"""Stage4 v2 — 회전·후진 명령 강제 배정 + 접지 요-슬립 벌점 (평지/험지).

설계 근거 (Stage4 안정화 성공 후 개선 1라운드):
- 명령 커버리지 결함 보정: Uniform 샘플에서 '제자리회전(vx=vy=0, wz만)'은
  측도 0이라 한 번도 안 나오고, 뚜렷한 후진(vx ≤ −0.15)도 저빈도.
  음성 명령 최종목표("돌아", "뒤로 가")의 학습 데이터가 구조적으로 부족.
  → UniformVelocityCommand 서브클래스(Stage4V2VelocityCommand)가 리샘플 후
    15% env를 제자리회전, 15% env를 뚜렷한 후진으로 강제 배정.
- foot_yaw_slip 벌점: 제자리회전을 '접지 발바닥 비틀기(트위스트 슬립)'로
  게이밍하지 않고 발을 들어 딛는 스텝 턴으로 돌게 유도 (mdp_stage4 참조).
- track_ang_vel_z_exp 1.0 → 1.5: 회전 명령 비중 증가에 맞춰 요 추종 강화.
- 험지판(Rough): stage4_rough_env_cfg 패턴을 이식하되 서브지형 비율을 v2로
  변경 — 경사 비중 확대(0.05→0.20, 최대 0.35rad)·계단 40%·flat 축소.

실행:
  isaaclab.sh -p rl_walking/scripts/train.py --task Biped12-Velocity-Stage4V2-v0 \
      --headless --num_envs 4096

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import copy
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import mdp_stage4
from .rough_env_cfg import BIPED12_TERRAINS_CFG, Biped12CurriculumCfg
from .stage4_env_cfg import _FEET_SENSOR_CFG, Biped12Stage4DREnvCfg

# 로봇 관절체(articulation)의 발 바디 — foot_yaw_slip의 body_ang_vel_w 인덱스용.
# 접촉센서(_FEET_SENSOR_CFG)와 반드시 같은 [왼발, 오른발] 순서 + preserve_order.
_FEET_BODY_CFG = SceneEntityCfg(
    "robot",
    body_names=["left_ankle_r_joint", "right_ankle_r_joint"],
    preserve_order=True,
)


##
# 커스텀 명령항 — 제자리회전·후진 강제 배정
##


class Stage4V2VelocityCommand(UniformVelocityCommand):
    """Uniform 리샘플 후 일부 env에 제자리회전/뚜렷한 후진을 강제 배정.

    r ~ U(0,1) 한 번으로 상호배타 분할 (겹침 없음 → 비율이 정확히 15%/15%):
      r < rel_spin_envs                          → 제자리회전 env
      rel_spin_envs ≤ r < spin+rel_backward_envs → 뚜렷한 후진 env
    - 제자리회전: vx=vy=0, wz는 ±spin_ang_vel_range 균등 재샘플.
      is_heading_env=False (heading 제어가 매 스텝 wz를 덮어쓰는 것 차단),
      is_standing_env=False (_update_command의 서기 zero화가 회전 명령을
      지우는 것 차단 — 이 둘을 안 끄면 강제 배정이 무효가 되는 함정).
    - 후진: vx만 backward_vel_x_range 균등 재샘플, vy·wz(heading 제어 포함)는
      부모 샘플 유지. is_standing_env=False만 해제 (후진이 0으로 지워지지 않게).
    - is_spin_env/is_backward_env 버퍼: 실측 검증·디버깅용 마스크.
    """

    cfg: "Stage4V2VelocityCommandCfg"

    def __init__(self, cfg: "Stage4V2VelocityCommandCfg", env):
        super().__init__(cfg, env)
        # 강제 배정 마스크 (검증 스크립트가 비중 실측에 사용)
        self.is_spin_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_backward_env = torch.zeros_like(self.is_spin_env)

    def _resample_command(self, env_ids: Sequence[int]):
        # 부모의 균등 샘플 (vx·vy·wz·heading·standing 결정) 먼저 수행
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # 단일 균등 난수로 상호배타 그룹 분할
        r = torch.rand(len(ids), device=self.device)
        spin_mask = r < self.cfg.rel_spin_envs
        back_mask = (~spin_mask) & (r < self.cfg.rel_spin_envs + self.cfg.rel_backward_envs)
        spin_ids = ids[spin_mask]
        back_ids = ids[back_mask]

        # ── 제자리회전: vx=vy=0, wz ±0.5~1.0 균등 재샘플 ──
        if spin_ids.numel() > 0:
            self.vel_command_b[spin_ids, 0] = 0.0
            self.vel_command_b[spin_ids, 1] = 0.0
            mag = torch.empty(spin_ids.numel(), device=self.device).uniform_(
                *self.cfg.spin_ang_vel_range
            )
            sign = torch.where(
                torch.rand(spin_ids.numel(), device=self.device) < 0.5,
                torch.tensor(-1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            self.vel_command_b[spin_ids, 2] = sign * mag
            self.is_heading_env[spin_ids] = False  # heading이 wz를 덮어쓰지 않게
            self.is_standing_env[spin_ids] = False  # 서기 zero화 제외

        # ── 뚜렷한 후진: vx만 재샘플, wz(heading 제어)·vy는 유지 ──
        if back_ids.numel() > 0:
            self.vel_command_b[back_ids, 0] = torch.empty(
                back_ids.numel(), device=self.device
            ).uniform_(*self.cfg.backward_vel_x_range)
            self.is_standing_env[back_ids] = False  # 서기 zero화 제외

        # 실측용 마스크 갱신 (리샘플된 env만 초기화 후 재기록)
        self.is_spin_env[ids] = False
        self.is_backward_env[ids] = False
        self.is_spin_env[spin_ids] = True
        self.is_backward_env[back_ids] = True


@configclass
class Stage4V2VelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Stage4V2VelocityCommand 페어 cfg — 강제 배정 비율·범위."""

    class_type: type = Stage4V2VelocityCommand

    rel_spin_envs: float = 0.15
    """제자리회전 강제 배정 비율 (리샘플 env 중)."""

    spin_ang_vel_range: tuple[float, float] = (0.5, 1.0)
    """제자리회전 wz 크기 범위 [rad/s] — 부호는 50:50 랜덤 (±0.5~1.0)."""

    rel_backward_envs: float = 0.15
    """뚜렷한 후진 강제 배정 비율 (제자리회전과 상호배타)."""

    backward_vel_x_range: tuple[float, float] = (-0.4, -0.15)
    """후진 vx 범위 [m/s] — 균등 재샘플."""


##
# 평지 v2
##


@configclass
class Biped12Stage4V2EnvCfg(Biped12Stage4DREnvCfg):
    """Stage4 v2 = Stage4-DR(클록·이력·미러·방폭·DR) + 회전/후진 명령 + 요-슬립 벌점."""

    def __post_init__(self):
        super().__post_init__()
        # ── 명령항 교체: Uniform → Stage4V2 (기존 클래스 불변 — 새 cfg로 대체) ──
        # DR에서 확정된 분포를 그대로 계승: vx(-0.4,0.8)·vy±0.25·wz±1.0·heading±π·서기10%
        old = self.commands.base_velocity
        self.commands.base_velocity = Stage4V2VelocityCommandCfg(
            asset_name=old.asset_name,
            resampling_time_range=old.resampling_time_range,
            rel_standing_envs=old.rel_standing_envs,
            rel_heading_envs=old.rel_heading_envs,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            debug_vis=old.debug_vis,
            ranges=old.ranges,
        )
        # ── 요 추종 강화 (회전 명령 비중 증가에 맞춤) 1.0 → 1.5 ──
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        # ── stand_still 게이트 교체 (적대 리뷰 CONFIRMED 수정): 원본은 선속도
        #    노름만 게이트해서 제자리회전 env(vx=vy=0)를 '서기'로 오분류 —
        #    스텝 턴의 관절 이탈을 벌점해 접지 비틀기 게이밍을 조장. SE(2)
        #    3성분 전부로 게이트하는 v2판으로 교체 (서기 env 동작은 동일).
        #    gait_phase_reward의 동일 결함은 mdp_stage4에서 직접 수정.
        self.rewards.stand_still.func = mdp_stage4.stand_still_full_command
        # ── 접지 요-슬립 벌점 (cmd 게이팅 없음 — mdp_stage4.foot_yaw_slip 참조) ──
        self.rewards.foot_yaw_slip = RewTerm(
            func=mdp_stage4.foot_yaw_slip,
            weight=-0.5,
            params={
                "sensor_cfg": _FEET_SENSOR_CFG,
                "asset_cfg": _FEET_BODY_CFG,
                "threshold": 5.0,
            },
        )


@configclass
class Biped12Stage4V2EnvCfg_PLAY(Biped12Stage4V2EnvCfg):
    """재생/평가용: 소수 env, 노이즈·외란 끔, 전진 명령 고정 (stage4 PLAY 패턴)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        # v2 강제 배정도 끔 — 고정 전진 재생이 목적
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class Biped12Stage4V2MarchEnvCfg(Biped12Stage4V2EnvCfg):
    """제자리 걷기(march)판 — 명령 0 = 정지 대신 클록 리듬 '들었다 놨다'.

    용도: 실물 접지 첫 동적 테스트 (이동 없이 걷기 역학만 검증 — 낙상 위험 최소).
    ⚠ 이 정책은 '조용히 서기'가 불가능 (명령 0 = 스텝). 배포 노드에서 정지는
    반드시 일반 체크포인트(R4 model_12994)로 전환할 것.

    R2~R4 확정 오버라이드를 cfg에 명문화 — 리줌 CLI 누락 실수 원천 차단.
    """

    def __post_init__(self):
        super().__post_init__()
        # ── R2~R4 확정값 명문화 (종전 CLI 오버라이드) ──
        self.rewards.track_ang_vel_z_exp.weight = 2.0       # R2
        self.rewards.flat_orientation_l2.weight = -3.0      # R3
        self.rewards.feet_gap.params["min_gap"] = 0.22      # R4
        # ── march 핵심: 명령 0 env 실효 40% + 위상 보상 상시 가동 ──
        # spin 15%/backward 15% 강제 배정이 독립 난수로 standing을 덮어쓰므로
        # (적대 리뷰 확인: 실효 = 설정 × 0.70) 0.571로 설정해 실효 40% 확보
        self.commands.base_velocity.rel_standing_envs = 0.571
        self.rewards.gait_phase.params["always_walk"] = True
        # 정지자세 고정 벌점 제거 — 제자리 스텝의 관절 이탈을 벌하지 않게
        self.rewards.stand_still = None
        # 위치 앵커 (march 1라운드 실측 드리프트 2.0m/30s 교정): 원점 0.3m
        # 데드존 밖 선형 벌점 — 명령 0 env만
        self.rewards.march_anchor = RewTerm(
            func=mdp_stage4.march_position_hold,
            weight=-1.0,
            params={"dead_zone": 0.3, "cap": 2.0},
        )


@configclass
class Biped12Stage4V2MarchEnvCfg_PLAY(Biped12Stage4V2MarchEnvCfg):
    """march 재생/평가 — 전 env 명령 0 (순수 제자리 걷기)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


##
# 험지 v2 — 서브지형 비율 변경판
##

# v2 서브지형 비율 (합 = 1.0): 경사 확대·계단 40%·flat 축소·boxes류(curbs) 최소화
_V2_TERRAIN_PROPORTIONS = {
    "slope": 0.20,
    "flat": 0.15,
    "random_rough": 0.20,
    "stairs_down": 0.20,
    "stairs_up": 0.20,
    "curbs": 0.05,  # boxes류 (MeshRandomGridTerrainCfg 불규칙 높이 격자)
}


def _make_v2_terrains_cfg():
    """BIPED12_TERRAINS_CFG의 v2 비율판 생성 — 원본은 불변(deepcopy).

    slope는 비중 확대(0.05→0.20)에 맞춰 최대 경사도 0.2→0.35rad로 확장.
    비율 합=1 검증 포함 (키 오타·서브지형 추가 누락이 여기서 잡히도록).
    """
    cfg = copy.deepcopy(BIPED12_TERRAINS_CFG)
    for name, proportion in _V2_TERRAIN_PROPORTIONS.items():
        cfg.sub_terrains[name].proportion = proportion  # 키 오타면 즉시 KeyError
    cfg.sub_terrains["slope"].slope_range = (0.0, 0.35)
    total = sum(t.proportion for t in cfg.sub_terrains.values())
    assert abs(total - 1.0) < 1e-6, f"v2 서브지형 비율 합 {total} != 1.0"
    return cfg


@configclass
class Biped12Stage4V2RoughEnvCfg(Biped12Stage4V2EnvCfg):
    """Stage4V2-Rough = V2(명령·요슬립) + 지형 커리큘럼 (stage4_rough 패턴 이식)."""

    def __post_init__(self):
        super().__post_init__()
        # ── 지형 교체: plane → generator(v2 비율) + curriculum ──
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=_make_v2_terrains_cfg(),
            max_init_terrain_level=3,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum = Biped12CurriculumCfg()
        # ── 험지 자세 페널티 완화 (rough_env_cfg 확정값) ──
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0.0  # 계단에선 절대높이 무의미
        # ── 험지 푸시 완화 ±0.3 (계단 모서리 접촉과 중첩 시 폭발 촉발원) ──
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.3, 0.3), "y": (-0.3, 0.3)
        }


# 커리큘럼 지형은 '컬럼 단위(num_cols=10 → 10% 단위)'로 배정되므로 비율을
# 10% 배수로 설계해 설정=실현을 보장한다 (적대 리뷰 발견: 15% 등 어중간한
# 값은 양자화로 왜곡 — v3 최초안에서 stairs_up이 리줌 원본의 절반인 10%로
# 떨어지는 문제 확인). cumsum 배정식은 terrain_generator._generate_curriculum_
# terrains 참조. 라벨 주의: 피라미드("slope")는 정상 스폰이라 '내리막 시작',
# 역피라미드("slope_inv")는 구덩이 스폰이라 '오르막 시작'이다 (리뷰 교정).
_V3_SLOPE_PROPORTIONS = {
    "flat": 0.10,
    "random_rough": 0.10,
    "stairs_down": 0.20,   # v2 실현 배치(2컬럼)와 동일 — 계단 퇴행 차단
    "stairs_up": 0.20,     # v2 실현 배치(2컬럼)와 동일
    "slope": 0.20,         # 내리막 시작 (피라미드 정상 스폰)
    "curbs": 0.00,         # v2에서도 실현 0컬럼(죽은 설정) — 명시적 0
    "slope_inv": 0.20,     # 오르막 시작 (역피라미드 구덩이 스폰) — 신설
}


def _make_v3_slope_terrains_cfg():
    """v2 지형에 역피라미드 경사(slope_inv)를 신설 — 경사 계열 실현 40%.

    목적: '다양한 각도의 경사 적응' — 각도는 난이도(0~1)×최대 0.35rad(20°),
    접근 방향은 리셋 요 ±180°(env_cfg reset_base)로 이미 전방위 커버.
    여기서는 내리막/오르막 '시작 상황'을 각 20%로 균형 노출한다.
    실현 컬럼(10열): flat 1 | random_rough 1 | stairs_down 2 | stairs_up 2 |
    slope 2 | slope_inv 2 — v2 대비 flat·rough가 1컬럼씩 줄고 오르막 경사 신설.
    """
    cfg = _make_v2_terrains_cfg()
    cfg.sub_terrains["slope_inv"] = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=0.20, slope_range=(0.0, 0.35), platform_width=2.0,
        border_width=0.25,
    )
    for name, proportion in _V3_SLOPE_PROPORTIONS.items():
        cfg.sub_terrains[name].proportion = proportion  # 키 오타면 즉시 KeyError
    total = sum(t.proportion for t in cfg.sub_terrains.values())
    assert abs(total - 1.0) < 1e-6, f"v3 서브지형 비율 합 {total} != 1.0"
    # 컬럼 양자화 검증 — 설정 비율이 10열 배정과 정확히 일치해야 한다
    props = [t.proportion for t in cfg.sub_terrains.values()]
    cum, cols = [], []
    acc = 0.0
    for p in props:
        acc += p
        cum.append(acc)
    for c in range(10):
        v = c / 10 + 0.001
        cols.append(next(i for i, s in enumerate(cum) if v < s))
    realized = [cols.count(i) / 10 for i in range(len(props))]
    assert all(abs(r - p) < 1e-9 for r, p in zip(realized, props)), (
        f"컬럼 양자화 괴리: 설정 {props} vs 실현 {realized}")
    return cfg


@configclass
class Biped12Stage4V2SlopeEnvCfg(Biped12Stage4V2RoughEnvCfg):
    """경사 보강 학습판 — 험지 v2에서 지형 비율만 교체 (관측·보상 동일 → 리줌 호환)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_v3_slope_terrains_cfg()
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class Biped12Stage4V2SlopeEnvCfg_PLAY(Biped12Stage4V2SlopeEnvCfg):
    """경사 보강판 재생/평가 — rough PLAY와 동일 패턴."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class Biped12Stage4V2RoughEnvCfg_PLAY(Biped12Stage4V2RoughEnvCfg):
    """재생/평가용: 소수 env, 노이즈·외란 끔, 저속 전진 고정 (rough PLAY 패턴)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # 전 난이도 열에 골고루 스폰 + 승급 커리큘럼 끔
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        # v2 강제 배정도 끔 — 고정 저속 전진 재생이 목적
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class Biped12Stage4V2MarchDRv2EnvCfg(Biped12Stage4V2MarchEnvCfg):
    """march + 0802 컴플라이언스 정합 (walk DRv2와 동일 오버라이드).

    근거: 구 march(14993)도 구식 DR(게인 0.8~1.2) 출신 — 실물(직렬 컴플라이언스
    실증)에 얹기 전 컴플라이언스 조건 검증·재학습용. walk 판정 전례: 구정책
    컴플라이언스 낙상 85.9% vs DRv2 1.6%.
    """

    def __post_init__(self):
        super().__post_init__()
        from rl_walking.stage4_env_cfg import apply_compliance_drv2
        apply_compliance_drv2(self)


@configclass
class Biped12Stage4V2MarchDRv3EnvCfg(Biped12Stage4V2MarchDRv2EnvCfg):
    """march-DRv3 — march-DRv2 + 액션 채터 제거 (stage4 DRv3와 동일 처방)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.12


@configclass
class Biped12Stage4V2MarchDRv4EnvCfg(Biped12Stage4V2MarchDRv3EnvCfg):
    """march-DRv4 — 0802 march_v3 실물 데뷔 1차 실측 반영.

    실측 실패 모드 (영상 프레임 + 블랙박스 + 발압력 교차 검증): 첫 스텝에서
    몸통이 요(yaw)로 비틀리며 앞-왼쪽으로 낙하 (줄은 느슨했음 — 자체 실패).
    발하중 108N→30N 붕괴는 결과. 학습 공백 3종을 정조준:
    ① 발-바닥 마찰 랜덤화 — march 가족은 마찰 0.8/0.6 '고정' 학습이라 실물
       실험실 바닥(미지 마찰)의 요 슬립을 한 번도 경험 못 함. 요 비틀림 직격.
    ② 푸시 외란에 요 성분 추가 + 간격 단축 — 착지 '쿵' 임팩트(시뮬보다 단단한
       실바닥)가 만드는 요·병진 교란의 회복 훈련.
    ③ 초기 자세 기울기 — 실물은 기울기 ~2.8°에서 정책이 켜짐 (시뮬은 항상
       수직 시작). 첫 틱 분포 밖 문제 제거.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.25, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.6, 0.6),
        }
        self.events.reset_base.params["pose_range"]["roll"] = (-0.07, 0.07)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.07, 0.07)


@configclass
class Biped12Stage4V2MarchDRv5EnvCfg(Biped12Stage4V2MarchDRv4EnvCfg):
    """march-DRv5 — 0803 '측정 기반' 재정합 (무한 튜닝 탈피 선언판).

    0802 실물 첫 march 블랙박스(61틱)의 sysid + 리플레이 대조 실측을 DR 중심에
    반영. 추측으로 범위를 '넓힌' v4와 달리 v5는 측정으로 '조준'한다:
    ① 루프 지연: 실측 명령→관측 왕복 80~160ms(4~8틱). 종전 DR 0~2틱은 실물의
       절반도 못 커버 — max_delay_steps 6으로 확대 (관측 지연 근사 포함).
    ② 유효 강성: 실측 추종비 무릎 0.27·발목 0.17~0.60 — 종전 하한 0.4는 낙관.
       하한 0.2로 확대 (상한 유지 — 파이프라인 개선 시에도 유효하도록).
    ③ 케이던스: 클록 1.4→1.1Hz (2.2걸음/s) — "천천히 안정적으로" (0803 지시).
       배포 러너의 클록 주파수도 체크포인트별로 일치시켜야 함 (계약 변경!).
    ④ 스탠스 무릎 굴곡 밴드 8~20° 보상 — "약간 굽혀 눌러주는" 스탠스 (0803
       지시), 니락(과신전 스톱 기대기) 전략 구조적 대체.
    ⑤ 착지 충격 벌점 — "쿵" 억제 (초과 수직접촉력 180N+ 선형).
    """

    def __post_init__(self):
        super().__post_init__()
        # ① 루프 지연 (실측 4~8틱): v5b에서 2~6틱 중심화 — 0~1틱 학습은 실물에
        #    없는 조건이라 낭비 + 지연 없는 env가 지연 env의 학습 신호를 희석.
        self.actions.joint_pos.max_delay_steps = 6
        self.actions.joint_pos.min_delay_steps = 2
        # ② 유효 강성 하한 확대 (실측 무릎 0.27·발목 0.17)
        self.events.actuator_gains.params[
            "stiffness_distribution_params"] = (0.2, 1.1)
        # ③ 클록 1.1Hz — 관측·보상 동일 갱신 (위상 정의 공유 필수)
        self.observations.policy.gait_clock.params["freq"] = 1.1
        self.rewards.gait_phase.params["freq"] = 1.1
        # ④ 스탠스 무릎 굴곡 밴드 (v5b: 1.0→2.5 — v5 평가에서 중앙 −2°로
        #    미달성, 직립·앵커 항에 밀렸음)
        self.rewards.knee_stance_flexion = RewTerm(
            func=mdp_stage4.knee_stance_flexion,
            weight=2.5,
            params={
                "sensor_cfg": _FEET_SENSOR_CFG,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["left_knee_joint", "right_knee_joint"],
                    preserve_order=True,
                ),
            },
        )
        # ⑥ v5b: 지연 하 채터 재발(반전 0.41) 대응 — action_rate 추가 증강.
        #    지연 환경에선 고주파 액션의 비용이 낮아 보여 채터가 되살아난다.
        self.rewards.action_rate_l2.weight = -0.20
        # ⑤ 착지 충격 벌점 (300N 착지 = -1.2/틱 대역)
        self.rewards.feet_impact = RewTerm(
            func=mdp_stage4.feet_impact_force,
            weight=-0.01,
            params={"sensor_cfg": _FEET_SENSOR_CFG},
        )


@configclass
class Biped12Stage4V2MarchDRv5EnvCfg_PLAY(Biped12Stage4V2MarchDRv5EnvCfg):
    """v5 재생/평가 — 표준 조건(DR 고정·외란 없음·전 env 명령 0), 클록 1.1Hz 유지."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # DR·지연·마찰을 표준 공칭으로 고정 (v5의 DR 오버라이드 되돌림)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        self.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.actions.joint_pos.max_delay_steps = 0
        self.events.reset_base.params["pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 0.0)


@configclass
class Biped12Stage4V2StandEnvCfg(Biped12Stage4V2MarchDRv5EnvCfg):
    """스탠드 전용 정책 — '조용히 서서 능동 균형' (0803 실물 관찰 반영 신규).

    실물 관찰: 정지 = stage8 동결 홀드(kp150 조각상) → "아슬아슬, 안정적으로
    서있는 느낌이 아님". 균형은 강성이 아니라 제어에서 나온다 — 명령 0에서
    발을 떼지 않고 스탠스 무릎 굴곡(굽혀 눌러주기)로 서서 푸시를 흡수하는
    전용 정책을 학습한다. 관측 245 계약 동일(클록은 v5와 같은 1.1Hz로 흐르지만
    정책이 무시하도록 학습) — 배포 스택 무변경 투입 가능.

    v5 대비 오버라이드: 전 env 명령 0, gait_phase 클록 보상 제거 → 양발 접촉
    보상으로 대체(스텝 금지 방향), march_anchor 강화, 푸시는 v5 그대로(회복
    훈련이 핵심 목적).
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_spin_envs = 0.0
        self.commands.base_velocity.rel_backward_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # 클록 리듬 스텝 대신 양발 접촉 유지 보상 (always_walk 해제 = 명령 0
        # → stand_reward 경로)
        self.rewards.gait_phase.params["always_walk"] = False
        # 제자리 앵커 강화 (스탠드는 드리프트 자체가 실패)
        self.rewards.march_anchor.weight = -2.0
        self.rewards.march_anchor.params["dead_zone"] = 0.15
        # v3 (0803): stand_v2 평가 기울기 18.7° 불합격 — '기울어 서기' 평형 차단.
        # ① 직립 보상 강화 (march -3 → -8: 스탠드는 직립이 제1 목표)
        self.rewards.flat_orientation_l2.weight = -8.0
        # ② CoM 랜덤화를 실측 대역으로 축소 — DRv5의 x −0.20~0.05는 보행
        #    강건성용 극단 포함이라 스탠드에선 '기울어 서기'를 정답으로 만든다.
        #    payload 실측(com_report: 정합 CoM 발목 뒤 1~4cm) 중심.
        self.events.base_com.params["com_range"]["x"] = (-0.08, 0.03)


@configclass
class Biped12Stage4V2StandDRv6EnvCfg(Biped12Stage4V2StandEnvCfg):
    """stand DRv6 — 0803 실물 발진 2건의 실측 갭 반영 재학습판 (stand_v4 후보).

    실기 근거 (stand_v3 제로샷 2회 블랙박스·candump 전수 분석):
    ① 총 루프지연 실측 4~5틱 + 기계 위상 — 학습 DR(2~6틱) 상단 미달 대역에서
       5~9Hz 정책 주도 리밋사이클 (타깃이 실측을 80~100ms 선행, 진폭비 1.3~1.5)
    ② 관측 계단: freeze-then-jump (정지 홀드율 48~71%, 최장 200ms, 점프 0.8~2°)
    ③ qvel 관측 절단: 정지 시 48~71%가 정확히 0 — 감쇠 신호 무력
    ④ 발목F ±1° 백래시 디더 (힙의 10배)
    ⑤ 인수(takeover) 자세가 리셋 분포(±4.6°) 밖 — 실측 최대 24.7°
    미모델 잔여(문서화): 배포 타깃 LPF 2.5Hz·러너 슬루 4°/틱은 학습 액션 경로에
    없음(둘 다 진폭 축소 = 안정 방향 편차), 홀드 런의 시간 상관(iid 근사).
    """

    def __post_init__(self):
        super().__post_init__()
        # ① 지연 DR을 실측 대역으로 상향 (2~6 → 3~7틱).
        #    1차 베이크(0803 16:50 런) 실측: 상한 8틱은 kp150 저댐핑에서 시뮬
        #    자체가 발진→물리폭발 (보상 -30만 스파이크, 종반 낙상 97%) — 실물
        #    불안정의 시뮬 재현. 베이크2에서 7틱도 폭발 재발 → 상한 6틱(DRv5 검증 대역)+하한 3틱 상향만 유지.
        self.actions.joint_pos.min_delay_steps = 3
        self.actions.joint_pos.max_delay_steps = 6
        # ②④ 관절각 관측: 양자화 계단 + 발목F 디더
        self.observations.policy.joint_pos.func = \
            mdp_stage4.joint_pos_rel_staircase
        self.observations.policy.joint_pos.params = {
            "quant_step_rad": 0.007, "ankle_dither_rad": 0.0175}
        # ③ 관절속도 관측: 드롭아웃 (감쇠를 관측이 아닌 액션 형성으로)
        self.observations.policy.joint_vel.func = \
            mdp_stage4.joint_vel_rel_dropout
        self.observations.policy.joint_vel.params = {"zero_prob": 0.5}
        # ⑤ 리셋 자세 분포 확대: 인수 불일치 커버 (±0.08 → ±0.2 rad ≈ ±11.5°)
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)


@configclass
class Biped12Stage4V2StandV5EnvCfg(Biped12Stage4V2StandEnvCfg):
    """stand_v5 — 0803 3차 실기(5초 자립 후 뒤 넘어짐→창발 보호스텝 104초) 반영.

    실기 판정: 발진(5~9Hz)은 브리지 완화 5종으로 정복 — 남은 병목은 질량 기하
    (실측 CoM 발목 뒤 1~4cm, 미모델 전장 1.51kg 뒤·위)와 스텝 금지 보상.
    설계 5지렛대:
    ① 몸의 진실: 전장 payload를 실측 대역으로 명시 (mass +1.0~2.0kg,
       base CoM x −0.10~−0.02(뒤 확정)·z +0.05~0.15(위)) — "혹시"가 아니라
       "확실히 뒤로 쏠린 몸"을 전제로 발목 전략·전경사 자세를 학습.
    ② 스텝 회복 합법화: 3차 실기의 창발 보호스텝이 생존 전략이었음 —
       양발 접촉 보상 절반, 앵커 완화(위기 스텝 허용, 복귀는 유지).
    ③ 관측 현실화 저용량: DRv6 베이크 1~3 폭발 부검 — 고용량 관측 부식이
       난동→물리폭발 유발. 계단 0.17°·발목 디더 0.46°·qvel 드롭 25%로 감량
       (실물 분포 방향은 유지, 워밍스타트가 적응 가능한 선).
    ④ 지연 하한 실측 정합: 3~6틱 (상단은 DRv5 검증 대역 유지).
    ⑤ 발진 벌점 강화: action_rate −0.12 → −0.20 (지연 루프 감쇠를 액션
       형성으로 확보 — qvel 관측은 실물에서 못 믿음이 실측됨).
    """

    def __post_init__(self):
        super().__post_init__()
        # ① 몸의 진실 — 전장 payload 실측 대역
        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 2.0)
        self.events.base_com.params["com_range"] = {
            "x": (-0.10, -0.02),
            "y": (-0.02, 0.02),
            "z": (0.05, 0.15),
        }
        # ② 스텝 회복 합법화
        self.rewards.gait_phase.weight *= 0.5
        self.rewards.march_anchor.weight = -1.0
        self.rewards.march_anchor.params["dead_zone"] = 0.3
        # ③ 관측 현실화 (저용량)
        self.observations.policy.joint_pos.func = \
            mdp_stage4.joint_pos_rel_staircase
        self.observations.policy.joint_pos.params = {
            "quant_step_rad": 0.003, "ankle_dither_rad": 0.008}
        self.observations.policy.joint_vel.func = \
            mdp_stage4.joint_vel_rel_dropout
        self.observations.policy.joint_vel.params = {"zero_prob": 0.25}
        # ④ 지연 실측 정합 (하한만 상향 — 상단은 검증 대역)
        self.actions.joint_pos.min_delay_steps = 3
        self.actions.joint_pos.max_delay_steps = 6
        # ⑤ 발진 벌점 + 인수 자세 커버
        self.rewards.action_rate_l2.weight = -0.20
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)


@configclass
class Biped12Stage4V2StandV6EnvCfg(Biped12Stage4V2StandV5EnvCfg):
    """stand_v6 — 4차 실기(0803 밤) 판정 반영: 전방 과보상 편향 교정.

    실기 실측: stand_v5는 하중 자립 90%(발력 106N/117.7N) 달성했으나 상시
    '앞으로 밀기' 편향 — v3의 후방 밀림이 전방으로 반전 = payload 학습 대역
    (+1.0~2.0kg, CoM x −0.10~−0.02)이 실효 뒤쏠림보다 과대해 과보상을 학습.
    이진 탐색 조임: payload 질량·CoM 대역을 실측 중심으로 절반 축소.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass.params["mass_distribution_params"] = (1.2, 1.6)
        self.events.base_com.params["com_range"] = {
            "x": (-0.08, -0.03),
            "y": (-0.02, 0.02),
            "z": (0.05, 0.12),
        }


@configclass
class Biped12Stage4V2StandV7EnvCfg(Biped12Stage4V2StandV6EnvCfg):
    """stand_v7 — 무개입 낙하 실측(0804) 반영: payload '위치' 가정 제거.

    무오염 파형 확정: v6b는 균형 잡힌 실물(트림0 자립 3회 실증) 위에서 1초부터
    전방 기울임을 단조 누적 (+1.6→+19°, 발진·외란 없음) = '뒤로 쏠린 몸' 가정
    (payload 뒤 배치 학습)이 낳은 순수 편향. 질량 결손 1.5kg은 실재하므로
    유지하되, 위치를 중앙 대역으로 — 몸이 균형임을 전제로 학습한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.05, 0.12),
        }


def _make_flat_only_terrains_cfg():
    """전부 평지 — 서기 학습에 경사·계단은 '기울어 서기'를 정답으로 만든다.

    v2 지형은 최대 19° 경사(slope_range 0.35rad)와 계단을 20%씩 섞는다. 보행
    강건성에는 옳지만, 서기 정책에게는 '지면이 기울어 있을 수 있다'가 전제로
    깔려 평지 실물에서 불필요한 기울임을 낳는다. 실물 시험장은 평평한 바닥이다.
    """
    cfg = copy.deepcopy(BIPED12_TERRAINS_CFG)
    for name in cfg.sub_terrains:
        cfg.sub_terrains[name].proportion = 0.0
    cfg.sub_terrains["flat"].proportion = 1.0
    return cfg


@configclass
class Biped12Stage4V2StandV8EnvCfg(Biped12Stage4V2StandV7EnvCfg):
    """stand_v8 — '가만히 서 있기'가 성립하는 환경으로 교정 (2026-08-08).

    **왜 v5~v7이 전부 같은 방식으로 실패했는가.**
    실물은 전 관절 0°에서 stage8 고정 홀드만으로 발력 96%·몇 분씩 선다.
    능동 균형이 필요 없는 몸이다. 그런데 학습 환경은 정반대로,
    **'서 있기만 하기'가 성립할 수 없도록 설계돼 있었다** — env_cfg.py 의
    push_robot 주석이 그렇게 명시한다. 8~12초마다 ±0.3m/s 푸시, 매 리셋마다
    전 관절 ±4.6°+관절속도 ±0.3rad/s, 최대 19° 경사 타일.
    그 환경에서 최적 전략은 '끊임없이 능동 보정하기'이고, 그 전략을 정적으로
    안정한 실물에 넣으면 **밀어내는 것이 당연한 귀결**이다.
    (5차 v6b 실측: 발력 96%로 혼자 서 있다가 정책 권한 alpha 0.84에서 하중
    이탈 시작, 1.4초 만에 −1.8→+19.2° 단조 전방. 발진도 스텝도 없었다.)

    v7까지 다섯 세대가 바꾼 축은 payload/CoM 랜덤화뿐이었고, DR 스윕에서
    실물 거동(+4.06°/s)을 재현한 env 는 128개 중 0개, 역산 CoM 은 +417mm 로
    물리적으로 불가능 — **CoM 가설은 기각됐다.** 바꿀 축은 외란 설계다.

    v7 대비 변경 (측정 기반 DR — 지연·강성·컴플라이언스·payload 대역은 유지):
      ① push_robot 제거 — 실물 판정은 무개입 낙하다. 외란 복원력은 서기가
         성립한 다음 세대에서 되돌린다.
      ② 초기 관절 교란 ±4.6°→±1.1°, 관절속도 ±0.3→±0.05 rad/s.
         실물 핸드오프는 잘 정의된 자세에서 시작한다.
      ③ 리셋 몸통 기울임 ±4°→±1.1° (착지 재현성 실측 ~2°).
      ④ 지형 전부 평지 — 실물 시험장은 평평한 바닥이다.

    판정: eval_drift_dr.py 로 **DR 켠 채** 전방 추세를 본다(Play 성적으로
    승격하지 않는다 — 그 잣대가 다섯 세대를 잘못 고른 원인이었다).
    합격 기준은 '낙상률'이 아니라 **실물 거동을 재현하지 않는가**이다.
    """

    def __post_init__(self):
        super().__post_init__()
        # ① 푸시 제거 — '서 있기'가 유효 전략이 되게 한다
        self.events.push_robot = None
        # ② 초기 관절 교란을 실물 수준으로
        self.events.reset_robot_joints.params["position_range"] = (-0.02, 0.02)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.05, 0.05)
        # ③ 리셋 몸통 기울임 축소
        for _k in ("roll", "pitch"):
            if _k in self.events.reset_base.params["pose_range"]:
                self.events.reset_base.params["pose_range"][_k] = (-0.02, 0.02)
        # ④ 평지 고정 (커리큘럼은 올릴 난이도가 없으므로 끈다)
        self.scene.terrain.terrain_generator = _make_flat_only_terrains_cfg()
        self.scene.terrain.terrain_generator.curriculum = False
        # ⑤ 무릎 굽힌 '스탠바이' 스탠스를 실제로 달성시킨다 (승윤님 지시).
        #    사람도 무릎을 편 차렷보다 살짝 굽힌 스탠바이가 안정적이다 —
        #    편 무릎은 하드스톱에 기대는 자세라 양방향 제어 권한이 없다
        #    (march_v3에서 '니락' 전략을 실제로 학습한 전례가 있다).
        #    밴드 보상은 v5b부터 있었으나 직립·앵커 항에 밀려 미달했으므로
        #    가중을 올리고 밴드를 스탠바이 대역(8~16°)으로 좁힌다.
        self.rewards.knee_stance_flexion.weight = 5.0
        self.rewards.knee_stance_flexion.params["band_rad"] = (0.14, 0.28)


@configclass
class Biped12Stage4V2MarchV6EnvCfg(Biped12Stage4V2MarchDRv5EnvCfg):
    """march_v6 — '바닥을 긁는다'(0808 실물 관찰) 교정: 발 들기 + 느린 걸음.

    **원인은 보상 설계의 공백이었다.** 보행은 feet_air_time(공중 시간, 문턱
    0.4s)과 게이트 클록으로만 만들어지고, **발 높이를 보상하는 항이 어디에도
    없었다.** 정책에게는 '떠 있기'만 충족하고 최소로 들어 스치듯 지나가는 것이
    보상상 최적이다.

    실측(march_v5b, 32env×15s, cmd 0.3, 스윙 1221회): 스윙 최고점 중앙 23.3mm,
    **p10 7.5mm · 최소 1.1mm · 20mm 미만이 41.6%**. 실물 바닥의 미세 요철이나
    발이 완전 수평이 아닌 순간에 닿는다 — 긁히는 것이 당연하다.

    처방 두 가지:
      ① feet_swing_clearance 보상 신설 — 스윙 발바닥을 45mm까지 선형 보상 후
         포화(과도한 들기는 에너지·안정성 손해라 보상하지 않는다).
      ② 게이트 클록 1.1 → 0.85Hz — 승윤님 관찰 "좀 천천히 걸어야겠다".
         한 스윙에 쓸 시간이 29% 늘어 같은 높이를 더 여유롭게 만든다.
         **배포 시 CKPT_GAIT_FREQ["march_v6"]=0.85 등록 필수** — 클록 불일치는
         위상 어긋난 관측이 되어 보행을 파괴한다.
    """

    def __post_init__(self):
        super().__post_init__()
        # ① 발 들기 보상 (스윙 중에만, 45mm 포화)
        self.rewards.feet_clearance = RewTerm(
            func=mdp_stage4.feet_swing_clearance,
            weight=2.0,
            params={
                "sensor_cfg": _FEET_SENSOR_CFG,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_r_joint", "right_ankle_r_joint"],
                    preserve_order=True,
                ),
                "target_m": 0.045,
            },
        )
        # ② 느린 걸음 — 관측·보상 클록을 반드시 함께 바꾼다(위상 정의 공유)
        self.observations.policy.gait_clock.params["freq"] = 0.85
        self.rewards.gait_phase.params["freq"] = 0.85


@configclass
class Biped12Stage4V2MarchV6bEnvCfg(Biped12Stage4V2MarchV6EnvCfg):
    """march_v6b — v6의 과용량 교정. 진단은 맞았고 처방 용량이 틀렸다.

    v6 실측(64env×30s, cmd 0.3): 발 들기는 완벽히 해결됐다(스윙 최고점 중앙
    23.3→76.9mm, 20mm 미만 41.6→2.8%). 그러나 **걷지를 못한다** —
    전진 0.25m(기대 9.0m)·속도추종오차 0.234·낙상 23.4%·직립 9.13°·
    발목롤 5.26rad/s(AK45-36 한계 5.0 **위반**)·액션레이트 0.128.
    발을 높이 드는 데 보상이 쏠려 전진이 밀렸고, 클록 0.85Hz는 같은 명령
    속도에 더 긴 보폭을 요구해 부담을 키웠다.

    v6 대비:
      ① clearance 가중 2.0 → 0.8 (전진·추종 항과 균형)
      ② 목표 높이 45 → 35mm (v5b 실측 23mm 대비 충분히 높고, 사람 보행
         clearance 15~25mm의 상한 부근. 과도한 들기는 에너지·발목속도 부담)
      ③ 클록 0.85 → 1.0Hz (v5b 1.1에서 완만히만 늦춘다)
      **배포 시 CKPT_GAIT_FREQ["march_v6b"]=1.0 등록 필수.**
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_clearance.weight = 0.8
        self.rewards.feet_clearance.params["target_m"] = 0.035
        self.observations.policy.gait_clock.params["freq"] = 1.0
        self.rewards.gait_phase.params["freq"] = 1.0


@configclass
class Biped12Stage4V2MarchV6cEnvCfg(Biped12Stage4V2MarchV6bEnvCfg):
    """march_v6c — v6b의 유일한 배포 차단 사유(발목롤 속도 위반) 제거.

    v6b 실측: 낙상 25.0→15.6%·20mm미만 스윙 41.6→17.2%·직립 8.03→7.72°로
    전부 개선했으나 **발목롤 5.68rad/s (AK45-36 한계 5.0 위반)**. 발을 더 들면
    발목롤이 더 빨라지는 구조적 부작용이다. joint_vel_l2(전 관절 균등 −5e-4)는
    특정 관절의 한계 초과를 잡지 못하므로 전용 벌점을 건다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_r_speed = RewTerm(
            func=mdp_stage4.joint_vel_over_limit,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["left_ankle_r_joint", "right_ankle_r_joint"],
                    preserve_order=True,
                ),
                "limit_rad_s": 4.5,   # 한계 5.0에 10% 마진
            },
        )


@configclass
class Biped12Stage4V2StandV9EnvCfg(Biped12Stage4V2StandV8EnvCfg):
    """stand_v9 — 캘리브레이션 상수 편향에 적응하는 서기 (실기 보류 해소).

    v8 실측(64env×8s): 영점 오차 2°에서 낙상 4/64, **3°에서 28/64(43.8%)·
    최대 기울임 32.1°.** IMU 오차는 3°에서 4/64로 훨씬 관대하다.
    원인은 관측 노이즈가 전부 Unoise(매 스텝 제로평균)뿐이고 **에피소드 내내
    유지되는 상수 편향을 학습한 적이 없기 때문**이다.

    더 위험한 것: boot_pose_check 자세 허용오차가 AK70 관절에서 8°(TOL)라
    **44% 낙상을 일으키는 3° 영점 오차가 게이트를 '일치'로 통과한다.**
    실물 영점은 매달림 육안 정렬로 잡았고 기구학 검증 이력이 없다.

    처방: 관절각 관측에 에피소드 상수 편향 ±3°(관절별 독립), 중력벡터에
    상수 피치 회전 ±3°(몸통 하나). 정책이 '내 기준이 틀어져 있을 수 있다'를
    전제로 중력·접촉·이력에서 실제 자세를 추론하도록 강제한다.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        # 에피소드마다 편향 재추첨 (리셋 이벤트)
        self.events.calib_bias = EventTerm(
            func=mdp_stage4.resample_calib_bias,
            mode="reset",
            params={"joint_bias_rad": (-0.052, 0.052),   # ±3°
                    "imu_tilt_rad": (-0.052, 0.052)},    # ±3°
        )
        # 관측 함수 교체 — 노이즈·클리핑·이력 설정은 기존 항의 것을 그대로 유지
        self.observations.policy.joint_pos.func = mdp_stage4.joint_pos_rel_biased
        self.observations.policy.projected_gravity.func = \
            mdp_stage4.projected_gravity_biased


@configclass
class Biped12Stage4V2StandV10EnvCfg(Biped12Stage4V2StandV9EnvCfg):
    """stand_v10 — **실물 질량 분포 정합** (0810). 두 달치 실패의 구조적 원인 교정.

    **왜 이게 근본 교정인가.**
    0809 실기에서 정책 없이 관절을 kp150 으로 굳히고 손을 떼자 로봇이
    **−19°/s 로 뒤로** 무너졌다(−0.9° → −20.0° 를 1.0초). 정책도 제어도 없는
    순수 역학이라 질량 분포 식별에 이상적인 시험이다. 같은 지표로 시뮬을
    역산하니(eval_com_fit.py, 질량 12.001kg 고정·CoM x 64 env 스윕):

        각속도 = +0.938 × CoM_x[mm] + 10.62   (r = +0.846)
        → 실물 −19°/s 를 내는 **전신 CoM x = 발목축 뒤 31.6mm**

    대조:
        biped12_payload.urdf  12.001kg · CoM 뒤 **30.7mm**  ← 1mm 차이로 일치
        biped12_base.urdf     10.491kg · CoM 뒤 13.5mm     ← **여태 학습에 사용**

    즉 결손 1.51kg 을 뒤 150·위 100mm 에 배치한 payload 자산이 처음부터
    옳았고, 학습만 그것을 안 쓰고 있었다. v3~v9 정책들은 **CoM 이 실물의
    절반만큼만 뒤에 있는 몸**을 균형잡는 법을 배웠다 — 실물에서 후방 보정이
    구조적으로 부족했던 이유다. 정책을 아무리 갈아도 없어지지 않는 갭이었다.

    v9 대비 변경 (v9 의 캘리브레이션 편향 랜덤화는 유지):
      질량   add_base_mass 1.2~1.6 → **1.35~1.67** (실물 12.0kg 정합, 중심 1.51)
      CoM x  base_com −0.03~+0.02 → **−0.043~−0.021**
             (전신 CoM −36.8 ~ −25.3mm, 중심 −30.7mm = 실측 대역)
    y·z 는 v7 대역 유지. 랜덤화 **폭**은 좁히지 않는다 — 중심만 옮긴다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass.params["mass_distribution_params"] = (1.35, 1.67)
        self.events.base_com.params["com_range"] = {
            "x": (-0.043, -0.021),
            "y": (-0.02, 0.02),
            "z": (0.05, 0.12),
        }


@configclass
class Biped12Stage4V2StandV12EnvCfg(Biped12Stage4V2StandV10EnvCfg):
    """stand_v12 — **직렬 컴플라이언스(SEA) 정합** (0810). 관측측 결손 교정.

    **왜 이게 v3~v11 전부를 무효화하는 교정인가.**
    엔코더는 로터(감속기 입력축)에 붙어 있고, 그 아래 감속기 백래시·구조 휨이
    직렬 스프링이다. 0802 첫 자립 기록이 이것을 이미 말하고 있었다 —
    **몸통이 5.7° 기울었는데 12관절 로터 오차는 전부 ≤0.7°**. 링크는 5.7°
    움직였는데 정책은 0.7° 밖에 못 봤다는 뜻이다.

    로터 관성이 작아 준정적으로 풀면 (kp: 서보, k: 구조):

        링크가 받는 실효 강성   k_eff  = kp·k/(kp+k)
        엔코더가 읽는 값        θ_r    = θ_l + τ_link/k
                                       = θ_l + kp/(kp+k)·(θ_target − θ_l)

    kp150·k43.5 이면 **관측의 77.5% 가 자기 명령의 메아리**이고 실제 상태는
    22.5% 뿐이다. 여태 이 효과를 '게인 스케일 하향'으로만 근사해 왔는데
    그것은 위 두 줄 중 **첫 줄만** 흉내낸 것이다. 정책이 변형에 눈이 먼다는
    둘째 줄이 통째로 빠져 있었고, v3~v11 은 전부 없는 감각을 가정하고 배웠다.

    **시뮬 확증 (v6b, 독립적인 두 스윕이 루프이득 축 하나로 합쳐짐).**
    이득 kp/k 가 1.4→1.9 를 넘는 순간 E_smth 가 3.95 → 9.6e6 으로 점프한다.
    실물 v6b 측정 E_smth = 4.40 → **실물은 발산 문턱 바로 아래**에서 돌고
    있었다. "부들부들 떨면서 서려다 균형을 잃고 쓰러진" 거동과 일치한다.
    그리고 **SEA 를 켜야만** 앞뒤 추세 부호가 뒤 → 앞으로 뒤집힌다 —
    CoM 보정으로도 지연 스윕으로도 재현 못 하던 실물 전방낙상(+4.06°/s)을
    재현한 첫 기전이다.

    v10 대비 변경:
      관측  joint_pos → joint_pos_rel_sea (θ_link + τ/k). 245 관측 계약 유지.
      구동  reset 이벤트로 env별 k_eff 기록 (actuator_gains **뒤**에 배치)
      DR    k_struct 60~400 Nm/rad — 실물 추정 43.5~107 을 넉넉히 감싼다.
            하나로 못 박지 않는 이유: 그 값이 틀리면 v3~v9 실패의 반복이다.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        self.observations.policy.joint_pos.func = mdp_stage4.joint_pos_rel_sea
        # actuator_gains 뒤에 와야 게인 DR 결과를 kp 로 읽는다 (직렬 합성)
        self.events.sea_stiffness = EventTerm(
            func=mdp_stage4.resample_sea_stiffness,
            mode="reset",
            params={"k_range": (60.0, 400.0)},
        )


@configclass
class Biped12Stage4V2StandV13EnvCfg(Biped12Stage4V2StandV6EnvCfg):
    """stand_v13 — **불확실성 전체를 담는 환경** (0810). 세대 경쟁의 종료.

    **왜 v3~v11 의 승격 판정이 전부 무의미했는가 (0810 대조 실측).**
    같은 정책·같은 배포 래퍼로 평가 환경만 바꿔 재니 순위가 뒤집힌다:

        정책     v6환경(푸시·거친지형)   v8환경(평지·무푸시)
        v6b            3/64                 20/64
        v7            16/64                  1/64
        v9            36/64                  2/64
        v11             —                   36/64

    **각 정책은 자기 학습 환경에서만 잘한다.** 즉 다섯 세대의 승격은
    "방금 바꾼 환경에 가장 잘 맞는 정책"을 고른 것이고, 실물 성능과 무관하다.
    환경을 바꿔 가며 세대를 뽑는 한 이 자기추적은 끝나지 않는다.

    **해법은 환경을 다시 고르는 것이 아니라 불확실성을 통째로 담는 것.**
    승윤님 설계 철학과 정확히 같다 — "기구에 차이가 있어도 자연스럽게 기능".
    어느 정책도 '자기 집'이 없는 분포에서 학습시키면, 이긴 정책은 환경에
    맞춰진 것이 아니라 **불확실성에 강한** 것이다.

    **담는 불확실성 (전부 실측·역산 근거):**
      CoM x  −0.10 ~ +0.02 m
             역산 구간이 초기 기울기 가정에 지배된다(0810 B): 물성 가정
             (마찰·반사관성)은 −31.0~−32.7mm 로 1.7mm 밖에 안 흔드는데,
             초기 피치 ±2° 가 −17.8~−42.9mm 로 25mm 를 흔든다. 그 초기
             기울기는 IMU 영점에 의존하고, IMU 영점이야말로 지금 못 믿는 값.
             → 점추정을 버리고 대역으로 학습한다.
      질량   +0.0 ~ +3.5 kg (총 10.5~14.0 kg)
             URDF base 10.491, payload판 12.001. 승윤님 추정 실물 ~13kg.
             질량은 넘어짐 역학에서 대부분 상쇄되지만(각가속도 = g·d/L²)
             토크 요구와 서보 권한은 비례하므로 대역으로 덮는다.
      IMU    피치·롤 상수 편향 ±5° (v9 의 ±3° 확대)
             근거: imu_mount_calib.py 가 '수직 매달림'을 영점 기준으로 쓰는데,
             줄에 매단 몸은 CoM 이 매단 점 밑에 오도록 걸리므로 베이스가
             연직이 아니다. 그 기울기가 1:1 로 영구 편향이 된다.
             0810 A 실측: −4.5° 편향이 실물 전방추세(+4.48 vs 실물 +4.06)와
             떨림(2.16° vs 실물 2~4°)을 동시에 재현.
      영점   관절 상수 편향 ±3° (v9 유지)
      푸시·지형·게인·지연·마찰: v6 대역 유지 (섭동 판정 원칙)

    v6 을 부모로 고른 이유: v7~v11 이 좁힌 축(CoM·푸시 제거)이 전부 순위
    역전의 원인이었고, v6 대역이 실측 −31.6mm 를 이미 포함하고 있었다.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        # x 상한을 +0.02 → +0.06 으로 넓힘 (0810 정책거동 역산):
        # v6b 를 실물 조건에 넣고 base_com x 만 스윕하니 추세가 깨끗한 직선
        # (추세 = +285.5·x −1.83, r=0.991). 실물 +4.06°/s 를 내는 값은
        # **x = +20.6mm**(전신 CoM 발목축 뒤 8mm)이고, v6b 학습대역
        # (−80~−30mm, 전신 −53~−31mm)에서 100mm 가까이 떨어져 있다.
        # → 정책이 '있지도 않은 후방 편중'을 보정해 앞으로 미는 기전.
        # 0804 트림0 안정(−1.7° 정지)과도 맞고, 0809 역산 −31.6mm 와는
        # 어긋난다(그 사이 m8 시프트·재앵커 — 영점 오염 의심).
        # 어느 쪽이 맞는지 모르므로 **양쪽을 다 담는다.**
        self.events.base_com.params["com_range"] = {
            "x": (-0.10, 0.06),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.15),
        }
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 3.5)
        # 캘리브레이션 상수 편향 — 매 스텝 제로평균 노이즈로는 못 가르친다.
        # 에피소드 내내 유지되는 편향이어야 '치우친 기준에 적응'을 배운다.
        self.events.calib_bias = EventTerm(
            func=mdp_stage4.resample_calib_bias,
            mode="reset",
            params={"joint_bias_rad": (-0.052, 0.052),    # ±3°
                    "imu_tilt_rad": (-0.087, 0.087)},     # ±5°
        )
        self.observations.policy.joint_pos.func = mdp_stage4.joint_pos_rel_biased
        self.observations.policy.projected_gravity.func = \
            mdp_stage4.projected_gravity_biased


class Biped12Stage4V2StandV14EnvCfg(Biped12Stage4V2StandV13EnvCfg):
    """stand_v14 — **실질량 17.62kg · 압력패드 8점 실접촉 자산으로 전환** (0904).

    **왜 자산을 바꾸는가 (0903~0904 실측).**
      - 저울 실측 17.62kg. 구 자산 10.49(+payload 최대 3.5) 로는 어떤 세대도
        실물 질량을 본 적이 없었다. 17.6kg 에서 v6b 낙상 27% → 66% (F3b),
        피치 진동 RMS 1.9~2.3° → 3.1~3.7°(실물 2~4° 대역 진입).
      - 실접촉은 발바닥이 아니라 RA30P 패드 8개(1.6mm 돌출) — usd_final3 는
        패드 콜리전 8개를 갖는다(횡폭 ±38mm, 헐 ±45 보다 좁음 → 좌우 낙상 지배).
      - 0단계 강체홀드(영자세 스폰) 0/64·−0.11°/s 로 실물 정지와 일치 —
        자산 충실도 합격 후 전환한다.

    **질량 대역.** URDF 가 실질량이므로 add_base_mass 는 0 중심:
      −1.0 ~ +1.5 kg (배선·부속·장착물 불확실성, 이전 대역폭 유지).
    **CoM x 대역.** v13 (−0.10~+0.06) 유지 — 실질량 재적합 +14~17mm 포함
      (F3c: 추세 = +492·com_x − 3.82, r 0.972).
    **나머지** IMU ±5°·영점 ±3°·푸시·지형·게인·지연·마찰: v13 그대로.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = (
            "/home/ryu/humanoid_leg_test1/rl_walking/assets/usd_final3/biped12.usd")
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.5)
        # 0905 감사 확정(critical): Biped12Stage4DREnvCfg 의 link_mass_calib(전 링크 ×1.09~1.19,
        # 10.49kg 시절 '실물 12kg 정합'용)가 상속돼 실질량 자산이 19.2~21.0kg 으로 학습·평가됐다.
        # v14~v17 은 이 상태로 학습됨(v18 부터 정정). URDF 가 실측이므로 스케일 1.0 고정.
        if getattr(self.events, "link_mass_calib", None):
            self.events.link_mass_calib.params["mass_distribution_params"] = (1.0, 1.0)


class Biped12Stage4V2StandV14sEnvCfg(Biped12Stage4V2StandV14EnvCfg):
    """stand_v14s — v14 + **착지 비대칭 랜덤화** (승윤님 제안 0904).

    실물은 무게 때문에 발이 대칭으로 착지하지 못한다: 한 발이 5cm 앞, 요가
    몇 도 틀어진 채 스탠딩이 시작된다. reset 마다 발마다 전후 ±3cm(상대차
    최대 6cm)·요 ±5° 를 관절각으로 심어 그 자세에서 시작하게 한다.
    v14 와 이것만 다르다 — 효과 귀속을 위한 대조 세대.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.stance_asym = EventTerm(
            func=mdp_stage4.reset_stance_asymmetry,
            mode="reset",
            params={"fore_aft_m": (-0.03, 0.03), "yaw_rad": (-0.087, 0.087)},
        )


class Biped12Stage4V2StandV15EnvCfg(Biped12Stage4V2StandV14sEnvCfg):
    """stand_v15 — v14s + **배포·캘리브레이션 현실화 3축** (0904 꼬리물기).

    승윤님 진단: "캘리브레이션을 하고 내려도 로봇 무게 때문에 착지부터 비틀려
    있고 랜덤이다. 캘리브레이션 도구도 완벽하지 않다." 정책이 완벽한 로봇을
    본 적이 없게 만든다:
      1. **배포 래퍼 LPF 를 학습에 내장** — 실물 브리지는 정책 출력을 EMA
         β=0.27 로 평활한다. 정책은 자기 출력이 깎이는 걸 본 적이 없었다.
         env 별 β∈[0.2, 1.0] (1.0=필터 없음도 포함해 계약 유지).
      2. **착지 비틀림** — reset 시 몸통 롤·피치 ±3° (실물 착지는 줄·무게로
         항상 조금 기울어 시작). velocity 는 여전히 0 (외란이 아니라 초기조건).
      3. **캘리브레이션 오차 관절군별 확대** — 발목 ±6° (0809 재앵커 시프트
         4~5.5° 실측), 무릎 ±4° (좌우 다리 길이차 ≈ 무릎각 편향), 나머지 ±3°.
      + **물리 실현성 제약**: 토크 포화 벌점(|τ|/τ_lim>0.6) · 소프트리밋 위반 벌점 −5
        (0904 명목 평가: 전 세대 hip_r 하드스톱 풀토크 악용 발견).
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.lpf_beta_range = (0.2, 1.0)
        # 0905 감사: ±3° 는 부모(DRv4 ±4°)보다 좁힌 값이었다 → ±4° 유지. 초기 '속도'는 실물
        # 착지(줄에 매달려 정지)에 없으므로 0 — docstring 과 하네스 실제를 일치시킴 (v18~).
        self.events.reset_base.params["pose_range"]["roll"] = (-0.07, 0.07)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.07, 0.07)
        _vr = self.events.reset_base.params.get("velocity_range")
        if isinstance(_vr, dict):
            for _k in list(_vr):
                _vr[_k] = (0.0, 0.0)
        if getattr(self.events, "reset_robot_joints", None):
            self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.calib_bias.params["imu_roll_rad"] = (-0.052, 0.052)   # 롤 편향 ±3° (실측 +2.6~3.1°)
        # CoM x 대역 하한 −0.10 → −0.05 (0904 v14 판정): 실질량 자산의 명목 CoM 은
        # 발목축 위(+0.2mm, 시뮬 실측)이고 뒤꿈치 지지 한계가 −74mm 라, −0.10 은
        # 물리적으로 설 수 없는 에피소드(강체홀드 DR ON 56/64·배포 DR ON 89% 낙상)
        # 만 보탠다. 상한 +0.06(발끝 73~85mm 안쪽) 유지. 실질량 재적합 +16mm 포함.
        self.events.base_com.params["com_range"]["x"] = (-0.05, 0.06)
        self.events.calib_bias.params["ankle_bias_rad"] = (-0.105, 0.105)
        self.events.calib_bias.params["knee_bias_rad"] = (-0.070, 0.070)
        # ── 물리 실현성 제약 2종 (0904 명목 평가 발견) ─────────────────────────
        # v6b~v14 전 세대가 hip_r 을 하드스톱 ±30° 에 밀어붙여 토크 한계 25Nm 를
        # 43~92% 시간 유지(v6b 는 hip_a 87%). 시뮬에서 공짜인 하드스톱을 악용하는
        # 자세(오리발/안짱다리) — 실물 AK70 연속정격(~피크 1/3) 초과 → 과열·브라운아웃.
        # 보상은 이 둘만 추가/강화 (승윤님: 보상 복잡화 금지, 실물 실현성은 예외).
        from isaaclab.managers import RewardTermCfg as _RewTerm
        self.rewards.torque_saturation = _RewTerm(
            func=mdp_stage4.torque_saturation, weight=-2.0, params={"thresh": 0.6})
        self.rewards.dof_pos_limits.weight = -5.0


class Biped12Stage4V2StandV16EnvCfg(Biped12Stage4V2StandV15EnvCfg):
    """stand_v16 — v15 + **모터·센서 부하 현실화 3축** (0904 꼬리물기).

      1. **토크 한계 env별 0.6~1.0 스케일** — 연속정격·발열 디레이팅·브라운아웃
         (5차 실기 우측 분기). 정책이 피크 토크를 상수로 믿지 않게.
      2. **관절 마찰 확대 0~0.3** (PhysX 관절마찰계수; 0단계 스윕에서 0.2 는
         강체홀드 거동을 눈에 띄게 바꿈) — 감속기 정지마찰·부하 의존 마찰.
      3. **발 접촉 관측 5틱 지연** — 실물 foot_n 이 정확히 5틱(100ms) 스테일
         (IMU 감사 실측). 학습은 0틱이었다.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.effort_scale = EventTerm(
            func=mdp_stage4.resample_effort_scale,
            mode="reset",
            params={"scale_range": (0.6, 1.0)},
        )
        if getattr(self.events, "joint_friction", None):
            self.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.3)
        self.observations.policy.feet_contact.params["delay_steps"] = 5


class Biped12Stage4V2StandV17EnvCfg(Biped12Stage4V2StandV16EnvCfg):
    """stand_v17 — v16 + **엔코더 사각지대·센서 지연 2축** (0904 꼬리물기).

      1. **감속기 백래시 히스테리시스 ±0.3~1.5°** — 명령이 방향을 바꿀 때
         데드밴드 동안 출력이 안 움직인다(출력측 백래시). 정책은 결과 관절각을
         관측하므로 '엔코더 사각지대'(로터는 움직였는데 링크가 안 움직인 경우)
         변형은 아직 미포함 — 0802 실측(기울기 5.7° 에 관절오차 0.7°)의 완전
         재현은 다음 과제.
      2. **IMU 자세 지연 0~3틱** — iAHRS 자세 필터 지연. 자이로(각속도)는 그대로.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.backlash_rad_range = (0.005, 0.026)
        self.events.calib_bias.params["imu_lag_steps"] = (0, 3)
        self.observations.policy.projected_gravity.func = mdp_stage4.projected_gravity_lagged


class Biped12Stage4V2StandV18EnvCfg(Biped12Stage4V2StandV17EnvCfg):
    """stand_v18 — **0905 적대 감사 확정 결함을 반영한 '정정 세대'** (설계 축은 v17 과 동일).

    새 축 없음. v14~v17 이 잘못된 nominal 위에서 학습됐다는 것이 감사(100 에이전트, 확정 31)로
    드러나 그 정정본만 다시 학습한다:
      1. **link_mass_calib 1.0 고정** — 전 링크 ×1.09~1.19 가 상속돼 실질량 자산이 19.2~21.0kg
         으로 학습됐었다. 저울 17.62kg 그대로.
      2. 액션텀 순서 지연→슬루→LPF (실물 브리지: 러너 슬루 → LPF 0.27 → 노드 슬루).
      3. 발접촉 지연 정확히 5틱(off-by-one 수정) + 리셋 초기화, IMU 지연 링버퍼 리셋 초기화.
      4. IMU 롤 장착 편향 ±3° 추가(실측 +2.6~3.1°; 기존은 피치만).
      5. 착지 롤/피치 ±4°(부모 대역 유지), 초기 속도 0(실물 착지는 정지).
      6. effort_scale 이 액추에이터 effort_limit 도 갱신(토크 포화 지표 정합).
    부모 클래스 코드 자체가 정정됐으므로 이 클래스는 '이름표'다 — v17 웜스타트로 6000it.
    """

    def __post_init__(self):
        super().__post_init__()


class Biped12Stage4V2StandV19EnvCfg(Biped12Stage4V2StandV18EnvCfg):
    """stand_v19 — v18 + **액션 지연 대역 3~8틱** (0905 배포 계약 감사 B4, 0906 야간 개선 루프).

    실측 루프지연은 4~8틱(0803 블랙박스 sysid)인데 v15~v18 학습 대역은 3~6틱 — 실측 상단 7~8틱이
    학습 분포 밖이었다. 이 축 하나만 바꾼다. 대조군 = v18c(같은 웜스타트 v18b/120998·같은 6000it·V18 env).
    판정(judge_gen h3 배포 경로)에서 v18c 대비 낙상·E_smth·τ가 같은 방향으로 좋아질 때만 채택 후보.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 3
        self.actions.joint_pos.max_delay_steps = 8


class Biped12Stage4V2StandV20EnvCfg(Biped12Stage4V2StandV19EnvCfg):
    """stand_v20 — v19 + **서보 강성 DR 하한 0.2 → 0.6** (0906 새벽 정착 실험).

    근거: 정책 없이 자세 0 을 PD 홀드(정착 1s)만 해도 DR ON 환경의 62/64 가 뒤로 기운다(피치 평균 −8°, 게이트 ±3° 통과 2/64).
    축 단독 제거 실험에서 원인은 actuator_gains 강성 DR(0.2~1.1×) 하나 — 착지 기울기·CoM·질량·영점·비대칭·마찰·아마추어·지연·
    백래시는 무관. 명목(강성 1.0)은 −1.9° 로 안정하고 실물 트림0 자립(−1.7°, 0804)과 일치한다. 즉 강성 0.2~0.5× 몸체는
    실물이 보인 정적 자세를 재현하지 못하는 DR 표본이다(하한 스윕: 0.5→18/64, 0.6→31/64, 0.8→55/64 통과).
    0.6 은 '정적 자세를 절반 이상 재현' 하는 경계 — 동적 컴플라이언스(구조 강성 43.5Nm/rad 실측)는 SEA 모델로 옮기는 것이 정도이며
    이 세대는 그 전까지의 중간 후보. 대조군 = v19(같은 웜스타트·같은 6000it, 강성 0.2~1.1). 판정은 V6 고정환경·h3·h3g 공통 조건으로만.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.6, 1.1)
