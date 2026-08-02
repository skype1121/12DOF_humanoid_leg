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
