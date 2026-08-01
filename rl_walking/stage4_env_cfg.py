"""Stage4 — 클록 기반 주기 보상 + 관측 이력 + 미러 대칭 전면 재학습 (평지).

설계 근거:
- Siekmann et al. 2021 (Periodic Reward Composition): 게이트 클록(sin/cos)을 관측에
  넣고 위상-접촉 일치에 보상 — 좌우 π 오프셋으로 대칭 걸음을 구조적으로 강제.
  Stage3 교훈(2026-07-25): 수렴 정책에 대칭 벌점 패치는 게이밍당함(종종걸음/이중딛기).
  → 초기부터 위상으로 형성한다. feet_air_time·step_time_regulator는 클록이 리듬을
  담당하므로 제거 (이중 리듬 신호 충돌 방지).
- Yu et al. 2018 (미러 손실): agents.Biped12Stage4PPORunnerCfg의 symmetry_cfg 참조.
- Lee et al. 2020 (관측 이력): 단일 프레임 대신 이력 5 스택 — 접촉/지연 등
  부분관측 보완 (DR 시 액션 지연 1틱을 정책이 이력으로 추정 가능).

관측 (단일 프레임 49 = 기존 45 + 클록 2 + 접촉 2; 항별 history_length=5):
  정책 입력 245 = 49×5, 크리틱 260 = 245 + base_lin_vel 15. 레이아웃 상수와 미러
  변환은 symmetry.py에 명시 — 항 순서/이력 변경 시 반드시 동기화할 것.

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import mdp_rewards as custom
from . import mdp_stage4
from .env_cfg import Biped12FlatEnvCfg

# 접촉 센서 발 바디 — 반드시 [왼발, 오른발] 순서 + preserve_order=True.
# 접촉센서 바디 나열은 USD DFS 순회 순서라 이름 순서와 다를 수 있는 함정
# (사양·log_picture/03). SceneEntityCfg가 '이름으로' 인덱스를 해석한다.
_FEET_SENSOR_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["left_ankle_r_joint", "right_ankle_r_joint"],
    preserve_order=True,
)


@configclass
class Stage4ObservationsCfg:
    """관측 재정의 — 기존 45 + 클록 2 + 접촉 2, 항별 이력 5 (flatten)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """액터: 실물 측정 가능 신호만 (IMU, 엔코더, 직전 액션, 클록, 발 힘센서).

        클리핑은 물리폭발 NaN 방어 (Stage3Polish ⑥-0 확정 진범 방어 계승) —
        정상 보행 범위의 10배 이상이라 학습 신호 불변. 항 순서 변경 금지
        (symmetry.py 미러 레이아웃과 1:1).
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-20.0, 20.0),
            history_length=5,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-10.0, 10.0),
            history_length=5,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-60.0, 60.0),
            history_length=5,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-10.0, 10.0),
            history_length=5,
            flatten_history_dim=True,
        )
        # 게이트 클록 [sin, cos] — 실물 배포 노드가 동일 공식으로 재현 가능
        gait_clock = ObsTerm(
            func=mdp_stage4.gait_clock,
            params={"freq": mdp_stage4.GAIT_FREQ_HZ},
            history_length=5,
            flatten_history_dim=True,
        )
        # 발 접촉 이진 [L, R] — 실물 RA30P 대응 (>5N)
        feet_contact = ObsTerm(
            func=mdp_stage4.feet_contact_binary,
            params={"sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0},
            history_length=5,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """크리틱 특권 관측 — 동일 이력 5 (크리틱 입력 = policy 245 + 여기 15 = 260)."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-15.0, 15.0),
            history_length=5,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Biped12Stage4FlatEnvCfg(Biped12FlatEnvCfg):
    """Stage4 평지 — 주기 보상 + 이력 관측 + stage2 수준 명령/외란 + 방폭."""

    observations: Stage4ObservationsCfg = Stage4ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # ── 명령 분포: stage2 수준 (전진 0.2~0.8, 측방 ±0.25, 요 ±1.0, 서기 10%) ──
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 외란: stage2 수준 (4~8s 간격, ±0.5 m/s)
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5)
        }

        # ── 주기 보상 (Siekmann 간이판) — 클록이 리듬 담당 ──
        self.rewards.gait_phase = RewTerm(
            func=mdp_stage4.gait_phase_reward,
            weight=2.0,
            params={
                "sensor_cfg": _FEET_SENSOR_CFG,
                "command_name": "base_velocity",
                "freq": mdp_stage4.GAIT_FREQ_HZ,
                "force_threshold": 5.0,
            },
        )
        # 기존 리듬 항 제거 — 클록과 이중 신호 충돌 방지 (step_time_regulator는
        # stage2 계열에만 존재, 본 cfg는 base 상속이라 원래 없음)
        self.rewards.feet_air_time = None

        # ── Stage3Polish에서 검증된 항 유지 ──
        # 발 간격 벌점 (모아걷기·교차보행 방지)
        self.rewards.feet_gap = RewTerm(
            func=custom.feet_too_close,
            weight=-5.0,
            params={
                "min_gap": 0.18,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_r_joint"]),
            },
        )
        # 뒤뚱(롤·피치 각속도) 감쇠 — Polish 확정값
        self.rewards.ang_vel_xy_l2.weight = -0.1

        # ── 방폭 3중 방어 (Stage3Polish ⑤·⑥-0 계승: 캡핑 5종 + 솔버 12/2;
        #    관측 클리핑은 Stage4ObservationsCfg에 선언) ──
        self.rewards.dof_acc_l2.func = custom.joint_acc_l2_capped
        self.rewards.joint_vel_l2.func = custom.joint_vel_l2_capped
        self.rewards.ang_vel_xy_l2.func = custom.ang_vel_xy_l2_capped
        self.rewards.lin_vel_z_l2.func = custom.lin_vel_z_l2_capped
        self.rewards.feet_contact_forces.func = custom.contact_forces_capped
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2


@configclass
class Biped12Stage4DREnvCfg(Biped12Stage4FlatEnvCfg):
    """Stage4-DR — 실물 정합 랜덤화 (stage2_env_cfg.Biped12FlatStage3EnvCfg 계승)."""

    def __post_init__(self):
        super().__post_init__()
        # 후진 보행 복원 (음성 명령 "뒤로 가" — 원 커리큘럼도 발현 단계는 전진만,
        # DR 단계에서 후진 추가했던 방식. 적대 리뷰 지적 반영)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        # 실물 질량 정합: 실측 ~12kg → 전 링크 1.09~1.19 스케일 (11.4~12.5kg 브래키팅)
        self.events.link_mass_calib = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (1.09, 1.19),
                "operation": "scale",  # recompute_inertia 기본 True
            },
        )
        # 페이로드: 배터리 0~5kg (base 가산) + CoM 위/앞/뒤 부착 브래키팅
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.base_com.params["com_range"] = {
            "x": (-0.08, 0.08),   # 앞/뒤 부착
            "y": (-0.03, 0.03),
            "z": (0.0, 0.15),     # 위 부착 (배터리는 아래로 안 달림)
        }
        # 액션 지연 0~1틱 (실측 루프 지연 ~20ms 정합) — 이력 관측이 추정을 보조
        self.actions.joint_pos.max_delay_steps = 1
        # 마찰 확대 (지면: Berkeley Humanoid 최종 범위 / 관절: 0~0.1)
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.15, 1.0)
        self.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.1)


@configclass
class Biped12Stage4DRv2EnvCfg(Biped12Stage4DREnvCfg):
    """Stage4-DRv2 — 실물 컴플라이언스 정합 (2026-08-02 실측 반영 보강학습용).

    근거 (실기 2건):
    ① 첫 자립 기립: 몸통 기울기 5.7°인데 12관절 로터 오차 전부 ≤0.7° —
       로터 엔코더 사각의 직렬 컴플라이언스(감속기 백래시·프린트 구조 휨) 실증.
    ② 첫 정책 라이브(공중): 수 초 내 발진("부들부들→튕김") — 유효강성이
       표기 kp150보다 크게 낮은 실물에서 기존 DR(게인 0.8~1.2)은 부족.
    ImplicitActuator에 직렬탄성(SEA) 모델이 없으므로 1차 근사 = 게인 스케일
    하향 확대 + 위상지연(액션 지연) 확대 + 관절마찰(백래시 근사) 확대.
    질량은 payload 실측 반영: 미모델 전장 1.51kg이 base 뒤(−X)·위 장착 —
    CoM 랜덤화를 뒤쪽 비대칭으로 확장 (com_report.py 실측: 정합 CoM은
    발목 뒤 1~4cm 대역).
    """

    def __post_init__(self):
        super().__post_init__()
        # 유효강성 하향 확대: 0.8~1.2 → 0.4~1.1 (직렬 탄성 1차 근사 — 하한이 핵심)
        self.events.actuator_gains.params[
            "stiffness_distribution_params"] = (0.4, 1.1)
        # 댐핑도 하향 포함: 저댐핑 발진 영역을 학습이 직접 겪게
        self.events.actuator_gains.params[
            "damping_distribution_params"] = (0.5, 1.2)
        # 전장 실측: base 뒤·위 1.51kg — CoM 뒤(−X) 비대칭 브래키팅
        self.events.base_com.params["com_range"] = {
            "x": (-0.20, 0.05),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.15),
        }
        # 컴플라이언스 = 추가 위상지연 → 액션 지연 0~2틱 (기존 0~1)
        self.actions.joint_pos.max_delay_steps = 2
        # 백래시 1차 근사: 관절 마찰 상한 확대
        self.events.joint_friction.params[
            "friction_distribution_params"] = (0.0, 0.15)


@configclass
class Biped12Stage4FlatEnvCfg_PLAY(Biped12Stage4FlatEnvCfg):
    """재생/평가용: 소수 env, 노이즈·외란 끔, 전진 명령 고정 (기존 PLAY 패턴)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
