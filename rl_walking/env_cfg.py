"""Biped12 velocity-tracking 학습 환경 (평지) — Isaac Lab manager-based RL env.

설계 근거 (docs/RL보행_작업로그.md, log_picture/00_0242 리서치 참조):
- 실물 정합: 물리 200Hz × decimation 4 = 정책 50Hz (실물 Stage8 노드 틱과 동일),
  슬루 4°/틱 액션(SlewJointPositionAction), kp150/kd5(±20% DR), tau 25Nm.
- 액터 관측 = 실물에서 IMU(iAHRS)+엔코더로 측정 가능한 것만
  (base_lin_vel 제외 — Berkeley Humanoid Lite 방식). 크리틱은 특권 관측 추가.
- 보상 = Isaac Lab G1 biped 항 + 소형 12DOF 스케일 (BHL biped 가중치 계열)
  + 저kd 대응(관절속도/액션레이트 페널티, 접지충격 페널티).
- 서기 함정 회피: air_time 명령 게이팅 + 전진 위주 명령 + 리셋 랜덤화 + 주기 푸시.
- 발이 전후 대칭이므로 인체식 푸시오프 셰이핑 없음 (실측: 역효과).
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .actions import SlewJointPositionActionCfg
from .biped12_cfg import BIPED12_CFG

# 발 링크 (URDF 링크명이 관절명과 동일한 SolidWorks 산출물 특성)
FEET = ".*_ankle_r_joint"


##
# Scene
##


@configclass
class Biped12SceneCfg(InteractiveSceneCfg):
    """평지 + 로봇 + 접촉센서."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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
    robot = BIPED12_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # 텍스처 없는 돔라이트 (Nucleus 다운로드 의존 제거 — 헤드리스 안정성)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 1단계: 전진 위주 (서기 함정 회피 — 0.3m/s 데드존 아래는 안 뽑음)
            lin_vel_x=(0.3, 0.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    # 실물 슬루(4°/틱@50Hz) 내장 위치 액션. scale 0.25rad — 소형 관절가동범위 기준
    joint_pos = SlewJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """액터: 실물 측정 가능 신호만 (IMU 자이로+중력벡터, 엔코더, 직전 액션)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """크리틱 특권 관측 (액터 관측에 추가로 붙음 — obs_groups 참조)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 실물 측정 불가 — 크리틱 전용

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """도메인 랜덤화 — 소형(9.94kg) 스케일. 발현 우선, 강건화는 커리큘럼에서 확대."""

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            # dynamic <= static 물리 일관성 강제 (리뷰 지적 반영)
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # 젯슨/카메라(D455)/배선 미장착분을 설계 단계에서 흡수
            "mass_distribution_params": (-0.5, 1.5),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # 실물 게인 구현 오차·전압 강하 (kd<=5 전이의 핵심 랜덤화)
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # 반사관성 기준값(ak70 0.003 / 발목롤 0.0236)의 0.5~1.5배 — 미실측 오차 흡수
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    # -- reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.08, 0.08),
            "velocity_range": (-0.3, 0.3),
        },
    )

    # -- interval (외란: '서 있기만 하기'를 반드시 실패하게 만드는 장치이기도 함)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """보상 — 근거는 각 항 주석. 가중치 출처: G1 cfg / BHL biped / humanoid-gym."""

    # -- 과제 (양수 우세: 생존+추종이 이득이어야 서기·자폭 함정이 없다)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,  # 소형 12DOF는 강하게 (BHL biped)
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 걷지 않으면 못 얻는 보상 (명령 게이팅 + 단일지지만 인정 → 호핑 차단)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,  # 발현 우선 상향 (G1 0.25 → 걷기 나오면 감축 예정)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 0.4,
        },
    )

    # -- 자세·안정 (kd5 측방 오버슈트 대응이 핵심)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.65}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- 걸음새 품질
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    # 접지 충격 완화 — 저댐핑 측방 진동의 가진원 차단 (Quiet Walking 계열)
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.002,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 150.0,
        },
    )
    # 다리 벌어짐/꼬임 방지 (측방 안정, 사람다운 자세)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_a_joint", ".*_hip_r_joint"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_r_joint"])},
    )

    # -- 정칙화 (저 kd에서 정책이 스스로 감쇠를 만들게 유도)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*", ".*_hip_a_joint"])},
    )

    # -- 접촉·종료
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_f_joint"],
            ),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)  # 소형 스케일
    # 명령 0일 때 기본자세 유지 (서기 모드)
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # 실측 낙상 기준(30°)보다 여유 — 동적 회복 동작을 허용
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})


##
# Env cfg
##


@configclass
class Biped12FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: Biped12SceneCfg = Biped12SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 정책 50Hz = 실물 Stage8 노드 틱과 동일 (이식 시 재보정 불필요)
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Biped12FlatEnvCfg_PLAY(Biped12FlatEnvCfg):
    """재생/평가용: 소수 env, 노이즈·외란 끔, 전진 명령 고정."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
