"""2·3단계 커리큘럼 — 걷기 발현(런1) 후 강건화·다재화·이식정합.

2단계 (발현→강건화 순서 원칙):
- 명령 확장: 전진 0.2~0.8 + 측방 ±0.25 + 요 ±1.0 + 서기 10% (BHL biped 최종 범위 계열)
- 외란 강화: ±0.5 m/s, 4~8s 간격 (kd5 약축 보강)
- 마찰 확대: 0.2~1.25 (Berkeley Humanoid 최종 범위)
- 걸음새 다듬기: action_rate/joint_vel 페널티 상향 (진동 억제 — 저 kd 이식 대비)

3단계 (페이로드 + sim2real 갭 최소화 — 승윤님 요청 2026-07-19 새벽):
- 실물 질량 정합: URDF 9.94kg vs 실물 ~12kg → 전 링크 1.15~1.26 스케일 (관성 재계산)
- 페이로드 0~5kg: base 질량 + CoM 오프셋 (배터리 위/앞/뒤 부착 브래키팅)
- 액션 지연 0~1틱(0~20ms) env별 랜덤 — 실측 루프 지연 정합
- 관절 마찰 확대 (0~0.1)

사용: --resume --load_run <직전 런> 으로 워밍스타트 (관측/액션 차원 불변).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .env_cfg import Biped12FlatEnvCfg


@configclass
class Biped12FlatStage2EnvCfg(Biped12FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 명령 다양화 (서기 혼합 — stand_still 보상이 이제 실제로 작동)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 외란 강화
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5)
        }
        # 마찰 확대
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.15, 1.0)
        # 부드러움 (실물 저 kd 진동 대비)
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_vel_l2.weight = -1.0e-3
        # 걷기는 이미 발현 — 에어타임 가중 절반으로 (품질 항으로 강등)
        self.rewards.feet_air_time.weight = 0.5


@configclass
class Biped12FlatStage3EnvCfg(Biped12FlatStage2EnvCfg):
    """3단계: 페이로드(배터리 0~5kg 위/앞/뒤) + 실물 질량 정합 + 지연 랜덤화."""

    def __post_init__(self):
        super().__post_init__()
        # 실물 질량 정합: 실측 ~12kg → 전 링크 1.15~1.26 스케일 (11.5~12.5kg 브래키팅)
        self.events.link_mass_calib = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (1.15, 1.26),
                "operation": "scale",  # recompute_inertia 기본 True
            },
        )
        # 페이로드: 배터리 0~5kg (질량은 base에 가산, CoM은 위/앞/뒤 부착 범위)
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.base_com.params["com_range"] = {
            "x": (-0.08, 0.08),   # 앞/뒤 부착
            "y": (-0.03, 0.03),
            "z": (0.0, 0.15),     # 위 부착 (배터리는 아래로 안 달림)
        }
        # 액션 지연 0~1틱 (실측 루프 지연 ~20ms 정합)
        self.actions.joint_pos.max_delay_steps = 1
        # 관절 마찰 확대
        self.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.1)
