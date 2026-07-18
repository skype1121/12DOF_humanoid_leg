"""2단계 커리큘럼 — 걷기 발현(런1) 후 강건화·다재화.

변경점 (리서치의 발현→강건화 순서 원칙):
- 명령 확장: 전진 0.2~0.8 + 측방 ±0.25 + 요 ±1.0 + 서기 10% (BHL biped 최종 범위 계열)
- 외란 강화: ±0.5 m/s, 4~8s 간격 (kd5 약축 보강)
- 마찰 확대: 0.2~1.25 (Berkeley Humanoid 최종 범위)
- 걸음새 다듬기: action_rate/joint_vel 페널티 상향 (진동 억제 — 저 kd 이식 대비)

사용: --resume True --load_run <런1> 으로 워밍스타트.
"""
from isaaclab.utils import configclass

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
