"""Stage4 험지 — 클록 주기 보상 + 이력 관측 + DR 위에 지형만 추가.

설계:
- Biped12Stage4DREnvCfg 상속: 방폭 3중 방어(페널티 캡핑 5종 + 솔버 12/2 +
  관측 클리핑)와 실물 정합 DR(질량·페이로드·지연·마찰)을 그대로 계승.
  여기서는 rough_env_cfg 패턴의 지형 교체·terrain 커리큘럼·푸시 완화만 얹는다.
- 블라인드 유지: Stage4ObservationsCfg에는 height scan 항이 없다(고유수용감각만).
  rough_env_cfg와 동일하게 지형 관측은 일절 추가하지 않는다 (D455는 향후).
- 지형·커리큘럼: rough_env_cfg의 BIPED12_TERRAINS_CFG(낮은 계단 2~6cm 믹스)와
  Biped12CurriculumCfg(terrain_levels_vel)를 재사용 — 단일 출처 유지.
- 명령 분포는 DR 것 유지(후진 -0.4 포함 — 음성 명령 "뒤로 가" 보존 결정).
  rough_env_cfg의 전진 저속 협소화는 이식하지 않는다.

실행:
  isaaclab.sh -p rl_walking/scripts/train.py --task Biped12-Velocity-Stage4-Rough-v0 \
      --headless --num_envs 4096 --resume True --load_run <Stage4 DR 런>

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .rough_env_cfg import BIPED12_TERRAINS_CFG, Biped12CurriculumCfg
from .stage4_env_cfg import Biped12Stage4DREnvCfg


@configclass
class Biped12Stage4RoughEnvCfg(Biped12Stage4DREnvCfg):
    """Stage4-Rough = Stage4-DR(클록·이력·미러·방폭·DR) + 지형 커리큘럼."""

    def __post_init__(self):
        super().__post_init__()
        # ── 지형 교체: plane → generator + curriculum (rough_env_cfg 패턴 그대로) ──
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=BIPED12_TERRAINS_CFG,
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
        self.rewards.base_height_l2.weight = 0.0   # 계단에선 절대높이 무의미
        # feet_air_time threshold 완화는 이식 불가 — Stage4는 클록이 리듬을 담당해
        # feet_air_time 자체가 제거됨 (stage4_env_cfg 참조)
        # ── 험지 푸시 완화 ±0.3 (계단 모서리 접촉과 중첩 시 폭발 촉발원;
        #    DR의 간격 4~8s는 유지, 세기만 ±0.5 → ±0.3) ──
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.3, 0.3), "y": (-0.3, 0.3)
        }


@configclass
class Biped12Stage4RoughEnvCfg_PLAY(Biped12Stage4RoughEnvCfg):
    """재생/평가용: 소수 env, 노이즈·외란 끔, 저속 전진 고정 (rough PLAY 패턴)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # 전 난이도 열에 골고루 스폰 + 승급 커리큘럼 끔 (rough PLAY 패턴)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
