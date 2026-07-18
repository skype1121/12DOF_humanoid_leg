"""Biped12 험지/낮은 계단 환경 — 평지 성공 후의 스트레치 목표.

설계:
- 블라인드(고유수용감각만): 실물에 지형 센서 배선이 아직 없음 (D455는 향후;
  Blackwell TiledCamera 행업 이슈로 비전 학습은 현 스택에서 보류).
  ANYmal 블라인드 계단 선례대로 저단차는 촉각-반응으로 충분.
- 낮은 계단: 단높이 2~6cm (로봇 다리 ~40cm, 발 18cm 스케일에 맞춤).
- terrain curriculum: 쉬운 지형에서 시작해 성공 시 어려운 열로 승급.
- 평지 정책 체크포인트에서 --resume으로 이어 학습 (커리큘럼 워밍스타트).

실행:
  isaaclab.sh -p rl_walking/scripts/train.py --task Biped12-Velocity-Rough-v0 \
      --headless --num_envs 4096 --resume True --load_run <평지런>
"""
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .stage2_env_cfg import Biped12FlatStage3EnvCfg

# 소형 이족용 저난도 지형 믹스 (단위 m)
BIPED12_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,          # 난이도 열 (커리큘럼 축)
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.01, 0.04), noise_step=0.01,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.02, 0.06),   # 낮은 계단 2~6cm
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.02, 0.06),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.2), platform_width=2.0,
            border_width=0.25,
        ),
        # 연석/문턱 모사: 불규칙 높이 격자 (턱 2~5cm)
        "curbs": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.05),
            platform_width=2.0,
        ),
    },
)


@configclass
class Biped12RoughEnvCfg(Biped12FlatStage3EnvCfg):
    """험지 = 3단계(실물질량·페이로드·지연) 위에 지형만 추가 — 올인원 최종형."""

    def __post_init__(self):
        super().__post_init__()
        # 지형 교체: plane → generator + curriculum
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
        # 험지에선 자세 페널티 완화 + 무릎 들어올림 여유
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0.0   # 계단에선 절대높이 무의미
        self.rewards.feet_air_time.params["threshold"] = 0.4
        # 명령: 전진 위주 저속 (계단은 요 회전 최소화)
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)


@configclass
class Biped12CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class Biped12RoughEnvCfg_PLAY(Biped12RoughEnvCfg):
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
