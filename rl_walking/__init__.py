"""12DOF 하체 로봇 RL 보행 학습 패키지 (Isaac Lab v2.3.2 외부 태스크).

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
학습 환경은 실물 이식 제약(kd<=5, 50Hz, 슬루 4°/tick, tau 25Nm,
발목롤 AK45-36 속도한계 5rad/s)을 내장한다.

사용:
    cd /home/ryu/humanoid_leg_test1
    /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py \
        --task Biped12-Velocity-Flat-v0 --headless --num_envs 4096
"""
try:
    import gymnasium as gym
except ImportError:      # 배포 장비(젯슨 등)엔 gymnasium 없음 — 등록은 학습 전용,
    gym = None           # deploy/ 하위(policy_runner 등) import 는 그대로 가능해야 함

# 엔트리포인트는 문자열 — import 시점에 isaaclab을 요구하지 않는다 (lazy)
if gym is None:
    def _noop_register(**_kw):
        pass
    class _GymStub:
        register = staticmethod(_noop_register)
    gym = _GymStub()
gym.register(
    id="Biped12-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.env_cfg:Biped12FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.env_cfg:Biped12FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Flat-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage2_env_cfg:Biped12FlatStage2EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Flat-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage2_env_cfg:Biped12FlatStage3EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Flat-Stage3Polish-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage2_env_cfg:Biped12FlatStage3PolishEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_env_cfg:Biped12Stage4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_env_cfg:Biped12Stage4DREnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_env_cfg:Biped12Stage4FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_rough_env_cfg:Biped12Stage4RoughEnvCfg",
        # 러너 재사용: 같은 biped12_stage4 폴더에 이어 쌓음 (--resume 워밍스타트 전제)
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_rough_env_cfg:Biped12Stage4RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Stage4 v2: 회전/후진 명령 강제 배정 + foot_yaw_slip 벌점 (DR 포함)
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2EnvCfg",
        # 러너 재사용: 같은 biped12_stage4 폴더에 이어 쌓음 (미러 대칭 손실 포함)
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Stage4 v2 험지: 서브지형 비율 v2 (slope 0.20·stairs 0.40·flat 0.15)
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 제자리 걷기판: 명령 0 = 클록 리듬 스텝 (정지 불가 — 노드에서 전환)
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Slope-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 경사 보강판: slope 계열 실현 40% (내리막 시작 0.20 + 오르막 시작
        # 0.20 신설) — 계단 상/하 2컬럼씩 v2와 동일 유지 (10% 양자화 설계)
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2SlopeEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Slope-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2SlopeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.rough_env_cfg:Biped12RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.rough_env_cfg:Biped12RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12FlatPPORunnerCfg",
    },
)
