"""12DOF 하체 로봇 RL 보행 학습 패키지 (Isaac Lab v2.3.2 외부 태스크).

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
학습 환경은 실물 이식 제약(kd<=5, 50Hz, 슬루 4°/tick, tau 25Nm,
발목롤 AK45-36 속도한계 5rad/s)을 내장한다.

사용:
    cd /home/ryu/humanoid_leg_test1
    /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py \
        --task Biped12-Velocity-Flat-v0 --headless --num_envs 4096
"""
import gymnasium as gym

# 엔트리포인트는 문자열 — import 시점에 isaaclab을 요구하지 않는다 (lazy)
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
