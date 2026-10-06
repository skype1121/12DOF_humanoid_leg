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
    id="Biped12-Velocity-Flat-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.env_cfg:Biped12FlatEnvCfg_DEMO",
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
    id="Biped12-Velocity-Stage4-DRv2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 0802 실측(직렬 컴플라이언스·전장 CoM) 반영 보강학습 — DR에서 --resume 권장
        "env_cfg_entry_point": "rl_walking.stage4_env_cfg:Biped12Stage4DRv2EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4-DRv3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_env_cfg:Biped12Stage4DRv3EnvCfg",
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
    id="Biped12-Velocity-Stage4V2-March-DRv2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # march + 0802 컴플라이언스 정합 — 구 march(14993)에서 --resume 권장
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchDRv2EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-DRv3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchDRv3EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-DRv4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchDRv4EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-DRv5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchDRv5EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-DRv5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchDRv5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-DRv6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandDRv6EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV5EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV6EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-V6c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchV6cEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-V6b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchV6bEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-March-V6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2MarchV6EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV10EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV12EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV18EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V19-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV19EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV20EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V17-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV17EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV16EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V15-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV15EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V14s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV14sEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V14-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV14EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V13-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV13EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV9EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV8EnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
    },
)

gym.register(
    id="Biped12-Velocity-Stage4V2-Stand-V7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.stage4_v2_env_cfg:Biped12Stage4V2StandV7EnvCfg",
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


# ── Hanging-to-Stand (0908 저녁): 보조지지 스케줄 커리큘럼 S1/S2/S3 — rl_walking/hang_stand_env_cfg.py ──
for _stage in ("S1", "S2", "S3", "S1r", "S2r", "S3r", "S4m", "S5s", "S5k", "S5u", "S5r", "S5v", "S5w", "S6", "S7", "S7b", "S7c", "S7d", "S8a", "S8aS1r", "S8aS3r", "S8b", "S8n", "S8an", "S8ac", "S8acL", "S8nc", "S8anc", "S8c", "S8ac3", "S8a3", "S8ac4", "S8ac5", "S8ac6", "S9a", "S9b", "S10a", "CableS3r", "CableFS3r", "S8ac4d", "S8ac4q", "S8ac4r", "S8ac4rn", "S8ac4rp", "S8ac4rq", "S8ac4r5", "S8ac4s", "S8ac4k", "S8ac4g", "S8ac4kg", "S8ac4k5", "S8ac4k4", "S8ac4k6", "S8ac4z", "S8ac4rd", "S8ac4rdp", "S8ac4rdA", "S9c", "S9cHold", "S9d", "S9dHold", "G1", "G1Hold", "G2", "G2Hold", "G3", "G3Hold", "G4", "G4Hold", "G5", "G5Hold"):   # *r = 축소 DR (0908 2차), S4m = 혼합 스케줄+실측 대역 DR (3차)
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.hang_stand_env_cfg:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPORunnerCfg",
        },
    )

# 0920 G6: 홀드 사전 러너(init_noise_std 0.25) + 실측 대역 DR
for _stage in ("G6", "G6Hold", "G7", "G7Hold",    # 0921 G7: 지속 외력·회복 스텝(같은 홀드 사전 러너)
                "G8", "G8a", "G8b", "G8c", "G8d", "G8Hold",   # 0921 G8: 회복 보너스·상태 게이트 수정 + 분리 실험
                "G9", "G9Hold",                               # 0921 G9: 무릎 밴드 보상 게이트 단일 변경
                "G10", "G10Hold",                             # 0922 G10: 명령 기준 한계 벌점 단일 변경
                "G11a", "G11aHold", "G11b", "G11bHold",
                "G11c", "G11cHold",
                "G12", "G12Hold", "G13", "G13Hold", "G14a", "G14aHold", "G14b", "G14bHold",
                "G15q", "G15u", "G15w", "G15d", "G15k", "G16o", "G16a", "G16i", "G16d", "G17o", "G17r", "G17p", "G17rp", "G18o", "G18d", "G18q", "G18f", "G19v", "G19n", "G20d", "G20s", "G21a1", "G21a2", "G21a3", "G21a", "G21b", "G21c1", "G21c2", "G21d", "G22a", "G22c", "G22w"):                            # 0929 G12: 액션 지연 3~8 → 9~14 단일 변경                          # 0929 G11c: 회복 품질 보상 5→10 단일 변경     # 0928 G11: 발목 정지 마찰+편향 절반 / +정착 보너스 게이트 해제
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.hang_stand_env_cfg:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPOHoldPriorRunnerCfg",
        },
    )


# 1003 G22R·W1 (승윤님 설계 지시서): 부모 G21a-11300 독립 두 분기 — 로그 실험 디렉터리 분리(biped12_g22r / biped12_w1)
for _stage, _agent in (("G22R", "Biped12G22RPPORunnerCfg"), ("W1", "Biped12W1PPORunnerCfg"), ("W1b", "Biped12W1PPORunnerCfg"), ("G22R2", "Biped12G22RPPORunnerCfg"), ("StepB0", "Biped12StepB0PPORunnerCfg"), ("StepB1", "Biped12StepB0PPORunnerCfg"), ("StepI0", "Biped12StepB0PPORunnerCfg"), ("StepA0", "Biped12StepB0PPORunnerCfg"), ("StepA0b", "Biped12StepB0PPORunnerCfg"), ("StepA0c", "Biped12StepB0PPORunnerCfg"), ("StepA0d", "Biped12StepB0PPORunnerCfg"), ("StepA0e", "Biped12StepB0PPORunnerCfg"), ("StepA0f", "Biped12StepB0PPORunnerCfg"), ("StepA0g", "Biped12StepB0PPORunnerCfg"), ("StepA0h", "Biped12StepB0PPORunnerCfg"), ("StepA0i", "Biped12StepB0PPORunnerCfg"), ("StepA0j", "Biped12StepB0PPORunnerCfg"), ("StepA0ic", "Biped12StepB0PPORunnerCfg"), ("StepA0k", "Biped12StepB0PPORunnerCfg"), ("StepA0l", "Biped12StepB0PPORunnerCfg"), ("StepA0m", "Biped12StepB0PPORunnerCfg"), ("StepA0n", "Biped12StepB0PPORunnerCfg"), ("W2a", "Biped12StepB0PPORunnerCfg"), ("G23Ra", "Biped12StepB0PPORunnerCfg"), ("W2b", "Biped12StepB0PPORunnerCfg"), ("G23Rb", "Biped12StepB0PPORunnerCfg"), ("W2c", "Biped12StepB0PPORunnerCfg"), ("W2d", "Biped12StepB0PPORunnerCfg"), ("W2e", "Biped12StepB0PPORunnerCfg"), ("G24Ra", "Biped12StepB0PPORunnerCfg"), ("W2f", "Biped12StepB0PPORunnerCfg"), ("G24Rb", "Biped12StepB0PPORunnerCfg"), ("W2g", "Biped12StepB0PPORunnerCfg"), ("G24Rc", "Biped12StepB0PPORunnerCfg"), ("W2h", "Biped12StepB0PPORunnerCfg"), ("W2i", "Biped12StepB0PPORunnerCfg"), ("W2j", "Biped12StepB0PPORunnerCfg"), ("G24Rd", "Biped12StepB0PPORunnerCfg"), ("G24Re", "Biped12StepB0PPORunnerCfg"), ("G24Rf", "Biped12StepB0PPORunnerCfg"), ("G24Rg", "Biped12StepB0PPORunnerCfg"), ("G24Rh", "Biped12StepB0PPORunnerCfg"), ("G25", "Biped12StepB0SafePPORunnerCfg"), ("G24RaSafe", "Biped12StepB0SafePPORunnerCfg"), ("G25D", "Biped12StepB0SafePPORunnerCfg"), ("W2hSafe", "Biped12StepB0SafePPORunnerCfg"), ("W2k", "Biped12StepB0SafePPORunnerCfg"), ("G24Rcue", "Biped12StepB0SafePPORunnerCfg"), ("G24Rcap", "Biped12StepB0SafePPORunnerCfg"), ("W2v", "Biped12StepB0SafePPORunnerCfg"), ("W2v2", "Biped12StepB0SafePPORunnerCfg"), ("W2w", "Biped12StepB0SafePPORunnerCfg"), ("W2s", "Biped12StepB0SafePPORunnerCfg"), ("W2o", "Biped12StepB0PPORunnerCfg"), ("W2q", "Biped12StepB0PPORunnerCfg"), ("W2qS", "Biped12StepB0PPORunnerCfg"), ("W2qe", "Biped12StepB0PPORunnerCfg"), ("W2q2", "Biped12StepB0PPORunnerCfg"), ("W2q2e", "Biped12StepB0PPORunnerCfg"), ("W2r", "Biped12StepB0PPORunnerCfg"), ("W2re", "Biped12StepB0PPORunnerCfg"), ("W2gs", "Biped12StepB0PPORunnerCfg"), ("W2gse", "Biped12StepB0PPORunnerCfg"), ("WOM", "Biped12StepB0PPORunnerCfg"), ("WOMX", "Biped12StepB0PPORunnerCfg"), ("G24RaD5", "Biped12StepB0PPORunnerCfg"), ("G24RaR4", "Biped12StepB0PPORunnerCfg"), ("W2om", "Biped12StepB0PPORunnerCfg"), ("W2omA", "Biped12StepB0PPORunnerCfg"), ("W2omL", "Biped12StepB0PPORunnerCfg"), ("W2omM", "Biped12StepB0PPORunnerCfg"), ("W2omP", "Biped12StepB0PPORunnerCfg"), ("W2omP2", "Biped12StepB0PPORunnerCfg"), ("W2omP3", "Biped12StepB0PPORunnerCfg"), ("W2omI", "Biped12StepB0PPORunnerCfg"), ("W2omR1", "Biped12StepB0PPORunnerCfg"), ("W2omR2", "Biped12StepB0PPORunnerCfg"), ("W2omF", "Biped12StepB0PPORunnerCfg"), ("W2omFe", "Biped12StepB0PPORunnerCfg"), ("W2ome", "Biped12StepB0PPORunnerCfg"), ("W2omd", "Biped12StepB0PPORunnerCfg"), ("W3Base", "Biped12StepB0PPORunnerCfg"), ("W3A", "Biped12StepB0PPORunnerCfg"), ("W3C", "Biped12StepB0PPORunnerCfg"), ("W3U", "Biped12StepB0PPORunnerCfg"), ("W3CI", "Biped12StepB0PPORunnerCfg"), ("W3A2", "Biped12StepB0PPORunnerCfg"), ("W3S1", "Biped12StepB0PPORunnerCfg"), ("W3S1b", "Biped12StepB0PPORunnerCfg"), ("W3A3", "Biped12StepB0PPORunnerCfg")):
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.hang_stand_env_cfg:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": f"rl_walking.agents:{_agent}",
        },
    )
