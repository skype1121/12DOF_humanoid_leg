"""1009 새 몸(hip_r 양쪽 기구 고정 + PLA 링크) 학습 태스크 등록(학습 전용) — 젯슨 배포 묶음에 들어가는 rl_walking/__init__.py 를 바꾸지 않으려고 따로 둔다.
학습 스크립트(rl_walking/scripts/train.py)가 환경변수 RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1009 일 때만 이 모듈을 읽는다.
환경 설정: rl_walking/newbody_1009.py · 사전 기준: reports/real_1009/NEWBODY_TRAIN.md"""
import gymnasium as gym

# 회복(G24R-a) 새 몸 판 — 에이전트 설정은 G24R-a 와 같은 것(Biped12StepB0PPORunnerCfg, 실험 폴더 biped12_stepb0)
for _stage in ("G24RaNB",):
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.newbody_1009:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
        },
    )

# 1010 2세대: G24RaNB + 앞 외란 힘 범위 상한 확대(단일 변경) — 위 등록과 같은 에이전트 설정
gym.register(
    id="Biped12-HangStand-G24RaNBf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandG24RaNBfEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1010 2단계(종종걸음): 걷기 태스크 W3S3d + 새 몸 — 기준선 평가용(에이전트 설정은 W3 계열과 같은 것)
gym.register(
    id="Biped12-HangStand-W3S3dNB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1010 종종걸음 학습 태스크: W3S3dNB + 명령 혼합(제자리 0.5·앞뒤 0.5·옆 0·회전 0)
gym.register(
    id="Biped12-HangStand-W3S3dNBj-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1010 걷는 중 좌우 흔들림 시험: NBj + 좌우 기울기 벌점 한 항
gym.register(
    id="Biped12-HangStand-W3S3dNBjR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjREnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1010 빠른 걸음: NBj + 걸음 시계 1.5 Hz
gym.register(
    id="Biped12-HangStand-W3S3dNBjF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjFEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1010 밤: NBj + 걷는 중 과요구 벌점(자르기 전 요구 토크)
gym.register(
    id="Biped12-HangStand-W3S3dNBjT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjTEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1011: NBjT + 힙 외전 토크 한계 0.55–0.8 배·나머지 0.8–1.0 배(리셋마다)
gym.register(
    id="Biped12-HangStand-W3S3dNBjL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjLEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)

# 1011: NBjL 완화판 — 힙 외전 0.72–0.92 배(18–23 N·m)·나머지 0.9–1.0 배
gym.register(
    id="Biped12-HangStand-W3S3dNBjM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_walking.newbody_1009:Biped12HangStandW3S3dNBjMEnvCfg",
        "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
    },
)
