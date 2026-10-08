"""1008 밤 추가 학습 태스크 등록(학습 전용) — 젯슨 배포 묶음에 들어가는 rl_walking/__init__.py 를 바꾸지 않으려고 따로 둔다.
학습 스크립트(rl_walking/scripts/train.py)가 환경변수 RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1008 일 때만 이 모듈을 읽는다."""
import gymnasium as gym

for _stage in ("W3S3x",):
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.hang_stand_env_cfg:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12StepB0PPORunnerCfg",
        },
    )

# 1008 낮: 매달림 시작 혼합 서기 세대(G21a 계열 — 에이전트 설정은 G 계열과 같은 것)
for _stage in ("G21h", "G21h2"):
    gym.register(
        id=f"Biped12-HangStand-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"rl_walking.hang_stand_env_cfg:Biped12HangStand{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": "rl_walking.agents:Biped12Stage4PPOHoldPriorRunnerCfg",
        },
    )
