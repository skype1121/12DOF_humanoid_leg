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
