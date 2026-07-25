"""policy_runner 폐루프 시뮬 검증 — 실행기 경로로 관측을 조립해 보행이 되는지.

검증 내용 (§3-2 오프라인 검증의 시뮬판 — 전부 결정적 판정):
  1. 관측 조립 정합: 러너가 원신호로 조립한 45차원 벡터를 env 관측 버퍼와 직접 대조
  2. 슬루 정합: 러너의 슬루 후 목표각 vs env 액션항 처리 결과 대조
  3. 보행 유지: 15초 무낙상 (전진 거리는 참고 지표 — 단일 env 편차가 커서 판정 제외)

실행: OMNI_KIT_ACCEPT_EULA=YES /home/ryu/IsaacLab/isaaclab.sh -p \
  rl_walking/deploy/sim_validate_runner.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--secs", type=float, default=15.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")

import rl_walking  # noqa: F401  (태스크 등록)
from rl_walking.deploy.policy_runner import JOINT_ORDER, PolicyRunner
from rl_walking.env_cfg import Biped12FlatEnvCfg_PLAY

OUT = "/home/ryu/humanoid_leg_test1/log_picture/실행기_폐루프검증.txt"
MODEL_DIR = "/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_flat/2026-07-25_21-59-19/exported"
CMD = (0.5, 0.0, 0.0)

env_cfg = Biped12FlatEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 60.0
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

env = gym.make("Biped12-Velocity-Flat-Play-v0", cfg=env_cfg)
uenv = env.unwrapped
robot = uenv.scene["robot"]
act_term = uenv.action_manager.get_term("joint_pos")

runner = PolicyRunner(MODEL_DIR)

# 러너 관절순서 == env 관절순서 확인 (이름 기준)
assert list(JOINT_ORDER) == list(robot.data.joint_names), \
    f"관절순서 불일치: {robot.data.joint_names}"

rep = [f"policy_runner 폐루프 검증 — 모델 {MODEL_DIR}", f"명령 {CMD}, {args_cli.secs:.0f}s"]
obs_dict, _ = env.reset()
cmd_term = uenv.command_manager.get_term("base_velocity")

S = int(args_cli.secs * 50)
target_err_max = 0.0
obs_err_max = 0.0
fell = False
runner.reset()
x0 = None

with torch.inference_mode():
    for i in range(S):
        cmd_term.vel_command_b[:] = torch.tensor([CMD], device=uenv.device)
        gyro = robot.data.root_ang_vel_b[0].cpu().numpy()
        grav = robot.data.projected_gravity_b[0].cpu().numpy()
        qpos = robot.data.joint_pos[0].cpu().numpy()
        qvel = robot.data.joint_vel[0].cpu().numpy()

        prev_last = runner.last_action.copy()
        slewed = runner.step(gyro, grav, np.array(CMD), qpos, qvel)
        raw = runner.last_action.copy()
        # 관측 조립 정합: env가 이 틱에 정책에 줬을 관측과 직접 대조
        env_obs = obs_dict["policy"][0].cpu().numpy()
        my_obs = runner.assemble_obs(gyro, grav, np.array(CMD), qpos, qvel, prev_last)
        obs_err_max = max(obs_err_max, float(np.abs(env_obs - my_obs).max()))

        obs_dict, _, terminated, truncated, _ = env.step(
            torch.tensor(raw, dtype=torch.float32, device=uenv.device).unsqueeze(0)
        )
        # env가 처리한 목표각(슬루 후)과 러너 목표각 대조
        env_target = act_term.processed_actions[0].cpu().numpy()
        target_err_max = max(target_err_max, float(np.abs(env_target - slewed).max()))
        if x0 is None:
            x0 = float(robot.data.root_pos_w[0, 0])
        if bool(terminated[0]):
            fell = True
            rep.append(f"!! {i/50:.1f}s 낙상 종료")
            break

dist = float(robot.data.root_pos_w[0, 0]) - (x0 or 0.0)
rep.append(f"전진 거리: {dist:.2f} m ({args_cli.secs:.0f}s, 명령 0.5m/s)")
rep.append(f"러너 vs env 관측 45차원 최대 오차: {obs_err_max:.2e}")
rep.append(f"러너 vs env 목표각 최대 오차: {target_err_max:.2e} rad")
rep.append(f"낙상: {'있음' if fell else '없음'}")
ok = (not fell) and obs_err_max < 1e-4 and target_err_max < 1e-4
rep.append(f"판정: {'PASS' if ok else 'FAIL'}")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
simulation_app.close()
