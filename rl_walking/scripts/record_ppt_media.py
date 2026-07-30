"""PPT 발표자료 영상 녹화 — 3모드 (RecordVideo, headless).

  march : 제자리걷기 단일 로봇 (model_14993 + PositionHold, 명령 0)
  turn  : 방향 전환 단일 로봇 (R4 model_12994, 서기→좌회전→우회전)
  falls : 학습 초기 연출 — 미학습(랜덤 초기화) 정책 × 병렬 env, 계속 쓰러짐

실행: OMNI_KIT_ACCEPT_EULA=YES isaaclab.sh -p rl_walking/scripts/record_ppt_media.py \
        --mode march --headless
출력: log_picture/_rec/<mode>-step-0.mp4 (판정은 파일 존재로 — isaaclab.sh는 exit0 함정)
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: F401  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["march", "turn", "falls"], required=True)
p.add_argument("--out_dir", default="/home/ryu/humanoid_leg_test1/log_picture/_rec")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import os

import torch

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab.utils.math as mu

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

MODE = {
    "march": dict(task="Biped12-Velocity-Stage4V2-March-Play-v0", envs=1, steps=700,
                  run="2026-07-26_21-41-30", ckpt="model_14993.pt",
                  eye=(2.3, 1.9, 1.25), lookat=(0.0, 0.0, 0.55)),
    "turn": dict(task="Biped12-Velocity-Stage4V2-Play-v0", envs=1, steps=760,
                 run="2026-07-26_14-53-03", ckpt="model_12994.pt",
                 eye=(2.3, 1.9, 1.25), lookat=(0.0, 0.0, 0.55)),
    "falls": dict(task="Biped12-Velocity-Stage4V2-Play-v0", envs=16, steps=330,
                  run=None, ckpt=None,
                  eye=(7.5, 5.5, 3.2), lookat=(0.0, 0.0, 0.4)),
}
m = MODE[a.mode]

ac = load_cfg_from_registry(m["task"], "rsl_rl_cfg_entry_point")
ec = load_cfg_from_registry(m["task"], "env_cfg_entry_point")
ec.scene.num_envs = m["envs"]
ec.seed = 42
ec.episode_length_s = 120.0 if a.mode != "falls" else 20.0
ec.viewer.eye = m["eye"]
ec.viewer.lookat = m["lookat"]
ec.viewer.resolution = (1280, 720)
if a.mode != "falls":
    ec.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    ec.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    ec.commands.base_velocity.heading_command = False   # heading이 wz 덮어쓰기 차단

env = gym.make(m["task"], cfg=ec, render_mode="rgb_array")
os.makedirs(a.out_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(env, video_folder=a.out_dir, name_prefix=a.mode,
                               step_trigger=lambda s: s == 0,
                               video_length=m["steps"], disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device)
if m["run"]:
    ckpt = get_checkpoint_path(
        os.path.abspath(os.path.join("/home/ryu/humanoid_leg_test1", "logs",
                                     "rsl_rl", ac.experiment_name)),
        m["run"], m["ckpt"])
    r.load(ckpt)
    print(f"[record] checkpoint: {ckpt}", flush=True)
else:
    print("[record] 미학습 랜덤 초기화 정책 (학습 0스텝 연출)", flush=True)
pol = r.get_inference_policy(device=env.unwrapped.device)
pnn = r.alg.policy

u = env.unwrapped
robot = u.scene["robot"]
dt = float(u.step_dt)

with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    ct = u.command_manager.get_term("base_velocity")
    if a.mode != "falls":
        ct.is_standing_env[:] = False
    p0 = robot.data.root_pos_w[:, :2].clone()
    for step in range(m["steps"]):
        t = step * dt
        if a.mode == "march":
            # PositionHold 상위 폐루프 (eval_march와 동일 수식, kp 0.8)
            err_w = p0 - robot.data.root_pos_w[:, :2]
            _, _, yaw = mu.euler_xyz_from_quat(robot.data.root_quat_w)
            c, s = torch.cos(yaw), torch.sin(yaw)
            ct.vel_command_b[:, 0] = (0.8 * (c * err_w[:, 0] + s * err_w[:, 1])).clamp(-0.15, 0.15)
            ct.vel_command_b[:, 1] = (0.8 * (-s * err_w[:, 0] + c * err_w[:, 1])).clamp(-0.15, 0.15)
            ct.vel_command_b[:, 2] = 0.0
        elif a.mode == "turn":
            wz = 0.0 if t < 2.0 else (0.6 if t < 8.0 else (-0.6 if t < 14.0 else 0.0))
            ct.vel_command_b[:, 0] = 0.0
            ct.vel_command_b[:, 1] = 0.0
            ct.vel_command_b[:, 2] = wz
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)

env.close()
print(f"[record] DONE mode={a.mode} steps={m['steps']}", flush=True)
app.close()
