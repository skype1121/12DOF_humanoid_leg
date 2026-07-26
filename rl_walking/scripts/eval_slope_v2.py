"""경사 단독 통과율 평가 (P5 신설) — Stage4V2-Rough 정책을 경사 100% 지형에서 판정.

지형: v2 서브지형 중 slope만 남기고 비율 1.0 — 난이도 0.5~1.0 (경사 ≈ 10~20°).
측정: 낙상 env 비율, vx 추종오차, 직립도(중력투영), 케이던스. 20s × 32env.
판정: 낙상 <10% PASS, 10~30% 한계, >30% FAIL (페이로드 평가와 동일 기준).
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Rough-Play-v0")
p.add_argument("--num_envs", type=int, default=0, help="0=태스크 기본값(32)")
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--difficulty", type=float, nargs=2, default=(0.5, 1.0))
p.add_argument("--out", required=True)
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import os

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

import gymnasium as gym

import isaaclab.utils.math as mu

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

T = a.task
ac = load_cfg_from_registry(T, "rsl_rl_cfg_entry_point")
ac = cli_args.update_rsl_rl_cfg(ac, a)
ec = load_cfg_from_registry(T, "env_cfg_entry_point")
if a.num_envs > 0:
    ec.scene.num_envs = a.num_envs
ec.episode_length_s = 60.0
ec.seed = ac.seed
ec.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

# ── 경사 단독 지형: slope만 남기고 비율 1.0, 난이도 고정 범위 ──
tg = ec.scene.terrain.terrain_generator
tg.sub_terrains = {"slope": tg.sub_terrains["slope"]}
tg.sub_terrains["slope"].proportion = 1.0
tg.curriculum = False
tg.difficulty_range = tuple(a.difficulty)
slope_lo = tg.sub_terrains["slope"].slope_range[1] * a.difficulty[0]
slope_hi = tg.sub_terrains["slope"].slope_range[1] * a.difficulty[1]

env = gym.make(T, cfg=ec)
env = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device)
ckpt = get_checkpoint_path(
    os.path.abspath(os.path.join("/home/ryu/humanoid_leg_test1", "logs", "rsl_rl", ac.experiment_name)),
    ac.load_run, ac.load_checkpoint)
r.load(ckpt)
pol = r.get_inference_policy(device=env.unwrapped.device)
pnn = r.alg.policy

u = env.unwrapped
robot = u.scene["robot"]
cs = u.scene.sensors["contact_forces"]
FEET = ("left_ankle_r_joint", "right_ankle_r_joint")
sensor_feet = [cs.body_names.index(n) for n in FEET]
N = u.num_envs
dt = float(u.step_dt)
dur = a.steps * dt
CMD_X = float(ec.commands.base_velocity.ranges.lin_vel_x[0])

falls = torch.zeros(N, device=u.device)
sum_vx_err = 0.0
lean_sum = 0.0
steps_cnt = torch.zeros(N, 2, device=u.device)

with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    for _ in range(a.steps):
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)
        falls += u.termination_manager.terminated.float()
        vel_yaw = mu.quat_apply_inverse(
            mu.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        sum_vx_err += float((vel_yaw[:, 0] - CMD_X).abs().mean())
        pg = robot.data.projected_gravity_b
        lean_sum += float(torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1, 1))).mean())
        steps_cnt += cs.compute_first_contact(dt)[:, sensor_feet].float()

n_fall = int((falls > 0).sum())
rate = n_fall / N * 100
verdict = "PASS" if rate < 10 else ("한계" if rate <= 30 else "FAIL")
cad = float(steps_cnt.sum()) / N / dur

rep = [
    "경사 단독 통과율 평가 (P5) — slope 100% 지형",
    f"checkpoint: {ckpt}",
    f"조건: {N} env × {dur:.0f}s, cmd_x={CMD_X} m/s, 난이도 {a.difficulty[0]}~{a.difficulty[1]}"
    f" (경사 {np.degrees(slope_lo):.1f}~{np.degrees(slope_hi):.1f}°)",
    "=" * 66,
    f"낙상: {n_fall}/{N} env ({rate:.0f}%) → {verdict}",
    f"vx 추종오차: {sum_vx_err / a.steps:.3f} m/s",
    f"평균 직립도(중력 대비): {lean_sum / a.steps:.2f}°",
    f"케이던스: {cad:.2f} 걸음/s",
]
open(a.out, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
app.close()
