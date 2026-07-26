"""측면 푸시 강건성 A/B 프로브 — 심투리얼 1순위 원칙의 직접 측정.

절차: 64env × 30s, cmd 0.5 전진(요0, 리샘플 차단). 5초마다 몸 기준 순수 측면
방향(좌우 교대)으로 속도 임펄스 주입 — 등급 0.3/0.5/0.7/1.0 m/s별 별도 롤아웃.
지표: 낙상 env 비율(등급별), 결과는 파일 기록 (kit 종료 시 stdout 유실 방어).
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Play-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

import gymnasium as gym

import isaaclab.utils.math as mu

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

ac = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
ac = cli_args.update_rsl_rl_cfg(ac, a)
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point")
ec.scene.num_envs = a.num_envs
ec.seed = ac.seed
ec.episode_length_s = 60.0
ec.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
ec.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

env = gym.make(a.task, cfg=ec)
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
N = a.num_envs
STEPS = 1500  # 30s
PUSH_EVERY = 250  # 5s
MAGS = [0.3, 0.5, 0.7, 1.0]


@torch.inference_mode()
def rollout(mag):
    env.reset()
    obs = env.get_observations()
    falls = torch.zeros(N, device=u.device)
    for i in range(STEPS):
        if i > 0 and i % PUSH_EVERY == 0:
            sign = 1.0 if (i // PUSH_EVERY) % 2 == 0 else -1.0
            yq = mu.yaw_quat(robot.data.root_quat_w)
            lat = torch.zeros(N, 3, device=u.device)
            lat[:, 1] = sign * mag  # 몸 기준 좌우축(y) — 전진축은 x
            push_w = mu.quat_apply(yq, lat)
            v = robot.data.root_vel_w.clone()
            v[:, :3] += push_w
            robot.write_root_velocity_to_sim(v)
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)
        falls += u.termination_manager.terminated.float()
    return int((falls > 0).sum())


rep = [f"측면 푸시 강건성 — {a.label}", f"checkpoint: {ckpt}",
       f"조건: {N}env x 30s, cmd 0.5 전진, 5s마다 좌우 교대 측면 속도 임펄스",
       "=" * 60]
for m in MAGS:
    nf = rollout(m)
    rate = nf / N * 100
    verdict = "PASS" if rate < 10 else ("한계" if rate <= 30 else "FAIL")
    line = f"푸시 {m:.1f} m/s | 낙상 {nf:3d}/{N} ({rate:4.0f}%) | {verdict}"
    rep.append(line)
    print(line, flush=True)
open(a.out, "w").write("\n".join(rep) + "\n")
print("saved:", a.out, flush=True)
env.close()
app.close()
