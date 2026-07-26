"""제자리 걷기(march) 평가 — '안정적으로 들었다 놨다' 판정.

측정 (32env × 30s, 명령 0):
- 제자리 이탈: 시작점 대비 XY 변위 (평균/최대) — 목표 <0.3m
- 케이던스(채터 제거 0.1s 병합): 목표 2.8 클록 락 | 에어타임 좌우 대칭비 ≤1.1
- 스윙 높이(발 들어올림), 롤 p95, 낙상
결과는 파일 기록 필수 (kit app.close()가 stdout 버퍼를 유실시키는 함정).

실행: isaaclab.sh -p rl_walking/scripts/eval_march.py --headless --load_run <런>
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-March-Play-v0")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--out", default="/home/ryu/humanoid_leg_test1/log_picture/제자리걷기평가.txt")
p.add_argument("--pos_hold", action="store_true",
               help="PositionHold 상위 폐루프 켜기 (위치오차→소속도 명령)")
p.add_argument("--ph_kp", type=float, default=0.8)
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
cs = u.scene.sensors["contact_forces"]
FEET = ("left_ankle_r_joint", "right_ankle_r_joint")
body_feet = [robot.body_names.index(n) for n in FEET]
sensor_feet = [cs.body_names.index(n) for n in FEET]
N = a.num_envs
dt = float(u.step_dt)
dur = a.steps * dt

falls = 0
ROLL, FZ, CONT, DISP = [], [], [], []
with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    p0 = robot.data.root_pos_w[:, :2].clone()
    ct = u.command_manager.get_term("base_velocity")
    if a.pos_hold:
        # standing env 제로화(_update_command)가 pos_hold 명령을 덮어쓰지 않게
        ct.is_standing_env[:] = False
    for _ in range(a.steps):
        if a.pos_hold:
            # PositionHold 벡터판 — deploy.PositionHold.update와 동일 수식
            err_w = p0 - robot.data.root_pos_w[:, :2]        # 월드 오차
            _, _, yaw = mu.euler_xyz_from_quat(robot.data.root_quat_w)
            c, s = torch.cos(yaw), torch.sin(yaw)
            bx = (c * err_w[:, 0] + s * err_w[:, 1])
            by = (-s * err_w[:, 0] + c * err_w[:, 1])
            ct.vel_command_b[:, 0] = (a.ph_kp * bx).clamp(-0.15, 0.15)
            ct.vel_command_b[:, 1] = (a.ph_kp * by).clamp(-0.15, 0.15)
            ct.vel_command_b[:, 2] = 0.0
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)
        falls += int(u.termination_manager.terminated.sum())
        DISP.append((robot.data.root_pos_w[:, :2] - p0).norm(dim=1).cpu().numpy().copy())
        rr, _, _ = mu.euler_xyz_from_quat(robot.data.root_quat_w)
        rr = torch.where(rr > np.pi, rr - 2 * np.pi, rr)
        ROLL.append(rr.cpu().numpy().copy())
        FZ.append(robot.data.body_pos_w[:, body_feet, 2].cpu().numpy().copy())
        CONT.append((cs.data.net_forces_w[:, sensor_feet, :].norm(dim=-1) > 5.0).cpu().numpy().copy())

DISP = np.array(DISP)
ROLL = np.degrees(np.abs(np.array(ROLL)))
FZ = np.array(FZ)          # (T, N, 2)
CONT = np.array(CONT)      # (T, N, 2)

# 채터 제거(공중<0.1s 병합) 케이던스·에어타임 — analyze_cadence와 동일 규칙
MIN_AIR = 5
def foot_stats(c, z):
    c = c.astype(bool).copy()
    t = 0
    while t < len(c):
        if not c[t]:
            s = t
            while t < len(c) and not c[t]:
                t += 1
            if s > 0 and t < len(c) and (t - s) < MIN_AIR:
                c[s:t] = True
        else:
            t += 1
    n, airs, clrs = 0, [], []
    z0 = z[c].mean() if c.any() else 0.0
    t = 1
    while t < len(c):
        if c[t] and not c[t - 1]:
            n += 1
            s = t - 1
            while s >= 0 and not c[s]:
                s -= 1
            airs.append((t - 1 - s) * dt)
            clrs.append(float(z[s + 1:t].max() - z0) if t > s + 1 else 0.0)
        t += 1
    return n, airs, clrs

nL = nR = 0
airL, airR, clrL, clrR = [], [], [], []
for e in range(N):
    n, ar, cl = foot_stats(CONT[:, e, 0], FZ[:, e, 0]); nL += n; airL += ar; clrL += cl
    n, ar, cl = foot_stats(CONT[:, e, 1], FZ[:, e, 1]); nR += n; airR += ar; clrR += cl
cad = (nL + nR) / N / dur
mAL, mAR = np.mean(airL) if airL else 0, np.mean(airR) if airR else 0
ratio = max(mAL, 1e-9) / max(mAR, 1e-9)
sym = max(ratio, 1.0 / ratio)

drift_final = DISP[-1]
ok_drift = float(np.mean(drift_final < 0.3)) * 100
verdict = ("PASS" if falls == 0 and np.mean(drift_final) < 0.3 and 2.5 <= cad <= 3.1
           and sym <= 1.1 else "FAIL")

rep = [
    "제자리 걷기(march) 평가 — '들었다 놨다' 판정",
    f"checkpoint: {ckpt}",
    f"조건: {N}env x {dur:.0f}s, 명령 (0,0,0)",
    "=" * 64,
    f"낙상: {falls}회",
    f"제자리 이탈(30s): 평균 {np.mean(drift_final):.2f}m | 최대 {np.max(drift_final):.2f}m"
    f" | <0.3m 비율 {ok_drift:.0f}%",
    f"케이던스(채터제거): {cad:.2f} 걸음/s (클록 2.8)",
    f"에어타임: 좌 {mAL:.3f}s / 우 {mAR:.3f}s — 대칭비 {sym:.2f}",
    f"스윙 높이: 좌 {np.mean(clrL)*100:.1f}cm / 우 {np.mean(clrR)*100:.1f}cm",
    f"롤 p95: {np.percentile(ROLL, 95):.2f}도",
    f"판정: {verdict} (기준: 낙상0 · 이탈평균<0.3m · 케이던스 2.5~3.1 · 대칭비<=1.1)",
]
open(a.out, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
app.close()
