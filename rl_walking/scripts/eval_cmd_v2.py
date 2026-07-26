"""Stage4 V2 명령추종 실측 — 제자리회전(0,0,0.6) / 후진(-0.3,0,0) 판정용 (headless).

측정: 누적 요 회전(명령 적분 대비 %), 접지 중 발 요슬립 |w_z| 평균,
      케이던스, 좌우 접촉 역위상(피어슨 상관), 좌/우 에어타임 비, 낙상.
spin3.py 기반 — V2 정책, 명령 고정(리샘플 차단), num_envs 통계.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Play-v0")
p.add_argument("--cmd", type=float, nargs=3, required=True, help="vx vy wz")
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--out", required=True, help="리포트 txt 경로")
p.add_argument("--npz", default="", help="접촉 궤적 npz 경로 (선택)")
p.add_argument("--label", default="")
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
ec.scene.num_envs = a.num_envs
ec.episode_length_s = 60.0  # 30s 측정 중 타임아웃 리셋 방지
ec.seed = ac.seed
ec.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
ec.commands.base_velocity.heading_command = False
ec.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)  # 측정 중 리샘플 차단

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
body_feet = [robot.body_names.index(n) for n in FEET]

N = a.num_envs
dev = u.device
CMD = torch.tensor([a.cmd], device=dev).repeat(N, 1)
dt = float(u.step_dt)
dur = a.steps * dt

falls = 0
yaw_acc = torch.zeros(N, device=dev)
prev_yaw = None
slip_sum = 0.0
slip_cnt = 0
steps_cnt = torch.zeros(N, 2, device=dev)
air_L, air_R = [], []
contacts = []
sum_vx_err = 0.0

with torch.inference_mode():
    env.reset()
    ct = u.command_manager.get_term("base_velocity")
    ct.vel_command_b[:] = CMD
    obs = env.get_observations()
    for i in range(a.steps):
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)
        ct.vel_command_b[:] = CMD

        term = u.termination_manager.terminated
        falls += int(term.sum())

        _, _, y = mu.euler_xyz_from_quat(robot.data.root_quat_w)
        if prev_yaw is not None:
            dy = y - prev_yaw
            dy = torch.where(dy > np.pi, dy - 2 * np.pi, dy)
            dy = torch.where(dy < -np.pi, dy + 2 * np.pi, dy)
            dy = torch.where(dones.bool(), torch.zeros_like(dy), dy)  # 리셋 스텝 제외
            yaw_acc += dy
        prev_yaw = y.clone()

        forces = cs.data.net_forces_w[:, sensor_feet, :]
        in_contact = forces.norm(dim=-1) > 1.0  # (N, 2)
        fyv = robot.data.body_ang_vel_w[:, body_feet, 2].abs()
        m = in_contact
        slip_sum += float(fyv[m].sum())
        slip_cnt += int(m.sum())

        fc = cs.compute_first_contact(dt)[:, sensor_feet]
        steps_cnt += fc.float()
        lat = cs.data.last_air_time[:, sensor_feet]
        if fc[:, 0].any():
            air_L.extend(lat[:, 0][fc[:, 0]].cpu().tolist())
        if fc[:, 1].any():
            air_R.extend(lat[:, 1][fc[:, 1]].cpu().tolist())

        vel_yaw = mu.quat_apply_inverse(mu.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        sum_vx_err += float((vel_yaw[:, 0] - a.cmd[0]).abs().mean())

        contacts.append(in_contact.cpu().numpy().copy())

C = np.array(contacts, dtype=float)  # (steps, N, 2)
# 좌우 역위상: env별 좌/우 접촉 시퀀스 피어슨 상관 평균
corrs = []
for e in range(N):
    l, rr = C[:, e, 0], C[:, e, 1]
    if l.std() > 1e-6 and rr.std() > 1e-6:
        corrs.append(float(np.corrcoef(l, rr)[0, 1]))
antiphase = float(np.mean(corrs)) if corrs else float("nan")

mAL = float(np.mean(air_L)) if air_L else 0.0
mAR = float(np.mean(air_R)) if air_R else 0.0
air_ratio = max(mAL, 1e-9) / max(mAR, 1e-9)
air_ratio_sym = max(air_ratio, 1.0 / air_ratio)
cad = float(steps_cnt.sum()) / N / dur
slip_mean = slip_sum / max(slip_cnt, 1)
cmd_yaw_int = a.cmd[2] * dur
yaw_mean = float(yaw_acc.mean())
rot_pct = yaw_mean / cmd_yaw_int * 100 if abs(cmd_yaw_int) > 1e-9 else float("nan")

rep = []
rep.append(f"Stage4 V2 명령추종 실측{(' — ' + a.label) if a.label else ''}")
rep.append(f"checkpoint: {ckpt}")
rep.append(f"조건: {N} env × {dur:.0f}s, cmd=({a.cmd[0]}, {a.cmd[1]}, {a.cmd[2]}) [vx,vy,wz]")
rep.append("=" * 66)
rep.append(f"낙상: {falls}회 / {N}env")
if abs(a.cmd[2]) > 1e-9:
    rep.append(f"누적 요 회전: 평균 {yaw_mean:.2f} rad (명령 적분 {cmd_yaw_int:.1f} rad 대비 {rot_pct:.1f}%)")
    rep.append(f"  env별: {[round(float(v), 2) for v in yaw_acc.cpu().tolist()]}")
rep.append(f"접지 중 발 요슬립 |w_z| 평균: {slip_mean:.3f} rad/s (접촉샘플 {slip_cnt})")
if abs(a.cmd[0]) > 1e-9:
    rep.append(f"속도 추종 오차 |vx-cmd|: 평균 {sum_vx_err / a.steps:.3f} m/s")
rep.append(f"케이던스: {cad:.2f} 걸음/s (양발 첫접촉 합산)")
rep.append(f"좌우 접촉 역위상(피어슨): {antiphase:.3f}")
rep.append(f"에어타임: 좌 {mAL:.3f} s / 우 {mAR:.3f} s — 비 L/R {air_ratio:.2f} (대칭비 {air_ratio_sym:.2f})")
if a.npz:
    np.savez_compressed(a.npz, contact=C, checkpoint=ckpt, cmd=np.array(a.cmd))
    rep.append(f"접촉 궤적 저장: {a.npz}")

open(a.out, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
app.close()
