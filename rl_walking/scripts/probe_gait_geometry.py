"""전진 보행 기하 품질 프로브 — 발 간격·중심 스웨이·롤·스윙 클리어런스.

측정 (32env × 30s, cmd 0.5 전진, 요0 스폰):
- 스탠스 폭: 헤딩 프레임 발 좌우(y) 간격 — 평균/최소/p5 (훈련 벌점 문턱 0.18m 대조)
- 중심 스웨이: 골반 y(헤딩 프레임) 진동 — 스텝 주기 대역 피크-피크
- 롤/피치 각: 평균·p95
- 스윙 클리어런스: 스윙 중 발 최고 높이 (지면 대비)
- 기준 자세 대비: 리셋 직후 스탠스 폭
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Play-v0")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=1500)
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

MARK = open("/tmp/gait_geom_mark.txt", "w")


def mark(s):
    MARK.write(s + "\n")
    MARK.flush()
    print("MARK:" + s, flush=True)


mark("ENV_READY")
GAP, SWAY, ROLL, PITCH, FZ_L, FZ_R, CONT = [], [], [], [], [], [], []
with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    # 리셋 직후 기준 스탠스 폭
    fp0 = robot.data.body_pos_w[:, body_feet, :]
    yq0 = mu.yaw_quat(robot.data.root_quat_w)
    rel0 = mu.quat_apply_inverse(yq0, fp0[:, 0] - fp0[:, 1])
    gap0 = float(rel0[:, 1].abs().mean())
    mark("LOOP_START")
    for i in range(a.steps):
        if i % 250 == 0:
            mark(f"STEP_{i}")
        obs, _, dones, _ = env.step(pol(obs))
        pnn.reset(dones)
        fp = robot.data.body_pos_w[:, body_feet, :]      # (N,2,3)
        yq = mu.yaw_quat(robot.data.root_quat_w)
        rel = mu.quat_apply_inverse(yq, fp[:, 0] - fp[:, 1])  # L−R 헤딩프레임
        GAP.append(rel[:, 1].abs().cpu().numpy().copy())
        root_rel = mu.quat_apply_inverse(yq, robot.data.root_pos_w - robot.data.root_pos_w.mean(0, keepdim=True))
        SWAY.append(robot.data.root_pos_w[:, 1].cpu().numpy().copy())
        rr, pp, _ = mu.euler_xyz_from_quat(robot.data.root_quat_w)
        rr = torch.where(rr > np.pi, rr - 2 * np.pi, rr)
        pp = torch.where(pp > np.pi, pp - 2 * np.pi, pp)
        ROLL.append(rr.cpu().numpy().copy())
        PITCH.append(pp.cpu().numpy().copy())
        FZ_L.append(fp[:, 0, 2].cpu().numpy().copy())
        FZ_R.append(fp[:, 1, 2].cpu().numpy().copy())
        CONT.append((cs.data.net_forces_w[:, sensor_feet, :].norm(dim=-1) > 5.0).cpu().numpy().copy())

mark("LOOP_DONE")
GAP = np.array(GAP); SWAY = np.array(SWAY)
ROLL = np.degrees(np.array(ROLL)); PITCH = np.degrees(np.array(PITCH))
FZ_L = np.array(FZ_L); FZ_R = np.array(FZ_R); CONT = np.array(CONT)

# 발바닥 접지 시 z(오프셋) 기준 스윙 클리어런스
z0_L = FZ_L[CONT[:, :, 0]].mean(); z0_R = FZ_R[CONT[:, :, 1]].mean()
clr_L = []; clr_R = []
for e in range(N):
    zl = FZ_L[:, e]; cl = CONT[:, e, 0]
    zr = FZ_R[:, e]; cr = CONT[:, e, 1]
    # 스윙 구간별 최고점
    for z, c, z0, out in ((zl, cl, z0_L, clr_L), (zr, cr, z0_R, clr_R)):
        t = 0
        while t < len(z):
            if not c[t]:
                s = t
                while t < len(z) and not c[t]:
                    t += 1
                if t - s >= 3:
                    out.append(float(z[s:t].max() - z0))
            else:
                t += 1

# 중심 좌우 스웨이: env별 y 시계열 표준편차·피크-피크(중앙 60% 트림)
sw_std = SWAY.std(axis=0)
sw_pp = np.percentile(SWAY, 95, axis=0) - np.percentile(SWAY, 5, axis=0)

mark("POST_DONE")
rep = []
rep.append("===== 전진 보행 기하 품질 =====")
rep.append(f"checkpoint: {ckpt}")
rep.append(f"조건: {N}env x {a.steps*dt:.0f}s, cmd 0.5m/s, 요0")
rep.append(f"[발 간격] 기준자세 {gap0:.3f}m | 보행 중 평균 {GAP.mean():.3f} | "
           f"p5 {np.percentile(GAP,5):.3f} | 최소 {GAP.min():.3f} (훈련 벌점 문턱 0.18)")
rep.append(f"[중심 스웨이] 좌우 y 표준편차 평균 {sw_std.mean()*100:.1f}cm | "
           f"p5-p95 폭 평균 {sw_pp.mean()*100:.1f}cm")
rep.append(f"[롤] 평균 |roll| {np.abs(ROLL).mean():.2f}도 | p95 {np.percentile(np.abs(ROLL),95):.2f}도")
rep.append(f"[피치] 평균 pitch {PITCH.mean():+.2f}도 (전경사+) | p95 |pitch| {np.percentile(np.abs(PITCH),95):.2f}도")
rep.append(f"[스윙 높이] 좌 평균 {np.mean(clr_L)*100:.1f}cm / 우 {np.mean(clr_R)*100:.1f}cm")
rep.append(f"[접지율] 좌 {CONT[:,:,0].mean()*100:.0f}% / 우 {CONT[:,:,1].mean()*100:.0f}%")
out = "/tmp/gait_geom_result.txt"
open(out, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)

env.close()
app.close()
