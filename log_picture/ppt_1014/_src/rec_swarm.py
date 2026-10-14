"""발표용: 걷기 정책을 많은 병렬 로봇에 넣고 카메라가 한 대 → 수십 대 → 오른쪽으로 훑는 장면을 프레임으로 저장.

저장소 밖(scratchpad) 임시 스크립트 — 저장소 코드·설정은 건드리지 않는다.
환경 설정은 평가기(eval_w3.py)와 같은 방식: 보조 0·커리큘럼 끔·밀기 외란 끔·명령 고정.
실행: isaaclab.sh -p -u rec_swarm.py --headless --out <dir> [--preview]
출력: <out>/f%05d.jpg + <out>/stats.json  (판정은 파일로)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: F401  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3d-v0")
p.add_argument("--ckpt", default="/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt")
p.add_argument("--out", required=True)
p.add_argument("--envs", type=int, default=1024)
p.add_argument("--spacing", type=float, default=2.2)
p.add_argument("--cmd", default="0.10,0,0", help="'vx,vy,wz' 고정 명령, 또는 native(학습 환경의 명령 분포 그대로)")
p.add_argument("--warm", type=int, default=250, help="녹화 전 진행 틱")
p.add_argument("--dur", type=float, default=17.0, help="녹화 길이[s]")
p.add_argument("--every", type=int, default=1, help="몇 틱마다 한 장(1 = 50 fps)")
p.add_argument("--res", default="1920x1080")
p.add_argument("--seed", type=int, default=3)
p.add_argument("--preview", action="store_true", help="경로 위 몇 장만 저장")
p.add_argument("--path", default="C")
p.add_argument("--lead", type=float, default=5.0, help="주인공을 대열 앞으로 내놓는 거리[m]")
p.add_argument("--hloop", type=float, default=0.5, help="헤딩 P 고리 이득(배포·평가기와 같은 식, 0 = 끔)")
p.add_argument("--yaw", type=float, default=0.10, help="스폰 요 무작위 반폭[rad] (학습 설정은 ±3.14) — 대열을 맞추려고 좁힘")
p.add_argument("--yaw0", type=float, default=3.14159265, help="스폰 요 중심[rad]")
p.add_argument("--xy", type=float, default=0.12, help="스폰 위치 무작위 반폭[m] (학습 설정은 ±0.3)")
p.add_argument("--stagger", type=float, default=1.82, help="걸음 시작 시각을 이 폭[s] 안에서 흩어 위상을 섞는다(0 = 전원 동시)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
NATIVE = a.cmd == "native"
os.environ["STEP_ASSIST_ALPHA"] = "0.0"
os.environ.setdefault("TURN_ALPHA", "0")
if not NATIVE:
    os.environ["W3_GAIT_ENABLE"] = "1"
    os.environ["W3_CMD"] = a.cmd
    os.environ["W3_HLOOP"] = "0"
app = AppLauncher(a).app

import json
import math
import time

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401
import rl_walking.tasks_extra_1009  # noqa: F401

from isaaclab_tasks.utils import load_cfg_from_registry

W, H = (int(v) for v in a.res.split("x"))
ac = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point")
ec.scene.num_envs = a.envs
ec.scene.env_spacing = a.spacing
ec.seed = a.seed
ec.curriculum = None
ec.episode_length_s = 600.0                      # 녹화 중 시간 초과 일제 리셋 방지
for _ev in ("disturbance", "push_robot"):
    if getattr(ec.events, _ev, None) is not None:
        setattr(ec.events, _ev, None)
if not NATIVE:
    c = ec.commands.base_velocity
    c.resampling_time_range = (1e6, 1e6)
    c.rel_standing_envs = 0.0
if a.yaw >= 0:
    _pr = ec.events.reset_base.params["pose_range"]
    _pr["yaw"] = (a.yaw0 - a.yaw, a.yaw0 + a.yaw); _pr["x"] = (-a.xy, a.xy); _pr["y"] = (-a.xy, a.xy)
ec.viewer.resolution = (W, H)
ec.viewer.origin_type = "world"

env = gym.make(a.task, cfg=ec, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
u = env.unwrapped
robot = u.scene["robot"]
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device)
r.load(a.ckpt)
pol = r.get_inference_policy(device=u.device)
pnn = r.alg.policy
cmd_term = u.command_manager.get_term("base_velocity")
CMD = None if NATIVE else [float(v) for v in a.cmd.split(",")]

org = u.scene.env_origins
xs, ys = org[:, 0], org[:, 1]
# 로봇은 월드 −x 를 본다(스폰 요 ≈ 180°). 카메라는 −x 쪽에서 +x 를 보므로 화면 오른쪽 = −y.
# 주인공 = 맨 앞줄(x 최소)·맨 왼쪽(y 최대) 모서리 로봇.
score = (xs - xs.min()).abs() + (ys - ys.max()).abs()
hero = int(score.argmin())
H0 = np.array([float(xs[hero]), float(ys[hero]), 0.0])
print(f"[swarm] envs {a.envs} hero {hero} origin {H0.tolist()} x[{float(xs.min()):.1f},{float(xs.max()):.1f}] y[{float(ys.min()):.1f},{float(ys.max()):.1f}]", flush=True)
SPANX = float(xs.max() - xs.min())
SPANY = float(ys.max() - ys.min())

# ---- 카메라 경로: (시각 s, 눈, 보는 점) — 주인공 기준 상대좌표(x: 대열 깊이 방향, −y: 화면 오른쪽)
PATHS = {
    "A": [
        (0.0, (-1.55, 0.55, 1.95), (0.05, 0.0, 0.30)),
        (2.6, (-1.75, 0.65, 2.10), (0.05, 0.0, 0.30)),
        (6.6, (-9.0, 4.0, 7.5), (7.0, -4.0, 0.0)),
        (10.6, (-11.0, -0.30 * SPANY, 11.0), (14.0, -0.36 * SPANY, 0.0)),
        (15.0, (-16.0, -0.60 * SPANY, 20.0), (0.40 * SPANX, -0.52 * SPANY, 0.0)),
    ],
    "B": [
        (0.0, (-1.55, 0.55, 1.95), (0.05, 0.0, 0.30)),
        (2.4, (-1.80, 0.70, 2.15), (0.05, 0.0, 0.30)),
        (6.2, (-7.5, 3.5, 6.0), (8.0, -5.0, 0.0)),
        (10.0, (-7.0, -0.42 * SPANY, 6.5), (0.30 * SPANX, -0.50 * SPANY, 0.0)),
        (13.6, (-9.0, -0.92 * SPANY, 13.0), (0.42 * SPANX, -0.52 * SPANY, 0.0)),
        (17.0, (-16.0, -1.06 * SPANY, 30.0), (0.46 * SPANX, -0.50 * SPANY, 0.0)),
    ],
    "C": [
        (0.0, (-1.55, 0.55, 1.85), (0.0, 0.0, 0.33)),
        (2.2, (-1.72, 0.62, 1.97), (0.0, 0.0, 0.33)),
        (6.0, (-7.0, 3.0, 8.5), (a.lead + 1.5, -4.0, 0.0)),
        (9.8, (a.lead + 2.0, -0.42 * SPANY, 7.0), (a.lead + 15.0, -0.42 * SPANY - 7.0, 0.0)),
        (13.2, (-6.0, -0.90 * SPANY, 12.0), (a.lead + 0.30 * SPANX, -0.62 * SPANY, 0.0)),
        (17.0, (-12.0, -1.05 * SPANY, 24.0), (a.lead + 0.40 * SPANX, -0.52 * SPANY, 0.0)),
    ],
}
KEYS = PATHS[a.path]
a_tilt_min = 27.0
HS = H0 + np.array([-a.lead, 0.0, 0.0])            # 주인공 시작점(대열 앞)
KT = np.array([k[0] for k in KEYS])
KE = np.array([k[1] for k in KEYS], dtype=float)
KL = np.array([k[2] for k in KEYS], dtype=float)


def _cr(P, t):
    """시각 불균등 Catmull-Rom(양 끝 접선 0 → 시작·끝에서 멈춘 듯 부드럽게)."""
    t = min(max(t, KT[0]), KT[-1])
    i = int(np.searchsorted(KT, t, side="right") - 1)
    i = min(max(i, 0), len(KT) - 2)
    t0, t1 = KT[i], KT[i + 1]
    s = (t - t0) / (t1 - t0)

    def tan(j):
        if j == 0 or j == len(KT) - 1:
            return np.zeros(3)
        return (P[j + 1] - P[j - 1]) / (KT[j + 1] - KT[j - 1])
    m0, m1 = tan(i) * (t1 - t0), tan(i + 1) * (t1 - t0)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * P[i] + h10 * m0 + h01 * P[i + 1] + h11 * m1


def cam_at(t):
    """주인공이 걸어 나가는 만큼 초반에는 따라간다(가중 1 → 0)."""
    hp = robot.data.root_pos_w[hero]
    live = np.array([float(hp[0]), float(hp[1]), 0.0])
    w = 1.0 - min(max((t - KT[1]) / (KT[2] - KT[1]), 0.0), 1.0)
    w = w * w * (3 - 2 * w)
    base = HS + w * (live - HS)
    e, l = base + _cr(KE, t), base + _cr(KL, t)
    hz = math.hypot(l[0] - e[0], l[1] - e[1])
    zmin = l[2] + hz * math.tan(math.radians(a_tilt_min))            # 위쪽 화면 끝이 지평선 아래에 머물게
    if e[2] < zmin:
        e[2] = zmin
    return e, l


def set_cam(t):
    e, l = cam_at(t)
    u.viewport_camera_controller.update_view_location(eye=tuple(e), lookat=tuple(l))


PSI = {"ref": None}


def yaw_of(q):
    w_, x_, y_, z_ = q.unbind(-1)
    return torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))


def set_cmd():
    if CMD is None:
        return
    cmd_term.vel_command_b[:, 0] = CMD[0]
    cmd_term.vel_command_b[:, 1] = CMD[1]
    cmd_term.vel_command_b[:, 2] = CMD[2]
    if a.hloop > 0 and PSI["ref"] is not None:          # wz = clip(K·wrap(ψ* − ψ), ±0.4), ψ* = 대열 공통 방향
        err = PSI["ref"] - yaw_of(robot.data.root_quat_w)
        err = torch.atan2(torch.sin(err), torch.cos(err))
        cmd_term.vel_command_b[:, 2] = (CMD[2] + a.hloop * err).clamp(-0.4, 0.4)
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False


os.makedirs(a.out, exist_ok=True)
dt = float(u.step_dt)
NREC = int(round(a.dur / dt))
falls = 0
hero_resets = 0
n = 0
PREV_T = [0.0, 2.2, 4.1, 6.0, 7.9, 9.8, 11.5, 13.2, 15.1, 17.0]
PREV_T = [t for t in PREV_T if t <= a.dur + 1e-6]
T_ON = 1.0 + torch.rand(a.envs, device=u.device) * a.stagger      # env 별 걸음 시작 시각[s]
prev_ticks = {int(round(t / dt)): t for t in PREV_T}
with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    _y = yaw_of(robot.data.root_quat_w)
    PSI["ref"] = torch.atan2(torch.sin(_y).mean(), torch.cos(_y).mean()).expand(a.envs).clone()
    fw = torch.stack([torch.cos(_y), torch.sin(_y)], -1).mean(0)
    print(f"[swarm] spawn yaw mean {math.degrees(float(PSI['ref'][0])):.1f} deg, mean yaw dir (x,y) = ({float(fw[0]):+.2f},{float(fw[1]):+.2f})", flush=True)
    for i in range(a.warm):
        if i == 40 and a.lead > 0:                      # 지지 해제 뒤·걸음 시작 전에 주인공만 대열 앞으로
            hid = torch.tensor([hero], device=u.device)
            rs = robot.data.root_state_w[hid].clone()
            rs[:, 0] -= a.lead
            robot.write_root_pose_to_sim(rs[:, :7], env_ids=hid)
            robot.write_root_velocity_to_sim(torch.zeros_like(rs[:, 7:]), env_ids=hid)
        set_cmd()
        if a.stagger > 0 and getattr(u, "_ge_cmd", None) is not None:
            u._ge_cmd[:] = (i * dt) > T_ON
        obs, _, dones, extras = env.step(pol(obs))
        pnn.reset(dones)
    _hp = robot.data.root_pos_w[hero]
    print(f"[swarm] hero pos after warm ({float(_hp[0]):.2f},{float(_hp[1]):.2f},{float(_hp[2]):.2f}) start ({HS[0]:.2f},{HS[1]:.2f})", flush=True)
    print(f"[swarm] warm done, tilt mean {float(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))).mean()):.2f} deg", flush=True)
    set_cam(0.0)
    for _ in range(12):                              # 첫 렌더는 빈 화면 — 버림
        u.sim.render(); u.render()
    T0 = time.time()
    for i in range(NREC + 1):
        t = i * dt
        if a.preview:
            if i in prev_ticks:
                set_cam(t)
                for _ in range(3):
                    u.sim.render(); u.render()
                Image.fromarray(u.render().astype("uint8")).save(os.path.join(a.out, f"pv_{prev_ticks[i]:04.1f}.jpg"), quality=92)
                print(f"[swarm] preview t={t:.1f}", flush=True)
        elif i % a.every == 0:
            set_cam(t)
            img = u.render()
            Image.fromarray(img.astype("uint8")).save(os.path.join(a.out, f"f{n:05d}.jpg"), quality=96, subsampling=0)
            n += 1
            if n % 100 == 0:
                print(f"[swarm] frames {n} t+{time.time() - T0:.0f}s", flush=True)
        set_cmd()
        obs, _, dones, extras = env.step(pol(obs))
        pnn.reset(dones)
        nd = int(dones.sum())
        falls += nd
        hero_resets += int(dones[hero])
json.dump({"task": a.task, "ckpt": a.ckpt, "envs": a.envs, "spacing": a.spacing, "cmd": a.cmd, "seed": a.seed, "dt": dt,
           "frames": n, "every": a.every, "resets_during_rec": falls, "hero_resets": hero_resets, "hero": hero, "path": a.path},
          open(os.path.join(a.out, "stats.json"), "w"))
print(f"[swarm] DONE frames={n} resets={falls} hero_resets={hero_resets}", flush=True)
env.close()
app.close()
