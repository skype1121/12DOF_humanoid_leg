"""발표용: 같은 학습 실행(G6)의 체크포인트 여러 개를 같은 학습 환경에서 재생해 프레임 저장.

저장소 밖(scratchpad) 임시 스크립트 — 저장소 코드·설정은 건드리지 않는다.
실행: isaaclab.sh -p -u rec_stages.py --headless --ckpts 0,100,4500 --out <dir>
출력: <out>/it<N>/f%05d.png + <out>/it<N>/stats.json  (판정은 파일로)
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: F401  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-G6-v0")
p.add_argument("--run", default="/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4/2026-09-20_15-06-59_g6")
p.add_argument("--ckpts", default="0,100,4500")
p.add_argument("--out", required=True)
p.add_argument("--envs", type=int, default=16)
p.add_argument("--spacing", type=float, default=1.6)
p.add_argument("--steps", type=int, default=800)
p.add_argument("--every", type=int, default=2)
p.add_argument("--res", default="1000x800")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--no_render", action="store_true")
p.add_argument("--win", default="0,100000", help="저장할 틱 구간 lo,hi")
p.add_argument("--cam_test", action="store_true", help="카메라 후보만 몇 장 찍고 종료")
p.add_argument("--eye", default="3.4,3.0,1.5")
p.add_argument("--look", default="-0.6,-0.6,0.35")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = not a.no_render
app = AppLauncher(a).app

import json
import os

import gymnasium as gym
import torch
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

from isaaclab_tasks.utils import load_cfg_from_registry

W, H = (int(v) for v in a.res.split("x"))
ac = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point")
ec.scene.num_envs = a.envs
ec.scene.env_spacing = a.spacing
ec.seed = a.seed
ec.viewer.resolution = (W, H)
ec.viewer.origin_type = "world"

env = gym.make(a.task, cfg=ec, render_mode=(None if a.no_render else "rgb_array"))
env = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
u = env.unwrapped
robot = u.scene["robot"]
r = OnPolicyRunner(env, ac.to_dict(), log_dir=None, device=ac.device)
org = u.scene.env_origins
cx, cy = float(org[:, 0].mean()), float(org[:, 1].mean())
print("[rec] env origins x", sorted(set(round(float(v), 2) for v in org[:, 0])), "y", sorted(set(round(float(v), 2) for v in org[:, 1])), flush=True)


def set_cam(eye, look):
    u.viewport_camera_controller.update_view_location(
        eye=(cx + eye[0], cy + eye[1], eye[2]), lookat=(cx + look[0], cy + look[1], look[2]))


def grab(path):
    img = u.render()
    if img is None:
        return False
    Image.fromarray(img.astype("uint8")).save(path)
    return True


os.makedirs(a.out, exist_ok=True)
EYE = tuple(float(v) for v in a.eye.split(","))
LOOK = tuple(float(v) for v in a.look.split(","))

if a.cam_test:
    cands = {
        "F": ((3.2, 2.5, 2.45), (-0.15, -0.15, 0.10)),
        "G": ((3.4, 2.7, 2.6), (-0.15, -0.15, 0.10)),
        "H": ((3.0, 2.4, 2.1), (-0.15, -0.15, 0.15)),
        "I": ((3.6, 2.2, 2.45), (-0.15, -0.10, 0.10)),
    }
    r.load(os.path.join(a.run, "model_4500.pt"))
    pol = r.get_inference_policy(device=u.device)
    with torch.inference_mode():
        env.reset()
        obs = env.get_observations()
        for i in range(60):
            obs, _, dones, _ = env.step(pol(obs))
        set_cam(*cands["F"])
        for _ in range(12):
            u.sim.render(); u.render()
        for k, (e, l) in cands.items():
            set_cam(e, l)
            for _ in range(4):
                u.sim.render(); u.render()
            print("[cam]", k, grab(os.path.join(a.out, f"cam_{k}.png")), flush=True)
    env.close()
    app.close()
    sys.exit(0)

WLO, WHI = (int(v) for v in a.win.split(","))
for spec in a.ckpts.split(","):
    # "라벨=경로" 또는 숫자(기본 run 의 model_N.pt)
    if "=" in spec:
        ck, path = spec.split("=", 1)
    else:
        ck, path = spec, os.path.join(a.run, f"model_{spec}.pt")
    r.load(path)
    pol = r.get_inference_policy(device=u.device)
    pnn = r.alg.policy
    od = os.path.join(a.out, f"it{ck}")
    os.makedirs(od, exist_ok=True)
    torch.manual_seed(a.seed)
    falls = 0
    timeouts = 0
    n = 0
    tilt_log = []
    with torch.inference_mode():
        env.reset()
        u.reset()                       # 체크포인트마다 같은 시작(전 env 리셋)
        obs = env.get_observations()
        if not a.no_render:
            set_cam(EYE, LOOK)
            for _ in range(12):             # 첫 렌더는 빈 화면 — 버림
                u.sim.render(); u.render()
        for i in range(a.steps):
            obs, _, dones, extras = env.step(pol(obs))
            pnn.reset(dones)
            to = extras.get("time_outs")
            nd = int(dones.sum())
            nt = int(to.sum()) if to is not None else 0
            falls += nd - nt
            timeouts += nt
            g = robot.data.projected_gravity_b
            tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
            tilt_log.append([round(float(tilt.mean()), 2), round(float(tilt.max()), 2), nd - nt])
            if (not a.no_render) and WLO <= i < WHI and i % a.every == 0:
                if grab(os.path.join(od, f"f{n:05d}.png")):
                    n += 1
    json.dump({"ckpt": ck, "path": path, "task": a.task, "envs": a.envs, "steps": a.steps, "dt": float(u.step_dt),
               "frames": n, "falls": falls, "timeouts": timeouts, "seed": a.seed, "tilt": tilt_log},
              open(os.path.join(od, "stats.json"), "w"))
    print(f"[rec] it{ck}: frames={n} falls={falls} timeouts={timeouts}", flush=True)

env.close()
print("[rec] DONE", flush=True)
app.close()
