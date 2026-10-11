"""종종걸음 **모션 시연용 애니메이션**(1011) — 강화학습·정책 없음, 물리 균형 없음.
발끝 궤적(들림 1 cm 안팎·짧은 보폭·빠른 박자)을 스크립트로 주고, 수치 야코비안으로 관절각을 풀어 매 프레임 관절·몸통 상태를 직접 써 넣은 뒤 렌더링한다.
이 영상은 "이런 모양으로 걷는다"를 보여 주는 그림이며, 이 궤적으로 실제로 균형을 잡고 걸을 수 있다는 근거가 아니다.
사용: isaaclab.sh -p reports/real_1009/newbody/shuffle_anim.py --out_tag shuffle1 [--lift 0.012 --stride 0.025 --period 0.4 --seconds 8]
"""
import argparse, math, os, subprocess, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3dNB-v0")
p.add_argument("--out_tag", default="shuffle1")
p.add_argument("--lift", type=float, default=0.012, help="발 들림 [m]")
p.add_argument("--stride", type=float, default=0.025, help="한 걸음 보폭 [m]")
p.add_argument("--period", type=float, default=0.4, help="한 주기(두 걸음) [s]")
p.add_argument("--swing_frac", type=float, default=0.35, help="주기 중 한 발이 떠 있는 비율")
p.add_argument("--sway", type=float, default=0.004, help="몸통 좌우 흔들림 [m](시각용)")
p.add_argument("--seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
import rl_walking.tasks_extra_1009  # noqa: F401,E402
from isaaclab.utils import math as M   # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

DATA = "/home/ryu/humanoid_leg_test1"
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 1
cfg.viewer.resolution = (1280, 720)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
robot = u.scene["robot"]; dev = u.device; sim = u.sim; dt = float(sim.get_physics_dt())
env.reset()
names = list(robot.joint_names)
print("[joints]", names, flush=True)
q0 = robot.data.default_joint_pos.clone()
for j, n in enumerate(names):
    if "hip_r" in n:
        q0[:, j] = 0.0


def raw_step(q):
    robot.set_joint_position_target(q); robot.write_data_to_sim(); sim.step(render=False); robot.update(dt)


def put(root_pos, root_quat, q):
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=1))
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    raw_step(q)


# 1) 물리로 잠깐 세워 서 있는 몸통 자세를 얻는다(기본 자세 PD 유지 0.5 s)
for _ in range(int(0.5 / dt)):
    raw_step(q0)
p0 = robot.data.root_pos_w.clone(); r0 = robot.data.root_quat_w.clone()
up = M.quat_apply(r0, torch.tensor([[0.0, 0.0, 1.0]], device=dev))
print(f"[settle] root z {float(p0[0, 2]):.3f} up_z {float(up[0, 2]):.3f}", flush=True)
assert float(up[0, 2]) > 0.95, "기본 자세로 0.5 s 서 있지 못함"
qs = robot.data.joint_pos.clone()                 # 하중 받은 실제 관절각(처짐 포함)
fwd = M.quat_apply(r0, torch.tensor([[1.0, 0.0, 0.0]], device=dev)); fwd[:, 2] = 0; fwd = fwd / fwd.norm()
lat = torch.tensor([[-float(fwd[0, 1]), float(fwd[0, 0]), 0.0]], device=dev)

feet = {s: robot.find_bodies(f"{s}_ankle_r.*")[0][0] for s in ("left", "right")}
print("[feet]", {s: robot.body_names[b] for s, b in feet.items()}, flush=True)
legj = {s: [names.index(f"{s}_{k}_joint") for k in ("hip_f", "knee", "ankle_f")] for s in ("left", "right")}


def foot_rel(side):
    b = feet[side]
    pr = M.quat_apply_inverse(robot.data.root_quat_w, robot.data.body_pos_w[:, b] - robot.data.root_pos_w)[0]
    qr = M.quat_mul(M.quat_conjugate(robot.data.root_quat_w), robot.data.body_quat_w[:, b])
    return pr.clone(), qr.clone()


# 2) 수치 야코비안: (발 앞뒤 x, 발 높이 z, 발 피치) ← (hip_f, knee, ankle_f). 공중에서 잰다(접촉 영향 없음)
air = p0.clone(); air[:, 2] += 0.4
put(air, r0, qs); base = {s: foot_rel(s) for s in feet}
J = {}
d = 0.02
for s in feet:
    cols = []
    for j in legj[s]:
        q = qs.clone(); q[:, j] += d
        put(air, r0, q); pr, qr = foot_rel(s)
        dq = M.quat_mul(qr, M.quat_conjugate(base[s][1])); aa = M.axis_angle_from_quat(dq)[0]
        cols.append(torch.stack([(pr[0] - base[s][0][0]) / d, (pr[2] - base[s][0][2]) / d, aa[1] / d]))
    J[s] = torch.stack(cols, dim=1)
    print(f"[J {s}]\n", J[s].cpu().numpy().round(3), flush=True)
Jinv = {s: torch.linalg.inv(J[s]) for s in feet}

# 3) 걸음 궤적
T = float(args.period); fs = float(args.swing_frac); S = float(args.stride); h = float(args.lift)
v = S / ((1.0 - fs) * T)                         # 지지 구간 동안 발이 몸통 대비 S 만큼 뒤로 간다
print(f"[gait] 주기 {T} s(초당 {2 / T:.1f} 걸음) · 보폭 {S * 100:.1f} cm · 들림 {h * 100:.1f} cm · 속도 {v:.3f} m/s", flush=True)


def foot_target(ph):
    """위상 ph∈[0,1) → 몸통 기준 발 변위 (dx, dz). 0…fs = 스윙, 나머지 = 지지."""
    if ph < fs:
        s_ = ph / fs
        return -S / 2 + S * (0.5 - 0.5 * math.cos(math.pi * s_)), h * math.sin(math.pi * s_) ** 2
    s_ = (ph - fs) / (1.0 - fs)
    return S / 2 - S * s_, 0.0


vdir = os.path.join(DATA, "log_picture", f"video_{args.out_tag}"); os.makedirs(vdir, exist_ok=True)
N = int(args.seconds * 50); ramp = 1.0
for i in range(N):
    t = i / 50.0
    g = min(1.0, max(0.0, (t - 1.0) / 0.6)) * min(1.0, max(0.0, (args.seconds - 1.0 - t) / 0.6))   # 앞뒤 1 s 는 서기
    q = qs.clone(); zs = {}
    for s, off in (("left", 0.0), ("right", 0.5)):
        dx, dz = foot_target((t / T + off) % 1.0)
        zs[s] = dz
        dqj = Jinv[s] @ torch.tensor([g * dx, g * dz, 0.0], device=dev)
        for k, j in enumerate(legj[s]):
            q[0, j] += dqj[k]
    if i == 0:
        x_trav = 0.0
    else:
        x_trav += g * v / 50.0
    sway = g * float(args.sway) * math.sin(2 * math.pi * t / T)          # 지지 발 쪽으로 살짝
    pos = p0 + fwd * x_trav + lat * sway
    put(pos, r0, q)
    eye = pos[0] + lat[0] * 1.7 + fwd[0] * 0.9; look = pos[0].clone(); look[2] -= 0.22
    u.viewport_camera_controller.update_view_location(eye=(float(eye[0]), float(eye[1]), float(p0[0, 2]) + 0.05), lookat=(float(look[0]), float(look[1]), float(look[2])))
    img = u.render()
    if img is None:
        continue
    im = Image.fromarray(img.astype("uint8")); dr = ImageDraw.Draw(im)
    dr.text((12, 10), f"SCRIPTED ANIMATION (no RL, no balance physics)  t={t:4.2f}s", fill=(255, 255, 0))
    dr.text((12, 32), f"{2 / T:.1f} steps/s  stride {S * 100:.1f} cm  lift {h * 100:.1f} cm  speed {v:.2f} m/s   foot z L/R = {zs['left'] * g * 1000:4.1f} / {zs['right'] * g * 1000:4.1f} mm", fill=(255, 255, 0))
    im.save(os.path.join(vdir, f"f{i:05d}.png"))
mp4 = os.path.join(DATA, "log_picture", f"video_{args.out_tag}.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "50", "-i", os.path.join(vdir, "f%05d.png"), "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4])
print("[video]", mp4, flush=True)
sys.stdout.flush()
os._exit(0)
