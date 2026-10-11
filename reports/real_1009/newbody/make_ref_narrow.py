"""좁은 종종걸음 참조 동작 만들기(1011): 왼발 스윙 반주기 23 표본의 관절각을 공중 IK 로 푼다(오른발 스윙은 학습 쪽에서 거울).
발 중심을 각각 안쪽으로 d_in, 골반을 지지발 쪽으로 sway·sin(πv), 스윙발을 창(0.25–0.75) 안에서 h·sin² 로 든다. 몸통 자세 유지, 전진 없음.
출력 형식 = reports/gen_step/ref_step_primitive_flex.npz 와 같은 키(q_ref (23,12) 절대각, 로봇 관절 순서). 순수 기구학 — 균형을 보장하지 않는다(학습의 모방 보상 목표로만 씀).
사용: isaaclab.sh -p -u reports/real_1009/newbody/make_ref_narrow.py --d_in 0.03 --sway 0.015 --lift 0.02 --out reports/real_1009/newbody/ref_narrow_a.npz"""
import argparse, math, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3dNB-v0")
p.add_argument("--d_in", type=float, default=0.03); p.add_argument("--sway", type=float, default=0.015); p.add_argument("--lift", type=float, default=0.02)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p); args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import gymnasium as gym, numpy as np, torch          # noqa: E402
import isaaclab_tasks, rl_walking, rl_walking.tasks_extra_1009  # noqa: F401,E402
from isaaclab.utils import math as M   # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
N = 23
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point"); cfg.scene.num_envs = N
env = gym.make(args.task, cfg=cfg); u = env.unwrapped
robot = u.scene["robot"]; dev = u.device; sim = u.sim; dt = float(sim.get_physics_dt()); env.reset()
names = list(robot.joint_names); hr = [j for j, n in enumerate(names) if "hip_r" in n]
q0 = robot.data.default_joint_pos.clone(); q0[:, hr] = robot.data.joint_pos[:, hr]; lim = robot.data.soft_joint_pos_limits.clone()


def put(pos, quat, q):
    robot.write_root_pose_to_sim(torch.cat([pos, quat], 1)); robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q); u.scene.write_data_to_sim(); sim.step(render=False); u.scene.update(dt)


SIDES = ("left", "right"); fb = {s: robot.find_bodies(f"{s}_ankle_r.*")[0][0] for s in SIDES}
legj = {s: [names.index(f"{s}_{k}_joint") for k in ("hip_f", "hip_a", "knee", "ankle_f", "ankle_r")] for s in SIDES}
p0 = robot.data.root_pos_w.clone(); p0[:, 2] += 0.4; r0 = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)


def foot_rel(s):
    pr = M.quat_apply_inverse(robot.data.root_quat_w, robot.data.body_pos_w[:, fb[s]] - robot.data.root_pos_w)
    return pr.clone(), M.quat_mul(M.quat_conjugate(robot.data.root_quat_w), robot.data.body_quat_w[:, fb[s]]).clone()


put(p0, r0, q0); base = {s: foot_rel(s) for s in SIDES}
v = torch.linspace(0.0, 1.0, N, device=dev)
sway = args.sway * torch.sin(math.pi * v)                       # 왼발 스윙 → 골반은 오른쪽(−y) → 발은 몸통 기준 +y
vv = ((v - 0.25) / 0.5).clamp(0, 1); dz = args.lift * torch.sin(math.pi * vv) ** 2
o = torch.zeros(N, device=dev)
tgt = {"left": torch.stack([o, -args.d_in + sway, dz], 1), "right": torch.stack([o, args.d_in + sway, o], 1)}


def err(s):
    pr, qr = foot_rel(s); aa = M.axis_angle_from_quat(M.quat_mul(qr, M.quat_conjugate(base[s][1])))
    return torch.cat([pr - (base[s][0] + tgt[s]), aa[:, :2]], 1)


q = q0.clone(); d = 0.01
for _ in range(8):
    put(p0, r0, q); e0 = {s: err(s) for s in SIDES}; J = {s: torch.zeros(N, 5, 5, device=dev) for s in SIDES}
    for k in range(5):
        qq = q.clone()
        for s in SIDES:
            qq[:, legj[s][k]] += d
        put(p0, r0, qq)
        for s in SIDES:
            J[s][:, :, k] = (err(s) - e0[s]) / d
    for s in SIDES:
        q[:, legj[s]] += torch.linalg.solve(J[s], -e0[s].unsqueeze(-1)).squeeze(-1).clamp(-0.3, 0.3)
    q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
put(p0, r0, q); res = max(float(err(s).norm(dim=1).max()) for s in SIDES)
q[:, hr] = 0.0
z = dict(np.load("/home/ryu/humanoid_leg_test1/reports/gen_step/ref_step_primitive_flex.npz"))
qn = q.cpu().numpy().astype(np.float32); z["q_ref"] = qn
np.savez(args.out, **z)
print(f"[ref] 잔차 최대 {res * 1000:.3f} · 발 중심 간격 {float(base['left'][0][0, 1] - base['right'][0][0, 1]) - 2 * args.d_in:.3f} m · 관절 {names}", flush=True)
for k in (0, 6, 11, 16, 22):
    print(f"[ref] v={k / 22:.2f} deg", np.rad2deg(qn[k]).round(1), flush=True)
print("[out]", args.out, flush=True); sys.stdout.flush(); os._exit(0)
