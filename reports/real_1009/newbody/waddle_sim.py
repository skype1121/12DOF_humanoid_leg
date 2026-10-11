"""WADDLE-W1 대조군(1011): **정책 없는 스크립트 궤적**을 같은 몸 시뮬에 물리로 넣는다(지시 8 항).
궤적 = 몸통 자세를 유지한 채 골반을 좌우로 D 만큼 평행 이동 → 반대 발을 h 만큼 수직으로 들기 → 내리기 → 반대쪽. 제자리(전진 없음), 피드백 없음(개루프).
관절 목표는 공중에서 수치 야코비안 뉴턴 반복으로 푼 발 위치 IK(발 x·y·z·롤·피치 ← hip_f·hip_a·knee·ankle_f·ankle_r).
구동 = 환경 구동기 그대로(kp 150·kd 5, 암시적 PD), 목표 갱신 50 Hz, 지연 없음(초 단위 준정적 동작이라 80 ms 지연은 무시). env.step 을 거치지 않으므로 밀기·명령 이벤트 없음.
env 마다 (플랜트, D, h)가 다르다. 플랜트 1 = 무릎 유효 강성 32 N·m/rad(kp 150 과 직렬 탄성 40 의 합성 — 실물 무릎 처짐 10–12° 모사, 가정).
사용: isaaclab.sh -p -u reports/real_1009/newbody/waddle_sim.py --out_tag w1 [--reps 3]
"""
import argparse, itertools, math, os, sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3dNB-v0")
p.add_argument("--out_tag", default="w1")
p.add_argument("--shifts", default="0.03,0.05,0.07,0.09,0.11")
p.add_argument("--lifts", default="0.01,0.02")
p.add_argument("--plants", default="0,1")
p.add_argument("--leans", default="0", help="지지쪽으로 몸통 롤 기울임 [deg] 목록")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--cycles", type=int, default=2)
p.add_argument("--t_shift", type=float, default=2.5, help="중앙 → 한쪽 이동 시간 [s](한쪽 → 반대쪽은 2 배)")
p.add_argument("--t_lift", type=float, default=0.8)
p.add_argument("--t_hold", type=float, default=0.6)
p.add_argument("--t_dwell", type=float, default=0.5)
p.add_argument("--soft_knee_kp", type=float, default=32.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
import rl_walking.tasks_extra_1009  # noqa: F401,E402
from isaaclab.utils import math as M   # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

DATA = "/home/ryu/humanoid_leg_test1"
combos = [(int(pl), float(D), float(h), float(le)) for pl, D, h, le in itertools.product(args.plants.split(","), args.shifts.split(","), args.lifts.split(","), args.leans.split(","))]
combos = [c for c in combos for _ in range(args.reps)]
N = len(combos)
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = N
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]; cs = u.scene.sensors["contact_forces"]; dev = u.device; sim = u.sim; dt = float(sim.get_physics_dt())
env.reset()
names = list(robot.joint_names)
PL = torch.tensor([c[0] for c in combos], device=dev); Dv = torch.tensor([c[1] for c in combos], device=dev); Hv = torch.tensor([c[2] for c in combos], device=dev); Lv = torch.deg2rad(torch.tensor([c[3] for c in combos], device=dev))
try:
    z = torch.zeros(N, robot.num_bodies, 3, device=dev); robot.set_external_force_and_torque(z, z)
except Exception as e:  # noqa: BLE001
    print("[warn] 외력 초기화 실패:", e, flush=True)
kid = [names.index("left_knee_joint"), names.index("right_knee_joint")]
soft = torch.nonzero(PL == 1).flatten()
if len(soft):
    robot.write_joint_stiffness_to_sim(torch.full((len(soft), 2), float(args.soft_knee_kp), device=dev), joint_ids=kid, env_ids=soft)
q0 = robot.data.default_joint_pos.clone()
hr = [j for j, n in enumerate(names) if "hip_r" in n]
q0[:, hr] = robot.data.joint_pos[:, hr]            # 고정 관절은 환경이 잠근 각 그대로
lim = robot.data.soft_joint_pos_limits.clone()


def step(q, n=1):
    for _ in range(n):
        robot.set_joint_position_target(q); u.scene.write_data_to_sim(); sim.step(render=False); u.scene.update(dt)


def put(root_pos, root_quat, q):
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=1))
    robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    step(q)


SIDES = ("left", "right")
fb = {s: robot.find_bodies(f"{s}_ankle_r.*")[0][0] for s in SIDES}
fc = {s: cs.find_bodies(f"{s}_ankle_r.*")[0][0] for s in SIDES}
legj = {s: [names.index(f"{s}_{k}_joint") for k in ("hip_f", "hip_a", "knee", "ankle_f", "ankle_r")] for s in SIDES}
print("[feet]", {s: robot.body_names[b] for s, b in fb.items()}, "N", N, flush=True)

# 1) 물리로 세움
step(q0, int(1.0 / dt))
p0 = robot.data.root_pos_w.clone(); r0 = robot.data.root_quat_w.clone()
mg = float(robot.data.default_mass[0].sum()) * 9.81


def fz():
    return torch.stack([cs.data.net_forces_w[:, fc[s], 2] for s in SIDES], 1)


print(f"[settle] root z {p0[:, 2].mean():.3f} · 발 수직력 합/체중 {float((fz().sum(1) / mg).mean()):.2f} (1 에 가까워야 지지 외력 없음) · mg {mg:.1f} N", flush=True)


def foot_rel(s):
    b = fb[s]
    pr = M.quat_apply_inverse(robot.data.root_quat_w, robot.data.body_pos_w[:, b] - robot.data.root_pos_w)
    qr = M.quat_mul(M.quat_conjugate(robot.data.root_quat_w), robot.data.body_quat_w[:, b])
    return pr.clone(), qr.clone()


air = p0.clone(); air[:, 2] += 0.4
put(air, r0, q0); base = {s: foot_rel(s) for s in SIDES}
print("[geom] 발 중심 y(몸통 기준, env0) L/R", float(base["left"][0][0, 1]), float(base["right"][0][0, 1]), "· x", float(base["left"][0][0, 0]), "· z", float(base["left"][0][0, 2]), flush=True)


def err(s, tgt):
    pr, qr = foot_rel(s)
    aa = M.axis_angle_from_quat(M.quat_mul(qr, M.quat_conjugate(base[s][1])))
    ph = tgt["phi"]; b = base[s][0] + tgt[s]; c, sn = torch.cos(ph), torch.sin(ph)
    want = torch.stack([b[:, 0], b[:, 1] * c - b[:, 2] * sn, b[:, 1] * sn + b[:, 2] * c], 1)     # R_x(phi): 몸통이 발에 대해 −phi 만큼 롤
    return torch.cat([pr - want, aa[:, :1] - ph[:, None], aa[:, 1:2]], dim=1)          # (N, 5)


def solve(tgt, q_init):
    """tgt[s] = (N,3) 발 위치 변위(몸통 기준). 반환 q, 잔차 노름, 한계 걸림 여부."""
    q = q_init.clone(); d = 0.01
    for _ in range(8):
        put(air, r0, q); e0 = {s: err(s, tgt) for s in SIDES}
        J = {s: torch.zeros(N, 5, 5, device=dev) for s in SIDES}
        for k in range(5):
            qq = q.clone()
            for s in SIDES:
                qq[:, legj[s][k]] += d
            put(air, r0, qq)
            for s in SIDES:
                J[s][:, :, k] = (err(s, tgt) - e0[s]) / d
        for s in SIDES:
            dq = torch.linalg.solve(J[s], -e0[s].unsqueeze(-1)).squeeze(-1).clamp(-0.3, 0.3)
            q[:, legj[s]] += dq
        q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
    put(air, r0, q)
    res = torch.stack([err(s, tgt).norm(dim=1) for s in SIDES], 1).amax(1)
    hit = ((q <= lim[..., 0] + 1e-4) | (q >= lim[..., 1] - 1e-4))
    hit[:, hr] = False
    return q, res, hit.any(1)


def T(dy, dzl=None, dzr=None, phi=None):
    zl = torch.zeros(N, device=dev) if dzl is None else dzl; zr = torch.zeros(N, device=dev) if dzr is None else dzr
    o = torch.zeros(N, device=dev)
    return {"left": torch.stack([o, dy, zl], 1), "right": torch.stack([o, dy, zr], 1), "phi": o if phi is None else phi}


# 골반을 왼쪽으로 D → 발은 몸통 기준 −D(몸통 +y = 왼쪽)
K = {}; RES = {}; HIT = {}
K["KL"], RES["KL"], HIT["KL"] = solve(T(-Dv, phi=-Lv), q0)
K["KLr"], RES["KLr"], HIT["KLr"] = solve(T(-Dv, dzr=Hv, phi=-Lv), K["KL"])
K["KR"], RES["KR"], HIT["KR"] = solve(T(Dv, phi=Lv), q0)
K["KRl"], RES["KRl"], HIT["KRl"] = solve(T(Dv, dzl=Hv, phi=Lv), K["KR"])
K["K0"] = q0
ikres = torch.stack([RES[k] for k in RES], 1).amax(1); ikhit = torch.stack([HIT[k] for k in HIT], 1).any(1)
for s in SIDES:
    dqd = torch.rad2deg(K["KL"][:, legj[s]] - q0[:, legj[s]])
    i = int(torch.argmax(Dv))
    print(f"[ik] {s} KL−K0 (D={float(Dv[i]):.2f}) hip_f,hip_a,knee,ankle_f,ankle_r [deg] {dqd[i].cpu().numpy().round(1)}", flush=True)
print(f"[ik] 잔차 최대 {float(ikres.max()) * 1000:.2f} mm·mrad · 한계 걸림 env {int(ikhit.sum())}/{N}", flush=True)

# 2) 다시 세우고 궤적 재생
put(p0, r0, robot.data.joint_pos.clone() * 0 + q0)
step(q0, int(1.0 / dt))
pz0 = robot.data.root_pos_w[:, 2].clone(); fz0 = {s: robot.data.body_pos_w[:, fb[s], 2].clone() for s in SIDES}
fxy0 = {s: robot.data.body_pos_w[:, fb[s], :2].clone() for s in SIDES}
seq = [(args.t_shift, "KL")]
for _ in range(args.cycles):
    seq += [(args.t_dwell, "KL"), (args.t_lift, "KLr"), (args.t_hold, "KLr"), (args.t_lift, "KL"), (args.t_dwell, "KL"), (2 * args.t_shift, "KR"),
            (args.t_dwell, "KR"), (args.t_lift, "KRl"), (args.t_hold, "KRl"), (args.t_lift, "KR"), (args.t_dwell, "KR"), (2 * args.t_shift, "KL")]
seq += [(args.t_shift, "K0"), (1.0, "K0")]
sub = int(round(0.02 / dt))
fell = torch.zeros(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float("nan"), device=dev)
tilt_max = torch.zeros(N, 2, device=dev); tq_max = torch.zeros(N, len(names), device=dev)
lift_ok = []; lift_rise = []; lift_fz = []; slip_max = torch.zeros(N, device=dev)
trace = []
prev = "K0"; t = 0.0
for dur, key in seq:
    n = max(1, int(round(dur / 0.02))); hold = (key == prev) and key in ("KLr", "KRl")
    sw = "right" if key == "KLr" else "left"; st_ = "left" if sw == "right" else "right"
    h_fz = torch.zeros(N, device=dev); h_rise = torch.full((N,), 9.0, device=dev)
    for i in range(n):
        a = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
        step(K[prev] + a * (K[key] - K[prev]), sub); t += 0.02
        g = robot.data.projected_gravity_b
        roll = torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2])); pitch = torch.rad2deg(torch.atan2(-g[:, 0], -g[:, 2]))
        new = (~fell) & ((g[:, 2] > -0.85) | (robot.data.root_pos_w[:, 2] < 0.75 * pz0))
        t_fall = torch.where(new, torch.full_like(t_fall, t), t_fall); fell |= new
        live = (~fell).float()
        tilt_max = torch.maximum(tilt_max, torch.stack([roll.abs(), pitch.abs()], 1) * live.unsqueeze(1))
        tq_max = torch.maximum(tq_max, robot.data.applied_torque.abs() * live.unsqueeze(1))
        F = fz(); rise = torch.stack([robot.data.body_pos_w[:, fb[s], 2] - fz0[s] for s in SIDES], 1)
        if hold:
            k = SIDES.index(sw)
            h_fz = torch.maximum(h_fz, F[:, k]); h_rise = torch.minimum(h_rise, rise[:, k])
            ks = SIDES.index(st_)
            slip_max = torch.maximum(slip_max, (robot.data.body_pos_w[:, fb[st_], :2] - fxy0[st_]).norm(dim=1) * live)
        trace.append(torch.cat([roll[:, None], pitch[:, None], F, rise, robot.data.applied_torque[:, [names.index("left_hip_a_joint"), names.index("right_hip_a_joint")]]], 1).cpu().numpy())
    if hold:
        lift_ok.append(((h_fz < 5.0) & (h_rise > 0.005) & ~fell).cpu()); lift_rise.append(h_rise.cpu()); lift_fz.append(h_fz.cpu())
    prev = key
lift_ok = torch.stack(lift_ok, 1); lift_rise = torch.stack(lift_rise, 1); lift_fz = torch.stack(lift_fz, 1)
tq = tq_max.cpu(); jn = lambda k: [j for j, n in enumerate(names) if k in n]  # noqa: E731
out = os.path.join(DATA, "log_picture", f"waddle_{args.out_tag}.txt")
L = []
L.append(f"WADDLE 스크립트 대조군 · 태스크 {args.task} · env {N} · 주기 {args.cycles} · 이동 {args.t_shift} s·들기 {args.t_lift} s·유지 {args.t_hold} s · 들기 시도 {lift_ok.shape[1]} 회/env")
L.append(f"발 중심 간격(몸통 기준) {float(base['left'][0][0, 1] - base['right'][0][0, 1]):.3f} m · IK 잔차 최대 {float(ikres.max()) * 1000:.2f} · 발 수직력 합/체중(세움 뒤) {float((fz().sum(1) / mg).mean()):.2f}")
L.append("플랜트\tD[cm]\th[cm]\tn\t낙상\t들기성공/시도\t든높이최소[mm]\t든발력최대[N]\t롤max\t피치max\thip_a토크max\t무릎max\t발목롤max\t지지발이동max[mm]\tIK한계")
for pl, D, h, le in sorted(set(combos)):
    m = ((PL == pl) & (Dv == D) & (Hv == h) & ((Lv - math.radians(le)).abs() < 1e-5)).cpu(); mi = torch.nonzero(m).flatten()
    L.append(f"{'무릎연성' if pl else '명목'}{le:.0f}도\t{D * 100:.0f}\t{h * 100:.0f}\t{len(mi)}\t{int(fell.cpu()[mi].sum())}\t{int(lift_ok[mi].sum())}/{lift_ok[mi].numel()}\t"
             f"{float(lift_rise[mi].min()) * 1000:.1f}\t{float(lift_fz[mi].max()):.0f}\t{float(tilt_max.cpu()[mi, 0].max()):.1f}\t{float(tilt_max.cpu()[mi, 1].max()):.1f}\t"
             f"{float(tq[mi][:, jn('hip_a')].max()):.1f}\t{float(tq[mi][:, jn('knee')].max()):.1f}\t{float(tq[mi][:, jn('ankle_r')].max()):.1f}\t{float(slip_max.cpu()[mi].max()) * 1000:.1f}\t{int(ikhit.cpu()[mi].sum())}")
open(out, "w").write("\n".join(L) + "\n")
np.savez_compressed(os.path.join(DATA, "log_picture", f"waddle_{args.out_tag}.npz"), trace=np.stack(trace), combos=np.array(combos), fell=fell.cpu().numpy(), t_fall=t_fall.cpu().numpy(),
                    kd=np.stack([(K[k] - q0).cpu().numpy() for k in ("KL", "KLr", "KR", "KRl")]), lift_ok=lift_ok.numpy(), lift_rise=lift_rise.numpy(), lift_fz=lift_fz.numpy(), tq_max=tq.numpy(), names=np.array(names))
print("\n".join(L), flush=True)
print("[out]", out, flush=True)
sys.stdout.flush()
os._exit(0)
