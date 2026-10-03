"""1003 진단(외부 조언 5): 이륙 직전 150 ms 의 체중 이동 — '체중 이동 부족' vs '스윙 생성 문제' 분리.
시뮬 실제 접촉력(패드 센서 아님)으로 이륙(접촉 상실 ≥ 60 ms)마다 직전 150 ms 평균을 기록:
  지지발 Fz/mg · 드는 발 Fz/mg · 골반 횡이동 비 s = (골반 y − 두 발 중점 y)/(지지발 y − 중점 y) (0 = 가운데, 1 = 지지발 위)
  · 질량 가중 몸 중심 횡이동 비(같은 정의, 링크 원점 근사) · 지지 다리 힙롤(hip_a)·발목롤(ankle_r) [deg].
이륙 결과 분류: 유효 스텝(재착지 & 수평 이동 ≥ 2 cm) / 짧은 들림(그 외).
조건(env 균등): 앞 F_f N 램프(상승 0.5·유지 1.0 s) · 뒤 F_b N 충격(0.3 s) — 평가 sf/b18 과 같은 모양, 밀기 t0 = 3 s."""
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-G22R-v0"); p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--F_f", type=float, default=12.0); p.add_argument("--F_b", type=float, default=18.0); p.add_argument("--t0", type=float, default=3.0)
p.add_argument("--out", default="diag_pretakeoff_out.txt")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import os, numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.curriculum = None; ec.events.push_robot = None
ec.actions.joint_pos.lpf_beta_range = (0.27, 0.27); ec.actions.joint_pos.min_delay_steps = ec.actions.joint_pos.max_delay_steps = 1
ec.episode_length_s = a.t0 + 6.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
N = U.num_envs; back = (torch.arange(N, device=dev) % 2) == 1
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
hs = mdp_hang._state(U); robot = U.scene["robot"]
q = robot.data.root_quat_w; ex = torch.zeros(N, 3, device=dev); ex[:, 0] = 1.0
fw = mdp_hang._quat_apply(q, ex)[:, :2]; fw = fw / fw.norm(dim=-1, keepdim=True)
hs["dst_kind"][:] = 0.0; hs["dst_F"][:] = 0.0
hs["dst_kind"][:, 0] = torch.where(back, 2.0, 1.0); hs["dst_t0"][:, 0] = a.t0
hs["dst_rise"][:, 0] = 0.5; hs["dst_hold"][:, 0] = torch.where(back, 0.3, 1.0); hs["dst_F"][:, 0] = torch.where(back, a.F_b, a.F_f)
hs["dst_dir"][:, 0] = fw * torch.where(back, -1.0, 1.0).unsqueeze(1)
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
mg = float(mdp_hang.G) * robot.data.default_mass.sum(1).to(dev)
mass = robot.data.default_mass.to(dev)
jn = list(robot.data.joint_names); ja = [jn.index("left_hip_a_joint"), jn.index("right_hip_a_joint")]; jr = [jn.index("left_ankle_r_joint"), jn.index("right_ankle_r_joint")]
H = 8                                    # 150 ms ≈ 7.5 틱
buf = {k: [] for k in ("fz", "sp", "sc", "ha", "ar")}
prev_c = torch.ones(N, 2, device=dev); air = torch.zeros(N, 2, device=dev); pend = torch.zeros(N, 2, dtype=torch.bool, device=dev)
lift_xy = torch.zeros(N, 2, 2, device=dev); snap = {}
events = []                              # (cond, outcome, stance_fz, lift_fz, s_pelvis, s_com, hip_a, ankle_r)
alive = torch.ones(N, dtype=torch.bool, device=dev)
steps = int((a.t0 + 5.5) / U.step_dt)
for i in range(steps):
    t = i * U.step_dt
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    alive &= ~d.bool()
    F = U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :]; fz = F[..., 2] / mg.unsqueeze(1); c = (F.norm(dim=-1) > 5.0).float()
    yaw = torch.atan2(2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3] + robot.data.root_quat_w[:, 1] * robot.data.root_quat_w[:, 2]),
                      1 - 2 * (robot.data.root_quat_w[:, 2] ** 2 + robot.data.root_quat_w[:, 3] ** 2))
    ly = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    fy = (robot.data.body_pos_w[:, bid, :2] * ly.unsqueeze(1)).sum(-1)          # (N,2) 발 횡좌표
    mid = fy.mean(1); py = (robot.data.root_pos_w[:, :2] * ly).sum(-1)
    com = (robot.data.body_pos_w * mass.unsqueeze(-1)).sum(1) / mass.sum(1, keepdim=True); cy = (com[:, :2] * ly).sum(-1)
    half = (fy - mid.unsqueeze(1))                                               # (N,2) 각 발까지 횡거리(부호)
    sp = (py - mid).unsqueeze(1) / half.where(half.abs() > 1e-3, torch.full_like(half, 1e-3))   # (N,2): 발 k 를 지지발로 볼 때 비
    sc = (cy - mid).unsqueeze(1) / half.where(half.abs() > 1e-3, torch.full_like(half, 1e-3))
    jp = torch.rad2deg(robot.data.joint_pos)
    for k, val in (("fz", fz), ("sp", sp), ("sc", sc), ("ha", jp[:, ja]), ("ar", jp[:, jr])):
        buf[k].append(val.clone()); buf[k] = buf[k][-H:]
    if t >= a.t0:
        lifted = (prev_c > 0.5) & (c < 0.5) & alive.unsqueeze(1)
        for e, f in zip(*torch.where(lifted)):
            e, f = int(e), int(f); s = 1 - f                                     # 지지발 = 반대 발
            m = lambda k, j: float(torch.stack([b[e, j] for b in buf[k]]).mean())
            snap[(e, f)] = dict(cond="뒤" if bool(back[e]) else "앞", stance_fz=m("fz", s), lift_fz=m("fz", f), s_p=m("sp", s), s_c=m("sc", s),
                                ha=m("ha", s), ar=m("ar", s), t_air=0.0)
            lift_xy[e, f] = robot.data.body_pos_w[e, bid[f], :2]
        air = torch.where(c < 0.5, air + U.step_dt, torch.zeros_like(air))
        landed = (prev_c < 0.5) & (c > 0.5)
        for e, f in zip(*torch.where(landed)):
            e, f = int(e), int(f); sn = snap.pop((e, f), None)
            if sn is None: continue
            disp = float((robot.data.body_pos_w[e, bid[f], :2] - lift_xy[e, f]).norm())
            ok = disp >= 0.02
            events.append((sn["cond"], "유효 스텝" if ok else "짧은 들림", sn["stance_fz"], sn["lift_fz"], sn["s_p"], sn["s_c"], sn["ha"], sn["ar"]))
    prev_c = c
for (e, f), sn in snap.items():
    events.append((sn["cond"], "미착지", sn["stance_fz"], sn["lift_fz"], sn["s_p"], sn["s_c"], sn["ha"], sn["ar"]))
L = [f"이륙 직전 150 ms (시뮬 실제 접촉력) · ckpt {ck} · 앞 {a.F_f} N 램프 / 뒤 {a.F_b} N 충격 · env {N}",
     "| 조건 | 결과 | n | 지지발 Fz/mg | 드는 발 Fz/mg | 골반 횡이동 비 | 몸 중심 횡이동 비 | 지지 힙롤 ° | 지지 발목롤 ° |", "|---|---|---|---|---|---|---|---|---|"]
for cond in ("앞", "뒤"):
    for oc in ("유효 스텝", "짧은 들림", "미착지"):
        E = [x for x in events if x[0] == cond and x[1] == oc]
        if not E: L.append(f"| {cond} | {oc} | 0 |" + " |" * 6); continue
        A = np.array([x[2:] for x in E], dtype=float); md = np.median(A, 0); q1 = np.percentile(A, 25, 0); q3 = np.percentile(A, 75, 0)
        L.append(f"| {cond} | {oc} | {len(E)} | " + " | ".join(f"{md[j]:.2f} [{q1[j]:.2f}, {q3[j]:.2f}]" for j in range(6)) + " |")
L.append("값 = 중앙 [사분위]. 횡이동 비 0 = 두 발 가운데, 1 = 지지발 위. 해석 기준(조언): 지지발 Fz ≈ 0.5 mg 면 체중 이동 부족, 0.7–0.8 mg 인데 못 나가면 스윙 생성 문제.")
open(f"/home/ryu/humanoid_leg_test1/reports/gen_g22r/{a.out}", "w").write("\n".join(L) + "\n"); os._exit(0)
