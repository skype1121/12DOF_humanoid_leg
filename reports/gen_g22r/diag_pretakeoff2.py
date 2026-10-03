"""1003 진단 v2(외부 조언 2차): 이륙 직전 150 ms — 체중 이동 부족 / 스윙 생성 부족 / 서기 국소해 분리. 시뮬 실제 접촉력 기준(패드 아님).
그룹: A 뒤 F_b N 충격 — 유효 스텝(양성 대조) · A' 뒤 짧은 들림 · B 앞 F_f N 램프 — 짧은 들림 · B' 앞 유효 스텝 · C 앞 — 이륙 없이 낙상.
기준 시각: 이륙(접촉 상실 ≥ 60 ms 로 확인된 이륙의 시작 틱). C 는 몸통 기울기가 처음 6.5°(회복 게이트 문턱)를 넘은 틱, 지지발 = 더 실린 발.
지표(그룹별 중앙 [사분위], n):
  지지발 Fz/mg 평균·최대(−150 ms→0) · 드는 발 Fz/mg 감소율((−150 ms 값 − 0 값)/−150 ms 값) · s_pelvis·s_com 평균(−150→0)
  (s = (y − y_mid)/d, d = |y_L − y_R|/2, 지지발 쪽 + — 0 가운데, +1 지지발 중심) · 지지 힙롤·발목롤 변화(0 − (−150 ms)) [deg]
  · 이륙 순간 몸통 롤·피치 [deg]·|ω_xy| · 이륙 뒤 150–300 ms 스윙발 수평 변위 [cm] · 공중 시간 [s] · 최종 스텝 변위 [cm]."""
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-G22R-v0"); p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--F_f", type=float, default=15.0); p.add_argument("--F_b", type=float, default=18.0); p.add_argument("--t0", type=float, default=3.0)
p.add_argument("--out", default="diag_pretakeoff2_out.txt")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import os, math, numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.curriculum = None; ec.events.push_robot = None
ec.actions.joint_pos.lpf_beta_range = (0.27, 0.27); ec.actions.joint_pos.min_delay_steps = ec.actions.joint_pos.max_delay_steps = 1
ec.episode_length_s = a.t0 + 6.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device; dt = U.step_dt
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
N = U.num_envs; back = (torch.arange(N, device=dev) % 2) == 1
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
hs = mdp_hang._state(U); robot = U.scene["robot"]
q0 = robot.data.root_quat_w; ex = torch.zeros(N, 3, device=dev); ex[:, 0] = 1.0
fw = mdp_hang._quat_apply(q0, ex)[:, :2]; fw = fw / fw.norm(dim=-1, keepdim=True)
hs["dst_kind"][:] = 0.0; hs["dst_F"][:] = 0.0
hs["dst_kind"][:, 0] = torch.where(back, 2.0, 1.0); hs["dst_t0"][:, 0] = a.t0
hs["dst_rise"][:, 0] = 0.5; hs["dst_hold"][:, 0] = torch.where(back, 0.3, 1.0); hs["dst_F"][:, 0] = torch.where(back, a.F_b, a.F_f)
hs["dst_dir"][:, 0] = fw * torch.where(back, -1.0, 1.0).unsqueeze(1)
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
mg = float(mdp_hang.G) * robot.data.default_mass.sum(1).to(dev); mass = robot.data.default_mass.to(dev)
jn = list(robot.data.joint_names); ja = [jn.index("left_hip_a_joint"), jn.index("right_hip_a_joint")]; jr = [jn.index("left_ankle_r_joint"), jn.index("right_ankle_r_joint")]
H = 8
steps = int((a.t0 + 5.5) / dt)
# 전 시계열 기록(CPU) — 이벤트 뒤 분석
rec = {k: [] for k in ("fz", "sp", "sc", "ha", "ar", "roll", "pitch", "w", "fxy", "c", "alive")}
alive = torch.ones(N, dtype=torch.bool, device=dev)
for i in range(steps):
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    alive &= ~d.bool()
    F = U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :]
    qw = robot.data.root_quat_w
    yaw = torch.atan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))
    ly = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    fy = (robot.data.body_pos_w[:, bid, :2] * ly.unsqueeze(1)).sum(-1); mid = fy.mean(1); dd = (fy[:, 0] - fy[:, 1]).abs() / 2
    py = (robot.data.root_pos_w[:, :2] * ly).sum(-1)
    com = (robot.data.body_pos_w * mass.unsqueeze(-1)).sum(1) / mass.sum(1, keepdim=True); cy = (com[:, :2] * ly).sum(-1)
    sgn_foot = torch.sign(fy - mid.unsqueeze(1))                       # 각 발 쪽 부호
    sp = ((py - mid) / dd.clamp(min=1e-3)).unsqueeze(1) * sgn_foot   # (N,2): 발 k 를 지지발로 볼 때 +
    sc = ((cy - mid) / dd.clamp(min=1e-3)).unsqueeze(1) * sgn_foot
    pg = robot.data.projected_gravity_b
    jp = torch.rad2deg(robot.data.joint_pos)
    rec["fz"].append((F[..., 2] / mg.unsqueeze(1)).cpu()); rec["sp"].append(sp.cpu()); rec["sc"].append(sc.cpu())
    rec["ha"].append(jp[:, ja].cpu()); rec["ar"].append(jp[:, jr].cpu())
    rec["roll"].append(torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2])).cpu()); rec["pitch"].append(torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])).cpu())
    rec["w"].append(robot.data.root_ang_vel_b[:, :2].norm(dim=1).cpu()); rec["fxy"].append(robot.data.body_pos_w[:, bid, :2].cpu())
    rec["c"].append((F.norm(dim=-1) > 5.0).cpu()); rec["alive"].append(alive.cpu().clone())
R = {k: torch.stack(v_).numpy() for k, v_ in rec.items()}     # (T, N, ...)
T = R["fz"].shape[0]; i0 = int(a.t0 / dt); bk = back.cpu().numpy()
tilt = np.hypot(R["roll"], R["pitch"])
ev = []
for e in range(N):
    c = R["c"][:, e]; took = False
    for f in (0, 1):
        k = i0
        while k < T - 1:
            if c[k - 1, f] and not c[k, f] and R["alive"][k, e]:
                j = k
                while j < T and not c[j, f]: j += 1
                air = (j - k) * dt
                if air < 0.06: k = j; continue
                took = True; s = 1 - f; k0 = max(k - H, 0)
                disp = float(np.linalg.norm(R["fxy"][min(j, T - 1), e, f] - R["fxy"][k, e, f])) if j < T else float("nan")
                d300 = float(np.linalg.norm(R["fxy"][min(k + int(0.3 / dt), T - 1), e, f] - R["fxy"][min(k + int(0.15 / dt), T - 1), e, f]))
                fz_s = R["fz"][k0:k + 1, e, s]; fz_l = R["fz"][k0:k + 1, e, f]
                grp = ("A 뒤 유효 스텝" if disp >= 0.02 else "A' 뒤 짧은 들림") if bk[e] else ("B' 앞 유효 스텝" if disp >= 0.02 else "B 앞 짧은 들림")
                ev.append((grp, fz_s.mean(), fz_s.max(), (fz_l[0] - fz_l[-1]) / max(fz_l[0], 1e-3), R["sp"][k0:k + 1, e, s].mean(), R["sc"][k0:k + 1, e, s].mean(),
                           R["ha"][k, e, s] - R["ha"][k0, e, s], R["ar"][k, e, s] - R["ar"][k0, e, s], R["roll"][k, e], R["pitch"][k, e], R["w"][k, e],
                           100 * d300, air, 100 * disp if not math.isnan(disp) else float("nan")))
                k = j
            else:
                k += 1
    if not bk[e] and not took and not R["alive"][-1, e]:
        ks = np.where((tilt[i0:, e] > 6.5) & R["alive"][i0:, e])[0]
        if len(ks):
            k = i0 + int(ks[0]); k0 = max(k - H, 0); s = int(np.argmax(R["fz"][k, e])); f = 1 - s
            fz_s = R["fz"][k0:k + 1, e, s]; fz_l = R["fz"][k0:k + 1, e, f]
            ev.append(("C 앞 이륙 없이 낙상", fz_s.mean(), fz_s.max(), (fz_l[0] - fz_l[-1]) / max(fz_l[0], 1e-3), R["sp"][k0:k + 1, e, s].mean(), R["sc"][k0:k + 1, e, s].mean(),
                       R["ha"][k, e, s] - R["ha"][k0, e, s], R["ar"][k, e, s] - R["ar"][k0, e, s], R["roll"][k, e], R["pitch"][k, e], R["w"][k, e],
                       float("nan"), float("nan"), float("nan")))
cols = ["지지 Fz/mg 평균", "지지 Fz/mg 최대", "드는 발 Fz 감소율", "s_pelvis", "s_com", "Δ지지 힙롤°", "Δ지지 발목롤°", "롤°", "피치°", "|ω|", "150–300ms 변위 cm", "공중 s", "스텝 변위 cm"]
L = [f"이륙 직전 150 ms 진단 v2 · ckpt {ck} · 앞 {a.F_f} N 램프(상승 0.5·유지 1.0 s) / 뒤 {a.F_b} N 충격(0.3 s) · env {N} (반반) · 지연 1틱·LPF 0.27",
     "| 그룹 | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
for g in ("A 뒤 유효 스텝", "A' 뒤 짧은 들림", "B 앞 짧은 들림", "B' 앞 유효 스텝", "C 앞 이륙 없이 낙상"):
    E = np.array([x[1:] for x in ev if x[0] == g], dtype=float)
    if len(E) == 0:
        L.append(f"| {g} | 0 |" + " |" * len(cols)); continue
    cells = []
    for j in range(len(cols)):
        col = E[:, j][~np.isnan(E[:, j])]
        cells.append("—" if len(col) == 0 else f"{np.median(col):.2f} [{np.percentile(col, 25):.2f}, {np.percentile(col, 75):.2f}]")
    L.append(f"| {g} | {len(E)} | " + " | ".join(cells) + " |")
nf = int((~bk & ~R["alive"][-1]).sum()); nb = int((bk & ~R["alive"][-1]).sum())
L.append(f"낙상: 앞 {nf}/{int((~bk).sum())} · 뒤 {nb}/{int(bk.sum())}. 값 = 중앙 [사분위]. s: 0 = 두 발 가운데, +1 = 지지발 중심. C 의 기준 시각 = 기울기 6.5° 첫 도달, 지지발 = 더 실린 발.")
open(f"/home/ryu/humanoid_leg_test1/reports/gen_g22r/{a.out}", "w").write("\n".join(L) + "\n"); os._exit(0)
