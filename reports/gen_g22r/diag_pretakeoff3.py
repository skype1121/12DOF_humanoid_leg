"""1003 진단 v3(외부 조언 3차: C군 밀기 시작 기준·좌우 미러 정규화·지지 하중 4종·−150/−100/−50/0 ms 3칸) — v2 기반.
원 v2 설명: 1003 진단 v2(외부 조언 2차): 이륙 직전 150 ms — 체중 이동 부족 / 스윙 생성 부족 / 서기 국소해 분리. 시뮬 실제 접촉력 기준(패드 아님).
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
p.add_argument("--out", default="diag_pretakeoff3_out.txt")
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
rec = {k: [] for k in ("fz", "sp", "sc", "ha", "ar", "roll", "pitch", "w", "fxy", "c", "alive", "shk", "thg")}
bn = list(robot.data.body_names)
BI = {s: [bn.index(f"{lr}_{s}_joint") for lr in ("left", "right")] for s in ("hip_f", "knee", "ankle_f")}
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
    bp = robot.data.body_pos_w
    def lat_tilt(lo, hi):     # 아래→위 벡터의 횡 기울기 [deg], 위 끝이 그 발(지지발 후보) 쪽이면 + (발별 부호 = sgn_foot)
        vv = bp[:, BI[hi]] - bp[:, BI[lo]]                                    # (N,2,3)
        lat = (vv[..., :2] * ly.unsqueeze(1)).sum(-1)
        return torch.rad2deg(torch.atan2(lat, vv[..., 2].clamp(min=1e-3))) * sgn_foot
    rec["shk"].append(lat_tilt("ankle_f", "knee").cpu()); rec["thg"].append(lat_tilt("knee", "hip_f").cpu())
R = {k: torch.stack(v_).numpy() for k, v_ in rec.items()}     # (T, N, ...)
T = R["fz"].shape[0]; i0 = int(a.t0 / dt); bk = back.cpu().numpy()
tilt = np.hypot(R["roll"], R["pitch"])
BINS = ((-7, -5), (-5, -2), (-2, 0))     # 틱 상대 구간 ≈ −150~−100 · −100~−50 · −50~0 ms (50 Hz, 1틱 20 ms)
def feats(e, k, s, f):
    """기준 틱 k, 지지발 s, 드는 발 f — 지지발 쪽 + 로 미러 정규화된 값."""
    k0 = max(k - 7, 0); w = slice(k0, k + 1)
    fz_s, fz_f = R["fz"][w, e, s], R["fz"][w, e, f]
    d = dict(fz_mean=fz_s.mean(), fz_peak=fz_s.max(), fz_ev=R["fz"][k, e, s], fz_sw_ev=R["fz"][k, e, f],
             sp=R["sp"][w, e, s].mean(), sc=R["sc"][w, e, s].mean(),
             d_shk=R["shk"][k, e, s] - R["shk"][k0, e, s], d_thg=R["thg"][k, e, s] - R["thg"][k0, e, s],
             roll=R["roll"][k, e] * (1 if s == 0 else -1), pitch=R["pitch"][k, e], w=R["w"][k, e])
    for bi, (lo, hi) in enumerate(BINS):
        ww = slice(max(k + lo, 0), k + hi + 1)
        d[f"b{bi}_sc"] = R["sc"][ww, e, s].mean(); d[f"b{bi}_fs"] = R["fz"][ww, e, s].mean(); d[f"b{bi}_ff"] = R["fz"][ww, e, f].mean()
    return d
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
                took = True; s_ = 1 - f
                disp = float(np.linalg.norm(R["fxy"][min(j, T - 1), e, f] - R["fxy"][k, e, f]))
                d300 = float(np.linalg.norm(R["fxy"][min(k + int(0.3 / dt), T - 1), e, f] - R["fxy"][min(k + int(0.15 / dt), T - 1), e, f]))
                grp = ("A 뒤 유효 스텝" if disp >= 0.02 else "A' 뒤 짧은 들림") if bk[e] else ("B' 앞 유효 스텝" if disp >= 0.02 else "B 앞 짧은 들림")
                d = feats(e, k - 1, s_, f); d.update(d300=100 * d300, air=air, disp=100 * disp, t_ev=(k - i0) * dt)
                ev.append((grp, d)); k = j
            else:
                k += 1
    if not bk[e] and not took and not R["alive"][-1, e]:
        ks = np.where((tilt[i0:, e] > 6.5) & R["alive"][i0:, e])[0]
        if len(ks):
            k = i0 + int(ks[0]); s_ = int(np.argmax(R["fz"][k, e])); f = 1 - s_
            d = feats(e, k, s_, f); d.update(t_ev=(k - i0) * dt)
            # 밀기 시작 → 6.5° 구간 안의 '시도' 최대치 (지지발 = 6.5° 시점 더 실린 발, 지지 쪽 +)
            ww = slice(i0, k + 1)
            fz_light0 = max(R["fz"][i0, e, f], 1e-3)
            d.update(onset_fs=R["fz"][i0, e, s_], onset_ff=R["fz"][i0, e, f], onset_sc=R["sc"][i0, e, s_],
                     try_unload=1 - R["fz"][ww, e, f].min() / fz_light0, try_sc=np.abs(R["sc"][ww, e, s_]).max(),
                     try_shk=np.abs(R["shk"][ww, e, s_] - R["shk"][i0, e, s_]).max())
            ev.append(("C 앞 이륙 없이 낙상", d))
G = ("A 뒤 유효 스텝", "A' 뒤 짧은 들림", "B 앞 짧은 들림", "B' 앞 유효 스텝", "C 앞 이륙 없이 낙상")
def cell(g, key, nd=2):
    x = np.array([d[key] for gg, d in ev if gg == g and key in d], dtype=float); x = x[~np.isnan(x)]
    return "—" if len(x) == 0 else f"{np.median(x):.{nd}f} [{np.percentile(x, 25):.{nd}f}, {np.percentile(x, 75):.{nd}f}]"
def table(title, keys):
    out = [f"\n### {title}", "| 그룹 | n | " + " | ".join(k for k, _ in keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for g in G:
        n = sum(1 for gg, _ in ev if gg == g)
        out.append(f"| {g} | {n} | " + " | ".join(cell(g, kk) for _, kk in keys) + " |")
    return out
L = [f"# 이륙 직전 진단 v3 · ckpt {ck}", f"앞 {a.F_f} N 램프(상승 0.5·유지 1.0 s) / 뒤 {a.F_b} N 충격(0.3 s) · env {N} (반반) · 지연 1틱·LPF 0.27 · 시뮬 실제 접촉력",
     "기준 시각: A/B = 이륙 직전 틱(접촉 상실 ≥ 60 ms 로 확인된 이륙), C = 기울기 6.5° 첫 도달. 모든 횡 값은 **지지발 쪽 +**(좌우 미러 정규화). 값 = 중앙 [사분위]."]
L += table("하중·횡이동(−150→0 ms)", [("지지 Fz/mg 평균", "fz_mean"), ("지지 Fz/mg 최고", "fz_peak"), ("지지 Fz/mg 이륙 순간", "fz_ev"),
            ("드는 발 Fz/mg 이륙 순간", "fz_sw_ev"), ("s_pelvis", "sp"), ("s_com", "sc")])
L += table("자세·운동(지지발 쪽 +)", [("Δ지지 정강이 횡기울기°", "d_shk"), ("Δ지지 허벅지 횡기울기°", "d_thg"), ("몸통 롤°(지지 쪽 +)", "roll"),
            ("피치°", "pitch"), ("|ω_xy|", "w"), ("밀기→기준 시각 s", "t_ev")])
L += table("시간 순서(3칸: −150~−100 · −100~−50 · −50~0 ms)", [("s_com ①", "b0_sc"), ("s_com ②", "b1_sc"), ("s_com ③", "b2_sc"),
            ("지지 Fz ①", "b0_fs"), ("지지 Fz ②", "b1_fs"), ("지지 Fz ③", "b2_fs"), ("드는 Fz ①", "b0_ff"), ("드는 Fz ②", "b1_ff"), ("드는 Fz ③", "b2_ff")])
L += table("스윙(이륙 뒤)", [("150–300 ms 변위 cm", "d300"), ("공중 s", "air"), ("스텝 변위 cm", "disp")])
L += table("C 전용: 밀기 시작 시점 값·밀기→6.5° 구간 시도 최대", [("밀기 시작 지지 Fz", "onset_fs"), ("밀기 시작 가벼운 발 Fz", "onset_ff"), ("밀기 시작 s_com", "onset_sc"),
            ("가벼운 발 하중 최대 감소율", "try_unload"), ("|s_com| 최대", "try_sc"), ("|Δ정강이 횡기울기| 최대°", "try_shk")])
nf = int((~bk & ~R["alive"][-1]).sum()); nb = int((bk & ~R["alive"][-1]).sum())
L.append(f"\n낙상: 앞 {nf}/{int((~bk).sum())} · 뒤 {nb}/{int(bk.sum())}. 정강이·허벅지 횡기울기 = 아래→위 링크 벡터의 횡 성분 각(위 끝이 지지발 쪽이면 +). 50 Hz 기록이라 3칸은 2–3 틱씩.")
open(f"/home/ryu/humanoid_leg_test1/reports/gen_g22r/{a.out}", "w").write("\n".join(L) + "\n"); os._exit(0)
