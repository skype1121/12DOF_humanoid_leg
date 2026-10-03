"""NEXT 2 준비(1004): 시뮬 성공 스텝 궤적 추출 → 위상 정규화 기준 동작의 원자료.
정책 롤아웃(외란 = 평가와 같은 모양: 뒤 F_b N 충격 0.3 s / 앞 F_f N 램프 상승 0.5·유지 1.0 s)에서 valid 스텝
(예정 아님 — 실제 이륙 ≥ 60 ms & 재착지 수평 ≥ 2 cm)마다 이륙 기준 [−PRE, +POST] 틱 창을 저장.
좌우 표준화: 오른발 스윙 사건은 시뮬 관절 미러(symmetry._JOINT_PERM/_JOINT_SIGN)와 횡 성분 부호 반전으로 '왼발 스윙' 형으로 바꾼다.
저장(npz): q[n,W,12](rad, 관절 순서 = env), qd, fz[n,W,2](/mg, [스윙, 지지]), c[n,W,2], foot_rel[n,W,2,3](골반 기준 요 프레임 [앞,옆,위] m, [스윙, 지지]),
roll/pitch[n,W](deg, 롤은 스윙 쪽 +), h[n,W](골반고), i_td[n](착지 틱, 창 기준), grp[n](0 뒤·1 앞), disp[n](m), default_q(12).
요약(txt): 그룹별 중앙 타이밍 — 하중 소거 시작(스윙 Fz 가 0.45 mg 아래로 내려가는 이륙 전 마지막 시점), 공중 시간, 착지→하중 수용(스윙 Fz > 0.25 mg 100 ms)."""
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-G22R-v0"); p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--F_f", type=float, default=15.0); p.add_argument("--F_b", type=float, default=18.0); p.add_argument("--front_frac", type=float, default=0.75)
p.add_argument("--t0", type=float, default=3.0); p.add_argument("--seed", type=int, default=1); p.add_argument("--out", default="ref_steps")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import os, numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from rl_walking.symmetry import _JOINT_PERM, _JOINT_SIGN  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.seed = a.seed; ec.curriculum = None; ec.events.push_robot = None
ec.actions.joint_pos.lpf_beta_range = (0.27, 0.27); ec.actions.joint_pos.min_delay_steps = ec.actions.joint_pos.max_delay_steps = 1
ec.episode_length_s = a.t0 + 6.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device; dt = U.step_dt
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
N = U.num_envs; front = torch.rand(N, device=dev) < a.front_frac
robot = U.scene["robot"]; jn = list(robot.data.joint_names)
from rl_walking.deploy.policy_runner_stage4 import JOINT_ORDER  # noqa
assert jn == list(JOINT_ORDER), "env 관절 순서 ≠ JOINT_ORDER (미러 순열 전제)"
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
hs = mdp_hang._state(U)
ex = torch.zeros(N, 3, device=dev); ex[:, 0] = 1.0
fw = mdp_hang._quat_apply(robot.data.root_quat_w, ex)[:, :2]; fw = fw / fw.norm(dim=-1, keepdim=True)
hs["dst_kind"][:] = 0.0; hs["dst_F"][:] = 0.0
hs["dst_kind"][:, 0] = torch.where(front, 1.0, 2.0); hs["dst_t0"][:, 0] = a.t0
hs["dst_rise"][:, 0] = 0.5; hs["dst_hold"][:, 0] = torch.where(front, 1.0, 0.3); hs["dst_F"][:, 0] = torch.where(front, a.F_f, a.F_b)
hs["dst_dir"][:, 0] = fw * torch.where(front, 1.0, -1.0).unsqueeze(1)
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
mg = float(mdp_hang.G) * robot.data.default_mass.sum(1).to(dev)
REC = {k: [] for k in ("q", "qd", "fz", "c", "frel", "roll", "pitch", "h", "fxy", "alive", "hx")}
alive = torch.ones(N, dtype=torch.bool, device=dev)
steps = int((a.t0 + 5.5) / dt)
for i in range(steps):
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    alive &= ~d.bool()
    F = U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :]
    qw = robot.data.root_quat_w
    yaw = torch.atan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))
    hx = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1); hy = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    rel = robot.data.body_pos_w[:, bid, :] - robot.data.root_pos_w.unsqueeze(1)
    frel = torch.stack([(rel[..., :2] * hx.unsqueeze(1)).sum(-1), (rel[..., :2] * hy.unsqueeze(1)).sum(-1), rel[..., 2]], -1)
    pg = robot.data.projected_gravity_b
    REC["q"].append(robot.data.joint_pos.cpu()); REC["qd"].append(robot.data.joint_vel.cpu())
    REC["fz"].append((F[..., 2] / mg.unsqueeze(1)).cpu()); REC["c"].append((F.norm(dim=-1) > 5.0).cpu())
    REC["frel"].append(frel.cpu()); REC["roll"].append(torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2])).cpu())
    REC["pitch"].append(torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])).cpu()); REC["h"].append(robot.data.root_pos_w[:, 2].cpu())
    REC["fxy"].append(robot.data.body_pos_w[:, bid, :2].cpu()); REC["alive"].append(alive.cpu().clone()); REC["hx"].append(hx.cpu())
R = {k: torch.stack(x).numpy() for k, x in REC.items()}
T = R["q"].shape[0]; i0 = int(a.t0 / dt); PRE, POST = 15, 30
perm = np.array(_JOINT_PERM); sgn = np.array(_JOINT_SIGN, dtype=np.float32)
fr = front.cpu().numpy()
out = {k: [] for k in ("q", "qd", "fz", "c", "frel", "roll", "pitch", "h", "i_td", "grp", "disp", "dfwd", "t_since", "t_unload", "t_air", "t_load")}
for e in range(N):
    c = R["c"][:, e]
    for f in (0, 1):
        k = i0
        while k < T - 1:
            if c[k - 1, f] and not c[k, f] and R["alive"][k, e]:
                j = k
                while j < T and not c[j, f]: j += 1
                air = (j - k) * dt
                if air < 0.06 or j >= T or k - PRE < 0 or k + POST >= T: k = j; continue
                disp = float(np.linalg.norm(R["fxy"][j, e, f] - R["fxy"][k, e, f]))
                if disp < 0.02 or not R["alive"][min(j + 10, T - 1), e]: k = j; continue
                s_ = 1 - f; w = slice(k - PRE, k + POST)
                q = R["q"][w, e]; qd = R["qd"][w, e]
                if f == 1:      # 오른발 스윙 → 왼발 스윙 표준형
                    q = q[:, perm] * sgn; qd = qd[:, perm] * sgn
                fz = R["fz"][w, e][:, [f, s_]]; cc = R["c"][w, e][:, [f, s_]]; frl = R["frel"][w, e][:, [f, s_]].copy()
                rl = R["roll"][w, e].copy()
                if f == 1: frl[..., 1] *= -1; rl *= -1                       # 횡 성분: 스윙 쪽 + 로
                fzs = R["fz"][:, e, f]; kk = k - 1
                while kk > k - PRE and fzs[kk] < 0.45: kk -= 1
                lt = 0.0; tl = float("nan")
                for jj in range(j, min(j + 25, T)):
                    lt = lt + dt if fzs[jj] > 0.25 else 0.0
                    if lt >= 0.10: tl = (jj - j) * dt; break
                for kname, val in (("q", q), ("qd", qd), ("fz", fz), ("c", cc), ("frel", frl), ("roll", rl), ("pitch", R["pitch"][w, e]), ("h", R["h"][w, e])):
                    out[kname].append(val)
                out["i_td"].append(PRE + (j - k)); out["grp"].append(int(fr[e])); out["disp"].append(disp)
                out["dfwd"].append(float(((R["fxy"][j, e, f] - R["fxy"][k, e, f]) * R["hx"][k, e]).sum())); out["t_since"].append((k - i0) * dt)
                out["t_unload"].append((k - kk) * dt); out["t_air"].append(air); out["t_load"].append(tl)
                k = j
            else:
                k += 1
O = {k: np.array(v_) for k, v_ in out.items()}
O["default_q"] = robot.data.default_joint_pos[0].cpu().numpy(); O["dt"] = np.array(dt); O["pre"] = np.array(PRE)
D = "/home/ryu/humanoid_leg_test1/reports/gen_step"
np.savez_compressed(f"{D}/{a.out}.npz", **O)
L = [f"성공 스텝 추출 · ckpt {ck} · 앞 {a.F_f} N 램프 {int(fr.sum())} env / 뒤 {a.F_b} N 충격 {int((~fr).sum())} env · seed {a.seed}"]
for g, nm in ((0, "뒤"), (1, "앞")):
    m = O["grp"] == g if len(O["grp"]) else np.array([], bool)
    n = int(m.sum())
    if n == 0: L.append(f"{nm}: 0"); continue
    med = lambda x: f"{np.nanmedian(x[m]):.3f} [{np.nanpercentile(x[m], 25):.3f}, {np.nanpercentile(x[m], 75):.3f}]"
    dx = O["frel"][m, O["i_td"][m] - 1 if False else 0, 0, 0] if False else None
    td_dx = np.array([O["frel"][i, O["i_td"][i], 0, 0] - O["frel"][i, PRE, 0, 0] for i in np.where(m)[0]])
    L.append(f"{nm}: n {n} · 하중 소거 시작→이륙 {med(O['t_unload'])} s · 공중 {med(O['t_air'])} s · 착지→하중 수용 {med(O['t_load'])} s · "
             f"수평 변위 {med(O['disp'])} m · 월드 앞축 변위(+앞) {med(O['dfwd'])} m · 앞 스텝(+2 cm 이상) {int((O['dfwd'][m] >= 0.02).sum())}건 · 밀기→이륙 {med(O['t_since'])} s · 골반 기준 스윙발 앞축 이동(착지−이륙) {np.median(td_dx):+.3f} m")
open(f"{D}/{a.out}.txt", "w").write("\n".join(L) + "\n"); os._exit(0)
