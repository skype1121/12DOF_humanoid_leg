"""STEP primitive fail-fast 평가(1004) — 명령 스텝 모드에서 '첫 비대칭 하중 → 이륙 → 착지 → 하중 수용'이 생기는지.
env: StepB 계열 태스크(관측 계약 G21a 그대로), 학습 DR 유지, 외란 없음, LPF β 0.27·지연 고정, 명령 리샘플·커리큘럼 끔.
절차: 0–T_STAND s 전 env 서기(명령 0) → 이후 T_STEP s 동안 절반 env 스텝 모드(vx 0.05), 나머지 서기 유지.
스텝 모드 지표(예정 반주기 = 1.1 Hz 클록 위상, 앞 반주기 왼발 스윙 — mdp_hang.sb0_step 과 같은 일정):
  예정 반주기마다 전환 구간(앞 16 %) 의 (지지 Fz − 스윙 Fz)/mg 평균, 지지 Fz 최대, 스윙 Fz 최소 →
  load_ready(지지 > 0.65 mg) · unload(스윙 < 0.35 mg) 비율, valid 이륙(예정 스윙발 ≥ 60 ms 공중) · 착지(그 뒤 재접촉 & 수평 ≥ 2 cm) ·
  하중 수용(착지 발 > 0.25 mg 100 ms) 비율(예정 반주기 대비). 서기 그룹: 무요청 이륙(≥ 60 ms) env 비율. 낙상: 그룹별."""
import argparse, os, sys, math
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
REPO = "/home/ryu/humanoid_leg_test1"; sys.path.insert(0, REPO)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-StepB1-v0"); p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--delay", type=int, default=1); p.add_argument("--seed", type=int, default=1)
p.add_argument("--t_stand", type=float, default=2.0); p.add_argument("--t_step", type=float, default=8.0); p.add_argument("--vx", type=float, default=0.05)
p.add_argument("--freq", type=float, default=1.1); p.add_argument("--out_tag", default="step")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from rl_walking import mdp_hang  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.seed = a.seed; ec.curriculum = None
c_ = ec.commands.base_velocity; c_.resampling_time_range = (1e6, 1e6); c_.rel_standing_envs = 1.0
ec.events.disturbance = None
if getattr(ec.events, "push_robot", None) is not None: ec.events.push_robot = None
act = ec.actions.joint_pos; act.lpf_beta_range = (0.27, 0.27); act.min_delay_steps = act.max_delay_steps = a.delay
if hasattr(act, "delay_probs"): act.delay_probs = ()
ec.episode_length_s = a.t_stand + a.t_step + 1.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device; dt = U.step_dt
ck = get_checkpoint_path(os.path.join(REPO, "logs", "rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
N = U.num_envs; stepg = (torch.arange(N, device=dev) % 2) == 0
robot = U.scene["robot"]; sensor = U.scene.sensors["contact_forces"]; fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
mg = float(mdp_hang.G) * robot.data.default_mass.sum(1).to(dev); ar = torch.arange(N, device=dev)
cmd = U.command_manager.get_term("base_velocity")
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
z = lambda: torch.zeros(N, device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev)
half_prev = torch.full((N,), -1.0, device=dev)
h_asym_s, h_asym_n, h_fst_max, h_fsw_min = z(), z(), z(), torch.full((N,), 9.0, device=dev)
h_to, h_td, h_ld, ld_t = z(), z(), z(), z()
S = {k: 0.0 for k in ("n_half", "asym_sum", "asym_n", "ready", "unload", "to", "td", "ld")}
prev_c = torch.ones(N, 2, device=dev); air = torch.zeros(N, 2, device=dev); lift_xy = torch.zeros(N, 2, 2, device=dev)
unsol = torch.zeros(N, dtype=torch.bool, device=dev)
steps = int((a.t_stand + a.t_step) / dt)
for i in range(steps):
    t_wall = i * dt
    on = t_wall >= a.t_stand
    cmd.vel_command_b[:, 0] = torch.where(stepg & torch.tensor(on, device=dev), a.vx, 0.0); cmd.vel_command_b[:, 1:] = 0.0
    cmd.is_standing_env[:] = False; cmd.is_heading_env[:] = False
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    alive &= ~d.bool()
    F = sensor.data.net_forces_w[:, fid, :]; c = (F.norm(dim=-1) > 5.0).float(); fz = F[..., 2] / mg.unsqueeze(1)
    fxy = robot.data.body_pos_w[:, bid, :2]
    t = U.episode_length_buf.to(torch.float32) * dt
    u = torch.remainder(a.freq * t, 1.0); half = (u >= 0.5).float(); vv = torch.remainder(2.0 * u, 1.0)
    sw = half.long(); stc = 1 - sw
    lifted = (prev_c > 0.5) & (c < 0.5); landed = (prev_c < 0.5) & (c > 0.5)
    lift_xy = torch.where(lifted.unsqueeze(-1), fxy, lift_xy)
    air_prev = air.clone(); air = torch.where(c < 0.5, air + dt, torch.zeros_like(air))
    newh = half != half_prev
    fin = newh & (half_prev >= 0) & stepg & alive & torch.tensor(on, device=dev)
    if bool(fin.any()):
        S["n_half"] += float(fin.sum()); S["asym_sum"] += float(h_asym_s[fin].sum()); S["asym_n"] += float(h_asym_n[fin].sum())
        S["ready"] += float((h_fst_max[fin] > 0.65).sum()); S["unload"] += float((h_fsw_min[fin] < 0.35).sum())
        S["to"] += float(h_to[fin].sum()); S["td"] += float(h_td[fin].sum()); S["ld"] += float(h_ld[fin].sum())
    for x in (h_asym_s, h_asym_n, h_fst_max, h_to, h_td, h_ld, ld_t):
        x[newh] = 0.0
    h_fsw_min[newh] = 9.0; half_prev = half
    tr = vv < 0.16
    fst, fsw = fz[ar, stc], fz[ar, sw]
    h_asym_s += torch.where(tr, fst - fsw, torch.zeros_like(fst)); h_asym_n += tr.float()
    h_fst_max = torch.where(tr, torch.maximum(h_fst_max, fst), h_fst_max); h_fsw_min = torch.where(tr, torch.minimum(h_fsw_min, fsw), h_fsw_min)
    sw_air = air[ar, sw]
    h_to = torch.maximum(h_to, (sw_air >= 0.06).float())
    disp = (fxy[ar, sw] - lift_xy[ar, sw]).norm(dim=-1)
    h_td = torch.maximum(h_td, (landed[ar, sw] & (air_prev[ar, sw] >= 0.06) & (disp >= 0.02)).float())
    ld_t = torch.where((h_td > 0.5) & (fsw > 0.25), ld_t + dt, torch.zeros_like(ld_t))
    h_ld = torch.maximum(h_ld, (ld_t >= 0.10).float())
    if on:
        unsol |= (~stepg) & alive & (air >= 0.06).any(1)
    prev_c = c
n = max(S["n_half"], 1.0)
fall_s = 100.0 * float((~alive[stepg]).float().mean()); fall_q = 100.0 * float((~alive[~stepg]).float().mean())
L = [f"STEP fail-fast 평가 · {a.task} · {ck}", f"지연 {a.delay} · LPF 0.27 · seed {a.seed} · env {N}(스텝 {int(stepg.sum())}/서기 {int((~stepg).sum())}) · 서기 {a.t_stand}s → 스텝 {a.t_step}s · vx {a.vx}",
     f"예정 반주기 {int(S['n_half'])} · 전환 구간 하중 비대칭 평균 {S['asym_sum'] / max(S['asym_n'], 1):+.3f} mg · load_ready(지지>0.65) {100 * S['ready'] / n:.1f} % · unload(스윙<0.35) {100 * S['unload'] / n:.1f} %",
     f"valid 이륙 {100 * S['to'] / n:.1f} % · 착지 {100 * S['td'] / n:.1f} % · 하중 수용 {100 * S['ld'] / n:.1f} % (예정 반주기 대비)",
     f"낙상: 스텝 그룹 {fall_s:.1f} % · 서기 그룹 {fall_q:.1f} % · 서기 그룹 무요청 이륙 env {100.0 * float(unsol[~stepg].float().mean()):.1f} %"]
txt = "\n".join(L); os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
open(os.path.join(REPO, "log_picture", f"step_eval_{a.out_tag}.txt"), "w").write(txt + "\n"); print(txt, flush=True)
os._exit(0)
