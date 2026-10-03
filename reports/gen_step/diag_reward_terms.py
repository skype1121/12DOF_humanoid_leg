"""1004 구조 진단: 같은 정책(G21a-11300)을 스텝 모드에서 보조 α=2(스텝 발생, 개루프 이륙 ~30 %)와 α=0(서기)로 돌려
보상 항별 '틱당 평균 기여(가중치·dt 포함)'를 비교 — 스텝을 하면 어느 항에서 손해/이득인지(A0·A0b·A0c 가 모두 이륙 억제로 수렴한 이유 후보).
스텝 모드 env 만 집계(전 env 스텝 모드 vx 0.05), 2 s 서기 뒤 8 s. 이륙이 실제 일어난 env-틱과 아닌 틱으로도 나눠 본다(α=2)."""
import argparse, sys, os
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-StepA0c-v0"); p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--alpha", type=float, default=2.0); p.add_argument("--out", default="diag_reward_terms")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
os.environ["STEP_ASSIST_ALPHA"] = str(a.alpha)
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.curriculum = None
c_ = ec.commands.base_velocity; c_.resampling_time_range = (1e6, 1e6); c_.rel_standing_envs = 1.0
act = ec.actions.joint_pos; act.lpf_beta_range = (0.27, 0.27); act.min_delay_steps = act.max_delay_steps = 1
ec.episode_length_s = 11.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device; dt = U.step_dt
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
cmd = U.command_manager.get_term("base_velocity"); RM = U.reward_manager
names = list(RM.active_terms); N = U.num_envs
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
fid = mdp_hang._feet_body_ids(U)
ret = torch.zeros(N, device=dev); acc_all = np.zeros(len(names)); acc_air = np.zeros(len(names)); acc_gnd = np.zeros(len(names)); n_all = n_air = n_gnd = 0
alive = torch.ones(N, dtype=torch.bool, device=dev); falls = 0
for i in range(int(10.0 / dt)):
    on = i * dt >= 2.0
    cmd.vel_command_b[:, 0] = 0.05 if on else 0.0; cmd.vel_command_b[:, 1:] = 0.0; cmd.is_standing_env[:] = False; cmd.is_heading_env[:] = False
    with torch.inference_mode():
        obs, rew, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    if not on:
        continue
    ret += torch.where(alive, rew.to(dev), torch.zeros_like(ret))
    step_r = RM._step_reward.detach().cpu().numpy() * dt          # (N, terms): 가중치 곱해진 값 × dt = 실효 기여
    c = (U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0)
    air = (~c).any(1).cpu().numpy(); al = alive.cpu().numpy()
    acc_all += step_r[al].sum(0); n_all += int(al.sum())
    acc_air += step_r[al & air].sum(0); n_air += int((al & air).sum())
    acc_gnd += step_r[al & ~air].sum(0); n_gnd += int((al & ~air).sum())
    falls += int((alive & d.bool()).sum()); alive &= ~d.bool()
L = [f"# 보상 항별 틱당 실효 기여(×dt) · {a.task} · α={a.alpha} · ckpt {ck}", f"스텝 모드 env {N} · 8 s · 낙상 {falls}/{N} · 한 발이라도 공중인 env-틱 비율 {100.0 * n_air / max(n_all, 1):.1f} %",
     "| 항 | 전체 평균 | 공중 틱 평균 | 접지 틱 평균 | 공중−접지 |", "|---|---|---|---|---|"]
rows = []
for j, nm in enumerate(names):
    A_ = acc_all[j] / max(n_all, 1); B_ = acc_air[j] / max(n_air, 1); C_ = acc_gnd[j] / max(n_gnd, 1)
    rows.append((B_ - C_, nm, A_, B_, C_))
for dlt, nm, A_, B_, C_ in sorted(rows):
    L.append(f"| {nm} | {A_:+.5f} | {B_:+.5f} | {C_:+.5f} | {dlt:+.5f} |")
L.append(f"env 당 8 s 보상 합(낙상 뒤 0): 평균 {float(ret.mean()):+.3f} · 생존 env {float(ret[alive].mean()) if alive.any() else float('nan'):+.3f} · 낙상 env {float(ret[~alive].mean()) if (~alive).any() else float('nan'):+.3f}")
L.append(f"합계: 전체 {acc_all.sum() / max(n_all, 1):+.4f} · 공중 {acc_air.sum() / max(n_air, 1):+.4f} · 접지 {acc_gnd.sum() / max(n_gnd, 1):+.4f} (틱당)")
open(f"/home/ryu/humanoid_leg_test1/reports/gen_step/{a.out}_a{a.alpha}.md", "w").write("\n".join(L) + "\n"); os._exit(0)
