"""1003 진단: G22R 환경에서 G21a-11300 정책으로 굴려 '무장 안 된 이륙'·낙상이 에피소드 어느 시각에 나는지(스폰 착지 과도인지) 확인."""
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-G22R-v0"); p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--steps", type=int, default=750)
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import os, numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.curriculum = None
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=U.device)
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
N = U.num_envs; first_un = np.full(N, np.nan); t_end = np.full(N, np.nan); why = [""] * N; ever = np.zeros(N, bool)
cat0 = None; done_once = np.zeros(N, bool)
for i in range(a.steps):
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    if cat0 is None:
        cat0 = mdp_hang._g22(U)["cat"].cpu().numpy().copy()
    st = mdp_hang._g2(U); t = i * U.step_dt
    un = st["ev_unarmed"].cpu().numpy() > 0.5
    m = un & np.isnan(first_un) & ~done_once; first_un[m] = t
    dd = d.cpu().numpy().astype(bool) & ~done_once
    tm = U.termination_manager
    for k in np.where(dd)[0]:
        t_end[k] = t; why[k] = "timeout" if bool(tm.time_outs[k]) else ",".join(n for n in tm.active_terms if bool(tm.get_term(n)[k]))
    done_once |= dd
q = cat0 < 0.5
OUT = []
print = lambda *x, **k: OUT.append(" ".join(str(y) for y in x))
print(f"무외란 env {q.sum()} · 첫 무장 안 된 이륙 있음 {np.isfinite(first_un[q]).sum()} · 그 시각 분포(s): {np.round(np.nanpercentile(first_un[q], [10, 50, 90]), 2) if np.isfinite(first_un[q]).any() else '-'}")
print(f"  시각 히스토그램(0.5 s 칸): {np.histogram(first_un[q][np.isfinite(first_un[q])], bins=np.arange(0, a.steps * U.step_dt + 0.5, 0.5))[0].tolist()}")
for c, nm in ((0, '무외란'), (1, '앞'), (2, '뒤')):
    m = cat0 == c; fe = m & np.isfinite(t_end) & np.array([w != 'timeout' for w in why])
    print(f"{nm}: env {m.sum()} · 첫 에피소드 종료(낙상) {fe.sum()} · 낙상 시각 분위 {np.round(np.nanpercentile(t_end[fe], [10, 50, 90]), 2) if fe.any() else '-'} · 사유 {dict(zip(*np.unique([why[k] for k in np.where(fe)[0]], return_counts=True))) if fe.any() else {}}")
open("/home/ryu/humanoid_leg_test1/reports/gen_g22r/diag_spont_out.txt", "w").write("\n".join(OUT) + "\n"); os._exit(0)
