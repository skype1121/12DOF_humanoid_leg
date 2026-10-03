"""1004 진단(OSS 라운드 6): 보조 α=2 개루프(G21a)에서 실제 이륙·착지가 반주기 위상의 어디에 오는지, 그리고 sb0_step 추적기의 착지 이벤트(ev_td)가
실제 착지 중 몇 %를 집계하는지. 위상 v ∈ [0,1) (반주기 0.455 s, 23틱; 창: 전환 <0.16, 스윙 <0.50, 착지/하중 <0.80, 정착)."""
import argparse, sys, os
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-StepA0c-v0"); p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--alpha", type=float, default=2.0); p.add_argument("--out", default="diag_phase_events")
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
cmd = U.command_manager.get_term("base_velocity"); N = U.num_envs; fid = mdp_hang._feet_body_ids(U)
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
prev_c = torch.ones(N, 2, dtype=torch.bool, device=dev); air = torch.zeros(N, 2, device=dev)
to_v, td_v, to_sched, td_sched, td_cnt = [], [], [], [], 0; ev_cnt = 0.0
for i in range(int(10.0 / dt)):
    on = i * dt >= 2.0
    cmd.vel_command_b[:, 0] = 0.05 if on else 0.0; cmd.vel_command_b[:, 1:] = 0.0; cmd.is_standing_env[:] = False; cmd.is_heading_env[:] = False
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    c = U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
    t = U.episode_length_buf.to(torch.float32) * dt; u = torch.remainder(1.1 * t, 1.0); half = (u >= 0.5).long(); vv = torch.remainder(2.0 * u, 1.0)
    air = torch.where(~c, air + dt, torch.zeros_like(air))
    if on:
        lifted = prev_c & ~c; landed = (~prev_c) & c
        for e, f in zip(*torch.where(lifted)):
            to_v.append(float(vv[e])); to_sched.append(int(half[e]) == int(f))
        for e, f in zip(*torch.where(landed)):
            td_v.append(float(vv[e])); td_sched.append(int(half[e]) == int(f)); td_cnt += 1
        ev_cnt += float(mdp_hang._sb0(U)["ev_td"].sum())
    prev_c = c
to_v, td_v = np.array(to_v), np.array(td_v); to_s, td_s = np.array(to_sched), np.array(td_sched)
H = lambda x: np.histogram(x, bins=np.linspace(0, 1, 11))[0].tolist()
L = [f"# 위상별 이륙·착지 (α={a.alpha}, G21a, {N} env × 8 s 스텝 모드)",
     f"이륙 {len(to_v)}건 · 예정 스윙발의 이륙 {100 * to_s.mean() if len(to_s) else 0:.0f} % · 위상 히스토그램(0.1 칸): {H(to_v)}",
     f"착지 {len(td_v)}건 · 착지 순간 그 발이 '현재 반주기 예정 스윙발'인 비율 {100 * td_s.mean() if len(td_s) else 0:.0f} % · 위상 히스토그램: {H(td_v)}",
     f"추적기 ev_td 집계 {ev_cnt:.0f}건 = 실제 착지의 {100 * ev_cnt / max(td_cnt, 1):.1f} %",
     "창: 전환 v<0.16 · 스윙 <0.50 · 착지/하중 <0.80 · 정착 ≥0.80. 착지 시 '예정 스윙발 아님' 이면 sb0_step 은 그 착지를 이벤트로 세지 않는다(반주기 경계 넘김)."]
open(f"/home/ryu/humanoid_leg_test1/reports/gen_step/{a.out}.md", "w").write("\n".join(L) + "\n"); os._exit(0)
