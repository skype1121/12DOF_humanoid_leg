"""1004 W2-9 구조 진단: W2-8 정책(24791)에 회전 보조 α(TURN_ALPHA) 를 고정해 회전시킨 경우(α 2)와 안 시킨 경우(α 0)의
보상 항별 '틱당 실효 기여(가중치·dt 포함)' 비교 — 학습 중 정책이 보조를 상쇄(회전 추종비 0.49→0.32)하는 이유(어느 항이 회전을 벌하나).
전 env 걷기 vx 0.10, wz ±0.3(반반), 2 s 서기 뒤 8 s."""
import argparse, sys, os
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-W2i-v0"); p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--wz", type=float, default=0.3); p.add_argument("--out", default="diag_turn_terms")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
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
cmd = U.command_manager.get_term("base_velocity"); RM = U.reward_manager; names = list(RM.active_terms); N = U.num_envs
robot = U.scene["robot"]
sgn = torch.where(torch.arange(N, device=dev) % 2 == 0, 1.0, -1.0)
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
acc = np.zeros(len(names)); n = 0; alive = torch.ones(N, dtype=torch.bool, device=dev); falls = 0; ratio_s = 0.0; ratio_n = 0
for i in range(int(10.0 / dt)):
    on = i * dt >= 2.0
    cmd.vel_command_b[:, 0] = 0.10 if on else 0.0; cmd.vel_command_b[:, 1] = 0.0; cmd.vel_command_b[:, 2] = (a.wz * sgn) if on else 0.0
    cmd.is_standing_env[:] = False; cmd.is_heading_env[:] = False
    with torch.inference_mode():
        obs, rew, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    if not on or i * dt < 3.0:
        continue
    step_r = RM._step_reward.detach().cpu().numpy() * dt; al = alive.cpu().numpy()
    acc += step_r[al].sum(0); n += int(al.sum())
    wr = robot.data.root_ang_vel_w[:, 2]; ratio_s += float((wr * sgn / a.wz)[alive].sum()); ratio_n += int(alive.sum())
    falls += int((alive & d.bool()).sum()); alive &= ~d.bool()
alpha = os.environ.get("TURN_ALPHA", "?")
L = [f"# α={alpha} · {a.task} · {ck} · wz ±{a.wz} · vx 0.10 · env {N} · 낙상 {falls} · 회전 추종비 {ratio_s / max(ratio_n, 1):.3f}"]
L += [f"{nm}\t{acc[j] / max(n, 1):+.6f}" for j, nm in enumerate(names)]
open(f"/home/ryu/humanoid_leg_test1/reports/gen_step/{a.out}_a{alpha}.tsv", "w").write("\n".join(L) + "\n"); os._exit(0)
