"""1004 진단(W2 요 표류 2회): 스텝 모드 보행에서 몸통 요 변화가 어디서 생기는지 분해.
틱마다 몸통 요 Δ 를 (a) 한 발 지지(스윙 중) (b) 양발 지지 로 나눠 누적, 접지한 발의 요 회전(발 링크 요 Δ, 접촉 중)을 '지지발 회전/슬립'으로 누적,
힙 회전 관절(hip_r) 각도 평균. 부호는 env 별 표류 방향(전체 Δ요 부호)에 맞춰 + = 표류 방향으로 정렬."""
import argparse, sys, os
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-W2b-v0"); p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--vx", type=float, default=0.10); p.add_argument("--out", default="diag_yaw_drift")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym  # noqa
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); import isaaclab_tasks, rl_walking  # noqa
from rl_walking import mdp_hang  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa
from rsl_rl.runners import OnPolicyRunner  # noqa
ec = load_cfg_from_registry(a.task, "env_cfg_entry_point"); ac = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point"), a)
ec.scene.num_envs = a.num_envs; ec.curriculum = None; ec.events.disturbance = None
c_ = ec.commands.base_velocity; c_.resampling_time_range = (1e6, 1e6); c_.rel_standing_envs = 1.0
act = ec.actions.joint_pos; act.lpf_beta_range = (0.27, 0.27); act.min_delay_steps = act.max_delay_steps = 1
ec.episode_length_s = 11.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device; dt = U.step_dt
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
cmd = U.command_manager.get_term("base_velocity"); N = U.num_envs; robot = U.scene["robot"]
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U); jn = list(robot.data.joint_names); jr = [jn.index("left_hip_r_joint"), jn.index("right_hip_r_joint")]
yaw_of = lambda q: torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
alive = torch.ones(N, dtype=torch.bool, device=dev); prev_y = None; prev_fy = None
z = lambda: torch.zeros(N, device=dev)
d_single, d_double, d_air, foot_rot, hipr = z(), z(), z(), z(), torch.zeros(N, 2, device=dev); n_hr = 0
wrap = lambda x: torch.atan2(torch.sin(x), torch.cos(x))
for i in range(int(10.0 / dt)):
    on = i * dt >= 2.0
    cmd.vel_command_b[:, 0] = a.vx if on else 0.0; cmd.vel_command_b[:, 1:] = 0.0; cmd.is_standing_env[:] = False; cmd.is_heading_env[:] = False
    with torch.inference_mode():
        obs, _, d, _ = v.step(pol(obs))
    obs = obs[0] if isinstance(obs, tuple) else obs
    alive &= ~d.bool()
    y = yaw_of(robot.data.root_quat_w); fy = yaw_of(robot.data.body_quat_w[:, bid, :])
    c = U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
    if on and prev_y is not None and i * dt >= 3.0:
        dy = wrap(y - prev_y) * alive.float(); nc = c.sum(1)
        d_single += torch.where(nc == 1, dy, torch.zeros_like(dy)); d_double += torch.where(nc == 2, dy, torch.zeros_like(dy)); d_air += torch.where(nc == 0, dy, torch.zeros_like(dy))
        dfy = wrap(fy - prev_fy); foot_rot += (dfy * c.float()).sum(1) * alive.float()
        hipr += torch.rad2deg(robot.data.joint_pos[:, jr]) * alive.float().unsqueeze(1); n_hr += 1
    prev_y, prev_fy = y, fy
tot = d_single + d_double + d_air; sg = torch.sign(tot); sg[sg == 0] = 1.0
al = alive
f = lambda x: float(torch.rad2deg(x * sg)[al].mean())
L = [f"# 요 표류 분해 · {a.task} · vx {a.vx} · ckpt {ck} · 7 s 스텝 모드 · 생존 {int(al.sum())}/{N}",
     f"총 Δ요(표류 방향 +) {f(tot):.2f}° = 한 발 지지 중 {f(d_single):.2f}° + 양발 지지 중 {f(d_double):.2f}° + 양발 공중 {f(d_air):.2f}°",
     f"접지 발 요 회전 합(지지발 회전/슬립, 표류 방향 +) {f(foot_rot):.2f}° · 힙 회전 관절 평균 왼/오 {float((hipr[:, 0] / max(n_hr, 1))[al].mean()):+.2f}/{float((hipr[:, 1] / max(n_hr, 1))[al].mean()):+.2f}°",
     f"표류 방향 분포: 왼 회전 env {int(((tot > 0) & al).sum())} · 오른 회전 env {int(((tot < 0) & al).sum())} · |총| 중앙 {float(torch.rad2deg(tot.abs())[al].median()):.1f}°"]
open(f"/home/ryu/humanoid_leg_test1/reports/gen_step/{a.out}.md", "w").write("\n".join(L) + "\n"); os._exit(0)
