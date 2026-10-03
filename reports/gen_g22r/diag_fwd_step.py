"""1003 진단(대조): 앞 15 N 램프 밀기(평가 sf15 와 같은 모양: 상승 0.5 s·유지 1.0 s)에서 정책 출력에 **스크립트 앞 스텝**을 더하면
낙상이 줄어드는가 — '앞 스텝이 물리·기구학적으로 회복 수단이 되는가'(가능하면 G22R 실패 = 탐색 문제) 를 가른다.
그룹(env 균등): A 정책만 · B 왼발 앞 스텝(힙 +A1) · C 왼발 큰 스텝(힙 +A2) · D 오른발 앞 스텝(힙 +A1).
스텝 궤적(관절 프레임, 정책 출력에 더함, 시작 = 밀기 시작 + t_trig): 0–0.15 s 힙 +A/2·무릎 굽힘 30° → 0.15–0.35 s 힙 +A·무릎 굽힘 10° → 이후 힙 +A 유지.
부호: 왼 힙앞뒤 + = 굽힘(앞), 왼 무릎 − = 굽힘; 오른쪽은 반대 부호(관절 방향 기록 joint-direction-verified)."""
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl"); import cli_args  # noqa
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-G22R-v0"); p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--F", type=float, default=15.0); p.add_argument("--t0", type=float, default=3.0); p.add_argument("--t_trig", type=float, default=0.4)
p.add_argument("--knee", type=float, default=30.0); p.add_argument("--A1", type=float, default=15.0); p.add_argument("--A2", type=float, default=25.0); p.add_argument("--out", default="diag_fwd_step_out.txt")
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
ec.episode_length_s = a.t0 + 7.0
env = gym.make(a.task, cfg=ec); v = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions); U = env.unwrapped; dev = U.device
ck = get_checkpoint_path(os.path.join("/home/ryu/humanoid_leg_test1/logs/rsl_rl", ac.experiment_name), ac.load_run, ac.load_checkpoint)
r = OnPolicyRunner(v, ac.to_dict(), log_dir=None, device=ac.device); r.load(ck); pol = r.get_inference_policy(device=dev)
N = U.num_envs; grp = torch.arange(N, device=dev) % 4
obs = v.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
# 외란 스케줄을 평가 sf15 와 같게 덮어쓴다(슬롯 0 만, 몸통 +x)
hs = mdp_hang._state(U); robot = U.scene["robot"]
q = robot.data.root_quat_w; ex = torch.zeros(N, 3, device=dev); ex[:, 0] = 1.0
fw = mdp_hang._quat_apply(q, ex)[:, :2]; fw = fw / fw.norm(dim=-1, keepdim=True)
hs["dst_kind"][:] = 0.0; hs["dst_F"][:] = 0.0
hs["dst_kind"][:, 0] = 1.0; hs["dst_t0"][:, 0] = a.t0; hs["dst_rise"][:, 0] = 0.5; hs["dst_hold"][:, 0] = 1.0; hs["dst_F"][:, 0] = a.F
hs["dst_dir"][:, 0] = fw
SC = 0.25 * 180.0 / math.pi            # 액션 1 = 14.3°
def script(tr, amp, side):
    """(N,12) 액션 오프셋. side 0 = 왼, 1 = 오른."""
    off = torch.zeros(N, 12, device=dev)
    if tr < 0: return off
    hip = amp / 2 if tr < 0.15 else amp; knee = a.knee if tr < 0.15 else (a.knee / 3 if tr < 0.35 else 0.0)
    sgn = 1.0 if side == 0 else -1.0
    off[:, 0 + side] = sgn * hip / SC; off[:, 6 + side] = -sgn * knee / SC
    return off
fell = torch.zeros(N, dtype=torch.bool, device=dev); t_f = torch.full((N,), float("nan"), device=dev)
lift = torch.zeros(N, 2, device=dev); prev_c = torch.ones(N, 2, device=dev); fid = mdp_hang._feet_body_ids(U)
pre_fell = torch.zeros(N, dtype=torch.bool, device=dev)
steps = int((a.t0 + 6.5) / U.step_dt)
for i in range(steps):
    t = i * U.step_dt; tr = t - (a.t0 + a.t_trig)
    with torch.inference_mode():
        act = pol(obs)
        add = torch.zeros_like(act)
        add[grp == 1] = script(tr, a.A1, 0)[grp == 1]; add[grp == 2] = script(tr, a.A2, 0)[grp == 2]; add[grp == 3] = script(tr, a.A1, 1)[grp == 3]
        obs, _, d, _ = v.step(act + add)
    obs = obs[0] if isinstance(obs, tuple) else obs
    d = d.bool()
    if t < a.t0:
        pre_fell |= d
    else:
        new = d & ~fell & ~pre_fell; t_f[new] = t - a.t0; fell |= d
    c = (U.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0).float()
    if t >= a.t0:
        lift += ((prev_c > 0.5) & (c < 0.5) & ~fell.unsqueeze(1)).float()
    prev_c = c
L = [f"앞 {a.F} N 램프(상승 0.5 s·유지 1.0 s) · 스텝 시작 = 밀기 +{a.t_trig}s · ckpt {ck}"]
for g, nm in enumerate(("A 정책만", f"B 왼발 스텝 힙+{a.A1}°", f"C 왼발 큰 스텝 힙+{a.A2}°", f"D 오른발 스텝 힙+{a.A1}°")):
    m = (grp == g) & ~pre_fell; n = int(m.sum()); nf = int((fell & m).sum())
    tf = t_f[fell & m]
    L.append(f"{nm}: n {n} · 낙상 {nf} ({100.0 * nf / max(n, 1):.1f} %) · 낙상 시각 중앙 {float(tf.median()) if len(tf) else float('nan'):.2f}s · 발 들기 평균 왼/오 {float(lift[m, 0].mean()):.2f}/{float(lift[m, 1].mean()):.2f}")
open(f"/home/ryu/humanoid_leg_test1/reports/gen_g22r/{a.out}", "w").write("\n".join(L) + "\n"); os._exit(0)
