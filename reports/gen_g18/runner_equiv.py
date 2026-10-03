"""1003 2단계: 학습 env ↔ 배포 러너(PolicyRunnerStage4) 관측·행동 동치 검사 (G21a 진단 실물 준비).
Isaac env(관측 잡음·지연 끔, 액션 LPF β 고정)를 정책으로 굴리며, 매 틱 env 0 의 원시 입력
(자이로·중력·명령·엔코더 관절각·엔코더 유한차분 속도·접촉 비트)을 러너에 넣어 관측 블록별·정책 출력 최대 오차를 낸다.
러너 계약 인자(--act_beta/--jvel_beta/--jvel_ema_mode/--gait/--pose)는 브리지가 같은 체크포인트에 주는 값을
Kit 밖(시스템 파이썬)에서 rl_bridge_node 로 뽑아 넘긴다 — Kit 안에서 브리지 모듈 import 시 env 생성이 멈췄다(1003)."""
import argparse, sys, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
for k, d in (("--task", None), ("--model", None), ("--out", None), ("--act_beta", "none"), ("--jvel_beta", "none"),
             ("--jvel_ema_mode", "v2"), ("--pose", None)):
    p.add_argument(k, default=d)
p.add_argument("--gait", type=float, default=1.1); p.add_argument("--lpf", type=float, default=1.0)
p.add_argument("--steps", type=int, default=600); p.add_argument("--trace", type=int, default=0); p.add_argument("--sim_restart2", type=int, default=0); p.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import numpy as np, torch, gymnasium as gym
REPO = "/home/ryu/humanoid_leg_test1"; sys.path.insert(0, REPO)
import isaaclab_tasks  # noqa
import rl_walking  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4, JOINT_ORDER
f = lambda s: None if s == "none" else float(s)
pose = json.loads(a.pose)
print(f"[계약] {a.task} act β={a.act_beta} jvel β={a.jvel_beta} mode={a.jvel_ema_mode} gait {a.gait} lpf {a.lpf}", flush=True)
cfg = parse_env_cfg(a.task, num_envs=a.num_envs)
P = cfg.observations.policy
P.enable_corruption = False
for n_ in ("base_ang_vel", "gyro_ema", "joint_vel_ema"):
    t = getattr(P, n_); 
    if "noise" in t.params: t.params["noise"] = 0.0
    if "max_lag" in t.params: t.params["max_lag"] = 0
cfg.actions.joint_pos.lpf_beta_range = (a.lpf, a.lpf)
env = gym.make(a.task, cfg=cfg).unwrapped
om = env.observation_manager
names = list(om.active_terms["policy"]); dims = om.group_obs_term_dim["policy"]
off = {}; o = 0
for n_, d_ in zip(names, dims):
    sz = int(np.prod(d_)); off[n_] = (o, sz); o += sz
jn = list(env.scene["robot"].data.joint_names); jid = [jn.index(j) for j in JOINT_ORDER]
dflt = env.scene["robot"].data.default_joint_pos[0].cpu().numpy().astype(np.float32)
print(f"[env] obs {o} · 관절 순서 env→러너 {jid} · env 기본자세−러너 pose 최대 {np.abs(dflt[jid]-np.asarray(pose)).max():.2e}", flush=True)
pol = torch.jit.load(f"{a.model}/policy.pt", map_location=env.device)
runner = PolicyRunnerStage4(a.model, backend="torch", gait_freq=a.gait, default_pose_rad=pose, jvel_ema_mode=a.jvel_ema_mode,
                            act_obs_ema_beta=f(a.act_beta), jvel_obs_ema_beta=f(a.jvel_beta))
def newest(ob, name):
    a0, sz = off[name]; d = sz // 5
    return ob[a0 + 4 * d: a0 + 5 * d]
obs, _ = env.reset()
mo = {n_: 0.0 for n_ in names}; ma = 0.0; resets = 0; worst = None
for t in range(a.steps):
    o_sim = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
    if t == 0 or int(env.episode_length_buf[0]) == 0:
        runner.reset(); resets += 1
    if a.sim_restart2 and int(env.episode_length_buf[0]) == 1:
        # 시뮬 _ema_state 는 episode_length_buf ≤ 1(두 틱) 에서 재시작, 러너는 첫 틱만 — 기존 차이(1003 발견). 이 규약만 맞춰 나머지를 비교
        for k in ("gyro_ema", "joint_vel_ema"): runner._ema.pop(k, None)
    jp = newest(o_sim, "joint_pos")          # env 순서 상대각
    qpos = (jp + dflt)[jid]
    if hasattr(env, "_enc_vel"):       # v2 계열(G18~): 유한차분 원값 버퍼
        qvel = env._enc_vel[0].detach().cpu().numpy().astype(np.float32)[jid]
    else:                              # 구버전(G12 등): joint_vel 관측(잡음 0) 최신 틀 = 유한차분 원값
        qvel = newest(o_sim, "joint_vel")[jid]
    runner.step(newest(o_sim, "base_ang_vel"), newest(o_sim, "projected_gravity"), newest(o_sim, "velocity_commands"),
                qpos, qvel, newest(o_sim, "feet_contact"))
    o_run = runner.last_obs
    if t < a.trace:
        for n_ in ("actions", "gyro_ema", "joint_vel_ema"):
            print(f"[trace] t{t} ep{int(env.episode_length_buf[0])} {n_:13s} sim {np.round(newest(o_sim, n_)[:3], 4)} run {np.round(newest(o_run, n_)[:3], 4)}", flush=True)
        print(f"[trace] t{t} gyro_raw {np.round(newest(o_sim, 'base_ang_vel'), 4)} act_mgr {np.round(env.action_manager.action[0, :3].cpu().numpy(), 4)} run_last_raw {np.round(runner.last_action[:3], 4)}", flush=True)
    for n_ in names:
        a0, sz = off[n_]; e = float(np.abs(o_sim[a0:a0 + sz] - o_run[a0:a0 + sz]).max())
        if e > mo[n_]: mo[n_] = e
        if e > 1e-3 and worst is None: worst = (t, n_, e)
    with torch.no_grad():
        a_sim = pol(obs["policy"])
    ma = max(ma, float(np.abs(a_sim[0].cpu().numpy() - runner.last_action).max()))  # 둘 다 네트워크 출력 순서
    obs, _, _, _, _ = env.step(a_sim)
with open(a.out, "w") as F:
    F.write(f"{a.task} · {a.model}\n계약 act β={a.act_beta} jvel β={a.jvel_beta} mode={a.jvel_ema_mode} gait {a.gait} lpf {a.lpf}\n")
    F.write(f"sim_restart2={a.sim_restart2} · 틱 {a.steps} · 러너 리셋 {resets} · 첫 1e-3 초과 {worst}\n")
    for n_ in names: F.write(f"관측 {n_:18s} 최대 |sim − runner| = {mo[n_]:.3e}\n")
    F.write(f"정책 출력(raw) 최대 |sim − runner| = {ma:.3e}\n")
print(open(a.out).read(), flush=True)
app.close()
