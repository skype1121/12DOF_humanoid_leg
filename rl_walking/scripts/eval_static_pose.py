"""고정 자세 정적 안정성 — 관절을 묶어놓고 시뮬 로봇이 서 있는지 본다.

posture_probe.py 의 판별력을 담보하기 위한 사전 검증. 실물에서 "정책의 평형
자세에 고정했더니 넘어졌다"가 의미를 가지려면, **같은 고정 자세가 시뮬에서는
선다**는 것이 먼저 참이어야 한다. 정책이 능동으로 균형을 잡아 안 넘어진 것과,
관절을 고정해도 서는 것은 다른 명제다.

정책을 쓰지 않고 상수 액션으로 목표 관절각을 유지한다:
    target = default + ACTION_SCALE × action   →   action = (pose − default)/0.25

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_static_pose.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V7-v0 --num_envs 64 --steps 250 \
      --pose policy_v7 --out_tag v7DR
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V7-v0")
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--pose", default="policy_v7",
                    choices=("policy_v7", "zero", "knee_only", "ankle_only"))
parser.add_argument("--out_tag", type=str, default="")
parser.add_argument("--nominal", action="store_true",
                    help="게인 정격·지연0 — 실물 stage8 홀드에 가까운 조건")
parser.add_argument("--policy", action="store_true",
                    help="대조군: 고정 액션 대신 실제 정책을 돌린다. 같은 무외란 "
                         "설정에서 정책이 서면 장치가 건전하다는 증거가 된다.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym          # noqa: E402
import math                      # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"정적자세_{args_cli.out_tag}.txt")

#: posture_probe.py 와 동일한 자세표 (deg, 관절 프레임)
POSE_POLICY_V7 = {
    "left_hip_f_joint": -6.1, "right_hip_f_joint": -4.8,
    "left_hip_a_joint": -0.8, "right_hip_a_joint": 2.1,
    "left_hip_r_joint": -1.3, "right_hip_r_joint": -2.1,
    "left_knee_joint": -10.6, "right_knee_joint": 11.0,
    "left_ankle_f_joint": 12.6, "right_ankle_f_joint": -2.0,
    "left_ankle_r_joint": -0.4, "right_ankle_r_joint": -2.0,
}
POSE_ZERO = {k: 0.0 for k in POSE_POLICY_V7}
POSES = {
    "policy_v7": POSE_POLICY_V7, "zero": POSE_ZERO,
    "knee_only": dict(POSE_ZERO, **{"left_knee_joint": -10.6,
                                    "right_knee_joint": 11.0}),
    "ankle_only": dict(POSE_ZERO, **{"left_ankle_f_joint": 12.6,
                                     "right_ankle_f_joint": -2.0}),
}
DEFAULT_RAD = {
    "left_hip_f_joint": 0.0, "right_hip_f_joint": 0.0,
    "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0,
    "left_knee_joint": -0.10471975511965977,
    "right_knee_joint": 0.10471975511965977,
    "left_ankle_f_joint": 0.0244, "right_ankle_f_joint": -0.0244,
    "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0,
}
ACTION_SCALE = 0.25

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = 0
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
# **정적 안정성만 본다.** 이 환경들은 3~8초마다 랜덤 푸시를 걸고 리셋 시 몸통을
# ±0.07rad(±4°) 기울여 스폰한다. 그대로 두면 "수동 상태로 외란을 버티는가"를
# 재게 되어 정적 안정성 질문에 답하지 못한다 (0808 실측: 이 함정에 걸렸다).
if args_cli.nominal:
    # 실물 stage8 홀드에 가깝게: 게인 정격(kp150/kd5) · 액션 지연 0 · 마찰 고정.
    # 실물은 이 조건에서 고정 관절만으로 몇 분씩 선다.
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    env_cfg.actions.joint_pos.min_delay_steps = 0
    env_cfg.actions.joint_pos.max_delay_steps = 0
    if getattr(env_cfg.events, "joint_friction", None) is not None:
        env_cfg.events.joint_friction.params[
            "friction_distribution_params"] = (0.0, 0.0)
env_cfg.events.push_robot = None
for k in ("roll", "pitch"):
    if k in env_cfg.events.reset_base.params["pose_range"]:
        env_cfg.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
# 관절 리셋은 **건드리지 않는다**. position_range 의 의미가 term 마다 다르고
# (scale 이면 1.0=기본, offset 이면 1.0=+57°), 오해해서 (1.0,1.0)을 주면 전
# 관절이 1rad 어긋난 채 스폰해 즉시 붕괴한다 — 0808 실측으로 대조군이
# 100% 낙상해 잡았다.
env_cfg.episode_length_s = 60.0

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=None)
uenv = env.unwrapped
robot = uenv.scene["robot"]
dev = uenv.device
N = args_cli.num_envs

# 정책 액션 순서 = robot.data.joint_names (Isaac DOF 순)
jn = list(robot.data.joint_names)
pose = POSES[args_cli.pose]
act = torch.zeros(N, len(jn), device=dev)
for i, name in enumerate(jn):
    tgt = math.radians(pose.get(name, 0.0))
    act[:, i] = (tgt - DEFAULT_RAD.get(name, 0.0)) / ACTION_SCALE

policy = None
if args_cli.policy:
    from isaaclab_tasks.utils import get_checkpoint_path
    from rsl_rl.runners import OnPolicyRunner
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl",
                                            agent_cfg.experiment_name))
    ckpt = get_checkpoint_path(log_root, agent_cfg.load_run,
                               agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=dev)
    print(f"[대조군] 정책 사용: {ckpt}")

obs = env.get_observations()
pg_log, alive = [], torch.ones(N, dtype=torch.bool, device=dev)
death = torch.full((N,), args_cli.steps, dtype=torch.long, device=dev)
for i in range(args_cli.steps):
    with torch.inference_mode():
        if policy is not None:
            obs, _, _, _ = env.step(policy(obs))
        else:
            env.step(act)
    pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
    term = uenv.termination_manager.terminated
    death[alive & term] = i
    alive &= ~term

pg = np.stack(pg_log)
fwd = np.degrees(np.arctan2(pg[:, :, 0], -pg[:, :, 2]))
dt = 1.0 / 50.0
t = np.arange(pg.shape[0]) * dt
dead = death.cpu().numpy()
W = min(int(3.0 / dt), pg.shape[0])
slopes = np.full(N, np.nan)
for e in range(N):
    n = min(W, max(dead[e], 2))
    if n < 10:
        continue
    A = np.vstack([t[:n], np.ones(n)]).T
    slopes[e] = np.linalg.lstsq(A, fwd[:n, e], rcond=None)[0][0]

fell = int((dead < args_cli.steps).sum())
lines = []
def P(s=""):
    print(s)
    lines.append(s)

P("=" * 70)
P(f"고정 자세 정적 안정성 — {args_cli.pose}")
P(f"태스크: {args_cli.task}   {N} env × {args_cli.steps/50:.1f}s   정책 없음")
P("=" * 70)
P("자세(관절 프레임 deg): " + ", ".join(
    f"{k.replace('_joint','')} {v:+.1f}" for k, v in sorted(pose.items())
    if abs(v) > 0.05) or "(전 관절 0)")
P("")
P(f"■ 낙상: {fell}/{N} ({100.0*fell/N:.1f}%)")
P(f"■ 초기 3s 전방 추세: 평균 {np.nanmean(slopes):+.2f}°/s, "
  f"최대 {np.nanmax(slopes):+.2f}°/s")
P(f"■ 종료 시 전방 기울임: 평균 {fwd[-1].mean():+.1f}°, "
  f"최대 {fwd[-1].max():+.1f}°, 최소 {fwd[-1].min():+.1f}°")
P("")
if fell == 0:
    P("→ 이 고정 자세는 시뮬에서 **정적으로 안정**하다.")
    P("   실물이 같은 자세에서 넘어지면 계통 오차의 직접 증거가 된다.")
else:
    P("→ 이 고정 자세는 시뮬에서도 서지 못한다(정책의 능동 균형이 필요).")
    P("   실물 고정 자세 시험은 판별력이 없다 — 실험 설계를 바꿔야 한다.")
P("=" * 70)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
