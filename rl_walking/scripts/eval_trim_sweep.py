"""트림 격자 스윕 — 정적으로 서는 착지 자세를 찾는다 (env당 조합 1개).

**왜 필요한가 (0809 실측).**
곧은 다리(전 관절 0)는 시뮬·실물 모두 **뒤로** 넘어진다(시뮬 −4.55°/s,
실물 −19°/s). 반대로 정책 stand_v8 의 평형 자세를 그대로 **고정**하면
시뮬에서 **앞으로 +26.2°/s** 로 100% 낙상한다 — 그 자세는 정책이 매 틱
능동으로 잡을 때만 성립하기 때문이다.
→ 두 자세 사이 어딘가에 **정적으로 서는 착지 자세**가 있다. 그것을 찾는다.

찾은 자세는 실물에서 hang_stand 트림 3채널로 그대로 재현할 수 있다:
    a(발목) : left_ankle_f = +a,  right_ankle_f = −a
    h(힙)   : left_hip_f   = −h,  right_hip_f   = +h
    k(무릎) : left_knee    = −k,  right_knee    = +k
(하드웨어맵 direction 과 TRIM_CH gain 으로 확정한 변환 — policy_v8 평형
 자세가 a12.5/h6.4/k10.4 로 환산되는 것으로 검증됨.)

한 프로세스에서 env 마다 다른 조합을 돌린다 — 조건당 Isaac 을 띄우면
격자 하나에 몇 시간이 걸린다(0809 실측: 조건 1개 1~2분).

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_trim_sweep.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V8-v0 --steps 400 --out_tag v8env
"""
import argparse
import itertools
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V8-v0")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out_tag", type=str, default="")
parser.add_argument("--a", type=str, default="0,3,6,9,12")
parser.add_argument("--h", type=str, default="0,2,4,6")
parser.add_argument("--k", type=str, default="0,3,6,9")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

A = [float(x) for x in args_cli.a.split(",")]
H = [float(x) for x in args_cli.h.split(",")]
K = [float(x) for x in args_cli.k.split(",")]
COMBOS = list(itertools.product(A, H, K))
args_cli.num_envs = len(COMBOS)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"트림스윕_{args_cli.out_tag}.txt")

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


def trim_to_pose(a, h, k):
    """트림 3채널 → 관절 프레임 목표각 [deg]."""
    return {
        "left_ankle_f_joint": +a, "right_ankle_f_joint": -a,
        "left_hip_f_joint": -h, "right_hip_f_joint": +h,
        "left_knee_joint": -k, "right_knee_joint": +k,
    }


env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = len(COMBOS)
env_cfg.seed = 0
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
# 정적 안정성만 본다 — 외란·초기교란·랜덤화 전부 제거 (0809 하네스 교훈)
env_cfg.events.push_robot = None
for _k in ("roll", "pitch"):
    if _k in env_cfg.events.reset_base.params["pose_range"]:
        env_cfg.events.reset_base.params["pose_range"][_k] = (0.0, 0.0)
if getattr(env_cfg.events, "reset_robot_joints", None):
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
# ⚠ **초기 몸통 속도 랜덤화 차단** (0810 발견).
# `reset_base` 는 pose_range 뿐 아니라 **velocity_range** 를 갖는다:
#   x/y ±0.3 m/s · z ±0.2 · roll/pitch/yaw **±0.3 rad/s (=±17°/s)**
# 순수 역학 식별 시험에 ±17°/s 초기 각속도가 섞이면 −19°/s 낙하 속도
# 측정이 통째로 오염된다. pose_range 만 0 으로 놓고 안심했던 것이 결함이었다.
_vr = env_cfg.events.reset_base.params.get("velocity_range")
if isinstance(_vr, dict):
    for _k in list(_vr):
        _vr[_k] = (0.0, 0.0)
env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
env_cfg.actions.joint_pos.min_delay_steps = 0
env_cfg.actions.joint_pos.max_delay_steps = 0
if getattr(env_cfg.events, "base_com", None):
    env_cfg.events.base_com.params["com_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
env_cfg.episode_length_s = 60.0

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=None)
uenv = env.unwrapped
robot = uenv.scene["robot"]
dev = uenv.device
N = len(COMBOS)

jn = list(robot.data.joint_names)
act = torch.zeros(N, len(jn), device=dev)
for e, (a, h, k) in enumerate(COMBOS):
    pose = trim_to_pose(a, h, k)
    for i, name in enumerate(jn):
        tgt = math.radians(pose.get(name, 0.0))
        act[e, i] = (tgt - DEFAULT_RAD.get(name, 0.0)) / ACTION_SCALE

env.get_observations()
pg_log = []
alive = torch.ones(N, dtype=torch.bool, device=dev)
death = torch.full((N,), args_cli.steps, dtype=torch.long, device=dev)
for i in range(args_cli.steps):
    with torch.inference_mode():
        env.step(act)
    pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
    term = uenv.termination_manager.terminated
    death[alive & term] = i
    alive &= ~term

pg = np.stack(pg_log)
fwd = np.degrees(np.arctan2(pg[:, :, 0], -pg[:, :, 2]))
lat = np.degrees(np.arctan2(pg[:, :, 1], -pg[:, :, 2]))
dt = 1.0 / 50.0
t = np.arange(pg.shape[0]) * dt
d = death.cpu().numpy()
W = min(int(3.0 / dt), pg.shape[0])
rows = []
for e in range(N):
    n = min(W, max(int(d[e]), 2))
    sl = float("nan")
    if n >= 10:
        A_ = np.vstack([t[:n], np.ones(n)]).T
        sl = float(np.linalg.lstsq(A_, fwd[:n, e], rcond=None)[0][0])
    last = int(d[e]) - 1 if d[e] < args_cli.steps else -1
    rows.append({
        "a": COMBOS[e][0], "h": COMBOS[e][1], "k": COMBOS[e][2],
        "fell": bool(d[e] < args_cli.steps),
        "t_fall": (float(d[e]) * dt if d[e] < args_cli.steps else None),
        "slope": sl, "fwd_end": float(fwd[last, e]),
        "lat_end": float(lat[last, e]),
        "fwd_absmax": float(np.abs(fwd[:max(1, int(d[e])), e]).max()),
    })

lines = []
def P(s=""):
    print(s)
    lines.append(s)

ok = [r for r in rows if not r["fell"]]
P("=" * 78)
P(f"트림 격자 스윕 — {args_cli.task}")
P(f"{N} 조합 × {args_cli.steps/50:.1f}s, 정책 없음, 외란·랜덤화 전부 제거")
P(f"a(발목) {A}   h(힙) {H}   k(무릎) {K}")
P("=" * 78)
P(f"■ 끝까지 선 조합: {len(ok)}/{N}")
P("")
if ok:
    ok.sort(key=lambda r: (abs(r["fwd_end"]) + abs(r["lat_end"]), abs(r["slope"])))
    P(f"{'a':>5}{'h':>5}{'k':>5}{'초기추세':>11}{'종료앞뒤':>10}{'종료좌우':>10}{'최대|앞뒤|':>11}")
    P("-" * 60)
    for r in ok[:20]:
        P(f"{r['a']:>5.1f}{r['h']:>5.1f}{r['k']:>5.1f}{r['slope']:>+10.2f}"
          f"{r['fwd_end']:>+9.2f}°{r['lat_end']:>+9.2f}°{r['fwd_absmax']:>10.2f}°")
    P("-" * 60)
    b = ok[0]
    P(f"■ 최적: a={b['a']:.1f} h={b['h']:.1f} k={b['k']:.1f} "
      f"— 종료 앞뒤 {b['fwd_end']:+.2f}° 추세 {b['slope']:+.2f}°/s")
    P("  → 실물 hang_stand 트림에 그대로 투입 가능")
else:
    P("끝까지 선 조합이 없다 — 격자를 넓히거나 모델(CoM)을 먼저 보정해야 한다.")
P("")
P("■ 넘어진 조합의 방향 분포 (부호로 CoM 위치를 읽는다)")
back = [r for r in rows if r["fell"] and r["slope"] < 0]
fore = [r for r in rows if r["fell"] and r["slope"] > 0]
P(f"   뒤로 {len(back)}개 · 앞으로 {len(fore)}개")
if back:
    P(f"   뒤로 넘어진 조합의 평균 a={np.mean([r['a'] for r in back]):.1f} "
      f"h={np.mean([r['h'] for r in back]):.1f} k={np.mean([r['k'] for r in back]):.1f}")
if fore:
    P(f"   앞으로 넘어진 조합의 평균 a={np.mean([r['a'] for r in fore]):.1f} "
      f"h={np.mean([r['h'] for r in fore]):.1f} k={np.mean([r['k'] for r in fore]):.1f}")
P("=" * 78)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
np.savez(os.path.join(REPO, "demo_output", f"trim_sweep_{args_cli.out_tag}.npz"),
         combos=np.array(COMBOS), fwd=fwd, lat=lat, death=d)
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
