"""CoM 역산 — 실물 낙하(−19°/s)를 재현하는 무게중심 위치를 찾는다.

**real-to-sim 의 핵심 단계.** 0809 실기에서 정책 없이 관절을 kp150 으로 굳히고
손을 떼자 로봇이 **−19°/s 로 뒤로** 무너졌다(−0.9° → −20.0° 를 1.0초에).
같은 조건을 시뮬에서 돌리면:
    base URDF   10.491kg · CoM 발목축 뒤 13.5mm  →  −4.66°/s   (4배 부족)
    payload URDF 12.001kg · CoM 뒤 30.7mm        →  −28.9°/s   (1.5배 과다)
실물은 그 사이다. 이 스크립트가 그 지점을 찾는다.

방법: 질량을 실물(12.0kg)로 고정하고 base 바디의 CoM x 만 env 마다 다르게
뽑아, 곧은 다리 고정 자세의 후방 낙하 속도와의 관계를 적합한다.
정책도 외란도 없는 순수 역학이므로 CoM 이 지배 변수다.

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_com_fit.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V8-v0 --num_envs 64 --out_tag fit
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V8-v0")
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--com_lo", type=float, default=-0.075)
parser.add_argument("--com_hi", type=float, default=-0.010)
parser.add_argument("--target", type=float, default=-19.0,
                    help="맞출 실물 낙하 각속도 [°/s]")
parser.add_argument("--axis", type=str, default="fwd",
                    help="fwd=전후 CoM(피치 낙하) | lat=좌우 CoM(롤 낙하)")
parser.add_argument("--init_pitch", type=float, default=0.0,
                    help="초기 몸통 피치 [°] — 같은 CoM 이 나오는지 보는 대조")
parser.add_argument("--friction", type=float, default=None,
                    help="관절 마찰 고정값 (무시 가정의 민감도)")
parser.add_argument("--armature_scale", type=float, default=1.0,
                    help="반사관성 배율 (감속기 관성 가정의 민감도)")
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
OUT = os.path.join(REPO, "log_picture", f"CoM역산_{args_cli.out_tag}.txt")

#: URDF 실측 (biped12_base.urdf 전개): base+pelvis 병합 3.9856kg,
#: 다리 6.505kg 이고 다리 CoM x ≈ −0.57mm (좌우대칭이라 원점 근처).
M_LEGS, X_LEGS = 6.505, -0.00057
M_BASEBODY_0 = 3.9856
PAYLOAD_KG = 1.51                      # 실물 12.0kg 정합
ACTION_SCALE = 0.25
DEFAULT_RAD = {
    "left_knee_joint": -0.10471975511965977,
    "right_knee_joint": 0.10471975511965977,
    "left_ankle_f_joint": 0.0244, "right_ankle_f_joint": -0.0244,
}

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = 0
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
# 순수 역학만 — 외란·초기교란·게인/지연 랜덤화 전부 제거
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
# 질량은 실물로 고정, CoM x 만 넓게 스윕
env_cfg.events.add_base_mass.params["mass_distribution_params"] = (PAYLOAD_KG, PAYLOAD_KG)
if args_cli.axis == "lat":
    # 좌우 CoM 은 롤 낙하로 잰다 — 좌우대칭이라 명목상 0 이어야 한다.
    env_cfg.events.base_com.params["com_range"] = {
        "x": (-0.0307, -0.0307), "y": (args_cli.com_lo, args_cli.com_hi),
        "z": (0.0041, 0.0041)}
else:
    env_cfg.events.base_com.params["com_range"] = {
        "x": (args_cli.com_lo, args_cli.com_hi), "y": (0.0, 0.0), "z": (0.0041, 0.0041)}
# 초기 기울기 — 역산이 초기조건에 의존하면 그 숫자는 못 믿는다
if args_cli.init_pitch != 0.0:
    _ip = math.radians(args_cli.init_pitch)
    env_cfg.events.reset_base.params["pose_range"]["pitch"] = (_ip, _ip)
# 물성 가정의 민감도 — 무시했던 항을 켜 보고 결과가 몇 mm 흔들리는지 본다
if args_cli.friction is not None:
    env_cfg.events.joint_friction.params[
        "friction_distribution_params"] = (args_cli.friction, args_cli.friction)
if args_cli.armature_scale != 1.0:
    for _a in env_cfg.scene.robot.actuators.values():
        _a.armature = _a.armature * args_cli.armature_scale
env_cfg.episode_length_s = 60.0

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=None)
uenv = env.unwrapped
robot = uenv.scene["robot"]
dev = uenv.device
N = args_cli.num_envs

# env 별 실제 base 바디 CoM 읽기 → 전신 CoM 으로 환산
com_bb = None
try:
    coms = robot.root_physx_view.get_coms().to("cpu").numpy()
    coms = coms.reshape(N, -1, coms.shape[-1])
    com_bb = coms[:, 0, 1 if args_cli.axis == "lat" else 0].copy()
except Exception as exc:
    print(f"[warn] CoM 읽기 실패: {exc}")

m_bb = M_BASEBODY_0 + PAYLOAD_KG
M_TOT = m_bb + M_LEGS
com_body = (None if com_bb is None
            else (m_bb * com_bb + M_LEGS * X_LEGS) / M_TOT)

# 곧은 다리(전 관절 0) 고정
jn = list(robot.data.joint_names)
act = torch.zeros(N, len(jn), device=dev)
for i, name in enumerate(jn):
    act[:, i] = (0.0 - DEFAULT_RAD.get(name, 0.0)) / ACTION_SCALE

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
if args_cli.axis == "lat":
    # 롤 낙하 — 좌우 CoM 을 같은 방식으로 역산
    fwd = np.degrees(np.arctan2(pg[:, :, 1], -pg[:, :, 2]))
dt = 1.0 / 50.0
t = np.arange(pg.shape[0]) * dt
d = death.cpu().numpy()

# 낙하 각속도 — 실물과 같은 방식: −3° 통과 후 −20° 도달까지의 기울기
slopes = np.full(N, np.nan)
for e in range(N):
    y = fwd[:, e]
    idx = np.where(y < -3.0)[0]
    if not len(idx):
        continue
    i0 = idx[0]
    idx2 = np.where(y < -20.0)[0]
    i1 = idx2[0] if len(idx2) else min(len(y) - 1, i0 + 100)
    if i1 - i0 < 5:
        continue
    A = np.vstack([t[i0:i1], np.ones(i1 - i0)]).T
    slopes[e] = np.linalg.lstsq(A, y[i0:i1], rcond=None)[0][0]

lines = []
def P(s=""):
    print(s)
    lines.append(s)

P("=" * 76)
P(f"CoM 역산 — 실물 낙하 {args_cli.target:+.1f}°/s 를 재현하는 무게중심")
P(f"{args_cli.task} · {N} env · 곧은 다리 고정 · 정책/외란 없음")
P(f"질량 {M_TOT:.3f} kg 고정 (실물 정합), base 바디 CoM x 스윕")
P("=" * 76)
ok = ~np.isnan(slopes)
P(f"낙하 발생: {int(ok.sum())}/{N}")
if com_body is not None and ok.sum() >= 5:
    x = com_body[ok] * 1000.0        # mm, 전신 CoM (발목축 전후, − = 뒤)
    y = slopes[ok]
    P(f"전신 CoM x 범위 {x.min():+.1f} ~ {x.max():+.1f} mm")
    P(f"낙하 각속도 범위 {y.min():+.1f} ~ {y.max():+.1f} °/s")
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = float(np.corrcoef(x, y)[0, 1])
    P("")
    P(f"■ 적합: 각속도 = {m:+.3f} × CoM_x[mm] {b:+.2f}   (r = {r:+.3f})")
    if abs(m) > 1e-6:
        xt = (args_cli.target - b) / m
        P("")
        P(f"■ **실물 {args_cli.target:+.1f}°/s 를 내는 전신 CoM x = {xt:+.1f} mm**")
        P(f"   (발목축 기준 {'뒤' if xt < 0 else '앞'}쪽 {abs(xt):.1f} mm)")
        P("")
        P("   비교:  base URDF   −13.5 mm (학습에 사용 중)")
        P("          payload URDF −30.7 mm")
        P(f"   → 내일 실측이 이 값 부근이면 모델 보정 확정.")
    P("")
    P(f"{'CoM_x(mm)':>11}{'낙하°/s':>11}")
    P("-" * 24)
    order = np.argsort(x)
    for k in order[::max(1, len(order) // 14)]:
        P(f"{x[k]:>11.1f}{y[k]:>11.1f}")
else:
    P("표본 부족 — 스윕 범위를 넓히거나 steps 를 늘려야 한다.")
P("=" * 76)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
