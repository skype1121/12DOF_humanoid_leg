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
                    choices=("policy_v8", "policy_v7", "zero", "knee_only", "ankle_only"))
parser.add_argument("--out_tag", type=str, default="")
parser.add_argument("--payload", action="store_true",
                    help="결손 1.51kg 을 payload URDF 대로 배치 (뒤 150·위 100mm)")
parser.add_argument("--flat", action="store_true",
                    help="지형을 평면으로 고정 (경사 타일 제거)")
parser.add_argument("--trace", action="store_true",
                    help="시간별 골반높이·관절추종오차·발접촉력 파형 출력")
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
#: stand_v8 평형 자세 = 실물에서 쓸 '무릎 굽힌 스탠바이' (trim a+12.5/h+6.4/k+10.4)
POSE_POLICY_V8 = {
    "left_hip_f_joint": -7.82, "right_hip_f_joint": 5.02,
    "left_hip_a_joint": 0.02, "right_hip_a_joint": -0.37,
    "left_hip_r_joint": 2.00, "right_hip_r_joint": 0.74,
    "left_knee_joint": -10.24, "right_knee_joint": 10.52,
    "left_ankle_f_joint": 13.67, "right_ankle_f_joint": -11.40,
    "left_ankle_r_joint": 0.34, "right_ankle_r_joint": -0.17,
}
POSE_ZERO = {k: 0.0 for k in POSE_POLICY_V7}
POSES = {
    "policy_v8": POSE_POLICY_V8, "policy_v7": POSE_POLICY_V7, "zero": POSE_ZERO,
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
if args_cli.payload:
    # biped12_payload.urdf 를 이벤트로 재현 — USD 재생성 없이 동일한 질량 분포.
    # payload 1.51kg @ base 프레임 (−150, 0, +100)mm 를 base 바디(=base+pelvis
    # 고정관절 병합, 3.986kg)에 더하면 병합 CoM 이 (−31.7, +0.3, +4.1)mm 이동한다.
    # 결과: 전신 12.001kg · CoM 발목축 뒤 30.7mm (base 모델은 10.491kg · 뒤 13.5mm).
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (1.51, 1.51)
    env_cfg.events.base_com.params["com_range"] = {
        "x": (-0.0317, -0.0317), "y": (0.0003, 0.0003), "z": (0.0041, 0.0041)}

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
    # CoM·질량 랜덤화도 끈다. 이게 남아 있으면 '시뮬이 정적으로 못 선다'가
    # 아니라 'CoM 뽑기에 따라 못 선다'를 재는 것이 된다 — 전혀 다른 명제다.
    if not args_cli.payload and getattr(env_cfg.events, "base_com", None) is not None:
        env_cfg.events.base_com.params["com_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    for _t in ("add_base_mass", "base_mass", "randomize_rigid_body_mass",
               "payload"):
        _e = getattr(env_cfg.events, _t, None)
        if (not args_cli.payload and _e is not None
                and "mass_distribution_params" in getattr(_e, "params", {})):
            _e.params["mass_distribution_params"] = (0.0, 0.0)
    if getattr(env_cfg.events, "physics_material", None) is not None:
        _p = env_cfg.events.physics_material.params
        _p["static_friction_range"] = (0.8, 0.8)
        _p["dynamic_friction_range"] = (0.6, 0.6)
if args_cli.flat:
    # 이 환경들은 최대 19° 경사 타일을 생성한다(stage4_v2_env_cfg _make_v2_terrains_cfg).
    # 수동 로봇은 경사에서 당연히 넘어가므로, '평지 정적 안정성'을 물으려면
    # 지형을 평면으로 못박아야 한다 — 결정론인데 일부만 넘어진 원인이 이것이었다.
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
env_cfg.events.push_robot = None
for k in ("roll", "pitch"):
    if k in env_cfg.events.reset_base.params["pose_range"]:
        env_cfg.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
# 관절 리셋: 이 환경은 reset_joints_by_**offset** 으로 전 관절에 ±0.08rad
# (±4.6°)와 관절속도 ±0.3rad/s 를 준다(env_cfg.py:230). 정적 안정성을 물으려면
# 이 초기 교란을 0으로 만들어야 한다 — 안 그러면 매 env가 다른 교란에서
# 출발해 '정적으로 서는가'가 아니라 '무작위 교란에서 회복하는가'를 재게 된다.
# (offset 규약이므로 0.0 = 교란 없음. scale 규약의 1.0 과 혼동 금지.)
if args_cli.nominal and getattr(env_cfg.events, "reset_robot_joints", None):
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
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

cs = uenv.scene.sensors.get("contact_forces")
feet_ids = None
if cs is not None:
    try:
        feet_ids = [cs.body_names.index(n) for n in
                    ("left_ankle_r_joint", "right_ankle_r_joint")]
    except ValueError:
        feet_ids = None
# 목표 관절각(rad) — 추종 오차를 재려면 명령값이 필요하다
tgt_rad = torch.tensor(
    [math.radians(pose.get(n, 0.0)) for n in jn], device=dev)

obs = env.get_observations()
pg_log, alive = [], torch.ones(N, dtype=torch.bool, device=dev)
death = torch.full((N,), args_cli.steps, dtype=torch.long, device=dev)
trace = []
term_hits = {}
for i in range(args_cli.steps):
    with torch.inference_mode():
        if policy is not None:
            obs, _, _, _ = env.step(policy(obs))
        else:
            env.step(act)
    pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
    if args_cli.trace:
        h = robot.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
        jerr = (robot.data.joint_pos - tgt_rad).abs()
        fz = (cs.data.net_forces_w[:, feet_ids, 2].sum(dim=1)
              if feet_ids is not None else torch.zeros_like(h))
        trace.append([
            float(h.mean()), float(h.min()),
            float(torch.rad2deg(jerr).mean()), float(torch.rad2deg(jerr).max()),
            float(fz.mean()), float(alive.float().mean()),
        ])
    term = uenv.termination_manager.terminated
    # **어느 종료항이 트립했는지** 센다. '낙상 83%'가 진짜 낙상인지
    # 종료 판정 아티팩트인지는 이걸 봐야만 갈린다.
    for _tn in uenv.termination_manager.active_terms:
        if _tn == "time_out":
            continue
        try:
            _d = uenv.termination_manager.get_term(_tn)
        except Exception:
            continue
        term_hits[_tn] = term_hits.get(_tn, 0) + int((alive & _d).sum())
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
P("■ 종료 사유별: " + (", ".join(f"{k} {v}" for k, v in sorted(
    term_hits.items(), key=lambda kv: -kv[1]) if v) or "없음"))
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
if args_cli.trace and trace:
    P("")
    P("  t(s)  골반h(m) 최저h  관절오차° 최대°   발접촉N   생존율")
    for k in range(0, len(trace), max(1, len(trace)//18)):
        r = trace[k]
        P(f"  {k*dt:5.2f} {r[0]:8.3f}{r[1]:7.3f}{r[2]:9.2f}{r[3]:7.2f}"
          f"{r[4]:10.1f}{r[5]*100:8.0f}%")
P("=" * 70)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
