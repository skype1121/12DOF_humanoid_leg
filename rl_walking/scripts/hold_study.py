"""착지 완료 상태 홀드 연구 (0918 밤, 승윤님 조건 5번): 기하 비교 → 지면 스폰 kp150/kd4 홀드, 변수 하나씩.

  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/hold_study.py --headless \
      --mode kinematic --pose_json <실측 엔코더각 JSON> --tag geo_enc            # 관절을 실측각에 '고정'(PD 없음) → 발평평 기하·CoM·렌더
      --mode hold --pose_json <v5 명령 JSON> --kd 4 --num_envs 32 --seconds 10 --tag hold_base   # 지면 스폰 홀드
        [--knee_bias -16 --hip_bias 0 --ankle_bias 0]   물리 목표 = 명령 + 편향 (엔코더 뒤 처짐 가설; '엔코더' = 물리 − 편향)
        [--zero_offset hip:2,knee:0,ankle:0]           명령에 더하는 영점 오프셋(관절 프레임 deg, 왼쪽 기준·오른쪽 부호 반전)
        [--ground_k 2000 --ground_d 40]                 지면 연성(PhysX compliant contact); 0 = 강체
        [--joint_friction 0.0] [--joint_damping 0.0]    관절 마찰 계수 / 관절 점성감쇠(모든 관절)
        [--drop_mm 5] [--pitch_jitter_deg 1.0]          스폰 초기조건: 발바닥-지면 간격·초기 피치 균등 잡음
출력: log_picture/_hold/<tag>.txt (전 env 집계: 낙상률·낙상 시각·몸통각 시계열·관절 추종오차·접촉), <tag>_envs.csv, 렌더 png(kinematic).
원칙: 생존 env 만의 평균으로 정합을 말하지 않는다 — 전 env 통계와 생존/낙상 분리 통계를 모두 적는다.
"""
import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: F401  # isort: skip

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Stand-V14-v0")
p.add_argument("--mode", choices=("kinematic", "hold"), default="hold")
p.add_argument("--pose_json", required=True)
p.add_argument("--tag", default="hold")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--kd", type=float, default=4.0)
p.add_argument("--kp", type=float, default=150.0)
p.add_argument("--knee_bias", type=float, default=0.0, help="왼무릎 기준 deg (음수 = 더 굽음); 오른쪽 부호 반전")
p.add_argument("--hip_bias", type=float, default=0.0)
p.add_argument("--ankle_bias", type=float, default=0.0)
p.add_argument("--zero_offset", type=str, default="", help="hip:a,knee:b,ankle:c [deg, 왼쪽 기준] 명령에 가산")
p.add_argument("--ground_k", type=float, default=0.0)
p.add_argument("--ground_d", type=float, default=0.0)
p.add_argument("--joint_friction", type=float, default=0.0)
p.add_argument("--joint_damping", type=float, default=0.0)
p.add_argument("--com_x", type=float, default=0.0, help="base 링크 CoM 전후 이동 [m] (+앞; base_com 이벤트 고정값) — URDF CoM 오차 가설 시험용")
p.add_argument("--ankle_kp_scale", type=float, default=1.0, help="발목앞뒤 관절만 kp 배율 (0919 실측: 발목 전동부 8° 유격·마찰 대역 가설 — 대역 안 저강성 근사)")
p.add_argument("--ankle_friction", type=float, default=0.0, help="발목앞뒤 관절만 PhysX 관절 마찰 계수 (히스테리시스 '버팀' 근사)")
p.add_argument("--knee_kp_scale", type=float, default=1.0, help="무릎 관절만 kp 배율 (직렬 스프링 근사: 유효 강성 kp·k/(kp+k))")
p.add_argument("--ankle_play_deg", type=float, default=0.0, help="발목앞뒤 관절 기계 유격 반폭 [deg] — DeadzoneImplicit(유격 안 강성 0, kd 유지). 0919 실측 발목 대역 ~8° → 반폭 4")
p.add_argument("--knee_play_deg", type=float, default=0.0, help="무릎 관절 기계 유격 반폭 [deg]")
p.add_argument("--ankle_coulomb", type=float, default=0.0, help="발목앞뒤 유격 안 쿨롱 마찰 토크 [Nm] (DeadzoneCoulombImplicit; --ankle_play_deg 와 함께)")
p.add_argument("--push_seq", type=str, default="", help="밀기 에뮬레이션 'b:2.0:80:0.3,f:5.0:80:0.3' = (방향 b/f, 시작 s, 힘 N, 지속 s) base 링크에 수평력")
p.add_argument("--drop_mm", type=float, default=5.0)
p.add_argument("--pitch_jitter_deg", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--render", action="store_true", help="hold 모드에서도 정착 시점 렌더")
p.add_argument("--out_dir", default="/home/ryu/humanoid_leg_test1/log_picture/_hold")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import gymnasium as gym                 # noqa: E402
from PIL import Image                   # noqa: E402
import isaaclab_tasks                   # noqa: F401,E402
sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking                       # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
import isaaclab.utils.math as mu        # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/rl_walking/scripts")
import pose_geom as pg_fk               # noqa: E402  URDF FK (발평평 피치)

DEFAULT_RAD = {"left_knee_joint": -0.10471975511965977, "right_knee_joint": 0.10471975511965977,
               "left_ankle_f_joint": 0.0244, "right_ankle_f_joint": -0.0244}
ACTION_SCALE = 0.25
SAG = ("hip_f", "knee", "ankle_f")

raw = json.load(open(a.pose_json, encoding="utf-8"))
pose_cmd = {k: float(v) for k, v in raw.get("trims_joint_frame_deg", raw).items()}


def side_sign(n):
    return 1.0 if n.startswith("left") else -1.0


def zero_off(n):
    if not a.zero_offset:
        return 0.0
    d = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in a.zero_offset.split(",")}
    for key, val in d.items():
        if key in n:
            return side_sign(n) * val
    return 0.0


def bias(n):
    if "knee" in n:
        return side_sign(n) * a.knee_bias
    if "hip_f" in n:
        return side_sign(n) * a.hip_bias
    if "ankle_f" in n:
        return side_sign(n) * a.ankle_bias
    return 0.0


# 물리 목표(deg) = 명령 + 영점 오프셋 + 처짐 편향
phys_deg = {n: pose_cmd.get(n, 0.0) + zero_off(n) + bias(n) for n in pose_cmd}

ec = load_cfg_from_registry(a.task, "env_cfg_entry_point")
ec.scene.num_envs = a.num_envs
ec.seed = a.seed
ec.episode_length_s = 600.0
ec.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
ec.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
ec.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
for k in ("roll", "pitch", "yaw", "x", "y", "z"):
    if k in ec.events.reset_base.params["pose_range"]:
        ec.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
if "velocity_range" in ec.events.reset_base.params:
    for k in list(ec.events.reset_base.params["velocity_range"]):
        ec.events.reset_base.params["velocity_range"][k] = (0.0, 0.0)
ec.events.push_robot = None
ec.events.actuator_gains.params["stiffness_distribution_params"] = (a.kp / 150.0, a.kp / 150.0)
ec.events.actuator_gains.params["damping_distribution_params"] = (a.kd / 5.0, a.kd / 5.0)
ec.actions.joint_pos.min_delay_steps = 0
ec.actions.joint_pos.max_delay_steps = 0
if getattr(ec.events, "joint_friction", None) is not None:
    ec.events.joint_friction.params["friction_distribution_params"] = (a.joint_friction, a.joint_friction)
if getattr(ec.events, "base_com", None) is not None:
    ec.events.base_com.params["com_range"] = {"x": (a.com_x, a.com_x), "y": (0.0, 0.0), "z": (0.0, 0.0)}
for _t in ("add_base_mass", "base_mass", "randomize_rigid_body_mass", "payload"):
    _e = getattr(ec.events, _t, None)
    if _e is not None and "mass_distribution_params" in getattr(_e, "params", {}):
        _e.params["mass_distribution_params"] = (0.0, 0.0)
if getattr(ec.events, "physics_material", None) is not None:
    ec.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
    ec.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
if getattr(ec.events, "reset_robot_joints", None):
    ec.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    ec.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
ec.scene.terrain.terrain_type = "plane"
ec.scene.terrain.terrain_generator = None
if a.ground_k > 0:
    ec.scene.terrain.physics_material.compliant_contact_stiffness = a.ground_k
    ec.scene.terrain.physics_material.compliant_contact_damping = a.ground_d
ec.viewer.resolution = (1280, 960)
ec.viewer.eye = (1.5, 0.0, 0.5)
ec.viewer.lookat = (0.0, 0.0, 0.45)

if a.ankle_play_deg > 0 or a.knee_play_deg > 0:
    # 유격 관절만 DeadzoneImplicit 로 분리 (기존 ak70 그룹에서 해당 관절 제외)
    from rl_walking.actuators_play import DeadzoneImplicitActuatorCfg
    import re as _re
    acts = ec.scene.robot.actuators
    base = acts["ak70"]
    keep = []
    for ex in (base.joint_names_expr if isinstance(base.joint_names_expr, (list, tuple)) else [base.joint_names_expr]):
        if a.ankle_play_deg > 0 and _re.fullmatch(ex, "left_ankle_f_joint"): continue
        if a.knee_play_deg > 0 and _re.fullmatch(ex, "left_knee_joint"): continue
        keep.append(ex)
    acts["ak70"] = ImplicitActuatorCfg(joint_names_expr=keep, effort_limit_sim=base.effort_limit_sim, velocity_limit_sim=base.velocity_limit_sim,
                                       stiffness=base.stiffness, damping=base.damping, armature=base.armature)
    from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg
    for nm, sub, play in (("ankle_f_play", ".*_ankle_f_joint", a.ankle_play_deg), ("knee_play", ".*_knee_joint", a.knee_play_deg)):
        if play > 0:
            if nm == "ankle_f_play" and a.ankle_coulomb > 0:
                acts[nm] = DeadzoneCoulombImplicitActuatorCfg(joint_names_expr=[sub], effort_limit_sim=base.effort_limit_sim, velocity_limit_sim=base.velocity_limit_sim,
                                                              stiffness=base.stiffness, damping=base.damping, armature=base.armature, play_rad=math.radians(play), coulomb_nm=a.ankle_coulomb)
            else:
                acts[nm] = DeadzoneImplicitActuatorCfg(joint_names_expr=[sub], effort_limit_sim=base.effort_limit_sim, velocity_limit_sim=base.velocity_limit_sim,
                                                       stiffness=base.stiffness, damping=base.damping, armature=base.armature, play_rad=math.radians(play))
env = gym.make(a.task, cfg=ec, render_mode="rgb_array")
u = env.unwrapped
robot = u.scene["robot"]
dev = u.device
N = a.num_envs
jn = list(robot.data.joint_names)
dt_ctrl = u.step_dt
steps = int(round(a.seconds / dt_ctrl))
os.makedirs(a.out_dir, exist_ok=True)

phys_rad = torch.tensor([math.radians(phys_deg.get(n, 0.0)) for n in jn], device=dev)
act = torch.zeros(N, len(jn), device=dev)
for i, n in enumerate(jn):
    act[:, i] = (float(phys_rad[i]) - DEFAULT_RAD.get(n, 0.0)) / ACTION_SCALE

# URDF FK: 발 평평일 때 몸통 피치(+앞) · 발 링크 원점 대비 base 높이
links_u, joints_u = pg_fk.load("/home/ryu/humanoid_leg_test1/rl_walking/assets/biped12_final3.urdf")
w = pg_fk.fk(links_u, joints_u, phys_deg)
foot_links = [j["child"] for j in joints_u if "ankle_r_joint" in j["name"]]
rf, _ = w[foot_links[0]]
up = rf[:, 1]
phi_flat = math.atan2(-up[0], up[2])          # 발바닥을 수평으로 만드는 y축 회전 = 몸통 피치(+앞)
R = pg_fk.rot_axis(np.array([0, 1, 0]), phi_flat)
base_over_foot = -float(np.mean([(R @ w[f][1])[2] for f in foot_links]))   # base 원점 − 발 링크 원점 (수직)

cs = u.scene.sensors.get("contact_forces")
bn = list(robot.body_names)
feet_idx = [k for k, n in enumerate(bn) if "ankle_r" in n]
feet_cs = None
if cs is not None:
    try:
        feet_cs = [cs.body_names.index(n) for n in ("left_ankle_r_joint", "right_ankle_r_joint")]
    except ValueError:
        feet_cs = None


def pitch_roll_deg():
    pgv = robot.data.projected_gravity_b
    return (torch.rad2deg(torch.atan2(pgv[:, 0], -pgv[:, 2])), torch.rad2deg(torch.atan2(pgv[:, 1], -pgv[:, 2])))


def foot_pitch_deg():
    """발 링크 프레임의 '위' 축(영자세에서 base +Z)이 월드 Z 와 이루는 전후 기울기 [deg] (0 = 평평)."""
    q = robot.data.body_quat_w[:, feet_idx[0]]
    upv = mu.quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3))   # 발 링크 프레임 +Y = 영자세에서 위(pose_geom 과 동일)
    fwd = torch.tensor([0.0, -1.0, 0.0], device=dev)                                   # 학습 월드 전방 = −Y
    return torch.rad2deg(torch.atan2((upv * fwd).sum(1), upv[:, 2]))                   # + = 발끝 아래(앞으로 기움)


def snap(name, eye, lookat=(0.0, 0.0, 0.42)):
    u.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
    for _ in range(3):
        u.render()
    img = u.render()
    Image.fromarray(np.asarray(img)).save(os.path.join(a.out_dir, f"{a.tag}_{name}.png"))


def place_standing(drop_m, jitter_deg):
    """관절을 물리 목표에 놓고, 발이 평평하게 닿도록 base 자세·높이를 둔다 (초기 속도 0)."""
    q = phys_rad.unsqueeze(0).expand(N, -1).clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    root = robot.data.default_root_state.clone()
    root[:, 7:] = 0.0
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    jit = (torch.rand(N, generator=g) * 2 - 1) * math.radians(jitter_deg)
    pitch = torch.full((N,), phi_flat) + jit
    qp = mu.quat_from_euler_xyz(torch.zeros(N), pitch, torch.zeros(N)).to(dev)
    root[:, 3:7] = mu.quat_mul(root[:, 3:7], qp)
    # 높이: 발 링크 원점이 지면 위 sole_h 에 오도록. sole_h(발 링크 원점~바닥) 은 스폰 뒤 실측으로 보정한다.
    root[:, 2] = u.scene.env_origins[:, 2] + base_over_foot + SOLE_H + drop_m
    root[:, :2] = u.scene.env_origins[:, :2]
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_data_to_sim()


SOLE_H = 0.0
env.reset()
# 관절별 강성/마찰 덮어쓰기 (리셋 이벤트 뒤)
def _ids(sub):
    return [k for k, n in enumerate(jn) if sub in n]
with torch.inference_mode():
    if a.ankle_kp_scale != 1.0:
        ids = _ids("ankle_f_joint"); robot.write_joint_stiffness_to_sim(torch.full((N, len(ids)), a.kp * a.ankle_kp_scale, device=dev), joint_ids=ids)
    if a.knee_kp_scale != 1.0:
        ids = _ids("knee_joint"); robot.write_joint_stiffness_to_sim(torch.full((N, len(ids)), a.kp * a.knee_kp_scale, device=dev), joint_ids=ids)
    if a.ankle_friction > 0:
        ids = _ids("ankle_f_joint"); robot.write_joint_friction_coefficient_to_sim(torch.full((N, len(ids)), a.ankle_friction, device=dev), joint_ids=ids)
# 발 링크 원점 ~ 바닥 거리 실측: 기본 스폰(강체 접촉 정착) 뒤 발 링크 z
with torch.inference_mode():
    for _ in range(25):
        env.step(act)
    fz0 = robot.data.body_pos_w[:, feet_idx, 2].mean().item() - u.scene.env_origins[:, 2].mean().item()
SOLE_H = max(0.0, fz0)

rows = []
with torch.inference_mode():
    if a.mode == "kinematic":
        place_standing(0.0, 0.0)
        for i in range(30):
            place_standing(0.0, 0.0)
            env.step(act)
        pt, rl = pitch_roll_deg()
        fp = foot_pitch_deg()
        masses = robot.root_physx_view.get_masses().to(dev)[0]
        com_w = (robot.data.body_com_pos_w[0] * masses[:, None]).sum(0) / masses.sum()
        foot_p = robot.data.body_pos_w[0, feet_idx].mean(0)
        fwd_axis = torch.tensor([0.0, -1.0, 0.0], device=dev)
        com_ahead = float(((com_w - foot_p) * fwd_axis).sum() * 100)
        lines = [f"[kinematic] pose_json {a.pose_json} 물리각(deg): " + " ".join(f"{n.replace('_joint','')}{phys_deg.get(n,0):+.1f}" for n in jn if abs(phys_deg.get(n, 0)) > 0.05),
                 f"URDF FK 발평평 몸통피치 {math.degrees(phi_flat):+.2f}° | 시뮬 관절 고정 뒤 몸통 pitch {float(pt[0]):+.2f}° roll {float(rl[0]):+.2f}° | 발 기울기 {float(fp[0]):+.2f}°",
                 f"전신 CoM 발목축 대비 앞 {com_ahead:+.2f} cm (URDF 질량 17.62 kg 기준)"]
        snap("side", (0.0, -1.6, 0.45)); snap("front", (1.6, 0.0, 0.45)); snap("iso", (1.1, -1.1, 0.7))
        open(os.path.join(a.out_dir, f"{a.tag}.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    else:
        place_standing(a.drop_mm / 1000.0, a.pitch_jitter_deg)
        pt, rl = pitch_roll_deg(); fp = foot_pitch_deg()
        h0 = (robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2])
        print(f"[spawn] 몸통 pitch {float(pt.mean()):+.2f}° (설계 {math.degrees(phi_flat):+.2f}) 발 기울기 {float(fp.mean()):+.2f}° base z {float(h0.mean()):.3f} sole_h {SOLE_H:.4f}")
        fell = torch.full((N,), -1.0, device=dev)
        pitch_hist = []; err_hist = []; contact_hist = []; fz_hist = []
        sample_t = sorted(set(int(round(t / dt_ctrl)) for t in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, a.seconds] + [x * 0.5 for x in range(int(a.seconds * 2))]))
        pushes = []
        for tok in [t for t in a.push_seq.split(",") if t.strip()]:
            d, t0, fN, dur = tok.split(":"); pushes.append((d.strip(), float(t0), float(fN), float(dur)))
        base_id = bn.index("base") if "base" in bn else 0
        fwd_axis_w = torch.tensor([0.0, -1.0, 0.0], device=dev)     # 학습 월드 전방 = −Y
        for i in range(steps):
            tnow = i * dt_ctrl; fvec = torch.zeros(N, 1, 3, device=dev)
            for d, t0, fN, dur in pushes:
                if t0 <= tnow < t0 + dur:
                    fvec[:, 0, :] += (fwd_axis_w if d == "f" else -fwd_axis_w) * fN
            if pushes:
                robot.set_external_force_and_torque(fvec, torch.zeros_like(fvec), body_ids=[base_id])
            env.step(act)
            pt, rl = pitch_roll_deg()
            h = robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
            down = (pt.abs() > 40) | (rl.abs() > 40) | (h < 0.30)
            newly = down & (fell < 0)
            fell[newly] = i * dt_ctrl
            if i in sample_t or i == steps - 1:
                q = robot.data.joint_pos
                err = torch.rad2deg(phys_rad.unsqueeze(0) - q)          # 물리 목표 − 실제 (= 실물 '명령−엔코더'와 같은 부호, 편향 제외)
                if feet_cs is not None:
                    f = cs.data.net_forces_w[:, feet_cs, 2].clamp(min=0)
                    both = ((f > 5).sum(1) == 2).float()
                else:
                    f = torch.zeros(N, 2, device=dev); both = torch.zeros(N, device=dev)
                pitch_hist.append((i * dt_ctrl, pt.clone(), rl.clone())); err_hist.append((i * dt_ctrl, err.clone())); contact_hist.append((i * dt_ctrl, both.clone())); fz_hist.append((i * dt_ctrl, f.clone()))
            if a.render and i == int(round(2.0 / dt_ctrl)):
                snap("t2s_side", (0.0, -1.6, 0.45)); snap("t2s_iso", (1.1, -1.1, 0.7))
        alive = fell < 0
        n_fall = int((~alive).sum())
        lines = [f"[hold] pose_json {a.pose_json} kp {a.kp} kd {a.kd} | 편향 무릎 {a.knee_bias:+.1f} 힙 {a.hip_bias:+.1f} 발목 {a.ankle_bias:+.1f} | 영점오프셋 '{a.zero_offset}' | 지면 k {a.ground_k:g}/d {a.ground_d:g} | 관절마찰 {a.joint_friction:g} 감쇠 {a.joint_damping:g} | com_x {a.com_x:+.3f} m | 발목 kp×{a.ankle_kp_scale:g} 마찰 {a.ankle_friction:g} 유격 {a.ankle_play_deg:g}° 쿨롱 {a.ankle_coulomb:g}Nm 무릎 kp×{a.knee_kp_scale:g} 유격 {a.knee_play_deg:g}° | push '{a.push_seq}' | drop {a.drop_mm:g} mm 피치잡음 ±{a.pitch_jitter_deg:g}° | env {N} × {a.seconds:g} s",
                 "명령(deg): " + " ".join(f"{n.replace('_joint','')}{pose_cmd.get(n,0):+.1f}" for n in jn if abs(pose_cmd.get(n, 0)) > 0.05),
                 f"스폰: 몸통 pitch 설계 {math.degrees(phi_flat):+.2f}° 실측 {float(pitch_hist[0][1].mean()):+.2f}° (t=0.1 s) | 발 기울기(t=0) {float(fp.mean()):+.2f}° | 발 원점 높이 {SOLE_H:.4f} m",
                 f"낙상: {n_fall}/{N} ({100*n_fall/N:.1f} %)" + (f"  낙상 시각 중앙 {float(fell[~alive].median()):.2f} s, 최소 {float(fell[~alive].min()):.2f}, 최대 {float(fell[~alive].max()):.2f} s" if n_fall else ""),
                 "몸통 피치 시계열 [deg] (전 env 평균±표준편차 | 생존 env 평균 | 낙상 env 평균):"]
        for (t, pt_, rl_), (_, both_) in zip(pitch_hist, contact_hist):
            al = fell < 0
            al_t = (fell < 0) | (fell > t)   # 시각 t 에 아직 서 있던 env
            s_all = f"{float(pt_.mean()):+6.2f}±{float(pt_.std()):4.2f}"
            s_alive = f"{float(pt_[al].mean()):+6.2f}" if al.any() else "  n/a "
            s_fall = f"{float(pt_[~al].mean()):+6.2f}" if (~al).any() else "  n/a "
            lines.append(f"  t={t:5.2f}s  {s_all} | 생존 {s_alive} | 낙상 {s_fall} | 롤 {float(rl_.mean()):+5.2f} | 양발접촉 {100*float(both_[al_t].mean()) if al_t.any() else 0:5.1f} % (서있던 env {int(al_t.sum())})")
        # 추종오차: 최종 시점, 생존 env 와 전 env
        t_end, err_end = err_hist[-1]
        lines.append(f"관절 추종오차(물리목표−실제, deg) @t={t_end:.1f}s — 생존 env 평균 / 전 env 평균:")
        for k, n in enumerate(jn):
            if any(s in n for s in SAG):
                ea = float(err_end[alive, k].mean()) if alive.any() else float("nan")
                lines.append(f"   {n:20s} {ea:+6.2f} / {float(err_end[:, k].mean()):+6.2f}")
        t2, err2 = err_hist[min(3, len(err_hist) - 1)]
        lines.append(f"관절 추종오차 @t={t2:.1f}s (전 env 평균): " + " ".join(f"{n.replace('_joint','')}{float(err2[:, k].mean()):+.2f}" for k, n in enumerate(jn) if any(s in n for s in SAG)))
        _, fz_end = fz_hist[-1]
        lines.append(f"발 수직력 @t={t_end:.1f}s 생존 env 평균 L {float(fz_end[alive, 0].mean()) if alive.any() else float('nan'):.1f} / R {float(fz_end[alive, 1].mean()) if alive.any() else float('nan'):.1f} N (체중 172.9 N)")
        open(os.path.join(a.out_dir, f"{a.tag}.txt"), "w").write("\n".join(lines) + "\n")
        with open(os.path.join(a.out_dir, f"{a.tag}_envs.csv"), "w") as f:
            f.write("env,fell_s,pitch_end,roll_end\n")
            pt_e, rl_e = pitch_hist[-1][1], pitch_hist[-1][2]
            for e in range(N):
                f.write(f"{e},{float(fell[e]):.2f},{float(pt_e[e]):.2f},{float(rl_e[e]):.2f}\n")
        print("\n".join(lines))
env.close()
app.close()
