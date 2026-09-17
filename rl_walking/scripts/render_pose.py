"""고정 자세를 시뮬(Isaac Lab)에 올려 kp150 홀드로 세우고 옆·앞에서 캡처 + 정적 계측 (0917).

  OMNI_KIT_ACCEPT_EULA=YES ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/render_pose.py \
      --pose_json jetson/calib_data/level_pose_0914.json --tag level_v3 --headless
출력: log_picture/_pose/<tag>_{side,front,iso}.png, <tag>_metrics.txt
계측: 몸통 pitch/roll(시뮬 IMU 상당), 관절 추종오차, 발 접촉력, 전신 CoM(발목축·발 기준), 낙상 여부.
조건: eval_static_pose --nominal --flat 와 같음(게인 정격·지연 0·CoM/질량 랜덤 0·푸시 0·평지).
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
p.add_argument("--task", default="Biped12-Velocity-Stage4V2-Stand-V14-v0", help="V14 = 실질량 17.62 kg·패드 접촉 자산(usd_final3)")
p.add_argument("--pose_json", required=True, help="trims_joint_frame_deg 를 가진 JSON(관절 프레임 deg)")
p.add_argument("--tag", default="pose")
p.add_argument("--kd_scale", type=float, default=1.0, help="정격 kd 대비 배율 (kd4 = 0.8)")
p.add_argument("--settle_steps", type=int, default=150)
p.add_argument("--hold_steps", type=int, default=500)
p.add_argument("--out_dir", default="/home/ryu/humanoid_leg_test1/log_picture/_pose")
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

DEFAULT_RAD = {
    "left_knee_joint": -0.10471975511965977, "right_knee_joint": 0.10471975511965977,
    "left_ankle_f_joint": 0.0244, "right_ankle_f_joint": -0.0244,
}
ACTION_SCALE = 0.25
raw = json.load(open(a.pose_json, encoding="utf-8"))
pose = raw.get("trims_joint_frame_deg", raw)

ec = load_cfg_from_registry(a.task, "env_cfg_entry_point")
ec.scene.num_envs = 1
ec.seed = 0
ec.episode_length_s = 120.0
ec.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
ec.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
ec.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
ec.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
for k in ("roll", "pitch"):
    if k in ec.events.reset_base.params["pose_range"]:
        ec.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
ec.events.push_robot = None
ec.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
ec.events.actuator_gains.params["damping_distribution_params"] = (a.kd_scale, a.kd_scale)
ec.actions.joint_pos.min_delay_steps = 0
ec.actions.joint_pos.max_delay_steps = 0
if getattr(ec.events, "joint_friction", None) is not None:
    ec.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.0)
if getattr(ec.events, "base_com", None) is not None:
    ec.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
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
ec.viewer.resolution = (1280, 960)
ec.viewer.eye = (1.5, 0.0, 0.5)
ec.viewer.lookat = (0.0, 0.0, 0.45)

env = gym.make(a.task, cfg=ec, render_mode="rgb_array")
u = env.unwrapped
robot = u.scene["robot"]
dev = u.device
jn = list(robot.data.joint_names)
act = torch.zeros(1, len(jn), device=dev)
for i, n in enumerate(jn):
    act[0, i] = (math.radians(float(pose.get(n, 0.0))) - DEFAULT_RAD.get(n, 0.0)) / ACTION_SCALE
tgt = torch.tensor([math.radians(float(pose.get(n, 0.0))) for n in jn], device=dev)
os.makedirs(a.out_dir, exist_ok=True)

cs = u.scene.sensors.get("contact_forces")
feet = None
if cs is not None:
    try:
        feet = [cs.body_names.index(n) for n in ("left_ankle_r_joint", "right_ankle_r_joint")]
    except ValueError:
        feet = None

def whole_body_com():
    masses = robot.root_physx_view.get_masses().to(dev)[0]                   # (B,)
    com_w = robot.data.body_com_pos_w[0]                                     # (B,3)
    m = masses.sum()
    return (com_w * masses[:, None]).sum(0) / m, float(m)

def snap(name):
    img = u.render()
    Image.fromarray(np.asarray(img)).save(os.path.join(a.out_dir, f"{a.tag}_{name}.png"))

env.reset()
fell = None
with torch.inference_mode():
    for i in range(a.settle_steps + a.hold_steps):
        env.step(act)
        if fell is None and bool(u.termination_manager.terminated[0]):
            fell = i
        if i == a.settle_steps:
            # 카메라 이동 뒤 렌더 파이프라인이 따라오도록 몇 스텝 워밍업(첫 프레임 공백 방지)
            def view(eye, name):
                u.viewport_camera_controller.update_view_location(eye=eye, lookat=(0.0, 0.0, 0.42))
                for _ in range(4):
                    env.step(act); u.render()
                snap(name)
            view((1.5, 0.0, 0.5), "side")      # 로봇 왼쪽 옆 (전방 = −Y)
            view((-1.5, 0.0, 0.5), "side_r")   # 오른쪽 옆
            view((0.0, -1.5, 0.5), "front")
            view((1.1, -1.0, 0.75), "iso")
            # 계측(정착 시점)
            pg = robot.data.projected_gravity_b[0].cpu().numpy()
            pitch_deg = math.degrees(math.atan2(pg[0], -pg[2])); roll_deg = math.degrees(math.atan2(pg[1], -pg[2]))
            q = robot.data.joint_pos[0]; err = torch.rad2deg(q - tgt)
            com, mtot = whole_body_com()
            # 발목 축 위치(발 링크 원점) — 앞뒤 축은 학습 월드 기준 −Y 가 전방
            bn = list(robot.body_names); fi = [k for k, n in enumerate(bn) if "ankle_r" in n]
            foot_p = robot.data.body_pos_w[0, fi].mean(0)
            fwd_axis = torch.tensor([0.0, -1.0, 0.0], device=dev)
            com_ahead_cm = float(((com - foot_p) * fwd_axis).sum() * 100)
            fz = float(cs.data.net_forces_w[0, feet, 2].sum()) if feet is not None else float("nan")
            fzl = float(cs.data.net_forces_w[0, feet[0], 2]) if feet is not None else float("nan")
            fzr = float(cs.data.net_forces_w[0, feet[1], 2]) if feet is not None else float("nan")
            h = float(robot.data.root_pos_w[0, 2] - u.scene.env_origins[0, 2])
            lines = [f"pose_json {a.pose_json}", f"task {a.task} kd_scale {a.kd_scale} settle {a.settle_steps} steps",
                     f"총질량(시뮬) {mtot:.3f} kg | 골반높이 {h:.3f} m",
                     f"몸통 pitch(+앞) {pitch_deg:+.2f}° roll(+왼) {roll_deg:+.2f}°",
                     f"전신 CoM: 발목축 대비 앞 {com_ahead_cm:+.2f} cm, 높이 {float(com[2]-u.scene.env_origins[0,2]):.3f} m",
                     f"발 수직력 합 {fz:.1f} N (L {fzl:.1f} / R {fzr:.1f}) — 체중 {mtot*9.81:.1f} N",
                     "관절 추종오차(실제−목표, deg): " + " ".join(f"{n.replace('_joint','')}{float(err[k]):+.1f}" for k, n in enumerate(jn)),
                     "관절 실제각(deg): " + " ".join(f"{n.replace('_joint','')}{math.degrees(float(q[k])):+.1f}" for k, n in enumerate(jn))]
    lines.append(f"홀드 {a.hold_steps/50:.0f} s 결과: " + ("낙상 없음(정적 자립)" if fell is None else f"낙상 step {fell} ({fell/50:.1f} s)"))
    pg = robot.data.projected_gravity_b[0].cpu().numpy()
    lines.append(f"최종 몸통 pitch {math.degrees(math.atan2(pg[0], -pg[2])):+.2f}° roll {math.degrees(math.atan2(pg[1], -pg[2])):+.2f}°")
    u.viewport_camera_controller.update_view_location(eye=(1.5, 0.0, 0.5), lookat=(0.0, 0.0, 0.42))
    for _ in range(4):
        env.step(act); u.render()
    snap("end_side")
open(os.path.join(a.out_dir, f"{a.tag}_metrics.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print(f"[render_pose] DONE {a.tag}", flush=True)
env.close(); app.close()
