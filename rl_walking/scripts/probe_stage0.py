"""0단계 강체홀드 실패 프로브 — eval_ladder 0단계와 동일 조건, 초반 궤적 계측.
어느 방향으로/얼마나 빨리/어떤 관절이 움직여 무너지는지 직접 본다."""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: F401  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped12-Velocity-Stage4V2-Stand-V6-v0")
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import rl_walking  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

ACTION_SCALE = 0.25

cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
cfg.scene.num_envs = args_cli.num_envs
if args_cli.usd:
    cfg.scene.robot.spawn.usd_path = args_cli.usd
cfg.seed = 0
cfg.episode_length_s = 60.0
for rng in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    setattr(cfg.commands.base_velocity.ranges, rng, (0.0, 0.0))
cfg.events.push_robot = None
cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
for k in ("roll", "pitch"):
    if k in cfg.events.reset_base.params["pose_range"]:
        cfg.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
vr = cfg.events.reset_base.params.get("velocity_range")
if isinstance(vr, dict):
    for k in list(vr):
        vr[k] = (0.0, 0.0)
if getattr(cfg.events, "reset_robot_joints", None):
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
if getattr(cfg.curriculum, "terrain_levels", None) is not None:
    cfg.curriculum.terrain_levels = None
cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
cfg.actions.joint_pos.min_delay_steps = 0
cfg.actions.joint_pos.max_delay_steps = 0
if getattr(cfg.events, "base_com", None):
    cfg.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
if getattr(cfg.events, "calib_bias", None):
    cfg.events.calib_bias.params["joint_bias_rad"] = (0.0, 0.0)
    cfg.events.calib_bias.params["imu_tilt_rad"] = (0.0, 0.0)
if getattr(cfg.events, "add_base_mass", None):
    m = cfg.events.add_base_mass.params["mass_distribution_params"]
    mid = 0.5 * (m[0] + m[1])
    cfg.events.add_base_mass.params["mass_distribution_params"] = (mid, mid)
if getattr(cfg.events, "joint_friction", None):
    cfg.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.0)
if getattr(cfg.events, "physics_material", None):
    pm = cfg.events.physics_material.params
    pm["static_friction_range"] = (0.8, 0.8)
    pm["dynamic_friction_range"] = (0.6, 0.6)
if getattr(cfg.events, "joint_armature", None):
    cfg.events.joint_armature.params["armature_distribution_params"] = (1.0, 1.0)

env = gym.make(args_cli.task, cfg=cfg)
env = RslRlVecEnvWrapper(env, clip_actions=100.0)
uenv = env.unwrapped
robot = uenv.scene["robot"]
dev = uenv.device
N = args_cli.num_envs
jn = robot.data.joint_names
print("PhysX 관절 순서:", jn, flush=True)
import omni.usd
_stage = omni.usd.get_context().get_stage()
for _p in ("/World/envs/env_0/Robot/base/collisions/pelvis/node_STL_BINARY_/mesh",
           "/World/envs/env_0/Robot/base/collisions/pelvis",
           "/World/envs/env_0/Robot/base/collisions"):
    _prim = _stage.GetPrimAtPath(_p)
    if _prim:
        _a = _prim.GetAttribute("physics:approximation")
        print(f"[검증] {_p} 존재={bool(_prim)} approx={_a.Get() if _a else '속성없음'}", flush=True)
    else:
        print(f"[검증] {_p} 없음", flush=True)


hold = (torch.zeros_like(robot.data.default_joint_pos) - robot.data.default_joint_pos) / ACTION_SCALE
env.get_observations()

print(f"{'k':>3s} {'z(m)':>7s} {'pitch°':>8s} {'roll°':>8s}  |qpos|max(관절)  term", flush=True)
for k in range(args_cli.steps):
    with torch.inference_mode():
        obs, _r, _d, _x = env.step(hold)
    if k == 0:
        cs = uenv.scene["contact_forces"]
        f = cs.data.net_forces_w[0].cpu().numpy()
        bn = cs.body_names
        print("k=0 바디별 접촉힘 |F| (N):", flush=True)
        for name, ff in zip(bn, f):
            mag = float(np.linalg.norm(ff))
            if mag > 0.1:
                print(f"    {name:28s} {mag:9.2f}  ({ff[0]:+.1f},{ff[1]:+.1f},{ff[2]:+.1f})", flush=True)
    root = robot.data.root_pos_w[0]
    pg = robot.data.projected_gravity_b[0].cpu().numpy()
    pitch = np.degrees(np.arctan2(pg[0], -pg[2]))
    roll = np.degrees(np.arctan2(pg[1], -pg[2]))
    q = robot.data.joint_pos[0].cpu().numpy()
    i = int(np.abs(q).argmax())
    term = uenv.termination_manager.terminated[0].item()
    if k % 3 == 0 or term:
        print(f"{k:3d} {root[2].item():7.4f} {pitch:8.2f} {roll:8.2f}  "
              f"{np.degrees(q[i]):+7.2f}°({jn[i]})  {'💀' if term else ''}", flush=True)
    if term:
        # 어떤 종료조건이 발화했는지
        tm = uenv.termination_manager
        for name in tm.active_terms:
            v = tm.get_term(name)[0].item() if hasattr(tm, "get_term") else None
            if v:
                print(f"    발화 종료조건: {name}", flush=True)
        break

env.close()
simulation_app.close()
