"""접촉 전수조사 진단 — 후방 전도의 실제 원인 규명.

부호는 프로브로 정상 확인됨. 남은 용의자:
(1) 자기충돌 컨벡스헐 간섭 (구 자산과 충돌 근사가 다름) → --no-self-collision 비교
(2) 특정 관절 토크 레일링 → 관절별 인가 토크 로깅
(3) 접촉 바디 오인 → 센서 자체 인덱스로 >1N 바디 전수 리스팅

실행:
  isaaclab.sh -p rl_walking/scripts/diagnose_contacts.py --tag SC_ON --headless
  isaaclab.sh -p rl_walking/scripts/diagnose_contacts.py --tag SC_OFF --no-self-collision --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--no-self-collision", action="store_true")
parser.add_argument("--kp", type=float, default=250.0)
parser.add_argument("--kd", type=float, default=25.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys

import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv

from rl_walking.env_cfg import Biped12FlatEnvCfg

OUT = "/home/ryu/humanoid_leg_test1/log_picture/06_접촉전수조사.txt"
rep = []


def p(s=""):
    rep.append(str(s))


cfg = Biped12FlatEnvCfg()
cfg.scene.num_envs = 1
cfg.events.push_robot = None
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {
    k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
cfg.observations.policy.enable_corruption = False
cfg.events.actuator_gains = None
cfg.events.joint_friction = None
cfg.events.joint_armature = None
for g in cfg.scene.robot.actuators.values():
    g.stiffness = args.kp
    g.damping = args.kd
if args.no_self_collision:
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
robot = env.scene["robot"]
cs = env.scene.sensors["contact_forces"]
sensor_bodies = cs.body_names  # 센서 자체 순서 (관절체 순서와 다를 수 있음!)

jn = robot.data.joint_names
sag = [jn.index(n) for n in ("left_hip_f_joint", "left_knee_joint", "left_ankle_f_joint")]

p(f"--- {args.tag}: kp={args.kp:.0f}/kd={args.kd:.0f}, self_collision="
  f"{not args.no_self_collision} ---")
p(f"센서 바디 순서: {sensor_bodies}")
p("step |  z    | pgx    | L발목f q/tau | L무릎 q/tau | 접촉>1N 바디(힘)")

for i in range(120):
    acts = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    _, _, terminated, _, _ = env.step(acts)
    if i % 10 == 0 or terminated.any():
        forces = cs.data.net_forces_w[0].norm(dim=-1)
        touching = [
            f"{sensor_bodies[b]}({float(forces[b]):.0f})"
            for b in range(len(sensor_bodies))
            if float(forces[b]) > 1.0
        ]
        q = robot.data.joint_pos[0]
        tau = robot.data.applied_torque[0]
        p(f"{i:4d} | {float(robot.data.root_pos_w[0,2]):.3f} | "
          f"{float(robot.data.projected_gravity_b[0,0]):+.3f} | "
          f"{float(q[sag[2]]):+.3f}/{float(tau[sag[2]]):+6.1f} | "
          f"{float(q[sag[1]]):+.3f}/{float(tau[sag[1]]):+6.1f} | "
          f"{' '.join(touching)}"
          + ("  << 종료!" if terminated.any() else ""))
    if terminated.any() and i > 3:
        break

with open(OUT, "a") as f:
    f.write("\n".join(rep) + "\n\n")
print("\n".join(rep), flush=True)
env.close()
simulation_app.close()
