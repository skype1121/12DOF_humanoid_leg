"""12관절 부호 규약 실측 프로브 — 변환 자산의 관절 방향을 무중력에서 확정.

배경: A/B 진단에서 구 검증 게인(kp250/kd25)으로도 후방 전도 → 기본자세의
무릎/발목 트림 부호가 이 자산의 규약과 어긋났을 가능성. 추측 대신 실측한다.

방법: 중력 OFF, 루트를 매 스텝 항등자세로 동결(자유베이스 반작용 제거).
각 관절에 해부학적 '기대 방향' 델타를 명령하고 자식 링크의 변위/자세 변화를 기록.
기대치(joint_direction 실측 + ANAT_SIGN, base 프레임: +X전방 +Y좌 +Z상):
  hip_f  굴곡(left +0.3 / right -0.3)  → 무릎 링크 +X (앞으로)
  hip_a  내전(left +0.2 / right -0.2)  → 발 링크가 몸 중앙쪽 (left: -Y, right: +Y)
  hip_r  외회전(left +0.3 / right -0.3)→ 발 yaw (left: +yaw, right: -yaw)
  knee   굴곡(left -0.4 / right +0.4)  → 발목 링크 -X (뒤로) & +Z (위로)
  ankle_f 배굴(left +0.3 / right -0.3) → 발 링크 +X (앞으로 ~2.4cm)
  ankle_r 외반(left +0.2 / right +0.2) → 발 롤 (left: -roll(바깥쪽), right: +roll)

실행: /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/probe_signs.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys

import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat

from rl_walking.biped12_cfg import BIPED12_CFG

OUT = "/home/ryu/humanoid_leg_test1/log_picture/05_관절부호_실측프로브.txt"
rep = []


def p(s=""):
    rep.append(str(s))


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))

cfg = BIPED12_CFG.replace(prim_path="/World/Robot")
cfg.spawn.rigid_props.disable_gravity = True
cfg.init_state.pos = (0.0, 0.0, 1.0)
robot = Articulation(cfg)
sim.reset()
robot.update(sim.get_physics_dt())

names = robot.data.joint_names
jmap = {n: i for i, n in enumerate(names)}
default = robot.data.default_joint_pos.clone()

ROOT_POSE = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=robot.device)
ZERO6 = torch.zeros(1, 6, device=robot.device)

# (관절, 델타, 마커 링크, 해부학 기대)
PROBES = [
    ("left_hip_f_joint", +0.3, "left_knee_joint", "무릎 +X(앞) [굴곡]"),
    ("right_hip_f_joint", -0.3, "right_knee_joint", "무릎 +X(앞) [굴곡]"),
    ("left_hip_a_joint", +0.2, "left_ankle_r_joint", "발 -Y(중앙쪽) [내전]"),
    ("right_hip_a_joint", -0.2, "right_ankle_r_joint", "발 +Y(중앙쪽) [내전]"),
    ("left_hip_r_joint", +0.3, "left_ankle_r_joint", "발 +yaw [외회전]"),
    ("right_hip_r_joint", -0.3, "right_ankle_r_joint", "발 -yaw [외회전]"),
    ("left_knee_joint", -0.4, "left_ankle_f_joint", "발목 -X(뒤)+Z(위) [굴곡]"),
    ("right_knee_joint", +0.4, "right_ankle_f_joint", "발목 -X(뒤)+Z(위) [굴곡]"),
    ("left_ankle_f_joint", +0.3, "left_ankle_r_joint", "발 +X(앞) [배굴]"),
    ("right_ankle_f_joint", -0.3, "right_ankle_r_joint", "발 +X(앞) [배굴]"),
    ("left_ankle_r_joint", +0.2, "left_ankle_r_joint", "발 롤 [외반]"),
    ("right_ankle_r_joint", +0.2, "right_ankle_r_joint", "발 롤 [외반]"),
]


def freeze_root():
    robot.write_root_pose_to_sim(ROOT_POSE)
    robot.write_root_velocity_to_sim(ZERO6)


def settle(target, n):
    for _ in range(n):
        freeze_root()
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())


def body_state(bname):
    bid = robot.find_bodies(bname)[0][0]
    pos = robot.data.body_pos_w[0, bid].clone()
    quat = robot.data.body_quat_w[0, bid].clone()
    r, pch, y = euler_xyz_from_quat(quat.unsqueeze(0))
    return pos, torch.tensor([float(r[0]), float(pch[0]), float(y[0])])


p("12관절 부호 규약 실측 프로브 (중력 OFF, 루트 동결)")
p("델타는 '해부학적 기대 방향' — 측정이 기대와 맞으면 규약 일치")
p("=" * 78)

for jname, delta, marker, expect in PROBES:
    # 초기화: 전관절 0 (트림 없는 순수 중립)
    zero = torch.zeros_like(default)
    robot.write_joint_state_to_sim(zero, torch.zeros_like(zero))
    freeze_root()
    settle(zero, 20)
    pos0, rpy0 = body_state(marker)

    target = zero.clone()
    target[0, jmap[jname]] = delta
    settle(target, 80)
    pos1, rpy1 = body_state(marker)
    reached = float(robot.data.joint_pos[0, jmap[jname]])

    dp = (pos1 - pos0).cpu()
    dr = (rpy1 - rpy0).cpu()
    # 라디안 랩어라운드 정리
    dr = torch.atan2(torch.sin(dr), torch.cos(dr))
    p(f"{jname:22s} cmd={delta:+.2f} 도달={reached:+.3f} | 마커 {marker}")
    p(f"   Δpos[m]  x={float(dp[0]):+.4f} y={float(dp[1]):+.4f} z={float(dp[2]):+.4f}")
    p(f"   Δrpy[rad] r={float(dr[0]):+.3f} p={float(dr[1]):+.3f} y={float(dr[2]):+.3f}")
    p(f"   기대: {expect}")
    p("")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
simulation_app.close()
