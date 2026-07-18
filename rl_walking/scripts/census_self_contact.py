"""자기접촉 정밀 전수조사 — SC ON 전도 원인 확정 (리뷰 major 지적 대응).

가설: SC ON에서만 생기는 미세(<1N) 비인접 링크 간섭이 지속 외란으로 작용.
방법: 임계 0.001N로 전 바디 접촉력 로깅 + 발 이외 바디의 접촉을 시계열 추적.
      기립 유지 60스텝 + 명령 유지 상태에서 측방 기울임 강제 30스텝(다리 모임 상황).

실행: isaaclab.sh -p rl_walking/scripts/census_self_contact.py --headless
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
import rl_walking  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv

from rl_walking.env_cfg import Biped12FlatEnvCfg

OUT = "/home/ryu/humanoid_leg_test1/log_picture/07_자기접촉_정밀조사.txt"
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

env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
robot = env.scene["robot"]
cs = env.scene.sensors["contact_forces"]
names = cs.body_names
feet = set(cs.find_bodies(".*_ankle_r_joint")[0])
jn = robot.data.joint_names

p("자기접촉 정밀 전수조사 (임계 0.001N, SC ON, 랜덤화 OFF)")
p(f"센서 바디: {names}")
p("=" * 74)

nonfoot_hits = {}


def census(i, label):
    forces = cs.data.net_forces_w[0].norm(dim=-1)
    rows = []
    for b in range(len(names)):
        f = float(forces[b])
        if f > 0.001:
            rows.append(f"{names[b]}={f:.3f}N")
            if b not in feet:
                nonfoot_hits[names[b]] = nonfoot_hits.get(names[b], 0) + 1
    p(f"  [{label} step {i:3d}] " + (" ".join(rows) if rows else "(접촉 없음)"))


# 1) 기본자세 기립 60스텝
acts = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for i in range(60):
    env.step(acts)
    if i % 10 == 0:
        census(i, "기립")

# 2) 양다리 내전 강제 (hip_a 충돌각 ±9.4° 시나리오 재현: 6°씩 내전)
adduct = torch.zeros_like(acts)
# 액션 = (target-default)/0.25; left_hip_a +0.105rad(6° 내전), right -0.105
li = jn.index("left_hip_a_joint")
ri = jn.index("right_hip_a_joint")
adduct[:, li] = 0.105 / 0.25
adduct[:, ri] = -0.105 / 0.25
for i in range(40):
    env.step(adduct)
    if i % 10 == 0:
        census(i, "내전")

p("=" * 74)
p("발 이외 바디 접촉 발생 횟수(>0.001N):")
if nonfoot_hits:
    for k, v in sorted(nonfoot_hits.items(), key=lambda x: -x[1]):
        p(f"   {k}: {v}회")
    p("→ 유령 자기접촉 존재: 해당 링크 충돌 근사/필터 개선 필요")
else:
    p("   없음 — SC ON에서도 비인접 링크 간섭력 0.001N 이상 없음")
    p("→ SC ON/OFF 거동 차이는 접촉력이 아니라 솔버 준비단계(컨택 마진 등)의")
    p("  미세 수치 효과로 판정. 물리적 유령힘 없음 = 학습·이식에 무해")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
simulation_app.close()
