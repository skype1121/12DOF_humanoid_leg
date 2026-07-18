"""학습 환경 스폰 검증 — 학습 시작 전 필수 게이트.

검증 항목:
1. 관절 이름/순서 (Isaac dof 순서는 런타임 확인 필수 — 하드웨어맵 매핑용)
2. 기본자세로 100스텝(2s) 제자리: root 높이 유지(≈0.65), 중력벡터 (0,0,-1),
   낙상/종료 없음  → 관절 부호·리밋·게인(kp150/kd5)·질량이 전부 맞아야 통과
3. 관측 차원 (policy 45 = 3+3+3+12+12+12, critic 3)
4. 발 접촉력 존재 (접촉센서 배선 확인)
5. 보상 유한값 (NaN 없음)

실행: /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/validate_env.py --headless
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

OUT = "/home/ryu/humanoid_leg_test1/log_picture/03_스폰검증_리포트.txt"
rep = []


def p(s=""):
    rep.append(str(s))


def fmt(t):
    return [round(float(v), 4) for v in t]


cfg = Biped12FlatEnvCfg()
cfg.scene.num_envs = 4
# 검증은 결정론적으로: 외란·리셋 랜덤화 전부 끔
cfg.events.push_robot = None
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.events.reset_base.params["velocity_range"] = {
    k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
}
cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
cfg.observations.policy.enable_corruption = False

env = ManagerBasedRLEnv(cfg=cfg)
obs, _ = env.reset()
robot = env.scene["robot"]

p("biped12 학습환경 스폰 검증 리포트 (자동 생성)")
p("=" * 70)
p("1) 관절 순서 (Isaac dof order — 하드웨어맵 매핑에 사용):")
for i, n in enumerate(robot.data.joint_names):
    p(f"   [{i:2d}] {n}")
p(f"   바디: {robot.data.body_names}")
p(f"   기본 관절각[rad]: {fmt(robot.data.default_joint_pos[0])}")

p("=" * 70)
p("2) 관측 차원:")
for k, v in obs.items():
    p(f"   {k}: {tuple(v.shape)}")

# 기본자세 유지 100스텝 (액션 0 = 기본자세 목표, 개루프 PD kp150/kd5)
n_fall = 0
for i in range(100):
    acts = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    obs, rew, terminated, truncated, info = env.step(acts)
    n_fall += int(terminated.sum())

p("=" * 70)
p("3) 100스텝(2s) 제자리 유지 결과 (개루프 — 약간의 드리프트는 정상):")
p(f"   root z (기대≈0.63~0.66): {fmt(robot.data.root_pos_w[:, 2])}")
p(f"   projected_gravity[0] (기대≈[0,0,-1]): {fmt(robot.data.projected_gravity_b[0])}")
p(f"   관절각[0] (기대≈기본자세): {fmt(robot.data.joint_pos[0])}")
p(f"   중도 종료 횟수: {n_fall} (0이어야 정상)")
p(f"   보상 (유한해야 함): {fmt(rew)}")

p("=" * 70)
p("4) 발 접촉력 [N] (양발 접지 시 ~49N/발):")
cs = env.scene.sensors["contact_forces"]
# 주의: 센서 바디 순서(DFS)는 관절체 바디 순서(BFS)와 다르다 —
# 반드시 센서 자체의 find_bodies로 인덱싱 (리뷰에서 잡힌 배선 버그 수정)
ids, names = cs.find_bodies(".*_ankle_r_joint")
forces = cs.data.net_forces_w[:, ids, :].norm(dim=-1)
p(f"   {names}: {[fmt(f) for f in forces]}")

ok = (
    n_fall == 0
    and torch.isfinite(rew).all()
    and float(robot.data.root_pos_w[:, 2].min()) > 0.55
    and abs(float(robot.data.projected_gravity_b[0, 2]) + 1.0) < 0.1
    # 양발 모두 실접촉 확인 (배선 오류가 게이트에서 잡히도록)
    and bool((forces[0] > 20.0).all())
)
p("=" * 70)
p(f"판정: {'PASS' if ok else 'FAIL'}")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)

env.close()
simulation_app.close()
