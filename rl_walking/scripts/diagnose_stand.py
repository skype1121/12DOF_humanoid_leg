"""A/B 게인 진단 — 개루프 기립 안정성으로 자산 버그 vs kd5 플랜트 특성 분리.

A: kp250/kd25 (구 시뮬 보행 검증 게인 — 이걸로 넘어지면 변환 자산 버그)
B: kp150/kd5  (실물 이식 게인 — A가 서고 B만 넘어지면 저댐핑 플랜트 특성)

각 케이스 6초(300스텝) 로깅: root z, 기울기, 발 z/접촉력, 토크 최대치, 종료 원인.

실행: /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/diagnose_stand.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--kp", type=float, required=True)
parser.add_argument("--kd", type=float, required=True)
parser.add_argument("--tag", type=str, default="")
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

OUT = "/home/ryu/humanoid_leg_test1/log_picture/04_기립AB진단_kp250kd25_vs_kp150kd5.txt"
rep = []


def p(s=""):
    rep.append(str(s))


def fmt(t, nd=3):
    return [round(float(v), nd) for v in t]


def run_case(tag, kp, kd):
    cfg = Biped12FlatEnvCfg()
    cfg.scene.num_envs = 2
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.observations.policy.enable_corruption = False
    # 게인 DR 끄고 지정값 고정
    cfg.events.actuator_gains = None
    cfg.events.joint_friction = None
    cfg.events.joint_armature = None
    for g in cfg.scene.robot.actuators.values():
        g.stiffness = kp
        g.damping = kd

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    robot = env.scene["robot"]
    cs = env.scene.sensors["contact_forces"]
    # 두 인덱스 공간 분리: 접촉력은 센서 순서(DFS), 위치는 관절체 순서(BFS)
    fids, _ = cs.find_bodies(".*_ankle_r_joint")
    bids, _ = robot.find_bodies(".*_ankle_r_joint")

    p(f"--- {tag}: kp={kp}, kd={kd} ---")
    p("  step |  z0   | pgx    pgy   | 발z L/R      | 접촉N L/R     | tau_max | term")
    n_term = 0
    for i in range(300):
        acts = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        _, _, terminated, truncated, _ = env.step(acts)
        n_term += int(terminated.sum())
        if i % 25 == 0 or (terminated.any() and n_term <= 3):
            pg = robot.data.projected_gravity_b[0]
            fz = robot.data.body_pos_w[0, bids, 2]
            fN = cs.data.net_forces_w[0, fids, :].norm(dim=-1)
            tau = float(robot.data.applied_torque[0].abs().max())
            term_names = ""
            if terminated.any():
                for tname in env.termination_manager.active_terms:
                    tt = env.termination_manager.get_term(tname)
                    if tt.any():
                        term_names += tname + " "
            p(f"  {i:4d} | {float(robot.data.root_pos_w[0,2]):.3f} | "
              f"{float(pg[0]):+.3f} {float(pg[1]):+.3f} | "
              f"{float(fz[0]):.3f} {float(fz[1]):.3f} | "
              f"{float(fN[0]):5.1f} {float(fN[1]):5.1f} | "
              f"{tau:5.1f} | {term_names}")
    p(f"  총 종료(2env×300스텝): {n_term}")
    env.close()
    return n_term


# 프로세스당 env 1개만 안전 (SimulationContext 싱글턴) — 케이스별 별도 실행
run_case(args.tag or f"kp{args.kp:.0f}/kd{args.kd:.0f}", args.kp, args.kd)

with open(OUT, "a") as f:
    f.write("\n".join(rep) + "\n\n")
print("\n".join(rep), flush=True)
simulation_app.close()
