"""페이로드(배터리/짐) 강건성 그리드 평가 — 논문식 부하 실험.

시나리오: 골반에 각형 배터리 부착 — 위/앞/뒤 × 질량 {0,1,2,3,5}kg.
구현: base 링크 질량 증가 + 합성 CoM 이동을 PhysX 뷰에 직접 기록.
      (payload CoM: 위=(0,0,+0.13), 앞=(+0.07,0,+0.05), 뒤=(-0.07,0,+0.05) [m])
각 조건: 64 env × 15s, cmd 0.4m/s 전진 — 낙상률/추종오차/직립도 기록.

실행: isaaclab.sh -p rl_walking/scripts/eval_payload.py --headless [--load_run ...]
      Stage4: --task Biped12-Velocity-Stage4-Play-v0
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
# eval_policy.py와 동일 패턴: 태스크 등록 kwargs에서 agent/env cfg 해석 —
# experiment_name(로그 디렉토리)·관측 차원이 태스크와 항상 일치.
# 기본값은 종전과 동일 (기존 호출 무변경).
parser.add_argument("--task", type=str, default="Biped12-Velocity-Flat-Play-v0",
                    help="평가 태스크 (…-Play-v0). Stage4: Biped12-Velocity-Stage4-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--secs", type=float, default=15.0)
parser.add_argument("--cmd_x", type=float, default=0.4)
parser.add_argument("--mass_scale", type=float, default=1.0,
                    help="전 링크 질량 스케일 (실물 12kg 정합 = 1.144, 12URDF0725 자중 10.49kg 기준)")
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

import gymnasium as gym

import isaaclab.utils.math as math_utils

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"페이로드평가{args_cli.out_tag}.txt")

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_x, args_cli.cmd_x)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
env_cfg.episode_length_s = 60.0

log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_nn = runner.alg.policy

uenv = env.unwrapped
robot = uenv.scene["robot"]
base_id = robot.find_bodies("base")[0][0]
N = args_cli.num_envs
STEPS = int(args_cli.secs * 50)

# 원본 질량/CoM 백업 (PhysX 뷰 — CPU 텐서 요구)
view = robot.root_physx_view
masses0 = view.get_masses().clone() * args_cli.mass_scale  # 실물 질량 정합 스케일
coms0 = view.get_coms().clone()
inertias0 = view.get_inertias().clone() * args_cli.mass_scale
ALL = torch.arange(N, dtype=torch.int32)
if args_cli.mass_scale != 1.0:
    view.set_inertias(inertias0, ALL)

POSITIONS = {
    "위(+z13cm)": (0.0, 0.0, 0.13),
    "앞(+x7cm)": (0.07, 0.0, 0.05),
    "뒤(-x7cm)": (-0.07, 0.0, 0.05),
}
MASSES = [0.0, 1.0, 2.0, 3.0, 5.0]


def set_payload(m_add, offset):
    masses = masses0.clone()
    coms = coms0.clone()
    m0 = masses0[:, base_id]
    masses[:, base_id] = m0 + m_add
    if m_add > 0:
        off = torch.tensor(offset, dtype=coms.dtype)
        # 합성 CoM = (m0*com0 + m_add*payload_pos) / (m0+m_add)
        com_base = coms0[:, base_id, :3]
        coms[:, base_id, :3] = (m0.unsqueeze(-1) * com_base + m_add * off) / (
            (m0 + m_add).unsqueeze(-1)
        )
    view.set_masses(masses, ALL)
    view.set_coms(coms, ALL)


@torch.inference_mode()
def rollout():
    # 전체(리셋 포함)를 inference_mode로 — env 내부 버퍼 모드 혼용 금지
    env.unwrapped.reset()
    obs = env.get_observations()
    falls = torch.zeros(N, device=uenv.device)
    verr = 0.0
    lean_sum = 0.0
    for _ in range(STEPS):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_nn.reset(dones)
        falls += uenv.termination_manager.terminated.float()
        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3]
        )
        verr += float((vel_yaw[:, 0] - args_cli.cmd_x).abs().mean())
        pg = robot.data.projected_gravity_b
        lean_sum += float(torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1, 1))).mean())
    n_fall_env = int((falls > 0).sum())
    return n_fall_env, verr / STEPS, lean_sum / STEPS


rep = []
rep.append(f"페이로드 강건성 그리드 평가 — checkpoint: {resume_path}")
rep.append(f"조건: {N} env × {args_cli.secs:.0f}s, cmd_x={args_cli.cmd_x} m/s. 로봇 자중 10.49kg (12URDF0725)")
rep.append(f"판정: 낙상env비율 <10% = PASS, 10~30% = 한계, >30% = FAIL")
rep.append("=" * 74)
rep.append(f"{'부착위치':14s} {'질량':>5s} | {'낙상env':>8s} | {'추종오차':>9s} | {'평균기울기':>9s} | 판정")

baseline = None
for pos_name, off in POSITIONS.items():
    for m in MASSES:
        if m == 0.0 and baseline is not None:
            continue  # 0kg은 한 번만
        set_payload(m, off)
        nf, ve, ln = rollout()
        rate = nf / N * 100
        verdict = "PASS" if rate < 10 else ("한계" if rate <= 30 else "FAIL")
        label = "기준(0kg)" if m == 0.0 else pos_name
        rep.append(f"{label:14s} {m:4.0f}kg | {nf:3d}/{N} ({rate:4.0f}%) | "
                   f"{ve:7.3f}m/s | {ln:7.2f}° | {verdict}")
        print(rep[-1], flush=True)
        if m == 0.0:
            baseline = (nf, ve, ln)

set_payload(0.0, (0, 0, 0))  # 복원
open(OUT, "w").write("\n".join(rep) + "\n")
print("saved:", OUT, flush=True)
env.close()
simulation_app.close()
