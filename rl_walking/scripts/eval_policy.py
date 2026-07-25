"""학습된 정책 정량 평가 — 걸음새 품질 지표 + 이식 제약 준수 검사.

지표: 낙상률, 속도 추종 오차, 직립도, 케이던스/에어타임/양발지지율,
      관절속도 최대(AK45-36 5rad/s 검사), 토크 통계, 액션 레이트, 전진 거리.
부수 효과: 체크포인트를 exported/policy.pt(.onnx)로 익스포트 (이식 준비).

실행: isaaclab.sh -p rl_walking/scripts/eval_policy.py --headless \
        [--load_run 2026-07-19_*] [--steps 1500] [--num_envs 64] [--cmd_x 0.5]
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
# Stage4 리뷰 수정: 태스크 하드코딩 제거 — Stage4는 experiment_name(biped12_stage4)과
# 관측 차원(245)이 달라, Flat 고정이면 '무경고로 이전 flat 체크포인트'를 평가하는 함정.
# 기본값은 종전과 동일 (기존 호출 무변경).
parser.add_argument("--task", type=str, default="Biped12-Velocity-Flat-Play-v0",
                    help="평가 태스크 (…-Play-v0). Stage4: Biped12-Velocity-Stage4-Play-v0")
parser.add_argument("--steps", type=int, default=1500, help="평가 스텝 (50Hz — 1500=30s)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--cmd_x", type=float, default=0.5)
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

import gymnasium as gym

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"정책평가{args_cli.out_tag}.txt")

# 태스크 등록 kwargs에서 agent/env cfg를 해석 — experiment_name(로그 디렉토리)과
# 관측 차원이 태스크와 항상 일치 (Flat 기본값이면 종전 클래스와 동일 객체)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
# 평가 명령 고정: 전진 cmd_x, 요 0 (heading 유지)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_x, args_cli.cmd_x)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
env_cfg.episode_length_s = 60.0  # 평가 중 타임아웃 리셋 방지

log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_nn = runner.alg.policy
normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

# 익스포트 (이식 준비물)
export_dir = os.path.join(os.path.dirname(resume_path), "exported")
export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")

uenv = env.unwrapped
robot = uenv.scene["robot"]
cs = uenv.scene.sensors["contact_forces"]
sensor_feet = [cs.body_names.index(n) for n in ("left_ankle_r_joint", "right_ankle_r_joint")]
jn = robot.data.joint_names
ankle_r_ids = [jn.index("left_ankle_r_joint"), jn.index("right_ankle_r_joint")]

N = args_cli.num_envs
dev = uenv.device
falls = torch.zeros(N, device=dev)
sum_verr = 0.0
sum_lean = 0.0
p_lean_max = 0.0
sum_h = 0.0
dsup = 0.0
ssup = 0.0
air_times = []
steps_cnt = torch.zeros(N, 2, device=dev)
tau_abs = []
jvel_ankle_r_max = 0.0
jvel_all_max = 0.0
act_rate = 0.0
prev_act = None
x0 = robot.data.root_pos_w[:, 0].clone()
traj = {k: [] for k in ("root", "pg", "q", "act", "contact")}

obs = env.get_observations()
import isaaclab.utils.math as math_utils

for i in range(args_cli.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_nn.reset(dones)

    term = uenv.termination_manager.terminated  # 타임아웃 제외 실제 낙상
    falls += term.float()

    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3]
    )
    sum_verr += float((vel_yaw[:, 0] - args_cli.cmd_x).abs().mean())
    pg = robot.data.projected_gravity_b
    lean = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1, 1)))
    sum_lean += float(lean.mean())
    p_lean_max = max(p_lean_max, float(lean.quantile(0.95)))
    sum_h += float(robot.data.root_pos_w[:, 2].mean())

    in_contact = cs.data.net_forces_w[:, sensor_feet, :].norm(dim=-1) > 1.0
    both = in_contact.all(dim=1).float().mean()
    single = (in_contact.sum(dim=1) == 1).float().mean()
    dsup += float(both)
    ssup += float(single)
    fc = cs.compute_first_contact(uenv.step_dt)[:, sensor_feet]
    if fc.any():
        air_times.append(float(cs.data.last_air_time[:, sensor_feet][fc].mean()))
    steps_cnt += fc.float()

    tau_abs.append(float(robot.data.applied_torque.abs().mean()))
    jv = robot.data.joint_vel.abs()
    jvel_ankle_r_max = max(jvel_ankle_r_max, float(jv[:, ankle_r_ids].max()))
    jvel_all_max = max(jvel_all_max, float(jv.max()))
    if prev_act is not None:
        act_rate += float((actions - prev_act).abs().mean())
    prev_act = actions.clone()

    traj["root"].append(robot.data.root_pos_w[0].cpu().numpy().copy())
    traj["pg"].append(pg[0].cpu().numpy().copy())
    traj["q"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    traj["act"].append(actions[0].cpu().numpy().copy())
    traj["contact"].append(in_contact[0].cpu().numpy().copy())

S = args_cli.steps
dist = (robot.data.root_pos_w[:, 0] - x0)
dur_s = S / 50.0
rep = []
rep.append(f"정책 평가 — checkpoint: {resume_path}")
rep.append(f"조건: {N} env × {dur_s:.0f}s, cmd_x={args_cli.cmd_x} m/s (요 0)")
rep.append("=" * 66)
rep.append(f"낙상: {int(falls.sum())}회 / {N}env — 낙상률 {float(falls.sum())/N*100:.1f}% "
           f"(무낙상 env {int((falls==0).sum())}/{N})")
rep.append(f"전진 거리: 평균 {float(dist.mean()):.2f} m (기대 {args_cli.cmd_x*dur_s:.1f}), "
           f"min {float(dist.min()):.2f} / max {float(dist.max()):.2f}")
rep.append(f"속도 추종 오차 |v-cmd|: 평균 {sum_verr/S:.3f} m/s")
rep.append(f"직립도: 평균 기울기 {sum_lean/S:.2f}°, p95 최대 {p_lean_max:.2f}° "
           f"(FSM 데모 피크 ~15-18° 참고)")
rep.append(f"골반 높이 평균: {sum_h/S:.3f} m (직립 0.649)")
rep.append(f"게이트: 단일지지 {ssup/S*100:.1f}% / 양발지지 {dsup/S*100:.1f}% / "
           f"공중 {100-(ssup+dsup)/S*100:.1f}%")
cad = float(steps_cnt.sum()) / N / dur_s
rep.append(f"걸음: 평균 에어타임 {np.mean(air_times) if air_times else 0:.3f} s, "
           f"케이던스 {cad:.2f} 걸음/s ({cad*60:.0f}spm; 사람 보행 ~1.5-2.0걸음/s)")
rep.append(f"토크: 평균 |tau| {np.mean(tau_abs):.2f} Nm (한계 25)")
rep.append(f"관절속도 최대: 전체 {jvel_all_max:.2f} rad/s | "
           f"발목롤 {jvel_ankle_r_max:.2f} rad/s (AK45-36 한계 5.0 — "
           f"{'준수 OK' if jvel_ankle_r_max <= 5.0 else '위반!!'})")
rep.append(f"액션 레이트 평균: {act_rate/max(S-1,1):.4f} (슬루 여유 지표)")

os.makedirs(os.path.join(REPO, "demo_output"), exist_ok=True)
npz = os.path.join(REPO, "demo_output", f"rl_eval_traj{args_cli.out_tag}.npz")
np.savez_compressed(npz, **{k: np.array(v) for k, v in traj.items()},
                    joint_names=np.array(jn), checkpoint=resume_path)
rep.append(f"궤적 저장: {npz} (env0 전체 스텝)")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
simulation_app.close()
