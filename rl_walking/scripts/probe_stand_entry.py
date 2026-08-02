"""서기 진입 프로브 — 실물 진입 조건(무릎 편 스탠드 + 명령 0)의 시뮬 재현 (2026-08-02).

배경: 실물에서 정책 서기 켜는 순간 출력이 틱마다 요동(발진 가드 트립 ×2).
러너는 폐루프 비트일치 복원됐으므로, 남은 질문은 "같은 진입 조건에서 시뮬
정책도 요동치는가"다.

시나리오:
  A sim-native: env 기본 리셋(기본자세+노이즈, 정책이 익숙한 진입)
  B real-entry : 리셋 직후 관절을 '실물 스탠드'(무릎 0°=곧게 폄, 나머지 기본자세)
                 로 강제 + 속도 0 — 실물 핸드오프 직후와 동일
지표: 첫 50틱의 관절별 raw 방향반전 수·진폭, 액션레이트, 낙상, 정착 여부.

실행: isaaclab.sh -p rl_walking/scripts/probe_stand_entry.py --headless \
        [--load_run 2026-08-02_04-31-34] [--steps 200]
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped12-Velocity-Stage4-Play-v0")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=4)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401

import gymnasium as gym

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

print("[CKPT] cfg 로드 시작", flush=True)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
env_cfg.episode_length_s = 60.0
# 명령 완전 0 (순수 서기) — ranges 전부 0이면 standing 비율 무관하게 명령 0
# (rel_standing_envs 오버라이드는 env 생성 행 의심으로 제거 — eval과 동일 패턴 유지)
bv = env_cfg.commands.base_velocity
bv.ranges.lin_vel_x = (0.0, 0.0)
bv.ranges.lin_vel_y = (0.0, 0.0)
bv.ranges.ang_vel_z = (0.0, 0.0)
bv.ranges.heading = (0.0, 0.0)

log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
print("[CKPT] gym.make 시작", flush=True)
env = gym.make(args_cli.task, cfg=env_cfg)
print("[CKPT] gym.make 완료", flush=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
print("[CKPT] runner.load 시작", flush=True)
runner.load(resume_path)
print("[CKPT] runner.load 완료", flush=True)
policy = runner.get_inference_policy(device=env.unwrapped.device)

uenv = env.unwrapped
robot = uenv.scene["robot"]
jn = robot.data.joint_names
knee_ids = [jn.index("left_knee_joint"), jn.index("right_knee_joint")]
print(f"[probe] checkpoint: {resume_path}")


def run_scenario(label, force_straight_knees):
    print(f"[CKPT] {label} reset", flush=True)
    obs, _ = env.reset()
    print(f"[CKPT] {label} reset 완료", flush=True)
    if force_straight_knees:
        # 실물 진입 재현: 무릎 0°(곧게 폄), 나머지 기본자세, 속도 0, 몸통 수직
        jpos = robot.data.default_joint_pos.clone()
        jpos[:, knee_ids] = 0.0
        jvel = torch.zeros_like(robot.data.default_joint_vel)
        robot.write_joint_state_to_sim(jpos, jvel)
        root = robot.data.default_root_state.clone()
        root[:, :3] += uenv.scene.env_origins
        root[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        uenv.scene.write_data_to_sim()
        # 강제 상태가 관측에 반영되도록 0액션 1틱 (정책 진입 직전 상태 확정)
        obs, *_ = env.step(torch.zeros(args_cli.num_envs, 12, device=uenv.device))

    acts = []
    heights = []
    for k in range(args_cli.steps):
        with torch.inference_mode():
            a = policy(obs)
        obs, *_ = env.step(a)
        if k % 25 == 0:
            print(f"[CKPT] step {k}", flush=True)
        acts.append(a[0].detach().cpu().numpy().copy())
        heights.append(float(robot.data.root_pos_w[0, 2] - uenv.scene.env_origins[0, 2]))
    acts = np.array(acts)                      # (T, 12)
    win = acts[:50]
    d = np.diff(win, axis=0)
    flips = ((d[1:] * d[:-1]) < -1e-6).sum(0)  # 관절별 방향반전
    rate = np.abs(np.diff(acts, axis=0)).mean()
    tail_rate = np.abs(np.diff(acts[-50:], axis=0)).mean()
    print(f"\n== {label} ==")
    print(f"raw |max| 첫10틱: {np.abs(acts[:10]).max():.2f} | 전체: {np.abs(acts).max():.2f}")
    print(f"첫 50틱 방향반전(관절별): {list(flips)}")
    print(f"액션레이트 평균: 초반 {rate:.3f} / 종반 {tail_rate:.3f}")
    print(f"골반높이: 시작 {heights[0]:.3f} → 끝 {heights[-1]:.3f} (0.4 미만이면 낙상)")
    print("raw[0] 첫 틱 12관절:", " ".join(f"{v:+.2f}" for v in acts[0]))
    print("raw 틱0~7 (left_hip_a):", " ".join(f"{acts[t, 2]:+.2f}" for t in range(8)))
    return acts


run_scenario("A: sim-native 리셋 (기본자세)", False)
# B(강제 자세 주입)는 PhysX 웨지 이력으로 기본 비활성 — 필요시 --with-b
if "--with-b" in sys.argv:
    run_scenario("B: real-entry (무릎 곧게 편 스탠드)", True)
simulation_app.close()
