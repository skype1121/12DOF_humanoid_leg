"""공중 매달기(행잉) 테스트 사전 시뮬레이션 — 데이터 수집.

실물 테스트 예정 상황 재현: 로봇을 갠트리/하네스에 단단히 고정해 발이 공중에
뜬 상태(fix_root_link=True, base z=1.0m)에서 이식용 정책이 보행 궤적을
만들어내는지 기록. 같은 명령 세트로 지상(ground) 기준 롤아웃도 수집해
analyze_hang.py에서 정량 비교한다.

조건: DR 전부 제거(실물 1대 = 명목 파라미터), 질량만 실물 12kg 정합 스케일.
env 6개 = 명령 케이스 6종 (서기/전진 0.3/전진 0.5/후진/횡보/회전).

실행 (모드당 별도 프로세스 — kit 1env 원칙):
  isaaclab.sh -p rl_walking/scripts/eval_hang.py --headless --mode hang --video
  isaaclab.sh -p rl_walking/scripts/eval_hang.py --headless --mode ground
  Stage4: --task Biped12-Velocity-Stage4-Play-v0 (load_run 기본값 없음 = 최신 런;
          Stage4 PLAY는 액션지연 0틱 — 태스크 cfg 설정을 그대로 따름)
분석 (kit 불필요):
  _isaac_sim/python.sh rl_walking/scripts/analyze_hang.py
"""
import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

DEFAULT_TASK = "Biped12-Velocity-Flat-Play-v0"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["hang", "ground"], required=True)
# eval_policy.py와 동일 패턴: 태스크 등록 kwargs에서 agent/env cfg 해석.
# 기본값은 종전과 동일 (기존 호출 무변경).
parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                    help="평가 태스크 (…-Play-v0). Stage4: Biped12-Velocity-Stage4-Play-v0")
parser.add_argument("--mass_scale", type=float, default=1.144,
                    help="전 링크 질량 스케일 (실물 12kg 정합 = 1.144, 12URDF0725 자중 10.49kg 기준; 구 자산은 1.21)")
parser.add_argument("--settle_secs", type=float, default=2.0)
parser.add_argument("--record_secs", type=float, default=30.0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500, help="영상 스텝 (500=10s)")
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
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

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

REPO = "/home/ryu/humanoid_leg_test1"
HANG_Z = 1.0  # 발끝 지면 여유 ~0.35m

# (이름, vx, vy, wz) — 실물 행잉 테스트에서 시도할 명령 세트
CASES = [
    ("서기 0.0", 0.0, 0.0, 0.0),
    ("전진 0.3", 0.3, 0.0, 0.0),
    ("전진 0.5", 0.5, 0.0, 0.0),
    ("후진 0.3", -0.3, 0.0, 0.0),
    ("횡보 0.2", 0.0, 0.2, 0.0),
    ("회전 0.3+0.5", 0.3, 0.0, 0.5),
]

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
if not args_cli.load_run and args_cli.task == DEFAULT_TASK:
    # 이식용 3단계 최종 (기본값) — biped12_flat 로그에만 존재하는 런이므로 Flat 한정
    agent_cfg.load_run = "2026-07-19_05-32-42"

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = len(CASES)
env_cfg.seed = agent_cfg.seed
env_cfg.episode_length_s = 120.0  # 기록 중 타임아웃 리셋 방지

# 명령: env별 고정 세트를 매 스텝 직접 기록 — heading 제어/리샘플 전부 차단
cmd = env_cfg.commands.base_velocity
cmd.heading_command = False
cmd.rel_heading_envs = 0.0
cmd.rel_standing_envs = 0.0
cmd.resampling_time_range = (1.0e6, 1.0e6)
cmd.ranges.lin_vel_x = (0.0, 0.0)
cmd.ranges.lin_vel_y = (0.0, 0.0)
cmd.ranges.ang_vel_z = (0.0, 0.0)
cmd.ranges.heading = None

# 이식 정책이 학습한 실측 루프 지연(~1틱@50Hz)을 평가에도 적용 (리뷰 반영 —
# PLAY 기본 0틱이면 실물보다 낙관적). 리셋 랜덤(0~1틱)은 rollout에서 1틱 고정.
# Flat 기본 태스크에만 강제 — 다른 태스크(Stage4 PLAY 등)는 해당 cfg의 지연
# 설정을 그대로 따른다 (Stage4 PLAY=0틱 → _delay 버퍼 자체가 없어 rollout에서 스킵).
if args_cli.task == DEFAULT_TASK:
    env_cfg.actions.joint_pos.max_delay_steps = 1

# DR 제거 (실물 1대 = 명목 조건). 질량만 실물 정합 스케일.
ev = env_cfg.events
ev.physics_material = None
ev.base_com = None
ev.actuator_gains = None
ev.joint_friction = None
ev.joint_armature = None
ev.reset_base = None          # 스폰 자세 그대로 (리셋 랜덤화 없음)
ev.reset_robot_joints = None
ev.add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "mass_distribution_params": (args_cli.mass_scale, args_cli.mass_scale),
        "operation": "scale",
    },
)

if args_cli.mode == "hang":
    # 갠트리 고정 근사: 루트 링크를 공중에 강체 고정
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, HANG_Z)

# 영상 카메라: env2(전진 0.5) 근접
env_cfg.viewer.origin_type = "env"
env_cfg.viewer.env_index = 2
if args_cli.mode == "hang":
    env_cfg.viewer.eye = (1.7, 1.7, 1.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.75)
else:
    env_cfg.viewer.eye = (1.7, 1.7, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

video_dir = os.path.join(REPO, "demo_output", f"hang_video_{args_cli.mode}{args_cli.out_tag}")
env = gym.make(args_cli.task, cfg=env_cfg,
               render_mode="rgb_array" if args_cli.video else None)
if args_cli.video:
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length, disable_logger=True,
    )
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)
policy_nn = runner.alg.policy

uenv = env.unwrapped
robot = uenv.scene["robot"]
cs = uenv.scene.sensors["contact_forces"]
cmd_term = uenv.command_manager.get_term("base_velocity")
act_term = uenv.action_manager.get_term("joint_pos")
# 액션 지연 버퍼(_delay)는 max_delay_steps>0일 때만 생성됨 (actions.py) —
# 지연 0 cfg(Stage4 PLAY)에서는 fill_(1) 자체를 스킵해야 AttributeError가 없다.
ACT_DELAY_ON = getattr(act_term, "_max_delay", 0) > 0

FOOT_NAMES = ["left_ankle_r_joint", "right_ankle_r_joint"]  # 발 링크명 = 관절명
foot_ids, _ = robot.find_bodies(FOOT_NAMES, preserve_order=True)
sensor_feet = [cs.body_names.index(n) for n in FOOT_NAMES]  # 센서는 DFS 순서 — 이름으로

N = len(CASES)
dev = uenv.device
CMD = torch.tensor([[c[1], c[2], c[3]] for c in CASES], dtype=torch.float32, device=dev)
S_SETTLE = int(args_cli.settle_secs * 50)
S_RECORD = int(args_cli.record_secs * 50)

rec = {k: [] for k in ("q", "dq", "tau", "act", "foot_b", "ffoot", "root", "quat", "angvel")}
dones_total = 0
max_foot_force = 0.0
dones_env = torch.zeros(N, dtype=torch.long, device=dev)   # env별 낙상/리셋 횟수
first_done_rec = np.full(N, -1, dtype=np.int64)            # 첫 done의 기록스텝 인덱스


def snap(t):
    """t는 텐서 — 기록용 numpy 복사."""
    return t.detach().cpu().numpy().copy()


@torch.inference_mode()
def rollout():
    global dones_total, max_foot_force
    uenv.reset()
    cmd_term.vel_command_b[:] = CMD
    if ACT_DELAY_ON:
        act_term._delay.fill_(1)  # 실측 지연 1틱 고정 (리셋 랜덤 0~1 방지)
    obs = env.get_observations()
    for i in range(S_SETTLE + S_RECORD):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        policy_nn.reset(dones)
        cmd_term.vel_command_b[:] = CMD
        if ACT_DELAY_ON:
            act_term._delay.fill_(1)
        d = dones.view(-1).bool()
        if d.any():
            dones_total += int(d.sum())
            dones_env += d.long()
            for e in torch.nonzero(d).flatten().tolist():
                if first_done_rec[e] == -1:
                    first_done_rec[e] = i - S_SETTLE  # 음수 = settle 중 발생

        ff = cs.data.net_forces_w[:, sensor_feet, :].norm(dim=-1)  # [N,2]
        if args_cli.mode == "hang":
            max_foot_force = max(max_foot_force, float(ff.max()))
        if i < S_SETTLE:
            continue
        bp = (robot.data.body_link_pos_w if hasattr(robot.data, "body_link_pos_w")
              else robot.data.body_pos_w)
        fp = bp[:, foot_ids, :] - robot.data.root_pos_w.unsqueeze(1)
        quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        foot_b = math_utils.quat_apply_inverse(quat, fp.reshape(-1, 3)).reshape(N, 2, 3)
        rec["q"].append(snap(robot.data.joint_pos))
        rec["dq"].append(snap(robot.data.joint_vel))
        rec["tau"].append(snap(robot.data.applied_torque))
        rec["act"].append(snap(actions))
        rec["foot_b"].append(snap(foot_b))
        rec["ffoot"].append(snap(ff))
        rec["root"].append(snap(robot.data.root_pos_w))
        rec["quat"].append(snap(robot.data.root_quat_w))
        rec["angvel"].append(snap(robot.data.root_ang_vel_b))


rollout()

os.makedirs(os.path.join(REPO, "demo_output"), exist_ok=True)
npz_path = os.path.join(REPO, "demo_output", f"hang_traj_{args_cli.mode}{args_cli.out_tag}.npz")
np.savez_compressed(
    npz_path,
    **{k: np.array(v) for k, v in rec.items()},
    joint_names=np.array(robot.data.joint_names),
    foot_names=np.array(FOOT_NAMES),
    case_names=np.array([c[0] for c in CASES]),
    cmds=CMD.cpu().numpy(),
    dt=0.02,
    mode=args_cli.mode,
    task=args_cli.task,
    mass_scale=args_cli.mass_scale,
    checkpoint=resume_path,
    settle_secs=args_cli.settle_secs,
    record_secs=args_cli.record_secs,
    dones_total=dones_total,
    dones_per_env=dones_env.cpu().numpy(),
    first_done_rec=first_done_rec,
    max_foot_force=max_foot_force,
)

# 헤드리스 stdout 유실 대비 — 성공 판정은 이 파일로
sanity = [
    f"mode={args_cli.mode} task={args_cli.task} mass_scale={args_cli.mass_scale} "
    f"checkpoint={resume_path}",
    f"steps: settle {S_SETTLE} + record {S_RECORD} (50Hz), "
    + ("액션지연 1틱 고정" if ACT_DELAY_ON else "액션지연 0틱 (태스크 cfg 기준)"),
    f"dones(리셋) 발생: {dones_total}회 (0이어야 정상) — env별 {dones_env.tolist()}",
    f"기록 스텝 수: {len(rec['q'])}",
]
if args_cli.mode == "hang":
    sanity.append(f"행잉 중 발 접촉력 최대: {max_foot_force:.4f} N (0이어야 진짜 공중)")
if args_cli.video:
    sanity.append(f"영상 폴더: {video_dir}")
sanity.append(f"궤적 저장: {npz_path}")
txt = "\n".join(sanity) + "\n"
open(os.path.join(REPO, "demo_output", f"hang_sanity_{args_cli.mode}{args_cli.out_tag}.txt"),
     "w").write(txt)
print(txt, flush=True)

env.close()
simulation_app.close()
