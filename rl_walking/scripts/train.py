# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports (공식 rsl_rl 스크립트 폴더의 cli_args 사용)
import sys
sys.path.insert(0, '/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl')
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--gp_coef", type=float, default=0.0,
                    help="G18: SmoothPPO 관측 블록 야코비안 벌점 계수(0=기존 PPO 그대로)")
parser.add_argument("--gp_block", type=str, default="165:225",
                    help="벌점 관측 구간 'a:b'(기본 last_action 이력 165–224) 또는 'all'")
parser.add_argument("--restore_lr", action="store_true", default=False,
                    help="G18: 이어 학습 시 체크포인트 optimizer 의 학습률을 alg.learning_rate 로 복원(rsl_rl 3.1.2 누락, 5.5.1 ppo.py:405)")
parser.add_argument("--c5_stand", type=str, default="",
                    help="PA1(1006): 파이프라인 정렬 회복 학습 — 고정 서기 정책 '<run>:<model_N.pt>'(실험 디렉터리 기준) 또는 .pt 경로. "
                         "비우면 끔(기존 동작 그대로). rl_walking/c5_train.py")
parser.add_argument("--c5_frac", type=float, default=0.75, help="PA1: 파이프라인 env 비율(나머지 = 정책 상시 앵커 env)")
parser.add_argument("--c5_critic_warmup", type=int, default=0, help="PA1: critic 예열 반복 수(mask 전부 0·액터 고정)")
parser.add_argument("--c5_arm_s", type=float, default=1.0,
                    help="PA1: 에피소드 시작 뒤 감지기 무장 지연[s](그동안 G21a 고정, 스폰 과도 오탐 방지)")
parser.add_argument("--c5_no_rescale", action="store_true", default=False,
                    help="PA1: mask advantage 표본비 배율(전/mask, 상한 10) 끔 — 기본은 켬(손실 평균 희석 보정)")
parser.add_argument("--init_expand", type=str, default="",
                    help="W3-B(1006): 관측 확장 웜스타트 '<run>:<model_N.pt>'(실험 디렉터리 기준) 또는 .pt 경로. 체크포인트 run 의 "
                         "params/env.yaml 과 새 env 관측 항을 이름으로 대조 — 기존 항 열 복사·새 항 열 0, 나머지 층 그대로, "
                         "optimizer 새로 시작. 비우면 끔(기존 동작 그대로). --resume 과 동시 사용 불가. rl_walking/obs_expand.py")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# biped12 커스텀 태스크 등록 — 이 스크립트가 속한 저장소(작업트리 포함)의 rl_walking. 본 저장소에서는 종전 경로와 같다.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isfile(os.path.join(_REPO_ROOT, "rl_walking", "__init__.py")):
    _REPO_ROOT = "/home/ryu/humanoid_leg_test1"
sys.path.insert(0, _REPO_ROOT)
import rl_walking  # noqa: F401
if os.environ.get("RL_WALKING_EXTRA_TASKS"):      # 1008: 학습 전용 추가 태스크 등록 모듈(쉼표 구분). 비우면 종전 그대로
    import importlib as _il_extra
    for _m_extra in os.environ["RL_WALKING_EXTRA_TASKS"].split(","):
        _il_extra.import_module(_m_extra.strip()); print(f"[train] 추가 태스크 모듈 {_m_extra.strip()}", flush=True)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.gp_coef > 0:
        import rsl_rl.runners.on_policy_runner as _opr
        from rl_walking.smooth_ppo import SmoothPPO
        SmoothPPO.GP_COEF = float(args_cli.gp_coef)
        SmoothPPO.GP_SL = None if args_cli.gp_block == "all" else slice(*map(int, args_cli.gp_block.split(":")))
        _opr.PPO = SmoothPPO
        print(f"[G18] SmoothPPO 야코비안 벌점 coef={SmoothPPO.GP_COEF} block={SmoothPPO.GP_SL}")
    if args_cli.c5_stand:
        # PA1: 고정 서기 정책 + 감지기 파이프라인 래퍼 + MaskedPPO (꺼져 있으면 이 블록 전체를 건너뜀)
        if agent_cfg.class_name != "OnPolicyRunner" or args_cli.distributed:
            raise ValueError("--c5_stand 는 단일 GPU OnPolicyRunner 에서만")
        from rl_walking import c5_train as _c5
        env = _c5.install(env, agent_cfg, args_cli, log_root_path,
                          resume_path if agent_cfg.resume else None)
    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    if args_cli.init_expand:
        # W3-B: 관측 확장 웜스타트 (꺼져 있으면 이 블록 전체를 건너뜀)
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation" or args_cli.distributed:
            raise ValueError("--init_expand 는 --resume·Distillation·분산 학습과 함께 쓸 수 없음")
        from rl_walking.obs_expand import init_expand_runner
        init_expand_runner(runner, env, args_cli.init_expand, log_root_path, log_dir)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.restore_lr:
            runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
            print(f"[G18] 학습률 복원 = {runner.alg.learning_rate:.2e}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
