"""PolicyRunnerStage4 폐루프 시뮬 검증 — 실행기 경로로 245 관측을 조립해 보행 확인.

검증 내용 (sim_validate_runner.py의 Stage4판 — 전부 결정적 판정):
  1. 관측 조립 정합: 러너가 원신호+내부 이력/클록으로 조립한 245차원 벡터를
     env 관측 버퍼(policy 그룹)와 매 틱 직접 대조 — 이력 백필·클록 위상·접촉
     이진화가 학습과 1:1인지 한 번에 검증된다
  2. 슬루 정합: 러너의 슬루 후 목표각 vs env 액션항(processed_actions) 대조
  3. 보행 유지: 15초 무낙상 (전진 거리는 참고 지표)

판정: 관측 최대오차 < 1e-4 AND 목표각 최대오차 < 1e-4 AND 무낙상 → PASS

실행: OMNI_KIT_ACCEPT_EULA=YES /home/ryu/IsaacLab/isaaclab.sh -p \
  rl_walking/deploy/sim_validate_runner_stage4.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--secs", type=float, default=15.0)
parser.add_argument("--model-dir", type=str, default=None,
                    help="exported/ 경로 (기본: walk 기본 모델)")
parser.add_argument("--task", type=str, default="Biped12-Velocity-Stage4-Play-v0")
parser.add_argument("--gait-freq", type=float, default=1.4,
                    help="러너 클록 주파수 [Hz] — 체크포인트 학습 클록과 일치 필수 (v5=1.1)")
parser.add_argument("--march", action="store_true",
                    help="march 검증 모드: 명령 0, 전진 판정 생략")
parser.add_argument("--hang-nominal", action="store_true",
                    help="(0911) HangStand 계약 대조 모드: 관측 노이즈 OFF·EMA 항 내부 노이즈 OFF·접촉 임계 5 N 고정(러너와 동일)·"
                         "액션 지연 0·푸시 OFF·명령 0. 러너 320 조립이 학습 env 관측과 1:1 인지 판정 (DR 로 인한 가짜 오차 제거)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.hang_nominal:
    args_cli.march = True   # 명령 0·전진 판정 생략
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")

import rl_walking  # noqa: F401  (태스크 등록)
from isaaclab.managers import SceneEntityCfg

from rl_walking import mdp_stage4
from rl_walking.deploy.policy_runner_stage4 import (
    CONTACT_FORCE_THRESHOLD_N, DEFAULT_MODEL_DIR_STAGE4, JOINT_ORDER,
    OBS_DIM_STAGE4, PolicyRunnerStage4, _TERMS_BY_DIM, HISTORY_LEN,
)
from rl_walking.stage4_env_cfg import Biped12Stage4FlatEnvCfg_PLAY

_suffix = "_march" if args_cli.march else ""
OUT = ("/home/ryu/humanoid_leg_test1/log_picture/"
       f"실행기_폐루프검증_stage4{_suffix}.txt")
CMD = (0.0, 0.0, 0.0) if args_cli.march else (0.5, 0.0, 0.0)
MODEL_DIR = args_cli.model_dir or DEFAULT_MODEL_DIR_STAGE4

if args_cli.task == "Biped12-Velocity-Stage4-Play-v0":
    env_cfg = Biped12Stage4FlatEnvCfg_PLAY()
else:
    from isaaclab_tasks.utils import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 60.0
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
if args_cli.hang_nominal:
    # 러너는 노이즈·지연·랜덤 임계를 모르므로, 계약 대조에서는 env 쪽 교란을 전부 끈다.
    _P = env_cfg.observations.policy
    _P.enable_corruption = False
    for _nm in ("gyro_ema", "joint_vel_ema"):                      # S8a raw+EMA 항의 내부 노이즈 (float 진폭 또는 cfg 객체)
        _t = getattr(_P, _nm, None)
        if _t is not None and "noise" in _t.params:
            _nz = _t.params["noise"]
            if hasattr(_nz, "n_min"):
                _nz.n_min = 0.0; _nz.n_max = 0.0
            elif isinstance(_nz, (int, float)):
                _t.params["noise"] = 0.0
    _fc = getattr(_P, "feet_contact", None)
    if _fc is not None:
        if "threshold_range" in _fc.params:
            _fc.params["threshold_range"] = (float(CONTACT_FORCE_THRESHOLD_N), float(CONTACT_FORCE_THRESHOLD_N))
        if "threshold" in _fc.params:
            _fc.params["threshold"] = float(CONTACT_FORCE_THRESHOLD_N)
        if "delay_steps" in _fc.params:      # 학습 env 는 접촉 비트를 5틱 지연시킨다(실물 10 Hz S&H 근사) — 조립 등가성 검사에선 0
            _fc.params["delay_steps"] = 0
    # 학습용 센서 손상 모델(항 함수 내부) 무효화 — 조립 등가성만 보기 위해
    _jp = getattr(_P, "joint_pos", None)
    if _jp is not None:
        for _k in ("quant_step_rad", "ankle_dither_rad"):
            if _k in _jp.params: _jp.params[_k] = 0.0
    _jv = getattr(_P, "joint_vel", None)
    if _jv is not None and "zero_prob" in _jv.params:
        _jv.params["zero_prob"] = 0.0
    _cb = getattr(env_cfg.events, "calib_bias", None)          # 영점 편향·IMU 편향·IMU 지연 리샘플 이벤트
    if _cb is not None:
        for _k, _v in list(_cb.params.items()):
            if isinstance(_v, tuple) and len(_v) == 2 and all(isinstance(t, (int, float)) for t in _v):
                _cb.params[_k] = (0, 0) if all(isinstance(t, int) for t in _v) else (0.0, 0.0)
            elif isinstance(_v, dict) and _k == "fixed_bias_deg":
                _cb.params[_k] = {j: 0.0 for j in JOINT_ORDER}
        print(f"[hang-nominal] calib_bias params → {_cb.params}")
    _aj = env_cfg.actions.joint_pos
    if hasattr(_aj, "min_delay_steps"):
        _aj.min_delay_steps = 0; _aj.max_delay_steps = 0
    if hasattr(_aj, "lpf_beta_range"):      # env 액션항의 LPF/백래시 DR — 러너(슬루만)와 목표각 비교를 위해 무효화
        _aj.lpf_beta_range = (1.0, 1.0)
    if hasattr(_aj, "backlash_rad_range"):
        _aj.backlash_rad_range = (0.0, 0.0)
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    print("[hang-nominal] corruption OFF · EMA noise OFF · contact thr 5N · delay 0 · push OFF · cmd 0")

env = gym.make(args_cli.task, cfg=env_cfg)
uenv = env.unwrapped
robot = uenv.scene["robot"]
act_term = uenv.action_manager.get_term("joint_pos")

runner = PolicyRunnerStage4(MODEL_DIR, gait_freq=args_cli.gait_freq)

# 러너 관절순서 == env 관절순서 확인 (이름 기준)
assert list(JOINT_ORDER) == list(robot.data.joint_names), \
    f"관절순서 불일치: {robot.data.joint_names}"

# 접촉 이진 입력 — 학습 관측(_FEET_SENSOR_CFG)과 동일 정의로 센서에서 재현
# ([왼발, 오른발] preserve_order — 접촉센서 바디 나열은 USD DFS라 이름순과 다를 수 있음)
feet_cfg = SceneEntityCfg(
    "contact_forces",
    body_names=["left_ankle_r_joint", "right_ankle_r_joint"],
    preserve_order=True,
)
feet_cfg.resolve(uenv.scene)

rep = [f"PolicyRunnerStage4 폐루프 검증 — 모델 {MODEL_DIR} "
       f"(task={args_cli.task}, clock={args_cli.gait_freq}Hz)",
       f"명령 {CMD}, {args_cli.secs:.0f}s, 관측 {runner.obs_dim}차원(이력5·클록·접촉{'·raw+EMA' if runner.obs_dim != OBS_DIM_STAGE4 else ''})"]
obs_dict, _ = env.reset()
# 0911: 245 하드코딩 → 러너가 모델에서 검출한 차원(245 또는 320 S8a) 과 env 정책 관측 차원을 대조
assert obs_dict["policy"].shape[1] == runner.obs_dim, \
    f"env policy 관측 {obs_dict['policy'].shape[1]} != 러너 {runner.obs_dim} (모델·태스크 불일치)"
cmd_term = uenv.command_manager.get_term("base_velocity")

S = int(args_cli.secs * 50)
target_err_max = 0.0
obs_err_max = 0.0
fell = False
runner.reset()
x0 = None

with torch.inference_mode():
    for i in range(S):
        cmd_term.vel_command_b[:] = torch.tensor([CMD], device=uenv.device)
        gyro = robot.data.root_ang_vel_b[0].cpu().numpy()
        grav = robot.data.projected_gravity_b[0].cpu().numpy()
        qpos = robot.data.joint_pos[0].cpu().numpy()
        qvel = robot.data.joint_vel[0].cpu().numpy()
        contact = mdp_stage4.feet_contact_binary(
            uenv, sensor_cfg=feet_cfg, threshold=CONTACT_FORCE_THRESHOLD_N,
        )[0].cpu().numpy()

        # 러너 스텝 (내부에서 이력 push + 245 조립 + 클록 tick — env와 동주기)
        slewed = runner.step(gyro, grav, np.array(CMD), qpos, qvel, contact)
        raw = runner.last_action.copy()

        # 관측 조립 정합: env가 이 틱에 정책에 줬을 관측(이력 포함)과 직접 대조
        env_obs = obs_dict["policy"][0].cpu().numpy()
        _d = np.abs(env_obs - runner.last_obs)
        obs_err_max = max(obs_err_max, float(_d.max()))
        # 항별 최대 오차 (0911): 어느 블록이 어긋나는지 — 이력 5 × 항 차원 순서 (_term_offsets 와 동일 산술)
        if "_obs_err_term" not in globals():
            _obs_err_term = {}
            _off = 0
            _term_slices = []
            for _tn, _td in _TERMS_BY_DIM.get(runner.obs_dim, ()):
                _term_slices.append((_tn, _off, _off + HISTORY_LEN * _td)); _off += HISTORY_LEN * _td
        for _tn, _a, _b in _term_slices:
            _obs_err_term[_tn] = max(_obs_err_term.get(_tn, 0.0), float(_d[_a:_b].max()))
            if i >= 100:   # 리셋 후 2 s 이후(EMA 재시드 과도·이력 백필 제외)의 정상상태 오차
                _k2 = _tn + "@2s+"
                _obs_err_term[_k2] = max(_obs_err_term.get(_k2, 0.0), float(_d[_a:_b].max()))

        obs_dict, _, terminated, truncated, _ = env.step(
            torch.tensor(raw, dtype=torch.float32, device=uenv.device).unsqueeze(0)
        )
        # env가 처리한 목표각(슬루 후)과 러너 목표각 대조
        env_target = act_term.processed_actions[0].cpu().numpy()
        target_err_max = max(target_err_max, float(np.abs(env_target - slewed).max()))
        if x0 is None:
            x0 = float(robot.data.root_pos_w[0, 0])
        if bool(terminated[0]):
            fell = True
            rep.append(f"!! {i/50:.1f}s 낙상 종료")
            break

dist = float(robot.data.root_pos_w[0, 0]) - (x0 or 0.0)
rep.append(f"전진 거리: {dist:.2f} m ({args_cli.secs:.0f}s, 명령 0.5m/s)")
rep.append(f"러너 vs env 관측 {runner.obs_dim}차원 최대 오차: {obs_err_max:.2e}")
if "_obs_err_term" in globals() and _obs_err_term:
    rep.append("  항별 최대 오차: " + ", ".join(f"{k}={v:.2e}" for k, v in _obs_err_term.items()))
rep.append(f"러너 vs env 목표각 최대 오차: {target_err_max:.2e} rad")
rep.append(f"낙상: {'있음' if fell else '없음'}")
ok = (not fell) and obs_err_max < 1e-4 and target_err_max < 1e-4
rep.append(f"판정: {'PASS' if ok else 'FAIL'}")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)
env.close()
simulation_app.close()
