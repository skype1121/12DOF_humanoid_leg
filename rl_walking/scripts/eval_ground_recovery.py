#!/usr/bin/env python3
"""지면 절차 평가 (0920 밤, 승윤님 절차) — 홀드 사전 검사 → 같은 홀드 명령에서 인계 → 동일 외란 → 회복(스텝 허용) → 재정지.

절차(실물 0918 확정 절차와 같은 순서):
  1. 지면 스폰(v5 홀드 명령 = env 기본자세, 편향 DR 반영) → --precheck_s 동안 정책 없이 kp150/kd4 홀드 = "무개입 자립 확인".
     사전 검사 통과 = 마지막 0.5 s 양발 접촉·|피치|≤10°·|롤|≤6°·|ω_xy|≤0.3·|v_xy|≤0.1·생존. 실패 env(불안정 초기조건)는 버리지 않고 별도 집계.
  2. 인계: 배포 체인(러너 슬루 → LPF β → 블렌딩)을 **홀드 명령각**(스폰 물리각·엔코더각 아님)에서 시작. --hold 면 인계 없이 홀드 지속(비교 기준).
  3. 외란(--dist, 인계 시각 기준 s): push:방향:시작:힘N:지속s (순간 밀기) · ramp:방향:시작:최대N:상승s:유지s (느린 외력 증가 후 해제).
     방향 b/f/l/r. 기울어진 초기조건은 --tilt_pitch_deg/--tilt_roll_deg (스폰 자세, 자동수평 없음).
  4. 판정: 외란마다 [낙상 / 스텝(발 들기) / 회복(외란 종료 뒤 --recover_s 안에 0.5 s 연속 정지 기준 도달) / 회복 시간 / 최대 기울기],
     정지 구간 불필요 움직임(발 들기·action-rate·명령 부호반전), 발목 한계 의존(한계 1° 이내 체류 비율), 토크 포화.
  홀드 사전 검사 통과율과 인계 후 성공률을 분리해 보고. 등록·배포 없음. 시뮬 결과는 정합 증명이 아니다.

  isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py --headless --task Biped12-HangStand-G6-v0 --num_envs 256 \
      --load_run <run> --checkpoint model_4500.pt [--nominal] [--hold] --dist "push:b:3:60:0.3;push:f:8:60:0.3;ramp:b:13:30:4:2" --out_tag g6_rec
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip  # noqa: E402
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser(description="지면 절차 평가: 홀드 사전검사 → 인계 → 동일 외란 → 회복")
parser.add_argument("--task", type=str, default="Biped12-HangStand-G6-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--hold", action="store_true", help="정책 없이 홀드 지속(비교 기준)")
parser.add_argument("--nominal", action="store_true", help="DR 끔(명목: 식별 플랜트 대역 중심)")
parser.add_argument("--nominal_play_zero", action="store_true")
parser.add_argument("--com_fix", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--precheck_s", type=float, default=3.0, help="홀드 사전 검사 시간 [s]")
parser.add_argument("--after_s", type=float, default=20.0, help="인계 뒤 관측 시간 [s]")
parser.add_argument("--no_chain", action="store_true", help="배포 체인 없이 env 원생 액션(학습 조건 대조)")
parser.add_argument("--lpf_beta", type=float, default=None)
parser.add_argument("--blend_s", type=float, default=None)
parser.add_argument("--blend_offset_deg", type=str, default="", help="인계 **출발 자세**를 정책 기본자세에서 이만큼 띄운다 'hip:2.4,knee:0.9,ankle:3.9' (좌우 거울). 0922 실물 1회차: 홀드 명령이 정책 기본보다 발목 +3.87°·힙 +2.40/−2.81°·무릎 +0.89/−1.72° 높았고, 블렌딩이 그만큼 자세를 내리며 몸이 앞으로 기울었다")
parser.add_argument("--slew_deg", type=float, default=None)
parser.add_argument("--ankle_play_deg", type=float, default=None, help="발목앞뒤 유격 반폭 고정 [deg] (0919 실측 대역 ~8° → 반폭 4). 실물 발산 재현 스윕용")
parser.add_argument("--ankle_coulomb", type=float, default=None, help="발목앞뒤 쿨롱 마찰 [Nm] 고정")
parser.add_argument("--ankle_kp_scale", type=float, default=None, help="발목앞뒤 kp 배율 (명령→몸 응답 이득 저하 재현)")
parser.add_argument("--ankle_stiction", type=float, default=None,
                    help="발목 유격 안 정지 마찰 켜기, 값 = 쿨롱 대비 배율 (0928). 미지정 = 종전(정지 마찰 없음)")
parser.add_argument("--land_s", type=float, default=0.0,
                    help="스폰 직후 부드러운 착지 [s]: 몸통에 체중만큼의 위쪽 지지력을 주고 0 까지 선형 감소 "
                         "(0928: 실물은 줄로 내려 착지해 스폰 충격이 없다). 0=끔")
parser.add_argument("--pb_chain", type=str, default="",
                    help="사슬 합 보존 편향 'lo_deg,hi_deg,leg_noise_deg' — 힙 = 무릎 − 발목 + c + n (0928)")
parser.add_argument("--pb_fixed", type=str, default="",
                    help="관절별 고정 편향 '관절=deg;관절=deg' (관절 프레임 원값, 0929 좌우 비대칭 시험)")
parser.add_argument("--level_s", type=float, default=0.0,
                    help="사전검사 앞에 자동수평 단계 [s] (0928: 실물 절차 착지→자동수평→인계 재현). 0=끔")
parser.add_argument("--level_rate", type=float, default=2.0, help="자동수평 힙 트림 최대 속도 [deg/s] (실물 0.5)")
parser.add_argument("--level_gain", type=float, default=0.5, help="자동수평 비례 이득 [deg 트림 / deg 편차 / 틱]")
parser.add_argument("--pb_set", type=str, default="",
                    help="default_pose_bias 파라미터 직접 지정 'key=lo,hi;key=lo,hi' [rad] (0928 세분 분해)")
parser.add_argument("--com_set", type=str, default="",
                    help="base_com com_range 성분 직접 지정 'x=lo,hi;y=lo,hi;z=lo,hi' [m] (0928 세분 분해)")
parser.add_argument("--dr_off", type=str, default="",
                    help="DR 축을 쉼표로 나열해 그 축만 명목화 (나머지는 DR 유지). "
                         "축: pose,gains,delay,com,posebias,play,mass,friction,calib,asym. "
                         "0928 분해 실험용 — 홀드를 무너뜨리는 축을 단일 변수로 찾는다.")
parser.add_argument("--delay_steps", type=str, default=None, help="액션 지연 lo,hi [틱] (기본: 명목 4,4 / DR 학습 대역)")
parser.add_argument("--dist", type=str, default="", help="외란 목록 ';' 구분, 시각은 인계 기준 [s]")
parser.add_argument("--recover_s", type=float, default=6.0, help="외란 종료 뒤 회복 판정 창 [s]")
parser.add_argument("--tilt_pitch_deg", type=float, default=0.0, help="스폰 몸통 피치 [deg, +앞] (기울어진 초기조건)")
parser.add_argument("--tilt_roll_deg", type=float, default=0.0)
parser.add_argument("--cmd_deadband_deg", type=float, default=0.0, help="배포 체인 출력 뒤 명령 데드밴드 [deg]: 직전 명령 대비 변화가 이보다 작으면 명령을 바꾸지 않는다(정숙용 배포측 레버, 학습 변경 없음)")
parser.add_argument("--freeze_after", type=float, default=None, help="인계 기준 이 시각[s] 이후 **정책 출력 전체를 고정**(마지막 액션 유지) — 흔들림이 명령에서 오는지 플랜트에서 오는지 가르는 대조군")
parser.add_argument("--dump_envs", type=int, default=0)
parser.add_argument("--enc_obs", type=str, default="", choices=("", "legacy", "v2"),
                    help="1001: 정책 관측의 엔코더 관절각·속도·속도 EMA 함수를 강제 — v2 = 실물 배포와 같은 의미(속도 EMA 에 진짜 속도), "
                         "legacy = G4~G15 학습 조건(속도 EMA 에 0+잡음). 비우면 태스크 설정 그대로.")
parser.add_argument("--out_tag", type=str, default="rec")
parser.add_argument("--video", action="store_true", help="측면 카메라 프레임(10 Hz)·mp4 저장 + 힘 화살표 주석 (env --video_env)")
parser.add_argument("--video_env", type=int, default=0)
parser.add_argument("--video_every", type=int, default=5, help="프레임 간격 [틱]")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json                      # noqa: E402
import gymnasium as gym          # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _pair(s):
    lo, hi = (float(v) for v in s.split(","))
    return (lo, hi)


# ── env 설정 ──────────────────────────────────────────────────────────────
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
if args_cli.seed is not None:
    env_cfg.seed = int(args_cli.seed); agent_cfg.seed = int(args_cli.seed)
env_cfg.events.push_robot = None                     # 외란은 이 스크립트가 동일하게 건다
if getattr(env_cfg.events, "disturbance", None) is not None:      # G7: env 자체 외란 스케줄 끔(평가 외란은 --dist 로만)
    env_cfg.events.disturbance = None
for _k in ("onset_hold_range", "onset_blend_range"):              # G7: env 온셋 인계 끔(이 스크립트의 홀드→체인 인계가 절차)
    if hasattr(env_cfg.actions.joint_pos, _k):
        setattr(env_cfg.actions.joint_pos, _k, (0.0, 0.0))
if hasattr(env_cfg.actions.joint_pos, "lpf_beta_range") and not args_cli.no_chain:
    env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)         # 필터는 배포 체인 1개만
_hp = env_cfg.events.hang_support_reset.params
assert float(_hp["r0_range"][1]) <= 1e-6, f"지면 절차 평가는 지지 0 태스크에서만 (r0_range={_hp['r0_range']})"
_DR_AXES = ("pose", "gains", "delay", "com", "posebias", "play", "mass", "friction", "calib", "asym")
_OFF = {x.strip() for x in (args_cli.dr_off or "").split(",") if x.strip()}
_bad = _OFF - set(_DR_AXES)
if _bad:
    raise SystemExit(f"--dr_off 미지의 축 {sorted(_bad)} (허용 {list(_DR_AXES)})")


def _nom(axis):
    """이 축을 명목화하는가 — --nominal 이면 전부, 아니면 --dr_off 에 적힌 축만."""
    return args_cli.nominal or axis in _OFF


if args_cli.nominal or _OFF:
    ev = env_cfg.events
    if _nom("pose"):
        for _k in ("roll", "pitch"):
            if _k in ev.reset_base.params["pose_range"]:
                ev.reset_base.params["pose_range"][_k] = (0.0, 0.0)
        if getattr(ev, "reset_robot_joints", None):
            ev.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    if _nom("gains"):
        ev.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        ev.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    # 0922 정정: 명목 지연을 4틱으로 고정해 온 것은 **실물보다 낙관적**이었다.
    #   실측 루프지연(0911 realdata_model_id §5): 중앙 6.3틱 · IQR 5.7~7.0 · p95 8.9.
    #   지연 스윕에서 발목 명령 최대가 4틱 34.4° → 9틱 51.6° 로 단조 증가하고, 실물 2회차 56.7° 는 9~11틱 구간이다.
    #   → 명목 = 실측 중앙 6틱. 강건성 판정은 --delay_steps 9,9(p95)로 따로 본다.
    if _nom("delay"):
        env_cfg.actions.joint_pos.min_delay_steps = 6; env_cfg.actions.joint_pos.max_delay_steps = 6
        if hasattr(env_cfg.actions.joint_pos, "lpf_beta_range"): env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)
        if hasattr(env_cfg.actions.joint_pos, "backlash_rad_range"): env_cfg.actions.joint_pos.backlash_rad_range = (0.0, 0.0)
    if _nom("com") and getattr(ev, "base_com", None):
        ev.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    if _nom("posebias") and getattr(ev, "default_pose_bias", None):
        _pb = ev.default_pose_bias.params
        for _k in ("hip_f_rad", "knee_rad", "ankle_f_rad"):
            if _pb.get(_k) is not None: _pb[_k] = (0.0, 0.0)
        for _k in ("shared_hip_f_rad", "shared_knee_rad", "shared_ankle_f_rad"):
            if _pb.get(_k) is not None:
                _c = 0.5 * (float(_pb[_k][0]) + float(_pb[_k][1])); _pb[_k] = (_c, _c)
        _pb["reset_noise_rad"] = 0.0
    for _nm, _key in ((("ankle_f_gains", "stiffness_distribution_params") if _nom("gains") else (None, None)),
                      (("ankle_f_friction", "friction_distribution_params") if _nom("friction") else (None, None))):
        _e = getattr(ev, _nm, None) if _nm else None
        if _e is not None:
            _c = 0.5 * (float(_e.params[_key][0]) + float(_e.params[_key][1])); _e.params[_key] = (_c, _c)
    if _nom("calib") and getattr(ev, "calib_bias", None):
        for _k in list(ev.calib_bias.params):
            if _k.endswith("_rad") and ev.calib_bias.params[_k] is not None: ev.calib_bias.params[_k] = (0.0, 0.0)
        if "imu_lag_steps" in ev.calib_bias.params: ev.calib_bias.params["imu_lag_steps"] = (0, 0)
    if _nom("asym"):
        for _e in ("stance_asym", "effort_scale"):
            if getattr(ev, _e, None) is not None: setattr(ev, _e, None)
    if _nom("mass") and getattr(ev, "add_base_mass", None):
        _op = ev.add_base_mass.params.get("operation", "add")
        ev.add_base_mass.params["mass_distribution_params"] = ((1.0, 1.0) if _op == "scale" else (0.0, 0.0))
    if _nom("friction"):
        if getattr(ev, "joint_friction", None): ev.joint_friction.params["friction_distribution_params"] = (0.0, 0.0)
        if getattr(ev, "physics_material", None):
            ev.physics_material.params["static_friction_range"] = (0.8, 0.8); ev.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        if getattr(ev, "joint_armature", None): ev.joint_armature.params["armature_distribution_params"] = (1.0, 1.0)
    from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg as _DZC
    for _k, _a in (list(env_cfg.scene.robot.actuators.items()) if _nom("play") else []):
        if getattr(_a, "play_rad_range", None) is None: continue
        if isinstance(_a, _DZC) and not args_cli.nominal_play_zero:
            _pc = 0.5 * (float(_a.play_rad_range[0]) + float(_a.play_rad_range[1])); _a.play_rad_range = None
            _a.play_rad = ({k: v * _pc for k, v in _a.play_rad.items()} if isinstance(_a.play_rad, dict) else float(_a.play_rad) * _pc)
            if getattr(_a, "coulomb_range", None) is not None:
                _cc = 0.5 * (float(_a.coulomb_range[0]) + float(_a.coulomb_range[1])); _a.coulomb_range = None
                _a.coulomb_nm = ({k: v * _cc for k, v in _a.coulomb_nm.items()} if isinstance(_a.coulomb_nm, dict) else float(_a.coulomb_nm) * _cc)
        else:
            _a.play_rad_range = None; _a.play_rad = 0.0
# 0922 실물 재현 스윕: 발목 유격·쿨롱·kp (명목 고정값으로 덮어씀)
if args_cli.ankle_play_deg is not None or args_cli.ankle_coulomb is not None:
    from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg as _DZC2
    for _k, _a in list(env_cfg.scene.robot.actuators.items()):
        if not isinstance(_a, _DZC2):
            continue
        _a.play_rad_range = None; _a.coulomb_range = None
        if args_cli.ankle_play_deg is not None:
            _a.play_rad = math.radians(float(args_cli.ankle_play_deg))
        if args_cli.ankle_coulomb is not None:
            _a.coulomb_nm = float(args_cli.ankle_coulomb)
        print(f"[plant] {_k}: 유격 {_a.play_rad} rad · 쿨롱 {_a.coulomb_nm} Nm (고정)")
if args_cli.ankle_kp_scale is not None:
    for _k, _a in list(env_cfg.scene.robot.actuators.items()):
        _ex = _a.joint_names_expr if isinstance(_a.joint_names_expr, (list, tuple)) else [_a.joint_names_expr]
        if any("ankle_f" in e for e in _ex):
            _st = _a.stiffness
            _a.stiffness = ({k2: v2 * args_cli.ankle_kp_scale for k2, v2 in _st.items()} if isinstance(_st, dict)
                            else float(_st) * args_cli.ankle_kp_scale)
            print(f"[plant] {_k}: 발목 kp ×{args_cli.ankle_kp_scale} → {_a.stiffness}")
if args_cli.com_fix is not None and getattr(env_cfg.events, "base_com", None):
    env_cfg.events.base_com.params["com_range"]["x"] = (float(args_cli.com_fix), float(args_cli.com_fix))
if args_cli.enc_obs:
    from rl_walking import mdp_hang as _mh
    _P = env_cfg.observations.policy
    _sfx = "_v2" if args_cli.enc_obs == "v2" else "_legacy"
    for _nm, _fn in (("joint_pos", "encoder_joint_pos_rel"), ("joint_vel", "encoder_joint_vel"), ("joint_vel_ema", "encoder_joint_vel_ema")):
        _t = getattr(_P, _nm, None)
        if _t is None:
            continue
        _t.func = getattr(_mh, _fn + _sfx)
        _t.params = {k: v for k, v in dict(_t.params or {}).items() if k in ("alpha", "noise")}
    print(f"[enc_obs] 정책 관측 엔코더 함수 = {args_cli.enc_obs}", flush=True)
if args_cli.delay_steps:
    _dlo, _dhi = _pair(args_cli.delay_steps)
    env_cfg.actions.joint_pos.min_delay_steps = int(_dlo); env_cfg.actions.joint_pos.max_delay_steps = int(_dhi)
if args_cli.ankle_stiction is not None:
    from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg as _DZC2
    _n_st = 0
    for _k, _a in list(env_cfg.scene.robot.actuators.items()):
        if isinstance(_a, _DZC2):
            _a.stiction = True; _a.stiction_ratio = float(args_cli.ankle_stiction); _n_st += 1
    print(f"[stiction] 발목 유격 정지 마찰 ON — 배율 {args_cli.ankle_stiction:g} · 대상 액추에이터 {_n_st}개", flush=True)
    if _n_st == 0:
        raise SystemExit("--ankle_stiction: DeadzoneCoulomb 액추에이터가 없다")
if args_cli.pb_chain:
    _lo, _hi, _nz = (float(x) for x in args_cli.pb_chain.split(","))
    _pbc = env_cfg.events.default_pose_bias.params
    _pbc["chain_target_rad"] = (math.radians(_lo), math.radians(_hi)); _pbc["chain_leg_noise_rad"] = math.radians(_nz)
    print(f"[pb_chain] 사슬 합 목표 {_lo:+.1f}~{_hi:+.1f}° · 다리별 잡음 ±{_nz:.1f}°", flush=True)
if args_cli.pb_fixed:
    _fx = {}
    for _p in [x for x in args_cli.pb_fixed.split(";") if x.strip()]:
        _k, _v = _p.split("="); _fx[_k.strip()] = math.radians(float(_v))
    env_cfg.events.default_pose_bias.params["fixed_bias_rad"] = _fx
    print(f"[pb_fixed] 고정 편향 {args_cli.pb_fixed}", flush=True)
def _parse_set(txt):
    out = {}
    for part in [x for x in (txt or "").split(";") if x.strip()]:
        k, v = part.split("=")
        lo, hi = (float(z) for z in v.split(","))
        out[k.strip()] = (lo, hi)
    return out
if args_cli.pb_set:
    _pbp = env_cfg.events.default_pose_bias.params
    for _k, _v in _parse_set(args_cli.pb_set).items():
        if _k not in _pbp:
            raise SystemExit(f"--pb_set 미지의 키 {_k} (있는 키 {sorted(_pbp)})")
        _pbp[_k] = (float(_v[0]) if _k == "reset_noise_rad" else _v)   # reset_noise_rad 는 스칼라
    print(f"[pb_set] default_pose_bias ← {_parse_set(args_cli.pb_set)}")
if args_cli.com_set:
    _cr = dict(env_cfg.events.base_com.params["com_range"])
    _cr.update(_parse_set(args_cli.com_set))
    env_cfg.events.base_com.params["com_range"] = _cr
    print(f"[com_set] base_com com_range ← {_cr}")
_pr = env_cfg.events.reset_base.params["pose_range"]
if args_cli.tilt_pitch_deg != 0.0:
    _r = math.radians(args_cli.tilt_pitch_deg); _pr["pitch"] = (_r, _r)     # env 규약: +피치 = 앞으로 기울임(projected_gravity x>0) 과 같은 부호로 둔다(아래 실측 표에서 확인)
if args_cli.tilt_roll_deg != 0.0:
    _r = math.radians(args_cli.tilt_roll_deg); _pr["roll"] = (_r, _r)
T_pre = float(args_cli.precheck_s) + float(args_cli.level_s); T_after = float(args_cli.after_s)
env_cfg.episode_length_s = T_pre + T_after + 1.0

if args_cli.video:
    env_cfg.viewer.resolution = (1280, 720)
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=("rgb_array" if args_cli.video else None))
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
uenv = env.unwrapped
robot = uenv.scene["robot"]
dev = uenv.device
N = args_cli.num_envs

policy = None; resume_path = "(hold: 액션 0 지속)"
if not args_cli.hold:
    log_root = os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=dev)

obs = env.get_observations()
_DEF = robot.data.default_joint_pos.clone()      # 정책 기본자세(= 액션 오프셋) — 시뮬 홀드 명령각
_BOFF = torch.zeros_like(_DEF)                  # 인계 출발 자세 오프셋(실물 홀드 명령과 정책 기본의 차이 재현)
if args_cli.blend_offset_deg:
    _jn_all = list(robot.data.joint_names)
    for _tok in args_cli.blend_offset_deg.split(","):
        _k, _v = _tok.split(":"); _r = math.radians(float(_v))
        for _ci, _nm in enumerate(_jn_all):
            _sgn = 1.0 if _nm.startswith("left") else -1.0
            if (_k == "hip" and "hip_f" in _nm) or (_k == "knee" and "knee" in _nm) or (_k == "ankle" and "ankle_f" in _nm):
                _BOFF[:, _ci] = _r * _sgn
    print(f"[blend] 인계 출발 자세 오프셋 {args_cli.blend_offset_deg} (좌우 거울) 적용")
_ASCALE = 0.25
_term = uenv.action_manager.get_term("joint_pos")
_orig_process = _term.process_actions
_chain = None; _chain_on = [False]; _LAST_CMD = [None]; _LAST_RAW = [None]; _DB_PREV = [None]
use_chain = (policy is not None) and (not args_cli.no_chain)
chain_desc = "없음(홀드)" if policy is None else ("없음(env 원생 액션)" if args_cli.no_chain else "")
if use_chain:
    from rl_walking.deploy.deploy_chain import DeployChain, BLEND_S, STAND_LPF_BETA  # noqa: E402
    from rl_walking.deploy.policy_runner import SLEW_RAD  # noqa: E402
    _kw = {}
    if args_cli.lpf_beta is not None: _kw["lpf_beta"] = float(args_cli.lpf_beta)
    if args_cli.blend_s is not None: _kw["blend_s"] = float(args_cli.blend_s)
    if args_cli.slew_deg is not None: _kw["slew_rad"] = math.radians(float(args_cli.slew_deg))
    chain_desc = (f"러너 슬루 {args_cli.slew_deg if args_cli.slew_deg is not None else math.degrees(SLEW_RAD):.1f}°/틱 → LPF β"
                  f"{args_cli.lpf_beta if args_cli.lpf_beta is not None else STAND_LPF_BETA} → 블렌딩 "
                  f"{args_cli.blend_s if args_cli.blend_s is not None else BLEND_S}s (출발 = 홀드 명령각)")

    def _hooked(a):
        _LAST_RAW[0] = _DEF + _ASCALE * a          # 정책이 '요청' 한 관절 목표(체인·클램프 통과 전)
        if not _chain_on[0]:
            _orig_process(a); _LAST_CMD[0] = _term.processed_actions.clone(); return
        out = _chain.step(a)
        if args_cli.cmd_deadband_deg > 0.0:
            _db = math.radians(args_cli.cmd_deadband_deg)
            if _DB_PREV[0] is None:
                _DB_PREV[0] = out.clone()
            else:
                _DB_PREV[0] = torch.where((out - _DB_PREV[0]).abs() >= _db, out, _DB_PREV[0])
            out = _DB_PREV[0]
        _LAST_CMD[0] = out.clone()
        _orig_process((out - _DEF) / _ASCALE)
        _term._raw_actions[:] = a          # 정책 '이전 액션' 관측은 원생 출력(브리지와 동일)
    _term.process_actions = _hooked
else:
    def _hooked2(a):
        _LAST_RAW[0] = _DEF + _ASCALE * a
        _orig_process(a); _LAST_CMD[0] = _term.processed_actions.clone()
    _term.process_actions = _hooked2

# ── 외란 파싱 ──
class Dist:
    def __init__(self, tok):
        p = tok.strip().split(":")
        self.kind = p[0]; self.dir = p[1]; self.t0 = float(p[2]); self.F = float(p[3])
        if self.kind == "push":
            self.dur = float(p[4]); self.t_end = self.t0 + self.dur
        elif self.kind == "ramp":
            self.rise = float(p[4]); self.holdt = float(p[5]); self.t_end = self.t0 + self.rise + self.holdt
        else:
            raise ValueError(tok)
    def force_at(self, t):   # t = 인계 기준 [s]
        if t < self.t0 or t >= self.t_end: return 0.0
        if self.kind == "push": return self.F
        return self.F * min(1.0, (t - self.t0) / max(self.rise, 1e-3))
    def label(self):
        return (f"{self.kind} {self.dir} t={self.t0:.1f}s {self.F:.0f}N " + (f"{self.dur:.2f}s" if self.kind == "push" else f"↑{self.rise:.1f}s 유지 {self.holdt:.1f}s"))

dists = [Dist(t) for t in args_cli.dist.split(";") if t.strip()]
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
_EX = torch.tensor([1.0, 0.0, 0.0], device=dev); _EY = torch.tensor([0.0, 1.0, 0.0], device=dev)
def _dir_vec(d):
    """env 별 방향(월드): 몸통 프레임 +x = 앞(projected_gravity x>0 = 앞 기울임 규약), +y = 왼. 수평 성분만, 정규화.
    (0920 교훈: set_external_force_and_torque 기본은 링크 로컬 프레임이라 월드 벡터를 넘기면 방향이 뒤틀린다 → is_global=True; 스폰 요(≈180°)에 무관하게 몸통 프레임으로 정의)"""
    q = robot.data.root_quat_w
    ax = _EX if d in ("f", "b") else _EY
    v = quat_apply(q, ax.unsqueeze(0).expand(q.shape[0], 3)); v[:, 2] = 0.0
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return v if d in ("f", "l") else -v
_bn = list(robot.data.body_names); _base_id = _bn.index("base") if "base" in _bn else 0

# ── 센서·상수 ──
_cs = uenv.scene["contact_forces"]; _cbn = list(_cs.body_names)
_fl = next(i for i, n in enumerate(_cbn) if n.startswith("left_ankle_r")); _fr = next(i for i, n in enumerate(_cbn) if n.startswith("right_ankle_r"))
jn = list(robot.data.joint_names); J = len(jn)
_eff = robot.data.joint_effort_limits
_lim = robot.data.soft_joint_pos_limits            # (N,J,2)
_rb = list(robot.data.body_names)
_FT = [_rb.index("left_ankle_r_joint"), _rb.index("right_ankle_r_joint")]   # 발 링크(스텝 계측용)
_ank = [k for k, n in enumerate(jn) if "ankle_f" in n]
STEPS_PRE = int(round(T_pre * 50)); STEPS = STEPS_PRE + int(round(T_after * 50))
_blend_ticks = int(round((args_cli.blend_s if args_cli.blend_s is not None else 1.5) * 50)) if use_chain else 0

def _rp():
    pg = robot.data.projected_gravity_b
    return torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])), torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2]))

# ── 0928 자동수평 에뮬레이션 (실물 콘솔 auto_level_tick 과 같은 법칙) ──
#   힙 앞뒤 트림: 좌 +off / 우 −off (off>0 = 굴곡 증가 = 몸통 앞으로). err = 목표(0°) − 피치.
#   양발 접촉 · |편차| ≤ 8°(초과 시 일시정지) · |ω_y| < 0.06 rad/s 일 때만 적분, 0.3° 안이면 정지, |off| ≤ 10°.
#   실물은 0.5°/s 로 ~20 s 걸린다. 시뮬은 기본 2°/s 로 빠르게(--level_rate).
LEVEL_STEPS = int(round(float(args_cli.level_s) * 50))
_lvl = torch.zeros(N, device=dev)                # 트림 [deg]
_lvl_term = uenv.action_manager.get_term("joint_pos")
_jn = list(robot.joint_names)
_jids = getattr(_lvl_term, "_joint_ids", slice(None))
_jlist = list(range(len(_jn))) if isinstance(_jids, slice) else [int(x) for x in _jids]
_LH = _jlist.index(_jn.index("left_hip_f_joint")); _RH = _jlist.index(_jn.index("right_hip_f_joint"))
_lvl_base = None
_lvl_pitch0 = None

_TERM_CNT = {}
# ── 기록 ──
alive = torch.ones(N, dtype=torch.bool, device=dev)
death_step = torch.full((N,), -1, dtype=torch.long, device=dev)
pitch_death = torch.full((N,), float("nan"), device=dev); roll_death = torch.full((N,), float("nan"), device=dev)
pre_both = torch.ones(N, dtype=torch.bool, device=dev); pre_pmax = torch.zeros(N, device=dev); pre_rmax = torch.zeros(N, device=dev)
pre_w = torch.zeros(N, device=dev); pre_v = torch.zeros(N, device=dev); pre_pitch_sum = torch.zeros(N, device=dev); pre_n = 0
prev_both = torch.zeros(N, dtype=torch.bool, device=dev); ever_both = torch.zeros(N, dtype=torch.bool, device=dev)
lift_total = torch.zeros(N, device=dev)
# 외란별
D = len(dists)
d_fell = torch.zeros(N, D, dtype=torch.bool, device=dev); d_lift = torch.zeros(N, D, device=dev); d_rec = torch.zeros(N, D, dtype=torch.bool, device=dev)
d_trec = torch.full((N, D), float("nan"), device=dev); d_pmax = torch.zeros(N, D, device=dev); d_alive0 = torch.zeros(N, D, dtype=torch.bool, device=dev)
d_quiet_run = torch.zeros(N, D, device=dev)
# 정지 구간(외란 없음, 블렌딩+2 s 뒤) 지표
# 0921 스텝 지표 분해(승윤님): 이륙 · 이동 · 재접촉 · 하중 지지 · 회복 성공을 따로, 실패한 시도도 남긴다
st_air = torch.zeros(N, 2, dtype=torch.bool, device=dev)       # 발별 공중 여부
st_t0 = torch.zeros(N, 2, device=dev); st_f0 = torch.zeros(N, 2, device=dev)
st_alive0 = torch.zeros(N, 2, dtype=torch.bool, device=dev)
st_land_t = torch.full((N, 2), -1.0, device=dev)               # 재접촉 시각(하중 관측용)
st_load_sum = torch.zeros(N, 2, device=dev); st_load_n = torch.zeros(N, 2, device=dev)
n_takeoff = torch.zeros(N, device=dev); n_surv_takeoff = torch.zeros(N, device=dev)
n_land = torch.zeros(N, device=dev); n_load = torch.zeros(N, device=dev)
sw_travel = torch.zeros(N, device=dev); sw_n = torch.zeros(N, device=dev)
BW = 17.62 * 9.81
ank_err_sum = torch.zeros(N, device=dev); ank_err_max = torch.zeros(N, device=dev)
ank_cmd_max = torch.zeros(N, device=dev); ank_over = torch.zeros(N, device=dev); ank_n = torch.zeros(N, device=dev)
ank_over_deg = torch.zeros(N, device=dev)   # 관절별 부호 있는 소프트 한계 초과량 최대 [deg]
q_n = torch.zeros(N, device=dev); q_lift = torch.zeros(N, device=dev); q_ar2 = torch.zeros(N, device=dev); q_rev = torch.zeros(N, device=dev)
q_w = torch.zeros(N, device=dev); q_qd2 = torch.zeros(N, device=dev); q_p2 = torch.zeros(N, device=dev); q_pm = torch.zeros(N, device=dev)
# 인계 뒤 전체: 한계 근접·포화
post_n = torch.zeros(N, device=dev); lim_ank = torch.zeros(N, device=dev); lim_any = torch.zeros(N, J, device=dev); sat = torch.zeros(N, device=dev)
final_quiet = torch.zeros(N, dtype=torch.bool, device=dev); fq_run = torch.zeros(N, device=dev)
# 0930: 흔들림 성장/감쇠 판정 — 인계 뒤 1 s 구간마다 env 별 피치 최대-최소 [deg] (살아 있는 틱만). 실물 0929 G12 '느린 흔들림이 커지다 넘어짐' 대조용.
NB_OSC = int(math.ceil(float(T_after)))
osc_max = torch.full((N, NB_OSC), float("-inf"), device=dev); osc_min = torch.full((N, NB_OSC), float("inf"), device=dev)
_geo = {}   # 사전 검사 끝 시점 링크 기울기(실물 폰 수평계 대조용)
# 외란 방향 검증: 힘 창 시작/끝의 피치·롤·골반 위치(앞축 성분) 변화
d_p0 = torch.full((N, D), float("nan"), device=dev); d_p1 = torch.full((N, D), float("nan"), device=dev)
d_r0 = torch.full((N, D), float("nan"), device=dev); d_r1 = torch.full((N, D), float("nan"), device=dev)
d_x0 = torch.full((N, D), float("nan"), device=dev); d_x1 = torch.full((N, D), float("nan"), device=dev)
_VID = None
if args_cli.video:
    from PIL import Image, ImageDraw  # noqa: E402
    _vdir = os.path.join(REPO, "log_picture", f"video_{args_cli.out_tag}"); os.makedirs(_vdir, exist_ok=True)
    _VID = {"dir": _vdir, "n": 0}
    _ve = int(args_cli.video_env); _vo = uenv.scene.env_origins[_ve]
    _cam_state = {}
    def _cam(i):
        # 진짜 측면: 로봇 앞축(몸통 +x, 첫 프레임에서 고정)에 수직인 방향에서 본다 → 앞/뒤 밀기가 화면 좌우로 보임
        if "perp" not in _cam_state:
            fw0 = _dir_vec("f")[_ve]; perp = torch.tensor([-float(fw0[1]), float(fw0[0]), 0.0]); perp = perp / perp.norm()
            _cam_state["perp"] = perp; _cam_state["fw0"] = fw0.clone()
        perp = _cam_state["perp"]
        eye = (float(_vo[0]) + 2.2 * float(perp[0]), float(_vo[1]) + 2.2 * float(perp[1]), float(_vo[2]) + 0.55); look = (float(_vo[0]), float(_vo[1]), float(_vo[2]) + 0.40)
        uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=look)
        vd = torch.tensor([look[0] - eye[0], look[1] - eye[1], 0.0]); vd = vd / vd.norm()
        _cam_state["right"] = torch.tensor([vd[1], -vd[0], 0.0])     # 카메라 오른쪽 = view × up(z) 의 수평 성분
    def _frame(i, t_rel, phase, f_now, fvec_w, pitch, roll, contact):
        _cam(i)
        img = uenv.render()
        if img is None: return
        im = Image.fromarray(img.astype("uint8")); dr = ImageDraw.Draw(im)
        fw = _dir_vec("f")[_ve].cpu(); rt = _cam_state["right"]
        sdx_f = float(fw[0] * rt[0] + fw[1] * rt[1])                       # 앞축의 화면 가로 성분(+ = 오른쪽)
        fwv = fvec_w[_ve, 0].cpu(); sdx_p = float(fwv[0] * rt[0] + fwv[1] * rt[1])
        txt = [f"t={i/50.0:5.2f}s  t_rel={t_rel:+5.2f}s  phase={phase}", f"pitch={float(pitch[_ve]):+6.2f} deg (+ = fwd lean)  roll={float(roll[_ve]):+6.2f}  contact L/R={int(contact[_ve,0])}/{int(contact[_ve,1])}",
               f"robot forward(world xy)=({float(fw[0]):+.2f},{float(fw[1]):+.2f})  => screen {'RIGHT' if sdx_f > 0 else 'LEFT'}",
               f"force={f_now:4.0f} N  world xy=({float(fwv[0]):+.1f},{float(fwv[1]):+.1f})  => screen {'RIGHT' if sdx_p > 0 else ('LEFT' if sdx_p < 0 else '-')}"]
        for k, tline in enumerate(txt): dr.text((12, 10 + 22 * k), tline, fill=(255, 255, 0))
        cx, cy = 640, 120
        gx = 90.0 * (1 if sdx_f > 0 else -1); dr.line((cx, cy, cx + gx, cy), fill=(0, 255, 0), width=4); dr.text((cx + gx * 1.15 - 30, cy - 8), "FWD", fill=(0, 255, 0))
        if f_now != 0.0 and sdx_p != 0.0:
            rx = 120.0 * (1 if sdx_p > 0 else -1)
            dr.line((cx, cy + 40, cx + rx, cy + 40), fill=(255, 60, 60), width=6); dr.text((cx + rx * 1.15 - 30, cy + 32), "PUSH", fill=(255, 60, 60))
        im.save(os.path.join(_vdir, f"f{_VID['n']:05d}.png")); _VID["n"] += 1
_bi = {k: _bn.index(k) for k in ("left_hip_f_joint", "left_knee_joint", "left_ankle_f_joint", "left_ankle_r_joint", "right_hip_f_joint", "right_knee_joint", "right_ankle_f_joint", "right_ankle_r_joint") if k in _bn}
prev_a = None; prev_cmd = None; prev_dcmd = None; _FROZEN = [None]
pitch_tl = []; force_tl = []
_DUMP = []
if args_cli.dump_envs > 0:
    os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
    _DUMP = [open(os.path.join(REPO, "log_picture", f"recbb_{args_cli.out_tag}_env{k}.jsonl"), "w", encoding="utf-8") for k in range(min(args_cli.dump_envs, N))]

QUIET_W, QUIET_V, QUIET_QD, QUIET_TICKS = 0.15, 0.05, 0.30, 25

# 0928: 스폰 시점 편향·CoM 을 env 별로 보존 (낙상 뒤 리셋되면 값이 바뀌므로 루프 전에 떠 둔다)
_BIAS0 = getattr(uenv, "_default_pose_bias", None)
_BIAS0 = _BIAS0.clone() if isinstance(_BIAS0, torch.Tensor) else None
try:
    _COM0 = robot.root_physx_view.get_coms()[:, _base_id, :3].clone().cpu()
except Exception:  # noqa: BLE001
    _COM0 = None
for i in range(STEPS):
    t_rel = (i - STEPS_PRE) / 50.0                          # 인계 기준 시각
    if i == STEPS_PRE and use_chain:                        # 인계: 체인을 홀드 명령각에서 시작
        _land = _DEF + _BOFF          # 인계 출발 자세(= 실물 홀드 명령). 기본은 정책 기본자세와 동일.
        _chain = DeployChain(_DEF, _land, qpos0=_land.clone(), stand=True, **_kw)
        _chain_on[0] = True
    # 외란 힘 (모든 env 동일)
    fvec = torch.zeros(N, 1, 3, device=dev); f_now = 0.0
    if i >= STEPS_PRE:
        for d in dists:
            fN = d.force_at(t_rel)
            if fN != 0.0:
                fvec[:, 0, :] += _dir_vec(d.dir) * fN; f_now += fN
    if float(args_cli.land_s) > 0.0 and i < int(round(float(args_cli.land_s) * 50)) + 1:
        # 0928 수정: 골반 한 점에 주면 무게중심과 어긋나 회전력이 생긴다(0.72 s 에 +20° 전도).
        #   → 링크마다 자기 질량만큼 위로 받친다 = 중력을 잠시 줄이는 것과 같다(회전력 없음).
        if "_LAND_M" not in globals():
            globals()["_LAND_M"] = robot.root_physx_view.get_masses().to(dev)          # (N, B)
        _frac = max(0.0, 1.0 - i / max(1.0, float(args_cli.land_s) * 50))
        _B = _LAND_M.shape[1]
        _fw = torch.zeros(N, _B, 3, device=dev); _fw[..., 2] = _LAND_M * 9.81 * _frac
        _q = robot.data.body_quat_w.reshape(-1, 4)
        _fb = quat_apply_inverse(_q, _fw.reshape(-1, 3)).reshape(N, _B, 3)
        robot.set_external_force_and_torque(_fb, torch.zeros_like(_fb))
    elif float(args_cli.land_s) > 0.0 and i == int(round(float(args_cli.land_s) * 50)) + 1:
        _z = torch.zeros(N, robot.num_bodies, 3, device=dev)
        robot.set_external_force_and_torque(_z, _z)
    if dists:
        # 0920 검증: Isaac Lab 2.3.2 의 is_global=True 가 실제로는 로컬로 걸렸다(1 env 검증에서 뒤 밀기에 골반이 앞으로) → 월드 힘을 base 쿼터니언으로 직접 로컬 변환해 기본 경로(로컬)로 건다
        fvec_b = quat_apply_inverse(robot.data.root_quat_w, fvec[:, 0, :]).unsqueeze(1)
        robot.set_external_force_and_torque(fvec_b, torch.zeros_like(fvec_b), body_ids=[_base_id])
    with torch.inference_mode():
        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else policy(obs)
        if args_cli.freeze_after is not None and policy is not None and t_rel >= args_cli.freeze_after:
            if _FROZEN[0] is None:
                _FROZEN[0] = a.clone(); print(f"[freeze] t_rel={t_rel:.2f}s 정책 출력 고정", flush=True)
            a = _FROZEN[0]
        obs, _, dones, _ = env.step(a)
    term = uenv.termination_manager.terminated
    pitch, roll = _rp()
    w_xy = robot.data.root_ang_vel_b[:, :2].norm(dim=-1); v_xy = robot.data.root_lin_vel_b[:, :2].norm(dim=-1)
    qd_rms = (robot.data.joint_vel ** 2).mean(1).sqrt()
    f = _cs.data.net_forces_w[:, [_fl, _fr], :].norm(dim=-1); contact = f > 5.0
    both = contact[:, 0] & contact[:, 1]
    if i >= STEPS_PRE:
        _ob = min(int(t_rel), NB_OSC - 1); _ok = alive & ~term
        osc_max[:, _ob] = torch.where(_ok, torch.maximum(osc_max[:, _ob], pitch), osc_max[:, _ob])
        osc_min[:, _ob] = torch.where(_ok, torch.minimum(osc_min[:, _ob], pitch), osc_min[:, _ob])
    if LEVEL_STEPS > 0 and i < LEVEL_STEPS:
        if _lvl_base is None:
            _lvl_base = _lvl_term._offset[:, [_LH, _RH]].clone(); _lvl_pitch0 = pitch.clone()
        _err = 0.0 - pitch
        _wy = robot.data.root_ang_vel_b[:, 1].abs()
        _ok = both & (_err.abs() <= 8.0) & (_err.abs() > 0.3) & (_wy < 0.06)
        _step = torch.clamp(float(args_cli.level_gain) * _err, -float(args_cli.level_rate) / 50.0, float(args_cli.level_rate) / 50.0)
        _lvl = torch.clamp(_lvl + torch.where(_ok, _step, torch.zeros_like(_step)), -10.0, 10.0)
        _lvl_term._offset[:, _LH] = _lvl_base[:, 0] + torch.deg2rad(_lvl)
        _lvl_term._offset[:, _RH] = _lvl_base[:, 1] - torch.deg2rad(_lvl)
        if i == LEVEL_STEPS - 1:
            _pa = pitch[alive]
            print(f"[자동수평] {args_cli.level_s:.1f}s 후 생존 {int(alive.sum())}/{N} · 피치 평균 "
                  f"{float(_lvl_pitch0[alive].mean()) if alive.any() else float('nan'):+.2f}° → "
                  f"{float(_pa.mean()) if alive.any() else float('nan'):+.2f}° (표준편차 {float(_pa.std()) if alive.sum()>1 else 0:.2f}) · "
                  f"트림 평균 {float(_lvl[alive].mean()) if alive.any() else 0:+.2f}° 범위 "
                  f"{float(_lvl[alive].min()) if alive.any() else 0:+.1f}~{float(_lvl[alive].max()) if alive.any() else 0:+.1f}°", flush=True)
    cmd = _LAST_CMD[0] if _LAST_CMD[0] is not None else _DEF
    a_ = alive.clone()
    newly = a_ & term
    death_step[newly] = i; pitch_death[newly] = pitch[newly]; roll_death[newly] = roll[newly]
    if bool(newly.any()):     # 0928: 어떤 종료 조건이 걸렸는지 집계 (낙상 판정 검증용)
        _tm = uenv.termination_manager
        for _nm in _tm.active_terms:
            _v = _tm.get_term(_nm)
            _c = int((_v & newly).sum())
            if _c:
                _TERM_CNT[_nm] = _TERM_CNT.get(_nm, 0) + _c
    ever_both |= (a_ & both)
    lift = (a_ & ever_both & prev_both & ~both).float()
    lift_total += lift
    quiet = a_ & both & (w_xy < QUIET_W) & (v_xy < QUIET_V) & (qd_rms < QUIET_QD)
    if i >= STEPS_PRE:      # ── 발목 명령 통계(실물 대조) ──
        _ac = torch.rad2deg(cmd[:, _ank]).abs().max(1).values
        _ae2 = (torch.rad2deg(cmd[:, _ank] - robot.data.joint_pos[:, _ank])).abs().max(1).values
        ank_err_sum[a_] += _ae2[a_]; ank_err_max = torch.maximum(ank_err_max, _ae2 * a_.float())
        ank_cmd_max = torch.maximum(ank_cmd_max, _ac * a_.float())
        # 0922 버그 수정: |명령| 을 두 발목 soft_max 의 절대값 최소(=오른발 +18.5°)와 비교하고 있었다.
        # 오른 발목 가동범위는 [-39.0, +18.5] 이고 실제 명령은 -28° 라 해당 경계는 soft_min 이다.
        # 그 결과 정상 기본자세(±28°)가 매 틱 '한계 초과'로 집계됐다(gs_v6 에서 92.2 %).
        # → 관절별 부호 있는 초과량으로 판정한다(학습 보상 action_limit_excess 와 같은 규약).
        _cmd_ank = torch.rad2deg(cmd[:, _ank])
        _lo_ank = torch.rad2deg(_lim[:, _ank, 0]); _hi_ank = torch.rad2deg(_lim[:, _ank, 1])
        _ov = ((_cmd_ank - _hi_ank).clamp(min=0.0) + (_lo_ank - _cmd_ank).clamp(min=0.0)).max(1).values
        ank_over[a_] += (_ov[a_] > 0.0).float(); ank_over_deg = torch.maximum(ank_over_deg, _ov * a_.float())
        ank_n[a_] += 1.0
    if i >= STEPS_PRE:      # ── 스텝 단계 집계 ──
        _fw = _dir_vec("f"); _fp = robot.data.body_pos_w[:, _FT, :2] - robot.data.root_pos_w[:, None, :2]
        _ffwd = (_fp * _fw[:, None, :2]).sum(-1)               # (N,2) 발-골반 앞축 거리
        for c in (0, 1):
            off = a_ & ever_both & ~st_air[:, c] & ~contact[:, c]
            st_air[off, c] = True; st_t0[off, c] = t_rel; st_f0[off, c] = _ffwd[off, c]; st_alive0[off, c] = True
            n_takeoff += off.float()
            on = a_ & st_air[:, c] & contact[:, c]
            st_air[on, c] = False; st_land_t[on, c] = t_rel; st_load_sum[on, c] = 0.0; st_load_n[on, c] = 0.0
            n_land += on.float()
            sw_travel[on] += (_ffwd[on, c] - st_f0[on, c]); sw_n += on.float()
            # 생존 이륙: 이륙 후 0.3 s 이상 살아 있음(낙상 부수 접촉상실과 구분)
            surv = a_ & st_alive0[:, c] & ((t_rel - st_t0[:, c]) >= 0.3)
            n_surv_takeoff += surv.float(); st_alive0[surv, c] = False
            # 재접촉 후 0.3 s 하중 적분
            win = a_ & (st_land_t[:, c] >= 0) & ((t_rel - st_land_t[:, c]) <= 0.3)
            st_load_sum[win, c] += f[win, c]; st_load_n[win, c] += 1.0
            done = a_ & (st_land_t[:, c] >= 0) & ((t_rel - st_land_t[:, c]) > 0.3) & (st_load_n[:, c] > 0)
            ok = done & ((st_load_sum[:, c] / st_load_n[:, c].clamp(min=1)) >= 0.3 * BW)
            n_load += ok.float(); st_land_t[done, c] = -1.0
    if i < STEPS_PRE:
        if i == STEPS_PRE - 1 and len(_bi) == 8:            # 링크 기울기(수직 기준, + = 아래쪽 끝이 앞): 실물 폰 수평계(허벅지/정강이/발등)와 같은 정의
            _fw = _dir_vec("f"); _pw = robot.data.body_pos_w
            def _tilt(a_, b_):     # a→b 벡터가 아래 수직에서 앞으로 기운 각 [deg]
                v = _pw[:, _bi[b_]] - _pw[:, _bi[a_]]
                return torch.rad2deg(torch.atan2((v * _fw).sum(1), -v[:, 2]))
            _upf = quat_apply(robot.data.body_quat_w[:, _bi["left_ankle_r_joint"]], torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3))   # 발 링크 +Y = 영자세 위(hold_study 규약)
            _geo = {"thigh_L": _tilt("left_hip_f_joint", "left_knee_joint"), "shank_L": _tilt("left_knee_joint", "left_ankle_f_joint"),
                    "thigh_R": _tilt("right_hip_f_joint", "right_knee_joint"), "shank_R": _tilt("right_knee_joint", "right_ankle_f_joint"),
                    "foot_L": torch.rad2deg(torch.atan2((_upf * _fw).sum(1), _upf[:, 2])), "pitch": pitch.clone(),
                    "q_L": torch.rad2deg(robot.data.joint_pos[:, [k for k, n in enumerate(jn) if n in ("left_hip_f_joint", "left_knee_joint", "left_ankle_f_joint")]])}
        if i >= STEPS_PRE - 25:                              # 사전 검사 창 = 마지막 0.5 s
            pre_both &= both | ~a_; pre_pmax = torch.maximum(pre_pmax, pitch.abs()); pre_rmax = torch.maximum(pre_rmax, roll.abs())
            pre_w += w_xy; pre_v += v_xy; pre_pitch_sum += pitch; pre_n += 1
    else:
        post_n[a_] += 1.0
        near = ((robot.data.joint_pos - _lim[:, :, 0]).abs() < math.radians(1.0)) | ((_lim[:, :, 1] - robot.data.joint_pos).abs() < math.radians(1.0))
        lim_any[a_] += near[a_].float(); lim_ank[a_] += near[a_][:, _ank].any(1).float()
        tq = robot.data.applied_torque.abs(); sat[a_] += (tq[a_] >= 0.95 * _eff[a_]).any(1).float()
        # 외란별 창
        for k, d in enumerate(dists):
            if abs(t_rel - d.t0) < 1e-6 or (i == STEPS_PRE and d.t0 <= 0.0):
                d_alive0[:, k] = a_
                _fwk = _dir_vec("f"); d_p0[:, k] = pitch; d_r0[:, k] = roll; d_x0[:, k] = (robot.data.root_pos_w[:, :2] * _fwk[:, :2]).sum(1)
            if abs(t_rel - d.t_end) < 1e-6:
                _fwk = _dir_vec("f"); d_p1[:, k] = pitch; d_r1[:, k] = roll; d_x1[:, k] = (robot.data.root_pos_w[:, :2] * _fwk[:, :2]).sum(1)
            in_win = (t_rel >= d.t0) and (t_rel < d.t_end + args_cli.recover_s)
            if in_win:
                d_fell[:, k] |= newly
                d_lift[:, k] += lift
                d_pmax[:, k] = torch.maximum(d_pmax[:, k], pitch.abs() * a_.float())
                if t_rel >= d.t_end:
                    d_quiet_run[:, k] = torch.where(quiet, d_quiet_run[:, k] + 1.0, torch.zeros_like(d_quiet_run[:, k]))
                    hit = (d_quiet_run[:, k] >= QUIET_TICKS) & ~d_rec[:, k] & a_
                    d_rec[hit, k] = True; d_trec[hit, k] = (t_rel - d.t_end) - (QUIET_TICKS - 1) / 50.0
        # 정지 구간: 블렌딩 + 2 s 뒤이며 어떤 외란 창에도 안 들어감
        in_any = any((t_rel >= d.t0) and (t_rel < d.t_end + args_cli.recover_s) for d in dists)
        if (i - STEPS_PRE) >= _blend_ticks + 100 and not in_any:
            q_n[a_] += 1.0; q_lift += lift; q_w[a_] += w_xy[a_]; q_qd2[a_] += qd_rms[a_] ** 2; q_p2[a_] += pitch[a_] ** 2; q_pm[a_] += pitch[a_]
            if prev_a is not None: q_ar2[a_] += ((a - prev_a)[a_] ** 2).mean(1)
            if prev_cmd is not None:
                dcmd = cmd - prev_cmd
                if prev_dcmd is not None:
                    rev = ((dcmd * prev_dcmd) < 0) & (dcmd.abs() > math.radians(0.3)) & (prev_dcmd.abs() > math.radians(0.3))
                    q_rev[a_] += rev[a_].float().mean(1)
        # 최종 정지(마지막 1 s 연속 정지)
        fq_run = torch.where(quiet, fq_run + 1.0, torch.zeros_like(fq_run))
    if prev_cmd is not None: prev_dcmd = (cmd - prev_cmd).clone()
    prev_cmd = cmd.clone(); prev_a = a.clone(); prev_both = both.clone()
    if _VID is not None and i % max(1, args_cli.video_every) == 0:
        _frame(i, t_rel, "hold" if i < STEPS_PRE else ("policy" if policy is not None else "hold(cont)"), f_now, fvec, pitch, roll, contact)
    for k, fh in enumerate(_DUMP):
        if bool(alive[k]):
            _fwd_k = _dir_vec("f")[k]
            # 스텝 계측: 발-골반 상대 위치를 몸통 앞축/좌축에 투영 [m], 발별 수직 접촉력 [N], 골반 전진 속도
            _rp_k = robot.data.root_pos_w[k]
            _lf_k = torch.tensor([-float(_fwd_k[1]), float(_fwd_k[0])], device=dev)
            _ft = {}
            for _nm, _fbi in (("L", _FT[0]), ("R", _FT[1])):      # 0921: 변수명 _bi 는 상단 링크 기울기용 dict 와 충돌 → _fbi
                _rel = robot.data.body_pos_w[k, _fbi, :2] - _rp_k[:2]
                _ft[f"foot_fwd_{_nm}"] = round(float((_rel * _fwd_k[:2]).sum()), 4)
                _ft[f"foot_lat_{_nm}"] = round(float((_rel * _lf_k).sum()), 4)
            _fz = _cs.data.net_forces_w[k, [_fl, _fr], 2]
            _ft["fz_L"] = round(float(_fz[0]), 1); _ft["fz_R"] = round(float(_fz[1]), 1)
            _ft["v_fwd"] = round(float((robot.data.root_lin_vel_w[k, :2] * _fwd_k[:2]).sum()), 4)
            _ft["base_z"] = round(float(robot.data.root_pos_w[k, 2] - uenv.scene.env_origins[k, 2]), 4)
            _ft["tau"] = [round(float(v), 2) for v in robot.data.applied_torque[k]]
            if _LAST_RAW[0] is not None:
                _ft["raw_deg"] = [round(math.degrees(float(_LAST_RAW[0][k, j])), 2) for j in range(J)]
            _ft["eff"] = [round(float(v), 1) for v in _eff[k]]
            fh.write(json.dumps({"t": round(i / 50.0, 3), "t_rel": round(t_rel, 3), "phase": "hold" if i < STEPS_PRE else ("policy" if policy is not None else "hold_cont"), "force_N": f_now, **_ft,
                                 "force_w": [round(float(v), 2) for v in fvec[k, 0]], "fwd_w": [round(float(v), 3) for v in _fwd_k[:2]], "root_xy": [round(float(v), 4) for v in robot.data.root_pos_w[k, :2]],
                                 "pitch": round(float(pitch[k]), 3), "roll": round(float(roll[k]), 3),
                                 "cmd_deg": [round(math.degrees(float(cmd[k, j])), 2) for j in range(J)],
                                 "q_deg": [round(math.degrees(float(robot.data.joint_pos[k, j])), 2) for j in range(J)],
                                 "contact": [int(contact[k, 0]), int(contact[k, 1])], "w_xy": round(float(w_xy[k]), 3), "v_xy": round(float(v_xy[k]), 3)}) + "\n")
    alive &= ~term
    if i % 25 == 0:
        pitch_tl.append((round(t_rel, 2), float(pitch[alive].mean()) if alive.any() else float("nan"), float(alive.float().mean()), f_now,
                         float(both[alive].float().mean()) if alive.any() else float("nan")))
for fh in _DUMP:
    fh.flush(); fh.close()
final_quiet = alive & (fq_run >= 50)

# ── 판정 ──
fell = death_step >= 0
fell_pre = fell & (death_step < STEPS_PRE)
pre_pass = (~fell_pre) & pre_both & (pre_pmax <= 10.0) & (pre_rmax <= 6.0) & (pre_w / max(pre_n, 1) <= 0.3) & (pre_v / max(pre_n, 1) <= 0.1)
pre_pitch = pre_pitch_sum / max(pre_n, 1)
fell_post = fell & (death_step >= STEPS_PRE)
t_fall_post = (death_step.float() - STEPS_PRE) / 50.0

def pct(m, g):
    n = int(g.sum()); return (100.0 * float((m & g).sum()) / n) if n else float("nan"), n
def mean_g(x, g):
    return float(x[g].mean()) if g.any() else float("nan")

L = []
P = L.append
P("=" * 100)
P(f"지면 절차 평가 — {args_cli.task} | {'홀드(비교 기준)' if policy is None else resume_path}")
P("인자: " + " ".join(sys.argv[1:]))
P(f"env {N} · 사전 검사(홀드) {T_pre:.1f}s → 인계 → {T_after:.1f}s | DR {'OFF(명목·식별 플랜트 중심)' if args_cli.nominal else 'ON'} com_fix={args_cli.com_fix} seed={env_cfg.seed} | 배포 경로: {chain_desc}")
P(f"스폰 기울기: 피치 {args_cli.tilt_pitch_deg:+.1f}° 롤 {args_cli.tilt_roll_deg:+.1f}° | 외란: " + ("; ".join(d.label() for d in dists) if dists else "없음") + f" | 회복 창 {args_cli.recover_s:.0f}s (정지 = 양발접촉·|ω|<{QUIET_W}·|v|<{QUIET_V}·관절속도RMS<{QUIET_QD}, {QUIET_TICKS}틱 연속)")
P("=" * 100)
n_pass = int(pre_pass.sum())
_n_fpre = int(fell_pre.sum()); _n_nocon = int((~pre_both & ~fell_pre).sum()); _n_p10 = int(((pre_pmax > 10) & ~fell_pre).sum()); _n_r6 = int(((pre_rmax > 6) & ~fell_pre).sum())
_n_mov = int(((((pre_w / max(pre_n, 1)) > 0.3) | ((pre_v / max(pre_n, 1)) > 0.1)) & ~fell_pre).sum())
P(f"[1] 홀드 사전 검사(무개입 자립 {T_pre:.1f}s) 통과: {n_pass}/{N} ({100*n_pass/N:.1f} %)  — 홀드 중 낙상 {_n_fpre} · 접촉 상실 {_n_nocon} · 피치>10° {_n_p10} · 롤>6° {_n_r6} · 동요 {_n_mov}")
P(f"    홀드 정착 피치(마지막 0.5 s 평균): 전체 {mean_g(pre_pitch, ~fell_pre):+.2f}° (표준편차 {float(pre_pitch[~fell_pre].std()) if (~fell_pre).sum() > 1 else 0:.2f}) · 통과 env {mean_g(pre_pitch, pre_pass):+.2f}° · 실패 env {mean_g(pre_pitch, ~pre_pass & ~fell_pre):+.2f}°")
if _geo:
    _ok = ~fell_pre
    P(f"    링크 기울기 @사전검사 끝(생존 env 평균, 수직 기준 + = 아랫끝이 앞) [시뮬]: 몸통 피치 {mean_g(_geo['pitch'], _ok):+.2f}° · 허벅지 L {mean_g(_geo['thigh_L'], _ok):+.2f}° / R {mean_g(_geo['thigh_R'], _ok):+.2f}° · 정강이 L {mean_g(_geo['shank_L'], _ok):+.2f}° / R {mean_g(_geo['shank_R'], _ok):+.2f}° · 발등 L {mean_g(_geo['foot_L'], _ok):+.2f}° · 관절 L 힙/무릎/발목 {[round(float(v), 2) for v in _geo['q_L'][_ok].mean(0)]}")
    P("      실물 [측정 0919 폰 수평계, 왼쪽]: IMU −5.6° 에서 허벅지 +33.1 / 정강이 −18.8 / 발등 0.0 ; IMU +4° 에서 +24.9 / −28.2 / 0 (같은 v5 명령)")
groups = [("전체", torch.ones_like(pre_pass)), ("사전 검사 통과", pre_pass), ("사전 검사 실패(불안정 초기조건, 별도 보존)", ~pre_pass)]
for gname, g in groups:
    n = int(g.sum())
    if n == 0:
        P(f"[{gname}] env 0"); continue
    surv = alive & g
    P(f"[{gname}] env {n}: 인계 뒤 생존 {int(surv.sum())}/{n} ({100*int(surv.sum())/n:.1f} %) · 최종 정지(마지막 1 s) {int((final_quiet & g).sum())}/{n} · 인계 뒤 낙상 {int((fell_post & g).sum())}"
      + (f" (낙상 시각 중앙 {float(t_fall_post[fell_post & g].median()):.2f}s, 블렌딩 창 안 {int(((t_fall_post < _blend_ticks / 50.0) & fell_post & g).sum())})" if (fell_post & g).any() else ""))
    for k, d in enumerate(dists):
        g0 = g & d_alive0[:, k]; n0 = int(g0.sum())
        if n0 == 0:
            P(f"    외란 {k+1} [{d.label()}]: 시작 시 생존 0"); continue
        nf = int((d_fell[:, k] & g0).sum()); ns = int(((d_lift[:, k] >= 1) & g0).sum()); nr = int((d_rec[:, k] & g0).sum())
        nrs = int((d_rec[:, k] & (d_lift[:, k] >= 1) & g0).sum()); nrn = int((d_rec[:, k] & (d_lift[:, k] < 1) & g0).sum())
        tr = d_trec[:, k][d_rec[:, k] & g0]
        P(f"    외란 {k+1} [{d.label()}]: 시작 시 생존 {n0} → 낙상 {nf} ({100*nf/n0:.1f} %) · 스텝(발 들기≥1) {ns} ({100*ns/n0:.1f} %) · 회복 {nr} ({100*nr/n0:.1f} %) = 스텝 없이 {nrn} + 스텝 후 {nrs}"
          + (f" · 회복 시간 중앙 {float(tr.median()):.2f}s (p90 {float(tr.quantile(0.9)):.2f})" if nr else "") + f" · 창 내 최대|피치| 평균 {mean_g(d_pmax[:, k], g0):.1f}° · 발 들기 평균 {mean_g(d_lift[:, k], g0):.2f}회")
    gq = g & (q_n > 0)
    if gq.any():
        qn = q_n[gq]
        P(f"    정지 구간(외란 없음, 블렌딩+2 s 뒤; env 평균): 발 들기 {float(q_lift[gq].mean()):.2f}회 · action-rate RMS {float((q_ar2[gq] / qn).sqrt().mean()):.3f} · 명령 부호반전 비율 {float((q_rev[gq] / qn).mean()):.3f}/틱 · |ω| {float((q_w[gq] / qn).mean()):.3f} rad/s · 관절속도 RMS {float((q_qd2[gq] / qn).sqrt().mean()):.3f} · 피치 평균 {float((q_pm[gq] / qn).mean()):+.2f}° RMS {float((q_p2[gq] / qn).sqrt().mean()):.2f}°")
    gp = g & (post_n > 0)
    if gp.any():
        pn = post_n[gp]
        worst = (lim_any[gp] / pn.unsqueeze(1)).mean(0); kw = int(worst.argmax())
        P(f"    한계 의존(인계 뒤, 한계 1° 이내 체류 비율): 발목앞뒤(둘 중 하나) {100*float((lim_ank[gp] / pn).mean()):.1f} % · 최대 관절 {jn[kw]} {100*float(worst[kw]):.1f} % | 토크 포화(어느 관절이든 ≥95 %) {100*float((sat[gp] / pn).mean()):.2f} % 틱")
for k, d in enumerate(dists):
    g0 = d_alive0[:, k] & ~torch.isnan(d_p1[:, k])
    if not g0.any(): continue
    dp = (d_p1[:, k] - d_p0[:, k])[g0]; drl = (d_r1[:, k] - d_r0[:, k])[g0]; dx = (d_x1[:, k] - d_x0[:, k])[g0]
    exp = {"b": "골반 −앞·피치 −(뒤로 기움)", "f": "골반 +앞·피치 +", "l": "롤 +(왼)", "r": "롤 −"}[d.dir]
    ok = {"b": (dx.mean() < 0) and (dp.mean() < 0), "f": (dx.mean() > 0) and (dp.mean() > 0), "l": drl.mean() > 0, "r": drl.mean() < 0}[d.dir]
    P(f"[방향 검증] 외란 {k+1} [{d.label()}] 힘 창 동안(env 평균): 골반 앞축 이동 {100*float(dx.mean()):+.1f} cm · Δ피치 {float(dp.mean()):+.2f}° · Δ롤 {float(drl.mean()):+.2f}° — 기대 {exp} → {'일치' if ok else '불일치!'}")
if float(ank_n.sum()) > 0:
    _ae = (ank_err_sum / ank_n).mean(); _amax = float(ank_cmd_max.max()); _emax = float(ank_err_max.max())
    P(f"[발목 명령] |명령−실측| 평균 {float(_ae):.1f}° · env 최대 {_emax:.1f}° · |명령| 최대 {_amax:.1f}° · "
      f"소프트 한계 초과 틱 {100*float(ank_over.sum()/ank_n.sum()):.1f} % · 초과량 최대 {float(ank_over_deg.max()):.1f}°   "
      f"(0922 실물 2회차: 평균 7.3 · 최대 28.3 · 명령 최대 56.7 · 한계 클램프 43회)")
_tk = int(n_takeoff.sum()); _sv = int(n_surv_takeoff.sum()); _ld = int(n_land.sum()); _lo = int(n_load.sum())
P(f"[스텝 단계 분해] 이륙 {_tk}회 → 생존 이륙(0.3 s 이상 생존) {_sv} → 재접촉 {_ld} → 하중 지지(0.3 s 평균 ≥30 % 체중) {_lo}"
  + (f" · 재접촉한 발의 앞축 이동 평균 {100*float(sw_travel.sum()/sw_n.sum().clamp(min=1)):+.1f} cm" if float(sw_n.sum()) > 0 else "")
  + f" · 실패한 시도(이륙했으나 재접촉 못 함) {_tk - _ld}")
P("시간축 (인계 기준 s | 생존 env 평균 피치 | 생존 비율 | 외란 힘 N | 양발접촉 비율):")
for t, p_, al, fN, bc in pitch_tl:
    P(f"   {t:6.1f}  {p_:+6.2f}  {al:5.2f}  {fN:5.0f}  {bc:5.2f}")
txt = "\n".join(L)
print(txt)
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"ground_recovery_{args_cli.out_tag}.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
P(f"[종료 조건] {_TERM_CNT if _TERM_CNT else '없음'}")
import csv  # noqa: E402
with open(os.path.join(REPO, "log_picture", f"ground_recovery_{args_cli.out_tag}_envs.csv"), "w", newline="") as fh:
    wcsv = csv.writer(fh)
    _bcols = [f"bias_{n}" for n in robot.joint_names] if _BIAS0 is not None else []
    _ccols = ["com_x", "com_y", "com_z"] if _COM0 is not None else []
    wcsv.writerow(["env", "pre_pass", "pre_pitch", "fell", "t_fall_rel", "alive_end", "final_quiet", "lifts_total"] + [f"d{k+1}_{s}" for k in range(D) for s in ("fell", "lifts", "recovered", "t_rec", "pmax")] + _bcols + _ccols + [f"p2p_{k}" for k in range(NB_OSC)] + ["takeoffs", "surv_takeoffs", "swing_travel_mean_cm"])
    for e in range(N):
        row = [e, int(pre_pass[e]), round(float(pre_pitch[e]), 2), int(fell[e]), round(float(t_fall_post[e]), 2) if fell[e] else "", int(alive[e]), int(final_quiet[e]), float(lift_total[e])]
        for k in range(D):
            row += [int(d_fell[e, k]), float(d_lift[e, k]), int(d_rec[e, k]), (round(float(d_trec[e, k]), 2) if d_rec[e, k] else ""), round(float(d_pmax[e, k]), 1)]
        if _BIAS0 is not None:
            row += [round(math.degrees(float(v)), 3) for v in _BIAS0[e]]
        if _COM0 is not None:
            row += [round(float(v), 4) for v in _COM0[e]]
        _pp = (osc_max[e] - osc_min[e]).tolist()
        row += [(round(v, 2) if math.isfinite(v) else "") for v in _pp]
        # 1001: 스텝 계측(인계 뒤, 양발 접지 이후 이륙 수·0.3 s 생존 이륙 수·착지당 발 전후 이동 평균[cm, − = 뒤로])
        row += [float(n_takeoff[e]), float(n_surv_takeoff[e]),
                (round(float(sw_travel[e] / sw_n[e]) * 100.0, 2) if float(sw_n[e]) > 0 else "")]
        wcsv.writerow(row)
if _VID is not None and _VID["n"] > 0:
    import subprocess  # noqa: E402
    _mp4 = os.path.join(REPO, "log_picture", f"video_{args_cli.out_tag}.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(50 // max(1, args_cli.video_every)), "-i", os.path.join(_VID["dir"], "f%05d.png"), "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", _mp4], check=True, timeout=300)
        print(f"[video] {_mp4} ({_VID['n']} frames, {50 // max(1, args_cli.video_every)} fps)")
    except Exception as _e:
        print("[video] ffmpeg 실패:", _e)
env.close()
simulation_app.close()
