"""Hanging-to-Stand 평가 — 지지 스케줄 아래서 고정 Kp 홀드 / 정책의 전이 성공률을 잰다 (0908 저녁).

핵심 지표 = **지지가 0 이 된 뒤 5초 생존율**. 그 외: 접지(양발 동시 접촉) 도달률, 지지 0 도달률(살아서),
낙상률·방향, env별 최대 |롤|·|피치| 평균, 최대 각속도, 종료 시 접촉 상태, 종료 시 관절각.
판정은 산출물 파일로 한다 (CLAUDE.md 5): log_picture/hang2stand_<tag>.txt (+ demo_output/hang2stand_<tag>.npz).

실행 (PYTHONPATH 필수 — eval_drift_dr 함정 0908):
  PYTHONPATH=/home/ryu/humanoid_leg_test1 ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_hang2stand.py --headless \
     --task Biped12-HangStand-S1-v0 --hold --num_envs 64 --out_tag s1_hold            # 고정 Kp150 (액션 0)
  ... --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --out_tag s1_v18b   # 정책 (h3 배포체인)
  옵션: --nominal(DR 끔) --com_fix X --r0 lo,hi --hold_s lo,hi --release_s lo,hi --clearance lo,hi --no_chain
"""
import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped12-HangStand-S1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=None, help="기본 = env episode_length_s")
parser.add_argument("--hold", action="store_true", help="정책 없이 액션 0 = DEFAULT_POSE kp150 홀드 (고정 Kp 기준선)")
parser.add_argument("--no_chain", action="store_true", help="배포 체인(러너 슬루→LPF→블렌딩) 끔 — env 원생 액션")
parser.add_argument("--nominal", action="store_true", help="DR 전부 끔 (명목 몸체)")
parser.add_argument("--com_fix", type=float, default=None, help="base_com x 고정 [m]")
parser.add_argument("--ankle_kp_scale", type=float, default=None, help="G1: 발목앞뒤 kp 배율 고정(이벤트 ankle_f_gains) — 가설 플랜트 지정용")
parser.add_argument("--ankle_friction", type=float, default=None, help="G1: 발목앞뒤 관절마찰 고정(이벤트 ankle_f_friction)")
parser.add_argument("--shared_bias", type=str, default=None, help="0918 S9d: 좌우 공유 물리 편향 고정 'hip,knee,ankle' [deg, 왼쪽 기준] (예 '0,-13,0'); --nominal 은 학습 대역 중심으로 고정")
parser.add_argument("--usd", type=str, default="rl_walking/assets/usd_final3/biped12.usd")
parser.add_argument("--r0", type=str, default=None); parser.add_argument("--hold_s", type=str, default=None)
parser.add_argument("--release_s", type=str, default=None); parser.add_argument("--clearance", type=str, default=None)
parser.add_argument("--stand_s", type=float, default=5.0, help="지지 0 이후 생존 판정 시간 [s]")
parser.add_argument("--no_push", action="store_true", help="푸시 외란 끔 (전이 능력과 푸시 강건성 분리 — 실물 윈치 시험엔 푸시가 없다)")
parser.add_argument("--noise_off", type=str, default="", help="관측 노이즈 채널 제거(쉼표): gyro,grav,jpos,jvel,contact,all — S8 원인 분해용")
parser.add_argument("--noise_scale", type=float, default=1.0, help="남은 관측 노이즈 대역 스케일(0.25/0.5/0.75/1.0)")
parser.add_argument("--no_obs_noise", action="store_true", help="관측 노이즈(enable_corruption) 끔 — 정책 고유 정지도 측정용(학습 노이즈 ±0.2rad/s 자이로 등이 잔떨림을 만드는지 분리)")
parser.add_argument("--lpf_beta", type=float, default=None, help="배포 체인 stand LPF β 덮어쓰기 (기본 deploy_chain.STAND_LPF_BETA=0.27; 작을수록 더 부드러움) — 재학습 없이 스탠바이 흔들림을 잡는 배포측 레버 시험")
parser.add_argument("--imu_roll_bias_deg", type=float, default=None, help="IMU 롤 장착 편향을 전 env 상수로 고정 [deg] (0910 실물 재현: 실물 콘솔 lat +1.0~1.4°)")
parser.add_argument("--com_fix_y", type=float, default=0.0, help="횡 CoM 고정 편심 [m] (0910 진단: 실물 편하중 재현)")
parser.add_argument("--contact_perturb", type=str, default="none",
                    help="actor 접촉 obs 교란(정답·물리·보상 불변, 0911 접촉 의존도 평가): "
                         "none/hold(10Hz S&H)/delay3(60ms)/realistic(S&H+한발누락)/onefoot(오른발 버스트0)/right0(오른발 상시0)/right0_after10/left1(왼발 상시1)/left1_after10/"
                         "chatter(랜덤 플립)/both1/both0(진단 고정 대조군)")
parser.add_argument("--perturb_seed", type=int, default=0, help="접촉 교란 RNG seed (독립 seed 평가용)")
parser.add_argument("--seed", type=int, default=None, help="env/torch seed 덮어쓰기 (독립 seed 평가용; 기본 agent_cfg.seed 고정 → 같은 설정은 결정론적 동일 결과)")
parser.add_argument("--delay_steps", type=str, default=None,
                    help="액션 지연 틱 lo,hi 고정 (예 7,7 / 9,9). 실측 정책루프 지연 중앙 6.3틱·p95 8.9틱 대비 민감도; v5 학습 대역 3~8")
parser.add_argument("--dump_envs", type=int, default=0,
                    help="처음 N개 env 의 시계열을 실물 블랙박스 형식으로 log_picture/simbb_<tag>_env<i>.jsonl 에 기록 "
                         "(jetson/scripts/analyze_hang2stand_trial.py 로 시뮬/실물 '최초 이상 발생 순서' 비교, 0911)")
parser.add_argument("--dump_obs", action="store_true", default=False, help="덤프 레코드에 정책 관측 벡터(obs, 320/245) 포함 — 실물 블랙박스 obs 와 항별 대조용(0914)")
parser.add_argument("--flip_joints", type=str, default="", help="0914 좌표계 가설 시험: 지정 관절(쉼표)을 정책 기준으로 물리 반전 — 액션 목표는 기본자세 기준 부호 반전(지연 버퍼·보정기 뒤 = 최종 물리 목표), 관측 joint_pos/joint_vel/joint_vel_ema 의 해당 열도 부호 반전(엔코더가 자기 일관되게 뒤집힌 상황 = 실물 발목롤 추정)")
parser.add_argument("--yaw_damp", type=float, default=0.0, help="0914 공중 응답 기준 시험: 힘 모델에 요 감쇠 토크 −c·ω_z [Nm·s/rad] 추가(sig4 의 요 회전 77 °/s 오염 방지; 기본 0 = 종전 동일)")
parser.add_argument("--joint_offset", type=str, default="", help="0914 부호 확인용: 최종 물리 목표에 상수 오프셋 [deg] 가산 'left_ankle_r_joint:10,right_ankle_r_joint:10' (지연 버퍼·보정기·반전 뒤 = 실물 stage8 에 주는 관절 프레임 목표와 같은 자리)")
parser.add_argument("--rotor_obs", type=str, default="", help="0914 실물 로터측 엔코더 에뮬레이션: 지정 관절(쉼표, all=12관절)의 관측 joint_pos/joint_vel/joint_vel_ema 를 물리 관절각 대신 명령 목표(슬루·지연 뒤 물리 목표) 기반 값으로 대체 — 구조 휨(엔코더 뒤 컴플라이언스)이 정책에 안 보이는 실물 조건. --gain_scale 과 함께 쓰면 휘지만 엔코더는 목표를 따르는 플랜트")
parser.add_argument("--ground_compliance", type=str, default="", help="0914 실물 패드·구조 연성 에뮬레이션: 지면 재질 compliant contact 'stiffness_N_per_m,damping' (PhysX 5 compliant contact; 0 = 강체 접촉=종전)")
parser.add_argument("--rotor_tau", type=float, default=0.05, help="--rotor_obs 의 모터측 각 = 목표의 1차 지연(τ s; 구동기 대역폭 에뮬, 속도는 그 미분). 0 = 목표 그대로(계단 속도 스파이크 — 폐기된 첫 판)")
parser.add_argument("--dump_feet_quat", action="store_true", default=False, help="덤프에 발 링크(*_ankle_r_link) 월드 쿼터니언(wxyz)·발바닥 아래 벡터를 기록 — 관절 부호의 물리 의미 확인")
parser.add_argument("--hook_dxy", type=str, default=None, help="0913: 힘 모델(줄) 작용점을 몸통 로컬 (dx,dy)[m] 로 고정 (y+ = 왼쪽). 실물 후크가 무게중심에서 옆으로 벗어난 경우 재현 — 매 스텝 st[px,py] 덮어씀")
parser.add_argument("--roll_comp", type=str, default=None,
                    help="0913 브리지형 느린 기울기 보정기: 'K,tau_s,limit_deg[,joints[,sign]]' — 몸통 롤(projected gravity, 1차 저역 τ)을 K 배로 관절 목표 오프셋(±limit) 에 더한다(액션 체인 최종 목표 뒤 = 브리지가 최종 명령에 더하는 것과 동일). joints=ankle_r|hip_a, sign=±1. 기본은 양발 진 접촉일 때만 작동(아니면 필터 0).")
parser.add_argument("--roll_comp_lr", type=str, default="1,1", help="보정기 좌/우 관절 부호(예 '1,-1'; hip_a 는 좌우 규약이 반대)")
parser.add_argument("--roll_comp_gate_src", type=str, default="obs", choices=["true", "obs"], help="보정기 게이트의 접촉 출처: true=시뮬 진 접촉력(>5 N), obs=정책이 보는 접촉 비트(교란 포함 = 실물 브리지와 동일)")
parser.add_argument("--roll_comp_site", type=str, default="pre", choices=["pre", "post"], help="보정기 삽입 자리: pre=액션 항 지연 버퍼 앞(오프셋이 전송 지연·슬루·백래시를 통과 = 브리지와 같은 자리, 기본), post=액션 체인 최종 목표 뒤(구동 모델 우회 — 상한 참고용)")
parser.add_argument("--roll_comp_gate", type=str, default="both", choices=["both", "any", "none"], help="보정기 작동 조건: both=양발 접촉, any=한 발, none=항상 (실물 브리지는 none 을 거부한다 — 시뮬 진단 전용)")
parser.add_argument("--roll_comp_debounce_s", type=float, default=0.2, help="보정기 게이트 디바운스 [s] — 접촉 비트가 이 시간 미만으로 끊기면 실효 게이트 유지(필터·오프셋 유지). 브리지 기본 0.2")
parser.add_argument("--roll_comp_slew_deg", type=float, default=0.25, help="보정기 출력 슬루 한계 [deg/틱] — 오프셋 계단 금지(0913 F2). 브리지 기본 0.25 (=12.5°/s)")
parser.add_argument("--roll_comp_roll_clamp_deg", type=float, default=15.0, help="보정기 입력 롤 클램프 [deg] — 공중·낙상 중 큰 기울기가 필터를 끌고 가지 못하게. 브리지 기본 15")
parser.add_argument("--calib_fixed_action", action="store_true", help="--calib_fixed 편향을 관측뿐 아니라 **액션에도**(진 목표 = 정책 목표 − b) 적용 = 실제 영점 오차(브리지가 틀린 영점으로 변환) 재현. 기본은 종전대로 관측만")
parser.add_argument("--calib_fixed", type=str, default="", help="관절 고정 편향 주입 'left_ankle_r_joint:3.6,right_hip_r_joint:3.4' [deg] — 실측 좌우 비대칭 재현")
parser.add_argument("--anchor_x", type=float, default=None, help="[케이블 태스크] 앵커 전후 오프셋 고정 [m] (+앞) — 줄 모멘트(후크 비연직) 가설 H2b 스윕용")
parser.add_argument("--anchor_y", type=float, default=None, help="[케이블 태스크] 앵커 좌우 오프셋 고정 [m]")
parser.add_argument("--anchor_h", type=float, default=None, help="[케이블 태스크] 앵커 높이(후크 위) 고정 [m] (기본 DR 1.0~1.6)")
parser.add_argument("--hook_z", type=float, default=None, help="[케이블 태스크] 후크 높이(base 위) 고정 [m] (기본 DR 0.25~0.40)")
parser.add_argument("--anchor_vx", type=float, default=None, help="[케이블] 앵커 전후 드리프트 속도 [m/s] (시간 가변 당김 가설)")
parser.add_argument("--anchor_vy", type=float, default=None, help="[케이블] 앵커 좌우 드리프트 속도 [m/s]")
parser.add_argument("--ramp_roll_torque", type=str, default=None, help="준정적 롤 토크 램프 'rate[Nm/s],cap[Nm],start[s]' (힘 모델; 실물 준정적 이탈 재현 시도)")
parser.add_argument("--level_torque", action="store_true", help="힘 모델의 수평 유지 토크(lvl_k) 를 켠다. 기본 OFF — 0912 확인: Isaac Lab set_forces_and_torques 가 positions 와 함께 주어진 torques 를 버려 v5 까지의 모든 학습·평가에서 이 토크는 작용한 적이 없다(학습 물리와 같게 두려면 OFF)")
parser.add_argument("--ramp_dr", action="store_true", help="태스크 env 의 준정적 롤 램프 DR(S8ac4q/S8ac4r 계열 ramp_rate_range)을 평가에서도 켠다. 기본 OFF — 0913 정정: 종전엔 S8ac4r env 평가에 램프 DR 이 그대로 섞여(2~12 Nm 무작위) v5 S8ac4 수치와 비교 불가였음")
parser.add_argument("--two_point_d", type=float, default=None, help="앞뒤 2점 현수 후크 간격 d [m] 고정 (0913 승윤님: 줄은 몸통 앞·뒤 2점) — 팽팽한 동안 피치 구속 토크 cap=T·d/2. 기본 None=종전 단일 후크")
parser.add_argument("--joint_play_deg", type=float, default=None, help="관절 기계 유격 반폭 [deg] (전 관절): 링크가 모터 ±play 안에서 자유(강성·감쇠 0) — 0912 재배치 실측 2~3°. 명시적 DeadzonePD 액추에이터로 교체(평가 전용)")
parser.add_argument("--joint_play_lat_deg", type=float, default=None, help="횡 관절(hip_a·ankle_r) 유격 반폭 [deg] — --joint_play_deg 와 함께 쓰면 횡 관절만 이 값(실측: 횡 ≤0.9°, 피치 1.5~3.7°)")
parser.add_argument("--backlash_deg", type=float, default=None, help="액션 항 명령측 백래시(히스테리시스) 반폭 고정 [deg] — 학습 DR 은 0.3~1.5°(S8ac4). 실측 유격 2~3° 를 명령측 모델로 넣었을 때의 감도(기계 유격 DeadzoneImplicit 와 대조). --nominal 뒤에 적용")
parser.add_argument("--ankle_roll_kp_scale", type=float, default=None, help="발목 롤(ankle_roll 액추에이터 그룹)만 kp 스케일 (0913 §3.1: 롤의 4~6° 가 관절 엔코더에 안 보임 → 롤 DOF 실효 강성 저하 가설)")
parser.add_argument("--gain_scale", type=float, default=None, help="액추에이터 강성(kp) 스케일 고정 (전 관절, 감쇠 불변) — 0913: 시뮬 강체 kp150 vs 실물 구조 강성 k_struct 43.5~107(k_eff≈34~62 Nm/rad) 차이 아래에서 정책의 능동 여유(정책 vs --hold 램프 파괴 토크 차) 측정용. --nominal 뒤에 적용")
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _pair(s):
    lo, hi = (float(v) for v in s.split(","))
    return (lo, hi)


agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
if args_cli.seed is not None:   # 독립 seed 평가(0911): 기본은 agent_cfg.seed 고정 → 같은 설정은 결정론적 동일 결과
    env_cfg.seed = int(args_cli.seed); agent_cfg.seed = int(args_cli.seed)
    print(f"[seed] env/agent seed = {args_cli.seed}")
if args_cli.usd:
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd
use_chain = (not args_cli.hold) and (not args_cli.no_chain)
if use_chain and hasattr(env_cfg.actions.joint_pos, "lpf_beta_range"):
    env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)     # 배포 판정: 필터는 체인 1개만
# 정지 서기 명령 · 요 고정 · 초기 속도 0 (실물 착지는 정지)
if args_cli.ground_compliance:
    _gk, _gd = [float(x) for x in args_cli.ground_compliance.split(",")]
    _pm = env_cfg.scene.terrain.physics_material
    if hasattr(_pm, "compliant_contact_stiffness"):
        _pm.compliant_contact_stiffness = _gk; _pm.compliant_contact_damping = _gd
        print(f"[ground_compliance] 지면 compliant contact stiffness={_gk:g} N/m damping={_gd:g}")
    else:
        raise SystemExit("[ground_compliance] RigidBodyMaterialCfg 에 compliant_contact_stiffness 없음 — Isaac Lab 버전 확인")
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
_vr = env_cfg.events.reset_base.params.get("velocity_range")
if isinstance(_vr, dict):
    for _k in list(_vr):
        _vr[_k] = (0.0, 0.0)
if getattr(env_cfg.events, "reset_robot_joints", None):
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
# 지지 스케줄 오버라이드
_hp = env_cfg.events.hang_support_reset.params
if args_cli.r0: _hp["r0_range"] = _pair(args_cli.r0)
if args_cli.hold_s: _hp["hold_range"] = _pair(args_cli.hold_s)
if args_cli.release_s: _hp["release_range"] = _pair(args_cli.release_s)
if not args_cli.level_torque:
    _hp["level_gain_range"] = (0.0, 0.0); _hp["level_damp_range"] = (0.0, 0.0)   # 학습 당시 실효 물리(토크 미적용)와 동일
if args_cli.two_point_d is not None:
    _hp["two_point_d_range"] = (float(args_cli.two_point_d), float(args_cli.two_point_d))
if not args_cli.ramp_dr and _hp.get("ramp_rate_range") is not None:
    _hp["ramp_rate_range"] = None; _hp["ramp_cap_range"] = None; _hp["ramp_start_range"] = None; _hp["ramp_sign_random"] = False
    print("[ramp] 태스크의 램프 DR 을 평가에서 끔 (--ramp_dr 로 유지 가능; --ramp_roll_torque 고정값은 별개)")
if args_cli.ramp_roll_torque:
    _v = [float(x) for x in args_cli.ramp_roll_torque.split(",")]; assert len(_v) == 3
    _hp["ramp_roll_torque"] = tuple(_v)
if args_cli.yaw_damp:
    _hp["yaw_damp"] = float(args_cli.yaw_damp)
for _nm, _key in (("anchor_x", "anchor_x_range"), ("anchor_y", "anchor_y_range"), ("anchor_h", "anchor_h_range"), ("hook_z", "hook_z_range"), ("anchor_vx", "anchor_vx_range"), ("anchor_vy", "anchor_vy_range")):
    _v = getattr(args_cli, _nm)
    if _v is not None:
        assert _hp.get("cable", False), f"--{_nm} 는 케이블 태스크(cable=True)에서만 의미가 있다"
        _hp[_key] = (float(_v), float(_v))
if args_cli.clearance: env_cfg.events.reset_base.params["pose_range"]["z"] = _pair(args_cli.clearance)
if args_cli.nominal:
    for _k in ("roll", "pitch"):
        if _k in env_cfg.events.reset_base.params["pose_range"]:
            env_cfg.events.reset_base.params["pose_range"][_k] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_robot_joints", None):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    env_cfg.actions.joint_pos.min_delay_steps = 4
    env_cfg.actions.joint_pos.max_delay_steps = 4
    if hasattr(env_cfg.actions.joint_pos, "lpf_beta_range"):
        env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)
    if hasattr(env_cfg.actions.joint_pos, "backlash_rad_range"):
        env_cfg.actions.joint_pos.backlash_rad_range = (0.0, 0.0)
    if getattr(env_cfg.events, "base_com", None):
        env_cfg.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    if getattr(env_cfg.events, "default_pose_bias", None):
        # 0918: 독립 편향(영점 오차) 0, 공유 편향(식별된 직렬 처짐)은 대역 중심 = 명목 플랜트
        _pb = env_cfg.events.default_pose_bias.params
        for _k in ("hip_f_rad", "knee_rad", "ankle_f_rad"):
            if _pb.get(_k) is not None:
                _pb[_k] = (0.0, 0.0)
        for _k in ("shared_hip_f_rad", "shared_knee_rad", "shared_ankle_f_rad"):
            if _pb.get(_k) is not None:
                _c = 0.5 * (float(_pb[_k][0]) + float(_pb[_k][1])); _pb[_k] = (_c, _c)
        _pb["reset_noise_rad"] = 0.0
    for _nm, _key in (("ankle_f_gains", "stiffness_distribution_params"), ("ankle_f_friction", "friction_distribution_params")):
        _ev = getattr(env_cfg.events, _nm, None)
        if _ev is not None:
            _c = 0.5 * (float(_ev.params[_key][0]) + float(_ev.params[_key][1])); _ev.params[_key] = (_c, _c)   # 명목 = 대역 중심(G1 가설 플랜트)
    if getattr(env_cfg.events, "calib_bias", None):
        for _k in list(env_cfg.events.calib_bias.params):
            if _k.endswith("_rad") and env_cfg.events.calib_bias.params[_k] is not None:
                env_cfg.events.calib_bias.params[_k] = (0.0, 0.0)
        if "imu_lag_steps" in env_cfg.events.calib_bias.params:
            env_cfg.events.calib_bias.params["imu_lag_steps"] = (0, 0)
    for _ev in ("stance_asym", "effort_scale"):
        if getattr(env_cfg.events, _ev, None) is not None:
            setattr(env_cfg.events, _ev, None)
    if getattr(env_cfg.events, "add_base_mass", None):
        _op = env_cfg.events.add_base_mass.params.get("operation", "add")
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = ((1.0, 1.0) if _op == "scale" else (0.0, 0.0))
    if getattr(env_cfg.events, "joint_friction", None):
        env_cfg.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.0)
    if getattr(env_cfg.events, "physics_material", None):
        env_cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
    if getattr(env_cfg.events, "joint_armature", None):
        env_cfg.events.joint_armature.params["armature_distribution_params"] = (1.0, 1.0)
    env_cfg.events.push_robot = None
if args_cli.no_push:
    env_cfg.events.push_robot = None
if args_cli.no_obs_noise:
    env_cfg.observations.policy.enable_corruption = False
_off = {t.strip() for t in args_cli.noise_off.split(",") if t.strip()}
_P = env_cfg.observations.policy
for _nm, _term in (("gyro", _P.base_ang_vel), ("grav", _P.projected_gravity), ("jpos", _P.joint_pos), ("jvel", _P.joint_vel)):
    if _nm in _off or "all" in _off:
        _term.noise = None
    elif _term.noise is not None and args_cli.noise_scale != 1.0:
        _term.noise.n_min = float(_term.noise.n_min) * args_cli.noise_scale
        _term.noise.n_max = float(_term.noise.n_max) * args_cli.noise_scale
for _nm, _ch in (("gyro_ema", "gyro"), ("joint_vel_ema", "jvel")):          # S8 raw+EMA 항: 내부 노이즈도 같은 규칙
    _t = getattr(_P, _nm, None)
    if _t is not None and "noise" in _t.params:
        if _ch in _off or "all" in _off or args_cli.no_obs_noise:
            _t.params["noise"] = 0.0
        else:
            _t.params["noise"] = float(_t.params["noise"]) * args_cli.noise_scale
if ("contact" in _off or "all" in _off) and "threshold_range" in _P.feet_contact.params:
    _P.feet_contact.params["threshold_range"] = None
if _off or args_cli.noise_scale != 1.0:
    print(f"[noise] off={sorted(_off)} scale={args_cli.noise_scale}")
if args_cli.calib_fixed and getattr(env_cfg.events, "calib_bias", None):
    _fb = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args_cli.calib_fixed.split(",") if ":" in kv}
    env_cfg.events.calib_bias.params["fixed_bias_deg"] = _fb
    print(f"[calib] 고정 편향 주입 {_fb}")
if args_cli.imu_roll_bias_deg is not None and getattr(env_cfg.events, "calib_bias", None):
    _rb = math.radians(args_cli.imu_roll_bias_deg)
    env_cfg.events.calib_bias.params["imu_roll_rad"] = (_rb, _rb)
    print(f"[imu] 롤 편향 고정 {args_cli.imu_roll_bias_deg:+.2f}°")
if args_cli.ankle_kp_scale is not None and getattr(env_cfg.events, "ankle_f_gains", None) is not None:
    env_cfg.events.ankle_f_gains.params["stiffness_distribution_params"] = (args_cli.ankle_kp_scale, args_cli.ankle_kp_scale)
if args_cli.ankle_friction is not None and getattr(env_cfg.events, "ankle_f_friction", None) is not None:
    env_cfg.events.ankle_f_friction.params["friction_distribution_params"] = (args_cli.ankle_friction, args_cli.ankle_friction)
if args_cli.shared_bias is not None and getattr(env_cfg.events, "default_pose_bias", None):
    _h, _k, _a = [math.radians(float(x)) for x in args_cli.shared_bias.split(",")]
    env_cfg.events.default_pose_bias.params.update({"shared_hip_f_rad": (_h, _h), "shared_knee_rad": (_k, _k), "shared_ankle_f_rad": (_a, _a)})
    print(f"[shared_bias] 공유 물리 편향 고정 hip {math.degrees(_h):+.1f}° knee {math.degrees(_k):+.1f}° ankle {math.degrees(_a):+.1f}° (왼쪽 기준)")
if getattr(env_cfg.events, "base_com", None) and (args_cli.com_fix is not None or args_cli.com_fix_y):
    if args_cli.com_fix is not None:
        # 종전 동작 유지: x 고정 + y 고정(com_fix_y, 기본 0) + z 0  (results_table 비교 호환)
        _cy = args_cli.com_fix_y or 0.0
        env_cfg.events.base_com.params["com_range"] = {"x": (args_cli.com_fix, args_cli.com_fix), "y": (_cy, _cy), "z": (0.0, 0.0)}
    else:
        # ★정정(0911, COMY_FIX): --com_fix_y 단독은 종전엔 --com_fix 없이는 통째로 무시됐다(죽은 인자 —
        # ±0.01/±0.02 4런이 기준선과 자릿수까지 동일해 발견). x·z 는 env 범위 유지, y 만 고정.
        _cr = dict(env_cfg.events.base_com.params.get("com_range", {}))
        _cr["y"] = (args_cli.com_fix_y, args_cli.com_fix_y)
        env_cfg.events.base_com.params["com_range"] = _cr
    print(f"[com] base_com com_range = {env_cfg.events.base_com.params['com_range']}")
def _play_map(names_expr, play_deg, lat_deg):
    """액추에이터 그룹의 관절 정규식 → {관절유형 정규식: rad}. 그룹이 매칭하는 유형만 넣는다(미매칭 키는 Isaac Lab 이 거부).
    횡 관절(hip_a·ankle_r)은 lat_deg 가 주어지면 그 값."""
    import re as _re
    types = ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r")
    exprs = names_expr if isinstance(names_expr, (list, tuple)) else [names_expr]
    out = {}
    for t in types:
        if any(_re.fullmatch(ex, f"left_{t}_joint") for ex in exprs):
            v = lat_deg if (lat_deg is not None and t in ("hip_a", "ankle_r")) else play_deg
            out[f".*_{t}_joint"] = math.radians(float(v))
    return out


if args_cli.backlash_deg is not None and hasattr(env_cfg.actions.joint_pos, "backlash_rad_range"):
    _bl = math.radians(float(args_cli.backlash_deg)); env_cfg.actions.joint_pos.backlash_rad_range = (_bl, _bl)
    print(f"[backlash] 명령측 백래시 고정 ±{args_cli.backlash_deg:.2f}°")
if args_cli.ankle_roll_kp_scale is not None:
    _ar = env_cfg.scene.robot.actuators.get("ankle_roll")
    assert _ar is not None, "ankle_roll 액추에이터 그룹 없음"
    _ar.stiffness = float(_ar.stiffness) * float(args_cli.ankle_roll_kp_scale)
    print(f"[ankle_roll] kp ×{args_cli.ankle_roll_kp_scale:.2f} → {_ar.stiffness:.1f} (kd 불변)")
if args_cli.joint_play_deg is not None:
    from rl_walking.actuators_play import DeadzoneImplicitActuatorCfg
    _acts = env_cfg.scene.robot.actuators
    for _k, _a in list(_acts.items()):
        _acts[_k] = DeadzoneImplicitActuatorCfg(joint_names_expr=_a.joint_names_expr,
                                               effort_limit_sim=getattr(_a, "effort_limit_sim", None), velocity_limit_sim=getattr(_a, "velocity_limit_sim", None),
                                               stiffness=_a.stiffness, damping=_a.damping, armature=getattr(_a, "armature", None), friction=getattr(_a, "friction", None),
                                               play_rad=_play_map(_a.joint_names_expr, float(args_cli.joint_play_deg), args_cli.joint_play_lat_deg))
    print(f"[play] 관절 유격 ±{args_cli.joint_play_deg:.2f}° (횡 {args_cli.joint_play_lat_deg}) (DeadzoneImplicit 내재 드라이브 목표 이동, 그룹 {list(_acts)}; 유격 안 감쇠 = 내재 kd)")
if args_cli.nominal:
    for _k, _a in list(env_cfg.scene.robot.actuators.items()):   # 0913 감사: S8ac4rp 계열의 내장 유격 DR 도 명목에서 끔(고정 유격은 --joint_play_deg 로만)
        if getattr(_a, "play_rad_range", None) is not None and args_cli.joint_play_deg is None:
            _a.play_rad_range = None; _a.play_rad = 0.0; print(f"[nominal] 액추에이터 {_k} 유격 DR OFF")
if args_cli.gain_scale is not None and getattr(env_cfg.events, "actuator_gains", None):
    _gs = float(args_cli.gain_scale)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (_gs, _gs)
    print(f"[gain] 액추에이터 강성 스케일 고정 ×{_gs:.2f} (감쇠 불변)")
if getattr(args_cli, "delay_steps", None):
    # 0911 실측(realdata_model_id.md §5): 정책루프 지연 중앙 6.3틱(IQR 5.7~7.0, p95 8.9) vs 학습 대역 3~8틱 → 상단 민감도
    _dlo, _dhi = _pair(args_cli.delay_steps)
    env_cfg.actions.joint_pos.min_delay_steps = int(_dlo); env_cfg.actions.joint_pos.max_delay_steps = int(_dhi)
    print(f"[delay] action delay steps = {int(_dlo)}..{int(_dhi)}")
T_s = args_cli.seconds if args_cli.seconds else float(env_cfg.episode_length_s)
env_cfg.episode_length_s = T_s + 1.0      # 평가 창 안에서 time_out 이 끼지 않게

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
uenv = env.unwrapped
if args_cli.calib_fixed and args_cli.calib_fixed_action:
    assert args_cli.nominal, "--calib_fixed_action 은 --nominal 과 함께만(DR ON 이면 무작위 편향 벡터 전체가 액션에도 빠져 의미가 달라짐 — 0913 감사)"
    # 0913: 진짜 영점 오차 = 관측 +b 이면서 명령은 −b (정책 프레임에서 일관). 액션 항의 최종 목표(슬루·LPF·백래시 뒤)에서 b 를 뺀다.
    _zt = uenv.action_manager.get_term("joint_pos")
    _zo = _zt.process_actions
    def _z_patched(actions, _t=_zt, _o=_zo):
        _o(actions)
        _b = getattr(uenv, "_calib_joint_bias", None)
        if _b is not None:
            _t._processed_actions = _t._processed_actions - _b[:, _t._joint_ids]
    _zt.process_actions = _z_patched
    print("[calib] 고정 편향을 액션에도 적용(진 목표 = 정책 목표 − b) — 실제 영점 오차 재현")
robot = uenv.scene["robot"]
N = args_cli.num_envs
dev = uenv.device
STEPS = int(round(T_s * 50))

policy = None
resume_path = "(hold: 액션 0)"
if not args_cli.hold:
    log_root = os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=dev)

obs = env.get_observations()
_DEF = robot.data.default_joint_pos.clone()
_LAND = torch.zeros_like(_DEF)
_ASCALE = 0.25
chain_desc = "없음(env 원생 액션)"
_LAST_OUT = [None]   # 배포 체인이 마지막으로 보낸 관절 목표(rad) — 시뮬 블랙박스 덤프용
if use_chain:
    from rl_walking.deploy.deploy_chain import DeployChain, BLEND_S, STAND_LPF_BETA  # noqa: E402
    from rl_walking.deploy.policy_runner import SLEW_RAD  # noqa: E402
    _term = uenv.action_manager.get_term("joint_pos")
    _chain = DeployChain(_DEF, _LAND, qpos0=robot.data.joint_pos.clone(), stand=True, **({"lpf_beta": args_cli.lpf_beta} if args_cli.lpf_beta is not None else {}))
    _orig = _term.process_actions

    def _hooked(a):
        if torch.any(_term._needs_init):
            _ids = torch.nonzero(_term._needs_init).squeeze(-1)
            _chain.reset(_ids, robot.data.joint_pos)
        out = _chain.step(a)
        _LAST_OUT[0] = out          # 덤프용: 배포 체인이 실제로 보낸 관절 목표(rad)
        _orig((out - _DEF) / _ASCALE)
        _term._raw_actions[:] = a

    _term.process_actions = _hooked
    chain_desc = f"h3 chain: 러너 슬루 {math.degrees(SLEW_RAD):.1f}°/틱 → LPF β{args_cli.lpf_beta if args_cli.lpf_beta is not None else STAND_LPF_BETA} → 블렌딩 {BLEND_S}s"

# ── 기록 (첫 에피소드만 — 종료 후 자동 리셋된 env 의 새 에피소드는 무시) ──
alive = torch.ones(N, dtype=torch.bool, device=dev)
death_step = torch.full((N,), -1, dtype=torch.long, device=dev)
touch_t = torch.full((N,), -1.0, device=dev); rel_t = torch.full((N,), -1.0, device=dev)
max_roll = torch.zeros(N, device=dev); max_pitch = torch.zeros(N, device=dev); max_w = torch.zeros(N, device=dev)
pitch_at_rel = torch.full((N,), float("nan"), device=dev)
pitch_death = torch.full((N,), float("nan"), device=dev); roll_death = torch.full((N,), float("nan"), device=dev)
r_at_death = torch.full((N,), float("nan"), device=dev)
ramp_at_death = torch.full((N,), float("nan"), device=dev)   # 0913: 종료(낙상) 순간의 램프 롤 토크 [Nm] — 정책 vs 홀드 능동 여유
ratio_trace = []; contact_trace = []; z_trace = []; pitch_trace = []
last_q = torch.zeros_like(robot.data.joint_pos); last_contact = torch.zeros(N, 2, device=dev)
_cs = uenv.scene["contact_forces"]; _bn = list(_cs.body_names)
_fl = _bn.index("left_ankle_r_joint"); _fr = _bn.index("right_ankle_r_joint")
# 물리 실현성 (0908 승윤님 "접지 자세가 실물로 될까"): 지지 0 이후 생존 스텝의 관절 토크 피크·RMS·포화(≥95% 한계), 발 간격
_rb = list(robot.data.body_names); _bl, _br = _rb.index("left_ankle_r_joint"), _rb.index("right_ankle_r_joint")
_nj = robot.data.joint_pos.shape[1]; _eff = robot.data.joint_effort_limits
_tau_peak = torch.zeros(_nj, device=dev); _tau_rms2 = torch.zeros(_nj, device=dev); _tau_sat = torch.zeros(_nj, device=dev); _tau_n = 0
_stance_w = []; _stance_dx = []
# 정지도 (0909 승윤님 "착지 후 흔들림 불가"): 지지 0 이후 생존 스텝의 |ω_xy| 평균·|v_xy| 평균·관절속도 RMS·발 들기 횟수(양발 접촉 후 한 발 이탈 전이)
_st_w = torch.zeros(N, device=dev); _st_v = torch.zeros(N, device=dev); _st_qd2 = torch.zeros(N, device=dev); _st_n = torch.zeros(N, device=dev)
_lift_cnt = torch.zeros(N, device=dev); _prev_both = torch.zeros(N, dtype=torch.bool, device=dev); _ever_both = torch.zeros(N, dtype=torch.bool, device=dev)
_ss_w = torch.zeros(N, device=dev); _ss_v = torch.zeros(N, device=dev); _ss_qd2 = torch.zeros(N, device=dev); _ss_n = torch.zeros(N, device=dev); _ss_lift = torch.zeros(N, device=dev)
_SS_FROM = STEPS - 150   # 마지막 3s = 정상상태(스탠바이) 구간
_ss_ar2 = torch.zeros(N, device=dev); _ss_r2 = torch.zeros(N, device=dev); _ss_p2 = torch.zeros(N, device=dev); _ss_rpk = torch.zeros(N, device=dev); _ss_ppk = torch.zeros(N, device=dev); _prev_a = None
_ss_aa2 = torch.zeros(N, device=dev); _ss_pm = torch.zeros(N, device=dev); _prev_a2 = None   # 액션 2차차분(채터 겨냥 보상의 직접 지표)·정착 피치 부호 평균
_air_q = torch.zeros(N, 12, device=dev); _air_n = torch.zeros(N, device=dev)   # 0910: 공중(지지비>0.9) 관절각 평균 — 실물 공중 비대칭(왼 힙a +11 vs 오른 +6) 대조
_jn = list(robot.joint_names); _jidx = {n: i for i, n in enumerate(_jn)}
# ── 0913 브리지형 느린 기울기 보정기(--roll_comp): 정책 밖 외부 루프. 실물 브리지에서 IMU 롤(맵 편향 보정 후)을 저역 필터해 관절 목표에 더하는 것과 같은 자리 ──
_RC_ON = bool(args_cli.roll_comp)
_RC_OBS_BITS = None
if _RC_ON:
    _p = [x.strip() for x in args_cli.roll_comp.split(",")]
    _rcK = float(_p[0]); _rcTau = max(float(_p[1]), 1e-3); _rcLim = math.radians(float(_p[2]))
    _rcJ = _p[3] if len(_p) > 3 else "ankle_r"; _rcSign = float(_p[4]) if len(_p) > 4 else 1.0
    _rcLR = [float(x) for x in args_cli.roll_comp_lr.split(",")]
    _rct = uenv.action_manager.get_term("joint_pos")
    _rc_tj = list(range(_nj)) if isinstance(_rct._joint_ids, slice) else [int(i) for i in _rct._joint_ids]
    _rc_cols = [_rc_tj.index(_jidx[f"left_{_rcJ}_joint"]), _rc_tj.index(_jidx[f"right_{_rcJ}_joint"])]
    _rc_f = torch.zeros(N, device=dev); _rc_off = torch.zeros(N, device=dev)
    _rc_acc = torch.zeros(N, device=dev); _rc_n = torch.zeros(N, device=dev); _rc_sat = torch.zeros(N, device=dev)
    _rc_dt = float(uenv.step_dt)
    # 0913 적대검수 F2 (브리지 rl_bridge_node.RollComp 와 같은 법칙): 게이트 디바운스 → 유지, 실효 해제 뒤엔
    # 같은 τ 로 0 을 향해 감쇠, 출력은 틱당 ±slew 로 제한, 입력 롤은 ±roll_clamp 로 자른다. 오프셋 계단 금지.
    _rc_gate_off_ticks = torch.zeros(N, device=dev)      # env별 연속 raw-gate-off 틱 수
    _rc_deb_ticks = max(1, int(round(args_cli.roll_comp_debounce_s / _rc_dt)))
    _rc_slew_r = math.radians(args_cli.roll_comp_slew_deg)
    _rc_rollcl = math.radians(args_cli.roll_comp_roll_clamp_deg)
    _rc_o = _rct.process_actions
    def _rc_patched(actions, _t=_rct, _o=_rc_o):
        """post 자리: 액션 체인 최종 목표 뒤에 가산(구동 모델 우회 — 상한 참고용). 법칙은 _rc_core 공유."""
        _o(actions)
        off = _rc_core()
        _t._processed_actions[:, _rc_cols[0]] += _rcLR[0] * off
        _t._processed_actions[:, _rc_cols[1]] += _rcLR[1] * off
    def _rc_core(_t=_rct):
        """게이트·필터·오프셋 계산(한 스텝 1회). 반환: 오프셋[rad] (N,).

        브리지 RollComp.update 와 같은 법칙(0913 F2): 롤 클램프 → raw 게이트 → 디바운스(유지) →
        해제 시 같은 τ 로 감쇠 → 출력 슬루 제한. 게이트 접촉원은 --roll_comp_gate_src(기본 obs =
        정책이 보는 비트 = 브리지와 동일). 시뮬엔 HangGuard 가 없으므로 브리지의 '접지 확인(F3)'
        추가 조건은 없다 — 실물이 더 보수적인 쪽(같은 조건 + 접지 확인)이다.
        """
        global _rc_f, _rc_off, _rc_acc, _rc_n, _rc_sat, _rc_gate_off_ticks
        g = robot.data.projected_gravity_b
        roll = torch.atan2(g[:, 1], -g[:, 2])
        roll = torch.where(torch.isfinite(roll), roll, torch.zeros_like(roll))
        roll = torch.clamp(roll, -_rc_rollcl, _rc_rollcl)
        fz = _cs.data.net_forces_w[:, [_fl, _fr], :].norm(dim=-1) > 5.0
        if args_cli.roll_comp_gate_src == "obs" and _RC_OBS_BITS is not None: fz = _RC_OBS_BITS
        if args_cli.roll_comp_gate == "both": raw_gate = fz[:, 0] & fz[:, 1]
        elif args_cli.roll_comp_gate == "any": raw_gate = fz[:, 0] | fz[:, 1]
        else: raw_gate = torch.ones_like(fz[:, 0])
        _rc_gate_off_ticks = torch.where(raw_gate, torch.zeros_like(_rc_gate_off_ticks), _rc_gate_off_ticks + 1.0)
        eff_gate = raw_gate | (_rc_gate_off_ticks < _rc_deb_ticks)
        _f_upd = _rc_f + (_rc_dt / _rcTau) * (roll - _rc_f)          # raw 게이트: 정상 갱신
        _f_dec = _rc_f + (_rc_dt / _rcTau) * (0.0 - _rc_f)           # 실효 해제: 0 으로 감쇠
        _rc_f = torch.where(raw_gate, _f_upd, torch.where(eff_gate, _rc_f, _f_dec))
        _tgt = torch.clamp(_rcSign * _rcK * _rc_f, -_rcLim, _rcLim)
        _rc_off = _rc_off + torch.clamp(_tgt - _rc_off, -_rc_slew_r, _rc_slew_r)
        _rc_acc += _rc_off.abs() * eff_gate; _rc_n += eff_gate.float()
        _rc_sat += ((_rc_off.abs() >= _rcLim - 1e-6) & eff_gate).float()
        return _rc_off
    def _rc_pre(target):
        off = _rc_core()
        out = target.clone()
        out[:, _rc_cols[0]] += _rcLR[0] * off
        out[:, _rc_cols[1]] += _rcLR[1] * off
        return out
    if args_cli.roll_comp_site == "pre":
        _rct.pre_delay_hook = _rc_pre
    else:
        _rct.process_actions = _rc_patched
    print(f"[roll_comp] K={_rcK} tau={_rcTau}s limit=±{math.degrees(_rcLim):.1f}° joints={_rcJ} sign={_rcSign:+.0f} lr={_rcLR} gate={args_cli.roll_comp_gate}/{args_cli.roll_comp_gate_src} site={args_cli.roll_comp_site} cols={_rc_cols} dt={_rc_dt} "
          f"debounce={args_cli.roll_comp_debounce_s}s({_rc_deb_ticks}틱) slew={args_cli.roll_comp_slew_deg}°/틱 roll_clamp=±{args_cli.roll_comp_roll_clamp_deg}° (F2 계단 금지 = 브리지와 같은 법칙)")
try:
    from rl_walking.mdp_hang import _feet_cfg as _fcfg, _support_gone_mask
    _fc = _fcfg(); _fc.resolve(env.unwrapped.scene); _FEET = _fc.name; _FEET_IDS = _fc.body_ids
except Exception:
    _FEET, _FEET_IDS = None, None

# ── 접촉 obs 교란 (0911 접촉 의존도 평가) — actor 관측의 feet_contact 슬라이스만 변조.
#    보상(feet_contact_gated)·종료·평가 정답은 센서를 직접 읽으므로 불변 = 교란은 actor 에만. ──
_cs_lo = _cs_hi = None
try:
    _om = uenv.observation_manager
    _terms = _om.active_terms["policy"]; _dims = _om.group_obs_term_dim["policy"]
    _off = 0; _layout = []
    for _nm, _dm in zip(_terms, _dims):
        _d = _dm[0] if isinstance(_dm, (list, tuple)) else int(_dm)
        _layout.append((_nm, _off, _d))
        if _nm == "feet_contact":
            _cs_lo, _cs_hi = _off, _off + _d
        _off += _d
    if _RC_ON and args_cli.roll_comp_gate_src == "obs":
        print(f"[roll_comp] 게이트 출처 obs: feet_contact 슬라이스={_cs_lo}:{_cs_hi}" + (" — 없음 → 진 접촉력으로 대체(브리지와 다름!)" if _cs_hi is None else " (마지막 2칸 [new_L,new_R])"))
    if args_cli.contact_perturb != "none":
        print(f"[obs 레이아웃] policy dim={_off}, feet_contact 슬라이스={_cs_lo}:{_cs_hi} "
              f"(항 {len(_terms)}개). 교란={args_cli.contact_perturb} seed={args_cli.perturb_seed}")
        if _cs_lo is None:
            print("[경고] feet_contact 항을 못 찾음 — 교란 미적용")
except Exception as _e:
    print(f"[경고] obs 레이아웃 조회 실패: {_e} — 교란 미적용")


class ContactPerturb:
    """actor 접촉 obs 교란기. obs[:, cs:ce] (L/R 이진 접촉)만 변조. 정답/물리/보상 불변."""
    def __init__(self, mode, N, dev, cs, ce, seed=0):
        self.mode, self.N, self.dev, self.cs, self.ce = mode, N, dev, cs, ce
        self.g = torch.Generator(device="cpu"); self.g.manual_seed(int(seed))
        self.held = None; self.buf = []; self.every = 5   # 50Hz→10Hz S&H

    def _rand(self, shape, p):
        return (torch.rand(shape, generator=self.g).to(self.dev) < p)

    def __call__(self, obs, i):
        if self.mode == "none" or self.cs is None:
            return obs
        # rsl_rl 래퍼는 TensorDict(키 'policy')를 준다 — 텐서 슬라이싱은 TensorDict 에 못 쓰므로
        # 'policy' 텐서만 꺼내 변조하고 다시 담는다 (critic 키 등은 불변).
        td = None
        t = obs
        if not torch.is_tensor(obs):
            td = obs
            t = obs["policy"]
        c = t[:, self.cs:self.ce]
        p = c.clone()
        if self.mode == "both1":
            p[:] = 1.0
        elif self.mode == "both0":
            p[:] = 0.0
        elif self.mode == "hold":
            if self.held is None or i % self.every == 0:
                self.held = c.clone()
            p = self.held.clone()
        elif self.mode == "delay3":
            self.buf.append(c.clone())
            if len(self.buf) > 3:
                p = self.buf.pop(0)
        elif self.mode == "left1":              # 0913 반박 검증: 왼발 접촉 비트 **상시 1**(false positive) — 실측: T5 내내 왼발 살아있는 패드 0·고착 L2 패드가 비트 1 을 만듦
            p[:, 0::2] = 1.0
        elif self.mode == "left1_after10":      # 10 s 뒤부터 왼발 상시 1
            if i >= 500:
                p[:, 0::2] = 1.0
        elif self.mode == "right0":             # 0913: 오른발 접촉 비트 **상시 0** — 이력 5×[L,R] 평탄화(폭 10, [old_L,old_R,…,new_L,new_R]) 라 오른발 열 전체(1::2)를 지운다
            p[:, 1::2] = 0.0                    #   (0913 감사 정정: 종전 p[:,1] 은 가장 오래된 오른발 비트 1칸만 지워 사실상 무효였음 — cm_*/cm2_* 결과 무효)
        elif self.mode == "right0_after10":     # 0913: 10 s 뒤부터 오른발 상시 0 (안정 서기 뒤 오판 발생)
            if i >= 500:
                p[:, 1::2] = 0.0
        elif self.mode == "onefoot":            # 오른발 2s 주기·앞 0.3s 0 오판 (실물 오른발 언로드 모사) — 0913 정정: 오른발 열 전체
            if (i % 100) < 15:
                p[:, 1::2] = 0.0
        elif self.mode == "chatter":            # 랜덤 단발 누락(양발 각 8%/스텝 0)
            p = p * (~self._rand(c.shape, 0.08)).float()
        elif self.mode == "realistic":          # 10Hz S&H + 지터 + 한발 짧은 누락(각 3%/스텝)
            if self.held is None or i % self.every == 0:
                self.held = c.clone()
            p = self.held.clone() * (~self._rand(c.shape, 0.03)).float()
        t = t.clone()
        t[:, self.cs:self.ce] = p
        if td is not None:
            td = td.clone()
            td["policy"] = t
            return td
        return t


_cperturb = ContactPerturb(args_cli.contact_perturb, N, dev, _cs_lo, _cs_hi, args_cli.perturb_seed)

# ── 0914 관절 물리 반전(좌표계 가설): 액션 최종 목표 + 관측 열 부호 반전 ──
_FLIP_COLS = []
_FLIP_OBS_IDX = []
if args_cli.flip_joints:
    _fj = [x.strip() for x in args_cli.flip_joints.split(",") if x.strip()]
    _FLIP_COLS = [_jidx[n] for n in _fj]
    try:
        for _nm, _o, _d in _layout:
            if _nm in ("joint_pos", "joint_vel", "joint_vel_ema") and _d % 12 == 0:
                _H = _d // 12
                for _h in range(_H):
                    for _c in _FLIP_COLS:
                        _FLIP_OBS_IDX.append(_o + _h * 12 + _c)
    except NameError:
        pass
    _FLIP_OBS_IDX = torch.tensor(sorted(set(_FLIP_OBS_IDX)), dtype=torch.long, device=dev) if _FLIP_OBS_IDX else None
    _rct_f = uenv.action_manager.get_term("joint_pos")
    _prev_hook = getattr(_rct_f, "pre_delay_hook", None)
    _flip_cols_t = torch.tensor(_FLIP_COLS, dtype=torch.long, device=dev)
    def _flip_hook(target, _ph=_prev_hook):
        out = target if _ph is None else _ph(target)
        out = out.clone()
        out[:, _flip_cols_t] = 2.0 * _DEF[:, _flip_cols_t] - out[:, _flip_cols_t]
        return out
    _rct_f.pre_delay_hook = _flip_hook
    print(f"[flip] 물리 반전 관절 {_fj} (열 {_FLIP_COLS}), 관측 반전 인덱스 {0 if _FLIP_OBS_IDX is None else int(_FLIP_OBS_IDX.numel())}개 (기존 pre_delay_hook {'래핑' if _prev_hook is not None else '없음'})")
# ── 0914 상수 관절 오프셋(부호 확인): 최종 물리 목표에 가산 ──
if args_cli.joint_offset:
    _jo = {}
    for _kv in args_cli.joint_offset.split(","):
        _k, _v = _kv.split(":"); _jo[_jidx[_k.strip()]] = math.radians(float(_v))
    _jo_cols = torch.tensor(list(_jo.keys()), dtype=torch.long, device=dev); _jo_vals = torch.tensor(list(_jo.values()), dtype=torch.float32, device=dev)
    _rct_o = uenv.action_manager.get_term("joint_pos"); _prev_o = getattr(_rct_o, "pre_delay_hook", None)
    def _offset_hook(target, _ph=_prev_o):
        out = target if _ph is None else _ph(target)
        out = out.clone(); out[:, _jo_cols] = out[:, _jo_cols] + _jo_vals; return out
    _rct_o.pre_delay_hook = _offset_hook
    print(f"[joint_offset] {args_cli.joint_offset} (열 {list(_jo.keys())}) — 최종 물리 목표에 가산")
_FEET_BODY = [k for k, n in enumerate(robot.body_names) if "ankle_r" in n] if args_cli.dump_feet_quat else []   # final3 자산은 링크 이름이 *_ankle_r_joint
if args_cli.dump_feet_quat:
    print(f"[dump_feet_quat] 발 링크 body idx {_FEET_BODY} = {[robot.body_names[k] for k in _FEET_BODY]}")
# ── 0914 로터측 관측 에뮬레이션 ──
_ROT_COLS = []; _ROT_IDX = {}; _rot_hist = None; _rot_ema = None; _rot_prev = None
if args_cli.rotor_obs:
    _rn = _jn if args_cli.rotor_obs.strip() == "all" else [x.strip() for x in args_cli.rotor_obs.split(",") if x.strip()]
    _ROT_COLS = [_jidx[n] for n in _rn]
    try:
        for _nm, _o, _d in _layout:
            if _nm in ("joint_pos", "joint_vel", "joint_vel_ema") and _d % 12 == 0:
                _ROT_IDX[_nm] = (_o, _d // 12)
    except NameError:
        pass
    _rot_cols_t = torch.tensor(_ROT_COLS, dtype=torch.long, device=dev)
    print(f"[rotor_obs] 관절 {_rn} (열 {_ROT_COLS}) 관측을 물리 목표로 대체: 항 {list(_ROT_IDX.keys())}")
def _rotor_obs(obs):
    """정책 관측의 joint_pos/joint_vel/joint_vel_ema(선택 열)를 '직전 틱 물리 목표' 기반 값으로 교체. 이력 5프레임은 자체 링버퍼."""
    global _rot_hist, _rot_ema, _rot_prev
    if not _ROT_COLS or not _ROT_IDX:
        return obs
    tgt = _LAST_OUT[0]
    if tgt is None:
        return obs
    q_t = (tgt - _DEF)[:, _rot_cols_t]
    if _rot_prev is None:                                   # 첫 틱: 물리 관절각으로 초기화(목표 0 인공물 방지)
        _rot_prev = (robot.data.joint_pos - _DEF)[:, _rot_cols_t].clone(); _rot_ema = torch.zeros_like(q_t)
        _rotor_obs.q_m = _rot_prev.clone()
    _a = 1.0 if args_cli.rotor_tau <= 0 else min(1.0, 0.02 / float(args_cli.rotor_tau))
    _rotor_obs.q_m = _rotor_obs.q_m + _a * (q_t - _rotor_obs.q_m)   # 모터측 각 = 목표의 1차 지연
    q_cmd = _rotor_obs.q_m
    qd_cmd = (q_cmd - _rot_prev) / 0.02; _rot_prev = q_cmd.clone()
    _rot_ema = 0.9 * _rot_ema + 0.1 * qd_cmd
    frame = {"joint_pos": q_cmd, "joint_vel": qd_cmd, "joint_vel_ema": _rot_ema}
    t = obs["policy"] if not torch.is_tensor(obs) else obs
    t = t.clone()
    if _rot_hist is None:
        _rot_hist = {}
    for _nm, (_o, _H) in _ROT_IDX.items():
        buf = _rot_hist.get(_nm)
        if buf is None:
            buf = frame[_nm].unsqueeze(1).repeat(1, _H, 1)
        else:
            buf = torch.cat([buf[:, 1:], frame[_nm].unsqueeze(1)], dim=1)
        _rot_hist[_nm] = buf
        for _h in range(_H):
            t[:, _o + _h * 12 + _rot_cols_t] = buf[:, _h]
    if torch.is_tensor(obs):
        return t
    obs = dict(obs); obs["policy"] = t; return obs
def _flip_obs(obs):
    if _FLIP_OBS_IDX is None or not _FLIP_COLS:
        return obs
    if torch.is_tensor(obs):
        obs = obs.clone(); obs[:, _FLIP_OBS_IDX] = -obs[:, _FLIP_OBS_IDX]; return obs
    t = obs["policy"].clone(); t[:, _FLIP_OBS_IDX] = -t[:, _FLIP_OBS_IDX]
    obs = dict(obs); obs["policy"] = t; return obs

# ── 시뮬 블랙박스 덤프 (0911): 실물 rl_bridge 블랙박스와 같은 키·같은 JOINT_ORDER(좌우 교대)로 기록 ──
import json as _json
_JO_REAL = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
            "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
            "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
_DUMP = []; _dump_dead = []
_LAST_OBS = [None]
if args_cli.dump_envs > 0:
    os.makedirs("log_picture", exist_ok=True)
    _DUMP = [open(f"log_picture/simbb_{args_cli.out_tag}_env{k}.jsonl", "w", encoding="utf-8") for k in range(min(args_cli.dump_envs, N))]
    _dump_dead = [False] * len(_DUMP)
    _jsel = [_jidx[n] for n in _JO_REAL]
    print(f"[dump] {len(_DUMP)}개 env 시계열 → log_picture/simbb_{args_cli.out_tag}_env*.jsonl (JOINT_ORDER 좌우교대로 재배열)")


def _dump_step(i, a, roll, pitch, contact, term, st):
    if not _DUMP:
        return
    _tgt = _LAST_OUT[0] if _LAST_OUT[0] is not None else (_DEF + a * _ASCALE)
    _q = robot.data.joint_pos; _qd = robot.data.joint_vel; _w = robot.data.root_ang_vel_b
    _sg = _support_gone_mask(st)
    _ts = 1.0e9 + i * 0.02
    for k, fh in enumerate(_DUMP):
        if _dump_dead[k]:
            continue
        row = {"timestamp": _ts, "live": True, "checkpoint": f"sim:{args_cli.out_tag}",
               "rp_deg": [float(roll[k]), float(pitch[k])],
               "gyro_rad_s": [float(x) for x in _w[k]],
               "qpos_rad": [float(_q[k, j]) for j in _jsel], "qvel_rad_s": [float(_qd[k, j]) for j in _jsel],
               "raw_action": [float(a[k, j]) for j in _jsel],
               "targets_deg_joint": {n: float(torch.rad2deg(_tgt[k, _jidx[n]])) for n in _JO_REAL},
               "contact": [float(contact[k, 0]), float(contact[k, 1])],
               "hang_state": "CONTACT_ACQUIRED" if float(st["touch_t"][k]) >= 0.0 else "PRE_CONTACT_HANG_ALLOWED",
               "support_r": float(st["r"][k]), "support_gone": bool(_sg[k]), "blend_alpha": 1.0}
        if args_cli.dump_obs and _LAST_OBS[0] is not None:
            row["obs"] = [float(x) for x in _LAST_OBS[0][k]]
        if _FEET_BODY:
            _bq_f = robot.data.body_quat_w[k, _FEET_BODY]          # (2,4) wxyz
            _dn_f = []
            for _qf in _bq_f:
                _fw, _fx, _fy, _fz = [float(v) for v in _qf]
                # R(q)·(0,0,−1): 발 링크 프레임 −Z 의 월드 벡터 (링크 프레임이 회전돼 있으면 발바닥 법선이 아닐 수 있음 — 쿼터니언 차이로 판정)
                _dn_f.append([-(2.0 * (_fx * _fz + _fw * _fy)), -(2.0 * (_fy * _fz - _fw * _fx)), -(1.0 - 2.0 * (_fx * _fx + _fy * _fy))])
            row["feet_quat_wxyz"] = [[float(v) for v in _qf] for _qf in _bq_f]; row["feet_down_w"] = _dn_f
        if bool(st.get("cable_on", False)):   # 케이블 모델 진단: 실제 적용 장력·앵커-후크 거리 (0912, 편심 스윕이 비트동일 → 힘 작용 여부 확인용)
            row["cbl_T"] = float(st["cbl_T"][k]); row["cbl_dist"] = float(st["cbl_dist"][k]); row["cbl_L0"] = float(st["cbl_L0"][k]); row["cbl_d0"] = float(st["cbl_d0"][k])
            row["base_z"] = float(robot.data.root_pos_w[k, 2])
        fh.write(_json.dumps(row) + "\n")
        if bool(term[k]):
            fh.write(_json.dumps({"timestamp": _ts, "event": "stop_all", "reason": "sim: terminated(fell)"}) + "\n")
            fh.flush()          # DR 조건은 앱 종료가 네이티브라 버퍼가 유실된다 — 즉시 flush
            _dump_dead[k] = True
        elif i % 50 == 0:
            fh.flush()


t0 = time.time()
_ramp_prev = torch.zeros(N, device=dev); _r_prev = torch.full((N,), float("nan"), device=dev)
for i in range(STEPS):
    with torch.inference_mode():
        if args_cli.hook_dxy:
            _hs = getattr(uenv, "_hang", None)
            if _hs is not None and "px" in _hs:
                _hdx, _hdy = _pair(args_cli.hook_dxy); _hs["px"][:] = float(_hdx); _hs["py"][:] = float(_hdy)
        _DEF.copy_(robot.data.default_joint_pos)   # 0918 S9c: env 별 물리 기본자세 편향 DR 뒤 배포체인 기준 갱신
        obs = _cperturb(obs, i)          # actor 접촉 obs 교란(정답/물리 불변) — none 이면 무변
        obs = _flip_obs(obs)              # 0914 관절 반전 가설: 관측 열 부호 반전(물리 불변)
        obs = _rotor_obs(obs)             # 0914 로터측 엔코더 에뮬레이션(관측만)
        if _RC_ON and args_cli.roll_comp_gate_src == "obs" and _cs_hi is not None:
            _tt = obs["policy"] if not torch.is_tensor(obs) else obs
            _RC_OBS_BITS = _tt[:, _cs_hi - 2:_cs_hi] > 0.5     # 이력 마지막 칸 [new_L, new_R] = 정책이 이번 틱에 보는 비트
        _LAST_OBS[0] = obs["policy"] if not torch.is_tensor(obs) else obs   # 덤프용(0914): 이번 틱 정책 입력
        a = torch.zeros(N, 12, device=dev) if policy is None else policy(obs)
        _stp = getattr(uenv, "_hang", None)
        if _stp is not None:             # 스텝 '전' 상태 보존 — env.step 은 종료 env 를 즉시 리셋한다(0913 정정: 종전 r_at_death 는 리셋값 r0 을 읽었음)
            _r_prev = _stp["r"].clone()
            if "ramp_rate" in _stp:
                _rd = _stp["ramp_rate"] * (_stp["t"] - _stp["ramp_start"]).clamp(min=0.0)
                _capm = _stp["ramp_cap"].abs(); _ramp_prev = torch.maximum(torch.minimum(_rd, _capm), -_capm)
        obs, _, dones, _ = env.step(a)
    st = uenv._hang
    term = uenv.termination_manager.terminated
    pg = robot.data.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])); roll = torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2]))
    w = robot.data.root_ang_vel_b.norm(dim=-1)
    f = _cs.data.net_forces_w[:, [_fl, _fr], :].norm(dim=-1)
    contact = (f > 5.0).float()
    _dump_step(i, a, roll, pitch, contact, term, st)
    a_ = alive
    max_roll[a_] = torch.maximum(max_roll[a_], roll[a_].abs()); max_pitch[a_] = torch.maximum(max_pitch[a_], pitch[a_].abs())
    max_w[a_] = torch.maximum(max_w[a_], w[a_])
    touch_t = torch.where(a_ & (touch_t < 0) & (st["touch_t"] >= 0), st["touch_t"], touch_t)
    newly_rel = a_ & (rel_t < 0) & (st["released_t"] >= 0)
    rel_t = torch.where(newly_rel, st["released_t"], rel_t); pitch_at_rel[newly_rel] = pitch[newly_rel]
    newly_dead = a_ & term
    death_step[newly_dead] = i; pitch_death[newly_dead] = pitch[newly_dead]; roll_death[newly_dead] = roll[newly_dead]
    r_at_death[newly_dead] = _r_prev[newly_dead]          # 스텝 전 지지비 (리셋 전 값)
    ramp_at_death[newly_dead] = _ramp_prev[newly_dead]    # 스텝 전 램프 토크
    last_q[a_] = robot.data.joint_pos[a_]; last_contact[a_] = contact[a_]
    _post = a_ & _support_gone_mask(st)   # item 4c: 케이블이면 실제 장력<=eps (힘 모델은 r<=0 동일)
    _both_now = (contact[:, 0] > 0) & (contact[:, 1] > 0)
    _ever_both |= (a_ & _both_now)
    _lift_evt = (a_ & _ever_both & _prev_both & ~_both_now).float()
    _lift_cnt += _lift_evt
    _air = a_ & (st["r"] > 0.9)
    if _air.any():
        _air_q[_air] += robot.data.joint_pos[_air]; _air_n[_air] += 1.0
    if i >= _SS_FROM:
        _ss = a_ & _support_gone_mask(st)   # item 4c: 실제 지지 소멸 (스케줄 아님)
        _ss_w[_ss] += robot.data.root_ang_vel_b[_ss, :2].norm(dim=-1); _ss_v[_ss] += robot.data.root_lin_vel_b[_ss, :2].norm(dim=-1)
        _ss_qd2[_ss] += (robot.data.joint_vel[_ss] ** 2).mean(1); _ss_n[_ss] += 1.0; _ss_lift += _lift_evt
        if _prev_a is not None:
            _ss_ar2[_ss] += ((a - _prev_a)[_ss] ** 2).mean(1)
            if _prev_a2 is not None:
                _ss_aa2[_ss] += ((a - 2.0 * _prev_a + _prev_a2)[_ss] ** 2).mean(1)
        _ss_r2[_ss] += roll[_ss] ** 2; _ss_p2[_ss] += pitch[_ss] ** 2; _ss_pm[_ss] += pitch[_ss]
        _ss_rpk[_ss] = torch.maximum(_ss_rpk[_ss], roll[_ss].abs()); _ss_ppk[_ss] = torch.maximum(_ss_ppk[_ss], pitch[_ss].abs())
    _prev_a2 = None if _prev_a is None else _prev_a.clone(); _prev_a = a.clone()
    _prev_both = _both_now.clone()
    if _post.any():
        _st_w[_post] += robot.data.root_ang_vel_b[_post, :2].norm(dim=-1); _st_v[_post] += robot.data.root_lin_vel_b[_post, :2].norm(dim=-1)
        _st_qd2[_post] += (robot.data.joint_vel[_post] ** 2).mean(1); _st_n[_post] += 1.0
        _t = robot.data.applied_torque[_post].abs()
        _tau_peak = torch.maximum(_tau_peak, _t.max(0).values); _tau_rms2 += (_t ** 2).sum(0)
        _tau_sat += (_t >= 0.95 * _eff[_post]).float().sum(0); _tau_n += int(_post.sum())
        _pw = robot.data.body_pos_w[_post][:, [_bl, _br], :]
        _stance_w.append((_pw[:, 0, 1] - _pw[:, 1, 1]).abs().mean().item()); _stance_dx.append((_pw[:, 0, 0] - _pw[:, 1, 0]).abs().mean().item())
    alive &= ~term
    ratio_trace.append(st["r"].mean().item()); contact_trace.append(contact[alive].mean().item() if alive.any() else float("nan"))
    z_trace.append(robot.data.root_pos_w[alive, 2].mean().item() if alive.any() else float("nan"))
    pitch_trace.append(pitch[alive].mean().item() if alive.any() else float("nan"))

for _fh in _DUMP:            # 덤프 파일 닫기 (앱 종료 전에 — 네이티브 종료로 유실 방지)
    try: _fh.flush(); _fh.close()
    except OSError: pass
# ── 지표 ──
fell = death_step >= 0
touched = touch_t >= 0
released = rel_t >= 0
t_final = STEPS / 50.0
surv_after = torch.where(fell, death_step.float() / 50.0, torch.full_like(rel_t, t_final)) - rel_t
# 0911 Codex 지적 수정: '안 넘어짐(~fell)'만으로 성공 처리하면 방출이 늦어 방출 후 stand_s 를 못 채운 env 도 성공으로 셈.
# 성공 = 방출 도달 & 방출 후 stand_s 이상 생존(넘어지지 않고). 방출 후 stand_s 미만은 '미검증'으로 성공 제외.
stand5 = released & (~fell) & (surv_after >= args_cli.stand_s)
_inconcl = int((released & (~fell) & (surv_after < args_cli.stand_s)).sum())  # 방출은 됐으나 관측창이 짧아 미검증
def pct(x): return 100.0 * float(x.float().mean())
def dirs(p, r):
    m = ~torch.isnan(p)
    back = int(((p < 0) & (p.abs() >= r.abs()) & m).sum()); fwd = int(((p > 0) & (p.abs() >= r.abs()) & m).sum())
    left = int(((r > 0) & (r.abs() > p.abs()) & m).sum()); right = int(((r < 0) & (r.abs() > p.abs()) & m).sum())
    return back, fwd, left, right
b, fw, l, rgt = dirs(pitch_death, roll_death)
jn = list(robot.data.joint_names)
q_end = torch.rad2deg(last_q)
out_lines = []
P = out_lines.append
P("=" * 74); P(f"Hanging-to-Stand 평가 — {args_cli.task}"); P(f"인자: {' '.join(sys.argv[1:])}")
P(f"체크포인트: {resume_path}"); P(f"배포 경로: {chain_desc}"); P(f"env {N} × {T_s:.1f}s, DR {'OFF(명목)' if args_cli.nominal else 'ON'}, com_fix={args_cli.com_fix}, 푸시 {'OFF' if (args_cli.no_push or args_cli.nominal) else 'ON'}, 관측노이즈 {'OFF' if args_cli.no_obs_noise else 'ON'}")
P(f"지지 스케줄: r0 {_hp['r0_range']} 유지 {_hp['hold_range']}s 방출 {_hp['release_range']}s, 간격 z {env_cfg.events.reset_base.params['pose_range'].get('z')}")
P("=" * 74)
P(f"접지(양발 동시) 도달      : {int(touched.sum())}/{N} ({pct(touched):.1f}%)  평균 접지 시각 {float(touch_t[touched].mean()) if touched.any() else float('nan'):.2f}s")
P(f"지지 0 도달(살아서)        : {int(released.sum())}/{N} ({pct(released):.1f}%)  평균 방출완료 시각 {float(rel_t[released].mean()) if released.any() else float('nan'):.2f}s  그때 피치 {float(pitch_at_rel[released].mean()) if released.any() else float('nan'):+.2f}°")
P(f"**지지 0 이후 {args_cli.stand_s:.0f}s 생존**   : {int(stand5.sum())}/{N} ({pct(stand5):.1f}%)   ← 핵심 지표")
if bool(torch.isfinite(ramp_at_death).any()):
    _rv = ramp_at_death[torch.isfinite(ramp_at_death)].abs()
    _q = torch.quantile(_rv, torch.tensor([0.1, 0.5, 0.9], device=_rv.device))
    P(f"램프 롤 토크 at 낙상 [Nm] (낙상 env {int(_rv.numel())}/{N}): p10 {float(_q[0]):.2f} · 중앙 {float(_q[1]):.2f} · p90 {float(_q[2]):.2f} · 최소 {float(_rv.min()):.2f} · 최대 {float(_rv.max()):.2f}")
P(f"**지지 0 이후 {args_cli.stand_s:.0f}s 생존**(미검증 {_inconcl}: 방출 후 관측창 <{args_cli.stand_s:.0f}s)")
P(f"낙상(조기 종료)            : {int(fell.sum())}/{N} ({pct(fell):.1f}%)  방향 뒤 {b} · 앞 {fw} · 왼 {l} · 오른 {rgt}")
if fell.any():
    P(f"   낙상 시 지지비 평균 {float(r_at_death[fell].mean()):.2f}, 낙상 시각 평균 {float(death_step[fell].float().mean())/50:.2f}s, "
      f"방출 후 낙상 {int((fell & released).sum())} / 방출 전 낙상 {int((fell & ~released).sum())}")
P(f"env별 최대 |롤| 평균 {float(max_roll.mean()):.2f}°  최대 |피치| 평균 {float(max_pitch.mean()):.2f}°  최대 각속도 평균 {float(max_w.mean()):.2f} rad/s")
_lc = last_contact
P(f"종료 시 접촉: 양발 {int(((_lc[:,0]>0)&(_lc[:,1]>0)).sum())} · 왼만 {int(((_lc[:,0]>0)&(_lc[:,1]==0)).sum())} · 오른만 {int(((_lc[:,0]==0)&(_lc[:,1]>0)).sum())} · 무접촉 {int(((_lc[:,0]==0)&(_lc[:,1]==0)).sum())}")
P("종료 시 관절각 평균 [deg] (생존 env):")
_al = ~fell
if _al.any():
    for k, n in enumerate(jn):
        P(f"   {n:<22} {float(q_end[_al, k].mean()):+7.2f}")
else:
    P("   (생존 env 없음)")
if _tau_n > 0:
    P("물리 실현성 — 지지 0 이후 생존 스텝 (AK 한계 대비): 관절 τ피크 /한계 τRMS 포화%")
    for k, n in enumerate(jn):
        P(f"   {n:<22} {float(_tau_peak[k]):6.2f} {float(_eff[0, k]):5.1f} {float((_tau_rms2[k] / _tau_n).sqrt()):6.2f} {100.0 * float(_tau_sat[k]) / _tau_n:5.1f}")
    P(f"   발 간격(좌우, 발목롤 링크) 평균 {sum(_stance_w)/len(_stance_w)*1000:.0f} mm · 앞뒤 어긋남 {sum(_stance_dx)/len(_stance_dx)*1000:.0f} mm")
_ok = _st_n > 0
if _ok.any():
    P(f"정지도 — 지지 0 이후 생존 스텝 env 평균: |각속도xy| {float((_st_w[_ok]/_st_n[_ok]).mean()):.3f} rad/s · |수평속도| {float((_st_v[_ok]/_st_n[_ok]).mean()):.3f} m/s · "
      f"관절속도 RMS {float(((_st_qd2[_ok]/_st_n[_ok]).sqrt()).mean()):.3f} rad/s · 발 들기 {float(_lift_cnt.mean()):.2f}회/env (양발 접촉 후)")
_ok2 = _ss_n > 0
if _ok2.any():
    P(f"**정상상태(마지막 3s, 스탠바이)** env 평균: |각속도xy| {float((_ss_w[_ok2]/_ss_n[_ok2]).mean()):.3f} rad/s · |수평속도| {float((_ss_v[_ok2]/_ss_n[_ok2]).mean()):.3f} m/s · "
      f"관절속도 RMS {float(((_ss_qd2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.3f} rad/s · 발 들기 {float(_ss_lift.mean()):.2f}회/env")
    P(f"   정상상태 추가: action-rate RMS {float(((_ss_ar2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.4f} · 롤 RMS {float(((_ss_r2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.2f}° 피크 {float(_ss_rpk[_ok2].mean()):.2f}° · "
      f"피치 RMS {float(((_ss_p2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.2f}° 피크 {float(_ss_ppk[_ok2].mean()):.2f}° · "
      f"액션 2차차분 RMS {float(((_ss_aa2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.4f} · 정착 피치 평균 {float((_ss_pm[_ok2]/_ss_n[_ok2]).mean()):+.2f}°")
    _flz = robot.data.net_forces_w if False else None  # (placeholder)
    try:
        _fs = env.unwrapped.scene.sensors[_FEET].data.net_forces_w[:, _FEET_IDS, 2].clamp(min=0.0)
        _fl = float(_fs[_ok2, 0].mean()); _fr = float(_fs[_ok2, 1].mean())
        P(f"정상상태 하중: 왼 {_fl:.0f}N / 오른 {_fr:.0f}N · 불균형 {abs(_fl-_fr)/max(_fl+_fr,1):.2f}")
    except Exception as _e:
        pass
_okA = _air_n > 0
if _okA.any():
    _mq = torch.rad2deg((_air_q[_okA] / _air_n[_okA, None]).mean(0))
    def _J(n): return float(_mq[_jidx[n]])
    P(f"공중 자세(지지비>0.9, 관절각 평균°): 힙a L {_J('left_hip_a_joint'):+.1f} / R {_J('right_hip_a_joint'):+.1f} · 힙f L {_J('left_hip_f_joint'):+.1f} / R {_J('right_hip_f_joint'):+.1f} · "
      f"힙r L {_J('left_hip_r_joint'):+.1f} / R {_J('right_hip_r_joint'):+.1f} · 무릎 L {_J('left_knee_joint'):+.1f} / R {_J('right_knee_joint'):+.1f} · 발목f L {_J('left_ankle_f_joint'):+.1f} / R {_J('right_ankle_f_joint'):+.1f}"
      f"   (실물 0910 공중: 힙a +11/+6 · 힙f −10/−0.5 · 힙r +3/+8 · 무릎 −0.6/−10)")
P("시간축 (0.5s 간격): t  지지비  접촉비  base z  피치")
for k in range(0, STEPS, 25):
    P(f"   {k/50:5.1f}  {ratio_trace[k]:5.2f}  {contact_trace[k]:5.2f}  {z_trace[k]:5.3f}  {pitch_trace[k]:+6.2f}")
P(f"소요 {time.time()-t0:.0f}s")
txt = "\n".join(out_lines)
if _RC_ON:
    _m = (_rc_n > 0)
    txt += ("\n보정기(--roll_comp %s, lr %s, gate %s): 작동 스텝 env 평균 %.0f · |오프셋| 평균 %.2f° · 포화(±limit) 비율 %.1f %%" % (
        args_cli.roll_comp, args_cli.roll_comp_lr,
        f"{args_cli.roll_comp_gate}/{args_cli.roll_comp_gate_src}/{args_cli.roll_comp_site}/deb{args_cli.roll_comp_debounce_s}/slew{args_cli.roll_comp_slew_deg}/clamp{args_cli.roll_comp_roll_clamp_deg}", _rc_n.mean().item(),
        math.degrees((_rc_acc[_m] / _rc_n[_m]).mean().item()) if _m.any() else 0.0,
        100.0 * (_rc_sat[_m] / _rc_n[_m]).mean().item() if _m.any() else 0.0))
print(txt)
tag = args_cli.out_tag or time.strftime("%m%d_%H%M%S")
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True); os.makedirs(os.path.join(REPO, "demo_output"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"hang2stand_{tag}.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
np.savez(os.path.join(REPO, "demo_output", f"hang2stand_{tag}.npz"), death_step=death_step.cpu().numpy(), touch_t=touch_t.cpu().numpy(),
         rel_t=rel_t.cpu().numpy(), max_roll=max_roll.cpu().numpy(), max_pitch=max_pitch.cpu().numpy(), max_w=max_w.cpu().numpy(),
         pitch_death=pitch_death.cpu().numpy(), roll_death=roll_death.cpu().numpy(), q_end=q_end.cpu().numpy(),
         ratio_trace=np.array(ratio_trace), contact_trace=np.array(contact_trace), z_trace=np.array(z_trace), pitch_trace=np.array(pitch_trace))
print(f"[산출물] log_picture/hang2stand_{tag}.txt")
env.close()
simulation_app.close()
