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
parser.add_argument("--level_fade_s", type=float, default=0.0, help="1004: 인계(사전검사 끝) 뒤 자동수평 트림을 이 시간 동안 선형으로 0 까지 소멸 — 실물 브리지 온셋 블렌딩(1.5 s)과 같음. 0 = 트림 유지(종전)")
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
parser.add_argument("--enc_obs", type=str, default="", choices=("", "legacy", "v2", "ema0"),
                    help="1001: 정책 관측의 엔코더 관절각·속도·속도 EMA 함수를 강제 — v2 = 실물 배포와 같은 의미(속도 EMA 에 진짜 속도), "
                         "legacy = G4~G15 학습 조건(속도 EMA 에 0+잡음). 비우면 태스크 설정 그대로.")
parser.add_argument("--obs_hold", type=int, default=0,
                    help="1001: 실물 브리지 관절 상태 샘플홀드 모사 — 관절마다 N틱에 한 번 갱신(env·관절별 무작위 위상, 라운드로빈 폴링 흉내). "
                         "정책 관측 joint_pos = 마지막 갱신값, joint_vel = 브리지 추정기(값 그대로면 0, 바뀌면 Δq/(틱×머문 틱), 머문 틱 상한 12), "
                         "joint_vel_ema = 그 속도의 EMA(v2 의미). --enc_obs v2 필요. 0 = 끔")
parser.add_argument("--knee_sea", type=float, default=None,
                    help="1001: 무릎을 직렬 탄성 액추에이터로(값 = 스프링 k Nm/rad, 예 42). 고정 무릎 공유 편향(처짐 정적 모델)은 끈다")
parser.add_argument("--knee_play_deg", type=float, default=None,
                    help="1001: 무릎 유격(데드밴드) 반폭 [deg] — DeadzoneImplicit(유격 안 자유, 엔코더 관측은 모터측). 고정 무릎 공유 편향은 끈다")
parser.add_argument("--knee_keep_bias", action="store_true", help="--knee_play_deg 와 함께: 고정 무릎 공유 편향(평형 자세)은 유지하고 유격만 동적으로 더한다")
parser.add_argument("--knee_sea_damp", type=float, default=0.3)
parser.add_argument("--knee_sea_J", type=float, default=0.01)
parser.add_argument("--replay_bb", type=str, default="",
                    help="1001: 실물 브리지 블랙박스(jsonl)의 관절 목표각(targets_deg_joint)을 인계 뒤 정책 출력 대신 그대로 재생(오픈루프). "
                         "실물 기본자세(CROUCH_V5)에서 벗어난 양을 시뮬 env 기본자세(편향 포함)에 더한다. 정책·플랜트 분리용")
parser.add_argument("--replay_t0", type=float, default=0.0, help="재생 시작 = 블랙박스 첫 틱 + 이 시간 [s]")
parser.add_argument("--contact_real", action="store_true",
                    help="1001 G17: 정책 접촉 관측을 실물 계약으로(mdp_hang.feet_contact_real: 발 평평 θ_off 2–6°·10 Hz 유지·1틱 전송) — 모든 정책 공통 적용")
parser.add_argument("--last_action_ema", type=float, default=0.0,
                    help="진단: 정책 관측 last_action 을 EMA(α) 로 느리게 만든 값으로 대체(0=끔). 25 Hz 2주기 자기루프(last_action 경유) 가설의 폐루프 대조")
parser.add_argument("--obs_ema", type=str, default="",
                    help="진단: 정책 관측 항을 시간 EMA 로 늦춤 'term:α,term:α' (예 base_ang_vel:0.1,joint_vel:0.1) — 폐루프 떨림 경로 귀속(배포 계약 아님)")
parser.add_argument("--gyro_delay", type=int, default=0,
                    help="1001: 정책 관측 자이로(base_ang_vel·gyro_ema 입력)를 N 정책 틱 늦게 — 실물 IMU 경로 지연 가설 시험")
parser.add_argument("--contact_drop", type=str, default="",
                    help="1001: 정책 관측의 발 접촉 비트만 강제로 0 (물리 접촉·채점 불변) — 'L|R|B:t_rel:dur' ';' 구분, t_rel 은 인계 기준 [s]. "
                         "실물 3 Hz 진동이 '한 발 접촉 비트 상실'에서 시작되는지 보는 관측 교란 대조")
parser.add_argument("--switch_to", type=str, default="",
                    help="진단(1004, C4 근거): '<run>:<model_N.pt>' 두 번째(회복) 정책. 감지기(IMU 관측만: 기울기·|ω_xy|, 히스테리시스)가 켜면 회복 정책, 꺼지면 주(서기) 정책. 두 정책 모두 같은 관측·명령 계약")
parser.add_argument("--min_height", type=float, default=None, help="진단(1004): base_too_low 종료 높이 덮어쓰기 [m] (거짓 낙상 대조, 기본 cfg 0.45)")
parser.add_argument("--ramp_off", action="store_true", help="진단(1004): hang_support 준정적 롤 토크 램프 DR 끔(학습 DR 의 가정 외란 — 평가 교란 대조)")
parser.add_argument("--sw_ref_tau", type=float, default=0.0, help="1004: 감지기 기울기를 기준 자세 대비 편차로(기준 = 인계 시점 롤·피치, 서기·조용·양발일 때 EMA τ[s] 갱신, |ref|≤8°). 0 = 절대 기울기(종전)")
parser.add_argument("--sw_ref_init_s", type=float, default=0.0, help="1004: 인계 뒤 이 시간 동안은 기준 자세가 현재 자세를 그대로 따라감(편차 0 → ω 로만 켜짐) — 인계 블렌딩 과도 오탐 방지")
parser.add_argument("--sw_off_contact", type=str, default="both", choices=("both", "any", "none"), help="1004 밤: 감지기 끄기 조건의 접촉 요구 — both(종전)/any(한 발 이상)/none(IMU 정숙만). 실물 패드 0 고착 대응")
parser.add_argument("--sw_oracle", action="store_true", help="1004 밤 진단: 외란 시작 틱에 즉시 회복 정책으로 전환(오라클 감지기) — 감지·인계 지연의 상한 효과 측정")
parser.add_argument("--sw_rec_vx", type=float, default=None, help="1005 진단: 회복 정책에만 관측 명령 vx 를 이 값으로(스텝 모드 신호 0.05) — 서기 정책 관측·환경 명령 불변. velocity_commands 5 프레임 전부 덮어씀")
parser.add_argument("--sw_rec_vx_s", type=float, default=0.0, help="1005: --sw_rec_vx 를 회복 모드 켜진 뒤 이 시간[s] 동안만(0 = 회복 모드 내내) — 반사 스텝 창, 이후 cmd 0 으로 정착")
parser.add_argument("--cue_state", type=str, default="", help="1005 활성화 구조: vx cue 상태 기반 종료 'tmin,hold,w,tmax' — 켜진 뒤 tmin s 지나고 양발 접촉·|ω_xy|<w 가 hold s 지속되면 종료, tmax 넘으면 다음 양발 지지에서 종료(래치, 감지기 꺼질 때까지 재점화 없음)")
parser.add_argument("--cue_dir", type=float, default=None, help="1005 ACT2: 방향 조건 cue — cue 시작 순간 몸통 전진속도(base x, m/s) > 이 값일 때만 cue(아니면 이번 활성화는 cue 없음). 정책은 전진 스텝만 학습")
parser.add_argument("--sw_early", type=str, default="", help="1005 ④ 정착: 상태 기반 조기 복귀 'v,tilt,hold' — 첫 외란 뒤 첫 하중 수용이 있었고 |v_xy|<v m/s·감지기 기울기<tilt° 가 hold s 지속되면 회복 모드 강제 해제(G21a 복귀)")
parser.add_argument("--cue_align", type=str, default="", help="1005 (B) 정착: cue 종료를 '발 앞뒤 정렬'로 — 'tmin,dx_cm,tmax': 켜진 뒤 tmin s 지나고 양발 접촉·두 발 앞축 거리 < dx_cm 이면 종료, tmax 넘으면 다음 양발 지지에서 종료")
parser.add_argument("--cue_clock", type=str, default="", choices=("", "reset"), help="1005 D: cue 시작 시 회복 정책 관측 gait_clock 을 '하중 적은 발이 바로 스윙' 위상에서 재시작(cue 동안만; 이후 환경 클록)")
parser.add_argument("--sw_monitor", action="store_true", help="감지기만 돌리고 전환하지 않음(주 정책 유지) — 감지 순간 상태 snapshot 비교용(C5 원인 B)")
parser.add_argument("--sw_on", type=str, default="4.0,0.5", help="전환 켜기: 기울기 deg, |ω_xy| rad/s (어느 하나 초과)")
parser.add_argument("--sw_off", type=str, default="2.5,0.25,1.0,1.5", help="전환 끄기: 기울기 deg, |ω| rad/s 둘 다 미만 & 양발 접촉이 calm s 연속 & 켜진 뒤 최소 hold s")
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
if args_cli.ramp_off:
    # 1004 정정: None 으로 지우면 난수 호출 수가 줄어 이후 DR 표본(편향·CoM)이 전부 바뀐다(짝 비교 붕괴) → 범위를 0 으로 두어 같은 수의 난수를 소비
    _hp["ramp_rate_range"] = (0.0, 0.0); _hp["ramp_cap_range"] = (0.0, 0.0); print("[ramp_off] 준정적 롤 토크 램프 0 (RNG 소비 유지, 진단)", flush=True)
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
    _sfx = "_legacy" if args_cli.enc_obs == "legacy" else "_v2"     # ema0 = v2 위치·속도 + EMA 0 (배포 러너 ema0 모드)
    for _nm, _fn in (("joint_pos", "encoder_joint_pos_rel"), ("joint_vel", "encoder_joint_vel"), ("joint_vel_ema", "encoder_joint_vel_ema")):
        _t = getattr(_P, _nm, None)
        if _t is None:
            continue
        _t.func = getattr(_mh, _fn + _sfx)
        _t.params = {k: v for k, v in dict(_t.params or {}).items() if k in ("alpha", "noise")}
    if args_cli.enc_obs == "ema0" and getattr(_P, "joint_vel_ema", None) is not None:
        def _ema_zero(env, alpha: float = 0.25, noise: float = 1.5, asset_cfg=None):
            a = env.scene[(asset_cfg or _mh._base_cfg()).name]
            return torch.zeros_like(a.data.joint_pos)
        _P.joint_vel_ema.func = _ema_zero
    print(f"[enc_obs] 정책 관측 엔코더 함수 = {args_cli.enc_obs}", flush=True)
if args_cli.obs_hold > 0:
    if args_cli.enc_obs not in ("v2", "ema0"):
        raise SystemExit("--obs_hold 는 --enc_obs v2 | ema0 와 함께 (배포 러너 의미)")
    from rl_walking import mdp_hang as _mh
    _HN = int(args_cli.obs_hold)

    def _held(env, asset_cfg=None):
        """(pos, vel) — 스텝당 한 번 갱신(common_step_counter 캐시). 리셋 첫 틱은 현재값·속도 0·머문 틱 1."""
        enc = _mh.encoder_joint_pos_v2(env, asset_cfg)
        st = getattr(env, "_hold_st", None)
        step = int(env.common_step_counter)
        if st is None or st["pos"].shape != enc.shape:
            g = torch.Generator(device="cpu"); g.manual_seed(1234)
            st = {"phase": torch.randint(0, _HN, enc.shape, generator=g).to(enc.device), "pos": enc.clone(),
                  "vel": torch.zeros_like(enc), "hold": torch.ones_like(enc), "step": None}
            env._hold_st = st
        if st["step"] != step:
            upd = ((step + st["phase"]) % _HN) == 0
            new_pos = torch.where(upd, enc, st["pos"])
            changed = new_pos != st["pos"]
            st["vel"] = torch.where(changed, (new_pos - st["pos"]) / (float(env.step_dt) * st["hold"]), torch.zeros_like(enc))
            st["hold"] = torch.where(changed, torch.ones_like(st["hold"]), torch.clamp(st["hold"] + 1, max=12.0))
            st["pos"] = new_pos; st["step"] = step
            fresh = (env.episode_length_buf == 0).view(-1, 1)
            if bool(fresh.any()):
                st["pos"] = torch.where(fresh, enc, st["pos"]); st["vel"] = torch.where(fresh, torch.zeros_like(enc), st["vel"])
                st["hold"] = torch.where(fresh, torch.ones_like(st["hold"]), st["hold"])
        return st["pos"], st["vel"]

    def _hold_pos_rel(env, asset_cfg=None):
        a = env.scene[(asset_cfg or _mh._base_cfg()).name]
        return _held(env, asset_cfg)[0] - a.data.default_joint_pos

    def _hold_vel(env, asset_cfg=None):
        return _held(env, asset_cfg)[1].clone()

    def _hold_vel_ema(env, alpha: float = 0.25, noise: float = 1.5, asset_cfg=None):
        x = _held(env, asset_cfg)[1]
        if noise > 0.0:
            x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
        return _mh._ema_state(env, "_s8_hold_jvel_ema", x, alpha).clone()

    _P = env_cfg.observations.policy
    _hold_terms = [("joint_pos", _hold_pos_rel), ("joint_vel", _hold_vel)] + ([("joint_vel_ema", _hold_vel_ema)] if args_cli.enc_obs == "v2" else [])
    for _nm, _fn in _hold_terms:
        _t = getattr(_P, _nm, None)
        if _t is not None:
            _t.func = _fn
    print(f"[obs_hold] 관절 상태 샘플홀드 {_HN}틱 (브리지 qvel 추정기 모사)", flush=True)
if args_cli.knee_sea is not None:
    import re as _re2
    from isaaclab.actuators import ImplicitActuatorCfg as _Impl2
    from rl_walking.actuators_play import SeriesElasticPDActuatorCfg
    _acts = env_cfg.scene.robot.actuators; _b = _acts["ak70"]
    _ex = _b.joint_names_expr if isinstance(_b.joint_names_expr, (list, tuple)) else [_b.joint_names_expr]
    _keep = [e for e in _ex if not _re2.fullmatch(e, "left_knee_joint")]
    _acts["ak70"] = _Impl2(joint_names_expr=_keep, effort_limit_sim=_b.effort_limit_sim, velocity_limit_sim=_b.velocity_limit_sim,
                          stiffness=_b.stiffness, damping=_b.damping, armature=_b.armature)
    _acts["knee_sea"] = SeriesElasticPDActuatorCfg(
        joint_names_expr=[".*_knee_joint"], effort_limit=_b.effort_limit_sim, velocity_limit=_b.velocity_limit_sim,
        stiffness=_b.stiffness, damping=_b.damping, armature=0.0,
        spring_k=float(args_cli.knee_sea), spring_damping=float(args_cli.knee_sea_damp), motor_inertia=float(args_cli.knee_sea_J),
        sim_dt=float(env_cfg.sim.dt))
    _pb = getattr(env_cfg.events, "default_pose_bias", None)
    if _pb is not None and "shared_knee_rad" in _pb.params:
        _pb.params["shared_knee_rad"] = None
    print(f"[knee_sea] 무릎 직렬 탄성 k={args_cli.knee_sea} c={args_cli.knee_sea_damp} J={args_cli.knee_sea_J} dt={env_cfg.sim.dt} · 공유 무릎 편향 끔", flush=True)
if args_cli.knee_play_deg is not None:
    import re as _re3, math as _m3
    from isaaclab.actuators import ImplicitActuatorCfg as _Impl3
    from rl_walking.actuators_play import DeadzoneImplicitActuatorCfg as _DZ3
    _acts = env_cfg.scene.robot.actuators; _b = _acts["ak70"]
    _ex = _b.joint_names_expr if isinstance(_b.joint_names_expr, (list, tuple)) else [_b.joint_names_expr]
    _keep = [e for e in _ex if not _re3.fullmatch(e, "left_knee_joint")]
    _acts["ak70"] = _Impl3(joint_names_expr=_keep, effort_limit_sim=_b.effort_limit_sim, velocity_limit_sim=_b.velocity_limit_sim,
                          stiffness=_b.stiffness, damping=_b.damping, armature=_b.armature)
    _acts["knee_play"] = _DZ3(joint_names_expr=[".*_knee_joint"], effort_limit_sim=_b.effort_limit_sim, velocity_limit_sim=_b.velocity_limit_sim,
                              stiffness=_b.stiffness, damping=_b.damping, armature=_b.armature,
                              play_rad=_m3.radians(args_cli.knee_play_deg), play_rad_range=None)
    _pb = getattr(env_cfg.events, "default_pose_bias", None)
    if _pb is not None and "shared_knee_rad" in _pb.params and not args_cli.knee_keep_bias:
        _pb.params["shared_knee_rad"] = None
    print(f"[knee_play] 무릎 유격 반폭 {args_cli.knee_play_deg}° · 공유 무릎 편향 {'유지' if args_cli.knee_keep_bias else '끔'}", flush=True)
if args_cli.contact_real:
    from rl_walking import mdp_hang as _mh5
    _fc5 = env_cfg.observations.policy.feet_contact
    _kp5 = {k: v for k, v in dict(_fc5.params).items() if k in ("sensor_cfg", "threshold", "threshold_range")}
    _kp5.update({"hold_ticks": 5, "extra_delay": 1, "tilt_off_deg": (6.0, 20.0)})
    _fc5.func = _mh5.feet_contact_real; _fc5.params = _kp5
    print("[contact_real] 정책 접촉 관측 = 실물 계약(평평·10 Hz·1틱)", flush=True)
if args_cli.last_action_ema > 0:
    _LA = env_cfg.observations.policy.actions; _la_orig = _LA.func; _ALPHA = float(args_cli.last_action_ema)
    def _la_ema(env, action_name=None):
        x = _la_orig(env) if action_name is None else _la_orig(env, action_name=action_name)
        st = getattr(env, "_la_ema_st", None); step = int(env.common_step_counter)
        if st is None or st["y"].shape != x.shape:
            st = {"y": x.clone(), "step": None}; env._la_ema_st = st
        if st["step"] != step:
            fresh = env.episode_length_buf == 0
            st["y"] = torch.where(fresh.view(-1, 1), x, st["y"] + _ALPHA * (x - st["y"])); st["step"] = step
        return st["y"].clone()
    _LA.func = _la_ema
    print(f"[last_action_ema] 정책 last_action 관측 = EMA α{_ALPHA} (진단 전용, 배포 계약 아님)")
if args_cli.gyro_delay > 0:
    from collections import deque as _dq
    from rl_walking import mdp_hang as _mh4
    _GD = int(args_cli.gyro_delay)
    _P4 = env_cfg.observations.policy
    _ga = _P4.base_ang_vel; _ga_orig = _ga.func
    def _delayed(env, key, x):
        st = getattr(env, key, None)
        step = int(env.common_step_counter)
        if st is None:
            st = {"q": _dq(maxlen=_GD + 1), "step": None, "out": x}; setattr(env, key, st)
        if st["step"] != step:
            st["q"].append(x.clone()); st["step"] = step
            st["out"] = st["q"][0]
        return st["out"]
    def _ang_delayed(env, asset_cfg=None):
        x = _ga_orig(env) if asset_cfg is None else _ga_orig(env, asset_cfg=asset_cfg)
        return _delayed(env, "_gd_ang", x).clone()
    _ga.func = _ang_delayed
    _ge = getattr(_P4, "gyro_ema", None)
    if _ge is not None:
        def _gyro_ema_delayed(env, alpha: float = 0.25, noise: float = 0.2, asset_cfg=None):
            asset = env.scene[(asset_cfg or _mh4._base_cfg()).name]
            x = _delayed(env, "_gd_ema_src", asset.data.root_ang_vel_b.clone())
            if noise > 0.0:
                x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
            return _mh4._ema_state(env, "_s8_gyro_ema", x, alpha).clone()
        _ge.func = _gyro_ema_delayed
    print(f"[gyro_delay] 자이로 관측 {_GD} 틱 지연", flush=True)
if args_cli.contact_drop:
    _CD = []
    for _it in args_cli.contact_drop.split(";"):
        _side, _t, _d = _it.split(":")
        _CD.append(({"L": [0], "R": [1], "B": [0, 1]}[_side.strip().upper()], float(_t), float(_d)))
    _fc = getattr(env_cfg.observations.policy, "feet_contact", None)
    if _fc is None:
        raise SystemExit("--contact_drop: 정책 관측에 feet_contact 항 없음")
    _fc_orig = _fc.func

    import inspect as _insp
    _fc_names = set(_insp.signature(_fc_orig).parameters) - {"env"}

    def _fc_drop(env, sensor_cfg=None, threshold: float = 5.0, delay_steps: int = 0, threshold_range=None,
                 hold_ticks=None, extra_delay=None, tilt_off_deg=None):
        # 1004 밤: --contact_real(feet_contact_real: hold_ticks·extra_delay·tilt_off_deg) 와 기본 feet_contact 모두 지원 — 원 함수가 받는 인자만 전달
        _kw = dict(sensor_cfg=sensor_cfg, threshold=threshold, delay_steps=delay_steps, threshold_range=threshold_range,
                   hold_ticks=hold_ticks, extra_delay=extra_delay, tilt_off_deg=tilt_off_deg)
        out = _fc_orig(env, **{k: v for k, v in _kw.items() if k in _fc_names and v is not None}).clone()
        t_rel = env.episode_length_buf.to(out.dtype) * float(env.step_dt) - float(args_cli.precheck_s)
        for cols, t0_, d_ in _CD:
            m = (t_rel >= t0_) & (t_rel < t0_ + d_)
            if bool(m.any()):
                for c_ in cols:
                    out[m, c_] = 0.0
        return out
    _fc.func = _fc_drop
    print(f"[contact_drop] 정책 접촉 비트 강제 0: {args_cli.contact_drop}", flush=True)
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
if args_cli.min_height is not None and getattr(env_cfg.terminations, "base_too_low", None) is not None:
    env_cfg.terminations.base_too_low.params["minimum_height"] = float(args_cli.min_height); print(f"[min_height] base_too_low {args_cli.min_height} m (진단)", flush=True)
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
policy2 = None
if args_cli.switch_to and policy is not None:
    _r2, _c2 = args_cli.switch_to.split(":")
    _p2 = get_checkpoint_path(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name), _r2, _c2)
    runner2 = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner2.load(_p2); policy2 = runner2.get_inference_policy(device=dev)
    _SW_ON = [float(x) for x in args_cli.sw_on.split(",")]; _SW_OFF = [float(x) for x in args_cli.sw_off.split(",")]
    sw_mode = torch.zeros(N, dtype=torch.bool, device=dev); sw_t_on = torch.zeros(N, device=dev); sw_calm = torch.zeros(N, device=dev)
    sw_ticks = torch.zeros(N, device=dev); swc_n = torch.zeros(N, device=dev); sw_ticks_pre_dist = torch.zeros(N, device=dev)
    swc_first = torch.full((N,), float('nan'), device=dev); swc_pre = torch.zeros(N, device=dev)   # 1004 SW-1b: 첫 켜짐 시각(인계 기준)·외란 전 켜짐 수 (sw_n 은 스윙 계측과 이름 충돌 — 분리)
    _HW = 10   # 1004 C5 handoff 계측: 첫 외란 뒤 전환 전후 ±_HW 틱
    hof_ring_obs = None; hof_ring_a = None; hof_k = torch.full((N,), -1, dtype=torch.long, device=dev)
    HOF = {}
    print(f"[switch] 주 정책 {resume_path} ↔ 회복 정책 {_p2} · 켜기 {_SW_ON} · 끄기 {_SW_OFF}", flush=True)

obs = env.get_observations()
_OBS_EMA = []; _OBS_EMA_ST = {}
if args_cli.obs_ema:
    _names = list(uenv.observation_manager.active_terms["policy"]); _dims = uenv.observation_manager.group_obs_term_dim["policy"]
    _off = {}; _o = 0
    for _n, _d in zip(_names, _dims):
        _sz = 1
        for _v in (_d if isinstance(_d, (tuple, list)) else (_d,)): _sz *= int(_v)
        _off[_n] = (_o, _o + _sz); _o += _sz
    for _tok in args_cli.obs_ema.split(","):
        _n, _al = _tok.split(":"); _OBS_EMA.append((*_off[_n], float(_al)))
    _OBS_LAYOUT = dict(_off)
    print(f"[obs_ema] {[(n, _off[n]) for n in [t.split(':')[0] for t in args_cli.obs_ema.split(',')]]} 총 {_o}", flush=True)
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
_REPLAY = None; _REPLAY_I = [0]; _REAL_DEF = None
if args_cli.replay_bb:
    import json as _js
    from rl_walking.hang_stand_env_cfg import CROUCH_V5_DEFAULT_RAD as _CV5
    _rows = [_js.loads(l) for l in open(args_cli.replay_bb)]
    _rows = [r for r in _rows if "event" not in r and r.get("targets_deg_joint")]
    _t0r = _rows[0]["timestamp"] + float(args_cli.replay_t0)
    _rows = [r for r in _rows if r["timestamp"] >= _t0r]
    _jn = list(robot.joint_names)
    _REPLAY = torch.tensor([[math.radians(r["targets_deg_joint"][j]) for j in _jn] for r in _rows], device=dev, dtype=torch.float32)
    _REAL_DEF = torch.tensor([_CV5[j] for j in _jn], device=dev, dtype=torch.float32)
    print(f"[replay] {args_cli.replay_bb} · {len(_rows)} 틱({len(_rows)/50:.1f} s) 재생, 시작 +{args_cli.replay_t0}s", flush=True)
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
        if _REPLAY is not None:
            k = min(_REPLAY_I[0], _REPLAY.shape[0] - 1); _REPLAY_I[0] += 1
            out = _DEF + (_REPLAY[k] - _REAL_DEF).unsqueeze(0)
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
_OSC_RING = []; osc_rev_max = torch.zeros(N, device=dev); osc_flag_any = torch.zeros(N, dtype=torch.bool, device=dev); osc_flag_rec = torch.zeros(N, dtype=torch.bool, device=dev); osc_flag_hipf = torch.zeros(N, dtype=torch.bool, device=dev)
_HIPF = [list(robot.data.joint_names).index(n) for n in ('left_hip_f_joint', 'right_hip_f_joint')]
fall_pitch_prev = torch.zeros(N, device=dev); fall_roll_prev = torch.zeros(N, device=dev)
term_reason = torch.full((N,), -1, dtype=torch.long, device=dev); fall_h_prev = torch.zeros(N, device=dev)
lat_sep0 = torch.zeros(N, device=dev); lat_sep_min = torch.full((N,), 9.0, device=dev); lat_roll_max = torch.zeros(N, device=dev); lat_v_max = torch.zeros(N, device=dev); lat_ss_t = torch.zeros(N, device=dev)
t_sep_min = torch.full((N,), float('nan'), device=dev); t_roll8 = torch.full((N,), float('nan'), device=dev); t_sep18 = torch.full((N,), float('nan'), device=dev); h_min_post = torch.full((N,), 9.0, device=dev)
td_sep_s = torch.zeros(N, device=dev); td_sep_n = torch.zeros(N, device=dev); td_sep_min = torch.full((N,), 9.0, device=dev)
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
# 1005 실패 5분류 계측(첫 외란 기준): 첫 이륙·첫 착지·첫 하중 수용 시각, cue 종료 뒤 이륙 수·이동, 회복 모드 해제 시각, 서기 복귀 뒤 진동
_NAN_ = lambda: torch.full((N,), float('nan'), device=dev)
to_first_t = _NAN_(); td_first_t = _NAN_(); load_first_t = _NAN_(); rec_off_t = _NAN_(); cue_end_rec = _NAN_()
to_after_cue = torch.zeros(N, device=dev); cue_xy0 = torch.full((N, 2), float('nan'), device=dev); xy_last = torch.zeros(N, 2, device=dev)
osc_flag_post = torch.zeros(N, dtype=torch.bool, device=dev)
# 1005 ③a 진단: 외란 뒤 1·2번째 착지의 캡처 여유(발 − ξ, 몸통 앞/왼축, cm)·걸음 길이·시각·피치·피치율
_MASS = robot.root_physx_view.get_masses().to(dev).float(); _MTOT = _MASS.sum(-1, keepdim=True)
td_k = torch.zeros(N, dtype=torch.long, device=dev)
TDM = {k: _NAN_() for k in ('t1','mf1','ml1','len1','p1','pr1','t2','mf2','ml2','len2','p2','pr2')}
early_hold = torch.zeros(N, device=dev)
n_land = torch.zeros(N, device=dev); n_load = torch.zeros(N, device=dev)
st_rl = torch.zeros(N, 2, dtype=torch.bool, device=dev); n_relift = torch.zeros(N, device=dev); n_loadfail_stay = torch.zeros(N, device=dev)   # 1004: 하중 실패 분해 — 0.3 s 창 안 재이륙 / 붙어 있었는데 하중 미달
n_to_lr = torch.zeros(2, device=dev); sw_sq = torch.zeros(N, device=dev)   # 1003 G22R 리포트: 좌/우 이륙 수·스텝 변위 제곱합(표준편차)
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
        if policy2 is not None and i >= STEPS_PRE:
            _pg = robot.data.projected_gravity_b
            _tilt = torch.rad2deg(torch.acos((-_pg[:, 2]).clamp(-1.0, 1.0))); _w = robot.data.root_ang_vel_b[:, :2].norm(dim=-1)
            _cf = _cs.data.net_forces_w[:, [_fl, _fr], :].norm(dim=-1) > 5.0
            if args_cli.contact_drop:      # 1004 밤: 패드 고장은 감지기 접촉에도 같게(실물 감지기 = 패드 비트)
                for _cols, _t0, _d in _CD:
                    if _t0 <= t_rel < _t0 + _d:
                        for _c in _cols:
                            _cf[:, _c] = False
            if float(args_cli.sw_ref_tau) > 0.0:      # 1004 상대 기울기 감지기(실물 서기 기울기 3–4° 대응)
                _pp_, _rr_ = _rp(); _rp2 = torch.stack([_rr_, _pp_], -1)
                if i == STEPS_PRE or "sw_ref" not in dir():
                    sw_ref = _rp2.clone()
                if t_rel < float(args_cli.sw_ref_init_s):
                    sw_ref = _rp2.clone()
                _upd = (~sw_mode) & (_w < 0.1) & _cf[:, 0] & _cf[:, 1]
                sw_ref = torch.where(_upd.unsqueeze(1), sw_ref + (_rp2 - sw_ref) * (0.02 / float(args_cli.sw_ref_tau)), sw_ref).clamp(-8.0, 8.0)
                _tilt = (_rp2 - sw_ref).norm(dim=-1)
            _on = (_tilt > _SW_ON[0]) | (_w > _SW_ON[1])
            _cq = {"both": _cf[:, 0] & _cf[:, 1], "any": _cf[:, 0] | _cf[:, 1], "none": torch.ones_like(_cf[:, 0])}[args_cli.sw_off_contact]
            _quiet = (_tilt < _SW_OFF[0]) & (_w < _SW_OFF[1]) & _cq
            sw_calm = torch.where(sw_mode & _quiet, sw_calm + 0.02, torch.zeros_like(sw_calm))
            sw_t_on = torch.where(sw_mode, sw_t_on + 0.02, torch.zeros_like(sw_t_on))
            _swoff = sw_mode & (sw_calm >= _SW_OFF[2]) & (sw_t_on >= _SW_OFF[3])
            if args_cli.sw_early:
                _ev, _et, _eh = [float(x) for x in args_cli.sw_early.split(',')]
                _ec = sw_mode & ~torch.isnan(load_first_t) & (robot.data.root_lin_vel_w[:, :2].norm(dim=-1) < _ev) & (_tilt < _et)
                early_hold = torch.where(_ec, early_hold + 0.02, torch.zeros_like(early_hold))
                _swoff = _swoff | (sw_mode & (early_hold >= _eh))
            if args_cli.sw_oracle and dists:
                _on = _on | torch.full_like(_on, any(d.t0 <= t_rel < d.t0 + 0.02 for d in dists))
            _new_on = (~sw_mode) & _on
            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff
            if dists and t_rel >= min(d.t0 for d in dists):
                rec_off_t = torch.where(_swoff & torch.isnan(rec_off_t), torch.full_like(rec_off_t, t_rel), rec_off_t)
            swc_first = torch.where(_new_on & torch.isnan(swc_first), torch.full_like(swc_first, t_rel), swc_first)
            if dists and t_rel < min(d.t0 for d in dists):
                swc_pre += _new_on.float()
            sw_ticks += (sw_mode & alive).float()
            if dists and t_rel < min(d.t0 for d in dists):
                sw_ticks_pre_dist += (sw_mode & alive).float()
            if args_cli.sw_rec_vx is not None:
                if "_VC_OFF" not in dir():
                    _nm_ = list(uenv.observation_manager.active_terms["policy"]); _dm_ = uenv.observation_manager.group_obs_term_dim["policy"]
                    _o_ = 0; _LAY_ = {}
                    for _n_, _d_ in zip(_nm_, _dm_):
                        _sz_ = 1
                        for _v_ in (_d_ if isinstance(_d_, (tuple, list)) else (_d_,)): _sz_ *= int(_v_)
                        _LAY_[_n_] = (_o_, _o_ + _sz_); _o_ += _sz_
                    _VC_OFF = _LAY_["velocity_commands"]; _GC_OFF = _LAY_.get("gait_clock")
                    _GC_F = 1.1
                    try:
                        _GC_F = float(uenv.observation_manager._group_obs_term_cfgs["policy"][_nm_.index("gait_clock")].params.get("freq", 1.1))
                    except Exception:
                        pass
                    cue_act = torch.zeros(N, dtype=torch.bool, device=dev); cue_done = torch.zeros(N, dtype=torch.bool, device=dev)
                    cue_hold = torch.zeros(N, device=dev); cue_t0 = torch.zeros(N, device=dev); cue_phi0 = torch.zeros(N, device=dev)
                    print(f"[sw_rec_vx] velocity_commands 열 {_VC_OFF} → vx {args_cli.sw_rec_vx} · 상태 종료 '{args_cli.cue_state}' · 클록 '{args_cli.cue_clock}' (gait_clock {_GC_OFF}, {_GC_F} Hz)", flush=True)
                # cue 상태기: 감지기 새로 켜짐 → 활성, 종료 조건 → 래치 해제(감지기 꺼지면 리셋)
                _st_on = _new_on & ~cue_act & ~cue_done
                if args_cli.cue_dir is not None:
                    _fwd_ = robot.data.root_lin_vel_b[:, 0] > float(args_cli.cue_dir)
                    cue_done = cue_done | (_st_on & ~_fwd_); _st_on = _st_on & _fwd_
                cue_act = cue_act | _st_on; cue_t0 = torch.where(_st_on, torch.full_like(cue_t0, t_rel), cue_t0)
                _Lf = _cs.data.net_forces_w[:, _fl, :].norm(dim=-1); _Rf = _cs.data.net_forces_w[:, _fr, :].norm(dim=-1)
                cue_phi0 = torch.where(_st_on, torch.where(_Lf < _Rf, torch.full_like(cue_phi0, math.pi + 0.05), torch.full_like(cue_phi0, 0.05)), cue_phi0)
                if args_cli.cue_align:
                    _amin, _adx, _amax = [float(x) for x in args_cli.cue_align.split(",")]
                    _fwa = _dir_vec("f"); _fpa = robot.data.body_pos_w[:, _FT, :2]
                    _dxa = 100.0 * ((_fpa[:, 0] - _fpa[:, 1]) * _fwa[:, :2]).sum(-1).abs()
                    _ela = t_rel - cue_t0; _bca = _cf[:, 0] & _cf[:, 1]
                    _end = cue_act & _bca & (((_ela >= _amin) & (_dxa < _adx)) | (_ela >= _amax))
                elif args_cli.cue_state:
                    _tmin, _hold, _wq, _tmax = [float(x) for x in args_cli.cue_state.split(",")]
                    _el = t_rel - cue_t0; _bc = _cf[:, 0] & _cf[:, 1]
                    cue_hold = torch.where(cue_act & _bc & (_w < _wq), cue_hold + 0.02, torch.zeros_like(cue_hold))
                    _end = cue_act & (((_el >= _tmin) & (cue_hold >= _hold)) | ((_el >= _tmax) & _bc))
                elif float(args_cli.sw_rec_vx_s) > 0.0:
                    _end = cue_act & ((t_rel - cue_t0) >= float(args_cli.sw_rec_vx_s))
                else:
                    _end = torch.zeros_like(cue_act)
                cue_end_rec = torch.where(_end & torch.isnan(cue_end_rec), torch.full_like(cue_end_rec, t_rel), cue_end_rec) if "cue_end_rec" in dir() else torch.where(_end, torch.full((N,), t_rel, device=dev), torch.full((N,), float('nan'), device=dev))
                cue_act = cue_act & ~_end; cue_done = (cue_done | _end) & sw_mode
                _vxm_ = cue_act & sw_mode
                _obs2 = obs.clone(); _p2o = _obs2["policy"].clone()
                _vc_ = _p2o[:, _VC_OFF[0]:_VC_OFF[1]:3]
                _p2o[:, _VC_OFF[0]:_VC_OFF[1]:3] = torch.where(_vxm_.unsqueeze(1), torch.full_like(_vc_, float(args_cli.sw_rec_vx)), _vc_)
                if args_cli.cue_clock == "reset" and _GC_OFF is not None:
                    _ph = cue_phi0 + 2.0 * math.pi * _GC_F * (t_rel - cue_t0)
                    _gc_ = _p2o[:, _GC_OFF[0]:_GC_OFF[1]].view(N, 5, 2)
                    _k_ = torch.arange(5, device=dev, dtype=torch.float32) - 4.0
                    _phk = _ph.unsqueeze(1) + _k_.unsqueeze(0) * (2.0 * math.pi * _GC_F * 0.02)
                    _new_gc = torch.stack([torch.sin(_phk), torch.cos(_phk)], -1)
                    _p2o[:, _GC_OFF[0]:_GC_OFF[1]] = torch.where(_vxm_.view(N, 1, 1), _new_gc, _gc_).reshape(N, 10)
                _obs2["policy"] = _p2o
                cue_ticks = (cue_ticks if "cue_ticks" in dir() else torch.zeros(N, device=dev)) + (_vxm_ & alive).float()
                _a2 = policy2(_obs2); _a1 = a
            else:
                _a2 = policy2(obs); _a1 = a
            if args_cli.sw_monitor:
                pass
            else:
                a = torch.where(sw_mode.unsqueeze(1), _a2, a)
            # ── C5 handoff 계측 ──
            _po = obs["policy"]
            if hof_ring_obs is None:
                hof_ring_obs = torch.zeros(_HW, N, _po.shape[1], device=dev); hof_ring_a = torch.zeros(_HW, N, a.shape[1], device=dev)
                _jn = robot.data.joint_pos.shape[1]
                for _k, _shp in (("obs", (2*_HW+1, _po.shape[1])), ("a_exec", (2*_HW+1, a.shape[1])), ("a1", (2*_HW+1, a.shape[1])), ("a2", (2*_HW+1, a.shape[1])),
                                 ("tilt", (2*_HW+1,)), ("w", (2*_HW+1,)), ("vxy", (2*_HW+1,)), ("q", (2*_HW+1, _jn)), ("qd", (2*_HW+1, _jn)), ("gyro", (2*_HW+1, 3)),
                                 ("fz", (2*_HW+1, 2)), ("mode", (2*_HW+1,)), ("t_sw", ())):
                    HOF[_k] = torch.full((N,) + _shp, float("nan"), device=dev)
            _first_after = _new_on & (hof_k < 0) & alive & ((not dists) or (t_rel >= min(d.t0 for d in dists)))
            if bool(_first_after.any()):
                hof_k[_first_after] = 0; HOF["t_sw"][_first_after] = t_rel
                for _j in range(_HW):     # 전환 전 _HW 틱(링: 가장 오래된 것부터)
                    _src = (i - _HW + _j) % _HW
                    HOF["obs"][_first_after, _j] = hof_ring_obs[_src, _first_after]; HOF["a_exec"][_first_after, _j] = hof_ring_a[_src, _first_after]
            _rec = (hof_k >= 0) & (hof_k <= _HW)
            if bool(_rec.any()):
                _ix = (_HW + hof_k[_rec])
                _ri = torch.nonzero(_rec).squeeze(1)
                _fz = _cs.data.net_forces_w[:, [_fl, _fr], 2]
                for _k, _v in (("obs", _po), ("a_exec", a), ("a1", _a1), ("a2", _a2), ("tilt", _tilt), ("w", _w), ("vxy", robot.data.root_lin_vel_w[:, :2].norm(dim=-1)),
                               ("q", robot.data.joint_pos - _DEF), ("qd", robot.data.joint_vel), ("gyro", robot.data.root_ang_vel_b), ("fz", _fz), ("mode", sw_mode.float())):
                    HOF[_k][_ri, _ix] = _v[_ri].to(HOF[_k].dtype)
                hof_k[_rec] += 1
            hof_ring_obs[i % _HW] = _po; hof_ring_a[i % _HW] = a
        if args_cli.freeze_after is not None and policy is not None and t_rel >= args_cli.freeze_after:
            if _FROZEN[0] is None:
                _FROZEN[0] = a.clone(); print(f"[freeze] t_rel={t_rel:.2f}s 정책 출력 고정", flush=True)
            a = _FROZEN[0]
        obs, _, dones, _ = env.step(a)
        if _OBS_EMA:
            _po = obs["policy"]
            for (_a0, _b0, _al) in _OBS_EMA:
                _x = _po[:, _a0:_b0]
                _st = _OBS_EMA_ST.get(_a0)
                _y = _x.clone() if _st is None else _st + _al * (_x - _st)
                _OBS_EMA_ST[_a0] = _y; _po[:, _a0:_b0] = _y
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
    if LEVEL_STEPS > 0 and float(args_cli.level_fade_s) > 0.0 and i >= STEPS_PRE and _lvl_base is not None:
        _fk = max(0.0, 1.0 - (i - STEPS_PRE) / (float(args_cli.level_fade_s) * 50.0))
        _lvl_term._offset[:, _LH] = _lvl_base[:, 0] + torch.deg2rad(_lvl) * _fk
        _lvl_term._offset[:, _RH] = _lvl_base[:, 1] - torch.deg2rad(_lvl) * _fk
    cmd = _LAST_CMD[0] if _LAST_CMD[0] is not None else _DEF
    # 1004 밤: 브리지 발진 가드 중주파 규칙(0.5 s 창, |Δ목표|>0.7° 스텝의 방향 반전 ≥9 & 진폭 ≥2.5°)을 명령 목표각에 적용 — 실물 9 Hz 힙 진동 재현 여부
    if i >= STEPS_PRE:
        _OSC_RING.append(cmd.clone())
        if len(_OSC_RING) > 25:
            _OSC_RING.pop(0)
        if len(_OSC_RING) >= 6:
            _W = torch.stack(_OSC_RING, 0)                      # (T, N, J)
            _dq = _W[1:] - _W[:-1]
            _big = _dq.abs() > math.radians(0.7)
            _sg = torch.sign(_dq) * _big
            # 연속한 '큰 스텝' 간 부호 반전 수(작은 스텝은 건너뜀 — 부호 이벤트 근사)
            _prev = torch.zeros_like(_sg[0]); _rev = torch.zeros_like(_sg[0])
            for _k in range(_sg.shape[0]):
                _s = _sg[_k]
                _rev += ((_s != 0) & (_prev != 0) & (_s != _prev)).float()
                _prev = torch.where(_s != 0, _s, _prev)
            _p2p = _W.max(0).values - _W.min(0).values
            _flag = (_rev >= 9) & (_p2p >= math.radians(2.5))      # (N, J)
            osc_rev_max = torch.maximum(osc_rev_max, _rev.max(1).values * alive.float())
            osc_flag_any = osc_flag_any | (_flag.any(1) & alive)
            if policy2 is not None:
                osc_flag_rec = osc_flag_rec | (_flag.any(1) & alive & sw_mode)
            osc_flag_hipf = osc_flag_hipf | (_flag[:, _HIPF].any(1) & alive)
            if policy2 is not None:
                osc_flag_post = osc_flag_post | (_flag.any(1) & alive & ~sw_mode & ~torch.isnan(rec_off_t))
    a_ = alive.clone()
    newly = a_ & term
    death_step[newly] = i; pitch_death[newly] = pitch[newly]; roll_death[newly] = roll[newly]
    # 1004: 종료 틱은 이미 자동 리셋된 상태(피치≈0) — 낙상 방향은 직전 틱 값으로
    if "_pp_prev" in dir():
        fall_pitch_prev[newly] = _pp_prev[newly]; fall_roll_prev[newly] = _rr_prev[newly]; fall_h_prev[newly] = _hh_prev[newly]
    _pp_prev = pitch.clone(); _rr_prev = roll.clone(); _hh_prev = robot.data.root_pos_w[:, 2].clone()
    if bool(newly.any()):     # 0928: 어떤 종료 조건이 걸렸는지 집계 (낙상 판정 검증용)
        _tm = uenv.termination_manager
        for _nm in _tm.active_terms:
            _v = _tm.get_term(_nm)
            _c = int((_v & newly).sum())
            if _c:
                _TERM_CNT[_nm] = _TERM_CNT.get(_nm, 0) + _c
                _sel = _v & newly
                term_reason[_sel & (term_reason < 0)] = list(_tm.active_terms).index(_nm)
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
    if i == STEPS_PRE:      # 1004: 인계 시점 램프 DR 값 기록(종료 뒤 자동 리셋이 재표본하므로 여기서)
        from rl_walking import mdp_hang as _mh2
        _hst = _mh2._state(uenv); _RAMP = (_hst["ramp_rate"].clone(), _hst["ramp_cap"].clone(), _hst["ramp_start"].clone())
    if i >= STEPS_PRE:      # ── 스텝 단계 집계 ──
        _fw = _dir_vec("f"); _fp = robot.data.body_pos_w[:, _FT, :2] - robot.data.root_pos_w[:, None, :2]
        _ffwd = (_fp * _fw[:, None, :2]).sum(-1)               # (N,2) 발-골반 앞축 거리
        # 1004 좌우 진단: 발 횡간격(몸통 좌우축 투영, 왼−오, + = 정상 배치)·롤·골반 횡속도 — 첫 외란 시작 뒤
        _fl_ = _dir_vec("l"); _fy = (_fp * _fl_[:, None, :2]).sum(-1)
        _sep = _fy[:, 0] - _fy[:, 1]
        if i == STEPS_PRE:
            lat_sep0[:] = _sep
        if (not dists) or t_rel >= min(d.t0 for d in dists):
            _bothc = contact[:, 0] & contact[:, 1]
            _ok_ = a_ & ~term
            _newmin = _ok_ & _bothc & (_sep < lat_sep_min)
            t_sep_min = torch.where(_newmin, torch.full_like(t_sep_min, t_rel), t_sep_min)
            lat_sep_min = torch.where(_ok_ & _bothc, torch.minimum(lat_sep_min, _sep), lat_sep_min)
            _r8 = _ok_ & (roll.abs() > 8.0) & torch.isnan(t_roll8)
            t_roll8 = torch.where(_r8, torch.full_like(t_roll8, t_rel), t_roll8)
            _n18 = _ok_ & _bothc & (_sep < 0.18) & torch.isnan(t_sep18)
            t_sep18 = torch.where(_n18, torch.full_like(t_sep18, t_rel), t_sep18)
            h_min_post = torch.where(_ok_, torch.minimum(h_min_post, robot.data.root_pos_w[:, 2]), h_min_post)
            lat_roll_max = torch.where(_ok_, torch.maximum(lat_roll_max, roll.abs()), lat_roll_max)
            _vl = (robot.data.root_lin_vel_w[:, :2] * _fl_[:, :2]).sum(-1).abs()
            lat_v_max = torch.where(_ok_, torch.maximum(lat_v_max, _vl), lat_v_max)
            _ss = _ok_ & (contact[:, 0] ^ contact[:, 1])
            lat_ss_t += _ss.float() * 0.02
        for c in (0, 1):
            off = a_ & ever_both & ~st_air[:, c] & ~contact[:, c]
            _rl = off & (st_land_t[:, c] >= 0) & ((t_rel - st_land_t[:, c]) <= 0.3)
            st_rl[_rl, c] = True                                   # 창 안 재이륙 태그(판정 정의는 종전 그대로)
            st_air[off, c] = True; st_t0[off, c] = t_rel; st_f0[off, c] = _ffwd[off, c]; st_alive0[off, c] = True
            n_takeoff += off.float()
            n_to_lr[c] += off.float().sum()
            on = a_ & st_air[:, c] & contact[:, c]
            st_air[on, c] = False; st_land_t[on, c] = t_rel; st_rl[on, c] = False; st_load_sum[on, c] = 0.0; st_load_n[on, c] = 0.0
            n_land += on.float()
            if dists and t_rel >= min(d.t0 for d in dists) and bool((on & (td_k < 2)).any()):
                _cpw = robot.data.body_com_pos_w if hasattr(robot.data, 'body_com_pos_w') else robot.data.body_pos_w
                _cvw = robot.data.body_com_lin_vel_w if hasattr(robot.data, 'body_com_lin_vel_w') else robot.data.body_lin_vel_w
                _com = (_cpw * _MASS[..., None]).sum(1) / _MTOT; _cv = (_cvw * _MASS[..., None]).sum(1) / _MTOT
                _fpw = robot.data.body_pos_w[:, _FT[c], :]
                _om = torch.sqrt(9.81 / (_com[:, 2] - _fpw[:, 2]).clamp(min=0.2))
                _xi = _com[:, :2] + _cv[:, :2] / _om[:, None]; _dm = _fpw[:, :2] - _xi
                _mf = 100.0 * (_dm * _fw[:, :2]).sum(-1); _ml = 100.0 * (_dm * _fl_[:, :2]).sum(-1)
                _pr = robot.data.root_ang_vel_b[:, 1]
                for _kk in (1, 2):
                    _m = on & (td_k == _kk - 1)
                    for _nm, _v in (('t', torch.full_like(_mf, t_rel)), ('mf', _mf), ('ml', _ml), ('len', 100.0 * (_ffwd[:, c] - st_f0[:, c])), ('p', pitch), ('pr', _pr)):
                        TDM[f'{_nm}{_kk}'] = torch.where(_m, _v, TDM[f'{_nm}{_kk}'])
                td_k = td_k + on.long()
            if dists and t_rel >= min(d.t0 for d in dists):
                to_first_t = torch.where(off & torch.isnan(to_first_t), torch.full_like(to_first_t, t_rel), to_first_t)
                td_first_t = torch.where(on & ~torch.isnan(to_first_t) & torch.isnan(td_first_t), torch.full_like(td_first_t, t_rel), td_first_t)
                to_after_cue += (off & ~torch.isnan(cue_end_rec)).float()
            if "_sep" in dir():
                td_sep_s += torch.where(on, _sep, torch.zeros_like(_sep)); td_sep_n += on.float()
                td_sep_min = torch.where(on, torch.minimum(td_sep_min, _sep), td_sep_min)
            sw_travel[on] += (_ffwd[on, c] - st_f0[on, c]); sw_n += on.float()
            sw_sq[on] += (_ffwd[on, c] - st_f0[on, c]) ** 2
            # 생존 이륙: 이륙 후 0.3 s 이상 살아 있음(낙상 부수 접촉상실과 구분)
            surv = a_ & st_alive0[:, c] & ((t_rel - st_t0[:, c]) >= 0.3)
            n_surv_takeoff += surv.float(); st_alive0[surv, c] = False
            # 재접촉 후 0.3 s 하중 적분
            win = a_ & (st_land_t[:, c] >= 0) & ((t_rel - st_land_t[:, c]) <= 0.3)
            st_load_sum[win, c] += f[win, c]; st_load_n[win, c] += 1.0
            done = a_ & (st_land_t[:, c] >= 0) & ((t_rel - st_land_t[:, c]) > 0.3) & (st_load_n[:, c] > 0)
            ok = done & ((st_load_sum[:, c] / st_load_n[:, c].clamp(min=1)) >= 0.3 * BW)
            load_first_t = torch.where(ok & ~torch.isnan(td_first_t) & torch.isnan(load_first_t), torch.full_like(load_first_t, t_rel), load_first_t)
            n_load += ok.float(); n_relift += (done & ~ok & st_rl[:, c]).float(); n_loadfail_stay += (done & ~ok & ~st_rl[:, c]).float()
            st_land_t[done, c] = -1.0; st_rl[done, c] = False
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
    xy_last = torch.where((alive & ~term).unsqueeze(1), robot.data.root_pos_w[:, :2], xy_last)
    cue_xy0 = torch.where((~torch.isnan(cue_end_rec) & torch.isnan(cue_xy0[:, 0])).unsqueeze(1), xy_last, cue_xy0)
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
if policy2 is not None:
    _m = pre_pass
    P(f"[전환] 회복 정책 사용 시간 비율 {100*float(sw_ticks[_m].sum())/max(1.0, float(_m.sum())*T_after*50):.1f} % · 켜짐 횟수 env 평균 {float(swc_n[_m].mean()):.2f} · 한 번이라도 켜진 env {100*float((swc_n[_m]>0).float().mean()):.1f} % · 첫 켜짐 시각 중앙 {float(swc_first[_m & ~torch.isnan(swc_first)].median()) if bool((_m & ~torch.isnan(swc_first)).any()) else float('nan'):.2f} s"
      + (f" · 외란 전 켜짐(오탐) env {100*float((sw_ticks_pre_dist[_m]>0).float().mean()):.1f} %" if dists else ""))
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
P(f"[하중 실패 분해] 재접촉 {_ld} = 하중 지지 {_lo} + 미달·창 안 같은 발 재이륙 {int(n_relift.sum())} + 미달·붙어 있음 {int(n_loadfail_stay.sum())} + 창 중 낙상/종료 {_ld - _lo - int(n_relift.sum()) - int(n_loadfail_stay.sum())}")
P(f"[스텝 단계 분해] 이륙 {_tk}회 → 생존 이륙(0.3 s 이상 생존) {_sv} → 재접촉 {_ld} → 하중 지지(0.3 s 평균 ≥30 % 체중) {_lo}"
  + (f" · 재접촉한 발의 앞축 이동 평균 {100*float(sw_travel.sum()/sw_n.sum().clamp(min=1)):+.1f} cm" if float(sw_n.sum()) > 0 else "")
  + f" · 실패한 시도(이륙했으나 재접촉 못 함) {_tk - _ld}")
if float(sw_n.sum()) > 0:
    _m = float(sw_travel.sum() / sw_n.sum()); _sd = (max(float(sw_sq.sum() / sw_n.sum()) - _m * _m, 0.0)) ** 0.5
    P(f"[스텝 통계] 재접촉 발 앞축 이동 평균 {100*_m:+.1f} cm · 표준편차 {100*_sd:.1f} cm · 이륙 왼/오 {int(n_to_lr[0])}/{int(n_to_lr[1])}"
      + (f" (왼 비율 {100*float(n_to_lr[0])/max(float(n_to_lr.sum()),1):.0f} %)" if float(n_to_lr.sum()) > 0 else ""))
P("시간축 (인계 기준 s | 생존 env 평균 피치 | 생존 비율 | 외란 힘 N | 양발접촉 비율):")
for t, p_, al, fN, bc in pitch_tl:
    P(f"   {t:6.1f}  {p_:+6.2f}  {al:5.2f}  {fN:5.0f}  {bc:5.2f}")
P(f"[종료 조건] {_TERM_CNT if _TERM_CNT else '없음'}")
txt = "\n".join(L)
print(txt)
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"ground_recovery_{args_cli.out_tag}.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
import csv  # noqa: E402
if policy2 is not None and HOF:
    import numpy as _np
    _np.savez_compressed(os.path.join(REPO, "log_picture", f"ground_recovery_{args_cli.out_tag}_handoff.npz"),
                         fell=fell.cpu().numpy(), pre_pass=pre_pass.cpu().numpy(), monitor=bool(args_cli.sw_monitor), obs_layout=json.dumps(_OBS_LAYOUT) if "_OBS_LAYOUT" in dir() else "",
                         **{k: v.cpu().numpy() for k, v in HOF.items()})
with open(os.path.join(REPO, "log_picture", f"ground_recovery_{args_cli.out_tag}_envs.csv"), "w", newline="") as fh:
    wcsv = csv.writer(fh)
    _bcols = [f"bias_{n}" for n in robot.joint_names] if _BIAS0 is not None else []
    _ccols = ["com_x", "com_y", "com_z"] if _COM0 is not None else []
    wcsv.writerow(["env", "pre_pass", "pre_pitch", "fell", "t_fall_rel", "alive_end", "final_quiet", "lifts_total"] + [f"d{k+1}_{s}" for k in range(D) for s in ("fell", "lifts", "recovered", "t_rec", "pmax")] + _bcols + _ccols + [f"p2p_{k}" for k in range(NB_OSC)] + ["takeoffs", "surv_takeoffs", "swing_travel_mean_cm", "fall_pitch", "fall_roll", "sep0_cm", "sep_min_ds_cm", "roll_max_post", "vlat_max", "single_support_s", "term_reason", "fall_h_prev", "t_sep_min", "t_sep18", "t_roll8", "h_min_post", "td_sep_mean_cm", "td_sep_min_cm", "ramp_rate", "ramp_cap", "ramp_start", "osc_rev_max", "osc_flag", "osc_flag_rec", "osc_flag_hipf"] + (["sw_on_n", "sw_first_t", "sw_on_pre", "sw_time_s", "cue_end_t", "cue_time_s", "to_first_t", "td_first_t", "load_first_t", "to_after_cue", "drift_after_cue_cm", "rec_off_t", "osc_flag_post"] if policy2 is not None else []) + ["td1_t", "td1_mf_cm", "td1_ml_cm", "td1_len_cm", "td1_pitch", "td1_prate", "td2_t", "td2_mf_cm", "td2_ml_cm", "td2_len_cm", "td2_pitch", "td2_prate"])
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
                (round(float(sw_travel[e] / sw_n[e]) * 100.0, 2) if float(sw_n[e]) > 0 else ""),
                (round(float(fall_pitch_prev[e]), 1) if fell[e] else ""), (round(float(fall_roll_prev[e]), 1) if fell[e] else ""),
                round(100 * float(lat_sep0[e]), 1), (round(100 * float(lat_sep_min[e]), 1) if float(lat_sep_min[e]) < 8.0 else ""),
                round(float(lat_roll_max[e]), 1), round(float(lat_v_max[e]), 3), round(float(lat_ss_t[e]), 2),
                (list(uenv.termination_manager.active_terms)[int(term_reason[e])] if int(term_reason[e]) >= 0 else ""), (round(float(fall_h_prev[e]), 3) if fell[e] else ""),
                *[("" if math.isnan(float(x[e])) else round(float(x[e]), 2)) for x in (t_sep_min, t_sep18, t_roll8)], round(float(h_min_post[e]), 3),
                (round(100 * float(td_sep_s[e] / td_sep_n[e]), 1) if float(td_sep_n[e]) > 0 else ""), (round(100 * float(td_sep_min[e]), 1) if float(td_sep_min[e]) < 8 else ""),
                round(float(_RAMP[0][e]), 3), round(float(_RAMP[1][e]), 2), round(float(_RAMP[2][e]), 2),
                int(osc_rev_max[e]), int(osc_flag_any[e]), int(osc_flag_rec[e]), int(osc_flag_hipf[e])]
        if policy2 is not None:
            row += [float(swc_n[e]), (round(float(swc_first[e]), 2) if not math.isnan(float(swc_first[e])) else ""), float(swc_pre[e]), round(float(sw_ticks[e]) / 50.0, 2)]
            _fx_ = lambda v: "" if math.isnan(float(v)) else round(float(v), 2)
            _ct_ = (float(cue_ticks[e]) / 50.0) if "cue_ticks" in dir() else 0.0
            _dr_ = (100.0 * float((xy_last[e] - cue_xy0[e]).norm())) if not math.isnan(float(cue_xy0[e, 0])) else ""
            row += [_fx_(cue_end_rec[e]), round(_ct_, 2), _fx_(to_first_t[e]), _fx_(td_first_t[e]), _fx_(load_first_t[e]), float(to_after_cue[e]), (round(_dr_, 1) if _dr_ != "" else ""), _fx_(rec_off_t[e]), int(osc_flag_post[e])]
        row += [("" if math.isnan(float(TDM[k][e])) else round(float(TDM[k][e]), 2)) for k in ('t1','mf1','ml1','len1','p1','pr1','t2','mf2','ml2','len2','p2','pr2')]
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
