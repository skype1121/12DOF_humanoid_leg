#!/usr/bin/env python3
"""Stage8 12-axis AK MIT real-control node.

Current deployment uses the main PC, ROS2 topics, and SocketCAN (default
`can0`; 젯슨 라이브 버스는 `can1` — 젯슨에서는 --can-channel can1 로 기동).

프레임 규약 (2026-07-27 실측 캘리브레이션):
  baseline_deg·latest_actual_deg·SET_JOINT_TARGET 상대각은 전부 '모터 인코더
  프레임' deg. config/joint_limits_12dof.json 리밋 표는 'URDF 관절 프레임' —
  절대각 리밋 검사 전 joint_deg = direction × (motor_deg − stand_zero_deg)
  변환 필수 (아래 _load_frame_constants / _set_joint_target 참조).
"""

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocols.ak_mit_command import (  # noqa: E402
    TAU_FF_DEFAULT,
    V_DES_DEFAULT,
    pack_ak_mit_command,
)
from protocols.ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402
from robot_runtime.joint_limits import (  # noqa: E402
    check_absolute_deg,
    get_absolute_limits_deg,
)
from robot_runtime.dof12_mapping import (  # noqa: E402
    JOINT_NAMES_12DOF,
    get_isaac_dof_index,
    get_joint_config,
    get_joint_name_to_isaac_dof_index,
    default_config_path,
    get_joint_name_to_motor_id,
    validate_12dof_mapping,
)
# 같은 폴더의 단일 파일 모듈로 가져온다 — jetson/ 패키지 파일(__init__)은 배포 대상이 아니다
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(1, str(Path(__file__).resolve().parent))
from mech_fixed import load_fixed as load_mech_fixed  # noqa: E402


COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
JOINT_STATES_TOPIC = "/humanoid/joint_states"

REAL_CAN_WRITE_ENABLED = True
READ_ONLY_CAN_ENABLED = True
# 기본 'can0' (메인 PC 호환). 젯슨 라이브 버스는 'can1' — main() 의
# --can-channel 인자로 교체 가능 (모듈 전역이라 전 경로에 일괄 적용).
CAN_CHANNEL = "can0"
CAN_BITRATE = 1000000
CAN_RESTART_MS = 100
CAN_COMMAND_TIMEOUT_SEC = 3.0
#: CAN 송신(명령 프레임) 타임아웃 (초) — bus.send(timeout=…) 로 그대로 전달.
#: 기본 0.2 는 종전 하드코딩 값과 동일 — 기본 기동의 버스 거동은 바뀌지 않는다.
#: 선택형 진단 손잡이: H-D1(버스 정체로 send 가 프레임마다 블로킹 → 12관절
#: × 0.2 s 로 단일 스레드 실행기가 잠겨 status 갭) 판별 시
#: --can-send-timeout 0.01 로 재기동하면 정체가 '블로킹' 대신 '즉시 실패 +
#: can_send_timeouts 증가'로 드러난다. ⚠ 낮추면 실패 거동이 바뀐다(송신
#: 드롭) — 진단 목적 외 상시 사용은 별도 검토 대상.
CAN_SEND_TIMEOUT_SEC = 0.2
CAN_SEND_TIMEOUT_MIN_SEC = 0.001
CAN_SEND_TIMEOUT_MAX_SEC = 1.0
#: 송신 타임아웃 판정 문구 (python-can 4.6.1 socketcan 소스 실측: TX 큐가
#: timeout 안에 쓰기 가능해지지 않으면 CanOperationError("Transmit buffer
#: full")). 인터페이스마다 문구가 달라 '타임아웃 예산을 다 쓴 CanError'
#: 경로도 함께 본다 (_is_can_send_timeout). 그 외 CanError(예: 인터페이스
#: 다운)는 세지 않는다 — 세면 H-D1 을 거짓으로 확증하게 된다.
CAN_SEND_TIMEOUT_ERROR_MARKERS = (
    "transmit buffer full",
    "buffer full",
    "no buffer space",   # ENOBUFS = TX 큐 포화 (같은 정체 현상)
    "timeout",
    "timed out",
)
CAN_FEEDBACK_RECV_TIMEOUT_SEC = 0.0
CAN_FEEDBACK_MAX_FRAMES_PER_TICK = 64
CAN_FEEDBACK_REQUEST_TIMEOUT_SEC = 0.01
CAN_FEEDBACK_REQUEST_PERIOD_SEC = 0.02
CAN_FEEDBACK_TX_BACKOFF_SEC = 1.0
FC_READONLY_REQUEST_DATA = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]

CONTROL_PERIOD_SEC = 0.02
STATUS_PERIOD_SEC = 0.01
MAX_TARGET_STEP_DEG_PER_TICK = 4.0
#: 0930: 발행 → stage8 처리까지 이보다 늦은 목표 명령을 '늦음'으로 센다. 기록·표시 전용 — 명령 처리 동작은 바꾸지 않는다.
CMD_STALE_MS = 60.0

# ---------------------------------------------------------------------------
# 1006 실효 지연 단축 옵션 — 전부 기본 꺼짐(종전 동작 그대로). 배포 시 인자로 선택.
#   (지연 예산표·측정: jetson/bench/latency_mock/latency_bench.py 머리말, 커밋 메시지)
#   --feedback-after-send : 명령 프레임을 보낸 '직후' 같은 틱 안에서 그 응답(=피드백)을 최대
#       FEEDBACK_WAIT_SEC 기다려 읽고 status 를 바로 발행. 종전엔 이번 틱 응답을 다음 틱
#       (20 ms 뒤) 처음에야 읽었다 → 관측 나이 −1 틱. 송신 전 폴링(종전)은 그대로 둔다.
#   --send-on-command     : SET_JOINT_TARGETS 를 받은 즉시 송신(종전: 다음 제어 타이머까지
#       0–20 ms 대기). 명령이 끊기면 배경(감시) 송신이 종전 주기로 이어 받는다(공백 ≤ 30 ms).
#   --control-hz 100      : 제어(송신) 주기 50 → 100 Hz. 피드백도 10 ms 마다 갱신.
#   ※ 송신 주기를 바꾸는 옵션(--send-on-command·--control-hz 100)을 켜면 슬루 제한을 '틱당 4°' 대신 '시간당 200°/s(= 4°/20 ms), 틱당 최대 4°'로
#     적용한다 — 송신이 잦아져도 슬루 상한(가드)은 종전보다 느슨해지지 않는다(같거나 더 엄격).
# ---------------------------------------------------------------------------
FEEDBACK_AFTER_SEND = False
FEEDBACK_WAIT_SEC = 0.004   # 12 명령 + 12 응답 ≈ 24 프레임 × 130 µs ≈ 3.1 ms (1 Mbps) + 여유
FEEDBACK_WAIT_MIN_SEC = 0.0005
FEEDBACK_WAIT_MAX_SEC = 0.008
SEND_ON_COMMAND = False
#: 명령 즉시 송신 간 최소 간격 — DDS 적체가 한꺼번에 풀릴 때 CAN TX 큐 폭주(송신 블로킹, H-D1) 차단.
SEND_ON_COMMAND_MIN_INTERVAL_SEC = 0.004
#: 마지막 명령 즉시 송신 뒤 이 시간 안이면 '명령 스트림 살아 있음'(브리지 20 ms × 1.25).
#: 살아 있으면 50 Hz 모드의 배경 송신은 쉬고(명령이 송신을 맡음), 끊기면 배경 송신이 종전 주기로 이어 받는다.
COMMAND_STREAM_ACTIVE_SEC = 0.025
#: 배경(감시) 송신 판정 해상도·허용 오차 [s] — 5 ms 타이머로 '마지막 송신 뒤 주기 − 2.5 ms' 경과 시 송신.
BACKGROUND_CHECK_SEC = 0.005
BACKGROUND_TOL_SEC = 0.0025
#: 슬루 시간 상한 [deg/s] — 종전 틱당 4°·50 Hz 와 같은 값 (옵션 모드에서만 사용).
MAX_TARGET_SLEW_DEG_PER_SEC = MAX_TARGET_STEP_DEG_PER_TICK / 0.02
ALLOWED_CONTROL_HZ = (50, 100)


# ---------------------------------------------------------------------------
# 1010 요구 토크 상한 — 기본 꺼짐(0 = 종전 프레임과 바이트 동일). --target-torque-cap-nm X 로 켠다.
#   MIT 구동기는 kp·(목표−실측) 을 그대로 토크로 요구받는다. 시뮬 구동기는 한계(25 N·m)에서 조용히 자르지만 실물에는 그 자르기가 없다.
#   1010 실물: 걷기 켠 0.19 s 뒤 right_hip_a 목표−실측 14°(kp 150 → 약 37 N·m 요구) → 모터 8 응답 정지.
#   켜면 **보내는 프레임의 목표만** 최신 실측각 ± degrees(X / kp) 안으로 묶는다(위치 항 요구 토크 ≤ X). 내부 commanded 목표·슬루·가드는 그대로 —
#   시뮬의 토크 자르기와 같은 꼴(실측이 따라오면 목표가 풀린다). 실측이 낡은 관절·실측이 없는 관절은 묶지 않는다(종전 그대로 — 동결 유지 경로를 건드리지 않는다).
# ---------------------------------------------------------------------------
TARGET_TORQUE_CAP_NM = 0.0
TARGET_TORQUE_CAP_RANGE_NM = (10.0, 25.0)
#: 관절 이름에 든 문자열별 상한(--target-torque-cap-joint hip_a=20 — 여러 번 가능). 맞는 항목이 있으면 전체 상한 대신 그 값(여럿 맞으면 가장 작은 값).
#: 근거(시뮬 1010): 힙 외전 한계만 20 N·m 로 낮춰도 왕복 낙상 6 → 8/214, 전 관절을 20 으로 낮추면 60/418 — 그래서 관절별로 따로 준다.
TARGET_TORQUE_CAP_BY_NAME = {}


def parse_target_torque_cap(value):
    cap = float(value)
    if cap == 0.0:
        return 0.0
    lo, hi = TARGET_TORQUE_CAP_RANGE_NM
    if not (lo <= cap <= hi) or not math.isfinite(cap):
        raise ValueError(f"--target-torque-cap-nm {value} 불가 (0 = 끔, 또는 {lo:g}~{hi:g})")
    return cap


def parse_target_torque_cap_joints(items):
    """['hip_a=20', ...] → {'hip_a': 20.0}. 이름 조각은 12 관절 중 하나 이상에 들어 있어야 하고 값은 상한 범위 안(0 불가 — 끄려면 항목을 빼라)."""
    out = {}
    for item in items or []:
        key, sep, val = str(item).partition("=")
        key = key.strip()
        if not sep or not key or not any(key in j for j in JOINT_NAMES_12DOF):
            raise ValueError(f"--target-torque-cap-joint {item!r} 불가 (형식 이름조각=N·m, 이름조각은 관절명 일부)")
        cap = parse_target_torque_cap(val)
        if cap == 0.0:
            raise ValueError(f"--target-torque-cap-joint {item!r}: 0 불가")
        out[key] = cap
    return out


def torque_cap_enabled():
    return TARGET_TORQUE_CAP_NM > 0.0 or bool(TARGET_TORQUE_CAP_BY_NAME)


def torque_cap_err_deg(joint, cap_nm=None):
    """그 관절의 허용 |목표−실측| [deg] = degrees(cap / kp). 상한이 꺼졌거나 kp ≤ 0 이면 None."""
    cap = TARGET_TORQUE_CAP_NM if cap_nm is None else float(cap_nm)
    if cap_nm is None:
        named = [v for k, v in TARGET_TORQUE_CAP_BY_NAME.items() if k in joint]
        if named:
            cap = min(named)
    kp = float(get_joint_config(joint)["kp"])
    if cap <= 0.0 or kp <= 0.0:
        return None
    return math.degrees(cap / kp)


def time_based_slew_enabled():
    """송신 주기를 바꾸는 옵션(명령 즉시 송신·100 Hz)이 켜졌는가 → 슬루를 시간 기준으로.

    --feedback-after-send 단독은 송신 주기를 바꾸지 않으므로 종전 틱당 4° 그대로."""
    return bool(SEND_ON_COMMAND or abs(CONTROL_PERIOD_SEC - 0.02) > 1e-9)


def latency_options_status():
    return {"feedback_after_send": bool(FEEDBACK_AFTER_SEND),
            "feedback_wait_ms": round(FEEDBACK_WAIT_SEC * 1000.0, 3),
            "send_on_command": bool(SEND_ON_COMMAND),
            "control_hz": round(1.0 / CONTROL_PERIOD_SEC, 3),
            "time_based_slew": time_based_slew_enabled(),
            "slew_deg_per_sec": MAX_TARGET_SLEW_DEG_PER_SEC}


def parse_feedback_wait_ms(value):
    """--feedback-wait-ms 검증 → 초. 범위 밖·비유한값은 ValueError."""
    try:
        ms = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"--feedback-wait-ms 숫자 아님: {value!r}")
    sec = ms / 1000.0
    if not math.isfinite(sec) or not (FEEDBACK_WAIT_MIN_SEC <= sec <= FEEDBACK_WAIT_MAX_SEC):
        raise ValueError(
            f"--feedback-wait-ms 범위 밖: {value!r} "
            f"({FEEDBACK_WAIT_MIN_SEC * 1000:g}~{FEEDBACK_WAIT_MAX_SEC * 1000:g} ms)")
    return sec
FEEDBACK_TIMEOUT_SEC = 3.0
#: 피드백 공백 무장해제 — armed 상태에서 피드백 두절이 이 시간을 넘으면 자동
#: 무장해제+베이스라인 폐기 (0802 감사: 세션 중단 후 살아남은 armed 노드가
#: 수 시간 뒤 모터 전원 인가 순간, 전원사이클로 프레임까지 어긋난 낡은
#: 베이스라인으로 명령을 쏘는 사고 경로 차단. 정상 라이브는 피드백 공백이
#: 3초를 넘지 않으므로 영향 없음. 재무장 = 재베이스라인 강제)
STALE_DISARM_SEC = 60.0

#: 인수 홀드 — hang_stand 핸드오프가 남긴 마지막 명령(각도·게인) 상태 파일.
#: ⚠ 실측 확정(0802): AK 모터는 MIT ENTER(=FC 폴링 프레임)를 다시 받으면 물고
#: 있던 명령을 리셋하고 무토크가 된다 — 종전 "폴링 후 베이스라인" 인수는 폴링
#: 순간 모터가 하나씩 풀려 하중에 무릎이 처지는 사고를 냈음 (벤치: kp30 홀드가
#: ENTER 1발에 1초 내 낙하). 인수 홀드 = 기동 첫 틱부터 상태 파일의 명령을
#: 그대로 스트리밍 (명령 프레임이 피드백도 반환 → ENTER 폴링 불필요). ARM 시
#: 정상 경로로 무봉합 전환. armed 중에도 ENTER 폴링은 금지 (240ms마다 20ms
#: 미세 무토크 공백을 만들던 원인).
TAKEOVER_STATE_FILE = "/tmp/hang_handoff_state.json"
#: 적대검수 0803: 120s → 10s. 정상 핸드오프 소비는 <1s — 넓은 창은 '삭제
#: 실패 파일을 재기동 stage8이 재소비해 이동한 로봇을 옛 자세로 요격'하는
#: 창만 넓힌다.
TAKEOVER_STATE_MAX_AGE_SEC = 10.0
#: --wait-takeover: 선기동 대기 모드 — 핸드오프 '전에' stage8을 띄워두면 상태
#: 파일이 나타나는 즉시(한 틱 내) 인수 홀드를 시작한다. 대기 중에는 폴링 포함
#: 어떤 CAN 송신도 하지 않음 (ENTER가 hang_stand의 홀드를 리셋하므로).
#: 이로써 '핸드오프 → 프로세스 기동 4~8초' 무토크 공백(모터 자체 타임아웃
#: 릴리즈 → 하중 시 즉시 붕괴)이 ~20ms로 줄어든다.
WAIT_TAKEOVER = False

COMMAND_ARM_ALL = "ARM_ALL"
COMMAND_DISARM_ALL = "DISARM_ALL"
COMMAND_SET_BASELINE_ALL = "SET_BASELINE_FROM_CURRENT_ALL"
COMMAND_SET_JOINT_TARGET = "SET_JOINT_TARGET"
COMMAND_SET_JOINT_TARGETS = "SET_JOINT_TARGETS"   # 배치판 (지연 수술 0803)
COMMAND_STOP_ALL = "STOP_ALL"
COMMAND_CHECK_CAN = "CHECK_CAN"
COMMAND_RECONNECT_CAN = "RECONNECT_CAN"

CAN_ID_BY_JOINT = get_joint_name_to_motor_id()
JOINT_BY_CAN_ID = {motor_id: joint for joint, motor_id in CAN_ID_BY_JOINT.items()}
ISAAC_DOF_INDEX_BY_JOINT = get_joint_name_to_isaac_dof_index()
JOINT_BY_ISAAC_DOF_INDEX = {
    dof_index: joint for joint, dof_index in ISAAC_DOF_INDEX_BY_JOINT.items()
}
DEFAULT_LIMIT_GAIN_BY_JOINT = {
    joint: {
        "target_limit_deg": float(get_joint_config(joint)["target_limit_deg"]),
        "hard_limit_deg": float(get_joint_config(joint)["hard_limit_deg"]),
        "kp": float(get_joint_config(joint)["kp"]),
        "kd": float(get_joint_config(joint)["kd"]),
    }
    for joint in JOINT_NAMES_12DOF
}

# Stage8 accepts all 12 joint commands, while REAL_CAN_WRITE_ENABLED remains
# False by default so actual CAN writes stay blocked unless changed on Jetson.
ALLOWED_JOINTS = list(JOINT_NAMES_12DOF)

#: 기구 고정 관절 (1009) — 하드웨어맵 관절 항목의 선택 필드 "mech_fixed": true 로 **명시 선언**한 관절.
#: 모터를 떼고 관절 0°(= 모터 프레임 stand_zero_deg)에 굳힌 관절이라 CAN 에 응답하지 않는다. 이 관절만:
#:   - CAN 송신 전부 생략(FC 폴링·MIT 명령·인수 홀드),
#:   - 피드백 합성(실측 = stand_zero_deg, 나이 0, 신선·유효, 속도·토크 0) → 기준점·ARM·가드가 12 관절 그대로 돈다,
#:   - 들어온 목표는 검증만 하고 버린다(항상 stand_zero_deg 에 있다).
#: 선언이 없으면 빈 집합 = 종전과 완전히 같은 동작. 선언되지 않은 관절의 무응답·stale 가드는 그대로다(완화 없음).
#: 선언 값이 bool 이 아니면 여기서 ValueError(기동 차단). 원본: jetson/scripts/mech_fixed.py
MECH_FIXED_JOINTS = frozenset(load_mech_fixed(default_config_path())[0])
#: FC 폴링(읽기 요청) 대상 — 기구 고정 관절 제외. 선언 없으면 12 관절 그대로(종전 순서).
POLL_JOINTS = [joint for joint in JOINT_NAMES_12DOF if joint not in MECH_FIXED_JOINTS]


# ---------------------------------------------------------------------------
# 프레임 변환 상수 (2026-07-27 실측 캘리브레이션 — config/robot_12dof_hardware_map.json)
#   joint_deg = direction × (motor_deg − stand_zero_deg)   (모터 → URDF 관절 프레임)
# check_absolute_deg 의 리밋 표(config/joint_limits_12dof.json)는 URDF 관절 프레임 —
# 모터 인코더 절대각을 그대로 넣으면 안 되고 반드시 위 식으로 변환 후 검사한다.
# 주의: jetson/scripts/rl_bridge_node.py 에도 동일 규약의 로더가 있다 (두 노드는
# 서로 다른 머신에서 돌 수 있어 자체완결 유지) — 규약 변경 시 두 곳 동시 수정.
# ---------------------------------------------------------------------------

def _load_frame_constants():
    """하드웨어맵에서 direction/stand_zero_deg 로드 — 12관절 전수 검증, 결함 시 기동 차단."""
    direction_by_joint = {}
    stand_zero_by_joint = {}
    problems = []
    for joint in JOINT_NAMES_12DOF:
        config = get_joint_config(joint)
        joint_ok = True
        direction = config.get("direction")
        if isinstance(direction, bool) or direction not in (-1, 1):
            problems.append(f"{joint}: direction={direction!r} (−1/+1 만 허용)")
            joint_ok = False
        stand_zero = config.get("stand_zero_deg")
        if (isinstance(stand_zero, bool)
                or not isinstance(stand_zero, (int, float))
                or not math.isfinite(float(stand_zero))
                or abs(float(stand_zero)) >= 120.0):
            problems.append(
                f"{joint}: stand_zero_deg={stand_zero!r} (유한 float, |SZ|<120° 필요)")
            joint_ok = False
        if not joint_ok:
            continue
        direction_by_joint[joint] = int(direction)
        stand_zero_by_joint[joint] = float(stand_zero)
    if problems:
        raise ValueError(
            "하드웨어맵 프레임 상수 검증 실패 — 절대각 리밋 검사의 프레임 변환 불가, "
            "기동 중단:\n  " + "\n  ".join(problems))
    return direction_by_joint, stand_zero_by_joint


DIRECTION_BY_JOINT, STAND_ZERO_DEG_BY_JOINT = _load_frame_constants()

# 모터 프레임 백스톱 (2026-07-27 적대리뷰 수정): |모터절대각 − stand_zero| 가
# 해당 관절 하드리밋 '폭'+여유를 넘으면 무조건 거부하는 2차 방어선.
# 종전 rom_sweep2 절대 min/max 표는 사용 금지 — 그 스윕은 영점 앵커 前 인코더
# 프레임 기록이라 현재(v4) 프레임과 절대값이 다르다 (같은 이유로 당시 3층
# 대조도 '폭' 기준만 수행). 절대값을 그대로 쓰면 정상 보행 엔벨로프를 오거부해
# 스윙 중 다리를 얼릴 수 있었음 (좌무릎 −41° 초과 굽힘 거부 등, 4관점 리뷰
# 전원 CONFIRMED). '폭'은 프레임 무관 실측이고 12관절 모두 리밋이 0을 걸치므로
# |joint| ≤ 폭 이 항상 성립 → 정상 타깃 오거부 불가능. DIR 부호와 무관하게
# 성립해 방향 상수 오염도 잡는다 (SZ 오염은 관절 프레임 검사가 잡음).
MOTOR_DEV_BACKSTOP_DEG_BY_JOINT = {
    _j: (float(_lim["hard_max"]) - float(_lim["hard_min"])) + 6.0
    for _j, _lim in get_absolute_limits_deg().items()
}


@dataclass(frozen=True)
class Stage8MitPreview:
    motor_id: int
    joint: str
    isaac_dof_index: int
    p_des_rad: float
    v_des_rad_s: float
    kp: float
    kd: float
    tau_ff: float
    frame_bytes: list
    frame_hex: str


def parse_json_object(message_data):
    try:
        payload = json.loads(message_data)
    except json.JSONDecodeError as error:
        return None, f"invalid_json: {error}"
    if not isinstance(payload, dict):
        return None, "payload_not_object"
    return payload, ""


def parse_can_send_timeout(value):
    """--can-send-timeout 값 → float 초. 비수치·범위 밖은 ValueError.

    범위 0.001~1.0 s: 아래로는 select 해상도 밑이라 무의미, 위로는 12관절 ×
    타임아웃이 제어 틱(20 ms)을 통째로 삼키는 영역.
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid --can-send-timeout: {value!r} (초 단위 숫자)") from None
    if not math.isfinite(seconds):
        raise ValueError(f"invalid --can-send-timeout: {value!r} (유한한 숫자)")
    if not (CAN_SEND_TIMEOUT_MIN_SEC <= seconds <= CAN_SEND_TIMEOUT_MAX_SEC):
        raise ValueError(
            f"--can-send-timeout out of range: {seconds}"
            f" (허용 {CAN_SEND_TIMEOUT_MIN_SEC}~{CAN_SEND_TIMEOUT_MAX_SEC} 초)")
    return seconds


def _is_can_send_timeout(error, elapsed_sec=None, timeout_sec=None):
    """bus.send 예외가 '송신 타임아웃(TX 큐 정체)'인지 판정.

    can 모듈을 못 불러오는 환경(오프라인 테스트)에서는 문구만으로 판정한다.
    """
    try:
        import can as _can
    except Exception:  # noqa: BLE001 — can 미설치 환경은 문구로만 판정
        _can = None
    if _can is not None and not isinstance(error, _can.CanError):
        return False
    text = str(error).lower()
    if any(marker in text for marker in CAN_SEND_TIMEOUT_ERROR_MARKERS):
        return True
    # 문구를 모르는 인터페이스 대비: 타임아웃 예산을 다 쓰고 실패한 경우
    if elapsed_sec is not None and timeout_sec:
        return float(elapsed_sec) >= 0.9 * float(timeout_sec)
    return False


def _number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_12axis_mit_preview(joint, target_deg):
    config = get_joint_config(joint)
    target_rad = math.radians(float(target_deg))
    kp = float(config["kp"])
    kd = float(config["kd"])
    frame_bytes = pack_ak_mit_command(
        target_rad,
        v_des_rad_s=V_DES_DEFAULT,
        kp=kp,
        kd=kd,
        tau_ff=TAU_FF_DEFAULT,
    )
    frame_hex = " ".join(f"{value:02X}" for value in frame_bytes)
    return Stage8MitPreview(
        motor_id=int(config["motor_id"]),
        joint=joint,
        isaac_dof_index=get_isaac_dof_index(joint),
        p_des_rad=target_rad,
        v_des_rad_s=float(V_DES_DEFAULT),
        kp=kp,
        kd=kd,
        tau_ff=float(TAU_FF_DEFAULT),
        frame_bytes=frame_bytes,
        frame_hex=frame_hex,
    )


class CommandTiming:
    """브리지가 목표 명령에 넣는 seq·sent_ts·src 로 '발행 → stage8 처리' 지연·누락·출처 변경을 잰다 (0930).

    0929 실물: 이 구간이 200–220 ms 였고(하드웨어맵 재파싱으로 실행기가 밀림, 1e31a27 에서 수정) 아무 기록에도 직접 안 보였다.
    기록 전용 — 명령 수락·거부는 그대로. 필드가 없는 옛 발행자(콘솔·stage8_cmd·구 브리지)는 측정만 빠지고 그대로 통과."""

    def __init__(self):
        self.n = 0
        self.last_age_ms = None
        self.stale = 0
        self.gaps = 0
        self.regress = 0
        self.src = None
        self.src_changes = 0
        self.last_seq = None
        self._win = collections.deque()

    @staticmethod
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

    def update(self, now, payload):
        src = payload.get("src")
        if isinstance(src, str) and src != self.src:
            if self.src is not None:
                self.src_changes += 1
            self.src = src
            self.last_seq = None
        seq = payload.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            if self.last_seq is not None:
                if seq > self.last_seq + 1:
                    self.gaps += seq - self.last_seq - 1
                elif seq <= self.last_seq:
                    self.regress += 1
            self.last_seq = seq
        sent = payload.get("sent_ts")
        if self._num(sent):
            age = (float(now) - float(sent)) * 1000.0
            self.n += 1
            self.last_age_ms = round(age, 1)
            if age > CMD_STALE_MS:
                self.stale += 1
            self._win.append((float(now), age))
            self._prune(now)

    def _prune(self, now):
        while self._win and float(now) - self._win[0][0] > 1.0:
            self._win.popleft()

    def snapshot(self, now):
        self._prune(now)
        ages = sorted(a for _, a in self._win)
        return {
            "n": self.n, "last_age_ms": self.last_age_ms,
            "p50_age_ms_1s": round(ages[len(ages) // 2], 1) if ages else None,
            "max_age_ms_1s": round(ages[-1], 1) if ages else None,
            "count_1s": len(ages), "stale_over_ms": CMD_STALE_MS, "stale": self.stale,
            "seq_gaps": self.gaps, "seq_regress": self.regress,
            "src": self.src, "src_changes": self.src_changes,
        }


class Stage8TwelveAxisMitControlCore:
    def __init__(self, now_fn=time.time):
        self.now_fn = now_fn
        self.armed = False
        self.baseline_deg_by_joint = {}
        self.baseline_set_time = None
        self.baseline_source = None                 # "takeover_hold" | "actual" (0918)
        self.baseline_hold_minus_actual_deg = {}    # 홀드 명령 − 실측 [deg] (0918 진단)
        self.warnings = []
        self.errors = []
        self.last_takeover_event = None   # 인수홀드 수명 이벤트 (명령 처리로 소거 안 됨)
        #: 스트리밍 이력 래치 — 한 번이라도 홀드/armed 명령 프레임을 보냈으면
        #: 이후 ENTER 폴링을 영구 금지 (0802 실측: ENTER=홀드 리셋 → 하중 하
        #: 능동 릴리즈). 이 상태에서 재무장하려면 노드 재기동(=명시 프로토콜).
        self._ever_streamed = False
        self.takeover_hold = self._load_takeover_state()
        self._takeover_start = None
        self._waiting_takeover = bool(WAIT_TAKEOVER) and self.takeover_hold is None
        self._wait_warn_time = None
        # 기구 고정 관절의 실측 = stand_zero_deg 합성(모터 프레임). 나이·신선도는 _joint_feedback_age_sec 가 합성한다.
        self.latest_actual_deg_by_joint = {
            joint: float(STAND_ZERO_DEG_BY_JOINT[joint]) for joint in MECH_FIXED_JOINTS}
        self.mech_fixed_target_drops = 0   # 고정 관절로 들어와 버린 목표 수 (status 노출)
        self.latest_feedback_by_joint = {}
        self.torque_cap_counts = {}        # 1010 요구 토크 상한이 목표를 묶은 송신 수(관절별, status 노출)
        self.latest_feedback_time_by_joint = {}
        self.latest_feedback_time = None
        self.desired_relative_deg_by_joint = {joint: 0.0 for joint in JOINT_NAMES_12DOF}
        self.commanded_relative_deg_by_joint = {joint: 0.0 for joint in JOINT_NAMES_12DOF}
        self.last_preview_by_joint = {}
        self.command_seq = 0
        self.cmd_timing = CommandTiming()   # 0930: 목표 명령 지연·누락 기록 (동작 불변)
        self.last_command = None
        self.last_reject_reason = None
        self.soft_clamp_counts = {}   # 관절별 soft 경계 클램프 누적 (status 노출)
        self.feedback_jump_drops = {}  # 관절별 점프 필터 드롭 (연속 카운트)
        self.can_bus = None
        self.can_bus_open_attempted = False
        self.can_status = "UNKNOWN"
        self.can_link_state = "UNKNOWN"
        self.can_bitrate = None
        self.can_last_error = None
        self.can_last_stdout = ""
        self.can_last_stderr = ""
        self.can_rx_count = 0
        self.can_tx_readonly_count = 0
        self.can_rx_ignored_count = 0
        #: bus.send 가 타임아웃 예산 내 송신 실패(TX 큐 대기/버퍼 포화)로
        #: 낸 CanError 누적 — H-D1 버스 정체 판별용 (status 노출)
        self.can_send_timeouts = 0
        self.can_decode_error = None
        self.feedback_stale_reason = "no_feedback"
        self.baseline_missing_joints = list(JOINT_NAMES_12DOF)
        self._next_readonly_request_index = 0
        self._last_readonly_request_time = 0.0
        self._readonly_request_backoff_until = 0.0
        self.accepted = False
        self.executed = False
        self.running = False
        #: 1006 지연 옵션 계측 — 마지막 armed 송신 시각(monotonic)·송신 경로별 횟수·송신 후 피드백 대기.
        self.last_armed_send_mono = None
        self.last_command_send_mono = None
        self._tick_sent_ids = []          # 이번 틱에 명령 프레임을 보낸 모터 ID (송신 후 피드백·즉시 status 발행 판정)
        self._tick_source = None
        self.send_counts = {"timer": 0, "command": 0, "watchdog": 0, "hold": 0}
        self.fb_after_send = {"ticks": 0, "complete": 0, "replies": 0,
                              "last_wait_ms": None, "max_wait_ms": 0.0}

        valid, reason = validate_12dof_mapping()
        self.mapping_valid = bool(valid)
        if reason:
            self.errors.append(f"mapping_invalid:{reason}")

    def handle_joint_states_data(self, message_data):
        payload, error = parse_json_object(message_data)
        if error:
            self._set_error(error)
            return False

        actuals = self._extract_actuals(payload)
        if not actuals:
            self._set_error("joint_actual_missing")
            return False

        self.latest_actual_deg_by_joint.update(actuals)
        self.latest_feedback_time = self.now_fn()
        self._clear_errors()
        return True

    def handle_command_data(self, message_data):
        payload, error = parse_json_object(message_data)
        if error:
            self._reject(error)
            return self.get_status()
        return self.handle_command(payload)

    def handle_command(self, payload):
        command = str(payload.get("command", "")).strip().upper()
        self.command_seq += 1
        self.last_command = command
        self.last_reject_reason = None
        self.warnings = []
        self.accepted = False
        self.executed = False

        if command == COMMAND_ARM_ALL:
            # 재무장 = 신선한 베이스라인 강제 (적대리뷰 B2: 종전엔 DISARM으로
            # 가드를 피해 살아남은 옛 베이스라인에 ARM만 다시 걸면 그대로 발사)
            if self.baseline_set_time is None or (
                    self.now_fn() - self.baseline_set_time > 30.0):
                return self._reject(
                    "baseline_stale_for_arm:SET_BASELINE_FROM_CURRENT_ALL 먼저")
            # 최종점검 S6: 부분 베이스라인 ARM 금지 — 빠진 관절은 명령이 조용히
            # 건너뛰어져 일부만 움직이는 비대칭 발사가 됨. 12/12 전수 요구.
            if not self._baseline_is_set():
                return self._reject(
                    "baseline_incomplete_for_arm:"
                    + ",".join(self._baseline_missing_joints()))
            self.armed = True
            if self.takeover_hold is not None:
                self.last_takeover_event = "armed_handover"
            self.takeover_hold = None   # 인수 홀드 종료 — armed 경로가 이어받음
            # 슬루 기준점을 '현재 실측'으로 초기화 (적대 시뮬 검증: 종전 0 고정은
            # 캡처 후 자세가 변했으면 첫 명령이 베이스라인 자세로 점프 —
            # kp150×6° ≈ 15.7Nm급 순간 인장. 러너의 qpos 초기화와 대칭)
            # 0918: 기준이 인수 홀드 명령각이면 슬루 기준점 = 0 (명령 연속). 홀드가
            # 직전 틱까지 정확히 그 명령을 스트리밍했으므로 점프가 없고, 실측(처진
            # 값)으로 초기화하면 첫 명령이 처진 각도로 떨어졌다 되돌아오는 침하 딥이 생긴다.
            for joint in self.baseline_deg_by_joint:
                if self.baseline_source == "takeover_hold":
                    self.commanded_relative_deg_by_joint[joint] = 0.0
                    continue
                actual = self.latest_actual_deg_by_joint.get(joint)
                if actual is not None:
                    rel = float(actual) - self.baseline_deg_by_joint[joint]
                    self.commanded_relative_deg_by_joint[joint] = rel
            self.accepted = True
            return self.get_status()

        if command == COMMAND_DISARM_ALL:
            self.armed = False
            self.running = False
            if self.takeover_hold is not None:
                self.last_takeover_event = "disarm_cancel"
            self.takeover_hold = None   # 운영자 의도 = 릴리즈 경로 (release_all)
            # 주의: 스트리밍 이력이 있으면 ENTER 폴링은 재개되지 않는다
            # (_ever_streamed — 폴링=능동 릴리즈). 모터는 마지막 명령을 물고
            # 있으므로 이후 절차 = stack_down → release_all.
            self.accepted = True
            return self.get_status()

        if command == COMMAND_SET_BASELINE_ALL:
            return self._set_baseline_from_current_all()

        if command == COMMAND_STOP_ALL:
            for joint in JOINT_NAMES_12DOF:
                self.desired_relative_deg_by_joint[joint] = self.commanded_relative_deg_by_joint[joint]
            self.running = False
            self.accepted = True
            return self.get_status()

        if command in (COMMAND_CHECK_CAN, COMMAND_RECONNECT_CAN):
            # 최종점검 S4: CHECK_CAN은 전모터 ENTER 일제사(=홀드 리셋),
            # RECONNECT_CAN은 ip link down/up(=버스 절단) — 대기/홀드/무장/
            # 스트리밍 이력 상태에서는 침묵·홀드 계약을 파괴하므로 거부.
            if (self._waiting_takeover or self.takeover_hold is not None
                    or self.armed or self._ever_streamed):
                return self._reject(
                    f"bus_owner_active:{command} 금지 (대기/홀드/무장 중 — "
                    "진단은 stack_down 후)")
            if command == COMMAND_CHECK_CAN:
                return self._check_can_status()
            return self._reconnect_can()

        if command == COMMAND_SET_JOINT_TARGET:
            return self._set_joint_target(payload)

        if command == COMMAND_SET_JOINT_TARGETS:
            self.cmd_timing.update(self.now_fn(), payload)
            return self._set_joint_targets_batch(payload)

        return self._reject("unsupported_command")

    def _load_takeover_state(self):
        """hang_stand 핸드오프 상태 파일 로드 (1회 소비). {joint: (deg, kp, kd)}."""
        try:
            with open(TAKEOVER_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        # 소비 표시는 읽기와 분리 (최종점검 S9): 삭제 실패(권한 등)해도 로드는
        # 진행하되 흔적을 남긴다 — ts 게이트(120s)가 낡은 재소비를 막는다.
        try:
            os.remove(TAKEOVER_STATE_FILE)
        except OSError:
            self.last_takeover_event = "state_file_remove_failed"
        try:
            if self.now_fn() - float(data["ts"]) > TAKEOVER_STATE_MAX_AGE_SEC:
                self.last_takeover_event = "state_file_too_old"
                return None
            hold = {}
            for mid_s, v in data["motors"].items():
                joint = JOINT_BY_CAN_ID[int(mid_s)]
                if joint in MECH_FIXED_JOINTS:
                    continue   # 기구 고정 — 홀드 송신 대상이 아니다(파일에 있어도 쓰지 않는다)
                deg = float(v["deg"])
                kp = min(500.0, max(0.0, float(v["kp"])))
                kd = min(5.0, max(0.0, float(v["kd"])))
                if not math.isfinite(deg) or abs(deg) > 180.0:
                    self.last_takeover_event = f"state_file_deg_invalid:{joint}"
                    return None
                # 최종점검 S2: 홀드 각도에도 리밋 표를 적용 — 파손·위조 파일이
                # 기구 한계 밖 홀드를 첫 틱부터 스트리밍하는 경로 차단.
                # deg=모터 절대각 → 관절 프레임 변환 후 soft 검사 + 모터 백스톱.
                joint_deg = DIRECTION_BY_JOINT[joint] * (
                    deg - STAND_ZERO_DEG_BY_JOINT[joint])
                if check_absolute_deg(joint, joint_deg) != "ok":
                    self.last_takeover_event = (
                        f"state_file_limit:{joint}:joint{joint_deg:.1f}deg")
                    return None
                dev = abs(deg - STAND_ZERO_DEG_BY_JOINT[joint])
                if dev > MOTOR_DEV_BACKSTOP_DEG_BY_JOINT[joint]:
                    self.last_takeover_event = (
                        f"state_file_backstop:{joint}:{dev:.1f}deg")
                    return None
                hold[joint] = (deg, kp, kd)
            # 기구 고정 관절을 뺀 전부가 있어야 수락 (선언 없으면 종전대로 12/12)
            need = len(JOINT_NAMES_12DOF) - len(MECH_FIXED_JOINTS)
            if len(hold) != need:
                self.last_takeover_event = f"state_file_incomplete:{len(hold)}/{need}"
                return None
            return hold
        except (KeyError, TypeError, ValueError):
            self.last_takeover_event = "state_file_parse_error"
            return None

    def _takeover_hold_tick(self):
        """인수 홀드 스트리밍 — 비무장 구간에서 hang_stand의 마지막 명령을 유지.

        명령 프레임이 피드백을 반환하므로 별도 ENTER 폴링 불필요 (ENTER는 홀드
        명령을 리셋하는 실측 거동이 있어 금지). ARM 시 정상 경로로 전환.
        """
        if not REAL_CAN_WRITE_ENABLED or self.takeover_hold is None:
            return []
        now = self.now_fn()
        if self._takeover_start is None:
            self._takeover_start = now
        elif (now - self._takeover_start > 300.0
              and self.last_takeover_event != "hold_age_over_300s"):
            # 적대검수 0803: 종전 '300s 타임아웃 캔슬'은 하중 지지 중 능동 침묵
            # = 모터 타임아웃(~200-300ms 브레이크) 낙하 확정 경로였다 (캔슬
            # 근거였던 '모터가 마지막 명령을 물고 있음'은 0802 실측으로 폐기).
            # 홀드는 시간과 무관하게 유지 — 경고만 남긴다. 좀비 홀드(모터
            # 전원사이클)는 아래 피드백 두절 가드가 담당.
            self.last_takeover_event = "hold_age_over_300s"
            self._warn("takeover_hold_age_over_300s")
        hold_age = now - self._takeover_start
        # 최종점검 S3: 홀드 중 피드백 가드 — 명령 프레임이 피드백을 겸하므로
        # 정상 홀드에선 매 틱 갱신된다. 두절(worst age > STALE_DISARM_SEC,
        # 기동 유예 5s 후 무피드백 포함) = 모터 전원 오프/버스 사망 신호 →
        # 홀드 중단 (전원 복귀 시 낡은 명령이 요격하는 사고 방지).
        worst = None
        mismatch = None
        worst_dev = 0.0
        worst_dev_joint = None
        for joint in self.takeover_hold:
            age = self._joint_feedback_age_sec(joint)
            if age is None:
                worst = float("inf") if hold_age > 5.0 else worst
                continue
            worst = age if worst is None else max(worst, age)
            if self._joint_feedback_is_fresh(joint):
                actual = self.latest_actual_deg_by_joint.get(joint)
                deg = self.takeover_hold[joint][0]
                if actual is not None:
                    dev = float(actual) - deg
                    if abs(dev) > abs(worst_dev):
                        worst_dev = dev
                        worst_dev_joint = joint
                    if abs(dev) > 20.0:
                        mismatch = f"{joint}:{dev:+.1f}deg"
        if worst is not None and worst > STALE_DISARM_SEC:
            self.takeover_hold = None
            self.last_takeover_event = "hold_feedback_gap_cancel"
            self._warn("takeover_hold_feedback_gap_cancel")
            return []
        # 실측-홀드 괴리 = 경고만, 캔슬 금지 (0802 밤 붕괴 2건 교훈).
        # 근거: ①상태 파일은 기록 후 수십 ms 내 소비돼 '낡은 파일 대괴리'가
        # 실경로에 없음(절대각·백스톱은 로더가 검증) ②캔슬→침묵→모터 자체
        # 타임아웃(~200-300ms 실측, 브레이크 모드)→하중 낙하로 사고를 '키움'
        # (candump 실측: 인수 공백 303ms 낙하를 kp150이 급제동한 것이 '발작').
        # 어떤 괴리에서도 버티는 것이 정답 — 회수는 운용자(줄 인장→stack_down
        # →release_all)의 몫.
        if mismatch is not None:
            if self.last_takeover_event != f"hold_disturbance:{mismatch}":
                self.last_takeover_event = f"hold_disturbance:{mismatch}"
                self._warn("takeover_hold_disturbance")
                print(f"[takeover] 괴리 감지 — 홀드 유지하며 버팀: {mismatch}",
                      flush=True)
        # 관측성: 매초 최대 괴리 로그 (다음 사고 분석용 — stage8 블랙박스 부재
        # 보완)
        if now - getattr(self, "_hold_log_t", 0.0) >= 1.0:
            self._hold_log_t = now
            print(f"[takeover] hold {hold_age:.0f}s 최대괴리 "
                  f"{worst_dev:+.2f}° ({worst_dev_joint if worst_dev else '-'})",
                  flush=True)
        self._ever_streamed = True
        sent_ids = []
        for joint, (deg, kp, kd) in self.takeover_hold.items():
            frame = pack_ak_mit_command(
                math.radians(deg), v_des_rad_s=V_DES_DEFAULT,
                kp=kp, kd=kd, tau_ff=TAU_FF_DEFAULT)
            if self._maybe_write_frame(CAN_ID_BY_JOINT[joint], frame):
                sent_ids.append(CAN_ID_BY_JOINT[joint])
        return sent_ids

    def _tick_timing_status(self):
        """1006: 최근 ≤ 250 틱 간격 통계(ms) — 평균·p50·p99·최대·20 ms ±5 ms 밖 비율. 계측 전용."""
        buf = getattr(self, "_tick_dt_ms", None)
        if not buf:
            return {"n": 0}
        xs = sorted(buf); n = len(xs)
        q = lambda f: round(xs[min(n - 1, int(f * (n - 1)))], 3)
        return {"n": n, "mean_ms": round(sum(xs) / n, 3), "p50_ms": q(0.5), "p99_ms": q(0.99), "max_ms": round(xs[-1], 3),
                "out_of_band_frac": round(sum(1 for x in xs if abs(x - CONTROL_PERIOD_SEC * 1000.0) > 5.0) / n, 4)}

    def control_tick(self, source="timer"):
        """제어 틱. source = "timer"(종전 주기 타이머) | "command"(--send-on-command 즉시 송신) |
        "watchdog"(즉시 송신 모드에서 명령이 끊겼을 때의 감시 송신). 동작은 source 와 무관하게 같고
        계측(send_counts)만 다르다."""
        self.executed = False
        self._tick_sent_ids = []
        self._tick_source = source
        # 1006 계측(읽기 전용, 동작 무관): 틱 간격 링버퍼 → status.tick_timing(지터 판정용)
        _now = time.monotonic()
        _lt = getattr(self, "_tick_last_mono", None)
        if _lt is not None:
            _buf = getattr(self, "_tick_dt_ms", None)
            if _buf is None:
                _buf = self._tick_dt_ms = collections.deque(maxlen=250)
            _buf.append((_now - _lt) * 1000.0)
        self._tick_last_mono = _now
        # 선기동 대기: 상태 파일이 나타날 때까지 CAN 완전 침묵 (송신 0 —
        # hang_stand가 아직 버스를 소유·홀드 중이므로 ENTER 폴링도 금지)
        if self._waiting_takeover:
            # 0802 붕괴#2 candump 실측: 인수 공백 303ms 중 모터 자체 명령
            # 타임아웃(~200-300ms)이 걸려 하중 낙하 → kp150 급제동 '발작'.
            # 공백의 대부분은 첫 송신 시점의 python-can 임포트+버스 오픈 비용
            # — 대기 중에 미리 열어둔다 (소켓 바인드는 수동적, 송신 0 유지).
            if self.can_bus is None:
                try:
                    import can as _can
                    self._get_or_open_can_bus(_can)
                except Exception:
                    pass
            self.takeover_hold = self._load_takeover_state()
            if self.takeover_hold is not None:
                # 적대검수 0803: 대기 중 선오픈 소켓에는 hang_stand 스트리밍
                # (~1200프레임/s, 수 분)이 쌓여 커널 RX 큐가 포화된다 — 인수
                # 첫 틱부터 낡은 프레임 수천 개가 '신선한 피드백'으로 소비되는
                # 것을 차단하기 위해 백로그를 전량 폐기하고 시작한다.
                drained = 0
                if self.can_bus is not None:
                    try:
                        while self.can_bus.recv(timeout=0.0) is not None:
                            drained += 1
                            if drained > 20000:
                                break
                    except Exception:  # noqa: BLE001 — 드레인 실패는 치명 아님
                        pass
                if drained:
                    print(f"[takeover] 인수 전 RX 백로그 {drained}프레임 폐기",
                          flush=True)
            if self.takeover_hold is None:
                now = self.now_fn()
                # 핸드셰이크 마커 — hang_stand 는 이 파일이 신선할 때만 핸드
                # 오프를 발화한다 (수신자 없는 핸드오프 = 상태 파일만 남고
                # 모터 자체 타임아웃 릴리즈). 적대검수 0803: 버스 선오픈이
                # '성공한 상태'에서만 touch — 송신 불능(채널 오설정·미개통)인
                # 노드가 수신자 행세하는 것 차단.
                if self.can_bus is not None:
                    try:
                        with open("/tmp/stage8_takeover_waiting", "w") as f:
                            f.write(str(now))
                    except OSError:
                        pass
                if self._wait_warn_time is None:
                    self._wait_warn_time = now
                elif now - self._wait_warn_time > 30.0:
                    self._wait_warn_time = now
                    self._warn("wait_takeover_still_waiting")
                return self.get_status()
            self._waiting_takeover = False
        self._ensure_readonly_can_bus()
        # ENTER 폴링은 '홀드 중인 모터를 리셋'하므로 (0802 실측) 인수 홀드
        # 또는 armed 송신 중에는 금지 — 두 경로 모두 명령 프레임이 피드백을 겸함.
        # 최종점검 S1: 스트리밍 이력이 있으면(홀드 해제·DISARM 후 포함) 폴링을
        # 영구 금지 — 종전엔 홀드 이탈 즉시 폴링 재개 = 하중 하 능동 릴리즈였음.
        if self.takeover_hold is not None:
            self._tick_sent_ids = self._takeover_hold_tick() or []
            if self._tick_sent_ids:   # 1006 계측(읽기 전용): 인수 홀드 송신 틱 수 — latency_check 송신 주기 판정용
                self.send_counts["hold"] = self.send_counts.get("hold", 0) + 1
        elif not self.armed and not self._ever_streamed:
            self._request_next_readonly_feedback()
        self._poll_can_feedback()
        if not self.armed or not self._has_any_baseline():
            # 인수 홀드도 명령 프레임 응답이 피드백 — 옵션이 켜졌으면 같은 틱 안에서 읽는다.
            self._await_feedback_after_send(self._tick_sent_ids)
            return self.get_status()

        # 피드백 공백 무장해제 가드 — 장기 두절(모터 전원 오프·분기 사망) 후
        # 복귀 시 낡은(프레임 시프트 가능) 베이스라인으로 명령이 나가는 것을
        # 원천 차단. 판정은 '관절별 최악 나이' 기준 (적대리뷰 B1: 전역 신선도는
        # 모터 1개만 살아 있어도 갱신되어 11개 두절을 영영 못 잡음).
        if REAL_CAN_WRITE_ENABLED:
            worst_age = None
            for joint in self.baseline_deg_by_joint:
                age = self._joint_feedback_age_sec(joint)
                if age is None:
                    worst_age = float("inf")
                    break
                worst_age = age if worst_age is None else max(worst_age, age)
            if worst_age is not None and worst_age > STALE_DISARM_SEC:
                self.armed = False
                self.running = False
                self.baseline_deg_by_joint = {}
                self.baseline_set_time = None
                self._warn("feedback_gap_auto_disarm")
                return self.get_status()

        # 적대검수 0803: 종전 '전역 stale 3초 → 송신 전면 중단'은 하중 지지 중
        # 침묵 = 모터 타임아웃(~200-300ms 브레이크) 낙하 확정 + 자기지속 래치
        # (armed 중 피드백은 명령 응답이 유일한 소스라 무송신→무응답→영영
        # stale). 인수홀드 경로와 동일한 시맨틱으로 통일: stale이면 램프를
        # '동결'(desired 진행 금지)하되 마지막 commanded 목표는 계속 스트리밍
        # — 모터 생존 유지 + 응답 복구 경로 확보. 완전 이탈은 위의 60s
        # 최악나이 무장해제가 담당.
        global_stale = REAL_CAN_WRITE_ENABLED and not self._feedback_is_fresh()
        if global_stale:
            self._warn("feedback_stale_freeze_hold")

        self.running = True
        moved = False
        self.last_preview_by_joint = {}
        if REAL_CAN_WRITE_ENABLED:
            self._ever_streamed = True   # armed 송신 개시 — 이후 ENTER 폴링 영구 금지

        if not self._feedback_is_fresh() and not REAL_CAN_WRITE_ENABLED:
            self._warn("feedback_stale_dry_run")

        # 슬루 상한: 기본(옵션 꺼짐) = 종전 그대로 틱당 4°. 옵션 모드 = 시간 기준 200°/s × 경과,
        # 틱당 최대 4° — 송신이 잦아져도(명령 즉시 송신·100 Hz) 시간당 슬루는 종전 이하로 유지.
        now_mono = time.monotonic()
        step_limit = MAX_TARGET_STEP_DEG_PER_TICK
        if time_based_slew_enabled():
            dt = (CONTROL_PERIOD_SEC if self.last_armed_send_mono is None
                  else max(0.0, now_mono - self.last_armed_send_mono))
            step_limit = min(MAX_TARGET_STEP_DEG_PER_TICK, MAX_TARGET_SLEW_DEG_PER_SEC * dt)
        self.last_armed_send_mono = now_mono
        if source == "command":
            self.last_command_send_mono = now_mono
        self.send_counts[source] = self.send_counts.get(source, 0) + 1

        for joint in JOINT_NAMES_12DOF:
            if joint not in self.baseline_deg_by_joint:
                continue
            if joint in MECH_FIXED_JOINTS:
                continue   # 기구 고정 — 모터 없음, 송신 생략(상대 명령은 0 그대로)
            joint_stale = (REAL_CAN_WRITE_ENABLED
                           and not self._joint_feedback_is_fresh(joint))
            if joint_stale:
                self._warn(f"joint_feedback_stale:{joint}")
            current = self.commanded_relative_deg_by_joint[joint]
            if global_stale or joint_stale:
                next_value = current   # 동결 — 진행 없이 현 목표 재송신
            else:
                desired = self.desired_relative_deg_by_joint[joint]
                next_value = self._step_toward(current, desired, step_limit)
            if next_value != current:
                moved = True
            self.commanded_relative_deg_by_joint[joint] = next_value

            absolute_target_deg = self.baseline_deg_by_joint[joint] + next_value
            if torque_cap_enabled() and not (global_stale or joint_stale):
                # 1010 요구 토크 상한: 보내는 목표만 실측 ± 허용 오차로 묶는다(내부 commanded 는 그대로 — 위 머리말)
                err_lim = torque_cap_err_deg(joint)
                meas = self.latest_actual_deg_by_joint.get(joint)
                if err_lim is not None and meas is not None:
                    capped = min(max(absolute_target_deg, float(meas) - err_lim), float(meas) + err_lim)
                    if capped != absolute_target_deg:
                        self.torque_cap_counts[joint] = self.torque_cap_counts.get(joint, 0) + 1
                        absolute_target_deg = capped
            preview = build_12axis_mit_preview(joint, absolute_target_deg)
            self.last_preview_by_joint[joint] = preview
            sent = self._maybe_write_frame(preview.motor_id, preview.frame_bytes)
            if sent:
                self._tick_sent_ids.append(preview.motor_id)
            self.executed = sent or self.executed

        self._await_feedback_after_send(self._tick_sent_ids)
        if moved and self.last_reject_reason is None:
            self.accepted = True
        return self.get_status()

    def _await_feedback_after_send(self, sent_ids):
        """--feedback-after-send: 방금 보낸 명령 프레임의 응답을 같은 틱 안에서 읽는다.

        보낸 모터가 모두 응답하거나 FEEDBACK_WAIT_SEC 가 지나면 멈춘다(상한 블로킹 = 대기 시간).
        응답 해석은 송신 전 폴링과 같은 _handle_can_feedback_message(형상 검증·점프 필터 동일).
        제때 못 온 응답은 종전처럼 다음 틱의 송신 전 폴링이 읽는다. 옵션 꺼짐이면 아무것도 안 한다."""
        if (not FEEDBACK_AFTER_SEND or not sent_ids or self.can_bus is None
                or not READ_ONLY_CAN_ENABLED):
            return 0
        pending = set(int(m) for m in sent_ids)
        t0 = time.monotonic()
        deadline = t0 + FEEDBACK_WAIT_SEC
        replies = 0
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                message = self.can_bus.recv(timeout=remaining)
            except Exception as error:
                self.can_status = "ERROR"
                self.can_last_error = str(error)
                self._set_error(f"can_recv_failed:{error}")
                break
            if message is None:
                break
            if self._handle_can_feedback_message(message):
                replies += 1
                pending.discard(int(message.arbitration_id))
        waited_ms = (time.monotonic() - t0) * 1000.0
        st = self.fb_after_send
        st["ticks"] += 1
        st["replies"] += replies
        st["complete"] += 0 if pending else 1
        st["last_wait_ms"] = round(waited_ms, 2)
        st["max_wait_ms"] = round(max(st["max_wait_ms"], waited_ms), 2)
        return replies

    def wants_background_send(self, now_mono=None):
        """--send-on-command 모드의 armed 배경(감시) 송신 판정 — 5 ms 타이머가 묻는다.

        명령 스트림이 살아 있으면(마지막 명령 송신 < 25 ms) 50 Hz 모드에선 송신하지 않는다(명령이 맡음).
        100 Hz 모드에선 명령 사이 중간 재송신(피드백 10 ms 갱신). 명령이 끊기면 종전 주기로 계속 송신 —
        모터 명령 공백 상한 ≈ 25 ms + 5 ms(종전 20 ms). 옵션 꺼짐·비무장이면 False(종전 타이머 경로)."""
        if not SEND_ON_COMMAND or not self.armed:
            return False
        now_mono = time.monotonic() if now_mono is None else now_mono
        since_any = (float("inf") if self.last_armed_send_mono is None
                     else now_mono - self.last_armed_send_mono)
        since_cmd = (float("inf") if self.last_command_send_mono is None
                     else now_mono - self.last_command_send_mono)
        due = since_any >= CONTROL_PERIOD_SEC - BACKGROUND_TOL_SEC
        if since_cmd < COMMAND_STREAM_ACTIVE_SEC:
            return due and CONTROL_PERIOD_SEC < 0.02 - 1e-9
        return due

    def should_send_on_command(self, now_mono=None):
        """방금 처리한 명령으로 즉시 송신할지 — 옵션 켜짐·armed·배치 목표 수용·최소 간격 경과."""
        if not SEND_ON_COMMAND or not self.armed or not self.accepted:
            return False
        if self.last_command != COMMAND_SET_JOINT_TARGETS:
            return False
        if self.last_command_send_mono is None:
            return True
        now_mono = time.monotonic() if now_mono is None else now_mono
        # 명령 즉시 송신끼리만 최소 간격 — DDS 적체가 한꺼번에 풀린 명령 묶음이 송신 폭주가 되는 것 차단.
        # 그 사이 명령은 목표만 갱신되고(최신값이 이김) 다음 송신에 실린다.
        return now_mono - self.last_command_send_mono >= SEND_ON_COMMAND_MIN_INTERVAL_SEC

    def _set_baseline_from_current_all(self):
        available = [
            joint for joint in JOINT_NAMES_12DOF
            if (
                joint in self.latest_actual_deg_by_joint
                and self._joint_feedback_is_fresh(joint)
            )
        ]
        self.baseline_missing_joints = [
            joint for joint in JOINT_NAMES_12DOF
            if joint not in available
        ]
        # 기구 고정 관절은 합성 피드백으로 항상 available 이다 — '실측 전무' 거부는 살아 있는 관절로만 판정
        # (선언 없으면 종전과 같은 식). 종전 거부가 고정 관절 때문에 통과로 바뀌지 않게 한다.
        if not [joint for joint in available if joint not in MECH_FIXED_JOINTS]:
            if not self._feedback_is_fresh():
                self.feedback_stale_reason = self._feedback_stale_reason()
                return self._reject("feedback_stale:all_baseline_missing")
            return self._reject("baseline_missing_actual:all")

        # 0918 핸드오프 침하 수정: 인수 홀드 중이면 기준 = 홀드 명령각(hang_stand 의
        # 마지막 명령). 실측으로 잡으면 하중 관절의 PD 추종오차만큼 토크가 풀려 ARM
        # 직후 한 번 더 처진다 — 0918 실기: 무릎 −30.4→−35.6°·발목 1.3° 침하 →
        # 무게중심 후방 이동 → 뒤꿈치 강체 회전 낙상(정책 미개입). 실측 신선도
        # 게이트는 그대로 두고(무피드백 ARM 금지), 값만 명령각을 쓴다.
        hold = self.takeover_hold
        self.baseline_source = "takeover_hold" if hold else "actual"
        self.baseline_deg_by_joint = {
            joint: (float(hold[joint][0]) if hold and joint in hold
                    else float(self.latest_actual_deg_by_joint[joint]))
            for joint in available
        }
        # 진단: 홀드 명령 − 실측 (= 하중 처짐, 사후 분석용)
        self.baseline_hold_minus_actual_deg = (
            {joint: round(float(hold[joint][0])
                          - float(self.latest_actual_deg_by_joint[joint]), 3)
             for joint in available if joint in hold}
            if hold else {})
        self.baseline_set_time = self.now_fn()
        for joint in JOINT_NAMES_12DOF:
            self.desired_relative_deg_by_joint[joint] = 0.0
            self.commanded_relative_deg_by_joint[joint] = 0.0
        self.accepted = True
        return self.get_status()

    def _set_joint_target(self, payload):
        if not self.armed:
            return self._reject("not_armed")
        if "joints" in payload or "joints_deg" in payload:
            return self._reject("multi_joint_payload_rejected")

        joint = str(payload.get("joint", "")).strip()
        why = self._validate_and_stage_target(joint, payload.get("target_deg"))
        if why is not None:
            return self._reject(why)
        self.accepted = True
        return self.get_status()

    def _set_joint_targets_batch(self, payload):
        """배치 타깃 (지연 수술 0803): 브리지가 틱당 12개 메시지 대신 1개.

        종전 12msg/틱 = 600msg/s의 DDS·콜백·GIL 부하가 젯슨 제어 루프 지터
        (=실측 루프 지연의 소프트웨어 몫)에 기여. 검증은 단건 경로와 완전
        동일(_validate_and_stage_target 공용). 일부 관절 거부 시 해당 관절만
        미갱신하고 사유를 모아 보고 — 수용된 관절은 정상 진행 (관절별 독립
        스트리밍 시맨틱 유지).
        """
        if not self.armed:
            return self._reject("not_armed")
        targets = payload.get("targets")
        if not isinstance(targets, dict) or not targets:
            return self._reject("targets_invalid")
        rejected = []
        for joint, deg in targets.items():
            why = self._validate_and_stage_target(str(joint).strip(), deg)
            if why is not None:
                rejected.append(why)
        if rejected:
            self.last_reject_reason = "batch_partial:" + ";".join(rejected[:4])
        self.accepted = len(rejected) < len(targets)
        return self.get_status()

    def _validate_and_stage_target(self, joint, raw_target):
        """단건 타깃 검증+스테이징 — 성공 시 None, 실패 시 거부 사유 문자열."""
        if joint not in JOINT_NAMES_12DOF:
            return "unknown_joint"
        if joint not in ALLOWED_JOINTS:
            return f"joint_not_allowed:{joint}"
        if joint not in self.baseline_deg_by_joint:
            self.baseline_missing_joints = self._baseline_missing_joints()
            return f"baseline_missing:{joint}"
        if REAL_CAN_WRITE_ENABLED and not self._joint_feedback_is_fresh(joint):
            return f"joint_feedback_stale:{joint}"

        try:
            target_deg = float(raw_target)
        except (TypeError, ValueError):
            return "target_invalid"
        if not math.isfinite(target_deg):
            # 적대검수 0803: NaN은 모든 리밋 비교(<=, abs)를 False로 통과해
            # 클램프·백스톱 전 계층을 무력화 — 파싱 직후 차단.
            return "target_not_finite"

        config = get_joint_config(joint)
        if abs(target_deg) > float(config["hard_limit_deg"]):
            return "hard_limit_exceeded"
        if abs(target_deg) > float(config["target_limit_deg"]):
            return "target_limited"

        # 절대각 검사 (2026-07-26 신설, 2026-07-27 프레임 수정): 종전 상대각 ±180°
        # 검사는 사실상 무제한 — 절대각(baseline+상대) 기준 비대칭 소프트/하드
        # 리밋을 추가 적용한다. 원본 config/joint_limits_12dof.json (RL 실측
        # 엔벨로프 + 해부학·자기충돌 한계).
        # 프레임 버그 수정: baseline·target 은 '모터 인코더 프레임' deg 인데
        # 리밋 표는 'URDF 관절 프레임' — joint = DIR×(motor−SZ) 변환 후 검사한다
        # (종전엔 모터 절대각을 그대로 넣어 리밋이 엉뚱한 프레임에 적용됐음).
        absolute_deg = self.baseline_deg_by_joint[joint] + target_deg  # 모터 프레임
        joint_abs_deg = DIRECTION_BY_JOINT[joint] * (
            absolute_deg - STAND_ZERO_DEG_BY_JOINT[joint])             # 관절 프레임
        limit_verdict = check_absolute_deg(joint, joint_abs_deg)
        if limit_verdict != "ok":
            # 최종점검(0802): soft 초과는 거부가 아니라 '경계 클램프'로 전환.
            # 근거 — march_v3 는 시뮬에서 무릎을 과신전 스톱에 눌러 기대는
            # 니락 전략을 학습(스탠스 3.4% 틱, 타깃은 스톱 너머 최대 +20°).
            # 거부하면 무릎 명령의 상당수가 조용히 무시돼 시뮬과 다른 폐루프,
            # 통과시키면 실물 기구 스톱을 kp150 으로 압박. 시뮬도 스톱에서
            # 물리적으로 포화됐으므로 soft 경계(스톱 1° 안쪽) 클램프가 시뮬
            # 물리와 정합. 클램프는 status 카운터로 노출 — 조용한 개입 금지.
            # 단 경계 초과 25° 이상은 폭주 신호로 보고 거부한다.
            # (러너의 구 클램프 사고와 다른 클래스: 그건 엔벨로프 한복판 왜곡,
            # 이건 물리 스톱 위치의 포화 재현. soft 표는 배포 정책 실측
            # 엔벨로프 ∪ 구 게이트 표로 재생성돼 정상 보행에선 발동하지 않는다.)
            lim = get_absolute_limits_deg()[joint]
            clamped = min(max(joint_abs_deg, lim["soft_min"]), lim["soft_max"])
            excess = abs(joint_abs_deg - clamped)
            if excess > 25.0:
                return (f"limit_runaway:{joint}:joint{joint_abs_deg:.1f}deg"
                        f"(경계+{excess:.1f}deg, motor{absolute_deg:.1f}deg)")
            self.soft_clamp_counts[joint] = self.soft_clamp_counts.get(joint, 0) + 1
            absolute_deg = (STAND_ZERO_DEG_BY_JOINT[joint]
                            + DIRECTION_BY_JOINT[joint] * clamped)     # 모터 프레임 역변환
            target_deg = absolute_deg - self.baseline_deg_by_joint[joint]
            joint_abs_deg = clamped

        # 모터 프레임 백스톱: stand_zero 로부터의 편차가 하드리밋 폭+여유를
        # 넘으면 거부 — DIR 부호와 무관한 2차 방어선 (프레임 근거는 상수 정의부).
        dev = abs(absolute_deg - STAND_ZERO_DEG_BY_JOINT[joint])
        dev_limit = MOTOR_DEV_BACKSTOP_DEG_BY_JOINT[joint]
        if dev > dev_limit:
            return (f"motor_dev_backstop:{joint}:|motor{absolute_deg:.1f}deg-"
                    f"SZ{STAND_ZERO_DEG_BY_JOINT[joint]:.1f}deg|="
                    f"{dev:.1f}deg > {dev_limit:.1f}deg")

        if joint in MECH_FIXED_JOINTS:
            # 기구 고정 — 검증은 위에서 그대로 거쳤고(불량 목표는 종전대로 거부), 통과한 목표는 버린다.
            # 거부가 아니므로 last_reject_reason 은 건드리지 않는다.
            self.mech_fixed_target_drops += 1
            return None
        self.desired_relative_deg_by_joint[joint] = target_deg
        return None

    def _baseline_is_set(self):
        return all(joint in self.baseline_deg_by_joint for joint in JOINT_NAMES_12DOF)

    def _has_any_baseline(self):
        return any(joint in self.baseline_deg_by_joint for joint in JOINT_NAMES_12DOF)

    def _baseline_missing_joints(self):
        return [
            joint for joint in JOINT_NAMES_12DOF
            if joint not in self.baseline_deg_by_joint
        ]

    def _feedback_is_fresh(self):
        if self.latest_feedback_time is None:
            return False
        return self.now_fn() - self.latest_feedback_time <= FEEDBACK_TIMEOUT_SEC

    def _feedback_age_sec(self):
        if self.latest_feedback_time is None:
            return None
        return max(0.0, self.now_fn() - self.latest_feedback_time)

    def _joint_feedback_age_sec(self, joint):
        if joint in MECH_FIXED_JOINTS:
            return 0.0   # 기구 고정 — 합성 피드백(항상 신선). 살아 있는 관절의 나이 판정에는 영향 없음
        timestamp = self.latest_feedback_time_by_joint.get(joint)
        if timestamp is None:
            return None
        return max(0.0, self.now_fn() - timestamp)

    def _joint_feedback_is_fresh(self, joint):
        age = self._joint_feedback_age_sec(joint)
        return age is not None and age <= FEEDBACK_TIMEOUT_SEC

    def _feedback_stale_reason(self):
        age = self._feedback_age_sec()
        if age is None:
            return "no_feedback"
        return f"feedback_age_sec:{age:.3f}>timeout:{FEEDBACK_TIMEOUT_SEC:.3f}"

    def _step_toward(self, current, desired, step_limit=None):
        limit = MAX_TARGET_STEP_DEG_PER_TICK if step_limit is None else float(step_limit)
        delta = float(desired) - float(current)
        if abs(delta) <= limit:
            return float(desired)
        direction = 1.0 if delta > 0.0 else -1.0
        return float(current) + direction * limit

    def _extract_actuals(self, payload):
        actuals = {}
        joints_deg = payload.get("joints_deg")
        if isinstance(joints_deg, dict):
            for joint in JOINT_NAMES_12DOF:
                value = _number_or_none(joints_deg.get(joint))
                if value is not None:
                    actuals[joint] = value

        joints = payload.get("joints")
        if isinstance(joints, dict):
            for joint in JOINT_NAMES_12DOF:
                joint_data = joints.get(joint)
                if not isinstance(joint_data, dict):
                    continue
                value = _number_or_none(joint_data.get("actual_deg"))
                if value is not None:
                    actuals[joint] = value
        for joint in MECH_FIXED_JOINTS:
            actuals.pop(joint, None)   # 기구 고정 — 합성값(stand_zero_deg)을 외부 값으로 덮지 않는다
        return actuals

    def _maybe_write_frame(self, motor_id, frame_bytes):
        if not REAL_CAN_WRITE_ENABLED:
            return False

        send_started = None
        try:
            import can
            bus = self._get_or_open_can_bus(can)
            message = can.Message(
                arbitration_id=int(motor_id),
                data=list(frame_bytes),
                is_extended_id=False,
            )
            send_started = time.monotonic()
            bus.send(message, timeout=CAN_SEND_TIMEOUT_SEC)
            self.can_status = "OK"
            self.can_last_error = None
            return True
        except Exception as error:
            # 타임아웃도 기존 실패 경로(can_status ERROR·_set_error)를 그대로
            # 탄다 — 카운터만 추가로 센다(진단 가시화, 가드 완화 아님).
            elapsed = None if send_started is None else (
                time.monotonic() - send_started)
            if _is_can_send_timeout(error, elapsed, CAN_SEND_TIMEOUT_SEC):
                self.can_send_timeouts += 1
            self.can_status = "ERROR"
            self.can_last_error = str(error)
            self._set_error(str(error))
            return False

    def _poll_can_feedback(self):
        if self.can_bus is None or not READ_ONLY_CAN_ENABLED:
            return 0

        decoded_count = 0
        for _index in range(CAN_FEEDBACK_MAX_FRAMES_PER_TICK):
            try:
                message = self.can_bus.recv(timeout=CAN_FEEDBACK_RECV_TIMEOUT_SEC)
            except Exception as error:
                self.can_status = "ERROR"
                self.can_last_error = str(error)
                self._set_error(f"can_recv_failed:{error}")
                break
            if message is None:
                break
            if self._handle_can_feedback_message(message):
                decoded_count += 1
        return decoded_count

    def _request_next_readonly_feedback(self):
        if self.can_bus is None or not READ_ONLY_CAN_ENABLED:
            return False
        if not POLL_JOINTS:
            return False
        now = self.now_fn()
        if now < self._readonly_request_backoff_until:
            return False
        if now - self._last_readonly_request_time < CAN_FEEDBACK_REQUEST_PERIOD_SEC:
            return False
        joint = POLL_JOINTS[self._next_readonly_request_index % len(POLL_JOINTS)]
        self._next_readonly_request_index += 1
        sent = self._send_readonly_feedback_request(CAN_ID_BY_JOINT[joint])
        if sent:
            self._last_readonly_request_time = now
        return sent

    def _request_all_readonly_feedback_once(self):
        sent = 0
        if self.can_bus is None or not READ_ONLY_CAN_ENABLED:
            return sent
        for joint in POLL_JOINTS:
            if self._send_readonly_feedback_request(CAN_ID_BY_JOINT[joint]):
                sent += 1
        return sent

    def _send_readonly_feedback_request(self, motor_id):
        if self.can_bus is None or not READ_ONLY_CAN_ENABLED:
            return False
        try:
            import can
            request = can.Message(
                arbitration_id=int(motor_id),
                data=list(FC_READONLY_REQUEST_DATA),
                is_extended_id=False,
            )
            self.can_bus.send(request, timeout=CAN_FEEDBACK_REQUEST_TIMEOUT_SEC)
            self.can_tx_readonly_count += 1
            return True
        except Exception as error:
            self.can_status = "ERROR"
            self.can_last_error = str(error)
            self._readonly_request_backoff_until = self.now_fn() + CAN_FEEDBACK_TX_BACKOFF_SEC
            self._set_error(f"readonly_feedback_request_failed:{error}")
            return False

    def _ensure_readonly_can_bus(self):
        if self.can_bus is not None or not READ_ONLY_CAN_ENABLED:
            return self.can_bus is not None
        if self.can_bus_open_attempted and self.can_status == "ERROR":
            return False
        self.can_bus_open_attempted = True
        try:
            import can
            self._get_or_open_can_bus(can)
            self.can_status = "BUS_OPEN_READ_ONLY" if not REAL_CAN_WRITE_ENABLED else "BUS_OPEN"
            self.can_last_error = None
            return True
        except Exception as error:
            self.can_status = "ERROR"
            self.can_last_error = str(error)
            self._set_error(f"can_bus_open_failed:{error}")
            return False

    def _handle_can_feedback_message(self, message):
        data = list(message.data)
        if data == FC_READONLY_REQUEST_DATA:
            self.can_rx_ignored_count += 1
            return False
        if len(data) != 8:
            self.can_rx_ignored_count += 1
            self.can_decode_error = f"ignored_frame_len:{len(data)}"
            return False
        try:
            feedback = decode_ak_mit_feedback(data)
        except Exception as error:
            self.can_decode_error = str(error)
            self.can_rx_ignored_count += 1
            return False

        # 프레임 형상 검증 (0802 march 데뷔 트립 교훈): 진짜 응답은
        # data[0]==arbitration_id∈1..12. 종전 arbitration_id 폴백은 '명령
        # 프레임'(data[0]=위치 상위바이트 0x7F~)까지 피드백으로 삼켜 ±수백°
        # 쓰레기 관절각을 만들었다 (candump 전수조사: 진짜 모터 피드백 54.6만
        # 프레임 중 쓰레기 0 — 쓰레기 41.5만 전부 폴백 오귀속).
        mid = int(feedback.motor_id)
        if mid != int(message.arbitration_id) or mid not in JOINT_BY_CAN_ID:
            self.can_rx_ignored_count += 1
            self.can_decode_error = (
                f"frame_shape_reject:arb{int(message.arbitration_id)}:d0_{mid}")
            return False
        joint = JOINT_BY_CAN_ID[mid]
        if joint in MECH_FIXED_JOINTS:
            # 기구 고정으로 선언된 모터 ID 의 프레임 — 모터가 없어야 하는 자리다. 합성값을 덮지 않고 버린다(계수·사유 노출).
            self.can_rx_ignored_count += 1
            self.can_decode_error = f"mech_fixed_frame_ignored:arb{mid}"
            return False

        now = self.now_fn()
        # 점프 필터: 직전 피드백이 신선(0.5s 내)한데 한 프레임에 60°+ 점프는
        # 물리 불가 — 드롭하고 카운트. 10연속이면 재동기 수용 (전원사이클 등).
        new_pos = float(feedback.position_deg)
        prev_pos = self.latest_actual_deg_by_joint.get(joint)
        prev_ts = self.latest_feedback_time_by_joint.get(joint)
        if (prev_pos is not None and prev_ts is not None
                and now - prev_ts < 0.5 and abs(new_pos - prev_pos) > 60.0):
            n = self.feedback_jump_drops.get(joint, 0) + 1
            self.feedback_jump_drops[joint] = n
            if n < 10:
                self.can_rx_ignored_count += 1
                return False
        self.feedback_jump_drops[joint] = 0
        self.latest_actual_deg_by_joint[joint] = new_pos
        self.latest_feedback_by_joint[joint] = feedback
        self.latest_feedback_time_by_joint[joint] = now
        self.latest_feedback_time = now
        self.can_rx_count += 1
        self.can_decode_error = None
        if feedback.error_ok is not True:
            self._warn(f"feedback_error:{joint}:{feedback.error_raw}")
        return True

    def _check_can_status(self):
        link_ok = self._refresh_can_link_status()
        if self.can_bus is not None:
            self.can_status = "BUS_OPEN_READ_ONLY" if not REAL_CAN_WRITE_ENABLED else "BUS_OPEN"
            self.can_last_error = None
        elif self.can_link_state == "UP":
            self.can_status = "LINK_UP_READ_ONLY" if READ_ONLY_CAN_ENABLED else "LINK_UP"
            if link_ok:
                self.can_last_error = None
            if READ_ONLY_CAN_ENABLED:
                self._ensure_readonly_can_bus()
        elif self.can_link_state == "DOWN":
            self.can_status = "LINK_DOWN"
        elif self.can_last_error:
            self.can_status = "ERROR"
        else:
            self.can_status = "UNKNOWN"
        self._request_all_readonly_feedback_once()
        self._poll_can_feedback()
        self.accepted = True
        return self.get_status()

    def _reconnect_can(self):
        self._close_can_bus()
        if not self._can_channel_is_allowed(CAN_CHANNEL):
            self.can_status = "BRINGUP_FAILED"
            self.can_last_error = f"invalid_can_channel:{CAN_CHANNEL}"
            return self._reject(self.can_last_error)

        commands = (
            ["ip", "link", "set", CAN_CHANNEL, "down"],
            [
                "ip",
                "link",
                "set",
                CAN_CHANNEL,
                "type",
                "can",
                "bitrate",
                str(CAN_BITRATE),
                "restart-ms",
                str(CAN_RESTART_MS),
            ],
            ["ip", "link", "set", CAN_CHANNEL, "up"],
        )
        for command in commands:
            result = self._run_allowed_ip_command(command)
            if result.returncode != 0:
                self.can_status = "BRINGUP_FAILED"
                self.can_last_error = self._format_command_error(command, result)
                self._set_error(self.can_last_error)
                return self.get_status()

        self._refresh_can_link_status()
        self.can_status = "BRINGUP_OK_READ_ONLY" if not REAL_CAN_WRITE_ENABLED else "BRINGUP_OK"
        self.can_last_error = None

        if READ_ONLY_CAN_ENABLED or REAL_CAN_WRITE_ENABLED:
            try:
                import can
                self._get_or_open_can_bus(can)
                self.can_status = "BUS_OPEN_READ_ONLY" if not REAL_CAN_WRITE_ENABLED else "BUS_OPEN"
                self._request_all_readonly_feedback_once()
                self._poll_can_feedback()
            except Exception as error:
                self.can_status = "BRINGUP_OK_BUS_OPEN_FAILED"
                self.can_last_error = str(error)
                self._warn(f"can_bus_open_failed:{error}")
        self.accepted = True
        return self.get_status()

    def _get_or_open_can_bus(self, can_module):
        if self.can_bus is None:
            self.can_bus = can_module.Bus(
                interface="socketcan",
                channel=CAN_CHANNEL,
                receive_own_messages=False,
            )
        return self.can_bus

    def _close_can_bus(self):
        if self.can_bus is None:
            return
        try:
            self.can_bus.shutdown()
        except Exception as error:
            self.can_last_error = str(error)
            self._warn(f"can_bus_shutdown_failed:{error}")
        finally:
            self.can_bus = None
            self.can_bus_open_attempted = False

    def _refresh_can_link_status(self):
        if not self._can_channel_is_allowed(CAN_CHANNEL):
            self.can_link_state = "UNKNOWN"
            self.can_last_error = f"invalid_can_channel:{CAN_CHANNEL}"
            return False
        result = self._run_allowed_ip_command(["ip", "-details", "link", "show", CAN_CHANNEL])
        if result.returncode != 0:
            self.can_link_state = "UNKNOWN"
            self.can_last_error = self._format_command_error(["ip", "-details", "link", "show", CAN_CHANNEL], result)
            self._set_error(self.can_last_error)
            return False

        output = result.stdout or ""
        state_match = re.search(r"\bstate\s+([A-Z]+)\b", output)
        bitrate_match = re.search(r"\bbitrate\s+([0-9]+)\b", output)
        self.can_link_state = state_match.group(1) if state_match else "UNKNOWN"
        self.can_bitrate = int(bitrate_match.group(1)) if bitrate_match else None
        self.can_last_error = None
        return True

    def _run_allowed_ip_command(self, command):
        self._validate_allowed_ip_command(command)
        try:
            result = subprocess.run(
                command,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=CAN_COMMAND_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(
                command,
                124,
                stdout=error.stdout or "",
                stderr=f"timeout:{error}",
            )
        except Exception as error:
            result = subprocess.CompletedProcess(command, 1, stdout="", stderr=str(error))
        self.can_last_stdout = result.stdout or ""
        self.can_last_stderr = result.stderr or ""
        return result

    def _validate_allowed_ip_command(self, command):
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("can_command_must_be_string_list")
        if not self._can_channel_is_allowed(CAN_CHANNEL):
            raise ValueError(f"invalid_can_channel:{CAN_CHANNEL}")

        allowed = (
            ["ip", "link", "set", CAN_CHANNEL, "down"],
            [
                "ip",
                "link",
                "set",
                CAN_CHANNEL,
                "type",
                "can",
                "bitrate",
                str(CAN_BITRATE),
                "restart-ms",
                str(CAN_RESTART_MS),
            ],
            ["ip", "link", "set", CAN_CHANNEL, "up"],
            ["ip", "-details", "link", "show", CAN_CHANNEL],
        )
        if command not in allowed:
            raise ValueError(f"can_command_not_allowed:{command}")

    def _can_channel_is_allowed(self, channel):
        return re.fullmatch(r"can[0-9]+", str(channel or "")) is not None

    def _format_command_error(self, command, result):
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "no_output"
        return f"can_command_failed:{' '.join(command)}:{result.returncode}:{detail}"

    def _reject(self, reason):
        self.last_reject_reason = reason
        self.accepted = False
        if reason not in self.errors:
            self.errors.append(reason)
        return self.get_status()

    def _takeover_status(self):
        """인수홀드 수명 상태 — status 노출 (최종점검 S5·S7: 무음 이탈 금지)."""
        holding = self.takeover_hold is not None
        out = {
            "waiting": self._waiting_takeover,
            "holding": holding,
            # 기구 고정 관절은 홀드 송신은 없지만 제자리에 굳어 있다 — 12 관절 기준 계수에 포함하고
            # 그중 몇 개가 고정인지 아래에 따로 적는다(선언이 있을 때만 필드 추가).
            "hold_joint_count": (len(self.takeover_hold) + len(MECH_FIXED_JOINTS)) if holding else 0,
            "hold_age_sec": (self.now_fn() - self._takeover_start
                             if holding and self._takeover_start is not None
                             else None),
            "ever_streamed": self._ever_streamed,
            "last_event": self.last_takeover_event,
        }
        if MECH_FIXED_JOINTS:
            out["hold_mech_fixed_count"] = len(MECH_FIXED_JOINTS) if holding else 0
        return out

    def _warn(self, reason):
        if reason not in self.warnings:
            self.warnings.append(reason)

    def _set_error(self, reason):
        if reason not in self.errors:
            self.errors.append(reason)

    def _clear_errors(self):
        self.errors = []
        self.last_reject_reason = None

    def get_status(self):
        joint_status = {}
        for joint in JOINT_NAMES_12DOF:
            preview = self.last_preview_by_joint.get(joint)
            preview_data = None
            if preview is not None:
                preview_data = {
                    "motor_id": preview.motor_id,
                    "joint": preview.joint,
                    "isaac_dof_index": preview.isaac_dof_index,
                    "p_des_rad": preview.p_des_rad,
                    "v_des_rad_s": preview.v_des_rad_s,
                    "kp": preview.kp,
                    "kd": preview.kd,
                    "tau_ff": preview.tau_ff,
                    "frame_hex": preview.frame_hex,
                }
            joint_status[joint] = {
                "allowed": joint in ALLOWED_JOINTS,
                "motor_id": CAN_ID_BY_JOINT[joint],
                "isaac_dof_index": ISAAC_DOF_INDEX_BY_JOINT[joint],
                "baseline_deg": self.baseline_deg_by_joint.get(joint),
                "latest_actual_deg": self.latest_actual_deg_by_joint.get(joint),
                "feedback_age_sec": self._joint_feedback_age_sec(joint),
                "feedback_stale": not self._joint_feedback_is_fresh(joint),
                "feedback_valid": (joint in self.latest_feedback_by_joint
                                   or joint in MECH_FIXED_JOINTS),
                "desired_relative_deg": self.desired_relative_deg_by_joint[joint],
                "commanded_relative_deg": self.commanded_relative_deg_by_joint[joint],
                "preview": preview_data,
            }
            feedback = self.latest_feedback_by_joint.get(joint)
            if feedback is not None:
                joint_status[joint].update(
                    {
                        "feedback_velocity_rad_s": feedback.velocity_rad_s,
                        "feedback_torque_est": feedback.torque_est,
                        "feedback_temperature_raw": feedback.temperature_raw,
                        "feedback_error_raw": feedback.error_raw,
                        "feedback_error_ok": feedback.error_ok,
                    }
                )
            elif joint in MECH_FIXED_JOINTS:
                # 합성값임을 구분하는 표지 + 속도·토크 0 (온도·오류 바이트는 모터가 없어 싣지 않는다)
                joint_status[joint].update(
                    {
                        "mech_fixed": True,
                        "feedback_velocity_rad_s": 0.0,
                        "feedback_torque_est": 0.0,
                    }
                )

        return {
            "timestamp": self.now_fn(),
            "node": "stage8_12axis_mit_control_node",
            "command_topic": COMMAND_TOPIC,
            "status_topic": STATUS_TOPIC,
            "joint_states_topic": JOINT_STATES_TOPIC,
            "real_can_write_enabled": REAL_CAN_WRITE_ENABLED,
            "read_only_can_enabled": READ_ONLY_CAN_ENABLED,
            "can_channel": CAN_CHANNEL,
            "can_status": self.can_status,
            "can_link_state": self.can_link_state,
            "can_bitrate": self.can_bitrate,
            "can_last_error": self.can_last_error,
            "can_rx_count": self.can_rx_count,
            "can_tx_readonly_count": self.can_tx_readonly_count,
            "can_rx_ignored_count": self.can_rx_ignored_count,
            "can_send_timeouts": self.can_send_timeouts,
            "can_send_timeout_sec": CAN_SEND_TIMEOUT_SEC,
            "can_decode_error": self.can_decode_error,
            "feedback_stale": not self._feedback_is_fresh(),
            "feedback_age_sec": self._feedback_age_sec(),
            "feedback_stale_reason": None if self._feedback_is_fresh() else self._feedback_stale_reason(),
            "allowed_joints": list(ALLOWED_JOINTS),
            "mech_fixed_joints": [joint for joint in JOINT_NAMES_12DOF if joint in MECH_FIXED_JOINTS],
            "mech_fixed_target_drops": self.mech_fixed_target_drops,
            "joint_count": len(JOINT_NAMES_12DOF),
            "armed": self.armed,
            "running": self.running,
            "accepted": self.accepted,
            "executed": self.executed,
            "command_seq": self.command_seq,
            "cmd_timing": self.cmd_timing.snapshot(self.now_fn()),
            "last_command": self.last_command,
            "last_reject_reason": self.last_reject_reason,
            "soft_clamp_counts": dict(self.soft_clamp_counts),
            "soft_clamp_total": sum(self.soft_clamp_counts.values()),
            "takeover": self._takeover_status(),
            "baseline_source": self.baseline_source,
            "baseline_hold_minus_actual_deg": self.baseline_hold_minus_actual_deg,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "mapping_valid": self.mapping_valid,
            "motor_id_by_joint": dict(CAN_ID_BY_JOINT),
            "isaac_dof_index_by_joint": dict(ISAAC_DOF_INDEX_BY_JOINT),
            "default_limit_gain_by_joint": dict(DEFAULT_LIMIT_GAIN_BY_JOINT),
            "control_period_sec": CONTROL_PERIOD_SEC,
            # 1006 지연 옵션(기본 전부 꺼짐)과 계측: 송신 경로별 횟수·송신 후 피드백 대기
            "latency_options": latency_options_status(),
            "send_counts": dict(self.send_counts),
            "fb_after_send": dict(self.fb_after_send),
            "tick_timing": self._tick_timing_status(),
            "max_target_step_deg_per_tick": MAX_TARGET_STEP_DEG_PER_TICK,
            # 1010 요구 토크 상한: 켠 때만 싣는다(꺼짐 = status 가 종전과 같은 키)
            **({"target_torque_cap_nm": TARGET_TORQUE_CAP_NM, "target_torque_cap_by_name": dict(TARGET_TORQUE_CAP_BY_NAME),
                "target_torque_cap_counts": dict(self.torque_cap_counts)} if torque_cap_enabled() else {}),
            "baseline_set": self._has_any_baseline(),
            "baseline_complete": self._baseline_is_set(),
            "baseline_count": len(self.baseline_deg_by_joint),
            "baseline_missing": not self._has_any_baseline(),
            "baseline_missing_joints": self._baseline_missing_joints(),
            "joints": joint_status,
        }


class Stage8TwelveAxisMitControlNode:
    def __init__(self, rclpy_module, string_type):
        self.rclpy = rclpy_module
        self.String = string_type
        self.node = rclpy_module.create_node("stage8_12axis_mit_control_node")
        self.core = Stage8TwelveAxisMitControlCore()
        self._last_cmd_status_pub = 0.0
        self.status_publisher = self.node.create_publisher(string_type, STATUS_TOPIC, 10)
        self.command_subscription = self.node.create_subscription(
            string_type,
            COMMAND_TOPIC,
            self._on_command_message,
            10,
        )
        self.joint_states_subscription = self.node.create_subscription(
            string_type,
            JOINT_STATES_TOPIC,
            self._on_joint_states_message,
            10,
        )
        self._last_status_pub = 0.0
        self.control_timer = self.node.create_timer(CONTROL_PERIOD_SEC, self._on_control_timer)
        if FEEDBACK_AFTER_SEND:
            self.status_timer = self.node.create_timer(STATUS_PERIOD_SEC, self._on_status_timer)
        else:
            self.status_timer = self.node.create_timer(STATUS_PERIOD_SEC, self.publish_status)
        # 1006 --send-on-command: armed 배경(감시) 송신 판정 타이머 (5 ms 해상도). 옵션 꺼짐이면 만들지 않는다.
        self.watchdog_timer = (self.node.create_timer(BACKGROUND_CHECK_SEC, self._on_watchdog_timer)
                               if SEND_ON_COMMAND else None)

    def _on_command_message(self, message):
        self.core.handle_command_data(message.data)
        sent_now = False
        if self.core.should_send_on_command():
            # 1006 --send-on-command: 다음 제어 타이머(0–20 ms)를 기다리지 않고 지금 송신.
            # 검증·슬루(시간 기준)·동결 규칙은 타이머 틱과 같은 control_tick 경로 그대로.
            self.core.control_tick(source="command")
            sent_now = bool(self.core._tick_sent_ids)
        # 지연 수술 0803: 종전 '명령마다 status 발행'은 라이브 브리지(12명령/틱)
        # 에서 ~700Hz JSON 직렬화 폭풍이었다 (전체 status ~수 KB — 젯슨 CPU 잠식
        # = 제어 루프 지터 = 실측 루프 지연의 소프트웨어 몫). 5ms 캡 —
        # stage8_cmd 검증 왕복은 100Hz 주기 발행이 어차피 10ms 내 보장.
        now = time.time()
        if sent_now and FEEDBACK_AFTER_SEND:
            self._last_cmd_status_pub = now
            self.publish_status()          # 방금 읽은 신선한 피드백을 바로 내보낸다
        elif now - self._last_cmd_status_pub >= 0.005:
            self._last_cmd_status_pub = now
            self.publish_status()

    def _on_joint_states_message(self, message):
        self.core.handle_joint_states_data(message.data)

    def _on_control_timer(self):
        if SEND_ON_COMMAND and self.core.armed:
            return   # armed 송신은 명령 즉시 송신 + 배경(감시) 타이머가 맡는다
        self._tick_and_maybe_publish("timer")

    def _on_watchdog_timer(self):
        if self.core.wants_background_send():
            self._tick_and_maybe_publish("watchdog")

    def _tick_and_maybe_publish(self, source):
        self.core.control_tick(source=source)
        if FEEDBACK_AFTER_SEND and self.core._tick_sent_ids:
            self.publish_status()

    def _on_status_timer(self):
        # --feedback-after-send 모드: 송신 직후 이미 발행했으면(5 ms 내) 주기 발행 생략 — 직렬화 부하 억제
        if time.time() - self._last_status_pub < STATUS_PERIOD_SEC * 0.5:
            return
        self.publish_status()

    def publish_status(self):
        self._last_status_pub = time.time()
        message = self.String()
        message.data = json.dumps(self.core.get_status(), ensure_ascii=False, sort_keys=True)
        self.status_publisher.publish(message)

    def destroy(self):
        self.core._close_can_bus()
        self.node.destroy_node()


def load_ros2():
    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError as error:
        return None, None, error
    return rclpy, String, None


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage8 12-axis AK MIT real-control node")
    parser.add_argument(
        "--can-channel", default=CAN_CHANNEL,
        help="SocketCAN 채널 (기본 can0 — 메인 PC 호환). 젯슨 라이브 버스는 can1 "
             "이므로 젯슨 배포 시 --can-channel can1 로 기동할 것")
    parser.add_argument(
        "--wait-takeover", action="store_true",
        help="선기동 대기: 핸드오프 상태 파일이 나타날 때까지 CAN 무송신 대기 "
             "후 즉시 인수 홀드 (무토크 공백 ~20ms화). hang_stand 핸드오프 전에 "
             "이 플래그로 먼저 기동해둘 것")
    parser.add_argument(
        "--can-send-timeout", default=None,
        help=f"CAN 송신 타임아웃 초 ({CAN_SEND_TIMEOUT_MIN_SEC}~"
             f"{CAN_SEND_TIMEOUT_MAX_SEC}). 미지정(기본) = 모듈 상수 "
             f"{CAN_SEND_TIMEOUT_SEC} 유지 = 종전과 동일 동작. H-D1 버스 정체 "
             "판별 시에만 낮춘다 (예: --can-send-timeout 0.01)")
    # 1006 실효 지연 단축 옵션 — 전부 기본 꺼짐 (미지정 = 종전 동작과 동일)
    parser.add_argument(
        "--feedback-after-send", action="store_true",
        help="명령 송신 직후 같은 틱 안에서 응답(피드백)을 읽고 status 를 바로 발행 "
             "(관측 나이 약 −1 틱). 대기 상한은 --feedback-wait-ms")
    parser.add_argument(
        "--feedback-wait-ms", default=None,
        help=f"--feedback-after-send 응답 대기 상한 ms ({FEEDBACK_WAIT_MIN_SEC * 1000:g}~"
             f"{FEEDBACK_WAIT_MAX_SEC * 1000:g}, 기본 {FEEDBACK_WAIT_SEC * 1000:g})")
    parser.add_argument(
        "--send-on-command", action="store_true",
        help="SET_JOINT_TARGETS 수신 즉시 CAN 송신 (종전: 다음 제어 타이머까지 0–20 ms 대기). "
             "슬루는 시간 기준 200°/s 로 유지, 명령이 25 ms 넘게 끊기면 배경 송신이 종전 주기로 이어 받음")
    parser.add_argument(
        "--control-hz", type=int, default=None, choices=list(ALLOWED_CONTROL_HZ),
        help="제어(송신) 주기 Hz (기본 50 = 종전). 100 은 버스 부하 2배·실물 미검증")
    parser.add_argument(
        "--target-torque-cap-nm", default=None,
        help=f"1010 요구 토크 상한 N·m (0/미지정 = 끔 = 종전, {TARGET_TORQUE_CAP_RANGE_NM[0]:g}~{TARGET_TORQUE_CAP_RANGE_NM[1]:g}). "
             "보내는 목표를 실측각 ± degrees(상한/kp) 안으로 묶는다")
    parser.add_argument(
        "--target-torque-cap-joint", action="append", default=None, metavar="이름조각=N·m",
        help="관절 이름에 이 조각이 든 관절만의 상한(예: hip_a=20). 여러 번 줄 수 있다. 전체 상한 없이 이것만 줘도 된다")
    return parser


def apply_latency_args(args):
    """지연 옵션 인자를 모듈 전역에 반영. 잘못된 값은 ValueError (기동 거부)."""
    global FEEDBACK_AFTER_SEND, FEEDBACK_WAIT_SEC, SEND_ON_COMMAND, CONTROL_PERIOD_SEC, TARGET_TORQUE_CAP_NM, TARGET_TORQUE_CAP_BY_NAME
    cap_by_name = parse_target_torque_cap_joints(getattr(args, "target_torque_cap_joint", None))
    cap = TARGET_TORQUE_CAP_NM
    if getattr(args, "target_torque_cap_nm", None) is not None:
        cap = parse_target_torque_cap(args.target_torque_cap_nm)
    wait = FEEDBACK_WAIT_SEC
    if getattr(args, "feedback_wait_ms", None) is not None:
        wait = parse_feedback_wait_ms(args.feedback_wait_ms)
    period = CONTROL_PERIOD_SEC
    hz = getattr(args, "control_hz", None)
    if hz is not None:
        if hz not in ALLOWED_CONTROL_HZ:
            raise ValueError(f"--control-hz {hz} 불가 (허용 {ALLOWED_CONTROL_HZ})")
        period = 1.0 / float(hz)
    fas = bool(getattr(args, "feedback_after_send", False))
    if fas and wait > 0.5 * period:
        # 대기가 주기의 절반을 넘으면 단일 스레드 실행기가 명령 콜백을 그만큼 못 받는다
        raise ValueError(f"--feedback-wait-ms {wait * 1000:g} 가 제어 주기 절반({period * 500:g} ms) 초과")
    FEEDBACK_AFTER_SEND = fas
    FEEDBACK_WAIT_SEC = wait
    SEND_ON_COMMAND = bool(getattr(args, "send_on_command", False))
    CONTROL_PERIOD_SEC = period
    TARGET_TORQUE_CAP_NM = cap
    TARGET_TORQUE_CAP_BY_NAME = cap_by_name


def main():
    global CAN_CHANNEL, WAIT_TAKEOVER, CAN_SEND_TIMEOUT_SEC

    parser = build_arg_parser()
    # ROS2 런치가 붙이는 --ros-args 등 미지 인자는 무시 (parse_known_args)
    args, _unknown = parser.parse_known_args()
    if re.fullmatch(r"can[0-9]+", str(args.can_channel)) is None:
        print(f"[Stage8-12Axis] invalid --can-channel: {args.can_channel!r} "
              "(can0/can1/... 형식만 허용)")
        return 1
    if args.can_send_timeout is not None:
        try:
            CAN_SEND_TIMEOUT_SEC = parse_can_send_timeout(args.can_send_timeout)
        except ValueError as error:
            print(f"[Stage8-12Axis] {error}")
            return 1
    try:
        apply_latency_args(args)
    except ValueError as error:
        print(f"[Stage8-12Axis] {error}")
        return 1
    CAN_CHANNEL = str(args.can_channel)
    WAIT_TAKEOVER = bool(args.wait_takeover)

    rclpy, String, error = load_ros2()
    if rclpy is None:
        print(f"[Stage8-12Axis] ROS2 import failed: {error}")
        return 1

    rclpy.init()
    node = Stage8TwelveAxisMitControlNode(rclpy, String)
    print(f"[Stage8-12Axis] command topic = {COMMAND_TOPIC}")
    print(f"[Stage8-12Axis] status topic = {STATUS_TOPIC}")
    print(f"[Stage8-12Axis] CAN channel = {CAN_CHANNEL}")
    print(f"[Stage8-12Axis] CAN send timeout = {CAN_SEND_TIMEOUT_SEC} s")
    if torque_cap_enabled():      # 1010 요구 토크 상한 — 기동 로그에 남긴다(현장에서 켜졌는지 확인용)
        print(f"[Stage8-12Axis] 요구 토크 상한 켬: 전 관절 {TARGET_TORQUE_CAP_NM:g} N·m, 관절별 {TARGET_TORQUE_CAP_BY_NAME} "
              f"(허용 |목표−실측| = degrees(상한/kp))", flush=True)
    print(f"[Stage8-12Axis] latency options = {latency_options_status()}")
    print(f"[Stage8-12Axis] REAL_CAN_WRITE_ENABLED = {REAL_CAN_WRITE_ENABLED}")
    print(f"[Stage8-12Axis] ALLOWED_JOINTS = {ALLOWED_JOINTS}")
    if MECH_FIXED_JOINTS:
        print("[Stage8-12Axis] 기구 고정(송신 제외·피드백 합성) = "
              f"{[(CAN_ID_BY_JOINT[j], j) for j in JOINT_NAMES_12DOF if j in MECH_FIXED_JOINTS]}")
    print("[Stage8-12Axis] 12-axis MIT control node started")
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
