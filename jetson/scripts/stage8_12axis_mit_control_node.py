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
    get_joint_name_to_motor_id,
    validate_12dof_mapping,
)


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
CAN_FEEDBACK_RECV_TIMEOUT_SEC = 0.0
CAN_FEEDBACK_MAX_FRAMES_PER_TICK = 64
CAN_FEEDBACK_REQUEST_TIMEOUT_SEC = 0.01
CAN_FEEDBACK_REQUEST_PERIOD_SEC = 0.02
CAN_FEEDBACK_TX_BACKOFF_SEC = 1.0
FC_READONLY_REQUEST_DATA = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]

CONTROL_PERIOD_SEC = 0.02
STATUS_PERIOD_SEC = 0.01
MAX_TARGET_STEP_DEG_PER_TICK = 4.0
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


class Stage8TwelveAxisMitControlCore:
    def __init__(self, now_fn=time.time):
        self.now_fn = now_fn
        self.armed = False
        self.baseline_deg_by_joint = {}
        self.baseline_set_time = None
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
        self.latest_actual_deg_by_joint = {}
        self.latest_feedback_by_joint = {}
        self.latest_feedback_time_by_joint = {}
        self.latest_feedback_time = None
        self.desired_relative_deg_by_joint = {joint: 0.0 for joint in JOINT_NAMES_12DOF}
        self.commanded_relative_deg_by_joint = {joint: 0.0 for joint in JOINT_NAMES_12DOF}
        self.last_preview_by_joint = {}
        self.command_seq = 0
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
        self.can_decode_error = None
        self.feedback_stale_reason = "no_feedback"
        self.baseline_missing_joints = list(JOINT_NAMES_12DOF)
        self._next_readonly_request_index = 0
        self._last_readonly_request_time = 0.0
        self._readonly_request_backoff_until = 0.0
        self.accepted = False
        self.executed = False
        self.running = False

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
            for joint in self.baseline_deg_by_joint:
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
            if len(hold) != len(JOINT_NAMES_12DOF):
                self.last_takeover_event = f"state_file_incomplete:{len(hold)}/12"
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
            return
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
            return
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
        for joint, (deg, kp, kd) in self.takeover_hold.items():
            frame = pack_ak_mit_command(
                math.radians(deg), v_des_rad_s=V_DES_DEFAULT,
                kp=kp, kd=kd, tau_ff=TAU_FF_DEFAULT)
            self._maybe_write_frame(CAN_ID_BY_JOINT[joint], frame)

    def control_tick(self):
        self.executed = False
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
            self._takeover_hold_tick()
        elif not self.armed and not self._ever_streamed:
            self._request_next_readonly_feedback()
        self._poll_can_feedback()
        if not self.armed or not self._has_any_baseline():
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

        for joint in JOINT_NAMES_12DOF:
            if joint not in self.baseline_deg_by_joint:
                continue
            joint_stale = (REAL_CAN_WRITE_ENABLED
                           and not self._joint_feedback_is_fresh(joint))
            if joint_stale:
                self._warn(f"joint_feedback_stale:{joint}")
            current = self.commanded_relative_deg_by_joint[joint]
            if global_stale or joint_stale:
                next_value = current   # 동결 — 진행 없이 현 목표 재송신
            else:
                desired = self.desired_relative_deg_by_joint[joint]
                next_value = self._step_toward(current, desired)
            if next_value != current:
                moved = True
            self.commanded_relative_deg_by_joint[joint] = next_value

            absolute_target_deg = self.baseline_deg_by_joint[joint] + next_value
            preview = build_12axis_mit_preview(joint, absolute_target_deg)
            self.last_preview_by_joint[joint] = preview
            self.executed = self._maybe_write_frame(preview.motor_id, preview.frame_bytes) or self.executed

        if moved and self.last_reject_reason is None:
            self.accepted = True
        return self.get_status()

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
        if not available:
            if not self._feedback_is_fresh():
                self.feedback_stale_reason = self._feedback_stale_reason()
                return self._reject("feedback_stale:all_baseline_missing")
            return self._reject("baseline_missing_actual:all")

        self.baseline_deg_by_joint = {
            joint: float(self.latest_actual_deg_by_joint[joint])
            for joint in available
        }
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

    def _step_toward(self, current, desired):
        delta = float(desired) - float(current)
        if abs(delta) <= MAX_TARGET_STEP_DEG_PER_TICK:
            return float(desired)
        direction = 1.0 if delta > 0.0 else -1.0
        return float(current) + direction * MAX_TARGET_STEP_DEG_PER_TICK

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
        return actuals

    def _maybe_write_frame(self, motor_id, frame_bytes):
        if not REAL_CAN_WRITE_ENABLED:
            return False

        try:
            import can
            bus = self._get_or_open_can_bus(can)
            message = can.Message(
                arbitration_id=int(motor_id),
                data=list(frame_bytes),
                is_extended_id=False,
            )
            bus.send(message, timeout=0.2)
            self.can_status = "OK"
            self.can_last_error = None
            return True
        except Exception as error:
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
        if not JOINT_NAMES_12DOF:
            return False
        now = self.now_fn()
        if now < self._readonly_request_backoff_until:
            return False
        if now - self._last_readonly_request_time < CAN_FEEDBACK_REQUEST_PERIOD_SEC:
            return False
        joint = JOINT_NAMES_12DOF[self._next_readonly_request_index % len(JOINT_NAMES_12DOF)]
        self._next_readonly_request_index += 1
        sent = self._send_readonly_feedback_request(CAN_ID_BY_JOINT[joint])
        if sent:
            self._last_readonly_request_time = now
        return sent

    def _request_all_readonly_feedback_once(self):
        sent = 0
        if self.can_bus is None or not READ_ONLY_CAN_ENABLED:
            return sent
        for joint in JOINT_NAMES_12DOF:
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
        return {
            "waiting": self._waiting_takeover,
            "holding": holding,
            "hold_joint_count": len(self.takeover_hold) if holding else 0,
            "hold_age_sec": (self.now_fn() - self._takeover_start
                             if holding and self._takeover_start is not None
                             else None),
            "ever_streamed": self._ever_streamed,
            "last_event": self.last_takeover_event,
        }

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
                "feedback_valid": joint in self.latest_feedback_by_joint,
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
            "can_decode_error": self.can_decode_error,
            "feedback_stale": not self._feedback_is_fresh(),
            "feedback_age_sec": self._feedback_age_sec(),
            "feedback_stale_reason": None if self._feedback_is_fresh() else self._feedback_stale_reason(),
            "allowed_joints": list(ALLOWED_JOINTS),
            "joint_count": len(JOINT_NAMES_12DOF),
            "armed": self.armed,
            "running": self.running,
            "accepted": self.accepted,
            "executed": self.executed,
            "command_seq": self.command_seq,
            "last_command": self.last_command,
            "last_reject_reason": self.last_reject_reason,
            "soft_clamp_counts": dict(self.soft_clamp_counts),
            "soft_clamp_total": sum(self.soft_clamp_counts.values()),
            "takeover": self._takeover_status(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "mapping_valid": self.mapping_valid,
            "motor_id_by_joint": dict(CAN_ID_BY_JOINT),
            "isaac_dof_index_by_joint": dict(ISAAC_DOF_INDEX_BY_JOINT),
            "default_limit_gain_by_joint": dict(DEFAULT_LIMIT_GAIN_BY_JOINT),
            "control_period_sec": CONTROL_PERIOD_SEC,
            "max_target_step_deg_per_tick": MAX_TARGET_STEP_DEG_PER_TICK,
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
        self.control_timer = self.node.create_timer(CONTROL_PERIOD_SEC, self._on_control_timer)
        self.status_timer = self.node.create_timer(STATUS_PERIOD_SEC, self.publish_status)

    def _on_command_message(self, message):
        self.core.handle_command_data(message.data)
        # 지연 수술 0803: 종전 '명령마다 status 발행'은 라이브 브리지(12명령/틱)
        # 에서 ~700Hz JSON 직렬화 폭풍이었다 (전체 status ~수 KB — 젯슨 CPU 잠식
        # = 제어 루프 지터 = 실측 루프 지연의 소프트웨어 몫). 5ms 캡 —
        # stage8_cmd 검증 왕복은 100Hz 주기 발행이 어차피 10ms 내 보장.
        now = time.time()
        if now - self._last_cmd_status_pub >= 0.005:
            self._last_cmd_status_pub = now
            self.publish_status()

    def _on_joint_states_message(self, message):
        self.core.handle_joint_states_data(message.data)

    def _on_control_timer(self):
        self.core.control_tick()

    def publish_status(self):
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


def main():
    global CAN_CHANNEL, WAIT_TAKEOVER

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
    # ROS2 런치가 붙이는 --ros-args 등 미지 인자는 무시 (parse_known_args)
    args, _unknown = parser.parse_known_args()
    if re.fullmatch(r"can[0-9]+", str(args.can_channel)) is None:
        print(f"[Stage8-12Axis] invalid --can-channel: {args.can_channel!r} "
              "(can0/can1/... 형식만 허용)")
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
    print(f"[Stage8-12Axis] REAL_CAN_WRITE_ENABLED = {REAL_CAN_WRITE_ENABLED}")
    print(f"[Stage8-12Axis] ALLOWED_JOINTS = {ALLOWED_JOINTS}")
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
