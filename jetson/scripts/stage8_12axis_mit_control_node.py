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

COMMAND_ARM_ALL = "ARM_ALL"
COMMAND_DISARM_ALL = "DISARM_ALL"
COMMAND_SET_BASELINE_ALL = "SET_BASELINE_FROM_CURRENT_ALL"
COMMAND_SET_JOINT_TARGET = "SET_JOINT_TARGET"
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
        self.warnings = []
        self.errors = []
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
            self.armed = True
            self.accepted = True
            return self.get_status()

        if command == COMMAND_DISARM_ALL:
            self.armed = False
            self.running = False
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

        if command == COMMAND_CHECK_CAN:
            return self._check_can_status()

        if command == COMMAND_RECONNECT_CAN:
            return self._reconnect_can()

        if command == COMMAND_SET_JOINT_TARGET:
            return self._set_joint_target(payload)

        return self._reject("unsupported_command")

    def control_tick(self):
        self.executed = False
        self._ensure_readonly_can_bus()
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

        if REAL_CAN_WRITE_ENABLED and not self._feedback_is_fresh():
            self.running = False
            return self._reject("feedback_stale_real_blocked")

        self.running = True
        moved = False
        self.last_preview_by_joint = {}

        if not self._feedback_is_fresh() and not REAL_CAN_WRITE_ENABLED:
            self._warn("feedback_stale_dry_run")

        for joint in JOINT_NAMES_12DOF:
            if joint not in self.baseline_deg_by_joint:
                continue
            if REAL_CAN_WRITE_ENABLED and not self._joint_feedback_is_fresh(joint):
                self._warn(f"joint_feedback_stale:{joint}")
                continue
            desired = self.desired_relative_deg_by_joint[joint]
            current = self.commanded_relative_deg_by_joint[joint]
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
        if joint not in JOINT_NAMES_12DOF:
            return self._reject("unknown_joint")
        if joint not in ALLOWED_JOINTS:
            return self._reject(f"joint_not_allowed:{joint}")
        if joint not in self.baseline_deg_by_joint:
            self.baseline_missing_joints = self._baseline_missing_joints()
            return self._reject(f"baseline_missing:{joint}")
        if REAL_CAN_WRITE_ENABLED and not self._joint_feedback_is_fresh(joint):
            return self._reject(f"joint_feedback_stale:{joint}")

        try:
            target_deg = float(payload.get("target_deg"))
        except (TypeError, ValueError):
            return self._reject("target_invalid")

        config = get_joint_config(joint)
        if abs(target_deg) > float(config["hard_limit_deg"]):
            return self._reject("hard_limit_exceeded")
        if abs(target_deg) > float(config["target_limit_deg"]):
            return self._reject("target_limited")

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
            return self._reject(
                f"{limit_verdict}:{joint}:joint{joint_abs_deg:.1f}deg"
                f"(motor{absolute_deg:.1f}deg)")

        # 모터 프레임 백스톱: stand_zero 로부터의 편차가 하드리밋 폭+여유를
        # 넘으면 거부 — DIR 부호와 무관한 2차 방어선 (프레임 근거는 상수 정의부).
        dev = abs(absolute_deg - STAND_ZERO_DEG_BY_JOINT[joint])
        dev_limit = MOTOR_DEV_BACKSTOP_DEG_BY_JOINT[joint]
        if dev > dev_limit:
            return self._reject(
                f"motor_dev_backstop:{joint}:|motor{absolute_deg:.1f}deg-"
                f"SZ{STAND_ZERO_DEG_BY_JOINT[joint]:.1f}deg|="
                f"{dev:.1f}deg > {dev_limit:.1f}deg")

        self.desired_relative_deg_by_joint[joint] = target_deg
        self.accepted = True
        return self.get_status()

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

        joint = JOINT_BY_CAN_ID.get(int(feedback.motor_id))
        if joint is None:
            joint = JOINT_BY_CAN_ID.get(int(message.arbitration_id))
        if joint is None:
            self.can_decode_error = f"unknown_feedback_motor_id:{feedback.motor_id}"
            self.can_rx_ignored_count += 1
            return False

        now = self.now_fn()
        self.latest_actual_deg_by_joint[joint] = float(feedback.position_deg)
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
    global CAN_CHANNEL

    parser = argparse.ArgumentParser(
        description="Stage8 12-axis AK MIT real-control node")
    parser.add_argument(
        "--can-channel", default=CAN_CHANNEL,
        help="SocketCAN 채널 (기본 can0 — 메인 PC 호환). 젯슨 라이브 버스는 can1 "
             "이므로 젯슨 배포 시 --can-channel can1 로 기동할 것")
    # ROS2 런치가 붙이는 --ros-args 등 미지 인자는 무시 (parse_known_args)
    args, _unknown = parser.parse_known_args()
    if re.fullmatch(r"can[0-9]+", str(args.can_channel)) is None:
        print(f"[Stage8-12Axis] invalid --can-channel: {args.can_channel!r} "
              "(can0/can1/... 형식만 허용)")
        return 1
    CAN_CHANNEL = str(args.can_channel)

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
