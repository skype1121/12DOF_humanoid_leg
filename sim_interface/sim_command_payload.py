"""Pure Python SIM command payload builder.

UI/command_bus에서 안전 검사를 통과한 joint command를 ROS/Isaac 쪽 dry-run
payload로 넘기기 위한 공통 helper다.

주의:
- Isaac import 없음
- ROS publish 없음
- CAN/AK/Jetson import 없음
- 단위는 degree로 고정한다. Isaac 적용 직전
  sim_interface/isaac_12dof_command_applier.py에서 radian으로 변환한다.
"""

import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control_logic.config_loader import load_robot_config  # noqa: E402
from control_logic.safety_filter import SafetyFilter  # noqa: E402
from robot_runtime.dof12_mapping import JOINT_NAMES_12  # noqa: E402


ALLOWED_SIM_PAYLOAD_MODES = ("SIM", "SIM_TO_REAL")
SIM_TARGET_UNIT = SafetyFilter.TARGET_UNIT
SIM_LIMIT_POLICY = SafetyFilter.LIMIT_POLICY


def _is_plain_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_mode(mode):
    if mode not in ALLOWED_SIM_PAYLOAD_MODES:
        allowed = ", ".join(ALLOWED_SIM_PAYLOAD_MODES)
        raise ValueError(f"SIM payload mode must be one of: {allowed}")


def _validate_joint_commands(joint_commands):
    if not isinstance(joint_commands, dict):
        raise ValueError("joint_commands must be dict")
    if not joint_commands:
        raise ValueError("joint_commands must not be empty")

    extra_joints = [joint_name for joint_name in joint_commands if joint_name not in JOINT_NAMES_12]
    if extra_joints:
        raise ValueError(f"unknown 12DOF joint: {extra_joints[0]}")


def _validate_target_deg(joint_name, target_deg, config):
    if not _is_plain_number(target_deg):
        raise ValueError(f"target_deg must be numeric: {joint_name}")
    target_deg = float(target_deg)
    if not math.isfinite(target_deg):
        raise ValueError(f"target_deg must be finite: {joint_name}")

    joint_limits = config["joint_limits_deg"][joint_name]
    min_deg = float(joint_limits["min"])
    max_deg = float(joint_limits["max"])
    if target_deg < min_deg or target_deg > max_deg:
        raise ValueError(
            "joint command limit exceeded: "
            f"joint_name={joint_name} "
            f"target_deg={target_deg} "
            f"min_deg={min_deg} "
            f"max_deg={max_deg} "
            f"unit={SIM_TARGET_UNIT} "
            f"policy={SIM_LIMIT_POLICY}"
        )
    return target_deg


def _ordered_joint_commands(joint_commands, config):
    ordered = {}
    for joint_name in JOINT_NAMES_12:
        if joint_name in joint_commands:
            ordered[joint_name] = _validate_target_deg(
                joint_name,
                joint_commands[joint_name],
                config,
            )
    return ordered


def build_sim_joint_payload(mode, joint_commands, source="ui", timestamp=None):
    """안전 검사를 통과한 12DOF joint command를 SIM bridge payload로 만든다.

    joint_commands의 key 순서는 입력 순서를 믿지 않고 JOINT_NAMES_12 순서로 재정렬한다.
    limit 초과 정책은 safety_filter와 동일하게 clamp가 아니라 reject다.
    """
    _validate_mode(mode)
    _validate_joint_commands(joint_commands)

    payload_timestamp = time.time() if timestamp is None else timestamp
    if not _is_plain_number(payload_timestamp):
        raise ValueError("timestamp must be numeric")
    payload_timestamp = float(payload_timestamp)
    if not math.isfinite(payload_timestamp):
        raise ValueError("timestamp must be finite")

    config = load_robot_config()
    ordered_commands = _ordered_joint_commands(joint_commands, config)

    return {
        "mode": mode,
        "target_unit": SIM_TARGET_UNIT,
        "joint_commands": ordered_commands,
        "source": str(source),
        "timestamp": payload_timestamp,
    }
