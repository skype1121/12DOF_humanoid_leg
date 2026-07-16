"""Stage8 12DOF mapping helpers.

This module is intentionally read-only and does not import CAN or python-can.
It keeps 12DOF CAN order and Isaac DOF order tied through joint-name keyed
dictionaries.
"""

import json
import math
from pathlib import Path


CONFIG_RELATIVE_PATH = Path("config") / "robot_12dof_hardware_map.json"
EXPECTED_JOINT_COUNT = 12
EXPECTED_MOTOR_IDS = tuple(range(1, 13))
EXPECTED_ISAAC_DOF_INDICES = tuple(range(12))

REQUIRED_JOINT_FIELDS = (
    "joint_name",
    "side",
    "body_group",
    "ui_label",
    "motor_id",
    "isaac_joint_name",
    "isaac_dof_aliases",
    "isaac_dof_index",
    "direction",
    "sign",
    "target_limit_deg",
    "hard_limit_deg",
    "kp",
    "kd",
    "enabled",
    "dry_run",
)


def project_root():
    return Path(__file__).resolve().parents[1]


def default_config_path():
    return project_root() / CONFIG_RELATIVE_PATH


def load_12dof_config(path=None):
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_joints(path=None):
    data = load_12dof_config(path)
    joints = data.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("12DOF map must contain a joints object")
    return joints


def _ordered_joint_names_by(field_name, path=None):
    joints = _load_joints(path)
    try:
        return tuple(
            joint_name
            for _value, joint_name in sorted(
                (int(joint[field_name]), joint_name)
                for joint_name, joint in joints.items()
            )
        )
    except KeyError as exc:
        raise ValueError(f"missing {field_name} in 12DOF map") from exc


JOINT_NAMES_12 = _ordered_joint_names_by("motor_id")
CAN_ID_JOINT_ORDER = JOINT_NAMES_12
ISAAC_DOF_JOINT_ORDER = _ordered_joint_names_by("isaac_dof_index")


def _is_plain_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_joint(joint_name, joints=None):
    joints = _load_joints() if joints is None else joints
    if joint_name not in joints:
        raise KeyError(f"unknown 12DOF joint: {joint_name}")
    return joints[joint_name]


def get_joint_names():
    return list(CAN_ID_JOINT_ORDER)


def get_isaac_dof_joint_names():
    return list(ISAAC_DOF_JOINT_ORDER)


def get_joint_config(joint_name):
    return dict(_require_joint(joint_name))


def get_motor_id(joint_name):
    return int(_require_joint(joint_name)["motor_id"])


def get_isaac_dof_index(joint_name):
    return int(_require_joint(joint_name)["isaac_dof_index"])


def get_limits(joint_name):
    joint = _require_joint(joint_name)
    return {
        "target_limit_deg": float(joint["target_limit_deg"]),
        "hard_limit_deg": float(joint["hard_limit_deg"]),
    }


def get_gains(joint_name):
    joint = _require_joint(joint_name)
    return {
        "kp": float(joint["kp"]),
        "kd": float(joint["kd"]),
    }


def get_joint_name_for_motor_id(motor_id):
    motor_id = int(motor_id)
    mapping = get_motor_id_to_joint_name()
    if motor_id not in mapping:
        raise KeyError(f"unknown 12DOF motor_id: {motor_id}")
    return mapping[motor_id]


def get_joint_name_for_isaac_dof_index(isaac_dof_index):
    isaac_dof_index = int(isaac_dof_index)
    mapping = get_isaac_dof_index_to_joint_name()
    if isaac_dof_index not in mapping:
        raise KeyError(f"unknown 12DOF isaac_dof_index: {isaac_dof_index}")
    return mapping[isaac_dof_index]


def get_joint_name_to_motor_id():
    joints = _load_joints()
    return {
        joint_name: int(joints[joint_name]["motor_id"])
        for joint_name in CAN_ID_JOINT_ORDER
    }


def get_motor_id_to_joint_name():
    return {
        motor_id: joint_name
        for joint_name, motor_id in get_joint_name_to_motor_id().items()
    }


def get_joint_name_to_isaac_dof_index():
    joints = _load_joints()
    return {
        joint_name: int(joints[joint_name]["isaac_dof_index"])
        for joint_name in CAN_ID_JOINT_ORDER
    }


def get_isaac_dof_index_to_joint_name():
    return {
        isaac_dof_index: joint_name
        for joint_name, isaac_dof_index in get_joint_name_to_isaac_dof_index().items()
    }


def can_order_differs_from_isaac_order():
    return CAN_ID_JOINT_ORDER != ISAAC_DOF_JOINT_ORDER


def validate_12dof_mapping(path=None):
    data = load_12dof_config(path)
    joints = data.get("joints")
    if not isinstance(joints, dict):
        return False, "joints must be an object"
    if len(joints) != EXPECTED_JOINT_COUNT:
        return False, "joint count must be 12"
    expected_can_order = _ordered_joint_names_by("motor_id", path)
    expected_isaac_order = _ordered_joint_names_by("isaac_dof_index", path)
    if set(expected_can_order) != set(expected_isaac_order):
        return False, "CAN order and Isaac order contain different joints"

    motor_ids = []
    isaac_indices = []
    for joint_name in expected_can_order:
        joint = joints[joint_name]
        if not isinstance(joint, dict):
            return False, f"joint config must be an object: {joint_name}"
        missing = [field for field in REQUIRED_JOINT_FIELDS if field not in joint]
        if missing:
            return False, f"missing fields for {joint_name}: {', '.join(missing)}"
        if joint["joint_name"] != joint_name:
            return False, f"joint_name mismatch: {joint_name}"
        if joint["isaac_joint_name"] != joint_name:
            return False, f"isaac_joint_name mismatch: {joint_name}"
        if joint["side"] not in ("left", "right"):
            return False, f"side invalid: {joint_name}"
        if not isinstance(joint["body_group"], str) or not joint["body_group"]:
            return False, f"body_group invalid: {joint_name}"
        if not isinstance(joint["ui_label"], str) or not joint["ui_label"]:
            return False, f"ui_label invalid: {joint_name}"
        aliases = joint["isaac_dof_aliases"]
        if not isinstance(aliases, list) or not aliases:
            return False, f"isaac_dof_aliases invalid: {joint_name}"
        if joint["isaac_joint_name"] not in aliases:
            return False, f"isaac_dof_aliases must include isaac_joint_name: {joint_name}"
        if not all(isinstance(alias, str) and alias for alias in aliases):
            return False, f"isaac_dof_aliases must be non-empty strings: {joint_name}"
        if not isinstance(joint["enabled"], bool):
            return False, f"enabled must be bool: {joint_name}"
        if not isinstance(joint["dry_run"], bool):
            return False, f"dry_run must be bool: {joint_name}"

        for field in ("motor_id", "isaac_dof_index", "direction", "sign"):
            value = joint[field]
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"{field} must be an integer: {joint_name}"
        if joint["direction"] not in (-1, 1) or joint["sign"] not in (-1, 1):
            return False, f"direction/sign invalid: {joint_name}"

        for field in ("target_limit_deg", "hard_limit_deg", "kp", "kd"):
            value = joint[field]
            if not _is_plain_number(value) or not math.isfinite(float(value)):
                return False, f"{field} must be finite: {joint_name}"
        if float(joint["target_limit_deg"]) <= 0.0:
            return False, f"target_limit_deg must be positive: {joint_name}"
        if float(joint["hard_limit_deg"]) < float(joint["target_limit_deg"]):
            return False, f"hard limit smaller than target limit: {joint_name}"

        motor_ids.append(int(joint["motor_id"]))
        isaac_indices.append(int(joint["isaac_dof_index"]))

    if tuple(sorted(motor_ids)) != EXPECTED_MOTOR_IDS:
        return False, "motor_id set must be 1..12"
    if tuple(sorted(isaac_indices)) != EXPECTED_ISAAC_DOF_INDICES:
        return False, "isaac_dof_index set must be 0..11"

    motor_order = tuple(
        joint_name
        for _motor_id, joint_name in sorted(
            (int(joint["motor_id"]), joint_name)
            for joint_name, joint in joints.items()
        )
    )
    isaac_order = tuple(
        joint_name
        for _dof_index, joint_name in sorted(
            (int(joint["isaac_dof_index"]), joint_name)
            for joint_name, joint in joints.items()
        )
    )
    if motor_order != expected_can_order:
        return False, "motor_id order is not stable"
    if isaac_order != expected_isaac_order:
        return False, "isaac_dof_index order is not stable"

    return True, ""


JOINT_NAMES_12DOF = JOINT_NAMES_12
JOINT_NAME_TO_MOTOR_ID = get_joint_name_to_motor_id()
MOTOR_ID_TO_JOINT_NAME = get_motor_id_to_joint_name()
JOINT_NAME_TO_ISAAC_DOF_INDEX = get_joint_name_to_isaac_dof_index()
ISAAC_DOF_INDEX_TO_JOINT_NAME = get_isaac_dof_index_to_joint_name()

# Stage8/Stage9 compatibility aliases. Keep these dicts joint-name keyed so
# CAN ID order and Isaac DOF index order cannot be accidentally conflated.
CAN_ID_BY_JOINT = JOINT_NAME_TO_MOTOR_ID
JOINT_BY_CAN_ID = MOTOR_ID_TO_JOINT_NAME
ISAAC_DOF_INDEX_BY_JOINT = JOINT_NAME_TO_ISAAC_DOF_INDEX
JOINT_BY_ISAAC_DOF_INDEX = ISAAC_DOF_INDEX_TO_JOINT_NAME
DEFAULT_LIMIT_GAIN_BY_JOINT = {
    joint_name: {
        "target_limit_deg": float(get_joint_config(joint_name)["target_limit_deg"]),
        "hard_limit_deg": float(get_joint_config(joint_name)["hard_limit_deg"]),
        "kp": float(get_joint_config(joint_name)["kp"]),
        "kd": float(get_joint_config(joint_name)["kd"]),
    }
    for joint_name in JOINT_NAMES_12DOF
}
