import json
from pathlib import Path
from pprint import pprint

from robot_runtime.dof12_mapping import JOINT_NAMES_12, get_joint_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "robot_config.json"

REQUIRED_CONFIG_KEYS = (
    "project_name",
    "default_mode",
    "available_modes",
    "target_status_values",
    "motor_status_values",
    "safety_status_values",
    "joint_names",
    "joint_limits_deg",
    "ak_motor_count",
    "ak_motor_status_values",
    "command_rate_limit",
    "safety_rules",
)


def load_robot_config(config_path=None):
    """로봇 공통 설정 JSON 파일을 읽고 검증한 뒤 딕셔너리로 반환한다."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    # 설정 파일 존재 여부를 먼저 확인한다.
    if not path.exists():
        raise FileNotFoundError(f"Robot config file not found: {path}")

    # JSON 문법 오류는 파일 경로를 포함한 ValueError로 변환한다.
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in config file {path}: {error}") from error

    _apply_12dof_hardware_map(config)
    _validate_config(config)
    return config


def _apply_12dof_hardware_map(config):
    """하드웨어 joint 정의는 robot_12dof_hardware_map.json을 단일 원본으로 쓴다.

    robot_config.json의 joint_names/joint_limits_deg/ak_motor_count는 기존 설정
    파일 형식 호환을 위해 남겨두지만, 런타임에서는 hardware map에서 파생한 값으로
    덮어쓴다.
    """
    config["joint_names"] = list(JOINT_NAMES_12)
    config["ak_motor_count"] = len(JOINT_NAMES_12)
    config["joint_limits_deg"] = {}
    for joint_name in JOINT_NAMES_12:
        joint_config = get_joint_config(joint_name)
        target_limit = float(joint_config["target_limit_deg"])
        config["joint_limits_deg"][joint_name] = {
            "min": -target_limit,
            "max": target_limit,
        }


def _validate_config(config):
    """설정 딕셔너리의 필수 구조와 기본 값을 검증한다."""
    if not isinstance(config, dict):
        raise ValueError("Robot config must be a JSON object")

    # 필수 최상위 키가 모두 있는지 확인한다.
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    _validate_default_mode(config)
    _validate_joint_limits(config)
    _validate_12dof_joint_names(config)
    _validate_ak_motor_count(config)


def _validate_default_mode(config):
    """default_mode가 available_modes 목록에 포함되는지 검사한다."""
    available_modes = config["available_modes"]
    default_mode = config["default_mode"]

    if not isinstance(available_modes, list):
        raise ValueError("available_modes must be a list")

    if default_mode not in available_modes:
        raise ValueError("default_mode must be included in available_modes")


def _validate_joint_limits(config):
    """joint_names와 joint_limits_deg의 일관성을 검사한다."""
    joint_names = config["joint_names"]
    joint_limits = config["joint_limits_deg"]

    if not isinstance(joint_names, list):
        raise ValueError("joint_names must be a list")

    if not isinstance(joint_limits, dict):
        raise ValueError("joint_limits_deg must be an object")

    for joint_name in joint_names:
        if joint_name not in joint_limits:
            raise ValueError(f"Missing joint limit for {joint_name}")

        limit = joint_limits[joint_name]
        if not isinstance(limit, dict):
            raise ValueError(f"Invalid joint limit for {joint_name}: must be an object")

        if "min" not in limit:
            raise ValueError(f"Missing min joint limit for {joint_name}")

        if "max" not in limit:
            raise ValueError(f"Missing max joint limit for {joint_name}")

        min_value = limit["min"]
        max_value = limit["max"]

        if min_value >= max_value:
            raise ValueError(
                f"Invalid joint limit for {joint_name}: min must be smaller than max"
            )


def _validate_12dof_joint_names(config):
    """control_logic 설정은 항상 robot_runtime.dof12_mapping의 12축 순서를 따른다."""
    joint_names = tuple(config["joint_names"])
    if joint_names != tuple(JOINT_NAMES_12):
        raise ValueError("joint_names must match JOINT_NAMES_12 from dof12_mapping")


def _validate_ak_motor_count(config):
    """AK 모터 개수가 1 이상의 정수인지 검사한다."""
    ak_motor_count = config["ak_motor_count"]

    if not isinstance(ak_motor_count, int) or ak_motor_count != len(JOINT_NAMES_12):
        raise ValueError("ak_motor_count must match 12DOF joint count")


if __name__ == "__main__":
    robot_config = load_robot_config()
    pprint(robot_config)
