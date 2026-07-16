"""Stage9 12DOF SIM-only command buffer.

The UI writes joint-name keyed dry commands here. Isaac-side scripts can read
the JSON file and apply targets in simulation. This module does not use ROS,
CAN, Jetson helpers, or real motor control code.
"""

import json
import math
from pathlib import Path

from robot_runtime.dof_mapping import DofMapping
from robot_runtime.robot_model import RobotModel
from sim_interface.sim_command_payload import build_sim_joint_payload


SIM_12DOF_COMMAND_FILE_PATH = Path("/tmp/humanoid_12dof_sim_command.json")
SIM_12DOF_TARGET_LIMIT_CAP_DEG = 180.0
SIM_12DOF_COMMAND_SOURCE = "stage9_ui_sim_12dof"
SIM_12DOF_COMMAND_KIND = "SET_12DOF_SIM_TARGETS"
ROBOT_MODEL = RobotModel.load_12dof()
DOF_MAPPING = DofMapping(ROBOT_MODEL)
JOINT_NAMES_12DOF = ROBOT_MODEL.joint_names


def get_sim_target_limit_deg(joint_name):
    configured_limit = float(ROBOT_MODEL.get_joint(joint_name).target_limit_deg)
    return min(SIM_12DOF_TARGET_LIMIT_CAP_DEG, configured_limit)


SIM_TARGET_LIMIT_DEG_BY_JOINT = {
    joint_name: get_sim_target_limit_deg(joint_name)
    for joint_name in JOINT_NAMES_12DOF
}


def clamp_sim_target_deg(joint_name, target_deg):
    if joint_name not in JOINT_NAMES_12DOF:
        raise KeyError(f"unknown 12DOF joint: {joint_name}")
    if isinstance(target_deg, bool) or not isinstance(target_deg, (int, float)):
        raise ValueError("target_deg must be numeric")
    target_deg = float(target_deg)
    if not math.isfinite(target_deg):
        raise ValueError("target_deg must be finite")
    limit = SIM_TARGET_LIMIT_DEG_BY_JOINT[joint_name]
    return max(-limit, min(limit, target_deg))


def build_12dof_sim_command(joint_targets_deg):
    if not isinstance(joint_targets_deg, dict):
        raise ValueError("joint_targets_deg must be a dict")

    targets = {
        joint_name: clamp_sim_target_deg(
            joint_name,
            joint_targets_deg.get(joint_name, 0.0),
        )
        for joint_name in JOINT_NAMES_12DOF
    }
    bridge_payload = build_sim_joint_payload(
        mode="SIM",
        joint_commands=targets,
        source=SIM_12DOF_COMMAND_SOURCE,
    )
    return {
        "timestamp": bridge_payload["timestamp"],
        "source": bridge_payload["source"],
        "mode": bridge_payload["mode"],
        "target_unit": bridge_payload["target_unit"],
        "command": SIM_12DOF_COMMAND_KIND,
        "robot_prim_path": "/World/URDF12DOF",
        "articulation_root": "/World/URDF12DOF",
        "joint_count": len(JOINT_NAMES_12DOF),
        "joint_commands": bridge_payload["joint_commands"],
        "joint_targets_deg": targets,
        "isaac_dof_index_by_joint": DOF_MAPPING.snapshot()["isaac_dof_index_by_joint"],
        "sim_target_limit_deg_by_joint": dict(SIM_TARGET_LIMIT_DEG_BY_JOINT),
        "mapping_policy": "joint_name_dict_only",
    }


def write_12dof_sim_command(joint_targets_deg, path=SIM_12DOF_COMMAND_FILE_PATH):
    payload = build_12dof_sim_command(joint_targets_deg)
    command_path = Path(path)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = command_path.with_suffix(command_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(command_path)
    return payload


def read_12dof_sim_command(path=SIM_12DOF_COMMAND_FILE_PATH):
    command_path = Path(path)
    with command_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("12DOF SIM command payload must be an object")
    return payload


class Stage9Sim12DofCommandBuffer:
    def __init__(self, path=SIM_12DOF_COMMAND_FILE_PATH):
        self.path = Path(path)
        self.joint_targets_deg = {joint_name: 0.0 for joint_name in JOINT_NAMES_12DOF}

    def set_joint_target(self, joint_name, target_deg):
        self.joint_targets_deg[joint_name] = clamp_sim_target_deg(joint_name, target_deg)
        return self.write()

    def write(self):
        return write_12dof_sim_command(self.joint_targets_deg, self.path)

    def snapshot(self):
        return {
            "path": str(self.path),
            "joint_targets_deg": dict(self.joint_targets_deg),
            "sim_target_limit_deg_by_joint": dict(SIM_TARGET_LIMIT_DEG_BY_JOINT),
        }
