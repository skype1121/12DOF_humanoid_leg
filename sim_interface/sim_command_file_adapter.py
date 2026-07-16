"""Safe adapter for writing SIM command files after command_bus approval.

UI code should not instantiate Stage9Sim12DofCommandBuffer directly. The UI first
routes commands through command_bus/safety_filter/robot_state, then calls this
adapter to preserve the existing Stage9 SIM command file workflow.
"""

from robot_runtime.dof12_mapping import JOINT_NAMES_12DOF
from sim_interface.isaac_12dof_command_buffer import (
    SIM_12DOF_COMMAND_FILE_PATH,
    Stage9Sim12DofCommandBuffer,
)


class Stage9SimCommandFileAdapter:
    """command_bus 뒤쪽에서만 SIM command file을 쓰는 adapter."""

    def __init__(self, robot_state, path=SIM_12DOF_COMMAND_FILE_PATH):
        self.robot_state = robot_state
        self.command_buffer = Stage9Sim12DofCommandBuffer(path)

    @property
    def path(self):
        return self.command_buffer.path

    def can_write_for_current_mode(self):
        return self.robot_state.get_mode() in ("SIM", "SIM_TO_REAL", "REAL_TO_SIM")

    def write_joint_command(self, joint_name, target_deg):
        """command_bus가 허용한 단일 joint 목표를 SIM command file에 반영한다."""
        if self.robot_state.get_mode() not in ("SIM", "SIM_TO_REAL"):
            raise RuntimeError("SIM command file write is allowed only in SIM/SIM_TO_REAL")
        return self.command_buffer.set_joint_target(joint_name, target_deg)

    def write_home_targets(self):
        """command_bus home 요청 이후 12개 관절 목표를 0 deg로 기록한다."""
        if self.robot_state.get_mode() not in ("SIM", "SIM_TO_REAL"):
            raise RuntimeError("SIM home write is allowed only in SIM/SIM_TO_REAL")
        targets = {joint_name: 0.0 for joint_name in JOINT_NAMES_12DOF}
        return self.write_targets(targets)

    def write_targets_from_feedback(self, targets_deg):
        """REAL_TO_SIM feedback mirror를 SIM command file에 반영한다.

        이 경로는 실제 write가 아니라 read-only feedback mirror다.
        """
        if self.robot_state.get_mode() != "REAL_TO_SIM":
            raise RuntimeError("feedback mirror write is allowed only in REAL_TO_SIM")
        return self.write_targets(targets_deg)

    def write_targets(self, targets_deg):
        if not isinstance(targets_deg, dict):
            raise ValueError("targets_deg must be dict")
        for joint_name, target_deg in targets_deg.items():
            self.command_buffer.joint_targets_deg[joint_name] = target_deg
        return self.command_buffer.write()

    def snapshot(self):
        return self.command_buffer.snapshot()
