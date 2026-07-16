"""CommandBuffer는 안전 검사를 통과한 명령만 저장한다."""

from dataclasses import asdict, dataclass
import time

from control_logic.command_models import JointCommand
from robot_runtime.command_types import ModeCommand, RobotCommand, SafetyCommand
from robot_runtime.robot_model import RobotModel


@dataclass(frozen=True)
class RuntimeSnapshot:
    mode: str
    connection_state: str
    motor_enable_state: str
    estop_active: bool
    joint_targets: dict
    joint_feedback: dict
    timestamp: float

    def to_dict(self):
        return asdict(self)


class CommandBuffer:
    """CommandBus 뒤쪽에서 mode adapter가 읽을 명령 snapshot을 보관한다."""

    def __init__(self, robot_model=None):
        self.robot_model = robot_model or RobotModel.load_12dof()
        self.latest_joint_targets_deg = {
            joint_name: 0.0 for joint_name in self.robot_model.joint_names
        }
        self.latest_joint_commands = {}
        self.latest_mode_command = None
        self.latest_robot_command = None
        self.latest_safety_command = None
        self.command_log = []
        self.last_update_time = None

    def push_joint_command(self, command):
        if not isinstance(command, JointCommand):
            raise TypeError("command must be JointCommand")
        self.robot_model.get_joint(command.joint_name)
        self.latest_joint_targets_deg[command.joint_name] = float(command.target_deg)
        self.latest_joint_commands[command.joint_name] = command.to_dict()
        self._record("joint_command", command.to_dict())
        return command

    def push_mode_command(self, mode, source="command_bus"):
        command = ModeCommand(mode=mode, source=source)
        self.latest_mode_command = command.to_dict()
        self._record("mode_command", self.latest_mode_command)
        return command

    def push_robot_command(self, command_name, source="command_bus"):
        command = RobotCommand(command=command_name, source=source)
        self.latest_robot_command = command.to_dict()
        self._record("robot_command", self.latest_robot_command)
        return command

    def push_safety_command(self, command_name, source="command_bus"):
        command = SafetyCommand(command=command_name, source=source)
        self.latest_safety_command = command.to_dict()
        self._record("safety_command", self.latest_safety_command)
        return command

    def get_latest_joint_targets(self):
        return {
            joint_name: float(self.latest_joint_targets_deg[joint_name])
            for joint_name in self.robot_model.joint_names
        }

    def snapshot(self):
        return {
            "joint_names": list(self.robot_model.joint_names),
            "joint_targets_deg": self.get_latest_joint_targets(),
            "latest_joint_commands": {
                joint_name: dict(command)
                for joint_name, command in self.latest_joint_commands.items()
            },
            "latest_mode_command": (
                None if self.latest_mode_command is None else dict(self.latest_mode_command)
            ),
            "latest_robot_command": (
                None if self.latest_robot_command is None else dict(self.latest_robot_command)
            ),
            "latest_safety_command": (
                None if self.latest_safety_command is None else dict(self.latest_safety_command)
            ),
            "last_update_time": self.last_update_time,
            "command_log": [dict(item) for item in self.command_log],
        }

    def build_runtime_snapshot(self, robot_state, feedback_snapshot=None):
        feedback_snapshot = feedback_snapshot or {}
        return RuntimeSnapshot(
            mode=robot_state.get_mode(),
            connection_state=robot_state.get_target_status(),
            motor_enable_state=robot_state.get_motor_status(),
            estop_active=robot_state.estop_active,
            joint_targets=self.get_latest_joint_targets(),
            joint_feedback=dict(feedback_snapshot),
            timestamp=time.time(),
        )

    def _record(self, command_type, payload):
        self.last_update_time = time.time()
        self.command_log.append(
            {
                "command_type": command_type,
                "payload": dict(payload),
                "timestamp": self.last_update_time,
            }
        )
