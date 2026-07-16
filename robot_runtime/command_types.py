"""Runtime command dataclasses.

기존 control_logic.command_models의 JointCommand/FeedbackSample/BridgeStatus는
그대로 재사용하고, mode/safety/robot command 타입만 추가한다.
"""

from dataclasses import asdict, dataclass
import time

from control_logic.command_models import BridgeStatus, FeedbackSample, JointCommand
from robot_runtime.mode_types import normalize_mode


def _timestamp(value=None):
    return time.time() if value is None else float(value)


@dataclass(frozen=True)
class ModeCommand:
    mode: str
    source: str = "ui"
    timestamp: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "mode", normalize_mode(self.mode))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp or None))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RobotCommand:
    command: str
    source: str = "ui"
    timestamp: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "command", str(self.command))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp or None))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SafetyCommand:
    command: str
    source: str = "ui"
    timestamp: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "command", str(self.command))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp or None))

    def to_dict(self):
        return asdict(self)
