"""Shared command/feedback data models for dry-run bridge validation.

이 파일은 순수 Python dataclass만 제공한다. Isaac, ROS, CAN, AK, Jetson 코드를
import하지 않으므로 UI/SIM/REAL dry-run 테스트에서 안전하게 사용할 수 있다.
"""

from dataclasses import asdict, dataclass
import math
import time

from robot_runtime.dof12_mapping import JOINT_NAMES_12


VALID_COMMAND_MODES = ("SIM", "REAL", "SIM_TO_REAL", "REAL_TO_SIM")
VALID_FEEDBACK_SOURCES = ("sim", "real")


def _now_if_none(timestamp):
    return time.time() if timestamp is None else float(timestamp)


def _require_finite_number(field_name, value, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _require_joint_name(joint_name):
    if joint_name not in JOINT_NAMES_12:
        raise ValueError(f"unknown 12DOF joint: {joint_name}")
    return str(joint_name)


def _require_source(source):
    if source not in VALID_FEEDBACK_SOURCES:
        raise ValueError("feedback source must be sim or real")
    return str(source)


@dataclass(frozen=True)
class JointCommand:
    joint_name: str
    target_deg: float
    source: str
    mode: str
    timestamp: float

    def __post_init__(self):
        object.__setattr__(self, "joint_name", _require_joint_name(self.joint_name))
        object.__setattr__(self, "target_deg", _require_finite_number("target_deg", self.target_deg))
        object.__setattr__(self, "source", str(self.source))
        if self.mode not in VALID_COMMAND_MODES:
            raise ValueError(f"invalid command mode: {self.mode}")
        object.__setattr__(self, "timestamp", _require_finite_number("timestamp", self.timestamp))

    @classmethod
    def create(cls, joint_name, target_deg, source="ui", mode="SIM", timestamp=None):
        return cls(
            joint_name=joint_name,
            target_deg=target_deg,
            source=source,
            mode=mode,
            timestamp=_now_if_none(timestamp),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FeedbackSample:
    joint_name: str
    position_deg: float | None
    velocity_dps: float | None
    torque_nm: float | None
    source: str
    timestamp: float
    status: str

    def __post_init__(self):
        object.__setattr__(self, "joint_name", _require_joint_name(self.joint_name))
        object.__setattr__(
            self,
            "position_deg",
            _require_finite_number("position_deg", self.position_deg, allow_none=True),
        )
        object.__setattr__(
            self,
            "velocity_dps",
            _require_finite_number("velocity_dps", self.velocity_dps, allow_none=True),
        )
        object.__setattr__(
            self,
            "torque_nm",
            _require_finite_number("torque_nm", self.torque_nm, allow_none=True),
        )
        object.__setattr__(self, "source", _require_source(self.source))
        object.__setattr__(self, "timestamp", _require_finite_number("timestamp", self.timestamp))
        object.__setattr__(self, "status", str(self.status or "UNKNOWN"))

    @classmethod
    def create(
        cls,
        joint_name,
        position_deg=None,
        velocity_dps=None,
        torque_nm=None,
        source="sim",
        timestamp=None,
        status="OK",
    ):
        return cls(
            joint_name=joint_name,
            position_deg=position_deg,
            velocity_dps=velocity_dps,
            torque_nm=torque_nm,
            source=source,
            timestamp=_now_if_none(timestamp),
            status=status,
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BridgeStatus:
    source: str
    connected: bool
    last_rx_time: float | None
    last_tx_time: float | None
    last_error: str | None

    def __post_init__(self):
        object.__setattr__(self, "source", _require_source(self.source))
        object.__setattr__(self, "connected", bool(self.connected))
        object.__setattr__(
            self,
            "last_rx_time",
            _require_finite_number("last_rx_time", self.last_rx_time, allow_none=True),
        )
        object.__setattr__(
            self,
            "last_tx_time",
            _require_finite_number("last_tx_time", self.last_tx_time, allow_none=True),
        )
        last_error = None if self.last_error is None else str(self.last_error)
        object.__setattr__(self, "last_error", last_error)

    @classmethod
    def create(
        cls,
        source,
        connected=False,
        last_rx_time=None,
        last_tx_time=None,
        last_error=None,
    ):
        return cls(
            source=source,
            connected=connected,
            last_rx_time=last_rx_time,
            last_tx_time=last_tx_time,
            last_error=last_error,
        )

    def to_dict(self):
        return asdict(self)
