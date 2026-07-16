"""RobotModel 계층.

12DOF 하체 정의를 config/mapping에서 읽어 UI, Safety, Isaac, REAL adapter가
같은 joint spec을 보게 한다. 추후 상체가 추가되어도 joint spec만 늘리면
동일한 조회 API를 계속 사용할 수 있다.
"""

from dataclasses import asdict, dataclass

from robot_runtime.dof12_mapping import (
    CAN_ID_JOINT_ORDER,
    load_12dof_config,
    validate_12dof_mapping,
)


def _infer_body_group(joint_name, side):
    if side in ("left", "right") and any(part in joint_name for part in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    if side in ("left", "right"):
        return f"{side}_body"
    return "body"


@dataclass(frozen=True)
class JointSpec:
    joint_name: str
    body_group: str
    side: str
    can_id: int
    isaac_joint_name: str
    isaac_dof_aliases: tuple
    isaac_dof_index: int
    direction: int
    sign: int
    target_limit_deg: float
    hard_limit_deg: float
    kp: float
    kd: float
    enabled: bool
    dry_run: bool
    ui_label: str

    @classmethod
    def from_config(cls, joint_name, config):
        side = str(config.get("side", ""))
        aliases = tuple(str(alias) for alias in config.get("isaac_dof_aliases", (config.get("isaac_joint_name") or joint_name,)))
        return cls(
            joint_name=joint_name,
            body_group=str(config.get("body_group") or _infer_body_group(joint_name, side)),
            side=side,
            can_id=int(config["motor_id"]),
            isaac_joint_name=str(config.get("isaac_joint_name") or joint_name),
            isaac_dof_aliases=aliases,
            isaac_dof_index=int(config["isaac_dof_index"]),
            direction=int(config.get("direction", 1)),
            sign=int(config.get("sign", 1)),
            target_limit_deg=float(config["target_limit_deg"]),
            hard_limit_deg=float(config["hard_limit_deg"]),
            kp=float(config["kp"]),
            kd=float(config["kd"]),
            enabled=bool(config.get("enabled", False)),
            dry_run=bool(config.get("dry_run", True)),
            ui_label=str(config.get("ui_label") or joint_name),
        )

    def target_min_deg(self):
        return -abs(self.target_limit_deg)

    def target_max_deg(self):
        return abs(self.target_limit_deg)

    def to_dict(self):
        return asdict(self)

    def to_legacy_config(self):
        data = self.to_dict()
        data["motor_id"] = self.can_id
        return data


class RobotModel:
    """로봇 자유도 정의의 단일 조회 지점."""

    def __init__(self, name, version, joint_specs, raw_config=None):
        self.name = str(name)
        self.version = str(version)
        self._joint_specs = tuple(joint_specs)
        self._by_name = {spec.joint_name: spec for spec in self._joint_specs}
        self._by_can_id = {spec.can_id: spec for spec in self._joint_specs}
        self._by_isaac_dof_index = {spec.isaac_dof_index: spec for spec in self._joint_specs}
        self.raw_config = dict(raw_config or {})
        self._validate_unique_keys()

    @classmethod
    def load_12dof(cls, path=None):
        valid, reason = validate_12dof_mapping(path)
        if not valid:
            raise ValueError(f"invalid 12DOF mapping: {reason}")
        data = load_12dof_config(path)
        joints = data["joints"]
        specs = [
            JointSpec.from_config(joint_name, joints[joint_name])
            for joint_name in CAN_ID_JOINT_ORDER
        ]
        return cls(
            name=data.get("robot_name", "humanoid_12dof_lower_body"),
            version=data.get("version", "12dof"),
            joint_specs=specs,
            raw_config=data,
        )

    @property
    def joint_specs(self):
        return self._joint_specs

    @property
    def joint_names(self):
        return tuple(spec.joint_name for spec in self._joint_specs)

    @property
    def joint_count(self):
        return len(self._joint_specs)

    @property
    def body_groups(self):
        return tuple(dict.fromkeys(spec.body_group for spec in self._joint_specs))

    def joints_by_group(self, body_group):
        return tuple(spec for spec in self._joint_specs if spec.body_group == body_group)

    def has_joint(self, joint_name):
        return joint_name in self._by_name

    def get_joint(self, joint_name):
        try:
            return self._by_name[joint_name]
        except KeyError as exc:
            raise KeyError(f"unknown joint: {joint_name}") from exc

    def get_by_can_id(self, can_id):
        try:
            return self._by_can_id[int(can_id)]
        except KeyError as exc:
            raise KeyError(f"unknown CAN ID: {can_id}") from exc

    def get_by_isaac_dof_index(self, dof_index):
        try:
            return self._by_isaac_dof_index[int(dof_index)]
        except KeyError as exc:
            raise KeyError(f"unknown Isaac DOF index: {dof_index}") from exc

    def can_id_by_joint(self):
        return {spec.joint_name: spec.can_id for spec in self._joint_specs}

    def joint_by_can_id(self):
        return {spec.can_id: spec.joint_name for spec in self._joint_specs}

    def isaac_dof_index_by_joint(self):
        return {spec.joint_name: spec.isaac_dof_index for spec in self._joint_specs}

    def joint_by_isaac_dof_index(self):
        return {spec.isaac_dof_index: spec.joint_name for spec in self._joint_specs}

    def limits_for(self, joint_name):
        spec = self.get_joint(joint_name)
        return {
            "min": spec.target_min_deg(),
            "max": spec.target_max_deg(),
            "target_limit_deg": spec.target_limit_deg,
            "hard_limit_deg": spec.hard_limit_deg,
        }

    def config_for_joint(self, joint_name):
        return self.get_joint(joint_name).to_legacy_config()

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "joint_count": self.joint_count,
            "joint_names": list(self.joint_names),
            "body_groups": list(self.body_groups),
            "joints": {spec.joint_name: spec.to_dict() for spec in self._joint_specs},
        }

    def _validate_unique_keys(self):
        if len(self._by_name) != len(self._joint_specs):
            raise ValueError("duplicate joint_name in RobotModel")
        if len(self._by_can_id) != len(self._joint_specs):
            raise ValueError("duplicate CAN ID in RobotModel")
        if len(self._by_isaac_dof_index) != len(self._joint_specs):
            raise ValueError("duplicate Isaac DOF index in RobotModel")
