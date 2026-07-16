from pprint import pprint

from robot_runtime.dof12_mapping import JOINT_NAMES_12
from robot_runtime.robot_model import RobotModel


class RobotState:
    """UI, 시뮬레이터, 실제 로봇 제어 코드가 공유할 로봇 상태 저장 클래스."""

    VALID_MODES = ("SIM", "REAL", "SIM_TO_REAL", "REAL_TO_SIM")
    VALID_TARGET_STATUSES = ("Disconnected", "Connecting", "Connected")
    VALID_MOTOR_STATUSES = ("Disabled", "Enabled")
    VALID_SAFETY_STATUSES = ("OK", "E-STOP ACTIVE")
    VALID_AK_MOTOR_STATUSES = ("Disconnected", "Connecting", "Connected")

    JOINT_NAMES = JOINT_NAMES_12

    AK_MOTOR_IDS = tuple(range(1, 13))

    def __init__(self, robot_model=None):
        self.robot_model = robot_model or RobotModel.load_12dof()
        self.JOINT_NAMES = self.robot_model.joint_names
        self.AK_MOTOR_IDS = tuple(
            sorted(spec.can_id for spec in self.robot_model.joint_specs)
        )

        # 전체 로봇 운용 상태를 저장한다.
        self.current_mode = "SIM"
        self.target_status = "Disconnected"
        self.motor_status = "Disabled"
        self.safety_status = "OK"

        # 각 관절의 현재 목표/상태 각도를 degree 단위로 저장한다.
        self.joint_angles = {joint_name: 0.0 for joint_name in self.JOINT_NAMES}

        # AK Motor 1~12의 연결 상태를 저장한다.
        self.ak_motor_status = {
            motor_id: "Disconnected" for motor_id in self.AK_MOTOR_IDS
        }

        # UI/브리지/테스트에서 상태 원인을 확인하기 위한 최근 상태값이다.
        self.last_command_source = "init"
        self.latest_joint_commands = {
            joint_name: 0.0 for joint_name in self.JOINT_NAMES
        }
        self.latest_motor_feedback = {}
        self.latest_sim_feedback = {}
        self.last_sim_rx_time = None
        self.last_real_rx_time = None
        self.sim_connected = False
        self.real_connected = False
        self.sim_feedback_stale = True
        self.real_feedback_stale = True
        self.bridge_status = {
            "sim": {
                "source": "sim",
                "connected": False,
                "last_rx_time": None,
                "last_tx_time": None,
                "last_error": None,
            },
            "real": {
                "source": "real",
                "connected": False,
                "last_rx_time": None,
                "last_tx_time": None,
                "last_error": None,
            },
        }
        self.last_error = ""

    def set_mode(self, mode):
        """현재 운용 모드를 설정한다."""
        self._validate_value("mode", mode, self.VALID_MODES)
        self.current_mode = mode

    def get_mode(self):
        """현재 운용 모드를 반환한다."""
        return self.current_mode

    def set_target_status(self, status):
        """시뮬레이터/실제 로봇 타깃 연결 상태를 설정한다."""
        self._validate_value("target_status", status, self.VALID_TARGET_STATUSES)
        self.target_status = status

    def get_target_status(self):
        """시뮬레이터/실제 로봇 타깃 연결 상태를 반환한다."""
        return self.target_status

    def set_motor_status(self, status):
        """모터 활성화 상태를 설정한다."""
        self._validate_value("motor_status", status, self.VALID_MOTOR_STATUSES)
        self.motor_status = status

    def get_motor_status(self):
        """모터 활성화 상태를 반환한다."""
        return self.motor_status

    def set_safety_status(self, status):
        """안전 상태를 설정한다."""
        self._validate_value("safety_status", status, self.VALID_SAFETY_STATUSES)
        self.safety_status = status

    def get_safety_status(self):
        """안전 상태를 반환한다."""
        return self.safety_status

    def set_joint_angle(self, joint_name, angle_deg):
        """지정한 관절의 각도를 degree 단위로 저장한다."""
        self._validate_joint_name(joint_name)
        angle_deg = float(angle_deg)
        self.joint_angles[joint_name] = angle_deg
        self.latest_joint_commands[joint_name] = angle_deg

    def get_joint_angle(self, joint_name):
        """지정한 관절의 각도를 반환한다."""
        self._validate_joint_name(joint_name)
        return self.joint_angles[joint_name]

    def get_all_joint_angles(self):
        """모든 관절 각도를 복사본으로 반환한다."""
        return dict(self.joint_angles)

    @property
    def target_connected(self):
        return self.target_status == "Connected"

    @property
    def motor_enabled(self):
        return self.motor_status == "Enabled"

    @property
    def estop_active(self):
        return self.safety_status == "E-STOP ACTIVE"

    def set_ak_motor_status(self, motor_id, status):
        """지정한 AK 모터의 연결 상태를 설정한다."""
        self._validate_motor_id(motor_id)
        self._validate_value("ak_motor_status", status, self.VALID_AK_MOTOR_STATUSES)
        self.ak_motor_status[motor_id] = status

    def get_ak_motor_status(self, motor_id):
        """지정한 AK 모터의 연결 상태를 반환한다."""
        self._validate_motor_id(motor_id)
        return self.ak_motor_status[motor_id]

    def get_all_ak_motor_status(self):
        """모든 AK 모터 연결 상태를 복사본으로 반환한다."""
        return dict(self.ak_motor_status)

    def reset_joint_angles(self):
        """모든 관절 각도를 0도로 초기화한다."""
        for joint_name in self.JOINT_NAMES:
            self.joint_angles[joint_name] = 0.0
            self.latest_joint_commands[joint_name] = 0.0

    def reset_all_ak_motors(self):
        """모든 AK 모터 상태를 Disconnected로 초기화한다."""
        for motor_id in self.AK_MOTOR_IDS:
            self.ak_motor_status[motor_id] = "Disconnected"

    def get_snapshot(self):
        """디버깅용으로 현재 전체 상태를 딕셔너리로 반환한다."""
        return {
            "current_mode": self.current_mode,
            "target_connected": self.target_connected,
            "motor_enabled": self.motor_enabled,
            "estop_active": self.estop_active,
            "target_status": self.target_status,
            "motor_status": self.motor_status,
            "safety_status": self.safety_status,
            "joint_angles": self.get_all_joint_angles(),
            "ak_motor_status": self.get_all_ak_motor_status(),
            "last_command_source": self.last_command_source,
            "latest_joint_commands": dict(self.latest_joint_commands),
            "latest_motor_feedback": dict(self.latest_motor_feedback),
            "latest_sim_feedback": dict(self.latest_sim_feedback),
            "last_sim_rx_time": self.last_sim_rx_time,
            "last_real_rx_time": self.last_real_rx_time,
            "sim_connected": self.sim_connected,
            "real_connected": self.real_connected,
            "sim_feedback_stale": self.sim_feedback_stale,
            "real_feedback_stale": self.real_feedback_stale,
            "bridge_status": {
                "sim": dict(self.bridge_status.get("sim", {})),
                "real": dict(self.bridge_status.get("real", {})),
            },
            "last_error": self.last_error,
        }

    def record_command_source(self, source):
        self.last_command_source = str(source)

    def set_last_error(self, error):
        self.last_error = str(error or "")

    def update_motor_feedback(self, joint_name, feedback):
        self._validate_joint_name(joint_name)
        self.latest_motor_feedback[joint_name] = dict(feedback)

    def update_sim_feedback(self, joint_name, feedback):
        self._validate_joint_name(joint_name)
        self.latest_sim_feedback[joint_name] = dict(feedback)

    def update_feedback_snapshot(self, feedback_snapshot):
        """FeedbackStateBuffer snapshot을 RobotState에 반영한다."""
        if not isinstance(feedback_snapshot, dict):
            raise ValueError("feedback_snapshot must be dict")

        latest_sim = feedback_snapshot.get("latest_sim_feedback", {})
        latest_real = feedback_snapshot.get("latest_motor_feedback", {})
        if isinstance(latest_sim, dict):
            for joint_name, feedback in latest_sim.items():
                self.update_sim_feedback(joint_name, feedback)
        if isinstance(latest_real, dict):
            for joint_name, feedback in latest_real.items():
                self.update_motor_feedback(joint_name, feedback)

        self.last_sim_rx_time = feedback_snapshot.get("last_sim_rx_time")
        self.last_real_rx_time = feedback_snapshot.get("last_real_rx_time")
        self.sim_connected = bool(feedback_snapshot.get("sim_connected"))
        self.real_connected = bool(feedback_snapshot.get("real_connected"))
        self.sim_feedback_stale = bool(feedback_snapshot.get("sim_feedback_stale", True))
        self.real_feedback_stale = bool(feedback_snapshot.get("real_feedback_stale", True))

        bridge_status = feedback_snapshot.get("bridge_status")
        if isinstance(bridge_status, dict):
            for source in ("sim", "real"):
                source_status = bridge_status.get(source)
                if isinstance(source_status, dict):
                    self.bridge_status[source] = dict(source_status)

        real_status = self.bridge_status.get("real", {})
        real_error = real_status.get("last_error") if isinstance(real_status, dict) else None
        if real_error:
            self.set_last_error(real_error)

    def _validate_value(self, field_name, value, valid_values):
        """상태 문자열이 허용된 값인지 검사한다."""
        if value not in valid_values:
            valid_text = ", ".join(valid_values)
            raise ValueError(
                f"Invalid {field_name}: {value!r}. Valid values: {valid_text}"
            )

    def _validate_joint_name(self, joint_name):
        """관절 이름이 등록된 이름인지 검사한다."""
        if joint_name not in self.JOINT_NAMES:
            valid_text = ", ".join(self.JOINT_NAMES)
            raise ValueError(
                f"Invalid joint_name: {joint_name!r}. Valid joint names: {valid_text}"
            )

    def _validate_motor_id(self, motor_id):
        """RobotModel에 등록된 CAN ID인지 검사한다."""
        if motor_id not in self.AK_MOTOR_IDS:
            valid_text = ", ".join(str(item) for item in self.AK_MOTOR_IDS)
            raise ValueError(
                "Invalid motor_id: {!r}. Valid motor_id: {}".format(motor_id, valid_text)
            )


if __name__ == "__main__":
    robot_state = RobotState()
    pprint(robot_state.get_snapshot())
