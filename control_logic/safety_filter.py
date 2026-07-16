import sys
import math
from pathlib import Path
from pprint import pprint


# 직접 실행할 때도 프로젝트 루트 기준 import가 동작하도록 경로를 보정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from control_logic.config_loader import load_robot_config
    from control_logic.robot_state import RobotState
    from robot_runtime.dof12_mapping import JOINT_NAMES_12
    from robot_runtime.robot_model import RobotModel
except ModuleNotFoundError:
    from config_loader import load_robot_config
    from robot_state import RobotState
    from robot_runtime.dof12_mapping import JOINT_NAMES_12
    from robot_runtime.robot_model import RobotModel


class SafetyFilter:
    """UI 또는 command_bus에서 들어온 명령의 안전 허용 여부만 검사한다."""

    LIMIT_POLICY = "reject"
    TARGET_UNIT = "deg"

    def __init__(self, robot_state, config, real_can_write_enabled=False, robot_model=None):
        # 상태 변경은 command_bus가 담당하므로 여기서는 참조만 보관한다.
        self.robot_state = robot_state
        self.config = config
        self.real_can_write_enabled = bool(real_can_write_enabled)
        self.robot_model = robot_model or RobotModel.load_12dof()
        self.expected_joint_names = tuple(self.robot_model.joint_names or JOINT_NAMES_12)

    def can_change_mode(self, new_mode):
        """운용 모드 변경 가능 여부를 검사한다."""
        if new_mode not in self.config["available_modes"]:
            return self._blocked(f"Invalid mode: {new_mode}")

        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            return self._blocked("Cannot change mode while E-STOP is active")

        # 모터가 켜진 상태에서는 모드 전환 중 물리 타깃이 바뀔 수 있으므로 차단한다.
        if self.robot_state.get_motor_status() == "Enabled":
            return self._blocked("Cannot change mode while motor is enabled")

        return self._allowed("Mode change allowed")

    def can_connect_target(self):
        """시뮬레이터 또는 실제 로봇 타깃 연결 시도 가능 여부를 검사한다."""
        # E-STOP ACTIVE 상태에서도 연결 시도 자체는 모터 구동이 아니므로 허용한다.
        return self._allowed("Target connection allowed")

    def can_enable_motor(self):
        """모터 Enable 가능 여부를 검사한다."""
        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            return self._blocked("Cannot enable motor while E-STOP is active")

        if self.robot_state.get_target_status() != "Connected":
            return self._blocked("Cannot enable motor before target connection")

        if self.robot_state.get_safety_status() != "OK":
            return self._blocked("Cannot enable motor while safety status is not OK")

        # 이미 Enabled인 상태는 위험 조건이 아니므로 중복 요청으로 보고 허용한다.
        if self.robot_state.get_motor_status() == "Enabled":
            return self._allowed("Motor already enabled")

        return self._allowed("Motor enable allowed")

    def can_send_real_write(self):
        """실제 CAN write는 명시 플래그가 켜진 경우에만 허용한다."""
        if not self.real_can_write_enabled:
            return self._blocked("REAL CAN write disabled by safety flag")
        return self._allowed("REAL CAN write flag enabled")

    def can_disable_motor(self):
        """모터 Disable 가능 여부를 검사한다."""
        # 모터를 끄는 명령은 현재 상태와 관계없이 항상 안전한 방향의 명령으로 본다.
        return self._allowed("Motor disable allowed")

    def can_clear_estop(self):
        """E-STOP 해제 요청 가능 여부를 검사한다."""
        # 이 메서드는 허용 여부만 반환하며 motor_status를 Enabled로 바꾸지 않는다.
        return self._allowed("Clear E-STOP allowed")

    def can_estop(self):
        """E-STOP 요청 가능 여부를 검사한다."""
        # E-STOP은 언제든지 안전한 방향의 명령으로 허용한다.
        return self._allowed("E-STOP allowed")

    def can_home(self):
        """Home 요청 가능 여부를 검사한다."""
        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            # 현재 단계의 Home은 실제 모터 명령이 아니라 UI 리셋용 동작으로 간주한다.
            return self._allowed("Home allowed for UI reset only while E-STOP is active")

        return self._allowed("Home allowed")

    def can_send_joint_command(self, joint_name, target_deg):
        """개별 관절 목표 각도 명령 가능 여부를 검사한다."""
        if joint_name not in self.expected_joint_names:
            return self._blocked(f"Invalid joint name: {joint_name}")

        if self.robot_state.get_target_status() != "Connected":
            return self._blocked("Cannot send joint command before target connection")

        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            return self._blocked("Cannot send joint command while E-STOP is active")

        if self.robot_state.get_safety_status() != "OK":
            return self._blocked("Cannot send joint command while safety status is not OK")

        if self.robot_state.get_motor_status() != "Enabled":
            return self._blocked("Cannot send joint command while motor is disabled")

        # joint command 값의 단위는 degree이며, limit 초과는 clamp하지 않고 reject한다.
        try:
            target_deg = float(target_deg)
        except (TypeError, ValueError):
            return self._blocked(f"Invalid target_deg: {target_deg}")

        joint_limits = self.config["joint_limits_deg"][joint_name]
        min_deg = float(joint_limits["min"])
        max_deg = float(joint_limits["max"])

        # 설정된 관절별 절대 각도 범위를 벗어나는 명령은 차단한다.
        if target_deg < min_deg or target_deg > max_deg:
            return self._blocked(
                "Joint limit exceeded: "
                f"joint_name={joint_name} "
                f"target_deg={target_deg} "
                f"min_deg={min_deg} "
                f"max_deg={max_deg} "
                f"unit={self.TARGET_UNIT} "
                f"policy={self.LIMIT_POLICY}"
            )

        current_deg = float(self.robot_state.get_joint_angle(joint_name))
        max_delta = float(
            self.config["command_rate_limit"]["max_joint_delta_deg_per_command"]
        )
        delta_deg = abs(target_deg - current_deg)

        # 한 번의 명령에서 너무 큰 각도 변화가 발생하지 않도록 제한한다.
        if delta_deg > max_delta:
            return self._blocked(
                f"Joint command delta {delta_deg} deg exceeds limit {max_delta} deg"
            )

        return self._allowed("Joint command allowed")

    def can_send_sim_joint_command(self, joint_name, target_deg):
        """UI -> SIM/SIM_TO_REAL payload 생성 가능 여부를 검사한다."""
        if self.robot_state.get_mode() not in ("SIM", "SIM_TO_REAL"):
            return self._blocked("SIM payload only allowed in SIM or SIM_TO_REAL mode")

        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            return self._blocked("Cannot send joint command while E-STOP is active")

        if self.robot_state.get_motor_status() != "Enabled":
            return self._blocked("Cannot send joint command while motor is disabled")

        target_result = self._check_joint_target_deg(joint_name, target_deg)
        if not target_result["allowed"]:
            return target_result

        return self._allowed("SIM joint payload allowed")

    def can_send_real_joint_command(self, joint_name, target_deg):
        """UI -> REAL/SIM_TO_REAL 실제 write 가능 여부를 검사한다.

        REAL_CAN_WRITE_ENABLED가 False이거나 확인 불가하면 정상적으로 차단한다.
        이번 프로젝트 복구 단계에서는 이 플래그를 활성화하지 않는다.
        """
        if self.robot_state.get_mode() not in ("REAL", "SIM_TO_REAL"):
            return self._blocked("REAL write only allowed in REAL or SIM_TO_REAL mode")

        if self.robot_state.get_safety_status() == "E-STOP ACTIVE":
            return self._blocked("Cannot send joint command while E-STOP is active")

        if self.robot_state.get_target_status() != "Connected":
            return self._blocked("Cannot send joint command before target connection")

        if self.robot_state.get_motor_status() != "Enabled":
            return self._blocked("Cannot send joint command while motor is disabled")

        write_result = self.can_send_real_write()
        if not write_result["allowed"]:
            return write_result

        target_result = self._check_joint_target_deg(joint_name, target_deg)
        if not target_result["allowed"]:
            return target_result

        return self._allowed("REAL joint write allowed")

    def can_accept_sim_feedback(self, joint_name, timestamp):
        """SIM -> UI feedback 반영 가능 여부를 검사한다."""
        return self._can_accept_feedback("sim", joint_name, timestamp)

    def can_accept_real_feedback(self, joint_name, timestamp):
        """REAL -> UI feedback 반영 가능 여부를 검사한다."""
        return self._can_accept_feedback("real", joint_name, timestamp)

    def _allowed(self, reason):
        """허용 결과를 표준 딕셔너리 형식으로 반환한다."""
        print(f"[COMMAND OK] {reason}")
        return {"allowed": True, "reason": reason}

    def _blocked(self, reason):
        """차단 결과를 표준 딕셔너리 형식으로 반환한다."""
        print(f"[SAFETY BLOCK] {self._translate_block_reason(reason)}")
        return {"allowed": False, "reason": reason}

    def _translate_block_reason(self, reason):
        translations = {
            "Cannot change mode while E-STOP is active": "E-STOP 상태에서는 mode change 불가",
            "Cannot enable motor before target connection": "target 미연결 상태에서는 motor enable 불가",
            "Cannot enable motor while E-STOP is active": "E-STOP 상태에서는 motor enable 불가",
            "Cannot enable motor while safety status is not OK": "안전 상태가 OK가 아니면 motor enable 불가",
            "Cannot change mode while motor is enabled": "motor enabled 상태에서는 SIM/REAL 모드 변경 불가",
            "Cannot send joint command before target connection": "target 미연결 상태에서는 joint command 불가",
            "Cannot send joint command while E-STOP is active": "E-STOP 상태에서는 joint command 불가",
            "Cannot send joint command while safety status is not OK": "안전 상태가 OK가 아니면 joint command 불가",
            "Cannot send joint command while motor is disabled": "motor disabled 상태에서는 joint command 불가",
            "REAL CAN write disabled by safety flag": "REAL_CAN_WRITE_ENABLED=False 상태에서는 실제 CAN write 불가",
        }
        if reason in translations:
            return translations[reason]
        if reason.startswith("Invalid joint name:"):
            return f"12축 기대 joint 이름이 아님: {reason.split(':', 1)[1].strip()}"
        if reason.startswith("Invalid target_deg:"):
            return f"joint command 값이 deg 숫자가 아님: {reason.split(':', 1)[1].strip()}"
        if reason.startswith("Joint limit exceeded:"):
            return "joint command가 deg limit을 초과하여 reject"
        if reason == "SIM payload only allowed in SIM or SIM_TO_REAL mode":
            return "SIM payload는 SIM/SIM_TO_REAL 모드에서만 생성 가능"
        if reason == "REAL write only allowed in REAL or SIM_TO_REAL mode":
            return "REAL write는 REAL/SIM_TO_REAL 모드에서만 가능"
        if reason.startswith("Invalid feedback source:"):
            return f"feedback source가 유효하지 않음: {reason.split(':', 1)[1].strip()}"
        if reason.startswith("Invalid feedback timestamp:"):
            return f"feedback timestamp가 유효하지 않음: {reason.split(':', 1)[1].strip()}"
        return reason

    def _check_joint_target_deg(self, joint_name, target_deg):
        if joint_name not in self.expected_joint_names:
            return self._blocked(f"Invalid joint name: {joint_name}")

        try:
            target_deg = float(target_deg)
        except (TypeError, ValueError):
            return self._blocked(f"Invalid target_deg: {target_deg}")

        if not math.isfinite(target_deg):
            return self._blocked(f"Invalid target_deg: {target_deg}")

        joint_limits = self.config["joint_limits_deg"][joint_name]
        min_deg = float(joint_limits["min"])
        max_deg = float(joint_limits["max"])
        if target_deg < min_deg or target_deg > max_deg:
            return self._blocked(
                "Joint limit exceeded: "
                f"joint_name={joint_name} "
                f"target_deg={target_deg} "
                f"min_deg={min_deg} "
                f"max_deg={max_deg} "
                f"unit={self.TARGET_UNIT} "
                f"policy={self.LIMIT_POLICY}"
            )
        return {"allowed": True, "reason": "target_deg_valid"}

    def _can_accept_feedback(self, source, joint_name, timestamp):
        if source not in ("sim", "real"):
            return self._blocked(f"Invalid feedback source: {source}")

        if joint_name not in self.expected_joint_names:
            return self._blocked(f"Invalid joint name: {joint_name}")

        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return self._blocked(f"Invalid feedback timestamp: {timestamp}")

        if not math.isfinite(timestamp):
            return self._blocked(f"Invalid feedback timestamp: {timestamp}")

        return self._allowed(f"{source.upper()} feedback accepted")


if __name__ == "__main__":
    config = load_robot_config()
    robot_state = RobotState()
    safety_filter = SafetyFilter(robot_state, config)

    print("Initial safety checks")
    pprint(
        {
            "can_change_mode_REAL": safety_filter.can_change_mode("REAL"),
            "can_connect_target": safety_filter.can_connect_target(),
            "can_enable_motor": safety_filter.can_enable_motor(),
            "can_disable_motor": safety_filter.can_disable_motor(),
            "can_clear_estop": safety_filter.can_clear_estop(),
            "can_home": safety_filter.can_home(),
            "can_send_left_knee_1deg": safety_filter.can_send_joint_command("left_knee_joint", 1.0),
        }
    )

    print("\nConnected and enabled safety checks")
    robot_state.set_target_status("Connected")
    robot_state.set_motor_status("Enabled")
    pprint(
        {
            "can_enable_motor_again": safety_filter.can_enable_motor(),
            "can_change_mode_REAL": safety_filter.can_change_mode("REAL"),
            "can_send_left_knee_3deg": safety_filter.can_send_joint_command("left_knee_joint", 3.0),
            "can_send_left_knee_20deg": safety_filter.can_send_joint_command("left_knee_joint", 20.0),
            "can_send_unknown_joint": safety_filter.can_send_joint_command(
                "unknown_joint", 0.0
            ),
            "can_send_bad_target": safety_filter.can_send_joint_command("left_knee_joint", "bad"),
        }
    )

    print("\nE-STOP safety checks")
    robot_state.set_motor_status("Disabled")
    robot_state.set_safety_status("E-STOP ACTIVE")
    pprint(
        {
            "can_change_mode_SIM": safety_filter.can_change_mode("SIM"),
            "can_connect_target": safety_filter.can_connect_target(),
            "can_home": safety_filter.can_home(),
            "can_enable_motor": safety_filter.can_enable_motor(),
        }
    )
