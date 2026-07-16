import sys
import threading
from pathlib import Path
from pprint import pprint


# 직접 실행할 때도 프로젝트 루트 기준 import가 동작하도록 경로를 보정한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from control_logic.command_buffer import CommandBuffer
    from control_logic.command_models import JointCommand
    from control_logic.config_loader import load_robot_config
    from control_logic.robot_state import RobotState
    from control_logic.safety_filter import SafetyFilter
    from robot_runtime.robot_model import RobotModel
except ModuleNotFoundError:
    from command_buffer import CommandBuffer
    from command_models import JointCommand
    from config_loader import load_robot_config
    from robot_state import RobotState
    from safety_filter import SafetyFilter
    from robot_runtime.robot_model import RobotModel


class CommandBus:
    """UI 명령을 안전 검사 후 RobotState에 반영하는 내부 명령 처리 클래스."""

    REAL_TARGET_MODES = ("REAL", "SIM_TO_REAL", "REAL_TO_SIM")

    def __init__(self, robot_state, safety_filter, config, robot_model=None, command_buffer=None):
        self.robot_state = robot_state
        self.safety_filter = safety_filter
        self.config = config
        self.robot_model = robot_model or RobotModel.load_12dof()
        self.command_buffer = command_buffer or CommandBuffer(self.robot_model)

        # 나중에 ROS2 publish를 붙일 수 있도록 허용된 명령만 기록한다.
        self.command_history = []
        self.pending_commands = []
        self._lock = threading.Lock()

    def submit_command(self, command_type, **payload):
        """UI에서 들어온 명령을 메인 제어 루프가 처리할 수 있도록 큐에 넣는다."""
        command = {"command_type": command_type, "payload": dict(payload)}
        with self._lock:
            self.pending_commands.append(command)
        return command

    def get_next_command(self):
        """대기 중인 명령을 하나 꺼낸다. 명령이 없으면 None을 반환한다."""
        with self._lock:
            if not self.pending_commands:
                return None

            return self.pending_commands.pop(0)

    def has_pending_commands(self):
        """처리 대기 중인 명령이 있는지 반환한다."""
        with self._lock:
            return bool(self.pending_commands)

    def request_mode_change(self, new_mode):
        """운용 모드 변경 요청을 처리한다."""
        old_mode = self.robot_state.get_mode()
        print(f"[MODE] {old_mode} -> {new_mode} 요청")
        safety_result = self.safety_filter.can_change_mode(new_mode)
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        # 현재는 실제 연결 과정 없이 테스트용으로 Connecting 후 Connected로 전환한다.
        self.robot_state.set_target_status("Connecting")
        self.robot_state.set_mode(new_mode)
        self.robot_state.set_target_status("Connected")

        # 모드 전환 후 모터가 자동 Enabled가 되지 않도록 명시적으로 Disabled로 둔다.
        self.robot_state.set_motor_status("Disabled")

        self._record_command("mode_change", {"new_mode": new_mode})
        self.command_buffer.push_mode_command(new_mode)
        self.robot_state.record_command_source("command_bus.mode_change")
        return self._result(True, f"Mode changed to {new_mode}")

    def request_connect_target(self):
        """시뮬레이터 또는 실제 로봇 타깃 연결 요청을 처리한다."""
        safety_result = self.safety_filter.can_connect_target()
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        current_mode = self.robot_state.get_mode()

        # 현재는 실제 연결 코드가 없으므로 즉시 Connected로 처리한다.
        self.robot_state.set_target_status("Connecting")

        if current_mode in self.REAL_TARGET_MODES:
            self._set_all_ak_motor_status("Connecting")

        self.robot_state.set_target_status("Connected")

        if current_mode in self.REAL_TARGET_MODES:
            self._set_all_ak_motor_status("Connected")

        self._record_command("connect_target", {"mode": current_mode})
        self.command_buffer.push_robot_command("connect_target")
        self.robot_state.record_command_source("command_bus.connect_target")
        return self._result(True, f"Target connected for mode: {current_mode}")

    def request_motor_enable(self):
        """모터 Enable 요청을 처리한다."""
        safety_result = self.safety_filter.can_enable_motor()
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        self.robot_state.set_motor_status("Enabled")

        self._record_command("motor_enable", {})
        self.command_buffer.push_robot_command("motor_enable")
        self.robot_state.record_command_source("command_bus.motor_enable")
        return self._result(True, "Motor enabled")

    def request_motor_disable(self):
        """모터 Disable 요청을 처리한다."""
        safety_result = self.safety_filter.can_disable_motor()
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        self.robot_state.set_motor_status("Disabled")

        # 실제 로봇 관련 모드에서는 모터 Disable 시 AK 모터 연결 상태도 끊긴 것으로 둔다.
        current_mode = self.robot_state.get_mode()
        if current_mode in self.REAL_TARGET_MODES:
            self._set_all_ak_motor_status("Disconnected")

        self._record_command("motor_disable", {"mode": current_mode})
        self.command_buffer.push_robot_command("motor_disable")
        self.robot_state.record_command_source("command_bus.motor_disable")
        return self._result(True, "Motor disabled")

    def request_clear_estop(self):
        """E-STOP 해제 요청을 처리한다."""
        safety_result = self.safety_filter.can_clear_estop()
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        self.robot_state.set_safety_status("OK")

        # E-STOP을 해제해도 모터가 자동 Enabled가 되지 않도록 Disabled를 유지한다.
        self.robot_state.set_motor_status("Disabled")

        self._record_command("clear_estop", {})
        self.command_buffer.push_safety_command("clear_estop")
        self.robot_state.record_command_source("command_bus.clear_estop")
        return self._result(True, "E-STOP cleared - motor remains disabled")

    def request_estop(self):
        """E-STOP 요청을 최우선으로 처리한다."""
        # E-STOP은 안전 검사 없이 항상 즉시 반영한다.
        self.robot_state.set_safety_status("E-STOP ACTIVE")
        self.robot_state.set_motor_status("Disabled")
        self._set_all_ak_motor_status("Disconnected")

        self._record_command("estop", {})
        self.command_buffer.push_safety_command("estop")
        self.robot_state.record_command_source("command_bus.estop")
        return self._result(True, "E-STOP pressed - motor disabled")

    def request_home(self):
        """Home 요청을 처리한다."""
        safety_result = self.safety_filter.can_home()
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        # 현재 Home은 실제 모터 명령이 아니라 UI/상태 리셋용이다.
        self.robot_state.reset_joint_angles()

        self._record_command("home", {})
        self.command_buffer.push_robot_command("home")
        self.robot_state.record_command_source("command_bus.home")
        return self._result(True, "Home requested - joint angles reset to 0 deg")

    def request_joint_command(self, joint_name, target_deg):
        """개별 관절 목표 각도 명령을 처리한다."""
        safety_result = self.safety_filter.can_send_joint_command(joint_name, target_deg)
        if not safety_result["allowed"]:
            return self._result(False, safety_result["reason"])

        target_deg = float(target_deg)
        self.robot_state.set_joint_angle(joint_name, target_deg)

        self._record_command(
            "joint_command", {"joint_name": joint_name, "target_deg": target_deg}
        )
        self.command_buffer.push_joint_command(
            JointCommand.create(
                joint_name,
                target_deg,
                source="command_bus",
                mode=self.robot_state.get_mode(),
            )
        )
        self.robot_state.record_command_source("command_bus.joint_command")
        print(f"[COMMAND OK] {self.robot_state.get_mode()} 모드 joint command accepted: {joint_name} {target_deg:.2f} deg")
        return self._result(
            True, f"Joint command accepted: {joint_name} -> {target_deg:.2f} deg"
        )

    def get_command_history(self):
        """허용되어 기록된 명령 이력의 복사본을 반환한다."""
        with self._lock:
            return [dict(item) for item in self.command_history]

    def get_command_snapshot(self):
        """ModeAdapter가 읽을 수 있는 허용 명령 snapshot을 반환한다."""
        return self.command_buffer.snapshot()

    def _record_command(self, command_type, payload):
        """나중에 ROS2 publish로 확장할 수 있도록 허용된 명령을 기록한다."""
        with self._lock:
            self.command_history.append(
                {
                    "command_type": command_type,
                    "payload": dict(payload),
                }
            )

    def _set_all_ak_motor_status(self, status):
        """RobotState에 등록된 모든 AK 모터 상태를 같은 값으로 설정한다."""
        for motor_id in self.robot_state.AK_MOTOR_IDS:
            self.robot_state.set_ak_motor_status(motor_id, status)

    def _result(self, success, message):
        """CommandBus 표준 결과 딕셔너리를 만든다."""
        if success:
            self.robot_state.set_last_error("")
        else:
            self.robot_state.set_last_error(message)
        return {
            "success": success,
            "message": message,
            "snapshot": self.robot_state.get_snapshot(),
        }


if __name__ == "__main__":
    config = load_robot_config()
    robot_state = RobotState()
    safety_filter = SafetyFilter(robot_state, config)
    command_bus = CommandBus(robot_state, safety_filter, config)

    print("명령 처리 결과")
    pprint(command_bus.request_connect_target())
    pprint(command_bus.request_motor_enable())
    pprint(command_bus.request_joint_command("left_knee_joint", 3.0))
    pprint(command_bus.request_joint_command("left_knee_joint", 20.0))
    pprint(command_bus.request_estop())

    print("\n최종 상태")
    pprint(robot_state.get_snapshot())

    print("\n명령 이력")
    pprint(command_bus.get_command_history())
