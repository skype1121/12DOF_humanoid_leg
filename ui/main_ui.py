import argparse
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control_logic.command_bus import CommandBus  # noqa: E402
from control_logic.config_loader import load_robot_config  # noqa: E402
from control_logic.robot_state import RobotState  # noqa: E402
from control_logic.safety_filter import SafetyFilter  # noqa: E402
from ros_interface.stage8_command_adapter import Stage8RosCommandAdapter  # noqa: E402
from robot_runtime.mode_types import VALID_RUNTIME_MODES  # noqa: E402
from robot_runtime.robot_model import RobotModel  # noqa: E402
from sim_interface.sim_command_file_adapter import (  # noqa: E402
    SIM_12DOF_COMMAND_FILE_PATH,
    Stage9SimCommandFileAdapter,
)


STAGE8_12AXIS_MIT_COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
STAGE8_12AXIS_MIT_STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
UI_ROS_NODE_NAME = "humanoid_12dof_control_ui"
SIM_12DOF_COMMAND_FILE_TEXT = "/tmp/humanoid_12dof_sim_command.json"
ISAAC_STATE_FILE_PATH = Path("/tmp/humanoid_isaac_state.json")

BG_COLOR = "#101214"
PANEL_COLOR = "#171a1d"
ROW_COLOR = "#1f2327"
TEXT_COLOR = "#f4f7fa"
MUTED_TEXT_COLOR = "#a7b0b8"
BORDER_COLOR = "#343a40"
GREEN = "#00c853"
YELLOW = "#ffb300"
RED = "#ff1744"
BLUE = "#2979ff"
ORANGE = "#ff6d00"
PURPLE = "#aa00ff"
CYAN = "#00bcd4"
WHITE = "#ffffff"
BLACK = "#050505"
BUTTON_BG = ORANGE
BUTTON_ACTIVE = "#d94800"
BUTTON_DISABLED = "#202327"
FONT_NORMAL = ("Consolas", 10)
FONT_BOLD = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 17, "bold")
FONT_SMALL = ("Consolas", 9)

SLIDER_PUBLISH_INTERVAL_SEC = 0.05
STATUS_HEARTBEAT_STALE_SEC = 2.0
TARGET_CONNECTED_HEARTBEAT_SEC = 1.5
ISAAC_STATE_STALE_SEC = 2.0
ROS_SPIN_INTERVAL_MS = 10
LOG_MAX_LINES = 400

STATE_DISCONNECTED = "DISCONNECTED"
STATE_CONNECTING = "CONNECTING"
STATE_CONNECTED = "CONNECTED"
STATE_FAILED = "FAILED"
STATE_DRY_RUN = "DRY_RUN"
STATE_BLOCKED_BY_SAFETY = "BLOCKED_BY_SAFETY"
STATE_ESTOP = "ESTOP"


ROBOT_MODEL = RobotModel.load_12dof()
JOINT_NAMES = ROBOT_MODEL.joint_names
JOINT_NAMES_12DOF = JOINT_NAMES
JOINT_DISPLAY_NAMES = {
    joint_name: ROBOT_MODEL.get_joint(joint_name).ui_label
    for joint_name in JOINT_NAMES
}

JOINT_LEGACY_DISPLAY_TEST_ALIASES = ("L hip F", "R ankle R")

JOINT_ROWS_12DOF = [
    {
        "joint_name": joint_name,
        "short_label": ROBOT_MODEL.get_joint(joint_name).ui_label,
        "motor_id": ROBOT_MODEL.get_joint(joint_name).can_id,
        "isaac_dof_index": ROBOT_MODEL.get_joint(joint_name).isaac_dof_index,
        "config": ROBOT_MODEL.config_for_joint(joint_name),
    }
    for joint_name in ROBOT_MODEL.joint_names
]


class HumanoidControlUI:
    """12DOF-only control UI for SIM and Stage8 REAL commands."""

    MODES = VALID_RUNTIME_MODES
    JOINT_NAMES = JOINT_NAMES_12DOF

    def __init__(self, root, use_ros=False):
        self.root = root
        self.root.title("휴머노이드 12자유도 제어")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 720)
        self.root.configure(bg=BG_COLOR)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.use_ros = use_ros
        self.current_mode = tk.StringVar(value="SIM")
        self.target_status = tk.StringVar(value=STATE_DISCONNECTED)
        self.real_can_status = tk.StringVar(value=STATE_DRY_RUN)
        self.estop_status = tk.StringVar(value="OK")
        self.last_command = tk.StringVar(value="--")
        self.last_reject_reason = tk.StringVar(value="--")
        self.warnings = tk.StringVar(value="--")
        self.errors = tk.StringVar(value="--")
        self.armed_status = tk.StringVar(value="DISARMED")
        self.allowed_joints = tk.StringVar(value="--")
        self.command_seq = tk.StringVar(value="--")
        self.status_heartbeat = tk.StringVar(value="--")
        self.can_channel = tk.StringVar(value="can0")
        self.can_status = tk.StringVar(value="알 수 없음")
        self.can_counts = tk.StringVar(value="tx 0 / rx 0 / skip 0")
        self.can_decode = tk.StringVar(value="--")

        self.connected = False
        self.target_connected_check_active = False
        self.target_connect_requested_at = None
        self.estop_active = False
        self.armed = False
        self.baseline_set = False
        self.real_can_write_enabled = False
        self.last_status_time = None
        self.stage8_status_payload = {}
        self._last_publish_time_by_joint = {}
        self._pending_slider_after_by_joint = {}
        self._pending_slider_value_by_joint = {}
        self._last_accepted_joint_deg = {
            joint_name: 0.0 for joint_name in JOINT_NAMES_12DOF
        }
        self._slider_rejected = {
            joint_name: False for joint_name in JOINT_NAMES_12DOF
        }
        self._updating_slider_programmatically = False
        self._last_real_to_sim_mirror_payload = None
        self._last_status_log_values = {}
        self._last_runtime_status_key = None
        self._heartbeat_after_id = None
        self._ros_spin_after_id = None
        self.mode_change_in_progress = False
        self.connect_in_progress = False
        self.log_line_count = 0

        self.joint_vars = {}
        self.joint_sliders = {}
        self.joint_target_labels = {}
        self.joint_actual_labels = {}
        self.joint_status_labels = {}
        self.joint_row_labels = {}
        self.motor_status_labels = {}
        self.command_buttons = []
        self.mode_buttons = {}
        self.status_value_labels = {}

        self.robot_model = ROBOT_MODEL
        self.robot_config = load_robot_config()
        self.robot_state = RobotState(robot_model=self.robot_model)
        self.safety_filter = SafetyFilter(
            self.robot_state,
            self.robot_config,
            robot_model=self.robot_model,
        )
        self.command_bus = CommandBus(
            self.robot_state,
            self.safety_filter,
            self.robot_config,
            robot_model=self.robot_model,
        )
        self.sim_command_adapter = Stage9SimCommandFileAdapter(
            self.robot_state,
            SIM_12DOF_COMMAND_FILE_PATH,
        )

        self.ros_error = ""
        self.stage8_ros_adapter = Stage8RosCommandAdapter(
            UI_ROS_NODE_NAME,
            STAGE8_12AXIS_MIT_COMMAND_TOPIC,
            STAGE8_12AXIS_MIT_STATUS_TOPIC,
            self._on_stage8_status_message,
        )

        self._build_ui()
        self._log_event("INFO", f"UI loaded {self.robot_model.joint_count} joints from RobotModel")
        self._log_event("INFO", "joint order: " + ", ".join(self.robot_model.joint_names))
        self._log_event("INFO", "Stage8 REAL 경로 준비: can0 / 12축 / ROS2 topic")
        self._refresh_controls()
        self._schedule_heartbeat_refresh()
        if self.use_ros:
            # --ros로 실행하면 버튼 클릭 전에도 ROS graph에 UI 노드와 topic이 보이게 만든다.
            self._start_stage8_ros_client()

    def _build_ui(self):
        container = tk.Frame(self.root, bg=BG_COLOR)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        container.columnconfigure(0, weight=4, minsize=760)
        container.columnconfigure(1, weight=0, minsize=340)
        container.rowconfigure(2, weight=4)
        container.rowconfigure(3, weight=1)

        self._build_header(container)
        self._build_connection_panel(container)
        self._build_joint_control(container)
        self._build_side_status(container)
        self._build_log_panel(container)

    def _panel(self, parent, title):
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            bd=1,
            relief=tk.SOLID,
            font=FONT_BOLD,
            padx=10,
            pady=8,
        )
        return frame

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=BG_COLOR)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)

        tk.Label(
            header,
            text="휴머노이드 12자유도 제어",
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=FONT_TITLE,
        ).grid(row=0, column=0, sticky="w")

        mode_frame = tk.Frame(header, bg=BG_COLOR)
        mode_frame.grid(row=0, column=1, sticky="e")
        for mode in self.MODES:
            button = self._button(
                mode_frame,
                mode,
                lambda selected=mode: self._set_mode(selected),
                width=14,
            )
            button.pack(side=tk.LEFT, padx=3)
            self.mode_buttons[mode] = button

        status = tk.Frame(header, bg=BG_COLOR)
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._status_chip(status, "모드", self.current_mode, 0)
        self._status_chip(status, "대상", self.target_status, 1)
        self._status_chip(status, "실물 송신", self.real_can_status, 2)
        self._status_chip(status, "비상정지", self.estop_status, 3)

    def _build_connection_panel(self, parent):
        panel = self._panel(parent, "연결 / 안전")
        panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for index in range(11):
            panel.columnconfigure(index, weight=1)

        actions = (
            ("대상 연결", self.connect_target),
            ("연결 해제", self.disconnect_target),
            ("CAN 상태확인", self.check_can),
            ("CAN 재연결", self.reconnect_can),
            ("제어 준비", self.arm_all),
            ("제어 해제", self.disarm_all),
            ("현재자세 기준설정", self.set_baseline),
            ("홈 0°", self.home_all_joints),
            ("전체 정지", self.stop_all),
            ("비상정지", self.estop),
            ("비상정지 해제", self.clear_estop),
        )
        for column, (label, command) in enumerate(actions):
            button = self._button(panel, label, command)
            button.grid(row=0, column=column, sticky="ew", padx=3, pady=2)
            self.command_buttons.append(button)

    def _build_joint_control(self, parent):
        panel = self._panel(parent, "12자유도 관절 제어")
        panel.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            panel,
            bg=PANEL_COLOR,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = tk.Scrollbar(panel, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.joint_canvas = canvas
        self.joint_scrollbar = scrollbar
        self.joint_rows_frame = tk.Frame(canvas, bg=PANEL_COLOR)
        self.joint_canvas_window = canvas.create_window(
            (0, 0),
            window=self.joint_rows_frame,
            anchor="nw",
        )

        self.joint_rows_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(self.joint_canvas_window, width=event.width),
        )

        for column in range(8):
            self.joint_rows_frame.columnconfigure(column, weight=1 if column == 3 else 0)

        headers = ("Group", "ID", "이름", "목표 슬라이더", "목표각", "실제각", "제한", "상태")
        for column, text in enumerate(headers):
            tk.Label(
                self.joint_rows_frame,
                text=text,
                bg=PANEL_COLOR,
                fg=MUTED_TEXT_COLOR,
                font=FONT_SMALL,
            ).grid(row=0, column=column, sticky="ew", padx=4, pady=(0, 5))

        grid_row = 1
        current_group = None
        for row in JOINT_ROWS_12DOF:
            joint_spec = self.robot_model.get_joint(row["joint_name"])
            if joint_spec.body_group != current_group:
                current_group = joint_spec.body_group
                tk.Label(
                    self.joint_rows_frame,
                    text=current_group,
                    bg="#222830",
                    fg=CYAN,
                    font=FONT_BOLD,
                    anchor="w",
                ).grid(row=grid_row, column=0, columnspan=8, sticky="ew", padx=3, pady=(6, 2))
                grid_row += 1
            self.create_joint_row(self.joint_rows_frame, grid_row, row)
            grid_row += 1

        self._build_motor_status_grid(self.joint_rows_frame, grid_row + 1)

    def create_joint_row(self, parent, row_index, row):
        joint_name = row["joint_name"]
        motor_id = int(row["motor_id"])
        limit = self._target_limit_for_mode(joint_name)
        value = tk.DoubleVar(value=0.0)
        self.joint_vars[joint_name] = value

        cells = (
            row["config"].get("body_group", "--"),
            str(motor_id),
            row["short_label"],
        )
        self.joint_row_labels[joint_name] = []
        for column, text in enumerate(cells):
            label = tk.Label(
                parent,
                text=text,
                bg=ROW_COLOR,
                fg=TEXT_COLOR,
                font=FONT_NORMAL,
                anchor="w",
            )
            label.grid(row=row_index, column=column, sticky="ew", padx=3, pady=2)
            self.joint_row_labels[joint_name].append(label)

        slider = tk.Scale(
            parent,
            from_=-limit,
            to=limit,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            showvalue=False,
            variable=value,
            command=lambda raw, name=joint_name: self._on_slider_changed(name, raw),
            bg=ROW_COLOR,
            fg=TEXT_COLOR,
            troughcolor="#56616b",
            highlightthickness=0,
            length=260,
        )
        slider.grid(row=row_index, column=3, sticky="ew", padx=4, pady=2)
        slider.bind(
            "<ButtonRelease-1>",
            lambda _event, name=joint_name: self._on_slider_released(name),
        )
        self.joint_sliders[joint_name] = slider

        self.joint_target_labels[joint_name] = self._row_label(parent, row_index, 4, "0.0")
        self.joint_actual_labels[joint_name] = self._row_label(parent, row_index, 5, "--")
        self._row_label(parent, row_index, 6, f"+/-{limit:.0f}")
        self.joint_status_labels[joint_name] = self._row_label(
            parent,
            row_index,
            7,
            "대기",
        )

    def _build_side_status(self, parent):
        side = tk.Frame(parent, bg=BG_COLOR)
        side.grid(row=2, column=1, sticky="nsew")
        side.configure(width=340)
        side.grid_propagate(False)
        side.rowconfigure(0, weight=1)
        side.rowconfigure(1, weight=0)

        status = self._panel(side, "Stage8 상태")
        status.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        status.columnconfigure(1, weight=1)
        pairs = (
            ("최근명령", self.last_command),
            ("거부사유", self.last_reject_reason),
            ("경고", self.warnings),
            ("오류", self.errors),
            ("제어준비", self.armed_status),
            ("허용축", self.allowed_joints),
            ("실물송신", self.real_can_status),
            ("CAN 채널", self.can_channel),
            ("CAN 상태", self.can_status),
            ("CAN 카운트", self.can_counts),
            ("CAN 디코드", self.can_decode),
            ("명령번호", self.command_seq),
            ("수신간격", self.status_heartbeat),
        )
        for row, (label, variable) in enumerate(pairs):
            tk.Label(status, text=label, bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR, font=FONT_SMALL).grid(
                row=row, column=0, sticky="w", padx=3, pady=2
            )
            tk.Label(
                status,
                textvariable=variable,
                bg=PANEL_COLOR,
                fg=TEXT_COLOR,
                font=FONT_SMALL,
                width=24,
                anchor="w",
                wraplength=210,
                justify=tk.LEFT,
            ).grid(
                row=row, column=1, sticky="w", padx=3, pady=2
            )

        self._build_rl_panel(side)

    # ------------------------------------------------------------------
    # RL 보행 패널 (2026-07-26) — 시뮬 전용: Isaac Sim MCP(8766) 라이브 컨트롤러의
    # rl_walk/rl_cmd/rl_stop/halt 를 조종한다. 실물 경로(Stage8)는 향후 젯슨
    # 브리지에서 — 이 패널은 SIM 검증용이며 CAN에 아무것도 보내지 않는다.
    # ------------------------------------------------------------------

    def _build_rl_panel(self, side):
        panel = self._panel(side, "RL 보행 (시뮬)")
        panel.grid(row=1, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)

        tk.Label(panel, text="정책", bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR,
                 font=FONT_SMALL).grid(row=0, column=0, sticky="w", padx=3)
        self.rl_ckpt = tk.StringVar(value="walk")
        ckpt_box = ttk.Combobox(
            panel, textvariable=self.rl_ckpt, state="readonly", width=10,
            values=("walk", "march", "rough", "walk_r3"))
        ckpt_box.grid(row=0, column=1, sticky="w", padx=3, pady=2)

        self.rl_heading_hold = tk.BooleanVar(value=True)
        tk.Checkbutton(
            panel, text="헤딩 유지", variable=self.rl_heading_hold,
            bg=PANEL_COLOR, fg=TEXT_COLOR, selectcolor=ROW_COLOR,
            activebackground=PANEL_COLOR, font=FONT_SMALL,
        ).grid(row=0, column=2, sticky="w", padx=3)

        self.rl_cmd_vars = {}
        for r, (key, label, lo, hi) in enumerate((
                ("cmd_x", "전진 vx", -0.4, 0.8),
                ("cmd_y", "측방 vy", -0.25, 0.25),
                ("wz", "회전 wz", -1.0, 1.0)), start=1):
            tk.Label(panel, text=label, bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR,
                     font=FONT_SMALL).grid(row=r, column=0, sticky="w", padx=3)
            var = tk.DoubleVar(value=0.5 if key == "cmd_x" else 0.0)
            self.rl_cmd_vars[key] = var
            tk.Scale(
                panel, from_=lo, to=hi, resolution=0.05, orient=tk.HORIZONTAL,
                variable=var, bg=PANEL_COLOR, fg=TEXT_COLOR,
                highlightthickness=0, troughcolor=ROW_COLOR, length=150,
                command=lambda _v, k=key: self._rl_on_cmd_changed(k),
            ).grid(row=r, column=1, columnspan=2, sticky="ew", padx=3)

        btns = tk.Frame(panel, bg=PANEL_COLOR)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        self.rl_buttons = []
        for idx, (text, cmd) in enumerate((
                ("RL 시작", self.rl_start),
                ("정지(서기)", self.rl_stop),
                ("즉시정지", self.rl_halt),
                ("리셋(재시작)", self.rl_reset))):
            b = self._button(btns, text, cmd, width=10)
            b.grid(row=idx // 2, column=idx % 2, padx=2, pady=2)
            self.rl_buttons.append(b)

        self.rl_status = tk.StringVar(value="미연결")
        tk.Label(panel, textvariable=self.rl_status, bg=PANEL_COLOR,
                 fg=TEXT_COLOR, font=FONT_SMALL, anchor="w", justify=tk.LEFT,
                 wraplength=300).grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=3, pady=(2, 3))

        self._rl_queue = queue.Queue()
        self._rl_last_cmd_sent = 0.0
        self._rl_poll_after_id = self.root.after(700, self._rl_poll)

    # ---- Isaac MCP 소켓 (walk_ui.IsaacLink 패턴 축약판 — 워커 스레드 전송) ----

    def _rl_send(self, payload, label=""):
        def worker():
            try:
                import socket as _socket
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(8.0)
                sock.connect(("localhost", int(os.environ.get("ISAAC_MCP_PORT", 8766))))
                code = ("import sim_walking.live_controller as LC; "
                        f"import json; print('RLUI:'+json.dumps(LC.command({payload!r}), ensure_ascii=False))")
                req = json.dumps({"type": "simulation.execute_script",
                                  "params": {"code": code}})
                sock.sendall(req.encode())
                chunks = b""
                while True:
                    part = sock.recv(1 << 16)
                    if not part:
                        break
                    chunks += part
                    try:
                        resp = json.loads(chunks.decode())
                        break
                    except json.JSONDecodeError:
                        continue
                sock.close()
                raw = json.dumps(resp, ensure_ascii=False)
                marker = raw.find("RLUI:")
                if marker >= 0:
                    body = raw[marker + 5:]
                    end = body.find("\\n")
                    body = body[:end] if end >= 0 else body
                    self._rl_queue.put((label, json.loads(body.replace('\\"', '"'))))
                else:
                    self._rl_queue.put((label, {"ok": False, "error": "응답 파싱 실패"}))
            except Exception as ex:
                self._rl_queue.put((label, {"ok": False, "error": f"{type(ex).__name__}: {ex}"}))
        threading.Thread(target=worker, daemon=True).start()

    def _rl_poll(self):
        try:
            while True:
                label, resp = self._rl_queue.get_nowait()
                self._rl_render(label, resp)
        except queue.Empty:
            pass
        # 주기 상태 조회 (RL 패널은 시뮬 전용 — 부담 적은 700ms)
        self._rl_send({"cmd": "status"}, label="status")
        self._rl_poll_after_id = self.root.after(700, self._rl_poll)

    def _rl_render(self, label, resp):
        if not isinstance(resp, dict):
            return
        if not resp.get("ok"):
            self.rl_status.set(f"오류: {resp.get('error', '?')[:120]}")
            if label != "status":
                self._log_event("WARN", f"RL {label} 실패: {resp.get('error', '?')}")
            return
        mode = resp.get("mode", "?")
        parts = [f"모드 {mode}"]
        if "forward_m" in resp:
            parts.append(f"전진 {resp['forward_m']}m")
        if "lean_ap" in resp:
            parts.append(f"기울기 {resp['lean_ap']}°/{resp.get('lean_lat', '?')}°")
        if mode == "RL":
            parts.append(f"정책 {resp.get('rl_checkpoint', '?')}")
            cmd = resp.get("rl_cmd")
            if cmd:
                parts.append(f"cmd({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})")
            if resp.get("heading_hold"):
                parts.append("헤딩유지")
        self.rl_status.set(" | ".join(str(p) for p in parts))
        if label in ("rl_walk", "rl_stop", "halt"):
            self._log_event("INFO", f"RL {label}: {resp}")

    def _rl_on_cmd_changed(self, _key):
        now = time.time()
        if now - self._rl_last_cmd_sent < 0.15:   # 슬라이더 디바운스
            return
        self._rl_last_cmd_sent = now
        self._rl_send({"cmd": "rl_cmd",
                       "cmd_x": float(self.rl_cmd_vars["cmd_x"].get()),
                       "cmd_y": float(self.rl_cmd_vars["cmd_y"].get()),
                       "wz": float(self.rl_cmd_vars["wz"].get())})

    def rl_start(self):
        if self.estop_active:
            self._log_event("SAFETY_BLOCK", "비상정지 중 — RL 시작 차단")
            return
        ck = self.rl_ckpt.get()
        payload = {"cmd": "rl_walk", "checkpoint": ck,
                   "cmd_x": float(self.rl_cmd_vars["cmd_x"].get()),
                   "cmd_y": float(self.rl_cmd_vars["cmd_y"].get()),
                   "wz": float(self.rl_cmd_vars["wz"].get()),
                   "heading_hold": bool(self.rl_heading_hold.get())}
        if ck == "march":
            payload.update({"cmd_x": 0.0, "cmd_y": 0.0, "wz": 0.0})
        self._rl_send(payload, label="rl_walk")
        self._log_event("INFO", f"RL 시작 요청: {payload}")

    def rl_stop(self):
        self._rl_send({"cmd": "rl_stop"}, label="rl_stop")

    def rl_halt(self):
        self._rl_send({"cmd": "halt"}, label="halt")

    def rl_reset(self):
        """넘어짐(FALLEN) 복구: 씬을 초기 자세로 리셋 — RL 재시작 전 필수."""
        self._rl_send({"cmd": "reset"}, label="reset")
        self._log_event("INFO", "RL 씬 리셋 요청 (넘어짐 복구)")

    def _build_motor_status_grid(self, parent, row_index):
        motors = self._panel(parent, "모터 상태")
        motors.grid(row=row_index, column=0, columnspan=8, sticky="ew", padx=3, pady=(14, 2))
        for column in range(3):
            motors.columnconfigure(column, weight=1, uniform="motor_status")

        for index, row in enumerate(JOINT_ROWS_12DOF):
            joint_name = row["joint_name"]
            motor_id = self.robot_model.get_joint(joint_name).can_id
            cell = tk.Frame(motors, bg=ROW_COLOR, bd=1, relief=tk.SOLID)
            cell.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=4,
                pady=4,
            )
            cell.columnconfigure(1, weight=1)

            tk.Label(
                cell,
                text=f"ID {motor_id}",
                bg=ROW_COLOR,
                fg=MUTED_TEXT_COLOR,
                font=FONT_SMALL,
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=(7, 5), pady=(5, 1))
            tk.Label(
                cell,
                text=row["short_label"],
                bg=ROW_COLOR,
                fg=TEXT_COLOR,
                font=FONT_BOLD,
                anchor="w",
            ).grid(
                row=0,
                column=1,
                sticky="ew",
                padx=(0, 7),
                pady=(5, 1),
            )
            label = tk.Label(
                cell,
                text="알 수 없음\n실 -- | 목 --",
                bg=ROW_COLOR,
                fg=MUTED_TEXT_COLOR,
                font=FONT_SMALL,
                anchor="w",
                justify=tk.LEFT,
                wraplength=220,
            )
            label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=7, pady=(0, 6))
            self.motor_status_labels[joint_name] = label

    def _build_log_panel(self, parent):
        panel = self._panel(parent, "로그")
        panel.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            panel,
            height=6,
            bg="#08090a",
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            font=FONT_SMALL,
            wrap=tk.WORD,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_scrollbar = tk.Scrollbar(panel, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.tag_configure("INFO", foreground=TEXT_COLOR)
        self.log_text.tag_configure("MODE", foreground=CYAN)
        self.log_text.tag_configure("JOINT", foreground=GREEN)
        self.log_text.tag_configure("WARN", foreground=YELLOW)
        self.log_text.tag_configure("ERROR", foreground=RED)
        self.log_text.tag_configure("SAFETY_BLOCK", foreground=ORANGE)
        self.log_text.tag_configure("DRY_RUN", foreground=YELLOW)
        self._log_event("INFO", "12자유도 UI 준비 완료")
        self._log_event("INFO", f"SIM 명령 파일: {SIM_12DOF_COMMAND_FILE_TEXT}")

    def _button(self, parent, text, command, width=12):
        button = tk.Button(
            parent,
            text=text,
            bg=self._button_base_color(text),
            fg=self._button_text_color(text),
            activebackground=BUTTON_ACTIVE,
            activeforeground=WHITE,
            disabledforeground="#6b7280",
            relief=tk.RAISED,
            bd=2,
            font=FONT_BOLD,
            width=width,
        )
        button.configure(command=lambda: self._run_button_command(button, command))
        return button

    def _run_button_command(self, button, command):
        self._flash_button_press(button)
        try:
            return command()
        except Exception as error:
            self.errors.set(str(error))
            self._log_event("ERROR", f"UI action exception: {error}")
            self._refresh_controls()
            return False

    def _button_base_color(self, text):
        return BUTTON_BG

    def _button_text_color(self, text):
        return WHITE

    def _configure_if_changed(self, widget, **options):
        changed = {}
        for key, value in options.items():
            if str(widget.cget(key)) != str(value):
                changed[key] = value
        if changed:
            widget.configure(**changed)

    def _set_button_visual(self, button, enabled):
        text = str(button.cget("text"))
        if enabled:
            self._configure_if_changed(
                button,
                state=tk.NORMAL,
                bg=self._button_base_color(text),
                fg=self._button_text_color(text),
                relief=tk.RAISED,
            )
            return
        self._configure_if_changed(
            button,
            state=tk.DISABLED,
            bg=BUTTON_DISABLED,
            fg="#6b7280",
            relief=tk.SOLID,
        )

    def _flash_button_press(self, button):
        if str(button.cget("state")) == tk.DISABLED:
            return
        text = str(button.cget("text"))
        button.configure(bg=BUTTON_ACTIVE, fg=WHITE, relief=tk.SUNKEN)
        self.root.after(
            140,
            lambda: button.configure(
                bg=self._button_base_color(text),
                fg=self._button_text_color(text),
                relief=tk.RAISED,
            ),
        )

    def _status_chip(self, parent, label, variable, column):
        frame = tk.Frame(parent, bg=PANEL_COLOR, bd=1, relief=tk.SOLID)
        frame.grid(row=0, column=column, sticky="ew", padx=3)
        parent.columnconfigure(column, weight=1)
        tk.Label(frame, text=label, bg=PANEL_COLOR, fg=MUTED_TEXT_COLOR, font=FONT_SMALL).pack(
            side=tk.LEFT, padx=(8, 4), pady=4
        )
        value_label = tk.Label(frame, textvariable=variable, bg=PANEL_COLOR, fg=TEXT_COLOR, font=FONT_BOLD)
        value_label.pack(
            side=tk.LEFT, padx=(0, 8), pady=4
        )
        self.status_value_labels[label] = value_label

    def _refresh_status_chip_colors(self):
        target_label = self.status_value_labels.get("대상")
        if target_label is not None:
            target = self.target_status.get()
            if target == STATE_CONNECTED:
                self._configure_if_changed(target_label, fg=GREEN)
            elif target == STATE_CONNECTING:
                self._configure_if_changed(target_label, fg=YELLOW)
            elif target in (STATE_FAILED, STATE_BLOCKED_BY_SAFETY):
                self._configure_if_changed(target_label, fg=RED)
            else:
                self._configure_if_changed(target_label, fg=MUTED_TEXT_COLOR)

        estop_label = self.status_value_labels.get("비상정지")
        if estop_label is not None:
            self._configure_if_changed(estop_label, fg=RED if self.estop_active else GREEN)

        write_label = self.status_value_labels.get("실물 송신")
        if write_label is not None:
            status = self.real_can_status.get()
            if status == STATE_BLOCKED_BY_SAFETY:
                self._configure_if_changed(write_label, fg=RED)
            elif status == STATE_DRY_RUN:
                self._configure_if_changed(write_label, fg=YELLOW)
            else:
                self._configure_if_changed(write_label, fg=RED if self.real_can_write_enabled else YELLOW)

    def _row_label(self, parent, row, column, text):
        label = tk.Label(parent, text=text, bg=ROW_COLOR, fg=TEXT_COLOR, font=FONT_SMALL, anchor="w")
        label.grid(row=row, column=column, sticky="ew", padx=3, pady=2)
        return label

    def _set_mode(self, mode):
        if mode not in self.MODES:
            return
        if self.mode_change_in_progress:
            self.last_reject_reason.set("mode_change_in_progress")
            self._log_event("SAFETY_BLOCK", "mode 변경 중복 요청 차단")
            return False
        old_mode = self.robot_state.get_mode()
        self.mode_change_in_progress = True
        self._log_event("MODE", f"{old_mode} -> {mode} 요청")
        self._refresh_controls()
        try:
            result = self.command_bus.request_mode_change(mode)
            if not result["success"]:
                self.last_reject_reason.set(result["message"])
                self._log_event("SAFETY_BLOCK", f"모드 변경 차단: {result['message']}")
                return False
            self.current_mode.set(self.robot_state.get_mode())
            self.target_connected_check_active = False
            self.target_connect_requested_at = None
            self.connected = False
            self.armed = False
            self.baseline_set = False
            self.target_status.set(STATE_DISCONNECTED)
            self.armed_status.set("DISARMED")
            self.robot_state.set_target_status("Disconnected")
            if mode in ("REAL", "SIM_TO_REAL") and not self.real_can_write_enabled:
                self.real_can_status.set(STATE_DRY_RUN)
                self._log_event("DRY_RUN", "REAL_CAN_WRITE_ENABLED=False, REAL 경로 차단")
            self._log_event("MODE", f"모드 변경 성공: {old_mode} -> {mode}")
            return True
        finally:
            self.mode_change_in_progress = False
            self._sync_ui_from_runtime_snapshot(log_changes=True)
            self._refresh_controls()

    def connect_target(self):
        if self.connect_in_progress or self.target_status.get() == STATE_CONNECTING:
            self.last_reject_reason.set("connect_in_progress")
            self._log_event("SAFETY_BLOCK", "connect 중복 요청 차단")
            return False
        if self.connected and self.target_status.get() == STATE_CONNECTED:
            self._log_event("INFO", "대상 연결 요청 무시: 이미 CONNECTED")
            return False
        self.connect_in_progress = True
        self.target_connected_check_active = True
        self.target_connect_requested_at = time.time()
        self.connected = False
        self.target_status.set(STATE_CONNECTING)
        self.robot_state.set_target_status("Connecting")
        self._log_event("INFO", f"connect target 요청: mode={self.current_mode.get()}")
        self._refresh_controls()
        try:
            if self.current_mode.get() == "SIM":
                ready, reason = self._is_sim_ready()
                if not ready:
                    self.connected = False
                    self.target_connected_check_active = False
                    self.target_status.set(STATE_DISCONNECTED)
                    self.robot_state.set_target_status("Disconnected")
                    self.last_reject_reason.set(reason)
                    self._log_event("SAFETY_BLOCK", f"SIM 대상 연결 차단: {reason}")
                    return False

            bus_result = self.command_bus.request_connect_target()
            if not bus_result["success"]:
                self.last_reject_reason.set(bus_result["message"])
                self.target_status.set(STATE_BLOCKED_BY_SAFETY)
                self._log_event("SAFETY_BLOCK", f"대상 연결 차단: {bus_result['message']}")
                return False
            if self.current_mode.get() == "SIM":
                self.connected = True
                self.target_connected_check_active = False
                self.target_status.set(STATE_CONNECTED)
                self._log_event("INFO", "SIM 대상 연결 완료: motor는 DISARMED 상태 유지")
                return True
            self.robot_state.set_target_status("Connecting")
            self._start_stage8_ros_client()
            self._update_target_connection_state()
            self._log_event("INFO", "대상 연결 요청: Stage8 상태 수신 대기")
            return True
        finally:
            if self.current_mode.get() == "SIM":
                self.connect_in_progress = False
            self._refresh_controls()

    def disconnect_target(self):
        self.target_connected_check_active = False
        self.target_connect_requested_at = None
        self.connect_in_progress = False
        self.connected = False
        self.armed = False
        self.baseline_set = False
        self.armed_status.set("DISARMED")
        self.target_status.set(STATE_DISCONNECTED)
        self.command_bus.request_motor_disable()
        self.robot_state.set_target_status("Disconnected")
        self._log_event("INFO", "연결 해제: STOP_ALL을 제외한 Stage8 명령 전송 차단")
        self._refresh_controls()
        return True

    def check_can(self):
        self._start_stage8_ros_client()
        self._refresh_can_status_from_payload()
        self._log_event("INFO", f"CAN 상태확인 요청: Stage8 topic 사용, 채널={self.can_channel.get()} 상태={self.can_status.get()}")
        return self.send_stage8_command({"command": "CHECK_CAN"}, allow_can_command=True)

    def reconnect_can(self):
        self._start_stage8_ros_client()
        self._log_event("INFO", "CAN 재연결 요청: Stage8 topic 사용")
        return self.send_stage8_command({"command": "RECONNECT_CAN"}, allow_can_command=True)

    def arm_all(self):
        if not self.connected:
            self.last_reject_reason.set("target_not_connected")
            self._log_event("SAFETY_BLOCK", "제어 준비 차단: CONNECTED 확정 후에만 ARM 가능")
            self._refresh_controls()
            return False
        if self.robot_state.motor_enabled:
            self.last_reject_reason.set("motor_already_enabled")
            self._log_event("INFO", "제어 준비 요청 무시: 이미 ARMED")
            self._refresh_controls()
            return False

        if self.current_mode.get() == "SIM":
            ready, reason = self._is_sim_ready()
            if not ready:
                self._mark_sim_not_ready(reason)
                self.last_reject_reason.set(reason)
                self._log_event("SAFETY_BLOCK", f"SIM 제어 준비 차단: {reason}")
                self._refresh_controls()
                return False
            self._log_event("INFO", "SIM motor enable request: 제어 준비")
            result = self.command_bus.request_motor_enable()
            if result["success"]:
                self.armed = True
                self.armed_status.set("ARMED")
            else:
                self.last_reject_reason.set(result["message"])
            self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"motor enable result: {result['message']}")
            self._refresh_controls()
            return bool(result["success"])

        self._start_stage8_ros_client()
        self._log_event("INFO", "motor enable request: 제어 준비")
        self._log_event("INFO", "제어 준비: Stage8이 관절 목표 명령을 받을 준비 상태, 단독 구동 명령 아님")
        sent = self.send_stage8_command({"command": "ARM_ALL"})
        if sent:
            result = self.command_bus.request_motor_enable()
            if result["success"]:
                self.armed_status.set("ARMED")
            self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"motor enable result: {result['message']}")
        self._refresh_controls()
        return sent

    def disarm_all(self):
        if self.current_mode.get() == "SIM":
            self._log_event("INFO", "SIM motor disable request: 제어 해제")
            result = self.command_bus.request_motor_disable()
            self.armed = False
            self.baseline_set = False
            self.armed_status.set("DISARMED")
            self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"motor disable result: {result['message']}")
            self._refresh_controls()
            return bool(result["success"])

        self._start_stage8_ros_client()
        self._log_event("INFO", "motor disable request: 제어 해제")
        result = self.command_bus.request_motor_disable()
        self.armed = False
        self.baseline_set = False
        self.armed_status.set("DISARMED")
        self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"motor disable result: {result['message']}")
        sent = self.send_stage8_command({"command": "DISARM_ALL"})
        self._refresh_controls()
        return sent

    def set_baseline(self):
        mode = self.current_mode.get()
        if mode == "SIM":
            self._log_event("WARN", "현재자세 기준설정 생략: SIM 모드는 직접 목표각/HOME만 사용")
            return False
        self._start_stage8_ros_client()
        self._log_event("INFO", "현재자세 기준설정: 현재 실제 자세를 이후 상대 목표각의 기준으로 사용")
        return self.send_stage8_command({"command": "SET_BASELINE_FROM_CURRENT_ALL"})

    def stop_all(self):
        self._log_event("WARN", "전체 정지 요청")
        result = self.command_bus.request_motor_disable()
        self.armed = False
        self.armed_status.set("DISARMED")
        self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"motor disable result: {result['message']}")
        self.send_stage8_command({"command": "STOP_ALL"}, force_stop=True)
        self._refresh_controls()

    def home_all_joints(self):
        if self.estop_active:
            self.last_reject_reason.set("blocked_by_estop")
            self._log_event("SAFETY_BLOCK", "홈 차단: 비상정지 활성")
            return False

        mode = self.current_mode.get()
        if mode == "REAL_TO_SIM":
            self.last_reject_reason.set("real_to_sim_read_only")
            self._log_event("WARN", "홈 비활성: REAL_TO_SIM은 실제값을 SIM에 미러링하는 읽기 모드")
            return False

        if mode not in ("SIM", "REAL", "SIM_TO_REAL"):
            return False

        if not self.connected or not self.robot_state.motor_enabled:
            self.last_reject_reason.set("target_or_motor_not_ready")
            self._log_event("SAFETY_BLOCK", "홈 차단: Connect + ARM 이후에만 HOME 가능")
            return False

        if mode in ("SIM", "SIM_TO_REAL"):
            ready, reason = self._is_sim_ready()
            if not ready:
                self._mark_sim_not_ready(reason)
                self.last_reject_reason.set(reason)
                self._log_event("SAFETY_BLOCK", f"SIM 홈 차단: {reason}")
                return False

        home_result = self.command_bus.request_home()
        if not home_result["success"]:
            self.last_reject_reason.set(home_result["message"])
            self._log_event("SAFETY_BLOCK", f"홈 차단: {home_result['message']}")
            return False

        sim_ok = True
        real_ok = True
        if mode in ("SIM", "SIM_TO_REAL"):
            sim_ok = self.send_home_sim_targets()
        if mode in ("REAL", "SIM_TO_REAL"):
            self._start_stage8_ros_client()
            if not self.real_can_write_enabled:
                self._log_event("DRY_RUN", "홈 0° Stage8 목표 명령 요청: REAL_CAN_WRITE_ENABLED=False")
            real_ok = self.send_home_real_targets()
        if not (sim_ok and real_ok):
            self.last_reject_reason.set("home_send_failed")
            self._log_event("SAFETY_BLOCK", "홈 실패: 일부 HOME 명령 전송 실패")
            return False

        self._updating_slider_programmatically = True
        try:
            for joint_name in JOINT_NAMES_12DOF:
                self.joint_vars[joint_name].set(0.0)
                self._last_accepted_joint_deg[joint_name] = 0.0
                self._slider_rejected[joint_name] = False
                self.joint_target_labels[joint_name].configure(text="0.0")
        finally:
            self._updating_slider_programmatically = False
        return True

    def send_home_sim_targets(self):
        try:
            self.sim_command_adapter.write_home_targets()
        except Exception as error:
            self._log_event("ERROR", f"SIM 홈 명령 거부: {error}")
            return False
        self._log_event("INFO", "SIM 홈 목표각 기록: 12개 관절 기준상대 0.0 deg")
        return True

    def send_home_real_targets(self):
        sent = 0
        for joint_name in JOINT_NAMES_12DOF:
            payload = {
                "command": "SET_JOINT_TARGET",
                "joint": joint_name,
                "target_deg": 0.0,
            }
            if self.send_stage8_command(payload, allow_home_target=True):
                sent += 1
        self._log_event("INFO", f"Stage8 홈 목표 명령 전송: {sent}/{len(JOINT_NAMES_12DOF)}")
        return sent == len(JOINT_NAMES_12DOF)

    def estop(self):
        result = self.command_bus.request_estop()
        self.estop_active = True
        self.estop_status.set(STATE_ESTOP)
        self.stop_all()
        # RL 시뮬 보행도 즉시 동결 (halt) — 안전 일관성 (2026-07-26)
        try:
            self.rl_halt()
        except Exception:
            pass
        self._log_event("SAFETY_BLOCK", f"비상정지 활성: {result['message']}")
        self._refresh_controls()

    def clear_estop(self):
        result = self.command_bus.request_clear_estop()
        self.estop_active = False
        self.estop_status.set("OK")
        self._log_event("INFO" if result["success"] else "SAFETY_BLOCK", f"비상정지 해제: {result['message']}")
        self._refresh_controls()

    def _on_slider_changed(self, joint_name, raw_value):
        if self._updating_slider_programmatically:
            return
        now = time.time()
        last_time = self._last_publish_time_by_joint.get(joint_name, 0.0)
        if now - last_time < SLIDER_PUBLISH_INTERVAL_SEC:
            self._pending_slider_value_by_joint[joint_name] = float(raw_value)
            if joint_name not in self._pending_slider_after_by_joint:
                delay_ms = max(1, int((SLIDER_PUBLISH_INTERVAL_SEC - (now - last_time)) * 1000))
                self._pending_slider_after_by_joint[joint_name] = self.root.after(
                    delay_ms,
                    lambda name=joint_name: self._flush_pending_slider(name),
                )
            return
        self._last_publish_time_by_joint[joint_name] = now
        self._route_joint_target(joint_name, float(raw_value))

    def _flush_pending_slider(self, joint_name):
        self._pending_slider_after_by_joint.pop(joint_name, None)
        if joint_name not in self._pending_slider_value_by_joint:
            return
        value = self._pending_slider_value_by_joint.pop(joint_name)
        self._last_publish_time_by_joint[joint_name] = time.time()
        self._route_joint_target(joint_name, float(value))

    def _on_slider_released(self, joint_name):
        if joint_name in self._pending_slider_after_by_joint:
            after_id = self._pending_slider_after_by_joint.pop(joint_name)
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
            if joint_name in self._pending_slider_value_by_joint:
                value = self._pending_slider_value_by_joint.pop(joint_name)
                self._last_publish_time_by_joint[joint_name] = time.time()
                self._route_joint_target(joint_name, float(value))

        if not self._slider_rejected.get(joint_name, False):
            return

        last_accepted = float(self._last_accepted_joint_deg.get(joint_name, 0.0))
        slider = self.joint_sliders.get(joint_name)
        if slider is not None:
            self._updating_slider_programmatically = True
            try:
                slider.set(last_accepted)
            finally:
                self._updating_slider_programmatically = False
        if joint_name in self.joint_target_labels:
            self.joint_target_labels[joint_name].configure(text=f"{last_accepted:.1f}")
        self._slider_rejected[joint_name] = False
        self._pending_slider_value_by_joint.pop(joint_name, None)
        self._log_event("INFO", f"{joint_name} slider rollback: {last_accepted:.2f} deg")

    def _command_result_fields(self, result, requested_deg):
        if not isinstance(result, dict):
            return False, "invalid_command_result", None
        allowed = bool(result.get("allowed", result.get("success", False)))
        reason = str(result.get("reason", result.get("message", "")))
        applied_deg = result.get("applied_deg", requested_deg if allowed else None)
        if applied_deg is not None:
            applied_deg = float(applied_deg)
        return allowed, reason, applied_deg

    def _route_joint_target(self, joint_name, target_deg):
        target_deg = self._clamp_joint_value(joint_name, target_deg)
        self.joint_target_labels[joint_name].configure(text=f"{target_deg:.1f}")
        mode = self.current_mode.get()
        self._log_event("JOINT", f"{joint_name} target={target_deg:.2f} deg mode={mode}")
        if mode in ("SIM", "SIM_TO_REAL"):
            ready, reason = self._is_sim_ready()
            if not ready:
                self._mark_sim_not_ready(reason)
                self._slider_rejected[joint_name] = True
                self.last_reject_reason.set(reason)
                self._log_event("SAFETY_BLOCK", f"SIM 관절 명령 차단: {reason}")
                return False
        command_result = self.command_bus.request_joint_command(joint_name, target_deg)
        allowed, reason, applied_deg = self._command_result_fields(command_result, target_deg)
        if not allowed:
            self._slider_rejected[joint_name] = True
            self.last_reject_reason.set(reason)
            self._log_event("SAFETY_BLOCK", f"관절 명령 차단: {reason}")
            return False

        if applied_deg is None:
            applied_deg = target_deg
        self._last_accepted_joint_deg[joint_name] = float(applied_deg)
        self._slider_rejected[joint_name] = False
        self.joint_target_labels[joint_name].configure(text=f"{float(applied_deg):.1f}")
        self._log_event("JOINT", f"{joint_name} accepted={float(applied_deg):.2f} deg")

        if mode == "SIM":
            return self.send_sim_command(joint_name, applied_deg)
        if mode == "REAL":
            return self.send_real_command(joint_name, applied_deg)
        if mode == "SIM_TO_REAL":
            sim_ok = self.send_sim_command(joint_name, applied_deg)
            real_ok = self.send_real_command(joint_name, applied_deg)
            return sim_ok and real_ok
        if mode == "REAL_TO_SIM":
            self._log_event("WARN", "REAL_TO_SIM 읽기 모드: 슬라이더 명령 무시")
            return False
        return False

    def send_sim_command(self, joint_name, deg):
        ready, reason = self._is_sim_ready()
        if not ready:
            self._mark_sim_not_ready(reason)
            self.last_reject_reason.set(reason)
            self._log_event("SAFETY_BLOCK", f"SIM 명령 차단: {reason}")
            return False
        try:
            self.sim_command_adapter.write_joint_command(joint_name, deg)
        except Exception as error:
            self._log_event("ERROR", f"SIM 명령 거부: {error}")
            return False
        self._log_event("INFO", f"SIM apply: {joint_name} {float(deg):.2f} deg")
        return True

    def send_real_command(self, joint_name, deg):
        if not self.real_can_write_enabled:
            self.real_can_status.set(STATE_DRY_RUN)
            self._log_event("DRY_RUN", "REAL_CAN_WRITE_ENABLED=False, CAN 송신 차단")
        payload = {
            "command": "SET_JOINT_TARGET",
            "joint": joint_name,
            "target_deg": float(self._clamp_joint_value(joint_name, deg)),
        }
        return self.send_stage8_command(payload)

    def send_stage8_command(self, payload, force_stop=False, allow_home_target=False, allow_can_command=False):
        command = str(payload.get("command", ""))
        if not force_stop and not self._can_send_command(
            command,
            allow_home_target=allow_home_target,
            allow_can_command=allow_can_command,
        ):
            self.last_reject_reason.set("blocked_by_ui_safety")
            self._log_event("SAFETY_BLOCK", f"Stage8 명령 차단: {payload}")
            return False
        if not self.stage8_ros_adapter.is_ready:
            self.last_reject_reason.set("stage8_publisher_unavailable")
            self._log_event("ERROR", "Stage8 publisher 사용 불가")
            return False

        ok, reason, message_data = self.stage8_ros_adapter.publish(payload)
        if not ok:
            self.errors.set(reason)
            self._log_event("ERROR", f"Stage8 전송 실패: {reason}")
            return False

        self.last_command.set(command)
        self._log_event("INFO", f"Stage8 명령 전송: {message_data}")
        return True

    def handle_stage8_status(self, status):
        if not isinstance(status, dict):
            return
        self.stage8_status_payload = status
        self.last_status_time = time.time()
        self._update_target_connection_state()

        self.armed = bool(status.get("armed"))
        self.armed_status.set("ARMED" if self.armed else "DISARMED")
        self.baseline_set = bool(status.get("baseline_set"))
        self.real_can_write_enabled = bool(status.get("real_can_write_enabled"))
        self.real_can_status.set("ENABLED_WARNING" if self.real_can_write_enabled else STATE_DRY_RUN)
        if not self.real_can_write_enabled:
            self._log_status_detail_once("real_can_write", "REAL_CAN_WRITE_ENABLED=False, DRY_RUN")
        self.last_command.set(self._compact_text(status.get("last_command") or "--", 28))
        self.last_reject_reason.set(self._compact_text(status.get("last_reject_reason") or "--", 80))
        self.command_seq.set(str(status.get("command_seq") or "--"))
        self._set_presence_status(self.warnings, "warnings", status.get("warnings"))
        self._set_presence_status(self.errors, "errors", status.get("errors"))
        self._refresh_can_counters_from_payload(status)

        allowed = status.get("allowed_joints")
        if isinstance(allowed, list):
            self.allowed_joints.set(f"{len(allowed)}/{len(JOINT_NAMES_12DOF)}")
            self._log_status_detail_once("allowed_joints", ",".join(str(joint_name) for joint_name in allowed))
        self._refresh_can_status_from_payload()

        joints = status.get("joints") if isinstance(status.get("joints"), dict) else {}
        self._update_joint_status(joints)
        if self.current_mode.get() == "REAL_TO_SIM":
            self._mirror_real_status_to_sim(joints)
        self._refresh_controls()

    def _on_stage8_status_message(self, message):
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.errors.set(f"status_json:{error}")
            self._log_event("ERROR", f"Stage8 status JSON decode 실패: {error}")
            return
        self.handle_stage8_status(status)

    def _update_joint_status(self, joints):
        for joint_name in JOINT_NAMES_12DOF:
            joint_status = joints.get(joint_name) if isinstance(joints, dict) else None
            if not isinstance(joint_status, dict):
                self._set_joint_row_connected(joint_name, False)
                self._configure_if_changed(
                    self.motor_status_labels[joint_name],
                    text="알 수 없음\n실 -- | 목 -- | 현 --\n기준0 --",
                    fg=MUTED_TEXT_COLOR,
                )
                continue

            actual = self._number_or_none(joint_status.get("latest_actual_deg"))
            target = self._number_or_none(joint_status.get("commanded_relative_deg"))
            baseline = self._number_or_none(joint_status.get("baseline_deg"))
            feedback_valid = bool(joint_status.get("feedback_valid"))
            feedback_stale = bool(joint_status.get("feedback_stale"))
            state = "연결" if feedback_valid and not feedback_stale and actual is not None else "알 수 없음"
            color = GREEN if state == "연결" else MUTED_TEXT_COLOR
            connected = state == "연결"
            self._set_joint_row_connected(joint_name, connected)

            actual_text = "--" if actual is None else f"{actual:.2f}"
            target_text = "--" if target is None else f"{target:.2f}"
            baseline_text = "--" if baseline is None else f"{baseline:.2f}"
            relative_text = "--" if actual is None or baseline is None else f"{actual - baseline:+.2f}"
            self._configure_if_changed(self.joint_actual_labels[joint_name], text=f"{actual_text}")
            self._configure_if_changed(
                self.motor_status_labels[joint_name],
                text=(
                    f"{state}\n"
                    f"실 {actual_text} | 목 {target_text} | 현 {relative_text}\n"
                    f"기준0 {baseline_text}"
                ),
                fg=color,
            )

    def _set_joint_row_connected(self, joint_name, connected):
        color = GREEN if connected else TEXT_COLOR
        for label in self.joint_row_labels.get(joint_name, []):
            self._configure_if_changed(label, fg=color)
        status_label = self.joint_status_labels.get(joint_name)
        if status_label is not None:
            self._configure_if_changed(status_label, fg=color)

    def _mirror_real_status_to_sim(self, joints):
        targets = {}
        for joint_name in JOINT_NAMES_12DOF:
            joint_status = joints.get(joint_name) if isinstance(joints, dict) else None
            if not isinstance(joint_status, dict):
                return False
            actual = self._number_or_none(joint_status.get("latest_actual_deg"))
            feedback_valid = bool(joint_status.get("feedback_valid"))
            feedback_stale = bool(joint_status.get("feedback_stale"))
            if actual is None or not feedback_valid or feedback_stale:
                return False
            targets[joint_name] = self._clamp_joint_value(joint_name, actual)

        payload_key = tuple((joint_name, round(targets[joint_name], 4)) for joint_name in JOINT_NAMES_12DOF)
        if payload_key == self._last_real_to_sim_mirror_payload:
            return False

        try:
            self.sim_command_adapter.write_targets_from_feedback(targets)
        except Exception as error:
            self.errors.set(str(error))
            self._log_event("ERROR", f"REAL_TO_SIM 미러 실패: {error}")
            return False

        self._last_real_to_sim_mirror_payload = payload_key
        self._log_event("INFO", "REAL_TO_SIM 실제값을 SIM command file로 미러링")
        return True

    def _start_stage8_ros_client(self):
        if self.stage8_ros_adapter.is_ready:
            return
        ok, reason = self.stage8_ros_adapter.start()
        if not ok:
            self.ros_error = reason
            self.errors.set(reason)
            self.target_status.set(STATE_FAILED)
            self.robot_state.set_target_status("Disconnected")
            self.connect_in_progress = False
            self._log_event("ERROR", f"ROS client 시작 실패: {reason}")
            return
        self._log_event("INFO", "UI ROS adapter 활성화: Stage8 topic publish/subscribe 생성")
        self._log_event("INFO", f"ROS node: {UI_ROS_NODE_NAME}")
        self._log_event("INFO", f"Stage8 명령 topic: {STAGE8_12AXIS_MIT_COMMAND_TOPIC}")
        self._log_event("INFO", f"Stage8 상태 topic: {STAGE8_12AXIS_MIT_STATUS_TOPIC}")
        if self._ros_spin_after_id is None:
            self._ros_spin_after_id = self.root.after(ROS_SPIN_INTERVAL_MS, self._spin_ros_once)

    def _spin_ros_once(self):
        if not self.stage8_ros_adapter.is_ready:
            self._ros_spin_after_id = None
            return
        ok, reason = self.stage8_ros_adapter.spin_once()
        if not ok:
            self.errors.set(reason)
            self._log_event("ERROR", f"ROS spin 실패: {reason}")
        self._ros_spin_after_id = self.root.after(ROS_SPIN_INTERVAL_MS, self._spin_ros_once)

    def _stop_stage8_ros_client(self):
        self.stage8_ros_adapter.stop()
        self._ros_spin_after_id = None

    def _read_isaac_state(self):
        try:
            with ISAAC_STATE_FILE_PATH.open("r", encoding="utf-8") as state_file:
                payload = json.load(state_file)
        except FileNotFoundError:
            return None, "isaac_state_missing"
        except json.JSONDecodeError:
            return None, "isaac_state_json_error"
        except OSError as error:
            return None, f"isaac_state_read_error:{error}"
        if not isinstance(payload, dict):
            return None, "isaac_state_not_dict"
        return payload, ""

    def _is_sim_ready(self):
        state, reason = self._read_isaac_state()
        if state is None:
            return False, reason

        timestamp = self._number_or_none(state.get("timestamp"))
        if timestamp is None:
            return False, "isaac_state_timestamp_missing"
        age = max(0.0, time.time() - timestamp)
        if age > ISAAC_STATE_STALE_SEC:
            return False, f"isaac_state_stale:{age:.1f}s"

        if not bool(state.get("simulation_playing")):
            return False, "isaac_sim_not_playing"
        if not bool(state.get("state_valid")):
            return False, f"isaac_state_invalid:{state.get('state_reason') or 'unknown'}"
        if str(state.get("state_reason") or "") != "ok":
            return False, f"isaac_state_reason:{state.get('state_reason')}"

        internal_joints = state.get("internal_joints")
        if not isinstance(internal_joints, dict):
            return False, "isaac_internal_joints_missing"
        missing = [joint_name for joint_name in JOINT_NAMES_12DOF if joint_name not in internal_joints]
        if missing:
            return False, f"isaac_internal_joints_incomplete:{len(missing)}"

        return True, "isaac_ready"

    def _mark_sim_not_ready(self, reason):
        self.connected = False
        self.target_connected_check_active = False
        self.connect_in_progress = False
        self.armed = False
        self.baseline_set = False
        self.armed_status.set("DISARMED")
        self.target_status.set(STATE_DISCONNECTED)
        self.robot_state.set_target_status("Disconnected")
        if self.robot_state.motor_enabled:
            self.command_bus.request_motor_disable()
        self.status_heartbeat.set(self._compact_text(reason, 24))

    def _can_send_command(self, command, allow_home_target=False, allow_can_command=False):
        if command == "STOP_ALL":
            return True
        mode = self.current_mode.get()
        ros_ready = self.stage8_ros_adapter.is_ready

        if allow_can_command and command == "CHECK_CAN":
            return ros_ready
        if allow_can_command and command == "RECONNECT_CAN":
            return mode in ("REAL", "SIM_TO_REAL") and ros_ready

        target_ready = self.connected

        if command == "DISARM_ALL":
            return mode in ("REAL", "SIM_TO_REAL") and ros_ready

        if self.estop_active:
            return False

        if command == "ARM_ALL":
            return mode in ("REAL", "SIM_TO_REAL") and target_ready and ros_ready
        if command == "SET_BASELINE_FROM_CURRENT_ALL":
            return (
                mode in ("REAL", "SIM_TO_REAL")
                and target_ready
                and ros_ready
            )
        if mode not in ("REAL", "SIM_TO_REAL"):
            return False
        if not self.connected and not (allow_home_target and target_ready):
            return False
        if not ros_ready:
            return False
        if command == "SET_JOINT_TARGET" and not allow_home_target and not (self.armed and self.baseline_set):
            return False
        return True

    def _sync_ui_from_runtime_snapshot(self, log_changes=False):
        """RobotState/CommandBuffer snapshot을 UI 표시 기준으로 반영한다."""
        snapshot = self.robot_state.get_snapshot()
        command_snapshot = self.command_bus.get_command_snapshot()

        runtime_mode = snapshot.get("current_mode", self.current_mode.get())
        if self.current_mode.get() != runtime_mode:
            self.current_mode.set(runtime_mode)

        self.estop_active = bool(snapshot.get("estop_active"))
        self.estop_status.set(STATE_ESTOP if self.estop_active else "OK")

        if runtime_mode == "SIM":
            ready, reason = self._is_sim_ready()
            if bool(snapshot.get("target_connected", False)) and not ready:
                self._mark_sim_not_ready(reason)
            else:
                self.connected = bool(snapshot.get("target_connected", False)) and ready
                self.target_status.set(STATE_CONNECTED if self.connected else STATE_DISCONNECTED)
            self.armed = bool(snapshot.get("motor_enabled", False)) and self.connected
            self.armed_status.set("ARMED" if self.armed else "DISARMED")
        elif self.estop_active:
            self.target_status.set(STATE_ESTOP)

        self.real_can_status.set(
            "ENABLED_WARNING" if self.real_can_write_enabled else STATE_DRY_RUN
        )
        if self.estop_active:
            self.real_can_status.set(STATE_BLOCKED_BY_SAFETY)

        targets = command_snapshot.get("joint_targets_deg", {})
        for joint_name, target_deg in targets.items():
            if joint_name in self.joint_target_labels:
                self.joint_target_labels[joint_name].configure(text=f"{float(target_deg):.1f}")

        status_key = (
            runtime_mode,
            snapshot.get("target_status"),
            snapshot.get("motor_status"),
            snapshot.get("safety_status"),
            self.target_status.get(),
            self.real_can_status.get(),
        )
        if log_changes and status_key != self._last_runtime_status_key:
            self._last_runtime_status_key = status_key
            self._log_event(
                "INFO",
                "RuntimeState snapshot "
                f"mode={runtime_mode} "
                f"target={snapshot.get('target_status')} "
                f"motor={snapshot.get('motor_status')} "
                f"safety={snapshot.get('safety_status')} "
                f"ui_target={self.target_status.get()}",
            )

    def _refresh_controls(self):
        self._refresh_status_chip_colors()
        mode = self.current_mode.get()
        for mode_name, button in self.mode_buttons.items():
            enabled = not self.mode_change_in_progress and not self.connect_in_progress
            if mode_name != mode and self.robot_state.motor_enabled:
                enabled = False
            if not enabled:
                self._configure_if_changed(
                    button,
                    state=tk.DISABLED,
                    bg=BUTTON_DISABLED,
                    fg="#6b7280",
                    relief=tk.SOLID,
                )
            elif mode_name == mode:
                self._configure_if_changed(
                    button,
                    state=tk.NORMAL,
                    bg=BUTTON_ACTIVE,
                    fg=WHITE,
                    relief=tk.SUNKEN,
                )
            else:
                self._configure_if_changed(
                    button,
                    state=tk.NORMAL,
                    bg=BUTTON_BG,
                    fg=WHITE,
                    relief=tk.RAISED,
                )

        readonly = mode == "REAL_TO_SIM"
        sim_or_real = mode in ("SIM", "REAL", "SIM_TO_REAL")
        for slider in self.joint_sliders.values():
            real_mode_blocked = mode in ("REAL", "SIM_TO_REAL") and not self.connected
            self._configure_if_changed(
                slider,
                state=tk.DISABLED
                if readonly or self.estop_active or real_mode_blocked or self.connect_in_progress
                else tk.NORMAL
            )
        for button in self.command_buttons:
            text = str(button.cget("text"))
            if text == "전체 정지":
                self._set_button_visual(button, True)
            elif text == "비상정지 해제":
                self._set_button_visual(button, True)
            elif text == "대상 연결":
                self._set_button_visual(button, not self.connect_in_progress and not self.connected)
            elif text == "연결 해제":
                self._set_button_visual(button, self.target_connected_check_active or self.connected)
            elif text == "CAN 상태확인":
                self._set_button_visual(button, True)
            elif text == "CAN 재연결":
                enabled = mode in ("REAL", "SIM_TO_REAL")
                self._set_button_visual(button, enabled)
            elif text == "제어 준비":
                enabled = (
                    mode in ("SIM", "REAL", "SIM_TO_REAL")
                    and self.connected
                    and not self.robot_state.motor_enabled
                    and not self.estop_active
                )
                self._set_button_visual(button, enabled)
            elif text == "제어 해제":
                enabled = mode in ("SIM", "REAL", "SIM_TO_REAL") and self.robot_state.motor_enabled
                self._set_button_visual(button, enabled)
            elif text == "현재자세 기준설정":
                enabled = (
                    mode in ("REAL", "SIM_TO_REAL")
                    and self.connected
                    and not self.estop_active
                )
                self._set_button_visual(button, enabled)
            elif text == "홈 0°":
                real_ready = (
                    mode == "REAL"
                    and self.connected
                )
                sim_to_real_ready = (
                    mode == "SIM_TO_REAL"
                    and self.connected
                )
                sim_ready = mode == "SIM" and self.connected and self.robot_state.motor_enabled
                real_ready = real_ready and self.robot_state.motor_enabled
                sim_to_real_ready = sim_to_real_ready and self.robot_state.motor_enabled
                enabled = not self.estop_active and (sim_ready or real_ready or sim_to_real_ready)
                self._set_button_visual(button, enabled)
            elif text == "비상정지":
                self._set_button_visual(button, True)
            else:
                self._set_button_visual(button, sim_or_real and not self.estop_active)

    def _schedule_heartbeat_refresh(self):
        self._update_target_connection_state()
        self._sync_ui_from_runtime_snapshot(log_changes=True)
        self._refresh_status_chip_colors()
        self._heartbeat_after_id = self.root.after(500, self._schedule_heartbeat_refresh)

    def _update_target_connection_state(self):
        if self.current_mode.get() == "SIM":
            ready, reason = self._is_sim_ready()
            if self.robot_state.get_target_status() == "Connected" and not ready:
                self._mark_sim_not_ready(reason)
                return
            self.connected = self.robot_state.get_target_status() == "Connected" and ready
            self.target_status.set(STATE_CONNECTED if self.connected else STATE_DISCONNECTED)
            self.status_heartbeat.set("SIM" if self.connected else "--")
            self.connect_in_progress = False
            return

        if self.last_status_time is None:
            self.status_heartbeat.set("--")
            self.connected = False
            if (
                self.target_connected_check_active
                and self.target_connect_requested_at is not None
                and time.time() - self.target_connect_requested_at > STATUS_HEARTBEAT_STALE_SEC
            ):
                elapsed = time.time() - self.target_connect_requested_at
                self.status_heartbeat.set(f"FAILED {elapsed:.1f}s")
                self.target_status.set(STATE_FAILED)
                self.robot_state.set_target_status("Disconnected")
                self.connect_in_progress = False
                self._log_status_detail_once(
                    "target_connect_failed",
                    f"target heartbeat 미수신: {elapsed:.1f}s 경과, FAILED 처리",
                )
                return
            self.target_status.set(STATE_CONNECTING if self.target_connected_check_active else STATE_DISCONNECTED)
            self.robot_state.set_target_status("Connecting" if self.target_connected_check_active else "Disconnected")
            return

        age = max(0.0, time.time() - self.last_status_time)
        if age > STATUS_HEARTBEAT_STALE_SEC:
            self.status_heartbeat.set(f"오래됨 {age:.1f}s")
        else:
            self.status_heartbeat.set(f"{age:.1f}s")

        if not self.target_connected_check_active:
            self.connected = False
            self.target_status.set(STATE_DISCONNECTED)
            self.robot_state.set_target_status("Disconnected")
            self.connect_in_progress = False
        elif (
            self.target_connect_requested_at is not None
            and self.last_status_time < self.target_connect_requested_at
        ):
            self.connected = False
            self.target_status.set(STATE_CONNECTING)
            self.robot_state.set_target_status("Connecting")
        elif age <= TARGET_CONNECTED_HEARTBEAT_SEC:
            self.connected = True
            self.target_status.set(STATE_CONNECTED)
            self.robot_state.set_target_status("Connected")
            self.connect_in_progress = False
        else:
            self.connected = False
            self.target_status.set(STATE_FAILED)
            self.robot_state.set_target_status("Disconnected")
            self.connect_in_progress = False

    def _refresh_can_status_from_payload(self):
        status = self.stage8_status_payload if isinstance(self.stage8_status_payload, dict) else {}
        can_payload = status.get("can")
        if not isinstance(can_payload, dict):
            can_payload = status.get("can_status")
        if not isinstance(can_payload, dict):
            can_payload = {}

        channel = (
            can_payload.get("channel")
            or can_payload.get("interface")
            or status.get("can_channel")
            or status.get("can_interface")
            or "can0"
        )
        state = (
            can_payload.get("status")
            or can_payload.get("state")
            or status.get("can_state")
            or status.get("can_status_text")
            or status.get("can_status")
            or "UNKNOWN"
        )
        if isinstance(state, dict):
            state = state.get("status") or state.get("state") or "UNKNOWN"

        self.can_channel.set(self._compact_text(channel, 24))
        self.can_status.set(self._translate_status_text(state))

    def _refresh_can_counters_from_payload(self, status):
        if not isinstance(status, dict):
            self.can_counts.set("tx 0 / rx 0 / skip 0")
            self.can_decode.set("--")
            return
        tx_count = int(status.get("can_tx_readonly_count") or 0)
        rx_count = int(status.get("can_rx_count") or 0)
        ignored_count = int(status.get("can_rx_ignored_count") or 0)
        self.can_counts.set(f"tx {tx_count} / rx {rx_count} / skip {ignored_count}")
        self.can_decode.set(self._compact_text(status.get("can_decode_error") or "--", 60))

    def _target_limit_for_mode(self, joint_name):
        return float(self.robot_model.get_joint(joint_name).target_limit_deg)

    def _clamp_joint_value(self, joint_name, value):
        limit = self._target_limit_for_mode(joint_name)
        value = float(value)
        return max(-limit, min(limit, value))

    def _number_or_none(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _set_joined_status(self, variable, value):
        if isinstance(value, list):
            variable.set(",".join(str(item) for item in value) or "--")
        elif value:
            variable.set(str(value))
        else:
            variable.set("--")

    def _compact_text(self, value, limit=80):
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    def _set_presence_status(self, variable, key, value):
        present = bool(value)
        variable.set("있음" if present else "없음")
        if present:
            self._log_status_detail_once(key, value)

    def _log_status_detail_once(self, key, value):
        text = self._compact_text(value, 300)
        if self._last_status_log_values.get(key) == text:
            return
        self._last_status_log_values[key] = text
        self._log(f"Stage8 상태 {key}: {text}")

    def _translate_status_text(self, value):
        raw = str(value or "UNKNOWN").upper()
        translations = {
            "UNKNOWN": "알 수 없음",
            "OK": "정상",
            "ERROR": "오류",
            "RECONNECTING": "재연결 중",
            "NOT SUPPORTED": "미지원",
            "NOT_SUPPORTED": "미지원",
            "BUS_OPEN_READ_ONLY": "읽기 버스 열림",
            "BUS_OPEN": "버스 열림",
            "LINK_UP_READ_ONLY": "링크 UP 읽기",
            "LINK_UP": "링크 UP",
            "LINK_DOWN": "링크 DOWN",
            "BRINGUP_OK_READ_ONLY": "CAN 설정 완료 읽기",
            "BRINGUP_OK": "CAN 설정 완료",
            "BRINGUP_FAILED": "CAN 설정 실패",
            "BRINGUP_OK_BUS_OPEN_FAILED": "CAN 설정 후 버스 열기 실패",
            "LINK_UP_BUS_OPEN_FAILED": "링크 UP 버스 열기 실패",
        }
        return translations.get(raw, self._compact_text(value, 40))

    def _log_event(self, level, message):
        level = str(level or "INFO").upper()
        if level not in ("INFO", "MODE", "JOINT", "WARN", "ERROR", "SAFETY_BLOCK", "DRY_RUN"):
            level = "INFO"
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] [{level}] {message}\n", level)
        self.log_line_count += 1
        while self.log_line_count > LOG_MAX_LINES:
            self.log_text.delete("1.0", "2.0")
            self.log_line_count -= 1
        self.log_text.see(tk.END)

    def _log(self, message):
        self._log_event("INFO", message)

    def _on_close(self):
        for after_id in list(self._pending_slider_after_by_joint.values()):
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._pending_slider_after_by_joint.clear()
        for after_id in (self._heartbeat_after_id, self._ros_spin_after_id,
                         getattr(self, "_rl_poll_after_id", None)):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        self._stop_stage8_ros_client()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ros",
        action="store_true",
        help="UI 시작 시 ROS2 node/publisher/subscriber를 생성",
    )
    args = parser.parse_args()

    print(f"[main_ui] --ros = {args.ros}", flush=True)

    root = tk.Tk()
    HumanoidControlUI(root, use_ros=args.ros)
    root.mainloop()


if __name__ == "__main__":
    main()
