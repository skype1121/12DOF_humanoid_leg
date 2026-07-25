#!/usr/bin/env python3
"""CAN-silent mock benchmark for the Stage8 12-axis MIT control node.

Runs the REAL message-processing code (json parse -> validation chain ->
joint mapping -> slew -> MIT frame packing) of
jetson/scripts/stage8_12axis_mit_control_node.py:Stage8TwelveAxisMitControlCore
under a synthetic SET_JOINT_TARGET load, with every physical transport mocked:

  * NO CAN:  sys.modules['can'] is replaced by a mock module BEFORE the node
             module is imported, and core.can_bus is pre-set to a FakeCanBus.
             The mock Bus factory RAISES if anything ever tries to open a real
             socketcan bus (call count asserted == 0 at the end), and
             socket.socket is patched to raise on AF_CAN as a second barrier.
             To avoid mock-optimism, FakeCanBus reproduces python-can's real
             user-space TX/RX work over AF_UNIX socketpairs: real
             build_can_frame packing, select() writability wait, socket send
             syscall on TX; select() + recvmsg + real dissect_can_frame + real
             can.Message construction on RX, plus python-can's per-send logging
             calls. Only the kernel CAN driver/wire is absent.
  * NO ROS:  the core class is instantiated directly (it is ROS-free by
             design). The per-message full-status echo that the real node
             performs in _on_command_message (handle_command_data +
             publish_status json.dumps + utf-8 encode, the data-copy half of
             CDR serialization) is replicated faithfully, as is the 100 Hz
             status timer (2 publishes per 20 ms window) and the 50 Hz
             control_tick. The constant per-publish rmw/DDS overhead that
             cannot run without ROS is measured by an ISOLATED subprocess
             calibration (ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1, throwaway
             topic; never the real node entrypoint, zero CAN) and reported in
             'rclpy_calibration' so the deadline margin can be corrected
             analytically ('window_busy_ms_with_ros_estimate').
  * NO real node process, NO subprocess: CHECK_CAN / RECONNECT_CAN commands
             are never sent, so the `ip link` paths are never reached.

Load pattern (per task spec):
  - 12 joints x 50 Hz synchronous burst: 12 SET_JOINT_TARGET msgs every 20 ms
    for 10 s = 6000 msgs (600 msg/s), plus a 2x margin run (1200 msg/s), plus
    a diagnostic 1x run with the per-message status echo removed.

Measured:
  - throughput (msg/s), per-20ms-window busy-time distribution (p50/p95/p99/
    max), 20 ms deadline overrun rate, per-operation microbench breakdown,
    mock CAN TX frame rate.

Run:  python3 rl_walking/deploy/bench_stage8_mock.py
"""

import importlib.util
import json
import logging
import math
import os
import select
import socket as socket_module
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NODE_PATH = PROJECT_ROOT / "jetson" / "scripts" / "stage8_12axis_mit_control_node.py"

CONTROL_PERIOD_SEC = 0.02
STATUS_PUBLISHES_PER_WINDOW = 2  # 100 Hz status timer = 2 per 20 ms window
DURATION_SEC = 10.0

# ---------------------------------------------------------------------------
# 1. Mock 'can' module — injected BEFORE the node module is imported so every
#    lazy `import can` inside the node resolves to this. The real python-can
#    library is imported ONLY to borrow its pure frame pack/unpack functions
#    and Message class (so the node pays the real per-frame construction cost);
#    can.Bus is replaced by a factory that RAISES, and socket.socket is patched
#    to refuse AF_CAN, so no CAN device can ever be opened.
# ---------------------------------------------------------------------------

REAL_BUS_CREATE_CALLS = []

# Borrow real python-can pieces (pure python; imports open no device).
import can as _real_can_pkg  # noqa: E402
from can.interfaces.socketcan.socketcan import (  # noqa: E402
    build_can_frame as real_build_can_frame,
    dissect_can_frame as real_dissect_can_frame,
)

RealCanMessage = _real_can_pkg.Message

# Safety barrier 2: any AF_CAN socket creation raises, whatever the caller.
_AF_CAN = getattr(socket_module, "AF_CAN", 29)
_ORIG_SOCKET_CLS = socket_module.socket
AF_CAN_SOCKET_ATTEMPTS = []


class _NoCanSocket(_ORIG_SOCKET_CLS):
    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        if family == _AF_CAN:
            AF_CAN_SOCKET_ATTEMPTS.append(family)
            raise RuntimeError("bench_stage8_mock: AF_CAN socket creation is blocked")
        super().__init__(family, type, proto, fileno)


socket_module.socket = _NoCanSocket


def _blocked_bus_factory(*args, **kwargs):
    REAL_BUS_CREATE_CALLS.append((args, kwargs))
    raise RuntimeError("bench_stage8_mock: real CAN Bus creation is blocked")


mock_can = types.ModuleType("can")
mock_can.Message = RealCanMessage  # real construction cost, no transport
mock_can.Bus = _blocked_bus_factory
mock_can.__bench_mock__ = True
sys.modules["can"] = mock_can

assert getattr(sys.modules["can"], "__bench_mock__", False), "can mock not active"
assert "rclpy" not in sys.modules, "rclpy must never be loaded in this benchmark"

# ---------------------------------------------------------------------------
# 2. Import the real node module by file path (imports config/mapping/protocol
#    code only; opens nothing at import time).
# ---------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("stage8_node_under_test", NODE_PATH)
node_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(node_mod)

assert sys.modules["can"] is mock_can, "can module was replaced during import"
assert node_mod.REAL_CAN_WRITE_ENABLED is True  # exercise the real TX code path
JOINTS = list(node_mod.JOINT_NAMES_12DOF)
CAN_ID_BY_JOINT = dict(node_mod.CAN_ID_BY_JOINT)
FC_REQUEST = list(node_mod.FC_READONLY_REQUEST_DATA)


# ---------------------------------------------------------------------------
# 3. FakeCanBus — drop-in for python-can Bus. Each TX (MIT command or FC read
#    request) enqueues one decodable feedback reply frame, mimicking AK motor
#    behaviour, so the node's freshness gates and RX decode path run for real.
#
#    Fidelity: reproduces python-can SocketcanBus's user-space work so the
#    mock does not undercount the real per-frame cost —
#      TX  = 2x log.debug + log.getChild + time.time + build_can_frame (struct
#            pack) + select([],[w],[]) + socket.send syscall (AF_UNIX wire)
#      RX  = select([r],[],[]) + recvmsg syscall + dissect_can_frame (struct
#            unpack) + real can.Message construction
#    Missing vs real: kernel CAN driver latency and TX-buffer backpressure
#    (bus.send(timeout=0.2) can block up to 200 ms on a congested bus).
# ---------------------------------------------------------------------------

_pycan_log = logging.getLogger("can.interfaces.socketcan.socketcan")


def _fake_feedback_frame(motor_id):
    # pos_raw=0x7FFF (~0 rad), vel_raw=0x7FF (~0), torque_raw=0x7FF (~0),
    # temp=30, error=0 -> decode_ak_mit_feedback passes, error_ok=True
    return [int(motor_id) & 0xFF, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 30, 0]


class FakeCanBus:
    def __init__(self):
        # AF_UNIX datagram socketpairs stand in for the socketcan socket:
        # same syscall pattern, no CAN device involved.
        self._tx_a, self._tx_b = socket_module.socketpair(
            socket_module.AF_UNIX, socket_module.SOCK_DGRAM)
        self._rx_a, self._rx_b = socket_module.socketpair(
            socket_module.AF_UNIX, socket_module.SOCK_DGRAM)
        for sock in (self._tx_a, self._tx_b, self._rx_a, self._rx_b):
            sock.setblocking(False)
        self.tx_mit_frames = 0
        self.tx_fc_requests = 0
        self.rx_delivered = 0
        self.rx_dropped = 0

    def queue_feedback(self, motor_id):
        frame = real_build_can_frame(RealCanMessage(
            arbitration_id=int(motor_id),
            data=_fake_feedback_frame(motor_id),
            is_extended_id=False,
        ))
        try:
            self._rx_a.send(frame)
        except (BlockingIOError, OSError):
            self.rx_dropped += 1  # RX overflow drops, like a real kernel queue

    def send(self, message, timeout=None):
        # Mirrors python-can SocketcanBus.send user-space sequence.
        _pycan_log.debug("We've been asked to write a message to the bus")
        logger_tx = _pycan_log.getChild("tx")
        logger_tx.debug("sending: %s", message)
        _started = time.time()
        if timeout is None:
            timeout = 0
        data = real_build_can_frame(message)
        ready = select.select([], [self._tx_a], [], timeout)[1]
        if not ready:
            raise RuntimeError("bench fake bus: transmit buffer full")
        self._tx_a.send(data)
        try:
            self._tx_b.recv(72)  # the "wire" consumes the TX frame
        except (BlockingIOError, OSError):
            pass
        if list(message.data) == FC_REQUEST:
            self.tx_fc_requests += 1
        else:
            self.tx_mit_frames += 1
        self.queue_feedback(message.arbitration_id)

    def recv(self, timeout=None):
        # Mirrors BusABC.recv -> _recv_internal -> capture_message.
        ready = select.select([self._rx_b], [], [], timeout if timeout else 0)[0]
        if not ready:
            return None
        try:
            cf, _ancillary, _msg_flags, _addr = self._rx_b.recvmsg(72, 64)
        except (BlockingIOError, OSError):
            return None
        can_id, can_dlc, _flags, frame_data = real_dissect_can_frame(cf)
        self.rx_delivered += 1
        return RealCanMessage(
            timestamp=time.time(),
            arbitration_id=can_id & 0x1FFFFFFF,
            is_extended_id=False,
            is_remote_frame=False,
            is_error_frame=False,
            channel="mock0",
            dlc=can_dlc,
            data=frame_data,
        )

    def shutdown(self):
        for sock in (self._tx_a, self._tx_b, self._rx_a, self._rx_b):
            try:
                sock.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 4. Core setup: ARM + fake feedback + baseline, exactly through the public
#    command/tick interfaces (no internal state shortcuts except bus injection)
# ---------------------------------------------------------------------------

def make_armed_core():
    core = node_mod.Stage8TwelveAxisMitControlCore(now_fn=time.monotonic)
    assert core.mapping_valid, f"mapping invalid: {core.errors}"
    bus = FakeCanBus()
    core.can_bus = bus  # injection point: _get_or_open_can_bus only creates when None
    core.can_bus_open_attempted = True

    st = core.handle_command_data(json.dumps({"command": "ARM_ALL"}))
    assert st["accepted"], st

    for joint in JOINTS:
        bus.queue_feedback(CAN_ID_BY_JOINT[joint])
    core.control_tick()  # decodes seeded feedback -> all joints fresh

    st = core.handle_command_data(json.dumps({"command": "SET_BASELINE_FROM_CURRENT_ALL"}))
    assert st["accepted"] and st["baseline_complete"], (
        st["last_reject_reason"], st["baseline_missing_joints"])
    return core, bus


def publish_status_mock(core):
    """Replicates node.publish_status(): full status dict + json.dumps + the
    utf-8 encode/copy that rmw CDR serialization performs on String.data.
    The remaining constant per-publish rmw/DDS cost is measured separately
    (rclpy_calibration) and applied analytically in the report."""
    payload = json.dumps(core.get_status(), ensure_ascii=False, sort_keys=True)
    return payload.encode("utf-8")


# ---------------------------------------------------------------------------
# 5. Load runs
# ---------------------------------------------------------------------------

def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def run_load(rate_multiplier, duration_sec=DURATION_SEC, per_message_echo=True):
    core, bus = make_armed_core()
    msgs_per_window = 12 * rate_multiplier
    n_windows = int(round(duration_sec / CONTROL_PERIOD_SEC))
    busy_times = []
    accepted = rejected = 0
    echo_bytes = 0
    tick_executed = 0
    tx_before = bus.tx_mit_frames

    start = time.perf_counter()
    for w in range(n_windows):
        t_phase = w * CONTROL_PERIOD_SEC
        t0 = time.perf_counter()
        # --- burst: 12*mult SET_JOINT_TARGET messages, like the ROS callback ---
        for k in range(msgs_per_window):
            joint = JOINTS[k % 12]
            target = 20.0 * math.sin(2.0 * math.pi * 0.5 * t_phase + (k % 12))
            msg = json.dumps(
                {"command": "SET_JOINT_TARGET", "joint": joint, "target_deg": target}
            )
            st = core.handle_command_data(msg)  # parse + validate + map (real code)
            if st["accepted"]:
                accepted += 1
            else:
                rejected += 1
            if per_message_echo:
                echo_bytes += len(publish_status_mock(core))  # node echoes status per msg
        # --- 50 Hz control tick: slew + 12x MIT pack + (mock) TX + RX decode ---
        st = core.control_tick()
        if st["executed"]:
            tick_executed += 1
        # --- 100 Hz status timer: 2 publishes per window ---
        for _ in range(STATUS_PUBLISHES_PER_WINDOW):
            echo_bytes += len(publish_status_mock(core))
        busy = time.perf_counter() - t0
        busy_times.append(busy)
        # fixed-rate pacing (sleep only if window finished early; no catch-up debt)
        target_t = start + (w + 1) * CONTROL_PERIOD_SEC
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)
    elapsed = time.perf_counter() - start

    total_msgs = msgs_per_window * n_windows
    s = sorted(busy_times)
    misses = sum(1 for b in busy_times if b > CONTROL_PERIOD_SEC)
    return {
        "rate_multiplier": rate_multiplier,
        "per_message_echo": per_message_echo,
        "duration_sec_wall": round(elapsed, 3),
        "windows": n_windows,
        "messages_total": total_msgs,
        "messages_accepted": accepted,
        "messages_rejected": rejected,
        "throughput_msg_per_s": round(total_msgs / elapsed, 1),
        "offered_msg_per_s": round(total_msgs / duration_sec, 1),
        "window_busy_ms": {
            "p50": round(percentile(s, 0.50) * 1000, 3),
            "p95": round(percentile(s, 0.95) * 1000, 3),
            "p99": round(percentile(s, 0.99) * 1000, 3),
            "max": round(s[-1] * 1000, 3),
            "mean": round(statistics.fmean(busy_times) * 1000, 3),
        },
        "deadline_20ms_miss": misses,
        "deadline_20ms_miss_rate_pct": round(100.0 * misses / n_windows, 2),
        "ticks_executed_tx": tick_executed,
        "mock_can_tx_mit_frames": bus.tx_mit_frames - tx_before,
        "mock_can_tx_mit_frames_per_s": round((bus.tx_mit_frames - tx_before) / elapsed, 1),
        "mock_can_fc_requests": bus.tx_fc_requests,
        "mock_can_rx_decoded_frames": core.can_rx_count,
        "mock_can_rx_dropped": bus.rx_dropped,
        "status_echo_bytes_total": echo_bytes,
        "core_errors": core.errors[-3:],
        "core_warnings": core.warnings[-3:],
    }


# ---------------------------------------------------------------------------
# 6. Microbench: per-operation cost breakdown (bottleneck attribution)
# ---------------------------------------------------------------------------

def microbench(n=2000):
    core, _bus = make_armed_core()
    msg = json.dumps({"command": "SET_JOINT_TARGET", "joint": JOINTS[0], "target_deg": 10.0})

    def timeit(fn, reps):
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        s = sorted(samples)
        return {
            "p50_us": round(percentile(s, 0.50) * 1e6, 1),
            "p95_us": round(percentile(s, 0.95) * 1e6, 1),
            "max_us": round(s[-1] * 1e6, 1),
        }

    results = {}
    results["handle_command_data_only"] = timeit(lambda: core.handle_command_data(msg), n)
    results["get_status_dict_build"] = timeit(lambda: core.get_status(), n)
    results["publish_status_full_json"] = timeit(lambda: publish_status_mock(core), n)
    results["control_tick_12joint"] = timeit(lambda: core.control_tick(), 500)
    sample_status = publish_status_mock(core)
    results["status_json_size_bytes"] = len(sample_status)
    return results


# ---------------------------------------------------------------------------
# 7. rclpy publish/deliver calibration — ISOLATED subprocess.
#    Quantifies the per-message rmw/DDS overhead the mock cannot replicate:
#    String() build + .data assign + publish() (serialize), and full
#    pub->sub->callback delivery, for both the 11.6 KB status echo and the
#    ~85 B SET_JOINT_TARGET command. Runs on a throwaway topic with
#    ROS_DOMAIN_ID=77 + ROS_LOCALHOST_ONLY=1 so it can never touch the real
#    robot graph; involves no CAN of any kind; is NOT the real node entrypoint.
# ---------------------------------------------------------------------------

_RCLPY_CALIB_CODE = r'''
import json, statistics, sys, time
import rclpy
from std_msgs.msg import String

def pct(s, q):
    s = sorted(s)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]

rclpy.init()
node = rclpy.create_node("bench_stage8_pub_calib_ephemeral")
pub = node.create_publisher(String, "/bench_stage8_pub_calib_topic", 10)
received = []
sub = node.create_subscription(
    String, "/bench_stage8_pub_calib_topic", lambda m: received.append(len(m.data)), 10)

# wait for pub<->sub discovery
deadline = time.monotonic() + 5.0
while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.05)

out = {"discovered": pub.get_subscription_count() >= 1}
payloads = {"status_11637B": "x" * 11637,
            "command_85B": json.dumps({"command": "SET_JOINT_TARGET",
                                       "joint": "hip_pitch_l", "target_deg": 12.34567})}
N = 2000
for name, payload in payloads.items():
    # (a) publish-side cost only: String build + data assign + publish()
    samples = []
    for _ in range(N):
        t0 = time.perf_counter()
        msg = String(); msg.data = payload; pub.publish(msg)
        samples.append(time.perf_counter() - t0)
        if len(received) > 3000:
            received.clear()
        rclpy.spin_once(node, timeout_sec=0)  # drain so history never saturates
    s = sorted(samples)
    out[name + "_publish_us"] = {"p50": round(pct(s, .5) * 1e6, 1),
                                 "p95": round(pct(s, .95) * 1e6, 1),
                                 "max": round(s[-1] * 1e6, 1)}
    # settle: drain every in-flight message before round-trip timing
    settle_end = time.monotonic() + 0.5
    while time.monotonic() < settle_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    received.clear()
    # (b) full delivery: publish -> DDS -> subscription callback consumed
    samples = []
    for _ in range(500):
        n0 = len(received)
        t0 = time.perf_counter()
        msg = String(); msg.data = payload; pub.publish(msg)
        while len(received) == n0:
            rclpy.spin_once(node, timeout_sec=0.05)
        samples.append(time.perf_counter() - t0)
    s = sorted(samples)
    out[name + "_pub_to_callback_us"] = {"p50": round(pct(s, .5) * 1e6, 1),
                                         "p95": round(pct(s, .95) * 1e6, 1),
                                         "max": round(s[-1] * 1e6, 1)}
node.destroy_node()
rclpy.shutdown()
print(json.dumps(out))
'''


def run_rclpy_calibration():
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = "77"          # isolated from any real robot graph
    env["ROS_LOCALHOST_ONLY"] = "1"      # never leaves this machine
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RCLPY_CALIB_CODE],
            capture_output=True, text=True, timeout=120, env=env, check=False,
        )
    except Exception as error:  # calibration is best-effort, never fatal
        return {"ok": False, "error": str(error)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "")[-500:]}
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as error:
        return {"ok": False, "error": f"parse:{error}", "raw": proc.stdout[-500:]}
    data["ok"] = True
    data["note"] = ("isolated subprocess, ROS_DOMAIN_ID=77, localhost-only, "
                    "throwaway topic; intra-host DDS — a remote subscriber adds "
                    "network cost on the subscriber side, not in this node's loop")
    return data


def ros_adjusted_window_busy(report):
    """Analytic correction for the rmw/DDS work the mock loop cannot run.
    Per 20 ms window at 1x load, add:
      - (12 echo + 2 timer) status publishes x measured publish() cost
        (String build + data assign + serialize) — paid in the node process;
      - 12 inbound command deliveries x measured full pub->callback cost.
        Upper bound: that figure also contains the sender's publish side,
        which in reality is paid by the commanding process, not this node."""
    calib = report.get("rclpy_calibration") or {}
    if not calib.get("ok"):
        return None
    stat_pub = calib.get("status_11637B_publish_us", {})
    cmd_rt = calib.get("command_85B_pub_to_callback_us", {})
    if not stat_pub or not cmd_rt:
        return None
    out = []
    for run in report["runs"]:
        mult = run["rate_multiplier"]
        echoes = (12 * mult if run["per_message_echo"] else 0) + STATUS_PUBLISHES_PER_WINDOW
        inbound = 12 * mult
        for level in ("p50", "p95"):
            extra_ms = (echoes * stat_pub[level] + inbound * cmd_rt[level]) / 1000.0
            run.setdefault("window_busy_ms_with_ros_estimate", {})[level] = round(
                run["window_busy_ms"][level] + extra_ms, 3)
        out.append(run["window_busy_ms_with_ros_estimate"])
    return out


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    report = {
        "bench": "stage8_mock_no_can",
        "node_file": str(NODE_PATH),
        "python": sys.version.split()[0],
        "safety": {},
        "microbench": microbench(),
        "runs": [],
    }
    report["runs"].append(run_load(rate_multiplier=1))               # 600 msg/s
    report["runs"].append(run_load(rate_multiplier=2))               # 1200 msg/s
    report["runs"].append(run_load(rate_multiplier=1,                # echo removed
                                   per_message_echo=False))

    report["rclpy_calibration"] = run_rclpy_calibration()
    ros_adjusted_window_busy(report)

    report["safety"] = {
        "mock_can_module_active": getattr(sys.modules["can"], "__bench_mock__", False),
        "real_bus_create_attempts": len(REAL_BUS_CREATE_CALLS),
        "af_can_socket_attempts": len(AF_CAN_SOCKET_ATTEMPTS),
        "rclpy_loaded": "rclpy" in sys.modules,
        "socketcan_opened": False,
    }
    assert report["safety"]["real_bus_create_attempts"] == 0
    assert report["safety"]["af_can_socket_attempts"] == 0
    assert report["safety"]["rclpy_loaded"] is False

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
