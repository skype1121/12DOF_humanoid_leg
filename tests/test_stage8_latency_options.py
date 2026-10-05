"""stage8 1006 지연 단축 옵션 회귀 (오프라인 전용 — 실 CAN·ROS 없음, 가짜 버스).

지킬 계약:
  1. 기본(인자 없음) = 종전 동작: 옵션 전부 꺼짐, 슬루 틱당 4° 고정, 송신 후 대기 없음.
  2. --feedback-after-send: 보낸 모터의 응답을 같은 틱 안에서 읽는다(대기 상한 준수).
  3. --send-on-command / --control-hz 100: 슬루가 시간 기준(200°/s, 틱당 ≤ 4°)으로 바뀌어
     송신이 잦아져도 시간당 슬루가 종전보다 커지지 않는다.
  4. 즉시 송신·배경 송신 판정: armed·배치 목표·최소 간격, 명령 두절 시 배경 송신 재개.
  5. 인자 검증: 대기 범위·주기 절반 상한·허용 주파수.

Run: python3 -m pytest tests/test_stage8_latency_options.py -q
"""
import math
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402

try:
    import can as can_module
except Exception:  # pragma: no cover
    can_module = None

OPTION_GLOBALS = ("FEEDBACK_AFTER_SEND", "FEEDBACK_WAIT_SEC", "SEND_ON_COMMAND", "CONTROL_PERIOD_SEC")


class FakeBus:
    """send 를 기록하고, reply=True 면 명령 위치를 그대로 담은 AK 응답을 recv 큐에 넣는다."""

    def __init__(self, reply=True):
        self.sent = []
        self.rx = []
        self.reply = reply
        self.recv_timeouts = []

    def send(self, message, timeout=None):
        self.sent.append(message)
        if self.reply:
            mid = int(message.arbitration_id)
            d = list(message.data)
            self.rx.append(can_module.Message(arbitration_id=mid, is_extended_id=False,
                                              data=[mid, d[0], d[1], 0x7F, 0xF7, 0xFF, 30, 0]))

    def recv(self, timeout=None):
        self.recv_timeouts.append(timeout)
        if self.rx:
            return self.rx.pop(0)
        if timeout:
            time.sleep(min(float(timeout), 0.02))
        return None


def mid_soft_motor_deg(joint):
    lim = s8.get_absolute_limits_deg()[joint]
    return (s8.STAND_ZERO_DEG_BY_JOINT[joint]
            + s8.DIRECTION_BY_JOINT[joint] * 0.5 * (lim["soft_min"] + lim["soft_max"]))


@unittest.skipIf(can_module is None, "python-can 없음")
class LatencyOptionsTest(unittest.TestCase):
    def setUp(self):
        saved = {k: getattr(s8, k) for k in OPTION_GLOBALS}
        self.addCleanup(lambda: [setattr(s8, k, v) for k, v in saved.items()])
        tmp = tempfile.TemporaryDirectory(prefix="stage8-latency-")
        self.addCleanup(tmp.cleanup)
        p = patch.object(s8, "TAKEOVER_STATE_FILE", str(Path(tmp.name) / "absent.json"))
        p.start()
        self.addCleanup(p.stop)
        self.t = 1000.0
        self.bus = FakeBus()
        self.core = s8.Stage8TwelveAxisMitControlCore(now_fn=lambda: self.t)
        self.core.can_bus = self.bus
        p2 = patch.object(s8.Stage8TwelveAxisMitControlCore, "_get_or_open_can_bus",
                          lambda core, mod: self.bus)
        p2.start()
        self.addCleanup(p2.stop)
        for j in s8.JOINT_NAMES_12DOF:
            self.core.latest_actual_deg_by_joint[j] = mid_soft_motor_deg(j)
            self.core.latest_feedback_time_by_joint[j] = self.t
        self.core.latest_feedback_time = self.t
        self.core._set_baseline_from_current_all()
        st = self.core.handle_command({"command": s8.COMMAND_ARM_ALL})
        self.assertTrue(self.core.armed, st.get("last_reject_reason"))

    def apply(self, argv):
        args, _ = s8.build_arg_parser().parse_known_args(argv)
        s8.apply_latency_args(args)

    # 1. 기본 = 종전
    def test_defaults_are_legacy(self):
        self.apply([])
        self.assertFalse(s8.FEEDBACK_AFTER_SEND)
        self.assertFalse(s8.SEND_ON_COMMAND)
        self.assertEqual(s8.CONTROL_PERIOD_SEC, 0.02)
        self.assertFalse(s8.time_based_slew_enabled())
        opts = self.core.get_status()["latency_options"]
        self.assertEqual((opts["feedback_after_send"], opts["send_on_command"], opts["control_hz"]),
                         (False, False, 50.0))

    def test_default_slew_is_4deg_per_tick_regardless_of_timing(self):
        self.apply([])
        j = s8.JOINT_NAMES_12DOF[0]
        self.core.desired_relative_deg_by_joint[j] = 10.0
        self.core.control_tick()
        self.core.control_tick()          # 몇 µs 뒤 — 종전 규약은 시간 무관 틱당 4°
        self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[j], 8.0)

    def test_default_does_not_wait_for_replies(self):
        self.apply([])
        self.bus.rx.clear()
        self.bus.reply = False
        self.core.control_tick()
        self.assertEqual(len(self.bus.sent), 12)
        self.assertTrue(all(to == 0.0 for to in self.bus.recv_timeouts))   # 종전 비블로킹 폴링만
        self.assertEqual(self.core.fb_after_send["ticks"], 0)

    # 2. 송신 직후 피드백
    def test_feedback_after_send_reads_replies_same_tick(self):
        self.apply(["--feedback-after-send"])
        self.t += 0.02
        self.core.control_tick()
        for j in s8.JOINT_NAMES_12DOF:
            self.assertEqual(self.core.latest_feedback_time_by_joint[j], self.t)
        self.assertEqual(self.bus.rx, [])
        self.assertEqual(self.core.fb_after_send["complete"], 1)
        self.assertEqual(self.core.fb_after_send["replies"], 12)

    def test_feedback_after_send_wait_is_bounded(self):
        self.apply(["--feedback-after-send", "--feedback-wait-ms", "2"])
        self.bus.reply = False
        t0 = time.monotonic()
        self.core.control_tick()
        waited = time.monotonic() - t0
        self.assertLess(waited, 0.002 + 0.025)     # 상한 2 ms (+ 가짜 버스 sleep 1회·스케줄 여유)
        self.assertEqual(self.core.fb_after_send["complete"], 0)
        self.assertTrue(all(to is not None and to <= 0.002 + 1e-9
                            for to in self.bus.recv_timeouts if to))

    # 3. 시간 기준 슬루
    def test_time_based_slew_limits_rate(self):
        self.apply(["--send-on-command"])
        self.assertTrue(s8.time_based_slew_enabled())
        j = s8.JOINT_NAMES_12DOF[0]
        self.core.desired_relative_deg_by_joint[j] = 30.0
        clock = [500.0]
        with patch.object(s8.time, "monotonic", lambda: clock[0]):
            self.core.control_tick(source="command")     # 첫 송신: dt = 주기 → 4°
            self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[j], 4.0)
            clock[0] += 0.005
            self.core.control_tick(source="command")     # 5 ms 뒤 → 1°
            self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[j], 5.0)
            clock[0] += 0.100
            self.core.control_tick(source="watchdog")    # 오래 쉬어도 틱당 4° 상한
            self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[j], 9.0)

    def test_time_based_slew_never_exceeds_legacy_rate(self):
        for argv in (["--send-on-command"], ["--control-hz", "100"],
                     ["--send-on-command", "--control-hz", "100"]):
            self.apply(argv)
            self.core.last_armed_send_mono = None
            j = s8.JOINT_NAMES_12DOF[3]
            self.core.commanded_relative_deg_by_joint[j] = 0.0
            self.core.desired_relative_deg_by_joint[j] = 40.0
            rng = random.Random(7)
            clock = [100.0]
            hist = []
            with patch.object(s8.time, "monotonic", lambda: clock[0]):
                for _ in range(200):
                    clock[0] += rng.uniform(0.0005, 0.03)
                    self.core.control_tick(source="command")
                    hist.append((clock[0], self.core.commanded_relative_deg_by_joint[j]))
                    if hist[-1][1] >= 40.0:
                        break
            for (ta, va), (tb, vb) in zip(hist, hist[1:]):
                self.assertLessEqual(vb - va, s8.MAX_TARGET_SLEW_DEG_PER_SEC * (tb - ta) + 1e-9, argv)
                self.assertLessEqual(vb - va, s8.MAX_TARGET_STEP_DEG_PER_TICK + 1e-9, argv)

    # 4. 즉시 송신·배경 송신 판정
    def test_should_send_on_command_rules(self):
        self.apply([])
        msg = {"command": "SET_JOINT_TARGETS", "targets": {s8.JOINT_NAMES_12DOF[0]: 0.5}}
        self.core.handle_command(msg)
        self.assertFalse(self.core.should_send_on_command())          # 옵션 꺼짐
        self.apply(["--send-on-command"])
        self.core.handle_command(msg)
        self.assertTrue(self.core.accepted)
        self.assertTrue(self.core.should_send_on_command(now_mono=10.0))
        self.core.last_command_send_mono = 10.0
        self.assertFalse(self.core.should_send_on_command(now_mono=10.002))   # 최소 간격 4 ms
        self.assertTrue(self.core.should_send_on_command(now_mono=10.0041))
        self.core.handle_command({"command": "STOP_ALL"})
        self.assertFalse(self.core.should_send_on_command(now_mono=11.0))     # 배치 목표만
        self.core.handle_command({"command": "DISARM_ALL"})
        self.core.handle_command(msg)
        self.assertFalse(self.core.should_send_on_command(now_mono=12.0))     # 비무장 거부

    def test_background_send_50hz(self):
        self.apply(["--send-on-command"])
        c = self.core
        c.last_armed_send_mono = c.last_command_send_mono = 50.0
        self.assertFalse(c.wants_background_send(now_mono=50.020))   # 명령 스트림 살아 있음 → 명령이 송신
        self.assertFalse(c.wants_background_send(now_mono=50.024))
        self.assertTrue(c.wants_background_send(now_mono=50.0251))   # 25 ms 두절 → 배경 송신 재개
        c.last_armed_send_mono = 50.030                               # 배경 송신 뒤엔 종전 주기
        self.assertFalse(c.wants_background_send(now_mono=50.045))
        self.assertTrue(c.wants_background_send(now_mono=50.0476))
        c.armed = False
        self.assertFalse(c.wants_background_send(now_mono=60.0))     # 비무장 = 종전 타이머 경로

    def test_background_send_100hz_refreshes_mid_cycle(self):
        self.apply(["--send-on-command", "--feedback-after-send", "--control-hz", "100"])
        c = self.core
        c.last_armed_send_mono = c.last_command_send_mono = 50.0
        self.assertFalse(c.wants_background_send(now_mono=50.005))
        self.assertTrue(c.wants_background_send(now_mono=50.0076))   # 명령 사이 중간 재송신

    # 5. 인자 검증
    def test_arg_validation(self):
        for argv in (["--feedback-after-send", "--feedback-wait-ms", "0.1"],
                     ["--feedback-after-send", "--feedback-wait-ms", "9"],
                     ["--feedback-after-send", "--feedback-wait-ms", "nan"],
                     ["--feedback-after-send", "--feedback-wait-ms", "6", "--control-hz", "100"]):
            args, _ = s8.build_arg_parser().parse_known_args(argv)
            with self.assertRaises(ValueError, msg=argv):
                s8.apply_latency_args(args)
        with self.assertRaises(SystemExit):
            s8.build_arg_parser().parse_args(["--control-hz", "200"])
        self.apply(["--feedback-after-send", "--feedback-wait-ms", "4", "--control-hz", "100"])
        self.assertTrue(math.isclose(s8.FEEDBACK_WAIT_SEC, 0.004))
        self.assertTrue(math.isclose(s8.CONTROL_PERIOD_SEC, 0.01))


if __name__ == "__main__":
    unittest.main()
