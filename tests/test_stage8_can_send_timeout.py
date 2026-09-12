"""stage8 선택형 CAN 송신 타임아웃 회귀 (오프라인 전용).

실 CAN/ROS 를 전혀 쓰지 않는다 — 버스는 가짜 객체로 갈아끼우고(`_get_or_open_can_bus`
패치), 핸드오프 상태 파일은 존재하지 않는 임시 경로로 돌린다(실 파일 1회 소비 방지).

Run: python3 -m pytest tests/test_stage8_can_send_timeout.py -v
     (또는 python3 -m unittest discover -s tests -p test_stage8_can_send_timeout.py -v)
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402

try:
    import can as can_module
except Exception:  # pragma: no cover - python-can 없는 환경
    can_module = None

#: 종전 하드코딩 값. 이 상수가 바뀌면 '기본 동작 동일' 보장이 깨진 것.
LEGACY_SEND_TIMEOUT_SEC = 0.2

LEGACY_CAN_STATUS_KEYS = (
    "can_channel", "can_status", "can_link_state", "can_bitrate",
    "can_last_error", "can_rx_count", "can_tx_readonly_count",
    "can_rx_ignored_count", "can_decode_error",
)


class FakeBus:
    """bus.send 호출 인자만 기록하는 가짜 버스 (실 송신 없음)."""

    def __init__(self, raise_error=None):
        self.sent = []
        self.raise_error = raise_error

    def send(self, message, timeout=None):
        self.sent.append((message, timeout))
        if self.raise_error is not None:
            raise self.raise_error


class ArgParseTests(unittest.TestCase):
    """CLI 계약: 기본값·범위 검사·기존 인자 보존 (ROS/CAN 무관)."""

    def parse(self, argv):
        return s8.build_arg_parser().parse_known_args(argv)

    def test_default_argv_keeps_legacy_timeout(self):
        args, unknown = self.parse([])
        self.assertIsNone(args.can_send_timeout)   # 기본 = 상수 유지
        self.assertEqual(unknown, [])
        self.assertEqual(s8.CAN_SEND_TIMEOUT_SEC, LEGACY_SEND_TIMEOUT_SEC)

    def test_existing_cli_contract_unchanged(self):
        args, unknown = self.parse(
            ["--can-channel", "can1", "--wait-takeover",
             "--ros-args", "-r", "__ns:=/x"])
        self.assertEqual(args.can_channel, "can1")
        self.assertTrue(args.wait_takeover)
        self.assertIsNone(args.can_send_timeout)
        self.assertEqual(unknown, ["--ros-args", "-r", "__ns:=/x"])

    def test_cli_value_parsed_as_string_then_validated(self):
        args, _ = self.parse(["--can-send-timeout", "0.01"])
        self.assertEqual(s8.parse_can_send_timeout(args.can_send_timeout), 0.01)

    def test_accepted_range(self):
        for value in ("0.001", "0.01", "0.02", "0.2", "1.0", 0.5):
            with self.subTest(value=value):
                seconds = s8.parse_can_send_timeout(value)
                self.assertTrue(
                    s8.CAN_SEND_TIMEOUT_MIN_SEC <= seconds <= s8.CAN_SEND_TIMEOUT_MAX_SEC)

    def test_rejected_values(self):
        for value in ("0", "0.0009", "1.0001", "2", "-0.1", "abc", "", None,
                      "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    s8.parse_can_send_timeout(value)

    def test_boundaries_exact(self):
        self.assertEqual(s8.parse_can_send_timeout("0.001"), s8.CAN_SEND_TIMEOUT_MIN_SEC)
        self.assertEqual(s8.parse_can_send_timeout("1.0"), s8.CAN_SEND_TIMEOUT_MAX_SEC)


@unittest.skipIf(can_module is None, "python-can 미설치")
class SendPathTests(unittest.TestCase):
    """_maybe_write_frame: 타임아웃 전달·카운터·기존 실패 경로."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="stage8-send-timeout-")
        self.addCleanup(tmp.cleanup)
        # 실 핸드오프 상태 파일을 테스트가 소비(삭제)하지 못하게 격리
        p = patch.object(s8, "TAKEOVER_STATE_FILE",
                         str(Path(tmp.name) / "absent_handoff_state.json"))
        p.start()
        self.addCleanup(p.stop)
        self.core = s8.Stage8TwelveAxisMitControlCore()
        self.frame = bytes(range(8))

    def write(self, bus, timeout_sec=None):
        self.core._get_or_open_can_bus = lambda _can_module: bus
        if timeout_sec is None:
            return self.core._maybe_write_frame(3, self.frame)
        with patch.object(s8, "CAN_SEND_TIMEOUT_SEC", timeout_sec):
            return self.core._maybe_write_frame(3, self.frame)

    def test_default_send_timeout_is_legacy_value(self):
        bus = FakeBus()
        self.assertTrue(self.write(bus))
        message, timeout = bus.sent[0]
        self.assertEqual(timeout, LEGACY_SEND_TIMEOUT_SEC)   # 종전과 동일
        self.assertEqual(message.arbitration_id, 3)
        self.assertEqual(bytes(message.data), self.frame)    # 프레임 바이트 불변
        self.assertFalse(message.is_extended_id)
        self.assertEqual(self.core.can_status, "OK")
        self.assertEqual(self.core.can_send_timeouts, 0)

    def test_cli_value_propagates_to_send(self):
        bus = FakeBus()
        self.assertTrue(self.write(bus, timeout_sec=0.01))
        self.assertEqual(bus.sent[0][1], 0.01)

    def test_send_timeout_counts_and_keeps_error_path(self):
        bus = FakeBus(raise_error=can_module.CanOperationError("Transmit buffer full"))
        self.assertFalse(self.write(bus))
        self.assertEqual(self.core.can_send_timeouts, 1)
        self.assertEqual(self.core.can_status, "ERROR")      # 기존 처리 경로 유지
        self.assertIn("Transmit buffer full", self.core.can_last_error)
        self.assertIn("Transmit buffer full", self.core.errors)

    def test_non_timeout_can_error_not_counted(self):
        bus = FakeBus(raise_error=can_module.CanOperationError(
            "Couldn't send CAN frame due to OS Error: Network is down"
            " The CAN interface appears to be down."))
        self.assertFalse(self.write(bus))
        self.assertEqual(self.core.can_send_timeouts, 0)     # 정체로 오집계 금지
        self.assertEqual(self.core.can_status, "ERROR")
        self.assertTrue(self.core.errors)

    def test_non_can_exception_not_counted(self):
        bus = FakeBus(raise_error=RuntimeError("timeout-like text but not CanError"))
        self.assertFalse(self.write(bus))
        self.assertEqual(self.core.can_send_timeouts, 0)
        self.assertEqual(self.core.can_status, "ERROR")

    def test_budget_exhausted_can_error_counted(self):
        # 문구를 모르는 인터페이스: 예산을 다 쓰고 실패하면 타임아웃으로 센다
        self.assertTrue(s8._is_can_send_timeout(
            can_module.CanOperationError("unknown-wording"), 0.0099, 0.01))
        self.assertFalse(s8._is_can_send_timeout(
            can_module.CanOperationError("unknown-wording"), 0.0005, 0.01))

    def test_status_json_keys_additive(self):
        status = self.core.get_status()
        for key in LEGACY_CAN_STATUS_KEYS:
            self.assertIn(key, status)                       # 기존 키 유지
        self.assertEqual(status["can_send_timeouts"], 0)
        self.assertEqual(status["can_send_timeout_sec"], LEGACY_SEND_TIMEOUT_SEC)
        bus = FakeBus(raise_error=can_module.CanOperationError("Transmit buffer full"))
        self.write(bus)
        self.assertEqual(self.core.get_status()["can_send_timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
