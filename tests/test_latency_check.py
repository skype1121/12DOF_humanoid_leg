"""latency_check 판정 로직 회귀 (오프라인 — ROS·CAN 없음, 합성 status 열).

지킬 계약:
  1. 정상 열은 PASS, 각 이상(틱 지터·응답 누락·stale·CAN 오류·옵션 불일치·재기동)은 FAIL.
  2. 측정 불가(짧음·송신 없음·표본 없음)는 INCONCLUSIVE — PASS 로 치지 않는다.
  3. 기준 상수는 사전 고정값(문서·배포 게이트와 같은 값).
  4. 도구는 읽기 전용 — 퍼블리셔·CAN 접근 코드가 없다.

Run: python3 -m pytest tests/test_latency_check.py -q
"""
import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import latency_check as L  # noqa: E402


def run(expect="off", **kw):
    s = L.summarize(L.synth_samples(**kw), expect)
    v, rows, incon = L.judge(s)
    return v, {r[0]: r for r in rows}, incon, s


class CriteriaTest(unittest.TestCase):
    def test_pre_registered_values(self):
        c = L.CRITERIA
        self.assertEqual(c["tick_p99_ms_max"], 25.0)
        self.assertEqual(c["tick_out_of_band_frac_max"], 0.02)
        self.assertEqual(c["fas_complete_frac_min"], 0.99)
        self.assertEqual(c["stale_samples_max"], 0)
        self.assertEqual(c["can_send_timeouts_delta_max"], 0)
        self.assertEqual(c["can_decode_error_samples_max"], 0)
        self.assertEqual(L.CRITERIA_VERSION, "lat_1006_v1")


class VerdictTest(unittest.TestCase):
    def test_base_pass_and_fas_row_na(self):
        v, rows, incon, s = run("off")
        self.assertEqual(v, "PASS", incon)
        self.assertEqual(rows["fas 같은 틱 응답 완결률"][3], "N/A")
        self.assertAlmostEqual(s["rx_per_send"], 1.0, places=3)

    def test_fas_pass(self):
        v, rows, _, s = run("fas", fas=True)
        self.assertEqual(v, "PASS")
        self.assertEqual(rows["fas 같은 틱 응답 완결률"][3], "PASS")
        self.assertGreaterEqual(s["fas_complete_frac"], 0.99)

    def test_fas_soc_expectation(self):
        self.assertEqual(run("fas,soc", fas=True, soc=True)[0], "PASS")
        self.assertEqual(run("fas,soc", fas=True, soc=False)[0], "FAIL")

    def test_option_mismatch_fails(self):
        v, rows, _, _ = run("fas", fas=False)   # 파일은 바꿨는데 stage8 재기동 안 함
        self.assertEqual(v, "FAIL")
        self.assertEqual(rows["실제 로드 옵션 = 기대"][3], "FAIL")
        self.assertEqual(run("off", fas=True)[0], "FAIL")

    def test_missed_responses(self):
        v, rows, _, s = run("fas", fas=True, missed_every=50)   # 2 % 틱 미완결
        self.assertEqual(v, "FAIL")
        self.assertEqual(rows["fas 같은 틱 응답 완결률"][3], "FAIL")
        v, rows, _, _ = run("off", missed_every=5)               # base 응답 누락
        self.assertEqual(rows["응답/송신 (missed response)"][3], "FAIL")

    def test_stale_and_feedback_age(self):
        v, rows, _, s = run("off", stale_at=12.0)
        self.assertEqual(v, "FAIL")
        self.assertGreater(s["stale_samples"], 0)
        samples = L.synth_samples()
        samples[100][1]["joints"]["j3"]["feedback_age_sec"] = 0.25
        s = L.summarize(samples, "off")
        v, rows, _ = L.judge(s)
        self.assertEqual(v, "FAIL")
        self.assertEqual(s["feedback_age_by_joint"]["j3"]["max"], 0.25)

    def test_can_errors(self):
        self.assertEqual(run("off", timeouts_at=5.0)[0], "FAIL")
        samples = L.synth_samples()
        samples[50][1]["can_decode_error"] = "frame_shape_reject:arb3:d0_7"
        samples[60][1]["can_status"] = "ERROR"
        samples[60][1]["can_last_error"] = "Transmit buffer full"
        s = L.summarize(samples, "off")
        v, rows, _ = L.judge(s)
        self.assertEqual(v, "FAIL")
        self.assertEqual(s["can_decode_error_samples"], 1)
        self.assertEqual(s["can_status_abnormal_samples"], 1)
        self.assertIn("Transmit buffer full", s["can_last_error_values"])
        samples = L.synth_samples()
        samples[-1][1]["errors"] = ["can_send_failed:x"]
        self.assertEqual(L.judge(L.summarize(samples, "off"))[0], "FAIL")

    def test_jitter(self):
        self.assertEqual(run("off", tick_p99=25.5)[0], "FAIL")
        self.assertEqual(run("off", tick_p99=24.9)[0], "PASS")
        self.assertEqual(run("off", tick_max=60.0)[0], "FAIL")
        self.assertEqual(run("off", oob=0.03)[0], "FAIL")

    def test_send_rate(self):
        self.assertEqual(run("off", hz=40.0)[0], "FAIL")   # control_hz≠50 이면 옵션 불일치로도 FAIL
        v, rows, _, _ = run(None, hz=40.0)
        self.assertEqual(rows["송신 틱/초"][3], "PASS")       # 송신 주기 자체는 control_hz 대비

    def test_status_gap(self):
        samples = [x for x in L.synth_samples() if not (10.0 <= x[0] < 10.3)]
        v, rows, _ = L.judge(L.summarize(samples, "off"))
        self.assertEqual(v, "FAIL")
        self.assertEqual({r[0]: r for r in rows}["status 수신 공백 최대 s"][3], "FAIL")

    def test_restart_detected(self):
        self.assertEqual(run("off", restart_at=15.0)[0], "FAIL")

    def test_inconclusive(self):
        self.assertEqual(run("off", duration=10.0)[0], "INCONCLUSIVE")
        samples = L.synth_samples()
        for _, st in samples:       # 대기 중(송신 없음)
            st["send_counts"] = {"timer": 0, "command": 0, "watchdog": 0, "hold": 0}
            st["can_rx_count"] = 0
        v, _, incon = L.judge(L.summarize(samples, "off"))
        self.assertNotEqual(v, "PASS")
        self.assertTrue(any("송신 틱" in m for m in incon))
        self.assertEqual(L.judge(L.summarize([], "off"))[0], "INCONCLUSIVE")

    def test_hold_state(self):
        v, _, _, s = run("off", armed=False)
        self.assertEqual(v, "PASS")
        self.assertEqual(s["state"], "takeover_hold")
        self.assertEqual(s["send_ticks_by_source"]["timer"], 0)

    def test_save_json_last_copy(self):
        with tempfile.TemporaryDirectory() as d:
            last = L.save_json(str(Path(d) / "x.json"), {"verdict": "PASS"}, "fas,soc")
            self.assertTrue(last.endswith("last_fas_soc.json"))
            self.assertEqual(json.load(open(last))["verdict"], "PASS")

    def test_selftest(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(L.selftest(), 0)
        self.assertIn("ALL PASS", buf.getvalue())


class ReadOnlyTest(unittest.TestCase):
    def test_no_publisher_or_can(self):
        src = (PROJECT_ROOT / "jetson/scripts/latency_check.py").read_text()
        tree = ast.parse(src)
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn("create_publisher", names)
        self.assertNotIn("publish", names)
        imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("can", imports)
        self.assertNotIn("subprocess", imports)
        self.assertNotIn("can", mods)


if __name__ == "__main__":
    unittest.main()
