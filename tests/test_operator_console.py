"""Offline console regressions. Every hardware/subprocess path is blocked/mocked.

Run: python3 -m unittest discover -s tests -p test_operator_console.py -v
"""
import collections
import copy
import json
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from jetson.scripts import operator_console as c
from jetson.scripts.console_demo import Demo
from jetson.scripts.console_state import TrialState


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="console-tests-")
        self.addCleanup(self.tmp.cleanup)
        self.demo = Demo()
        self.session = c.SessionLog()
        self.trial = TrialState(c.CKPT_DEFAULT)
        self.patches = patch.dict(c.__dict__, DEMO=True, DEMO_SOURCE=self.demo,
            TRIAL=self.trial, SESSION=self.session, SESSION_DIR=self.tmp.name,
            TEL=c.Telemetry(), RUN=c.Runner(), BUSY=False, REQUESTS=collections.OrderedDict(),
            PROCESS_HEALTH={"ts": c.now(), "external": [], "error": None})
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.addCleanup(self.session.close)
        for function in ("run", "Popen"):
            p = patch.object(c.subprocess, function, side_effect=AssertionError("real process forbidden"))
            p.start()
            self.addCleanup(p.stop)

    def payload(self, action, **extra):
        state = c.console_snapshot()["trial"]
        return {"action": action, "boot_id": state["boot_id"], "revision": state["revision"],
                "request_id": c.uuid.uuid4().hex, "confirmed": True, **extra}

    def request(self, action, **extra):
        return c.dispatch_action(self.payload(action, **extra))

    def step(self, action, **extra):
        code, response = self.request(action, **extra)
        self.assertEqual(code, 200, (action, response))
        self.assertTrue(response["ok"], (action, response))
        c.console_snapshot()

    def prepare(self, mode="ground", until="stand"):
        self.step("configure", mode=mode, checkpoint=c.CKPT_DEFAULT)
        for action in self.trial.flow():
            if action == "ground" and mode == "ground":
                self.step("op_touchdown")
            self.step(action)
            self.assertIn(action, self.trial.completed, (action, self.trial.pending, self.trial.fault))
            if action == until:
                break

    def test_ground_flow_checks_actual_policy_and_requires_manual_slack(self):
        self.prepare()
        self.assertFalse(self.trial.fault)
        self.assertEqual(self.trial.measurement["state"], "waiting")
        self.assertIsNone(self.trial.slack_at)
        self.assertEqual(c.console_snapshot()["bridge"]["loaded_checkpoint"], c.CKPT_DEFAULT)

    def test_air_flow_no_contact_until_touchdown(self):
        self.prepare("air")
        self.assertEqual(self.demo.contact, [False, False])
        self.assertEqual(self.request("op_slack_done")[0], 409)
        self.step("op_touchdown")
        self.step("op_slack_done")
        self.assertEqual(self.trial.measurement["state"], "measuring")

    def test_force_values_do_not_gate_contact(self):
        telemetry = c.Telemetry()
        telemetry.receive("foot", {"timestamp": c.now(), "left_contact": True, "right_contact": False,
                                  "left_n": float("nan"), "right_n": 9999999})
        view = telemetry.view()
        self.assertTrue(view["fresh"]["foot"])
        self.assertEqual(view["foot"]["contact"], [True, False])

    def test_legacy_force_cannot_fabricate_contacts(self):
        telemetry = c.Telemetry()
        telemetry.receive("foot", {"timestamp": c.now(), "left_n": 1000, "right_n": 1000})
        self.assertFalse(telemetry.view()["fresh"]["foot"])
        self.assertIsNone(telemetry.view()["foot"]["contact"])

    def test_nested_takeover_contract_and_bad_baseline(self):
        telemetry = c.Telemetry()
        telemetry.receive("status", {"timestamp": c.now(), "takeover": {"waiting": True, "holding": False},
                                     "joints": {"x": {"baseline_deg": float("nan")}}})
        view = telemetry.view()["stage8"]
        self.assertTrue(view["waiting"])
        self.assertFalse(view["holding"])
        self.assertEqual(view["baseline"], 0)
        self.assertFalse(view["feedback_ok"])

    def test_handoff_allowed_before_takeover_without_joint_feedback_R9(self):
        """0912 현장(R9): 인계 전 stage8 은 --wait-takeover(CAN 침묵)라 관절 피드백이 없다 — 인계 게이트는
        waiting+status 신선+kp150 홀드만 요구해야 하고, 12관절 신선은 인계 뒤 단계(기준점·ARM·정책 로드·발사)에만 요구한다."""
        self.prepare(until="ground")
        view = Demo().snapshot()
        now = view.get("server_time") or c.now()
        view["server_time"] = now
        view["session"] = {"active": True}
        view["fresh"] = dict.fromkeys(("status", "imu", "foot"), True); view["fresh"]["bridge"] = False
        view["stage8"].update(waiting=True, holding=False, armed=False, baseline=0, feedback_ok=False, joints={})
        view.setdefault("procs", {})["hang_stand"] = {"alive": True, "hold": {"kp": 150.0, "ts": now}}
        view["procs"]["stage8"] = {"alive": True}
        view["foot"]["contact"] = [True, True]
        self.trial.update(view)
        self.assertIsNone(self.trial.reason("handoff"), self.trial.reason("handoff"))
        # 인계 뒤: 홀드 상태이지만 12관절 피드백이 없으면 기준점은 막혀야 한다
        self.trial.completed.add("handoff")
        view["stage8"].update(waiting=False, holding=True)
        view["procs"]["hang_stand"] = {"alive": False, "rc": 0}   # 인계 뒤 자세 유지 스크립트는 종료됨(겹침 래치 회피)
        self.trial.update(view)
        self.assertIsNotNone(self.trial.reason("baseline"))
        self.assertIn("피드백", self.trial.reason("baseline"))

    def test_stale_missing_future_and_nonfinite_source_timestamps(self):
        for stamp in (None, c.now()-2, c.now()+10, float("nan")):
            with self.subTest(stamp=stamp):
                telemetry = c.Telemetry()
                telemetry.receive("foot", {"timestamp": stamp, "left_contact": True, "right_contact": True})
                self.assertFalse(telemetry.view()["fresh"]["foot"])
                self.assertIsNone(telemetry.view()["foot"]["contact"])

    def test_invalid_imu_is_not_healthy_zero(self):
        for quat in ([0, 0, 0, 0], [1, float("nan"), 0, 0], [1, 0]):
            telemetry = c.Telemetry()
            telemetry.receive("imu", {"timestamp": c.now(), "quat_wxyz": quat, "gyro_rad_s": [0, 0, 0]})
            self.assertFalse(telemetry.view()["fresh"]["imu"])
            self.assertIsNone(telemetry.view()["imu"])

    def test_bridge_event_latches_without_replacing_snapshot_age(self):
        telemetry = c.Telemetry()
        telemetry.receive("bridge", {"timestamp": c.now(), "checkpoint": c.CKPT_DEFAULT, "armed": True})
        stamp = telemetry.t["bridge"]
        telemetry.receive("bridge", {"event": "stop_all", "reason": "osc_guard"})
        telemetry.receive("bridge", {"event": "informational", "reason": "later event"})
        self.assertEqual(telemetry.t["bridge"], stamp)
        view = telemetry.view()["bridge"]
        self.assertEqual(view["checkpoint"], c.CKPT_DEFAULT)
        self.assertEqual(view["fault"], "osc_guard")

    def test_skipped_arm_and_legacy_actions_rejected(self):
        self.assertEqual(self.request("arm")[0], 409)
        for name in ("march", "trim", "zero_recapture", "disarm", "posture", "rm"):
            self.assertEqual(self.request(name)[0], 403)

    def test_idempotency_and_stale_tabs_and_boot(self):
        payload = self.payload("can_up")
        other_tab = self.payload("can_up")
        with patch.object(self.demo, "act", wraps=self.demo.act) as execute:
            result = c.dispatch_action(payload)
            self.assertEqual(c.dispatch_action(payload), result)
            self.assertEqual(execute.call_count, 1)
        self.assertEqual(c.dispatch_action(other_tab)[0], 409)
        changed = {**payload, "action": "sensors"}
        self.assertEqual(c.dispatch_action(changed)[0], 409)
        self.assertEqual(self.request("sensors", boot_id="oldboot")[0], 409)

    def test_confirm_required_and_configuration_injection_rejected(self):
        self.prepare(until="stage8")
        self.assertEqual(self.request("hang_stand", confirmed=False)[0], 400)
        self.assertEqual(self.request("configure", mode="air", checkpoint=c.CKPT_DEFAULT)[0], 409)

    def test_selected_checkpoint_cannot_be_overridden_by_stand_request(self):
        self.prepare(mode="air", until="bridge")
        with patch.object(self.demo, "act", wraps=self.demo.act) as execute:
            self.step("stand", checkpoint="stand_v7", hang_arm=False)
            execute.assert_called_once_with("stand", {"checkpoint": c.CKPT_DEFAULT, "hang_arm": True})

    def test_command_acceptance_does_not_complete_stage(self):
        self.step("can_up")
        with patch.object(self.demo, "act", return_value=(True, "started")):
            self.step("sensors")
        self.assertNotIn("sensors", self.trial.completed)
        self.assertEqual(self.trial.pending["action"], "sensors")
        self.assertEqual(self.request("boot_pose")[0], 409)
        self.demo.act("sensors", {})
        c.console_snapshot()
        self.assertIn("sensors", self.trial.completed)

    def test_policy_load_without_debug_allows_stand(self):
        self.prepare(until="bridge")
        view = c.console_snapshot()
        self.assertFalse(view["fresh"]["bridge"])
        self.assertTrue(view["trial"]["allowed"]["stand"]["enabled"])

    def test_wrong_policy_load_never_completes_bridge(self):
        self.prepare(until="arm")
        original = self.demo.act
        def wrong(action, payload):
            return original(action, {**payload, "checkpoint": "stand_v7"})
        with patch.object(self.demo, "act", side_effect=wrong):
            self.step("bridge")
        self.assertNotIn("bridge", self.trial.completed)
        self.assertFalse(c.console_snapshot()["trial"]["allowed"]["stand"]["enabled"])

    def test_timer_5_30_and_retension_interrupt(self):
        self.prepare()
        clock = [10.0]
        self.trial.clock = lambda: clock[0]
        self.step("op_slack_done")
        clock[0] += 5.1
        view = c.console_snapshot()["trial"]
        self.assertTrue(view["measurement"]["observed_5s"])
        self.assertNotIn("success", view["measurement"])
        clock[0] += 25
        self.assertTrue(c.console_snapshot()["trial"]["measurement"]["observed_30s"])
        self.step("op_retension")
        elapsed = self.trial.measurement["elapsed"]
        clock[0] += 100
        c.console_snapshot()
        self.assertEqual(self.trial.measurement["state"], "interrupted")
        self.assertEqual(self.trial.measurement["elapsed"], elapsed)
        self.assertEqual(self.request("op_slack_done")[0], 409)

    def test_assist_stale_contact_and_guard_invalidate(self):
        for kind in ("assist", "stale", "contact", "guard", "record"):
            with self.subTest(kind=kind):
                self.trial.measurement = {"state": "waiting", "elapsed": 0, "reason": "test"}
                self.trial.slack_at = None
                self.trial.fault = None
                self.trial.completed = set(self.trial.flow())
                self.session.ensure("test", manual=True)
                view = Demo().snapshot()
                view.update(session={"active": True}, procs={"stage8": {"alive": True}, "bridge": {"alive": True}})
                view["fresh"] = dict.fromkeys(("status", "bridge", "imu", "foot"), True)
                view["stage8"].update(armed=True, running=True, feedback_ok=True)
                view["bridge"].update(armed=True, checkpoint=c.CKPT_DEFAULT)
                view["foot"]["contact"] = [True, True]
                self.trial.update(view)
                self.trial.operator_event("op_slack_done", {})
                if kind == "assist": self.trial.operator_event("op_assist", {})
                if kind == "stale": view["fresh"]["imu"] = False
                if kind == "contact": view["foot"]["contact"] = [False, True]
                if kind == "guard": view["bridge"]["fault"] = "osc_guard"
                if kind == "record": view["session"]["active"] = False
                self.trial.update(view)
                self.assertEqual(self.trial.measurement["state"], "interrupted")

    def test_stop_bypasses_every_gate_and_actual_ack_is_separate(self):
        self.prepare(until="stage8")
        self.trial.begin("hang_stand", c.now())
        c.BUSY = True
        self.assertIsNone(self.trial.reason("estop"))
        with patch.object(self.demo, "act", return_value=(False, "ROS disconnected")):
            code, result = c.dispatch_action({"action": "estop"})
        self.assertEqual(code, 200)
        self.assertFalse(result["ok"])
        self.assertFalse(c.console_snapshot()["trial"]["stop"]["confirmed"])
        self.assertIsNone(self.trial.pending)
        c.BUSY = False
        self.demo.act("estop", {})
        self.assertTrue(c.console_snapshot()["trial"]["stop"]["confirmed"])
        self.assertEqual(self.request("hang_stand")[0], 409)

    def test_old_stop_status_sequence_is_not_new_ack(self):
        self.prepare(until="stage8")
        self.demo.act("estop", {})
        c.console_snapshot()
        with patch.object(self.demo, "act", return_value=(True, "published only")):
            c.dispatch_action({"action": "estop"})
        self.assertFalse(c.console_snapshot()["trial"]["stop"]["confirmed"])

    def test_partial_estop_publish_still_attempts_both_paths(self):
        with patch.object(c.TEL, "publish", side_effect=[RuntimeError("broken"), True]) as publish:
            ok, message = c.act_estop()
        self.assertFalse(ok)
        self.assertEqual(publish.call_count, 2)
        self.assertIn("STOP_ALL 발행", message)

    def test_support_event_allowed_while_pending(self):
        self.step("can_up")
        with patch.object(self.demo, "act", return_value=(True, "waiting")):
            self.step("sensors")
        self.step("support_confirm")
        self.step("op_assist")
        self.assertTrue(self.trial.support)

    def test_recovery_order_and_new_trial(self):
        self.prepare()
        self.assertEqual(self.request("release")[0], 409)
        self.assertEqual(self.request("stack_down")[0], 409)
        self.assertEqual(self.request("session_end")[0], 409)
        self.step("support_confirm")
        self.step("stack_down")
        self.step("release")
        self.step("session_end")
        self.assertFalse(self.session.status()["active"])
        self.step("new_trial")
        self.assertEqual(c.TRIAL.number, 2)
        self.assertEqual(c.TRIAL.completed, set())
        self.assertIsNone(c.TRIAL.fault)

    def test_external_and_stale_process_check_block_release(self):
        self.trial.support = True
        self.trial.completed.add("stack_down")
        view = self.demo.snapshot()
        view["session"] = {"active": True}
        view["external_controls"] = [123]
        self.trial.update(view)
        self.assertIsNotNone(self.trial.reason("release"))
        view["external_controls"] = []
        view["process_check_ok"] = False
        self.trial.update(view)
        self.assertIsNotNone(self.trial.reason("release"))

    def test_recorder_contains_events_and_compact_trial_samples(self):
        self.prepare()
        self.step("op_slack_done")
        c.SESSION.sample(c.console_snapshot())
        self.step("op_assist")
        rows = [json.loads(line) for line in (Path(self.session.dir)/"timeline.jsonl").read_text().splitlines()]
        self.assertTrue(any(r.get("event") == "measurement_interrupted" for r in rows))
        samples = [r for r in rows if r.get("kind") == "s"]
        self.assertEqual(samples[-1]["sensor_contact"], [True, True])
        self.assertNotIn("allowed", samples[-1]["trial"])
        self.assertTrue(samples[-1]["unverified_force_fields"])

    def test_json_nonfinite_sanitizer(self):
        json.dumps(c.json_safe({"a": float("nan"), "b": [float("inf")]}), allow_nan=False)

    def test_recorder_failure_prevents_action_but_never_stop(self):
        with patch.object(self.session, "ensure", side_effect=OSError("disk full")):
            self.assertEqual(self.request("can_up")[0], 503)
            self.assertEqual(c.dispatch_action({"action": "estop"})[0], 200)

    def test_stop_during_blocking_request_does_not_resurrect_pending(self):
        started, finish = threading.Event(), threading.Event()
        def slow(action, payload):
            if action == "can_up":
                started.set()
                finish.wait(2)
            return True, "demo"
        response = []
        with patch.object(self.demo, "act", side_effect=slow):
            p = self.payload("can_up")
            thread = threading.Thread(target=lambda: response.append(c.dispatch_action(p)))
            thread.start()
            self.assertTrue(started.wait(1))
            c.dispatch_action({"action": "estop"})
            finish.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(response[0][1]["ok"])
        self.assertIsNone(self.trial.pending)
        self.assertNotIn("can_up", self.trial.completed)

    def test_analyze_does_not_fallback_to_latest_file(self):
        with patch.object(c.RUN, "snapshot", return_value={}):
            with self.assertRaises(ValueError):
                c.act_analyze()

    def test_output_contracts_for_hold_policy_and_release(self):
        def parse(key, lines):
            proc = c.Proc.__new__(c.Proc)
            proc.key, proc.argv, proc.started = key, [], c.now()
            proc.lines, proc.hold = collections.deque(maxlen=300), {}
            proc.loaded_checkpoint = proc.fault = proc.blackbox_path = proc.release_error = None
            proc.all_pass = proc.release_ok = False
            proc.logf = None
            proc.p = Mock(stdout=io.StringIO(lines), pid=42)
            proc.p.wait.return_value = 0
            proc.p.poll.return_value = 0
            proc._pump()
            return proc.snapshot()
        hang = parse("hang_stand", "[운용대기  29.0s] kp150.0 트림 a0.00/h0.00/k0.00°\n")
        self.assertEqual(hang["hold"]["phase"], "운용대기")
        self.assertEqual(hang["hold"]["kp"], 150)
        bridge = parse("bridge", "RL 브리지 기동 — LIVE, checkpoint=stand_hang_v5\n블랙박스 기록: /tmp/test.jsonl\nSTOP_ALL (sensor_stale)\nlater normal line\n")
        self.assertEqual(bridge["loaded_checkpoint"], "stand_hang_v5")
        self.assertEqual(bridge["blackbox_path"], "/tmp/test.jsonl")
        self.assertIn("sensor_stale", bridge["fault"])
        release = parse("release", "[release_all] 완료 — 송신실패 12건 (버스 상태 확인!)\n")
        self.assertFalse(release["release_ok"])
        self.assertIsNotNone(release["release_error"])
        good = parse("release", "[release_all] 12모터 소프트 릴리즈+MIT 해제 완료 — 무토크\n")
        self.assertTrue(good["release_ok"])

    def test_release_rc_zero_with_transmit_errors_is_failure(self):
        self.step("support_confirm")
        self.step("stack_down")
        original = self.demo.act
        def fail(action, payload):
            result = original(action, payload)
            self.demo.procs["release"]["release_error"] = "송신실패 12건"
            return result
        with patch.object(self.demo, "act", side_effect=fail):
            self.step("release")
        self.assertNotIn("release", self.trial.completed)
        self.assertIsNotNone(self.trial.fault)
        self.assertEqual(self.request("new_trial")[0], 409)

    def test_recovery_not_blocked_by_recorder_failure(self):
        self.prepare()
        with patch.object(self.session, "ensure", side_effect=OSError("disk full")):
            self.step("support_confirm")
            self.step("stack_down")
            self.step("release")
        self.assertIn("release", self.trial.completed)
        self.assertIsNotNone(self.trial.fault)

    def test_pending_timeout_is_not_success(self):
        self.step("can_up")
        clock = [0]
        self.trial.clock = lambda: clock[0]
        with patch.object(self.demo, "act", return_value=(True, "request accepted")):
            self.step("sensors")
        clock[0] = 31
        c.console_snapshot()
        self.assertIsNone(self.trial.pending)
        self.assertFalse(self.trial.last_result["ok"])
        self.assertNotIn("sensors", self.trial.completed)

    def test_baseline_requires_new_matching_command_sequence(self):
        self.prepare(until="handoff")
        self.demo.baseline = 12
        self.demo.command = "SET_BASELINE_FROM_CURRENT_ALL"
        c.console_snapshot()
        with patch.object(self.demo, "act", return_value=(True, "published")):
            self.step("baseline")
        self.demo.procs["stage8_baseline"] = {"rc": 0, "alive": False}
        c.console_snapshot()
        self.assertNotIn("baseline", self.trial.completed)
        self.demo.seq += 1
        c.console_snapshot()
        self.assertIn("baseline", self.trial.completed)

    def test_sender_death_and_can_busoff_latch(self):
        self.prepare(until="hang_stand")
        self.demo.procs["hang_stand"].update(alive=False, rc=0)
        c.console_snapshot()
        self.assertIsNotNone(self.trial.fault)
        self.assertEqual(self.request("gain_ramp")[0], 409)
        self.trial.fault = None
        view = self.demo.snapshot()
        view.update(session={"active": True}, can={"state": "BUS-OFF"})
        self.trial.update(view)
        self.assertIn("BUS-OFF", self.trial.fault)

    def test_process_check_fails_closed_and_does_not_kill(self):
        with patch.object(c.subprocess, "run", return_value=Mock(returncode=0, stdout="4321\n")):
            c.poll_processes()
            self.assertEqual(c.PROCESS_HEALTH["external"], [4321])
        with patch.object(c.subprocess, "run", side_effect=OSError("pgrep unavailable")):
            c.poll_processes()
            self.assertIsNotNone(c.PROCESS_HEALTH["error"])


if __name__ == "__main__":
    unittest.main()
