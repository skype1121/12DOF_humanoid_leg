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

    #: 1008: 콘솔 메뉴에 남은 정책(합성 키)은 공중 정책 시작(hang_arm)을 받지 않는다. 공중 방식의 콘솔 경로(코드는 그대로 남아 있다)는
    #: 메뉴를 시험용으로 바꿔 정리된 서기 키로 계속 고정한다.
    #: 1008 오후: 남긴 정책이 공중 정책 시작을 받는 합성 키로 바뀌어, 기본 키로도 공중 방식을 탈 수 있다(아래 *_kept_policy 시험들).
    #: 걷기 키가 아닌 서기 키의 공중 절차(종전 hang_arm 계약)는 이 시험용 메뉴로 계속 고정한다.
    AIR_KEY = "stand_crouch_g12"
    #: 공중 정책 시작을 받지 않는 걷기 키(지면 시작 학습) — 시험용 메뉴에 올려 '공중 방식 거부'를 계속 고정한다
    GROUND_ONLY_WALK_KEYS = ("stand_w3s3d_15800", "stand_kew3_g21a_g24ra_w3s3d15800", "stand_kew3q_g21a_g24ra_w3s3d15800")

    def use_air_key(self):
        p = patch.object(c, "CKPT_MENU", [self.AIR_KEY]); p.start(); self.addCleanup(p.stop)
        return self.AIR_KEY

    def prepare(self, mode="ground", until="stand", checkpoint=None):
        self.step("configure", mode=mode, checkpoint=checkpoint or c.CKPT_DEFAULT)
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

    def test_kept_policy_is_the_air_start_key_and_accepts_both_modes(self):
        """1008 오후: 남긴 정책 = 공중 정책 시작을 받는 합성 키(종전 합성 키 + 공중·착지 구간 정책). 시험 방식의 기본값은 '공중'이고, 공중·접지 설정을 둘 다 받는다."""
        self.assertEqual(c.CKPT_DEFAULT, "stand_kew3qa_g21a_g24ra_w3s3d15800")
        self.assertEqual((c.MODE_DEFAULT, self.trial.mode, c.console_snapshot()["trial"]["mode"]), ("air", "air", "air"))
        self.assertEqual(TrialState("stand_crouch_g12").mode, "ground")                      # 공중 정책 시작 키가 아니면 종전 기본값
        self.assertEqual(TrialState("stand_kew3q_g21a_g24ra_w3s3d15800").mode, "ground")
        self.assertEqual(c.console_snapshot()["trial"]["allowed"]["policy_on"]["label"], "정책 ON(공중에서 바로 구동)")
        self.step("configure", mode="ground", checkpoint=c.CKPT_DEFAULT)
        self.assertEqual(self.trial.mode, "ground")
        self.assertEqual(c.console_snapshot()["trial"]["allowed"]["policy_on"]["label"], "정책 ON(착지 자동 인수)")
        self.step("configure", mode="air", checkpoint=c.CKPT_DEFAULT)
        self.assertEqual(self.trial.mode, "air")
        self.assertEqual(self.request("configure", mode="hang", checkpoint=c.CKPT_DEFAULT)[0], 400)
        self.assertEqual(self.trial.mode, "air")

    def test_new_trial_keeps_the_mode_of_the_previous_trial(self):
        """'새 시험 준비'는 앞 시험의 방식·정책을 그대로 넘긴다(기본값으로 되돌리지 않는다) — 접지 방식으로 바꿔 둔 운영자의 선택이 다음 시험에서 조용히 공중으로 바뀌지 않게."""
        for mode in ("ground", "air"):
            self.step("configure", mode=mode, checkpoint=c.CKPT_DEFAULT)
            c.TRIAL.completed.add("release")                                                 # (회수·토크 해제까지 끝난 상태)
            number = c.TRIAL.number
            self.step("new_trial")
            self.assertEqual((c.TRIAL.mode, c.TRIAL.checkpoint, c.TRIAL.number), (mode, c.CKPT_DEFAULT, number + 1))
            self.assertEqual(c.console_snapshot()["trial"]["mode"], mode)

    def test_ground_only_walk_keys_still_refuse_air_policy_start(self):
        """공중 정책 시작을 받지 않는 걷기 키(지면 시작으로만 학습·평가 — AIR1 시뮬: 착지 낙상 다수)는 메뉴에 올라와 있어도 공중 방식 설정을 서버가 사유를 대고 거부한다."""
        for key in self.GROUND_ONLY_WALK_KEYS:
            with patch.object(c, "CKPT_MENU", [key]):
                before = (self.trial.mode, self.trial.checkpoint)
                code, res = self.request("configure", mode="air", checkpoint=key)
                self.assertEqual(code, 400, (key, res))
                self.assertIn("공중 정책 시작(hang_arm)을 받지 않습니다", res["message"])
                self.assertIn("접지 후 정책 인계", res["message"])
                self.assertEqual((self.trial.mode, self.trial.checkpoint), before)           # 설정 불변
                self.step("configure", mode="ground", checkpoint=key)                        # 접지 방식은 받는다
                self.assertEqual((self.trial.mode, self.trial.checkpoint), ("ground", key))
                self.assertEqual(TrialState(key).mode, "ground")
            self.trial.mode, self.trial.checkpoint = before

    def test_cleaned_up_policies_are_refused_by_the_server(self):
        for key in ("stand_crouch_g12", "stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "stand_w3s3d_15800", "stand_kew3_g21a_g24ra_w3s3d15800",
                    "stand_kew3q_g21a_g24ra_w3s3d15800"):
            code, res = self.request("configure", mode="ground", checkpoint=key)
            self.assertEqual(code, 400, (key, res))
            self.assertIn("정리된 정책", res["message"])
            self.assertEqual(self.trial.checkpoint, c.CKPT_DEFAULT)

    def test_air_flow_no_contact_until_touchdown(self):
        self.prepare("air", checkpoint=self.use_air_key())
        self.assertEqual(self.demo.contact, [False, False])
        self.assertEqual(self.request("op_slack_done")[0], 409)
        self.step("op_touchdown")
        self.step("op_slack_done")
        self.assertEqual(self.trial.measurement["state"], "measuring")

    def test_air_flow_with_the_kept_policy(self):
        """1008 오후: 남긴 정책(공중 정책 시작 키)으로 단계별 공중 절차 — 접촉 없이 stand 까지 가고, stand 는 hang_arm 으로 나간다(요청이 끼워 넣은 값은 무시)."""
        self.prepare("air", until="bridge")
        self.assertEqual(self.demo.contact, [False, False])
        with patch.object(self.demo, "act", wraps=self.demo.act) as execute:
            self.step("stand", checkpoint="stand_v7", hang_arm=False, touchdown_start=True)
            execute.assert_called_once_with("stand", {"checkpoint": c.CKPT_DEFAULT, "hang_arm": True})
        self.assertIn("stand", self.trial.completed)
        self.assertFalse(self.trial.fault)
        self.assertFalse(self.trial.policy_on)
        self.assertEqual(self.request("op_slack_done")[0], 409)                              # 매달린 채(무접촉)로는 무보조 확인 불가
        self.step("op_touchdown")
        self.step("op_slack_done")
        self.assertEqual(self.trial.measurement["commander"], "policy")

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

    def test_pad_channel_keys_flow_from_adapter_to_console(self):
        """0913: 어댑터 채널별 출력이 콘솔 foot dict 로 그대로 흐르는가 (현장 확인 1순위).

        0912 실물 상황 재현 — 왼[2] 고착 패드는 bias 5000 으로 판정 제외, 살아있는
        왼[0]/왼[3]만 눌림. 접촉 계약(contact/pads_on)은 이 추가로 바뀌지 않는다.
        """
        from jetson.scripts import foot_force_adapter_node as ffa
        bias = [0.0, 0.0, 5000.0, 0.0]
        raw = {"timestamp": c.now(), "left": [420, 150, 587, 900], "right": [0, 0, 250, 30]}
        out = ffa.pressure_to_foot_force(raw, ffa.NEWTON_PER_COUNT, 0.0, 0.0,
                                         ffa.PAD_CONTACT_COUNTS, bias, None)
        _state, _warn, err = ffa.attach_dead_diag(out, ffa.new_dead_state(), raw,
                                                  ffa.PAD_CONTACT_COUNTS, bias, None)
        self.assertIsNone(err)
        telemetry = c.Telemetry()
        telemetry.receive("foot", out)
        foot = telemetry.view()["foot"]
        self.assertEqual(foot["pads_counts"], [[420, 150, 0, 900], [0, 0, 250, 30]])
        self.assertEqual(foot["pads_on_mask"], [[True, False, False, True],
                                                [False, False, True, False]])
        self.assertEqual(foot["dead_mask"], [[False] * 4, [False] * 4])
        self.assertIn("판정제외(bias) L2", foot["dead_note"])
        self.assertEqual(foot["contact"], [True, True])
        self.assertEqual(foot["pads_on"], [2, 1])
        json.dumps(c.json_safe(foot), allow_nan=False)

    def test_missing_or_malformed_pad_channels_are_none_not_zero(self):
        telemetry = c.Telemetry()
        base = {"timestamp": c.now(), "left_contact": True, "right_contact": False}
        telemetry.receive("foot", base)                      # 구버전 어댑터: 새 키 없음
        foot = telemetry.view()["foot"]
        self.assertEqual(foot["contact"], [True, False])     # 접촉 계약은 그대로
        for key in ("pads_counts", "pads_on_mask", "dead_mask", "dead_note"):
            self.assertIsNone(foot[key], key)
        telemetry.receive("foot", dict(
            base, timestamp=c.now(), left_pads_counts=[1, 2, 3],
            right_pads_counts=[0, 0, 0, float("nan")], left_pads_on_mask=[1, 0, 0, 0],
            right_pads_on_mask=[True] * 4, dead_mask=[[False] * 4], dead_note="   "))
        foot = telemetry.view()["foot"]
        self.assertIsNone(foot["pads_counts"])               # 길이 3·NaN 은 0 으로 위장하지 않는다
        self.assertEqual(foot["pads_on_mask"], [None, [True] * 4])   # 1/0 은 bool 계약 아님
        self.assertIsNone(foot["dead_mask"])                 # [[4]] 한 발만 = 형식 오류
        self.assertIsNone(foot["dead_note"])
        self.assertEqual(foot["contact"], [True, False])
        self.demo.act("sensors", {})                         # --demo 화면도 새 키를 낸다
        self.demo.act("op_touchdown", {})
        demo_foot = c.console_snapshot()["foot"]
        self.assertEqual([sum(m) for m in demo_foot["pads_on_mask"]], demo_foot["pads_on"])
        self.assertEqual([len(row) for row in demo_foot["pads_counts"]], [4, 4])

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
        for name in ("march", "trim", "disarm", "posture", "rm"):   # zero_recapture 는 0914부터 정식 영점 도구
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
        key = self.use_air_key()
        self.prepare(mode="air", until="bridge", checkpoint=key)
        with patch.object(self.demo, "act", wraps=self.demo.act) as execute:
            self.step("stand", checkpoint="stand_v7", hang_arm=False)
            execute.assert_called_once_with("stand", {"checkpoint": key, "hang_arm": True})

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

    # ---- 1005 정책 ON(단순화 기동) ------------------------------------------------------
    def wait_policy_on(self):
        worker = c.POLICY_ON_WORKER
        self.assertIsNotNone(worker)
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive(), "정책 ON 순서기가 끝나지 않음")
        c.console_snapshot()

    def test_policy_on_runs_existing_steps_and_waits_for_touchdown(self):
        self.prepare(until="gain_ramp")
        self.assertEqual(self.demo.contact, [False, False])        # 매달린 상태
        calls = []
        original = self.demo.act
        def spy(action, payload):
            calls.append((action, dict(payload)))
            return original(action, payload)
        with patch.object(self.demo, "act", side_effect=spy):
            self.step("policy_on")
            self.wait_policy_on()
        self.assertFalse(self.trial.fault, self.trial.fault)
        self.assertEqual([a for a, _ in calls], list(c.POLICY_ON_STEPS))   # 기존 단계를 순서대로, 한 번씩
        stand = dict(calls)["stand"]
        self.assertEqual(stand, {"checkpoint": c.CKPT_DEFAULT, "hang_arm": False, "touchdown_start": True})
        self.assertTrue(set(c.POLICY_ON_STEPS) | {"policy_on"} <= self.trial.completed)
        snap = c.console_snapshot()["trial"]
        self.assertIn("착지 대기", snap["phase"])
        self.assertEqual(self.trial.commander(), "hold")            # 착지 전 = 홀드(정책 아님)
        self.assertEqual(self.request("policy_on")[0], 409)          # 중복 실행 거부
        self.assertEqual(self.request("stand")[0], 409)              # 단계 재실행 거부
        self.step("op_touchdown")                                   # 착지 → 브리지 게이트 POLICY
        self.assertNotIn("착지 대기", c.console_snapshot()["trial"]["phase"])
        self.assertEqual(self.trial.commander(), "policy")
        self.step("op_slack_done")
        self.assertEqual(self.trial.measurement["commander"], "policy")

    def _policy_on_air(self, checkpoint=None):
        self.prepare("air", until="gain_ramp", checkpoint=checkpoint)
        self.assertEqual(self.demo.contact, [False, False])        # 매달린 상태
        calls = []
        original = self.demo.act
        def spy(action, payload):
            calls.append((action, dict(payload)))
            return original(action, payload)
        with patch.object(self.demo, "act", side_effect=spy):
            code, res = self.request("policy_on")
            self.assertEqual(code, 200, res)
            self.assertIn("공중에서 바로 구동", res["message"])
            self.assertNotIn("착지 확인 뒤에만", res["message"])
            self.wait_policy_on()
        return calls

    def test_policy_on_air_mode_starts_policy_in_the_air(self):
        """1008 오후 공중 방식의 정책 ON: 같은 단계를 같은 순서로 한 번씩 — 마지막 stand 만 hang_arm 으로 나간다(착지 자동 인수 touchdown_start 아님).
        착지 대기가 없다: 정책이 공중에서 구동 중(명령 주체 = 정책), 화면 이름·기록 문구도 그렇게 말한다."""
        self.assertEqual(c.console_snapshot()["trial"]["allowed"]["policy_on"]["label"], "정책 ON(공중에서 바로 구동)")
        calls = self._policy_on_air()
        self.assertFalse(self.trial.fault, self.trial.fault)
        self.assertEqual([a for a, _ in calls], list(c.POLICY_ON_STEPS))   # 기존 단계를 순서대로, 한 번씩
        stand = dict(calls)["stand"]
        self.assertEqual(stand, {"checkpoint": c.CKPT_DEFAULT, "hang_arm": True, "touchdown_start": False})
        self.assertEqual(c.bridge_cmd_payload("stand", {k: stand[k] for k in ("hang_arm", "touchdown_start")}), {"cmd": "stand", "hang_arm": True})
        self.assertTrue(set(c.POLICY_ON_STEPS) | {"policy_on"} <= self.trial.completed)
        self.assertTrue(self.trial.policy_on)
        snap = c.console_snapshot()["trial"]
        self.assertNotIn("착지 대기", snap["phase"])
        self.assertIn("정책 제어", snap["phase"])
        self.assertIsNone(self.demo.start_gate)                      # 착지 자동 인수 게이트를 만들지 않았다
        self.assertEqual(self.trial.commander(), "policy")           # 공중에서 이미 정책이 명령 중
        armed = [e for e in self.trial.history if e.get("event") == "policy_on_armed"]
        self.assertEqual(len(armed), 1)
        self.assertIn("공중 정책 구동", armed[0]["message"])
        self.assertNotIn("착지 대기", armed[0]["message"])
        self.assertEqual(self.request("policy_on")[0], 409)          # 중복 실행 거부
        self.assertEqual(self.request("stand")[0], 409)              # 단계 재실행 거부
        self.assertEqual(self.request("op_slack_done")[0], 409)      # 매달린 채로는 무보조 확인 불가(양발 접촉 필요)
        self.step("op_touchdown")
        self.step("op_slack_done")
        self.assertEqual(self.trial.measurement["commander"], "policy")

    def test_policy_on_follows_the_mode_for_a_non_walk_key_too(self):
        """정책 ON 의 마지막 단계는 시험 방식만 본다: 걷기 키가 아닌 서기 키(시험용 메뉴)의 공중 방식도 hang_arm 으로 나간다(종전 단계별 공중 절차와 같은 인자)."""
        key = self.use_air_key()
        calls = self._policy_on_air(checkpoint=key)
        self.assertFalse(self.trial.fault, self.trial.fault)
        self.assertEqual(dict(calls)["stand"], {"checkpoint": key, "hang_arm": True, "touchdown_start": False})
        self.assertEqual(self.trial.commander(), "policy")

    def test_policy_on_blocks_other_actions_while_running_and_estop_aborts(self):
        self.prepare(until="gain_ramp")
        gate = threading.Event()
        original = self.demo.act
        def slow(action, payload):
            if action == "baseline":
                gate.wait(5)
            return original(action, payload)
        with patch.object(self.demo, "act", side_effect=slow):
            self.step("policy_on")
            for _ in range(200):
                if (self.trial.macro or {}).get("step") == "baseline":
                    break
                time.sleep(0.01)
            self.assertEqual((self.trial.macro or {}).get("step"), "baseline")
            self.assertEqual(self.request("arm")[0], 409)            # 순서 건너뛰기 거부
            self.assertEqual(self.request("policy_on")[0], 409)      # 중복 클릭 거부
            self.assertIn("정책 ON", c.console_snapshot()["trial"]["allowed"]["bridge"]["reason"])
            c.dispatch_action({"action": "estop"})                   # 정지는 언제나
            gate.set()
            self.wait_policy_on()
        self.assertIsNone(self.trial.macro)
        self.assertIsNotNone(self.trial.fault)
        self.assertNotIn("arm", self.trial.completed)
        self.assertNotIn("stand", self.trial.completed)

    def test_policy_on_step_failure_stops_and_latches(self):
        self.prepare(until="gain_ramp")
        original = self.demo.act
        def fail_arm(action, payload):
            if action == "arm":
                return False, "ARM 송신 실패(모의)"
            return original(action, payload)
        with patch.object(self.demo, "act", side_effect=fail_arm):
            self.step("policy_on")
            self.wait_policy_on()
        self.assertIsNone(self.trial.macro)
        self.assertIn("ARM", self.trial.fault)
        self.assertNotIn("bridge", self.trial.completed)
        self.assertNotIn("policy_on", self.trial.completed)
        self.assertNotIn("bridge", self.demo.procs)                  # 정책 프로세스는 띄우지 않았다

    def test_policy_on_preconditions(self):
        self.step("configure", mode="ground", checkpoint=c.CKPT_DEFAULT)
        self.assertEqual(self.request("policy_on")[0], 409)          # 준비 전
        self.prepare(until="gain_ramp")
        self.demo.act("op_touchdown", {})                            # 양발 접촉 상태 → 거부(매달린 상태에서만)
        self.assertIn("양발 접촉", c.console_snapshot()["trial"]["allowed"]["policy_on"]["reason"])
        self.demo.contact = [False, False]
        self.assertTrue(c.console_snapshot()["trial"]["allowed"]["policy_on"]["enabled"])
        self.demo.contact = [True, True]
        self.step("ground")                                          # 단계별 절차를 시작하면 정책 ON 불가(섞지 않음)
        self.assertIn("단계별", c.console_snapshot()["trial"]["allowed"]["policy_on"]["reason"])

    def test_old_stepwise_stand_payload_unchanged(self):
        self.assertEqual(c.bridge_cmd_payload("stand", {"hang_arm": False, "touchdown_start": False}), {"cmd": "stand"})
        self.prepare(until="bridge")
        with patch.object(self.demo, "act", wraps=self.demo.act) as execute:
            self.step("stand")
            execute.assert_called_once_with("stand", {"checkpoint": c.CKPT_DEFAULT, "hang_arm": False})
        self.assertFalse(self.trial.policy_on)


if __name__ == "__main__":
    unittest.main()


def test_poll_processes_excludes_readonly_preview_only():
    """0921 사고: 정지한 robotctl motors 의 자식 zero_anchor --preview(읽기 전용)가 '외부 제어' 로 잡혀
    강제 초기화가 영구히 막혔다. 읽기 전용 미리보기만 제외하고, 정지(T) 제어 프로세스와 확인 불가 PID 는 남긴다."""
    import io
    from unittest.mock import Mock, patch

    fake = {111: "/usr/bin/python3 /x/zero_anchor.py --preview --json",   # 읽기 전용 → 제외
            222: "/usr/bin/python3 /x/stage8_12axis_mit_control_node.py",  # 정지여도 남김(토크 물고 있을 수 있음)
            333: "/usr/bin/python3 /x/zero_anchor.py --write"}             # 쓰기 → 남김
    real_open = open

    def fake_open(path, mode="rb", *a, **kw):
        if isinstance(path, str) and path.startswith("/proc/"):
            pid = int(path.split("/")[2])
            if pid not in fake:
                raise OSError("no such pid")
            return io.BytesIO(fake[pid].replace(" ", "\0").encode())
        return real_open(path, mode, *a, **kw)

    with patch.object(c.subprocess, "run", return_value=Mock(returncode=0, stdout="111 222 333 999\n")), \
         patch("builtins.open", fake_open):
        c.poll_processes()
    assert sorted(c.PROCESS_HEALTH["external"]) == [222, 333, 999], c.PROCESS_HEALTH["external"]


def test_release_no_bus_completes_step_with_reason():
    """0922: 모터 전원이 꺼지면 ACK 가 없어 release_all 송신이 전부 실패한다.
    종전엔 이를 '실패' 로 처리해 토크 해제 단계가 영구히 막혔다(0918·0921). 이제 NO_BUS 는 사유를 남기고 단계를 넘긴다.
    반대로 일부만 응답하는 PARTIAL 은 그대로 실패여야 한다(허브 반접촉 등 진짜 이상)."""
    from jetson.scripts import console_state as cs

    def run(verdict, rc):
        t = cs.TrialState("stand_crouch_v5")
        t.begin("release", 100.0)
        t.pending["launched"] = True
        view = {"procs": {"release": {"alive": False, "rc": rc, "release_verdict": verdict,
                                      "release_ok": verdict == "OK", "release_error": None}},
                "stage8": {}, "bridge": {}, "fresh": {}, "received": {}, "server_time": 101.0}
        t.update(view)
        return t

    t = run("NO_BUS", 4)
    assert "release" in t.completed, "NO_BUS 는 단계를 넘겨야 한다(막다른 길 금지)"
    assert t.last_result["ok"] is True and "버스 무응답" in t.last_result["message"]
    assert t.release_no_bus is True
    assert "전원 차단" in t.snapshot()["owner"], t.snapshot()["owner"]

    t = run("PARTIAL", 5)
    assert "release" not in t.completed, "일부 응답은 실패로 남아야 한다"
    assert t.last_result["ok"] is False

    t = run("OK", 0)
    assert "release" in t.completed and t.last_result["ok"] is True
    assert not getattr(t, "release_no_bus", False)


def test_slack_confirm_allows_hold_only_and_records_commander():
    """0928: 무보조 확인이 정책 armed 만 허용해 홀드 자립을 기록할 수 없었다(70세션 전수 0건).
    이제 홀드(stage8 ARM·스트리밍, 정책 미명령)도 허용하되 명령 주체를 기록해 절대 섞지 않는다."""
    from jetson.scripts import console_state as cs

    def mk(completed, s8, br, fresh, contact=(True, True), t=100.0):
        st = cs.TrialState("stand_crouch_v6")
        st.completed = set(completed)
        st.view = {"stage8": s8, "bridge": br, "fresh": fresh,
                   "foot": {"contact": list(contact)}, "procs": {},
                   "session": {"active": True}, "received": {}, "server_time": t}
        return st

    S8_ON = {"armed": True, "running": True, "holding": True, "baseline": 12, "feedback_ok": True}   # 실제 콘솔 뷰에는 항상 있음(operator_console.py feedback_ok)
    FRESH_ALL = {"status": True, "imu": True, "foot": True, "bridge": True, "joints": True}

    # 1) 정책이 명령 중 → "policy" (종전 동작 유지)
    t = mk(("arm", "bridge", "stand"), S8_ON, {"armed": True}, FRESH_ALL)
    assert t.commander() == "policy", t.commander()

    # 2) 홀드만 (정책 미명령) → "hold" — 종전엔 None 이라 버튼이 막혔다
    t = mk(("arm",), S8_ON, {"armed": False}, dict(FRESH_ALL, bridge=False))
    assert t.commander() == "hold", t.commander()

    # 3) stage8 이 ARM 아니면 None
    t = mk(("arm",), {"armed": False, "running": True}, {"armed": False}, FRESH_ALL)
    assert t.commander() is None
    # 4) arm 단계 미완료면 None (홀드가 명령 중이라 볼 수 없다)
    t = mk((), S8_ON, {"armed": False}, FRESH_ALL)
    assert t.commander() is None
    # 5) status 피드백이 낡으면 None
    t = mk(("arm",), S8_ON, {"armed": False}, dict(FRESH_ALL, status=False))
    assert t.commander() is None


def test_slack_confirm_hold_measurement_invalidated_if_policy_takes_over():
    """홀드 자립 측정 중 정책이 명령을 잡으면 그 시험은 무효 — 두 자립을 섞지 않는다."""
    from jetson.scripts import console_state as cs
    S8_ON = {"armed": True, "running": True, "holding": True, "baseline": 12, "feedback_ok": True}   # 실제 콘솔 뷰에는 항상 있음(operator_console.py feedback_ok)
    FRESH = {"status": True, "imu": True, "foot": True, "bridge": False, "joints": True}

    now = [100.0]                                   # 측정 경과는 주입 시계(TrialState.clock)로 잰다 — server_time 과 같이 움직인다
    t = cs.TrialState("stand_crouch_v6", clock=lambda: now[0])
    t.completed = {"arm"}
    base = {"stage8": S8_ON, "bridge": {"armed": False}, "fresh": FRESH,
            "foot": {"contact": [True, True]}, "procs": {},
            "session": {"active": True}, "received": {}, "server_time": 100.0}
    t.update(dict(base))
    t.operator_event("op_slack_done", {})
    assert t.measurement["state"] == "measuring"
    assert t.measurement["commander"] == "hold", t.measurement
    assert "홀드" in t.measurement["reason"], t.measurement["reason"]

    # 홀드 유지 → 계속 측정, 5초 관측 표시
    now[0] = 106.0
    t.update(dict(base, server_time=106.0))
    assert t.measurement["state"] == "measuring", t.measurement
    assert t.measurement.get("observed_5s") is True, t.measurement

    # 정책이 명령을 잡음 → 무효
    now[0] = 107.0
    t.update(dict(base, server_time=107.0, bridge={"armed": True},
                  fresh=dict(FRESH, bridge=True)))
    assert t.measurement["state"] == "interrupted", t.measurement
    assert "정책이 명령을 잡음" in t.measurement["reason"], t.measurement["reason"]


def test_slack_confirm_still_needs_contact_and_fresh_sensors():
    """완화는 제어기 조건에만 적용된다 — 접촉·센서 신선도 요구는 그대로여야 한다."""
    from jetson.scripts import console_state as cs
    S8_ON = {"armed": True, "running": True, "holding": True, "baseline": 12, "feedback_ok": True}   # 실제 콘솔 뷰에는 항상 있음(operator_console.py feedback_ok)
    FRESH = {"status": True, "imu": True, "foot": True, "bridge": False, "joints": True}

    def reason(contact=(True, True), fresh=None):
        t = cs.TrialState("stand_crouch_v6")
        t.completed = {"arm"}
        t.update({"stage8": S8_ON, "bridge": {"armed": False}, "fresh": fresh or FRESH,
                  "foot": {"contact": list(contact)}, "procs": {},
                  "session": {"active": True}, "received": {}, "server_time": 100.0})
        return t.reason("op_slack_done")

    assert reason() is None, reason()                       # 홀드·양발·신선 → 허용
    assert reason(contact=(False, True)) is not None        # 한 발만 → 거부
    assert reason(fresh=dict(FRESH, imu=False)) is not None  # IMU 낡음 → 거부


def test_page_policy_on_wording_follows_the_mode():
    """1008 오후: 정책 ON 의 화면 문구는 시험 방식을 따른다 — 접지 방식의 버튼·설명·확인 문구는 종전 글자 그대로이고, 공중 방식일 때만 '공중에서 바로 구동' 문구로 바뀐다
    (마크업·스크립트 조각 확인. 스크립트를 실제로 돌려 본 것은 아니다)."""
    page = c.render_page()
    ground_note = "공중 수평 자세(kp150)에서 한 번 — 인계·기준점·ARM·정책 로드를 같은 게이트로 진행, 정책은 양발 착지 확인 뒤에만 시작"
    assert '<button data-action="policy_on" id="policyOnBtn">정책 ON (착지 자동 인수)</button><small id="policyOnNote">' + ground_note + "</small>" in page
    assert "const POLICY_ON_GROUND={label:'정책 ON (착지 자동 인수)',note:'" + ground_note + "'};" in page
    assert "function confirmText(action){return action==='policy_on'&&state?.trial.mode==='air'?POLICY_ON_AIR.confirm:confirmations[action]}" in page
    assert "text('confirmText',custom?.text||confirmText(action))" in page and "text('confirmText',custom?.text||confirmations[action])" not in page
    air = page.split("const POLICY_ON_AIR={", 1)[1].split("};", 1)[0]
    for part in ("정책이 공중에서 바로 구동", "다리가 움직일 수 있", "줄이 로봇을 지지", "발이 바닥에서 떨어져", "다리 주변에 사람이 없는지", "지그를 천천히 내리세요",
                 "서기 정책(KE)으로 자동으로 넘깁니다", "걷기 버튼이 잠겨"):
        assert part in air, part
    assert "착지(0.3 s)·정착을 브리지가 확인한 뒤에만 시작" not in air           # 접지 방식의 문장이 공중 문구에 섞이지 않는다
    ground = page.split("const confirmations={policy_on:'", 1)[1].split("',air_gait_on:", 1)[0]
    assert "정책은 양발 착지(0.3 s)·정착을 브리지가 확인한 뒤에만 시작" in ground and "공중에서 바로" not in ground
    assert "text('policyOnBtn',t.mode==='air'?(t.allowed.policy_on?.label||'정책 ON(공중에서 바로 구동)'):POLICY_ON_GROUND.label)" in page
    assert "text('policyOnNote',t.mode==='air'?POLICY_ON_AIR.note:POLICY_ON_GROUND.note)" in page
    assert "w.mode==='air'?'공중·착지 정책(착지 정착 뒤 KE 인계)'" in page       # 걷기 패널의 모드 표시


def test_policy_on_level_pose_and_ramp_preconditions():
    """1005 정책 ON(실물 경로): 수평 기준 자세 적용·램프 완료·에어 보행 정지가 선행조건 — 누락이면 인수 순간 큰 자세 이동이 된다(0918 낙상 기전)."""
    from jetson.scripts import console_state as cs
    ref = {j: 0.0 for j in c.JOINT_NAMES}
    ref.update(left_knee_joint=-38.5, right_knee_joint=38.5)

    def reason(jt_cur, jt_req=None, gait=False, ref_override="unset"):
        t = cs.TrialState("stand_crouch_g21a")
        t.completed = set(cs.PREP)
        jt = {j: {"req": (jt_req or jt_cur)[j], "cur": jt_cur[j]} for j in c.JOINT_NAMES}
        t.view = {"server_time": 1000.0, "fresh": {"status": True, "imu": True, "foot": True, "bridge": False},
                  "stage8": {"waiting": True}, "bridge": {}, "foot": {"contact": [False, False]},
                  "procs": {"hang_stand": {"alive": True, "hold": {"kp": 150.0, "ts": 999.5}}, "stage8": {"alive": True}},
                  "process_check_ok": True, "session": {"active": True}, "external_controls": [],
                  "hang_hold": {"kp": 150.0, "joint_trim": jt, "gait": {"active": gait}},
                  "level_pose_ref": ref if ref_override == "unset" else ref_override}
        return t.reason("policy_on")

    assert reason(dict(ref)) is None, reason(dict(ref))
    zero = {j: 0.0 for j in c.JOINT_NAMES}
    assert "수평 기준 자세 적용" in reason(zero)                       # 수평 자세 누락(무릎 0°)
    assert "램프" in reason(dict(ref, left_knee_joint=-30.0), dict(ref))  # 램프 진행 중
    assert "에어 보행" in reason(dict(ref), gait=True)
    assert reason(dict(ref, left_knee_joint=-35.0)) is None             # 미세 조정(±5° 이내)은 허용
    assert "level_pose" in reason(dict(ref), ref_override=None)          # level_pose.json 없음 → 거부(0914 상수 폴백을 기준으로 쓰지 않음)


# 1006: stage8 지연 단축 옵션 전달 — 기본(빈 값)은 종전 기동 인자와 같아야 한다
def test_stage8_latency_args_default_empty(monkeypatch):
    monkeypatch.delenv("STAGE8_LATENCY_OPTS", raising=False)
    assert c.stage8_latency_args() == []
    assert c.stage8_latency_args("") == []


def test_stage8_latency_args_allowlist():
    import pytest
    assert c.stage8_latency_args("fas") == ["--feedback-after-send"]
    assert c.stage8_latency_args("fas,soc") == ["--feedback-after-send", "--send-on-command"]
    assert c.stage8_latency_args("fas, w4") == ["--feedback-after-send", "--feedback-wait-ms", "4"]
    for bad in ("hz100", "--control-hz 100", "w20", "x"):
        with pytest.raises(ValueError):
            c.stage8_latency_args(bad)


def test_stage8_latency_file(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGE8_LATENCY_OPTS", raising=False)
    monkeypatch.setattr(c, "STAGE8_LATENCY_FILE", str(tmp_path / "none"))
    assert c.stage8_latency_args() == []                      # 파일 없음 = 종전
    f = tmp_path / "opts"; f.write_text("fas   # 1단계\n")
    monkeypatch.setattr(c, "STAGE8_LATENCY_FILE", str(f))
    assert c.stage8_latency_args() == ["--feedback-after-send"]
    f.write_text("\n")
    assert c.stage8_latency_args() == []                      # 빈 파일 = 종전
