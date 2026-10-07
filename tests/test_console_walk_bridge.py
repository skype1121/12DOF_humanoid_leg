"""콘솔 걷기 패널 ↔ **실제 브리지 노드 코드** 계약 시험(1007). ROS·로봇·젯슨·정책 파일 불필요.

  python3 -m pytest -q tests/test_console_walk_bridge.py

jetson/scripts/selftest_w3_bridge_node.py 의 시험대(가짜 rclpy·가짜 시계·스텁 정책)를 빌려 rl_bridge_node.main() 을 그대로 띄우고,
콘솔(operator_console)의 **실제 경로**를 그 노드에 맞물린다(DEMO 아님):
    콘솔 dispatch_action → Telemetry.publish(JSON) → 노드 _on_cmd
    노드가 응답 토픽·디버그 토픽에 발행한 JSON → 콘솔 Telemetry.receive → console_snapshot → TrialState.walk
브리지 기동 인자는 콘솔이 만드는 것(bridge_live_argv) + 시험 전용 게이트 우회(--force-frame --force-gains)다.

확인: 콘솔이 만든 인자로 브리지가 걷기 세션으로 뜬다(두 걷기 키) · 수락(기한이 이 요청으로 걸림) → 같은 req_id 의 burst_end → 서기 복귀
(합성 키는 KE 인계까지) · '정착 대기'가 풀린 직후의 켜기를 실제 감독기가 수락 · 정지 래치/재무장 · 콘솔이 볼 수 없는 이유의 거부와 사유 표시 ·
걷기 끄기 · estop · guard_mode 표시의 근거(응답·디버그) · 클릭당 발행 1 회.
**계약·순서 검증이다 — 정책은 스텁이고 접촉은 클록 위상으로 만든 가정 모형이라 폐루프 성능·실물 안전을 증명하지 않는다.**
"""
import collections
import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

from jetson.scripts import operator_console as c
from jetson.scripts.console_state import TrialState, WALK_KEYS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "jetson", "scripts")
ROS_MODULES = ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg")

pytestmark = pytest.mark.skipif(c.WALK_IMPORT_ERROR is not None, reason=f"걷기 모듈 없음: {c.WALK_IMPORT_ERROR}")


@pytest.fixture(scope="module")
def bench():
    """브리지 노드 시험대와 브리지 모듈(시험대가 쓰는 최상위 이름 그대로). 가짜 rclpy 는 시험이 끝나면 sys.modules 에서 되돌린다."""
    saved = {k: sys.modules.get(k) for k in ROS_MODULES}
    added = SCRIPTS not in sys.path
    if added:
        sys.path.insert(0, SCRIPTS)
    import selftest_w3_bridge_node as st
    import rl_bridge_node as b
    yield st, b
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    if added and SCRIPTS in sys.path:
        sys.path.remove(SCRIPTS)


def fake_proc(key, loaded=None):
    """콘솔이 띄워 추적 중인(살아 있는) 프로세스의 대역 — 실제 프로세스는 만들지 않는다."""
    proc = c.Proc.__new__(c.Proc)
    proc.key, proc.argv, proc.started, proc.rc = key, [key], c.now(), None
    proc.lines, proc.hold = collections.deque(maxlen=300), {}
    proc.loaded_checkpoint, proc.warmup, proc.all_pass, proc.fault = loaded, None, False, None
    proc.blackbox_path, proc.release_ok, proc.release_error, proc.release_verdict, proc.logf = None, False, None, None, None
    proc.p = Mock(pid=4242)
    proc.p.poll.return_value = None
    return proc


class Link:
    """콘솔 ↔ 브리지 노드 배선 + 시간 진행. 콘솔은 DEMO 가 아니라 실제 Telemetry 경로를 쓴다."""

    def __init__(self, node, st, b, clock, key):
        self.node, self.st, self.b, self.clock, self.key = node, st, b, clock, key
        self.H = st.Harness(node, b, clock)
        self.sent = []                       # 콘솔이 브리지 명령 토픽에 발행한 것
        self.stage8 = []                     # 콘솔이 stage8 명령 토픽에 발행한 것(estop 의 STOP_ALL)
        self.console_gyro = None             # 콘솔이 보는 IMU 각속도(None = 브리지와 같은 값)
        self._cursor = {b.BRIDGE_ACK_TOPIC: 0, b.BRIDGE_DEBUG_TOPIC: 0}
        self.stages = []                     # 걷기 버튼 단계의 변화(순서)

    def publish(self, which, obj):
        if which == "stage8":
            self.stage8.append(dict(obj))
            return True
        self.sent.append(json.loads(json.dumps(obj)))
        self.node._on_cmd(self.st._Msg(json.dumps(obj)))          # 콘솔 발행 → (직렬화) → 브리지 구독 콜백
        return True

    def pump(self):
        for topic, name in ((self.b.BRIDGE_ACK_TOPIC, "ack"), (self.b.BRIDGE_DEBUG_TOPIC, "bridge")):
            msgs = self.node._pubs[topic].msgs
            for raw in msgs[self._cursor[topic]:]:
                c.TEL.receive(name, json.loads(raw))
            self._cursor[topic] = len(msgs)

    def feed_console(self):
        """콘솔이 직접 구독하는 stage8 상태·IMU·발 접촉 — 브리지에 넣는 것과 같은 합성 입력."""
        now = self.clock.t
        joints = {j: {"baseline_deg": 0.0, "latest_actual_deg": 0.0, "feedback_valid": True, "feedback_stale": False,
                      "feedback_age_sec": 0.01, "commanded_relative_deg": 0.0} for j in self.b.JOINT_ORDER}
        c.TEL.receive("status", {"timestamp": now, "armed": True, "running": True, "accepted": True, "joints": joints,
                                 "last_command": "SET_ALL", "command_seq": 5})
        c.TEL.receive("imu", {"timestamp": now, "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "gyro_rad_s": list(self.console_gyro or self.H.gyro)})
        cl, cr = self.H.contact()
        c.TEL.receive("foot", {"timestamp": now, "left_contact": bool(cl), "right_contact": bool(cr)})

    def tick(self, seconds=None):
        n = 1 if seconds is None else max(1, int(round(seconds / self.b.TICK_SEC)))
        for _ in range(n):
            self.H.tick(1)                   # 브리지 한 틱: 입력 주입 → _tick → 시계 + 0.02
            self.pump()
            self.feed_console()
            c.PROCESS_HEALTH["ts"] = self.clock.t
            c.console_snapshot()
            cur = c.TRIAL.walk.cur
            s = (cur["req_id"], cur["stage"]) if cur else None
            if s is not None and (not self.stages or self.stages[-1] != s):
                self.stages.append(s)

    def walk(self):
        return c.console_snapshot()["walk"]

    def payload(self, action, **extra):
        state = c.console_snapshot()
        return {"action": action, "boot_id": state["trial"]["boot_id"], "revision": state["trial"]["revision"],
                "request_id": c.uuid.uuid4().hex, "confirmed": True, "seen": state["server_time"], **extra}

    def req(self, action, **extra):
        return c.dispatch_action(self.payload(action, **extra))

    def until(self, pred, limit_s, what):
        t0 = self.clock.t
        while not pred(self.walk()):
            assert self.clock.t - t0 <= limit_s, f"{what}: {limit_s:g} s 안에 일어나지 않음 — {self.walk()['current']} / {self.walk()['reason']}"
            self.tick()
        return round(self.clock.t - t0, 2)


def run_session(bench, tmp_path, monkeypatch, key, guard, body):
    """콘솔이 만든 기동 인자로 실제 브리지 main() 을 띄우고(가짜 ROS), spin 자리에서 body(link) 를 돌린다."""
    st, b = bench
    clock = st.FakeClock()
    monkeypatch.setenv("HOME", str(tmp_path))                     # 블랙박스 = tmp/logs (실제 ~/logs 를 건드리지 않는다)
    st._patch(b, clock)                                           # 가짜 시계·스텁 정책·guard_mode safe
    if guard == "data":                                           # 실제 하드웨어맵의 값(data)을 그대로 쓰는 변형 — 경고 표시의 근거 확인용
        real_load = b._st_real_load

        def load_data():
            frames = real_load()
            frames.guard_mode = "data"
            return frames
        b.load_frame_map = load_data
    argv = c.bridge_live_argv(key)
    assert argv[:2] == [sys.executable, os.path.join(c.SCRIPTS, "rl_bridge_node.py")] and argv[2:5] == ["--live", "--checkpoint", key]
    node_argv = argv[2:] + ["--force-frame", "--force-gains"]      # 시험 전용: 프레임·게인 게이트 우회(실물 기동에는 붙지 않는다)
    done = {}

    def driver(node):
        assert node.w3sup is not None, "콘솔 인자로 띄운 브리지가 걷기 세션(감독기)이 아님"
        assert node.core.kew3 is WALK_KEYS[key]["composite"] and node.core.w3cmd and node.core.w3hl is None, "합성 여부·헤딩 고리 off 가 인자와 다름"
        link = Link(node, st, b, clock, key)
        trial = TrialState(key, clock=lambda: clock.t)
        trial.completed = set(trial.flow()) - {"stand"}            # 기존 단계별 절차로 정책 로드까지 끝난 상태에서 출발
        run = c.Runner()
        run.procs = {"bridge": fake_proc("bridge", loaded=key), "stage8": fake_proc("stage8")}
        tel = c.Telemetry()
        tel.publish = link.publish
        session = c.SessionLog()
        with patch.dict(c.__dict__, DEMO=False, TRIAL=trial, TEL=tel, RUN=run, SESSION=session, SESSION_DIR=str(tmp_path / "sessions"),
                        BUSY=False, REQUESTS=collections.OrderedDict(), PROCESS_HEALTH={"ts": clock.t, "external": [], "error": None},
                        now=lambda: clock.t, W3_SEQ_FILE=str(tmp_path / "w3_seq"),
                        JOINT_TRIM_FILE=str(tmp_path / "jt.json"), HANG_STATUS_FILE=str(tmp_path / "hs.json"),
                        AIR_GAIT_FILE=str(tmp_path / "ag.json"), LEVEL_POSE_FILE=str(tmp_path / "lp.json")), \
                patch.object(c.subprocess, "run", side_effect=AssertionError("real process forbidden")), \
                patch.object(c.subprocess, "Popen", side_effect=AssertionError("real process forbidden")):
            session.ensure("walk bridge contract", manual=True)
            link.H.tick(5)                                         # 센서 흐름(정책은 아직 비활성 — 디버그 토픽 없음)
            link.feed_console()
            ok, _msg = c.act_bridge_cmd("stand", {"hang_arm": False, "touchdown_start": False})      # 콘솔의 기존 stand 경로
            assert ok and link.sent == [{"cmd": "stand"}]
            link.tick(2.6)                                         # 인계 워밍업 + 온셋 블렌딩 1.5 s
            assert not trial.fault, trial.fault
            trial.completed.add("stand")                           # (콘솔의 stand 응답 확인에 해당 — 아래부터 걷기 패널)
            link.tick(0.1)
            assert not trial.fault, trial.fault
            try:
                body(link)
                done["ok"] = True
            finally:
                session.close("walk bridge contract")
    rc = st._run_main(b, node_argv, driver)
    assert rc == 0 and done.get("ok"), f"브리지 main rc {rc}"


CASES = [(key, guard) for key in WALK_KEYS for guard in ("safe", "data")]


@pytest.mark.parametrize("key,guard", CASES)
def test_walk_button_lifecycle_against_real_bridge_node(bench, tmp_path, monkeypatch, key, guard):
    composite = WALK_KEYS[key]["composite"]
    moves = [m for m in c.walk_moves() if m["enabled"]]
    short = min(moves, key=lambda m: m["dur_s"])
    long = max(moves, key=lambda m: m["dur_s"])

    def body(link):
        node, b = link.node, link.b
        # --- 세션 표시: 걷기 키·guard_mode(stand 응답과 디버그 프레임에서)·stand 응답은 걷기 버튼 기록과 무관 ---
        w = link.walk()
        assert w["session"] and w["profile"]["checkpoint"] == key and w["composite"] is composite, w
        assert w["guard_mode"] == guard and w["bridge"]["fresh"] and w["bridge"]["standing"] and w["current"] is None, w
        assert (w["mode"] == "ke") is composite and w["bridge"]["heading_loop"] is False
        assert w["recent_acks"][-1]["cmd"] == "stand" and w["recent_acks"][-1]["accepted"] and w["recent_acks"][-1]["match"] is None
        # --- ① 수락 → burst_end(같은 req_id) → 서기 복귀 ---
        link.until(lambda w: w["available"], 3.0, "정책 시작 뒤 걷기 버튼 열림")
        n_sent = len(link.sent)
        code, res = link.req("w3_move", move=short["name"])
        assert code == 200 and res["ok"] and len(link.sent) == n_sent + 1, (code, res)
        cmd = link.sent[-1]
        assert cmd == {"cmd": "step", "enable": True, "req_id": cmd["req_id"], "seq": cmd["seq"], "heading_loop": False,
                       **{k: short[k] for k in ("vx", "vy", "wz", "dur_s")}}, cmd
        link.tick()
        cur = link.walk()["current"]
        assert cur["stage"] == "walking" and cur["deadline"] is True and cur["ack_action"] == "set_want" and not cur["late_ack"], cur
        assert tuple(node.core.cmd) == (short["vx"], short["vy"], short["wz"]) and node.w3sup.burst_req == cmd["req_id"]
        walked = link.until(lambda w: w["current"]["stage"] != "walking", short["dur_s"] + 1.0, "기한 종료")
        cur = link.walk()["current"]
        assert cur["stage"] == "ending" and cur["end_by"] == "burst" and not cur["note"], cur       # 디버그 추정이 아니라 burst_end 응답으로
        assert short["dur_s"] - 0.06 <= walked <= short["dur_s"] + 0.1, walked      # 끄기는 브리지의 기한에(콘솔 타이머 없음)
        assert tuple(node.core.cmd) == (0.0, 0.0, 0.0) and not node.core.runner._w3_clock.want
        if composite:                                              # 래치가 꺼진 뒤에도 KE 로 넘기기 전에는 '서기 복귀'가 아니다
            link.until(lambda w: w["bridge"]["ge"] is False, 1.0, "반주기 래치 해제")
            w = link.walk()
            assert w["mode"] == "w3" and w["current"]["stage"] == "ending" and not w["bridge"]["standing"], w
        settled = link.until(lambda w: w["current"]["stage"] == "done", 5.0, "서기 복귀")
        w = link.walk()
        assert w["bridge"]["standing"] and (w["mode"] == "ke") is composite and w["current"]["stand_seen"], w
        if composite:                                              # 래치가 꺼진 뒤 조용한 양발 서기 0.5 s 를 기다린 다음에야 KE 로 넘긴다
            assert settled >= 0.5 - 0.02, settled
        assert node.core.active and not link.H.stop_alls, "걷기 버튼 한 번이 STOP_ALL 을 냈다"
        assert [s for _rid, s in link.stages if _rid == cmd["req_id"]] == ["walking", "ending", "done"], link.stages
        acks = [json.loads(m) for m in node._pubs[b.BRIDGE_ACK_TOPIC].msgs if json.loads(m).get("req_id") == cmd["req_id"]]
        assert [a.get("event") for a in acks] == [None, "burst_end"] and all(a["accepted"] for a in acks), acks
        # --- ② 서기 복귀 직후에는 '정착 대기'로 잠겨 있다가, 풀린 직후의 켜기를 실제 감독기가 수락한다(연타가 브리지 거부로 끝나지 않는다) ---
        w = link.walk()
        assert not w["available"] and ("정착 대기" in w["reason"] or "양발 접촉" in w["reason"]), w["reason"]
        reopen = link.until(lambda w: w["available"], 3.0, "정착 뒤 버튼 다시 열림")
        code, res = link.req("w3_move", move=short["name"])
        link.tick()
        cur = link.walk()["current"]
        assert code == 200 and cur["stage"] == "walking", (reopen, cur)
        link.until(lambda w: w["current"]["stage"] == "done", 6.0, "둘째 걷기의 서기 복귀")
        # --- ③ 정지 래치: 콘솔이 걸러 발행하지 않는다 → 재무장 → 다시 가능 ---
        link.until(lambda w: w["controls"]["w3_stop"]["enabled"], 1.0, "정지 래치 버튼")
        code, res = link.req("w3_stop")
        link.tick()
        w = link.walk()
        assert code == 200 and w["command"]["stage"] == "accepted" and w["bridge"]["stop_latched"] is True and node.w3sup.stop_latched, w["command"]
        n_sent = len(link.sent)
        code, res = link.req("w3_move", move=short["name"])
        assert code == 409 and "정지 래치" in res["message"] and len(link.sent) == n_sent, (code, res)
        code, res = link.req("w3_rearm")
        link.tick()
        w = link.walk()
        assert code == 200 and w["command"]["kind"] == "rearm" and w["command"]["stage"] == "accepted", w["command"]
        assert w["bridge"]["stop_latched"] is False and not node.w3sup.stop_latched
        # --- ④ 콘솔이 볼 수 없는 이유의 브리지 거부: 사유가 그대로 표시되고, 다음 클릭은 새 seq 로 수락 ---
        link.console_gyro = list(link.H.gyro)                     # 콘솔 IMU 는 조용, 브리지 입력만 흔들림
        link.H.gyro = (0.3, 0.0, 0.0)                             # 감독기 정착 문턱(0.25 rad/s)은 넘고 KE 회복 감지 문턱(0.5)은 안 넘는 값
        link.tick(0.3)
        link.until(lambda w: w["available"], 3.0, "콘솔 선행조건 통과")
        code, res = link.req("w3_move", move=short["name"])
        link.tick()
        cur = link.walk()["current"]
        assert code == 200 and cur["stage"] == "rejected" and cur["by"] == "bridge" and "정착 미달" in cur["why"], cur
        assert not node.core.runner._w3_clock.want, "거부된 켜기가 걷기를 켰다"
        rejected_seq = link.sent[-1]["seq"]
        link.H.gyro = tuple(link.console_gyro)
        link.console_gyro = None
        link.tick(1.3)
        # --- ⑤ 긴 버튼을 걷는 중에 걷기 끄기 → 걷기 끝 요청됨(운영자 끄기) → 서기 복귀, 뒤늦은 burst_end 없음 ---
        link.until(lambda w: w["available"], 3.0, "버튼 열림")
        code, res = link.req("w3_move", move=long["name"])
        link.tick(0.5)
        cur = link.walk()["current"]
        assert code == 200 and cur["stage"] == "walking" and link.sent[-1]["seq"] > rejected_seq, cur
        long_rid = link.sent[-1]["req_id"]
        code, res = link.req("w3_off")
        link.tick()
        w = link.walk()
        assert code == 200 and w["command"]["kind"] == "off" and w["command"]["stage"] == "accepted", w["command"]
        assert w["current"]["stage"] == "ending" and w["current"]["end_by"] == "off", w["current"]
        link.until(lambda w: w["current"]["stage"] == "done", 5.0, "끄기 뒤 서기 복귀")
        link.tick(long["dur_s"] + 0.5)
        assert not any(json.loads(m).get("event") == "burst_end" and json.loads(m).get("req_id") == long_rid
                       for m in node._pubs[b.BRIDGE_ACK_TOPIC].msgs), "운영자가 끈 걷기에 뒤늦은 burst_end"
        # --- ⑥ 걷는 중 정지 요청(estop): 브리지 estop + stage8 STOP_ALL 동시 → 걷기 버튼 '중단', 이후 걷기·재무장 불가, 끄기는 가능 ---
        link.until(lambda w: w["available"], 3.0, "버튼 열림")
        link.req("w3_move", move=long["name"])
        link.tick(0.4)
        assert link.walk()["current"]["stage"] == "walking"
        n_stop = len(link.H.stop_alls)
        code, res = c.dispatch_action({"action": "estop"})
        assert code == 200 and res["ok"] and link.sent[-1] == {"cmd": "estop"} and link.stage8[-1] == {"command": "STOP_ALL"}, (code, res)
        assert len(link.H.stop_alls) == n_stop + 1 and not node.core.active
        link.pump()
        c.console_snapshot()
        cur = link.walk()["current"]
        assert cur["stage"] == "interrupted" and "정지" in cur["why"], cur
        n_sent = len(link.sent)
        for action, extra in (("w3_move", {"move": short["name"]}), ("w3_rearm", {})):
            code, res = link.req(action, **extra)
            assert code == 409 and len(link.sent) == n_sent, (action, code, res)
        code, res = link.req("w3_off")
        assert code == 200 and res["ok"] and link.sent[-1]["cmd"] == "step" and link.sent[-1]["enable"] is False, (code, res)
        link.pump()
        c.console_snapshot()
        assert link.walk()["command"]["kind"] == "off" and link.walk()["command"]["stage"] == "accepted"
        # --- 타임라인: 요청·응답·기한 종료·서기 복귀 확인이 사건으로 남았다 ---
        with open(os.path.join(c.SESSION.dir, "timeline.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        events = [r["event"] for r in rows if r.get("kind") == "trial_event" and str(r.get("event", "")).startswith("walk")]
        assert events[:4] == ["walk_request", "walk_ack", "walk_burst_end", "walk_done"], events[:8]
        for name in ("walk_cmd_ack", "walk_end", "walk_interrupted"):
            assert name in events, (name, events)
        assert sum(1 for r in rows if r.get("kind") == "bridge_ack") == len(node._pubs[b.BRIDGE_ACK_TOPIC].msgs)

    run_session(bench, tmp_path, monkeypatch, key, guard, body)


def test_non_walk_session_has_no_ack_topic_and_console_refuses(bench, tmp_path, monkeypatch):
    """걷기 키가 아닌 종전 세션: 브리지는 응답 토픽을 열지 않고(감독기 없음), 콘솔은 걷기 명령을 아예 발행하지 않는다."""
    st, b = bench
    clock = st.FakeClock()
    monkeypatch.setenv("HOME", str(tmp_path))
    st._patch(b, clock)
    key = "stand_crouch_g21a"
    argv = c.bridge_live_argv(key)
    assert argv[2:] == ["--live", "--checkpoint", key]             # 기존 키의 인자는 종전 그대로
    seen = {}

    def driver(node):
        assert node.w3sup is None and b.BRIDGE_ACK_TOPIC not in node._pubs
        trial = TrialState(key, clock=lambda: clock.t)
        trial.completed = set(trial.flow())
        with patch.dict(c.__dict__, DEMO=False, TRIAL=trial, TEL=c.Telemetry(), RUN=c.Runner(), SESSION=c.SessionLog(),
                        SESSION_DIR=str(tmp_path / "sessions"), BUSY=False, REQUESTS=collections.OrderedDict(),
                        PROCESS_HEALTH={"ts": clock.t, "external": [], "error": None}, now=lambda: clock.t,
                        W3_SEQ_FILE=str(tmp_path / "w3_seq")), \
                patch.object(c.TEL, "publish", side_effect=AssertionError("걷기 세션이 아닌데 발행")):
            assert trial.reason("w3_move") is not None and trial.reason("w3_off") is not None and trial.reason("w3_stop") is not None
            seen["ok"] = True
    rc = st._run_main(b, argv[2:] + ["--force-frame", "--force-gains"], driver)
    assert rc == 0 and seen.get("ok")
