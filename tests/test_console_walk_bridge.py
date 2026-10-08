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
import math
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


def run_session(bench, tmp_path, monkeypatch, key, guard, body, start=None):
    """콘솔이 만든 기동 인자로 실제 브리지 main() 을 띄우고(가짜 ROS), spin 자리에서 body(link) 를 돌린다.
    start(1008): 정책 시작을 시험이 직접 하는 경우의 함수 start(link, trial) — 주면 아래의 기본 시작(콘솔의 지면 stand 발행 → 2.6 s → 완료 처리)을 하지 않는다."""
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
        if node.core.kew3:                    # 1008: 콘솔 인자로 뜬 합성 세션의 KE 감지기 갇힘 해소 = 그 키의 값(첫 합성 키는 꺼짐 — 종전 그대로)
            det = node.core.runner.det
            assert (det.requiet, det.requiet_contact) == REQUIET[key], (key, det.requiet, det.requiet_contact)
            assert bool(node.core.runner.has_air) is (key == AIR_START_KEY), f"{key}: 공중·착지 러너 유무가 기대와 다름"       # 1008 오후: 공중 정책 시작 키에만
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
            if start is not None:                                  # 1008: 시험이 정책 시작을 직접 한다(공중 시작 경로)
                try:
                    start(link, trial)
                    body(link)
                    done["ok"] = True
                finally:
                    session.close("walk bridge contract")
                return
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
#: 1008: 합성 키별 KE 감지기 갇힘 해소 (requiet, 접촉 요구) — 여기 따로 적은 기대값(브리지 명세를 읽어 오지 않는다)
REQUIET = {"stand_kew3_g21a_g24ra_w3s3d15800": (None, "both"), "stand_kew3q_g21a_g24ra_w3s3d15800": ((2.0, 0.3), "both"),
           "stand_kew3qa_g21a_g24ra_w3s3d15800": ((2.0, 0.3), "both")}       # 1008 오후 셋째 합성 키(+ 공중·착지 정책): 서기 KE·감지기 설정은 둘째 합성 키와 같다
#: 1008 오후: 공중 정책 시작(stand + hang_arm)을 받는 걷기 키 — 여기 따로 적은 기대값(콘솔 WALK_KEYS·브리지 명세를 읽어 오지 않는다)
AIR_START_KEY = "stand_kew3qa_g21a_g24ra_w3s3d15800"


def test_walk_keys_and_console_bridge_argv_are_pinned():
    """1008: 이 계약 시험이 도는 걷기 키 = 1007 의 두 키 + 둘째 합성 키(갇힘 해소). 콘솔이 만드는 브리지 인자는 두 합성 키가 같다(키 이름만 다름).
    1008 오후: + 셋째 합성 키(둘째 합성 키 + 공중·착지 구간 서기 정책 · 공중 정책 시작). 인자는 다른 합성 키와 같다."""
    assert list(WALK_KEYS) == ["stand_w3s3d_15800", "stand_kew3_g21a_g24ra_w3s3d15800", "stand_kew3q_g21a_g24ra_w3s3d15800", AIR_START_KEY]
    assert [k for k in WALK_KEYS if WALK_KEYS[k].get("air_start")] == [AIR_START_KEY] and "공중 정책 시작 가능" in WALK_KEYS[AIR_START_KEY]["label"]
    assert [k for k, _g in CASES] == [k for k in WALK_KEYS for _ in range(2)] and set(REQUIET) == {k for k in WALK_KEYS if WALK_KEYS[k]["composite"]}
    assert c.bridge_live_argv("stand_w3s3d_15800")[2:] == ["--live", "--checkpoint", "stand_w3s3d_15800", "--experimental-w3", "--w3-heading-loop", "off"]
    for key in REQUIET:
        assert c.bridge_live_argv(key)[2:] == ["--live", "--checkpoint", key, "--experimental-w3", "--experimental-kew3", "--w3-heading-loop", "off"]
        assert c.bridge_selftest_argv(key)[2:] == ["--selftest", "--checkpoint", key, "--experimental-w3", "--experimental-kew3"]
    assert "갇힘 해소(시뮬 평가 구성)" in WALK_KEYS["stand_kew3q_g21a_g24ra_w3s3d15800"]["label"] and "실물 미검증" in WALK_KEYS["stand_kew3q_g21a_g24ra_w3s3d15800"]["label"]


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


@pytest.mark.parametrize("key", list(REQUIET))
def test_recovery_mode_lock_in_walk_panel_and_requiet_release(bench, tmp_path, monkeypatch, key):
    """1008 갇힘 해소 대조(콘솔 ↔ 실제 브리지 노드, 두 합성 키에 같은 입력): 브리지 IMU 가 기준과 4.5° 다른 자세로 조용히 서서 KE 회복 모드에 갇히면
    걷기 패널이 브리지 사유와 함께 잠긴다(콘솔은 발행하지 않는다). 첫 합성 키는 2.6 s 뒤에도 그대로이고 자세를 되돌려야 풀린다.
    둘째 합성 키(갇힘 해소 2.0 s)는 그 자세 그대로 약 2 s 뒤 패널이 다시 열리고 걷기 버튼이 수락돼 서기 복귀까지 간다.
    **계약·순서 검증이다 — 스텁 정책·가정 접촉 모형이라 실물에서 갇힘이 얼마나 생기는지·해소가 안전한지는 증명하지 않는다.**"""
    short = min((m for m in c.walk_moves() if m["enabled"]), key=lambda m: m["dur_s"])

    def body(link):
        node = link.node
        det = node.core.runner.det
        link.until(lambda w: w["available"], 3.0, "정책 시작 뒤 걷기 버튼 열림")
        link.tick(2.0)                                             # KE 감지기 기준 자세 고정(인계 뒤 1.5 s 추종 창)
        link.H.pitch = math.radians(4.5)                           # 브리지 IMU 만 기운다(콘솔이 보는 IMU 는 수평) — 켜기 3° 초과·조기 복귀 자격 5° 미만·끄기 1.5° 초과
        link.tick(0.3)
        w = link.walk()
        assert node.core.runner.last_mode == 1 and not w["available"] and "회복 모드" in w["reason"], w["reason"]
        n_sent = len(link.sent)
        code, res = link.req("w3_move", move=short["name"])
        assert code == 409 and "회복 모드" in res["message"] and len(link.sent) == n_sent, (code, res)
        link.tick(2.3)                                             # 기울인 뒤 2.6 s — 조용한 양발 서기
        w = link.walk()
        if REQUIET[key][0] is None:
            assert node.core.runner.last_mode == 1 and det.n_requiet == 0 and not w["available"] and "회복 모드" in w["reason"], w["reason"]
            code, res = link.req("w3_move", move=short["name"])
            assert code == 409 and "회복 모드" in res["message"] and len(link.sent) == n_sent, (code, res)
            link.H.pitch = 0.0                                     # 자세를 되돌려야만 풀린다
            link.until(lambda w: w["available"], 4.0, "자세 복귀 뒤 버튼 열림")
            assert det.n_requiet == 0
        else:
            assert node.core.runner.last_mode == 0 and det.n_requiet == 1 and det.n_early == 0, (node.core.runner.last_mode, det.n_requiet)
            link.until(lambda w: w["available"], 3.0, "갇힘 해소 뒤 버튼 다시 열림")
            assert w["bridge"]["fresh"] and link.walk()["mode"] == "ke"
        code, res = link.req("w3_move", move=short["name"])       # 풀린 뒤에는 두 키 모두 걷기 버튼이 수락돼 서기 복귀까지 간다
        link.tick()
        assert code == 200 and res["ok"] and link.walk()["current"]["stage"] == "walking", (code, res, link.walk()["current"])
        link.until(lambda w: w["current"]["stage"] == "done", 6.0, "서기 복귀")
        assert node.core.active and not link.H.stop_alls

    run_session(bench, tmp_path, monkeypatch, key, "safe", body)


def _console_stand(link, trial):
    """콘솔의 단계 'stand' 를 실제 경로로 낸다(dispatch_action → TrialState 게이트 → 발행 → 브리지 노드). 공중(무접촉)에서 시작한다.
    프로세스 목록 검사(pgrep)는 시험에서 돌리지 않는다 — 신선도만 맞춰 둔다."""
    link.H.force_contact = (False, False)
    link.H.tick(5)                                                 # 센서 흐름(정책은 아직 비활성 — 디버그 토픽 없음)
    link.feed_console()
    c.PROCESS_HEALTH["ts"] = link.clock.t
    with patch.object(c, "poll_processes", lambda: None):
        return link.req("stand")


def test_air_mode_stand_from_console_starts_policy_in_the_air_on_real_bridge_node(bench, tmp_path, monkeypatch):
    """1008 공중 방식(콘솔 ↔ 실제 브리지 노드): 공중 정책 시작 키에서 콘솔의 stand 는 hang_arm 으로 나가고 브리지가 받는다 —
    매달린 채(무접촉) 러너가 AIR 모드(공중·착지 정책)로 구동되고(HangGuard PRE_CONTACT), 콘솔의 stand 단계는 단계별 공중 절차와 같은 응답 확인으로 완료된다.
    AIR 동안 걷기 패널은 사유("공중·착지 정책 구간…")와 함께 잠겨 있고(콘솔이 발행하지 않는다), 직접 보낸 켜기도 브리지가 거부한다.
    착지(양발 접촉)·정착 뒤 러너가 서기 KE 로 넘기면 패널이 정착 대기를 거쳐 열리고, 걷기 버튼이 수락돼 서기 복귀까지 간다.
    **계약·순서 검증이다 — 스텁 정책·가정 접촉이라 실물 공중 거동·착지 성공을 증명하지 않는다.**"""
    key = AIR_START_KEY
    short = min((m for m in c.walk_moves() if m["enabled"]), key=lambda m: m["dur_s"])

    def start(link, trial):
        node, b = link.node, link.b
        run = node.core.runner
        assert trial.mode == "air" and trial.checkpoint == key, "공중 정책 시작 키의 TrialState 기본 방식이 air 가 아님"
        assert run.has_air and run.mode == "ke"
        code, res = _console_stand(link, trial)
        assert code == 200 and res["ok"] and link.sent == [{"cmd": "stand", "hang_arm": True}], (code, res, link.sent)
        assert node.core.active and node.hang_guard.hang_mode and node.hang_guard.state == b.HangGuard.PRE_CONTACT and run.mode == "air"
        link.tick(0.2)
        assert "stand" in trial.completed and not trial.fault and trial.pending is None, (trial.pending, trial.fault, trial.last_result)
        link.tick(2.6)                                             # 인계 워밍업 + 온셋 블렌딩 — 그동안도 매달린 채

    def body(link):
        node, b = link.node, link.b
        run = node.core.runner
        snap = c.console_snapshot()
        assert snap["bridge"]["hang_state"] == b.HangGuard.PRE_CONTACT and snap["bridge"]["armed"] and snap["foot"]["contact"] == [False, False]
        assert snap["trial"]["mode"] == "air" and "정책 제어" in snap["trial"]["phase"] and c.TRIAL.commander() == "policy"
        assert snap["trial"]["allowed"]["policy_on"]["label"] == "정책 ON(공중에서 바로 구동)"
        dbg = json.loads(node._pubs[b.BRIDGE_DEBUG_TOPIC].msgs[-1])
        assert dbg["kew3"]["mode"] == "air" and dbg["c5"]["mode"] == 0 and run.mode == "air" and run.last_step_kind == 3
        w = link.walk()
        assert w["session"] and w["mode"] == "air" and not w["available"] and not w["bridge"]["standing"] and not w["bridge"]["walking"], w
        assert "공중·착지 정책 구간(착지 정착 뒤 KE 인계 대기)" in w["reason"], w["reason"]
        n_sent = len(link.sent)
        code, res = link.req("w3_move", move=short["name"])
        assert code == 409 and "공중·착지 정책 구간" in res["message"] and len(link.sent) == n_sent, (code, res)     # 콘솔이 발행하지 않는다
        link.publish("bridge", {"cmd": "step", "enable": True, "req_id": "direct", "seq": 2 ** 40, "vx": short["vx"]})   # 콘솔을 우회한 켜기 → 브리지가 거부
        ack = json.loads(node._pubs[b.BRIDGE_ACK_TOPIC].msgs[-1])
        assert ack["req_id"] == "direct" and not ack["accepted"] and "양발 접촉" in ack["why"] and ack["kew3"]["mode"] == "air", ack
        assert not run._w3_clock.want
        link.tick(0.5)
        assert run.mode == "air" and run.n_to_w3 == 0 and node.core.active and not link.H.stop_alls
        # 착지(지그를 내림): 양발 접촉·조용함 → 러너가 KE 로 넘긴다 → 콘솔의 정착 대기 뒤 걷기 버튼이 열리고 수락된다
        link.H.force_contact = (True, True)
        link.tick(0.5)
        w = link.walk()
        assert run.mode == "air" and w["mode"] == "air" and not w["available"] and "공중·착지 정책 구간" in w["reason"], w["reason"]   # 닿았어도 인계 전에는 잠김
        t_ke = link.until(lambda w: run.mode == "ke", 1.0, "착지 정착 뒤 KE 인계")
        assert 0.4 <= t_ke <= 0.6 and run.n_air_to_ke == 1 and node.hang_guard.state == b.HangGuard.CONTACT_ACQUIRED, t_ke      # 양발 접촉 연속 1.0 s(앞의 0.5 s 포함)
        link.tick()
        w = link.walk()
        assert w["mode"] == "ke" and not w["available"] and "공중·착지 정책 구간" not in w["reason"], w["reason"]
        link.H.force_contact = None                               # 가정 접촉 모형(서기 = 양발, 걷는 중 = 위상에 따라 한발)
        link.until(lambda w: w["available"], 4.0, "KE 인계·정착 뒤 걷기 버튼 열림")
        code, res = link.req("w3_move", move=short["name"])
        link.tick()
        assert code == 200 and res["ok"] and link.walk()["current"]["stage"] == "walking", (code, res, link.walk()["current"])
        link.until(lambda w: w["current"]["stage"] == "done", 6.0, "서기 복귀")
        assert node.core.active and not link.H.stop_alls and not c.TRIAL.fault and link.walk()["mode"] == "ke" and run.n_air_to_ke == 1

    run_session(bench, tmp_path, monkeypatch, key, "safe", body, start=start)


@pytest.mark.parametrize("key", [k for k in WALK_KEYS if k != AIR_START_KEY])
def test_bridge_refuses_hang_arm_for_other_walk_keys_even_if_console_state_says_air(bench, tmp_path, monkeypatch, key):
    """공중 정책 시작을 받지 않는 걷기 키: 콘솔의 시험 설정은 공중 방식을 거부한다(다른 시험). 그 문을 우회해 콘솔 상태가 air 가 되더라도
    (시험에서 직접 바꿔 넣음) **브리지가 hang_arm 을 거부**한다 — 정책은 시작되지 않고, 콘솔의 stand 단계는 응답 확인 시간 초과로 실패·고장 래치된다."""
    def start(link, trial):
        node = link.node
        assert trial.mode == "ground"
        trial.mode = "air"                                         # (시험 전용: 설정 게이트를 우회한 상태)
        code, res = _console_stand(link, trial)
        assert code == 200 and link.sent == [{"cmd": "stand", "hang_arm": True}], (code, res, link.sent)
        assert not node.core.active and node._launches == 0 and getattr(node.core.runner, "mode", "ke") == "ke", "브리지가 hang_arm 을 받아 정책을 시작함"
        ack = json.loads(node._pubs[link.b.BRIDGE_ACK_TOPIC].msgs[-1])
        assert ack["cmd"] == "stand" and not ack["accepted"], ack
        why = next((m for _lv, m in node._log.lines if "hang_arm 불가" in m), None)
        # 거부 사유에 허용 키가 적힌다 — 합성 플래그로 뜬 세션(합성 키)에는 공중 정책 시작 키, W3 단독 세션(합성 플래그 없음)에는 '없음'
        assert why is not None and ((AIR_START_KEY in why) if WALK_KEYS[key]["composite"] else ("없음" in why and AIR_START_KEY not in why)), why
        link.tick(8.5)                                             # stand 응답 확인 상한 8 s
        assert "stand" not in trial.completed and trial.fault and "정책 제어 시작" in trial.fault, (trial.fault, trial.last_result)
        assert not node.core.active and link.b.BRIDGE_DEBUG_TOPIC in node._pubs and not node._pubs[link.b.BRIDGE_DEBUG_TOPIC].msgs

    run_session(bench, tmp_path, monkeypatch, key, "safe", lambda link: None, start=start)


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
