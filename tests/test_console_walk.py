"""콘솔 걷기 조작 패널(1007) 오프라인 시험. ROS·서브프로세스·실제 로그·실제 순번 파일을 쓰지 않는다.

  python3 -m pytest -q tests/test_console_walk.py

- 콘솔 --selftest 의 걷기 패널 시험(selftest_walk)을 그대로 돌린다 — 버튼 표의 두 판(공개 목록 w3_moves_public 이 있는 판·없는 판) 모두.
- HTTP 경로: 페이지·/api/state·/api/action 을 실제 서버(임시 포트)로 지나며 재전송·새로고침·교차 출처·오래된 요청을 본다.
- 상태기 단위 시험: 응답과 디버그 프레임의 도착 순서, 늦은 수락 응답, 다른 요청의 넘겨받음, 형식이 어긋난 브리지 상태.
합성 브리지(console_demo.DemoW3Bridge)의 판정은 실제 감독기·래치 클록이지만 접촉·모드 전환은 모사다 — 계약·순서 검증이다.
"""
import copy
import html
import json
import re
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jetson.scripts import operator_console as c
from jetson.scripts import console_state as cs

pytestmark = pytest.mark.skipif(c.WALK_IMPORT_ERROR is not None, reason=f"걷기 모듈 없음: {c.WALK_IMPORT_ERROR}")
VAL = ("label", "vx", "vy", "wz", "dur_s")


@pytest.fixture(autouse=True)
def no_real_process():
    """걷기 패널 경로는 프로세스를 띄우거나 pgrep 을 부르지 않는다."""
    with patch.object(c.subprocess, "run", side_effect=AssertionError("real process forbidden")), \
            patch.object(c.subprocess, "Popen", side_effect=AssertionError("real process forbidden")):
        yield


class PublicApi:
    """공개 목록이 **있는** 판의 모양(주 작업자 커밋 6c62defc 의 w3_deploy 와 같은 API)을 원본 모듈 위에 덧씌운다 — 원본에 없을 때의 대역."""

    def __init__(self, real, disabled):
        self._real = real
        self.W3_MOVES = copy.deepcopy(real.W3_MOVES)
        for name, why in disabled.items():
            self.W3_MOVES[name].update(enabled=False, why_disabled=why)

    def __getattr__(self, key):
        return getattr(self._real, key)

    def w3_move_enabled(self, name):
        return bool(self.W3_MOVES[name].get("enabled", True))

    def w3_moves_public(self):
        return [{"name": k, **{f: m[f] for f in VAL}, "enabled": bool(m.get("enabled", True)), "why_disabled": m.get("why_disabled")}
                for k, m in self.W3_MOVES.items()]

    def w3_move_cmd(self, name, limits=None, allow_disabled=False):
        if name in self.W3_MOVES and not allow_disabled and not self.W3_MOVES[name].get("enabled", True):
            raise ValueError(f"비활성 걷기 버튼 {name!r}: {self.W3_MOVES[name].get('why_disabled', '사유 미기록')}")
        return self._real.w3_move_cmd(name)


class NoPublicApi:
    """공개 목록이 **없는** 판(1007 초판)의 모양: w3_moves_public·w3_move_enabled 가 없고 w3_move_cmd 는 비활성을 거르지 않는다."""

    def __init__(self, real):
        self._real = real
        self.W3_MOVES = real.W3_MOVES

    def __getattr__(self, key):
        if key in ("w3_moves_public", "w3_move_enabled"):
            raise AttributeError(key)
        return getattr(self._real, key)

    def w3_move_cmd(self, name, limits=None):
        try:
            return self._real.w3_move_cmd(name, allow_disabled=True)      # 옛 함수에는 비활성 검사가 없었다(대역 안에서만 쓰는 인자)
        except TypeError:
            return self._real.w3_move_cmd(name)


def test_selftest_walk_native_table(tmp_path):
    lines = c.selftest_walk(str(tmp_path))
    assert len(lines) >= 6 and any("⑫" in line for line in lines), lines


def test_selftest_walk_with_public_list_and_disabled_buttons(tmp_path):
    """공개 목록(w3_moves_public)이 있고 두 버튼이 비활성인 판 — 원본이 이미 그 판이면 그대로, 아니면 같은 API 의 대역으로."""
    real = c._W3
    if callable(getattr(real, "w3_moves_public", None)) and any(not m["enabled"] for m in real.w3_moves_public()):
        api = real
    else:
        api = PublicApi(real, {"right1": "X14 대역: 방향 부호 일치 63 %", "back1": "X14 대역: 방향 부호 일치 80 %"})
    with patch.object(c, "_W3", api):
        moves = {m["name"]: m for m in c.walk_moves()}
        off = [n for n, m in moves.items() if not m["enabled"]]
        assert off and all(moves[n]["why_disabled"] for n in off)
        for n in off:
            code, message = c.walk_move_block(n)
            assert code == 409 and "비활성" in message and moves[n]["why_disabled"] in message
            with pytest.raises(ValueError, match="비활성"):
                c.walk_cmd_payload("move", "rid", 1, n)              # 원본 함수도 거부한다(콘솔은 우회 인자를 넘기지 않는다)
        lines = c.selftest_walk(str(tmp_path))
    assert any("공개 목록 있음" in line and f"비활성 {len(off)}" in line for line in lines), lines[0]
    assert any("원본 비활성" in line for line in lines), lines[-1]


def test_selftest_walk_without_public_list_fallback(tmp_path):
    """공개 목록이 없는 판(대체 경로): 표에 enabled False 가 있으면 원본 함수가 거르지 않아도 콘솔이 거른다."""
    real = c._W3
    with patch.object(c, "_W3", NoPublicApi(real)):
        assert not callable(getattr(c._W3, "w3_moves_public", None))
        assert [m["name"] for m in c.walk_moves()] == list(real.W3_MOVES)
        lines = c.selftest_walk(str(tmp_path))
        assert any("공개 목록 없음(대체 경로)" in line for line in lines), lines[0]
        with patch.dict(real.W3_MOVES["left1"], {"enabled": False, "why_disabled": "대체 경로 시험"}):
            assert c._W3.w3_move_cmd("left1")["vy"] == real.W3_MOVES["left1"]["vy"]      # 옛 원본 함수는 통과시킨다 …
            assert c.walk_move_block("left1") == (409, "왼쪽 옆걸음: 비활성 걷기 버튼 — 대체 경로 시험")   # … 콘솔이 막는다
            with c.WalkBench(str(tmp_path), c.WALK_CKPTS[0]) as bn:
                bn.prepare()
                code, res = bn.req("w3_move", move="left1")
                assert code == 409 and "대체 경로 시험" in res["message"] and bn.sent == []


def test_sim_reference_is_labelled_and_only_on_enabled_buttons(tmp_path):
    """시뮬 참고 이동량은 '시뮬 중앙값·실물 미확인' 표기와 함께 활성 버튼에만 실린다(명령 값이 아니다)."""
    assert "시뮬" in c.WALK_SIM_REF_NOTE and "실물 미확인" in c.WALK_SIM_REF_NOTE
    assert set(c.WALK_SIM_REF) <= set(c._W3.W3_MOVES)
    with patch.dict(c._W3.W3_MOVES["fwd1"], {"enabled": False, "why_disabled": "시험"}):
        with c.WalkBench(str(tmp_path), c.WALK_CKPTS[0]) as bn:
            buttons = {b["name"]: b for b in bn.walk()["buttons"]}
            assert buttons["fwd1"]["sim_ref"] is None and buttons["fwd1"]["allowed"] is False
            assert buttons["fwd3"]["sim_ref"] == c.WALK_SIM_REF["fwd3"]
            off_button = c.render_page().split('data-walk="fwd1"', 1)[1].split("</button>", 1)[0]
            assert c.WALK_SIM_REF["fwd1"] not in off_button and "비활성" in off_button, off_button
    page = c.render_page()
    assert f'<small>시뮬 {c.WALK_SIM_REF["fwd3"]}</small>' in page


def test_walk_keys_are_the_1007_pair_plus_the_requiet_composite_key():
    """1008: 걷기 키 = 1007 의 두 키(이름·순서·표시·기동 인자 그대로) + 둘째 합성 키(첫 합성 키 + KE 감지기 갇힘 해소 = 시뮬 합성 평가 구성).
    그 밖의 키는 없다. 둘째 합성 키의 브리지 인자는 첫 합성 키와 같다(다른 것은 브리지 명세의 감지기 설정뿐).
    1008 오후: + 셋째 합성 키(둘째 합성 키 + 공중·착지 구간 서기 정책 G21h — 공중 정책 시작을 받는 키). 맨 뒤에 붙고 앞의 세 키는 그대로다."""
    old = {"stand_w3s3d_15800": ["--experimental-w3"], "stand_kew3_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"]}
    new = {"stand_kew3q_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"],
           "stand_kew3qa_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"],
           "stand_kew3qa_g21a_g24ra_w3nbjd2_17798": ["--experimental-w3", "--experimental-kew3"]}      # 1010: 새 몸 종종걸음 키(맨 뒤)
    assert list(cs.WALK_KEYS) == list(old) + list(new) == c.WALK_CKPTS and set(c.WALK_BRIDGE_ARGS) == set(old) | set(new)
    assert cs.WALK_KEYS["stand_w3s3d_15800"] == {"composite": False, "label": "걷기 실험 · W3 단독"}
    assert cs.WALK_KEYS["stand_kew3_g21a_g24ra_w3s3d15800"] == {"composite": True, "label": "걷기 실험 · 서기 KE + 걷기 W3 합성"}
    q = cs.WALK_KEYS["stand_kew3q_g21a_g24ra_w3s3d15800"]
    assert set(q) == {"composite", "label"} and q["composite"] is True
    # 표시 이름: 걷기 실험 키임 · 무엇이 다른지(갇힘 해소) · 시뮬 평가 구성임 · 실물 미검증. 앞의 두 키 표시에는 그 문구가 없다
    assert all(s in q["label"] for s in ("걷기 실험", "서기 KE + 걷기 W3 합성 + 갇힘 해소(시뮬 평가 구성)", "실물 미검증")), q["label"]
    assert not any("갇힘 해소" in cs.WALK_KEYS[k]["label"] or "실물 미검증" in cs.WALK_KEYS[k]["label"] for k in old)
    assert len({v["label"] for v in cs.WALK_KEYS.values()}) == len(cs.WALK_KEYS)
    # 넷째 걷기 키(공중 정책 시작): air_start 표시는 이 키에만 있고, 앞의 세 항목에는 그 키 자체가 없다(= False 로 읽는다)
    h = cs.WALK_KEYS["stand_kew3qa_g21a_g24ra_w3s3d15800"]
    assert set(h) == {"composite", "air_start", "label"} and h["composite"] is True and h["air_start"] is True
    assert h["label"] == "걷기 실험 · 서기 KE + 걷기 W3 합성 + 갇힘 해소 + 공중·착지 정책(G21h) · 공중 정책 시작 가능 · 실물 미검증"
    assert all(s in h["label"] for s in ("걷기 실험", "서기 KE + 걷기 W3 합성 + 갇힘 해소 + 공중·착지 정책(G21h)", "공중 정책 시작 가능", "실물 미검증")), h["label"]
    air = ["stand_kew3qa_g21a_g24ra_w3s3d15800", "stand_kew3qa_g21a_g24ra_w3nbjd2_17798"]      # 1010: 새 몸 종종걸음 키도 공중 정책 시작을 받는다
    others = [k for k in cs.WALK_KEYS if k not in air]
    assert all("air_start" not in cs.WALK_KEYS[k] and "공중 정책 시작" not in cs.WALK_KEYS[k]["label"] and "G21h" not in cs.WALK_KEYS[k]["label"] for k in others)
    assert [k for k in cs.WALK_KEYS if cs.walk_air_start(k)] == air
    assert not cs.walk_air_start("stand_crouch_g12") and not cs.walk_air_start("모르는 키") and not cs.walk_air_start(None)
    assert all(cs.default_mode(k) == "air" for k in air) and all(cs.default_mode(k) == "ground" for k in others + list(c.CKPT_CHOICES) + [None])
    for key, flags in {**old, **new}.items():
        assert c.WALK_BRIDGE_ARGS[key] == flags
        assert c.bridge_live_argv(key)[2:] == ["--live", "--checkpoint", key] + flags + ["--w3-heading-loop", "off"]
        assert c.bridge_selftest_argv(key)[2:] == ["--selftest", "--checkpoint", key] + flags
    for key in c.CKPT_CHOICES:                                  # 기존 선택지의 기동 인자는 그대로(실험 플래그 없음)
        assert c.bridge_live_argv(key)[2:] == ["--live", "--checkpoint", key] and c.bridge_selftest_argv(key)[2:] == ["--selftest", "--checkpoint", key]
    assert c.CKPT_ALL == c.CKPT_CHOICES + list(old) + list(new)      # 콘솔이 기동 인자를 아는 키(고를 수 있는 키가 아니다 — 아래 메뉴 시험)


def test_policy_menu_is_the_single_composite_key():
    """1008 승윤님 지시(안 쓰는 정책 정리): 고를 수 있는 정책 = 합성 키 하나. 드롭다운·기본값·시험 설정·정책 로드·계약 검사가 모두 그 키만 받는다.
    1008 오후: 그 한 키 = 공중 정책 시작을 받는 합성 키(종전 합성 키 + 공중·착지 구간 서기 정책 G21h). 시험 방식의 기본값도 그에 맞춰 '공중'이다.
    종전 메뉴 키는 정리된 키가 됐다."""
    keep = "stand_kew3qa_g21a_g24ra_w3nbjd2_17798"      # 1010 저녁: 새 몸 종종걸음 키 하나
    assert c.CKPT_MENU_KEYS == (keep,) and c.CKPT_MENU == [keep] and c.CKPT_DEFAULT == keep and c.TRIAL.checkpoint == keep
    assert c.MODE_DEFAULT == "air" == c.TRIAL.mode == cs.TrialState(keep).mode and cs.TrialState("stand_kew3q_g21a_g24ra_w3s3d15800").mode == "ground"
    assert "정리된 정책" in c.ckpt_menu_reason("stand_kew3q_g21a_g24ra_w3s3d15800")
    page = c.render_page()
    assert re.findall(r'<option value="(stand[A-Za-z0-9_]*)">', page) == [keep]
    assert f'<option value="{keep}">{keep} — {html.escape(cs.WALK_KEYS[keep]["label"])}</option>' in page
    assert not any(f"<option>{k}</option>" in page or f'<option value="{k}">' in page for k in c.CKPT_ALL if k != keep)
    assert c.ckpt_menu_reason(keep) is None
    for k in c.CKPT_ALL:
        if k != keep:
            assert "정리된 정책" in c.ckpt_menu_reason(k), k
            with pytest.raises(ValueError, match="정리된 정책"):
                c.act_bridge_live({"checkpoint": k})
    assert c.ckpt_menu_reason("stand_v7") == "허용되지 않은 체크포인트입니다" and c.ckpt_menu_reason(None) == "허용되지 않은 체크포인트입니다"
    started = []
    with patch.object(c.RUN, "start", side_effect=lambda key, argv, cwd=None: (started.append((key, list(argv))), (True, "mock"))[1]):
        c.act_bridge_live({})                                            # 키를 안 주면 남긴 정책
        c.ACTIONS["selftest"][0]({"checkpoint": "stand_crouch_g12"})     # 정리된 키를 줘도 계약 검사는 남긴 정책으로
    flags = ["--experimental-w3", "--experimental-kew3"]
    assert [a[1][2:] for a in started] == [["--live", "--checkpoint", keep] + flags + ["--w3-heading-loop", "off"], ["--selftest", "--checkpoint", keep] + flags]


def test_policy_menu_never_falls_back_to_an_old_policy_when_walk_module_is_missing():
    """걷기 모듈을 못 불러오면 메뉴가 빈다 — 옛 서기 정책으로 조용히 넘어가지 않고 정책 로드·계약 검사가 사유를 대며 거부한다."""
    with patch.object(c, "WALK_IMPORT_ERROR", "ImportError: 시험"), patch.object(c, "CKPT_MENU", []):
        assert "걷기 모듈" in c.ckpt_menu_reason(c.CKPT_DEFAULT)
        with patch.object(c.RUN, "start", side_effect=AssertionError("띄우면 안 됨")):
            for call in (lambda: c.act_bridge_live({}), lambda: c.ACTIONS["selftest"][0]({})):
                with pytest.raises(ValueError, match="걷기 모듈"):
                    call()
        assert not re.findall(r'<option value="(stand[A-Za-z0-9_]*)">', c.render_page())      # 드롭다운은 비고, 페이지는 만들어진다


def test_sim_reference_of_the_requiet_key_equals_what_the_first_composite_key_shows(tmp_path):
    """1008: 참고 이동량(X14)은 갇힘 해소를 켠 구성으로 잰 값이다 — 둘째 합성 키는 첫 합성 키에 지금 보이는 것과 같은 숫자를 보인다(표는 키별이 아니다)."""
    first, second = "stand_kew3_g21a_g24ra_w3s3d15800", "stand_kew3q_g21a_g24ra_w3s3d15800"
    third = "stand_kew3qa_g21a_g24ra_w3s3d15800"                # 1008 오후: 표가 키별이 아니므로 셋째 합성 키도 같은 숫자를 보인다(KE 인계 뒤의 서기·걷기 정책이 둘째 키와 같다)
    enabled = {m["name"] for m in c.walk_moves() if m["enabled"]}
    shown = {}
    for key in (first, second, third):
        with c.WalkBench(str(tmp_path), key) as bn:
            bn.prepare()                                        # 이 키로 시험을 설정해 정책 실행 단계까지(걷기 세션)
            w = bn.walk()
            assert w["session"] and w["profile"]["checkpoint"] == key and w["profile"]["label"] == cs.WALK_KEYS[key]["label"], w["profile"]
            assert "실물 미확인" in w["sim_ref_note"]
            shown[key] = {b["name"]: b["sim_ref"] for b in w["buttons"]}
    assert shown[second] == shown[first] == shown[third]
    assert {n: v for n, v in shown[second].items() if v} == {n: v for n, v in c.WALK_SIM_REF.items() if n in enabled} and any(shown[second].values())


# ---------------------------------------------------------------- HTTP 경로(실제 서버, 임시 포트)
class Http:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def get(self, path):
        with urlopen(self.base + path, timeout=5) as r:
            return r.status, r.read().decode()

    def post(self, body, headers=None):
        req = Request(self.base + "/api/action", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
        try:
            with urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except HTTPError as e:
            return e.code, json.loads(e.read())


@pytest.mark.parametrize("key", [k for k in cs.WALK_KEYS if cs.WALK_KEYS[k]["composite"]])     # 1008: 합성 키 둘 다(종전에는 마지막 키 = 첫 합성 키 하나)
def test_http_duplicate_post_refresh_reconnect_and_stale(tmp_path, key):
    """중복 클릭(같은 요청 재전송)·새로고침(새 GET)·재접속(새 요청 식별자)·교차 출처·오래된 요청으로 걷기가 반복되지 않는다."""
    with c.WalkBench(str(tmp_path), key) as bn:
        bn.prepare()
        server = ThreadingHTTPServer(("127.0.0.1", 0), c.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            http = Http(server.server_address[1])
            code, page = http.get("/")
            assert code == 200 and page.count('data-walk="') == len(c._W3.W3_MOVES) and 'id="walkCard"' in page
            state = json.loads(http.get("/api/state")[1])
            assert state["walk"]["available"] and state["demo"] is True
            body = {"action": "w3_move", "move": "fwd3", "confirmed": True, "seen": state["server_time"],
                    "boot_id": state["trial"]["boot_id"], "revision": state["trial"]["revision"], "request_id": "walk-http-0001"}
            # 교차 출처 요청은 종전 규칙대로 거부(아무것도 나가지 않는다)
            assert http.post(body, {"Origin": "http://evil.example"})[0] == 403 and bn.sent == []
            first = http.post(body)
            again = http.post(body)                                   # 응답을 못 받은 브라우저의 재전송·중복 클릭
            assert first == again and first[0] == 200 and first[1]["ok"] and len(bn.sent) == 1, (first, again)
            # 새로고침 = 페이지·상태를 새로 받는다 — 진행 중인 걷기 버튼이 그대로 보이고 버튼은 잠겨 있다
            assert http.get("/")[0] == 200
            fresh = json.loads(http.get("/api/state")[1])["walk"]
            assert fresh["current"]["req_id"] == bn.sent[0]["req_id"] and fresh["current"]["stage"] == "walking"
            assert not fresh["available"] and not any(b["enabled"] for b in fresh["buttons"])
            # 재접속한 다른 화면이 같은 버튼을 다시 누른다(새 요청 식별자) → 서버가 진행 중이라 거부
            code, res = http.post(dict(body, request_id="walk-http-0002", seen=fresh and json.loads(http.get("/api/state")[1])["server_time"]))
            assert code == 409 and "진행 중" in res["message"] and len(bn.sent) == 1, (code, res)
            # 걷는 중에도 걷기 끄기는 받는다(revision·boot_id 없이도 — 안전 방향)
            code, res = http.post({"action": "w3_off", "request_id": "walk-http-0003"})
            assert code == 200 and res["ok"] and bn.sent[-1]["cmd"] == "step" and bn.sent[-1]["enable"] is False, (code, res)
            assert http.post({"action": "w3_off", "request_id": "walk-http-0003"})[0] == 200 and len(bn.sent) == 2     # 같은 요청 = 한 번만 발행
            bn.until(lambda w: w["current"]["stage"] == "done", 6.0, "서기 복귀")
            bn.until(lambda w: w["available"], 3.0, "버튼 다시 열림")
            # 연결이 끊겼다 뒤늦게 도착한 요청(화면 시각이 오래됨) → 거부, 발행 없음
            state = json.loads(http.get("/api/state")[1])
            late = dict(body, request_id="walk-http-0004", revision=state["trial"]["revision"], seen=state["server_time"] - 20.0)
            code, res = http.post(late)
            assert code == 409 and "오래됐습니다" in res["message"] and len(bn.sent) == 2, (code, res)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


# ---------------------------------------------------------------- 상태기 단위 시험(가짜 응답·디버그 프레임)
KEY = "stand_kew3_g21a_g24ra_w3s3d15800"


class Rig:
    """TrialState + WalkState 를 가짜 뷰로 직접 돌린다(콘솔·데모 없이). t = 콘솔 시계, 브리지 timestamp 는 호출자가 준다."""

    def __init__(self, key=KEY):
        self.t = 100.0
        self.trial = cs.TrialState(key, clock=lambda: self.t)
        self.trial.completed = set(self.trial.flow())
        self.walk = self.trial.walk
        self.key = key
        self.frame(ts=99.9)

    def view(self, ts, want=False, ge=False, mode="ke", sup=None, fresh=True, recv=None, contact=(True, True)):
        sup = dict({"last_seq": None, "stop_latched": False, "latch_reason": None, "freeze": None}, **(sup or {}))
        bridge = {"live": True, "armed": True, "checkpoint": self.key, "loaded_checkpoint": self.key, "blend": 1.0, "ts": ts,
                  "w3": {"want": want, "ge": ge, "ge_obs": 0.0, "u": 0.0, "f": 0.0}, "w3_sup": sup, "cmd": [0.0, 0.0, 0.0]}
        if cs.WALK_KEYS[self.key]["composite"]:
            bridge["kew3"] = {"mode": mode, "block": None}
        return {"server_time": self.t, "fresh": {"status": True, "imu": True, "foot": True, "bridge": fresh},
                "received": {"bridge": self.t if recv is None else recv, "status": self.t},
                "stage8": {"armed": True, "running": True, "feedback_ok": True}, "bridge": bridge,
                "foot": {"contact": list(contact)}, "imu": {"gyro": [0.01, 0.01, 0.0]},
                "procs": {"bridge": {"alive": True}, "stage8": {"alive": True}}, "process_check_ok": True,
                "session": {"active": True}, "external_controls": [], "walk_error": None,
                "walk_ready": {"contact_s": 1.0, "quiet_s": 1.0, "quiet_w": 0.25}}

    def frame(self, dt=0.05, **kw):
        self.t += dt
        self.trial.update(self.view(**kw))
        return self.walk.cur

    def send(self, name="fwd1", dur_s=0.6, seq=7):
        rid = f"con-test-{name}-{seq}"
        self.walk.begin(self.trial, "move", rid, seq, {"cmd": "step", "enable": True, "req_id": rid, "seq": seq, "dur_s": dur_s},
                        name=name, label=name)
        return rid

    def ack(self, rid, ts, accepted=True, event=None, burst=True, why="ok"):
        ack = {"req_id": rid, "cmd": "step", "accepted": accepted, "why": why, "action": "set_want", "timestamp": ts,
               "sup": {"burst": {"req_id": rid, "dur_s": 0.6, "t_off": ts + 0.6}} if burst and accepted and not event else {}}
        if event:
            ack["event"] = event
        self.walk.feed([{"t": self.t, "ack": ack}])


def test_older_debug_frame_after_accept_ack_does_not_end_the_walk():
    """응답과 디버그 프레임은 다른 토픽이라 순서가 보장되지 않는다 — 수락 '전'에 찍힌 프레임(want 0)이 늦게 와도 걷기 끝으로 보지 않는다."""
    rig = Rig()
    for _ in range(30):
        rig.frame(ts=rig.t)                               # 서기 정착
    assert rig.trial.reason("w3_move") is None, rig.trial.reason("w3_move")
    rid = rig.send()
    rig.ack(rid, ts=200.0)                                # 브리지 시계 200.0 에 수락
    assert rig.frame(ts=199.98, want=False)["stage"] == "walking"      # 수락 전 프레임이 뒤늦게 도착
    assert rig.frame(ts=200.0, want=False)["stage"] == "walking"       # 같은 시각도 '뒤'가 아니다
    assert rig.frame(ts=200.02, want=True, ge=True, mode="w3")["stage"] == "walking"
    cur = rig.frame(ts=200.70, want=False, ge=True, mode="w3")          # 수락 뒤 프레임이 want 0 → 걷기 끝 요청 관측
    assert cur["stage"] == "ending" and cur["end_by"] == "observed"
    assert rig.frame(ts=200.72, want=False, ge=False, mode="w3")["stage"] == "ending"     # 합성: KE 인계 전
    assert rig.frame(ts=201.30, want=False, ge=False, mode="ke")["stage"] == "done"


def test_air_landing_phase_blocks_walk_buttons_until_ke_handoff():
    """1008 공중 정책 시작 키: 브리지 러너가 공중·착지 구간(kew3.mode "air")인 동안은 서기도 걷기도 아니다 — 걷기 버튼은 사유와 함께 잠기고
    (양발이 닿아도), '정착 대기' 시계는 서기 KE 로 넘어간(mode "ke") 뒤에야 돈다."""
    rig = Rig("stand_kew3qa_g21a_g24ra_w3s3d15800")
    assert rig.trial.mode == "air"                              # 공중 정책 시작 키의 기본 방식
    for _ in range(40):
        rig.frame(ts=rig.t, mode="air", contact=(False, False))     # 매달린 채 공중·착지 정책 구동
    st = rig.walk.bridge_status(rig.trial)
    assert st["mode"] == "air" and st["standing"] is False and st["walking"] is False and rig.walk.snapshot(rig.trial)["mode"] == "air"
    why = rig.trial.reason("w3_move")
    assert why and "공중·착지 정책 구간(착지 정착 뒤 KE 인계 대기)" in why, why
    for _ in range(40):
        rig.frame(ts=rig.t, mode="air")                         # 양발이 닿았지만 아직 공중·착지 구간 — 접촉만으로는 열리지 않는다
    why = rig.trial.reason("w3_move")
    assert why and "공중·착지 정책 구간(착지 정착 뒤 KE 인계 대기)" in why and rig.walk._steady_since is None, why
    assert not rig.trial.fault and rig.walk.cur is None
    rig.frame(ts=rig.t, mode="ke")                              # 서기 KE 로 인계 → 이제부터 정착 대기
    why = rig.trial.reason("w3_move")
    assert why and "서기 정착 대기" in why, why
    for _ in range(30):
        rig.frame(ts=rig.t, mode="ke")
    assert rig.trial.reason("w3_move") is None and rig.walk.snapshot(rig.trial)["mode"] == "ke"
    other = Rig()                                               # 다른 합성 키의 문구는 그대로(W3 모드일 때)
    for _ in range(30):
        other.frame(ts=other.t, mode="w3")
    assert "걷기 정책(W3) 모드" in other.trial.reason("w3_move")


def test_burst_end_ack_then_same_tick_frame_is_not_yet_standing():
    rig = Rig("stand_w3s3d_15800")
    for _ in range(30):
        rig.frame(ts=rig.t)
    rid = rig.send()
    rig.ack(rid, ts=300.0)
    rig.frame(ts=300.02, want=True, ge=True)
    rig.ack(rid, ts=300.62, event="burst_end")
    cur = rig.frame(ts=300.62, want=False, ge=False)       # 기한 종료와 같은 틱의 프레임(같은 timestamp) — 뒤의 프레임을 기다린다
    assert cur["stage"] == "ending" and cur["end_by"] == "burst"
    assert rig.frame(ts=300.64, want=False, ge=True)["stage"] == "ending"
    assert rig.frame(ts=300.90, want=False, ge=False)["stage"] == "done"


def test_stale_frames_never_count_as_standing():
    """끊긴(오래된) 디버그 프레임의 want 0·ge 0 은 서기의 근거가 아니다 — '서기 복귀 확인됨'이 되지 않고, 수신 끊김은 고장 래치로 간다."""
    rig = Rig("stand_w3s3d_15800")
    for _ in range(30):
        rig.frame(ts=rig.t)
    rid = rig.send()
    rig.ack(rid, ts=400.0)
    rig.ack(rid, ts=400.6, event="burst_end")
    cur = rig.frame(ts=401.0, want=False, ge=False, fresh=False)
    assert rig.walk.bridge_status(rig.trial)["standing"] is False
    assert cur["stage"] == "interrupted" and "수신 끊김" in cur["why"] and not cur["stand_seen"], cur
    cur = rig.frame(ts=401.1, want=False, ge=False, fresh=True)          # 다시 신선한 프레임 — 중단 기록에 '이후 서기 확인'만 남는다
    assert cur["stage"] == "interrupted" and cur["stand_seen"], cur
    assert rig.trial.reason("w3_move") is not None                      # 고장 래치 뒤에는 걷기 불가(회수 뒤 새 시험)


def test_late_accept_ack_is_adopted_but_nothing_is_resent():
    rig = Rig()
    for _ in range(30):
        rig.frame(ts=rig.t)
    rid = rig.send(name="fwd3", dur_s=2.4)
    for _ in range(45):                                    # 2.25 s 동안 응답 없음(로봇은 걷는 중으로 보인다)
        rig.frame(ts=rig.t + 400.0, want=True, ge=True, mode="w3")
    cur = rig.walk.cur
    assert cur["stage"] == "no_ack" and cur["seen_walking"] and not cur["stand_seen"]
    assert "응답 없음" in rig.trial.reason("w3_move")
    rig.ack(rid, ts=rig.t + 399.0)                         # 뒤늦게 도착한 수락 응답(같은 req_id)
    cur = rig.frame(ts=rig.t + 400.0, want=True, ge=True, mode="w3")
    assert cur["stage"] == "walking" and cur["late_ack"] is True and cur["deadline"] is True
    rig.ack("con-other-fwd1-1", ts=rig.t + 400.0, event="burst_end")       # 남의 요청의 기한 종료 — 이 기록과 무관
    assert rig.frame(ts=rig.t + 400.0, want=True, ge=True, mode="w3")["stage"] == "walking"


def test_other_request_taking_over_and_relatch_are_interruptions():
    rig = Rig("stand_w3s3d_15800")
    for _ in range(30):
        rig.frame(ts=rig.t)
    rid = rig.send()
    rig.ack(rid, ts=500.0)
    cur = rig.frame(ts=500.1, want=True, ge=True, sup={"burst": {"req_id": "cli-host-1-9", "dur_s": 5.0, "t_off": 505.0}})
    assert cur["stage"] == "interrupted" and "다른 요청" in cur["why"]
    rid = rig.send(seq=8)
    rig.ack(rid, ts=510.0)
    cur = rig.frame(ts=510.2, want=False, ge=True, sup={"stop_latched": True, "latch_reason": "guard: lift_guard"})
    assert cur["stage"] == "interrupted" and "guard: lift_guard" in cur["why"]
    assert "정지 래치(guard: lift_guard)" in rig.trial.reason("w3_move") or "정지 래치" in rig.trial.reason("w3_move")
    rid = rig.send(seq=9)
    rig.ack(rid, ts=520.0)
    rig.ack(rid, ts=520.6, event="burst_end")
    cur = rig.frame(ts=520.7, want=True, ge=True)          # 서기 확인 전에 걷기가 다시 켜짐(콘솔 밖 명령)
    assert cur["stage"] == "interrupted" and "다시 켜졌습니다" in cur["why"]


def test_malformed_bridge_state_is_never_read_as_standing_or_unlatched():
    rig = Rig("stand_w3s3d_15800")
    for _ in range(30):
        rig.frame(ts=rig.t)
    assert rig.trial.reason("w3_move") is None
    for broken in ({"want": 0, "ge": 0}, {"want": None, "ge": False}, {"ge": False}):
        view = rig.view(ts=rig.t)
        view["bridge"]["w3"] = broken
        rig.trial.update(view)
        assert "읽지 못했습니다" in rig.trial.reason("w3_move"), (broken, rig.trial.reason("w3_move"))
        assert rig.walk.bridge_status(rig.trial)["standing"] is False
    view = rig.view(ts=rig.t)
    view["bridge"]["w3_sup"]["stop_latched"] = None
    rig.trial.update(view)
    assert "읽지 못했습니다" in rig.trial.reason("w3_move")
    view = rig.view(ts=rig.t)
    del view["bridge"]["w3_sup"]
    rig.trial.update(view)
    assert "w3·w3_sup" in rig.trial.reason("w3_move")


def test_fault_latch_interrupts_and_estop_reason_is_recorded():
    rig = Rig()
    for _ in range(30):
        rig.frame(ts=rig.t)
    rid = rig.send()
    rig.ack(rid, ts=600.0)
    rig.frame(ts=600.1, want=True, ge=True, mode="w3")
    rig.trial.latch("sensor_stale")
    cur = rig.walk.cur
    assert cur["stage"] == "interrupted" and "sensor_stale" in cur["why"]
    assert "정지·고장" in rig.trial.reason("w3_move") and rig.trial.reason("w3_off") is None      # 끄기는 고장 래치 뒤에도 받는다
    events = [e["event"] for e in rig.trial.history if e["event"].startswith("walk")]
    assert events == ["walk_request", "walk_ack", "walk_interrupted"], events


def test_settle_wait_follows_bridge_enable_thresholds():
    """걷기 직후에는 '조용한 양발 서기'가 브리지 켜기 조건만큼 이어질 때까지 콘솔이 잠가 둔다(표시용 — 최종 판정은 브리지)."""
    rig = Rig("stand_w3s3d_15800")
    rig.frame(ts=rig.t)
    assert "정착 대기" in rig.trial.reason("w3_move")
    for _ in range(22):                                    # 1.10 s — 아직(여유 0.15 s 포함 1.15 s)
        rig.frame(ts=rig.t)
    assert "정착 대기" in rig.trial.reason("w3_move")
    rig.frame(ts=rig.t)
    rig.frame(ts=rig.t)
    assert rig.trial.reason("w3_move") is None
    rig.frame(ts=rig.t, contact=(True, False))             # 한발 접촉 → 시계가 다시 0 부터
    assert "양발 접촉" in rig.trial.reason("w3_move")
    rig.frame(ts=rig.t)
    assert "정착 대기" in rig.trial.reason("w3_move")
    view = rig.view(ts=rig.t)
    view["imu"]["gyro"] = [0.4, 0.0, 0.0]                  # 흔들림(|ω_xy| ≥ 문턱) → 정착으로 세지 않는다
    for _ in range(40):
        rig.t += 0.05
        view["server_time"] = view["received"]["bridge"] = rig.t
        rig.trial.update(view)
    assert "정착 대기" in rig.trial.reason("w3_move")
