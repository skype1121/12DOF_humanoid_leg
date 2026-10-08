"""robotctl prep 이 hang 을 흡수한 뒤의 순서·확인·건너뛰기 (1008). 콘솔 서버는 가짜(상태 dict) — 명령 순서와 '모터가 움직이기 전 확인'만 본다."""
import importlib.util
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("robotctl", os.path.join(HERE, "..", "jetson", "scripts", "robotctl.py"))
PREP5 = ["can_up", "sensors", "boot_pose", "selftest", "stage8"]


class FakeConsole:
    """콘솔 /api/state·/api/action 대역: 액션을 받으면 곧바로 완료로 친다(hang 단계는 hang_hold 상태도 바꾼다)."""

    def __init__(self, completed=(), refuse=()):
        self.completed = list(completed); self.sent = []; self.refuse = set(refuse); self.hold = {}

    def state(self):
        labels = {k: {"enabled": True, "reason": None, "label": k} for k in PREP5 + ["hang_stand", "gain_ramp"]}
        return {"trial": {"completed": list(self.completed), "pending": None, "fault": None, "last_result": None, "allowed": labels,
                          "boot_id": "b", "revision": 1, "phase": "prep"},
                "hang_hold": dict(self.hold)}

    def act(self, action, extra=None, confirmed=True, quiet=False):
        self.sent.append(action)
        if action in self.refuse:
            return False
        self.completed.append(action)
        if action == "hang_stand":
            self.hold = {"phase": "에어유지", "kp": 20.0}
        if action == "gain_ramp":
            self.hold = {"phase": "운용대기", "kp": 150.0}
        return True


@pytest.fixture
def rc(monkeypatch):
    mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def wire(rc, monkeypatch, fake, answers=()):
    monkeypatch.setattr(rc, "state", fake.state); monkeypatch.setattr(rc, "act", fake.act)
    asked = []; it = iter(answers)

    def _input(prompt):
        asked.append(prompt)
        return next(it)
    monkeypatch.setattr("builtins.input", _input)
    return asked


def args(**kw):
    return types.SimpleNamespace(**{"yes": False, "no_hang": False, **kw})


def test_prep_runs_prep_steps_then_asks_once_then_hangs(rc, monkeypatch, capsys):
    fake = FakeConsole(); asked = wire(rc, monkeypatch, fake, answers=["y"])
    rc.cmd_prep(args())
    assert fake.sent == PREP5 + ["hang_stand", "gain_ramp"]
    assert len(asked) == 1 and "줄이 로봇을 지지" in asked[0]          # 모터가 움직이기 전 한 번만 묻는다
    out = capsys.readouterr().out
    assert "robotctl policy-on" in out and "robotctl hang" not in out      # 다음 안내에 hang 이 더는 없다


def test_prompt_comes_after_prep_steps_and_before_any_motion(rc, monkeypatch):
    fake = FakeConsole(); order = []
    monkeypatch.setattr(rc, "state", fake.state)
    monkeypatch.setattr(rc, "act", lambda a, *k, **kw: (order.append(a), fake.act(a))[1])
    monkeypatch.setattr("builtins.input", lambda p: (order.append("ASK"), "y")[1])
    rc.cmd_prep(args())
    assert order == PREP5 + ["ASK", "hang_stand", "gain_ramp"]


def test_decline_sends_no_motion_and_exits_nonzero(rc, monkeypatch, capsys):
    fake = FakeConsole(); wire(rc, monkeypatch, fake, answers=["n"])
    with pytest.raises(SystemExit) as e:
        rc.cmd_prep(args())
    assert e.value.code == 1 and fake.sent == PREP5                     # 준비 단계까지만, 모터 명령 없음
    assert "공중 자세 유지는 시작하지 않았습니다" in capsys.readouterr().out


def test_yes_flag_skips_prompt(rc, monkeypatch):
    fake = FakeConsole(); asked = wire(rc, monkeypatch, fake)
    rc.cmd_prep(args(yes=True))
    assert asked == [] and fake.sent[-2:] == ["hang_stand", "gain_ramp"]


def test_no_hang_stops_before_motion_without_asking(rc, monkeypatch, capsys):
    fake = FakeConsole(); asked = wire(rc, monkeypatch, fake)
    rc.cmd_prep(args(no_hang=True))
    assert fake.sent == PREP5 and asked == []
    assert "--no-hang" in capsys.readouterr().out


def test_completed_steps_are_skipped_including_hang(rc, monkeypatch, capsys):
    fake = FakeConsole(completed=PREP5 + ["hang_stand", "gain_ramp"]); asked = wire(rc, monkeypatch, fake)
    rc.cmd_prep(args())
    assert fake.sent == [] and asked == []                               # 전부 끝났으면 아무것도 보내지 않고 묻지도 않는다
    assert "이미 완료" in capsys.readouterr().out


def test_resume_from_gain_ramp_when_hang_stand_already_done(rc, monkeypatch):
    fake = FakeConsole(completed=PREP5 + ["hang_stand"]); fake.hold = {"phase": "에어유지", "kp": 20.0}
    asked = wire(rc, monkeypatch, fake, answers=["y"])
    rc.cmd_prep(args())
    assert fake.sent == ["gain_ramp"] and len(asked) == 1               # 게인만 남았어도 한 번 묻는다(강성이 올라간다)


def test_prep_step_refusal_stops_before_hang(rc, monkeypatch):
    fake = FakeConsole(refuse={"boot_pose"}); asked = wire(rc, monkeypatch, fake)
    with pytest.raises(SystemExit):
        rc.cmd_prep(args())
    assert fake.sent == ["can_up", "sensors", "boot_pose"] and asked == []   # 영점 확인이 막히면 그 뒤(공중 자세 포함)로 가지 않는다


def test_hang_refusal_exits_without_gain_ramp(rc, monkeypatch):
    fake = FakeConsole(refuse={"hang_stand"}); wire(rc, monkeypatch, fake, answers=["y"])
    with pytest.raises(SystemExit):
        rc.cmd_prep(args())
    assert fake.sent == PREP5 + ["hang_stand"]


def test_hang_command_still_works_as_alias(rc, monkeypatch):
    fake = FakeConsole(completed=PREP5); asked = wire(rc, monkeypatch, fake, answers=["y"])
    rc.cmd_hang(args())
    assert fake.sent == ["hang_stand", "gain_ramp"] and len(asked) == 1
    fake2 = FakeConsole(completed=PREP5); wire(rc, monkeypatch, fake2, answers=["n"])
    with pytest.raises(SystemExit):
        rc.cmd_hang(args())
    assert fake2.sent == []


class FakePolicyOnConsole:
    """정책 ON 대역: policy_on 을 받으면 곧바로 완료로 친다. mode = 콘솔의 시험 방식(None 이면 state 에 그 필드가 없다 — 옛 콘솔)."""

    def __init__(self, mode, refuse=False, fault=None):
        self.mode, self.refuse, self.fault = mode, refuse, fault
        self.completed, self.sent = [], []

    def state(self):
        trial = {"checkpoint": "stand_kew3qa_g21a_g24ra_w3s3d15800", "completed": list(self.completed), "pending": None, "fault": self.fault,
                 "last_result": None, "macro": None, "boot_id": "b", "revision": 1, "phase": "공중 자세 유지",
                 "allowed": {"policy_on": {"enabled": True, "reason": None, "label": "정책 ON"}}}
        if self.mode is not None:
            trial["mode"] = self.mode
        return {"trial": trial, "imu": {"fwd": 1.0, "lat": -0.5}}

    def act(self, action, extra=None, confirmed=True, quiet=False):
        self.sent.append(action)
        if self.refuse:
            return False
        self.completed.append(action)
        return True


GROUND_PROMPT = ("정책 ON: 줄(지그)이 로봇을 지지하고 발이 바닥에서 떨어져 있으며 수평 자세(level)가 적용됐나요? "
                 "정책은 착지 확인 뒤에만 시작합니다 [y/N] ")
GROUND_DONE = ("✓ 착지 대기 — 정책 미구동·수평 자세 유지 중. 이제 지그를 해제하세요(양발 착지 0.3 s·정착 뒤 정책 자동 시작). "
               "'robotctl watch' 로 관찰, 문제 시 robotctl estop")


def test_policy_on_air_mode_says_policy_starts_in_the_air(rc, monkeypatch, capsys):
    """1008 오후 공중 방식: 확인 문구가 줄 지지·발이 뜸·수평 자세·다리 주변 사람 없음을 묻고 '공중에서 바로 구동'이라고 말한다. 끝나면 공중 구동 중·지그를 내리라고 안내한다."""
    fake = FakePolicyOnConsole("air"); asked = wire(rc, monkeypatch, fake, answers=["y"])
    rc.cmd_policy_on(args())
    assert fake.sent == ["policy_on"] and len(asked) == 1
    for part in ("공중 방식", "줄(지그)이 로봇을 지지", "발이 바닥에서 떨어져", "수평 자세(level)", "다리 주변에 사람이 없나요", "공중에서 바로 구동", "착지를 기다리지 않음"):
        assert part in asked[0], (part, asked[0])
    assert "착지 확인 뒤에만" not in asked[0]
    out = capsys.readouterr().out
    assert "방식 air(공중 정책 시작)" in out and "✓ 정책이 공중에서 구동 중" in out and "지그를 천천히 내리세요" in out and "서기 KE 로 자동 인계" in out
    assert "착지 대기" not in out and "robotctl estop" in out


@pytest.mark.parametrize("mode", ["ground", None])
def test_policy_on_ground_mode_keeps_the_touchdown_wording(rc, monkeypatch, capsys, mode):
    """접지 방식(그리고 방식 필드가 없는 옛 콘솔): 확인 문구·완료 안내가 종전 글자 그대로다 — 정책은 착지 확인 뒤에만 시작한다."""
    fake = FakePolicyOnConsole(mode); asked = wire(rc, monkeypatch, fake, answers=["y"])
    rc.cmd_policy_on(args())
    assert fake.sent == ["policy_on"] and asked == [GROUND_PROMPT]
    out = capsys.readouterr().out
    assert GROUND_DONE in out and "공중에서" not in out and "방식 air" not in out


@pytest.mark.parametrize("mode", ["air", "ground"])
def test_policy_on_decline_sends_nothing_and_yes_flag_skips_prompt(rc, monkeypatch, mode):
    fake = FakePolicyOnConsole(mode); asked = wire(rc, monkeypatch, fake, answers=["n"])
    with pytest.raises(SystemExit) as e:
        rc.cmd_policy_on(args())
    assert e.value.code == 2 and fake.sent == [] and len(asked) == 1          # 거절 = 아무것도 보내지 않는다
    fake = FakePolicyOnConsole(mode); asked = wire(rc, monkeypatch, fake)
    rc.cmd_policy_on(args(yes=True))
    assert fake.sent == ["policy_on"] and asked == []


@pytest.mark.parametrize("mode", ["air", "ground"])
def test_policy_on_refusal_or_fault_exits_nonzero_without_success_message(rc, monkeypatch, capsys, mode):
    fake = FakePolicyOnConsole(mode, refuse=True); wire(rc, monkeypatch, fake, answers=["y"])
    with pytest.raises(SystemExit) as e:
        rc.cmd_policy_on(args())
    assert e.value.code == 1 and fake.sent == ["policy_on"]
    fake = FakePolicyOnConsole(mode, fault="정책 ON 중단(제어기 ARM): 모의")
    monkeypatch.setattr(fake, "act", lambda action, *a, **k: (fake.sent.append(action), True)[1])      # 접수는 됐지만 완료되지 않고 고장 래치
    wire(rc, monkeypatch, fake, answers=["y"])
    with pytest.raises(SystemExit):
        rc.cmd_policy_on(args())
    out = capsys.readouterr().out
    assert "정책 ON 중단" in out and "✓ 정책이 공중에서 구동 중" not in out and "✓ 착지 대기" not in out


class FakeStepConsole(FakePolicyOnConsole):
    """단계별 명령 대역: 어떤 액션이든 받으면 곧바로 완료로 친다(허용 표시는 전부 켜 둔다)."""

    def state(self):
        s = super().state()
        s["trial"]["allowed"] = {k: {"enabled": True, "reason": None, "label": k} for k in ("handoff", "baseline", "arm", "bridge", "stand", "auto_level_off")}
        return s


@pytest.mark.parametrize("mode", ["air", "ground", None])
def test_stepwise_handoff_and_stand_prompts_follow_the_mode(rc, monkeypatch, capsys, mode):
    """기본 방식이 air 가 되면서 단계별 명령의 확인 문구도 방식을 따른다: 접지 방식(그리고 방식 필드 없는 옛 콘솔)은 종전 글자 그대로,
    공중 방식은 '줄이 지지·발이 뜸'(인계)·'공중에서 바로 구동'(정책 시작)을 묻는다. 보내는 액션·순서는 방식과 무관하게 같다."""
    fake = FakeStepConsole(mode); asked = wire(rc, monkeypatch, fake, answers=["y", "y"])
    rc.cmd_handoff(args())
    rc.cmd_stand(args())
    assert fake.sent == ["handoff", "baseline", "arm", "bridge", "stand"] and len(asked) == 2
    if mode == "air":
        assert all(s in asked[0] for s in ("공중 방식", "줄(지그)이 로봇을 지지", "발이 바닥에서 떨어져", "robotctl config ground")) and "발이 바닥에 있고" not in asked[0]
        assert all(s in asked[1] for s in ("공중 방식", "공중에서 바로 구동", "다리 주변에 사람이 없나요")) and "줄 느슨" not in asked[1]
    else:
        assert asked == ["제어 인계: 발이 바닥에 있고 줄이 느슨하며 몸통이 거의 수평인가요? [y/N] ",
                         "정책 시작(stand_kew3qa_g21a_g24ra_w3s3d15800): 줄 느슨·손 떼기 확인? [y/N] "]
    assert ("방식 air" in capsys.readouterr().out) is (mode == "air")


def test_status_line_shows_the_air_landing_phase(rc):
    """상태 한 줄(robotctl status·watch): 브리지 합성 러너가 공중·착지 구간(kew3.mode air)이면 그렇게 적고, KE 로 넘어가면 사라진다(표시 전용)."""
    st = {"trial": {"phase": "정책 제어 · 지지 제거"}, "bridge": {"live": True, "loaded_checkpoint": "stand_kew3qa_g21a_g24ra_w3s3d15800", "kew3": {"mode": "air"}}}
    assert "공중·착지 정책 구간(KE 인계 전·걷기 불가)" in rc.line(st) and "정책 LIVE stand_kew3qa_g21a_g24ra_w3s3d15800" in rc.line(st)
    for kew3 in ({"mode": "ke"}, {"mode": "w3"}, None):
        st["bridge"]["kew3"] = kew3
        assert "공중·착지" not in rc.line(st)
    assert "공중·착지" not in rc.line({"trial": {}, "bridge": None})


def test_docstring_describes_both_modes(rc):
    doc = rc.__doc__
    assert "air    = 공중 정책 시작(hang_arm)" in doc and "ground = 착지 자동 인수(touchdown start)" in doc
    assert "방식 air(1008 오후): **정책이 공중에서 바로 구동**" in doc and "방식 ground: 착지 대기" in doc
    assert "남긴 정책에서 거부된다" not in doc                                 # 옛 문장(공중 방식은 거부)이 남아 있지 않다


def test_cli_parses_new_prep_flags(rc, monkeypatch):
    seen = []
    monkeypatch.setattr(rc, "cmd_prep", lambda a: seen.append((a.yes, a.no_hang)))      # main() 은 호출 시점의 전역 cmd_prep 을 묶는다
    for argv in (["robotctl", "prep"], ["robotctl", "prep", "-y"], ["robotctl", "prep", "--no-hang"]):
        monkeypatch.setattr(rc.sys, "argv", argv)
        rc.main()
    assert seen == [(False, False), (True, False), (False, True)]
