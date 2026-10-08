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


def test_cli_parses_new_prep_flags(rc, monkeypatch):
    seen = []
    monkeypatch.setattr(rc, "cmd_prep", lambda a: seen.append((a.yes, a.no_hang)))      # main() 은 호출 시점의 전역 cmd_prep 을 묶는다
    for argv in (["robotctl", "prep"], ["robotctl", "prep", "-y"], ["robotctl", "prep", "--no-hang"]):
        monkeypatch.setattr(rc.sys, "argv", argv)
        rc.main()
    assert seen == [(False, False), (True, False), (False, True)]
