#!/usr/bin/env python3
"""모터 CAN 채널의 단일 원본 (2026-09-07).

배경: 라이브 버스가 PCAN USB(can1)에서 다른 어댑터로 바뀔 수 있는데, 채널 이름이
스크립트 15곳에 `default="can1"` 로 흩어져 있었다. 한 곳이라도 빠지면 회수 경로가
없는 버스를 열고 조용히 실패한다(release_all 이 그렇게 되면 릴리즈가 안 나간다).

해석 순서 — 먼저 정해진 것이 이긴다:
  1. 명시 인자 (`--channel can0`)
  2. 환경변수 ROBOT_CAN_CHANNEL
  3. 파일 ~/rl_calib/can_channel  (한 줄, 예: can0)
  4. 레거시 기본값 can1  (0727~0809 실기 전체가 이 버스)

자동 탐지는 **일부러 넣지 않는다.** 올라와 있는 CAN 인터페이스를 골라 잡으면
모터가 없는 빈 버스를 "찾았다"고 선언하게 되고, 그 상태로 stage8 을 띄우면
인수 실패를 채널 문제가 아닌 다른 원인으로 오진하게 된다. 채널은 사람이 못 박는다.

사용:
    from can_channel import resolve
    ap.add_argument("--channel", default=None)      # None → resolve()
    ch = resolve(args.channel)
셀프테스트: python3 can_channel.py --selftest
"""
from __future__ import annotations

import os
import re
import sys

LEGACY = "can1"                       # 0727~0809 실기 버스 (PCAN USB)
ENV_VAR = "ROBOT_CAN_CHANNEL"
CONF = os.path.expanduser("~/rl_calib/can_channel")
_VALID = re.compile(r"^can[0-9]+$")


def _clean(v):
    """빈 값·주석·공백을 걸러 유효한 채널 이름만 돌려준다."""
    if not v:
        return None
    v = v.split("#", 1)[0].strip()
    return v if _VALID.match(v) else None


def resolve(explicit=None, conf=CONF, env=None):
    env = os.environ if env is None else env
    for v in (explicit, env.get(ENV_VAR)):
        c = _clean(v)
        if c:
            return c
    try:
        with open(conf, encoding="utf-8") as fh:
            for line in fh:
                c = _clean(line)
                if c:
                    return c
    except OSError:
        pass
    return LEGACY


def source(explicit=None, conf=CONF, env=None):
    """어디서 정해졌는지 — 로그에 남겨 오진을 막는다."""
    env = os.environ if env is None else env
    if _clean(explicit):
        return "인자"
    if _clean(env.get(ENV_VAR)):
        return f"환경변수 {ENV_VAR}"
    try:
        with open(conf, encoding="utf-8") as fh:
            for line in fh:
                if _clean(line):
                    return f"파일 {conf}"
    except OSError:
        pass
    return f"기본값(레거시 {LEGACY})"


def exists(ch):
    return os.path.exists(f"/sys/class/net/{ch}")


def is_up(ch):
    try:
        with open(f"/sys/class/net/{ch}/operstate", encoding="utf-8") as fh:
            return fh.read().strip() in ("up", "unknown")
    except OSError:
        return False


def describe(explicit=None):
    ch = resolve(explicit)
    st = "없음" if not exists(ch) else ("UP" if is_up(ch) else "DOWN")
    return f"{ch} [{st}] ← {source(explicit)}"


def _selftest():
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"{name}: {got!r} != {want!r}")

    tmp = "/tmp/_can_channel_selftest"
    env = {}
    chk("기본값", resolve(conf="/nonexistent", env=env), LEGACY)
    chk("명시 인자", resolve("can0", conf="/nonexistent", env=env), "can0")
    chk("잘못된 인자는 무시", resolve("eth0", conf="/nonexistent", env=env), LEGACY)
    chk("환경변수", resolve(conf="/nonexistent", env={ENV_VAR: "can3"}), "can3")
    chk("인자가 환경변수를 이긴다",
        resolve("can0", conf="/nonexistent", env={ENV_VAR: "can3"}), "can0")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("# 주석\n\n  can0  \n")
    chk("파일(주석·공백)", resolve(conf=tmp, env=env), "can0")
    chk("환경변수가 파일을 이긴다", resolve(conf=tmp, env={ENV_VAR: "can3"}), "can3")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("# 전부 주석\n")
    chk("빈 파일 → 레거시", resolve(conf=tmp, env=env), LEGACY)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("can0 # PCAN 대신 내장 버스\n")
    chk("파일 줄끝 주석", resolve(conf=tmp, env=env), "can0")
    os.unlink(tmp)
    chk("source 표기", source("can0", conf="/nonexistent", env=env), "인자")
    if fails:
        print("[selftest] can_channel FAIL\n  " + "\n  ".join(fails))
        return 1
    print("[selftest] can_channel: 우선순위 인자>환경>파일>레거시 · 주석·공백·"
          "무효값 처리 — ALL PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(describe(sys.argv[1] if len(sys.argv) > 1 else None))
