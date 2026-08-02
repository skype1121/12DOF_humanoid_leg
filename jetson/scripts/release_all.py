"""전체 릴리즈 — 현재각 홀드에서 게인 램프다운 후 MIT 해제 (estop 프로토콜 최종 단계).

⚠ 반드시 줄(카라비너)이 로봇을 받는 상태에서 실행할 것.
   서 있는 로봇에 쏘면 (부드럽게) 주저앉는다.

v2.1 (2026-08-02, 적대리뷰 반영):
  - 인터리브 홀드: 모터별로 '읽기 성공 즉시' kp12 홀드 1발을 쏘고 다음 모터로
    넘어감 — 종전 일괄 방식은 버스 반죽음(뒷번호 재시도 소진) 시 앞번호가
    ~1초 무토크로 방치되어, 이 도구가 막으려던 자유낙하 회생 트립을 재현했음.
  - 시그널(SIGINT/SIGTERM/SIGHUP) → 즉시 최종 시퀀스로 점프. 최종 시퀀스
    (제로게인 3연발 + MIT 해제)는 finally에 있어 어떤 경로로 죽어도 실행됨 —
    종전엔 램프 중 Ctrl-C면 모터가 kp6 홀드를 문 채 MIT 래치로 방치됐음.
  - 원리 유지: 실측 현재각 홀드(점프 없음) → 2s 게인 램프다운(회생 미미)
    → 제로게인+EXIT. --fast = 즉시 릴리즈 (버스 사망 직전 비상 전용).

사용: python3 release_all.py [--channel can1] [--fast]
"""
import argparse
import math
import signal
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
IDS = list(range(1, 13))
ZERO = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)
KP_SOFT = 12.0     # 홀드 시작 게인 — 스윙만 잡을 저강성 (하중 지지 불가 수준)
KD_SOFT = 0.8
RAMP_S = 2.0
TICK = 0.02

SIG = []


def on_signal(signum, frame):
    SIG.append(signum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--fast", action="store_true",
                    help="즉시 제로게인 (구버전) — 버스 사망 직전 비상 전용. "
                         "관절이 영점에서 멀면 자유낙하 스윙→회생으로 SMPS 트립 위험")
    a = ap.parse_args()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    bus = can.Bus(channel=a.channel, interface="socketcan")
    fail = 0

    def tx(m, data):
        bus.send(can.Message(arbitration_id=m, data=data, is_extended_id=False))

    def hold(m, p, g):
        tx(m, pack_ak_mit_command(math.radians(p), 0.0,
                                  KP_SOFT * g, KD_SOFT * g, 0.0))

    try:
        if not a.fast:
            # 인터리브 읽기+홀드: 읽힌 모터는 그 자리에서 즉시 kp12 홀드 점유
            for m in IDS:
                try:
                    tx(m, ENTER)
                except Exception:
                    pass
            time.sleep(0.02)
            pos = {}
            for m in IDS:
                if SIG:
                    break
                got = None
                for _ in range(3):
                    try:
                        tx(m, ZERO)
                    except Exception:
                        break
                    t_end = time.monotonic() + 0.03
                    while time.monotonic() < t_end:
                        r = bus.recv(timeout=0.03)
                        if r is not None and not r.is_error_frame \
                                and len(r.data) == 8 and r.data[0] == m:
                            got = decode_ak_mit_feedback(
                                bytes(r.data)).position_deg
                            break
                    if got is not None:
                        break
                if got is not None:
                    pos[m] = got
                    try:
                        hold(m, got, 1.0)   # 즉시 홀드 — 무토크 공백 최소화
                    except Exception:
                        pass
                # 이미 읽힌 모터들 홀드 재송신 (읽기 지연 동안 명령 신선 유지)
                for pm, pp in pos.items():
                    try:
                        hold(pm, pp, 1.0)
                    except Exception:
                        pass
            missing = sorted(set(IDS) - set(pos))
            if missing:
                print(f"[release_all] 모터 {missing} 읽기 실패 — 해당 모터는 "
                      f"즉시 제로게인 (아마 무전원)")
            if pos and not SIG:
                print(f"[release_all] {len(pos)}모터 현재각 홀드 → "
                      f"{RAMP_S:.0f}s 게인 램프다운")
                t0 = time.monotonic()
                while not SIG:
                    g = 1.0 - (time.monotonic() - t0) / RAMP_S
                    if g <= 0.0:
                        break
                    for m, p in pos.items():
                        try:
                            hold(m, p, g)
                        except Exception:
                            fail += 1
                    time.sleep(TICK)
            if SIG:
                print(f"[release_all] 시그널 {SIG[0]} — 최종 시퀀스로 점프")
        return 0
    finally:
        # 최종 시퀀스는 어떤 경로(정상/시그널/예외)로도 반드시 실행
        for _ in range(3):
            for m in IDS:
                try:
                    tx(m, ZERO)
                except Exception:
                    fail += 1
            time.sleep(0.005)
        for m in IDS:
            try:
                tx(m, EXIT)
            except Exception:
                fail += 1
        if fail:
            print(f"[release_all] 완료 — 송신실패 {fail}건 (버스 상태 확인!)")
        else:
            print("[release_all] 12모터 소프트 릴리즈+MIT 해제 완료 — 무토크")
        try:
            bus.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
