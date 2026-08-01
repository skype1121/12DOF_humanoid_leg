"""전체 릴리즈 — 현재각 홀드에서 게인 램프다운 후 MIT 해제 (estop 프로토콜 최종 단계).

⚠ 반드시 줄(카라비너)이 로봇을 받는 상태에서 실행할 것.
   서 있는 로봇에 쏘면 (부드럽게) 주저앉는다.

v2 (2026-08-02): 즉시 제로게인 → 소프트 램프다운으로 변경.
  사고 실증: 무릎 40° 굽힌 채 즉시 무토크를 쏘자 다리가 중력 자유낙하로
  스윙("팍") → 모터 역구동 회생 전류 → SMPS 과전압 보호 래치(사망처럼 보임).
  SMPS는 배터리와 달리 회생을 흡수 못 한다. 어떤 자세에서도 안전하려면:
  ① 실측 현재각을 그대로 홀드 목표로 (점프 없음, 에어램프와 같은 원리)
  ② 저게인(kp12)에서 2초 선형 램프다운 → 관절이 천천히 정착 (회생 미미)
  ③ 마지막에 제로게인 3연발 + MIT 해제 (기존과 동일)
  읽기 실패 모터는 구버전 경로(즉시 제로게인)로 폴백. --fast = 구버전 즉시
  릴리즈 (버스가 죽어가는 비상시 전용 — 회생 위험 감수).

사용: python3 release_all.py [--channel can1] [--fast]
"""
import argparse
import math
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


def tx(bus, mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


def read_positions(bus):
    """모터별 현재각 1회 읽기 (ENTER + 제로명령 → 응답). 실패 모터는 제외."""
    pos = {}
    for m in IDS:
        try:
            tx(bus, m, ENTER)
        except Exception:
            continue
    time.sleep(0.02)
    for m in IDS:
        got = None
        for _ in range(3):
            try:
                tx(bus, m, ZERO)
            except Exception:
                break
            t_end = time.monotonic() + 0.03
            while time.monotonic() < t_end:
                r = bus.recv(timeout=0.03)
                if r is not None and not r.is_error_frame \
                        and len(r.data) == 8 and r.data[0] == m:
                    got = decode_ak_mit_feedback(bytes(r.data)).position_deg
                    break
            if got is not None:
                break
        if got is not None:
            pos[m] = got
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--fast", action="store_true",
                    help="즉시 제로게인 (구버전) — 버스 사망 직전 비상 전용. "
                         "관절이 영점에서 멀면 자유낙하 스윙→회생으로 SMPS 트립 위험")
    a = ap.parse_args()
    bus = can.Bus(channel=a.channel, interface="socketcan")
    fail = 0
    try:
        if not a.fast:
            pos = read_positions(bus)
            missing = sorted(set(IDS) - set(pos))
            if missing:
                print(f"[release_all] 모터 {missing} 읽기 실패 — 해당 모터는 "
                      f"즉시 제로게인 폴백")
            if pos:
                print(f"[release_all] {len(pos)}모터 현재각 홀드 → "
                      f"{RAMP_S:.0f}s 게인 램프다운")
                t0 = time.monotonic()
                while True:
                    el = time.monotonic() - t0
                    g = 1.0 - el / RAMP_S
                    if g <= 0.0:
                        break
                    for m, p in pos.items():
                        try:
                            tx(bus, m, pack_ak_mit_command(
                                math.radians(p), 0.0,
                                KP_SOFT * g, KD_SOFT * g, 0.0))
                        except Exception:
                            fail += 1
                    time.sleep(TICK)
        for _ in range(3):
            for m in IDS:
                try:
                    tx(bus, m, ZERO)
                except Exception:
                    fail += 1
            time.sleep(0.005)
        for m in IDS:
            try:
                tx(bus, m, EXIT)
            except Exception:
                fail += 1
        if fail:
            print(f"[release_all] 완료 — 송신실패 {fail}건 (버스 상태 확인!)")
            return 1
        print("[release_all] 12모터 소프트 릴리즈+MIT 해제 완료 — 무토크")
        return 0
    finally:
        try:
            bus.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
