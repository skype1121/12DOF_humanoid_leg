"""전체 릴리즈 — 12모터 제로게인 3연발 + MIT 해제 (estop 프로토콜 최종 단계).

⚠ 반드시 줄(캘리브레이션 지그/카라비너)이 로봇을 받는 상태에서 실행할 것.
   서 있는 로봇에 쏘면 그대로 주저앉는다.

용도: 정책 라이브 estop 프로토콜의 마지막 단계 —
  ① 브리지에 estop 명령 (STOP_ALL = 현재 자세 동결 홀드, 게인 유지)
  ② 줄 재인장 (로봇을 줄이 받게)
  ③ 이 스크립트 실행 → 무토크 매달림
hang_stand.release_all()과 동일 패턴의 독립 도구 (스택이 죽어도 쓸 수 있게).

사용: python3 release_all.py [--channel can1]
"""
import argparse
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402

EXIT = bytes([0xFF] * 7 + [0xFD])
IDS = list(range(1, 13))
ZERO = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can1")
    a = ap.parse_args()
    bus = can.Bus(channel=a.channel, interface="socketcan")
    sent_fail = 0
    try:
        for _ in range(3):
            for m in IDS:
                try:
                    bus.send(can.Message(arbitration_id=m, data=ZERO,
                                         is_extended_id=False))
                except Exception:
                    sent_fail += 1
            time.sleep(0.005)
        for m in IDS:
            try:
                bus.send(can.Message(arbitration_id=m, data=EXIT,
                                     is_extended_id=False))
            except Exception:
                sent_fail += 1
        if sent_fail:
            print(f"[release_all] 완료 — 송신실패 {sent_fail}건 (버스 상태 확인!)")
            return 1
        print("[release_all] 12모터 제로게인+MIT 해제 완료 — 무토크")
        return 0
    finally:
        try:
            bus.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
