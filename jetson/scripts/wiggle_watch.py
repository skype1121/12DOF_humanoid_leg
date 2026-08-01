"""흔들기 테스트 — 12모터 제로토크 연속 폴링, 응답 빠지는 순간을 실시간 표시.

로봇 무토크 매달림 상태에서 실행. CAN 하니스를 구간별로 살살 흔들며
어느 구간에서 응답이 빠지는지 관찰. 모터는 절대 움직이지 않음
(MIT 진입 + 全제로 게인·토크 명령 = 제로토크 읽기, 승인된 패턴).

종료: touch /tmp/wiggle_stop  (종료 시 MIT 해제 보장)
"""
import os
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
IDS = list(range(1, 13))
STOP = "/tmp/wiggle_stop"
ZERO = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)

bus = can.Bus(channel="can1", interface="socketcan")


def tx(mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


try:
    for _ in range(2):
        for m in IDS:
            tx(m, ENTER)
        time.sleep(0.02)
    hits = {m: 0 for m in IDS}
    sweeps = 0
    t0 = time.monotonic()
    t_line = t0
    last_miss = None
    while not os.path.exists(STOP):
        got = set()
        send_fail = False
        for m in IDS:
            try:
                tx(m, ZERO)
            except Exception as e:
                send_fail = True
                print(f"[{time.monotonic()-t0:7.1f}s] !! 송신실패 {type(e).__name__}"
                      " — 버스 순단 (지금 만진 구간이 범인!)", flush=True)
                time.sleep(0.3)
                break
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            r = bus.recv(timeout=0.01)
            if r is None or r.is_error_frame or len(r.data) != 8:
                continue
            mid = r.data[0]
            if mid in hits:
                got.add(mid)
        sweeps += 1
        for m in got:
            hits[m] += 1
        miss = tuple(m for m in IDS if m not in got)
        if miss != last_miss and not send_fail:
            t = time.monotonic() - t0
            if miss:
                print(f"[{t:7.1f}s] 결측 발생: {list(miss)}", flush=True)
            else:
                print(f"[{t:7.1f}s] 전체 복구 12/12", flush=True)
            last_miss = miss
        if time.monotonic() - t_line >= 1.0:
            rates = " ".join(f"{m}:{100*hits[m]//max(sweeps,1)}" for m in IDS)
            print(f"[{time.monotonic()-t0:7.1f}s] 응답률% {rates} ({sweeps}스윕)",
                  flush=True)
            hits = {m: 0 for m in IDS}
            sweeps = 0
            t_line = time.monotonic()
finally:
    for _ in range(3):
        for m in IDS:
            try:
                tx(m, ZERO)
            except Exception:
                pass
        time.sleep(0.005)
    for m in IDS:
        try:
            tx(m, EXIT)
        except Exception:
            pass
    print("[wiggle] 종료 — MIT 해제", flush=True)
