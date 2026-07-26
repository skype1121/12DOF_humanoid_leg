"""에어 스탠딩 — 매달린 로봇을 스탠드 자세로 동기 램프 후 유지 (2026-07-27 첫 12모터 동시 제어).

전제: 로봇 매달림(발 비접촉), 부호 검증 완료, 스탠드 스냅샷(모터 프레임) 확보.
타깃은 '모터 프레임 절대각'이라 direction/스케일 변환이 필요 없다 (자기 엔코더 기준 복귀).

안전 설계:
  - 전 모터 현재각 실측 → 목표까지 50Hz 동기 선형 램프 (최대 5°/s, 최소 6초)
  - kp 20 / kd 1 (운용게인 미만) — 매달림 무부하라 충분
  - 가드: |실측−명령| > 15° (어느 모터든) / 모터별 피드백 0.5s 두절 / 총시간 초과
    → 즉시 전 모터 제로게인 3연발 + MIT 해제
  - 정지 스위치: /tmp/pose_hold_stop 파일 생기면 즉시 릴리즈 (touch로 원격 정지)
  - 어떤 경로로 끝나도 finally에서 릴리즈 보장. 릴리즈 후 로봇은 줄에 매달려 처짐.

사용: python3 pose_hold.py [--hold 30] [--kp 20] [--kd 1.0] [--channel can1]
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
STOP_FILE = "/tmp/pose_hold_stop"
IDS = list(range(1, 13))

#: 스탠드 스냅샷 v2 (stand_snapshot_v2_20260727 — 승윤님이 수평 스탠드로 정렬해
#: 매달아 둔 상태를 정지 캡처. v1('들고 정렬' 추정 구간)은 골반 기울어진 자세로
#: 판명 — 에어스탠딩 시 힙 ±30° 젖힘(기지개 자세) 관찰로 폐기)
STAND_DEG = {1: 31.55, 2: 3.60, 3: 8.27, 4: 12.36, 5: 33.56, 6: 4.12,
             7: 44.84, 8: 8.23, 9: -0.43, 10: 21.17, 11: 6.61, 12: 9.17}

#: 실측 ROM (rom_sweep2) — 목표 클램프 최후방어
MEAS_ROM = {
    1: (-114.4, 36.2), 2: (-13.5, 20.9), 3: (-82.3, 89.0), 4: (-64.3, 52.2),
    5: (-30.6, 41.1), 6: (-25.4, 34.0), 7: (14.4, 153.8), 8: (-22.7, 27.2),
    9: (-77.3, 61.6), 10: (-68.0, 64.0), 11: (-47.9, 42.9), 12: (-22.5, 36.6),
}

DEV_ABORT_DEG = 15.0
FB_TIMEOUT_S = 0.5
RAMP_RATE_DEG_S = 5.0
TICK = 0.02


def tx(bus, mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


def release_all(bus):
    for _ in range(3):
        for m in IDS:
            try:
                tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
        time.sleep(0.005)
    for m in IDS:
        try:
            tx(bus, m, EXIT)
        except Exception:
            pass


def drain(bus, pos, seen):
    """수신 큐 비우며 모터별 최신 위치·시각 갱신."""
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            return
        if msg.is_error_frame or len(msg.data) < 6:
            continue
        m = msg.data[0]
        if m in pos:
            try:
                pos[m] = decode_ak_mit_feedback(bytes(msg.data)).position_deg
                seen[m] = time.time()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=30.0)
    ap.add_argument("--kp", type=float, default=20.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--targets-json", default="",
                    help="모터별 타깃 override {\"1\": deg, ...} — 미지정 시 STAND_DEG")
    a = ap.parse_args()
    kp = min(a.kp, 25.0)
    kd = min(a.kd, 2.0)
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    bus = can.Bus(channel=a.channel, interface="socketcan")
    abort = None
    try:
        for m in IDS:
            tx(bus, m, ENTER)
            time.sleep(0.005)
        time.sleep(0.05)

        # 시작각 실측 (제로게인 3회 중앙값)
        p0 = {}
        for m in IDS:
            reads = []
            for _ in range(3):
                tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                t_e = time.time() + 0.03
                while time.time() < t_e:
                    r = bus.recv(timeout=0.03)
                    if r is not None and not r.is_error_frame \
                            and len(r.data) >= 6 and r.data[0] == m:
                        reads.append(decode_ak_mit_feedback(bytes(r.data)).position_deg)
                        break
            if len(reads) < 2:
                print(f"[FAIL] 모터 {m} 시작각 읽기 실패 — 중단", flush=True)
                return 1
            p0[m] = statistics.median(reads)

        base = STAND_DEG
        if a.targets_json:
            base = {int(k): float(v) for k, v in
                    json.load(open(a.targets_json)).items()}
        tgt = {m: min(max(base[m], MEAS_ROM[m][0] + 3), MEAS_ROM[m][1] - 3)
               for m in IDS}
        dmax = max(abs(tgt[m] - p0[m]) for m in IDS)
        t_ramp = max(6.0, dmax / RAMP_RATE_DEG_S)
        print("모터별 이동량: " + " ".join(
            f"{m}:{tgt[m] - p0[m]:+.1f}" for m in IDS), flush=True)
        print(f"[pose_hold] 램프 {t_ramp:.1f}s → 유지 {a.hold:.0f}s "
              f"(kp={kp} kd={kd})", flush=True)

        pos = dict(p0)
        seen = {m: time.time() for m in IDS}
        t0 = time.time()
        t_end = t0 + t_ramp + a.hold
        t_show = t0
        while time.time() < t_end:
            now = time.time()
            frac = min(1.0, (now - t0) / t_ramp)
            for m in IDS:
                cmd = p0[m] + frac * (tgt[m] - p0[m])
                try:
                    tx(bus, m, pack_ak_mit_command(
                        math.radians(cmd), 0.0, kp, kd, 0.0))
                except Exception as e:
                    abort = f"송신실패 {type(e).__name__}"
                    break
            if abort:
                break
            drain(bus, pos, seen)
            for m in IDS:
                cmd = p0[m] + frac * (tgt[m] - p0[m])
                if abs(pos[m] - cmd) > DEV_ABORT_DEG:
                    abort = f"모터{m} 이탈 {pos[m] - cmd:+.1f}°"
                    break
                if now - seen[m] > FB_TIMEOUT_S:
                    abort = f"모터{m} 피드백 두절"
                    break
            if abort or os.path.exists(STOP_FILE):
                abort = abort or "정지 스위치"
                break
            if now - t_show >= 1.0:
                t_show = now
                errs = {m: pos[m] - tgt[m] for m in IDS}
                worst = max(errs, key=lambda k: abs(errs[k]))
                phase = "램프" if frac < 1.0 else "유지"
                print(f"[{phase} {now - t0:5.1f}s] 최대오차 모터{worst} "
                      f"{errs[worst]:+.2f}° | 평균 {statistics.mean(abs(v) for v in errs.values()):.2f}°",
                      flush=True)
            time.sleep(max(0.0, TICK - (time.time() - now)))

        if abort:
            print(f"[ABORT] {abort} — 전체 릴리즈", flush=True)
            return 2
        print("[완료] 유지 종료 — 릴리즈 (로봇이 천천히 처집니다)", flush=True)
        return 0
    finally:
        release_all(bus)


if __name__ == "__main__":
    raise SystemExit(main())
