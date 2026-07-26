"""부호 검증 — 모터 1개씩 미소동작 (기본 +2°, 초저게인) — 2026-07-27 설계.

목적: 실물 모터 +회전이 URDF 관절 +방향과 같은지(direction=+1/−1) 관절별 확정.
로봇은 반드시 '매달린 상태(발 비접촉)'에서, 승윤님이 육안 관찰하며 1개씩 실행.

안전 설계 ("모터 팍팍 금지" 원칙):
  - kp 기본 3 (클램프 ≤12 — 운용게인 15~30 미만), kd 기본 0.5 (≤2), τ=0 고정
    → 2° 오차 시 토크 ~0.1Nm: 손으로 쉽게 저지되는 힘
  - 첫 명령 목표 = '현재각 그대로' → 점프 물리적으로 불가, 이후 0.1°/20ms 램프
  - 3중 자동중단: 이동량 > 3×delta+1° (폭주) / |속도| > 1.5rad/s (과속) /
    피드백 0.3s 두절 (통신상실) → 즉시 제로게인 3연발 + MIT 해제
  - 어떤 경로로 끝나든 finally에서 제로게인+해제 보장, 복귀 램프로 원위치

사용: python3 sign_check.py --id 4 [--delta 2] [--kp 3] [--kd 0.5]
결과 해석: '+명령 → 움직인 해부학 방향'을 관찰 보고 → 시뮬 표와 대조해 direction 확정.
"""
import argparse
import math
import statistics
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])

#: 실측 ROM (모터 프레임 deg, rom_sweep2_20260727) — 목표각 가드용
MEAS_ROM = {
    1: (-114.4, 36.2), 2: (-13.5, 20.9), 3: (-82.3, 89.0), 4: (-64.3, 52.2),
    5: (-30.6, 41.1), 6: (-25.4, 34.0), 7: (14.4, 153.8), 8: (-22.7, 27.2),
    9: (-77.3, 61.6), 10: (-68.0, 64.0), 11: (-47.9, 42.9), 12: (-22.5, 36.6),
}

#: URDF +방향의 해부학 의미 (Isaac 실측 2026-07-18, memory joint-direction-verified)
URDF_PLUS_MEANING = {
    1: "왼허벅지 앞으로 (굽힘)", 2: "왼다리 안쪽으로 (모음)",
    3: "왼발끝 바깥 회전 (외회전)", 4: "왼무릎 폄",
    5: "왼발끝 위로 (배굴)", 6: "왼발바닥 바깥 기울임 (외번)",
    7: "오른허벅지 뒤로 (폄)", 8: "오른다리 바깥으로 (벌림)",
    9: "오른발끝 안쪽 회전 (내회전)", 10: "오른무릎 굽힘",
    11: "오른발끝 아래로 (저굴)", 12: "오른발바닥 바깥 기울임 (외번)",
}


def tx(bus, mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


def exchange(bus, mid, pos_deg, kp, kd, timeout=0.03):
    """위치명령 1회 송신 → 피드백 수신 (없으면 None). τ=0, vel=0 고정."""
    tx(bus, mid, pack_ak_mit_command(math.radians(pos_deg), 0.0, kp, kd, 0.0))
    t_e = time.time() + timeout
    while time.time() < t_e:
        m = bus.recv(timeout=max(0.0, t_e - time.time()))
        if m is not None and not m.is_error_frame and len(m.data) >= 6 \
                and m.data[0] == mid:
            return decode_ak_mit_feedback(bytes(m.data))
    return None


def safe_release(bus, mid):
    """제로게인 3연발 + MIT 해제 — 실패해도 삼키고 끝까지 시도."""
    for _ in range(3):
        try:
            tx(bus, mid, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
            time.sleep(0.005)
        except Exception:
            pass
    try:
        tx(bus, mid, EXIT)
    except Exception:
        pass


def ramp(bus, mid, start, goal, kp, kd, p0, delta, state):
    """0.1°/20ms 램프 — 3중 가드. 중단 사유 반환 (None=정상)."""
    cur = start
    step = 0.1 if goal >= start else -0.1
    last_fb = time.time()
    while True:
        cur = min(cur + step, goal) if step > 0 else max(cur + step, goal)
        try:
            fb = exchange(bus, mid, cur, kp, kd)
        except Exception as e:  # 송신 실패 = 버스 이상
            return f"송신실패 {type(e).__name__}"
        now = time.time()
        if fb is not None:
            last_fb = now
            state["pos"] = fb.position_deg
            if abs(fb.position_deg - p0) > abs(delta) + 6.0:
                return "폭주감지"
            if abs(fb.velocity_rad_s) > 1.5:
                return "과속감지"
        elif now - last_fb > 0.3:
            return "통신상실"
        if cur == goal:
            return None
        time.sleep(0.018)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--hold", type=float, default=0.6)
    ap.add_argument("--channel", default="can1")
    a = ap.parse_args()
    mid = a.id
    delta = max(-10.0, min(10.0, a.delta))   # 관찰 가능 한계 (±10°)
    kp = min(a.kp, 12.0)                     # 운용게인(15~30) 미만 강제
    kd = min(a.kd, 2.0)

    bus = can.Bus(channel=a.channel, interface="socketcan")
    print(f"[sign_check] 모터 {mid}: {delta:+.1f}° @ kp={kp} kd={kd}")
    print(f"  URDF +방향 의미: {URDF_PLUS_MEANING.get(mid, '?')}")
    state = {"pos": None}
    try:
        tx(bus, mid, ENTER)
        time.sleep(0.05)
        reads = []
        for _ in range(5):
            fb = exchange(bus, mid, 0.0, 0.0, 0.0)   # 제로게인 읽기
            if fb is not None:
                reads.append(fb.position_deg)
            time.sleep(0.02)
        if len(reads) < 3:
            print("[FAIL] 시작각 읽기 실패 — 통신 확인")
            return 1
        p0 = statistics.median(reads)
        print(f"  시작각 p0 = {p0:+.2f}°")
        lo, hi = MEAS_ROM[mid]
        if not (lo + 3.0 <= p0 + delta <= hi - 3.0):
            print(f"[거부] 목표 {p0 + delta:+.1f}°가 실측 ROM({lo:+.1f}~{hi:+.1f}) "
                  f"3° 안전띠 침범 — --delta {-delta:+.0f} 로 반대방향 시도")
            return 3

        abort = ramp(bus, mid, p0, p0 + delta, kp, kd, p0, delta, state)
        if abort:
            print(f"[ABORT] 전진 램프 중단: {abort} (현재 {state['pos']}°)")
            return 2
        t_e = time.time() + a.hold
        last = state["pos"]
        while time.time() < t_e:
            fb = exchange(bus, mid, p0 + delta, kp, kd)
            if fb is not None:
                last = fb.position_deg
            time.sleep(0.02)
        dp = (last - p0) if last is not None else 0.0

        abort = ramp(bus, mid, p0 + delta, p0, kp, kd, p0, delta, state)
        if abort:
            print(f"[경고] 복귀 램프 중단: {abort} — 제로게인 해제로 마무리")
        print(f"  도달각 = {last:+.2f}°  변위 dp = {dp:+.2f}°")
        if abs(dp) < 0.3:
            print(f"[결과] 무동작 (마찰/중력에 못 이김) → --kp {min(kp*2, 12):.0f} 로 재시도")
        else:
            print(f"[결과] 엔코더 기준 {'+' if dp > 0 else '−'}방향 {abs(dp):.1f}° 이동")
            print(f"  → 승윤님 관찰: 실제로 '{URDF_PLUS_MEANING.get(mid)}' 이었으면 direction=+1,")
            print("     반대 방향이었으면 direction=-1 로 기록")
        return 0
    finally:
        safe_release(bus, mid)


if __name__ == "__main__":
    raise SystemExit(main())
