"""정적 제자리걸음 (Plan B, 2026-08-03) — RL 독립 개루프 스크립트 보행.

배경: RL march의 심투리얼 갭(측정: 루프지연 4~8틱·마찰·컴플라이언스)이 닫히는
동안, 위치제어 기반 준정적(quasi-static) 제자리걸음으로 "일단 걷는" 베이스라인
확보 (0803 승윤님: "천천히 안정적으로" — 준정적 보행의 정의 그 자체).

⚠ 미검증 신규 도구 — 실행 순서 (벤치 검증 사다리):
  1. --probe: 공중 매달림에서 발목롤 ±1° 프로브 → 기울기 방향 육안 확인으로
     SHIFT_SIGN 확정 (해부학 부호 실측 전엔 체중이동 방향을 신뢰할 수 없음)
  2. 공중에서 전체 사이클 재생 (모양 확인, kp30)
  3. 접지 + 게인램프 후 shift만 (lift 0) — 체중이동 검증
  4. lift 소량부터 (2~3°)

동작 시퀀스 (hang_stand와 동일 UX):
  기동 → 현재각 캡처 → ENTER → kp20 에어램프(스탠드 영점) → touch /tmp/hang_ground
  + /tmp/hang_gain → kp150 램프 → touch /tmp/staticmarch_go → 사이클 시작:
    [T/4] 체중 왼쪽 (양 발목롤 shift_amp) → [T/4] 오른발 리프트·착지
    → [T/4] 체중 오른쪽 → [T/4] 왼발 리프트·착지  (코사인 블렌딩)
  조절: /tmp/staticmarch_shift (deg), /tmp/staticmarch_lift (deg),
        /tmp/staticmarch_period (s)
  종료: /tmp/staticmarch_stop (진폭 램프다운→게인 램프다운→홀드 유지, 릴리즈
        안 함 — 접지 하중 전제) / kill 파일은 비상용 즉시 릴리즈

안전:
  - 이탈 가드(진폭 연동)·피드백 두절 가드·모터 폴트 가드 → 접지 소프트 램프다운
  - 진폭 하드캡: shift ≤ 6°, lift ≤ 12°, period ≥ 2.5s
  - 명령 슬루 4°/틱 (stage8와 동일 규약)

사용: python3 static_march.py [--probe] [--channel can1]
"""
import argparse
import json
import math
import os
import signal
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
IDS = list(range(1, 13))

GO_FILE = "/tmp/staticmarch_go"
STOP_FILE = "/tmp/staticmarch_stop"
KILL_FILE = "/tmp/staticmarch_kill"
SHIFT_FILE = "/tmp/staticmarch_shift"
LIFT_FILE = "/tmp/staticmarch_lift"
PERIOD_FILE = "/tmp/staticmarch_period"
GROUND_FILE = "/tmp/hang_ground"
GAIN_FILE = "/tmp/hang_gain"

KP_AIR, KD_AIR = 20.0, 1.0
KP_OP, KD_OP = 150.0, 5.0
RAMP_RATE_DEG_S = 5.0
GAIN_RAMP_S = 8.0
SLEW_DEG_TICK = 4.0
TICK = 0.02

SHIFT_DEFAULT, SHIFT_MAX = 3.0, 6.0
LIFT_DEFAULT, LIFT_MAX = 0.0, 12.0
PERIOD_DEFAULT, PERIOD_MIN = 4.0, 2.5
AMP_RAMP_S = 6.0

#: 체중이동(발목롤 m6/m12) 모터 raw 부호 — ⚠ --probe로 실측 확정 전 잠정값.
#: 해부학: ankle_r은 좌우 '같은' 관절부호 예외 (joint-direction 실측 메모) —
#: 모터 raw 방향은 하드웨어맵 direction이 흡수. +shift = 잠정 '왼쪽 기울기'.
SHIFT_SIGN = {6: +1.0, 12: +1.0}
#: 리프트 조합 (hang_leg_swing 실측 부호 재사용): 힙F 굴곡 m1+/m7−,
#: 무릎 굽힘 m4+/m10−, 발목F 보상 m5+/m11− (트림 부호와 동일)
LIFT_R = {7: -1.0, 10: -1.0, 11: -0.8}   # 오른발 들기
LIFT_L = {1: +1.0, 4: +1.0, 5: +0.8}     # 왼발 들기

SIG = []


def on_signal(signum, frame):
    SIG.append(signum)


def say(*a, **k):
    k["flush"] = True
    try:
        print(*a, **k)
    except OSError:
        pass


def tx(bus, m, data):
    bus.send(can.Message(arbitration_id=m, data=data, is_extended_id=False))


def read_file_float(path, default, lo, hi):
    try:
        return max(lo, min(hi, float(open(path).read().strip())))
    except (OSError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="발목롤 부호 프로브: ±1° 교대 — 기울기 방향 육안 확인용")
    ap.add_argument("--channel", default="can1")
    a = ap.parse_args()

    for f in (GO_FILE, STOP_FILE, KILL_FILE, SHIFT_FILE, LIFT_FILE, PERIOD_FILE,
              GROUND_FILE, GAIN_FILE):
        try:
            os.remove(f)
        except OSError:
            pass
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    hw = json.load(open(HW_MAP))
    sz = {int(m["id"]): float(m["stand_zero_deg"]) for m in hw["motors"]}

    bus = can.Bus(channel=a.channel, interface="socketcan")
    say(f"[static_march] 기동 (probe={a.probe}) — ENTER + 현재각 캡처")
    for m in IDS:
        tx(bus, m, ENTER)
        time.sleep(0.005)
    time.sleep(0.05)

    pos = {}
    seen = {}
    for m in IDS:
        for _ in range(5):
            tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
            te = time.monotonic() + 0.03
            while time.monotonic() < te:
                r = bus.recv(timeout=0.03)
                if (r is not None and not r.is_error_frame and len(r.data) == 8
                        and r.data[0] == m and r.arbitration_id == m):
                    pos[m] = decode_ak_mit_feedback(bytes(r.data)).position_deg
                    seen[m] = time.monotonic()
                    break
            if m in pos:
                break
        if m not in pos:
            say(f"[FAIL] 모터 {m} 시작각 읽기 실패")
            return 1

    big = [m for m in IDS if abs(sz[m] - pos[m]) > 12.0]
    if big:
        say(f"[FAIL] 모터 {big} 스탠드에서 12° 초과 이탈 — 손 정렬 후 재실행")
        return 1

    kp, kd = KP_AIR, KD_AIR
    t0 = time.monotonic()
    t_ramp = max(6.0, max(abs(sz[m] - pos[m]) for m in IDS) / RAMP_RATE_DEG_S)
    p_start = dict(pos)
    cmd_prev = dict(pos)
    grounded = False
    gain_t0 = None
    marching = False
    march_t0 = None
    amp_scale = 0.0
    soft = None
    say(f"[static_march] 에어램프 {t_ramp:.1f}s → 이후: 접지=touch {GROUND_FILE}, "
        f"게인=touch {GAIN_FILE}, 시작=touch {GO_FILE}")

    def offsets(t_march, shift_amp, lift_amp, period):
        """사이클 위상 → 모터별 오프셋 deg. 4상 코사인 블렌딩."""
        ph = (t_march % period) / period            # 0~1
        out = {m: 0.0 for m in IDS}
        # 체중이동: 사인파 (0~0.5 왼쪽, 0.5~1 오른쪽)
        w = math.sin(2.0 * math.pi * ph)
        for m, s in SHIFT_SIGN.items():
            out[m] += s * shift_amp * w
        # 리프트: 체중이 반대쪽에 실린 구간 중앙에서 벨 곡선
        def bell(c, width=0.12):
            d = min(abs(ph - c), 1.0 - abs(ph - c))
            return math.cos(min(d / width, 1.0) * math.pi / 2.0) ** 2 \
                if d < width else 0.0
        br = bell(0.25)   # 왼쪽 지지 구간 → 오른발 리프트
        bl = bell(0.75)
        for m, s in LIFT_R.items():
            out[m] += s * lift_amp * br
        for m, s in LIFT_L.items():
            out[m] += s * lift_amp * bl
        return out

    while True:
        now = time.monotonic()
        t = now - t0
        frac = min(1.0, t / t_ramp)
        base = {m: p_start[m] + frac * (sz[m] - p_start[m]) for m in IDS}

        if os.path.exists(KILL_FILE) or len(SIG) >= 2:
            say("[비상] 즉시 릴리즈")
            for _ in range(3):
                for m in IDS:
                    tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                time.sleep(0.02)
            return 1
        if (os.path.exists(STOP_FILE) or SIG) and soft is None:
            soft = now
            say("[소프트 정지] 진폭 3s → 게인 2s 램프다운 → kp20 홀드 유지 "
                "(릴리즈 없음 — 줄 확인 후 release_all)")

        if not grounded and os.path.exists(GROUND_FILE):
            grounded = True
            say("[접지] 확인")
        if gain_t0 is None and frac >= 1.0 and os.path.exists(GAIN_FILE):
            gain_t0 = now
            say(f"[게인램프] kp20→{KP_OP:.0f} ({GAIN_RAMP_S:.0f}s)")
        if gain_t0 is not None and soft is None:
            g = min(1.0, (now - gain_t0) / GAIN_RAMP_S)
            kp = KP_AIR + g * (KP_OP - KP_AIR)
            kd = KD_AIR + g * (KD_OP - KD_AIR)
        if (not marching and kp >= KP_OP - 1e-6
                and (os.path.exists(GO_FILE) or a.probe)):
            marching = True
            march_t0 = now
            say("[시작] " + ("프로브 모드 (발목롤 ±1°)" if a.probe else "정적 마치"))

        shift_amp = read_file_float(SHIFT_FILE, SHIFT_DEFAULT, 0.0, SHIFT_MAX)
        lift_amp = read_file_float(LIFT_FILE, LIFT_DEFAULT, 0.0, LIFT_MAX)
        period = read_file_float(PERIOD_FILE, PERIOD_DEFAULT, PERIOD_MIN, 20.0)

        if soft is not None:
            amp_scale = max(0.0, amp_scale - TICK / 3.0)
            if amp_scale <= 0.0:
                kp = max(KP_AIR, kp - (KP_OP - KP_AIR) * TICK / 2.0)
                kd = max(KD_AIR, kd - (KD_OP - KD_AIR) * TICK / 2.0)
        elif marching:
            amp_scale = min(1.0, (now - march_t0) / AMP_RAMP_S)

        off = {m: 0.0 for m in IDS}
        if marching and march_t0 is not None:
            if a.probe:
                w = 1.0 if int((now - march_t0) / 3.0) % 2 == 0 else -1.0
                for m, s in SHIFT_SIGN.items():
                    off[m] = s * 1.0 * w * amp_scale
            else:
                raw = offsets(now - march_t0, shift_amp, lift_amp, period)
                off = {m: v * amp_scale for m, v in raw.items()}

        abort = None
        for m in IDS:
            cmd = base[m] + off[m]
            cmd = cmd_prev[m] + max(-SLEW_DEG_TICK,
                                    min(SLEW_DEG_TICK, cmd - cmd_prev[m]))
            cmd_prev[m] = cmd
            tx(bus, m, pack_ak_mit_command(
                math.radians(cmd), 0.0, kp, kd, 0.0))
        te = time.monotonic() + 0.012
        got = set()
        while time.monotonic() < te and len(got) < 12:
            r = bus.recv(timeout=0.004)
            if (r is not None and not r.is_error_frame and len(r.data) == 8
                    and 1 <= r.data[0] <= 12 and r.data[0] == r.arbitration_id):
                fb = decode_ak_mit_feedback(bytes(r.data))
                m = int(fb.motor_id)
                pos[m] = fb.position_deg
                seen[m] = time.monotonic()
                got.add(m)
                dev_lim = max(6.0, 1.0 * (abs(off[m]) + 4.0))
                if abs(pos[m] - cmd_prev[m]) > dev_lim:
                    abort = f"모터{m} 이탈 {pos[m] - cmd_prev[m]:+.1f}°"
        stale = [m for m in IDS if time.monotonic() - seen.get(m, 0) > 0.5]
        if stale:
            abort = f"모터{stale} 피드백 두절"
        if abort and soft is None:
            soft = now
            say(f"[가드] {abort} — 소프트 정지 진입 (줄 확인!)")
        if soft is not None and kp <= KP_AIR + 1e-6 and amp_scale <= 0.0:
            say("[완료] kp20 홀드 유지 중 — 줄 인장 후 release_all로 마무리")
            # kp20 홀드 지속 (하중 시 릴리즈 금지) — 운용자가 종료 판단
            while not os.path.exists(KILL_FILE) and len(SIG) < 2:
                for m in IDS:
                    tx(bus, m, pack_ak_mit_command(
                        math.radians(cmd_prev[m]), 0.0, KP_AIR, KD_AIR, 0.0))
                time.sleep(TICK)
            for _ in range(3):
                for m in IDS:
                    tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                time.sleep(0.02)
            return 0

        sl = TICK - (time.monotonic() - now)
        if sl > 0:
            time.sleep(sl)


if __name__ == "__main__":
    sys.exit(main())
