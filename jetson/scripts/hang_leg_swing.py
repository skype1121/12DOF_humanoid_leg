"""행잉 레그 스윙 — 매달린 로봇의 스크립트 보행 스윙 데모 (영상 촬영용, 2026-08-02).

⚠ 반드시 공중 매달림(발 무접촉)에서만 실행. 정책이 아니라 순수 궤적 재생이라
  발진 위험이 없음 (피드백 루프 없는 개루프 사인파 + 저게인 kp30).
  0802 "에어 라이브 금지"는 정책(폐루프) 얘기 — 이 도구는 개루프라 해당 없음.

동작:
  1. 에어램프: 현재각 → 스탠드 영점 (kp20, 최대 5°/s) — hang_stand와 동일
  2. 스윙: 힙 F 좌우 반대위상 사인파 + 무릎 굽힘 (걷기 모양), 진폭 0→목표 5s 램프
     raw 부호 (웅크림 실측): 힙굴곡 m1+/m7−, 무릎굽힘 m4+/m10− → 좌우 반대위상은
     m1·m7 같은 부호, m4·m10 반대 부호로 구현됨 (아래 식 참조)
  3. 조절 (라이브): /tmp/legswing_amp (힙 진폭 deg, 기본 10, 0~18 클램프)
                    /tmp/legswing_freq (Hz, 기본 0.5, 0.2~0.8 클램프)
  4. 종료: /tmp/legswing_stop → 진폭 3s 램프다운 → 게인 2s 램프다운 → MIT 해제
          /tmp/legswing_kill → 즉시 릴리즈 (비상)

안전:
  - 가드: |실측−명령| > 20° / 피드백 0.5s 두절 / 모터 폴트 2회 → 즉시 릴리즈
  - 무릎 진폭 = 힙 진폭 × 1.2 (스윙 모양), ROM 여유 대비 최대 18×1.2=21.6° < 35°
  - 종료는 항상 소프트 (진폭·게인 램프다운) — SMPS 회생 트립 방지 (release_all v2 교훈)

사용: python3 hang_leg_swing.py [--hold 120] [--channel can1]
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
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
IDS = list(range(1, 13))
STOP_FILE = "/tmp/legswing_stop"
KILL_FILE = "/tmp/legswing_kill"
AMP_FILE = "/tmp/legswing_amp"
FREQ_FILE = "/tmp/legswing_freq"

KP_AIR, KD_AIR = 20.0, 1.0
KP_SWING, KD_SWING = 30.0, 1.5
RAMP_RATE_DEG_S = 5.0
AMP_RAMP_S = 5.0
AMP_DEFAULT, AMP_MIN, AMP_MAX = 10.0, 0.0, 18.0
FREQ_DEFAULT, FREQ_MIN, FREQ_MAX = 0.5, 0.2, 0.8
KNEE_RATIO = 1.2
DEV_ABORT_DEG = 20.0
FB_TIMEOUT_S = 0.5
FAULT_ABORT_N = 2
TICK = 0.02
SOFT_AMP_S = 3.0
SOFT_GAIN_S = 2.0

SWING_SIGN = {1: +1.0, 7: +1.0}       # 힙F: 반대위상 굴곡 = raw 둘 다 +sin
KNEE_SIGN = {4: +1.0, 10: -1.0}       # 무릎: raw 부호 반대 (굽힘 m4+/m10−)


def say(*a, **k):
    k["flush"] = True
    try:
        print(*a, **k)
    except OSError:
        pass


def read_param(path, cur, lo, hi):
    try:
        with open(path) as f:
            return min(hi, max(lo, float(f.read().strip())))
    except Exception:
        return cur


def swing_offsets(phase, amp_hip):
    """위상(rad)→모터별 raw 오프셋 deg. 힙 반대위상 사인 + 무릎 스텝 굽힘."""
    off = {m: 0.0 for m in IDS}
    s = math.sin(phase)
    for m, g in SWING_SIGN.items():
        off[m] = g * amp_hip * s
    amp_knee = amp_hip * KNEE_RATIO
    # 무릎: 힙이 앞으로 나갈 때 굽힘 (반주기 반파 사인, 좌우 반대위상)
    kl = max(0.0, math.sin(phase)) * amp_knee
    kr = max(0.0, math.sin(phase + math.pi)) * amp_knee
    off[4] = KNEE_SIGN[4] * kl
    off[10] = KNEE_SIGN[10] * kr
    return off


def selftest():
    """궤적 순수 검증 — 부호·연속성·진폭 한계."""
    # 진폭 한계: 모든 위상에서 |오프셋| ≤ amp×ratio
    for k in range(200):
        ph = k * 0.05
        off = swing_offsets(ph, AMP_MAX)
        assert all(abs(v) <= AMP_MAX * KNEE_RATIO + 1e-9 for v in off.values())
    # 반대위상: 힙 raw 동부호(해부학 반대), 무릎은 같은 위상에 한쪽만 굽힘
    off = swing_offsets(math.pi / 2, 10.0)
    assert off[1] == off[7] == 10.0
    assert off[4] == 12.0 and off[10] == 0.0, (off[4], off[10])
    off2 = swing_offsets(3 * math.pi / 2, 10.0)
    assert off2[1] == off2[7] == -10.0
    assert off2[4] == 0.0 and off2[10] == -12.0
    # 연속성: 틱 간 점프 ≤ 진폭×w×dt (0.8Hz, 18°에서도 < 2°/틱)
    prev = swing_offsets(0.0, AMP_MAX)
    wmax = 2 * math.pi * FREQ_MAX
    for k in range(1, 400):
        cur = swing_offsets(k * wmax * TICK, AMP_MAX)
        step = max(abs(cur[m] - prev[m]) for m in IDS)
        assert step <= AMP_MAX * KNEE_RATIO * wmax * TICK + 1e-6, step
        prev = cur
    print("[selftest] 궤적 부호·진폭한계·연속성 PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=120.0, help="스윙 유지 시간(초)")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    for f in (STOP_FILE, KILL_FILE, AMP_FILE, FREQ_FILE):
        if os.path.exists(f):
            os.remove(f)

    try:
        hw = json.load(open(HW_MAP))
        zero = {int(j["motor_id"]): float(j["stand_zero_deg"])
                for j in hw["joints"].values()}
    except Exception as e:
        raise SystemExit(f"[중단] 하드웨어맵 읽기 실패: {e}")

    bus = can.Bus(channel=a.channel, interface="socketcan")

    def tx(m, data):
        bus.send(can.Message(arbitration_id=m, data=data, is_extended_id=False))

    def release(soft_from=None):
        """소프트 릴리즈: 게인 램프다운 후 제로게인+EXIT (회생 트립 방지)."""
        if soft_from is not None:
            t0 = time.monotonic()
            while time.monotonic() - t0 < SOFT_GAIN_S:
                g = 1.0 - (time.monotonic() - t0) / SOFT_GAIN_S
                for m, p in soft_from.items():
                    try:
                        tx(m, pack_ak_mit_command(
                            math.radians(p), 0.0,
                            max(0.0, KP_SWING * g), max(0.0, KD_SWING * g), 0.0))
                    except Exception:
                        pass
                time.sleep(TICK)
        for _ in range(3):
            for m in IDS:
                try:
                    tx(m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                except Exception:
                    pass
            time.sleep(0.005)
        for m in IDS:
            try:
                tx(m, EXIT)
            except Exception:
                pass

    abort = None
    cmd_now = {}
    try:
        for _ in range(2):
            for m in IDS:
                tx(m, ENTER)
            time.sleep(0.02)

        # 시작각 읽기
        pos = {}
        for m in IDS:
            for _ in range(5):
                tx(m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                t_e = time.monotonic() + 0.03
                while time.monotonic() < t_e:
                    r = bus.recv(timeout=0.03)
                    if r is not None and not r.is_error_frame \
                            and len(r.data) == 8 and r.data[0] == m:
                        pos[m] = decode_ak_mit_feedback(
                            bytes(r.data)).position_deg
                        break
                if m in pos:
                    break
            if m not in pos:
                say(f"[FAIL] 모터 {m} 시작각 읽기 실패 — 중단")
                return 1

        big = [m for m in IDS if abs(zero[m] - pos[m]) > 25.0]
        if big:
            say(f"[FAIL] 모터 {big} 시작각이 영점에서 25° 초과 — 매달림 정렬 후 재실행")
            return 1

        p0 = dict(pos)
        dmax = max(abs(zero[m] - p0[m]) for m in IDS)
        t_ramp = max(4.0, dmax / RAMP_RATE_DEG_S)
        say(f"[legswing] 에어램프 {t_ramp:.1f}s → 스윙 (kp{KP_AIR:.0f}→{KP_SWING:.0f})")

        seen = {m: time.monotonic() for m in IDS}
        fault_cnt = {}
        fault_t = {}
        amp_req = AMP_DEFAULT
        freq = FREQ_DEFAULT
        amp_cur = 0.0
        phase = 0.0
        t0 = time.monotonic()
        t_show = t0
        t_prev = t0
        swinging = False
        soft = False

        while True:
            now = time.monotonic()
            dt = min(0.1, now - t_prev)
            t_prev = now
            frac = min(1.0, (now - t0) / t_ramp)
            if frac >= 1.0 and not swinging:
                swinging = True
                t_swing0 = now
                say("[legswing] 스윙 시작 — 진폭 5s 램프업 "
                    f"(amp={amp_req:.0f}° freq={freq:.2f}Hz; "
                    f"echo N > {AMP_FILE} 로 조절)")
            if swinging:
                amp_req = read_param(AMP_FILE, amp_req, AMP_MIN, AMP_MAX)
                freq = read_param(FREQ_FILE, freq, FREQ_MIN, FREQ_MAX)
                ramp_target = 0.0 if soft else amp_req
                rate = (AMP_MAX / SOFT_AMP_S) if soft else (AMP_MAX / AMP_RAMP_S)
                if amp_cur < ramp_target:
                    amp_cur = min(amp_cur + rate * dt, ramp_target)
                else:
                    amp_cur = max(amp_cur - rate * dt, ramp_target)
                phase += 2 * math.pi * freq * dt
                if now - t0 > a.hold + t_ramp and not soft:
                    soft = True
                    say("[legswing] 유지시간 종료 — 진폭 램프다운")
                if soft and amp_cur <= 0.0:
                    say("[legswing] 스윙 종료 — 소프트 릴리즈")
                    break

            kp = KP_AIR + (KP_SWING - KP_AIR) * frac
            kd = KD_AIR + (KD_SWING - KD_AIR) * frac
            off = swing_offsets(phase, amp_cur) if swinging \
                else {m: 0.0 for m in IDS}
            for m in IDS:
                c = p0[m] + frac * (zero[m] - p0[m]) + off[m]
                cmd_now[m] = c
                try:
                    tx(m, pack_ak_mit_command(
                        math.radians(c), 0.0, kp, kd, 0.0))
                except Exception as e:
                    abort = f"송신실패 {type(e).__name__}"
                    break
            if abort:
                break

            while True:
                r = bus.recv(timeout=0.0)
                if r is None:
                    break
                if r.is_error_frame or len(r.data) != 8:
                    continue
                mid = r.data[0]
                if mid not in pos:
                    continue
                try:
                    fb = decode_ak_mit_feedback(bytes(r.data))
                    pos[mid] = fb.position_deg
                    seen[mid] = time.monotonic()
                    if not fb.error_ok:
                        t = time.monotonic()
                        fault_cnt[mid] = 1 if t - fault_t.get(mid, -9.9) > 1.0 \
                            else fault_cnt.get(mid, 0) + 1
                        fault_t[mid] = t
                except Exception:
                    pass

            for m in IDS:
                if abs(pos[m] - cmd_now[m]) > DEV_ABORT_DEG:
                    abort = f"모터{m} 이탈 {pos[m] - cmd_now[m]:+.1f}°"
                    break
                if now - seen[m] > FB_TIMEOUT_S:
                    abort = f"모터{m} 피드백 두절"
                    break
            if any(c >= FAULT_ABORT_N for c in fault_cnt.values()):
                abort = "모터 폴트 2회"
            if abort or os.path.exists(KILL_FILE):
                abort = abort or "비상 정지(kill)"
                break
            if os.path.exists(STOP_FILE) and not soft:
                try:
                    os.remove(STOP_FILE)
                except OSError:
                    pass
                if not swinging:
                    say("[legswing] 정지 요청 (에어램프 중) — 소프트 릴리즈")
                    break
                soft = True
                say("[legswing] 정지 요청 — 진폭 램프다운")

            if now - t_show >= 1.0:
                t_show = now
                errs = [abs(pos[m] - cmd_now[m]) for m in IDS]
                say(f"[{'스윙' if swinging else '에어램프'} {now - t0:5.1f}s] "
                    f"amp{amp_cur:4.1f}° f{freq:.2f}Hz kp{kp:.0f} "
                    f"| 최대오차 {max(errs):.2f}° 평균 {statistics.mean(errs):.2f}°")
            time.sleep(max(0.0, TICK - (time.monotonic() - now)))

        if abort:
            say(f"[ABORT] {abort} — 즉시 릴리즈 (매달림이라 무해)")
            release()
            return 2
        release(soft_from=cmd_now)
        say("[legswing] 완료 — 무토크")
        return 0
    finally:
        try:
            release()
        except Exception:
            pass
        try:
            bus.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
