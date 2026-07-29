"""매달림 기립 — 지그에 매달린 로봇을 스탠드로 홀드 후, 줄을 풀며 체중 이전 (2026-07-30).

절차 (승윤님 수동 하강 + 원격 파일 트리거):
  1. AIR_RAMP   : 현재각→스탠드(하드웨어맵 v5) 저게인 동기 램프 (kp20/kd1, 최대 5°/s)
  2. AIR_HOLD   : 매달림 유지. 트림 프로브 — /tmp/ankle_trim_deg 에 8 쓰면 발목F
                  (m5 +/m11 −)를 0.5°/s로 램프. 무부하에서 하드스톱에 닿으면 추종오차
                  증분이 벌어짐 → 3틱 연속 초과 시 트림 동결+경고 (저토크라 무해).
                  프로브 후 0으로 되돌릴 것 (0 복귀 시 베이스라인 리셋 — 재프로브 가능).
  3. GAIN_RAMP  : /tmp/hang_gain 생성 시 kp20/kd1 → kp150/kd5 선형 램프 (8s).
                  전 모터 오차 3° 미만일 때만 수락 (아니면 파일 삭제+거부 사유 출력).
                  ※ 반드시 발이 땅에 닿기 전에. 완료 후 줄을 천천히 풀어 착지.
  4. 착지 선언  : 발이 닿으면 touch /tmp/hang_ground — 무부하 스톨가드 해제
                  (접지 후엔 발목 오차가 정상이므로). 이후 트림 8→10° 라이브 조정.
                  오탐 동결은 touch /tmp/hang_trim_unfreeze 로 해제.
  5. 종료       : /tmp/hang_stand_stop → 소프트 릴리즈: 명령 동결 + 게인만 3s 램프다운
                  (줄 재인장 후 사용!). /tmp/hang_stand_kill → 즉시 릴리즈 (비상).
                  --hold 초과 시: 저게인이면 소프트 릴리즈, 운용게인(하중 가능성)이면
                  자동 릴리즈 없이 경고 반복 — 파일 트리거로만 종료.

안전:
  - 가드: |실측−명령| > 15°(어느 모터든) / 피드백 0.5s 두절 / 모터 폴트 2회 관측
    → 즉시 릴리즈 (줄이 최후 안전망). 폴트 1회는 경고만 (일시 스파이크 디바운스).
  - 시그널: 1회(SIGINT/SIGTERM/SIGHUP) → 소프트 정지 경로, 2회 → 즉시 릴리즈.
    SSH 단절(pty 사망) 시에도 print OSError를 삼켜 소프트 경로가 완주됨.
  - 릴리즈 = 제로게인 3연발 + MIT 해제. 어떤 경로로 끝나도 finally 보장.
  - ROM 클램프는 '스탠드 영점 상대값'(프레임 불변). 발목 트림은 클램프 대신
    무부하 스톨가드로 보호 (m5 상한 41.1°는 스윕 미완 의심 — 실측으로 확인).
  - 게인 상한 kp150/kd5 (하드웨어맵 승인값, kd5=실물 상한 규칙)
  - 제어 시계는 monotonic (NTP 스텝 보정에 면역)

사용: python3 hang_stand.py [--hold 900] [--channel can1]
"""
import argparse
import json
import math
import os
import signal
import statistics
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
STOP_FILE = "/tmp/hang_stand_stop"
KILL_FILE = "/tmp/hang_stand_kill"
GAIN_FILE = "/tmp/hang_gain"
GROUND_FILE = "/tmp/hang_ground"
TRIM_FILE = "/tmp/ankle_trim_deg"
UNFREEZE_FILE = "/tmp/hang_trim_unfreeze"
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
IDS = list(range(1, 13))

#: 실측 ROM의 '스탠드 영점 상대값' (rom_sweep2 − stand_v4, 프레임 불변).
ROM_REL = {
    1: (-146.4, 4.2), 2: (-15.7, 18.7), 3: (-91.0, 80.3), 4: (-78.5, 38.0),
    5: (-64.9, 6.8), 6: (-30.4, 29.1), 7: (-30.3, 109.1), 8: (-31.5, 18.4),
    9: (-75.8, 63.1), 10: (-89.9, 42.1), 11: (-53.5, 37.3), 12: (-30.8, 28.3),
}

KP_AIR, KD_AIR = 20.0, 1.0
KP_MAX, KD_MAX = 150.0, 5.0
DEV_ABORT_DEG = 15.0
FB_TIMEOUT_S = 0.5
RAMP_RATE_DEG_S = 5.0
GAIN_RAMP_S = 8.0
GAIN_GATE_DEG = 3.0
SOFT_RELEASE_S = 3.0
TRIM_RATE_DEG_S = 0.5
TRIM_MIN, TRIM_MAX = -2.0, 12.0
TRIM_STALL_AIR_DEG = 2.5
TRIM_STALL_TICKS = 3
FAULT_ABORT_N = 2
TICK = 0.02
L_ANK, R_ANK = 5, 11

SIG_STOP = []


def on_signal(signum, frame):
    SIG_STOP.append(signum)


def say(*args, **kw):
    """print 래퍼 — SSH 단절로 pty가 죽어도 (OSError) 제어 루프를 못 죽이게."""
    kw["flush"] = True
    try:
        print(*args, **kw)
    except OSError:
        pass


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


def drain(bus, pos, seen, fault, fault_cnt, fault_t):
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            return
        m = msg.data[0] if len(msg.data) == 8 else None
        if msg.is_error_frame or m not in pos:
            continue
        try:
            fb = decode_ak_mit_feedback(bytes(msg.data))
            pos[m] = fb.position_deg
            seen[m] = time.monotonic()
            if not fb.error_ok:
                t = time.monotonic()
                # 1s 시간창 디바운스 — 무관한 단발 스파이크 2건의 누적 방지
                fault_cnt[m] = 1 if t - fault_t.get(m, -9.9) > 1.0 \
                    else fault_cnt.get(m, 0) + 1
                fault_t[m] = t
                fault[m] = fb.error_raw
        except Exception:
            pass


def load_stand_zero():
    try:
        hw = json.load(open(HW_MAP))
        zero = {int(j["motor_id"]): float(j["stand_zero_deg"])
                for j in hw["joints"].values()}
    except Exception as e:
        raise SystemExit(f"[중단] 하드웨어맵 읽기 실패 {HW_MAP}: {e} — "
                         f"boot_pose_check로 프레임 확정된 맵이 필요합니다")
    missing = [m for m in IDS if m not in zero]
    if missing:
        raise SystemExit(f"[중단] 하드웨어맵에 모터 {missing} stand_zero 없음")
    return zero


def read_trim_req(cur_req):
    try:
        with open(TRIM_FILE) as f:
            v = float(f.read().strip())
        return min(TRIM_MAX, max(TRIM_MIN, v))
    except Exception:
        return cur_req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=900.0,
                    help="유지 상한(초). 운용게인 상태면 초과해도 자동 릴리즈 안 함")
    ap.add_argument("--channel", default="can1")
    a = ap.parse_args()

    for f in (STOP_FILE, KILL_FILE, GAIN_FILE, GROUND_FILE, TRIM_FILE,
              UNFREEZE_FILE):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                raise SystemExit(f"[중단] 트리거 파일 삭제 불가 {f}: {e} — "
                                 f"수동 삭제 후 재실행 (타 계정 소유?)")
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    zero = load_stand_zero()
    tgt = {m: min(max(zero[m], zero[m] + ROM_REL[m][0] + 3),
                  zero[m] + ROM_REL[m][1] - 3) for m in IDS}
    for m in IDS:
        if abs(tgt[m] - zero[m]) > 0.01:
            say(f"[주의] 모터{m} 스탠드 영점이 ROM 마진 밖 — "
                f"{zero[m]:+.2f}→{tgt[m]:+.2f} 클램프")

    try:
        bus = can.Bus(channel=a.channel, interface="socketcan")
    except Exception as e:
        raise SystemExit(f"[중단] CAN 열기 실패({a.channel}): {e} — can_up.sh 후 재실행")

    abort = None
    soft_stop = False
    try:
        for m in IDS:
            tx(bus, m, ENTER)
            time.sleep(0.005)
        time.sleep(0.05)

        p0 = {}
        for m in IDS:
            reads = []
            for _ in range(5):
                tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                t_e = time.monotonic() + 0.03
                while time.monotonic() < t_e:
                    r = bus.recv(timeout=0.03)
                    if r is not None and not r.is_error_frame \
                            and len(r.data) == 8 and r.data[0] == m:
                        reads.append(
                            decode_ak_mit_feedback(bytes(r.data)).position_deg)
                        break
                if len(reads) >= 3:
                    break
            if len(reads) < 3:
                say(f"[FAIL] 모터 {m} 시작각 읽기 실패 — 중단")
                return 1
            p0[m] = statistics.median(reads)
            if SIG_STOP:
                say("[중단] 시그널 수신 — 시작 전 종료")
                return 1

        big = [m for m in IDS if abs(tgt[m] - p0[m]) > 12.0]
        if big:
            say(f"[FAIL] 모터 {big} 시작각이 스탠드에서 12° 초과 이탈 — "
                f"매달림 직립 정렬 후 재실행 (boot_pose_check 먼저)")
            return 1

        dmax = max(abs(tgt[m] - p0[m]) for m in IDS)
        t_ramp = max(6.0, dmax / RAMP_RATE_DEG_S)
        say("모터별 이동량: " + " ".join(
            f"{m}:{tgt[m] - p0[m]:+.1f}" for m in IDS))
        say(f"[hang_stand] 에어램프 {t_ramp:.1f}s → 유지 (kp{KP_AIR:.0f} 시작)")

        pos = dict(p0)
        seen = {m: time.monotonic() for m in IDS}
        fault = {}
        fault_cnt = {}
        fault_t = {}
        fault_warned = set()
        kp, kd = KP_AIR, KD_AIR
        gain_t0 = None
        gain_skip_mtime = None      # 삭제 불가한 거부된 gain 파일 mtime (무시 목록)
        unfreeze_skip_mtime = None  # 삭제 불가한 unfreeze 파일 mtime (1회만 적용)
        trim_req = 0.0
        trim_cur = 0.0
        trim_frozen = None
        trim_base = None
        stall_cnt = {L_ANK: 0, R_ANK: 0}
        grounded = False
        t0 = time.monotonic()
        t_show = t0
        t_prev = t0
        t_warn = 0.0
        t_warn_ground = 0.0

        def cmd_of(m, frac):
            c = p0[m] + frac * (tgt[m] - p0[m])
            if m == L_ANK:
                c += trim_cur
            elif m == R_ANK:
                c -= trim_cur
            return c

        while True:
            now = time.monotonic()
            dt = min(0.1, now - t_prev)
            t_prev = now
            frac = min(1.0, (now - t0) / t_ramp)
            operational = kp > KP_AIR + 1e-6

            if now - t0 > a.hold:
                if operational:
                    if now - t_warn > 30.0:
                        t_warn = now
                        say("[경고] 유지 상한 초과 — 하중 가능성으로 자동 릴리즈 "
                            "안 함. stop/kill 파일로 종료하세요")
                else:
                    soft_stop = True
                    say("[시간초과] 저게인 상태 — 소프트 릴리즈")
                    break

            if not grounded and os.path.exists(GROUND_FILE):
                grounded = True
                say("[착지 선언] 스톨가드 해제 — 트림 라이브 조정 가능")
            if operational and not grounded and trim_req > 0.0 \
                    and now - t_warn_ground > 30.0:
                t_warn_ground = now
                say("[안내] 착지했으면 touch /tmp/hang_ground — 선언 전엔 "
                    "스톨가드가 부하 오차를 하드스톱으로 오인할 수 있음")

            if trim_frozen is not None and os.path.exists(UNFREEZE_FILE):
                try:
                    mt = os.path.getmtime(UNFREEZE_FILE)
                except OSError:
                    mt = None
                if unfreeze_skip_mtime is None or mt != unfreeze_skip_mtime:
                    try:
                        os.remove(UNFREEZE_FILE)
                    except OSError:
                        unfreeze_skip_mtime = mt
                    trim_frozen = None
                    # trim_cur>0 상태라 0크로싱 재캡처가 안 옴 — 즉시 재캡처해
                    # 스톨가드를 재장전한 채로 트림 재상승 허용
                    trim_base = {m: pos[m] - cmd_of(m, frac)
                                 for m in (L_ANK, R_ANK)}
                    stall_cnt = {L_ANK: 0, R_ANK: 0}
                    say("[동결 해제] 베이스라인 재캡처 — 트림 재조정 가능")

            if gain_t0 is None and frac >= 1.0 and os.path.exists(GAIN_FILE):
                try:
                    mt = os.path.getmtime(GAIN_FILE)
                except OSError:
                    mt = None
                if gain_skip_mtime is None or mt != gain_skip_mtime:
                    errs_now = {m: abs(pos[m] - cmd_of(m, frac)) for m in IDS}
                    bad = [m for m in IDS if errs_now[m] >= GAIN_GATE_DEG]
                    if bad:
                        try:
                            os.remove(GAIN_FILE)
                        except OSError:
                            gain_skip_mtime = mt
                        say(f"[거부] 게인램프 — 모터 {bad} 오차 {GAIN_GATE_DEG}° "
                            f"이상 (걸림/스톨 의심). 해소 후 다시 touch")
                    else:
                        gain_t0 = now
                        say("[게인램프] kp20/kd1 → kp150/kd5 (8s) — 완료 후 "
                            "줄을 천천히 풀어 착지시키세요")
            if gain_t0 is not None:
                g = min(1.0, (now - gain_t0) / GAIN_RAMP_S)
                kp = KP_AIR + g * (KP_MAX - KP_AIR)
                kd = KD_AIR + g * (KD_MAX - KD_AIR)

            if frac >= 1.0:
                trim_req = read_trim_req(trim_req)
                lim = trim_req if trim_frozen is None else min(trim_req, trim_frozen)
                step = TRIM_RATE_DEG_S * dt
                new_trim = min(trim_cur + step, lim) if trim_cur < lim \
                    else max(trim_cur - step, lim)
                if trim_base is None and new_trim > 0.0 >= trim_cur:
                    # 0 상향 크로싱에서 베이스라인 캡처 (부동소수 정확일치 금지)
                    trim_base = {m: pos[m] - cmd_of(m, frac)
                                 for m in (L_ANK, R_ANK)}
                    stall_cnt = {L_ANK: 0, R_ANK: 0}
                elif trim_base is not None and new_trim <= 0.0:
                    trim_base = None    # 0 복귀 — 다음 프로브에서 재캡처
                trim_cur = new_trim

            for m in IDS:
                try:
                    tx(bus, m, pack_ak_mit_command(
                        math.radians(cmd_of(m, frac)), 0.0, kp, kd, 0.0))
                except Exception as e:
                    abort = f"송신실패 {type(e).__name__}"
                    break
            if abort:
                break
            drain(bus, pos, seen, fault, fault_cnt, fault_t)

            if not grounded and trim_cur > 0.3 and trim_frozen is None \
                    and trim_base is not None:
                for m in (L_ANK, R_ANK):
                    inc = abs((pos[m] - cmd_of(m, frac)) - trim_base[m])
                    stall_cnt[m] = stall_cnt[m] + 1 if inc > TRIM_STALL_AIR_DEG else 0
                    if stall_cnt[m] >= TRIM_STALL_TICKS:
                        trim_frozen = max(0.0, trim_cur - 1.0)
                        say(f"[프로브] 모터{m} 오차 증분 {inc:+.1f}° — 하드스톱 "
                            f"의심, 트림 {trim_frozen:.1f}°로 동결 "
                            f"(오탐이면 touch {UNFREEZE_FILE})")
                        break

            hard_fault = [m for m, c in fault_cnt.items() if c >= FAULT_ABORT_N]
            if hard_fault:
                abort = "모터 폴트 " + " ".join(
                    f"{m}:0x{fault[m]:X}" for m in sorted(hard_fault))
                break
            for m in fault:
                if m not in fault_warned:
                    fault_warned.add(m)
                    say(f"[경고] 모터{m} 폴트 1회 관측 (0x{fault[m]:X}) — "
                        f"{FAULT_ABORT_N}회 시 릴리즈")
            for m in IDS:
                if abs(pos[m] - cmd_of(m, frac)) > DEV_ABORT_DEG:
                    abort = f"모터{m} 이탈 {pos[m] - cmd_of(m, frac):+.1f}°"
                    break
                if now - seen[m] > FB_TIMEOUT_S:
                    abort = f"모터{m} 피드백 두절"
                    break
            if len(SIG_STOP) >= 2:
                abort = "시그널 2회 — 즉시 릴리즈"
            if abort or os.path.exists(KILL_FILE):
                abort = abort or "비상 정지(kill)"
                break
            if os.path.exists(STOP_FILE) or SIG_STOP:
                soft_stop = True
                if SIG_STOP:
                    say(f"[시그널 {SIG_STOP[0]}] 소프트 정지 경로")
                break

            if now - t_show >= 1.0:
                t_show = now
                errs = {m: pos[m] - cmd_of(m, frac) for m in IDS}
                worst = max(errs, key=lambda k: abs(errs[k]))
                phase = ("에어램프" if frac < 1.0 else
                         "게인램프" if gain_t0 and kp < KP_MAX - 1e-6 else
                         "에어유지" if gain_t0 is None else
                         "하중유지" if grounded else "운용대기")
                say(f"[{phase} {now - t0:5.1f}s] kp{kp:5.1f} 트림{trim_cur:4.1f}° "
                    f"| 최대오차 모터{worst} {errs[worst]:+.2f}° | 평균 "
                    f"{statistics.mean(abs(v) for v in errs.values()):.2f}°")
            time.sleep(max(0.0, TICK - (time.monotonic() - now)))

        if abort:
            say(f"[ABORT] {abort} — 즉시 릴리즈 (줄이 받습니다)")
            return 2
        if soft_stop:
            # 명령 동결 + 게인만 램프다운: 위치오차 성분 토크가 스텝 소실 없이
            # 게인에 비례해 서서히 빠짐. kill·폴트·추가 시그널은 계속 감시.
            say("[소프트정지] 명령 동결, 게인 3s 램프다운 — 줄 확인!")
            frozen = {m: cmd_of(m, min(1.0, (time.monotonic() - t0) / t_ramp))
                      for m in IDS}
            kp0, kd0 = kp, kd
            g0 = time.monotonic()
            while time.monotonic() - g0 < SOFT_RELEASE_S:
                if os.path.exists(KILL_FILE) or len(SIG_STOP) >= 2 \
                        or any(c >= FAULT_ABORT_N for c in fault_cnt.values()):
                    say("[소프트정지 중단] 즉시 릴리즈")
                    break
                g = 1.0 - (time.monotonic() - g0) / SOFT_RELEASE_S
                for m in IDS:
                    try:
                        tx(bus, m, pack_ak_mit_command(
                            math.radians(frozen[m]), 0.0,
                            max(2.0, kp0 * g), max(0.3, kd0 * g), 0.0))
                    except Exception:
                        pass
                drain(bus, pos, seen, fault, fault_cnt, fault_t)
                time.sleep(TICK)
            say("[완료] 릴리즈")
        return 0
    finally:
        release_all(bus)


if __name__ == "__main__":
    raise SystemExit(main())
