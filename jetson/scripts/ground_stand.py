"""지면 기립 — 접지 웅크림 자세에서 천천히 일어서기 (2026-07-27 v3, 31건 적대검수 반영).

시나리오 (호이스트 없음 — 고정 로프 + 사람 보조):
  로봇이 이미 발 접촉 + 무릎 굽힘 자세로 로프에 지탱돼 있다. 이 자세에서
  강성만 올린 뒤(움직임 없음) 아주 천천히 스탠드 타깃으로 일어선다.
  끝낼 때는 역재생으로 웅크림(로프가 받치는 기하)까지 내려간 뒤 해제한다.

단계: READ → ENGAGE(현자세 강성상승) → RISE(1.5°/s 기립) → HOLD
      → DESCEND(역재생, gs_descend) → HOLD_CROUCH → RELEASE(gs_release)

지면 안전 철학 (에어와 반대 — v3에서 검수 반영 강화):
  - 이상 시 기본 동작 = 동결(타깃+게인+단계 전부) + 유지 + 경고. 자동 릴리즈 없음.
  - 정상 해제(gs_release)는 HOLD_CROUCH(웅크림 복귀 후)에서만 + 3초 카운트다운
    (매초 출력, 파일 삭제로 언제든 취소·취소 확인 출력).
  - 비상 즉시해제 = /tmp/gs_release_now (모든 단계, 카운트다운 없음) — 로봇이
    이미 넘어져 로프에 걸렸는데 모터가 버티며 싸우는 상황 전용.
  - 최후 자동해제: |오차|>25°가 신선한 샘플로 3틱 연속 (걸림/폭주 — 유지가 더 위험).
  - Ctrl+C: 마지막 명령·게인 그대로 유지 루프 전환. 연타해도 절대 해제 안 됨
    (해제는 파일로만). 유지 루프에도 drain·25°가드·상태출력 유지.
  - 릴리즈 송신은 실패 카운트로 검증 — 실패 시 "미확인" 경고 (버스 사망 시
    모터는 마지막 명령 유지 — 로프·손 확보 후 배선 점검).
  - 재실행 오발동 방지: --supported 플래그 필수 (시작 시 제로토크 읽기가 있어서
    로봇이 로프/웅크림으로 지지된 상태가 아니면 순간 무토크 낙하).

파일 명령: 멈춰=/tmp/gs_freeze  재개=/tmp/gs_resume  앉아=/tmp/gs_descend
           해제=/tmp/gs_release  비상즉시해제=/tmp/gs_release_now

사용: python3 ground_stand.py --supported [--channel can1] [--kp-final 150]
      [--kd-final 5] [--engage-sec 10] [--rise-rate 1.5] [--targets-json 파일]
      python3 ground_stand.py --selftest
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

FREEZE_FILE = "/tmp/gs_freeze"
RESUME_FILE = "/tmp/gs_resume"
DESCEND_FILE = "/tmp/gs_descend"
RELEASE_FILE = "/tmp/gs_release"
RELEASE_NOW_FILE = "/tmp/gs_release_now"
ALL_FILES = (FREEZE_FILE, RESUME_FILE, DESCEND_FILE, RELEASE_FILE, RELEASE_NOW_FILE)
IDS = list(range(1, 13))
AK45 = (6, 12)

#: 스탠드 영점 v4 (stand_snapshot_v4_20260727) — 모터 프레임 절대각 (직립)
STAND_DEG = {1: 31.99, 2: 2.20, 3: 8.69, 4: 14.22, 5: 34.28, 6: 4.95,
             7: 44.71, 8: 8.82, 9: -1.52, 10: 21.87, 11: 5.61, 12: 8.29}

ENGAGE_KP, ENGAGE_KD = 20.0, 1.0
FREEZE_DEV_DEG = 12.0
RELEASE_DEV_DEG = 25.0
RELEASE_DEV_TICKS = 3          # 25° 연속 틱 (단일 샘플 글리치 방어)
FRESH_S = 0.1                  # 가드 판단에 쓸 샘플 신선도
TORQUE_WARN = 20.0             # AK70 τ 경고 (AK45는 스케일 미확정 — 표시만)
TEMP_WARN_DELTA = 15           # 시작 대비 raw 상승 경고
FB_WARN_S = 1.0
TICK = 0.02


def gain_at(f, kp_final, kd_final):
    f = min(1.0, max(0.0, f))
    return (ENGAGE_KP + f * (kp_final - ENGAGE_KP),
            ENGAGE_KD + f * (kd_final - ENGAGE_KD))


def interp(p0, tgt, frac):
    return {m: p0[m] + frac * (tgt[m] - p0[m]) for m in IDS}


def selftest():
    assert gain_at(0.0, 150, 5) == (20.0, 1.0)
    assert gain_at(1.0, 150, 5) == (150.0, 5.0)
    assert gain_at(9.9, 150, 5) == (150.0, 5.0)
    kp, kd = gain_at(0.5, 150, 5)
    assert abs(kp - 85.0) < 1e-9 and abs(kd - 3.0) < 1e-9
    p0 = {m: 0.0 for m in IDS}
    t = {m: 10.0 for m in IDS}
    assert interp(p0, t, 0.0) == p0 and interp(p0, t, 1.0) == t
    assert all(abs(v - 5.0) < 1e-12 for v in interp(p0, t, 0.5).values())
    assert sorted(STAND_DEG) == IDS
    assert all(math.isfinite(v) and abs(v) < 120 for v in STAND_DEG.values())
    print("[selftest] ground_stand v3: 게인 스케줄·보간·타깃 무결성 ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supported", action="store_true",
                    help="로봇이 로프/웅크림으로 지지된 상태임을 확인 (필수 — "
                         "시작 시 제로토크 읽기가 있어 미지지 상태면 낙하)")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--kp-final", type=float, default=150.0)
    ap.add_argument("--kd-final", type=float, default=5.0)
    ap.add_argument("--engage-sec", type=float, default=10.0)
    ap.add_argument("--rise-rate", type=float, default=1.5)
    ap.add_argument("--targets-json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.supported:
        print("[거부] --supported 플래그 필요 — 시작 직후 제로토크 읽기 구간이 "
              "있어, 로봇이 로프/웅크림/사람으로 지지된 상태가 아니면 낙하합니다.\n"
              "       지지 확인 후: python3 ground_stand.py --supported")
        return 1

    sys.path.insert(0, "/home/mama/rl_calib")
    import can  # noqa: E402
    from ak_mit_command import pack_ak_mit_command  # noqa: E402
    from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402
    ENTER = bytes([0xFF] * 7 + [0xFC])
    EXIT = bytes([0xFF] * 7 + [0xFD])

    kp_final = min(a.kp_final, 200.0)
    kd_final = min(a.kd_final, 5.0)     # 실물 kd 상한 규칙 = 프로토콜 상한
    rise_rate = min(max(a.rise_rate, 0.3), 3.0)
    for f in ALL_FILES:
        if os.path.exists(f):
            os.remove(f)

    tgt = dict(STAND_DEG)
    if a.targets_json:
        tgt = {int(k): float(v) for k, v in json.load(open(a.targets_json)).items()}

    bus = can.Bus(channel=a.channel, interface="socketcan")

    def tx(mid, data):
        bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))

    def release_all(reason):
        """제로게인 3연발 + EXIT — 송신 실패 카운트로 결과 검증."""
        fails = 0
        for _ in range(3):
            for m in IDS:
                try:
                    tx(m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                except Exception:
                    fails += 1
            time.sleep(0.005)
        for m in IDS:
            try:
                tx(m, EXIT)
            except Exception:
                fails += 1
        if fails:
            print(f"[경고] 릴리즈({reason}) 송신 {fails}건 실패 — 미확인! 모터가 "
                  "아직 강성 유지 중일 수 있음. 로프·손 확보 후 배선 점검", flush=True)
        else:
            print(f"[릴리즈 완료] ({reason}) — 로봇이 로프/웅크림에 실립니다", flush=True)
        return fails

    pos, tau, temp, seen = {}, {m: 0.0 for m in IDS}, {m: 0 for m in IDS}, {}

    def drain():
        while True:
            try:
                msg = bus.recv(timeout=0.0)
            except Exception:
                return
            if msg is None:
                return
            if msg.is_error_frame or len(msg.data) < 6:
                continue
            m = msg.data[0]
            if m in tau:
                try:
                    fb = decode_ak_mit_feedback(bytes(msg.data))
                    pos[m] = fb.position_deg
                    tau[m] = fb.torque_est
                    temp[m] = fb.temperature_raw
                    seen[m] = time.time()
                except Exception:
                    pass

    released = False
    cmd = None
    cur_kp, cur_kd = ENGAGE_KP, ENGAGE_KD

    def guard_tick(now, dev_state):
        """25° 연속·신선 샘플 가드 — 릴리즈 필요 시 True. 메인·Ctrl+C 루프 공용."""
        if cmd is None:
            return False
        worst_m, worst_dev = None, 0.0
        for m in IDS:
            if m in pos and now - seen.get(m, 0) < FRESH_S:
                d = abs(pos[m] - cmd[m])
                if d > worst_dev:
                    worst_m, worst_dev = m, d
        if worst_m is not None and worst_dev > RELEASE_DEV_DEG:
            dev_state["n"] += 1
            if dev_state["n"] >= RELEASE_DEV_TICKS:
                print(f"[최후릴리즈] 모터{worst_m} 오차 {worst_dev:.1f}° > "
                      f"{RELEASE_DEV_DEG}° ({RELEASE_DEV_TICKS}틱 연속) — "
                      "걸림/폭주 판단", flush=True)
                return True
        else:
            dev_state["n"] = 0
        return False

    try:
        for m in IDS:
            try:
                tx(m, ENTER)
            except Exception:
                print(f"[FAIL] 모터 {m} ENTER 송신 실패 — 버스 확인 후 재시도", flush=True)
                return 1
            time.sleep(0.005)
        time.sleep(0.05)

        p0 = {}
        for m in IDS:
            reads = []
            for _ in range(3):
                try:
                    tx(m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                    t_e = time.time() + 0.03
                    while time.time() < t_e:
                        r = bus.recv(timeout=0.03)
                        if r is not None and not r.is_error_frame \
                                and len(r.data) >= 6 and r.data[0] == m:
                            reads.append(decode_ak_mit_feedback(
                                bytes(r.data)).position_deg)
                            break
                except Exception:
                    pass
            if len(reads) < 2:
                print(f"[FAIL] 모터 {m} 시작각 읽기 실패 — 토크 인가 전 중단·해제",
                      flush=True)
                release_all("시작각 실패, 게인 인가 전이라 무해")
                released = True
                return 1
            p0[m] = statistics.median(reads)
        pos.update(p0)
        now0 = time.time()
        seen.update({m: now0 for m in IDS})
        # 온도 기준값: p0 읽기는 drain을 안 거쳐 temp가 비어 있음 — 제로게인
        # 1라운드 폴링으로 채운 뒤 캡처 (안 채워진 모터는 이후 첫 수신값으로 보정)
        for m in IDS:
            try:
                tx(m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
        time.sleep(0.05)
        drain()
        temp0 = {m: (temp[m] if temp[m] else None) for m in IDS}

        dmax = max(abs(tgt[m] - p0[m]) for m in IDS)
        t_rise = max(4.0, dmax / rise_rate)
        print(f"[ground_stand v3] ENGAGE {a.engage_sec:.0f}s (kp20→{kp_final:.0f})"
              f" → RISE {t_rise:.0f}s ({rise_rate}°/s, 최대이동 {dmax:.1f}°)"
              f" → HOLD  | 예상 기립완료 ~{time.strftime('%H:%M:%S', time.localtime(now0 + a.engage_sec + t_rise))}",
              flush=True)
        print("이동량: " + " ".join(f"{m}:{tgt[m] - p0[m]:+.1f}" for m in IDS), flush=True)
        print("※ 기울거나 미끄러지면 → 즉시 로프를 당기세요 (모터는 계속 버팁니다). "
              "파일명령: 멈춰=gs_freeze 재개=gs_resume 앉아=gs_descend "
              "해제(웅크림 한정)=gs_release 비상즉시=gs_release_now", flush=True)

        phase = "ENGAGE"
        phase_t0 = time.time()
        engage_frozen_el = 0.0
        rise_frac = 0.0
        frozen = False
        frozen_cmd = None
        frozen_gains = None
        dev_state = {"n": 0}
        release_at = None
        torque_hot = {m: None for m in IDS}
        t_show = 0.0
        warned_fb = 0.0
        warned_temp = 0.0
        cd_last_print = -1

        while True:
            now = time.time()

            # ---- 비상 즉시해제 (모든 단계, 카운트다운 없음) ----
            if os.path.exists(RELEASE_NOW_FILE):
                print("[비상즉시해제] gs_release_now 감지", flush=True)
                release_all("비상")
                released = True
                return 3

            # ---- 단계 진행 (동결 시 완전 정지: 타깃·게인·타이머) ----
            if not frozen:
                if phase == "ENGAGE":
                    el = now - phase_t0
                    cmd = dict(p0)
                    cur_kp, cur_kd = gain_at(el / a.engage_sec, kp_final, kd_final)
                    if el >= a.engage_sec:
                        phase, phase_t0 = "RISE", now
                        print("[RISE] 기립 시작 — 로프 그대로, 손은 골반 스팟. "
                              "이상하면 '멈춰'!", flush=True)
                elif phase == "RISE":
                    cur_kp, cur_kd = kp_final, kd_final
                    rise_frac = min(1.0, rise_frac + TICK / t_rise)
                    cmd = interp(p0, tgt, rise_frac)
                    if rise_frac >= 1.0:
                        phase, phase_t0 = "HOLD", now
                        print("[HOLD] 기립 완료 — 유지 중. (이 단계에선 '멈춰'가 "
                              "의미 없음 — 위험하면 로프를 당기세요) 착석복귀 = "
                              "gs_descend", flush=True)
                elif phase == "HOLD":
                    cur_kp, cur_kd = kp_final, kd_final
                    cmd = interp(p0, tgt, rise_frac)
                    if os.path.exists(DESCEND_FILE):
                        os.remove(DESCEND_FILE)
                        phase, phase_t0 = "DESCEND", now
                        print("[DESCEND] 착석 복귀 시작 (역재생)", flush=True)
                elif phase == "DESCEND":
                    cur_kp, cur_kd = kp_final, kd_final
                    rise_frac = max(0.0, rise_frac - TICK / t_rise)
                    cmd = interp(p0, tgt, rise_frac)
                    if rise_frac <= 0.0:
                        phase, phase_t0 = "HOLD_CROUCH", now
                        print("[HOLD_CROUCH] 웅크림 복귀 완료 — 로프·손 확인 후 "
                              "gs_release (3초 카운트다운, 파일 삭제=취소)", flush=True)
                else:  # HOLD_CROUCH
                    cur_kp, cur_kd = kp_final, kd_final
                    cmd = dict(p0)
            else:
                cmd = frozen_cmd
                cur_kp, cur_kd = frozen_gains

            # ---- 송신 ----
            for m in IDS:
                try:
                    tx(m, pack_ak_mit_command(
                        math.radians(cmd[m]), 0.0, cur_kp, cur_kd, 0.0))
                except Exception:
                    pass
            drain()

            # ---- 가드 ----
            if guard_tick(now, dev_state):
                release_all("25도 최후")
                released = True
                return 2

            fresh_worst_m, fresh_worst = None, 0.0
            for m in IDS:
                if now - seen.get(m, 0) < FRESH_S:
                    d = abs(pos[m] - cmd[m])
                    if d > fresh_worst:
                        fresh_worst_m, fresh_worst = m, d
            want_freeze = os.path.exists(FREEZE_FILE)
            if want_freeze:
                try:
                    os.remove(FREEZE_FILE)
                except OSError:
                    pass
            if not frozen and (want_freeze or fresh_worst > FREEZE_DEV_DEG):
                if phase in ("HOLD", "HOLD_CROUCH") and want_freeze \
                        and fresh_worst <= FREEZE_DEV_DEG:
                    print("[안내] 이 단계는 타깃이 이미 고정 — '멈춰'가 바꿀 것이 "
                          "없습니다. 위험하면 로프를 당기세요 (모터 유지됨)", flush=True)
                else:
                    frozen = True
                    frozen_cmd = dict(cmd)
                    frozen_gains = (cur_kp, cur_kd)
                    if phase == "ENGAGE":
                        engage_frozen_el = now - phase_t0
                    why = "수동" if want_freeze else \
                        f"모터{fresh_worst_m} 오차 {fresh_worst:.1f}°"
                    print(f"[FREEZE] {why} — 타깃·게인·진행 전부 동결 유지. "
                          "재개=gs_resume, 비상=gs_release_now", flush=True)
            if frozen and os.path.exists(RESUME_FILE):
                os.remove(RESUME_FILE)
                frozen = False
                frozen_cmd = None
                frozen_gains = None
                if phase == "ENGAGE":       # 게인 램프 연속성 복원
                    phase_t0 = now - engage_frozen_el
                print("[RESUME] 동결 해제 — 진행 재개", flush=True)

            if seen:
                fb_age = now - min(seen.values())
                if fb_age > FB_WARN_S and now - warned_fb > 5.0:
                    warned_fb = now
                    print("[경고] 피드백 두절 — 버스 이상 시 모터는 마지막 명령 "
                          "유지. 로프·손 확보 후 배선 점검", flush=True)

            for m in IDS:
                hot = abs(tau[m]) > TORQUE_WARN and m not in AK45
                if hot and torque_hot[m] is None:
                    torque_hot[m] = now
                elif not hot:
                    torque_hot[m] = None
                if torque_hot[m] and now - torque_hot[m] > 3.0:
                    torque_hot[m] = now + 2.0
                    print(f"[경고] 모터{m} 토크 {tau[m]:+.1f} 지속 — 하중 집중",
                          flush=True)
            for m in IDS:                      # 기준 미확보 모터는 첫 수신값으로
                if temp0[m] is None and temp[m]:
                    temp0[m] = temp[m]
            deltas = [temp[m] - temp0[m] for m in IDS
                      if temp0[m] is not None and temp[m]]
            hot_temp = max(deltas) if deltas else 0
            if hot_temp > TEMP_WARN_DELTA and now - warned_temp > 10.0:
                warned_temp = now
                print(f"[경고] 온도 raw +{hot_temp} 상승 — 길게 끌지 말고 착석 "
                      "복귀 고려", flush=True)

            # ---- 정상 해제 (웅크림 한정, 카운트다운 취소 가능) ----
            if os.path.exists(RELEASE_FILE):
                if phase != "HOLD_CROUCH":
                    os.remove(RELEASE_FILE)
                    print(f"[거부] 해제는 웅크림 복귀 후에만 ({phase} 단계) — "
                          "gs_descend 먼저. 진짜 비상이면 gs_release_now", flush=True)
                elif release_at is None:
                    release_at = now + 3.0
                    cd_last_print = -1
                    print("[릴리즈] 3초 카운트다운 — 로프·손 최종 확인! "
                          "(rm gs_release = 취소)", flush=True)
                if release_at is not None:
                    remain = int(math.ceil(release_at - now))
                    if remain != cd_last_print and remain > 0:
                        cd_last_print = remain
                        print(f"[릴리즈] {remain}…", flush=True)
                    if now >= release_at:
                        release_all("정상 종료")
                        released = True
                        return 0
            elif release_at is not None:
                release_at = None
                print("[릴리즈 취소됨] — 유지 계속", flush=True)

            if now - t_show >= 1.0:
                t_show = now
                errs = {m: pos[m] - cmd[m] for m in IDS}
                we = max(errs, key=lambda k: abs(errs[k]))
                print(f"[{phase}{'/FROZEN' if frozen else ''} "
                      f"{now - phase_t0:4.0f}s kp{cur_kp:.0f} "
                      f"진행{rise_frac * 100:3.0f}%] 오차max m{we} {errs[we]:+.2f}° | "
                      f"τ무릎 {tau[4]:+.1f}/{tau[10]:+.1f} 힙 {tau[1]:+.1f}/{tau[7]:+.1f} "
                      f"발목F {tau[5]:+.1f}/{tau[11]:+.1f} 롤raw {tau[6]:+.1f}/{tau[12]:+.1f} | "
                      f"tempΔ {hot_temp}", flush=True)
            time.sleep(max(0.0, TICK - (time.time() - now)))

    except KeyboardInterrupt:
        # Ctrl+C = 마지막 명령·게인 유지 루프. 연타 절대 해제 금지 (파일로만).
        print(f"\n[Ctrl+C] 마지막 명령 유지 (kp{cur_kp:.0f}) — 해제는 "
              f"{RELEASE_FILE}(3초) 또는 {RELEASE_NOW_FILE}(즉시)", flush=True)
        if cmd is None:
            release_all("토크 인가 전 — 무해")   # 게인 걸기 전이면 해제가 안전
            released = True
            return 0
        hold_cmd = dict(cmd)
        dev_state = {"n": 0}
        t_show2 = 0.0
        release_at2 = None
        while True:
            try:
                now = time.time()
                if os.path.exists(RELEASE_NOW_FILE):
                    release_all("비상")
                    released = True
                    return 3
                for m in IDS:
                    try:
                        tx(m, pack_ak_mit_command(
                            math.radians(hold_cmd[m]), 0.0, cur_kp, cur_kd, 0.0))
                    except Exception:
                        pass
                drain()
                if guard_tick(now, dev_state):
                    release_all("25도 최후")
                    released = True
                    return 2
                if os.path.exists(RELEASE_FILE):
                    if release_at2 is None:
                        release_at2 = now + 3.0
                        print("[릴리즈] 3초 카운트다운 (rm = 취소)", flush=True)
                    elif now >= release_at2:
                        if os.path.exists(RELEASE_FILE):   # 최종 재확인
                            release_all("Ctrl+C 후 파일 해제")
                            released = True
                            return 0
                        release_at2 = None
                elif release_at2 is not None:
                    release_at2 = None
                    print("[릴리즈 취소됨]", flush=True)
                if now - t_show2 >= 2.0:
                    t_show2 = now
                    if pos:
                        we = max(IDS, key=lambda k: abs(pos[k] - hold_cmd[k]))
                        print(f"[유지중 kp{cur_kp:.0f}] 오차max m{we} "
                              f"{pos[we] - hold_cmd[we]:+.2f}°", flush=True)
                time.sleep(TICK)
            except KeyboardInterrupt:
                print("[무시] Ctrl+C 연타 — 해제는 파일로만 "
                      f"({RELEASE_FILE} / {RELEASE_NOW_FILE})", flush=True)
                continue
    finally:
        if not released:
            print("[비정상 종료] 릴리즈 안 함 — 모터는 마지막 명령 유지 중. "
                  "로프·손 확보 후 수동 해제 필요", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
