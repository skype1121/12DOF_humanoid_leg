"""부팅 인코더 프레임 체크 — 전원 인가 후 첫 실행 도구 (2026-07-28, 검수 2라운드 22건 반영).

배경: AK 모터 엔코더는 로터(감속기 입력축) 기준 절대식이라, 전원이 끊기면
멀티턴 카운트를 잃고 관절각이 창 크기(AK70-10=36°, AK45-36=10°)의 정수배(k)만큼
어긋난 프레임으로 깨어날 수 있다 (2026-07-27 실증: 웅크림 자세 순단 후
모터 1·5·11이 정확히 36° 시프트).

원리: 로봇을 '기준표를 캡처했던 자세'에 두고 실행하면
    reading = ref + k×window + 자세오차
이므로 k = round((reading−ref)/window), 잔차 = 나머지. k≠0 = 프레임 어긋남.

제로토크 폴링만 사용 — 모터는 절대 움직이지 않는다 (zero_torque_read 검증 패턴).

사용:
  python3 boot_pose_check.py --ref stand              # 하드웨어맵 stand_zero 기준
  python3 boot_pose_check.py --ref journal            # 마지막 관측 저널 기준
  python3 boot_pose_check.py --ref hang_snapshot.json # 임의 기준 JSON
  ... --save v5_ref.json    # 현재 읽기를 새 기준표(예: v5)로 저장

판정 가이드 (종료코드 0 = 12모터 전부 검사되고 전부 일치일 때만):
  ✓ 일치            → 그대로 진행
  ✗ 프레임 어긋남    → --ref stand일 때만 stand_zero 보정값 제안 (다른 기준은
                      프레임 변화 사실만 보고 — 보정값 산출 불가)
  ⚠ 자세불일치      → 로봇 자세가 기준 자세와 다름 — 자세를 맞추고 재실행
  ⚠ 폴트/산포/응답률 → 해당 모터 신뢰 불가 — 원인 해결 후 재실행
  기준없음/무응답    → 해당 모터는 미검사 — OK 아님 (전수 검사 강제)
주의: 발목롤(6·12)은 창이 10°라 자세오차가 ±5°를 넘으면 k 판정이 원리적으로
불가 — 발을 기준 자세와 같은 상태(자유 늘어짐 or 평평)로 두고 실행할 것.
저널: '✓ 일치'로 검증된 모터의 관측만 병합 갱신 (시프트·폴트·불안정 읽기는
저널에 절대 안 들어감 — 어긋난 프레임이 다음 실행의 기준이 되는 것 차단).
"""
import argparse
import json
import os
import signal
import statistics
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER_MIT = bytes([0xFF] * 7 + [0xFC])
EXIT_MIT = bytes([0xFF] * 7 + [0xFD])
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
JOURNAL = "/home/mama/rl_calib/last_pose_journal.json"

#: 관절 1회전당 로터 1회전 창 (deg) — 감속비가 정수라 시프트는 항상 이 정수배
WINDOW = {m: (10.0 if m in (6, 12) else 36.0) for m in range(1, 13)}
#: 자세오차 허용 (deg) — 매달림 재현성 ~2° 실측 기준 여유. AK45는 창의 절반 미만 강제
TOL = {m: (3.5 if m in (6, 12) else 8.0) for m in range(1, 13)}
POLL_N = 15          #: 모터당 폴링 횟수
MIN_VALID = 12       #: 유효 응답 최소 (미만이면 응답률 저조 — 반접촉 전조 의심)


def load_json_guarded(path, what):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        hint = (" — 최초 실행이면 --ref stand로 실행 (저널은 실행마다 자동 생성)"
                if path == JOURNAL else "")
        raise SystemExit(f"[중단] {what} 없음: {path}{hint}")
    except (json.JSONDecodeError, ValueError):
        raise SystemExit(f"[중단] {what} 파손(JSON 오류): {path} — "
                         f"--ref stand 사용 후 재생성 권장")


def load_hw():
    hw = load_json_guarded(HW_MAP, "하드웨어맵")
    out = {}
    for j, c in (hw.get("joints") or hw).items():
        if isinstance(c, dict) and "motor_id" in c:
            out[int(c["motor_id"])] = {"joint": j,
                                       "stand_zero": c.get("stand_zero_deg")}
    return out


def load_ref(spec, hwinfo):
    """기준표 로드 → ({motor_id: deg}, 설명). stand/journal/파일경로 지원."""
    if spec == "stand":
        ref = {m: v["stand_zero"] for m, v in hwinfo.items()
               if v["stand_zero"] is not None}
        return ref, "하드웨어맵 stand_zero (직립 정렬 자세)"
    path = JOURNAL if spec == "journal" else spec
    d = load_json_guarded(path, "저널" if spec == "journal" else "기준파일")
    try:
        if isinstance(d, dict) and "pos_deg" in d:       # 저널 형식
            now = time.time()
            age = now - float(d.get("ts", now))
            desc = f"저널 {path} ({age/60:.0f}분 전 실행)"
            mts = d.get("motor_ts", {}) or {}
            stale = [k for k, t in mts.items() if now - float(t) > age + 60]
            if stale:
                desc += (f" ⚠ 모터 {sorted(int(s) for s in stale)}는 그보다 "
                         f"오래된 관측(결측으로 이월된 값)")
            return ({int(k): float(v) for k, v in d["pos_deg"].items()}, desc)
        ref = {}
        for k, v in d.items():                            # 스냅샷/일반 형식
            if not str(k).isdigit():
                continue
            if isinstance(v, dict):
                val = v.get("median_deg", v.get("now"))
            else:
                val = v
            if val is not None:
                ref[int(k)] = float(val)
        return ref, f"기준파일 {path}"
    except (TypeError, AttributeError, ValueError, KeyError):
        raise SystemExit(f"[중단] 기준파일 형식 이상: {path} — 모터별 "
                         f"median_deg를 가진 스냅샷/저널 JSON이어야 함")


def poll_stats(bus, mid, n=POLL_N):
    """제로토크(전부-0) 폴링 n회 → 통계 dict. 샘플 단위 예외 흡수.

    med=None이면 무응답. spread(max−min)로 폴링 중 순단/프레임 점프 검출,
    n_ok 저조는 반접촉 전조 신호.
    """
    vals, err = [], None
    for _ in range(n):
        try:
            bus.send(can.Message(arbitration_id=mid,
                                 data=pack_ak_mit_command(0, 0, 0, 0, 0),
                                 is_extended_id=False))
        except Exception:
            time.sleep(0.02)
            continue
        t_end = time.time() + 0.05
        try:
            while time.time() < t_end:
                msg = bus.recv(timeout=max(0.0, t_end - time.time()))
                if msg is None:
                    break
                if not msg.is_error_frame and len(msg.data) == 8 \
                        and msg.data[0] == mid:
                    fb = decode_ak_mit_feedback(bytes(msg.data))
                    vals.append(fb.position_deg)
                    if not fb.error_ok:
                        err = fb.error_raw
                    break
        except Exception:   # recv/decode 이상 — 이 샘플만 버리고 계속
            pass
        time.sleep(0.01)
    return {"med": statistics.median(vals) if len(vals) >= 3 else None,
            "n_ok": len(vals), "n_try": n, "err": err,
            "spread": (max(vals) - min(vals)) if vals else 0.0}


def save_json_atomic(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="stand",
                    help="stand | journal | 기준 JSON 경로")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--save", default="",
                    help="현재 읽기를 새 기준표 JSON으로 저장 (예: v5)")
    ap.add_argument("--no-journal", action="store_true",
                    help="저널 갱신 생략 (기본: '✓ 일치' 검증분만 병합 기록)")
    a = ap.parse_args()
    if a.save and os.path.abspath(a.save) == JOURNAL:
        raise SystemExit(f"[중단] --save 경로가 저널({JOURNAL})과 동일 — "
                         f"저널은 자동 관리되므로 다른 경로를 쓸 것")

    hwinfo = load_hw()
    ref, ref_desc = load_ref(a.ref, hwinfo)
    if not ref:
        raise SystemExit("[중단] 기준표가 비어 있음 — 파일 형식 확인 "
                         "(모터별 median_deg 필요, sweep 출력은 기준으로 못 씀)")
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    try:
        bus = can.Bus(channel=a.channel, interface="socketcan")
    except Exception as e:
        raise SystemExit(f"[중단] CAN {a.channel} 열기 실패"
                         f"({type(e).__name__}: {e}) — can_up.sh로 캔업 후 재실행")
    print(f"[boot_pose_check] 기준: {ref_desc}")
    print("전제: 로봇이 지금 '기준표를 캡처했던 자세'에 있어야 판정 유효\n")

    readings, exit_fail = {}, []
    interrupted = False
    try:
        for m in range(1, 13):
            if stop:
                interrupted = True
                break
            try:
                bus.send(can.Message(arbitration_id=m, data=ENTER_MIT,
                                     is_extended_id=False))
            except Exception:
                continue
            time.sleep(0.02)
            readings[m] = poll_stats(bus, m)
    finally:
        for m in range(1, 13):
            try:
                bus.send(can.Message(arbitration_id=m, data=EXIT_MIT,
                                     is_extended_id=False))
                time.sleep(0.005)
            except Exception:
                exit_fail.append(m)
        if exit_fail:
            print(f"[경고] 모터 {exit_fail} MIT 해제 송신 실패 — 제로게인 "
                  f"상태라 무토크지만, 버스 복구 후 재실행으로 해제 권장")

    shifted, pose_bad, dead = [], [], []
    unchecked, faulted, unstable, good = [], [], [], set()
    print(f"{'모터':>4} {'관절':24s} {'읽기':>8} {'기준':>8} "
          f"{'k':>3} {'잔차':>7}  판정")
    for m in range(1, 13):
        joint = hwinfo.get(m, {}).get("joint", "?")
        st = readings.get(m)
        if st is None:
            unchecked.append(m)
            print(f"{m:>4} {joint:24s} {'미폴링(중단/송신실패)':>8}")
            continue
        if st["med"] is None:
            dead.append(m)
            print(f"{m:>4} {joint:24s} {'무응답':>8}")
            continue
        flags = ""
        if st["err"]:
            faulted.append(m)
            flags += f" ⚠폴트(err={st['err']})"
        if st["spread"] > TOL[m]:
            unstable.append(m)
            flags += f" ⚠산포{st['spread']:.1f}°(폴링 중 점프?)"
        elif st["n_ok"] < MIN_VALID:
            unstable.append(m)
            flags += f" ⚠응답률{st['n_ok']}/{st['n_try']}(반접촉?)"
        r = st["med"]
        if m not in ref:
            unchecked.append(m)
            print(f"{m:>4} {joint:24s} {r:>+8.2f} {'기준없음':>8}"
                  f"{'':>3} {'':>7}  미검사{flags}")
            continue
        w = WINDOW[m]
        delta = r - ref[m]
        k = round(delta / w)
        resid = delta - k * w
        if abs(resid) > TOL[m]:
            verdict = "⚠ 자세불일치(잔차 큼)"
            pose_bad.append(m)
        elif k != 0:
            verdict = f"✗ 프레임 {k:+d}창 어긋남"
            shifted.append((m, k))
        else:
            verdict = "✓ 일치"
            if not flags:
                good.add(m)
        print(f"{m:>4} {joint:24s} {r:>+8.2f} {ref[m]:>+8.2f} "
              f"{k:>+3d} {resid:>+7.2f}  {verdict}{flags}")

    print()
    problems = bool(dead or pose_bad or shifted or unchecked
                    or faulted or unstable)
    io_fail = False
    if interrupted:
        print("[중단됨] 사용자 인터럽트 — 판정 불완전")
    if dead:
        print(f"[결측] 모터 {dead} 무응답 — 전원/커넥터 확인 후 재실행")
    if unchecked:
        print(f"[미검사] 모터 {unchecked} 기준값 없음/미폴링 — 전수 검사 전엔 "
              f"통과 아님 (--ref stand 재실행 또는 완전한 기준표 사용)")
    if faulted:
        print(f"[폴트] 모터 {faulted} 에러 플래그 — 위치값 신뢰 불가, 원인 확인")
    if unstable:
        print(f"[불안정] 모터 {unstable} 산포/응답률 이상 — 커넥터·전원 점검 "
              f"후 재실행 (판정 신뢰 불가)")
    if pose_bad:
        print(f"[보류] 모터 {pose_bad} 잔차 초과 — 로봇 자세를 기준 자세로 "
              f"맞춘 뒤 재실행 (지금은 k 판정 신뢰 불가)")
    if shifted and not pose_bad and not unstable:
        if a.ref == "stand":
            print("[프레임 어긋남] 보정 제안 (현재 프레임 기준 새 stand_zero):")
            for m, k in shifted:
                sz = hwinfo[m]["stand_zero"]
                print(f"  모터 {m}: stand_zero {sz:+.2f} → "
                      f"{sz + k * WINDOW[m]:+.2f}  (k={k:+d}×{WINDOW[m]:.0f}°)")
            print("  → 하드웨어맵에 적용하거나, --save로 영점 재캡처(권장) 후 갱신")
        else:
            print(f"[프레임 어긋남] 기준({a.ref}) 대비 시프트 검출 — 이 기준의 "
                  f"프레임 상태를 모르므로 stand_zero 보정값은 산출 불가.")
            print("  → 직립 정렬 후 --ref stand로 재실행하거나 --save로 재앵커")
    if not problems:
        print("[OK] 12모터 전수 검사·프레임 일치 — 기립 등 다음 단계 진행 가능")

    if a.save:
        out = {str(m): {"joint": hwinfo.get(m, {}).get("joint", "?"),
                        "median_deg": readings[m]["med"]}
               for m in readings if readings[m]["med"] is not None}
        missing = sorted(set(range(1, 13)) - {int(k) for k in out})
        out["_meta"] = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ref_compared": a.ref, "tool": "boot_pose_check",
                        "partial_missing": missing,
                        "suspect_faulted": faulted, "suspect_unstable": unstable}
        try:
            save_json_atomic(a.save, out)
            if missing or faulted or unstable:
                print(f"[저장⚠] 기준표({12 - len(missing)}/12, 결측 {missing}, "
                      f"의심 {sorted(set(faulted + unstable))}) → {a.save} — "
                      f"완전·클린판 재캡처 권장")
            else:
                print(f"[저장] 새 기준표(12/12 클린) → {a.save}")
        except OSError as e:
            io_fail = True
            print(f"[저장실패] {a.save}: {e}")

    if not a.no_journal:
        # '✓ 일치'로 검증된 모터만 저널 갱신 — 시프트/폴트/불안정 읽기가
        # 다음 실행의 기준이 되는 자기세탁 차단
        trusted = {str(m): readings[m]["med"] for m in good}
        if trusted:
            now = time.time()
            old = {}
            try:
                old = json.load(open(JOURNAL))
                if not isinstance(old, dict):
                    old = {}
            except Exception:
                old = {}
            pos = dict(old.get("pos_deg", {}) or {})
            mts = dict(old.get("motor_ts", {}) or {})
            pos.update(trusted)
            mts.update({m: now for m in trusted})
            try:
                save_json_atomic(JOURNAL, {"ts": now, "pos_deg": pos,
                                           "motor_ts": mts})
                held = sorted(int(k) for k in pos if k not in trusted)
                note = f" (미검증 {held}는 이전 관측 유지)" if held else ""
                print(f"[저널] 검증 관측 {len(trusted)}모터 갱신{note} → {JOURNAL}")
            except OSError as e:
                io_fail = True
                print(f"[저널실패] {e}")
        else:
            print("[저널] '✓ 일치' 검증 관측 없음 — 저널 갱신 생략 "
                  "(어긋난 프레임을 기준으로 삼지 않기 위함)")

    return 1 if (problems or io_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
