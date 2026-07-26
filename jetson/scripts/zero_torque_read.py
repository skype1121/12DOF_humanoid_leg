"""제로 토크 엔코더 읽기 — 실물 캘리브레이션 전용 (2026-07-27, 승윤님 승인).

원리: AK 모터 MIT 모드 진입(FF..FC) 후 전부-0 명령(kp=0, kd=0, tau=0)으로
폴링 — 토크 출력이 0이라 모터는 완전히 힘 빠진 상태 그대로, 절대 안 움직임.
응답 프레임에서 엔코더 각도만 읽는다. 종료 시 모드 해제(FF..FD).

모드:
  probe    --id 1                 모터 1개 검증 (진입→읽기 1회→해제)
  sweep    --ids 1-12 --out f.json  수동 가동범위 스윕 기록 (관절별 min/max,
                                    Ctrl+C로 종료·저장. 1초마다 현황 출력)
  snapshot --ids 1-12 --out f.json  현재 자세 스냅샷 (20회 중앙값)

참고: 발목롤(AK45-36, 모터 6·12) 위치 스케일 = 1.0 확정 (2026-07-27 폰 각도기
실측, 오차 0.8%) — 보정 불필요. 속도 디코드만 과대판독(--vmax 상향으로 대응).
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

ENTER_MIT = bytes([0xFF] * 7 + [0xFC])
EXIT_MIT = bytes([0xFF] * 7 + [0xFD])
JOINT_BY_MOTOR = {}
try:
    _hw = json.load(open("/home/mama/rl_calib/robot_12dof_hardware_map.json"))
    for _j, _c in (_hw.get("joints") or _hw).items():
        if isinstance(_c, dict) and "motor_id" in _c:
            JOINT_BY_MOTOR[int(_c["motor_id"])] = _j
except Exception:
    pass


def parse_ids(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def zero_cmd_frame():
    return pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)


def send(bus, mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


def poll_once(bus, mid, timeout=0.05):
    """0토크 명령 1회 송신 → 해당 모터 응답 수신·디코드 (없으면 None)."""
    send(bus, mid, zero_cmd_frame())
    t_end = time.time() + timeout
    while time.time() < t_end:
        msg = bus.recv(timeout=max(0.0, t_end - time.time()))
        if msg is None:
            return None
        if len(msg.data) >= 6 and msg.data[0] == mid:
            return decode_ak_mit_feedback(bytes(msg.data))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "sweep", "snapshot"])
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--ids", default="1-12")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    bus = can.Bus(channel=a.channel, interface="socketcan")

    if a.mode == "probe":
        mid = a.id
        joint = JOINT_BY_MOTOR.get(mid, "?")
        print(f"[probe] 모터 {mid} ({joint}) — MIT 진입 → 0토크 읽기 → 해제")
        send(bus, mid, ENTER_MIT)
        time.sleep(0.05)
        fb = poll_once(bus, mid, timeout=0.2)
        send(bus, mid, EXIT_MIT)
        if fb is None:
            print("[FAIL] 응답 없음 — 모터 전원/배선/ID 확인")
            return 1
        print(f"[OK] pos={fb.position_deg:+.2f}° vel={fb.velocity_rad_s:+.3f}rad/s "
              f"tau={fb.torque_est:+.2f} temp={fb.temperature_raw} "
              f"err={'OK' if fb.error_ok else fb.error_raw}")
        return 0

    ids = parse_ids(a.ids)
    stats = {m: {"min": None, "max": None, "now": None, "n": 0, "miss": 0}
             for m in ids}
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    for m in ids:
        send(bus, m, ENTER_MIT)
        time.sleep(0.01)
    print(f"[{a.mode}] 모터 {ids} 0토크 폴링 시작 (Ctrl+C로 종료·저장)")

    def build_out():
        out = {}
        for m in ids:
            s = stats[m]
            rec = {"joint": JOINT_BY_MOTOR.get(m, "?"), "motor_id": m,
                   "min_deg": s["min"], "max_deg": s["max"],
                   "samples": s["n"], "miss": s["miss"],
                   "ak45_scale_unconfirmed": m in (6, 12)}
            if a.mode == "snapshot" and samples[m]:
                rec["median_deg"] = statistics.median(samples[m])
            out[str(m)] = rec
        return out

    def autosave():
        if not a.out:
            return
        tmp = a.out + ".part"
        json.dump(build_out(), open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, a.out)

    samples = {m: [] for m in ids}
    t_show = time.time()
    t_save = time.time()
    try:
        while not stop:
            for m in ids:
                st = stats[m]
                try:
                    fb = poll_once(bus, m, timeout=0.01)
                except Exception:   # 송신버퍼 폭주 등 버스 이상 — 결측 처리 후 계속
                    st["miss"] += 1
                    continue
                if fb is None:
                    st["miss"] += 1
                    continue
                d = fb.position_deg
                st["now"] = d
                st["n"] += 1
                st["min"] = d if st["min"] is None else min(st["min"], d)
                st["max"] = d if st["max"] is None else max(st["max"], d)
                if a.mode == "snapshot":
                    samples[m].append(d)
            if a.mode == "snapshot" and all(
                    len(v) >= 20 for v in samples.values()):
                break
            if time.time() - t_show > 1.0:
                t_show = time.time()
                row = " | ".join(
                    f"{m}:{s['now']:+.1f}({s['min']:+.1f}~{s['max']:+.1f})"
                    for m, s in stats.items() if s["now"] is not None)
                miss = {m: s["miss"] for m, s in stats.items() if s["miss"]}
                print(row + (f"  miss={miss}" if miss else ""), flush=True)
            if time.time() - t_save > 5.0:      # 크래시 대비 주기 자동저장
                t_save = time.time()
                try:
                    autosave()
                except Exception:
                    pass
            time.sleep(0.005)
    finally:
        for m in ids:
            try:
                send(bus, m, EXIT_MIT)
            except Exception:   # 버스 이상으로 해제 실패 — 저장은 계속
                pass

    out = build_out()
    for m in ids:
        s = stats[m]
        rec = out[str(m)]
        print(f"모터 {m:2d} {rec['joint']:24s} "
              f"min {s['min'] if s['min'] is not None else '-':>8} "
              f"max {s['max'] if s['max'] is not None else '-':>8} "
              f"(n={s['n']}, miss={s['miss']})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("saved:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
