"""candump 인수 로그 분석기 — 핸드오프→stage8 "빡" 프레임 판독 (2026-08-02).

입력: candump -L 로그 (takeover_capture.sh 산출물)
  형식: (타임스탬프) can1 ID#HEX16
출력:
  1) 모터별 압축 이벤트 스트림 — 명령 종류(ENTER/EXIT/ZERO/CMD)나 디코드 값
     (pos/kp/kd)이 바뀌는 순간만 출력 (동일 프레임 반복은 접음)
  2) 모터별 '최대 명령 공백' 전후 대조표 — hang_stand 마지막 프레임 vs
     stage8 첫 프레임 (위치·게인 차이가 "빡"의 크기)

사용: python3 analyze_takeover.py ~/logs/takeover_*.candump [--motor 10]
      python3 analyze_takeover.py --selftest
"""
import argparse
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocols.ak_mit_command import pack_ak_mit_command  # noqa: E402
from protocols.ak_mit_decoder import AK70_10_LIMITS  # noqa: E402

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")
ENTER_HEX = "FF" * 7 + "FC"
EXIT_HEX = "FF" * 7 + "FD"


def uint_to_float(u, lo, hi, bits):
    span = float(hi) - float(lo)
    return float(lo) + span * float(u) / ((1 << bits) - 1)


def decode_cmd(data, limits=AK70_10_LIMITS):
    """명령 프레임 8바이트 → (pos_deg, vel, kp, kd, tau). pack의 정확한 역."""
    b = data
    p_int = (b[0] << 8) | b[1]
    v_int = (b[2] << 4) | (b[3] >> 4)
    kp_int = ((b[3] & 0x0F) << 8) | b[4]
    kd_int = (b[5] << 4) | (b[6] >> 4)
    tau_int = ((b[6] & 0x0F) << 8) | b[7]
    return (
        math.degrees(uint_to_float(p_int, limits.position_min_rad,
                                   limits.position_max_rad, 16)),
        uint_to_float(v_int, limits.velocity_min_rad_s,
                      limits.velocity_max_rad_s, 12),
        uint_to_float(kp_int, limits.kp_min, limits.kp_max, 12),
        uint_to_float(kd_int, limits.kd_min, limits.kd_max, 12),
        uint_to_float(tau_int, limits.torque_min, limits.torque_max, 12),
    )


def classify(hexdata):
    if hexdata == ENTER_HEX:
        return "ENTER", None
    if hexdata == EXIT_HEX:
        return "EXIT", None
    if len(hexdata) != 16:
        return "OTHER", None
    dec = decode_cmd(bytes.fromhex(hexdata))
    if dec[2] < 0.5 and dec[3] < 0.05:
        return "ZERO", dec
    return "CMD", dec


def parse(path):
    """[(ts, motor_id, kind, decoded)] — 명령 방향(ID 1..12)만."""
    out = []
    for line in open(path):
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        ts = float(m.group(1))
        can_id = int(m.group(2), 16)
        if not 1 <= can_id <= 12:
            continue  # 모터→호스트 피드백/기타는 제외
        kind, dec = classify(m.group(3).upper())
        out.append((ts, can_id, kind, dec))
    return out


def changed(a, b):
    if a is None or b is None:
        return a is not b
    return (abs(a[0] - b[0]) > 0.2 or abs(a[2] - b[2]) > 1.0
            or abs(a[3] - b[3]) > 0.1)


def fmt(dec):
    if dec is None:
        return ""
    return (f"pos={dec[0]:+8.2f}° v={dec[1]:+.1f} kp={dec[2]:6.1f} "
            f"kd={dec[3]:4.2f} tau={dec[4]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="candump -L 로그 경로")
    ap.add_argument("--motor", type=int, default=None, help="특정 모터만")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.log:
        ap.error("로그 경로 필요 (또는 --selftest)")

    events = parse(a.log)
    if not events:
        print("명령 프레임 없음 — 로그/채널 확인")
        return 1
    t0 = events[0][0]
    ids = sorted({e[1] for e in events})
    print(f"프레임 {len(events)}개, 모터 {ids}, 구간 "
          f"{events[-1][0] - t0:.1f}s\n")

    print("== 모터별 이벤트 스트림 (변화 시점만) ==")
    last = {}
    for ts, mid, kind, dec in events:
        if a.motor and mid != a.motor:
            continue
        pk, pd = last.get(mid, (None, None))
        if kind != pk or (kind in ("CMD", "ZERO") and changed(dec, pd)):
            print(f"[{ts - t0:8.3f}s] m{mid:02d} {kind:5s} {fmt(dec)}")
            last[mid] = (kind, dec)

    print("\n== 모터별 최대 명령 공백 전후 대조 (인수 지점 후보) ==")
    for mid in ids:
        if a.motor and mid != a.motor:
            continue
        cmds = [(ts, kind, dec) for ts, i, kind, dec in events
                if i == mid and kind in ("CMD", "ZERO")]
        if len(cmds) < 2:
            continue
        gaps = [(cmds[k + 1][0] - cmds[k][0], k) for k in range(len(cmds) - 1)]
        gap, k = max(gaps)
        before, after = cmds[k], cmds[k + 1]
        dpos = (after[2][0] - before[2][0]) if before[2] and after[2] else None
        print(f"m{mid:02d}: 공백 {gap:6.3f}s @ {before[0] - t0:.3f}s")
        print(f"   전: {before[1]:5s} {fmt(before[2])}")
        print(f"   후: {after[1]:5s} {fmt(after[2])}"
              + (f"   Δpos={dpos:+.2f}°" if dpos is not None else ""))
    return 0


def selftest():
    import tempfile
    lines = []
    t = 1000.0
    # hang_stand 홀드: m3 pos 8.69° kp150 → 공백 2s → stage8: pos 8.40° kp150
    for k in range(5):
        fb = pack_ak_mit_command(math.radians(8.69), 0.0, 150.0, 5.0, 0.0)
        lines.append(f"({t + k * 0.02:.6f}) can1 003#"
                     + "".join(f"{v:02X}" for v in fb))
    lines.append(f"({t + 0.5:.6f}) can1 003#{ENTER_HEX}")
    for k in range(3):
        fb = pack_ak_mit_command(math.radians(8.40), 0.0, 150.0, 5.0, 0.0)
        lines.append(f"({t + 2.1 + k * 0.02:.6f}) can1 003#"
                     + "".join(f"{v:02X}" for v in fb))
    fb0 = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)
    lines.append(f"({t + 3.0:.6f}) can1 003#"
                 + "".join(f"{v:02X}" for v in fb0))
    lines.append(f"({t + 3.1:.6f}) can1 003#{EXIT_HEX}")
    lines.append(f"({t + 0.1:.6f}) can1 000#0301020304050607")  # 피드백 무시 확인
    f = tempfile.NamedTemporaryFile("w", suffix=".candump", delete=False)
    f.write("\n".join(lines))
    f.close()

    ev = parse(f.name)
    kinds = [k for _, _, k, _ in ev]
    assert "ENTER" in kinds and "EXIT" in kinds and "ZERO" in kinds, kinds
    cmds = [d for _, _, k, d in ev if k == "CMD"]
    assert abs(cmds[0][0] - 8.69) < 0.05, cmds[0]   # 위치 왕복 (양자화 오차 내)
    assert abs(cmds[0][2] - 150.0) < 0.5 and abs(cmds[0][3] - 5.0) < 0.05
    assert abs(cmds[-1][0] - 8.40) < 0.05
    # 공백 탐지: 최대 갭이 인수 지점(0.58→2.1s)에 위치해야 함
    c3 = [(ts, k, d) for ts, i, k, d in ev if i == 3 and k in ("CMD", "ZERO")]
    gaps = [(c3[k + 1][0] - c3[k][0], k) for k in range(len(c3) - 1)]
    gap, k = max(gaps)
    assert 1.9 < gap < 2.1, gap
    print("[selftest] 파싱·디코드 왕복·ENTER/EXIT/ZERO 분류·공백 탐지 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
