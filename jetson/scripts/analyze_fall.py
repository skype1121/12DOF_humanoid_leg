#!/usr/bin/env python3
"""무개입 낙하 판정기 — rl_bridge 블랙박스(jsonl)에서 자립/기울임을 실측 판정.

5차(v6b) 무개입 낙하는 눈과 스냅샷으로 읽었다. 그때 확정된 사실:
"균형 잡힌 몸 위에서 1초부터 전방으로 단조 누적(+1.6→+19°), 발진도 스텝도
없음" = payload 뒤 배치 학습의 순수 산물. 6차(v7)의 질문은 단 하나 —
**그 단조 전방 밀기가 사라졌는가**. 이 스크립트가 그 답을 숫자로 낸다.

사용:
    python3 jetson/scripts/analyze_fall.py ~/rl_bridge_bb/rl_bridge_XXXX.jsonl
    python3 jetson/scripts/analyze_fall.py <v7.jsonl> --compare <v6b.jsonl>

판정 지표는 원격 오판 사고(IMU 스냅샷 1장으로 '섰다' 오판) 이후 확립된
**발력 총합 %**를 1순위로, 기울임 시계열을 2순위로 쓴다.
"""
import argparse
import json
import math
import os
import sys

BODY_WEIGHT_N = 117.7      # 실물 12kg × 9.81
LOAD_STANDING = 85.0       # 이 % 이상이면 하중을 정책이 받고 있다고 본다
LOAD_LOST = 50.0           # 이 % 아래로 떨어지면 줄/사람에게 넘어간 것
LEAN_FALL_DELTA = 15.0     # 시작 자세 대비 이만큼 더 기울면 자립 종료
EARLY_WINDOW_S = 3.0       # "1초부터 단조 누적" 을 잡아내는 초기 창
MIN_GRADE_S = 1.5          # 이보다 짧은 구간은 등급을 매기지 않는다

# v6b 5차 무개입 낙하 실측 (비교 기준선)
V6B_BASELINE = {"slope": 6.0, "lean_end": 19.0, "standing_s": 2.0}


def grav_to_angles(quat_wxyz):
    """quat(wxyz) → (전방기울임°, 좌우기울임°, 총기울기°).

    브리지와 동일한 projected-gravity 식을 쓴다 (rl_bridge_node.tick).
    gx>0 = 앞으로 기울어짐 (y축 +a 회전에서 gx=sin(a) — 실물 IMU 규약과 일치).
    """
    w, x, y, z = quat_wxyz
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    fwd = math.degrees(math.atan2(gx, -gz))
    lat = math.degrees(math.atan2(gy, -gz))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
    return fwd, lat, tilt


def linfit(xs, ys):
    """최소제곱 (기울기, 절편, R²). 표본 부족·분산 0이면 (0,평균,0)."""
    n = len(xs)
    if n < 3:
        return 0.0, (sum(ys) / n if n else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    icept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy < 1e-12:
        return slope, icept, 0.0
    ss_res = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
    return slope, icept, max(0.0, 1.0 - ss_res / syy)


def monotonic_frac(xs, ys, bin_s=0.2):
    """0.2s 빈으로 다운샘플한 뒤 '증가한 구간' 비율.

    20ms 원틱으로 부호를 세면 센서 노이즈가 0.5로 희석된다. v6b의 단조
    누적은 초 단위 추세였으므로 빈 평균 후 차분해야 실체가 보인다.
    """
    if not xs:
        return 0.0, 0
    bins = {}
    for t, v in zip(xs, ys):
        k = int(t / bin_s)
        bins.setdefault(k, []).append(v)
    keys = sorted(bins)
    means = [sum(bins[k]) / len(bins[k]) for k in keys]
    if len(means) < 2:
        return 0.0, len(means)
    ups = sum(1 for a, b in zip(means, means[1:]) if b > a)
    return ups / (len(means) - 1), len(means)


def sparkline(vals, width=60):
    """시계열을 한 줄로. 축(0°)은 '-' 로 표시."""
    if not vals:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    samp = [vals[i] for i in range(0, len(vals), step)][:width]
    lo, hi = min(samp), max(samp)
    if hi - lo < 1e-9:
        return "─" * len(samp)
    out = []
    for v in samp:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def load(path):
    """jsonl → (틱 레코드 리스트, 이벤트 리스트). 손상 줄은 건너뛴다."""
    ticks, events, bad = [], [], 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if "event" in o:
                events.append(o)
            elif "quat_wxyz" in o:
                ticks.append(o)
    return ticks, events, bad


def analyze(path):
    ticks, events, bad = load(path)
    if not ticks:
        raise SystemExit(f"[실패] 틱 레코드가 없습니다: {path}")

    # t0 = 정책이 실제로 모터를 지휘하기 시작한 첫 틱 (live+armed).
    # 브리지 기동 시각이 아니다 — ARM 전 관측만 흐르는 구간이 길다.
    armed = [t for t in ticks if t.get("live") and t.get("armed")]
    if not armed:
        raise SystemExit(
            f"[실패] live+armed 구간이 없습니다 (정책 미투입): {path}\n"
            f"       총 틱 {len(ticks)}개 — ARM_ALL 이 안 됐거나 브리지가 "
            f"--live 없이 떴을 수 있습니다.")
    t0 = armed[0]["timestamp"]

    rows = []
    for t in armed:
        rel = t["timestamp"] - t0
        fwd, lat, tilt = grav_to_angles(t["quat_wxyz"])
        fn = t.get("foot_n") or [0.0, 0.0]
        load_pct = 100.0 * (float(fn[0]) + float(fn[1])) / BODY_WEIGHT_N
        rows.append({"t": rel, "fwd": fwd, "lat": lat, "tilt": tilt,
                     "load": load_pct, "contact": t.get("contact"),
                     "alpha": t.get("blend_alpha", 1.0)})

    dur = rows[-1]["t"]

    # 기준 자세 = 블렌딩 완료 직후 0.3s 평균. 착지 기하 탓에 시작부터 기울어
    # 있는 경우가 흔한데(0803 실기: 시작 −16°), 절대각으로 자립을 판정하면
    # 발력 92%로 멀쩡히 서 있는 시행이 "자립 0초"가 된다. 자립 여부는
    # **발력**이 1순위이고, 각도는 **시작 대비 변화량**으로만 본다.
    settled = [r for r in rows if r["alpha"] >= 0.99]
    if settled:
        base_win = [r["fwd"] for r in settled if r["t"] <= settled[0]["t"] + 0.3]
        fwd0 = sum(base_win) / len(base_win)
    else:
        fwd0 = rows[0]["fwd"]

    # 실효 자립 = 발력을 잃거나(줄로 넘어감) 시작 대비 크게 기울 때까지.
    # 블렌딩 구간(alpha<1)은 정책 권한이 부분적이라 판정에서 제외한다.
    # **조건은 반드시 지속되어야 한다** — 10Hz 발압력의 1펄스 결손 하나가
    # 스텝테스트를 통째로 무효화한 전례가 있다. 순간값으로 끊으면 15초를
    # 버틴 시행이 0.4초로 찍힌다.
    # 핵심: **낙상은 되돌아오지 않는다.** 회복한 딥은 낙상이 아니다.
    # (0803 실기: 인수 0.8초에 발력이 17%로 꺼졌다가 회복해 15초를 섰다.
    #  순간값·지속조건 어느 쪽으로 봐도 그 시행은 "0.4초 자립"으로 오독된다.)
    def terminal_fall(pred, hold_s):
        """pred가 로그 끝까지 회복 없이 이어진 구간의 시작. 없으면 None."""
        t_bad = None
        for r in settled:
            if pred(r):
                if t_bad is None:
                    t_bad = r["t"]
            else:
                t_bad = None          # 회복했다 = 딥이었다
        if t_bad is None or settled[-1]["t"] - t_bad < hold_s:
            return None               # 끝자락 순간값은 낙상으로 세지 않는다
        return t_bad

    standing_s = dur
    stand_reason = "끝까지 유지"
    t_load = terminal_fall(lambda r: r["load"] < LOAD_LOST, 0.30)
    t_lean = terminal_fall(
        lambda r: abs(r["fwd"] - fwd0) > LEAN_FALL_DELTA, 0.20)
    if t_load is not None and (t_lean is None or t_load <= t_lean):
        standing_s, stand_reason = t_load, f"발력 {LOAD_LOST:.0f}% 미만 (미회복)"
    elif t_lean is not None:
        standing_s, stand_reason = (
            t_lean, f"시작 대비 {LEAN_FALL_DELTA:.0f}° 초과 기울임 (미회복)")

    # 회복한 딥은 낙상이 아니지만 그 자체로 진단 가치가 있다 (인수 전이 손실).
    dips = [r for r in settled
            if r["load"] < LOAD_LOST and r["t"] < standing_s]

    early = [r for r in rows if r["t"] <= EARLY_WINDOW_S]
    slope, _, r2 = linfit([r["t"] for r in early], [r["fwd"] for r in early])
    mono, nbins = monotonic_frac([r["t"] for r in early],
                                 [r["fwd"] for r in early])

    # 발력은 **자립 구간에서만** 평균낸다. 낙상 후 구간을 섞으면 멀쩡히
    # 서 있다가 넘어진 시행이 "처음부터 줄에 매달려 있었다"로 오독된다.
    loads = [r["load"] for r in rows
             if r["alpha"] >= 0.99 and r["t"] <= standing_s]
    fwds = [r["fwd"] for r in rows]

    return {
        "path": path, "bad_lines": bad, "n_ticks": len(rows),
        "ckpt": armed[0].get("checkpoint", "?"),
        "dur": dur, "hz": (len(rows) - 1) / dur if dur > 0 else 0.0,
        "standing_s": standing_s, "stand_reason": stand_reason,
        "settled_s": (settled[-1]["t"] - settled[0]["t"]) if settled else 0.0,
        "dip_n": len(dips), "dip_t": dips[0]["t"] if dips else None,
        "dip_min": min((r["load"] for r in dips), default=None),
        "guard_events": [e.get("event") for e in events
                         if e.get("event") in ("osc_guard", "osc_guard_tgt",
                                               "lift_guard")],
        "slope": slope, "slope_r2": r2, "mono": mono, "nbins": nbins,
        "fwd0": fwd0, "fwd_start": rows[0]["fwd"], "fwd_end": rows[-1]["fwd"],
        "fwd_max": max(fwds), "fwd_min": min(fwds),
        "lat_absmax": max(abs(r["lat"]) for r in rows),
        "load_mean": (sum(loads) / len(loads)) if loads else 0.0,
        "load_min": min(loads) if loads else 0.0,
        "load_max": max(loads) if loads else 0.0,
        "events": [(e["timestamp"] - t0, e.get("event"), e.get("reason", ""))
                   for e in events if e["timestamp"] >= t0],
        "rows": rows,
    }


def verdict(a):
    """6차 핵심 질문에 대한 등급 + 가드 단서 (가드로 끝난 시행은 무조건 단서)."""
    grade, headline, action = _grade(a)
    if grade in ("A", "B") and a["guard_events"]:
        action += (f"  ※ 단, 이 시행은 가드"
                   f"({'/'.join(sorted(set(a['guard_events'])))})로 끝났습니다 — "
                   f"발진 원인을 규명하기 전에는 다음 단계로 넘어가지 마세요.")
    return grade, headline, action


def _grade(a):
    """등급 산출 본체."""
    slope, mono, stand = a["slope"], a["mono"], a["standing_s"]
    # 등급을 매길 자격이 있는 시행인지 먼저 본다. 0.8초짜리 가드 종료 로그에
    # 3빈 회귀로 "전방 단조 밀기"를 선언하면 그건 판정이 아니라 착각이다.
    if a["settled_s"] < MIN_GRADE_S:
        return ("—", f"판정 불가 — 블렌딩 후 유효 구간이 {a['settled_s']:.1f}s "
                f"뿐입니다 (최소 {MIN_GRADE_S:.1f}s)",
                "시행이 너무 일찍 끝났습니다. 무엇이 끝냈는지(가드·착지·"
                "인장)를 해결하고 다시 받으세요.")
    if a["guard_events"] and stand < 2.5:
        return ("—", f"판정 불가 — 가드({'/'.join(sorted(set(a['guard_events'])))})"
                f"가 {stand:.1f}s에 시행을 끊었습니다",
                "균형이 아니라 가드가 결말을 정했습니다. 가드 트립 원인을 "
                "먼저 규명해야 v7 판정이 성립합니다.")
    # 전방 단조 누적 = v6b 병증. 단조비율만 쓰면 "1초 평탄 후 램프" 같은
    # 실제 파형에서 앞 빈들이 비증가로 세어져 놓친다 — 직선성(R²)을 병기.
    pushing = slope > 2.0 and (mono >= 0.65 or a["slope_r2"] >= 0.7)
    if pushing and stand < 4.0:
        return ("C", "무변화 — v6b와 같은 전방 단조 밀기가 재현됐습니다",
                "payload 위치 가정은 원인이 아니었습니다. 다음 용의자: "
                "관측 지연(+1틱)·컴플라이언스(로터 사각)·발압력 지연.")
    if slope < -2.0 and mono <= 0.3 and stand < 4.0:
        return ("D", "역전 — 이번엔 뒤로 단조 누적됩니다 (v7 과교정)",
                "com_range를 v6b와 v7 사이(x 약 −0.05~0.0)로 좁혀 재베이크.")
    if stand >= 8.0 and abs(a["fwd_end"]) < 5.0:
        return ("A", "완전 성공 — 8초 이상 자립, 기울임 5° 이내 유지",
                "march 사다리(5초) 진입 가능합니다.")
    if stand >= 4.0 or (not pushing and stand >= 2.5):
        return ("B", "개선 확인 — 단조 밀기는 사라졌으나 자립이 짧습니다",
                "발진/횡동요/발력 손실 중 무엇이 끝냈는지 이벤트·발력을 보세요.")
    return ("C", "판정 애매 — 자립이 짧은데 전방 단조 패턴은 뚜렷하지 않습니다",
            "블렌딩 구간·가드 트립·착지 기하(초기 기울임)를 먼저 의심하세요.")


def report(a):
    print("=" * 68)
    print(f"무개입 낙하 판정 — {os.path.basename(a['path'])}")
    print("=" * 68)
    print(f"정책: {a['ckpt']}   정책구동 틱 {a['n_ticks']}개 / "
          f"{a['dur']:.1f}s ({a['hz']:.1f}Hz)"
          + (f"   손상줄 {a['bad_lines']}" if a["bad_lines"] else ""))
    print()
    print(f"■ 실효 자립 시간: {a['standing_s']:.1f}s  ({a['stand_reason']})")
    print(f"■ 발력 총합: 평균 {a['load_mean']:.0f}% / 최소 {a['load_min']:.0f}%"
          f" / 최대 {a['load_max']:.0f}%  (체중 {BODY_WEIGHT_N:.0f}N 기준)")
    print("   (자립 구간 기준 — 낙상 후 구간 제외)")
    if a["dip_t"] is not None:
        print(f"   ※ 회복된 발력 딥: {a['dip_t']:.1f}s에 최저 {a['dip_min']:.0f}% "
              f"({a['dip_n']}틱) — 낙상은 아니나 인수 전이 손실 신호")
    if a["load_mean"] >= LOAD_STANDING:
        print("   → 정책이 하중을 받고 있었습니다 (줄에 매달린 상태 아님).")
    elif a["load_mean"] >= LOAD_LOST:
        print("   → 대부분 받고 있으나 일부를 줄이 분담했습니다. "
              "줄 인장을 더 풀 여지가 있는지 확인하세요.")
    else:
        print("   → 하중이 줄/사람에게 상당 부분 넘어가 있었습니다. "
              "자립 판정으로 쓰기 전에 착지·인장 상태를 확인하세요.")
    print()
    print(f"■ 전방 기울임: 시작 {a['fwd_start']:+.1f}° → 종료 "
          f"{a['fwd_end']:+.1f}°  (최대 {a['fwd_max']:+.1f}° / 최소 "
          f"{a['fwd_min']:+.1f}°)   좌우 최대 |{a['lat_absmax']:.1f}|°")
    print(f"   기준 자세(블렌딩 후) {a['fwd0']:+.1f}° → 순변화 "
          f"{a['fwd_end'] - a['fwd0']:+.1f}°"
          + ("   ※ 시작부터 크게 기울어 있었습니다 — 착지 기하를 의심하세요"
             if abs(a["fwd0"]) > 8.0 else ""))
    print(f"■ 초기 {EARLY_WINDOW_S:.0f}s 추세: {a['slope']:+.2f}°/s "
          f"(R²={a['slope_r2']:.2f}), 단조증가 {a['mono']*100:.0f}% "
          f"({a['nbins']}빈)")
    print(f"   v6b 기준선: 약 {V6B_BASELINE['slope']:+.1f}°/s, "
          f"{V6B_BASELINE['lean_end']:.0f}°까지 누적, 자립 "
          f"~{V6B_BASELINE['standing_s']:.0f}s")
    print()
    print("  전방기울임 시계열 (좌=투입 직후, 우=종료)")
    print(f"  {sparkline([r['fwd'] for r in a['rows']])}")
    print(f"  {sparkline([r['load'] for r in a['rows']])}  ← 발력 총합")
    print()
    if a["events"]:
        print("■ 이벤트")
        for t, ev, why in a["events"]:
            print(f"   {t:6.2f}s  {ev}" + (f"  ({why})" if why else ""))
    else:
        print("■ 이벤트: 없음 (가드 무트립·STOP_ALL 없음)")
    print()
    grade, headline, action = verdict(a)
    print(f"■ 판정 [{grade}급] {headline}")
    print(f"   다음: {action}")
    print("=" * 68)


def compare(a, b):
    print()
    print("=" * 68)
    print("비교 — 이번(좌) vs 기준(우)")
    print("=" * 68)
    rows = [
        ("정책", a["ckpt"], b["ckpt"]),
        ("실효 자립", f"{a['standing_s']:.1f}s", f"{b['standing_s']:.1f}s"),
        ("초기 추세", f"{a['slope']:+.2f}°/s", f"{b['slope']:+.2f}°/s"),
        ("단조증가", f"{a['mono']*100:.0f}%", f"{b['mono']*100:.0f}%"),
        ("종료 기울임", f"{a['fwd_end']:+.1f}°", f"{b['fwd_end']:+.1f}°"),
        ("발력 평균", f"{a['load_mean']:.0f}%", f"{b['load_mean']:.0f}%"),
    ]
    for label, x, y in rows:
        print(f"  {label:<10} {x:>16}   |  {y:>16}")
    d = a["slope"] - b["slope"]
    print()
    print(f"  전방 밀기 변화: {d:+.2f}°/s "
          + ("(개선 — 전방 누적이 줄었습니다)" if d < -1.0
             else "(악화 — 전방 누적이 늘었습니다)" if d > 1.0
             else "(유의차 없음)"))
    print("=" * 68)


def main():
    p = argparse.ArgumentParser(description="무개입 낙하 블랙박스 판정기")
    p.add_argument("jsonl", help="rl_bridge 블랙박스 (~/rl_bridge_bb/*.jsonl)")
    p.add_argument("--compare", help="비교 대상 블랙박스 (예: v6b 5차)")
    p.add_argument("--csv", help="틱 시계열을 CSV로 저장")
    args = p.parse_args()

    a = analyze(args.jsonl)
    report(a)
    if args.compare:
        compare(a, analyze(args.compare))
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("t_s,fwd_deg,lat_deg,tilt_deg,load_pct,blend_alpha\n")
            for r in a["rows"]:
                f.write(f"{r['t']:.3f},{r['fwd']:.3f},{r['lat']:.3f},"
                        f"{r['tilt']:.3f},{r['load']:.2f},{r['alpha']:.3f}\n")
        print(f"CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
