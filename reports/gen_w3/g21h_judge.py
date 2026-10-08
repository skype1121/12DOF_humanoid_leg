#!/usr/bin/env python3
"""G21h 후보 판정(reports/gen_w3/AIR_START_1008.md §8) — 사용: g21h_judge.py <체크포인트 번호> [A|B|all] [seed 목록(A 팔) 기본 1,2]

A = 공중 시작(§1 의 P1–P3, 대조는 기존 정책의 AIR1 대조와 후보 자신의 대조 둘 다) + 견고성(공중 20 s·방출 1 s)
B = 지면 비퇴보: B1 서기 35 s · B2 KE 회복 · B3 걷기 버튼(X14) · B4 혼합 왕복(X7). 기준선 = 지금 배포 구성(G21a)의 같은 하네스·같은 seed 실행."""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/reports/gen_w3")
LP = "/home/ryu/humanoid_leg_test1/log_picture"
N = sys.argv[1]; WHAT = sys.argv[2] if len(sys.argv) > 2 else "all"
SEEDS = tuple(int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("1", "2")))
ARMS = {"C": (-1.5, 2.5, 35.5), "A2": (2.0, 6.0, 39.0), "A2i": (2.0, 6.0, 39.0), "A6": (6.0, 10.0, 43.0), "A6i": (6.0, 10.0, 43.0), "A20i": (20.0, 24.0, 57.0), "A2ir1": (2.0, 3.0, 36.0)}
LABEL = {"C": "홀드로 착지 뒤 인계", "A2": "공중 2 s · 감지기 그대로", "A2i": "공중 2 s · 접지 전 감지기 정지", "A6": "공중 6 s · 감지기 그대로", "A6i": "공중 6 s · 접지 전 감지기 정지",
         "A20i": "공중 20 s · 감지기 정지", "A2ir1": "공중 2 s · 감지기 정지 · 방출 1 s"}
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
def q90(v):
    v = np.asarray([x for x in v if x == x], float)
    return float(np.percentile(v, 90)) if len(v) else float("nan")
pc = lambda a, n: f"{a} / {n} ({100*a/n:.1f} %)" if n else "—"
def load_air(prefix, arm, seeds):
    rel0, zero, btn = ARMS[arm]; R = dict(n=0, f_pre=0, f_all30=0, alive_zero=0, f_zero5=0, alive_btn=0, acc=0, miss=[], air=[], exc=[], spd=[], air_on=0, air_n=0, td_rec=0, td_n=0, wmax=[])
    for s in seeds:
        tag = f"{prefix}_{arm}_s{s}"; f = f"{LP}/ground_recovery_{tag}_envs.csv"; k = f"{LP}/kwloop_{tag}.npz"; t = f"{LP}/transall_{tag}.npz"
        if not (os.path.exists(f) and os.path.exists(k) and os.path.exists(t)):
            R["miss"].append(s); continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        tf = np.array([float(r["t_fall_rel"]) if (r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan")) else np.inf for r in rows]); n = len(rows); R["n"] += n
        R["f_pre"] += int((tf < 0).sum()); R["f_all30"] += int((tf <= zero + 30.0).sum())
        az = tf > zero; R["alive_zero"] += int(az.sum()); R["f_zero5"] += int((az & (tf <= zero + 5.0)).sum())
        ev = json.loads(str(np.load(k, allow_pickle=True)["events"])); ab = tf > btn
        R["alive_btn"] += int(ab.sum()); R["acc"] += sum(1 for e in range(n) if ab[e] and any(x[1] == "enable" and x[0] >= btn - 0.05 for x in ev[e]))
        tr = np.load(t, allow_pickle=True); cont = tr["cont"] > 0.5; mode = tr["mode"].astype(bool); q = tr["q"]; w = tr["w"]; alive = tr["alive"].astype(bool)
        for e in range(q.shape[1]):
            anyc = cont[:, e].any(1); k1 = int(np.argmax(anyc)) if anyc.any() else q.shape[0]; k1 = min(k1, int(alive[:, e].sum()))
            if k1 >= 10:
                R["air_n"] += 1; R["air"].append(k1 * 0.02); R["air_on"] += bool(mode[:k1, e].any())
                R["exc"].append(float(np.degrees(np.ptp(q[:k1, e], axis=0).max()))); R["spd"].append(float(np.abs(np.diff(q[:k1, e], axis=0)).max() * 50.0)); R["wmax"].append(float(w[:k1, e].max()))
            both = cont[:, e].all(1)
            if both.any() and alive[int(np.argmax(both)), e]:
                R["td_n"] += 1; R["td_rec"] += bool(mode[int(np.argmax(both)), e])
    return R
def part_a():
    base = load_air("tr_air1", "C", (1, 2)); P = f"tr_h{N}"
    D = {a: load_air(P, a, (1,) if a in ("A20i", "A2ir1") else SEEDS) for a in ARMS}
    c = D["C"]
    print(f"## A. 공중 시작 — G21h-{N} (seed {','.join(map(str, SEEDS))}; 견고성 팔은 seed 1)")
    print("| 팔 | env | 처음부터 지지 0 뒤 30 s 까지 낙상 | 기존 정책 대조(0/512) 대비 p | 후보 대조 대비 | 지지 0 뒤 5 s 낙상 | 걷기 버튼 수락 |"); print("|---|---|---|---|---|---|---|")
    print(f"| (기준) 기존 정책 G21a · 홀드로 착지 뒤 인계 | {base['n']} | {pc(base['f_all30'], base['n'])} | — | — | {pc(base['f_zero5'], base['alive_zero'])} | {pc(base['acc'], base['alive_btn'])} |")
    for a, r in D.items():
        if r["n"] == 0: print(f"| {a} {LABEL[a]} | (없음 {r['miss']}) | | | | | |"); continue
        pb = fisher(r["f_all30"], r["n"] - r["f_all30"], base["f_all30"], base["n"] - base["f_all30"]) if base["n"] else float("nan")
        dc = "—" if a == "C" or c["n"] == 0 else f"{100*r['f_all30']/r['n'] - 100*c['f_all30']/c['n']:+.1f} %p · p {fisher(r['f_all30'], r['n'] - r['f_all30'], c['f_all30'], c['n'] - c['f_all30']):.3g}"
        print(f"| {a} {LABEL[a]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_all30'], r['n'])} | {pb:.3g} | {dc} | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
    print(); print("| 팔 | 공중 구간(추적 env) | 공중 시간 중앙 | 공중에서 회복 모드 켜짐 | 공중 관절 이동폭 p90 | 공중 관절 속도 최대 p90 | 공중 몸통 각속도 최대 p90 | 양발 착지 순간 회복 모드 |"); print("|---|---|---|---|---|---|---|---|")
    for a, r in D.items():
        if r["air_n"] == 0: print(f"| {a} | 0 | — | — | — | — | — | {pc(r['td_rec'], r['td_n'])} |"); continue
        print(f"| {a} | {r['air_n']} | {np.median(r['air']):.1f} s | {pc(r['air_on'], r['air_n'])} | {q90(r['exc']):.1f}° | {q90(r['spd']):.2f} rad/s | {q90(r['wmax']):.2f} rad/s | {pc(r['td_rec'], r['td_n'])} |")
    print(); allok = True
    for a in ("A2i", "A6i", "A20i", "A2ir1", "A2", "A6"):
        r = D[a]
        if r["n"] == 0 or r["miss"]: print(f"- {a}: 실행 미완 {r['miss']}"); allok = allok and a in ("A2", "A6"); continue
        fr = 100 * r["f_all30"] / r["n"]; pb = fisher(r["f_all30"], r["n"] - r["f_all30"], base["f_all30"], base["n"] - base["f_all30"])
        okc = c["n"] > 0 and not c["miss"]; fc = 100 * c["f_all30"] / c["n"] if c["n"] else float("nan"); pcx = fisher(r["f_all30"], r["n"] - r["f_all30"], c["f_all30"], c["n"] - c["f_all30"]) if c["n"] else float("nan")
        p1 = (fr <= 100 * base["f_all30"] / base["n"] + 3.0 and pb >= 0.05) and (okc and fr <= fc + 3.0 and pcx >= 0.05)
        p2 = r["air_n"] > 0 and (100 * r["air_on"] / r["air_n"] <= 5.0) and q90(r["exc"]) <= 20.0 and q90(r["spd"]) <= 3.0
        z5 = 100 * r["f_zero5"] / max(r["alive_zero"], 1); bz5 = 100 * base["f_zero5"] / max(base["alive_zero"], 1); acc = 100 * r["acc"] / max(r["alive_btn"], 1)
        p3 = (z5 <= bz5 + 3.0) and acc >= 95.0
        need = a in ("A2i", "A6i", "A20i", "A2ir1")
        res = (p1 and p2 and p3) if a in ("A2i", "A6i") else p1
        if need: allok = allok and res
        print(f"- {a}{'' if need else ' (참고 — 승격 조건 아님)'}: P1 {'충족' if p1 else '미달'}(낙상 {fr:.1f} % · 기존 대조 대비 p {pb:.3g} · 후보 대조 {fc:.1f} % 대비 p {pcx:.3g})"
              + (f" · P2 {'충족' if p2 else '미달'}(회복 모드 {100*r['air_on']/max(r['air_n'],1):.1f} % · 이동폭 {q90(r['exc']):.1f}° · 속도 {q90(r['spd']):.2f}) · P3 {'충족' if p3 else '미달'}(지지 0 뒤 5 s {z5:.1f} % · 수락 {acc:.1f} %)" if a in ("A2i", "A6i", "A2", "A6") else ""))
    print(f"- **A 묶음: {'충족' if allok else '미달 또는 미완'}**")
    return allok
def stand35(tags):
    rows = []
    for t in tags:
        f = f"{LP}/ground_recovery_{t}_envs.csv"
        if not os.path.exists(f): return None
        rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["pre_pass"] == "1"]
    fell = sum(1 for r in rows if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan") and float(r["t_fall_rel"]) <= 35.0)
    quiet = sum(1 for r in rows if r["fell"] != "1" and r["final_quiet"] == "1")
    return dict(n=len(rows), fell=fell, quiet=quiet)
def part_b():
    ok = True; print(f"## B. 지면 비퇴보 — G21h-{N} 대 기준선(G21a)")
    # B1
    b0 = stand35([f"rg_f1008_0611_s_nd_s{s}" for s in (1, 2, 3)]); b1 = stand35([f"rg_h{N}_s_nd_s{s}" for s in (1, 2, 3)])
    if b1 is None: print("- B1 서기 35 s: 실행 미완"); ok = False
    else:
        d = 100 * b1["fell"] / b1["n"] - 100 * b0["fell"] / b0["n"]; p = fisher(b1["fell"], b1["n"] - b1["fell"], b0["fell"], b0["n"] - b0["fell"]); r = d <= 3.0 and p >= 0.05; ok &= r
        print(f"- B1 서기 단독 35 s: 낙상 {pc(b1['fell'], b1['n'])} 대 기준선 {pc(b0['fell'], b0['n'])} ({d:+.1f} %p, p {p:.3g}) · 마지막 1 s 정지 {pc(b1['quiet'], b1['n'] - b1['fell'])} 대 {pc(b0['quiet'], b0['n'] - b0['fell'])} → {'충족' if r else '미달'}")
    # B2
    import keq_judge as K
    def ke(arm, cond):
        n = fell = ever = 0; miss = []
        for s in (21, 22, 23):
            A = K.load(arm, cond, 2, s)
            if A is None: miss.append(s); continue
            m = A["mode"] & A["alive"]
            for e in range(A["pre"].shape[0]):
                if A["pre"][e]: n += 1; fell += bool(A["fell"][e]); ever += bool(m[:, e].any())
        return dict(n=n, fell=fell, ever=ever, miss=miss)
    for cond, lab in (("nd60", "무외란 60 s"), ("b18", "뒤 밀기 18 N"), ("sf15", "앞 램프 15 N")):
        a0 = ke("KE", cond); a1 = ke(f"KEh{N}", cond)
        if a1["n"] == 0 or a1["miss"]: print(f"- B2 KE {lab}: 실행 미완 {a1['miss']}"); ok = False; continue
        d = 100 * a1["fell"] / a1["n"] - 100 * a0["fell"] / a0["n"]; p = fisher(a1["fell"], a1["n"] - a1["fell"], a0["fell"], a0["n"] - a0["fell"]); r = d <= 3.0 and p >= 0.05
        ex = ""
        if cond == "nd60":
            de = 100 * a1["ever"] / a1["n"] - 100 * a0["ever"] / a0["n"]; r = r and de <= 3.0; ex = f" · 회복 모드 진입 {pc(a1['ever'], a1['n'])} 대 {pc(a0['ever'], a0['n'])} ({de:+.1f} %p)"
        ok &= r
        print(f"- B2 KE {lab}: 낙상 {pc(a1['fell'], a1['n'])} 대 기준선 {pc(a0['fell'], a0['n'])} ({d:+.1f} %p, p {p:.3g}){ex} → {'충족' if r else '미달'}")
    # B3
    import x19_judge as J
    c0 = J.judge([f"tr_x14_d2_s{s}" for s in (1, 2)]); c1 = J.judge([f"tr_h{N}_x14_d2_s{s}" for s in (1, 2)])
    if c1 is None: print("- B3 걷기 버튼(X14): 실행 미완"); ok = False
    else:
        f1, n1 = c1["fall"]; f0, n0 = c0["fall"]; a1, an1 = c1["acc"]; smin = min((100 * c1["rows"][b]["sign"][0] / c1["rows"][b]["sign"][1]) for b in J.ACTIVE if c1["rows"].get(b, {}).get("sign"))
        d = 100 * f1 / n1 - 100 * f0 / n0; p = J.fisher(f1, n1 - f1, f0, n0 - f0); r = d <= 3.0 and p >= 0.05 and 100 * a1 / an1 >= 99.0 and smin >= 95.0; ok &= r
        print(f"- B3 걷기 버튼 대본 58 s: 낙상 {pc(f1, n1)} 대 기준선 {pc(f0, n0)} ({d:+.1f} %p, p {p:.3g}) · 수락 {100*a1/an1:.2f} % · 활성 버튼 방향 최저 {smin:.1f} % → {'충족' if r else '미달'}")
    # B4
    def x7(tags):
        n = f = 0
        for t in tags:
            fp = f"{LP}/ground_recovery_{t}_envs.csv"
            if not os.path.exists(fp): return None
            rows = [r for r in csv.DictReader(open(fp, encoding="utf-8")) if r["pre_pass"] == "1"]; n += len(rows); f += sum(1 for r in rows if r["fell"] == "1")
        return n, f
    d0 = x7([f"tr_base_x7_d2_s{s}" for s in (1, 2)]); d1 = x7([f"tr_h{N}_x7_d2_s{s}" for s in (1, 2)])
    if d1 is None: print("- B4 혼합 왕복(X7): 실행 미완"); ok = False
    else:
        d = 100 * d1[1] / d1[0] - 100 * d0[1] / d0[0]; p = fisher(d1[1], d1[0] - d1[1], d0[1], d0[0] - d0[1]); r = d <= 3.0 and p >= 0.05; ok &= r
        print(f"- B4 혼합 왕복 48 s: 낙상 {pc(d1[1], d1[0])} 대 기준선 {pc(d0[1], d0[0])} ({d:+.1f} %p, p {p:.3g}) → {'충족' if r else '미달'}")
    print(f"- **B 묶음: {'충족' if ok else '미달 또는 미완'}**")
    return ok
ra = part_a() if WHAT in ("A", "all") else None
if WHAT == "all": print()
rb = part_b() if WHAT in ("B", "all") else None
