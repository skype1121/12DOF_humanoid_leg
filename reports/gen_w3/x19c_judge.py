#!/usr/bin/env python3
"""X19c 판정(TRANSITION_W3KE.md §5.22f): 서기 단독 키의 패드 신호 끊김 내성 — 인계 뒤 5 s·30 s·35 s 낙상, 대조 대비 Fisher."""
import csv, math, os
LP = "/home/ryu/humanoid_leg_test1/log_picture"
def load(tags):
    rows = []
    for t in tags:
        f = f"{LP}/ground_recovery_{t}_envs.csv"
        if not os.path.exists(f): return None
        rows += [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["pre_pass"] == "1"]
    return rows
def falls(rows, t):
    return sum(1 for r in rows if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan") and float(r["t_fall_rel"]) <= t)
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
ARMS = (("대조(접촉 비트 정상)", [f"rg_f1008_0611_s_nd_s{s}" for s in (1, 2, 3)]), ("SL 왼발 비트 0 고착", [f"x19c_SL_s{s}" for s in (1, 2, 3)]), ("SB 양발 비트 0 고착", [f"x19c_SB_s{s}" for s in (1, 2, 3)]))
ref = load(ARMS[0][1]); rf, rn = falls(ref, 35.0), len(ref)
print("| 팔 | 사전검사 통과 env | 5 s 낙상 | 30 s 낙상 | 35 s 낙상 | 대조 대비 · Fisher p | 끝까지 선 개체의 발 들림(중앙·p90) |"); print("|---|---|---|---|---|---|---|")
for nm, tags in ARMS:
    r = load(tags)
    if r is None: print(f"| {nm} | (없음) | | | | | |"); continue
    n = len(r); f5, f30, f35 = falls(r, 5.0), falls(r, 30.0), falls(r, 35.0)
    lif = sorted(float(x["lifts_total"]) for x in r if x["fell"] != "1" and x["lifts_total"] not in ("", "nan"))
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
    d = "—" if nm.startswith("대조") else f"{100*f35/n - 100*rf/rn:+.2f} %p · {fisher(f35, n - f35, rf, rn - rf):.3g}"
    print(f"| {nm} | {n} | {f5} ({100*f5/n:.1f} %) | {f30} ({100*f30/n:.1f} %) | {f35} ({100*f35/n:.2f} %) | {d} | {q(lif, .5):.0f} · {q(lif, .9):.0f} |")
