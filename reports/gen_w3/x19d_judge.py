#!/usr/bin/env python3
"""X19d 판정(TRANSITION_W3KE.md §5.22h): 회복 키 KE 의 패드 신호 끊김 내성(무외란 60 s) — 낙상·회복 모드 켜짐·모드 시간, 대조 대비 Fisher."""
import math, sys
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/reports/gen_w3")
import keq_judge as K
import numpy as np
SEEDS = (21, 22, 23)
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
def fall_times(arm):
    out = []
    for s in SEEDS:
        A = K.load(arm, "nd60", 2, s)
        if A is None: continue
        out += [round(float(A["tf"][e]), 1) for e in range(A["pre"].shape[0]) if A["pre"][e] and A["fell"][e]]
    return sorted(out)
ref = K.nd_stats("KE", 2, SEEDS)
print("| 팔 | 사전검사 통과 env | 60 s 낙상 | 대조 대비 · Fisher p | 회복 모드 켜짐(회/env·분) | 모드 시간 | 낙상 시각(인계 뒤 s) |"); print("|---|---|---|---|---|---|---|")
for nm, arm in (("대조(접촉 비트 정상)", "KE"), ("KEL 왼발 비트 0 고착", "KEL"), ("KEB 양발 비트 0 고착", "KEB")):
    r = K.nd_stats(arm, 2, SEEDS)
    if r["n"] == 0 or r["miss"]: print(f"| {nm} | (없음 {r['miss']}) | | | | | |"); continue
    d = "—" if arm == "KE" else f"{100*r['fell']/r['n'] - 100*ref['fell']/ref['n']:+.2f} %p · {fisher(r['fell'], r['n'] - r['fell'], ref['fell'], ref['n'] - ref['fell']):.3g}"
    print(f"| {nm} | {r['n']} | {r['fell']} ({100*r['fell']/r['n']:.2f} %) | {d} | {r['rate']:.3f} | {100*r['frac']:.2f} % | {fall_times(arm)} |")
