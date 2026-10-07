#!/usr/bin/env python3
"""X19e 판정(TRANSITION_W3KE.md §5.22j): 감지기 기준 자세 갱신의 접촉 요구를 '한 발 이상'으로 바꾼 시뮬 전용 수정안 — E1(왼발 0 헛켜짐)·E2(정상 무외란)·E3(밀기)."""
import math, sys
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/reports/gen_w3")
import keq_judge as K
import numpy as np
SEEDS = (21, 22, 23)
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
def stats(arm, cond):
    n = fell = ever = end = 0; mt = at = 0; miss = []
    for s in SEEDS:
        A = K.load(arm, cond, 2, s)
        if A is None: miss.append(s); continue
        m = A["mode"] & A["alive"]
        for e in range(A["pre"].shape[0]):
            if not A["pre"][e]: continue
            n += 1; fell += bool(A["fell"][e]); ever += bool(m[:, e].any()); end += bool(m[-1, e]); mt += int(m[:, e].sum()); at += int(A["alive"][:, e].sum())
    return dict(n=n, fell=fell, ever=ever, end=end, frac=(mt / at if at else float("nan")), miss=miss)
pc = lambda a, n: f"{a} / {n} ({100*a/n:.1f} %)" if n else "(없음)"
print("| 조건 | 감지기 | 통과 env | 낙상 | 회복 모드에 들어감 | 끝에 회복 모드 | 모드 시간 |"); print("|---|---|---|---|---|---|---|")
R = {}
for cond, lab, arms in (("nd60", "무외란 60 s · 정상 접촉", (("종전", "KE"), ("수정안", "KEr"))), ("nd60", "무외란 60 s · 왼발 비트 0", (("종전", "KEL"), ("수정안", "KELr"))),
                        ("b18", "뒤 밀기 18 N", (("종전", "KE"), ("수정안", "KEr"))), ("sf15", "앞 램프 15 N", (("종전", "KE"), ("수정안", "KEr")))):
    for dn, arm in arms:
        r = stats(arm, cond); R[(arm, cond)] = r
        if r["n"] == 0 or r["miss"]: print(f"| {lab} | {dn} | (없음 {r['miss']}) | | | | |"); continue
        print(f"| {lab} | {dn} | {r['n']} | {pc(r['fell'], r['n'])} | {pc(r['ever'], r['n'])} | {pc(r['end'], r['n'])} | {100*r['frac']:.2f} % |")
ok = all(R[k]["n"] > 0 and not R[k]["miss"] for k in R)
if ok:
    c, l, n2, b0, b1, s0, s1 = R[("KE", "nd60")], R[("KELr", "nd60")], R[("KEr", "nd60")], R[("KE", "b18")], R[("KEr", "b18")], R[("KE", "sf15")], R[("KEr", "sf15")]
    pp = lambda r, k: 100 * r[k] / r["n"]
    e1 = (pp(l, "ever") <= pp(c, "ever") + 3) and (100 * l["frac"] <= 5) and (pp(l, "fell") <= pp(c, "fell") + 3)
    e2 = (pp(n2, "fell") <= pp(c, "fell") + 3) and (pp(n2, "ever") <= pp(c, "ever") + 3)
    pb = fisher(b1["fell"], b1["n"] - b1["fell"], b0["fell"], b0["n"] - b0["fell"]); ps = fisher(s1["fell"], s1["n"] - s1["fell"], s0["fell"], s0["n"] - s0["fell"])
    e3 = (pp(b1, "fell") <= pp(b0, "fell") + 3 and pb >= 0.05) and (pp(s1, "fell") <= pp(s0, "fell") + 3 and ps >= 0.05)
    print(); print(f"- E1 {'충족' if e1 else '미달'}: 왼발 0 + 수정안 — 들어감 {pp(l,'ever'):.1f} % (대조 {pp(c,'ever'):.1f} % + 3) · 모드 시간 {100*l['frac']:.2f} % (≤ 5) · 낙상 {pp(l,'fell'):.2f} % (대조 {pp(c,'fell'):.2f} % + 3)")
    print(f"- E2 {'충족' if e2 else '미달'}: 정상 접촉 + 수정안 — 낙상 {pp(n2,'fell'):.2f} % · 들어감 {pp(n2,'ever'):.1f} % (대조 {pp(c,'fell'):.2f} %·{pp(c,'ever'):.1f} %)")
    print(f"- E3 {'충족' if e3 else '미달'}: 뒤 밀기 {pp(b1,'fell'):.1f} % 대 {pp(b0,'fell'):.1f} % (p {pb:.3g}) · 앞 램프 {pp(s1,'fell'):.1f} % 대 {pp(s0,'fell'):.1f} % (p {ps:.3g})")
    print(f"- **판정: {'E1–E3 충족 → 수정안 후보(시뮬)' if (e1 and e2 and e3) else '미달 있음 → 제안하지 않음'}**")
else:
    print("\n(실행 미완 — 판정 보류)")
