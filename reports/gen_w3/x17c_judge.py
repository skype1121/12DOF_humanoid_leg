"""X17c 판정(TRANSITION_W3KE.md §5.20e) — 전 관절 클램프의 회복 영향, 토크 한계 1.0 배·seed 3–6·팔 4 개.
사용: python3 reports/gen_w3/x17c_judge.py [--seeds 3,4,5,6]
낙상 = 사전검사 통과 env 의 인계 뒤 전체 낙상(ground_recovery_<tag>_envs.csv 의 fell). 팔 간 비교는 Fisher 정확 검정(러너 루프 팔은 실행 간 궤적이 갈려 env 짝이 엄밀하지 않다 — 짝 수도 함께 적는다)."""
import argparse, csv, os
from math import comb
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
ARMS = [("c0", "클램프 없음(대조)"), ("c1", "현 표"), ("c2", "힙 넓힌 제안 표"), ("c3", "발목만(힙·무릎 풀기)")]
LAB = {"b18": "뒤 밀기 18 N", "sf15": "앞 램프 15 N"}


def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    pr = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)   # noqa: E731
    p0 = pr(a); return sum(pr(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(tag):
    p = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("pre_pass", "1") in ("1", "True", "true"):
            out[int(r["env"])] = r["fell"] in ("1", "True", "true")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="3,4,5,6"); a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    print(f"## X17c — 전 관절 클램프의 회복 영향(토크 한계 1.0 배, seed {a.seeds})")
    print("| 조건 | 팔 | 낙상 / 통과 (Wilson 95 %) | 대조 대비 차 | Fisher p(대 대조) | env 짝: 대조만 / 팔만 | 판정(차 ≤ +3 %p ∧ p ≥ 0.05) |"); print("|---|---|---|---|---|---|---|")
    verdict = {}
    for c in ("b18", "sf15"):
        D = {}
        for k, _ in ARMS:
            rows = {}; miss = []
            for s in seeds:
                r = load(f"tr_x17c_{k}_{c}_d2_s{s}")
                if r is None:
                    miss.append(s); continue
                rows.update({(s, e): f for e, f in r.items()})
            D[k] = (rows, miss)
        base, bm = D["c0"]
        for k, nm in ARMS:
            rows, miss = D[k]
            if not rows:
                print(f"| {LAB[c]} | {nm} | (없음 seed {miss}) | | | | |"); continue
            f = sum(rows.values()); n = len(rows); lo, hi = wilson(f, n); note = f" · 미완 seed {miss}" if miss else ""
            if k == "c0":
                print(f"| {LAB[c]} | {nm} | {f} / {n} ({100*f/n:.1f} %, {100*lo:.1f}–{100*hi:.1f}){note} | — | — | — | — |"); continue
            bf = sum(base.values()); bn = len(base); d = 100 * f / n - 100 * bf / bn; p = fisher(f, n - f, bf, bn - bf)
            both = [x for x in rows if x in base]; only_b = sum(1 for x in both if base[x] and not rows[x]); only_a = sum(1 for x in both if rows[x] and not base[x])
            ok = (d <= 3.0) and (p >= 0.05); verdict[(c, k)] = ok
            print(f"| {LAB[c]} | {nm} | {f} / {n} ({100*f/n:.1f} %, {100*lo:.1f}–{100*hi:.1f}){note} | {d:+.1f} %p | {p:.3g} | {only_b} / {only_a} | {'무해' if ok else '**유해**'} |")
    for k, nm in ARMS[1:]:
        v = [verdict.get((c, k)) for c in ("b18", "sf15")]
        print(f"- {nm}: " + ("자료 부족" if None in v else ("**두 조건 모두 무해**" if all(v) else "유해(한 조건 이상)")))


if __name__ == "__main__":
    main()
