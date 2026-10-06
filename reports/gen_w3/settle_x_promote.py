"""STOPEST §13c/§13d 판정: 후보(odicwN·odicwH)·제거 팔(xK·xDS·xMIN·xV, + 기존 odic = −|ω|)의 §8 P1–P5(vs imu) + 후보 기준 (2)(vs odicw 짝 낙상, 두 지연 합).
settle_promote.judge 를 그대로 쓴다(같은 P1–P5 코드·같은 imu 기준선 실행 — §11 V0).
사용: python3 reports/gen_w3/settle_x_promote.py [팔 ...] → stopest_results/settle_x_promote.txt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import settle_promote as SP  # noqa: E402

CAND = ("odicwN", "odicwH")


def pair_falls(arm, ref, P):
    """후보 기준 (2): 외란 3 조건 × 지연 2·3 합, 짝 낙상 '팔만' vs 'ref만' — 팔만 > ref만 ∧ p < 0.05 이면 FAIL."""
    b = c = 0; ok = True
    for pfx in ("sab", "sabd3"):
        for cnd in SP.DIST:
            A = SP.load(pfx, ref, cnd); B = SP.load(pfx, arm, cnd)
            if A is None or B is None:
                P(f"  (2) {pfx} {cnd} 자료 부족"); ok = False; continue
            if not SP.rng_same(A, B):
                P(f"  (2) {pfx} {cnd} RNG 짝 불일치 → 무효"); ok = False; continue
            for k in A:
                if A[k]["pre_pass"] != "1":
                    continue
                x, y = A[k]["fell"] == "1", B[k]["fell"] == "1"
                b += x and not y; c += y and not x
    p = SP.mcnemar(b, c)
    res = ok and not (c > b and p < 0.05)
    P(f"  (2) 짝 낙상(두 지연·외란 3 조건) {ref}만 {b} / {arm}만 {c} · p {p:.3g} → {'PASS' if res else 'FAIL'}")
    return res


def main(arms):
    L = []
    P = lambda x="": (L.append(x), print(x, flush=True))   # noqa: E731
    P("# STOPEST §13 시뮬 판정 — P1–P5(§8, vs imu) · 후보 (2) vs odicw · 보조 vs odicw")
    R = {}
    for pfx, dl, strict in (("sab", 2, True), ("sabd3", 3, False)):
        P(f"\n## 지연 {dl} ({'주 판정' if strict else '강건성 — P1 방향만'})")
        for arm in arms:
            P(f"### {arm}")
            R[(arm, dl)] = SP.judge(pfx, arm, strict, P)
            if R[(arm, dl)] is not None and arm != "odicw":
                SP.secondary(pfx, arm, "odicw", P)
    P("\n## 후보 기준 (2)")
    C2 = {a: pair_falls(a, "odicw", P) for a in arms if a in CAND}
    P("\n## 결론")
    for arm in arms:
        v = (R.get((arm, 2)), R.get((arm, 3)))
        s = f"- {arm}: P1–P5 지연 2 {v[0]} · 지연 3 {v[1]}"
        if arm in C2:
            s += f" · (2) {C2[arm]}"
        P(s)
    with open(os.path.join(HERE, "stopest_results", "settle_x_promote.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("odicwN", "odicwH", "xK", "xDS", "xMIN", "xV", "odic"))
