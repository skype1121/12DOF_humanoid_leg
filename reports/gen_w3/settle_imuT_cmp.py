"""STOPEST §13e: '늦게 끄기만' 대조 imuT vs odicw — 시간 정합·짝 낙상(조건별·합)·imuT P1–P5.
사용: python3 reports/gen_w3/settle_imuT_cmp.py → stopest_results/settle_imuT_cmp.txt
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import settle_promote as SP  # noqa: E402

f = SP.f


def main():
    L = []
    P = lambda x="": (L.append(x), print(x, flush=True))   # noqa: E731
    P("# §13e imuT(현행 imu, 조기 복귀 최소 시간 0.6→1.1 s) 대조 — odicw 의 오도메트리 고유 효과 분리")
    for pfx, dl in (("sab", 2), ("sabd3", 3)):
        P(f"\n## 지연 {dl}")
        D = {a: {c: SP.load(pfx, a, c) for c in ("nd",) + SP.DIST} for a in ("imu", "imuT", "odicw")}
        if any(v is None for a in D for v in D[a].values()):
            P("  자료 부족"); continue
        for c in ("nd",) + SP.DIST:
            if not (SP.rng_same(D["imu"][c], D["imuT"][c]) and SP.rng_same(D["imu"][c], D["odicw"][c])):
                P(f"  {c} RNG 짝 불일치 → 무효"); return
        # 시간 정합
        for a in ("imuT", "odicw"):
            dt = [f(D[a][c][k]["sw_time_s"]) - f(D["imu"][c][k]["sw_time_s"]) for c in SP.DIST for k in D["imu"][c]
                  if D["imu"][c][k]["pre_pass"] == "1" and D["imu"][c][k]["fell"] != "1" and D[a][c][k]["fell"] != "1"]
            P(f"  회복 모드 시간 차({a} − imu) 평균 {np.mean(dt):+.3f} s")
        tb = tc = 0
        for c in SP.DIST:
            A, B, C = D["imuT"][c], D["odicw"][c], D["imu"][c]
            ks = [k for k in A if A[k]["pre_pass"] == "1"]
            b = sum(1 for k in ks if A[k]["fell"] == "1" and B[k]["fell"] != "1"); cc = sum(1 for k in ks if B[k]["fell"] == "1" and A[k]["fell"] != "1")
            bi = sum(1 for k in ks if C[k]["fell"] == "1" and A[k]["fell"] != "1"); ci = sum(1 for k in ks if A[k]["fell"] == "1" and C[k]["fell"] != "1")
            fr = {a: 100 * np.mean([D[a][c][k]["fell"] == "1" for k in ks]) for a in ("imu", "imuT", "odicw")}
            P(f"  {c}: 낙상 imu {fr['imu']:.1f} · imuT {fr['imuT']:.1f} · odicw {fr['odicw']:.1f}% · 짝 imuT만 {b} / odicw만 {cc} (p {SP.mcnemar(b, cc):.3g}) · 짝 imu만 {bi} / imuT만 {ci} (p {SP.mcnemar(bi, ci):.3g})")
            tb += b; tc += cc
        P(f"  외란 3 조건 합: imuT만 {tb} / odicw만 {tc} · p {SP.mcnemar(tb, tc):.3g} → {'오도메트리 고유 효과 지지' if (tc < tb and SP.mcnemar(tb, tc) < 0.05) else '늦게 끈 효과와 구분 안 됨'}")
        P(f"### imuT P1–P5 (vs imu)")
        SP.judge(pfx, "imuT", dl == 2, P)
    with open(os.path.join(HERE, "stopest_results", "settle_imuT_cmp.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
