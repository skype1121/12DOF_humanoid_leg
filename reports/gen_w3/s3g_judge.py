"""W3S3g 판정(TRANSITION_W3KE.md §5.19 G1–G4) — 왕복 대본(앞 0.06 × 5 주기)에서 쏠림·요·낙상을 대조(W3S3e 같은 체크포인트)·기준선(X7c C0)과 나란히.
사용: python3 reports/gen_w3/s3g_judge.py --cand tr_s3g16299_ff --ctl tr_s3e16299_ff [--base tr_x7c_c0] [--seeds 1,2]
G1 기전: 다섯째 걷기의 걷는 중 평균 롤 표준편차 ≤ 대조 0.6 배 ∧ 다섯째 걷기 요 속도 표준편차 ≤ 대조 0.6 배.
G2: 다섯째 걷기 |Δψ| 중앙 ≤ 첫 걷기의 1.5 배. G3: 주기 2–5 낙상 ≤ 기준선의 절반 ∧ Fisher p < 0.1. G4: 첫 주기 낙상 0–1 ∧ 4 s 전진 중앙 ≥ 기준선의 0.85 배."""
import argparse, csv, os
import numpy as np
from math import comb
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
STARTS = [2.0, 11.0, 20.0, 29.0, 38.0]; W = 4.0; END = 47.0


def fisher(a, b, c, d):
    """2×2 양측 Fisher 정확검정 p."""
    n = a + b + c + d; r1 = a + b; c1 = a + c
    p0 = comb(r1, a) * comb(n - r1, c1 - a) / comb(n, c1); p = 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        px = comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
        if px <= p0 * (1 + 1e-9):
            p += px
    return min(1.0, p)


def arm(prefix, seeds):
    Y = []; Wr = []; R = []; F = []; n1 = f1 = n2 = f2 = 0; got = []
    for s in seeds:
        tag = f"{prefix}_d2_s{s}"; pk = os.path.join(LP, f"kwloop_{tag}.npz"); pc = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
        if not (os.path.exists(pk) and os.path.exists(pc)):
            continue
        got.append(tag); z = np.load(pk, allow_pickle=True); t = z["t"].astype(float); pre = z["pre_pass"].astype(bool)
        tf = np.full(pre.shape[0], np.inf)
        for r in csv.DictReader(open(pc)):
            if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
                tf[int(r["env"])] = float(r["t_fall_rel"])
        K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        yaw = np.degrees(np.unwrap(z["yaw"].astype(float), axis=0)); xy = z["xy"].astype(float)
        a1 = pre & (tf >= STARTS[0]); n1 += int(a1.sum()); f1 += int((a1 & (tf < STARTS[1])).sum())
        for c in range(1, 5):
            t1 = STARTS[c + 1] if c + 1 < 5 else END; al = pre & (tf >= STARTS[c]); n2 += int(al.sum()); f2 += int((al & (tf < t1)).sum())
        ok = pre & (tf >= END)
        Y.append(np.stack([yaw[K(s0 + W), ok] - yaw[K(s0), ok] for s0 in STARTS], 1)); Wr.append(np.stack([(yaw[K(s0 + 4), ok] - yaw[K(s0 + 2), ok]) / 2.0 for s0 in STARTS], 1))
        k0 = K(STARTS[0]); c0, s0_ = np.cos(np.radians(yaw[k0, ok])), np.sin(np.radians(yaw[k0, ok])); d = xy[K(STARTS[0] + W), ok] - xy[k0, ok]; F.append(d[:, 0] * c0 + d[:, 1] * s0_)
        pt = os.path.join(LP, f"transall_{tag}.npz")
        if os.path.exists(pt):
            zt = np.load(pt, allow_pickle=True); n = zt["roll"].shape[1]; okn = ok[:n]; roll = zt["roll"].astype(float)
            R.append(np.stack([roll[K(s0 + 2):K(s0 + 4), okn].mean(0) for s0 in STARTS], 1))
    if not got:
        return None
    Y = np.concatenate(Y); Wr = np.concatenate(Wr); F = np.concatenate(F); R = np.concatenate(R) if R else None
    return dict(tags=got, n=len(Y), dpsi=np.median(np.abs(Y), 0), wstd=Wr.std(0), rstd=(R.std(0) if R is not None else None), nR=(len(R) if R is not None else 0), fwd=float(np.median(F)), n1=n1, f1=f1, n2=n2, f2=f2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--ctl", default=""); ap.add_argument("--base", default="tr_x7c_c0"); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--name", default=""); ap.add_argument("--mode", default="g", choices=["g", "k1"], help="g = W3S3g 의 G1–G4 · k1 = W3_FINAL_SELECT.md §6 의 재현 기준 K1(기준선은 --base, 같은 seed)")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    C = arm(a.cand, seeds); B = arm(a.base, seeds); T = arm(a.ctl, seeds) if a.ctl else None
    if C is None or B is None:
        print("자료 없음:", "후보" if C is None else "", "기준선" if B is None else ""); return 2
    f5 = lambda v: " → ".join(f"{x:.1f}" for x in v)   # noqa: E731
    print(f"## {a.name or a.cand} — 실행 {', '.join(C['tags'])}")
    print("| 팔 | 완주 env | 첫 주기 낙상 | 주기 2–5 낙상 | 걷기 \\|Δψ\\| 중앙 1 → 5 [°] | 걷는 중 요 속도 표준편차 1 → 5 [°/s] | 걷는 중 평균 롤 표준편차 1 → 5 [°] | 첫 걷기 4 s 전진 [m] |"); print("|---|---|---|---|---|---|---|---|")
    for nm, A in (("기준선 " + a.base, B), ("대조 " + a.ctl, T), ("후보 " + a.cand, C)):
        if A is None:
            continue
        print(f"| {nm} | {A['n']} | {A['f1']} / {A['n1']} | {A['f2']} / {A['n2']} ({100 * A['f2'] / max(A['n2'], 1):.2f} %) | {f5(A['dpsi'])} | {f5(A['wstd'])} | {f5(A['rstd']) if A['rstd'] is not None else '—'} (n {A['nR']}) | {A['fwd']:+.3f} |")
    if a.mode == "k1":
        r2 = C["dpsi"][4] / C["dpsi"][0]; rw = C["wstd"][4] / B["wstd"][4]; rc = C["f2"] / max(C["n2"], 1); rb = B["f2"] / max(B["n2"], 1)
        k = [("다섯째 / 첫 걷기 |Δψ| 중앙 ≤ 1.5", r2 <= 1.5, f"{r2:.2f} 배(기준선 {B['dpsi'][4] / B['dpsi'][0]:.2f} 배)"),
             ("다섯째 걷기 요 속도 표준편차 ≤ 기준선의 0.6 배", rw <= 0.6, f"{C['wstd'][4]:.1f} 대 {B['wstd'][4]:.1f} °/s = {rw:.2f} 배"),
             ("주기 2–5 낙상 ≤ 기준선 + 1 %p", 100 * (rc - rb) <= 1.0 + 1e-9, f"{C['f2']} / {C['n2']} ({100 * rc:.2f} %) 대 {B['f2']} / {B['n2']} ({100 * rb:.2f} %), 차 {100 * (rc - rb):+.2f} %p"),
             ("첫 주기 낙상 ≤ 1", C["f1"] <= 1, f"{C['f1']} 건(기준선 {B['f1']})")]
        for nm, okk, val in k:
            print(f"- K1 · {nm}: {val} → {'충족' if okk else '미달'}")
        print("- **K1 종합**: " + ("충족" if all(x[1] for x in k) else "미달") + f" · 참고: 첫 걷기 4 s 전진 {C['fwd']:+.3f} m(기준선 {B['fwd']:+.3f}, 명령 0.24 m)")
        return 0
    res = []
    if T is not None and C["rstd"] is not None and T["rstd"] is not None:
        r_roll = C["rstd"][4] / T["rstd"][4]; r_w = C["wstd"][4] / T["wstd"][4]; g1 = r_roll <= 0.6 and r_w <= 0.6
        print(f"- **G1 기전**: 다섯째 걷기 롤 표준편차 후보/대조 = {r_roll:.2f} · 요 속도 표준편차 후보/대조 = {r_w:.2f} (기준 둘 다 ≤ 0.6) → {'충족' if g1 else '미달'}"); res.append(("G1", g1))
    r2 = C["dpsi"][4] / C["dpsi"][0]; g2 = r2 <= 1.5
    print(f"- **G2 요 누적**: 다섯째 / 첫 걷기 |Δψ| 중앙 = {r2:.2f} 배 (기준 ≤ 1.5, 기준선 {B['dpsi'][4] / B['dpsi'][0]:.2f} 배" + (f", 대조 {T['dpsi'][4] / T['dpsi'][0]:.2f} 배" if T else "") + f") → {'충족' if g2 else '미달'}"); res.append(("G2", g2))
    p = fisher(C["f2"], C["n2"] - C["f2"], B["f2"], B["n2"] - B["f2"]); rc = C["f2"] / max(C["n2"], 1); rb = B["f2"] / max(B["n2"], 1); g3 = (rc <= 0.5 * rb) and p < 0.1
    print(f"- **G3 낙상**: 주기 2–5 {100 * rc:.2f} % 대 기준선 {100 * rb:.2f} % (비 {rc / rb if rb > 0 else float('nan'):.2f}, Fisher p {p:.3f}; 기준 절반 이하 ∧ p < 0.1) → {'충족' if g3 else '미달'}"); res.append(("G3", g3))
    g4 = C["f1"] <= 1 and C["fwd"] >= 0.85 * B["fwd"]
    print(f"- **G4 해치지 않음**: 첫 주기 낙상 {C['f1']} (≤ 1) · 첫 걷기 4 s 전진 {C['fwd']:+.3f} m (기준선 {B['fwd']:+.3f} 의 {C['fwd'] / B['fwd']:.2f} 배, 기준 ≥ 0.85) → {'충족' if g4 else '미달'}"); res.append(("G4", g4))
    print("- 종합: " + " · ".join(f"{k} {'충족' if v else '미달'}" for k, v in res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
