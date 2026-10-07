"""STOPEST §16 — imuH 독립 seed 판정: settle_promote.judge(P1–P5, seed 집합 덮어쓰기) + P6(해제 뒤 1 s 안 참 |ω_xy| ≥ 1.0 또는 재켜짐) + 보고용(dwell·정상 자세 복귀).
사용: python3 reports/gen_w3/settle_promote_x.py [out.txt] [seeds=7,8,9,10,11,12] [arm=imuH]
"""
import csv, glob, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settle_promote as sp  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
f = sp.f


def p6_env_flags(pfx, arm, c, s, win=50, w_thr=1.0):
    p = os.path.join(REPO, "log_picture", f"ground_recovery_{pfx}_{arm}_{c}_s{s}_trace.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p); mode = z["mode"]; w = z["w_true"].astype(np.float32); pp = z["pre_pass"]
    T, N = mode.shape; flag = np.zeros(N, dtype=bool); n_off = np.zeros(N, dtype=int)
    for t in range(1, T):
        off = mode[t - 1] & ~mode[t]
        if not off.any():
            continue
        e = np.nonzero(off)[0]; n_off[e] += 1
        t1 = min(T, t + win)
        reon = mode[t + 1:t1].any(0) if t + 1 < t1 else np.zeros(N, dtype=bool)
        big = (w[t:t1] >= w_thr).any(0)
        flag[e] |= (reon | big)[e]
    return flag, n_off, pp


def dwell_stats(pfx, arm, c, seeds):
    tv = []; nr = []
    for s in seeds:
        p = sp.LPD.format(pfx, arm, c, s)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r["pre_pass"] != "1" or r["fell"] == "1":
                continue
            x = f(r.get("sw_time_s"));  y = f(r.get("nrm_t"))
            if x is not None: tv.append(x)
            if y is not None: nr.append(y)
    q = lambda v, p: (float(np.percentile(v, p)) if v else float("nan"))
    return q(tv, 50), q(tv, 90), q(nr, 50), q(nr, 90), len(tv), len(nr)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    seeds = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "7,8,9,10,11,12").split(","))
    arm = sys.argv[3] if len(sys.argv) > 3 else "imuH"
    sp.SEEDS = seeds
    L = []; P = lambda x="": (L.append(x), print(x, flush=True))
    P(f"# STOPEST §16 imuH 독립 seed 판정 — seed {seeds} · 팔 {arm} vs imu")
    R = {}
    for pfx, dl, strict in (("sab", 2, True), ("sabd3", 3, False)):
        P(f"\n## 지연 {dl} ({'P1 엄격' if strict else 'P1 방향'})")
        R[(dl, "p15")] = sp.judge(pfx, arm, strict, P)
        # P6
        bi = ci = n = 0; fa = []; fb = []; miss = 0
        for c in sp.DIST:
            for s in seeds:
                A = p6_env_flags(pfx, "imu", c, s); B = p6_env_flags(pfx, arm, c, s)
                if A is None or B is None:
                    miss += 1; continue
                m = A[2] & B[2]
                fa += list(A[0][m]); fb += list(B[0][m]); n += int(m.sum())
                bi += int((A[0] & ~B[0] & m).sum()); ci += int((B[0] & ~A[0] & m).sum())
        if n:
            p6 = (np.mean(fb) <= np.mean(fa) + 0.02)
            P(f"  P6 해제 뒤 1 s 안 |ω|≥1.0 또는 재켜짐(env 비율): imu {100*np.mean(fa):.1f}% · {arm} {100*np.mean(fb):.1f}% · 짝 imu만 {bi} / {arm}만 {ci} (n {n}, p {sp.mcnemar(bi, ci):.3g}){' · 누락 '+str(miss) if miss else ''} → {'PASS' if p6 else 'FAIL'}")
            R[(dl, "p6")] = bool(p6)
        else:
            P("  P6: 추적 파일 없음"); R[(dl, "p6")] = None
        for c in ("nd",) + sp.DIST:
            a = dwell_stats(pfx, "imu", c, seeds); b = dwell_stats(pfx, arm, c, seeds)
            P(f"  보고 {c}: 회복 모드 시간 중앙/p90 imu {a[0]:.2f}/{a[1]:.2f} s · {arm} {b[0]:.2f}/{b[1]:.2f} s | 정상 자세 복귀 nrm_t 중앙/p90 imu {a[2]:.2f}/{a[3]:.2f} · {arm} {b[2]:.2f}/{b[3]:.2f} (n {a[4]}/{b[4]})")
    P("\n## 결론")
    ok = all(v is True for v in R.values())
    P(f"- {arm}: 지연 2 P1–P5 {R.get((2,'p15'))} P6 {R.get((2,'p6'))} · 지연 3 P1–P5 {R.get((3,'p15'))} P6 {R.get((3,'p6'))} → {'통과(승격 제안)' if ok else '불통과 또는 보류'}")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
