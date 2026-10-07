"""X13 요약(TORQUE_BUDGET_1007.md §5) — 토크 한계 배율별(env i → i % 4) 구간 낙상·전진·토크.
사용: python3 reports/gen_w3/x13_summ.py [--seeds 1,2] [--prefix tr_x13]
구간(인계 기준): 제자리 걸음 2–8 s · 앞 0.06 8–16 s · 정지·서기 16–20 s · 밀기 뒤 20–24 s. 낙상 분모 = 구간 시작 때 생존(통과 env)."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
SEG = [("제자리 걸음", 2.0, 8.0), ("앞 0.06", 8.0, 16.0), ("정지·서기", 16.0, 20.0), ("밀기 뒤", 20.0, 24.5)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--prefix", default="tr_x13", help="실행 이름 접두(예: tr_s3f16299_x13)"); a = ap.parse_args()
    acc = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        p = os.path.join(LP, f"kwloop_{a.prefix}_d2_s{s}.npz"); q = os.path.join(LP, f"ground_recovery_{a.prefix}_d2_s{s}_envs.csv")
        if not (os.path.exists(p) and os.path.exists(q)):
            print("없음", p); continue
        z = np.load(p, allow_pickle=True); pre = z["pre_pass"].astype(bool); sc = z["effort_scale"].astype(float); t = z["t"].astype(float); xy = z["xy"].astype(float); yaw = z["yaw"].astype(float)
        tf = np.full(pre.shape[0], np.inf)
        for r in csv.DictReader(open(q)):
            if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
                tf[int(r["env"])] = float(r["t_fall_rel"])
        K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        zt = None
        pt = os.path.join(LP, f"transall_{a.prefix}_d2_s{s}.npz")
        if os.path.exists(pt):
            zt = np.load(pt, allow_pickle=True)
        for g in sorted(set(np.round(sc, 3))):
            m = pre & (np.abs(sc - g) < 1e-6); A = acc.setdefault(g, dict(n=[0] * 4, f=[0] * 4, fwd=[], rms=[], sat=[]))
            for i, (nm, t0, t1) in enumerate(SEG):
                al = m & (tf >= t0); A["n"][i] += int(al.sum()); A["f"][i] += int((al & (tf < t1)).sum())
            ok = m & (tf >= 16.0); d = xy[K(16.0)] - xy[K(8.0)]; c0, s0 = np.cos(yaw[K(8.0)]), np.sin(yaw[K(8.0)]); A["fwd"] += list((c0 * d[:, 0] + s0 * d[:, 1])[ok])
            if zt is not None and "tq" in zt.files:
                tq = np.abs(zt["tq"].astype(np.float32)); n128 = tq.shape[1]; mm = m[:n128] & (tf[:n128] >= 16.0)
                if mm.any():
                    X = tq[K(9.0):K(16.0)][:, mm]; lim = 25.0 * g
                    A["rms"].append(np.sqrt((X[..., [2, 3, 6, 7]] ** 2).mean((0, 1)))); A["sat"].append((X[..., [2, 3, 6, 7]] >= lim - 0.5).mean((0, 1)))
    print("| 토크 한계 배율(AK70 한계) | 제자리 걸음 낙상 | 앞 걷기 낙상 | 정지·서기 낙상 | 밀기(뒤 12 N) 뒤 낙상 | 걷기 구간 합(2–16 s) | 8 s 전진 중앙 [m] | 걷기 중 RMS: 힙 외전 L/R·무릎 L/R [N·m] | 한계 포화 시간 [%] |"); print("|---|---|---|---|---|---|---|---|---|")
    base = None
    for g in sorted(acc, reverse=True):
        A = acc[g]; cells = []
        for i in range(4):
            lo, hi = wilson(A["f"][i], max(A["n"][i], 1)); cells.append(f"{A['f'][i]} / {A['n'][i]} ({100*A['f'][i]/max(A['n'][i],1):.1f} %)")
        wf = A["f"][0] + A["f"][1]; wn = A["n"][0]; lo, hi = wilson(wf, max(wn, 1)); wr = 100 * wf / max(wn, 1)
        if base is None:
            base = wr
        rms = np.mean(A["rms"], 0) if A["rms"] else None; sat = 100 * np.mean(A["sat"], 0) if A["sat"] else None
        print(f"| {g:g} ({25*g:.1f} N·m) | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {wf} / {wn} ({wr:.1f} %, {100*lo:.1f}–{100*hi:.1f}) · 기준 대비 {wr - base:+.1f} %p | {np.median(A['fwd']):+.2f} | "
              + (f"{rms[0]:.1f}/{rms[1]:.1f} · {rms[2]:.1f}/{rms[3]:.1f}" if rms is not None else "—") + " | " + (f"{sat[0]:.0f}/{sat[1]:.0f} · {sat[2]:.0f}/{sat[3]:.0f}" if sat is not None else "—") + " |")


if __name__ == "__main__":
    main()
