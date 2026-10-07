"""X15 요약(TORQUE_BUDGET_1007.md §8) — KE(서기·회복)의 토크 한계 민감도. 걷기 없음, 배율 env i → i % 4.
사용: python3 reports/gen_w3/x15_summ.py [--seeds 1,2] [--conds nd,b18,sf15] [--prefix tr_x15]
구간(인계 기준): 외란 전 0–16 s · 외란 뒤 16–22 s(6 s) · 전체 0–24 s. 낙상 분모 = 구간 시작 때 생존(사전 검사 통과 env)."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
LAB = {"nd": "무외란", "b18": "뒤 밀기 18 N", "sf15": "앞 램프 15 N"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--conds", default="nd,b18,sf15"); ap.add_argument("--prefix", default="tr_x15"); a = ap.parse_args()
    acc = {}; tor = {}; rec = {}
    for c in a.conds.split(","):
        for s in [int(x) for x in a.seeds.split(",")]:
            tag = f"{a.prefix}_{c}_d2_s{s}"; p = os.path.join(LP, f"kwloop_{tag}.npz"); q = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
            if not (os.path.exists(p) and os.path.exists(q)):
                print("(없음)", tag); continue
            z = np.load(p, allow_pickle=True); pre = z["pre_pass"].astype(bool); sc = np.round(z["effort_scale"].astype(float), 3); t = z["t"].astype(float); kind = z["kind"]
            tf = np.full(pre.shape[0], np.inf)
            for r in csv.DictReader(open(q)):
                if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
                    tf[int(r["env"])] = float(r["t_fall_rel"])
            K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
            pt = os.path.join(LP, f"transall_{tag}.npz"); zt = np.load(pt, allow_pickle=True) if os.path.exists(pt) else None
            for g in sorted(set(sc)):
                m = pre & (sc == g); A = acc.setdefault((c, g), dict(n=0, f0=0, n1=0, f1=0, fa=0)); A["n"] += int(m.sum()); A["f0"] += int((m & (tf < 16.0)).sum())
                al = m & (tf >= 16.0); A["n1"] += int(al.sum()); A["f1"] += int((al & (tf < 22.0)).sum()); A["fa"] += int((m & np.isfinite(tf)).sum())
                # 회복 모드(kind 1) 시간 비율: 외란 전 5–15 s / 외란 뒤 16–22 s
                R = rec.setdefault((c, g), dict(pre=[], post=[])); k5, k15, k16, k22 = K(5), K(15), K(16), K(22)
                for e in np.nonzero(m & (tf >= 16.0))[0]:
                    R["pre"].append(float((kind[k5:k15, e] == 1).mean())); R["post"].append(float((kind[k16:k22, e] == 1).mean()) if tf[e] >= 22 else np.nan)
                if zt is not None and c == "nd":
                    n = zt["tq"].shape[1]; mm = m[:n] & (tf[:n] >= 16.0)
                    if mm.any():
                        tq = np.abs(zt["tq"].astype(float)[K(5):K(15)][:, mm]); lim = 25.0 * g
                        T = tor.setdefault(g, dict(rms=[], sat=[])); T["rms"].append(np.sqrt((tq ** 2).mean(axis=(0, 1)))); T["sat"].append((tq >= lim - 0.5).mean(axis=(0, 1)))
    gs = sorted({g for (_, g) in acc}, reverse=True)
    print("| 조건 | 배율(AK70 한계) | 통과 env | 외란 전 낙상(0–16 s) | 외란 뒤 6 s 낙상 | 전체 낙상 | 1.0 배 대비(외란 뒤) | 회복 모드 시간: 외란 전 / 뒤 |"); print("|---|---|---|---|---|---|---|---|")
    ok = {g: True for g in gs}
    for c in a.conds.split(","):
        base = acc.get((c, 1.0))
        for g in gs:
            A = acc.get((c, g))
            if A is None:
                continue
            r1 = A["f1"] / max(A["n1"], 1); lo, hi = wilson(A["f1"], A["n1"]); ra = A["fa"] / max(A["n"], 1); d = ""
            if base is not None:
                dd = 100 * (ra - base["fa"] / max(base["n"], 1)); d = f"전체 {dd:+.1f} %p"; ok[g] = ok[g] and (dd <= 3.0 + 1e-9)
            R = rec[(c, g)]
            print(f"| {LAB.get(c, c)} | {g:g} ({25 * g:.1f} N·m) | {A['n']} | {A['f0']} ({100 * A['f0'] / max(A['n'], 1):.1f} %) | {A['f1']} / {A['n1']} ({100 * r1:.1f} %, {100 * lo:.0f}–{100 * hi:.0f}) | {A['fa']} ({100 * ra:.1f} %) | {d} | {100 * np.nanmean(R['pre']) if R['pre'] else float('nan'):.1f} % / {100 * np.nanmean(R['post']) if R['post'] else float('nan'):.1f} % |")
    print("- 미리 정한 해석(세 조건 모두 전체 낙상이 1.0 배 대비 +3 %p 이하): " + " · ".join(f"{g:g} 배 {'버팀' if ok[g] else '못 버팀'}" for g in gs if g != 1.0))
    if tor:
        print("| 배율 | 서기 5–15 s 관절 RMS [N·m]: 힙앞뒤 L/R · 힙외전 L/R · 힙회전 L/R · 무릎 L/R · 발목앞뒤 L/R · 발목롤 L/R | 한계 포화 시간 %(무릎 L/R) |"); print("|---|---|---|")
        for g in gs:
            if g in tor:
                r = np.mean(tor[g]["rms"], 0); s = 100 * np.mean(tor[g]["sat"], 0)
                print(f"| {g:g} | " + " · ".join(f"{r[2 * i]:.1f}/{r[2 * i + 1]:.1f}" for i in range(6)) + f" | {s[6]:.1f}/{s[7]:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
