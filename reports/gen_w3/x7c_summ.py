"""X7c·X7d·S3E 요약(TRANSITION_W3KE.md §5.11d–f, S3E_DESIGN.md E1–E2) — 앞 0.06 × 5 주기 왕복에서 팔별 주기 2–5 낙상과 걷기 한 번의 요 변화.
사용: python3 reports/gen_w3/x7c_summ.py [--arms c0=tr_x7c_c0,c1=tr_x7c_c1,...] [--seeds 1,2] [--ref c0]
낙상: 주기 시작 → 다음 주기 시작 전(마지막은 47 s), 분모 = 주기 시작 때 생존(등록 정의). 대조 대비 Fisher 정확(비짝) + 보조로 env 짝 McNemar(주기 2–5 중 한 번이라도 낙상).
요 변화: 걷기 명령이 앞 0.06 인 4 s 구간(팔별 시작 시각 offs 반영)의 |Δψ| 중앙."""
import argparse, csv, os, sys
import numpy as np
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
CYC = [2.0, 11.0, 20.0, 29.0, 38.0]


def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    pr = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)   # noqa: E731
    p0 = pr(a); return sum(pr(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if pr(x) <= p0 + 1e-12)


def load(prefix, s):
    p = os.path.join(LP, f"kwloop_{prefix}_d2_s{s}.npz"); q = os.path.join(LP, f"ground_recovery_{prefix}_d2_s{s}_envs.csv")
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    z = np.load(p, allow_pickle=True); pre = z["pre_pass"].astype(bool); tf = np.full(pre.shape[0], np.inf)
    for r in csv.DictReader(open(q)):
        if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
            tf[int(r["env"])] = float(r["t_fall_rel"])
    return dict(t=z["t"].astype(float), yaw=z["yaw"].astype(float), pre=pre, tf=tf, kind=z["kind"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="c0=tr_x7c_c0,c1=tr_x7c_c1,c2=tr_x7c_c2,c3=tr_x7c_c3:1.5,c4=tr_x7c_c4,d1=tr_x7d_d1")
    ap.add_argument("--seeds", default="1,2"); ap.add_argument("--ref", default="c0")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]; R = {}
    for item in a.arms.split(","):
        nm, rest = item.split("="); pref, _, off = rest.partition(":"); off = float(off) if off else 0.0
        n0 = [0] * 5; nf = [0] * 5; dy = [[] for _ in range(5)]; anyf = {}; got = []; rec2 = 0; nh = 0
        for s in seeds:
            D = load(pref, s)
            if D is None:
                continue
            got.append(s); t = D["t"]; K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
            for c, ts in enumerate(CYC):
                t1 = CYC[c + 1] if c + 1 < 5 else 47.0
                al = D["pre"] & (D["tf"] >= ts); fl = al & (D["tf"] < t1); n0[c] += int(al.sum()); nf[c] += int(fl.sum())
                ok = D["pre"] & (D["tf"] >= ts + off + 4.5); k0, k1 = K(ts + off), K(ts + off + 4.0)
                d = np.degrees(np.arctan2(np.sin(D["yaw"][k1] - D["yaw"][k0]), np.cos(D["yaw"][k1] - D["yaw"][k0]))); dy[c].append(np.abs(d[ok]))
                if c >= 1:
                    for e in np.nonzero(al)[0]:
                        anyf[(s, int(e))] = anyf.get((s, int(e)), False) or bool(fl[e])
            # KE 인계 뒤 2 s 안 회복 모드 켜짐(kind 1) env·인계 수
            kd = D["kind"]
            for e in np.nonzero(D["pre"])[0]:
                tr = np.nonzero((kd[1:, e] != 2) & (kd[:-1, e] == 2))[0] + 1
                for k in tr:
                    nh += 1; rec2 += bool((kd[k:k + 100, e] == 1).any())
        R[nm] = dict(n0=n0, nf=nf, dy=[np.concatenate(x) if x else np.zeros(0) for x in dy], anyf=anyf, seeds=got, rec2=rec2, nh=nh)
    ref = R.get(a.ref)
    print("| 팔 | seed | 첫 주기 낙상 | 주기 2–5 낙상 | 대조 대비 Fisher p | env 짝(팔만 / 대조만) · p | 걷기 \\|Δψ\\| 중앙: 주기 1 → 5 [°] | 다섯째 / 대조 다섯째 | 인계 뒤 2 s 안 회복 켜짐 |"); print("|---|---|---|---|---|---|---|---|---|")
    for nm, r in R.items():
        if not r["seeds"]:
            print(f"| {nm} | — | (자료 없음) |"); continue
        f25 = sum(r["nf"][1:]); n25 = sum(r["n0"][1:]); med = [float(np.median(x)) if len(x) else float("nan") for x in r["dy"]]
        if ref and nm != a.ref and ref["seeds"] and set(r["seeds"]) <= set(ref["seeds"]):
            # 같은 seed 만으로 대조 재집계
            rf = sum(ref["nf"][1:]); rn = sum(ref["n0"][1:]); pf = fisher(f25, n25 - f25, rf, rn - rf) if set(r["seeds"]) == set(ref["seeds"]) else float("nan")
            ks = [k for k in r["anyf"] if k in ref["anyf"]]; b = sum(1 for k in ks if r["anyf"][k] and not ref["anyf"][k]); c_ = sum(1 for k in ks if ref["anyf"][k] and not r["anyf"][k])
            mref = float(np.median(ref["dy"][4])) if len(ref["dy"][4]) else float("nan")
            extra = f"{pf:.3f} | {b} / {c_} · {mcnemar(b, c_):.3f}"; ratio = f"{med[4] / mref:.2f}"
        else:
            extra = "— | —"; ratio = "1.00" if nm == a.ref else "—"
        print(f"| {nm} | {','.join(map(str, r['seeds']))} | {r['nf'][0]} / {r['n0'][0]} | {f25} / {n25} ({100*f25/max(n25,1):.2f} %) | {extra} | {' → '.join(f'{m:.1f}' for m in med)} | {ratio} | {r['rec2']} / {r['nh']} ({100*r['rec2']/max(r['nh'],1):.1f} %) |")


if __name__ == "__main__":
    main()
