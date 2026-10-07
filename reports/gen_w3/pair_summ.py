"""팔 간 짝 비교 표(전환 평가 env 별 CSV 공용) — 같은 seed·env 를 짝으로 McNemar.
사용: python3 reports/gen_w3/pair_summ.py --ref direct=tr_direct --arm latchd=tr_latchd --arm none=tr_none --conds b18x,sf15x,f12x [--seeds 1,2] [--delay 2] [--metric d1_fell]
metric: d1_fell(첫 외란 뒤 낙상) | d2_fell | fell_sw5(전환 뒤 5 s) | fell(인계 뒤 전체). 분모 = 두 팔 모두 사전검사 통과 ∧ 걷기 중 낙상 아님."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def load(prefix, cond, seeds, delay):
    D = {}; got = []
    for s in seeds:
        p = os.path.join(LP, f"ground_recovery_{prefix}_{cond}_d{delay}_s{s}_envs.csv")
        if not os.path.exists(p):
            continue
        got.append(s)
        for r in csv.DictReader(open(p)):
            if r["pre_pass"] == "1" and r.get("fell_walk", "0") != "1":
                D[(s, int(r["env"]))] = r
    return D, got


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--arm", action="append", default=[]); ap.add_argument("--conds", required=True)
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--delay", default="2"); ap.add_argument("--metric", default="d1_fell"); ap.add_argument("--extra", default="nrm_sw_t,cmd_jump0,sw_on_n")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]; rn, rp = a.ref.split("="); arms = [x.split("=") for x in a.arm]
    ex = [x for x in a.extra.split(",") if x]
    for cond in a.conds.split(","):
        R, gr = load(rp, cond, seeds, a.delay)
        print(f"## 조건 {cond} · 지표 {a.metric} · 기준 팔 {rn}(seed {gr})")
        print("| 팔 | seed | n(공통) | 낙상 | Wilson 95 % | 기준 팔 낙상(같은 env) | 차 | 짝(기준만 / 이 팔만) p | " + " | ".join(f"{e} 중앙" for e in ex) + " |")
        print("|---|---|---|---|---|---|---|---|" + "---|" * len(ex))
        fr_all = sum(R[k].get(a.metric) == "1" for k in R); lo, hi = wilson(fr_all, max(len(R), 1))
        print(f"| {rn} | {','.join(map(str, gr))} | {len(R)} | {fr_all} ({100*fr_all/max(len(R),1):.1f} %) | {100*lo:.1f}–{100*hi:.1f} | — | — | — | " +
              " | ".join(f"{np.nanmedian([fl(R[k], e) for k in R]) if R else float('nan'):.2f}" for e in ex) + " |")
        for nm, pf in arms:
            D, gd = load(pf, cond, seeds, a.delay); ks = sorted(set(R) & set(D))
            if not ks:
                print(f"| {nm} | {','.join(map(str, gd))} | 0 | (자료 없음) |"); continue
            fd = sum(D[k].get(a.metric) == "1" for k in ks); fr = sum(R[k].get(a.metric) == "1" for k in ks); lo, hi = wilson(fd, len(ks))
            b = sum(1 for k in ks if R[k].get(a.metric) == "1" and D[k].get(a.metric) != "1"); c = sum(1 for k in ks if D[k].get(a.metric) == "1" and R[k].get(a.metric) != "1")
            print(f"| {nm} | {','.join(map(str, gd))} | {len(ks)} | {fd} ({100*fd/len(ks):.1f} %) | {100*lo:.1f}–{100*hi:.1f} | {fr} ({100*fr/len(ks):.1f} %) | {100*(fd-fr)/len(ks):+.1f} %p | {b} / {c} p {mcnemar(b, c):.3g} | " +
                  " | ".join(f"{np.nanmedian([fl(D[k], e) for k in ks]):.2f}" for e in ex) + " |")
        print()


if __name__ == "__main__":
    main()
