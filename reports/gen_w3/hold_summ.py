"""홀드 창 외란 요약(TRANSITION_W3KE.md 창 외란·보강): tr_<arm>_h<cond>_d2_s<seed> — 밀기 시각(9 s)의 상태(W3 STAND / 수동 홀드 / KE)별 외란 뒤 낙상.
같은 (seed, env) 는 9 s 까지 세 팔이 같은 궤적(정착 시각 같음)이므로 상태만 다른 짝 비교가 된다.
사용: python3 reports/gen_w3/hold_summ.py [--t 9.0]
"""
import argparse, csv, glob, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trans_summ import wilson, mcnemar  # noqa: E402
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def state(arm, r, t):
    ts, tsw = r.get("tr_settle", ""), r.get("tr_sw", "")
    if arm == "none" or ts == "" or float(ts) > t:
        return "W3STAND"
    if arm == "restart":
        return "hold" if (tsw == "" or float(tsw) > t) else "KE"
    return "KE" if (tsw != "" and float(tsw) <= t) else "W3STAND"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--t", type=float, default=9.0); a = ap.parse_args()
    D = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(REPO, "log_picture", "ground_recovery_tr_*_h*_d2_s*_envs.csv"))):
        m = re.search(r"tr_(restart|none|reset)_h(\w+?)_d2_s(\d+)_envs", os.path.basename(p))
        if not m:
            continue
        arm, cond, s = m.group(1), m.group(2), int(m.group(3))
        for r in csv.DictReader(open(p)):
            D[(cond, arm)][(s, int(r["env"]))] = r
    for cond in sorted({k[0] for k in D}):
        print(f"\n## 조건 h{cond} (밀기 {a.t} s)")
        if cond.endswith("nd"):
            for arm in ("restart",):
                R = D.get((cond, arm), {}); ks = [k for k, r in R.items() if r["pre_pass"] == "1"]
                fw = sum(R[k].get("fell_win") == "1" for k in ks); f5 = sum(R[k].get("fell_sw5") == "1" for k in ks)
                pk = sorted(float(R[k]["pk_pw"]) for k in ks if R[k].get("pk_pw", "") != "")
                print(f"  {arm}: n {len(ks)} · 홀드 창 낙상 {fw} · KE 뒤 5 s 낙상 {f5} · 창 피크 |피치| 중앙 {pk[len(pk)//2] if pk else float('nan'):.1f}° p95 {pk[int(0.95*len(pk))] if pk else float('nan'):.1f}°")
            continue
        print("| 팔 | 밀기 시점 상태 | n | 외란 뒤 낙상 | Wilson 95 % |"); print("|---|---|---|---|---|")
        st = {}
        for arm in ("restart", "none", "reset"):
            R = D.get((cond, arm), {}); cnt = defaultdict(lambda: [0, 0])
            for k, r in R.items():
                if r["pre_pass"] != "1" or r.get("fell_walk") == "1":
                    continue
                s_ = state(arm, r, a.t); st[(arm, k)] = (s_, r.get("d1_fell") == "1"); cnt[s_][0] += 1; cnt[s_][1] += r.get("d1_fell") == "1"
            for s_, (n, f_) in sorted(cnt.items()):
                lo, hi = wilson(f_, n); print(f"| {arm} | {s_} | {n} | {f_} ({100*f_/max(n,1):.1f} %) | {100*lo:.1f}–{100*hi:.1f} |")
        # 짝: 같은 env 가 restart=hold, none=W3STAND, reset=KE 인 경우
        for A, B, sa, sb in (("restart", "none", "hold", "W3STAND"), ("restart", "reset", "hold", "KE"), ("none", "reset", "W3STAND", "KE")):
            b = c = n = 0
            for (arm, k), (s_, f_) in st.items():
                if arm != A or s_ != sa or (B, k) not in st or st[(B, k)][0] != sb:
                    continue
                n += 1; g = st[(B, k)][1]; b += f_ and not g; c += g and not f_
            if n:
                print(f"- 짝 {sa}({A}) vs {sb}({B}): {sa}만 {b} / {sb}만 {c} (n {n}, McNemar p {mcnemar(b, c):.3g})")


if __name__ == "__main__":
    main()
