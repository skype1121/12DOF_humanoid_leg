"""Q5 요약(TRANSITION_W3KE.md): 걷는 중 외란 직후 4 팔(walk·keKE·stopW3·stopKE) — 팔별 낙상률(Wilson), walk 기준 짝 McNemar, 외란 시 보행 위상(8 구간)별 낙상.
위상: ge 켬 = w3_on + (env mod K)/K × 0.909 s, 외란 t0 → u = ((t0 − t_on) × 1.1) mod 1 (앞 명령 = 왼발 리드, u < 0.5 = 왼발 스윙 반주기).
사용: python3 reports/gen_w3/q5_summ.py [--out reports/gen_w3/q5_result.md] [--tag q5]
"""
import argparse, csv, glob, math, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trans_summ import wilson, mcnemar, fisher2, fp  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARMS = ("walk", "keKE", "keKEc", "stopW3", "stopKE")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "reports", "gen_w3", "q5_result.md"))
    ap.add_argument("--tag", default="q5"); ap.add_argument("--w3_on", type=float, default=2.0); ap.add_argument("--t0", type=float, default=5.0); ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()
    D = defaultdict(dict)   # (arm, dist, delay) -> (seed, env) -> row
    for p in sorted(glob.glob(os.path.join(REPO, "log_picture", f"ground_recovery_tr_*_{a.tag}*_envs.csv"))):
        m = re.search(rf"tr_(\w+?)_{a.tag}(\w+?)_d(\d)_s(\d+)_envs", os.path.basename(p))
        if not m:
            continue
        arm, dist, dl, s = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        for r in csv.DictReader(open(p)):
            D[(arm, dist, dl)][(s, int(r["env"]))] = r
    L = []; P = L.append
    P(f"# Q5 걷는 중 외란 직후 4 팔 비교 (자동 생성 q5_summ.py, 태그 {a.tag})")
    for dist, dl in sorted({(k[1], k[2]) for k in D}):
        P(f"\n## 외란 {dist} · 지연 {dl} (사전검사 통과 env, seed 합)")
        P("| 팔 | seed | n | 외란 뒤 낙상 | Wilson 95 % | walk 대비 짝(walk만 / 팔만, p) | KE 회복 켜짐 | 서기 정상 자세 도달 |")
        P("|---|---|---|---|---|---|---|---|")
        base = D.get(("walk", dist, dl))
        for arm in ARMS:
            R = D.get((arm, dist, dl))
            if not R:
                continue
            ks = [k for k, r in R.items() if r["pre_pass"] == "1"]; n = len(ks)
            fell = sum(R[k]["fell"] == "1" for k in ks); lo, hi = wilson(fell, n)
            pair = "—"
            if base and arm != "walk":
                b = c = both = nn = 0; ok = True
                for k in ks:
                    rb = base.get(k)
                    if rb is None or rb["pre_pass"] != "1":
                        continue
                    if fp(rb) != fp(R[k]):
                        ok = False
                    nn += 1; x = rb["fell"] == "1"; y = R[k]["fell"] == "1"; b += x and not y; c += y and not x; both += x and y
                p = mcnemar(b, c) if ok else fisher2(b + both, nn - b - both, c + both, nn - c - both)
                pair = f"{b} / {c} ({'McNemar' if ok else 'Fisher'} p {p:.3g})"
            swon = sum((float(R[k].get("sw_on_n") or 0) > 0) for k in ks); nrm = sum(R[k].get("nrm_t", "") != "" for k in ks)
            P(f"| {arm} | {','.join(str(s) for s in sorted({k[0] for k in ks}))} | {n} | {fell} ({100*fell/max(n,1):.1f} %) | {100*lo:.1f}–{100*hi:.1f} | {pair} | {swon} | {nrm} ({100*nrm/max(n,1):.0f} %) |")
        # 위상별
        P(f"\n외란 시 보행 위상별 낙상(env mod {a.K} → u, 앞 명령 왼발 리드: u < 0.5 왼발 스윙 반주기):")
        hdr = "| u |" + "".join(f" {arm} |" for arm in ARMS if D.get((arm, dist, dl))); P(hdr); P("|---|" + "---|" * (hdr.count("|") - 2))
        ph = {}
        for k in range(a.K):
            t_on = a.w3_on + k / a.K * 0.909; ph[k] = ((a.t0 - t_on) * 1.1) % 1.0
        for k in sorted(range(a.K), key=lambda x: ph[x]):
            row = f"| {ph[k]:.2f} |"
            for arm in ARMS:
                R = D.get((arm, dist, dl))
                if not R:
                    continue
                ks = [kk for kk, r in R.items() if r["pre_pass"] == "1" and kk[1] % a.K == k]
                f_ = sum(R[kk]["fell"] == "1" for kk in ks)
                row += f" {f_}/{len(ks)} ({100*f_/max(len(ks),1):.0f} %) |"
            P(row)
    txt = "\n".join(L); print(txt)
    open(a.out, "w", encoding="utf-8").write(txt + "\n"); print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
