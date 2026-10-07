"""X7b 요약(TRANSITION_W3KE.md §5.11c) — 걷기 구간별 낙상(구간 시작 → 다음 걷기 시작 전, 분모 = 구간 시작 때 생존)과 걷기 시작 자세.
사용: python3 reports/gen_w3/x7b_summ.py [--seeds 1,2] [--scen bf,ff,orig] [--prefix tr_x7b]
자세 지표(걷기 켜기 요청 직전 0.5 s 평균): 발 엇갈림(왼 − 오른 발의 골반 기준 앞뒤 거리, cm) · 골반-발 앞뒤 거리 평균(cm, + = 발이 골반보다 앞) · 발 간격(좌우, cm) · 투영 중력 x(앞 기울기, °근사)."""
import argparse, csv, json, os
import numpy as np
from math import comb
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    pr = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)   # noqa: E731
    p0 = pr(a); return sum(pr(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if pr(x) <= p0 + 1e-12)


def load(tag):
    p = os.path.join(LP, f"kwloop_{tag}.npz"); q = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    z = np.load(p, allow_pickle=True); rows = {int(r["env"]): r for r in csv.DictReader(open(q))}
    N = z["pre_pass"].shape[0]; tf = np.full(N, np.inf)
    for e, r in rows.items():
        if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
            tf[e] = float(r["t_fall_rel"])
    script = json.loads(str(z["script"]))
    return dict(z=z, tf=tf, pre=z["pre_pass"].astype(bool), script=script, t=z["t"].astype(float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2"); ap.add_argument("--scen", default="bf,ff,orig"); ap.add_argument("--prefix", default="tr_x7b")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    tab = {}
    for sc in a.scen.split(","):
        segs = None; agg = None; got = []
        for s in seeds:
            D = load(f"{a.prefix}_{sc}_d2_s{s}")
            if D is None:
                continue
            got.append(s); walks = [(tt, c) for (tt, w, c) in D["script"] if w]; t_end = float(D["t"][-1]) + 0.02
            if segs is None:
                segs = walks; agg = [dict(n=0, f=0, st=[], stf=[]) for _ in walks]
            z = D["z"]; has_pos = "ffwd" in z.files and z["ffwd"].shape[0] > 0
            for k, (tt, c) in enumerate(walks):
                t1 = walks[k + 1][0] if k + 1 < len(walks) else t_end
                alive0 = D["pre"] & (D["tf"] >= tt); fell = alive0 & (D["tf"] < t1)
                agg[k]["n"] += int(alive0.sum()); agg[k]["f"] += int(fell.sum())
                if has_pos:
                    k1 = int(np.searchsorted(D["t"], tt - 1e-6)); k0 = max(0, k1 - 25)
                    ff = z["ffwd"][k0:k1].astype(float).mean(0); fl = z["flat"][k0:k1].astype(float).mean(0); pg = z["pg"][k0:k1].astype(float).mean(0)
                    feat = np.stack([100 * (ff[:, 0] - ff[:, 1]), 100 * ff.mean(1), 100 * (fl[:, 0] - fl[:, 1]), np.degrees(np.arcsin(np.clip(pg[:, 0], -1, 1)))], 1)
                    agg[k]["st"].append(feat[alive0 & ~fell]); agg[k]["stf"].append(feat[fell])
        if segs is None:
            print(f"## {sc}: 자료 없음"); continue
        tab[sc] = agg
        print(f"## 대본 {sc} (seed {got})")
        print("| 걷기 | 시작 s · 명령 | 시작 때 생존 | 구간 낙상 | 시작 자세(생존 env 중앙): 발 엇갈림 cm(|·| 중앙) · 골반-발 앞뒤 cm · 발 간격 cm · 앞 기울기 ° | 낙상 env 의 같은 값(중앙) |"); print("|---|---|---|---|---|---|")
        for k, (tt, c) in enumerate(segs):
            g = agg[k]; st = np.concatenate(g["st"]) if g["st"] else np.zeros((0, 4)); sf = np.concatenate(g["stf"]) if g["stf"] else np.zeros((0, 4))
            d = (lambda M: f"{np.median(M[:,0]):+.1f} ({np.median(np.abs(M[:,0])):.1f}) · {np.median(M[:,1]):+.1f} · {np.median(M[:,2]):.1f} · {np.median(M[:,3]):+.2f}" if len(M) else "—")
            print(f"| {k+1} | {tt:g} · {c} | {g['n']} | {g['f']} ({100*g['f']/max(g['n'],1):.1f} %) | {d(st)} | {d(sf)} (n {len(sf)}) |")
    if "ff" in tab and len(tab["ff"]) >= 5:
        g = tab["ff"]; lo = (g[0]["f"] + g[1]["f"], g[0]["n"] + g[1]["n"]); hi = (g[3]["f"] + g[4]["f"], g[3]["n"] + g[4]["n"])
        print(f"- H8(누적): S_ff 걷기 1·2 낙상 {lo[0]}/{lo[1]} 대 걷기 4·5 낙상 {hi[0]}/{hi[1]} · Fisher 양측 p {fisher(lo[0], lo[1]-lo[0], hi[0], hi[1]-hi[0]):.3f} → {'기각(4·5 ≤ 1·2)' if hi[0] <= lo[0] else '방향 일치(단측 p 는 양측의 절반 근처)'}")
    if "bf" in tab and "ff" in tab:
        b = tab["bf"][1]; f2 = tab["ff"][1]
        print(f"- H7(직전 뒤 걷기): S_bf 앞 걷기 낙상 {b['f']}/{b['n']} ({100*b['f']/max(b['n'],1):.1f} %) 대 S_ff 둘째 걷기 {f2['f']}/{f2['n']} ({100*f2['f']/max(f2['n'],1):.1f} %) → {'기각(< 1 %)' if 100*b['f']/max(b['n'],1) < 1.0 else ('지지 조건 충족' if 100*b['f']/max(b['n'],1) >= 2.0 and b['f']/max(b['n'],1) > f2['f']/max(f2['n'],1) else '미결')}")


if __name__ == "__main__":
    main()
