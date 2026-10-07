"""X2e 추적 분석(TRANSITION_W3KE.md §5.6c) — 감지기 기준 자세 오염 가설(H1) 판정용.
입력: log_picture/transall_tr_{latchd3,latchdT,restartT}_<cond>_d2_s*.npz (사본 평가기 --trace_all 128).
env 별(사전검사 통과 ∧ 램프 시작 시 생존): 램프 직전 기준 오차(기준 피치 − 직전 1 s 감지 피치 평균)·기준 롤 오차 · 램프 뒤 회복 모드 첫 켜짐 지연 ·
램프 직전 자세(참 피치, 두 발의 골반 대비 앞축 평균·앞뒤 어긋남, 발 좌우 간격, 골반 높이) · 낙상.
사용: python3 reports/gen_w3/x2e_trace.py [--cond sf15] [--t0 16.0]
"""
import argparse, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LP = os.path.join(REPO, "log_picture")
ARMS = [("restartT", "restart(정착 뒤, 기준 창 1.5 s)"), ("latchdT", "X2d(래치 직후, 기준 창 1.5 s)"), ("latchd3", "X2e(래치 직후, 기준 창 3.0 s)")]


def load(arm, cond, t0):
    out = {}
    for p in sorted(glob.glob(os.path.join(LP, f"transall_tr_{arm}_{cond}_d2_s*.npz"))):
        s = int(re.search(r"_s(\d+)\.npz", p).group(1)); z = np.load(p, allow_pickle=True)
        t = z["t"].astype(np.float64); k0 = int(np.searchsorted(t, t0 - 1e-6)); k1 = int(np.searchsorted(t, t0 - 1.0 - 1e-6))
        alive = z["alive"].astype(bool); pre = z["pre_pass"].astype(bool); mode = z["mode"].astype(bool); ke = z["ke"].astype(bool)
        ref = z["ref"].astype(np.float64); rp2 = z["rp2"].astype(np.float64)       # (T, N, 2) = (롤, 피치)[°]
        ffwd = z["ffwd"].astype(np.float64); flat = z["flat"].astype(np.float64); h = z["h"].astype(np.float64); pit = z["pitch"].astype(np.float64)
        tsw = z["tr_sw"].astype(np.float64)
        for e in range(alive.shape[1]):
            if not pre[e] or not alive[k0 - 1, e] or not ke[k0 - 1, e]:
                continue
            on = np.nonzero(mode[k0:, e])[0]
            # 램프 직전 이미 회복 모드면 지연 0 으로 치지 않고 따로 표시
            pre_on = bool(mode[k0 - 1, e])
            lat = (float(t[k0 + on[0]] - t0) if len(on) else float("nan"))
            fell = bool(z["fell"][e]) and float(z["t_fall_post"][e]) >= t0 - 1e-6
            out[(s, e)] = dict(ref_p=ref[k0 - 1, e, 1], ref_r=ref[k0 - 1, e, 0], det_p=float(np.nanmean(rp2[k1:k0, e, 1])), det_r=float(np.nanmean(rp2[k1:k0, e, 0])),
                               lat=lat, pre_on=pre_on, fell=fell, pitch=float(np.nanmean(pit[k1:k0, e])),
                               fmean=100 * float(np.nanmean(ffwd[k1:k0, e, :])), stag=100 * float(np.nanmean(ffwd[k1:k0, e, 0] - ffwd[k1:k0, e, 1])),
                               sep=100 * float(np.nanmean(flat[k1:k0, e, 0] - flat[k1:k0, e, 1])), h=float(np.nanmean(h[k1:k0, e])), tsw=float(tsw[e]),
                               ref_cap_p=float("nan"))
    return out


def med(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return (float("nan"), float("nan"), float("nan")) if len(v) == 0 else (float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", default="sf15"); ap.add_argument("--t0", type=float, default=16.0)
    a = ap.parse_args()
    D = {arm: load(arm, a.cond, a.t0) for arm, _ in ARMS}
    print(f"## X2e 추적 — 조건 {a.cond}, 외란 시작 {a.t0:.1f} s")
    print("| 팔 | n | 낙상 | 기준 오차 피치[°] 중앙 (10–90 %) | |기준 오차| 중앙 | 기준 오차 롤 중앙 | 램프 뒤 회복 켜짐 지연[s] 중앙 (10–90 %) | 안 켜짐 | 직전 이미 켜짐 | 참 피치[°] | 발 앞축 평균[cm] | 앞뒤 어긋남 |중앙|[cm] | 좌우 간격[cm] | 골반 높이[m] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, name in ARMS:
        d = D[arm]
        if not d:
            print(f"| {name} | 0 | (자료 없음) |"); continue
        v = list(d.values()); n = len(v); f = sum(x["fell"] for x in v)
        ep = [x["ref_p"] - x["det_p"] for x in v]; er = [x["ref_r"] - x["det_r"] for x in v]
        m, lo, hi = med(ep); ml, llo, lhi = med([x["lat"] for x in v if not x["pre_on"]])
        print(f"| {name} | {n} | {f} ({100*f/n:.1f} %) | {m:+.2f} ({lo:+.2f}…{hi:+.2f}) | {med(np.abs(ep))[0]:.2f} | {med(er)[0]:+.2f} | {ml:.2f} ({llo:.2f}–{lhi:.2f}) | "
              f"{sum(1 for x in v if not np.isfinite(x['lat']))} | {sum(x['pre_on'] for x in v)} | {med([x['pitch'] for x in v])[0]:+.2f} | {med([x['fmean'] for x in v])[0]:+.2f} | "
              f"{med(np.abs([x['stag'] for x in v]))[0]:.2f} | {med([x['sep'] for x in v])[0]:.1f} | {med([x['h'] for x in v])[0]:.3f} |")
    # 짝 비교
    for (a1, n1), (a2, n2) in ((ARMS[1], ARMS[2]), (ARMS[0], ARMS[1]), (ARMS[0], ARMS[2])):
        ks = sorted(set(D[a1]) & set(D[a2]))
        if not ks:
            continue
        b = sum(1 for k in ks if D[a1][k]["fell"] and not D[a2][k]["fell"]); c = sum(1 for k in ks if D[a2][k]["fell"] and not D[a1][k]["fell"])
        de = np.array([abs(D[a1][k]["ref_p"] - D[a1][k]["det_p"]) - abs(D[a2][k]["ref_p"] - D[a2][k]["det_p"]) for k in ks])
        dl = np.array([D[a1][k]["lat"] - D[a2][k]["lat"] for k in ks if not D[a1][k]["pre_on"] and not D[a2][k]["pre_on"]]); dl = dl[np.isfinite(dl)]
        print(f"- 짝 {a1} vs {a2} (n {len(ks)}): 낙상 {a1}만 {b} / {a2}만 {c} (McNemar p {mcnemar(b, c):.3g}) · |기준 오차| 차 중앙 {np.median(de):+.2f}° · 켜짐 지연 차 중앙 {np.median(dl) if len(dl) else float('nan'):+.3f} s (n {len(dl)})")
    # 팔 안에서: 기준 오차(피치) 부호·크기별 낙상률, 켜짐 지연
    for arm, name in ARMS:
        d = D[arm]
        if not d:
            continue
        v = list(d.values()); ep = np.array([x["ref_p"] - x["det_p"] for x in v]); fl = np.array([x["fell"] for x in v]); lat = np.array([x["lat"] for x in v])
        print(f"- {name}: 기준 오차 구간별 낙상률·켜짐 지연 중앙 — " + " · ".join(
            f"[{lo:+.0f},{hi:+.0f})° n {int(m.sum())} 낙상 {100*fl[m].mean():.0f} % 지연 {np.nanmedian(lat[m]):.2f}s"
            for lo, hi in ((-9, -1), (-1, -0.3), (-0.3, 0.3), (0.3, 1), (1, 9)) for m in [(ep >= lo) & (ep < hi)] if m.sum() >= 5))


if __name__ == "__main__":
    main()
