#!/usr/bin/env python3
"""AIR4 탐색(AIR_START_1008.md §14): 기운 채 매달림에서 공중 정책의 거동 — env 별 매달림 피치(공중 구간 중앙)로 묶어 본다.
사용: air4_tilt.py <log_picture 폴더> <태그> [<태그> ...]"""
import csv, json, os, sys
import numpy as np
LP = sys.argv[1]; EDGES = [-99, -4, -2, 0, 1, 2.5, 4, 6, 99]
def load(tag):
    rows = list(csv.DictReader(open(f"{LP}/ground_recovery_{tag}_envs.csv", encoding="utf-8")))
    tf = np.array([float(r["t_fall_rel"]) if (r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan")) else np.inf for r in rows])
    tr = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True); ka = np.load(f"{LP}/kwair_{tag}.npz", allow_pickle=True)
    ev = json.loads(str(np.load(f"{LP}/kwloop_{tag}.npz", allow_pickle=True)["events"]))
    q = np.degrees(tr["q"]); cont = tr["cont"] > 0.5; alive = tr["alive"].astype(bool); pitch = tr["pitch"]; T, E, J = q.shape
    out = []
    for e in range(E):
        anyc = cont[:, e].any(1); k1 = int(np.argmax(anyc)) if anyc.any() else T; k1 = min(k1, int(alive[:, e].sum()))
        if k1 < 60: continue
        p_h = float(np.median(pitch[25:min(k1, 400), e])); p0 = float(np.median(pitch[5:25, e]))
        exc = float(np.ptp(q[:k1, e], axis=0).max()); dev_end = float(np.abs(q[k1 - 1, e] - q[0, e]).max())
        out.append(dict(p=p_h, p0=p0, exc=exc, dev=dev_end, amax=float(ka["amax"][e]), amax_all=float(ka["amax_all"][e]), hold=bool(ka["hold"][e]), fell=bool(tf[e] <= 14.0 + 30.0), fell_air=bool(tf[e] < 8.0),
                        ho=any(x[1] == "air_to_ke" for x in ev[e]), acc=any(x[1] == "enable" for x in ev[e]), t_air=k1 * 0.02))
    return out
for tag in sys.argv[2:]:
    try: R = load(tag)
    except Exception as ex: print(f"{tag}: 읽기 실패 {ex!r}"); continue
    print(f"\n### {tag} — env {len(R)} (공중 구간이 있는 것)")
    print("| 매달림 피치(공중 중앙) | env | 공중 관절 이동폭 p50 · p90 · 최대 | 접촉 직전 자세 이탈 p50 · p90 | 매달린 동안 \\|a\\| 최대 p50 · p90 · 최대 | \\|a\\| > 6 인 env | 차단 발동 | 낙상(지지 끝 뒤 30 s 까지) | KE 인계 | 걷기 버튼 수락 |"); print("|---|---|---|---|---|---|---|---|---|---|")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        g = [r for r in R if lo <= r["p"] < hi]
        if not g: continue
        ex = np.array([r["exc"] for r in g]); dv = np.array([r["dev"] for r in g]); am = np.array([r["amax"] for r in g]); n = len(g)
        lab = f"{lo:+g} ~ {hi:+g}°".replace("-99 ~ ", "< ").replace(" ~ +99°", "° 이상")
        print(f"| {lab} | {n} | {np.percentile(ex, 50):.0f} · {np.percentile(ex, 90):.0f} · {ex.max():.0f}° | {np.percentile(dv, 50):.0f} · {np.percentile(dv, 90):.0f}° | {np.nanpercentile(am, 50):.1f} · {np.nanpercentile(am, 90):.1f} · {np.nanmax(am):.3g} | {int((am > 6).sum())} | {sum(r['hold'] for r in g)} | {sum(r['fell'] for r in g)} ({100*sum(r['fell'] for r in g)/n:.0f} %) | {sum(r['ho'] for r in g)} | {sum(r['acc'] for r in g)} |")
    P = np.array([r["p"] for r in R]); print(f"- 매달림 피치 범위 {P.min():+.1f} ~ {P.max():+.1f}° (중앙 {np.median(P):+.1f}°) · 전체 낙상 {sum(r['fell'] for r in R)} / {len(R)} · 차단 발동 {sum(r['hold'] for r in R)}")
