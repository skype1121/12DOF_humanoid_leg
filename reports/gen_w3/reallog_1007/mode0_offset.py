"""서기 모드(mode 0) 중 '조용한 양발 서기인데 기준 대비 기울기 ≥ 1.5°' 구간 — 실물에서 조용히 선 자세가 감지기 기준(τ 10 s EMA)에서 얼마나·얼마나 오래 벗어나는가.
(회복 모드 중이었다면 기준이 얼어 있으므로 이 편차가 그대로 '꺼짐 차단'이 된다 — 그 크기·지속의 실측 분포.)"""
import sys, pickle, os
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
FILES = ["rl_bridge_20261004_221209.jsonl", "rl_bridge_20261004_221920.jsonl", "rl_bridge_20261004_223349.jsonl"]
for fn in FILES:
    D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb")); R = L.replay(D, "rv3"); t = D["t"]
    c = D["c"] > 0.5; both = c[:, 0] & c[:, 1]; w = D["wxy"]; tilt = R["tilt"]; m0 = (D["mode"] == 0) & (t >= 1.5)
    q = m0 & (w < 0.3) & both
    print(f"\n== {fn}: 서기 모드 {m0.sum()*0.02:.1f} s · 그중 조용∧양발 {q.sum()*0.02:.1f} s · 조용∧양발 틱의 기준 대비 기울기: 중앙 {np.median(tilt[q]):.2f}° p90 {np.percentile(tilt[q],90):.2f}° p99 {np.percentile(tilt[q],99):.2f}° 최대 {tilt[q].max():.2f}° · ≥1.5° 틱 {100*(tilt[q]>=1.5).mean():.1f} % ({(tilt[q]>=1.5).sum()*0.02:.1f} s) · ≥2.0° {100*(tilt[q]>=2.0).mean():.1f} %")
    rr = L.runs(q & (tilt >= 1.5))
    rr = [(s, e) for (s, e) in rr if (e - s) >= 10]
    for (s, e) in rr:
        d = D["rp"][s:e] - R["ref"][s:e]
        print(f"   t={t[s]:6.2f}–{t[e-1]:6.2f} ({(e-s)*0.02:.2f} s) 기울기 최대 {tilt[s:e].max():.2f}° · 평균 Δ롤 {d[:,0].mean():+.2f} Δ피치 {d[:,1].mean():+.2f} · |ω_xy| 최대 {w[s:e].max():.3f} · 뒤이어 {'회복 켜짐' if (e < D['n'] and D['mode'][min(e+25, D['n']-1)] == 1) else '서기 유지'}")
    # 자세 자체(기준과 무관)의 느린 이동: 조용∧양발 틱의 롤·피치 범위
    print(f"   조용∧양발 틱 자세(편향 보정) 롤 p1–p99 {np.percentile(D['rp'][q,0],1):+.2f}…{np.percentile(D['rp'][q,0],99):+.2f}° · 피치 p1–p99 {np.percentile(D['rp'][q,1],1):+.2f}…{np.percentile(D['rp'][q,1],99):+.2f}° · 기준 이동 범위 롤 {R['ref'][m0,0].min():+.2f}…{R['ref'][m0,0].max():+.2f} 피치 {R['ref'][m0,1].min():+.2f}…{R['ref'][m0,1].max():+.2f}")
