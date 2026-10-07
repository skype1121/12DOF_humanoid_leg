"""기록된 회복 활성화 안의 '조용한 구간'(원 자이로 |ω_xy|<0.3 ∧ 접촉 한 발 이상, 연속 ≥ 0.5 s) 전부 — 그때 자세가 기준에서 얼마나 떨어져 있었나."""
import sys, pickle, os
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
FILES = [("rl_bridge_20261004_221209.jsonl", "시험2"), ("rl_bridge_20261004_221920.jsonl", "시험3(무효)"), ("rl_bridge_20261004_223349.jsonl", "시험4")]
rows = []
for fn, lab in FILES:
    D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb")); R = L.replay(D, "rv3"); t = D["t"]
    c = D["c"] > 0.5; both = c[:, 0] & c[:, 1]; anyc = c[:, 0] | c[:, 1]; w = D["wxy"]; tilt = R["tilt"]
    for a, (i0, i1) in enumerate(L.activations(D["mode"])):
        for (s, e) in L.runs((w[i0:i1] < 0.3) & anyc[i0:i1]):
            if (e - s) < 25:
                continue
            sl = slice(i0 + s, i0 + e); d = D["rp"][sl] - R["ref"][sl]
            rows.append((lab, a + 1, t[i0 + s], (e - s) * 0.02, both[sl].mean(), np.median(tilt[sl]), (tilt[sl] >= 1.5).mean(), tilt[sl].max(), np.median(d[:, 0]), np.median(d[:, 1]), np.median(w[sl])))
print("시험 · 활성화 · 시작 t · 길이 s · 양발 비율 · 기준 대비 기울기 중앙° · ≥1.5° 틱 비율 · 최대° · Δ롤 중앙 · Δ피치 중앙 · |ω_xy| 중앙")
for r in rows:
    print(f"{r[0]:<9} {r[1]}  {r[2]:6.2f}  {r[3]:5.2f}  {100*r[4]:3.0f}%  {r[5]:5.2f}  {100*r[6]:3.0f}%  {r[7]:5.2f}  {r[8]:+5.2f}  {r[9]:+5.2f}  {r[10]:.3f}")
n = len(rows); big = sum(1 for r in rows if r[5] >= 1.5)
tot = sum(r[3] for r in rows); tot15 = sum(r[3] * r[6] for r in rows)
print(f"조용한 구간 {n} 개(합 {tot:.1f} s) 중 기울기 중앙 ≥ 1.5° 인 구간 {big} 개 · 조용한 틱 가운데 ≥1.5° 인 시간 {tot15:.2f} s ({100*tot15/tot:.0f} %)")
