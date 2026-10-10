#!/usr/bin/env python3
"""1010 밤: 걷기 수락 뒤 창의 **요구 토크**(자르기 전) = kp·(목표−실측) − kd·속도, kp 150·kd 5(AK70 관절; 발목 롤 AK45 는 게인이 달라 표에서 뺀다).
입력: transall_<tag>.npz(--trace_qd) + kwloop_<tag>.npz. chk_load.py 의 창 정의 그대로. 사용: demand.py <tag> ..."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import chk_load as C
KP, KD = 150.0, 5.0; AK70 = C.J8; names = [C.JN[j] for j in AK70]
print("| 실행 | 창 | 표본 | 요구 최대 N·m(AK70 8 관절 중 최대) p50/p95/최대 | L_hip_a | R_hip_a | L_knee | R_knee | 0.2 s 안 hip_a 목표−실측 최대 ° p50/p95 | 낙상 |"); print("|---|---|---|---|---|---|---|---|---|---|")
for tag in sys.argv[1:]:
    D = C.load(tag); W = C.windows(D); dem = KP * (D["cmd"] - D["q"]) - KD * D["qd"]
    k = np.load(f"{C.LP}/kwloop_{tag}.npz", allow_pickle=True); ok = k["pre_pass"]; fall = f"{int((k['fell'] & ok).sum())}/{int(ok.sum())}"
    for typ, lab in (("start_inpl", "제자리 시작 1 s"), ("start_fwd", "앞 0.06 시작 1 s"), ("steady_fwd", "앞 0.06 정상")):
        rows = [(e, a, b) for e, ws in W.items() for (n, a, b) in ws if n == typ]
        if not rows: continue
        mx = np.array([np.abs(dem[a:b, e][:, AK70]).max(0) for e, a, b in rows])           # (표본, 8)
        e02 = np.array([np.degrees(np.abs(D["cmd"][a:a + 10, e][:, [2, 3]] - D["q"][a:a + 10, e][:, [2, 3]]).max()) for e, a, b in rows])
        cell = lambda v: "%.0f / %.0f / %.0f" % tuple(np.percentile(v, [50, 95, 100]))
        print(f"| {tag} | {lab} | {len(rows)} | {cell(mx.max(1))} | " + " | ".join(cell(mx[:, names.index(n)]) for n in ("L_hip_a", "R_hip_a", "L_knee", "R_knee")) + f" | {np.percentile(e02, 50):.1f} / {np.percentile(e02, 95):.1f} | {fall} |")
