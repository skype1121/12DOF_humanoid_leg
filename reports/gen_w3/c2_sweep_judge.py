"""정지 시점 스윕 판정(CPU, 1006 문헌 권고 3) — c2_landscape arm X(stop_td_k = 1..4 / stop_oracle) × cue(B3 형 / B6 형).
사용: python3 c2_sweep_judge.py <태그접두> <seed...>   입력 log_picture/c2unit_<접두>_{B3,B6}_{k1,k2,k3,k4,or}_s<seed>.npz
사전 기준(실행 전 기록, C2_DESIGN §8b): 충격·램프 각각, 고정 k 중 최선의 외란 창 낙상률이 오라클 대비 +3 %p 이내(점 추정)이고
  짝(같은 env 번호) 부트스트랩 95 % 상한 ≤ +5 %p 이면 '고정 N 보 정지로 속도 추정기 의존 제거 가능'."""
import sys, os
import numpy as np
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
pfx = sys.argv[1]; seeds = sys.argv[2:] or ["1"]
STOPS = ("k1", "k2", "k3", "k4", "or", "ov")
REF = os.environ.get("SWEEP_REF", "ov")      # 1006 밤: 기준 오라클 = 속도 오라클(ov) — 종전 or(지지영역 포함)은 언제나 첫 착지로 약함


def load(cue, stp):
    parts = []
    for sd in seeds:
        p = f"{LP}/c2unit_{pfx}_{cue}_{stp}_s{sd}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); parts.append({k: z[k] for k in z.files})
    if not parts:
        return None
    keys = [k for k in parts[0] if parts[0][k].ndim == 1 and parts[0][k].shape[0] > 1]
    return {k: np.concatenate([q[k] for q in parts]) for k in keys}


rng = np.random.default_rng(0)
L = [f"정지 시점 스윕 · {pfx} · seed {' '.join(seeds)}"]
verdict = {}
for cue in ("B3", "B6"):
    R = {s: load(cue, s) for s in STOPS}
    R = {s: v for s, v in R.items() if v is not None}
    if REF not in R:
        L.append(f"[{cue}] 기준 오라클 {REF} 없음"); continue
    base = R[REF]; cat = base["cat"]
    valid = np.ones_like(cat, dtype=bool)
    for v in R.values():
        valid &= v["fell_pre"] < 0.5
    for lab, m in (("충격", (cat > 0.5) & (base["kind0"] > 1.5) & valid), ("램프", (cat > 0.5) & (base["kind0"] > 0.5) & (base["kind0"] < 1.5) & valid)):
        n = int(m.sum())
        if n < 5:
            continue
        fo = R[REF]["fell_w"][m]
        row = []
        best = None
        for s in [x for x in STOPS if x in R]:
            X = R[s]; f = X["fell_w"][m]
            d = f - fo; idx = np.arange(n); bs = np.array([d[rng.choice(idx, n)].mean() for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.append(f"{s}: 낙상 {100*f.mean():5.1f} % (Δ오라클 {100*d.mean():+5.1f} %p, 95 % {100*lo:+.1f}~{100*hi:+.1f}) · 정착 {X['settle_ev_w'][m].mean():.2f} · 요청 시 착지 {np.nanmedian(np.where(X['req_w'][m] + X['timeout_w'][m] > 0, X['n_td_req'][m], np.nan)) if (X['req_w'][m] + X['timeout_w'][m]).sum() else float('nan'):.0f} · 시간 초과 {X['timeout_w'][m].mean():.2f} · 표류 {np.nanmedian(X['drift_w'][m]):.0f} cm")
            if s not in ("or", "ov") and (best is None or d.mean() < best[1]):
                best = (s, d.mean(), hi)
        ok = best is not None and best[1] <= 0.03 and best[2] <= 0.05
        verdict[(cue, lab)] = (ok, best)
        L.append(f"[{cue} · {lab} env {n}] 최선 고정 {best[0]} Δ {100*best[1]:+.1f} %p (95 % 상한 {100*best[2]:+.1f}) → {'PASS(고정 N 보로 충분)' if ok else 'FAIL(상태 정보 필요)'}")
        L += ["    " + r for r in row]
L.append("종합: " + " · ".join(f"{c}/{l} {'PASS' if v[0] else 'FAIL'}" for (c, l), v in verdict.items()))
txt = "\n".join(L); print(txt)
open(f"{LP}/c2sweep_judge_{pfx}.txt", "w").write(txt + "\n")
