#!/usr/bin/env python3
"""롤 기울기에 대한 정책 보정 응답 분석 (0913, 승윤님 지시: '더 움직이는지'가 아니라 필요한 순간 적절히 보정하고 추종하는지).

입력: 브리지 블랙박스(rl_bridge_*.jsonl) 또는 시뮬 덤프(simbb_*_env*.jsonl) — 같은 키(rp_deg, qpos_rad, targets_deg_joint, contact, foot_n, hang_state).
창: 접지 확정(hang_state CONTACT*) 부터 낙상/종료까지. 보정기가 있으면 정책 출력은 targets_deg_joint_pre_rc.
지표:
  gain_asym  = d(발목롤 비대칭 모드 (L−R)/2)/d(roll)  [°/°]  — 좌우 축이 반대(왼 +X·오른 −X)라 (L−R)/2 가 '두 발 같은 세계 방향' = 진짜 롤 대응 모드
  gain_sym   = d((L+R)/2)/d(roll)                       [°/°]  — 대칭(안쪽/바깥쪽) 모드
  gain_hip   = d(힙롤 (L−R)/2)/d(roll)                   [°/°]  — 힙 롤 축은 좌우 같은 부호라 (L−R)/2 가 대칭 모드, (L+R)/2 가 같은 방향 모드 → 둘 다 보고
  latency_s  = |roll| 가 1° 넘은 뒤 발목롤 비대칭 목표가 사전 평균에서 0.5° 이상 벗어날 때까지
  track_rms  = 목표−측정 RMS [°] (발목롤·힙롤), lag_s = 목표 변화가 측정에 나타나는 상호상관 지연
  survive_s  = 접지 확정 뒤 양발 이탈까지, left_off_s = 왼발 비트 0 까지
회귀는 |roll| 0.5~8° 구간(접지 뒤)에서 최소제곱. 표본 <50틱이면 '-'.
"""
import argparse, json, math, sys
JN = ["left_hip_f_joint","right_hip_f_joint","left_hip_a_joint","right_hip_a_joint","left_hip_r_joint","right_hip_r_joint",
      "left_knee_joint","right_knee_joint","left_ankle_f_joint","right_ankle_f_joint","left_ankle_r_joint","right_ankle_r_joint"]
def _tg(r):
    t = r.get("targets_deg_joint_pre_rc")
    if isinstance(t, list): return t
    t = r["targets_deg_joint"]
    return [t[n] for n in JN] if isinstance(t, dict) else list(t)
def _slope(xs, ys):
    n = len(xs)
    if n < 50: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    return sxy/sxx if sxx > 1e-9 else None
def _xcorr_lag(a, b, dt, max_lag_s=0.6):
    """b 가 a 를 얼마나 늦게 따르는지(초). 표본 부족이면 None."""
    n = min(len(a), len(b))
    if n < 100: return None
    a = a[:n]; b = b[:n]; ma = sum(a)/n; mb = sum(b)/n; a = [x-ma for x in a]; b = [x-mb for x in b]
    best, bl = -1e18, 0
    for lag in range(0, int(max_lag_s/dt)+1):
        c = sum(a[i]*b[i+lag] for i in range(n-lag))
        if c > best: best, bl = c, lag
    return bl*dt
def analyze(recs, dt=0.02):
    recs = [r for r in recs if isinstance(r.get("rp_deg"), list) and isinstance(r.get("qpos_rad"), list) and isinstance(r.get("contact"), list)]
    if not recs: return {"error": "no records"}
    ic = next((i for i, r in enumerate(recs) if str(r.get("hang_state", "")).startswith("CONTACT")), None)
    if ic is None:  # 시뮬 덤프엔 hang_state 가 없을 수 있음 → 양발 접촉 첫 시각
        ic = next((i for i, r in enumerate(recs) if r["contact"][0] > 0.5 and r["contact"][1] > 0.5), None)
    if ic is None: return {"error": "no contact"}
    t0 = recs[ic]["timestamp"]
    seg = recs[ic:]
    iend = next((i for i, r in enumerate(seg) if r["contact"][0] < 0.5 and r["contact"][1] < 0.5), None)
    survive = (seg[iend]["timestamp"]-t0) if iend is not None else (seg[-1]["timestamp"]-t0)
    il = next((r["timestamp"]-t0 for r in seg if r["contact"][0] < 0.5 and r["contact"][1] > 0.5), None)
    win = seg[:iend] if iend is not None else seg
    roll = [r["rp_deg"][0] for r in win]
    T = [_tg(r) for r in win]; Q = [[math.degrees(x) for x in r["qpos_rad"]] for r in win]
    ank_asym_t = [(t[10]-t[11])/2 for t in T]; ank_sym_t = [(t[10]+t[11])/2 for t in T]
    hip_asym_t = [(t[2]-t[3])/2 for t in T]; hip_sym_t = [(t[2]+t[3])/2 for t in T]
    ank_asym_q = [(q[10]-q[11])/2 for q in Q]
    # 사전 평균(접지 뒤 첫 1 s)
    n0 = min(len(win), 50); pre_roll = sum(roll[:n0])/n0; pre_aa = sum(ank_asym_t[:n0])/n0
    sel = [i for i in range(len(win)) if 0.5 <= abs(roll[i]-pre_roll) <= 8.0]
    xs = [roll[i]-pre_roll for i in sel]
    out = {"n_ticks": len(win), "survive_s": round(survive, 2), "left_off_s": (round(il, 2) if il is not None else None),
           "roll_pre": round(pre_roll, 2), "roll_min": round(min(roll), 2), "roll_max": round(max(roll), 2),
           "gain_ank_asym": _slope(xs, [ank_asym_t[i]-pre_aa for i in sel]),
           "gain_ank_sym": _slope(xs, [ank_sym_t[i] for i in sel]),
           "gain_hip_asym": _slope(xs, [hip_asym_t[i] for i in sel]),
           "gain_hip_sym": _slope(xs, [hip_sym_t[i] for i in sel])}
    # 지연: |Δroll|>1° 이후 비대칭 목표 변화 0.5° 이상
    i1 = next((i for i in range(len(win)) if abs(roll[i]-pre_roll) > 1.0), None)
    i2 = next((i for i in range(i1, len(win)) if abs(ank_asym_t[i]-pre_aa) > 0.5), None) if i1 is not None else None
    out["latency_s"] = (round((i2-i1)*dt, 2) if (i1 is not None and i2 is not None) else None)
    out["roll_rate_max_deg_s"] = round(min((roll[i+25]-roll[i])/(25*dt) for i in range(0, len(roll)-25)), 2) if len(roll) > 30 else None
    # 추종
    err_a = [T[i][10]-Q[i][10] for i in range(len(win))] + [T[i][11]-Q[i][11] for i in range(len(win))]
    err_h = [T[i][2]-Q[i][2] for i in range(len(win))] + [T[i][3]-Q[i][3] for i in range(len(win))]
    out["track_rms_ankle_r"] = round(math.sqrt(sum(e*e for e in err_a)/max(1, len(err_a))), 2)
    out["track_rms_hip_a"] = round(math.sqrt(sum(e*e for e in err_h)/max(1, len(err_h))), 2)
    out["track_lag_s_ankle_asym"] = _xcorr_lag(ank_asym_t, ank_asym_q, dt)
    for k in ("gain_ank_asym", "gain_ank_sym", "gain_hip_asym", "gain_hip_sym"):
        if out[k] is not None: out[k] = round(out[k], 3)
    return out
def _selftest():
    # 합성: 롤이 2 s 동안 0→6° 로 늘고 목표가 이득 0.5 °/° 로 반대로 움직이며 측정이 3틱 늦게 따름
    dt = 0.02; recs = []
    for i in range(400):
        t = i*dt; roll = 0.0 if t < 1.0 else min(6.0, (t-1.0)*3.0)
        tg = [0.0]*12; tg[10] = -0.5*roll; tg[11] = +0.5*roll      # (L−R)/2 = −0.5·roll
        q = [0.0]*12
        if i >= 3: q[10] = math.radians(-0.5*(0.0 if (i-3)*dt < 1.0 else min(6.0, ((i-3)*dt-1.0)*3.0))); q[11] = -q[10]
        recs.append({"timestamp": t, "rp_deg": [roll, 0.0], "qpos_rad": q, "targets_deg_joint": tg, "contact": [1.0, 1.0], "foot_n": [80, 80], "hang_state": "CONTACT_ACQUIRED"})
    o = analyze(recs, dt)
    assert abs(o["gain_ank_asym"] + 0.5) < 0.02, o
    assert o["latency_s"] is not None and o["latency_s"] <= 0.1, o
    assert o["track_lag_s_ankle_asym"] is not None and abs(o["track_lag_s_ankle_asym"] - 0.06) < 0.025, o
    assert o["survive_s"] > 7.9, o
    print("[selftest] analyze_roll_response: 이득 −0.5 °/° · 지연 ≤0.1 s · 추종 지연 0.06 s · 생존 — PASS")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--dt", type=float, default=0.02)
    a = ap.parse_args()
    if a.selftest: _selftest(); sys.exit(0)
    print("| 파일 | 생존 s | 왼발 이탈 s | 롤 사전/최소 | 최대 롤속도 °/s | 이득 발목 비대칭 °/° | 발목 대칭 | 힙 비대칭 | 힙 대칭 | 지연 s | 추종 RMS 발목/힙 ° | 추종 lag s |")
    for f in a.files:
        recs = []
        for l in open(f, encoding="utf-8", errors="replace"):
            try: recs.append(json.loads(l))
            except Exception: pass
        o = analyze(recs, a.dt)
        if "error" in o: print(f"| {f} | {o['error']} |"); continue
        print(f"| {f.split('/')[-1]} | {o['survive_s']} | {o['left_off_s']} | {o['roll_pre']}/{o['roll_min']} | {o['roll_rate_max_deg_s']} | {o['gain_ank_asym']} | {o['gain_ank_sym']} | {o['gain_hip_asym']} | {o['gain_hip_sym']} | {o['latency_s']} | {o['track_rms_ankle_r']}/{o['track_rms_hip_a']} | {o['track_lag_s_ankle_asym']} |")
