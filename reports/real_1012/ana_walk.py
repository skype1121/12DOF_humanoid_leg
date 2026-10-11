#!/usr/bin/env python3
"""실물 1012 03:53 제자리걸음 시도 분석: 걷기 켠 뒤 구간의 섞기·접촉·몸통 자세/요·요구 토크·추정 토크. 사용: ana_walk.py <블랙박스.jsonl>"""
import json, sys, math, numpy as np
rows = []; t_on = t_off = None
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except Exception: continue
    if d.get("event") == "w3_step" and d.get("enable"): t_on = d["timestamp"]
    if d.get("event") == "stop_all": t_off = d["timestamp"]
    if "qpos_rad" in d and "event" not in d: rows.append(d)
J = list(rows[0]["targets_deg_joint"].keys()); print("관절 순서", [j.replace("_joint", "") for j in J])
t = np.array([r["timestamp"] for r in rows]); 
def col(f): return np.array([f(r) for r in rows], dtype=float)
tg = np.array([[r["targets_deg_joint"][j] for j in J] for r in rows])
q = np.degrees(np.array([r["qpos_rad"] for r in rows])); qd = np.array([r["qvel_rad_s"] for r in rows])
print("qpos 길이", q.shape, "키 예", {k: (type(rows[0][k]).__name__) for k in ("contact", "foot_n", "rp_deg", "tau_est_nm", "s8_cmd", "kew3", "gyro_rad_s")})
ct = np.array([r["contact"] for r in rows], dtype=float); rp = np.array([r["rp_deg"] for r in rows], dtype=float)
qu = np.array([r["quat_wxyz"] for r in rows], dtype=float); yaw = np.degrees(np.arctan2(2 * (qu[:, 0] * qu[:, 3] + qu[:, 1] * qu[:, 2]), 1 - 2 * (qu[:, 2] ** 2 + qu[:, 3] ** 2)))
gy = np.array([r["gyro_rad_s"] for r in rows], dtype=float)
al = np.array([(r.get("kew3") or {}).get("blend_a", np.nan) for r in rows]); md = [(r.get("kew3") or {}).get("mode") for r in rows]
tau = np.array([r["tau_est_nm"] if isinstance(r.get("tau_est_nm"), list) else [np.nan] * 12 for r in rows], dtype=float)
fn = np.array([r["foot_n"] if isinstance(r.get("foot_n"), list) else [np.nan, np.nan] for r in rows], dtype=float)
# 관절각 순서가 targets 와 같은지: 서기 구간에서 target−q 평균이 작은지 본다
i_on = int(np.searchsorted(t, t_on)); i_off = int(np.searchsorted(t, t_off)) if t_off else len(t)
print(f"틱 {len(t)} · 걷기 켬 t={t_on - t[0]:.2f}s(틱 {i_on}) · 정지 t={(t_off or t[-1]) - t[0]:.2f}s · 걷기 길이 {(t[i_off - 1] - t_on):.2f}s · 틱 간격 중앙 {np.median(np.diff(t)) * 1000:.1f} ms")
st = slice(max(0, i_on - 500), i_on)
print(f"[서기 직전 10 s] 롤 {rp[st, 0].mean():+.2f}±{rp[st, 0].std():.2f}° 피치 {rp[st, 1].mean():+.2f}±{rp[st, 1].std():.2f}° · 요 변화 {yaw[i_on - 1] - yaw[st][0]:+.2f}° · 접촉 L/R {ct[st, 0].mean():.2f}/{ct[st, 1].mean():.2f} · |목표−각| 최대 {np.abs(tg[st] - q[st]).max():.1f}°")
err = tg - q; dem = 150.0 * np.radians(err) - 5.0 * qd
names = [j.replace("_joint", "") for j in J]
print(f"[서기 전체] 요구 |kp·e − kd·v| 관절별 P99:", {n: round(float(np.percentile(np.abs(dem[100:i_on, k]), 99)), 1) for k, n in enumerate(names) if "hip_r" not in n})
print("t[s]  α    접촉L R  발N L/R        롤    피치   요Δ    ωz    목표Δ(0.02s)max  요구:Lhip_a Rhip_a Lknee Rknee Lank_r Rank_r   추정τ:Rhip_a")
ia = [names.index(x) for x in ("left_hip_a", "right_hip_a", "left_knee", "right_knee", "left_ankle_r", "right_ankle_r")]
for i in range(i_on - 5, i_off, 5):
    dtg = np.abs(tg[i] - tg[i - 1]); dtg[[names.index("left_hip_r"), names.index("right_hip_r")]] = 0
    print(f"{t[i] - t_on:5.2f} {al[i]:4.2f}  {int(ct[i, 0])} {int(ct[i, 1])}   {fn[i, 0]:6.1f}/{fn[i, 1]:6.1f}  {rp[i, 0]:+5.1f} {rp[i, 1]:+5.1f} {yaw[i] - yaw[i_on]:+6.1f} {gy[i, 2]:+5.2f}   {dtg.max():4.1f}          " + " ".join(f"{dem[i, k]:+6.1f}" for k in ia) + f"     {tau[i, ia[1]]:+6.1f}")
w = slice(i_on, i_off)
print("[걷기 구간] 요구 최대:", {n: round(float(np.abs(dem[w, k]).max()), 1) for k, n in enumerate(names) if "hip_r" not in n})
print("[걷기 구간] 추정 토크 최대:", {n: round(float(np.nanmax(np.abs(tau[w, k]))), 1) for k, n in enumerate(names) if "hip_r" not in n})
print(f"[걷기 구간] 0.2 s 목표 변화 최대 {np.abs(np.delete(tg[i_on + 10] - tg[i_on - 1], [4, 5])).max():.1f}° · 요 변화 {yaw[i_off - 1] - yaw[i_on]:+.1f}° · 롤 범위 {rp[w, 0].min():+.1f}~{rp[w, 0].max():+.1f}° · 피치 {rp[w, 1].min():+.1f}~{rp[w, 1].max():+.1f}°")
print(f"[걷기 구간] 접촉 비율 L {ct[w, 0].mean():.2f} R {ct[w, 1].mean():.2f} · 양발 무접촉 틱 {int(((ct[w, 0] < .5) & (ct[w, 1] < .5)).sum())} · 피드백 나이 최대 {max(r.get('fb_age_max_s') or 0 for r in rows[i_on:i_off]):.3f}s")
print("마지막 정지 사유:", rows[i_off - 1].get("stop_reason"), "| 가드:", rows[i_off - 1].get("osc_tripped"), "| 소프트 클램프 합:", rows[i_off - 1].get("soft_clamp_total"))
