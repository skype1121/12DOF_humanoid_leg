#!/usr/bin/env python3
"""세션 타임라인에서 한 시각 구간의 관절 피드백(나이·온도 원값·각도·토크)을 관절별로 요약한다(로컬 파일만).
사용: ana_tl.py <timeline.jsonl> <시작 epoch> <끝 epoch> [기준 epoch(정지 시각)]"""
import json, sys, numpy as np
P, A, Bq = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); REF = float(sys.argv[4]) if len(sys.argv) > 4 else A
ORD = ["left_hip_f","left_hip_a","left_hip_r","left_knee","left_ankle_f","left_ankle_r","right_hip_f","right_hip_a","right_hip_r","right_knee","right_ankle_f","right_ankle_r"]
rows = []; other = []
for ln in open(P, "rb"):
    try: r = json.loads(ln.decode("utf-8", "replace"))
    except Exception: continue
    ts = r.get("ts")
    if ts is None or ts < A or ts > Bq: continue
    (rows if r.get("kind") == "s" else other).append(r)
print(f"s 행 {len(rows)}  그 밖 {len(other)}  키 예: {[k for k in rows[0].keys() if k!='joints']}")
for r in other: print(f"  {r['ts']-REF:+8.2f}s {r.get('kind')}: " + json.dumps({k: v for k, v in r.items() if k not in ('kind','t','ts','request_id')}, ensure_ascii=False)[:230])
T = np.array([r["ts"] - REF for r in rows])
def col(j, k):
    out = []
    for r in rows:
        v = (r.get("joints") or {}).get(j + "_joint", {}).get(k)
        out.append(np.nan if v is None else float(v))
    return np.array(out)
print("관절(모터)        | 나이 최대 s @t | 나이>0.1 s 구간(t 기준)          | 온도원값 전→후(최소) | 각도 전→후 Δ° | stale/invalid 행수")
for i, j in enumerate(ORD):
    age = col(j, "feedback_age_sec"); tmp = col(j, "feedback_temperature_raw"); ang = col(j, "latest_actual_deg")
    st = np.array([bool((r.get("joints") or {}).get(j + "_joint", {}).get("feedback_stale")) for r in rows])
    iv = np.array([not bool((r.get("joints") or {}).get(j + "_joint", {}).get("feedback_valid", True)) for r in rows])
    bad = age > 0.1; seg = ""
    if bad.any():
        idx = np.flatnonzero(bad); cuts = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[cuts + 1]]); ends = np.concatenate([idx[cuts], [idx[-1]]])
        seg = " ".join(f"[{T[a]:+.2f}..{T[b]:+.2f}]" for a, b in zip(starts, ends))
    pre = T < -0.3; post = T > (T[bad][-1] + 0.3 if bad.any() else 0.5)
    a0 = np.nanmedian(ang[pre][-10:]) if pre.any() else np.nan; a1 = np.nanmedian(ang[post][:10]) if post.any() else np.nan
    t0 = np.nanmedian(tmp[pre][-10:]) if pre.any() else np.nan; t1 = np.nanmedian(tmp[post][:10]) if post.any() else np.nan
    print(f"{j:14s}({i+1:2d}) | {np.nanmax(age):6.2f} @{T[np.nanargmax(age)]:+6.2f} | {seg:32s} | {t0:5.0f}→{t1:5.0f} (min {np.nanmin(tmp):.0f}) | {a0:+7.2f}→{a1:+7.2f} Δ{a1-a0:+7.2f} | {int(st.sum())}/{int(iv.sum())}")
print("시간축: t, live, armed, s8 last_command, fwd, lat, contact, (나이>0.1 s 인 모터 번호)")
for k in range(len(rows)):
    if -1.6 <= T[k] <= 6.0 and (k % 2 == 0 or abs(T[k]) < 1.0):
        r = rows[k]; over = [i + 1 for i, j in enumerate(ORD) if ((r.get("joints") or {}).get(j + "_joint", {}).get("feedback_age_sec") or 0) > 0.1]
        print(f"  {T[k]:+6.2f} live={r.get('live')} armed={r.get('armed')} s8={r.get('stage8_last_command')} fwd={r.get('fwd')} lat={r.get('lat')} c={r.get('sensor_contact')} over={over}")
