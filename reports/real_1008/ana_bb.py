#!/usr/bin/env python3
"""브리지 블랙박스 한 개를 읽어 공중 → 착지 → 인계 구간과 정지 원인을 요약한다(로컬 파일만).
사용: ana_bb.py <rl_bridge_*.jsonl> [끝에서 몇 틱 자세히=70]"""
import json, sys, numpy as np
B = sys.argv[1]; NT = int(sys.argv[2]) if len(sys.argv) > 2 else 70
rows = []
for ln in open(B, "rb"):
    try: rows.append(json.loads(ln.decode("utf-8", "replace")))
    except Exception: pass
ev = [r for r in rows if "event" in r and "live" not in r]
tk = [r for r in rows if "qpos_rad" in r]
t0 = tk[0]["timestamp"]
print(f"틱 {len(tk)}  길이 {tk[-1]['timestamp']-t0:.2f} s  시작 {t0:.3f}")
for r in ev:
    print(f"  이벤트 t={r['timestamp']-t0:+.3f}", {k: v for k, v in r.items() if k not in ('timestamp', 'w3', 'sup', 'kew3', 'guard_mode')}, "| kew3", (r.get('kew3') or {}).get('event'))
JN = list(tk[0]["targets_deg_joint"].keys())
print("관절 순서(블랙박스):", [j.replace('_joint', '') for j in JN])
T = np.array([r["timestamp"] - t0 for r in tk])
Q = np.degrees(np.array([r["qpos_rad"] for r in tk])); V = np.array([r["qvel_rad_s"] for r in tk])
TG = np.array([[r["targets_deg_joint"][j] for j in JN] for r in tk])
A = np.array([r["raw_action"] for r in tk]) if tk[0].get("raw_action") is not None else None
C = np.array([r["contact"] for r in tk]); RP = np.array([r["rp_deg"] for r in tk]); G = np.array([r["gyro_rad_s"] for r in tk])
mode = [(r.get("kew3") or {}).get("mode") for r in tk]; dev = np.array([(r.get("kew3") or {}).get("air_dev") or 0 for r in tk])
fam = np.array([r.get("fb_age_max_s") or 0 for r in tk]); faj = [r.get("fb_age_max_joint") for r in tk]; sta = np.array([r.get("status_age_s") or 0 for r in tk])
dt = np.diff(T); print(f"틱 간격 중앙 {np.median(dt)*1e3:.1f} ms  p99 {np.percentile(dt,99)*1e3:.1f}  최대 {dt.max()*1e3:.1f} ms (t={T[1:][dt.argmax()]:.2f})")
print(f"rp_deg 의미 확인용 첫 틱 {RP[0]}  fb_age_max 중앙 {np.median(fam)*1e3:.0f} ms  최대 {fam.max()*1e3:.0f} ms  status_age 최대 {sta.max()*1e3:.0f} ms")
# 구간
def first(mask):
    i = np.flatnonzero(mask); return int(i[0]) if len(i) else None
i_c1 = first(C.max(1) > 0.5); i_c2 = first(C.min(1) > 0.5); i_ke = first(np.array([m != 'air' for m in mode]))
print(f"첫 접촉 t={T[i_c1] if i_c1 is not None else None}  양발 접촉 t={T[i_c2] if i_c2 is not None else None}  KE 인계 t={T[i_ke] if i_ke is not None else None}  끝 t={T[-1]:.2f}")
air = slice(0, i_c1 if i_c1 is not None else len(tk))
q0 = TG[0]
print("공중 구간: 목표−시작목표 최대 |°|", np.round(np.abs(TG[air] - TG[0]).max(0), 1).tolist())
print("공중 구간: 측정 관절 이동폭(°)   ", np.round(Q[air].max(0) - Q[air].min(0), 1).tolist())
print(f"공중 구간: air_dev 최대 {dev[air].max():.1f}°  몸통 rp 범위 r[{RP[air,0].min():.1f},{RP[air,0].max():.1f}] p[{RP[air,1].min():.1f},{RP[air,1].max():.1f}]  |gyro| 최대 {np.abs(G[air]).max(0).round(2).tolist()}")
if A is not None: print(f"공중 구간: |raw_action| 최대 {np.abs(A[air]).max():.2f}  전체 최대 {np.abs(A).max():.2f} (t={T[np.abs(A).max(1).argmax()]:.2f})")
# 피드백 동결: 관절별 마지막으로 값이 바뀐 시각
print("관절별 마지막 qpos 변화 시각(끝 기준 −s) / 동결 길이:")
for j, n in enumerate(JN):
    ch = np.flatnonzero(np.abs(np.diff(Q[:, j])) > 1e-9)
    last = T[ch[-1] + 1] if len(ch) else None
    # 가장 긴 무변화 구간
    idx = np.concatenate([[0], ch + 1, [len(T) - 1]]); gaps = np.diff(T[idx]); g = gaps.argmax()
    print(f"  {n.replace('_joint',''):14s} 마지막 변화 t={last:.3f} (끝−{T[-1]-last:.3f}s)  최장 무변화 {gaps.max()*1e3:.0f} ms @t={T[idx[g]]:.2f}")
print(f"--- 끝 {NT} 틱")
print(" t     mode c  fbmax(ms) joint          status(ms) | pitch  roll | gyro xyz           | 바뀐 관절 수 | max|목표변화|/틱° | max|a|")
for i in range(max(1, len(tk) - NT), len(tk)):
    nchg = int((np.abs(Q[i] - Q[i - 1]) > 1e-9).sum()); dtg = np.abs(TG[i] - TG[i - 1]).max()
    am = np.abs(A[i]).max() if A is not None else float('nan')
    ov = tk[i].get("fb_age_over")
    print(f"{T[i]:6.2f} {str(mode[i]):4s} {int(C[i,0])}{int(C[i,1])} {fam[i]*1e3:7.0f} {str(faj[i]).replace('_joint',''):14s} {sta[i]*1e3:7.0f} | {RP[i,1]:+5.1f} {RP[i,0]:+5.1f} | {G[i,0]:+.2f} {G[i,1]:+.2f} {G[i,2]:+.2f} | {nchg:2d} | {dtg:5.2f} | {am:5.2f}" + (f" | over={ov}" if ov else ""))
