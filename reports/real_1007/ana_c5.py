"""실물 C5/KE 브리지 블랙박스 한 파일 요약 — 회복 활성화별 지속·끝·복귀 자세, 정지 사건, (선택) 젯슨 세션 타임라인에서 정지 전후 관절 피드백 나이.
사용: python3 reports/real_1007/ana_c5.py <rl_bridge_*.jsonl> [--timeline]   (--timeline 은 젯슨에 읽기 전용 ssh — 최신 세션 폴더에서 정지 시각 ±3 s 만 조회)
주의(CLAUDE.md): 패드는 접촉 0/1 로만 본다. 밀기·보조 표식이 없으면 성공 집계에 쓰지 않는다. rp_deg 는 장착 편향 보정 전 값이라 기준(ref_deg)과 직접 빼지 않고 '직전 1 s 대비'로 본다."""
import json, subprocess, sys
import numpy as np

NAMES = ["LhF", "RhF", "LhA", "RhA", "LhR", "RhR", "Lkn", "Rkn", "LaF", "RaF", "LaR", "RaR"]


def vec(x, n):
    if isinstance(x, dict):
        x = list(x.values())
    if x is None:
        return [np.nan] * n
    x = list(x)[:n]
    return [float(v) if isinstance(v, (int, float)) else np.nan for v in x] + [np.nan] * (n - len(x))


def main():
    p = sys.argv[1]; T = []; M = []; RP = []; Q = []; C = []; G = []; NE = []; NON = []; CUE = []; ev = []; ck = set(); gm = set()
    for line in open(p, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("c5") is None:
            if "event" in d:
                ev.append(d)
            if d.get("checkpoint"):
                ck.add(d.get("checkpoint"))
            continue
        ck.add(d.get("checkpoint")); gm.add(d.get("guard_mode")); c5 = d["c5"]
        T.append(d["timestamp"]); M.append(int(c5["mode"])); NE.append(c5.get("n_early", 0)); NON.append(c5.get("n_on", 0)); CUE.append(c5.get("cue", 0))
        RP.append(vec(d.get("rp_deg"), 2)); C.append(vec(d.get("contact"), 2)); G.append(vec(d.get("gyro_rad_s"), 3)); Q.append(vec(d.get("qpos_rad"), 12))
    if not T:
        print(f"{p}: C5 틱 없음(정책 {ck}) · 사건 {[(e.get('event'), e.get('reason')) for e in ev][-4:]}"); return
    T0 = T[0]; T = np.array(T) - T0; M = np.array(M); RP = np.array(RP, float); Q = np.array(Q, float); C = np.array(C, float); G = np.array(G, float)
    w = np.hypot(G[:, 0], G[:, 1]); both = (C > 0.5).all(1)
    print(f"== {p}\n정책 {ck} · guard_mode {gm} · {T[-1]:.1f} s · 회복 모드 {100*M.mean():.1f} % · 켜짐 {int(NON[-1])} 회 · 조기 복귀 {int(NE[-1])} 회 · cue 틱 {int(np.sum(CUE))} · 양발 접촉 비율 {np.nanmean(both):.2f}")
    q = (T > 3) & (M == 0) & (w < 0.1)
    if q.sum() > 50:
        print(f"조용한 서기: 롤 {np.nanmedian(RP[q,0]):+.1f}° 피치 {np.nanmedian(RP[q,1]):+.1f}° · |ω| 중앙 {np.nanmedian(w[q]):.3f}")
    for e in ev:
        print(f"  사건 t {e.get('timestamp', T0) - T0:7.2f}s: " + str({k: v for k, v in e.items() if k != 'timestamp' and not isinstance(v, (list, dict))})[:200])
    on = np.nonzero(np.diff(M) == 1)[0] + 1; off = np.nonzero(np.diff(M) == -1)[0] + 1
    for a in on:
        has_off = (off > a).any(); b = off[off > a][0] if has_off else len(M) - 1
        pre = slice(max(0, a - 50), a); rp_pre = np.nanmean(RP[pre], 0); seg = slice(a, b + 1); dev = np.linalg.norm(RP[seg] - rp_pre, axis=1)
        best = cur = 0
        for qv in ((w[seg] < 0.3) & both[seg]):
            cur = cur + 1 if qv else 0; best = max(best, cur)
        lifts = int((np.diff(both[seg].astype(int)) == -1).sum())
        line = (f"  회복 t {T[a]:6.2f}s · 지속 {T[b]-T[a]:5.2f}s · {'서기 복귀' if has_off else '안 꺼짐(끝까지)'} · 직전 대비 최대 편차 {np.nanmax(dev):.1f}° · |ω| 최대 {np.nanmax(w[seg]):.2f}"
                f" · 한발 구간 {lifts} 회 · 조용한 양발 최장 {best*0.02:.2f}s")
        if has_off:
            post = slice(min(len(M) - 1, b + 25), min(len(M), b + 125))
            if post.stop - post.start > 10:
                dq = np.degrees(np.nanmean(Q[post], 0) - np.nanmean(Q[pre], 0)); rpp = np.nanmean(RP[post], 0)
                line += (f" · 복귀 뒤 0.5–2.5 s: 자세 Δ(롤,피치) [{rpp[0]-rp_pre[0]:+.1f}, {rpp[1]-rp_pre[1]:+.1f}]° · 관절 평균 |Δ| {np.nanmean(np.abs(dq)):.1f}° 최대 {NAMES[int(np.nanargmax(np.abs(dq)))]} {dq[int(np.nanargmax(np.abs(dq)))]:+.1f}°"
                         f" · |ω| {np.nanmean(w[post]):.2f} · 그 구간 회복 재켜짐 {int(M[post].any())}")
        print(line)
    stops = [e for e in ev if e.get("event") == "stop_all"]
    if stops and "--timeline" in sys.argv:
        ts = stops[-1]["timestamp"]
        code = r'''
import json,glob,os
d0=sorted(glob.glob('/home/mama/logs/sessions/session_*'), key=os.path.getmtime)[-1]
short={'left_hip_f_joint':'LhF','right_hip_f_joint':'RhF','left_hip_a_joint':'LhA','right_hip_a_joint':'RhA','left_hip_r_joint':'LhR','right_hip_r_joint':'RhR','left_knee_joint':'Lkn','right_knee_joint':'Rkn','left_ankle_f_joint':'LaF','right_ankle_f_joint':'RaF','left_ankle_r_joint':'LaR','right_ankle_r_joint':'RaR'}
TS=%r; worst={}; first={}; tl=[]
for line in open(d0+'/timeline.jsonl', errors='ignore'):
    if '"joints"' not in line: continue
    try: d=json.loads(line)
    except Exception: continue
    t=d.get('ts')
    if not isinstance(t,(int,float)) or not (-3.0 < t-TS < 0.6): continue
    tl.append((round(t-TS,2), d.get('tilt'), d.get('foot_pads_on')))
    for k,v in d['joints'].items():
        a=v.get('feedback_age_sec') if isinstance(v,dict) else None
        if isinstance(a,(int,float)):
            s=short.get(k,k); worst[s]=max(worst.get(s,0),a)
            if a>0.05 and s not in first: first[s]=round(t-TS,2)
print(os.path.basename(d0), '| 정지 전후 관절 피드백 나이 최대[s]:', {k:round(v,3) for k,v in sorted(worst.items(), key=lambda x:-x[1])})
print('  나이 0.05 s 를 처음 넘은 시각(정지 기준):', first, '| dead mask', d.get('foot_dead_mask'))
print('  기울기(정지 기준 s, °):', [(a,b) for a,b,_ in tl[::6]][-12:])
''' % ts
        r = subprocess.run(["sshpass", "-p", "***", "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8", "mama@172.16.35.89", "python3 -"], input=code, capture_output=True, text=True, timeout=60)
        print("  타임라인:", (r.stdout or r.stderr).strip()[:900])


if __name__ == "__main__":
    main()
