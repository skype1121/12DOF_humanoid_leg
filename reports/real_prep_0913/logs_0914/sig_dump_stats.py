"""시뮬 덤프(simbb_<tag>_env*.jsonl) 의 낙상 서명 통계: 방출 완료 → 왼/오른발 접촉 0 순서·시각, 그때 롤, 롤 속도(자이로 x 최대·중앙), 관절 스윙, 정책 목표 스윙.
사용: python3 sig_dump_stats.py <tag> [--release 18.98]"""
import json, glob, sys, statistics as st, numpy as np
tag=sys.argv[1]; rel=float(sys.argv[sys.argv.index("--release")+1]) if "--release" in sys.argv else None
files=sorted(glob.glob(f"log_picture/simbb_{tag}_env*.jsonl"))
J=["left_hip_f_joint","right_hip_f_joint","left_hip_a_joint","right_hip_a_joint","left_hip_r_joint","right_hip_r_joint","left_knee_joint","right_knee_joint","left_ankle_f_joint","right_ankle_f_joint","left_ankle_r_joint","right_ankle_r_joint"]
rows=[]
for f in files:
    R=[json.loads(l) for l in open(f) if l.strip()]
    ev=[r for r in R if "event" in r]; R=[r for r in R if isinstance(r.get("rp_deg"),list) and isinstance(r.get("contact"),list)]
    t0=R[0]["timestamp"]
    tg=next((r["timestamp"]-t0 for r in R if r.get("support_gone")),None)
    fell=bool(ev)
    tend=R[-1]["timestamp"]-t0
    after=[r for r in R if tg is not None and r["timestamp"]-t0>=tg]
    def first0(k): return next((r["timestamp"]-t0 for r in after if r["contact"][k]==0),None)
    tL,tR=first0(0),first0(1)
    tfirst=min([x for x in (tL,tR) if x is not None],default=None)
    roll_at=None; gx_pre=None; slope=None
    if tfirst is not None:
        rr=next(r for r in after if r["timestamp"]-t0>=tfirst); roll_at=rr["rp_deg"][0]
        pre=[r for r in after if r["timestamp"]-t0<=tfirst]
        gx_pre=max(abs(r["gyro_rad_s"][0])*57.3 for r in pre) if pre else None
        win=[r for r in pre if r["timestamp"]-t0>=tfirst-2.0]
        if len(win)>2: slope=(win[-1]["rp_deg"][0]-win[0]["rp_deg"][0])/max(1e-6,win[-1]["timestamp"]-win[0]["timestamp"])
    # joint & target swing in [release, min(end, release+6)]
    win2=[r for r in after if r["timestamp"]-t0<=tg+6.0] if tg is not None else []
    def pp(vals): return (max(vals)-min(vals)) if vals else None
    sw_ar=pp([r["qpos_rad"][10]*57.3 for r in win2]); sw_ha=pp([r["qpos_rad"][2]*57.3 for r in win2])
    sw_tar=pp([r["targets_deg_joint"]["left_ankle_r_joint"] for r in win2]); sw_tha=pp([r["targets_deg_joint"]["left_hip_a_joint"] for r in win2])
    maxroll=max(abs(r["rp_deg"][0]) for r in R)
    rows.append(dict(env=f[-7:-6],fell=fell,tend=tend,tg=tg,tL=tL,tR=tR,first=("L" if tL is not None and (tR is None or tL<=tR) else ("R" if tR is not None else "-")),roll_at=roll_at,gx=gx_pre,slope=slope,sw_ar=sw_ar,sw_ha=sw_ha,sw_tar=sw_tar,sw_tha=sw_tha,maxroll=maxroll,endroll=R[-1]["rp_deg"][0]))
fmt=lambda v,f="%.2f": "—" if v is None else (f%v)
print(f"tag {tag}: envs {len(rows)}, fell {sum(r['fell'] for r in rows)}")
print("| env | 낙상 | 종료 s | 방출완료 s | 첫 이탈 발 | L0 s | R0 s | 이탈 시 롤 ° | 이탈 전 |gx| max °/s | 이탈 전 2 s 롤 기울기 °/s | 관절 스윙 ankle_r L/hip_a L ° (방출~+6 s) | 목표 스윙 ankle_r L/hip_a L ° | 최대|롤| ° | 종료 롤 ° |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['env']} | {'Y' if r['fell'] else 'n'} | {r['tend']:.1f} | {fmt(r['tg'])} | {r['first']} | {fmt(r['tL'])} | {fmt(r['tR'])} | {fmt(r['roll_at'])} | {fmt(r['gx'],'%.1f')} | {fmt(r['slope'])} | {fmt(r['sw_ar'],'%.1f')}/{fmt(r['sw_ha'],'%.1f')} | {fmt(r['sw_tar'],'%.1f')}/{fmt(r['sw_tha'],'%.1f')} | {r['maxroll']:.1f} | {r['endroll']:+.1f} |")
def med(k): v=[r[k] for r in rows if r[k] is not None]; return st.median(v) if v else None
print("중앙값: 이탈 전 |gx| max %s °/s · 롤 기울기 %s °/s · 관절 스윙 ankle_r %s° hip_a %s° · 목표 스윙 %s/%s° · 첫 이탈 L %d / R %d"%(fmt(med('gx'),'%.1f'),fmt(med('slope')),fmt(med('sw_ar'),'%.1f'),fmt(med('sw_ha'),'%.1f'),fmt(med('sw_tar'),'%.1f'),fmt(med('sw_tha'),'%.1f'),sum(r['first']=='L' for r in rows),sum(r['first']=='R' for r in rows)))
