"""signchk 덤프 비교: zero 대비 p10/m10/hipa 의 발 링크 회전(월드) 축·각 → 발목롤 +10° 가 물리적으로 어느 쪽 가장자리를 내리는지.
월드 프레임: x 앞, +y 왼, z 위(현 USD). 앞축(x) 둘레 +회전 = 왼(+y) 가장자리 위·오른 가장자리 아래."""
import json, numpy as np, sys
def q2R(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def last_quats(tag):
    R=[json.loads(l) for l in open(f"log_picture/simbb_{tag}_env0.jsonl") if l.strip()]; R=[r for r in R if r.get("feet_quat_wxyz")]
    r=R[-1]; return [q2R(q) for q in r["feet_quat_wxyz"]], np.degrees(r["qpos_rad"]), r["timestamp"]-R[0]["timestamp"], len(R)
R0,q0,t0,n0=last_quats("signchk3_zero")
for tag in ["signchk3_p10","signchk3_m10","signchk3_hipa"]:
    try: R1,q1,t1,n1=last_quats(tag)
    except Exception as e: print(tag,"missing",e); continue
    print(f"== {tag}: records {n1} t={t1:.1f}s | ankle_r q L/R {q1[10]:+.1f}/{q1[11]:+.1f} (zero {q0[10]:+.1f}/{q0[11]:+.1f}) | hip_a q L/R {q1[2]:+.1f}/{q1[3]:+.1f}")
    for name,Ra,Rb in [("L foot",R0[0],R1[0]),("R foot",R0[1],R1[1])]:
        D=Rb@Ra.T; ang=np.degrees(np.arccos(np.clip((np.trace(D)-1)/2,-1,1)))
        ax=np.array([D[2,1]-D[1,2],D[0,2]-D[2,0],D[1,0]-D[0,1]]); ax=ax/(np.linalg.norm(ax)+1e-12)
        rx=ax[0]*ang   # 앞축(x) 둘레 회전 성분(°): + = 왼 가장자리 위 = 오른 가장자리 아래
        side="left" if name.startswith("L") else "right"
        if abs(rx)>0.5:
            inner_down = (rx>0) if side=="left" else (rx<0)   # 왼발: 오른(안쪽) 가장자리 아래 = rx>0 ; 오른발: 안쪽=왼 가장자리 아래 = rx<0
            print(f"   {name}: 회전 {ang:.1f}° 축 {np.round(ax,2).tolist()} → 앞축 성분 {rx:+.1f}° → {'안쪽 가장자리 아래(외번)' if inner_down else '바깥 가장자리 아래(내번)'}")
        else:
            print(f"   {name}: 회전 {ang:.1f}° 축 {np.round(ax,2).tolist()} (앞축 성분 {rx:+.1f}°)")
