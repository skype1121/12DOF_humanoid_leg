import json,sys,numpy as np
f=sys.argv[1]; wins=[tuple(map(float,w.split('-'))) for w in sys.argv[2:]]
rows=[json.loads(l) for l in open(f)]; t=[r for r in rows if 'event' not in r]; t0=t[0]['timestamp']
s=np.array([r['timestamp']-t0 for r in t])
th=np.array([r['rp_deg'][1] for r in t]); w=np.degrees(np.array([r['gyro_rad_s'][1] for r in t]))
q=np.degrees(np.array([r['qpos_rad'] for r in t]))
tg=lambda j: np.array([r['targets_deg_joint'][j] for r in t])
# 앞뒤 성분 (좌 +, 우 − 규약) : 발목 = 배굴(+), 힙 = 굴곡(+)
ank_c=(tg('left_ankle_f_joint')-tg('right_ankle_f_joint'))/2; ank_q=(q[:,8]-q[:,9])/2
hip_c=(tg('left_hip_f_joint')-tg('right_hip_f_joint'))/2;     hip_q=(q[:,0]-q[:,1])/2
kn_c=(tg('left_knee_joint')-tg('right_knee_joint'))/2;        kn_q=(q[:,6]-q[:,7])/2
dt=0.02
def xlag(a,b,maxl=60):
    """b 가 a 보다 몇 틱 늦나 (상관 최대, 부호 포함 상관값)"""
    a=a-a.mean(); b=b-b.mean(); best=None
    for L in range(-maxl,maxl+1):
        if L>=0: c=np.corrcoef(a[:len(a)-L],b[L:])[0,1]
        else: c=np.corrcoef(a[-L:],b[:len(b)+L])[0,1]
        if best is None or abs(c)>abs(best[1]): best=(L,c)
    return best
def ph(a,b,fr0):
    """주파수 fr0 에서 b 의 위상 − a 의 위상 (deg), b 진폭/a 진폭"""
    n=len(a); win=np.hanning(n); A=np.fft.rfft((a-a.mean())*win); B=np.fft.rfft((b-b.mean())*win)
    fr=np.fft.rfftfreq(n,dt); k=np.argmin(abs(fr-fr0))
    return np.degrees(np.angle(B[k]/A[k])), abs(B[k])/abs(A[k])
for a,b in wins:
    m=(s>=a)&(s<b); x=th[m]
    F=np.abs(np.fft.rfft((x-x.mean())*np.hanning(len(x)))); fr=np.fft.rfftfreq(len(x),dt); k=np.argmax(F[1:])+1; f0=fr[k]
    print(f"\n== {a:.0f}-{b:.0f}s  피치 주파수 {f0:.2f} Hz, 피치 p2p {x.max()-x.min():.1f}°")
    for name,sig in [('발목 명령',ank_c),('발목 실제',ank_q),('힙 명령',hip_c),('힙 실제',hip_q),('무릎 명령',kn_c)]:
        p,g=ph(th[m],sig[m],f0); L,c=xlag(th[m],sig[m])
        print(f"  {name:6s}: 피치 대비 위상 {p:+7.1f}°  이득 {g:5.2f}  | 상관 최대 지연 {L*dt:+.2f}s (r={c:+.2f})")
    for name,cc,qq in [('발목',ank_c,ank_q),('힙',hip_c,hip_q)]:
        L,c=xlag(cc[m],qq[m]); p,g=ph(cc[m],qq[m],f0)
        print(f"  {name} 명령→실제: 위상 {p:+6.1f}°, 추종비 {g:.2f}, 지연 {L*dt:+.2f}s (r={c:+.2f})")
