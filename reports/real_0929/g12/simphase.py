import json,sys,numpy as np
dt=0.02
def load(f):
    r=[json.loads(l) for l in open(f)]; r=[a for a in r if a['phase']=='policy']
    s=np.array([a['t_rel'] for a in r]); th=np.array([a['pitch'] for a in r])
    c=np.array([a['cmd_deg'] for a in r]); q=np.array([a['q_deg'] for a in r]); ct=np.array([a['contact'] for a in r])
    return s,th,c,q,ct
def xlag(a,b,maxl=40):
    a=a-a.mean(); b=b-b.mean(); best=(0,0)
    for L in range(0,maxl+1):
        if len(a)-L<20: break
        cc=np.corrcoef(a[:len(a)-L],b[L:])[0,1]
        if abs(cc)>abs(best[1]): best=(L,cc)
    return best
def ph(a,b,f0):
    n=len(a); w=np.hanning(n); A=np.fft.rfft((a-a.mean())*w); B=np.fft.rfft((b-b.mean())*w); fr=np.fft.rfftfreq(n,dt); k=np.argmin(abs(fr-f0))
    return np.degrees(np.angle(B[k]/A[k])), abs(B[k])/abs(A[k])
for spec in sys.argv[1:]:
    f,a,b=spec.split(':'); a=float(a); b=float(b)
    s,th,c,q,ct=load(f); m=(s>=a)&(s<b)
    if m.sum()<40: print(f, '짧음', m.sum()); continue
    x=th[m]; F=np.abs(np.fft.rfft((x-x.mean())*np.hanning(len(x)))); fr=np.fft.rfftfreq(len(x),dt); f0=fr[np.argmax(F[1:])+1]
    ankc=(c[:,8]-c[:,9])/2; ankq=(q[:,8]-q[:,9])/2; hipc=(c[:,0]-c[:,1])/2; hipq=(q[:,0]-q[:,1])/2
    La,ra=xlag(ankc[m],ankq[m]); Lh,rh=xlag(hipc[m],hipq[m])
    pa,ga=ph(th[m],ankc[m],f0); pa2,ga2=ph(ankc[m],ankq[m],f0)
    print(f"{f.replace('recbb_g12e_',''):34s} {a:4.1f}-{b:4.1f}s 주파수 {f0:4.2f}Hz 피치p2p {x.max()-x.min():5.1f}° | 발목명령 vs 피치 {pa:+6.0f}° | 발목 명령→실제 지연 {La*dt:.2f}s 추종 {ga2:.2f} | 힙 지연 {Lh*dt:.2f}s | 양접 {(ct[m].min(1)>0).mean():.0%}")
