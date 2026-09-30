import json,sys,numpy as np
f=sys.argv[1]
rows=[json.loads(l) for l in open(f)]
t=[r for r in rows if 'event' not in r]
ts=np.array([r['timestamp'] for r in t]); ts-=ts[0]
rp=np.array([r['rp_deg'] if r.get('rp_deg') else [np.nan,np.nan] for r in t],float)
gy=np.array([r['gyro_rad_s'] for r in t],float)
ct=np.array([r['contact'] for r in t],float)
al=np.array([r['blend_alpha'] or 0 for r in t],float)
q=np.degrees(np.array([r['qpos_rad'] for r in t],float))
ra=np.array([r['raw_action'] if r.get('raw_action') else [np.nan]*12 for r in t],float)
hs=[r['hang_state'] for r in t]
W=float(sys.argv[2]) if len(sys.argv)>2 else 5.0
T1=float(sys.argv[3]) if len(sys.argv)>3 else 1e9
print(f"{'구간':>9} {'α':>4} {'롤':>6} {'피치':>6} {'|ω|rms':>6} {'양접':>4} {'무접':>4} {'반전':>5} {'주파':>5} {'무릎L':>6} {'무릎R':>6}")
for a in np.arange(0,min(ts[-1],T1),W):
    m=(ts>=a)&(ts<a+W)
    if m.sum()<20: continue
    g=gy[m]; w=np.sqrt((g**2).sum(1).mean())
    both=(ct[m].min(1)>0.5).mean(); none=(ct[m].max(1)<0.5).mean()
    d=np.diff(ra[m],axis=0); rev=np.nanmean((np.sign(d[1:])*np.sign(d[:-1])<0))
    x=g[:,1]-g[:,1].mean(); F=np.abs(np.fft.rfft(x)); fr=np.fft.rfftfreq(len(x),0.02); k=np.argmax(F[1:])+1
    print(f"{a:4.0f}-{a+W:3.0f}s {al[m].mean():4.2f} {np.nanmean(rp[m,0]):+6.1f} {np.nanmean(rp[m,1]):+6.1f} {w:6.3f} {both:4.0%} {none:4.0%} {rev:5.2f} {fr[k]:5.2f} {q[m,6].mean():+6.1f} {q[m,7].mean():+6.1f}")
from collections import Counter; print(Counter(hs))
