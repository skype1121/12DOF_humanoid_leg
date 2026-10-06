"""W3 걷기→정지(stopdec 기록, W3A3 제자리·W3S1b 앞 0.06) 이상 운동학 상한: 참 발 위치−골반 위치 유한차분(지지 = 접촉 힘>5 N 두 틱) → 주기 평균. 엔코더·IMU 오차 없음."""
import numpy as np, sys
S='/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/stopdec/'
W=45
def tm(x):
    cs=np.cumsum(np.concatenate([np.zeros((1,)+x.shape[1:]),x],0),0); o=np.full(x.shape,np.nan); o[W-1:]=(cs[W:]-cs[:-W])/W; return o
for f in ('a3_stop','s1b_f06_stop','s1b_f06_nostop'):
    d=np.load(S+f+'.npz'); t=d['t']; root=d['root']; feet=d['feet'][...,:2]; cc=d['cc']; alive=d['alive']; comv=d['comv']
    rel=feet-root[:,:,None,:]
    ok=cc[1:]&cc[:-1]
    v=-(rel[1:]-rel[:-1])/0.02
    w=ok.sum(-1); vi=np.where(w[...,None]>0,(v*ok[...,None]).sum(2)/np.maximum(w,1)[...,None],np.nan)
    # hold last for invalid
    for k in range(1,vi.shape[0]):
        bad=np.isnan(vi[k,:,0]); vi[k,bad]=vi[k-1,bad]
    vi=np.nan_to_num(vi)
    vb=tm(vi); cb=tm(comv[1:])
    sb=np.linalg.norm(vb,axis=-1); sc=np.linalg.norm(cb,axis=-1)
    ts=d['t_stop']; tt=t[1:]
    rel_t=tt[:,None]-ts[None,:]
    m=alive[1:]&np.isfinite(sc)
    e=np.linalg.norm(vb-cb,axis=-1)
    print(f, 'err rms all', np.sqrt(np.nanmean(e[m]**2)).round(3))
    for lo,hi in ((-4,0),(0,1),(1,2),(2,4),(4,8)):
        mk=m&(rel_t>=lo)&(rel_t<hi)
        if mk.sum()==0: continue
        print(f'  t-stop [{lo},{hi}): true|vbar| p50/p90 {np.percentile(sc[mk],50):.3f}/{np.percentile(sc[mk],90):.3f} est p50/p90 {np.percentile(sb[mk],50):.3f}/{np.percentile(sb[mk],90):.3f} err rms {np.sqrt((e[mk]**2).mean()):.3f}  est<0.07 {100*(sb[mk]<0.07).mean():.0f}%')
