import re,sys,numpy as np,datetime as dt
rx=re.compile(r'\((\S+ \S+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[8\]\s+(.*)')
def load(f):
    cmd={}; rep={}; last={}
    for l in open(f):
        m=rx.match(l.strip())
        if not m: continue
        ts=dt.datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S.%f').timestamp()
        mid=int(m.group(2),16); b=bytes(int(x,16) for x in m.group(3).split())
        if b[:7]==b'\xff'*7: continue
        is_rep = b[0]==mid and b[7]<0x10 and mid in last and ts-last[mid]<0.005
        if is_rep:
            rep.setdefault(mid,[]).append((ts,((b[1]<<8)|b[2]), ((b[3]<<4)|(b[4]>>4)), (((b[4]&0xF)<<8)|b[5])))
        else:
            cmd.setdefault(mid,[]).append((ts,((b[0]<<8)|b[1]), ((b[3]&0xF)<<8)|b[4]))
            last[mid]=ts
    return {k:np.array(v,float) for k,v in cmd.items()},{k:np.array(v,float) for k,v in rep.items()}
def xlag(a,b,maxl=40):
    a=a-a.mean(); b=b-b.mean(); best=(0,-2)
    for L in range(0,maxl):
        c=np.corrcoef(a[:len(a)-L],b[L:])[0,1]
        if c>best[1]: best=(L,c)
    return best
if __name__=='__main__':
    f=sys.argv[1]; t_a=float(sys.argv[2]); t_b=float(sys.argv[3])
    c,r=load(f); t0=min(v[0,0] for v in c.values())
    rad=25.0/65535  # 16bit → ±12.5 rad
    print(f"기록 시작 {dt.datetime.fromtimestamp(t0)}  창 {t_a}-{t_b}s")
    for mid in sorted(c):
        C=c[mid]; R=r.get(mid)
        if R is None or len(R)<100: print(mid,'응답 부족'); continue
        g=np.arange(t0+t_a,t0+t_b,0.02)
        cp=np.interp(g,C[:,0],C[:,1])*rad; rp=np.interp(g,R[:,0],R[:,1])*rad
        L,cc=xlag(cp,rp); amp_c=np.ptp(cp); amp_r=np.ptp(rp)
        dts=R[:,0]-np.interp(R[:,0],C[:,0],C[:,0])
        print(f"모터{mid:2d} 명령→응답위치 지연 {L*20:4d} ms (r={cc:.2f}) | 명령 p2p {np.degrees(amp_c):5.1f}° 응답 p2p {np.degrees(amp_r):5.1f}° | 응답수 {len(R)} 명령수 {len(C)}")
