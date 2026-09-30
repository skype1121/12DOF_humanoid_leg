"""candump(-ta 절대시각) → 모터별 명령/응답 시계열. AK MIT: 명령 id=모터, pos16|vel12|kp12|kd12|t12, 응답 id=0 data[0]=모터."""
import re,sys,numpy as np,datetime as dt
P_MAX,V_MAX,KP_MAX,KD_MAX={}, {}, 500.0, 5.0
# AK70-10: P±12.5 rad, V±50, T±25 / AK45-36(발목롤 5·11): P±12.5, V±8, T±...(응답 해석은 위치만 정밀 비교)
def u2f(x,lo,hi,bits): return x*(hi-lo)/((1<<bits)-1)+lo
def load(f):
    cmd={}; rep={}
    rx=re.compile(r'\((\S+ \S+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[\d\]\s+(.*)')
    for l in open(f):
        m=rx.match(l.strip())
        if not m: continue
        ts=dt.datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S.%f').timestamp()
        cid=int(m.group(2),16); b=bytes(int(x,16) for x in m.group(3).split())
        if len(b)!=8 and not (cid==0 and len(b)>=6): continue
        if cid==0:
            mid=b[0]; p=(b[1]<<8)|b[2]; v=(b[3]<<4)|(b[4]>>4); i=((b[4]&0xF)<<8)|b[5]
            rep.setdefault(mid,[]).append((ts,u2f(p,-12.5,12.5,16),u2f(v,-50,50,12),u2f(i,-25,25,12)))
        elif 1<=cid<=12 and len(b)==8:
            if b in (bytes.fromhex('FFFFFFFFFFFFFFFC'),bytes.fromhex('FFFFFFFFFFFFFFFD'),bytes.fromhex('FFFFFFFFFFFFFFFE')): continue
            p=(b[0]<<8)|b[1]; kp=((b[3]&0xF)<<8)|b[4]; kd=(b[5]<<4)|(b[6]>>4)
            cmd.setdefault(cid,[]).append((ts,u2f(p,-12.5,12.5,16),u2f(kp,0,KP_MAX,12),u2f(kd,0,KD_MAX,12)))
    return {k:np.array(v) for k,v in cmd.items()},{k:np.array(v) for k,v in rep.items()}
if __name__=='__main__':
    c,r=load(sys.argv[1])
    for mid in sorted(c):
        a=c[mid]; b=r.get(mid,np.zeros((0,4)))
        print(f"모터{mid:2d} 명령 {len(a):6d} ({1/np.median(np.diff(a[:,0])):.0f}Hz) kp~{np.median(a[:,2]):.0f} kd~{np.median(a[:,3]):.2f} | 응답 {len(b):6d}"+(f" ({1/np.median(np.diff(b[:,0])):.0f}Hz)" if len(b)>2 else ""))
