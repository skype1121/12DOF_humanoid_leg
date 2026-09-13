"""공중 손굴리기 테스트: 롤을 기울여 '유지'한 구간마다 정책 목표 변화(Δ)와 이득(°/°), 지연을 뽑는다.
사용: python3 air_hold_gain.py <rl_bridge_*.jsonl> [--thr 3] [--hold 1.5]
- 기준(pre) = live 시작 후 처음 5 s 의 평균(정지 매달림).
- 구간 = |roll−pre_roll| ≥ thr 가 hold s 이상 지속. 구간 값 = 마지막 60 % 평균(정착 부분).
- 지연 = 롤이 thr/2 를 넘은 시각 → 발목 비대칭 모드 또는 힙 대칭 모드 변화가 최종 Δ의 50 % 에 도달한 시각.
"""
import json, sys, statistics as st
p=sys.argv[1]; thr=3.0; hold=1.5
if "--thr" in sys.argv: thr=float(sys.argv[sys.argv.index("--thr")+1])
if "--hold" in sys.argv: hold=float(sys.argv[sys.argv.index("--hold")+1])
R=[json.loads(l) for l in open(p) if l.strip()]
L=[r for r in R if r.get("live") and isinstance(r.get("rp_deg"),list) and r.get("targets_deg_joint")]
t0=L[0]["timestamp"]
def tg(r,n): return r["targets_deg_joint"][n]
J=["left_ankle_r_joint","right_ankle_r_joint","left_hip_a_joint","right_hip_a_joint"]
pre=[r for r in L if r["timestamp"]-t0<=5.0]
pr=st.mean(r["rp_deg"][0] for r in pre); pt={n:st.mean(tg(r,n) for r in pre) for n in J}
print(f"live {L[-1]['timestamp']-t0:.1f} s | pre roll {pr:+.2f}° | pre tgt aL {pt[J[0]]:+.2f} aR {pt[J[1]]:+.2f} hL {pt[J[2]]:+.2f} hR {pt[J[3]]:+.2f}")
# segments
segs=[]; cur=None
for r in L:
    t=r["timestamp"]-t0; d=r["rp_deg"][0]-pr
    if abs(d)>=thr:
        if cur is None: cur=[t,t,[]]
        cur[1]=t; cur[2].append(r)
    else:
        if cur is not None:
            if cur[1]-cur[0]>=hold: segs.append(cur)
            cur=None
if cur is not None and cur[1]-cur[0]>=hold: segs.append(cur)
print(f"유지 구간(|Δroll|≥{thr}°, ≥{hold}s): {len(segs)}개")
print("| 구간 s | 길이 s | Δroll ° | Δ발목 L/R ° | 발목 비대칭 (L−R)/2 | Δ힙 L/R ° | 힙 대칭 (L+R)/2 | 이득 발목비대칭 °/° | 이득 힙대칭 °/° | 지연 s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a,b,rows in segs:
    tail=rows[int(len(rows)*0.4):]
    dr=st.mean(r["rp_deg"][0] for r in tail)-pr
    dj={n:st.mean(tg(r,n) for r in tail)-pt[n] for n in J}
    aa=(dj[J[0]]-dj[J[1]])/2; hs=(dj[J[2]]+dj[J[3]])/2
    # delay: roll crosses thr/2 -> mode reaches 50% of final
    tc=None
    for r in L:
        t=r["timestamp"]-t0
        if a-3<=t<=b and abs(r["rp_deg"][0]-pr)>=thr/2: tc=t; break
    td=None
    mode=lambda r: ((tg(r,J[0])-pt[J[0]])-(tg(r,J[1])-pt[J[1]]))/2 if abs(aa)>=abs(hs) else ((tg(r,J[2])-pt[J[2]])+(tg(r,J[3])-pt[J[3]]))/2
    fin=aa if abs(aa)>=abs(hs) else hs
    if tc is not None and abs(fin)>0.3:
        for r in L:
            t=r["timestamp"]-t0
            if t>=tc and abs(mode(r))>=0.5*abs(fin) and (mode(r)*fin>0): td=t-tc; break
    print(f"| {a:.1f}~{b:.1f} | {b-a:.1f} | {dr:+.2f} | {dj[J[0]]:+.2f}/{dj[J[1]]:+.2f} | {aa:+.2f} | {dj[J[2]]:+.2f}/{dj[J[3]]:+.2f} | {hs:+.2f} | {aa/dr:+.3f} | {hs/dr:+.3f} | {'—' if td is None else f'{td:.2f}'} |")
