"""실물(공중 손굴리기) vs 시뮬(공중 롤 램프) 정책 관측 항별 대조 + 반사실 교체(어느 항이 응답 크기를 가르는가).
사용: python3 obs_counterfactual.py <real jsonl> <sim glob prefix e.g. log_picture/simbb_airobs_v5_rampR6> [--real_rest a,b] [--real_hold a,b]
"""
import json, sys, glob, numpy as np, torch
real=sys.argv[1]; simpre=sys.argv[2]
def arg(k,d):
    return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
rr=[float(x) for x in arg("--real_rest","44,54").split(",")]; rh=[float(x) for x in arg("--real_hold","36,40").split(",")]
sp=[float(x) for x in arg("--sim_pre","0.5,2.5").split(",")]; sq=[float(x) for x in arg("--sim_plateau","30,39").split(",")]
TERMS=[("base_ang_vel",3),("projected_gravity",3),("velocity_commands",3),("joint_pos",12),("joint_vel",12),("actions",12),("gait_clock",2),("feet_contact",2),("gyro_ema",3),("joint_vel_ema",12)]
H=5; OFF={}; o=0
for n,d in TERMS: OFF[n]=(o,o+H*d,d); o+=H*d
assert o==320
pol=torch.jit.load(arg("--policy","models/stand_hang_v5/policy.pt"),map_location="cpu"); pol.eval()
def act(obs):
    with torch.no_grad(): return pol(torch.tensor(np.asarray(obs,dtype=np.float32)).unsqueeze(0)).squeeze(0).numpy()
def load(path, live_only=True):
    R=[json.loads(l) for l in open(path) if l.strip()]
    return [r for r in R if r.get("live") and isinstance(r.get("obs"),list) and len(r["obs"])==320]
def window(L,a,b):
    t0=L[0]["timestamp"]; return [r for r in L if a<=r["timestamp"]-t0<=b]
def term(obs,n,frame=-1):
    a,b,d=OFF[n]; return np.array(obs[a:b]).reshape(H,d)[frame]
def set_term(obs,n,val_frame):   # 모든 이력 프레임을 같은 값으로
    a,b,d=OFF[n]; obs=np.array(obs,dtype=np.float32).copy(); obs[a:b]=np.tile(np.asarray(val_frame,dtype=np.float32),H); return obs
J=["hip_f L","hip_f R","hip_a L","hip_a R","hip_r L","hip_r R","knee L","knee R","ankle_f L","ankle_f R","ankle_r L","ankle_r R"]
SC=np.degrees(0.25)   # action → target deg
def summ(da):   # 관심 관절 Δ목표(°)
    return f"ankle_r L/R {da[10]*SC:+.2f}/{da[11]*SC:+.2f} · hip_a L/R {da[2]*SC:+.2f}/{da[3]*SC:+.2f} · |Δ|max {np.abs(da).max()*SC:.2f}°"
RL=load(real); Rrest=window(RL,*rr); Rhold=window(RL,*rh)
sims=[load(f) for f in sorted(glob.glob(simpre+"_env*.jsonl"))]
Spre=[r for L in sims for r in window(L,*sp)]; Splat=[r for L in sims for r in window(L,*sq)]
print(f"real rest {len(Rrest)} ticks / hold {len(Rhold)} ticks | sim envs {len(sims)} pre {len(Spre)} plateau {len(Splat)} ticks")
def mean_obs(W): return np.mean([r["obs"] for r in W],0)
mR,mH,mP,mQ=mean_obs(Rrest),mean_obs(Rhold),mean_obs(Spre),mean_obs(Splat)
print("\n## 항별 평균(최신 프레임) real rest | real hold | sim pre | sim plateau")
for n,d in TERMS:
    print(f"{n:18s} R.rest {np.round(term(mR,n),3).tolist()}\n{'':18s} R.hold {np.round(term(mH,n),3).tolist()}\n{'':18s} S.pre  {np.round(term(mP,n),3).tolist()}\n{'':18s} S.plat {np.round(term(mQ,n),3).tolist()}")
print("\n## 정책 출력(평균 obs 에 대한 네트워크 출력; Δ = 대비 rest/pre)")
aR,aH,aP,aQ=act(mR),act(mH),act(mP),act(mQ)
print("real hold − real rest :",summ(aH-aR)); print("sim plateau − sim pre :",summ(aQ-aP))
gR=term(mR,"projected_gravity"); gH=term(mH,"projected_gravity"); gP=term(mP,"projected_gravity"); gQ=term(mQ,"projected_gravity")
print("gravity real rest→hold",np.round(gR,3),"→",np.round(gH,3)," sim pre→plat",np.round(gP,3),"→",np.round(gQ,3))
print("\n## 반사실 1: 중력만 교체(이력 5프레임 전부)")
print("real rest + real hold gravity     :",summ(act(set_term(mR,"projected_gravity",gH))-aR))
print("real rest + sim plateau gravity   :",summ(act(set_term(mR,"projected_gravity",gQ))-aR))
print("sim pre   + sim plateau gravity   :",summ(act(set_term(mP,"projected_gravity",gQ))-aP))
print("sim pre   + real hold gravity     :",summ(act(set_term(mP,"projected_gravity",gH))-aP))
print("\n## 반사실 2: real rest(+real hold 중력) 에서 항 하나씩 sim pre 값으로 교체 → 응답이 커지는 항 = 실물이 시뮬과 다른 '맥락'")
base=set_term(mR,"projected_gravity",gH); a0=act(base)
for n,d in TERMS:
    if n=="projected_gravity": continue
    ob=set_term(base,n,term(mP,n)); print(f"  swap {n:16s} → Δ vs real-rest: {summ(act(ob)-aR)}   (vs 미교체 |Δ|max {np.abs(a0-aR).max()*SC:.2f}°)")
print("\n## 반사실 3: sim pre(+sim plateau 중력) 에서 항 하나씩 real rest 값으로 교체 → 응답이 죽는 항")
base=set_term(mP,"projected_gravity",gQ); a0=act(base)
for n,d in TERMS:
    if n=="projected_gravity": continue
    ob=set_term(base,n,term(mR,n)); print(f"  swap {n:16s} → Δ vs sim-pre: {summ(act(ob)-aP)}   (vs 미교체 |Δ|max {np.abs(a0-aP).max()*SC:.2f}°)")
print("\n## 실제 틱 단위(평균 아님): real hold 틱들의 출력 − real rest 평균 출력 (분포)")
dh=np.array([act(r["obs"])-aR for r in Rhold]); print("  real hold ticks: ankle_r L/R mean %+.2f/%+.2f°, hip_a L/R %+.2f/%+.2f°"%(dh[:,10].mean()*SC,dh[:,11].mean()*SC,dh[:,2].mean()*SC,dh[:,3].mean()*SC))
dq=np.array([act(r["obs"])-aP for r in Splat]); print("  sim plateau ticks: ankle_r L/R mean %+.2f/%+.2f°, hip_a L/R %+.2f/%+.2f°"%(dq[:,10].mean()*SC,dq[:,11].mean()*SC,dq[:,2].mean()*SC,dq[:,3].mean()*SC))
