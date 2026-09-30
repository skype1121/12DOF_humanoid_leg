import csv,math,os
os.chdir('/home/ryu/humanoid_leg_test1/log_picture')
def f(x): return float(x) if x not in ('',None) else math.nan
print(f"{'조건':16s} {'n':>4s} {'낙상(t>3)':>9s} {'낙상 전체':>9s} {'생존 성장':>9s} {'성장 합':>8s} {'감쇠':>7s} {'지속흔들림':>9s} {'말기 p2p 중앙':>12s}")
for ctrl,dls in (('g12',(6,11,14,17)),('hold',(11,17))):
  for dl in dls:
    for ds in ('nd','f8','b8'):
      n=fall3=fall=grow=damp=sus=0; late=[]
      for sd in (1,2,3):
        p=f'ground_recovery_osc_{ctrl}_d{dl}_{ds}_s{sd}_envs.csv'
        for r in csv.DictReader(open(p)):
          if r['pre_pass']!='1': continue
          n+=1; pp=[f(r[f'p2p_{k}']) for k in range(14)]
          if r['fell']=='1':
            fall+=1; t=f(r['t_fall_rel'])
            if t>3: fall3+=1
            continue
          early=max(pp[1:4]); lt=max(pp[6:14]); late.append(max(pp[10:14]))
          if lt>=4 and lt>=2*early: grow+=1
          if max(pp[10:14])<=2: damp+=1
          else: sus+=1
      late.sort(); med=late[len(late)//2] if late else math.nan
      pc=lambda x: f"{x}({100*x/n:.1f}%)"
      print(f"{ctrl+' d'+str(dl)+' '+ds:16s} {n:4d} {pc(fall3):>9s} {pc(fall):>9s} {pc(grow):>9s} {pc(grow+fall3):>8s} {pc(damp):>7s} {pc(sus):>9s} {med:12.2f}")
