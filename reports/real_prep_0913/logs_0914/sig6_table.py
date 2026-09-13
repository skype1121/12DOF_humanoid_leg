import re, sys, glob
S="/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/evals/"
rows={}
for f in ["sig3_chain.log","sig5_chain.log","sig6_chain.log"]:
    for line in open(S+f, encoding="utf-8", errors="replace"):
        m=re.match(r"(sig\d)_(v5|v6|v6d)_(\S+): (\d+)/(\d+) \(([\d.]+)%\)",line)
        if not m: continue
        cond=m.group(3); pol=m.group(2)
        post=re.search(r"방출 후 낙상 (\d+) / 방출 전 낙상 (\d+)",line)
        roll=re.search(r"최대 \|롤\| 평균 ([\d.]+)°",line)
        ronly=re.search(r"오른만 (\d+)",line)
        # sig6 lines truncated the contact string; read from result file for right-only count
        rf=f"log_picture/hang2stand_{m.group(1)}_{pol}_{cond}.txt"
        try:
            txt=open(rf).read(); ro=re.search(r"오른만 (\d+)",txt); ronly=ro
        except FileNotFoundError: pass
        cell=f"{m.group(4)}/{m.group(5)}"
        if post and int(post.group(1))+int(post.group(2))>0: cell+=f" (후 {post.group(1)}·전 {post.group(2)})"
        if roll: cell+=f" 롤 {roll.group(1)}°"
        if ronly and int(ronly.group(1))>0: cell+=f" 오른만 {ronly.group(1)}"
        rows.setdefault(cond,{})[pol]=cell
order=["g05","g05_comyR02","g05_comyR03","g05_comyR04","g04_comyR02","g04_comyR03","g07_comyR04","g07_comyR06","g05_hookL05","g05_hookL05_comyR02","hookL0.03","hookL0.05","hookL0.08","hookR0.05","hookL0.05_g07"]
label={"g05":"kp ×0.5 단독","g05_comyR02":"kp ×0.5 + CoM 오른 2 cm","g05_comyR03":"kp ×0.5 + CoM 오른 3 cm","g05_comyR04":"kp ×0.5 + CoM 오른 4 cm","g04_comyR02":"kp ×0.4 + CoM 오른 2 cm","g04_comyR03":"kp ×0.4 + CoM 오른 3 cm","g07_comyR04":"kp ×0.7 + CoM 오른 4 cm","g07_comyR06":"kp ×0.7 + CoM 오른 6 cm","g05_hookL05":"kp ×0.5 + 후크 왼쪽 5 cm","g05_hookL05_comyR02":"kp ×0.5 + 후크 왼 5 + CoM 오른 2","hookL0.03":"후크 왼쪽 3 cm","hookL0.05":"후크 왼쪽 5 cm","hookL0.08":"후크 왼쪽 8 cm","hookR0.05":"후크 오른쪽 5 cm","hookL0.05_g07":"후크 왼쪽 5 cm + kp ×0.7"}
print("| 조건 (부분지지 r0 0.5 홀드 15 s → 방출 4 s, 명목 64 env) | v5 | v6 | v6d |")
print("|---|---|---|---|")
for c in order:
    if c not in rows and c not in label: continue
    r=rows.get(c,{})
    print(f"| {label.get(c,c)} | {r.get('v5','—')} | {r.get('v6','—')} | {r.get('v6d','—')} |")
missing=[c for c in rows if c not in order]
if missing: print("(미배치 조건:", missing, ")")
