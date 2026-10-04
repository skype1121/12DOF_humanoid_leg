"""끝까지 성공률 = 사전 검사 통과 & 미낙상 / 전체 env (검토 4회차: 홀드·회복 민감도 반대 방향 → 둘 다 반영). 사용: e2e.py <태그> [조건...] (seed 1–3)"""
import sys, csv, os
T = sys.argv[1]
for c in sys.argv[2:] or ["sf12", "sf15"]:
    n = ok = pp = 0
    for sd in (1, 2, 3):
        f = f"/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{T}_{c}_s{sd}_envs.csv"
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f)):
            n += 1; pp += r["pre_pass"] == "1"; ok += r["pre_pass"] == "1" and r["fell"] != "1"
    print(f"{T} {c}: 사전 검사 통과 {pp}/{n} · 끝까지 성공 {ok}/{n} = {100*ok/max(n,1):.1f} %")
