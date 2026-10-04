"""고정 스코어카드 (1004 밤): 태그별 C5 평가 결과를 같은 조건·seed 로 집계.
사용: scorecard.py --seeds 1 2 --conds sf12 sf15 fi40 fs80 fi60 nd -- <태그...>
낙상 % = 사전 검사 통과 env 중 낙상, nd/d3nd 는 흔들림 지속 %(p2p 5–8 s > 2°), 끝까지 성공 % 병기 안 함(필요 시 e2e.py)."""
import csv, os, sys
args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); j = i + 1; out = []
        while j < len(args) and not args[j].startswith("--"):
            out.append(args[j]); j += 1
        return out
    return default
seeds = [int(x) for x in opt("--seeds", ["1", "2"])]
conds = opt("--conds", ["sf12", "sf15", "fi40", "fs80", "fi60", "nd"])
tags = args[args.index("--") + 1:] if "--" in args else []
f = lambda x: float(x) if x not in ("", None) else float("nan")
def cell(tag, c):
    n = k = 0
    for sd in seeds:
        p = f"/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{tag}_{c}_s{sd}_envs.csv"
        if not os.path.exists(p): return None
        for r in csv.DictReader(open(p)):
            if r["pre_pass"] != "1": continue
            n += 1
            if c in ("nd", "d3nd"):
                k += r["fell"] != "1" and max(f(r[f"p2p_{i}"]) for i in (5, 6, 7)) > 2
            else:
                k += r["fell"] == "1"
    return 100.0 * k / max(n, 1)
print("| 태그 | " + " | ".join(conds) + " |"); print("|---|" + "---|" * len(conds))
for t in tags:
    vals = [cell(t, c) for c in conds]
    print(f"| {t} | " + " | ".join("—" if v is None else f"{v:.1f}" for v in vals) + " |")
