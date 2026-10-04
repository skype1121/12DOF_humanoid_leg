"""C5 전환 기준 집계: 정상 서기 오탐(t>0.5 s 첫 켜짐, 외란 전), 채터링(외란 없는 조건 env 당 켜짐 ≥2), 복귀 뒤 재발산(외란 뒤 켜짐 ≥2 또는 서기 복귀 후 낙상).
사용: python3 switch_stats.py <태그> [seed...]"""
import sys, csv, os
T = sys.argv[1]; SEEDS = [int(x) for x in sys.argv[2:]] or [1, 2, 3, 4]
def rows(c, sd):
    p = f"/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{T}_{c}_s{sd}_envs.csv"
    return [r for r in csv.DictReader(open(p)) if r["pre_pass"] == "1"] if os.path.exists(p) else []
f = lambda x: float(x) if x not in ("", None) else None
for c in ("nd", "d3nd"):
    R = [r for sd in SEEDS for r in rows(c, sd)]
    if not R: continue
    fa = sum(1 for r in R if f(r["sw_first_t"]) is not None and f(r["sw_first_t"]) > 0.5)
    late_on = sum(1 for r in R if (f(r["sw_on_n"]) or 0) - (1 if f(r["sw_first_t"]) is not None and f(r["sw_first_t"]) <= 0.5 else 0) >= 1)
    chat = sum(1 for r in R if (f(r["sw_on_n"]) or 0) >= 2)
    print(f"{c}: n {len(R)} · 정상 서기 오탐(첫 켜짐 t>0.5 s) {fa} ({100*fa/len(R):.1f} %) · 인계 뒤 켜짐 있는 env {late_on} ({100*late_on/len(R):.1f} %) · 켜짐 ≥2 (채터링 후보) {chat}")
for c in ("sf10", "sf12", "sf15", "sb10", "b18", "sl8"):
    R = [r for sd in SEEDS for r in rows(c, sd)]
    if not R: continue
    post = lambda r: (f(r["sw_on_n"]) or 0) - (f(r["sw_on_pre"]) or 0)
    re = sum(1 for r in R if post(r) >= 2)
    surv = [r for r in R if r["fell"] != "1" and post(r) >= 1]
    print(f"{c}: n {len(R)} · 낙상 {sum(r['fell']=='1' for r in R)} · 외란 뒤 켜짐 env {sum(1 for r in R if post(r)>=1)} · 외란 뒤 재켜짐(복귀 뒤 재발산 후보) {re} ({100*re/len(R):.1f} %) · 회복 정책 사용 시간 평균 {sum(f(r['sw_time_s']) for r in R)/len(R):.2f} s")
