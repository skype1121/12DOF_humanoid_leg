"""1005 밤 자세 복귀 집계: python3 posture.py <태그> [태그...] -c 조건... -s seed..."""
import sys, csv, os, statistics as st
args = sys.argv[1:]; tags = []; conds = []; seeds = [1, 2]; mode = "t"
for a in args:
    if a == "-c": mode = "c"; continue
    if a == "-s": mode = "s"; seeds = []; continue
    {"t": tags, "c": conds, "s": seeds}[mode].append(int(a) if mode == "s" else a)
f = lambda x: None if x in ("", None) else float(x)
q = lambda v: f"{st.median(v):.2f}/{sorted(v)[int(0.9*(len(v)-1))]:.2f}" if v else "—"
for c in conds:
    print(f"== {c}")
    for t in tags:
        R = []
        for sd in seeds:
            p = f"log_picture/ground_recovery_{t}_{c}_s{sd}_envs.csv"
            if os.path.exists(p): R += [r for r in csv.DictReader(open(p)) if r["pre_pass"] == "1"]
        if not R: continue
        al = [r for r in R if r["fell"] != "1"]
        rt = [f(r["rest_t"]) for r in al if f(r.get("rest_t")) is not None]
        nr = len(al) - len(rt)
        print(f"  {t:6s} n {len(R)} 낙상 {100*(len(R)-len(al))/len(R):.1f} % · 생존 {len(al)} 중 미복귀 {nr} ({100*nr/max(1,len(al)):.0f} %) · 복귀 시간 중앙/p90 {q(rt)} s · 끝 관절편차 {q([f(r['end_dq_deg']) for r in al if f(r.get('end_dq_deg')) is not None])}° · 끝 높이차 {q([f(r['end_dh_cm']) for r in al if f(r.get('end_dh_cm')) is not None])} cm · 끝 기울기차 {q([f(r['end_dtilt_deg']) for r in al if f(r.get('end_dtilt_deg')) is not None])}° · 끝 ω {q([f(r['end_w']) for r in al if f(r.get('end_w')) is not None])}")
