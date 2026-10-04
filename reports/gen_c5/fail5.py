"""1005 회복 활성화 실패 5분류: python3 fail5.py <태그> [태그...] -c 조건... -s seed...
① 낙상 & 외란 뒤 이륙 없음 ② 이륙 & 착지 없이 낙상 ③a 착지 후 낙상 ③b 생존 비정숙 ④ 생존 & (cue 종료 뒤 이륙 ≥2 또는 drift >10 cm) ⑤ 서기 복귀 뒤 진동.
정착 시간 = 회복 모드 해제 시각(rec_off_t) − 첫 외란(2.0 s)."""
import sys, csv, os, statistics as st
args = sys.argv[1:]; tags = []; conds = []; seeds = [1, 2]; mode = "t"
for a in args:
    if a == "-c": mode = "c"; continue
    if a == "-s": mode = "s"; seeds = []; continue
    {"t": tags, "c": conds, "s": seeds}[mode].append(int(a) if mode == "s" else a)
f = lambda x: None if x in ("", None) else float(x)
for c in conds:
    print(f"== {c}")
    for t in tags:
        R = []
        for sd in seeds:
            p = f"log_picture/ground_recovery_{t}_{c}_s{sd}_envs.csv"
            if os.path.exists(p): R += [r for r in csv.DictReader(open(p)) if r["pre_pass"] == "1"]
        if not R: print(f"  {t}: 없음"); continue
        n = len(R); k = dict(F1=0, F2=0, F3a=0, F3b=0, F4=0, F5=0, ok=0)
        settle = []; cue = []; drift = []; tak = []
        for r in R:
            fell = r["fell"] == "1"; to = f(r.get("to_first_t")); td = f(r.get("td_first_t"))
            if fell:
                if to is None: k["F1"] += 1
                elif td is None: k["F2"] += 1
                else: k["F3a"] += 1
                continue
            if r["final_quiet"] != "1": k["F3b"] += 1; continue
            dr = f(r.get("drift_after_cue_cm")); ac = f(r.get("to_after_cue")) or 0
            if (ac >= 2) or (dr is not None and dr > 10): k["F4"] += 1; continue
            if r.get("osc_flag_post") == "1": k["F5"] += 1; continue
            k["ok"] += 1
        for r in R:
            if r["fell"] == "1": continue
            ro = f(r.get("rec_off_t"))
            if ro is not None: settle.append(ro - 2.0)
            cue.append(f(r.get("cue_time_s")) or 0.0); tak.append(f(r["takeoffs"]) or 0)
            if f(r.get("drift_after_cue_cm")) is not None: drift.append(f(r["drift_after_cue_cm"]))
        fall = k["F1"] + k["F2"] + k["F3a"]
        md = lambda v: f"{st.median(v):.2f}" if v else "—"
        print(f"  {t:8s} n {n:3d} 낙상 {100*fall/n:5.1f} % [①{k['F1']} ②{k['F2']} ③a{k['F3a']}] ③b {k['F3b']} ④ {k['F4']} ⑤ {k['F5']} 정상 {100*k['ok']/n:5.1f} % · 정착 중앙 {md(settle)} s (n {len(settle)}) · cue {md(cue)} s · 이륙 {md(tak)} · drift {md(drift)} cm")
