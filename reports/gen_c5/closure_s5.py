"""회복 closure 분해: 회복 뒤 서기 복귀 실패(S3 비정착·S5 추가 스텝/표류)가 어느 신호에서 나오는지."""
import sys, math
src = open(__file__.replace("closure_s5.py", "stages_cmp.py")).read().split("A, B = sys.argv")[0]
ns = {}; exec(src, ns)
f = lambda r, k: float(r[k]) if r.get(k) not in ("", None) else float("nan")
for tag in sys.argv[1].split(","):
    for c in sys.argv[2].split(","):
        R = ns["load"](tag, c)
        if not R: continue
        cl = {k: ns["classify"](r) for k, r in R.items()}
        S5 = [R[k] for k in R if cl[k] == "S5"]; S3 = [R[k] for k in R if cl[k] == "S3"]; OK = [R[k] for k in R if cl[k] == "OK"]
        ac = [f(r, "to_after_cue") for r in S5]; dr = [f(r, "drift_after_cue_cm") for r in S5]
        od = sum(1 for a, d in zip(ac, dr) if a < 2 and d > 10); os_ = sum(1 for a, d in zip(ac, dr) if a >= 2 and not d > 10)
        med = lambda v: sorted(x for x in v if x == x)[len([x for x in v if x == x]) // 2] if any(x == x for x in v) else float("nan")
        print(f"{tag} {c}: n {len(R)} | S5 {len(S5)}(표류만 {od}·추가스텝만 {os_}·둘다 {len(S5)-od-os_}, 추가이륙 중앙 {med(ac):.0f}, 표류 중앙 {med(dr):.1f} cm, cue 길이 중앙 {med([f(r,'cue_time_s') for r in S5]):.2f}s)"
              f" | S3 {len(S3)}(end_w 중앙 {med([f(r,'end_w') for r in S3]):.2f}) | OK {len(OK)}(cue {med([f(r,'cue_time_s') for r in OK]):.2f}s, 추가이륙 {med([f(r,'to_after_cue') for r in OK]):.0f}, 표류 {med([f(r,'drift_after_cue_cm') for r in OK]):.1f} cm)")

if len(sys.argv) > 3:   # 착지 사슬: 보폭·포획 여유(전방 mf, 측방 ml)를 S5 vs OK 로
    for tag in sys.argv[1].split(","):
        for c in sys.argv[2].split(","):
            R = ns["load"](tag, c)
            if not R: continue
            cl = {k: ns["classify"](r) for k, r in R.items()}
            for g in ("OK", "S5", "F2b"):
                L = [R[k] for k in R if cl[k] == g]
                if not L: continue
                med = lambda key: (lambda v: sorted(v)[len(v) // 2] if v else float("nan"))([f(r, key) for r in L if f(r, key) == f(r, key)])
                print(f"  {tag} {c} {g:3s} n {len(L):3d} | 1보: t {med('td1_t'):.2f}s 보폭 {med('td1_len_cm'):+.1f} 여유 앞 {med('td1_mf_cm'):+.1f} 옆 {med('td1_ml_cm'):+.1f} cm | 2보: t {med('td2_t'):.2f}s 보폭 {med('td2_len_cm'):+.1f} 여유 앞 {med('td2_mf_cm'):+.1f} 옆 {med('td2_ml_cm'):+.1f} cm")
