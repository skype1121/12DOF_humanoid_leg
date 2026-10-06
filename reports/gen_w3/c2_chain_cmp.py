"""회복 체인 판정(CPU, 승윤님 회복 정의 1006) — eval_ground_recovery CSV 로 env 별 단계 통과:
  ① 회복 스텝(외란 조건: 첫 이륙 있음; nd 는 해당 없음) → ② 운동량 억제(정지 신호 뒤 이륙 < 2 & 표류 ≤ 10 cm — KE 'cue 종료', c2 '첫 정지 요청')
  → ③ 양발(end_both = 1) → ④ 수평 기준 정상 자세(nrm_t 있음 = |피치|·|롤| < 3°·|ω| < 0.2·양발·명목 관절 < 5°·높이 ≥ −2 cm 0.5 s, 그리고 abn_frac ≤ 0.10)
  → ⑤ STAND 복귀(rec_off_t 있음 = KE 서기 인계 / c2 ge 0 래치; nd 는 해당 없음). 체인 성공 = 낙상 없음 & ①–⑤.
사용: python3 c2_chain_cmp.py <태그A> <태그B> cond... (seed 1·2, 사전 검사 통과 env). 주의: eval_ground_recovery 의 nrm_t 0.5 s 지속은
1006 밤 중복 블록 수정 뒤 코드에서만 정확(그 전 KABS2 는 0.25 s) — 비교는 같은 코드로 다시 잰 태그끼리."""
import csv, os, sys
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
f = lambda x: None if x in ("", None) else float(x)


def load(t, c):
    R = {}
    for s in (1, 2, 3, 4, 5, 6):
        p = f"{LP}/ground_recovery_{t}_{c}_s{s}_envs.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r["pre_pass"] == "1":
                    R[(s, r["env"])] = r
    return R


def stages(r, nd):
    if r["fell"] == "1":
        return [False] * 5, False
    s1 = True if nd else f(r.get("to_first_t")) is not None
    ac = f(r.get("to_after_cue")) or 0.0; dr = f(r.get("drift_after_cue_cm"))
    s2 = (ac < 2) and (dr is None or dr <= 10.0)
    s3 = r.get("end_both") in ("1", "1.0")
    abn = f(r.get("abn_frac"))
    if nd:      # nd: 평가기가 nrm_t 를 외란 조건에서만 계산 → 끝 자세(수평·명목 관절·양발)와 abn_frac 로 판정(같은 문턱)
        ap, ar, dq = f(r.get("abs_pitch")), f(r.get("abs_roll")), f(r.get("dq_nom_deg"))
        s4 = (ap is not None and abs(ap) < 3.0 and abs(ar) < 3.0 and dq < 5.0) and (abn is None or abn <= 0.10)
    else:
        s4 = f(r.get("nrm_t")) is not None and (abn is None or abn <= 0.10)
    s5 = True if nd else f(r.get("rec_off_t")) is not None
    st = [s1, s2, s3, s4, s5]
    return st, all(st)


tags = sys.argv[1:3]; conds = sys.argv[3:]
print("조건 | 태그 | n | 낙상 % | ①스텝 ②억제 ③양발 ④정상자세 ⑤복귀 (누적 통과 %) | 체인 성공 % | abs 피치/롤 중앙 · 명목편차 중앙 · abn_frac 평균")
for c in conds:
    for t in tags:
        R = load(t, c)
        if not R:
            print(f"{c} | {t} | 없음"); continue
        n = len(R); nd = c.startswith("nd")
        cum = [0] * 5; ok = 0; fell = 0
        for r in R.values():
            st, a = stages(r, nd)
            fell += r["fell"] == "1"; ok += a
            run = True
            for k in range(5):
                run = run and st[k]; cum[k] += run
        med = lambda key: (lambda v: sorted(v)[len(v) // 2] if v else float("nan"))([abs(f(r[key])) for r in R.values() if r["fell"] != "1" and f(r.get(key)) is not None])
        abn = [f(r["abn_frac"]) for r in R.values() if r["fell"] != "1" and f(r.get("abn_frac")) is not None]
        print(f"{c} | {t} | {n} | {100*fell/n:5.1f} | " + " ".join(f"{100*x/n:5.1f}" for x in cum) + f" | {100*ok/n:5.1f} | "
              f"{med('abs_pitch'):.2f}/{med('abs_roll'):.2f}° · {med('dq_nom_deg'):.2f}° · {sum(abn)/max(len(abn),1):.3f}")
