"""1006 회복 단계 분해 비교(트랙 B 병목 분석 에이전트의 분류 재사용).
사용: python3 stages_cmp.py <태그A> <태그B> cond1 cond2 ...  (seed 1·2, pre_pass env)
분류: F0 무스텝 낙상 · F1 착지 실패 · F2a 1보 뒤 낙상 · F2b 다보 뒤 낙상 · S3 비정착 · S4 자세 미복귀 · S5 추가 스텝/표류 · S6 진동 · OK.
엄격 성공 = OK + OK무스텝(사용자 정의: 정지·정착·자세 복귀). 짝 비교 = 같은 env 의 (F2b+S5) 여부."""
import csv, os, sys, math
LP = "/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{}_{}_s{}_envs.csv"
f = lambda x: None if x in ("", None) else float(x)


def load(t, c):
    R = {}
    for s in (1, 2):
        p = LP.format(t, c, s)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if r["pre_pass"] == "1":
                    R[(s, r["env"])] = r
    return R


def classify(r):
    fell = r["fell"] == "1"; to = f(r.get("to_first_t")); td = f(r.get("td_first_t")); td2 = f(r.get("td2_t"))
    if fell:
        if to is None: return "F0"
        if td is None: return "F1"
        if td2 is None: return "F2a"
        return "F2b"
    if r["final_quiet"] != "1": return "S3"
    dh = f(r.get("end_dh_cm")); dt = f(r.get("end_dtilt_deg")); dq = f(r.get("end_dq_deg"))
    if "end_dh_cm" in r and ((dh is not None and dh < -2) or (dt is not None and dt > 3) or (dq is not None and dq > 5)): return "S4"
    ac = f(r.get("to_after_cue")) or 0; dr = f(r.get("drift_after_cue_cm"))
    if ac >= 2 or (dr is not None and dr > 10): return "S5"
    if r.get("osc_flag_post") == "1": return "S6"
    return "OK"


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


A, B = sys.argv[1], sys.argv[2]
K = ["F0", "F1", "F2a", "F2b", "S3", "S4", "S5", "S6", "OK"]
print(f"조건 | 태그 | n | " + " ".join(K) + " | 낙상 % | 다보낙상+추가스텝 % | 엄격 성공 %")
for c in sys.argv[3:]:
    RA, RB = load(A, c), load(B, c)
    for tag, R in ((A, RA), (B, RB)):
        if not R:
            print(f"{c} | {tag} | 없음"); continue
        cnt = {k: 0 for k in K}
        for r in R.values(): cnt[classify(r)] += 1
        n = len(R)
        print(f"{c} | {tag} | {n} | " + " ".join(f"{100*cnt[k]/n:4.1f}" for k in K) +
              f" | {100*sum(cnt[k] for k in K[:4])/n:4.1f} | {100*(cnt['F2b']+cnt['S5'])/n:4.1f} | {100*cnt['OK']/n:4.1f}")
    keys = set(RA) & set(RB)
    if keys:
        bad = lambda r: classify(r) in ("F2b", "S5")
        b = sum(1 for k in keys if bad(RA[k]) and not bad(RB[k])); cc = sum(1 for k in keys if bad(RB[k]) and not bad(RA[k]))
        f2a = lambda r: classify(r) == "F2a"
        b2 = sum(1 for k in keys if f2a(RA[k]) and not f2a(RB[k])); c2 = sum(1 for k in keys if f2a(RB[k]) and not f2a(RA[k]))
        print(f"   짝(n {len(keys)}): 다보낙상+추가스텝 {A}만 {b} / {B}만 {cc} · p {mcnemar(b, cc):.3g}  |  1보 뒤 낙상 {A}만 {b2} / {B}만 {c2} · p {mcnemar(b2, c2):.3g}")
