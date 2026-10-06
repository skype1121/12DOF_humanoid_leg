"""병목 2 폐루프 A/B 집계: KE 조기 복귀 판정기 imu(현행) vs odo θ0.07 / θ0.05 (settle_ab_run.sh 산출물).
짝 = 같은 (seed, env). 먼저 RNG 경로 동일성(bias_*·com_*·pre_pass·pre_pitch 열 일치)을 확인하고, 깨지면 그 조건은 짝 비교를 하지 않는다.
분류는 reports/gen_c5/stages_cmp.py 와 같은 식(엄격 성공 = OK).
사용: python3 settle_ab_cmp.py [out.txt]"""
import csv, os, sys, math
import numpy as np

PFX = os.environ.get("SAB_PFX", "sab")
LP = "/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_" + PFX + "_{}_{}_s{}_envs.csv"
SEEDS = (1, 2, 3, 4, 5, 6); CONDS = ("nd", "sf15", "fi40", "b18"); ARMS = tuple(os.environ.get("SAB_ARMS", "imu o07 o05").split())
f = lambda x: None if x in ("", None) else float(x)


def classify(r):      # stages_cmp.py 와 동일
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


def load(a, c, s):
    p = LP.format(a, c, s)
    return {r["env"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else None


def main(out):
    L = []
    P = lambda x="": (L.append(x), print(x, flush=True))
    P("# KE 조기 복귀 판정기 폐루프 A/B — imu(현행 기울기<6°·|ω|<0.4·양발 0.2 s) vs odo(|v̄|<θ·양발≥0.3 s·켜진 뒤≥0.9 s·0.2 s)")
    P("G11a 128 env × seed 1–6 · 지연 2 · 접촉 v2 · kabs KE 인자 · 사전검사 통과 env 만 · 짝 = 같은 (seed, env)")
    MET = ("abs_pitch", "abs_roll", "dq_nom_deg", "end_both", "nrm_t", "abn_frac", "to_after_cue", "to_after_off", "sw_time_s", "sw_on_n", "rec_off_t")
    for c in CONDS:
        D = {a: {} for a in ARMS}; miss = []
        for a in ARMS:
            for s in SEEDS:
                R = load(a, c, s)
                if R is None:
                    miss.append(f"{a}s{s}"); continue
                for e, r in R.items():
                    D[a][(s, e)] = r
        P(f"\n## {c}" + (f"  (없음: {' '.join(miss)})" if miss else ""))
        # RNG 경로 동일성
        rng_ok = {}
        for a in ARMS[1:]:
            keys = set(D["imu"]) & set(D[a]); bad = 0
            for k in keys:
                r0, r1 = D["imu"][k], D[a][k]
                cols = [x for x in r0 if isinstance(x, str) and (x.startswith("bias_") or x.startswith("com_"))] + ["pre_pass", "pre_pitch"]
                if any(r0.get(x) != r1.get(x) for x in cols):
                    bad += 1
            rng_ok[a] = (bad == 0 and len(keys) > 0)
            P(f"RNG 경로 동일성 imu↔{a}: 짝 {len(keys)} · 불일치 {bad} → {'OK' if rng_ok[a] else '깨짐(짝 비교 무효)'}")
        for a in ARMS:
            R = {k: r for k, r in D[a].items() if r["pre_pass"] == "1"}
            if not R:
                continue
            n = len(R); cls = [classify(r) for r in R.values()]
            fell = np.mean([r["fell"] == "1" for r in R.values()]); ok = np.mean([x == "OK" for x in cls])
            # seed 별 변동
            fs = [np.mean([r["fell"] == "1" for k, r in R.items() if k[0] == s]) for s in SEEDS if any(k[0] == s for k in R)]
            oks = [np.mean([classify(r) == "OK" for k, r in R.items() if k[0] == s]) for s in SEEDS if any(k[0] == s for k in R)]
            surv = [r for r in R.values() if r["fell"] != "1"]
            def mm(col, rows=surv):
                v = [f(r.get(col)) for r in rows]; v = [x for x in v if x is not None and not math.isnan(x)]
                return (np.mean(v) if v else float("nan"))
            cnt = {k: cls.count(k) for k in ("F0", "F1", "F2a", "F2b", "S3", "S4", "S5", "S6")}
            P(f"{a:4s} n {n:4d} | 낙상 {100*fell:5.1f}% (seed sd {100*np.std(fs):.1f}) | 엄격 {100*ok:5.1f}% (seed sd {100*np.std(oks):.1f}) | "
              + " ".join(f"{k}{v}" for k, v in cnt.items()) + " |생존 평균: " + " ".join(f"{m} {mm(m):.3g}" for m in MET))
        for a in ARMS[1:]:
            if not rng_ok.get(a):
                continue
            keys = [k for k in set(D["imu"]) & set(D[a]) if D["imu"][k]["pre_pass"] == "1"]
            for nm, fn in (("낙상", lambda r: r["fell"] == "1"), ("엄격 성공", lambda r: classify(r) == "OK"),
                           ("다보낙상+추가스텝", lambda r: classify(r) in ("F2b", "S5"))):
                b = sum(1 for k in keys if fn(D["imu"][k]) and not fn(D[a][k])); cc = sum(1 for k in keys if fn(D[a][k]) and not fn(D["imu"][k]))
                P(f"   짝 imu↔{a} {nm}: imu만 {b} / {a}만 {cc} · McNemar p {mcnemar(b, cc):.3g}")
            # 같은 env 생존 짝의 연속 지표 차(odo − imu)
            both = [k for k in keys if D["imu"][k]["fell"] != "1" and D[a][k]["fell"] != "1"]
            for col in ("sw_time_s", "to_after_off", "abs_pitch", "dq_nom_deg", "nrm_t", "abn_frac"):
                d = [f(D[a][k].get(col)) - f(D["imu"][k].get(col)) for k in both if f(D[a][k].get(col)) is not None and f(D["imu"][k].get(col)) is not None]
                d = [x for x in d if not math.isnan(x)]
                if d:
                    P(f"   짝 생존 {len(d)} {col}: {a}−imu 평균 {np.mean(d):+.3f} (95% ±{1.96*np.std(d)/math.sqrt(len(d)):.3f})")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
