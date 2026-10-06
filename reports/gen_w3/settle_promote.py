"""STOPEST §8/§8a 사전 승격 기준 P1–P5 판정(KE 종료 판정기 odi/odic vs imu) · §11 odicw(같은 P1–P5 + 보조 odicw vs odic).
사용: python3 settle_promote.py [out.txt] [팔 ...]   — 지연 2 = 태그 sab_*, 지연 3 = sabd3_*. 팔 생략 = odi odic(§9 재현).
      python3 settle_promote.py stopest_results/settle_promote_odicw.txt odicw   — §11
"""
import csv, os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LPD = "/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{}_{}_{}_s{}_envs.csv"
SEEDS = (1, 2, 3, 4, 5, 6); DIST = ("sf15", "fi40", "b18")
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


def load(pfx, arm, c):
    D = {}
    for s in SEEDS:
        p = LPD.format(pfx, arm, c, s)
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p)):
            D[(s, r["env"])] = r
    return D


def rng_same(A, B):
    if set(A) != set(B): return False
    for k in A:
        cols = [x for x in A[k] if isinstance(x, str) and (x.startswith("bias_") or x.startswith("com_"))] + ["pre_pass", "pre_pitch"]
        if any(A[k].get(x) != B[k].get(x) for x in cols): return False
    return True


def judge(pfx, arm, strict_p1, P):
    res = {}; okall = True
    data = {c: (load(pfx, "imu", c), load(pfx, arm, c)) for c in ("nd",) + DIST}
    if any(a is None or b is None for a, b in data.values()):
        P(f"  {arm}: 자료 부족 — 판정 보류"); return None
    for c, (A, B) in data.items():
        if not rng_same(A, B):
            P(f"  {arm}: {c} RNG 짝 불일치 → 판정 무효"); return None
    keys = {c: [k for k in data[c][0] if data[c][0][k]["pre_pass"] == "1"] for c in data}
    # P1
    b = cc = 0
    for c in DIST:
        A, B = data[c]
        b += sum(1 for k in keys[c] if A[k]["fell"] == "1" and B[k]["fell"] != "1")
        cc += sum(1 for k in keys[c] if B[k]["fell"] == "1" and A[k]["fell"] != "1")
    p1p = mcnemar(cc, b)
    p1 = (cc < b and p1p < 0.05) if strict_p1 else (cc <= b)
    P(f"  P1 낙상 짝(외란 3 조건): imu만 {b} / {arm}만 {cc} · p {p1p:.3g} → {'PASS' if p1 else 'FAIL'}")
    # P2
    p2 = True; parts = []
    for c in DIST:
        A, B = data[c]; ka = keys[c]
        ra = np.mean([classify(A[k]) == "OK" for k in ka]); rb = np.mean([classify(B[k]) == "OK" for k in ka])
        parts.append(f"{c} {100*ra:.1f}→{100*rb:.1f}"); p2 &= (rb - ra) >= -0.02
    bo = sum(1 for c in DIST for k in keys[c] if classify(data[c][0][k]) == "OK" and classify(data[c][1][k]) != "OK")
    co = sum(1 for c in DIST for k in keys[c] if classify(data[c][1][k]) == "OK" and classify(data[c][0][k]) != "OK")
    worse_sig = co < bo and mcnemar(bo, co) < 0.05
    p2 &= not worse_sig
    P(f"  P2 엄격 성공: {' · '.join(parts)} · 짝 imu만 {bo} / {arm}만 {co} (p {mcnemar(bo, co):.3g}) → {'PASS' if p2 else 'FAIL'}")
    # P3
    p3 = True; parts = []
    for col, lim in (("abs_pitch", 0.5), ("dq_nom_deg", 0.5), ("abn_frac", 0.02)):
        d = []
        for c in DIST:
            A, B = data[c]
            for k in keys[c]:
                if A[k]["fell"] != "1" and B[k]["fell"] != "1":
                    x, y = f(A[k].get(col)), f(B[k].get(col))
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        d.append(y - x)
        up = np.mean(d) + 1.96 * np.std(d) / math.sqrt(len(d))
        parts.append(f"{col} 차 {np.mean(d):+.3f} 상한 {up:+.3f}(≤{lim})"); p3 &= up <= lim
    P(f"  P3 끝 자세: {' · '.join(parts)} → {'PASS' if p3 else 'FAIL'}")
    # P4
    la = []; lb = []; dt = []
    for c in DIST:
        A, B = data[c]
        for k in keys[c]:
            if A[k]["fell"] != "1": la.append(f(A[k]["sw_time_s"]) > 4.0)
            if B[k]["fell"] != "1": lb.append(f(B[k]["sw_time_s"]) > 4.0)
            if A[k]["fell"] != "1" and B[k]["fell"] != "1": dt.append(f(B[k]["sw_time_s"]) - f(A[k]["sw_time_s"]))
    p4 = (np.mean(lb) <= np.mean(la) + 0.03) and (np.mean(dt) <= 1.0)
    P(f"  P4 장기 체류(>4 s): imu {100*np.mean(la):.1f}% · {arm} {100*np.mean(lb):.1f}% · 회복 시간 차 {np.mean(dt):+.2f} s → {'PASS' if p4 else 'FAIL'}")
    # P5
    A, B = data["nd"]; kn = keys["nd"]
    fa = np.mean([B[k]["fell"] == "1" for k in kn]); sa = np.mean([f(A[k]["sw_on_n"]) for k in kn]); sb = np.mean([f(B[k]["sw_on_n"]) for k in kn])
    oa = np.mean([classify(A[k]) == "OK" for k in kn]); ob = np.mean([classify(B[k]) == "OK" for k in kn])
    p5 = fa == 0 and sb <= sa + 0.02 and ob >= oa - 0.01
    P(f"  P5 무외란: 낙상 {100*fa:.1f}% · 켜짐 {sa:.3f}→{sb:.3f} · 엄격 {100*oa:.1f}→{100*ob:.1f} → {'PASS' if p5 else 'FAIL'}")
    allp = bool(p1 and p2 and p3 and p4 and p5)
    P(f"  ⇒ {arm}: {'통과' if allp else '불통과'}")
    return allp


def secondary(pfx, arm, ref, P):
    """§11 보조(판정 아님): arm vs ref 짝(둘 다 생존, 외란 3 조건) — 끈 뒤 진동·재켜짐(sw_on_n ≥ 2)·끈 뒤 이륙."""
    data = {c: (load(pfx, ref, c), load(pfx, arm, c)) for c in DIST}
    if any(a is None or b is None for a, b in data.values()):
        P(f"  보조 {arm} vs {ref}: 자료 부족"); return
    for c, (A, B) in data.items():
        if not rng_same(A, B):
            P(f"  보조 {arm} vs {ref}: {c} RNG 짝 불일치 → 무효"); return
    for nm, fn in (("끈 뒤 진동 osc_flag_post", lambda r: r.get("osc_flag_post") == "1"),
                   ("재켜짐 sw_on_n≥2", lambda r: (f(r.get("sw_on_n")) or 0) >= 2),
                   ("끈 뒤 이륙 to_after_off≥1", lambda r: (f(r.get("to_after_off")) or 0) >= 1),
                   ("낙상", lambda r: r["fell"] == "1")):
        b = c_ = n = 0
        for c in DIST:
            A, B = data[c]
            for k in A:
                if A[k]["pre_pass"] != "1": continue
                if nm != "낙상" and (A[k]["fell"] == "1" or B[k]["fell"] == "1"): continue
                n += 1; x, y = fn(A[k]), fn(B[k]); b += x and not y; c_ += y and not x
        P(f"  보조 {nm}: {ref}만 {b} / {arm}만 {c_} (짝 {n}, p {mcnemar(b, c_):.3g})")
    dt = []
    for c in DIST:
        A, B = data[c]
        for k in A:
            if A[k]["pre_pass"] == "1" and A[k]["fell"] != "1" and B[k]["fell"] != "1":
                dt.append(f(B[k]["sw_time_s"]) - f(A[k]["sw_time_s"]))
    P(f"  보조 회복 모드 시간 차({arm} − {ref}) 평균 {np.mean(dt):+.3f} s · 차 ≠ 0 인 짝 {100 * np.mean(np.abs(dt) > 1e-6):.1f}%")


def main(out, arms=("odi", "odic")):
    L = []
    P = lambda x="": (L.append(x), print(x, flush=True))
    P(f"# STOPEST 사전 승격 기준 판정 (KE 종료 판정기) — 팔 {' '.join(arms)} vs imu" + (" · §11" if "odicw" in arms else " · §8/§8a"))
    R = {}
    for pfx, dl, strict in (("sab", 2, True), ("sabd3", 3, False)):
        P(f"\n## 지연 {dl} ({'주 판정' if strict else '강건성 확인 — P1 방향만'})")
        for arm in arms:
            P(f"### {arm}")
            R[(arm, dl)] = judge(pfx, arm, strict, P)
            if arm == "odicw":
                secondary(pfx, arm, "odic", P)
    P("\n## 결론")
    for arm in arms:
        v = (R.get((arm, 2)), R.get((arm, 3)))
        P(f"- {arm}: 지연 2 {v[0]} · 지연 3 {v[1]} → {'승격 후보' if all(x is True for x in v) else '미승격'}")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", tuple(sys.argv[2:]) or ("odi", "odic"))
