"""STOPEST §16 — imuH 독립 seed 판정: settle_promote.judge(P1–P5, seed 집합 덮어쓰기) + P6(해제 뒤 1 s 안 참 |ω_xy| ≥ 1.0 또는 재켜짐) + 보고용(dwell·정상 자세 복귀).
사용: python3 reports/gen_w3/settle_promote_x.py [out.txt] [seeds=7,8,9,10,11,12] [arm=imuH]
"""
import csv, glob, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settle_promote as sp  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
f = sp.f


def p6_env_flags(pfx, arm, c, s, win=50, w_thr=1.0):
    p = os.path.join(REPO, "log_picture", f"ground_recovery_{pfx}_{arm}_{c}_s{s}_trace.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p); mode = z["mode"]; w = z["w_true"].astype(np.float32); pp = z["pre_pass"]
    T, N = mode.shape; flag = np.zeros(N, dtype=bool); n_off = np.zeros(N, dtype=int)
    for t in range(1, T):
        off = mode[t - 1] & ~mode[t]
        if not off.any():
            continue
        e = np.nonzero(off)[0]; n_off[e] += 1
        t1 = min(T, t + win)
        reon = mode[t + 1:t1].any(0) if t + 1 < t1 else np.zeros(N, dtype=bool)
        big = (w[t:t1] >= w_thr).any(0)
        flag[e] |= (reon | big)[e]
    return flag, n_off, pp


def dwell_stats(pfx, arm, c, seeds):
    tv = []; nr = []
    for s in seeds:
        p = sp.LPD.format(pfx, arm, c, s)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r["pre_pass"] != "1" or r["fell"] == "1":
                continue
            x = f(r.get("sw_time_s"));  y = f(r.get("nrm_t"))
            if x is not None: tv.append(x)
            if y is not None: nr.append(y)
    q = lambda v, p: (float(np.percentile(v, p)) if v else float("nan"))
    return q(tv, 50), q(tv, 90), q(nr, 50), q(nr, 90), len(tv), len(nr)


def judge_v2(pfx, arm, strict_p1, P):
    """STOPEST §17 v2: P1·P3·P4 는 v1 과 같은 식, P2′(짝 비열등 + 조건별 −4.5 %p)·P5′(짝 낙상 ≤ +1·켜짐 ≤ +0.02·엄격 ≥ −1 %p)."""
    data = {c: (sp.load(pfx, "imu", c), sp.load(pfx, arm, c)) for c in ("nd",) + sp.DIST}
    if any(a is None or b is None for a, b in data.values()):
        P(f"  {arm}: 자료 부족 — 판정 보류"); return None
    for c, (A, B) in data.items():
        if not sp.rng_same(A, B):
            P(f"  {arm}: {c} RNG 짝 불일치 → 판정 무효"); return None
    keys = {c: [k for k in data[c][0] if data[c][0][k]["pre_pass"] == "1"] for c in data}
    b = cc = 0
    for c in sp.DIST:
        A, B = data[c]
        b += sum(1 for k in keys[c] if A[k]["fell"] == "1" and B[k]["fell"] != "1"); cc += sum(1 for k in keys[c] if B[k]["fell"] == "1" and A[k]["fell"] != "1")
    p1p = sp.mcnemar(cc, b); p1 = (cc < b and p1p < 0.05) if strict_p1 else (cc <= b)
    P(f"  P1 낙상 짝: imu만 {b} / {arm}만 {cc} · p {p1p:.3g} → {'PASS' if p1 else 'FAIL'}")
    p2 = True; parts = []
    for c in sp.DIST:
        A, B = data[c]; ka = keys[c]
        ra = np.mean([sp.classify(A[k]) == "OK" for k in ka]); rb = np.mean([sp.classify(B[k]) == "OK" for k in ka])
        parts.append(f"{c} {100*ra:.1f}→{100*rb:.1f}"); p2 &= (rb - ra) >= -0.045
    bo = sum(1 for c in sp.DIST for k in keys[c] if sp.classify(data[c][0][k]) == "OK" and sp.classify(data[c][1][k]) != "OK")
    co = sum(1 for c in sp.DIST for k in keys[c] if sp.classify(data[c][1][k]) == "OK" and sp.classify(data[c][0][k]) != "OK")
    p2 &= not (co < bo and sp.mcnemar(bo, co) < 0.05)
    P(f"  P2′ 엄격 성공: {' · '.join(parts)} · 짝 imu만 {bo} / {arm}만 {co} (p {sp.mcnemar(bo, co):.3g}) → {'PASS' if p2 else 'FAIL'}")
    p3 = True; parts = []
    for col, lim in (("abs_pitch", 0.5), ("dq_nom_deg", 0.5), ("abn_frac", 0.02)):
        d = []
        for c in sp.DIST:
            A, B = data[c]
            for k in keys[c]:
                if A[k]["fell"] != "1" and B[k]["fell"] != "1":
                    x, y = f(A[k].get(col)), f(B[k].get(col))
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        d.append(y - x)
        up = np.mean(d) + 1.96 * np.std(d) / math.sqrt(len(d)); parts.append(f"{col} 상한 {up:+.3f}(≤{lim})"); p3 &= up <= lim
    P(f"  P3 끝 자세: {' · '.join(parts)} → {'PASS' if p3 else 'FAIL'}")
    la = []; lb = []; dt = []
    for c in sp.DIST:
        A, B = data[c]
        for k in keys[c]:
            if A[k]["fell"] != "1": la.append(f(A[k]["sw_time_s"]) > 4.0)
            if B[k]["fell"] != "1": lb.append(f(B[k]["sw_time_s"]) > 4.0)
            if A[k]["fell"] != "1" and B[k]["fell"] != "1": dt.append(f(B[k]["sw_time_s"]) - f(A[k]["sw_time_s"]))
    p4 = (np.mean(lb) <= np.mean(la) + 0.03) and (np.mean(dt) <= 1.0)
    P(f"  P4 장기 체류(>4 s): imu {100*np.mean(la):.1f}% · {arm} {100*np.mean(lb):.1f}% · 회복 시간 차 {np.mean(dt):+.2f} s → {'PASS' if p4 else 'FAIL'}")
    A, B = data["nd"]; kn = keys["nd"]
    fa_only = sum(1 for k in kn if A[k]["fell"] == "1" and B[k]["fell"] != "1"); fb_only = sum(1 for k in kn if B[k]["fell"] == "1" and A[k]["fell"] != "1")
    sa = np.mean([f(A[k]["sw_on_n"]) for k in kn]); sb = np.mean([f(B[k]["sw_on_n"]) for k in kn])
    oa = np.mean([sp.classify(A[k]) == "OK" for k in kn]); ob = np.mean([sp.classify(B[k]) == "OK" for k in kn])
    p5 = (fb_only <= fa_only + 1) and (sb <= sa + 0.02) and (ob >= oa - 0.01)
    P(f"  P5′ 무외란: 짝 낙상 imu만 {fa_only} / {arm}만 {fb_only} · 켜짐 {sa:.3f}→{sb:.3f} · 엄격 {100*oa:.1f}→{100*ob:.1f} → {'PASS' if p5 else 'FAIL'}")
    allp = bool(p1 and p2 and p3 and p4 and p5); P(f"  ⇒ {arm}(v2 P1–P5′): {'통과' if allp else '불통과'}")
    return allp


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        out = sys.argv[2] if len(sys.argv) > 2 else ""; seeds = tuple(int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "13,14,15,16,17,18").split(","))
        arm = sys.argv[4] if len(sys.argv) > 4 else "imuH"; sp.SEEDS = seeds
        L = []; P = lambda x="": (L.append(x), print(x, flush=True))
        P(f"# STOPEST §17 imuH 확증 v2 — seed {seeds} · 팔 {arm} vs imu"); R = {}
        for pfx, dl, strict in (("sab", 2, True), ("sabd3", 3, False)):
            P(f"\n## 지연 {dl}"); R[(dl, "p15")] = judge_v2(pfx, arm, strict, P)
            bi = ci = n = 0; fa = []; fb = []
            for c in sp.DIST:
                for s_ in seeds:
                    A = p6_env_flags(pfx, "imu", c, s_); B = p6_env_flags(pfx, arm, c, s_)
                    if A is None or B is None:
                        continue
                    m = A[2] & B[2]; fa += list(A[0][m]); fb += list(B[0][m]); n += int(m.sum())
                    bi += int((A[0] & ~B[0] & m).sum()); ci += int((B[0] & ~A[0] & m).sum())
            if n:
                p6 = bool(np.mean(fb) <= np.mean(fa) + 0.02); R[(dl, "p6")] = p6
                P(f"  P6 해제 뒤 1 s 재흔들림·재켜짐: imu {100*np.mean(fa):.1f}% · {arm} {100*np.mean(fb):.1f}% (짝 {bi}/{ci}, n {n}) → {'PASS' if p6 else 'FAIL'}")
            else:
                R[(dl, "p6")] = None; P("  P6: 추적 없음")
        ok = all(v is True for v in R.values())
        P(f"\n## 결론\n- {arm} v2: {R} → {'전부 통과 — 승격 제안(사용자 결정)' if ok else '불통과 — 권장 최종 철회'}")
        if out:
            open(out, "w").write("\n".join(L) + "\n")
        return
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    seeds = tuple(int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "7,8,9,10,11,12").split(","))
    arm = sys.argv[3] if len(sys.argv) > 3 else "imuH"
    sp.SEEDS = seeds
    L = []; P = lambda x="": (L.append(x), print(x, flush=True))
    P(f"# STOPEST §16 imuH 독립 seed 판정 — seed {seeds} · 팔 {arm} vs imu")
    R = {}
    for pfx, dl, strict in (("sab", 2, True), ("sabd3", 3, False)):
        P(f"\n## 지연 {dl} ({'P1 엄격' if strict else 'P1 방향'})")
        R[(dl, "p15")] = sp.judge(pfx, arm, strict, P)
        # P6
        bi = ci = n = 0; fa = []; fb = []; miss = 0
        for c in sp.DIST:
            for s in seeds:
                A = p6_env_flags(pfx, "imu", c, s); B = p6_env_flags(pfx, arm, c, s)
                if A is None or B is None:
                    miss += 1; continue
                m = A[2] & B[2]
                fa += list(A[0][m]); fb += list(B[0][m]); n += int(m.sum())
                bi += int((A[0] & ~B[0] & m).sum()); ci += int((B[0] & ~A[0] & m).sum())
        if n:
            p6 = (np.mean(fb) <= np.mean(fa) + 0.02)
            P(f"  P6 해제 뒤 1 s 안 |ω|≥1.0 또는 재켜짐(env 비율): imu {100*np.mean(fa):.1f}% · {arm} {100*np.mean(fb):.1f}% · 짝 imu만 {bi} / {arm}만 {ci} (n {n}, p {sp.mcnemar(bi, ci):.3g}){' · 누락 '+str(miss) if miss else ''} → {'PASS' if p6 else 'FAIL'}")
            R[(dl, "p6")] = bool(p6)
        else:
            P("  P6: 추적 파일 없음"); R[(dl, "p6")] = None
        for c in ("nd",) + sp.DIST:
            a = dwell_stats(pfx, "imu", c, seeds); b = dwell_stats(pfx, arm, c, seeds)
            P(f"  보고 {c}: 회복 모드 시간 중앙/p90 imu {a[0]:.2f}/{a[1]:.2f} s · {arm} {b[0]:.2f}/{b[1]:.2f} s | 정상 자세 복귀 nrm_t 중앙/p90 imu {a[2]:.2f}/{a[3]:.2f} · {arm} {b[2]:.2f}/{b[3]:.2f} (n {a[4]}/{b[4]})")
    P("\n## 결론")
    ok = all(v is True for v in R.values())
    P(f"- {arm}: 지연 2 P1–P5 {R.get((2,'p15'))} P6 {R.get((2,'p6'))} · 지연 3 P1–P5 {R.get((3,'p15'))} P6 {R.get((3,'p6'))} → {'통과(승격 제안)' if ok else '불통과 또는 보류'}")
    if out:
        open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
