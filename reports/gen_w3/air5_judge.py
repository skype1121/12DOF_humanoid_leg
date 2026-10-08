#!/usr/bin/env python3
"""AIR5 판정(reports/gen_w3/AIR_START_1008.md §15) — 앞 절반은 air3_judge 와 같은 계산(Q1–Q3, 대조는 AIR3 seed 3), 뒤 절반은 기운 매달림(Q5)·차단 발동(Q6).\n원문: AIR3 판정(§13): 정정한 지지 모형·브리지 등록 합성 키에서 3 단 러너(AIR → KE ⇄ W3)의 공중 정책 구동. 기준은 AIR2(§10)의 Q1–Q3 그대로.
대조는 매달림 시간을 맞춘 홀드 착지(같은 접두).

사용: air5_judge.py [태그 접두=tr_air5] [명목 팔 seed=3] [T 팔 seed=3,5]
낙상은 에피소드 처음(매달림)부터 센다. 공중 거동은 추적한 앞 128 env(인계 ~ 첫 발 접촉)."""
import csv, json, math, os, sys
import numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
PF = sys.argv[1] if len(sys.argv) > 1 else "tr_air5"
SEEDS = tuple(int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("3",)))
#: 팔 → (지지 끝[인계 기준 s], 걷기 버튼 시각, 대조 팔, 설명, seed 제한)
ARMS = {"A2": (6.0, 39.0, "C", "힘 모형 · 공중 2 s · 방출 4 s", None), "A2r1": (3.0, 36.0, "Cr1", "힘 모형 · 공중 2 s · 방출 1 s", None),
        "L2": (8.0, 41.0, "CL6", "줄 길이 · 공중 2 s · 하강 6 s", None), "L20": (26.0, 59.0, "CL20", "줄 길이 · 공중 20 s · 하강 6 s", None),
        "L2s": (17.0, 50.0, "CLs", "줄 길이 · 공중 2 s · 하강 15 s(느림)", None)}
#: 대조 = AIR3 의 같은 seed(3) 대조 팔(홀드로 착지 뒤 지면에서 인계 — 공중 러너를 쓰지 않으므로 공중 정책이 바뀌어도 같은 실행)
CTRL = {"C": ("tr_air3", -1.5 + 4.0, 35.5, "힘 모형 · 홀드로 착지 뒤 인계(방출 4 s)", (3,)), "Cr1": ("tr_air3", -0.5, 33.0, "힘 모형 · 홀드 착지 뒤 인계(방출 1 s)", (3,)),
        "CL6": ("tr_air3", -1.5, 33.0, "줄 길이 · 홀드 3 s 매달림 · 하강 6 s", (3,)), "CL20": ("tr_air3", -1.5, 33.0, "줄 길이 · 홀드 23 s 매달림 · 하강 6 s", (3,)),
        "CLs": ("tr_air3", -1.5, 33.0, "줄 길이 · 홀드 착지 뒤 인계(하강 15 s)", (3,))}
LAG = 6      # 접촉 관측(feet_contact_real_v2)의 감지 지연 상한 [틱]: 10 Hz 유지(5 틱) + 전송 1 틱
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
q90 = lambda v: float(np.percentile(np.asarray([x for x in v if x == x], float), 90)) if len([x for x in v if x == x]) else float("nan")
pc = lambda a, n: f"{a} / {n} ({100*a/n:.1f} %)" if n else "—"
def load(prefix, arm, zero, btn, seeds):
    R = dict(n=0, f_all30=0, alive_zero=0, f_zero5=0, alive_btn=0, acc=0, miss=[], air=[], exc=[], spd=[], spd_air=[], wmax=[], air_n=0, ho=0, ho_n=0, ho_dt=[], rec_air=0)
    for s in seeds:
        tag = f"{prefix}_{arm}_s{s}"; f = f"{LP}/ground_recovery_{tag}_envs.csv"; k = f"{LP}/kwloop_{tag}.npz"; t = f"{LP}/transall_{tag}.npz"
        if not (os.path.exists(f) and os.path.exists(k) and os.path.exists(t)):
            R["miss"].append(s); continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        tf = np.array([float(r["t_fall_rel"]) if (r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan")) else np.inf for r in rows]); n = len(rows); R["n"] += n
        R["f_all30"] += int((tf <= zero + 30.0).sum())
        az = tf > zero; R["alive_zero"] += int(az.sum()); R["f_zero5"] += int((az & (tf <= zero + 5.0)).sum())
        ev = json.loads(str(np.load(k, allow_pickle=True)["events"])); ab = tf > btn
        R["alive_btn"] += int(ab.sum()); R["acc"] += sum(1 for e in range(n) if ab[e] and any(x[1] == "enable" and x[0] >= btn - 0.05 for x in ev[e]))
        tr = np.load(t, allow_pickle=True); tt = tr["t"].astype(float); tt = tt - tt[0]; cont = tr["cont"] > 0.5; mode = tr["mode"].astype(bool); q = tr["q"]; w = tr["w"]; alive = tr["alive"].astype(bool)
        for e in range(q.shape[1]):
            anyc = cont[:, e].any(1); k1 = int(np.argmax(anyc)) if anyc.any() else q.shape[0]; k1 = min(k1, int(alive[:, e].sum()))
            if k1 >= 10:
                R["air_n"] += 1; R["air"].append(k1 * 0.02); R["rec_air"] += bool(mode[:k1, e].any())
                R["exc"].append(float(np.degrees(np.ptp(q[:k1, e], axis=0).max()))); R["spd"].append(float(np.abs(np.diff(q[:k1, e], axis=0)).max() * 50.0)); R["wmax"].append(float(w[:k1, e].max()))
                # §13 수정 1: 접촉 관측 모형의 지연 상한(10 Hz 유지 5 틱 + 전송 1 틱 = LAG 틱)만큼 앞에서 끊은 공중 구간의 관절 속도 — 감지 직전 창은 실제로는 착지 충격이다
                R["spd_air"].append(float(np.abs(np.diff(q[:max(k1 - LAG, 2), e], axis=0)).max() * 50.0))
            both = cont[:, e].all(1)
            if both.any():                                    # 양발 첫 접촉 뒤 10 s 까지 살아 있던 개체: 그 안에 KE 로 넘어갔는가
                tb = float(tt[int(np.argmax(both))]); th = [x[0] for x in ev[e] if x[1] == "air_to_ke"]
                if tf[e] > tb + 10.0:
                    R["ho_n"] += 1; R["ho"] += bool(th and th[0] <= tb + 10.0)
                    if th: R["ho_dt"].append(th[0] - tb)
    return R
C = {}
for cn, (pf, zero, btn, lab, sd) in CTRL.items():
    C[cn] = load(pf or PF, cn.replace("Cr1", "Cr1"), zero, btn, sd or SEEDS)
D = {a: load(PF, a, v[0], v[1], v[4] or SEEDS) for a, v in ARMS.items()}
print(f"## AIR5 — G21h2 · 정정 지지 모형 · 3 단 러너(AIR → KE ⇄ W3) 공중 정책 구동 ({PF}, seed {','.join(map(str, SEEDS))}, 낙상 분모 = 전체 env)")
print("| 팔 | env | 처음부터 지지 끝 뒤 30 s 까지 낙상 | 대조 | 대조 대비 · Fisher p | 지지 끝 뒤 5 s 낙상 | 걷기 버튼 수락 |"); print("|---|---|---|---|---|---|---|")
for cn, r in C.items():
    if r["n"]: print(f"| (대조) {cn} {CTRL[cn][3]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_all30'], r['n'])} | — | — | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
    else: print(f"| (대조) {cn} {CTRL[cn][3]} | (없음 {r['miss']}) | | | | | |")
for a, r in D.items():
    c = C[ARMS[a][2]]
    if r["n"] == 0: print(f"| {a} {ARMS[a][3]} | (없음 {r['miss']}) | | | | | |"); continue
    d = "—" if c["n"] == 0 else f"{100*r['f_all30']/r['n'] - 100*c['f_all30']/c['n']:+.1f} %p · {fisher(r['f_all30'], r['n'] - r['f_all30'], c['f_all30'], c['n'] - c['f_all30']):.3g}"
    print(f"| {a} {ARMS[a][3]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_all30'], r['n'])} | {ARMS[a][2]} {pc(c['f_all30'], c['n'])} | {d} | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
print(); print("| 팔 | 공중 구간(추적 env) | 공중 시간 중앙 | 공중 관절 이동폭 p90 | 공중 관절 속도 최대 p90 (감지 6 틱 전까지) | 같은 값 · 등록 창(감지 직전까지 = 착지 충격 포함) | 공중 몸통 각속도 최대 p90 | 양발 첫 접촉 뒤 10 s 안에 KE 인계 | 접촉 → 인계 시간 중앙·p90 |"); print("|---|---|---|---|---|---|---|---|---|")
for a, r in D.items():
    if r["air_n"] == 0: print(f"| {a} | 0 | — | — | — | — | — | — | — |"); continue
    hd = f"{np.median(r['ho_dt']):.1f} · {np.percentile(r['ho_dt'], 90):.1f} s" if r["ho_dt"] else "—"
    print(f"| {a} | {r['air_n']} | {np.median(r['air']):.1f} s | {q90(r['exc']):.1f}° | {q90(r['spd_air']):.2f} rad/s | {q90(r['spd']):.2f} rad/s{' ★' if q90(r['spd']) > 3.0 else ''} | {q90(r['wmax']):.2f} rad/s | {pc(r['ho'], r['ho_n'])} | {hd} |")
print(); allok = True
for a, r in D.items():
    c = C[ARMS[a][2]]
    if r["n"] == 0 or r["miss"] or c["n"] == 0 or c["miss"]: print(f"- {a}: 실행 미완(팔 {r['miss']} · 대조 {c['miss']})"); allok = False; continue
    fr = 100 * r["f_all30"] / r["n"]; fc = 100 * c["f_all30"] / c["n"]; p = fisher(r["f_all30"], r["n"] - r["f_all30"], c["f_all30"], c["n"] - c["f_all30"])
    q1 = fr <= fc + 3.0 and p >= 0.05
    q2 = r["air_n"] > 0 and q90(r["exc"]) <= 20.0 and q90(r["spd_air"]) <= 3.0          # §13 수정 1(속도 창). 이동폭은 등록 창 그대로
    q2reg = r["air_n"] > 0 and q90(r["exc"]) <= 20.0 and q90(r["spd"]) <= 3.0
    z5 = 100 * r["f_zero5"] / max(r["alive_zero"], 1); cz5 = 100 * c["f_zero5"] / max(c["alive_zero"], 1); acc = 100 * r["acc"] / max(r["alive_btn"], 1); ho = 100 * r["ho"] / max(r["ho_n"], 1)
    q3 = ho >= 99.0 and z5 <= cz5 + 3.0 and acc >= 95.0
    allok = allok and q1 and q2 and q3
    print(f"- {a}: Q1 {'충족' if q1 else '미달'}(낙상 {fr:.1f} % 대 대조 {fc:.1f} %, p {p:.3g}) · Q2 {'충족' if q2 else '미달'}(이동폭 {q90(r['exc']):.1f}° · 속도 {q90(r['spd_air']):.2f}{'' if q2reg == q2 else ' — 등록 창으로는 ' + format(q90(r['spd']), '.2f') + ' 미달'}) · "
          f"Q3 {'충족' if q3 else '미달'}(인계 {ho:.1f} % · 지지 끝 뒤 5 s {z5:.1f} % 대 {cz5:.1f} % · 수락 {acc:.1f} %)")
print(f"- **공중 팔 전체: {'Q1–Q3 충족' if allok else '미달 또는 미완'}**")

# ---------------- Q5 기운 매달림(T 팔) · Q6 차단 발동 ----------------
TSEEDS = tuple(int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("3", "5")))
EDGES = [-99, -2, 0, 1, 2.5, 4, 6, 99]; BINS = [(-2, 0), (0, 1), (1, 2.5), (2.5, 4), (4, 6)]
T = []; tmiss = []
for sd in TSEEDS:
    tag = f"{PF}_T_s{sd}"; f = f"{LP}/ground_recovery_{tag}_envs.csv"
    if not all(os.path.exists(x) for x in (f, f"{LP}/kwloop_{tag}.npz", f"{LP}/transall_{tag}.npz", f"{LP}/kwair_{tag}.npz")):
        tmiss.append(sd); continue
    rows = list(csv.DictReader(open(f, encoding="utf-8"))); tf = np.array([float(r["t_fall_rel"]) if (r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan")) else np.inf for r in rows])
    tr = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True); ka = np.load(f"{LP}/kwair_{tag}.npz", allow_pickle=True); ev = json.loads(str(np.load(f"{LP}/kwloop_{tag}.npz", allow_pickle=True)["events"]))
    q = np.degrees(tr["q"]); cont = tr["cont"] > 0.5; alive = tr["alive"].astype(bool); pitch = tr["pitch"]; n = len(rows)
    assert q.shape[1] == n, f"{tag}: 전 env 추적이 아님({q.shape[1]} / {n})"
    for e in range(n):
        anyc = cont[:, e].any(1); k1 = int(np.argmax(anyc)) if anyc.any() else q.shape[0]; k1 = min(k1, int(alive[:, e].sum()))
        air = k1 >= 60
        T.append(dict(p=float(np.median(pitch[25:min(k1, 400), e])) if air else float("nan"), exc=float(np.ptp(q[:k1, e], axis=0).max()) if air else float("nan"), air=air,
                      fell=bool(tf[e] <= 14.0 + 30.0), alive_btn=bool(tf[e] > 47.0), ho=any(x[1] == "air_to_ke" for x in ev[e]), acc=any(x[1] == "enable" and x[0] >= 46.95 for x in ev[e]),
                      hold=int(ka["n_hold"][e]) > 0, dev=float(ka["dev"][e]) if "dev" in ka.files else float("nan"), amax=float(ka["amax"][e])))
print(); print(f"### Q5 기운 매달림 — T 팔(매달림 목표 피치 −6 ~ +6°, seed {','.join(map(str, TSEEDS))}{' · 미완 ' + str(tmiss) if tmiss else ''})")
ok5 = bool(T) and not tmiss
if T:
    n = len(T); nf = sum(r["fell"] for r in T); st = [r for r in T if not r["fell"]]; nho = sum(r["ho"] for r in st); ab = [r for r in T if r["alive_btn"]]; nacc = sum(r["acc"] for r in ab)
    print("| 매달림 피치(공중 중앙) | env | 공중 관절 이동폭 p50 · p90 · 최대 | 매달린 동안 목표 이탈 최대 p90 · 최대 | 낙상 | 넘어지지 않은 개체의 KE 인계 | 차단 발동 |"); print("|---|---|---|---|---|---|---|")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        g = [r for r in T if r["air"] and lo <= r["p"] < hi]
        if not g: continue
        ex = np.array([r["exc"] for r in g]); dv = np.array([r["dev"] for r in g]); gs = [r for r in g if not r["fell"]]; h = sum(r["ho"] for r in gs)
        lab = f"{lo:+g} ~ {hi:+g}°".replace("-99 ~ ", "< ").replace(" ~ +99°", "° 이상")
        print(f"| {lab} | {len(g)} | {np.percentile(ex, 50):.0f} · {np.percentile(ex, 90):.0f} · {ex.max():.0f}° | {np.nanpercentile(dv, 90):.0f} · {np.nanmax(dv):.0f}° | {sum(r['fell'] for r in g)} | {pc(h, len(gs))} | {sum(r['hold'] for r in g)} |")
        if (lo, hi) in BINS and len(g) >= 20:
            ok5 = ok5 and np.percentile(ex, 90) <= 20.0 and (100 * h / max(len(gs), 1)) >= 95.0
    P = np.array([r["p"] for r in T if r["air"]])
    c1 = 100 * nf / n <= 3.0; c2 = 100 * nho / max(len(st), 1) >= 97.0; c3 = 100 * nacc / max(len(ab), 1) >= 95.0; ok5 = ok5 and c1 and c2 and c3
    print(f"- 전체 {n} env · 매달림 피치 {np.nanmin(P):+.1f} ~ {np.nanmax(P):+.1f}° · 낙상 {pc(nf, n)}({'충족' if c1 else '미달'} ≤ 3 %) · 넘어지지 않은 개체의 KE 인계 {pc(nho, len(st))}({'충족' if c2 else '미달'} ≥ 97 %) · 걷기 버튼 수락 {pc(nacc, len(ab))}({'충족' if c3 else '미달'} ≥ 95 %)")
print(f"- **Q5: {'충족' if ok5 else '미달 또는 미완'}**")
print(); print("### Q6 공중 폭주 차단 발동(≤ 1 %)")
ok6 = True; tot = [0, 0]
for arm, seeds in [(a_, SEEDS) for a_ in ARMS] + [("T", TSEEDS)]:
    for sd in seeds:
        f = f"{LP}/kwair_{PF}_{arm}_s{sd}.npz"
        if not os.path.exists(f): print(f"- {arm} s{sd}: 없음"); ok6 = False; continue
        ka = np.load(f, allow_pickle=True); nh = int((ka["n_hold"] > 0).sum()); n = len(ka["n_hold"]); tot[0] += nh; tot[1] += n
        dv = ka["dev"] if "dev" in ka.files else np.full(n, np.nan)
        print(f"- {arm} s{sd}: 발동 {pc(nh, n)} · 매달린 동안 목표 이탈 최대의 p50 {np.nanpercentile(dv, 50):.1f}° · p99 {np.nanpercentile(dv, 99):.1f}° · 최대 {np.nanmax(dv):.1f}°(문턱 30°)"); ok6 = ok6 and 100 * nh / n <= 1.0
print(f"- **Q6: {'충족' if ok6 else '미달 또는 미완'}** (합 {pc(tot[0], tot[1]) if tot[1] else '—'})")
print(); print(f"## AIR5 종합: Q1–Q3 {'충족' if allok else '미달 또는 미완'} · Q5 {'충족' if ok5 else '미달 또는 미완'} · Q6 {'충족' if ok6 else '미달 또는 미완'} (Q7 은 selftest·러너 동치로 따로)")
