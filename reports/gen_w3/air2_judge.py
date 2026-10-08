#!/usr/bin/env python3
"""AIR2 판정(reports/gen_w3/AIR_START_1008.md §10): 3 단 러너(AIR → KE ⇄ W3)로 공중에서 정책을 구동한 채 착지시키는 경로.

사용: air2_judge.py [태그 접두=tr_air2] [seed 목록=1,2]
낙상은 에피소드 처음(매달림)부터 센다. 공중 거동은 추적한 앞 128 env(인계 ~ 첫 발 접촉)."""
import csv, json, math, os, sys
import numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
PF = sys.argv[1] if len(sys.argv) > 1 else "tr_air2"
SEEDS = tuple(int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("1", "2")))
#: 팔 → (지지 끝[인계 기준 s], 걷기 버튼 시각, 대조 팔, 설명, seed 제한)
ARMS = {"A2": (6.0, 39.0, "C", "힘 모형 · 공중 2 s · 방출 4 s", None), "A6": (10.0, 43.0, "C", "힘 모형 · 공중 6 s · 방출 4 s", None), "A2r1": (3.0, 36.0, "Cr1", "힘 모형 · 공중 2 s · 방출 1 s", (1,)),
        "L2": (8.0, 41.0, "CL6", "줄 길이 · 공중 2 s · 하강 6 s", None), "L20": (26.0, 59.0, "CL6", "줄 길이 · 공중 20 s · 하강 6 s", None), "L60": (66.0, 99.0, "CL6", "줄 길이 · 공중 60 s · 하강 6 s", (1,)),
        "L2f": (4.0, 37.0, "CLf", "줄 길이 · 공중 2 s · 하강 2 s(빠름)", None), "L2s": (17.0, 50.0, "CLs", "줄 길이 · 공중 2 s · 하강 15 s(느림)", None)}
CTRL = {"C": ("tr_air1", -1.5 + 4.0, 35.5, "힘 모형 · 홀드로 착지 뒤 인계(AIR1 대조, 방출 4 s)", (1, 2)), "Cr1": ("tr_air1", -0.5, 33.0, "힘 모형 · 홀드 착지 뒤 인계(방출 1 s)", (1,)),
        "CL6": (None, -1.5, 33.0, "줄 길이 · 홀드로 착지 뒤 인계(하강 6 s)", None), "CLf": (None, -1.5, 33.0, "줄 길이 · 홀드 착지 뒤 인계(하강 2 s)", None), "CLs": (None, -1.5, 33.0, "줄 길이 · 홀드 착지 뒤 인계(하강 15 s)", None)}
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
q90 = lambda v: float(np.percentile(np.asarray([x for x in v if x == x], float), 90)) if len([x for x in v if x == x]) else float("nan")
pc = lambda a, n: f"{a} / {n} ({100*a/n:.1f} %)" if n else "—"
def load(prefix, arm, zero, btn, seeds):
    R = dict(n=0, f_all30=0, alive_zero=0, f_zero5=0, alive_btn=0, acc=0, miss=[], air=[], exc=[], spd=[], wmax=[], air_n=0, ho=0, ho_n=0, ho_dt=[], rec_air=0)
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
print(f"## AIR2 — 3 단 러너(AIR → KE ⇄ W3) 공중 정책 구동 ({PF}, seed {','.join(map(str, SEEDS))}, 낙상 분모 = 전체 env)")
print("| 팔 | env | 처음부터 지지 끝 뒤 30 s 까지 낙상 | 대조 | 대조 대비 · Fisher p | 지지 끝 뒤 5 s 낙상 | 걷기 버튼 수락 |"); print("|---|---|---|---|---|---|---|")
for cn, r in C.items():
    if r["n"]: print(f"| (대조) {cn} {CTRL[cn][3]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_all30'], r['n'])} | — | — | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
    else: print(f"| (대조) {cn} {CTRL[cn][3]} | (없음 {r['miss']}) | | | | | |")
for a, r in D.items():
    c = C[ARMS[a][2]]
    if r["n"] == 0: print(f"| {a} {ARMS[a][3]} | (없음 {r['miss']}) | | | | | |"); continue
    d = "—" if c["n"] == 0 else f"{100*r['f_all30']/r['n'] - 100*c['f_all30']/c['n']:+.1f} %p · {fisher(r['f_all30'], r['n'] - r['f_all30'], c['f_all30'], c['n'] - c['f_all30']):.3g}"
    print(f"| {a} {ARMS[a][3]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_all30'], r['n'])} | {ARMS[a][2]} {pc(c['f_all30'], c['n'])} | {d} | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
print(); print("| 팔 | 공중 구간(추적 env) | 공중 시간 중앙 | 공중 관절 이동폭 p90 | 공중 관절 속도 최대 p90 | 공중 몸통 각속도 최대 p90 | 양발 첫 접촉 뒤 10 s 안에 KE 인계 | 접촉 → 인계 시간 중앙·p90 |"); print("|---|---|---|---|---|---|---|---|")
for a, r in D.items():
    if r["air_n"] == 0: print(f"| {a} | 0 | — | — | — | — | — | — |"); continue
    hd = f"{np.median(r['ho_dt']):.1f} · {np.percentile(r['ho_dt'], 90):.1f} s" if r["ho_dt"] else "—"
    print(f"| {a} | {r['air_n']} | {np.median(r['air']):.1f} s | {q90(r['exc']):.1f}° | {q90(r['spd']):.2f} rad/s | {q90(r['wmax']):.2f} rad/s | {pc(r['ho'], r['ho_n'])} | {hd} |")
print(); allok = True
for a, r in D.items():
    c = C[ARMS[a][2]]
    if r["n"] == 0 or r["miss"] or c["n"] == 0 or c["miss"]: print(f"- {a}: 실행 미완(팔 {r['miss']} · 대조 {c['miss']})"); allok = False; continue
    fr = 100 * r["f_all30"] / r["n"]; fc = 100 * c["f_all30"] / c["n"]; p = fisher(r["f_all30"], r["n"] - r["f_all30"], c["f_all30"], c["n"] - c["f_all30"])
    q1 = fr <= fc + 3.0 and p >= 0.05
    q2 = r["air_n"] > 0 and q90(r["exc"]) <= 20.0 and q90(r["spd"]) <= 3.0
    z5 = 100 * r["f_zero5"] / max(r["alive_zero"], 1); cz5 = 100 * c["f_zero5"] / max(c["alive_zero"], 1); acc = 100 * r["acc"] / max(r["alive_btn"], 1); ho = 100 * r["ho"] / max(r["ho_n"], 1)
    q3 = ho >= 99.0 and z5 <= cz5 + 3.0 and acc >= 95.0
    allok = allok and q1 and q2 and q3
    print(f"- {a}: Q1 {'충족' if q1 else '미달'}(낙상 {fr:.1f} % 대 대조 {fc:.1f} %, p {p:.3g}) · Q2 {'충족' if q2 else '미달'}(이동폭 {q90(r['exc']):.1f}° · 속도 {q90(r['spd']):.2f}) · "
          f"Q3 {'충족' if q3 else '미달'}(인계 {ho:.1f} % · 지지 끝 뒤 5 s {z5:.1f} % 대 {cz5:.1f} % · 수락 {acc:.1f} %)")
print(f"- **공중 팔 전체: {'Q1–Q3 충족' if allok else '미달 또는 미완'}**")
