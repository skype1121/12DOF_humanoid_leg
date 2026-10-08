#!/usr/bin/env python3
"""AIR1 판정(reports/gen_w3/AIR_START_1008.md §1): 합성 키를 공중에서 구동한 채 착지시키는 경로 대 홀드로 착지 뒤 인계(대조).

낙상은 에피소드 처음(매달림)부터 센다 — 절차 전체의 실패율. 공중 거동은 추적한 앞 128 env(인계 ~ 첫 발 접촉)로 잰다."""
import csv, json, math, os, sys
import numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
#: 팔 → (인계 기준 방출 시작, 지지 0, 걷기 버튼 시각) [s]
ARMS = {"C": (-1.5, 2.5, 35.5), "A2": (2.0, 6.0, 39.0), "A2i": (2.0, 6.0, 39.0), "A6": (6.0, 10.0, 43.0), "A6i": (6.0, 10.0, 43.0)}
#: AIR1b 민감도 팔(방출 1·2 s, seed 1) — `air1_judge.py b` 로 본다
ARMS_B = {"C": (-1.5, 2.5, 35.5), "Cr1": (-1.5, -0.5, 33.0), "A2i": (2.0, 6.0, 39.0), "A2ir2": (2.0, 4.0, 37.0), "A2ir1": (2.0, 3.0, 36.0), "A2": (2.0, 6.0, 39.0), "A2r1": (2.0, 3.0, 36.0)}
LABEL_B = {"Cr1": "대조 · 방출 1 s", "A2ir2": "공중 2 s · 감지기 정지 · 방출 2 s", "A2ir1": "공중 2 s · 감지기 정지 · 방출 1 s", "A2r1": "공중 2 s · 감지기 그대로 · 방출 1 s"}
LABEL = {"C": "대조: 홀드로 착지 뒤 인계", "A2": "공중 2 s · 감지기 그대로", "A2i": "공중 2 s · 접지 전 감지기 정지", "A6": "공중 6 s · 감지기 그대로", "A6i": "공중 6 s · 접지 전 감지기 정지"}
#: AIR1c 진단 팔(접지 확인 전 접촉 관측 (1, 1) 고정) — `air1_judge.py c`
ARMS_C = {"C": (-1.5, 2.5, 35.5), "A2i": (2.0, 6.0, 39.0), "A2ic": (2.0, 6.0, 39.0), "A6i": (6.0, 10.0, 43.0), "A6ic": (6.0, 10.0, 43.0)}
MODE_C = len(sys.argv) > 1 and sys.argv[1] == "c"
MODE_B = len(sys.argv) > 1 and sys.argv[1] == "b"
SEEDS = (1,) if MODE_B else (1, 2) if MODE_C else tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("1", "2")))
def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a); return min(1.0, sum(p(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))
def q90(v):
    v = np.asarray([x for x in v if x == x], float)
    return float(np.percentile(v, 90)) if len(v) else float("nan")
def load(arm):
    rel0, zero, btn = ARMS[arm]; R = dict(n=0, f_pre=0, f_all30=0, alive_zero=0, f_zero5=0, alive_btn=0, acc=0, miss=[], air=[], exc=[], spd=[], air_on=0, air_n=0, td_rec=0, td_n=0, wmax=[], pre_pass=0)
    for s in SEEDS:
        tag = f"tr_air1_{arm}_s{s}"; f = f"{LP}/ground_recovery_{tag}_envs.csv"; k = f"{LP}/kwloop_{tag}.npz"; t = f"{LP}/transall_{tag}.npz"
        if not (os.path.exists(f) and os.path.exists(k) and os.path.exists(t)):
            R["miss"].append(s); continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        tf = np.array([float(r["t_fall_rel"]) if (r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan")) else np.inf for r in rows])
        N = len(rows); R["n"] += N; R["pre_pass"] += sum(r["pre_pass"] == "1" for r in rows)
        R["f_pre"] += int((tf < 0).sum()); R["f_all30"] += int((tf <= zero + 30.0).sum())
        az = tf > zero; R["alive_zero"] += int(az.sum()); R["f_zero5"] += int((az & (tf <= zero + 5.0)).sum())
        z = np.load(k, allow_pickle=True); ev = json.loads(str(z["events"]))
        ab = tf > btn
        R["alive_btn"] += int(ab.sum()); R["acc"] += sum(1 for e in range(N) if ab[e] and any(x[1] == "enable" and x[0] >= btn - 0.05 for x in ev[e]))
        tr = np.load(t, allow_pickle=True); tt = tr["t"].astype(float); tt = tt - tt[0]; cont = tr["cont"] > 0.5; mode = tr["mode"].astype(bool); q = tr["q"]; w = tr["w"]; alive = tr["alive"].astype(bool)
        for e in range(q.shape[1]):
            anyc = cont[:, e].any(1); k1 = int(np.argmax(anyc)) if anyc.any() else len(tt)      # 첫 발 접촉 틱(인계 뒤)
            k1 = min(k1, int(alive[:, e].sum()))
            if k1 >= 10:                                                                        # 공중 구간 0.2 s 이상
                R["air_n"] += 1; R["air"].append(k1 * 0.02); R["air_on"] += bool(mode[:k1, e].any())
                R["exc"].append(float(np.degrees(np.ptp(q[:k1, e], axis=0).max()))); R["spd"].append(float(np.abs(np.diff(q[:k1, e], axis=0)).max() * 50.0)); R["wmax"].append(float(w[:k1, e].max()))
            both = cont[:, e].all(1)
            if both.any() and alive[int(np.argmax(both)), e]:
                R["td_n"] += 1; R["td_rec"] += bool(mode[int(np.argmax(both)), e])
    return R
pc = lambda a, n: f"{a} / {n} ({100*a/n:.1f} %)" if n else "—"
if MODE_C:
    ARMS = ARMS_C; LABEL = {**LABEL, "A2ic": "공중 2 s · 감지기 정지 · 접촉 관측 (1, 1)", "A6ic": "공중 6 s · 감지기 정지 · 접촉 관측 (1, 1)"}
if MODE_B:
    ARMS = ARMS_B; LABEL = {**{k: v + " · 방출 4 s" for k, v in LABEL.items()}, **LABEL_B}
D = {a: load(a) for a in ARMS}
print(f"## AIR1 — 공중 정책 구동 대 홀드 착지 뒤 인계 (seed {','.join(map(str, SEEDS))}, 낙상 분모 = 전체 env)")
print("| 팔 | env | 인계 전 낙상 | 처음부터 지지 0 뒤 30 s 까지 낙상 | 대조 대비 · Fisher p | 지지 0 뒤 5 s 낙상(그때 서 있던 개체) | 걷기 버튼 수락(서 있던 개체) |"); print("|---|---|---|---|---|---|---|")
c = D["C"]
for a, r in D.items():
    if r["n"] == 0: print(f"| {a} {LABEL[a]} | (없음 {r['miss']}) | | | | | |"); continue
    d = "—" if a == "C" or c["n"] == 0 else f"{100*r['f_all30']/r['n'] - 100*c['f_all30']/c['n']:+.1f} %p · {fisher(r['f_all30'], r['n'] - r['f_all30'], c['f_all30'], c['n'] - c['f_all30']):.3g}"
    print(f"| {a} {LABEL[a]}{' (미완 ' + str(r['miss']) + ')' if r['miss'] else ''} | {r['n']} | {pc(r['f_pre'], r['n'])} | {pc(r['f_all30'], r['n'])} | {d} | {pc(r['f_zero5'], r['alive_zero'])} | {pc(r['acc'], r['alive_btn'])} |")
print(); print("| 팔 | 공중 구간(추적 env) | 공중 시간 중앙 | 공중에서 회복 모드 켜짐 | 공중 관절 이동폭 p90 | 공중 관절 속도 최대 p90 | 공중 몸통 각속도 최대 p90 | 양발 착지 순간 회복 모드 |"); print("|---|---|---|---|---|---|---|---|")
for a, r in D.items():
    if r["air_n"] == 0: print(f"| {a} | 0 | — | — | — | — | — | {pc(r['td_rec'], r['td_n'])} |"); continue
    print(f"| {a} | {r['air_n']} | {np.median(r['air']):.1f} s | {pc(r['air_on'], r['air_n'])} | {q90(r['exc']):.1f}° | {q90(r['spd']):.2f} rad/s | {q90(r['wmax']):.2f} rad/s | {pc(r['td_rec'], r['td_n'])} |")
print()
if c["n"] and not c["miss"] and not MODE_B and not MODE_C:      # P1–P3 판정 줄은 §1 의 네 팔에만(민감도·진단 모드는 표만)
    for a in ("A2", "A6", "A2i", "A6i"):
        r = D[a]
        if r["n"] == 0 or r["miss"]: print(f"- {a}: 실행 미완 {r['miss']}"); continue
        p1d = 100 * r["f_all30"] / r["n"] - 100 * c["f_all30"] / c["n"]; p1p = fisher(r["f_all30"], r["n"] - r["f_all30"], c["f_all30"], c["n"] - c["f_all30"])
        p1 = p1d <= 3.0 and p1p >= 0.05
        p2 = r["air_n"] > 0 and (100 * r["air_on"] / r["air_n"] <= 5.0) and q90(r["exc"]) <= 20.0 and q90(r["spd"]) <= 3.0
        z5 = 100 * r["f_zero5"] / max(r["alive_zero"], 1); cz5 = 100 * c["f_zero5"] / max(c["alive_zero"], 1); acc = 100 * r["acc"] / max(r["alive_btn"], 1)
        p3 = (z5 <= cz5 + 3.0) and acc >= 95.0
        print(f"- {a}: P1 {'충족' if p1 else '미달'}({p1d:+.1f} %p, p {p1p:.3g}) · P2 {'충족' if p2 else '미달'}(회복 모드 {100*r['air_on']/max(r['air_n'],1):.1f} % · 이동폭 {q90(r['exc']):.1f}° · 속도 {q90(r['spd']):.2f}) · P3 {'충족' if p3 else '미달'}(지지 0 뒤 5 s {z5:.1f} % 대 {cz5:.1f} % · 수락 {acc:.1f} %) → **{'P1–P3 충족' if (p1 and p2 and p3) else '미달'}**")
