#!/usr/bin/env python3
"""골반 관절 고정 비교 요약(1009) — out/ 의 결과 파일만 읽는다. 사전 기준: 서기 = nd30 낙상 ≤ 2 % 그리고 C0 대비 +2 %p 이내 / 밀기 = 조건마다 C0 대비 +5 %p 이내."""
import csv, os, re, sys
H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, "out")
CONDS = ["C0", "R2", "RR", "RL", "A2", "AR", "AL"]; MEAS = ["nd30", "l18", "r18", "b18", "fi40", "sf15"]
LBL = {"C0": "없음(대조군)", "R2": "hip_r 좌+우", "RR": "hip_r 우", "RL": "hip_r 좌", "A2": "hip_a 좌+우", "AR": "hip_a 우", "AL": "hip_a 좌"}
MD = {"nd30": "무외란 30 s", "l18": "왼 18 N×0.3 s", "r18": "오른 18 N×0.3 s", "b18": "뒤 18 N×0.3 s", "fi40": "앞 40 N×0.2 s", "sf15": "앞 램프 15 N"}


def load(c, m):
    t = os.path.join(O, f"ground_recovery_lk_{c}_{m}.txt"); cv = os.path.join(O, f"ground_recovery_lk_{c}_{m}_envs.csv")
    if not (os.path.exists(t) and os.path.exists(cv)):
        return None
    txt = open(t, encoding="utf-8").read(); rows = list(csv.DictReader(open(cv)))
    d = dict(N=len(rows), pre=sum(r["pre_pass"] == "1" for r in rows))
    blk = txt.split("[사전 검사 통과] env")[1].split("[사전 검사 실패")[0]
    if m.startswith("nd30"):
        d["n"] = int(re.match(r"\s*(\d+):", blk).group(1)); d["f"] = int(re.search(r"인계 뒤 낙상 (\d+)", blk).group(1))
    else:
        g = re.search(r"외란 1 \[.*?\]: 시작 시 생존 (\d+) → 낙상 (\d+)", blk); d["n"] = int(g.group(1)); d["f"] = int(g.group(2))
        d["dir_ok"] = "일치" in re.search(r"\[방향 검증\].*", txt).group(0) and "불일치" not in re.search(r"\[방향 검증\].*", txt).group(0)
        d["dir"] = re.search(r"\[방향 검증\].*", txt).group(0)
    g = re.search(r"\[고정\].*?전 env ([\d.]+)°", txt); d["lock"] = float(g.group(1)) if g else None
    d["pp"] = {int(r["env"]) for r in rows if r["pre_pass"] == "1"}
    key = "fell" if m.startswith("nd30") else "d1_fell"
    d["fs"] = {int(r["env"]) for r in rows if r["pre_pass"] == "1" and r[key] == "1"}
    d["fall_all"] = sum(r["fell"] == "1" for r in rows)
    d["mt"] = os.path.getmtime(t)
    return d


D = {(c, m): load(c, m) for c in CONDS for m in MEAS}
pc = lambda f, n: f"{f}/{n} = {100*f/n:.1f} %" if n else "—"
L = []; P = L.append
P("| 조건 | 고정 | " + " | ".join(MD[m] for m in MEAS) + " | 고정 관절 최대 변위 [°] | 사전검사 통과/전체(nd30) |"); P("|---|---|" + "---|" * (len(MEAS) + 2))
V = {}
for c in CONDS:
    cells = []; locks = []
    for m in MEAS:
        d = D[(c, m)]; b = D[("C0", m)]
        if d is None: cells.append("미실행"); continue
        s = pc(d["f"], d["n"])
        if c != "C0" and b is not None:
            s += f" (Δ {100*d['f']/d['n'] - 100*b['f']/b['n']:+.1f} %p)"
        cells.append(s)
        if d["lock"] is not None: locks.append(d["lock"])
    d0 = D[(c, "nd30")]
    P(f"| {c} | {LBL[c]} | " + " | ".join(cells) + f" | {('%.3f' % max(locks)) if locks else '—'} | {(str(d0['pre']) + '/' + str(d0['N'])) if d0 else '—'} |")
    V[c] = max(locks) if locks else 0.0
P("")
P("| 조건 | 고정 유효(<0.3°) | 서기 가능(nd30 ≤ 2 % 그리고 C0 +2 %p 이내) | 밀기 내성 유지(조건마다 C0 +5 %p 이내) | 넘은 밀기 조건 |"); P("|---|---|---|---|---|")
for c in CONDS[1:]:
    if any(D[(c, m)] is None or D[("C0", m)] is None for m in MEAS):
        P(f"| {c} | — | 미완 | 미완 | |"); continue
    r = lambda m, cc=c: 100 * D[(cc, m)]["f"] / D[(cc, m)]["n"]
    st = (r("nd30") <= 2.0) and (r("nd30") - r("nd30", "C0") <= 2.0)
    bad = [f"{m} {r(m) - r(m, 'C0'):+.1f}" for m in MEAS[1:] if r(m) - r(m, "C0") > 5.0]
    P(f"| {c} | {'유효' if V[c] < 0.3 else '무효'} | {'합격' if st else '불합격'} | {'합격' if not bad else '불합격'} | {', '.join(bad)} |")
P("")
P("같은 env 짝 비교(두 조건 모두 사전검사 통과한 env 만; 낙상 = C0만 / 고정만):")
for c in CONDS[1:]:
    s = []
    for m in MEAS:
        d = D[(c, m)]; b = D[("C0", m)]
        if d is None or b is None: continue
        cm = d["pp"] & b["pp"]; s.append(f"{m} {len((b['fs'] - d['fs']) & cm)}/{len((d['fs'] - b['fs']) & cm)} (공통 {len(cm)})")
    P(f"- {c}: " + " · ".join(s))
P("")
SUPP = ["l30", "r30"]
DS = {(c, m): load(c, m) for c in CONDS for m in SUPP}
if any(v is not None for v in DS.values()):
    P("보조(사전 기준 판정에 넣지 않음) — 옆 밀기 30 N×0.2 s (걷기 평가기 W-dist 에서 쓰던 세기를 서기에 적용):")
    P("| 조건 | 왼 30 N×0.2 s | 오른 30 N×0.2 s | 고정 최대 변위 [°] |"); P("|---|---|---|---|")
    for c in CONDS:
        cells = []; lk = []
        for m in SUPP:
            d = DS[(c, m)]; b0 = DS[("C0", m)]
            if d is None: cells.append("미실행"); continue
            x = pc(d["f"], d["n"])
            if c != "C0" and b0: x += f" (Δ {100*d['f']/d['n'] - 100*b0['f']/b0['n']:+.1f} %p)"
            cells.append(x)
            if d["lock"] is not None: lk.append(d["lock"])
        P(f"| {c} | " + " | ".join(cells) + f" | {('%.3f' % max(lk)) if lk else '—'} |")
    P("")
    D.update(DS); MEAS = MEAS + SUPP
P("밀기 방향 검증(평가기 내장, 힘 창 동안 응답):")
for m in MEAS[1:]:
    for c in CONDS:
        d = D[(c, m)]
        if d: P(f"- {c} {m}: {'일치' if d['dir_ok'] else '불일치!'} — {d['dir'][:160]}")
print("\n".join(L))


# ── 1009 추가: hip_r 고정 + 모터 제거 질량 감소 · seed 흩어짐 ──
MEAS0 = ["nd30", "l18", "r18", "b18", "fi40", "sf15"]
def loadx(c, m, sd=1):
    t = m if sd == 1 else f"{m}_s{sd}"
    d = load(c, t)
    if d is None:
        d = load(c, t + "_b0")      # 고정 변위 0.3° 초과로 무효 → 한계 폭 0 으로 다시 돌린 것
        if d is not None: d["b0"] = True
    return d
def rate(d): return 100 * d["f"] / d["n"]
def massline(c, m):
    t = os.path.join(O, f"ground_recovery_lk_{c}_{m}.txt")
    if not os.path.exists(t): return ""
    g = re.search(r"전체 질량 env 평균 [\d.]+→([\d.]+) kg", open(t, encoding="utf-8").read()); return g.group(1) if g else ""
MC = [("R2pla", "R2"), ("R2m3", "R2"), ("R2m5", "R2"), ("RRm3", "RR"), ("RLm3", "RL")]
if any(loadx(c, "nd30") for c, _ in MC):
    print("\n### 질량 감소 조건 (seed 1 · 384 env) — 괄호: C0 대비 / 질량 변화 없는 같은 고정 대비 [%p]")
    print("| 조건 | " + " | ".join(MD[m] for m in MEAS0) + " | 고정 최대 변위 [°] | 전체 질량 [kg] |"); print("|---|" + "---|" * (len(MEAS0) + 2))
    for c in ["C0", "R2", "RR", "RL"]:
        print(f"| {c} | " + " | ".join(pc(D[(c, m)]["f"], D[(c, m)]["n"]) for m in MEAS0) + f" | {'—' if c == 'C0' else '%.3f' % max(D[(c, m)]['lock'] for m in MEAS0)} | 17.635 |")
    for c, ref in MC:
        cells = []; lk = []; ok = True
        for m in MEAS0:
            d = loadx(c, m)
            if d is None: cells.append("미실행"); ok = False; continue
            cells.append(f"{pc(d['f'], d['n'])} ({rate(d) - rate(D[('C0', m)]):+.1f} / {rate(d) - rate(D[(ref, m)]):+.1f})"); lk.append(d["lock"])
        print(f"| {c} | " + " | ".join(cells) + f" | {('%.3f' % max(lk)) if lk else '—'} | {massline(c, 'nd30')} |")
    print("\n| 조건 | 고정 유효 | 서기 가능 | 밀기 내성 유지 | C0 +5 %p 넘은 밀기 | 같은 고정(질량 불변) 대비 ±5 %p 넘게 달라진 칸 |"); print("|---|---|---|---|---|---|")
    for c, ref in MC:
        ds = {m: loadx(c, m) for m in MEAS0}
        if any(v is None for v in ds.values()): print(f"| {c} | 취소(무외란만 실행)" + (f" | {'합격' if ds['nd30'] and rate(ds['nd30']) <= 2.0 and rate(ds['nd30']) - rate(D[('C0', 'nd30')]) <= 2.0 else '불합격'} | 미실행 | | |" if ds["nd30"] else " | | | | |")); continue
        st = rate(ds["nd30"]) <= 2.0 and rate(ds["nd30"]) - rate(D[("C0", "nd30")]) <= 2.0
        bad = [f"{m} {rate(ds[m]) - rate(D[('C0', m)]):+.1f}" for m in MEAS0[1:] if rate(ds[m]) - rate(D[("C0", m)]) > 5.0]
        dif = [f"{m} {rate(ds[m]) - rate(D[(ref, m)]):+.1f}" for m in MEAS0 if abs(rate(ds[m]) - rate(D[(ref, m)])) > 5.0]
        print(f"| {c} | {'유효' if max(d['lock'] for d in ds.values()) < 0.3 else '무효'} | {'합격' if st else '불합격'} | {'합격' if not bad else '불합격'} | {', '.join(bad)} | {', '.join(dif)} |")
if loadx("C0", "nd30", 2):
    print("\n### seed 흩어짐 (C0·R2·R2pla, seed 1·2·3, 각 384 env) — 낙상 %: seed1 / seed2 / seed3 → 합")
    print("| 조건 | " + " | ".join(MD[m] for m in MEAS0) + " |"); print("|---|" + "---|" * len(MEAS0))
    POOL = {}
    for c in ("C0", "R2", "R2pla"):
        cells = []
        for m in MEAS0:
            ds = [loadx(c, m, sd) for sd in (1, 2, 3)]
            if any(d is None for d in ds): cells.append("미완"); continue
            f = sum(d["f"] for d in ds); n = sum(d["n"] for d in ds); POOL[(c, m)] = (f, n)
            cells.append(" / ".join(f"{rate(d):.1f}" + ("†" if d.get("b0") else "") for d in ds) + f" → {pc(f, n)}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    print("\n합(3 seed) 기준 C0 대비 차 [%p]:")
    for c in ("R2", "R2pla"):
        out = []
        for m in MEAS0:
            if (c, m) in POOL and ("C0", m) in POOL:
                a = POOL[(c, m)]; b = POOL[("C0", m)]; dv = 100 * a[0] / a[1] - 100 * b[0] / b[1]
                out.append(f"{m} {dv:+.1f}" + (" (기준 넘음)" if (m != "nd30" and dv > 5.0) or (m == "nd30" and (dv > 2.0 or 100 * a[0] / a[1] > 2.0)) else ""))
        print(f"- {c}: " + " · ".join(out))
    print("seed 별 C0 대비 차 [%p] (seed1/2/3):")
    for c in ("R2", "R2pla"):
        out = []
        for m in MEAS0:
            ds = [(loadx(c, m, sd), loadx("C0", m, sd)) for sd in (1, 2, 3)]
            if all(a and b for a, b in ds): out.append(f"{m} " + "/".join(f"{rate(a) - rate(b):+.1f}" for a, b in ds))
        print(f"- {c}: " + " · ".join(out))

# 정착 자세(무외란 30 s)
def pose(tag):
    t = os.path.join(O, f"ground_recovery_lk_{tag}.txt")
    if not os.path.exists(t): return None
    out = {}
    for ln in open(t, encoding="utf-8"):
        if ln.startswith("[정착 자세]"):
            k = "hold" if "홀드" in ln else "end"
            g = re.search(r"env (\d+) 평균\): 피치 ([-+\d.]+)° \(env 표준편차 ([\d.]+)\) · 롤 ([-+\d.]+)° \(표준편차 ([\d.]+)\) · 골반 높이 ([\d.]+) m", ln)
            q = {a: float(b) for a, b in re.findall(r"(\w+_joint)=([-+\d.]+)\(", ln)}
            out[k] = dict(n=int(g.group(1)), p=float(g.group(2)), ps=float(g.group(3)), r=float(g.group(4)), rs=float(g.group(5)), h=float(g.group(6)), q=q)
    return out or None
PT = [("C0", ["C0_nd30p", "C0_nd30_s2", "C0_nd30_s3"]), ("R2", ["R2_nd30p", "R2_nd30_s2", "R2_nd30_s3"]), ("R2pla", ["R2pla_nd30", "R2pla_nd30_s2", "R2pla_nd30_s3"])]
if pose("R2pla_nd30"):
    print("\n### 정착 자세 (무외란, 사전검사 통과·끝까지 생존 env 평균) — seed 1 / 2 / 3")
    for k, lb in (("hold", "홀드(정책 없음) 마지막 0.5 s"), ("end", "정책 서기 마지막 2 s(인계 뒤 28–30 s)")):
        print(f"\n{lb}\n\n| 조건 | env 수 | 몸통 피치 [°] (+앞) | 롤 [°] | 골반 높이 [m] | 왼 무릎 [°] | 오른 무릎 [°] | 왼/오 힙 앞뒤 [°] | 왼/오 발목 앞뒤 [°] |"); print("|---|---|---|---|---|---|---|---|---|")
        for c, tags in PT:
            ps = [pose(t) for t in tags]; ps = [p[k] for p in ps if p]
            if not ps: continue
            j = lambda f: " / ".join(f(p) for p in ps)
            print(f"| {c} | {j(lambda p: str(p['n']))} | {j(lambda p: '%+.2f' % p['p'])} | {j(lambda p: '%+.2f' % p['r'])} | {j(lambda p: '%.4f' % p['h'])} | {j(lambda p: '%+.2f' % p['q']['left_knee_joint'])} | {j(lambda p: '%+.2f' % p['q']['right_knee_joint'])} | {j(lambda p: '%+.1f,%+.1f' % (p['q']['left_hip_f_joint'], p['q']['right_hip_f_joint']))} | {j(lambda p: '%+.1f,%+.1f' % (p['q']['left_ankle_f_joint'], p['q']['right_ankle_f_joint']))} |")
