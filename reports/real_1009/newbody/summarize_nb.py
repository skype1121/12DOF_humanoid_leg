#!/usr/bin/env python3
"""새 몸 후보 평가 요약(NEWBODY_TRAIN.md §5 사전 기준 그대로) — ev/out/ground_recovery_nb_<팔>_<측정>_s<seed>.{txt,_envs.csv,_modefall.csv} 를 읽는다.
무효(고정 관절 변위 ≥ 0.3°) 실행은 run_lock2.sh 가 out/ 에 복사하지 않으므로 여기 없다 → log_picture/ 에서 값만 참고로 읽어 '무효' 로 표시.
사용: summarize_nb.py [팔 ...]  → 표(마크다운)를 stdout 으로."""
import csv
import os
import re
import sys
from math import sqrt

H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "ev", "out"); LP = os.path.join(H, "..", "..", "..", "log_picture")
MEAS = ["nd30", "l18", "r18", "b18", "fi40", "sf15"]
NAME = {"nd30": "무외란 30 s", "l18": "왼 18 N", "r18": "오른 18 N", "b18": "뒤 18 N", "fi40": "앞 40 N", "sf15": "앞 램프 15 N"}
THR = {"fi40": 29.5, "sf15": 22.6, "b18": 12.6}          # 사전 기준 P1·P2·P3 [%]
ARMS = sys.argv[1:] or ["bz", "bo", "c1_22300", "c1_22800", "c1_23292", "ct_22300", "ct_22800", "ct_23292", "c2_22300", "c2_22800", "c2_23292"]
LABEL = {"bz": "기준선 G24Ra-21793 (관측 0 켬)", "bo": "기준선 G24Ra-21793 (관측 0 끔)"}
for _c in ("22300", "22800", "23292"):
    LABEL[f"c1_{_c}"] = f"후보 NB-s1 {_c}"; LABEL[f"c2_{_c}"] = f"후보 NB-s2 {_c}"; LABEL[f"ct_{_c}"] = f"대조 g24rcont {_c}(원 몸 학습)"


def wil(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def fmt(k, n):
    p, lo, hi = wil(k, n)
    return f"{k}/{n} = {p:.1f} % ({lo:.1f}–{hi:.1f})" if n else "—"


def load(arm, m, s):
    tag = f"nb_{arm}_{m}_s{s}"; t = os.path.join(OUT, f"ground_recovery_{tag}.txt"); valid = True
    if not os.path.isfile(t):
        t = os.path.join(LP, f"ground_recovery_{tag}.txt"); valid = False
        if not os.path.isfile(t):
            return None
    txt = open(t, errors="ignore").read()
    if not valid and "무효!" not in txt:
        return None      # out/ 에 없고 무효 표시도 없음 = 다른 이유로 실패한 실행 → 쓰지 않음
    c = t[:-4] + "_envs.csv"
    rows = list(csv.DictReader(open(c)))
    d = {"valid": valid, "tag": tag, "rows": {int(r["env"]): r for r in rows}}
    pp = [r for r in rows if r["pre_pass"] == "1"]
    d["n"] = len(pp); d["k"] = sum(r["fell"] == "1" for r in pp)
    al = [r for r in pp if r["alive_end"] == "1"]; d["alive"] = len(al); d["quiet"] = sum(r["final_quiet"] == "1" for r in al)
    mm = re.search(r"최대 변위: 전 env ([\d.]+)°", txt); d["lockdev"] = float(mm.group(1)) if mm else float("nan")
    mm = re.search(r"\[사전 검사 통과\] env \d+:.*\n\s+정지 구간.*?action-rate RMS ([\d.]+).*?\|ω\| ([\d.]+) rad/s · 관절속도 RMS ([\d.]+)", txt)
    d["q"] = tuple(float(x) for x in mm.groups()) if mm else None
    mm = re.search(r"회복 정책 사용 시간 비율 ([\d.]+) %", txt); d["rec_pct"] = float(mm.group(1)) if mm else float("nan")
    mm = re.search(r"회복 모드 중 (\d+) · 서기 모드 중 (\d+) \(외란 뒤 회복 모드가 한 번도 안 켜짐 (\d+) · 켜졌다가 서기로 돌아온 뒤 (\d+)\)", txt)
    d["mode"] = tuple(int(x) for x in mm.groups()) if mm else None
    d["zero"] = "lock_obs_zero 켬" in txt
    return d


D = {a: {m: {s: load(a, m, s) for s in (1, 2, 3, 4)} for m in MEAS} for a in ARMS}
D = {a: v for a, v in D.items() if any(x for m in MEAS for x in v[m].values())}
ARMS = list(D)


def pooled(a, m, seeds=None):
    xs = [(s, x) for s, x in D[a][m].items() if x and x["valid"] and (seeds is None or s in seeds)]
    return sum(x["k"] for _, x in xs), sum(x["n"] for _, x in xs), [s for s, _ in xs]


print("### 낙상/사전검사 통과 env — 유효 seed 합 (95 % Wilson) · 괄호 = 쓴 평가 seed, 무효 seed 는 †로 따로\n")
print("| 팔 | " + " | ".join(NAME[m] for m in MEAS) + " |"); print("|---|" + "---|" * len(MEAS))
for a in ARMS:
    cells = []
    for m in MEAS:
        k, n, ss = pooled(a, m); inv = [f"s{s} {x['k']}/{x['n']}({x['lockdev']:.3f}°)" for s, x in D[a][m].items() if x and not x["valid"]]
        cells.append((fmt(k, n) + f" [s{','.join(map(str, ss))}]" if n else "미실행") + (" †" + ";".join(inv) if inv else ""))
    print(f"| {LABEL.get(a, a)} | " + " | ".join(cells) + " |")
print("\nseed 별 낙상 % (유효만):")
for a in ARMS:
    print(f"- {LABEL.get(a, a)}: " + " · ".join(f"{m} " + "/".join(f"{100 * x['k'] / x['n']:.1f}" for s, x in sorted(D[a][m].items()) if x and x["valid"]) for m in MEAS))

if "bz" in D:
    print("\n### 기준선(관측 0 켬) 대비 — 양쪽 다 유효한 seed 만, 같은 env 짝(둘 다 사전검사 통과)\n")
    print("| 팔 | 측정 | seed | 팔 | 기준선 | 차 [%p] | 짝: 기준선만 낙상 / 팔만 낙상 (공통 env) |"); print("|---|---|---|---|---|---|---|")
    for a in ARMS:
        if a == "bz":
            continue
        for m in MEAS:
            ss = [s for s in (1, 2, 3, 4) if D[a][m][s] and D[a][m][s]["valid"] and D["bz"][m][s] and D["bz"][m][s]["valid"]]
            if not ss:
                continue
            ka, na, _ = pooled(a, m, ss); kb, nb, _ = pooled("bz", m, ss); ob = oa = cm = 0
            for s in ss:
                ra, rb = D[a][m][s]["rows"], D["bz"][m][s]["rows"]
                for e in ra:
                    if ra[e]["pre_pass"] == "1" and rb[e]["pre_pass"] == "1":
                        cm += 1; fa = ra[e]["fell"] == "1"; fb = rb[e]["fell"] == "1"; ob += (fb and not fa); oa += (fa and not fb)
            print(f"| {LABEL.get(a, a)} | {NAME[m]} | {','.join(map(str, ss))} | {fmt(ka, na)} | {fmt(kb, nb)} | {100 * ka / na - 100 * kb / nb:+.1f} | {ob} / {oa} ({cm}) |")

print("\n### 정숙성(무외란 30 s, 사전검사 통과 env 정지 구간, seed 평균) · 밀기 뒤 정지 복귀(살아남은 env 중 최종 정지)\n")
print("| 팔 | action-rate RMS | \\|ω\\| [rad/s] | 관절속도 RMS | 회복 정책 사용 시간 % | 최종 정지/생존: 뒤 18 N | 앞 40 N | 앞 램프 15 N |"); print("|---|---|---|---|---|---|---|---|")
Q = {}
for a in ARMS:
    xs = [x for x in D[a]["nd30"].values() if x and x["valid"] and x["q"]]
    q = [sum(x["q"][i] for x in xs) / len(xs) for i in range(3)] if xs else [float("nan")] * 3
    rp = sum(x["rec_pct"] for x in xs) / len(xs) if xs else float("nan")
    fq = {}
    for m in ("b18", "fi40", "sf15"):
        ys = [x for x in D[a][m].values() if x and x["valid"]]; al = sum(x["alive"] for x in ys); qu = sum(x["quiet"] for x in ys); fq[m] = (qu, al)
    Q[a] = (q, rp, fq)
    print(f"| {LABEL.get(a, a)} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {rp:.2f} | " + " | ".join(f"{fq[m][0]}/{fq[m][1]} = {100 * fq[m][0] / max(1, fq[m][1]):.1f} %" for m in ("b18", "fi40", "sf15")) + " |")

print("\n### 낙상 틱에 행동을 낸 정책(사전검사 통과 낙상 env, 유효 seed 합)\n")
print("| 팔 | 측정 | 낙상 | 회복 모드 중 | 서기 모드 중: 회복이 한 번도 안 켜짐 | 서기 모드 중: 회복 뒤 서기로 돌아온 다음 |"); print("|---|---|---|---|---|---|")
for a in ARMS:
    for m in ("b18", "fi40", "sf15", "nd30"):
        ys = [x for x in D[a][m].values() if x and x["valid"] and x["mode"]]
        if not ys:
            continue
        r = sum(x["mode"][0] for x in ys); s0 = sum(x["mode"][2] for x in ys); s1 = sum(x["mode"][3] for x in ys); k = sum(x["k"] for x in ys)
        print(f"| {LABEL.get(a, a)} | {NAME[m]} | {k} | {r} ({100 * r / max(1, k):.0f} %) | {s0} | {s1} |")

if "bz" in D:
    print("\n### 사전 기준 판정 (유효 seed 합, 기준선 = 관측 0 켬)\n")
    print("| 팔 | S1 무외란 ≤ 2 % ∧ 기준선 +1 %p 이내 | P1 앞 40 N ≤ 29.5 % | P2 앞 램프 ≤ 22.6 % | P3 뒤 18 N ≤ 12.6 % | Q1 정숙성(+10 % 이내 ∧ 회복 사용 +1 %p 이내) | Q2 정지 복귀(−5 %p 이내) | 종합 |"); print("|---|---|---|---|---|---|---|---|")
    kb, nb, _ = pooled("bz", "nd30"); pb = 100 * kb / max(1, nb); qb, rb_, fb = Q["bz"]
    for a in ARMS:
        if a in ("bz", "bo"):
            continue
        k, n, _ = pooled(a, "nd30")
        if not n:
            continue
        pa = 100 * k / n; s1 = pa <= 2.0 and pa - pb <= 1.0; res = [("통과" if s1 else "미달") + f" ({pa:.1f} vs {pb:.1f})"]; ok = s1
        for m in ("fi40", "sf15", "b18"):
            k, n, _ = pooled(a, m); p = 100 * k / max(1, n); g = n > 0 and p <= THR[m]; ok = ok and g; res.append(("통과" if g else "미달") + f" ({p:.1f})")
        q, rp, fq = Q[a]; q1 = all(q[i] <= qb[i] * 1.10 for i in range(3)) and rp <= rb_ + 1.0; ok = ok and q1
        res.append(("통과" if q1 else "미달") + " (" + "/".join(f"{100 * (q[i] / qb[i] - 1):+.0f} %" for i in range(3)) + f", 회복 {rp - rb_:+.2f} %p)")
        dq = [100 * fq[m][0] / max(1, fq[m][1]) - 100 * fb[m][0] / max(1, fb[m][1]) for m in ("b18", "fi40", "sf15")]; q2 = all(x >= -5.0 for x in dq); ok = ok and q2
        res.append(("통과" if q2 else "미달") + " (" + "/".join(f"{x:+.1f}" for x in dq) + " %p)")
        print(f"| {LABEL.get(a, a)} | " + " | ".join(res) + f" | **{'통과' if ok else '미달'}** |")
