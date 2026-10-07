"""KE 감지기 갇힘 해소 후보(requiet) 확증 배터리 판정 — reports/gen_w3/KE_REQUIET.md §4 의 K1–K7 을 그대로 채점한다(기준 수정 금지).
입력: log_picture/transall_keq_{KE,KEq,KEH,KEHq}_{cond}_d{D}_s{seed}.npz (사본 평가기 추적 128 env) · rqlog_keq_*.json(재기준 발동 [t, env, 묵은 기준 대비 기울기°]).
사용: python3 reports/gen_w3/keq_judge.py [--seeds 21,22,23] [--aux]
짝 = 같은 seed·env. 분모 규칙은 각 기준 옆에 적는다. 미완 실행이 있으면 '미완'으로 표시하고 판정하지 않는다."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
T1, T2 = 4.0, 12.0


def load(arm, cond, d, seed):
    p = os.path.join(LP, f"transall_keq_{arm}_{cond}_d{d}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True); t = z["t"].astype(float)
    R = dict(t=t, mode=z["mode"].astype(bool), alive=z["alive"].astype(bool), pre=z["pre_pass"].astype(bool), fell=z["fell"].astype(bool), tf=z["t_fall_post"].astype(float))
    q = os.path.join(LP, f"rqlog_keq_{arm}_{cond}_d{d}_s{seed}.json")
    R["fires"] = json.load(open(q)).get("fires", []) if os.path.exists(q) else []
    R["rqlog_missing"] = (not os.path.exists(q)) and arm.endswith("q")       # 후보 팔인데 발동 기록 파일이 없으면 K7 을 믿을 수 없다(독립 검토 지적)
    return R


def two_dist(a, b, cond, d, seeds):
    """bb·fb·padL: 둘째 외란(12 s) 기준. 반환 dict: n(둘 다 12 s 생존), 낙상 수, 짝 불일치, 직전 회복 모드 비율(팔별), 미완 seed."""
    out = dict(n=0, fa=0, fb=0, onlyA=0, onlyB=0, stuckA=0, stuckB=0, nA=0, nB=0, miss=[], keys=[])
    for s in seeds:
        A = load(a, cond, d, s); B = load(b, cond, d, s)
        if A is None or B is None:
            out["miss"].append(s); continue
        k2 = int(np.searchsorted(A["t"], T2 - 1e-6))
        for e in range(A["pre"].shape[0]):
            if not (A["pre"][e] and B["pre"][e]):
                continue
            aA = bool(A["alive"][k2 - 1, e]); aB = bool(B["alive"][k2 - 1, e])
            if aA:
                out["nA"] += 1; out["stuckA"] += bool(A["mode"][k2 - 1, e])
            if aB:
                out["nB"] += 1; out["stuckB"] += bool(B["mode"][k2 - 1, e])
            if aA and aB:
                x = bool(A["fell"][e] and A["tf"][e] >= T2 - 1e-6); y = bool(B["fell"][e] and B["tf"][e] >= T2 - 1e-6)
                out["n"] += 1; out["fa"] += x; out["fb"] += y; out["onlyA"] += (x and not y); out["onlyB"] += (y and not x); out["keys"].append(((s, e), x, y))
    return out


def one_dist(a, b, cond, d, seeds):
    out = dict(n=0, fa=0, fb=0, onlyA=0, onlyB=0, miss=[])
    for s in seeds:
        A = load(a, cond, d, s); B = load(b, cond, d, s)
        if A is None or B is None:
            out["miss"].append(s); continue
        for e in range(A["pre"].shape[0]):
            if A["pre"][e] and B["pre"][e]:
                x = bool(A["fell"][e]); y = bool(B["fell"][e]); out["n"] += 1; out["fa"] += x; out["fb"] += y; out["onlyA"] += (x and not y); out["onlyB"] += (y and not x)
    return out


def nd_stats(arm, d, seeds):
    n = fell = 0; ons = 0; env_min = 0.0; mode_t = 0; alive_t = 0; miss = []
    for s in seeds:
        A = load(arm, "nd60", d, s)
        if A is None:
            miss.append(s); continue
        m = A["mode"] & A["alive"]
        for e in range(A["pre"].shape[0]):
            if not A["pre"][e]:
                continue
            n += 1; fell += bool(A["fell"][e]); ons += int((m[1:, e] & ~m[:-1, e]).sum()); env_min += float(A["alive"][:, e].sum()) * 0.02 / 60.0
            mode_t += int(m[:, e].sum()); alive_t += int(A["alive"][:, e].sum())
    return dict(n=n, fell=fell, rate=ons / env_min if env_min > 0 else float("nan"), frac=mode_t / alive_t if alive_t else float("nan"), miss=miss)


def fires_stats(arm, conds_d, seeds, base=None):
    """반환: 합산 (발동 수, 6° 초과, 2 s 안 낙상), 조건별 표 행, 발동 기록 누락 실행 목록. base = 기준 팔 이름이면 '2 s 안 낙상' 중 기준 팔에서도 같은 env 가 ±0.5 s 안에 넘어진 건수를 함께 센다."""
    nf = big = f2 = 0; rows = []; missing = []
    for cond, d in conds_d:
        c_nf = c_big = c_f2 = c_same = c_pre = 0
        for s in seeds:
            A = load(arm, cond, d, s)
            if A is None:
                continue
            if A["rqlog_missing"]:
                missing.append(f"{cond}/d{d}/s{s}")
            B = load(base, cond, d, s) if base else None
            for f in A["fires"]:
                tt, e, tilt = float(f[0]), int(f[1]), float(f[2]); c_nf += 1; c_big += tilt > 6.0; c_pre += bool(A["pre"][e])
                hit = bool(A["fell"][e] and tt - 1e-6 <= float(A["tf"][e]) <= tt + 2.0 + 1e-6); c_f2 += hit
                if hit and B is not None and bool(B["fell"][e]) and abs(float(B["tf"][e]) - float(A["tf"][e])) <= 0.5:
                    c_same += 1
        nf += c_nf; big += c_big; f2 += c_f2
        if c_nf:
            rows.append(f"{cond}(지연 {d}) 발동 {c_nf}(통과 env {c_pre}) · 6° 초과 {c_big} · 2 s 안 낙상 {c_f2}(그중 기준 팔도 같은 시각 낙상 {c_same})")
    return nf, big, f2, rows, missing


def pct(a, n):
    return f"{a} ({100*a/n:.1f} %)" if n else "—"


def judge(a, b, seeds, label):
    print(f"## {label}: {b} 대 {a} (seed {seeds}, 짝 = 같은 seed·env)")
    res = {}; inc = []
    # K1·K2
    tot = dict(n=0, fa=0, fb=0, oa=0, ob=0); ind = {}
    print("| 조건 | 짝 n(12 s 둘 다 생존) | 둘째 외란 뒤 낙상 기준 / 후보 | 기준만 / 후보만 · p | 12 s 직전 회복 모드: 기준 / 후보 |"); print("|---|---|---|---|---|")
    k2ok = True
    for cond in ("bb", "fb"):
        r = two_dist(a, b, cond, 2, seeds); inc += [f"{cond}/s{s}" for s in r["miss"]]
        print(f"| {cond} | {r['n']} | {pct(r['fa'], r['n'])} / {pct(r['fb'], r['n'])} | {r['onlyA']} / {r['onlyB']} · {mcnemar(r['onlyA'], r['onlyB']):.3g} | {pct(r['stuckA'], r['nA'])} / {pct(r['stuckB'], r['nB'])} |")
        tot["n"] += r["n"]; tot["fa"] += r["fa"]; tot["fb"] += r["fb"]; tot["oa"] += r["onlyA"]; tot["ob"] += r["onlyB"]
        k2ok = k2ok and r["nB"] > 0 and (100.0 * r["stuckB"] / r["nB"] <= 2.0)
        for key, x, y in r["keys"]:
            v = ind.setdefault(key, [0, 0]); v[0] += x; v[1] += y
    d1 = 100.0 * (tot["fa"] - tot["fb"]) / tot["n"] if tot["n"] else float("nan"); p1 = mcnemar(tot["oa"], tot["ob"])
    ia = sum(1 for v in ind.values() if v[0] > v[1]); ib = sum(1 for v in ind.values() if v[1] > v[0])
    res["K1"] = (d1 >= 3.0 and p1 < 0.05 and tot["oa"] > tot["ob"]); res["K2"] = k2ok
    print(f"- K1 이득: bb+fb 낙상 {pct(tot['fa'], tot['n'])} → {pct(tot['fb'], tot['n'])}, 감소 {d1:+.1f} %p, 짝 {tot['oa']} / {tot['ob']} p {p1:.3g} · 개체 단위(두 조건 접음) {ia} / {ib} p {mcnemar(ia, ib):.3g} → {'충족' if res['K1'] else '미달'}")
    print(f"- K2 갇힘(후보, 조건별 ≤ 2 %): {'충족' if k2ok else '미달'}")
    # K3
    B = C = 0; k3 = True
    print("| 단일 외란 | 짝 n | 낙상 기준 / 후보 | 차이 | 기준만 / 후보만 · p |"); print("|---|---|---|---|---|")
    for cond in ("b18", "fi40", "sf15"):
        r = one_dist(a, b, cond, 2, seeds); inc += [f"{cond}/s{s}" for s in r["miss"]]
        d = 100.0 * (r["fb"] - r["fa"]) / r["n"] if r["n"] else float("nan"); k3 = k3 and r["n"] > 0 and d <= 3.0 + 1e-9; B += r["onlyB"]; C += r["onlyA"]
        print(f"| {cond} | {r['n']} | {pct(r['fa'], r['n'])} / {pct(r['fb'], r['n'])} | {d:+.1f} %p | {r['onlyA']} / {r['onlyB']} · {mcnemar(r['onlyA'], r['onlyB']):.3g} |")
    worse = (B > C and mcnemar(B, C) < 0.05); res["K3"] = k3 and not worse
    print(f"- K3 단일 외란 비열등: 합친 짝 기준만 {C} / 후보만 {B}, p {mcnemar(B, C):.3g} → {'충족' if res['K3'] else '미달'}")
    # K4
    na = nd_stats(a, 2, seeds); nb = nd_stats(b, 2, seeds); inc += [f"nd60/s{s}" for s in sorted(set(na["miss"]) | set(nb["miss"]))]
    res["K4"] = (nb["fell"] <= na["fell"] + 1) and (nb["rate"] <= na["rate"] * 1.2 + 1e-12) and (nb["frac"] <= na["frac"] + 1e-12)
    print(f"- K4 무외란 60 s: 낙상 {na['fell']}/{na['n']} → {nb['fell']}/{nb['n']} · 회복 켜짐 {na['rate']:.3f} → {nb['rate']:.3f} 회/env·분 · 회복 모드 시간 비율 {100*na['frac']:.3f} → {100*nb['frac']:.3f} % → {'충족' if res['K4'] else '미달'}")
    # K5
    r = two_dist(a, b, "padL", 2, seeds); inc += [f"padL/s{s}" for s in r["miss"]]
    st = 100.0 * r["stuckB"] / r["nB"] if r["nB"] else float("nan"); d5 = 100.0 * (r["fb"] - r["fa"]) / r["n"] if r["n"] else float("nan")
    res["K5"] = (st <= 5.0) and (d5 <= 3.0 + 1e-9)
    print(f"- K5 패드 고장(padL): 12 s 직전 회복 모드 기준 {pct(r['stuckA'], r['nA'])} / 후보 {pct(r['stuckB'], r['nB'])} · 둘째 외란 뒤 낙상 {pct(r['fa'], r['n'])} → {pct(r['fb'], r['n'])} ({d5:+.1f} %p, 짝 {r['onlyA']} / {r['onlyB']}) → {'충족' if res['K5'] else '미달'}")
    # K6 (지연 3, seed 21·22)
    s3 = [s for s in seeds if s in (21, 22)]
    r1 = one_dist(a, b, "sf15", 3, s3); r2 = two_dist(a, b, "bb", 3, s3); n3a = nd_stats(a, 3, s3); n3b = nd_stats(b, 3, s3)
    inc += [f"d3 sf15/s{s}" for s in r1["miss"]] + [f"d3 bb/s{s}" for s in r2["miss"]] + [f"d3 nd60/s{s}" for s in sorted(set(n3a["miss"]) | set(n3b["miss"]))]
    d61 = 100.0 * (r1["fb"] - r1["fa"]) / r1["n"] if r1["n"] else float("nan"); st6 = 100.0 * r2["stuckB"] / r2["nB"] if r2["nB"] else float("nan")
    res["K6"] = (d61 <= 3.0 + 1e-9) and (st6 <= 2.0) and (n3b["fell"] <= n3a["fell"] + 1)
    print(f"- K6 지연 3: sf15 낙상 {pct(r1['fa'], r1['n'])} → {pct(r1['fb'], r1['n'])} ({d61:+.1f} %p) · bb 갇힘 기준 {pct(r2['stuckA'], r2['nA'])} / 후보 {pct(r2['stuckB'], r2['nB'])} · nd60 낙상 {n3a['fell']}/{n3a['n']} → {n3b['fell']}/{n3b['n']} → {'충족' if res['K6'] else '미달'}")
    # K7
    nf, big, f2, k7rows, k7miss = fires_stats(b, [(c, 2) for c in ("bb", "fb", "b18", "fi40", "sf15", "padL", "nd60")] + [(c, 3) for c in ("bb", "sf15", "nd60")], seeds, base=a)
    res["K7"] = nf > 0 and (100.0 * big / nf <= 5.0) and (100.0 * f2 / nf <= 2.0) and not k7miss
    print(f"- K7 돌려주는 자세(전 조건 합산 — §4 해석 고정 주석): 재기준 발동 {nf} 회 · 묵은 기준 대비 6° 초과 {pct(big, nf)} · 발동 뒤 2 s 안 낙상 {pct(f2, nf)} → {'충족' if res['K7'] else '미달'}")
    for r_ in k7rows:
        print(f"    조건별: {r_}")
    if k7miss:
        print(f"    ⚠ 발동 기록(rqlog) 없는 후보 실행: {', '.join(k7miss)} — K7 미달로 처리")
    if inc:
        print(f"- **미완 실행**: {', '.join(inc)} → 판정 보류(위의 '충족/미달'은 지금 있는 실행만의 중간값이다)")
    else:
        print(f"- **판정**: {'K1–K7 전부 충족 → 채택 제안(결정은 사용자)' if all(res.values()) else '미달 기준 ' + ', '.join(k for k, v in res.items() if not v) + ' → 기록하고 값 조정 없음'}")
    return res, inc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="21,22,23"); ap.add_argument("--aux", action="store_true")
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    judge("KE", "KEq", seeds, "주 판정")
    if a.aux:
        print(); print("## 보조(판정 아님): imuH 결합 — nd60·bb·fb 만 실행됨, 나머지 줄은 자료 없음")
        judge("KEH", "KEHq", seeds, "보조")
