"""G3/G4: 회복 체크포인트 선택 — 같은 감지기(상대 v1: 4/2.5, τ10)·같은 평가 seed 로 21793 vs cont-lr 마지막 4 체크포인트.
지표: 편향 0 sf12(seed 1–3)·sf15(seed 1–2) 낙상, 힙 편향 곡선(−1.5/0/+1.5/+3°, sf12 seed 1–3) 사전검사 통과 env 중 낙상·끝까지 성공, 민감도 S = (낙상(+1.5) − 낙상(−1.5))/3°."""
import csv, os, statistics as st
L = "/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{}_{}_s{}_envs.csv"
def stat(tag, c, seeds):
    n = f = tot = ok = 0
    for sd in seeds:
        p = L.format(tag, c, sd)
        if not os.path.exists(p): return None
        for r in csv.DictReader(open(p)):
            tot += 1
            if r["pre_pass"] == "1":
                n += 1; f += r["fell"] == "1"; ok += r["fell"] != "1"
    return (100 * f / max(n, 1), 100 * ok / max(tot, 1))
C = {"21793 (G24R-a)": {"b0": "rel_c5b", "m15": "relm15_c5b", "p15": "relp15_c5b", "p30": "relp30_c5b"}}
for ck in (23000, 23100, 23200, 23292):
    C[f"cont-lr {ck}"] = {k: f"k4clr{ck}{k}" for k in ("b0", "m15", "p15", "p30")}
print("| 후보 | sf12 낙상 | sf15 낙상 | 편향 −1.5 / 0 / +1.5 / +3° 낙상 (끝까지 성공) | S %/° | 최악 +3° 끝까지 성공 |")
print("|---|---|---|---|---|---|")
rows = {}
for nm, t in C.items():
    a = stat(t["b0"], "sf12", (1, 2, 3)); b = stat(t["b0"], "sf15", (1, 2))
    cur = {k: stat(t[k], "sf12", (1, 2, 3)) for k in ("m15", "b0", "p15", "p30")}
    if a is None or b is None or None in cur.values():
        print(f"| {nm} | (미완) | | | | |"); continue
    S = (cur["p15"][0] - cur["m15"][0]) / 3.0
    rows[nm] = (a[0], b[0], S, cur["p30"][1])
    print(f"| {nm} | {a[0]:.1f} % | {b[0]:.1f} % | " + " / ".join(f"{cur[k][0]:.1f} ({cur[k][1]:.0f})" for k in ("m15", "b0", "p15", "p30")) + f" | {S:.1f} | {cur['p30'][1]:.1f} % |")
ck = [v for k, v in rows.items() if k.startswith("cont-lr")]
if ck:
    for i, nm in enumerate(("sf12", "sf15", "S", "+3° 끝까지")):
        vals = [v[i] for v in ck]
        print(f"cont-lr 4 체크포인트 {nm}: 평균 {st.mean(vals):.1f} · sd {st.pstdev(vals):.1f} · 범위 {min(vals):.1f}–{max(vals):.1f}" + (f" · 21793 {rows['21793 (G24R-a)'][i]:.1f}" if '21793 (G24R-a)' in rows else ""))
