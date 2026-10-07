"""HD1 판정(HEADING_SESSION_1007.md §7) — 헤딩 고리 켬 대 끔, X7 대본, 배포 코드 폐루프.
사용: python3 reports/gen_w3/hd1_judge.py [--seeds 1,2,3]
H1a 낙상(48 s, 통과 env): 켬 − 끔 ≤ +1 %p 충족 / > +5 %p 불합격 / 그 사이 미결(비짝 차이의 95 % 구간 병기, env 짝 McNemar 는 보조).
H1b 방향 유지: 비회전 걷기 구간(1·3·4·5)의 |Δψ|(걷기 켜짐 → KE 인계, 그 구간을 마친 env) 중앙이 켬에서 끔의 절반 이하.
H1c 고장: 켬 팔에서 헤딩 고리 고장으로 끊긴 비회전 구간 ≤ 2 %(분모 = 구간 시작 때 생존 env × 구간)."""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
STARTS = [2.0, 11.0, 20.0, 29.0, 38.0]; NONTURN = [0, 2, 3, 4]; NAMES = ["앞", "회전", "옆", "뒤", "앞"]


def load(arm, s):
    p = os.path.join(LP, f"kwloop_tr_hd1{arm}_d2_s{s}.npz"); q = os.path.join(LP, f"ground_recovery_tr_hd1{arm}_d2_s{s}_envs.csv")
    if not (os.path.exists(p) and os.path.exists(q)):
        return None
    z = np.load(p, allow_pickle=True); pre = z["pre_pass"].astype(bool); tf = np.full(pre.shape[0], np.inf)
    for r in csv.DictReader(open(q)):
        if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
            tf[int(r["env"])] = float(r["t_fall_rel"])
    return dict(t=z["t"].astype(float), yaw=z["yaw"].astype(float), pre=pre, tf=tf, ev=json.loads(str(z["events"])), wzc=(z["wzc"].astype(float) if "wzc" in z.files and z["wzc"].shape[0] else None))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2,3"); a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    R = {}; miss = []
    for arm in ("off", "on"):
        n = f = 0; dpsi = [[] for _ in range(5)]; segs = [0] * 5; faults = [0] * 5; fellset = {}; sat = []
        for s in seeds:
            D = load(arm, s)
            if D is None:
                miss.append(f"{arm}/s{s}"); continue
            t = D["t"]; K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
            for e in np.nonzero(D["pre"])[0]:
                n += 1; fell = bool(np.isfinite(D["tf"][e])); f += fell; fellset[(s, int(e))] = fell
                E = D["ev"][e]; en = [tt for tt, nm in E if nm == "enable"]; tk = [tt for tt, nm in E if nm == "to_ke"]; hf = [tt for tt, nm in E if nm == "hl_fault"]
                for c, ts in enumerate(STARTS):
                    if D["tf"][e] < ts:
                        break
                    segs[c] += 1
                    t1 = STARTS[c + 1] if c + 1 < 5 else 1e9
                    faults[c] += any(ts <= x < t1 for x in hf)
                    e0 = [x for x in en if ts - 1e-6 <= x < t1]; k0 = [x for x in tk if x > (e0[0] if e0 else 1e9) and x < t1 + 5]
                    if e0 and k0 and D["tf"][e] > k0[0]:
                        d = D["yaw"][K(k0[0]), e] - D["yaw"][K(e0[0]), e]; dpsi[c].append(abs(np.degrees(np.arctan2(np.sin(d), np.cos(d)))))
            if D["wzc"] is not None:
                sat.append(float(np.mean(np.abs(D["wzc"][:, D["pre"]]) >= 0.39)))
        R[arm] = dict(n=n, f=f, dpsi=dpsi, segs=segs, faults=faults, fell=fellset, sat=sat)
    off, on = R["off"], R["on"]
    print(f"## HD1 (seed {seeds}) — 통과 env 끔 {off['n']} · 켬 {on['n']}" + (f" · **미완 {', '.join(miss)}**" if miss else ""))
    if not off["n"] or not on["n"]:
        return
    pa, pb = off["f"] / off["n"], on["f"] / on["n"]; d = 100 * (pb - pa); se = 100 * np.sqrt(pa * (1 - pa) / off["n"] + pb * (1 - pb) / on["n"])
    ks = [k for k in on["fell"] if k in off["fell"]]; b = sum(1 for k in ks if on["fell"][k] and not off["fell"][k]); c = sum(1 for k in ks if off["fell"][k] and not on["fell"][k])
    h1a = "충족" if d <= 1.0 else ("불합격" if d > 5.0 else "미결")
    print(f"- H1a 낙상(48 s): 끔 {off['f']}/{off['n']} ({100*pa:.1f} %) · 켬 {on['f']}/{on['n']} ({100*pb:.1f} %) · 차 {d:+.1f} %p (95 % 구간 {d-1.96*se:+.1f} ~ {d+1.96*se:+.1f}) · env 짝 켬만 {b} / 끔만 {c} (p {mcnemar(b, c):.3f}) → {h1a}")
    print("| 걷기 구간 | 끔 \\|Δψ\\| 중앙 (n) | 켬 \\|Δψ\\| 중앙 (n) | 켬 / 끔 | 켬 팔 고리 고장 / 구간 |"); print("|---|---|---|---|---|")
    mo = []; mn = []
    for cidx in range(5):
        a0 = np.array(off["dpsi"][cidx]); a1 = np.array(on["dpsi"][cidx])
        print(f"| {cidx+1} {NAMES[cidx]} | {np.median(a0):.1f} ({len(a0)}) | {np.median(a1):.1f} ({len(a1)}) | {np.median(a1)/max(np.median(a0),1e-9):.2f} | {on['faults'][cidx]} / {on['segs'][cidx]} ({100*on['faults'][cidx]/max(on['segs'][cidx],1):.1f} %) |")
        if cidx in NONTURN:
            mo += list(a0); mn += list(a1)
    ratio = np.median(mn) / max(np.median(mo), 1e-9); h1b = "충족" if ratio <= 0.5 else "미달"
    nf = sum(on["faults"][c] for c in NONTURN); ns = sum(on["segs"][c] for c in NONTURN); h1c = "충족" if 100 * nf / max(ns, 1) <= 2.0 else "미달"
    print(f"- H1b 방향 유지(비회전 구간 합): \\|Δψ\\| 중앙 끔 {np.median(mo):.1f}° → 켬 {np.median(mn):.1f}° (비 {ratio:.2f}) → {h1b}")
    print(f"- H1c 고리 고장으로 끊긴 비회전 구간: {nf} / {ns} ({100*nf/max(ns,1):.1f} %) → {h1c}" + (f" · 고리 wz 포화(≥ 0.39 rad/s) 시간 비율 {100*np.mean(on['sat']):.2f} %" if on["sat"] else ""))
    if miss:
        print("- **미완 — 판정 보류**(위는 중간값)")
    else:
        print(f"- **판정**: {'세 기준 충족 → 현행 규칙에서 고리 켬이 합성 구조에서도 성립' if (h1a == '충족' and h1b == '충족' and h1c == '충족') else '불합격 또는 미결(' + ', '.join(k for k, v in (('H1a', h1a), ('H1b', h1b), ('H1c', h1c)) if v != '충족') + ') → 첫 실물 걷기는 고리 끔'}")


if __name__ == "__main__":
    main()
