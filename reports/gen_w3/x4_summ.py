"""X4 집계(TRANSITION_W3KE.md §5.6d) — 깨끗한 KE 서기에서 가벼운 밀기로 회복 모드를 한 번 켠 뒤의 앞 램프 낙상(H3).
입력: log_picture/transall_tr_x4{C,Pb,Pf}_sf15_d2_s*.npz (사본 평가기 추적). 분모 = 사전검사 통과 ∧ 램프 시작(16 s) 시 생존 ∧ KE 활성.
사용: python3 reports/gen_w3/x4_summ.py"""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
T_PUSH, T_RAMP = 9.0, 16.0


def load(c):
    D = {}
    for p in sorted(glob.glob(os.path.join(LP, f"transall_tr_x4{c}_sf15_d2_s*.npz"))):
        s = int(re.search(r"_s(\d+)\.npz", p).group(1)); z = np.load(p, allow_pickle=True)
        t = z["t"].astype(float); kp = int(np.searchsorted(t, T_PUSH - 1e-6)); kr = int(np.searchsorted(t, T_RAMP - 1e-6))
        alive = z["alive"].astype(bool); pre = z["pre_pass"].astype(bool); mode = z["mode"].astype(bool); ke = z["ke"].astype(bool)
        ref = z["ref"].astype(float); rp2 = z["rp2"].astype(float); ffwd = z["ffwd"].astype(float); pit = z["pitch"].astype(float)
        fell = z["fell"].astype(bool); tf = z["t_fall_post"].astype(float)
        for e in range(alive.shape[1]):
            if not pre[e] or not ke[kp - 1, e] or not alive[kp - 1, e]:
                continue
            a_r = bool(alive[kr - 1, e])
            D[(s, e)] = dict(alive_ramp=a_r, fell_push=bool(fell[e] and T_PUSH - 1e-6 <= tf[e] < T_RAMP - 1e-6), fell_ramp=bool(a_r and fell[e] and tf[e] >= T_RAMP - 1e-6),
                             act=bool(mode[kp:kr, e].any()), act_s=float(mode[kp:kr, e].sum() * 0.02), pre_on=bool(mode[kr - 1, e]), on_before_push=bool(mode[:kp, e].any()),
                             ref_err=float(ref[kr - 1, e, 1] - np.nanmean(rp2[kr - 50:kr, e, 1])), pitch=float(np.nanmean(pit[kr - 50:kr, e])),
                             fmean=100 * float(np.nanmean(ffwd[kr - 50:kr, e, :])), stag=100 * float(np.nanmean(ffwd[kr - 50:kr, e, 0] - ffwd[kr - 50:kr, e, 1])),
                             lat=(lambda on: float(t[kr + on[0]] - T_RAMP) if len(on) else float("nan"))(np.nonzero(mode[kr:, e])[0]))
    return D


def main():
    D = {c: load(c) for c in ("C", "Pb", "Pf")}
    names = {"C": "C 램프만", "Pb": "Pb 뒤 12 N(9 s) + 램프", "Pf": "Pf 앞 12 N(9 s) + 램프"}
    print("| 조건 | seed | n(9 s 생존) | 첫 밀기 낙상 | 램프 시작 생존 | 램프 뒤 낙상 | Wilson 95 % | 9–16 s 회복 모드 켜진 env | 램프 직전 켜짐 | 켜진 env 램프 낙상 | 안 켜진 env 램프 낙상 | 램프 직전 참 피치·발 앞축 평균·|어긋남| 중앙 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ("C", "Pb", "Pf"):
        v = list(D[c].values())
        if not v:
            print(f"| {names[c]} | 자료 없음 |"); continue
        seeds = sorted({k[0] for k in D[c]}); n = len(v); fp = sum(x["fell_push"] for x in v); ar = [x for x in v if x["alive_ramp"]]; fr = sum(x["fell_ramp"] for x in ar)
        lo, hi = wilson(fr, len(ar)); act = [x for x in ar if x["act"]]; non = [x for x in ar if not x["act"]]
        print(f"| {names[c]} | {','.join(map(str, seeds))} | {n} | {fp} | {len(ar)} | {fr} ({100*fr/max(len(ar),1):.1f} %) | {100*lo:.1f}–{100*hi:.1f} | {len(act)} ({100*len(act)/max(len(ar),1):.1f} %) | {sum(x['pre_on'] for x in ar)} | "
              f"{sum(x['fell_ramp'] for x in act)}/{len(act)} ({100*sum(x['fell_ramp'] for x in act)/max(len(act),1):.1f} %) | {sum(x['fell_ramp'] for x in non)}/{len(non)} ({100*sum(x['fell_ramp'] for x in non)/max(len(non),1):.1f} %) | "
              f"{np.median([x['pitch'] for x in ar]):+.2f}° · {np.median([x['fmean'] for x in ar]):+.2f} cm · {np.median(np.abs([x['stag'] for x in ar])):.2f} cm |")
    for c in ("Pb", "Pf"):
        ks = sorted(k for k in set(D["C"]) & set(D[c]) if D["C"][k]["alive_ramp"] and D[c][k]["alive_ramp"])
        if not ks:
            continue
        b = sum(1 for k in ks if D["C"][k]["fell_ramp"] and not D[c][k]["fell_ramp"]); cc = sum(1 for k in ks if D[c][k]["fell_ramp"] and not D["C"][k]["fell_ramp"])
        fC = sum(D["C"][k]["fell_ramp"] for k in ks); fP = sum(D[c][k]["fell_ramp"] for k in ks)
        print(f"- 짝 C vs {c} (둘 다 램프 시작 생존, n {len(ks)}): 램프 낙상 C {fC} ({100*fC/len(ks):.1f} %) / {c} {fP} ({100*fP/len(ks):.1f} %) · 차 {100*(fP-fC)/len(ks):+.1f} %p · C만 {b} / {c}만 {cc} (McNemar p {mcnemar(b, cc):.3g})")
        # 켜진 env 만(같은 env 의 C 와 비교) — 상관 수준
        ka = [k for k in ks if D[c][k]["act"]]; kn = [k for k in ks if not D[c][k]["act"]]
        for nm, kk in (("회복 모드가 켜졌던 env", ka), ("안 켜졌던 env", kn)):
            if kk:
                fC2 = sum(D["C"][k]["fell_ramp"] for k in kk); fP2 = sum(D[c][k]["fell_ramp"] for k in kk)
                b2 = sum(1 for k in kk if D["C"][k]["fell_ramp"] and not D[c][k]["fell_ramp"]); c2 = sum(1 for k in kk if D[c][k]["fell_ramp"] and not D["C"][k]["fell_ramp"])
                print(f"    {nm} (n {len(kk)}): 같은 env 의 C {fC2} ({100*fC2/len(kk):.1f} %) → {c} {fP2} ({100*fP2/len(kk):.1f} %) · C만 {b2} / {c}만 {c2} (p {mcnemar(b2, c2):.3g})")


if __name__ == "__main__":
    main()
