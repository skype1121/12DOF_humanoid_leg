"""X10 요약(reports/gen_w3/ANKLE_RATE_X10.md) — 발목 슬루 제한 팔 대 기준선(KE-q 배터리의 KE 팔), 같은 seed·env 짝.
사용: python3 reports/gen_w3/x10_summ.py [--seeds 21,22] [--arms R2,RF2,R1] [--conds sf15,b18,fi40,bb] [--cmdstats]
출력 1: 조건별 낙상(기준선·팔)·차이·McNemar, 팔별 '비용 없음' 판정(조건별 +3 %p 이하 ∧ 합친 짝 검정에서 유의한 악화 없음).
출력 2(--cmdstats): 추적이 있는 실행에서 회복 활성화별 발목 롤·피치 목표의 1 s 스윙, 틱당 변화, 추종 오차(|목표 − 실측|), 적용 토크 — 실물 서명(1 s 에 16–26°, 오차 11–15°)과 비교."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
JN = ["LhF", "RhF", "LhA", "RhA", "LhR", "RhR", "Lkn", "Rkn", "LaF", "RaF", "LaR", "RaR"]


def load(tag):
    p = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    if not os.path.exists(p):
        return None
    return {int(r["env"]): r for r in csv.DictReader(open(p)) if r["pre_pass"] == "1"}


def outcomes(a):
    seeds = [int(x) for x in a.seeds.split(",")]; conds = a.conds.split(",")
    for arm in a.arms.split(","):
        print(f"## 팔 {arm} 대 기준선 KE(같은 seed·env 짝, seed {seeds})")
        print("| 조건 | 짝 n | 기준선 낙상 | 팔 낙상 | 차이 | 팔만 / 기준선만 | p |"); print("|---|---|---|---|---|---|---|")
        B = C = 0; ok = True; miss = []
        for cond in conds:
            n = fb = fa = b = c = 0
            for s in seeds:
                R = load(f"keq_KE_{cond}_d2_s{s}"); X = load(f"x10_{arm}_{cond}_d2_s{s}")
                if R is None or X is None:
                    miss.append(f"{cond}/s{s}{'(기준선)' if R is None else ''}"); continue
                for e in set(R) & set(X):
                    r = R[e]["fell"] == "1"; x = X[e]["fell"] == "1"; n += 1; fb += r; fa += x; b += (x and not r); c += (r and not x)
            if n == 0:
                print(f"| {cond} | 0 | — | — | — | — | — |"); ok = False; continue
            d = 100.0 * (fa - fb) / n; B += b; C += c; ok = ok and d <= 3.0 + 1e-9
            print(f"| {cond} | {n} | {fb} ({100*fb/n:.1f} %) | {fa} ({100*fa/n:.1f} %) | {d:+.1f} %p{'' if d <= 3.0 + 1e-9 else ' ✗'} | {b} / {c} | {mcnemar(b, c):.3f} |")
        p = mcnemar(B, C); worse = (B > C and p < 0.05)
        print(f"- 합친 짝: 팔만 낙상 {B} / 기준선만 낙상 {C}, p {p:.3f} · 판정: {'미완(' + ', '.join(miss) + ')' if miss else ('비용 없음 기준 충족' if ok and not worse else '기준 미달')}\n")


def cmdstats(a):
    print("## 회복 활성화별 발목 명령 통계(추적 128 env, 활성화 = 회복 모드 켜짐 → 1 s 창)")
    print("| 실행 | 활성화 수 | 발목 롤 1 s 스윙 중앙 / p90 / 최대 [°] | 스윙 ≥ 16° 비율 | 발목 롤 틱당 |Δ| 최대 중앙 [°] | 발목 롤 |목표 − 실측| 최대 중앙 / p90 [°] | 오차 ≥ 11° 비율 | 발목 롤 |τ| 최대 중앙 / p90 [N·m] | 발목 피치 1 s 스윙 중앙 / p90 [°] |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ["R0"] + a.arms.split(","):
        for cond in a.conds.split(","):
            for s in [int(x) for x in a.seeds.split(",")]:
                tag = f"x10_{arm}_{cond}_d2_s{s}"; p = os.path.join(LP, f"transall_{tag}.npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p, allow_pickle=True)
                if "cmd" not in z.files:
                    continue
                cmd = np.degrees(z["cmd"].astype(np.float32)); mode = z["mode"]; alive = z["alive"]
                q = np.degrees(z["q"].astype(np.float32)) if "q" in z.files else None; tq = z["tq"].astype(np.float32) if "tq" in z.files else None
                sw = []; dmax = []; er = []; tqm = []; swf = []
                for e in range(mode.shape[1]):
                    on = np.nonzero(mode[1:, e] & ~mode[:-1, e])[0] + 1
                    for k in on:
                        w = slice(k, min(k + 50, mode.shape[0]))
                        if not alive[w, e].all():
                            w = slice(k, k + max(1, int(np.argmin(alive[w, e]))))
                        c = cmd[w, e]
                        if c.shape[0] < 5:
                            continue
                        sw.append(float((c[:, 10:].max(0) - c[:, 10:].min(0)).max())); swf.append(float((c[:, 8:10].max(0) - c[:, 8:10].min(0)).max()))
                        dmax.append(float(np.abs(np.diff(c[:, 10:], axis=0)).max()))
                        if q is not None:
                            er.append(float(np.abs(c[:, 10:] - q[w, e, 10:]).max()))
                        if tq is not None:
                            tqm.append(float(np.abs(tq[w, e, 10:]).max()))
                if not sw:
                    continue
                sw = np.array(sw); f = lambda v, pc: float(np.percentile(v, pc))   # noqa: E731
                erq = (f"{f(er, 50):.1f} / {f(er, 90):.1f}" if er else "—"); erf = (f"{100*np.mean(np.array(er) >= 11):.0f} %" if er else "—")
                tqs = (f"{f(tqm, 50):.1f} / {f(tqm, 90):.1f}" if tqm else "—")
                print(f"| {arm} {cond} s{s} | {len(sw)} | {f(sw, 50):.1f} / {f(sw, 90):.1f} / {sw.max():.1f} | {100*np.mean(sw >= 16):.0f} % | {f(dmax, 50):.2f} | {erq} | {erf} | {tqs} | {f(swf, 50):.1f} / {f(swf, 90):.1f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="21,22"); ap.add_argument("--arms", default="R2,RF2,R1"); ap.add_argument("--conds", default="sf15,b18,fi40,bb"); ap.add_argument("--cmdstats", action="store_true")
    a = ap.parse_args(); outcomes(a)
    if a.cmdstats:
        cmdstats(a)
