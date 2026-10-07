"""배포 코드 합성 러너 폐루프(--runner_loop) 집계 — X6 동등성(TRANSITION_W3KE.md §5.10) 및 왕복 대본 공용.
입력: log_picture/ground_recovery_<tag>_envs.csv + log_picture/kwloop_<tag>.npz. 대조 = 평가기 direct 팔 CSV(같은 seed·env).
사용: python3 reports/gen_w3/kw_summ.py [--arm tr_kew3] [--ref tr_direct] [--conds nd,b18,sf15] [--seeds 1,2,3] [--t_dist 16]"""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import mcnemar, wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def rows(tag):
    p = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    return {int(r["env"]): r for r in csv.DictReader(open(p))} if os.path.exists(p) else None


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="tr_kew3"); ap.add_argument("--ref", default="tr_direct")
    ap.add_argument("--conds", default="nd,b18,sf15"); ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--t_dist", type=float, default=16.0)
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    for cond in a.conds.split(","):
        A = {}; R = {}; EV = {}; KIND = {}; T = {}; REJ = {}
        for s in seeds:
            ra, rr = rows(f"{a.arm}_{cond}_d2_s{s}"), rows(f"{a.ref}_{cond}_d2_s{s}")
            pz = os.path.join(LP, f"kwloop_{a.arm}_{cond}_d2_s{s}.npz")
            if ra is None or not os.path.exists(pz):
                continue
            z = np.load(pz, allow_pickle=True); ev = json.loads(str(z["events"])); rej = json.loads(str(z["rej"]))
            for e, r in ra.items():
                if r["pre_pass"] == "1":
                    A[(s, e)] = r; EV[(s, e)] = ev[e]; REJ[(s, e)] = rej[e]
            if rr is not None:
                for e, r in rr.items():
                    if r["pre_pass"] == "1":
                        R[(s, e)] = r
            KIND[s] = z["kind"]; T[s] = z["t"]
        if not A:
            print(f"## {cond}: 자료 없음"); continue
        ks = sorted(A); n = len(ks); got = sorted({k[0] for k in ks})
        c = lambda col: sum(A[k].get(col) == "1" for k in ks)
        en_dly = []; stop2ke = []; rec_pre = 0; rec_any = 0; n_enable_rej = 0
        for k in ks:
            ev = EV[k]; t_req = next((t for t, nm in ev if nm == "enable_req"), None); t_en = next((t for t, nm in ev if nm == "enable"), None)
            if t_req is not None and t_en is not None:
                en_dly.append(t_en - t_req)
            t_st = next((t for t, nm in ev if nm == "stop_req"), None); t_ke = next((t for t, nm in ev if nm == "to_ke" and (t_st is None or t >= t_st)), None)
            if t_st is not None and t_ke is not None:
                stop2ke.append(t_ke - t_st)
            ons = [t for t, nm in ev if nm == "rec_on"]
            rec_any += bool(ons); rec_pre += any(t < (a.t_dist if cond != "nd" else 1e9) for t in ons)
            n_enable_rej += REJ[k] is not None
        fto = np.array([fl(A[k], "tr_first_to") for k in ks]); nrm = np.array([fl(A[k], "nrm_sw_t") for k in ks]); jump = np.array([fl(A[k], "cmd_jump0") for k in ks])
        print(f"## 조건 {cond} — 합성 러너 폐루프 {a.arm} (seed {got}, 사전검사 통과 n {n})")
        print(f"- 낙상: 걷기 중 {c('fell_walk')} · 래치~KE 창 {c('fell_win')} · KE 뒤 5 s {c('fell_sw5')} · 외란 뒤 {c('d1_fell') if cond != 'nd' else '—'} · 인계 뒤 전체 {c('fell')}")
        print(f"- 켜기: 요청→수락 중앙 {np.median(en_dly) if en_dly else float('nan'):.2f}s (최대 {max(en_dly) if en_dly else float('nan'):.2f}s, 거부 1 회 이상 {n_enable_rej}) · 수락→첫 이륙 중앙 {np.nanmedian(fto):.2f}s · 1 s 안 {100*float((fto <= 1.0).sum())/n:.1f} %")
        print(f"- 정지: 요청→KE 인계 중앙 {np.median(stop2ke) if stop2ke else float('nan'):.2f}s (p90 {np.percentile(stop2ke, 90) if stop2ke else float('nan'):.2f}s) · 정착 상한 도달 {c('tr_settle_to')} · KE 첫 틱 목표 점프 중앙 {np.nanmedian(jump):.2f}° · 정상 자세 복귀(KE 기준) 중앙 {np.nanmedian(nrm):.2f}s")
        print(f"- 회복 모드: 외란 전 켜진 env {rec_pre} ({100*rec_pre/n:.1f} %) · 한 번이라도 {rec_any} ({100*rec_any/n:.1f} %)")
        if R:
            kk = sorted(k for k in set(A) & set(R) if A[k].get("fell_walk") != "1" and R[k].get("fell_walk") != "1")
            for col, nm in ((("d1_fell", "외란 뒤 낙상") if cond != "nd" else ("fell_sw5", "KE 뒤 5 s 낙상")), ("fell", "인계 뒤 전체 낙상")):
                fa = sum(A[k].get(col) == "1" for k in kk); fr = sum(R[k].get(col) == "1" for k in kk)
                b = sum(1 for k in kk if R[k].get(col) == "1" and A[k].get(col) != "1"); cc = sum(1 for k in kk if A[k].get(col) == "1" and R[k].get(col) != "1")
                lo, hi = wilson(fa, len(kk))
                print(f"- 짝({a.ref} 대비, n {len(kk)}) {nm}: 합성 러너 {fa} ({100*fa/len(kk):.1f} %, {100*lo:.1f}–{100*hi:.1f}) / {a.ref} {fr} ({100*fr/len(kk):.1f} %) · 차 {100*(fa-fr)/len(kk):+.1f} %p · {a.ref}만 {b} / 합성만 {cc} (p {mcnemar(b, cc):.3g})")
        print()


if __name__ == "__main__":
    main()
