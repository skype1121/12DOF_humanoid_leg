"""합성 러너 폐루프 왕복 대본 집계(TRANSITION_W3KE.md X7) — kwloop_<tag>.npz 의 사건·행동 주체 시간열 + env CSV.
구간 정의(env 별 사건 기준): 걷기 k = k 번째 enable ~ 다음 stop_req / 정지 전환 k = stop_req ~ to_ke / KE 서기 k = to_ke ~ 다음 enable(또는 끝).
사용: python3 reports/gen_w3/kw_scen_summ.py --tags tr_kwsc_nd_d2_s1,tr_kwsc_nd_d2_s2 [--name 이름]"""
import argparse, collections, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from trans_summ import wilson   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--name", default="")
    a = ap.parse_args()
    n = 0; nwalk_script = None; falls = collections.Counter(); fall_t = []; en_dly = collections.defaultdict(list); never = collections.Counter(); s2k = collections.defaultdict(list)
    cap = collections.Counter(); rec_ke = collections.Counter(); rec_tot = 0; stuck_at_req = collections.Counter(); rq = 0; done_all = 0; survivors = 0; n_w3 = []; n_ke = []
    dfell = collections.Counter(); dn = 0; first_rej = collections.Counter(); kinds = np.zeros(3)
    for tag in a.tags.split(","):
        pz = os.path.join(LP, f"kwloop_{tag}.npz"); pc = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
        if not (os.path.exists(pz) and os.path.exists(pc)):
            print(f"(없음 {tag})"); continue
        z = np.load(pz, allow_pickle=True); ev = json.loads(str(z["events"])); rej = json.loads(str(z["rej"])); script = json.loads(str(z["script"])); evt = z["ev_t"]
        t = z["t"]; kind = z["kind"]; pp = z["pre_pass"].astype(bool); fell = z["fell"].astype(bool); tf = z["t_fall_post"]
        ge_arr = z["ge"].astype(bool)
        rows = {int(r["env"]): r for r in csv.DictReader(open(pc))}
        walks = [k for k, (tt, w, c) in enumerate(script) if w]; nwalk_script = len(walks)
        for e in np.nonzero(pp)[0]:
            n += 1; E = ev[e]; dead = bool(fell[e] and tf[e] >= 0); t_end = float(tf[e]) if dead else float(t[-1]) + 0.02
            for kk in (0, 1, 2):
                kinds[kk] += int((kind[:, e] == kk).sum())
            en = [tt for tt, nm in E if nm == "enable"]; rq_t = [tt for tt, nm in E if nm == "enable_req"]; st = [tt for tt, nm in E if nm == "stop_req"]; tk = [tt for tt, nm in E if nm == "to_ke"]
            n_w3.append(int(z["n_to_w3"][e])); n_ke.append(int(z["n_to_ke"][e])); rq += sum(1 for tt, nm in E if nm == "requiet")
            rons = [tt for tt, nm in E if nm == "rec_on"]; roffs = [tt for tt, nm in E if nm == "rec_off"]; rec_tot += len(rons)
            # 걷기 구간별: 대본 요청 시각 → 수락
            for j, k in enumerate(walks):
                t_req = float(evt[k][e])
                if t_req >= t_end:
                    continue                                   # 그 전에 넘어짐 — 구간 미도달
                acc = [x for x in en if x >= t_req - 1e-6 and (j + 1 >= len(walks) or x < float(evt[walks[j + 1]][e]))]
                if acc:
                    en_dly[j].append(acc[0] - t_req)
                else:
                    never[j] += 1
                # 요청 시각에 회복 모드였나(갇힘·회복 중)
                on_at = sum(1 for x in rons if x <= t_req) > sum(1 for x in roffs if x <= t_req)
                stuck_at_req[j] += bool(on_at)
            for j, s_ in enumerate(st):
                nxt = [x for x in tk if x >= s_]
                if nxt:
                    s2k[j].append(nxt[0] - s_)
                    # 상한 도달 = 래치(ge 꺼짐) 뒤 정착 상한 3.0 s 를 다 쓴 인계. (1007 정정: 종전 식 '요청 → 인계 ≥ 3.4 s' 는 요청 → 래치가 0.1–0.5 s 라 상한 도달을 못 잡았다 — 독립 검토 지적)
                    k_s = int(np.searchsorted(t, s_ - 1e-6)); k_k = int(np.searchsorted(t, nxt[0] - 1e-6)); g = ge_arr[k_s:k_k + 1, e]
                    off = np.nonzero(~g)[0]
                    if len(off):
                        cap[j] += (nxt[0] - float(t[k_s + off[0]])) >= 3.0 - 0.05
            for j, k_ in enumerate(tk):                        # KE 서기 구간 j(전환 뒤): 다음 enable 전까지 회복 켜짐 수
                nx = min([x for x in en if x > k_] + [t_end]); rec_ke[j] += any(k_ <= x < nx for x in rons)
            if dead:
                # 낙상 직전 행동 주체·구간
                ki = int(np.searchsorted(t, tf[e] - 1e-6)) - 1; kd = int(kind[max(ki, 0), e]) if ki >= 0 else -1
                last = [(tt, nm) for tt, nm in E if tt <= tf[e] and nm in ("enable", "stop_req", "to_ke")]
                ph = {"enable": "걷기", "stop_req": "정지 전환", "to_ke": "KE 서기"}.get(last[-1][1], "시작 KE 서기") if last else "시작 KE 서기"
                idx = sum(1 for tt, nm in last if nm == last[-1][1]) if last else 0
                falls[f"{ph} {idx}" if last else ph] += 1; fall_t.append(float(tf[e]))
            else:
                survivors += 1; done_all += (len(en) >= len(walks) and len(tk) >= len(st) and len(st) >= len(walks))
            r = rows.get(int(e))
            if r is not None:
                dn += 1
                for dk in ("d1_fell", "d2_fell", "d3_fell", "d4_fell"):
                    if r.get(dk) == "1":
                        dfell[dk] += 1
            if rej[e] is not None:
                first_rej[rej[e][1][:24]] += 1
    if n == 0:
        print("자료 없음"); return
    ft = sum(falls.values()); lo, hi = wilson(ft, n)
    print(f"## 왕복 대본 {a.name or a.tags} — 사전검사 통과 n {n}, 대본 걷기 구간 {nwalk_script}")
    print(f"- 낙상 전체 {ft} ({100*ft/n:.1f} %, {100*lo:.1f}–{100*hi:.1f}) · 구간별 {dict(sorted(falls.items()))}")
    print(f"- 끝까지 생존 {survivors} 중 모든 걷기·복귀 완료 {done_all} ({100*done_all/max(survivors,1):.1f} %) · env 당 KE→W3 전환 중앙 {np.median(n_w3):.0f} / W3→KE {np.median(n_ke):.0f}")
    print("- 걷기 켜기(대본 요청 → 감독기 수락): " + " · ".join(f"[{j+1}] 중앙 {np.median(en_dly[j]) if en_dly[j] else float('nan'):.2f}s p95 {np.percentile(en_dly[j], 95) if en_dly[j] else float('nan'):.2f}s 끝내 못 켬 {never[j]} 요청 때 회복 모드 {stuck_at_req[j]}" for j in range(nwalk_script or 0)))
    print("- 정지(요청 → KE 인계): " + " · ".join(f"[{j+1}] 중앙 {np.median(s2k[j]):.2f}s p95 {np.percentile(s2k[j], 95):.2f}s 상한 도달 {cap[j]}" for j in sorted(s2k)))
    print("- KE 서기 구간에서 회복 모드가 켜진 env: " + " · ".join(f"[{j+1}] {rec_ke[j]}" for j in sorted(rec_ke)) + f" · 회복 켜짐 합 {rec_tot} · 재기준 발동 합 {rq}")
    print(f"- 행동 주체 틱 비율: KE 서기 {100*kinds[0]/kinds.sum():.1f} % · KE 회복 {100*kinds[1]/kinds.sum():.1f} % · W3 {100*kinds[2]/kinds.sum():.1f} % · 첫 켜기 거부 사유 {dict(first_rej)}")
    if dfell:
        print(f"- 외란별 낙상(CSV, n {dn}): " + " · ".join(f"{k} {v} ({100*v/dn:.1f} %)" for k, v in sorted(dfell.items())))


if __name__ == "__main__":
    main()
