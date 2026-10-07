"""열린 고리 반사실 재생: 같은 기록 입력에 RV4(끄기 접촉 any)·KE(현재 브리지 stand_c5kr 설정) 감지기를 돌린다.
주의: 재생 감지기의 모드는 실제 실행된 정책을 바꾸지 않는다(기록된 몸 움직임 = 그때 실제로 돈 정책의 결과). 첫 분기 이후는 참고만."""
import sys, pickle, os, glob
import numpy as np
sys.path.insert(0, ".")
import c5lib as L

def describe(D, R, kind, t_min=0.0):
    t = D["t"]; c = D["c"] > 0.5; both = c[:, 0] & c[:, 1]; anyc = c[:, 0] | c[:, 1]; n = D["n"]
    rows = []
    for (i0, i1) in L.activations(R["mode"]):
        ended = i1 < n; seg = slice(i0, i1)
        dur = (t[i1] - t[i0]) if ended else (t[-1] - t[i0] + 0.02)
        why = R["off_why"][i1] if ended else "끝까지"
        tl = R["tilt"][seg]; w = D["wxy"][seg]; wd = R["w"][seg]
        pkt, pkw = float(tl.max()), float(wd.max())
        elig = (pkt >= 5.0) or (pkw >= 1.0)
        # 조용한데 켜진 채: 원 자이로 |ω_xy|<0.3 ∧ 양발 연속 ≥ 2 s / 한발 이상
        qa = [(s, e) for (s, e) in L.runs((w < 0.3) & both[seg]) if (e - s) >= 100]
        qb = [(s, e) for (s, e) in L.runs((w < 0.3) & anyc[seg]) if (e - s) >= 100]
        def fmt(q):
            return "; ".join(f"t={t[i0+s]:.1f}+{(e-s)*0.02:.1f}s 기울기 중앙 {np.median(tl[s:e]):.2f}°(≥1.5° {100*(tl[s:e]>=1.5).mean():.0f}%)" for (s, e) in q) or "없음"
        rows.append(dict(i0=i0, i1=i1, t_on=float(t[i0]), dur=float(dur), why=why, pkt=pkt, pkw=pkw, elig=elig, qa=qa, qb=qb, qa_s=fmt(qa), qb_s=fmt(qb),
                         qa_len=sum((e - s) * 0.02 for (s, e) in qa), qb_len=sum((e - s) * 0.02 for (s, e) in qb),
                         trig=("기울기" if R["tilt"][i0] > 3.0 else "") + ("|ω|" if R["w"][i0] > 0.5 else "")))
    return rows

if __name__ == "__main__":
    FILES = ["rl_bridge_20261004_221209.jsonl", "rl_bridge_20261004_221920.jsonl", "rl_bridge_20261004_223349.jsonl"]
    for fn in FILES:
        D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb"))
        print(f"\n===== {fn} ({D['t'][-1]:.1f} s) — 기록(RV3 실제) 회복 모드 {int(D['mode'].sum())} 틱")
        for kind in ("rv3", "rv4", "ke"):
            R = L.replay(D, kind)
            rows = describe(D, R, kind)
            agree = float((R["mode"] == D["mode"]).mean())
            print(f"  [{kind}] 회복 모드 {int(R['mode'].sum())} 틱({100*R['mode'].mean():.1f} %) · 켜짐 {len(rows)} · 조기 복귀 {int(R['n_early'][-1])} · 기록 모드와 일치 틱 {100*agree:.1f} %")
            for r in rows:
                print(f"     켜짐 t={r['t_on']:6.2f} [{r['trig']}] 지속 {r['dur']:5.2f} s 끔={r['why']:<5} 최대 기울기 {r['pkt']:5.2f}° 최대 |ω|(감지) {r['pkw']:.2f} 조기복귀 자격 {'O' if r['elig'] else 'X'} | 조용∧양발≥2s: {r['qa_s']} | 조용∧한발이상≥2s: {r['qb_s']}")
