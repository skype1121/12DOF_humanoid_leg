"""보조(열린 고리): C5 가 아닌 실물 서기 로그(0928–1004, 서기 정책 단독)에 KE 감지기를 재생 — '밀린 뒤 조용히 선 자세가 감지기 기준과 1.5° 넘게 어긋나 꺼지지 않는' 구간이 있는가.
주의: 이 로그들에서 실제로 돈 것은 서기 정책뿐이다(회복 정책·전환 없음). 재생 모드는 몸 움직임에 아무 영향도 주지 않았다 → KE 폐루프 거동의 증거가 아니라
'실물 서기에서 자세가 기준 대비 얼마나 움직이는가'의 측정이다."""
import sys, pickle, os, glob
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
from replay_cf import describe
FILES = [("reports/real_0928/rl_bridge_20260928_164644.jsonl", None), ("reports/real_0929/rl_bridge_20260929_140230.jsonl", None),
         ("reports/real_0929/rl_bridge_20260929_210129.jsonl", None), ("reports/real_1001/rl_bridge_20261001_184814.jsonl", None),
         ("reports/real_1001/rl_bridge_20261001_185555.jsonl", None), ("reports/real_1001/rl_bridge_20261001_193901.jsonl", None),
         ("reports/real_1002/rl_bridge_20261002_143511.jsonl", None), ("reports/real_1003/rl_bridge_20261003_174906.jsonl", None),
         ("reports/real_1004/rl_bridge_20261004_220509.jsonl", None)]
tot = dict(acts=0, light=0, quiet2=0)
for rel, _ in FILES:
    p = os.path.join(L.REPO, rel); pk = os.path.join(L.S, os.path.basename(p) + ".pkl")
    if os.path.exists(pk):
        D = pickle.load(open(pk, "rb"))
    else:
        D = L.load(p); pickle.dump(D, open(pk, "wb"))
    t = D["t"]; c = D["c"] > 0.5; both = c[:, 0] & c[:, 1]; none = ~(c[:, 0] | c[:, 1])
    # 회수(양발 무접촉 1 s 이상 지속) 시작 시각 — 그 뒤는 서기가 아니므로 표시
    lift = [(s, e) for (s, e) in L.runs(none) if (e - s) >= 50]
    t_lift = t[lift[0][0]] if lift else None
    R = L.replay(D, "ke"); rows = describe(D, R, "ke")
    bias = np.unique(D["bias"], axis=0).tolist()
    print(f"\n===== {os.path.basename(p)} ck={D['ck']} {t[-1]:.1f} s · bias {bias} · guard {D['gm']} · 양발 무접촉 ≥1 s 첫 시각 {None if t_lift is None else round(float(t_lift),1)} · KE 재생: 켜짐 {len(rows)} · 회복 모드 {100*R['mode'].mean():.1f} % · 조기 복귀 {int(R['n_early'][-1])}")
    for r in rows:
        after_lift = (t_lift is not None and r["t_on"] + r["dur"] > t_lift)
        flag = " [회수/무접촉 구간 포함]" if after_lift else ""
        seg = slice(r["i0"], r["i1"]); tl = R["tilt"][seg]; w = D["wxy"][seg]; cb = both[seg]
        q_t, q_w = tl < 1.5, w < 0.3
        blk = f"기울기만 {int((~q_t & q_w & cb).sum())} · |ω| 포함 {int((~q_w).sum())} · 양발 아님만 {int((q_t & q_w & ~cb).sum())} · 조용 {int((q_t & q_w & cb).sum())}"
        print(f"   켜짐 t={r['t_on']:6.2f} [{r['trig']:<6}] 지속 {r['dur']:5.2f} s 끔={r['why']:<5} 최대 기울기 {r['pkt']:5.2f}° |ω| {r['pkw']:.2f} 조기자격 {'O' if r['elig'] else 'X'} | 틱: {blk} | 조용∧양발≥2s: {r['qa_s']}{flag}")
        if not after_lift:
            tot["acts"] += 1; tot["light"] += (not r["elig"]); tot["quiet2"] += (len(r["qa"]) > 0)
print("\n합계(회수 구간 제외):", tot)
