"""G14 사전 기준(G14_CRITERIA.md) 판정. 지표 정의는 G13 과 동일(g13_verdict.stats)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from g13_verdict import stats, c1, c1n, c2, pct  # noqa: E402  (import 시 g13 표도 출력됨)
base = stats('osc_g12')
fam = {f: {ck: stats(f'osc_{f}_{ck}') for ck in (10000, 10500, 11000, 11499)} for f in ('g12c', 'g13', 'g14a', 'g14b')}
print("\n== G14 가설 판정: 같은 체크포인트에서 무외란 흔들림 지속(d11+d14) ==")
for a, b in (('g14a', 'g12c'), ('g14b', 'g13')):
    wins = 0; rows = []
    for ck in (10000, 10500, 11000, 11499):
        sa, sb = fam[a][ck], fam[b][ck]
        if sa is None or sb is None: rows.append(f"{ck}: 결과 없음"); continue
        w = c2(sa) < c2(sb); wins += w
        rows.append(f"{ck}: {a} {c2(sa)} vs {b} {c2(sb)} {'✓' if w else '✗'}")
    print(f"  {a} vs {b}: " + " | ".join(rows) + f"  → {wins}/4 쌍 ({'지지' if wins >= 3 else '미지지'})")
print("\n== G14 후보 판정 (G12-9500 대비): C1 ≤ 0.7×185 = 129.5, C2 ≤ 29, C3 무외란 d11 낙상 ≤ 2.3 %·감쇠 ≥ 90.6 % ==")
for f in ('g14a', 'g14b'):
    for ck in (10000, 10500, 11000, 11499):
        s = fam[f][ck]
        if s is None: continue
        nd = s[(11, 'nd')]
        ok1 = c1(s) <= 0.7 * c1(base); ok2 = c2(s) <= c2(base)
        ok3 = pct(nd['fall'], nd['n']) <= 2.3 and pct(nd['damp'], nd['n']) >= 90.6
        print(f"  {f}-{ck}: C1 {c1(s)} {'✓' if ok1 else '✗'} | C2 {c2(s)} {'✓' if ok2 else '✗'} | C3 낙상 {pct(nd['fall'], nd['n']):.1f}% 감쇠 {pct(nd['damp'], nd['n']):.1f}% {'✓' if ok3 else '✗'}"
              + ("  ← 통과" if ok1 and ok2 and ok3 else ""))
