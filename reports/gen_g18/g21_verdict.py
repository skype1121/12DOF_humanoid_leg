"""G21 판정(G21_DESIGN_CRITERIA.md, 1002 15:2x 사전 기준) — 선별 seed 1, 체크포인트 11300·11600.
보존 기준: b18 생존 ≥88 % · 재접촉/이륙 ≥95 % · 하중/재접촉 ≥95 % · nd 발 들기 <5 %. 단계 비교는 대조 대비 흔들림(nd·d3nd)."""
import sys, re, math
sys.path.insert(0, '/home/ryu/humanoid_leg_test1/reports/gen_g17')
import g17_verdict as V
LP = V.LP
FAMS = [('g21o', None), ('g21a1', 'g21o'), ('g21a2', 'g21o'), ('g21a3', 'g21o'), ('g21a', 'g21o'), ('g21b', 'g21a'),
        ('g21c1', 'g21b'), ('g21c2', 'g21c1'), ('g21d', 'g21c2'), ('g21a3s43', 'g21o')]
CKS = (11300, 11600)

def chain(tag):
    s = open(f'{LP}/ground_recovery_{tag}.txt', encoding='utf-8').read()
    m = re.search(r'이륙 (\d+)회 → 생존 이륙\(0.3 s 이상 생존\) (\d+) → 재접촉 (\d+) → 하중 지지[^)]*\) (\d+)', s)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0, 0)

def keep(M, tag_b18):
    to, ok, td, ld = chain(tag_b18)
    surv = 100 - M['b18']['fall']
    r_td = 100 * td / to if to else math.nan; r_ld = 100 * ld / td if td else math.nan
    bad = []
    if not surv >= 88: bad.append(f'b18 생존 {surv:.1f}<88')
    if not r_td >= 95: bad.append(f'재접촉/이륙 {r_td:.0f}<95')
    if not r_ld >= 95: bad.append(f'하중/재접촉 {r_ld:.0f}<95')
    if not M['nd']['lift'] < 5: bad.append(f"nd 발들기 {M['nd']['lift']:.1f}≥5")
    return bad, f'b18 생존 {surv:.1f} · 사슬 {to}→{td}→{ld}'

B1 = V.load(V.G12, 'model_9500.pt', 'g17b_g12', (1,))
MB1 = {c: (V.summarize(B1[c]) if c in B1 else V.NAN) for c in V.CONDS}
print(f"{'낙상 %':16s} " + ' '.join(f'{c:>5s}' for c in V.SHOW)); print(V.table('G12-9500', MB1))
S = {}
rows = [('g19v-10700(기준)', V.run_dir('g19v'), 10700, 'g18_g19v_10700')]
for fam, _ in FAMS:
    for ck in CKS:
        rows.append((f'{fam}-{ck}', V.run_dir(fam), ck, f'g21_{fam}_{ck}'))
for name, rd, ck, pfx in rows:
    if rd is None: continue
    R = V.load(rd, f'model_{ck}.pt', pfx, (1,))
    if R is None: continue
    M = {c: (V.summarize(R[c], B1[c]) if c in R and c in B1 else V.NAN) for c in V.CONDS}
    kb, kd = keep(M, f'{pfx}_b18_s1'); c0 = V.c0(M, MB1, V.TH['screen']); ok1, k, sc, note = V.c1(M, MB1)
    S[name] = M
    print(V.table(name, M) + f"\n    보존 {'✓' if not kb else '✗ ' + '; '.join(kb)} ({kd}) | C0 {'✓' if not c0 else '✗ ' + '; '.join(c0)} | C1 {'✓' if ok1 else '✗'} ({k}/3 {sc:+.1f}%p{note})")
print('\n[단계 비교] 흔들림 지속 nd / d3nd (%), 같은 체크포인트')
for fam, ref in FAMS:
    if ref is None: continue
    for ck in CKS:
        a, b = S.get(f'{fam}-{ck}'), S.get(f'{ref}-{ck}')
        if a and b:
            print(f"  {fam}-{ck} vs {ref}: nd {a['nd']['sus']:.1f} vs {b['nd']['sus']:.1f} · d3nd {a['d3nd']['sus']:.1f} vs {b['d3nd']['sus']:.1f} · 부호반전 {a['nd']['rev']:.3f} vs {b['nd']['rev']:.3f}")
