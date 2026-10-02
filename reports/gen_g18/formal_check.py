"""선별 C0+C1 통과 체크포인트의 정식 판정 (G17_CRITERIA §2–3 + 부록 §8 정식 문턱, eval seed 2–4) + 독립 seed 5–6 재확인.
사용: formal_check.py <fam(예 g20d)> <ck(예 10700)>  — 파일 g18_{fam}_{ck}_{cond}_s{seed}, 기준선 g17b_g12_*."""
import sys
sys.path.insert(0, '/home/ryu/humanoid_leg_test1/reports/gen_g17')
import g17_verdict as V
fam, ck = sys.argv[1], int(sys.argv[2])
rd = V.run_dir(fam)
for seeds, label in (((2, 3, 4), '정식 seed 2–4'), ((5, 6), '독립 seed 5–6')):
    B = V.load(V.G12, 'model_9500.pt', 'g17b_g12', seeds)
    R = V.load(rd, f'model_{ck}.pt', f'g18_{fam}_{ck}', seeds)
    if B is None or R is None:
        print(f'[{label}] 대기'); continue
    MB = {c: (V.summarize(B[c]) if c in B else V.NAN) for c in V.CONDS}
    M = {c: (V.summarize(R[c], B[c]) if c in R and c in B else V.NAN) for c in V.CONDS}
    bad = V.c0(M, MB, V.TH['formal']); ok1, k, sc, note = V.c1(M, MB)
    print(f'[{label}]\n' + f"{'낙상 %':14s} " + ' '.join(f'{c:>5s}' for c in V.SHOW))
    print(V.table('G12-9500', MB)); print(V.table(f'{fam}-{ck}', M))
    pair = ' '.join(f"{c}:{M[c]['only_c']}/{M[c]['only_b']}(p={V.mcnemar(M[c]['only_c'], M[c]['only_b']):.3g})" for c in ('sf10', 'sb10', 'b18'))
    print(f"  C0 {'통과' if not bad else '탈락 — ' + '; '.join(bad)} | C1 {'통과' if ok1 else '탈락'} ({k}/3, 점수 {sc:+.1f}%p{note}) | 짝 McNemar(후보만/기준만 낙상) {pair}")
