"""G18 사전 기준(G18_DESIGN_CRITERIA.md §3) 판정 — 1002 08시, G18 결과 0건 시점 작성. G17 판정 도구(g17_verdict.py)의 함수·문턱 재사용.
파일: g18_{fam}_{ck}_{cond}_s{seed} (eval17.sh), 기준선 g17b_g12_*. 없는 결과는 '대기'."""
import sys, math
sys.path.insert(0, '/home/ryu/humanoid_leg_test1/reports/gen_g17')
import g17_verdict as V
FAMS = ('g18o', 'g18j', 'g18d', 'g18q', 'g18f', 'g18f43', 'g19v', 'g19n', 'g20d', 'g20s')
CKS = V.CKS

def main():
    B1 = V.load(V.G12, 'model_9500.pt', 'g17b_g12', (1,))
    if B1 is None: print('기준선 대기'); return
    MB1 = {c: (V.summarize(B1[c]) if c in B1 else V.NAN) for c in V.CONDS}
    hdr = f"{'낙상 %':14s} " + ' '.join(f'{c:>5s}' for c in V.SHOW)
    print('[선별] seed 1 (G17 과 같은 조건·기준선)\n' + hdr); print(V.table('G12-9500', MB1))
    S = {}
    for fam in FAMS:
        rd = V.run_dir(fam)
        for ck in CKS:
            R = V.load(rd, f'model_{ck}.pt', f'g18_{fam}_{ck}', (1,)) if rd else None
            if R is None: continue
            M = {c: (V.summarize(R[c], B1[c]) if c in R and c in B1 else V.NAN) for c in V.CONDS}
            bad = V.c0(M, MB1, V.TH['screen']); ok1, k, sc, note = V.c1(M, MB1)
            S[(fam, ck)] = (M, bad, sc)
            print(V.table(f'{fam}-{ck}', M) + f"  C0{'✓' if not bad else '✗ ' + '; '.join(bad)} C1({k}/3 {sc:+.1f}%p{note})")
    print('\n[가설] (같은 체크포인트·seed 1 짝, 대조 G18o; 단일 학습 seed → 지지는 잠정)')
    def judge(fam, rule):
        w = [rule(S[(fam, ck)], S[('g18o', ck)]) for ck in CKS if (fam, ck) in S and ('g18o', ck) in S]
        return f'{sum(w)}/{len(w)} → ' + ('대기' if len(w) < 4 else ('지지(잠정)' if sum(w) >= 3 else '미지지'))
    smooth = lambda a, o: a[0]['nd']['rev'] <= 0.3 * o[0]['nd']['rev'] and a[2] >= o[2] - 10
    print('  H-j (G18j):', judge('g18j', smooth))
    print('  H-d (G18d):', judge('g18d', smooth))
    print('  H-q (G18q):', judge('g18q', lambda a, o: a[0]['b18']['fall'] <= o[0]['b18']['fall'] - 10
                                 and a[0]['nd']['lift'] <= o[0]['nd']['lift'] + 2 and a[0]['b8']['lift'] <= o[0]['b8']['lift'] + 3))
    print('  H-f (G18f):', judge('g18f', lambda a, o: a[0]['nd']['rev'] <= 0.3 * o[0]['nd']['rev'] and a[0]['b18']['fall'] <= o[0]['b18']['fall'] + 10 and a[2] >= o[2] - 10))
    w = [S[('g18f43', ck)] for ck in CKS if ('g18f43', ck) in S and ('g18o', ck) in S]
    if w: print('  H-f 복제 (G18f43 vs G18o):', judge('g18f43', lambda a, o: a[0]['nd']['rev'] <= 0.3 * o[0]['nd']['rev'] and a[0]['b18']['fall'] <= o[0]['b18']['fall'] + 10 and a[2] >= o[2] - 10))
    hv = lambda a, o: a[0]['nd']['rev'] <= 0.3 * o[0]['nd']['rev'] and a[0]['b18']['fall'] <= o[0]['b18']['fall'] + 10 and a[2] >= o[2] - 10
    def judge_vs(fam, ref):
        w = [hv(S[(fam, ck)], S[(ref, ck)]) for ck in CKS if (fam, ck) in S and (ref, ck) in S]
        return f'{sum(w)}/{len(w)} → ' + ('대기' if len(w) < 4 else ('지지(잠정)' if sum(w) >= 3 else '미지지'))
    print('  H-v (G19v vs G18f):', judge_vs('g19v', 'g18f')); print('  H-n (G19n vs G18f):', judge_vs('g19n', 'g18f'))
    h20 = lambda a, o: ((a[0]['nd']['sus'] <= o[0]['nd']['sus'] - 3) or a[0]['nd']['sus'] <= 3.9) and a[0]['d3nd']['sus'] <= 0.5 * o[0]['d3nd']['sus'] \
        and a[0]['nd']['rev'] <= 0.008 and a[0]['b18']['fall'] <= o[0]['b18']['fall'] + 10 and a[2] >= o[2] - 10
    for fam in ('g20d', 'g20s'):
        w = [h20(S[(fam, ck)], S[('g19v', ck)]) for ck in CKS if (fam, ck) in S and ('g19v', ck) in S]
        print(f'  H-20 ({fam} vs G19v): {sum(w)}/{len(w)} → ' + ('대기' if len(w) < 4 else ('지지(잠정)' if sum(w) >= 3 else '미지지')))
    print('\n[배포 자격] 선별 C0 통과 체크포인트:', [f'{f}-{c}' for (f, c), v in S.items() if not v[1]] or '없음',
          '— 통과 시 정식 seed 2–4 + 독립 seed 5–6 필요(G18_DESIGN_CRITERIA §3)')

if __name__ == '__main__':
    main()
