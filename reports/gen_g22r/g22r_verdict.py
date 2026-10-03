"""G22R 판정표 (DESIGN_CRITERIA.md 사전 기준) — eval17.sh 결과 파일만 읽는다(머리글·DR 짝 검사는 g17_verdict.one 재사용).
사용: g22r_verdict.py <후보 load_run(biped12_stage4 안 이름)> <ckpt 번호> <접두> [seeds, 기본 1]
기준선: G21a-11300 (seed 1 = 접두 g21, seed 2–4 = g18; 새 조건 sf5/sf8/sf12 는 접두 g22b)."""
import sys, re, math
sys.path.insert(0, '/home/ryu/humanoid_leg_test1/reports/gen_g17')
import g17_verdict as V
V.ARGS.update({'sf5': '--dist ramp:f:2:5:0.5:1.0', 'sf8': '--dist ramp:f:2:8:0.5:1.0', 'sf12': '--dist ramp:f:2:12:0.5:1.0'})
CONDS = ('nd', 'd3nd', 'sf5', 'sf8', 'sf10', 'sf12', 'sf15', 'sb10', 'b18', 'sl8')
G21A = '2026-10-02_21-23-08_g21a'
LP = V.LP


def txt(tag):
    try:
        return open(f'{LP}/ground_recovery_{tag}.txt', encoding='utf-8').read()
    except OSError:
        return None


def steps(s):
    m = re.search(r'이륙 (\d+)회 → 생존 이륙\(0.3 s 이상 생존\) (\d+) → 재접촉 (\d+) → 하중 지지\(0.3 s 평균 ≥30 % 체중\) (\d+)', s or '')
    st = re.search(r'\[스텝 통계\] 재접촉 발 앞축 이동 평균 ([+-][0-9.]+) cm · 표준편차 ([0-9.]+) cm · 이륙 왼/오 (\d+)/(\d+)', s or '')
    rec = re.search(r'외란 1 \[[^\]]*\]: 시작 시 생존 (\d+) → 낙상 \d+ \([0-9.]+ %\) · 스텝\(발 들기≥1\) (\d+)[^·]*· 회복 \d+[^·]*(?:·[^·]*)?(?:· 회복 시간 중앙 ([0-9.]+)s)?', s or '')
    tr = re.search(r'회복 시간 중앙 ([0-9.]+)s', s or '')
    return dict(to=int(m[1]) if m else 0, td=int(m[3]) if m else 0, load=int(m[4]) if m else 0,
                mean=float(st[1]) if st else math.nan, sd=float(st[2]) if st else math.nan,
                L=int(st[3]) if st else None, R=int(st[4]) if st else None,
                n0=int(rec[1]) if rec else 0, stepper=int(rec[2]) if rec else 0, trec=float(tr[1]) if tr else math.nan)


def row(load_run, ck, pfx_of, seeds):
    out = {}
    for c in CONDS:
        res = {}; agg = dict(to=0, td=0, load=0, n0=0, stepper=0); means = []; sds = []; L = R = 0; trs = []
        for sd in seeds:
            pfx = pfx_of(c, sd)
            r = V.one(f'{pfx}_{c}_s{sd}', load_run, f'model_{ck}.pt', sd, c)
            if r is None:
                res = None; break
            res[sd] = r
            k = steps(txt(f'{pfx}_{c}_s{sd}'))
            for key in agg: agg[key] += k[key]
            if not math.isnan(k['mean']): means.append(k['mean']); sds.append(k['sd'])
            if k['L'] is not None: L += k['L']; R += k['R']
            if not math.isnan(k['trec']): trs.append(k['trec'])
        if res is None:
            out[c] = None; continue
        sm = V.summarize(res)
        out[c] = dict(fall=sm['fall'], sus=sm['sus'], lift=sm['lift'], **agg,
                      mean=sum(means) / len(means) if means else math.nan, sd=sum(sds) / len(sds) if sds else math.nan,
                      Lr=(100.0 * L / (L + R)) if (L + R) else math.nan, trec=sum(trs) / len(trs) if trs else math.nan)
    return out


def fmt(x, nd=1):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def table(name, M):
    print(f'\n## {name}')
    print('| 조건 | 낙상 % | 흔들림 지속 % | 발 들기 env % | 스텝 env % | 이륙 | 착지 | 하중 | 착지/이륙 | 하중/착지 | 변위 평균 cm | 변위 sd | 왼 이륙 % | 회복 시간 중앙 s |')
    print('|' + '---|' * 15)
    for c in CONDS:
        m = M.get(c)
        if m is None:
            print(f'| {c} | 대기 |' + ' |' * 13); continue
        r1 = 100.0 * m['td'] / m['to'] if m['to'] else math.nan; r2 = 100.0 * m['load'] / m['td'] if m['td'] else math.nan
        sp = 100.0 * m['stepper'] / m['n0'] if m['n0'] else math.nan
        print(f"| {c} | {fmt(m['fall'])} | {fmt(m['sus'])} | {fmt(m['lift'])} | {fmt(sp)} | {m['to']} | {m['td']} | {m['load']} | {fmt(r1)} | {fmt(r2)} |"
              f" {fmt(m['mean'])} | {fmt(m['sd'])} | {fmt(m['Lr'], 0)} | {fmt(m['trec'], 2)} |")


def judge(M):
    """사전 기준(DESIGN_CRITERIA 판정 G22R). 대기 조건이 있으면 '대기'."""
    need = ('nd', 'd3nd', 'sf15', 'b18')
    if any(M.get(c) is None for c in need):
        return '대기'
    bad, warn = [], []
    if M['nd']['sus'] > 2.3: bad.append(f"무외란 흔들림 {M['nd']['sus']:.1f} > 2.3")
    if M['d3nd']['sus'] > 2.9: bad.append(f"지연3 흔들림 {M['d3nd']['sus']:.1f} > 2.9")
    f15 = M['sf15']; sp = 100.0 * f15['stepper'] / f15['n0'] if f15['n0'] else 0.0
    if f15['fall'] > 30: bad.append(f"앞15 낙상 {f15['fall']:.1f} > 30")
    if sp < 50: bad.append(f"앞15 스텝 env {sp:.1f} < 50")
    if f15['to'] and 100.0 * f15['td'] / f15['to'] < 90: bad.append(f"앞15 착지/이륙 {100.0 * f15['td'] / f15['to']:.0f} < 90")
    if f15['td'] and 100.0 * f15['load'] / f15['td'] < 90: bad.append(f"앞15 하중/착지 {100.0 * f15['load'] / f15['td']:.0f} < 90")
    b = M['b18']
    if b['fall'] > 15: bad.append(f"뒤18 낙상 {b['fall']:.1f} > 15")
    if b['td'] and 100.0 * b['load'] / b['td'] < 90: bad.append(f"뒤18 하중/착지 {100.0 * b['load'] / b['td']:.0f} < 90")
    if M['nd']['lift'] >= 5: bad.append(f"무외란 발 들기 {M['nd']['lift']:.1f} ≥ 5")
    trs = [M[c]['trec'] for c in ('sf10', 'sf15', 'sb10', 'b18') if M.get(c) and not math.isnan(M[c]['trec'])]
    if trs and sorted(trs)[len(trs) // 2] > 1.0: bad.append(f"정착 시간 중앙 {sorted(trs)[len(trs) // 2]:.2f} > 1.0 s")
    for c in ('sf15', 'b18'):
        if M.get(c) and not math.isnan(M[c]['Lr']) and (M[c]['Lr'] >= 80 or M[c]['Lr'] <= 20): warn.append(f"{c} 좌우 편향 왼 {M[c]['Lr']:.0f} %")
    return ('통과' if not bad else '탈락 — ' + '; '.join(bad)) + (' | 경고: ' + '; '.join(warn) if warn else '')


if __name__ == '__main__':
    run, ck, pfx = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seeds = tuple(int(x) for x in sys.argv[4:]) or (1,)
    base_pfx = lambda c, sd: ('g22b' if c in ('sf5', 'sf8', 'sf12') else ('g21' if sd == 1 else 'g18')) + '_g21a_11300'
    B = row(G21A, 11300, base_pfx, seeds)
    C = row(run, ck, lambda c, sd: pfx, seeds)
    print(f'# G22R 판정 — 후보 {run} model_{ck} · seeds {seeds} · 기준선 G21a-11300')
    table('G21a-11300 (기준선)', B); table(f'후보 {pfx}', C)
    print('\n판정(사전 기준):', judge(C))
    print('기준선 G21a 같은 기준:', judge(B))
