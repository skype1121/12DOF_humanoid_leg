"""G16 사전 기준(reports/gen_g16/G16_CRITERIA.md) 판정 — 1001 02:3x 작성, G16 결과를 보기 전에 커밋한다. 없는 결과는 '대기'.

평가 파일 이름: 기준선 osc15_g12v2_9500_d{d}_{c}_s{s} / 후보 g16_{fam}_{ck}_d{d}_{c}_s{s} / σ_ck osc15_g12c_{ck}_d3_b18_s{s}
각 결과는 머리글(인자 줄)과 DR 표본 짝(같은 seed·조건의 기준선과 pre_pass·com_x·편향 열 동일)을 검사한 뒤에만 쓴다(규칙 5)."""
import csv, math, os, re, statistics, sys, glob
LP = '/home/ryu/humanoid_leg_test1/log_picture'
LOGS = '/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4'
FAMS = ('g16o', 'g16o43', 'g16a', 'g16a43', 'g16d')
CKS = (10500, 10800, 11100, 11499)
C0_CONDS = [(1, 'nd'), (3, 'nd'), (3, 'b8'), (6, 'nd'), (11, 'nd')]
C1_CONDS = [(1, 'b18'), (3, 'b18'), (6, 'b18')]
ALL = C0_CONDS + C1_CONDS
P = lambda a, b: 100.0 * a / b if b else math.nan
def f(x): return float(x) if x not in ('', None) else math.nan

def run_dir(fam):
    hits = sorted(d for d in glob.glob(f'{LOGS}/*_{fam}') if os.path.isdir(d))
    return os.path.basename(hits[-1]) if hits else None

def header_ok(txt, load_run, ck, dl, sd, enc='v2'):
    line = next((l for l in open(txt, encoding='utf-8') if l.startswith('인자:')), '')
    want = [f'--load_run {load_run}', f'--checkpoint {ck}', f'--delay_steps {dl},{dl}', f'--seed {sd}', '--num_envs 128']
    if enc: want.append(f'--enc_obs {enc}')
    miss = [w for w in want if w not in line]
    return not miss, miss

def one(tag, load_run, ck, dl, sd, enc='v2'):
    """(행 목록, 사전검사 통과 블록 텍스트) 또는 None. 머리글 불일치는 예외(섞인 결과를 조용히 쓰지 않는다)."""
    c, t = f'{LP}/ground_recovery_{tag}_envs.csv', f'{LP}/ground_recovery_{tag}.txt'
    if not (os.path.exists(c) and os.path.exists(t)): return None
    ok, miss = header_ok(t, load_run, ck, dl, sd, enc)
    if not ok: raise SystemExit(f'머리글 불일치 {tag}: {miss}')
    s = open(t, encoding='utf-8').read()
    blk = s.split('[사전 검사 통과]')[1].split('[사전 검사 실패')[0] if '[사전 검사 통과]' in s else ''
    m1 = re.search(r'토크 포화\(어느 관절이든 ≥95 %\) ([0-9.]+) % 틱', blk)
    m2 = re.search(r'명령 부호반전 비율 ([0-9.]+)/틱', blk)
    return list(csv.DictReader(open(c))), (float(m1[1]) if m1 else math.nan), (float(m2[1]) if m2 else math.nan)

def summarize(res_by_seed, base_by_seed=None):
    """seed 별 (rows, sat, rev) → 합산 지표. base_by_seed 가 있으면 짝 검사 + 짝 차이(후보만 낙상 − 기준만 낙상)."""
    n = fall = gf = sus = damp = lift = 0; sats = []; revs = []; only_c = only_b = 0
    for sd, (rows, sat, rev) in res_by_seed.items():
        sats.append(sat); revs.append(rev)
        brows = base_by_seed[sd][0] if base_by_seed else None
        if brows is not None:
            key = lambda r: (r['pre_pass'], r['com_x'], r['bias_left_knee_joint'], r['bias_right_ankle_f_joint'])
            if [key(r) for r in rows] != [key(r) for r in brows]:
                raise SystemExit('DR 표본 짝 불일치 (seed %s) — 같은 평가 env·seed 가 아님' % sd)
        for i, r in enumerate(rows):
            if r['pre_pass'] != '1': continue
            n += 1; pp = [f(r[f'p2p_{k}']) for k in range(14)]
            lift += f(r['lifts_total']) > 0
            if brows is not None:
                fb = brows[i]['fell'] == '1'; fc = r['fell'] == '1'
                only_c += fc and not fb; only_b += fb and not fc
            if r['fell'] == '1':
                fall += 1
                if f(r['t_fall_rel']) > 3: gf += 1
                continue
            if max(pp[6:14]) >= 4 and max(pp[6:14]) >= 2 * max(pp[1:4]): gf += 1
            if max(pp[10:14]) <= 2: damp += 1
            else: sus += 1
    return dict(n=n, fall=P(fall, n), gf=P(gf, n), sus=P(sus, n), damp=P(damp, n), lift=P(lift, n),
                sat=sum(sats) / len(sats), rev=sum(revs) / len(revs), only_c=only_c, only_b=only_b)

G12 = '2026-09-29_14-32-50_g12'
def base(dl, ds, seeds):
    out = {}
    for sd in seeds:
        r = one(f'osc15_g12v2_9500_d{dl}_{ds}_s{sd}', G12, 'model_9500.pt', dl, sd)
        if r is None: return None
        out[sd] = r
    return out

def cand(fam, ck, dl, ds, seeds):
    rd = run_dir(fam)
    if rd is None: return None
    out = {}
    for sd in seeds:
        r = one(f'g16_{fam}_{ck}_d{dl}_{ds}_s{sd}', rd, f'model_{ck}.pt', dl, sd)
        if r is None: return None
        out[sd] = r
    return out

# 사전 기준 문턱 (G16_CRITERIA.md 와 같은 값) — 선별(seed 1, n≈103) / 정식(seed 1–3, n≈311)
TH = {
    'screen': dict(nd13_fall=2, nd13_sus=3, nd13_damp=3, b8_gf=3, d6_fall=3, d6_sus=4, d11_fall=4, d11_sus=6,
                   sat_nd=0.3, rev_nd=0.005, sat_b18=2.0, lift=3),
    'formal': dict(nd13_fall=1, nd13_sus=1.5, nd13_damp=2, b8_gf=2, d6_fall=2, d6_sus=3, d11_fall=3, d11_sus=5,
                   sat_nd=0.2, rev_nd=0.004, sat_b18=1.5, lift=2),
}
def c0(mc, mb, th, skip_d11=False):
    """mc/mb: {(dl,ds): summary}. 실패 항목 목록(빈 목록 = 통과). skip_d11: G16d 보정(§7) — d11 항목 미적용."""
    bad = []
    for dl in (1, 3):
        c, b = mc[(dl, 'nd')], mb[(dl, 'nd')]
        if c['fall'] > b['fall'] + th['nd13_fall']: bad.append(f'd{dl} 무외란 낙상 {c["fall"]:.1f}>{b["fall"]:.1f}+{th["nd13_fall"]}')
        if c['sus'] > b['sus'] + th['nd13_sus']: bad.append(f'd{dl} 흔들림지속 {c["sus"]:.1f}>{b["sus"]:.1f}+{th["nd13_sus"]}')
        if c['damp'] < b['damp'] - th['nd13_damp']: bad.append(f'd{dl} 감쇠 {c["damp"]:.1f}<{b["damp"]:.1f}-{th["nd13_damp"]}')
    c, b = mc[(3, 'b8')], mb[(3, 'b8')]
    if c['gf'] > b['gf'] + th['b8_gf']: bad.append(f'd3 뒤8N 성장·낙상 {c["gf"]:.1f}>{b["gf"]:.1f}+{th["b8_gf"]}')
    for dl, kf, ks in ((6, 'd6_fall', 'd6_sus'), (11, 'd11_fall', 'd11_sus')):
        if skip_d11 and dl == 11:
            continue
        c, b = mc[(dl, 'nd')], mb[(dl, 'nd')]
        if c['fall'] > b['fall'] + th[kf]: bad.append(f'd{dl} 무외란 낙상 {c["fall"]:.1f}>{b["fall"]:.1f}+{th[kf]}')
        if c['sus'] > b['sus'] + th[ks]: bad.append(f'd{dl} 흔들림지속 {c["sus"]:.1f}>{b["sus"]:.1f}+{th[ks]}')
    nds = [(dl, 'nd') for dl in (1, 3, 6, 11)]
    sc = sum(mc[k]['sat'] for k in nds) / 4; sb = sum(mb[k]['sat'] for k in nds) / 4
    rc = sum(mc[k]['rev'] for k in nds) / 4; rb = sum(mb[k]['rev'] for k in nds) / 4
    lc = sum(mc[k]['lift'] for k in nds) / 4; lb = sum(mb[k]['lift'] for k in nds) / 4
    if sc > sb + th['sat_nd']: bad.append(f'무외란 토크포화 {sc:.2f}>{sb:.2f}+{th["sat_nd"]}')
    if rc > rb + th['rev_nd']: bad.append(f'무외란 부호반전 {rc:.3f}>{rb:.3f}+{th["rev_nd"]}')
    if lc > lb + th['lift']: bad.append(f'무외란 발들기 {lc:.1f}>{lb:.1f}+{th["lift"]}')
    s18c = sum(mc[(d, 'b18')]['sat'] for d in (1, 3, 6)) / 3; s18b = sum(mb[(d, 'b18')]['sat'] for d in (1, 3, 6)) / 3
    if s18c > s18b + th['sat_b18']: bad.append(f'뒤18N 토크포화 {s18c:.2f}>{s18b:.2f}+{th["sat_b18"]}')
    return bad

def b18mean(m): return sum(m[(d, 'b18')]['fall'] for d in (1, 3, 6)) / 3

def main():
    seeds1 = (1,)
    B1 = {k: base(*k, seeds1) for k in ALL}
    if any(v is None for v in B1.values()):
        print('기준선(G12-9500 v2) 결과 대기:', [k for k, v in B1.items() if v is None]); return
    MB1 = {k: summarize(v) for k, v in B1.items()}
    # σ_ck: G12c d3 b18 (legacy 관측, seed 1–3) 4 체크포인트 낙상률의 표준편차
    g12c = []
    for ck in CKS:
        rows = []
        for sd in (1, 2, 3):
            r = one(f'osc15_g12c_{ck}_d3_b18_s{sd}', '2026-09-30_12-05-46_g12c', f'model_{ck}.pt', 3, sd, enc=None)
            if r is None: rows = None; break
            rows.append(r)
        if rows: g12c.append(summarize({i: r for i, r in enumerate(rows)})['fall'])
    sig = statistics.stdev(g12c) if len(g12c) == 4 else math.nan
    thr = max(15.0, 2 * sig) if sig == sig else 15.0
    print(f'σ_ck(G12c d3 뒤18N 낙상, 4 체크포인트) = {sig:.1f} %p → C1 문턱 = {thr:.1f} %p' + ('' if sig == sig else ' (σ_ck 대기 — 15 %p 임시)'))
    cols = ['d1nd낙', 'd3nd낙', 'd3nd지', 'd3b8', 'd6nd낙', 'd6nd지', 'd11지', 'd1b18', 'd3b18', 'd6b18', '포화', '반전']
    def row(m):
        return [m[(1, 'nd')]['fall'], m[(3, 'nd')]['fall'], m[(3, 'nd')]['sus'], m[(3, 'b8')]['gf'], m[(6, 'nd')]['fall'],
                m[(6, 'nd')]['sus'], m[(11, 'nd')]['sus'], m[(1, 'b18')]['fall'], m[(3, 'b18')]['fall'], m[(6, 'b18')]['fall'],
                sum(m[(d, 'nd')]['sat'] for d in (1, 3, 6, 11)) / 4, sum(m[(d, 'nd')]['rev'] for d in (1, 3, 6, 11)) / 4]
    print('\n[선별] eval seed 1, 체크포인트 4개 (기준선 = G12-9500 v2 관측, 같은 DR 표본 짝)')
    print(f"{'후보':16s} " + ' '.join(f'{c:>6s}' for c in cols) + '  선별C0')
    print(f"{'G12-9500(v2)':16s} " + ' '.join(f'{x:6.1f}' if i < 10 else f'{x:6.3f}' for i, x in enumerate(row(MB1))))
    screen = {}
    for fam in FAMS:
        for ck in CKS:
            R = {k: cand(fam, ck, *k, seeds1) for k in ALL}
            if any(v is None for v in R.values()): continue
            M = {k: summarize(R[k], B1[k]) for k in ALL}
            bad = c0(M, MB1, TH['screen'], skip_d11=(fam == 'g16d')); screen[(fam, ck)] = (M, bad)
            print(f"{fam + '-' + str(ck):16s} " + ' '.join(f'{x:6.1f}' if i < 10 else f'{x:6.3f}' for i, x in enumerate(row(M)))
                  + ('  ✓' if not bad else '  ✗ ' + '; '.join(bad)))
    # 체크포인트 선택(사전 규칙): 선별 C0 통과 중 뒤 18 N 평균(d1·d3·d6) 최소, 5 %p 이내 동률이면 d6+d11 흔들림 지속 합이 작은 것, 다음 토크포화
    print('\n[체크포인트 선택] (선별 C0 통과 중 뒤18N 평균 최소 → 5 %p 동률이면 d6+d11 흔들림 → 토크포화)')
    pick = {}
    for fam in FAMS:
        ok = [(ck, M) for (fm, ck), (M, bad) in screen.items() if fm == fam and not bad]
        done = sum(1 for (fm, _ck) in screen if fm == fam)
        if done < 4: print(f'  {fam}: 대기 ({done}/4)'); continue
        if not ok: print(f'  {fam}: 선별 C0 통과 체크포인트 없음 → 탈락'); continue
        best = min(b18mean(M) for _, M in ok)
        tied = [(ck, M) for ck, M in ok if b18mean(M) <= best + 5]
        tied.sort(key=lambda x: (x[1][(6, 'nd')]['sus'] + x[1][(11, 'nd')]['sus'], sum(x[1][(d, 'nd')]['sat'] for d in (1, 3, 6, 11))))
        pick[fam] = tied[0][0]
        print(f'  {fam}: {pick[fam]} (뒤18N 평균 {b18mean(tied[0][1]):.1f} %)')
    # 정식 C0 (선택 체크포인트, seed 1–3)
    print('\n[정식 C0] 선택 체크포인트 · eval seed 1–3 (기준선 seed 1–3)')
    B3 = {k: base(*k, (1, 2, 3)) for k in ALL}
    formal = {}
    if any(v is None for v in B3.values()):
        print('  기준선 seed 1–3 대기:', [k for k, v in B3.items() if v is None])
    else:
        MB3 = {k: summarize(v) for k, v in B3.items()}
        for fam, ck in pick.items():
            R = {k: cand(fam, ck, *k, (1, 2, 3)) for k in ALL}
            if any(v is None for v in R.values()): print(f'  {fam}-{ck}: 대기'); continue
            M = {k: summarize(R[k], B3[k]) for k in ALL}
            bad = c0(M, MB3, TH['formal'], skip_d11=(fam == 'g16d')); formal[fam] = (ck, M, bad)
            pair = ' '.join(f"{k[0]}{k[1]}:{M[k]['only_c']}/{M[k]['only_b']}" for k in C1_CONDS)
            print(f"  {fam}-{ck}: {'통과' if not bad else '탈락 — ' + '; '.join(bad)} | 뒤18N 낙상 d1/d3/d6 "
                  f"{M[(1,'b18')]['fall']:.1f}/{M[(3,'b18')]['fall']:.1f}/{M[(6,'b18')]['fall']:.1f} (기준 {MB3[(1,'b18')]['fall']:.1f}/{MB3[(3,'b18')]['fall']:.1f}/{MB3[(6,'b18')]['fall']:.1f}) 짝(후보만/기준만 낙상) {pair}")
    # 가설 판정
    print('\n[가설 H-o] 관측 v2 학습(G16o)의 비퇴보: seed 42·43 각각 선택 체크포인트가 정식 C0 통과')
    for fam in ('g16o', 'g16o43'):
        v = formal.get(fam)
        print(f'  {fam}: ' + ('대기' if v is None else ('통과' if not v[2] else '탈락')))
    print(f'\n[가설 H-a] require_both 제거(G16a) → 뒤 18 N 낙상 감소 (G16o 대비, 체크포인트 4개 × seed 1, d1·d3·d6 평균)')
    def fam_b18(fam):
        v = [b18mean(screen[(fam, ck)][0]) for ck in CKS if (fam, ck) in screen]
        return v if len(v) == 4 else None
    o42, o43, a42, a43 = (fam_b18(x) for x in ('g16o', 'g16o43', 'g16a', 'g16a43'))
    if o42 is None or a42 is None:
        print('  대기')
    else:
        ctrl = o42 + (o43 or []); ctrl_mean = sum(ctrl) / len(ctrl)
        spread = abs(sum(o42) / 4 - sum(o43) / 4) if o43 else math.nan
        for fam, a, o in (('g16a', a42, o42), ('g16a43', a43, o43)):
            if a is None or o is None: print(f'  {fam}: 대기'); continue
            am = sum(a) / 4; diff = ctrl_mean - am
            wins = sum(x < y for x, y in zip(a, o))
            worst_ok = max(a) <= ctrl_mean
            anchor = am <= b18mean(MB1) + 2
            spread_ok = (diff > spread) if spread == spread else None
            ok = diff >= thr and wins >= 3 and worst_ok and anchor and (spread_ok is not False)
            print(f'  {fam}: 평균 {am:.1f} vs 대조 {ctrl_mean:.1f} (차 {diff:+.1f}, 문턱 {thr:.1f}) · 체크포인트 승 {wins}/4 · 최악≤대조평균 {worst_ok} · '
                  f'기준선 닻(≤{b18mean(MB1)+2:.1f}) {anchor} · seed 폭({spread:.1f}) 초과 {spread_ok} → {"지지" if ok else "미지지"}'
                  + ('' if o43 else ' (잠정: 대조 seed 1개)'))
    print('\n[G16d] 판정은 아침 실측 지연 뒤(명령 경로 p95 ≤ 5틱일 때만 자격). 여기서는 선별 결과만 표시.')

if __name__ == '__main__':
    main()
