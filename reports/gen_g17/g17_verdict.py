"""G17 사전 기준(G17_CRITERIA.md) 판정 — 1002 02시, G17 평가 결과를 보기 전에 작성. 없는 결과는 '대기'.
파일: 기준선 g17b_g12_{c}_s{s} / 후보 g17_{fam}_{ck}_{c}_s{s} (eval17.sh). 머리글·DR 짝 검사 뒤에만 사용."""
import csv, math, os, re, glob
LP = '/home/ryu/humanoid_leg_test1/log_picture'
LOGS = '/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4'
FAMS = ('g17o', 'g17o43', 'g17r', 'g17r43', 'g17p', 'g17rp')
CKS = (10100, 10400, 10700, 11000)
CONDS = ('nd', 'ndg2', 'ndg3', 'd0nd', 'd3nd', 'f8', 'b8', 'sl8', 'sf10', 'sb10', 'sf15', 'b18', 'sf10g2', 'd5nd', 'd5b8')
# 조건 인자(eval17.sh 와 같은 표) — 머리글 대조용 (부록 §8)
ARGS = {'nd': '', 'ndg2': '--gyro_delay 2', 'ndg3': '--gyro_delay 3', 'd0nd': '', 'd3nd': '', 'd5nd': '',
        'f8': '--dist push:f:2:8:0.3', 'b8': '--dist push:b:2:8:0.3', 'sl8': '--dist ramp:l:2:8:0.5:1.0',
        'sf10': '--dist ramp:f:2:10:0.5:1.0', 'sb10': '--dist ramp:b:2:10:0.5:1.0', 'sf15': '--dist ramp:f:2:15:0.5:1.0',
        'b18': '--dist push:b:2:18:0.3', 'sf10g2': '--gyro_delay 2 --dist ramp:f:2:10:0.5:1.0', 'd5b8': '--dist push:b:2:8:0.3'}
DL = {'d0nd': 0, 'd3nd': 3, 'd5nd': 5, 'd5b8': 5}
G12C = '2026-09-30_12-05-46_g12c'
G12 = '2026-09-29_14-32-50_g12'
P = lambda a, b: 100.0 * a / b if b else math.nan
def f(x): return float(x) if x not in ('', None) else math.nan

def run_dir(fam):
    hits = sorted(d for d in glob.glob(f'{LOGS}/*_{fam}') if os.path.isdir(d))
    return os.path.basename(hits[-1]) if hits else None

def one(tag, load_run, ck, sd, cond):
    c, t = f'{LP}/ground_recovery_{tag}_envs.csv', f'{LP}/ground_recovery_{tag}.txt'
    if not (os.path.exists(c) and os.path.exists(t)): return None
    s = open(t, encoding='utf-8').read()
    line = next((l for l in s.splitlines() if l.startswith('인자:')), '')
    want = [f'--load_run {load_run}', f'--checkpoint {ck}', f'--delay_steps {DL.get(cond, 1)},{DL.get(cond, 1)}', f'--seed {sd}',
            '--num_envs 128', '--enc_obs v2', '--contact_real', '--after_s 8', '--task Biped12-HangStand-G11a-v0'] + ([ARGS[cond]] if ARGS[cond] else [])
    if 'gyro_delay' not in ARGS[cond] and '--gyro_delay' in line: want.append('(gyro_delay 없어야 함)')
    if '--dist' not in ARGS[cond] and '--dist' in line: want.append('(dist 없어야 함)')
    miss = [w for w in want if w not in line]
    if miss: raise SystemExit(f'머리글 불일치 {tag}: {miss}')
    if '[사전 검사 통과]' not in s: return None
    blk = s.split('[사전 검사 통과]')[1].split('[사전 검사 실패')[0]
    m1 = re.search(r'토크 포화\(어느 관절이든 ≥95 %\) ([0-9.]+) % 틱', blk)
    m2 = re.search(r'명령 부호반전 비율 ([0-9.]+)/틱', blk)
    m3 = re.search(r'이륙 (\d+)회 → 생존 이륙\(0.3 s 이상 생존\) (\d+)', s)
    return (list(csv.DictReader(open(c))), float(m1[1]) if m1 else math.nan, float(m2[1]) if m2 else math.nan,
            (int(m3[1]), int(m3[2])) if m3 else (0, 0))

def summarize(res, base=None):
    n = fall = gf = sus = lift = 0; sats = []; revs = []; to = [0, 0]; oc = ob = 0
    for sd, (rows, sat, rev, tk) in res.items():
        sats.append(sat); revs.append(rev); to[0] += tk[0]; to[1] += tk[1]
        brows = base[sd][0] if base else None
        if brows is not None:
            key = lambda r: (r['pre_pass'], r['com_x'], r['bias_left_knee_joint'], r['bias_right_ankle_f_joint'])
            if [key(r) for r in rows] != [key(r) for r in brows]:
                raise SystemExit(f'DR 표본 짝 불일치 (seed {sd})')
        for i, r in enumerate(rows):
            if r['pre_pass'] != '1': continue
            n += 1; pp = [f(r[f'p2p_{k}']) for k in range(8)]   # --after_s 8 (부록 §8)
            lift += f(r['lifts_total']) > 0
            if brows is not None:
                fb, fc = brows[i]['fell'] == '1', r['fell'] == '1'; oc += fc and not fb; ob += fb and not fc
            if r['fell'] == '1':
                fall += 1; gf += f(r['t_fall_rel']) > 3; continue
            if max(pp[4:8]) >= 4 and max(pp[4:8]) >= 2 * max(pp[1:4]): gf += 1
            if max(pp[5:8]) > 2: sus += 1
    return dict(n=n, fall=P(fall, n), gf=P(gf, n), sus=P(sus, n), lift=P(lift, n), sat=sum(sats) / len(sats),
                rev=sum(revs) / len(revs), takeoff=to[0], takeoff_ok=to[1], only_c=oc, only_b=ob)

def load(load_run, ck, pfx, seeds):
    out = {}
    for c in CONDS:
        d = {}
        for sd in seeds:
            r = one(f'{pfx}_{c}_s{sd}', load_run, ck, sd, c)
            if r is None: return None
            d[sd] = r
        out[c] = d
    return out

TH = {'screen': dict(nd_fall=2, nd_sus=3, nd_lift=3, rev=0.008, sat=0.3, ndg2=3, f8=3, b8=4, sl8=5, dl_fall=2, dl_sus=3),
      'formal': dict(nd_fall=1, nd_sus=2, nd_lift=2, rev=0.006, sat=0.2, ndg2=2, f8=2, b8=3, sl8=5, dl_fall=1, dl_sus=2)}

def c0(m, b, th):
    bad = []
    chk = lambda cond, val, lim, name: bad.append(f'{name} {val:.1f}>{lim:.1f}') if val > lim else None
    chk('nd', m['nd']['fall'], b['nd']['fall'] + th['nd_fall'], 'nd 낙상')
    chk('nd', m['nd']['sus'], b['nd']['sus'] + th['nd_sus'], 'nd 흔들림지속')
    chk('nd', m['nd']['lift'], b['nd']['lift'] + th['nd_lift'], 'nd 발들기')
    if m['nd']['rev'] > th['rev']: bad.append(f"nd 부호반전 {m['nd']['rev']:.3f}>{th['rev']}")
    if m['nd']['sat'] > b['nd']['sat'] + th['sat']: bad.append(f"nd 토크포화 {m['nd']['sat']:.2f}")
    chk('ndg2', m['ndg2']['fall'], b['ndg2']['fall'] + th['ndg2'], 'ndg2 낙상')
    chk('ndg3', m['ndg3']['fall'], b['ndg3']['fall'] + th['ndg2'], 'ndg3 낙상')
    for c in ('d0nd', 'd3nd'):
        chk(c, m[c]['fall'], b[c]['fall'] + th['dl_fall'], f'{c} 낙상')
        chk(c, m[c]['sus'], b[c]['sus'] + th['dl_sus'], f'{c} 흔들림지속')
    chk('f8', m['f8']['fall'], b['f8']['fall'] + th['f8'], 'f8 낙상')
    chk('b8', m['b8']['gf'], b['b8']['gf'] + th['b8'], 'b8 성장·낙상')
    chk('sl8', m['sl8']['fall'], b['sl8']['fall'] + th['sl8'], 'sl8 낙상')
    return bad

def c1(m, b):
    """(통과 여부, 만족 수, 점수, 메모). 점수 = sf10·sb10·b18 낙상 상대 감소 평균."""
    s1 = m['sf10']['fall'] <= 0.5 * b['sf10']['fall']
    s2 = (m['sb10']['fall'] <= b['sb10']['fall'] + 2) if b['sb10']['fall'] < 10 else (m['sb10']['fall'] <= 0.5 * b['sb10']['fall'])
    s3 = m['b18']['fall'] <= b['b18']['fall'] - 15
    worse = [c for c in ('sf10', 'sb10', 'b18') if m[c]['fall'] > b[c]['fall'] + 5]
    absd = [b[c]['fall'] - m[c]['fall'] for c in ('sf10', 'sb10', 'b18')]   # 절대 %p 감소 (부록 §8)
    k = s1 + s2 + s3
    return (k >= 2 and not worse), k, sum(absd) / 3, (f'악화 {worse}' if worse else '')

def mcnemar(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

SHOW = ('nd', 'ndg2', 'ndg3', 'd0nd', 'd3nd', 'f8', 'b8', 'sl8', 'sf10', 'sb10', 'sf15', 'b18', 'sf10g2', 'd5nd', 'd5b8')
def table(name, m):
    return (f"{name:14s} " + ' '.join(f"{m[c]['fall']:5.1f}" for c in SHOW) +
            f"  sus{m['nd']['sus']:4.1f} lift{m['nd']['lift']:4.1f} rev{m['nd']['rev']:.3f} 이륙{m['b18']['takeoff']}/{m['b18']['takeoff_ok']}")

def main():
    hdr = f"{'낙상 %':14s} " + ' '.join(f'{c:>5s}' for c in SHOW)
    B1 = load(G12, 'model_9500.pt', 'g17b_g12', (1,))
    if B1 is None: print('기준선 seed 1 대기'); return
    MB1 = {c: summarize(v) for c, v in B1.items()}
    print('[선별] seed 1\n' + hdr); print(table('G12-9500', MB1))
    for ck in CKS:   # 이어 학습 대조(판정 미사용, 부록 §8)
        R = load(G12C, f'model_{ck}.pt', f'g17c_g12c_{ck}', (1,))
        if R is not None: print(table(f'[대조]G12c-{ck}', {c: summarize(R[c], B1[c]) for c in CONDS}))
    screen = {}
    for fam in FAMS:
        rd = run_dir(fam)
        for ck in CKS:
            R = load(rd, f'model_{ck}.pt', f'g17_{fam}_{ck}', (1,)) if rd else None
            if R is None: continue
            M = {c: summarize(R[c], B1[c]) for c in CONDS}
            bad = c0(M, MB1, TH['screen']); ok1, k, sc, note = c1(M, MB1)
            screen[(fam, ck)] = (M, bad, sc)
            print(table(f'{fam}-{ck}', M) + f"  C0{'✓' if not bad else '✗ ' + '; '.join(bad)} C1({k}/3 점수{sc:+.1f}%p{note})")
    print('\n[체크포인트 선택] 선별 C0 통과 중 C1 점수(절대 %p) 최대(5 %p 동률 → nd 부호반전)')
    pick = {}
    for fam in FAMS:
        done = [(ck, v) for (fm, ck), v in screen.items() if fm == fam]
        if len(done) < 4: print(f'  {fam}: 대기 ({len(done)}/4)'); continue
        ok = [(ck, v) for ck, v in done if not v[1]]
        if not ok: print(f'  {fam}: 선별 C0 통과 없음 → 탈락'); continue
        best = max(v[2] for _, v in ok)
        tied = sorted([(ck, v) for ck, v in ok if v[2] >= best - 5], key=lambda x: x[1][0]['nd']['rev'])
        pick[fam] = tied[0][0]; print(f'  {fam}: {pick[fam]} (점수 {tied[0][1][2]:+.1f}%p)')
    FS = (2, 3, 4)   # 정식 = 선택에 쓰지 않은 seed (부록 §8)
    B3 = load(G12, 'model_9500.pt', 'g17b_g12', FS)
    print('\n[정식] seed 2–4')
    if B3 is None: print('  기준선 seed 2–4 대기'); return
    MB3 = {c: summarize(v) for c, v in B3.items()}
    print(hdr); print(table('G12-9500', MB3))
    formal = {}
    for fam, ck in pick.items():
        R = load(run_dir(fam), f'model_{ck}.pt', f'g17_{fam}_{ck}', FS)
        if R is None: print(f'  {fam}-{ck}: 대기'); continue
        M = {c: summarize(R[c], B3[c]) for c in CONDS}
        bad = c0(M, MB3, TH['formal']); ok1, k, sc, note = c1(M, MB3)
        formal[fam] = (ck, M, bad, ok1, sc)
        pair = ' '.join(f"{c}:{M[c]['only_c']}/{M[c]['only_b']}(p={mcnemar(M[c]['only_c'], M[c]['only_b']):.3f})" for c in ('sf10', 'sb10', 'b18'))
        print(table(f'{fam}-{ck}', M) + f"  C0{'✓' if not bad else '✗ ' + '; '.join(bad)} C1{'✓' if ok1 else '✗'}({k}/3 {sc:+.1f}%p{note}) 짝(후보만/기준만 낙상) {pair}")
    passed = sorted([(v[4], fam) for fam, v in formal.items() if not v[2] and v[3]], reverse=True)
    print('\n[가설] H-o: g17o·g17o43 각각 정식 C0+C1 →', {k: ('대기' if k not in formal else ('통과' if not formal[k][2] and formal[k][3] else '탈락')) for k in ('g17o', 'g17o43')})
    print('  H-r·H-p 는 같은 체크포인트·seed 1 선별 표로 판정(문서 §5) — 아래 표 참조')
    for fam, ref, key in (('g17r', 'g17o', 'rev'), ('g17r43', 'g17o43', 'rev'), ('g17p', 'g17o', 'sfmean')):
        wins = []
        for ck in CKS:
            a, o = screen.get((fam, ck)), screen.get((ref, ck))
            if not (a and o): continue
            if key == 'rev':
                wins.append(a[0]['nd']['rev'] <= 0.7 * o[0]['nd']['rev'] and a[2] >= o[2] - 10)   # nd 부호반전만 (부록 §8)
            else:
                am = (a[0]['sf10']['fall'] + a[0]['sf15']['fall']) / 2; om = (o[0]['sf10']['fall'] + o[0]['sf15']['fall']) / 2
                wins.append(am <= om - 15)
        print(f'  {fam} vs {ref}: 체크포인트 승 {sum(wins)}/{len(wins)} → ' + ('지지' if len(wins) == 4 and sum(wins) >= 3 else ('대기' if len(wins) < 4 else '미지지')))
    print('  H-r 지지 = g17r·g17r43 둘 다 체크포인트 승 ≥3/4 (하나만 = 잠정)')
    print('\n[배포 판정] ' + (f'후보 {passed[0][1]}-{formal[passed[0][1]][0]} (C1 점수 {passed[0][0]:+.1f}%p) — '
          + ('seed 복제 계열' if passed[0][1] in ('g17o', 'g17o43', 'g17r', 'g17r43') else '단일 seed 계열 → 독립 seed 5–6 재평가에서 C1 유지 필요')
          + ' · 아침 실측 d* ≥ 3 이면 보류 · G12-9500 대조와 함께 실물 비교' if passed else '통과 후보 없음 → G12-9500 유지'))

if __name__ == '__main__':
    main()
