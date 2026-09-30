"""G15 사전 기준(reports/gen_g15/G15_CRITERIA.md) 판정. 없는 결과는 '대기'로 표시."""
import csv, math, os, sys
os.chdir('/home/ryu/humanoid_leg_test1/log_picture')
def f(x): return float(x) if x not in ('', None) else math.nan
def stat(prefix):
    n = gf = sus = damp = fall = 0
    for sd in (1, 2, 3):
        p = f'ground_recovery_{prefix}_s{sd}_envs.csv'
        if not os.path.exists(p): return None
        for r in csv.DictReader(open(p)):
            if r['pre_pass'] != '1': continue
            n += 1; pp = [f(r[f'p2p_{k}']) for k in range(14)]
            if r['fell'] == '1':
                fall += 1
                if f(r['t_fall_rel']) > 3: gf += 1
                continue
            if max(pp[6:14]) >= 4 and max(pp[6:14]) >= 2 * max(pp[1:4]): gf += 1
            if max(pp[10:14]) <= 2: damp += 1
            else: sus += 1
    return dict(n=n, gf=gf, sus=sus, damp=damp, fall=fall)
P = lambda a, b: 100.0 * a / b if b else math.nan
CONDS = [(1, 'nd'), (1, 'f8'), (1, 'b8'), (3, 'nd'), (3, 'f8'), (3, 'b8'), (11, 'nd'), (11, 'b8'), (3, 'b18')]
def base_prefix(dl, ds):
    if dl in (1, 3) and ds != 'b18': return f'osc_short_g12_d{dl}_{ds}'
    if dl == 11: return f'osc_g12_d{dl}_{ds}'
    return f'osc15_g12base_9500_d{dl}_{ds}'
def fam_prefix(fam, ck, dl, ds): return f'osc15_{fam}_{ck}_d{dl}_{ds}'
def table(get):
    t = {}
    for dl, ds in CONDS:
        s = get(dl, ds)
        if s is None: return None
        t[(dl, ds)] = s
    return t
def metrics(t):
    m = {}
    for dl in (1, 3):
        nd = t[(dl, 'nd')]
        m[f'd{dl}_fall'] = P(nd['fall'], nd['n']); m[f'd{dl}_damp'] = P(nd['damp'], nd['n']); m[f'd{dl}_sus'] = P(nd['sus'], nd['n'])
        m[f'd{dl}_push'] = P(t[(dl, 'f8')]['gf'] + t[(dl, 'b8')]['gf'], t[(dl, 'f8')]['n'] + t[(dl, 'b8')]['n'])
    m['d11_sus'] = P(t[(11, 'nd')]['sus'], t[(11, 'nd')]['n']); m['d11_b8'] = P(t[(11, 'b8')]['gf'], t[(11, 'b8')]['n'])
    m['b18_fall'] = P(t[(3, 'b18')]['fall'], t[(3, 'b18')]['n'])
    m['stress'] = m['d11_sus'] + m['d11_b8'] + m['b18_fall']
    return m
base = table(lambda dl, ds: stat(base_prefix(dl, ds)))
if base is None: print('기준선 결과 대기'); sys.exit(0)
B = metrics(base)
cols = ['d1_fall', 'd1_damp', 'd1_sus', 'd1_push', 'd3_fall', 'd3_damp', 'd3_sus', 'd3_push', 'd11_sus', 'd11_b8', 'b18_fall', 'stress']
print(f"{'후보':14s} " + " ".join(f"{c:>8s}" for c in cols))
print(f"{'G12-9500':14s} " + " ".join(f"{B[c]:8.1f}" for c in cols))
def c0(m):
    ok = True
    for dl in (1, 3):
        ok &= m[f'd{dl}_fall'] <= B[f'd{dl}_fall'] + 1 and m[f'd{dl}_damp'] >= B[f'd{dl}_damp'] - 2
        ok &= m[f'd{dl}_sus'] <= B[f'd{dl}_sus'] + 1 and m[f'd{dl}_push'] <= B[f'd{dl}_push'] + 1
    return ok
def c1(m):
    keys = ('d11_sus', 'd11_b8', 'b18_fall')
    imp = [k for k in keys if B[k] > 0 and m[k] <= 0.7 * B[k]]
    worse = [k for k in keys if m[k] > B[k] + 2]
    return bool(imp) and not worse, imp, worse
res = {}
for fam in ('g12c', 'g15q', 'g15u', 'g15w', 'g15d', 'g15k'):
    for ck in (10500, 11499):
        t = table(lambda dl, ds: stat(fam_prefix(fam, ck, dl, ds)))
        if t is None: print(f"{fam}-{ck:<8d} 대기"); continue
        m = metrics(t); res[(fam, ck)] = m
        ok0 = c0(m); ok1, imp, worse = c1(m)
        print(f"{fam+'-'+str(ck):14s} " + " ".join(f"{m[c]:8.1f}" for c in cols) + f"  C0 {'✓' if ok0 else '✗'} C1 {'✓' if ok1 else '✗'}"
              + (f" 개선{imp}" if imp else "") + (f" 퇴보{worse}" if worse else ""))
print("\n가설(같은 체크포인트에서 G12c 대비 지연 11 무외란 흔들림 지속이 2쌍 모두 적음):")
for fam in ('g15q', 'g15u', 'g15w', 'g15d', 'g15k'):
    pairs = [(ck, res.get((fam, ck)), res.get(('g12c', ck))) for ck in (10500, 11499)]
    if any(a is None or b is None for _, a, b in pairs): print(f"  {fam}: 대기"); continue
    wins = sum(a['d11_sus'] < b['d11_sus'] for _, a, b in pairs)
    print(f"  {fam}: " + " | ".join(f"{ck}: {a['d11_sus']:.1f} vs {b['d11_sus']:.1f}" for ck, a, b in pairs) + f" → {'지지' if wins == 2 else '미지지'}")
