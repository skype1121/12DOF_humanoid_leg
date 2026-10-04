import sys, math
sys.path.insert(0, '/home/ryu/humanoid_leg_test1/reports/gen_g17'); import g17_verdict as V
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
def counts(tag, run, ck, sd, c):
    r = V.one(tag, run, ck, sd, c)
    if r is None: return None
    n = sus = 0; flags = []
    for row in r[0]:
        if row['pre_pass'] != '1': flags.append(None); continue
        n += 1
        pp = [f(row[f'p2p_{k}']) for k in range(8)]
        s = row['fell'] != '1' and max(pp[5:8]) > 2
        sus += s; flags.append(s)
    return n, sus, flags
B = ('2026-10-02_21-23-08_g21a', 'model_11300.pt'); C = ('2026-10-04_10-08-41_g24ra_seed1_from_w2e20294', 'model_21793.pt')
for c in ('nd', 'd3nd'):
    tb = [0, 0]; tc = [0, 0]; oc = ob = 0
    for sd in (1, 2, 3, 4):
        bp = 'g21_g21a_11300' if sd == 1 else 'g18_g21a_11300'
        b = counts(f'{bp}_{c}_s{sd}', *B, sd, c); k = counts(f'g24ra_21793_{c}_s{sd}', *C, sd, c)
        if b is None or k is None: print(c, sd, '없음', b is None, k is None); continue
        tb[0] += b[0]; tb[1] += b[1]; tc[0] += k[0]; tc[1] += k[1]
        for x, y in zip(b[2], k[2]):
            if x is None or y is None: continue
            oc += y and not x; ob += x and not y
        print(f'{c} s{sd}: G21a {b[1]}/{b[0]}  G24R-a {k[1]}/{k[0]}')
    def ci(s, n):
        p = s / max(n, 1); z = 1.96; d = 1 + z*z/n; cen = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
        return f'{100*p:.1f} % [{100*(cen-h):.1f}, {100*(cen+h):.1f}]'
    nd_ = oc + ob
    pv = sum(math.comb(nd_, i) for i in range(0, min(oc, ob) + 1)) * 2 / 2**nd_ if nd_ else 1.0
    print(f'{c} 합계: G21a {tb[1]}/{tb[0]} = {ci(tb[1], tb[0])} · G24R-a {tc[1]}/{tc[0]} = {ci(tc[1], tc[0])} · 짝 불일치 후보만 {oc} / 기준만 {ob} · 부호검정 p={min(pv,1):.3g}')
