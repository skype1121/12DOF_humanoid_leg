"""G13 사전 기준(G13_CRITERIA.md) 판정표. 사전검사 통과 env 만 집계."""
import csv, math, os, sys
os.chdir('/home/ryu/humanoid_leg_test1/log_picture')
def f(x): return float(x) if x not in ('', None) else math.nan
def stats(prefix):
    """prefix 예: osc_g13_10000 → {(dl,ds): dict}"""
    out = {}
    for dl in (11, 14, 17):
        for ds in ('nd', 'f8', 'b8'):
            n = gf = sus = damp = fall = 0
            for sd in (1, 2, 3):
                p = f'ground_recovery_{prefix}_d{dl}_{ds}_s{sd}_envs.csv'
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
            out[(dl, ds)] = dict(n=n, gf=gf, sus=sus, damp=damp, fall=fall)
    return out
def c1(s): return sum(s[(dl, ds)]['gf'] for dl in (11, 14, 17) for ds in ('f8', 'b8'))
def c1n(s): return sum(s[(dl, ds)]['n'] for dl in (11, 14, 17) for ds in ('f8', 'b8'))
def c2(s): return s[(11, 'nd')]['sus'] + s[(14, 'nd')]['sus']
def pct(a, b): return 100.0 * a / b if b else math.nan
base = stats('osc_g12')     # 재현 시험의 G12-9500 (osc_g12_d{dl}_{ds}_s{sd})
fams = {}
for fam in ('g13', 'g12c', 'g14a', 'g14b'):
    cks = sorted({int(x.split('_')[4]) for x in os.listdir('.') if x.startswith(f'ground_recovery_osc_{fam}_') and x.endswith('_envs.csv')})
    fams[fam] = {ck: stats(f'osc_{fam}_{ck}') for ck in cks}
    fams[fam] = {k: v for k, v in fams[fam].items() if v is not None}
print(f"{'후보':14s} {'C1 밀기6조건 성장·낙상':>22s} {'C2 무외란 지속흔들림':>18s} {'무외란d11 낙상':>13s} {'무외란d11 감쇠':>13s}")
def row(name, s):
    nd = s[(11, 'nd')]
    print(f"{name:14s} {c1(s):5d}/{c1n(s)} ({pct(c1(s), c1n(s)):4.1f}%)      {c2(s):5d}           {pct(nd['fall'], nd['n']):5.1f}%       {pct(nd['damp'], nd['n']):5.1f}%")
row('G12-9500', base)
for fam in fams:
    for ck, s in sorted(fams[fam].items()): row(f'{fam}-{ck}', s)
# 선택 규칙: C3(대 같은 계열의 선택본이 아니라 G12c 기준) — 대조군 G12c 는 C3 을 자기 자신 대비로 볼 수 없으므로 C3 은 G12-9500 대비로 선택, 판정은 G12c 선택본 대비
def c3_ok(s, ref):
    a, b = s[(11, 'nd')], ref[(11, 'nd')]
    return pct(a['fall'], a['n']) <= pct(b['fall'], b['n']) + 1 and pct(a['damp'], a['n']) >= pct(b['damp'], b['n']) - 2
sel = {}
for fam in fams:
    ok = {ck: s for ck, s in fams[fam].items() if c3_ok(s, base)}
    pool = ok or fams[fam]
    if not pool: continue
    sel[fam] = min(pool, key=lambda ck: c1(pool[ck]))
    print(f"선택 {fam}: {sel[fam]} (C3 vs G12-9500 통과 {sorted(ok)})")
if 'g13' in sel and 'g12c' in sel:
    g, c = fams['g13'][sel['g13']], fams['g12c'][sel['g12c']]
    C1 = c1(g) <= 0.7 * c1(c) and c1(g) <= 0.7 * c1(base)
    C2 = c2(g) <= c2(c)
    C3 = c3_ok(g, c)
    print(f"\n판정 G13-{sel['g13']} vs G12c-{sel['g12c']}: C1 {c1(g)} ≤ 0.7×{c1(c)}={0.7*c1(c):.1f} & ≤ 0.7×{c1(base)}={0.7*c1(base):.1f} → {'통과' if C1 else '탈락'} | C2 {c2(g)} ≤ {c2(c)} → {'통과' if C2 else '탈락'} | C3 → {'통과' if C3 else '탈락'}")
    print("지연별 밀기 성장·낙상 % (앞/뒤):")
    for nm, s in (('G12-9500', base), (f"G12c-{sel['g12c']}", c), (f"G13-{sel['g13']}", g)):
        print(f"  {nm:14s} " + "  ".join(f"d{dl} {pct(s[(dl,'f8')]['gf'], s[(dl,'f8')]['n']):4.1f}/{pct(s[(dl,'b8')]['gf'], s[(dl,'b8')]['n']):4.1f}" for dl in (11, 14, 17)))
