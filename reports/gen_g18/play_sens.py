"""유격 감도 보고: ps_{policy}_{base|p2|p0}_nd — 흔들림 지속 % 와 dump env 피치 지배 주파수·p2p (인계 3 s 이후)."""
import csv, json, glob, sys, numpy as np
def st(t):
    R = [r for r in csv.DictReader(open(f'log_picture/ground_recovery_{t}_envs.csv')) if r['pre_pass'] == '1']
    alive = [r for r in R if r['fell'] != '1']
    sus = sum(max(float(r[f'p2p_{k}']) for k in (5, 6, 7)) > 2 for r in alive)
    fs, amps = [], []
    for f in glob.glob(f'log_picture/recbb_{t}_env*.jsonl'):
        k = [x for x in (json.loads(l) for l in open(f)) if isinstance(x, dict) and x.get('t_rel', -1) >= 3.0]
        if len(k) < 100: continue
        key = [kk for kk in k[0] if 'pitch' in kk][0]
        p = np.array([x[key] for x in k]); p -= p.mean()
        F = np.abs(np.fft.rfft(p * np.hanning(len(p)))); fr = np.fft.rfftfreq(len(p), 0.02); m = fr > 0.2
        fs.append(fr[m][F[m].argmax()]); amps.append(p.max() - p.min())
    return len(R), len(R) - len(alive), 100 * sus / max(len(R), 1), (np.median(fs) if fs else float('nan')), (np.median(amps) if amps else float('nan'))
for p in sys.argv[1:]:
    for v in ('base', 'p2', 'p0'):
        t = f'ps_{p}_{v}_nd'
        try: n, fl, s, f0, a = st(t); print(f'{t:18s} n={n} 낙상 {fl} 흔들림지속 {s:5.1f} % · 지배 {f0:.2f} Hz · 피치 p2p {a:.2f}°')
        except FileNotFoundError: print(f'{t:18s} 대기')
