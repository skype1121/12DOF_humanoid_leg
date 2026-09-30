"""1001 OSC 재현 v2 A/B — 관측만 다른(legacy vs v2) G12-9500 평가에서 실물 0929 서명(2 Hz·양발 접촉 저하·발목롤 진동)이
v2 에서만 나오는지 본다. 사전 계획(G16_DESIGN_v2 §0.1): 낙상률이 아니라 서명으로 판정.

입력: log_picture/recbb_{osc_g12|osc_g12v2}_d{11,14}_{nd,f8,b8}_s{1,2,3}_env{0..5}.jsonl (env 6개 덤프)
     + ground_recovery_*_envs.csv (128 env 집계: 성장·흔들림 지속·낙상).
서명 env: 인계 뒤 4–14 s 피치 p2p > 2° 이고 지배 주파수 1.5–2.5 Hz. 보조: 인계 2 s 뒤 양발 접촉 비율, 발목롤 표준편차."""
import csv, glob, json, math, os
import numpy as np
LP = '/home/ryu/humanoid_leg_test1/log_picture'


def trace(path):
    rows = [json.loads(l) for l in open(path)]
    r = [x for x in rows if x['t_rel'] >= 0]
    if len(r) < 100:
        return None
    t = np.array([x['t_rel'] for x in r]); p = np.array([x['pitch'] for x in r])
    alive = len(r) >= int(13.5 / 0.02)
    m = (t >= 4) & (t < 14)
    if m.sum() < 200:
        return dict(alive=False)
    x = p[m] - p[m].mean(); F = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), 0.02)
    k = np.argmax(F[1:]) + 1
    c = np.array([x['contact'] for x in r])[t >= 2]
    ar = np.array([[x['q_deg'][10], x['q_deg'][11]] for x in r])[t >= 2]
    return dict(alive=alive, p2p=float(p[m].max() - p[m].min()), f=float(fr[k]), both=float((c.min(1) > 0).mean()),
                ank_r_std=float(ar.std(0).mean()))


def agg(prefix):
    n = gf = sus = fall = 0
    for sd in (1, 2, 3):
        pth = f'{LP}/ground_recovery_{prefix}_s{sd}_envs.csv'
        if not os.path.exists(pth):
            return None
        for r in csv.DictReader(open(pth)):
            if r['pre_pass'] != '1':
                continue
            n += 1; pp = [float(r[f'p2p_{k}']) if r[f'p2p_{k}'] else math.nan for k in range(14)]
            if r['fell'] == '1':
                fall += 1; gf += float(r['t_fall_rel'] or 0) > 3; continue
            if max(pp[6:14]) >= 4 and max(pp[6:14]) >= 2 * max(pp[1:4]): gf += 1
            if max(pp[10:14]) > 2: sus += 1
    return n, 100 * gf / n, 100 * sus / n, 100 * fall / n


def main():
    print(f"{'조건':10s} | {'관측':6s} | {'128env: 성장·낙상 / 지속 / 낙상 %':34s} | 덤프 env: 서명(2Hz·p2p>2°) / 살아남음 · p2p 중앙 · 양발 접촉 · 발목롤 σ°")
    for d in (11, 14):
        for c in ('nd', 'f8', 'b8'):
            for mode, pre in (('legacy', 'osc_g12'), ('v2', 'osc_g12v2')):
                a = agg(f'{pre}_d{d}_{c}')
                tr = [trace(f) for f in sorted(glob.glob(f'{LP}/recbb_{pre}_d{d}_{c}_s[123]_env*.jsonl'))]
                tr = [x for x in tr if x and x.get('alive')]
                if a is None:
                    print(f'd{d} {c:4s}   | {mode:6s} | 대기'); continue
                sig = [x for x in tr if x['p2p'] > 2 and 1.5 <= x['f'] <= 2.5]
                med = np.median([x['p2p'] for x in tr]) if tr else math.nan
                both = np.mean([x['both'] for x in tr]) if tr else math.nan
                ars = np.mean([x['ank_r_std'] for x in tr]) if tr else math.nan
                print(f'd{d} {c:4s}   | {mode:6s} | {a[1]:6.1f} / {a[2]:6.1f} / {a[3]:6.1f} (n={a[0]})        | '
                      f'{len(sig)}/{len(tr)} · {med:.2f}° · {100 * both:.0f} % · {ars:.2f}')
    print('\n판정(사전): v2 에서만 서명 env 가 나오고(legacy 0) 양발 접촉이 낮아지면 "v2 관측이 실물 2 Hz 서명의 후보 기전" 지지. 아니면 미지지.')


if __name__ == '__main__':
    main()
