"""한 발 하중 상실 뒤 고주파 튐 서명 — 실물 블랙박스와 시뮬 덤프에 같은 정의 적용(사전 기준, 1001).
사건: 한 발 법선력이 5 N 아래로 처음 떨어진 시각 t0. 창: t0 ~ t0+1.5 s.
서명(튐) = 창 안 피치 p2p > 5° 이고 피치 지배 주파수 ≥ 2 Hz. 보조: 창 안 |ωy| 최대, 접촉 비트 전환 횟수."""
import json, sys, glob, numpy as np

def feat(t, pitch, wy, fL, fR, cL, cR, dt):
    lo = np.where((np.minimum(fL, fR) < 5.0))[0]
    if len(lo) == 0:
        return None
    i0 = lo[0]; n = int(1.5 / dt)
    sl = slice(i0, min(len(t), i0 + n))
    p = pitch[sl]
    if len(p) < int(0.8 / dt):
        return None
    x = p - p.mean(); F = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), dt); k = np.argmax(F[1:]) + 1
    sw = int(np.sum(np.abs(np.diff(cL[sl]))) + np.sum(np.abs(np.diff(cR[sl]))))
    return dict(t0=float(t[i0]), p2p=float(np.ptp(p)), f=float(fr[k]), wmax=float(np.abs(wy[sl]).max()), switches=sw,
                bounce=bool(np.ptp(p) > 5 and fr[k] >= 2.0))

def real(path):
    rows = [json.loads(l) for l in open(path)]
    r = [x for x in rows if 'event' not in x and x.get('rp_deg') and x.get('foot_n')]
    t = np.array([x['timestamp'] for x in r]); t -= t[0]
    return feat(t, np.array([x['rp_deg'][1] for x in r]), np.array([x['gyro_rad_s'][1] for x in r]),
                np.array([x['foot_n'][0] for x in r]), np.array([x['foot_n'][1] for x in r]),
                np.array([x['contact'][0] for x in r]), np.array([x['contact'][1] for x in r]), 0.02)

def sim(tag):
    out = []
    for f in sorted(glob.glob(f'log_picture/recbb_{tag}_env*.jsonl')):
        d = [json.loads(l) for l in open(f)]
        d = [x for x in d if x['t_rel'] >= 1.9]
        if len(d) < 60: continue
        t = np.array([x['t_rel'] for x in d])
        res = feat(t, np.array([x['pitch'] for x in d]), np.gradient(np.radians([x['pitch'] for x in d]), 0.02),
                   np.array([x['fz_L'] for x in d]), np.array([x['fz_R'] for x in d]),
                   np.array([x['contact'][0] for x in d], float), np.array([x['contact'][1] for x in d], float), 0.02)
        out.append(res)
    return out

if __name__ == '__main__':
    for p in sys.argv[1:]:
        if p.endswith('.jsonl'):
            print('실물', p.split('/')[-1], real(p))
        else:
            res = sim(p); ev = [x for x in res if x]
            b = sum(x['bounce'] for x in ev)
            print(f"시뮬 {p}: 덤프 {len(res)} · 한발 하중상실 {len(ev)} · 튐 {b}"
                  + (f" · 중앙 p2p {np.median([x['p2p'] for x in ev]):.1f}° f {np.median([x['f'] for x in ev]):.1f}Hz |ωy| {np.median([x['wmax'] for x in ev]):.2f} 접촉전환 {np.median([x['switches'] for x in ev]):.0f}" if ev else ""))
