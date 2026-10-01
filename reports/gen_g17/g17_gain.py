"""G17 보고 지표(판정 미사용, G17_CRITERIA §3 보조): 실물 1001 블랙박스 관측을 체크포인트 actor 에 넣어
(1) 튐 창 2–4 Hz 액션 진폭, (2) 자이로·관절속도 블록만 흔들 때 진폭, (3) 조용한 서기 창 진폭 을 G12-9500 대비 % 로 낸다.
개루프 분석 — 폐루프에서 튐이 줄었다는 증명이 아니다(실물로만 확인). 사용: g17_gain.py <run_dir/model_N.pt> ..."""
import json, sys, numpy as np, torch
BB = [('reports/real_1001/rl_bridge_20261001_185555.jsonl', 8.3, 9.6), ('reports/real_1001/rl_bridge_20261001_193901.jsonl', 38.2, 39.6)]
QUIET = ('reports/real_1001/rl_bridge_20261001_184814.jsonl', 20.0, 40.0)
GYRO, JVEL = (0, 15), (105, 165)
def actor(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    ks = sorted({int(k.split('.')[1]) for k in sd if k.startswith('actor.') and k.endswith('.weight')})
    W = [(sd[f'actor.{k}.weight'], sd[f'actor.{k}.bias']) for k in ks]
    def pi(x):
        x = torch.tensor(x, dtype=torch.float32)
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < len(W) - 1: x = torch.nn.functional.elu(x)
        return x.detach().numpy()
    return pi
def band(a, lo=2.0, hi=4.0):
    a = a - a.mean(0); F = np.fft.rfft(a * np.hanning(len(a))[:, None], axis=0); f = np.fft.rfftfreq(len(a), 0.02)
    m = (f >= lo) & (f <= hi); return float(np.sqrt((np.abs(F[m][:, [0, 1, 6, 7, 8, 9]]) ** 2).sum()))
def load(fn):
    r = [json.loads(l) for l in open(fn)]; r = [x for x in r if 'event' not in x and x.get('obs')]
    t = np.array([x['timestamp'] for x in r]); t -= t[0]
    return t, np.array([x['obs'] for x in r], dtype=np.float32)
DATA = [(load(fn), w0, w1) for fn, w0, w1 in BB]
QD = load(QUIET[0])
def metrics(pi):
    out = []
    for (t, O), w0, w1 in DATA:
        m = (t >= w0) & (t < w1); pre = (t >= w0 - 1.0) & (t < w0)
        full = band(pi(O[m]))
        blk = []
        for a, b in (GYRO, JVEL):
            Y = np.repeat(O[pre].mean(0, keepdims=True), m.sum(), 0); Y[:, a:b] = O[m][:, a:b]; blk.append(band(pi(Y)))
        out.append((full, *blk))
    t, O = QD; m = (t >= QUIET[1]) & (t < QUIET[2])
    q = float(np.sqrt(((pi(O[m]) - pi(O[m]).mean(0)) ** 2).mean()))
    return np.array(out), q
ref, refq = metrics(actor('logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/model_9500.pt'))
print('기준 G12-9500 = 100 %. 칸: 튐창(2회차/3회차) 전체 · 자이로만 · 관절속도만 | 조용한 서기 액션 표준편차')
for p in sys.argv[1:]:
    m, q = metrics(actor(p)); r = 100 * m / ref
    print(f"{'/'.join(p.split('/')[-2:]):48s} 전체 {r[0,0]:4.0f}/{r[1,0]:4.0f} · 자이로 {r[0,1]:4.0f}/{r[1,1]:4.0f} · 관절속도 {r[0,2]:4.0f}/{r[1,2]:4.0f} | 조용 {100*q/refq:4.0f}")
