"""실물 3 Hz 흔들림 원인 분해: 실물 블랙박스 관측(320)을 G12-9500 actor 에 넣어 명령 재현 → 관측 묶음을 하나씩 '흔들림 직전 1 s 평균'으로
고정(제거 시험)하거나, 그 묶음만 남기고 나머지를 고정(단독 시험)해 액션의 2–4 Hz 대역 진폭이 얼마나 남는지 본다. 개루프 분석(폐루프 증명 아님)."""
import json, sys, numpy as np, torch
BLK = [('자이로', 0, 15), ('중력', 15, 30), ('명령', 30, 45), ('관절각', 45, 105), ('관절속도', 105, 165), ('직전액션', 165, 225),
       ('클록', 225, 235), ('접촉', 235, 245), ('자이로EMA', 245, 260), ('속도EMA', 260, 320)]
sd = torch.load('logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/model_9500.pt', map_location='cpu', weights_only=False)['model_state_dict']
ks = sorted({int(k.split('.')[1]) for k in sd if k.startswith('actor.') and k.endswith('.weight')})
W = [(sd[f'actor.{k}.weight'], sd[f'actor.{k}.bias']) for k in ks]
def pi(x):
    x = torch.tensor(x, dtype=torch.float32)
    for i, (w, b) in enumerate(W):
        x = x @ w.T + b
        if i < len(W) - 1: x = torch.nn.functional.elu(x)
    return x.detach().numpy()
def band(a, lo=2.0, hi=4.0):   # (T,12) → 2–4 Hz 대역 RMS (힙앞뒤 0,1 + 발목앞뒤 8,9 + 무릎 6,7)
    a = a - a.mean(0); F = np.fft.rfft(a * np.hanning(len(a))[:, None], axis=0); f = np.fft.rfftfreq(len(a), 0.02)
    m = (f >= lo) & (f <= hi); cols = [0, 1, 6, 7, 8, 9]
    return float(np.sqrt((np.abs(F[m][:, cols]) ** 2).sum()))
for fn, w0, w1 in (('reports/real_1002/rl_bridge_20261002_143511.jsonl', 44.3, 45.9),):
    r = [json.loads(l) for l in open(fn)]; r = [x for x in r if 'event' not in x and x.get('obs')]
    t = np.array([x['timestamp'] for x in r]); t -= t[0]
    O = np.array([x['obs'] for x in r], dtype=np.float32); A = np.array([x['raw_action'] for x in r])
    m = (t >= w0) & (t < w1); pre = (t >= w0 - 1.0) & (t < w0)
    base = pi(O[m]); err = np.abs(base - A[m]).max()
    b0 = band(base)
    print(f'== {fn.split("/")[-1]} 창 {w0}–{w1}s: 재현 오차 {err:.3f} · 2–4 Hz 액션 진폭(기준) {b0:.2f}')
    print(f"   {'묶음':8s} {'제거하면 남는 %':>14s} {'이것만 흔들면 %':>15s}")
    for name, a, b in BLK:
        X = O[m].copy(); X[:, a:b] = O[pre][:, a:b].mean(0)
        Y = np.repeat(O[pre].mean(0, keepdims=True), m.sum(), 0); Y[:, a:b] = O[m][:, a:b]
        print(f'   {name:8s} {100*band(pi(X))/b0:13.0f}% {100*band(pi(Y))/b0:14.0f}%')
