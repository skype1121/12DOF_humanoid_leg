"""G18 사전검사(CPU, 보고 전용): 실물 1001 조용한 서기 관측점(184814 블랙박스 20–40 s, 40 점)에서
(1) 정책 평균 출력의 관측 블록별 야코비안 ‖∂μ/∂x_blk‖²_F, (2) last_action '자기루프' — 다른 관측은 고정, 출력만 actions 이력
(165–224, 오래된→최신 12×5)으로 되먹여 120틱 굴린 뒤 마지막 60틱의 틱 간 부호반전 비율·액션 표준편차.
플랜트 없는 개루프 지표 — 폐루프 떨림의 원인 증명이 아니다(OSS 분석 1 가설 확인용). 사용: jac_selfloop.py <ckpt.pt> ..."""
import json, os, sys, numpy as np, torch
EMA = float(os.environ.get('LA_EMA', '1.0'))   # G18f 계약: last_action 관측 = 원시 액션 EMA(β) — LA_EMA=0.27
BLK = {'자이로': (0, 15), '관절속도': (105, 165), '직전액션': (165, 225), '관절각': (45, 105)}
def actor(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    ks = sorted({int(k.split('.')[1]) for k in sd if k.startswith('actor.') and k.endswith('.weight')})
    W = [(sd[f'actor.{k}.weight'], sd[f'actor.{k}.bias']) for k in ks]
    def pi(x):
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < len(W) - 1: x = torch.nn.functional.elu(x)
        return x
    return pi
r = [json.loads(l) for l in open('reports/real_1001/rl_bridge_20261001_184814.jsonl')]
r = [x for x in r if 'event' not in x and x.get('obs')]
t = np.array([x['timestamp'] for x in r]); t -= t[0]
O = np.array([x['obs'] for x in r], dtype=np.float32)
X0 = torch.tensor(O[(t >= 20) & (t < 40)][::25])
print(f'관측점 {len(X0)}개 (실물 조용한 서기). last_action 되먹임 EMA β={EMA}. 칸: 블록 야코비안² (자이로·관절속도·직전액션·관절각) | 자기루프 부호반전/틱 · 액션 표준편차')
for p in sys.argv[1:]:
    pi = actor(p)
    x = X0.clone().requires_grad_(True); mu = pi(x)
    J = torch.stack([torch.autograd.grad(mu[:, k].sum(), x, retain_graph=True)[0] for k in range(mu.shape[1])], 1)
    jn = {n: float(J[:, :, a:b].pow(2).sum((1, 2)).mean()) for n, (a, b) in BLK.items()}
    with torch.no_grad():
        x = X0.clone(); x[:, 165:225] += 0.02 * torch.randn(len(x), 60)
        acts = []
        for k in range(120):
            a = pi(x); acts.append(a)
            new = EMA * a + (1.0 - EMA) * x[:, 213:225]
            x[:, 165:213] = x[:, 177:225].clone(); x[:, 213:225] = new
        A = torch.stack(acts[60:])                      # (60, P, 12)
        d = A[1:] - A[:-1]
        rev = float(((d[1:] * d[:-1]) < 0).float().mean())
        sd = float(A.std(0).mean())
    print(f"{'/'.join(p.split('/')[-2:]):46s} " + ' '.join(f'{n} {v:6.2f}' for n, v in jn.items()) + f" | 루프 반전 {rev:.3f} · 표준편차 {sd:.4f}")
