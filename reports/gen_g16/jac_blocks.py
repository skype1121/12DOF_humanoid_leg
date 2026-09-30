"""정책 입력 블록별 게인(야코비안 노름) — 같은 관측점(0929 실물 로그 0–10 s 조용한 서기, G12-9500 관측)에서 체크포인트 비교.
‖J_blk‖ = 액션(12) 에 대한 블록 입력의 야코비안 프로베니우스 노름, 관측점 평균. 폐루프 성능이 아니라 '어느 입력에 얼마나 반응하나'의 지표."""
import json, sys, glob
import torch
REPO = '/home/ryu/humanoid_leg_test1'
LOG = f'{REPO}/reports/real_0929/g12/rl_bridge_20260929_210129.jsonl'
BLK = [('자이로', 0, 15), ('중력', 15, 30), ('명령', 30, 45), ('관절각', 45, 105), ('관절속도', 105, 165), ('직전액션', 165, 225),
       ('클록', 225, 235), ('접촉', 235, 245), ('자이로EMA', 245, 260), ('속도EMA', 260, 320)]


def actor(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    ks = sorted({int(k.split('.')[1]) for k in sd if k.startswith('actor.') and k.endswith('.weight')})
    W = [(sd[f'actor.{k}.weight'], sd[f'actor.{k}.bias']) for k in ks]
    def f(x):
        for i, (w, b) in enumerate(W):
            x = x @ w.T + b
            if i < len(W) - 1:
                x = torch.nn.functional.elu(x)
        return x
    return f


def main():
    rows = [json.loads(l) for l in open(LOG)]
    obs = [r['obs'] for r in rows if r.get('obs') is not None]
    X = torch.tensor(obs[:500:5], dtype=torch.float32)          # 0–10 s 조용한 구간 100점
    runs = {'g12': '2026-09-29_14-32-50_g12', 'g12c': '2026-09-30_12-05-46_g12c', 'g16o': '2026-10-01_01-46-29_g16o'}
    cks = [('g12', 9500)] + [(r, c) for c in (10500, 10800, 11100, 11499) for r in ('g12c', 'g16o')]
    print(f"{'체크포인트':14s} " + ' '.join(f'{n:>7s}' for n, _, _ in BLK))
    for r, c in cks:
        f = actor(f'{REPO}/logs/rsl_rl/biped12_stage4/{runs[r]}/model_{c}.pt')
        J = torch.func.vmap(torch.func.jacrev(f))(X)             # (P, 12, 320)
        vals = [float(J[:, :, a:b].pow(2).sum((1, 2)).sqrt().mean()) for _, a, b in BLK]
        print(f'{r + "-" + str(c):14s} ' + ' '.join(f'{v:7.2f}' for v in vals))


if __name__ == '__main__':
    main()
