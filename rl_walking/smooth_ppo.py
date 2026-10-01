"""G18 학습 알고리즘 변형 (rsl_rl 3.1.2 PPO 하위 클래스, site-packages 무수정).

SmoothPPO: 정책 평균 출력의 **관측 블록 한정 야코비안 벌점** ‖∂μ/∂x_blk‖²_F (Hutchinson 추정, create_graph).
  근거(1002 OSS 분석 1, 오프라인 가설): G17o 의 25 Hz 정지 떨림 = last_action 관측(165–224) 경유 2주기 자기루프 —
  G17o ‖J_last‖² 5.6 vs G12-9500 0.93. LCP(smooth-humanoid-locomotion ppo_rma.py:156-159) 는 전 관측 log-prob 기울기에 벌점을
  걸어 자이로 이득까지 깎을 수 있어, 여기서는 블록만 제약한다(GP_SL). GP_SL=None 이면 전 관측(LCP 류 대조).
  매 미니배치 optimizer.zero_grad() 직후 벌점 기울기를 먼저 누적하고, 이어지는 PPO loss.backward() 와 같은 clip·step 을 탄다.
"""
import torch
from rsl_rl.algorithms import PPO


class SmoothPPO(PPO):
    GP_COEF = 0.0
    GP_SL = slice(165, 225)
    GP_BS = 4096

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._gp_sum = 0.0; self._gp_n = 0
        zg = self.optimizer.zero_grad

        def zero_grad_gp(*aa, **kk):
            zg(*aa, **kk)
            if self.GP_COEF <= 0 or self.storage is None or self.storage.observations is None:
                return
            obs = self.storage.observations
            x = self.policy.get_actor_obs(obs.flatten(0, 1))
            idx = torch.randint(0, x.shape[0], (min(self.GP_BS, x.shape[0]),), device=x.device)
            x = x[idx].detach().clone().requires_grad_(True)
            mu = self.policy.actor(self.policy.actor_obs_normalizer(x))
            g, = torch.autograd.grad((mu * torch.randn_like(mu)).sum(), x, create_graph=True)
            gb = g if self.GP_SL is None else g[:, self.GP_SL]
            pen = gb.pow(2).sum(1).mean()
            (self.GP_COEF * pen).backward()
            self._gp_sum += float(pen.detach()); self._gp_n += 1

        self.optimizer.zero_grad = zero_grad_gp

    def update(self):
        d = super().update()
        if self._gp_n:
            d["jac_block_sq"] = self._gp_sum / self._gp_n
        self._gp_sum = 0.0; self._gp_n = 0
        return d
