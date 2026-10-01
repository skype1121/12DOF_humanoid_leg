"""SmoothPPO: 블록 야코비안 벌점이 그 블록 이득을 줄이고(다른 블록은 덜), update 가 jac_block_sq 를 보고한다 (CPU, Isaac 없이)."""
import sys, torch
sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rl_walking.smooth_ppo import SmoothPPO


def _jac_sq(pol, x, sl):
    x = x.clone().requires_grad_(True); mu = pol.actor(x)
    J = torch.stack([torch.autograd.grad(mu[:, k].sum(), x, retain_graph=True)[0] for k in range(mu.shape[1])], 1)
    return float(J[:, :, sl].pow(2).sum((1, 2)).mean())


def test_block_penalty_reduces_block_gain():
    torch.manual_seed(0)
    N, T, D, A = 64, 8, 40, 4
    obs = TensorDict({"policy": torch.randn(N, D)}, batch_size=[N])
    pol = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, A, actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32])
    SmoothPPO.GP_COEF = 1.0; SmoothPPO.GP_SL = slice(10, 20); SmoothPPO.GP_BS = 256
    alg = SmoothPPO(pol, num_learning_epochs=1, num_mini_batches=2, learning_rate=1e-3, schedule="fixed")
    alg.init_storage("rl", N, T, obs, [A])
    x0 = torch.randn(256, D)
    b0, o0 = _jac_sq(pol, x0, slice(10, 20)), _jac_sq(pol, x0, slice(20, 30))
    for it in range(30):
        for t in range(T):
            o = TensorDict({"policy": torch.randn(N, D)}, batch_size=[N])
            alg.act(o)
            alg.process_env_step(o, torch.zeros(N), torch.zeros(N, dtype=torch.long), {})
        alg.compute_returns(o)
        d = alg.update()
    assert "jac_block_sq" in d
    b1, o1 = _jac_sq(pol, x0, slice(10, 20)), _jac_sq(pol, x0, slice(20, 30))
    print(f"블록 {b0:.4f}→{b1:.4f} · 다른 블록 {o0:.4f}→{o1:.4f}")
    assert b1 < 0.5 * b0
    assert b1 / b0 < o1 / o0


if __name__ == "__main__":
    test_block_penalty_reduces_block_gain(); print("OK")
