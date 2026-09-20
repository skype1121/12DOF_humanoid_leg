"""G7 항 단위 테스트 (Isaac 없이): 온셋 인계 가중치, 외란 프로파일, 외란 창 판정. 0921."""
import math
import sys

import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
from rl_walking.onset import onset_alpha  # noqa: E402
from rl_walking import mdp_hang  # noqa: E402


def test_onset_alpha_shape_and_monotone():
    hold = torch.tensor([0.5, 0.0, 1.0]); blend = torch.tensor([1.0, 1.5, 0.0])
    a0 = onset_alpha(torch.tensor([0.2, 0.0, 0.5]), hold, blend)
    assert torch.allclose(a0, torch.tensor([0.0, 0.0, 0.0]))                       # 홀드 중 0 (blend 0 이면 hold 전 0)
    amid = onset_alpha(torch.tensor([1.0, 0.75, 1.0]), hold, blend)
    assert abs(float(amid[0]) - 0.5) < 1e-6 and abs(float(amid[1]) - 0.5) < 1e-6   # 블렌딩 중간 = 0.5
    assert float(amid[2]) == 1.0                                                      # blend 0: hold 뒤 즉시 1
    a1 = onset_alpha(torch.tensor([2.0, 2.0, 2.0]), hold, blend)
    assert torch.allclose(a1, torch.ones(3))
    ts = torch.linspace(0, 3, 301)
    seq = onset_alpha(ts, torch.full_like(ts, 0.5), torch.full_like(ts, 1.0))
    assert bool((seq[1:] >= seq[:-1] - 1e-7).all())                                   # 단조 증가
    assert float(seq[51] - seq[50]) < 1e-3 and float(seq[150] - seq[149]) < 1e-3     # 양 끝 기울기 ≈ 0 (부드러운 출발/도착)


def test_dist_profile_ramp_and_impulse():
    kind = torch.tensor([[1.0, 2.0, 0.0]]); t0 = torch.tensor([[2.0, 6.0, 0.0]]); rise = torch.tensor([[2.0, 1.0, 1.0]]); hold = torch.tensor([[1.0, 0.3, 0.0]])
    def prof(t):
        return mdp_hang._dist_profile(torch.tensor([[t]]), kind, t0, rise, hold)[0]
    assert torch.allclose(prof(1.0), torch.tensor([0.0, 0.0, 0.0]))
    assert abs(float(prof(3.0)[0]) - 0.5) < 1e-6                 # 램프 중간
    assert float(prof(4.5)[0]) == 1.0                            # 유지 구간
    assert float(prof(5.1)[0]) == 0.0                            # 해제
    assert float(prof(6.1)[1]) == 1.0 and float(prof(6.4)[1]) == 0.0   # 충격 0.3 s
    assert float(prof(9.0)[2]) == 0.0                            # kind 0 = 없음


class _FakeEnv:
    def __init__(self, N=4, dev="cpu"):
        self.num_envs = N; self.device = dev; self.step_dt = 0.02


def test_dist_window_and_end_time():
    env = _FakeEnv()
    st = mdp_hang._state(env)
    st["dst_kind"][0, 0] = 1.0; st["dst_t0"][0, 0] = 2.0; st["dst_rise"][0, 0] = 2.0; st["dst_hold"][0, 0] = 1.0    # 2~5 s 램프
    st["dst_kind"][1, 0] = 2.0; st["dst_t0"][1, 0] = 4.0; st["dst_hold"][1, 0] = 0.3                                 # 4~4.3 충격
    ends = mdp_hang._dist_end_time(st)
    assert abs(float(ends[0, 0]) - 5.0) < 1e-6 and abs(float(ends[1, 0]) - 4.3) < 1e-6
    st["t"][:] = 1.0
    assert not bool(mdp_hang.dist_window_mask(env, 3.0).any())
    st["t"][:] = 3.0
    m = mdp_hang.dist_window_mask(env, 3.0); assert bool(m[0]) and not bool(m[1]) and not bool(m[2])
    st["t"][:] = 7.0
    m = mdp_hang.dist_window_mask(env, 3.0); assert bool(m[0]) and bool(m[1])      # 해제 뒤 3 s 안
    st["t"][:] = 8.5
    m = mdp_hang.dist_window_mask(env, 3.0); assert not bool(m[0]) and not bool(m[1])


def test_dist_force_vector_direction():
    env = _FakeEnv(N=1)
    st = mdp_hang._state(env)
    st["dst_kind"][0, 0] = 2.0; st["dst_t0"][0, 0] = 1.0; st["dst_hold"][0, 0] = 0.5; st["dst_F"][0, 0] = 10.0
    st["dst_dir"][0, 0] = torch.tensor([0.0, -1.0])
    st["t"][:] = 1.2
    F = mdp_hang._dist_force_w(st, torch.tensor([0]))
    assert torch.allclose(F, torch.tensor([[0.0, -10.0]]))
    st["t"][:] = 1.6
    assert torch.allclose(mdp_hang._dist_force_w(st, torch.tensor([0])), torch.zeros(1, 2))


def test_quat_apply_matches_rotation():
    # yaw 90° : x → y
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    v = mdp_hang._quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(v, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


if __name__ == "__main__":
    for fn in (test_onset_alpha_shape_and_monotone, test_dist_profile_ramp_and_impulse, test_dist_window_and_end_time,
               test_dist_force_vector_direction, test_quat_apply_matches_rotation):
        fn(); print("PASS", fn.__name__)
