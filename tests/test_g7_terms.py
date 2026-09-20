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


def test_recovery_gate_state_based():
    """G8 상태 기반 게이트: 외란 없이 기울기만으로도 열리고, 실제 안정된 뒤에만 닫힌다(시간 기준 아님)."""
    import types
    env = _FakeEnv(N=3)
    st = mdp_hang._state(env)
    tilt = [0.0, 0.0, 0.0]; w = [0.0, 0.0, 0.0]; both = [True, True, True]
    mdp_hang._balance_state = lambda e, *a, **k: (torch.tensor(both), torch.tensor(tilt), torch.tensor(w), torch.zeros(3), torch.zeros(3))
    step = lambda: mdp_hang.recovery_gate_step(env, None, tilt_ok=5.0, w_ok=0.35, v_ok=0.12, settle_s=0.3, max_open_s=8.0)
    step(); assert float(st["rec_open"][0]) == 0.0                      # 평온 → 닫힘
    tilt = [8.0, 0.0, 0.0]; step(); assert float(st["rec_open"][0]) == 1.0   # 외란 이벤트 없이 기울기만으로 열림
    tilt = [0.0, 0.0, 0.0]
    for _ in range(10):        # 0.2 s 정상 — 아직 settle_s(0.3) 미만
        step()
    assert float(st["rec_open"][0]) == 1.0 and float(st["rec_ev"][0]) == 0.0
    for _ in range(6):         # 0.32 s 누적 → 닫힘 + 회복 이벤트 1
        step()
    assert float(st["rec_open"][0]) == 0.0 and float(st["rec_ev"][0]) == 1.0
    st["dst_F_now"][1] = 12.0; step()
    assert float(st["rec_open"][1]) == 1.0                              # 외란 힘이 걸려 있으면 열림


def test_recovery_gate_max_open_guard():
    """계속 움직여 벌점을 피하는 전략 차단: 연속 열림이 max_open_s 를 넘으면 강제로 닫힌다."""
    env = _FakeEnv(N=1)
    st = mdp_hang._state(env); st["rec_open"][:] = 0.0; st["rec_open_t"][:] = 0.0; st["rec_settle_run"][:] = 0.0
    mdp_hang._balance_state = lambda e, *a, **k: (torch.tensor([True]), torch.tensor([20.0]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    for _ in range(int(2.0 / 0.02)):
        mdp_hang.recovery_gate_step(env, None, max_open_s=1.0)
    assert float(st["rec_open"][0]) == 0.0


if __name__ == "__main__":
    for fn in (test_onset_alpha_shape_and_monotone, test_dist_profile_ramp_and_impulse, test_dist_window_and_end_time,
               test_dist_force_vector_direction, test_quat_apply_matches_rotation,
               test_recovery_gate_state_based, test_recovery_gate_max_open_guard):
        fn(); print("PASS", fn.__name__)
