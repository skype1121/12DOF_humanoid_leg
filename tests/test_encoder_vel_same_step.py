"""1001: 같은 스텝에 joint_vel(먼저)·joint_vel_ema(나중)가 둘 다 엔코더 속도를 부를 때의 동작 고정.

- legacy(기존 이름·G4~G15 학습 조건): 두 번째 호출이 속도 0 을 받는다 → joint_vel_ema 채널은 잡음뿐 (결함, 재현 보존용).
- v2: 스텝 캐시 → 두 호출 모두 실제 속도.
"""
import sys, types
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rl_walking import mdp_hang as m  # noqa: E402


def _env():
    class Data: pass
    class Asset:
        def __init__(s):
            s.data = Data(); s.data.joint_pos = torch.zeros(2, 3); s.data.default_joint_pos = torch.zeros(2, 3); s.actuators = {}
    class Term:
        def __init__(s, a): s.a = a; s._joint_ids = slice(None)
        @property
        def processed_actions(s): return s.a.data.joint_pos.clone()
    class AM:
        def __init__(s, a): s.t = Term(a)
        def get_term(s, n): return s.t
    a = Asset()
    env = types.SimpleNamespace(scene={"robot": a}, action_manager=AM(a), step_dt=0.02,
                                episode_length_buf=torch.tensor([5, 5]), common_step_counter=0, num_envs=2)
    return env, a, types.SimpleNamespace(name="robot", joint_ids=None)


def _run(vel_fn, ema_fn, steps=6):
    env, a, cfg = _env(); out = []
    for k in range(steps):
        a.data.joint_pos = torch.full((2, 3), 0.02 * (k + 1))            # 실제 1.0 rad/s
        env.common_step_counter = k + 1; env.episode_length_buf = torch.tensor([5 + k, 5 + k])
        v = vel_fn(env, asset_cfg=cfg); e = ema_fn(env, alpha=1.0, noise=0.0, asset_cfg=cfg)   # alpha 1 = EMA 가 입력 그대로
        out.append((float(v[0, 0]), float(e[0, 0])))
    return out


def test_legacy_second_call_sees_zero_velocity():
    out = _run(m.encoder_joint_vel, m.encoder_joint_vel_ema)
    assert all(abs(v - 1.0) < 1e-5 for v, _ in out[1:]), out          # 먼저 부른 joint_vel 은 정상
    assert all(abs(e) < 1e-9 for _, e in out), out                    # 나중 joint_vel_ema 는 항상 0 (결함)
    assert m.encoder_joint_vel_legacy is m.encoder_joint_vel


def test_v2_both_calls_see_true_velocity():
    out = _run(m.encoder_joint_vel_v2, m.encoder_joint_vel_ema_v2)
    assert all(abs(v - 1.0) < 1e-5 and abs(e - 1.0) < 1e-5 for v, e in out[1:]), out


def _lockstep(pos_fn, vel_fn, elbs, qs):
    env, a, cfg = _env(); out = []
    for k, (elb, q) in enumerate(zip(elbs, qs)):
        a.data.joint_pos = torch.full((2, 3), q)
        env.common_step_counter = k + 1; env.episode_length_buf = torch.tensor([elb, elb])
        p = pos_fn(env, asset_cfg=cfg); v = vel_fn(env, asset_cfg=cfg)
        out.append((float(p[0, 0]), float(v[0, 0])))
    return out


def test_v2_equals_legacy_pos_and_first_vel_except_reset_edge():
    """1001 (검토 지적: 'G16o = 관측 수정만' 확인): 유격 0·같은 궤적에서 joint_pos_rel·joint_vel(먼저 부르는 쪽) 은
    리셋 가장자리를 빼면 legacy 와 v2 가 같다. 알려진 차이 = 리셋 뒤 둘째 틱(elb==1): legacy 속도 0, v2 는 실제 차분."""
    elbs = [0, 1, 2, 3, 4, 5]
    qs = [0.10, 0.13, 0.17, 0.20, 0.26, 0.29]
    L = _lockstep(m.encoder_joint_pos_rel, m.encoder_joint_vel, elbs, qs)
    V = _lockstep(m.encoder_joint_pos_rel_v2, m.encoder_joint_vel_v2, elbs, qs)
    for k in range(len(elbs)):
        assert abs(L[k][0] - V[k][0]) < 1e-6, (k, L[k], V[k])            # 위치: 전 틱 동일
    assert L[0][1] == 0.0 and V[0][1] == 0.0                              # elb 0: 둘 다 0
    assert L[1][1] == 0.0 and abs(V[1][1] - 1.5) < 1e-4                   # elb 1: 알려진 차이 (0.03/0.02)
    for k in range(2, len(elbs)):
        assert abs(L[k][1] - V[k][1]) < 1e-4, (k, L[k], V[k])            # elb ≥ 2: 동일
