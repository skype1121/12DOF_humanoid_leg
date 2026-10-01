"""1001 G17 관측 계약: 자이로 env 별 지연, 접촉 10 Hz 유지·전송 지연·발 기울기 꺼짐 (Isaac 없이)."""
import sys, types, math
import torch
sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
from rl_walking import mdp_hang as m  # noqa: E402
# Isaac 없이: isaaclab.utils.math.quat_apply_inverse 와 mdp_stage4 를 최소 스텁으로
def _qinv(q, v):
    w, x, y, z = q.unbind(-1); u = torch.stack([-x, -y, -z], -1)
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(u, t, dim=-1)
for _n in ("isaaclab", "isaaclab.utils", "isaaclab.utils.math"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["isaaclab.utils.math"].quat_apply_inverse = _qinv
s4 = types.ModuleType("rl_walking.mdp_stage4"); s4.feet_contact_binary = None
sys.modules["rl_walking.mdp_stage4"] = s4


def _env(N=4):
    class D: pass
    a = types.SimpleNamespace(data=D())
    a.data.root_ang_vel_b = torch.zeros(N, 3)
    a.data.body_quat_w = torch.tensor([1.0, 0, 0, 0]).repeat(N, 2, 1)
    a.find_bodies = lambda names, preserve_order=True: ([0, 1], names)
    env = types.SimpleNamespace(scene={"robot": a}, common_step_counter=0, episode_length_buf=torch.full((N,), 5))
    return env, a, types.SimpleNamespace(name="robot")


def test_gyro_lag_per_env():
    env, a, cfg = _env()
    out = []
    for k in range(8):
        env.common_step_counter = k + 1
        a.data.root_ang_vel_b = torch.full((4, 3), float(k))
        o = m.base_ang_vel_lagged(env, max_lag=3, asset_cfg=cfg)
        assert torch.equal(o, m.base_ang_vel_lagged(env, max_lag=3, asset_cfg=cfg))   # 같은 스텝 재호출 동일
        out.append(o[:, 0])
    lag = env._gyro_lag_st["lag"]
    for e in range(4):
        assert abs(float(out[-1][e]) - (7 - int(lag[e]))) < 1e-6


def _qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return torch.tensor([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


# 발 링크 +Y = 위 → 평평하게 선 발 = 월드 X 축 +90° 회전 (1002 fcr_diag: 정착 서기 발 좌표 중력 (0,−1.00,0))
FLAT = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])


def test_contact_hold_delay_and_tilt():
    env, a, cfg = _env(N=2)
    a.data.body_quat_w = FLAT.repeat(2, 2, 1)
    force = {"v": torch.ones(2, 2)}
    s4_orig = s4.feet_contact_binary
    s4.feet_contact_binary = lambda env_, sc, thr, d, tr: force["v"].clone()
    try:
        sc = types.SimpleNamespace(body_names=["L", "R"])
        seq = []
        for k in range(20):
            env.common_step_counter = k + 1
            if k == 8:
                force["v"] = torch.zeros(2, 2)          # 접촉 상실
            seq.append(m.feet_contact_real(env, sc, hold_ticks=5, extra_delay=1, tilt_off_deg=(3.0, 3.0), asset_cfg=cfg)[:, 0].clone())
        s = torch.stack(seq)                              # (20, 2)
        assert float(s[2:8].min()) == 1.0                 # 평평·힘 접촉 = 1 (1차 구현 결함의 회귀 시험: 스폰 자세 무관)
        for e in range(2):
            first0 = int((s[:, e] == 0).nonzero()[0])
            assert 9 <= first0 <= 8 + 5 + 1, (e, first0)  # 표본 유지(≤5틱) + 전송 1틱 안에 반영
        # 기울기: 발을 10° 기울이면 힘이 있어도 꺼짐
        force["v"] = torch.ones(2, 2)
        ang = math.radians(10.0)
        qy = torch.tensor([math.cos(ang / 2), 0, math.sin(ang / 2), 0])
        a.data.body_quat_w = _qmul(qy, FLAT).repeat(2, 2, 1)   # 평평 발을 월드 Y 축으로 10° 기울임
        for k in range(20, 32):
            env.common_step_counter = k + 1
            o = m.feet_contact_real(env, sc, hold_ticks=5, extra_delay=1, tilt_off_deg=(3.0, 3.0), asset_cfg=cfg)
        assert float(o.sum()) == 0.0
    finally:
        s4.feet_contact_binary = s4_orig
