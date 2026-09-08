"""Hanging-to-Stand 보조지지(assisted support) MDP 항 — 0908 저녁.

**왜 이 환경인가.** 0908 실기 4회: 전관절 0° kp150 홀드는 체중 68~72% 에서 뒤꿈치 강체 회전으로
넘어진다(관절 오차 ≤0.3°, 토크 ≤0.46Nm — 발목이 휘는 게 아니라 CoP 가 발 밖). 줄을 완전히 풀어도
못 선다(승윤님). 시뮬 대조군: 명목 홀드는 CoM −0.03m 에서 64/64 같은 식으로 넘어지고, v18b 정책은
그 CoM 에서 41% 만 선다. 즉 **고정 자세→핸드오프 프로토콜 자체가 이 로봇에선 성립하지 않는다.**
정책이 매달림→접지→하중 이전→자립을 하나의 에피소드로 배워야 한다.

**윈치는 모델링하지 않는다.** base 에 세계좌표 수직 힘 F = r(t)·m·g 를 건다(로프 등가 지지력).
  r(t): 에피소드마다 r0 ∈ r0_range → t_hold 동안 유지 → t_rel 동안 선형으로 0 → 이후 자유 기립.
  실물 줄의 흔들림·후크 위치 = (a) 수평 성분: 로프 길이 L 의 진자 복원력 −(F/L)·(xy − xy0) + 작은 상수
  편향, (b) 작용점 오프셋(로컬 x,y ±point_xy) → 힘×오프셋 = 몸통 피치/롤 모멘트(0908 두 줄 아이볼트
  기하 재현). 힘은 base 링크 프레임으로 직접 회전해 넣는다(is_global 캐시 의존 회피).

**정책 관측에는 넣지 않는다** — 실물에서 지지비를 못 잰다. 보상·평가·(향후) 크리틱에서만 쓴다.
정책은 발접촉·중력·관절 상태의 변화로 전이를 스스로 감지해야 한다(관측 245 계약 불변 = v18b 웜스타트 가능).
"""
from __future__ import annotations

import torch

try:  # 셀프테스트는 Isaac 없이 돈다 — 무거운 import 는 지연
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply_inverse as _quat_inv
except ImportError:  # pragma: no cover
    SceneEntityCfg = None
    _quat_inv = None


def _base_cfg():
    return SceneEntityCfg("robot", body_names="base")


def _feet_cfg():
    return SceneEntityCfg("contact_forces")

G = 9.81


def _state(env):
    """env 에 붙는 per-env 스케줄/상태 텐서 (resample_calib_bias 와 같은 패턴)."""
    st = getattr(env, "_hang", None)
    if st is None or st["r0"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device

        def z():
            return torch.zeros(N, device=dev)

        st = {"r0": z(), "t_hold": z(), "t_rel": z(), "t": z(), "r": z(), "mass": z(),
              "bx": z(), "by": z(), "px": z(), "py": z(), "L": torch.ones(N, device=dev),
              "x0": z(), "y0": z(),
              "released_t": torch.full((N,), -1.0, device=dev),   # r 가 0 에 닿은 시각 (−1 = 아직)
              "touch_t": torch.full((N,), -1.0, device=dev)}      # 양발 첫 동시 접촉 시각
        env._hang = st
    return st


def _ratio(st):
    t = st["t"]
    dec = (t - st["t_hold"]) / st["t_rel"].clamp(min=1e-3)
    r = torch.where(t <= st["t_hold"], st["r0"], st["r0"] * (1.0 - dec))
    return r.clamp(min=0.0)


def _apply(env, st, ids, asset_cfg):
    """지지력을 base 링크 프레임으로 넣는다. ids: long tensor."""
    asset = env.scene[asset_cfg.name]
    r = st["r"][ids]
    Fz = r * st["mass"][ids] * G                                    # 수직 지지력 [N], 세계 +z
    pos = asset.data.root_pos_w[ids]
    # 진자 복원력: 로프 길이 L 인 진자, 수평 변위에 비례 (F/L). 힘이 0 이면 자동으로 0.
    kx = Fz / st["L"][ids]
    Fx = -kx * (pos[:, 0] - st["x0"][ids]) + Fz * st["bx"][ids]
    Fy = -kx * (pos[:, 1] - st["y0"][ids]) + Fz * st["by"][ids]
    F_w = torch.stack([Fx, Fy, Fz], dim=-1)                          # (n,3) 세계
    F_b = _quat_inv(asset.data.root_quat_w[ids], F_w)               # base 링크 프레임
    forces = F_b.unsqueeze(1)                                       # (n,1,3)
    positions = torch.stack([st["px"][ids], st["py"][ids], torch.zeros_like(Fz)], dim=-1).unsqueeze(1)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torch.zeros_like(forces), positions=positions,
        body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)


def hang_support_reset(env, env_ids, r0_range=(0.5, 0.7), hold_range=(0.5, 2.0), release_range=(2.0, 8.0),
                       rope_len_range=(1.0, 2.0), bias_frac=0.01, point_xy=0.03, asset_cfg=None):
    """mode="reset": 에피소드마다 지지 스케줄·로프 기하를 새로 뽑고 첫 지지력을 즉시 건다."""
    asset_cfg = asset_cfg or _base_cfg()
    st = _state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = len(ids)

    def U(lo, hi):
        return torch.empty(n, device=env.device).uniform_(float(lo), float(hi))

    st["r0"][ids] = U(*r0_range)
    st["t_hold"][ids] = U(*hold_range)
    st["t_rel"][ids] = U(*release_range)
    st["L"][ids] = U(*rope_len_range)
    st["bx"][ids] = U(-bias_frac, bias_frac)
    st["by"][ids] = U(-bias_frac, bias_frac)
    st["px"][ids] = U(-point_xy, point_xy)
    st["py"][ids] = U(-point_xy, point_xy)
    st["t"][ids] = 0.0
    st["released_t"][ids] = -1.0
    st["touch_t"][ids] = -1.0
    asset = env.scene[asset_cfg.name]
    # 총질량: startup 질량 DR(add_base_mass) 반영. PhysX 뷰 get_masses() 는 CPU readback 이라
    # 매 리셋마다 부르면 4096env 학습이 ~7it/min 로 느려진다(0908 실측) — startup 뒤 불변이므로 1회 캐시.
    cache = getattr(env, "_hang_mass_all", None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = asset.root_physx_view.get_masses().sum(1).to(env.device)
        env._hang_mass_all = cache
    st["mass"][ids] = cache[ids]
    pos = asset.data.root_pos_w[ids]
    st["x0"][ids] = pos[:, 0]
    st["y0"][ids] = pos[:, 1]
    st["r"][ids] = st["r0"][ids]
    _apply(env, st, ids, asset_cfg)


def hang_support_step(env, env_ids, asset_cfg=None, sensor_cfg=None, threshold: float = 5.0):
    """mode="interval", interval_range_s=(0,0): 매 스텝 지지비 갱신·힘 적용·전이 시각 기록."""
    asset_cfg = asset_cfg or _base_cfg()
    sensor_cfg = sensor_cfg or _feet_cfg()
    st = _state(env)
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    st["t"][ids] += env.step_dt
    st["r"] = _ratio(st)
    newly = (st["r"] <= 0.0) & (st["released_t"] < 0.0)
    st["released_t"][newly] = st["t"][newly]
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    nt = both & (st["touch_t"] < 0.0)
    st["touch_t"][nt] = st["t"][nt]
    _apply(env, st, ids, asset_cfg)


# ───────────────────────────── 보상 ─────────────────────────────

def hang_ratio(env) -> torch.Tensor:
    return _state(env)["r"]


def survive_after_release(env) -> torch.Tensor:
    """지지 0 이후 살아 있는 매 스텝 +1 — '자유 기립 생존' 을 강하게."""
    return (_state(env)["r"] <= 0.0).to(torch.float32)


def hang_quiet(env, r_min: float = 0.5, asset_cfg=None) -> torch.Tensor:
    """매달린 동안(r > r_min) 관절속도 제곱합 — 공중 발버둥 억제 (0803 사고 클래스)."""
    st = _state(env)
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    w = (st["r"] > r_min).to(torch.float32)
    return w * torch.sum(asset.data.joint_vel ** 2, dim=1)


def feet_contact_gated(env, sensor_cfg=None, threshold: float = 5.0) -> torch.Tensor:
    """양발 접촉 보상 × (1 − r): 지지가 클수록 접촉을 요구하지 않는다 — 공중에서 발 뻗기 유도 금지."""
    st = _state(env)
    sensor_cfg = sensor_cfg or _feet_cfg()
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1).to(torch.float32)
    return both * (1.0 - st["r"]).clamp(min=0.0)


def selftest():
    """ROS/Isaac 없이 스케줄 함수만 검증."""
    dev = "cpu"
    st = {"r0": torch.tensor([0.6, 1.0], device=dev), "t_hold": torch.tensor([1.0, 2.0]),
          "t_rel": torch.tensor([4.0, 2.0]), "t": torch.tensor([0.0, 0.0])}
    assert torch.allclose(_ratio(st), torch.tensor([0.6, 1.0]))
    st["t"] = torch.tensor([1.0, 2.0]); assert torch.allclose(_ratio(st), torch.tensor([0.6, 1.0]))
    st["t"] = torch.tensor([3.0, 3.0]); assert torch.allclose(_ratio(st), torch.tensor([0.3, 0.5]))
    st["t"] = torch.tensor([5.0, 4.0]); assert torch.allclose(_ratio(st), torch.tensor([0.0, 0.0]))
    st["t"] = torch.tensor([9.0, 9.0]); assert torch.allclose(_ratio(st), torch.tensor([0.0, 0.0])), "0 아래로 안 감"
    _selftest_posture()
    print("[selftest] mdp_hang: 지지비 스케줄(유지·선형방출·하한0)·안정 자세 항 — ALL PASS")


if __name__ == "__main__":
    selftest()


# ───────────────── 안정 자세 보상 (0908 21:00, 승윤님 "좀 안정적인 자세면 좋겠는데") ─────────────────
# 실물 유사 조건의 stand_hang_v1 종료 자세가 비대칭(왼힙굽힘 −18.7 vs 오른 +8.5, 발목 +17.3 vs −6.1)·엇갈림(앞뒤 33mm)이라
# 흔들려 보인다. 토크·클램프는 문제 없음(포화 0%). 보상에 좌우 대칭·발 배치 밴드가 없어서 생긴 자세 → 두 항을 추가한다.
_PAIRS = ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r")
_PAIR_SIGN = {"hip_f": 1.0, "hip_a": 1.0, "hip_r": 1.0, "knee": 1.0, "ankle_f": 1.0, "ankle_r": -1.0}  # symmetry.py: ankle_r 만 동부호


def _pair_index(env, asset_cfg):
    key = "_hang_pairs"
    cache = getattr(env, key, None)
    if cache is None:
        names = list(env.scene[asset_cfg.name].data.joint_names)
        li, ri, sg = [], [], []
        for p in _PAIRS:
            li.append(names.index(f"left_{p}_joint")); ri.append(names.index(f"right_{p}_joint")); sg.append(_PAIR_SIGN[p])
        cache = (torch.tensor(li, device=env.device), torch.tensor(ri, device=env.device), torch.tensor(sg, device=env.device))
        setattr(env, key, cache)
    return cache


def pose_symmetry(env, asset_cfg=None) -> torch.Tensor:
    """좌우 대칭 편차 Σ(q_L + s·q_R)² [rad²] — s=+1(부호 반대 5쌍: 대칭이면 합 0), ankle_r 은 s=−1(동부호: 차 0)."""
    asset_cfg = asset_cfg or _base_cfg()
    q = env.scene[asset_cfg.name].data.joint_pos
    li, ri, sg = _pair_index(env, asset_cfg)
    return torch.sum((q[:, li] + sg * q[:, ri]) ** 2, dim=1)


def stance_shape(env, gap_range=(0.22, 0.32), max_stagger: float = 0.04, asset_cfg=None) -> torch.Tensor:
    """발 배치 벌점: 좌우 간격이 밴드 밖이면 초과분, 앞뒤 어긋남이 max_stagger 넘으면 초과분 [m] (진행방향 기준 투영)."""
    asset_cfg = asset_cfg or _feet_body_cfg()
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    d = feet[:, 0, :] - feet[:, 1, :]
    h = asset.data.heading_w
    lat = torch.abs(-torch.sin(h) * d[:, 0] + torch.cos(h) * d[:, 1])
    lon = torch.abs(torch.cos(h) * d[:, 0] + torch.sin(h) * d[:, 1])
    lo, hi = gap_range
    return (lo - lat).clamp(min=0.0) + (lat - hi).clamp(min=0.0) + (lon - max_stagger).clamp(min=0.0)


def _feet_body_cfg():
    return SceneEntityCfg("robot", body_names=[".*_ankle_r_joint"])


def _selftest_posture():
    class _A:  # 최소 더미 — joint_names·joint_pos 만
        pass
    q = torch.zeros(2, 12)
    names = [f"{s}_{p}_joint" for p in _PAIRS for s in ("left", "right")]
    li = torch.tensor([names.index(f"left_{p}_joint") for p in _PAIRS]); ri = torch.tensor([names.index(f"right_{p}_joint") for p in _PAIRS])
    sg = torch.tensor([_PAIR_SIGN[p] for p in _PAIRS])
    # env0: 대칭 자세(hip_f L=+0.2,R=−0.2 · ankle_r L=R=+0.1) → 0 ; env1: 비대칭
    q[0, names.index("left_hip_f_joint")] = 0.2; q[0, names.index("right_hip_f_joint")] = -0.2
    q[0, names.index("left_ankle_r_joint")] = 0.1; q[0, names.index("right_ankle_r_joint")] = 0.1
    q[1, names.index("left_hip_f_joint")] = 0.2; q[1, names.index("right_hip_f_joint")] = 0.0
    val = torch.sum((q[:, li] + sg * q[:, ri]) ** 2, dim=1)
    assert abs(val[0]) < 1e-9 and abs(val[1] - 0.04) < 1e-9, val
    # stance_shape 수학: 간격 0.27(밴드 안)·어긋남 0 → 0 ; 간격 0.36 → 0.04 ; 어긋남 0.10 → 0.06
    lo, hi, ms = 0.22, 0.32, 0.04
    f = lambda lat, lon: max(0, lo - lat) + max(0, lat - hi) + max(0, lon - ms)
    assert f(0.27, 0.0) == 0 and abs(f(0.36, 0.0) - 0.04) < 1e-9 and abs(f(0.27, 0.10) - 0.06) < 1e-9
    print("[selftest] 안정 자세 항: 대칭 부호(ankle_r 동부호)·발 배치 밴드 — PASS")
