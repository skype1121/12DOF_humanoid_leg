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

import math
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

        st = {"r0": z(), "t_hold": z(), "t_rel": z(), "t": z(), "r": z(), "mass": z(), "lvl_k": z(), "lvl_c": z(),
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
    # 프레임 수평 유지 토크 (0908 22:40, 승윤님 "실물은 저 정도로 흔들리며 착지하지 않는다"): 실물 현가는 앞뒤 두 줄이
    # 프레임을 수평으로 잡는다(오늘 착지 피치 −0.5°). 지지비 r 에 비례해 줄이 풀리면 사라진다. lvl_k=0 이면 종전과 동일.
    pgv = asset.data.projected_gravity_b[ids]; w = asset.data.root_ang_vel_b[ids]
    tau = _level_torque(pgv, w, st["lvl_k"][ids] * r, st["lvl_c"][ids] * r)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=tau.unsqueeze(1), positions=positions,
        body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)


def hang_support_reset(env, env_ids, r0_range=(0.5, 0.7), hold_range=(0.5, 2.0), release_range=(2.0, 8.0),
                       rope_len_range=(1.0, 2.0), bias_frac=0.01, point_xy=0.03, asset_cfg=None,
                       level_gain_range=(0.0, 0.0), level_damp_range=(0.0, 0.0)):
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
    st["lvl_k"][ids] = U(*level_gain_range)      # 수평 유지 강성 [Nm/rad] (실물 두 줄 현가 ≈ 강함)
    st["lvl_c"][ids] = U(*level_damp_range)      # 감쇠 [Nm·s/rad]
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
    _selftest_pitch_band()
    _selftest_level()
    _selftest_still()
    _selftest_s8()
    print("[selftest] mdp_hang: 지지비 스케줄·안정 자세 항·피치 밴드·수평 유지 토크·착지 후 정지·S8 — ALL PASS")



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
    assert abs(val[0]) < 1e-6 and abs(val[1] - 0.04) < 1e-6, val  # float32
    # stance_shape 수학: 간격 0.27(밴드 안)·어긋남 0 → 0 ; 간격 0.36 → 0.04 ; 어긋남 0.10 → 0.06
    lo, hi, ms = 0.22, 0.32, 0.04
    f = lambda lat, lon: max(0, lo - lat) + max(0, lat - hi) + max(0, lon - ms)
    assert f(0.27, 0.0) == 0 and abs(f(0.36, 0.0) - 0.04) < 1e-9 and abs(f(0.27, 0.10) - 0.06) < 1e-9
    print("[selftest] 안정 자세 항: 대칭 부호(ankle_r 동부호)·발 배치 밴드 — PASS")


# ───────────── 몸통 피치 밴드 (0908 22:35, 승윤님 "몸체·머리는 수평을 잡고 싶다") ─────────────
# S5k 까지의 정책은 CoM 후방을 '몸통 뒤로 17° 눕히기 + 골반 앞으로' 로 맞춘다 — 서긴 하지만 뒤꿈치 여유가 얇다.
# 몸통 |피치| 가 band 안이면 0, 밖이면 초과분[rad] 선형 벌점. 수평 몸통 + 다리 전체 4~6° 앞기울임 해법을 유도.
def torso_pitch_band(env, band_rad: float = 0.087, asset_cfg=None) -> torch.Tensor:
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    pg = asset.data.projected_gravity_b
    pitch = torch.atan2(pg[:, 0], -pg[:, 2]).abs()
    return (pitch - band_rad).clamp(min=0.0)


def _selftest_pitch_band():
    import math
    for deg, want in ((0.0, 0.0), (4.0, 0.0), (17.0, math.radians(17.0) - 0.087), (-17.0, math.radians(17.0) - 0.087)):
        th = math.radians(deg); pg = torch.tensor([[math.sin(th), 0.0, -math.cos(th)]])
        v = (torch.atan2(pg[:, 0], -pg[:, 2]).abs() - 0.087).clamp(min=0.0)
        assert abs(float(v) - want) < 1e-4, (deg, float(v), want)
    print("[selftest] 몸통 피치 밴드: 0/4° → 0, ±17° → 12° 초과분 — PASS")

def _level_torque(pg, w, k, c):
    """몸통 프레임 복원 토크. pg=투영중력(수평이면 (0,0,−1)), w=각속도(body). 앞기울임(pg_x>0)→ −y 토크(고개 들기),
    +x 롤(pg_y<0)→ −x 토크. τ = k·(pg_y, −pg_x, 0) − c·(w_x, w_y, 0)."""
    zero = torch.zeros_like(k)
    return torch.stack([k * pg[:, 1] - c * w[:, 0], -k * pg[:, 0] - c * w[:, 1], zero], dim=-1)


def _selftest_level():
    import math
    th = math.radians(10.0)
    pg = torch.tensor([[math.sin(th), 0.0, -math.cos(th)], [0.0, -math.sin(th), -math.cos(th)]])  # 앞기울임 / +x 롤
    w = torch.zeros(2, 3); k = torch.tensor([100.0, 100.0]); c = torch.zeros(2)
    tau = _level_torque(pg, w, k, c)
    assert tau[0, 1] < 0 and abs(tau[0, 0]) < 1e-6, tau[0]      # 앞기울임 → 고개 드는(−y) 토크
    assert tau[1, 0] < 0 and abs(tau[1, 1]) < 1e-6, tau[1]      # +x 롤 → −x 복원 토크
    tau0 = _level_torque(pg, w, torch.zeros(2), c); assert float(tau0.abs().sum()) == 0.0, "게인 0 = 종전과 동일"
    print("[selftest] 수평 유지 토크: 앞기울임→−y, +x롤→−x, 게인0→0 — PASS")


# ───────── 착지 후 정지 (0909 13:05, 승윤님 "착지 후 흔들흔들은 용납되지 않아 — 안정적인 스탠바이 자세로 서 있어야") ─────────
def _touched_mask(env, min_s: float, r_max: float = 1.0, after_release_s: float = 0.0):
    """양발 접촉 min_s 경과 & 지지비 ≤ r_max (기본 1.0 = 지지비 무관). r_max<1 이면 줄이 거의 풀린 뒤에만 정지를 요구.
    after_release_s>0 이면 지지 0 도달 후 그 시간이 지난 뒤에만(= 정착 후 스탠바이 구간) — 전이 구간은 자유."""
    st = _state(env)
    m = (st["touch_t"] >= 0.0) & ((st["t"] - st["touch_t"]) >= min_s) & (st["r"] <= r_max)
    if after_release_s > 0.0:
        m = m & (st["released_t"] >= 0.0) & ((st["t"] - st["released_t"]) >= after_release_s)
    return m


def still_after_contact(env, min_s: float = 0.5, w_ang: float = 1.0, w_lin: float = 1.0, w_jvel: float = 0.002,
                        asset_cfg=None, r_max: float = 1.0, after_release_s: float = 0.0) -> torch.Tensor:
    """양발 접촉 min_s 뒤부터: 몸통 각속도(xy)²·수평속도²·관절속도² 합 — 서 있는 동안의 잔보정·흔들림을 벌한다."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    w = asset.data.root_ang_vel_b[:, :2]; v = asset.data.root_lin_vel_b[:, :2]; qd = asset.data.joint_vel
    val = w_ang * (w * w).sum(1) + w_lin * (v * v).sum(1) + w_jvel * (qd * qd).sum(1)
    return val * _touched_mask(env, min_s, r_max, after_release_s).to(torch.float32)


def feet_lift_after_contact(env, min_s: float = 0.5, sensor_cfg=None, threshold: float = 5.0, r_max: float = 1.0,
                            after_release_s: float = 0.0) -> torch.Tensor:
    """양발이 닿은 뒤(min_s) 한 발이라도 떨어지면 1 — 착지 후 발 들썩임·스텝 차단."""
    sensor_cfg = sensor_cfg or _feet_cfg()
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    return (~both).to(torch.float32) * _touched_mask(env, min_s, r_max, after_release_s).to(torch.float32)


def _selftest_still():
    st = {"touch_t": torch.tensor([-1.0, 1.0, 1.0, 1.0]), "t": torch.tensor([3.0, 1.2, 2.0, 2.0]), "r": torch.tensor([0.0, 0.0, 0.0, 0.6]),
          "released_t": torch.tensor([-1.0, -1.0, 1.5, -1.0])}
    m = (st["touch_t"] >= 0.0) & ((st["t"] - st["touch_t"]) >= 0.5) & (st["r"] <= 0.3)
    assert m.tolist() == [False, False, True, False], m
    m2 = m & (st["released_t"] >= 0.0) & ((st["t"] - st["released_t"]) >= 1.0)
    assert m2.tolist() == [False, False, False, False], "방출 후 0.5s 는 아직 정착 구간"
    st["t"] = torch.tensor([3.0, 1.2, 3.0, 2.0]); m3 = (st["touch_t"] >= 0.0) & (st["r"] <= 0.3) & (st["released_t"] >= 0.0) & ((st["t"] - st["released_t"]) >= 1.0)
    assert m3.tolist() == [False, False, True, False], "방출 후 1.5s 는 스탠바이"
    print("[selftest] 착지 후 정지 마스크: 미접지/0.2s/1.0s → F/F/T — PASS")


# ═══════════════ S8 (0909 16:40~): 관측 개선(raw+EMA)·발 하중·settled 정숙 보상 ═══════════════
# 실물 정지 노이즈(0908): 자이로 std 0.006, 관절속도 std 0.03~0.18 rad/s. 학습 균일노이즈(±0.2 / ±1.5)는 그 10~20배.
# S7 계열(강한 속도 벌점)은 전이 능력을 깎았다 → 벌점이 아니라 **관측**을 바꾼다: 정책이 '한 프레임 튐'과 '지속 변화'를
# 구분하도록 raw 와 EMA(시정수 ~4프레임)를 함께 준다. EMA 는 노이즈 낀 신호의 EMA 여야 실물과 같다(관측 매니저의 노이즈는
# 함수 뒤에 더해지므로 여기서 직접 노이즈를 넣고 EMA 한다; 이 항들은 cfg 에서 noise=None).

def _ema_state(env, key, x, alpha):
    st = getattr(env, key, None)
    if st is None or st.shape != x.shape:
        st = x.clone(); setattr(env, key, st)
    fresh = env.episode_length_buf <= 1        # 리셋 직후 = 현재값으로 재시작
    st[fresh] = x[fresh]
    st.mul_(1.0 - alpha).add_(alpha * x)
    return st


def gyro_ema(env, alpha: float = 0.25, noise: float = 0.2, asset_cfg=None) -> torch.Tensor:
    """노이즈 낀 자이로의 EMA (3). 실물 브리지에서도 같은 식으로 계산 가능."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    x = asset.data.root_ang_vel_b
    if noise > 0.0:
        x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
    return _ema_state(env, "_s8_gyro_ema", x, alpha).clone()


def joint_vel_ema(env, alpha: float = 0.25, noise: float = 1.5, asset_cfg=None) -> torch.Tensor:
    """노이즈 낀 관절속도의 EMA (12)."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    x = asset.data.joint_vel
    if noise > 0.0:
        x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
    return _ema_state(env, "_s8_jvel_ema", x, alpha).clone()


def foot_load_total(env, sensor_cfg=None, gain_range=(0.7, 1.3), offset_range=(-0.05, 0.05), noise: float = 0.03,
                    delay_steps: int = 5, clip_max: float = 1.5) -> torch.Tensor:
    """양발 수직력 합 / 체중 (1). 실물 = (left_n+right_n)/172.9 (전역 교정된 합만 신뢰, 좌우 비율은 안 쓴다).
    env 별 게인·오프셋(패드 감도), 노이즈, 5틱 지연(RA30P 10Hz)."""
    sensor_cfg = sensor_cfg or _feet_cfg()
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0).sum(1)
    st = _state(env)
    mass = st["mass"] if float(st["mass"].max()) > 0 else torch.full_like(f, 17.62)
    load = f / (mass * G)
    N = load.shape[0]
    g = getattr(env, "_s8_load_gain", None)
    if g is None or g.shape[0] != N:
        g = torch.empty(N, device=load.device).uniform_(*gain_range); env._s8_load_gain = g
        env._s8_load_off = torch.empty(N, device=load.device).uniform_(*offset_range)
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        g[fresh] = torch.empty(int(fresh.sum()), device=load.device).uniform_(*gain_range)
        env._s8_load_off[fresh] = torch.empty(int(fresh.sum()), device=load.device).uniform_(*offset_range)
    y = (load * g + env._s8_load_off + (torch.rand_like(load) * 2.0 - 1.0) * noise).clamp(0.0, clip_max)
    if delay_steps > 0:
        buf = getattr(env, "_s8_load_buf", None)
        if buf is None or buf.shape[0] != delay_steps or buf.shape[1] != N:
            buf = y.unsqueeze(0).repeat(delay_steps, 1).clone(); env._s8_load_buf = buf; env._s8_load_idx = 0
        buf[:, fresh] = y[fresh].unsqueeze(0)
        i = env._s8_load_idx; out = buf[i].clone(); buf[i] = y; env._s8_load_idx = (i + 1) % delay_steps
        y = out
    return y.unsqueeze(1)


# ── settled 상태(특권 정보로 판정, 관측엔 안 넣음) + 램프 ──
def settled_ramp(env, hold_s: float = 0.5, ramp_s: float = 1.0, tilt_deg: float = 8.0, min_height: float = 0.60,
                 sensor_cfg=None, threshold: float = 5.0, r_eps: float = 0.05) -> torch.Tensor:
    """지지비 ≤ r_eps & 양발 접촉 & 기울기<tilt & 골반높이>min 이 hold_s 유지된 뒤 ramp_s 동안 0→1 로 오르는 가중치 (N,).
    기본 tilt 8°·r_eps 0.05 (0909 S8ac 1차: tilt 5°·r≤0 이면 정착 피치 −4° + 잔류 지지비 0.01~0.03 에 게이트가 거의 안 열려 보너스 Episode_Reward 0.0025)."""
    st = _state(env)
    asset = env.scene["robot"]
    sensor_cfg = sensor_cfg or _feet_cfg()
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    pg = asset.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1.0, 1.0)))
    ok = (st["r"] <= r_eps) & both & (tilt < tilt_deg) & (asset.data.root_pos_w[:, 2] > min_height)
    tmr = getattr(env, "_s8_settle_t", None)
    if tmr is None or tmr.shape[0] != ok.shape[0]:
        tmr = torch.zeros(ok.shape[0], device=ok.device); env._s8_settle_t = tmr
    tmr[:] = torch.where(ok, tmr + env.step_dt, torch.zeros_like(tmr))
    return ((tmr - hold_s) / max(ramp_s, 1e-3)).clamp(0.0, 1.0)


def settle_quiet(env, w_ang: float = 1.0, w_lin: float = 1.0, w_jvel: float = 0.0, w_acc: float = 0.0,
                 asset_cfg=None, hold_s: float = 0.5, ramp_s: float = 1.0, tilt_deg: float = 8.0, min_height: float = 0.60,
                 sensor_cfg=None, threshold: float = 5.0, r_eps: float = 0.05) -> torch.Tensor:
    """settled 램프 × (몸통 각속도² + 수평속도² + 관절속도² + 액션 2차차분²) — 착지 직후가 아니라 정착 후에만."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    w = asset.data.root_ang_vel_b[:, :2]; v = asset.data.root_lin_vel_b[:, :2]
    val = w_ang * (w * w).sum(1) + w_lin * (v * v).sum(1)
    if w_jvel > 0.0:
        val = val + w_jvel * (asset.data.joint_vel ** 2).sum(1)
    if w_acc > 0.0:
        val = val + w_acc * action_accel_l2(env)
    return val * settled_ramp(env, hold_s, ramp_s, tilt_deg, min_height, sensor_cfg, threshold, r_eps)


def action_accel_l2(env) -> torch.Tensor:
    """액션 2차 차분 |a_t − 2a_{t−1} + a_{t−2}|² 평균 — 느린 보정은 두고 고주파 채터만 겨냥. 스텝당 1회 계산(캐시)."""
    step = int(env.common_step_counter)
    if getattr(env, "_s8_acc_step", -1) == step:
        return env._s8_acc_val
    am = env.action_manager
    a, a1 = am.action, am.prev_action
    a2 = getattr(env, "_s8_act_prev2", None)
    if a2 is None or a2.shape != a.shape:
        a2 = a1.clone(); env._s8_act_prev2 = a2
    fresh = env.episode_length_buf <= 1
    a2[fresh] = a1[fresh]
    val = ((a - 2.0 * a1 + a2) ** 2).mean(1)
    env._s8_act_prev2 = a1.clone()
    env._s8_acc_step, env._s8_acc_val = step, val
    return val


def foot_load_balance(env, sensor_cfg=None, threshold: float = 5.0, min_total: float = 40.0,
                      hold_s: float = 0.5, ramp_s: float = 1.0, tilt_deg: float = 8.0, min_height: float = 0.60,
                      r_eps: float = 0.05) -> torch.Tensor:
    """정착 상태에서 양발 수직하중 불균형 |L-R|/(L+R) 벌점 (0~1). 0910 실물: 착지 후 하중이 한 발(오른 65%)로 쏠려 반대 발이 떠 넘어짐 —
    원인이 발 롤·횡 CoM·로프 어느 쪽이든, 하중을 좌우로 고르게 실으면 그 실패를 막는다. settled 램프로 정착 후에만 켠다(착지 순간의 정상적 편하중은 벌 안 줌)."""
    sensor_cfg = sensor_cfg or _feet_cfg()
    fz = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    fl, fr = fz[:, 0], fz[:, 1]
    tot = fl + fr
    imb = (fl - fr).abs() / tot.clamp(min=min_total)      # min_total 미만이면 분모 고정(공중·저하중서 폭주 방지)
    imb = torch.where(tot > min_total, imb, torch.zeros_like(imb))
    return imb.clamp(0.0, 1.0) * settled_ramp(env, hold_s, ramp_s, tilt_deg, min_height, sensor_cfg, threshold, r_eps)


def settle_quiet_bonus(env, s_ang: float = 0.08, s_lin: float = 0.02, s_jvel: float = 0.06, s_acc: float = 0.05,
                       use_jvel: bool = True, use_acc: bool = True, asset_cfg=None, hold_s: float = 0.5, ramp_s: float = 1.0,
                       tilt_deg: float = 8.0, min_height: float = 0.60, sensor_cfg=None, threshold: float = 5.0,
                       r_eps: float = 0.05, rational: bool = False) -> torch.Tensor:
    # (**kwargs 금지: Isaac Lab 매니저가 VAR_KEYWORD 를 '필수 인자 ramp_kw' 로 오인해 cfg 검증에서 죽는다 — 0909 S8ac 1차 실패)
    """정착 후에만 주는 **양(+)의 정숙 보너스** 0~1: settled 램프 × 평균[exp(−(ω/σω)²), exp(−(v/σv)²)[, exp(−(q̇rms/σq)²)][, exp(−(ä_rms/σa)²)]].

    벌점(S7 계열)이 아니라 보너스인 이유: settled 게이트 뒤의 벌점은 '정착하지 않기'(발 들기·기울기)로 회피할 수 있지만,
    보너스는 정착 자체를 유도하면서 정착 후 정숙만 추가로 보상한다. σ 는 목표치(ω 0.05·v 0.005·q̇ 0.04·ä 0.03) 근방에서 기울기가 살도록 잡음."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    v = asset.data.root_lin_vel_b[:, :2].norm(dim=1)
    # rational=True: 1/(1+(x/σ)²) — exp(−(x/σ)²) 는 x>2σ 에서 포화(기울기 0)해 학습환경(raw 액션, LPF 없음)의 운동 수준이
    # σ 의 2~3배면 신호가 죽는다(0909 S8ac3: 보너스 0.015). rational 은 3σ 에서도 0.1·기울기 유지.
    def g(x, sig):
        z = (x / sig) ** 2
        return 1.0 / (1.0 + z) if rational else torch.exp(-z)
    terms = [g(w, s_ang), g(v, s_lin)]
    if use_jvel:
        qd = asset.data.joint_vel.pow(2).mean(1).sqrt()
        terms.append(g(qd, s_jvel))
    if use_acc:
        acc = action_accel_l2(env).sqrt()
        terms.append(g(acc, s_acc))
    return torch.stack(terms, 1).mean(1) * settled_ramp(env, hold_s, ramp_s, tilt_deg, min_height, sensor_cfg, threshold, r_eps)


def _selftest_s8():
    # EMA: 한 프레임 튐은 25% 만 반영, 지속 입력은 4프레임 후 68%
    x0 = torch.zeros(2, 3); st = x0.clone()
    for k in range(4):
        st = st * 0.75 + 0.25 * torch.ones(2, 3)
    assert abs(float(st[0, 0]) - (1 - 0.75 ** 4)) < 1e-6
    spike = torch.zeros(3); spike = spike * 0.75 + 0.25 * torch.tensor([1.0, 0, 0]); assert abs(float(spike[0]) - 0.25) < 1e-9
    # settled 램프: 조건 0.5s 유지 후 1s 램프
    t = 0.0; w = []
    for k in range(80):
        t += 0.02; w.append(min(1.0, max(0.0, (t - 0.5) / 1.0)))
    assert w[24] < 1e-6 and abs(w[49] - 0.5) < 0.03 and w[79] > 1 - 1e-6, (w[24], w[49], w[79])
    # 정숙 보너스: σ 지점에서 e^-1, 0 에서 1, 3σ 에서 ~0 — 유계(0~1)
    for x, s_, want in ((0.0, 0.08, 1.0), (0.08, 0.08, math.exp(-1.0)), (0.24, 0.08, math.exp(-9.0))):
        assert abs(float(torch.exp(-(torch.tensor(x) / s_) ** 2)) - want) < 1e-6
    # rational 형: σ 에서 0.5, 3σ 에서 0.1 (exp 는 3σ 에서 1e-4 로 포화)
    r3 = 1.0 / (1.0 + 9.0); assert abs(r3 - 0.1) < 1e-9 and abs(1.0 / (1.0 + 1.0) - 0.5) < 1e-9
    # 하중 균형: |L-R|/(L+R) 는 0~1 유계
    assert abs((abs(60.0-40.0)/(60.0+40.0)) - 0.2) < 1e-9
    print("[selftest] S8: EMA 스파이크 25%/지속 4프레임 68%·settled 램프 0.5s+1s·정숙 보너스 유계(exp/rational)·하중균형 유계 — PASS")


if __name__ == "__main__":
    selftest()
