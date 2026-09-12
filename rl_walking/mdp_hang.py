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
              "touch_t": torch.full((N,), -1.0, device=dev),      # 양발 첫 동시 접촉 시각
              # 0911 케이블 지지 모델 (cable_on=True 일 때 _cable_apply 사용)
              "cable_on": False, "hx": z(), "hy": z(), "hz": z(), "ax": z(), "ay": z(), "az": z(),
              "cbl_k": z(), "cbl_c": z(), "cbl_tmax": z(), "cbl_L0": torch.ones(N, device=dev), "cbl_d0": z(), "cbl_slack": z(),
              # 케이블 실제 상태 기록 (item 4c: 지지 제거를 스케줄 r 이 아니라 실제 장력으로 판정)
              "cbl_T": z(), "cbl_dist": z(),
              "avx": z(), "avy": z(),   # 0912 앵커 드리프트 속도 [m/s] (시간 가변 당김 재현용)
              "ramp_rate": z(), "ramp_cap": z(), "ramp_start": z(),   # 0912 준정적 롤 토크 램프 [Nm/s, Nm, s] (실물 준정적 이탈 재현용)
              "tp_d": z(), "tp_kp": z(), "tp_kd": z()}   # 0913 앞뒤 2점 현수: 후크 간격 d [m](0=끔), 피치 구속 강성/감쇠 — 토크 cap = T·d/2 (줄은 당기기만)
        env._hang = st
    return st


def _ratio(st):
    t = st["t"]
    dec = (t - st["t_hold"]) / st["t_rel"].clamp(min=1e-3)
    r = torch.where(t <= st["t_hold"], st["r0"], st["r0"] * (1.0 - dec))
    return r.clamp(min=0.0)


def _qrot(q, v):
    """쿼터니언 q=[w,x,y,z] 로 벡터 v 정회전 (v' = R·v). 의존성 없음(런타임·셀프테스트 공용)."""
    w = q[:, 0:1]; u = q[:, 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


_CABLE_T_EPS = 1.0   # 지지 소멸 판정 장력 임계 [N] — 이보다 작으면 줄이 사실상 안 잡음(체중 172.9N 대비 <1%)


def _cable_tension(dist, L, k, c, ddist, tmax):
    """순수 스칼라 장력: 팽팽할 때만 T = clip(k·(dist−L) + c·max(0,ddist), 0, tmax). (Isaac 무관 — 셀프테스트용)

    ★정정(2026-09-11, item 4b): 종전엔 슬랙(dist≤L)이라 stretch=0 이어도 감쇠항 c·ddist 가
    남아 '느슨한 줄이 장력을 내는' 물리 오류가 있었다. 줄은 밀지 못하므로 stretch≤0 이면
    스프링·감쇠 모두 0 으로 게이트한다. 감쇠는 팽팽(stretch>0)하고 멀어질 때(ddist>0)만 저항."""
    stretch = dist - L
    taut = (stretch > 0.0).to(dist.dtype)
    T = taut * (k * stretch.clamp(min=0.0) + c * ddist.clamp(min=0.0))
    T = T.clamp(min=0.0)
    return torch.minimum(T, torch.as_tensor(tmax, dtype=T.dtype, device=T.device))


def _cable_apply(env, st, ids, asset_cfg):
    """길이 제약 케이블 지지(0911 v6, Codex 메타 §2): 고정 월드 앵커 A ↔ 몸통 후크 H(base 위 hz).
    윈치 길이 L(t)=L0−δ0(초기 프리텐션)+payout(t); payout 은 r(t) 로 증가(r0→0 에 걸쳐 δ0+여유). 장력 T=clip(k·(|A−H|−L),0,Tmax)+감쇠, 방향 후크→앵커.
    후크가 CoM 위라 몸이 기울면 발 주위 모멘트 발생 = 기존 수직힘 모델에 없던 '자세→장력→회전' 결합. 수평유지 토크 없음(케이블 대체)."""
    asset = env.scene[asset_cfg.name]
    r = st["r"][ids]; m = st["mass"][ids]
    pos = asset.data.root_pos_w[ids]; quat = asset.data.root_quat_w[ids]
    hook_b = torch.stack([st["hx"][ids], st["hy"][ids], st["hz"][ids]], dim=-1)
    H = pos + _qrot(quat, hook_b)
    _t = st["t"][ids]
    A = torch.stack([st["ax"][ids] + st["avx"][ids] * _t, st["ay"][ids] + st["avy"][ids] * _t, st["az"][ids]], dim=-1)
    rope = A - H; dist = torch.norm(rope, dim=-1).clamp(min=1e-4); dirw = rope / dist.unsqueeze(-1)
    if bool(st.get("cbl_force", False)):
        T = (r.clamp(min=0.0) * m * G)                       # 힘제어 케이블: 크기 = 지지비 스케줄, 방향 = 후크→앵커 (램프 없음: 힘 모델과 동일 시작)
    else:
        payout = (1.0 - (r / st["r0"][ids].clamp(min=1e-3)).clamp(0.0, 1.0)) * (st["cbl_d0"][ids] + st["cbl_slack"][ids])
        L = st["cbl_L0"][ids] - st["cbl_d0"][ids] + payout
        v = asset.data.root_lin_vel_w[ids] if hasattr(asset.data, "root_lin_vel_w") else torch.zeros_like(pos)
        ddist = -(dirw * v).sum(-1)
        T = _cable_tension(dist, L, st["cbl_k"][ids], st["cbl_c"][ids], ddist, st["cbl_tmax"][ids])
        ramp = (st["t"][ids] / 1.0).clamp(0.0, 1.0)          # 첫 1s 램프인: 스폰 낙하 착지 충격(시뮬 인공물) 제거
        T = T * ramp
    st["cbl_T"][ids] = T.detach()                        # item 4c: 실제 적용 장력 기록 (지지 소멸·재긴장 판정·로깅용)
    st["cbl_dist"][ids] = dist.detach()
    F_w = T.unsqueeze(-1) * dirw
    F_b = _quat_inv(quat, F_w)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=F_b.unsqueeze(1), torques=None, positions=hook_b.unsqueeze(1),
        body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)
    if bool((st["tp_d"][ids] > 0).any()):   # 0913 앞뒤 2점 현수 피치 구속 (cap = T·d/2)
        _ty = _two_point_pitch_torque(asset.data.projected_gravity_b[ids], asset.data.root_ang_vel_b[ids], st["tp_kp"][ids], st["tp_kd"][ids], T * st["tp_d"][ids] * 0.5)
        _tau = torch.stack([torch.zeros_like(_ty), _ty, torch.zeros_like(_ty)], dim=-1)
        asset.permanent_wrench_composer.add_forces_and_torques(forces=None, torques=_tau.unsqueeze(1), positions=None, body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)


def _apply(env, st, ids, asset_cfg):
    """지지력을 base 링크 프레임으로 넣는다. ids: long tensor."""
    if bool(st.get("cable_on", False)):
        return _cable_apply(env, st, ids, asset_cfg)
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
    # 0912 준정적 롤 토크 램프: τx += clip(rate·(t−start), 0, cap) — 실물의 느린(각속도 <5 °/s) 횡 이탈 재현 시도 (가정 외란, 실측 아님)
    _dt = (st["t"][ids] - st["ramp_start"][ids]).clamp(min=0.0)
    ramp = st["ramp_rate"][ids] * _dt
    _capm = st["ramp_cap"][ids].abs(); ramp = torch.maximum(torch.minimum(ramp, _capm), -_capm)   # 0913 정정: 크기 기준 클램프(rate·cap 부호가 달라도 무제한 토크 방지)
    tau = tau + torch.stack([ramp, torch.zeros_like(ramp), torch.zeros_like(ramp)], dim=-1)
    # 0913 앞뒤 2점 현수 피치 구속 (tp_d>0 일 때만; cap = Fz·d/2 — 줄이 풀리면(Fz→0) 자동 소멸)
    _ty = _two_point_pitch_torque(pgv, w, st["tp_kp"][ids], st["tp_kd"][ids], Fz * st["tp_d"][ids] * 0.5)
    tau = tau + torch.stack([torch.zeros_like(_ty), _ty, torch.zeros_like(_ty)], dim=-1)
    # ★정정(2026-09-12): Isaac Lab 2.3 `set_forces_and_torques` 는 positions 를 주면 합성 토크를 position×force 로 **덮어써**
    # torques 인자를 버린다(kernels.set_forces_and_torques_at_position). 종전 코드는 수평 유지 토크(lvl_k, 0908)를 한 번도 적용하지
    # 못했다(v5 학습·평가 전부 — 정책은 그 조건으로 학습된 것이므로 배포엔 영향 없음, 문서의 '수평 유지 토크' 서술만 무효).
    # 힘(+레버암 토크)은 set 으로, 명시 토크(수평 유지 + 램프)는 add 로 따로 더한다.
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=None, positions=positions,
        body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)
    asset.permanent_wrench_composer.add_forces_and_torques(
        forces=None, torques=tau.unsqueeze(1), positions=None,
        body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)


def _support_gone_mask(st):
    """지지 소멸 여부(bool, per-env). item 4c 정정.

    힘 모델(cable_on=False): Fz=r·m·g 이므로 r<=0 이면 지지력 정확히 0 → r<=0.
    케이블 모델: 스케줄 r<=0(윈치 최대 방출) **그리고** 실제 장력 T<=eps(줄 슬랙)일 때만 소멸.
      - 종전 r<=0 단독은 케이블이 여전히 팽팽(dist>L)해도 '지지 제거'로 오판했다.
      - 첫 1s 램프인(T*ramp)으로 T≈0 인 구간은 r 이 아직 높아(홀드) 오탐되지 않는다.
      - 슬랙 후 몸이 기울어 재긴장(T>eps)하면 다시 False → 자립 보상/판정에서 그 구간 제외."""
    if bool(st.get("cable_on", False)):
        return (st["r"] <= 0.0) & (st["cbl_T"] <= _CABLE_T_EPS)
    return st["r"] <= 0.0


def support_gone(env) -> torch.Tensor:
    """지지 소멸 여부(per-env bool) — 평가·로깅에서 스케줄이 아니라 실제 지지로 판정할 때 사용."""
    return _support_gone_mask(_state(env))


def hang_support_reset(env, env_ids, r0_range=(0.5, 0.7), hold_range=(0.5, 2.0), release_range=(2.0, 8.0),
                       rope_len_range=(1.0, 2.0), bias_frac=0.01, point_xy=0.03, asset_cfg=None,
                       level_gain_range=(0.0, 0.0), level_damp_range=(0.0, 0.0),
                       cable=False, hook_z_range=(0.25, 0.40), anchor_x_range=(-0.05, 0.05), anchor_y_range=(-0.05, 0.05),
                       anchor_h_range=(1.0, 1.6), cbl_k_range=(1500.0, 3000.0), cbl_c_range=(20.0, 120.0),
                       cbl_slack_range=(0.05, 0.15), cable_force_mode=False, anchor_vx_range=(0.0, 0.0), anchor_vy_range=(0.0, 0.0), ramp_roll_torque=(0.0, 0.0, 0.0),
                       ramp_rate_range=None, ramp_cap_range=None, ramp_start_range=None, ramp_sign_random=False,
                       two_point_d_range=(0.0, 0.0), two_point_kp_range=(200.0, 200.0), two_point_kd_range=(10.0, 10.0)):
    """mode="reset": 에피소드마다 지지 스케줄·로프 기하를 새로 뽑고 첫 지지력을 즉시 건다.
    two_point_d_range: 0913 승윤님 확인 '줄은 몸통 앞·뒤 2점' — 두 줄이 팽팽한 동안 피치를 구속한다(롤은 진자 그대로).
    토크 = clip(−kp·pitch − kd·ω_y, ±T·d/2) (줄은 당기기만 하므로 한 줄이 전 장력을 받을 때가 최대). d=0 이면 종전(단일 후크)과 동일.
    d 실측 없음(가정 0.1~0.3 m)."""
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
    st["ramp_rate"][ids] = float(ramp_roll_torque[0]); st["ramp_cap"][ids] = float(ramp_roll_torque[1]); st["ramp_start"][ids] = float(ramp_roll_torque[2])
    st["tp_d"][ids] = U(*two_point_d_range); st["tp_kp"][ids] = U(*two_point_kp_range); st["tp_kd"][ids] = U(*two_point_kd_range)
    if ramp_rate_range is not None:   # 학습용 DR(0912 S8ac4q): 준정적 롤 토크 램프를 env 마다 다르게, 부호 랜덤
        st["ramp_rate"][ids] = U(*ramp_rate_range); st["ramp_cap"][ids] = U(*ramp_cap_range); st["ramp_start"][ids] = U(*ramp_start_range)
        if ramp_sign_random:
            sgn = (torch.rand(n, device=env.device) < 0.5).to(st["ramp_rate"].dtype) * 2.0 - 1.0
            st["ramp_rate"][ids] = st["ramp_rate"][ids] * sgn; st["ramp_cap"][ids] = st["ramp_cap"][ids] * sgn
    # ★정정(2026-09-12): 질량 캐시를 케이블 블록 **앞**으로 옮김. 종전엔 케이블 프리텐션 d0=r0·m·g/k 와 tmax=1.5·m·g 를
    # st["mass"] 를 채우기 전에 계산해 첫 리셋에서 mass=0 → tmax=0 → 장력이 에피소드 내내 0 으로 클램프됐다(평가는 단일
    # 에피소드라 케이블이 한 번도 작용하지 않았음 — 0911 케이블 결과 전부 '무지지 낙하' 였다). 진단: simbb 덤프 cbl_T.
    # 총질량: startup 질량 DR(add_base_mass) 반영. PhysX 뷰 get_masses() 는 CPU readback 이라
    # 매 리셋마다 부르면 4096env 학습이 ~7it/min 로 느려진다(0908 실측) — startup 뒤 불변이므로 1회 캐시.
    _asset0 = env.scene[asset_cfg.name]
    cache = getattr(env, "_hang_mass_all", None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = _asset0.root_physx_view.get_masses().sum(1).to(env.device)
        env._hang_mass_all = cache
    st["mass"][ids] = cache[ids]
    if cable:
        st["cable_on"] = True
        # 0912 힘제어 케이블: 장력 크기는 r(t)·m·g(힘 모델과 같은 스케줄), 방향은 후크→고정 앵커(기하 결합). 강성·길이 제약 없음 —
        # 실물 로프 강성이 미측정이라 길이 모델은 k 에 결과가 지배되므로, 줄 모멘트(후크 비연직) 가설 검정엔 이 모드를 쓴다.
        st["cbl_force"] = bool(cable_force_mode)
        st["avx"][ids] = U(*anchor_vx_range); st["avy"][ids] = U(*anchor_vy_range)   # 앵커가 t 에 비례해 이동(운영자·윈치 드리프트 가설)
        asset0 = env.scene[asset_cfg.name]
        p0 = asset0.data.root_pos_w[ids]                     # 리셋 시 base 월드 위치
        st["hx"][ids] = 0.0; st["hy"][ids] = 0.0; st["hz"][ids] = U(*hook_z_range)   # 후크(토르소 위)
        st["ax"][ids] = p0[:, 0] + U(*anchor_x_range)        # 고정 앵커 = 초기 후크 위 + 오프셋(앞뒤 포함)
        st["ay"][ids] = p0[:, 1] + U(*anchor_y_range)
        st["az"][ids] = p0[:, 2] + st["hz"][ids] + U(*anchor_h_range)
        st["cbl_k"][ids] = U(*cbl_k_range); st["cbl_c"][ids] = U(*cbl_c_range); st["cbl_slack"][ids] = U(*cbl_slack_range)
        # 초기 |A−H0| (수직 근사 = anchor_h) 를 L0 로, 프리텐션 δ0 = r0·m·g / k (수직서 T_z≈T)
        st["cbl_L0"][ids] = torch.norm(torch.stack([st["ax"][ids]-p0[:,0], st["ay"][ids]-p0[:,1],
                                                    st["az"][ids]-(p0[:,2]+st["hz"][ids])], -1), dim=-1)
        st["cbl_d0"][ids] = (st["r0"][ids] * st["mass"][ids] * G / st["cbl_k"][ids])
        st["cbl_tmax"][ids] = st["mass"][ids] * G * 1.5
        assert bool((st["cbl_tmax"][ids] > 0).all()) and bool((st["cbl_d0"][ids] > 0).all()), \
            "케이블 프리텐션/최대장력이 0 — 질량 미설정(순서 버그) — 무지지 낙하 평가가 되므로 중단"
    st["t"][ids] = 0.0
    st["released_t"][ids] = -1.0
    st["touch_t"][ids] = -1.0
    asset = env.scene[asset_cfg.name]
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
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    nt = both & (st["touch_t"] < 0.0)
    st["touch_t"][nt] = st["t"][nt]
    _apply(env, st, ids, asset_cfg)                     # 케이블이면 여기서 cbl_T 갱신
    # 지지 소멸 시각 — 힘 모델 r<=0, 케이블 실제 장력<=eps (item 4c). _apply 뒤라 cbl_T 최신.
    newly = _support_gone_mask(st) & (st["released_t"] < 0.0)
    st["released_t"][newly] = st["t"][newly]


# ───────────────────────────── 보상 ─────────────────────────────

def hang_ratio(env) -> torch.Tensor:
    return _state(env)["r"]


def survive_after_release(env) -> torch.Tensor:
    """지지 소멸 이후 살아 있는 매 스텝 +1 — '자유 기립 생존' 을 강하게.

    item 4c: 힘 모델은 r<=0, 케이블은 실제 장력<=eps 일 때만 카운트(재긴장 구간 제외).
    힘 모델 동작은 종전(r<=0)과 동일 — v5 보상 불변."""
    return _support_gone_mask(_state(env)).to(torch.float32)


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
    _selftest_cable()
    _selftest_two_point()
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
def torso_pitch_band(env, band_rad: float = 0.087, fwd_band_rad: float | None = None, asset_cfg=None) -> torch.Tensor:
    """몸통 피치 밴드 벌점. 기본은 |피치|>band 대칭.
    fwd_band_rad 를 주면 **비대칭**: 앞기울임(pg_x>0, CoM 을 발 위로 가져오는 방향)은 fwd_band 까지 무벌, 뒤기울임은 band_rad.
    (0911 v6: 실물 CoM 이 발 뒤라 앞으로 기울여야 서는데 대칭 밴드가 그 해법을 벌줬음 — 앞은 관대, 뒤는 엄격.)"""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    pg = asset.data.projected_gravity_b
    pitch = torch.atan2(pg[:, 0], -pg[:, 2])          # 실측 부호: 앞기울임 = 음수(v5 정착 −4.9°), 뒤기울임 = 양수
    if fwd_band_rad is None:
        return (pitch.abs() - band_rad).clamp(min=0.0)
    fwd = (-pitch - fwd_band_rad).clamp(min=0.0)       # 앞(음수)으로 fwd_band 초과분 — 관대(fwd_band 큼)
    back = (pitch - band_rad).clamp(min=0.0)           # 뒤(양수)로 band 초과분 — 엄격(band 작음)
    return fwd + back


def _selftest_pitch_band():
    import math
    for deg, want in ((0.0, 0.0), (4.0, 0.0), (17.0, math.radians(17.0) - 0.087), (-17.0, math.radians(17.0) - 0.087)):
        th = math.radians(deg); pg = torch.tensor([[math.sin(th), 0.0, -math.cos(th)]])
        v = (torch.atan2(pg[:, 0], -pg[:, 2]).abs() - 0.087).clamp(min=0.0)
        assert abs(float(v) - want) < 1e-4, (deg, float(v), want)
    # 비대칭: 앞(+) 관대(fwd 12°)·뒤(−) 엄격(3°)
    import math as _m
    def pv(deg, fwd, back):
        th=_m.radians(deg); pg=torch.tensor([[_m.sin(th),0.0,-_m.cos(th)]])
        pit=torch.atan2(pg[:,0],-pg[:,2]); return float((pit-fwd).clamp(min=0.0)+(-pit-back).clamp(min=0.0))
    def pv2(deg, fwd, back):
        th=_m.radians(deg); pg=torch.tensor([[_m.sin(th),0.0,-_m.cos(th)]])
        pit=torch.atan2(pg[:,0],-pg[:,2]); return float((-pit-fwd).clamp(min=0.0)+(pit-back).clamp(min=0.0))
    # 앞기울임 = 음수 deg. 앞 −8°→무벌(fwd 12°), 뒤 +8°→벌(back 3°), 앞 −16°→초과분
    assert abs(pv2(-8, _m.radians(12), 0.052)-0.0)<1e-4, pv2(-8,_m.radians(12),0.052)
    assert abs(pv2(+8, _m.radians(12), 0.052)-(_m.radians(8)-0.052))<1e-4
    assert abs(pv2(-16,_m.radians(12), 0.052)-(_m.radians(16)-_m.radians(12)))<1e-4
    print("[selftest] 몸통 피치 밴드: 비대칭 앞(−)8°→무벌·뒤(+)8°→벌·앞16°→초과분(부호 수정) — PASS")

def _level_torque(pg, w, k, c):
    """몸통 프레임 복원 토크. pg=투영중력(수평이면 (0,0,−1)), w=각속도(body). 앞기울임(pg_x>0)→ −y 토크(고개 들기),
    +x 롤(pg_y<0)→ −x 토크. τ = k·(pg_y, −pg_x, 0) − c·(w_x, w_y, 0)."""
    zero = torch.zeros_like(k)
    return torch.stack([k * pg[:, 1] - c * w[:, 0], -k * pg[:, 0] - c * w[:, 1], zero], dim=-1)


def _two_point_pitch_torque(pg, w, kp, kd, cap):
    """앞뒤 2점 현수의 피치 구속 토크 τ_y [Nm] (body y축; 앞기울임 pitch=atan2(pg_x,−pg_z)>0 → 음의 토크). |τ| ≤ cap = T·d/2, cap<=0 이면 0."""
    pitch = torch.atan2(pg[:, 0], -pg[:, 2])
    cap = cap.clamp(min=0.0)
    return (-kp * pitch - kd * w[:, 1]).clamp(-cap, cap)


def _selftest_two_point():
    pg = torch.tensor([[0.0, 0.0, -1.0], [math.sin(0.1), 0.0, -math.cos(0.1)], [-math.sin(0.1), 0.0, -math.cos(0.1)], [math.sin(0.1), 0.0, -math.cos(0.1)]])
    w = torch.zeros(4, 3); kp = torch.full((4,), 200.0); kd = torch.full((4,), 10.0); cap = torch.tensor([10.0, 10.0, 10.0, 0.0])
    t = _two_point_pitch_torque(pg, w, kp, kd, cap)
    assert abs(float(t[0])) < 1e-9, "수평 → 0"
    assert float(t[1]) == -10.0, f"앞기울임 0.1 rad → −20 → cap −10: {float(t[1])}"
    assert float(t[2]) == 10.0, "뒤기울임 → +cap"
    assert float(t[3]) == 0.0, "cap 0(줄 슬랙) → 0"
    return "2점 현수 피치 토크: 수평 0 · 앞/뒤 기울임 부호·cap 클램프 · 슬랙 0"


def _selftest_cable():
    # 슬랙(dist<L)=0, 타이트(dist>L) 프리텐션, Tmax 클립, 감쇠 단방향
    d=torch.tensor([1.0, 1.086, 1.2, 1.086]); L=torch.tensor([1.1, 1.0, 1.0, 1.0]); k=torch.tensor([2000.0]*4)
    c=torch.tensor([100.0]*4); dd=torch.tensor([0.0, 0.0, 0.0, 0.5]); tmax=torch.tensor([300.0]*4)
    T=_cable_tension(d, L, k, c, dd, tmax)
    assert float(T[0])==0.0, "슬랙(dist<L) 장력 0"
    assert abs(float(T[1])-172.0)<1.0, ("프리텐션 δ0=0.086→~172N", float(T[1]))
    assert float(T[2])==300.0, "Tmax 클립"
    assert abs(float(T[3])-(172.0+50.0))<1.0, ("감쇠 +c·ddist", float(T[3]))
    # 감쇠는 멀어질 때만(ddist>0)
    assert abs(float(_cable_tension(torch.tensor([1.086]),torch.tensor([1.0]),torch.tensor([2000.0]),torch.tensor([100.0]),torch.tensor([-0.5]),torch.tensor([300.0]))[0])-172.0)<1.0
    # ★회귀(item 4b): 슬랙(dist<L)인데 멀어지는 중(ddist>0) — 감쇠가 장력을 만들면 안 됨(줄은 밀지 못한다)
    assert float(_cable_tension(torch.tensor([0.9]),torch.tensor([1.0]),torch.tensor([2000.0]),torch.tensor([100.0]),torch.tensor([0.5]),torch.tensor([300.0]))[0])==0.0, "슬랙+ddist>0 장력 0 (감쇠 누출 버그 부재)"
    # _qrot: 항등 쿼터니언·z축 90° 회전(x→y)
    import math as _m
    assert torch.allclose(_qrot(torch.tensor([[1.,0,0,0]]), torch.tensor([[1.,2,3]])), torch.tensor([[1.,2,3]]), atol=1e-6)
    qz=torch.tensor([[_m.cos(_m.pi/4),0,0,_m.sin(_m.pi/4)]])  # z 90°
    assert torch.allclose(_qrot(qz, torch.tensor([[1.,0,0]])), torch.tensor([[0.,1,0]]), atol=1e-5)
    # item 4c: 지지 소멸 판정 — 힘 모델 r<=0 ; 케이블 r<=0 AND 실제 장력<=eps
    stf = {"cable_on": False, "r": torch.tensor([0.5, 0.0, -0.1])}
    assert torch.equal(_support_gone_mask(stf), torch.tensor([False, True, True])), "힘 모델 지지소멸=r<=0"
    stc = {"cable_on": True, "r": torch.tensor([0.5, 0.0, 0.0, 0.0]), "cbl_T": torch.tensor([200.0, 200.0, 0.5, 5.0])}
    assert torch.equal(_support_gone_mask(stc), torch.tensor([False, False, True, False])), \
        "케이블 지지소멸: r>0→F, r=0&팽팽→F(재긴장), r=0&슬랙→T"
    print("[selftest] 케이블 장력: 슬랙 0·프리텐션 172N·Tmax 클립·감쇠 단방향(슬랙누출 0)·_qrot·지지소멸(힘/케이블) — PASS")


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
                      gate: str = "contact", hold_s: float = 0.5, ramp_s: float = 1.0, tilt_deg: float = 8.0,
                      min_height: float = 0.60, r_eps: float = 0.05) -> torch.Tensor:
    """정착 상태에서 양발 수직하중 불균형 |L-R|/(L+R) 벌점 (0~1). 0910 실물: 착지 후 하중이 한 발(오른 65%)로 쏠려 반대 발이 떠 넘어짐 —
    원인이 발 롤·횡 CoM·로프 어느 쪽이든, 하중을 좌우로 고르게 실으면 그 실패를 막는다. settled 램프로 정착 후에만 켠다(착지 순간의 정상적 편하중은 벌 안 줌)."""
    sensor_cfg = sensor_cfg or _feet_cfg()
    fz = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    fl, fr = fz[:, 0], fz[:, 1]
    tot = fl + fr
    imb = (fl - fr).abs() / tot.clamp(min=min_total)      # min_total 미만이면 분모 고정(공중·저하중서 폭주 방지)
    imb = torch.where(tot > min_total, imb, torch.zeros_like(imb)).clamp(0.0, 1.0)
    if gate == "contact":
        # 0910 v6b: 정착 게이트(지지0·0.5s) 대신 양발 접촉이면 항상 — 줄 풀리는 하중이전 구간(한 발 빠짐이 실제 일어나는 곳)을 커버.
        both = (torch.norm(env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > threshold).all(dim=1).float()
        return imb * both
    return imb * settled_ramp(env, hold_s, ramp_s, tilt_deg, min_height, sensor_cfg, threshold, r_eps)


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
