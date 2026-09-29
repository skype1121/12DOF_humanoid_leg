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
import os

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


_DST_K = 3   # 에피소드당 외란 슬롯 수


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
              "ramp_rate": z(), "ramp_cap": z(), "ramp_start": z(), "yaw_c": z(),  # 0914 공중 응답 기준 시험용 요 감쇠 [Nm·s/rad] (기본 0 = 종전 동일)   # 0912 준정적 롤 토크 램프 [Nm/s, Nm, s] (실물 준정적 이탈 재현용)
              "tp_d": z(), "tp_kp": z(), "tp_kd": z(),
              # 0921 G7 외란 스케줄(에피소드당 최대 K 회): kind 0=없음 1=지속 램프 2=순간 충격; 방향은 월드 xy 단위벡터
              "dst_kind": torch.zeros(N, _DST_K, device=dev), "dst_t0": torch.zeros(N, _DST_K, device=dev), "dst_rise": torch.ones(N, _DST_K, device=dev),
              "dst_hold": torch.zeros(N, _DST_K, device=dev), "dst_F": torch.zeros(N, _DST_K, device=dev), "dst_dir": torch.zeros(N, _DST_K, 2, device=dev),
              "dst_F_now": z(), "dst_last_end": torch.full((N,), -1.0, device=dev), "dst_pending": torch.zeros(N, device=dev), "dst_quiet_run": z(), "dst_bonus": z(),
              # 0921 G8: 상태 기반 회복 게이트 — 외란 이벤트 유무와 무관하게 '균형을 잃은 동안' 열린다(서서히 기우는 경우 포함)
              "rec_open": z(), "rec_open_t": z(), "rec_settle_run": z(), "rec_ev": z(), "rec_guard": z()}   # 0913 앞뒤 2점 현수: 후크 간격 d [m](0=끔), 피치 구속 강성/감쇠 — 토크 cap = T·d/2 (줄은 당기기만)
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
    _yz = -st["yaw_c"][ids] * w[:, 2]                                   # 0914 요 감쇠(기본 0)
    tau = tau + torch.stack([torch.zeros_like(_yz), torch.zeros_like(_yz), _yz], dim=-1)
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
    # 0921 G7: 외란(지속 램프·순간 충격) — 월드 수평 힘을 base 로컬로 변환해 더한다(is_global 미작동 확인 0920)
    Fd = _dist_force_w(st, ids)                                     # (n,2) 월드 xy [N]
    st["dst_F_now"][ids] = Fd.norm(dim=-1)
    if bool((st["dst_F_now"][ids] > 0.0).any()):
        Fd3 = torch.cat([Fd, torch.zeros_like(Fd[:, :1])], dim=-1)
        Fd_b = _quat_inv(asset.data.root_quat_w[ids], Fd3)
        asset.permanent_wrench_composer.add_forces_and_torques(
            forces=Fd_b.unsqueeze(1), torques=None, positions=None,
            body_ids=asset_cfg.body_ids, env_ids=ids, is_global=False)


def _dist_profile(t, kind, t0, rise, hold):
    """외란 크기 배율 0~1 (텐서, broadcast). kind1 램프: t0 부터 rise 동안 0→1, hold 동안 1, 그 뒤 0. kind2 충격: t0 부터 hold 동안 1."""
    dt = t - t0
    ramp = (dt / rise.clamp(min=1e-3)).clamp(0.0, 1.0)
    in_ramp = (kind == 1) & (dt >= 0.0) & (dt < rise + hold)
    in_imp = (kind == 2) & (dt >= 0.0) & (dt < hold)
    return torch.where(in_ramp, ramp, torch.zeros_like(ramp)) + in_imp.to(ramp.dtype)


def _dist_force_w(st, ids):
    """(n,2) 월드 xy 힘 = Σ_k F_k · profile_k(t) · dir_k."""
    t = st["t"][ids].unsqueeze(1)
    prof = _dist_profile(t, st["dst_kind"][ids], st["dst_t0"][ids], st["dst_rise"][ids], st["dst_hold"][ids])   # (n,K)
    return ((st["dst_F"][ids] * prof).unsqueeze(-1) * st["dst_dir"][ids]).sum(1)


def _dist_end_time(st):
    """(N,K) 각 외란의 종료 시각."""
    return st["dst_t0"] + torch.where(st["dst_kind"] == 1, st["dst_rise"] + st["dst_hold"], st["dst_hold"])


def disturbance_reset(env, env_ids, n_range=(1, 3), t0_range=(2.0, 10.0), gap_range=(3.0, 7.0), ramp_prob=0.6,
                      F_ramp_range=(3.0, 20.0), rise_range=(1.0, 4.0), hold_range=(1.0, 3.0),
                      F_imp_range=(5.0, 30.0), dur_imp_range=(0.2, 0.4), lateral_prob=0.3, lateral_scale=0.6,
                      p_active=1.0, asset_cfg=None):
    """mode="reset": 에피소드마다 외란 스케줄을 뽑는다 (승윤님 0920: 지속 외력·느린 균형 상실·앞뒤 회복 스텝).
    방향은 스폰 시점 몸통 앞축(월드) 기준 앞/뒤(확률 1−lateral_prob) 또는 좌/우(힘 ×lateral_scale). 값은 전부 [가정]."""
    asset_cfg = asset_cfg or _base_cfg()
    st = _state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = len(ids); dev = env.device; K = _DST_K
    def U(lo, hi, shape=(n, K)):
        return torch.empty(*shape, device=dev).uniform_(float(lo), float(hi))
    cnt = torch.randint(int(n_range[0]), int(n_range[1]) + 1, (n,), device=dev)
    active = (torch.rand(n, device=dev) < float(p_active))
    slot = torch.arange(K, device=dev).unsqueeze(0)
    on = (slot < cnt.unsqueeze(1)) & active.unsqueeze(1)
    is_ramp = torch.rand(n, K, device=dev) < float(ramp_prob)
    kind = torch.where(on, torch.where(is_ramp, torch.ones_like(slot, dtype=torch.float32).expand(n, K), 2.0 * torch.ones(n, K, device=dev)), torch.zeros(n, K, device=dev))
    rise = U(*rise_range); hold = torch.where(is_ramp, U(*hold_range), U(*dur_imp_range))
    F = torch.where(is_ramp, U(*F_ramp_range), U(*F_imp_range))
    # 시각: 첫 외란 t0, 다음은 이전 종료 + gap
    dur = torch.where(is_ramp, rise + hold, hold)
    t0 = torch.zeros(n, K, device=dev); t0[:, 0] = U(*t0_range, (n,))
    for k in range(1, K):
        t0[:, k] = t0[:, k - 1] + dur[:, k - 1] + U(*gap_range, (n,))
    # 방향: 몸통 +x(앞) 월드 수평
    asset = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w[ids]
    ex = torch.zeros(n, 3, device=dev); ex[:, 0] = 1.0
    fw = _quat_apply(q, ex)[:, :2]; fw = fw / fw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    lf = torch.stack([-fw[:, 1], fw[:, 0]], dim=-1)
    lat = torch.rand(n, K, device=dev) < float(lateral_prob)
    sgn = torch.where(torch.rand(n, K, device=dev) < 0.5, -1.0, 1.0)
    d = torch.where(lat.unsqueeze(-1), lf.unsqueeze(1).expand(n, K, 2), fw.unsqueeze(1).expand(n, K, 2)) * sgn.unsqueeze(-1)
    F = torch.where(lat, F * float(lateral_scale), F)
    st["dst_kind"][ids] = kind; st["dst_t0"][ids] = t0; st["dst_rise"][ids] = rise; st["dst_hold"][ids] = hold
    st["dst_F"][ids] = F; st["dst_dir"][ids] = d
    st["dst_F_now"][ids] = 0.0; st["dst_last_end"][ids] = -1.0; st["dst_pending"][ids] = 0.0; st["dst_quiet_run"][ids] = 0.0; st["dst_bonus"][ids] = 0.0


def _quat_apply(q, v):
    """(n,4) wxyz × (n,3) → (n,3) (Isaac 없이도 돌도록 자체 구현)."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    t = 2.0 * torch.cross(torch.cat([x, y, z], 1), v, dim=-1)
    return v + w * t + torch.cross(torch.cat([x, y, z], 1), t, dim=-1)


def dist_force_now(env) -> torch.Tensor:
    """(N,) 지금 걸린 외란 힘 크기 [N] (보상·평가용, 관측 아님)."""
    return _state(env)["dst_F_now"]


def _feet_body_ids(env, sensor_name: str = "contact_forces"):
    """접촉 센서에서 **발 링크만** 골라낸 인덱스(캐시). 0921 결함: SceneEntityCfg("contact_forces") 는 body_ids 가
    slice(None)(전 바디)라 매니저가 해석해 주지 않는 직접 호출에서는 '양발 접촉'이 언제나 False 였다 → 게이트 상시 열림→래치."""
    ids = getattr(env, "_hang_feet_ids", None)
    if ids is None:
        names = list(env.scene.sensors[sensor_name].body_names)
        ids = [i for i, n in enumerate(names) if "ankle_r" in n]
        assert len(ids) == 2, f"발 링크 2개를 못 찾음: {names}"
        env._hang_feet_ids = ids
    return ids


def _balance_state(env, sensor_cfg=None, threshold: float = 5.0, asset_cfg=None):
    """(both_contact, tilt_deg, |ω_xy|, |v_xy|, q̇_rms) — 게이트·보너스가 공유하는 원시 상태."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    if sensor_cfg is not None and not isinstance(getattr(sensor_cfg, "body_ids", slice(None)), slice):
        f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        f = env.scene.sensors["contact_forces"].data.net_forces_w[:, _feet_body_ids(env), :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    pg = asset.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1.0, 1.0)))
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    v = asset.data.root_lin_vel_b[:, :2].norm(dim=1)
    qd = asset.data.joint_vel.pow(2).mean(1).sqrt()
    return both, tilt, w, v, qd


def recovery_gate_step(env, env_ids, tilt_ok: float = 6.5, w_ok: float = 1.5, v_ok: float = 0.5, settle_s: float = 0.3,
                       max_open_s: float = 4.0, lift_tilt: float = 3.0, sensor_cfg=None, threshold: float = 5.0, asset_cfg=None):
    """mode="interval", interval_range_s=(0,0): **상태 기반** 회복 게이트를 매 스텝 갱신한다 (0921 G8, 승윤님 지적 2).

    종전(G7)은 '외란 해제 + 3 s' 라는 시간 창이었다 → 4 s 째 필요한 회복 스텝이 자세·접촉 벌점을 받았고, 외란 이벤트 없이
    서서히 기울 때는 창이 아예 열리지 않아 스텝을 막았다. 여기서는 **실제 균형 상태**로 연다/닫는다:
      열림  : 외란 힘이 걸려 있거나, 기울기>tilt_ok, |ω|>w_ok, |v|>v_ok, 또는 (한 발 비접촉 & 기울기>lift_tilt)
      닫힘  : 위 조건이 모두 정상인 상태가 settle_s 연속 유지되면 (= 실제로 회복함)
      안전장치: 연속 열림이 max_open_s 를 넘으면 실제 안정될 때까지 닫아 둔다(계속 움직여 벌점을 회피하는 전략 차단).

    **임계는 학습 중 실측으로 잡았다**(0921, 액션 잡음 포함 1024 env): 정상 서기 |ω| p50 0.38·p90 0.80·p99 1.3,
    |v| p50 0.08·p90 0.17, 기울기 p50 1.8·p90 3.7·p99 4.8°. 외란 표본과 ω·v 분포가 거의 겹쳐 **속도 기준은 정상 서기를
    구분하지 못한다**(첫 시도: w_ok 0.35 → 게이트 97 % 열림, 자세 보상 소멸·에피소드 3.1 s). 그래서 기울기(정상 p99 4.8 <
    tilt_ok 6.5)를 주 신호로, ω·v 는 격한 이탈만 잡는 안전망(1.5·0.5)으로 둔다. 발 들기는 기울기와 함께일 때만 연다
    (정상 서기 중 우발적 발 들림으로 벌점이 꺼지지 않게)."""
    st = _state(env)
    both, tilt, w, v, _qd = _balance_state(env, sensor_cfg, threshold, asset_cfg)
    unbal = (st["dst_F_now"] > 0.0) | (tilt > tilt_ok) | (w > w_ok) | (v > v_ok) | ((~both) & (tilt > lift_tilt))
    st["rec_settle_run"] = torch.where(unbal, torch.zeros_like(st["rec_settle_run"]), st["rec_settle_run"] + env.step_dt)
    settled = st["rec_settle_run"] >= settle_s
    open_now = st["rec_open"] > 0.5
    # 안전장치 래치: 연속 열림이 max_open_s 를 넘으면 '실제로 안정될 때까지' 게이트를 닫아 둔다(계속 움직여 벌점을 회피하는 전략 차단).
    st["rec_open_t"] = torch.where(open_now, st["rec_open_t"] + env.step_dt, torch.zeros_like(st["rec_open_t"]))
    guard = (st["rec_guard"] > 0.5) | (st["rec_open_t"] > max_open_s)
    guard = guard & ~settled                                     # 실제 회복하면 래치 해제
    want_open = unbal & ~guard
    open_next = torch.where(want_open, torch.ones_like(st["rec_open"]),
                            torch.where(settled | guard, torch.zeros_like(st["rec_open"]), st["rec_open"]))
    # 회복 이벤트: 열려 있다가 **실제 안정으로** 닫혔다 = 균형을 잃었다가 되찾음(안전장치로 닫힌 경우는 제외)
    st["rec_ev"] = (open_now & (open_next <= 0.5) & settled).to(torch.float32)
    st["rec_open"] = open_next
    st["rec_open_t"] = torch.where(open_next > 0.5, st["rec_open_t"], torch.zeros_like(st["rec_open_t"]))
    st["rec_guard"] = guard.to(torch.float32)
    if os.environ.get("HANG_GATE_DEBUG"):
        _n = getattr(env, "_gate_dbg_n", 0) + 1; env._gate_dbg_n = _n
        if _n % 50 == 0:
            q = lambda x, m: (float(x[m].quantile(0.5)), float(x[m].quantile(0.9)), float(x[m].quantile(0.99))) if bool(m.any()) else (0., 0., 0.)
            calm = (st["dst_F_now"] <= 0.0) & both & (tilt < 5.0)      # 외란 없고 양발 접촉 = '정상 서기' 표본
            dist = st["dst_F_now"] > 0.0
            print(f"[gate] step {_n} open {float(st['rec_open'].mean()):.3f} both {float(both.float().mean()):.3f} ev {float(st['rec_ev'].sum()):.0f} "
                  f"| calm n{int(calm.sum())} w(p50/p90/p99) {q(w, calm)[0]:.2f}/{q(w, calm)[1]:.2f}/{q(w, calm)[2]:.2f} "
                  f"v {q(v, calm)[0]:.3f}/{q(v, calm)[1]:.3f}/{q(v, calm)[2]:.3f} tilt {q(tilt, calm)[0]:.1f}/{q(tilt, calm)[1]:.1f}/{q(tilt, calm)[2]:.1f} "
                  f"| dist n{int(dist.sum())} w {q(w, dist)[0]:.2f}/{q(w, dist)[1]:.2f} v {q(v, dist)[0]:.3f}/{q(v, dist)[1]:.3f} tilt {q(tilt, dist)[0]:.1f}/{q(tilt, dist)[1]:.1f}", flush=True)


def recovery_open(env) -> torch.Tensor:
    """(N,) 회복 게이트가 열려 있으면 1 (진단·게이트용)."""
    return _state(env)["rec_open"]


def recovery_quality(env, s_ang: float = 0.30, s_lin: float = 0.10, s_jvel: float = 0.50, tilt_ok: float = 6.5,
                     sensor_cfg=None, threshold: float = 5.0, asset_cfg=None) -> torch.Tensor:
    """**회복 게이트가 열려 있는 동안만** 주는 등급형(rational) 정숙 보상 0~1 — 문턱 미달로 0 에 갇히지 않는다.
    σ 는 학습 중 실제 운동 수준(평가 관측 |ω|≈0.1~0.2·q̇≈0.11, 학습은 액션 잡음으로 더 큼)을 감싸도록 넉넉히 잡는다.
    G7 의 recovery_bonus(경성 문턱 0.15/0.05/0.3 + 0.5 s 연속)는 학습 내내 0 이었다 — 그 결함의 교체항."""
    both, tilt, w, v, qd = _balance_state(env, sensor_cfg, threshold, asset_cfg)
    def g(x, sig):
        return 1.0 / (1.0 + (x / sig) ** 2)
    q = (g(w, s_ang) + g(v, s_lin) + g(qd, s_jvel)) / 3.0
    q = q * both.to(q.dtype) * (tilt < tilt_ok).to(q.dtype)
    return q * _state(env)["rec_open"]


def recovery_event(env) -> torch.Tensor:
    """(N,) 균형을 잃었다가 실제로 되찾은 스텝에 1 (recovery_gate_step 이 계산). 성공 판정과 같은 의미의 1 회성 보상."""
    return _state(env)["rec_ev"]


def dist_window_mask(env, recover_s: float = 3.0) -> torch.Tensor:
    """외란이 걸려 있거나 해제 뒤 recover_s 안이면 True — 자세·발접촉·정숙 항의 게이트(그 창에서는 스텝 허용)."""
    st = _state(env)
    t = st["t"].unsqueeze(1)
    ends = _dist_end_time(st)
    on = st["dst_kind"] > 0
    started = on & (t >= st["dst_t0"])
    recent = (started & (t < ends + float(recover_s))).any(1)
    return recent


def _resolve_nested(env, params):
    """래핑된 항의 params 안 SceneEntityCfg 는 매니저가 못 푼다(최상위만 resolve) → 여기서 한 번 resolve."""
    if params is None:
        return {}
    if not params.get("_resolved", False):
        for k, v in list(params.items()):
            if SceneEntityCfg is not None and isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)
        params["_resolved"] = True
    return {k: v for k, v in params.items() if k != "_resolved"}


def state_gated(env, func, params: dict | None = None) -> torch.Tensor:
    """기존 보상 항 func 를 **회복 게이트가 닫힌 동안(= 실제로 안정된 동안)** 에만 적용 (0921 G8).
    균형을 잃은 동안에는 자세·접촉·정숙 벌점이 0 → 필요한 스텝을 시간과 무관하게 허용한다."""
    val = func(env, **_resolve_nested(env, params))
    return val * (1.0 - _state(env)["rec_open"]).to(val.dtype)


def open_gated(env, func, params: dict | None = None) -> torch.Tensor:
    """func 를 회복 게이트가 열린 동안에만 적용(창 안 약한 action-rate 등)."""
    val = func(env, **_resolve_nested(env, params))
    return val * _state(env)["rec_open"].to(val.dtype)


def rest_gated(env, func, recover_s: float = 3.0, params: dict | None = None) -> torch.Tensor:
    """기존 보상 항 func 를 '외란 창 밖(정지 구간)' 에서만 적용. 스텝이 필요한 창에서는 자세·발 들기 벌점이 0."""
    val = func(env, **_resolve_nested(env, params))
    return val * (~dist_window_mask(env, recover_s)).to(val.dtype)


def window_gated(env, func, recover_s: float = 3.0, params: dict | None = None) -> torch.Tensor:
    """func 를 외란 창 안에서만 적용(예: 창 안에서 약한 action-rate)."""
    val = func(env, **_resolve_nested(env, params))
    return val * dist_window_mask(env, recover_s).to(val.dtype)


def recovery_bonus(env, recover_s: float = 6.0, quiet_s: float = 0.5, s_ang: float = 0.15, s_lin: float = 0.05, s_jvel: float = 0.3,
                   sensor_cfg=None, threshold: float = 5.0, asset_cfg=None) -> torch.Tensor:
    """외란 해제 뒤 recover_s 안에 quiet_s 연속으로 '조용히 서기'(양발 접촉·|ω|<s_ang·|v|<s_lin·관절속도RMS<s_jvel)에 도달하면
    1 회성 보너스 1.0 (승윤님 0920: 성공 = 단순 생존이 아니라 외란 해제 후 다시 조용히 서는 것). 매 스텝 0/1."""
    st = _state(env)
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    sensor_cfg = sensor_cfg or _feet_cfg()
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :]
    both = (torch.norm(f, dim=-1) > threshold).all(dim=1)
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1); v = asset.data.root_lin_vel_b[:, :2].norm(dim=1)
    qd = asset.data.joint_vel.pow(2).mean(1).sqrt()
    quiet = both & (w < s_ang) & (v < s_lin) & (qd < s_jvel)
    t = st["t"]
    # 외란 종료 감지: 어느 슬롯이든 end 가 지금 스텝 안에 들어오면 pending 시작
    ends = _dist_end_time(st); on = st["dst_kind"] > 0
    just_ended = (on & (ends > t.unsqueeze(1) - env.step_dt) & (ends <= t.unsqueeze(1))).any(1)
    st["dst_pending"] = torch.where(just_ended, torch.ones_like(t), st["dst_pending"])
    st["dst_last_end"] = torch.where(just_ended, t, st["dst_last_end"])
    active = (st["dst_F_now"] > 0.0)
    st["dst_quiet_run"] = torch.where(quiet & ~active, st["dst_quiet_run"] + env.step_dt, torch.zeros_like(t))
    pend = st["dst_pending"] > 0.5
    hit = pend & (st["dst_quiet_run"] >= quiet_s) & ((t - st["dst_last_end"]) <= recover_s)
    expired = pend & ((t - st["dst_last_end"]) > recover_s)
    st["dst_pending"] = torch.where(hit | expired, torch.zeros_like(t), st["dst_pending"])
    st["dst_bonus"] = hit.to(torch.float32)
    return st["dst_bonus"]


def action_limit_excess(env, action_term: str = "joint_pos", asset_cfg=None) -> torch.Tensor:
    """정책이 **요청한** 관절 목표가 소프트 한계를 넘은 양의 합 [rad] (N,).

    0922 실물 2회차: 발목 명령이 56.7° 까지 갔다(소프트 한계 39°). stage8 이 잘라 줘서 관절은 38° 에 박혔고,
    정책은 한계 밖을 계속 요청하며 감겼다(windup) → 접지 상실 → 낙상.
    기존 `dof_pos_limits` 벌점은 **실측 관절각** 기준이라 명령이 한계를 넘어도 0 이다 — 그래서 학습이 이를 막지 못했다.
    이 항은 클램프 **전** 요청값을 본다(target = offset + scale × raw_action)."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    term = env.action_manager.get_term(action_term)
    raw = term.raw_actions
    off = getattr(term, "_offset", None)
    scale = getattr(term, "_scale", None)
    if off is None or scale is None:
        return torch.zeros(raw.shape[0], device=raw.device)
    tgt = off + scale * raw
    lim = asset.data.soft_joint_pos_limits[:, term._joint_ids, :] if hasattr(term, "_joint_ids") else asset.data.soft_joint_pos_limits
    over = (tgt - lim[..., 1]).clamp(min=0.0) + (lim[..., 0] - tgt).clamp(min=0.0)
    return over.sum(dim=1)


def set_joint_limits_from_config(env, env_ids, config_path: str, asset_cfg=None, physics: str = "hard", soft: str = "soft", joints: str = ".*_ankle_f_joint"):
    """mode="startup": 배포 config(config/joint_limits_12dof.json, limits_deg[joint] = soft_min/soft_max/hard_min/hard_max)에서
    관절 한계를 읽어 물리 한계(기본 hard)와 soft 한계(벌점·명령 클램프 기준, 기본 soft)를 시뮬에 쓴다 — 승윤님 0920 '발목 한계는
    실제 검증 범위와 먼저 맞춰줘'. USD 에 구운 구값(발목앞뒤 ±30°)을 덮어쓴다. 명시된 관절만 바꾼다."""
    import json
    asset_cfg = asset_cfg or _base_cfg()
    asset = env.scene[asset_cfg.name]
    d = json.load(open(config_path, encoding="utf-8")); lim = d.get("limits_deg", d)
    import re as _re
    names = list(asset.joint_names)
    ids = [k for k, n in enumerate(names) if n in lim and _re.fullmatch(joints, n)]
    if not ids:
        return
    N = asset.data.joint_pos_limits.shape[0]
    hard = torch.stack([torch.tensor([math.radians(lim[names[k]][f"{physics}_min"]), math.radians(lim[names[k]][f"{physics}_max"])]) for k in ids]).to(env.device)
    softl = torch.stack([torch.tensor([math.radians(lim[names[k]][f"{soft}_min"]), math.radians(lim[names[k]][f"{soft}_max"])]) for k in ids]).to(env.device)
    asset.write_joint_position_limit_to_sim(hard.unsqueeze(0).expand(N, -1, -1).clone(), joint_ids=ids, warn_limit_violation=False)
    asset.data.soft_joint_pos_limits[:, ids, :] = softl.unsqueeze(0)
    env._joint_limits_from_config = {names[k]: (float(lim[names[k]][f"{soft}_min"]), float(lim[names[k]][f"{soft}_max"]), float(lim[names[k]][f"{physics}_min"]), float(lim[names[k]][f"{physics}_max"])) for k in ids}
    mask = torch.zeros(len(names), dtype=torch.bool, device=env.device); mask[ids] = True
    env._soft_clamp_mask = mask      # 액션 항 clamp_to_soft_limits 는 이 관절들만 자른다(그 외 관절은 USD soft(0.9×) 로 자르면 기본자세를 깨뜨릴 수 있음)
    print("[limits] config →", {n: v for n, v in env._joint_limits_from_config.items()})


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
                       two_point_d_range=(0.0, 0.0), two_point_kp_range=(200.0, 200.0), two_point_kd_range=(10.0, 10.0),
                       yaw_damp=0.0):
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

    for _k in ("rec_open", "rec_open_t", "rec_settle_run", "rec_ev", "rec_guard"):   # 0921: 회복 게이트 상태도 에피소드마다 초기화(래치 영구화 방지)
        if _k in st:
            st[_k][ids] = 0.0
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
    st["yaw_c"][ids] = float(yaw_damp)   # 0914: 공중 롤 응답 기준 시험에서 요 회전(0913 sig4 오염 원인) 억제용
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


# ─────────────────────── 0918 S9c: 물리 기본자세 편향 DR ───────────────────────
def resample_default_pose_bias(env, env_ids, hip_f_rad=(-0.21, 0.21), knee_rad=(-0.087, 0.087),
                               ankle_f_rad=(-0.14, 0.14), reset_noise_rad: float = 0.05,
                               action_term: str = "joint_pos", asset_cfg=None,
                               shared_hip_f_rad=None, shared_knee_rad=None, shared_ankle_f_rad=None,
                               chain_target_rad=None, chain_leg_noise_rad: float = 0.0,
                               fixed_bias_rad=None):
    """에피소드마다 env 별로 '물리 기본자세'를 nominal + 편향으로 바꾼다.

    모델: 실물 영점 오차·사슬 오프셋 = 정책 프레임(엔코더 D)이 물리 자세와 어긋나는 것. 시뮬 등가 =
    env 의 default_joint_pos(관측 joint_pos_rel 기준 · 액션 오프셋 기준)를 nominal+b 로 두는 것 —
    정책은 nominal 프레임에서 '0' 을 보지만 물리는 b 만큼 다른 자세. 0917 실측: 같은 관절각에서 시뮬 몸통 +5°
    vs 실물 −1~+9°, 수평 서기 힙 15~26°(URDF 9.5°) → 힙 ±12°, 발목 ±8°, 무릎 ±5° 로 시작.
    action term 의 _offset(use_default_offset 캐시)과 관절 상태(리셋 자세)도 같이 갱신한다.
    """
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    names = list(asset.data.joint_names)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not hasattr(env, "_nominal_default_joint_pos"):
        env._nominal_default_joint_pos = asset.data.default_joint_pos[0].clone()
        env._default_pose_bias = torch.zeros_like(asset.data.default_joint_pos)
    nom = env._nominal_default_joint_pos
    bias = torch.zeros(len(ids), len(names), device=env.device)
    for rng, key in ((hip_f_rad, "_hip_f_joint"), (knee_rad, "_knee_joint"), (ankle_f_rad, "_ankle_f_joint")):
        if rng is None:
            continue
        cols = torch.tensor([i for i, n in enumerate(names) if n.endswith(key)], device=env.device)
        bias[:, cols] = torch.empty(len(ids), len(cols), device=env.device).uniform_(float(rng[0]), float(rng[1]))
    # 0918 S9d: 좌우 거울 공유 편향 — 왼쪽 관절 값 기준(rng), 오른쪽은 부호 반전(힙앞뒤·무릎·발목앞뒤는 좌우 반대 부호 규약).
    # 용도 = 양다리에 같은 하중으로 생기는 엔코더 뒤 직렬 처짐(무릎 −8~−17° 식별)을 per-env 한 값으로 가산. 위 독립 편향(영점 오차)과 합산.
    for rng, key in ((shared_hip_f_rad, "_hip_f_joint"), (shared_knee_rad, "_knee_joint"), (shared_ankle_f_rad, "_ankle_f_joint")):
        if rng is None:
            continue
        v = torch.empty(len(ids), 1, device=env.device).uniform_(float(rng[0]), float(rng[1]))
        for i, n in enumerate(names):
            if n.endswith(key):
                bias[:, i] += (v[:, 0] if n.startswith("left") else -v[:, 0])
    # 0928: 사슬 합 보존 — 힙 편향을 무릎·발목 편향에 맞춰 정한다.
    #   독립 힙/무릎 편향은 사슬 합(굴곡 기준 힙−무릎+발목)을 −22~−6° 로 뿌려, 착지 몸통을 실물이
    #   한 번도 보이지 않은 각도로 기울였다(홀드 실패의 주 예측변수, 상관 −0.37).
    #   실물: 오른다리 힙 +7.4°·무릎 +9.1° 가 함께 움직여 사슬 합 ±1.2° 유지(0928 자립),
    #   착지 직후 몸통 −3.6~0°(0918·0928) ↔ 시뮬 사슬 합 −14~−10°(명목 −14 → 몸통 −4 · 자동수평 +3.6 → 0).
    #   → 힙 = 무릎 − 발목 + c + n  (c ~ U(chain_target) env 공통, n ~ U(±chain_leg_noise) 다리별)
    if chain_target_rad is not None:
        c = torch.empty(len(ids), device=env.device).uniform_(float(chain_target_rad[0]), float(chain_target_rad[1]))
        for side, sg in (("left", 1.0), ("right", -1.0)):
            ih = names.index(f"{side}_hip_f_joint"); ik = names.index(f"{side}_knee_joint"); ia = names.index(f"{side}_ankle_f_joint")
            k_flex = -sg * bias[:, ik]           # 굴곡 + 로 통일 (왼 무릎 − / 오른 무릎 + 가 굴곡)
            a_flex = sg * bias[:, ia]
            n = torch.empty(len(ids), device=env.device).uniform_(-float(chain_leg_noise_rad), float(chain_leg_noise_rad))
            h_flex = k_flex - a_flex + c + n
            bias[:, ih] = sg * h_flex
    # 0929: 관절별 고정 편향 (실물 좌우 비대칭 재현 시험). {관절 이름: rad} — 모든 env 에 같은 값 가산.
    #   default_joint_pos 에 들어가므로 물리 자세만 바뀌고 정책 관측(joint_pos_rel)에는 0 으로 보인다 = '숨은 비대칭'.
    if fixed_bias_rad:
        for jn, v in dict(fixed_bias_rad).items():
            bias[:, names.index(jn)] += float(v)
    new = nom.unsqueeze(0) + bias
    asset.data.default_joint_pos[ids] = new
    env._default_pose_bias[ids] = bias
    term = env.action_manager.get_term(action_term)
    off = getattr(term, "_offset", None)
    if isinstance(off, torch.Tensor):
        jid = getattr(term, "_joint_ids", slice(None))
        off[ids] = new[:, jid] if not isinstance(jid, slice) else new
    # 리셋 자세 = 새 기본자세 + 작은 잡음 (reset_robot_joints 가 앞서 구 기본자세로 썼을 수 있으므로 덮어씀)
    q = new + torch.empty_like(new).uniform_(-reset_noise_rad, reset_noise_rad)
    lo, hi = asset.data.soft_joint_pos_limits[ids, :, 0], asset.data.soft_joint_pos_limits[ids, :, 1]
    q = torch.clamp(q, lo, hi)
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


# ─────────────── 0920 엔코더(로터측) 관측 — 유격 안 이동은 실물 엔코더에 보이지 않는다 ───────────────
# 실측(0919): 발목앞뒤 전동부 ~8° 유격·마찰 대역. 하중이 바뀌면 출력(링크)은 최대 8° 움직이지만 로터 엔코더는 1.5° 만 움직인다.
# 시뮬은 발목을 DeadzoneImplicit(유격 반폭 play_rad, rl_walking/actuators_play.py)로 모델링하는데, 기본 관측 joint_pos_rel /
# joint_vel_rel 은 **물리 관절각(출력측)** 을 준다 → 실물 엔코더(모터측) 계약과 다르다. 아래 함수는 로터각을 재구성한다:
#     θ_enc = θ_j + clamp(θ_ref − θ_j, −play, +play)
#   유격 안(|θ_ref−θ_j| ≤ play): 강성 0 이라 로터는 목표에 그대로 앉아 있음 → θ_enc = θ_ref (링크가 어디 있든 엔코더 불변).
#   유격 밖: 로터가 링크를 밀고 있어 θ_enc = θ_j ± play (PD 는 로터–목표 오차로 작동하므로 로터 = 링크 + 경계).
#   θ_ref = 액션 항의 **최종 송출 목표**(SlewJointPositionAction.processed_actions — 지연·슬루·LPF·백래시가 모두 적용된 뒤 값;
#   Articulation.data.joint_pos_target 과 같고, DeadzoneImplicit.compute 는 새 텐서를 만들어 이 값을 오염시키지 않는다 — 0920 감사).
#   유격 0 인 관절은 θ_enc = θ_j 라 기존 joint_pos_rel 과 동일(회귀 안전).
# 리셋 직후(episode_length_buf == 0)에는 전 에피소드의 목표가 남아 있으므로 θ_ref := θ_j 로 두어 θ_enc = θ_j (엔코더 = 링크에서 출발).


def _enc_asset(env, asset_cfg):
    """(asset, joint_ids) — asset_cfg None 이면 'robot'·전 관절. Isaac 없는 더미 env 도 통과."""
    name = getattr(asset_cfg, "name", None) or "robot"
    jid = getattr(asset_cfg, "joint_ids", None)
    if jid is None:
        jid = slice(None)
    return env.scene[name], jid


def _enc_play(env, asset, play_rad=None) -> torch.Tensor:
    """유격 반폭 텐서 (num_envs, num_joints) [rad].
    (a) play_rad 인자: float 또는 {관절정규식(fullmatch): rad}. 어느 정규식도 안 맞으면 ValueError(설정 오타 방지, Isaac 규약과 같음).
    (b) None: 로봇 액추에이터 중 `_play` 텐서를 가진 것(DeadzoneImplicit/DeadzonePD/DeadzoneCoulombImplicit)에서 env·관절별 값을
        joint_indices 로 모아 온다(리셋마다 재샘플되는 DR 값이 그대로 반영). 유격 액추에이터가 없으면 전부 0."""
    q = asset.data.joint_pos
    N, J = q.shape
    if play_rad is not None:
        import re as _re
        cache = getattr(env, "_enc_play_cache", None)
        key = (id(asset), repr(play_rad), J)
        if cache is not None and cache[0] == key and cache[1].device == q.device:
            return cache[1]
        names = list(getattr(asset, "joint_names", None) or asset.data.joint_names)
        vec = torch.zeros(J, device=q.device, dtype=q.dtype)
        if isinstance(play_rad, dict):
            for pat, val in play_rad.items():
                hit = [i for i, n in enumerate(names) if _re.fullmatch(pat, n)]
                if not hit:
                    raise ValueError(f"encoder obs: play_rad 정규식 '{pat}' 이 어느 관절에도 안 맞음: {names}")
                vec[hit] = float(val)
        else:
            vec[:] = float(play_rad)
        p = vec.abs().unsqueeze(0).expand(N, J)
        env._enc_play_cache = (key, p)
        return p
    p = torch.zeros(N, J, device=q.device, dtype=q.dtype)
    acts = getattr(asset, "actuators", None) or {}
    for act in acts.values():
        pl = getattr(act, "_play", None)
        if not isinstance(pl, torch.Tensor):
            continue
        idx = getattr(act, "joint_indices", slice(None))
        p[:, idx] = pl.to(dtype=p.dtype, device=p.device).abs()
    return p


def _enc_target(env, asset, action_term="joint_pos") -> torch.Tensor:
    """θ_ref (num_envs, num_joints): 액션 항의 최종 송출 목표(processed_actions)를 전 관절 공간에 놓는다(항 밖 관절은 θ_j).
    항이 없으면 sim 목표(asset.data.joint_pos_target), 그것도 없으면 θ_j(→ 유격 효과 0)."""
    q = asset.data.joint_pos
    ref = q.clone()
    am = getattr(env, "action_manager", None)
    term = None
    if am is not None and action_term:
        try:
            term = am.get_term(action_term)
        except (KeyError, ValueError, AttributeError):
            term = None
    if term is not None:
        tgt = getattr(term, "processed_actions", None)
        if isinstance(tgt, torch.Tensor) and tgt.shape[0] == q.shape[0]:
            jid = getattr(term, "_joint_ids", None)
            if jid is None:
                jid = slice(None)
            ref[:, jid] = tgt.to(dtype=ref.dtype, device=ref.device)
            return ref
    tgt = getattr(asset.data, "joint_pos_target", None)
    if isinstance(tgt, torch.Tensor) and tgt.shape == q.shape:
        return tgt.clone()
    return ref


def encoder_joint_pos(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None) -> torch.Tensor:
    """로터(엔코더) 절대각 θ_enc (num_envs, |joint_ids|) [rad] — 진단·다른 항의 재료. 정책 관측은 encoder_joint_pos_rel 을 쓴다."""
    asset, jid = _enc_asset(env, asset_cfg)
    q = asset.data.joint_pos
    ref = _enc_target(env, asset, action_term)
    elb = getattr(env, "episode_length_buf", None)
    if isinstance(elb, torch.Tensor) and elb.shape[0] == q.shape[0]:
        fresh = (elb == 0).view(-1, 1)
        if bool(fresh.any()):
            ref = torch.where(fresh, q, ref)
    play = _enc_play(env, asset, play_rad)
    enc = q + torch.clamp(ref - q, -play, play)
    return enc[:, jid]


def encoder_joint_pos_rel(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None) -> torch.Tensor:
    """**엔코더(로터측) 관절각 − 기본자세** (num_envs, |joint_ids|). joint_pos_rel 의 실물 계약판.

    이 관측을 쓰는 env 는 **실물 엔코더 계약(로터측)과 일치**한다: 물리 관절각(출력측)과 엔코더각의 차이 = **유격 안 이동**
    (θ_enc − θ_j = clamp(θ_ref − θ_j, ±play)). 실물에서 하중으로 링크가 유격 안을 8° 움직여도 엔코더는 안 움직이는 것과 같은 식으로,
    정책은 유격 안 이동을 관절각으로는 볼 수 없고 IMU·접촉·(밖으로 밀렸을 때의) 경계값으로만 추정해야 한다.
    - play: play_rad 인자(dict 정규식→rad / float) 또는 로봇 액추에이터(Deadzone*Actuator)의 `_play` 에서 env 별 자동 획득(_enc_play).
    - θ_ref: action_term(기본 "joint_pos") 의 processed_actions = 지연·슬루·LPF·백래시가 모두 적용된 실제 송출 목표(_enc_target).
    - 유격 0 인 관절·리셋 직후·유격 액추에이터가 없는 env 에서는 joint_pos_rel 과 동일.
    - default_joint_pos 는 asset.data 의 현재값(resample_default_pose_bias 가 바꾼 env 별 기본자세 포함)."""
    asset, jid = _enc_asset(env, asset_cfg)
    enc = encoder_joint_pos(env, asset_cfg, action_term, play_rad)
    return enc - asset.data.default_joint_pos[:, jid]


def encoder_link_gap(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None) -> torch.Tensor:
    """θ_enc − θ_j (num_envs, |joint_ids|): 정책이 못 보는 유격 안 이동. **크리틱 특권 관측·진단·기록용** (actor 에 넣지 않는다)."""
    asset, jid = _enc_asset(env, asset_cfg)
    enc = encoder_joint_pos(env, asset_cfg, action_term, play_rad)
    return enc - asset.data.joint_pos[:, jid]


def encoder_joint_vel(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None) -> torch.Tensor:
    """**엔코더각 θ_enc 의 유한차분 속도** (num_envs, |joint_ids|) [rad/s], dt = env.step_dt. joint_vel_rel 의 실물 계약판
    (실물 브리지 qvel 도 로터 엔코더 관측의 유한차분). 물리 관절속도(출력측)와 달리 유격 안 이동은 속도에도 나타나지 않는다.
    - 버퍼 env._enc_prev(전 관절 θ_enc)·env._enc_step(마지막 갱신 스텝)·env._enc_vel(마지막 값). 같은 스텝(common_step_counter)
      안의 재호출(예: encoder_joint_vel_ema 가 같은 스텝에 다시 부름, 크리틱 그룹)은 버퍼를 다시 밀지 않고 같은 값을 준다.
    - 리셋 직후(episode_length_buf == 0)에는 0 이고 버퍼를 현재 θ_enc 로 다시 시작한다(전 에피소드와의 차분 금지).
    - default_joint_vel 이 있으면 뺀다(joint_vel_rel 규약, 이 로봇은 0)."""
    asset, jid = _enc_asset(env, asset_cfg)
    enc = encoder_joint_pos(env, asset_cfg, action_term, play_rad)   # (N, |jid|)
    full = enc
    step = int(getattr(env, "common_step_counter", 0))
    prev = getattr(env, "_enc_prev", None)
    if prev is None or prev.shape != full.shape or prev.device != full.device:
        env._enc_prev = full.clone()
        env._enc_step = step
        env._enc_vel = torch.zeros_like(full)
    elif getattr(env, "_enc_step", None) != step:
        dt = float(getattr(env, "step_dt", 0.02))
        env._enc_vel = (full - prev) / dt
        env._enc_prev = full.clone()
        env._enc_step = step
    vel = env._enc_vel.clone()
    elb = getattr(env, "episode_length_buf", None)
    if isinstance(elb, torch.Tensor) and elb.shape[0] == full.shape[0]:
        fresh = elb == 0
        if bool(fresh.any()):
            vel[fresh] = 0.0
            env._enc_prev[fresh] = full[fresh]
            env._enc_vel[fresh] = 0.0
    dv = getattr(asset.data, "default_joint_vel", None)
    if isinstance(dv, torch.Tensor) and dv.shape[0] == vel.shape[0]:
        vel = vel - dv[:, jid]
    return vel


def encoder_joint_vel_ema(env, alpha: float = 0.25, noise: float = 1.5, asset_cfg=None,
                          action_term: str = "joint_pos", play_rad=None) -> torch.Tensor:
    """노이즈 낀 **엔코더 속도**의 EMA — joint_vel_ema(S8a) 와 같은 규약(alpha·noise 는 함수 안에서, cfg noise=None; 버퍼 _ema_state 로
    episode_length_buf ≤ 1 에서 현재값 재시작). 버퍼 키 '_s8_enc_jvel_ema' (기존 '_s8_jvel_ema' 와 독립)."""
    x = encoder_joint_vel(env, asset_cfg, action_term, play_rad)
    if noise > 0.0:
        x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
    return _ema_state(env, "_s8_enc_jvel_ema", x, alpha).clone()


def _selftest_encoder():
    """torch 만으로 θ_enc 식 검증 (pytest: tests/test_encoder_obs.py 가 더미 env 로 전 경로를 덮는다)."""
    q = torch.tensor([[0.10, -0.20]]); ref = torch.tensor([[0.13, -0.40]]); play = torch.tensor([[0.07, 0.07]])
    enc = q + torch.clamp(ref - q, -play, play)
    assert torch.allclose(enc, torch.tensor([[0.13, -0.27]]))     # 안: = ref / 밖: = q − play
    assert torch.allclose(q + torch.clamp(ref - q, -0.0, 0.0), q)  # 유격 0: = q
    print("[selftest] encoder obs: 유격 안 θ_ref·밖 θ_j±play·유격 0 항등 — PASS")


# ─────────────────────── 0920 엔코더측 관측 (유격 관절: 실물 로터 엔코더는 유격 안 이동을 못 봄) ───────────────────────
def _play_by_joint(env, asset):
    """(N, J) 유격 반폭 [rad]. DeadzoneImplicitActuator 계열의 _play 텐서를 관절 인덱스에 배치, 그 외 0."""
    key = "_enc_play_cache"
    N = asset.data.joint_pos.shape[0]
    play = torch.zeros_like(asset.data.joint_pos)
    for act in asset.actuators.values():
        pl = getattr(act, "_play", None)
        if pl is None:
            continue
        idx = act.joint_indices
        if isinstance(idx, slice):
            play[:, idx] = pl
        else:
            play[:, torch.as_tensor(idx, device=play.device)] = pl
    return play


def _encoder_pos(env, asset, action_term="joint_pos"):
    """θ_enc = θ_j + clamp(θ_ref − θ_j, −play, +play): 유격 안에서는 모터가 목표에 붙어 있고(θ_enc≈θ_ref), 밖에서는 유격만큼 어긋난 채 링크를 민다."""
    q = asset.data.joint_pos
    try:
        term = env.action_manager.get_term(action_term)
        ref = term.processed_actions
    except Exception:
        ref = q
    play = _play_by_joint(env, asset)
    return q + torch.clamp(ref - q, -play, play)


def encoder_joint_pos_rel(env, asset_cfg=None, action_term: str = "joint_pos") -> torch.Tensor:
    """엔코더각 − 기본자세 (12). 유격 0 이면 mdp.joint_pos_rel 과 동일. 실물 계약: 로터측 엔코더."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    return _encoder_pos(env, asset, action_term) - asset.data.default_joint_pos


def _encoder_vel(env, asset, action_term):
    enc = _encoder_pos(env, asset, action_term)
    prev = getattr(env, "_enc_prev", None)
    if prev is None or prev.shape != enc.shape:
        prev = enc.clone(); env._enc_prev = prev
    fresh = env.episode_length_buf <= 1
    prev[fresh] = enc[fresh]
    vel = (enc - prev) / float(env.step_dt)
    env._enc_prev = enc.clone()
    return vel


def encoder_joint_vel(env, asset_cfg=None, action_term: str = "joint_pos") -> torch.Tensor:
    """엔코더각 유한차분 속도 (12). 유격 안 링크 이동은 0 으로 보인다(실물과 동일)."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    return _encoder_vel(env, asset, action_term)


def encoder_joint_vel_ema(env, alpha: float = 0.25, noise: float = 1.5, asset_cfg=None, action_term: str = "joint_pos") -> torch.Tensor:
    """엔코더 속도의 EMA (12) — joint_vel_ema 와 같은 규약(노이즈·alpha·리셋)."""
    asset = env.scene[(asset_cfg or _base_cfg()).name]
    x = _encoder_vel(env, asset, action_term)
    if noise > 0.0:
        x = x + (torch.rand_like(x) * 2.0 - 1.0) * noise
    return _ema_state(env, "_s8_enc_jvel_ema", x, alpha).clone()
