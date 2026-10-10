"""1009 새 몸(골반 회전 hip_r 양쪽을 PLA 출력물로 0° 에 굳힌 구성) 학습 환경 — 학습 전용, 기존 파일 불변.

배경: reports/real_1009/lock_eval/RESULT.md, 사전 기준 reports/real_1009/NEWBODY_TRAIN.md.
G24R-a(Biped12HangStandG24RaEnvCfg) 설정에 아래만 더한다. 보상·외란·무작위화·지연·게인·관측/행동 차원(320/12)은 그대로.

  1) 물리 고정: left/right_hip_r_joint 의 위치 한계를 env 마다 한 번 뽑은 고정각 θ ~ U(±2°) 로 묶는다(startup, 하한 = 상한).
     θ 는 학습 실행 내내 그 env 에 고정(에피소드가 바뀌어도 같은 값) — 실물 조립 오차를 가정한 범위다(**가정**, 실측 없음).
     리셋 때 그 관절 상태를 θ·속도 0 으로 다시 놓는다(reset_robot_joints·default_pose_bias 가 쓴 값 덮어씀).
     기본자세(default_joint_pos)는 0° 로 되돌려 둔다 — 한계 쓰기가 기본자세를 θ 로 잘라 넣는데, 그대로 두면
     resample_default_pose_bias 가 env 0 의 θ 를 전 env 의 명목 기본자세로 잡는다.
     정책의 그 관절 명령은 평가기(eval_ground_recovery_lock.py)와 같이 한계에 막혀 무시된다(목표각은 자르지 않는다 —
     PD 토크가 한계와 내부에서 상쇄. 토크 벌점 항에는 그 토크가 보인다).
  2) 질량: 변형 URDF(biped12_final3_hiprpla.urdf)의 네 링크 inertial(질량·질량 중심·관성 텐서)을 startup 마지막에 덮어쓴다.
     base 의 질량·질량 중심 무작위화(add_base_mass 0–0.5 kg, base_com)는 다른 링크라 그대로 남는다.
     asset.data.default_mass 도 같이 고친다(보상의 체중 정규화 mg 가 새 몸 무게를 쓰도록).
  3) 관측 계약: 실물 stage8 은 기구 고정 관절의 피드백을 합성한다(실측 = stand_zero_deg, 속도 0 — 커밋 b15c343e).
     그래서 정책이 보는 그 관절의 joint_pos·joint_vel·joint_vel_ema 열은 **정확히 0**(잡음도 없음)으로 둔다.
     직전 행동(actions) 열은 정책 자신의 출력이라 그대로다(브리지도 정책 입력·LPF 상태는 그대로 둔다).

이벤트 순서: 이벤트 관리자는 cfg.__dict__ 순서로 돈다. 아래 세 항은 __post_init__ 끝에서 새 속성으로 붙어 각 모드의 마지막이 된다
(joint_limits_config·add_base_mass·base_com·link_mass_calib 뒤). 실행 중 확인은 reports/real_1009/newbody/check_newbody_env.py.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .hang_stand_env_cfg import Biped12HangStandG24RaEnvCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
NB_URDF = os.path.join(_HERE, "assets", "biped12_final3_hiprpla.urdf")
NB_LOCK_JOINTS = ("left_hip_r_joint", "right_hip_r_joint")
NB_LINKS = ("left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint")
NB_LOCK_OFFSET_DEG = (-2.0, 2.0)      # 가정: 실물 조립 오차 범위(실측 없음)
NB_NOMINAL_TOTAL_KG = 16.2595         # 변형 URDF 링크 질량 합(무작위화 전)


def _read_urdf_inertials(urdf_path: str, links) -> dict:
    root = ET.parse(urdf_path).getroot()
    by = {l.get("name"): l.find("inertial") for l in root.findall("link")}
    out = {}
    for nm in links:
        it = by[nm]; o = it.find("origin"); I = it.find("inertia")
        assert all(abs(float(v)) < 1e-12 for v in (o.get("rpy") or "0 0 0").split()), f"{nm}: inertial rpy ≠ 0 은 지원 안 함"
        g = lambda k: float(I.get(k))  # noqa: E731
        out[nm] = {
            "mass": float(it.find("mass").get("value")),
            "com": [float(v) for v in o.get("xyz").split()],
            # 규약(1009 평가기에서 자산 읽음값으로 확인): PhysX 뷰 get/set_inertias 9 성분 = 링크 축 기준 전체 텐서
            "inertia": [g("ixx"), g("ixy"), g("ixz"), g("ixy"), g("iyy"), g("iyz"), g("ixz"), g("iyz"), g("izz")],
        }
    return out


def nb_apply_link_inertials(env, env_ids, urdf_path: str = NB_URDF, links=NB_LINKS):
    """mode="startup"(마지막): 변형 URDF 의 inertial 로 네 링크를 덮어쓰고, 적용 뒤 읽은 값으로 확인한다. 불일치면 중단."""
    asset = env.scene["robot"]; pv = asset.root_physx_view
    names = list(asset.body_names); spec = _read_urdf_inertials(urdf_path, links)
    m0 = pv.get_masses().clone(); c0 = pv.get_coms().clone(); i0 = pv.get_inertias().clone()
    m1 = m0.clone(); c1 = c0.clone(); i1 = i0.clone()
    eid = torch.arange(m0.shape[0], device=m0.device)
    for nm, s in spec.items():
        b = names.index(nm)
        m1[:, b] = s["mass"]; c1[:, b, :3] = torch.tensor(s["com"], dtype=c0.dtype); i1[:, b] = torch.tensor(s["inertia"], dtype=i0.dtype)
    pv.set_masses(m1, eid); pv.set_coms(c1, eid); pv.set_inertias(i1, eid)
    m2 = pv.get_masses(); c2 = pv.get_coms(); i2 = pv.get_inertias()
    parts = []; ok = True
    for nm, s in spec.items():
        b = names.index(nm)
        em = float((m2[:, b] - s["mass"]).abs().max()); ec = float((c2[:, b, :3].cpu() - torch.tensor(s["com"])).abs().max())
        ei = float((i2[:, b].cpu() - torch.tensor(s["inertia"])).abs().max())
        good = em < 1e-6 and ec < 1e-6 and ei < 1e-8; ok = ok and good
        parts.append(f"{nm} {float(m0[0, b]):.5f}→{float(m2[0, b]):.5f} kg(목표 {s['mass']:.5f}) com {[round(float(v), 5) for v in c2[0, b, :3]]} "
                     f"오차 질량 {em:.1e}·com {ec:.1e}·관성 {ei:.1e} {'일치' if good else '불일치!'}")
    # 보상의 체중 정규화(mg = g·default_mass 합)가 새 몸 무게를 쓰도록 기본 질량도 같이 고친다(base 는 건드리지 않음)
    dm = asset.data.default_mass; others = [k for k in range(len(names)) if names[k] not in spec]
    for nm, s in spec.items():
        dm[:, names.index(nm)] = s["mass"]
    tot = m2.sum(-1); base_extra = (m2[:, others].sum(-1) - m0[0:1, others].sum(-1))   # env 0 대비가 아니라 아래에서 명목과 비교
    nominal_now = float(dm.sum(-1).mean())
    info = (f"[newbody][질량] {os.path.basename(urdf_path)} | " + " || ".join(parts)
            + f" | 전체 질량: 종전 env 평균 {float(m0.sum(-1).mean()):.4f} → 지금 env 평균 {float(tot.mean()):.4f} kg (범위 {float(tot.min()):.4f}–{float(tot.max()):.4f}; "
              f"범위는 base 질량 무작위화 몫) · default_mass 합(무작위화 전 명목) {nominal_now:.4f} kg (기대 {NB_NOMINAL_TOTAL_KG})")
    print(info, flush=True)
    env._nb_mass = {"m_set": m2.clone(), "c_set": c2.clone(), "i_set": i2.clone(), "info": info, "links": list(spec), "nominal": nominal_now}
    del base_extra
    if not ok or abs(nominal_now - NB_NOMINAL_TOTAL_KG) > 2e-3:
        raise RuntimeError("[newbody] 링크 inertial 적용 확인 실패 — 학습하지 않음: " + info)


def nb_lock_joints_startup(env, env_ids, joint_names=NB_LOCK_JOINTS, offset_deg=NB_LOCK_OFFSET_DEG, band_deg: float = 0.0):
    """mode="startup"(joint_limits_config 뒤): env·관절마다 고정각 θ ~ U(offset_deg) 를 한 번 뽑아 위치 한계를 [θ−band, θ+band] 로 묶는다."""
    asset = env.scene["robot"]; names = list(asset.data.joint_names)
    ids = [names.index(n) for n in joint_names]; N = asset.data.joint_pos.shape[0]
    tgt = torch.empty(N, len(ids), device=env.device).uniform_(math.radians(offset_deg[0]), math.radians(offset_deg[1]))
    hb = math.radians(float(band_deg))
    old = asset.data.joint_pos_limits[0, ids].clone(); d0 = asset.data.default_joint_pos[:, ids].clone()
    asset.write_joint_position_limit_to_sim(torch.stack([tgt - hb, tgt + hb], dim=-1), joint_ids=ids, warn_limit_violation=False)
    asset.data.default_joint_pos[:, ids] = d0        # 한계 쓰기가 기본자세를 θ 로 잘라 넣는다 → 명목(0°)으로 되돌림(모듈 설명 참조)
    asset.write_joint_state_to_sim(tgt, torch.zeros_like(tgt), joint_ids=ids)
    env._nb_lock = {"ids": ids, "names": list(joint_names), "tgt": tgt, "band": hb}
    print(f"[newbody][고정] {list(joint_names)} ids {ids} · 고정각 θ ~ U({offset_deg[0]}, {offset_deg[1]})° env 별(가정) · 읽음: 최소 {float(torch.rad2deg(tgt.min())):.3f}° "
          f"최대 {float(torch.rad2deg(tgt.max())):.3f}° 평균 {float(torch.rad2deg(tgt.mean())):.3f}° · 반폭 {band_deg}° · 종전 한계(env0) {torch.rad2deg(old).cpu().tolist()}° "
          f"· 기본자세(env0) {torch.rad2deg(asset.data.default_joint_pos[0, ids]).cpu().tolist()}°", flush=True)


def nb_lock_joints_reset(env, env_ids):
    """mode="reset"(마지막): 고정 관절 상태를 θ·속도 0 으로 다시 놓는다."""
    st = getattr(env, "_nb_lock", None)
    if st is None:
        return
    asset = env.scene["robot"]
    e = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    q = st["tgt"][e]
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=st["ids"], env_ids=e)


def nb_obs_fixed_zero(env, func, params: dict, joint_names=NB_LOCK_JOINTS, noise=None) -> torch.Tensor:
    """관측 감싸개: 원 관측 함수 값(+ 원 설정의 균일 잡음)을 낸 뒤 기구 고정 관절 열만 정확히 0 으로 둔다(실물 합성 피드백 계약).
    noise=(n_min, n_max): 원 ObsTerm 의 가산 균일 잡음을 여기서 더한다(항의 noise 는 None 으로 옮김 — 고정 열에 잡음이 남지 않게)."""
    x = func(env, **params).clone()
    if noise is not None:
        x = x + torch.empty_like(x).uniform_(float(noise[0]), float(noise[1]))
    cols = getattr(env, "_nb_obs_cols", None)
    if cols is None:
        names = list(env.scene["robot"].data.joint_names)
        assert x.shape[-1] == len(names), f"관측 열 수 {x.shape[-1]} ≠ 관절 수 {len(names)}"
        cols = [names.index(n) for n in joint_names]; env._nb_obs_cols = cols
    x[:, cols] = 0.0
    return x


@configclass
class Biped12HangStandG24RaNBEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-a-NB (1009): G24R-a + **새 몸**(hip_r 양쪽 물리 고정 θ~U(±2°) env 별 · 네 링크 inertial = 변형 URDF · 고정 관절 관측 열 0).
    보상·외란 분포·무작위화·지연·게인·PPO·관측/행동 차원 불변 → G24Ra-21793 에서 이어 학습. 학습 전용(관측 감싸개가 잡음을 항상 더한다)."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        for k in ("nb_link_inertials", "nb_lock_startup", "nb_lock_reset"):
            assert getattr(ev, k, None) is None, f"이벤트 이름 충돌 {k}"
        # 새 속성 = __dict__ 끝 → 각 모드의 마지막 항
        ev.nb_link_inertials = EventTerm(func=nb_apply_link_inertials, mode="startup", params={"urdf_path": NB_URDF, "links": NB_LINKS})
        ev.nb_lock_startup = EventTerm(func=nb_lock_joints_startup, mode="startup",
                                       params={"joint_names": NB_LOCK_JOINTS, "offset_deg": NB_LOCK_OFFSET_DEG, "band_deg": 0.0})
        ev.nb_lock_reset = EventTerm(func=nb_lock_joints_reset, mode="reset", params={})
        P = self.observations.policy
        for name in ("joint_pos", "joint_vel", "joint_vel_ema"):
            t = getattr(P, name)
            nz = t.noise
            if nz is not None:
                assert getattr(nz, "operation", "add") == "add" and "uniform" in str(nz.func), f"{name}: 가산 균일 잡음만 옮길 수 있음 ({nz})"
            t.params = {"func": t.func, "params": dict(t.params or {}), "joint_names": NB_LOCK_JOINTS,
                        "noise": (float(nz.n_min), float(nz.n_max)) if nz is not None else None}
            t.func = nb_obs_fixed_zero
            t.noise = None


# ───────────────────────── 1010 2세대: 앞 외란 힘 범위 상한만 평가 조건까지 확대 (G24RaNBf) ─────────────────────────
NBF_FRONT_RAMP_N = (10.0, 15.0)     # 종전 R3 앞 램프 10–12 N → 상한 15 N(평가 앞 램프 15 N)
NBF_FRONT_IMP_N = (15.0, 40.0)      # 종전 R3 앞 충격 15–18 N(= 램프 범위 × 1.5) → 상한 40 N(평가 앞 40 N × 0.2 s)


def nbf_disturbance_reset(env, env_ids, base: dict, front_ramp_N=NBF_FRONT_RAMP_N, front_imp_N=NBF_FRONT_IMP_N):
    """mode="reset": 원 함수 mdp_hang.disturbance_reset_g22r(G24R-a 인자 그대로)를 부른 뒤, **앞 범주 env 의 힘 크기만** 다시 뽑는다.
    램프(kind 1) ~ U(front_ramp_N), 충격(kind 2) ~ U(front_imp_N). 범주 비율·횟수·시각·상승/유지/지속 시간·방향·뒤 범주·무외란 범주는 원 함수 값 그대로.
    하한은 종전과 같다(램프 10 N, 충격 15 N). 충격 지속 0.2–0.4 s 도 그대로라 충격량 상한은 7.2 → 16 N·s 가 된다(평가 8 N·s)."""
    from . import mdp_hang as _mh
    _mh.disturbance_reset_g22r(env, env_ids, **base)
    hs = _mh._state(env); g = _mh._g22(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if len(ids) == 0:
        return
    kind = hs["dst_kind"][ids]; F = hs["dst_F"][ids]; front = (g["cat"][ids] == 1).unsqueeze(1)
    U = lambda r: torch.empty_like(F).uniform_(float(r[0]), float(r[1]))  # noqa: E731
    F = torch.where(front & (kind == 1), U(front_ramp_N), F)
    F = torch.where(front & (kind == 2), U(front_imp_N), F)
    hs["dst_F"][ids] = F


@configclass
class Biped12HangStandG24RaNBfEnvCfg(Biped12HangStandG24RaNBEnvCfg):
    """G24R-a-NBf (1010): G24RaNB + **앞 외란 힘 범위 상한만 확대**(앞 램프 10–12 → 10–15 N, 앞 충격 15–18 → 15–40 N) 단일 변경.
    하한·뒤·무외란 비율·시간·다른 이벤트·보상·질량·고정·관측 전부 G24RaNB 와 같다. 사전 등록 reports/real_1009/NEWBODY_TRAIN.md §9."""

    def __post_init__(self):
        super().__post_init__()
        d = self.events.disturbance
        d.params = {"base": dict(d.params), "front_ramp_N": NBF_FRONT_RAMP_N, "front_imp_N": NBF_FRONT_IMP_N}      # base = 원 함수 인자 그대로
        d.func = nbf_disturbance_reset


# ───────────────────────── 1010 2단계(종종걸음): 새 몸을 W3 걷기 태스크에 얹기 ─────────────────────────
from .hang_stand_env_cfg import Biped12HangStandW3S3dEnvCfg  # noqa: E402


def nb_apply_newbody(cfg):
    """환경 설정 cfg 에 새 몸 세 가지(네 링크 inertial · hip_r 물리 고정 θ~U(±2°) · 고정 관절 관측 열 0)를 얹는다.
    Biped12HangStandG24RaNBEnvCfg.__post_init__ 과 같은 내용(그 클래스는 그대로 둔다). 주의: reset 의 stance_asym 이 hip_r 에 넣는 발 요 ±5° 는
    마지막 reset 항 nb_lock_reset 이 θ 로 덮는다 → 새 몸에서 발 요 어긋남 = 고정각 θ(±2°)뿐이다(G24RaNB 도 같다)."""
    ev = cfg.events
    for k in ("nb_link_inertials", "nb_lock_startup", "nb_lock_reset"):
        assert getattr(ev, k, None) is None, f"이벤트 이름 충돌 {k}"
    ev.nb_link_inertials = EventTerm(func=nb_apply_link_inertials, mode="startup", params={"urdf_path": NB_URDF, "links": NB_LINKS})
    ev.nb_lock_startup = EventTerm(func=nb_lock_joints_startup, mode="startup",
                                   params={"joint_names": NB_LOCK_JOINTS, "offset_deg": NB_LOCK_OFFSET_DEG, "band_deg": 0.0})
    ev.nb_lock_reset = EventTerm(func=nb_lock_joints_reset, mode="reset", params={})
    P = cfg.observations.policy
    for name in ("joint_pos", "joint_vel", "joint_vel_ema"):
        t = getattr(P, name)
        nz = t.noise
        if nz is not None:
            assert getattr(nz, "operation", "add") == "add" and "uniform" in str(nz.func), f"{name}: 가산 균일 잡음만 옮길 수 있음 ({nz})"
        t.params = {"func": t.func, "params": dict(t.params or {}), "joint_names": NB_LOCK_JOINTS,
                    "noise": (float(nz.n_min), float(nz.n_max)) if nz is not None else None}
        t.func = nb_obs_fixed_zero
        t.noise = None


@configclass
class Biped12HangStandW3S3dNBEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3d-NB (1010): 현 걷기 후보의 학습 태스크 W3S3d + 새 몸(nb_apply_newbody)만. 명령·보상·커리큘럼·지연·관측/행동 차원 불변.
    용도: 현 걷기 정책(W3S3d-15800)을 새 몸에 그대로 넣은 기준선 평가(eval_w3_nb.py). 학습용 변형은 reports/real_1009/NEWBODY_WALK.md."""

    def __post_init__(self):
        super().__post_init__()
        nb_apply_newbody(self)


NBJ_MIX3 = (0.5, 0.5, 0.0, 0.0)     # STEP 중 (제자리, 앞뒤, 옆, 회전) — 원 W3S3d (0.25, 0.2, 0.25, 0.3)


@configclass
class Biped12HangStandW3S3dNBjEnvCfg(Biped12HangStandW3S3dNBEnvCfg):
    """W3S3d-NBj (1010 '종종걸음'): W3S3dNB + **명령 혼합만 변경** — STEP 중 제자리 0.5·앞뒤 0.5(앞 0.03–0.10·뒤 0.03–0.06 반반, 원 설정)·옆 0·회전 0.
    STAND 30 %·보상·참조·관측·지연·커리큘럼 항·질량·고정은 W3S3dNB 와 같다(회전 보상 w3_turn·w3_ytrack 은 wz = 0 에서 0).
    사전 기준·계획 reports/real_1009/NEWBODY_WALK.md. W3S3d-15800 에서 이어 학습."""

    def __post_init__(self):
        super().__post_init__()
        p = dict(self.events.w3_cmd.params); p["mix3"] = NBJ_MIX3
        self.events.w3_cmd.params = p


# ---------------------------------------------------------------- 1010 걷는 중 좌우 흔들림 시험(느낌 보기) ----------------------------------------------------------------
def nb_roll_sq(env) -> torch.Tensor:
    """몸통 좌우 기울기 제곱 g_y²(≈ 롤 rad²) — 벌점 크기(양수). STEP 게이트는 w3_step_gated 가 건다."""
    return env.scene["robot"].data.projected_gravity_b[:, 1] ** 2


@configclass
class Biped12HangStandW3S3dNBjREnvCfg(Biped12HangStandW3S3dNBjEnvCfg):
    """W3S3d-NBjR (1010 승윤님 '조금만 해봐 느낌만 보자'): NBj 와 전부 같고 **보상 항 하나만 추가** — 걷는 중(STEP) 좌우 기울기 제곱 벌점.
    근거(측정): NBj-s2-16299 걷기 좌우 기울기 폭 17°(RMS 5.5° → g_y² 평균 ≈ 0.009), 기존 w3_orient(−2)는 초당 −0.02 로 사실상 무효.
    가중 −30 = 걷기 초당 약 −0.28. 시작 = NBj-s2-16299 이어 학습(300 반복). 승격용이 아니라 반응을 보는 한 점 — 기준은 NEWBODY_WALK.md §15."""
    W_ROLL = -30.0          # 비주석 클래스 상수(설정 필드 아님)

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_walking import w3 as _w3
        self.rewards.w3_roll_sq = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_ROLL), params={"func": nb_roll_sq, "params": {}})


# ---------------------------------------------------------------- 1010 빠른 걸음(승윤님 "걸음 빠르게") ----------------------------------------------------------------
NBJF_F_STEP = 1.5       # 위상 주파수 [Hz] — 한 주기 = 왼·오른 한 걸음씩. 종전 1.1(한발 지지 0.455 s) → 1.5(0.333 s)


@configclass
class Biped12HangStandW3S3dNBjFEnvCfg(Biped12HangStandW3S3dNBjEnvCfg):
    """W3S3d-NBjF (1010): NBj 와 전부 같고 **걸음 시계 주파수만** 1.1 → 1.5 Hz. 보상·참조·관측 차원·명령·지연·질량·고정 불변
    (위상 기반 항은 env._gf 를 읽어 따라온다; 한 주기 평균 창 _v_box 등의 길이 1/1.1 s 는 그대로 — 평균 창이 1.36 주기가 된다).
    근거: 걷는 중 좌우 기울기 폭 17°·옆 이동 11.6 cm 는 느린 걸음(한발 지지 0.455 s)의 무게 이동. 선형 역진자 어림으로 옆 이동이 약 40 % 준다.
    배포 계약 변경: 러너 W3Config.f_step 을 같은 값으로(새 키). 시작 = NBj-s2-16299 이어 학습. 기준 reports/real_1009/NEWBODY_WALK.md §16."""

    def __post_init__(self):
        super().__post_init__()
        p = dict(self.events.w3_gait.params); p["f_step"] = float(NBJF_F_STEP)
        self.events.w3_gait.params = p


# ---------------------------------------------------------------- 1010 밤: 과요구 벌점(실물 모터 8 정지 뒤) ----------------------------------------------------------------
def nb_over_demand(env, thresh: float = 0.8, joint_names=NB_LOCK_JOINTS) -> torch.Tensor:
    """PD 가 **요구한** 토크(자르기 전 computed_torque)의 한계 대비 비율이 thresh 를 넘는 초과분 제곱합 — 굳은 관절 제외. 벌점 크기(양수).
    왜: 시뮬 구동기는 한계를 넘는 요구를 조용히 자른다(applied_torque ≤ 한계) → 정책이 목표를 크게 튀겨도 공짜다. 실물 MIT 구동기는 kp·오차를 그대로 받는다
    (1010 실물: 걷기 켠 0.19 s 뒤 right_hip_a 목표−실측 14° = 요구 약 37 N·m → 모터 8 응답 정지). 기존 torque_saturation 은 잘린 값이라 비율이 1 에서 멈춘다."""
    robot = env.scene["robot"]
    c = getattr(env, "_nb_od_cols", None)
    if c is None:
        lock = set(robot.find_joints(list(joint_names))[0])
        c = env._nb_od_cols = torch.tensor([j for j in range(robot.num_joints) if j not in lock], device=env.device)
    r = robot.data.computed_torque[:, c].abs() / (robot.data.joint_effort_limits[:, c] + 1e-6)
    return torch.clamp(r - thresh, min=0.0).pow(2).sum(dim=1)


@configclass
class Biped12HangStandW3S3dNBjTEnvCfg(Biped12HangStandW3S3dNBjEnvCfg):
    """W3S3d-NBjT (1010 밤, 승윤님 "토크 상한이랑 급격한 튐에 벌점"): NBj 와 전부 같고 **보상 항 하나만 추가** — 걷는 중 과요구 벌점(nb_over_demand, 문턱 0.8).
    가중 −5(어림: 힙 외전이 12 % 틱 동안 요구비 1.5 → 초당 약 −0.3). 시작 = NBj-s2-16299 이어 학습. 기준 reports/real_1009/NEWBODY_WALK.md §17."""
    W_OD = -5.0             # 비주석 클래스 상수(설정 필드 아님)

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_walking import w3 as _w3
        self.rewards.w3_over_demand = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_OD), params={"func": nb_over_demand, "params": {}})


# ---------------------------------------------------------------- 1011 낮은 토크로 걷기(승윤님 "정책을 새로 구우는 방향 — 30 은 심해") ----------------------------------------------------------------
def nb_resample_effort_by_joint(env, env_ids, named: dict | None = None, other_range: tuple = (1.0, 1.0)):
    """리셋마다 관절 토크 한계 배율을 다시 뽑는다(env 별 한 값씩). named = {관절 이름 조각: (lo, hi)} — 조각이 든 관절은 그 범위, 나머지는 other_range.
    mdp_stage4.resample_effort_scale 과 같은 쓰기 경로(PhysX 한계 + 액추에이터 모델 한계). 정책은 한계를 관측하지 못한다."""
    robot = env.scene["robot"]
    if not hasattr(env, "_effort_nominal"):
        env._effort_nominal = robot.data.joint_effort_limits.clone()
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if len(ids) == 0:
        return
    s = torch.empty(len(ids), 1, device=env.device).uniform_(*other_range).repeat(1, robot.num_joints)
    for key, (lo, hi) in (named or {}).items():
        cols = [j for j, n in enumerate(robot.joint_names) if key in n]
        assert cols, f"nb_resample_effort_by_joint: 맞는 관절 없음 {key!r}"
        s[:, cols] = torch.empty(len(ids), 1, device=env.device).uniform_(lo, hi)
    new = env._effort_nominal[ids] * s
    robot.write_joint_effort_limit_to_sim(new, env_ids=ids)
    for act in robot.actuators.values():
        if hasattr(act, "effort_limit") and torch.is_tensor(act.effort_limit) and act.effort_limit.dim() == 2:
            act.effort_limit[ids] = new[:, act.joint_indices]


@configclass
class Biped12HangStandW3S3dNBjLEnvCfg(Biped12HangStandW3S3dNBjTEnvCfg):
    """W3S3d-NBjL (1011): NBjT(과요구 벌점 −5, 문턱 = 그 env 의 한계 × 0.8) + **힙 외전 토크 한계를 리셋마다 0.55–0.8 배(13.75–20 N·m), 나머지 관절 0.8–1.0 배**.
    근거(시뮬 NBj-s2, hip_a 한계만): 25 N·m 낙상 6/214 · 20 → 8 · 17.5 → 14 · 15 → 39/214, 좌우 기울기 폭 17 → 13 → 9 → 5°. 지금 정책은 15 에서 무너진다.
    목표: 힙 외전 15–20 N·m 안에서 성립하는 걸음(실물 상한 20 이 거의 걸리지 않는 걸음). 벌점만(NBjT)으로는 요구 p95 37–38 N·m·낙상 증가였다.
    시작 = NBj-s2-16299 이어 학습. 기준 reports/real_1009/NEWBODY_WALK.md §18."""
    HIP_A_SCALE = (0.55, 0.8)       # 비주석 클래스 상수(설정 필드 아님)
    OTHER_SCALE = (0.8, 1.0)

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.nb_effort = EventTerm(func=nb_resample_effort_by_joint, mode="reset",
                                          params={"named": {"hip_a": tuple(self.HIP_A_SCALE)}, "other_range": tuple(self.OTHER_SCALE)})


@configclass
class Biped12HangStandW3S3dNBjMEnvCfg(Biped12HangStandW3S3dNBjLEnvCfg):
    """W3S3d-NBjM (1011, 승윤님 "실패하면 상한 23 으로도"): NBjL 과 같은 구조, 범위만 완화 — hip_a 0.72–0.92 배(18–23 N·m), 나머지 0.9–1.0 배.
    NBjL(13.75–20 N·m)은 학습 보상이 50 → 33–37 에서 회복하지 못했다(반복 900 시점). 실물 상한 23 N·m 에 맞춘 걸음을 노린다. 기준 §18(문턱만 23)."""
    HIP_A_SCALE = (0.72, 0.92)
    OTHER_SCALE = (0.9, 1.0)
