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
  3) 관측 계약: 실물 stage8 은 기구 고정 관절의 피드백을 합성한다(실측 = stand_zero_deg, 속도 0 — 커밋 e1e92131).
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


@configclass
class Biped12HangStandW3S3dNBjDEnvCfg(Biped12HangStandW3S3dNBjMEnvCfg):
    """W3S3d-NBjD (1011): NBjM + **예정 스윙 창을 (0.12, 0.88) → (0.25, 0.75)** — 예정 양발 지지 24 → 50 %(반주기 0.455 s 중 0.11 → 0.23 s).
    근거(NBj-s2 시뮬, 걷기 켠 뒤 틱별 중앙): 지지쪽 힙 외전 요구는 스윙발 이륙 **직전** 0.1 s 에 30–37 N·m 로 솟고(25 초과의 83 % 가 이륙 전), 한발 지지 중에는 2–5 N·m.
    즉 과요구는 한발로 버티는 힘이 아니라 0.11 s 안에 무게를 넘기는 밀기다 — 넘길 시간을 늘린다. 바뀌는 것은 w3_periodic·w3_height 의 win 뿐(W3S3tD 와 같은 방식),
    클록·관측·유효 스텝 판정·모방 창 불변 → 배포 러너 변경 없음. 기준 §19."""
    SWING_WIN = (0.25, 0.75)

    def __post_init__(self):
        super().__post_init__()
        from rl_walking import w3 as _w3
        rw = self.rewards
        assert rw.w3_periodic.func is _w3.w3_periodic_contact and rw.w3_height.func is _w3.w3_swing_height
        _win = (float(self.SWING_WIN[0]), float(self.SWING_WIN[1]))
        rw.w3_periodic.params = dict(rw.w3_periodic.params or {}, win=_win)
        rw.w3_height.params = dict(rw.w3_height.params or {}, win=_win)


@configclass
class Biped12HangStandW3S3dNBjEEnvCfg(Biped12HangStandW3S3dNBjDEnvCfg):
    """W3S3d-NBjE (1011 05:20): NBjD 와 전부 같고 **힙 외전 한계 범위만 0.72–0.92 → 0.64–0.84 배(16–21 N·m)**. 시작 = NBjD-s1-17798 이어 학습.
    근거: NBjD 는 걷기 전환 솟음 중앙 20–21·p95 25 N·m, env 별 최대 p95 29–30 N·m — 과도 기준(≤ 23) 미달. NBjL(13.75–20, 좁은 양발 지지)은 폭발했으므로 넓은 양발 지지 위에서 한 단계만 낮춘다. 기준 §20."""
    HIP_A_SCALE = (0.64, 0.84)


def nb_hip_a_demand_abs(env, lim_nm: float = 20.0) -> torch.Tensor:
    """힙 외전 PD 요구 토크(자르기 전)의 **절대값**이 lim_nm 을 넘는 초과 비율 e 의 (e + e²) 합 — 벌점 크기(양수).
    nb_over_demand 는 env 별 무작위 한계 대비 제곱이라 드문 솟음(26 N·m 수준)엔 값이 작다. 이 항은 실물 상한(20 N·m) 기준 고정 문턱 + 선형 항으로 작은 초과에도 기울기를 준다."""
    robot = env.scene["robot"]
    c = getattr(env, "_nb_ha_cols", None)
    if c is None:
        c = env._nb_ha_cols = torch.tensor([j for j, n in enumerate(robot.joint_names) if "hip_a" in n], device=env.device)
    e = torch.clamp(robot.data.computed_torque[:, c].abs() / float(lim_nm) - 1.0, min=0.0)
    return (e + e.pow(2)).sum(dim=1)


@configclass
class Biped12HangStandW3S3dNBjHEnvCfg(Biped12HangStandW3S3dNBjEEnvCfg):
    """W3S3d-NBjH (1011 14:50, 승윤님 "순간 요구를 26 까지 안 나오게 — 20 으로 자르는 게 아니라"): NBjE 와 전부 같고 **보상 항 하나만 추가** —
    걷는 중 힙 외전 요구가 20 N·m 를 넘는 초과분 벌점(nb_hip_a_demand_abs, 가중 −20). 시작 = NBjE-s1-18997 이어 학습. 기준 NEWBODY_WALK.md §22."""
    W_HAD = -20.0           # 비주석 클래스 상수(설정 필드 아님)

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_walking import w3 as _w3
        self.rewards.w3_hip_a_demand = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_HAD), params={"func": nb_hip_a_demand_abs, "params": {}})


@configclass
class Biped12HangStandW3S3dNBjIEnvCfg(Biped12HangStandW3S3dNBjHEnvCfg):
    """W3S3d-NBjI (1011 16:35, 승윤님 "가능하면 20 으로 맞춰"): NBjH 와 같고 벌점 문턱 20 → 16 N·m, 가중 −20 → −40.
    순간 최대(env 별 최대 p95)를 20 이하로 두려면 문턱이 목표보다 낮아야 한다(NBjH 는 문턱 20 에서 p95 24). 시작 = NBjH-s2-20196. 기준 NEWBODY_WALK.md §23."""
    W_HAD = -40.0
    HAD_LIM = 16.0

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_hip_a_demand.params["params"] = {"lim_nm": float(self.HAD_LIM)}


@configclass
class Biped12HangStandW3S3dNBjSEnvCfg(Biped12HangStandW3S3dNBjEEnvCfg):
    """W3S3d-NBjS (1011 18:35, 승윤님 "발은 1 cm 만 띄워도 되고 진짜 종종걸음 모션만"): NBjE 와 전부 같고 **발 높이만** —
    스윙 높이 목표 5 → 1.2 cm(σ 2 → 0.8 cm), 유효 스텝 정점 기준 3 → 0.5 cm. 시작 = NBjE-s1-18997. 시뮬 영상용(실물 배포 후보 아님)."""
    SWING_H = 0.012
    SWING_SIG = 0.008
    MIN_APEX = 0.005

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        rw.w3_height.params = dict(rw.w3_height.params or {}, h=float(self.SWING_H), sig=float(self.SWING_SIG))
        self.events.w3_track.params = dict(self.events.w3_track.params or {}, min_apex=float(self.MIN_APEX))


# ---------------------------------------------------------------- 1011 밤: WADDLE-W1 초저속 뒤뚱 보행(승윤님 지시) ----------------------------------------------------------------
def nb_set_vbox_period(env, env_ids, period_s: float = 1.0 / 1.1):
    """mode="startup": 한 주기 이동 평균 속도(_v_box) 창 길이를 걸음 시계 주기에 맞춘다(시계를 늦추면 종전 0.91 s 창은 한 주기를 못 덮어 흔들림이 남는다)."""
    env._w3_vbox_period = float(period_s)


@configclass
class Biped12HangStandW3S3dNBjWEnvCfg(Biped12HangStandW3S3dNBjEEnvCfg):
    """W3S3d-NBjW (1011 WADDLE-W1a): NBjE 와 같고 **걸음 시계·발 높이·명령 범위만** 바꾼다. 시작 = NBjE-s1-18997 이어 학습.
    - 걸음 시계 1.1 → F_STEP Hz(0.4 = 한 발 스텝 1.25 s, 예정 스윙 0.625 s·양발 지지 0.625 s). 속도 평균 창도 한 주기로.
    - 스윙 높이 목표 5 → 2 cm(σ 2 → 1 cm), 유효 스텝 정점 기준 3 → 1 cm.
    - 명령: 제자리 0.2·앞뒤 0.8(뒤 범위 0 → 뒤 표본은 제자리와 같다 = 제자리 60 %·앞 40 %), 앞 속도 VX(한 걸음 2–4 cm 어림).
    힙 외전 한계 무작위화(16–21 N·m)·과요구 벌점·미끄러짐 벌점·보상 가중·관측/행동 차원은 NBjE 그대로.
    배포 계약 변경: 러너 f_step 을 같은 값으로. 기준 reports/real_1009/WADDLE_W1.md."""
    F_STEP = 0.4
    VX = (0.02, 0.035)
    SWING_H = 0.02
    SWING_SIG = 0.01
    MIN_APEX = 0.01
    MIX3 = (0.2, 0.8, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards; ev = self.events
        ev.w3_gait.params = dict(ev.w3_gait.params or {}, f_step=float(self.F_STEP))
        ev.w3_cmd.params = dict(ev.w3_cmd.params or {}, mix3=tuple(self.MIX3), vx_range=tuple(self.VX), vx_back_range=(0.0, 0.0))
        rw.w3_height.params = dict(rw.w3_height.params or {}, h=float(self.SWING_H), sig=float(self.SWING_SIG))
        ev.w3_track.params = dict(ev.w3_track.params or {}, min_apex=float(self.MIN_APEX))
        ev.nb_vbox = EventTerm(func=nb_set_vbox_period, mode="startup", params={"period_s": 1.0 / float(self.F_STEP)})


@configclass
class Biped12HangStandW3S3dNBjXEnvCfg(Biped12HangStandW3S3dNBjWEnvCfg):
    """W3S3d-NBjX (1011 WADDLE-W1b): NBjW 와 같고 걸음 시계만 0.6 Hz(한 발 스텝 0.83 s), 앞 속도 0.025–0.05(한 걸음 2–4 cm)."""
    F_STEP = 0.6
    VX = (0.025, 0.05)


@configclass
class Biped12HangStandW3S3dNBjYEnvCfg(Biped12HangStandW3S3dNBjWEnvCfg):
    """W3S3d-NBjY (1011 WADDLE-W1c, 21:25 추가): NBjW 와 같고 걸음 시계만 0.8 Hz(한 발 스텝 0.625 s), 앞 속도 0.03–0.06.
    근거: W1a(0.4 Hz)는 유효 스텝이 0 으로 무너졌고 W1b(0.6 Hz)는 기준선의 1/3 수준에서 정체 — 동적 걸음에서 덜 벗어난 한 단계."""
    F_STEP = 0.8
    VX = (0.03, 0.06)


# ---------------------------------------------------------------- 1011 밤: 좁은 걸음(승윤님 "저렇게 종종걸음으로") ----------------------------------------------------------------
@configclass
class Biped12HangStandW3S3dNBjNEnvCfg(Biped12HangStandW3S3dNBjEEnvCfg):
    """W3S3d-NBjN (1011 22:40): NBjE 와 전부 같고 **착지 목표 발 간격만** 좁힌다 — 목표 반폭 0.17 → W_HALF_N(0.14 = 서기 간격 28 cm),
    폭 벌점 문턱(2w − 0.05)도 같은 w 로. 걸음 시계 1.1 Hz·발 높이·명령·한계 무작위화 불변 → 배포 러너 계약 변경 없음.
    근거: 지금 걸음은 착지 간격 34–36 cm(목표 0.17 이 1006 에 실측 넓은 걸음에 맞춰 올려진 값). 좌우로 미는 힘은 발 간격에 비례(선형 역진자 어림) —
    좁히면 이륙 직전 힙 외전 밀기가 줄 것이라는 예상(미검증). 시작 = NBjE-s1-18997. 기준 reports/real_1009/WADDLE_W1.md §5."""
    W_HALF_N = 0.14

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        ev.w3_track.params = dict(ev.w3_track.params or {}, nominal_half_width=float(self.W_HALF_N))
        ev.w3_lat.params = dict(ev.w3_lat.params or {}, nominal_half_width=float(self.W_HALF_N))


@configclass
class Biped12HangStandW3S3dNBjOEnvCfg(Biped12HangStandW3S3dNBjNEnvCfg):
    """W3S3d-NBjO (1011 22:40): NBjN 과 같고 목표 반폭 0.11(발 간격 22 cm)."""
    W_HALF_N = 0.11


def nb_width_excess(env, width_m: float = 0.24) -> torch.Tensor:
    """걷는 중 좌우 발 간격이 width_m 을 넘는 만큼[m](양수, 매 틱) — w3_lat_tracker 가 갱신한 현재 간격. STEP 게이트는 w3_step_gated 가 건다."""
    from rl_walking import w3 as _w3
    return (_w3._lat(env)["width"] - float(width_m)).clamp(min=0.0)


@configclass
class Biped12HangStandW3S3dNBjPEnvCfg(Biped12HangStandW3S3dNBjNEnvCfg):
    """W3S3d-NBjP (1011 22:47): NBjN(목표 반폭 0.12) + **간격 초과 벌점**(걷는 중 매 틱, 간격 − 24 cm 의 양수분 × −5).
    근거: N1·N2 학습 초반 300 반복 동안 착지 보상이 오르지 않음(착지 사건 보상만으로는 간격을 당기는 신호가 약함 — 1006 W3S1e 때와 같은 현상). 시작 = NBjE-s1-18997."""
    W_HALF_N = 0.12
    W_WEX = -5.0

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_walking import w3 as _w3
        self.rewards.w3_width_excess = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_WEX), params={"func": nb_width_excess, "params": {"width_m": 2.0 * float(self.W_HALF_N)}})


NB_REF_NARROW = os.path.join(os.path.dirname(_HERE), "reports", "real_1009", "newbody", "ref_narrow_a.npz")


def nb_imit_lateral(env, ref_path: str = NB_REF_NARROW, sig_q: float = 0.10) -> torch.Tensor:
    """좌우 관절(양쪽 hip_a·ankle_r) 모방 — 반주기 위상의 참조각과의 차: exp(−Σdq²/(4σ²)), 전 구간. STEP 게이트는 w3_step_gated 가 건다.
    σ 0.10 rad(w3_imit 와 같은 값): 0.04 로 시작했더니 지금 걸음과의 차(관절당 RMS 5.5°)에서 값이 0.003 으로 죽어 기울기가 없었다(22:55 측정) → 0.10 이면 약 0.4."""
    from rl_walking import mdp_hang as H
    R = H._sbi_ref(env, ref_path)
    asset = env.scene[H._base_cfg().name]
    if getattr(env, "_nb_latidx", None) is None:
        jn = list(asset.joint_names)
        env._nb_latidx = torch.tensor([jn.index(f"{sd}_{j}_joint") for j in ("hip_a", "ankle_r") for sd in ("left", "right")], device=env.device)
    u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(env.num_envs, device=env.device)
    half = (u >= 0.5).long(); v = torch.remainder(2.0 * u, 1.0); k = (v * 23.0).long().clamp(0, 22)
    dq = asset.data.joint_pos[:, env._nb_latidx] - R["q"][half, k][:, env._nb_latidx]
    return torch.exp(-(dq * dq).sum(1) / (4.0 * sig_q ** 2))


@configclass
class Biped12HangStandW3S3dNBjZEnvCfg(Biped12HangStandW3S3dNBjPEnvCfg):
    """W3S3d-NBjZ (1011 22:50, 승윤님 "레퍼런스 잡아주면 안돼?"·"보폭 줄고 발 빠르게"): NBjP + **좌우 관절 참조 모방**(역기구학으로 만든 좁은 종종걸음:
    발 간격 22.7 cm·골반 흔들림 1.5 cm·발 2 cm, `newbody/make_ref_narrow.py`) + **걸음 시계 1.1 → 1.4 Hz**(초당 2.8 걸음).
    한 번에 세 가지(간격 벌점·참조·시계)가 바뀌므로 무엇의 효과인지는 가르지 못한다 — 결과만 본다. 배포 계약 변경: 러너 f_step 1.4."""
    F_STEP_Z = 1.4
    W_IMLAT = 1.5

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as RewTerm
        from rl_walking import w3 as _w3
        ev = self.events
        ev.w3_gait.params = dict(ev.w3_gait.params or {}, f_step=float(self.F_STEP_Z))
        ev.nb_vbox = EventTerm(func=nb_set_vbox_period, mode="startup", params={"period_s": 1.0 / float(self.F_STEP_Z)})
        self.rewards.w3_imit_lat = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_IMLAT), params={"func": nb_imit_lateral, "params": {}})
