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
