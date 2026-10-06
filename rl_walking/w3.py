"""W3 (1006) — 명령 조건 보행 세대 핵심 모듈. 1차 목표 = Stage 0 제자리 좌우 교대 스텝 엔진(보조 0 자립).

구성(reports/gen_w3/W3_DESIGN.md):
  - gait_enable(env._ge, bool) + 관측 게이트(env._ge_obs, 0.2 s 램프): 0 = STAND(클록 관측 (0,0)), 1 = STEP(클록 회전).
    0→1 은 위상 0(왼발 스윙)부터, 1→0 은 반주기 경계에서만(스윙 중 끄지 않음 — 래치).
  - 위상 생성기: env._gu(주기 단위), env._gf(Hz). 기존 _gait_phase/gait_clock/gait_u 가 그대로 읽는다.
  - 명령 분포(Stage 0): STAND 30 %·STEP0(명령 0) 70 %. 평가는 env var W3_GAIT_ENABLE·W3_CMD 로 고정.
  - 참조 잔차 훅(W3-C) + 보조 scaffold(W3-A): 행동 지연 버퍼 앞(pre_delay_hook). 기여는 env._w3_ref_last / _w3_assist_last 에 기록.
  - 스텝 추적기 + STEP 전용 보상(주기 접촉·스윙 높이·유효 스텝·착지 목표·모방). STAND 보상은 기존 항을 ge 0 게이트로.
모든 보상은 gait_enable 로 분리: STEP 에서 가만히 서면 총점 ≤ 0 이 되도록(단 작은 생존 항 + 큰 종료 벌점으로 조기 낙상 해킹 방지).
"""
from __future__ import annotations

import math
import os

import torch

from . import mdp_hang as H


# ---------------------------------------------------------------- 상태 ----------------------------------------------------------------
def _st(env):
    s = getattr(env, "_w3", None)
    if s is None or s["prev_c"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        z = lambda *sh: torch.zeros(*(sh or (N,)), device=dev)
        s = {"prev_c": torch.ones(N, 2, device=dev), "air_t": z(N, 2), "lift_z": z(N, 2), "apex": z(N, 2), "lift_xy": z(N, 2, 2),
             "lift_half": torch.full((N, 2), -1.0, device=dev), "valid_swing": z(N, 2), "load_t": z(N, 2), "load_win": z(N, 2), "pending": z(N, 2),
             "ev_takeoff": z(N, 2), "ev_td": z(N, 2), "ev_valid": z(N, 2), "ev_wrong": z(N, 2), "td_err": z(N, 2), "half": torch.full((N,), -1.0, device=dev),
             "valid_in_half": z(), "ev_miss": z(), "last_valid_foot": torch.full((N,), -1.0, device=dev), "ev_alt": z(),
             "n_valid": z(N, 2), "n_sched": z(), "n_alt": z()}
        env._w3 = s
    return s


def _ge(env):
    g = getattr(env, "_ge", None)
    return g if g is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# ---------------------------------------------------------------- 명령 + gait_enable ----------------------------------------------------------------
def install_w3_cmd(env, env_ids, p_step: float = 0.7, stage: int = 0):
    """mode="startup": 명령 리샘플 패치 — Stage 0 = STAND(ge 0, 명령 0) / STEP0(ge 1, 명령 0). 서기 env(is_standing_env)도 ge 로 덮는다.
    평가 고정: W3_GAIT_ENABLE=0|1, W3_CMD='vx,vy,wz'."""
    N, dev = env.num_envs, env.device
    env._ge = torch.zeros(N, dtype=torch.bool, device=dev)
    env._ge_cmd = torch.zeros(N, dtype=torch.bool, device=dev)      # 명령이 원하는 ge(래치 전)
    env._ge_obs = torch.zeros(N, device=dev)
    env._w3_stage = int(stage)
    term = env.command_manager.get_term("base_velocity")
    orig = term._resample_command

    def _resample(ids):
        orig(ids)
        ids = torch.as_tensor(ids, device=dev, dtype=torch.long) if not isinstance(ids, slice) else torch.arange(N, device=dev)
        n = ids.numel()
        if n == 0:
            return
        want = torch.rand(n, device=dev) < p_step
        fge = os.environ.get("W3_GAIT_ENABLE")
        if fge is not None:
            want = torch.full((n,), fge == "1", dtype=torch.bool, device=dev)
        c = term.vel_command_b
        c[ids] = 0.0
        fc = os.environ.get("W3_CMD")
        if fc:
            v = torch.tensor([float(x) for x in fc.split(",")], device=dev)
            c[ids] = v
        env._ge_cmd[ids] = want
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[ids] = False                             # 명령 0 고정은 위에서; 서기 여부는 ge 가 정함

    term._resample_command = _resample


def w3_gait_update(env, env_ids, f_step: float = 1.1, ramp_s: float = 0.2):
    """mode="interval"(0 s): 위상 누적 + gait_enable 래치 + 관측 램프. 에피소드 시작 = ge 즉시 반영·위상 0."""
    N, dev = env.num_envs, env.device
    if getattr(env, "_gu", None) is None or env._gu.shape[0] != N:
        env._gu = torch.zeros(N, device=dev); env._gf = torch.zeros(N, device=dev)
    want = getattr(env, "_ge_cmd", torch.zeros(N, dtype=torch.bool, device=dev))
    ge = _ge(env).clone()
    fresh = env.episode_length_buf <= 1
    on = want & ~ge                                                       # 0→1: 위상 0 부터
    u_next = env._gu + f_step * env.step_dt
    cross = (env._gu < 0.5) & (u_next >= 0.5) | (u_next >= 1.0)           # 반주기 경계 통과
    off = ~want & ge & cross                                              # 1→0: 반주기 경계에서만
    ge = torch.where(fresh, want, torch.where(on, torch.ones_like(ge), torch.where(off, torch.zeros_like(ge), ge)))
    reset_u = fresh | on | ~ge
    env._gu = torch.where(reset_u, torch.zeros_like(env._gu), torch.remainder(u_next, 1.0))
    env._gf = torch.where(ge, torch.full_like(env._gf, f_step), torch.zeros_like(env._gf))
    env._ge = ge
    k = env.step_dt / max(ramp_s, 1e-6)
    env._ge_obs = torch.where(fresh, ge.float(), (env._ge_obs + torch.where(ge, k, -k)).clamp(0.0, 1.0))


# ---------------------------------------------------------------- 참조 잔차 + 보조 ----------------------------------------------------------------
def install_w3_ref(env, env_ids, ref_path: str = "", ref_scale: float = 1.0, flex_gain: float = 1.0, lead_ticks: int = 6,
                   shift_amp: float = 0.0, shift_lead: float = 0.10, shift_joints: dict | None = None, assist_alpha0: float = 0.0):
    """mode="startup": pre_delay_hook 에 W3 참조 잔차를 건다(gait_enable 1 env 만).
    q_ref = ref_scale·(flex_gain·굽힘 펄스(A0h ref, 스윙발, lead 앞당김) + 관상면 무게 이동) + α_s·굽힘 펄스(보조 scaffold).
    무게 이동: y(u) = −shift_amp·sin(2π(u + shift_lead)) (u<0.5 왼 스윙 → 골반을 오른(지지)발 쪽으로), 관절 잔차 = Σ k_j·y(u)
    (shift_joints = {관절명: k} — 부호·크기는 기하 계산으로 정함; 비면 0).
    평가 진단 env var: W3_REF_SCALE(참조 배율 덮어쓰기), STEP_ASSIST_ALPHA(보조 α 덮어쓰기)."""
    R = H._sbi_ref(env, ref_path)
    term = env.action_manager.get_term("joint_pos")
    asset = env.scene[H._base_cfg().name]
    jn = list(asset.joint_names)
    J = torch.zeros(len(jn), device=env.device)
    for name, k in (shift_joints or {}).items():
        J[jn.index(name)] = float(k)
    env._w3_alpha = float(assist_alpha0)
    prev = term.pre_delay_hook

    def hook(target):
        if prev is not None:
            target = prev(target)
        ge = _ge(env).float().unsqueeze(1)
        rs = float(os.environ.get("W3_REF_SCALE", ref_scale * float(getattr(env, "_w3_ref_alpha", 1.0))))   # _w3_ref_alpha: 참조 내재화 커리큘럼
        a_s = float(os.environ.get("STEP_ASSIST_ALPHA", getattr(env, "_w3_alpha", 0.0)))
        u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(env.num_envs, device=env.device)
        ul = torch.remainder(u + lead_ticks * env.step_dt * getattr(env, "_gf", torch.zeros_like(u)), 1.0)
        half = (ul >= 0.5).long(); v = torch.remainder(2.0 * ul, 1.0); k = (v * 23.0).long().clamp(0, 22)
        fg = float(os.environ.get("W3_FLEX_GAIN", flex_gain)); sa = float(os.environ.get("W3_SHIFT_AMP", shift_amp))
        flex = (R["q"][half, k] - R["q"][half, 0])
        y = -sa * torch.sin(2.0 * math.pi * (u + shift_lead))
        shift = y.unsqueeze(1) * J.unsqueeze(0)
        ref = rs * (fg * flex + shift) * ge
        assist = a_s * flex * ge
        env._w3_ref_last = ref; env._w3_assist_last = assist
        return target + ref + assist

    term.pre_delay_hook = hook


def w3_ref_curriculum(env, env_ids, alphas: tuple = (1.0, 0.75, 0.5, 0.25, 0.0), min_full: float = 0.6, min_iters: int = 60,
                      min_sched: float = 20000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """1006 W3-CI: 참조 잔차 배율 감쇠(내재화) — A0m 규칙(절대 문턱: 블록 유효 스텝률 ≥ min_full 이 회복돼야 다음 단계, 체류 ≥ min_iters).
    마지막(0)에서 계속 학습. 상대 규칙(톱니) 금지."""
    s = _st(env); g = getattr(env, "_w3_rcur", None)
    if g is None:
        g = {"stage": int(start_stage), "since": 0, "base_s": 0.0, "base_v": 0.0, "last": float("nan")}; env._w3_rcur = g
        env._w3_ref_alpha = alphas[g["stage"]]
    n_s = float(s["n_sched"].sum()); n_v = float(s["n_valid"].sum())
    it = int(env.common_step_counter) // steps_per_iter
    if n_s - g["base_s"] >= min_sched:
        r = (n_v - g["base_v"]) / (n_s - g["base_s"]); g["last"] = r
        ok = r >= min_full and it - g["since"] >= min_iters and g["stage"] < len(alphas) - 1
        print(f"[W3 참조 블록] 반복 {it} 참조={alphas[g['stage']]}: 유효 스텝률 {r:.3f}" + (" → 감쇠" if ok else ""), flush=True)
        if ok:
            g["stage"] += 1; g["since"] = it; env._w3_ref_alpha = alphas[g["stage"]]
        g["base_s"], g["base_v"] = n_s, n_v
    return {"ref_alpha": float(env._w3_ref_alpha), "block_valid": g["last"]}


def w3_yaw_drift(env) -> torch.Tensor:
    """STEP 명령 요 0 의 실제 요 각속도 |ω_z| — 벌점 크기(양수). 1006 W3-C 400: 30 s 에 요 70° 표류."""
    asset = env.scene[H._base_cfg().name]
    c = env.command_manager.get_command("base_velocity")
    return (asset.data.root_ang_vel_w[:, 2] - c[:, 2]).abs() * _ge(env).float()


def w3_assist_curriculum(env, env_ids, alphas: tuple = (1.0, 0.75, 0.5, 0.25, 0.0), min_full: float = 0.25, min_iters: int = 50,
                         min_sched: float = 20000.0, steps_per_iter: int = 24):
    """보조 scaffold 감쇠 — A0m 규칙(절대 문턱): 블록 유효 스텝률(유효/예정 반주기) ≥ min_full 이고 체류 ≥ min_iters 면 다음 α.
    상대 규칙(톱니) 금지(W2omP3 교훈). 마지막 α(0) 에서 계속 학습."""
    s = _st(env); g = getattr(env, "_w3_cur", None)
    if g is None:
        g = {"stage": 0, "since": 0, "base_s": 0.0, "base_v": 0.0, "last": float("nan")}; env._w3_cur = g
        env._w3_alpha = alphas[0]
    n_s = float(s["n_sched"].sum()); n_v = float(s["n_valid"].sum())
    it = int(env.common_step_counter) // steps_per_iter
    if n_s - g["base_s"] >= min_sched:
        r = (n_v - g["base_v"]) / (n_s - g["base_s"]); g["last"] = r
        ok = r >= min_full and it - g["since"] >= min_iters and g["stage"] < len(alphas) - 1
        print(f"[W3 보조 블록] 반복 {it} α={alphas[g['stage']]}: 유효 스텝률 {r:.3f}" + (" → 감쇠" if ok else ""), flush=True)
        if ok:
            g["stage"] += 1; g["since"] = it; env._w3_alpha = alphas[g["stage"]]
        g["base_s"], g["base_v"] = n_s, n_v
    return {"alpha": float(env._w3_alpha), "block_valid": g["last"]}


# ---------------------------------------------------------------- 스텝 추적기 ----------------------------------------------------------------
def w3_step_tracker(env, env_ids, threshold: float = 5.0, min_air_s: float = 0.12, min_apex: float = 0.03, load_frac: float = 0.5,
                    load_s: float = 0.10, load_win_s: float = 0.30, nominal_half_width: float = 0.10, sig_place: float = 0.04):
    """mode="interval"(0 s): 발별 유효 스텝 = 이륙 → 의미 있는 스윙(공중 ≥ min_air_s·정점 ≥ min_apex) → 착지(예정 스윙발, 그 반주기 또는 직후 1/4 주기 안)
    → 하중 수용(착지 후 load_win_s 안 Fz > load_frac·mg 누적 load_s). 예정 아닌 발의 유효 스텝 = ev_wrong(두드리기·한발 피벗 감시).
    착지 오차 td_err = 착지 발 위치(골반 기준 요 프레임) − 목표 p*(명령 0: (0, ±nominal_half_width)) 노름."""
    s = _st(env)
    asset = env.scene[H._base_cfg().name]
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    F = sensor.data.net_forces_w[:, fid, :]; c = (F.norm(dim=-1) > threshold).float()
    mg = float(H.G) * asset.data.default_mass.sum(1).to(F.device); fz = F[..., 2] / mg.unsqueeze(1)
    bid = H._g21_bids(env); fp = asset.data.body_pos_w[:, bid, :]; fxy = fp[..., :2]; fzp = fp[..., 2]
    dt = env.step_dt; N = env.num_envs
    ge = _ge(env).float()
    fresh = env.episode_length_buf <= 1
    if bool(fresh.any()):
        for k_, v_ in s.items():
            if k_ in ("n_valid", "n_sched", "n_alt"):
                continue
            v_[fresh] = 0.0
        s["prev_c"][fresh] = 1.0; s["half"][fresh] = -1.0; s["lift_half"][fresh] = -1.0; s["last_valid_foot"][fresh] = -1.0
    u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(N, device=env.device)
    half = (u >= 0.5).float()                                             # 0 = 왼발 스윙 반주기, 1 = 오른발
    new_half = (half != s["half"]) & (ge > 0.5)
    # 놓친 스텝: 끝난 반주기에 예정 발 유효 스텝이 없었음
    s["ev_miss"] = (new_half & (s["half"] >= 0) & (s["valid_in_half"] < 0.5)).float()
    s["n_sched"] += (new_half & (s["half"] >= 0)).float()
    s["valid_in_half"] = torch.where(new_half, torch.zeros_like(s["valid_in_half"]), s["valid_in_half"])
    s["half"] = torch.where(ge > 0.5, half, torch.full_like(half, -1.0))
    lifted = (s["prev_c"] > 0.5) & (c < 0.5); landed = (s["prev_c"] < 0.5) & (c > 0.5); air = c < 0.5
    s["ev_takeoff"] = lifted.float()
    s["lift_z"] = torch.where(lifted, fzp, s["lift_z"]); s["lift_xy"] = torch.where(lifted.unsqueeze(-1), fxy, s["lift_xy"])
    s["lift_half"] = torch.where(lifted, half.unsqueeze(1).expand(-1, 2), s["lift_half"])
    s["apex"] = torch.where(lifted, torch.zeros_like(s["apex"]), torch.maximum(s["apex"], (fzp - s["lift_z"]) * air.float()))
    s["air_t"] = torch.where(air, s["air_t"] + dt, torch.zeros_like(s["air_t"]))
    s["valid_swing"] = torch.where(lifted, torch.zeros_like(s["valid_swing"]),
                                   torch.where(air & (s["air_t"] >= min_air_s) & (s["apex"] >= min_apex), torch.ones_like(s["valid_swing"]), s["valid_swing"]))
    # 착지 위치 오차(골반 요 프레임)
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rel = fxy - asset.data.root_pos_w[:, None, :2]
    rx = rel[..., 0] * cy.unsqueeze(1) + rel[..., 1] * sy.unsqueeze(1); ry = -rel[..., 0] * sy.unsqueeze(1) + rel[..., 1] * cy.unsqueeze(1)
    tgt_y = torch.tensor([nominal_half_width, -nominal_half_width], device=env.device).unsqueeze(0)
    err = torch.sqrt(rx ** 2 + (ry - tgt_y) ** 2)
    td_ok = landed & (s["valid_swing"] > 0.5)
    s["ev_td"] = td_ok.float()
    s["td_err"] = torch.where(td_ok, err, torch.zeros_like(err))
    s["pending"] = torch.where(td_ok, torch.ones_like(s["pending"]), s["pending"])
    s["load_win"] = torch.where(td_ok, torch.zeros_like(s["load_win"]), s["load_win"] + dt * s["pending"])
    s["load_t"] = torch.where(td_ok, torch.zeros_like(s["load_t"]), s["load_t"] + dt * ((fz > load_frac) & (s["pending"] > 0.5)).float())
    done = (s["pending"] > 0.5) & (s["load_t"] >= load_s)
    expire = (s["pending"] > 0.5) & (s["load_win"] > load_win_s)
    sched_foot = half.long()                                              # 예정 스윙발(0 왼·1 오)
    is_sched = torch.stack([sched_foot == 0, sched_foot == 1], dim=1) | (s["lift_half"] == torch.stack([torch.zeros_like(half), torch.ones_like(half)], 1))
    s["ev_valid"] = (done & is_sched).float() * ge.unsqueeze(1)
    s["ev_wrong"] = (done & ~is_sched).float() * ge.unsqueeze(1)
    s["pending"] = torch.where(done | expire, torch.zeros_like(s["pending"]), s["pending"])
    s["valid_swing"] = torch.where(landed, torch.zeros_like(s["valid_swing"]), s["valid_swing"])
    anyv = s["ev_valid"].sum(1) > 0
    foot = torch.argmax(s["ev_valid"], dim=1).float()
    s["ev_alt"] = (anyv & (s["last_valid_foot"] >= 0) & (foot != s["last_valid_foot"])).float()
    s["last_valid_foot"] = torch.where(anyv, foot, s["last_valid_foot"])
    s["valid_in_half"] = torch.maximum(s["valid_in_half"], anyv.float())
    s["n_valid"] += s["ev_valid"]; s["n_alt"] += s["ev_alt"]
    s["prev_c"] = c


# ---------------------------------------------------------------- 보상(모두 gait_enable 분리) ----------------------------------------------------------------
def w3_value(env, name: str) -> torch.Tensor:
    s = _st(env); v = s[name]
    return v.sum(1) if v.dim() == 2 else v


def w3_place_reward(env, sig: float = 0.04) -> torch.Tensor:
    """유효 착지 사건마다 exp(−td_err²/σ²) (명령 0 = 제자리 목표)."""
    s = _st(env)
    return (torch.exp(-(s["td_err"] / sig) ** 2) * s["ev_td"]).sum(1) * _ge(env).float()


def w3_periodic_contact(env, threshold: float = 5.0, win: tuple = (0.12, 0.88), w_force: float = 1.0, w_vel: float = 1.0) -> torch.Tensor:
    """Siekmann형(연속값, 양수 = 벌점 크기): 예정 스윙발의 스윙 창 안 Fz/mg + 예정 지지발 수평 속도 |v|. STEP(ge 1)만.
    STEP 에서 가만히 서면 스윙 창마다 스윙발 하중 ≈ 0.5 mg 로 계속 벌점."""
    asset = env.scene[H._base_cfg().name]
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    F = sensor.data.net_forces_w[:, fid, :]
    mg = float(H.G) * asset.data.default_mass.sum(1).to(F.device)
    fz = (F[..., 2] / mg.unsqueeze(1)).clamp(min=0.0)
    u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(env.num_envs, device=env.device)
    half = (u >= 0.5).long(); v = torch.remainder(2.0 * u, 1.0)
    inw = ((v >= win[0]) & (v < win[1])).float()
    ar = torch.arange(env.num_envs, device=env.device)
    bid = H._g21_bids(env)
    fv = asset.data.body_lin_vel_w[:, bid, :2].norm(dim=-1)
    pen = w_force * fz[ar, half] * inw + w_vel * fv[ar, 1 - half]
    return pen * _ge(env).float()


def w3_swing_height(env, h: float = 0.05, sig: float = 0.02, win: tuple = (0.12, 0.88)) -> torch.Tensor:
    """예정 스윙발 높이(지지발 기준) 추종 exp(−(Δz − h·sin(π·v'))²/σ²), 스윙 창 안만. STEP 만."""
    asset = env.scene[H._base_cfg().name]
    bid = H._g21_bids(env); fz = asset.data.body_pos_w[:, bid, 2]
    u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(env.num_envs, device=env.device)
    half = (u >= 0.5).long(); v = torch.remainder(2.0 * u, 1.0)
    vp = ((v - win[0]) / (win[1] - win[0])).clamp(0.0, 1.0)
    inw = ((v >= win[0]) & (v < win[1])).float()
    ar = torch.arange(env.num_envs, device=env.device)
    dz = fz[ar, half] - fz[ar, 1 - half]
    zr = h * torch.sin(math.pi * vp)
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    air = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1)[ar, half] <= 5.0).float()   # 1006 단위시험: 서기도 창 가장자리에서 +0.36/s 받음 → 공중일 때만
    return torch.exp(-((dz - zr) / sig) ** 2) * inw * air * _ge(env).float()


def w3_imitation(env, ref_path: str = "", sig_q: float = 0.10, win: tuple = (0.25, 0.75)) -> torch.Tensor:
    """모방(밀집, 정책 자신의 관절을 채점) — **스윙발 굽힘 관절(hip_f·knee·ankle_f)만, 스윙 창 가운데(win)만**:
    exp(−Σ(q − q_ref)² / (n σ²)). 창 가운데는 참조 굽힘이 커서 가만히 서면 점수가 낮다(감사: 전 관절·전 구간 모방은 서기에 66 % 공짜). STEP 만."""
    R = H._sbi_ref(env, ref_path)
    asset = env.scene[H._base_cfg().name]
    jn = list(asset.joint_names)
    if getattr(env, "_w3_flexidx", None) is None:
        env._w3_flexidx = torch.tensor([[jn.index(f"{side}_{j}_joint") for j in ("hip_f", "knee", "ankle_f")] for side in ("left", "right")],
                                       device=env.device)
    u = env._gu if getattr(env, "_gu", None) is not None else torch.zeros(env.num_envs, device=env.device)
    half = (u >= 0.5).long(); v = torch.remainder(2.0 * u, 1.0); k = (v * 23.0).long().clamp(0, 22)
    q_ref = R["q"][half, k]
    idx = env._w3_flexidx[half]                                          # (N,3) 스윙발 굽힘 관절
    dq = torch.gather(asset.data.joint_pos, 1, idx) - torch.gather(q_ref, 1, idx)
    inw = ((v >= win[0]) & (v < win[1])).float()
    r = torch.exp(-(dq * dq).sum(1) / (3.0 * sig_q ** 2))
    return r * inw * _ge(env).float()


def w3_alive_step(env) -> torch.Tensor:
    """STEP 작은 생존 항(조기 낙상 해킹 방지 — 검토자). STEP 서기 총점은 주기 접촉 벌점으로 음수 유지."""
    return _ge(env).float()


def w3_drift(env) -> torch.Tensor:
    """STEP 명령 0 의 수평 이동(몸통 xy 속도 노름) — 벌점 크기(양수)."""
    asset = env.scene[H._base_cfg().name]
    return asset.data.root_lin_vel_w[:, :2].norm(dim=-1) * _ge(env).float()


def w3_stand_gated(env, func, params: dict | None = None) -> torch.Tensor:
    """기존(서기) 보상을 STAND(ge 0)에만."""
    val = func(env, **H._resolve_nested(env, params))
    return val * (~_ge(env)).to(val.dtype)


def w3_step_gated(env, func, params: dict | None = None) -> torch.Tensor:
    val = func(env, **H._resolve_nested(env, params))
    return val * _ge(env).to(val.dtype)
