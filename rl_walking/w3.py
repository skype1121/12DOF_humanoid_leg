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
             "ev_takeoff": z(N, 2), "ev_td": z(N, 2), "ev_valid": z(N, 2), "ev_wrong": z(N, 2), "td_err": z(N, 2), "td_dx": z(N, 2), "td_err_h": z(N, 2), "td_dx_h": z(N, 2), "half": torch.full((N,), -1.0, device=dev),
             "valid_in_half": z(), "ev_miss": z(), "last_valid_foot": torch.full((N,), -1.0, device=dev), "ev_alt": z(),
             "n_valid": z(N, 2), "n_sched": z(), "n_alt": z()}
        env._w3 = s
    return s


def _ge(env):
    g = getattr(env, "_ge", None)
    return g if g is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# ---------------------------------------------------------------- 명령 + gait_enable ----------------------------------------------------------------
def install_w3_cmd(env, env_ids, p_step: float = 0.7, stage: int = 0, mix: tuple = (0.43, 0.285, 0.285), vx_range: tuple = (0.03, 0.10),
                   vx_back_range: tuple | None = None, mix2: tuple = (0.3, 0.3, 0.4), vy_range: tuple = (0.03, 0.04),
                   mix3: tuple = (0.3, 0.3, 0.0, 0.4), wz_range: tuple = (0.1, 0.2), p_hloop: float = 0.0,
                   mix4: tuple = (0.2, 0.15, 0.15, 0.2, 0.3), mix_pat: tuple = (0.4, 0.2, 0.15, 0.25), mix_scale: float = 0.0):
    """mode="startup": 명령 리샘플 패치 — Stage 0 = STAND(ge 0, 명령 0) / STEP0(ge 1, 명령 0). 서기 env(is_standing_env)도 ge 로 덮는다.
    Stage 1 = STEP 중 제자리 / 앞 / 뒤(vx). Stage 2(1006 W3S2) = STEP 중 제자리 mix2[0] / 앞뒤 mix2[1](반반) / 옆 mix2[2](±vy, |vy| ∈ env._w3_vy_range —
    w3_vy_curriculum 이 상한을 넓힌다). Stage 0/1 경로는 종전과 비트 동일(난수 소비 순서 포함).
    Stage 3(1006 W3S3) = STEP 중 제자리 mix3[0] / 앞뒤 mix3[1] / 옆 mix3[2] / 제자리 회전 mix3[3](±wz, |wz| ∈ env._w3_wz_range — w3_wz_curriculum).
    Stage 3 은 리샘플 원래 wz 를 env._w3_wz0 에, 외부 헤딩 고리 표시(확률 p_hloop, STEP·비회전 명령만)를 env._w3_hl 에 남긴다(w3_heading_loop 가 읽음).
    **Stage 3 은 command term 의 heading_command 를 꺼야 한다**(켜 두면 _update_command 가 매 틱 wz 를 clip(·, ang_vel_z 범위) 로 덮음 — 1006 측정 사실).
    Stage 4(1006 W3S4, 혼합) = STEP 중 제자리 mix4[0] / 앞뒤 mix4[1] / 옆 mix4[2] / 회전 mix4[3](단일 축 — Stage 3 과 같은 범위·부호 규칙) /
    혼합 mix4[4]. 혼합 안 조합 = mix_pat(앞뒤+회전 · 앞뒤+옆 · 옆+회전 · 셋 다), 축마다 부호 독립 반반, 크기 = lo + s·U·(hi − lo)
    (단일 축 범위 env._w3_vx_range/_vx_back/_vy_range/_wz_range 를 따름, s = env._w3_mix_scale — w3_mix_curriculum 이 0 → 0.5 → 1).
    헤딩 고리 표시는 Stage 3 과 같다(STEP·wz 0 명령 — 혼합 앞뒤+옆 포함). reports/gen_w3/S4_DESIGN.md.
    평가 고정: W3_GAIT_ENABLE=0|1, W3_CMD='vx,vy,wz'."""
    N, dev = env.num_envs, env.device
    env._ge = torch.zeros(N, dtype=torch.bool, device=dev)
    env._ge_cmd = torch.zeros(N, dtype=torch.bool, device=dev)      # 명령이 원하는 ge(래치 전)
    env._ge_obs = torch.zeros(N, device=dev)
    env._w3_stage = int(stage); env._w3_mix = tuple(mix); env._w3_vx_range = tuple(vx_range)
    env._w3_vx_back = tuple(vx_back_range) if vx_back_range else tuple(vx_range)
    env._w3_mix2 = tuple(mix2); env._w3_vy_range = tuple(vy_range)
    if int(stage) >= 3:                                                   # Stage 3 상태(그 밖 stage 는 속성도 만들지 않음 — 비트 동일)
        env._w3_mix3 = tuple(mix3); env._w3_wz_range = tuple(wz_range); env._w3_p_hloop = float(p_hloop)
        env._w3_wz0 = torch.zeros(N, device=dev); env._w3_hl = torch.zeros(N, dtype=torch.bool, device=dev)
    if int(stage) >= 4:                                                   # Stage 4 상태(stage 0–3 은 만들지 않음 — 비트 동일)
        env._w3_mix4 = tuple(mix4); env._w3_mix_pat = tuple(mix_pat); env._w3_mix_scale = float(mix_scale)
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
        if env._w3_stage == 1:      # Stage 1: STEP env 중 앞 / 뒤 범주(vx), 나머지 제자리 — 비율은 env._w3_mix(기본 제자리 0.43·앞 0.285·뒤 0.285)
            mix = getattr(env, "_w3_mix", (0.43, 0.285, 0.285))
            r = torch.rand(n, device=dev)
            fwd = r >= mix[0]; back = r >= mix[0] + mix[1]
            mag = torch.empty(n, device=dev).uniform_(*getattr(env, "_w3_vx_range", (0.03, 0.10)))
            magb = torch.empty(n, device=dev).uniform_(*getattr(env, "_w3_vx_back", getattr(env, "_w3_vx_range", (0.03, 0.10))))
            vx = torch.where(back, -magb, torch.where(fwd, mag, torch.zeros_like(mag)))
            c[ids, 0] = torch.where(want, vx, torch.zeros_like(vx))
        elif env._w3_stage == 2:    # Stage 2(1006 W3S2): 제자리 / 앞뒤(반반, Stage 1 범위 — 리허설) / 옆(±vy, 부호 반반)
            m2 = getattr(env, "_w3_mix2", (0.3, 0.3, 0.4))
            r = torch.rand(n, device=dev); r2 = torch.rand(n, device=dev)
            fb = (r >= m2[0]) & (r < m2[0] + m2[1]); lat = r >= m2[0] + m2[1]
            mag = torch.empty(n, device=dev).uniform_(*env._w3_vx_range)
            magb = torch.empty(n, device=dev).uniform_(*env._w3_vx_back)
            magy = torch.empty(n, device=dev).uniform_(*env._w3_vy_range)
            neg = r2 < 0.5
            vx = torch.where(fb, torch.where(neg, -magb, mag), torch.zeros_like(mag))
            vy = torch.where(lat, torch.where(neg, -magy, magy), torch.zeros_like(magy))
            c[ids, 0] = torch.where(want, vx, torch.zeros_like(vx))
            c[ids, 1] = torch.where(want, vy, torch.zeros_like(vy))
        elif env._w3_stage >= 4:    # Stage 4(1006 W3S4): 단일 축 4 범주(Stage 3 과 같은 식) + 혼합(축별 독립 부호, 크기 커리큘럼)
            m4 = env._w3_mix4; mp = env._w3_mix_pat
            r = torch.rand(n, device=dev); r2 = torch.rand(n, device=dev); r3 = torch.rand(n, device=dev)
            rp = torch.rand(n, device=dev); sg = torch.rand(n, 3, device=dev) < 0.5; uu = torch.rand(n, 3, device=dev)
            e1 = m4[0]; e2 = e1 + m4[1]; e3 = e2 + m4[2]; e4 = e3 + m4[3]
            fb = (r >= e1) & (r < e2); lat = (r >= e2) & (r < e3); turn = (r >= e3) & (r < e4); mixd = r >= e4
            p1 = mp[0]; p2 = p1 + mp[1]; p3 = p2 + mp[2]
            # 조합: rp < p1 앞뒤+회전 · p1–p2 앞뒤+옆 · p2–p3 옆+회전 · ≥ p3 셋 다
            mx = mixd & ((rp < p2) | (rp >= p3))
            my = mixd & (rp >= p1)
            mw = mixd & ((rp < p1) | (rp >= p2))
            mag = torch.empty(n, device=dev).uniform_(*env._w3_vx_range)
            magb = torch.empty(n, device=dev).uniform_(*env._w3_vx_back)
            magy = torch.empty(n, device=dev).uniform_(*env._w3_vy_range)
            magw = torch.empty(n, device=dev).uniform_(*env._w3_wz_range)
            neg = r2 < 0.5
            s_ = float(env._w3_mix_scale)
            lohi = lambda rg, k: rg[0] + s_ * uu[:, k] * (rg[1] - rg[0])
            hx = torch.where(sg[:, 0], -lohi(env._w3_vx_back, 0), lohi(env._w3_vx_range, 0))
            hy = torch.where(sg[:, 1], -lohi(env._w3_vy_range, 1), lohi(env._w3_vy_range, 1))
            hw = torch.where(sg[:, 2], -lohi(env._w3_wz_range, 2), lohi(env._w3_wz_range, 2))
            zero = torch.zeros(n, device=dev)
            vx = torch.where(fb, torch.where(neg, -magb, mag), torch.where(mx, hx, zero))
            vy = torch.where(lat, torch.where(neg, -magy, magy), torch.where(my, hy, zero))
            wz = torch.where(turn, torch.where(neg, -magw, magw), torch.where(mw, hw, zero))
            c[ids, 0] = torch.where(want, vx, zero)
            c[ids, 1] = torch.where(want, vy, zero)
            c[ids, 2] = torch.where(want, wz, zero)
            env._w3_hl[ids] = want & (wz == 0) & (r3 < env._w3_p_hloop)      # 외부 헤딩 고리: STEP·wz 0 명령의 일부(Stage 3 과 같은 규약)
        elif env._w3_stage == 3:    # Stage 3(1006 W3S3): 제자리 / 앞뒤 / 옆 / 제자리 회전(±wz, 부호 반반) — 대각·이동+회전 없음
            m3 = env._w3_mix3
            r = torch.rand(n, device=dev); r2 = torch.rand(n, device=dev); r3 = torch.rand(n, device=dev)
            e1 = m3[0]; e2 = e1 + m3[1]; e3 = e2 + m3[2]
            fb = (r >= e1) & (r < e2); lat = (r >= e2) & (r < e3); turn = r >= e3
            mag = torch.empty(n, device=dev).uniform_(*env._w3_vx_range)
            magb = torch.empty(n, device=dev).uniform_(*env._w3_vx_back)
            magy = torch.empty(n, device=dev).uniform_(*env._w3_vy_range)
            magw = torch.empty(n, device=dev).uniform_(*env._w3_wz_range)
            neg = r2 < 0.5
            vx = torch.where(fb, torch.where(neg, -magb, mag), torch.zeros_like(mag))
            vy = torch.where(lat, torch.where(neg, -magy, magy), torch.zeros_like(magy))
            wz = torch.where(turn, torch.where(neg, -magw, magw), torch.zeros_like(magw))
            c[ids, 0] = torch.where(want, vx, torch.zeros_like(vx))
            c[ids, 1] = torch.where(want, vy, torch.zeros_like(vy))
            c[ids, 2] = torch.where(want, wz, torch.zeros_like(wz))
            env._w3_hl[ids] = want & ~turn & (r3 < env._w3_p_hloop)          # 외부 헤딩 고리: STEP·비회전 명령의 일부(배포: 운영자 wz 0 + 고리)
        fc = os.environ.get("W3_CMD")
        if fc:
            v = torch.tensor([float(x) for x in fc.split(",")], device=dev)
            c[ids] = v
        if env._w3_stage >= 3:
            env._w3_wz0[ids] = c[ids, 2]
        env._ge_cmd[ids] = want
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[ids] = False                             # 명령 0 고정은 위에서; 서기 여부는 ge 가 정함

    term._resample_command = _resample


def w3_gait_update(env, env_ids, f_step: float = 1.1, ramp_s: float = 0.2, lead_vy_cmin: float = 0.0, lead_wz_cmin: float = 0.0):
    """mode="interval"(0 s): 위상 누적 + gait_enable 래치 + 관측 램프. 에피소드 시작 = ge 즉시 반영·위상 0.
    lead_vy_cmin > 0(Stage 2): STEP 시작(fresh·0→1) 때 명령 vy < −lead_vy_cmin 이면 위상 0.5(오른발 리드 — 가는 쪽 발이 먼저 나감).
    vy > 0 은 종전대로 0(왼발 리드). 0(기본) = 종전과 비트 동일.
    lead_wz_cmin > 0(Stage 3): 옆 명령이 없고(|vy| < 0.02) wz < −lead_wz_cmin(오른쪽 회전)이면 위상 0.5(오른발 리드 — 도는 쪽 발이 먼저 바깥으로).
    0(기본) = 종전과 비트 동일."""
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
    u0 = torch.zeros_like(env._gu)
    if lead_vy_cmin > 0.0:
        vy = env.command_manager.get_command("base_velocity")[:, 1]
        u0 = torch.where((vy < -lead_vy_cmin) & ge & (fresh | on), torch.full_like(u0, 0.5), u0)
    if lead_wz_cmin > 0.0:
        cw = env.command_manager.get_command("base_velocity")
        u0 = torch.where((cw[:, 1].abs() < 0.02) & (cw[:, 2] < -lead_wz_cmin) & ge & (fresh | on), torch.full_like(u0, 0.5), u0)
    env._gu = torch.where(reset_u, u0, torch.remainder(u_next, 1.0))
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
                         min_sched: float = 20000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """보조 scaffold 감쇠 — A0m 규칙(절대 문턱): 블록 유효 스텝률(유효/예정 반주기) ≥ min_full 이고 체류 ≥ min_iters 면 다음 α.
    상대 규칙(톱니) 금지(W2omP3 교훈). 마지막 α(0) 에서 계속 학습."""
    s = _st(env); g = getattr(env, "_w3_cur", None)
    if g is None:
        g = {"stage": int(start_stage), "since": 0, "base_s": 0.0, "base_v": 0.0, "last": float("nan")}; env._w3_cur = g
        env._w3_alpha = alphas[g["stage"]]
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
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((N,), 1.1, device=env.device)
    tgt_x = (cmd[:, 0] / (4.0 * f_.clamp(min=0.5))).unsqueeze(1)          # Raibert형: 반 지지 시간 이동(명령 0 → 0 = Stage 0 과 같음)
    tgt_y = tgt_y + (cmd[:, 1] / (4.0 * f_.clamp(min=0.5))).unsqueeze(1)
    err = torch.sqrt((rx - tgt_x) ** 2 + (ry - tgt_y) ** 2)
    td_ok = landed & (s["valid_swing"] > 0.5)
    s["ev_td"] = td_ok.float()
    s["td_err"] = torch.where(td_ok, err, torch.zeros_like(err))
    s["td_err_h"] = torch.where(td_ok, err, s["td_err_h"])                # 1006 레드팀: 유효 스텝(하중 수용) 때 지급하려고 착지값 보관
    d = fxy - s["lift_xy"]                                                # 스윙발 이륙→착지 월드 변위(요 프레임 x) — 몸 기울임으로 못 만드는 진행량
    dx = d[..., 0] * cy.unsqueeze(1) + d[..., 1] * sy.unsqueeze(1)
    s["td_dx"] = torch.where(td_ok, dx, torch.zeros_like(dx))
    s["td_dx_h"] = torch.where(td_ok, dx, s["td_dx_h"])
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


def _v_ema(env, tau: float = 0.5):
    """1006 S1 단위시험: 순간 속도는 걸음 흔들림(±0.1 m/s)이 명령 차(0.08)를 덮는다 → 한 보행 주기 수준 EMA(τ 0.5 s) 월드 수평 속도.
    틱당 1회 갱신(같은 틱 재호출은 캐시), 에피소드 시작에 현재 속도로 초기화."""
    asset = env.scene[H._base_cfg().name]
    v = asset.data.root_lin_vel_w[:, :2]
    tick = int(env.common_step_counter)
    st = getattr(env, "_w3_vema", None)
    if st is None or st[0].shape[0] != env.num_envs:
        st = [v.clone(), -1]; env._w3_vema = st
    if st[1] != tick:
        a = min(1.0, env.step_dt / tau)
        fresh = (env.episode_length_buf <= 1).unsqueeze(1)
        st[0] = torch.where(fresh, v, st[0] + a * (v - st[0])); st[1] = tick
    return st[0]


def _v_box(env, period_s: float = 1.0 / 1.1):
    """1006 S1 단위시험 3차: EMA(τ 0.5)는 0.55 Hz 좌우 흔들림(±0.14 m/s)을 절반(≈0.07)만 깎아 명령 0.03 추종을 덮음 →
    정확히 한 보행 주기 이동 평균 = (p_t − p_{t−T})/T (주기 성분 소거). 에피소드 앞 T 동안은 시작점 기준."""
    asset = env.scene[H._base_cfg().name]
    p = asset.data.root_pos_w[:, :2]
    tick = int(env.common_step_counter)
    K = max(2, int(round(period_s / env.step_dt)))
    st = getattr(env, "_w3_vbox", None)
    if st is None or st["buf"].shape[0] != env.num_envs or st["buf"].shape[1] != K:
        st = {"buf": p.unsqueeze(1).repeat(1, K, 1).clone(), "i": 0, "tick": -1, "v": torch.zeros_like(p)}; env._w3_vbox = st
    if st["tick"] != tick:
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            st["buf"][fresh] = p[fresh].unsqueeze(1)
        i = st["i"]; old = st["buf"][:, i].clone()                     # K 틱 전(또는 에피소드 시작) 위치
        st["buf"][:, i] = p; st["i"] = (i + 1) % K
        n = env.episode_length_buf.clamp(min=1, max=K).float().unsqueeze(1)
        st["v"] = (p - old) / (n * env.step_dt); st["tick"] = tick
    return st["v"]


def _v_err(env, box: bool = False):
    asset = env.scene[H._base_cfg().name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    cmd = env.command_manager.get_command("base_velocity")
    cw = torch.stack([cmd[:, 0] * torch.cos(yaw) - cmd[:, 1] * torch.sin(yaw), cmd[:, 0] * torch.sin(yaw) + cmd[:, 1] * torch.cos(yaw)], -1)
    return ((_v_box(env) if box else _v_ema(env)) - cw).norm(dim=-1), cmd[:, :2].norm(dim=-1)


def w3_drift(env) -> torch.Tensor:
    """STEP 수평 속도 오차 |v − v_cmd|(명령 0 이면 이동 자체) — 벌점 크기(양수)."""
    e, _ = _v_err(env)
    return e * _ge(env).float()


def w3_track(env, sig: float = 0.05, cmin: float = 0.02) -> torch.Tensor:
    """Stage 1+: 명령 ≠ 0 일 때만 exp(−|v − v_cmd|²/σ²), σ 0.05(작은 속도에서 부호 구분 — 넓은 σ 는 서기에 공짜 점수)."""
    e, cn = _v_err(env)
    return torch.exp(-(e / sig) ** 2) * (cn >= cmin).float() * _ge(env).float()


def w3_track_rel(env, sig: float = 0.05, cmin: float = 0.02) -> torch.Tensor:
    """Stage 1 v2: 서기 대비 개선분 exp(−e²/σ²) − exp(−|c|²/σ²) — 서기(v 0) = 0 점, 반대 방향 < 0. 명령 0 은 0(Stage 0 과 같음).
    속도 = 한 보행 주기 이동 평균(_v_box, 흔들림 소거)."""
    e, cn = _v_err(env, box=True)
    return (torch.exp(-(e / sig) ** 2) - torch.exp(-(cn / sig) ** 2)) * (cn >= cmin).float() * _ge(env).float()


def w3_stride(env, sig: float = 0.03, cmin: float = 0.02) -> torch.Tensor:
    """Stage 1 v2: 유효 착지 사건마다 스윙발 변위 x 가 ΔX* = vx/f 에 맞는 정도 − 제자리 디딤(Δx 0)의 점수 → 제자리 = 0, 반대 = 음수.
    명령 0 은 0(Stage 0 과 같음). 지지발 미끄럼·몸 기울임으로는 못 얻음(스윙발 자체 변위)."""
    s = _st(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    tx = (cmd[:, 0] / f_.clamp(min=0.5)).unsqueeze(1)
    r = (torch.exp(-((s["td_dx"] - tx) / sig) ** 2) - torch.exp(-(tx / sig) ** 2)) * s["ev_td"]
    return r.sum(1) * (cmd[:, 0].abs() >= cmin).float() * _ge(env).float()


def _yaw_box(env, period_s: float = 1.0 / 1.1):
    """한 보행 주기 순 요 속도 = wrap(ψ_t − ψ_{t−T})/T — 걸음마다의 요 흔들림 소거, 누적 표류만 남김."""
    asset = env.scene[H._base_cfg().name]
    q = asset.data.root_quat_w
    psi = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    tick = int(env.common_step_counter)
    K = max(2, int(round(period_s / env.step_dt)))
    st = getattr(env, "_w3_ybox", None)
    if st is None or st["buf"].shape[0] != env.num_envs or st["buf"].shape[1] != K:
        st = {"buf": psi.unsqueeze(1).repeat(1, K).clone(), "i": 0, "tick": -1, "w": torch.zeros_like(psi)}; env._w3_ybox = st
    if st["tick"] != tick:
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            st["buf"][fresh] = psi[fresh].unsqueeze(1)
        i = st["i"]; old = st["buf"][:, i].clone()
        st["buf"][:, i] = psi; st["i"] = (i + 1) % K
        n = env.episode_length_buf.clamp(min=1, max=K).float()
        d = torch.atan2(torch.sin(psi - old), torch.cos(psi - old))
        st["w"] = d / (n * env.step_dt); st["tick"] = tick
    return st["w"]


def _gy_box(env, period_s: float = 1.0 / 1.1):
    """한 보행 주기 평균 옆 기울기 = 몸통 좌표 투영 중력 y 의 이동 평균(K 틱) — 걸음마다의 좌우 흔들림은 소거, 한쪽으로 쏠린 걸음(lean)만 남김.
    1007 claude2(TRANSITION_W3KE.md §5.19): 왕복에서 걷는 중 평균 롤과 요 속도가 매 걷기 r 0.67–0.93·기울기 ≈ 2 (°/s)/° 로 묶여 있고, 멈췄다 다시 걸을 때마다 커진다."""
    asset = env.scene[H._base_cfg().name]
    gy = asset.data.projected_gravity_b[:, 1]
    tick = int(env.common_step_counter)
    K = max(2, int(round(period_s / env.step_dt)))
    st = getattr(env, "_w3_gybox", None)
    if st is None or st["buf"].shape[0] != env.num_envs or st["buf"].shape[1] != K:
        st = {"buf": gy.unsqueeze(1).repeat(1, K).clone(), "i": 0, "tick": -1, "m": gy.clone()}; env._w3_gybox = st
    if st["tick"] != tick:
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            st["buf"][fresh] = gy[fresh].unsqueeze(1)
        i = st["i"]
        st["buf"][:, i] = gy; st["i"] = (i + 1) % K
        st["m"] = st["buf"].mean(dim=1); st["tick"] = tick
    return st["m"]


def w3_lean_net(env) -> torch.Tensor:
    """W3S3g: 한 주기 평균 옆 기울기 |ḡ_y|(≈ 평균 롤 rad) — 벌점 크기(양수), STEP 에서만. 제곱형(w3_orient)은 1–2° 쏠림에 기울기가 거의 없어 선형으로 둔다."""
    return _gy_box(env).abs() * _ge(env).float()


def w3_yaw_net(env) -> torch.Tensor:
    """W3-A3: 한 주기 순 요 속도 오차 |ψ̇_T − wz_cmd| — 벌점 크기. 순간 |ω_z|(w3_yaw_drift)는 걸음 요 흔들림이 지배해 표류를 못 잡음(W3-A2 69 → 65°)."""
    c = env.command_manager.get_command("base_velocity")
    return (_yaw_box(env) - c[:, 2]).abs() * _ge(env).float()


def w3_drift_net(env) -> torch.Tensor:
    """W3-A3: 한 주기 이동평균 속도 오차 |v_T − v_cmd| — 벌점 크기(EMA 잔여 흔들림 0.07 m/s 제거)."""
    e, _ = _v_err(env, box=True)
    return e * _ge(env).float()


def w3_place_valid(env, sig: float = 0.04) -> torch.Tensor:
    """1006 레드팀 결함 1·4 수정판: 착지 오차 보상을 **유효 스텝(공중·정점·하중 수용 통과) 사건에만** — 착지 사건마다 지급하던 w3_place_reward 는
    두드리기·잘게 딛기로 반주기당 여러 번 받을 수 있었음(목표 폭이 틀려 죽어 있던 동안은 잠복)."""
    s = _st(env)
    return (torch.exp(-(s["td_err_h"] / sig) ** 2) * s["ev_valid"]).sum(1) * _ge(env).float()


def w3_stride_valid(env, sig: float = 0.03, cmin: float = 0.02) -> torch.Tensor:
    """w3_stride 의 유효 스텝 지급판(같은 식, ev_td → ev_valid, 착지 변위는 보관값)."""
    s = _st(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    tx = (cmd[:, 0] / f_.clamp(min=0.5)).unsqueeze(1)
    r = (torch.exp(-((s["td_dx_h"] - tx) / sig) ** 2) - torch.exp(-(tx / sig) ** 2)) * s["ev_valid"]
    return r.sum(1) * (cmd[:, 0].abs() >= cmin).float() * _ge(env).float()


def w3_stand_gated(env, func, params: dict | None = None) -> torch.Tensor:
    """기존(서기) 보상을 STAND(ge 0)에만."""
    val = func(env, **H._resolve_nested(env, params))
    return val * (~_ge(env)).to(val.dtype)


def w3_step_gated(env, func, params: dict | None = None) -> torch.Tensor:
    val = func(env, **H._resolve_nested(env, params))
    return val * _ge(env).to(val.dtype)


# ================================================================ Stage 2(1006 W3S2) — 옆걸음(vy) ================================================================
# 설계·사전 통과 기준: reports/gen_w3/S2_DESIGN.md. 기존 w3_step_tracker·w3_stride 는 손대지 않는다(Stage 0/1 비트 동일) —
# 옆 변위·폭은 별도 상태(env._w3lat)를 쓰는 후행 추적기(w3_lat_tracker, w3_step_tracker **뒤** 정의 순서)가 계산한다.
def _lat(env):
    L = getattr(env, "_w3lat", None)
    if L is None or L["td_dy"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        z = lambda *sh: torch.zeros(*sh, device=dev)
        L = {"td_dx": z(N, 2), "td_dy": z(N, 2), "width": z(N), "td_width": z(N, 2), "width_bad": z(N, 2), "cross": z(N, 2),
             # 누적 카운터(커리큘럼·로그용, 리셋 안 함): 옆 명령 착지 수·부호 정답 수·STEP 착지 수·폭 위반 수
             "n_lat_td": z(1), "n_lat_ok": z(1), "n_td": z(1), "n_bad": z(1)}
        env._w3lat = L
    return L


def _yaw_of(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def w3_lat_tracker(env, env_ids, nominal_half_width: float = 0.10, width_margin: float = 0.05, cmin: float = 0.02):
    """mode="interval"(0 s), **w3_step_tracker 뒤**: 이번 틱 유효 착지(_st.ev_td)마다
    - td_dx / td_dy = 스윙발 이륙→착지 **월드 변위**의 요 프레임 성분(몸 기울임·지지발 미끄럼으로 못 만드는 진행량; td_dx 는 w3_step_tracker 와 같은 식).
    - td_width = 착지 순간 좌우 발 간격 W = y_왼 − y_오(골반 요 프레임, 부호 있음). width_bad = W < 2w − margin, cross = W < 0(교차).
    상태는 env._w3lat(기존 env._w3 키 불변)."""
    s = _st(env); L = _lat(env)
    asset = env.scene[H._base_cfg().name]
    bid = H._g21_bids(env); fxy = asset.data.body_pos_w[:, bid, :2]
    yaw = _yaw_of(asset.data.root_quat_w); cy, sy = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    rel = fxy - asset.data.root_pos_w[:, None, :2]
    ry = -rel[..., 0] * sy + rel[..., 1] * cy                             # (N,2) 왼·오 발 요 프레임 y
    W = ry[:, 0] - ry[:, 1]
    L["width"] = W
    td = s["ev_td"] > 0.5
    d = fxy - s["lift_xy"]
    dx = d[..., 0] * cy + d[..., 1] * sy; dy = -d[..., 0] * sy + d[..., 1] * cy
    zero = torch.zeros_like(dx)
    L["td_dx"] = torch.where(td, dx, zero); L["td_dy"] = torch.where(td, dy, zero)
    Wb = W.unsqueeze(1).expand(-1, 2)
    L["td_width"] = torch.where(td, Wb, zero)
    wmin = 2.0 * nominal_half_width - width_margin
    L["width_bad"] = (td & (Wb < wmin)).float(); L["cross"] = (td & (Wb < 0.0)).float()
    ge = _ge(env)
    cmd = env.command_manager.get_command("base_velocity")
    tdg = td & ge.unsqueeze(1)
    lat = (cmd[:, 1].abs() >= cmin).unsqueeze(1) & tdg
    ok = lat & (dy * torch.sign(cmd[:, 1]).unsqueeze(1) > 0.0)
    L["n_lat_td"] += lat.float().sum(); L["n_lat_ok"] += ok.float().sum()
    L["n_td"] += tdg.float().sum(); L["n_bad"] += (L["width_bad"] * tdg.float()).sum()


def w3_stride2(env, sig: float = 0.03, cmin: float = 0.02, width_gate: bool = True, k_opp: float = 0.5) -> torch.Tensor:
    """Stage 2: 스윙발 이륙→착지 변위 2D 목표 ΔP* = (vx/f, vy/f) — 유효 착지마다
    폭 게이트·[exp(−‖Δp − ΔP*‖²/σ²) − exp(−‖ΔP*‖²/σ²)] + k_opp·clip(Δp·ΔP*/‖ΔP*‖², −1, 0)
    (제자리 디딤 = 0, 기울임만(착지 없음) = 0, 반대 방향 < 0). 둘째 항(반대 방향 벌점)이 필요한 이유(CPU 단위시험 1차):
    첫 항만이면 |vy| 0.05 반대 = −0.10, 0.08 반대 = 0.000(= 제자리) — 기준점 exp(−‖ΔP*‖²/σ²) 가 큰 명령에서 0 으로 사라져 방향 구분이 없어진다.
    **옆 명령(|vy| ≥ cmin)만** 이 식 + 폭 게이트(width_bad·교차 착지는 첫 항 0 점; 반대 벌점은 유지).
    |vy| < cmin 이면 w3_stride(1D x)를 그대로 돌려준다 → 앞뒤·제자리 리허설은 Stage 1 과 비트 동일."""
    base = w3_stride(env, sig=sig, cmin=cmin)
    s = _st(env); L = _lat(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    fc = f_.clamp(min=0.5)
    tx = (cmd[:, 0] / fc).unsqueeze(1); ty = (cmd[:, 1] / fc).unsqueeze(1)
    e2 = (L["td_dx"] - tx) ** 2 + (L["td_dy"] - ty) ** 2
    r = (torch.exp(-e2 / sig ** 2) - torch.exp(-(tx ** 2 + ty ** 2) / sig ** 2)) * s["ev_td"]
    if width_gate:
        r = r * (1.0 - L["width_bad"])
    n2 = (tx ** 2 + ty ** 2).clamp(min=1e-8)
    r = r + k_opp * ((L["td_dx"] * tx + L["td_dy"] * ty) / n2).clamp(-1.0, 0.0) * s["ev_td"]
    lat = r.sum(1) * _ge(env).float()
    return torch.where(cmd[:, 1].abs() >= cmin, lat, base)


def w3_width_pen(env, w_cross: float = 1.0) -> torch.Tensor:
    """Stage 2 폭 제약 벌점 크기(양수, 사건): 착지 순간 W < 2w − margin 이면 1, 교차(W < 0)면 1 + w_cross. STEP 만.
    (명목 폭 ≈ 2w 의 정상 걸음은 0 점 — S2_DESIGN 반사실 시험에서 서기·Stage 1 궤적 위반률 확인)."""
    L = _lat(env)
    return (L["width_bad"] + w_cross * L["cross"]).sum(1) * _ge(env).float()


def w3_vy_curriculum(env, env_ids, vmax: tuple = (0.04, 0.06, 0.08), vmin: float = 0.03, min_sign: float = 0.90, max_bad: float = 0.02,
                     min_iters: int = 100, min_lat_td: float = 3000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """|vy| 상한 커리큘럼(0.04 → 0.06 → 0.08). 블록(옆 명령 착지 ≥ min_lat_td) 판정 — 부호 정답률 ≥ min_sign 이고
    폭 위반률(전 STEP 착지) < max_bad, 체류 ≥ min_iters 면 다음 상한. 절대 문턱(톱니 금지). 이전 단계 회귀는 평가(eval_w3)에서 본다."""
    L = _lat(env); g = getattr(env, "_w3_vycur", None)
    if g is None:
        g = {"stage": int(start_stage), "since": 0, "b_lat": 0.0, "b_ok": 0.0, "b_td": 0.0, "b_bad": 0.0, "sign": float("nan"), "bad": float("nan")}
        env._w3_vycur = g
        env._w3_vy_range = (vmin, vmax[g["stage"]])
    n_lat, n_ok, n_td, n_bad = (float(L[k].sum()) for k in ("n_lat_td", "n_lat_ok", "n_td", "n_bad"))
    it = int(env.common_step_counter) // steps_per_iter
    if n_lat - g["b_lat"] >= min_lat_td:
        sg = (n_ok - g["b_ok"]) / (n_lat - g["b_lat"]); bd = (n_bad - g["b_bad"]) / max(n_td - g["b_td"], 1.0)
        g["sign"], g["bad"] = sg, bd
        adv = sg >= min_sign and bd < max_bad and it - g["since"] >= min_iters and g["stage"] < len(vmax) - 1
        print(f"[W3S2 vy 블록] 반복 {it} |vy|≤{vmax[g['stage']]}: 부호 정답 {sg:.3f} · 폭 위반 {bd:.3f}" + (" → 상한 확대" if adv else ""), flush=True)
        if adv:
            g["stage"] += 1; g["since"] = it; env._w3_vy_range = (vmin, vmax[g["stage"]])
        g["b_lat"], g["b_ok"], g["b_td"], g["b_bad"] = n_lat, n_ok, n_td, n_bad
    return {"vy_max": float(env._w3_vy_range[1]), "lat_sign_ok": g["sign"], "width_bad": g["bad"]}


# ================================================================ Stage 3(1006 W3S3) — 제자리 회전(wz). 기본 꺼짐(W3S3 계열 설정을 고를 때만) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/S3_DESIGN.md. CPU 단위시험: rl_walking/scripts/unit_w3_s3.py.
# 기존 상태(env._w3·env._w3lat)·함수는 손대지 않는다 — 회전 상태는 env._w3turn, 후행 추적기(w3_turn_tracker, w3_step_tracker **뒤** 정의 순서).
# 요 측정은 프레임 독립형: 발 링크 쿼터니언의 **월드 상대 회전** q_now ⊗ q_ref* 의 요 성분(발 링크 축 규약과 무관), 몸통 대비 비틀림은 발 연결선 방향으로.
def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _qconj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def _rel_yaw(q_now, q_ref):
    """월드 프레임 상대 회전 R_now·R_refᵀ 의 요[rad] — 링크 축 규약과 무관(기울기 작을 때 순수 수직축 회전량)."""
    r = _qmul(q_now, _qconj(q_ref)); w, x, y, z = r.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _feet_line_yaw(env):
    """발 연결선 방향 요 ψ_fl = atan2(p_왼 − p_오) − π/2 (두 발이 정면 나란히면 = 몸통 요). 몸통만 비틀면 변하지 않는다."""
    asset = env.scene[H._base_cfg().name]
    bid = H._g21_bids(env); fxy = asset.data.body_pos_w[:, bid, :2]
    d = fxy[:, 0] - fxy[:, 1]
    return torch.atan2(d[:, 1], d[:, 0]) - 0.5 * math.pi


def _turn(env):
    T_ = getattr(env, "_w3turn", None)
    if T_ is None or T_["td_dpsi"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        z = lambda *sh: torch.zeros(*sh, device=dev)
        T_ = {"q_lift": torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(N, 2, 1), "q_prev": None, "td_dpsi": z(N, 2),
              "pivot": z(N, 2), "twist": z(N), "c_prev": torch.ones(N, 2, device=dev),
              # 누적(커리큘럼·로그, 리셋 안 함): 회전 명령 착지 수·부호 정답·요 이득 분자/분모·지지발 피벗 |요|·스윙 착지 |요|·부호 있는 합
              "n_turn_td": z(1), "n_turn_ok": z(1), "gain_num": z(1), "gain_den": z(1), "piv_abs": z(1), "swing_abs": z(1),
              "piv_sgn": z(1), "swing_sgn": z(1)}
        env._w3turn = T_
    return T_


def w3_turn_tracker(env, env_ids, threshold: float = 5.0, cmin: float = 0.08):
    """mode="interval"(0 s), **w3_step_tracker 뒤**: 발별
    - td_dpsi = 스윙발 이륙→착지 **월드 요 변위**(_st.ev_td 사건에만, 그 외 0). 목표 Δψ* = wz/f.
    - pivot = 이번 틱 **접촉 유지 중인** 발의 요 변화[rad](지지발 비틀림 미끄럼·한발 피벗 — 감시 지표, 보상 아님).
    - twist = wrap(ψ_골반 − ψ_발연결선)(몸통만 비틀기 감시).
    상태는 env._w3turn(기존 env._w3·env._w3lat 키 불변)."""
    s = _st(env); T_ = _turn(env)
    asset = env.scene[H._base_cfg().name]
    bid = H._g21_bids(env); qf = asset.data.body_quat_w[:, bid, :]
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    c = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > threshold)
    fresh = (env.episode_length_buf <= 1)
    if T_["q_prev"] is None:
        T_["q_prev"] = qf.clone()
    if bool(fresh.any()):
        T_["q_lift"][fresh] = qf[fresh]; T_["q_prev"][fresh] = qf[fresh]; T_["c_prev"][fresh] = 1.0
    lifted = s["ev_takeoff"] > 0.5
    T_["q_lift"] = torch.where(lifted.unsqueeze(-1), qf, T_["q_lift"])
    td = s["ev_td"] > 0.5
    dpsi = _rel_yaw(qf, T_["q_lift"])
    T_["td_dpsi"] = torch.where(td, dpsi, torch.zeros_like(dpsi))
    stance = c & (T_["c_prev"] > 0.5) & ~fresh.unsqueeze(1)
    dtick = _rel_yaw(qf, T_["q_prev"])
    T_["pivot"] = torch.where(stance, dtick, torch.zeros_like(dtick))
    psi = _yaw_of(asset.data.root_quat_w)
    T_["twist"] = _wrap(psi - _feet_line_yaw(env))
    T_["q_prev"] = qf.clone(); T_["c_prev"] = c.float()
    # 누적 지표(STEP·회전 명령만)
    ge = _ge(env)
    wz = env.command_manager.get_command("base_velocity")[:, 2]
    tm = ge & (wz.abs() >= cmin); sg = torch.sign(wz).unsqueeze(1)
    tdg = td & tm.unsqueeze(1)
    T_["n_turn_td"] += tdg.float().sum(); T_["n_turn_ok"] += (tdg & (dpsi * sg > 0.0)).float().sum()
    T_["gain_num"] += (_yaw_box(env) * torch.sign(wz) * tm.float()).sum(); T_["gain_den"] += (wz.abs() * tm.float()).sum()
    T_["piv_abs"] += (T_["pivot"].abs() * tm.unsqueeze(1).float()).sum(); T_["swing_abs"] += (T_["td_dpsi"].abs() * tdg.float()).sum()
    T_["piv_sgn"] += (T_["pivot"] * sg * tm.unsqueeze(1).float()).sum(); T_["swing_sgn"] += (T_["td_dpsi"] * sg * tdg.float()).sum()


def w3_turn_stride(env, sig: float = 0.0873, cmin: float = 0.08, k_opp: float = 0.5) -> torch.Tensor:
    """Stage 3: 스윙발 착지 사건마다 발 요 변위 Δψ 가 Δψ* = wz/f 에 맞는 정도(Stage 2 w3_stride2 와 같은 설계):
    [exp(−(Δψ − Δψ*)²/σ²) − exp(−Δψ*²/σ²)] + k_opp·clip(Δψ·Δψ*/Δψ*², −1, 0). σ 5°(0.0873 rad).
    제자리 디딤(Δψ 0) = 0, 몸통만 비틀기(발 요 불변) = 0, 반대 방향 < 0. |wz| < cmin 이면 **정확히 0**(Stage 0–2 와 총 보상 비트 동일)."""
    s = _st(env); T_ = _turn(env)
    wz = env.command_manager.get_command("base_velocity")[:, 2]
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    tp = (wz / f_.clamp(min=0.5)).unsqueeze(1)
    d = T_["td_dpsi"]
    r = (torch.exp(-((d - tp) / sig) ** 2) - torch.exp(-(tp / sig) ** 2)) * s["ev_td"]
    r = r + k_opp * ((d * tp) / (tp * tp).clamp(min=1e-8)).clamp(-1.0, 0.0) * s["ev_td"]
    on = (wz.abs() >= cmin) & _ge(env)
    return torch.where(on, r.sum(1), torch.zeros_like(wz))


def _feet_yaw_box(env, period_s: float = 1.0 / 1.1):
    """한 보행 주기 발 연결선 요 순속도 = wrap(ψ_fl,t − ψ_fl,t−T)/T — 걸음 주기 성분 소거. 몸통 비틀기로는 못 만든다."""
    psi = _feet_line_yaw(env)
    tick = int(env.common_step_counter)
    K = max(2, int(round(period_s / env.step_dt)))
    st = getattr(env, "_w3_fybox", None)
    if st is None or st["buf"].shape[0] != env.num_envs or st["buf"].shape[1] != K:
        st = {"buf": psi.unsqueeze(1).repeat(1, K).clone(), "i": 0, "tick": -1, "w": torch.zeros_like(psi)}; env._w3_fybox = st
    if st["tick"] != tick:
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            st["buf"][fresh] = psi[fresh].unsqueeze(1)
        i = st["i"]; old = st["buf"][:, i].clone()
        st["buf"][:, i] = psi; st["i"] = (i + 1) % K
        n = env.episode_length_buf.clamp(min=1, max=K).float()
        st["w"] = _wrap(psi - old) / (n * env.step_dt); st["tick"] = tick
    return st["w"]


def w3_yaw_track_rel(env, sig: float = 0.2, cmin: float = 0.08, src: str = "feet") -> torch.Tensor:
    """Stage 3: 회전 추종 개선분 exp(−(ψ̇_T − wz)²/σ²) − exp(−wz²/σ²) — 가만히 = 0, 반대 < 0. |wz| < cmin 이면 정확히 0.
    src="feet"(기본): 발 연결선 요 한 주기 순속도(_feet_yaw_box) — 몸통만 톱니형으로 비틀어 지수형 추종 점수를 따는 해킹 차단
    (CPU 단위시험 [C]: src="pelvis" 는 톱니 비틀기만으로 + 점수 → 기각). 순 요 오차 선형 벌점 w3_yaw_net(골반)은 그대로 둔다."""
    wz = env.command_manager.get_command("base_velocity")[:, 2]
    w = _feet_yaw_box(env) if src == "feet" else _yaw_box(env)
    r = torch.exp(-((w - wz) / sig) ** 2) - torch.exp(-(wz / sig) ** 2)
    on = (wz.abs() >= cmin) & _ge(env)
    return torch.where(on, r, torch.zeros_like(wz))


def w3_heading_loop(env, env_ids, kp: float = 0.5, wz_max: float = 0.4):
    """mode="interval"(0 s), 선택(W3S3h): **배포와 같은 외부 헤딩 P 고리** — env._w3_hl(리샘플 때 STEP·비회전 명령 일부) env 에서
    wz 명령 = clip(wz₀ + kp·wrap(ψ* − ψ), ±wz_max), ψ* = 걷기 시작(에피소드 시작 또는 고리 켜짐) 요에서 wz₀ 로 적분(비회전 명령은 wz₀ = 0 → ψ* 고정).
    그 외 env 는 wz = wz₀(리샘플 값) 그대로, ψ* = 현재 요로 따라간다. W3_HLOOP=0 이면 끔(eval_w3 는 자체 고리 --heading_loop 사용).
    관측·보상은 이 고리 출력(배포에서 정책이 보는 값)을 본다. 이벤트는 command 갱신 뒤에 돌아 다음 틱 관측·보상에 반영된다."""
    if os.environ.get("W3_HLOOP", "1") == "0" or getattr(env, "_w3_hl", None) is None:
        return
    term = env.command_manager.get_term("base_velocity")
    asset = env.scene[H._base_cfg().name]
    psi = _yaw_of(asset.data.root_quat_w)
    tgt = getattr(env, "_w3_hl_tgt", None)
    if tgt is None or tgt.shape[0] != env.num_envs:
        tgt = psi.clone()
    act = env._w3_hl & _ge(env)
    prev_act = getattr(env, "_w3_hl_prev", None)
    if prev_act is None or prev_act.shape[0] != env.num_envs:
        prev_act = torch.zeros_like(act)
    start = (env.episode_length_buf <= 1) | (act & ~prev_act)
    wz0 = env._w3_wz0
    tgt = torch.where(start | ~act, psi, _wrap(tgt + wz0 * env.step_dt))
    err = _wrap(tgt - psi)
    wz = (wz0 + kp * err).clamp(-wz_max, wz_max)
    term.vel_command_b[:, 2] = torch.where(act, wz, wz0)
    env._w3_hl_tgt = tgt; env._w3_hl_prev = act


def w3_wz_curriculum(env, env_ids, vmax: tuple = (0.2, 0.3, 0.4), vmin: float = 0.1, min_sign: float = 0.90, min_gain: float = 0.5,
                     min_iters: int = 100, min_turn_td: float = 3000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """|wz| 상한 커리큘럼(0.2 → 0.3 → 0.4, 하한 0.1). 블록(회전 명령 착지 ≥ min_turn_td) 판정 — 착지 발 요 부호 정답률 ≥ min_sign 이고
    요 이득(골반 한 주기 순 요 속도·부호 / |wz|) ≥ min_gain, 체류 ≥ min_iters 면 다음 상한. 절대 문턱(톱니 금지). 피벗 비율은 로그만."""
    T_ = _turn(env); g = getattr(env, "_w3_wzcur", None)
    if g is None:
        g = {"stage": int(start_stage), "since": 0, "b_td": 0.0, "b_ok": 0.0, "b_gn": 0.0, "b_gd": 0.0, "b_pa": 0.0, "b_sa": 0.0,
             "sign": float("nan"), "gain": float("nan"), "pivot_frac": float("nan")}
        env._w3_wzcur = g
        env._w3_wz_range = (vmin, vmax[g["stage"]])
    # 피벗 비율 = 명령 방향 부호 기여 피벗 / (피벗 + 스윙). |·| 합은 제자리 걸음에도 지지발 요 떨림(≈ 21–25°/s, 1006 GPU 지형)이 지배해 쓰지 않는다.
    n_td, n_ok, gn, gd, pa, sa = (float(T_[k].sum()) for k in ("n_turn_td", "n_turn_ok", "gain_num", "gain_den", "piv_sgn", "swing_sgn"))
    it = int(env.common_step_counter) // steps_per_iter
    if n_td - g["b_td"] >= min_turn_td:
        sg = (n_ok - g["b_ok"]) / (n_td - g["b_td"]); gain = (gn - g["b_gn"]) / max(gd - g["b_gd"], 1e-9)
        _dp, _ds = pa - g["b_pa"], sa - g["b_sa"]
        pf = _dp / (_dp + _ds) if abs(_dp + _ds) > 1e-9 else float("nan")
        g["sign"], g["gain"], g["pivot_frac"] = sg, gain, pf
        adv = sg >= min_sign and gain >= min_gain and it - g["since"] >= min_iters and g["stage"] < len(vmax) - 1
        print(f"[W3S3 wz 블록] 반복 {it} |wz|≤{vmax[g['stage']]}: 발 요 부호 정답 {sg:.3f} · 요 이득 {gain:.3f} · 피벗 비율 {pf:.3f}"
              + (" → 상한 확대" if adv else ""), flush=True)
        if adv:
            g["stage"] += 1; g["since"] = it; env._w3_wz_range = (vmin, vmax[g["stage"]])
        g["b_td"], g["b_ok"], g["b_gn"], g["b_gd"], g["b_pa"], g["b_sa"] = n_td, n_ok, gn, gd, pa, sa
    return {"wz_max": float(env._w3_wz_range[1]), "turn_sign_ok": g["sign"], "yaw_gain": g["gain"], "pivot_frac": g["pivot_frac"]}


# ================================================================ Stage 4(1006 W3S4) — 혼합 명령(vx·vy·wz 동시, 곡선 보행). 기본 꺼짐(W3S4 설정을 고를 때만) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/S4_DESIGN.md. CPU 단위시험: rl_walking/scripts/unit_w3_s4.py.
# 기존 함수·상태(env._w3·_w3lat·_w3turn)는 손대지 않는다(install_w3_cmd 의 stage 4 분기만 추가). 새 상태 env._w3mix, 후행 추적기 w3_mix_tracker
# (w3_turn_tracker **뒤** 정의 순서). 새 보상(w3_stride4·w3_track_rel4·w3_drift_net4)은 |wz| < wcmin 또는 병진 명령 없음이면 기존 함수 값을 그대로
# 돌려준다(torch.where 로 기존 텐서 원소 선택 → Stage 1–3 명령에서 비트 동일).
#
# 곡선 착지 목표(정상 상태 강체 운동학): 몸이 본체 프레임 일정 속도 (vx, vy)·요 속도 wz 로 움직이고 발이 매 주기 같은 본체 프레임 위치 p 에 착지하면,
# 같은 발의 연속 착지 변위(= 이륙→착지 변위, 지지 중 미끄럼 없음)는 **직전 착지 순간 골반 요 프레임**에서
#     ΔP* = R(θ/2)·sinc(θ/2)·(vx, vy)/f  +  (R(θ) − I)·p,      θ = wz/f,  sinc(x) = sin x / x,   p = (0, ±w)(왼 +, 오 −)
# 첫 항 = 몸 경로 현(chord)(방향은 반주기 회전분 θ/2 만큼 돌아감), 둘째 항 = 발 횡 오프셋의 회전(안쪽 발은 덜, 바깥 발은 더 감 — 지레 항).
# 이 프레임은 그 발이 놓인 순간의 몸 방향이고, 지지 중 발 피벗이 없으면 = 이륙 순간 그 발의 요 → 브리프의 "이륙 요 프레임" 해석.
# (이륙 순간 **골반** 요는 주기 중간이라 이미 θ/2 를 돌아 있어, 그 프레임에서 θ/2 를 또 돌리면 이중 회전 — 단위시험 [B] 변형 비교.)
# 발 요 목표 Δψ* = wz/f 는 Stage 3 w3_turn_stride 그대로(곡선에서도 같은 식).
def _mix(env):
    M = getattr(env, "_w3mix", None)
    if M is None or M["psi_td"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        z = lambda *sh: torch.zeros(*sh, device=dev)
        M = {"psi_td": z(N, 2), "psi_ref": z(N, 2), "psi_lo": z(N, 2), "c_prev": torch.ones(N, 2, device=dev), "td_d": z(N, 2, 2), "init": False,
             # 누적(커리큘럼·로그, 리셋 안 함): 혼합 명령 착지 수 · 축별 활성 착지 수/부호 정답(지레·회전 보정 뒤) · 폭 위반
             "n_td": z(1), "n_x": z(1), "ok_x": z(1), "n_y": z(1), "ok_y": z(1), "n_w": z(1), "ok_w": z(1), "n_bad": z(1)}
        env._w3mix = M
    return M


def _rot2(a, v):
    """각 a(…) 로 (…,2) 벡터 회전."""
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _sinc(x):
    small = x.abs() < 1e-6
    return torch.where(small, torch.ones_like(x), torch.sin(x) / torch.where(small, torch.ones_like(x), x))


def w3_curve_target(cmd, f, nominal_half_width: float = 0.144, rot_frac: float = 0.5, lever: bool = True, sinc: bool = True):
    """곡선 착지 목표 ΔP*(N, 2발, 2) — 측정 프레임이 '직전 착지 골반 요 + (0.5 − rot_frac)·θ' 일 때의 식.
    기본(rot_frac 0.5·lever·sinc) = 직전 착지 골반 요 프레임의 정확한 강체 운동학. rot_frac 0 = 주기 중간(이륙 순간 골반) 프레임용.
    lever=False·sinc=False·rot_frac 0.5 = 브리프 문자식 R(θ/2)·(vx, vy)/f. wz 0 이면 (vx, vy)/f(Stage 2 w3_stride2 와 같은 목표)."""
    fc = f.clamp(min=0.5)
    th = cmd[:, 2] / fc
    v = torch.stack([cmd[:, 0] / fc, cmd[:, 1] / fc], -1)
    if sinc:
        v = v * _sinc(0.5 * th).unsqueeze(-1)
    T = _rot2(rot_frac * th, v).unsqueeze(1).repeat(1, 2, 1)
    if lever:
        p = torch.tensor([[0.0, nominal_half_width], [0.0, -nominal_half_width]], device=cmd.device, dtype=cmd.dtype).unsqueeze(0).expand(cmd.shape[0], 2, 2)
        th2 = th.unsqueeze(1).expand(-1, 2)
        T = T + _rot2(-(0.5 - rot_frac) * th2, _rot2(th2, p) - p)
    return T


def w3_mix_tracker(env, env_ids, threshold: float = 5.0, cmin: float = 0.02, wcmin: float = 0.08, nominal_half_width: float = 0.144):
    """mode="interval"(0 s), **w3_turn_tracker 뒤**: 발별
    - psi_td = 마지막 착지(접촉 0→1) 순간 골반 요(에피소드 시작 = 그때 요), psi_ref = 이륙 순간 고정한 psi_td(= 이번 스윙의 측정 프레임),
      psi_lo = 이륙 순간 골반 요(변형 비교용).
    - td_d = 유효 착지(_st.ev_td) 사건의 이륙→착지 **월드** 변위(N,2,2), 그 외 0.
    - 혼합 명령(활성 축 ≥ 2: |vx|·|vy| ≥ cmin, |wz| ≥ wcmin) STEP 착지의 축별 부호 정답: u = R(−θ/2)·(Δp_ref − 지레 항)
      의 x·y 부호 vs vx·vy(θ = **측정** 착지 발 요 변화 _w3turn.td_dpsi — 회전 오류와 병진 부호를 분리), 발 요 Δψ 부호 vs wz. 폭 위반은 _w3lat.width_bad. 상태 env._w3mix(기존 키 불변)."""
    s = _st(env); M = _mix(env)
    asset = env.scene[H._base_cfg().name]
    psi = _yaw_of(asset.data.root_quat_w)
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    c = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > threshold).float()
    fresh = env.episode_length_buf <= 1
    if not M["init"]:
        M["psi_td"] = psi.unsqueeze(1).repeat(1, 2).clone(); M["psi_ref"] = M["psi_td"].clone(); M["psi_lo"] = M["psi_td"].clone(); M["init"] = True
    if bool(fresh.any()):
        p2 = psi[fresh].unsqueeze(1).expand(-1, 2)
        M["psi_td"][fresh] = p2; M["psi_ref"][fresh] = p2; M["psi_lo"][fresh] = p2; M["c_prev"][fresh] = 1.0
    lifted = s["ev_takeoff"] > 0.5
    M["psi_ref"] = torch.where(lifted, M["psi_td"], M["psi_ref"])
    M["psi_lo"] = torch.where(lifted, psi.unsqueeze(1).expand(-1, 2), M["psi_lo"])
    td = s["ev_td"] > 0.5
    bid = H._g21_bids(env); fxy = asset.data.body_pos_w[:, bid, :2]
    d = fxy - s["lift_xy"]
    M["td_d"] = torch.where(td.unsqueeze(-1), d, torch.zeros_like(d))
    landed = (M["c_prev"] < 0.5) & (c > 0.5) & ~fresh.unsqueeze(1)
    # 축별 부호(혼합 명령 STEP 착지만) — psi_td 갱신 전(이번 스윙의 측정 프레임 = psi_ref)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    ax = cmd[:, 0].abs() >= cmin; ay = cmd[:, 1].abs() >= cmin; aw = cmd[:, 2].abs() >= wcmin
    mixd = (ax.float() + ay.float() + aw.float()) >= 2.0
    tdm = td & (mixd & _ge(env)).unsqueeze(1)
    if bool(tdm.any()):
        dl = _rot2(-M["psi_ref"], M["td_d"])
        T_ = getattr(env, "_w3turn", None)
        # 지레·현 회전 보정은 **측정** 발 요 변화로(회전이 틀려도 병진 부호를 따로 판정 — 명령 θ 로 보정하면 회전 반대 착지의 지레 오차 2w·sinθ 가 x 부호를 뒤집음)
        th = T_["td_dpsi"] if T_ is not None else (cmd[:, 2] / f_.clamp(min=0.5)).unsqueeze(1).expand(-1, 2)
        p = torch.tensor([[0.0, nominal_half_width], [0.0, -nominal_half_width]], device=dl.device, dtype=dl.dtype).unsqueeze(0)
        u = _rot2(-0.5 * th, dl - (_rot2(th, p) - p))
        tx = tdm & ax.unsqueeze(1); ty = tdm & ay.unsqueeze(1); tw = tdm & aw.unsqueeze(1)
        M["n_td"] += tdm.float().sum()
        M["n_x"] += tx.float().sum(); M["ok_x"] += (tx & (u[..., 0] * torch.sign(cmd[:, 0]).unsqueeze(1) > 0)).float().sum()
        M["n_y"] += ty.float().sum(); M["ok_y"] += (ty & (u[..., 1] * torch.sign(cmd[:, 1]).unsqueeze(1) > 0)).float().sum()
        if T_ is not None:
            M["n_w"] += tw.float().sum(); M["ok_w"] += (tw & (T_["td_dpsi"] * torch.sign(cmd[:, 2]).unsqueeze(1) > 0)).float().sum()
        L = getattr(env, "_w3lat", None)
        if L is not None:
            M["n_bad"] += (L["width_bad"] * tdm.float()).sum()
    M["psi_td"] = torch.where(landed, psi.unsqueeze(1).expand(-1, 2), M["psi_td"])
    M["c_prev"] = c


def w3_stride4(env, sig: float = 0.03, cmin: float = 0.02, wcmin: float = 0.08, width_gate: bool = True, k_opp: float = 0.5,
               nominal_half_width: float = 0.144, frame: str = "td", rot_frac: float = 0.5, lever: bool = True, sinc: bool = True) -> torch.Tensor:
    """Stage 4: 회전 + 병진 명령(|wz| ≥ wcmin 이고 |vx| 또는 |vy| ≥ cmin)이면 곡선 착지 목표(w3_curve_target; 측정 프레임 frame="td" = 직전 착지 골반 요,
    "lo" = 이륙 순간 골반 요)로 w3_stride2 와 같은 식: 폭 게이트·[exp(−‖Δp − ΔP*‖²/σ²) − exp(−‖ΔP*‖²/σ²)] + k_opp·clip(Δp·ΔP*/‖ΔP*‖², −1, 0).
    그 밖(단일 축·제자리 회전·명령 0)은 **w3_stride2 값 그대로**(Stage 1–3 비트 동일)."""
    base = w3_stride2(env, sig=sig, cmin=cmin, width_gate=width_gate, k_opp=k_opp)
    M = getattr(env, "_w3mix", None)
    if M is None:
        return base
    s = _st(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    T = w3_curve_target(cmd, f_, nominal_half_width, rot_frac=rot_frac, lever=lever, sinc=sinc)
    yf = M["psi_ref"] if frame == "td" else M["psi_lo"]
    dl = _rot2(-yf, M["td_d"])
    e2 = ((dl - T) ** 2).sum(-1); n2 = (T ** 2).sum(-1)
    r = (torch.exp(-e2 / sig ** 2) - torch.exp(-n2 / sig ** 2)) * s["ev_td"]
    L = getattr(env, "_w3lat", None)
    if width_gate and L is not None:
        r = r * (1.0 - L["width_bad"])
    r = r + k_opp * ((dl * T).sum(-1) / n2.clamp(min=1e-8)).clamp(-1.0, 0.0) * s["ev_td"]
    cur = r.sum(1) * _ge(env).float()
    on = (cmd[:, 2].abs() >= wcmin) & ((cmd[:, 0].abs() >= cmin) | (cmd[:, 1].abs() >= cmin))
    return torch.where(on, cur, base)


def _v_err_curve(env):
    """회전 중 병진 추종 오차: 한 주기 이동 평균 속도(_v_box) vs 명령 (vx, vy) 를 **같은 창의 실제 몸 요 경로**로 적분한 평균
    R(ψ_mid)·sinc(Δψ/2)·(vx, vy), ψ_mid = ψ − Δψ/2, Δψ = 창 동안 순 요 변화(_yaw_box × 창 길이). 현재 요로 돌린 명령(_v_err)은
    곡선에서 반 창만큼 늦은 방향을 비교해 |v|·Δψ/2 수준의 체계 오차를 만든다(단위시험 [C])."""
    asset = env.scene[H._base_cfg().name]
    psi = _yaw_of(asset.data.root_quat_w)
    vb = _v_box(env); wb = _yaw_box(env)
    K = max(2, int(round((1.0 / 1.1) / env.step_dt)))
    n = env.episode_length_buf.clamp(min=1, max=K).float()
    dpsi = wb * n * env.step_dt
    cmd = env.command_manager.get_command("base_velocity")
    cw = _rot2(psi - 0.5 * dpsi, cmd[:, :2] * _sinc(0.5 * dpsi).unsqueeze(1))
    return (vb - cw).norm(dim=-1), cmd[:, :2].norm(dim=-1)


def w3_track_rel4(env, sig: float = 0.05, cmin: float = 0.02, wcmin: float = 0.08) -> torch.Tensor:
    """Stage 4: 회전 + 병진 명령(|wz| ≥ wcmin 이고 |vx| 또는 |vy| ≥ cmin)이면 곡선 보정 추종 개선분(_v_err_curve),
    그 밖은 **w3_track_rel 값 그대로**(비트 동일). 병진 추종은 실제 요 경로 기준이라 회전과 분리된다(회전은 w3_yaw_net·w3_ytrack·w3_turn 이 채점)."""
    base = w3_track_rel(env, sig=sig, cmin=cmin)
    e, cn = _v_err_curve(env)
    cur = (torch.exp(-(e / sig) ** 2) - torch.exp(-(cn / sig) ** 2)) * (cn >= cmin).float() * _ge(env).float()
    c = env.command_manager.get_command("base_velocity")
    on = (c[:, 2].abs() >= wcmin) & ((c[:, 0].abs() >= cmin) | (c[:, 1].abs() >= cmin))
    return torch.where(on, cur, base)


def w3_drift_net4(env, cmin: float = 0.02, wcmin: float = 0.08) -> torch.Tensor:
    """Stage 4: 회전 + 병진 명령(|wz| ≥ wcmin 이고 |vx| 또는 |vy| ≥ cmin)이면 곡선 보정 순 표류 |v_T − 곡선 명령 평균|,
    그 밖은 **w3_drift_net 값 그대로**(비트 동일)."""
    base = w3_drift_net(env)
    e, _ = _v_err_curve(env)
    c = env.command_manager.get_command("base_velocity")
    on = (c[:, 2].abs() >= wcmin) & ((c[:, 0].abs() >= cmin) | (c[:, 1].abs() >= cmin))
    return torch.where(on, e * _ge(env).float(), base)


def w3_mix_curriculum(env, env_ids, scales: tuple = (0.0, 0.5, 1.0), min_sign: float = 0.90, max_bad: float = 0.02, min_iters: int = 100,
                      min_mix_td: float = 3000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """혼합 크기 커리큘럼(env._w3_mix_scale 0 → 0.5 → 1: 각 축 크기 = lo + s·U·(hi − lo)). 블록(혼합 착지 ≥ min_mix_td) 판정 —
    활성 축마다 부호 정답(지레·회전 보정 뒤 x·y, 발 요) ≥ min_sign 이고 폭 위반률 < max_bad, 체류 ≥ min_iters 면 다음 단계. 절대 문턱(톱니 금지)."""
    M = _mix(env); g = getattr(env, "_w3_mixcur", None)
    keys = ("n_td", "n_x", "ok_x", "n_y", "ok_y", "n_w", "ok_w", "n_bad")
    if g is None:
        g = {"stage": int(start_stage), "since": 0, "b": {k: 0.0 for k in keys}, "sx": float("nan"), "sy": float("nan"), "sw": float("nan"), "bad": float("nan")}
        env._w3_mixcur = g
        env._w3_mix_scale = float(scales[g["stage"]])
    cur = {k: float(M[k].sum()) for k in keys}
    it = int(env.common_step_counter) // steps_per_iter
    dd = {k: cur[k] - g["b"][k] for k in keys}
    if dd["n_td"] >= min_mix_td:
        rate = lambda ok, nn: dd[ok] / dd[nn] if dd[nn] > 0 else float("nan")
        g["sx"], g["sy"], g["sw"] = rate("ok_x", "n_x"), rate("ok_y", "n_y"), rate("ok_w", "n_w")
        g["bad"] = dd["n_bad"] / max(dd["n_td"], 1.0)
        vals = [x for x in (g["sx"], g["sy"], g["sw"]) if x == x]
        adv = bool(vals) and min(vals) >= min_sign and g["bad"] < max_bad and it - g["since"] >= min_iters and g["stage"] < len(scales) - 1
        print(f"[W3S4 혼합 블록] 반복 {it} 크기 s={scales[g['stage']]}: 부호 x {g['sx']:.3f} · y {g['sy']:.3f} · 요 {g['sw']:.3f} · 폭 위반 {g['bad']:.3f}"
              + (" → 크기 확대" if adv else ""), flush=True)
        if adv:
            g["stage"] += 1; g["since"] = it; env._w3_mix_scale = float(scales[g["stage"]])
        g["b"] = cur
    return {"mix_scale": float(env._w3_mix_scale), "mix_sign_x": g["sx"], "mix_sign_y": g["sy"], "mix_sign_w": g["sw"], "mix_width_bad": g["bad"]}


# ================================================================ W3S3t* (1007) — 걷기 토크 예산. 기본 꺼짐(W3S3tK 설정을 고를 때만) ================================================================
# 설계·사전 기준: reports/gen_w3/TORQUE_W3T_DESIGN.md · CPU 단위시험: rl_walking/scripts/unit_w3t.py. 기존 함수·상태(env._w3·_w3lat·_w3turn·_w3mix)는 손대지 않는다.
def w3_stance_knee_over(env, max_flex_rad: float = 0.785, threshold: float = 5.0, cap_rad: float = 0.6) -> torch.Tensor:
    """W3S3tK: 걷는 동안(ge 1) **접지한 다리**의 무릎 굽힘이 max_flex_rad 를 넘는 초과분[rad] — 접지 다리 평균, 선형(상한 cap_rad). 벌점 크기(양수).
    무릎 굽힘 = 물리 관절각(왼 무릎 −·오른 무릎 + 가 굽힘 — mdp_stage4.knee_stance_flexion 과 같은 부호 보정). 공중(스윙) 다리는 자유, 두 발 다 공중이면 0.
    선형인 이유: 현 걸음의 지지 무릎은 64°(문턱 45° 에서 19° 밖)라, 밴드형(knee_stance_flexion: 밴드 밖 0.15 rad 에서 0 으로 포화)은 기울기가 없다.
    근거(reports/gen_w3/torque_phase.py bounds (B)): 한발 지지 무릎 정적 토크는 굽힘에 거의 비례 — 65° 21.3 → 45° 15.6 → 30° 11.0 N·m."""
    asset = env.scene[H._base_cfg().name]
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    c = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > threshold).float()                     # (N,2) [왼, 오]
    if getattr(env, "_w3_kneeidx", None) is None:
        jn = list(asset.joint_names)
        env._w3_kneeidx = torch.tensor([jn.index("left_knee_joint"), jn.index("right_knee_joint")], device=env.device)
    q = asset.data.joint_pos[:, env._w3_kneeidx]
    flex = torch.stack([-q[:, 0], q[:, 1]], dim=1)                                                 # 굽힘 + 로 통일
    over = (flex - max_flex_rad).clamp(min=0.0, max=cap_rad)
    val = (over * c).sum(1) / c.sum(1).clamp(min=1.0)
    return val * _ge(env).float()


def w3_stride_lin(env, cmin: float = 0.02) -> torch.Tensor:
    """W3S3tF: 유효 착지 사건(_st.ev_td)마다 스윙발 이륙→착지 변위 Δp 의 **목표 보폭 방향 진행 비율** clip(Δp·ΔP*/‖ΔP*‖², −1, 1), ΔP* = (vx/f, vy/f).
    제자리 디딤 = 0, 목표 보폭 이상 = 1(더 가도 늘지 않음 — 과속 유인 없음), 반대 방향 = 음수. 병진 명령(|vx| 또는 |vy| ≥ cmin)·STEP 에서만, 그 밖(명령 0·제자리 회전·서기)은 정확히 0.
    w3_stride2 의 반대 방향 벌점 항(k_opp·clip(·, −1, 0))을 양수 쪽까지 편 것과 같은 식이다. 변위는 w3_lat_tracker 의 _w3lat.td_dx·td_dy(스윙발 자체의 월드 변위 — 몸 기울임·지지발 미끄럼으로는 못 얻는다).
    선형인 이유(reports/gen_w3/fwd_incentive.py): 가우시안형 w3_stride(σ 0.03 m)·w3_track(σ 0.05 m/s)은 env 마다 다른 속도 치우침(학습 조건 한 주기 평균 오차 0.062 m/s ≈ 보폭 0.056 m)이
    폭보다 크면 평평해져, 전진을 절반으로 줄여도 잃는 보상이 0.004–0.038/s 뿐이다. 선형은 같은 조건에서 0.08–0.14/s 를 유지한다."""
    s = _st(env); L = _lat(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    fc = f_.clamp(min=0.5)
    tx = (cmd[:, 0] / fc).unsqueeze(1); ty = (cmd[:, 1] / fc).unsqueeze(1)
    n2 = (tx ** 2 + ty ** 2).clamp(min=1e-8)
    r = ((L["td_dx"] * tx + L["td_dy"] * ty) / n2).clamp(-1.0, 1.0) * s["ev_td"]
    on = ((cmd[:, 0].abs() >= cmin) | (cmd[:, 1].abs() >= cmin)) & _ge(env)
    return torch.where(on, r.sum(1), torch.zeros_like(cmd[:, 0]))
