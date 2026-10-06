"""W3D (1007) — 걷는 중 외란 → 회복 스텝 → 운동량 억제 → 정상 WALK/STAND 복귀를 **같은 W3 스텝 엔진** 위에서 학습하는 설정의 코어.

설계·사전 통과 기준: reports/gen_w3/WD_DESIGN.md. CPU 반사실 단위시험: rl_walking/scripts/unit_w3d.py.
기본 꺼짐 — 이 모듈은 W3D* 환경 설정에서만 쓰인다(기존 W3/c2 함수·상태는 손대지 않음: env._w3·_w3lat·_w3turn·_c2 키 불변).

구성
  1. 외란 스케줄(reset, ev.disturbance 자리 — 평가기 eval_w3/eval_ground_recovery 가 'disturbance' 를 None 으로 지우면 같이 꺼진다)
     - 에피소드 p_dist 에 1–2 회. 시각 t0 ~ U(t0_range)(보행 주기 0.909 s 보다 훨씬 넓음 → 보행 위상 균일 분산), 다음은 앞 종료 + gap.
     - 모양: 충격(p_imp, 지속 0.2–0.4 s) 또는 짧은 램프(상승 0.3–1.0 s·유지 0.2–1.0 s). 방향 = **발생 순간 몸통** 앞/뒤/왼/오른(dir_p).
     - 크기 커리큘럼: 충격 F ~ U(levels[lvl]) N, 램프 F ~ U(ramp_lo, ramp_frac·상한). lvl 은 w3d_curriculum 이 올린다(절대 문턱, 톱니 금지).
     - **걷는 env 만**: 발생 직전 틱에 gait_enable 이 꺼져 있으면(STAND) 그 외란은 취소(서기 외란은 KE/G21a 경로의 몫).
     - 정지 시나리오(p_stop): 외란 끝 + U(stop_delay) 에 정지 명령(_ge_cmd 0·명령 0, 에피소드 끝까지) → STAND 복귀.
       (평가 eval_w3 --dist ';stop' 과 같은 계약. 실물: "밀리면 stop" 이 걷기 유지보다 위험했던 기준선 → 학습 분포에 포함)
  2. 회복 창(충격만): 충격 시작 ~ +win_s 동안 유효 착지(_st.ev_td) 최대 max_td 개 = '회복 착지'.
     점수 = 포획 차분형 v2(w3_c2.capture_diff_score — 직전 양발 지지영역 대비 ξ_eq 거리 감소, clip 0–1; c2 에서 기준 2 통과한 식 그대로).
  3. 보상(전부 가산 — 기존 걷기 보상 항은 그대로 둔다 = 외란 구간에도 걷기 보상 유지):
     - w3d_cap   = Σ 회복 착지 cap_df                                         (방향 회복 스텝 > 원래 보폭 > 반대)
     - w3d_rplace = Σ 회복 착지 max(0, gain − place_nom)                      (place 가중과 같은 가중 → 회복 착지의 착지 항 = max(명목, gain))
     - w3d_rstride = Σ 회복 착지 max(0, gain·s_max − stride_nom)              (stride 가중과 같은 가중 → 보폭 항 = max(명목, gain·명목 최대점))
       gain = clip((Φ_명목착지 − Φ_실제)/1 cm, 0, 1): 같은 순간 W3 명목 착지 목표에 디뎠을 때보다 실제 착지가 더 포획했는가.
       → 명목보다 포획을 개선한 회복 착지는 착지·보폭 점수를 잃지 않는다('회복 쪽이 손해 보지 않게'). 명목·반대 착지는 gain 0 → 보전 0.
     - w3d_ret  (1 회/외란): 외란 끝 + 0.3 s 뒤 (외란 전 같은 env 기준 대비) 주기 평균 속도 추종 오차 ≤ 전 + 0.03·기울기 ≤ 전 + 2°·|ω_xy| ≤ 1.5·전 + 0.05
       (eval_w3 WALK 판정과 같은 꼴)가 ret_hold 유지 →
       max(0, 1 − (t − t_on)/ret_T). 정지 시나리오 env 는 대상 아님.
     - w3d_sret (1 회/외란, 정지 시나리오): 래치(ge 0) 뒤 양발·기울기 < 3°·|ω_xy| < 0.2·|v| < 0.08 이 0.5 s 유지 → max(0, 1 − (t − t_stop)/sret_T).
     외란이 없거나 걷지 않는 env 는 새 항이 정확히 0 → 무외란 env 의 보상·관측·명령은 W3S3d 와 같다(p_dist 0 이면 난수 소비도 없음 = 학습 전체 비트 동일).
  4. 숨기 유인: 모드 전환(ge)은 명령(시간 규칙)만 바꾼다 — 정책 상태로 회복 창·정지 시점을 열고 닫을 수 없다(c2 의 상태 조건 정지 요청이 만든 유인 제거).
     얼어붙기(스텝 중단)는 기존 놓친 반주기 −10·주기 접촉 벌점이 그대로 막는다(단위시험 [B]).

배포 계약: actor 관측(320)·critic(335)·행동 처리 불변 → W3S3d 체크포인트에서 --resume. 배포 러너 변경 없음(보상·외란은 학습 전용).
"""
from __future__ import annotations

import math
import os

import torch

from . import mdp_hang as H
from . import w3 as W3
from . import w3_c2 as C2

DIR_B = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))     # f, b, l, r (몸통 프레임)
DIR_NAMES = ("f", "b", "l", "r")
LEVELS = ((6.0, 10.0), (6.0, 14.0), (6.0, 18.0), (6.0, 22.0))  # 충격 F 범위 [N] (지속 0.2–0.4 s → 1.2–8.8 N·s). 평가 곡선 8–18 N·0.3 s 를 L2 까지 덮음


# ---------------------------------------------------------------- 상태 ----------------------------------------------------------------
def _st(env):
    s = getattr(env, "_w3d", None)
    K = H._DST_K
    if s is None or s["w_on"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        z = lambda *sh: torch.zeros(*sh, device=dev)
        b = lambda *sh: torch.zeros(*sh, dtype=torch.bool, device=dev)
        s = {"dir_b": z(N, K, 2), "dir_i": torch.full((N, K), -1, dtype=torch.long, device=dev), "cancel": b(N, K), "applied": b(N, K),
             "stop_want": b(N), "t_stop": torch.full((N,), float("inf"), device=dev), "stop_on": b(N), "t_stop_on": torch.full((N,), float("inf"), device=dev),
             # 회복 창(충격)
             "w_on": b(N), "w_t0": z(N), "w_ntd": z(N), "w_F": z(N, 2), "ph_on": torch.full((N,), -1.0, device=dev),
             # 착지 사건(발별): 회복 착지 여부·차분형 점수·Φ 전/후
             "td_rec": b(N, 2), "cap_df": z(N, 2), "gain": z(N, 2), "phi_b": z(N, 2), "phi_a": z(N, 2), "last_xy": None,
             # 복귀(외란 단위)
             "r_on": b(N), "r_t0": z(N), "r_tend": z(N), "r_run": z(N), "r_done": b(N), "ev_ret": z(N),
             "s_run": z(N), "s_done": b(N), "ev_sret": z(N), "pre": z(N, 3), "pre_n": z(N), "pre_f": z(N, 3), "post": z(N, 3), "spost": z(N, 3),
             # 에피소드 기록(커리큘럼)
             "ep_imp": b(N), "ep_lvl": torch.zeros(N, dtype=torch.long, device=dev),
             "lvl": 0, "stats": {k: 0.0 for k in ("imp_ep", "imp_fall", "applied", "canceled", "rec_td", "cap_sum", "ret_paid", "ret_due",
                                                   "sret_paid", "sret_due", "stop_issued")},
             "lvl_stats": {}}
        env._w3d = s
    return s


def _yaw(env):
    return W3._yaw_of(env.scene[H._base_cfg().name].data.root_quat_w)


def _rot(yaw, v):
    """(N,) 요로 (N,K,2) 또는 (N,2) 회전."""
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < v.dim() - 1:
        c = c.unsqueeze(-1); s_ = s_.unsqueeze(-1)
    return torch.stack([c * v[..., 0] - s_ * v[..., 1], s_ * v[..., 0] + c * v[..., 1]], -1)


# ---------------------------------------------------------------- 외란 스케줄(reset) ----------------------------------------------------------------
def w3d_dist_reset(env, env_ids, p_dist: float = 0.5, n_range: tuple = (1, 2), t0_range: tuple = (3.0, 15.0), gap_range: tuple = (3.0, 5.0),
                   p_imp: float = 0.7, imp_dur: tuple = (0.2, 0.4), ramp_rise: tuple = (0.3, 1.0), ramp_hold: tuple = (0.2, 1.0),
                   ramp_lo: float = 3.0, ramp_frac: float = 0.6, dir_p: tuple = (0.35, 0.25, 0.2, 0.2), levels: tuple = LEVELS,
                   p_stop: float = 0.3, stop_delay: tuple = (0.0, 1.0)):
    """mode="reset": 걷기 외란 스케줄. p_dist ≤ 0 이면 아무것도 하지 않는다(난수 소비 0 — W3S3d 와 학습 전체 비트 동일)."""
    if p_dist <= 0.0:
        return
    s = _st(env); hs = H._state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)
    n = ids.numel()
    if n == 0:
        return
    dev = env.device; K = H._DST_K
    # 지난 에피소드 집계(충격이 실제로 걸린 env 의 종료 = 낙상인가) — 커리큘럼 입력
    tm = getattr(env, "termination_manager", None)
    fell = tm.terminated[ids].bool() if (tm is not None and hasattr(tm, "terminated")) else torch.zeros(n, dtype=torch.bool, device=dev)
    had = s["ep_imp"][ids]
    S = s["stats"]
    S["imp_ep"] += float(had.sum()); S["imp_fall"] += float((had & fell).sum())
    for lv in s["ep_lvl"][ids][had].unique().tolist():
        m = had & (s["ep_lvl"][ids] == lv)
        d = s["lvl_stats"].setdefault(int(lv), [0.0, 0.0]); d[0] += float(m.sum()); d[1] += float((m & fell).sum())
    # 새 스케줄
    U = lambda lo, hi, *sh: torch.empty(*sh, device=dev).uniform_(float(lo), float(hi))
    act = torch.rand(n, device=dev) < float(p_dist)
    cnt = torch.randint(int(n_range[0]), int(n_range[1]) + 1, (n,), device=dev)
    slot = torch.arange(K, device=dev).unsqueeze(0)
    on = (slot < cnt.unsqueeze(1)) & act.unsqueeze(1)
    imp = torch.rand(n, K, device=dev) < float(p_imp)
    lo, hi = levels[min(int(s["lvl"]), len(levels) - 1)]
    F = torch.where(imp, U(lo, hi, n, K), U(ramp_lo, max(ramp_lo, ramp_frac * hi), n, K))
    rise = U(*ramp_rise, n, K)
    hold = torch.where(imp, U(*imp_dur, n, K), U(*ramp_hold, n, K))
    dur = torch.where(imp, hold, rise + hold)
    t0 = torch.zeros(n, K, device=dev); t0[:, 0] = U(*t0_range, n)
    for k in range(1, K):
        t0[:, k] = t0[:, k - 1] + dur[:, k - 1] + U(*gap_range, n)
    cp = torch.tensor(dir_p, device=dev, dtype=torch.float32).cumsum(0); cp = cp / cp[-1]
    di = torch.searchsorted(cp, torch.rand(n, K, device=dev).contiguous(), right=True).clamp(max=3)
    db = torch.tensor(DIR_B, device=dev)[di]
    stop = act & (torch.rand(n, device=dev) < float(p_stop))
    t_stop = t0[:, 0] + dur[:, 0] + U(*stop_delay, n)
    kind = torch.where(on, torch.where(imp, torch.full_like(F, 2.0), torch.ones_like(F)), torch.zeros_like(F))
    hs["dst_kind"][ids] = kind; hs["dst_t0"][ids] = t0; hs["dst_rise"][ids] = torch.where(imp, torch.ones_like(rise), rise)
    hs["dst_hold"][ids] = hold; hs["dst_F"][ids] = torch.where(on, F, torch.zeros_like(F)); hs["dst_dir"][ids] = 0.0
    hs["dst_F_now"][ids] = 0.0; hs["dst_last_end"][ids] = -1.0; hs["dst_pending"][ids] = 0.0; hs["dst_quiet_run"][ids] = 0.0; hs["dst_bonus"][ids] = 0.0
    s["dir_b"][ids] = db; s["dir_i"][ids] = torch.where(on, di, torch.full_like(di, -1))
    s["cancel"][ids] = False; s["applied"][ids] = False
    s["stop_want"][ids] = stop; s["t_stop"][ids] = torch.where(stop, t_stop, torch.full_like(t_stop, float("inf")))
    s["stop_on"][ids] = False; s["t_stop_on"][ids] = float("inf")
    for k_ in ("w_on", "r_on", "r_done", "s_done", "ep_imp"):
        s[k_][ids] = False
    for k_ in ("w_t0", "w_ntd", "r_t0", "r_tend", "r_run", "s_run", "ev_ret", "ev_sret", "pre_n"):
        s[k_][ids] = 0.0
    s["pre"][ids] = 0.0; s["pre_f"][ids] = 0.0; s["spost"][ids] = 0.0
    s["w_F"][ids] = 0.0; s["ph_on"][ids] = -1.0
    s["td_rec"][ids] = False; s["cap_df"][ids] = 0.0; s["gain"][ids] = 0.0; s["phi_b"][ids] = 0.0; s["phi_a"][ids] = 0.0
    s["ep_lvl"][ids] = int(s["lvl"])


# ---------------------------------------------------------------- 정지 시나리오 + W3 엔진(interval, w3_gait 자리) ----------------------------------------------------------------
def w3d_gait_update(env, env_ids, f_step: float = 1.1, ramp_s: float = 0.2, lead_vy_cmin: float = 0.0, lead_wz_cmin: float = 0.0):
    """mode="interval"(0 s): **w3_gait_update 대체**(같은 자리). 정지 시나리오 env(실제 외란이 걸렸고 정지 시각 지남)만 _ge_cmd 0·명령 0 으로 덮은 뒤
    W3 엔진(w3_gait_update, 반주기 경계 래치 그대로)을 부른다. 명령 리샘플(command_manager, 이벤트 전)이 다시 켜도 같은 틱 엔진 전에 덮인다.
    정지 대상이 없으면 w3_gait_update 와 같다(비트 동일)."""
    s = getattr(env, "_w3d", None)
    if s is not None and bool(s["stop_want"].any()):
        hs = H._state(env); t = hs["t"]
        go = s["stop_want"] & s["applied"].any(1) & (t >= s["t_stop"])
        if bool(go.any()):
            first = go & ~s["stop_on"]
            s["t_stop_on"] = torch.where(first, t, s["t_stop_on"]); s["stats"]["stop_issued"] += float(first.sum())
            s["stop_on"] = s["stop_on"] | go
            if getattr(env, "_ge_cmd", None) is not None:
                env._ge_cmd = torch.where(go, torch.zeros_like(env._ge_cmd), env._ge_cmd)
            term = env.command_manager.get_term("base_velocity")
            term.vel_command_b[go] = 0.0
            if getattr(env, "_w3_wz0", None) is not None:
                env._w3_wz0[go] = 0.0
    W3.w3_gait_update(env, env_ids, f_step=f_step, ramp_s=ramp_s, lead_vy_cmin=lead_vy_cmin, lead_wz_cmin=lead_wz_cmin)


# ---------------------------------------------------------------- 틱 갱신(interval) ----------------------------------------------------------------
def w3d_dist_step(env, env_ids, win_s: float = 1.2, max_td: int = 3, ret_after: float = 0.3, ret_tol_v: float = 0.03, ret_tilt: float = 2.0,
                  ret_w: float = 0.05, pre_tau: float = 2.0, ret_hold: float = 0.5, ret_T: float = 4.0, sret_tilt: float = 4.0, sret_w: float = 0.6, sret_v: float = 0.2,
                  sret_hold: float = 0.5, sret_T: float = 4.0):
    """mode="interval"(0 s), **w3_gait·w3_track 뒤**(정의 순서): ① 발생 전 슬롯 방향을 현재 몸통 요로 갱신(발생 틱엔 직전 틱 방향으로 고정)
    ② 다음 틱 발생인데 STAND 면 취소 ③ 발생 기록(충격 → 회복 창) ④ 복귀 판정(1 회/외란 보상 사건). 정지 명령은 w3d_gait_update 가 건다."""
    s = _st(env); hs = H._state(env)
    if not bool((hs["dst_kind"] > 0).any()) and not bool(s["r_on"].any()):
        s["ev_ret"].zero_(); s["ev_sret"].zero_()
        return
    dev = env.device; dt = env.step_dt
    t = hs["t"]; tt = t.unsqueeze(1)
    ge = W3._ge(env)
    kind = hs["dst_kind"]
    pre = (kind > 0) & (tt < hs["dst_t0"])
    yaw = _yaw(env)
    hs["dst_dir"] = torch.where(pre.unsqueeze(-1), _rot(yaw, s["dir_b"]), hs["dst_dir"])
    nxt = pre & (hs["dst_t0"] <= tt + dt + 1e-6)
    cx = nxt & ~ge.unsqueeze(1)
    if bool(cx.any()):
        hs["dst_kind"] = torch.where(cx, torch.zeros_like(kind), kind); hs["dst_F"] = torch.where(cx, torch.zeros_like(hs["dst_F"]), hs["dst_F"])
        s["cancel"] |= cx; s["stats"]["canceled"] += float(cx.sum())
    kind = hs["dst_kind"]
    new = (kind > 0) & (tt >= hs["dst_t0"]) & ~s["applied"]
    S = s["stats"]
    if bool(new.any()):
        s["applied"] |= new; S["applied"] += float(new.sum())
        newi = new & (kind == 2)
        ni = newi.any(1)
        k_first = torch.argmax(new.float(), dim=1)
        ar = torch.arange(env.num_envs, device=dev)
        t0k = hs["dst_t0"][ar, k_first]
        tend = t0k + torch.where(kind[ar, k_first] == 1, hs["dst_rise"][ar, k_first] + hs["dst_hold"][ar, k_first], hs["dst_hold"][ar, k_first])
        nn = new.any(1)
        s["w_on"] = torch.where(ni, torch.ones_like(s["w_on"]), s["w_on"] & ~nn)          # 램프가 새로 오면 이전 충격 창도 닫는다
        s["w_t0"] = torch.where(ni, t0k, s["w_t0"]); s["w_ntd"] = torch.where(ni, torch.zeros_like(s["w_ntd"]), s["w_ntd"])
        s["w_F"] = torch.where(ni.unsqueeze(1), (hs["dst_F"][ar, k_first].unsqueeze(1) * hs["dst_dir"][ar, k_first]), s["w_F"])
        gu = getattr(env, "_gu", None)
        if gu is not None:
            s["ph_on"] = torch.where(nn, gu, s["ph_on"])
        s["ep_imp"] |= ni
        # 복귀 판정 재무장(외란 단위)
        s["r_on"] = s["r_on"] | nn; s["r_t0"] = torch.where(nn, t0k, s["r_t0"]); s["r_tend"] = torch.where(nn, tend, s["r_tend"])
        s["r_run"] = torch.where(nn, torch.zeros_like(s["r_run"]), s["r_run"]); s["r_done"] = s["r_done"] & ~nn
        s["s_run"] = torch.where(nn, torch.zeros_like(s["s_run"]), s["s_run"]); s["s_done"] = s["s_done"] & ~nn
        s["pre_f"] = torch.where(nn.unsqueeze(1), s["pre"], s["pre_f"])
    # 회복 창 닫기(시간·착지 수)
    s["w_on"] = s["w_on"] & (t <= s["w_t0"] + win_s) & (s["w_ntd"] < max_td)
    # 복귀 판정(보상 전용 특권량)
    asset = env.scene[H._base_cfg().name]
    pg = asset.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1.0, 1.0)))
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    s["ev_ret"] = torch.zeros_like(s["ev_ret"]); s["ev_sret"] = torch.zeros_like(s["ev_sret"])
    # 외란 전 같은 env 의 걷기 기준값(EMA τ pre_tau, STEP·외력 없음·복귀 판정 밖에서만 갱신) — 1007 스모크: 절대 문턱(4°·0.5 rad/s)은
    # 학습 DR 걷기 자체(기울기 중앙 8.6°·|ω| 1.09)가 못 넘어 지급 0/183 → eval_w3 WALK 판정과 같은 '외란 전 대비' 규칙으로 교체
    ev_, _ = W3._v_err(env, box=True)
    cur = torch.stack([ev_, tilt, w], 1)
    quiet = ge & (hs["dst_F_now"] <= 0.0) & ~(s["r_on"] & ~s["r_done"])
    a_ = min(1.0, dt / max(pre_tau, 1e-6))
    s["pre"] = torch.where(quiet.unsqueeze(1), torch.where((s["pre_n"] < 0.5).unsqueeze(1), cur, s["pre"] + a_ * (cur - s["pre"])), s["pre"])
    s["pre_n"] = torch.where(quiet, s["pre_n"] + 1.0, s["pre_n"])
    # 판정 값 = 한 주기(τ 0.91 s) EMA — 순간값은 걸음 흔들림이 지배(1007 스모크 v1: 순간값 판정 지급 2/129)
    b_ = min(1.0, dt / 0.91)
    s["post"] = torch.where((env.episode_length_buf <= 1).unsqueeze(1), cur, s["post"] + b_ * (cur - s["post"]))
    walk = s["r_on"] & ~s["r_done"] & ~s["stop_want"]
    if bool(walk.any()):
        late = t >= s["r_tend"] + ret_after
        pf = s["pre_f"]
        po = s["post"]
        ok = walk & late & ge & (po[:, 0] <= pf[:, 0] + ret_tol_v) & (po[:, 1] <= pf[:, 1] + ret_tilt) & (po[:, 2] <= 1.5 * pf[:, 2] + ret_w)
        s["r_run"] = torch.where(ok, s["r_run"] + dt, torch.zeros_like(s["r_run"]))
        pay = walk & (s["r_run"] >= ret_hold)
        tout = walk & ~pay & (t - s["r_t0"] > ret_T)
        s["ev_ret"] = torch.where(pay, (1.0 - (t - s["r_t0"]) / ret_T).clamp(min=0.0), s["ev_ret"])
        S["ret_paid"] += float(pay.sum()); S["ret_due"] += float((pay | tout).sum())
        s["r_done"] = s["r_done"] | pay | tout
    sw = s["r_on"] & s["stop_on"] & ~s["s_done"]
    if bool(sw.any()):
        sensor = env.scene.sensors["contact_forces"]
        both = (sensor.data.net_forces_w[:, H._feet_body_ids(env), :].norm(dim=-1) > 5.0).all(dim=1)
        v = asset.data.root_lin_vel_w[:, :2].norm(dim=1)
        # 1007 학습 결함: 순간값 문턱(3°·0.2·0.08)은 학습의 확률적 행동(탐색 잡음)으로 0 지급(같은 seed 결정적 102/154 vs 확률적 0/119)
        # → 한 주기 EMA 값(정지 래치 뒤부터 누적)으로 판정. 평가(eval_w3 STAND 정상)는 절대 순간 기준 그대로.
        c_ = torch.stack([tilt, w, v], 1)
        k_ = min(1.0, dt / 0.91)
        s["spost"] = torch.where((sw & ge).unsqueeze(1), c_, s["spost"] + k_ * (c_ - s["spost"]))
        sp = s["spost"]
        ok = sw & ~ge & both & (sp[:, 0] < sret_tilt) & (sp[:, 1] < sret_w) & (sp[:, 2] < sret_v)
        s["s_run"] = torch.where(ok, s["s_run"] + dt, torch.zeros_like(s["s_run"]))
        pay = sw & (s["s_run"] >= sret_hold)
        tout = sw & ~pay & (t - s["t_stop_on"] > sret_T)
        s["ev_sret"] = torch.where(pay, (1.0 - (t - s["t_stop_on"]) / sret_T).clamp(min=0.0), s["ev_sret"])
        S["sret_paid"] += float(pay.sum()); S["sret_due"] += float((pay | tout).sum())
        s["s_done"] = s["s_done"] | pay | tout


# ---------------------------------------------------------------- 회복 착지 추적(interval) ----------------------------------------------------------------
def w3d_tracker(env, env_ids, df_margin: float = 0.01, df_scale: float = 0.05, nom_hw: float = 0.144, gain_scale: float = 0.01):
    """mode="interval"(0 s), **w3_step_tracker·w3d_dist_step 뒤**: 이번 틱 유효 착지(ev_td) 중 회복 창 안의 것 = 회복 착지,
    포획 차분형 v2(w3_c2.capture_diff_score, ξ_eq = ξ + F/(mω0²), 직전 양발 지지영역 = 다른 발 + 착지 발의 이륙 전 위치) 기록."""
    s = _st(env); s3 = W3._st(env)
    asset = env.scene[H._base_cfg().name]
    bid = H._g21_bids(env); fp = asset.data.body_pos_w[:, bid, :]
    sensor = env.scene.sensors["contact_forces"]
    cont = sensor.data.net_forces_w[:, H._feet_body_ids(env), :].norm(dim=-1) > 5.0
    if s["last_xy"] is None or s["last_xy"].shape[0] != env.num_envs:
        s["last_xy"] = fp[..., :2].clone()
    fresh = (env.episode_length_buf <= 1).view(-1, 1, 1)
    s["last_xy"] = torch.where(fresh, fp[..., :2], s["last_xy"])
    td = s3["ev_td"] > 0.5
    rec = s["w_on"] & W3._ge(env)
    s["td_rec"] = td & rec.unsqueeze(1)
    if not bool(s["td_rec"].any()):
        s["cap_df"] = torch.zeros_like(s["cap_df"]); s["gain"] = torch.zeros_like(s["gain"]); s["phi_b"] = torch.zeros_like(s["phi_b"]); s["phi_a"] = torch.zeros_like(s["phi_a"])
        s["last_xy"] = torch.where(cont.unsqueeze(-1) & ~td.unsqueeze(-1), fp[..., :2], s["last_xy"])
        return
    yaw = _yaw(env)
    com, vc = C2._com_state(asset)
    zf = fp[..., 2].min(1).values
    om = torch.sqrt(C2.G / (com[:, 2] - zf).clamp(min=0.2))
    xi = com[:, :2] + vc[:, :2] / om.unsqueeze(1)
    hs = H._state(env)
    Fw = H._dist_force_w(hs, torch.arange(env.num_envs, device=env.device))
    mtot = asset.data.default_mass.sum(1).to(Fw.device)
    xi_eq = xi + Fw / (mtot * om ** 2).unsqueeze(1)
    corners = C2.foot_corners(fp[..., :2], yaw)
    corners_prev = C2.foot_corners(s["last_xy"], yaw)
    df, pb, pa = C2.capture_diff_score(xi_eq, corners, cont, td, margin=df_margin, scale=df_scale, corners_prev=corners_prev)
    # 손해 보전 자격(1007 단위시험 [B] 1차 실패 반영): 같은 순간 '명목 착지'(W3 착지 목표 위치, 반폭 nom_hw)를 했을 때의 Φ 와 비교해
    # 실제 착지가 더 포획했는가 — gain = clip((Φ_명목 − Φ_실제)/gain_scale, 0, 1). 차분형 cap_df 만으로 보전하면 부분 개선(cap_df 0.2–0.3)에서
    # 방향 회복 스텝이 원래 보폭보다 손해(뒤 밀기 −0.12)였다. 반대·명목 착지는 gain 0 → 보전 없음.
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    fc = f_.clamp(min=0.5)
    tl = torch.stack([(cmd[:, 0] / (4.0 * fc)).unsqueeze(1).expand(-1, 2),
                      torch.tensor([nom_hw, -nom_hw], device=env.device).unsqueeze(0) + (cmd[:, 1] / (4.0 * fc)).unsqueeze(1)], -1)   # (N,2,2) 골반 요 프레임
    nom_xy = asset.data.root_pos_w[:, None, :2] + _rot(yaw, tl)
    cn = C2.foot_corners(nom_xy, yaw)
    phi_nom = torch.zeros_like(df)
    for c_ in (0, 1):
        o_ = 1 - c_
        pts = torch.cat([corners[:, o_], cn[:, c_]], 1)
        msk = torch.cat([cont[:, o_:o_ + 1].expand(-1, 4), torch.ones_like(cont[:, :1]).expand(-1, 4)], 1)
        phi_nom[:, c_] = (C2.support_dist(xi_eq, pts, msk) - df_margin).clamp(min=0.0)
    phi_act = (pa - df_margin).clamp(min=0.0)
    gain = ((phi_nom - phi_act) / gain_scale).clamp(0.0, 1.0)
    z = torch.zeros_like(df)
    s["cap_df"] = torch.where(s["td_rec"], df, z); s["phi_b"] = torch.where(s["td_rec"], pb, z); s["phi_a"] = torch.where(s["td_rec"], pa, z)
    s["gain"] = torch.where(s["td_rec"], gain, z)
    s["w_ntd"] = s["w_ntd"] + s["td_rec"].float().sum(1)
    S = s["stats"]; S["rec_td"] += float(s["td_rec"].sum()); S["cap_sum"] += float(s["cap_df"].sum())
    s["last_xy"] = torch.where(cont.unsqueeze(-1) & ~td.unsqueeze(-1), fp[..., :2], s["last_xy"])


# ---------------------------------------------------------------- 보상(가산, 회복 착지·복귀 사건에서만 ≠ 0) ----------------------------------------------------------------
def _place_nom(env, sig: float = 0.04) -> torch.Tensor:
    """(N,2) w3_place_reward 의 발별 값(같은 식) — 합·ge 곱 전."""
    s3 = W3._st(env)
    return torch.exp(-(s3["td_err"] / sig) ** 2) * s3["ev_td"]


def stride2_per_foot(env, sig: float = 0.03, cmin: float = 0.02, width_gate: bool = True, k_opp: float = 0.5):
    """(N,2) 발별 w3_stride2 값(같은 식; 옆 명령이면 2D 식, 아니면 w3_stride 1D 식)과 (N,) 명목 최대점 s_max = 1 − exp(−‖ΔP*‖²/σ²)
    (명령이 활성 축이 없으면 0 — w3_stride2 가 0 을 주는 경우). 단위시험 [A] 가 Σ발·ge = w3_stride2 를 확인한다."""
    s = W3._st(env); L = W3._lat(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    fc = f_.clamp(min=0.5)
    tx = (cmd[:, 0] / fc).unsqueeze(1); ty = (cmd[:, 1] / fc).unsqueeze(1)
    # 1D(w3_stride)
    r1 = (torch.exp(-((s["td_dx"] - tx) / sig) ** 2) - torch.exp(-(tx / sig) ** 2)) * s["ev_td"]
    r1 = r1 * (cmd[:, 0].abs() >= cmin).float().unsqueeze(1)
    # 2D(w3_stride2 옆)
    e2 = (L["td_dx"] - tx) ** 2 + (L["td_dy"] - ty) ** 2
    r2 = (torch.exp(-e2 / sig ** 2) - torch.exp(-(tx ** 2 + ty ** 2) / sig ** 2)) * s["ev_td"]
    if width_gate:
        r2 = r2 * (1.0 - L["width_bad"])
    n2 = (tx ** 2 + ty ** 2).clamp(min=1e-8)
    r2 = r2 + k_opp * ((L["td_dx"] * tx + L["td_dy"] * ty) / n2).clamp(-1.0, 0.0) * s["ev_td"]
    lat = (cmd[:, 1].abs() >= cmin).unsqueeze(1)
    r = torch.where(lat, r2, r1)
    smax_lat = 1.0 - torch.exp(-(tx ** 2 + ty ** 2) / sig ** 2).squeeze(1)
    smax_x = (1.0 - torch.exp(-(tx / sig) ** 2)).squeeze(1) * (cmd[:, 0].abs() >= cmin).float()
    smax = torch.where(lat.squeeze(1), smax_lat, smax_x)
    return r, smax


def w3d_cap(env) -> torch.Tensor:
    s = _st(env)
    return (s["cap_df"] * s["td_rec"].float()).sum(1)


def w3d_relief_place(env, sig: float = 0.04) -> torch.Tensor:
    s = _st(env)
    if not bool(s["td_rec"].any()):
        return torch.zeros(env.num_envs, device=env.device)
    return ((s["gain"] - _place_nom(env, sig)).clamp(min=0.0) * s["td_rec"].float()).sum(1)


def w3d_relief_stride(env, sig: float = 0.03, cmin: float = 0.02, width_gate: bool = True, k_opp: float = 0.5) -> torch.Tensor:
    s = _st(env)
    if not bool(s["td_rec"].any()):
        return torch.zeros(env.num_envs, device=env.device)
    r, smax = stride2_per_foot(env, sig=sig, cmin=cmin, width_gate=width_gate, k_opp=k_opp)
    return ((s["gain"] * smax.unsqueeze(1) - r).clamp(min=0.0) * s["td_rec"].float()).sum(1)


def stride4_per_foot(env, sig: float = 0.03, cmin: float = 0.02, wcmin: float = 0.08, width_gate: bool = True, k_opp: float = 0.5,
                     nominal_half_width: float = 0.144, frame: str = "td", rot_frac: float = 0.5, lever: bool = True, sinc: bool = True):
    """(N,2) 발별 w3_stride4 값(같은 식: 회전 + 병진 명령이면 곡선 착지 목표, 그 밖은 stride2_per_foot)과 (N,2) 발별 명목 최대점
    s_max = 1 − exp(−‖ΔP*‖²/σ²)(곡선 목표는 발마다 다름 — 안쪽/바깥 발). 단위시험 [A4] 가 Σ발·ge = w3_stride4 를 확인한다."""
    rb, sb = stride2_per_foot(env, sig=sig, cmin=cmin, width_gate=width_gate, k_opp=k_opp)
    sb = sb.unsqueeze(1).expand(-1, 2)
    M = getattr(env, "_w3mix", None)
    if M is None:
        return rb, sb
    s3 = W3._st(env)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    T = W3.w3_curve_target(cmd, f_, nominal_half_width, rot_frac=rot_frac, lever=lever, sinc=sinc)
    yf = M["psi_ref"] if frame == "td" else M["psi_lo"]
    dl = W3._rot2(-yf, M["td_d"])
    e2 = ((dl - T) ** 2).sum(-1); n2 = (T ** 2).sum(-1)
    r = (torch.exp(-e2 / sig ** 2) - torch.exp(-n2 / sig ** 2)) * s3["ev_td"]
    L = getattr(env, "_w3lat", None)
    if width_gate and L is not None:
        r = r * (1.0 - L["width_bad"])
    r = r + k_opp * ((dl * T).sum(-1) / n2.clamp(min=1e-8)).clamp(-1.0, 0.0) * s3["ev_td"]
    on = ((cmd[:, 2].abs() >= wcmin) & ((cmd[:, 0].abs() >= cmin) | (cmd[:, 1].abs() >= cmin))).unsqueeze(1)
    return torch.where(on, r, rb), torch.where(on, 1.0 - torch.exp(-n2 / sig ** 2), sb)


def w3d_relief_stride4(env, sig: float = 0.03, cmin: float = 0.02, wcmin: float = 0.08, nominal_half_width: float = 0.144) -> torch.Tensor:
    """W3S4d 용 보폭 손해 보전(w3_stride4 와 같은 가중): 회복 착지마다 max(0, gain·s_max − stride4_발) — 곡선 착지 목표와 일관."""
    s = _st(env)
    if not bool(s["td_rec"].any()):
        return torch.zeros(env.num_envs, device=env.device)
    r, smax = stride4_per_foot(env, sig=sig, cmin=cmin, wcmin=wcmin, nominal_half_width=nominal_half_width)
    return ((s["gain"] * smax - r).clamp(min=0.0) * s["td_rec"].float()).sum(1)


def w3d_value(env, name: str) -> torch.Tensor:
    return _st(env)[name]


# ---------------------------------------------------------------- 커리큘럼·로그 ----------------------------------------------------------------
def w3d_curriculum(env, env_ids, max_fall: float = 0.25, min_ep: float = 1500.0, min_iters: int = 100, steps_per_iter: int = 24, n_levels: int = len(LEVELS),
                   start_level: int = 0):
    """충격 크기 단계(LEVELS) 진급: 현재 단계에서 충격이 실제로 걸린 에피소드 ≥ min_ep 이고 그 낙상률 ≤ max_fall, 체류 ≥ min_iters 반복.
    절대 문턱(톱니 금지). 로그 = 단계·블록 낙상률·회복 착지 평균 점수·복귀 지급률·정지 복귀 지급률."""
    s = _st(env)
    g = getattr(env, "_w3d_cur", None)
    if g is None:
        g = {"since": 0, "b": {}, "fall": float("nan")}
        env._w3d_cur = g; s["lvl"] = int(start_level)
    it = int(env.common_step_counter) // steps_per_iter
    lv = int(s["lvl"]); d = s["lvl_stats"].get(lv, [0.0, 0.0])
    b = g["b"].get(lv, [0.0, 0.0])
    ne, nf = d[0] - b[0], d[1] - b[1]
    if ne >= min_ep:
        g["fall"] = nf / ne
        adv = g["fall"] <= max_fall and it - g["since"] >= min_iters and lv < n_levels - 1
        print(f"[W3D 충격 블록] 반복 {it} 단계 {lv} F≤{LEVELS[min(lv, len(LEVELS) - 1)][1]:.0f} N: 충격 에피소드 {ne:.0f} 낙상 {g['fall']:.3f}"
              + (" → 단계 상승" if adv else ""), flush=True)
        g["b"][lv] = [d[0], d[1]]
        if adv:
            s["lvl"] = lv + 1; g["since"] = it
    S = s["stats"]
    return {"level": float(s["lvl"]), "imp_fall_block": g["fall"], "applied": S["applied"], "canceled_stand": S["canceled"],
            "cap_mean": S["cap_sum"] / max(S["rec_td"], 1.0), "rec_td": S["rec_td"],
            "ret_rate": S["ret_paid"] / max(S["ret_due"], 1.0), "sret_rate": S["sret_paid"] / max(S["sret_due"], 1.0), "stop_issued": S["stop_issued"]}
