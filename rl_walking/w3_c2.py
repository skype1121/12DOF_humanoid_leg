"""closure 2 구조 경로 c2 (1006) — W3 스텝 엔진 위의 '필요한 만큼 딛고 멈추는' 회복.

설계·단위시험 계획: reports/gen_w3/C2_DESIGN.md. 기본 꺼짐 — 이 모듈은 W3C2* 환경 설정에서만 쓰인다(기존 W3/G 계열 불변).

구성
  1. supervisor(배포 가능 신호만: IMU 기울기·|ω_xy|·접촉 0/1·시간·유효 착지 사건) 상태기계
       STAND(0) → REC(1, ge 1) → STOPREQ(2, ge 요청 0, 엔진이 반주기 경계에서 래치) → POST(3, ge 0 확인 뒤 정착 감시) → STAND
     - 켜짐: 감지기(기울기 > on_tilt 또는 |ω_xy| > on_w, det_delay 틱 지연) & 무장(학습 외란 힘이 걸렸거나 해제 뒤 arm_s 안)
       — 무장은 학습 전용(G16a 자기 개방 차단). 배포에서는 armed = 항상 참으로 같은 함수를 쓴다.
     - 정지 요청(상태 조건): REC ≥ min_rec_s & 유효 착지 ≥ min_td & 최근 win_s 동안 max|ω_xy| < stop_w & max 기울기 < stop_tilt
       (선택: 운동학 속도 추정 |v| < stop_v — 배포엔 추정기 필요) 또는 REC ≥ tmax_s(시간 초과, 별도 집계).
     - STOPREQ 중 감지기가 다시 켜지면(시간 초과 전) REC 로 취소.
     - 래치 해제(실제 ge 0)는 W3 엔진(w3_gait_update)의 반주기 경계 규칙 그대로 — c2 는 _ge_cmd 만 바꾼다.
  2. 보상(critic/보상 전용 특권량 — actor 관측 계약 불변: 320 차원·velocity_commands·gait_clock 그대로)
     - c2_place_reward: 회복 중(REC/STOPREQ) 유효 착지는 포획점 목표, 그 밖은 W3 w3_place_reward 와 **같은 식·같은 값**.
       목표(골반 요 프레임, 발별) = W3 명목 목표(명령 0 → (0, ±w)) + v_CoM/ω0 + b·v/√(|v|²+v_ref²) (|오프셋| ≤ max_off).
       v = 0 이면 W3 명목 목표와 정확히 같다(연속성).
     - ev_settle: POST 에서 양발·기울기·|ω|·|v| 정착이 settle_s 유지되면 1회(외란 1건당 최대 1회 — 새 외력 상승 에지에서만 재무장).
     - 초과 스텝 벌점 없음(1006 결정: 생존 env 가 오히려 더 많이 딛는다).
  3. 외란 없는 에피소드(g22r 범주 0)는 supervisor 가 건드리지 않음 → Stage 1(W3S1c)과 같은 명령·보상.

순수 계산(c2_core, capture_target_rel, place_value)은 Isaac 없이 CPU 단위시험(rl_walking/tests_w3_c2.py).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from . import w3 as W3

STAND, REC, STOPREQ, POST = 0, 1, 2, 3
G = 9.81


# ---------------------------------------------------------------- 파라미터 ----------------------------------------------------------------
@dataclass
class C2Params:
    on_tilt: float = 4.0          # KE 감지기 sw_on 과 같은 값(deg)
    on_w: float = 0.5             # rad/s
    det_delay: int = 2            # 감지 지연 [틱] (KE det_from_obs ≈ 관측 지연)
    arm_s: float = 2.5            # 외력 해제 뒤 무장 유지 [s] (학습 전용)
    min_rec_s: float = 0.9        # 정지 요청 최소 REC 시간 ≈ 한 주기(두 걸음)
    min_td: int = 1               # 정지 요청 최소 유효 착지 수
    win_s: float = 0.45           # 정숙 창 ≈ 반주기(1.1 Hz)
    stop_w: float = 0.6           # 창 안 max|ω_xy| 문턱 [rad/s] — **C2CAL 실측으로 확정 전 가정값**
    stop_tilt: float = 4.0        # 창 안 max 기울기 문턱 [deg] — 가정값
    stop_v: float = -1.0          # > 0 이면 |v| 조건 추가(운동학 속도 추정 대리, 잡음 kv_noise) — C2v 변형
    kv_noise: float = 0.03        # [m/s] 속도 추정 잡음 σ(학습 DR, 가정)
    tmax_s: float = 4.0           # REC 시간 초과 → 정지 요청
    refr_s: float = 0.5           # POST 진입 뒤 재켜짐 불응 [s]
    post_s: float = 3.0           # POST 유지 뒤 STAND
    settle_tilt: float = 3.0      # 정착(보상 전용, 특권 |v| 사용 가능)
    settle_w: float = 0.3
    settle_v: float = 0.08
    settle_s: float = 0.3


def _P(params: dict | None) -> C2Params:
    p = C2Params()
    for k, v in (params or {}).items():
        if not hasattr(p, k):
            raise KeyError(f"C2Params 에 없는 키: {k}")
        setattr(p, k, type(getattr(p, k))(v))
    return p


# ---------------------------------------------------------------- 상태 ----------------------------------------------------------------
def new_state(N: int, dev, P: C2Params, dt: float) -> dict:
    K = max(1, int(round(P.win_s / dt)))
    D = max(0, int(P.det_delay))
    z = lambda *sh: torch.zeros(*sh, device=dev)
    return {"mode": torch.zeros(N, dtype=torch.long, device=dev), "t": z(N), "n_td": z(N), "refr": z(N),
            "wbuf": z(N, K), "tbuf": z(N, K), "bi": 0, "dbuf": torch.zeros(N, D + 1, dtype=torch.bool, device=dev), "di": 0,
            "arm_until": z(N), "prev_force": torch.zeros(N, dtype=torch.bool, device=dev),
            "settle_run": z(N), "paid": torch.zeros(N, dtype=torch.bool, device=dev), "elig": torch.zeros(N, dtype=torch.bool, device=dev),
            # 이번 틱 사건(0/1)
            "ev_trig": z(N), "ev_req": z(N), "ev_timeout": z(N), "ev_cancel": z(N), "ev_latch": z(N), "ev_settle": z(N),
            # 착지 사건 보조(발별): 회복 중 착지였는지, 포획 목표 오차
            "td_rec": torch.zeros(N, 2, dtype=torch.bool, device=dev), "cap_err": z(N, 2),
            "K": K, "D": D}


def reset_envs(st: dict, mask: torch.Tensor):
    """에피소드 시작 env 의 상태 초기화(버퍼 포함)."""
    if not bool(mask.any()):
        return
    st["mode"][mask] = STAND
    for k in ("t", "n_td", "refr", "arm_until", "settle_run"):
        st[k][mask] = 0.0
    st["wbuf"][mask] = 0.0; st["tbuf"][mask] = 0.0; st["dbuf"][mask] = False
    st["prev_force"][mask] = False; st["paid"][mask] = False; st["elig"][mask] = False
    st["td_rec"][mask] = False; st["cap_err"][mask] = 0.0


# ---------------------------------------------------------------- 순수 상태기계 ----------------------------------------------------------------
def c2_core(st: dict, tilt: torch.Tensor, w: torch.Tensor, armed: torch.Tensor, td_any: torch.Tensor, dt: float, P: C2Params,
            v_est: torch.Tensor | None = None) -> torch.Tensor:
    """한 틱 supervisor 갱신 → want(bool, N) = 이번 틱 엔진에 요청할 gait_enable.
    입력은 배포 가능 신호(기울기 deg, |ω_xy| rad/s, 직전 틱 유효 착지 사건)와 무장(학습 전용, 배포 = 참). v_est 는 C2v 변형에서만."""
    N = tilt.shape[0]
    # 정숙 창(최근 K 틱 최대값)
    K = st["K"]; i = st["bi"]
    st["wbuf"][:, i] = w; st["tbuf"][:, i] = tilt; st["bi"] = (i + 1) % K
    wmax = st["wbuf"].max(1).values; tmax = st["tbuf"].max(1).values
    # 감지 지연(링 버퍼: D 틱 전 값)
    det_now = (tilt > P.on_tilt) | (w > P.on_w)
    D = st["D"]
    if D > 0:
        j = st["di"]; st["dbuf"][:, j] = det_now; st["di"] = (j + 1) % (D + 1)
        det = st["dbuf"][:, st["di"]]                    # 가장 오래된 칸 = D 틱 전
    else:
        det = det_now
    m = st["mode"]
    for k in ("ev_trig", "ev_req", "ev_timeout", "ev_cancel"):
        st[k] = torch.zeros(N, device=tilt.device)
    st["refr"] = (st["refr"] - dt).clamp(min=0.0)
    # STAND / POST → REC
    start = ((m == STAND) | (m == POST)) & det & armed & (st["refr"] <= 0.0)
    # REC 진행
    in_rec = m == REC
    t_rec = st["t"] + dt
    n_td = st["n_td"] + td_any.float()
    calm = (wmax < P.stop_w) & (tmax < P.stop_tilt)
    if P.stop_v > 0.0 and v_est is not None:
        calm = calm & (v_est < P.stop_v)
    req = in_rec & (t_rec >= P.min_rec_s) & (n_td >= P.min_td) & calm
    tout = in_rec & ~req & (t_rec >= P.tmax_s)
    # STOPREQ 취소(시간 초과로 들어온 것은 취소 불가)
    in_sr = m == STOPREQ
    cancel = in_sr & det & (st["t"] + dt < P.tmax_s) & armed
    # POST → STAND
    post_end = (m == POST) & (st["t"] + dt >= P.post_s) & ~start
    new_m = m.clone()
    new_m = torch.where(start, torch.full_like(m, REC), new_m)
    new_m = torch.where(req | tout, torch.full_like(m, STOPREQ), new_m)
    new_m = torch.where(cancel, torch.full_like(m, REC), new_m)
    new_m = torch.where(post_end, torch.full_like(m, STAND), new_m)
    # 시간·착지 수
    st["t"] = torch.where(start, torch.zeros_like(st["t"]),
                          torch.where(in_rec | in_sr | (m == POST), st["t"] + dt, torch.zeros_like(st["t"])))
    st["t"] = torch.where(post_end, torch.zeros_like(st["t"]), st["t"])
    st["n_td"] = torch.where(start, torch.zeros_like(n_td), torch.where(in_rec | in_sr, n_td, st["n_td"]))
    st["elig"] = st["elig"] | start                     # 정착 보상 자격 = 이번 회복이 실제로 켜졌음
    st["ev_trig"] = start.float(); st["ev_req"] = req.float(); st["ev_timeout"] = tout.float(); st["ev_cancel"] = cancel.float()
    st["mode"] = new_m
    return new_m == REC


def c2_after_latch(st: dict, ge_after: torch.Tensor, P: C2Params):
    """엔진 갱신 뒤: STOPREQ 이고 실제 ge 가 0 이 되었으면 POST(불응기 시작)."""
    lat = (st["mode"] == STOPREQ) & ~ge_after
    st["mode"] = torch.where(lat, torch.full_like(st["mode"], POST), st["mode"])
    st["t"] = torch.where(lat, torch.zeros_like(st["t"]), st["t"])
    st["refr"] = torch.where(lat, torch.full_like(st["refr"], P.refr_s), st["refr"])
    st["settle_run"] = torch.where(lat, torch.zeros_like(st["settle_run"]), st["settle_run"])
    st["ev_latch"] = lat.float()


def c2_settle_update(st: dict, both: torch.Tensor, tilt: torch.Tensor, w: torch.Tensor, v: torch.Tensor, force_on: torch.Tensor,
                     dt: float, P: C2Params):
    """정착 사건(보상 전용): POST 에서 조건(외력 없음 포함) settle_s 유지 → 1회.
    자격 = 회복이 실제로 켜졌음(elig) & 이번 외란(외력 상승 에지 이후)에서 아직 미지급(paid) — 같은 외란 안 재켜짐 파밍 차단(1006 단위시험 6c 에서 발견:
    정착 상태에서 새 외력이 걸리는 틱에 즉시 지급되던 결함 → 외력 중 정착 불인정 + 상승 에지에서 정착 시간 0)."""
    rise = force_on & ~st["prev_force"]
    st["paid"] = st["paid"] & ~rise
    st["elig"] = st["elig"] & ~(rise & ((st["mode"] == STAND) | (st["mode"] == POST)))   # 새 외란은 새 켜짐이 있어야 자격(6c-2: 지난 외란 자격 잔존 결함)
    st["prev_force"] = force_on.clone()
    ok = (st["mode"] == POST) & both & ~force_on & (tilt < P.settle_tilt) & (w < P.settle_w) & (v < P.settle_v)
    st["settle_run"] = torch.where(ok, st["settle_run"] + dt, torch.zeros_like(st["settle_run"]))
    ev = (st["settle_run"] >= P.settle_s) & st["elig"] & ~st["paid"]
    st["paid"] = st["paid"] | ev
    st["elig"] = st["elig"] & ~ev
    st["ev_settle"] = ev.float()


# ---------------------------------------------------------------- 포획점 목표(순수) ----------------------------------------------------------------
def capture_target_rel(nom_x: torch.Tensor, nom_y: torch.Tensor, v_yaw: torch.Tensor, omega0: torch.Tensor,
                       b: float = 0.02, v_ref: float = 0.05, max_off: float = 0.25):
    """골반 요 프레임 발별 착지 목표 (N,2) x, y.
    nom_x/nom_y (N,2) = W3 명목 목표(w3_step_tracker 와 같은 식), v_yaw (N,2) = CoM 수평속도(요 프레임), omega0 (N,) = √(g/z).
    오프셋 = v/ω0 (ICP − CoM) + b·v/√(|v|²+v_ref²)(정지용 여유, v→0 에서 0), 노름 ≤ max_off. v = 0 이면 명목 목표와 정확히 같다."""
    vn = v_yaw.norm(dim=-1, keepdim=True)
    off = v_yaw / omega0.clamp(min=1e-3).unsqueeze(1) + b * v_yaw / torch.sqrt(vn ** 2 + v_ref ** 2)
    on = off.norm(dim=-1, keepdim=True)
    off = off * torch.where(on > max_off, max_off / on.clamp(min=1e-9), torch.ones_like(on))
    return nom_x + off[:, :1], nom_y + off[:, 1:2]


def place_value(ev_td: torch.Tensor, td_err: torch.Tensor, td_rec: torch.Tensor, cap_err: torch.Tensor, ge: torch.Tensor,
                sig: float = 0.04, sig_cap: float = 0.05) -> torch.Tensor:
    """(N,) 착지 보상 값. 회복 착지가 아니면 W3 w3_place_reward 와 같은 식(같은 σ) → 무외란 env 는 값이 완전히 같다."""
    r_nom = torch.exp(-(td_err / sig) ** 2)
    r_cap = torch.exp(-(cap_err / sig_cap) ** 2)
    return (torch.where(td_rec, r_cap, r_nom) * ev_td).sum(1) * ge.float()


# ---------------------------------------------------------------- env 어댑터 ----------------------------------------------------------------
def _c2(env, P: C2Params | None = None) -> dict:
    st = getattr(env, "_c2", None)
    if st is None or st["mode"].shape[0] != env.num_envs:
        P = P or getattr(env, "_c2_P", None) or C2Params()
        st = new_state(env.num_envs, env.device, P, env.step_dt)
        st["stats"] = {k: 0.0 for k in ("trig", "req", "timeout", "cancel", "latch", "settle", "rec_ticks", "c2_ticks")}
        env._c2 = st
    return st


def _c2_mask(env) -> torch.Tensor:
    """외란 범주(g22r cat 1 앞 / 2 뒤) 에피소드 = c2 회복 에피소드. 범주 0 은 Stage 1 그대로."""
    g = getattr(env, "_g22", None)
    if g is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return g["cat"] > 0.5


def _imu_signals(env):
    asset = env.scene["robot"]
    pg = asset.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1.0, 1.0)))
    w = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    return asset, tilt, w


def c2_gait_update(env, env_ids, f_step: float = 1.1, ramp_s: float = 0.2, c2: dict | None = None):
    """mode="interval"(0 s): **w3_gait_update 대체** — supervisor(c2 에피소드만 _ge_cmd·명령 덮기) → W3 엔진 래치 → 래치 확인.
    c2 에피소드가 아닌 env 는 _ge_cmd·명령을 건드리지 않는다(= w3_gait_update 그대로)."""
    from . import mdp_hang as H
    P = getattr(env, "_c2_P", None)
    if P is None:
        prm = dict(c2 or {})
        for kv in filter(None, os.environ.get("C2_PARAMS", "").split(",")):   # 평가·반사실 진단 전용 덮어쓰기 'k=v,...'
            k, v = kv.split("="); prm[k.strip()] = float(v)
        P = _P(prm); env._c2_P = P
        env._c2_arm_off = os.environ.get("C2_ARM", "") == "stand"          # 진단 팔 B1: 회복 켜짐 금지(발목 전략만)
        env._c2_deploy = os.environ.get("C2_ARM", "") == "deploy"         # 평가: 배포 의미(전 env supervisor·무장 항상 참 — nd 오탐 포함)
        print(f"[c2] supervisor 파라미터 {P} · 켜짐 금지 {env._c2_arm_off} · 배포 의미 {env._c2_deploy}", flush=True)
    st = _c2(env, P)
    fresh = env.episode_length_buf <= 1
    reset_envs(st, fresh)
    deploy = getattr(env, "_c2_deploy", False)
    m2 = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if deploy else _c2_mask(env)
    asset, tilt, w = _imu_signals(env)
    hs = H._state(env)
    force_on = hs["dst_F_now"] > 0.0
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    st["arm_until"] = torch.where(force_on, t + P.arm_s, st["arm_until"])
    armed = (t < st["arm_until"]) & m2 & (not getattr(env, "_c2_arm_off", False))
    if deploy:
        armed = m2.clone()
    en = getattr(env, "_c2_enable", None)                       # 평가 스크립트용 (N,) bool: 인계 전 홀드 구간 등에서 켜짐 금지
    if en is not None:
        armed = armed & en
    s3 = W3._st(env)
    td_any = s3["ev_td"].sum(1) > 0.5                           # 직전 틱 유효 착지(보상과 같은 20 ms 지연 규약)
    v_est = None
    if P.stop_v > 0.0:
        v_est = (asset.data.root_lin_vel_b[:, :2] + P.kv_noise * torch.randn(env.num_envs, 2, device=env.device)).norm(dim=1)
    want = c2_core(st, tilt, w, armed, td_any, env.step_dt, P, v_est)
    if getattr(env, "_ge_cmd", None) is not None:
        env._ge_cmd = torch.where(m2, want, env._ge_cmd)
        term = env.command_manager.get_term("base_velocity")
        term.vel_command_b[m2] = 0.0                              # 회복 중 명령 0(cue 없음) — 관측 명령 채널은 그대로 0
    W3.w3_gait_update(env, env_ids, f_step=f_step, ramp_s=ramp_s)
    c2_after_latch(st, W3._ge(env), P)
    sensor = env.scene.sensors["contact_forces"]; fid = H._feet_body_ids(env)
    both = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0).all(dim=1)
    v = asset.data.root_lin_vel_w[:, :2].norm(dim=1)
    c2_settle_update(st, both, tilt, w, v, force_on, env.step_dt, P)
    S = st["stats"]
    for k, e in (("trig", "ev_trig"), ("req", "ev_req"), ("timeout", "ev_timeout"), ("cancel", "ev_cancel"), ("latch", "ev_latch"), ("settle", "ev_settle")):
        S[k] += float(st[e].sum())
    S["rec_ticks"] += float((st["mode"] == REC).sum()); S["c2_ticks"] += float(m2.sum())


def _com_state(asset):
    m = asset.data.default_mass.to(asset.data.root_pos_w.device)
    cp = asset.data.body_com_pos_w if hasattr(asset.data, "body_com_pos_w") else asset.data.body_pos_w
    cv = asset.data.body_com_lin_vel_w if hasattr(asset.data, "body_com_lin_vel_w") else asset.data.body_lin_vel_w
    com = (cp * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    vc = (cv * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    return com, vc


def c2_tracker(env, env_ids, nominal_half_width: float = 0.10, b: float = 0.02, v_ref: float = 0.05, max_off: float = 0.25):
    """mode="interval"(0 s), **w3_step_tracker 뒤**: 이번 틱 유효 착지(ev_td)마다 회복 중 여부·포획 목표 오차 기록(특권 CoM — 보상 전용)."""
    from . import mdp_hang as H
    st = _c2(env); s3 = W3._st(env)
    td = s3["ev_td"] > 0.5
    if not bool(td.any()):
        st["td_rec"] = torch.zeros_like(st["td_rec"]); return
    asset = env.scene["robot"]
    bid = H._g21_bids(env); fp = asset.data.body_pos_w[:, bid, :]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    cy, sy = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    rel = fp[..., :2] - asset.data.root_pos_w[:, None, :2]
    rx = rel[..., 0] * cy + rel[..., 1] * sy; ry = -rel[..., 0] * sy + rel[..., 1] * cy
    # W3 명목 목표(w3_step_tracker 와 같은 식)
    cmd = env.command_manager.get_command("base_velocity")
    f_ = env._gf if getattr(env, "_gf", None) is not None else torch.full((env.num_envs,), 1.1, device=env.device)
    nom_x = (cmd[:, 0] / (4.0 * f_.clamp(min=0.5))).unsqueeze(1).expand(-1, 2)
    nom_y = torch.tensor([nominal_half_width, -nominal_half_width], device=env.device).unsqueeze(0) + (cmd[:, 1] / (4.0 * f_.clamp(min=0.5))).unsqueeze(1)
    com, vc = _com_state(asset)
    vy = torch.stack([vc[:, 0] * cy[:, 0] + vc[:, 1] * sy[:, 0], -vc[:, 0] * sy[:, 0] + vc[:, 1] * cy[:, 0]], -1)
    zf = fp[..., 2].min(1).values
    om = torch.sqrt(G / (com[:, 2] - zf).clamp(min=0.2))
    tx, ty = capture_target_rel(nom_x, nom_y, vy, om, b=b, v_ref=v_ref, max_off=max_off)
    err = torch.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
    rec = ((st["mode"] == REC) | (st["mode"] == STOPREQ)) & W3._ge(env) & _c2_mask(env)
    st["td_rec"] = td & rec.unsqueeze(1)
    st["cap_err"] = torch.where(st["td_rec"], err, torch.zeros_like(err))


# ---------------------------------------------------------------- 보상 ----------------------------------------------------------------
def c2_place_reward(env, sig: float = 0.04, sig_cap: float = 0.05) -> torch.Tensor:
    """w3_place_reward 대체: 회복 착지 = 포획 목표, 나머지 = W3 명목(같은 값)."""
    s3 = W3._st(env); st = _c2(env)
    return place_value(s3["ev_td"], s3["td_err"], st["td_rec"], st["cap_err"], W3._ge(env), sig, sig_cap)


def c2_value(env, name: str) -> torch.Tensor:
    return _c2(env)[name]


def c2_icp_obs(env, clip: float = 0.3) -> torch.Tensor:
    """critic 전용(W3C2i 변형): ICP − CoM = v_CoM/ω0 (요 프레임, (N,2)) [m]. actor 에는 넣지 않는다(유격 8° → 추정 오차 ~7 cm)."""
    from . import mdp_hang as H
    asset = env.scene["robot"]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    com, vc = _com_state(asset)
    bid = H._g21_bids(env); zf = asset.data.body_pos_w[:, bid, 2].min(1).values
    om = torch.sqrt(G / (com[:, 2] - zf).clamp(min=0.2))
    vx = vc[:, 0] * cy + vc[:, 1] * sy; vyy = -vc[:, 0] * sy + vc[:, 1] * cy
    return (torch.stack([vx, vyy], -1) / om.unsqueeze(1)).clamp(-clip, clip)


def c2_log(env, env_ids) -> dict:
    """CurrTerm(효과 없음, 집계 로그): 누적 사건 수와 REC 점유율."""
    S = _c2(env)["stats"]
    return {"c2_trig": S["trig"], "c2_req": S["req"], "c2_timeout": S["timeout"], "c2_cancel": S["cancel"],
            "c2_latch": S["latch"], "c2_settle": S["settle"], "c2_rec_frac": S["rec_ticks"] / max(S["c2_ticks"], 1.0)}
