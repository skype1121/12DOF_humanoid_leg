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
    tilt_ref_tau: float = 0.0     # > 0: 기울기 = 기준 중력 방향 대비 편차(KE --sw_ref_tau 와 같은 개념, 배포 가능 — IMU 만). 기준은 STAND·양발·|ω|<0.1 일 때 EMA τ[s] 갱신, |기준 기울기| ≤ 8°.
                                  #     0 = 절대 기울기(종전). 1006 C2EVAL 스모크: 절대 기울기 감지기는 무외란 서기에서 외란 전 켜짐 21 % → 후보 수정.
    tilt_ref_init_s: float = 1.5  # 인계(감지 허용) 뒤 이 시간 동안 기준 = 현재 자세(블렌딩 과도 오탐 방지, KE --sw_ref_init_s)
    det_delay: int = 2            # 감지 지연 [틱] (KE det_from_obs ≈ 관측 지연)
    arm_s: float = 2.5            # 외력 해제 뒤 무장 유지 [s] (학습 전용)
    min_rec_s: float = 0.9        # 정지 요청 최소 REC 시간 ≈ 한 주기(두 걸음)
    min_td: int = 1               # 정지 요청 최소 유효 착지 수
    win_s: float = 0.45           # 정숙 창 ≈ 반주기(1.1 Hz)
    stop_w: float = 0.6           # 창 안 max|ω_xy| 문턱 [rad/s] — **C2CAL 실측으로 확정 전 가정값**
    stop_tilt: float = 4.0        # 창 안 max 기울기 문턱 [deg] — 가정값
    stop_v: float = -1.0          # > 0 이면 |v| 조건 추가(운동학 속도 추정 대리, 잡음 kv_noise) — C2v 변형
    kv_noise: float = 0.03        # [m/s] 속도 추정 백색 잡음 σ(학습 DR, 가정) — 주기 평균 45 틱이면 실효 ≈ 0.005 m/s 로 묻힘(레드팀 a)
    kv_bias: float = 0.0          # [m/s] 에피소드 고정 편향 σ(벡터, 리셋마다 재표본) — 운동학 추정기 계통 오차 대리(가정)
    kv_lf: float = 0.0            # [m/s] 저주파(OU, τ kv_lf_tau) 잡음 σ — 주기 평균에 남는 상관 오차(가정)
    kv_lf_tau: float = 1.0
    v_box_s: float = 0.0          # > 0: |v| 조건을 이 길이(한 보행 주기 0.909 s) 이동평균 속도 벡터의 크기로. 1006 C2CAL: 순간 |v| 는 제자리 걸음 흔들림이
                                  #     중앙 0.32 m/s 라 정지 조건이 영원히 안 열림 → 주기 평균(제자리 p90 0.058 vs 회복 p90 0.25–0.34 m/s)로 분리됨
    tmax_s: float = 4.0           # REC 시간 초과 → 정지 요청
    stop_td_k: int = 0            # 반사실 정지 시점 스윕(1006 문헌 권고 3): > 0 이면 정지 요청 = 회복 유효 착지 k 번째(상태 조건·취소 무시, 시간 초과는 유지)
    stop_oracle: float = 0.0      # 1 = 오라클(착지 후 참 ξ_eq ∈ 양발 지지영역 — 1006 스윕에서 언제나 첫 착지 = 약함), 2 = 속도 오라클(회복 착지 뒤 0.5 s 안
                                  #     참 주기 평균 속도 |v̄_T| < 0.07, 잡음 없음 — '운동량 소멸' 특권 기준). 진단 전용
    cancel_calm: float = 0.0      # > 0: STOPREQ 취소 조건 = 감지기 발화(종전) 대신 '정지 규칙 미충족'(창 max|ω|·기울기·|v̄| 중 하나라도 문턱 초과).
                                  #     1006 C2EVAL(c2v): 감지 문턱(4°·0.5)을 제자리 걸음 흔들림이 늘 넘어 요청의 90 % 가 취소됨(b18 요청 8.3·취소 7.6·래치 0.7)
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
            "td_rec": torch.zeros(N, 2, dtype=torch.bool, device=dev), "cap_err": z(N, 2), "cap_mg": z(N, 2), "cap_df": z(N, 2), "phi_b": z(N, 2), "phi_a": z(N, 2), "oracle_ok": torch.zeros(N, dtype=torch.bool, device=dev),
            "gref": torch.zeros(N, 3, device=dev), "en_t": z(N), "K": K, "D": D,
            "vbuf": z(N, max(1, int(round(P.v_box_s / dt))), 2), "vi": 0, "kvb": z(N, 2), "kvlf": z(N, 2),
            "vtbuf": z(N, max(2, int(round((1.0 / 1.1) / dt))), 2), "vti": 0, "since_rtd": torch.full((N,), 1e9, device=dev)}


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
            v_est: torch.Tensor | None = None, oracle_ok: torch.Tensor | None = None) -> torch.Tensor:
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
    forced = P.stop_td_k > 0 or P.stop_oracle > 0.0
    if P.stop_td_k > 0:
        req = in_rec & (n_td >= P.stop_td_k)
    elif P.stop_oracle > 0.0:
        req = in_rec & (oracle_ok if oracle_ok is not None else torch.zeros_like(in_rec))
    tout = in_rec & ~req & (t_rec >= P.tmax_s)
    # STOPREQ 취소(시간 초과로 들어온 것은 취소 불가)
    in_sr = m == STOPREQ
    _cdet = ~calm if P.cancel_calm > 0.0 else det
    cancel = in_sr & _cdet & (st["t"] + dt < P.tmax_s) & armed & (not forced)
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


def capture_margin_score(foot_xy: torch.Tensor, xi_xy: torch.Tensor, v_xy: torch.Tensor, scale: float = 0.05, v_min: float = 0.10) -> torch.Tensor:
    """(N,2) 이동 방향 포획 여유 점수(1006 기준 2 재설계): m = (발 − ξ)·v̂ (ξ = ICP = CoM + v/ω0, 월드 수평), 점수 = clip(m/scale, −1, 1).
    발이 이동 방향으로 ICP 보다 앞(m > 0) = 포획 → +, 뒤 = 넘어지는 쪽 → −. |v| ≤ v_min 이면 0(운동량 없음 — 제자리 디딤이 공짜 점수를 받지 않음).
    R2(G24Rcap) _capture_margin_ev 와 같은 정의(발별)."""
    vn = v_xy.norm(dim=-1, keepdim=True)
    vh = v_xy / vn.clamp(min=1e-9)
    m = ((foot_xy - xi_xy.unsqueeze(1)) * vh.unsqueeze(1)).sum(-1)
    return torch.where(vn > v_min, (m / scale).clamp(-1.0, 1.0), torch.zeros_like(m))


FOOT_FRONT, FOOT_BACK, FOOT_HW = 0.085, 0.065, 0.03     # 발목축 기준 발 사각형 [m] (실측 150 mm 평발: 앞 85·뒤 65, 반폭 3 cm — eval ho_state 와 같음)


def foot_corners(fxy: torch.Tensor, yaw: torch.Tensor, front: float = FOOT_FRONT, back: float = FOOT_BACK, hw: float = FOOT_HW) -> torch.Tensor:
    """(N,2,4,2) 발별 사각형 꼭짓점(월드 xy). 축 = 골반 요(발 요 대신 근사 — 회복 중 발 요 차이는 문서화된 근사)."""
    loc = torch.tensor([[front, hw], [front, -hw], [-back, -hw], [-back, hw]], device=fxy.device, dtype=fxy.dtype)
    c, s = torch.cos(yaw), torch.sin(yaw)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)       # (N,2,2)
    rot = torch.einsum("nij,kj->nki", R, loc)                                       # (N,4,2)
    return fxy.unsqueeze(2) + rot.unsqueeze(1)


def support_dist(p: torch.Tensor, corners: torch.Tensor, mask: torch.Tensor, K: int = 64) -> torch.Tensor:
    """(N,) 점 p 와 (마스크된) 꼭짓점 볼록 껍질 사이 거리(바깥 > 0, 안 ≤ 0 근사). corners (N,M,2), mask (N,M).
    지지 함수 방식: max_u [u·p − max_v u·v] (u = K 방향, 오차 ≤ 1 − cos(π/K) ≈ 0.1 %). 유효 꼭짓점 없음 → +inf."""
    ang = torch.arange(K, device=p.device, dtype=p.dtype) * (2 * math.pi / K)
    U = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                          # (K,2)
    hv = torch.einsum("nmd,kd->nmk", corners, U)                                    # (N,M,K)
    hv = torch.where(mask.unsqueeze(-1), hv, torch.full_like(hv, -float("inf")))
    h = hv.max(1).values                                                           # (N,K)
    d = (p @ U.T - h).max(1).values
    return torch.where(mask.any(1), d, torch.full_like(d, float("inf")))


def capture_diff_score(xi: torch.Tensor, corners: torch.Tensor, cont: torch.Tensor, td: torch.Tensor, margin: float = 0.01, scale: float = 0.05,
                       corners_prev: torch.Tensor | None = None):
    """발별(N,2) 차분형 포획 점수와 (Φ_전, Φ_후). 착지 발 c: 후 = 다른 발(접촉 시) + 착지 발(현재 위치).
    전 = corners_prev 가 있으면 **직전 양발 지지영역**(다른 발 + 착지 발의 이륙 전 위치) — 제자리로 되딛으면 0 점(1006 v2: 다른 발만을 '전'으로
    쓰면 ξ 가 두 발 사이(횡)에 있을 때 어떤 착지든 만점 → 스윕 오라클·2d 가 무의미해진 결함), 없으면 다른 발만(v1).
    Φ = max(0, dist − margin). 이미 포획(Φ_전 = 0) → 0 점, 지지 없던 착지(전 = 공집합) → 0 점(정의 불가)."""
    N = xi.shape[0]
    out = torch.zeros(N, 2, device=xi.device); pb = torch.zeros(N, 2, device=xi.device); pa = torch.zeros(N, 2, device=xi.device)
    flat = corners.reshape(N, 8, 2)
    for c in (0, 1):
        o = 1 - c
        mb = torch.zeros(N, 2, dtype=torch.bool, device=xi.device); mb[:, o] = cont[:, o]
        ma = mb.clone(); ma[:, c] = True
        if corners_prev is not None:
            fb = torch.cat([corners[:, o], corners_prev[:, c]], 1)                     # (N,8,2) 다른 발 현재 + 착지 발 이륙 전
            mbb = torch.cat([cont[:, o:o + 1].expand(-1, 4), torch.ones(N, 4, dtype=torch.bool, device=xi.device)], 1)
            db = support_dist(xi, fb, mbb)
        else:
            db = support_dist(xi, flat, mb.repeat_interleave(4, dim=1))
        da = support_dist(xi, flat, ma.repeat_interleave(4, dim=1))
        phb = (db - margin).clamp(min=0.0); pha = (da - margin).clamp(min=0.0)
        ok = td[:, c] & torch.isfinite(phb)
        sc = ((phb - pha) / scale).clamp(0.0, 1.0)
        out[:, c] = torch.where(ok, sc, torch.zeros_like(sc))
        pb[:, c] = torch.where(ok, phb, torch.zeros_like(phb)); pa[:, c] = torch.where(td[:, c], da, torch.zeros_like(da))
    return out, pb, pa


def place_value(ev_td: torch.Tensor, td_err: torch.Tensor, td_rec: torch.Tensor, cap_err: torch.Tensor, ge: torch.Tensor,
                sig: float = 0.04, sig_cap: float = 0.05, cap_score: torch.Tensor | None = None) -> torch.Tensor:
    """(N,) 착지 보상 값. 회복 착지가 아니면 W3 w3_place_reward 와 같은 식(같은 σ) → 무외란 env 는 값이 완전히 같다.
    cap_score 를 주면 회복 착지 점수를 그것으로(여유형), 없으면 포획 목표 오차형 exp(−e²/σ²)(종전)."""
    r_nom = torch.exp(-(td_err / sig) ** 2)
    r_cap = torch.exp(-(cap_err / sig_cap) ** 2) if cap_score is None else cap_score
    return (torch.where(td_rec, r_cap, r_nom) * ev_td).sum(1) * ge.float()


# ---------------------------------------------------------------- env 어댑터 ----------------------------------------------------------------
def _get_P(env, c2: dict | None = None) -> C2Params:
    """supervisor 파라미터를 한 곳에서 해석(1006 레드팀 결함 수정): 보상·커리큘럼이 supervisor 이벤트보다 먼저 _c2() 를 부르면 종전엔
    기본 C2Params()(v_box_s 0 → 속도 버퍼 길이 1 = 순간 |v|)로 상태를 만들고 다시 안 만들었다. 이제 어느 쪽이 먼저든
    env.cfg.events.w3_gait.params["c2"] + C2_PARAMS env var 로 같은 P 를 만든다."""
    P = getattr(env, "_c2_P", None)
    if P is not None:
        return P
    prm = dict(c2) if c2 is not None else {}
    if c2 is None:
        try:
            prm = dict(env.cfg.events.w3_gait.params.get("c2", {}) or {})
        except AttributeError:
            prm = {}
    for kv in filter(None, os.environ.get("C2_PARAMS", "").split(",")):   # 평가·반사실 진단 전용 덮어쓰기 'k=v,...'
        k, v = kv.split("="); prm[k.strip()] = float(v)
    P = _P(prm); env._c2_P = P
    env._c2_arm_off = os.environ.get("C2_ARM", "") == "stand"          # 진단 팔 B1: 회복 켜짐 금지(발목 전략만)
    env._c2_deploy = os.environ.get("C2_ARM", "") in ("deploy", "march")   # 평가: 배포 의미(전 env supervisor·무장 항상 참 — nd 오탐 포함)
    env._c2_march = os.environ.get("C2_ARM", "") == "march"           # C2CAL: 전 env ge 1 고정
    print(f"[c2] supervisor 파라미터 {P} · 켜짐 금지 {env._c2_arm_off} · 배포 의미 {env._c2_deploy}", flush=True)
    return P


def _c2(env, P: C2Params | None = None) -> dict:
    st = getattr(env, "_c2", None)
    P = P or _get_P(env)
    K = max(1, int(round(P.win_s / env.step_dt))); V = max(1, int(round(P.v_box_s / env.step_dt)))
    if st is None or st["mode"].shape[0] != env.num_envs or st["K"] != K or st["vbuf"].shape[1] != V or st["D"] != max(0, int(P.det_delay)):
        old = st["stats"] if st is not None and "stats" in st else None
        st = new_state(env.num_envs, env.device, P, env.step_dt)
        st["stats"] = old or {k: 0.0 for k in ("trig", "req", "timeout", "cancel", "latch", "settle", "rec_ticks", "c2_ticks")}
        env._c2 = st
    return st


def _c2_mask(env) -> torch.Tensor:
    """외란 범주(g22r cat 1 앞 / 2 뒤) 에피소드 = c2 회복 에피소드. 범주 0 은 Stage 1 그대로."""
    g = getattr(env, "_g22", None)
    if g is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return g["cat"] > 0.5


def tilt_dev_deg(g: torch.Tensor, gref: torch.Tensor) -> torch.Tensor:
    """(N,) 두 단위 중력 방향(몸통 프레임) 사이 각 [deg] — 상대 기울기 감지기(순수 계산, CPU 시험 대상)."""
    gn = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-9); rn = gref / gref.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.rad2deg(torch.acos((gn * rn).sum(-1).clamp(-1.0, 1.0)))


def update_gref(gref: torch.Tensor, g: torch.Tensor, upd: torch.Tensor, init: torch.Tensor, dt: float, tau: float, max_deg: float = 8.0) -> torch.Tensor:
    """기준 중력 방향 EMA: init 이면 현재값, upd 이면 dt/τ 만큼 따라감, 기준 기울기(아래 방향 대비)는 max_deg 로 제한."""
    a = min(1.0, dt / max(tau, 1e-6))
    r = torch.where(init.unsqueeze(1), g, torch.where(upd.unsqueeze(1), gref + a * (g - gref), gref))
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    down = torch.tensor([0.0, 0.0, -1.0], device=g.device).expand_as(r)
    ang = tilt_dev_deg(r, down)
    over = ang > max_deg
    if bool(over.any()):           # 아래 방향과 기준 사이를 max_deg 로 되돌림(구면 선형 근사)
        k = (max_deg / ang.clamp(min=1e-6)).unsqueeze(1)
        rr = down + k * (r - down); r = torch.where(over.unsqueeze(1), rr / rr.norm(dim=-1, keepdim=True), r)
    return r


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
    P = _get_P(env, c2)
    st = _c2(env, P)
    fresh = env.episode_length_buf <= 1
    reset_envs(st, fresh)
    deploy = getattr(env, "_c2_deploy", False)
    m2 = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if deploy else _c2_mask(env)
    asset, tilt, w = _imu_signals(env)
    if P.tilt_ref_tau > 0.0:                                  # 상대 기울기 감지기(배포 가능: IMU 중력 방향만)
        g = asset.data.projected_gravity_b
        en0 = getattr(env, "_c2_enable", None)
        en_now = en0 if en0 is not None else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        st["en_t"] = torch.where(en_now & ~fresh, st["en_t"] + env.step_dt, torch.zeros_like(st["en_t"]))
        init = fresh | (st["gref"].norm(dim=-1) < 0.5) | (st["en_t"] < P.tilt_ref_init_s)
        sensor0 = env.scene.sensors["contact_forces"]
        both0 = (sensor0.data.net_forces_w[:, H._feet_body_ids(env), :].norm(dim=-1) > 5.0).all(dim=1)
        upd = (st["mode"] == STAND) & both0 & (w < 0.1)
        st["gref"] = update_gref(st["gref"], g, upd, init, env.step_dt, P.tilt_ref_tau)
        tilt = tilt_dev_deg(g, st["gref"])
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
    if P.v_box_s > 0.0:                                        # 주기 이동평균(배포 = 다리 운동학 속도 추정기의 주기 평균, 학습 = 참 속도 + 잡음)
        if P.kv_bias > 0.0:
            st["kvb"] = torch.where(fresh.unsqueeze(1), P.kv_bias * torch.randn(env.num_envs, 2, device=env.device), st["kvb"])
        if P.kv_lf > 0.0:
            a_ = env.step_dt / max(P.kv_lf_tau, 1e-6)
            st["kvlf"] = st["kvlf"] * (1 - a_) + P.kv_lf * math.sqrt(2 * a_) * torch.randn(env.num_envs, 2, device=env.device)
            st["kvlf"] = torch.where(fresh.unsqueeze(1), torch.zeros_like(st["kvlf"]), st["kvlf"])
        vn = asset.data.root_lin_vel_w[:, :2] + P.kv_noise * torch.randn(env.num_envs, 2, device=env.device) + st["kvb"] + st["kvlf"]
        st["vbuf"][fresh] = vn[fresh].unsqueeze(1)
        st["vbuf"][:, st["vi"]] = vn; st["vi"] = (st["vi"] + 1) % st["vbuf"].shape[1]
        st["vbar"] = st["vbuf"].mean(1).norm(dim=-1)
    if P.stop_v > 0.0 and P.v_box_s > 0.0:
        v_est = st["vbar"]
    elif P.stop_v > 0.0:
        v_est = (asset.data.root_lin_vel_b[:, :2] + P.kv_noise * torch.randn(env.num_envs, 2, device=env.device)).norm(dim=1)
    orc = st["oracle_ok"]
    if P.stop_oracle >= 2.0:                                    # 속도 오라클(참값, 특권): 회복 착지 뒤 0.5 s 안 & 참 주기 평균 속도 < 0.07
        vt = asset.data.root_lin_vel_w[:, :2]
        st["vtbuf"][fresh] = vt[fresh].unsqueeze(1)
        st["vtbuf"][:, st["vti"]] = vt; st["vti"] = (st["vti"] + 1) % st["vtbuf"].shape[1]
        rtd = st["td_rec"].any(1)
        st["since_rtd"] = torch.where(rtd, torch.zeros_like(st["since_rtd"]), st["since_rtd"] + env.step_dt)
        st["since_rtd"] = torch.where(fresh, torch.full_like(st["since_rtd"], 1e9), st["since_rtd"])
        orc = (st["since_rtd"] <= 0.5) & (st["vtbuf"].mean(1).norm(dim=-1) < 0.07)
    want = c2_core(st, tilt, w, armed, td_any, env.step_dt, P, v_est, oracle_ok=orc)
    if getattr(env, "_c2_march", False):
        want = torch.ones_like(want) if en is None else en.clone()     # 홀드(인계 전) 구간은 ge 0
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


def c2_tracker(env, env_ids, nominal_half_width: float = 0.10, cap_half_width: float = -1.0, b: float = 0.02, v_ref: float = 0.05, max_off: float = 0.25,
               mg_scale: float = 0.05, mg_vmin: float = 0.10, df_margin: float = 0.01, df_scale: float = 0.05):
    """mode="interval"(0 s), **w3_step_tracker 뒤**: 이번 틱 유효 착지(ev_td)마다 회복 중 여부·포획 목표 오차 기록(특권 CoM — 보상 전용)."""
    from . import mdp_hang as H
    st = _c2(env); s3 = W3._st(env)
    td = s3["ev_td"] > 0.5
    asset = env.scene["robot"]
    _fp0 = asset.data.body_pos_w[:, H._g21_bids(env), :2]
    _cont0 = env.scene.sensors["contact_forces"].data.net_forces_w[:, H._feet_body_ids(env), :].norm(dim=-1) > 5.0
    if "last_xy" not in st or st["last_xy"].shape[0] != env.num_envs:
        st["last_xy"] = _fp0.clone()
    if not bool(td.any()):
        st["td_rec"] = torch.zeros_like(st["td_rec"]); st["oracle_ok"] = torch.zeros_like(st["oracle_ok"])
        st["last_xy"] = torch.where(_cont0.unsqueeze(-1), _fp0, st["last_xy"]); return
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
    if cap_half_width > 0.0:      # 레드팀 c: 포획 목표 y 명목 폭(실측 발 간격 반폭 ≈ 0.18) — 기본 −1 = W3 명목 0.10(종전)
        nom_y = nom_y + (torch.tensor([cap_half_width, -cap_half_width], device=env.device) - torch.tensor([nominal_half_width, -nominal_half_width], device=env.device)).unsqueeze(0)
    tx, ty = capture_target_rel(nom_x, nom_y, vy, om, b=b, v_ref=v_ref, max_off=max_off)
    err = torch.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
    rec = ((st["mode"] == REC) | (st["mode"] == STOPREQ)) & W3._ge(env) & _c2_mask(env)
    st["td_rec"] = td & rec.unsqueeze(1)
    st["cap_err"] = torch.where(st["td_rec"], err, torch.zeros_like(err))
    xi = com[:, :2] + vc[:, :2] / om.unsqueeze(1)                 # 여유형(보상 옵션·진단용으로 항상 기록)
    mg = capture_margin_score(fp[..., :2], xi, vc[:, :2], scale=mg_scale, v_min=mg_vmin)
    st["cap_mg"] = torch.where(st["td_rec"], mg, torch.zeros_like(mg))
    # 차분형(1006 문헌 권고 1): Φ = max(0, dist(ξ_eq, 지지영역) − 여유), 점수 = clip((Φ_착지전 − Φ_착지후)/scale, 0, 1).
    # ξ_eq = ξ + F/(m·ω0²) — 지속 외력 F(월드 수평, 특권)가 있으면 정적 평형에 필요한 CoP 가 CoM 보다 F 방향으로 F/(mω0²) 앞
    # (LIPM ẍ = ω0²(x − p) + F/m → 평형 p = x + F/(mω0²)). 권고문의 '뺀다'는 F 를 반력 기준으로 본 표기 — 부호는 이 유도를 따름.
    hs = H._state(env)
    Fw = H._dist_force_w(hs, torch.arange(env.num_envs, device=env.device))
    mtot = asset.data.default_mass.sum(1).to(Fw.device)
    xi_eq = xi + Fw / (mtot * om ** 2).unsqueeze(1)
    sensor = env.scene.sensors["contact_forces"]
    cont = sensor.data.net_forces_w[:, H._feet_body_ids(env), :].norm(dim=-1) > 5.0
    corners = foot_corners(fp[..., :2], yaw)
    corners_prev = foot_corners(st["last_xy"], yaw)                     # 착지 발의 이륙 전(마지막 접촉) 위치
    df, pb, pa = capture_diff_score(xi_eq, corners, cont, td, margin=df_margin, scale=df_scale, corners_prev=corners_prev)
    st["cap_df"] = torch.where(st["td_rec"], df, torch.zeros_like(df))
    st["phi_b"] = torch.where(st["td_rec"], pb, torch.zeros_like(pb)); st["phi_a"] = torch.where(st["td_rec"], pa, torch.zeros_like(pa))
    st["oracle_ok"] = (st["td_rec"] & (pa <= 0.0)).any(1)
    st["last_xy"] = torch.where(cont.unsqueeze(-1) & ~td.unsqueeze(-1), fp[..., :2], st["last_xy"])   # 착지 틱 발은 다음 틱부터 갱신


# ---------------------------------------------------------------- 보상 ----------------------------------------------------------------
def c2_place_reward(env, sig: float = 0.04, sig_cap: float = 0.05, cap_mode: str = "err") -> torch.Tensor:
    """w3_place_reward 대체: 회복 착지 = 포획 점수(cap_mode "err" = 목표 오차형(종전 기본) / "margin" = 이동 방향 포획 여유형), 나머지 = W3 명목(같은 값)."""
    s3 = W3._st(env); st = _c2(env)
    cs = st["cap_mg"] if cap_mode == "margin" else (st["cap_df"] if cap_mode == "diff" else None)
    return place_value(s3["ev_td"], s3["td_err"], st["td_rec"], st["cap_err"], W3._ge(env), sig, sig_cap, cap_score=cs)


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
