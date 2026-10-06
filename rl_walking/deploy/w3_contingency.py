"""W3 걷기 세션 2 차 대비책(1007 강화 연구, **기본 꺼짐 · 실물 미검증 · 젯슨 미배포**) — 약점 대장 reports/gen_w3/CONTINGENCY.md.

브리지가 `--w3-monitor` 를 명시해야 만들어진다(없으면 BridgeCore.w3mon = None → 이 모듈은 import 만 되고 어떤 경로도 타지 않는다).
순수 파이썬·numpy 객체: 시간 = 틱 수 × dt(브리지 50 Hz). 판단만 하고, 실행(걷기 정지·켜기 거부·고리 끄기)은 호출자가 한다.

감시 항목(설정 키 0 = 그 항목 끔):
  A 자세 경계   걷는 중(clock.want ∨ ge) **한 주기 저역(EMA τ walk_tau_s)** 기울기의 걷기 시작 전 서기 평균 대비 편차 > walk_tilt_deg 가
                walk_tilt_s 지속, 또는 |ω_xy| 저역 RMS > walk_w 가 walk_w_s 지속 → 걷기 정지(명령 0 + ge 0, 반주기 경계) = 같은 정책의 STAND.
                (순간값은 쓰지 않는다 — 시뮬 정상 걷기에서 좌우 체중 이동 롤이 순간 편차 p99 25°·|ω_xy| 중앙 0.9 rad/s, cont_trace_ana)
  B 접촉 이상   걷는 중: 한 발 연속 무접촉 > swing_max_s · 양발 무접촉 > both_off_s · 한 발 접촉 전이 수(창 chatter_win_s) > chatter_n
                → 걷기 정지. 서 있을 때: 최근 stand_chatter_win_s 안 전이 ≥ stand_chatter_n → 켜기 거부(걷기 금지 상태).
                접촉 신선도(노드가 주는 발 센서 나이) > contact_max_age_s → 걷는 중 정지 / 서 있을 때 켜기 거부.
                **고장 값을 0/1 로 바꾸지 않는다** — 판단만 하고 정책 입력은 건드리지 않는다(정책 계약 불변).
  C 실효 지연   노드가 최종 송출 목표(관절 프레임, LPF 뒤)와 엔코더를 넘기면 창 lat_win_s 의 1 차 차분 교차상관으로
                목표→엔코더 지연[틱]을 추정(관절 lat_joints, 여기 부족하면 추정 없음). 추정 ≥ lat_max_ticks 가 lat_confirm 번 연속
                → 걷는 중 정지 / 서 있을 때 켜기 거부. 같은 추정식으로 시뮬 지연–추정 대응표와 실물 블랙박스를 잰다(문서 §지연).
  D 요 표류     서 있고 조용할 때(양발·|ω_xy| < 0.1) IMU 요를 펼쳐 최근 yaw_win_s 선형 기울기 r[°/min] 추정(구간 ≥ yaw_min_span_s).
                |r| > yaw_rate_max_dpm → 헤딩 고리 금지(세션 끝까지, 진단 사유 기록). 아니면 고리 타임아웃을 yaw_budget_deg/|r| 분으로 줄임.
                요 점프: |wrap(Δψ) − ω_z·dt| > yaw_jump_deg 한 틱 → 고리 금지 + 고리 걷기 중이면 정지.
  E 고리 wz 상한 병진 명령(vx 또는 vy ≠ 0) 중 헤딩 고리 출력 |wz| ≤ hl_wz_translate (W3S3d 는 단일 축 학습 — 병진+회전은 학습 밖).
  F 걷기 시간   연속 걷기 > walk_max_s → 정지.
  G 정지 정착   걷기 → 서기(ge 1→0) 뒤 settle_s 안에 양발 접촉 ∧ |ω_xy| < settle_w 가 settle_hold_s 이어지지 않으면 기록(note).

mode: "act" = 위 조치 실행, "note" = 같은 판정을 기록만(guard_mode data 와 같은 뜻 — 오트립 계수용). 어느 쪽이든 사건은 events 에 쌓인다.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class W3MonitorConfig:
    mode: str = "act"                 # "act" | "note"
    dt: float = 0.02
    # A 자세 경계 (걷는 중) — 문턱은 시뮬 정상 걷기 분포의 상단 + 여유(CONTINGENCY §A 표), 실물 미측정
    walk_tau_s: float = 0.91          # 저역 시정수 = 한 보행 주기(1.1 Hz)
    walk_tilt_deg: float = 8.0        # [보정 전 가정] — §A 표로 확정
    walk_tilt_s: float = 0.2
    walk_w: float = 1.8               # |ω_xy| 저역 RMS [rad/s] [보정 전 가정]
    walk_w_s: float = 0.2
    # B 접촉
    swing_max_s: float = 0.8          # 반주기 0.455 s 의 약 1.75 배
    both_off_s: float = 0.2
    chatter_win_s: float = 1.0
    chatter_n: int = 6                # 정상 걷기 = 한 발 2 전이/주기(0.91 s)
    stand_chatter_win_s: float = 2.0
    stand_chatter_n: int = 2
    contact_max_age_s: float = 0.25   # 발 센서 나이(10 Hz → 2 펄스 결손). 지속 요구 없음 — 노드 FOOT_STALE_SEC 0.35 하드 정지 전에 걷기만 멈춤(간헐 결손 대비)
    # C 실효 지연
    lat_max_ticks: float = 0.0        # 0 = 끔
    lat_win_s: float = 2.0
    lat_every: int = 10
    lat_confirm: int = 3
    lat_maxlag: int = 15
    lat_min_std: float = 0.002        # 목표 1 차 차분 표준편차 [rad/틱] 이 이 값 미만인 관절은 제외(들뜸 부족)
    lat_min_corr: float = 0.5
    lat_min_joints: int = 3           # 들뜸 문턱을 넘은 관절이 이만큼 있어야 추정(실물 조용한 서기 2 관절 창에서 8–14 틱 가짜 추정 — §C)
    lat_walk_only: bool = True        # 걷는 중(보행 들뜸)에만 추정·판정 — 서 있을 때는 직전 걷기의 판정(lat_bad)으로 켜기만 거부
    # 스택 지연(0930 적체형): stage8 이 잰 명령 발행→처리 1 s 중앙 나이·피드백(status) 나이 — 노드가 넘긴다
    s8_age_max_ms: float = 40.0       # 실물 1001–1004 중앙 p99 ≤ 13.6 ms, 0929 적체 ≈ 200 ms
    fb_age_max_s: float = 0.08        # 실물 status_age p99 ≤ 0.015 s, 최대 0.049 s (하드 정지 SENSOR_STALE 0.2 s)
    health_hold_s: float = 0.3        # 센서 나이 판정 지속 시간(한 틱 튐 무시)
    # D 요 표류
    yaw_rate_max_dpm: float = 2.5
    yaw_budget_deg: float = 5.0
    yaw_win_s: float = 60.0
    yaw_min_span_s: float = 20.0
    yaw_jump_deg: float = 3.0          # 한 틱 |Δψ − ω_z·dt| — 1° 는 실물 틱 지터(최대 ~60 ms)에서 가짜 점프 1 건(1001 재생), 3° = 150°/s 불일치
    # E 고리 wz 상한(병진 중)
    hl_wz_translate: float = 0.0      # 0 = 끔(고리 기본 ±0.4 그대로)
    # F 걷기 시간
    walk_max_s: float = 0.0           # 0 = 끔
    # G 정지 정착(기록만)
    settle_s: float = 3.0
    settle_w: float = 0.5
    settle_hold_s: float = 0.5


#: 범위 — 문턱을 성공률 목적으로 풀 수 없게(넘으면 기동 거부). 0 허용 키 = 그 항목 끔.
W3_MON_RANGES = {
    "walk_tau_s": (0.2, 3.0), "walk_tilt_deg": (0.0, 20.0), "walk_tilt_s": (0.02, 1.0), "walk_w": (0.0, 4.0), "walk_w_s": (0.02, 1.0),
    "swing_max_s": (0.0, 2.0), "both_off_s": (0.0, 1.0), "chatter_win_s": (0.2, 5.0), "chatter_n": (0, 30),
    "stand_chatter_win_s": (0.2, 10.0), "stand_chatter_n": (0, 30), "contact_max_age_s": (0.0, 0.35),
    "lat_max_ticks": (0.0, 15.0), "lat_win_s": (0.5, 10.0), "lat_every": (1, 100), "lat_confirm": (1, 20),
    "lat_maxlag": (4, 30), "lat_min_std": (0.0, 0.05), "lat_min_corr": (0.0, 0.99), "lat_min_joints": (1, 4),
    "s8_age_max_ms": (0.0, 200.0), "fb_age_max_s": (0.0, 0.2), "health_hold_s": (0.0, 2.0),
    "yaw_rate_max_dpm": (0.0, 30.0), "yaw_budget_deg": (1.0, 15.0), "yaw_win_s": (10.0, 600.0),
    "yaw_min_span_s": (5.0, 300.0), "yaw_jump_deg": (0.0, 10.0),
    "hl_wz_translate": (0.0, 0.4), "walk_max_s": (0.0, 600.0),
    "settle_s": (0.5, 10.0), "settle_w": (0.1, 2.0), "settle_hold_s": (0.1, 3.0),
}
_INT_KEYS = {"chatter_n", "stand_chatter_n", "lat_every", "lat_confirm", "lat_maxlag", "lat_min_joints"}
_BOOL_KEYS = {"lat_walk_only"}

#: 지연 추정 관절(W3 걷기에서 크게 움직이는 힙 피치·무릎) — 이름은 JOINT_ORDER 기준
LAT_JOINTS = ("left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint")


def parse_w3_monitor(text):
    """None/''/'off' → None(끔) · 'on' → 기본 · 'note' → 기본+기록만 · JSON 객체 → 덮어쓰기(모르는 키·범위 밖 ValueError)."""
    if text is None or str(text).strip() in ("", "off"):
        return None
    s = str(text).strip()
    if s == "on":
        return W3MonitorConfig()
    if s == "note":
        return W3MonitorConfig(mode="note")
    d = json.loads(s)
    if not isinstance(d, dict):
        raise ValueError("W3 감시 설정은 'on'|'note'|'off' 또는 JSON 객체")
    kw = {}
    for k, v in d.items():
        if k == "mode":
            if v not in ("act", "note"):
                raise ValueError(f"W3 감시 mode {v!r} — act|note")
            kw[k] = v
            continue
        if k in _BOOL_KEYS:
            if not isinstance(v, bool):
                raise ValueError(f"W3 감시 {k}: true/false")
            kw[k] = v
            continue
        if k not in W3_MON_RANGES:
            raise ValueError(f"W3 감시: 모르는 키 {k!r} (허용 mode, {', '.join(sorted(W3_MON_RANGES))})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"W3 감시 {k}: 유한 수 필요 (받음 {v!r})")
        lo, hi = W3_MON_RANGES[k]
        if not lo <= float(v) <= hi:
            raise ValueError(f"W3 감시 {k}={v} 범위 밖 [{lo}, {hi}]")
        if k in _INT_KEYS and float(v) != int(v):
            raise ValueError(f"W3 감시 {k}: 정수 필요")
        kw[k] = int(v) if k in _INT_KEYS else float(v)
    return W3MonitorConfig(**kw)


def _wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


def _rp_deg(grav):
    """투영중력 → (롤, 피치) [deg] — c5_switch.roll_pitch_deg 와 같은 정의(롤 = atan2(g_y, −g_z), 피치 = atan2(g_x, −g_z))."""
    gx, gy, gz = (float(v) for v in grav)
    return math.degrees(math.atan2(gy, -gz)), math.degrees(math.atan2(gx, -gz))


def lag_xcorr(tgt, enc, maxlag=15, min_std=0.002, min_corr=0.5, min_joints=1):
    """목표·엔코더 (T, J) → (지연 틱 | None, 상관, 쓴 관절 수). 1 차 차분 정규 교차상관을 관절 합산(공분산 합 / 표준편차 곱 합).
    lag L = enc[t] 가 tgt[t − L] 을 가장 닮는 L ∈ [0, maxlag]. 들뜸 부족·상관 미달이면 None."""
    a = np.diff(np.asarray(tgt, dtype=np.float64), axis=0)
    b = np.diff(np.asarray(enc, dtype=np.float64), axis=0)
    T = a.shape[0]
    if T < maxlag + 10:
        return None, float("nan"), 0
    use = np.std(a, axis=0) >= min_std
    if int(use.sum()) < max(1, int(min_joints)):
        return None, float("nan"), 0
    a = a[:, use] - a[:, use].mean(0)
    b = b[:, use] - b[:, use].mean(0)
    best, bc = None, -2.0
    for L in range(0, maxlag + 1):
        x, y = a[: T - L], b[L:]
        num = float(np.sum(x * y))
        den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        c = num / den if den > 0 else -2.0
        if c > bc:
            best, bc = L, c
    if bc < min_corr:
        return None, bc, int(use.sum())
    return best, bc, int(use.sum())


class W3WalkMonitor:
    """걷기 세션 감시기. tick() 은 정책 틱마다 1 회(러너 계산 전), track()·health() 는 노드가 같은 틱에 부른다."""

    def __init__(self, cfg: W3MonitorConfig, joint_order=None):
        self.cfg = cfg
        self.act = cfg.mode == "act"
        jo = list(joint_order) if joint_order is not None else None
        self.lat_idx = [jo.index(j) for j in LAT_JOINTS] if jo is not None else [0, 1, 2, 3]
        self.events = []                 # 노드가 비운다: {"event": "w3_mon", "kind", "why", "action", "t"}
        self.reset_session()

    # ---- 세션(프로세스) 수명 상태: 요 표류 추정·고리 금지는 정책 재시작에도 유지 ----
    def reset_session(self):
        self.n = 0
        self.hl_ban = None               # 고리 금지 사유(세션 끝까지)
        self.yaw_rate_dpm = None
        self._yq = deque()               # (t, 펼친 요 rad, 구간 번호) — 조용한 서기 표본
        self._yq_seg = 0
        self._yq_in = False
        self._yaw_unwrap = None
        self._yaw_prev = None
        self._lat_buf = deque(maxlen=max(16, int(round(self.cfg.lat_win_s / self.cfg.dt)) + 1))
        self.lat_est = None
        self.lat_corr = None
        self.lat_hits = 0
        self.lat_bad = False
        self.health_bad = None
        self._hb_t0 = None
        self.n_trips = {}
        self.reset_walk_state()

    def reset_walk_state(self):
        self.walking = False
        self.t_walk0 = None
        self.rp_ref = None
        self.tilt_t0 = None
        self.w_t0 = None
        self.off_t0 = [None, None]
        self.both_off_t0 = None
        self.trans = (deque(), deque())
        self.trans_stand = (deque(), deque())   # 서기(걷기·래치 해제 대기 아님) 중 전이만 — 걷기 직후 켜기 판정에 걷기 전이를 섞지 않는다
        self.c_prev = None
        self.t_stop = None
        self.settle_t0 = None
        self.stop_pending = None         # act 모드 걷기 정지 요청(호출자가 실행하고 지운다)
        self.last_dev = None
        self.last_w = None
        self.last_wrms = None
        self._stopped_this_walk = False
        self._rp_lp = None               # 기울기 (롤, 피치) EMA — 서기·걷기 내내 갱신(걷기 시작 기준 = 그 순간 값 = 서기 평균)
        self._w2_lp = None

    @property
    def t(self):
        return self.n * self.cfg.dt

    def _emit(self, kind, why, action):
        self.n_trips[kind] = self.n_trips.get(kind, 0) + 1
        self.events.append({"event": "w3_mon", "kind": kind, "why": why, "action": action if self.act else f"note:{action}",
                            "t": round(self.t, 3)})

    def _stop(self, kind, why):
        """걷기 정지 판정 — act 면 요청을 세우고(호출자 실행), note 면 기록만. 같은 걷기에서 한 번만."""
        if self.stop_pending is not None or self._stopped_this_walk:
            return
        self._stopped_this_walk = True
        self._emit(kind, why, "stop_walk")
        if self.act:
            self.stop_pending = (kind, why)

    # ---- 매 정책 틱 ----
    def tick(self, *, walking, contact2, gyro, grav, yaw, hl_latched=False):
        c = self.cfg
        self.n += 1
        t = self.t
        cl, cr = bool(contact2[0] > 0.5), bool(contact2[1] > 0.5)
        try:
            w = math.hypot(float(gyro[0]), float(gyro[1]))
            wz = float(gyro[2])
        except (TypeError, ValueError, IndexError):
            w, wz = float("nan"), float("nan")
        rp = _rp_deg(grav)
        self.last_w = w
        a_ = min(1.0, c.dt / c.walk_tau_s)
        if self._rp_lp is None:
            self._rp_lp = rp
        else:
            self._rp_lp = (self._rp_lp[0] + a_ * (rp[0] - self._rp_lp[0]), self._rp_lp[1] + a_ * (rp[1] - self._rp_lp[1]))
        if math.isfinite(w):
            self._w2_lp = w * w if self._w2_lp is None else self._w2_lp + a_ * (w * w - self._w2_lp)
        # 걷기 전이
        if walking and not self.walking:
            self.walking, self.t_walk0, self.rp_ref = True, t, self._rp_lp
            self.tilt_t0 = self.w_t0 = self.both_off_t0 = None
            self.off_t0 = [None, None]
            self.t_stop = self.settle_t0 = None
            self._stopped_this_walk = False
        elif not walking and self.walking:
            self.walking = False
            self.t_stop, self.settle_t0 = t, None
            self.stop_pending = None
        # 접촉 전이 기록(서기·걷기 공통)
        cc = (cl, cr)
        if self.c_prev is not None:
            for k in (0, 1):
                if cc[k] != self.c_prev[k]:
                    self.trans[k].append(t)
                    if not self.walking:
                        self.trans_stand[k].append(t)
        self.c_prev = cc
        keep = max(c.chatter_win_s, c.stand_chatter_win_s)
        for k in (0, 1):
            while self.trans[k] and self.trans[k][0] < t - keep - 1e-9:
                self.trans[k].popleft()
            while self.trans_stand[k] and self.trans_stand[k][0] < t - keep - 1e-9:
                self.trans_stand[k].popleft()
        # D 요: 점프·표류 표본
        self._yaw_step(t, yaw, wz, cl and cr and math.isfinite(w) and w < 0.1 and not walking, hl_latched and walking, w)
        if self.walking:
            self._walk_checks(t, rp, w, cc)
        else:
            self._settle_check(t, w, cl and cr)

    def _walk_checks(self, t, rp, w, cc):
        c = self.cfg
        rp_lp = self._rp_lp
        dev = math.hypot(rp_lp[0] - self.rp_ref[0], rp_lp[1] - self.rp_ref[1])
        self.last_dev = dev
        w_inst = w
        w = math.sqrt(self._w2_lp) if (self._w2_lp is not None and math.isfinite(w_inst)) else float("nan")
        self.last_wrms = w
        if c.walk_tilt_deg > 0:
            if dev > c.walk_tilt_deg:
                self.tilt_t0 = t if self.tilt_t0 is None else self.tilt_t0
                if t - self.tilt_t0 >= c.walk_tilt_s - 1e-9:
                    self._stop("tilt", f"걷기 중 저역 기울기 편차 {dev:.1f}° > {c.walk_tilt_deg:g}° ({c.walk_tilt_s:g}s)")
            else:
                self.tilt_t0 = None
        if c.walk_w > 0:
            if not math.isfinite(w) or w > c.walk_w:
                self.w_t0 = t if self.w_t0 is None else self.w_t0
                if t - self.w_t0 >= c.walk_w_s - 1e-9:
                    self._stop("w_xy", f"걷기 중 |ω_xy| 저역 RMS {w:.2f} > {c.walk_w:g} rad/s ({c.walk_w_s:g}s)")
            else:
                self.w_t0 = None
        for k in (0, 1):
            if not cc[k]:
                self.off_t0[k] = t if self.off_t0[k] is None else self.off_t0[k]
                if c.swing_max_s > 0 and t - self.off_t0[k] >= c.swing_max_s - 1e-9:
                    self._stop("swing_long", f"{'왼' if k == 0 else '오른'}발 무접촉 {t - self.off_t0[k]:.2f}s > {c.swing_max_s:g}s")
            else:
                self.off_t0[k] = None
        if not cc[0] and not cc[1]:
            self.both_off_t0 = t if self.both_off_t0 is None else self.both_off_t0
            if c.both_off_s > 0 and t - self.both_off_t0 >= c.both_off_s - 1e-9:
                self._stop("both_off", f"걷기 중 양발 무접촉 {t - self.both_off_t0:.2f}s")
        else:
            self.both_off_t0 = None
        if c.chatter_n > 0:
            for k in (0, 1):
                n = sum(1 for x in self.trans[k] if x > t - c.chatter_win_s + 1e-9)
                if n > c.chatter_n:
                    self._stop("chatter", f"{'왼' if k == 0 else '오른'}발 접촉 전이 {n} 회/{c.chatter_win_s:g}s > {c.chatter_n}")
        if c.walk_max_s > 0 and t - self.t_walk0 >= c.walk_max_s - 1e-9:
            self._stop("walk_max", f"연속 걷기 {t - self.t_walk0:.1f}s ≥ 상한 {c.walk_max_s:g}s")
        if self.lat_bad:
            self._stop("latency", f"실효 지연 추정 {self.lat_est} 틱 ≥ {c.lat_max_ticks:g} ({c.lat_confirm} 회 연속)")
        if self.health_bad is not None:
            self._stop("health", self.health_bad)

    def _settle_check(self, t, w, both):
        c = self.cfg
        if self.t_stop is None:
            return
        if both and math.isfinite(w) and w < c.settle_w:
            self.settle_t0 = t if self.settle_t0 is None else self.settle_t0
            if t - self.settle_t0 >= c.settle_hold_s - 1e-9:
                self.t_stop = None                       # 정착 확인
                return
        else:
            self.settle_t0 = None
        if t - self.t_stop >= c.settle_s - 1e-9:
            self._emit("settle", f"걷기 정지 뒤 {c.settle_s:g}s 안에 정착(양발·|ω_xy| < {c.settle_w:g} {c.settle_hold_s:g}s) 안 됨", "note")
            self.t_stop = None

    def _yaw_step(self, t, yaw, wz, quiet_stand, hl_walk, w_xy=0.0):
        c = self.cfg
        if yaw is None or not math.isfinite(float(yaw)):
            if hl_walk:
                self._stop("yaw_nan", "IMU 요 비유한")
            if self.hl_ban is None:
                self.hl_ban = "IMU 요 비유한"
                self._emit("hl_ban", self.hl_ban, "hl_off")
            return
        yaw = float(yaw)
        if self._yaw_prev is not None:
            d = _wrap(yaw - self._yaw_prev)
            self._yaw_unwrap += d
            if (c.yaw_jump_deg > 0 and math.isfinite(wz) and math.isfinite(w_xy) and w_xy < 0.5
                    and abs(d - wz * c.dt) > math.radians(c.yaw_jump_deg)):
                # 몸통이 크게 기우는 중(|ω_xy| ≥ 0.5)에는 몸통 z 자이로 ≠ 월드 요 속도 — 실물 낙하·밀기에서 1.6–2.8° 가짜 점프(재생 3 건)
                why = f"IMU 요 점프 {math.degrees(d):+.2f}° (자이로 적분 {math.degrees(wz * c.dt):+.2f}°)"
                if self.hl_ban is None:
                    self.hl_ban = why
                    self._emit("hl_ban", why, "hl_off")
                if hl_walk:
                    self._stop("yaw_jump", why)
        else:
            self._yaw_unwrap = yaw
        self._yaw_prev = yaw
        # 정지 표류: 조용한 서기 구간(연속 run)마다 따로 평균을 뺀 합동 기울기 — 구간 사이의 실제 회전(밀기·회복 스텝)은 기울기에 안 들어간다.
        if quiet_stand:
            if not self._yq_in:
                self._yq_seg += 1
            self._yq.append((t, self._yaw_unwrap, self._yq_seg))
        self._yq_in = bool(quiet_stand)
        while self._yq and self._yq[0][0] < t - c.yaw_win_s:
            self._yq.popleft()
        if self.n % 25 == 0 and len(self._yq) >= 10:
            arr = np.array(self._yq)
            num = den = span = 0.0
            for sg in np.unique(arr[:, 2]):
                s_ = arr[arr[:, 2] == sg]
                if s_[-1, 0] - s_[0, 0] < 3.0 - 1e-9:      # 3 s 미만 조각은 버린다(정착 과도)
                    continue
                tt = s_[:, 0] - s_[:, 0].mean()
                num += float(np.sum(tt * (s_[:, 1] - s_[:, 1].mean()))); den += float(np.sum(tt * tt)); span += s_[-1, 0] - s_[0, 0]
            if span < c.yaw_min_span_s - 1e-9 or den <= 0.0:
                return
            r = num / den                                   # rad/s
            self.yaw_rate_dpm = math.degrees(r) * 60.0
            if c.yaw_rate_max_dpm > 0 and abs(self.yaw_rate_dpm) > c.yaw_rate_max_dpm and self.hl_ban is None:
                self.hl_ban = f"IMU 요 정지 표류 {self.yaw_rate_dpm:+.2f}°/min > {c.yaw_rate_max_dpm:g}"
                self._emit("hl_ban", self.hl_ban, "hl_off")

    # ---- 노드 입력(LPF 뒤 송출 목표·엔코더, 센서 나이) ----
    def track(self, target_rad, qpos_rad):
        c = self.cfg
        if c.lat_max_ticks <= 0:
            return
        tg = np.asarray(target_rad, dtype=np.float64)[self.lat_idx]
        q = np.asarray(qpos_rad, dtype=np.float64)[self.lat_idx]
        self._lat_buf.append(np.concatenate([tg, q]))
        if c.lat_walk_only and not self.walking:
            self._lat_buf.clear()                     # 걷기 들뜸 창만(서기·전이 표본 섞지 않음)
            return
        if self.n % c.lat_every != 0 or len(self._lat_buf) < self._lat_buf.maxlen:
            return
        arr = np.asarray(self._lat_buf)
        k = len(self.lat_idx)
        L, corr, _nj = lag_xcorr(arr[:, :k], arr[:, k:], c.lat_maxlag, c.lat_min_std, c.lat_min_corr, c.lat_min_joints)
        self.lat_corr = None if not math.isfinite(corr) else round(corr, 3)
        if L is None:
            return                                   # 들뜸 부족 — 판정 보류(연속 계수 유지)
        self.lat_est = L
        if L >= c.lat_max_ticks - 1e-9:
            self.lat_hits += 1
            if self.lat_hits >= c.lat_confirm and not self.lat_bad:
                self.lat_bad = True
                self._emit("latency", f"실효 지연 추정 {L} 틱(상관 {corr:.2f}) ≥ {c.lat_max_ticks:g} — {c.lat_confirm} 회 연속", "inhibit")
        else:
            self.lat_hits = 0
            self.lat_bad = False

    def health(self, foot_age_s=None, fb_age_s=None, s8_p50_ms=None):
        """노드 입력(매 틱, tick() 뒤): 발 센서 나이·stage8 피드백 나이·stage8 명령 발행→처리 1 s 중앙 나이. None = 그 입력 없음(판정 안 함).
        문턱 초과가 health_hold_s 지속되면 걷기 금지(걷는 중 정지·켜기 거부), 모두 정상이면 즉시 해제."""
        c = self.cfg

        def over(v, thr):
            return thr > 0 and v is not None and (not math.isfinite(float(v)) or float(v) > thr)
        rs, now_bad = [], None
        if over(foot_age_s, c.contact_max_age_s):          # 즉시(지속 요구 없음)
            now_bad = f"발 센서 나이 {foot_age_s:.3f}s > {c.contact_max_age_s:g}s(오래된 접촉 — 0/1 대체 없이 걷기 금지)"
        if over(fb_age_s, c.fb_age_max_s):
            rs.append(f"관절 피드백 나이 {fb_age_s:.3f}s > {c.fb_age_max_s:g}s")
        if over(s8_p50_ms, c.s8_age_max_ms):
            rs.append(f"stage8 명령 처리 나이 중앙 {s8_p50_ms:.1f} ms > {c.s8_age_max_ms:g} ms(스택 지연 적체 의심)")
        if not rs:
            self._hb_t0 = None
        else:
            self._hb_t0 = self.t if self._hb_t0 is None else self._hb_t0
        held = bool(rs) and self.t - self._hb_t0 >= c.health_hold_s - 1e-9
        parts = ([now_bad] if now_bad else []) + (rs if held else [])
        if not parts:
            self.health_bad = None
            return
        bad = "; ".join(parts)
        if self.health_bad is None:
            self._emit("health", bad, "inhibit")
        self.health_bad = bad

    # ---- 조회 ----
    def enable_gate(self):
        """W3Supervisor.enable_preconditions 의 추가 조건(act 모드만 거부)."""
        if not self.act:
            return True, "ok"
        c = self.cfg
        if self.health_bad is not None:
            return False, f"감시: {self.health_bad}"
        if self.lat_bad:
            return False, f"감시: 실효 지연 추정 {self.lat_est} 틱 ≥ {c.lat_max_ticks:g}"
        if c.stand_chatter_n > 0:
            t = self.t
            for k in (0, 1):
                n = sum(1 for x in self.trans_stand[k] if x > t - c.stand_chatter_win_s + 1e-9)
                if n >= c.stand_chatter_n:
                    return False, f"감시: 서기 중 {'왼' if k == 0 else '오른'}발 접촉 전이 {n} 회/{c.stand_chatter_win_s:g}s(채터·패드 의심)"
        return True, "ok"

    def hl_allowed(self):
        if not self.act or self.hl_ban is None:
            return True, "ok"
        return False, self.hl_ban

    def hl_timeout_s(self, base_s):
        """요 표류율로 줄인 고리 타임아웃 = min(기본, 예산°/|r| 분). 추정 없으면 기본."""
        if not self.act or self.yaw_rate_dpm is None or abs(self.yaw_rate_dpm) < 1e-6:
            return base_s
        return min(base_s, 60.0 * self.cfg.yaw_budget_deg / abs(self.yaw_rate_dpm))

    def hl_wz_cap(self, vx, vy):
        c = self.cfg
        if not self.act or c.hl_wz_translate <= 0 or (vx == 0.0 and vy == 0.0):
            return None
        return c.hl_wz_translate

    def take_stop(self):
        s, self.stop_pending = self.stop_pending, None
        return s

    def snapshot(self):
        def r(v, n):
            return None if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)) else round(float(v), n)
        return {"mode": self.cfg.mode, "walking": self.walking, "dev_deg": r(self.last_dev, 2), "w_xy": r(self.last_w, 3), "w_rms": r(self.last_wrms, 3),
                "lat": self.lat_est, "lat_corr": self.lat_corr, "lat_bad": self.lat_bad,
                "yaw_rate_dpm": r(self.yaw_rate_dpm, 3), "hl_ban": self.hl_ban, "health": self.health_bad,
                "trips": dict(self.n_trips)}


def config_dict(cfg: W3MonitorConfig) -> dict:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


# ------------------------------------------------------------------ C5 회복 모드 체류 상한(기록·선택적 동결) — 회복 세션 전용
class RecoveryOvertime:
    """C5 회복 모드(1) 연속 체류 > max_s → 사건 1 회(같은 활성화에서). action 'note'(기록) | 'freeze'(호출자가 지연 동결).
    근거: SwitchDetector 에 회복 모드 시간 상한이 없다(끄기 = calm 1.0 s 또는 조기 복귀뿐) — 패드 0 고착(off_contact both)·
    진동 지속 시 무기한 회복 정책. 시뮬 장기 체류(> 4 s) 3.8–8.3 % (STOPEST P4)."""

    def __init__(self, max_s=6.0, action="note", dt=0.02):
        if not (1.0 <= float(max_s) <= 30.0):
            raise ValueError(f"회복 체류 상한 {max_s} 범위 [1, 30] 밖")
        if action not in ("note", "freeze"):
            raise ValueError(f"회복 체류 조치 {action!r} — note|freeze")
        self.max_s, self.action, self.dt = float(max_s), action, float(dt)
        self.t_in = 0.0
        self.fired = False

    def update(self, mode):
        if int(mode) == 1:
            self.t_in += self.dt
            if self.t_in >= self.max_s - 1e-9 and not self.fired:
                self.fired = True
                return {"event": "c5_overtime", "why": f"회복 모드 {self.t_in:.2f}s ≥ 상한 {self.max_s:g}s", "action": self.action}
        else:
            self.t_in, self.fired = 0.0, False
        return None


def parse_c5_overtime(text):
    """None/'' → None · 'note:6' · 'freeze:6' 형식."""
    if text is None or str(text).strip() == "":
        return None
    s = str(text).strip()
    if ":" not in s:
        raise ValueError("--c5-overtime 형식 'note:초' 또는 'freeze:초'")
    a, v = s.split(":", 1)
    return RecoveryOvertime(max_s=float(v), action=a)
