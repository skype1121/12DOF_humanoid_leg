"""W3 실험 기능(1006, **기본 꺼짐·실물 미검증·젯슨 미배포**) — W3-A 400 제자리 교대 스텝 엔진의 배포 관측·명령 계약.

HTURN2(rl_walking/deploy/hturn2.py)와 같은 방식: 러너는 `w3=None`(기본)이면 이 모듈을 import 조차 하지 않아 종전과 비트 동일,
브리지는 `--experimental-w3` 를 명시해야 체크포인트 키(W3_CKPT_KEY)가 등록된다. 기본 구조에 영구 고정하지 않는다.
문서: reports/W3_BRIDGE_FLAG_1006.md · 동치 검증: rl_walking/deploy/selftest_w3_equiv.py (시뮬 원본 소스를 torch CPU 로 실행해 대조).

시뮬 원본(이 파일은 그 식을 numpy float32 로 옮긴 것):
  (a) rl_walking/w3.py w3_gait_update (mode interval 0 s = 매 env.step, 명령 갱신 뒤·관측 앞):
        want = env._ge_cmd (명령이 원하는 gait_enable)
        fresh = episode_length_buf ≤ 1          (러너 틱 0·1)
        on    = want ∧ ¬ge                      (0→1: 즉시, 위상 0 = 왼발 스윙부터)
        u'    = u + f_step·dt                   (f_step 1.1 Hz, dt 0.02)
        cross = (u < 0.5 ∧ u' ≥ 0.5) ∨ u' ≥ 1   (반주기 경계 통과)
        off   = ¬want ∧ ge ∧ cross              (1→0: 반주기 경계에서만 — 스윙 중 끄지 않는 래치)
        ge    = fresh ? want : on ? 1 : off ? 0 : ge
        u     = (fresh ∨ on ∨ ¬ge) ? 0 : u' mod 1
        f     = ge ? f_step : 0
        ge_obs = fresh ? ge : clamp(ge_obs ± dt/0.2, 0, 1)   (0.2 s 램프)
  (b) rl_walking/mdp_stage4.gait_clock: [sin 2πu, cos 2πu] × ge_obs   (_gait_phase 는 env._gu 가 있으면 2π·_gu)
      → 서기(ge 0, 램프 끝) = (0, 0). 램프 내림 중에는 u = 0 이므로 (0, ge_obs).
  (c) 명령: Stage 0 = velocity_commands (0,0,0) + gait_enable. ge 는 관측에 따로 들어가지 않는다(클록 게이트로만 보인다).
  (d) 행동 체인 = 기존 stand 계열과 같음(러너 target → 러너 슬루 → 브리지 LPF β0.27 → 노드 슬루). W3 보조(assist)·참조(ref)는
      배포에 없다 — W3-A 400 은 보조 α 0·참조 0 평가 통과(LOG a8a5db22). 이 모듈에 행동 잔차는 없다.

틱 대응(러너 n번째 호출 = env episode_length_buf n): 관측 n 은 그 틱의 want_n 으로 래치를 한 번 갱신한 뒤 조립한다
(ManagerBasedRLEnv.step: buf+1 → 리셋 → command.compute(_ge_cmd 갱신) → interval(w3_gait_update) → 관측).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: 브리지 체크포인트 키(플래그를 켤 때만 등록). **stand 가족 접두사** — Stage 0 명령은 서기와 같은 (0,0,0)이라
#: stand 경로(heading_hold 강제 OFF·LPF β0.27·명령 0)를 그대로 탄다. 시작은 항상 STAND(ge 0), `step` 명령으로 켠다.
W3_CKPT_KEY = "stand_w3a_11699"
#: W3-A 400 학습 run (biped12_stepb0). exported/ 는 아직 없다 — 보고서의 내보내기 절차(이번 작업에서 실행하지 않음).
W3_RUN_REL = "logs/rsl_rl/biped12_stepb0/2026-10-06_11-40-53_w3A_s1_from_g21a"
W3_MODEL = "model_11699.pt"
W3_TASK = "Biped12-HangStand-W3A-v0"

_F32 = np.float32


@dataclass(frozen=True)
class W3Config:
    """W3 계약 상수 — 기본값 = hang_stand_env_cfg.Biped12HangStandW3BaseEnvCfg(w3_gait f_step 1.1) · w3_gait_update(ramp_s 0.2)."""
    f_step: float = 1.1          # STEP 위상 주파수 [Hz]
    ramp_s: float = 0.2          # 관측 게이트 램프 [s]
    dt: float = 0.005 * 4        # env.step_dt (sim dt 0.005 × decimation 4) — 시뮬과 같은 python float 식


W3_DEFAULT = W3Config()


class W3GaitClock:
    """gait_enable 래치 + 위상 + 관측 램프 — w3_gait_update 의 단일 env 판(float32).

    set_want(bool) 로 명령이 원하는 ge 를 주고, update(tick) 를 러너가 관측 조립 때 틱당 정확히 1회 부른다
    (tick = 러너 _tick, 조립 전 값 = env episode_length_buf).
    """

    def __init__(self, cfg: W3Config = W3_DEFAULT):
        self.cfg = cfg
        self._du = _F32(cfg.f_step * cfg.dt)                    # torch: float32 텐서 + python float → float32 연산
        self._k = _F32(cfg.dt / max(cfg.ramp_s, 1e-6))          # torch.where(ge, k, −k) → float32
        self.want = False
        self.reset()

    def reset(self) -> None:
        """에피소드 초기화 — 위상·래치·램프 0. want 도 False(서기)로: 브리지가 시작 뒤 step 명령으로만 켠다."""
        self.u = _F32(0.0)
        self.f = _F32(0.0)
        self.ge = False
        self.ge_obs = _F32(0.0)
        self.want = False
        self.events = []                 # 진단: (tick, "on"|"off") 래치 전이

    def set_want(self, want: bool) -> None:
        self.want = bool(want)

    def update(self, tick: int) -> None:
        fresh = int(tick) <= 1
        want, ge = self.want, self.ge
        on = want and not ge
        u_next = _F32(self.u + self._du)
        cross = (self.u < _F32(0.5) and u_next >= _F32(0.5)) or (u_next >= _F32(1.0))
        off = (not want) and ge and cross
        if fresh:
            ge_new = want
        elif on:
            ge_new = True
        elif off:
            ge_new = False
        else:
            ge_new = ge
        if ge_new != ge:
            self.events.append((int(tick), "on" if ge_new else "off"))
        reset_u = fresh or on or (not ge_new)
        self.u = _F32(0.0) if reset_u else _F32(np.remainder(u_next, _F32(1.0)))
        self.f = _F32(self.cfg.f_step) if ge_new else _F32(0.0)
        self.ge = ge_new
        if fresh:
            self.ge_obs = _F32(1.0) if ge_new else _F32(0.0)
        else:
            self.ge_obs = _F32(min(max(_F32(self.ge_obs + (self._k if ge_new else -self._k)), _F32(0.0)), _F32(1.0)))

    def clock_obs(self) -> np.ndarray:
        """[sin 2πu, cos 2πu] × ge_obs (float32) — mdp_stage4.gait_clock(_gait_phase = 2π·_gu)."""
        ph = _F32(2.0 * math.pi) * self.u
        return np.array([np.sin(ph), np.cos(ph)], dtype=np.float32) * self.ge_obs

    def snapshot(self) -> dict:
        return {"want": bool(self.want), "ge": bool(self.ge), "ge_obs": float(self.ge_obs), "u": float(self.u), "f": float(self.f)}


def step_enable_allowed(active: bool, warmed: bool, blend_done: bool) -> tuple:
    """브리지 `step` 켜기 선행조건 → (허용, 사유). 끄기는 항상 허용(래치가 반주기 경계에서 내린다)."""
    if not active:
        return False, "정책 미실행(stand 로 W3 키를 먼저 시작)"
    if not warmed:
        return False, "인계 워밍업 미완료"
    if not blend_done:
        return False, "온셋 블렌딩 진행 중"
    return True, "ok"


# ------------------------------------------------------------------ 서기 정책(G21a) ↔ W3 상태 이어붙이기 (문서 §5, 브리지 미연결)
def carry_runner_state(src, dst) -> None:
    """**실험·브리지 미연결** — src 러너(예: G21a)의 관측 이력·관측 필터·last_action·슬루 기준·틱을 dst(W3 러너)로 옮긴다.

    전제: 같은 관측 레이아웃(320)·같은 기본자세·같은 관측 필터 β. 어긋나면 ValueError.
    gait_clock 이력은 옮기지 않고 dst 계약 값으로 채운다: dst 는 STAND(ge 0, ge_obs 0)로 시작하므로 (0,0) 5틀
    — src(G21a)의 1.1 Hz 회전 클록은 W3 STAND 학습 분포(클록 (0,0))에 없던 입력이다.
    dst 의 W3 클록은 reset 상태(ge 0)로 두되, 틱을 옮기므로 fresh 규칙(틱 ≤ 1)은 다시 걸리지 않는다(래치 경로로만 켜짐).
    위험(보고서 §5): last_action·act EMA 는 G21a 출력 분포 — W3 정책이 그 상태에서 학습된 적이 없다(W3 에피소드는 늘 fresh 시작).
    """
    if src.obs_dim != dst.obs_dim or src.terms != dst.terms:
        raise ValueError(f"관측 레이아웃 다름: {src.obs_dim} vs {dst.obs_dim}")
    if not np.array_equal(src.default_pose, dst.default_pose):
        raise ValueError("기본자세 다름 — 관측 joint_pos·행동 오프셋 기준이 바뀐다")
    if (src.act_obs_ema_beta, src.jvel_obs_ema_beta) != (dst.act_obs_ema_beta, dst.jvel_obs_ema_beta):
        raise ValueError("관측 필터 β 다름")
    if getattr(dst, "_w3_clock", None) is None:
        raise ValueError("dst 가 W3 러너가 아님")
    for name, h in src._hist.items():
        d = dst._hist[name]
        d._buf = h._buf.copy()
        d._pushes = h._pushes
    dst._hist["gait_clock"]._buf[:] = 0.0                         # W3 STAND 계약 (0,0)
    dst._ema = {k: v.copy() for k, v in src._ema.items()}
    dst.last_action = src.last_action.copy()
    dst.last_jvel_ema = None if src.last_jvel_ema is None else src.last_jvel_ema.copy()
    dst._prev_emitted = None if src._prev_emitted is None else src._prev_emitted.copy()
    dst._tick = src._tick
    dst._w3_clock.reset()


# ------------------------------------------------------------------ W3 세션 명령·정지 감독기 (1006 감사 H3–H6·L1 수정, 순수 파이썬)
# 브리지는 `--experimental-w3 --checkpoint stand_w3a_11699` 세션에서만 이 감독기를 만든다(그 밖에는 None = 종전 경로 그대로).
# ROS 를 모르는 순수 객체 — 판단은 여기서, 브리지는 결정(action)을 실행하고 응답을 발행만 한다(오프라인 셀프테스트 대상).
#
#  H3: step 은 req_id·seq(단조 증가) 필수, seq ≤ 마지막 seq 는 중복/오래됨으로 거부(1 Hz 반복 pub = 같은 seq → 거부).
#      stop/estop/가드 트립/STOP_ALL 은 **정지 래치** — 명시적 rearm(새 seq) 전까지 모든 켜기 거부(0.3 s 늦은 enable 의 위상 0 재기동 차단).
#      끄기(enable false)는 안전 방향이라 seq 가 없거나 오래돼도 적용한다(응답에 표시).
#  H4/H5: W3 세션에서 walk/march 는 항상 거부(가족 대체로 drv2·march_v5b 가 뜨던 경로), stand 는 정책 실행 중이면 거부
#      (래치 없이 ge 0·러너 리셋되던 재발사). 이 세션에서 정책 교체는 불가.
#  L1: 켜기 선행조건 = 종전(실행·워밍업·블렌딩) + 양발 접촉(정책이 받는 contact2 와 같은 비트) 연속 enable_contact_s,
#      기울기(정책이 받는 투영중력 = 장착 편향 보정 후) ≤ enable_tilt_deg, |ω_xy| < enable_quiet_w 연속 enable_quiet_s,
#      관측 신선(마지막 observe ≤ obs_max_age_s), ge 꺼짐(래치 해제 완료), 정지 래치 없음, 동결 대기 없음.
#  H2: 가드 트립(비하드) 시 즉시 STOP_ALL 대신 ge 0 → 반주기 래치 해제(clock.ge False) → 양발 접촉 freeze_contact_s·
#      |ω_xy| < freeze_quiet_w 연속 freeze_quiet_s → 동결. 트립부터 freeze_timeout_s 를 넘으면 그대로 동결하고 사유에 남긴다.
#      센서 공백·비유한·과속(OSC_VEL_HARD)·워밍업/착지 타임아웃·estop 은 지연 없이 즉시(정책을 더 돌릴 근거가 없다).

@dataclass(frozen=True)
class W3SafetyConfig:
    """W3 감독기 문턱 — 기본값은 보수적(실물 미측정: 실물 서기 |ω_xy|·기울기 분포로 현장 확인 필요)."""
    enable_contact_s: float = 1.0     # 켜기: 양발 접촉 연속 [s]
    enable_tilt_deg: float = 7.0      # 켜기: 몸통 기울기 상한 [deg] (투영중력 기준, 롤·피치 합성)
    enable_quiet_w: float = 0.25      # 켜기: |ω_xy| 문턱 [rad/s] (착지 인수 TD_QUIET_W 0.3 보다 엄격)
    enable_quiet_s: float = 1.0       # 켜기: 문턱 아래 연속 [s]
    obs_max_age_s: float = 0.1        # 켜기·지연 동결: 마지막 관측 나이 상한 [s] (정책 틱 5개)
    freeze_contact_s: float = 0.10    # 지연 동결: 양발 접촉 연속 [s]
    freeze_quiet_w: float = 0.8       # 지연 동결: |ω_xy| 문턱 [rad/s] (착지 직후 흔들림 허용 — 한발 지지 동결보다 낫다)
    freeze_quiet_s: float = 0.06      # 지연 동결: 문턱 아래 연속 [s]
    freeze_timeout_s: float = 1.5     # 트립 → 강제 동결 상한 [s] (반주기 0.455 s + 착지·확인 여유)


W3_SAFETY_DEFAULT = W3SafetyConfig()

#: 설정 허용 범위 — 문턱을 성공률 목적으로 풀지 못하게 상·하한을 둔다(넘으면 기동 거부).
W3_SAFETY_RANGES = {
    "enable_contact_s": (0.3, 5.0), "enable_tilt_deg": (2.0, 10.0), "enable_quiet_w": (0.05, 0.5),
    "enable_quiet_s": (0.3, 5.0), "obs_max_age_s": (0.04, 0.2), "freeze_contact_s": (0.04, 0.5),
    "freeze_quiet_w": (0.1, 1.5), "freeze_quiet_s": (0.0, 0.5), "freeze_timeout_s": (0.5, 3.0),
}


def parse_w3_safety(text) -> W3SafetyConfig:
    """'{"enable_tilt_deg": 6}' 같은 JSON 객체 문자열 → W3SafetyConfig. 모르는 키·비유한·범위 밖은 ValueError."""
    if text is None or str(text).strip() == "":
        return W3_SAFETY_DEFAULT
    import json
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("W3 안전 설정은 JSON 객체여야 함")
    kw = {}
    for k, v in d.items():
        if k not in W3_SAFETY_RANGES:
            raise ValueError(f"W3 안전 설정: 모르는 키 {k!r} (허용 {sorted(W3_SAFETY_RANGES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"W3 안전 설정 {k}: 유한 수 필요 (받음 {v!r})")
        lo, hi = W3_SAFETY_RANGES[k]
        if not lo <= float(v) <= hi:
            raise ValueError(f"W3 안전 설정 {k}={v} 범위 밖 [{lo}, {hi}]")
        kw[k] = float(v)
    return W3SafetyConfig(**kw)


def tilt_deg(grav) -> float:
    """투영중력(정책 관측과 같은 벡터) → 직립 대비 기울기 [deg]. 비정상이면 inf."""
    try:
        gx, gy, gz = (float(v) for v in grav)
    except (TypeError, ValueError):
        return float("inf")
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if not math.isfinite(n) or n < 1e-6:
        return float("inf")
    return math.degrees(math.acos(max(-1.0, min(1.0, -gz / n))))


def _check_req(d) -> tuple:
    """(req_id, seq, 오류문|None). req_id = 1~64자 문자열, seq = 1 이상 정수(bool 아님)."""
    rid, seq = d.get("req_id"), d.get("seq")
    if not isinstance(rid, str) or not (1 <= len(rid) <= 64):
        return rid, seq, "req_id(1~64자 문자열) 필수"
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        return rid, seq, "seq(1 이상 정수) 필수"
    return rid, seq, None


class W3Supervisor:
    """W3 세션 명령 상태기 + 지연 동결 순서기. 시간은 호출자가 넘긴다(now, 초). 상태는 브리지 프로세스 수명 동안 유지
    (stand 재시작으로 seq·정지 래치가 지워지지 않는다 — 재시작 뒤 켜기도 rearm 이 필요)."""

    FREEZE_WAIT_LATCH = "WAIT_LATCH"        # ge 0 요청 → 반주기 경계 대기(clock.ge 아직 1)
    FREEZE_WAIT_SUPPORT = "WAIT_SUPPORT"    # ge 0 → 양발 지지·정착 대기

    def __init__(self, cfg: W3SafetyConfig = W3_SAFETY_DEFAULT):
        self.cfg = cfg
        self.last_seq = None
        self.stop_latched = False
        self.latch_reason = None
        self.freeze_state = None
        self.freeze_t0 = None
        self.freeze_why = None
        self.freeze_latch_t = None
        self.last_obs_t = None
        self.both_t0 = None
        self.quiet_t0 = None
        self.fquiet_t0 = None
        self.tilt = None
        self.w_xy = None
        self.contact = None

    # ---- 관측(정책 틱마다, 가드·정책 계산 전) ----
    def observe(self, now, contact2, gyro, grav) -> None:
        cl, cr = bool(contact2[0] > 0.0), bool(contact2[1] > 0.0)
        try:
            w = math.hypot(float(gyro[0]), float(gyro[1]))
        except (TypeError, ValueError, IndexError):
            w = float("nan")
        self.contact = (cl, cr)
        self.w_xy = w
        self.tilt = tilt_deg(grav)
        self.both_t0 = (self.both_t0 if self.both_t0 is not None else now) if (cl and cr) else None
        quiet = math.isfinite(w) and w < self.cfg.enable_quiet_w
        self.quiet_t0 = (self.quiet_t0 if self.quiet_t0 is not None else now) if quiet else None
        fq = math.isfinite(w) and w < self.cfg.freeze_quiet_w
        self.fquiet_t0 = (self.fquiet_t0 if self.fquiet_t0 is not None else now) if fq else None
        self.last_obs_t = now

    @staticmethod
    def _dur(t0, now):
        return 0.0 if t0 is None else now - t0

    def enable_preconditions(self, now, *, active, warmed, blend_done, ge_now) -> tuple:
        ok, why = step_enable_allowed(active, warmed, blend_done)
        if not ok:
            return ok, why
        c, eps = self.cfg, 1e-9
        if self.stop_latched:
            return False, f"정지 래치({self.latch_reason}) — rearm(새 seq) 전까지 켜기 불가"
        if self.freeze_state is not None:
            return False, f"동결 대기 중({self.freeze_why})"
        if ge_now:
            return False, "직전 끄기의 반주기 래치 해제 전(ge 1) — 서기 확인 뒤 다시"
        if self.last_obs_t is None or now - self.last_obs_t > c.obs_max_age_s:
            return False, "관측 없음/낡음 — 정책 틱 확인"
        if self._dur(self.both_t0, now) < c.enable_contact_s - eps:
            return False, (f"양발 접촉 {self._dur(self.both_t0, now):.2f}s < {c.enable_contact_s:g}s "
                           f"(접촉 L{int(self.contact[0])} R{int(self.contact[1])})")
        if not (self.tilt is not None and self.tilt <= c.enable_tilt_deg):
            return False, f"기울기 {self.tilt:.1f}° > {c.enable_tilt_deg:g}°"
        if self._dur(self.quiet_t0, now) < c.enable_quiet_s - eps:
            return False, (f"정착 미달 |ω_xy| {self.w_xy:.2f} rad/s — 문턱 {c.enable_quiet_w:g} 아래 "
                           f"{self._dur(self.quiet_t0, now):.2f}s < {c.enable_quiet_s:g}s")
        return True, "ok"

    # ---- 명령 처리: 결정만 낸다 ----
    def handle(self, d, now, *, active, warmed, blend_done, ge_now, want_now) -> dict:
        """d = 파싱된 명령 dict. 반환 {"accepted", "why", "action", "want"}.
        action: "estop"(종전 estop 실행) · "stop"(종전 stop 실행 = ge 0) · "set_want"(want 적용) · "launch"(종전 stand 경로) ·
                "none"(실행 없음 — 거부, 또는 감독기 내부 상태만 바뀜)."""
        c = d.get("cmd", "")

        def out(acc, why, action="none", want=None):
            return {"accepted": bool(acc), "why": why, "action": action, "want": want}

        if c in ("estop", "stop"):
            _rid, seq, err = _check_req(d)
            if err is None and (self.last_seq is None or seq > self.last_seq):
                self.last_seq = seq
            self.stop_latched = True
            self.latch_reason = c
            return out(True, f"{c} — 정지 래치(rearm 전까지 켜기 거부)", c)
        if c == "step":
            v = d.get("enable", None)
            if not (isinstance(v, bool) or (v in (0, 1) and not isinstance(v, float))):
                return out(False, "enable 은 true/false 또는 0/1 필수")
            want = bool(v)
            _rid, seq, err = _check_req(d)
            stale = err is None and self.last_seq is not None and seq <= self.last_seq
            if not want:                       # 끄기 = 안전 방향: 형식·순번과 무관하게 적용
                if err is None and not stale:
                    self.last_seq = seq
                    note = ""
                else:
                    note = f" (주의: {err or f'오래된 seq {seq} ≤ {self.last_seq}'} — 끄기라 적용)"
                return out(True, "끄기 — 반주기 경계에서 서기" + note, "set_want", False)
            if err is not None:
                return out(False, err)
            if stale:
                return out(False, f"중복/오래된 seq {seq} ≤ 마지막 {self.last_seq}")
            self.last_seq = seq                # 형식이 맞는 새 seq 는 거부돼도 소비(재시도는 새 seq)
            if want_now and not self.stop_latched and self.freeze_state is None:
                return out(True, "이미 켜짐 — 변화 없음", "none")
            ok, why = self.enable_preconditions(now, active=active, warmed=warmed, blend_done=blend_done, ge_now=ge_now)
            if not ok:
                return out(False, why)
            return out(True, "켜기 — 다음 틱 위상 0(왼발 스윙)", "set_want", True)
        if c == "rearm":
            _rid, seq, err = _check_req(d)
            if err is not None:
                return out(False, err)
            if self.last_seq is not None and seq <= self.last_seq:
                return out(False, f"중복/오래된 seq {seq} ≤ 마지막 {self.last_seq}")
            self.last_seq = seq
            if self.freeze_state is not None:
                return out(False, f"동결 대기 중({self.freeze_why}) — rearm 불가")
            if active and (ge_now or want_now):
                return out(False, "스텝 래치가 아직 켜짐(ge/want 1) — 서기 전환 뒤 rearm")
            was = self.latch_reason
            self.stop_latched = False
            self.latch_reason = None
            return out(True, f"정지 래치 해제(이전 {was}) — 켜기는 별도 step(새 seq)", "none")
        if c in ("walk", "march"):
            return out(False, f"W3 세션에서 {c} 거부 — 정책 교체 금지(종전 가족 대체 drv2/march_v5b 경로 차단)")
        if c == "stand":
            if active:
                return out(False, "W3 정책 실행 중 stand 재발사 거부(러너 리셋·래치 우회) — 서기는 stop, 재시작은 estop·회수 뒤")
            if self.freeze_state is not None:
                return out(False, f"동결 대기 중({self.freeze_why})")
            return out(True, "stand 시작 요청(종전 게이트 적용)", "launch")
        if c == "turn":
            return out(False, "W3 세션에서 turn 미지원")
        return out(False, f"알 수 없는 명령 {c!r}")

    # ---- 가드 트립 → 지연 동결 ----
    def begin_freeze(self, now, why) -> bool:
        """처음 트립이면 True(새로 시작). 이미 대기 중이면 첫 사유·시각을 유지하고 False. 정지 래치를 건다."""
        self.stop_latched = True
        if self.freeze_state is not None:
            return False
        self.latch_reason = f"guard: {why}"
        self.freeze_state = self.FREEZE_WAIT_LATCH
        self.freeze_t0 = now
        self.freeze_why = str(why)
        self.freeze_latch_t = None
        return True

    def freeze_due(self, now, ge_now):
        """동결할 때면 (kind, 설명), 아니면 None. kind = "support"(래치 해제·양발 지지·정착 확인) | "timeout"."""
        if self.freeze_state is None:
            return None
        c = self.cfg
        if self.freeze_state == self.FREEZE_WAIT_LATCH and not ge_now:
            self.freeze_state = self.FREEZE_WAIT_SUPPORT
            self.freeze_latch_t = now
        waited = now - self.freeze_t0
        if self.freeze_state == self.FREEZE_WAIT_SUPPORT:
            fresh = self.last_obs_t is not None and now - self.last_obs_t <= c.obs_max_age_s
            if (fresh and self._dur(self.both_t0, now) >= c.freeze_contact_s - 1e-9
                    and self._dur(self.fquiet_t0, now) >= c.freeze_quiet_s - 1e-9):
                return "support", (f"래치 해제 +{self.freeze_latch_t - self.freeze_t0:.2f}s·양발 지지·정착 확인 "
                                   f"(트립 뒤 {waited:.2f}s)")
        if waited >= c.freeze_timeout_s - 1e-9:
            ct = self.contact or (False, False)
            w = self.w_xy if self.w_xy is not None else float("nan")
            return "timeout", (f"타임아웃 {c.freeze_timeout_s:g}s — {self.freeze_state} 상태로 동결 "
                               f"(접촉 L{int(ct[0])} R{int(ct[1])}, |ω_xy| {w:.2f})")
        return None

    def on_frozen(self, reason) -> None:
        """STOP_ALL 이 실제로 나갔다(어떤 경로든) — 동결 대기 해제, 정지 래치 유지."""
        self.freeze_state = None
        self.freeze_t0 = None
        self.freeze_latch_t = None
        if not self.stop_latched:
            self.latch_reason = f"stop_all: {reason}"
        self.stop_latched = True

    def snapshot(self) -> dict:
        def r(v, n):
            return None if v is None or not math.isfinite(v) else round(float(v), n)
        return {"last_seq": self.last_seq, "stop_latched": self.stop_latched, "latch_reason": self.latch_reason,
                "freeze": self.freeze_state, "freeze_why": self.freeze_why if self.freeze_state else None,
                "tilt_deg": r(self.tilt, 2), "w_xy": r(self.w_xy, 3),
                "contact": list(self.contact) if self.contact else None}
