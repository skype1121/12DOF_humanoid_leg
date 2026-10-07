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
    # 1007 리드 발 규칙(w3_gait_update lead_vy_cmin·lead_wz_cmin). 0 = 종전(W3-A: 위상 0 = 늘 왼발 리드)과 비트 동일.
    lead_vy_cmin: float = 0.0    # > 0: STEP 시작(fresh·0→1) 때 vy < −이 값이면 위상 0.5(오른발 리드)
    lead_wz_cmin: float = 0.0    # > 0: STEP 시작 때 |vy| < lead_lat_eps 이고 wz < −이 값이면 위상 0.5(오른발 리드)
    lead_lat_eps: float = 0.02   # 시뮬 소스의 하드코딩 상수(cw[:, 1].abs() < 0.02)


W3_DEFAULT = W3Config()

# ------------------------------------------------------------------ 1007 W3S3d-15800 (Stage 3: 제자리·앞뒤·옆·제자리 회전, 학습 지연 1–5)
#: 브리지 키(stand 가족 접두사 — 시작은 서기). 명령 채널(vx·vy·wz 단일 축)·리드 발 규칙·외부 헤딩 고리를 쓰는 첫 W3 키.
W3S3D_CKPT_KEY = "stand_w3s3d_15800"
W3S3D_RUN_REL = "logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d"
W3S3D_MODEL = "model_15800.pt"
W3S3D_TASK = "Biped12-HangStand-W3S3d-v0"
#: 학습 env.yaml events.w3_gait.params = {f_step 1.1, lead_vy_cmin 0.02, lead_wz_cmin 0.08}(W3S2 → W3S3 상속), ramp_s 기본 0.2.
W3S3D_CONFIG = W3Config(lead_vy_cmin=0.02, lead_wz_cmin=0.08)


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
        self.last_lead = None            # 1007 진단: 마지막 STEP 시작 리드 발("left"|"right", 리드 규칙 켜진 설정에서만)

    def set_want(self, want: bool) -> None:
        self.want = bool(want)

    def update(self, tick: int, cmd=None) -> None:
        """cmd = 이번 틱 정책 명령 관측(vx, vy, wz) — 리드 발 규칙(cfg.lead_*_cmin > 0)에서만 읽는다(시뮬: 명령 갱신 뒤 같은 틱 값)."""
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
        u0 = _F32(0.0)
        if ge_new and (fresh or on) and (self.cfg.lead_vy_cmin > 0.0 or self.cfg.lead_wz_cmin > 0.0):
            u0 = self.lead_phase(cmd)
        self.u = u0 if reset_u else _F32(np.remainder(u_next, _F32(1.0)))
        self.f = _F32(self.cfg.f_step) if ge_new else _F32(0.0)
        self.ge = ge_new
        if fresh:
            self.ge_obs = _F32(1.0) if ge_new else _F32(0.0)
        else:
            self.ge_obs = _F32(min(max(_F32(self.ge_obs + (self._k if ge_new else -self._k)), _F32(0.0)), _F32(1.0)))

    def lead_phase(self, cmd) -> np.float32:
        """STEP 시작 위상 — w3_gait_update 의 u0(float32 비교: 시뮬 텐서 float32 대 python 상수 = float32 비교).
        vy < −lead_vy_cmin → 0.5 / |vy| < lead_lat_eps ∧ wz < −lead_wz_cmin → 0.5 / 그 밖 0(왼발 리드). cmd 없으면 거부."""
        if cmd is None:
            raise ValueError("W3 리드 발 규칙: 이번 틱 명령(vx, vy, wz)이 필요")
        c = np.asarray(cmd, dtype=np.float32).reshape(-1)
        vy, wz = c[1], c[2]
        u0 = _F32(0.0)
        if self.cfg.lead_vy_cmin > 0.0 and vy < _F32(-self.cfg.lead_vy_cmin):
            u0 = _F32(0.5)
        if self.cfg.lead_wz_cmin > 0.0 and abs(vy) < _F32(self.cfg.lead_lat_eps) and wz < _F32(-self.cfg.lead_wz_cmin):
            u0 = _F32(0.5)
        self.last_lead = "right" if u0 != _F32(0.0) else "left"     # 진단(블랙박스) — 래치 전이 events 와 섞지 않는다
        return u0

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


# ------------------------------------------------------------------ 1007 명령 채널(vx·vy·wz) + 외부 헤딩 고리 — W3S3d 키에서만(그 밖 키는 None)
# 학습 분포(W3S3d env.yaml events.w3_cmd.params, stage 3): STEP 명령 = 제자리 / 앞뒤(vx +0.03–0.10, −0.03–0.06) / 옆(|vy| 0.03–0.04 —
# S3d 학습 중 커리큘럼 미진급, 시작 단계 0) / 제자리 회전(|wz| 0.1–0.2, 같은 이유) **중 한 축만**(대각·이동+회전 없음). 그 사이(0 < |v| < 하한)는
# 학습에 없던 값이다. 명령은 리샘플(8–12 s)마다 계단으로 바뀐다(STAND→STEP 은 ge 와 명령이 같은 틱에, STEP→STEP 은 위상 유지).
# → 배포 '슬루' = 진폭 램프가 아니라 **변경 간 최소 체류 시간**(램프는 학습에 없던 0–하한 사이 값을 지나간다). 범위 밖은 거부(클립하지 않음).

@dataclass(frozen=True)
class W3CmdLimits:
    """명령 채널 허용 범위 [m/s, rad/s] — 기본값 = W3S3d 학습 범위. 0 은 언제나 허용(제자리)."""
    vx_fwd: tuple = (0.03, 0.10)
    vx_back: tuple = (0.03, 0.06)
    vy: tuple = (0.03, 0.04)
    wz: tuple = (0.1, 0.2)
    min_dwell_s: float = 1.0      # 걷는 중 명령 변경 간 최소 간격 [s] (한 걸음 주기 0.91 s 이상)


W3S3D_CMD_LIMITS = W3CmdLimits()

#: 1007 콘솔 걷기 버튼: 시간 제한 걷기(버스트) 허용 범위 [s]. step 켜기·명령 변경에 dur_s 를 주면 그 시간이 지난 뒤 **브리지(감독기)가** 끄기를 요구한다 —
#: 콘솔·통신이 죽어도 걷기가 무한히 이어지지 않게 하는 기한이다(끄기는 종전과 같이 반주기 경계에서 서기).
W3_BURST_RANGE = (0.3, 10.0)
#: 1007 콘솔 걷기 버튼 표(단일 원본 — 콘솔·시뮬 평가가 같은 값을 읽는다). 전부 W3S3d 학습 범위 안의 단일 축 명령 + 지속 시간.
#: 걸음 주기 = 1 / f_step = 0.909 s(왼발·오른발 한 번씩). 끄기는 다음 반주기 경계에서 걸리므로 dur_s 0.6 → 한 주기(0.91 s), 2.4 → 세 주기(2.73 s), 2.0 → 2.5 주기(2.27 s).
#: 부호: +vx 앞 · +vy 왼 · +wz 왼쪽(반시계) 회전 — 실물 부호는 계획서 T6 에서 확인 전(현장 확인 필요).
#: "enabled": False 인 버튼은 X14(TRANSITION_W3KE.md §5.18 사전 등록 기준 B4)에서 미달 — 콘솔은 비활성으로 보이고 서버도 거부한다(w3_move_cmd). 값을 바꿔 다시 켜려면 새 등록·평가가 먼저다.
W3_MOVES = {
    "fwd1": {"label": "전진 한 걸음", "vx": 0.06, "vy": 0.0, "wz": 0.0, "dur_s": 0.6},
    "fwd3": {"label": "전진 세 걸음", "vx": 0.06, "vy": 0.0, "wz": 0.0, "dur_s": 2.4},
    "back1": {"label": "후진 한 걸음", "vx": -0.04, "vy": 0.0, "wz": 0.0, "dur_s": 0.6,
              "enabled": False, "why_disabled": "X14(1007 시뮬): 방향 부호 일치 80 %(기준 95 %) — 한 주기 변위(중앙 −2.4 cm)가 과도 잡음(±3 cm)에 묻힘"},
    "left1": {"label": "왼쪽 옆걸음", "vx": 0.0, "vy": 0.035, "wz": 0.0, "dur_s": 0.6},
    "right1": {"label": "오른쪽 옆걸음", "vx": 0.0, "vy": -0.035, "wz": 0.0, "dur_s": 0.6,
               "enabled": False, "why_disabled": "X14(1007 시뮬): 방향 부호 일치 63 %(기준 95 %) — 한 주기 변위가 시작 과도의 왼쪽 쏠림(3–5 cm)에 묻힘"},
    "turn_l": {"label": "좌측 회전", "vx": 0.0, "vy": 0.0, "wz": 0.15, "dur_s": 2.0},
    "turn_r": {"label": "우측 회전", "vx": 0.0, "vy": 0.0, "wz": -0.15, "dur_s": 2.0},
    "inplace": {"label": "제자리 걸음", "vx": 0.0, "vy": 0.0, "wz": 0.0, "dur_s": 2.0},
}


def w3_move_enabled(name) -> bool:
    return bool(W3_MOVES[name].get("enabled", True))


def w3_moves_public() -> list:
    """콘솔 표시용 버튼 목록(정의 순서) — [{name, label, vx, vy, wz, dur_s, enabled, why_disabled}]."""
    return [{"name": k, "label": m["label"], "vx": m["vx"], "vy": m["vy"], "wz": m["wz"], "dur_s": m["dur_s"],
             "enabled": bool(m.get("enabled", True)), "why_disabled": m.get("why_disabled")} for k, m in W3_MOVES.items()]


def w3_move_cmd(name, limits: "W3CmdLimits" = W3S3D_CMD_LIMITS, allow_disabled: bool = False) -> dict:
    """버튼 이름 → 검증된 step 켜기 필드 {vx, vy, wz, dur_s}(학습 범위·단일 축·기한 범위 검사, 위반 ValueError).
    비활성 버튼(enabled False)은 거부한다 — allow_disabled 는 시뮬 평가 전용(콘솔·브리지 경로에서는 쓰지 않는다)."""
    if name not in W3_MOVES:
        raise ValueError(f"모르는 걷기 버튼 {name!r} (허용 {sorted(W3_MOVES)})")
    m = W3_MOVES[name]
    if not allow_disabled and not m.get("enabled", True):
        raise ValueError(f"비활성 걷기 버튼 {name!r}: {m.get('why_disabled', '사유 미기록')}")
    vx, vy, wz = validate_w3_cmd({"vx": m["vx"], "vy": m["vy"], "wz": m["wz"]}, limits)
    dur = float(m["dur_s"])
    if not (W3_BURST_RANGE[0] - 1e-9 <= dur <= W3_BURST_RANGE[1] + 1e-9):
        raise ValueError(f"{name}: dur_s {dur} 범위 밖 {W3_BURST_RANGE}")
    return {"vx": vx, "vy": vy, "wz": wz, "dur_s": dur}
#: --w3-cmd-limits 로 넓힐 수 있는 상한(시뮬 평가 근거: vy 0.05–0.08 추종·wz 0.3 회전 — W3S3d-15800 확장 평가, 학습 범위 밖이라 기본은 안 씀).
W3_CMD_LIMIT_RANGES = {"vy_max": (0.03, 0.08), "wz_max": (0.1, 0.3), "min_dwell_s": (0.5, 10.0)}


def parse_w3_cmd_limits(text, base: W3CmdLimits = W3S3D_CMD_LIMITS) -> W3CmdLimits:
    """'{"vy_max": 0.06}' → W3CmdLimits. 상한·체류만 바꾼다(하한·앞뒤 범위 고정). 모르는 키·비유한·범위 밖은 ValueError."""
    if text is None or str(text).strip() == "":
        return base
    import json
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("W3 명령 범위 설정은 JSON 객체여야 함")
    kw = {}
    for k, v in d.items():
        if k not in W3_CMD_LIMIT_RANGES:
            raise ValueError(f"W3 명령 범위: 모르는 키 {k!r} (허용 {sorted(W3_CMD_LIMIT_RANGES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"W3 명령 범위 {k}: 유한 수 필요 (받음 {v!r})")
        lo, hi = W3_CMD_LIMIT_RANGES[k]
        if not lo <= float(v) <= hi:
            raise ValueError(f"W3 명령 범위 {k}={v} 범위 밖 [{lo}, {hi}]")
        kw[k] = float(v)
    return W3CmdLimits(vx_fwd=base.vx_fwd, vx_back=base.vx_back,
                       vy=(base.vy[0], kw.get("vy_max", base.vy[1])), wz=(base.wz[0], kw.get("wz_max", base.wz[1])),
                       min_dwell_s=kw.get("min_dwell_s", base.min_dwell_s))


def validate_w3_cmd(d, lim: W3CmdLimits) -> tuple:
    """명령 dict 의 vx·vy·wz(없으면 0) → (vx, vy, wz) python float. 단일 축 위반·축별 범위 밖·비유한·bool 은 ValueError(클립 없음)."""
    v = {}
    for k in ("vx", "vy", "wz"):
        x = d.get(k, 0.0)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise ValueError(f"{k} 는 유한 수(받음 {x!r})")
        v[k] = float(x)
    nz = [k for k in ("vx", "vy", "wz") if v[k] != 0.0]
    if len(nz) > 1:
        raise ValueError(f"단일 축만 허용(받음 {nz}) — W3S3d 학습 = 제자리/앞뒤/옆/회전 한 축, 혼합은 학습 밖")
    eps = 1e-6      # 1007: 1e-9 → 1e-6 — 경계값이 float32 를 한 번 거치면(0.1f = 0.1 + 1.5e-9, 0.2f = 0.2 + 3.0e-9) 거부되던 것 방지. 1e-6 m/s·rad/s 는 물리적으로 무의미한 여유

    def inr(x, r):
        return r[0] - eps <= abs(x) <= r[1] + eps
    if v["vx"] > 0.0 and not inr(v["vx"], lim.vx_fwd):
        raise ValueError(f"vx {v['vx']:+.3f} 는 앞 범위 [{lim.vx_fwd[0]}, {lim.vx_fwd[1]}] 밖(0 또는 범위 안)")
    if v["vx"] < 0.0 and not inr(v["vx"], lim.vx_back):
        raise ValueError(f"vx {v['vx']:+.3f} 는 뒤 범위 [−{lim.vx_back[1]}, −{lim.vx_back[0]}] 밖")
    if v["vy"] != 0.0 and not inr(v["vy"], lim.vy):
        raise ValueError(f"|vy| {abs(v['vy']):.3f} 는 범위 [{lim.vy[0]}, {lim.vy[1]}] 밖")
    if v["wz"] != 0.0 and not inr(v["wz"], lim.wz):
        raise ValueError(f"|wz| {abs(v['wz']):.3f} 는 범위 [{lim.wz[0]}, {lim.wz[1]}] 밖")
    return (v["vx"], v["vy"], v["wz"])


@dataclass(frozen=True)
class W3HeadingLoopConfig:
    """외부 헤딩 P 고리 — 식은 eval_w3 --heading_loop K(--hl_wzmax) · rl_walking/w3.py w3_heading_loop 와 같다:
        ψ* = 걷기 시작 요(래치), 이후 ψ* ← wrap(ψ* + wz₀·dt);  wz = clip(wz₀ + K·wrap(ψ* − ψ), ±wz_max).
    시뮬 근거: W3S3d-15800 120 s 요 표류 2.2–2.4°(K 0.5, ±0.4). 실물 안전 장치(시뮬에 없음):
      err_max_deg — |wrap(ψ* − ψ)| 가 넘으면 IMU 요 이상(점프·표류) 또는 실제 회전 이탈로 보고 고리 고장 → 걷기 정지(서기).
      timeout_s   — 한 번 걷기(ψ* 래치) 최대 시간. 넘으면 걷기 정지(시뮬 검증 길이 120 s = 실물 IMU 요 표류 누적의 상한).
    브리지는 운영자 wz₀ ≠ 0(회전 명령) 동안 고리를 끈다(W3S3h 학습 규약 = STEP·비회전 명령만) — 그때 ψ* 도 해제."""
    kp: float = 0.5
    wz_max: float = 0.4
    err_max_deg: float = 15.0
    timeout_s: float = 120.0
    dt: float = 0.005 * 4


W3_HL_DEFAULT = W3HeadingLoopConfig()
W3_HL_RANGES = {"kp": (0.1, 1.0), "wz_max": (0.1, 0.4), "err_max_deg": (3.0, 30.0), "timeout_s": (5.0, 300.0)}


def parse_w3_heading_loop(text):
    """None/'' → 기본(켜짐) · 'off' → None(끔) · JSON 객체 → W3HeadingLoopConfig(모르는 키·범위 밖 ValueError)."""
    if text is None or str(text).strip() == "":
        return W3_HL_DEFAULT
    if str(text).strip().lower() == "off":
        return None
    import json
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("W3 헤딩 고리 설정은 'off' 또는 JSON 객체")
    kw = {}
    for k, v in d.items():
        if k not in W3_HL_RANGES:
            raise ValueError(f"W3 헤딩 고리: 모르는 키 {k!r} (허용 {sorted(W3_HL_RANGES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"W3 헤딩 고리 {k}: 유한 수 필요 (받음 {v!r})")
        lo, hi = W3_HL_RANGES[k]
        if not lo <= float(v) <= hi:
            raise ValueError(f"W3 헤딩 고리 {k}={v} 범위 밖 [{lo}, {hi}]")
        kw[k] = float(v)
    return W3HeadingLoopConfig(**kw)


def wrap_pi(x: float) -> float:
    """atan2(sin x, cos x) — eval_w3._wrap · w3._wrap 과 같은 식(float64)."""
    return math.atan2(math.sin(x), math.cos(x))


class W3HeadingLoop:
    """헤딩 고리 상태기(순수 파이썬, 시간 = 갱신 횟수 × dt). update() 는 걷는 틱마다 1회, release() 는 정지·서기·회전 명령 때."""

    def __init__(self, cfg: W3HeadingLoopConfig = W3_HL_DEFAULT):
        self.cfg = cfg
        self.last_fault = None
        self.release()

    def release(self) -> None:
        self.tgt = None
        self.n = 0
        self.err = None
        self.wz = None
        self.fault = None

    @property
    def latched(self) -> bool:
        return self.tgt is not None

    def update(self, psi: float, wz0: float = 0.0):
        """현재 요 ψ, 운영자 wz₀ → (wz 명령, 고장 사유|None). 고장이면 wz = wz₀(보정 없음)이고 release 전까지 고장 유지."""
        c = self.cfg
        if self.fault is not None:
            return float(wz0), self.fault
        if not math.isfinite(float(psi)):
            self.fault = self.last_fault = "IMU 요 비유한"
            return float(wz0), self.fault
        if self.tgt is None:
            self.tgt = float(psi)
            self.n = 0
        else:
            self.tgt = wrap_pi(self.tgt + float(wz0) * c.dt)
            self.n += 1
        e = wrap_pi(self.tgt - float(psi))
        self.err = e
        if abs(e) > math.radians(c.err_max_deg):
            self.fault = self.last_fault = f"헤딩 오차 {math.degrees(e):+.1f}° > {c.err_max_deg:g}° (IMU 요 이상 또는 회전 이탈)"
            return float(wz0), self.fault
        if self.n * c.dt > c.timeout_s + 1e-9:
            self.fault = self.last_fault = f"헤딩 고리 타임아웃 {c.timeout_s:g}s(한 번 걷기 상한)"
            return float(wz0), self.fault
        self.wz = max(-c.wz_max, min(c.wz_max, float(wz0) + c.kp * e))
        return self.wz, None

    def snapshot(self) -> dict:
        return {"latched": self.latched, "tgt": None if self.tgt is None else round(self.tgt, 5),
                "err_deg": None if self.err is None else round(math.degrees(self.err), 3),
                "wz": None if self.wz is None else round(self.wz, 5), "t_s": round(self.n * self.cfg.dt, 2),
                "fault": self.fault, "last_fault": self.last_fault}


#: 브리지 W3 키 명세 — 키 → (run 상대 경로, 모델, 태스크, W3Config, 명령 채널 범위|None). None = 명령 (0,0,0) 고정(W3-A).
W3_SPECS = {
    W3_CKPT_KEY: (W3_RUN_REL, W3_MODEL, W3_TASK, W3_DEFAULT, None),
    W3S3D_CKPT_KEY: (W3S3D_RUN_REL, W3S3D_MODEL, W3S3D_TASK, W3S3D_CONFIG, W3S3D_CMD_LIMITS),
}
W3_KEYS = tuple(W3_SPECS)


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

    def __init__(self, cfg: W3SafetyConfig = W3_SAFETY_DEFAULT, cmd_limits: W3CmdLimits | None = None):
        self.cfg = cfg
        self.cmd_limits = cmd_limits     # 1007: None = 명령 채널 없음(W3-A, (0,0,0) 고정 — 종전 동작), 아니면 vx·vy·wz 검증 범위
        self.last_cmd_t = None           # 1007: 마지막으로 명령(켜기·변경)을 적용한 시각 — 변경 간 최소 체류
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
        # 1007 시간 제한 걷기(버스트): 기한(초, 호출자 시계)·요청 ID·길이. 한 번도 안 쓰면 snapshot 에 필드가 없다(종전 스키마 불변).
        self.burst_t_off = None
        self.burst_req = None
        self.burst_dur = None
        self.burst_used = False
        self.last_burst_end = None

    # ---- 1007 시간 제한 걷기(버스트) ----
    def burst_arm(self, now, req_id, dur_s) -> None:
        """수락·실행된 켜기/명령 변경 뒤에 호출자가 부른다. dur_s None = 기한 없음(종전 연속 걷기 — 걸려 있던 기한도 지운다)."""
        if dur_s is None:
            self.burst_t_off = self.burst_req = self.burst_dur = None
            return
        self.burst_used = True
        self.burst_t_off = float(now) + float(dur_s)
        self.burst_req = req_id
        self.burst_dur = float(dur_s)

    def burst_clear(self) -> None:
        self.burst_t_off = self.burst_req = self.burst_dur = None

    def burst_due(self, now, want_now):
        """기한이 지났고 아직 걷기 요청 중(want 1)이면 (req_id, dur_s, 지난 시간) 을 돌려주고 기한을 지운다 — 호출자는 step 끄기와 같은 실행을 한다.
        want 0(이미 꺼짐)이면 기한만 지우고 None."""
        if self.burst_t_off is None:
            return None
        if not want_now:
            self.burst_clear()
            return None
        if now < self.burst_t_off - 1e-9:
            return None
        info = (self.burst_req, self.burst_dur, float(now) - (self.burst_t_off - self.burst_dur))
        self.last_burst_end = {"req_id": self.burst_req, "dur_s": self.burst_dur, "t": float(now)}
        self.burst_clear()
        return info

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
    def handle(self, d, now, *, active, warmed, blend_done, ge_now, want_now, cmd_now=None) -> dict:
        """d = 파싱된 명령 dict. 반환 {"accepted", "why", "action", "want", "cmd", "hl"}.
        action: "estop"(종전 estop 실행) · "stop"(종전 stop 실행 = ge 0) · "set_want"(want 적용) · "launch"(종전 stand 경로) ·
                "set_cmd"(1007 명령 채널: 걷는 중 명령만 바꿈) · "none"(실행 없음 — 거부, 또는 감독기 내부 상태만 바뀜).
        1007 명령 채널(cmd_limits 있음): step 켜기에 vx·vy·wz(없으면 0)·heading_loop(bool, 없으면 True) — 켜기와 같은 틱에 적용(시뮬
        리샘플과 같음 → 리드 발 규칙이 이 명령을 본다). cmd = 검증된 (vx, vy, wz), hl = 헤딩 고리 사용 여부. 채널 없는 키는 0 아닌 속도 거부."""
        c = d.get("cmd", "")

        def out(acc, why, action="none", want=None, cmd=None, hl=None, dur=None):
            r = {"accepted": bool(acc), "why": why, "action": action, "want": want, "cmd": cmd, "hl": hl}
            if dur is not None:              # 1007 버스트: dur_s 를 준 명령에만 필드 추가(종전 결정 dict 불변)
                r["dur_s"] = dur
            return r

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
            cmd = hl = None
            dur = d.get("dur_s", None)         # 1007 버스트: 없으면 종전(기한 없음)
            if dur is not None:
                if (isinstance(dur, bool) or not isinstance(dur, (int, float)) or not math.isfinite(float(dur))
                        or not (W3_BURST_RANGE[0] - 1e-9 <= float(dur) <= W3_BURST_RANGE[1] + 1e-9)):
                    return out(False, f"dur_s 는 {W3_BURST_RANGE[0]:g}–{W3_BURST_RANGE[1]:g} s 의 수(받음 {dur!r})")
                dur = float(dur)
            if self.cmd_limits is None:        # 명령 채널 없는 키(W3-A): (0,0,0) 고정 — 0 아닌 속도는 조용히 버리지 않고 거부
                if any(isinstance(d.get(k), bool) or not isinstance(d.get(k, 0.0), (int, float)) or d.get(k, 0.0) != 0
                       for k in ("vx", "vy", "wz")):
                    return out(False, "이 키는 명령 채널 없음 — vx·vy·wz 는 0 만(제자리 스텝)")
            else:
                try:
                    cmd = validate_w3_cmd(d, self.cmd_limits)
                except ValueError as e:
                    return out(False, f"명령 거부: {e}")
                hl = d.get("heading_loop", True)
                if not isinstance(hl, bool):
                    return out(False, "heading_loop 은 true/false")
            if want_now and not self.stop_latched and self.freeze_state is None:
                if self.cmd_limits is None:
                    return out(True, "이미 켜짐 — 변화 없음", "none")
                cur = tuple(float(x) for x in cmd_now) if cmd_now is not None else None
                if cur == cmd:
                    return out(True, "이미 켜짐·같은 명령 — 변화 없음", "none")
                if self.last_cmd_t is not None and now - self.last_cmd_t < self.cmd_limits.min_dwell_s - 1e-9:
                    return out(False, f"명령 변경 간격 {now - self.last_cmd_t:.2f}s < {self.cmd_limits.min_dwell_s:g}s(체류) — 다시")
                self.last_cmd_t = now
                return out(True, f"명령 변경 {cur} → {cmd}(위상 유지)" + (f"·기한 {dur:g}s" if dur is not None else ""), "set_cmd", None, cmd, hl, dur)
            ok, why = self.enable_preconditions(now, active=active, warmed=warmed, blend_done=blend_done, ge_now=ge_now)
            if not ok:
                return out(False, why)
            if self.cmd_limits is None:
                return out(True, "켜기 — 다음 틱 위상 0(왼발 스윙)" + (f"·기한 {dur:g}s" if dur is not None else ""), "set_want", True, dur=dur)
            self.last_cmd_t = now
            return out(True, f"켜기 — 명령 {cmd}·헤딩 고리 {'켜짐' if hl else '꺼짐'}(다음 틱, 리드 발은 명령 부호 규칙)"
                       + (f"·기한 {dur:g}s" if dur is not None else ""), "set_want", True, cmd, hl, dur)
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
        self.burst_clear()                   # 1007: 가드 트립 뒤에는 기한 끄기가 따로 나가지 않게
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
        self.burst_clear()
        if not self.stop_latched:
            self.latch_reason = f"stop_all: {reason}"
        self.stop_latched = True

    def snapshot(self) -> dict:
        def r(v, n):
            return None if v is None or not math.isfinite(v) else round(float(v), n)
        return {"last_seq": self.last_seq, "stop_latched": self.stop_latched, "latch_reason": self.latch_reason,
                **({} if self.cmd_limits is None else {"last_cmd_t": self.last_cmd_t}),
                "freeze": self.freeze_state, "freeze_why": self.freeze_why if self.freeze_state else None,
                "tilt_deg": r(self.tilt, 2), "w_xy": r(self.w_xy, 3),
                "contact": list(self.contact) if self.contact else None,
                **({} if not self.burst_used else {"burst": (None if self.burst_t_off is None else
                                                          {"req_id": self.burst_req, "dur_s": self.burst_dur, "t_off": round(self.burst_t_off, 3)}),
                                                 "last_burst_end": self.last_burst_end})}
