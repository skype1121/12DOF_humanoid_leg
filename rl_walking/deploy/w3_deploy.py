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
