"""HTURN2 실험 기능(1006, **기본 꺼짐·실물 미검증**) — W2q 앞 보행 정책 + 회전 피드포워드 + HeadingHold 목표 각.

사용자 방향(1006): 걷기 실물 범위를 임시로 'forward walking + heading/turn control' 로 좁힌다. 이 모듈의 계약은
**experimental feature flag** 로만 쓰인다 — 러너는 `hturn2=None`(기본)이면 이 모듈을 전혀 호출하지 않아 종전과 비트 동일,
브리지는 `--experimental-hturn2` 를 명시해야 체크포인트 키(HTURN2_CKPT_KEY)가 등록된다. 기본 구조에 영구 고정하지 않는다.

시뮬 원본(이 파일은 그 식을 numpy float32 로 옮긴 것 — 동치 검증: rl_walking/deploy/selftest_hturn2_equiv.py):
  (a) cadence 위상 누적 = rl_walking/mdp_hang.py cadence_of_cmd / gait_phase_update (W2q, 비-omni)
        f = clamp(0.7 + 4|vx|, 0.8, 1.6) Hz  (걷기 모드 |vx| ≥ 0.03), 서기 f0 = 1.1 Hz
        u ← 0                       (episode_length_buf ≤ 1, 즉 러너 틱 0·1 — env 의 fresh 규칙)
        u ← (u + f·dt) mod 1         (그 밖, f 는 이번 틱 명령으로 계산)
        관측 gait_clock = [sin 2πu, cos 2πu]  (mdp_stage4._gait_phase: env._gu 가 있으면 2π·_gu)
  (b) 회전 피드포워드 = mdp_hang.install_turn_assist 의 hook (κ −0.455, HTURN2 α 1.5, lead 6틱)
        u' = (u + f·lead_ticks·dt) mod 1,  v = (2u') mod 1,  ss(x) = x²(3−2x)
        왼발 스윙(u' < 0.5): shL = ss(v), shR = 1 − ss(v) · 오른발 스윙: shL = 1 − ss(v), shR = ss(v)
        amp = α·κ·wz·[|vx| ≥ 0.03]
        left_hip_r += amp·shL,  right_hip_r += amp·shR     (raw rad, 관절 프레임)
      적용 위치: 시뮬 SlewJointPositionAction 에서 hook 은 target(=기본자세+0.25·raw) **직후**, 지연 버퍼·슬루·LPF **앞**.
      실물 체인(러너 target → 러너 슬루 → 브리지 LPF → 노드 슬루)에 대응하면 **러너 target 직후, 러너 슬루 앞**이다
      (시뮬 지연 버퍼는 실물 전송·응답 지연의 모형이라 실물에 대응 코드가 없다).
      관측 'actions' 는 계속 raw — 잔차는 관측에 들어가지 않는다(시뮬 last_action 도 raw).
  (c) HeadingHold 목표 = 시작 요 + 목표 변화, wz = clip(kp·wrap(ψ* − ψ), ±0.3)  (eval_walk.py --heading_hold 1 --hh_wz_max 0.3
      --heading_target). 러너 밖(브리지 BridgeCore)이 담당하고, 이 모듈은 상수만 둔다.

틱 대응(러너 n번째 호출 = env episode_length_buf n): 관측 n 은 명령 cmd_n·위상 u_n 으로 만들고, 그 관측의 행동에 붙는 피드포워드는
(u_n, f_n, cmd_n) 을 쓴다 — ManagerBasedRLEnv.step 순서(process_action → … → command.compute → interval 이벤트 → 관측)에서
다음 스텝 process_action 의 hook 이 보는 상태와 같다. ⚠ eval_walk.py 는 env.step 전에 명령을 직접 바꾸므로(set_cmd) 평가에서는
hook 의 명령(vx·wz)만 관측보다 1틱 새것이었다 — 위상(u·f)은 같다. 학습 순서(이 모듈)를 계약으로 둔다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: 브리지 체크포인트 키(플래그를 켤 때만 등록). walk 가족(stand·march 접두사 아님) — "walk" 명령으로만 선택된다.
HTURN2_CKPT_KEY = "walk_w2q25590_hturn2"
#: W2q 25590 학습 run (biped12_stepb0). exported/ 는 아직 없다 — reports/HTURN2_BRIDGE_FLAG_1006.md 의 내보내기 절차.
HTURN2_RUN_REL = "logs/rsl_rl/biped12_stepb0/2026-10-05_18-18-43_w2q_seed1_from_w2h24791"
HTURN2_MODEL = "model_25590.pt"


@dataclass(frozen=True)
class Hturn2Config:
    """HTURN2 계약 상수 — 기본값이 시뮬 HTURN2 평가 조건(W2qe + TURN_ALPHA 1.5, --heading_hold 1 --hh_wz_max 0.3)."""
    f0: float = 1.1           # 서기 모드 클록 [Hz] (gait_phase_update f0)
    a: float = 0.7            # cadence 절편
    b: float = 4.0            # cadence 기울기
    fmin: float = 0.8
    fmax: float = 1.6
    walk_min: float = 0.03    # 걷기 모드 문턱 |vx| (gait_phase_update·install_turn_assist walk_min)
    dt: float = 0.02          # env.step_dt (dt 0.005 × decimation 4)
    alpha: float = 1.5        # HTURN2 α (TURN_ALPHA)
    kappa: float = -0.455     # install_turn_assist κ
    lead_ticks: int = 6       # 피드포워드 위상 앞당김 [틱]
    hh_kp: float = 1.0        # HeadingHold kp
    hh_wz_max: float = 0.3    # HeadingHold |wz| 상한 (HTURN2; 브리지 종전 기본 0.6)
    vx_max: float = 0.15      # 브리지 walk 명령 상한 = W2q 학습 명령 범위 상한 (lin_vel_x 0.05~0.15)


HTURN2_DEFAULT = Hturn2Config()

_F32 = np.float32


def cadence_of_cmd(vx, cfg: Hturn2Config = HTURN2_DEFAULT) -> np.float32:
    """mdp_hang.cadence_of_cmd 의 float32 스칼라판. 걷기 모드면 clamp(a + b|vx|), 아니면 f0."""
    ax = abs(_F32(vx))
    if ax >= _F32(cfg.walk_min):
        return _F32(min(max(_F32(cfg.a) + _F32(cfg.b) * ax, _F32(cfg.fmin)), _F32(cfg.fmax)))
    return _F32(cfg.f0)


def walk_mode(vx, cfg: Hturn2Config = HTURN2_DEFAULT) -> bool:
    return bool(abs(_F32(vx)) >= _F32(cfg.walk_min))


class CadenceClock:
    """명령 연동 위상 누적기 — mdp_hang.gait_phase_update(비-omni, freeze_stand False)의 단일 env 판.

    update(tick, vx) 는 러너가 관측을 조립할 때 틱당 정확히 1회 부른다(tick = 러너 _tick, 조립 전 값).
    """

    def __init__(self, cfg: Hturn2Config = HTURN2_DEFAULT):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.u = _F32(0.0)
        self.f = _F32(self.cfg.f0)

    def update(self, tick: int, vx) -> np.float32:
        self.f = cadence_of_cmd(vx, self.cfg)
        if int(tick) <= 1:            # env fresh = episode_length_buf ≤ 1 → 위상 0
            self.u = _F32(0.0)
        else:
            self.u = _F32(np.remainder(_F32(self.u + self.f * _F32(self.cfg.dt)), _F32(1.0)))
        return self.u

    def clock_obs(self) -> np.ndarray:
        """[sin 2πu, cos 2πu] float32 — mdp_stage4._gait_phase(2π·_gu) → gait_clock."""
        ph = _F32(2.0 * math.pi) * self.u
        return np.array([np.sin(ph), np.cos(ph)], dtype=np.float32)


def _ss(x):
    return x * x * (_F32(3.0) - _F32(2.0) * x)


def turn_ff_residual(u, f, vx, wz, cfg: Hturn2Config = HTURN2_DEFAULT):
    """install_turn_assist hook 의 잔차 (left_hip_r, right_hip_r) [rad] float32. 서기 모드면 (0, 0)."""
    lead_s = cfg.lead_ticks * cfg.dt                     # 시뮬: lead_ticks * env.step_dt (python float)
    uu = _F32(np.remainder(_F32(_F32(u) + _F32(f) * _F32(lead_s)), _F32(1.0)))
    v = _F32(np.remainder(_F32(2.0) * uu, _F32(1.0)))
    rise = _ss(v)
    fall = _F32(1.0) - _ss(v)
    swing_l = uu < _F32(0.5)                             # 앞 반주기 = 왼발 스윙(sb0 규약)
    sh_l, sh_r = (rise, fall) if swing_l else (fall, rise)
    amp = _F32(cfg.alpha * cfg.kappa) * _F32(wz) * _F32(1.0 if walk_mode(vx, cfg) else 0.0)
    return _F32(amp * sh_l), _F32(amp * sh_r)


def heading_wz(ref, yaw, cfg: Hturn2Config = HTURN2_DEFAULT) -> float:
    """HeadingHold 식(eval_walk.py: atan2(sin, cos) 랩) — 브리지는 policy_runner_stage4.HeadingHold 를 kp·상한만 바꿔 쓴다."""
    err = math.atan2(math.sin(ref - yaw), math.cos(ref - yaw))
    return max(-cfg.hh_wz_max, min(cfg.hh_wz_max, cfg.hh_kp * err))
