"""기계 유격(백래시) PD 액추에이터 (0913): 링크가 모터 위치 ±play 안에서는 자유(강성·감쇠 전달 0), 밖에서만 kp·kd 가 작용.
실측 근거: 0912 재배치 후 로터 엔코더·바닥 진실 사이 어긋남 — 피치 관절(hip_f 0.7~3.1°, ankle_f 1.5~3.7°, hip_r 요 3.7°) 크고,
횡 관절(hip_a 0.7~0.9°, ankle_r ≤0.1°) 작음. 종전 시뮬(ImplicitActuator kp150)·actions.py 의 backlash(명령측 히스테리시스)는
외력 아래 링크 자유 회전을 모델하지 않았다.
- play_rad: float 또는 {관절정규식: rad} — 관절별 유격 반폭(고정).
- play_rad_range: (lo, hi) 이면 리셋마다 env·관절별로 [lo,hi]·(play_rad 비율) 로 재샘플(학습 DR). None 이면 고정.
평가(--joint_play_deg)·학습(S8ac4rp) 공용. 관측 계약 불변."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators.actuator_pd import IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.utils import configclass


class DeadzonePDActuator(IdealPDActuator):
    cfg: "DeadzonePDActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # 관절별 기준 유격 (num_envs, num_joints)
        self._play_base = self._parse_joint_parameter(cfg.play_rad, 0.0).clone()
        self._play = self._play_base.clone()
        self._rng_scale = None
        if cfg.play_rad_range is not None:
            lo, hi = (float(v) for v in cfg.play_rad_range)
            self._rng_scale = (lo, hi)
            self._resample(None)

    def _resample(self, env_ids):
        if self._rng_scale is None:
            return
        lo, hi = self._rng_scale
        if env_ids is None:
            n = self._play.shape[0]; idx = slice(None)
        else:
            idx = env_ids; n = len(env_ids)
        s = torch.empty(n, 1, device=self._play.device).uniform_(lo, hi)
        self._play[idx] = self._play_base[idx] * s

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if self._rng_scale is None or env_ids is None:
            return
        if isinstance(env_ids, slice):          # Isaac Lab Articulation.reset(env_ids=None) → slice(None) (0913 감사)
            self._resample(None); return
        if len(env_ids) > 0:
            ids = torch.as_tensor(env_ids, device=self._play.device, dtype=torch.long)
            self._resample(ids)

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        e = control_action.joint_positions - joint_pos
        b = self._play
        e_dz = torch.sign(e) * torch.clamp(e.abs() - b, min=0.0)
        ev = control_action.joint_velocities - joint_vel
        inside = e.abs() <= b
        # 유격 안: 기어 마찰 근사 = 링크 속도에 대한 점성 감쇠(kd × inside_damping_scale; 0 이면 완전 자유). 밖: 통상 PD 감쇠.
        d_in = -self.damping * joint_vel * float(self.cfg.inside_damping_scale)
        self.computed_effort = (self.stiffness * e_dz
                                + torch.where(inside, d_in, self.damping * ev)
                                + control_action.joint_efforts)
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class DeadzonePDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = DeadzonePDActuator
    play_rad: dict[str, float] | float = 0.0        # 유격 반폭 [rad] (관절별 dict 가능)
    play_rad_range: tuple[float, float] | None = None   # 학습 DR: 리셋마다 play_rad × U(lo,hi)
    inside_damping_scale: float = 1.0   # 유격 안 링크 감쇠(kd 배수, 기어 마찰 근사). 0.0 = 완전 자유(최악)


class DeadzoneImplicitActuator(ImplicitActuator):
    """유격을 **PhysX 내재 드라이브의 목표 이동**으로 구현(수치 안정): 매 물리 스텝 목표 = q + sign(e)·max(|e|−play, 0).
    → 유격 안에서는 강성 0(목표 = 현재각), 감쇠 kd 는 내재 드라이브가 항상 적용(기어 마찰 근사). 명시적 DeadzonePD 는 작은 관성 관절에서
    kd·dt/I 가 커 불안정(0913 진단: 홀드도 3 s 에 붕괴) → 이 변형을 기본으로 쓴다."""
    cfg: "DeadzoneImplicitActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._play_base = self._parse_joint_parameter(cfg.play_rad, 0.0).clone()
        self._play = self._play_base.clone()
        self._rng_scale = None
        if cfg.play_rad_range is not None:
            lo, hi = (float(v) for v in cfg.play_rad_range)
            self._rng_scale = (lo, hi)
            self._resample(None)

    def _resample(self, env_ids):
        if self._rng_scale is None:
            return
        lo, hi = self._rng_scale
        if env_ids is None:
            n = self._play.shape[0]; idx = slice(None)
        else:
            idx = env_ids; n = len(env_ids)
        s = torch.empty(n, 1, device=self._play.device).uniform_(lo, hi)
        self._play[idx] = self._play_base[idx] * s

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if self._rng_scale is None or env_ids is None:
            return
        if isinstance(env_ids, slice):          # Isaac Lab Articulation.reset(env_ids=None) → slice(None) (0913 감사)
            self._resample(None); return
        if len(env_ids) > 0:
            ids = torch.as_tensor(env_ids, device=self._play.device, dtype=torch.long)
            self._resample(ids)

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        e = control_action.joint_positions - joint_pos
        e_dz = torch.sign(e) * torch.clamp(e.abs() - self._play, min=0.0)
        control_action.joint_positions = joint_pos + e_dz
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DeadzoneImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DeadzoneImplicitActuator
    play_rad: dict[str, float] | float = 0.0
    play_rad_range: tuple[float, float] | None = None


class DeadzoneCoulombImplicitActuator(DeadzoneImplicitActuator):
    """0920: 유격(DeadzoneImplicit) + **유격 안 쿨롱 마찰 토크**. 실물 발목앞뒤 실측(0919): 하중 시 출력이 엔코더 불변으로 ~8° 움직이고
    그 안 임의 지점에 정지(마찰이 자세를 잡음). PhysX 관절 마찰계수는 반력 비례라 저하중에서 약함 → 일정 크기 쿨롱 토크를 피드포워드로 준다.
    τ_f = −τ_c · tanh(θ̇ / v0)  (v0 로 채터 완화). 유격 밖(|e|>play)에서는 PD 가 지배하지만 마찰은 항상 더해진다(기어 마찰)."""
    cfg: "DeadzoneCoulombImplicitActuatorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._tau_c_base = self._parse_joint_parameter(cfg.coulomb_nm, 0.0).clone()
        self._tau_c = self._tau_c_base.clone()
        self._v0 = float(cfg.coulomb_v0)
        self._c_rng = tuple(float(v) for v in cfg.coulomb_range) if cfg.coulomb_range is not None else None
        if self._c_rng is not None:
            self._resample_c(None)

    def _resample_c(self, env_ids):
        if self._c_rng is None:
            return
        if env_ids is None:
            n = self._tau_c.shape[0]; idx = slice(None)
        else:
            idx = env_ids; n = len(env_ids)
        s = torch.empty(n, 1, device=self._tau_c.device).uniform_(*self._c_rng)
        self._tau_c[idx] = self._tau_c_base[idx] * s

    def reset(self, env_ids):
        super().reset(env_ids)
        if self._c_rng is None or env_ids is None:
            return
        if isinstance(env_ids, slice):
            self._resample_c(None); return
        if len(env_ids) > 0:
            self._resample_c(torch.as_tensor(env_ids, device=self._tau_c.device, dtype=torch.long))

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        control_action = super().compute(control_action, joint_pos, joint_vel)
        ff = -self._tau_c * torch.tanh(joint_vel / self._v0)
        control_action.joint_efforts = ff if control_action.joint_efforts is None else control_action.joint_efforts + ff
        return control_action


@configclass
class DeadzoneCoulombImplicitActuatorCfg(DeadzoneImplicitActuatorCfg):
    class_type: type = DeadzoneCoulombImplicitActuator
    coulomb_nm: dict[str, float] | float = 0.0     # 쿨롱 마찰 토크 [Nm]
    coulomb_v0: float = 0.05                        # tanh 속도 스케일 [rad/s]
    coulomb_range: tuple[float, float] | None = None  # env 별 배율 U(lo,hi) (DR)
