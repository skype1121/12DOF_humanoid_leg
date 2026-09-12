"""기계 유격(백래시) PD 액추에이터 (0913): 링크가 모터 위치 ±play 안에서는 자유(강성·감쇠 전달 0), 밖에서만 kp·kd 가 작용.
실측 근거: 0912 재배치 후 로터 엔코더·바닥 진실 사이 2~3° 어긋남(reports/real_prep_0911/FINAL_REPORT.md §9.4 이후 관측).
종전 시뮬(ImplicitActuator kp150)·actions.py 의 backlash(명령측 히스테리시스)는 외력 아래 링크 자유 회전을 모델하지 않았다.
평가 전용(--joint_play_deg). 학습 자산·계약 불변."""
from __future__ import annotations

import torch
from isaaclab.actuators.actuator_pd import IdealPDActuator
from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg
from isaaclab.utils import configclass


class DeadzonePDActuator(IdealPDActuator):
    cfg: "DeadzonePDActuatorCfg"

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        e = control_action.joint_positions - joint_pos
        b = float(self.cfg.play_rad)
        e_dz = torch.sign(e) * torch.clamp(e.abs() - b, min=0.0)
        ev = control_action.joint_velocities - joint_vel
        inside = e.abs() <= b
        self.computed_effort = (self.stiffness * e_dz
                                + torch.where(inside, torch.zeros_like(ev), self.damping * ev)
                                + control_action.joint_efforts)
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class DeadzonePDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = DeadzonePDActuator
    play_rad: float = 0.0   # 유격 반폭 [rad] (링크가 모터 대비 ±play 자유)
