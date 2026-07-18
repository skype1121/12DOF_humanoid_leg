"""슬루 리밋 관절 위치 액션 — 실물 Stage8 노드의 4°/tick 제한을 시뮬에 동일 구현.

실물 경로: 정책(50Hz) → 노드가 목표각을 틱당 최대 4°씩만 이동(NODE_SLEW_DEG_PER_TICK).
시뮬에 이게 없으면 정책이 실물에서 포화되는 스텝 명령을 학습하고, 포화는 추가
지연처럼 작용해 저댐핑(kd5) 전도를 촉발한다 → 학습 중에도 같은 제한을 건다.

정책 스텝(50Hz, decimation 4 × dt 0.005) 1회당 목표각 변화량을 ±4°로 클램프.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class SlewJointPositionAction(JointPositionAction):
    """JointPositionAction + 스텝당 목표각 변화량 클램프 (실물 슬루 정합)."""

    cfg: "SlewJointPositionActionCfg"

    def __init__(self, cfg: "SlewJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._max_delta = float(cfg.max_delta_per_step)
        # 직전에 실제로 내보낸 목표각 (슬루의 기준점)
        self._prev_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # 리셋 직후 env는 현재 관절위치에서 슬루를 시작해야 함
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_init[:] = True
        else:
            self._needs_init[env_ids] = True

    def process_actions(self, actions: torch.Tensor):
        # scale/offset/clip 적용 (부모)
        super().process_actions(actions)
        # 리셋된 env: 기준점을 현재 관절위치로
        if torch.any(self._needs_init):
            ids = torch.nonzero(self._needs_init).squeeze(-1)
            self._prev_target[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
            self._needs_init[ids] = False
        # 슬루 클램프: 스텝당 ±max_delta
        delta = self._processed_actions - self._prev_target
        self._processed_actions = self._prev_target + torch.clamp(
            delta, -self._max_delta, self._max_delta
        )
        self._prev_target = self._processed_actions.clone()


@configclass
class SlewJointPositionActionCfg(JointPositionActionCfg):
    """슬루 리밋 관절 위치 액션 설정."""

    class_type: type[ActionTerm] = SlewJointPositionAction
    # 4° per 정책스텝(50Hz) = 실물 NODE_SLEW_DEG_PER_TICK와 동일
    max_delta_per_step: float = 0.06981317007977318
