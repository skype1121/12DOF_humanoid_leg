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
    """JointPositionAction + 스텝당 목표각 변화량 클램프 (실물 슬루 정합).

    max_delay_steps > 0 이면 env별 랜덤 액션 지연(0~max틱)도 적용 —
    실측 루프 지연(제어기 ~15-19ms + 서보 ~3ms ≈ 1틱@50Hz)의 sim2real 정합.
    지연은 슬루 이전 단계(정책 출력 도착 지연)로 모델링.
    """

    cfg: "SlewJointPositionActionCfg"

    def __init__(self, cfg: "SlewJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._max_delta = float(cfg.max_delta_per_step)
        # 직전에 실제로 내보낸 목표각 (슬루의 기준점)
        self._prev_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # 리셋 직후 env는 현재 관절위치에서 슬루를 시작해야 함
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # 액션 지연 링버퍼 + env별 지연량
        self._max_delay = int(cfg.max_delay_steps)
        if self._max_delay > 0:
            self._buf = torch.zeros(
                self._max_delay + 1, self.num_envs, self.action_dim, device=self.device
            )
            self._delay = torch.randint(
                0, self._max_delay + 1, (self.num_envs,), device=self.device
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_init[:] = True
            if self._max_delay > 0:
                self._delay = torch.randint_like(self._delay, 0, self._max_delay + 1)
        else:
            self._needs_init[env_ids] = True
            if self._max_delay > 0:
                ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
                    env_ids, device=self.device
                )
                self._delay[ids] = torch.randint(
                    0, self._max_delay + 1, (len(ids),), device=self.device
                )

    def process_actions(self, actions: torch.Tensor):
        # scale/offset/clip 적용 (부모)
        super().process_actions(actions)
        # 리셋된 env: 기준점(그리고 지연버퍼)을 현재 관절위치로
        if torch.any(self._needs_init):
            ids = torch.nonzero(self._needs_init).squeeze(-1)
            cur = self._asset.data.joint_pos[ids][:, self._joint_ids]
            self._prev_target[ids] = cur
            if self._max_delay > 0:
                self._buf[:, ids] = cur.unsqueeze(0)
            self._needs_init[ids] = False
        target = self._processed_actions
        # env별 액션 지연: 버퍼 밀고 d틱 前 목표를 꺼냄
        if self._max_delay > 0:
            self._buf = torch.roll(self._buf, 1, dims=0)
            self._buf[0] = target
            idx = self._delay.view(1, -1, 1).expand(1, self.num_envs, self.action_dim)
            target = torch.gather(self._buf, 0, idx).squeeze(0)
        # 슬루 클램프: 스텝당 ±max_delta
        delta = target - self._prev_target
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
    # 액션 지연 랜덤화 상한 [정책틱] — 0이면 비활성 (stage3에서 1로)
    max_delay_steps: int = 0
