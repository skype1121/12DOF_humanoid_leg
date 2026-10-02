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


from rl_walking.onset import onset_alpha  # noqa: E402


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
        # 0913 평가 전용 선택 훅: 지연 버퍼 **앞** 목표(rad)를 변조(브리지형 보정기 정합). 기본 None = 학습·종전 동작 불변.
        self.pre_delay_hook = None
        # 직전에 실제로 내보낸 목표각 (슬루의 기준점)
        self._prev_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # 리셋 직후 env는 현재 관절위치에서 슬루를 시작해야 함
        self._needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # 액션 지연 링버퍼 + env별 지연량 (min~max 균등 — 0803 실측 중심화)
        self._max_delay = int(cfg.max_delay_steps)
        self._min_delay = min(int(cfg.min_delay_steps), self._max_delay)
        if self._max_delay > 0:
            self._buf = torch.zeros(
                self._max_delay + 1, self.num_envs, self.action_dim, device=self.device
            )
            # 1003 G22w: 지연 표본 분포 가중(delay_probs, min..max 각 틱 확률). 비어 있으면 종전 균등.
            self._delay_p = None
            if len(tuple(cfg.delay_probs)) > 0:
                pr = torch.tensor(cfg.delay_probs, dtype=torch.float32, device=self.device)
                assert pr.numel() == self._max_delay - self._min_delay + 1, "delay_probs 길이 = max−min+1"
                self._delay_p = pr / pr.sum()
            self._delay = self._sample_delay(self.num_envs)
        # 배포 래퍼 EMA 저역필터 (0904, v15): 실물 브리지는 STAND_LPF_BETA=0.27 로
        # 정책 출력을 평활한 뒤 노드로 보낸다 — 학습엔 없던 필터라 정책이 자기
        # 출력이 깎이는 걸 본 적이 없다. env 별 β∈[lo,hi] (1.0 = 필터 없음).
        self._lpf_lo, self._lpf_hi = (float(v) for v in cfg.lpf_beta_range)
        self._use_lpf = self._lpf_hi < 1.0 or self._lpf_lo < 1.0
        if self._use_lpf:
            self._beta = torch.empty(self.num_envs, device=self.device).uniform_(
                self._lpf_lo, self._lpf_hi)
            self._lpf_state = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._bl_lo, self._bl_hi = (float(v) for v in cfg.backlash_rad_range)
        self._use_bl = self._bl_hi > 0.0
        if self._use_bl:
            self._bl = torch.empty(self.num_envs, 1, device=self.device).uniform_(self._bl_lo, self._bl_hi)
            self._bl_out = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._oh_lo, self._oh_hi = (float(v) for v in cfg.onset_hold_range)
        self._ob_lo, self._ob_hi = (float(v) for v in cfg.onset_blend_range)
        self._use_onset = (self._oh_hi > 0.0) or (self._ob_hi > 0.0)
        if self._use_onset:
            self._on_hold = torch.empty(self.num_envs, device=self.device).uniform_(self._oh_lo, self._oh_hi)
            self._on_blend = torch.empty(self.num_envs, device=self.device).uniform_(self._ob_lo, self._ob_hi)
            self._on_t = torch.zeros(self.num_envs, device=self.device)
        self._clamp_soft = bool(cfg.clamp_to_soft_limits)

    def _sample_delay(self, n: int) -> torch.Tensor:
        """env 별 지연 [틱] — 에피소드(리셋)마다 1회 추출해 고정. delay_probs 가 있으면 가중, 없으면 균등(종전)."""
        if getattr(self, "_delay_p", None) is not None:
            return torch.multinomial(self._delay_p, n, replacement=True) + self._min_delay
        return torch.randint(self._min_delay, self._max_delay + 1, (n,), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_init[:] = True
            if self._max_delay > 0:
                self._delay = self._sample_delay(self.num_envs)
            if self._use_lpf:
                self._beta.uniform_(self._lpf_lo, self._lpf_hi)
            if self._use_bl:
                self._bl.uniform_(self._bl_lo, self._bl_hi)
            if self._use_onset:
                self._on_hold.uniform_(self._oh_lo, self._oh_hi); self._on_blend.uniform_(self._ob_lo, self._ob_hi); self._on_t[:] = 0.0
        else:
            self._needs_init[env_ids] = True
            ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
                env_ids, device=self.device
            )
            if self._max_delay > 0:
                self._delay[ids] = self._sample_delay(len(ids))
            if self._use_lpf:
                self._beta[ids] = torch.empty(len(ids), device=self.device).uniform_(
                    self._lpf_lo, self._lpf_hi)
            if self._use_bl:
                self._bl[ids] = torch.empty(len(ids), 1, device=self.device).uniform_(
                    self._bl_lo, self._bl_hi)
            if self._use_onset:
                self._on_hold[ids] = torch.empty(len(ids), device=self.device).uniform_(self._oh_lo, self._oh_hi)
                self._on_blend[ids] = torch.empty(len(ids), device=self.device).uniform_(self._ob_lo, self._ob_hi)
                self._on_t[ids] = 0.0

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
            if self._use_lpf:
                self._lpf_state[ids] = cur
            if self._use_bl:
                self._bl_out[ids] = cur
            self._needs_init[ids] = False
        target = self._processed_actions
        if self.pre_delay_hook is not None:
            target = self.pre_delay_hook(target)
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
        # EMA 저역필터 — 0905 감사 확정: 실물 브리지 순서는 러너 슬루(4°/틱) → LPF β0.27 → 노드 슬루.
        # v15~v17 은 LPF→슬루 순서로 학습됐고(포화 구간에서 불일치), v18 부터 슬루→LPF 로 정합.
        if self._use_lpf:
            b = self._beta.view(-1, 1)
            self._lpf_state = b * self._processed_actions + (1.0 - b) * self._lpf_state
            self._processed_actions = self._lpf_state.clone()
        # 0921 온셋 인계: 홀드(목표 = 기본자세) → 스무스스텝 블렌딩 → 정책 (브리지 순서 슬루→LPF→블렌딩 과 동일 위치)
        if self._use_onset:
            alpha = onset_alpha(self._on_t, self._on_hold, self._on_blend).unsqueeze(1)
            hold_tgt = self._asset.data.default_joint_pos[:, self._joint_ids]
            self._processed_actions = hold_tgt * (1.0 - alpha) + self._processed_actions * alpha
            self._on_t += self._env.step_dt
        # 백래시 히스테리시스: 출력은 명령 주위 ±b 데드밴드 안에서는 움직이지 않는다
        if self._use_bl:
            cmd = self._processed_actions
            over = cmd - self._bl_out
            self._bl_out = torch.where(over > self._bl, cmd - self._bl,
                           torch.where(over < -self._bl, cmd + self._bl, self._bl_out))
            self._processed_actions = self._bl_out.clone()
        if self._clamp_soft:
            mask = getattr(self._env, "_soft_clamp_mask", None)     # set_joint_limits_from_config 가 만든 관절 마스크 (없으면 클램프 없음)
            if mask is not None:
                lim = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
                clamped = torch.maximum(torch.minimum(self._processed_actions, lim[..., 1]), lim[..., 0])
                m = mask[self._joint_ids].unsqueeze(0)
                self._processed_actions = torch.where(m, clamped, self._processed_actions)


@configclass
class SlewJointPositionActionCfg(JointPositionActionCfg):
    """슬루 리밋 관절 위치 액션 설정."""

    class_type: type[ActionTerm] = SlewJointPositionAction
    # 4° per 정책스텝(50Hz) = 실물 NODE_SLEW_DEG_PER_TICK와 동일
    max_delta_per_step: float = 0.06981317007977318
    # 액션 지연 랜덤화 상한 [정책틱] — 0이면 비활성 (stage3에서 1로)
    max_delay_steps: int = 0
    # 지연 하한 [정책틱] — 0803 실측(루프 4~8틱) 중심화용. max보다 크면 max로 클램프.
    min_delay_steps: int = 0
    # 지연 틱별 표본 확률(min..max 순, 길이 max−min+1). 빈 튜플 = 균등(종전). 1003 G22w: (0.10, 0.30, 0.25, 0.35)
    delay_probs: tuple = ()
    # 배포 래퍼 EMA β 랜덤 대역 (1.0,1.0)=비활성. 실물 STAND_LPF_BETA=0.27 포함하도록
    # v15 는 (0.2, 1.0).
    lpf_beta_range: tuple[float, float] = (1.0, 1.0)
    # 감속기 백래시 히스테리시스 반폭 [rad] env별 대역 (0,0)=비활성. v17 (0.005, 0.026)=0.3~1.5°.
    # 실물: 로터 엔코더는 감속기 뒤 백래시·벨트·프린트 변형을 못 본다(엔코더 사각지대,
    # 0802 실측 기울기 5.7° 에 관절오차 0.7°). 명령이 방향을 바꿀 때 |Δ|<b 동안 출력이 안 움직인다.
    backlash_rad_range: tuple[float, float] = (0.0, 0.0)
    # 0921 G7 온셋 인계(지면 절차 학습 안에): 리셋 뒤 onset_hold 동안 목표 = 기본자세(홀드), 이어 onset_blend 동안 스무스스텝으로
    # 홀드→정책 목표 크로스페이드(브리지 blend_alpha 와 같은 식). (0,0)=비활성(종전 동일). 정책은 인계 창을 학습 분포 안에서 본다.
    onset_hold_range: tuple[float, float] = (0.0, 0.0)
    onset_blend_range: tuple[float, float] = (0.0, 0.0)
    # 소프트 한계 명령 클램프(실물 stage8 경계클램프 정합): True 면 최종 목표를 asset.data.soft_joint_pos_limits 안으로 자른다.
    clamp_to_soft_limits: bool = False
