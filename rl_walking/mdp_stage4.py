"""Stage4 전용 mdp 함수 — 게이트 클록 관측 + 접촉 관측 + 주기 보상.

설계 근거:
- Siekmann et al. 2021 "Sim-to-Real Learning of All Common Bipedal Gaits via
  Periodic Reward Composition": 위상 클록으로 스탠스/스윙 구간을 정의하고
  구간-접촉 일치에 보상 — 좌우 π 오프셋이 대칭 걸음을 '정의 자체'로 강제.
  (Stage3 대칭 벌점 2회 실패 교훈: 수렴 후 패치가 아니라 초기부터 위상으로 형성)
- 접촉 관측은 실물 RA30P 발바닥 힘센서 대응 — 이진값(>5N)이라 sim2real 갭 최소.

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg

# 게이트 주파수 [Hz] — 케이던스 2.8보/s (= 1.4Hz × 양발). Stage3 실측 뚜벅걸음 대역.
GAIT_FREQ_HZ = 1.4


def _gait_phase(env, freq: float) -> torch.Tensor:
    """에피소드 경과시간 기반 게이트 위상 φ = 2π·f·t. (N,) [rad].

    episode_length_buf는 리셋 시 0 — 모든 env가 φ=0(왼발 스탠스 시작)에서 출발.
    관측(gait_clock)과 보상(gait_phase_reward)이 반드시 이 함수를 공유할 것
    (위상 정의가 어긋나면 클록이 무의미해진다).
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    return 2.0 * math.pi * freq * t


def gait_clock(env, freq: float = GAIT_FREQ_HZ) -> torch.Tensor:
    """게이트 클록 관측 [sin φ, cos φ]. (N, 2).

    실물 이식 시 배포 노드가 동일 공식(2π·1.4·t)으로 재현 가능 — 외부 입력 불필요.
    """
    phase = _gait_phase(env, freq)
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def feet_contact_binary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """발 접촉 이진 관측 (N, 2) — [왼발, 오른발] 순서, 법선힘 노름 > threshold[N].

    실물 RA30P 발바닥 힘센서 대응. sensor_cfg.body_ids는 SceneEntityCfg가
    '이름으로' 해석한 인덱스 (preserve_order=True 필수 — 접촉센서 바디 순서는
    USD DFS 순회라 좌/우가 이름 순서와 다를 수 있는 함정, log_picture/03 참조).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3)
    return (torch.norm(forces, dim=-1) > threshold).to(torch.float32)


def gait_phase_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    freq: float = GAIT_FREQ_HZ,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Siekmann 2021 간이판 주기 보상 — 위상-접촉 일치도. 값 0~1.

    왼발 스탠스 = sin(φ) ≥ 0 구간, 오른발 스탠스 = sin(φ) < 0 (π 오프셋).
    reward = Σ_발 [ I(자기 스탠스구간)·I(접촉) + I(스윙구간)·I(비접촉) ] / 2
           = ( I(왼스탠스)==I(왼접촉) + I(오른스탠스)==I(오른접촉) ) / 2.
    서기 명령(|cmd_xy| < 0.1)은 클록을 무시하고 양발 접촉 보상으로 대체
    — 서기 env가 제자리 발구름을 하지 않게.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3) [L, R]
    contact = torch.norm(forces, dim=-1) > force_threshold  # (N, 2) bool

    sin_phi = torch.sin(_gait_phase(env, freq))
    left_stance = sin_phi >= 0.0  # (N,) bool — 오른발 스탠스는 그 여집합 (π 오프셋)

    walk_reward = (
        (left_stance == contact[:, 0]).to(torch.float32)
        + ((~left_stance) == contact[:, 1]).to(torch.float32)
    ) * 0.5

    stand_reward = (contact[:, 0] & contact[:, 1]).to(torch.float32)

    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return torch.where(cmd > 0.1, walk_reward, stand_reward)
