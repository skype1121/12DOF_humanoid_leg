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

from isaaclab.envs import mdp
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
    서기 명령(|cmd_xy| + |wz| < 0.1)은 클록을 무시하고 양발 접촉 보상으로 대체
    — 서기 env가 제자리 발구름을 하지 않게.

    게이트에 |wz| 포함 (Stage4 v2 적대 리뷰 수정): 선속도 노름만 보면
    제자리회전 env(vx=vy=0, wz≠0)가 '서기'로 오분류되어 양발 접지 보상(2.0)이
    스텝 턴을 벌하고 접지 비틀기(요-슬립 게이밍)를 정확히 조장한다.
    서기 env는 명령 3성분 전부 0이라 이 변경에 영향 없음.
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

    cmd_vec = env.command_manager.get_command(command_name)
    cmd = torch.norm(cmd_vec[:, :2], dim=1) + torch.abs(cmd_vec[:, 2])
    return torch.where(cmd > 0.1, walk_reward, stand_reward)


def stand_still_full_command(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """서기(전 명령 ≈ 0) env만 기본자세 이탈 벌점 — SE(2) 3성분 전부 게이트.

    isaaclab_tasks의 stand_still_joint_deviation_l1은 선속도 노름만 보고
    게이트해서 제자리회전 env(vx=vy=0, wz≠0)까지 '서기'로 오분류 —
    스텝 턴에 필요한 관절 이탈을 벌점하는 충돌 (Stage4 v2 적대 리뷰 수정).
    v2 cfg가 stand_still.func를 이 함수로 교체한다 (v1 계열은 원본 유지).
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    return mdp.joint_deviation_l1(env, asset_cfg) * (cmd_mag < command_threshold)


def foot_yaw_slip(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 5.0,
    cap: float = 20.0,
) -> torch.Tensor:
    """접지 요-슬립 벌점 (Stage4 v2) — 접촉 중인 발의 월드 z축 |각속도| 합.

    제자리회전 명령을 '접지한 발바닥 비틀기(트위스트 슬립)'로 게이밍하지 않고
    발을 들어 딛는 스텝 턴으로 돌도록 유도. cmd 게이팅 없음 — 전진 보행 중에도
    접지발의 요 회전은 물리적으로 슬립이므로 일괄 벌점 (양수 반환, weight 음수).

    구현: 접촉(법선힘 노름 > threshold[N]) 마스크 × |body_ang_vel_w[..., 2]| 합,
    cap으로 클램프. 캡은 방폭 캡핑 5종과 같은 규약 (mdp_rewards 참조) — 실측
    정상값 최대 3.1(4096env 스모크) 대비 ~6배 여유라 학습 신호 불변, 계단
    모서리 물리폭발 시 각속도 스파이크(수백 rad/s)의 보상 오염만 차단.
    weight -0.5 × cap 20 = 스텝당 최대 -10 (다른 캡 항들과 동일 대역).
    sensor_cfg(접촉센서)와 asset_cfg(로봇 바디)는 반드시 같은 발 순서
    [왼발, 오른발] + preserve_order=True로 페어를 맞출 것 — 접촉센서 바디
    나열은 USD DFS 순회라 이름 순서와 다를 수 있는 함정 (log_picture/03).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 발, 3)
    in_contact = (torch.norm(forces, dim=-1) > threshold).to(torch.float32)  # (N, 발)
    asset = env.scene[asset_cfg.name]
    foot_yaw_vel = torch.abs(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2])  # (N, 발)
    return (foot_yaw_vel * in_contact).sum(dim=1).clamp_max(cap)
