"""캡핑된 페널티 항 — 물리 폭발이 가치함수를 오염시키지 못하게 상한.

배경: 계단 모서리 접촉 + 페이로드 + 자기충돌 조합에서 드물게 PhysX 폭발
(관절가속² ~1e12)이 발생, 보상 −1e5 스파이크 → 가치함수 오염 → NaN 크래시.
종료 판정은 다음 스텝에 잡지만 해당 스텝 보상은 이미 배치에 들어간다.

캡 값은 정상 보행 대비 10~100배 여유 — 학습 신호는 그대로, 폭발만 무력화.
(정상 보행 실측: acc² 합 ~1e4-1e5, vel² 합 ~50-300, ang_vel_xy² ~0.1-1)
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def joint_acc_l2_capped(env, cap: float = 1.0e7, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    v = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return v.clamp_max(cap)


def joint_vel_l2_capped(env, cap: float = 5.0e3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    v = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return v.clamp_max(cap)


def ang_vel_xy_l2_capped(env, cap: float = 100.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    v = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return v.clamp_max(cap)


def lin_vel_z_l2_capped(env, cap: float = 25.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    v = torch.square(asset.data.root_lin_vel_b[:, 2])
    return v.clamp_max(cap)


def contact_forces_capped(env, threshold: float, sensor_cfg: SceneEntityCfg, cap: float = 5.0e3):
    sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    return torch.sum(violation.clip(min=0.0), dim=1).clamp_max(cap)