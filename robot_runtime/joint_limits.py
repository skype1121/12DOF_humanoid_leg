"""12관절 절대각 제한 공용 로더 — config/joint_limits_12dof.json 단일 원본.

표준 라이브러리만 사용 (젯슨 노드·UI·RL 러너·시뮬 컨트롤러 공용).
semantics: soft = 정상 작동 범위(초과 거부), hard = 기구 보호 한계(무조건 거부).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "joint_limits_12dof.json"
_cache = None


def get_absolute_limits_deg() -> dict:
    """{관절명: {soft_min, soft_max, hard_min, hard_max}} (deg). 파일은 1회 로드."""
    global _cache
    if _cache is None:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        limits = data["limits_deg"]
        for name, lim in limits.items():
            if not (lim["hard_min"] <= lim["soft_min"] < lim["soft_max"] <= lim["hard_max"]):
                raise ValueError(f"joint_limits_12dof.json 순서 위반: {name} {lim}")
        _cache = limits
    return _cache


def clamp_targets_rad(named_rad: dict, mode: str = "soft") -> dict:
    """{관절명: rad} 목표를 soft(기본)/hard 절대각 범위로 클램프해 반환."""
    limits = get_absolute_limits_deg()
    lo_k, hi_k = (f"{mode}_min", f"{mode}_max")
    out = {}
    for name, rad in named_rad.items():
        lim = limits.get(name)
        if lim is None:
            out[name] = rad
            continue
        lo = math.radians(lim[lo_k])
        hi = math.radians(lim[hi_k])
        out[name] = min(max(float(rad), lo), hi)
    return out


def check_absolute_deg(joint: str, absolute_deg: float) -> str:
    """절대각 검사: 'ok' | 'soft_limit_exceeded' | 'hard_limit_exceeded' | 'unknown_joint'."""
    lim = get_absolute_limits_deg().get(joint)
    if lim is None:
        return "unknown_joint"
    if not (lim["hard_min"] <= absolute_deg <= lim["hard_max"]):
        return "hard_limit_exceeded"
    if not (lim["soft_min"] <= absolute_deg <= lim["soft_max"]):
        return "soft_limit_exceeded"
    return "ok"
