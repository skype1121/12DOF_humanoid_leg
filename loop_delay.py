#!/usr/bin/env python3
"""블랙박스에서 목표 명령 -> 관절 위치의 순수 지연과 1차 추종 시정수를 추정한다.

필수 외부 의존성은 numpy뿐이다.

사용 예:
    python loop_delay.py --selftest
    python loop_delay.py ~/logs/rl_bridge_20260905_120000.jsonl
    python loop_delay.py blackbox.jsonl --t-end 1234.5 --max-lag-ticks 15
"""

import argparse
import json
import math
import sys

import numpy as np


JOINT_NAMES = (
    "left_hip_f",
    "right_hip_f",
    "left_hip_a",
    "right_hip_a",
    "left_hip_r",
    "right_hip_r",
    "left_knee",
    "right_knee",
    "left_ankle_f",
    "right_ankle_f",
    "left_ankle_r",
    "right_ankle_r",
)

# 이보다 작은 목표 표준편차는 양자화와 자세 유지 오차만으로도 지연이 바뀔 수 있다.
MIN_TARGET_STD_DEG = 0.15
MIN_SAMPLES = 80


def _empty_joint(n_samples=0):
    return {
        "delay_ticks": None,
        "delay_ms": None,
        "tau_ms": None,
        "confidence": 0.0,
        "n_samples": int(n_samples),
    }


def _as_joint_vector(value):
    """길이 12인 유한 실수 벡터가 아니면 None을 반환한다."""
    # 명세는 리스트지만 기존(2026-08) 블랙박스는 target을 *_joint 키 dict로 썼다.
    if isinstance(value, dict):
        try:
            value = [
                value[name] if name in value else value[name + "_joint"]
                for name in JOINT_NAMES
            ]
        except KeyError:
            return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(array)):
        return None
    return array


def _prepare_rows(rows, t_end, max_lag_ticks):
    """원래 틱 인덱스를 보존한 채 입력을 배열화하고 공통 표본 마스크를 만든다."""
    kept = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            timestamp = float(row["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp):
            continue
        if t_end is not None and timestamp > float(t_end):
            continue
        kept.append((timestamp, row))

    if len(kept) < 2:
        raise ValueError("유효한 timestamp 행이 2개 미만입니다")

    timestamps = np.asarray([item[0] for item in kept], dtype=np.float64)
    positive_dt = np.diff(timestamps)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    if positive_dt.size == 0:
        raise ValueError("timestamp가 단조 증가하지 않거나 틱 주기를 구할 수 없습니다")
    tick_s = float(np.median(positive_dt))

    n_rows = len(kept)
    n_joints = len(JOINT_NAMES)
    targets = np.full((n_rows, n_joints), np.nan, dtype=np.float64)
    qpos_deg = np.full((n_rows, n_joints), np.nan, dtype=np.float64)
    base_valid = np.zeros(n_rows, dtype=bool)

    for i, (_, row) in enumerate(kept):
        target = _as_joint_vector(row.get("targets_deg_joint"))
        qpos = _as_joint_vector(row.get("qpos_rad"))
        try:
            blend_alpha = float(row.get("blend_alpha", float("nan")))
        except (TypeError, ValueError):
            blend_alpha = float("nan")
        enabled = (
            row.get("armed") is True
            and row.get("live") is True
            and math.isfinite(blend_alpha)
            and blend_alpha >= 1.0 - 1e-9
        )
        if target is not None:
            targets[i] = target
        if qpos is not None:
            qpos_deg[i] = np.rad2deg(qpos)
        base_valid[i] = enabled and target is not None and qpos is not None

    # 누락 행이나 큰 timestamp 점프를 가로질러 target[k-d]를 연결하지 않는다.
    contiguous = np.zeros(n_rows, dtype=bool)
    contiguous[0] = base_valid[0]
    if n_rows > 1:
        gaps = np.diff(timestamps)
        good_gap = (gaps > 0.25 * tick_s) & (gaps < 1.8 * tick_s)
        contiguous[1:] = base_valid[1:] & base_valid[:-1] & good_gap

    run_length = np.zeros(n_rows, dtype=np.int64)
    for i in range(n_rows):
        if not base_valid[i]:
            continue
        if i > 0 and contiguous[i]:
            run_length[i] = run_length[i - 1] + 1
        else:
            run_length[i] = 1

    # 모든 d 후보가 정확히 같은 표본으로 비교되도록 최대 지연만큼 워밍업한다.
    eligible = np.flatnonzero(run_length >= max_lag_ticks + 2)
    return tick_s, targets, qpos_deg, eligible


def _fit_one_joint(target, qpos, sample_index, tick_s, max_lag_ticks):
    """관절 하나에 대해 지연 d와 1차 지연 tau를 격자 최소제곱으로 찾는다."""
    n_samples = int(sample_index.size)
    if n_samples < MIN_SAMPLES:
        return _empty_joint(n_samples)

    current_target = target[sample_index]
    target_std = float(np.std(current_target))
    if not math.isfinite(target_std) or target_std < MIN_TARGET_STD_DEG:
        return _empty_joint(n_samples)

    y = qpos[sample_index]
    y_prev = qpos[sample_index - 1]
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(y_prev))):
        return _empty_joint(n_samples)

    # 실제 틱 주기에 비례한 격자다. 20 ms 루프에서는 40/80 ms를 정확히 포함한다.
    tau_in_ticks = np.asarray(
        [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
        dtype=np.float64,
    )
    tau_s_grid = tau_in_ticks * tick_s
    score = np.full((max_lag_ticks + 1, tau_s_grid.size), np.inf, dtype=np.float64)
    gain_grid = np.full_like(score, np.nan)

    for delay in range(max_lag_ticks + 1):
        delayed_target = target[sample_index - delay]
        u_mean = float(np.mean(delayed_target))
        u_centered = delayed_target - u_mean
        u_energy = float(np.dot(u_centered, u_centered))
        if u_energy <= 1e-12:
            continue

        for tau_i, tau_s in enumerate(tau_s_grid):
            a = math.exp(-tick_s / float(tau_s))
            # q[k]-a*q[k-1] = c*target[k-d]+b. c=(1-a)*DC_gain 이다.
            z = y - a * y_prev
            z_mean = float(np.mean(z))
            c = float(np.dot(u_centered, z - z_mean) / u_energy)
            b = z_mean - c * u_mean
            residual = z - (c * delayed_target + b)
            score[delay, tau_i] = float(np.mean(residual * residual))
            gain_grid[delay, tau_i] = c / (1.0 - a)

    if not np.any(np.isfinite(score)):
        return _empty_joint(n_samples)

    best_flat = int(np.nanargmin(score))
    best_delay, best_tau_i = np.unravel_index(best_flat, score.shape)
    best_score = float(score[best_delay, best_tau_i])
    best_tau_s = float(tau_s_grid[best_tau_i])
    best_gain = float(gain_grid[best_delay, best_tau_i])

    # d별 최적 tau의 잔차 프로파일에서 최저점이 평평한지를 본다.
    delay_profile = np.min(score, axis=1)
    alternatives = [
        float(value)
        for delay, value in enumerate(delay_profile)
        if abs(delay - best_delay) >= 2 and math.isfinite(float(value))
    ]
    alternative_score = min(alternatives) if alternatives else best_score

    # 입력이 없는 같은 AR(1) 예측과 비교해 목표가 위치 변화를 설명하는 정도를 잰다.
    best_a = math.exp(-tick_s / best_tau_s)
    z_best = y - best_a * y_prev
    null_residual = z_best - float(np.mean(z_best))
    null_score = float(np.mean(null_residual * null_residual))
    explained = max(0.0, 1.0 - best_score / max(null_score, 1e-12))

    explained_power = max(null_score - best_score, 1e-12)
    clarity_raw = max(0.0, alternative_score - best_score) / explained_power
    clarity = min(1.0, clarity_raw / 0.35)
    excitation = min(1.0, max(0.0, (target_std - MIN_TARGET_STD_DEG) / 0.85))
    # 좌표 부호나 스케일이 틀린 적합은 숫자가 선명해도 신뢰하지 않는다.
    gain_quality = min(1.0, max(0.0, best_gain) / 0.5)
    if best_gain > 2.0:
        gain_quality *= max(0.0, 1.0 - (best_gain - 2.0) / 2.0)
    confidence = excitation * math.sqrt(max(0.0, explained * clarity)) * gain_quality

    return {
        "delay_ticks": int(best_delay),
        "delay_ms": float(best_delay * tick_s * 1000.0),
        "tau_ms": float(best_tau_s * 1000.0),
        "confidence": float(min(1.0, max(0.0, confidence))),
        "n_samples": n_samples,
    }


def estimate_loop_delay(rows, t_end=None, max_lag_ticks=15):
    """정책 목표에서 실측 관절 위치까지의 순수 지연과 추종 tau를 추정한다.

    반환 형식은 ``{"joints": {관절명: 결과}, "overall": 전체 결과}`` 이다.
    confidence가 0.1 미만인 관절은 전체 중앙값에서 제외한다.
    """
    if isinstance(max_lag_ticks, bool) or int(max_lag_ticks) != max_lag_ticks:
        raise ValueError("max_lag_ticks는 0 이상의 정수여야 합니다")
    max_lag_ticks = int(max_lag_ticks)
    if max_lag_ticks < 0:
        raise ValueError("max_lag_ticks는 0 이상의 정수여야 합니다")

    tick_s, targets, qpos_deg, sample_index = _prepare_rows(
        rows, t_end, max_lag_ticks
    )
    joints = {}
    used_delays = []
    for joint_i, name in enumerate(JOINT_NAMES):
        result = _fit_one_joint(
            targets[:, joint_i],
            qpos_deg[:, joint_i],
            sample_index,
            tick_s,
            max_lag_ticks,
        )
        joints[name] = result
        if result["delay_ticks"] is not None and result["confidence"] >= 0.1:
            used_delays.append(result["delay_ticks"])

    overall = {
        "median_ticks": float(np.median(used_delays)) if used_delays else None,
        "tick_ms": float(tick_s * 1000.0),
        "n_joints_used": len(used_delays),
    }
    return {"joints": joints, "overall": overall}


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL {line_number}행 파싱 실패: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL {line_number}행이 객체(dict)가 아닙니다")
            rows.append(row)
    return rows


def _fmt(value, digits=2):
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _print_result(result):
    print("joint             delay[tick]  delay[ms]  tau[ms]  confidence  samples")
    print("----------------  -----------  ---------  -------  ----------  -------")
    for name in JOINT_NAMES:
        item = result["joints"][name]
        delay = "-" if item["delay_ticks"] is None else str(item["delay_ticks"])
        print(
            f"{name:16s}  {delay:>11s}  {_fmt(item['delay_ms']):>9s}"
            f"  {_fmt(item['tau_ms']):>7s}  {item['confidence']:10.3f}"
            f"  {item['n_samples']:7d}"
        )
    overall = result["overall"]
    print(
        "overall: median_ticks={}  tick_ms={}  n_joints_used={}".format(
            _fmt(overall["median_ticks"], 1),
            _fmt(overall["tick_ms"], 3),
            overall["n_joints_used"],
        )
    )


def _smooth_targets(rng, time_s):
    """관절마다 0.5~3 Hz 성분만 합친 재현 가능한 매끄러운 목표를 만든다."""
    targets = np.empty((time_s.size, len(JOINT_NAMES)), dtype=np.float64)
    for joint_i in range(len(JOINT_NAMES)):
        signal = np.zeros(time_s.size, dtype=np.float64)
        for _ in range(8):
            frequency = rng.uniform(0.5, 3.0)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            amplitude = rng.uniform(0.35, 0.9)
            signal += amplitude * np.sin(2.0 * math.pi * frequency * time_s + phase)
        targets[:, joint_i] = signal + rng.uniform(-8.0, 8.0)
    return targets


def _synthetic_rows(delay_ticks, tau_ms, seed):
    tick_s = 0.020
    n_ticks = 1400
    rng = np.random.default_rng(seed)
    time_s = np.arange(n_ticks, dtype=np.float64) * tick_s
    targets = _smooth_targets(rng, time_s)
    qpos_deg = np.empty_like(targets)
    qpos_deg[0] = targets[0]
    a = math.exp(-tick_s / (tau_ms / 1000.0))
    for tick in range(1, n_ticks):
        source_tick = max(0, tick - delay_ticks)
        qpos_deg[tick] = a * qpos_deg[tick - 1] + (1.0 - a) * targets[source_tick]

    # 요구된 0.05도 관측 잡음과 0.1도 양자화를 동시에 적용한다.
    measured_deg = qpos_deg + rng.normal(0.0, 0.05, size=qpos_deg.shape)
    measured_deg = np.round(measured_deg / 0.1) * 0.1
    qvel_rad_s = np.gradient(np.deg2rad(measured_deg), tick_s, axis=0)

    rows = []
    for tick in range(n_ticks):
        rows.append(
            {
                "timestamp": float(time_s[tick]),
                "targets_deg_joint": targets[tick].tolist(),
                "qpos_rad": np.deg2rad(measured_deg[tick]).tolist(),
                "qvel_rad_s": qvel_rad_s[tick].tolist(),
                "blend_alpha": 0.0 if tick < 25 else 1.0,
                "armed": True,
                "live": True,
                "foot_n": [86.45, 86.45],
            }
        )
    return rows


def _run_selftest():
    print("합성 self-test: 50 Hz, 0.5~3 Hz 목표, noise=0.05 deg, quant=0.1 deg")
    print("true_d  true_tau  est_d  joint_range  est_tau_med  conf_med  used  result")
    print("------  --------  -----  -----------  -----------  --------  ----  ------")
    passed = 0
    total = 0
    for delay_ticks in (4, 6, 8):
        for tau_ms in (40, 80):
            total += 1
            rows = _synthetic_rows(delay_ticks, tau_ms, seed=1000 + 10 * delay_ticks + tau_ms)
            result = estimate_loop_delay(rows, max_lag_ticks=15)
            estimates = [
                item["delay_ticks"]
                for item in result["joints"].values()
                if item["delay_ticks"] is not None
            ]
            taus = [
                item["tau_ms"]
                for item in result["joints"].values()
                if item["tau_ms"] is not None
            ]
            confidences = [item["confidence"] for item in result["joints"].values()]
            estimated = result["overall"]["median_ticks"]
            ok = (
                estimated is not None
                and abs(estimated - delay_ticks) <= 1.0
                and result["overall"]["n_joints_used"] > 0
            )
            passed += int(ok)
            joint_range = "-" if not estimates else f"{min(estimates)}..{max(estimates)}"
            print(
                f"{delay_ticks:6d}  {tau_ms:8.0f}  {_fmt(estimated, 1):>5s}"
                f"  {joint_range:>11s}  {_fmt(np.median(taus) if taus else None, 1):>11s}"
                f"  {np.median(confidences):8.3f}"
                f"  {result['overall']['n_joints_used']:4d}  {'PASS' if ok else 'FAIL'}"
            )
    print(f"SELFTEST: {passed}/{total} PASS")
    return passed == total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blackbox", nargs="?", help="입력 블랙박스 JSONL")
    parser.add_argument("--selftest", action="store_true", help="합성 데이터 자가시험")
    parser.add_argument("--t-end", type=float, default=None, help="분석 종료 timestamp(초)")
    parser.add_argument("--max-lag-ticks", type=int, default=15, help="최대 지연 후보(틱)")
    args = parser.parse_args(argv)

    if args.selftest:
        return 0 if _run_selftest() else 1
    if not args.blackbox:
        parser.error("blackbox JSONL 경로 또는 --selftest가 필요합니다")

    try:
        rows = _load_jsonl(args.blackbox)
        result = estimate_loop_delay(rows, args.t_end, args.max_lag_ticks)
    except (OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
