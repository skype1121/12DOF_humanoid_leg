#!/usr/bin/env python3
"""무개입 낙하 판정기 — rl_bridge 블랙박스(jsonl)에서 자립/기울임을 실측 판정.

5차(v6b) 무개입 낙하는 눈과 스냅샷으로 읽었다. 그때 확정된 사실:
"균형 잡힌 몸 위에서 1초부터 전방으로 단조 누적(+1.6→+19°), 발진도 스텝도
없음" = payload 뒤 배치 학습의 순수 산물. 6차(v7)의 질문은 단 하나 —
**그 단조 전방 밀기가 사라졌는가**. 이 스크립트가 그 답을 숫자로 낸다.

사용:
    python3 jetson/scripts/analyze_fall.py ~/rl_bridge_bb/rl_bridge_XXXX.jsonl
    python3 jetson/scripts/analyze_fall.py <v7.jsonl> --compare <v6b.jsonl>

판정 지표는 원격 오판 사고(IMU 스냅샷 1장으로 '섰다' 오판) 이후 확립된
**발력 총합 %**를 1순위로, 기울임 시계열을 2순위로 쓴다.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

try:
    import numpy as np          # 루프지연 추정에만 사용 — 없으면 그 절만 생략
except ImportError:             # pragma: no cover
    np = None

BODY_WEIGHT_N = 172.9      # 실물 17.62kg(저울, 2026-09-04) × 9.81 — 구 117.7(12kg 추정) 폐기
#: 0906 발력 어댑터 재보정(npc 0.01804→0.02650) 이전 블랙박스의 foot_n 은 참값의 117.7/172.9 배 —
#: 첫 live 틱 timestamp 가 컷오프(2026-09-06 00:00 KST) 이전이면 ×1.469 자동 보정 (--foot_scale 로 강제).
#: 백분율은 같은 카운트 앵커를 나누므로 보정 전후 모두 유효하다(4차 "90%" 는 90%).
LEGACY_FOOT_SCALE = 172.9 / 117.7
LEGACY_FOOT_CUTOFF = datetime(2026, 9, 6, tzinfo=timezone(timedelta(hours=9))).timestamp()
#: 관절별 토크 한계[Nm] — 시뮬 액추에이터 설정(biped12_cfg.py)과 동일 축:
#: AK70-10 25.0 / 발목롤 AK45-36 24.0. 블랙박스 tau_est_nm(0905 신설) 이 있을 때만 토크 요약.
JOINT_ORDER = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
               "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
               "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
TAU_LIMIT_NM = [25.0] * 10 + [24.0, 24.0]
TAU_SAT_FRAC = 0.95       # 시뮬 명목 판정(eval_drift_dr 물리 실현성)과 같은 문턱(|τ| ≥ 0.95·한계) — 0905 감사 C1
TAU_STALE_S = 0.1         # stage8 status 가 이보다 오래된 틱의 tau_est 는 동결값 → 토크표에서 제외 (0905 감사 C4)
ANKLE_R_JOINTS = ("left_ankle_r_joint", "right_ankle_r_joint")   # AK45-36: torque_est 디코드 스케일 미확정 (감사 C3)
#: 루프지연 추정 (0905 감사 B4): 목표(targets_deg_joint = 러너 출력, 블렌딩 전) → 실측 qpos 를
#: "순수 지연 d틱 + 1차 지연 τ" 모델로 격자 적합. 학습 대역 v15~v18 3~6틱 / v19 3~8틱, 0803 실측 4~8틱.
LAG_MAX_TICKS = 15
LAG_TAU_MS = (0, 20, 40, 60, 80, 120, 160, 240)
LAG_MIN_STD_DEG = 0.3     # 목표 신호 표준편차가 이보다 작은 관절은 '정지' — 추정 제외
LAG_MIN_R2 = 0.5          # 적합도 미달 관절은 중앙값 집계에서 제외
LAG_TRAIN_BAND = (3, 8)   # 현행 학습 대역 상한(v19); v15~v18 은 3~6
LOAD_STANDING = 85.0       # 이 % 이상이면 하중을 정책이 받고 있다고 본다
LOAD_LOST = 50.0           # 이 % 아래로 떨어지면 줄/사람에게 넘어간 것
LEAN_FALL_DELTA = 15.0     # 시작 자세 대비 이만큼 더 기울면 자립 종료
EARLY_WINDOW_S = 3.0       # "1초부터 단조 누적" 을 잡아내는 초기 창
MIN_GRADE_S = 1.5          # 이보다 짧은 구간은 등급을 매기지 않는다

# v6b 5차 무개입 낙하 실측 기준선 — 이 스크립트로 원본 블랙박스
# (0804 rl_bridge_20260804_130927.jsonl)를 재분석해 확정한 값.
V6B_BASELINE = {"slope": 4.06, "lean_end": 19.2, "standing_s": 1.4}


def grav_to_angles(quat_wxyz):
    """quat(wxyz) → (전방기울임°, 좌우기울임°, 총기울기°).

    브리지와 동일한 projected-gravity 식을 쓴다 (rl_bridge_node.tick).
    gx>0 = 앞으로 기울어짐 (y축 +a 회전에서 gx=sin(a) — 실물 IMU 규약과 일치).
    """
    w, x, y, z = quat_wxyz
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    fwd = math.degrees(math.atan2(gx, -gz))
    lat = math.degrees(math.atan2(gy, -gz))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
    return fwd, lat, tilt


def linfit(xs, ys):
    """최소제곱 (기울기, 절편, R²). 표본 부족·분산 0이면 (0,평균,0)."""
    n = len(xs)
    if n < 3:
        return 0.0, (sum(ys) / n if n else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    icept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy < 1e-12:
        return slope, icept, 0.0
    ss_res = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
    return slope, icept, max(0.0, 1.0 - ss_res / syy)


def monotonic_frac(xs, ys, bin_s=0.2):
    """0.2s 빈으로 다운샘플한 뒤 '증가한 구간' 비율.

    20ms 원틱으로 부호를 세면 센서 노이즈가 0.5로 희석된다. v6b의 단조
    누적은 초 단위 추세였으므로 빈 평균 후 차분해야 실체가 보인다.
    """
    if not xs:
        return 0.0, 0
    bins = {}
    for t, v in zip(xs, ys):
        k = int(t / bin_s)
        bins.setdefault(k, []).append(v)
    keys = sorted(bins)
    means = [sum(bins[k]) / len(bins[k]) for k in keys]
    if len(means) < 2:
        return 0.0, len(means)
    ups = sum(1 for a, b in zip(means, means[1:]) if b > a)
    return ups / (len(means) - 1), len(means)


def sparkline(vals, width=60):
    """시계열을 한 줄로. 축(0°)은 '-' 로 표시."""
    if not vals:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    step = max(1, len(vals) // width)
    samp = [vals[i] for i in range(0, len(vals), step)][:width]
    lo, hi = min(samp), max(samp)
    if hi - lo < 1e-9:
        return "─" * len(samp)
    out = []
    for v in samp:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def load(path):
    """jsonl → (틱 레코드 리스트, 이벤트 리스트). 손상 줄은 건너뛴다."""
    ticks, events, bad = [], [], 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if "event" in o:
                events.append(o)
            elif "quat_wxyz" in o:
                ticks.append(o)
    return ticks, events, bad


def torque_summary(rows):
    """관절별 |τ| 피크·RMS·포화 비율(틱 기준) — 시뮬 명목 판정의 물리 실현성 표와 같은 축.
    tau_est_nm 이 한 틱도 없으면 None (구 블랙박스)."""
    per = [[] for _ in JOINT_ORDER]
    for r in rows:
        tau = r.get("tau")
        if not tau or len(tau) != len(JOINT_ORDER):
            continue
        for k, v in enumerate(tau):
            if v is not None:
                per[k].append(abs(float(v)))
    if not any(per):
        return None
    out = []
    for k, name in enumerate(JOINT_ORDER):
        xs = per[k]
        if not xs:
            out.append((name, None, None, None, 0)); continue
        lim = TAU_LIMIT_NM[k]
        peak = max(xs)
        rms = math.sqrt(sum(x * x for x in xs) / len(xs))
        sat = 100.0 * sum(1 for x in xs if x >= TAU_SAT_FRAC * lim) / len(xs)
        out.append((name, peak, rms, sat, len(xs)))
    return out


def _first_order(ud, a, n):
    """1차 저역(EMA 계수 a)을 지수 커널 컨볼루션으로 — 초기 과도는 호출자가 warm 구간을 버린다."""
    if a >= 1.0:
        return ud[:n].copy(), 1
    L = int(math.ceil(math.log(1e-4) / math.log(1.0 - a))) + 1
    kern = a * (1.0 - a) ** np.arange(L)
    return np.convolve(ud, kern)[:n], L


def _fit_lag(u, y, dt, max_d=LAG_MAX_TICKS, taus_ms=LAG_TAU_MS):
    """y ≈ 1차지연(τ){u 를 d 틱 지연} — 잔차 RMS 최소 (d, τ). 반환 (d, τ_ms, R²).
    평균 오프셋(영점·PD 처짐)은 지연과 무관하므로 잔차에서 뺀다. 초기 과도(warm)는 공통으로 버린다."""
    n = len(u)
    best = (None, None, -1.0)
    for tau_ms in taus_ms:
        a = 1.0 if tau_ms == 0 else dt / (tau_ms / 1000.0 + dt)
        for d in range(0, max_d + 1):
            ud = np.concatenate([np.full(d, u[0]), u[:n - d]])
            if a >= 1.0:
                yh, L = ud, 1
            else:
                yh, L = _first_order(ud, a, n)
            warm = max(max_d, L)
            if n - warm < 40:
                continue
            res = y[warm:] - yh[warm:]
            res = res - res.mean()
            y0 = y[warm:] - y[warm:].mean()
            r2 = 1.0 - float(np.mean(res ** 2)) / (float(np.mean(y0 ** 2)) + 1e-12)
            if r2 > best[2]:
                best = (d, tau_ms, r2)
    return best


def clamp_summary(rows, t_end):
    """stage8 soft 클램프 증가분(자립 구간·전 구간)과 거부 사유 — 블랙박스에 없으면 None (0906 이전 브리지)."""
    have = [r for r in rows if r.get("clamp") is not None]
    if not have:
        return None
    stand = [r for r in have if r["t"] <= t_end]
    def delta(seq):
        if len(seq) < 2:
            return 0, {}
        d_tot = int(seq[-1]["clamp"]) - int(seq[0]["clamp"])
        per = {}
        a, b = seq[0].get("clamp_j") or [], seq[-1].get("clamp_j") or []
        for k, name in enumerate(JOINT_ORDER):
            if k < len(a) and k < len(b) and a[k] is not None and b[k] is not None and b[k] - a[k] > 0:
                per[name] = int(b[k] - a[k])
        return d_tot, per
    d_stand, per_stand = delta(stand)
    d_all, per_all = delta(have)
    first_clamp_t = next((r["t"] for i, r in enumerate(have[1:], 1) if r["clamp"] > have[i - 1]["clamp"]), None)
    rejects = sorted({str(r["reject"]) for r in have if r.get("reject")})
    return {"stand": d_stand, "per_stand": per_stand, "all": d_all, "per_all": per_all,
            "first_t": first_clamp_t, "rejects": rejects}


def loop_delay_summary(rows, t_end):
    """관절별 루프지연 추정 — 자립 구간(t≤t_end)·블렌딩 완료(alpha≥0.99) 틱만. numpy 없으면 None."""
    if np is None:
        return None
    sel = [r for r in rows if r["t"] <= t_end and r["alpha"] >= 0.99
           and isinstance(r.get("u"), dict) and r.get("q")]
    if len(sel) < 60:
        return {"n": len(sel), "joints": [], "median_ticks": None, "tick_ms": None, "n_good": 0,
                "why": "블렌딩 완료 후 자립 표본 60틱 미만"}
    ts = np.array([r["ts"] for r in sel], dtype=float)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.02
    U = np.array([[float(r["u"].get(j, float("nan"))) for j in JOINT_ORDER] for r in sel])
    Q = np.degrees(np.array([r["q"] for r in sel], dtype=float))
    joints = []
    for k, name in enumerate(JOINT_ORDER):
        u, y = U[:, k], Q[:, k]
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(y)) or u.std() < LAG_MIN_STD_DEG:
            joints.append((name, None, None, None, "정지(목표 분산 부족)"))
            continue
        d, tau, r2 = _fit_lag(u, y, dt)
        joints.append((name, d, tau, r2, "" if (r2 is not None and r2 >= LAG_MIN_R2) else "적합도 미달"))
    good = [d for _, d, _, r2, note in joints if d is not None and not note]
    return {"n": len(sel), "tick_ms": dt * 1000.0, "joints": joints, "n_good": len(good),
            "median_ticks": float(np.median(good)) if good else None}


def lag_selftest():
    """합성 목표(0.5~3Hz 정현 합) → 지연 d·1차 τ·잡음 0.05°·양자화 0.1° 로 실측을 만들고 d 를 ±1틱 안에 회수하는지."""
    assert np is not None, "numpy 필요"
    rng = np.random.default_rng(0)
    dt, n = 0.02, 1500
    t = np.arange(n) * dt
    fails = []
    print("  d[틱]  τ[ms]  → d̂  τ̂    R²")
    for d in (4, 6, 8):
        for tau in (40, 80):
            u = sum(rng.uniform(0.5, 2.0) * np.sin(2 * math.pi * f * t + rng.uniform(0, 6.28))
                    for f in (0.5, 1.1, 1.9, 2.7))
            ud = np.concatenate([np.full(d, u[0]), u[:n - d]])
            a = dt / (tau / 1000.0 + dt)
            y = np.empty(n); s = ud[0]
            for i in range(n):
                s = s + a * (ud[i] - s); y[i] = s
            y = np.round((y + 0.7 + rng.normal(0, 0.05, n)) / 0.1) * 0.1   # 오프셋 0.7° + 잡음 + 0.1° 양자화
            dh, th, r2 = _fit_lag(u, y, dt)
            ok = dh is not None and abs(dh - d) <= 1
            print(f"  {d:4d}  {tau:5d}   → {dh}  {th}  {r2:.3f}  {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append((d, tau, dh, th))
    # 플럼빙: rows 형태로 summary 호출
    rows = [{"t": i * dt, "alpha": 1.0, "ts": 1000.0 + i * dt,
             "u": {j: float(u[i]) for j in JOINT_ORDER}, "q": [math.radians(float(y[i]))] * 12}
            for i in range(n)]
    s = loop_delay_summary(rows, t_end=1e9)
    assert s and s["median_ticks"] is not None and abs(s["median_ticks"] - 8) <= 1, s
    assert not fails, f"루프지연 회수 실패: {fails}"
    print(f"[selftest] analyze_fall 루프지연 추정: 6조합 ±1틱 회수 · summary 중앙 {s['median_ticks']:.0f}틱 — ALL PASS")
    return 0


def analyze(path, foot_scale=None):
    ticks, events, bad = load(path)
    if not ticks:
        raise SystemExit(f"[실패] 틱 레코드가 없습니다: {path}")

    # t0 = 정책이 실제로 모터를 지휘하기 시작한 첫 틱 (live+armed).
    # 브리지 기동 시각이 아니다 — ARM 전 관측만 흐르는 구간이 길다.
    armed = [t for t in ticks if t.get("live") and t.get("armed")]
    if not armed:
        raise SystemExit(
            f"[실패] live+armed 구간이 없습니다 (정책 미투입): {path}\n"
            f"       총 틱 {len(ticks)}개 — ARM_ALL 이 안 됐거나 브리지가 "
            f"--live 없이 떴을 수 있습니다.")
    t0 = armed[0]["timestamp"]
    scale = foot_scale if foot_scale is not None else (
        LEGACY_FOOT_SCALE if t0 < LEGACY_FOOT_CUTOFF else 1.0)

    rows = []
    for t in armed:
        rel = t["timestamp"] - t0
        fwd, lat, tilt = grav_to_angles(t["quat_wxyz"])
        fn = t.get("foot_n") or [0.0, 0.0]
        load_pct = 100.0 * scale * (float(fn[0]) + float(fn[1])) / BODY_WEIGHT_N
        rows.append({"t": rel, "fwd": fwd, "lat": lat, "tilt": tilt,
                     "load": load_pct, "contact": t.get("contact"),
                     "alpha": t.get("blend_alpha", 1.0),
                     "tau": t.get("tau_est_nm"),
                     "status_age": t.get("status_age_s"),    # stage8 status 신선도 (0906 감사 C4)
                     "u": t.get("targets_deg_joint"), "q": t.get("qpos_rad"), "ts": t["timestamp"],
                     "clamp": t.get("soft_clamp_total"), "clamp_j": t.get("soft_clamp_counts"),
                     "reject": t.get("last_reject_reason")})   # stage8 클램프·거부 (0906 감사 B7)

    dur = rows[-1]["t"]

    # 분석 창 = 정책 투입(t0)부터 전부. **블렌딩 구간을 빼면 안 된다** —
    # 5차 v6b 실측에서 낙하 전체가 블렌딩 1.42s 안에서 끝났다(발력 96%를
    # 유지하다 alpha 0.7→0.9에서 붕괴). 블렌딩을 빼면 사건 자체가 사라져
    # "판정 불가"가 된다. 대신 낙상 시점의 alpha를 함께 보고해, 정책이
    # 얼마만큼의 권한을 쥔 상태에서 무너졌는지 드러낸다.
    settled = rows
    # 기준 자세 = 투입 직후 0.3s 평균. 착지 기하 탓에 시작부터 기울어 있는
    # 경우가 흔한데(0803 실기: 시작 −16°), 절대각으로 자립을 판정하면 발력
    # 92%로 멀쩡히 서 있던 시행이 "자립 0초"가 된다. 자립 여부는 **발력**이
    # 1순위이고, 각도는 **시작 대비 변화량**으로만 본다.
    base_win = [r["fwd"] for r in rows if r["t"] <= rows[0]["t"] + 0.3]
    fwd0 = sum(base_win) / len(base_win)

    # 실효 자립 = 발력을 잃거나(줄로 넘어감) 시작 대비 크게 기울 때까지.
    # 블렌딩 구간(alpha<1)은 정책 권한이 부분적이라 판정에서 제외한다.
    # **조건은 반드시 지속되어야 한다** — 10Hz 발압력의 1펄스 결손 하나가
    # 스텝테스트를 통째로 무효화한 전례가 있다. 순간값으로 끊으면 15초를
    # 버틴 시행이 0.4초로 찍힌다.
    # 핵심: **낙상은 되돌아오지 않는다.** 회복한 딥은 낙상이 아니다.
    # (0803 실기: 인수 0.8초에 발력이 17%로 꺼졌다가 회복해 15초를 섰다.
    #  순간값·지속조건 어느 쪽으로 봐도 그 시행은 "0.4초 자립"으로 오독된다.)
    def terminal_fall(pred, hold_s):
        """pred가 로그 끝까지 회복 없이 이어진 구간의 시작. 없으면 None."""
        t_bad = None
        for r in settled:
            if pred(r):
                if t_bad is None:
                    t_bad = r["t"]
            else:
                t_bad = None          # 회복했다 = 딥이었다
        if t_bad is None or settled[-1]["t"] - t_bad < hold_s:
            return None               # 끝자락 순간값은 낙상으로 세지 않는다
        return t_bad

    standing_s = dur
    stand_reason = "끝까지 유지"
    t_load = terminal_fall(lambda r: r["load"] < LOAD_LOST, 0.30)
    t_lean = terminal_fall(
        lambda r: abs(r["fwd"] - fwd0) > LEAN_FALL_DELTA, 0.20)
    if t_load is not None and (t_lean is None or t_load <= t_lean):
        standing_s, stand_reason = t_load, f"발력 {LOAD_LOST:.0f}% 미만 (미회복)"
    elif t_lean is not None:
        standing_s, stand_reason = (
            t_lean, f"시작 대비 {LEAN_FALL_DELTA:.0f}° 초과 기울임 (미회복)")

    # 회복한 딥은 낙상이 아니지만 그 자체로 진단 가치가 있다 (인수 전이 손실).
    dips = [r for r in settled
            if r["load"] < LOAD_LOST and r["t"] < standing_s]

    early = [r for r in rows if r["t"] <= EARLY_WINDOW_S]
    slope, _, r2 = linfit([r["t"] for r in early], [r["fwd"] for r in early])
    mono, nbins = monotonic_frac([r["t"] for r in early],
                                 [r["fwd"] for r in early])

    # 발력은 **자립 구간에서만** 평균낸다. 낙상 후 구간을 섞으면 멀쩡히
    # 서 있다가 넘어진 시행이 "처음부터 줄에 매달려 있었다"로 오독된다.
    loads = [r["load"] for r in rows if r["t"] <= standing_s]
    # 낙상 순간 정책이 쥐고 있던 권한 — 블렌딩 중 무너졌는지 완전 인수 후
    # 무너졌는지가 원인 해석을 가른다.
    alpha_at_fall = next((r["alpha"] for r in rows if r["t"] >= standing_s), 1.0)
    # 하중이 처음 빠지기 시작한 순간과 그때의 정책 권한. v6b는 alpha가
    # 0.7→0.9로 오르는 구간에서 발력 96%가 무너졌다 = 정책이 밀어낸 증거.
    depart = next((r for r in rows
                   if r["t"] <= standing_s and r["load"] < LOAD_STANDING), None)
    fwds = [r["fwd"] for r in rows]

    return {
        "path": path, "bad_lines": bad, "n_ticks": len(rows),
        "ckpt": armed[0].get("checkpoint", "?"),
        "dur": dur, "hz": (len(rows) - 1) / dur if dur > 0 else 0.0,
        "standing_s": standing_s, "stand_reason": stand_reason,
        "settled_s": dur, "alpha_at_fall": alpha_at_fall,
        "depart_t": depart["t"] if depart else None,
        "depart_alpha": depart["alpha"] if depart else None,
        # 가드가 자립 종료 직후(0.5s 내)에 울렸는지 — 그때만 "가드가 시행을
        # 끊었다"고 말할 수 있다. 넘어진 뒤 발이 뜨면서 울린 lift_guard를
        # 시행 무효 사유로 쓰면 진짜 낙상 데이터를 버리게 된다.
        "guard_ended": any(
            e.get("event") in ("osc_guard", "osc_guard_tgt", "lift_guard")
            and standing_s - 0.2 <= e["timestamp"] - t0 <= standing_s + 0.5
            for e in events),
        "dip_n": len(dips), "dip_t": dips[0]["t"] if dips else None,
        "dip_min": min((r["load"] for r in dips), default=None),
        "guard_events": [e.get("event") for e in events
                         if e.get("event") in ("osc_guard", "osc_guard_tgt",
                                               "lift_guard")],
        "slope": slope, "slope_r2": r2, "mono": mono, "nbins": nbins,
        "fwd0": fwd0, "fwd_start": rows[0]["fwd"], "fwd_end": rows[-1]["fwd"],
        "fwd_max": max(fwds), "fwd_min": min(fwds),
        "lat_absmax": max(abs(r["lat"]) for r in rows),
        "load_mean": (sum(loads) / len(loads)) if loads else 0.0,
        "load_min": min(loads) if loads else 0.0,
        "load_max": max(loads) if loads else 0.0,
        "events": [(e["timestamp"] - t0, e.get("event"), e.get("reason", ""))
                   for e in events if e["timestamp"] >= t0],
        "rows": rows,
        "foot_scale": scale,
        # 0905 감사 C2: 토크표는 자립 구간(t ≤ standing_s)만 — 낙상 후·STOP_ALL 홀드 틱을 넣으면
        # 시뮬(생존 스텝만 집계)과 축이 어긋난다. C4: status 가 TAU_STALE_S 보다 오래된 틱(동결값)은 제외.
        "tau": torque_summary([r for r in rows if r["t"] <= standing_s
                               and not (r["status_age"] is not None and r["status_age"] > TAU_STALE_S)]),
        "tau_stale": sum(1 for r in rows if r["t"] <= standing_s
                         and r["status_age"] is not None and r["status_age"] > TAU_STALE_S),
        "lag": loop_delay_summary(rows, standing_s),
        "clamp": clamp_summary(rows, standing_s),
    }


def verdict(a):
    """6차 핵심 질문에 대한 등급 + 가드 단서 (가드로 끝난 시행은 무조건 단서)."""
    grade, headline, action = _grade(a)
    if grade in ("A", "B") and a["guard_events"]:
        action += (f"  ※ 단, 이 시행은 가드"
                   f"({'/'.join(sorted(set(a['guard_events'])))})로 끝났습니다 — "
                   f"발진 원인을 규명하기 전에는 다음 단계로 넘어가지 마세요.")
    return grade, headline, action


def _grade(a):
    """등급 산출 본체."""
    slope, mono, stand = a["slope"], a["mono"], a["standing_s"]
    # 등급을 매길 자격이 있는 시행인지 먼저 본다. 0.8초짜리 가드 종료 로그에
    # 3빈 회귀로 "전방 단조 밀기"를 선언하면 그건 판정이 아니라 착각이다.
    if a["settled_s"] < MIN_GRADE_S:
        return ("—", f"판정 불가 — 정책 구동 구간이 {a['settled_s']:.1f}s "
                f"뿐입니다 (최소 {MIN_GRADE_S:.1f}s)",
                "시행이 너무 일찍 끝났습니다. 무엇이 끝냈는지(가드·착지·"
                "인장)를 해결하고 다시 받으세요.")
    if a["guard_ended"] and stand < 2.5:
        return ("—", f"판정 불가 — 가드({'/'.join(sorted(set(a['guard_events'])))})"
                f"가 {stand:.1f}s에 시행을 끊었습니다",
                "균형이 아니라 가드가 결말을 정했습니다. 가드 트립 원인을 "
                "먼저 규명해야 v7 판정이 성립합니다.")
    # 전방 단조 누적 = v6b 병증. 단조비율만 쓰면 "1초 평탄 후 램프" 같은
    # 실제 파형에서 앞 빈들이 비증가로 세어져 놓친다 — 직선성(R²)을 병기.
    pushing = slope > 2.0 and (mono >= 0.65 or a["slope_r2"] >= 0.7)
    if pushing and stand < 4.0:
        return ("C", "무변화 — v6b와 같은 전방 단조 밀기가 재현됐습니다",
                "payload 위치 가정은 원인이 아니었습니다. 다음 용의자: "
                "관측 지연(+1틱)·컴플라이언스(로터 사각)·발압력 지연.")
    if slope < -2.0 and mono <= 0.3 and stand < 4.0:
        return ("D", "역전 — 이번엔 뒤로 단조 누적됩니다 (v7 과교정)",
                "com_range를 v6b와 v7 사이(x 약 −0.05~0.0)로 좁혀 재베이크.")
    if stand >= 8.0 and abs(a["fwd_end"]) < 5.0:
        return ("A", "완전 성공 — 8초 이상 자립, 기울임 5° 이내 유지",
                "march 사다리(5초) 진입 가능합니다.")
    if stand >= 4.0 or (not pushing and stand >= 2.5):
        return ("B", "개선 확인 — 단조 밀기는 사라졌으나 자립이 짧습니다",
                "발진/횡동요/발력 손실 중 무엇이 끝냈는지 이벤트·발력을 보세요.")
    return ("C", "판정 애매 — 자립이 짧은데 전방 단조 패턴은 뚜렷하지 않습니다",
            "블렌딩 구간·가드 트립·착지 기하(초기 기울임)를 먼저 의심하세요.")


def report(a):
    print("=" * 68)
    print(f"무개입 낙하 판정 — {os.path.basename(a['path'])}")
    print("=" * 68)
    print(f"정책: {a['ckpt']}   정책구동 틱 {a['n_ticks']}개 / "
          f"{a['dur']:.1f}s ({a['hz']:.1f}Hz)"
          + (f"   손상줄 {a['bad_lines']}" if a["bad_lines"] else ""))
    print()
    print(f"■ 실효 자립 시간: {a['standing_s']:.1f}s  ({a['stand_reason']})"
          + (f"   ※ 낙상 시점 정책권한 alpha={a['alpha_at_fall']:.2f} "
             f"— 블렌딩 중 무너짐" if a["alpha_at_fall"] < 0.99 else ""))
    print(f"■ 발력 총합: 평균 {a['load_mean']:.0f}% / 최소 {a['load_min']:.0f}%"
          f" / 최대 {a['load_max']:.0f}%  (체중 {BODY_WEIGHT_N:.0f}N 기준"
          + (f", 구 어댑터 스케일 ×{a['foot_scale']:.3f} 보정" if abs(a["foot_scale"] - 1.0) > 1e-9 else "")
          + ")")
    print("   (자립 구간 기준 — 낙상 후 구간 제외)")
    if a["depart_t"] is not None:
        print(f"   하중 이탈 개시: {a['depart_t']:.2f}s "
              f"(그때 정책권한 alpha={a['depart_alpha']:.2f})"
              + ("  ← 정책이 권한을 쥐면서 무너졌습니다"
                 if a["depart_alpha"] < 0.99 else ""))
    if a["dip_t"] is not None:
        print(f"   ※ 회복된 발력 딥: {a['dip_t']:.1f}s에 최저 {a['dip_min']:.0f}% "
              f"({a['dip_n']}틱) — 낙상은 아니나 인수 전이 손실 신호")
    if a["load_mean"] >= LOAD_STANDING:
        print("   → 정책이 하중을 받고 있었습니다 (줄에 매달린 상태 아님).")
    elif a["load_mean"] >= LOAD_LOST:
        print("   → 대부분 받고 있으나 일부를 줄이 분담했습니다. "
              "줄 인장을 더 풀 여지가 있는지 확인하세요.")
    else:
        print("   → 하중이 줄/사람에게 상당 부분 넘어가 있었습니다. "
              "자립 판정으로 쓰기 전에 착지·인장 상태를 확인하세요.")
    print()
    print(f"■ 전방 기울임: 시작 {a['fwd_start']:+.1f}° → 종료 "
          f"{a['fwd_end']:+.1f}°  (최대 {a['fwd_max']:+.1f}° / 최소 "
          f"{a['fwd_min']:+.1f}°)   좌우 최대 |{a['lat_absmax']:.1f}|°")
    print(f"   기준 자세(투입 직후) {a['fwd0']:+.1f}° → 순변화 "
          f"{a['fwd_end'] - a['fwd0']:+.1f}°"
          + ("   ※ 시작부터 크게 기울어 있었습니다 — 착지 기하를 의심하세요"
             if abs(a["fwd0"]) > 8.0 else ""))
    print(f"■ 초기 {EARLY_WINDOW_S:.0f}s 추세: {a['slope']:+.2f}°/s "
          f"(R²={a['slope_r2']:.2f}), 단조증가 {a['mono']*100:.0f}% "
          f"({a['nbins']}빈)")
    print(f"   v6b 기준선: 약 {V6B_BASELINE['slope']:+.1f}°/s, "
          f"{V6B_BASELINE['lean_end']:.0f}°까지 누적, 자립 "
          f"~{V6B_BASELINE['standing_s']:.0f}s")
    print()
    print("  전방기울임 시계열 (좌=투입 직후, 우=종료)")
    print(f"  {sparkline([r['fwd'] for r in a['rows']])}")
    print(f"  {sparkline([r['load'] for r in a['rows']])}  ← 발력 총합")
    print()
    if a["events"]:
        print("■ 이벤트")
        for t, ev, why in a["events"]:
            print(f"   {t:6.2f}s  {ev}" + (f"  ({why})" if why else ""))
    else:
        print("■ 이벤트: 없음 (가드 무트립·STOP_ALL 없음)")
    if a.get("tau"):
        print(f"■ 실물 토크 (stage8 torque_est, 자립 구간 t≤{a['standing_s']:.1f}s 만"
              + (f", status 스테일 {a['tau_stale']}틱 제외" if a.get("tau_stale") else "")
              + ") — 시뮬 명목 판정 표와 같은 축")
        print(f"   관절                 |τ|피크  RMS   한계  포화%(≥{TAU_SAT_FRAC*100:.0f}%)  샘플")
        worst = None
        for name, peak, rms, sat, n in a["tau"]:
            k = JOINT_ORDER.index(name)
            if peak is None:
                print(f"   {name:<20} (기록 없음)"); continue
            print(f"   {name:<20} {peak:6.2f} {rms:6.2f} {TAU_LIMIT_NM[k]:5.0f}   {sat:6.1f}%     {n}"
                  + ("   (AK45 스케일 미확정 — 24Nm 대조 보류)" if name in ANKLE_R_JOINTS else ""))
            if worst is None or peak / TAU_LIMIT_NM[k] > worst[1]:
                worst = (name, peak / TAU_LIMIT_NM[k])
        if worst:
            print(f"   → 피크/한계 최대 {worst[1]*100:.0f}% ({worst[0]}). 시뮬 명목(v15~v17) 46~64% 와 대조.")
        print()
    elif a["rows"]:
        print("■ 실물 토크: 블랙박스에 tau_est_nm 없음 (0905 이전 브리지) — 토크 대조 불가")
        print()
    cl = a.get("clamp")
    if cl is None:
        print("■ stage8 클램프/거부: 블랙박스에 없음 (0906 이전 브리지)")
    else:
        per = ", ".join(f"{n.replace('_joint', '')} +{v}" for n, v in sorted(cl["per_stand"].items(), key=lambda x: -x[1])) or "없음"
        print(f"■ stage8 soft 클램프: 자립 구간 +{cl['stand']}회 ({per}) / 전 구간 +{cl['all']}회"
              + (f", 첫 클램프 {cl['first_t']:.2f}s" if cl["first_t"] is not None else "")
              + (f" · 거부 사유 {cl['rejects']}" if cl["rejects"] else " · 거부 없음"))
        if cl["stand"] > 0:
            print("   ⚠ 자립 중 클램프 = 정책 목표가 stage8 soft 경계 밖 — 시뮬 판정에는 없는 개입(감사 B7). 낙상 시각과 겹치면 경계표 재검토.")
    lag = a.get("lag")
    if lag is None:
        print("■ 루프지연 추정: numpy 없음 — 생략")
    elif not lag["joints"]:
        print(f"■ 루프지연 추정: 생략 ({lag.get('why', '표본 부족')}, 표본 {lag['n']}틱)")
    else:
        print(f"■ 루프지연 추정 (targets_deg_joint→qpos, 순수 지연 d + 1차 지연 τ 격자 적합; 블렌딩 완료 후 자립 {lag['n']}틱, 틱 {lag['tick_ms']:.1f}ms)")
        print("   관절                 d[틱]   τ[ms]    R²")
        for name, d, tau, r2, note in lag["joints"]:
            if d is None:
                print(f"   {name:<20}   —       —      —   {note}"); continue
            print(f"   {name:<20} {d:4d}   {tau:5d}   {r2:5.2f}   {note}")
        if lag["median_ticks"] is not None:
            med = lag["median_ticks"]
            band = "학습 대역 안" if LAG_TRAIN_BAND[0] <= med <= LAG_TRAIN_BAND[1] else "⚠ 학습 대역 밖"
            print(f"   → 중앙 d = {med:.0f}틱 ({med * lag['tick_ms']:.0f}ms), 유효 관절 {lag['n_good']}/12 — "
                  f"{band}(v19 {LAG_TRAIN_BAND[0]}~{LAG_TRAIN_BAND[1]}틱, v15~v18 3~6틱), 0803 실측 4~8틱")
        else:
            print("   → 유효 관절 없음 (정지 상태이거나 적합도 미달)")
    print()
    print("■ 정의 주의(감사 C5·C6): 실물 낙상 = 발력<50% 또는 시작 대비 +15° 기울임 / 시뮬 낙상 = 절대 40°·베이스 접촉 —")
    print("   낙상률·초기 3s 추세·E_smth 는 정의가 달라 시뮬 수치와 직접 비교하지 않는다(방향·순서·토크 축만 대조).")
    grade, headline, action = verdict(a)
    print(f"■ 판정 [{grade}급] {headline}")
    print(f"   다음: {action}")
    print("=" * 68)


def compare(a, b):
    print()
    print("=" * 68)
    print("비교 — 이번(좌) vs 기준(우)")
    print("=" * 68)
    rows = [
        ("정책", a["ckpt"], b["ckpt"]),
        ("실효 자립", f"{a['standing_s']:.1f}s", f"{b['standing_s']:.1f}s"),
        ("초기 추세", f"{a['slope']:+.2f}°/s", f"{b['slope']:+.2f}°/s"),
        ("단조증가", f"{a['mono']*100:.0f}%", f"{b['mono']*100:.0f}%"),
        ("종료 기울임", f"{a['fwd_end']:+.1f}°", f"{b['fwd_end']:+.1f}°"),
        ("발력 평균", f"{a['load_mean']:.0f}%", f"{b['load_mean']:.0f}%"),
    ]
    for label, x, y in rows:
        print(f"  {label:<10} {x:>16}   |  {y:>16}")
    d = a["slope"] - b["slope"]
    print()
    print(f"  전방 밀기 변화: {d:+.2f}°/s "
          + ("(개선 — 전방 누적이 줄었습니다)" if d < -1.0
             else "(악화 — 전방 누적이 늘었습니다)" if d > 1.0
             else "(유의차 없음)"))
    print("=" * 68)


def main():
    p = argparse.ArgumentParser(description="무개입 낙하 블랙박스 판정기")
    p.add_argument("jsonl", help="rl_bridge 블랙박스 (~/rl_bridge_bb/*.jsonl)")
    p.add_argument("--compare", help="비교 대상 블랙박스 (예: v6b 5차)")
    p.add_argument("--csv", help="틱 시계열을 CSV로 저장")
    p.add_argument("--foot_scale", type=float, default=None,
                   help="foot_n 배율 강제 (기본: 0906 이전 블랙박스는 자동 ×1.469, 이후 1.0)")
    p.add_argument("--selftest", action="store_true", help="루프지연 추정기 합성 검증(블랙박스 불필요)")
    if "--selftest" in sys.argv:
        return lag_selftest()
    args = p.parse_args()

    a = analyze(args.jsonl, args.foot_scale)
    report(a)
    if args.compare:
        compare(a, analyze(args.compare, args.foot_scale))
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("t_s,fwd_deg,lat_deg,tilt_deg,load_pct,blend_alpha\n")
            for r in a["rows"]:
                f.write(f"{r['t']:.3f},{r['fwd']:.3f},{r['lat']:.3f},"
                        f"{r['tilt']:.3f},{r['load']:.2f},{r['alpha']:.3f}\n")
        print(f"CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
