"""GaitFSM을 Session 위에서 실행하는 러너 (텔레메트리 기록 + 낙상 감지 + 캡처)."""
import json
import os
import numpy as np

from .sim_session import Session

FPS = 60.0


def count_steps(rows, swing_fn=None):
    """텔레메트리 rows(0.1s 간격)에서 '물리적으로 확인된 스텝' 수 집계.

    스텝 = 롤링 베이스라인(최근 2s 최솟값) 대비 +12mm 상승 후 2.5s 내
    복귀하는 완결 사이클. swing_fn(t)->side 가 주어지면 해당 시점에
    그 발이 스윙 '명령' 중이었을 때만 집계한다 (명령된 스텝의 물리적 확인).
    순수 물리 신호만으론 얕은 스윙(+14mm)과 LEAN 중 하중발 에지 롤링
    (+13.6mm, 완결 사이클 형성)이 분리 불가였음 — 실측."""
    steps = {"left": 0, "right": 0}
    up = {"left": False, "right": False}
    t_up = {"left": 0.0, "right": 0.0}
    z_base = {"left": 0.0, "right": 0.0}   # 이륙 시점 베이스라인 고정
    ok_up = {"left": False, "right": False}
    hist = {"left": [], "right": []}
    for r in rows:
        for side in ("left", "right"):
            z = r["foot_z"][side]
            h = hist[side]
            h.append(z)
            if len(h) > 20:      # rows는 0.1s 간격 → 2s 윈도우
                h.pop(0)
            b = min(h)
            if not up[side] and z > b + 0.012:
                up[side] = True
                t_up[side] = r["t"]
                z_base[side] = b
                ok_up[side] = (swing_fn is None) or \
                    (swing_fn(r["t"]) == side or swing_fn(t_up[side] - 0.2) == side)
            elif up[side]:
                # 상승 유지 중 스윙 명령이 걸리면 인정 (선행 리프트 케이스)
                if swing_fn is not None and not ok_up[side] \
                        and swing_fn(r["t"]) == side:
                    ok_up[side] = True
                # 복귀는 '이륙 시점' 베이스라인 기준 — 롤링 베이스라인이
                # 2s 만에 적응해 미복귀 상승을 사이클로 오인하는 것 방지
                if z < z_base[side] + 0.005:
                    up[side] = False
                    if r["t"] - t_up[side] < 2.5 and ok_up[side]:
                        steps[side] += 1
    return steps


def run_gait(s: Session, fsm, duration_s, tag="run",
             telem_every=6, capture_every=None, capture_dir=None,
             abort_on_fall=True, extra_stop=None,
             balance=True, bal_kp=0.010, bal_ki=0.03, bal_clamp=0.10,
             lat_kp=0.008, lat_kd=0.004, lat_clamp=0.12,
             yaw_kp=-0.008, yaw_clamp=0.12,
             push=None):
    """fsm.targets(t)를 60Hz로 구동.

    balance=True: BalanceFeedback(feedback.py) — 시상면(발목 P+I),
      관상면(hip_a P+D, fsm.lat_ref 추종), 요(hip_r P).
    returns: dict(metrics), telemetry rows 는 demo_output/telemetry 저장.
    """
    from .feedback import BalanceFeedback
    n_frames = int(duration_s * FPS)
    rows = []
    caps = []
    fell_at = None
    x0 = None
    fb = BalanceFeedback(s, bal_kp=bal_kp, bal_ki=bal_ki, bal_clamp=bal_clamp,
                         lat_kp=lat_kp, lat_kd=lat_kd, lat_clamp=lat_clamp,
                         yaw_kp=yaw_kp, yaw_clamp=yaw_clamp)
    obs_every = 2  # 폐루프 정책 관측 주기 (프레임)
    for f in range(n_frames):
        t = f / FPS
        # 폐루프 정책 지원: observe()가 있으면 관측 제공 (policy_interface 참조)
        if hasattr(fsm, "observe") and f % obs_every == 0:
            from .policy_interface import build_obs
            fsm.observe(build_obs(s, t))
        tg = fsm.targets(t)
        # 외란 주입: push=(t_start, dur_s, [fx,fy,fz] N) — 골반에 월드 힘
        if push and push[0] <= t < push[0] + push[1]:
            try:
                s.pelvis.apply_forces(np.array([push[2]], dtype=float),
                                      is_global=True)
            except Exception:
                pass
        if balance:
            ref = fsm.lat_ref(t) if hasattr(fsm, "lat_ref") else 0.0
            tg = fb.apply(tg, lat_ref=ref)
        s.cmd(named=tg)
        s.step(1)
        if f % telem_every == 0:
            te = s.telemetry()
            te["t"] = round(t, 3)
            rows.append(te)
            if x0 is None:
                x0 = (te["pelvis_x"], te["pelvis_y"])
            if abort_on_fall and s.fallen():
                fell_at = t
                break
            if extra_stop and extra_stop(te):
                break
        if capture_every and capture_dir and f % capture_every == 0:
            pth = os.path.join(capture_dir, f"{tag}_{f:05d}.png")
            s.capture(pth, wait=0)   # 비동기 — 물리 타이밍 유지
            caps.append(pth)

    te = s.telemetry()
    dx = te["pelvis_x"] - x0[0]
    dy = te["pelvis_y"] - x0[1]
    swing_fn = fsm.swing_side if hasattr(fsm, "swing_side") else None
    steps = count_steps(rows, swing_fn=swing_fn)
    metrics = {
        "steps_left": steps["left"], "steps_right": steps["right"],
        "steps_total": steps["left"] + steps["right"],
        "tag": tag,
        "duration_run": round((f + 1) / FPS, 2),
        "fell_at": fell_at,
        "forward_progress_m": round(-dy, 4),   # 전진 = -Y
        "lateral_drift_m": round(dx, 4),
        "final_pelvis_z": round(te["pelvis_z"], 4),
        "final_lean_ap": round(te["lean_ap"], 2),
        "final_lean_lat": round(te["lean_lat"], 2),
        "final_yaw": round(te["yaw"], 2),
        "max_jvel": round(max(r["jvel_max"] for r in rows), 3) if rows else None,
        "min_pelvis_z": round(min(r["pelvis_z"] for r in rows), 4) if rows else None,
        "max_abs_lean_ap": round(max(abs(r["lean_ap"]) for r in rows), 2) if rows else None,
        "max_abs_lean_lat": round(max(abs(r["lean_lat"]) for r in rows), 2) if rows else None,
        "n_captures": len(caps),
    }
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "demo_output", "telemetry")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"telem_{tag}.json"), "w") as fh:
        json.dump({"metrics": metrics, "rows": rows}, fh)
    return metrics
