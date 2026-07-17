"""GaitFSM을 Session 위에서 실행하는 러너 (텔레메트리 기록 + 낙상 감지 + 캡처)."""
import json
import os
import numpy as np

from .sim_session import Session

FPS = 60.0


def run_gait(s: Session, fsm, duration_s, tag="run",
             telem_every=6, capture_every=None, capture_dir=None,
             abort_on_fall=True, extra_stop=None,
             balance=True, bal_kp=0.010, bal_ki=0.03, bal_clamp=0.10,
             lat_kp=0.008, lat_kd=0.004, lat_clamp=0.12,
             yaw_kp=-0.008, yaw_clamp=0.12,
             push=None):
    """fsm.targets(t)를 60Hz로 구동.

    balance=True:
      - 시상면: lean_ap → 양발목 대칭 도르시 보정 (P+I). 감도 ~-168deg/rad.
      - 관상면: lean_lat → 양힙 hip_a 평행사변형 보정 (P+D).
        평행사변형 로킹에선 골반이 수평이어야 하므로 lean_lat 자체가 오차.
        raw + 는 양쪽 다 '골반 왼쪽(+X) 이동' 방향.
    returns: dict(metrics), telemetry rows 는 tmp json 저장.
    """
    from .sim_session import ANAT_SIGN
    n_frames = int(duration_s * FPS)
    rows = []
    caps = []
    fell_at = None
    x0 = None
    bal_i = 0.0
    lean_ap = 0.0
    lean_lat = 0.0
    lat_prev = None
    lat_rate = 0.0
    _yaw_cache = [0.0]
    sgnL = ANAT_SIGN["dorsiflexion"]["left"]
    sgnR = ANAT_SIGN["dorsiflexion"]["right"]
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
            if f % 2 == 0:
                lean_ap, lean_lat = s.lean()
                bal_i += bal_ki * lean_ap * (2.0 / FPS)
                bal_i = max(-bal_clamp, min(bal_clamp, bal_i))
                if lat_prev is not None:
                    lat_rate = (lean_lat - lat_prev) / (2.0 / FPS)
                lat_prev = lean_lat
            corr = max(-bal_clamp, min(bal_clamp, bal_kp * lean_ap + bal_i))
            tg["left_ankle_f_joint"] = tg.get("left_ankle_f_joint", 0.0) + sgnL * corr
            tg["right_ankle_f_joint"] = tg.get("right_ankle_f_joint", 0.0) + sgnR * corr
            # 관상면: (측정 - 의도 lat_ref) 오차만 교정. lean>ref -> 골반 오른쪽으로
            ref = fsm.lat_ref(t) if hasattr(fsm, "lat_ref") else 0.0
            lcorr = -(lat_kp * (lean_lat - ref) + lat_kd * lat_rate)
            lcorr = max(-lat_clamp, min(lat_clamp, lcorr))
            tg["left_hip_a_joint"] = tg.get("left_hip_a_joint", 0.0) + lcorr
            tg["right_hip_a_joint"] = tg.get("right_hip_a_joint", 0.0) + lcorr
            # 요(yaw): 진행방향 유지 — hip_r 대칭 보정 (부호는 실측 캘리브레이션)
            if yaw_kp:
                if f % 2 == 0:
                    _, _, _, yaw_now = s.pelvis_pose()
                    _yaw_cache[0] = yaw_now
                ycorr = max(-yaw_clamp, min(yaw_clamp, yaw_kp * _yaw_cache[0]))
                tg["left_hip_r_joint"] = tg.get("left_hip_r_joint", 0.0) + ycorr
                tg["right_hip_r_joint"] = tg.get("right_hip_r_joint", 0.0) + ycorr
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
    # 스텝 카운트: 발 z 상승 에지 (스윙 피크 0.065+ vs 롤링 노이즈 ≤0.064 실측 분리)
    steps = {"left": 0, "right": 0}
    up = {"left": False, "right": False}
    for r in rows:
        for side in ("left", "right"):
            z = r["foot_z"][side]
            if not up[side] and z > 0.064:
                steps[side] += 1
                up[side] = True
            elif up[side] and z < 0.056:
                up[side] = False
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
    outdir = "/home/ryu/.claude/jobs/88d43fa3/tmp"
    with open(os.path.join(outdir, f"telem_{tag}.json"), "w") as fh:
        json.dump({"metrics": metrics, "rows": rows}, fh)
    return metrics
