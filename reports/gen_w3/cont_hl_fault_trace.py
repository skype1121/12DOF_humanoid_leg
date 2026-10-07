"""1007 헤딩 고리 판별 분리 — E3 추적(eval_w3 --trace_n, 고리 켬 120 s)에서 종전 15° 즉시 정지 vs 판별기(HeadingFaultJudge) 비교(오프라인 재생).

추적의 요·투영중력·자이로(참 값, 시뮬 IMU 이상 없음)로 쿼터니언을 다시 만들고, 배포와 같은 W3HeadingLoop(같은 틱 요, 오차 상한만 30° 로 넓혀 관찰)을 돌린다.
- 종전: 오차 > 15° 첫 시각 = 배포였으면 걷기 정지(오정지 — 시뮬에는 IMU 이상이 없으므로 전부 오판).
- 판별: IMU 정합·15° 5 s 연속·반대 회전 판정 시각.
- 15° 를 넘은 env 가 그 뒤(시뮬에서는 멈추지 않고 계속 걸음) 넘어졌는지 = '그 오정지가 막아 준 낙상' 이 있었는지.
(eval 의 고리는 한 틱 전 요 — 배포 재현과 1 틱 차, parity §4.5 에서 영향 없음 확인.)
사용: python3 reports/gen_w3/cont_hl_fault_trace.py tag ...
"""
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import w3_contingency as C  # noqa: E402
from rl_walking.deploy import w3_deploy as W  # noqa: E402


def quat_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(roll / 2), math.sin(roll / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def ana(tag, t_skip=2.0):
    z = np.load(os.path.join(REPO, "log_picture", f"w3trace_{tag}.npz"), allow_pickle=True)
    dt = float(z["dt"]); T, N = z["yaw"].shape
    N = min(N, int(os.environ.get("HF_MAX_ENV", "32")))          # 파이썬 틱 재생이 느려(64 env × 6100 틱 > 20 분) 앞 N env 만
    tf = z["t_fall"]
    rows = []
    for e in range(N):
        J = C.HeadingFaultJudge(C.HeadingFaultConfig())
        hl = W.W3HeadingLoop(W.W3HeadingLoopConfig(err_max_deg=30.0, timeout_s=1e9 if False else 300.0))
        old = new = None; emax = 0.0; mis = 0.0
        kend = T if not np.isfinite(tf[e]) else int(tf[e] / dt) - 1
        for k in range(kend):
            g = z["grav"][k, e]
            pitch = math.asin(max(-1.0, min(1.0, float(g[0])))); roll = math.atan2(-float(g[1]), -float(g[2]))
            q = quat_zyx(float(z["yaw"][k, e]), pitch, roll)
            walk = bool(z["want"][k, e] > 0.5)
            err = wz = None
            if walk:
                wz, _f = hl.update(float(z["yaw"][k, e]), 0.0)
                err = hl.err
                if k * dt >= t_skip and err is not None:
                    emax = max(emax, abs(math.degrees(err)))
                    if old is None and abs(err) > math.radians(15.0):
                        old = round(k * dt, 2)
            r = J.update(q, z["gyro"][k, e], err, wz)
            if J.mis is not None:
                mis = max(mis, abs(math.degrees(J.mis)))
            if r is not None and new is None:
                new = (round(k * dt, 2), r[0])
        rows.append({"env": e, "old": old, "new": new, "err_max": round(emax, 1), "mis_max": round(mis, 2),
                     "fell": bool(np.isfinite(tf[e])), "t_fall": float(tf[e]) if np.isfinite(tf[e]) else None})
    n_old = sum(r["old"] is not None for r in rows); n_new = sum(r["new"] is not None for r in rows)
    fell_after_old = sum(1 for r in rows if r["old"] is not None and r["fell"] and r["t_fall"] > r["old"])
    fell_no_old = sum(1 for r in rows if r["old"] is None and r["fell"])
    kinds = {}
    for r in rows:
        if r["new"] is not None:
            kinds[r["new"][1]] = kinds.get(r["new"][1], 0) + 1
    s = {"tag": tag, "n": N, "old_15deg_stops": n_old, "new_fires": n_new, "new_kinds": kinds, "fell_after_old_stop": fell_after_old,
         "fell_without_old_stop": fell_no_old, "err_max_p50_p90": [float(np.percentile([r["err_max"] for r in rows], p)) for p in (50, 90)],
         "mis_max_max": max(r["mis_max"] for r in rows)}
    return s, rows


def main():
    out = {}
    for t in sys.argv[1:]:
        s, rows = ana(t)
        print(json.dumps(s, ensure_ascii=False))
        out[t] = {"summary": s, "rows": rows}
    json.dump(out, open(os.path.join(REPO, "log_picture", "cont_hl_fault_trace.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
