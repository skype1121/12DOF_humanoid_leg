"""1007 헤딩 고리 고장 판별 분리(HeadingFaultJudge) 오판 점검 — 오프라인(열린 고리 재생, 폐루프 효과 아님).

(1) 시뮬 녹화 log_picture/w3parity_rec_w3s3d_15800_d3.npz(W3S3d-15800, 3 env × 45 s, 지연 3, 자이로·IMU 지연 DR): 고리 켬 구간(hl_on)에
    배포와 같은 W3HeadingLoop(같은 틱 요)을 돌려 오차를 만들고, 판별기를 정책 관측 자이로(gyro, 지연·잡음)·참 자이로 각각으로.
    종전 규칙(오차 > 15° 즉시)과 새 규칙(IMU 불일치·15° 연속 dev_hold_s·요 속도 반대) 의 정지 시각 비교.
(2) 실물 블랙박스 1001–1004(정책 서기·밀기·C5, 실제 IMU·자이로): IMU 정합 불일치 분포(고리 없음 — A 신호 오판만).
(3) 주입 시험: 녹화에 IMU 요 점프 10°·표류 6 °/s 를 넣어 A 가 잡는지.
사용: python3 reports/gen_w3/cont_hl_fault.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import w3_contingency as C  # noqa: E402
from rl_walking.deploy import w3_deploy as W  # noqa: E402


def run_rec(gyro_key="gyro", inject=None, cfg=None):
    z = np.load(os.path.join(REPO, "log_picture", "w3parity_rec_w3s3d_15800_d3.npz"), allow_pickle=True)
    T, N = z["want"].shape
    cfg = cfg or C.HeadingFaultConfig()
    res = []
    for e in range(N):
        J = C.HeadingFaultJudge(cfg)
        hl = W.W3HeadingLoop(W.W3HeadingLoopConfig(err_max_deg=30.0, timeout_s=300.0))
        old_stop = new = None
        mis_max = 0.0
        dev_max = 0.0
        prev_on = False
        for k in range(T):
            q = [float(v) for v in z["quat"][k, e]]
            if inject is not None:
                q = inject(k, q)
            g = z[gyro_key][k, e]
            on = bool(z["hl_on"][k, e]) and bool(z["want"][k, e])
            err = wz = None
            if on:
                if not prev_on:
                    J.reset_walk()
                _, _, yv = C.euler_from_quat(q)
                wz, f = hl.update(yv, 0.0)
                err = hl.err
                if err is not None:
                    dev_max = max(dev_max, abs(math.degrees(err)))
                    if old_stop is None and abs(err) > math.radians(15.0):
                        old_stop = k * 0.02
            else:
                hl.release()
            prev_on = on
            r = J.update(q, g, err, wz)
            if J.mis is not None:
                mis_max = max(mis_max, abs(math.degrees(J.mis)))
            if r is not None and new is None:
                new = (round(k * 0.02, 2),) + r[:1] + (r[2],)
        res.append({"env": e, "old_15deg_stop_s": old_stop, "new_fire": new, "imu_mis_max_deg": round(mis_max, 2), "hl_err_max_deg": round(dev_max, 1)})
    return res


def run_real():
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl"))):
        if os.path.getsize(p) == 0:
            continue
        J = C.HeadingFaultJudge(C.HeadingFaultConfig())
        mis, fires, n = [], [], 0
        for ln in open(p):
            if '"quat_wxyz"' not in ln or '"gyro_rad_s"' not in ln:
                continue
            d = json.loads(ln)
            r = J.update(d["quat_wxyz"], d["gyro_rad_s"])
            n += 1
            if J.mis is not None:
                mis.append(abs(math.degrees(J.mis)))
            if r is not None:
                g = d["gyro_rad_s"]
                fires.append((round(n * 0.02, 1), r[1][:40], round(math.hypot(g[0], g[1]), 2)))
                J.reset(); J.reset_walk()
        out.append({"file": os.path.basename(p), "ticks": n, "mis_p99": round(float(np.percentile(mis, 99)), 2) if mis else None,
                    "mis_max": round(float(np.max(mis)), 2) if mis else None, "fires": fires[:6], "n_fires": len(fires)})
    return out


def jump(k, q, at=1500, deg=10.0):
    if k < at:
        return q
    w, x, y, z = q
    h = math.radians(deg) / 2
    cw, sw = math.cos(h), math.sin(h)                 # 월드 z 회전 앞곱
    return [cw * w - sw * z, cw * x - sw * y, cw * y + sw * x, cw * z + sw * w]


def drift(k, q, at=1500, dps=6.0):
    return jump(k, q, at=0, deg=0.0) if k < at else jump(k, q, at=0, deg=dps * (k - at) * 0.02)


def main():
    R = {"rec_gyro_obs": run_rec("gyro"), "rec_gyro_true": run_rec("gyro_true"),
         "rec_jump10": run_rec("gyro", inject=jump), "rec_drift6dps": run_rec("gyro", inject=drift), "real": run_real()}
    for k, v in R.items():
        print(k)
        for r in v:
            print("  ", r)
    json.dump(R, open(os.path.join(REPO, "log_picture", "cont_hl_fault.json"), "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
