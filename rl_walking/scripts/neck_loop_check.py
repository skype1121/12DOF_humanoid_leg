#!/usr/bin/env python3
"""목 센서 루프 안정성 검사 (0919, 승격 기준 항목): IMU 무반응·관절이 목표를 그대로 따르는 가짜 조건에서 500틱 — 발산/램프 폭주 검출.
  python3 rl_walking/scripts/neck_loop_check.py --exported logs/rsl_rl/biped12_stage4/<run>/exported --pose v5 [--ticks 500]
브리지 CHECKPOINTS 를 파일로 건드리지 않고 런타임에만 임시 등록한다(등록·배포 아님)."""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); sys.path.insert(0, "/home/ryu/humanoid_leg_test1/jetson/scripts")
import rl_bridge_node as B
POSES = {"v5": {"left_hip_f_joint": 0.5282, "right_hip_f_joint": -0.5282, "left_knee_joint": -0.6720, "right_knee_joint": 0.6720, "left_ankle_f_joint": 0.4887, "right_ankle_f_joint": -0.4887},
         "v3": {"left_hip_f_joint": 0.2618, "right_hip_f_joint": -0.2618, "left_knee_joint": -0.5236, "right_knee_joint": 0.5236, "left_ankle_f_joint": 0.3578, "right_ankle_f_joint": -0.3578}}
ap = argparse.ArgumentParser(); ap.add_argument("--exported", required=True); ap.add_argument("--pose", default="v5"); ap.add_argument("--ticks", type=int, default=500)
a = ap.parse_args()
name = "_neck_check"
B.CHECKPOINTS[name] = os.path.abspath(a.exported)
B.CKPT_DEFAULT_POSE[name] = [float(POSES[a.pose].get(j, 0.0)) for j in B.JOINT_ORDER]
B.CKPT_GAIT_FREQ[name] = 1.1
core = B.BridgeCore(name); core.cmd = (0.0, 0.0, 0.0); core.reset()
q = np.array(B.CKPT_DEFAULT_POSE[name], dtype=float); d0 = q.copy(); worst = 0.0; nonfinite = None
prev = q.copy(); ramp = np.zeros(12, dtype=int); ramp_max = 0; SLEW = math.radians(4.0) * 0.9
for i in range(a.ticks):
    out = core.tick((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), q, np.zeros(12), [1.0, 1.0])
    if not np.isfinite(out).all():
        nonfinite = i; break
    step = out - prev
    # 같은 방향으로 슬루 한계에 붙어 연속 이동하는 틱 수 (램프 폭주 서명: stand_crouch_v2 는 34틱 만에 비유한)
    same_dir = (np.sign(step) == np.sign(prev - d0 + 1e-12)) | (np.abs(prev - d0) < 1e-6)
    ramp = np.where((np.abs(step) >= SLEW) & same_dir, ramp + 1, 0); ramp_max = max(ramp_max, int(ramp.max()))
    worst = max(worst, float(np.abs(out - d0).max())); prev = out; q = out
# 판정: 비유한 = FAIL. 슬루 한계 연속 램프 ≥ 25틱(=100° 이동) = FAIL(폭주). 그 외 PASS — 최대 편차는 정보(기존 세대도 40~80° 과도 있음).
fail = nonfinite is not None or ramp_max >= 25
print(f"neck_loop pose={a.pose} ticks={a.ticks}: nonfinite_at={nonfinite} slew-ramp 연속 최대 {ramp_max}틱 max|target-default|={math.degrees(worst):.1f} deg  → {'FAIL(발산/램프 폭주)' if fail else 'PASS(유한·램프 폭주 없음)'}")
sys.exit(1 if fail else 0)
