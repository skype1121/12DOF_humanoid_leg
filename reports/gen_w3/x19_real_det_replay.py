#!/usr/bin/env python3
"""X19d 보조: 실물 블랙박스의 IMU·접촉을 배포 KE 감지기(kr 구성, c5_switch.SwitchDetector)에 열린 고리로 넣어 본다(읽기 전용).

묻는 것: 왼발 접촉 비트가 0 이면(실제 0 이었던 1005 22:03, 그리고 다른 세션의 무외란 서기 구간에 0 을 강제로 넣은 반사실)
감지기의 기준 자세가 얼고, 서기 자세의 느린 흐름이 켜기 문턱 3.0° 를 넘는가.
열린 고리의 한계: 감지기가 켜진 뒤의 몸 움직임은 재현하지 않는다(첫 켜짐까지만 뜻이 있다). 접촉 비트만 쓴다.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); sys.path.insert(0, "/home/ryu/humanoid_leg_test1/jetson/scripts")
from rl_walking.deploy.c5_switch import SwitchDetector
from rl_walking.deploy.replay_bridge_log import projected_gravity
KW = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
          early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))      # rl_bridge_node C5_DETECTOR_KW + C5_EXTRA[kr].det
def load(path, t0, t1):
    G, W, C = [], [], []
    for ln in open(path, encoding="utf-8", errors="replace"):
        try: d = json.loads(ln)
        except Exception: continue
        if d.get("contact") is None or d.get("quat_wxyz") is None: continue
        G.append(np.asarray(projected_gravity(np.asarray(d["quat_wxyz"], dtype=np.float64)), dtype=np.float64).reshape(3)); W.append(d["gyro_rad_s"][:3]); C.append(d["contact"][:2])
    a, b = int(t0 * 50), int(t1 * 50); return np.array(G)[a:b], np.array(W, float)[a:b], np.array(C, float)[a:b]
def replay(G, W, C, force_left0):
    det = SwitchDetector(**KW); first = None; dev = []; ref_move = 0.0; prev = None
    for i in range(len(G)):
        c = C[i].copy()
        if force_left0: c[0] = 0.0
        m = det.update(G[i], W[i], c)
        if det.ref is not None:
            rp = det.roll_pitch_deg(det.filtered(G[i], W[i])[0]) if False else SwitchDetector.roll_pitch_deg(G[i]); dev.append(float(np.linalg.norm(rp - det.ref)))
            if prev is not None: ref_move += float(np.abs(det.ref - prev).sum())
            prev = det.ref.copy()
        if m and first is None: first = i * 0.02
    return first, (max(dev[75:]) if len(dev) > 75 else float("nan")), ref_move
print("| 실물 구간 | 접촉 비트 | 길이 | 첫 회복 모드 켜짐 | 기준 대비 최대 편차(1.5 s 이후) | 기준 자세 누적 이동 |"); print("|---|---|---|---|---|---|")
for nm, p, t0, t1 in (("1005 22:03 전체(왼발 신호 실제 0)", "reports/real_1005/rl_bridge_20261005_220326.jsonl", 0, 75.1),
                      ("1003 무외란 서기 0–150 s", "reports/real_1003/rl_bridge_20261003_174906.jsonl", 0, 150),
                      ("1005 21:41 0–55 s(접촉 끊김 없는 구간)", "reports/real_1005/rl_bridge_20261005_214105.jsonl", 0, 55),
                      ("1007 조용한 서기 0–18 s", "reports/real_1007/rl_bridge_20261007_200441.jsonl", 0, 18),
                      ("1007 0–30.1 s(18 s 부터 롤이 천천히 기욺 — 원인 미확인)", "reports/real_1007/rl_bridge_20261007_200441.jsonl", 0, 30.1)):
    G, W, C = load(p, t0, t1)
    for lab, f in (("기록 그대로", False), ("왼발 0 강제(반사실)", True)):
        if nm.startswith("1005 22:03") and f: continue
        first, mx, mv = replay(G, W, C, f)
        print(f"| {nm} | {lab} | {len(G)*0.02:.0f} s | {'없음' if first is None else f'{first:.1f} s'} | {mx:.2f}° | {mv:.2f}° |")
