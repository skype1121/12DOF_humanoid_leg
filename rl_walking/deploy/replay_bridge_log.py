#!/usr/bin/env python3
"""실물 브리지 블랙박스 **로그 재생** 계약 검증 (2026-09-11).

기록된 원시 입력(gyro·quat·qpos·qvel·contact·cmd)을 배포 러너 `PolicyRunnerStage4` 에 순서대로
넣어, 러너가 조립한 관측(320)과 정책 raw 출력이 블랙박스에 기록된 `obs`·`raw_action` 과
일치하는지 본다. 이것은 **출력·계약 검증**(같은 입력 → 같은 관측·같은 출력)이지 폐루프 제어
성능 증명이 아니다(CLAUDE.md 규칙 6).

사용:
  PYTHONPATH=. python3 rl_walking/deploy/replay_bridge_log.py reports/real_0910_A1/trial1/rl_bridge_20260910_154841.jsonl \
      --model models/stand_hang_v5 --gait_freq 1.1
주의: 블랙박스는 obs/raw_action 을 소수 4자리로 반올림해 기록한다 → 허용 오차 5e-4 + 이력 누적을
감안해 |Δ| 통계로 보고한다. 중간 정지·재명령이 있는 시도(정지 이벤트 2개 이상)는 재명령 시점의
러너 reset 을 재현할 수 없으므로 첫 연속 구간만 검증한다.
"""
import argparse, json, math, sys
import numpy as np

JO = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
      "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
      "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def projected_gravity(q):
    w, x, y, z = q
    return (-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y)))


def load(path):
    ticks, stops = [], []
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("event") == "stop_all": stops.append(d)
        elif d.get("live") and d.get("obs") is not None and "qpos_rad" in d: ticks.append(d)
    return ticks, stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bb"); ap.add_argument("--model", default="models/stand_hang_v5")
    ap.add_argument("--gait_freq", type=float, default=1.1, help="v5=1.1 (CKPT_GAIT_FREQ), 그 외 1.4")
    ap.add_argument("--backend", default="auto"); ap.add_argument("--max_ticks", type=int, default=0)
    ap.add_argument("--tol_obs", type=float, default=2e-3); ap.add_argument("--tol_act", type=float, default=2e-3)
    # 0912 출력 민감도 검사(낙상 방지 증명 아님): 기록 입력을 바꿔 정책 출력이 얼마나 달라지는지
    ap.add_argument("--contact_hold", action="store_true", help="접촉 입력을 항상 (1,1) 로 유지")
    ap.add_argument("--gyro_scale", type=float, default=1.0, help="자이로 입력 배율")
    ap.add_argument("--grav_roll_deg", type=float, default=0.0, help="투영중력에 롤 오프셋[deg] 추가(IMU 장착 편향 가정)")
    ap.add_argument("--qvel_scale", type=float, default=1.0, help="관절속도 입력 배율(실물 관절속도 관측이 시뮬보다 작을 때의 출력 민감도)")
    ap.add_argument("--window", type=str, default="", help="민감도 집계 구간 't0,t1' [s, 첫 틱 기준]")
    a = ap.parse_args()
    from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4
    ticks, stops = load(a.bb)
    if not ticks:
        print("obs 가 기록된 정책 틱 없음"); return 2
    # 첫 연속 구간: 첫 정지 이벤트 이전까지 (재명령 시점의 reset 은 재현 불가)
    if stops:
        t_stop = stops[0]["timestamp"]; ticks = [t for t in ticks if t["timestamp"] <= t_stop]
    if a.max_ticks: ticks = ticks[:a.max_ticks]
    runner = PolicyRunnerStage4(a.model, backend=a.backend, gait_freq=a.gait_freq)
    runner.reset()
    n = len(ticks); dobs = np.zeros(n); dact = np.zeros(n); worst_idx = np.zeros(n, dtype=int)
    dim = None
    perturbed = a.contact_hold or a.gyro_scale != 1.0 or a.grav_roll_deg != 0.0 or a.qvel_scale != 1.0
    act_rec = np.zeros((n, 12)); act_new = np.zeros((n, 12)); tt = np.zeros(n)
    for i, t in enumerate(ticks):
        grav = projected_gravity(t["quat_wxyz"])
        if a.grav_roll_deg:
            # base 프레임 x 축(전방) 둘레로 grav 를 회전 = 롤 편향 주입
            th = np.radians(a.grav_roll_deg); gy, gz = grav[1], grav[2]
            grav = (grav[0], gy * np.cos(th) - gz * np.sin(th), gy * np.sin(th) + gz * np.cos(th))
        gyro = [v * a.gyro_scale for v in t["gyro_rad_s"]]
        contact = [1.0, 1.0] if a.contact_hold else t["contact"]
        cmd = tuple(t.get("cmd") or (0.0, 0.0, 0.0))
        qvel = [v * a.qvel_scale for v in t["qvel_rad_s"]]
        runner.step(gyro, grav, cmd, t["qpos_rad"], qvel, contact)
        act_rec[i] = np.asarray(t["raw_action"], dtype=np.float64); act_new[i] = np.asarray(runner.last_action, dtype=np.float64); tt[i] = t["timestamp"] - ticks[0]["timestamp"]
        rec_obs = np.asarray(t["obs"], dtype=np.float64); my_obs = np.asarray(runner.last_obs, dtype=np.float64)
        if dim is None:
            dim = my_obs.shape[0]
            if rec_obs.shape[0] != dim:
                print(f"관측 차원 불일치: 기록 {rec_obs.shape[0]} vs 러너 {dim}"); return 3
        d = np.abs(my_obs - rec_obs); dobs[i] = d.max(); worst_idx[i] = int(d.argmax())
        dact[i] = np.abs(np.asarray(runner.last_action, dtype=np.float64) - np.asarray(t["raw_action"], dtype=np.float64)).max()
    def pct(x, q): return float(np.percentile(x, q))
    print(f"# 로그 재생 계약 검증: {a.bb.split('/')[-1]}  (러너 {a.model}, f={a.gait_freq} Hz, obs {dim})")
    print(f"틱 {n}개 (첫 연속 구간), 기록 반올림 4자리 → 허용 obs {a.tol_obs}, act {a.tol_act}")
    print(f"|Δobs| max: median {np.median(dobs):.2e}  p95 {pct(dobs,95):.2e}  max {dobs.max():.2e}  → 초과 틱 {(dobs > a.tol_obs).sum()}/{n}")
    print(f"|Δraw_action| max: median {np.median(dact):.2e}  p95 {pct(dact,95):.2e}  max {dact.max():.2e}  → 초과 틱 {(dact > a.tol_act).sum()}/{n}")
    # 어느 관측 인덱스가 자주 최악인가 (항 경계와 대조: feet_contact 235:245, gait_clock 225:235, raw+EMA 245:320)
    from collections import Counter
    bad = [worst_idx[i] for i in range(n) if dobs[i] > a.tol_obs]
    if bad:
        c = Counter(bad).most_common(6)
        print("초과 틱의 최악 인덱스(상위): " + ", ".join(f"{k}(x{v})" for k, v in c))
    if perturbed:
        sel = np.ones(n, dtype=bool)
        if a.window:
            w0, w1 = (float(x) for x in a.window.split(",")); sel = (tt >= w0) & (tt <= w1)
        d = (act_new - act_rec) * 0.25 * 57.29578   # 목표각 환산 [deg]
        lat = [2, 3, 4, 5, 10, 11]
        print(f"# 출력 민감도(계약 검증 아님): contact_hold={a.contact_hold} gyro_scale={a.gyro_scale} qvel_scale={a.qvel_scale} grav_roll_deg={a.grav_roll_deg} 구간 {a.window or '전체'} ({int(sel.sum())} 틱)")
        print(f"  |Δ목표| deg: 전체 관절 최대 {np.abs(d[sel]).max():.2f}, 중앙 {np.median(np.abs(d[sel])):.3f} | 좌우축(hip_a/hip_r/ankle_r) 최대 {np.abs(d[sel][:, lat]).max():.2f}")
        print("  관절별 최대 |Δ목표| deg: " + " ".join(f"{np.abs(d[sel][:, k]).max():.2f}" for k in range(12)))
        return 0
    ok = (dobs > a.tol_obs).sum() == 0 and (dact > a.tol_act).sum() == 0
    print("판정: " + ("PASS — 러너 관측 조립·정책 출력이 기록과 일치(계약 검증; 폐루프 성능 아님)" if ok else
                     "FAIL/편차 — 항 경계별 원인 확인 필요(이력 백필·EMA 초기화·클록 위상 후보)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
