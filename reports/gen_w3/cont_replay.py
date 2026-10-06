"""1007 대비책 — 실물 블랙박스를 감시기(note 모드)에 재생해 서기 중 판정 입력의 실물 분포와 '켰다면 생겼을' 거부·사건을 센다.
(읽기 전용, 열린 고리 재생 = 계약·오트립 점검이지 폐루프 효과 증명이 아니다.)

재는 것: ① IMU 요 정지 표류율 r[°/min](양발 접촉·|ω_xy| < 0.1 표본, 60 s 창 선형 기울기 — 감시기 D 와 같은 식) 시계열·최대,
② 요 점프(|Δψ − ω_z·dt| > 1°/틱) 수, ③ 서기 접촉 채터로 켜기 거부였을 틱 비율, ④ 피드백 나이·stage8 명령 처리 나이로 걷기 금지였을 틱 비율,
⑤ 켜기 선행조건 기울기·|ω_xy| 와 같은 축의 서기 분포(참고). 정책 = 기록 당시 정책(G12·G21a·C5 서기) — W3 서기와 다르다.
W3 세션 블랙박스(w3 필드 있음)는 실제 걷기 구간을 걷기로 넣고 송출 목표·엔코더로 지연 추정도 돈다(현장 시험 뒤 오트립 계수 = D1 과 같은 쓰임).
사용: python3 reports/gen_w3/cont_replay.py [jsonl ...] → 표 + log_picture/cont_replay.json
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


def quat_grav(q):
    w, x, y, z = q
    return (2 * (x * z - w * y), 2 * (y * z + w * x), -(w * w - x * x - y * y + z * z))


def yaw_q(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def replay(path, cfg=None):
    """cfg: 감시 설정(기본 = note + 지연 판정 8 틱) — 현장 세션 뒤 '켰다면' 계수용으로 같은 함수를 쓴다."""
    from rl_walking.deploy.policy_runner import JOINT_ORDER
    m = C.W3WalkMonitor(cfg or C.W3MonitorConfig(mode="note", lat_max_ticks=8.0), JOINT_ORDER)
    n = n_gate_chat = n_health = 0
    rates, w_all, tilt_all, gap = [], [], [], 0
    t_prev = None
    with open(path) as f:
        for ln in f:
            if '"quat_wxyz"' not in ln or '"targets_deg_joint"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if t_prev is not None and ts is not None and ts - t_prev > 0.1:
                gap += 1                                  # 기록 공백(정책 정지 등) — 감시기 시간은 틱 수라 그대로 진행
            t_prev = ts
            q = d["quat_wxyz"]; g = d["gyro_rad_s"]; c = d.get("contact") or [0, 0]
            w3 = d.get("w3") or {}
            walking = bool(w3.get("want") or w3.get("ge"))         # W3 세션 블랙박스면 실제 걷기 구간, 아니면 서기
            m.tick(walking=walking, contact2=c, gyro=g, grav=quat_grav(q), yaw=yaw_q(q))
            tj, qp = d.get("targets_deg_joint"), d.get("qpos_rad")
            if tj and qp and (d.get("blend_alpha") is None or d["blend_alpha"] >= 1.0):
                from rl_walking.deploy.policy_runner import JOINT_ORDER
                m.track([math.radians(tj[j]) for j in JOINT_ORDER], qp)
            s8 = (d.get("s8_cmd") or {}).get("p50_age_ms_1s")
            m.health(foot_age_s=None, fb_age_s=d.get("status_age_s"), s8_p50_ms=s8)
            n += 1
            ok, why = C.W3WalkMonitor.enable_gate(type("X", (), {"act": True, "cfg": m.cfg, "health_bad": None, "lat_bad": False,
                                                                 "trans_stand": m.trans_stand, "t": m.t, "lat_est": None})())
            n_gate_chat += (not ok)
            n_health += m.health_bad is not None
            if m.n % 250 == 0 and m.yaw_rate_dpm is not None:
                rates.append(m.yaw_rate_dpm)
            w_all.append(math.hypot(g[0], g[1]))
            rpd = d.get("rp_deg")
            if rpd:
                tilt_all.append(math.hypot(rpd[0], rpd[1]))
    ev = {}
    for e in m.events:
        ev[e["kind"]] = ev.get(e["kind"], 0) + 1
    pr = lambda x: [round(float(v), 3) for v in np.percentile(x, [50, 90, 99])] if len(x) else None
    return {"file": os.path.relpath(path, REPO), "ticks": n, "gaps": gap, "yaw_rate_dpm_samples": len(rates),
            "yaw_rate_dpm_p50_abs": round(float(np.median(np.abs(rates))), 3) if rates else None,
            "yaw_rate_dpm_max_abs": round(float(np.max(np.abs(rates))), 3) if rates else None,
            "yaw_rate_dpm_last": round(rates[-1], 3) if rates else None,
            "events": ev, "frac_gate_chatter": round(n_gate_chat / max(n, 1), 4), "frac_health_bad": round(n_health / max(n, 1), 4),
            "w_xy_p50_90_99": pr(w_all), "tilt_deg_p50_90_99": pr(tilt_all)}


def main():
    ps = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl")))
    out = [replay(p) for p in ps if os.path.getsize(p) > 0]
    print("| 파일 | 틱 | 공백 | 요 표류 |r| 중앙/최대 [°/min] (표본) | 마지막 | 사건 | 채터 켜기 거부 틱 | 센서 나이 금지 틱 | |ω_xy| p50/90/99 | 기울기 p50/90/99 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['file'].split('/')[-1]} | {r['ticks']} | {r['gaps']} | {r['yaw_rate_dpm_p50_abs']} / {r['yaw_rate_dpm_max_abs']} ({r['yaw_rate_dpm_samples']}) | "
              f"{r['yaw_rate_dpm_last']} | {r['events']} | {100 * r['frac_gate_chatter']:.2f} % | {100 * r['frac_health_bad']:.2f} % | {r['w_xy_p50_90_99']} | {r['tilt_deg_p50_90_99']} |")
    json.dump(out, open(os.path.join(REPO, "log_picture", "cont_replay.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
