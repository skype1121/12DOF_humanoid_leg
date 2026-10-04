"""C5 감지기 실물 블랙박스 재생 (1004 검토 6회차) — 계약 검증, 폐루프 성능 증명 아님.
실물 정책 구간(live & armed & blend_alpha≥1 근사: 정책이 명령 중)의 rp_deg(장착 편향 보정 롤·피치)·gyro·contact 를 SwitchDetector 에 넣어
켜짐 횟수·회복 모드 체류 비율·기울기 분포를 본다. 사용: replay_detector.py <jsonl...>"""
import sys, json, math, os
import numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/rl_walking/deploy")
from c5_switch import SwitchDetector

for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f) if l.strip()]
    R = [r for r in R if r.get("live") and r.get("armed") and r.get("rp_deg") is not None]
    if not R:
        print(os.path.basename(f), "정책 구간 없음"); continue
    d = SwitchDetector(); modes = []; tilts = []; ws = []; on_t = []
    t0 = R[0]["timestamp"]
    for r in R:
        rl, pt = [math.radians(x) for x in r["rp_deg"][:2]]
        g = np.array([math.sin(pt), -math.sin(rl) * math.cos(pt), -math.cos(rl) * math.cos(pt)])   # 크기 1, tilt = acos(cos r cos p)
        prev = d.mode
        m = d.update(g, r["gyro_rad_s"], r.get("contact", [1, 1]))
        if m == 1 and prev == 0: on_t.append(round(r["timestamp"] - t0, 1))
        modes.append(m); tilts.append(SwitchDetector.tilt_deg(g)); ws.append(float(np.linalg.norm(r["gyro_rad_s"][:2])))
    modes = np.array(modes); tilts = np.array(tilts); ws = np.array(ws)
    dur = R[-1]["timestamp"] - t0
    print(f"{os.path.basename(f)} [{R[0].get('checkpoint')}] 정책 구간 {dur:.0f} s · {len(R)} 기록 · 켜짐 {d.n_on} 회(첫 시각 {on_t[:6]}) · 회복 모드 체류 {100*modes.mean():.1f} % · "
          f"기울기 중앙 {np.median(tilts):.2f}° (p10 {np.percentile(tilts,10):.2f}, p90 {np.percentile(tilts,90):.2f}) · 기울기<2.5° 비율 {100*(tilts<2.5).mean():.0f} % · |ω| p90 {np.percentile(ws,90):.3f} · 롤 중앙 {np.median([r['rp_deg'][0] for r in R]):+.2f}° 피치 중앙 {np.median([r['rp_deg'][1] for r in R]):+.2f}°")
