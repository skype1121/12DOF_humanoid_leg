"""1005 krF 감지기 실물 블랙박스 재생 — 실물 센서 잡음에서 오탐·재켜짐·체류(계약 검증, 폐루프 아님).
RV4(종전 C5) 와 krF(필터 α0.33 + 조기 복귀 + 불응기 1.5 s) 를 같은 로그에 재생. 사용: replay_krf_real.py <jsonl...>"""
import sys, json, math, os
import numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/rl_walking/deploy")
from c5_switch import SwitchDetector
BASE = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any")
CFG = {"RV4(C5)": dict(BASE), "krF": dict(BASE, early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33)}
for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f) if l.strip()]
    R = [r for r in R if r.get("live") and r.get("armed") and r.get("rp_deg") is not None]
    if not R:
        print(os.path.basename(f), "정책 구간 없음"); continue
    dur = R[-1]["timestamp"] - R[0]["timestamp"]
    out = []
    for name, kw in CFG.items():
        d = SwitchDetector(**kw); modes = []; ons = []; prev = 0
        for r in R:
            rl, pt = [math.radians(x) for x in r["rp_deg"][:2]]
            g = np.array([math.sin(pt), -math.sin(rl) * math.cos(pt), -math.cos(rl) * math.cos(pt)])
            m = d.update(g, r["gyro_rad_s"], r.get("contact", [1, 1]))
            if m == 1 and prev == 0: ons.append(r["timestamp"] - R[0]["timestamp"])
            prev = m; modes.append(m)
        gaps = np.diff(ons) if len(ons) > 1 else np.array([])
        rapid = int((gaps < 3.0).sum())       # 3 s 안 재켜짐(왕복 후보)
        out.append(f"{name}: 켜짐 {len(ons)} ({60*len(ons)/max(dur,1):.1f}/분) · 3 s 안 재켜짐 {rapid} · 체류 {100*np.mean(modes):.1f} % · 조기복귀 {getattr(d,'n_early',0)}")
    print(f"{os.path.basename(f)} [{R[0].get('checkpoint')}] {dur:.0f} s · " + " | ".join(out))
