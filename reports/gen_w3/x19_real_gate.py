#!/usr/bin/env python3
"""X19 보조: 실물 브리지 블랙박스의 접촉 비트·자이로·자세를 W3 감독기의 켜기 선행조건에 넣어 본다(읽기 전용, 열린 고리).

몸이 조용한 틱(|ω_xy| < 0.25 rad/s 가 1.0 s 연속 ∧ 기울기 ≤ 7°)만 골라, 그 가운데 접촉 조건(양발 접촉 N 틱 연속)도 성립하는 비율을 낸다.
= "몸은 걷기를 켜도 되는 상태인데 접촉 비트 때문에 거부되는" 비율. 밀기·회복 스텝·줄 회수 중의 발 들림은 조용한 틱에서 빠진다.
기울기는 블랙박스의 rp_deg(롤·피치 합성)로 근사한다(감독기는 장착 편향 보정 뒤 투영중력). 패드 비트만 쓴다 — 하중·CoP 해석 아님.
"""
import json, sys
import numpy as np
SESS = (("1003 서기(지면 0–245 s)", "reports/real_1003/rl_bridge_20261003_174906.jsonl"), ("1005 21:41", "reports/real_1005/rl_bridge_20261005_214105.jsonl"),
        ("1005 21:57", "reports/real_1005/rl_bridge_20261005_215721.jsonl"), ("1005 22:03", "reports/real_1005/rl_bridge_20261005_220326.jsonl"),
        ("1007 회복", "reports/real_1007/rl_bridge_20261007_200441.jsonl"))
def load(path):
    C, W, TL = [], [], []
    for ln in open(path, encoding="utf-8", errors="replace"):
        try: d = json.loads(ln)
        except Exception: continue
        c, g, rp = d.get("contact"), d.get("gyro_rad_s"), d.get("rp_deg")
        if c is None or g is None or rp is None: continue
        C.append([float(c[0]) > 0.5, float(c[1]) > 0.5]); W.append(float(np.hypot(g[0], g[1]))); TL.append(float(np.hypot(rp[0], rp[1])))
    return np.array(C, bool), np.array(W), np.array(TL)
def runlen(b):
    out = np.zeros(len(b), int); r = 0
    for i, v in enumerate(b):
        r = r + 1 if v else 0; out[i] = r
    return out
print("| 실물 세션 | 틱 | 조용한 틱(자이로·기울기 조건 성립) | 그중 접촉 조건도 성립: 1.0 s | 0.5 s | 0.3 s | 조용한 틱의 왼 / 오른 접촉 |"); print("|---|---|---|---|---|---|---|")
for nm, p in SESS:
    C, W, TL = load(p)
    if nm.startswith("1003"): C, W, TL = C[:245 * 50], W[:245 * 50], TL[:245 * 50]   # 245 s 이후는 줄 회수 중 공중(REAL_1003.md)
    calm = (runlen(W < 0.25) >= 50) & (TL <= 7.0); both = runlen(C.all(1)); n = int(calm.sum())
    if n == 0: print(f"| {nm} | {len(C)} | 0 | — | — | — | — |"); continue
    cells = " | ".join(f"{100*(both[calm] >= k).mean():.1f} %" for k in (50, 25, 15))
    print(f"| {nm} | {len(C)} | {n} ({100*n/len(C):.0f} %) | {cells} | {100*C[calm,0].mean():.1f} % / {100*C[calm,1].mean():.1f} % |")
