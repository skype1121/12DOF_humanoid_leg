#!/usr/bin/env python3
"""v5q(S8ac4q, 준정적 롤 토크 램프 DR) 사전등록 실험(promotion_criteria 부록 2) 수확·판정 — 결과 파일만 읽음. 2026-09-12 결과 전 작성.
규칙: (1) R05 ≥ 50 % 그리고 R02 ≥ v5 −3 pp (2) A ≥ v5 −1, B′ ≥ v5 −3, C ≥ v5 −1, H69 ≥ v5 −5 pp (3) 발 들기(A) ≤ v5 +0.5 (4) 실물 T5 재생 grav_roll ±5° 출력 Δ ≥ 3.0° (별도 실행값을 --replay_dq 로 입력)."""
import re, sys, pathlib, argparse
LP = pathlib.Path("log_picture")
def read(tag):
    f = LP / f"hang2stand_{tag}.txt"
    if not f.exists(): return None
    t = f.read_text(errors="replace"); m = re.search(r"5s 생존\*\*\s*:\s*(\d+)/(\d+) \(([\d.]+)%\)", t); fl = re.search(r"발 들기 ([\d.]+)회/env", t)
    return dict(pct=float(m.group(3)), n=int(m.group(1)), lift=float(fl.group(1)) if fl else None) if m else None
ap = argparse.ArgumentParser(); ap.add_argument("--replay_dq", type=float, default=None, help="v5q 실물 T5 재생 grav_roll ±5° 최대 |Δ목표| [deg]"); a = ap.parse_args()
rows = [("R05 램프 0.5 Nm/s (r0 0.5)", "v5_R05_256", "v5q_R05"), ("R02 램프 0.2 Nm/s", "v5_R02_256", "v5q_R02"), ("A 명목", "v5_nom_256", "v5q_nom_256"),
        ("B′ DR 푸시 OFF", "v5_dr_nopush", "v5q_dr_nopush"), ("C realistic 명목", "v5_nom_cReal", "v5q_nom_cReal"), ("H69 DR 푸시 OFF 지연 6~9", "v5_dr_nopush_delay69", "v5q_dr_nopush_delay69")]
print("| 조건 | v5 | v5q | Δ(pp) |"); print("|---|---|---|---|")
R = {}
for lab, a5, aq in rows:
    x, y = read(a5), read(aq); R[lab] = (x, y)
    print(f"| {lab} | {x['pct'] if x else '—'} | {y['pct'] if y else '—'} | {(y['pct']-x['pct']) if (x and y) else '—':+} |" if (x and y) else f"| {lab} | {x['pct'] if x else '—'} | {y['pct'] if y else '—'} | — |")
x, y = read("v5_nom_256"), read("v5q_nom_256")
print(f"| 발 들기/env (A) | {x['lift'] if x else '—'} | {y['lift'] if y else '—'} | |")
miss = [lab for lab, (x, y) in R.items() if x is None or y is None]
if miss: print("\n결과 누락:", miss); sys.exit(2)
g = lambda lab, i: R[lab][i]["pct"]
c1 = g("R05 램프 0.5 Nm/s (r0 0.5)", 1) >= 50.0 and g("R02 램프 0.2 Nm/s", 1) >= g("R02 램프 0.2 Nm/s", 0) - 3.0
c2 = (g("A 명목", 1) >= g("A 명목", 0) - 1.0 and g("B′ DR 푸시 OFF", 1) >= g("B′ DR 푸시 OFF", 0) - 3.0 and g("C realistic 명목", 1) >= g("C realistic 명목", 0) - 1.0 and g("H69 DR 푸시 OFF 지연 6~9", 1) >= g("H69 DR 푸시 OFF 지연 6~9", 0) - 5.0)
c3 = (y["lift"] is not None and x["lift"] is not None and y["lift"] <= x["lift"] + 0.5)
c4 = (a.replay_dq is not None and a.replay_dq >= 3.0)
ok = lambda c: "충족" if c else "**미충족**"
print(f"\n- (1) R05 {g('R05 램프 0.5 Nm/s (r0 0.5)',0)} → {g('R05 램프 0.5 Nm/s (r0 0.5)',1)} (≥50) · R02 {g('R02 램프 0.2 Nm/s',0)} → {g('R02 램프 0.2 Nm/s',1)} : {ok(c1)}")
print(f"- (2) A/B′/C/H69 퇴보 한도(−1/−3/−1/−5 pp) : {ok(c2)}")
print(f"- (3) 발 들기 {x['lift']} → {y['lift']} (≤ +0.5) : {ok(c3)}")
print(f"- (4) 실물 T5 재생 grav_roll ±5° Δ목표 {a.replay_dq} (≥3.0°, v5 ≤1.2°) : {ok(c4) if a.replay_dq is not None else '미입력'}")
verdict = c1 and c2 and c3 and c4
print(f"\n### 판정: {'**v5q 승격 후보(실물 적용은 승윤님 결정 뒤 별도 시험)**' if verdict else '**v5 유지 (준정적 보정 미해결)**'}")
sys.exit(0 if verdict else 1)
