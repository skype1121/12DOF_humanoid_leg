"""WOMS 전방향 평가 자동 판정(사전 기준, LOG.md 'WOMS'): python3 omni_judge.py <walk_eval txt> <ckpt>
1000 kill: 전진 유효 cadence < 0.3 Hz 또는 걷기 낙상 > 30 % 또는 명령 반응 전무.
3000 success: vx 기울기(0.05→0.20) ≥ 0.4, 뒤·옆 실제 속도 방향 일치 ≥ 50 %, 제자리 회전 |ωz| ≥ 0.2 & 유효 cadence > 0.5 Hz, 걷기 낙상 ≤ 10 %."""
import sys, re
fp, ck = sys.argv[1], int(sys.argv[2])
rows = {}
for l in open(fp):
    m = re.match(r"- \(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\) → \(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\) · 낙상 ([0-9.]+) % · 유효 cadence ([0-9.]+) Hz", l.strip())
    if m:
        c = tuple(float(x) for x in m.groups()[:3]); rows[c] = tuple(float(x) for x in m.groups()[3:])
if not rows:
    print("판정 불가: 전방향 표 없음"); sys.exit(1)
g = lambda c: rows.get(c)
fw = [g((0.05, 0, 0)), g((0.1, 0, 0)), g((0.2, 0, 0))]
slope = (fw[2][0] - fw[0][0]) / 0.15 if fw[0] and fw[2] else float("nan")
falls = max(r[3] for r in rows.values())
cad_fw = fw[1][4] if fw[1] else 0.0
resp = []
for c, r in rows.items():
    if c[0] < 0: resp.append(("뒤", r[0] < -0.03))
    if c[1] != 0: resp.append(("옆", r[1] * c[1] > 0 and abs(r[1]) > 0.02))
    if c[2] != 0: resp.append(("회전", r[2] * c[2] > 0 and abs(r[2]) > 0.1))
any_resp = any(ok for _, ok in resp) or slope > 0.2
yaw = [r for c, r in rows.items() if c[0] == 0 and c[1] == 0 and c[2] != 0]
yaw_ok = all(abs(r[2]) >= 0.2 and r[4] > 0.5 for r in yaw) if yaw else False
dir_ok = sum(ok for _, ok in resp) / max(1, len(resp))
print(f"[{ck}] vx 기울기 {slope:.2f} · 전진 유효 cadence {cad_fw:.2f} Hz · 최대 낙상 {falls:.1f} % · 방향 반응 {dir_ok*100:.0f} % ({resp}) · 제자리 회전 {'OK' if yaw_ok else 'X'}")
if ck == 1000 and (cad_fw < 0.3 or falls > 30 or not any_resp):
    print("결정: KILL (1000 기준)")
elif slope >= 0.4 and dir_ok >= 0.5 and yaw_ok and falls <= 10:
    print("결정: SUCCESS (최종 기준 충족)")
else:
    print("결정: CONTINUE")
