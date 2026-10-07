"""시험 4 회복 체류 19 s 중 '왼발 접촉 0' 구간 진단(추정): (1) 콘솔 타임라인의 패드 원값 (2) 엔코더 FK 로 본 좌우 발바닥 높이 차·발 기울기.
패드 값은 접촉 여부 진단용 원값으로만 본다(하중 해석 금지). FK 는 엔코더가 못 보는 무릎 처짐·발목 유격을 포함하지 못한다(추정)."""
import sys, pickle, os, json, datetime
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
from rl_walking.deploy.leg_odometry import fk_sole, LEG_CHAIN
fn = "rl_bridge_20261004_223349.jsonl"
D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb")); R = L.replay(D, "rv3"); t = D["t"]
idx = {s: [L.JO.index(j) for j, _, _ in LEG_CHAIN[s]] for s in ("left", "right")}
pL, RL = fk_sole(D["q"][:, idx["left"]], "left"); pR, RR = fk_sole(D["q"][:, idx["right"]], "right")
g = D["g"] / np.linalg.norm(D["g"], axis=1, keepdims=True)          # 몸통 프레임 '아래' 단위벡터(장착 편향 보정)
downL = (pL * g).sum(1); downR = (pR * g).sum(1)                    # 몸통 원점에서 발바닥점까지 '아래' 거리 [m]
dh = (downR - downL) * 100.0                                         # 왼발바닥이 오른발바닥보다 높은 정도 [cm] (+ = 왼발이 더 높음)
# 발바닥 법선(발 링크 +Y)이 연직(−g)과 이루는 각
nL = RL[:, :, 1]; nR = RR[:, :, 1]
tiltL = np.degrees(np.arccos(np.clip(np.abs((nL * g).sum(1)), -1, 1))); tiltR = np.degrees(np.arccos(np.clip(np.abs((nR * g).sum(1)), -1, 1)))
def seg(a, b): return (t >= a) & (t < b)
print("# 엔코더 FK(추정): 왼발바닥 − 오른발바닥 높이 차 dh[cm](+ = 왼발 높음), 발바닥 기울기[°], 접촉 비트 평균")
for nm, (a, b) in (("서기 20–40 s(양발)", (20, 40)), ("서기 40–58 s(양발)", (40, 58)), ("회복 스텝 61.8–64.8 s", (61.8, 64.8)), ("정지·왼접촉0 65.0–72.6 s", (65.0, 72.6)),
                   ("72.6–74.4 s", (72.6, 74.4)), ("재외란 74.6–78.2 s", (74.6, 78.2)), ("정착 78.7–80.8 s", (78.7, 80.8)), ("서기 81–85 s(양발)", (81, 85))):
    m = seg(a, b)
    print(f"  {nm:<26} dh 중앙 {np.median(dh[m]):+5.2f} (p5 {np.percentile(dh[m],5):+5.2f} · p95 {np.percentile(dh[m],95):+5.2f}) · 발바닥 기울기 왼 {np.median(tiltL[m]):4.1f}° 오른 {np.median(tiltR[m]):4.1f}° · "
          f"접촉 왼 {D['c'][m,0].mean():.2f} 오른 {D['c'][m,1].mean():.2f} · |ω_xy| 중앙 {np.median(D['wxy'][m]):.3f} · 왼 raw_sum 중앙 {np.nanmedian(D['raw_sum'][m,0]):.0f} 오른 {np.nanmedian(D['raw_sum'][m,1]):.0f}")
print("\n# 좌우 관절각 [deg] (JOINT_ORDER: 좌우 교대) — 서기 40–58 s 중앙 vs 65.0–72.6 s 중앙 vs 81–85 s 중앙")
qa = np.degrees(np.median(D["q"][seg(40, 58)], axis=0)); qb = np.degrees(np.median(D["q"][seg(65.0, 72.6)], axis=0)); qc = np.degrees(np.median(D["q"][seg(81, 85)], axis=0))
ta = np.median(D["tgt"][seg(40, 58)], axis=0); tb = np.median(D["tgt"][seg(65.0, 72.6)], axis=0)
for k, j in enumerate(L.JO):
    print(f"  {j:<20} 서기 {qa[k]:+7.2f} → 정지구간 {qb[k]:+7.2f} (Δ {qb[k]-qa[k]:+6.2f}) → 복귀 뒤 {qc[k]:+7.2f} | 명령목표 서기 {ta[k]:+7.2f} → 정지구간 {tb[k]:+7.2f} (Δ {tb[k]-ta[k]:+6.2f})")
print("\n# 추정 토크 tau_est_nm [N·m](stage8 추정 원값, JOINT_ORDER) — 서기 40–58 s 중앙 vs 65.0–72.6 s 중앙")
tqa = np.nanmedian(D["tau"][seg(40, 58)], axis=0); tqb = np.nanmedian(D["tau"][seg(65.0, 72.6)], axis=0)
for k, j in enumerate(L.JO):
    print(f"  {j:<20} {tqa[k]:+6.2f} → {tqb[k]:+6.2f}")
# 타임라인 패드 원값
tl = L.REPO + "/reports/real_1004/session_20261004_221741_eb7694/timeline.jsonl"
T0 = D["ts"][0]; rows = []
with open(tl, "rb") as f:
    for ln in f:
        try: d = json.loads(ln)
        except ValueError: continue
        if d.get("kind") != "s": continue
        tt = d["ts"] - T0
        if 55 <= tt <= 86: rows.append((tt, d))
print(f"\n# 콘솔 타임라인 's' 표본(시험 4, 블랙박스 시각 기준) {len(rows)} 개 — foot_pads_counts[왼 4, 오른 4]·foot_pads_on·sensor_contact·contact·foot_dead_mask·fresh")
print("  키:", sorted(rows[0][1].keys()))
last = None
for tt, d in rows:
    key = (tuple(d.get("foot_pads_on") or []), tuple(d.get("sensor_contact") or []), tuple(d.get("contact") or []) if d.get("contact") else None)
    if key != last or int(tt * 1) % 2 == 0 and abs(tt - round(tt)) < 0.03:
        print(f"  t={tt:6.2f} pads_counts={d.get('foot_pads_counts')} pads_on={d.get('foot_pads_on')} sensor_contact={d.get('sensor_contact')} contact={d.get('contact')} dead={d.get('foot_dead_mask')} fresh={d.get('fresh')} raw_sum={d.get('foot_raw_sum')} unverified={d.get('unverified_force_fields')}")
    last = key
