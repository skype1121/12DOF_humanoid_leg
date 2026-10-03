"""NEXT 2 기준 동작 생성(1004): ref_steps_*.npz 의 '뒤 유효 스텝'(표준형 = 왼발 스윙)에서 **공통 primitive 만** 뽑아 반주기 23틱 기준을 만든다.
시간: 하중 전환 4틱(이륙 전 k−4..k−1) · 스윙 8틱(이륙..착지를 8틱으로 시간 정규화) · 하중 수용 7틱(착지..+6) · 정착 4틱(+7..+10) — STEP-B0 일정과 같다.
가져오는 것: 접촉(스윙발 0/1), 스윙발 골반 기준 높이(z)·횡(y) 궤적, 좌우 방향 관절(hip_a·hip_r·ankle_r 양쪽)과 양 무릎의 중앙 궤적, 하중(fz) 중앙.
가져오지 않는 것: 뒤 스텝의 앞뒤 방향(hip_f·ankle_f, 발 x) — 뒤로 디딘 방향이라 앞 스텝에 쓰지 않는다(정정 1004: 앞 스텝 시연 0건).
앞뒤 방향 대신: 스윙발 x 목표 = 이륙 시 x + Δ·smoothstep(스윙 진행), Δ = +0.05 m (명령 스텝 3–8 cm 중앙). hip_f·ankle_f 기준 = 이륙 전(k−4) 값 고정(가중 낮음).
출력: ref_step_primitive.npz — q_ref[23,12](rad, env 관절 순서, 왼발 스윙 표준형), w_q[12](관절 가중: 좌우 방향·무릎 1, 앞뒤 0.3), foot_ref[23,3](스윙발 골반 기준 [x,y,z] m),
contact_ref[23,2]([왼 스윙발, 오른 지지발] 0/1), fz_ref[23,2], n_src."""
import numpy as np, sys
D = "/home/ryu/humanoid_leg_test1/reports/gen_step"
z = np.load(f"{D}/{sys.argv[1] if len(sys.argv) > 1 else 'ref_steps_g22r2_s1'}.npz")
m = (z["grp"] == 0) & (z["dfwd"] < -0.02)
idx = np.where(m)[0]; PRE = int(z["pre"]); W = z["q"].shape[1]
def seg(i):
    td = int(z["i_td"][i]); k = PRE
    if td + 10 >= W or td - k < 2: return None
    sw = np.linspace(k, td, 8, endpoint=False)
    t = np.concatenate([np.arange(k - 4, k), sw, np.arange(td, td + 7), np.arange(td + 7, td + 11)])
    lo = np.floor(t).astype(int); fr = (t - lo)[:, None]
    lerp = lambda X: X[lo] * (1 - fr.reshape(-1, *([1] * (X.ndim - 1)))) + X[np.minimum(lo + 1, W - 1)] * fr.reshape(-1, *([1] * (X.ndim - 1)))
    return lerp(z["q"][i]), lerp(z["frel"][i][:, 0, :]), lerp(z["fz"][i].astype(float)), td
Q, Fp, Fz = [], [], []
for i in idx:
    r = seg(i)
    if r is None: continue
    Q.append(r[0]); Fp.append(r[1]); Fz.append(r[2])
Q, Fp, Fz = np.array(Q), np.array(Fp), np.array(Fz)
q_ref = np.median(Q, 0); foot = np.median(Fp, 0); fz_ref = np.median(Fz, 0)
# 앞뒤 관절(hip_f idx 0/1, ankle_f idx 8/9) = 이륙 전 값 고정
for j in (0, 1, 8, 9):
    q_ref[:, j] = q_ref[0, j]
w_q = np.ones(12) * 1.0; w_q[[0, 1, 8, 9]] = 0.3
# 스윙발 x: 이륙(인덱스 4) x + Δ·smoothstep
x0 = foot[4, 0]; s = np.zeros(23); prog = (np.arange(8) + 1) / 8.0; s[4:12] = prog * prog * (3 - 2 * prog); s[12:] = 1.0
foot[:, 0] = foot[0, 0]; foot[4:, 0] = x0 + 0.05 * s[4:]
contact = np.ones((23, 2)); contact[4:12, 0] = 0.0
np.savez(f"{D}/ref_step_primitive.npz", q_ref=q_ref.astype(np.float32), w_q=w_q.astype(np.float32), foot_ref=foot.astype(np.float32),
         contact_ref=contact.astype(np.float32), fz_ref=fz_ref.astype(np.float32), n_src=np.array(len(Q)), default_q=z["default_q"])
dq = np.rad2deg(q_ref - z["default_q"][None])
print(f"원자료 뒤 유효 스텝 {len(Q)}건 → 23틱 기준")
print("스윙발 높이(골반 기준 z, cm, 0=이륙 전):", np.round(100 * (foot[:, 2] - foot[0, 2]), 1).tolist())
print("스윙발 횡(y, cm):", np.round(100 * foot[:, 1], 1).tolist())
print("하중 [스윙, 지지] (mg):", [f"{a:.2f}/{b:.2f}" for a, b in fz_ref])
print("관절 기본자세 대비(deg) 최대 변화 — 왼무릎", round(float(np.abs(dq[:, 6]).max()), 1), "왼 hip_a", round(float(np.abs(dq[:, 2]).max()), 1),
      "오른 hip_a", round(float(np.abs(dq[:, 3]).max()), 1), "왼 ankle_r", round(float(np.abs(dq[:, 10]).max()), 1), "오른 ankle_r", round(float(np.abs(dq[:, 11]).max()), 1))
