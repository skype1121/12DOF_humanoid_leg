"""실물 서기 무릎 토크 역산 — 브리지 블랙박스의 서기 자세로 정역학 모델 토크를 계산해 모터 토크 추정(tau_est_nm)과 견준다(CPU·numpy·scipy, 로컬 사본만 읽음).

사용: python3 reports/gen_w3/real_knee_torque.py [reports/real_1007/rl_bridge_20261007_200441.jsonl]

물음(TORQUE_BUDGET_1007.md 머리말): 실물 서기 무릎 τ 추정 3.4–4.4 N·m 대 시뮬 7.9–8.2 N·m 의 2 배 차이가 (i) 자세 (ii) 토크 추정 눈금·기어 마찰 (iii) 질량 중 무엇인가.
방법: 블랙박스의 조용한 서기 구간(회복 모드 아님·양발 접촉·|ω_xy| < 0.1)에서 엔코더 관절각·IMU 기울기(장착 편향 보정)·tau_est 의 중앙값을 뽑는다.
  엔코더 각은 물리 관절각이 아니다(무릎 뒤 직렬 처짐·발목 유격·영점 오프셋이 엔코더 뒤에 있다) → 엔코더 각을 그대로 넣으면 발바닥이 지면과 평행하지 않게 나온다(사슬 불일치).
  그래서 "물리 무릎각 = 엔코더 + 처짐 s_k" 를 0…전부까지 훑으며, 나머지 불일치는 힙·발목에 나눠 [발 평평 ∧ 발목 유지 토크 = τ_a] 를 만족하는 자세를 풀고
  그 자세의 양발 서기 무릎 토크(다리마다 골반 절반 + 자기 다리)를 계산한다. τ_a 는 −2·0·+2 N·m 세 값으로 훑는다 — 발목 tau_est 는 작고(|τ| < 1) 부호 규약·유격 마찰(약 2 N·m)이
  불확실해 압력 중심을 ±2.3 cm 안으로만 묶는다. 모델 = torque_phase.Model(rl_walking/assets/biped12_final3.urdf, 17.635 kg). 조용한 서기 창은 3–18 s(그 뒤는 천천히 기우는 구간).
한계: 정역학(가속 없음), 좌우 대칭 가정, tau_est 는 드라이버가 낸 값 그대로(전류 × 토크 상수 × 감속비로 추정 — 기어 효율·마찰 미포함), 패드 값은 쓰지 않는다(CLAUDE.md §6).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torque_phase as TP  # noqa: E402

JI = {n: i for i, n in enumerate(TP.JN)}


def load(path):
    T, M, Q, TAU, RP, B, G, C, TG = [], [], [], [], [], [], [], [], []
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("c5") is None or d.get("qpos_rad") is None:
            continue
        T.append(d["timestamp"]); M.append(int(d["c5"]["mode"])); Q.append(d["qpos_rad"]); TAU.append(d["tau_est_nm"]); RP.append(d["rp_deg"])
        B.append(d.get("imu_mount_bias_deg") or [0.0, 0.0]); G.append(d["gyro_rad_s"]); C.append(d["contact"])
        tg = d.get("targets_deg_joint") or {}; TG.append([tg.get(n, np.nan) for n in TP.JN])
    f = lambda x: np.array(x, float)                                              # noqa: E731
    T = f(T); return dict(t=T - T[0], mode=np.array(M), q=f(Q), tau=f(TAU), rp=f(RP) - f(B), gyro=f(G), cont=f(C), tgt=f(TG))


def solve_posture(model, knee_deg, pitch_deg, roll_deg, tau_ankle):
    """좌우 대칭 서기: 무릎 굽힘 knee_deg, 몸통 피치·롤(측정). 미지수 힙 굴곡 hf·발목 af — 조건: 발 앞뒤 패드 높이 같음 ∧ (골반 절반 + 다리)의 발목 유지 토크 = tau_ankle(측정 평균)."""
    gd = TP.gdir_from_rp(np.array([roll_deg]), np.array([pitch_deg])); up = -gd[0]
    half = model.halved("L")

    def build(x):
        q = np.zeros(12)
        q[JI["left_hip_f_joint"]] = x[0]; q[JI["left_knee_joint"]] = -np.radians(knee_deg); q[JI["left_ankle_f_joint"]] = x[1]
        q[JI["right_hip_f_joint"]] = -x[0]; q[JI["right_knee_joint"]] = np.radians(knee_deg); q[JI["right_ankle_f_joint"]] = -x[1]
        return q

    def res(x):
        q = build(x); R, p, a, pj = model.fk(q[None])
        fr = 0.5 * (p["leftfoot_leftfront_RA30P"][0] + p["leftfoot_rightfront_RA30P"][0]); bk = 0.5 * (p["leftfoot_leftback_RA30P"][0] + p["leftfoot_rightback_RA30P"][0])
        ta = half.hold_torque(q[None], gd, "L")[0][JI["left_ankle_f_joint"]]
        return [float((fr - bk) @ up) * 50.0, ta - tau_ankle]
    s = least_squares(res, x0=[np.radians(0.5 * knee_deg), np.radians(0.5 * knee_deg)])
    q = build(s.x); th = half.hold_torque(q[None], gd, "L")[0]
    return np.degrees(s.x[0]), np.degrees(s.x[1]), th[JI["left_knee_joint"]], th[JI["left_hip_f_joint"]], float(s.cost)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TP.REPO, "reports", "real_1007", "rl_bridge_20261007_200441.jsonl")
    d = load(path); model = TP.Model(os.path.join(TP.REPO, "rl_walking", "assets", "biped12_final3.urdf"))
    w = np.hypot(d["gyro"][:, 0], d["gyro"][:, 1]); both = (d["cont"] > 0.5).all(1)
    m = (d["t"] > 3.0) & (d["t"] < 18.0) & (d["mode"] == 0) & (w < 0.1) & both
    print(f"# real_knee_torque.py — {os.path.basename(path)} · 틱 {len(d['t'])}({d['t'][-1]:.1f} s) · 조용한 서기 틱 {int(m.sum())}({d['t'][m].min():.1f}–{d['t'][m].max():.1f} s)")
    med = lambda x: np.nanmedian(x[m], 0)                                         # noqa: E731
    q = med(d["q"]); tau = med(d["tau"]); rp = med(d["rp"]); tgt = med(d["tgt"])
    iq = np.percentile(d["tau"][m], [10, 90], axis=0)
    print("\n## 1. 조용한 서기의 측정 중앙값")
    print("| 관절 | 명령 ° | 엔코더 ° | tau_est N·m 중앙 [p10, p90] |"); print("|---|---|---|---|")
    for j in range(12):
        print(f"| {TP.SHORT[j]} | {tgt[j]:+.1f} | {np.degrees(q[j]):+.1f} | {tau[j]:+.2f} [{iq[0, j]:+.2f}, {iq[1, j]:+.2f}] |")
    print(f"- IMU(장착 편향 보정 뒤): 롤 {rp[0]:+.2f}° · 피치 {rp[1]:+.2f}°(+ = 왼/앞 기울임)")
    lsb = np.diff(np.unique(np.round(d["tau"][:, 6], 6))); lsb = lsb[lsb > 1e-6].min()
    print(f"- tau_est 눈금 한 칸 {lsb:.5f} N·m(= 50/4095 = {50 / 4095:.5f}: MIT 12 비트·±25 N·m 디코드 그대로) — 드라이버의 전류 기반 추정값이다")
    hfe, kne, afe = 0.5 * (np.degrees(q[0]) - np.degrees(q[1])), 0.5 * (-np.degrees(q[6]) + np.degrees(q[7])), 0.5 * (np.degrees(q[8]) - np.degrees(q[9]))
    kn_est = 0.5 * (abs(tau[6]) + abs(tau[7])); hf_est = 0.5 * (tau[0] - tau[1]); af_est = 0.5 * (tau[8] - tau[9])
    print(f"- 좌우 평균(굽힘 +): 엔코더 힙 {hfe:.1f}° · 무릎 {kne:.1f}° · 발목 {afe:.1f}° → 사슬 합(힙 − 무릎 + 발목) {hfe - kne + afe:+.1f}° 인데 몸통 피치는 {rp[1]:+.1f}°")
    print(f"  → 엔코더 각이 물리 각이라면 발바닥이 지면과 {hfe - kne + afe - rp[1]:.1f}° 어긋난다(서 있는 로봇에서 불가능) = 엔코더 뒤의 처짐·유격·영점 오프셋 합. 시뮬 KE 서기는 힙 25.6 − 무릎 54.9 + 발목 29.8 = +0.5°.")
    print(f"- tau_est 좌우 평균 크기: 무릎 {kn_est:.2f} · 힙 앞뒤 {hf_est:+.2f} · 발목 앞뒤 {af_est:+.2f} N·m")

    print("\n## 2. 물리 무릎각 가정별 모델 토크(발 평평 ∧ 몸통 피치 = IMU 인 자세; 발목 유지 토크 τ_a = −2 · 0 · +2 N·m) — 양발 서기, 다리마다 골반 절반 + 자기 다리")
    print("| 무릎 뒤 처짐 가정 s_k ° | 물리 무릎 ° | 풀린 힙 ° (τ_a 0; 엔코더와의 차) | 풀린 발목 ° (τ_a 0; 엔코더와의 차) | 모델 무릎 N·m: τ_a −2 · 0 · +2 | tau_est / 모델(τ_a 0) | 모델 힙 앞뒤 N·m (tau_est) |"); print("|---|---|---|---|---|---|---|")
    rows = []; mis = hfe - kne + afe - rp[1]
    for sk in (0.0, 5.0, 10.0, 12.5, mis):
        tk3 = [solve_posture(model, kne + sk, rp[1], 0.0, ta)[2] for ta in (-2.0, 0.0, 2.0)]
        hf, af, tk, th, cost = solve_posture(model, kne + sk, rp[1], 0.0, 0.0)
        rows.append((sk, tk, min(tk3), max(tk3)))
        print(f"| {sk:.1f}{' (불일치 전부)' if sk == mis else ''} | {kne + sk:.1f} | {hf:.1f} ({hf - hfe:+.1f}) | {af:.1f} ({af - afe:+.1f}) | {tk3[0]:.2f} · {tk3[1]:.2f} · {tk3[2]:.2f} | {kn_est / tk:.2f} | {th:+.2f} ({hf_est:+.2f}) |")
    qs = np.zeros(12)
    for nm_, val in (("left_hip_f_joint", 25.6), ("right_hip_f_joint", -25.6), ("left_knee_joint", -54.9), ("right_knee_joint", 54.9), ("left_ankle_f_joint", 29.8), ("right_ankle_f_joint", -29.8)):
        qs[JI[nm_]] = np.radians(val)
    tk_sim = float(model.halved("L").hold_torque(qs[None], TP.gdir_from_rp(np.array([0.0]), np.array([0.4])), "L")[0][JI["left_knee_joint"]])
    print(f"- 대조(시뮬 KE 서기의 실제 자세 힙 25.6·무릎 54.9·발목 29.8°·피치 +0.4° 를 같은 모델에): 모델 무릎 {tk_sim:.2f} N·m — 시뮬 적용 토크 7.9–8.2 N·m")
    lo, hi = min(r[2] for r in rows), max(r[3] for r in rows)
    sag = [r for r in rows if r[0] in (10.0, 12.5)]
    print("\n## 3. 가설별 판단")
    print(f"- (i) 자세: 물리 무릎각을 엔코더 값({kne:.1f}°, 처짐 0)부터 사슬 불일치 전부를 무릎에 준 값까지, 압력 중심을 ±2.3 cm 훑어도 모델 무릎 토크는 {lo:.1f}–{hi:.1f} N·m 다. tau_est {kn_est:.2f} 는 그 {kn_est / hi:.2f}–{kn_est / lo:.2f} 배.")
    print(f"  시뮬 서기 모델값 {tk_sim:.1f} 과의 차 {tk_sim - kn_est:.1f} N·m 가운데 자세가 설명할 수 있는 몫: 처짐이 0919 실측(+10–12.5°)대로면 {tk_sim - sag[1][1]:.1f}–{tk_sim - sag[0][1]:.1f} N·m"
          f"({100 * (tk_sim - sag[1][1]) / (tk_sim - kn_est):.0f}–{100 * (tk_sim - sag[0][1]) / (tk_sim - kn_est):.0f} %), 처짐이 0 이라고 쳐도 {tk_sim - rows[0][1]:.1f} N·m({100 * (tk_sim - rows[0][1]) / (tk_sim - kn_est):.0f} %)까지다.")
    print(f"- (iii) 질량: 모델은 17.635 kg(저울 17.62 kg). 무릎 토크는 질량에 비례하므로 tau_est 를 질량만으로 맞추려면 {17.635 * kn_est / tk_sim:.1f} kg 이어야 한다 — 저울값과 맞지 않는다.")
    print(f"- (ii) 남는 것 = 토크 추정 쪽: tau_est 는 모델의 {kn_est / hi:.2f}–{kn_est / lo:.2f} 배(처짐 10–12.5° 면 {kn_est / sag[1][1]:.2f}–{kn_est / sag[0][1]:.2f} 배). 이 기록만으로는 '추정 눈금(전류 → 토크 상수)'과 '기어 마찰이 정지 하중 일부를 받침'을 가르지 못한다.")
    th_h = solve_posture(model, kne + 12.5, rp[1], 0.0, 0.0)[3]
    print(f"  곁 단서: 힙 앞뒤도 tau_est {hf_est:+.2f} 대 모델 {th_h:+.2f} N·m 로 {hf_est / th_h:.2f} 배 — 무릎과 비슷한 비율이다(값이 작아 마찰·잡음과 같은 크기 — 단서일 뿐).")

    # 방향 의존(마찰 단서): 양발 접촉에서 무릎이 굽는 중/펴는 중의 tau_est
    print("\n## 4. 방향 의존 단서(판정 아님) — 양발 접촉 틱에서 무릎이 굽는 중·펴는 중의 |tau_est| (엔코더 속도 |q̇| > 0.05 rad/s)")
    dq = np.gradient(d["q"], 0.02, axis=0)
    for j, sgn, nm in ((6, -1.0, "왼 무릎"), (7, 1.0, "오른 무릎")):
        flexv = sgn * dq[:, j]; a_ = both & (d["t"] > 3.0)
        fx = a_ & (flexv > 0.05); ex = a_ & (flexv < -0.05); st_ = a_ & (np.abs(flexv) <= 0.02)
        g = lambda k: (f"{np.abs(d['tau'][k, j]).mean():.2f} (n {int(k.sum())}, 무릎각 {np.degrees(sgn * d['q'][k, j]).mean():.1f}°)" if k.sum() >= 5 else f"표본 부족(n {int(k.sum())})")   # noqa: E731
        print(f"- {nm}: 굽는 중 {g(fx)} · 펴는 중 {g(ex)} · 정지 {g(st_)}")
        if fx.sum() >= 5 and ex.sum() >= 5:
            up, dn = np.abs(d["tau"][ex, j]).mean(), np.abs(d["tau"][fx, j]).mean()
            print(f"    방향 평균 {0.5 * (up + dn):.2f} · 반차 {0.5 * (up - dn):.2f} N·m — 마찰만이 원인이면 방향 평균이 실제 하중(모델 {lo:.1f}–{hi:.1f})에 가까워야 한다(이 표본은 회복 동작 중이라 하중이 일정하지 않다 — 단서일 뿐)")
    one = (d["cont"][:, 0] < 0.5) ^ (d["cont"][:, 1] < 0.5)
    if one.sum() >= 3:
        st_leg = np.where(d["cont"][one, 0] > 0.5, np.abs(d["tau"][one, 6]), np.abs(d["tau"][one, 7]))
        print(f"- 한발 접촉 틱 {int(one.sum())} 개(회복 스텝 중): 지지 무릎 |tau_est| 평균 {st_leg.mean():.2f} · 최대 {st_leg.max():.2f} N·m (서기의 {st_leg.mean() / kn_est:.1f} 배; 정적 예상은 약 2.3 배 — 움직이는 중이라 참고만)")
    print("\n## 5. 가르는 데 필요한 현장 측정(승인 사항 — 이 스크립트는 실물에 아무것도 하지 않는다)")
    print("- 하중이 걸린 채 무릎을 천천히 굽혔다 펴는 왕복(예: 양발 서기에서 ±5°, 2°/s)에서 같은 각의 tau_est 를 방향별로 본다: 차이 = 2 × 기어 마찰, 평균 ≈ 실제 하중 × 추정 눈금.")
    print(f"  평균이 모델값({lo:.1f}–{hi:.1f} N·m)에 오면 눈금은 맞고 서기의 {kn_est:.1f} 는 마찰이 받친 것, 평균도 절반이면 눈금 쪽이다.")
    print("- 눈금 자체: 다리를 띄운 채 정강이를 수평 가까이 들고 알려진 추(질량·거리)를 달아 tau_est 변화 = m·g·L 인지 본다.")
    print("- 물리 무릎각: 서기 사진·각도기로 넓적다리–정강이 각을 재면 2 절의 어느 행인지(처짐 s_k) 정해진다.")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    main()
