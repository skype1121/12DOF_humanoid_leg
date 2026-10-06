"""실물 블랙박스 재생(계약 검증, 폐루프 증명 아님): LegOdometry 를 실물 신호(qpos_rad·gyro_rad_s·quat_wxyz·contact)에 그대로 돌린다.
참값이 없으므로 측정하는 것은 (1) 조용한 서기 구간의 추정 |v̄| 바닥(줄에 매달리지 않고 제자리 — 이동 0 가정, 운영자/영상 확인 필요),
(2) C5 회복 모드 구간의 |v̄| 분포, (3) 무릎 추정 토크(tau_est_nm)의 변동폭 → 엔코더가 못 보는 처짐 변동 = Δτ/k (k 35–45 N·m/rad, 0919 식별) — C2 가정(±5.5°)의 실물 크기 점검.
사용: python3 stopest_real_replay.py <rl_bridge_*.jsonl>...
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_walking.deploy.leg_odometry import LegOdometry, quat_to_R  # noqa: E402

JN = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint",
      "left_knee_joint", "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def run(path):
    L = [json.loads(l) for l in open(path)]
    L = [r for r in L if "event" not in r and r.get("qpos_rad") and r.get("quat_wxyz")]
    t = np.array([r["timestamp"] for r in L]); t = t - t[0]
    q = np.array([r["qpos_rad"] for r in L]); gy = np.array([r["gyro_rad_s"] for r in L]); qu = np.array([r["quat_wxyz"] for r in L])
    cf = np.array([r["contact"] for r in L]) > 0.5; tau = np.array([r["tau_est_nm"] for r in L])
    mode = np.array([((r.get("c5") or {}).get("mode", 0) or 0) for r in L]); hs = np.array([r.get("hang_state", "") for r in L])
    R = quat_to_R(qu); g = np.einsum("nji,j->ni", R, np.array([0, 0, -1.0]))
    odo = LegOdometry(JN, n=1)
    VB = np.zeros((len(L), 2)); FR = np.zeros(len(L))
    for k in range(len(L)):
        o = odo.step(q[k:k + 1], gy[k:k + 1], g[k:k + 1], cf[k:k + 1])
        VB[k] = o["v_bar"][0]; FR[k] = o["frac"][0]
    sb = np.linalg.norm(VB, axis=-1)
    free = hs == "FREE_STAND"
    # 조용한 서기: 회복 모드 0, 직전 3 s 안 모드 1 없음, 양발 접촉, |ω_xy| 작음(1 s 창 max < 0.3)
    m1 = np.convolve(mode.astype(float), np.ones(150), "full")[:len(L)] > 0
    wq = np.array([np.abs(gy[max(0, k - 50):k + 1, :2]).max() for k in range(len(L))])
    calm = free & (mode == 0) & ~m1 & cf.all(1) & (wq < 0.3) & (FR >= 0.99)
    rec = free & (mode == 1)
    print(f"\n## {os.path.basename(path)} — {t[-1]:.1f} s · FREE_STAND {free.mean()*100:.0f}% · 회복 모드 틱 {int(rec.sum())}")
    for nm, mk in (("조용한 서기", calm), ("회복 모드", rec)):
        if mk.sum() < 10:
            print(f"  {nm}: 표본 부족 {int(mk.sum())}"); continue
        print(f"  {nm} ({mk.sum()/50:.1f} s): 추정 |v̄| p50 {np.percentile(sb[mk],50):.3f} · p90 {np.percentile(sb[mk],90):.3f} · max {sb[mk].max():.3f} m/s · "
              f"θ0.07 초과(=움직임 판정) {100*(sb[mk] >= 0.07).mean():.1f}% · 유효비 p10 {np.percentile(FR[mk],10):.2f}")
    iL, iR = JN.index("left_knee_joint"), JN.index("right_knee_joint")
    for nm, mk in (("조용한 서기", calm), ("회복 모드", rec), ("FREE 전체", free)):
        if mk.sum() < 10:
            continue
        d = [np.percentile(tau[mk, i], 95) - np.percentile(tau[mk, i], 5) for i in (iL, iR)]
        print(f"  무릎 τ_est p5–p95 폭 [{nm}]: 왼 {d[0]:.1f} · 오른 {d[1]:.1f} N·m → 처짐 변동(k 40) 왼 {np.degrees(d[0]/40):.1f}° · 오른 {np.degrees(d[1]/40):.1f}°")
    # 회복 모드 켜짐 사건별: 켜짐 뒤 v̄ 최대·정착(θ0.07, 0.2 s 유지) 시각 vs 모드 꺼짐 시각
    on = np.where((mode[1:] == 1) & (mode[:-1] == 0))[0] + 1
    for k0 in on:
        off = np.where(mode[k0:] == 0)[0]; koff = k0 + off[0] if len(off) else len(L) - 1
        seg = slice(k0, min(len(L), k0 + 400))
        ok = (sb[seg] < 0.07) & (FR[seg] >= 0.5); run_ = 0; kd = None
        for j, v in enumerate(ok):
            run_ = run_ + 1 if v else 0
            if j >= 45 and run_ >= 10 and kd is None:
                kd = j
        print(f"  켜짐 t={t[k0]:.1f}s: 모드 꺼짐 +{t[koff]-t[k0]:.2f}s · 켜짐 뒤 8 s 추정 |v̄| max {sb[seg].max():.3f} · 판정(a) 정착 선언 +{(kd*0.02) if kd is not None else float('nan'):.2f}s")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
