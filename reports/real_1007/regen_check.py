"""실물 블랙박스에서 드라이버 재시작 직전의 순 기계 출력(Σ τ_est·q̇) — 구동(+)인가 회생(−)인가 (HARDWARE_P0.md §3 의 회생 가설 점검).
사용: python3 reports/real_1007/regen_check.py <블랙박스.jsonl> [...]
재시작 시점 = 발목 롤(목록 끝 두 관절 또는 이름에 ankle_r)의 실측 관절각이 0.15 s 넘게 멈춘 첫 틱(τ_est 가 0 코드로 굳기 전). 판정이 아니라 관찰이다 — τ_est 눈금이 절반일 수 있다(그러면 값은 2 배)."""
import json, sys
import numpy as np
JN = ["left_hip_f", "right_hip_f", "left_hip_a", "right_hip_a", "left_hip_r", "right_hip_r", "left_knee", "right_knee", "left_ankle_f", "right_ankle_f", "left_ankle_r", "right_ankle_r"]


def main():
    for p in sys.argv[1:]:
        rows = []
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "tau_est_nm" in r and "qvel_rad_s" in r and "qpos_rad" in r:
                rows.append(r)
        if len(rows) < 100:
            print(p, "틱 부족"); continue
        tau = np.array([r["tau_est_nm"] for r in rows], float); qv = np.array([r["qvel_rad_s"] for r in rows], float); q = np.array([r["qpos_rad"] for r in rows], float)
        t = np.array([r["timestamp"] for r in rows], float); t -= t[0]
        P = tau * qv; Pt = P.sum(1); neg = np.where(P < 0, P, 0).sum(1); pos = np.where(P > 0, P, 0).sum(1); cu = (tau[:, :10] ** 2).sum(1)
        print(f"## {p.split('/')[-1]} — {len(t)} 틱 · {t[-1]:.1f} s")
        print(f"- 세션 전체: 순 기계 출력 최소 {Pt.min():+.1f} W @ {t[Pt.argmin()]:.2f} s · 최대 {Pt.max():+.1f} W @ {t[Pt.argmax()]:.2f} s · 회생 합 p1 {np.percentile(neg, 1):+.1f} · 구동 합 p99 {np.percentile(pos, 99):+.1f} · Σ τ²(AK70) 최대 {cu.max():.0f} @ {t[cu.argmax()]:.2f} s")
        # 발목 롤 실측이 멈춘 구간(0.15 s 이상 값 불변) 찾기
        ev = []
        for j in (10, 11):
            same = np.abs(np.diff(q[:, j])) < 1e-9; k = 0
            while k < len(same):
                if same[k]:
                    k2 = k
                    while k2 < len(same) and same[k2]:
                        k2 += 1
                    if t[min(k2, len(t) - 1)] - t[k] >= 0.15 and np.abs(qv[max(k - 25, 0):k + 1]).max() > 0.5:      # 움직이던 중에 굳음
                        ev.append((t[k], j, t[min(k2, len(t) - 1)] - t[k]))
                    k = k2
                else:
                    k += 1
        ev.sort()
        if not ev:
            print("- 움직이던 중 발목 롤 실측이 0.15 s 넘게 멈춘 구간 없음"); continue
        seen = []
        for te, j, dur in ev:
            if any(abs(te - s0) < 1.0 for s0 in seen):
                continue
            seen.append(te); k = int(np.searchsorted(t, te)); a = max(k - 25, 0)
            w = slice(a, k + 1); kmin = a + int(Pt[w].argmin()); kcu = a + int(cu[w].argmax())
            print(f"- 발목 롤 멈춤 {te:.2f} s({JN[j]}, {dur:.2f} s 이상): 직전 0.5 s 의 순 기계 출력 평균 {Pt[w].mean():+.1f} W · 최소 {Pt[kmin]:+.1f} W @ {t[kmin]:.2f} s · 회생 합 최소 {neg[w].min():+.1f} W · 구동 합 최대 {pos[w].max():+.1f} W"
                  f" · Σ τ² 최대 {cu[kcu]:.0f} @ {t[kcu]:.2f} s(그 틱 순 출력 {Pt[kcu]:+.1f} W) · 세션 회생 순위: 최소값이 전체 틱의 하위 {100 * (Pt < Pt[kmin]).mean():.2f} %")
            jmin = np.argsort(P[kmin])[:3]; print("    최대 회생 틱의 관절별 출력 [W] 하위 3: " + ", ".join(f"{JN[i]} {P[kmin, i]:+.1f}(τ {tau[kmin, i]:+.1f}·q̇ {qv[kmin, i]:+.2f})" for i in jmin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
