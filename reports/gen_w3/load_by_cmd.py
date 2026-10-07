"""x12: W3 걷기 명령별 부하 — 추적(transall, 관절 토크 포함)에서 대본 구간별 Σ τ²(AK70 10 관절)·관절별 |τ| 를 집계(판정 없는 측정).
사용: python3 reports/gen_w3/load_by_cmd.py [TAG]   (기본 x12_w3load_cmds_d2_s21)
구간은 대본 시각 고정(명령 적용 뒤 1 s 는 제외). 창 = 0.5 s, 창 최대의 env·창 분포. 서기 기준 = KE 서기 구간(26–31·43–48·54–59 s 의 마지막 2 s)."""
import os, sys
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
SEG = [("제자리", 3, 8), ("앞 0.03", 9, 14), ("앞 0.06", 15, 20), ("앞 0.10", 21, 26), ("뒤 0.04", 32, 37), ("뒤 0.06", 38, 43), ("옆 0.035", 49, 54), ("회전 0.15", 60, 65)]
STAND = [(29, 31), (46, 48), (57, 59)]
JN = ["LhF", "RhF", "LhA", "RhA", "LhR", "RhR", "Lkn", "Rkn", "LaF", "RaF", "LaR", "RaR"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "x12_w3load_cmds_d2_s21"
    z = np.load(os.path.join(LP, f"transall_{tag}.npz"), allow_pickle=True)
    t = z["t"].astype(float); tq = z["tq"].astype(np.float32); alive = z["alive"]; pre = z["pre_pass"].astype(bool)
    K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
    s2 = (tq[:, :, :10] ** 2).sum(-1)
    base = np.concatenate([s2[K(a):K(b)][:, pre & alive[K(b) - 1]].ravel() for a, b in STAND]); b50 = float(np.median(base))
    print(f"## {tag} — 통과 env {int(pre.sum())} · 서기 Σ τ² 중앙 {b50:.0f} (p99 {np.percentile(base, 99):.0f})")
    print("| 명령 | 구간 끝 생존 | 0.5 s 창 최대 Σ τ²: 중앙 / p90 / 최대 | 서기 대비 배수(중앙 / p90) | 관절별 \\|τ\\| p95 상위 3 [N·m] |"); print("|---|---|---|---|---|")
    for nm, a, b in SEG:
        m = pre & alive[K(b) - 1]; W = []
        for k in range(K(a), K(b) - 24, 25):
            W.append(s2[k:k + 25][:, m].max(0))
        W = np.concatenate(W) if W else np.zeros(1)
        p95 = np.percentile(np.abs(tq[K(a):K(b)][:, m]).reshape(-1, 12), 95, axis=0); top = np.argsort(-p95)[:3]
        print(f"| {nm} | {int(m.sum())} | {np.median(W):.0f} / {np.percentile(W, 90):.0f} / {W.max():.0f} | {np.median(W)/b50:.1f} / {np.percentile(W, 90)/b50:.1f} | " + ", ".join(f"{JN[j]} {p95[j]:.1f}" for j in top) + " |")


if __name__ == "__main__":
    main()
