"""추적(transall, q·tq 포함)에서 회복 활성화별 부하 통계 — 실물 재시작 구간과 비교(reports/real_1007/README.md 보강 2).
사용: python3 reports/gen_w3/load_stats.py TAG [TAG...]     (TAG = out_tag, 예 x10_R0t_b18_d2_s21)
부하 대용: Σ τ²(AK70 10 관절) · 가지별 Σ|τ|(왼 1–5 / 오른 7–11) · 음의 기계 출력(회생) · 관절 최대 속도. 활성화 = 회복 모드 켜짐 → 2 s 창. --walk 이면 걷기 구간(ge) 전체를 0.5 s 창으로."""
import os, sys
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
L = [0, 2, 4, 6, 8]; R = [1, 3, 5, 7, 9]


def main():
    walk = "--walk" in sys.argv; tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    print("| 실행 | 창 수 | Σ τ² 최대: p10 / 중앙 / p90 / 최대 | 가지 Σ\\|τ\\| 최대 중앙(왼 / 오른) | 회생 출력 최저 중앙 [W] | 인출 출력 최대 중앙 [W] | 서기 Σ τ² 중앙 / p99 | 서기 대비 배수(중앙) |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in tags:
        p = os.path.join(LP, f"transall_{tag}.npz")
        if not os.path.exists(p):
            print(f"| {tag} | (추적 없음) |"); continue
        z = np.load(p, allow_pickle=True)
        if "tq" not in z.files:
            print(f"| {tag} | (tq 없음) |"); continue
        tq = z["tq"].astype(np.float32); q = z["q"].astype(np.float32); mode = z["mode"]; alive = z["alive"]; pp = z["pre_pass"]; t = z["t"]; ge = z["ge"]
        T, N, _ = tq.shape; qd = np.gradient(q, axis=0) / 0.02
        s2 = (tq[:, :, :10] ** 2).sum(-1); bl = np.abs(tq[:, :, L]).sum(-1); br = np.abs(tq[:, :, R]).sum(-1); P = (tq[:, :, :10] * qd[:, :, :10]).sum(-1)
        A = {"s2": [], "bl": [], "br": [], "pmin": [], "pmax": []}
        for e in range(N):
            if not pp[e]:
                continue
            if walk:
                idx = np.nonzero(ge[:, e] & alive[:, e])[0]; starts = idx[::25] if len(idx) else []
                wins = [slice(k, min(k + 25, T)) for k in starts]
            else:
                on = np.nonzero(mode[1:, e] & ~mode[:-1, e])[0] + 1; wins = [slice(k, min(k + 100, T)) for k in on]
            for w in wins:
                if not alive[w, e].all():
                    w = slice(w.start, w.start + max(1, int(np.argmin(alive[w, e]))))
                if w.stop - w.start < 5:
                    continue
                A["s2"].append(s2[w, e].max()); A["bl"].append(bl[w, e].max()); A["br"].append(br[w, e].max()); A["pmin"].append(P[w, e].min()); A["pmax"].append(P[w, e].max())
        if not A["s2"]:
            print(f"| {tag} | 0 |"); continue
        # 서기 기준: 인계 뒤 1–3.5 s 중 회복 모드·걷기 아님
        k0, k1 = int(np.searchsorted(t, 1.0)), int(np.searchsorted(t, 3.5)); m = (~mode[k0:k1]) & (~ge[k0:k1]) & alive[k0:k1] & pp[None, :]
        base = s2[k0:k1][m]; b50 = float(np.median(base)) if base.size else float("nan"); b99 = float(np.percentile(base, 99)) if base.size else float("nan")
        v = np.array(A["s2"]); f = lambda a, pc: float(np.percentile(a, pc))   # noqa: E731
        print(f"| {tag} | {len(v)} | {f(v,10):.0f} / {f(v,50):.0f} / {f(v,90):.0f} / {v.max():.0f} | {f(A['bl'],50):.0f} / {f(A['br'],50):.0f} | {f(A['pmin'],50):.0f} | {f(A['pmax'],50):.0f} | {b50:.0f} / {b99:.0f} | {f(v,50)/b50 if b50 == b50 and b50 > 0 else float('nan'):.1f} |")


if __name__ == "__main__":
    main()
