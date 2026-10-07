"""X18(TORQUE_BUDGET_1007.md §19): 부분지지 비율별 W3 걷기 부하·걸음 — 추적(transall, 관절 토크 포함)에서 집계. 판정 없는 측정 + 미리 적은 읽는 법.
사용: python3 reports/gen_w3/x18_summ.py [--seeds 21,22] [--fr 00,20,35,50]
부하 = AK70 10 관절 Σ τ², 0.5 s 창 최대. 기준 = **받침 없는(00) 실행의 서기(29–31 s) 중앙**(= 온 체중 서기). 명령 적용 뒤 1 s 는 제외."""
import argparse, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
SEG = [("제자리", 3, 8), ("앞 0.03", 9, 14), ("앞 0.06", 15, 20), ("앞 0.10", 21, 26)]
STAND = (29, 31)
READ_MAX = 6.5      # 읽는 법(§19): 창 최대의 p90 이 온 체중 서기의 6.5 배 이하 = 실물에서 재시작 없이 지나간 거친 회복(6.5–9 배)의 아래 끝 이하


def load(tag):
    f = os.path.join(LP, f"transall_{tag}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in ("t", "tq", "alive", "pre_pass", "cont", "vxy", "q")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="21,22"); ap.add_argument("--fr", default="00,20,35,50")
    a = ap.parse_args(); seeds = a.seeds.split(","); frs = a.fr.split(",")
    D = {(u, s): load(f"x18_u{u}_d2_s{s}") for u in frs for s in seeds}
    miss = [k for k, v in D.items() if v is None]
    if miss:
        print("없는 실행:", miss)
    ref = []
    for s in seeds:
        d = D.get(("00", s))
        if d is None:
            continue
        t = d["t"].astype(float); K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        s2 = (d["tq"].astype(np.float32)[:, :, :10] ** 2).sum(-1); m = d["pre_pass"].astype(bool)[:s2.shape[1]] & d["alive"][K(STAND[1]) - 1]
        ref.append(s2[K(STAND[0]):K(STAND[1])][:, m].ravel())
    if not ref:
        raise SystemExit("받침 없는(00) 실행이 없다 — 기준을 못 만든다")
    b50 = float(np.median(np.concatenate(ref)))
    print(f"## X18 — 부분지지 비율별 W3 걷기 부하 (seed {a.seeds}, 추적 env 128/실행) · 온 체중 서기 Σ τ² 중앙 {b50:.0f}")
    print("| 받침 | 명령 | 구간 끝 생존 / 통과 | 창 최대 Σ τ² 의 온 체중 서기 대비 배수: 중앙 / p90 / 최대 | 무릎 RMS L/R · 힙 외전 RMS L/R [N·m] | \\|τ\\| ≥ 24.5 시간 % (무릎 · 힙 외전) | 한발 지지 % | 5 s 전진 중앙 [m] |"); print("|---|---|---|---|---|---|---|---|")
    worst = {}
    for u in frs:
        for nm, a0, b0 in SEG:
            W = []; al = 0; n = 0; kn = []; ha = []; satk = []; sath = []; ss = []; fw = []
            for s in seeds:
                d = D.get((u, s))
                if d is None:
                    continue
                t = d["t"].astype(float); K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
                tq = d["tq"].astype(np.float32); na = tq.shape[1]; pre = d["pre_pass"].astype(bool)[:na]; m = pre & d["alive"][K(b0) - 1]
                n += int(pre.sum()); al += int(m.sum())
                if not m.any():
                    continue
                s2 = (tq[:, :, :10] ** 2).sum(-1)
                for k in range(K(a0), K(b0) - 24, 25):
                    W.append(s2[k:k + 25][:, m].max(0))
                x = tq[K(a0):K(b0)][:, m]
                kn.append(x[:, :, 6:8].reshape(-1, 2)); ha.append(x[:, :, 2:4].reshape(-1, 2))
                c = d["cont"][K(a0):K(b0)][:, m].astype(bool); ss.append((c.sum(-1) == 1).ravel())
                v = d["vxy"][K(a0):K(b0)][:, m].astype(np.float32); fw.append(v[:, :, 0].sum(0) * 0.02)
            if not W:
                print(f"| {u} % | {nm} | 0 / {n} | — | — | — | — | — |"); worst[u] = float("inf"); continue
            W = np.concatenate(W); kn = np.concatenate(kn); ha = np.concatenate(ha); ss = np.concatenate(ss); fw = np.concatenate(fw)
            r50, r90, rmx = np.median(W) / b50, np.percentile(W, 90) / b50, W.max() / b50
            worst[u] = max(worst.get(u, 0.0), r90)
            rms = lambda y: np.sqrt((y ** 2).mean(0))   # noqa: E731
            print(f"| {u} % | {nm} | {al} / {n} | {r50:.1f} / {r90:.1f} / {rmx:.1f} | {rms(kn)[0]:.1f}/{rms(kn)[1]:.1f} · {rms(ha)[0]:.1f}/{rms(ha)[1]:.1f} | "
                  f"{100 * (np.abs(kn) >= 24.5).mean():.1f} · {100 * (np.abs(ha) >= 24.5).mean():.1f} | {100 * ss.mean():.0f} | {np.median(fw):+.3f} |")
    print("- 읽는 법(§19, 실행 전에 적음): 네 명령 모두에서 창 최대 p90 ≤ " + f"{READ_MAX} 배인 가장 작은 받침 비율 → " +
          (next((f"**{u} %**" for u in frs if worst.get(u, float('inf')) <= READ_MAX), "없음(50 % 에서도 넘음)")) +
          " · 비율별 최악 p90: " + " · ".join(f"{u} % {worst.get(u, float('nan')):.1f} 배" for u in frs))


if __name__ == "__main__":
    main()
