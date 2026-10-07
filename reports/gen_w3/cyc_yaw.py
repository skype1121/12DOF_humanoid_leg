"""왕복 대본 실행의 주기별 요 변화·옆 이동·낙상 — 좌우 비대칭 누적(TRANSITION_W3KE.md §5.11e) 확인용.
사용: python3 reports/gen_w3/cyc_yaw.py --tags TAG[,TAG...] [--starts 2,11,20,29,38] [--walk_s 4] [--name 이름]
주기 = starts[i] 부터 starts[i+1] 전까지. 요 변화 = 주기 시작 → 시작 + walk_s (그 구간을 살아서 지난 env). 낙상 = 주기 안 낙상 / 주기 시작 생존."""
import argparse, csv, json, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--starts", default="2,11,20,29,38"); ap.add_argument("--walk_s", type=float, default=4.0)
    ap.add_argument("--name", default=""); ap.add_argument("--end", type=float, default=47.0)
    a = ap.parse_args(); starts = [float(x) for x in a.starts.split(",")]; C = len(starts)
    dy = [[] for _ in range(C)]; lat = [[] for _ in range(C)]; fwd = [[] for _ in range(C)]; n0 = [0] * C; nf = [0] * C; roll0 = [[] for _ in range(C)]; asym = [[] for _ in range(C)]; hl_f = 0; got = []
    for tag in a.tags.split(","):
        p = os.path.join(LP, f"kwloop_{tag}.npz"); q = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
        if not (os.path.exists(p) and os.path.exists(q)):
            continue
        got.append(tag); z = np.load(p, allow_pickle=True); rows = {int(r["env"]): r for r in csv.DictReader(open(q))}
        t = z["t"].astype(float); pre = z["pre_pass"].astype(bool); N = pre.shape[0]; tf = np.full(N, np.inf)
        for e, r in rows.items():
            if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
                tf[e] = float(r["t_fall_rel"])
        yaw = z["yaw"].astype(float); xy = z["xy"].astype(float); K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        pg = z["pg"].astype(float) if "pg" in z.files and z["pg"].shape[0] else None
        if "hl_nfault" in z.files:
            hl_f += int(z["hl_nfault"][pre].sum())
        for c, ts in enumerate(starts):
            t1 = starts[c + 1] if c + 1 < C else a.end
            al = pre & (tf >= ts); n0[c] += int(al.sum()); nf[c] += int((al & (tf < t1)).sum())
            ok = pre & (tf >= ts + a.walk_s + 0.5); k0, k1 = K(ts), K(ts + a.walk_s)
            d = np.degrees(np.arctan2(np.sin(yaw[k1] - yaw[k0]), np.cos(yaw[k1] - yaw[k0]))); dxy = xy[k1] - xy[k0]; cy, sy = np.cos(yaw[k0]), np.sin(yaw[k0])
            dy[c].append(d[ok]); fwd[c].append((cy * dxy[:, 0] + sy * dxy[:, 1])[ok]); lat[c].append((-sy * dxy[:, 0] + cy * dxy[:, 1])[ok])
            if pg is not None:
                roll0[c].append(np.degrees(np.arcsin(np.clip(pg[max(k0 - 10, 0):k0, :, 1].mean(0), -1, 1)))[ok])
    print(f"## {a.name or a.tags} — 실행 {len(got)} 개" + (f" · 헤딩 고리 고장 {hl_f} 건" if hl_f else ""))
    print("| 주기 | 시작 생존 | 주기 낙상 | 걷기 중 \\|요 변화\\| 중앙 / p90 [°] | 요 변화 부호 중앙 | 전진 중앙 [m] | \\|옆 이동\\| 중앙 / p90 [m] | 시작 \\|롤\\| 중앙 / p90 [°] |"); print("|---|---|---|---|---|---|---|---|")
    for c in range(C):
        d = np.concatenate(dy[c]) if dy[c] else np.zeros(0); f = np.concatenate(fwd[c]) if fwd[c] else np.zeros(0); l = np.concatenate(lat[c]) if lat[c] else np.zeros(0); r0 = np.concatenate(roll0[c]) if roll0[c] else np.zeros(0)
        rs = f"{np.median(np.abs(r0)):.2f} / {np.percentile(np.abs(r0), 90):.2f}" if len(r0) else "—"
        print(f"| {c+1} | {n0[c]} | {nf[c]} ({100*nf[c]/max(n0[c],1):.1f} %) | {np.median(np.abs(d)):.1f} / {np.percentile(np.abs(d), 90):.1f} | {np.median(d):+.1f} | {np.median(f):+.2f} | {np.median(np.abs(l)):.2f} / {np.percentile(np.abs(l), 90):.2f} | {rs} |")
    print(f"- 주기 2 이후 낙상 합 {sum(nf[1:])} / {sum(n0[1:])} ({100*sum(nf[1:])/max(sum(n0[1:]),1):.2f} %) · 첫 주기 {nf[0]} / {n0[0]}")


if __name__ == "__main__":
    main()
