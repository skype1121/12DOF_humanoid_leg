"""왕복 대본: 걷기 시작 시점의 상태(발 배치·관절·자세)가 그 걷기의 요 변화(부호 포함)를 얼마나 설명하는가 — 누적 원인 탐색(TRANSITION_W3KE.md §5.11e 후속).
사용: python3 reports/gen_w3/yaw_start_feat.py --tags tr_x7c_c0_d2_s1,tr_x7c_c0_d2_s2 [--starts 2,11,20,29,38] [--walk_s 4]
탐색 분석이다(사후). 상관은 원인이 아니다 — 후보 변수를 골라 '그 변수만 바꾼' 실험으로 넘긴다."""
import argparse, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--starts", default="2,11,20,29,38"); ap.add_argument("--walk_s", type=float, default=4.0)
    a = ap.parse_args(); starts = [float(x) for x in a.starts.split(",")]
    X = []; Y = []; CY = []; names = None; JN = None
    for tag in a.tags.split(","):
        pk = os.path.join(LP, f"kwloop_{tag}.npz"); pt = os.path.join(LP, f"transall_{tag}.npz")
        if not (os.path.exists(pk) and os.path.exists(pt)):
            print("없음", tag); continue
        zk = np.load(pk, allow_pickle=True); zt = np.load(pt, allow_pickle=True)
        t = zk["t"].astype(float); yaw = zk["yaw"].astype(float); pre = zk["pre_pass"].astype(bool); n = zt["q"].shape[1]
        tfp = zk["t_fall_post"].astype(float); fell = zk["fell"].astype(bool); tf = np.where(fell, tfp, np.inf)[:n]
        q = zt["q"].astype(float); ff = zt["ffwd"].astype(float); fl = zt["flat"].astype(float); roll = zt["roll"].astype(float); pitch = zt["pitch"].astype(float); h = zt["h"].astype(float)
        tq = zt["tq"].astype(float); cont = zt["cont"].astype(float)
        if JN is None:
            tr = np.load(os.path.join(LP, f"trans_{tag}.npz"), allow_pickle=True); JN = [str(x) for x in tr["joint_names"]]
        K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        for c, ts in enumerate(starts):
            k0, k1 = K(ts), K(ts + a.walk_s)
            ok = pre[:n] & (tf >= ts + a.walk_s + 0.5)
            ks = slice(max(k0 - 10, 0), k0)                                # 시작 직전 0.2 s 평균(서기 상태)
            feat = {
                "발 앞뒤 엇갈림(L−R) [m]": (ff[ks, :, 0] - ff[ks, :, 1]).mean(0),
                "발 앞뒤 평균 [m]": (ff[ks, :, 0] + ff[ks, :, 1]).mean(0) / 2,
                "발 좌우 간격(L−R) [m]": (fl[ks, :, 0] - fl[ks, :, 1]).mean(0),
                "발 좌우 중심 [m]": (fl[ks, :, 0] + fl[ks, :, 1]).mean(0) / 2,
                "롤 [rad]": roll[ks].mean(0), "피치 [rad]": pitch[ks].mean(0), "골반 높이 [m]": h[ks].mean(0),
            }
            for j, nm in enumerate(JN):
                feat[f"q {nm}"] = q[ks, :, j].mean(0)
            # 좌우 거울 차(같은 관절의 L − R 또는 L + R): 이름 규칙에 맞춰 짝을 찾는다
            for j, nm in enumerate(JN):
                if nm.startswith("left_") or nm.startswith("l_") or nm.startswith("L"):
                    base = nm.split("_", 1)[1] if "_" in nm else nm[1:]
                    for j2, nm2 in enumerate(JN):
                        if nm2 != nm and nm2.endswith(base) and (nm2.startswith("right_") or nm2.startswith("r_") or nm2.startswith("R")):
                            feat[f"q {base}: L−R"] = q[ks, :, j].mean(0) - q[ks, :, j2].mean(0); feat[f"q {base}: L+R"] = q[ks, :, j].mean(0) + q[ks, :, j2].mean(0)
            if names is None:
                names = list(feat)
            dy = np.degrees(wrap(yaw[k1, :n] - yaw[k0, :n]))
            X.append(np.stack([feat[k][ok] for k in names], 1)); Y.append(dy[ok]); CY.append(np.full(int(ok.sum()), c))
    X = np.concatenate(X); Y = np.concatenate(Y); CY = np.concatenate(CY)
    print(f"표본 {len(Y)} (걷기 × env), 요 변화 [°]: 평균 {Y.mean():+.2f} · 표준편차 {Y.std():.2f} · |중앙| {np.median(np.abs(Y)):.2f}")
    print("관절 이름:", JN)
    rows = []
    for i, nm in enumerate(names):
        x = X[:, i]
        if x.std() < 1e-9:
            continue
        r = float(np.corrcoef(x, Y)[0, 1]); rc = [float(np.corrcoef(x[CY == c], Y[CY == c])[0, 1]) if (CY == c).sum() > 10 and x[CY == c].std() > 1e-9 else float("nan") for c in range(len(starts))]
        sd = [float(x[CY == c].std()) for c in range(len(starts))]
        rows.append((abs(r), nm, r, rc, sd))
    rows.sort(reverse=True)
    print("| 특징(걷기 시작 직전 0.2 s 평균) | 전체 r | 주기별 r (1→5) | 표준편차 주기 1 → 5 |"); print("|---|---|---|---|")
    for _, nm, r, rc, sd in rows[:18]:
        print(f"| {nm} | {r:+.2f} | " + " ".join(f"{v:+.2f}" for v in rc) + f" | {sd[0]:.4f} → {sd[-1]:.4f} |")
    # 다중 선형: 상위 특징으로 설명되는 분산(주기 묶음 5 겹 교차검증)
    top = [names.index(nm) for _, nm, *_ in rows[:8]]
    Z = (X[:, top] - X[:, top].mean(0)) / (X[:, top].std(0) + 1e-12); rng = np.random.default_rng(0); idx = rng.permutation(len(Y)); pred = np.zeros_like(Y)
    for f in range(5):
        te = idx[f::5]; trn = np.setdiff1d(idx, te); A = np.c_[Z[trn], np.ones(len(trn))]; w, *_ = np.linalg.lstsq(A, Y[trn], rcond=None); pred[te] = np.c_[Z[te], np.ones(len(te))] @ w
    print(f"- 상위 8 특징 선형 모형의 교차검증 R² = {1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum():.2f}")
    # 직전 걷기의 요 변화와의 자기상관(같은 env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
