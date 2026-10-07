"""왕복 대본의 방향 틀어짐 누적 — 무엇이 커지는가(TRANSITION_W3KE.md §5.19). 탐색 분석(사후)이다.
사용: python3 reports/gen_w3/lean_yaw.py [--tags tr_x7c_c0_d2_s1,tr_x7c_c0_d2_s2] [--cont tr_x13_d2_s1,tr_x13_d2_s2] [--starts 2,11,20,29,38]
① 연속 걷기(멈춤 없음)에서 요 속도가 시간에 따라 커지는가 ② 같은 개체의 걷기 사이 요 방향 유지 ③ 걷는 중 평균 롤(쏠림)과 요 속도의 관계(주기별 r·기울기)
④ 서 있는 동안의 롤과 걷는 중 롤의 크기(부호 정렬 평균)."""
import argparse, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def load(tag, end):
    zk = np.load(os.path.join(LP, f"kwloop_{tag}.npz"), allow_pickle=True); pt = os.path.join(LP, f"transall_{tag}.npz")
    t = zk["t"].astype(float); yaw = np.degrees(np.unwrap(zk["yaw"].astype(float), axis=0)); pre = zk["pre_pass"].astype(bool)
    tf = np.where(zk["fell"].astype(bool), zk["t_fall_post"].astype(float), np.inf); ok = pre & (tf >= end)
    roll = None
    if os.path.exists(pt):
        zt = np.load(pt, allow_pickle=True); n = zt["roll"].shape[1]; roll = np.full((len(t), yaw.shape[1]), np.nan); roll[:, :n] = zt["roll"].astype(float)
    return t, yaw, ok, roll, zk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="tr_x7c_c0_d2_s1,tr_x7c_c0_d2_s2"); ap.add_argument("--cont", default="tr_x13_d2_s1,tr_x13_d2_s2")
    ap.add_argument("--starts", default="2,11,20,29,38"); ap.add_argument("--walk_s", type=float, default=4.0); ap.add_argument("--end", type=float, default=47.0); ap.add_argument("--name", default="")
    a = ap.parse_args(); starts = [float(x) for x in a.starts.split(",")]; C = len(starts)
    if a.cont:
        print("### ① 연속 걷기(제자리 2–8 s → 앞 0.06 8–16 s, 멈춤 없음, 토크 한계 1.0 배 env)")
        D4 = []
        for tag in a.cont.split(","):
            if not os.path.exists(os.path.join(LP, f"kwloop_{tag}.npz")):
                continue
            t, yaw, ok, _, zk = load(tag, 17.0); K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
            if "effort_scale" in zk.files:
                ok = ok & (np.abs(zk["effort_scale"].astype(float) - 1.0) < 1e-6)
            D4.append(np.stack([yaw[K(s0 + 4), ok] - yaw[K(s0), ok] for s0 in (2, 6, 10)], 1))
        if D4:
            D4 = np.concatenate(D4); print(f"- env {len(D4)} · 4 s 구간 요 변화 표준편차 [°] (2–6 / 6–10 / 10–14 s): " + " / ".join(f"{v:.1f}" for v in D4.std(0)) + " · |중앙| " + " / ".join(f"{v:.1f}" for v in np.median(np.abs(D4), 0)))
    Y = []; R = []; RS = []; W = []
    for tag in a.tags.split(","):
        if not os.path.exists(os.path.join(LP, f"kwloop_{tag}.npz")):
            print("없음", tag); continue
        t, yaw, ok, roll, _ = load(tag, a.end); K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        Y.append(np.stack([yaw[K(s + a.walk_s), ok] - yaw[K(s), ok] for s in starts], 1)); W.append(np.stack([(yaw[K(s + 4), ok] - yaw[K(s + 2), ok]) / 2.0 for s in starts], 1))
        if roll is not None:
            R.append(np.stack([roll[K(s + 2):K(s + 4), ok].mean(0) for s in starts], 1)); RS.append(np.stack([roll[K(s) - 10:K(s), ok].mean(0) for s in starts], 1))
    if not Y:
        return 1
    Y = np.concatenate(Y); W = np.concatenate(W); c = lambda x, y: float(np.corrcoef(x, y)[0, 1])   # noqa: E731
    print(f"### ② 왕복({a.name or a.tags}) — 다섯 걷기를 모두 마친 env {len(Y)}")
    print("- 걷기 한 번(4 s)의 요 변화 표준편차 [°]: " + " → ".join(f"{v:.1f}" for v in Y.std(0)) + " · 걷는 중(2–4 s) 요 속도 표준편차 [°/s]: " + " → ".join(f"{v:.1f}" for v in W.std(0)))
    print("- 같은 env 의 이웃한 걷기 요 변화 상관 r: " + " · ".join(f"{c(Y[:, i], Y[:, i + 1]):.2f}" for i in range(C - 1)) + f" · 첫째–다섯째 {c(Y[:, 0], Y[:, -1]):.2f}"
          + f" · 걷기 2–{C} 가 전부 같은 방향인 env {float((np.abs(np.sign(Y[:, 1:]).sum(1)) == C - 1).mean()) * 100:.0f} % (독립이면 {100 * 2 * 0.5 ** (C - 1):.1f} %)")
    if R:
        R = np.concatenate(R); RS = np.concatenate(RS); m = np.isfinite(R).all(1) & np.isfinite(RS).all(1); R = R[m]; RS = RS[m]; Wm = W[m]
        print(f"### ③ 걷는 중 평균 롤(쏠림)과 요 속도 — 추적 기록이 있는 env {int(m.sum())}")
        print("| 걷기 | 걷는 중 평균 롤 표준편차 [°] | r(걷는 중 롤, 요 속도) | 기울기 [(°/s)/°] | 걷기 직전 서 있는 롤 표준편차 [°] | r(직전 서기 롤, 요 속도) | 기울기 |"); print("|---|---|---|---|---|---|---|")
        for i in range(C):
            print(f"| {i + 1} | {R[:, i].std():.2f} | {c(R[:, i], Wm[:, i]):+.2f} | {np.polyfit(R[:, i], Wm[:, i], 1)[0]:+.2f} | {RS[:, i].std():.2f} | {c(RS[:, i], Wm[:, i]):+.2f} | {np.polyfit(RS[:, i], Wm[:, i], 1)[0]:+.2f} |")
        sg = np.sign(Wm[:, -1]); sg[sg == 0] = 1
        print("### ④ 다섯째 걷기의 요 방향으로 부호를 맞춘 평균(마지막 걷기는 선택 편향 포함)")
        print("| 걷기 | 직전 서 있는 롤 [°] | 걷는 중 롤 [°] | 걷는 중 요 속도 [°/s] |"); print("|---|---|---|---|")
        for i in range(C):
            print(f"| {i + 1} | {(RS[:, i] * sg).mean():+.2f} | {(R[:, i] * sg).mean():+.2f} | {(Wm[:, i] * sg).mean():+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
