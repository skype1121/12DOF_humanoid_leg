"""Q6c 오프라인 감지기 선택(TRANSITION_W3KE.md Q6) — 걷는 중 '뒤 밀기' 관측 기반 감지.
입력: log_picture/transall_tr_walk_q6{nd,b12,b18,f12,f18}_d2_s*.npz (개입 없는 걷기, 정책 관측 자이로·중력·접촉, 위상 u, ge).
신호(정책 관측만): 피치 θ = atan2(g_x, −g_z)[°], 롤 φ = atan2(g_y, −g_z)[°], 자이로 ω_y[rad/s].
후보:
  A  기울기 편차 크기 |(θ, φ) − EMA(τ 2 s)|            (rsup 식, 방향 = 피치 편차 부호)
  B  피치 편차 θ − EMA(τ 2 s)                           (피치 축만)
  C  위상 정합 피치 잔차: θ − μ_θ(u) (무외란 걷기의 위상별 평균, 32 구간)
  D  자이로 ω_y 저역(EMA α 0.33) + 위상 정합: ω_y − μ_ω(u)
  E  C 와 D 의 결합: z = (θ − μ_θ(u))/σ_θ(u) + k·(ω_y − μ_ω(u))/σ_ω(u)   (k 1)
'뒤' 판정 = 신호 < −문턱 이 hold 틱(기본 2) 연속. '앞' = 신호 > +문턱.
지표: 무외란 걷기 '뒤' 오경보 [회/env·분](nd 전체 + 외란 실행의 외란 전 구간) · 뒤 밀기 검출률(외란 시작 뒤 지연 예산 T 안) · 검출 지연 분포 · 앞 밀기에서 '뒤' 오판정률.
위상 통계(μ, σ)는 nd seed 1 로만 만들고(학습), 평가는 nd seed 2 + 외란 실행으로 한다(누설 방지).
사용: python3 reports/gen_w3/q6_detect.py [--budget 0.4] [--far 0.2]
"""
import argparse, glob, os, re, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LP = os.path.join(REPO, "log_picture")
DT = 0.02; NB = 32


def load(tag):
    out = []
    for p in sorted(glob.glob(os.path.join(LP, f"transall_tr_walk_q6{tag}_d2_s*.npz"))):
        z = np.load(p, allow_pickle=True); s = int(re.search(r"_s(\d+)\.npz", p).group(1))
        g = z["grav"].astype(np.float64); gy = z["gyro"].astype(np.float64)
        th = np.degrees(np.arctan2(g[..., 0], -g[..., 2])); ph = np.degrees(np.arctan2(g[..., 1], -g[..., 2]))
        out.append(dict(seed=s, t=z["t"].astype(np.float64), th=th, ph=ph, wy=gy[..., 1], wx=gy[..., 0], u=z["u"].astype(np.float64), ge=z["ge"].astype(bool),
                        alive=z["alive"].astype(bool), pre=z["pre_pass"].astype(bool), F=z["F"].astype(np.float64), fell=z["fell"].astype(bool), tfall=z["t_fall_post"].astype(np.float64)))
    return out


def ema(x, tau):
    a = DT / tau; y = np.empty_like(x); y[0] = x[0]
    for k in range(1, x.shape[0]):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    return y


def phase_stats(runs, t_lo=3.5):
    """무외란 걷기(ge 1, 생존, t ≥ t_lo)의 위상 구간별 평균·표준편차."""
    acc = {k: [[] for _ in range(NB)] for k in ("th", "wy")}
    for r in runs:
        m = r["ge"] & r["alive"] & (r["t"][:, None] >= t_lo) & r["pre"][None, :]
        b = np.clip((r["u"] * NB).astype(int), 0, NB - 1)
        for k in acc:
            for i in range(NB):
                acc[k][i].append(r[k][m & (b == i)])
    st = {}
    for k in acc:
        mu = np.array([np.mean(np.concatenate(v)) if sum(len(x) for x in v) else 0.0 for v in acc[k]])
        sd = np.array([np.std(np.concatenate(v)) if sum(len(x) for x in v) else 1.0 for v in acc[k]])
        st[k] = (mu, np.maximum(sd, 1e-3))
    return st


def signals(r, st):
    """(T, N) 신호 dict — 양(+) = 앞, 음(−) = 뒤."""
    th, ph, wy, u = r["th"], r["ph"], r["wy"], r["u"]
    b = np.clip((u * NB).astype(int), 0, NB - 1)
    th_ref = ema(th, 2.0); ph_ref = ema(ph, 2.0)
    devA = np.hypot(th - th_ref, ph - ph_ref) * np.sign(th - th_ref)
    wy_f = ema(wy, DT / 0.33)
    C = th - st["th"][0][b]; D = wy_f - st["wy"][0][b]
    E = C / st["th"][1][b] + D / st["wy"][1][b]
    return {"A": devA, "B": th - th_ref, "C": C, "D": D, "E": E}


def first_cross(sig, thr, hold, sign):
    """(T, N): sign·sig > thr 가 hold 틱 연속인 첫 틱 색인(없으면 −1)을 열마다."""
    x = (sign * sig) > thr
    run = np.zeros(sig.shape[1], dtype=int); out = np.full(sig.shape[1], -1)
    for k in range(sig.shape[0]):
        run = np.where(x[k], run + 1, 0)
        hit = (run >= hold) & (out < 0); out[hit] = k
    return out


def count_cross(sig, thr, hold, sign, mask, refr=50):
    """마스크 안에서의 발동 횟수(불응 refr 틱) — 열 합."""
    x = ((sign * sig) > thr) & mask
    run = np.zeros(sig.shape[1], dtype=int); last = np.full(sig.shape[1], -10**9); n = 0
    for k in range(sig.shape[0]):
        run = np.where(x[k], run + 1, 0)
        hit = (run >= hold) & (k - last >= refr); n += int(hit.sum()); last[hit] = k
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.4); ap.add_argument("--far", type=float, default=0.2)
    ap.add_argument("--hold", type=int, default=2); ap.add_argument("--t_on", type=float, default=3.5); ap.add_argument("--out", default=os.path.join(REPO, "reports", "gen_w3", "q6_detect_result.md"))
    a = ap.parse_args()
    nd = load("nd")
    if not nd:
        print("추적 파일 없음"); return
    train = [r for r in nd if r["seed"] == 1] or nd[:1]; test_nd = [r for r in nd if r["seed"] != 1] or nd
    st = phase_stats(train, a.t_on)
    D = {t: load(t) for t in ("b18", "b12", "f12", "f18")}
    L = []; P = lambda x="": (L.append(x), print(x))
    P(f"# Q6c 감지기 오프라인 선택 (지연 예산 {a.budget} s, 오경보 상한 {a.far} 회/env·분, 연속 {a.hold} 틱)")
    P(f"위상 통계 학습 = nd seed {[r['seed'] for r in train]}, 오경보 평가 = nd seed {[r['seed'] for r in test_nd]} + 외란 실행의 외란 전 구간")
    P(f"무외란 걷기 관측 피치 위상별 표준편차 중앙 {np.median(st['th'][1]):.2f}° · 평균의 위상 진폭 {np.ptp(st['th'][0]):.2f}° · ω_y 표준편차 중앙 {np.median(st['wy'][1]):.3f} rad/s")
    best = None
    P("\n| 후보 | 문턱 | 오경보 '뒤' [회/env·분] | b18 검출률(예산 안) | b18 검출 지연 중앙 s | b12 검출률 | f12 '뒤' 오판 | f18 '뒤' 오판 | b18 검출 env 중 낙상 env 비율 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for name in ("A", "B", "C", "D", "E"):
        # 문턱 후보: 무외란 평가 신호의 음(−) 쪽 분위수에서
        neg = []
        for r in test_nd:
            s = signals(r, st)[name]; m = r["ge"] & r["alive"] & (r["t"][:, None] >= a.t_on) & r["pre"][None, :]
            neg.append(-s[m])
        neg = np.concatenate(neg)
        cands = sorted(set(np.round(np.percentile(neg, [99, 99.5, 99.8, 99.9, 99.95, 99.99, 100]) * np.array([1, 1, 1, 1, 1, 1, 1.1]), 3)))
        rows = []
        for thr in cands:
            # 오경보
            nfa = 0; env_min = 0.0
            for r in test_nd + [x for v in D.values() for x in v]:
                s = signals(r, st)[name]; pre_push = np.ones_like(r["t"], dtype=bool) if r["F"].max() == 0 else (r["t"] < r["t"][np.argmax(r["F"] > 0)] - 0.02)
                m = r["ge"] & r["alive"] & (r["t"][:, None] >= a.t_on) & pre_push[:, None] & r["pre"][None, :]
                nfa += count_cross(s, thr, a.hold, -1.0, m); env_min += m.sum() * DT / 60.0
            far = nfa / max(env_min, 1e-9)
            res = {}
            for tag, runs in D.items():
                det = 0; n = 0; lat = []; det_fell = 0; fell_n = 0
                for r in runs:
                    k0 = int(np.argmax(r["F"] > 0)); kb = int(round(a.budget / DT))
                    s = signals(r, st)[name][k0:k0 + kb + 1]
                    ok = r["pre"] & r["alive"][k0] & r["ge"][k0]
                    fc = first_cross(s, thr, a.hold, -1.0)
                    hit = (fc >= 0) & ok; det += int(hit.sum()); n += int(ok.sum()); lat += list(fc[hit] * DT)
                    det_fell += int((hit & r["fell"]).sum()); fell_n += int((ok & r["fell"]).sum())
                res[tag] = (det / max(n, 1), float(np.median(lat)) if lat else float("nan"), n, det_fell, fell_n)
            rows.append((thr, far, res))
        # 선택 규칙: 오경보 ≤ far ∧ 앞 밀기 '뒤' 오판 ≤ 5 % 인 문턱 중 b18 검출 최대
        ok_rows = [x for x in rows if x[1] <= a.far and max(x[2].get("f12", (0,))[0], x[2].get("f18", (0,))[0]) <= 0.05]
        pick = max(ok_rows, key=lambda x: x[2].get("b18", (0,))[0]) if ok_rows else None
        for thr, far, res in rows:
            mark = " ←" if pick is not None and thr == pick[0] else ""
            g = lambda t: res.get(t, (float("nan"), float("nan"), 0, 0, 0))
            P(f"| {name}{mark} | {thr:.3f} | {far:.3f} | {100*g('b18')[0]:.0f} % (n {g('b18')[2]}) | {g('b18')[1]:.2f} | {100*g('b12')[0]:.0f} % | {100*g('f12')[0]:.0f} % | {100*g('f18')[0]:.0f} % | {g('b18')[3]}/{g('b18')[4]} |")
        if pick is not None and (best is None or pick[2]["b18"][0] > best[1][2]["b18"][0]):
            best = (name, pick)
    P("")
    if best is None:
        P("선택 가능한 후보 없음(오경보·앞 오판 조건을 만족하는 문턱 없음) → 관측 기반 감지 불가로 기록")
    else:
        nm, (thr, far, res) = best
        P(f"**선택: 후보 {nm}, 문턱 {thr:.3f}** — 오경보 {far:.3f} 회/env·분, b18 검출 {100*res['b18'][0]:.0f} %(지연 중앙 {res['b18'][1]:.2f} s), b12 {100*res['b12'][0]:.0f} %, 앞 오판 f12 {100*res['f12'][0]:.0f} %·f18 {100*res['f18'][0]:.0f} %")
        P("검출률 < 50 % 면 사전 규칙상 Q6d 는 하지 않는다." if res["b18"][0] < 0.5 else "검출률 ≥ 50 % → Q6d(폐루프) 진행 대상.")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
