"""C2CAL 분석(CPU) — eval_ground_recovery --c2 --c2_trace 궤적으로 정지 문턱을 사전 규칙대로 확정(reports/gen_w3/C2_DESIGN.md §4).
사용: python3 c2_cal.py <태그접두> <seed> [win_s=0.45]
입력: log_picture/c2trace_<접두>_{ndmarch,nddeploy,sf15ns,fi40ns,b18ns}_s<seed>.npz
규칙(사전): stop_w = (제자리 걸음 창 max|ω_xy|) p90 × 1.2, stop_tilt = (창 max 기울기) p90 + 1°.
  판정 A: 외란 뒤 회복(REC 이후) 틱 중 참 |v| > 0.15 인 틱의 ≥ 80 % 가 '정숙 아님'(창 max ≥ 문턱)이어야 IMU 만으로 운동량 소멸을 가름 → 아니면 c2v 주 경로.
  판정 B: 무외란 STAND(deploy) 켜짐 env ≤ 1 %. 참고: 제자리 걸음 중 감지 조건 발화 env 비율(무장 없는 배포에서 걸음 중 재켜짐 위험)."""
import sys, os
import numpy as np
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
pfx, sd = sys.argv[1], sys.argv[2]
WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
DT = 0.02; K = max(1, int(round(WIN / DT))); SKIP = 1.5      # 인계 블렌딩 과도 제외 [s]
ON_TILT, ON_W, DELAY = 4.0, 0.5, 2


def load(nm):
    p = f"{LP}/c2trace_{pfx}_{nm}_s{sd}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def winmax(x):          # (T,N) → 최근 K 틱 최대(앞쪽은 가용 틱만)
    T = x.shape[0]; out = np.empty_like(x)
    for t in range(T):
        out[t] = x[max(0, t - K + 1):t + 1].max(0)
    return out


def q(a, ps=(50, 90, 99)):
    a = a[np.isfinite(a)]
    return tuple(float(np.percentile(a, p)) for p in ps) if a.size else (np.nan,) * len(ps)


L = [f"C2CAL · {pfx} seed {sd} · 창 {WIN}s(K {K}) · 인계 뒤 {SKIP}s 제외"]
M = load("ndmarch")
if M is None:
    sys.exit("ndmarch 궤적 없음")
t = M["t_rel"]; post = t >= SKIP
alive = M["alive"].astype(bool) & M["pre_pass"].astype(bool)[None, :]
wm, tm = winmax(M["w"]), winmax(M["tilt"])
sel = post[:, None] & alive
pw, pt = q(wm[sel]), q(tm[sel])
stop_w = pw[1] * 1.2; stop_tilt = pt[1] + 1.0
L.append(f"(a) 무외란 제자리 걸음(ge 1 고정): 창 max|ω_xy| p50/p90/p99 {pw[0]:.3f}/{pw[1]:.3f}/{pw[2]:.3f} rad/s · 창 max 기울기 {pt[0]:.2f}/{pt[1]:.2f}/{pt[2]:.2f}° "
         f"· 순간 |v| p50/p90 {q(M['v'][sel])[0]:.3f}/{q(M['v'][sel])[1]:.3f} m/s · 생존 env {int(alive[-1].sum())}/{int(M['pre_pass'].sum())}")
L.append(f"    → 규칙 문턱 stop_w = {stop_w:.3f} rad/s · stop_tilt = {stop_tilt:.2f}° (현 가정값 0.6 · 4.0)")
det = (M["tilt"] > ON_TILT) | (M["w"] > ON_W)
L.append(f"    참고: 제자리 걸음 중 감지 조건(기울기>{ON_TILT}°|ω|>{ON_W}) 발화 env {100*np.mean((det & sel).any(0)[M['pre_pass'].astype(bool)]):.1f} % · 발화 틱 {100*det[sel].mean():.2f} %")
for _nmD, _labD in (("nddeploy", "절대 기울기"), ("nddeprel", "상대 기울기 τ10 s(사후 후보)")):
    D = load(_nmD)
    if D is None:
        continue
    a2 = D["alive"].astype(bool) & D["pre_pass"].astype(bool)[None, :]
    on = (D["mode"] >= 1) & a2 & (D["t_rel"] >= 0)[:, None]
    fr = 100 * np.mean(on.any(0)[D["pre_pass"].astype(bool)])
    detd = ((D["tilt"] > ON_TILT) | (D["w"] > ON_W)) & a2
    L.append(f"(c) 무외란 STAND(deploy, {_labD}): 켜짐 env {fr:.1f} % (기준 ≤ 1 %) → 판정 B {'PASS' if fr <= 1.0 else 'FAIL'} · 감지 발화 틱 {100*detd[a2].mean():.2f} % · 기울기 p50/p90/p99 {q(D['tilt'][a2])[0]:.2f}/{q(D['tilt'][a2])[1]:.2f}/{q(D['tilt'][a2])[2]:.2f}°")
rows = []
for nm in ("sf15ns", "fi40ns", "b18ns"):
    R = load(nm)
    if R is None:
        L.append(f"(b) {nm}: 궤적 없음"); continue
    a3 = R["alive"].astype(bool) & R["pre_pass"].astype(bool)[None, :]
    rec = np.maximum.accumulate((R["mode"] >= 1), axis=0) & a3            # 첫 켜짐 이후
    wm3, tm3 = winmax(R["w"]), winmax(R["tilt"])
    v = R["v"]
    for (sw, stl, lab) in ((stop_w, stop_tilt, "규칙"), (0.6, 4.0, "가정")):
        notcalm = (wm3 >= sw) | (tm3 >= stl)
        fast = rec & (v > 0.15); slow = rec & (v < 0.05)
        hit = notcalm[fast].mean() if fast.any() else np.nan
        tn = (~notcalm[slow]).mean() if slow.any() else np.nan
        rows.append((nm, lab, hit))
        L.append(f"(b) {nm} [{lab} 문턱 ω {sw:.3f}·기울기 {stl:.2f}]: |v|>0.15 틱 중 '정숙 아님' {100*hit:.1f} % (≥80 % 필요) · |v|<0.05 틱 중 '정숙' {100*tn:.1f} % "
                 f"· 회복 틱 {int(rec.sum())} (빠름 {int(fast.sum())}) · 켜진 env {int(rec.any(0).sum())}/{int(R['pre_pass'].sum())}")
    # 켜짐 → 규칙 문턱 첫 정숙 / → 참 |v|<0.05 첫 도달 시각(env 중앙)
    first = lambda m: np.where(m.any(0), m.argmax(0), -1)
    t_on = first(R["mode"] >= 1)
    calm = ~((wm3 >= stop_w) | (tm3 >= stop_tilt)); still = v < 0.05
    dts = []
    for e in range(calm.shape[1]):
        if t_on[e] < 0 or not R["pre_pass"][e]:
            continue
        c1 = np.where(calm[t_on[e] + 23:, e])[0]; c2 = np.where(still[t_on[e] + 23:, e])[0]
        if c1.size and c2.size:
            dts.append((c1[0] - c2[0]) * DT)
    if dts:
        L.append(f"    켜짐+0.45 s 이후 첫 '정숙'(규칙) − 첫 |v|<0.05: 중앙 {np.median(dts):+.2f} s (음수 = IMU 정숙이 운동량 소멸보다 먼저 → 이른 정지 위험) · n {len(dts)}")
# ── 추가(사후, 1006): 한 보행 주기 이동평균 속도 |v̄_T| — 순간 |v| 는 걸음 흔들림(제자리 걸음 중앙 ~0.3 m/s)이 지배해 운동량 지표가 못 된다.
P_T = max(2, int(round((1.0 / 1.1) / DT)))
def vbox(R):
    if "vxy" not in R:
        return None
    v = R["vxy"].astype(np.float64); c = np.cumsum(v, axis=0); out = np.empty(v.shape[:2])
    for t in range(v.shape[0]):
        lo = max(0, t - P_T + 1); s_ = c[t] - (c[lo - 1] if lo > 0 else 0.0)
        out[t] = np.linalg.norm(s_ / (t - lo + 1), axis=-1)
    return out
vbM = vbox(M)
if vbM is None:
    L.append("(d) 주기 평균 속도: 궤적에 vxy 없음(구 궤적) — --c2_trace 재실행 필요")
else:
    pv = q(vbM[sel]); stop_vb = max(pv[1] * 1.2, 0.03)
    L.append(f"(d) [사후 지표] 한 주기({P_T} 틱) 이동평균 |v̄_T|: 제자리 걸음 p50/p90/p99 {pv[0]:.3f}/{pv[1]:.3f}/{pv[2]:.3f} m/s → 문턱(p90×1.2) {stop_vb:.3f}")
    for nm in ("sf15ns", "fi40ns", "b18ns"):
        R = load(nm)
        if R is None or "vxy" not in R:
            continue
        a3 = R["alive"].astype(bool) & R["pre_pass"].astype(bool)[None, :]
        rec = np.maximum.accumulate((R["mode"] >= 1), axis=0) & a3
        vb = vbox(R); hi = rec & (vb > 0.15)
        L.append(f"    {nm}: 회복 중 |v̄_T|>0.15 틱 {int(hi.sum())} 중 문턱 초과 {100*(vb[hi] >= stop_vb).mean() if hi.any() else float('nan'):.1f} % · 회복 틱 중 |v̄_T| p50/p90 {q(vb[rec])[0]:.3f}/{q(vb[rec])[1]:.3f}")
ok = [h for (_, lab, h) in rows if lab == "규칙" and np.isfinite(h)]
if ok:
    L.append(f"판정 A(규칙 문턱, 외란 조건 모두 ≥ 80 %): {'PASS' if min(ok) >= 0.8 else 'FAIL → c2v(|v| 추정 조건) 주 경로'} (최소 {100*min(ok):.1f} %)")
txt = "\n".join(L); print(txt)
open(f"{LP}/c2cal_{pfx}_s{sd}.txt", "w").write(txt + "\n")
