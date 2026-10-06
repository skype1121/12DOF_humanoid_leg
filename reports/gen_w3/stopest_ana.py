"""병목 2 오프라인 분석: 다리 운동학 오도메트리(rl_walking/deploy/leg_odometry.py) 정확도 + 정착 판정기 분류 정확도.

입력: stopest_rec.py 가 남긴 npz(R1 KE fi40 · R2 KE sf15→b18 · R3 W3S1c+c2 정지 안 함 b18→fi40).
조건(추정기 입력):
  C0 이상   : 링크 관절각(유격 없음)·참 자이로·참 중력·접촉 힘 > 5 N·지연 0
  C1 배포   : 엔코더(유격 클램프 + 기본자세 편향 = 실물 프레임)·정책 관측 자이로/중력(잡음 ±0.2·IMU 지연 0–3)·정책 접촉 0/1(v2)·추가 지연 2 틱
  C2 배포+  : C1 + 엔코더가 못 보는 무릎 하중 처짐(서기 5.5° 기준, 하중비 선형: 스윙 0·한발 11°) + 자이로 상수 편향 ±0.03 + 지연 4 틱
  C3 최악   : C2 의 무릎 처짐 11°(실측 서기 처짐 전량이 동적이라는 가정, 한발 22°)
참값: CoM 수평 속도(질량 가중)·골반 속도, 미래 1 s CoM 순변위(D1), 2 s 안 낙상.
판정 라벨(T2): '움직임' = 2 s 안 낙상 또는 D1 > 8 cm, '정착' = D1 < 3 cm 이고 2 s 생존. 사이 = 제외.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_walking.deploy.leg_odometry import LegOdometry, quat_to_R  # noqa: E402

DT = 0.02; W = 45


def yaw_of(quat):
    w, x, y, z = (quat[..., i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def trailing_mean(x, w=W):
    T = x.shape[0]
    cs = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


def shift(x, d):
    if d <= 0:
        return x
    y = np.empty_like(x); y[d:] = x[:-d]; y[:d] = x[:1]
    return y


def load(path):
    d = np.load(path, allow_pickle=False)
    m = json.loads(str(d["meta"]))
    D = {k: d[k] for k in d.files if k != "meta"}
    T, N = D["q"].shape[:2]
    term = D["term"].astype(bool)
    alive = np.cumsum(term, 0) == 0          # 이번 틱까지 종료 없음(종료 틱 이후는 자동 리셋 쓰레기)
    D["alive"] = alive & D["pre_pass"][None].astype(bool)
    D["pre_pass"] = D["pre_pass"].astype(bool)
    D["t_rel"] = (np.arange(T) - m["steps_pre"] + 1) * DT
    D["meta"] = m
    # 2 s 안 낙상: 종료 틱
    t_term = np.where(term.any(0), term.argmax(0), 10 ** 9)
    D["t_term"] = t_term
    return D


def inputs(D, cond, rng):
    m = D["meta"]; jn = m["jn"]; T, N = D["q"].shape[:2]
    if cond == "C0":
        q = D["q"]; gy = D["av"]; pg = D["pg"]; cf = D["fn"] > 5.0; dly = 0
    elif cond in ("C1e", "C1i", "C1c"):          # 오차원 분리: 한 가지만 배포 신호로(나머지 이상)
        q = D["enc"] - D["pbias"] if cond == "C1e" else D["q"]
        gy = D["o_gyro"] if cond == "C1i" else D["av"]; pg = D["o_pg"] if cond == "C1i" else D["pg"]
        cf = (D["o_cf"] > 0.5) if cond == "C1c" else (D["fn"] > 5.0); dly = 0
    else:
        q = D["enc"] - D["pbias"]; gy = D["o_gyro"].copy(); pg = D["o_pg"]; cf = D["o_cf"] > 0.5
        dly = {"C1": 2, "C2": 4, "C3": 4, "C1d0": 0, "C2k": 4, "C2k70": 4, "C2t": 4}[cond]
        if cond in ("C2", "C3", "C2k", "C2k70", "C2t"):
            dk = np.radians(11.0 if cond == "C3" else 5.5)
            fn = D["fn"]; lf = fn / np.maximum(fn.sum(-1, keepdims=True), 1.0)          # 하중비(0–1), 접촉 없음 0
            q = q.copy()
            iL, iR = jn.index("left_knee_joint"), jn.index("right_knee_joint")
            # 물리 무릎 = 엔코더 + 처짐(왼쪽 굴곡 −, 오른쪽 +; 공유 편향 부호 규약) → 엔코더 = 물리 − 처짐
            q[..., iL] = q[..., iL] + dk * 2 * lf[..., 0]
            q[..., iR] = q[..., iR] - dk * 2 * lf[..., 1]
            if cond in ("C2k", "C2k70"):
                # 보상(가설): 접촉 0/1 로 하중비 추정(한발 1·양발 0.5) × 처짐 계수(정확 / 70 %) — 실물 패드 0/1 만으로 가능한 형
                c = cf.astype(float); lf_hat = c / np.maximum(c.sum(-1, keepdims=True), 1.0)
                kk = dk * (1.0 if cond == "C2k" else 0.7)
                q[..., iL] = q[..., iL] - kk * 2 * lf_hat[..., 0]
                q[..., iR] = q[..., iR] + kk * 2 * lf_hat[..., 1]
            if cond == "C2t":
                # 보상(가설): 무릎 토크(모터 전류)로 처짐 추정 = 참 하중비(대리) + 잡음 σ 15 %
                lf_hat = np.clip(lf + rng.normal(0, 0.15, size=lf.shape), 0, 1)
                q[..., iL] = q[..., iL] - dk * 2 * lf_hat[..., 0]
                q[..., iR] = q[..., iR] + dk * 2 * lf_hat[..., 1]
            gy = gy + rng.uniform(-0.03, 0.03, size=(1, N, 3))
    return q, gy, pg, cf, dly


def run_odo(D, cond, rng):
    zg = None
    if cond.endswith("g"):          # 발 높이 게이트 변형(C1g·C2tg …): 같은 입력 + LegOdometry(z_gate=1.5 cm)
        cond, zg = cond[:-1], 0.015
    q, gy, pg, cf, dly = inputs(D, cond, rng)
    T, N = q.shape[:2]
    odo = LegOdometry(D["meta"]["jn"], n=N, z_gate=zg)
    V = np.zeros((T, N, 2)); VB = np.zeros((T, N, 2)); VAL = np.zeros((T, N), bool); FR = np.zeros((T, N)); YAW = np.zeros((T, N))
    for t in range(T):
        o = odo.step(q[t], gy[t], pg[t], cf[t])
        V[t] = o["v_inst"]; VB[t] = o["v_bar"]; VAL[t] = o["valid"]; FR[t] = o["frac"]; YAW[t] = odo.yaw
    return {"V": shift(V, dly), "VB": shift(VB, dly), "VAL": shift(VAL, dly), "FR": shift(FR, dly), "YAW": shift(YAW, dly), "dly": dly}


def truth(D, yaw_est):
    """참 속도를 추정기 프레임(요 오프셋 Δ = ψ_true − ψ̂)으로."""
    yt = yaw_of(D["quat"]); dl = yt - yaw_est
    c, s = np.cos(-dl), np.sin(-dl)
    def rot(v):
        return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
    vC = rot(D["comv"][..., :2]); vP = rot(D["lv"][..., :2])
    return vC, vP


def labels(D):
    T, N = D["q"].shape[:2]
    com = D["com"][..., :2]; alive = D["alive"]; tt = D["t_term"]
    H1, H2 = 50, 100
    D1 = np.full((T, N), np.nan)
    D1[:T - H1] = np.linalg.norm(com[H1:] - com[:-H1], axis=-1)
    idx = np.arange(T)[:, None]
    fall2 = (tt[None, :] > idx) & (tt[None, :] <= idx + H2)
    surv2 = (tt[None, :] > idx + H2) & (idx + H2 < T)
    alive_f = np.zeros((T, N), bool); alive_f[:T - H1] = alive[H1:]
    # 다음 외란이 2 s 안에 시작하는 틱은 라벨 제외(미래 외란이 만든 낙상·변위를 현재 상태로 오인하지 않게)
    tr = D["t_rel"]; clean = np.ones(T, bool)
    for (_, _, t0, _, _) in D["meta"]["dists"]:
        clean &= ~((t0 > tr) & (t0 <= tr + 2.0))
    clean = clean[:, None]
    moving = alive & clean & (fall2 | (alive_f & (D1 > 0.08)))
    settled = alive & clean & surv2 & (D1 < 0.03)
    return moving, settled, D1


def post_mask(D, t_lo=0.0, t_hi=6.0):
    """외란 시작 기준 [t_lo, t_hi) 창(외란별 합집합)."""
    tr = D["t_rel"]; m = np.zeros(tr.shape, bool)
    for (_, _, t0, _, _) in D["meta"]["dists"]:
        m |= (tr >= t0 + t_lo) & (tr < t0 + t_hi)
    return m


def pre_mask(D):
    tr = D["t_rel"]; t0 = min([x[2] for x in D["meta"]["dists"]] + [99.0])
    return (tr >= 1.5) & (tr < t0)


def first_true_after(cond, t_start_idx, hold):
    """cond (T,N) 가 hold 틱 연속 참이 되는 첫 틱(≥ t_start_idx). 없으면 −1."""
    T, N = cond.shape
    run = np.zeros(N, int); out = np.full(N, -1)
    for t in range(t_start_idx, T):
        run = np.where(cond[t], run + 1, 0)
        hit = (run >= hold) & (out < 0)
        out[hit] = t
    return out


def main(paths, out_txt):
    rng = np.random.default_rng(0)
    L = []
    P = lambda s="": (L.append(s), print(s, flush=True))
    P("# 병목 2 — 다리 운동학 오도메트리·정착 판정 오프라인 표 (stopest_ana.py)")
    for path in paths:
        D = load(path); m = D["meta"]; T, N = D["q"].shape[:2]
        tag = os.path.basename(path)
        P(f"\n## {tag}: env {N}(사전검사 통과 {int(D['pre_pass'].sum())}) · 틱 {T} · 외란 {m['dists']} · c2={m['c2']}")
        P(f"정책 {m['ckpt']} {('→ ' + m['switch_to']) if m['switch_to'] else ''}")
        moving, settled, D1 = labels(D)
        pm = post_mask(D)[:, None] & D["alive"]; prm = pre_mask(D)[:, None] & D["alive"]
        P(f"라벨(외란 뒤 0–6 s, 생존 틱): 움직임 {int((moving & pm).sum())} · 정착 {int((settled & pm).sum())} · 제외 {int((pm & ~moving & ~settled).sum())}")
        res = {}
        for cond in ("C0", "C1e", "C1i", "C1c", "C1cg", "C1d0", "C1", "C1g", "C2", "C2g", "C2k", "C2k70", "C2t", "C2tg", "C3", "C3g"):
            o = run_odo(D, cond, rng); res[cond] = o
            vC, vP = truth(D, o["YAW"])
            vCb = trailing_mean(vC); vPb = trailing_mean(vP)
            sb = np.linalg.norm(o["VB"], axis=-1); sCb = np.linalg.norm(vCb, axis=-1)
            for nm, mk in (("외란 뒤", pm), ("외란 전 서기", prm)):
                mk = mk & np.isfinite(sCb)
                if mk.sum() == 0:
                    continue
                ev = np.linalg.norm(o["VB"] - vCb, axis=-1)[mk]; es = (sb - sCb)[mk]
                evp = np.linalg.norm(o["VB"] - vPb, axis=-1)[mk]
                ei = np.linalg.norm(o["V"] - vP, axis=-1)[mk & o["VAL"]]
                P(f"  {cond} {nm}: |v̄ 추정−v̄_CoM| RMS {np.sqrt((ev**2).mean()):.3f} p90 {np.percentile(ev, 90):.3f} · 크기 편향 {es.mean():+.3f} · "
                  f"골반 v̄ 대비 RMS {np.sqrt((evp**2).mean()):.3f} · 순간 골반 RMS {np.sqrt((ei**2).mean()):.3f} · 참 |v̄_CoM| p50/p90 {np.percentile(sCb[mk],50):.3f}/{np.percentile(sCb[mk],90):.3f} · 추정 p50/p90 {np.percentile(sb[mk],50):.3f}/{np.percentile(sb[mk],90):.3f} · 유효 {o['VAL'][mk].mean():.2f}")
        # ── 틱 분류(외란 뒤 0–6 s): 정착 판정 = |v̄| < θ & 유효비 ≥ 0.5 ──
        P("  틱 분류(외란 뒤 0.9–6 s, T2 라벨): 오탐 = 움직임 중 '정착' 판정 비율 / 미탐 = 정착 중 '움직임' 판정 비율")
        pm9 = post_mask(D, 0.9, 6.0)[:, None] & D["alive"]
        mv = moving & pm9; st = settled & pm9
        for cond in ("C0", "C1", "C1g", "C2", "C2g", "C2t", "C2tg", "C3", "C3g"):
            o = res[cond]; sb = np.linalg.norm(o["VB"], axis=-1)
            row = []
            for th in (0.05, 0.07, 0.10):
                pr = (sb < th) & (o["FR"] >= 0.5)
                row.append(f"θ{th:.2f} 오탐 {100*pr[mv].mean():.1f}% 미탐 {100*(~pr[st]).mean():.1f}%")
            P(f"    (a) 오도메트리 {cond}: " + " · ".join(row))
        # IMU 문턱 기준선(KE 조기 복귀 규칙 형): 기울기 < 6° & |ω_xy| < 0.4 & 양발
        pg = D["o_pg"] / np.linalg.norm(D["o_pg"], axis=-1, keepdims=True); tilt = np.degrees(np.arccos(np.clip(-pg[..., 2], -1, 1)))
        wxy = np.linalg.norm(D["o_gyro"][..., :2], axis=-1); both = (D["o_cf"] > 0.5).all(-1)
        imu = (tilt < 6) & (wxy < 0.4) & both
        P(f"    IMU 순간 규칙(기울기<6°·|ω|<0.4·양발): 오탐 {100*imu[mv].mean():.1f}% 미탐 {100*(~imu[st]).mean():.1f}%")
        # (b) 접촉 패턴 + IMU 저주파: 양발 연속 ≥ 0.5 s & LPF(τ0.2)|ω_xy| < 0.3
        ds = np.zeros((T, N)); wl = np.zeros((T, N)); a = DT / 0.2
        for t in range(T):
            ds[t] = np.where(both[t], (ds[t - 1] if t else 0) + DT, 0.0)
            wl[t] = (wl[t - 1] if t else wxy[0]) + a * (wxy[t] - (wl[t - 1] if t else wxy[0]))
        for dsT, wT in ((0.5, 0.3), (0.3, 0.4), (0.8, 0.25)):
            jb = (ds >= dsT) & (wl < wT)
            P(f"    (b) 양발≥{dsT}s·LPF|ω|<{wT}: 오탐 {100*jb[mv].mean():.1f}% 미탐 {100*(~jb[st]).mean():.1f}%")
        for cond in ("C1", "C2"):
            o = res[cond]; sb = np.linalg.norm(o["VB"], axis=-1)
            jab = (sb < 0.07) & (o["FR"] >= 0.5) & (ds >= 0.3)
            P(f"    (a)+(b) {cond} θ0.07 & 양발≥0.3s: 오탐 {100*jab[mv].mean():.1f}% 미탐 {100*(~jab[st]).mean():.1f}%")
        # ── 사건 수준: 외란별 첫 '정착' 선언(외란 뒤 ≥ 0.9 s, 0.2 s 유지) ──
        P("  사건(외란별 첫 정착 선언 — 외란 뒤 ≥0.9 s·0.2 s 유지): 거짓정착 = 선언 순간 T2 '움직임' · 미선언 = 6 s 안 선언 없음 · 지연 = 선언 − 참 정착 시작(T2 정착 0.5 s 연속의 첫 틱)")
        tr = D["t_rel"]
        judges = {}
        for cond in ("C1", "C1g", "C2", "C2g", "C2t", "C2tg", "C3g"):
            o = res[cond]; sb = np.linalg.norm(o["VB"], axis=-1)
            for th in (0.05, 0.07):
                judges[f"(a){cond} θ{th:.2f}"] = (sb < th) & (o["FR"] >= 0.5)
        judges["(b) 양발≥0.5s·|ω|<0.3"] = (ds >= 0.5) & (wl < 0.3)
        judges["(a)C1 θ0.07 + 양발≥0.3s"] = (np.linalg.norm(res["C1"]["VB"], axis=-1) < 0.07) & (res["C1"]["FR"] >= 0.5) & (ds >= 0.3)
        mode = D["mode"].astype(bool)
        for (_, _, t0, _, _) in m["dists"]:
            i0 = int(np.argmin(np.abs(tr - t0))); i9 = i0 + 45; iend = min(T, i0 + 300)
            ok_env = D["alive"][i0]
            if ok_env.sum() == 0:
                continue
            # 참 정착 시작
            st_run = first_true_after(settled[:iend], i0, 25)
            fell6 = (D["t_term"] > i0) & (D["t_term"] < iend)
            P(f"   외란 t={t0:.0f}s: 대상 env {int(ok_env.sum())} · 6 s 안 낙상 {int((fell6 & ok_env).sum())} · 참 정착 도달 {int(((st_run >= 0) & ok_env).sum())} (중앙 {np.median((st_run[(st_run>=0)&ok_env]-i0))*DT if ((st_run>=0)&ok_env).any() else float('nan'):.2f}s)")
            items = list(judges.items())
            # KE/c2 실제 감독기: 외란 뒤 회복 모드가 꺼진 첫 틱(켜졌던 env)
            if mode[i0:iend].any():
                on_first = np.where(mode[i0:iend].any(0), mode[i0:iend].argmax(0) + i0, -1)
                off = np.full(N, -1)
                for e in np.where(on_first >= 0)[0]:
                    z = np.where(~mode[on_first[e]:iend, e])[0]
                    off[e] = on_first[e] + z[0] if len(z) else -1
                items.append(("실제 감독기(모드 꺼짐)", off))
            for nm, J in items:
                if isinstance(J, np.ndarray) and J.ndim == 2:
                    tdec = first_true_after(J[:iend], i9, 10)
                else:
                    tdec = J
                sel = ok_env.copy()
                dec = (tdec >= 0) & sel
                # 선언 순간이 생존 틱이어야 의미(이미 넘어졌으면 제외)
                dec_alive = dec & (tdec < D["t_term"])
                fs = np.zeros(N, bool); fs[dec_alive] = moving[tdec[dec_alive], np.where(dec_alive)[0]]
                surv_st = sel & (st_run >= 0)
                miss = surv_st & ~dec_alive
                lat = (tdec - st_run)[dec_alive & surv_st] * DT
                P(f"     {nm:28s}: 선언 {int(dec_alive.sum()):3d} · 거짓정착 {int(fs.sum()):3d} ({100*fs.sum()/max(1,dec_alive.sum()):.1f}%) · "
                  f"참정착 env 중 미선언 {int(miss.sum())}/{int(surv_st.sum())} · 지연 중앙 {np.median(lat) if len(lat) else float('nan'):+.2f}s p90 {np.percentile(lat,90) if len(lat) else float('nan'):+.2f}s")
    open(out_txt, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
