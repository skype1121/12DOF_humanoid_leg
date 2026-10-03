"""1003 교대 진단(G12/ema0 ↔ G21a/native EMA) 블랙박스 자동 분석 — 승격 판정 아님, 비교만.

입력: 브리지 블랙박스 rl_bridge_*.jsonl (시도 하나 = 파일 하나, 또는 --trial 로 같은 파일 안 시각 구간 지정).
출력: 시도별 표(markdown·csv) + 밀기 시각 기준 정렬 그림.

지표 (전부 밀기 시작 t0 기준 시간 정렬):
  0. 무외란 서기(정책 정착 후 ~ t0−0.5 s): 피치각속도 2–4.5 Hz 대역 RMS, 앞/뒤 절반 비(즉시 성장 검출), 피치 표준편차.
  ① 3 Hz: 피치각속도 2–4.5 Hz 대역 포락선(힐베르트) 최대·성장률 λ[1/s](log 포락선 직선 맞춤, t0+0.2 ~ 유효 끝), 우세 주파수.
  ② 발 사슬(패드 **접촉 비트** 기준 — 하중 해석 아님, 영상 확인 필요): 발별 이륙(1→0, ≥0.1 s 유지) → 착지(0→1) →
     하중 수용 대리지표(착지 발 접촉 ≥0.3 s 유지 + 그 시점 양발 접촉) 시각.
  ③ 정착 시간: |ωx|,|ωy| < 0.15 rad/s 이고 |피치 − 밀기 전 기준| < 1.5° 가 1.0 s 연속 성립하는 첫 시각(t0 기준, 착지 기준).
  유효 끝(낙상·회수 등): FREE_STAND 이탈, 양발 무접촉 ≥0.2 s, stop_reason/osc_tripped, |Δ피치| > 15° 중 첫 시각.

밀기 시작 t0 우선순위: ① --push 파일:초(영상 손 접촉 시각을 블랙박스 시각으로 옮긴 값) ② --timeline 의 콘솔 '밀기 기록'(op_push)
버튼 시각(같은 젯슨 벽시계) ③ 자동(신뢰 낮음 — 1002 실측: 조용한 서기 피치 흔들림 p2p 최대 ~2° 가 약한 밀기(+1.2°)와 같은 크기라
피치만으로는 구분 불가, 1002 자체 점검에서 44.1 s 대신 10.4 s 를 잡음). 표의 t0원천 열로 구분한다.
밀기 세기 대리지표: t0 뒤 0.5 s 안 최대 |Δ피치| (힘 대신 IMU 응답 크기 — 같은 약한 밀기인지 교대 시도 간 확인용).
축 규약: rp_deg = [롤, 피치], gyro_rad_s[1] = 피치각속도 (1002 보고서와 같음). 시각 = 파일 첫 기록 기준 초.

사용:
  python3 reports/real_1003/ab_analyze.py ~/logs/rl_bridge_A.jsonl ~/logs/rl_bridge_B.jsonl ... --out reports/real_1003/ab
  python3 reports/real_1003/ab_analyze.py reports/real_1002/rl_bridge_20261002_143511.jsonl --label 1002_g12_v2   (자체 점검)
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

DT = 0.02
BAND = (2.0, 4.5)


def load(fn):
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    ev = [r for r in rows if "event" in r]
    rec = [r for r in rows if "event" not in r and "rp_deg" in r]
    if not rec:
        raise SystemExit(f"{fn}: 텔레메트리 기록 없음")
    t00 = rec[0]["timestamp"]
    t = np.array([r["timestamp"] - t00 for r in rec])
    g = lambda k, n: np.array([r.get(k) if r.get(k) is not None else [np.nan] * n for r in rec], dtype=float)
    d = dict(t=t, rp=g("rp_deg", 2), gyro=g("gyro_rad_s", 3), contact=g("contact", 2),
             blend=np.array([r.get("blend_alpha") or 0.0 for r in rec], dtype=float),
             state=[r.get("hang_state") for r in rec], live=np.array([bool(r.get("live")) for r in rec]),
             stop=[r.get("stop_reason") or (("osc:" + str(r["osc_tripped"])) if r.get("osc_tripped") else None) for r in rec],
             ckpt=sorted({r.get("checkpoint") for r in rec}), mode=sorted({str(r.get("jvel_ema_mode")) for r in rec}),
             filt=sorted({str(r.get("obs_filter", "기록 없음(1003 이전 브리지)")) for r in rec}),
             events=[(e.get("event"), e["timestamp"] - t00) for e in ev], t00=t00)
    return d


def bandpass(x):
    b, a = butter(2, [BAND[0] / (0.5 / DT), BAND[1] / (0.5 / DT)], btype="band")
    return filtfilt(b, a, x)


def policy_start(d):
    on = np.where((d["blend"] >= 0.999) & d["live"])[0]
    return d["t"][on[0]] if len(on) else None


def detect_push(d, ts):
    t, p = d["t"], d["rp"][:, 1]
    i0 = np.searchsorted(t, ts + 3.0)
    run = 0
    for i in range(max(i0, 100), len(t)):
        base = np.nanmedian(p[i - 100:i])
        run = run + 1 if abs(p[i] - base) > 0.4 else 0
        if run >= 3:
            j = i - 2
            while j > 0 and abs(p[j] - base) >= 0.1:
                j -= 1
            return t[j]
    return None


def valid_end(d, t0, base_pitch):
    t = d["t"]
    both0 = (d["contact"][:, 0] < 0.5) & (d["contact"][:, 1] < 0.5)
    for i in np.where(t >= t0)[0]:
        why = None
        if d["state"][i] not in (None, "FREE_STAND", "CONTACT_ACQUIRED"):
            why = f"상태 {d['state'][i]}"
        elif d["stop"][i]:
            why = f"정지 {d['stop'][i]}"
        elif abs(d["rp"][i, 1] - base_pitch) > 15:
            why = "|Δ피치|>15°"
        elif i >= 10 and both0[i - 9:i + 1].all():
            why = "양발 무접촉 ≥0.2 s"
        if why:
            return t[i], why
    return t[-1], "기록 끝"


def foot_chain(d, t0, t_end):
    t, c = d["t"], d["contact"]
    out = []
    for f, name in ((0, "왼"), (1, "오")):
        k = np.where((t >= t0) & (t <= t_end))[0]
        if not len(k):
            continue
        i = k[0]
        while i < k[-1]:
            if c[i, f] >= 0.5 and c[i + 1, f] < 0.5 and (c[i + 1:i + 6, f] < 0.5).all():
                to = t[i + 1]
                j = i + 1
                while j <= k[-1] and c[j, f] < 0.5:
                    j += 1
                td = t[j] if j <= k[-1] else None
                la = None
                if td is not None:
                    m = (t >= td) & (t < td + 0.3)
                    if td + 0.3 <= t_end and (c[m, f] >= 0.5).all() and (c[np.searchsorted(t, td + 0.3) - 1] >= 0.5).all():
                        la = td + 0.3
                out.append(dict(foot=name, takeoff=to - t0, touchdown=None if td is None else td - t0,
                                load=None if la is None else la - t0))
                i = j
            else:
                i += 1
    return sorted(out, key=lambda e: e["takeoff"])


def settle(d, t_from, t_end, base_pitch):
    t = d["t"]
    ok = ((np.abs(d["gyro"][:, 0]) < 0.15) & (np.abs(d["gyro"][:, 1]) < 0.15)
          & (np.abs(d["rp"][:, 1] - base_pitch) < 1.5))
    k = np.where((t >= t_from) & (t <= t_end))[0]
    need = int(round(1.0 / DT))
    run = 0
    for i in k:
        run = run + 1 if ok[i] else 0
        if run >= need:
            return t[i - need + 1]
    return None


def analyze(fn, label, push=None, t_lo=None, t_hi=None):
    d = load(fn)
    if t_lo is not None or t_hi is not None:   # 같은 파일 안 시도 구간
        m = (d["t"] >= (t_lo or -1e9)) & (d["t"] <= (t_hi or 1e9))
        for k in ("t", "rp", "gyro", "contact", "blend", "live"):
            d[k] = d[k][m]
        d["state"] = [s for s, keep in zip(d["state"], m) if keep]; d["stop"] = [s for s, keep in zip(d["stop"], m) if keep]
    t = d["t"]
    ts = policy_start(d)
    R = dict(label=label, file=os.path.basename(fn), ckpt="/".join(map(str, d["ckpt"])), jvel_ema_mode="/".join(d["mode"]),
             obs_filter="/".join(d["filt"]), policy_start=ts)
    if ts is None:
        R["note"] = "정책 블렌딩 완료 구간 없음"
        return R, d, None
    t0 = None if push is not None else detect_push(d, ts)
    src = push[1] if push is not None else ("자동(신뢰 낮음)" if t0 is not None else "없음")
    if push is not None:
        t0 = push[0]
    R["push_t0"] = t0; R["push_src"] = src
    wy = d["gyro"][:, 1]
    wyb = bandpass(np.nan_to_num(wy))
    env = np.abs(hilbert(wyb))
    # 0. 무외란 서기
    q_hi = (t0 - 0.5) if t0 is not None else t[-1]
    q = (t >= ts + 2.0) & (t < q_hi)
    if q.sum() > 100:
        h = np.where(q)[0]; a, b = h[: len(h) // 2], h[len(h) // 2:]
        rms = lambda ix: float(np.sqrt(np.mean(wyb[ix] ** 2)))
        R.update(quiet_s=float(q.sum() * DT), quiet_band_rms=rms(h), quiet_growth_ratio=rms(b) / max(rms(a), 1e-6),
                 quiet_pitch_std=float(np.nanstd(d["rp"][q, 1])))
    if t0 is None:
        R["note"] = "밀기 미검출 — --push 로 지정"
        return R, d, None
    base_pitch = float(np.nanmedian(d["rp"][(t >= t0 - 2.0) & (t < t0), 1]))
    t_end, why = valid_end(d, t0, base_pitch)
    R.update(base_pitch=base_pitch, valid_end=t_end - t0, end_reason=why)
    w = (t >= t0) & (t <= t_end)
    R["peak_dpitch"] = float(np.nanmax(np.abs(d["rp"][w, 1] - base_pitch))) if w.any() else None
    w5 = (t >= t0) & (t <= min(t0 + 0.5, t_end))
    R["push_dpitch_05"] = float(np.nanmax(np.abs(d["rp"][w5, 1] - base_pitch))) if w5.any() else None
    # ① 3 Hz
    fw = (t >= t0 + 0.2) & (t <= t_end)
    R["band_env_max"] = float(env[w].max()) if w.any() else None
    if fw.sum() >= 15:
        tt, ee = t[fw] - t0, np.log(np.maximum(env[fw], 1e-4))
        R["band_growth_per_s"] = float(np.polyfit(tt, ee, 1)[0])
        seg = wy[fw] - np.nanmean(wy[fw]); F = np.abs(np.fft.rfft(np.nan_to_num(seg) * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), DT); m = (f >= 1.0) & (f <= 8.0)
        R["dom_freq_hz"] = float(f[m][np.argmax(F[m])]) if m.any() else None
    # ② 발 사슬
    chain = foot_chain(d, t0, t_end)
    R["steps"] = len(chain)
    if chain:
        c0 = chain[0]
        R.update(first_foot=c0["foot"], takeoff=c0["takeoff"], touchdown=c0["touchdown"], load_accept=c0["load"])
    # ③ 정착
    # 정착 탐색은 마지막 착지 뒤(스텝이 있으면) 또는 t0+0.2 부터 — 착지 전 정착 판정 방지
    last_td = max([c["touchdown"] for c in chain if c["touchdown"] is not None], default=None)
    st = settle(d, t0 + max(0.2, last_td or 0.0), t_end, base_pitch)
    R["settle_from_push"] = None if st is None else st - t0
    R["settle_from_touchdown"] = None if (st is None or last_td is None) else st - t0 - last_td
    return R, d, dict(t0=t0, env=env, t_end=t_end, base=base_pitch, chain=chain)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v).replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", nargs="*", default=None, help="파일 순서대로 라벨")
    ap.add_argument("--push", action="append", default=[], help="파일이름:초 (밀기 시작 지정, 파일 첫 기록 기준)")
    ap.add_argument("--timeline", action="append", default=[], help="세션 timeline.jsonl — 콘솔 '밀기 기록'(op_push) 시각 사용")
    ap.add_argument("--trial", action="append", default=[], help="파일이름:시작초:끝초 (한 파일 안 여러 시도)")
    ap.add_argument("--out", default=None, help="출력 접두어 (…_table.md / .csv / .png)")
    a = ap.parse_args()
    push = {p.rsplit(":", 1)[0]: (float(p.rsplit(":", 1)[1]), "지정(영상 등)") for p in a.push}
    marks = []                                   # 콘솔 op_push 벽시계 시각
    for tl in a.timeline:
        for l in open(tl):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("event") == "op_push" or r.get("action") == "op_push" and r.get("event") in (None, "request"):
                if r.get("event") == "op_push" and r.get("ts"):
                    marks.append(float(r["ts"]))
    marks = sorted(set(marks))
    jobs = []
    for i, fn in enumerate(a.files):
        lab = a.label[i] if a.label and i < len(a.label) else os.path.basename(fn)
        tr = [x.split(":") for x in a.trial if x.split(":")[0] == os.path.basename(fn)]
        if tr:
            for k, (_, lo, hi) in enumerate(tr):
                jobs.append((fn, f"{lab}#{k + 1}", float(lo), float(hi)))
        else:
            jobs.append((fn, lab, None, None))
    res = []
    for fn, lab, lo, hi in jobs:
        pk = push.get(os.path.basename(fn))
        if pk is None and marks:
            first = next(json.loads(l) for l in open(fn) if l.strip() and "event" not in json.loads(l))
            t00 = first["timestamp"]; last = json.loads(open(fn).readlines()[-1])["timestamp"]
            mk = [m - t00 for m in marks if t00 <= m <= last and (lo is None or lo <= m - t00 <= hi)]
            if mk:
                pk = (mk[0], "콘솔 밀기 기록" + (f"(같은 구간 표시 {len(mk)}개 — 첫 번째)" if len(mk) > 1 else ""))
        if pk is not None and lo is not None and not (lo <= pk[0] <= hi):
            pk = None
        res.append(analyze(fn, lab, pk, lo, hi))
    cols = [("label", "시도"), ("ckpt", "정책"), ("jvel_ema_mode", "jvel_ema"), ("obs_filter", "관측필터"), ("push_t0", "t0[s]"),
            ("push_src", "t0원천"), ("quiet_band_rms", "무외란 3Hz RMS"), ("quiet_growth_ratio", "무외란 후/전"),
            ("band_env_max", "3Hz 포락 최대"), ("band_growth_per_s", "성장률 λ[1/s]"), ("dom_freq_hz", "우세 Hz"),
            ("push_dpitch_05", "밀기 0.5s Δ피치°"), ("peak_dpitch", "최대 Δ피치°"), ("first_foot", "첫 발"), ("takeoff", "이륙"), ("touchdown", "착지"),
            ("load_accept", "하중수용(대리)"), ("steps", "스텝 수"), ("settle_from_push", "정착(밀기 후)"),
            ("settle_from_touchdown", "정착(착지 후)"), ("valid_end", "유효 끝"), ("end_reason", "끝 사유")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for R, _, _ in res:
        lines.append("| " + " | ".join(fmt(R.get(k)) for k, _ in cols) + " |")
    notes = [f"- {R['label']}: {R['note']}" for R, _, _ in res if R.get("note")]
    txt = "\n".join(lines + ([""] + notes if notes else []) + ["",
          "시각은 밀기 시작 t0 기준 초. 발 사슬은 패드 접촉 비트 기준(하중 해석 아님, 영상으로 확인). 3 Hz = 피치각속도 2–4.5 Hz 대역 [rad/s]."])
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out + "_table.md", "w").write(txt + "\n")
        with open(a.out + "_table.csv", "w", newline="") as F:
            w = csv.writer(F); w.writerow([k for k, _ in cols])
            for R, _, _ in res:
                w.writerow([R.get(k) for k, _ in cols])
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib import font_manager as _fm
        _ko = [f.name for f in _fm.fontManager.ttflist if any(k in f.name for k in ("Nanum", "Noto Sans CJK", "Noto Sans KR", "UnDotum"))]
        if _ko:
            plt.rcParams["font.family"] = _ko[0]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for R, d, X in res:
            if X is None:
                continue
            tt = d["t"] - X["t0"]; m = (tt >= -3) & (tt <= 6)
            ls = "-" if "21a" in R["ckpt"] else "--"
            ax[0].plot(tt[m], d["rp"][m, 1] - X["base"], ls, label=R["label"])
            ax[1].plot(tt[m], d["gyro"][m, 1], ls, color=ax[0].lines[-1].get_color())
            ax[2].plot(tt[m], X["env"][m], ls, color=ax[0].lines[-1].get_color())
            col = ax[0].lines[-1].get_color(); off = 0.03 * len(ax[3].lines)
            ax[3].plot(tt[m], d["contact"][m, 0] + off, ls, color=col)
            ax[3].plot(tt[m], -1.5 + d["contact"][m, 1] + off, ls, color=col)
            for a_ in ax:
                a_.axvline(X["t_end"] - X["t0"], color=ax[0].lines[-1].get_color(), lw=0.6, alpha=0.5)
        ax[0].set_ylabel("Δ피치 [°]"); ax[1].set_ylabel("ωy [rad/s]"); ax[2].set_ylabel("3 Hz 포락 [rad/s]")
        ax[3].set_ylabel("접촉 왼(위)/오(아래)"); ax[3].set_xlabel("밀기 t0 기준 [s]  (실선 G21a, 점선 G12, 세로선 = 유효 끝)")
        for a_ in ax:
            a_.axvline(0, color="k", lw=0.8); a_.grid(alpha=0.3)
        ax[0].legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(a.out + "_aligned.png", dpi=110)
        print(f"저장: {a.out}_table.md / .csv / _aligned.png")


if __name__ == "__main__":
    main()
