#!/usr/bin/env python3
"""1011 재학습 지시 7 의 보고 표: 정책(실행 태그)별 왕복 성공률 · 힙 외전(8번 = R_hip_a, 2번 = L_hip_a) 요구 최대/제한 뒤 토크 ·
20/23/25 N·m 초과 빈도·연속 지속 · 요구 변화율 · 좌우 발 접지력 · 무게중심 궤적 · 걷기 전환 토크 파형.
요구 토크 = kp·(ptg − q) − kd·qd (ptg = PD 가 받는 목표, kp 150·kd 5). 입력: transall_<tag>.npz(--trace_qd: qd·ptg·fz·com) + kwloop_<tag>.npz.
사용: report7.py <표시이름=tag> ...   (표준 출력 = 마크다운)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chk_load as C
KP, KD, W_N = 150.0, 5.0, 16.26 * 9.81
TH = (20.0, 23.0, 25.0)


def runs(mask):
    """True 연속 구간 길이(틱) 목록."""
    out, n = [], 0
    for v in mask:
        if v: n += 1
        elif n: out.append(n); n = 0
    if n: out.append(n)
    return out


def one(name, tag):
    D = C.load(tag); a = np.load(f"{C.LP}/transall_{tag}.npz", allow_pickle=True); k = np.load(f"{C.LP}/kwloop_{tag}.npz", allow_pickle=True)
    assert "ptg" in a.files, f"{tag}: ptg 없음"
    ptg = a["ptg"].astype(np.float32); q, qd, tq, alive = D["q"], D["qd"], D["tq"], D["alive"]; na = D["na"]
    dem = KP * (ptg - q) - KD * qd; kind = D["kind"]; pp = D["pp"]; W = C.windows(D)
    fell = k["fell"][:na] & pp; walk = (kind == 2) & alive & pp[None, :]
    ev = [e for e in range(na) if pp[e]]
    R = {"name": name, "tag": tag, "fall": f"{int(fell.sum())}/{int(pp.sum())}", "ok%": 100 * (1 - fell.sum() / max(pp.sum(), 1)),
         "n_w3": int(k["n_to_w3"][:na][pp].sum()), "n_ke": int(k["n_to_ke"][:na][pp].sum())}
    for j, nm in ((3, "R"), (2, "L")):
        d = np.abs(dem[..., j]); t = np.abs(tq[..., j])
        mx = np.array([d[walk[:, e], e].max() for e in ev if walk[:, e].any()]); tm = np.array([t[walk[:, e], e].max() for e in ev if walk[:, e].any()])
        R[nm + "_dem"] = tuple(np.percentile(mx, [50, 95, 100])); R[nm + "_app"] = tuple(np.percentile(tm, [50, 95, 100]))
        R[nm + "_steady95"] = float(np.percentile(d[walk], 95)); R[nm + "_rms"] = float(np.sqrt(np.mean(d[walk] ** 2)))
        for th in TH:
            R[f"{nm}_fr{th:.0f}"] = 100 * float((d[walk] > th).mean())
            ln = [r for e in ev for r in runs((d[:, e] > th) & walk[:, e])]
            R[f"{nm}_du{th:.0f}"] = (0.0, 0.0) if not ln else (20 * float(np.percentile(ln, 95)), 20 * float(max(ln)))
        dd = np.abs(np.diff(dem[..., j], axis=0)); R[nm + "_rate95"] = float(np.percentile(dd[walk[1:]], 95))
    A = C.J8; dall = np.abs(dem[..., A]).max(-1)
    R["any_fr25"] = 100 * float((dall[walk] > 25).mean()); R["sat%"] = 100 * float((np.abs(tq[..., A]) >= 0.95 * 25).any(-1)[walk].mean())
    if "fz" in a.files:
        fz = a["fz"].astype(np.float32); com = a["com"].astype(np.float32); flat = a["flat"].astype(np.float32)
        st = (kind != 2) & alive & pp[None, :] & (D["t"][:, None] > 0.5)
        R["fz_stand"] = tuple(float(fz[..., i][st].mean()) for i in (0, 1)); R["fz_walk"] = tuple(float(fz[..., i][walk].mean()) for i in (0, 1))
        R["fz_peak95"] = tuple(float(np.percentile(fz[..., i][walk], 99)) for i in (0, 1))
        cl = com[..., 1]; half = np.abs(flat[..., 0] - flat[..., 1]) / 2          # 발 간격의 절반
        amp = np.array([np.percentile(cl[walk[:, e], e], 97.5) - np.percentile(cl[walk[:, e], e], 2.5) for e in ev if walk[:, e].sum() > 50])
        R["com_pp_cm"] = 100 * float(np.median(amp)); R["half_w_cm"] = 100 * float(np.median(half[walk])); R["com_fwd_cm"] = 100 * float(np.median(com[..., 0][walk]))
        R["com_stand_lat_cm"] = 100 * float(np.median(cl[st]))
    rows = [(e, i0) for e, ws in W.items() for (n, i0, i1) in ws if n == "start_inpl"]
    K = 40
    R["wave"] = {"R": np.median([np.abs(dem[i0:i0 + K, e, 3]) for e, i0 in rows], 0), "L": np.median([np.abs(dem[i0:i0 + K, e, 2]) for e, i0 in rows], 0),
                 "Rapp": np.median([np.abs(tq[i0:i0 + K, e, 3]) for e, i0 in rows], 0),
                 "cL": np.mean([a["cont"][i0:i0 + K, e, 0] for e, i0 in rows], 0), "cR": np.mean([a["cont"][i0:i0 + K, e, 1] for e, i0 in rows], 0),
                 "R95": np.percentile([np.abs(dem[i0:i0 + K, e, 3]) for e, i0 in rows], 95, 0), "n": len(rows)}
    if "com" in a.files:
        R["wave"]["com"] = 100 * np.median([a["com"][i0:i0 + K, e, 1].astype(np.float32) for e, i0 in rows], 0)
    return R


RS = [one(*(x.split("=", 1) if "=" in x else (x, x))) for x in sys.argv[1:]]
f3 = lambda t: "%.0f / %.0f / %.0f" % t
print("### 1. 성공률과 힙 외전 요구 토크 (걷는 틱 전체, env 별 최대의 p50 / p95 / 최대, N·m)\n")
print("| 정책 | 왕복 낙상 | 성공 % | 걷기 전환 켬/끔 | 8번(R) 요구 | 8번 제한 뒤 | 2번(L) 요구 | 2번 제한 뒤 | 평상 요구 p95 R/L | 요구 RMS R/L | 어느 관절이든 25 초과 틱 % | 포화 틱 % |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for R in RS:
    print(f"| {R['name']} | {R['fall']} | {R['ok%']:.1f} | {R['n_w3']}/{R['n_ke']} | {f3(R['R_dem'])} | {f3(R['R_app'])} | {f3(R['L_dem'])} | {f3(R['L_app'])} | {R['R_steady95']:.0f} / {R['L_steady95']:.0f} | {R['R_rms']:.1f} / {R['L_rms']:.1f} | {R['any_fr25']:.1f} | {R['sat%']:.1f} |")
print("\n### 2. 20 / 23 / 25 N·m 초과 — 빈도(걷는 틱 %)와 연속 지속(p95 / 최대, ms), 요구 변화율\n")
print("| 정책 | 8번 >20 | >23 | >25 | 8번 지속 >20 | >23 | >25 | 2번 >20 | >23 | >25 | 2번 지속 >25 | 요구 변화율 p95 R/L (N·m/틱) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for R in RS:
    du = lambda nm, th: "%.0f / %.0f" % R[f"{nm}_du{th}"]
    print(f"| {R['name']} | {R['R_fr20']:.1f} | {R['R_fr23']:.1f} | {R['R_fr25']:.1f} | {du('R', 20)} | {du('R', 23)} | {du('R', 25)} | {R['L_fr20']:.1f} | {R['L_fr23']:.1f} | {R['L_fr25']:.1f} | {du('L', 25)} | {R['R_rate95']:.1f} / {R['L_rate95']:.1f} |")
if all("fz_walk" in R for R in RS):
    print(f"\n### 3. 좌우 발 접지력(수직, N — 몸무게 {W_N:.0f} N)과 무게중심(두 발 중점 기준, cm)\n")
    print("| 정책 | 서기 평균 왼/오른 | 걷기 평균 왼/오른 | 걷기 99 % 왼/오른 | 무게중심 좌우 왕복 폭 | 발 간격의 절반 | 좌우 폭 ÷ 발 간격 | 무게중심 앞뒤(중앙) | 서기 때 좌우 치우침 |\n|---|---|---|---|---|---|---|---|---|")
    for R in RS:
        print(f"| {R['name']} | {R['fz_stand'][0]:.0f} / {R['fz_stand'][1]:.0f} | {R['fz_walk'][0]:.0f} / {R['fz_walk'][1]:.0f} | {R['fz_peak95'][0]:.0f} / {R['fz_peak95'][1]:.0f} | {R['com_pp_cm']:.1f} | {R['half_w_cm']:.1f} | {R['com_pp_cm'] / (2 * R['half_w_cm']):.2f} | {R['com_fwd_cm']:+.1f} | {R['com_stand_lat_cm']:+.1f} |")
print("\n### 4. 걷기 전환 파형 (제자리 걸음 켠 뒤 틱 20 ms 마다, 중앙값)\n")
for R in RS:
    w = R["wave"]; ks = list(range(0, 40, 2))
    print(f"**{R['name']}** (표본 {w['n']})\n\n| 틱 | " + " | ".join(str(i) for i in ks) + " |\n|---|" + "---|" * len(ks))
    print("| 8번 요구 N·m | " + " | ".join(f"{w['R'][i]:.0f}" for i in ks) + " |")
    print("| 8번 요구 p95 | " + " | ".join(f"{w['R95'][i]:.0f}" for i in ks) + " |")
    print("| 8번 제한 뒤 | " + " | ".join(f"{w['Rapp'][i]:.0f}" for i in ks) + " |")
    print("| 2번 요구 N·m | " + " | ".join(f"{w['L'][i]:.0f}" for i in ks) + " |")
    print("| 왼발 접지 % | " + " | ".join(f"{100 * w['cL'][i]:.0f}" for i in ks) + " |")
    print("| 오른발 접지 % | " + " | ".join(f"{100 * w['cR'][i]:.0f}" for i in ks) + " |")
    if "com" in w:
        print("| 무게중심 왼쪽(+) cm | " + " | ".join(f"{w['com'][i]:+.1f}" for i in ks) + " |")
    print()
