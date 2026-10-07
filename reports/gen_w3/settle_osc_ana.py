"""STOPEST §13b: 진동 주입 시뮬(--osc_inj --trace_npz) 분석 — 판정기 팔별 끄기·조기 해제·흔들림 노출·재켜짐·낙상.

입력: log_picture/ground_recovery_<tag>_trace.npz (틱별 mode·w_true·w_det·alive, osc_amp/gap env 별) + _envs.csv(낙상·사전검사).
지표(사전검사 통과 env, 인계 뒤 t ≥ 1.5 s):
  끄기 수 · 조기 해제 = 끈 뒤 1 s 안 참 |ω_xy| ≥ 1.0(생존 중) · 흔들림 노출 = 첫 켜짐 뒤 서기 모드 & 감지 |ω_xy| > 0.5 시간 ·
  재켜짐 = 켜짐 2 회 이상 · 마지막 버스트 끝 대비 마지막 끄기 지연 · 낙상.
사용: python3 settle_osc_ana.py calib <tag>            — 보정: 토크 진폭 구간별 버스트 중 참 |ω| 최대(중앙)·낙상
      python3 settle_osc_ana.py cmp <접두> <조건> 팔... — 팔 비교(seed 전부 합), 짝 McNemar 는 팔 쌍마다
"""
import csv
import glob
import math
import os
import sys

import numpy as np

LP = "/home/ryu/humanoid_leg_test1/log_picture"


def load(tag):
    z = np.load(os.path.join(LP, f"ground_recovery_{tag}_trace.npz"))
    rows = list(csv.DictReader(open(os.path.join(LP, f"ground_recovery_{tag}_envs.csv"))))
    return z, rows


def burst_end(z):
    f, lo, hi, t0, on, glo, ghi, n = [float(x) for x in str(z["osc_inj"]).split(",")]
    return t0 + n * on + (n - 1) * z["osc_gap"], (t0, on, int(n))


def per_env(z, rows):
    t = z["t_rel"]; mode = z["mode"].astype(bool); wt = z["w_true"].astype(np.float32); wd = z["w_det"].astype(np.float32); al = z["alive"].astype(bool)
    T, N = mode.shape
    pre = z["pre_pass"].astype(bool)
    tend, (t0, on, nb) = burst_end(z) if str(z["osc_inj"]) else (np.full(N, np.nan), (np.nan, 0, 0))
    out = []
    for e in range(N):
        if not pre[e]:
            continue
        m = mode[:, e]; a = al[:, e]
        # 생존 구간만(낙상 뒤 리셋된 틱 제외)
        dead = np.where(~a)[0]; Te = dead[0] if len(dead) else T
        m = m[:Te]; w1 = wt[:Te, e]; w2 = wd[:Te, e]; tt = t[:Te]
        offs = np.where(m[:-1] & ~m[1:])[0] + 1
        ons = np.where(~m[:-1] & m[1:])[0] + 1
        prem = sum(1 for k in offs if w1[k:min(Te, k + 50)].max() >= 1.0)
        expo = 0.0
        if len(ons) or (len(m) and m[0]):
            k0 = ons[0] if len(ons) else 0
            expo = float(((~m[k0:]) & (w2[k0:] > 0.5)).sum() / 50.0)
        last_off = tt[offs[-1]] if len(offs) and not m[-1] else np.nan
        burst_w = np.nan
        if nb:
            inb = (tt >= t0) & (tt < float(tend[e]))
            burst_w = float(w1[inb].max()) if inb.any() else np.nan
        out.append(dict(env=e, fell=rows[e]["fell"] == "1", n_on=len(ons) + int(len(m) > 0 and m[0]), n_off=len(offs), prem=prem, expo=expo,
                        lat=(last_off - float(tend[e])) if nb else np.nan, stuck=bool(len(m) and m[-1]), amp=float(z["osc_amp"][e]) if len(z["osc_amp"]) else np.nan,
                        gap=float(z["osc_gap"][e]) if len(z["osc_gap"]) else np.nan, burst_w=burst_w))
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def calib(tag):
    z, rows = load(tag)
    E = per_env(z, rows)
    amps = np.array([e["amp"] for e in E]); bw = np.array([e["burst_w"] for e in E]); fl = np.array([e["fell"] for e in E])
    print(f"# 보정 {tag} — 사전검사 통과 {len(E)} env")
    edges = np.linspace(amps.min(), amps.max(), 7)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (amps >= lo) & (amps <= hi)
        if s.any():
            print(f"  τ {lo:5.1f}–{hi:5.1f} N·m: n {s.sum():3d} · 버스트 중 참 |ω| 최대 중앙 {np.nanmedian(bw[s]):.2f} (p10 {np.nanpercentile(bw[s], 10):.2f}·p90 {np.nanpercentile(bw[s], 90):.2f}) · 낙상 {100 * fl[s].mean():.0f}%")


def cmp(pfx, cond, arms):
    L = []
    P = lambda s="": (L.append(s), print(s, flush=True))   # noqa: E731
    P(f"# §13b 진동 주입 비교 — {pfx} · {cond} · 팔 {' '.join(arms)}")
    D = {}
    for a in arms:
        D[a] = {}
        for f in sorted(glob.glob(os.path.join(LP, f"ground_recovery_{pfx}_{a}_{cond}_s*_trace.npz"))):
            tag = os.path.basename(f)[len("ground_recovery_"):-len("_trace.npz")]
            sd = tag.rsplit("_s", 1)[1]
            z, rows = load(tag)
            for e in per_env(z, rows):
                D[a][(sd, e["env"])] = e
    keys = sorted(set.intersection(*[set(D[a]) for a in arms])) if arms else []
    P(f"짝 env {len(keys)} (seed {sorted(set(k[0] for k in keys))})")
    for a in arms:
        E = [D[a][k] for k in keys]
        if not E:
            continue
        lat = [e["lat"] for e in E if np.isfinite(e["lat"]) and not e["fell"]]
        P(f"  {a:>7}: 낙상 {100 * np.mean([e['fell'] for e in E]):.1f}% · 끄기/env {np.mean([e['n_off'] for e in E]):.2f} · 조기 해제 env {100 * np.mean([e['prem'] > 0 for e in E]):.1f}% (총 {sum(e['prem'] for e in E)}) · "
          f"노출 평균 {np.mean([e['expo'] for e in E]):.3f} s · 재켜짐(≥2) {100 * np.mean([e['n_on'] >= 2 for e in E]):.1f}% · 마지막 끄기 지연 중앙 {np.median(lat) if lat else float('nan'):+.2f} s · "
          f"끝까지 켜짐 {100 * np.mean([e['stuck'] for e in E]):.1f}% · 버스트 중 참|ω| 최대 중앙 {np.nanmedian([e['burst_w'] for e in E]):.2f}")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for nm, fn in (("낙상", lambda e: e["fell"]), ("조기 해제", lambda e: e["prem"] > 0), ("노출>0", lambda e: e["expo"] > 0)):
                x = sum(1 for k in keys if fn(D[a][k]) and not fn(D[b][k])); y = sum(1 for k in keys if fn(D[b][k]) and not fn(D[a][k]))
                P(f"    {a} vs {b} {nm}: {a}만 {x} / {b}만 {y} · p {mcnemar(x, y):.3g}")
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopest_results")
    with open(os.path.join(od, f"settle_osc_{pfx}_{cond}.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "calib":
        calib(sys.argv[2])
    else:
        cmp(sys.argv[2], sys.argv[3], sys.argv[4:])
