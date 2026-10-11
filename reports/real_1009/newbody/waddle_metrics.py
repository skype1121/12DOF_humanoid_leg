#!/usr/bin/env python3
"""WADDLE-W1 판정 보조(1011): 왕복 평가 기록(transall_·kwloop_)에서 교대 스텝·전진·몸통 기울기·걷기 전환 목표 변화를 뽑는다.
스텝 = 걷기 모드에서 발 수직력 < 5 N 가 0.1 s 이상 이어진 뒤 다시 닿은 것(발 높이는 이 기록에 없다 — 정점 기준은 걷기 단독 평가에서).
사용: waddle_metrics.py <이름=tag:f_step> ..."""
import sys, json, numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
print("| 정책 | 낙상 | 걷기 중 한 발 완전 이륙 스텝(env·초당) | 예정 대비 % | 좌우 교대 비율 % | 공중 시간 중앙 s | 앞 명령 구간 전진 cm(중앙) | 한 걸음당 cm | 롤 p95° | 피치 p95° | 전환 0.2 s 목표 변화 최대°(중앙/p95) | 전환 한 틱 최대°(p95) | 스텝 0 인 env % |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1); tag, f = rest.split(":"); f = float(f)
    a = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True); k = np.load(f"{LP}/kwloop_{tag}.npz", allow_pickle=True)
    pp = a["pre_pass"].astype(bool); fell = a["fell"].astype(bool) & pp; t = a["t"]; mode = a["mode"]; alive = a["alive"].astype(bool); fz = a["fz"]; cmd = a["cmd"]
    n = mode.shape[1]; kind = k["kind"][:, :n]; walk = (kind[:len(t)] == 2) & alive; pp = k["pre_pass"][:n].astype(bool); fell = k["fell"][:n].astype(bool) & pp
    ptg = a["ptg"]; roll = np.abs(a["roll"]); pitch = np.abs(a["pitch"]); sc = 1.0 if np.nanmax(roll[walk]) > 1.6 else 57.2958
    nstep = np.zeros(n); alt = []; airs = []; wt = walk.sum(0) * 0.02; d02 = []; d1 = []; fwd = []; fsteps = []
    xy = k["xy"]; yaw = k["yaw"]; kt = k["t"]
    for e in range(n):
        if not pp[e]:
            continue
        seq = []
        for s in (0, 1):
            off = (fz[:, e, s] < 5.0) & walk[:, e]; i = 0
            while i < len(off):
                if off[i]:
                    j = i
                    while j < len(off) and off[j]:
                        j += 1
                    if j - i >= 5 and j < len(off) and walk[j, e]:
                        seq.append((j, s)); airs.append((j - i) * 0.02)
                    i = j
                else:
                    i += 1
        seq.sort(); nstep[e] = len(seq)
        if len(seq) > 1:
            alt.append(np.mean([seq[i][1] != seq[i - 1][1] for i in range(1, len(seq))]))
        w = np.nonzero(walk[:, e])[0]
        if len(w) and w[0] + 12 < len(t):
            i0 = w[0]; J = [j for j in range(12) if j not in (4, 5)]      # 4·5 = 기구 고정 hip_r(실물 미송신) 제외
            d02.append(np.rad2deg(np.abs(ptg[i0 + 10, e][J] - ptg[i0 - 1, e][J]).max())); d1.append(np.rad2deg(np.abs(np.diff(ptg[i0 - 1:i0 + 11, e][:, J], axis=0)).max()))
        # 앞 명령 구간: 걷기 모드이고 스크립트 명령 vx > 0 인 연속 구간의 전진(시작 요 방향)
        if fell[e]:
            continue
        ks = np.interp(t, kt, np.arange(len(kt))).astype(int)
        vx = k["want"][:, e] if False else None
    # 전진: kwloop 의 스크립트에서 앞 명령 걷기 구간
    scr = json.loads(str(k["script"]))
    ev_t = k["ev_t"]
    try:
        segs = [(i, s) for i, s in enumerate(scr)]
    except Exception:
        segs = []
    dist = []
    for i, s in segs:
        try:
            is_walk = bool(s[1]); c = s[2]
        except Exception:
            continue
        if not is_walk or c is None or abs(float(c[0])) < 1e-6 or i + 1 >= ev_t.shape[0]:
            continue
        for e in range(n):
            if not pp[e] or fell[e]:
                continue
            t0, t1 = float(ev_t[i, e]), float(ev_t[i + 1, e]); i0, i1 = np.searchsorted(kt, t0), min(np.searchsorted(kt, t1), len(kt) - 1)
            d = xy[i1, e] - xy[i0, e]; y0 = yaw[i0, e]; dist.append(d[0] * np.cos(y0) + d[1] * np.sin(y0))
            j0, j1 = np.searchsorted(t, t0), np.searchsorted(t, t1); ns = 0
            for s_ in (0, 1):
                off = (fz[j0:j1, e, s_] < 5.0).astype(int); ns += int((np.diff(off) == -1).sum())
            fsteps.append(ns)
    ok = pp
    sched = wt[ok] * 2 * f
    print(f"| {name} | {int(fell.sum())}/{int(pp.sum())} | {np.median(nstep[ok] / np.maximum(wt[ok], 1e-6)):.2f} | {100 * nstep[ok].sum() / max(sched.sum(), 1e-6):.0f} | {100 * np.mean(alt) if alt else float('nan'):.0f} | "
          f"{np.median(airs) if airs else float('nan'):.2f} | {100 * np.median(dist) if dist else float('nan'):.1f} | {100 * np.sum(dist) / max(np.sum(fsteps), 1) if dist else float('nan'):.2f} | "
          f"{np.nanpercentile(roll[walk].astype(np.float64) * sc, 95):.1f} | {np.nanpercentile(pitch[walk].astype(np.float64) * sc, 95):.1f} | {np.median(d02):.1f} / {np.percentile(d02, 95):.1f} | {np.percentile(d1, 95):.1f} | {100 * np.mean(nstep[ok] == 0):.0f} |")
