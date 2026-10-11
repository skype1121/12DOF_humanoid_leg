#!/usr/bin/env python3
"""걷기 시작 직후 취소(전환 중 서기 복귀) 판정(1012): 대본의 walk→stop 쌍마다 켠 시간, KE 복귀까지 걸린 시간, 그 사이 낙상을 센다.
사용: waddle_cancel.py <이름=tag> ..."""
import json, sys, numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
for arg in sys.argv[1:]:
    name, tag = arg.split("=", 1)
    k = np.load(f"{LP}/kwloop_{tag}.npz", allow_pickle=True); a = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True)
    t = k["t"]; kind = k["kind"]; n = kind.shape[1]; pp = k["pre_pass"][:n].astype(bool); fell = k["fell"][:n].astype(bool) & pp
    alive = a["alive"].astype(bool); ta = a["t"]; scr = json.loads(str(k["script"])); ev = k["ev_t"]
    print(f"**{name}** — 낙상 {int(fell.sum())}/{int(pp.sum())} · 걷기 전환 {int(k['n_to_w3'][:n][pp].sum())} · 서기 복귀 {int(k['n_to_ke'][:n][pp].sum())}")
    print("| 켠 시간 | 켜짐 | 서기 복귀까지(중앙 / 최대) | 복귀 못 함 | 켠 뒤 ~ 복귀 뒤 2 s 사이 낙상 |"); print("|---|---|---|---|---|")
    for i, s in enumerate(scr):
        if not s[1] or i + 1 >= len(scr):
            continue
        dur = scr[i + 1][0] - s[0]; back = []; nob = 0; nf = 0; non = 0
        for e in range(n):
            if not pp[e]:
                continue
            t0 = float(ev[i, e]); t1 = float(ev[i + 1, e]); i0 = int(np.searchsorted(t, t0)); i1 = int(np.searchsorted(t, t1))
            nxt = float(ev[i + 2, e]) if i + 2 < ev.shape[0] else t[-1]; i2 = int(np.searchsorted(t, nxt))
            j0 = int(np.searchsorted(ta, t0))
            if j0 >= len(ta) or not alive[j0, e]:
                continue
            if not (kind[i0:i1 + 2, e] == 2).any():
                continue
            non += 1
            w = np.nonzero(kind[i1:i2, e] != 2)[0]
            tb = (t[i1 + w[0]] - t1) if len(w) else None
            if tb is None:
                nob += 1
            else:
                back.append(tb)
            jend = int(np.searchsorted(ta, (t1 + (tb if tb is not None else 4.0)) + 2.0)); jend = min(jend, len(ta) - 1)
            if not alive[j0:jend + 1, e].all():
                nf += 1
        print(f"| {dur:.1f} s | {non} | {np.median(back) if back else float('nan'):.2f} / {np.max(back) if back else float('nan'):.2f} s | {nob} | {nf} |")
