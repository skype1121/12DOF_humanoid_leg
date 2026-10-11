#!/usr/bin/env python3
"""서기 → 걷기 전환 완화 판정표(1012): 왕복 평가 기록(transall_·kwloop_[·rblend_])에서 전환 구간을 따로 본다.
요구 토크 = kp·(ptg − q) − kd·qd (제한 전), tq = 시뮬이 실제로 넣은 토크(제한 뒤). 관절 순서 = 러너(좌우 교대): 8번 = 오른 hip_a = 열 3, hip_r(4·5)은 기구 고정이라 제외.
구간: A = 걷기 켠 뒤 0–0.2 s, B = 0.2–1 s, W = 걷는 틱 전체. 하드웨어 한계(가정) = AK70 25 N·m·발목 롤(AK45-36) 24 N·m.
사용: waddle_trans.py <이름=tag:f_step> ...   (표준 출력 = 마크다운)"""
import json, sys, numpy as np
LP = "/home/ryu/humanoid_leg_test1/log_picture"
KP, KD = 150.0, 5.0
J8, J2 = 3, 2
GROUPS = {"hip_f": [0, 1], "hip_a(2번)": [2], "무릎": [6, 7], "발목 앞뒤": [8, 9], "발목 롤": [10, 11]}
LIM = np.array([25, 25, 25, 25, 1e9, 1e9, 25, 25, 25, 25, 24, 24], np.float32)
P = lambda x, q: float(np.percentile(x, q)) if len(x) else float("nan")


def one(name, tag, f):
    a = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True); k = np.load(f"{LP}/kwloop_{tag}.npz", allow_pickle=True)
    t = a["t"]; n = a["q"].shape[1]; T = len(t)
    kind = k["kind"][:T, :n]; pp = k["pre_pass"][:n].astype(bool); fell = k["fell"][:n].astype(bool) & pp; alive = a["alive"].astype(bool)
    dem = KP * (a["ptg"].astype(np.float32) - a["q"].astype(np.float32)) - KD * a["qd"].astype(np.float32); tq = a["tq"].astype(np.float32)
    fz = a["fz"].astype(np.float32); u = a["u"].astype(np.float32); ptg = a["ptg"].astype(np.float32)
    roll = np.abs(a["roll"].astype(np.float64)); pitch = np.abs(a["pitch"].astype(np.float64))
    dfw = (a["ffwd"][:, :, 0].astype(np.float32) - a["ffwd"][:, :, 1].astype(np.float32)); dlt = (a["flat"][:, :, 0].astype(np.float32) - a["flat"][:, :, 1].astype(np.float32))
    walk = (kind == 2) & alive
    R = dict(name=name, fall=f"{int(fell.sum())}/{int(pp.sum())}", n_on=0)
    A8, B8, A8t, B8t = [], [], [], []; AB = {g: [] for g in GROUPS}; over = 0; ntr = 0; d02 = []; d1 = []
    t_lift = []; first_foot = []; air1 = []; t_fall_tr = 0; mism_tr = []; mism_st = []; supp = []; rp_tr = []; slip_tr = []; slip_st = []
    for e in range(n):
        if not pp[e]:
            continue
        w = walk[:, e]; on = np.nonzero(w[1:] & ~w[:-1])[0] + 1
        for i0 in on:
            if i0 + 50 >= T:
                continue
            ntr += 1; sA = slice(i0, i0 + 10); sB = slice(i0 + 10, i0 + 50); sAB = slice(i0, i0 + 50); s2 = slice(i0, min(i0 + 100, T))
            m = w[sAB]
            A8 += list(np.abs(dem[sA, e, J8])[w[sA]]); B8 += list(np.abs(dem[sB, e, J8])[w[sB]])
            A8t += list(np.abs(tq[sA, e, J8])[w[sA]]); B8t += list(np.abs(tq[sB, e, J8])[w[sB]])
            for g, js in GROUPS.items():
                AB[g] += list(np.abs(dem[sAB, e][:, js])[m].max(1)) if m.any() else []
            if m.any() and (np.abs(dem[sAB, e])[m] > LIM[None, :]).any():
                over += 1
            Jn = [j for j in range(12) if j not in (4, 5)]
            d02.append(np.degrees(np.abs(ptg[i0 + 10, e][Jn] - ptg[i0 - 1, e][Jn]).max())); d1.append(np.degrees(np.abs(np.diff(ptg[i0 - 1:i0 + 51, e][:, Jn], axis=0)).max()))
            if not alive[s2, e].all():
                t_fall_tr += 1
            # 첫 이륙(수직력 < 5 N 이 0.1 s 이상)
            best = None
            for s_ in (0, 1):
                off = (fz[i0:min(i0 + 200, T), e, s_] < 5.0); j = 0
                while j < len(off):
                    if off[j]:
                        j2 = j
                        while j2 < len(off) and off[j2]:
                            j2 += 1
                        if j2 - j >= 5:
                            if best is None or j < best[0]:
                                best = (j, s_, j2 - j)
                            break
                        j = j2
                    else:
                        j += 1
            if best is not None:
                t_lift.append(best[0] * 0.02); first_foot.append(best[1]); air1.append(best[2] * 0.02)
                j0, s_, L = best; supp.append(float(fz[i0 + j0:i0 + j0 + L, e, 1 - s_].min()))
            rp_tr.append(max(float(np.nanmax(roll[s2, e])), 0.0)); rp_tr.append(-max(float(np.nanmax(pitch[s2, e])), 0.0))
        # 위상–접지 어긋남: 예정 스윙 창(반주기 0.25–0.75)에서 스윙발이 닿아 있거나, 예정 지지발이 떠 있는 틱 비율
        v = np.remainder(2.0 * u[:, e], 1.0); sw = (u[:, e] >= 0.5).astype(int); inw = (v >= 0.25) & (v < 0.75) & w
        if inw.any():
            idx = np.nonzero(inw)[0]
            bad = (fz[idx, e, sw[idx]] > 5.0) | (fz[idx, e, 1 - sw[idx]] < 5.0)
            since = np.zeros(T, int); c = 10 ** 6
            for i in range(T):
                c = 0 if (w[i] and (i == 0 or not w[i - 1])) else c + 1; since[i] = c
            tr = since[idx] < 100
            mism_tr += list(bad[tr]); mism_st += list(bad[~tr])
        # 양발 지지 중 두 발 상대 이동(한 구간의 시작–끝, mm) — 한쪽이라도 미끄러지면 생긴다
        both = (fz[:, e, 0] > 20.0) & (fz[:, e, 1] > 20.0) & w; i = 0
        while i < T:
            if both[i]:
                j = i
                while j < T and both[j]:
                    j += 1
                if j - i >= 3:
                    d = 1000.0 * float(np.hypot(dfw[j - 1, e] - dfw[i, e], dlt[j - 1, e] - dlt[i, e]))
                    c = 10 ** 6
                    for i2 in range(i, max(i - 100, -1), -1):
                        if w[i2] and (i2 == 0 or not w[i2 - 1]):
                            c = i - i2; break
                    (slip_tr if c < 100 else slip_st).append(d)
                i = j
            else:
                i += 1
    W8 = np.abs(dem[:, :, J8])[walk & pp[None, :]]; W8t = np.abs(tq[:, :, J8])[walk & pp[None, :]]
    A8, B8, A8t, B8t = map(np.array, (A8, B8, A8t, B8t))
    rb = None
    try:
        rb = np.load(f"{LP}/rblend_{tag}.npz")
    except Exception:
        pass
    R.update(ntr=ntr, fall_tr=t_fall_tr,
             A8=f"{A8.mean():.1f} / {P(A8, 95):.1f} / {P(A8, 99):.1f} / {A8.max():.1f}", B8=f"{B8.mean():.1f} / {P(B8, 95):.1f} / {P(B8, 99):.1f} / {B8.max():.1f}",
             W8=f"{W8.mean():.1f} / {P(W8, 95):.1f} / {P(W8, 99):.1f} / {W8.max():.1f}", W8t=f"{W8t.mean():.1f} / {P(W8t, 95):.1f} / {P(W8t, 99):.1f} / {W8t.max():.1f}",
             ABt=f"{np.concatenate([A8t, B8t]).max():.1f}", over=f"{over}/{ntr}",
             grp=" · ".join(f"{g} {P(np.array(x), 99):.0f}/{np.max(x):.0f}" for g, x in AB.items()),
             d02=f"{np.median(d02):.1f} / {P(np.array(d02), 95):.1f}", d1=f"{P(np.array(d1), 95):.1f}",
             lift=f"{np.median(t_lift):.2f} s", left=f"{100 * np.mean(np.array(first_foot) == 0):.0f} %", air=f"{np.median(air1):.2f} s", nolift=f"{ntr - len(t_lift)}/{ntr}",
             supp=f"{np.median(supp):.0f} / {P(np.array(supp), 5):.0f} N",
             mism=f"{100 * np.mean(mism_tr):.0f} % / {100 * np.mean(mism_st):.0f} %", slip=f"{P(np.array(slip_tr), 95):.1f} / {P(np.array(slip_st), 95):.1f} mm",
             rp=f"{P(np.array([x for x in rp_tr if x >= 0]), 95):.1f}° / {P(np.array([-x for x in rp_tr if x < 0]), 95):.1f}°",
             dev=("—" if rb is None else f"{float(np.percentile(rb['dev'].max(1)[rb['dev'].max(1) > 0], 50)):.1f} / {float(rb['dev'].max()):.1f}°"),
             kerec=("—" if rb is None else f"{int(rb['ke_rec'].sum())}"),
             tinf=("—" if rb is None or len(rb["t_bl"]) == 0 else f"{1000 * float(np.mean(rb['t_no'])):.2f} → {1000 * float(np.mean(rb['t_bl'])):.2f} ms (최대 {1000 * float(np.max(rb['t_bl'])):.1f})"))
    return R


RS = []
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1); tag, f = (rest.split(":") + ["1.1"])[:2]
    try:
        RS.append(one(name, tag, float(f)))
    except FileNotFoundError as e:
        print(f"<!-- {name}: 기록 없음 {e} -->")
rows = [("왕복 낙상", "fall"), ("걷기 전환 횟수 / 전환 뒤 2 s 안 낙상", None), ("8번 요구 0–0.2 s (평균 / P95 / P99 / 최대)", "A8"), ("8번 요구 0.2–1 s", "B8"), ("8번 요구 걷는 틱 전체", "W8"),
        ("8번 제한 뒤 걷는 틱 전체", "W8t"), ("8번 제한 뒤 0–1 s 최대", "ABt"), ("0–1 s 에 어느 관절이든 한계 초과한 전환", "over"), ("다른 관절 0–1 s 요구 P99/최대", "grp"),
        ("0.2 s 목표 변화(중앙 / P95)", "d02"), ("0–1 s 한 틱 목표 변화 P95", "d1"), ("첫 이륙까지(중앙)", "lift"), ("첫 이륙이 왼발인 비율", "left"), ("첫 공중 시간(중앙)", "air"),
        ("4 s 안에 이륙 없음", "nolift"), ("첫 스윙 중 지지발 수직력 최소(중앙 / 하위 5 %)", "supp"), ("위상–접지 어긋남(전환 2 s / 그 뒤)", "mism"),
        ("양발 지지 중 두 발 상대 이동 P95(전환 2 s / 그 뒤)", "slip"), ("전환 2 s 롤 / 피치 최대의 P95", "rp"), ("내보낸 목표 − 걷기 정책 목표(전환별 최대의 중앙 / 최대)", "dev"),
        ("섞는 중 서기 러너가 회복 모드로 간 env", "kerec"), ("러너 한 틱 시간(평소 → 섞는 중)", "tinf")]
print("| 항목 | " + " | ".join(r["name"] for r in RS) + " |"); print("|---|" + "---|" * len(RS))
for lab, key in rows:
    print(f"| {lab} | " + " | ".join((f"{r['ntr']} / {r['fall_tr']}" if key is None else str(r[key])) for r in RS) + " |")
