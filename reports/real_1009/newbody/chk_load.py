#!/usr/bin/env python3
"""1010 점검 P1(b): 걷기 시작 1 s·정상 걷기·서기 창의 토크·기계 출력 분포(NEWBODY_CHECKS.md P1 정의 그대로).
입력: log_picture/transall_<tag>.npz(--trace_qd 로 qd 포함) + kwloop_<tag>.npz. 사용: chk_load.py <tag> [<tag> ...]  → 마크다운 표."""
import json
import os
import sys
import numpy as np

LP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "log_picture")
JN = ["L_hip_f", "R_hip_f", "L_hip_a", "R_hip_a", "L_hip_r", "R_hip_r", "L_knee", "R_knee", "L_ank_f", "R_ank_f", "L_ank_r", "R_ank_r"]
HIPR = [4, 5]; ANKR = [10, 11]
J10 = [j for j in range(12) if j not in HIPR]                 # hip_r 제외
J8 = [j for j in range(12) if j not in HIPR + ANKR]           # AK70, hip_r 제외
J10A = [j for j in range(12) if j not in ANKR]                # AK70 10 관절(원래 몸)
LIM = np.array([25.0] * 10 + [24.0] * 2)                      # 추적 최대값에서 읽은 명목 한계(AK70 25, 발목 롤 24)


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.nan,) * 3 if x.size == 0 else (np.percentile(x, 50), np.percentile(x, 95), x.max())


def f3(x, d=1):
    a, b, c = pct(x); return f"{a:.{d}f} / {b:.{d}f} / {c:.{d}f}"


def load(tag):
    a = np.load(os.path.join(LP, f"transall_{tag}.npz"), allow_pickle=True); k = np.load(os.path.join(LP, f"kwloop_{tag}.npz"), allow_pickle=True)
    t = a["t"].astype(float); na = a["tq"].shape[1]
    tq = a["tq"].astype(np.float32); q = a["q"].astype(np.float32); cmd = a["cmd"].astype(np.float32); alive = a["alive"].astype(bool)
    qd = a["qd"].astype(np.float32) if "qd" in a.files else None
    assert qd is not None, f"{tag}: qd 없음(--trace_qd 필요)"
    kind = k["kind"][:, :na]; pp = k["pre_pass"][:na].astype(bool); evt = k["ev_t"][:, :na]; script = json.loads(str(k["script"])); es = float(k["effort_scale"][0]) if "effort_scale" in k.files and np.isfinite(k["effort_scale"][0]) and k["effort_scale"][0] > 0 else 1.0
    return dict(t=t, tq=tq, q=q, qd=qd, cmd=cmd, alive=alive, kind=kind, pp=pp, evt=evt, script=script, es=es, na=na)


def windows(D):
    """env 별 창 목록: (이름, i0, i1). 걷기 구간 j = 대본 j 번째 걷기 요청 뒤 처음 W3(kind 2)로 넘어간 틱."""
    t = D["t"]; out = {e: [] for e in range(D["na"])}
    walks = [(i, c) for i, (tt, w, c) in enumerate(D["script"]) if w]
    for e in range(D["na"]):
        if not D["pp"][e]:
            continue
        i0 = int(np.searchsorted(t, 0.5)); i1 = int(np.searchsorted(t, 2.0))
        if D["alive"][i0:i1, e].all():
            out[e].append(("stand", i0, i1))
        for j, (si, c) in enumerate(walks):
            treq = float(D["evt"][si, e]); ir = int(np.searchsorted(t, treq - 1e-6))
            nxt = float(D["evt"][walks[j + 1][0], e]) if j + 1 < len(walks) else t[-1] + 1
            w3 = np.nonzero((D["kind"][ir:, e] == 2))[0]
            if w3.size == 0 or t[ir + w3[0]] >= nxt:
                continue
            a0 = ir + int(w3[0]); typ = "fwd" if abs(c[0]) > 1e-6 else "inpl"
            tstop = float(D["evt"][si + 1, e]) if si + 1 < len(D["script"]) else t[-1]; istop = int(np.searchsorted(t, tstop))
            if a0 + 50 <= len(t) and D["alive"][a0:a0 + 50, e].all():
                out[e].append((f"start_{typ}", a0, a0 + 50))
            if a0 + 100 < istop and D["alive"][a0 + 100:istop, e].all():
                out[e].append((f"steady_{typ}", a0 + 100, istop))
    return out


def stats(D, joints_pow, joints_t2):
    W = windows(D); tqa = np.abs(D["tq"]); powj = np.abs(D["tq"] * D["qd"])
    R = {}
    stand_t2 = {}
    for e, ws in W.items():
        for nm, i0, i1 in ws:
            if nm == "stand":
                stand_t2[e] = float((D["tq"][i0:i1, e][:, joints_t2] ** 2).sum(-1).mean())
    for e, ws in W.items():
        for nm, i0, i1 in ws:
            r = R.setdefault(nm, dict(n=0, tqmax=[], pmax=[], pmean=[], t2x=[], err02=[], sat=[], t2roll=[]))
            r["n"] += 1
            r["tqmax"].append(tqa[i0:i1, e].max(0))
            p = powj[i0:i1, e][:, joints_pow].sum(-1).astype(float); r["pmax"].append(p.max()); r["pmean"].append(p.mean())
            t2 = (D["tq"][i0:i1, e][:, joints_t2].astype(float) ** 2).sum(-1)
            if e in stand_t2 and stand_t2[e] > 0:
                r["t2x"].append(t2.max() / stand_t2[e])
                if len(t2) >= 5:      # 0.1 s 이동 평균 최대(실물 기록 형식에 가깝게)
                    r["t2roll"].append(np.convolve(t2, np.ones(5) / 5, "valid").max() / stand_t2[e])
            if nm.startswith("start"):
                i2 = i0 + 10; r["err02"].append(float(np.rad2deg(np.abs(D["cmd"][i2, e] - D["q"][i2, e])[J10 if joints_pow == J10 else list(range(12))].sum())))
            r["sat"].append(float((tqa[i0:i1, e][:, J10] >= 0.95 * LIM[J10] * D["es"]).any(1).mean()) * 100)
    return R


def main():
    tags = sys.argv[1:]
    allR = {}
    for tag in tags:
        D = load(tag); ob = "_ob_" in tag
        allR[tag] = stats(D, J10, J8)
        if ob:
            allR[tag + " (hip_r 포함)"] = stats(D, list(range(12)), J10A)
    names = ["stand", "start_inpl", "start_fwd", "steady_inpl", "steady_fwd"]
    lab = {"stand": "서기(인계 뒤 0.5–2.0 s)", "start_inpl": "제자리 시작 1 s", "start_fwd": "앞 0.06 시작 1 s", "steady_inpl": "제자리 정상", "steady_fwd": "앞 0.06 정상"}
    print("값 = env·구간 분포의 p50 / p95 / 최대. 기계 출력 = Σ|τ·ω| [W], Στ² 배수 = 창 최대 ÷ 그 env 서기 평균.\n")
    print("| 실행 | 창 | 표본(env·구간) | 기계 출력 창 최대 [W] | 기계 출력 창 평균 [W] | Στ² 순간 최대 ÷ 서기 | Στ² 0.1 s 평균 최대 ÷ 서기 | 0.2 s 뒤 Σ\\|목표−실측\\| [°] | 포화 틱 [%] |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, R in allR.items():
        for nm in names:
            if nm not in R:
                continue
            r = R[nm]
            print(f"| {tag} | {lab[nm]} | {r['n']} | {f3(r['pmax'])} | {f3(r['pmean'])} | {f3(r['t2x'])} | {f3(r['t2roll'])} | {f3(r['err02']) if r['err02'] else '—'} | {f3(r['sat'])} |")
    print("\n관절별 최대 |토크| [N·m] (창 안 최대의 env·구간 분포 p50 / p95 / 최대)\n")
    print("| 실행 | 창 | " + " | ".join(JN) + " |"); print("|---|---|" + "---|" * 12)
    for tag, R in allR.items():
        if "포함" in tag:
            continue
        for nm in names:
            if nm not in R:
                continue
            m = np.array(R[nm]["tqmax"])
            print(f"| {tag} | {lab[nm]} | " + " | ".join(f3(m[:, j]) for j in range(12)) + " |")
    if len(tags) >= 2:      # 첫 두 실행의 비(첫째 ÷ 둘째)
        A, B = allR[tags[0]], allR[tags[1] + " (hip_r 포함)"] if (tags[1] + " (hip_r 포함)") in allR else allR[tags[1]]
        B10 = allR[tags[1]]
        print(f"\n비 = {tags[0]} ÷ {tags[1]} (p50 · p95). 원래 몸 쪽은 hip_r 포함(전 관절) 값 / hip_r 제외 값 두 가지.\n")
        print("| 창 | 기계 출력 창 최대: 대 전 관절 | 대 hip_r 제외 | 기계 출력 창 평균: 대 전 관절 | 대 hip_r 제외 | Στ² 0.1 s 최대(절대값 아님, 서기 대비 배수): 대 AK70 10 | 대 8 |")
        print("|---|---|---|---|---|---|---|")
        for nm in names[1:]:
            if nm not in A or nm not in B:
                continue
            def rr(key, X, Y):
                a = pct(X[nm][key]); b = pct(Y[nm][key]); return f"{a[0] / b[0]:.2f} · {a[1] / b[1]:.2f}"
            print(f"| {lab[nm]} | {rr('pmax', A, B)} | {rr('pmax', A, B10)} | {rr('pmean', A, B)} | {rr('pmean', A, B10)} | {rr('t2roll', A, B)} | {rr('t2roll', A, B10)} |")


if __name__ == "__main__":
    main()
