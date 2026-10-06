"""1007 병목 2: odic 배포 구현(c5_switch.SwitchDetector settle) 실물 블랙박스 재생 — **계약 검증, 참값 없음, 폐루프 성능 증명 아님**.

배포 경로 그대로: 중력 = correct_gravity(projected_gravity(quat_wxyz), 기록된 imu_mount_bias_deg), 자이로 = gyro_rad_s, 접촉 = contact(정책 0/1),
엔코더 = qpos_rad(JOINT_ORDER, 관절 프레임), 무릎 τ = knee_tau_joint(tau_est_nm(모터 프레임) × direction, status_age_s ≤ 0.1 s).
감지기 = stand_c5kr 설정(C5_DETECTOR_KW + C5_EXTRA det) — imu(현행) vs odic(k̂ 스윕·τ 기준 2 종·보상 없음). 기록 위에서 열린 고리로 돈다
(재생 감지기의 모드는 실제 실행된 정책을 바꾸지 않는다 — 기록된 실제 모드는 1004 의 C5 RV3 감지기 결과).

측정: (1) 회복 사건(재생 감지기 켜짐)별 imu·odic 끄기 시각·끄기 경로, 기록된 실제 모드와의 대응
      (2) 조용한 서기 구간 |v̄| 분포·θ 초과율(odic 이 '움직임' 이라고 볼 오탐) — k̂ 별
      (3) tau_est_nm 무릎 신호: 부호(관절 프레임 서기 토크)·양자화·갱신률·status 신선도·고주파 잡음·Δτ↔Δq 지연(상호상관)
사용: python3 reports/gen_w3/odic_real_replay.py [파일...]  (생략 = reports/real_1001~1004 의 rl_bridge_*.jsonl)
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import c5_switch as C5  # noqa: E402
from rl_walking.deploy.policy_runner import JOINT_ORDER  # noqa: E402

_spec = importlib.util.spec_from_file_location("_bridge_for_replay", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)

DET_KW = dict(B.C5_DETECTOR_KW, **B.C5_EXTRA["stand_c5kr_g21a_g24ra"]["det"])
KNEE_DIR = {"left_knee_joint": -1, "right_knee_joint": -1}      # deploy/hwmap_jetson_1004·1005.json direction(0727 실물 검증) — 무릎은 양쪽 −1
ARMS = [("imu", None)] + [(f"odic k{k:g}", dict(k_hat=float(k))) for k in (30, 35, 42, 45, 60)] + \
       [("odic k42 abs", dict(k_hat=42.0, tau_ref="absolute")), ("odi(보상 없음)", dict(k_hat=0.0)),
        ("k42 부호반대(대조)", dict(k_hat=42.0, _flip=True))]
IL, IR = JOINT_ORDER.index("left_knee_joint"), JOINT_ORDER.index("right_knee_joint")
OFF_STATE = []      # odic 조기 복귀 순간의 (파일, t, 상대 기울기°, 필터 |ω_xy|, 부호, k̂) — 현행 IMU 조기 복귀 조건(6°·0.4) 밖에서 끈 사례 집계


def load(path):
    R = []
    for ln in open(path):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "event" in r or not r.get("qpos_rad") or not r.get("quat_wxyz"):
            continue
        R.append(r)
    if not R:
        return None
    d = {"t": np.array([r["timestamp"] for r in R]) - R[0]["timestamp"],
         "q": np.array([r["qpos_rad"] for r in R], float), "gy": np.array([r["gyro_rad_s"] for r in R], float),
         "quat": np.array([r["quat_wxyz"] for r in R], float), "c": np.array([r["contact"] for r in R], float),
         "tau": np.array([[np.nan if v is None else v for v in (r.get("tau_est_nm") or [None] * 12)] for r in R], float),
         "age": np.array([np.nan if r.get("status_age_s") is None else r["status_age_s"] for r in R], float),
         "mode": np.array([int((r.get("c5") or {}).get("mode", 0) or 0) for r in R]),
         "hs": np.array([str(r.get("hang_state")) for r in R]), "ck": R[0].get("checkpoint"),
         "bias": [r.get("imu_mount_bias_deg") or [0.0, 0.0] for r in R]}
    d["g"] = np.array([B.correct_gravity(B.projected_gravity(qq), b) for qq, b in zip(d["quat"], d["bias"])], float)
    return d


def replay(d, settle):
    flip = -1.0 if (settle or {}).get("_flip") else 1.0          # 대조군: 무릎 τ 부호를 일부러 뒤집는다(관절 프레임 변환 오류 모사)
    settle = None if settle is None else {k: v for k, v in settle.items() if k != "_flip"}
    det = C5.SwitchDetector(**DET_KW, **({} if settle is None else {"settle": settle}))
    n = len(d["t"]); mode = np.zeros(n, int); spd = np.full(n, np.nan); ev = []
    on_k = None; ne = 0
    for k in range(n):
        if settle is None:
            m = det.update(d["g"][k], d["gy"][k], d["c"][k])
        else:
            tk = B.knee_tau_joint({"left_knee_joint": d["tau"][k, IL], "right_knee_joint": d["tau"][k, IR]}, d["age"][k] if np.isfinite(d["age"][k]) else None,
                                  KNEE_DIR)
            if tk is not None and flip < 0:
                tk = (-tk[0], -tk[1])
            m = det.update(d["g"][k], d["gy"][k], d["c"][k], qpos_rad=d["q"][k], tau_knee_nm=tk)
            spd[k] = float(det.settle.spd)
        mode[k] = m
        if m == 1 and on_k is None:
            on_k = k
        if m == 0 and on_k is not None:
            ev.append((on_k, k, "early" if det.n_early > ne else "calm")); on_k = None
            if det.n_early > ne and settle is not None:
                OFF_STATE.append((d["name"], float(d["t"][k]), float(getattr(det, "last_tilt", np.nan)), float(np.linalg.norm(det.last_w[:2])), flip, settle.get("k_hat"), settle.get("tau_ref", "handover")))
        ne = det.n_early
    if on_k is not None:
        ev.append((on_k, None, "끝까지"))
    return mode, spd, ev, det


def quiet_mask(d):
    n = len(d["t"]); free = d["hs"] == "FREE_STAND"
    m1 = np.convolve(d["mode"].astype(float), np.ones(150), "full")[:n] > 0
    wq = np.array([np.abs(d["gy"][max(0, k - 50):k + 1, :2]).max() for k in range(n)])
    return free & (d["mode"] == 0) & ~m1 & (d["c"] > 0.5).all(1) & (wq < 0.3)


def tau_report(d, out):
    free = d["hs"] == "FREE_STAND"; qm = quiet_mask(d)
    tj = np.stack([KNEE_DIR["left_knee_joint"] * d["tau"][:, IL], KNEE_DIR["right_knee_joint"] * d["tau"][:, IR]], 1)
    qk = d["q"][:, [IL, IR]]
    for s, nm in ((0, "왼"), (1, "오른")):
        x = tj[free, s]; x = x[np.isfinite(x)]
        if len(x) < 50:
            continue
        dx = np.abs(np.diff(x)); nz = dx[dx > 1e-9]
        step = float(np.min(nz)) if len(nz) else float("nan")
        chg = float((dx > 1e-9).mean())
        xq = tj[qm, s]
        hf = float(np.std(xq[2:-2] - np.convolve(xq, np.ones(5) / 5, "same")[2:-2])) if qm.sum() > 60 else float("nan")
        # Δτ ↔ Δq_knee 상호상관(FREE): 지연 L > 0 = τ 가 q 보다 L 틱 늦게 반응
        a = np.diff(tj[:, s]); b = np.diff(qk[:, s]); ok = np.isfinite(a) & np.isfinite(b) & free[1:]
        best = (0, 0.0)
        for L in range(-6, 7):
            if L >= 0:
                aa, bb, oo = a[L:], b[:len(b) - L], ok[L:] & ok[:len(ok) - L]
            else:
                aa, bb, oo = a[:L], b[-L:], ok[:L] & ok[-L:]
            if oo.sum() > 100:
                cc = float(np.corrcoef(aa[oo], bb[oo])[0, 1])
                if abs(cc) > abs(best[1]):
                    best = (L, cc)
        out.append(f"    무릎 {nm}: 관절 프레임 τ 중앙 {np.median(x):+.2f} N·m(p5–p95 {np.percentile(x, 5):+.2f}…{np.percentile(x, 95):+.2f}) · 양자화 {step:.4f} · "
                   f"틱 갱신률 {100 * chg:.0f}% · 조용한 서기 고주파 σ {hf:.3f} N·m(= 처짐 {np.degrees(hf / 42):.2f}° @k42) · "
                   f"Δτ↔Δq 최대 상관 {best[1]:+.2f} @ 지연 {best[0]:+d} 틱")
    a = d["age"][free]; a = a[np.isfinite(a)]
    if len(a):
        out.append(f"    status_age_s p50 {np.percentile(a, 50) * 1e3:.1f} · p99 {np.percentile(a, 99) * 1e3:.1f} · max {a.max() * 1e3:.1f} ms · > 0.1 s {100 * (a > 0.1).mean():.2f}%")


def main(paths):
    out = []
    agg_q = {nm: [] for nm, _ in ARMS if nm != "imu"}
    ev_rows = []
    for p in paths:
        d = load(p)
        if d is None or len(d["t"]) < 200:
            continue
        name = os.path.basename(p); d["name"] = name
        qm = quiet_mask(d)
        out.append(f"\n## {name} — {d['ck']} · {d['t'][-1]:.1f} s · 조용한 서기 {qm.sum() / 50:.1f} s · 기록 회복 모드 {int(d['mode'].sum())} 틱")
        tau_report(d, out)
        res = {nm: replay(d, st) for nm, st in ARMS}
        for nm, (mode, spd, ev, det) in res.items():
            if nm != "imu":
                agg_q[nm].append(spd[qm])
        # 조용한 서기 오탐(odic 이 '움직임'으로 볼 틱)·재생 감지기 켜짐
        for nm in ("imu", "odic k42", "odic k35", "odic k60", "odi(보상 없음)", "k42 부호반대(대조)"):
            mode, spd, ev, det = res[nm]
            on_q = sum(1 for (k0, k1, why) in ev if qm[k0])
            s = f"    재생 {nm:>14}: 켜짐 {len(ev)} · 그중 조용한 서기에서 켜짐 {on_q} · 회복 모드 {int(mode.sum())} 틱 · 조기 복귀 {det.n_early}"
            if nm != "imu" and qm.sum() > 50:
                sq = spd[qm]
                s += f" · 조용한 서기 |v̄| p50 {np.nanpercentile(sq, 50):.3f}·p99 {np.nanpercentile(sq, 99):.3f}·max {np.nanmax(sq):.3f} · θ0.07 초과 {100 * np.nanmean(sq >= 0.07):.2f}%"
            if nm != "imu":
                s += f" · τ 판정 불가 틱 {det.settle.n_tau_bad}"
            out.append(s)
        # 회복 사건: imu 재생 켜짐 기준으로 각 팔의 같은 사건 끄기 시각
        mode_i, _, ev_i, _ = res["imu"]
        for (k0, k1, why) in ev_i:
            row = [name, d["t"][k0], (None if k1 is None else d["t"][k1] - d["t"][k0]), why]
            km = k0
            log_on = bool(d["mode"][max(0, k0 - 10):k0 + 10].any())
            lo = np.where(d["mode"][k0:] == 0)[0] if log_on else []
            row.append((d["t"][k0 + lo[0]] - d["t"][k0]) if log_on and len(lo) else None)
            for nm, _ in ARMS[1:]:
                _, spd, ev, _ = res[nm]
                hit = [(a, b, w) for (a, b, w) in ev if abs(a - km) <= 5]
                if hit:
                    a, b, w = hit[0]
                    row.append(((None if b is None else d["t"][b] - d["t"][a]), w, float(np.nanmax(spd[a:(b or len(spd))]))))
                else:
                    row.append(None)
            ev_rows.append(row)
    out.append("\n## 회복 사건(재생 imu 감지기 켜짐 기준) — 끄기까지 시간 [s] · 경로(early = 조기 복귀, calm = 조용함 1 s) · 사건 중 |v̄| 최대")
    out.append("  파일 · 켜짐 t · imu 끄기(경로) · 기록 실제 모드 끄기 | " + " | ".join(nm for nm, _ in ARMS[1:]))
    for r in ev_rows:
        f = lambda v: "—" if v is None else f"{v:.2f}"   # noqa: E731
        cells = []
        for c in r[5:]:
            cells.append("—" if c is None else f"{f(c[0])}({c[1]}, v̄max {c[2]:.2f})")
        out.append(f"  {r[0][-20:-6]} · {r[1]:.1f} · {f(r[2])}({r[3]}) · {f(r[4])} | " + " | ".join(cells))
    out.append("\n## 조용한 서기 |v̄| 합계(전 파일) — odic 이 '아직 움직임' 으로 볼 비율(θ0.07)")
    for nm, xs in agg_q.items():
        x = np.concatenate(xs) if xs else np.array([])
        x = x[np.isfinite(x)]
        if len(x):
            out.append(f"  {nm:>14}: n {len(x)} 틱({len(x) / 50:.0f} s) · p50 {np.percentile(x, 50):.4f} · p90 {np.percentile(x, 90):.4f} · p99 {np.percentile(x, 99):.4f} · "
                       f"max {x.max():.3f} · θ0.07 초과 {100 * (x >= 0.07).mean():.2f}% · θ0.05 초과 {100 * (x >= 0.05).mean():.2f}%")
    k42 = [r for r in OFF_STATE if r[5] == 42.0 and r[4] > 0 and r[6] == "handover"]
    out.append("\n## odic(k̂ 42) 조기 복귀 순간 상태 — 현행 IMU 조기 복귀 조건(상대 기울기 < 6°·|ω| < 0.4) 밖에서 끈 사례")
    for r in k42:
        out.append(f"  {r[0][-20:-6]} t {r[1]:.2f} s · 상대 기울기 {r[2]:.2f}° · |ω_xy| {r[3]:.3f} rad/s" + ("  ← IMU 조건 밖" if (r[2] >= 6.0 or r[3] >= 0.4) else ""))
    txt = "\n".join(out)
    print(txt)
    od = os.path.join(REPO, "reports", "gen_w3", "stopest_results"); os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "odic_real_replay.txt"), "w", encoding="utf-8") as fo:
        fo.write("# odic 실물 블랙박스 재생(1007) — 계약 검증, 참값 없음, 폐루프 성능 증명 아님\n" + txt + "\n")


if __name__ == "__main__":
    ps = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl")))
    main(ps)
