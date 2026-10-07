"""STOPEST §13 (1007): odicw 개선 후보·조건 제거 팔의 실물 블랙박스 재생 R1–R3 + 끈 뒤 재흔들림 — **계약 검증, 참값 없음, 폐루프 성능 증명 아님**.

팔 = settle_attack.make_detector(배포 c5_switch 그대로, stand_c5kr KE 감지기 위 변형). 입력 경로 = odic_real_replay(load·knee_tau_joint).
  R1: odic 가 IMU 조건 밖(필터 |ω_xy| ≥ 0.4 또는 상대 기울기 ≥ 6°)에서 조기 복귀한 순간마다 팔 모드 = 1(유지)인가.
  R2: 조용한 서기 θ0.07 초과율·'조용한 서기에서 켜짐' 수 — odicw 와 같은가(오도메트리 입력을 바꾸는 xK 는 값이 달라도 정상, 보고만).
  R3: imu 사건별 끄기 시각 (팔 − imu) > +3 s 인 사건 수 — odic 대비 새로 생긴 수(기준 0).
  추가(열린 고리): 조기 해제 = 끈 뒤 1 s 안 원 |ω_xy| ≥ 1.0 · 흔들림 노출 = FREE_STAND·첫 켜짐 뒤 서기 모드인데 감지 |ω_xy| > 0.5 인 시간.
사용: python3 reports/gen_w3/settle_x_replay.py [팔 ...] → stopest_results/settle_x_replay.txt
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odic_real_replay as O  # noqa: E402
import settle_attack as A  # noqa: E402

REF = ("imu", "odic", "odicw")


def run(d, arm):
    det = A.make_detector(arm); n = len(d["t"])
    mode = np.zeros(n, int); spd = np.full(n, np.nan); tilt = np.full(n, np.nan); wf = np.zeros(n); ev = []
    on_k = None; ne = 0
    for k in range(n):
        if arm in ("imu", "imuT", "imuH"):
            m = det.update(d["g"][k], d["gy"][k], d["c"][k])
        else:
            tk = O.B.knee_tau_joint({"left_knee_joint": d["tau"][k, O.IL], "right_knee_joint": d["tau"][k, O.IR]},
                                    d["age"][k] if np.isfinite(d["age"][k]) else None, O.KNEE_DIR)
            m = det.update(d["g"][k], d["gy"][k], d["c"][k], qpos_rad=d["q"][k], tau_knee_nm=tk)
            spd[k] = float(det.settle.spd); tilt[k] = getattr(det, "last_tilt", np.nan)
        wf[k] = float(np.linalg.norm(np.asarray(det.last_w)[:2]))
        mode[k] = m
        if m == 1 and on_k is None:
            on_k = k
        if m == 0 and on_k is not None:
            ev.append((on_k, k, "early" if det.n_early > ne else "calm")); on_k = None
        ne = det.n_early
    if on_k is not None:
        ev.append((on_k, None, "끝까지"))
    return dict(mode=mode, spd=spd, tilt=tilt, wf=wf, ev=ev)


def main(arms):
    paths = sorted(glob.glob(os.path.join(O.REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl")))
    arms = list(dict.fromkeys(list(REF) + list(arms)))
    out = []
    P = lambda s="": (out.append(s), print(s, flush=True))   # noqa: E731
    P("# §13 재생 R1–R3 + 끈 뒤 재흔들림(실물 1001–1004 블랙박스, 열린 고리·참값 없음) — 팔: " + " ".join(arms))
    agg = {a: dict(r1=[], qx=[0, 0], qon=0, r3=0, ticks=0, prem=0, offs=0, expo=0.0) for a in arms}
    for p in paths:
        d = O.load(p)
        if d is None or len(d["t"]) < 200:
            continue
        name = os.path.basename(p)[-20:-6]
        qm = O.quiet_mask(d); free = d["hs"] == "FREE_STAND"
        wraw = np.linalg.norm(d["gy"][:, :2], axis=1); n = len(d["t"])
        R = {a: run(d, a) for a in arms}
        # R1 기준 순간: odic 조기 복귀 중 IMU 조건 밖
        r1k = [k1 for (k0, k1, why) in R["odic"]["ev"] if why == "early" and k1 is not None
               and (R["odic"]["wf"][k1] >= 0.4 or R["odic"]["tilt"][k1] >= 6.0)]
        for a in arms:
            r = R[a]; g = agg[a]
            g["r1"] += [(name, float(d["t"][k]), int(r["mode"][k])) for k in r1k]
            sq = r["spd"][qm]
            if a not in ("imu", "imuT", "imuH"):
                g["qx"][0] += int(np.nansum(sq >= 0.07)); g["qx"][1] += int(qm.sum())
            g["qon"] += sum(1 for (k0, k1, w) in r["ev"] if qm[k0])
            g["ticks"] += int(r["mode"].sum())
            for (k0, k1, w) in r["ev"]:
                if k1 is None:
                    continue
                g["offs"] += 1
                if wraw[k1:min(n, k1 + 50)].max() >= 1.0:
                    g["prem"] += 1
            on = np.where(r["mode"] == 1)[0]
            if len(on):
                after = np.arange(n) >= on[0]
                g["expo"] += float(((r["mode"] == 0) & after & free & (r["wf"] > 0.5)).sum() / 50.0)
        # R3: imu 사건별
        for (k0, k1, why) in R["imu"]["ev"]:
            ti = (n if k1 is None else k1) - k0
            for a in arms:
                hit = [(b0, b1) for (b0, b1, w) in R[a]["ev"] if abs(b0 - k0) <= 5]
                if hit:
                    b0, b1 = hit[0]
                    if ((n if b1 is None else b1) - b0 - ti) / 50.0 > 3.0:
                        agg[a]["r3"] += 1
        P(f"\n## {name} — 길이 {d['t'][-1]:.0f} s · 조용한 서기 {qm.sum() / 50:.0f} s · R1 기준 순간 {len(r1k)}")
        for a in arms:
            r = R[a]
            P(f"  {a:>7}: 사건 {len(r['ev'])} · 경로 {''.join({'early': 'e', 'calm': 'c'}.get(w, '∞') for _, _, w in r['ev'])} · 회복 틱 {int(r['mode'].sum())}")
    P("\n## 합계")
    P("  팔 · R1 유지/기준 · R2 조용한 서기 θ0.07 초과(틱)·켜짐 · R3 imu 대비 +3 s 초과 사건 · 총 회복 틱 · 끄기 수 · 조기 해제(끈 뒤 1 s 원|ω|≥1) · 흔들림 노출 s")
    for a in arms:
        g = agg[a]
        r1 = f"{sum(m for _, _, m in g['r1'])}/{len(g['r1'])}"
        qx = "—" if a in ("imu", "imuT", "imuH") else f"{g['qx'][0]}/{g['qx'][1]} ({100 * g['qx'][0] / max(1, g['qx'][1]):.2f}%)"
        P(f"  {a:>7} · R1 {r1} · R2 {qx}·켜짐 {g['qon']} · R3 {g['r3']} · 회복 틱 {g['ticks']} · 끄기 {g['offs']} · 조기 해제 {g['prem']} · 노출 {g['expo']:.2f} s")
    P("\n## 판정(§11 R 정의, odicw 기준)")
    for a in arms:
        if a in ("imu", "odic", "imuT", "imuH"):
            continue
        g = agg[a]
        r1 = all(m == 1 for _, _, m in g["r1"])
        r2 = (g["qx"] == agg["odicw"]["qx"]) and (g["qon"] == agg["odicw"]["qon"])
        r3 = g["r3"] <= agg["odic"]["r3"]
        P(f"  {a}: R1 {'PASS' if r1 else 'FAIL'} · R2 {'PASS(동일)' if r2 else '다름(보고)'} · R3 {'PASS' if r3 else 'FAIL'}")
    od = os.path.join(O.REPO, "reports", "gen_w3", "stopest_results")
    with open(os.path.join(od, "settle_x_replay.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["odicwN", "odicwH", "xV", "xDS", "xMIN", "xK", "xTAU"])
