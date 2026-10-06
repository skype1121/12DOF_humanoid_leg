"""1007 병목 2 odicw(odic ∧ 감지 |ω_xy| < 0.4) 실물 블랙박스 재생 — STOPEST §11 R1–R3. **계약 검증, 참값 없음, 폐루프 성능 증명 아님**.

odic_real_replay.py 의 load·replay·quiet_mask 를 그대로 쓴다(배포 경로: correct_gravity·knee_tau_joint·SwitchDetector(settle)).
팔: imu(현행) · odic(k̂ 42, 인계 기준) · odicw(같은 + w_thr 0.4). 열린 고리 — 재생 감지기의 모드는 기록된 실제 정책을 바꾸지 않는다
(끄기 뒤 실제 몸 거동은 기록된 감지기(1004 C5 RV3·1001–1003 무회복) 아래의 것).
  R1: odic 가 필터 |ω_xy| ≥ 0.4(또는 상대 기울기 ≥ 6°)에서 조기 복귀한 순간마다 odicw 모드 = 1(유지)인가 + 그 사건의 odicw 최종 끄기 경로·시각·직전 0.2 s 원 |ω_xy| 최대
  R2: 조용한 서기 θ0.07 초과율·'조용한 서기에서 켜짐' 수 odicw = odic
  R3: imu 사건별 끄기 시각 (팔 − imu) > +3 s 인 사건 — odicw 에서 odic 대비 새로 생긴 수(기준 0), 총 회복 모드 틱
사용: python3 reports/gen_w3/odicw_real_replay.py [파일...] → stopest_results/odicw_real_replay.txt
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odic_real_replay as O  # noqa: E402

ARMS = [("imu", None), ("odic", dict(k_hat=42.0)), ("odicw", dict(k_hat=42.0, w_thr=0.4))]


def replay_ext(d, settle):
    """O.replay + 끄기 순간 감지 상태(상대 기울기·필터 |ω_xy|) — 사건 목록 (켜짐 k, 꺼짐 k, 경로, 기울기, 필터 ω)."""
    n0 = len(O.OFF_STATE)
    mode, spd, ev, det = O.replay(d, settle)
    st = O.OFF_STATE[n0:]
    del O.OFF_STATE[n0:]
    # 각 끄기의 감지 상태를 다시 계산하지 않고, 같은 감지기를 한 번 더 돌려 틱별 tilt·w 를 남긴다(결정론)
    det2 = O.C5.SwitchDetector(**O.DET_KW, **({} if settle is None else {"settle": dict(settle)}))
    tilt = np.full(len(d["t"]), np.nan); wf = np.full(len(d["t"]), np.nan); mode2 = np.zeros(len(d["t"]), int)
    for k in range(len(d["t"])):
        if settle is None:
            m = det2.update(d["g"][k], d["gy"][k], d["c"][k])
        else:
            tk = O.B.knee_tau_joint({"left_knee_joint": d["tau"][k, O.IL], "right_knee_joint": d["tau"][k, O.IR]},
                                    d["age"][k] if np.isfinite(d["age"][k]) else None, O.KNEE_DIR)
            m = det2.update(d["g"][k], d["gy"][k], d["c"][k], qpos_rad=d["q"][k], tau_knee_nm=tk)
        mode2[k] = m; tilt[k] = getattr(det2, "last_tilt", np.nan); wf[k] = float(np.linalg.norm(det2.last_w[:2]))
    assert np.array_equal(mode2, mode), "재생 결정론 위반"
    return mode, spd, ev, det, tilt, wf


def main(paths):
    out = []
    tot = {nm: 0 for nm, _ in ARMS}
    r1_rows = []; r2 = {nm: [0, 0, 0] for nm in ("odic", "odicw")}      # [θ 초과 틱, 조용한 서기 틱, 조용한 서기 켜짐]
    r3 = {nm: [] for nm in ("odic", "odicw")}
    ev_lines = []; wblock_tot = 0; early_rows = []
    for p in paths:
        d = O.load(p)
        if d is None or len(d["t"]) < 200:
            continue
        name = os.path.basename(p); d["name"] = name
        qm = O.quiet_mask(d)
        res = {nm: replay_ext(d, st) for nm, st in ARMS}
        wraw = np.linalg.norm(d["gy"][:, :2], axis=1)
        for nm in tot:
            tot[nm] += int(res[nm][0].sum())
        for nm in ("odic", "odicw"):
            mode, spd, ev, det, tilt, wf = res[nm]
            sq = spd[qm]
            r2[nm][0] += int(np.nansum(sq >= 0.07)); r2[nm][1] += int(np.isfinite(sq).sum())
            r2[nm][2] += sum(1 for (k0, k1, w) in ev if qm[k0])
        wblock_tot += res["odicw"][3].settle.n_w_block
        # odicw 조기 복귀 순간 상태(전부)
        _, _, evw, _, tw, ww = res["odicw"]
        for (k0, k1, why) in evw:
            if why == "early":
                early_rows.append((name, d["t"][k1], tw[k1], ww[k1], float(wraw[max(0, k1 - 10):k1 + 1].max())))
        # R1: odic 조기 복귀 중 IMU 조건 밖
        mo, _, evo, _, to_, wo = res["odic"]; mw = res["odicw"][0]
        for (k0, k1, why) in evo:
            if why != "early":
                continue
            outside = (to_[k1] >= 6.0) or (wo[k1] >= 0.4)
            # 같은 사건의 odicw 끄기: odicw 의 켜짐 구간 중 k1 을 포함(또는 k1 직전 끝난) 것
            hit = [(a, b, w) for (a, b, w) in evw if a <= k1 and (b is None or b >= k1 - 1)]
            a, b, w = hit[0] if hit else (None, None, "없음")
            held = bool(mw[k1 - 1] == 1 and mw[k1] == 1) if k1 >= 1 else False      # odic 가 끈 틱(k1)에서 odicw 는 여전히 1
            r1_rows.append(dict(f=name, t=d["t"][k1], tilt=to_[k1], wf=wo[k1], outside=outside, held=held,
                                w_off=(None if b is None else d["t"][b]), w_path=w, w_dt=(None if b is None else d["t"][b] - d["t"][k1]),
                                w_raw_pre=(None if b is None else float(wraw[max(0, b - 10):b + 1].max())),
                                o_raw_pre=float(wraw[max(0, k1 - 10):k1 + 1].max()),
                                o_raw_post=float(wraw[k1:k1 + 50].max()), w_raw_post=(None if b is None else float(wraw[b:b + 50].max()))))
        # R3: imu 사건별
        _, _, evi, _, _, _ = res["imu"]
        for (k0, k1, why) in evi:
            ti = (d["t"][k1] if k1 is not None else d["t"][-1]) - d["t"][k0]
            cells = []
            for nm in ("odic", "odicw"):
                ev = res[nm][2]
                h = [(a, b, w) for (a, b, w) in ev if abs(a - k0) <= 5]
                if h:
                    a, b, w = h[0]; ta = (d["t"][b] if b is not None else d["t"][-1]) - d["t"][a]
                    r3[nm].append((name, d["t"][k0], ta - ti))
                    cells.append(f"{ta:.2f}({w})")
                else:
                    cells.append("—")
            ev_lines.append(f"  {name[-20:-6]} · 켜짐 {d['t'][k0]:.1f} s · imu {ti:.2f}({why}) · odic {cells[0]} · odicw {cells[1]}")
    out.append("# odicw 실물 블랙박스 재생(1007, STOPEST §11 R1–R3) — 계약 검증, 참값 없음, 폐루프 성능 증명 아님")
    out.append(f"파일 {len(paths)} 개 · 감지기 stand_c5kr(KE) · odic/odicw k̂ 42 인계 기준 · odicw w_thr 0.4(감지 EMA 0.33 뒤 |ω_xy|)")
    out.append("\n## R1 — odic 조기 복귀 순간(전부)과 odicw")
    out.append("  파일 · t · 상대 기울기 · 필터 |ω_xy| · IMU 조건 밖 · odicw 유지 · odicw 끄기(경로, odic 대비 +s) · 끄기 직전 0.2 s 원 |ω| odic→odicw · 끈 뒤 1 s 원 |ω| 최대 odic→odicw")
    for r in r1_rows:
        fv = lambda v: "—" if v is None else f"{v:.2f}"   # noqa: E731
        out.append(f"  {r['f'][-20:-6]} · {r['t']:.2f} · {r['tilt']:.2f}° · {r['wf']:.3f} · {'밖' if r['outside'] else '안'} · {'유지' if r['held'] else '같이 끔'} · "
                   f"{fv(r['w_off'])}({r['w_path']}, {('+' + fv(r['w_dt'])) if r['w_dt'] is not None else '—'}) · {r['o_raw_pre']:.2f}→{fv(r['w_raw_pre'])} · "
                   f"{r['o_raw_post']:.2f}→{fv(r['w_raw_post'])}")
    outs = [r for r in r1_rows if r["outside"]]
    r1 = len(outs) > 0 and all(r["held"] for r in outs)
    out.append(f"  ⇒ R1: IMU 조건 밖 odic 조기 복귀 {len(outs)} 건 중 odicw 유지 {sum(r['held'] for r in outs)} → {'PASS' if r1 else 'FAIL'}")
    out.append(f"  odicw 가 |ω| 로 막은 틱 합계 {wblock_tot}")
    out.append("\n## odicw 조기 복귀 순간(전부) — 상대 기울기 · 필터 |ω_xy| · 직전 0.2 s 원 |ω_xy| 최대")
    for (nm, t, ti, wv, wr) in early_rows:
        out.append(f"  {nm[-20:-6]} t {t:.2f} · {ti:.2f}° · {wv:.3f} · {wr:.2f}" + ("  ← IMU 조건 밖" if (ti >= 6.0 or wv >= 0.4) else ""))
    out.append("\n## R2 — 조용한 서기 오판")
    for nm in ("odic", "odicw"):
        a, b, c = r2[nm]
        out.append(f"  {nm:>6}: θ0.07 초과 {a}/{b} 틱 ({100 * a / max(b, 1):.2f}%) · 조용한 서기에서 켜짐 {c}")
    rr2 = r2["odic"] == r2["odicw"]
    out.append(f"  ⇒ R2: {'PASS(동일)' if rr2 else 'FAIL(다름 — 구현 결함 의심)'}")
    out.append("\n## R3 — 회복 모드 지속(imu 사건 기준, 끄기까지 시간 [s])")
    out.extend(ev_lines)
    long_o = {(f, round(t, 2)) for (f, t, dt) in r3["odic"] if dt > 3.0}
    long_w = {(f, round(t, 2)) for (f, t, dt) in r3["odicw"] if dt > 3.0}
    new = long_w - long_o
    out.append(f"  imu 대비 +3 s 초과 사건: odic {len(long_o)} · odicw {len(long_w)} · odicw 에서 새로 생긴 {len(new)} {sorted(new) if new else ''}")
    out.append(f"  총 회복 모드 틱: imu {tot['imu']} · odic {tot['odic']} · odicw {tot['odicw']} (odicw − odic {tot['odicw'] - tot['odic']:+d} 틱 = {(tot['odicw'] - tot['odic']) / 50:+.2f} s)")
    rr3 = len(new) == 0
    out.append(f"  ⇒ R3: {'PASS' if rr3 else 'FAIL'}")
    out.append(f"\n## 결론(재생): R1 {'PASS' if r1 else 'FAIL'} · R2 {'PASS' if rr2 else 'FAIL'} · R3 {'PASS' if rr3 else 'FAIL'}")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "stopest_results", "odicw_real_replay.txt"), "w", encoding="utf-8") as fo:
        fo.write(txt + "\n")


if __name__ == "__main__":
    ps = sys.argv[1:] or sorted(glob.glob(os.path.join(O.REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl")))
    main(ps)
