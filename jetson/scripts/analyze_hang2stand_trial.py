#!/usr/bin/env python3
"""매달림→착지→지지제거→자립 시도 1회의 사후 분석 (2026-09-11).

사용:
  python3 jetson/scripts/analyze_hang2stand_trial.py <rl_bridge_*.jsonl> [--timeline session_*/timeline.jsonl] [--pads pads.jsonl]
  python3 jetson/scripts/analyze_hang2stand_trial.py --selftest

무엇을 하나 (원시 기록만 사용, CLAUDE.md 규칙 6):
  1. 구간 나누기: 워밍업/접촉 획득/지지 제거·자립/정지 — hang_state·contact·blend_alpha·stop 이벤트로.
  2. 1 s 빈 궤적: roll·pitch·|ω|·접촉 모드·관절 추종오차(이름 키)·액션 변화율.
  3. 접촉 비트 전환 목록(+ 미보정 raw_sum 상대치 — 하중 아님).
  4. 종료 직전 3 s: 추종오차·관절속도·|ω| → '관절 붕괴' vs '자세 유지한 채 접촉 상실' 구분.
  5. 최초 이상 발생 순서: roll·pitch·|ω|·접촉·추종오차·액션변화율의 이탈 시각을 정지 이벤트 기준으로 정렬(정렬 불확실성 명시).
  6. 타임라인이 있으면 운영자 이벤트(착지·방출·재인장·슬랙·보조)를 정지 기준 시각으로 정렬. 없으면 '윈치 미기록' 명시.
하지 않는 것: 발력%·CoP·CoM·좌우 하중 계산. 원인 확정(가설만 열거).
"""
import argparse, json, math, statistics, sys, time
from collections import Counter

JO = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
      "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
      "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
R2D = 57.29577951308232
TICK = 0.02           # 50 Hz
CONTACT_HOLD_S = 0.10  # 발압력 10 Hz S&H → 접촉 이탈 시각 불확실성
OPS = ("op_touchdown", "op_release_start", "op_release_stop", "op_retension", "op_slack_done", "op_assist", "support_confirm", "estop", "stand", "arm", "handoff", "ground")


def load_bb(path):
    ticks, events = [], []
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("event"): events.append(d)
        elif d.get("live") and "rp_deg" in d and "qpos_rad" in d: ticks.append(d)
    return ticks, events


def load_timeline(path):
    rows = []
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("kind") in ("action", "event", "action_start") and (d.get("action") in OPS or str(d.get("event", "")).startswith("op_")):
            try: rows.append((float(d.get("ts")), d.get("action") or d.get("event"), d.get("ok")))
            except (TypeError, ValueError): pass
    return rows


def gyro_mag(t):
    g = t.get("gyro_rad_s") or (0, 0, 0)
    return math.sqrt(sum(x * x for x in g)) * R2D


def track_err(t):
    q = t.get("qpos_rad"); tg = t.get("targets_deg_joint") or {}
    e = [abs(q[i] * R2D - tg[JO[i]]) for i in range(12) if JO[i] in tg]
    return sum(e) / len(e) if e else float("nan")


def act_rate(a, b):
    if not a or not b: return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def segments(ticks, events):
    t0 = ticks[0]["timestamp"]
    out = []
    # 워밍업: blend_alpha == 0 이면서 첫 명령 전(hold-prime 구간은 블랙박스에 기록되지 않을 수 있음)
    hs_prev, start = None, t0
    for t in ticks:
        hs = t.get("hang_state")
        if hs != hs_prev:
            if hs_prev is not None: out.append((hs_prev, start - t0, t["timestamp"] - t0))
            hs_prev, start = hs, t["timestamp"]
    out.append((hs_prev, start - t0, ticks[-1]["timestamp"] - t0))
    # 정지 이벤트는 여러 개일 수 있다(예: hang_guard 120 s 초과 후 재명령 → 마지막 lift_guard).
    # 기준은 **마지막** stop_all(=시도 종료). 전부 나열은 report 에서.
    stops = [e for e in events if e.get("event") == "stop_all"]
    stop = stops[-1] if stops else None
    return out, stop


def stable_window(ticks, max_len=1500):
    """기준선 창 = 가장 긴 접촉 (1,1) 연속 구간의 **앞 절반(최대 max_len 틱=30 s)**.

    구간 전체를 기준선으로 쓰면 접촉이 유지되는 동안 천천히 진행된 자세 표류가 기준선에
    흡수되고 스캔이 그 뒤에서 시작해 이탈 시각을 놓친다(셀프테스트로 확인). 앞 절반만 기준선으로
    두고 그 끝부터 스캔한다. (1,1) 구간이 짧으면 앞 절반 전체."""
    best, cur, cs = (0, 0), 0, 0
    for i, t in enumerate(ticks):
        if tuple(t.get("contact") or (0, 0)) == (1.0, 1.0):
            if cur == 0: cs = i
            cur += 1
            if cur > best[1] - best[0]: best = (cs, i + 1)
        else: cur = 0
    if best[1] - best[0] >= 500:          # (1,1) 구간 ≥ 10 s 일 때만 접촉 구간을 기준선으로 (σ 추정에 충분)
        lo, hi = best
        hi = lo + max(50, min((hi - lo) // 2, max_len))
    else:                                  # 공중 매달림 시도 등: 전반부(최대 30 s)를 기준선으로
        lo, hi = 0, max(50, min(len(ticks) // 2, max_len))
    return lo, hi


def onset_order(ticks, stop_ts):
    """신호별 최초 이탈 시각(정지 기준, 음수=정지 전). 규칙: 기준선 창의 median±4σ 또는 물리 임계를
    5틱 연속 초과. 접촉은 (1,1) 이 깨지는 첫 5틱. 정렬 불확실성: 틱 ±20 ms, 접촉 +100 ms(10 Hz S&H)."""
    lo, hi = stable_window(ticks)
    base = ticks[lo:hi]
    sig = {
        "roll": (lambda t: t["rp_deg"][0], 2.0),
        "pitch": (lambda t: t["rp_deg"][1], 2.0),
        "|ω|": (gyro_mag, 5.0),
        "추종오차": (track_err, 1.0),
    }
    res = {}
    for name, (f, phys) in sig.items():
        vals = [f(t) for t in base]; vals = [v for v in vals if not math.isnan(v)]
        if len(vals) < 10: res[name] = None; continue
        med = statistics.median(vals); sd = statistics.pstdev(vals) or 1e-6
        thr = max(4 * sd, phys)
        run = 0
        for t in ticks[hi:]:
            v = f(t)
            run = run + 1 if (not math.isnan(v) and abs(v - med) > thr) else 0
            if run >= 5: res[name] = t["timestamp"] - stop_ts - 4 * TICK; break
        else: res[name] = None
    # 액션 변화율
    prev = None; ar = []
    for t in base:
        a = t.get("raw_action"); ar.append(act_rate(a, prev) if prev else float("nan")); prev = a
    arv = [v for v in ar if not math.isnan(v)]
    if len(arv) > 10:
        med = statistics.median(arv); sd = statistics.pstdev(arv) or 1e-6; thr = 4 * sd
        prev = ticks[hi - 1].get("raw_action"); run = 0; res["액션변화율"] = None
        for t in ticks[hi:]:
            v = act_rate(t.get("raw_action"), prev); prev = t.get("raw_action")
            run = run + 1 if (not math.isnan(v) and v - med > thr) else 0
            if run >= 5: res["액션변화율"] = t["timestamp"] - stop_ts - 4 * TICK; break
    # 접촉
    run = 0; res["접촉(1,1) 이탈"] = None
    for t in ticks[hi:]:
        run = run + 1 if tuple(t.get("contact") or (0, 0)) != (1.0, 1.0) else 0
        if run >= 5: res["접촉(1,1) 이탈"] = t["timestamp"] - stop_ts - 4 * TICK; break
    return res, (ticks[lo]["timestamp"] - ticks[0]["timestamp"], ticks[hi - 1]["timestamp"] - ticks[0]["timestamp"])


def end_state(ticks, stop_reason=None):
    tN = ticks[-1]["timestamp"]
    rows = []
    for lab, a, b in (("종료-3~-2s", 3, 2), ("종료-2~-1s", 2, 1), ("종료-1~0s", 1, 0)):
        seg = [t for t in ticks if tN - a <= t["timestamp"] < tN - b + 1e-9]
        if not seg: continue
        te = statistics.mean(track_err(t) for t in seg)
        qv = statistics.mean(math.sqrt(sum(x * x for x in (t.get("qvel_rad_s") or [0] * 12))) * R2D for t in seg)
        wm = statistics.mean(gyro_mag(t) for t in seg)
        r = statistics.mean(t["rp_deg"][0] for t in seg); p = statistics.mean(t["rp_deg"][1] for t in seg)
        cc = Counter(tuple(t.get("contact") or (0, 0)) for t in seg).most_common(1)[0][0]
        rows.append((lab, te, qv, wm, r, p, cc))
    last = rows[-1] if rows else None
    verdict = None
    if last:
        dyn = last[3] > 20.0 or last[2] > 60.0          # |ω| > 20 °/s 또는 관절속도 > 60 °/s = 동적 낙상
        if last[1] > 5.0:
            verdict = "관절 붕괴 후보(추종오차 큼)"
        elif stop_reason and "terminated" in str(stop_reason):
            verdict = ("동적 낙상 종료(시뮬 기울기 종료, 접촉 비트는 아직 1,1)" if dyn else
                       "기울기 종료(준정적) — 실물 준정적 접촉 상실과 비교 대상")
        elif last[6] == (1.0, 1.0):
            verdict = "양발 접촉 유지한 채 종료(정지 없음/시간 종료) — 접촉 상실 아님"
        elif dyn:
            verdict = "동적 접촉 상실(|ω| 큼) — 실물 준정적 사례와 다른 부류"
        else:
            verdict = "자세 유지한 채 준정적 접촉 상실(추종오차·|ω| 작음) — 윈치 리프트/측면 표류/센서 중 판별 필요"
    return rows, verdict


def report(bb_path, timeline_path=None, pads_path=None, out=print, ref="stop"):
    ticks, events = load_bb(bb_path)
    if not ticks:
        out("정책 틱 0 — live 구간 없음"); return
    t0 = ticks[0]["timestamp"]; tN = ticks[-1]["timestamp"]
    segs, stop = segments(ticks, events)
    # 기준 시각: stop(마지막 정지, 기본) | support_gone(시뮬 덤프의 지지 소멸 첫 틱) | contact(첫 양발 접촉)
    if ref == "support_gone":
        _r = next((t for t in ticks if t.get("support_gone")), None)
        stop_ts = _r["timestamp"] if _r else (stop["timestamp"] if stop else tN)
    elif ref == "contact":
        _r = next((t for t in ticks if tuple(t.get("contact") or (0, 0)) == (1.0, 1.0)), None)
        stop_ts = _r["timestamp"] if _r else (stop["timestamp"] if stop else tN)
    else:
        stop_ts = stop["timestamp"] if stop else tN
    out(f"[기준 시각] ref={ref} → {stop_ts - t0:.1f}s (첫 틱 기준); 아래 '정지 기준' 상대시각은 이 기준이다")
    out(f"# 시도 분석: {bb_path.split('/')[-1]}")
    stops = [e for e in events if e.get("event") == "stop_all"]
    out(f"정책 틱 {len(ticks)}개, {time.strftime('%H:%M:%S', time.localtime(t0))}~{time.strftime('%H:%M:%S', time.localtime(tN))} ({tN - t0:.1f}s), "
        f"정지(마지막 기준): {stop.get('reason') if stop else '없음'}")
    if len(stops) > 1:
        out("  정지 이벤트 " + str(len(stops)) + "개: " + " | ".join(f"{e['timestamp'] - t0:+.1f}s {e.get('reason')}" for e in stops) + "  (중간 정지 후 재명령된 시도)")
    ck = ticks[0].get("checkpoint"); out(f"정책: {ck} · 첫 hang_state: {ticks[0].get('hang_state')} · blend_alpha 첫값 {ticks[0].get('blend_alpha')}")
    if stop and "warmup" in str(stop.get("reason")):
        out("⚠ 인계 워밍업 타임아웃 — stage8 status 스트림/CAN 점검 대상(정책 미구동).")
    out("\n## 1. 구간 (hang_state 기준, 시각은 첫 틱 기준 s)")
    for hs, a, b in segs: out(f"- {hs}: {a:.1f} ~ {b:.1f} s")
    out("  ※ 윈치 방출/재인장 구간은 블랙박스에 없다 — 타임라인 이벤트로만 알 수 있음.")
    # 1s 빈
    out("\n## 2. 1 s 빈 궤적 (roll° pitch° |ω|°/s 접촉 추종오차° 액션변화율)")
    bins = {}
    for t in ticks: bins.setdefault(int(t["timestamp"] - t0), []).append(t)
    keys = sorted(bins); show = keys if len(keys) <= 40 else keys[:8] + keys[-32:]
    prev = None
    for b in keys:
        ts = bins[b]; mid = ts[len(ts) // 2]
        cc = Counter(tuple(t.get("contact") or (0, 0)) for t in ts).most_common(1)[0][0]
        te = statistics.mean(track_err(t) for t in ts)
        ars = []
        for t in ts:
            a = t.get("raw_action"); ars.append(act_rate(a, prev) if prev else float("nan")); prev = a
        arv = [v for v in ars if not math.isnan(v)]
        if b in show:
            out(f"  {b:4d}  {mid['rp_deg'][0]:5.1f} {mid['rp_deg'][1]:5.1f} {max(gyro_mag(t) for t in ts):6.1f}  {str(cc):10s} {te:5.2f}  {statistics.mean(arv) if arv else float('nan'):.3f}")
        elif b == keys[8]: out("   ...")
    # 접촉 전환
    out("\n## 3. 접촉 비트 전환 (raw_sum 은 미보정 원값 — 하중 아님, 상대 변화만)")
    prevc = None; n = 0
    for t in ticks:
        c = tuple(t.get("contact") or (0, 0))
        if c != prevc:
            rs = t.get("foot_raw_sum") or ["?", "?"]
            out(f"  {t['timestamp'] - stop_ts:+7.1f}s(정지 기준) {prevc}→{c} raw_sum={rs} roll={t['rp_deg'][0]:.1f} pitch={t['rp_deg'][1]:.1f} hang={t.get('hang_state')}")
            prevc = c; n += 1
            if n > 30: out("  ...(30개 초과 생략)"); break
    dist = Counter(tuple(t.get("contact") or (0, 0)) for t in ticks)
    out("  분포: " + ", ".join(f"{k}:{v / len(ticks) * 100:.0f}%" for k, v in dist.most_common()))
    # 종료 직전
    out("\n## 4. 종료 직전 3 s (추종오차° qvelL2°/s |ω|°/s roll° pitch° 접촉)")
    rows, verdict = end_state(ticks, stop.get("reason") if stop else None)
    for lab, te, qv, wm, r, p, cc in rows: out(f"  {lab:12s} {te:6.2f} {qv:7.1f} {wm:6.1f} {r:5.1f} {p:5.1f} {cc}")
    if verdict: out(f"  → {verdict}")
    # 발생 순서
    res, (ws, we) = onset_order(ticks, stop_ts)
    out(f"\n## 5. 최초 이상 발생 순서 (정지 기준 s, 기준선 창 {ws:.1f}~{we:.1f}s; 틱 ±0.02 s, 접촉은 S&H 로 +{CONTACT_HOLD_S:.1f} s 불확실)")
    ordered = sorted(((v, k) for k, v in res.items() if v is not None))
    for v, k in ordered: out(f"  {v:+7.2f}s  {k}")
    for k, v in res.items():
        if v is None: out(f"     (없음)  {k} — 기준선 창 이후 이탈 없음/판정 불가")
    out("  해석 규칙: 순서를 미리 가정하지 않는다. roll/pitch 가 접촉보다 먼저면 자세 변화→접촉 상실, 접촉이 먼저면 센서/외부(윈치) 쪽 가설을 우선 검토. 추종오차가 먼저 커지면 관절 추종 문제.")
    # 타임라인
    out("\n## 6. 운영자 이벤트 (타임라인)")
    if timeline_path:
        rows = load_timeline(timeline_path)
        rows = [(ts, a, ok) for ts, a, ok in rows if t0 - 60 <= ts <= tN + 60]
        if rows:
            for ts, a, ok in rows: out(f"  {ts - stop_ts:+7.1f}s(정지 기준) {a} ok={ok}")
        else: out("  ⚠ 이 시도 창(±60 s)에 운영자 이벤트 기록 없음 — 윈치 조작 시각 불명.")
    else:
        out("  타임라인 미제공 — 윈치 조작 시각 불명(--timeline 로 session_*/timeline.jsonl 지정).")
    out("\n## 7. 가설 (확정 아님)")
    out("  H1 윈치 재인장/리프트로 발이 뜸 · H2 횡 편심/줄 측방 당김으로 롤 표류 · H3 한쪽 패드 감도 저하 · H4 정책 체중이동+lift_guard 상호작용 · H5 인계 레이스(워밍업 로그로 확인).")
    out("  판별: 3·5·6절의 순서와 영상(발·줄)로. 발력%·CoP 로 판정하지 않는다.")


def _synthetic():
    """합성 블랙박스: 10 s 안정 → roll 왼쪽 표류 → 오른발 접촉 소실 → lift_guard 정지."""
    rows = []; t0 = 1000.0
    for i in range(50 * 16):
        t = t0 + i * TICK
        roll = 1.0 if i < 500 else 1.0 - (i - 500) * 0.02          # 10 s 후 −0.02°/틱 표류 (→ −5°)
        contact = [1.0, 1.0] if i < 700 else ([1.0, 0.0] if i < 780 else [0.0, 0.0])
        q = [0.0] * 12; tg = {j: 0.2 for j in JO}   # 추종오차 0.2° 상수(자세 유지)
        rows.append({"timestamp": t, "live": True, "rp_deg": [roll, -2.9], "gyro_rad_s": [0.01, 0.01, 0.0],
                     "qpos_rad": q, "qvel_rad_s": [0.0] * 12, "raw_action": [0.0] * 12, "targets_deg_joint": tg,
                     "contact": contact, "foot_raw_sum": [2000, 1800 if i < 700 else 150],
                     "hang_state": "CONTACT_ACQUIRED", "blend_alpha": 1.0, "checkpoint": "synthetic"})
    rows.append({"timestamp": t0 + 800 * TICK, "event": "stop_all", "reason": "lift_guard: 양발 무접촉 0.4s 지속"})
    return rows


def selftest():
    import tempfile, os
    rows = _synthetic()
    p = os.path.join(tempfile.mkdtemp(), "rl_bridge_synth.jsonl")
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    ticks, events = load_bb(p)
    assert len(ticks) == 800 and events and events[0]["reason"].startswith("lift_guard")
    stop_ts = events[0]["timestamp"]
    res, _ = onset_order(ticks, stop_ts)
    assert res["roll"] is not None and res["접촉(1,1) 이탈"] is not None, res
    assert res["roll"] < res["접촉(1,1) 이탈"], ("합성: roll 표류가 접촉 이탈보다 먼저여야", res)
    assert res["추종오차"] is None, "추종오차 상수 → 이탈 없음"
    rows_, verdict = end_state(ticks)
    assert "자세 유지" in verdict, verdict
    buf = []; report(p, out=buf.append)
    txt = "\n".join(buf)
    assert "윈치 조작 시각 불명" in txt and "최초 이상 발생 순서" in txt
    print("[selftest] analyze_hang2stand_trial: 로드·발생순서(roll→접촉→정지)·종료판정(자세 유지)·리포트 — PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bb", nargs="?"); ap.add_argument("--timeline"); ap.add_argument("--pads")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ref", choices=("stop", "support_gone", "contact"), default="stop",
                    help="상대시각 기준: stop(마지막 정지) | support_gone(시뮬 덤프 지지 소멸) | contact(첫 양발 접촉)")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if not a.bb: ap.error("rl_bridge_*.jsonl 필요 (또는 --selftest)")
    report(a.bb, a.timeline, a.pads, ref=a.ref)


if __name__ == "__main__":
    main()
