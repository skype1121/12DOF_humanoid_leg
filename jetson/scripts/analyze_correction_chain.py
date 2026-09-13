#!/usr/bin/env python3
"""보정 전달 사슬 분석 (0912 리뷰 요구): 첫 패드 OFF 전후 ~ 낙상까지, 같은 시간축에
  패드 on/off · actor 접촉 관측 · IMU · 정책 원출력 → 체인 후 목표 → 제어기 수락 명령 → 관절 실제각/추종오차 · 정책 활성/블렌드/가드
를 놓고 세 경우를 가른다.
  ① 정책 원출력 자체가 보정하지 않았다          (raw_action 이 안정창 대비 거의 불변)
  ② 정책은 보정했으나 체인(슬루·LPF·클램프·블렌딩·가드)에서 명령이 달라졌다 (raw→target 전달비 낮음 / 클램프·동결)
  ③ 최종 목표는 변했으나 실제 관절이 추종하지 못했다 (target−qpos 추종오차 큼 / stage8 cmd 대비 act 지연)
기록되지 않은 항목은 '미기록'으로 표시한다. 발력%·CoP 계산 없음(패드는 on/off 만).
사용: python3 analyze_correction_chain.py <rl_bridge.jsonl> [--stiff stiff_cop.jsonl] [--pads pads_inc.jsonl] [--hwmap config/robot_12dof_hardware_map.json]
      python3 analyze_correction_chain.py --selftest
"""
import argparse, json, math, statistics as st, sys, time, os

JO = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
      "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
      "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
DEF_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0, 6.0, 1.4, -1.4, 0.0, 0.0]   # policy_runner.DEFAULT_POSE_RAD (deg)
ACTION_SCALE = 0.25; R2D = 57.29577951308232; TICK = 0.02
LEFT_BAD_PAD = 2      # 0912 승윤님: 왼발 index 2 패드 상시 눌림(고장) — on/off 판정에서 제외
PAD_ON = 80.0         # 어댑터 PAD_CONTACT_COUNTS (0912 시점)


def load_bb(path):
    ticks, events = [], []
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("event"): events.append(d)
        elif d.get("live") and "rp_deg" in d and "raw_action" in d: ticks.append(d)
    return ticks, events


def strip_roll_comp(ticks):
    """브리지 롤 보정기(roll_comp) 오프셋을 목표에서 제거해 '정책 체인 목표'를 복원한다 (0913 F7).

    전달비(Δ목표/Δ정책출력)는 정책이 낸 보정만 봐야 한다 — 외부 루프인 보정기 오프셋이 섞이면
    정책이 보정한 것처럼 보인다. 각 틱에 _tgt_pre(JOINT_ORDER 순 deg)를 넣는다.
      · targets_deg_joint_pre_rc 가 기록돼 있으면 그대로 사용(정확 — lr·부호 포함)
      · 없으면 roll_comp_joints(없으면 ankle_r)·lr=[1,1] 을 **가정**하고 경고를 반환한다
    반환 (활성 틱 수, |오프셋| 평균 deg, |오프셋| 최대 deg, 경고 문자열 또는 None). 추종오차는
    최종 명령 기준이라 건드리지 않는다(관절은 오프셋 포함 명령을 따라간다).
    """
    n = 0; acc = 0.0; mx = 0.0; warn = None
    for t in ticks:
        off = t.get("roll_comp_deg")
        if not isinstance(off, (int, float)):
            continue
        n += 1; acc += abs(off); mx = max(mx, abs(off))
        pre = t.get("targets_deg_joint_pre_rc")
        if isinstance(pre, list) and len(pre) == 12:
            t["_tgt_pre"] = [float(v) for v in pre]
            continue
        j = t.get("roll_comp_joints") or "ankle_r"
        warn = warn or f"targets_deg_joint_pre_rc 미기록 → 보정 관절 {j}·lr[1,1] **가정**하고 뺌(기록 버전 확인)"
        cols = [JO.index(f"left_{j}_joint"), JO.index(f"right_{j}_joint")] if f"left_{j}_joint" in JO else []
        pre = [t["targets_deg_joint"][JO[k]] for k in range(12)]
        for c in cols:
            pre[c] -= float(off)
        t["_tgt_pre"] = pre
    return n, (acc / n if n else 0.0), mx, warn


def tgt_pre(t, k):
    """전달비 계산용 목표[deg] — 보정기 오프셋을 뺀 값(없으면 기록 그대로)."""
    v = t.get("_tgt_pre")
    return v[k] if v is not None else t["targets_deg_joint"][JO[k]]


def load_pads(path, t0, t1):
    rows = []
    if not path or not os.path.exists(path): return rows
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        ts = d.get("timestamp")
        if isinstance(ts, (int, float)) and t0 <= ts <= t1 and "left" in d and "right" in d: rows.append(d)
    return rows


def pads_on(row):
    L = [i for i, v in enumerate(row["left"]) if i != LEFT_BAD_PAD and v > PAD_ON]
    Rr = [i for i, v in enumerate(row["right"]) if v > PAD_ON]
    return L, Rr


def load_stiff(path):
    rows = []
    if not path or not os.path.exists(path): return rows
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if "joints" in d and "t" in d: rows.append(d)
    return rows


def align_stiff(stiff, ticks, guess_unix):
    """stiff 't'(로거 기준 초) → unix 오프셋. 발력 합(fn) 상호상관으로 ±180 s 탐색. 반환 (offset, peak_corr)."""
    if not stiff: return None, None
    # 정렬 신호: 관절각(왼무릎·오른무릎·왼힙F 의 act 합) — 발력합보다 착지·기립 과도에서 상관이 뚜렷하다 (0912: fn 정렬은 상관 0.46~0.70 로 프레임 정합 실패)
    KEYS = ("left_knee_joint", "right_knee_joint", "left_hip_f_joint", "right_hip_f_joint")
    KI = [JO.index(k) for k in KEYS]
    def sact(r):
        v = [r["joints"].get(k, {}).get("act") for k in KEYS]
        return None if any(x is None for x in v) else sum(v)
    sf = [(r["t"], sact(r)) for r in stiff if sact(r) is not None]
    hw = getattr(align_stiff, "hw", None)
    if hw:
        bb = [(t["timestamp"], sum(hw[JO[k]]["stand_zero_deg"] + hw[JO[k]]["direction"] * t["qpos_rad"][k] * R2D for k in KI)) for t in ticks]
    else:
        bb = [(t["timestamp"], sum(t["qpos_rad"][k] * R2D for k in KI)) for t in ticks]
    if len(sf) < 50 or len(bb) < 50: return None, None
    # 블랙박스 구간을 0.05 s 격자로
    tb0, tb1 = bb[0][0], bb[-1][0]
    import bisect
    bt = [x[0] for x in bb]
    def bb_at(t):
        i = bisect.bisect_left(bt, t)
        if i <= 0: return bb[0][1]
        if i >= len(bb): return bb[-1][1]
        return bb[i][1]
    best = (None, -1.0)
    base = guess_unix - sf[0][0]
    for off in [base + k * 0.1 for k in range(-1800, 1801)]:
        xs, ys = [], []
        for t, v in sf[::3]:
            u = t + off
            if tb0 <= u <= tb1: xs.append(v); ys.append(bb_at(u))
        if len(xs) < 40: continue
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0: continue
        c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
        if c > best[1]: best = (off, c)
    return best


def contact_first_off(ticks, min_stable_s=2.0):
    """안정창(양발 접촉 연속 ≥ min_stable_s) 뒤 처음으로 한 발이라도 접촉 0 이 된 틱 인덱스와 안정창 (i0,i1)."""
    run0 = None
    for i, t in enumerate(ticks):
        c = tuple(t.get("contact") or (0, 0))
        if c == (1.0, 1.0):
            if run0 is None: run0 = i
            if (t["timestamp"] - ticks[run0]["timestamp"]) >= min_stable_s:
                # 이후 첫 이탈
                for j in range(i + 1, len(ticks)):
                    if tuple(ticks[j].get("contact") or (0, 0)) != (1.0, 1.0):
                        return j, (run0, j - 1)
                return None, (run0, len(ticks) - 1)
        else:
            run0 = None
    return None, (None, None)


def fall_index(ticks, i_from, roll_deg=10.0):
    for j in range(i_from, len(ticks)):
        if abs(ticks[j]["rp_deg"][0]) >= roll_deg or abs(ticks[j]["rp_deg"][1]) >= roll_deg: return j
    return len(ticks) - 1


def report(bb_path, stiff_path=None, pads_path=None, hwmap=None, out=print, bin_s=0.5):
    ticks, events = load_bb(bb_path)
    if not ticks: out("정책 틱 없음"); return
    rc_n, rc_mean, rc_max, rc_warn = strip_roll_comp(ticks)
    t0 = ticks[0]["timestamp"]
    i_off, (s0, s1) = contact_first_off(ticks)
    if i_off is None:
        out("첫 접촉 OFF 없음(양발 접촉이 끝까지 유지) — 분석 창을 만들 수 없음"); return
    i_fall = fall_index(ticks, i_off)
    stops = [e for e in events if e.get("event") == "stop_all"]
    w0 = max(0, i_off - int(3.0 / TICK)); w1 = min(len(ticks) - 1, i_fall + int(1.0 / TICK))
    stable = ticks[s0:s1 + 1]
    # 안정창 기준값
    raw_ref = [st.mean(t["raw_action"][k] for t in stable) for k in range(12)]
    tgt_ref = [st.mean(tgt_pre(t, k) for t in stable) for k in range(12)]
    q_ref = [st.mean(t["qpos_rad"][k] * R2D for t in stable) for k in range(12)]
    out(f"# 보정 전달 사슬: {os.path.basename(bb_path)}")
    out(f"정책 틱 {len(ticks)}개 ({ticks[-1]['timestamp'] - t0:.1f} s), 정지 이벤트: {[(e.get('reason') or '')[:30] for e in stops]}")
    if rc_n:
        out(f"보정기 roll_comp 작동: 틱 {rc_n}/{len(ticks)} · |오프셋| 평균 {rc_mean:.2f}° (최대 {rc_max:.2f}°) — "
            f"전달비 계산에서는 이 오프셋을 뺀 '정책 체인 목표'를 쓴다(추종오차는 최종 명령 기준)"
            + (f" · ⚠ {rc_warn}" if rc_warn else ""))
    out(f"안정창(양발 접촉 연속): {ticks[s0]['timestamp'] - t0:.2f}~{ticks[s1]['timestamp'] - t0:.2f} s ({ticks[s1]['timestamp'] - ticks[s0]['timestamp']:.1f} s) | 첫 접촉 OFF: {ticks[i_off]['timestamp'] - t0:.2f} s (contact={ticks[i_off].get('contact')}) | 롤/피치 10° 도달: {ticks[i_fall]['timestamp'] - t0:.2f} s")
    # 패드 / stiff
    pads = load_pads(pads_path, ticks[w0]["timestamp"] - 1, ticks[w1]["timestamp"] + 1) if pads_path else []
    stiff = load_stiff(stiff_path) if stiff_path else []
    hw = None
    if hwmap and os.path.exists(hwmap):
        m = json.load(open(hwmap)); hw = m.get("joints", m)
    align_stiff.hw = hw
    off, corr = (None, None)
    if stiff:
        # 파일명 stiff_cop_HHMM 로 대략 시작시각 추정 → 같은 날짜
        hhmm = os.path.basename(stiff_path).split("_")[-1].split(".")[0]
        day = time.strftime("%Y-%m-%d", time.localtime(t0))
        try: guess = time.mktime(time.strptime(f"{day} {hhmm[:2]}:{hhmm[2:4]}:00", "%Y-%m-%d %H:%M:%S"))
        except ValueError: guess = t0 - 600
        off, corr = align_stiff(stiff, ticks, guess)
        out(f"stage8 로그(stiff) 정렬: 오프셋 {off:.2f} s, 발력합 상호상관 {corr:.3f}" if off is not None else "stage8 로그(stiff) 정렬 실패 — 제어기 수락 명령은 미기록 취급")
    # 프레임 확인: stiff act(모터 절대) → 관절 deg = DIR×(act−SZ) vs 블랙박스 qpos
    if stiff and off is not None and hw:
        errs = []
        import bisect
        bt = [t["timestamp"] for t in ticks]
        for r in stiff:
            u = r["t"] + off
            if not (ticks[w0]["timestamp"] <= u <= ticks[w1]["timestamp"]): continue
            i = min(bisect.bisect_left(bt, u), len(ticks) - 1)
            for k, name in enumerate(JO):
                j = r["joints"].get(name) or {}
                if j.get("act") is None: continue
                jd = hw[name]["direction"] * (j["act"] - hw[name]["stand_zero_deg"])
                errs.append(abs(jd - ticks[i]["qpos_rad"][k] * R2D))
        out(f"프레임 정합(stiff act→관절 deg vs 블랙박스 qpos): |Δ| 중앙 {st.median(errs):.2f}° / p95 {sorted(errs)[int(0.95 * len(errs)) - 1]:.2f}° (n={len(errs)})" if errs else "프레임 정합: 표본 없음")
    # 빈 표
    groups = {"hip_a": (2, 3), "hip_r": (4, 5), "ankle_r": (10, 11), "hip_f": (0, 1), "knee": (6, 7), "ankle_f": (8, 9)}
    out(f"\n## 시간축 표 (첫 OFF 기준 s; Δ = 안정창 평균 대비; 정책출력 Δ = 0.25·Δraw [deg 환산]; 전달비 = Δ목표/Δ정책출력)")
    out("   t    | 패드 L(실)/R on | 접촉비트 | actor접촉(최신) | roll  pitch |ω|  | blend hang        | Δ정책출력 deg: hip_a L/R  hip_r L/R  ankle_r L/R | Δ목표 deg 같은순 | 추종오차(목표−실제) deg 같은순 | stage8 des−목표(모터) | 비고")
    tb = ticks[i_off]["timestamp"]
    import bisect
    pt = [p["timestamp"] for p in pads]
    stiff_by_t = [(r["t"] + off, r) for r in stiff] if (stiff and off is not None) else []
    sti = [x[0] for x in stiff_by_t]
    summary = {"raw_max": [0.0] * 12, "tgt_max": [0.0] * 12, "trk_max": [0.0] * 12, "deliv_max": [0.0] * 12, "clamp": 0, "osc": False, "blend_min": 1.0, "hang": set(), "stop_in_win": False}
    tcur = ticks[w0]["timestamp"]
    while tcur < ticks[w1]["timestamp"]:
        seg = [t for t in ticks if tcur <= t["timestamp"] < tcur + bin_s]
        if not seg: tcur += bin_s; continue
        rel = tcur - tb
        # 패드
        ps = [p for p in pads if tcur <= p["timestamp"] < tcur + bin_s]
        if ps:
            lon = st.mean(len(pads_on(p)[0]) for p in ps); ron = st.mean(len(pads_on(p)[1]) for p in ps); pad_s = f"{lon:.1f}/{ron:.1f}"
        else: pad_s = "미기록" if not pads_path else "-"
        cb = Counter_mode([tuple(t.get("contact") or (0, 0)) for t in seg])
        actor = Counter_mode([tuple(t["obs"][243:245]) if t.get("obs") else ("미기록",) for t in seg])
        roll = st.mean(t["rp_deg"][0] for t in seg); pitch = st.mean(t["rp_deg"][1] for t in seg)
        w = st.mean(math.sqrt(t["gyro_rad_s"][0] ** 2 + t["gyro_rad_s"][1] ** 2) * R2D for t in seg)
        blend = min(t.get("blend_alpha", 1.0) for t in seg); hang = seg[-1].get("hang_state", "?")[:10]
        summary["blend_min"] = min(summary["blend_min"], blend); summary["hang"].add(hang)
        d_raw = [st.mean(t["raw_action"][k] for t in seg) - raw_ref[k] for k in range(12)]
        d_out = [ACTION_SCALE * d * R2D for d in d_raw]
        d_tgt = [st.mean(tgt_pre(t, k) for t in seg) - tgt_ref[k] for k in range(12)]
        trk = [st.mean(t["targets_deg_joint"][JO[k]] - t["qpos_rad"][k] * R2D for t in seg) for k in range(12)]
        for k in range(12):
            summary["raw_max"][k] = max(summary["raw_max"][k], abs(d_out[k])); summary["tgt_max"][k] = max(summary["tgt_max"][k], abs(d_tgt[k])); summary["trk_max"][k] = max(summary["trk_max"][k], abs(trk[k]))
        clamp = sum(sum((t.get("soft_clamp_counts") or {}).values()) if isinstance(t.get("soft_clamp_counts"), dict) else 0 for t in seg)
        summary["clamp"] += clamp; summary["osc"] = summary["osc"] or any(t.get("osc_tripped") for t in seg)
        # stage8 des vs 목표(모터 프레임)
        dl = "미기록"
        if stiff_by_t:
            i0 = bisect.bisect_left(sti, tcur); i1 = bisect.bisect_left(sti, tcur + bin_s)
            ss = [r for _, r in stiff_by_t[i0:i1]]
            if ss:
                vals = []
                for k in (2, 3, 4, 5, 10, 11):
                    name = JO[k]
                    des = [r["joints"].get(name, {}).get("des") for r in ss]; base = [r["joints"].get(name, {}).get("base") for r in ss]
                    des = [d for d in des if d is not None]; base = [b for b in base if b is not None]
                    if des and base:
                        s8_abs = st.mean(des) + st.mean(base)
                        tm = st.mean(t["targets_deg_motor"][name] for t in seg)
                        vals.append(s8_abs - tm); summary["deliv_max"][k] = max(summary["deliv_max"][k], abs(s8_abs - tm))
                dl = " ".join(f"{v:+.1f}" for v in vals) if vals else "-"
        note = []
        if clamp: note.append(f"클램프{clamp}")
        stop_here = [e for e in stops if tcur <= e["timestamp"] < tcur + bin_s]
        if stop_here: note.append("STOP:" + (stop_here[0].get("reason") or "")[:18]); summary["stop_in_win"] = True
        f6 = lambda arr: " ".join(f"{arr[k]:+5.1f}" for k in (2, 3, 4, 5, 10, 11))
        out(f"{rel:+6.1f} | {pad_s:>9} | {cb} | {actor} | {roll:+5.1f} {pitch:+5.1f} {w:4.1f} | {blend:4.2f} {hang:10s} | {f6(d_out)} | {f6(d_tgt)} | {f6(trk)} | {dl} | {' '.join(note)}")
        tcur += bin_s
    # 요약·판정
    out("\n## 요약 (창 내 최대 |Δ|, deg)")
    out("관절        | Δ정책출력 | Δ목표 | 전달비 | 추종오차 | stage8 수락−목표")
    for k, name in enumerate(JO):
        ratio = (summary["tgt_max"][k] / summary["raw_max"][k]) if summary["raw_max"][k] > 0.3 else float("nan")
        dv = f"{summary['deliv_max'][k]:.1f}" if stiff_by_t and k in (2, 3, 4, 5, 10, 11) else "미기록"
        out(f"{name:18s} | {summary['raw_max'][k]:6.1f} | {summary['tgt_max'][k]:5.1f} | {ratio:5.2f} | {summary['trk_max'][k]:6.1f} | {dv}")
    big_raw = max(summary["raw_max"]); big_tgt = max(summary["tgt_max"]); big_trk = max(summary["trk_max"])
    lat_raw = max(summary["raw_max"][k] for k in (2, 3, 4, 5, 10, 11))
    out(f"\n블렌드 최소 {summary['blend_min']:.2f} · hang_state {sorted(summary['hang'])} · 소프트클램프 {summary['clamp']} · 발진가드 {summary['osc']} · 창 내 정지 {summary['stop_in_win']}")
    out("\n## 판정 (규칙: ① 정책출력 Δ 최대 < 2° 이면 '정책 원출력 미보정'; ② 전달비 < 0.5 또는 클램프/동결/블렌드<1 이면 '체인에서 변형'; ③ 추종오차 최대 > 3° 이면 '관절 미추종')")
    v = []
    v.append(f"① 정책 원출력: 최대 Δ {big_raw:.1f}° (좌우축 hip_a/hip_r/ankle_r 최대 {lat_raw:.1f}°) → " + ("**보정 출력 거의 없음**" if big_raw < 2.0 else "보정 출력 있음"))
    ratios = [summary["tgt_max"][k] / summary["raw_max"][k] for k in range(12) if summary["raw_max"][k] > 1.0]
    v.append(f"② 체인: 전달비 {('/'.join(f'{r:.2f}' for r in ratios) if ratios else 'n/a(출력 작음)')} · 블렌드 최소 {summary['blend_min']:.2f} · 클램프 {summary['clamp']} · 정지 {summary['stop_in_win']} → " + ("**체인 변형 의심**" if (ratios and min(ratios) < 0.5) or summary["clamp"] or summary["blend_min"] < 0.99 or summary["stop_in_win"] else "체인 정상(목표 = 출력 추종)"))
    v.append(f"③ 추종: 최대 추종오차 {big_trk:.1f}° → " + ("**관절 미추종**" if big_trk > 3.0 else "관절 추종 정상"))
    for x in v: out("  " + x)
    out("  주의: '보정 출력 없음'은 정책이 IMU·관절 입력에 대해 큰 목표 변화를 내지 않았다는 측정이며, 그 출력이 '충분했는가'의 판정이 아니다. 접촉 OFF 가 출력에 준 영향은 접촉 유지 재생(replay --contact_hold) 로 별도 검사.")


def Counter_mode(items):
    from collections import Counter
    c = Counter(items); k, n = c.most_common(1)[0]
    return f"{k}" if n == len(items) else f"{k}({n}/{len(items)})"


def selftest():
    import tempfile
    rows = []; t0 = 1000.0
    for i in range(1000):
        t = t0 + i * TICK
        contact = [1.0, 1.0] if i < 400 or i >= 990 else ([1.0, 0.0] if i < 500 else [0.0, 0.0])
        roll = 0.0 if i < 450 else min(15.0, (i - 450) * 0.05)
        raw = [0.0] * 12; raw[2] = 0.0 if i < 420 else 0.3    # hip_a 보정 출력 (0.3·0.25 rad = 4.3°)
        tgt = {JO[k]: DEF_DEG[k] + ACTION_SCALE * raw[k] * R2D * (0.9 if i >= 420 else 1.0) for k in range(12)}
        q = [math.radians(tgt[JO[k]]) for k in range(12)]
        rows.append({"live": True, "timestamp": t, "rp_deg": [roll, -3.0], "gyro_rad_s": [0, 0, 0], "raw_action": raw, "targets_deg_joint": tgt,
                     "targets_deg_motor": tgt, "qpos_rad": q, "contact": contact, "foot_n": [30, 30], "blend_alpha": 1.0, "hang_state": "CONTACT_ACQUIRED", "obs": [0.0] * 243 + contact + [0.0] * 75})
    p = os.path.join(tempfile.mkdtemp(), "bb.jsonl")
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    buf = []; report(p, out=buf.append)
    txt = "\n".join(buf)
    assert "첫 접촉 OFF: 8.00 s" in txt, txt[:500]
    assert "보정 출력 있음" in txt and "체인 정상" in txt and "관절 추종 정상" in txt, txt[-800:]
    # 0913 F7: 보정기 오프셋이 '정책이 보정한 것'으로 집계되지 않는지 — 정책 출력은 고정,
    # ankle_r 목표에만 보정기 오프셋 3° 램프. 제거 후 Δ목표 ≈ 0 이어야 한다.
    rows2 = []
    for i in range(1000):
        t = t0 + i * TICK
        contact = [1.0, 1.0] if i < 400 or i >= 990 else ([1.0, 0.0] if i < 500 else [0.0, 0.0])
        roll = 0.0 if i < 450 else min(15.0, (i - 450) * 0.05)
        raw = [0.0] * 12
        off = 0.0 if i < 420 else min(3.0, (i - 420) * 0.02)       # 보정기 오프셋 [deg]
        pre = [DEF_DEG[k] for k in range(12)]
        tgt = {JO[k]: pre[k] + (off if k in (10, 11) else 0.0) for k in range(12)}
        q = [math.radians(tgt[JO[k]]) for k in range(12)]
        rows2.append({"live": True, "timestamp": t, "rp_deg": [roll, -3.0], "gyro_rad_s": [0, 0, 0], "raw_action": raw,
                      "targets_deg_joint": tgt, "targets_deg_motor": tgt, "qpos_rad": q, "contact": contact,
                      "foot_n": [30, 30], "blend_alpha": 1.0, "hang_state": "CONTACT_ACQUIRED",
                      "obs": [0.0] * 243 + contact + [0.0] * 75,
                      "roll_comp_deg": off, "roll_comp_gate": True, "roll_comp_raw_gate": True,
                      "roll_comp_joints": "ankle_r", "targets_deg_joint_pre_rc": pre})
    p2 = os.path.join(tempfile.mkdtemp(), "bb_rc.jsonl")
    with open(p2, "w") as f:
        for r in rows2: f.write(json.dumps(r) + "\n")
    buf2 = []; report(p2, out=buf2.append)
    txt2 = "\n".join(buf2)
    assert "보정기 roll_comp 작동" in txt2 and "|오프셋| 평균" in txt2, txt2[:600]
    row = [l for l in buf2 if l.startswith("left_ankle_r_joint")][0]
    assert float(row.split("|")[2]) < 0.1, f"보정기 오프셋이 Δ목표로 새어 들어감: {row}"
    # pre 미기록(구 기록) 경로: 같은 로그에서 pre 를 빼면 가정·경고와 함께 같은 결과
    for r in rows2: r.pop("targets_deg_joint_pre_rc")
    p3 = os.path.join(tempfile.mkdtemp(), "bb_rc_nopre.jsonl")
    with open(p3, "w") as f:
        for r in rows2: f.write(json.dumps(r) + "\n")
    buf3 = []; report(p3, out=buf3.append)
    assert "⚠" in "\n".join(buf3) and float([l for l in buf3 if l.startswith("left_ankle_r_joint")][0].split("|")[2]) < 0.1
    print("[selftest] analyze_correction_chain: 안정창·첫 OFF·낙상 창·Δ출력/Δ목표/추종·판정 ①②③ + 보정기 오프셋 제거(기록/가정 2경로) — PASS")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("bb", nargs="?"); ap.add_argument("--stiff"); ap.add_argument("--pads"); ap.add_argument("--hwmap", default="config/robot_12dof_hardware_map.json"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    if not a.bb: ap.error("bb 경로 필요")
    report(a.bb, a.stiff, a.pads, a.hwmap)


if __name__ == "__main__":
    main()
