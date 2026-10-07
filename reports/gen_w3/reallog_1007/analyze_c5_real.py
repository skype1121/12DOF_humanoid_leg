"""1004 실물 C5(stand_c5_g21a_g24ra, 감지기 RV3) 회복 모드 활성화별 분석.
측정 사실 = 블랙박스 기록(c5.mode·switch·ref_deg·contact·gyro·quat). 계산값 = 배포본 감지기 재생으로 복원한 기준 대비 기울기·calm 타이머(모드·기준 전 틱 일치로 검증)."""
import sys, pickle, os, json, datetime
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
FILES = ["rl_bridge_20261004_221209.jsonl", "rl_bridge_20261004_221920.jsonl", "rl_bridge_20261004_223349.jsonl"]
LABEL = {"rl_bridge_20261004_221209.jsonl": "시험2(22:12)", "rl_bridge_20261004_221920.jsonl": "시험3(22:19, 손 보조·무효)", "rl_bridge_20261004_223349.jsonl": "시험4(22:33)"}
OFF_TILT, OFF_W, ON_TILT, ON_W = 1.5, 0.3, 3.0, 0.5
QUIET_MIN_S = 2.0
out = []
P = lambda s="": (print(s), out.append(s))
wall = lambda ts: datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-4]
ALL = []
for fn in FILES:
    D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb"))
    R = L.replay(D, "rv3")
    assert (R["mode"] == D["mode"]).all()
    t, mode, tilt, w, c = D["t"], D["mode"], R["tilt"], D["wxy"], D["c"] > 0.5
    both = c[:, 0] & c[:, 1]; anyc = c[:, 0] | c[:, 1]; none = ~anyc; Lonly = c[:, 0] & ~c[:, 1]; Ronly = ~c[:, 0] & c[:, 1]
    free = D["hs"] == "FREE_STAND"
    n = D["n"]
    P(f"\n================ {LABEL[fn]} — {fn} ================")
    P(f"정책 구간 {t[-1]:.2f} s({n} 틱, {wall(D['ts'][0])}–{wall(D['ts'][-1])}) · 회복 모드 {int(mode.sum())} 틱 = {mode.sum()*0.02:.2f} s = 전체의 {100*mode.mean():.1f} % "
      f"· FREE_STAND 구간({free.sum()*0.02:.1f} s) 중 {100*mode[free].mean():.1f} %")
    P(f"접촉 비트(전 구간): 양발 {100*both.mean():.1f} % · 왼발만 {100*Lonly.mean():.1f} % · 오른발만 {100*Ronly.mean():.1f} % · 무접촉 {100*none.mean():.1f} %")
    # 서기 모드 구간의 기준 대비 기울기·|ω| 분포(오탐 여유)
    m0 = (mode == 0) & (t > 1.5)
    P(f"서기 모드(인계 1.5 s 뒤) 기준 대비 기울기: 중앙 {np.median(tilt[m0]):.2f}° p95 {np.percentile(tilt[m0],95):.2f}° 최대 {tilt[m0].max():.2f}° · |ω_xy| 중앙 {np.median(w[m0]):.3f} p95 {np.percentile(w[m0],95):.3f} 최대 {w[m0].max():.3f} rad/s")
    acts = L.activations(mode)
    for a, (i0, i1) in enumerate(acts):
        ended = i1 < n
        seg = slice(i0, i1)
        dur = (t[i1] - t[i0]) if ended else (t[-1] - t[i0] + 0.02)
        trig = []
        if tilt[i0] > ON_TILT: trig.append(f"기울기 {tilt[i0]:.2f}°>3")
        if w[i0] > ON_W: trig.append(f"|ω| {w[i0]:.2f}>0.5")
        q_t, q_w, q_c = tilt[seg] < OFF_TILT, w[seg] < OFF_W, both[seg]
        quiet = q_t & q_w & q_c
        k = i1 - i0
        blk = {"기울기만": int((~q_t & q_w & q_c).sum()), "|ω|만": int((q_t & ~q_w & q_c).sum()), "접촉(양발 아님)만": int((q_t & q_w & ~q_c).sum()),
               "기울기+|ω|": int((~q_t & ~q_w & q_c).sum()), "기울기+접촉": int((~q_t & q_w & ~q_c).sum()), "|ω|+접촉": int((q_t & ~q_w & ~q_c).sum()),
               "셋 다": int((~q_t & ~q_w & ~q_c).sum()), "조용(calm 누적)": int(quiet.sum())}
        A = dict(file=fn, label=LABEL[fn], idx=a + 1, i0=int(i0), i1=int(i1), t_on=float(t[i0]), wall_on=wall(D["ts"][i0]), dur=float(dur), ended=bool(ended),
                 trig=" · ".join(trig), tilt_on=float(tilt[i0]), w_on=float(w[i0]), max_tilt=float(tilt[seg].max()), t_max_tilt=float(t[i0 + int(np.argmax(tilt[seg]))]),
                 max_w=float(w[seg].max()), max_w3=float(np.linalg.norm(D["gy"][seg], axis=1).max()),
                 ref_on=[float(v) for v in D["ref_log"][i0]], rp_on=[float(v) for v in D["rp"][i0]],
                 c_both=float(both[seg].mean()), c_L=float(Lonly[seg].mean()), c_R=float(Ronly[seg].mean()), c_none=float(none[seg].mean()), blk=blk,
                 off_path=("평온(calm 1.0 s) 경로" if ended else "꺼지지 않음 — 켜진 채 세션 정지"))
        P(f"\n[활성화 {a+1}] 켜짐 t={t[i0]:.2f} s ({wall(D['ts'][i0])}) · 지속 {dur:.2f} s · {'꺼짐 t=%.2f s — 평온 경로(RV3 는 이 경로뿐)' % t[i1] if ended else '세션 끝까지 켜진 채(마지막 틱 t=%.2f s)' % t[-1]}")
        P(f"   켜짐 사유(켜진 틱): {A['trig']} · 켜진 틱 기준 자세 ref=[롤 {A['ref_on'][0]:+.2f}, 피치 {A['ref_on'][1]:+.2f}]° · 자세 rp=[{A['rp_on'][0]:+.2f}, {A['rp_on'][1]:+.2f}]°")
        P(f"   활성 중 최대: 기준 대비 기울기 {A['max_tilt']:.2f}° (t={A['t_max_tilt']:.2f}) · |ω_xy| {A['max_w']:.2f} rad/s · |ω|(3축) {A['max_w3']:.2f}")
        P(f"   활성 중 접촉 비트: 양발 {100*A['c_both']:.0f} % · 왼발만 {100*A['c_L']:.0f} % · 오른발만 {100*A['c_R']:.0f} % · 무접촉 {100*A['c_none']:.0f} %")
        P(f"   꺼짐 조건 틱 분해({k} 틱): " + " · ".join(f"{kk} {vv}({100*vv/k:.0f}%)" for kk, vv in blk.items() if vv))
        # 조용한데 켜진 채: (A) 양발 & |ω|<0.3 연속 ≥ 2 s  (B) 한 발 이상 & |ω|<0.3 연속 ≥ 2 s  (C) |ω|<0.3 만 ≥ 2 s
        segs = {}
        for nm, msk in (("A 양발∧|ω|<0.3", q_w & q_c), ("B 한발이상∧|ω|<0.3", q_w & anyc[seg]), ("C |ω|<0.3(접촉 무관)", q_w)):
            rr = [(s, e) for (s, e) in L.runs(msk) if (e - s) * 0.02 >= QUIET_MIN_S]
            segs[nm] = []
            for (s, e) in rr:
                tl = tilt[i0 + s:i0 + e]; cc = c[i0 + s:i0 + e]
                rpm = np.median(D["rp"][i0 + s:i0 + e], axis=0)
                segs[nm].append(dict(t0=float(t[i0 + s]), len_s=float((e - s) * 0.02), tilt_med=float(np.median(tl)), tilt_min=float(tl.min()), tilt_max=float(tl.max()),
                                     frac_ge15=float((tl >= OFF_TILT).mean()), w_med=float(np.median(w[i0 + s:i0 + e])), w_max=float(w[i0 + s:i0 + e].max()),
                                     both=float((cc[:, 0] & cc[:, 1]).mean()), Lc=float(cc[:, 0].mean()), Rc=float(cc[:, 1].mean()),
                                     rp_med=[float(v) for v in rpm], d_rp=[float(v) for v in (rpm - D["ref_log"][i0])]))
            if segs[nm]:
                for sgm in segs[nm]:
                    P(f"   ▶ 조용한데 켜진 채 [{nm}] t={sgm['t0']:.2f}–{sgm['t0']+sgm['len_s']:.2f} s ({sgm['len_s']:.2f} s): 기준 대비 기울기 중앙 {sgm['tilt_med']:.2f}° (최소 {sgm['tilt_min']:.2f}, 최대 {sgm['tilt_max']:.2f}) · ≥1.5° 틱 {100*sgm['frac_ge15']:.0f} % · "
                      f"|ω_xy| 중앙 {sgm['w_med']:.3f} 최대 {sgm['w_max']:.3f} · 접촉 왼 {100*sgm['Lc']:.0f} % 오른 {100*sgm['Rc']:.0f} % 양발 {100*sgm['both']:.0f} % · 자세−기준 [Δ롤 {sgm['d_rp'][0]:+.2f}, Δ피치 {sgm['d_rp'][1]:+.2f}]°")
            else:
                P(f"   · 조용한데 켜진 채 [{nm}] ≥ {QUIET_MIN_S:.0f} s 구간: 없음 (가장 긴 연속 {max([ (e-s)*0.02 for (s,e) in L.runs(msk)] or [0.0]):.2f} s)")
        A["segs"] = segs
        # 켜진 뒤 처음 |ω|<0.3 이 0.5 s 이어진 시각(정착 추정) → 꺼짐까지 대기
        rr = [(s, e) for (s, e) in L.runs(q_w) if (e - s) >= 25]
        if rr:
            s0 = rr[0][0]; A["t_first_quiet"] = float(t[i0 + s0])
            P(f"   |ω_xy|<0.3 이 처음 0.5 s 이어진 시작 t={t[i0+s0]:.2f} s(켜진 뒤 {t[i0+s0]-t[i0]:.2f} s) → 그 뒤 {'꺼짐까지 %.2f s' % (t[i1]-t[i0+s0]) if ended else '세션 끝까지 %.2f s 동안 안 꺼짐' % (t[-1]-t[i0+s0])}")
        else:
            A["t_first_quiet"] = None
            P("   |ω_xy|<0.3 이 0.5 s 이어진 구간 없음(활성 내내 움직임)")
        # 전후 자세(조용한 서기): 전 = 켜지기 직전 2 s 의 조용(|ω|<0.1·양발) 틱 중앙, 후 = 꺼진 뒤 1–5 s 조용(|ω|<0.1·양발·서기 모드) 틱 중앙
        pre = slice(max(0, i0 - 100), i0); mk = (w[pre] < 0.1) & both[pre]
        A["pre_rp"] = [float(v) for v in np.median(D["rp"][pre][mk], axis=0)] if mk.sum() >= 10 else None
        if ended:
            j1 = acts[a + 1][0] if a + 1 < len(acts) else n
            post = slice(min(n, i1 + 50), min(j1, i1 + 250)); mk2 = (w[post] < 0.1) & both[post]
            A["post_rp"] = [float(v) for v in np.median(D["rp"][post][mk2], axis=0)] if mk2.sum() >= 10 else None
            A["post_n"] = int(mk2.sum()); A["rp_off"] = [float(v) for v in D["rp"][i1]]; A["tilt_off"] = float(tilt[i1])
            A["ref_after5"] = [float(v) for v in D["ref_log"][min(j1 - 1, i1 + 250)]]
            P(f"   전후 자세(편향 보정 [롤, 피치]°): 켜지기 전 2 s 조용 틱 중앙 {None if A['pre_rp'] is None else [round(v,2) for v in A['pre_rp']]} · 기준(켜짐) {[round(v,2) for v in A['ref_on']]} · 꺼진 틱 자세 {[round(v,2) for v in A['rp_off']]}(기준 대비 {A['tilt_off']:.2f}°) · "
              f"꺼진 뒤 1–5 s 조용 틱 중앙 {None if A['post_rp'] is None else [round(v,2) for v in A['post_rp']]}(n={A['post_n']}) → 기준(켜짐) 대비 "
              f"{'—' if A['post_rp'] is None else '%.2f°' % float(np.linalg.norm(np.array(A['post_rp'])-np.array(A['ref_on'])))}")
        else:
            A["post_rp"] = None
            P(f"   전 자세: 켜지기 전 2 s 조용 틱 중앙 {None if A['pre_rp'] is None else [round(v,2) for v in A['pre_rp']]} · 기준(켜짐) {[round(v,2) for v in A['ref_on']]} · 꺼짐 없음(후 자세 없음)")
        ALL.append(A)
    # 세션 종료 문맥
    P(f"세션 끝: 마지막 틱 t={t[-1]:.2f} s mode={int(mode[-1])} · 사건 {[(round(e.get('timestamp',0)-D['ts'][0],2), e.get('event'), (e.get('why') or e.get('reason') or e.get('state') or '')) for e in D['ev'] if e.get('event') in ('stop_all','lift_guard','osc_guard','osc_guard_tgt','hang_guard')]}")
json.dump(ALL, open(os.path.join(L.S, "activations_1004.json"), "w"), ensure_ascii=False, indent=1)
open(os.path.join(L.S, "analyze_c5_real.txt"), "w").write("\n".join(out) + "\n")
