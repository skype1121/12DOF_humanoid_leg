"""X14 판정(TRANSITION_W3KE.md §5.18) — 콘솔 걷기 버튼(시간 제한 걷기) 대본의 배포 코드 폐루프 결과.
사용: python3 reports/gen_w3/x14_judge.py [--tags tr_x14_d2_s1,tr_x14_d2_s2] [--name 이름]
B1 수락(요청 뒤 2 s 안 켜짐 ≥ 99 %) · B2 스스로 끝남(기한 + 0.5 s 안 gait 꺼짐 전부 ∧ KE 인계 p95 ≤ 3.5 s) · B3 낙상 ≤ 5 % · B4 방향 부호 일치 ≥ 95 %(제자리 제외).
변위 창(결과를 보기 전에 고정): 요청 시각 → min(다음 요청, 요청 + 기한 + 3 s), 요청 시각의 몸통 방향 기준(앞 +x, 왼쪽 +y, 왼쪽 회전 +요)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rl_walking.deploy import w3_deploy as W3D   # noqa: E402
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="tr_x14_d2_s1,tr_x14_d2_s2"); ap.add_argument("--name", default="")
    ap.add_argument("--moves", default="fwd1,turn_l,turn_r,left1,right1,back1,fwd3,inplace,fwd1,fwd1")
    a = ap.parse_args(); mv = a.moves.split(","); B = len(mv)
    st = [dict(n=0, acc=0, dly=[], off_late=0, off_n=0, off_dly=[], ke=[], ke_none=0, fall=0, sign_ok=0, sign_n=0, mag=[], yawabs=[], lat=[], fwd=[], rec=0) for _ in range(B)]
    N = 0; F = 0; got = []; end_s = None
    for tag in a.tags.split(","):
        pz = os.path.join(LP, f"kwloop_{tag}.npz")
        if not os.path.exists(pz):
            print(f"(없음 {tag})"); continue
        got.append(tag); z = np.load(pz, allow_pickle=True); ev = json.loads(str(z["events"])); script = json.loads(str(z["script"])); assert len(script) == B, (len(script), B)
        t = z["t"].astype(float); ge = z["ge"].astype(bool); pp = z["pre_pass"].astype(bool); fell = z["fell"].astype(bool); tf = np.where(fell, z["t_fall_post"].astype(float), np.inf)
        yaw = np.unwrap(z["yaw"].astype(float), axis=0); xy = z["xy"].astype(float); kind = z["kind"]; end_s = float(t[-1])
        K = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)   # noqa: E731
        treq = [float(s[0]) for s in script]
        for e in np.nonzero(pp)[0]:
            N += 1; F += bool(fell[e] and tf[e] <= end_s + 1); E = ev[e]
            en = [x for x, nm in E if nm == "enable"]; be = [x for x, nm in E if nm == "burst_end"]; tk = [x for x, nm in E if nm == "to_ke"]
            for b in range(B):
                s = st[b]; t0 = treq[b]; t1 = treq[b + 1] if b + 1 < B else end_s; m = W3D.W3_MOVES[mv[b]]; dur = float(m["dur_s"])
                if tf[e] <= t0:
                    continue                                              # 요청 전에 넘어짐 — 분모 제외
                s["n"] += 1; s["fall"] += bool(t0 < tf[e] <= t1); s["rec"] += bool(kind[K(t0), e] == 1)
                acc = [x for x in en if t0 - 1e-6 <= x <= t0 + 2.0 + 1e-6]
                if not acc:
                    continue
                s["acc"] += 1; te = acc[0]; s["dly"].append(te - t0)
                bx = [x for x in be if te < x <= te + dur + 1.0]
                if tf[e] > te + dur + 0.6:                                # 기한 + 0.5 s 까지 살아 있던 것만 '스스로 끝남' 분모
                    s["off_n"] += 1; k_due = K(te + dur); k_lim = K(te + dur + 0.5); g = ge[k_due:k_lim + 1, e]; off = np.nonzero(~g)[0]
                    if len(off) == 0 or not bx:
                        s["off_late"] += 1
                    else:
                        s["off_dly"].append(float(t[k_due + off[0]]) - (te + dur))
                    nk = [x for x in tk if x >= te + dur - 1e-6 and x < t1 + 1e-6]
                    if nk and tf[e] > nk[0]:
                        s["ke"].append(nk[0] - (te + dur))
                    elif tf[e] > t1:
                        s["ke_none"] += 1
                tw = min(t1, t0 + dur + 3.0)
                if tf[e] > tw:
                    k0, k1 = K(t0), K(tw); c0, s0 = np.cos(yaw[k0, e]), np.sin(yaw[k0, e]); d = xy[k1, e] - xy[k0, e]
                    fw = float(d[0] * c0 + d[1] * s0); lt = float(-d[0] * s0 + d[1] * c0); dy = float(np.degrees(yaw[k1, e] - yaw[k0, e]))
                    s["fwd"].append(fw); s["lat"].append(lt); s["yawabs"].append(abs(dy))
                    if abs(m["vx"]) > 0:
                        s["sign_n"] += 1; s["sign_ok"] += bool(np.sign(fw) == np.sign(m["vx"])); s["mag"].append(fw)
                    elif abs(m["vy"]) > 0:
                        s["sign_n"] += 1; s["sign_ok"] += bool(np.sign(lt) == np.sign(m["vy"])); s["mag"].append(lt)
                    elif abs(m["wz"]) > 0:
                        s["sign_n"] += 1; s["sign_ok"] += bool(np.sign(dy) == np.sign(m["wz"])); s["mag"].append(dy)
    if not got:
        return 1
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")   # noqa: E731
    print(f"## X14 {a.name} — 실행 {len(got)} 개({', '.join(got)}), 사전 검사 통과 env {N}")
    print("| # | 버튼 | 명령·기한 | 요청 때 생존 | 수락(2 s 안) | 켜짐 지연 중앙 / p95 [s] | 기한 + 0.5 s 안 꺼짐 | 기한 → 꺼짐 중앙 [s] | 기한 → KE 인계 중앙 / p95 [s] | 구간 낙상 | 방향 부호 일치 | 크기 중앙 (p10–p90) | 요 틀어짐 \\|Δψ\\| 중앙 [°] |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = dict(n=0, acc=0, off_n=0, off_late=0, ke=[], sign_n=0, sign_ok=0); worst_sign = 1.0; per_b_fail = []
    for b, s in enumerate(st):
        m = W3D.W3_MOVES[mv[b]]; unit = "m" if (abs(m["vx"]) > 0 or abs(m["vy"]) > 0) else ("°" if abs(m["wz"]) > 0 else "")
        mag = f"{q(s['mag'], 50):+.3f} ({q(s['mag'], 10):+.3f}–{q(s['mag'], 90):+.3f}) {unit}" if s["mag"] else (f"앞 {q(s['fwd'], 50):+.3f} m · 옆 {q(s['lat'], 50):+.3f} m" if s["fwd"] else "—")
        sg = f"{s['sign_ok']} / {s['sign_n']} ({100 * s['sign_ok'] / max(s['sign_n'], 1):.1f} %)" if s["sign_n"] else "—"
        print(f"| {b + 1} | {m['label']} | {m['vx']:+.3f}, {m['vy']:+.3f}, {m['wz']:+.2f} · {m['dur_s']} s | {s['n']} | {s['acc']} ({100 * s['acc'] / max(s['n'], 1):.1f} %) | {q(s['dly'], 50):.2f} / {q(s['dly'], 95):.2f} | {s['off_n'] - s['off_late']} / {s['off_n']} | {q(s['off_dly'], 50):.2f} | {q(s['ke'], 50):.2f} / {q(s['ke'], 95):.2f}"
              + (f" (미인계 {s['ke_none']})" if s["ke_none"] else "") + f" | {s['fall']} ({100 * s['fall'] / max(s['n'], 1):.1f} %) | {sg} | {mag} | {q(s['yawabs'], 50):.1f} |")
        tot["n"] += s["n"]; tot["acc"] += s["acc"]; tot["off_n"] += s["off_n"]; tot["off_late"] += s["off_late"]; tot["ke"] += s["ke"]; tot["sign_n"] += s["sign_n"]; tot["sign_ok"] += s["sign_ok"]
        if s["sign_n"]:
            r = s["sign_ok"] / s["sign_n"]; worst_sign = min(worst_sign, r)
            if r < 0.95:
                per_b_fail.append(f"{b + 1}:{m['label']} {100 * r:.1f} %")
    b1 = tot["acc"] / max(tot["n"], 1); lo, hi = wilson(F, N); kp95 = q(tot["ke"], 95)
    print(f"- **B1 수락**: {tot['acc']} / {tot['n']} ({100 * b1:.2f} %) — 기준 ≥ 99 % → {'충족' if b1 >= 0.99 else '미달'}" + f" · 요청 때 KE 회복 모드였던 건수 {sum(s['rec'] for s in st)}")
    print(f"- **B2 스스로 끝남**: 기한 + 0.5 s 안 gait 꺼짐 {tot['off_n'] - tot['off_late']} / {tot['off_n']} · 기한 → KE 인계 p95 {kp95:.2f} s (기준 전부 ∧ ≤ 3.5 s) → {'충족' if tot['off_late'] == 0 and kp95 <= 3.5 else '미달'}")
    print(f"- **B3 낙상**: {F} / {N} ({100 * F / max(N, 1):.2f} %, 95 % 구간 {100 * lo:.1f}–{100 * hi:.1f} %) — 기준 ≤ 5 % → {'충족' if F / max(N, 1) <= 0.05 else '미달'}")
    print(f"- **B4 방향**: 버튼별 최저 {100 * worst_sign:.1f} % (전체 {tot['sign_ok']} / {tot['sign_n']}) — 기준 버튼마다 ≥ 95 % → {'충족' if not per_b_fail else '미달: ' + ', '.join(per_b_fail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
