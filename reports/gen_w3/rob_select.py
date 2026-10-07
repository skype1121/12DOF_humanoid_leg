"""ROBUST_DESIGN §3 — envelope JSON(log_picture/w3env_<tag>.json)에서 그룹(ax 조합)별 채점 낙상률·이득을 뽑아 표로 내고, 코너 선택·통과 기준을 적용한다.
사용: python3 rob_select.py table <tag> [<tag> ...]            — 그룹 표
      python3 rob_select.py pick <tag_base> <tag_c1> <tag_c2> … — 코너(imu_pitch=-3, com_x=0.02) 낙상 최소 태그 출력(동률이면 앞)
      python3 rob_select.py judge <base_prefix> <cand_prefix>  — E1/E2/E3 기준 ①–⑥ 판정(태그 규약: <prefix>_{e2,e1f,e1z,e3h,e3k}_s<seed>)
"""
import glob, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LP = os.path.join(REPO, "log_picture")


def load(tag):
    p = os.path.join(LP, f"w3env_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def groups(tag):
    d = load(tag)
    if d is None:
        return None
    out = {}
    for g in d["groups"]:
        key = (g["cmd"],) + tuple(sorted((k, round(float(v), 4)) for k, v in g["ax"].items()))
        out[key] = g
    return out, bool(d.get("completed", True))


def base_key(gs, cmd):
    for k in gs:
        if k[0] == cmd and all(abs(v - (1.0 if a in ("kp", "kd", "kp_L", "kd_L", "play", "coul", "c_hold", "cmd_scale") else (2.0 if a == "delay" else 0.0))) < 1e-9 for a, v in k[1:]):
            return k
    return None


def fr(g):
    return 100.0 * float(g["fall_rate"])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def table(tags):
    for t in tags:
        r = groups(t)
        if r is None:
            print(f"# {t}: 없음"); continue
        gs, ok = r
        print(f"\n# {t} (완주 {ok})\n| cmd | ax | n_score | 낙상 % | 이득 x/y/yaw | alt |\n|---|---|---|---|---|---|")
        for k, g in gs.items():
            print(f"| {g['cmd']} | {g['ax']} | {g['n_score']} | {fr(g):.1f} | {_f(g.get('gain_x')):.2f}/{_f(g.get('gain_y')):.2f}/{_f(g.get('gain_yaw')):.2f} | {_f(g.get('alt')):.3f} |")


def corner(tag, cmd="f06", ax=(("com_x", 0.02), ("imu_pitch", -3.0))):
    r = groups(tag)
    if r is None:
        return None
    gs, _ = r
    for k, g in gs.items():
        if k[0] == cmd and dict(k[1:]) == {a: round(v, 4) for a, v in ax}:
            return fr(g)
    return None


def pick(tags):
    best = None
    for t in tags:
        c = corner(t)
        print(f"{t}: 코너 낙상 {c if c is None else round(c, 1)} %")
        if c is not None and (best is None or c < best[1] - 1e-9):
            best = (t, c)
    print("PICK", best[0] if best else "")


def judge(bp, cp, seed=1):
    def g(prefix, part, cmd, ax):
        r = groups(f"{prefix}_{part}_s{seed}")
        if r is None:
            return None
        gs, _ = r
        for k, gg in gs.items():
            if k[0] == cmd and dict(k[1:]) == {a: round(v, 4) for a, v in ax}:
                return gg
        return None
    res = []
    def chk(name, cond, val, lim, base):
        res.append(cond); print(f"{'PASS' if cond else 'FAIL'} {name}: 후보 {val} (기준 {lim}, 15800 {base})")
    c_b, c_c = g(bp, "e2", "f06", (("com_x", 0.02), ("imu_pitch", -3.0))), g(cp, "e2", "f06", (("com_x", 0.02), ("imu_pitch", -3.0)))
    if c_c: chk("① 코너 −3°×+2 cm 낙상 ≤ 10 %", fr(c_c) <= 10.0, round(fr(c_c), 1), 10, round(fr(c_b), 1) if c_b else None)
    for s in (-6.0, 6.0):
        b, c = g(bp, "e1f", "f06", (("imu_pitch", s),)), g(cp, "e1f", "f06", (("imu_pitch", s),))
        if c: chk(f"② IMU 피치 {s:+.0f}° 낙상 ≤ 5 %", fr(c) <= 5.0, round(fr(c), 1), 5, round(fr(b), 1) if b else None)
    b, c = g(bp, "e1f", "f06", (("com_x", 0.04),)), g(cp, "e1f", "f06", (("com_x", 0.04),))
    if c: chk("③ CoM +4 cm 낙상 ≤ 5 %", fr(c) <= 5.0, round(fr(c), 1), 5, round(fr(b), 1) if b else None)
    b, c = g(bp, "e1f", "f06", (("play", 2.0),)), g(cp, "e1f", "f06", (("play", 2.0),))
    if c: chk("④ 유격 ×2 앞 이득 ≥ 0.6", _f(c.get("gain_x")) >= 0.6, round(_f(c.get("gain_x")), 2), 0.6, round(_f(b.get("gain_x")), 2) if b else None)
    for part, nm in (("e3h", "hipf_L −4°"), ("e3k", "kp_L 1.3")):
        for cmd in ("z", "f06"):
            r = groups(f"{cp}_{part}_s{seed}"); rb = groups(f"{bp}_{part}_s{seed}")
            if r:
                gs, _ = r; cands = [gg for k, gg in gs.items() if k[0] == cmd]
                bb = [gg for k, gg in rb[0].items() if k[0] == cmd] if rb else []
                if cands: chk(f"⑤ 120 s {nm} {cmd} 낙상 ≤ 4 %", fr(cands[0]) <= 4.0, round(fr(cands[0]), 1), 4, round(fr(bb[0]), 1) if bb else None)
    for part in ("e1f", "e1z"):
        cmd = "f06" if part == "e1f" else "z"
        r = groups(f"{cp}_{part}_s{seed}"); rb = groups(f"{bp}_{part}_s{seed}")
        if r and rb:
            kb = base_key(r[0], cmd); kbb = base_key(rb[0], cmd)
            if kb and kbb:
                c, b = r[0][kb], rb[0][kbb]
                chk(f"⑥ 기준점 {cmd} 낙상 ≤ 15800+1 %p", fr(c) <= fr(b) + 1.0, round(fr(c), 1), f"≤{round(fr(b) + 1.0, 1)}", round(fr(b), 1))
                if cmd == "f06": chk("⑥ 기준점 앞 이득 15800 ± 0.15", abs(_f(c.get("gain_x")) - _f(b.get("gain_x"))) <= 0.15, round(_f(c.get("gain_x")), 2), "±0.15", round(_f(b.get("gain_x")), 2))
    print("ALL_PASS" if res and all(res) else "NOT_ALL_PASS", f"({sum(res)}/{len(res)})")


if __name__ == "__main__":
    m = sys.argv[1]
    {"table": lambda: table(sys.argv[2:]), "pick": lambda: pick(sys.argv[2:]), "judge": lambda: judge(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)}[m]()
