"""C5 낙상 짝 비교 (seed 1–4, 같은 DR 표본): 사용 python3 fall_pool.py <태그A> <태그B> [조건...]"""
import sys, csv, math, os
A, B = sys.argv[1], sys.argv[2]; CONDS = sys.argv[3:] or ["sf10", "sf12", "sf15"]
def rows(t):
    p = f"/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{t}_envs.csv"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None
def wil(s, n):
    if n == 0: return "—"
    p = s / n; z = 1.96; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f"{100*p:.1f} % [{100*(c-h):.1f}, {100*(c+h):.1f}]"
for c in CONDS:
    sa = sb = n = oa = ob = 0; per = []
    for sd in range(1, 9):
        ra, rb = rows(f"{A}_{c}_s{sd}"), rows(f"{B}_{c}_s{sd}")
        if ra is None or rb is None: continue
        fa_ = fb_ = nn = 0
        _keys = [k for k in ra[0] if k.startswith("bias_") or k.startswith("com_")]
        _bad = sum(1 for x, y in zip(ra, rb) if any(x[k] != y[k] for k in _keys if not k.startswith("bias_left_hip_f") and not k.startswith("bias_right_hip_f")))
        _badh = sum(1 for x, y in zip(ra, rb) if any(x[k] != y[k] for k in _keys))
        if _bad:
            print(f"  ⚠ DR 표본 불일치 seed {sd}: {_bad}/{len(ra)} env (힙 앞뒤 외 열) — 짝 비교 무효", flush=True)
        elif _badh:
            per.append(f"[s{sd} 힙 편향만 다름 {_badh}]")
        for x, y in zip(ra, rb):
            if x["pre_pass"] != "1" or y["pre_pass"] != "1": continue
            fa, fb = x["fell"] == "1", y["fell"] == "1"; nn += 1; fa_ += fa; fb_ += fb; oa += fa and not fb; ob += fb and not fa
        sa += fa_; sb += fb_; n += nn; per.append(f"s{sd} {fa_}/{fb_}")
    m = oa + ob; p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(oa, ob) + 1)) / 2**m) if m else 1.0
    print(f"{c}: {A} {wil(sa, n)} · {B} {wil(sb, n)} · n {n} · seed별 {', '.join(per)} · 짝 불일치 {A}만 {oa} / {B}만 {ob} · p {p:.3g}")
