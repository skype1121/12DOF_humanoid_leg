"""W3→KE 전환 평가 요약 (TRANSITION_W3KE.md §3 R1–R6) — log_picture/ground_recovery_tr_<mode>_<cond>_d<delay>_s<seed>_envs.csv 를 읽어
팔·조건·지연별 표와 사전 규칙 판정을 낸다. 짝(같은 seed·env, RNG 지문 bias/com 일치)이면 McNemar, 아니면 Fisher.
사용: python3 reports/gen_w3/trans_summ.py [--prefix tr] [--out reports/gen_w3/trans_result.md]
"""
import argparse, csv, glob, math, os, re
from collections import defaultdict
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LP = os.path.join(REPO, "log_picture")
MODES = ("restart", "reset", "direct", "latch", "latchd", "resetnow", "directnow", "nostop", "none", "walk")
f = lambda x: None if x in ("", None) else float(x)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fisher2(a, b, c, d):
    """2×2 [[a,b],[c,d]] 양측 Fisher 정확 검정(초기하, 더 극단 = 확률 ≤ 관측)."""
    n = a + b + c + d; r1 = a + b; c1 = a + c
    lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    def pr(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = pr(a); return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def load(prefix):
    rows = defaultdict(dict)   # (mode, cond, delay, seed) -> env -> row
    pat = re.compile(rf"ground_recovery_{prefix}_(\w+?)_(\w+?)_d(\d)_s(\d+)_envs\.csv$")
    for p in sorted(glob.glob(os.path.join(LP, f"ground_recovery_{prefix}_*_envs.csv"))):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        mode, cond, d, s = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(mode, cond, d, s)][int(r["env"])] = r
    return rows


def fp(r):
    """RNG 지문: 편향·CoM 열(bias_*·com_*)·pre_pitch."""
    return tuple(round(float(r[k]), 4) for k in r if k.startswith(("bias_", "com_")) and r[k] != "") + (round(float(r["pre_pitch"]), 2),)


def agg(rs, dist_k=1):
    """사전검사 통과 env 집계."""
    P = [r for r in rs if r["pre_pass"] == "1"]; n = len(P)
    g = lambda key: np.array([f(r[key]) for r in P], dtype=float)
    cnt = lambda key: int(sum(1 for r in P if r.get(key) == "1"))
    ke = np.array([r.get("tr_sw", "") != "" for r in P])
    out = dict(n=n, fell_post=cnt("fell"), fell_walk=cnt("fell_walk"), fell_win=cnt("fell_win"), fell_sw2=cnt("fell_sw2"), fell_sw5=cnt("fell_sw5"),
               ke_n=int(ke.sum()), geoff_n=int(sum(1 for r in P if r.get("tr_geoff", "") != "")), settle_to=cnt("tr_settle_to"),
               dist_fell=cnt(f"d{dist_k}_fell") if P and f"d{dist_k}_fell" in P[0] else None,
               posture_ok=sum(1 for r in P if r.get("nrm_t", "") != "" and (f(r.get("abn_frac")) or 0.0) <= 0.1 and r.get("end_both", "") in ("1", "1.0")) if P and "nrm_t" in P[0] else None,
               lifts=float(np.nanmedian(g("walk_lifts"))) if n else float("nan"))
    for k in ("pk_p2", "pk_r2", "pk_w2", "pk_pw", "pk_ww", "da_sw", "dcmd_sw", "cmd_jump0", "nrm_sw_t", "ke_first_on"):
        v = g(k) if n else np.array([])
        v = v[~np.isnan(v)]
        out[k + "_med"] = float(np.median(v)) if v.size else float("nan"); out[k + "_p95"] = float(np.percentile(v, 95)) if v.size else float("nan")
        out[k + "_max"] = float(v.max()) if v.size else float("nan")
    v = g("nrm_sw_t") if n else np.array([]); out["nrm5"] = float(np.mean((v <= 5.0) & ~np.isnan(v))) if v.size else float("nan")
    v = g("ke_on_pre") if n else np.array([]); out["ke_on_pre"] = float(np.mean(v > 0)) if v.size else float("nan")
    v = g("tr_settle") - g("tr_geoff") if n else np.array([]); v = v[~np.isnan(v)]; out["settle_med"] = float(np.median(v)) if v.size else float("nan")
    v = g("tr_sw") if n else np.array([]); v = v[~np.isnan(v)]; out["tsw_med"] = float(np.median(v)) if v.size else float("nan")
    return out


def pair_counts(A, B, key):
    """같은 seed·env 에서 A 만 / B 만 (key == '1'), 짝 가능 여부(지문)."""
    b = c = both = n = 0; fpok = True
    for k, ea in A.items():
        eb = B.get(k)
        if eb is None:
            continue
        for e, ra in ea.items():
            rb = eb.get(e)
            if rb is None or ra["pre_pass"] != "1" or rb["pre_pass"] != "1":
                continue
            if fp(ra) != fp(rb):
                fpok = False
            n += 1; xa = ra.get(key) == "1"; xb = rb.get(key) == "1"
            b += xa and not xb; c += xb and not xa; both += xa and xb
    return b, c, both, n, fpok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="tr"); ap.add_argument("--out", default=os.path.join(REPO, "reports", "gen_w3", "trans_result.md"))
    a = ap.parse_args(); rows = load(a.prefix)
    L = []; P = L.append
    P(f"# 전환 평가 요약 ({a.prefix}, 자동 생성 trans_summ.py) — 파일 {len(rows)} 개")
    conds = sorted({k[1] for k in rows}); delays = sorted({k[2] for k in rows})
    for d in delays:
        for cond in conds:
            P(f"\n## 지연 {d} · 조건 {cond} (사전검사 통과 env 합, seed 합)")
            P("| 팔 | seed | n | 걷기 중 낙상 | ge0~KE 창 낙상 | KE 뒤 2 s | KE 뒤 5 s | 외란 뒤 낙상 | 인계 뒤 낙상 합 | 정착 중앙 s(상한 초과) | t_sw 중앙 | 피크 |피치|/|롤|/|ω| 중앙(p95) | |Δaction| 중앙/max | |Δ목표| 중앙/max °/틱 | 첫 KE 점프 중앙/max ° | 감지기 오탐 % | 정상 자세 중앙 s(5 s 안 %) | stop 전 이륙 중앙 |")
            P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for mode in MODES:
                keys = sorted(k for k in rows if k[0] == mode and k[1] == cond and k[2] == d)
                if not keys:
                    continue
                rs = [r for k in keys for r in rows[k].values()]; o = agg(rs)
                lo, hi = wilson(o["fell_sw5"], o["n"])
                P(f"| {mode} | {','.join(str(k[3]) for k in keys)} | {o['n']} | {o['fell_walk']} | {o['fell_win']} | {o['fell_sw2']} | {o['fell_sw5']} ({100*lo:.1f}–{100*hi:.1f} %) | {o['dist_fell']} | {o['fell_post']} | "
                  f"{o['settle_med']:.2f}({o['settle_to']}) | {o['tsw_med']:.1f} | {o['pk_p2_med']:.1f}({o['pk_p2_p95']:.1f})/{o['pk_r2_med']:.1f}({o['pk_r2_p95']:.1f})/{o['pk_w2_med']:.2f}({o['pk_w2_p95']:.2f}) | "
                  f"{o['da_sw_med']:.3f}/{o['da_sw_max']:.3f} | {o['dcmd_sw_med']:.2f}/{o['dcmd_sw_max']:.2f} | {o['cmd_jump0_med']:.2f}/{o['cmd_jump0_max']:.2f} | {100*o['ke_on_pre']:.1f} | {o['nrm_sw_t_med']:.2f}({100*o['nrm5']:.0f}) | {o['lifts']:.0f} |")
            # 짝 비교(restart 기준)
            base = {k: rows[k] for k in rows if k[0] == "restart" and k[1] == cond and k[2] == d}
            if base:
                P("\n짝 비교(restart 기준, 같은 seed·env, KE 뒤 5 s 낙상 / 외란 뒤 낙상 / 인계 뒤 낙상 전체):")
                for mode in ("reset", "direct", "latch", "latchd", "resetnow", "directnow", "nostop", "none", "walk"):
                    other = {k: rows[k] for k in rows if k[0] == mode and k[1] == cond and k[2] == d}
                    if not other:
                        continue
                    A = {k[3]: v for k, v in base.items()}; B = {k[3]: v for k, v in other.items()}
                    parts = []
                    for key, nm in (("fell_sw5", "KE5s"), ("d1_fell", "외란"), ("fell", "전체")):
                        b, c, both, n, ok = pair_counts(A, B, key)
                        if n == 0:
                            continue
                        p = mcnemar(b, c) if ok else fisher2(b + both, n - b - both, c + both, n - c - both)
                        parts.append(f"{nm}: restart만 {b} / {mode}만 {c} (둘 다 {both}, n {n}, {'McNemar' if ok else 'Fisher(지문 불일치)'} p {p:.3f})")
                    P(f"- restart vs {mode}: " + " · ".join(parts))
    txt = "\n".join(L); print(txt)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
