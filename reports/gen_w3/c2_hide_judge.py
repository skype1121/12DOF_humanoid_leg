"""숨기 유인 분해·최소 유지 집합(CPU, 1007) — c2_landscape(term_w·cand_w 기록판) B1–B5 결과.
사용: python3 c2_hide_judge.py <태그접두> <seed...>
(가) 항별 분해: 외란 env, 보상 매니저 항별 창 합의 짝 평균 차 B1(서기만) − B3(회복) — 유인(+)을 만드는 항 순위.
(나) R1d(S) = R1 + Σ_{j∈S} 후보 j 의 회복 중(ge 1) 복제 기여. 후보 = 상태 게이트 서기 양(+) 항 3(무릎 굽힘·양발 접촉·정착 정숙 — 부작용 없는 복제)
     + 상태 읽기 항 3(명령 추종 2·생존). 사전 기준(1007, 실행 전): B1 − B3 ≤ 0(점 추정) 이고 그 R1d 로 기준 3(B3>B5>B2·B3−B2 ≥ 1)·5(B4−B3 ≤ 0) 유지 —
     만족하는 집합 중 크기 최소(동률이면 회복 중 추가 보상 합이 작은 것)를 '회복 중 유지할 항' 후보로."""
import sys, os, itertools
import numpy as np
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
pfx = sys.argv[1]; seeds = sys.argv[2:] or ["1"]


def load(arm):
    parts = []
    for sd in seeds:
        p = f"{LP}/c2unit_{pfx}_{arm}_s{sd}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); parts.append({k: z[k] for k in z.files})
    if not parts:
        return None
    out = {}
    for k in parts[0]:
        a = parts[0][k]
        if a.ndim >= 1 and a.shape[0] > 1 and k not in ("term_names", "cand_names"):
            out[k] = np.concatenate([q[k] for q in parts])
        else:
            out[k] = a
    return out


R = {a: load(a) for a in ("B1", "B2", "B3", "B4", "B5")}
if any(v is None or "term_w" not in v for v in R.values()):
    sys.exit(f"없는 팔/구버전: {[a for a, v in R.items() if v is None or 'term_w' not in v]}")
valid = np.ones_like(R["B1"]["cat"], dtype=bool)
for v in R.values():
    valid &= v["fell_pre"] < 0.5
dist = (R["B1"]["cat"] > 0.5) & valid
L = [f"숨기 유인 분해 · {pfx} · seed {' '.join(seeds)} · 외란 env {int(dist.sum())}"]
names = [str(x) for x in R["B1"]["term_names"]]
d = (R["B1"]["term_w"][dist] - R["B3"]["term_w"][dist]).mean(0)
tot = d.sum()
L.append(f"(가) 보상 매니저 항 합 B1 − B3 = {tot:+.3f}(R1 기준 차와 같은 축) — 기여 큰 순:")
for j in np.argsort(-np.abs(d))[:14]:
    L.append(f"    {names[j]:28s} {d[j]:+.3f}   (B1 {R['B1']['term_w'][dist][:, j].mean():+.3f} · B3 {R['B3']['term_w'][dist][:, j].mean():+.3f})")
r1 = lambda X: X["other_w"] + X["cap_w"] + X["settle_w"]
cn = [str(x) for x in R["B1"]["cand_names"]]
L.append(f"(나) 후보 회복 중 복제 기여(B3 외란 env 평균): " + " · ".join(f"{cn[j]} {R['B3']['cand_w'][dist][:, j].mean():+.3f}" for j in range(len(cn))))


def pd(A, B, f):
    x = f(R[A])[dist] - f(R[B])[dist]
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


rows = []
for k in range(0, len(cn) + 1):
    for S in itertools.combinations(range(len(cn)), k):
        f = (lambda S_: (lambda X: r1(X) + (X["cand_w"][:, list(S_)].sum(1) if S_ else 0.0)))(S)
        h, hse = pd("B1", "B3", f); d32, _ = pd("B3", "B2", f); d43, _ = pd("B4", "B3", f)
        v3, v5, v2 = f(R["B3"])[dist].mean(), f(R["B5"])[dist].mean(), f(R["B2"])[dist].mean()
        ok3 = (v3 > v5 > v2) and d32 >= 1.0; ok5 = d43 <= 0.0
        add = R["B3"]["cand_w"][dist][:, list(S)].sum(1).mean() if S else 0.0
        rows.append((h <= 0 and ok3 and ok5, len(S), add, S, h, hse, ok3, d32, ok5, d43))
L.append("    집합 | B1−B3(±SE) | 기준 3 (B3−B2) | 기준 5 (B4−B3) | 회복 중 추가분")
for r in sorted(rows, key=lambda r: (r[1], r[4]))[:20]:
    ok, n, add, S, h, hse, ok3, d32, ok5, d43 = r
    L.append(f"    {'+'.join(cn[j] for j in S) or '(없음=R1)':70s} | {h:+.2f}±{hse:.2f} | {'P' if ok3 else 'F'} {d32:+.2f} | {'P' if ok5 else 'F'} {d43:+.2f} | {add:+.2f}"
             + ("  ← 통과" if ok else ""))
best = [r for r in rows if r[0]]
if best:
    b = sorted(best, key=lambda r: (r[1], r[2]))[0]
    L.append(f"최소 유지 집합: {' + '.join(cn[j] for j in b[3])} (B1−B3 {b[4]:+.2f}±{b[5]:.2f}, 기준 3 B3−B2 {b[7]:+.2f}, 기준 5 B4−B3 {b[9]:+.2f})")
else:
    L.append("최소 유지 집합: 없음 — 6 후보 전부를 유지해도 숨기 유인이 남거나 기준 3·5 가 깨짐")
txt = "\n".join(L); print(txt)
open(f"{LP}/c2hide_judge_{pfx}.txt", "w").write(txt + "\n")
