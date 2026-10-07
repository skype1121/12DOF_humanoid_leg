"""X17 판정(TRANSITION_W3KE.md §5.20) — 전 관절 소프트 한계 클램프(실물 stage8 과 같은 자리) 대조.
사용: python3 reports/gen_w3/x17_judge.py [--seeds 1,2]
C1 회복(KE, 한계 1.0 배 그룹): 조건마다 전체 낙상이 대조(X15) + 3 %p 이하 ∧ env 짝 McNemar p ≥ 0.05. C2 걷기 왕복: s3g_judge 의 표(대조 X7c C0)로 본다.
보고: 관절별 클램프 발동 비율(추적 env 에서 러너 목표각이 soft 한계 밖인 틱 = 클램프가 자른 틱) — 서기(kind 0)·회복 모드(kind 1)·걷기(kind 2)별."""
import argparse, csv, json, math, os, sys
import numpy as np
from math import comb
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LAB = {"nd": "무외란", "b18": "뒤 밀기 18 N", "sf15": "앞 램프 15 N"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def falls(tag):
    pk = os.path.join(LP, f"kwloop_{tag}.npz"); pc = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    if not (os.path.exists(pk) and os.path.exists(pc)):
        return None
    z = np.load(pk, allow_pickle=True); pre = z["pre_pass"].astype(bool); sc = np.round(z["effort_scale"].astype(float), 3) if "effort_scale" in z.files else np.ones(pre.shape[0])
    f = np.zeros(pre.shape[0], bool)
    for r in csv.DictReader(open(pc)):
        if r["fell"] == "1":
            f[int(r["env"])] = True
    return pre, sc, f


def clamp_activity(tag, lim, names):
    pt = os.path.join(LP, f"transall_{tag}.npz"); pk = os.path.join(LP, f"kwloop_{tag}.npz")
    if not (os.path.exists(pt) and os.path.exists(pk)):
        return None
    zt = np.load(pt, allow_pickle=True); zk = np.load(pk, allow_pickle=True); cmd = zt["cmd"].astype(float); n = cmd.shape[1]
    kind = zk["kind"][:, :n]; alive = zt["alive"].astype(bool) if "alive" in zt.files else np.ones(cmd.shape[:2], bool)
    lo = np.array([math.radians(lim[x]["soft_min"]) for x in names]); hi = np.array([math.radians(lim[x]["soft_max"]) for x in names])
    at = (cmd > hi[None, None] + 1e-4) | (cmd < lo[None, None] - 1e-4)      # 추적의 cmd 는 러너가 낸 목표각(클램프 전) — 한계 밖이면 그 틱에 클램프가 자른다
    out = {}
    for kd, nm in ((0, "서기"), (1, "회복"), (2, "걷기")):
        m = (kind == kd) & alive
        out[nm] = (at[m].mean(0) if m.any() else np.full(len(names), np.nan), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="1,2"); a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]
    lim = json.load(open(os.path.join(REPO, "config", "joint_limits_12dof.json"), encoding="utf-8"))["limits_deg"]
    names = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
             "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
    print("### C1 회복(KE) — 전 관절 클램프(X17) 대 대조(X15), env 짝")
    print("| 조건 | 배율 | 짝 n | 낙상 대조 | 낙상 클램프 | 차 | 대조만 / 클램프만 · p |"); print("|---|---|---|---|---|---|---|")
    ok1 = True; miss = []
    for c in ("nd", "b18", "sf15"):
        acc = {}
        for s in seeds:
            A = falls(f"tr_x15_{c}_d2_s{s}"); B = falls(f"tr_x17_{c}_d2_s{s}")
            if A is None or B is None:
                miss.append(f"{c}/s{s}"); continue
            both = A[0] & B[0]
            for g in sorted(set(A[1])):
                m = both & (A[1] == g); d = acc.setdefault(g, [0, 0, 0, 0, 0]); d[0] += int(m.sum()); d[1] += int(A[2][m].sum()); d[2] += int(B[2][m].sum()); d[3] += int((A[2] & ~B[2])[m].sum()); d[4] += int((B[2] & ~A[2])[m].sum())
        for g in sorted(acc, reverse=True):
            n, fa, fb, oa, ob = acc[g]; p = mcnemar(oa, ob); dd = 100 * (fb - fa) / max(n, 1)
            mark = ""
            if abs(g - 1.0) < 1e-6:
                good = dd <= 3.0 + 1e-9 and p >= 0.05; ok1 = ok1 and good; mark = " ← " + ("무해" if good else "**유해**")
            print(f"| {LAB[c]} | {g:g} | {n} | {fa} ({100 * fa / max(n, 1):.1f} %) | {fb} ({100 * fb / max(n, 1):.1f} %) | {dd:+.1f} %p | {oa} / {ob} · {p:.3f}{mark} |")
    print(f"- **C1**: {'자료 부족 ' + ','.join(miss) if miss else ('무해' if ok1 else '유해')}")
    print("### 클램프 발동 비율 — 러너 목표각이 soft 한계 밖이라 잘린 틱의 비율 [%] (추적 env, 클램프 팔)")
    print("| 실행 | 구간(틱 수) | " + " | ".join(x.replace("_joint", "").replace("left_", "L ").replace("right_", "R ") for x in names) + " |"); print("|---|---|" + "---|" * len(names))
    for tag in [f"tr_x17_{c}_d2_s{s}" for c in ("b18", "sf15", "nd") for s in seeds] + [f"tr_x17_ff_d2_s{s}" for s in seeds]:
        ca = clamp_activity(tag, lim, names)
        if ca is None:
            continue
        for nm, (v, cnt) in ca.items():
            if cnt == 0:
                continue
            print(f"| {tag.replace('tr_x17_', '').replace('_d2', '')} | {nm} ({cnt}) | " + " | ".join(("—" if not np.isfinite(x) else (f"**{100 * x:.1f}**" if x >= 0.01 else f"{100 * x:.1f}")) for x in v) + " |")
    print("### C2 걷기 왕복 — 명령줄")
    print("python3 reports/gen_w3/s3g_judge.py --cand tr_x17_ff --ctl tr_x7c_c0 --name 'X17 걷기 왕복(전 관절 클램프) 대 C0'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
