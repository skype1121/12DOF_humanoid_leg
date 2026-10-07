"""STAND_MODE_CMP(1007) — W3S3d STAND 모드(ge 0) vs G21a 단독 vs KE(odicw) 짝 비교. 사전 기준 = STAND_MODE_CMP.md.

사용:
  python3 reports/regress/stand_mode_cmp.py plan            → runs/smc_<시각>/ (jobs.tsv·manifest.json) 경로 출력
  reports/regress/run_regress.sh --resume <dir>              → GPU 실행(잠금 공유·무효 자동 재실행)
  python3 reports/regress/stand_mode_cmp.py analyze <dir>   → <dir>/smc_result.md
"""
import json, math, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regress_lib as RL      # noqa: E402
import summarize as S         # noqa: E402

W3RUN, W3CK = "2026-10-07_00-11-12_w3S3d_from_s2d", "model_15800.pt"
G21 = ("2026-10-02_21-23-08_g21a", "model_11300.pt")
REC = "2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt"
DIST = {"nd": "", "f12": "--dist push:f:2:12:0.3", "b18": "--dist push:b:2:18:0.3", "sf15": "--dist ramp:f:2:15:0.5:1.0", "fi40": "--dist push:f:2:40:0.2"}
ARMS = {
    "w3": f"--load_run {W3RUN} --checkpoint {W3CK} {RL.STAND_CONTRACT} --c2 --c2_arm ge0",
    "g21": f"--load_run {G21[0]} --checkpoint {G21[1]} {RL.STAND_CONTRACT}",
    "ke": f"--load_run {G21[0]} --checkpoint {G21[1]} {RL.STAND_CONTRACT} {RL.KE_SWITCH.format(rec=REC)} {RL.KE_ARMS['odicw']}",
}
MAIN_SEEDS = (1, 2, 3); TR_SEEDS = (1, 2); TR_DELAYS = (3, 4)


def plan():
    rid = "smc" + time.strftime("%m%d_%H%M"); rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", rid + "_stand_mode_cmp")
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    jobs = []
    for sd in MAIN_SEEDS:
        for c in DIST:
            for a in ARMS:
                jobs.append((a, c, 2, sd))
    for d in TR_DELAYS:
        for sd in TR_SEEDS:
            for a in ARMS:
                jobs.append((a, "nd", d, sd))
    J = {}
    with open(os.path.join(rd, "jobs.tsv"), "w") as fh:
        for a, c, d, sd in jobs:
            jid = f"{a}-{c}-d{d}-s{sd}"; tag = f"rg_{rid}_{a}_{c}_d{d}_s{sd}"
            base = RL.GR_BASE.replace("--delay_steps 2,2", f"--delay_steps {d},{d}")
            args = re.sub(r"\s+", " ", f"{base} {ARMS[a]} --seed {sd} {DIST[c]}").strip()
            fh.write("\t".join([jid, "gr", tag, "900", args]) + "\n")
            J[jid] = dict(kind="gr", tag=tag, args=args, item=f"SMC-{a}-{c}-d{d}", seed=sd, cond=c, arm=a, delay=d, num_envs=128)
    ck = {"w3": RL.find_ckpt(W3RUN, W3CK), "g21a": RL.find_ckpt(*G21), "rec": RL.find_ckpt(*REC.split(":"))}
    man = dict(rid=rid, cfg=dict(mode="smc", rid=rid, profile="s3d", w3=f"{W3RUN}:{W3CK}", w3_task="(G11a 하네스, --c2 ge0)", stand=":".join(G21), rec=REC),
               mode_params={}, started=time.strftime("%Y-%m-%d %H:%M:%S"),
               code=dict(worktree=RL.WT, head=RL.sh("git rev-parse HEAD"), branch=RL.sh("git rev-parse --abbrev-ref HEAD"),
                         dirty=RL.sh("git status --porcelain -- rl_walking reports/regress | head -50").splitlines()),
               scripts={p: RL.md5(os.path.join(RL.WT, p)) for p in ("rl_walking/scripts/eval_ground_recovery.py", "rl_walking/w3_c2.py",
                                                                    "reports/regress/STAND_MODE_CMP.md", "reports/regress/stand_mode_cmp.py")},
               checkpoints={k: dict(path=v, md5=RL.md5(v)) for k, v in ck.items()}, data_repo=RL.DATA, jobs=J)
    json.dump(man, open(os.path.join(rd, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(rd)


def fisher(a, n1, b, n2):
    """양측 Fisher 정확 검정(2×2: a/n1 vs b/n2)."""
    K = a + b; N = n1 + n2
    def hp(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)
    p0 = hp(a); lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(hp(x) for x in range(lo, hi + 1) if hp(x) <= p0 * (1 + 1e-9)))


def analyze(rd):
    man = json.load(open(os.path.join(rd, "manifest.json")))
    st = {}
    for l in open(os.path.join(rd, "status.tsv")):
        x = l.rstrip("\n").split("\t"); st[x[0]] = x[1]
    D = {}   # (arm, cond, delay) → {(seed, env): row}
    bad = [j for j in man["jobs"] if st.get(j) != "유효"]
    for jid, J in man["jobs"].items():
        if st.get(jid) != "유효":
            continue
        for r in S.load_csv(J["tag"]) or []:
            D.setdefault((J["arm"], J["cond"], J["delay"]), {})[(J["seed"], r["env"])] = r
    M = {   # 지표: env 행 → bool
        "낙상": lambda r, c: r["fell"] == "1",
        "끝 정상 자세": lambda r, c: S.posture_ok(r, c == "nd"),
        "스텝 발생": lambda r, c: (S.ff(r.get("takeoffs")) or 0) > 0,
        "인계 과도 낙상(<2 s)": lambda r, c: r["fell"] == "1" and (S.ff(r.get("t_fall_rel")) if S.ff(r.get("t_fall_rel")) is not None else 99) < 2.0,
        "최종 정숙": lambda r, c: r["final_quiet"] == "1",
        "회복 모드 사용": lambda r, c: (S.ff(r.get("sw_on_n")) or 0) > 0,
    }
    L = []; P = L.append
    P(f"# STAND_MODE_CMP 결과 — {man['rid']}"); P("")
    P(f"- 코드 `{man['code']['head'][:10]}` · 작업 {len(man['jobs'])} · 무효 {len(bad)} {bad if bad else ''}")
    P("- 분모 = 사전검사 통과 env. 비교 = W3 − 대조(%p), 짝(같은 seed·env, RNG 지문 일치)이면 McNemar(W3만/대조만), 아니면 Fisher. 시뮬 판정 — 실물 확률 아님.")
    P("")

    def pp(arm, c, d):
        R = D.get((arm, c, d), {}); return {k: r for k, r in R.items() if r["pre_pass"] == "1"}

    def cmp(m, c, d, other, conds=None):
        conds = conds or [c]
        A = {}; B = {}
        for cc in conds:
            A.update({(cc,) + k: r for k, r in pp("w3", cc, d).items()}); B.update({(cc,) + k: r for k, r in pp(other, cc, d).items()})
        if not A or not B:
            return None
        fa = sum(M[m](r, k[0]) for k, r in A.items()); fb = sum(M[m](r, k[0]) for k, r in B.items())
        keys = set(A) & set(B)
        paired = bool(keys) and all(S.rng_key(A[k]) == S.rng_key(B[k]) for k in keys)
        if paired:
            b = sum(1 for k in keys if M[m](A[k], k[0]) and not M[m](B[k], k[0])); cc_ = sum(1 for k in keys if M[m](B[k], k[0]) and not M[m](A[k], k[0]))
            p = S.mcnemar(b, cc_); how = f"짝 {b}/{cc_}"
        else:
            p = fisher(fa, len(A), fb, len(B)); how = "Fisher"
        return dict(a=fa, na=len(A), b=fb, nb=len(B), d=100 * (fa / len(A) - fb / len(B)), p=p, how=how)

    P("## 주 시험(지연 2, seed 1–3)")
    P("| 조건 | 지표 | W3 ge0 | G21a | KE odicw | W3−G21a %p (p, 방식) | W3−KE %p (p, 방식) |"); P("|---|---|---|---|---|---|---|")
    for c in DIST:
        for m in ("낙상", "끝 정상 자세", "스텝 발생", "인계 과도 낙상(<2 s)", "최종 정숙", "회복 모드 사용"):
            cells = []
            for a in ("w3", "g21", "ke"):
                R = pp(a, c, 2); k = sum(M[m](r, c) for r in R.values())
                cells.append(f"{k}/{len(R)} = {100*k/max(len(R),1):.1f} %")
            x = [cmp(m, c, 2, o) for o in ("g21", "ke")]
            P(f"| {c} | {m} | " + " | ".join(cells) + " | " + " | ".join(f"{v['d']:+.1f} ({v['p']:.3g}, {v['how']})" if v else "—" for v in x) + " |")
    P("")
    P("## 외란 4 조건 합(지연 2)")
    dist4 = ["f12", "b18", "sf15", "fi40"]
    Q = {}
    for o in ("g21", "ke"):
        v = cmp("낙상", None, 2, o, dist4); Q[("dist", o)] = v
        if v:
            P(f"- 낙상 W3 {v['a']}/{v['na']} vs {o} {v['b']}/{v['nb']} → {v['d']:+.1f} %p (p {v['p']:.3g}, {v['how']})")
    P("")
    P("## 인계(착지) 과도 — nd, 지연 2(seed 1–3)·3·4(seed 1–2)")
    P("| 지연 | W3 과도 낙상 | G21a | KE | W3−G21a | W3−KE |"); P("|---|---|---|---|---|---|")
    for d in (2, 3, 4):
        cells = []
        for a in ("w3", "g21", "ke"):
            R = pp(a, "nd", d); k = sum(M["인계 과도 낙상(<2 s)"](r, "nd") for r in R.values()); fl = sum(r["fell"] == "1" for r in R.values())
            cells.append(f"{k}/{len(R)} (낙상 전체 {fl})")
        x = [cmp("인계 과도 낙상(<2 s)", "nd", d, o) for o in ("g21", "ke")]
        P(f"| {d} | " + " | ".join(cells) + " | " + " | ".join(f"{v['d']:+.1f} ({v['p']:.3g})" if v else "—" for v in x) + " |")
    # 지연 합(비짝 합산 — 지연마다 짝은 따로)
    for o in ("g21", "ke"):
        tot = [0, 0, 0, 0]
        for d in (2, 3, 4):
            v = cmp("인계 과도 낙상(<2 s)", "nd", d, o)
            if v:
                tot[0] += v["a"]; tot[1] += v["na"]; tot[2] += v["b"]; tot[3] += v["nb"]
        if tot[1] and tot[3]:
            Q[("tr", o)] = dict(d=100 * (tot[0] / tot[1] - tot[2] / tot[3]), p=fisher(tot[0], tot[1], tot[2], tot[3]))
            P(f"- 지연 2·3·4 합 과도 낙상 W3 {tot[0]}/{tot[1]} vs {o} {tot[2]}/{tot[3]} → {Q[('tr', o)]['d']:+.1f} %p (Fisher p {Q[('tr', o)]['p']:.3g})")
    for o in ("g21", "ke"):
        Q[("pose", o)] = cmp("끝 정상 자세", "nd", 2, o)
        Q[("each", o)] = [cmp("낙상", c, 2, o) for c in dist4]
    # W3 팔 무결성: 회복 모드 0
    w3on = sum((S.ff(r.get("sw_on_n")) or 0) > 0 for k3 in D if k3[0] == "w3" for r in D[k3].values())
    P(""); P(f"- W3 ge0 팔 무결성: 회복 모드 켜진 env {w3on} (0 이어야 함)")
    # 사전 규칙 판정
    P(""); P("## 사전 규칙 판정(STAND_MODE_CMP.md)")
    for o, nm in (("g21", "G21a"), ("ke", "KE")):
        why_w = []; why_b = []
        v = Q.get(("dist", o))
        if v and v["p"] < 0.05:
            (why_w if v["d"] >= 3 else why_b if v["d"] <= -3 else []).append(f"(i) 외란 합 {v['d']:+.1f} %p p {v['p']:.3g}")
        for c, v in zip(dist4, Q[("each", o)]):
            if v and v["p"] < 0.005:
                (why_w if v["d"] >= 5 else why_b if v["d"] <= -5 else []).append(f"(ii) {c} {v['d']:+.1f} %p p {v['p']:.3g}")
        v = Q.get(("tr", o))
        if v and v["p"] < 0.05:
            (why_w if v["d"] >= 2 else why_b if v["d"] <= -2 else []).append(f"(iii) 과도 {v['d']:+.1f} %p p {v['p']:.3g}")
        v = Q.get(("pose", o))
        if v and v["p"] < 0.05:
            (why_w if v["d"] <= -5 else why_b if v["d"] >= 5 else []).append(f"(iv) nd 끝 자세 {v['d']:+.1f} %p p {v['p']:.3g}")
        verdict = "W3 STAND 모드가 더 위험" if why_w else ("W3 STAND 모드가 더 안전" if why_b else "구별 안 됨(이 표본에서)")
        if why_w and why_b:
            verdict = "혼재(항목별로 반대)"
        P(f"- Q1 vs {nm}: **{verdict}** — " + ("; ".join(why_w + why_b) or "문턱 넘는 차 없음"))
    v = Q.get(("dist", "ke")); tr = Q.get(("tr", "ke")); nd = cmp("낙상", "nd", 2, "ke")
    ke_better = v and v["d"] >= 5 and v["p"] < 0.05
    ke_ok = not ((tr and tr["d"] <= -2 and tr["p"] < 0.05) or (nd and nd["d"] <= -2 and nd["p"] < 0.05))
    P(f"- Q2 걷기 정지 뒤 KE 인계: **{'권장' if (ke_better and ke_ok) else '권장 근거 없음(이 표본에서)'}** — 외란 합 W3−KE "
      + (f"{v['d']:+.1f} %p p {v['p']:.3g}" if v else "—") + f", KE 과도·nd 비열화 {'예' if ke_ok else '아니오'}."
      " 한계: 홀드→정책 인계에서 시작 — W3 걷기 정지→G21a 전환 과도는 미측정(별도 시험 필요).")
    open(os.path.join(rd, "smc_result.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    plan() if sys.argv[1] == "plan" else analyze(sys.argv[2])
