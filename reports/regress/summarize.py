"""고정 회귀 검사 요약(1007) — run_regress.sh 실행 디렉터리의 산출물을 THRESHOLDS.md 문턱으로 판정해 PASS/FAIL 표를 만든다.

사용: python3 reports/regress/summarize.py <run_dir> [--prev <이전 run_dir>]   (GPU 불필요, 다시 돌려도 같은 결과)
출력: <run_dir>/summary.md(사람용 표) + results.json(기계용 값) — 이전 실행(같은 모드, 자동 선택 또는 --prev) 대비 변화 포함.

판정 규칙(THRESHOLDS.md '판정' 열):
  ≤ / ≥ / 구간 / |x|≤  : 연속 지표 — 문턱 밖이면 FAIL.
  비율≤ p / 비율≥ p    : 횟수 지표 k/n — 점추정이 문턱 안이면 PASS, 밖이면서 Wilson 95 % 구간이 문턱을 걸치면 '경계'(표본 부족 — 전체 모드로 재확인),
                         구간 전체가 문턱 밖이면 FAIL.
  짝                   : 같은 (seed, env) 짝 McNemar — 후보 쪽 낙상이 유의하게(p < 0.05) 많으면 FAIL, 많지만 p ≥ 0.05 면 '경계'.
적용 열 '혼합' 행은 --profile s4d 에서만 판정(s3d 는 참고), 해당 작업이 없으면 '미실행'(빠른 모드에 없는 항목 등). 무효 실행의 항목은 '무효'.
지표 정의는 아래 METRICS(eval_w3 json·eval_ground_recovery envs.csv 열) — 회복 체인 단계 판정은 reports/gen_w3/c2_chain_cmp.py 와 같은 식."""
import csv, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("REGRESS_DATA_REPO", "/home/ryu/humanoid_leg_test1")
LP = os.path.join(DATA, "log_picture")
KE_REF = "sab_{arm}_{cond}_s{seed}"   # KE 기준선(STOPEST §7·§12 의 sab_imu·sab_odicw, 같은 인자) — 같은 seed 면 RNG 짝
ff = lambda x: None if x in ("", None) else float(x)


# ── 문턱 표 읽기 ────────────────────────────────────────────────────────────
def load_thresholds(path=os.path.join(HERE, "THRESHOLDS.md")):
    T = []
    for line in open(path):
        if not line.startswith("| "):
            continue
        c = [x.strip().replace("\\|", "|") for x in re.split(r"(?<!\\)\|", line.strip())[1:-1]]   # 표 안 '\|' 는 글자
        if len(c) < 7 or not re.match(r"^[SWK]-[\w.-]+$", c[0]):
            continue
        T.append(dict(id=c[0], name=c[1], judge=c[2], thr=c[3], base=c[4], why=c[5], scope=c[6]))
    return T


# ── 통계 ────────────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ── 회복 체인(c2_chain_cmp.py 와 같은 식) ────────────────────────────────────
def chain_ok(r, nd):
    if r["fell"] == "1":
        return False
    s1 = True if nd else ff(r.get("to_first_t")) is not None
    ac = ff(r.get("to_after_cue")) or 0.0; dr = ff(r.get("drift_after_cue_cm"))
    s2 = (ac < 2) and (dr is None or dr <= 10.0)
    s3 = r.get("end_both") in ("1", "1.0")
    s4 = posture_ok(r, nd)
    s5 = True if nd else ff(r.get("rec_off_t")) is not None
    return all([s1, s2, s3, s4, s5])


def posture_ok(r, nd):
    """정상 자세 복귀(c2_chain_cmp ④): 외란 = nrm_t 있음(|피치|·|롤| < 3°·|ω| < 0.2·양발·명목 관절 < 5°·높이 ≥ −2 cm 0.5 s) ∧ abn_frac ≤ 0.10,
    nd = 끝 자세(수평·명목 관절) ∧ abn_frac ≤ 0.10. 낙상 = 실패."""
    if r["fell"] == "1":
        return False
    abn = ff(r.get("abn_frac"))
    if nd:
        ap, ar, dq = ff(r.get("abs_pitch")), ff(r.get("abs_roll")), ff(r.get("dq_nom_deg"))
        return (ap is not None and abs(ap) < 3.0 and abs(ar) < 3.0 and dq is not None and dq < 5.0) and (abn is None or abn <= 0.10) \
            and r.get("end_both") in ("1", "1.0")
    return ff(r.get("nrm_t")) is not None and (abn is None or abn <= 0.10) and r.get("end_both") in ("1", "1.0")


def load_csv(tag):
    p = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def rng_key(r):
    return tuple(r.get(k) for k in r if isinstance(k, str) and (k.startswith("bias_") or k.startswith("com_"))) + (r.get("pre_pass"), r.get("pre_pitch"))


def paired(A, B):
    """A·B = {(seed, env): row}. RNG 지문이 전부 같을 때만 짝 비교. 반환 (짝 수, A만 낙상, B만 낙상, RNG 일치 여부)."""
    keys = sorted(set(A) & set(B))
    if not keys or any(rng_key(A[k]) != rng_key(B[k]) for k in keys):
        return len(keys), None, None, False
    keys = [k for k in keys if A[k]["pre_pass"] == "1"]
    a = sum(1 for k in keys if A[k]["fell"] == "1" and B[k]["fell"] != "1"); b = sum(1 for k in keys if B[k]["fell"] == "1" and A[k]["fell"] != "1")
    return len(keys), a, b, True


# ── 지표 추출 ───────────────────────────────────────────────────────────────
def collect(run_dir):
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    status = {}
    sp = os.path.join(run_dir, "status.tsv")
    if os.path.exists(sp):
        for l in open(sp):
            x = l.rstrip("\n").split("\t")
            if len(x) >= 2:
                status[x[0]] = dict(status=x[1], tries=x[2] if len(x) > 2 else "", note=x[3] if len(x) > 3 else "", t_s=x[4] if len(x) > 4 else "")
    by_item = {}   # item → [(job_id, meta)]
    for jid, J in man["jobs"].items():
        by_item.setdefault(J["item"], []).append((jid, J))
    V = {}         # 지표 ID → dict(kind='cont'|'rate'|'pair', ...)
    invalid_items = set()
    for item, L in by_item.items():
        good = [(j, J) for j, J in L if status.get(j, {}).get("status") == "유효"]
        if len(good) < len(L):
            invalid_items.add(item)
        if not good:
            continue
        kind = good[0][1]["kind"]
        V.setdefault("_items", set()).add(item)
        if kind == "w3":
            D = [json.load(open(os.path.join(LP, f"w3eval_{J['tag']}.json"))) for _, J in good]
            w3_metrics(item, D, V)
        else:
            rows = {}
            for _, J in good:
                R = load_csv(J["tag"])
                for r in R or []:
                    rows[(J["seed"], r["env"])] = r
            gr_metrics(item, good[0][1], rows, V)
            V.setdefault("_rows", {})[item] = rows
            V.setdefault("_seeds", {})[item] = sorted({J["seed"] for _, J in good})
    # KE 짝: odicw vs imu(같은 실행에 imu 가 없으면 저장 기준선 sab_imu 사용), 각 팔 vs 저장 기준선
    R = V.get("_rows", {})
    tot = [0, 0, 0]; how = set(); rng_bad = []
    for c in ("sf15", "fi40", "b18"):
        A = R.get(f"K-odicw-{c}")
        if A is None:
            continue
        B = R.get(f"K-imu-{c}")
        if B is None:
            B = ref_rows("imu", c, V["_seeds"][f"K-odicw-{c}"]); how.add("저장 기준선 sab_imu")
        else:
            how.add("같은 실행 imu")
        n, a, b, ok = paired(A, B)
        if not ok:
            rng_bad.append(c); continue
        tot[0] += n; tot[1] += a; tot[2] += b
    if tot[0]:
        V["K-pair.fall"] = dict(kind="pair", n=tot[0], cand=tot[1], base=tot[2], note=f"odicw만 낙상 {tot[1]} / imu만 {tot[2]} (외란 3 조건 합, {', '.join(sorted(how))})"
                                + (f" · RNG 불일치 {rng_bad}" if rng_bad else ""))
    for arm in ("odicw", "imu"):
        tot = [0, 0, 0]; bad = []
        for c in ("nd", "sf15", "fi40", "b18"):
            A = R.get(f"K-{arm}-{c}")
            if A is None:
                continue
            B = ref_rows(arm, c, V["_seeds"][f"K-{arm}-{c}"])
            n, a, b, ok = paired(A, B)
            if not ok:
                bad.append(c); continue
            tot[0] += n; tot[1] += a; tot[2] += b
        if tot[0] or bad:
            V[f"K-ref-{arm}.fall"] = dict(kind="pair", n=tot[0], cand=tot[1], base=tot[2],
                                         note=f"이번에만 낙상 {tot[1]} / 기준선(sab_{arm})에만 {tot[2]}" + (f" · RNG 불일치(짝 불가) {bad}" if bad else "")
                                         + (" · 0/0 = 비트 재현" if tot[0] and tot[1] == 0 and tot[2] == 0 else ""))
    return man, status, V, invalid_items


def ref_rows(arm, cond, seeds):
    out = {}
    for s in seeds:
        for r in load_csv(KE_REF.format(arm=arm, cond=cond, seed=s)) or []:
            out[(s, r["env"])] = r
    return out


def w3_metrics(item, D, V):
    def rate(idk, k, n):
        V[idk] = dict(kind="rate", k=int(k), n=int(n))

    def cont(idk, vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        V[idk] = dict(kind="cont", v=(sum(v) / len(v) if v else float("nan")), per=v)
    if item == "W-dist":
        G = {}
        for d in D:
            for g in d["dist"]["groups"]:
                lab = None
                for full, short in (("cmd=0,0.05,0;push:l:6:0:0.2", "l0"), ("cmd=0,0.05,0;push:l:6:30:0.2", "l30"),
                                    ("push:f:6:0:0.3", "f0"), ("push:f:6:12:0.3", "f12"), ("push:b:6:18:0.3", "b18")):
                    if g["label"] == full:
                        lab = short; break
                if lab is None:
                    continue
                a = G.setdefault(lab, [0, 0, 0]); a[0] += g["n"]; a[1] += g["falls"]; a[2] += g["recovered"]
        for lab, (n, fl, rc) in G.items():
            rate(f"W-dist.{lab}.fall", fl, n); rate(f"W-dist.{lab}.rec", rc, n)
        return
    k = sum(d.get("fall_score") or 0 for d in D); w = sum(d.get("fall_warm") or 0 for d in D); N = sum(d["num_envs"] for d in D)
    rate(f"{item}.fall", k, N - w); rate(f"{item}.warm", w, N)
    cont(f"{item}.vx", [d.get("fwd_mps") for d in D]); cont(f"{item}.vy", [d.get("lat_mps") for d in D])
    cont(f"{item}.alt", [d.get("alternation_rate") for d in D]); cont(f"{item}.cad", [d.get("valid_cadence_hz") for d in D])
    cont(f"{item}.yaw", [d.get("yaw_drift_deg") for d in D])
    cont(f"{item}.yg", [(d.get("turn") or {}).get("yaw_gain") for d in D])
    cont(f"{item}.gx", [(d.get("mix") or {}).get("gain_x") for d in D]); cont(f"{item}.gy", [(d.get("mix") or {}).get("gain_y") for d in D])
    st = [d.get("stop") for d in D if d.get("stop")]
    if st:
        n = sum(s["alive_at_stop"] for s in st)
        rate(f"{item}.settle", round(sum(s["settle_rate"] * s["alive_at_stop"] for s in st)), n)
        rate(f"{item}.sfall", sum(s["falls_after_stop"] for s in st), n)
        cont(f"{item}.foot", [s.get("foot_mid_drift_med_cm") for s in st])


def gr_metrics(item, J, rows, V):
    nd = J["cond"] == "nd"
    P = [r for r in rows.values() if r["pre_pass"] == "1"]; n = len(P)
    has_abs = bool(P) and all(k in P[0] for k in ("abs_pitch", "abs_roll", "dq_nom_deg", "end_both", "nrm_t", "abn_frac"))   # 1006 밤 이전 CSV 에는 없음
    V[f"{item}.fall"] = dict(kind="rate", k=sum(r["fell"] == "1" for r in P), n=n)
    V[f"{item}.quiet"] = dict(kind="rate", k=sum(r["final_quiet"] == "1" for r in P), n=n)
    if has_abs:
        V[f"{item}.pose"] = dict(kind="rate", k=sum(posture_ok(r, nd) for r in P), n=n)
        V[f"{item}.chain"] = dict(kind="rate", k=sum(chain_ok(r, nd) for r in P), n=n)
    V[f"{item}.false"] = dict(kind="rate", k=sum((ff(r.get("sw_on_n")) or 0) > 0 for r in P), n=n)
    V[f"{item}.long"] = dict(kind="rate", k=sum((ff(r.get("sw_time_s")) or 0) > 4 for r in P), n=n)
    V[f"{item}.pre"] = dict(kind="rate", k=n, n=len(rows))


# ── 판정 ────────────────────────────────────────────────────────────────────
def judge(t, v, profile):
    if t["scope"] == "혼합" and profile != "s4d":
        tag = "참고"
    else:
        tag = None
    j = t["judge"]; nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t["thr"])]
    if v["kind"] == "cont":
        x = v["v"]
        if x is None or math.isnan(x):
            return tag or "무효", "NaN"
        ok = {"≤": lambda: x <= nums[0], "≥": lambda: x >= nums[0], "구간": lambda: nums[0] <= x <= nums[1], "|x|≤": lambda: abs(x) <= nums[0]}[j]()
        return tag or ("PASS" if ok else "FAIL"), f"{x:+.3f}" if abs(x) < 10 else f"{x:.1f}"
    if v["kind"] == "rate":
        k, n = v["k"], v["n"]
        if n == 0:
            return tag or "무효", "n 0"
        p = k / n; lo, hi = wilson(k, n); s = f"{k}/{n} = {100*p:.1f} % [{100*lo:.0f}–{100*hi:.0f}]"
        if j == "비율≤":
            r = "PASS" if p <= nums[0] else ("경계" if lo <= nums[0] else "FAIL")
        else:
            r = "PASS" if p >= nums[0] else ("경계" if hi >= nums[0] else "FAIL")
        return tag or r, s
    if v["kind"] == "pair":
        b, c = v["cand"], v["base"]
        p = mcnemar(b, c)
        r = "PASS" if b <= c else ("FAIL" if p < 0.05 else "경계")
        return tag or r, f"{v['note']} · McNemar p {p:.3g} (짝 {v['n']})"
    return "무효", "?"


def prev_run(run_dir, man):
    runs = sorted(glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")), "*", "results.json")))
    cands = []
    for p in runs:
        d = os.path.dirname(p)
        if os.path.abspath(d) == os.path.abspath(run_dir):
            continue
        try:
            r = json.load(open(p))
        except Exception:   # noqa: BLE001
            continue
        if r.get("mode") == man["cfg"]["mode"] and r.get("started", "") < man.get("started", "~"):
            cands.append((r.get("started", ""), d))
    return sorted(cands)[-1][1] if cands else None


def fmt_val(v):
    if v is None:
        return "—"
    if v["kind"] == "cont":
        return f"{v['v']:+.3f}"
    if v["kind"] == "rate":
        return f"{100*v['k']/max(v['n'],1):.1f} %"
    return f"{v['cand']}/{v['base']}"


def num(v):
    if v is None:
        return None
    return v["v"] if v["kind"] == "cont" else (v["k"] / v["n"] if v["kind"] == "rate" and v["n"] else None)


def main():
    a = sys.argv[1:]; run_dir = os.path.abspath(a[0]); prev = a[a.index("--prev") + 1] if "--prev" in a else None
    man, status, V, invalid_items = collect(run_dir)
    profile = man["cfg"].get("profile", "s3d"); T = load_thresholds()
    prev = prev or prev_run(run_dir, man)
    PV = json.load(open(os.path.join(prev, "results.json")))["values"] if prev and os.path.exists(os.path.join(prev, "results.json")) else {}
    rows = []; cnt = {}
    for t in T:
        item = t["id"].split(".")[0]
        v = V.get(t["id"])
        if v is None:
            r, s = ("무효" if item in invalid_items else ("지표 없음" if item in V.get("_items", ()) else "미실행")), ""
        else:
            r, s = judge(t, v, profile)
        cnt[r] = cnt.get(r, 0) + 1
        pv = PV.get(t["id"]); d = ""
        if pv is not None and v is not None and num(pv) is not None and num(v) is not None:
            dd = num(v) - num(pv)
            d = f"{fmt_val(pv)} → {fmt_val(v)} (Δ {100*dd:+.1f} %p)" if v["kind"] == "rate" else f"{fmt_val(pv)} → {fmt_val(v)} (Δ {dd:+.3f})"
        elif pv is not None and v is not None and v["kind"] == "pair":
            d = f"{pv.get('cand')}/{pv.get('base')} → {v['cand']}/{v['base']}"
        rows.append((t, r, s, d))
    # 서기·KE: 이전 실행과 env 짝(같은 seed·RNG 지문 일치 시) — 낙상 새로 생김/사라짐
    pair_prev = []
    if prev:
        pm = json.load(open(os.path.join(prev, "manifest.json")))
        for item, R in V.get("_rows", {}).items():
            P = {}
            for jid, J in pm["jobs"].items():
                if J["item"] == item and J["seed"] in V["_seeds"][item]:
                    for r in load_csv(J["tag"]) or []:
                        P[(J["seed"], r["env"])] = r
            if P:
                n, a_, b_, ok = paired(R, P)
                pair_prev.append((item, n, a_, b_, ok))
    overall = "FAIL" if cnt.get("FAIL") else ("PASS(경계 있음)" if cnt.get("경계") else "PASS")
    if cnt.get("무효"):
        overall += " · 무효 항목 있음 — 재실행 필요"
    L = []
    P_ = L.append
    c = man["cfg"]
    P_(f"# 고정 회귀 검사 결과 — {man['rid']} ({c['mode']}, 프로필 {profile})")
    P_("")
    P_(f"- 종합: **{overall}** · " + " · ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    P_(f"- W3: `{c['w3_task']}` `{c['w3']}` · 서기 `{c['stand']}` · 회복 `{c['rec']}`")
    if c.get("w3_extra") or c.get("stand_extra") or c.get("ke_extra"):
        P_(f"- 추가 인자: w3 `{c.get('w3_extra','')}` · stand `{c.get('stand_extra','')}` · ke `{c.get('ke_extra','')}`")
    P_(f"- 코드 `{man['code']['head'][:10]}` ({man['code']['branch']}) · 작업 트리 변경 {len(man['code']['dirty'])} 건 · 체크포인트 md5 "
       + " · ".join(f"{k} {(v['md5'] or '없음')[:8]}" for k, v in man["checkpoints"].items()))
    P_(f"- 시작 {man.get('started')} · 끝 {man.get('finished', '—')} · 문턱 `THRESHOLDS.md` md5 {man['scripts'].get('reports/regress/THRESHOLDS.md', '')[:8]}"
       + f" · 이전 실행 비교: {os.path.basename(prev) if prev else '없음(첫 실행)'}")
    P_("- 시뮬 판정이다. 실물 성공 확률이 아니다. '경계' = 점추정은 문턱 밖이나 95 % 구간이 문턱을 걸침(표본 부족 → 전체 모드로 재확인).")
    P_("")
    sec = {"S": "STAND (G21a 단독, 지연 2·접촉 v2)", "W": "WALK·TURN·HEADING·DELAY·STOP·걷기 중 외란 (eval_w3)", "K": "RECOVERY·SETTLE·POSTURE RESTORE (KE 체인, eval_ground_recovery)"}
    for s_, title in sec.items():
        P_(f"## {title}")
        P_("| ID | 항목 | 판정 | 문턱 | 값 | 기준선(출처) | 이전 실행 대비 |")
        P_("|---|---|---|---|---|---|---|")
        for t, r, s, d in rows:
            if t["id"][0] != s_:
                continue
            mark = {"PASS": "PASS", "FAIL": "**FAIL**", "경계": "*경계*"}.get(r, r)
            e = lambda x: str(x).replace("|", "\\|")
            P_(f"| {t['id']} | {e(t['name'])} | {mark} | {e(t['judge'])} {t['thr']} | {e(s)} | {e(t['base'])} | {e(d)} |")
        P_("")
    if pair_prev:
        P_("## 이전 실행과 env 짝(같은 seed·RNG 지문) — 서기·KE")
        for item, n, a_, b_, ok in pair_prev:
            P_(f"- {item}: " + (f"짝 {n} · 이번에만 낙상 {a_} / 이전에만 {b_} · McNemar p {mcnemar(a_, b_):.3g}" if ok else f"RNG 지문 불일치(짝 {n}) — 짝 비교 불가(환경 코드·인자 변경 의심)"))
        P_("")
    P_("## 실행 유효성")
    P_("| 작업 | 상태 | 시도 | 시간 s | 비고 |")
    P_("|---|---|---|---|---|")
    for jid in man["jobs"]:
        s = status.get(jid, {"status": "미실행", "tries": "", "note": "", "t_s": ""})
        P_(f"| {jid} | {s['status']} | {s['tries']} | {s['t_s']} | {s['note']} |")
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(L) + "\n")
    vals = {k: v for k, v in V.items() if not k.startswith("_")}
    json.dump(dict(rid=man["rid"], mode=c["mode"], profile=profile, started=man.get("started"), overall=overall, counts=cnt,
                   verdicts={t["id"]: r for t, r, _, _ in rows}, values=vals, prev=os.path.basename(prev) if prev else None),
              open(os.path.join(run_dir, "results.json"), "w"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
