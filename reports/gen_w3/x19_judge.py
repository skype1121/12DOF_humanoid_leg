"""X19 판정(TRANSITION_W3KE.md §5.22) — 걷기·합성 구조의 접촉 관측 교란 내성. x14_judge.py 를 팔마다 돌려 수락·낙상·방향을 모으고 대조 C0 와 견준다.
사용: python3 reports/gen_w3/x19_judge.py
R1(D2)·R2(F2): 58 s 낙상 ≤ C0 + 3 %p ∧ Fisher p ≥ 0.05 ∧ 수락 ≥ 99 % ∧ 활성 버튼(1·2·3·4·7·9·10 번째)의 방향 부호 일치 ≥ 95 %. C0 는 같은 seed 묶음으로."""
import os, re, subprocess, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..")); LP = os.path.join(REPO, "log_picture")
ACTIVE = (1, 2, 3, 4, 7, 9, 10)


def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    pr = lambda x: comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)   # noqa: E731
    p0 = pr(a); return sum(pr(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def judge(tags):
    tags = [t for t in tags if os.path.exists(os.path.join(LP, f"kwloop_{t}.npz"))]
    if not tags:
        return None
    out = subprocess.run([sys.executable, os.path.join(HERE, "x14_judge.py"), "--tags", ",".join(tags), "--name", "x"], capture_output=True, text=True, cwd=REPO).stdout
    m1 = re.search(r"B1 수락\*\*: (\d+) / (\d+)", out); m3 = re.search(r"B3 낙상\*\*: (\d+) / (\d+)", out)
    if not (m1 and m3):
        return None
    rows = {}
    for ln in out.split("\n"):
        c = [x.strip() for x in ln.split("|")]
        if len(c) > 12 and c[1].isdigit():
            ms = re.match(r"(\d+) / (\d+)", c[11]); mf = re.match(r"(\d+)", c[10]); ma = re.match(r"(\d+) \(", c[5])
            rows[int(c[1])] = dict(sign=(int(ms.group(1)), int(ms.group(2))) if ms else None, fall=int(mf.group(1)) if mf else 0, alive=int(c[4]), acc=int(ma.group(1)) if ma else 0, label=c[2])
    return dict(tags=tags, acc=(int(m1.group(1)), int(m1.group(2))), fall=(int(m3.group(1)), int(m3.group(2))), rows=rows)


def main():
    T = lambda arm, seeds: [f"tr_x19_{arm}_d2_s{s}" for s in seeds]   # noqa: E731
    C0 = lambda seeds: [f"tr_x14_d2_s{s}" for s in seeds]             # noqa: E731
    c4 = judge(C0((1, 2, 3, 4))); c2 = judge(C0((1, 2)))
    print("## X19 — 접촉 관측 교란 내성(콘솔 버튼 대본 58 s, 합성 러너)")
    print("| 팔 | 실행 수 | 58 s 낙상 | C0 대비 차 · Fisher p | 버튼 수락 | 활성 버튼 방향 부호 일치(최저) | 판정 |"); print("|---|---|---|---|---|---|---|")
    def line(nm, r, ref, crit):
        if r is None:
            print(f"| {nm} | 0 | (없음) | | | | |"); return None
        f, n = r["fall"]; a, an = r["acc"]; smin = min((100 * r["rows"][b]["sign"][0] / r["rows"][b]["sign"][1]) for b in ACTIVE if r["rows"].get(b, {}).get("sign"))
        if ref is None or r is ref:
            print(f"| {nm} | {len(r['tags'])} | {f} / {n} ({100*f/n:.2f} %) | — | {a} / {an} ({100*a/an:.2f} %) | {smin:.1f} % | — |"); return None
        rf, rn = ref["fall"]; d = 100 * f / n - 100 * rf / rn; p = fisher(f, n - f, rf, rn - rf)
        ok = (d <= 3.0) and (p >= 0.05) and (100 * a / an >= 99.0) and (smin >= 95.0)
        print(f"| {nm} | {len(r['tags'])} | {f} / {n} ({100*f/n:.2f} %) | {d:+.2f} %p · {p:.3g} | {a} / {an} ({100*a/an:.2f} %) | {smin:.1f} % | {('**강건**' if ok else '**미달**') if crit else '(보고)'} |")
        return ok
    line("C0 대조(seed 1–4)", c4, None, False)
    r1 = line("D2 지연 2 틱", judge(T("D2", (1, 2, 3, 4))), c4, True); r2 = line("F2 뒤집힘 2 %", judge(T("F2", (1, 2, 3, 4))), c4, True)
    line("D2F5 지연 2 + 뒤집힘 5 %", judge(T("D2F5", (1, 2, 3, 4))), c4, False)
    line("C0 대조(seed 1–2)", c2, None, False)
    pl = judge(T("PL", (1, 2))); pw = judge(T("PW", (1, 2)))
    line("PL 왼 패드 30 s 부터 0", pl, c2, False); line("PW 걷는 중 2 s 누락(35–37 s)", pw, c2, False)
    for nm, r in (("C0(seed 1–2)", c2), ("PL", pl), ("PW", pw)):
        if r:
            print(f"- {nm} 버튼별 수락/생존·구간 낙상: " + " · ".join(f"{b}:{r['rows'][b]['acc']}/{r['rows'][b]['alive']}·{r['rows'][b]['fall']}" for b in sorted(r["rows"])))
    print(f"- R1(D2) {'충족' if r1 else ('미달' if r1 is not None else '자료 없음')} · R2(F2) {'충족' if r2 else ('미달' if r2 is not None else '자료 없음')}")


if __name__ == "__main__":
    main()
