#!/usr/bin/env python3
"""v5d(S8ac4d, 지연 DR 3~10) 사전등록 실험 결과 수확 → FINAL_REPORT §8.4 표 + 승격 규칙 판정.
결과 파일(log_picture/hang2stand_<tag>.txt)만 읽는다(stdout 은 DR-on 종료 시 소실). 2026-09-11 23:5x, 결과 보기 전에 작성.

규칙 해석(결과 전에 고정):
 (1) A: 명목 seed0·1·2 3-seed 평균이 v5 3-seed 평균 −1.0 pp 이상. B: DR+푸시 3-seed 평균이 v5 −3.0 pp 이상
     (부록의 "2-seed 평균"은 seed1·2 평균 — 함께 계산해 둘 다 표기, 판정은 3-seed 평균·2-seed 평균 모두 충족).
 (2) H69 DR 푸시 OFF seed0 ≥ v5 +10 pp  또는  H9 DR+푸시 ≥ v5 +10 pp;
     그리고 H69 DR 푸시 OFF seed1·2 각각 v5 같은 seed 대비 ≥ +5 pp (방향·크기 재현).
 (3) 발 들기/env: 명목 3-seed 평균이 v5 3-seed 평균을 넘지 않음(≤ +0.0).
 (4) 동일 체인·관측 320·게인 — 평가 인자 줄에서 태스크 S8ac4·h3 chain 확인.
하나라도 어긋나면 v5 유지."""
import re, sys, pathlib
LP = pathlib.Path("log_picture")
def read(tag):
    f = LP / f"hang2stand_{tag}.txt"
    if not f.exists(): return None
    t = f.read_text(errors="replace")
    m = re.search(r"5s 생존\*\*\s*:\s*(\d+)/(\d+) \(([\d.]+)%\)", t)
    fl = re.search(r"발 들기 ([\d.]+)회/env", t)
    chain = ("h3 chain" in t) and ("S8ac4-v0" in t)
    return dict(n=int(m.group(1)), N=int(m.group(2)), pct=float(m.group(3)), lift=float(fl.group(1)) if fl else None, chain=chain) if m else None
def pct(r): return "—" if r is None else f"{r['pct']:.1f}"
def lift(r): return "—" if r is None else f"{r['lift']:.2f}"
def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
rows = [  # (표 라벨, v5 태그들, v5d 태그들)
 ("A 명목 (seed0 / 1 / 2)", ["v5_nom_256","v5_nom_s1","v5_nom_s2"], ["v5d_nom_256","v5d_nom_s1","v5d_nom_s2"]),
 ("B DR+푸시 (seed0 / 1 / 2)", ["v5_dr_256","v5_dr_s1","v5_dr_s2"], ["v5d_dr_256","v5d_dr_s1","v5d_dr_s2"]),
 ("B′ DR 푸시 OFF", ["v5_dr_nopush"], ["v5d_dr_nopush"]),
 ("H69 명목", ["v5_nom_delay69"], ["v5d_nom_delay69"]),
 ("H69 DR 푸시 OFF (seed0 / 1 / 2)", ["v5_dr_nopush_delay69","v5_dr_nopush_delay69_s1","v5_dr_nopush_delay69_s2"], ["v5d_dr_nopush_delay69","v5d_dr_nopush_delay69_s1","v5d_dr_nopush_delay69_s2"]),
 ("H69 DR+푸시", ["v5_dr_delay69"], ["v5d_dr_delay69"]),
 ("H9 DR+푸시 지연 9", ["v5_dr_delay9"], ["v5d_dr_delay9"]),
 ("C realistic 명목", ["v5_nom_cReal"], ["v5d_nom_cReal"]),
]
R = {}
def get(tag):
    if tag not in R: R[tag] = read(tag)
    return R[tag]
print("| 조건 | v5 | v5d | Δ(pp) |"); print("|---|---|---|---|")
for lab, a, b in rows:
    va = [get(t) for t in a]; vb = [get(t) for t in b]
    ma, mb = mean([x and x["pct"] for x in va]), mean([x and x["pct"] for x in vb])
    d = "—" if (ma is None or mb is None) else f"{mb-ma:+.1f}"
    print(f"| {lab} | {' / '.join(pct(x) for x in va)} | {' / '.join(pct(x) for x in vb)} | {d} |")
la = [get(t) for t in rows[0][1]]; lb = [get(t) for t in rows[0][2]]
print(f"| 발 들기/env (A seed0 / 1 / 2) | {' / '.join(lift(x) for x in la)} | {' / '.join(lift(x) for x in lb)} | "
      f"{'—' if None in [mean([x and x['lift'] for x in la]), mean([x and x['lift'] for x in lb])] else f'{mean([x and x[chr(108)+chr(105)+chr(102)+chr(116)] for x in lb])-mean([x and x[chr(108)+chr(105)+chr(102)+chr(116)] for x in la]):+.2f}'} |")
missing = [t for t in R if R[t] is None]
if missing: print(f"\n결과파일 없음: {', '.join(missing)}")
# ---- 판정 ----
def P(tag): r=get(tag); return None if r is None else r["pct"]
def L(tag): r=get(tag); return None if r is None else r["lift"]
v = {}
A5 = mean([P("v5_nom_256"),P("v5_nom_s1"),P("v5_nom_s2")]); A5d = mean([P("v5d_nom_256"),P("v5d_nom_s1"),P("v5d_nom_s2")])
B5 = mean([P("v5_dr_256"),P("v5_dr_s1"),P("v5_dr_s2")]); B5d = mean([P("v5d_dr_256"),P("v5d_dr_s1"),P("v5d_dr_s2")])
B5_2 = mean([P("v5_dr_s1"),P("v5_dr_s2")]); B5d_2 = mean([P("v5d_dr_s1"),P("v5d_dr_s2")])
H5, H5d = P("v5_dr_nopush_delay69"), P("v5d_dr_nopush_delay69"); H9, H9d = P("v5_dr_delay9"), P("v5d_dr_delay9")
rep = [(P(f"v5_dr_nopush_delay69_s{s}"), P(f"v5d_dr_nopush_delay69_s{s}")) for s in (1,2)]
LF5 = mean([L("v5_nom_256"),L("v5_nom_s1"),L("v5_nom_s2")]); LF5d = mean([L("v5d_nom_256"),L("v5d_nom_s1"),L("v5d_nom_s2")])
ok = lambda c: "충족" if c else "**미충족**"
need = [A5,A5d,B5,B5d,B5_2,B5d_2,H5,H5d,H9,H9d,LF5,LF5d]+[x for p in rep for x in p]
if any(x is None for x in need):
    print("\n판정 보류: 결과 파일 누락(위 목록)."); sys.exit(2)
c1 = (A5d >= A5-1.0) and (B5d >= B5-3.0) and (B5d_2 >= B5_2-3.0)
c2a = H5d >= H5+10.0; c2b = H9d >= H9+10.0; c2r = all(d >= a+5.0 for a,d in rep)
c2 = (c2a or c2b) and c2r
c3 = LF5d <= LF5 + 1e-9
c4 = all(get(t) and get(t)["chain"] for t in ["v5d_nom_256","v5d_dr_256","v5d_dr_nopush_delay69"])
print("\n**승격 규칙 판정** (promotion_criteria.md 부록, 해석은 이 스크립트 docstring)")
print(f"- (1) A 3-seed 평균 v5 {A5:.2f} → v5d {A5d:.2f} (기준 ≥ {A5-1.0:.2f}) : {ok(A5d >= A5-1.0)};  B 3-seed 평균 {B5:.2f} → {B5d:.2f} (≥ {B5-3.0:.2f}) : {ok(B5d >= B5-3.0)};  B 2-seed(1·2) 평균 {B5_2:.2f} → {B5d_2:.2f} (≥ {B5_2-3.0:.2f}) : {ok(B5d_2 >= B5_2-3.0)}")
print(f"- (2) H69 DR 푸시 OFF seed0 {H5:.1f} → {H5d:.1f} (≥ {H5+10:.1f}) : {ok(c2a)};  H9 DR+푸시 {H9:.1f} → {H9d:.1f} (≥ {H9+10:.1f}) : {ok(c2b)};  seed1·2 재현(각 ≥ +5 pp) " + ", ".join(f"s{i+1} {a:.1f}→{d:.1f} ({d-a:+.1f})" for i,(a,d) in enumerate(rep)) + f" : {ok(c2r)}")
print(f"- (3) 발 들기/env 명목 3-seed 평균 v5 {LF5:.2f} → v5d {LF5d:.2f} : {ok(c3)}")
print(f"- (4) 동일 체인(h3)·태스크 S8ac4 평가 : {ok(c4)}")
verdict = c1 and c2 and c3 and c4
print(f"\n### 판정: {'**v5d 승격**' if verdict else '**v5 유지 (지연 대역 미해결)**'}")
sys.exit(0 if verdict else 1)
