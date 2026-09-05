"""세대 보고서 자동 생성 — reports/gen_<tag>/README.md

입력(있으면 사용, 없으면 '—'): demo_output/**/ladder_<tag>.json,
log_picture/**/드리프트DR_<tag>_<cond>.txt (cond: old_com0 old_com16 new_com0 new_com16 dr_on nom_old nom_new
xold_com0 xold_com16 xnew_com16), 플러시 자산은 <tag>_flush 로 같은 규칙.

실행: python3 rl_walking/scripts/gen_report.py <tag> [--title "stand_v17"] [--run RUN --ckpt CKPT --task TASK --usd USD]
(GPU 불필요 — 리포트 파일만 읽는다.)
"""
import argparse
import glob
import json
import os
import re

REPO = "/home/ryu/humanoid_leg_test1"
CONDS = [("old_com0", "구래퍼 com0"), ("old_com16", "구래퍼 com+16"), ("new_com0", "신래퍼 com0"),
         ("new_com16", "신래퍼 com+16"), ("dr_on", "자기 DR ON"), ("nom_old", "명목 구래퍼"),
         ("nom_new", "명목 신래퍼"), ("xold_com0", "V6고정 구래퍼 com0"), ("xold_com16", "V6고정 구래퍼 com+16"),
         ("xnew_com16", "V6고정 신래퍼 com+16")]


def find(pattern):
    hits = glob.glob(os.path.join(REPO, pattern), recursive=True)
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None


def parse_drift(path):
    d = {"file": os.path.relpath(path, REPO)}
    t = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"낙상\(조기 종료\): (\d+)/(\d+) \(([\d.]+)%\)", t)
    if m: d["fall"] = f"{m.group(1)}/{m.group(2)} ({m.group(3)}%)"; d["fall_pct"] = float(m.group(3))
    m = re.search(r"낙상 방향 — 뒤 (\d+) · 앞 (\d+) · 왼 (\d+) · 오른 (\d+)", t)
    if m: d["dir"] = f"뒤{m.group(1)}·앞{m.group(2)}·왼{m.group(3)}·오{m.group(4)}"
    m = re.search(r"평균 ([+\-\d.nan]+), 중앙 ([+\-\d.nan]+)", t)
    if m: d["trend"] = f"{m.group(1)} / {m.group(2)}"
    m = re.search(r"피치 진동 RMS\(추세 제거\) — 중앙 ([\d.]+)°", t)
    if m: d["pitch_rms"] = m.group(1)
    m = re.search(r"E_smth \(액션 2차차분\)\s+([\d.]+)", t)
    if m: d["e_smth"] = m.group(1)
    m = re.search(r"토크 포화 최대 (\S+) ([\d.]+)%", t)
    if m: d["tau_sat"] = f"{m.group(1).replace('_joint','')} {m.group(2)}%"
    m = re.search(r"발 요차 중앙 ([\d.]+)°", t)
    if m: d["yaw_diff"] = m.group(1)
    m = re.search(r"발 전후차 \|L−R\| 중앙 ([\d.]+)mm", t)
    if m: d["dx"] = m.group(1)
    m = re.search(r">2cm/s 비율 ([\d.]+)%", t)
    if m: d["slip"] = m.group(1)
    m = re.search(r"인자: (.*)", t)
    if m: d["args"] = m.group(1).strip()
    rows = re.findall(r"^\s+(\w+_joint)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", t, re.M)
    d["tau_rows"] = [(r[0], float(r[1]), float(r[2]), float(r[4])) for r in rows]   # name, peak, lim, sat%
    return d


def ladder_rows(tag):
    p = find(f"demo_output/**/ladder_{tag}.json")
    if not p: return None, []
    rows = json.load(open(p))
    return os.path.relpath(p, REPO), rows


def section(tag, title):
    out = [f"## {title} ({tag})", ""]
    lp, rows = ladder_rows(tag)
    if rows:
        out += [f"**검증 사다리** (`{lp}`)", "", "| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |", "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: (x["stage"], x["dr"])):
            out.append(f"| {r['stage']} | {'ON' if r['dr'] else 'OFF'} | {r['fall']}/{r['n']} | {r['trend_dps']:+.2f} | {r['e_smth']:.3f} | {r['energy']:.2f} | {'✓' if r['pass_gate'] else '✗'} |")
        out.append("")
    out += ["**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)", "",
            "| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |",
            "|---|---|---|---|---|---|---|---|---|"]
    parsed = {}
    for key, label in CONDS:
        p = find(f"log_picture/**/드리프트DR_{tag}_{key}.txt")
        if not p:
            out.append(f"| {label} | — | | | | | | | |"); continue
        d = parse_drift(p); parsed[key] = d
        out.append(f"| {label} | {d.get('fall','—')} | {d.get('dir','')} | {d.get('trend','')} | {d.get('pitch_rms','')}° | {d.get('e_smth','')} | {d.get('tau_sat','')} | {d.get('yaw_diff','')}° | {d.get('slip','')}% |")
    out.append("")
    nom = parsed.get("nom_new") or parsed.get("nom_old")
    if nom and nom.get("tau_rows"):
        out += ["**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%", "", "| 관절 | τ피크 | 한계 | 포화% |", "|---|---|---|---|"]
        for n, pk, lim, sat in nom["tau_rows"]:
            out.append(f"| {n.replace('_joint','')} | {pk:.1f} | {lim:.0f} | {sat:.1f} |")
        out.append("")
    if parsed:
        out += ["<details><summary>원본 파일·인자</summary>", ""]
        for key, d in parsed.items():
            out.append(f"- `{d['file']}` — `{d.get('args','')[:160]}`")
        out += ["", "</details>", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--title", default=None)
    ap.add_argument("--run", default=""); ap.add_argument("--ckpt", default=""); ap.add_argument("--task", default=""); ap.add_argument("--usd", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    title = a.title or f"stand_{a.tag}"
    out_dir = os.path.join(REPO, "reports", f"gen_{a.tag.replace('_fix','')}")
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# {title} — 세대 판정 보고 (자동 생성: gen_report.py)", ""]
    meta = [f"- run: `{a.run}`", f"- checkpoint: `{a.ckpt}`", f"- task: `{a.task}`", f"- 자산: `{a.usd}`"]
    md += [m for m in meta if not m.endswith("``")]
    if a.note: md += ["", a.note]
    md.append("")
    md += section(a.tag, "판정 — 기본 자산(usd_final3, 패드 돌출)")
    if find(f"log_picture/**/드리프트DR_{a.tag}_flush_old_com0.txt") or find(f"demo_output/**/ladder_{a.tag}_flush.json"):
        md += section(f"{a.tag}_flush", "판정 — 플러시 자산(usd_final3f, 패드 눌림)")
    md += ["---", "판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), "
           "V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화."]
    p = os.path.join(out_dir, "README.md")
    open(p, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(p)


if __name__ == "__main__":
    main()
