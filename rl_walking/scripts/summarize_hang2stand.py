#!/usr/bin/env python3
"""log_picture/hang2stand_*.txt 전부 → 조건별 마크다운 표 (kit 불필요). 사용: python3 rl_walking/scripts/summarize_hang2stand.py"""
import glob, os, re
rows = []
for f in sorted(glob.glob("log_picture/hang2stand_*.txt"), key=os.path.getmtime):
    t = open(f, encoding="utf-8").read(); tag = os.path.basename(f)[len("hang2stand_"):-4]
    if tag.startswith("smoke"): continue
    task = re.search(r"평가 — Biped12-HangStand-(\S+)-v0", t); ck = re.search(r"체크포인트: (.*)", t)
    cond = re.search(r"env (\d+) × ([0-9.]+)s, DR (\S+), com_fix=([^,\s]+)(?:, 푸시 (\S+))?", t)
    s5 = re.search(r"5s 생존\*?\*?\s*: (\d+)/(\d+) \(([0-9.]+)%\)", t); fall = re.search(r"낙상\(조기 종료\)\s*: (\d+)/\d+ \(([0-9.]+)%\)\s+방향 (.*)", t)
    pre = re.search(r"방출 전 낙상 (\d+)", t)
    pol = "고정Kp" if "hold" in (ck.group(1) if ck else "") else ("v18b" if "2026-09-05_19-37-39" in (ck.group(1) if ck else "") else re.sub(r".*biped12_stage4/(\S+)/model_(\d+).pt", r"\1/\2", ck.group(1) if ck else "?"))
    rows.append((tag, task.group(1) if task else "?", pol, cond.group(3) if cond else "?", cond.group(4) if cond else "?", (cond.group(5) or "ON") if cond else "?", cond.group(1) if cond else "?",
                 f"{s5.group(1)}/{s5.group(2)} ({s5.group(3)}%)" if s5 else "?", fall.group(2)+"%" if fall else "?", fall.group(3).strip() if fall else "?", pre.group(1) if pre else "0"))
print("| tag | 태스크 | 정책 | DR | com_fix | 푸시 | env | **지지0 이후 5s 생존** | 낙상 | 방향 | 방출전 낙상 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(r) + " |")
