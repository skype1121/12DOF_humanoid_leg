#!/usr/bin/env python3
"""학습 로그에서 '붕괴 전' 최고 체크포인트를 고른다 (0920 G6 사고 재발 방지).

G6(2026-09-20_15-06-59_g6)은 4500 부근에서 에피소드 813스텝·time_out 66% 였다가 5068 이후 15스텝·자세이탈 100% 로
붕괴했는데, 후처리가 '지지 0 생존 수' 동률(전부 0) 을 sort 로 깨며 model_5999(붕괴 뒤)를 BEST 로 골랐다.
평가 결함(체인 blend 출발 0·명목 유격 0)과 겹쳐 세대 전체가 0/256 으로 오판됐다.

규칙:
  1. rsl_rl 학습 로그(ANSI 포함)에서 반복별 Mean reward / Mean episode length / time_out 비율을 읽는다.
  2. 붕괴 = 최고 에피소드 길이 대비 `--collapse_frac`(기본 0.5) 아래로 떨어진 뒤 `--collapse_hold`(기본 20) 반복 연속 유지.
     붕괴 시작 이후 체크포인트는 후보에서 제외한다.
  3. 남은 체크포인트마다 그 저장 직전 `--window`(기본 100) 반복의 평균 에피소드 길이(기본) 또는 보상으로 순위.
  4. 출력: 상위 N 후보 + 붕괴 여부. 결정은 사람이 한다(평가 없이 등록·배포 금지).

  python3 rl_walking/scripts/pick_checkpoint.py <train.log> <run_dir> [--metric len|reward|timeout] [--top 3]
"""
import argparse
import glob
import os
import re


def parse_log(path):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="ignore").read())
    rows = []
    for blk in re.split(r"Learning iteration ", txt)[1:]:
        try:
            it = int(blk.split("/")[0])
        except ValueError:
            continue
        g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, blk))
        rows.append({"it": it, "reward": g(r"Mean reward: *(-?[\d.]+)"), "len": g(r"Mean episode length: *(-?[\d.]+)"),
                     "timeout": g(r"Episode_Termination/time_out: *(-?[\d.]+)"),
                     "bad_orientation": g(r"Episode_Termination/bad_orientation: *(-?[\d.]+)")})
    return [r for r in rows if r["len"] is not None]


def find_collapse(rows, frac=0.5, hold=20):
    """최고 에피소드 길이의 frac 아래가 hold 반복 연속이면 그 시작 반복을 돌려준다. 없으면 None."""
    if not rows:
        return None
    peak_i = max(range(len(rows)), key=lambda i: rows[i]["len"])
    thr = rows[peak_i]["len"] * frac
    for i in range(peak_i, len(rows) - hold + 1):
        if all(r["len"] < thr for r in rows[i:i + hold]):
            return rows[i]["it"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("run_dir")
    ap.add_argument("--metric", default="len", choices=("len", "reward", "timeout"))
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--collapse_frac", type=float, default=0.5)
    ap.add_argument("--collapse_hold", type=int, default=20)
    a = ap.parse_args()
    rows = parse_log(a.log)
    if not rows:
        print("로그에서 반복을 못 읽음"); return 2
    cks = sorted(int(re.search(r"model_(\d+)", p).group(1)) for p in glob.glob(os.path.join(a.run_dir, "model_*.pt")))
    if not cks:
        print("체크포인트 없음:", a.run_dir); return 2
    col = find_collapse(rows, a.collapse_frac, a.collapse_hold)
    peak = max(rows, key=lambda r: r["len"])
    print(f"반복 {rows[0]['it']}~{rows[-1]['it']} · 최고 에피소드 길이 {peak['len']:.0f}스텝 @ {peak['it']} · "
          f"붕괴 시작 {col if col is not None else '없음'} (기준: 최고의 {a.collapse_frac:.0%} 아래 {a.collapse_hold}회 연속)")
    by_it = {r["it"]: r for r in rows}
    scored = []
    for c in cks:
        if col is not None and c >= col:
            continue
        win = [by_it[i] for i in range(max(0, c - a.window), c + 1) if i in by_it and by_it[i][a.metric] is not None]
        if not win:
            continue
        scored.append((sum(r[a.metric] for r in win) / len(win), c, win[-1]))
    scored.sort(reverse=True)
    print(f"{'순위':<4}{'체크포인트':<14}{'창평균 ' + a.metric:>14}{'len':>8}{'reward':>9}{'timeout':>9}{'badori':>8}")
    for k, (sc, c, last) in enumerate(scored[:a.top], 1):
        print(f"{k:<4}model_{c}.pt{'':<3}{sc:>14.2f}{last['len']:>8.0f}{(last['reward'] or 0):>9.1f}{(last['timeout'] or 0):>9.2f}{(last['bad_orientation'] or 0):>8.2f}")
    excluded = [c for c in cks if col is not None and c >= col]
    if excluded:
        print(f"붕괴 뒤 제외 {len(excluded)}개: model_{excluded[0]}~model_{excluded[-1]} (최종 model_{cks[-1]} 포함 — '마지막=최고' 아님)")
    print("PICK", f"model_{scored[0][1]}.pt" if scored else "없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
