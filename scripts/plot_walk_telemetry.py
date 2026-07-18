#!/usr/bin/env python3
"""보행 텔레메트리 그래프 생성기.

runner가 demo_output/telemetry/telem_<tag>.json 으로 남긴 기록을
4패널 그래프(전진/기울기/발높이/골반높이)로 그린다.

사용법:
    python3 scripts/plot_walk_telemetry.py              # 가장 최근 기록
    python3 scripts/plot_walk_telemetry.py --tag demo   # 특정 태그
    python3 scripts/plot_walk_telemetry.py --list       # 기록 목록
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEM_DIR = os.path.join(REPO, "demo_output", "telemetry")


def find_telem(tag=None):
    files = sorted(glob.glob(os.path.join(TELEM_DIR, "telem_*.json")),
                   key=os.path.getmtime)
    if not files:
        sys.exit(f"텔레메트리 기록 없음: {TELEM_DIR} (데모를 먼저 실행하세요)")
    if tag:
        want = os.path.join(TELEM_DIR, f"telem_{tag}.json")
        if not os.path.exists(want):
            sys.exit(f"태그 '{tag}' 없음. --list 로 확인하세요.")
        return want
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="telem_<tag>.json 의 태그")
    ap.add_argument("--out", default=None, help="저장 경로(.png)")
    ap.add_argument("--list", action="store_true", help="기록 목록 출력")
    args = ap.parse_args()

    if args.list:
        for f in sorted(glob.glob(os.path.join(TELEM_DIR, "telem_*.json")),
                        key=os.path.getmtime):
            m = json.load(open(f)).get("metrics", {})
            print(f"{os.path.basename(f):32s} steps={m.get('steps_total')} "
                  f"fwd={m.get('forward_progress_m')}m fell={m.get('fell_at')}")
        return

    path = find_telem(args.tag)
    d = json.load(open(path))
    rows, m = d["rows"], d["metrics"]
    tag = m.get("tag", "run")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    fell = f", FELL at {m['fell_at']}s" if m.get("fell_at") else ", no fall"
    axes[0].plot(t, [-r["pelvis_y"] for r in rows], lw=2, color="#2563eb")
    axes[0].set_ylabel("forward [m]")
    axes[0].set_title(f"12DOF walk telemetry — {tag} "
                      f"({m.get('steps_total')} steps, "
                      f"{m.get('forward_progress_m')} m{fell})")
    axes[0].grid(alpha=.3)

    axes[1].plot(t, [r["lean_ap"] for r in rows], label="lean_ap (+back)",
                 color="#dc2626")
    axes[1].plot(t, [r["lean_lat"] for r in rows], label="lean_lat (+left)",
                 color="#16a34a")
    axes[1].axhline(30, ls="--", c="gray", lw=.8)
    axes[1].axhline(-30, ls="--", c="gray", lw=.8)
    axes[1].set_ylabel("lean [deg]")
    axes[1].legend()
    axes[1].grid(alpha=.3)

    axes[2].plot(t, [r["foot_z"]["left"] for r in rows], label="left foot z",
                 color="#7c3aed")
    axes[2].plot(t, [r["foot_z"]["right"] for r in rows], label="right foot z",
                 color="#ea580c")
    axes[2].axhline(0.050, ls=":", c="k", lw=.8)
    axes[2].set_ylabel("foot z [m]")
    axes[2].legend()
    axes[2].grid(alpha=.3)

    axes[3].plot(t, [r["pelvis_z"] for r in rows], color="#0f766e")
    axes[3].set_ylabel("pelvis z [m]")
    axes[3].set_xlabel("t [s]")
    axes[3].grid(alpha=.3)

    out = args.out or os.path.join(REPO, "demo_output", f"telemetry_{tag}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print("saved:", out)


if __name__ == "__main__":
    main()
