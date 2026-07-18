#!/usr/bin/env python3
"""sim→real 드라이런: 보행 궤적을 Stage8 명령 tape 로 변환·검증 (송신 없음).

사용법:
    python3 scripts/sim2real_dryrun.py --steps 4          # 4보 + 정지 tape
    python3 scripts/sim2real_dryrun.py --steps 4 --hz 10  # 스트림 주기 변경
    python3 scripts/sim2real_dryrun.py --shift-only       # 체중이동만(스텝 없음, 실물 1차 데모용)

출력: demo_output/sim2real_tape.jsonl + 검증 통계.
ROS/CAN 을 전혀 사용하지 않는다. tape 재생(실송신)은 별도 운영 절차
(jsonl 첫 줄 _meta.operator_procedure 참조)로만 한다.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sim_walking.gait_static import StaticGait, StaticGaitParams  # noqa: E402
from sim_walking.real_bridge import BridgeConfig, build_tape, write_tape_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--hz", type=float, default=12.5)
    ap.add_argument("--shift-only", action="store_true",
                    help="스텝 없이 체중이동 사이클만 (kd<=5 실물에서도 가능한 1차 데모)")
    ap.add_argument("--out", default=os.path.join(REPO, "demo_output",
                                                  "sim2real_tape.jsonl"))
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(REPO, "config", "sim_walk_params.json")))
    gp = dict(cfg["gait"])
    if args.shift_only:
        # 스텝 성분 제거 → 순수 체중이동(LEAN) 반복
        gp.update(step_hip_deg=0.0, hip_lift_deg=0.0, knee_lift_deg=0.0,
                  clear_deg=0.0)
        g = StaticGait(StaticGaitParams(**gp), num_steps=args.steps)
    else:
        g = StaticGait(StaticGaitParams(**gp), num_steps=args.steps)
    duration = g._t_stop() + g.stop_dur + 1.0

    tape, stats = build_tape(g, duration, BridgeConfig(stream_hz=args.hz))

    print(f"[tape] {len(tape)} commands, {duration:.1f}s, {args.hz}Hz/joint")
    print(f"[검증] violations = {len(stats['violations'])}")
    for v in stats["violations"][:5]:
        print("   ✗", v)
    print("[관절별 최대 절대각(deg)]")
    for k, v in sorted(stats["max_abs_deg"].items()):
        print(f"   {k:22s} {v:7.2f}")
    print(f"[관절별 최대 델타] max = "
          f"{max(stats['max_delta_deg'].values()) if stats['max_delta_deg'] else 0:.2f} deg "
          f"(허용 {stats['slew_capacity_deg']} deg)")
    path = write_tape_jsonl(tape, stats, args.out)
    print(f"[저장] {path}")

    ok = not stats["violations"]
    print("\n=== 판정:", "PASS ✅ tape 사용 가능" if ok else "FAIL ❌ 위반 존재 — 사용 금지", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
