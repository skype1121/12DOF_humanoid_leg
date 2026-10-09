#!/usr/bin/env python3
"""새 몸 걷기 평가 요약 — w3/out/w3eval_<몸>_<ckpt 이름>_<조건>_s<seed>.json. 사용: summarize_w3nb.py <접두> [<접두> ...] (예: ob_15800 nb_15800)"""
import json, os, sys
from math import sqrt
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w3", "out")
CONDS = [("st", "서기 명령(ge 0)"), ("z", "제자리 스텝"), ("f", "앞 0.06"), ("stop", "앞 0.06 → 8 s 에 정지")]


def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f} % ({100 * max(0, c - h):.1f}–{100 * (c + h):.1f})"


def load(pre, c):
    out = []
    for s in (1, 2, 3, 4, 5, 6):
        f = os.path.join(H, f"w3eval_{pre}_{c}_s{s}.json")
        if os.path.isfile(f):
            out.append(json.load(open(f)))
    return out


m = lambda xs: sum(xs) / len(xs)
print("| 팔 | 조건 | seed 수·env | 시작 과도(2 s) 낙상 | 채점 구간 낙상 | 이륙 / 착지 / 하중 / 교대 % | 유효 cadence Hz | 앞 속도 m/s | 옆 속도 m/s | 요 표류 ° | 피치 RMS ° | 토크 포화 틱 %: 전 관절 / hip_r 뺀 | hip_r \\|토크\\| N·m |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for pre in sys.argv[1:]:
    for c, cn in CONDS:
        js = load(pre, c)
        if not js:
            continue
        N = sum(j["num_envs"] for j in js); fw = sum(j["fall_warm"] for j in js); fs = sum(j["fall_score"] for j in js)
        g = lambda k: m([j[k] for j in js])
        print(f"| {pre} | {cn} | {len(js)}·{N} | {wil(fw, N)} | {wil(fs, N - fw)} | {100 * g('lift_rate'):.1f} / {100 * g('touchdown_rate'):.1f} / {100 * g('load_rate'):.1f} / {100 * g('alternation_rate'):.1f} | {g('valid_cadence_hz'):.3f} | "
              f"{g('fwd_mps'):+.4f} | {g('lat_mps'):+.4f} | {g('yaw_drift_deg'):.1f} | {g('pitch_rms_deg'):.2f} | {m([j['nb']['sat_any_pct'] for j in js]):.1f} / {m([j['nb']['sat_excl_hipr_pct'] for j in js]):.1f} | {m([j['nb']['hipr_tau_mean_nm'] for j in js]):.1f} |")
print()
for pre in sys.argv[1:]:
    js = load(pre, "stop")
    for j in js:
        print(f"- {pre} 정지 seed {j['seed']}: " + json.dumps(j.get("stop"), ensure_ascii=False)[:700])
    js = load(pre, "f")
    if js:
        sj = {}
        for j in js:
            for k, v in j["nb"]["sat_joint_pct"].items():
                sj[k] = sj.get(k, 0) + v / len(js)
        print(f"- {pre} 앞 0.06 관절별 포화 %: " + " · ".join(f"{k.replace('_joint', '')} {v:.1f}" for k, v in sj.items()) + f" · 보폭(앞) cm p10/50/90 {js[0].get('step_fwd_cm_p10_50_90')} · 스텝 높이 {js[0].get('step_height_cm_p10_50_90')}")
