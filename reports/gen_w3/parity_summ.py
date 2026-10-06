"""1007 parity 감사 — eval_w3_parity JSON → 채점 낙상(채점 시작 뒤) 표. 시작 과도 낙상(스폰·지지 해제·온셋, t < 서기+워밍업)은 따로 센다.

  python3 reports/gen_w3/parity_summ.py log_picture/w3par_<tag>.json [그룹 접두 ...]
"""
import json
import math
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    J = json.load(open(sys.argv[1]))
    pref = sys.argv[2:]
    t0 = None
    print("| 그룹 | 채점 n | 채점 낙상 | 95 % CI | 시작 과도 낙상 | 유효 스텝률 | vx | vy | 요 표류 평균/최대 ° | 고리 0<|wz|<0.1 | 고리 오차 >15° env |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, g in J["groups"].items():
        if pref and not any(k.startswith(p) for p in pref):
            continue
        ts = g["cfg"]["stand"] + J["warmup"]
        early = [t for t in g["t_fall"] if t < ts]
        late = [t for t in g["t_fall"] if t >= ts]
        n = g["n"] - len(early)
        lo, hi = wilson(len(late), n)
        he = g.get("hl_err_max_deg")
        nf = (sum(1 for x in he if x > 15.0) if he else None)
        hs = "—" if g["hl_small_frac"] is None else f"{g['hl_small_frac']:.2f}"
        hf = "—" if nf is None else f"{nf}/{len(he)}"
        print(f"| {k} | {n} | {len(late)} ({100 * len(late) / max(n, 1):.1f} %) | {100 * lo:.0f}–{100 * hi:.0f} % | {len(early)} | {g['valid_rate']:.3f} | "
              f"{g['vx']:+.3f} | {g['vy']:+.3f} | {g['yaw_drift_deg_mean']:.1f} / {g['yaw_drift_deg_max']:.1f} | {hs} | {hf} |")


if __name__ == "__main__":
    main()
