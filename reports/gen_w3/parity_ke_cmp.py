"""1007 parity 감사 — KE 감지기 타이머 float32(시뮬 원본) vs float64(배포 현행 imu) 짝 비교(같은 seed·조건, 사전 검사 통과 env 만).

  python3 reports/gen_w3/parity_ke_cmp.py   → log_picture/ground_recovery_par_ke_{f32,t64}_{조건}_s{seed}_envs.csv
지표: 낙상(env 단위 짝: f32 만 낙상 / t64 만 낙상), 회복 모드 시간, 꺼짐 시각 rec_off_t 차(같은 env).
"""
import csv
import os

LP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log_picture")


def rd(t):
    with open(os.path.join(LP, f"ground_recovery_{t}_envs.csv")) as f:
        return {r["env"]: r for r in csv.DictReader(f)}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


print("| 조건 | seed | 사전 통과 짝 | 낙상 f32 | 낙상 t64 | f32만 | t64만 | 회복 모드 s 평균 f32/t64 | 꺼짐 시각 차(t64−f32) 0/−/+ 건 |")
print("|---|---|---|---|---|---|---|---|---|")
tot = [0, 0, 0, 0, 0]
for C in ("nd", "fi40", "b18"):
    for S in (1, 2):
        try:
            A, B = rd(f"par_ke_f32_{C}_s{S}"), rd(f"par_ke_t64_{C}_s{S}")
        except FileNotFoundError:
            continue
        keys = [k for k in A if k in B and A[k]["pre_pass"] in ("1", "True", "1.0") and B[k]["pre_pass"] in ("1", "True", "1.0")]
        fa = [fl(A[k]["fell"]) > 0 for k in keys]; fb = [fl(B[k]["fell"]) > 0 for k in keys]
        oa = sum(1 for x, y in zip(fa, fb) if x and not y); ob = sum(1 for x, y in zip(fa, fb) if y and not x)
        ta = [fl(A[k]["sw_time_s"]) for k in keys]; tb = [fl(B[k]["sw_time_s"]) for k in keys]
        d0 = dm = dp = 0
        for k in keys:
            ra, rb = fl(A[k]["rec_off_t"]), fl(B[k]["rec_off_t"])
            if ra == ra and rb == rb:
                d = rb - ra
                d0 += abs(d) < 1e-6; dm += d < -1e-6; dp += d > 1e-6
        print(f"| {C} | {S} | {len(keys)} | {sum(fa)} | {sum(fb)} | {oa} | {ob} | {sum(ta)/len(ta):.2f} / {sum(tb)/len(tb):.2f} | {d0}/{dm}/{dp} |")
        tot = [tot[0] + len(keys), tot[1] + sum(fa), tot[2] + sum(fb), tot[3] + oa, tot[4] + ob]
print(f"| 합 | | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} | | |")
