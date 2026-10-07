"""W3D 평가 요약(1007): log_picture/w3eval_<PFX>_* 를 기준선 접두사와 짝으로 표로. python3 rl_walking/scripts/summarize_wd.py base d500 [d200 ...]
외란 곡선: 방향·크기별 낙상/n·정상 복귀·첫 회복 착지 힘 방향 변위 중앙(cm), seed 1·2 합 + McNemar 대신 두 비율 차와 단순 정규 근사 95 % 구간(짝 env 정보 없음 — 같은 seed·같은 DR 지문이므로 짝 비교로 해석하지 말 것).
회귀: 명령별 실제 속도·낙상·교대·요 이득 등 주요 지표."""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LP = os.path.join(REPO, "log_picture")


def J(tag):
    p = os.path.join(LP, f"w3eval_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def ci(a, n, b, m):
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    p, q = a / n, b / m
    se = math.sqrt(p * (1 - p) / n + q * (1 - q) / m)
    return (q - p) * 100, 196 * se


def dist_tab(pfxs):
    out = ["| 방향·F | " + " | ".join(f"{p} 낙상(s1+s2) · 복귀 · 첫착지Δ cm" for p in pfxs) + f" | 차({pfxs[-1]}−{pfxs[0]}) %p ±95 |", "|---" * (len(pfxs) + 2) + "|"]
    for D in "fblr":
        for gi, F in enumerate((0, 8, 12, 15, 18)):
            row = [f"{D} {F} N"]; agg = []
            for p in pfxs:
                fa = n = rc = 0; dd = []
                for S in (1, 2):
                    j = J(f"{p}_dist_{D}_s{S}")
                    if j is None:
                        continue
                    g = j["dist"]["groups"][gi]
                    fa += g["falls"]; n += g["n"]; rc += g["recovered"]
                    dd.append(g.get("step1_disp_dir_cm_p10_50_90", [float("nan")] * 3)[1])
                agg.append((fa, n))
                row.append(f"{fa}/{n} · {rc} · " + "/".join(f"{x:+.1f}" for x in dd))
            d, h = ci(agg[0][0], agg[0][1], agg[-1][0], agg[-1][1])
            row.append(f"{d:+.1f} ± {h:.1f}")
            out.append("| " + " | ".join(row) + " |")
    return out


def stop_tab(pfxs):
    out = ["| 그룹 | " + " | ".join(f"{p} 낙상 · 정상 복귀(s1+s2)" for p in pfxs) + " |", "|---" * (len(pfxs) + 1) + "|"]
    labs = None
    for p in pfxs:
        j = J(f"{p}_stop_s1")
        if j:
            labs = [g["label"] for g in j["dist"]["groups"]]; break
    for gi, lab in enumerate(labs or []):
        row = [f"`{lab}`"]
        for p in pfxs:
            fa = n = rc = 0
            for S in (1, 2):
                j = J(f"{p}_stop_s{S}")
                if j:
                    g = j["dist"]["groups"][gi]; fa += g["falls"]; n += g["n"]; rc += g["recovered"]
            row.append(f"{fa}/{n} · {rc}")
        out.append("| " + " | ".join(row) + " |")
    return out


def reg_tab(pfxs):
    keys = ["c0", "f06", "b04", "l05", "r05", "tl02", "tr02", "fl02", "fr02", "all3", "d4f06", "d4all3", "stop", "stand"]
    out = ["| 명령 | " + " | ".join(f"{p} vx/vy · 요°/s · 낙상(채점) · 교대" for p in pfxs) + " |", "|---" * (len(pfxs) + 1) + "|"]
    for k in keys + ["hl_c0", "hl_f06"]:
        row = [k]
        for p in pfxs:
            j = J(f"{p}_reg_{k}") or J(f"{p}_{k}")
            if j is None:
                row.append("—"); continue
            fs = j.get("fall_score"); fs = fs if fs is not None else j.get("fall_rate")
            fsn = f"{fs}" if not isinstance(fs, float) else f"{100 * fs:.1f} %"
            row.append(f"{j.get('fwd_mps', float('nan')):+.3f}/{j.get('lat_mps', float('nan')):+.3f} · {j.get('yaw_rate_dps', float('nan')):+.1f}"
                       f"{' (표류 ' + format(j.get('yaw_drift_deg', float('nan')), '.1f') + '°)' if k.startswith('hl') else ''} · {fsn} · {100 * j.get('alternation_rate', float('nan')):.1f}")
        out.append("| " + " | ".join(row) + " |")
    return out


if __name__ == "__main__":
    P = sys.argv[1:]
    print("\n".join(["## 외란 곡선(앞 0.06 걷기, 0.3 s 충격, 위상 분산, 롤 램프 끔, 지연 2)"] + dist_tab(P) + ["", "## 정지 시나리오(f)"] + stop_tab(P)
                    + ["", "## 회귀(외란 없음, seed 1)"] + reg_tab(P)))
