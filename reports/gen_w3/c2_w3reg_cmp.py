"""무외란 걷기 회귀 비교(CPU, 1007) — eval_w3 채점표(w3eval_<접두>_{z,tl,tr,lat,fwd}_s<seed>.txt)에서 핵심 지표를 뽑아 체크포인트끼리 나란히.
사용: python3 c2_w3reg_cmp.py <seed> <접두...>   예) c2_w3reg_cmp.py 1 w3r_g0 w3r_g400 w3r_gF
회귀 경보(사전, C2_DESIGN §14 감시 항목): 교대 < 95 % 또는 채점 구간 낙상 > 시작점 + 2 %p 또는 명령 0 영점 |앞·옆 속도| > 0.01 m/s(시작점보다 나쁠 때)."""
import os, re, sys
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
sd = sys.argv[1]; tags = sys.argv[2:]
CM = ("z", "tl", "tr", "lat", "fwd")


def grab(t, c):
    p = f"{LP}/w3eval_{t}_{c}_s{sd}.txt"
    if not os.path.exists(p):
        return None
    s = open(p).read()
    g = lambda pat: (lambda m: m.groups() if m else None)(re.search(pat, s))
    alt = g(r"교대율[^|]*\| ([\d.]+) %"); fall = g(r"낙상률 \| ([\d.]+) % \((\d+)/(\d+)\) — 워밍업\(시작 과도\) (\d+) · 채점 구간 (\d+)")
    drift = g(r"실제 앞 ([+\-\d.]+) / 옆 ([+\-\d.]+) m/s"); yaw = g(r"요 표류\(명령 대비 \|Δ요\|\) \| ([\d.]+)°"); ygain = g(r"요 이득\(실제/명령\) ([\d.]+)"); cad = g(r"유효 cadence[^|]*\| ([\d.]+) Hz")
    wz = g(r"실제 요 속도[^|]*\| ([+\-\d.]+)")
    return {"alt": float(alt[0]) if alt else float("nan"),
            "fall_score": 100.0 * int(fall[4]) / int(fall[2]) if fall else float("nan"),
            "vx": float(drift[0]) if drift else float("nan"), "vy": float(drift[1]) if drift else float("nan"),
            "yaw": float(yaw[0]) if yaw else float("nan"), "ygain": float(ygain[0]) if ygain else float("nan"), "cad": float(cad[0]) if cad else float("nan")}


print("명령 | 접두 | 교대 % | 채점 낙상 % | 실제 앞/옆 m/s | 요 표류° | 요 이득 | cadence Hz")
base = {c: grab(tags[0], c) for c in CM}
for c in CM:
    for t in tags:
        r = grab(t, c)
        if r is None:
            print(f"{c} | {t} | 없음"); continue
        b = base[c]
        warn = []
        if r["alt"] < 95: warn.append("교대<95")
        if b and r["fall_score"] > b["fall_score"] + 2: warn.append("낙상↑")
        if c == "z" and b and max(abs(r["vx"]), abs(r["vy"])) > max(0.01, max(abs(b["vx"]), abs(b["vy"]))): warn.append("영점↑")
        print(f"{c} | {t} | {r['alt']:.1f} | {r['fall_score']:.1f} | {r['vx']:+.3f}/{r['vy']:+.3f} | {r['yaw']:.1f} | {r['ygain']:.2f} | {r['cad']:.3f}" + (f"  ⚠ {' '.join(warn)}" if warn else ""))
