#!/usr/bin/env python3
"""몸통 피치 감도 스윕 (0917) — 서 있는 상태에서 관절 하나씩 대칭 ±STEP° 바꾸며 IMU 피치/롤을 기록.

목적: URDF 는 시상면 관절 1° 가 몸통 1° 로 가야 한다. 실물 감도가 1 에서 벗어난 관절이 정합 간극(발/발목 컴플라이언스·영점·IMU 편향)의 위치.
사용: robotctl 로 hang → level → 착지 후,  python3 ~/humanoid_repo/jetson/scripts/pitch_sweep.py [--step 3] [--settle 10]
안전: 각 스텝은 콘솔 joint_trim 액션(서버 게이트·소프트한계·30°/s 램프). |fwd|>ABORT_DEG 또는 Ctrl-C → 기준 자세 원복 후 종료. 줄은 느슨히 걸어 둘 것.
출력: ~/logs/pitch_sweep_<stamp>.json + 표 + 관절별 감도.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robotctl as rc

ABORT_DEG = 8.0
# (라벨, {관절: 부호}) — 대칭 규약(FK 검증): 힙앞뒤·무릎·발목앞뒤는 좌우 반대 부호
GROUPS = [("hip_f", {"left_hip_f_joint": +1, "right_hip_f_joint": -1}),
          ("knee", {"left_knee_joint": -1, "right_knee_joint": +1}),      # + = 더 굽힘
          ("ankle_f", {"left_ankle_f_joint": +1, "right_ankle_f_joint": -1})]  # + = 더 배굴

def snap(n_s):
    """n_s 초 동안 0.2 s 간격 샘플 → 평균/표준편차 + 실제 관절각 평균."""
    fw, la, gy, jp = [], [], [], {}
    t0 = time.time()
    while time.time() - t0 < n_s:
        st = rc.state(); im = st.get("imu") or {}; h = st.get("hang_hold") or {}
        fw.append(float(im.get("fwd", 0))); la.append(float(im.get("lat", 0)))
        g = im.get("gyro") or [0, 0, 0]; gy.append(max(abs(x) for x in g) * 57.3)
        for k, v in (h.get("joint_pos_deg") or {}).items(): jp.setdefault(k, []).append(float(v))
        time.sleep(0.2)
    m = lambda a: sum(a) / max(1, len(a)); sd = lambda a: (sum((x - m(a)) ** 2 for x in a) / max(1, len(a))) ** 0.5
    return {"fwd": m(fw), "fwd_sd": sd(fw), "lat": m(la), "gyro_max": max(gy) if gy else 0,
            "joints": {k: m(v) for k, v in jp.items()}, "contact": (rc.state().get("foot") or {}).get("contact")}

def set_joint(j, v):
    ok = rc.act("joint_trim", {"joint": j, "value": v}, quiet=True)
    if not ok: raise SystemExit(f"조정 거부: {j}={v}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=float, default=3.0); ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--measure", type=float, default=3.0); a = ap.parse_args()
    st = rc.state(); base = dict(st.get("joint_trim_req") or {})
    h = st.get("hang_hold") or {}
    if not (h.get("kp", 0) >= 150 and st["trial"]["allowed"].get("joint_trim", {}).get("enabled")):
        raise SystemExit("운용 게인(kp150)·개별 조정 가능 상태가 아닙니다 (hang → level → 착지 후 실행)")
    print("기준 자세:", {k: v for k, v in base.items() if abs(v) > 0.01}, flush=True)
    rows = []; stamp = time.strftime("%Y%m%d_%H%M%S")
    def restore():
        for j, v in base.items():
            if any(j in g for _, g in GROUPS): set_joint(j, v)
    def watch_settle(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            im = rc.state().get("imu") or {}
            if abs(float(im.get("fwd", 0))) > ABORT_DEG:
                print(f"!! fwd {im.get('fwd'):+.1f}° > {ABORT_DEG} → 원복", flush=True); restore(); raise SystemExit(2)
            time.sleep(0.5)
    try:
        print("▶ 기준 측정", flush=True); watch_settle(a.settle); r = snap(a.measure); r["label"] = "base0"; rows.append(r); print(f"   base0 fwd {r['fwd']:+.2f} lat {r['lat']:+.2f}", flush=True)
        for label, grp in GROUPS:
            for sgn in (+1, -1):
                d = sgn * a.step
                for j, s in grp.items(): set_joint(j, base.get(j, 0.0) + s * d)
                print(f"▶ {label} {d:+.0f}°", flush=True); watch_settle(a.settle); r = snap(a.measure); r["label"] = f"{label}{d:+.0f}"; rows.append(r)
                print(f"   {r['label']:10s} fwd {r['fwd']:+.2f} lat {r['lat']:+.2f} (sd {r['fwd_sd']:.2f}, gyro {r['gyro_max']:.0f})", flush=True)
                restore(); print("▶ 기준 복귀", flush=True); watch_settle(a.settle); r = snap(a.measure); r["label"] = f"base_after_{label}{d:+.0f}"; rows.append(r)
                print(f"   base      fwd {r['fwd']:+.2f} lat {r['lat']:+.2f}", flush=True)
    except KeyboardInterrupt:
        print("중단 → 원복", flush=True); restore(); return 1
    out = os.path.expanduser(f"~/logs/pitch_sweep_{stamp}.json"); json.dump({"base_req": base, "step": a.step, "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n== 감도 (Δ몸통피치 / Δ실제관절각, 왼쪽 관절 기준; URDF 기대 = 힙 +1, 무릎 −1(굽힘+), 발목 +1)")
    bases = [r for r in rows if r["label"].startswith("base")]; bf = sum(r["fwd"] for r in bases) / len(bases); bj = {}
    for r in bases:
        for k, v in r["joints"].items(): bj.setdefault(k, []).append(v)
    bj = {k: sum(v) / len(v) for k, v in bj.items()}
    for label, grp in GROUPS:
        jl = [j for j in grp if j.startswith("left")][0]; sens = []
        for r in rows:
            if r["label"].startswith(label) and not r["label"].startswith("base"):
                dj = r["joints"].get(jl, 0) - bj.get(jl, 0); df = r["fwd"] - bf
                if abs(dj) > 0.5: sens.append(df / dj); print(f"  {r['label']:10s} Δ관절 {dj:+.2f}° Δfwd {df:+.2f}° → 감도 {df/dj:+.2f}")
        if sens: print(f"  {label}: 평균 감도 {sum(sens)/len(sens):+.2f}")
    print(f"기준 fwd 평균 {bf:+.2f}° (표준편차 {max(r['fwd_sd'] for r in bases):.2f}) | 저장 {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
