#!/usr/bin/env python3
"""실효 강성 기록기 — 윈치를 천천히 풀며 관절각·토크·발력·기울임을 한 파일에 남긴다 (0908).

**왜 이 도구인가.**
0908 IMU 4방향 시험으로 부호 뒤집힘 가설이 기각되면서, 어제 착지의 "앞 +14.4°"는
실제 앞 기울임이 맞는 것으로 확정됐다. 그러면 승윤님이 눈으로 본 "뒤로 쏠림"과
IMU 가 동시에 참일 수 있는 설명은 **윈치 기하** 하나만 남는다 — 후크 한 점에
매달린 채 발이 닿으면 줄이 몸통을 연직선으로 끌어당기므로, 몸통 피치는
'어디로 넘어지려는가'가 아니라 '발이 후크 연직선의 앞/뒤 어디에 놓였는가'를 잰다.

**따라서 줄이 걸린 동안 몸통 기울임으로는 아무것도 판정할 수 없다.**
남는 관측량은 후크 기하와 무관한 것들이다 — 관절각, 토크, 발력. 이 도구는
그 셋을 (기울임과 함께) 시간축 하나에 남긴다.

**읽기 전용이다.** 어떤 토픽에도 publish 하지 않는다. 모터를 움직이지 않는다.
stage8 이 이미 띄워 놓은 상태 토픽을 구독만 한다.

재는 것:
  k_ctrl [Nm/rad]  관절별 Δτ / Δ(명령각 − 실제각). MIT 규약상 kp 와 같아야 한다.
                   kp(150) 보다 한참 낮으면 **토크 포화** — 정책으로도 못 고친다.
  k_body [Nm/rad]  Δ(발목굽힘 토크 합) / Δ(몸통 피치). 감속기·구조·패드
                   컴플라이언스가 전부 들어간 **몸통 수준 실효 강성**.
                   로터 엔코더는 관절측 처짐을 못 보므로 k_ctrl 로는 안 잡힌다.
  필요강성          m·g·h (역진자). k_body 가 이보다 작으면 정적으로 못 선다.
  토크 여유         관절별 |τ| 최대. 한계에 닿으면 능동 제어도 방법이 없다.

사용 (젯슨, ROS2 소싱 · stage8 ARMED · 홀드 중):
  python3 stiffness_log.py --seconds 90 --save ~/logs/stiff_$(date +%m%d_%H%M).jsonl
     → 기록 도는 동안 **윈치를 아주 천천히** 푼다. 발력 %가 올라가는 게 자료다.
  python3 stiffness_log.py --file ~/logs/stiff_0908_1500.jsonl --com_h 0.52
  python3 stiffness_log.py --selftest
"""
import argparse
import json
import math
import os
import sys
import time

STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"

BODY_WEIGHT_N = 172.9          # 17.62 kg 저울 실측 (0904)
MASS_KG = 17.62
G = 9.81
NOMINAL_KP = 150.0             # 하드웨어맵 전 관절 공통 [Nm/rad]
ANKLE_PITCH = ("left_ankle_f_joint", "right_ankle_f_joint")

# ────────────────────────── 순수 함수 (selftest 대상) ──────────────────────────


def ols(xs, ys):
    """최소자승 기울기·절편·R². 표본 부족/분산 0이면 (None, None, None)."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None, None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    icept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy / (sxx * syy)) if syy > 1e-12 else None
    return slope, icept, r2


def imu_pitch_deg(quat):
    """콘솔과 같은 규약: fwd>0 = 앞으로 기울임. quat 은 [w,x,y,z]."""
    if not quat or len(quat) != 4:
        return None
    w, x, y, z = (float(v) for v in quat)
    gx = -2.0 * (x * z - w * y)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return math.degrees(math.atan2(gx, -gz))


def row_from_msgs(status, imu, foot, t):
    """세 토픽의 마지막 페이로드 → 기록 1행. 없는 값은 None 으로 남긴다."""
    joints = {}
    for name, j in ((status or {}).get("joints") or {}).items():
        if not isinstance(j, dict):
            continue
        prev = j.get("preview") or {}
        joints[name] = {
            "act": j.get("latest_actual_deg"),
            "cmd": j.get("commanded_relative_deg"),
            "des": j.get("desired_relative_deg"),
            "base": j.get("baseline_deg"),
            "tau": j.get("feedback_torque_est"),
            "vel": j.get("feedback_velocity_rad_s"),
            "stale": j.get("feedback_stale"),
            "p_des": prev.get("p_des_rad"),
            "kp": prev.get("kp"),
        }
    ln = float((foot or {}).get("left_n") or 0.0)
    rn = float((foot or {}).get("right_n") or 0.0)
    q = (imu or {}).get("quat") or (imu or {}).get("quat_wxyz")
    return {
        "t": round(t, 3),
        "fwd": (lambda v: round(v, 2) if v is not None else None)(imu_pitch_deg(q)),
        "fn": [round(ln, 1), round(rn, 1)],
        "pct": round(100.0 * (ln + rn) / BODY_WEIGHT_N, 1),
        "armed": bool((status or {}).get("armed")),
        "joints": joints,
    }


def analyze(rows, com_h_m=0.52, kp=NOMINAL_KP):
    """기록 → 판정표. rows 는 row_from_msgs 형식."""
    out = {"n": len(rows), "com_h_m": com_h_m}
    usable = [r for r in rows if r.get("joints")]
    out["n_joint"] = len(usable)
    if usable:
        pcts = [r["pct"] for r in usable]
        out["pct"] = {"min": min(pcts), "max": max(pcts), "gain": max(pcts) - min(pcts)}
        fwds = [r["fwd"] for r in usable if r["fwd"] is not None]
        out["fwd"] = ({"min": min(fwds), "max": max(fwds), "span": max(fwds) - min(fwds)}
                      if fwds else None)

    # ── 관절별 k_ctrl = Δτ / Δ(cmd − act) ──
    per = {}
    names = sorted({n for r in usable for n in r["joints"]})
    for nm in names:
        xs, ys, taus = [], [], []
        for r in usable:
            j = r["joints"].get(nm) or {}
            if j.get("stale") or j.get("tau") is None or j.get("act") is None or j.get("cmd") is None:
                continue
            xs.append(math.radians(float(j["cmd"]) - float(j["act"])))
            ys.append(float(j["tau"]))
            taus.append(abs(float(j["tau"])))
        s, _, r2 = ols(xs, ys)
        per[nm] = {"n": len(xs), "k_ctrl": s, "r2": r2,
                   "tau_max": (max(taus) if taus else None),
                   "err_span_deg": (round(math.degrees(max(xs) - min(xs)), 2)
                                    if len(xs) >= 2 else None)}
    out["joints"] = per

    # ── k_body = Δ(발목굽힘 토크 합) / Δ(몸통 피치) ──
    xs, ys = [], []
    for r in usable:
        if r["fwd"] is None:
            continue
        tt = 0.0
        ok = True
        for nm in ANKLE_PITCH:
            j = r["joints"].get(nm) or {}
            if j.get("tau") is None or j.get("stale"):
                ok = False
                break
            tt += float(j["tau"])
        if ok:
            xs.append(math.radians(r["fwd"]))
            ys.append(tt)
    s, _, r2 = ols(xs, ys)
    # 앞으로 기울면(fwd+) 복원 토크는 뒤로 밀어야 하므로 기울기는 음수가 정상.
    out["k_body"] = {"n": len(xs), "slope": s, "r2": r2,
                     "k_eff": (-s if s is not None else None)}
    out["k_need"] = MASS_KG * G * com_h_m
    ke = out["k_body"]["k_eff"]
    out["verdict"] = (None if ke is None else
                      ("정적 안정 여유 있음" if ke > out["k_need"] else "정적 불안정 — 넘어짐"))
    return out


def report(a):
    L = []
    L.append(f"표본 {a['n']}행 (관절 있는 행 {a['n_joint']})  CoM 높이 가정 {a['com_h_m']:.2f} m")
    p = a.get("pct")
    if p:
        L.append(f"발력 {p['min']:.1f}% → {p['max']:.1f}%  (하중 이전 {p['gain']:.1f}%p)")
    f = a.get("fwd")
    if f:
        L.append(f"몸통 피치 {f['min']:+.2f}° ~ {f['max']:+.2f}°  (진폭 {f['span']:.2f}°)")
    L.append("")
    L.append(f"{'관절':<22}{'n':>5}{'k_ctrl[Nm/rad]':>16}{'R²':>7}{'오차폭[°]':>10}{'|τ|max':>9}")
    for nm, j in a["joints"].items():
        k = "—" if j["k_ctrl"] is None else f"{j['k_ctrl']:.1f}"
        r2 = "—" if j["r2"] is None else f"{j['r2']:.2f}"
        es = "—" if j["err_span_deg"] is None else f"{j['err_span_deg']:.2f}"
        tm = "—" if j["tau_max"] is None else f"{j['tau_max']:.2f}"
        L.append(f"{nm:<22}{j['n']:>5}{k:>16}{r2:>7}{es:>10}{tm:>9}")
    kb = a["k_body"]
    L.append("")
    L.append("── 몸통 수준 실효 강성 (후크 기하와 무관한 판정은 아님 — 주석 참조) ──")
    if kb["k_eff"] is None:
        L.append(f"k_body: 표본 부족 (n={kb['n']}) — 기울임이 안 변했거나 토크가 결측")
    else:
        L.append(f"k_body = {kb['k_eff']:.1f} Nm/rad   (R²={kb['r2']:.2f}, n={kb['n']})")
        L.append(f"필요   = m·g·h = {MASS_KG}·{G}·{a['com_h_m']} = {a['k_need']:.1f} Nm/rad")
        L.append(f"명목 kp 합(발목굽힘 2개) = {2*NOMINAL_KP:.0f} Nm/rad "
                 f"→ 실효/명목 = {kb['k_eff']/(2*NOMINAL_KP):.2f}배")
        L.append(f"판정: {a['verdict']}")
    return "\n".join(L)


# ────────────────────────────── 캡처 (ROS, lazy) ──────────────────────────────


def capture(seconds, path, hz=20.0):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("stiffness_log")
    last = {"status": None, "imu": None, "foot": None}

    def mk(key):
        def cb(msg):
            try:
                last[key] = json.loads(msg.data)
            except (ValueError, TypeError):
                pass
        return cb

    node.create_subscription(String, STATUS_TOPIC, mk("status"), 10)
    node.create_subscription(String, IMU_TOPIC, mk("imu"), 20)
    node.create_subscription(String, FOOT_TOPIC, mk("foot"), 20)

    rows, t0, nxt = [], time.time(), 0.0
    print(f"[기록] {path}  {seconds:.0f}초 — 지금부터 윈치를 아주 천천히 푸십시오.")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            while time.time() - t0 < seconds:
                rclpy.spin_once(node, timeout_sec=0.02)
                el = time.time() - t0
                if el < nxt:
                    continue
                nxt = el + 1.0 / hz
                r = row_from_msgs(last["status"], last["imu"], last["foot"], el)
                rows.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if int(el * hz) % int(hz * 2) == 0:
                    print(f"  t={el:5.1f}s  발력 {r['pct']:5.1f}%  "
                          f"피치 {r['fwd'] if r['fwd'] is not None else float('nan'):+6.2f}°  "
                          f"관절 {len(r['joints'])}/12  armed={r['armed']}")
    except KeyboardInterrupt:
        print("\n[기록] 중단 — 여기까지 저장됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rows


# ──────────────────────────────── selftest ────────────────────────────────


def selftest():
    s, i, r2 = ols([0, 1, 2, 3], [1, 3, 5, 7])
    assert abs(s - 2) < 1e-9 and abs(i - 1) < 1e-9 and abs(r2 - 1) < 1e-9, (s, i, r2)
    assert ols([1, 1, 1], [1, 2, 3]) == (None, None, None), "분산 0 이면 기울기 없음"
    assert ols([0, 1], [0, 1]) == (None, None, None), "표본 2개는 거부"

    assert abs(imu_pitch_deg([1, 0, 0, 0])) < 1e-9, "수평이면 0°"
    # +Y 축 회전 10° = 앞으로 기울임 → fwd > 0 (콘솔 규약과 일치)
    a = math.radians(10.0) / 2.0
    assert abs(imu_pitch_deg([math.cos(a), 0.0, math.sin(a), 0.0]) - 10.0) < 1e-6
    assert imu_pitch_deg(None) is None and imu_pitch_deg([1, 0, 0]) is None

    r = row_from_msgs({"armed": True, "joints": {"left_ankle_f_joint": {
        "latest_actual_deg": 1.0, "commanded_relative_deg": 2.0,
        "feedback_torque_est": 4.0, "feedback_stale": False}}},
        {"quat": [1, 0, 0, 0]}, {"left_n": 80.0, "right_n": 92.9}, 1.234)
    assert r["pct"] == 100.0, r["pct"]
    assert r["joints"]["left_ankle_f_joint"]["tau"] == 4.0
    assert r["fwd"] == 0.0 and r["armed"] is True
    assert row_from_msgs(None, None, None, 0.0)["joints"] == {}, "결측이어도 죽지 않는다"

    # 합성 기록: 발목 2개가 kp=150 을 정확히 내고, 몸통은 60 Nm/rad 로 처진다
    rows = []
    for k in range(40):
        fwd = 0.10 * k                       # 0 → 3.9°
        tau = -60.0 * math.radians(fwd) / 2  # 발목 1개 몫
        js = {nm: {"latest_actual_deg": 0.0,
                   "commanded_relative_deg": math.degrees(tau / 150.0),
                   "feedback_torque_est": tau, "feedback_stale": False}
              for nm in ANKLE_PITCH}
        rows.append(row_from_msgs({"armed": True, "joints": js},
                                  {"quat": [math.cos(math.radians(fwd)/2), 0.0,
                                            math.sin(math.radians(fwd)/2), 0.0]},
                                  {"left_n": 86.45, "right_n": 86.45}, k * 0.05))
    a = analyze(rows, com_h_m=0.52)
    for nm in ANKLE_PITCH:
        assert abs(a["joints"][nm]["k_ctrl"] - 150.0) < 1.0, a["joints"][nm]
    assert abs(a["k_body"]["k_eff"] - 60.0) < 0.5, a["k_body"]
    assert a["k_need"] > a["k_body"]["k_eff"], "60 < m·g·h(89.9) 이므로 불안정 판정이어야"
    assert "불안정" in a["verdict"], a["verdict"]
    b = analyze(rows, com_h_m=0.30)          # 필요강성 51.9 < 60 → 안정
    assert "여유" in b["verdict"], b["verdict"]
    assert isinstance(report(a), str) and "k_body" in report(a)
    assert analyze([])["n"] == 0, "빈 기록도 죽지 않는다"

    print("[selftest] stiffness_log: OLS·피치규약·행변환·k_ctrl·k_body·판정·빈입력 — ALL PASS")


def main():
    ap = argparse.ArgumentParser(description="실효 강성 기록·분석 (읽기 전용)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--save", help="기록 경로 (기본 ~/logs/stiff_<stamp>.jsonl)")
    ap.add_argument("--file", help="기존 기록을 분석만 한다")
    ap.add_argument("--com_h", type=float, default=0.52, help="CoM 높이 [m] — 필요강성 m·g·h 산출용")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0
    if a.file:
        rows = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    else:
        path = a.save or os.path.expanduser(
            f"~/logs/stiff_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        rows = capture(a.seconds, path, a.hz)
        print(f"\n[기록] {len(rows)}행 → {path}\n")
    print(report(analyze(rows, com_h_m=a.com_h)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
