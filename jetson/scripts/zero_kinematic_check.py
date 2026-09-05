#!/usr/bin/env python3
"""영점 기구학 검증 — 실물 영점이 URDF 영점과 같은지 실측으로 판정.

**왜 필요한가 (0809 시뮬 실측).**
stand_v8 은 영점 오차 3°에서 낙상 43.8%, 2°에서 6.3% 다. 절벽이 2~3° 사이에
있다. 그런데 실물 영점(stand_zero_deg)은 **매달림 육안 정렬로 잡았고 기구학
검증을 한 적이 없다** (그 이유로 v1을 폐기한 이력도 있다). 더 나쁜 것은
`boot_pose_check` 의 자세 허용오차가 AK70 관절에서 8° 라 **44% 낙상을 내는
3° 오차가 게이트를 '일치'로 통과한다.** 즉 지금 우리는 정책이 견딜 수 없는
크기의 오차를 검출할 수단이 없다.

**측정 원리.**
로봇이 두 발을 지면에 붙이고 서 있을 때, 발목 굽힘을 δ 만큼 바꾸면 몸통은
발목축을 중심으로 −δ 만큼 회전한다(발이 평평하고 다리가 강체라면 1:1).
따라서 발목 명령을 여러 값으로 바꾸며 IMU 몸통 피치를 재면:

    body_pitch = gain × ankle_cmd + intercept

  gain      1.0에서 크게 벗어나면 컴플라이언스·발 미끄러짐·기구학 스케일 문제
  intercept **발목 0°에서의 몸통 피치** = 영점 오차 + IMU 장착 오차의 합
            (IMU 장착은 imu_mount_calib.json 으로 이미 보정돼 있으므로,
             남는 것은 영점 오차로 해석한다)

정책은 쓰지 않는다. stage8 홀드만으로 천천히 움직이고, 각 지점에서 정지 후
평균을 낸다. 기울임·발력 자동 중단은 posture_probe 와 동일.

사용:
    python3 zero_kinematic_check.py                 # ±3° 5점 스윕
    python3 zero_kinematic_check.py --span 2 --pts 5
    python3 zero_kinematic_check.py --selftest
"""
import argparse
import json
import math
import os
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

RATE_HZ = 20.0
SLEW_DEG = 0.3            # 틱당 — 20Hz 에서 6°/s (posture_probe 보다 더 느리게)
SETTLE_S = 2.0            # 각 지점 도달 후 정지 대기
SAMPLE_S = 2.0            # 그 뒤 평균낼 구간
ABORT_FWD_DEG = 8.0
ABORT_LAT_DEG = 8.0
ABORT_LOAD_PCT = 60.0
BODY_WEIGHT_N = 172.9     # 실물 17.62kg 저울(0904)×9.81 — 어댑터 npc 0.02650(0906) 과 같은 참값 축. 구 117.7 폐기
#: 발목 굽힘은 좌우 거울 규약 — 순수 피치를 만들려면 부호를 반대로 준다
ANKLE_F = {"left_ankle_f_joint": +1.0, "right_ankle_f_joint": -1.0}


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return float("nan"), my, 0.0
    g = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - g * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy < 1e-12:
        return g, b, 1.0
    ss = sum((y - (g * x + b)) ** 2 for x, y in zip(xs, ys))
    return g, b, max(0.0, 1.0 - ss / syy)


def selftest():
    fails = []
    g, b, r2 = linfit([-3, -1.5, 0, 1.5, 3], [3.2, 1.7, 0.2, -1.3, -2.8])
    if abs(g + 1.0) > 0.05:
        fails.append(f"이득 산출 오류: {g:.3f} (기대 ≈ −1.0)")
    if abs(b - 0.2) > 0.05:
        fails.append(f"절편 산출 오류: {b:.3f} (기대 ≈ 0.2)")
    if r2 < 0.99:
        fails.append(f"R² 오류: {r2:.3f}")
    # 거울 규약: 좌우 부호가 반대여야 순수 피치
    if ANKLE_F["left_ankle_f_joint"] * ANKLE_F["right_ankle_f_joint"] > 0:
        fails.append("발목 좌우 부호가 같음 — 순수 피치가 아니라 롤이 섞인다")
    # 슬루 상한
    if SLEW_DEG * RATE_HZ > 8.0:
        fails.append("램프 속도가 너무 빠름")
    for f in fails:
        print(f"[FAIL] {f}")
    print("selftest: " + ("ALL PASS" if not fails else f"{len(fails)}건 실패"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description="영점 기구학 검증 (정책 없음)")
    ap.add_argument("--span", type=float, default=3.0, help="발목 스윕 ±범위[°]")
    ap.add_argument("--pts", type=int, default=5, help="측정 지점 수 (홀수 권장)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import rl_bridge_node as rb

    frames = rb.load_frame_map()
    rclpy.init(args=None)
    node = Node("zero_kinematic_check")
    st = {"status": None, "imu": None, "foot": None, "bridge": None,
          "t": {"status": 0.0, "imu": 0.0, "foot": 0.0, "bridge": 0.0}}

    def mk(k):
        def cb(m):
            try:
                st[k] = json.loads(m.data)
                st["t"][k] = time.time()
            except ValueError:
                pass
        return cb
    node.create_subscription(String, rb.STATUS_TOPIC, mk("status"), 10)
    node.create_subscription(String, rb.IMU_TOPIC, mk("imu"), 20)
    node.create_subscription(String, rb.FOOT_TOPIC, mk("foot"), 20)
    node.create_subscription(String, rb.BRIDGE_DEBUG_TOPIC, mk("bridge"), 10)
    pub = node.create_publisher(String, rb.COMMAND_TOPIC, 10)

    def spin(dt=0.05):
        rclpy.spin_once(node, timeout_sec=dt)

    print("상태 수신 대기…")
    t_end = time.time() + 8.0
    while time.time() < t_end and not (st["status"] and st["imu"] and st["foot"]):
        spin()
    if not (st["status"] and st["imu"] and st["foot"]):
        print("✗ 상태·IMU·발력 미수신 — 스택/센서 확인")
        return 2
    if not st["status"].get("armed"):
        print("✗ stage8 이 ARMED 가 아닙니다")
        return 2
    joints = st["status"].get("joints") or {}
    baseline = {j: v.get("baseline_deg") for j, v in joints.items()
                if isinstance(v, dict) and v.get("baseline_deg") is not None}
    if len(baseline) != 12:
        print(f"✗ 베이스라인 {len(baseline)}/12")
        return 2
    if (st["bridge"] or {}).get("live"):
        print("✗ 브리지가 live 입니다 — 두 주체가 같은 관절을 쓰면 안 됩니다")
        return 2

    def sense():
        q = (st["imu"] or {}).get("quat") or (st["imu"] or {}).get("quat_wxyz")
        fwd = lat = None
        if q and len(q) == 4:
            w, x, y, z = (float(v) for v in q)
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            fwd = math.degrees(math.atan2(gx, -gz))
            lat = math.degrees(math.atan2(gy, -gz))
        f = st["foot"] or {}
        pct = 100.0 * (float(f.get("left_n", 0)) + float(f.get("right_n", 0))) \
            / BODY_WEIGHT_N
        age = max(time.time() - st["t"]["imu"], time.time() - st["t"]["foot"])
        return fwd, lat, pct, age

    # 현재 발목각(관절 프레임)에서 출발
    cur = {}
    for j in ANKLE_F:
        v = joints.get(j) or {}
        md = v.get("feedback_deg")
        if md is None:
            md = (baseline.get(j) or 0.0)
        cur[j] = math.degrees(frames.motor_deg_to_joint_rad(j, float(md)))

    pts = [(-a.span + 2 * a.span * k / (a.pts - 1)) for k in range(a.pts)] \
        if a.pts > 1 else [0.0]
    # 0을 마지막에 한 번 더 — 이력(히스테리시스) 확인용
    pts = pts + [0.0]
    print(f"발목 굽힘 스윕: {', '.join(f'{p:+.1f}°' for p in pts)}  "
          f"(램프 {SLEW_DEG*RATE_HZ:.0f}°/s, 지점당 정지 {SETTLE_S:.0f}s + 평균 {SAMPLE_S:.0f}s)")
    f0, l0, p0, _ = sense()
    print(f"시작: 전방 {f0:+.2f}° 좌우 {l0:+.2f}° 발력 {p0:.0f}%")
    if p0 < ABORT_LOAD_PCT:
        print(f"✗ 발력 {p0:.0f}% — 로봇이 하중을 받고 서 있어야 측정이 성립합니다")
        return 2

    period = 1.0 / RATE_HZ
    results, aborted = [], None
    for target in pts:
        want = {j: target * s for j, s in ANKLE_F.items()}
        # ── 램프 ──
        while not aborted:
            done = True
            for j in ANKLE_F:
                d = want[j] - cur[j]
                if abs(d) > 1e-3:
                    cur[j] += max(-SLEW_DEG, min(SLEW_DEG, d))
                    done = False
            tg = {}
            for j, deg in cur.items():
                base = baseline.get(j)
                if base is None:
                    continue
                tg[j] = frames.joint_rad_to_motor_deg(j, math.radians(deg)) - base
            if tg:
                pub.publish(String(data=json.dumps(
                    {"command": "SET_JOINT_TARGETS", "targets": tg})))
            spin(period)
            fwd, lat, pct, age = sense()
            if age > 0.6:
                aborted = f"센서 끊김 {age:.1f}s"
            elif abs(fwd) > ABORT_FWD_DEG:
                aborted = f"전방 {fwd:+.1f}° 초과"
            elif abs(lat) > ABORT_LAT_DEG:
                aborted = f"좌우 {lat:+.1f}° 초과"
            elif pct < ABORT_LOAD_PCT:
                aborted = f"발력 {pct:.0f}% 미만"
            if done:
                break
        if aborted:
            break
        # ── 정지 대기 후 평균 ──
        t_end = time.time() + SETTLE_S
        while time.time() < t_end:
            spin(period)
        acc, n = [], 0
        t_end = time.time() + SAMPLE_S
        while time.time() < t_end:
            spin(period)
            fwd, lat, pct, _ = sense()
            acc.append((fwd, lat, pct))
            n += 1
        mf = sum(a_[0] for a_ in acc) / max(1, n)
        ml = sum(a_[1] for a_ in acc) / max(1, n)
        mp = sum(a_[2] for a_ in acc) / max(1, n)
        results.append((target, mf, ml, mp))
        print(f"  발목 {target:+5.1f}°  →  몸통 전방 {mf:+6.2f}°  "
              f"좌우 {ml:+6.2f}°  발력 {mp:5.1f}%")

    print("-" * 62)
    if aborted:
        print(f"■ 중단: {aborted}  (현 자세 홀드 — 되돌리지 않습니다)")
        return 1
    xs = [r[0] for r in results[:-1]]
    ys = [r[1] for r in results[:-1]]
    g, b, r2 = linfit(xs, ys)
    hyst = results[-1][1] - results[len(results) // 2][1] if len(results) > 2 else 0.0
    print(f"■ 적합: 몸통피치 = {g:+.3f} × 발목 {b:+.3f}   (R²={r2:.3f})")
    print(f"  이력(0° 복귀 차이): {hyst:+.2f}°")
    print()
    print(f"■ 이득 {g:+.3f} — 기대 −1.0 (발 평평·강체 가정)")
    if abs(abs(g) - 1.0) > 0.3:
        print("   → 1.0에서 크게 벗어남: 발 미끄러짐·컴플라이언스·기구학 스케일 의심")
    else:
        print("   → 기구학은 정합. 발이 평평하게 붙어 있고 전달이 정상이다.")
    print()
    print(f"■ **절편 {b:+.3f}° = 발목 0°에서의 몸통 피치**")
    print("   IMU 장착은 imu_mount_calib.json 으로 보정돼 있으므로,")
    print("   남는 값은 영점 오차로 해석한다.")
    a_ = abs(b)
    if a_ < 1.0:
        print(f"   → {a_:.2f}° — 양호. stand_v8(더 꼿꼿·부드러움) 사용 가능.")
    elif a_ < 2.0:
        print(f"   → {a_:.2f}° — 경계. v8 낙상 6% 대역. stand_v9 권장.")
    else:
        print(f"   → {a_:.2f}° — **위험**. v8은 3°에서 44% 낙상한다.")
        print("      stand_v9(3°에서 16%)를 쓰거나, 영점을 재앵커하고 재측정할 것.")
    print("-" * 62)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
