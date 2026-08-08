#!/usr/bin/env python3
"""정적 자세 시험 — 정책 없이 stage8 홀드만으로 목표 자세에 세워본다.

**왜 이게 다음 실험인가.**
DR 스윕 결과 실물 낙하(+4.06°/s 전방)가 학습 분포 128개 env 어디에도 없다.
CoM 역산값은 +417mm 로 물리적으로 불가능 — payload/CoM 가설은 기각됐다.
남은 것은 랜덤화가 표현하지 못하는 **계통 오차**(영점·기구학·부호 등 상수
편향)이고, 그것을 가르는 가장 값싼 측정이 이것이다:

  실물이 **전 관절 0°**(곧은 다리)에서는 발력 96%로 몇 분씩 혼자 선다.
  정책은 시뮬에서 **무릎 ∓10.8°·발목굽힘 +12.6/−2.0°·힙 −6.1/−4.8°**
  자세로 선다(DR 128/128 무낙상).
  → 그 자세를 실물에 **정책 없이** 넣어보면 답이 갈린다.
     서면  : 자세는 무죄 → 전이(속도·타이밍)나 폐루프의 문제
     넘어가면: 같은 관절각이 시뮬은 서고 실물은 못 선다 = 계통 오차 확정.
              그리고 **버티게 만드는 데 필요한 트림이 곧 오차의 크기**다.

정책이 개입하지 않으므로 발진이 원리적으로 불가능하고, 램프가 느려 언제든
중단할 수 있다. 넘어뜨리는 실험이 아니다.

사용:
    python3 posture_probe.py --pose policy_v7 --ramp 8       # 8초에 걸쳐 진입
    python3 posture_probe.py --pose zero --ramp 5            # 전 관절 0 복귀
    python3 posture_probe.py --pose custom --json '{"left_knee_joint": -10.8}'
    python3 posture_probe.py --selftest

안전:
- **자동 중단**: 전방/좌우 기울임이 한계를 넘거나, 발력이 한계 아래로
  떨어지거나, 센서가 끊기면 즉시 램프를 멈추고 그 자리에서 홀드한다.
  (되돌리지 않는다 — 급히 되돌리는 동작 자체가 위험하다. 사람이 판단한다.)
- 매 틱 슬루 제한. 기본 0.5°/틱(20Hz) = 10°/s 로 hang_stand 보다 느리다.
- stage8 이 ARMED 이고 baseline 12/12 일 때만 동작한다.
- 브리지가 live 면 거부 — 두 주체가 같은 관절을 쓰면 안 된다.
"""
import argparse
import json
import math
import os
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

#: 정책 stand_v8 의 시뮬(DR 128env, 낙상 0/128) 정착 평형 자세 [deg, 관절 프레임].
#: eval_drift_dr.py --out_tag stand_v8_DR 산출. v7 대비 스탠스가 크게 대칭화됐다
#: (발목 굽힘 비대칭 10.6°→2.3°, 무릎 ∓10.4° 스탠바이).
#: **실물 절차**: 정책 투입 **전에** 이 자세로 옮겨놓는다. 그러면 정책이
#: 움직일 거리가 0이 되어, 5차 붕괴(발력 96% → alpha 0.84에서 이탈)의
#: 직접 원인인 '하중 받은 채 자세 이동'이 사라진다.
POSE_POLICY_V8 = {
    "left_hip_f_joint": -7.82, "right_hip_f_joint": 5.02,
    "left_hip_a_joint": 0.02, "right_hip_a_joint": -0.37,
    "left_hip_r_joint": 2.00, "right_hip_r_joint": 0.74,
    "left_knee_joint": -10.24, "right_knee_joint": 10.52,
    "left_ankle_f_joint": 13.67, "right_ankle_f_joint": -11.40,
    "left_ankle_r_joint": 0.34, "right_ankle_r_joint": -0.17,
}

#: 정책 stand_v7 이 시뮬(DR 128env)에서 정착하는 평형 자세 [deg, 관절 프레임].
#: demo_output/rl_eval_trajstand_v7.npz 의 정착 구간 평균에서 산출.
POSE_POLICY_V7 = {
    "left_hip_f_joint": -6.1, "right_hip_f_joint": -4.8,
    "left_hip_a_joint": -0.8, "right_hip_a_joint": 2.1,
    "left_hip_r_joint": -1.3, "right_hip_r_joint": -2.1,
    "left_knee_joint": -10.6, "right_knee_joint": 11.0,
    "left_ankle_f_joint": 12.6, "right_ankle_f_joint": -2.0,
    "left_ankle_r_joint": -0.4, "right_ankle_r_joint": -2.0,
}
POSE_ZERO = {k: 0.0 for k in POSE_POLICY_V7}
#: 무릎만 정책값으로 — 발목/힙은 0. 어느 관절이 범인인지 분리할 때 쓴다.
POSE_KNEE_ONLY = dict(POSE_ZERO, **{"left_knee_joint": -10.6,
                                    "right_knee_joint": 11.0})
POSE_ANKLE_ONLY = dict(POSE_ZERO, **{"left_ankle_f_joint": 12.6,
                                     "right_ankle_f_joint": -2.0})
POSES = {"policy_v8": POSE_POLICY_V8, "policy_v7": POSE_POLICY_V7,
         "zero": POSE_ZERO,
         "knee_only": POSE_KNEE_ONLY, "ankle_only": POSE_ANKLE_ONLY}

RATE_HZ = 20.0
SLEW_DEG = 0.5           #: 틱당 최대 변화 — 20Hz 에서 10°/s
ABORT_FWD_DEG = 9.0      #: 발 지지면 앞여유 7.5° + 여유. 넘으면 회복 불가 영역
ABORT_LAT_DEG = 9.0
ABORT_LOAD_PCT = 55.0    #: 체중 대비. 이 아래면 줄/사람이 받기 시작한 것
SENSOR_STALE_SEC = 0.6
BODY_WEIGHT_N = 117.7


def build_targets(frames, pose_deg, baseline):
    """관절 프레임 목표각[deg] → stage8 상대 명령(baseline 기준)."""
    out = {}
    for j, deg in pose_deg.items():
        base = baseline.get(j)
        if base is None:
            continue
        motor_deg = frames.joint_rad_to_motor_deg(j, math.radians(float(deg)))
        out[j] = motor_deg - base
    return out


def slew(cur, want, limit=SLEW_DEG):
    d = want - cur
    return cur + max(-limit, min(limit, d))


def selftest():
    fails = []
    # ① 슬루가 한계를 넘지 않는다
    if abs(slew(0.0, 100.0) - SLEW_DEG) > 1e-9:
        fails.append("슬루 상한 위반")
    if abs(slew(0.0, -100.0) + SLEW_DEG) > 1e-9:
        fails.append("슬루 하한 위반")
    if abs(slew(5.0, 5.2) - 5.2) > 1e-9:
        fails.append("작은 변화에서 슬루가 목표를 지나침")
    # ② 램프가 유한 스텝에 수렴
    cur, n = 0.0, 0
    while abs(cur - 12.6) > 1e-6 and n < 10000:
        cur = slew(cur, 12.6)
        n += 1
    if n >= 10000:
        fails.append("램프 미수렴")
    # ③ 자세표 12관절 · 좌우 부호 규약(거울)
    for name, pose in POSES.items():
        if len(pose) != 12:
            fails.append(f"{name}: 관절 {len(pose)}개")
    if POSE_POLICY_V7["left_knee_joint"] * POSE_POLICY_V7["right_knee_joint"] > 0:
        fails.append("무릎 좌우 부호가 같음 — 거울 규약 위반")
    # ④ 중단 임계가 지지면 여유(7.5°)보다 크지 않게
    if ABORT_FWD_DEG > 12.0:
        fails.append("전방 중단 임계가 너무 큼 — 회복 불가 영역까지 감")
    # ⑤ 목표 변환 (프레임맵 스텁)
    class FM:
        def joint_rad_to_motor_deg(self, j, rad):
            return 10.0 + math.degrees(rad)
    got = build_targets(FM(), {"a": 5.0}, {"a": 10.0})
    if abs(got["a"] - 5.0) > 1e-9:
        fails.append(f"상대목표 산출 오류: {got}")
    if build_targets(FM(), {"a": 5.0}, {}) != {}:
        fails.append("baseline 없는 관절이 명령에 포함됨")
    for f in fails:
        print(f"[FAIL] {f}")
    print("selftest: " + ("ALL PASS" if not fails else f"{len(fails)}건 실패"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description="정적 자세 시험 (정책 없음)")
    ap.add_argument("--pose", default="policy_v8", choices=list(POSES) + ["custom"])
    ap.add_argument("--json", help="custom 자세 {관절명: deg}")
    ap.add_argument("--ramp", type=float, default=8.0, help="진입 시간(초) 목표")
    ap.add_argument("--hold", type=float, default=20.0, help="도달 후 관찰(초)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pose = dict(POSES.get(a.pose, {}))
    if a.pose == "custom":
        if not a.json:
            ap.error("--pose custom 에는 --json 이 필요합니다")
        pose = dict(POSE_ZERO)
        pose.update({k: float(v) for k, v in json.loads(a.json).items()})

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import rl_bridge_node as rb

    frames = rb.load_frame_map()
    rclpy.init(args=None)
    node = Node("posture_probe")
    state = {"status": None, "imu": None, "foot": None, "bridge": None,
             "t": {"status": 0.0, "imu": 0.0, "foot": 0.0, "bridge": 0.0}}

    def mk(k):
        def cb(m):
            try:
                state[k] = json.loads(m.data)
                state["t"][k] = time.time()
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
    while time.time() < t_end and not (state["status"] and state["imu"]
                                       and state["foot"]):
        spin()
    if not (state["status"] and state["imu"] and state["foot"]):
        print("✗ 상태·IMU·발력 중 수신되지 않은 것이 있습니다 — 스택/센서 확인")
        return 2

    st = state["status"]
    if not st.get("armed"):
        print("✗ stage8 이 ARMED 가 아닙니다 — 베이스라인·ARM 먼저")
        return 2
    joints = st.get("joints") or {}
    baseline = {j: v.get("baseline_deg") for j, v in joints.items()
                if isinstance(v, dict) and v.get("baseline_deg") is not None}
    if len(baseline) != 12:
        print(f"✗ 베이스라인 {len(baseline)}/12 — 12/12 여야 합니다")
        return 2
    if (state["bridge"] or {}).get("live"):
        print("✗ 브리지가 live 입니다 — 같은 관절을 두 주체가 쓰면 안 됩니다")
        return 2

    def sense():
        q = (state["imu"] or {}).get("quat") or (state["imu"] or {}).get("quat_wxyz")
        fwd = lat = None
        if q and len(q) == 4:
            w, x, y, z = (float(v) for v in q)
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            fwd = math.degrees(math.atan2(gx, -gz))
            lat = math.degrees(math.atan2(gy, -gz))
        f = state["foot"] or {}
        pct = 100.0 * (float(f.get("left_n", 0)) + float(f.get("right_n", 0))) \
            / BODY_WEIGHT_N
        age = max(time.time() - state["t"]["imu"], time.time() - state["t"]["foot"])
        return fwd, lat, pct, age

    fwd0, lat0, pct0, _ = sense()
    print(f"시작 자세: 전방 {fwd0:+.1f}°, 좌우 {lat0:+.1f}°, 발력 {pct0:.0f}%")
    print(f"목표 자세({a.pose}): " + ", ".join(
        f"{k.replace('_joint','')} {v:+.1f}" for k, v in sorted(pose.items())
        if abs(v) > 0.05) or "(전 관절 0)")
    print(f"슬루 {SLEW_DEG}°/틱 @{RATE_HZ:.0f}Hz — 중단: 전방|{ABORT_FWD_DEG}|° "
          f"좌우|{ABORT_LAT_DEG}|° 발력<{ABORT_LOAD_PCT:.0f}%")

    # 현재 관절각(관절 프레임)에서 출발 — 급격한 첫 스텝 방지
    cur = {}
    for j in pose:
        v = joints.get(j) or {}
        mdeg = v.get("feedback_deg")
        if mdeg is None:
            mdeg = (baseline.get(j) or 0.0) + (v.get("target_rel_deg") or 0.0)
        # 변환은 FrameMap 한 곳에서만 — 부호·영점 규약을 두 군데 구현하면
        # 언젠가 어긋나고, 그 어긋남은 조용히 자세로 나타난다.
        cur[j] = math.degrees(frames.motor_deg_to_joint_rad(j, float(mdeg)))

    period = 1.0 / RATE_HZ
    n_ramp = max(1, int(a.ramp * RATE_HZ))
    step = {j: (pose[j] - cur[j]) / n_ramp for j in pose}
    aborted = None
    t_stop = time.time() + a.ramp + a.hold
    i = 0
    while time.time() < t_stop:
        i += 1
        want = {j: (cur[j] + step[j] * min(i, n_ramp)) for j in pose}
        cur_cmd = {j: slew(cur[j], want[j]) for j in pose}
        cur = cur_cmd
        tg = build_targets(frames, cur, baseline)
        if tg:
            pub.publish(String(data=json.dumps(
                {"command": "SET_JOINT_TARGETS", "targets": tg})))
        spin(period)
        fwd, lat, pct, age = sense()
        if age > SENSOR_STALE_SEC:
            aborted = f"센서 끊김 {age:.1f}s"
        elif fwd is not None and abs(fwd) > ABORT_FWD_DEG:
            aborted = f"전방 기울임 {fwd:+.1f}° 초과"
        elif lat is not None and abs(lat) > ABORT_LAT_DEG:
            aborted = f"좌우 기울임 {lat:+.1f}° 초과"
        elif pct < ABORT_LOAD_PCT:
            aborted = f"발력 {pct:.0f}% 미만"
        if i % 10 == 0 or aborted:
            prog = min(100, int(100 * i / n_ramp))
            print(f"  [{prog:3d}%] 전방 {fwd:+5.1f}° 좌우 {lat:+5.1f}° "
                  f"발력 {pct:5.1f}%" + (f"   ← 중단: {aborted}" if aborted else ""))
        if aborted:
            break

    fwd, lat, pct, _ = sense()
    print("-" * 62)
    if aborted:
        print(f"■ 중단: {aborted}")
        print("  현 자세에서 홀드 중입니다 (되돌리지 않음 — 사람이 판단).")
        print("  → 같은 관절각이 시뮬에서는 128/128 서는데 실물이 못 섰다면,")
        print("     계통 오차(영점·기구학) 확정입니다. 어느 관절에서 무너졌는지")
        print("     knee_only / ankle_only 로 분리해 보세요.")
    else:
        print(f"■ 완주: 전방 {fwd:+.1f}° 좌우 {lat:+.1f}° 발력 {pct:.0f}%")
        print(f"  시작 대비 전방 {fwd - fwd0:+.1f}°.")
        print("  → 자세는 무죄입니다. 남은 용의자는 전이 속도·타이밍과 폐루프.")
    print(f"  판정 근거는 세션 타임라인에 기록돼 있습니다 (~/logs/sessions/).")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not aborted else 1


if __name__ == "__main__":
    sys.exit(main())
