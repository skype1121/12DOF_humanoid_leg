"""IMU 어댑터 — humanoid_imu(sensor_msgs/Imu)를 RL 브리지 JSON(std_msgs/String)으로 변환.

입력: /humanoid/imu/data (sensor_msgs/Imu, ~100Hz, 기존 humanoid_imu 노드)
  - orientation: iAHRS RPY로 만든 쿼터니언 (메시지 필드 순서는 x, y, z, w)
  - angular_velocity rad/s, linear_acceleration m/s²
출력: /humanoid/imu (std_msgs/String, 입력과 동일 레이트) — 브리지 계약 JSON:
  {"timestamp": epoch초 float, "gyro_rad_s": [x,y,z],
   "quat_wxyz": [w,x,y,z], "accel_m_s2": [x,y,z], "mount_calibrated": bool}

핵심 주의 1: sensor_msgs 쿼터니언은 필드가 (x,y,z,w) 순서지만 브리지 계약은
[w,x,y,z]. 이 재배열이 이 노드의 존재 이유이며, --selftest가
(x=1,y=2,z=3,w=4) → [4,1,2,3]을 실제로 검증한다.

핵심 주의 2: iAHRS가 몸통에 회전 장착되어 있다(수직 매달림 실측 투영중력
(−1.00,+0.03,−0.00) = IMU +X가 위). 시작 시 장착 보정 JSON(R_base_imu,
v_base = R @ v_imu; imu_mount_calib.py로 생성)을 탐색해 적용한다:
  - gyro/accel: v_base = R @ v_imu
  - orientation: world←base = R_world_imu @ R_base_imuᵀ  (world←imu에 장착
    회전 역을 우측 곱)
탐색 순서: 리포 config/imu_mount_calib.json → /home/mama/rl_calib/
imu_mount_calib.json. 없으면 항등(경고 로그), 있는데 직교/det+1이 아니면
기동 거부. "mount_calibrated"로 소비측에 적용 여부를 알린다.

구조: 변환은 순수 함수 imu_to_bridge_dict()로 분리, rclpy는 main() 안에서만
lazy import → --selftest는 ROS 소싱 없는 로컬 환경에서도 그대로 돈다.
3x3/쿼터니언 헬퍼도 순수 파이썬(math만) — numpy 비의존 유지.

QoS: 구독·발행 모두 depth 50, reliability 기본값(RELIABLE).

사용: python3 imu_adapter_node.py             (Jetson, ROS2 Humble 소싱 후)
      python3 imu_adapter_node.py --selftest  (ROS 불필요 — 변환 로직 검증)
"""
import argparse
import json
import math
import os

IN_TOPIC = "/humanoid/imu/data"
OUT_TOPIC = "/humanoid/imu"
QOS_DEPTH = 50  # 양쪽 동일
MOUNT_CALIB_NAME = "imu_mount_calib.json"


# ---------- 3x3/쿼터니언 순수 수학 (numpy 비의존) ----------

def mat_vec3(r, v):
    return [r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2]]


def mat_t3(r):
    return [[r[j][i] for j in range(3)] for i in range(3)]


def mat_mul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mat_det3(r):
    return (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))


def validate_rotation(r, eps=1e-6):
    """RRᵀ≈I 및 det=+1 검증 — 어긋나면 ValueError (한국어 사유 포함)."""
    for i in range(3):
        for j in range(3):
            rrt = sum(r[i][k] * r[j][k] for k in range(3))
            expect = 1.0 if i == j else 0.0
            if abs(rrt - expect) > eps:
                raise ValueError(
                    f"직교성 위반: (RRᵀ)[{i}][{j}]={rrt:+.9f} (허용 {eps})")
    d = mat_det3(r)
    if abs(d - 1.0) > eps:
        raise ValueError(f"행렬식 이상: det={d:+.9f} (+1 필요 — 반사 의심)")


def quat_wxyz_to_mat(w, x, y, z):
    """단위 쿼터니언 [w,x,y,z] → 회전행렬 (v_world = R @ v_body)."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def mat_to_quat_wxyz(r):
    """회전행렬 → 쿼터니언 [w,x,y,z] (트레이스 분기 — 수치 안정)."""
    t = r[0][0] + r[1][1] + r[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0  # s = 4w
        return [0.25 * s,
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s]
    if r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0  # s = 4x
        return [(r[2][1] - r[1][2]) / s,
                0.25 * s,
                (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s]
    if r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0  # s = 4y
        return [(r[0][2] - r[2][0]) / s,
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s]
    s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0  # s = 4z
    return [(r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s]


# ---------- 장착 보정 로드 ----------

def mount_calib_paths():
    """탐색 후보 경로: 리포 config/ (파일 기준 3단계 위) → Jetson rl_calib."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return [os.path.join(repo, "config", MOUNT_CALIB_NAME),
            os.path.join("/home/mama/rl_calib", MOUNT_CALIB_NAME)]


def load_mount_calib():
    """장착 보정 탐색·로드 → (R_base_imu | None, 경로 | None).

    파일이 있는데 회전행렬이 불량이면 ValueError — 호출측(main)이 기동 거부.
    """
    for path in mount_calib_paths():
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            calib = json.load(f)
        r = [[float(v) for v in row] for row in calib["R_base_imu"]]
        if len(r) != 3 or any(len(row) != 3 for row in r):
            raise ValueError(f"{path}: R_base_imu가 3x3이 아님")
        try:
            validate_rotation(r)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e
        return r, path
    return None, None


# ---------- 변환 (순수 함수) ----------

def imu_to_bridge_dict(stamp_sec, stamp_nanosec, quat_xyzw, gyro_xyz, accel_xyz,
                       mount_r=None):
    """Imu 메시지 필드값 → 브리지 계약 dict (순수 함수, ROS 비의존).

    quat_xyzw: 메시지 필드 순서 그대로 (x, y, z, w) 튜플 →
    출력 quat_wxyz는 [w, x, y, z]로 재배열된다.

    mount_r: 장착 회전 R_base_imu(3x3, v_base = R @ v_imu) 또는 None.
      None → 종전과 동일한 통과(재배열만), "mount_calibrated": false.
      행렬 → gyro/accel에 R 적용, orientation은 world←base =
      R_world_imu @ Rᵀ로 재구성 후 [w,x,y,z] 재산출, "mount_calibrated": true.
      (쿼터니언 노름이 0에 가까우면 회전 재구성 불가 → 그대로 통과.)
    """
    x, y, z, w = quat_xyzw
    if mount_r is None:
        gyro_out = [float(v) for v in gyro_xyz]
        accel_out = [float(v) for v in accel_xyz]
        quat_out = [float(w), float(x), float(y), float(z)]
        calibrated = False
    else:
        gyro_out = mat_vec3(mount_r, [float(v) for v in gyro_xyz])
        accel_out = mat_vec3(mount_r, [float(v) for v in accel_xyz])
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-9:
            quat_out = [float(w), float(x), float(y), float(z)]  # 재구성 불가
        else:
            r_wi = quat_wxyz_to_mat(w / n, x / n, y / n, z / n)
            r_wb = mat_mul3(r_wi, mat_t3(mount_r))  # world←base
            quat_out = mat_to_quat_wxyz(r_wb)
        calibrated = True
    return {
        "timestamp": stamp_sec + stamp_nanosec * 1e-9,
        "gyro_rad_s": gyro_out,
        "quat_wxyz": quat_out,
        "accel_m_s2": accel_out,
        "mount_calibrated": calibrated,
    }


def selftest():
    """ROS 없이 순수 변환 검증 — 실패 시 AssertionError로 즉사."""
    # 1) 쿼터니언 재배열: 필드 (x=1,y=2,z=3,w=4) → [w,x,y,z] = [4,1,2,3]
    d = imu_to_bridge_dict(1700000000, 500000000,
                           quat_xyzw=(1.0, 2.0, 3.0, 4.0),
                           gyro_xyz=(0.1, -0.2, 0.3),
                           accel_xyz=(0.0, 0.0, 9.81))
    assert d["quat_wxyz"] == [4.0, 1.0, 2.0, 3.0], d["quat_wxyz"]

    # 2) 타임스탬프: sec + nanosec*1e-9
    assert math.isclose(d["timestamp"], 1700000000.5, abs_tol=1e-6), d["timestamp"]

    # 3) 자이로·가속도는 순서 유지 그대로 통과
    assert d["gyro_rad_s"] == [0.1, -0.2, 0.3], d["gyro_rad_s"]
    assert d["accel_m_s2"] == [0.0, 0.0, 9.81], d["accel_m_s2"]

    # 4) JSON 직렬화 왕복 + 계약 필수 키 존재
    back = json.loads(json.dumps(d))
    assert back == d, "JSON 왕복 불일치"
    for k in ("timestamp", "gyro_rad_s", "quat_wxyz", "accel_m_s2",
              "mount_calibrated"):
        assert k in back, f"필수 키 누락: {k}"
    assert len(back["quat_wxyz"]) == 4 and len(back["gyro_rad_s"]) == 3
    assert back["mount_calibrated"] is False  # 보정 미지정 = false

    # 5) 항등 장착 회귀: mount_r=I면 종전 출력과 동일해야 함 (수치 오차 내)
    ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    hq = math.sin(0.2)  # 단위 쿼터니언: z축 0.4rad 회전, xyzw=(0,0,sin0.2,cos0.2)
    q_unit_xyzw = (0.0, 0.0, hq, math.cos(0.2))
    d_old = imu_to_bridge_dict(1, 0, q_unit_xyzw, (0.1, -0.2, 0.3),
                               (0.0, 0.0, 9.81))
    d_id = imu_to_bridge_dict(1, 0, q_unit_xyzw, (0.1, -0.2, 0.3),
                              (0.0, 0.0, 9.81), mount_r=ident)
    assert d_id["mount_calibrated"] is True
    assert d_id["gyro_rad_s"] == d_old["gyro_rad_s"]
    assert d_id["accel_m_s2"] == d_old["accel_m_s2"]
    # 쿼터니언은 행렬 왕복 경유 — 부호까지 포함해 1e-12 내 일치해야 함
    # (트레이스 분기는 w>0 쿼터니언을 그대로 복원)
    for a, b in zip(d_id["quat_wxyz"], d_old["quat_wxyz"]):
        assert abs(a - b) < 1e-12, (d_id["quat_wxyz"], d_old["quat_wxyz"])

    # 6) x-up 장착(실기 시나리오): base_x=(0,0,−1), base_y=(0,1,0),
    #    base_z=(1,0,0) (IMU 좌표) — 수직 매달림이면 IMU 중력 = (−1,0,0)
    r_bi = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    validate_rotation(r_bi)

    def check_attitude(r_wb_true, g_base_expect, label):
        """world←base 참값에서 입력 quat을 유도해 출력 검증.

        입력 orientation은 world←imu = R_wb_true @ R_base_imu (장착과 일관되게
        유도) → 출력 quat_wxyz는 world←base를 복원해야 하고, 출력 quat으로
        계산한 투영중력 R_wbᵀ@(0,0,−1)이 g_base_expect와 일치해야 한다.
        """
        r_wi = mat_mul3(r_wb_true, r_bi)
        qw, qx, qy, qz = mat_to_quat_wxyz(r_wi)
        out = imu_to_bridge_dict(0, 0, (qx, qy, qz, qw),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                 mount_r=r_bi)
        ow, ox, oy, oz = out["quat_wxyz"]
        r_out = quat_wxyz_to_mat(ow, ox, oy, oz)
        # 회전행렬 비교 (쿼터니언 부호 모호성 회피)
        err_r = max(abs(r_out[i][j] - r_wb_true[i][j])
                    for i in range(3) for j in range(3))
        assert err_r < 1e-9, f"{label}: world←base 오차 {err_r:.2e}"
        g_base = mat_vec3(mat_t3(r_out), [0.0, 0.0, -1.0])
        err_g = max(abs(g_base[i] - g_base_expect[i]) for i in range(3))
        assert err_g < 1e-9, f"{label}: 투영중력 {g_base} vs {g_base_expect}"
        return g_base

    # 6a) 매달림(베이스=월드 정렬): 출력 투영중력 ≈ (0,0,−1)
    ident3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    g = check_attitude(ident3, [0.0, 0.0, -1.0], "매달림")
    print(f"[selftest] x-up 장착 매달림: 출력 투영중력 = "
          f"({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}) OK")

    # 6b) 베이스 +X를 20° 아래로 (world Y축 +20° 회전) →
    #     투영중력 = (sin20°, 0, −cos20°)
    a20 = math.radians(20.0)
    r_wb_pitch = [[math.cos(a20), 0.0, math.sin(a20)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(a20), 0.0, math.cos(a20)]]
    check_attitude(r_wb_pitch, [math.sin(a20), 0.0, -math.cos(a20)], "앞기울임")

    # 6c) 자이로/가속도 리맵: R_bi @ (a,b,c) = (−c, b, a)
    d6 = imu_to_bridge_dict(0, 0, (0.0, 0.0, 0.0, 1.0),
                            gyro_xyz=(0.1, -0.2, 0.3),
                            accel_xyz=(1.0, 2.0, 3.0), mount_r=r_bi)
    assert all(abs(a - b) < 1e-12
               for a, b in zip(d6["gyro_rad_s"], [-0.3, -0.2, 0.1])), \
        d6["gyro_rad_s"]
    assert all(abs(a - b) < 1e-12
               for a, b in zip(d6["accel_m_s2"], [-3.0, 2.0, 1.0])), \
        d6["accel_m_s2"]
    assert d6["mount_calibrated"] is True

    # 7) 쿼터니언↔행렬 왕복 (4개 트레이스 분기 모두 커버)
    for q in ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
              [0.5, 0.5, 0.5, 0.5], [0.2, -0.4, 0.7, 0.5570]):
        n = math.sqrt(sum(v * v for v in q))
        qn = [v / n for v in q]
        q2 = mat_to_quat_wxyz(quat_wxyz_to_mat(*qn))
        # 부호 모호성: q와 −q는 같은 회전
        sgn = 1.0 if sum(a * b for a, b in zip(qn, q2)) >= 0 else -1.0
        assert all(abs(a - sgn * b) < 1e-9 for a, b in zip(qn, q2)), (qn, q2)

    print("[selftest] imu_adapter: 재배열 + 항등 회귀 + x-up 장착 회전 ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="sensor_msgs/Imu → 브리지 JSON(std_msgs/String) 어댑터")
    ap.add_argument("--selftest", action="store_true",
                    help="ROS 없이 변환 로직만 검증하고 종료")
    a, _ = ap.parse_known_args()  # --ros-args 통과 허용 (foot 어댑터와 동일)
    if a.selftest:
        return selftest()

    # 장착 보정: rclpy init 전에 로드 — 불량이면 기동 거부(안전 우선)
    try:
        mount_r, mount_path = load_mount_calib()
    except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
        print(f"[오류] IMU 장착 보정 파일 불량 — 기동 거부: {e}")
        print("       imu_mount_calib.py로 재보정하거나 파일을 제거하세요.")
        return 1

    # rclpy는 여기서만 import — selftest 경로가 ROS 미소싱 환경에서도 돌게 함
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String

    rclpy.init()
    node = Node("imu_adapter")
    if mount_r is None:
        node.get_logger().warning(
            "IMU 장착 보정 없음 — IMU 프레임 그대로 통과 "
            "(중력이 (0,0,−1)이 아니면 imu_mount_calib.py 실행)")
    else:
        node.get_logger().info(f"IMU 장착 보정 적용: {mount_path}")
        for name, row in zip(("base_x", "base_y", "base_z"), mount_r):
            node.get_logger().info(
                f"  {name}_in_imu = [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    qos = QoSProfile(depth=QOS_DEPTH)
    pub = node.create_publisher(String, OUT_TOPIC, qos)

    def cb(msg):
        d = imu_to_bridge_dict(
            msg.header.stamp.sec, msg.header.stamp.nanosec,
            (msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w),
            (msg.angular_velocity.x, msg.angular_velocity.y,
             msg.angular_velocity.z),
            (msg.linear_acceleration.x, msg.linear_acceleration.y,
             msg.linear_acceleration.z),
            mount_r=mount_r)
        pub.publish(String(data=json.dumps(d)))

    node.create_subscription(Imu, IN_TOPIC, cb, qos)
    node.get_logger().info(f"{IN_TOPIC} → {OUT_TOPIC} 변환 시작 (depth={QOS_DEPTH})")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
