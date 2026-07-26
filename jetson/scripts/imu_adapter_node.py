"""IMU 어댑터 — humanoid_imu(sensor_msgs/Imu)를 RL 브리지 JSON(std_msgs/String)으로 변환.

입력: /humanoid/imu/data (sensor_msgs/Imu, ~100Hz, 기존 humanoid_imu 노드)
  - orientation: iAHRS RPY로 만든 쿼터니언 (메시지 필드 순서는 x, y, z, w)
  - angular_velocity rad/s, linear_acceleration m/s²
출력: /humanoid/imu (std_msgs/String, 입력과 동일 레이트) — 브리지 계약 JSON:
  {"timestamp": epoch초 float, "gyro_rad_s": [x,y,z],
   "quat_wxyz": [w,x,y,z], "accel_m_s2": [x,y,z]}

핵심 주의: sensor_msgs 쿼터니언은 필드가 (x,y,z,w) 순서지만 브리지 계약은
[w,x,y,z]. 이 재배열이 이 노드의 존재 이유이며, --selftest가
(x=1,y=2,z=3,w=4) → [4,1,2,3]을 실제로 검증한다.

구조: 변환은 순수 함수 imu_to_bridge_dict()로 분리, rclpy는 main() 안에서만
lazy import → --selftest는 ROS 소싱 없는 로컬 환경에서도 그대로 돈다.

QoS: 구독·발행 모두 depth 50, reliability 기본값(RELIABLE).

사용: python3 imu_adapter_node.py             (Jetson, ROS2 Humble 소싱 후)
      python3 imu_adapter_node.py --selftest  (ROS 불필요 — 변환 로직 검증)
"""
import argparse
import json
import math

IN_TOPIC = "/humanoid/imu/data"
OUT_TOPIC = "/humanoid/imu"
QOS_DEPTH = 50  # 양쪽 동일


def imu_to_bridge_dict(stamp_sec, stamp_nanosec, quat_xyzw, gyro_xyz, accel_xyz):
    """Imu 메시지 필드값 → 브리지 계약 dict (순수 함수, ROS 비의존).

    quat_xyzw: 메시지 필드 순서 그대로 (x, y, z, w) 튜플 →
    출력 quat_wxyz는 [w, x, y, z]로 재배열된다.
    """
    x, y, z, w = quat_xyzw
    return {
        "timestamp": stamp_sec + stamp_nanosec * 1e-9,
        "gyro_rad_s": [float(v) for v in gyro_xyz],
        "quat_wxyz": [float(w), float(x), float(y), float(z)],
        "accel_m_s2": [float(v) for v in accel_xyz],
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
    for k in ("timestamp", "gyro_rad_s", "quat_wxyz", "accel_m_s2"):
        assert k in back, f"필수 키 누락: {k}"
    assert len(back["quat_wxyz"]) == 4 and len(back["gyro_rad_s"]) == 3

    print("[selftest] imu_adapter: 쿼터니언 재배열 (x,y,z,w)->[w,x,y,z] 포함 ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="sensor_msgs/Imu → 브리지 JSON(std_msgs/String) 어댑터")
    ap.add_argument("--selftest", action="store_true",
                    help="ROS 없이 변환 로직만 검증하고 종료")
    a, _ = ap.parse_known_args()  # --ros-args 통과 허용 (foot 어댑터와 동일)
    if a.selftest:
        return selftest()

    # rclpy는 여기서만 import — selftest 경로가 ROS 미소싱 환경에서도 돌게 함
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String

    rclpy.init()
    node = Node("imu_adapter")
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
             msg.linear_acceleration.z))
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
