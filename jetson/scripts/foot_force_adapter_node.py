"""발바닥 힘 어댑터 — 압력센서 JSON을 브리지 계약 foot_force JSON으로 변환.

입력: /humanoid/pressure/data (std_msgs/String, 10Hz, 기존 압력 노드 JSON):
  {"timestamp", "source", "values":[8 ints 0-1023], "left":[4], "right":[4],
   "raw_ads1115":[8], "voltages":[8]}
출력: /humanoid/foot_force (std_msgs/String, 입력과 동일 레이트):
  {"timestamp", "left_n", "right_n", "left_raw_sum", "right_raw_sum",
   "calibrated": false}

변환: left_n = NEWTON_PER_COUNT × sum(left 카운트), 우측 동일.
"calibrated": false는 소비측에 '이 뉴턴값은 아직 실측 보정 전'임을 알리는 표식.

강건성: JSON 파싱/형식 오류 메시지는 스킵하고 카운트 — 1·21·41…번째마다만
경고 로그 (스팸 방지). 변환은 순수 함수로 분리, rclpy는 main()에서만 lazy
import → --selftest는 ROS 소싱 없이 로컬에서 돈다.

사용: python3 foot_force_adapter_node.py             (Jetson, ROS2 Humble 소싱 후)
      python3 foot_force_adapter_node.py --selftest  (ROS 불필요 — 변환 검증)
      ros2 run 없이 파라미터: --ros-args -p newton_per_count:=0.08
"""
import argparse
import json
import math

IN_TOPIC = "/humanoid/pressure/data"
OUT_TOPIC = "/humanoid/foot_force"
QOS_DEPTH = 10  # 입력이 10Hz라 1초 버퍼면 충분
ERR_LOG_EVERY = 20

#: ★ 미보정 플레이스홀더 — 지면 스탠딩 전 실측 보정 필수! ★
#: 카운트→뉴턴 변환 계수. 로봇을 저울에 세워 총 카운트 합 vs 실제 체중(N)으로
#: 재산정해야 한다. 접촉 판정 임계 5N과 연동되므로 이 값이 틀리면 접촉 감지가
#: 통째로 틀어진다. ROS 파라미터 newton_per_count 로 런타임 override 가능.
NEWTON_PER_COUNT = 0.05


def pressure_to_foot_force(raw, newton_per_count):
    """압력 JSON dict → foot_force dict (순수 함수, ROS 비의존).

    필수: timestamp(숫자), left/right(각 4개 숫자 리스트).
    형식이 어긋나면 KeyError/ValueError/TypeError — 호출측이 잡아서 스킵.
    """
    left = raw["left"]
    right = raw["right"]
    if len(left) != 4 or len(right) != 4:
        raise ValueError(f"left/right 길이 오류: {len(left)}/{len(right)} (4 필요)")
    left_sum = sum(int(v) for v in left)
    right_sum = sum(int(v) for v in right)
    return {
        "timestamp": float(raw["timestamp"]),
        "left_n": newton_per_count * left_sum,
        "right_n": newton_per_count * right_sum,
        "left_raw_sum": left_sum,
        "right_raw_sum": right_sum,
        "calibrated": False,
    }


def selftest():
    """ROS 없이 순수 변환 검증 — 실패 시 AssertionError로 즉사."""
    # 1) 정상 샘플: 합산과 뉴턴 스케일링
    sample = json.dumps({
        "timestamp": 123.456, "source": "ads1115",
        "values": [100, 200, 300, 400, 10, 20, 30, 40],
        "left": [100, 200, 300, 400], "right": [10, 20, 30, 40],
        "raw_ads1115": [0] * 8, "voltages": [0.0] * 8,
    })
    d = pressure_to_foot_force(json.loads(sample), NEWTON_PER_COUNT)
    assert d["left_raw_sum"] == 1000, d["left_raw_sum"]
    assert d["right_raw_sum"] == 100, d["right_raw_sum"]
    assert math.isclose(d["left_n"], 0.05 * 1000), d["left_n"]     # = 50.0 N
    assert math.isclose(d["right_n"], 0.05 * 100), d["right_n"]    # = 5.0 N
    assert d["calibrated"] is False
    assert math.isclose(d["timestamp"], 123.456)

    # 2) 계수 override 반영 확인
    d2 = pressure_to_foot_force(json.loads(sample), 0.1)
    assert math.isclose(d2["left_n"], 100.0), d2["left_n"]

    # 3) JSON 직렬화 왕복 + 계약 필수 키
    back = json.loads(json.dumps(d))
    for k in ("timestamp", "left_n", "right_n",
              "left_raw_sum", "right_raw_sum", "calibrated"):
        assert k in back, f"필수 키 누락: {k}"

    # 4) 손상 입력은 반드시 예외로 튕겨야 함 (노드가 잡아서 스킵)
    bads = [
        {"timestamp": 1.0, "right": [1, 2, 3, 4]},                    # left 누락
        {"timestamp": 1.0, "left": [1, 2, 3], "right": [1, 2, 3, 4]},  # 길이 3
        {"timestamp": 1.0, "left": [1, 2, "x", 4], "right": [1, 2, 3, 4]},  # 비숫자
        {"left": [1, 2, 3, 4], "right": [1, 2, 3, 4]},                # timestamp 누락
    ]
    for bad in bads:
        try:
            pressure_to_foot_force(bad, NEWTON_PER_COUNT)
        except (KeyError, ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"손상 입력이 통과됨: {bad}")

    print("[selftest] foot_force_adapter: 합산·뉴턴 스케일·손상입력 거부 ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="압력 JSON → foot_force JSON(std_msgs/String) 어댑터")
    ap.add_argument("--selftest", action="store_true",
                    help="ROS 없이 변환 로직만 검증하고 종료")
    a, _ = ap.parse_known_args()  # --ros-args 통과 허용
    if a.selftest:
        return selftest()

    # rclpy는 여기서만 import — selftest 경로가 ROS 미소싱 환경에서도 돌게 함
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String

    rclpy.init()
    node = Node("foot_force_adapter")
    node.declare_parameter("newton_per_count", NEWTON_PER_COUNT)
    qos = QoSProfile(depth=QOS_DEPTH)
    pub = node.create_publisher(String, OUT_TOPIC, qos)
    stat = {"ok": 0, "err": 0}

    def cb(msg):
        # 파라미터를 매 콜백 재조회 (10Hz라 저렴) — 보정 작업 중
        # `ros2 param set … newton_per_count X` 가 즉시 반영되게 함
        npc = float(node.get_parameter("newton_per_count").value)
        try:
            out = pressure_to_foot_force(json.loads(msg.data), npc)
        except Exception as e:
            stat["err"] += 1
            if stat["err"] % ERR_LOG_EVERY == 1:  # 1·21·41…번째만 로그
                node.get_logger().warning(
                    f"압력 JSON 파싱 실패 누적 {stat['err']}건 "
                    f"(최근: {type(e).__name__}: {e}) — 스킵")
            return
        stat["ok"] += 1
        pub.publish(String(data=json.dumps(out)))

    node.create_subscription(String, IN_TOPIC, cb, qos)
    node.get_logger().info(
        f"{IN_TOPIC} → {OUT_TOPIC} 변환 시작 (newton_per_count="
        f"{float(node.get_parameter('newton_per_count').value)}, 미보정 플레이스홀더)")
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
