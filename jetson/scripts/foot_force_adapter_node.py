"""발바닥 힘 어댑터 — 압력센서 JSON을 브리지 계약 foot_force JSON으로 변환.

입력: /humanoid/pressure/data (std_msgs/String, 10Hz, 기존 압력 노드 JSON):
  {"timestamp", "source", "values":[8 ints 0-1023], "left":[4], "right":[4],
   "raw_ads1115":[8], "voltages":[8]}
출력: /humanoid/foot_force (std_msgs/String, 입력과 동일 레이트):
  {"timestamp", "left_n", "right_n", "left_raw_sum", "right_raw_sum",
   "calibrated": true}

변환: left_n = NEWTON_PER_COUNT × max(0, sum(left 카운트) − LEFT_BIAS_COUNTS),
우측 동일. "calibrated": true = 실측 보정 완료 (2026-08-02 자립 스탠딩 보정).

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

#: ★ 실측 보정 완료 (2026-08-02, 트림0 자립 스탠딩) ★
#: 절차: 무부하(매달림 0727) 카운트 = L328/R62 → 바이어스.
#:       자립 스탠딩(줄·손 무접촉, 0802) 20표본 평균 = L3356/R3559 카운트.
#:       체중 12.0kg×9.81 = 117.7N → npc = 117.7/((3356−328)+(3559−62)) = 0.01804.
#:       검산: L 54.6N + R 63.1N = 117.7N (좌우 46/54% — 당시 IMU 좌우 1.1°와 일치).
#: 구 플레이스홀더 0.05는 2.8배 과대 — 접촉 임계 5N 판정이 통째로 틀어졌었음.
#: ROS 파라미터 newton_per_count/left_bias_counts/right_bias_counts 로 override 가능.
NEWTON_PER_COUNT = 0.01804
LEFT_BIAS_COUNTS = 328.0
RIGHT_BIAS_COUNTS = 62.0


def pressure_to_foot_force(raw, newton_per_count,
                           left_bias=LEFT_BIAS_COUNTS,
                           right_bias=RIGHT_BIAS_COUNTS):
    """압력 JSON dict → foot_force dict (순수 함수, ROS 비의존).

    필수: timestamp(숫자), left/right(각 4개 숫자 리스트).
    형식이 어긋나면 KeyError/ValueError/TypeError — 호출측이 잡아서 스킵.
    바이어스 차감 후 음수는 0으로 바닥 처리 (무부하 노이즈가 음수 힘으로
    새어나가 접촉 판정을 흔들지 않게).
    """
    left = raw["left"]
    right = raw["right"]
    if len(left) != 4 or len(right) != 4:
        raise ValueError(f"left/right 길이 오류: {len(left)}/{len(right)} (4 필요)")
    left_sum = sum(int(v) for v in left)
    right_sum = sum(int(v) for v in right)
    return {
        "timestamp": float(raw["timestamp"]),
        "left_n": newton_per_count * max(0.0, left_sum - left_bias),
        "right_n": newton_per_count * max(0.0, right_sum - right_bias),
        "left_raw_sum": left_sum,
        "right_raw_sum": right_sum,
        "calibrated": True,
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
    assert math.isclose(d["left_n"],
                        NEWTON_PER_COUNT * (1000 - LEFT_BIAS_COUNTS)), d["left_n"]
    assert math.isclose(d["right_n"],
                        NEWTON_PER_COUNT * (100 - RIGHT_BIAS_COUNTS)), d["right_n"]
    assert d["calibrated"] is True
    assert math.isclose(d["timestamp"], 123.456)

    # 2) 계수 override 반영 확인 (바이어스 0으로 스케일만 검증)
    d2 = pressure_to_foot_force(json.loads(sample), 0.1, 0.0, 0.0)
    assert math.isclose(d2["left_n"], 100.0), d2["left_n"]

    # 2b) 바이어스 바닥 처리: 무부하 수준(바이어스 이하) 카운트 → 정확히 0N
    low = {"timestamp": 1.0, "left": [80, 80, 80, 80], "right": [10, 10, 10, 10]}
    d3 = pressure_to_foot_force(low, NEWTON_PER_COUNT)
    assert d3["left_n"] == 0.0 and d3["right_n"] == 0.0, (d3["left_n"], d3["right_n"])

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
    node.declare_parameter("left_bias_counts", LEFT_BIAS_COUNTS)
    node.declare_parameter("right_bias_counts", RIGHT_BIAS_COUNTS)
    qos = QoSProfile(depth=QOS_DEPTH)
    pub = node.create_publisher(String, OUT_TOPIC, qos)
    stat = {"ok": 0, "err": 0}

    def cb(msg):
        # 파라미터를 매 콜백 재조회 (10Hz라 저렴) — 보정 작업 중
        # `ros2 param set … newton_per_count X` 가 즉시 반영되게 함
        npc = float(node.get_parameter("newton_per_count").value)
        lb = float(node.get_parameter("left_bias_counts").value)
        rb = float(node.get_parameter("right_bias_counts").value)
        try:
            out = pressure_to_foot_force(json.loads(msg.data), npc, lb, rb)
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
        f"{float(node.get_parameter('newton_per_count').value)}, 바이어스 "
        f"L{float(node.get_parameter('left_bias_counts').value):.0f}/"
        f"R{float(node.get_parameter('right_bias_counts').value):.0f} — "
        f"실측 보정 2026-08-02)")
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
