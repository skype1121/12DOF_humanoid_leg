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

#: ★ 실측 보정 (2026-08-02 카운트 앵커, 2026-09-06 체중 재보정) ★
#: 절차: 무부하(매달림 0727) 카운트 = L328/R62 → 바이어스.
#:       자립 스탠딩(줄·손 무접촉, 0802) 20표본 평균 = L3356/R3559 카운트
#:       → 순카운트 (3356−328)+(3559−62) = 6525 = 체중 1.0 의 앵커.
#:       체중 = 17.62kg×9.81 = 172.9N (저울 실측 0904) → npc = 172.9/6525 = 0.02650.
#:       검산: L 80.2N + R 92.7N = 172.9N (좌우 46/54% — 당시 IMU 좌우 1.1°와 일치).
#: ⚠ 0802~0905 는 "12.0kg 추정 = 117.7N" 으로 npc 0.01804 를 썼다 — 보고 N 이 참값의 0.68배.
#:    그 시절 블랙박스(foot_n)는 ×1.469 (analyze_fall 이 날짜로 자동 보정). 백분율(발력/체중)은
#:    같은 카운트 앵커를 나누므로 구·신 모두 유효 — 4차 "90% 자립" 은 그대로 90% 다.
#: 구 플레이스홀더 0.05는 2.8배 과대 — 접촉 임계 5N 판정이 통째로 틀어졌었음.
#: 접촉 임계 5N(브리지 CONTACT_FORCE_THRESHOLD_N) = 시뮬 feet_contact threshold 5.0N 과 같은 참값 축
#:    (구 스케일에선 실효 7.35N 이었음).
#: ROS 파라미터 newton_per_count/left_bias_counts/right_bias_counts 로 override 가능.
NEWTON_PER_COUNT = 0.02650
LEFT_BIAS_COUNTS = 328.0
RIGHT_BIAS_COUNTS = 62.0
#: 0910 승윤님: 접지 판정은 '4개 패드 중 하나라도 눌리면 접지' (합산 5N 규칙은 발이 모서리로 닿을 때 왼발을 너무 일찍 비접촉으로 판정 →
#: 정책이 "왼발 떴다" 입력을 받고 반응, A1 3회 공통). 패드별 (count − pad_bias) > PAD_CONTACT_COUNTS 면 그 패드 눌림. 200 counts ≈ 5.3 N (0912 까지 80).
#: 0912: 왼발 index 2 패드 상시 눌림(고장, 429~728 카운트 표류) → 젯슨 sensors_up 에서 `-p left_pad_bias:="[0.0,0.0,5000.0,0.0]"` 로 판정 제외(패드 교체 전까지).
PAD_CONTACT_COUNTS = 200.0   # 0912 승윤님 결정: 80 → 200 카운트(≈5.3 N; 패드가 매우 민감해 80 은 스침도 눌림). 학습 접촉 임계 DR 2~20 N 안.


def pressure_to_foot_force(raw, newton_per_count,
                           left_bias=LEFT_BIAS_COUNTS,
                           right_bias=RIGHT_BIAS_COUNTS,
                           pad_contact_counts=PAD_CONTACT_COUNTS,
                           left_pad_bias=None, right_pad_bias=None):
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
    lpb = list(left_pad_bias) if left_pad_bias else [0.0] * 4
    rpb = list(right_pad_bias) if right_pad_bias else [0.0] * 4
    if len(lpb) != 4 or len(rpb) != 4:
        raise ValueError("pad bias 길이 오류 (4 필요)")
    l_on = [int(int(v) - float(b) > pad_contact_counts) for v, b in zip(left, lpb)]
    r_on = [int(int(v) - float(b) > pad_contact_counts) for v, b in zip(right, rpb)]
    return {
        "timestamp": float(raw["timestamp"]),
        "left_n": newton_per_count * max(0.0, left_sum - left_bias),
        "right_n": newton_per_count * max(0.0, right_sum - right_bias),
        "left_raw_sum": left_sum,
        "right_raw_sum": right_sum,
        # 0910: 패드 단위 접지 — 하나라도 눌리면 True (브리지 contact2 가 이 필드를 우선 사용)
        "left_contact": bool(any(l_on)),
        "right_contact": bool(any(r_on)),
        "left_pads_on": int(sum(l_on)),
        "right_pads_on": int(sum(r_on)),
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

    # 1b) 체중 앵커: 0802 자립 스탠딩 카운트(L3356/R3559) → 172.9N (저울 17.62kg) ±0.3N
    st = pressure_to_foot_force(
        {"timestamp": 0.0, "left": [3356, 0, 0, 0], "right": [3559, 0, 0, 0]}, NEWTON_PER_COUNT)
    tot = st["left_n"] + st["right_n"]
    assert abs(tot - 172.9) < 0.3, f"체중 앵커 어긋남: {tot:.2f}N ≠ 172.9N (npc={NEWTON_PER_COUNT})"

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

    # 4) 패드 단위 접지(0910): 한 패드만 눌려도 True, 예압 패드는 pad_bias 로 상쇄, 전부 무부하면 False
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 700, 0], "right": [0, 0, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is False and o["right_contact"] is False and o["left_pads_on"] == 0, o
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 700, 150], "right": [0, 90, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is True and o["left_pads_on"] == 1 and o["right_contact"] is True and o["right_pads_on"] == 1, o
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 760, 0], "right": [0, 0, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is False, "예압 패드 +60 카운트 요동은 접지 아님"
    # 5) 0912: 기본 임계 200 — 150 카운트 스침은 눌림 아님, 250 은 눌림; 고장 패드(왼 index 2)는 bias 5000 으로 제외
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 150, 600, 0], "right": [0, 0, 250, 0]}, 0.0265, 0.0, 0.0,
                               PAD_CONTACT_COUNTS, [0, 0, 5000, 0], None)
    assert PAD_CONTACT_COUNTS == 200.0 and o["left_contact"] is False and o["left_pads_on"] == 0 and o["right_contact"] is True and o["right_pads_on"] == 1, o
    print("[selftest] foot_force_adapter: 합산·뉴턴 스케일·손상입력 거부·패드 단위 접지(하나라도 눌리면 True, 예압 상쇄)·임계 200·고장 패드 제외 ALL PASS")
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
    node.declare_parameter("left_npc_scale", 1.0)   # 0911: per-foot 스케일(왼발 저평가 보정 — 균형스탠스 표본으로 정함, 기본 1.0=무변화)
    node.declare_parameter("right_npc_scale", 1.0)
    node.declare_parameter("newton_per_count", NEWTON_PER_COUNT)
    node.declare_parameter("left_bias_counts", LEFT_BIAS_COUNTS)
    node.declare_parameter("right_bias_counts", RIGHT_BIAS_COUNTS)
    node.declare_parameter("pad_contact_counts", PAD_CONTACT_COUNTS)
    node.declare_parameter("left_pad_bias", [0.0, 0.0, 0.0, 0.0])    # 패드별 무부하 카운트(예압 패드 타어) — 0910 왼[2] 700
    node.declare_parameter("right_pad_bias", [0.0, 0.0, 0.0, 0.0])
    qos = QoSProfile(depth=QOS_DEPTH)
    pub = node.create_publisher(String, OUT_TOPIC, qos)
    stat = {"ok": 0, "err": 0}

    def cb(msg):
        # 파라미터를 매 콜백 재조회 (10Hz라 저렴) — 보정 작업 중
        # `ros2 param set … newton_per_count X` 가 즉시 반영되게 함
        npc = float(node.get_parameter("newton_per_count").value)
        lsc = float(node.get_parameter("left_npc_scale").value); rsc = float(node.get_parameter("right_npc_scale").value)
        lb = float(node.get_parameter("left_bias_counts").value)
        rb = float(node.get_parameter("right_bias_counts").value)
        pcc = float(node.get_parameter("pad_contact_counts").value)
        lpb = [float(v) for v in node.get_parameter("left_pad_bias").value]
        rpb = [float(v) for v in node.get_parameter("right_pad_bias").value]
        try:
            out = pressure_to_foot_force(json.loads(msg.data), npc, lb, rb, pcc, lpb, rpb)
            out["left_n"] *= lsc; out["right_n"] *= rsc   # per-foot 스케일 (접촉 이진은 패드 카운트 기반이라 불변)
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
