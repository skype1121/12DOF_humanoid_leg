"""stage8 명령 전송기 — 발행 후 상태 검증·재시도 (유실 방지, 2026-08-02).

배경: ros2 topic pub --once 는 디스커버리 완료 전에 쏘면 조용히 유실됨
(0802 실사고: SET_BASELINE 유실 → 베이스라인 0/12 상태로 ARM → 재전송 순간
모터 튐과 겹침). 이 도구는 구독 확인 후 발행하고, 상태 토픽에서 효과를
검증할 때까지 재시도한다.

사용: python3 stage8_cmd.py baseline   # SET_BASELINE_FROM_CURRENT_ALL → 12/12 확인
      python3 stage8_cmd.py arm        # ARM_ALL → armed:true 확인
      python3 stage8_cmd.py disarm     # DISARM_ALL → armed:false 확인
      python3 stage8_cmd.py stop       # STOP_ALL (동결 홀드) → 발행 확인만
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CMD_TOPIC = "/humanoid/stage8_12axis_mit_command"
STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
COMMANDS = {
    "baseline": "SET_BASELINE_FROM_CURRENT_ALL",
    "arm": "ARM_ALL",
    "disarm": "DISARM_ALL",
    "stop": "STOP_ALL",
}


def verified(mode, status):
    joints = status.get("joints", {})
    n_base = sum(1 for v in joints.values()
                 if isinstance(v, dict) and v.get("baseline_deg") is not None)
    if mode == "baseline":
        return n_base == 12, f"베이스라인 {n_base}/12"
    if mode == "arm":
        return bool(status.get("armed")), f"armed={status.get('armed')}"
    if mode == "disarm":
        return not status.get("armed"), f"armed={status.get('armed')}"
    if mode == "stop":
        return True, "발행됨 (동결 홀드)"
    return False, "?"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"사용: stage8_cmd.py {{{'|'.join(COMMANDS)}}}")
        return 2
    mode = sys.argv[1]
    rclpy.init()
    node = Node("stage8_cmd_sender")
    pub = node.create_publisher(String, CMD_TOPIC, 10)
    latest = {}

    def on_status(msg):
        try:
            latest.update(json.loads(msg.data))
        except Exception:
            pass

    node.create_subscription(String, STATUS_TOPIC, on_status, 10)

    # 디스커버리 대기: stage8이 명령 구독자로 잡힐 때까지 (유실 원천 차단)
    t0 = time.monotonic()
    while pub.get_subscription_count() == 0:
        if time.monotonic() - t0 > 5.0:
            print("[FAIL] 명령 구독자 없음 — stage8 미기동?")
            return 1
        rclpy.spin_once(node, timeout_sec=0.1)

    payload = String(data=json.dumps({"command": COMMANDS[mode]}))
    for attempt in range(1, 4):
        pub.publish(payload)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            ok, detail = verified(mode, latest)
            if ok and latest:
                print(f"[OK] {COMMANDS[mode]} — {detail} (시도 {attempt})")
                node.destroy_node()
                rclpy.shutdown()
                return 0
        print(f"[재시도 {attempt}] {COMMANDS[mode]} — 최근 상태: "
              f"{verified(mode, latest)[1] if latest else '상태 미수신'}")
    print(f"[FAIL] {COMMANDS[mode]} 3회 실패 — stage8 로그 확인")
    node.destroy_node()
    rclpy.shutdown()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
