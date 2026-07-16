#!/usr/bin/env python3
"""Publish one Stage8 12-axis MIT test command as ROS2 std_msgs/String."""

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Send a Stage8 12-axis MIT test command.")
    parser.add_argument("--joint", default="left_knee_joint")
    parser.add_argument("--deg", type=float, default=1.0)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def build_payload(args):
    selected = [args.arm, args.baseline, args.stop]
    if sum(1 for value in selected if value) > 1:
        raise ValueError("choose only one of --arm, --baseline, or --stop")
    if args.arm:
        return {"command": "ARM_ALL"}
    if args.baseline:
        return {"command": "SET_BASELINE_FROM_CURRENT_ALL"}
    if args.stop:
        return {"command": "STOP_ALL"}
    return {
        "command": "SET_JOINT_TARGET",
        "joint": str(args.joint),
        "target_deg": float(args.deg),
    }


def load_ros2():
    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError as error:
        return None, None, error
    return rclpy, String, None


def main(argv=None):
    args = parse_args(argv)
    try:
        payload = build_payload(args)
    except ValueError as error:
        print(f"[Stage8CommandSender] error = {error}")
        return 2

    rclpy, String, error = load_ros2()
    if rclpy is None:
        print(f"[Stage8CommandSender] ROS2 import failed: {error}")
        return 1

    rclpy.init()
    node = rclpy.create_node("stage8_12axis_mit_test_command_sender")
    publisher = node.create_publisher(String, COMMAND_TOPIC, 10)
    message = String()
    message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    deadline = time.time() + 1.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    publisher.publish(message)
    rclpy.spin_once(node, timeout_sec=0.1)
    print(f"[Stage8CommandSender] topic = {COMMAND_TOPIC}")
    print(f"[Stage8CommandSender] payload = {message.data}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
