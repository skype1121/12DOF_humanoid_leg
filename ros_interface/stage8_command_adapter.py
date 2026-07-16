"""Stage8 ROS command adapter.

UI가 ROS2 publisher/subscriber 객체를 직접 만지지 않도록 분리한 얇은 adapter다.
이 파일은 CAN/AK 송신 코드를 import하지 않는다.
"""

import json


class Stage8RosCommandAdapter:
    def __init__(self, node_name, command_topic, status_topic, status_callback):
        self.node_name = node_name
        self.command_topic = command_topic
        self.status_topic = status_topic
        self.status_callback = status_callback
        self.rclpy = None
        self.string_type = None
        self.node = None
        self.command_publisher = None
        self.status_subscription = None
        self.last_error = ""

    @property
    def is_ready(self):
        return self.node is not None and self.command_publisher is not None and self.string_type is not None

    def start(self):
        if self.node is not None:
            return True, "already_started"
        try:
            import rclpy
            from std_msgs.msg import String
        except Exception as error:
            self.last_error = str(error)
            return False, f"ROS unavailable: {error}"

        try:
            if not rclpy.ok():
                rclpy.init()
            self.rclpy = rclpy
            self.string_type = String
            self.node = rclpy.create_node(self.node_name)
            self.command_publisher = self.node.create_publisher(
                String,
                self.command_topic,
                10,
            )
            self.status_subscription = self.node.create_subscription(
                String,
                self.status_topic,
                self.status_callback,
                10,
            )
            self.last_error = ""
            return True, "started"
        except Exception as error:
            self.last_error = str(error)
            return False, f"ROS client start failed: {error}"

    def publish(self, payload):
        if not self.is_ready:
            return False, "stage8_publisher_unavailable", ""
        message = self.string_type()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            self.command_publisher.publish(message)
        except Exception as error:
            self.last_error = str(error)
            return False, f"publish_failed:{error}", message.data
        self.last_error = ""
        return True, "published", message.data

    def spin_once(self):
        if self.node is None or self.rclpy is None:
            return True, "not_started"
        try:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as error:
            self.last_error = str(error)
            return False, f"spin_failed:{error}"
        return True, "spun"

    def stop(self):
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
        self.node = None
        self.command_publisher = None
        self.status_subscription = None
        self.string_type = None
