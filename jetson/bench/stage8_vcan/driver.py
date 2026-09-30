"""stage8 벤치 구동기: 베이스라인 → ARM → 50 Hz SET_JOINT_TARGETS(사인) → 기록. ROS_DOMAIN_ID 는 호출자가 격리."""
import json, math, sys, time, os
import rclpy
from std_msgs.msg import String
OUT = sys.argv[1]; STREAM_S = float(sys.argv[2])
CMD = "/humanoid/stage8_12axis_mit_command"; ST = "/humanoid/stage8_12axis_mit_status"
assert os.environ.get("ROS_DOMAIN_ID") not in (None, "", "0"), "격리 도메인에서만"
JOINTS = ["left_hip_f_joint", "left_hip_a_joint", "left_hip_r_joint", "left_knee_joint", "left_ankle_f_joint", "left_ankle_r_joint",
          "right_hip_f_joint", "right_hip_a_joint", "right_hip_r_joint", "right_knee_joint", "right_ankle_f_joint", "right_ankle_r_joint"]
rclpy.init(); node = rclpy.create_node("s8bench_driver")
pub = node.create_publisher(String, CMD, 10)
st_log = []; pub_log = []
def on_st(m):
    try: d = json.loads(m.data)
    except Exception: return
    st_log.append((time.time(), d.get("command_seq"), d.get("last_command"), d.get("armed"), d.get("running"), d.get("last_reject_reason")))
node.create_subscription(String, ST, on_st, 10)
def send(obj): pub.publish(String(data=json.dumps(obj)))
def spin_for(s):
    t = time.time() + s
    while time.time() < t: rclpy.spin_once(node, timeout_sec=0.005)
spin_for(3.0)
send({"command": "SET_BASELINE_FROM_CURRENT_ALL"}); spin_for(1.5)
send({"command": "ARM_ALL"}); spin_for(1.5)
print("armed?", st_log[-1] if st_log else None, flush=True)
t0 = time.time(); k = 0
def tick():
    global k
    t = time.time(); rel = 2.0 * math.sin(2 * math.pi * 0.7 * (t - t0))
    pub_log.append((t, rel)); send({"command": "SET_JOINT_TARGETS", "targets": {j: rel for j in JOINTS}}); k += 1
timer = node.create_timer(0.02, tick)
spin_for(STREAM_S)
node.destroy_timer(timer)
send({"command": "STOP_ALL"}); spin_for(0.5)
json.dump({"pub": pub_log, "st": st_log}, open(OUT, "w"))
print("published", len(pub_log), "status", len(st_log), flush=True)
node.destroy_node(); rclpy.shutdown()
