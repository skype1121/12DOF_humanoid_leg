"""RL 브리지 노드 — 정책 실행기 → Stage8 세트포인트 (실물 이식용, 2026-07-27).

구조 (전부 기검증 조각의 조립):
  센서 구독 → PolicyRunnerStage4(관측245 조립·비트일치 검증본·리밋 클램프 내장)
  → 50Hz 틱마다 12관절 목표각 → Stage8 SET_JOINT_TARGET ×12 발행
  (12×50Hz = 600msg/s — 목 벤치 검증 여유 PASS. 노드 측 게이트/슬루/절대각
  리밋 검사 그대로 통과하므로 브리지가 뚫려도 노드가 막는다.)

토픽 규약:
  구독 /humanoid/joint_states            Stage8 노드 발행 (JSON, actual deg)
  구독 /humanoid/stage8_12axis_mit_status  armed·baseline_deg 확인
  구독 /humanoid/imu                     iAHRS 드라이버 (JSON):
      {"timestamp": epoch초, "gyro_rad_s": [wx,wy,wz](body),
       "quat_wxyz": [w,x,y,z](world<-body)}          ※ 드라이버 미구현 — 규약 확정용
  구독 /humanoid/foot_force              RA30P 드라이버 (JSON):
      {"timestamp": epoch초, "left_n": 법선힘N, "right_n": N}   ※ 미구현
  구독 /humanoid/rl_bridge_command       조종 (JSON):
      {"cmd": "stand"|"walk"|"march"|"stop"|"estop",
       "vx":0.0,"vy":0.0,"wz":0.0, "heading_hold": true}
  발행 /humanoid/stage8_12axis_mit_command   (--live 일 때만)
  발행 /humanoid/rl_bridge_debug            매 틱 진단 (드라이런 목표각 포함)

안전 (기본 = 드라이런):
  - --live 없으면 Stage8 명령 발행 안 함 (debug 토픽만) — 이중: 노드 armed 게이트
  - 센서 신선도 0.2s 초과 → 즉시 정지(명령 중단 + STOP_ALL 1회) — 정책은 stale
    관측으로 오동작하므로 멈추는 게 안전
  - estop 명령 → STOP_ALL 발행 + 루프 정지
  - 체크포인트: walk=R4(12994) 기본. march 전환은 접지 테스트 단계에서만.
  - kd는 노드 하드웨어맵 관할(kd5) — 브리지는 게인 미접촉. 시뮬 kd25 금지 원칙.

드라이런 자가검증 (로봇·ROS 불필요):
  python3 jetson/scripts/rl_bridge_node.py --selftest
  → 목 센서(정지 직립)로 500틱 돌려 목표각 유한성·리밋·틱 주기 검증
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl_walking.deploy.policy_runner_stage4 import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N, HeadingHold, PolicyRunnerStage4,
    yaw_from_quat_wxyz)
from rl_walking.deploy.policy_runner import JOINT_ORDER  # noqa: E402

RATE_HZ = 50.0
TICK_SEC = 1.0 / RATE_HZ
SENSOR_STALE_SEC = 0.2
COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
JOINT_STATES_TOPIC = "/humanoid/joint_states"
IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"
BRIDGE_CMD_TOPIC = "/humanoid/rl_bridge_command"
BRIDGE_DEBUG_TOPIC = "/humanoid/rl_bridge_debug"

CHECKPOINTS = {
    "walk": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-07-26_14-53-03/exported"),
    "march": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-07-26_21-41-30/exported"),
}


class BridgeCore:
    """ROS 무관 코어 — 관측 딕셔너리 → 12관절 목표각(deg, 절대). 셀프테스트 공유."""

    def __init__(self, checkpoint="walk"):
        self.runner = PolicyRunnerStage4(CHECKPOINTS[checkpoint])
        self.ckpt_name = checkpoint
        self.hh = HeadingHold(kp=1.0, wz_limit=0.6)
        self.hh_ref_set = False
        self.cmd = (0.0, 0.0, 0.0)
        self.heading_hold = True
        self.active = False          # stand/walk/march 시작 여부

    def reset(self, checkpoint=None):
        if checkpoint and checkpoint != self.ckpt_name:
            self.runner = PolicyRunnerStage4(CHECKPOINTS[checkpoint])
            self.ckpt_name = checkpoint
        self.runner.reset()
        self.hh_ref_set = False

    def tick(self, gyro, quat_wxyz, qpos_rad, qvel_rad_s, contact2):
        """관측 → 목표각 12 [rad, 절대] (JOINT_ORDER 순)."""
        w, x, y, z = quat_wxyz
        # projected gravity (world -z 를 body 프레임으로)
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        grav = (gx, gy, gz)
        vx, vy, wz = self.cmd
        if self.heading_hold and abs(wz) < 1e-6:
            yaw = yaw_from_quat_wxyz(w, x, y, z)
            if not self.hh_ref_set:
                self.hh.set_ref(yaw)
                self.hh_ref_set = True
            wz = self.hh.update(yaw)
        return self.runner.step(gyro, grav, (vx, vy, wz),
                                qpos_rad, qvel_rad_s, contact2)


def run_selftest():
    """목 센서(정지 직립) 500틱 — 유한성·리밋·주기. ROS·로봇 불필요."""
    import numpy as np
    core = BridgeCore("walk")
    core.cmd = (0.0, 0.0, 0.0)
    core.reset()
    q = np.array(core.runner._limits_lo) * 0.0 if core.runner._limits_lo is not None \
        else np.zeros(12)
    gyro = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    qvel = np.zeros(12)
    t0 = time.perf_counter()
    for i in range(500):
        out = core.tick(gyro, quat, q, qvel, [1.0, 1.0])
        assert np.isfinite(out).all(), f"틱 {i}: 비유한 목표"
        if core.runner._limits_lo is not None:
            assert (out >= core.runner._limits_lo - 1e-9).all()
            assert (out <= core.runner._limits_hi + 1e-9).all()
        q = out  # 목표 추종 가정 (목)
    dt = (time.perf_counter() - t0) / 500
    assert dt < TICK_SEC, f"틱 연산 {dt*1000:.2f}ms ≥ 20ms — 젯슨에서 재확인 필요"
    print(f"[PASS] 브리지 셀프테스트: 500틱 유한·리밋 준수, 틱 {dt*1000:.2f}ms "
          f"(50Hz 예산 20ms, 백엔드 {core.runner.backend})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="실제 Stage8 명령 발행 (없으면 드라이런 — debug 토픽만)")
    ap.add_argument("--checkpoint", default="walk", choices=list(CHECKPOINTS))
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class RlBridgeNode(Node):
        def __init__(self):
            super().__init__("humanoid_rl_bridge")
            self.core = BridgeCore(args.checkpoint)
            self.live = bool(args.live)
            self.joint_deg = {}
            self.joint_ts = 0.0
            self.prev_qpos = None
            self.imu = None
            self.imu_ts = 0.0
            self.foot = None
            self.foot_ts = 0.0
            self.baseline = {}
            self.armed = False
            self.stopped_for_stale = False
            self.create_subscription(String, JOINT_STATES_TOPIC, self._on_js, 10)
            self.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
            self.create_subscription(String, IMU_TOPIC, self._on_imu, 50)
            self.create_subscription(String, FOOT_TOPIC, self._on_foot, 50)
            self.create_subscription(String, BRIDGE_CMD_TOPIC, self._on_cmd, 10)
            self.pub_cmd = self.create_publisher(String, COMMAND_TOPIC, 10)
            self.pub_dbg = self.create_publisher(String, BRIDGE_DEBUG_TOPIC, 10)
            self.timer = self.create_timer(TICK_SEC, self._tick)
            self.get_logger().info(
                f"RL 브리지 기동 — {'LIVE' if self.live else 'DRY-RUN'}, "
                f"checkpoint={self.core.ckpt_name}")

        # ---- 입력 콜백 ----
        def _on_js(self, msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            acts = d.get("joints") or d.get("actual_deg_by_joint") or {}
            if acts:
                self.joint_deg.update(
                    {k: float(v) for k, v in acts.items() if k in JOINT_ORDER})
                self.joint_ts = time.time()

        def _on_status(self, msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            self.armed = bool(d.get("armed"))
            js = d.get("joints") or {}
            for j, info in js.items():
                b = info.get("baseline_deg")
                if b is not None:
                    self.baseline[j] = float(b)

        def _on_imu(self, msg):
            try:
                d = json.loads(msg.data)
                self.imu = (tuple(d["gyro_rad_s"]), tuple(d["quat_wxyz"]))
                self.imu_ts = float(d.get("timestamp", time.time()))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        def _on_foot(self, msg):
            try:
                d = json.loads(msg.data)
                self.foot = (float(d["left_n"]), float(d["right_n"]))
                self.foot_ts = float(d.get("timestamp", time.time()))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        def _on_cmd(self, msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            c = d.get("cmd", "")
            if c == "estop":
                self._stop_all("estop")
                self.core.active = False
                return
            if c == "stop":
                self.core.cmd = (0.0, 0.0, 0.0)   # walk 정책 능동 서기 = 정지
                return
            if c in ("stand", "walk", "march"):
                ck = "march" if c == "march" else "walk"
                self.core.reset(checkpoint=ck)
                self.core.cmd = ((0.0, 0.0, 0.0) if c in ("stand", "march") else
                                 (float(d.get("vx", 0.3)), float(d.get("vy", 0.0)),
                                  float(d.get("wz", 0.0))))
                self.core.heading_hold = bool(d.get("heading_hold", True))
                self.core.active = True
                self.stopped_for_stale = False
                self.get_logger().info(f"RL {c} 시작 cmd={self.core.cmd}")

        # ---- 50Hz 틱 ----
        def _tick(self):
            now = time.time()
            if not self.core.active:
                return
            fresh = (now - self.joint_ts < SENSOR_STALE_SEC
                     and now - self.imu_ts < SENSOR_STALE_SEC
                     and now - self.foot_ts < SENSOR_STALE_SEC
                     and len(self.joint_deg) == 12)
            if not fresh:
                if not self.stopped_for_stale:
                    self._stop_all("sensor_stale")
                    self.stopped_for_stale = True
                return
            self.stopped_for_stale = False
            qpos = [math.radians(self.joint_deg[j]) for j in JOINT_ORDER]
            if self.prev_qpos is None:
                qvel = [0.0] * 12
            else:
                qvel = [(a - b) / TICK_SEC for a, b in zip(qpos, self.prev_qpos)]
            self.prev_qpos = qpos
            gyro, quat = self.imu
            contact2 = [1.0 if self.foot[0] > CONTACT_FORCE_THRESHOLD_N else 0.0,
                        1.0 if self.foot[1] > CONTACT_FORCE_THRESHOLD_N else 0.0]
            targets_rad = self.core.tick(gyro, quat, qpos, qvel, contact2)
            targets_deg = {j: math.degrees(float(v))
                           for j, v in zip(JOINT_ORDER, targets_rad)}
            dbg = {"timestamp": now, "live": self.live, "armed": self.armed,
                   "checkpoint": self.core.ckpt_name, "cmd": list(self.core.cmd),
                   "targets_deg": targets_deg}
            self.pub_dbg.publish(String(data=json.dumps(dbg)))
            if not (self.live and self.armed):
                return
            for j in JOINT_ORDER:
                base = self.baseline.get(j)
                if base is None:
                    continue   # baseline 미설정 관절은 노드가 어차피 거부
                rel = targets_deg[j] - base
                self.pub_cmd.publish(String(data=json.dumps(
                    {"command": "SET_JOINT_TARGET", "joint": j,
                     "target_deg": rel})))

        def _stop_all(self, reason):
            self.get_logger().warning(f"STOP_ALL ({reason})")
            if self.live:
                self.pub_cmd.publish(String(
                    data=json.dumps({"command": "STOP_ALL"})))

    rclpy.init()
    node = RlBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
