"""stage8 1010 요구 토크 상한(--target-torque-cap-nm) 회귀 (오프라인 전용 — 실 CAN·ROS 없음, 가짜 버스).

지킬 계약:
  1. 기본(인자 없음·0) = 종전: 보내는 프레임이 상한 코드가 없던 때와 바이트 동일, 묶은 횟수 0.
  2. 켜면 보내는 목표 = clip(내부 목표, 실측 ± degrees(상한/kp)). 내부 commanded 목표·슬루는 그대로.
  3. 실측이 따라오면 묶임이 풀린다(시뮬의 토크 자르기와 같은 꼴).
  4. 실측이 낡은 관절은 묶지 않는다(종전 동결 경로 그대로).
  5. 인자 검증: 0 = 끔, 범위 밖 거부.

Run: python3 -m pytest tests/test_stage8_torque_cap.py -q
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_stage8_latency_options import FakeBus, can_module, mid_soft_motor_deg  # noqa: E402

GLOBALS = ("FEEDBACK_AFTER_SEND", "FEEDBACK_WAIT_SEC", "SEND_ON_COMMAND", "CONTROL_PERIOD_SEC", "TARGET_TORQUE_CAP_NM", "TARGET_TORQUE_CAP_BY_NAME")


@unittest.skipIf(can_module is None, "python-can 없음")
class TorqueCapTest(unittest.TestCase):
    J = "right_hip_a_joint"          # 1010 실물에서 응답이 끊긴 관절(모터 8)

    def setUp(self):
        saved = {k: getattr(s8, k) for k in GLOBALS}
        self.addCleanup(lambda: [setattr(s8, k, v) for k, v in saved.items()])
        tmp = tempfile.TemporaryDirectory(prefix="stage8-cap-")
        self.addCleanup(tmp.cleanup)
        for name, value in (("TAKEOVER_STATE_FILE", str(Path(tmp.name) / "absent.json")),
                            ("MECH_FIXED_JOINTS", frozenset()), ("POLL_JOINTS", list(s8.JOINT_NAMES_12DOF))):
            p = patch.object(s8, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.t = 1000.0
        self.bus = FakeBus(reply=False)          # 실측을 시험이 직접 정한다(응답 없음)
        self.core = s8.Stage8TwelveAxisMitControlCore(now_fn=lambda: self.t)
        self.core.can_bus = self.bus
        p2 = patch.object(s8.Stage8TwelveAxisMitControlCore, "_get_or_open_can_bus", lambda core, mod: self.bus)
        p2.start()
        self.addCleanup(p2.stop)
        for j in s8.JOINT_NAMES_12DOF:
            self.core.latest_actual_deg_by_joint[j] = mid_soft_motor_deg(j)
            self.core.latest_feedback_time_by_joint[j] = self.t
        self.core.latest_feedback_time = self.t
        self.core._set_baseline_from_current_all()
        st = self.core.handle_command({"command": s8.COMMAND_ARM_ALL})
        self.assertTrue(self.core.armed, st.get("last_reject_reason"))
        self.mid = int(s8.get_joint_config(self.J)["motor_id"])
        self.base = self.core.baseline_deg_by_joint[self.J]

    def apply(self, argv):
        args, _ = s8.build_arg_parser().parse_known_args(argv)
        s8.apply_latency_args(args)

    def tick(self):
        self.bus.sent.clear()
        self.core.control_tick()
        return {int(m.arbitration_id): bytes(m.data) for m in self.bus.sent}

    def frame(self, joint, deg):
        return bytes(s8.build_12axis_mit_preview(joint, deg).frame_bytes)

    def run_ticks(self, n, desired):
        self.core.desired_relative_deg_by_joint[self.J] = desired
        out = []
        for _ in range(n):
            out.append(self.tick()[self.mid])
        return out

    def test_default_is_legacy_frames(self):
        for argv in ([], ["--target-torque-cap-nm", "0"]):
            self.setUp() if argv else None
            self.apply(argv)
            self.assertEqual(s8.TARGET_TORQUE_CAP_NM, 0.0)
            frames = self.run_ticks(5, 30.0)
            self.assertEqual(frames, [self.frame(self.J, self.base + min(30.0, 4.0 * (k + 1))) for k in range(5)])   # 종전: 틱당 4° 슬루 그대로
            st = self.core.get_status()
            self.assertFalse(any(k.startswith("target_torque_cap") for k in st))      # 꺼짐 = status 키도 종전 그대로
            self.assertEqual(self.core.torque_cap_counts, {})

    def test_cap_limits_sent_target_not_internal(self):
        self.apply(["--target-torque-cap-nm", "20"])
        kp = float(s8.get_joint_config(self.J)["kp"])
        e = math.degrees(20.0 / kp)
        self.assertAlmostEqual(s8.torque_cap_err_deg(self.J), e)
        frames = self.run_ticks(5, 30.0)
        want = [self.frame(self.J, self.base + min(4.0 * (k + 1), e)) for k in range(5)]
        self.assertEqual(frames, want)                                   # 실측(= 기준점)에서 e 넘게 못 간다
        self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[self.J], 20.0)   # 내부 목표는 슬루대로 진행
        n_capped = sum(1 for k in range(5) if 4.0 * (k + 1) > e)
        self.assertEqual(self.core.get_status()["target_torque_cap_counts"], {self.J: n_capped})
        self.assertLessEqual(kp * math.radians(e), 20.0 + 1e-9)          # 위치 항 요구 토크 ≤ 상한
        # 다른 관절(목표 = 실측)은 건드리지 않는다
        other = "left_knee_joint"
        self.assertEqual(self.tick()[int(s8.get_joint_config(other)["motor_id"])], self.frame(other, self.core.baseline_deg_by_joint[other]))

    def test_cap_releases_when_measurement_follows(self):
        self.apply(["--target-torque-cap-nm", "20"])
        e = s8.torque_cap_err_deg(self.J)
        self.run_ticks(8, 30.0)                                          # 내부 목표 30° 도달, 보내는 목표는 실측 + e
        self.core.latest_actual_deg_by_joint[self.J] = self.base + 12.0  # 관절이 따라왔다
        self.assertEqual(self.tick()[self.mid], self.frame(self.J, self.base + 12.0 + e))
        self.core.latest_actual_deg_by_joint[self.J] = self.base + 29.0  # 오차 1° — 상한 안
        before = dict(self.core.torque_cap_counts)
        self.assertEqual(self.tick()[self.mid], self.frame(self.J, self.base + 30.0))
        self.assertEqual(self.core.torque_cap_counts, before)
        self.core.latest_actual_deg_by_joint[self.J] = self.base + 50.0  # 반대쪽(밀려서 지나침)도 대칭
        self.assertEqual(self.tick()[self.mid], self.frame(self.J, self.base + 50.0 - e))

    def test_cap_includes_damping_term(self):
        """1011: 밀려 움직이는 관절에서도 kp·(목표−실측) − kd·속도 의 크기가 상한을 넘지 않는다."""
        class SimpleNamespace:                       # 피드백 대역: 속도·위치만 쓰고 나머지 필드는 0
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, name): return 0.0
        self.apply(["--target-torque-cap-nm", "25", "--target-torque-cap-joint", "hip_a=20"])
        cfg = s8.get_joint_config(self.J); kp, kd = float(cfg["kp"]), float(cfg["kd"])
        self.run_ticks(8, 30.0)
        for v in (-3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
            for want in (30.0, -30.0):
                self.core.desired_relative_deg_by_joint[self.J] = want
                for _ in range(20):
                    self.tick()
                self.core.latest_feedback_by_joint[self.J] = SimpleNamespace(velocity_rad_s=v, position_deg=self.base)
                self.core.latest_actual_deg_by_joint[self.J] = self.base
                self.tick()
                sent_deg = math.degrees(self.core.last_preview_by_joint[self.J].p_des_rad)
                tau = kp * math.radians(sent_deg - self.base) - kd * v
                self.assertLessEqual(abs(tau), 20.0 + 1e-6, (v, want, tau))
        # 속도를 모르면 종전(위치 항만)
        self.core.latest_feedback_by_joint[self.J] = None
        self.assertEqual(s8.torque_cap_damping_shift_deg(self.J, None), 0.0)
        self.assertEqual(s8.torque_cap_damping_shift_deg(self.J, SimpleNamespace(velocity_rad_s=float("nan"))), 0.0)

    def test_stale_joint_is_not_capped(self):
        self.apply(["--target-torque-cap-nm", "20"])
        self.run_ticks(3, 30.0)
        held = self.core.commanded_relative_deg_by_joint[self.J]
        self.core.latest_feedback_time_by_joint[self.J] = self.t - 10.0  # 그 관절 실측이 낡음 → 종전 동결(현 목표 재송신)
        counts = dict(self.core.torque_cap_counts)
        with patch.object(s8, "REAL_CAN_WRITE_ENABLED", True):
            got = self.tick()[self.mid]
        self.assertEqual(got, self.frame(self.J, self.base + held))
        self.assertEqual(self.core.torque_cap_counts, counts)

    def test_per_joint_cap_overrides_global(self):
        self.apply(["--target-torque-cap-nm", "25", "--target-torque-cap-joint", "hip_a=20"])
        kp = float(s8.get_joint_config(self.J)["kp"])
        self.assertAlmostEqual(s8.torque_cap_err_deg(self.J), math.degrees(20.0 / kp))
        self.assertAlmostEqual(s8.torque_cap_err_deg("left_hip_a_joint"), math.degrees(20.0 / float(s8.get_joint_config("left_hip_a_joint")["kp"])))
        knee = "right_knee_joint"
        self.assertAlmostEqual(s8.torque_cap_err_deg(knee), math.degrees(25.0 / float(s8.get_joint_config(knee)["kp"])))
        self.assertEqual(self.run_ticks(4, 30.0)[-1], self.frame(self.J, self.base + s8.torque_cap_err_deg(self.J)))
        self.assertEqual(self.core.get_status()["target_torque_cap_by_name"], {"hip_a": 20.0})
        # 전체 상한 없이 관절별만: 그 관절만 묶이고 나머지는 종전
        self.apply(["--target-torque-cap-joint", "hip_a=20"])
        self.assertEqual(s8.TARGET_TORQUE_CAP_NM, 25.0)      # apply 는 준 인자만 바꾼다(전체 상한은 앞 값 유지) — 끄려면 0 을 명시
        self.apply(["--target-torque-cap-nm", "0", "--target-torque-cap-joint", "hip_a=20"])
        self.assertIsNone(s8.torque_cap_err_deg(knee))
        self.assertIsNotNone(s8.torque_cap_err_deg(self.J))
        for bad in (["hip_a"], ["nojoint=20"], ["hip_a=0"], ["hip_a=30"]):
            with self.assertRaises(ValueError):
                self.apply(["--target-torque-cap-joint"] + bad)

    def test_arg_validation(self):
        for bad in ("5", "26", "-1", "nan"):
            with self.assertRaises(ValueError):
                self.apply(["--target-torque-cap-nm", bad])
        for ok in ("10", "25", "20.5"):
            self.apply(["--target-torque-cap-nm", ok])
            self.assertEqual(s8.TARGET_TORQUE_CAP_NM, float(ok))


if __name__ == "__main__":
    unittest.main()
