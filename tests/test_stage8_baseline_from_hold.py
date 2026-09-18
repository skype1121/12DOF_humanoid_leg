"""stage8 기준(baseline) = 인수 홀드 명령각 회귀 (오프라인 전용, 실 CAN 없음).

0918 실기: 핸드오프 → SET_BASELINE_FROM_CURRENT_ALL 이 하중 처진 실측을 기준으로
잡아 ARM 직후 무릎이 2.6° 더 처지고(PD 추종오차 소실) 뒤꿈치 회전 낙상. 수정 후엔
인수 홀드 중이면 기준 = hang_stand 마지막 명령각, 홀드가 없으면 종전대로 실측.

Run: python3 -m pytest tests/test_stage8_baseline_from_hold.py -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402

SAG_DEG = 2.6   # 0918 실측 무릎 처짐


class BaselineFromHoldTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="stage8-baseline-hold-")
        self.addCleanup(tmp.cleanup)
        p = patch.object(s8, "TAKEOVER_STATE_FILE",
                         str(Path(tmp.name) / "absent_handoff_state.json"))
        p.start()
        self.addCleanup(p.stop)
        self.t = 1000.0
        self.core = s8.Stage8TwelveAxisMitControlCore(now_fn=lambda: self.t)
        # 12관절 신선한 실측 = 명령보다 SAG 만큼 처진 값 (모터 프레임 deg)
        self.cmd = {j: float(s8.STAND_ZERO_DEG_BY_JOINT[j]) + 10.0
                    for j in s8.JOINT_NAMES_12DOF}
        for j in s8.JOINT_NAMES_12DOF:
            self.core.latest_actual_deg_by_joint[j] = self.cmd[j] - SAG_DEG
            self.core.latest_feedback_time_by_joint[j] = self.t

    def test_hold_present_uses_command_angle(self):
        self.core.takeover_hold = {j: (self.cmd[j], 150.0, 4.0)
                                   for j in s8.JOINT_NAMES_12DOF}
        st = self.core._set_baseline_from_current_all()
        self.assertTrue(self.core.accepted)
        self.assertEqual(st["baseline_source"], "takeover_hold")
        for j in s8.JOINT_NAMES_12DOF:
            self.assertAlmostEqual(self.core.baseline_deg_by_joint[j], self.cmd[j])
            self.assertAlmostEqual(st["baseline_hold_minus_actual_deg"][j], SAG_DEG, places=3)
        # ARM: 슬루 기준점 0 = 홀드 명령 그대로 이어감 (실측 초기화면 −SAG 딥)
        st = self.core.handle_command({"command": s8.COMMAND_ARM_ALL})
        self.assertTrue(self.core.armed, st.get("last_reject_reason"))
        for j in s8.JOINT_NAMES_12DOF:
            self.assertEqual(self.core.commanded_relative_deg_by_joint[j], 0.0)

    def test_no_hold_falls_back_to_actual(self):
        self.core.takeover_hold = None
        st = self.core._set_baseline_from_current_all()
        self.assertTrue(self.core.accepted)
        self.assertEqual(st["baseline_source"], "actual")
        self.assertEqual(st["baseline_hold_minus_actual_deg"], {})
        for j in s8.JOINT_NAMES_12DOF:
            self.assertAlmostEqual(self.core.baseline_deg_by_joint[j], self.cmd[j] - SAG_DEG)
        st = self.core.handle_command({"command": s8.COMMAND_ARM_ALL})
        self.assertTrue(self.core.armed, st.get("last_reject_reason"))
        for j in s8.JOINT_NAMES_12DOF:   # 종전 동작 유지: 실측 기준 초기화
            self.assertAlmostEqual(self.core.commanded_relative_deg_by_joint[j], 0.0)

    def test_stale_feedback_still_rejected_even_with_hold(self):
        self.core.takeover_hold = {j: (self.cmd[j], 150.0, 4.0)
                                   for j in s8.JOINT_NAMES_12DOF}
        self.t += s8.FEEDBACK_TIMEOUT_SEC + 1.0
        self.core._set_baseline_from_current_all()
        self.assertFalse(self.core._baseline_is_set())
        self.assertIsNotNone(self.core.last_reject_reason)


if __name__ == "__main__":
    unittest.main()
