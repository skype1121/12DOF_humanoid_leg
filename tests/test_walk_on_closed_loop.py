"""걷기 켜기 폐루프 회귀(오프라인 — 실 CAN·ROS·GPU 없음): stage8 코어(가짜 버스) ↔ 브리지 노드(가짜 ROS·엄격 로거) ↔ **실제 합성 키·실제 정책 파일**.

왜(1010 실물): 걷기로 넘어가는 순간마다 죽었다 — ① 브리지가 로그 등급 문제로 종료(정책 시작 직후), ② 걷기 켠 0.19 s 뒤 right_hip_a 목표−실측 14° →
모터 8 응답 정지. 여기서 고정하는 것:
  A. 콘솔과 같은 순서(stand 거부 → 재전송 수락 → 서기 → step 켜기(기한 2 s) → 스스로 끄기 → 서기)가 예외·STOP_ALL 없이 끝까지 간다.
  B. 요구 토크 상한을 켜면 **보내는 프레임**의 |목표−실측| 이 관절별 허용 오차(degrees(상한/kp))를 넘지 않는다 — 관절이 눌려 못 따라오는 경우에도.
  C. 상한을 끄면 같은 조건에서 그 오차를 넘는다(이 시험이 사고 꼴을 실제로 만든다는 대조).
가짜 모터: 명령 위치를 그대로 따라가되, right_hip_a(모터 8)만 틱당 0.3° 로 느리게 따라간다(하중에 눌린 관절). IMU 는 직립·양발 접촉 고정 — 물리가 아니라
소프트웨어 경로 시험이다(걸음의 성패를 보지 않는다).

Run: python3 -m pytest tests/test_walk_on_closed_loop.py -q
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mech_fixed as M  # noqa: E402  (시험대·가짜 버스·임시 맵 도우미)

b, s8, H = M.b, M.s8, M.H
KEY = "stand_kew3qa_g21a_g24ra_w3nbje1_18997"
SLOW_ID, SLOW_JOINT, SLOW_DEG_PER_TICK = 8, "right_hip_a_joint", 0.3
Q = 0.06          # 프레임 위치 16 비트 양자화(25 rad / 65535 = 0.022°)가 목표·실측에 한 번씩 드는 여유 [deg]


def dec_deg(d0, d1):
    return math.degrees(((d0 << 8) | d1) / 65535.0 * 25.0 - 12.5)


class LagBus(M.SimBus):
    """SimBus + 모터 8 만 느리게 따라간다. 보낸 위치 명령마다 (모터, 목표°, 보내기 직전 실측°) 를 적는다."""

    def __init__(self, pos_deg):
        super().__init__(pos_deg)
        self.posf = {int(m): float(d) for m, d in pos_deg.items()}
        self.log = []

    def send(self, message, timeout=None):
        mid, d = int(message.arbitration_id), bytes(message.data)
        self.frames.append((mid, d))
        if mid not in self.posf or d == M.EXIT:
            return
        if d not in (M.ENTER, M.ZERO_CMD):
            tgt = dec_deg(d[0], d[1])
            self.log.append((mid, tgt, self.posf[mid]))
            if mid == SLOW_ID:
                e = tgt - self.posf[mid]
                self.posf[mid] += max(-SLOW_DEG_PER_TICK, min(SLOW_DEG_PER_TICK, e))
            else:
                self.posf[mid] = tgt
        p = M.enc(self.posf[mid])
        self.rx.append(M.can.Message(arbitration_id=mid, data=[mid, p[0], p[1], 0x7F, 0xF7, 0xFF, 30, 0], is_extended_id=False))


def run(map_path, cap_args, walk_ticks=170):
    rec = {"acks": [], "modes": [], "stops": [], "exc": None}
    clk = H.FakeClock(2_000_000_000.0)
    real_make = b._make_runner
    H._patch(b, clk)
    b._make_runner = real_make                      # 실제 러너(합성 키 = 실제 정책 파일 4 개)
    b.HARDWARE_MAP_PATHS = (map_path,)
    saved = {k: getattr(s8, k) for k in ("TARGET_TORQUE_CAP_NM", "TARGET_TORQUE_CAP_BY_NAME")}
    args, _ = s8.build_arg_parser().parse_known_args(cap_args)
    s8.apply_latency_args(args)

    def driver(node):
        with M.declared(s8, map_path), patch.object(s8, "TAKEOVER_STATE_FILE", os.path.join(os.path.dirname(map_path), "absent.json")):
            bus = LagBus(M.stand_pos())
            core = s8.Stage8TwelveAxisMitControlCore(now_fn=clk.time)
            core.can_bus = bus
            rec["bus"], rec["core"] = bus, core
            n_cmd, n_ack = [0], [0]

            def tick(bridge=True, imu=True):
                core.control_tick()
                st = core.get_status()
                if bridge:
                    node._on_status(H._Msg(json.dumps(st)))
                    if imu:
                        node._on_imu(H._Msg(json.dumps({"gyro_rad_s": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "timestamp": clk.t})))
                    node._on_foot(H._Msg(json.dumps({"left_n": 40.0, "right_n": 40.0, "left_contact": True, "right_contact": True, "timestamp": clk.t})))
                    node._tick()
                    msgs = node._pubs[b.COMMAND_TOPIC].msgs
                    for m in msgs[n_cmd[0]:]:
                        core.handle_command_data(m)
                        if '"STOP_ALL"' in m:
                            rec["stops"].append((round(clk.t - 2_000_000_000.0, 2), node.stop_reason))
                    n_cmd[0] = len(msgs)
                    acks = node._pubs[b.BRIDGE_ACK_TOPIC].msgs
                    rec["acks"] += [json.loads(a) for a in acks[n_ack[0]:]]
                    n_ack[0] = len(acks)
                    kw = (node.core.kew3_snapshot() if node.core.kew3 else {}) or {}
                    rec["modes"].append(kw.get("mode"))
                clk.t += 0.02

            for _ in range(30):
                tick(bridge=False)
            core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
            core.handle_command({"command": s8.COMMAND_ARM_ALL})
            assert core.armed
            # 콘솔 '정책 ON' 순서: IMU 가 아직 안 온 틱에 stand → 거부, 다시 보내 수락(1010 실물에서 브리지가 죽은 수열)
            tick(imu=False)
            node._on_cmd(H._Msg(json.dumps({"cmd": "stand"})))
            tick(imu=False)
            node._on_cmd(H._Msg(json.dumps({"cmd": "stand"})))
            for _ in range(3):
                tick()
            node._on_cmd(H._Msg(json.dumps({"cmd": "stand"})))
            for _ in range(330):                     # 워밍업·블렌딩 + 걷기 선행조건(양발 접촉·조용함 1 s)
                tick()
            rec["i_walk"] = len(bus.log)
            node._on_cmd(H._Msg(json.dumps({"cmd": "step", "enable": True, "req_id": "t-inplace-1", "seq": 1, "heading_loop": False,
                                            "vx": 0.0, "vy": 0.0, "wz": 0.0, "dur_s": 2.0})))
            for _ in range(walk_ticks):
                tick()
            rec["i_end"] = len(bus.log)
            for _ in range(150):                     # 기한 뒤 스스로 끄고 서기로
                tick()
            rec["active"] = bool(node.core.active)
            rec["kew3"] = node.core.kew3_snapshot()
            rec["status"] = core.get_status()

    H.PUB_LOG.clear()
    try:
        with M.quiet():
            rc = H._run_main(b, ["--live", "--force-frame", "--force-gains", "--checkpoint", KEY, "--experimental-w3", "--experimental-kew3",
                                 "--w3-heading-loop", "off"], driver)
    finally:
        for k, v in saved.items():
            setattr(s8, k, v)
    rec["rc"] = rc
    return rec


def max_err(rec, mid, i0=None, i1=None):
    rows = [r for r in rec["bus"].log[(rec["i_walk"] if i0 is None else i0):(rec["i_end"] if i1 is None else i1)] if r[0] == mid]
    return max(abs(t - m) for _, t, m in rows)


@unittest.skipUnless(os.path.exists(os.path.join(b.REPO, b._W3.W3NBJ_RUN_REL, "exported", "policy.onnx")), "새 몸 걷기 정책 파일 없음")
class WalkOnClosedLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="walk-on-test-")
        cls.map = M.write_map(cls._tmp.name, "map_fixed.json", fixed=M.FIXED)
        cls._saved = {k: getattr(b, k) for k in ("time", "load_frame_map", "_make_runner", "HARDWARE_MAP_PATHS")}
        cls.cap = run(cls.map, ["--target-torque-cap-nm", "25", "--target-torque-cap-joint", "hip_a=20"])
        cls.nocap = run(cls.map, ["--target-torque-cap-nm", "0"])

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            setattr(b, k, v)
        cls._tmp.cleanup()

    def check_sequence(self, rec):
        self.assertEqual(rec["rc"], 0)
        self.assertEqual(rec["stops"], [], "걷기 왕복 중 STOP_ALL")
        stand = [a for a in rec["acks"] if a.get("cmd") == "stand"]
        self.assertEqual([a["accepted"] for a in stand], [False, False, True], [a.get("why") for a in stand])      # 거부 뒤 수락(로그 등급이 바뀌는 수열)
        step = [a for a in rec["acks"] if a.get("cmd") == "step"]
        self.assertEqual([(a["accepted"]) for a in step], [True, True], [a.get("why") for a in step])      # 켜기 + 기한(2 s) 끄기 응답
        self.assertIn("기한", step[1]["why"])
        self.assertIn("w3", rec["modes"])
        self.assertEqual((rec["kew3"]["n_to_w3"], rec["kew3"]["n_to_ke"], rec["kew3"]["mode"]), (1, 1, "ke"))        # 켜기 → 기한 뒤 스스로 서기
        self.assertTrue(rec["active"])
        self.assertEqual(rec["bus"].sent_to(M.FIXED_IDS), [])                                                        # 기구 고정 모터로는 아무것도 안 나간다

    def test_A_walk_round_trip_runs_with_cap(self):
        self.check_sequence(self.cap)

    def test_A_walk_round_trip_runs_without_cap(self):
        self.check_sequence(self.nocap)

    def test_B_cap_bounds_sent_error(self):
        lim_a = math.degrees(20.0 / float(s8.get_joint_config(SLOW_JOINT)["kp"]))
        self.assertLessEqual(max_err(self.cap, SLOW_ID), lim_a + Q)
        lim = math.degrees(25.0 / 150.0)
        for j, mid in s8.CAN_ID_BY_JOINT.items():
            if mid in M.FIXED_IDS or mid == SLOW_ID:
                continue
            self.assertLessEqual(max_err(self.cap, mid), lim + Q, j)
        counts = self.cap["status"]["target_torque_cap_counts"]
        # 1011: 상한이 걸리는지는 정책에 달렸다(NBj-s2 는 상한 없이 14.2°, NBjD-s2 8.3°, NBjE-s1 3.1° — 허용 7.64°). 상한 없는 실행이 허용값을 넘을 때만 걸려야 하고, 안 넘으면 한 번도 걸리지 않아야 한다.
        if max_err(self.nocap, SLOW_ID) > lim_a + Q:
            self.assertGreater(counts.get(SLOW_JOINT, 0), 0, counts)
        else:
            self.assertEqual(counts.get(SLOW_JOINT, 0), 0, counts)
        self.assertEqual(self.cap["status"]["target_torque_cap_by_name"], {"hip_a": 20.0})

    def test_C_without_cap_status_has_no_cap_keys(self):
        lim_a = math.degrees(20.0 / float(s8.get_joint_config(SLOW_JOINT)["kp"]))
        e_no, e_cap = max_err(self.nocap, SLOW_ID), max_err(self.cap, SLOW_ID)
        if e_no > lim_a + Q:                                                              # 대조: 상한 없는 쪽이 허용값을 넘는 정책이면 상한 켠 쪽이 더 작아야 한다
            self.assertGreater(e_no, e_cap)
        else:                                                                             # 넘지 않는 정책이면 상한은 아무것도 바꾸지 않는다
            self.assertAlmostEqual(e_no, e_cap, places=6)
        self.assertFalse(any(k.startswith("target_torque_cap") for k in self.nocap["status"]))


if __name__ == "__main__":
    unittest.main()
