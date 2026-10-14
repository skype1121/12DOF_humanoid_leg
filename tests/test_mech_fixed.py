"""기구 고정 관절(하드웨어맵 "mech_fixed": true) 대조 시험 A~E — 오프라인 전용 (1009).

실 CAN·ROS·젯슨·GPU 를 쓰지 않는다: 버스는 가짜(SimBus — 프레임 기록 + 살아 있는 모터의 응답 합성), ROS 는 가짜 rclpy
(jetson/scripts/selftest_w3_bridge_node.py 의 시험대), 시계는 가짜, 하드웨어맵은 임시 사본(실제 맵은 읽기만), /tmp 트리거
파일은 임시 폴더로 돌린다(hang_stand 의 os·open 을 시험 안에서만 바꿔 끼움 — 실제 /tmp 파일을 만들거나 지우지 않는다).
pgrep·sudo·ip 를 부르는 경로(release_all 게이트·ping_all 버스 복구)는 시험에서 막아 둔다.

  (A) 고정 선언 + 고정 모터(3·9) 무응답 → prep(boot_pose_check·ping_all·hang_stand 핸드오프) → stage8 인수 홀드·기준점 12/12·
      ARM 수락 → 브리지 stand 가 신선도 게이트를 지나 정책 목표를 보낸다. 고정 모터로 나간 CAN 프레임 0 개.
  (B) 고정 선언 + **살아 있는 관절 하나 stale** → 브리지 sensor_stale 정지·stage8 60 s 자동 무장 해제가 선언 없는 대조군과
      같은 틱에 일어난다. 시작부터 죽은 산 모터 = ARM 거부. hang_stand 는 피드백 두절로 중단.
  (C) 고정 선언 없음 + 모터 3·9 무응답 → 종전과 똑같이 거부(수정 전 커밋의 stage8·브리지와 프레임·발행 비트 동일).
  (D) 고정 선언 없음 → 송신 프레임 열·브리지 발행이 수정 전 커밋과 바이트 동일.
  (E) 맵 사본 두 개의 mech_fixed 불일치 → 브리지 기동(--live 포함) 거부.

(C)·(D)의 '수정 전' 은 git 의 BASE_REV 에서 꺼낸 원본을 같은 프로세스에 따로 올려 같은 입력으로 돌린 것이다(git 이 없으면 건너뜀).
**계약·순서 검증이다 — 폐루프 성능이나 실물 안전(고정 부품의 강성·선언과 실물의 일치)은 증명하지 않는다.**

Run: python3 -m pytest tests/test_mech_fixed.py -v
"""
import contextlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time as _real_time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO / "protocols"), str(REPO / "jetson" / "scripts"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import can  # noqa: E402

import boot_pose_check as bpc  # noqa: E402
import hang_stand as hs  # noqa: E402
import mech_fixed  # noqa: E402
import ping_all as pa  # noqa: E402
import release_all as ra  # noqa: E402
import rl_bridge_node as b  # noqa: E402
import selftest_w3_bridge_node as H  # noqa: E402
import zero_anchor as za  # noqa: E402
from ak_mit_command import pack_ak_mit_command  # noqa: E402
from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402

BASE_REV = "2a43786d"                 # 수정 전 기준 커밋 (C·D 대조군)
REAL_MAP = REPO / "config" / "robot_12dof_hardware_map.json"
FIXED = ("left_hip_r_joint", "right_hip_r_joint")
FIXED_IDS = (3, 9)
ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
ZERO_CMD = bytes(pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
CKPT = "stand_crouch_g21a"
POLICY_ACTION = 0.2                   # 스텁 정책의 상수 출력(12 관절 전부) → 목표 = 기본자세 + 0.25 × 0.2 rad


def write_map(directory, name, fixed=(), value=True):
    """실제 맵의 임시 사본 — fixed 관절에만 "mech_fixed": value 를 단다. 실제 맵 파일은 건드리지 않는다."""
    with open(REAL_MAP, encoding="utf-8") as f:
        m = json.load(f)
    for j in fixed:
        m["joints"][j]["mech_fixed"] = value
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f)
    return path


def enc(deg):
    fr = pack_ak_mit_command(math.radians(float(deg)), 0.0, 0.0, 0.0, 0.0)
    return fr[0], fr[1]


class Clock:
    """가짜 시계 — time()·monotonic() 이 같은 값, sleep 은 값만 올리고 hook 을 부른다. 그 밖(strftime 등)은 진짜 time."""

    def __init__(self, t0=2_000_000_000.0):
        self.t = float(t0)
        self.hook = None

    def time(self):
        return self.t

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += max(0.0, float(dt))

    def sleep(self, dt):
        self.advance(dt)
        if self.hook is not None:
            self.hook()

    def __getattr__(self, name):
        return getattr(_real_time, name)


class SimBus:
    """가짜 CAN 버스. 보낸 프레임을 전부 적고, 살아 있는 모터만 응답한다(명령 위치를 그대로 되받음 = 완전 추종).
    ENTER(FC)·제로 명령에는 현재 위치로 응답, EXIT 에는 응답 없음. 빈 수신은 기다린 시간만큼 가짜 시계를 올린다."""

    def __init__(self, pos_deg, dead=(), clock=None):
        self.pos = {int(m): enc(d) for m, d in pos_deg.items()}
        self.dead = set(dead)
        self.clock = clock
        self.frames = []
        self.rx = []

    def send(self, message, timeout=None):
        mid, d = int(message.arbitration_id), bytes(message.data)
        self.frames.append((mid, d))
        if mid in self.dead or mid not in self.pos or d == EXIT:
            return
        if d not in (ENTER, ZERO_CMD):
            self.pos[mid] = (d[0], d[1])
        p = self.pos[mid]
        self.rx.append(can.Message(arbitration_id=mid, data=[mid, p[0], p[1], 0x7F, 0xF7, 0xFF, 30, 0],
                                   is_extended_id=False))

    def recv(self, timeout=None):
        if self.rx:
            return self.rx.pop(0)
        if self.clock is not None and timeout:
            self.clock.advance(timeout)
        return None

    def shutdown(self):
        pass

    def sent_to(self, ids):
        return [f for f in self.frames if f[0] in set(ids)]


def stand_pos(offset=1.0):
    return {m: float(s8.STAND_ZERO_DEG_BY_JOINT[j]) + offset for j, m in s8.CAN_ID_BY_JOINT.items()}


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


@contextlib.contextmanager
def keep_signals():
    sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old = {s: signal.getsignal(s) for s in sigs}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


@contextlib.contextmanager
def argv(*args):
    old, sys.argv = sys.argv, ["prog", *args]
    try:
        yield
    finally:
        sys.argv = old


@contextlib.contextmanager
def declared(mod, map_path):
    """stage8 모듈의 고정 관절 집합을 map_path 의 선언으로 바꾼다(모듈이 기동 때 하는 것과 같은 함수로 읽는다)."""
    names = frozenset(mod.load_mech_fixed(map_path)[0])
    with patch.object(mod, "MECH_FIXED_JOINTS", names), \
            patch.object(mod, "POLL_JOINTS", [j for j in mod.JOINT_NAMES_12DOF if j not in names]):
        yield names


def load_base(rel_path, name):
    """BASE_REV 의 파일을 같은 __file__ 경로로 실행한 별도 모듈(경로·맵 해석 동일). git 이 없으면 None."""
    try:
        src = subprocess.check_output(["git", "show", f"{BASE_REV}:{rel_path}"], cwd=str(REPO), stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / rel_path)
    sys.modules[name] = mod
    exec(compile(src, mod.__file__ + f"@{BASE_REV}", "exec"), mod.__dict__)
    return mod


_BASE = {}


def base_modules():
    if not _BASE:
        _BASE["s8"] = load_base("jetson/scripts/stage8_12axis_mit_control_node.py", "stage8_base_rev")
        _BASE["b"] = load_base("jetson/scripts/rl_bridge_node.py", "rl_bridge_node_base_rev")
    return _BASE["s8"], _BASE["b"]


def closed_loop(s8m, bm, map_path, declare, dead=(), kill=None, stand_ticks=200, after_stop_s=0.0, takeover_file=None,
                bus=None, t0=2_000_000_000.0):
    """stage8 코어(가짜 버스) ↔ 브리지 노드(가짜 ROS) 폐루프. 틱마다 stage8 status → 브리지, 브리지 명령 → stage8.

    순서: stage8 기동(인수 상태 파일이 있으면 인수 홀드, 없으면 읽기 폴링) → 기준점 → ARM → 브리지 stand → stand_ticks 틱.
    kill = (stand 뒤 틱, 모터 ID): 그 틱부터 그 모터가 응답하지 않는다. 브리지가 정지하면 after_stop_s 동안 stage8 만 더 돌린다."""
    rec = {"arm_reject": None, "stop_tick": None, "stop_reason": None, "disarm_tick": None, "stale_warn_tick": None,
           "targets": [], "dbg": []}
    clk = H.FakeClock(t0)
    H._patch(bm, clk)

    def const_policy(obs):
        import numpy as np
        return np.full(12, POLICY_ACTION, dtype=np.float32)

    bm._make_runner = lambda ck, det_kw=None: bm.PolicyRunnerStage4(
        model_path=const_policy, obs_dim=320, gait_freq=1.1, default_pose_rad=bm.CKPT_DEFAULT_POSE.get(ck),
        act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    bm.HARDWARE_MAP_PATHS = (map_path,)
    ctx = declared(s8m, map_path) if declare else contextlib.nullcontext()
    absent = os.path.join(os.path.dirname(map_path), "absent_handoff_state.json")

    def driver(node):
        with ctx, patch.object(s8m, "TAKEOVER_STATE_FILE", takeover_file or absent):
            sim = bus if bus is not None else SimBus(stand_pos(), dead=dead)
            sim.clock = None
            core = s8m.Stage8TwelveAxisMitControlCore(now_fn=clk.time)
            core.can_bus = sim
            rec["core"], rec["bus"], rec["node"] = core, sim, node
            rec["holding_at_start"] = core.takeover_hold is not None
            n_cmd = [0]

            def tick(bridge=True):
                core.control_tick()
                st = core.get_status()
                if bridge:
                    node._on_status(H._Msg(json.dumps(st)))
                    node._on_imu(H._Msg(json.dumps({"gyro_rad_s": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                                                    "timestamp": clk.t})))
                    node._on_foot(H._Msg(json.dumps({"left_n": 40.0, "right_n": 40.0, "left_contact": True,
                                                     "right_contact": True, "timestamp": clk.t})))
                    node._tick()
                    msgs = node._pubs[bm.COMMAND_TOPIC].msgs
                    for m in msgs[n_cmd[0]:]:
                        core.handle_command_data(m)
                        d = json.loads(m)
                        if d.get("command") == "SET_JOINT_TARGETS":
                            rec["targets"].append(d["targets"])
                    n_cmd[0] = len(msgs)
                clk.t += 0.02
                return st

            for _ in range(30):
                tick(bridge=False)
            rec["hold_status"] = core.get_status()["takeover"]
            rec["baseline_status"] = core.handle_command({"command": s8m.COMMAND_SET_BASELINE_ALL})
            st = core.handle_command({"command": s8m.COMMAND_ARM_ALL})
            rec["arm_reject"] = st["last_reject_reason"]
            rec["armed_after_arm"] = bool(core.armed)
            tick()
            node._on_cmd(H._Msg(json.dumps({"cmd": "stand"})))
            k = 0
            while k < stand_ticks:
                if kill is not None and k == kill[0]:
                    sim.dead.add(kill[1])
                n0 = len(node._pubs[bm.COMMAND_TOPIC].msgs)
                tick()
                k += 1
                if any('"STOP_ALL"' in m for m in node._pubs[bm.COMMAND_TOPIC].msgs[n0:]):
                    rec["stop_tick"], rec["stop_reason"] = k, node.stop_reason
                    break
            rec["active_at_end"] = bool(node.core.active)
            rec["status_at_end"] = core.get_status()
            rec["dbg"] = [json.loads(m) for m in node._pubs[bm.BRIDGE_DEBUG_TOPIC].msgs]
            for i in range(int(after_stop_s / 0.02)):
                st = tick()
                if rec["stale_warn_tick"] is None and any(w.startswith("joint_feedback_stale:") for w in st["warnings"]):
                    rec["stale_warn_tick"], rec["stale_warn"] = i, [w for w in st["warnings"] if w.startswith("joint_feedback_stale:")]
                if rec["disarm_tick"] is None and not core.armed:
                    rec["disarm_tick"], rec["disarm_warnings"] = i, list(st["warnings"])
            rec["pub_log"] = list(H.PUB_LOG)

    H.PUB_LOG.clear()
    with quiet():
        rc = H._run_main(bm, ["--live", "--force-frame", "--force-gains", "--checkpoint", CKPT], driver)
    assert rc == 0, f"브리지 main rc {rc}"
    return rec


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="mech-fixed-test-")
        cls.d = cls._tmp.name
        cls.map_none = write_map(cls.d, "map_none.json")
        cls.map_fixed = write_map(cls.d, "map_fixed.json", FIXED)
        # 브리지 블랙박스(~/logs)를 임시 폴더로, 브리지 모듈에 끼운 시험용 치환은 끝나고 되돌린다
        cls._home = os.environ.get("HOME")
        os.environ["HOME"] = cls.d
        cls._saved = {m: {k: getattr(m, k) for k in ("time", "load_frame_map", "_make_runner", "HARDWARE_MAP_PATHS")}
                      for m in (b,)}
        cls._bias = list(b.IMU_MOUNT_BIAS_DEG)
        cls._ros = {k: sys.modules.get(k) for k in ("rclpy", "rclpy.node", "std_msgs", "std_msgs.msg")}

    @classmethod
    def tearDownClass(cls):
        for m, saved in cls._saved.items():
            for k, v in saved.items():
                setattr(m, k, v)
        b.IMU_MOUNT_BIAS_DEG[0], b.IMU_MOUNT_BIAS_DEG[1] = cls._bias
        for k, v in cls._ros.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if cls._home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = cls._home
        cls._tmp.cleanup()


class DeclarationTest(unittest.TestCase):
    def test_real_map_declares_nothing_and_module_defaults_are_unchanged(self):
        """실제 맵에는 선언이 없다 → stage8 의 고정 집합은 비고 폴링 대상은 12 관절 그대로."""
        self.assertEqual(mech_fixed.load_fixed(str(REAL_MAP)), ((), ()))
        self.assertEqual(s8.MECH_FIXED_JOINTS, frozenset())
        self.assertEqual(list(s8.POLL_JOINTS), list(s8.JOINT_NAMES_12DOF))

    def test_ids_come_from_map_not_code(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mech_fixed.load_fixed(write_map(d, "m.json", FIXED)), (FIXED, FIXED_IDS))
            self.assertEqual(mech_fixed.load_fixed(write_map(d, "k.json", ("left_knee_joint",))), (("left_knee_joint",), (4,)))
            self.assertEqual(mech_fixed.load_fixed(write_map(d, "f.json", FIXED, value=False)), ((), ()))
            with self.assertRaises(ValueError):
                mech_fixed.load_fixed(write_map(d, "bad.json", FIXED, value="true"))

    def test_selftest(self):
        with quiet():
            self.assertEqual(mech_fixed._selftest(), 0)


class Stage8UnitTest(_Base):
    def core(self, dead=(), takeover=None, t=1000.0):
        self.t = t
        path = os.path.join(self.d, "s8_state.json")
        if os.path.exists(path):
            os.remove(path)
        if takeover is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"ts": self.t, "motors": takeover}, f)
        with patch.object(s8, "TAKEOVER_STATE_FILE", path):
            core = s8.Stage8TwelveAxisMitControlCore(now_fn=lambda: self.t)
        core.can_bus = SimBus(stand_pos(), dead=dead)
        return core

    def ticks(self, core, n):
        for _ in range(n):
            core.control_tick()
            self.t += 0.02

    def hold_file(self, ids):
        by_id = {m: j for j, m in s8.CAN_ID_BY_JOINT.items()}
        return {str(m): {"deg": float(s8.STAND_ZERO_DEG_BY_JOINT[by_id[m]]) + 1.0, "kp": 150.0, "kd": 5.0} for m in ids}

    def test_A_baseline_arm_and_zero_frames_to_fixed(self):
        with declared(s8, self.map_fixed):
            core = self.core(dead=FIXED_IDS)
            self.ticks(core, 30)
            st = core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
            self.assertTrue(st["accepted"])
            self.assertEqual((st["baseline_count"], st["baseline_complete"], st["baseline_missing_joints"]), (12, True, []))
            for j in FIXED:
                js = st["joints"][j]
                self.assertEqual(js["baseline_deg"], s8.STAND_ZERO_DEG_BY_JOINT[j])
                self.assertEqual(js["latest_actual_deg"], s8.STAND_ZERO_DEG_BY_JOINT[j])
                self.assertEqual((js["mech_fixed"], js["feedback_valid"], js["feedback_stale"], js["feedback_age_sec"]),
                                 (True, True, False, 0.0))
                self.assertEqual((js["feedback_velocity_rad_s"], js["feedback_torque_est"]), (0.0, 0.0))
            self.assertNotIn("mech_fixed", st["joints"]["left_knee_joint"])
            self.assertEqual(st["mech_fixed_joints"], list(FIXED))
            st = core.handle_command({"command": s8.COMMAND_ARM_ALL})
            self.assertTrue(core.armed, st["last_reject_reason"])
            # 고정 관절로 들어온 목표: 검증 뒤 버림 — 거부 사유 없음·스테이징 없음. 산 관절은 그대로 반영
            st = core.handle_command({"command": s8.COMMAND_SET_JOINT_TARGETS,
                                      "targets": {j: 2.0 for j in s8.JOINT_NAMES_12DOF}})
            self.assertTrue(st["accepted"])
            self.assertIsNone(st["last_reject_reason"])
            self.assertEqual(st["mech_fixed_target_drops"], 2)
            self.ticks(core, 20)
            st = core.get_status()
            for j in s8.JOINT_NAMES_12DOF:
                want = 0.0 if j in FIXED else 2.0
                self.assertEqual(st["joints"][j]["desired_relative_deg"], want, j)
                self.assertEqual(st["joints"][j]["commanded_relative_deg"], want, j)
            self.assertEqual(core.can_bus.sent_to(FIXED_IDS), [])
            self.assertEqual({m for m, _ in core.can_bus.frames}, set(range(1, 13)) - set(FIXED_IDS))
            # 불량 목표는 고정 관절이어도 종전대로 거부로 드러난다
            st = core.handle_command({"command": s8.COMMAND_SET_JOINT_TARGETS, "targets": {FIXED[0]: float("nan")}})
            self.assertFalse(st["accepted"])
            self.assertIn("target_not_finite", st["last_reject_reason"])

    def test_A_takeover_hold_accepts_all_but_fixed(self):
        live = [m for m in range(1, 13) if m not in FIXED_IDS]
        with declared(s8, self.map_fixed):
            core = self.core(dead=FIXED_IDS, takeover=self.hold_file(live))
            self.assertIsNotNone(core.takeover_hold, core.last_takeover_event)
            self.assertEqual(len(core.takeover_hold), 10)
            self.ticks(core, 25)
            tk = core.get_status()["takeover"]
            self.assertEqual((tk["holding"], tk["hold_joint_count"], tk["hold_mech_fixed_count"]), (True, 12, 2))
            self.assertEqual(core.can_bus.sent_to(FIXED_IDS), [])
            st = core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
            self.assertEqual((st["baseline_count"], st["baseline_source"]), (12, "takeover_hold"))
            core.handle_command({"command": s8.COMMAND_ARM_ALL})
            self.assertTrue(core.armed)
            # 옛 hang_stand 가 12 개를 다 적은 파일도 수락하되 고정 관절 항목은 쓰지 않는다
            core = self.core(dead=FIXED_IDS, takeover=self.hold_file(range(1, 13)))
            self.assertEqual(sorted(core.takeover_hold), sorted(j for j in s8.JOINT_NAMES_12DOF if j not in FIXED))
            # 산 관절이 하나라도 빠진 파일은 종전대로 거부
            core = self.core(dead=FIXED_IDS, takeover=self.hold_file([m for m in live if m != 4]))
            self.assertIsNone(core.takeover_hold)
            self.assertEqual(core.last_takeover_event, "state_file_incomplete:9/10")

    def test_C_undeclared_takeover_file_without_3_9_is_rejected(self):
        core = self.core(takeover=self.hold_file([m for m in range(1, 13) if m not in FIXED_IDS]))
        self.assertIsNone(core.takeover_hold)
        self.assertEqual(core.last_takeover_event, "state_file_incomplete:10/12")

    def test_B_live_motor_dead_from_start_blocks_arm(self):
        with declared(s8, self.map_fixed):
            core = self.core(dead=FIXED_IDS + (4,))
            self.ticks(core, 30)
            st = core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
            self.assertEqual((st["baseline_count"], st["baseline_missing_joints"]), (11, ["left_knee_joint"]))
            st = core.handle_command({"command": s8.COMMAND_ARM_ALL})
            self.assertFalse(core.armed)
            self.assertEqual(st["last_reject_reason"], "baseline_incomplete_for_arm:left_knee_joint")

    def test_B_all_live_motors_dead_baseline_rejected_like_before(self):
        """고정 관절의 합성 피드백만으로는 기준점이 수락되지 않는다(종전 '실측 전무' 거부 유지)."""
        with declared(s8, self.map_fixed):
            core = self.core(dead=range(1, 13))
            self.ticks(core, 30)
            st = core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
            self.assertFalse(st["accepted"])
            self.assertEqual(st["last_reject_reason"], "feedback_stale:all_baseline_missing")
        core = self.core(dead=range(1, 13))      # 대조군: 선언 없음 — 같은 사유
        self.ticks(core, 30)
        st = core.handle_command({"command": s8.COMMAND_SET_BASELINE_ALL})
        self.assertEqual(st["last_reject_reason"], "feedback_stale:all_baseline_missing")

    def test_frame_from_declared_fixed_motor_is_ignored_and_visible(self):
        with declared(s8, self.map_fixed):
            core = self.core()
            p = enc(77.0)
            ok = core._handle_can_feedback_message(
                can.Message(arbitration_id=3, data=[3, p[0], p[1], 0x7F, 0xF7, 0xFF, 30, 0], is_extended_id=False))
            self.assertFalse(ok)
            self.assertEqual(core.latest_actual_deg_by_joint[FIXED[0]], s8.STAND_ZERO_DEG_BY_JOINT[FIXED[0]])
            self.assertEqual(core.can_decode_error, "mech_fixed_frame_ignored:arb3")


class ScriptsTest(_Base):
    """~/rl_calib 스크립트(가짜 버스·가짜 시계·임시 맵). 모듈 상수의 맵·저널 경로만 임시 파일로 바꾼다."""

    def run_bpc(self, map_path, dead, journal):
        clk = Clock()
        bus = SimBus(stand_pos(), dead=dead, clock=clk)
        with patch.object(bpc, "HW_MAP", map_path), patch.object(bpc, "JOURNAL", journal), patch.object(bpc, "time", clk), \
                patch.object(can, "Bus", lambda *a, **k: bus), argv("--ref", "stand"), keep_signals(), quiet() as out:
            rc = bpc.main()
        return rc, bus, out.getvalue(), clk

    def test_A_boot_pose_check_and_frame_gate(self):
        journal = os.path.join(self.d, "journal_A.json")
        with open(journal, "w", encoding="utf-8") as f:      # 모터를 떼기 전 실행이 남긴 옛 저널(고정 모터 항목 포함)
            json.dump({"ts": 1.0, "pos_deg": {str(m): 0.0 for m in range(1, 13)}, "motor_ts": {str(m): 1.0 for m in range(1, 13)}}, f)
        rc, bus, text, clk = self.run_bpc(self.map_fixed, FIXED_IDS, journal)
        self.assertEqual(rc, 0, text)
        self.assertIn("모터 [3, 9] 기구 고정 — 제외", text)
        self.assertIn("[OK] 10모터 전수 검사", text)
        self.assertEqual(bus.sent_to(FIXED_IDS), [])
        with open(journal, encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(sorted(int(k) for k in j["motor_ts"]), [m for m in range(1, 13) if m not in FIXED_IDS])
        self.assertEqual(sorted(j["pos_deg"]), sorted(j["motor_ts"]))
        # 브리지 프레임 게이트: 선언이 있으면 통과, 같은 저널이라도 선언이 없으면 종전대로 거부
        self.assertIsNone(b.frame_check_verdict(journal, 3600.0, now=clk.t, mech_fixed_ids=FIXED_IDS))
        why = b.frame_check_verdict(journal, 3600.0, now=clk.t)
        self.assertIn("[3, 9]", why)
        self.assertNotIn("기구 고정", why)

    def test_C_boot_pose_check_undeclared_dead_motors_fail(self):
        journal = os.path.join(self.d, "journal_C.json")
        rc, bus, text, _ = self.run_bpc(self.map_none, FIXED_IDS, journal)
        self.assertEqual(rc, 1)
        self.assertIn("[결측] 모터 [3, 9] 무응답", text)
        self.assertNotIn("기구 고정", text)
        self.assertTrue(bus.sent_to(FIXED_IDS))          # 선언이 없으면 종전대로 폴링한다

    def test_B_boot_pose_check_declared_but_live_motor_dead_fails(self):
        rc, _, text, _ = self.run_bpc(self.map_fixed, FIXED_IDS + (4,), os.path.join(self.d, "journal_B.json"))
        self.assertEqual(rc, 1)
        self.assertIn("[결측] 모터 [4] 무응답", text)

    def test_D_boot_pose_check_undeclared_all_alive_output_as_before(self):
        rc, bus, text, _ = self.run_bpc(self.map_none, (), os.path.join(self.d, "journal_D.json"))
        self.assertEqual(rc, 0, text)
        self.assertIn("[OK] 12모터 전수 검사·프레임 일치 — 기립 등 다음 단계 진행 가능", text)
        self.assertNotIn("기구 고정", text)
        self.assertEqual({m for m, _ in bus.frames}, set(range(1, 13)))

    def run_ping(self, map_path, dead):
        clk = Clock()
        bus = SimBus(stand_pos(), dead=dead, clock=clk)

        def no_recover(ch):
            raise AssertionError("버스 복구(sudo ip link)가 불림")
        with patch.object(pa, "HW_MAP", map_path), patch.object(pa, "time", clk), patch.object(pa, "open_bus", lambda ch: bus), \
                patch.object(pa, "bus_state", lambda ch: "ERROR-ACTIVE"), patch.object(pa, "recover", no_recover), \
                argv("--json"), quiet() as out:
            rc = pa.main()
        return rc, bus, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_ping_all(self):
        rc, bus, js = self.run_ping(self.map_fixed, FIXED_IDS)                      # A
        self.assertEqual((rc, js["dead"], js["mech_fixed"], len(js["ok"])), (0, [], [3, 9], 10))
        self.assertEqual(bus.sent_to(FIXED_IDS), [])
        rc, _, js = self.run_ping(self.map_fixed, FIXED_IDS + (4,))                 # B
        self.assertEqual((rc, js["dead"]), (1, [4]))
        rc, bus, js = self.run_ping(self.map_none, FIXED_IDS)                       # C
        self.assertEqual((rc, js["dead"]), (1, [3, 9]))
        self.assertNotIn("mech_fixed", js)
        rc, bus, js = self.run_ping(self.map_none, ())                              # D
        self.assertEqual((rc, len(js["ok"])), (0, 12))
        self.assertNotIn("mech_fixed", js)

    def run_release(self, map_path, dead):
        clk = Clock()
        bus = SimBus(stand_pos(), dead=dead, clock=clk)
        ra.SIG.clear()
        ra.__dict__.pop("VERDICT_CODE", None)
        with patch.object(ra, "HW_MAP", map_path), patch.object(ra, "time", clk), patch.object(ra, "gate_or_die", lambda f=None: []), \
                patch.object(ra, "IDS", list(range(1, 13))), patch.object(can, "Bus", lambda *a, **k: bus), \
                argv(), keep_signals(), quiet() as out:
            ra.main()
            code = ra.__dict__.get("VERDICT_CODE")
        return code, bus, out.getvalue()

    def test_release_all(self):
        code, bus, text = self.run_release(self.map_fixed, FIXED_IDS)               # A
        self.assertEqual(code, 0, text)
        self.assertIn("모터 [3, 9] 기구 고정 — 제외", text)
        self.assertIn("RELEASE_VERDICT=OK seen=10 fail=0", text)
        self.assertEqual(bus.sent_to(FIXED_IDS), [])
        code, bus, text = self.run_release(self.map_none, ())                       # D
        self.assertIn("[release_all] 12모터 소프트 릴리즈+MIT 해제 완료 — 무토크", text)
        self.assertIn("RELEASE_VERDICT=OK seen=12 fail=0", text)
        self.assertEqual({m for m, _ in bus.frames}, set(range(1, 13)))
        # 맵을 못 읽어도 회수는 막지 않는다 — 12 모터 전부 대상
        code, bus, text = self.run_release(os.path.join(self.d, "no_such_map.json"), ())
        self.assertIn("12모터 전부 대상으로 진행", text)
        self.assertEqual({m for m, _ in bus.frames}, set(range(1, 13)))

    def read_pose(self, map_path, dead):
        clk = Clock()
        bus = SimBus(stand_pos(), dead=dead, clock=clk)
        with patch.object(bpc, "HW_MAP", map_path), patch.object(bpc, "time", clk), patch.object(za, "time", clk), \
                patch.object(can.interface, "Bus", lambda *a, **k: bus), quiet() as out:
            pose = za.read_pose()
        return pose, bus, out.getvalue()

    def test_zero_anchor(self):
        pose, bus, text = self.read_pose(self.map_fixed, FIXED_IDS)                 # A
        self.assertEqual(sorted(pose), [m for m in range(1, 13) if m not in FIXED_IDS])
        self.assertIn("모터 [3, 9] 기구 고정 — 제외", text)
        self.assertEqual(bus.sent_to(FIXED_IDS), [])
        rep = za.analyze(pose)
        self.assertEqual((rep["n_bad"], rep["can_recapture"]), (0, True))
        self.assertNotIn(FIXED[0], [r["joint"] for r in rep["rows"]])              # 고정 관절 영점은 쓰기 대상이 아니다
        pose, _, _ = self.read_pose(self.map_fixed, FIXED_IDS + (4,))              # B
        rep = za.analyze(pose)
        self.assertEqual((rep["n_bad"], rep["can_recapture"], rep["can_multiple"]), (1, False, False))
        pose, _, text = self.read_pose(self.map_none, FIXED_IDS)                   # C
        rep = za.analyze(pose)
        self.assertEqual((len(pose), rep["n_bad"], rep["can_recapture"]), (12, 2, False))
        self.assertNotIn("기구 고정", text)
        with quiet():
            self.assertEqual(za.selftest(), 0)


class HangStandRig:
    """hang_stand.main() 을 가짜 버스·가짜 시계로 돌린다. /tmp 바로 아래 파일 접근은 전부 임시 폴더로 돌린다."""

    def __init__(self, directory, map_path, dead=()):
        self.dir = os.path.join(directory, f"hs_{id(self)}")
        os.makedirs(self.dir)
        self.map_path = map_path
        self.clk = Clock()
        self.bus = SimBus(stand_pos(), dead=dead, clock=self.clk)
        self.events = []           # [(경과 초, 함수)] — 한 번씩
        self.marker_from = None    # 이 경과 초부터 stage8 대기 표지를 계속 신선하게

    def path(self, p):
        p = str(p)
        return os.path.join(self.dir, os.path.basename(p)) if os.path.dirname(p) == "/tmp" else p

    def touch(self, p, text=""):
        with open(self.path(p), "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, p):
        with open(self.path(p), encoding="utf-8") as f:
            return json.load(f)

    def run(self):
        rig = self

        class _Path:
            def exists(self, p):
                return os.path.exists(rig.path(p))

            def getmtime(self, p):
                return os.path.getmtime(rig.path(p))

            def __getattr__(self, name):
                return getattr(os.path, name)

        class _Os:
            path = _Path()

            def remove(self, p):
                return os.remove(rig.path(p))

            def replace(self, a, c):
                return os.replace(rig.path(a), rig.path(c))

            def __getattr__(self, name):
                return getattr(os, name)

        t_start = self.clk.t
        pending = sorted(self.events, key=lambda e: e[0])

        def hook():
            el = rig.clk.t - t_start
            while pending and el >= pending[0][0]:
                pending.pop(0)[1]()
            if rig.marker_from is not None and el >= rig.marker_from:
                rig.touch("/tmp/stage8_takeover_waiting")
                os.utime(rig.path("/tmp/stage8_takeover_waiting"), (rig.clk.t, rig.clk.t))
            if el > 120.0:
                raise AssertionError("hang_stand 가 120 s(가짜 시계) 안에 끝나지 않음")
        self.clk.hook = hook
        hs.SIG_STOP.clear()
        hs.FB_TAU.clear()
        hs.FB_VEL.clear()
        limits = (str(REPO / "config" / "joint_limits_12dof.json"),)
        with patch.object(hs, "HW_MAP", self.map_path), patch.object(hs, "JOINT_LIMIT_FILES", limits), \
                patch.object(hs, "IDS", list(range(1, 13))), patch.object(hs, "time", self.clk), \
                patch.object(hs, "os", _Os()), patch.object(hs, "open", lambda p, *a, **k: open(rig.path(p), *a, **k), create=True), \
                patch.object(can, "Bus", lambda *a, **k: self.bus), argv(), keep_signals(), quiet() as out:
            try:
                rc = hs.main()
            except SystemExit as e:
                rc = e.code
        self.text = out.getvalue()
        return rc


class HangStandTest(_Base):
    def test_A_prep_level_handoff_then_stage8_and_bridge(self):
        """prep → level → 핸드오프(hang_stand) → stage8 인수 홀드·기준점·ARM → 브리지 stand: 고정 선언 + 모터 3·9 무응답."""
        rig = HangStandRig(self.d, self.map_fixed, dead=FIXED_IDS)
        rig.events = [
            (6.5, lambda: rig.touch(hs.GAIN_FILE)),
            (15.2, lambda: rig.touch(hs.JOINT_TRIM_FILE, json.dumps({"left_hip_f_joint": 5.0, FIXED[0]: 3.0, FIXED[1]: -3.0}))),
            (16.0, lambda: rig.touch(hs.GROUND_FILE)),
            (16.8, lambda: rig.touch(hs.HANDOFF_FILE)),
        ]
        rig.marker_from = 16.6
        rc = rig.run()
        self.assertEqual(rc, 3, rig.text[-2000:])
        self.assertIn("모터 [3, 9] 기구 고정 — 제외", rig.text)
        self.assertEqual(rig.bus.sent_to(FIXED_IDS), [])
        self.assertEqual({m for m, _ in rig.bus.frames}, set(range(1, 13)) - set(FIXED_IDS))
        st = rig.read(hs.STATUS_FILE)
        self.assertEqual(st["mech_fixed"], list(FIXED))
        self.assertEqual(len(st["joint_trim"]), 12)
        for j in FIXED:      # 조정 요청(±3°)은 따르지 않는다 — 0 으로 보고
            self.assertEqual(st["joint_trim"][j], {"req": 0.0, "cur": 0.0, "frozen": False, "mech_fixed": True})
            self.assertEqual((st["joint_pos_deg"][j], st["joint_cmd_deg"][j]), (0.0, 0.0))
            self.assertNotIn(j, st["err_deg"])
        self.assertAlmostEqual(st["joint_trim"]["left_hip_f_joint"]["cur"], 5.0, places=6)
        state = rig.read("/tmp/hang_handoff_state.json")
        self.assertEqual(sorted(int(k) for k in state["motors"]), [m for m in range(1, 13) if m not in FIXED_IDS])
        # 그 상태 파일 그대로 stage8 → 브리지
        rec = closed_loop(s8, b, self.map_fixed, declare=True, bus=rig.bus, takeover_file=rig.path("/tmp/hang_handoff_state.json"),
                          t0=float(state["ts"]) + 0.05)
        self.assertTrue(rec["holding_at_start"], rec["core"].last_takeover_event)
        self.assertEqual((rec["hold_status"]["hold_joint_count"], rec["hold_status"]["hold_mech_fixed_count"]), (12, 2))
        self.assertEqual((rec["baseline_status"]["baseline_count"], rec["baseline_status"]["baseline_source"]), (12, "takeover_hold"))
        self.assertTrue(rec["armed_after_arm"], rec["arm_reject"])
        self.check_policy_running(rec)

    def check_policy_running(self, rec):
        self.assertIsNone(rec["stop_tick"], rec["stop_reason"])
        self.assertTrue(rec["active_at_end"])
        self.assertGreater(len(rec["targets"]), 150)
        st = rec["status_at_end"]
        self.assertTrue(st["armed"])
        self.assertIsNone(st["last_reject_reason"])
        self.assertEqual(rec["bus"].sent_to(FIXED_IDS), [])
        self.assertGreater(st["mech_fixed_target_drops"], 150)
        ticks = [d for d in rec["dbg"] if d.get("raw_action") is not None]
        self.assertGreater(len(ticks), 150)
        i_fix = [b.JOINT_ORDER.index(j) for j in FIXED]
        for d in ticks:
            self.assertEqual(d["mech_fixed"], [j for j in b.JOINT_ORDER if j in FIXED])
            for i in i_fix:      # 정책 출력은 그대로(입력 쪽은 건드리지 않는다), 송출 목표만 관절 0°
                self.assertAlmostEqual(d["raw_action"][i], POLICY_ACTION, places=6)
        for tg in rec["targets"]:
            for j in FIXED:      # 상대 목표 = stand_zero − 기준점(stand_zero) = 0 (블렌딩 부동소수 오차만)
                self.assertLess(abs(tg[j]), 1e-9)
        last = rec["targets"][-1]
        self.assertTrue(all(abs(last[j]) > 1.0 for j in ("left_knee_joint", "right_knee_joint")), last)   # 산 관절은 정책 자세로 움직였다

    def test_C_undeclared_dead_motors_abort_at_start(self):
        rig = HangStandRig(self.d, self.map_none, dead=FIXED_IDS)
        self.assertEqual(rig.run(), 1)
        self.assertIn("[FAIL] 모터 3 시작각 읽기 실패 — 중단", rig.text)
        self.assertNotIn("기구 고정", rig.text)

    def test_B_live_motor_feedback_loss_aborts(self):
        """선언이 있어도 산 모터(4)의 피드백 두절은 종전대로 즉시 릴리즈 — 선언 없는 대조군과 같은 사유·같은 지연(±1 틱).
        (시작 구간의 모터별 ENTER 간격이 10 개 vs 12 개라 루프 위상이 10 ms 다르다 → 지연은 틱 단위로 비교한다.)"""
        got = []
        for map_path, dead in ((self.map_fixed, FIXED_IDS), (self.map_none, ())):
            rig = HangStandRig(self.d, map_path, dead=dead)
            killed = []

            def kill(r=rig, k=killed):
                r.bus.dead.add(4)
                k.append(r.clk.t)
            rig.events = [(3.0, kill)]
            rc = rig.run()
            line = [ln for ln in rig.text.splitlines() if ln.startswith("[ABORT]")]
            got.append((rc, line, rig.clk.t - killed[0]))
        for rc, line, delay in got:
            self.assertEqual(rc, 2)
            self.assertEqual(line, ["[ABORT] 모터4 피드백 두절 — 즉시 릴리즈 (줄이 받습니다)"])
            self.assertGreater(delay, hs.FB_TIMEOUT_S)
            self.assertLess(delay, hs.FB_TIMEOUT_S + 3 * hs.TICK)
        self.assertLess(abs(got[0][2] - got[1][2]), hs.TICK + 1e-6)

    def test_trim_channel_motor_declared_fixed_is_refused(self):
        with self.assertRaises(SystemExit):
            hs.live_ids(list(range(1, 13)), {4})
        self.assertEqual(hs.live_ids(list(range(1, 13)), set(FIXED_IDS)), [m for m in range(1, 13) if m not in FIXED_IDS])
        with quiet():
            self.assertEqual(hs.selftest(), 0)


class ClosedLoopTest(_Base):
    """stage8 ↔ 브리지 폐루프(가짜 버스·가짜 ROS). 읽기 폴링 기동 → 기준점 → ARM → stand."""

    def test_A_declared_fixed_dead_motors_policy_runs(self):
        rec = closed_loop(s8, b, self.map_fixed, declare=True, dead=FIXED_IDS)
        self.assertEqual(rec["baseline_status"]["baseline_count"], 12)
        self.assertTrue(rec["armed_after_arm"], rec["arm_reject"])
        HangStandTest.check_policy_running(self, rec)

    def test_B_live_joint_stale_stops_like_control(self):
        """산 관절(왼 무릎, 모터 4)이 stand 100 틱 뒤 응답을 멈춘다. 선언 있음(3·9 무응답) vs 선언 없음(12 모터) 대조군."""
        kw = dict(kill=(100, 4), stand_ticks=400, after_stop_s=s8.STALE_DISARM_SEC + 3.0)
        fix = closed_loop(s8, b, self.map_fixed, declare=True, dead=FIXED_IDS, **kw)
        ctl = closed_loop(s8, b, self.map_none, declare=False, **kw)
        for rec in (fix, ctl):
            self.assertIsNotNone(rec["stop_tick"])
            self.assertIn("sensor_stale", rec["stop_reason"])
            self.assertFalse(rec["active_at_end"])
            self.assertEqual(rec["stale_warn"], ["joint_feedback_stale:left_knee_joint"])
            self.assertIn("feedback_gap_auto_disarm", rec["disarm_warnings"])
            self.assertFalse(rec["core"].armed)
            self.assertEqual(rec["core"].baseline_deg_by_joint, {})
        # 가드가 느슨해지지 않았다: 정지·동결 경고·자동 무장 해제가 대조군과 같은 틱
        self.assertEqual((fix["stop_tick"], fix["stop_reason"]), (ctl["stop_tick"], ctl["stop_reason"]))
        self.assertLessEqual((fix["stop_tick"] - 100) * 0.02, b.STATUS_FEEDBACK_FRESH_SEC + b.SENSOR_STALE_SEC + 0.1)
        self.assertEqual(fix["stale_warn_tick"], ctl["stale_warn_tick"])
        self.assertEqual(fix["disarm_tick"], ctl["disarm_tick"])
        self.assertEqual(fix["bus"].sent_to(FIXED_IDS), [])

    def test_C_undeclared_dead_motors_rejected(self):
        rec = closed_loop(s8, b, self.map_none, declare=False, dead=FIXED_IDS, stand_ticks=300)
        self.assertEqual(rec["baseline_status"]["baseline_missing_joints"], list(FIXED))
        self.assertFalse(rec["armed_after_arm"])
        self.assertEqual(rec["arm_reject"], "baseline_incomplete_for_arm:left_hip_r_joint,right_hip_r_joint")
        # 정책은 한 틱도 돌지 않았다(브리지가 보낸 것은 인계 워밍업의 현 자세 유지 명령뿐이고 stage8 은 비무장이라 받지 않는다)
        self.assertFalse(any(d.get("raw_action") is not None for d in rec["dbg"]))
        self.assertFalse(rec["status_at_end"]["armed"])
        self.assertFalse(rec["active_at_end"])
        self.assertIn("sensor_stale", rec["stop_reason"])
        self.assertTrue(all(d in (ENTER,) for _, d in rec["bus"].frames))     # 버스로 나간 것은 읽기 폴링뿐(명령 프레임 0)
        self.assertNotIn("mech_fixed", json.dumps(rec["dbg"]))

    def same_as_base(self, **kw):
        s8_base, b_base = base_modules()
        if s8_base is None or b_base is None:
            self.skipTest(f"git 에서 {BASE_REV} 원본을 꺼낼 수 없음")
        try:
            new = closed_loop(s8, b, self.map_none, declare=False, **kw)
            old = closed_loop(s8_base, b_base, self.map_none, declare=False, **kw)
        finally:
            sys.modules.pop("stage8_base_rev", None)
        self.assertEqual(len(new["bus"].frames), len(old["bus"].frames))
        self.assertEqual(new["bus"].frames, old["bus"].frames)                 # CAN 프레임 열(모터 ID·8 바이트) 바이트 동일
        self.assertEqual(new["pub_log"], old["pub_log"])                       # 브리지 발행(명령·디버그) 순서·비트 동일
        for k in ("arm_reject", "stop_tick", "stop_reason", "active_at_end"):
            self.assertEqual(new[k], old[k], k)
        common = set(new["status_at_end"]) & set(old["status_at_end"]) - {"tick_timing"}   # tick_timing 은 실제 monotonic 시계
        for k in sorted(common):
            self.assertEqual(new["status_at_end"][k], old["status_at_end"][k], k)
        self.assertEqual(set(new["status_at_end"]) - set(old["status_at_end"]), {"mech_fixed_joints", "mech_fixed_target_drops"})
        return new

    def test_D_undeclared_frames_and_publications_identical_to_base(self):
        new = self.same_as_base(stand_ticks=300)
        self.assertGreater(len(new["bus"].frames), 3000)
        self.assertGreater(len(new["targets"]), 250)
        self.assertIsNone(new["stop_tick"])

    def test_D_undeclared_live_joint_stale_identical_to_base(self):
        new = self.same_as_base(kill=(100, 4), stand_ticks=400)
        self.assertIn("sensor_stale", new["stop_reason"])

    def test_C_undeclared_dead_motors_identical_to_base(self):
        new = self.same_as_base(dead=FIXED_IDS, stand_ticks=300)
        self.assertFalse(new["armed_after_arm"])


class MapCopyMismatchTest(_Base):
    def test_E_selftest_item(self):
        with quiet():
            self.assertIn("사본 불일치 거부", b._selftest_mech_fixed_map())

    def test_E_live_start_refused_on_mismatch(self):
        """저장소 사본과 ~/rl_calib 사본의 mech_fixed 가 다르면 브리지 --live 기동이 맵 로드에서 멈춘다(노드 생성 전)."""
        created = []
        H._patch(b, H.FakeClock())
        for pair in ((self.map_fixed, self.map_none), (self.map_none, self.map_fixed)):
            with patch.object(b, "HARDWARE_MAP_PATHS", pair), self.assertRaises(ValueError) as cm, quiet():
                H._run_main(b, ["--live", "--force-frame", "--force-gains", "--checkpoint", CKPT], created.append)
            self.assertIn("사본 불일치", str(cm.exception))
            self.assertIn("mech_fixed", str(cm.exception))
        self.assertEqual(created, [])
        same = (self.map_fixed, write_map(self.d, "map_fixed_copy.json", FIXED))                   # 양쪽이 같으면 뜬다
        with patch.object(b, "HARDWARE_MAP_PATHS", same), quiet():
            self.assertEqual(H._run_main(b, ["--live", "--force-frame", "--force-gains", "--checkpoint", CKPT], created.append), 0)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].mech_fixed, [j for j in b.JOINT_ORDER if j in FIXED])


if __name__ == "__main__":
    unittest.main()
