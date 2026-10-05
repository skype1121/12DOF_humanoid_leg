"""1005 단순화 기동(정책 ON → 지그 해제) — 브리지 **실제 노드 코드**를 가짜 ROS·모의 센서로 구동하는 폐루프 아닌 계약 시험.

시나리오(모의, 실물 아님): 공중 매달림(양발 비접촉, 몸통 피치 +6°) → stand(touchdown_start) → 3 s 대기 →
지그 해제 = 착지(접촉 채터 + |ω| 0.8 rad/s 충격 0.15 s, 몸통 피치 +1° 로 정착) → 정책 시작.
플랜트는 1차 지연 추종(관절)과 대본 IMU(정책에 반응하지 않음)뿐이다 — 정책 성능·균형 증명이 아니라
배포 경로의 순서·연속성·초기화 계약을 확인한다.

확인 항목: 착지 전 정책 미구동(러너 틱 0·명령 = 래치 홀드) · 홀드→정책 전환 목표 연속(틱당 ≤ 4°) · 비유한 없음 ·
정책 첫 틱 = 학습 리셋 계약(이력 백필·직전 액션 0) · C5 감지기 기준이 착지 뒤 자세로 캡처·시작 직후 회복 오탐 0 ·
대조군(종전 공중 정책 hang_arm, 같은 시나리오): 공중에서 정책 구동 + 착지 뒤 회복 모드 오탐 → 이번 게이트의 존재 이유.
종전 지상 stand 경로(touchdown_start 없음)는 즉시 정책·start_gate 필드 없음(스키마 불변).
"""
import functools
import importlib.util
import json
import math
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "stand_c5kr_g21a_g24ra"
DT = 0.02
SLEW_DEG = 4.0


def _load():
    spec = importlib.util.spec_from_file_location("rlbn_startup", os.path.join(REPO, "jetson/scripts/rl_bridge_node.py"))
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["rl_bridge_node.py"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


M = _load()
if not os.path.isdir(M.CHECKPOINTS[CKPT]):
    pytest.skip(f"{CKPT} 정책 파일 없음", allow_module_level=True)


def quat_rp(roll_deg, pitch_deg):
    """R = Ry(pitch)·Rx(roll) (요 0) → (w,x,y,z)."""
    r, p = math.radians(roll_deg) / 2, math.radians(pitch_deg) / 2
    return (math.cos(p) * math.cos(r), math.cos(p) * math.sin(r), math.sin(p) * math.cos(r), -math.sin(p) * math.sin(r))


class Sim:
    """가짜 rclpy + stage8 + 센서. M.main() 안에서 만들어진 노드를 잡아 틱을 직접 돌린다."""

    def __init__(self, monkeypatch, tmp_path, cmd, scenario, hang_rp=(1.0, 6.0), stand_rp=(0.5, 1.0),
                 t_touch=3.5, t_end=8.0, start_on_ground=False, hold_off=None):
        self.mp, self.cmd, self.scenario = monkeypatch, cmd, scenario
        self.hang_rp, self.stand_rp, self.t_touch, self.t_end = hang_rp, stand_rp, t_touch, t_end
        self.start_on_ground = start_on_ground
        self.hold_off = hold_off
        self.clock = [1_000_000.0]
        self.pubs = []           # (tick, topic, obj)
        self.logs = []
        self.node = None
        self.tick_i = 0
        monkeypatch.setenv("HOME", str(tmp_path))                 # 블랙박스 = tmp/logs (실제 ~/logs 오염 금지)
        real_time = M.time
        clock = self.clock

        class FakeTime:
            def __getattr__(self, k):
                return getattr(real_time, k)

            @staticmethod
            def time():
                return clock[0]
        monkeypatch.setattr(M, "time", FakeTime())
        self._install_ros()

    # ---- 가짜 ROS ----
    def _install_ros(self):
        sim = self

        class String:
            def __init__(self, data=""):
                self.data = data

        class Pub:
            def __init__(self, topic):
                self.topic = topic

            def publish(self, msg):
                sim.pubs.append((sim.tick_i, self.topic, json.loads(msg.data)))

        class Log:
            def _w(self, level, msg):
                sim.logs.append((level, str(msg)))

            def info(self, m): self._w("info", m)
            def warn(self, m): self._w("warn", m)
            def warning(self, m): self._w("warn", m)
            def error(self, m): self._w("error", m)

        class Node:
            def __init__(self, name):
                self._subs = {}
                sim.node = self

            def create_subscription(self, typ, topic, cb, qos):
                self._subs[topic] = cb

            def create_publisher(self, typ, topic, qos):
                return Pub(topic)

            def create_timer(self, period, cb):
                return None

            def get_logger(self):
                return Log()

            def destroy_node(self):
                pass

        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda *a, **k: None
        rclpy.shutdown = lambda *a, **k: None
        rclpy.spin = lambda node: sim.run(node)
        node_mod = types.ModuleType("rclpy.node")
        node_mod.Node = Node
        std = types.ModuleType("std_msgs")
        std_msg = types.ModuleType("std_msgs.msg")
        std_msg.String = String
        self.String = String
        for k, v in (("rclpy", rclpy), ("rclpy.node", node_mod), ("std_msgs", std), ("std_msgs.msg", std_msg)):
            self.mp.setitem(sys.modules, k, v)

    # ---- 모의 플랜트 ----
    def run(self, node):
        fr = node.frames
        self.fr = fr
        pose = M.ckpt_default_pose(CKPT)
        # 수평 홀드 자세 = 정책 기본자세 + 실물 같은 작은 편차(무릎 +1°, 발목 −1.5°)
        off = self.hold_off or {"left_knee_joint": 1.0, "right_knee_joint": -1.0, "left_ankle_f_joint": -1.5, "right_ankle_f_joint": 1.5}
        q_deg = {j: fr.joint_rad_to_motor_deg(j, pose[i] + math.radians(off.get(j, 0.0))) for i, j in enumerate(M.JOINT_ORDER)}
        self.baseline = dict(q_deg)                          # SET_BASELINE_FROM_CURRENT_ALL (공중 인수 홀드)
        self.cmd_rel = {j: 0.0 for j in M.JOINT_ORDER}        # stage8 스트리밍 명령(상대) — 인수 홀드
        self.want_rel = dict(self.cmd_rel)
        self.q = dict(q_deg)
        self.hold_abs0 = {j: self.baseline[j] + self.cmd_rel[j] for j in M.JOINT_ORDER}
        n = int(round(self.t_end / DT))
        t_cmd = 0.5
        for i in range(n):
            self.tick_i = i
            t = i * DT
            self.clock[0] += DT
            self._sensors(node, t)
            if abs(t - t_cmd) < DT / 2:
                node._on_cmd(self.String(json.dumps(self.cmd)))
            node._tick()
            self._stage8(i)
            self.scenario_hook(node, t, i)

    def scenario_hook(self, node, t, i):
        if self.scenario:
            self.scenario(self, node, t, i)

    def contact(self, t):
        if self.start_on_ground:
            return (True, True)
        dt = t - self.t_touch
        if dt < 0:
            return (False, False)
        chatter = [(True, False)] * 2 + [(True, True)] * 3 + [(False, True)] + [(True, False)] * 2
        k = int(round(dt / DT))
        return chatter[k] if k < len(chatter) else (True, True)

    def imu(self, t):
        if self.start_on_ground:
            return self.stand_rp, (0.0, 0.0, 0.0)
        dt = t - self.t_touch
        if dt < 0:
            return self.hang_rp, (0.0, 0.0, 0.0)
        a = min(1.0, dt / 0.2)
        rp = tuple(h + (s - h) * a for h, s in zip(self.hang_rp, self.stand_rp))
        w = (0.0, -0.8, 0.0) if dt < 0.15 else (0.0, 0.0, 0.0)          # 착지 충격 피치 각속도
        return rp, w

    def _sensors(self, node, t):
        ts = self.clock[0]
        rp, w = self.imu(t)
        node._on_imu(self.String(json.dumps({"timestamp": ts, "gyro_rad_s": list(w), "quat_wxyz": list(quat_rp(*rp))})))
        cl, cr = self.contact(t)
        node._on_foot(self.String(json.dumps({"timestamp": ts, "left_n": 0.0, "right_n": 0.0,
                                               "left_contact": cl, "right_contact": cr})))
        joints = {j: {"baseline_deg": self.baseline[j], "commanded_relative_deg": self.cmd_rel[j],
                      "feedback_age_sec": 0.02, "latest_actual_deg": self.q[j]} for j in M.JOINT_ORDER}
        node._on_status(self.String(json.dumps({"armed": True, "joints": joints})))

    def _stage8(self, i):
        for tick, topic, obj in self.pubs:
            if tick != i or topic != M.COMMAND_TOPIC:
                continue
            if obj.get("command") == "SET_JOINT_TARGETS":
                self.want_rel.update({j: float(v) for j, v in obj["targets"].items()})
            elif obj.get("command") == "STOP_ALL":
                self.want_rel = dict(self.cmd_rel)                       # 동결(토크 유지)
        for j in M.JOINT_ORDER:                                         # stage8 4°/틱 슬루 → 모터 1차 추종
            d = self.want_rel[j] - self.cmd_rel[j]
            self.cmd_rel[j] += max(-SLEW_DEG, min(SLEW_DEG, d))
            self.q[j] += 0.3 * (self.baseline[j] + self.cmd_rel[j] - self.q[j])

    # ---- 결과 정리 ----
    def dbg(self):
        return [(i, o) for i, tp, o in self.pubs if tp == M.BRIDGE_DEBUG_TOPIC and "event" not in o]

    def targets(self):
        """틱별 브리지 발행 절대 목표(모터 deg) — SET_JOINT_TARGETS 만."""
        out = []
        for i, tp, o in self.pubs:
            if tp == M.COMMAND_TOPIC and o.get("command") == "SET_JOINT_TARGETS":
                out.append((i, {j: self.baseline[j] + float(v) for j, v in o["targets"].items()}))
        return out

    def stops(self):
        return [o for _, tp, o in self.pubs if tp == M.COMMAND_TOPIC and o.get("command") == "STOP_ALL"]


def run_sim(monkeypatch, tmp_path, cmd, **kw):
    sim = Sim(monkeypatch, tmp_path, cmd, kw.pop("scenario", None), **kw)
    argv = sys.argv
    sys.argv = ["rl_bridge_node.py", "--live", "--force-frame", "--checkpoint", CKPT]
    try:
        assert M.main() == 0
    finally:
        sys.argv = argv
    return sim


def first_policy_tick(sim):
    for i, o in sim.dbg():
        if o.get("raw_action") is not None:
            return i, o
    return None, None


def max_step(sim):
    tg = sim.targets()
    worst, at = 0.0, None
    for (i0, a), (i1, b) in zip(tg, tg[1:]):
        for j in M.JOINT_ORDER:
            d = abs(b[j] - a[j])
            if d > worst:
                worst, at = d, (i1, j)
    return worst, at, tg


def test_touchdown_start_holds_in_air_then_starts_policy_continuously(monkeypatch, tmp_path):
    seen = {}

    def probe(sim, node, t, i):
        r = node.core.runner
        if t < sim.t_touch:
            seen.setdefault("air_runner_ticks", 0)
            seen["air_runner_ticks"] = max(seen["air_runner_ticks"], r.core._tick)
        if r.core._tick == 1 and "first_obs" not in seen:
            seen["first_obs"] = r.core.last_obs.copy()
            seen["first_terms"] = [(n, d, r.core._hist[n]._buf.shape[0]) for n, d in r.core.terms]
            seen["hang_state"] = node.hang_guard.state
            seen["start_t"] = t
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand", "touchdown_start": True}, scenario=probe)
    # (1) 공중: 정책 미구동, 명령 = 래치 홀드 그대로
    assert seen["air_runner_ticks"] == 0, "착지 전 정책이 계산됐다"
    air = [tg for i, tg in sim.targets() if i * DT < sim.t_touch]
    assert air, "공중 대기 중 홀드 명령이 없다"
    for tg in air:
        assert max(abs(tg[j] - sim.hold_abs0[j]) for j in M.JOINT_ORDER) < 1e-9, "공중 대기 명령이 홀드와 다르다"
    gates = [o.get("start_gate", {}).get("state") for i, o in sim.dbg() if i * DT < sim.t_touch and "start_gate" in o]
    assert gates and all(s != "POLICY" for s in gates)
    # (2) 시작 시점: 채터 뒤 연속 양발 0.3 s + 정착 뒤 (충격 0.15 s 동안은 시작 안 함)
    steady = sim.t_touch + 8 * DT                       # 채터 8틱 뒤 연속 양발
    assert seen["start_t"] >= steady + M.TD_CONFIRM_S - 1e-6, (seen["start_t"], steady)
    assert seen["start_t"] <= steady + M.TD_CONFIRM_S + M.TD_SETTLE_MAX_S + 0.05
    assert seen["hang_state"] == M.HangGuard.CONTACT_ACQUIRED            # hang_mode=False(종전 지상 리프트 가드)
    # (3) 학습 리셋 계약: 첫 관측의 이력 = 백필(모든 프레임 동일), 직전 액션 0
    obs, off = seen["first_obs"], 0
    for name, dim, L in seen["first_terms"]:
        block = obs[off:off + dim * L].reshape(L, dim)
        assert (block == block[-1]).all(), f"{name} 이력이 백필이 아님"
        if name == "actions":
            assert not block.any(), "첫 틱 직전 액션 ≠ 0"
        off += dim * L
    # (4) 연속성: 홀드 → 정책 전환 포함 전 구간 틱당 ≤ 4°, 첫 정책 틱 = 홀드(α=0)
    worst, at, tg = max_step(sim)
    assert worst <= SLEW_DEG + 1e-6, f"목표 점프 {worst:.3f}° @ {at}"
    i0, o0 = first_policy_tick(sim)
    first = dict(tg)[i0]
    assert max(abs(first[j] - sim.hold_abs0[j]) for j in M.JOINT_ORDER) < 1e-6, "첫 정책 틱이 홀드와 다르다"
    assert o0["blend_alpha"] == 0.0
    # (5) 비유한 없음
    for _, o in sim.dbg():
        for k in ("targets_deg_motor",):
            assert all(math.isfinite(v) for v in (o[k] or {}).values())
        if o.get("obs") is not None:
            assert all(math.isfinite(v) for v in o["obs"])
    # (6) C5: 시작 직후 회복 오탐 0, 기준 자세 = 착지 뒤 자세(공중 자세 아님)
    c5 = [o["c5"] for _, o in sim.dbg() if o.get("c5")]
    assert c5 and c5[-1]["n_on"] == 0, f"회복 모드 오탐 {c5[-1]['n_on']}회"
    ref = c5[-1]["ref_deg"]
    det = M.C5_DETECTOR_KW
    g_stand = M.correct_gravity(M.projected_gravity(quat_rp(*sim.stand_rp)))
    g_hang = M.correct_gravity(M.projected_gravity(quat_rp(*sim.hang_rp)))
    from rl_walking.deploy.c5_switch import SwitchDetector
    rp_s, rp_h = SwitchDetector.roll_pitch_deg(g_stand), SwitchDetector.roll_pitch_deg(g_hang)
    d_s = math.hypot(ref[0] - rp_s[0], ref[1] - rp_s[1])
    d_h = math.hypot(ref[0] - rp_h[0], ref[1] - rp_h[1])
    assert d_s < 0.5 and d_h > det["on_tilt_deg"], (ref, rp_s, rp_h)
    assert not sim.stops(), f"예상 밖 STOP_ALL {sim.stops()}"
    print(f"[touchdown_start] 시작 t={seen['start_t']:.2f}s(착지 {sim.t_touch}s) · 최대 틱 변화 {worst:.3f}° · "
          f"C5 켜짐 {c5[-1]['n_on']} · 기준 {ref} (착지 자세 {[round(v, 2) for v in rp_s]}, 공중 {[round(v, 2) for v in rp_h]})")


def test_control_air_policy_start_runs_in_air_and_false_triggers_recovery(monkeypatch, tmp_path):
    """대조군 — 같은 시나리오에서 종전 공중 정책 경로(hang_arm)만 바꾼다: 정책이 공중에서 구동되고, C5 기준이
    공중 자세로 잡혀 착지 직후 회복 모드가 켜진다(오탐). touchdown_start 가 이것을 없애는지 확인하는 비교 기준."""
    seen = {"air_ticks": 0}

    def probe(sim, node, t, i):
        if t < sim.t_touch:
            seen["air_ticks"] = node.core.runner.core._tick
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand", "hang_arm": True}, scenario=probe)
    c5 = [o["c5"] for _, o in sim.dbg() if o.get("c5")]
    assert seen["air_ticks"] > 100, "대조군: 공중에서 정책이 돌지 않았다"
    assert c5[-1]["n_on"] >= 1, "대조군: 착지 뒤 회복 모드 오탐이 재현되지 않았다"
    assert all("start_gate" not in o for _, o in sim.dbg())
    print(f"[control hang_arm] 공중 정책 틱 {seen['air_ticks']} · C5 켜짐 {c5[-1]['n_on']} · 기준 {c5[-1]['ref_deg']}")


def test_old_ground_stand_path_unchanged(monkeypatch, tmp_path):
    """종전 지상 stand(touchdown_start 없음): 워밍업 뒤 즉시 정책, start_gate 필드 없음, 첫 틱 = 홀드(블렌딩 α=0), 틱당 ≤ 4°."""
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand"}, start_on_ground=True, t_end=4.0)
    d = sim.dbg()
    assert d and all("start_gate" not in o for _, o in d)
    i0, o0 = first_policy_tick(sim)
    assert i0 is not None and i0 * DT <= 0.5 + 2 * DT, f"정책 시작 지연 {i0 * DT:.2f}s"
    worst, at, tg = max_step(sim)
    assert worst <= SLEW_DEG + 1e-6, (worst, at)
    first = dict(tg)[i0]
    assert max(abs(first[j] - sim.hold_abs0[j]) for j in M.JOINT_ORDER) < 1e-6
    c5 = [o["c5"] for _, o in d if o.get("c5")]
    assert c5[-1]["n_on"] == 0
    assert not sim.stops()


def test_touchdown_wait_timeout_freezes_hold_without_policy(monkeypatch, tmp_path):
    """착지 없이 대기 상한 → STOP_ALL 1회(동결 = 같은 홀드) + 정책 비활성 — 정책은 한 틱도 돌지 않는다. guard_mode=data 여도 정지."""
    monkeypatch.setattr(M, "TouchdownGate", functools.partial(M.TouchdownGate, max_wait_s=2.0))
    ticks = {"n": 0}

    def probe(sim, node, t, i):
        ticks["n"] = max(ticks["n"], node.core.runner.core._tick)
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand", "touchdown_start": True}, scenario=probe,
                  t_touch=99.0, t_end=4.0)
    assert ticks["n"] == 0
    assert len(sim.stops()) == 1
    assert sim.node.core.active is False
    assert any("touchdown_wait_timeout" in m for _, m in sim.logs)
    worst, at, _ = max_step(sim)
    assert worst < 1e-9


def test_touchdown_start_rejected_with_hang_arm(monkeypatch, tmp_path):
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand", "touchdown_start": True, "hang_arm": True}, t_end=1.5)
    assert sim.node.core.active is False and not sim.targets()
    assert any("[거부] touchdown_start" in m for _, m in sim.logs)


def test_touchdown_start_refuses_hold_pose_far_from_policy_default(monkeypatch, tmp_path):
    """수평 자세 적용 누락(무릎 ≈ 0°, 정책 기본 ∓38.5°) → 홀드 래치 시점에 정지(동결 = 같은 홀드), 정책 0틱, 홀드 이동 명령 0."""
    ticks = {"n": 0}

    def probe(sim, node, t, i):
        ticks["n"] = max(ticks["n"], node.core.runner.core._tick)
    sim = run_sim(monkeypatch, tmp_path, {"cmd": "stand", "touchdown_start": True}, scenario=probe,
                  hold_off={"left_knee_joint": 38.5, "right_knee_joint": -38.5}, t_end=6.0)
    assert ticks["n"] == 0 and sim.node.core.active is False
    assert len(sim.stops()) == 1 and any("touchdown_pose_mismatch" in m for _, m in sim.logs)
    worst, _, _ = max_step(sim)
    assert worst < 1e-9
