#!/usr/bin/env python3
"""W3 세션 브리지 노드 통합 셀프테스트 (1006 감사 H3–H6·L1) — ROS·로봇·젯슨 불필요(가짜 rclpy·가짜 시계).

  python3 jetson/scripts/selftest_w3_bridge_node.py

rl_bridge_node.main() 을 **그대로** 실행하되 rclpy/std_msgs 를 가짜 모듈로 바꿔 끼우고, rclpy.spin(node) 자리에서 시나리오를 돌린다.
노드의 _on_cmd → _on_cmd_w3 → W3Supervisor, _tick 의 관측·지연 동결·가드 연결을 실제 코드로 지난다. 바꿔 끼우는 것(시험 전용):
  - 정책: W3 exported 가 아직 없어 _make_runner 를 스텁 정책(0 출력) W3 러너로 대체 — 정책 출력·폐루프 물리는 검증 대상이 아니다.
  - 하드웨어맵 guard_mode: 실제 맵은 "data"(가드가 정지하지 않음)라 지연 동결 경로를 보려면 "safe" 로 덮는다(맵 파일은 건드리지 않음).
  - 발 접촉: W3 클록 위상으로 만든 가정 모형(u<0.5 왼발 스윙, u≥0.5 오른발 스윙, ge 0 뒤 0.1 s 착지 지연).
  - 블랙박스: HOME 을 임시 폴더로 돌려 ~/logs 를 더럽히지 않는다. --live 지만 발행은 가짜 퍼블리셔 기록뿐(CAN·stage8 없음).
시나리오: stand 수락·재발사 거부·walk/march 교체 거부, 블렌딩 중 켜기 거부, 정상 켜기, 중복 seq, 스텝 중 stand(러너 리셋 없음),
stop 뒤 0.3 s 늦은 켜기·1 Hz 반복 enable 거부, rearm 뒤 재개, 접촉 없음 켜기 거부, 한발 지지 중 발진 가드 트립 → 동결 지연(양발 지지에서
STOP_ALL), 과속 트립 즉시 동결, 동결 뒤 rearm 전 켜기 거부. 마지막으로 플래그 꺼짐 기동은 감독기·응답 토픽을 만들지 않음을 확인.
**계약·순서 검증이다 — 폐루프 성능이나 실물 안전을 증명하지 않는다.**
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time as _real_time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class FakeClock:
    def __init__(self, t0=2_000_000_000.0):
        self.t = t0

    def time(self):
        return self.t

    def __getattr__(self, name):          # strftime·sleep 등은 진짜 time 으로
        return getattr(_real_time, name)


class _Msg:
    def __init__(self, data=""):
        self.data = data


PUB_LOG = []          # 모든 퍼블리셔 발행을 순서대로 (토픽, 데이터) — 플래그 꺼짐 노드 비트 비교용


class _Pub:
    def __init__(self, topic):
        self.topic, self.msgs = topic, []

    def publish(self, m):
        self.msgs.append(m.data)
        PUB_LOG.append((self.topic, m.data))


class _Log:
    def __init__(self):
        self.lines = []

    def _w(self, lvl):
        return lambda s: self.lines.append((lvl, str(s)))

    def __getattr__(self, name):
        return self._w(name)


class _Node:
    def __init__(self, name):
        self._subs, self._pubs, self._timers, self._log = {}, {}, [], _Log()

    def create_subscription(self, typ, topic, cb, depth):
        self._subs[topic] = cb

    def create_publisher(self, typ, topic, depth):
        p = _Pub(topic)
        self._pubs[topic] = p
        return p

    def create_timer(self, dt, cb):
        self._timers.append(cb)

    def get_logger(self):
        return self._log

    def destroy_node(self):
        pass


def _install_fake_ros(driver):
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = driver
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    rclpy.node = node_mod
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _Msg
    std.msg = std_msg
    sys.modules.update({"rclpy": rclpy, "rclpy.node": node_mod, "std_msgs": std, "std_msgs.msg": std_msg})


def _run_main(b, argv, driver):
    _install_fake_ros(driver)
    old = sys.argv
    sys.argv = ["rl_bridge_node.py"] + argv
    try:
        return b.main()
    finally:
        sys.argv = old


class Harness:
    """노드 입력 주입·틱 구동. 접촉은 W3 클록 위상 모형(가정)."""

    def __init__(self, node, b, clock):
        self.n, self.b, self.clk = node, b, clock
        self.pose = b.CKPT_DEFAULT_POSE[b._W3.W3_CKPT_KEY]
        self.off_t = None
        self.force_contact = None
        self.gyro = (0.02, 0.01, 0.0)
        self.yaw = 0.0                     # 1007: IMU 요(헤딩 고리 시험). 0 이면 종전과 같은 쿼터니언 [1,0,0,0]

    @property
    def acks(self):
        p = self.n._pubs.get(self.b.BRIDGE_ACK_TOPIC)
        return [json.loads(m) for m in p.msgs] if p else []

    @property
    def stop_alls(self):
        return [m for m in self.n._pubs[self.b.COMMAND_TOPIC].msgs if '"STOP_ALL"' in m]

    def contact(self):
        if self.force_contact is not None:
            return self.force_contact
        c = self.n.core.runner._w3_clock if self.n.core.w3 else None
        if c is not None and c.ge:
            self.off_t = None
            return (True, False) if float(c.u) >= 0.5 else (False, True)
        self.off_t = self.clk.t if self.off_t is None else self.off_t
        return (True, True) if self.clk.t - self.off_t >= 0.1 - 1e-9 else (True, False)

    def feed(self):
        now = self.clk.t
        js = {}
        for j, v in zip(self.b.JOINT_ORDER, self.pose):
            m = self.n.frames.joint_rad_to_motor_deg(j, v)
            js[j] = {"baseline_deg": m, "latest_actual_deg": m, "feedback_age_sec": 0.0, "commanded_relative_deg": 0.0}
        self.n._on_status(_Msg(json.dumps({"armed": True, "joints": js})))
        qt = [1.0, 0.0, 0.0, 0.0] if self.yaw == 0.0 else [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        self.n._on_imu(_Msg(json.dumps({"gyro_rad_s": list(self.gyro), "quat_wxyz": qt, "timestamp": now})))
        cl, cr = self.contact()
        self.n._on_foot(_Msg(json.dumps({"left_n": 50.0 * cl, "right_n": 50.0 * cr, "left_contact": cl,
                                         "right_contact": cr, "timestamp": now})))

    def tick(self, k=1):
        for _ in range(k):
            self.feed()
            self.n._tick()
            self.clk.t += self.b.TICK_SEC

    def cmd(self, d):
        self.n._on_cmd(_Msg(json.dumps(d)))
        return self.acks[-1] if self.acks else None


def scenario(b, clock, out):
    def driver(node):
        H = Harness(node, b, clock)
        R = []
        assert node.w3sup is not None and b.BRIDGE_ACK_TOPIC in node._pubs, "W3 세션인데 감독기/응답 토픽 없음"
        H.tick(5)
        a = H.cmd({"cmd": "stand"})
        assert a["accepted"] and node.core.active and node.core.ckpt_name == b._W3.W3_CKPT_KEY, a
        R.append("stand 수락")
        for c in ("stand", "walk", "march"):
            a = H.cmd({"cmd": c, "vx": 0.3})
            assert not a["accepted"] and node.core.ckpt_name == b._W3.W3_CKPT_KEY, (c, a)
        R.append("실행 중 stand·walk·march 거부(키 유지)")
        H.tick(10)
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 1})
        assert not a["accepted"] and "블렌딩" in a["why"], a
        R.append("블렌딩 중 켜기 거부")
        H.tick(int(2.0 / b.TICK_SEC))
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 2})
        assert a["accepted"] and node.core.runner._w3_clock.want, a
        H.tick(1)
        assert node.core.runner._w3_clock.ge, "켜기 뒤 다음 틱 ge 1 아님"
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 2})
        assert not a["accepted"] and "중복" in a["why"], a
        R.append("정상 켜기·중복 seq 거부")
        H.tick(7)
        tk = node.core.runner._tick
        a = H.cmd({"cmd": "stand"})
        H.tick(1)
        assert not a["accepted"] and node.core.runner._tick == tk + 1 and node.core.runner._w3_clock.ge, "스텝 중 stand 가 러너를 리셋"
        R.append("스텝 중 stand 거부(러너 리셋 없음)")
        # stop → 0.3 s 늦은 켜기 / 1 Hz 반복 enable
        rep = {"cmd": "step", "enable": True, "req_id": "pub", "seq": 3}
        assert H.cmd(rep)["accepted"] is True     # 이미 켜짐 → 변화 없음(수락)
        a = H.cmd({"cmd": "stop"})
        assert a["accepted"] and not node.core.runner._w3_clock.want
        H.tick(int(0.3 / b.TICK_SEC))
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "late", "seq": 4})
        assert not a["accepted"] and "정지 래치" in a["why"], a
        rep_acc = []
        for _ in range(4):
            H.tick(int(1.0 / b.TICK_SEC))
            rep_acc.append(H.cmd(rep)["accepted"])
        assert rep_acc == [False] * 4 and not node.core.runner._w3_clock.ge and node.core.active, rep_acc
        assert not H.stop_alls, "stop 이 동결을 냈다(stop = ge 0 서기여야 함)"
        R.append("stop 뒤 0.3 s 늦은 켜기 거부·1 Hz 반복 4회 거부·서기 유지")
        a = H.cmd({"cmd": "rearm", "req_id": "r", "seq": 5})
        assert a["accepted"], a
        assert not H.cmd(rep)["accepted"], "rearm 뒤 옛 반복 메시지 수락"
        H.force_contact = (True, False)
        H.tick(int(1.5 / b.TICK_SEC))
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 6})
        assert not a["accepted"] and "양발" in a["why"], a
        R.append("rearm 수락·옛 메시지 거부·한발 접촉 켜기 거부")
        H.force_contact = None
        H.tick(int(1.2 / b.TICK_SEC))
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 7})
        assert a["accepted"], a
        # 한발 지지 중 발진 가드 트립 → 지연 동결
        H.tick(12)
        assert sum(H.contact()) == 1, "트립 시점이 한발 지지가 아님"
        real_osc = node.osc

        class TripOsc:
            def update(self, now, q, v):
                return "저주파 반전 시험"
        node.osc = TripOsc()
        n_trip = node.core.runner._tick
        H.tick(1)
        assert not H.stop_alls and node.core.active, "한발 지지 트립에서 즉시 동결"
        k = 0
        while not H.stop_alls and k < 200:
            H.tick(1)
            k += 1
        assert H.stop_alls and not node.core.active, "지연 동결이 끝내 나가지 않음"
        frz_contact = H.contact()
        sr = node.stop_reason
        assert "support" in sr and frz_contact == (True, True), (sr, frz_contact)
        dt_frz = (k + 1) * b.TICK_SEC
        assert dt_frz < b._W3.W3_SAFETY_DEFAULT.freeze_timeout_s
        R.append(f"한발 지지 트립 → {dt_frz:.2f}s 뒤 양발 지지에서 STOP_ALL({sr.split('→')[-1].strip()[:40]}…)")
        node.osc = real_osc
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 8})
        assert not a["accepted"], a
        # 재시작(stand) 뒤에도 래치 유지 → rearm 필요
        a = H.cmd({"cmd": "stand"})
        assert a["accepted"] and node.core.active, a
        H.tick(int(2.6 / b.TICK_SEC))
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 9})
        assert not a["accepted"] and "정지 래치" in a["why"], a
        assert H.cmd({"cmd": "rearm", "req_id": "r", "seq": 10})["accepted"]
        assert H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 11})["accepted"]
        H.tick(12)
        n_stop = len(H.stop_alls)

        class HardOsc:
            def update(self, now, q, v):
                return "과속 13.0 rad/s 시험"
        node.osc = HardOsc()
        H.tick(1)
        assert len(H.stop_alls) == n_stop + 1 and not node.core.active and "과속" in node.stop_reason, node.stop_reason
        node.osc = real_osc
        R.append("동결 뒤 재시작해도 래치 유지(rearm 필요)·과속 트립 즉시 동결")
        a = H.cmd({"cmd": "estop"})
        assert a["accepted"]
        n_ack = len(H.acks)
        node._on_cmd(_Msg('"estop"'))
        assert len(H.acks) == n_ack + 1 and not H.acks[-1]["accepted"]
        R.append(f"응답 {len(H.acks)}건(모든 명령)·비객체 JSON 응답")
        out["w3"] = R
        out["n_trip_tick"] = n_trip
    return driver


def scenario_s3d(b, clock, out):
    """1007 W3S3d 세션(명령 채널·리드 발·헤딩 고리) — 노드 _on_cmd → 감독기 → 코어 → 러너 관측까지 실제 코드.
    관측 인덱스: velocity_commands 최신 프레임 = obs[42:45](vx, vy, wz)."""
    def driver(node):
        H = Harness(node, b, clock)
        R = []
        k = b._W3.W3S3D_CKPT_KEY
        assert node.w3sup is not None and node.w3sup.cmd_limits is not None and node.core.w3cmd
        H.tick(5)
        assert H.cmd({"cmd": "stand"})["accepted"] and node.core.ckpt_name == k
        H.tick(int(2.6 / b.TICK_SEC))
        clk = node.core.runner._w3_clock
        sq = 0
        for bad, key in (({"vy": 0.06}, "범위"), ({"vx": 0.05, "wz": 0.1}, "단일 축"), ({"vx": 0.01}, "범위"), ({"wz": "x"}, "유한")):
            sq += 1
            a = H.cmd(dict({"cmd": "step", "enable": True, "req_id": "op", "seq": sq}, **bad))
            assert not a["accepted"] and key in a["why"] and not clk.want, (bad, a)
        R.append("범위 밖·혼합·사이값·형식 거부(켜지지 않음)")
        H.yaw = 0.3
        H.tick(3)
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "vy": -0.04})
        assert a["accepted"] and clk.want and node.core.cmd == (0.0, -0.04, 0.0), a
        H.tick(1)
        obs = node.core.runner.last_obs
        assert clk.ge and float(clk.u) == 0.5 and abs(float(obs[43]) + 0.04) < 1e-7 and float(obs[44]) == 0.0, (clk.u, obs[42:45])
        R.append("옆 −0.04 켜기 → 같은 틱 명령·오른발 리드(위상 0.5)·고리 래치 출력 0")
        H.yaw = 0.3 - 0.05                       # 요가 오른쪽으로 0.05 rad 틀어짐 → 고리 wz = +K·0.05
        H.tick(2)
        wz = float(node.core.runner.last_obs[44])
        assert abs(wz - 0.5 * 0.05) < 1e-4 and node.core.w3hl.latched, wz
        R.append(f"헤딩 오차 −2.9° → wz {wz:+.4f}(K 0.5)")
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "vx": 0.06})
        assert not a["accepted"] and "체류" in a["why"] and node.core.cmd == (0.0, -0.04, 0.0), a
        H.tick(int(1.0 / b.TICK_SEC))
        u_before = float(clk.u)
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "vx": 0.06})
        assert a["accepted"] and a["action"] == "set_cmd" and node.core.cmd == (0.06, 0.0, 0.0), a
        H.tick(1)
        assert abs(float(clk.u) - (u_before + 1.1 * 0.02) % 1.0) < 1e-6 and abs(float(node.core.runner.last_obs[42]) - 0.06) < 1e-7
        R.append("체류 1 s 전 변경 거부 → 뒤 변경 수락(위상 유지·명령 관측 반영)")
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "vx": 0.06})
        assert a["accepted"] and a["action"] == "none", a
        a = H.cmd({"cmd": "stop"})
        assert a["accepted"] and not clk.want and node.core.cmd == (0.0, 0.0, 0.0)
        H.tick(1)
        assert not node.core.w3hl.latched and float(node.core.runner.last_obs[44]) == 0.0
        R.append("stop → 명령 0·ψ* 해제")
        H.tick(int(1.5 / b.TICK_SEC))
        sq += 1
        assert H.cmd({"cmd": "rearm", "req_id": "r", "seq": sq})["accepted"]
        H.yaw = 1.0
        H.tick(int(1.2 / b.TICK_SEC))
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "wz": -0.15})
        assert a["accepted"] and node.core.cmd == (0.0, 0.0, -0.15), a
        H.tick(1)
        assert float(clk.u) == 0.5 and not node.core.w3hl.latched and abs(float(node.core.runner.last_obs[44]) + 0.15) < 1e-7
        R.append("회전 −0.15 켜기 → 오른발 리드·고리 끔(wz 그대로)")
        H.tick(int(1.0 / b.TICK_SEC))
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq})
        assert a["accepted"] and a["action"] == "set_cmd" and node.core.cmd == (0.0, 0.0, 0.0), a
        H.tick(2)
        assert node.core.w3hl.latched
        H.yaw = node.core.w3hl.tgt + math.radians(20.0)     # IMU 요 점프(오차 상한 15° 초과)
        n_err = sum(1 for lv, m in node._log.lines if "헤딩 고리 고장" in m)
        H.tick(1)
        assert not clk.want and node.core.cmd == (0.0, 0.0, 0.0) and node.core.active and not H.stop_alls
        assert sum(1 for lv, m in node._log.lines if "헤딩 고리 고장" in m) == n_err + 1
        R.append("IMU 요 점프 20° → 고리 고장 → 걷기 정지(서기, 동결 아님)·로그")
        H.tick(int(1.5 / b.TICK_SEC))
        sq += 1
        a = H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": sq, "vx": -0.04, "heading_loop": False})
        assert a["accepted"] and node.core.w3hl_on is False, a
        H.tick(1)
        assert float(clk.u) == 0.0 and not node.core.w3hl.latched
        R.append("뒤 −0.04·heading_loop false → 왼발 리드·고리 미사용")
        # 가드 트립(지연 동결 경로) → 명령 0
        H.tick(12)
        real_osc = node.osc

        class TripOsc:
            def update(self, now, q, v):
                return "저주파 반전 시험"
        node.osc = TripOsc()
        H.tick(1)
        node.osc = real_osc
        assert node.core.cmd == (0.0, 0.0, 0.0) and not clk.want, node.core.cmd
        R.append("가드 트립 → 명령 0·ge 0(지연 동결 경로)")
        assert H.cmd({"cmd": "estop"})["accepted"]
        out["s3d"] = R
    return driver


def scenario_off(b, clock, out):
    def driver(node):
        assert node.w3sup is None and node.pub_ack is None and b.BRIDGE_ACK_TOPIC not in node._pubs
        node._on_cmd(_Msg(json.dumps({"cmd": "step", "enable": True, "req_id": "a", "seq": 1})))
        assert not node.core.active
        out["off"] = "플래그 꺼짐: 감독기·응답 토픽 없음·step 무시(종전)"
    return driver


def _patch(b, clock):
    """시험 전용 치환: 가짜 시계·guard_mode safe·스텁 정책 러너(관측 의존 결정적 출력). 원래 함수는 한 번만 저장."""
    import numpy as np
    b.time = clock
    if not hasattr(b, "_st_real_load"):
        b._st_real_load = b.load_frame_map

    def load_safe():
        f = b._st_real_load()
        f.guard_mode = "safe"
        return f
    b.load_frame_map = load_safe

    def stub(obs):
        o = np.asarray(obs, dtype=np.float32)
        return np.tanh(o[:12] * np.float32(3.0) + o[12:24]).astype(np.float32)

    def make_runner(ck):
        kw = dict(model_path=stub, obs_dim=320, gait_freq=1.1, default_pose_rad=b.CKPT_DEFAULT_POSE.get(ck),
                  act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
        if ck in b.W3_CKPTS:
            kw["w3"] = b.W3_CKPTS[ck]
        return b.PolicyRunnerStage4(**kw)
    b._make_runner = make_runner


def _load_ref(rev):
    """기준 커밋의 rl_bridge_node.py 를 같은 __file__ 경로로 실행한 별도 모듈(REPO·맵 경로 동일)."""
    import subprocess
    src = subprocess.check_output(["git", "show", f"{rev}:jetson/scripts/rl_bridge_node.py"],
                                  cwd=os.path.dirname(os.path.dirname(HERE)))
    mod = types.ModuleType("rl_bridge_node_ref")
    mod.__file__ = os.path.join(HERE, "rl_bridge_node.py")
    sys.modules["rl_bridge_node_ref"] = mod
    exec(compile(src, mod.__file__ + f"@{rev}", "exec"), mod.__dict__)
    return mod


def replay_driver(b):
    """플래그 꺼짐(또는 비 W3 키) 세션의 결정적 입력열 — stand·step(무시)·turn·rearm(무시)·walk·stop·march·리프트 가드·
    발진 가드·과속·estop·불량 JSON."""
    def driver(node):
        clk = b.time
        pose = b.CKPT_DEFAULT_POSE.get(node.core.ckpt_name) or b.CKPT_DEFAULT_POSE["stand_crouch_g21a"]
        st = {"contact": (True, True), "k": 0}

        def feed():
            now = clk.t
            st["k"] += 1
            js = {}
            for i, (j, v) in enumerate(zip(b.JOINT_ORDER, pose)):
                m = node.frames.joint_rad_to_motor_deg(j, v) + 0.3 * math.sin(0.07 * st["k"] + i)
                js[j] = {"baseline_deg": m - 0.3, "latest_actual_deg": m, "feedback_age_sec": 0.0,
                         "commanded_relative_deg": 0.1, "feedback_torque_est": 0.5 * i}
            node._on_status(_Msg(json.dumps({"armed": True, "joints": js})))
            node._on_imu(_Msg(json.dumps({"gyro_rad_s": [0.03 * math.sin(0.1 * st["k"]), 0.02, 0.0],
                                          "quat_wxyz": [0.9998, 0.01, -0.015, 0.0], "timestamp": now})))
            cl, cr = st["contact"]
            node._on_foot(_Msg(json.dumps({"left_n": 40.0 * cl, "right_n": 40.0 * cr, "left_contact": cl,
                                           "right_contact": cr, "timestamp": now})))

        def tick(n):
            for _ in range(n):
                feed()
                node._tick()
                clk.t += b.TICK_SEC

        def cmd(d):
            node._on_cmd(_Msg(json.dumps(d)))

        class Osc:
            def __init__(self, why):
                self.why = why

            def update(self, now, q, v):
                return self.why
        tick(3)
        cmd({"cmd": "stand"}); tick(90)
        cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 1}); tick(5)
        cmd({"cmd": "turn", "heading_target_deg": 10}); cmd({"cmd": "rearm", "req_id": "a", "seq": 2}); tick(5)
        cmd({"cmd": "walk", "vx": 0.2}); tick(40)
        cmd({"cmd": "stop"}); tick(10)
        cmd({"cmd": "march"}); tick(30)
        cmd({"cmd": "stand"}); tick(20)
        st["contact"] = (False, False); tick(int(1.2 / b.TICK_SEC))          # 리프트 가드(safe) → STOP_ALL
        st["contact"] = (True, True)
        cmd({"cmd": "stand"}); tick(20)
        real = node.osc
        node.osc = Osc("저주파 반전 시험"); tick(3); node.osc = real            # 발진 가드 → STOP_ALL
        cmd({"cmd": "stand"}); tick(20)
        node.osc = Osc("과속 13 rad/s 시험"); tick(3); node.osc = real          # 과속 → STOP_ALL
        cmd({"cmd": "stand"}); tick(20)
        cmd({"cmd": "estop"}); node._on_cmd(_Msg('"estop"')); node._on_cmd(_Msg("{bad")); tick(5)
    return driver


def replay(b, argv, t0=2_000_000_000.0):
    PUB_LOG.clear()
    _patch(b, FakeClock(t0))
    rc = _run_main(b, argv, replay_driver(b))
    assert rc == 0, f"main rc {rc}"
    return list(PUB_LOG)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="68493c11", help="플래그 꺼짐 노드 비트 비교 기준 커밋(감독기 도입 전)")
    ap.add_argument("--ref-w3", default="95946e41", help="W3-A 세션 노드 비트 비교 기준 커밋(1007 명령 채널 도입 전)")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="w3node_")
    os.environ["HOME"] = tmp
    import rl_bridge_node as b
    out, fails = {}, 0
    clock = FakeClock()
    _patch(b, clock)
    tests = [("W3 세션 노드", ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint", b._W3.W3_CKPT_KEY],
              scenario, "w3"),
             ("W3S3d 세션 노드(1007)", ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint",
                                        b._W3.W3S3D_CKPT_KEY], scenario_s3d, "s3d"),
             ("플래그 꺼짐 노드", ["--force-frame", "--checkpoint", "stand_crouch_g21a"], scenario_off, "off")]
    for name, argv, sc, key in tests:
        try:
            if key == "off" and b.EXPERIMENTAL_W3:
                b.disable_experimental_w3()
            rc = _run_main(b, argv, sc(b, clock, out))
            assert rc == 0, f"main rc {rc}"
            v = out[key]
            print(f"[PASS] {name}: " + (" / ".join(v) if isinstance(v, list) else v))
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    # [E] 플래그 꺼짐·비 W3 키 노드 = 기준 커밋 노드 (발행 메시지 전부 순서·비트 비교)
    try:
        ref = _load_ref(a.ref)
        notes = []
        for argv in (["--live", "--force-frame", "--force-gains", "--checkpoint", "stand_crouch_g21a"],
                     ["--live", "--force-frame", "--force-gains", "--checkpoint", "stand_v18b"],
                     ["--force-frame", "--checkpoint", "stand_crouch_g21a"],
                     ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint", "stand_crouch_g21a"]):
            logs = []
            for m in (b, ref):
                for mm in (b, ref):
                    if mm.EXPERIMENTAL_W3:
                        mm.disable_experimental_w3()
                logs.append(replay(m, argv))
            la, lb = logs
            assert len(la) == len(lb), f"{argv}: 발행 수 {len(la)} vs 기준 {len(lb)}"
            for i, (x, y) in enumerate(zip(la, lb)):
                assert x == y, f"{argv}: {i}번째 발행 다름\n new {x[0]} {x[1][:300]}\n ref {y[0]} {y[1][:300]}"
            n_stop = sum('"STOP_ALL"' in d for t, d in la if t == b.COMMAND_TOPIC)
            assert n_stop >= 4 or "--live" not in argv, f"{argv}: STOP_ALL {n_stop} — 가드 경로를 안 지남"
            assert not any(t == b.BRIDGE_ACK_TOPIC for t, _ in la), "비 W3 세션이 응답 토픽에 발행"
            notes.append(f"{argv[-1]}{'·live' if '--live' in argv else '·dry'}{'·w3플래그' if '--experimental-w3' in argv else ''} "
                         f"{len(la)}건(STOP_ALL {n_stop})")
        print(f"[PASS] 플래그 꺼짐/비 W3 키 노드 = 기준 {a.ref} 발행 비트 동일: " + ", ".join(notes))
    except Exception as e:  # noqa: BLE001
        fails += 1
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 플래그 꺼짐 노드 비트 비교: {type(e).__name__}: {e}")
    # [G] 1007: W3-A 세션 노드(--experimental-w3 --checkpoint stand_w3a_11699) = 1007 변경 전 커밋 발행 비트 동일
    try:
        ref2 = _load_ref(a.ref_w3)
        argv = ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint", b._W3.W3_CKPT_KEY]
        logs = []
        for m in (b, ref2):
            for mm in (b, ref2):
                if mm.EXPERIMENTAL_W3:
                    mm.disable_experimental_w3()
            logs.append(replay(m, argv))
        la, lb = logs
        assert len(la) == len(lb), f"발행 수 {len(la)} vs 기준 {len(lb)}"
        for i, (x, y) in enumerate(zip(la, lb)):
            assert x == y, f"{i}번째 발행 다름\n new {x[0]} {x[1][:300]}\n ref {y[0]} {y[1][:300]}"
        n_ack = sum(1 for t, _ in la if t == b.BRIDGE_ACK_TOPIC)
        assert n_ack >= 5, n_ack
        print(f"[PASS] W3-A 세션 노드 = 기준 {a.ref_w3} 발행 비트 동일: {len(la)}건(응답 {n_ack})")
    except Exception as e:  # noqa: BLE001
        fails += 1
        import traceback
        traceback.print_exc()
        print(f"[FAIL] W3-A 세션 노드 비트 비교: {type(e).__name__}: {e}")
    try:   # 1007 기동 설정 거부: 범위 밖 헤딩 고리·명령 상한, 플래그 없이
        for argv in (["--experimental-w3", "--w3-heading-loop", '{"kp": 3}', "--selftest"],
                     ["--experimental-w3", "--w3-cmd-limits", '{"vy_max": 0.2}', "--selftest"],
                     ["--w3-heading-loop", "off", "--selftest"], ["--checkpoint", b._W3.W3S3D_CKPT_KEY, "--selftest"]):
            rc = _run_main(b, argv, lambda n: None)
            assert rc == 2, (argv, rc)
        print("[PASS] --w3-heading-loop·--w3-cmd-limits 범위 밖 거부·플래그 없이 거부·W3S3d 키 플래그 없이 거부(exit 2)")
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] 1007 기동 설정: {type(e).__name__}: {e}")
    try:   # --experimental-w3 플래그 단독 검증: 범위 밖 설정은 기동 거부
        rc = _run_main(b, ["--experimental-w3", "--w3-safety", '{"enable_tilt_deg": 40}', "--selftest"], lambda n: None)
        assert rc == 2, rc
        rc = _run_main(b, ["--w3-safety", '{"enable_tilt_deg": 5}', "--selftest"], lambda n: None)
        assert rc == 2, rc
        print("[PASS] --w3-safety: 범위 밖 거부(exit 2)·플래그 없이 거부(exit 2)")
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] --w3-safety: {type(e).__name__}: {e}")
    print("w3 node selftest:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
