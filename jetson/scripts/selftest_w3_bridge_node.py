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
1007 KE↔W3 합성 러너(--experimental-kew3): 합성 세션 시나리오(stand = KE → 켜기 다음 틱 W3 → 끄기 → 반주기 래치 → 조용한 양발 서기 0.5 s → KE,
KE 회복 모드 켜기 거부[응답 사유], stop 정지 래치·rearm, 한발 지지 트립 지연 동결, 재시작, --c5-overtime 은 KE 모드에서만) + 합성 플래그 꺼짐
(또는 켜도 다른 키) 세션의 발행이 연결 전 커밋(--ref-kew3)과 비트 동일 + 기동 거부 경로. 하위 러너는 스텁 정책, 묶는 것은 브리지 _kew3_compose.
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
        self.pitch = 0.0                   # 1007 감시 시험: 몸통 피치[rad](0 이면 종전 쿼터니언)
        self.s8 = None                     # 1007 감시 시험: stage8 cmd_timing.p50_age_ms_1s(None 이면 필드 없음 = 종전)
        self.foot_mute = False             # 1007 감시 시험: 발 메시지 끊김

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
        st = {"armed": True, "joints": js}
        if self.s8 is not None:
            st["cmd_timing"] = {"p50_age_ms_1s": self.s8}
        self.n._on_status(_Msg(json.dumps(st)))
        qt = [1.0, 0.0, 0.0, 0.0] if self.yaw == 0.0 else [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        if self.pitch != 0.0:              # 요 ψ 뒤 피치 θ (z-y 순)
            cy, sy, cp, sp = math.cos(self.yaw / 2), math.sin(self.yaw / 2), math.cos(self.pitch / 2), math.sin(self.pitch / 2)
            qt = [cy * cp, -sy * sp, cy * sp, sy * cp]
        self.n._on_imu(_Msg(json.dumps({"gyro_rad_s": list(self.gyro), "quat_wxyz": qt, "timestamp": now})))
        cl, cr = self.contact()
        if not self.foot_mute:
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


def scenario_burst(b, clock, out):
    """1007 시간 제한 걷기(콘솔 걷기 버튼의 브리지 쪽 안전장치): step 켜기에 dur_s → 기한 뒤 브리지가 스스로 끄기(명령 0·gait 0)·같은 req_id 의
    burst_end 응답. 기한 형식 검사, 기한 전 끄기·estop 이 기한을 지움, 걷는 중 명령 변경이 기한을 새로 검, dur_s 없는 켜기는 종전처럼 계속 걷기,
    버튼 표(W3_MOVES)의 모든 항목이 실제 감독기를 통과."""
    def driver(node):
        H = Harness(node, b, clock)
        R = []
        W = b._W3
        assert node.w3sup is not None and node.core.w3cmd
        H.tick(5)
        assert H.cmd({"cmd": "stand"})["accepted"]
        H.tick(int(2.6 / b.TICK_SEC))
        clk = node.core.runner._w3_clock
        sq = [0]

        def on(**kw):
            sq[0] += 1
            return H.cmd(dict({"cmd": "step", "enable": True, "req_id": f"btn-{sq[0]}", "seq": sq[0], "heading_loop": False}, **kw))

        def wait_stand():
            H.tick(int(1.5 / b.TICK_SEC))
            assert not clk.ge and not clk.want, (clk.ge, clk.want)
            H.tick(int(1.2 / b.TICK_SEC))            # 켜기 선행조건(양발 1 s·정착 1 s) 재충족

        for bad in (0.1, 11.0, "x", True):
            a = on(vx=0.06, dur_s=bad)
            assert not a["accepted"] and "dur_s" in a["why"] and not clk.want, (bad, a)
        R.append("기한 형식·범위 거부 4 종(켜지지 않음)")
        assert "burst" not in node.w3sup.snapshot(), "기한을 쓰기 전인데 snapshot 에 burst 필드"
        a = on(vx=0.06, dur_s=0.6)
        t0 = clock.t
        assert a["accepted"] and clk.want and node.core.cmd == (0.06, 0.0, 0.0) and a["sup"]["burst"]["dur_s"] == 0.6, a
        n_ack = len(H.acks)
        k = 0
        while clk.want and k < 200:
            H.tick(1); k += 1
        ran = clock.t - t0
        ends = [x for x in H.acks[n_ack:] if x.get("event") == "burst_end"]
        assert 0.6 - 1e-6 <= ran <= 0.6 + 2 * b.TICK_SEC and len(ends) == 1 and ends[0]["req_id"] == a["req_id"], (ran, ends)
        assert node.core.cmd == (0.0, 0.0, 0.0) and not clk.want and node.w3sup.burst_t_off is None
        t_off = clock.t
        while clk.ge:
            H.tick(1)
        R.append(f"전진 0.06·기한 0.6 s → {ran:.2f} s 뒤 브리지가 끄기(응답 burst_end 1 건, 명령 0) → 반주기 경계 {clock.t - t0:.2f} s 에 서기")
        assert not H.stop_alls, "기한 끄기가 STOP_ALL 을 냈다"
        wait_stand()
        a = on(wz=0.15, dur_s=2.0)                     # 기한 전 운영자 끄기 → 기한 해제(뒤에 burst_end 가 또 나오면 안 됨)
        assert a["accepted"] and clk.want
        H.tick(20)
        sq[0] += 1
        n_ack = len(H.acks)
        assert H.cmd({"cmd": "step", "enable": False, "req_id": "btn-off", "seq": sq[0]})["accepted"] and node.w3sup.burst_t_off is None
        H.tick(int(2.5 / b.TICK_SEC))
        assert not any(x.get("event") == "burst_end" for x in H.acks[n_ack:]) and not clk.want
        R.append("기한 전 끄기 → 기한 해제(뒤늦은 burst_end 없음)")
        H.tick(int(1.2 / b.TICK_SEC))
        a = on(vx=0.06, dur_s=5.0)                     # 걷는 중 명령 변경(+ 새 기한) → 새 기한만 유효
        assert a["accepted"]
        H.tick(int(1.1 / b.TICK_SEC))
        a2 = on(vx=0.04, dur_s=0.5)
        t1 = clock.t
        assert a2["accepted"] and a2["action"] == "set_cmd" and node.core.cmd == (0.04, 0.0, 0.0), a2
        k = 0
        while clk.want and k < 400:
            H.tick(1); k += 1
        assert 0.5 - 1e-6 <= clock.t - t1 <= 0.5 + 2 * b.TICK_SEC, clock.t - t1
        R.append("걷는 중 명령 변경 + 새 기한 0.5 s → 새 기한에 끄기(앞 기한 5 s 무효)")
        wait_stand()
        a = on(vx=0.06)                                # dur_s 없음 = 종전: 계속 걷기
        assert a["accepted"] and "dur_s" not in a and node.w3sup.burst_t_off is None
        H.tick(int(3.0 / b.TICK_SEC))
        assert clk.want and clk.ge, "기한 없는 켜기가 스스로 꺼짐"
        R.append("dur_s 없는 켜기 = 3 s 뒤에도 걷는 중(종전 동작)")
        a = on(vx=0.04, dur_s=3.0)                     # 기한 걸린 채 estop → 기한 해제·STOP_ALL 1 회
        H.tick(int(1.1 / b.TICK_SEC))
        a = on(vx=0.06, dur_s=3.0)
        assert a["accepted"] and node.w3sup.burst_t_off is not None, a
        n0 = len(H.stop_alls)
        assert H.cmd({"cmd": "estop"})["accepted"] and node.w3sup.burst_t_off is None and len(H.stop_alls) == n0 + 1
        R.append("기한 중 estop → 기한 해제·STOP_ALL")
        S = W.W3Supervisor(cmd_limits=W.W3S3D_CMD_LIMITS)   # 버튼 표 전 항목이 실제 감독기 형식·범위 검사를 통과
        t = 100.0
        for i in range(120):
            S.observe(t + i * b.TICK_SEC, (1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        for i, name in enumerate(W.W3_MOVES):
            c = W.w3_move_cmd(name)
            d = S.handle(dict({"cmd": "step", "enable": True, "req_id": f"m{i}", "seq": 1000 + i, "heading_loop": False}, **c), t + 2.38,
                         active=True, warmed=True, blend_done=True, ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
            assert d["accepted"] and d["dur_s"] == c["dur_s"] and tuple(d["cmd"]) == (c["vx"], c["vy"], c["wz"]), (name, d)
        R.append(f"버튼 표 {len(W.W3_MOVES)} 항목 감독기 통과")
        out["burst"] = R
    return driver


def scenario_mon(b, clock, out):
    """1007 2 차 대비책 감시(--w3-monitor on) 노드 통합: 자세(저역 피치 14°) → 걷기 정지·정지 래치·로그(동결 아님) → rearm 뒤 재개,
    stage8 명령 처리 나이 120 ms(0.3 s 지속) → 걷기 정지·켜기 거부(이유 '감시') → 정상화 뒤 허용, 발 메시지 0.3 s 끊김 → 걷기 정지
    (노드 하드 정지 0.35 s 전에), 블랙박스 w3_mon 필드."""
    def driver(node):
        H = Harness(node, b, clock)
        R = []
        assert node.core.w3mon is not None and node.core.w3cmd
        H.tick(5)
        assert H.cmd({"cmd": "stand"})["accepted"]
        H.tick(int(2.6 / b.TICK_SEC))
        clk = node.core.runner._w3_clock
        sq = [0]

        def nxt():
            sq[0] += 1
            return sq[0]

        def enable(**kw):
            return H.cmd(dict({"cmd": "step", "enable": True, "req_id": "op", "seq": nxt()}, **kw))

        def rearm():
            return H.cmd({"cmd": "rearm", "req_id": "r", "seq": nxt()})
        assert enable(vx=0.06)["accepted"]
        H.tick(int(2.0 / b.TICK_SEC))
        assert clk.ge and not node.w3sup.stop_latched
        n0 = len(H.stop_alls)
        H.pitch = math.radians(14.0)
        k = 0
        while clk.want and k < 100:
            H.tick(1); k += 1
        assert not clk.want and node.core.cmd == (0.0, 0.0, 0.0) and node.core.active and len(H.stop_alls) == n0, (k, node.core.cmd)
        assert node.w3sup.stop_latched and "monitor tilt" in node.w3sup.latch_reason, node.w3sup.latch_reason
        assert any("W3 감시 걷기 정지" in m for lv, m in node._log.lines)
        R.append(f"저역 피치 14° → {k * b.TICK_SEC:.2f}s 뒤 걷기 정지·정지 래치({node.w3sup.latch_reason[:24]}…)·동결 없음")
        H.pitch = 0.0
        H.tick(int(2.0 / b.TICK_SEC))
        a = enable(vx=0.06)
        assert not a["accepted"] and "정지 래치" in a["why"], a
        assert rearm()["accepted"]
        H.tick(int(1.2 / b.TICK_SEC))
        a = enable(vx=0.06)
        assert a["accepted"], a
        R.append("래치 → 켜기 거부 → rearm 뒤 재개")
        H.tick(int(1.0 / b.TICK_SEC))
        H.s8 = 120.0
        k = 0
        while clk.want and k < 100:
            H.tick(1); k += 1
        assert not clk.want and 0.28 <= k * b.TICK_SEC <= 0.36 and "monitor health" in node.w3sup.latch_reason, (k, node.w3sup.latch_reason)
        H.tick(int(1.5 / b.TICK_SEC))
        assert rearm()["accepted"]
        H.tick(int(1.2 / b.TICK_SEC))
        a = enable(vx=0.06)
        assert not a["accepted"] and "감시" in a["why"] and "stage8" in a["why"], a
        H.s8 = 1.0
        H.tick(2)
        a = enable(vx=0.06)
        assert a["accepted"], a
        R.append(f"stage8 명령 처리 나이 120 ms → {k * b.TICK_SEC:.2f}s 뒤 걷기 정지, 켜기 거부(감시)·정상화 뒤 허용")
        H.tick(int(1.0 / b.TICK_SEC))
        H.foot_mute = True
        k = 0
        while clk.want and k < 30:
            H.tick(1); k += 1
        H.foot_mute = False
        assert not clk.want and node.core.active and k * b.TICK_SEC < 0.35, (k, node.core.active, node.stop_reason)
        R.append(f"발 메시지 끊김 → {k * b.TICK_SEC:.2f}s 에 걷기 정지(하드 정지 0.35 s 전)")
        bb = [json.loads(m) for m in node._pubs[b.BRIDGE_DEBUG_TOPIC].msgs[-3:]]
        assert all("w3_mon" in d and d["w3_mon"]["mode"] == "act" for d in bb)
        R.append("디버그 w3_mon 필드")
        assert H.cmd({"cmd": "estop"})["accepted"]
        out["mon"] = R
    return driver


def scenario_s3d_benign(b, clock, out, key):
    """정상 입력 W3S3d 세션 — 감시 켬/끔 명령 토픽 비트 비교용(같은 입력열)."""
    def driver(node):
        H = Harness(node, b, clock)
        H.tick(5)
        assert H.cmd({"cmd": "stand"})["accepted"]
        H.tick(int(2.6 / b.TICK_SEC))
        assert H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 1, "vx": 0.06})["accepted"]
        H.yaw = 0.0
        for i in range(400):
            H.yaw = 0.02 * math.sin(i * 0.05)
            H.tick(1)
        H.cmd({"cmd": "step", "enable": False, "req_id": "op", "seq": 2})
        H.tick(100)
        out[key] = list(node._pubs[b.COMMAND_TOPIC].msgs)
    return driver


def scenario_off(b, clock, out):
    def driver(node):
        assert node.w3sup is None and node.pub_ack is None and b.BRIDGE_ACK_TOPIC not in node._pubs
        node._on_cmd(_Msg(json.dumps({"cmd": "step", "enable": True, "req_id": "a", "seq": 1})))
        assert not node.core.active
        out["off"] = "플래그 꺼짐: 감독기·응답 토픽 없음·step 무시(종전)"
    return driver


def scenario_kew3(b, clock, out):
    """1007 KE↔W3 합성 세션(--experimental-w3 --experimental-kew3 --checkpoint 합성 키) — 노드 _on_cmd → 감독기 → 러너 켜기 게이트 → 코어 → 합성 러너.
    하위 러너 둘은 스텁 정책(걷기 W3 러너 + C5Runner[실제 감지기 설정])이고 묶는 것은 브리지 _kew3_compose(전환 방식·정착 시간 = 실제 명세).
    접촉은 클록 위상 가정 모형(ge 1 한발 지지, ge 0 뒤 0.1 s 착지 지연). 순서·계약 검증이며 폐루프 성능 증명이 아니다."""
    def driver(node):
        H = Harness(node, b, clock)
        R = []
        key, core, dt = b.KEW3_CKPT_KEY, node.core, b.TICK_SEC
        run = core.runner
        clk = run._w3_clock
        assert node.w3sup is not None and core.kew3 and core.w3 and core.w3cmd and type(run).__name__ == "KeW3Runner"
        assert run.handoff == "carry" and run.settle_quiet_s == 0.5 and any("KE↔W3 합성 러너 켜짐" in m for lv, m in node._log.lines)
        sq = [0]

        def dbg():
            return json.loads(node._pubs[b.BRIDGE_DEBUG_TOPIC].msgs[-1])

        def nxt():
            sq[0] += 1
            return sq[0]

        def on(**kw):
            return H.cmd(dict({"cmd": "step", "enable": True, "req_id": "op", "seq": nxt(), "vx": 0.06}, **kw))

        def off():
            return H.cmd({"cmd": "step", "enable": False, "req_id": "op", "seq": nxt()})

        def rearm():
            return H.cmd({"cmd": "rearm", "req_id": "r", "seq": nxt()})

        def until(cond, limit, what):
            k = 0
            while not cond():
                H.tick(1)
                k += 1
                assert k <= limit, f"{what}: {limit} 틱 안에 일어나지 않음"
            return k

        class TripOsc:
            def update(self, now, q, v):
                return "저주파 반전 시험"
        # 시작: hang_arm 은 W3 키 규칙대로 거부, stand = KE 모드
        H.tick(5)
        a = H.cmd({"cmd": "stand", "hang_arm": True})
        assert not a["accepted"] and not core.active, a
        a = H.cmd({"cmd": "stand"})
        assert a["accepted"] and core.active and core.ckpt_name == key and run.mode == "ke" and a["kew3"]["mode"] == "ke", a
        H.tick(int(2.6 / dt))
        d = dbg()
        assert d["checkpoint"] == key and d["kew3"]["mode"] == "ke" and d["kew3"]["n_to_w3"] == 0 and d["kew3"]["block"] is None, d["kew3"]
        assert d["c5"]["kew3"]["mode"] == "ke" and d["c5"]["mode"] == 0 and {"raw_stand", "raw_rec", "ref_deg"} <= set(d["c5"])
        assert d["w3"]["ge"] is False and "w3_cmd" in d and "w3_sup" in d and not core.w3hl.latched and not H.stop_alls
        R.append("hang_arm 시작 거부·stand = KE 모드·디버그 kew3/c5(KE 필드 + kew3)")
        # KE → W3: 켜기 수락 뒤 첫 틱에 W3(STAND), 다음 틱 gait 0→1
        a = on()
        assert a["accepted"] and a["kew3"]["mode"] == "ke" and a["w3"]["want"] is True and core.cmd == (0.06, 0.0, 0.0), a
        H.tick(1)
        d = dbg()
        assert run.mode == "w3" and not clk.ge and d["kew3"]["mode"] == "w3" and d["kew3"]["event"] == "to_w3" and d["kew3"]["n_to_w3"] == 1, d["kew3"]
        assert d["c5"]["kew3"]["mode"] == "w3" and core.w3hl.latched
        H.tick(1)
        assert clk.ge and float(clk.u) == 0.0 and abs(float(run.last_obs[42]) - 0.06) < 1e-7
        R.append("켜기 수락(응답에 kew3) → 다음 틱 W3(STAND) → 그다음 틱 gait 0→1·왼발 리드·명령 관측")
        H.tick(100)
        a = H.cmd({"cmd": "stand"})
        assert not a["accepted"] and run.mode == "w3" and run.n_to_w3 == 1 and clk.ge, a
        # W3 → KE: 끄기 → 반주기 래치 → 조용한 양발 서기 0.5 s → KE(동결·재기동 없음)
        a = off()
        assert a["accepted"] and core.cmd == (0.0, 0.0, 0.0) and not clk.want, a
        k1 = until(lambda: not clk.ge, 30, "반주기 래치")
        assert run.mode == "w3" and dbg()["kew3"]["since_latch_s"] is not None
        k2 = until(lambda: run.mode == "ke", 60, "정착 뒤 KE 인계")
        d = dbg()
        assert 28 <= k2 <= 32 and d["kew3"]["event"] == "to_ke" and d["kew3"]["n_to_ke"] == 1 and len(d["raw_action"]) == 12, (k2, d["kew3"])
        assert core.active and not H.stop_alls and not core.w3hl.latched
        R.append(f"끄기 → 래치 {k1} 틱 → 착지 0.1 s + 조용한 양발 서기 0.5 s({k2} 틱) → KE(동결 없음)")
        # KE 회복 모드: 감독기는 수락(기울기 5°·조용·양발)하지만 러너 게이트가 거부 — 사유는 응답에
        H.tick(100)
        H.pitch = math.radians(4.5)          # 기준 대비 +4.5°: 켜기 3° 초과, 조기 복귀 허용 5° 미만, 끄기 1.5° 초과 → 회복 모드에 머묾
        H.tick(15)
        assert run.mode == "ke" and run.last_mode == 1 and dbg()["kew3"]["block"] and dbg()["c5"]["mode"] == 1, "시나리오 점검: KE 회복 모드"
        a = on()
        assert not a["accepted"] and a["action"] == "none" and "회복 모드" in a["why"] and a["kew3"]["block"] and not clk.want, a
        sup, node.w3sup = node.w3sup, None   # 감독기 없는 종전 step 경로(_on_cmd_base)도 같은 사유로 거부한다
        try:
            node._on_cmd(_Msg(json.dumps({"cmd": "step", "enable": True})))
        finally:
            node.w3sup = sup
        assert not clk.want and any("step 켜기 거부" in m and "회복 모드" in m for lv, m in node._log.lines)
        H.tick(10)
        assert run.mode == "ke" and run.n_to_w3 == 1 and core.cmd == (0.0, 0.0, 0.0)
        H.pitch = 0.0
        k3 = until(lambda: run.last_mode == 0, 120, "회복 모드 → 서기")
        a = on()
        assert a["accepted"], a
        H.tick(2)
        assert run.mode == "w3" and clk.ge and run.n_to_w3 == 2
        R.append(f"KE 회복 모드 켜기 거부(응답 사유·종전 step 경로 포함) → 서기 복귀 {k3} 틱 뒤 수락")
        # stop = 정지 래치(종전 의미): KE 로 돌아간 뒤에도 rearm 전 켜기 거부
        H.tick(50)
        a = H.cmd({"cmd": "stop"})
        assert a["accepted"] and not clk.want and core.cmd == (0.0, 0.0, 0.0), a
        until(lambda: run.mode == "ke", 80, "stop 뒤 KE 인계")
        assert core.active and not H.stop_alls and run.n_to_ke == 2
        a = on()
        assert not a["accepted"] and "정지 래치" in a["why"] and run.mode == "ke", a
        assert rearm()["accepted"]
        H.tick(int(1.2 / dt))
        assert on()["accepted"]
        # 한발 지지 중 가드 트립 → 지연 동결: 반주기 래치 → 양발 지지 확인 → STOP_ALL(KE 로 넘기기 전, W3 STAND 에서)
        H.tick(12)
        assert run.mode == "w3" and sum(H.contact()) == 1, "트립 시점이 한발 지지 걷기가 아님"
        real_osc, node.osc = node.osc, TripOsc()
        H.tick(1)
        assert not H.stop_alls and core.active, "한발 지지 트립에서 즉시 동결"
        k4 = until(lambda: bool(H.stop_alls), 200, "지연 동결")
        node.osc = real_osc
        assert not core.active and "support" in node.stop_reason and H.contact() == (True, True) and run.mode == "w3", (node.stop_reason, run.mode)
        R.append(f"stop 정지 래치(KE 뒤 켜기 거부 → rearm) / 한발 지지 트립 → {(k4 + 1) * dt:.2f}s 뒤 양발 지지(W3 STAND)에서 STOP_ALL")
        # 재시작(stand) = KE 새 시작, 래치 유지. KE 서기 중 트립은 그 틱에 동결
        a = H.cmd({"cmd": "stand"})
        assert a["accepted"] and core.active and run.mode == "ke" and not clk.want and not clk.ge and run.n_to_w3 == 3, a
        H.tick(int(2.6 / dt))
        a = on()
        assert not a["accepted"] and "정지 래치" in a["why"], a
        n_stop = len(H.stop_alls)
        node.osc = TripOsc()
        H.tick(1)
        node.osc = real_osc
        assert len(H.stop_alls) == n_stop + 1 and not core.active and run.mode == "ke" and "support" in node.stop_reason, node.stop_reason
        R.append("재시작 = KE 새 시작·래치 유지 / KE 서기 중 트립 = 같은 틱 동결")
        assert H.cmd({"cmd": "stand"})["accepted"]
        H.tick(20)
        assert H.cmd({"cmd": "estop"})["accepted"] and not core.active
        out["kew3"] = R
    return driver


def scenario_kew3_ov(b, clock, out):
    """1007 합성 세션 + --c5-overtime freeze:1 — 회복 모드 체류 상한은 **KE 모드에서만** 센다(입력 = 합성 러너 last_mode: W3 모드는 0).
    ① 걷기(W3 모드) 3 s 동안 사건 없음 ② KE 회복 모드 0.4 s 에서 (시험 전용: 켜기 게이트를 거치지 않고) W3 로 넘기면 체류 계수가 0 으로 —
    KE 내부 모드 값이 1 로 남아 있어도 W3 걷기 2 s 동안 사건 없음 ③ KE 로 돌아와 회복 모드에 1 s 머물면 양발 접촉 틱에 STOP_ALL(c5_overtime)."""
    def driver(node):
        H = Harness(node, b, clock)
        core, dt = node.core, b.TICK_SEC
        run = core.runner
        clk = run._w3_clock
        assert b.C5_OVERTIME is not None and core.kew3

        def no_event():
            return not H.stop_alls and not any("체류 상한" in m for lv, m in node._log.lines)

        def to_ke(seq):
            assert H.cmd({"cmd": "step", "enable": False, "req_id": "op", "seq": seq})["accepted"]
            k = 0
            while run.mode == "w3":
                H.tick(1)
                k += 1
                assert k <= 80, "끄기 뒤 KE 인계 없음"
            H.tick(100)                           # KE 감지기 기준 자세 고정(인계 뒤 1.5 s 추종 창)
        H.tick(5)
        assert H.cmd({"cmd": "stand"})["accepted"]
        H.tick(int(2.6 / dt))
        assert H.cmd({"cmd": "step", "enable": True, "req_id": "op", "seq": 1, "vx": 0.06})["accepted"]
        H.tick(150)                               # ① 걷기 3 s > 상한 1 s
        assert run.mode == "w3" and clk.ge and no_event()
        to_ke(2)
        H.pitch = math.radians(4.5)               # ② 기준 대비 +4.5° → 회복 모드(머묾)
        H.tick(20)
        assert run.mode == "ke" and run.last_mode == 1 and no_event(), "시나리오 점검: KE 회복 모드 0.4 s"
        core.cmd = (0.06, 0.0, 0.0)
        run.set_gait_enable(True)                 # 시험 전용: 게이트(runner_enable_gate)를 거치지 않은 직접 켜기
        H.tick(100)
        assert run.mode == "w3" and run.last_mode == 0 and run.ke.last_mode == 1 and no_event(), "W3 모드에서 회복 체류를 셈"
        to_ke(3)                                  # ③ 인계 때 감지기 새로(기준 = 지금 자세 4.5°)
        assert run.last_mode == 0 and no_event()
        H.pitch = math.radians(9.0)
        k = 0
        while not H.stop_alls:
            H.tick(1)
            k += 1
            assert k <= 100, "회복 모드 체류 상한 동결이 나가지 않음"
        assert node.stop_reason == "c5_overtime" and not core.active and run.mode == "ke" and 50 <= k <= 56, (node.stop_reason, k)
        assert any("C5 회복 모드 체류 상한" in m for lv, m in node._log.lines)
        out["kew3_ov"] = (f"걷기 3 s(W3 모드) 사건 없음 / KE 회복 모드 0.4 s 뒤 W3 로 넘어가면 체류 0(KE 내부 모드 1 이 남아도 걷기 2 s 사건 없음) / "
                          f"KE 회복 모드 체류 {k * dt:.2f}s 에 STOP_ALL(c5_overtime)")
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

    def stub_rec(obs):
        o = np.asarray(obs, dtype=np.float32)
        return np.tanh(o[12:24] * np.float32(2.0) - o[:12]).astype(np.float32)

    def make_runner(ck):
        kws = getattr(b, "KEW3_CKPTS", {})
        if ck in kws:                               # 1007 합성 키: 하위 스텁 러너 둘을 브리지의 합성 함수로 묶는다(전환 방식·정착 시간 = 실제 명세)
            return b._kew3_compose(ck, make_runner(kws[ck].w3), make_runner(kws[ck].ke))
        kw = dict(model_path=stub, obs_dim=320, gait_freq=1.1, default_pose_rad=b.CKPT_DEFAULT_POSE.get(ck),
                  act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
        if ck in getattr(b, "C5_PAIRS", {}):         # 1007: C5 키 = 스텁 정책 2 개 + 실제 감지기·cue 설정(브리지 _make_runner 와 같은 조립, odic·parity 옵션 제외)
            ex = b.C5_EXTRA.get(ck, {})
            kw.pop("model_path")
            return b.C5Runner(stub, stub_rec, detector=b.SwitchDetector(**b.C5_DETECTOR_KW, **ex.get("det", {})), **ex.get("runner", {}), **kw)
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


def replay_driver_c5(b):
    """1007 C5 키 세션의 결정적 입력열(기준 커밋 비트 비교용) — stand → 기울기 유지(회복 모드에 머묾 → --c5-overtime 이 있으면 체류 상한 동결) →
    재시작 → 기울기 펄스(회복 모드 왕복: 켜짐·cue·조기 복귀) → step(무시)·stop → 리프트 가드 → estop. 블랙박스 c5 필드가 틱마다 달라지는 구간을 지난다."""
    def driver(node):
        clk = b.time
        pose = b.CKPT_DEFAULT_POSE.get(node.core.ckpt_name) or b.CKPT_DEFAULT_POSE["stand_crouch_g21a"]
        st = {"contact": (True, True), "k": 0, "pitch": 0.0, "gy": 0.02}

        def feed():
            now = clk.t
            st["k"] += 1
            js = {}
            for i, (j, v) in enumerate(zip(b.JOINT_ORDER, pose)):
                m = node.frames.joint_rad_to_motor_deg(j, v) + 0.3 * math.sin(0.07 * st["k"] + i)
                js[j] = {"baseline_deg": m - 0.3, "latest_actual_deg": m, "feedback_age_sec": 0.0,
                         "commanded_relative_deg": 0.1, "feedback_torque_est": 0.5 * i}
            node._on_status(_Msg(json.dumps({"armed": True, "joints": js})))
            p = st["pitch"]
            node._on_imu(_Msg(json.dumps({"gyro_rad_s": [0.03 * math.sin(0.1 * st["k"]), st["gy"], 0.0],
                                          "quat_wxyz": [math.cos(p / 2), 0.0, math.sin(p / 2), 0.0], "timestamp": now})))
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
        tick(3)
        cmd({"cmd": "stand"}); tick(160)                         # 인계(블렌딩 1.5 s) + 감지기 기준 자세 고정(1.5 s)
        st["pitch"] = math.radians(4.5); tick(80)                # 기준 대비 +4.5° 유지 → 회복 모드 체류(조기 복귀 불가)
        st["pitch"] = 0.0
        cmd({"cmd": "stand"}); tick(160)
        st["pitch"] = math.radians(6.5); st["gy"] = 0.7; tick(1); st["gy"] = 0.02; tick(9)    # 펄스: 켜짐(앞 기울기 → cue)
        st["pitch"] = 0.0; tick(120)                             # 조기 복귀
        cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 1}); cmd({"cmd": "stop"}); tick(10)
        st["contact"] = (False, False); tick(int(1.2 / b.TICK_SEC)); st["contact"] = (True, True)   # 리프트 가드(safe) → STOP_ALL
        cmd({"cmd": "stand"}); tick(20)
        cmd({"cmd": "estop"}); tick(5)
    return driver


def replay(b, argv, t0=2_000_000_000.0, driver_fn=None):
    PUB_LOG.clear()
    _patch(b, FakeClock(t0))
    rc = _run_main(b, argv, (driver_fn or replay_driver)(b))
    assert rc == 0, f"main rc {rc}"
    return list(PUB_LOG)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="68493c11", help="플래그 꺼짐 노드 비트 비교 기준 커밋(감독기 도입 전)")
    ap.add_argument("--ref-w3", default="95946e41", help="W3-A 세션 노드 비트 비교 기준 커밋(1007 명령 채널 도입 전)")
    ap.add_argument("--ref-kew3", default="b79370ca", help="합성 플래그 꺼짐/다른 키 세션 노드 비트 비교 기준 커밋(1007 KE↔W3 합성 러너 연결 전)")
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
             ("W3S3d 시간 제한 걷기(1007 콘솔 걷기 버튼)", ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint",
                                                  b._W3.W3S3D_CKPT_KEY, "--w3-heading-loop", "off"], scenario_burst, "burst"),
             ("W3S3d 감시 켬 노드(1007 2 차 대비책)", ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint",
                                                   b._W3.W3S3D_CKPT_KEY, "--w3-monitor", "on"], scenario_mon, "mon"),
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
    try:   # 1007 2 차 대비책: W3S3d 정상 입력에서 감시 켬 = 끔 명령 토픽 비트 동일, --w3-monitor 거부 경로
        for mm in (b,):
            if not mm.EXPERIMENTAL_W3:
                pass
        res = {}
        for key, extra in (("m_off", []), ("m_on", ["--w3-monitor", "on"])):
            ck = FakeClock()
            _patch(b, ck)
            rc = _run_main(b, ["--live", "--force-frame", "--force-gains", "--experimental-w3", "--checkpoint", b._W3.W3S3D_CKPT_KEY] + extra,
                           scenario_s3d_benign(b, ck, res, key))
            assert rc == 0, rc
        la, lb = res["m_off"], res["m_on"]
        assert len(la) == len(lb) and la == lb, f"감시 켬/끔 명령 토픽 다름 ({len(la)} vs {len(lb)})"
        for argv in (["--w3-monitor", "on", "--selftest"], ["--experimental-w3", "--w3-monitor", '{"walk_tilt_deg": 40}', "--selftest"],
                     ["--checkpoint", "stand_crouch_g21a", "--c5-overtime", "note:6", "--selftest"]):
            rc = _run_main(b, argv, lambda n: None)
            assert rc == 2, (argv, rc)
        print(f"[PASS] W3S3d 정상 입력 감시 켬 = 끔 명령 토픽 비트 동일({len(la)}건)·--w3-monitor/--c5-overtime 거부 3 경로(exit 2)")
    except Exception as e:  # noqa: BLE001
        fails += 1
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 1007 감시 켬/끔 비트 비교: {type(e).__name__}: {e}")
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
    # ---- 1007 KE↔W3 합성 러너(--experimental-kew3) ----
    kw_mod = "rl_walking.deploy.kew3_runner"
    live = ["--live", "--force-frame", "--force-gains"]
    s3d, c5k = b._W3.W3S3D_CKPT_KEY, "stand_c5kr_g21a_g24ra"
    try:   # [K1] 합성 플래그 꺼짐(또는 켜도 다른 키) 세션 = 합성 러너 연결 전 커밋 발행 비트 동일 + 그때까지 kew3_runner 미적재(지연 import)
        assert kw_mod not in sys.modules, "합성 플래그를 켠 적이 없는데 kew3_runner 모듈이 올라와 있음(지연 import 깨짐)"
        refk = _load_ref(a.ref_kew3)
        assert not hasattr(refk, "KEW3_CKPTS"), f"{a.ref_kew3} 에 이미 합성 러너 연결이 있다 — 기준 커밋 확인"
        cases = [("W3S3d", live + ["--experimental-w3", "--checkpoint", s3d], None, replay_driver),
                 ("C5 kr", live + ["--checkpoint", c5k], None, replay_driver_c5),
                 ("C5 kr·체류 상한", live + ["--checkpoint", c5k, "--c5-overtime", "freeze:1"], None, replay_driver_c5),
                 ("W3S3d·합성 플래그 켬", live + ["--experimental-w3", "--checkpoint", s3d],
                  live + ["--experimental-w3", "--experimental-kew3", "--checkpoint", s3d], replay_driver),
                 ("C5 kr·W3+합성 플래그 켬", live + ["--experimental-w3", "--checkpoint", c5k],
                  live + ["--experimental-w3", "--experimental-kew3", "--checkpoint", c5k], replay_driver_c5)]
        notes = []
        for name, argv_ref, argv_new, drv in cases:
            if argv_new is None:                 # (합성 플래그를 켠 기동은 모듈 존재 확인을 위해 import 한다 — 꺼짐 세션만 본다)
                assert kw_mod not in sys.modules, f"{name}: 합성 플래그 꺼짐 세션이 kew3_runner 를 올림"
            logs = []
            for m, argv in ((b, argv_new or argv_ref), (refk, argv_ref)):
                for mm in (b, refk):
                    if mm.EXPERIMENTAL_W3:
                        mm.disable_experimental_w3()
                logs.append(replay(m, argv, driver_fn=drv))
            la, lb = logs
            assert len(la) == len(lb), f"{name}: 발행 수 {len(la)} vs 기준 {len(lb)}"
            for i, (x, y) in enumerate(zip(la, lb)):
                assert x == y, f"{name}: {i}번째 발행 다름\n new {x[0]} {x[1][:300]}\n ref {y[0]} {y[1][:300]}"
            dbgs = [json.loads(d) for t, d in la if t == b.BRIDGE_DEBUG_TOPIC]
            n_stop = sum('"STOP_ALL"' in d for t, d in la if t == b.COMMAND_TOPIC)
            n_ack = sum(1 for t, _ in la if t == b.BRIDGE_ACK_TOPIC)
            assert not any("kew3" in d for d in dbgs), f"{name}: 합성 키가 아닌 세션 디버그에 kew3 필드"
            if drv is replay_driver_c5:          # 입력열이 C5 분기(블랙박스 c5·회복 모드·체류 상한)를 실제로 지났는지
                ticks = [d for d in dbgs if d.get("raw_action") is not None]
                n_rec = sum(1 for d in ticks if d["c5"]["mode"] == 1)
                ov = sum(1 for d in dbgs if d.get("event") == "stop_all" and d.get("reason") == "c5_overtime")
                assert ticks and all(isinstance(d["c5"], dict) for d in ticks) and n_rec >= 30 and n_ack == 0, (name, len(ticks), n_rec, n_ack)
                assert ov == (1 if "--c5-overtime" in argv_ref else 0), f"{name}: c5_overtime 동결 {ov} 회"
                notes.append(f"{name} {len(la)}건(회복 모드 {n_rec} 틱·c5_overtime {ov}·STOP_ALL {n_stop})")
            else:
                assert n_ack >= 5 and n_stop >= 1 and all("w3" in d and "w3_cmd" in d for d in dbgs if d.get("raw_action") is not None), (name, n_ack, n_stop)
                notes.append(f"{name} {len(la)}건(응답 {n_ack}·STOP_ALL {n_stop})")
        print(f"[PASS] 합성 플래그 꺼짐/다른 키 세션 노드 = 기준 {a.ref_kew3} 발행 비트 동일·꺼짐 세션은 kew3_runner 미적재: " + ", ".join(notes))
    except Exception as e:  # noqa: BLE001
        fails += 1
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 합성 플래그 꺼짐 노드 비트 비교: {type(e).__name__}: {e}")
    kargv = live + ["--experimental-w3", "--experimental-kew3", "--checkpoint", b.KEW3_CKPT_KEY]
    for name, argv, sc, key in (("KE↔W3 합성 세션 노드(1007)", kargv, scenario_kew3, "kew3"),
                                ("KE↔W3 합성 세션 + 회복 체류 상한(1007)", kargv + ["--c5-overtime", "freeze:1"], scenario_kew3_ov, "kew3_ov")):
        try:
            ck = FakeClock()
            _patch(b, ck)
            rc = _run_main(b, argv, sc(b, ck, out))
            assert rc == 0, f"main rc {rc}"
            v = out[key]
            print(f"[PASS] {name}: " + (" / ".join(v) if isinstance(v, list) else v))
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    try:   # 기동 거부: 합성 키는 플래그 둘 다 필요, --runner-ema-fresh 2·--c5-settle·--c5-imu-f32 와 함께 불가
        K = b.KEW3_CKPT_KEY
        rej = (["--checkpoint", K, "--selftest"], ["--experimental-kew3", "--selftest"], ["--experimental-w3", "--checkpoint", K, "--selftest"],
               ["--experimental-w3", "--experimental-kew3", "--runner-ema-fresh", "2", "--selftest"],
               ["--experimental-w3", "--experimental-kew3", "--checkpoint", K, "--runner-ema-fresh", "2", "--selftest"],
               ["--experimental-w3", "--experimental-kew3", "--checkpoint", K, "--c5-settle", "odic", "--selftest"],
               ["--experimental-w3", "--experimental-kew3", "--checkpoint", K, "--c5-imu-f32", "--selftest"])
        for argv in rej:
            rc = _run_main(b, argv, lambda n: None)
            assert rc == 2, (argv, rc)
        assert b.RUNNER_EMA_FRESH == 1 and b.C5_SETTLE is None and not b.C5_IMU_F32, "거부된 기동이 전역 설정을 바꿈"
        print(f"[PASS] --experimental-kew3 기동 거부 {len(rej)} 경로(exit 2): 플래그 없이 합성 키·W3 플래그 없이·--runner-ema-fresh 2·--c5-settle·--c5-imu-f32")
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(f"[FAIL] --experimental-kew3 기동 거부: {type(e).__name__}: {e}")
    print("w3 node selftest:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
