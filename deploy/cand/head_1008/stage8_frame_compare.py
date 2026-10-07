#!/usr/bin/env python3
"""젯슨 판 ↔ 새 판 stage8(12 축 MIT 제어 노드) CAN 프레임 비교 — 같은 명령열에서 버스로 나가는 프레임이 같은지 본다.

  python3 stage8_frame_compare.py --repo <트리 A> --dump a.json
  python3 stage8_frame_compare.py --repo <트리 B> --dump b.json
  python3 stage8_frame_compare.py --compare a.json b.json      # 종료 코드 0 = 프레임·공통 상태 전부 같음, 1 = 다름, 2 = 실행 불가

**실 CAN·ROS 를 쓰지 않는다**: `can`·`rclpy`·`std_msgs` 를 가짜 모듈로 바꿔 끼운 뒤 그 트리의 stage8 `main()` 을 콘솔과 같은 기동 인자
(`--can-channel can1 --wait-takeover`, 그리고 인수 대기 없는 기동)로 그대로 돌린다. 가짜 `can.Bus` 는 프레임을 기록하고 명령 위치를 되받는
응답만 만든다(소켓을 열 방법이 없다). 시계는 가짜(틱 20 ms), 인수 상태 파일은 임시 폴더.
대본: ① 인수 대기(침묵) → 상태 파일 등장 → 인수 홀드 송신 → 기준점 → ARM → 브리지 형식 배치 목표(정상·슬루 초과·한계 밖·모르는 관절)
      → 단건 목표 → STOP_ALL → 재개 → 불량 JSON → 무장 중 CHECK_CAN 거부 → DISARM
      ② 인수 대기 없는 기동: 읽기 폴링 → 기준점 → ARM → 목표 → 피드백 두절(3 s 뒤 램프 동결·재송신, 60 s 뒤 자동 무장 해제) → 복귀 → 재무장
비교: 단계마다 버스로 나간 프레임(모터 ID·8 바이트)과 status JSON. status 는 두 판에 모두 있는 키만 비교하고 한쪽에만 있는 키는 따로 적는다.
**개루프 계약 비교다 — 실버스 타이밍·모터 응답·폐루프 성능은 보지 않는다.**"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import time as _real_time
import types

T0 = 2_000_000_000.0
DT = 0.02


class FakeClock:
    def __init__(self, t0=T0):
        self.t = t0

    def time(self):
        return self.t

    def __getattr__(self, name):          # monotonic·sleep·strftime 은 진짜 time 으로
        return getattr(_real_time, name)


class _CanMessage:
    def __init__(self, arbitration_id=0, data=None, is_extended_id=False, **_kw):
        self.arbitration_id = int(arbitration_id)
        self.data = bytearray(data or [])
        self.is_extended_id = bool(is_extended_id)


class _CanError(Exception):
    pass


class FakeBus:
    """프레임 기록 + 응답 합성. 명령 프레임(8 바이트) → 그 위치를 담은 응답, 읽기 폴링(FF…FC) → 현재 위치 응답. 소켓 없음."""

    def __init__(self, env):
        self.env = env
        self.rx = []
        self.reply = True
        self.pos = dict(env["pos0"])      # 모터 ID → 위치 2 바이트

    def send(self, message, timeout=None):
        mid, d = int(message.arbitration_id), list(message.data)
        self.env["frames"].append([mid, bytes(d).hex()])
        if not self.reply:
            return
        if d != self.env["fc"]:
            self.pos[mid] = (d[0], d[1])
        p = self.pos.get(mid)
        if p is not None:
            self.rx.append(_CanMessage(arbitration_id=mid, data=[mid, p[0], p[1], 0x7F, 0xF7, 0xFF, 30, 0]))

    def recv(self, timeout=None):
        return self.rx.pop(0) if self.rx else None

    def shutdown(self):
        pass


def install_fakes(env):
    can = types.ModuleType("can")
    can.Message = _CanMessage
    can.CanError = _CanError
    can.Bus = lambda *a, **k: env.setdefault("bus", FakeBus(env))
    can.__fake__ = True

    class _Pub:
        def __init__(self, topic):
            self.topic = topic

        def publish(self, m):
            env["status"].append(m.data)

    class _RosNode:
        def __init__(self):
            self.subs, self.timers = {}, []

        def create_publisher(self, typ, topic, depth):
            return _Pub(topic)

        def create_subscription(self, typ, topic, cb, depth):
            self.subs[topic] = cb

        def create_timer(self, period, cb):
            self.timers.append((float(period), cb))

        def destroy_node(self):
            pass

    class _String:
        def __init__(self, data=""):
            self.data = data

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.ok = lambda: False
    rclpy.shutdown = lambda *a, **k: None
    rclpy.create_node = lambda name: env.setdefault("node", _RosNode())
    rclpy.spin = lambda node: env["driver"](node)
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _String
    std.msg = std_msg
    sys.modules.update({"can": can, "rclpy": rclpy, "std_msgs": std, "std_msgs.msg": std_msg})
    return _String


def mid_soft_motor_deg(s8, joint):
    lim = s8.get_absolute_limits_deg()[joint]
    return s8.STAND_ZERO_DEG_BY_JOINT[joint] + s8.DIRECTION_BY_JOINT[joint] * 0.5 * (lim["soft_min"] + lim["soft_max"])


class Rig:
    """가짜 ROS 노드의 타이머·구독 콜백을 20 ms 틱으로 구동하고 단계별 프레임·status 를 모은다."""

    def __init__(self, s8, env, node, clk, String):
        self.s8, self.env, self.node, self.clk, self.String = s8, env, node, clk, String
        self.steps = []
        self.seq = 0
        self.cmd_cb = node.subs[s8.COMMAND_TOPIC]
        self.control = [cb for p, cb in node.timers if abs(p - s8.CONTROL_PERIOD_SEC) < 1e-9]
        self.status = [cb for p, cb in node.timers if abs(p - s8.STATUS_PERIOD_SEC) < 1e-9]
        self.other = [(p, cb) for p, cb in node.timers if abs(p - s8.CONTROL_PERIOD_SEC) >= 1e-9 and abs(p - s8.STATUS_PERIOD_SEC) >= 1e-9]

    def _flush(self, label):
        # status 는 수 KB JSON 이 틱마다 2 건 — 통째로 두면 덤프가 100 MB 를 넘는다 → 키별 값 해시만 남긴다(비교는 키 단위)
        st = [{k: hashlib.md5(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
               for k, v in json.loads(x).items()} for x in self.env["status"]]
        self.steps.append({"label": label, "frames": self.env["frames"][:], "status": st})
        del self.env["frames"][:]
        del self.env["status"][:]

    def cmd(self, payload, label=None):
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.cmd_cb(self.String(data))
        self._flush(label or ("cmd:" + (payload if isinstance(payload, str) else str(payload.get("command")))))

    def targets(self, rel_by_joint, label="targets"):
        self.seq += 1
        self.cmd({"command": self.s8.COMMAND_SET_JOINT_TARGETS, "targets": rel_by_joint, "seq": self.seq,
                  "sent_ts": self.clk.t, "src": "bridge:4242:2000000000"}, label)

    def tick(self, n=1, label="tick"):
        for _ in range(n):
            self.clk.t += DT / 2
            for cb in self.status:
                cb()
            self.clk.t += DT / 2
            for cb in self.control:
                cb()
            for cb in self.status:
                cb()
            self._flush(label)


def drive_console_flow(s8, env, clk, String, state_file):
    def driver(node):
        R = Rig(s8, env, node, clk, String)
        J = list(s8.JOINT_NAMES_12DOF)
        R.tick(10, "wait_takeover")                                   # 상태 파일 없음 — 침묵이어야 한다
        motors = {str(s8.CAN_ID_BY_JOINT[j]): {"deg": mid_soft_motor_deg(s8, j), "kp": 150.0, "kd": 5.0} for j in J}
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"ts": clk.t, "motors": motors}, f)
        R.tick(25, "takeover_hold")
        R.cmd({"command": s8.COMMAND_SET_BASELINE_ALL})
        R.tick(3, "after_baseline")
        R.cmd({"command": s8.COMMAND_ARM_ALL})
        for k in range(120):                                          # 브리지 형식 배치 목표(작은 사인)
            R.targets({j: round(1.5 * math.sin(0.11 * k + 0.5 * i), 4) for i, j in enumerate(J)})
            R.tick(1, "armed")
        R.targets({j: (12.0 if i % 2 == 0 else -9.0) for i, j in enumerate(J)}, "targets_big_step")   # 슬루 4°/틱 초과
        R.tick(6, "slew")
        R.targets({J[0]: 400.0, J[1]: -400.0, "no_such_joint": 1.0, J[2]: float("nan")}, "targets_out_of_limit")
        R.tick(3, "after_reject")
        R.cmd({"command": s8.COMMAND_SET_JOINT_TARGET, "joint": J[3], "target_deg": 2.0}, "single_target")
        R.tick(3, "after_single")
        R.cmd({"command": s8.COMMAND_STOP_ALL})
        R.tick(10, "frozen")
        for k in range(20):
            R.targets({j: round(-1.0 * math.sin(0.2 * k + i), 4) for i, j in enumerate(J)})
            R.tick(1, "resumed")
        R.cmd("{bad json", "bad_json")
        R.cmd({"command": s8.COMMAND_CHECK_CAN}, "check_can_while_armed")
        R.tick(2, "after_check")
        R.cmd({"command": s8.COMMAND_DISARM_ALL})
        R.tick(10, "disarmed")
        env["steps"] = R.steps
    return driver


def drive_plain_start(s8, env, clk, String, state_file):
    def driver(node):
        R = Rig(s8, env, node, clk, String)
        J = list(s8.JOINT_NAMES_12DOF)
        R.tick(40, "readonly_poll")                                   # 인수 대기 없음: 읽기 폴링으로 피드백 수집
        R.cmd({"command": s8.COMMAND_SET_BASELINE_ALL})
        R.cmd({"command": s8.COMMAND_ARM_ALL})
        for k in range(60):
            R.targets({j: round(2.0 * math.sin(0.09 * k + i), 4) for i, j in enumerate(J)})
            R.tick(1, "armed")
        env["bus"].reply = False                                      # 피드백 두절: 3 s 뒤 램프 동결(같은 목표 재송신) → STALE_DISARM_SEC 뒤 자동 무장 해제
        n_gap = int((float(s8.STALE_DISARM_SEC) + 2.0) / DT)
        for k in range(n_gap):
            if k % 10 == 0:
                R.targets({j: 1.0 for j in J}, "targets_no_fb")
            R.tick(1, "no_feedback")
        env["bus"].reply = True
        R.tick(20, "feedback_back")
        R.cmd({"command": s8.COMMAND_SET_BASELINE_ALL}, "baseline_after_gap")
        R.cmd({"command": s8.COMMAND_ARM_ALL}, "arm_after_gap")
        R.tick(5, "after_rearm_attempt")
        R.cmd({"command": s8.COMMAND_DISARM_ALL})
        env["steps"] = R.steps
    return driver


SCENARIOS = [("console_flow", ["--can-channel", "can1", "--wait-takeover"], drive_console_flow),
             ("plain_start", ["--can-channel", "can1"], drive_plain_start)]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def do_dump(repo, out_path):
    repo = os.path.abspath(repo)
    src = os.path.join(repo, "jetson", "scripts", "stage8_12axis_mit_control_node.py")
    if not os.path.isfile(src):
        print(f"stage8 없음: {repo}", file=sys.stderr)
        return 2
    sys.path.insert(0, repo)
    env = {"frames": [], "status": []}
    String = install_fakes(env)
    s8 = importlib.import_module("jetson.scripts.stage8_12axis_mit_control_node")
    assert os.path.abspath(s8.__file__) == src, (s8.__file__, src)
    assert getattr(sys.modules["can"], "__fake__", False), "가짜 can 모듈이 아님 — 중단"
    runs = []
    tmp = tempfile.mkdtemp(prefix="stage8cmp_")
    for name, argv, mk in SCENARIOS:
        clk = FakeClock()
        env.clear()
        env.update({"frames": [], "status": [], "fc": list(s8.FC_READONLY_REQUEST_DATA)})
        env["pos0"] = {}
        for j in s8.JOINT_NAMES_12DOF:
            fr = s8.build_12axis_mit_preview(j, mid_soft_motor_deg(s8, j)).frame_bytes
            env["pos0"][int(s8.CAN_ID_BY_JOINT[j])] = (fr[0], fr[1])
        state_file = os.path.join(tmp, f"{name}_handoff_state.json")
        s8.TAKEOVER_STATE_FILE = state_file                      # 실제 /tmp/hang_handoff_state.json 을 건드리지 않는다
        s8.time = clk                                            # 모듈의 time.time() → 가짜 시계
        s8.Stage8TwelveAxisMitControlCore.__init__.__defaults__ = (clk.time,)    # 노드가 만드는 코어의 now_fn
        env["driver"] = mk(s8, env, clk, String, state_file)
        old_argv = sys.argv
        sys.argv = ["stage8_12axis_mit_control_node.py"] + argv
        err = None
        try:
            rc = s8.main()
        except Exception as e:  # noqa: BLE001
            rc, err = -1, f"{type(e).__name__}: {e}"
        finally:
            sys.argv = old_argv
        runs.append({"name": name, "argv": argv, "rc": rc, "error": err, "steps": env.get("steps", [])})
    meta = {"repo": repo, "stage8_md5": _md5(src), "python": sys.version.split()[0],
            "latency_options": (s8.latency_options_status() if hasattr(s8, "latency_options_status") else None)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "runs": runs}, f, ensure_ascii=False)
    nf = sum(len(s["frames"]) for r in runs for s in r["steps"])
    ns = sum(len(s["status"]) for r in runs for s in r["steps"])
    bad = [f"{r['name']}: {r['error'] or 'rc ' + str(r['rc'])}" for r in runs if r["rc"] != 0]
    print(f"덤프 {out_path}: stage8 {meta['stage8_md5'][:8]} · 대본 {len(runs)} · 단계 {sum(len(r['steps']) for r in runs)} · 프레임 {nf} · status {ns}"
          + (f" · ★실행 오류 {bad}" if bad else ""))
    return 0 if not bad else 2


def do_compare(pa, pb):
    A, B = json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
    print(f"A: {A['meta']['repo']} stage8 {A['meta']['stage8_md5'][:8]} 지연 옵션 {A['meta']['latency_options']}")
    print(f"B: {B['meta']['repo']} stage8 {B['meta']['stage8_md5'][:8]} 지연 옵션 {B['meta']['latency_options']}")
    ok = True
    only_a, only_b = set(), set()
    for ra, rb in zip(A["runs"], B["runs"]):
        name = ra["name"]
        if ra["rc"] != 0 or rb["rc"] != 0:
            ok = False
            print(f"[다름] {name}: 실행 오류 A {ra['error'] or ra['rc']} / B {rb['error'] or rb['rc']}")
            continue
        sa, sb = ra["steps"], rb["steps"]
        nfa = sum(len(s["frames"]) for s in sa)
        nfb = sum(len(s["frames"]) for s in sb)
        fdiff = sdiff = None
        n_status = 0
        if len(sa) != len(sb):
            ok = False
            print(f"[다름] {name}: 단계 수 {len(sa)} vs {len(sb)}")
            continue
        for i, (x, y) in enumerate(zip(sa, sb)):
            if x["frames"] != y["frames"] and fdiff is None:
                fdiff = f"단계 #{i}({x['label']}): A {x['frames'][:3]}… {len(x['frames'])}개 / B {y['frames'][:3]}… {len(y['frames'])}개"
            if len(x["status"]) != len(y["status"]) and sdiff is None:
                sdiff = f"단계 #{i}({x['label']}): status 발행 수 {len(x['status'])} vs {len(y['status'])}"
            for du, dv in zip(x["status"], y["status"]):          # 키 → 값 해시
                only_a |= set(du) - set(dv)
                only_b |= set(dv) - set(du)
                n_status += 1
                bad = [k for k in set(du) & set(dv) if du[k] != dv[k]]
                if bad and sdiff is None:
                    sdiff = f"단계 #{i}({x['label']}): status 공통 키 {sorted(bad)[:8]} 의 값이 다름"
        labels = {}
        for s in sa:
            labels[s["label"]] = labels.get(s["label"], 0) + len(s["frames"])
        if fdiff or sdiff:
            ok = False
            print(f"[다름] {name}: " + " | ".join(x for x in (fdiff, sdiff) if x))
        else:
            print(f"[같음] {name} {' '.join(ra['argv'])}: 프레임 {nfa}/{nfb} 개 비트 동일 · status {n_status} 건 공통 키 동일 · 단계별 프레임 수 "
                  + ", ".join(f"{k} {v}" for k, v in labels.items() if v or k.startswith("wait")))
    print(f"status 에서 A 에만 있는 키: {sorted(only_a) or '없음'}")
    print(f"status 에서 B 에만 있는 키: {sorted(only_b) or '없음'}")
    print("결과: " + ("프레임·공통 status 전부 같음" if ok else "다름 있음"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--dump")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args()
    if a.compare:
        return do_compare(*a.compare)
    if a.dump:
        return do_dump(a.repo, os.path.abspath(a.dump))
    ap.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
