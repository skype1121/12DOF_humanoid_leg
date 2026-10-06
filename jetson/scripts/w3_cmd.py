#!/usr/bin/env python3
"""W3 세션 명령 도구 (1007 2 차 대비책 W10 — 콘솔 W3 통합(D4 a) 전 대체 경로, **실물 미검증·젯슨 미배포**)

`ros2 topic pub` 손 발행의 함정(같은 seq 반복·`-r` 반복 발행·응답 무시·재전송)을 막는 한 번 발행 도구.
  - seq = max(파일의 마지막 seq + 1, 현재 ms) — 파일 잠금(flock)·원자적 쓰기. 브리지를 다시 띄워도 단조 증가.
  - req_id = "cli-<호스트>-<pid>-<seq>". 명령은 **한 번만** 발행하고 같은 req_id 의 응답(/humanoid/rl_bridge_ack)을 기다린다.
  - 응답 없음 → 재전송하지 않고 종료 코드 3("상태 확인 뒤 판단"). 켜기는 절대 자동 재시도하지 않는다.
  - 켜기 명령 값은 로컬에서 w3_deploy.validate_w3_cmd(학습 범위)로 먼저 검사(최종 판정은 브리지 감독기).
  - estop 은 이 도구에 없다 — 비상정지는 콘솔 버튼(브리지 estop + stage8 STOP_ALL 동시)을 쓴다.
사용(젯슨, ROS 환경):
  python3 jetson/scripts/w3_cmd.py on [--vx 0.04 | --vy 0.03 | --wz 0.1] [--no-heading-loop]
  python3 jetson/scripts/w3_cmd.py off | stop | rearm
  python3 jetson/scripts/w3_cmd.py status          # 디버그 토픽 한 건: w3·w3_sup·w3_mon
  python3 jetson/scripts/w3_cmd.py --selftest      # ROS 불필요
종료 코드: 0 수락 · 1 거부 · 2 인자/로컬 검사 거부 · 3 응답 없음(타임아웃) · 4 ROS 오류.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from rl_walking.deploy import w3_deploy as _W3  # noqa: E402

CMD_TOPIC = "/humanoid/rl_bridge_command"      # rl_bridge_node.BRIDGE_CMD_TOPIC
ACK_TOPIC = "/humanoid/rl_bridge_ack"          # rl_bridge_node.BRIDGE_ACK_TOPIC
DBG_TOPIC = "/humanoid/rl_bridge_debug"        # rl_bridge_node.BRIDGE_DEBUG_TOPIC
SEQ_FILE = os.path.expanduser("~/.w3_cmd_seq")


def next_seq(path=SEQ_FILE, now_ms=None):
    """파일 잠금 아래 seq = max(마지막 + 1, 현재 ms) 를 원자적으로 기록하고 돌려준다."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        last = 0
        try:
            with open(path) as f:
                last = int(f.read().strip() or 0)
        except (FileNotFoundError, ValueError):
            last = 0
        seq = max(last + 1, now_ms)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(str(seq))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return seq


def build(action, seq, *, vx=0.0, vy=0.0, wz=0.0, heading_loop=True, limits=_W3.W3S3D_CMD_LIMITS, host=None, pid=None):
    """→ (명령 dict, req_id). on 은 학습 범위·단일 축 로컬 검사(위반 ValueError)."""
    req_id = f"cli-{host or socket.gethostname()}-{pid or os.getpid()}-{seq}"[:64]
    base = {"req_id": req_id, "seq": int(seq)}
    if action == "on":
        d = {"vx": float(vx), "vy": float(vy), "wz": float(wz)}
        _W3.validate_w3_cmd(d, limits)
        return dict(base, cmd="step", enable=True, heading_loop=bool(heading_loop), **d), req_id
    if action == "off":
        return dict(base, cmd="step", enable=False), req_id
    if action in ("stop", "rearm"):
        return dict(base, cmd=action), req_id
    raise ValueError(f"알 수 없는 동작 {action!r} — on|off|stop|rearm (estop 은 콘솔 버튼)")


def match_ack(text, req_id):
    """응답 문자열 → 같은 req_id 면 dict, 아니면 None."""
    try:
        a = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return a if isinstance(a, dict) and a.get("req_id") == req_id else None


def _selftest():
    import tempfile
    d = tempfile.mkdtemp(prefix="w3cmd_")
    p = os.path.join(d, "seq")
    s1 = next_seq(p, now_ms=1000)
    s2 = next_seq(p, now_ms=1000)          # 같은 ms → +1
    s3 = next_seq(p, now_ms=500)           # 시계가 뒤로 가도 단조
    s4 = next_seq(p, now_ms=5000)
    assert (s1, s2, s3, s4) == (1000, 1001, 1002, 5000), (s1, s2, s3, s4)
    c, rid = build("on", 7, vx=0.04, host="h", pid=1)
    assert c == {"req_id": "cli-h-1-7", "seq": 7, "cmd": "step", "enable": True, "heading_loop": True, "vx": 0.04, "vy": 0.0, "wz": 0.0}, c
    for bad in ({"vx": 0.2}, {"vx": 0.04, "wz": 0.1}, {"vy": 0.01}, {"vx": -0.08}):
        try:
            build("on", 8, **bad)
        except ValueError:
            continue
        raise AssertionError(f"로컬 검사 통과: {bad}")
    assert build("off", 9)[0]["enable"] is False and build("stop", 10)[0]["cmd"] == "stop" and build("rearm", 11)[0]["cmd"] == "rearm"
    try:
        build("estop", 12)
        raise AssertionError("estop 이 이 도구에서 발행됨")
    except ValueError:
        pass
    assert match_ack(json.dumps({"req_id": "x", "accepted": True}), "x")["accepted"] is True
    assert match_ack(json.dumps({"req_id": "y"}), "x") is None and match_ack("{bad", "x") is None
    # 브리지 감독기와 계약 일치: 도구가 만든 켜기·rearm 을 실제 W3Supervisor 가 형식 오류 없이 처리
    S = _W3.W3Supervisor(cmd_limits=_W3.W3S3D_CMD_LIMITS)
    c, _ = build("on", next_seq(p), vx=0.04)
    r = S.handle(c, 0.0, active=False, warmed=True, blend_done=True, ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
    assert not r["accepted"] and "정책 미실행" in r["why"], r          # 형식 통과·선행조건에서 거부(= 계약 일치)
    c2, _ = build("on", c["seq"], vx=0.04)
    r2 = S.handle(c2, 0.1, active=True, warmed=True, blend_done=True, ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
    assert not r2["accepted"] and "중복" in r2["why"], r2               # 같은 seq 재사용은 브리지가 거부
    print("[PASS] w3_cmd: seq 단조(같은 ms·시계 역행)·켜기 로컬 검사 4 종 거부·off/stop/rearm·estop 없음·응답 매칭·감독기 계약 일치")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", choices=("on", "off", "stop", "rearm", "status"))
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--no-heading-loop", action="store_true")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.action is None:
        ap.print_usage()
        return 2
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as e:
        print(f"ROS 없음: {e}", file=sys.stderr)
        return 4
    rclpy.init()
    node = Node("w3_cmd_cli")
    got = {}
    try:
        if a.action == "status":
            def on_dbg(m):
                try:
                    d = json.loads(m.data)
                except json.JSONDecodeError:
                    return
                got["d"] = {k: d.get(k) for k in ("checkpoint", "cmd", "w3", "w3_cmd", "w3_sup", "w3_mon", "stop_reason", "guard_mode")}
            node.create_subscription(String, DBG_TOPIC, on_dbg, 10)
            t_end = time.time() + a.timeout
            while "d" not in got and time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
            if "d" not in got:
                print("디버그 토픽 무소식 — 브리지 미실행 또는 DDS 정지 의심(0919 서명 확인)")
                return 3
            print(json.dumps(got["d"], ensure_ascii=False, indent=1))
            return 0
        try:
            seq = next_seq()
            cmd, rid = build(a.action, seq, vx=a.vx, vy=a.vy, wz=a.wz, heading_loop=not a.no_heading_loop)
        except ValueError as e:
            print(f"[로컬 거부] {e}")
            return 2

        def on_ack(m):
            r = match_ack(m.data, rid)
            if r is not None:
                got["a"] = r
        node.create_subscription(String, ACK_TOPIC, on_ack, 10)
        pub = node.create_publisher(String, CMD_TOPIC, 10)
        t_wait = time.time() + 0.5                       # 구독 연결 대기(발행은 한 번뿐이라 연결 전 발행 손실을 피한다)
        while pub.get_subscription_count() < 1 and time.time() < t_wait:
            rclpy.spin_once(node, timeout_sec=0.05)
        pub.publish(String(data=json.dumps(cmd)))
        print(f"발행 1 회: {json.dumps(cmd, ensure_ascii=False)}")
        t_end = time.time() + a.timeout
        while "a" not in got and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        if "a" not in got:
            print(f"응답 없음({a.timeout:g} s) — 같은 명령을 재전송하지 말 것. `w3_cmd.py status` 로 상태 확인 뒤 판단")
            return 3
        r = got["a"]
        print(f"{'수락' if r.get('accepted') else '거부'}: {r.get('why')}  (action {r.get('action')}, 래치 {(r.get('sup') or {}).get('stop_latched')})")
        return 0 if r.get("accepted") else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
