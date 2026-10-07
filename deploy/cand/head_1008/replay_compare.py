#!/usr/bin/env python3
"""젯슨 판 ↔ 새 판 브리지 발행 비교 — 같은 입력열을 두 저장소 트리의 **실제 브리지 노드 코드**에 넣어 발행·블랙박스가 같은지 본다.

목적: "기존 세션(서기 G21a·G12, 회복 kr·krh)이 새 판에서도 종전과 같은 명령을 낸다"를 selftest 의 기준 커밋이 아니라
**젯슨에 지금 있는 판**을 기준으로 직접 확인한다(저장소 selftest 의 비트 비교 기준은 68493c11·95946e41·b79370ca 로 젯슨 판보다 뒤다).

  # 트리마다 한 번(그 트리의 jetson/scripts/rl_bridge_node.py·rl_walking/deploy·정책 파일·하드웨어맵을 쓴다)
  python3 replay_compare.py --repo <트리 A> --dump a.json
  python3 replay_compare.py --repo <트리 B> --dump b.json
  python3 replay_compare.py --compare a.json b.json        # 종료 코드 0 = 전부 같음, 1 = 다름, 2 = 실행 불가
  python3 replay_compare.py --compare a.json b.json --new-bb-keys 키1,키2   # 새 판이 블랙박스 줄에만 더한 기록 필드를 빼고 비교(발행은 전부, A 에도 있는 키면 거부)

방법: rclpy·std_msgs 를 가짜 모듈로 바꾸고(ROS·CAN·로봇 불필요) 가짜 시계로 rl_bridge_node.main() 을 그대로 돌린다. 정책은 그 트리의
**실제 정책 파일**(그 트리의 _make_runner — onnxruntime 또는 torch), 하드웨어맵도 그 트리의 것이다. 입력열(관절·IMU·접촉·명령)은 결정적 합성값.
비교 대상: 모든 토픽 발행(명령 토픽의 목표각·STOP_ALL, 디버그 토픽의 관측 320·원시 행동·가드·C5 모드…)과 블랙박스 줄.
경우: 키 × 대본(서기/회복 펄스/공중 시작) × live·dry × 가드 모드(맵 그대로 = 젯슨은 data / safe 강제 = 가드 정지 경로).
      + 대본 err(g21a·krh): stage8 상태에 모터 오류 바이트·warnings·피드백 나이(정지 기준 아래·위)를 실어, 새 판의 오류 기록이 발행을 바꾸지 않는지 본다.
**개루프 계약 비교다 — 폐루프 성능이나 실물 안전을 증명하지 않는다**(CLAUDE.md 측정과 주장 6)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time as _real_time
import types

KEYS_BASIC = ["stand_crouch_g21a", "stand_crouch_g12", "stand_crouch_g12_ema0", "stand_crouch_g11a", "stand_v18b"]
KEYS_C5 = ["stand_c5_g21a_g24ra", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra"]
T0 = 2_000_000_000.0
FAKE_PID = 4242


class FakeClock:
    def __init__(self, t0=T0):
        self.t = t0

    def time(self):
        return self.t

    def __getattr__(self, name):          # strftime·sleep·monotonic 등은 진짜 time 으로
        return getattr(_real_time, name)


class _Msg:
    def __init__(self, data=""):
        self.data = data


class _Pub:
    def __init__(self, topic, log):
        self.topic, self._log = topic, log

    def publish(self, m):
        self._log.append([self.topic, m.data])

    def get_subscription_count(self):
        return 1


class _Log:
    def __init__(self, sink):
        self._sink = sink

    def __getattr__(self, name):
        return lambda s: self._sink.append([name, str(s)])


def _make_node_class(pub_log, log_lines):
    class _Node:
        def __init__(self, name):
            self._subs, self._pubs, self._timers = {}, {}, []
            self._log = _Log(log_lines)

        def create_subscription(self, typ, topic, cb, depth):
            self._subs[topic] = cb

        def create_publisher(self, typ, topic, depth):
            p = _Pub(topic, pub_log)
            self._pubs[topic] = p
            return p

        def create_timer(self, dt, cb):
            self._timers.append(cb)

        def get_logger(self):
            return self._log

        def destroy_node(self):
            pass
    return _Node


def _install_fake_ros(driver, pub_log, log_lines):
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = driver
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _make_node_class(pub_log, log_lines)
    rclpy.node = node_mod
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _Msg
    std.msg = std_msg
    sys.modules.update({"rclpy": rclpy, "rclpy.node": node_mod, "std_msgs": std, "std_msgs.msg": std_msg})


class Feed:
    """센서 입력 주입 + 틱 구동(두 판 공통 메시지 형식: stage8 status·IMU·발)."""

    def __init__(self, node, b, clk):
        self.n, self.b, self.clk = node, b, clk
        self.pose = b.CKPT_DEFAULT_POSE.get(node.core.ckpt_name) or [float(v) for v in b.DEFAULT_POSE_RAD]
        self.k = 0
        self.contact = (True, True)
        self.pitch = 0.0
        self.roll = 0.0
        self.gy = 0.02
        self.amp = 0.3          # 관절 흔들림 [deg]
        self.s8_full = False    # True 면 stage8 상태의 오류 필드(feedback_error_raw·feedback_error_ok·warnings·errors)까지 싣는다(err 대본)
        self.err = {}           # {관절: 0 아닌 오류 바이트}
        self.warn = []          # stage8 warnings
        self.age = {}           # {관절: feedback_age_sec}

    def feed(self):
        b, now = self.b, self.clk.t
        self.k += 1
        js = {}
        for i, (j, v) in enumerate(zip(b.JOINT_ORDER, self.pose)):
            m = self.n.frames.joint_rad_to_motor_deg(j, v) + self.amp * math.sin(0.07 * self.k + i)
            js[j] = {"baseline_deg": m - 0.3, "latest_actual_deg": m, "feedback_age_sec": float(self.age.get(j, 0.0)),
                     "commanded_relative_deg": 0.1, "feedback_torque_est": 0.5 * i, "feedback_temperature_raw": 30 + i}
            if self.s8_full:
                js[j]["feedback_error_raw"] = int(self.err.get(j, 0))
                js[j]["feedback_error_ok"] = not self.err.get(j, 0)
        st = {"armed": True, "joints": js}
        if self.s8_full:
            st["warnings"] = list(self.warn)
            st["errors"] = []
        self.n._on_status(_Msg(json.dumps(st)))
        p, r = self.pitch, self.roll
        quat = [math.cos(p / 2) * math.cos(r / 2), math.cos(p / 2) * math.sin(r / 2),
                math.sin(p / 2) * math.cos(r / 2), -math.sin(p / 2) * math.sin(r / 2)]
        self.n._on_imu(_Msg(json.dumps({"gyro_rad_s": [0.03 * math.sin(0.1 * self.k), self.gy, 0.01 * math.cos(0.05 * self.k)],
                                        "quat_wxyz": quat, "timestamp": now})))
        cl, cr = self.contact
        self.n._on_foot(_Msg(json.dumps({"left_n": 40.0 * cl, "right_n": 40.0 * cr, "left_contact": cl,
                                         "right_contact": cr, "timestamp": now})))

    def tick(self, n=1):
        for _ in range(n):
            self.feed()
            self.n._tick()
            self.clk.t += self.b.TICK_SEC

    def cmd(self, d):
        self.n._on_cmd(_Msg(json.dumps(d) if not isinstance(d, str) else d))


class _Osc:
    def __init__(self, why):
        self.why = why

    def update(self, now, q, v):
        return self.why


def drv_basic(b, clk):
    """서기 세션: stand → 정상 서기 → 기울기·각속도 변화 → 모르는 명령(step·turn·rearm) → stop → 재시작 → 리프트 가드 → 발진 가드 →
    과속 → estop·불량 JSON."""
    def driver(node):
        F = Feed(node, b, clk)
        F.tick(3)
        F.cmd({"cmd": "stand"}); F.tick(120)
        F.pitch = math.radians(2.0); F.gy = 0.15; F.tick(30)
        F.roll = math.radians(-1.5); F.pitch = math.radians(-1.0); F.gy = 0.02; F.tick(60)
        F.pitch = F.roll = 0.0
        F.cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 1}); F.tick(5)
        F.cmd({"cmd": "turn", "heading_target_deg": 10}); F.cmd({"cmd": "rearm", "req_id": "a", "seq": 2}); F.tick(5)
        F.cmd({"cmd": "stop"}); F.tick(10)
        F.cmd({"cmd": "stand"}); F.tick(40)
        F.contact = (True, False); F.tick(15); F.contact = (True, True); F.tick(10)      # 한발 접촉(가드 미발동 구간)
        F.contact = (False, False); F.tick(int(1.2 / b.TICK_SEC)); F.contact = (True, True)   # 양발 무접촉 1.2 s → 리프트 가드
        F.cmd({"cmd": "stand"}); F.tick(20)
        real = node.osc
        node.osc = _Osc("저주파 반전 시험"); F.tick(3); node.osc = real                    # 발진 가드
        F.cmd({"cmd": "stand"}); F.tick(20)
        node.osc = _Osc("과속 13 rad/s 시험"); F.tick(3); node.osc = real                  # 과속(하드)
        F.cmd({"cmd": "stand"}); F.tick(20)
        F.cmd({"cmd": "estop"}); F.cmd('"estop"'); F.cmd("{bad"); F.tick(5)
    return driver


def drv_c5(b, clk):
    """회복 세션: stand → 기준 자세 고정 → 기울기 유지(회복 모드 체류) → 재시작 → 앞 기울기 펄스(켜짐·cue·조기 복귀) →
    뒤 기울기 펄스 → 옆 기울기 펄스 + 한발 접촉 → stop → 리프트 가드 → estop."""
    def driver(node):
        F = Feed(node, b, clk)
        F.tick(3)
        F.cmd({"cmd": "stand"}); F.tick(160)
        F.pitch = math.radians(4.5); F.tick(80)
        F.pitch = 0.0
        F.cmd({"cmd": "stand"}); F.tick(160)
        F.pitch = math.radians(6.5); F.gy = 0.7; F.tick(1); F.gy = 0.02; F.tick(9)
        F.pitch = 0.0; F.tick(120)
        F.pitch = math.radians(-6.0); F.gy = -0.6; F.tick(1); F.gy = 0.02; F.tick(12)
        F.pitch = 0.0; F.tick(120)
        F.roll = math.radians(5.0); F.contact = (True, False); F.tick(8); F.contact = (True, True); F.tick(6)
        F.roll = 0.0; F.tick(150)
        F.cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 1}); F.cmd({"cmd": "stop"}); F.tick(10)
        F.contact = (False, False); F.tick(int(1.2 / b.TICK_SEC)); F.contact = (True, True)
        F.cmd({"cmd": "stand"}); F.tick(20)
        F.cmd({"cmd": "estop"}); F.tick(5)
    return driver


def drv_hang(b, clk):
    """공중 시작(hang_arm — 0908 부터 있던 경로): 무접촉 3 s → 한발 → 양발 착지 → 서기 → 재들림 → 재착지 → estop."""
    def driver(node):
        F = Feed(node, b, clk)
        F.contact = (False, False)
        F.tick(3)
        F.cmd({"cmd": "stand", "hang_arm": True}); F.tick(150)
        F.contact = (True, False); F.tick(10)
        F.contact = (True, True); F.tick(120)
        F.contact = (False, False); F.tick(30)
        F.contact = (True, True); F.tick(60)
        F.cmd({"cmd": "estop"}); F.tick(5)
    return driver


def drv_err(b, clk):
    """모터 오류 기록(1007) 경로: 서기 중 stage8 상태에 오류 바이트·warnings 가 실리고 피드백 나이가 커진다(정지 기준 아래 → 위).
    기준 판(젯슨 판)은 이 필드를 읽지 않는다 — 새 판이 그것을 블랙박스에 적으면서 발행(명령·디버그)과 가드 동작을 바꾸지 않는지 본다."""
    def driver(node):
        F = Feed(node, b, clk)
        F.s8_full = True
        AL, AR = "left_ankle_r_joint", "right_ankle_r_joint"
        assert AL in b.JOINT_ORDER and AR in b.JOINT_ORDER
        F.tick(3)
        F.cmd({"cmd": "stand"}); F.tick(100)
        F.err = {AL: 4}; F.warn = [f"feedback_error:{AL}:4"]; F.tick(3)                       # 오류 바이트 3 틱
        F.err = {}; F.warn = []; F.tick(20)
        F.err = {AL: 4, AR: 2}; F.warn = ["b", "a"]; F.tick(1); F.err = {}; F.warn = []; F.tick(20)   # 두 관절 1 틱
        F.age = {AL: 0.06, AR: 0.12}; F.tick(10); F.age = {}; F.tick(20)                      # 피드백 나이: 기록 하한(0.05 s) 이상·정지 기준 아래
        F.err = {AR: 7}; F.age = {AR: 0.35, AL: 0.35}; F.warn = ["feedback_stale:" + AR]; F.tick(15)   # 정지 기준 초과(두 판이 같은 가드를 타야 한다)
        F.err = {}; F.age = {}; F.warn = []; F.tick(10)
        F.cmd({"cmd": "stand"}); F.tick(40)
        F.cmd({"cmd": "estop"}); F.tick(5)
    return driver


SCENARIOS = {"basic": drv_basic, "c5": drv_c5, "hang": drv_hang, "err": drv_err}
KEYS_ERR = ("stand_crouch_g21a", "stand_c5krh_g21a_g24ra")


def load_bridge(repo):
    path = os.path.join(repo, "jetson", "scripts", "rl_bridge_node.py")
    spec = importlib.util.spec_from_file_location("rl_bridge_node_replay", path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = ["rl_bridge_node.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def cases(b):
    out = []
    for k in KEYS_BASIC:
        if k not in b.CHECKPOINTS:
            continue
        for live in (True, False):
            for guard in ("map", "safe"):
                if not live and guard == "safe":
                    continue                      # dry 는 한 가지만(발행 없음 경로 확인)
                out.append((k, "basic", live, guard))
    out.append(("stand_crouch_g21a", "hang", True, "map"))
    out.append(("stand_crouch_g21a", "hang", True, "safe"))
    for k in KEYS_C5:
        if k not in b.CHECKPOINTS:
            continue
        for guard in ("map", "safe"):
            out.append((k, "c5", True, guard))
    for k in KEYS_ERR:
        if k in b.CHECKPOINTS:
            for guard in ("map", "safe"):
                out.append((k, "err", True, guard))
    return out


def policy_files_ok(b, key):
    dirs = [b.CHECKPOINTS[key]]
    pair = getattr(b, "C5_PAIRS", {}).get(key)
    if pair:
        dirs = [b.CHECKPOINTS[pair[0]], pair[1]]
    return all(os.path.isfile(os.path.join(d, "policy.onnx")) or os.path.isfile(os.path.join(d, "policy.pt")) for d in dirs)


def run_case(b, repo, key, scen, live, guard):
    pub_log, log_lines = [], []
    clk = FakeClock()
    b.time = clk
    if not hasattr(b, "_rc_real_load"):
        b._rc_real_load = b.load_frame_map
    if guard == "safe":
        def load_safe():
            f = b._rc_real_load()
            f.guard_mode = "safe"
            return f
        b.load_frame_map = load_safe
    else:
        b.load_frame_map = b._rc_real_load
    argv = (["--live", "--force-frame", "--force-gains"] if live else ["--force-frame"]) + ["--checkpoint", key]
    _install_fake_ros(SCENARIOS[scen](b, clk), pub_log, log_lines)
    old_argv, old_pid = sys.argv, os.getpid
    home = tempfile.mkdtemp(prefix="replay_home_")
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = home                      # 블랙박스(~/logs)를 임시 폴더로
    os.getpid = lambda: FAKE_PID                   # 명령 출처(src)에 pid 가 들어간다 — 두 프로세스 비교를 위해 고정
    sys.argv = ["rl_bridge_node.py"] + argv
    err = None
    try:
        rc = b.main()
    except Exception as e:  # noqa: BLE001 — 한 경우의 예외를 기록하고 다음으로
        rc, err = -1, f"{type(e).__name__}: {e}"
    finally:
        sys.argv, os.getpid = old_argv, old_pid
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
    bb = []
    logd = os.path.join(home, "logs")
    if os.path.isdir(logd):
        for fn in sorted(os.listdir(logd)):
            with open(os.path.join(logd, fn), encoding="utf-8") as f:
                bb += [ln.rstrip("\n") for ln in f]
    norm = lambda s: s.replace(repo, "<REPO>").replace(home, "<HOME>")  # noqa: E731
    return {"key": key, "scenario": scen, "live": live, "guard": guard, "argv": argv, "rc": rc, "error": err,
            "pubs": [[t, norm(d)] for t, d in pub_log], "bb": [norm(x) for x in bb],
            "log": [[lv, norm(m)] for lv, m in log_lines]}


def do_dump(repo, out_path):
    repo = os.path.abspath(repo)
    if not os.path.isfile(os.path.join(repo, "jetson", "scripts", "rl_bridge_node.py")):
        print(f"브리지 없음: {repo}", file=sys.stderr)
        return 2
    os.chdir(repo)
    b = load_bridge(repo)
    import numpy as np
    runs, skipped = [], []
    for key, scen, live, guard in cases(b):
        if not policy_files_ok(b, key):
            skipped.append(key)
            continue
        runs.append(run_case(b, repo, key, scen, live, guard))
    backend = None
    try:      # 추론 백엔드 이름(onnx|torch) — 기록용
        backend = getattr(b._make_runner("stand_crouch_g21a"), "backend", None)
    except Exception:  # noqa: BLE001
        pass
    meta = {"repo": repo, "numpy": np.__version__, "python": sys.version.split()[0], "backend": backend,
            "bridge_md5": _md5(os.path.join(repo, "jetson", "scripts", "rl_bridge_node.py")),
            "skipped_no_policy": sorted(set(skipped))}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "runs": runs}, f, ensure_ascii=False)
    n_pub = sum(len(r["pubs"]) for r in runs)
    bad = [f"{r['key']}/{r['scenario']}: {r['error'] or 'rc ' + str(r['rc'])}" for r in runs if r["rc"] != 0]
    print(f"덤프 {out_path}: 경우 {len(runs)} · 발행 {n_pub} 건 · 블랙박스 {sum(len(r['bb']) for r in runs)} 줄 · 브리지 {meta['bridge_md5'][:8]} · "
          f"numpy {meta['numpy']} · 백엔드 {meta['backend']} · 정책 없음 생략 {meta['skipped_no_policy'] or '없음'}" + (f" · ★실행 오류 {bad}" if bad else ""))
    return 0 if not bad else 2


def _md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _first_diff(a, b):
    """두 JSON 문자열의 첫 차이를 사람이 읽게 요약(둘 다 객체면 키 단위)."""
    try:
        da, db = json.loads(a), json.loads(b)
    except (ValueError, TypeError):
        return f"A {a[:160]!r} / B {b[:160]!r}"
    if isinstance(da, dict) and isinstance(db, dict):
        only_a, only_b = sorted(set(da) - set(db)), sorted(set(db) - set(da))
        diff = [k for k in sorted(set(da) & set(db)) if da[k] != db[k]]
        parts = []
        if only_a:
            parts.append(f"A 에만 있는 키 {only_a}")
        if only_b:
            parts.append(f"B 에만 있는 키 {only_b}")
        for k in diff[:4]:
            parts.append(f"{k}: A {json.dumps(da[k], ensure_ascii=False)[:120]} / B {json.dumps(db[k], ensure_ascii=False)[:120]}")
        if len(diff) > 4:
            parts.append(f"… 다른 키 {len(diff)} 개")
        return " · ".join(parts) or "키·값 같음(문자열 표현만 다름)"
    return f"A {a[:160]!r} / B {b[:160]!r}"


def _strip_keys(s, keys):
    """JSON 객체 문자열에서 최상위 키 몇 개를 뺀 정규형(키가 없으면 원문 그대로)."""
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        return s
    if not isinstance(d, dict) or not (set(d) & keys):
        return s
    return json.dumps({k: v for k, v in d.items() if k not in keys}, ensure_ascii=False)


def _canon_bb(s, new_keys):
    """블랙박스 줄의 정규형 — 새 판이 더한 기록 필드(new_keys)를 뺀 나머지를 같은 방식으로 다시 직렬화한다(키 순서·숫자 형·값 보존).
    돌려주는 것: (정규형 문자열, 그 줄에 있던 new_keys 목록). 객체가 아닌 줄은 원문 그대로."""
    try:
        d = json.loads(s)
    except (ValueError, TypeError):
        return s, []
    if not isinstance(d, dict):
        return s, []
    return json.dumps({k: v for k, v in d.items() if k not in new_keys}, ensure_ascii=False), [k for k in d if k in new_keys]


def do_compare(pa, pb, ignore_keys=(), new_bb_keys=()):
    A, B = json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
    print(f"A: {A['meta']['repo']} 브리지 {A['meta']['bridge_md5'][:8]} numpy {A['meta']['numpy']} 백엔드 {A['meta'].get('backend')}")
    print(f"B: {B['meta']['repo']} 브리지 {B['meta']['bridge_md5'][:8]} numpy {B['meta']['numpy']} 백엔드 {B['meta'].get('backend')}")
    ign = set(k for k in ignore_keys if k)
    if ign:
        # 의도해서 더한 기록 필드를 사람이 확인한 뒤에만 여기에 적는다 — 나머지는 그대로 비트 비교된다.
        print(f"※ 비교에서 뺀 최상위 키(--ignore-keys): {sorted(ign)}")
        for D in (A, B):
            for r in D["runs"]:
                r["pubs"] = [[t, _strip_keys(d, ign)] for t, d in r["pubs"]]
                r["bb"] = [_strip_keys(x, ign) for x in r["bb"]]
    newk, new_note = set(k for k in new_bb_keys if k), ""
    if newk:
        # B(새 판)가 **블랙박스 줄에만** 더한 기록 필드. 발행(명령·디버그 토픽)은 건드리지 않고 전부 비교한다. A(기준 판)에 같은 키가 있으면
        # 새 필드가 아니므로 비교를 거부한다. 뺀 뒤에는 양쪽 줄을 같은 방식으로 다시 직렬화해 나머지 키·값·순서를 비교한다.
        cnt = {"A": {}, "B": {}}
        for tag, D in (("A", A), ("B", B)):
            for r in D["runs"]:
                rows = []
                for x in r["bb"]:
                    c, had = _canon_bb(x, newk)
                    for k in had:
                        cnt[tag][k] = cnt[tag].get(k, 0) + 1
                    rows.append(c)
                r["bb"] = rows
        if cnt["A"]:
            print(f"[오류] --new-bb-keys 의 키가 A(기준 판) 블랙박스에도 있다: {cnt['A']} — 새 필드가 아니다. 목록에서 빼고 다시 볼 것")
            return 1
        new_note = " · 새 판 블랙박스에만 있는 기록 필드(뺀 줄 수): " + (", ".join(f"{k} {cnt['B'].get(k, 0)}" for k in sorted(newk)))
        print(f"※ 새 판이 블랙박스 줄에 더한 기록 필드로 보고 뺀 키(--new-bb-keys, 블랙박스 줄에서만 — 발행은 전부 비교): {sorted(newk)}")
    ia = {(r["key"], r["scenario"], r["live"], r["guard"]): r for r in A["runs"]}
    ib = {(r["key"], r["scenario"], r["live"], r["guard"]): r for r in B["runs"]}
    n_same = n_diff = 0
    for k in sorted(set(ia) | set(ib)):
        name = f"{k[0]} · {k[1]} · {'live' if k[2] else 'dry'} · guard {k[3]}"
        if k not in ia or k not in ib:
            print(f"[생략] {name}: {'A' if k not in ia else 'B'} 에 없음(키 미등록 또는 정책 파일 없음)")
            continue
        ra, rb = ia[k], ib[k]
        if ra["rc"] != 0 or rb["rc"] != 0:
            n_diff += 1
            print(f"[다름] {name}: 실행 오류 A {ra['error'] or ra['rc']} / B {rb['error'] or rb['rc']}")
            continue
        notes = []
        for field, label in (("pubs", "발행"), ("bb", "블랙박스")):
            xa, xb = ra[field], rb[field]
            if xa == xb:
                continue
            i = next((i for i, (x, y) in enumerate(zip(xa, xb)) if x != y), min(len(xa), len(xb)))
            if i < min(len(xa), len(xb)):
                x, y = xa[i], xb[i]
                if field == "pubs":
                    what = f"토픽 {x[0]}" + ("" if x[0] == y[0] else f"→{y[0]}") + " " + _first_diff(x[1], y[1])
                else:
                    what = _first_diff(x, y)
            else:
                what = "한쪽이 더 김"
            n_ne = sum(1 for x, y in zip(xa, xb) if x != y) + abs(len(xa) - len(xb))
            notes.append(f"{label} {len(xa)}/{len(xb)} 건 중 다른 것 {n_ne} — 첫 차이 #{i}: {what}")
        n_stop = sum('"STOP_ALL"' in d for t, d in ra["pubs"])
        if notes:
            n_diff += 1
            print(f"[다름] {name}: " + " | ".join(notes))
        else:
            n_same += 1
            print(f"[같음] {name}: 발행 {len(ra['pubs'])} 건(STOP_ALL {n_stop}) 비트 동일·블랙박스 {len(ra['bb'])} 줄 " + ("새 필드 밖 키·값·순서 동일" if newk else "비트 동일"))
    print(f"결과: 같음 {n_same} · 다름 {n_diff}" + new_note)
    return 0 if n_diff == 0 and n_same > 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd(), help="저장소 트리(jetson/scripts/rl_bridge_node.py 가 있는 곳)")
    ap.add_argument("--dump", help="이 트리의 브리지를 돌려 결과를 JSON 으로 저장")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--ignore-keys", default="", help="비교에서 뺄 최상위 JSON 키(쉼표, 발행·블랙박스 양쪽에서) — 되도록 --new-bb-keys 를 쓸 것")
    ap.add_argument("--new-bb-keys", default="", help="새 판(B)이 블랙박스 줄에만 더한 기록 필드(쉼표). 블랙박스 줄에서만 빼고 발행은 전부 비교, A 에도 있으면 거부")
    a = ap.parse_args()
    if a.compare:
        return do_compare(*a.compare, ignore_keys=a.ignore_keys.split(","), new_bb_keys=a.new_bb_keys.split(","))
    if a.dump:
        return do_dump(a.repo, os.path.abspath(a.dump))
    ap.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
