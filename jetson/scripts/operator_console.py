#!/usr/bin/env python3
"""실기 운영 콘솔 — 실제 상태로 진행하는 기립 시험 조작반.

매 실기마다 터미널 여러 개를 오가며 순서를 외우는 대신, 한 화면에서:
  게이트 → 기립 → 핸드오프 → 무장 → 정책 → 판정 → 회수
를 순서대로 확인하고, **기울기·접촉·제어 상태·기록을 실시간으로**
본다. 폰으로도 열리므로 로봇 옆에서 줄을 잡은 채 조작할 수 있다.

    python3 jetson/scripts/operator_console.py          # http://<젯슨>:8800
    python3 jetson/scripts/operator_console.py --selftest

설계 원칙 (실기 사고 이력에서 나온 것들):
- **비상정지는 항상 눌린다.** 다른 어떤 상태에서도 비활성화되지 않는다.
  estop(브리지 동결) + STOP_ALL(stage8)을 동시에 쏜다.
- **임의 명령 실행 없음.** 액션은 아래 ACTIONS 화이트리스트가 전부다.
  웹에서 들어온 문자열이 셸로 가는 경로는 존재하지 않는다.
- **프로세스 종료는 추적 중인 PID로만.** pkill 금지(프로젝트 규칙 — 자기매칭
  사고 이력). 콘솔이 띄운 프로세스만 콘솔이 내린다.
- **순서 강제.** 선행 조건이 안 되면 서버가 거부한다(버튼 비활성화는 보조).
  ARM 전 베이스라인 12/12, 핸드오프 전 stage8 대기 등.
- **자동 실행 없음.** 콘솔은 아무것도 스스로 시작하지 않는다.
"""
import argparse
import collections
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import tempfile
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .console_state import TrialState, LABELS, EVENTS, CONTROL, finite
except ImportError:
    from console_state import TrialState, LABELS, EVENTS, CONTROL, finite

# --- 경로: 젯슨은 스크립트가 두 디렉토리로 갈려 있다 (0808 실측 확인) ---
CALIB = os.path.expanduser("~/rl_calib")
REPO = os.path.expanduser("~/humanoid_repo")
SCRIPTS = os.path.join(REPO, "jetson", "scripts")
LOGDIR = os.path.expanduser("~/logs")

STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"
BRIDGE_CMD_TOPIC = "/humanoid/rl_bridge_command"
BRIDGE_DEBUG_TOPIC = "/humanoid/rl_bridge_debug"

BODY_WEIGHT_N = 172.9        # 구 로그 필드 호환용. 화면·진행·성공 판정에 사용하지 않는다.
BRIDGE_JOINT_FRESH_SEC = 0.2  # rl_bridge_node.SENSOR_STALE_SEC 와 동일 — '12/12 신선' 판정 기준 (0911 R3)

#: 정책 드롭다운 — 첫 항목이 기본값(7차 데뷔 후보). 브리지 CHECKPOINTS 의 부분집합이어야 한다(selftest ⑧).
#: 0907: stand_v14 제거 — 패키지 GENS(v15~v18b)에 없어 젯슨에 파일이 없다(선택 시 로드 실패). 젯슨 구 자산 정리로
#: walk/march/drv2/march_drv2/march_v3/stand_v3/v5/v6 도 삭제됨 — 젯슨 잔존 = v15~v18b + v7/v8/v9/march_v5b.
CKPT_CHOICES = ["stand_hang_v5", "stand_hang_v6", "stand_v18b", "stand_hang_v4", "stand_hang_v3", "stand_hang_v2", "stand_hang_v1", "stand_v17", "stand_v15", "stand_v18", "stand_v16",
                "stand_v9", "stand_v8", "stand_v7", "march_v5b"]
CKPT_DEFAULT = CKPT_CHOICES[0]


def render_page():
    """PAGE 의 정책 드롭다운을 CKPT_CHOICES 로 채운다 (구판은 정적 목록 stand_v8~ 로 기본값이 legacy stand_v7 였음)."""
    return PAGE.replace("<!--CKPT_OPTIONS-->",
                        "".join(f"<option>{c}</option>" for c in CKPT_CHOICES))
STALE_SEC = 1.5              # 이보다 오래된 텔레메트리는 '끊김'으로 표시
#: can_up.sh 는 sudo가 필요하다. 젯슨 계정 암호(공개본에서 기본값 제거)를 환경변수로
#: 덮어쓸 수 있게 둔다 — 소스에 상수로 박아두지 않기 위함.
SUDO_PW = os.environ.get("JETSON_SUDO_PW", "")


def now():
    return time.time()


# ---------------------------------------------------------------------------
# 세션 레코더 — 모터 전원 이후 전 구간을 하나의 타임라인으로 남긴다
# ---------------------------------------------------------------------------
#: 브리지 블랙박스는 정책이 돌 때만 기록한다. 그래서 게이트·기립·**착지**
#: 구간이 통째로 비어 있었다. 0808 분석에서 "착지 기하가 세션 성패를
#: 지배한다"가 확인됐는데 정작 그 구간 데이터가 없어 매번 기억으로
#: 재구성해야 했다. 이 레코더가 그 공백을 메운다.
SESSION_DIR = os.path.join(LOGDIR, "sessions")
SAMPLE_HZ = 20.0
IDLE_ROTATE_SEC = 1800.0     # 30분 조용하면 다음 조작은 새 세션으로 본다
#: 이만큼 '실제 활동'이 없으면 세션을 자동으로 닫는다. 없으면 테스트로 버튼
#: 한 번 누른 세션이 영원히 열린 채 유휴 상태를 20Hz로 기록한다(0808 실측:
#: 37분·13MB·시간당 21MB). 진짜 실기 기록이 유휴 데이터에 파묻힌다.
IDLE_CLOSE_SEC = 600.0


class SessionLog:
    def __init__(self):
        self.lock = threading.RLock()
        self.name = None
        self.dir = None
        self.f = None
        self.manual = False
        self.t0 = None
        self.last = 0.0
        self.last_active = 0.0      # 마지막 '실제 활동' 시각
        self.n_sample = 0
        self.n_event = 0

    def ensure(self, reason="", manual=False):
        """세션이 없거나 오래 쉬었으면 새로 연다. 조작 시점 = 모터 전원 이후."""
        with self.lock:
            if self.f is not None and (now() - self.last) < IDLE_ROTATE_SEC:
                if manual:
                    self.manual = True      # 수동 선언은 뒤늦게도 존중한다
                return self.name
            self.manual = manual
            self._close_locked()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.name = f"session_{stamp}_{uuid.uuid4().hex[:6]}"
            self.dir = os.path.join(SESSION_DIR, self.name)
            os.makedirs(self.dir, exist_ok=True)
            self.f = open(os.path.join(self.dir, "timeline.jsonl"), "a")
            self.t0 = now()
            self.last = self.t0
            self.last_active = self.t0
            self.n_sample = self.n_event = 0
            self._write({"kind": "session_start", "reason": reason,
                         "wall": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"[세션] 시작 {self.name} ({reason})")
            return self.name

    def _write(self, obj):
        if self.f is None:
            return
        obj["t"] = round(now() - self.t0, 3)
        obj["ts"] = round(now(), 3)
        try:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.f.flush()
            self.last = now()
        except OSError:
            try:
                self.f.close()
            except OSError:
                pass
            self.f = None

    def event(self, kind, **kw):
        with self.lock:
            if self.f is None:
                return
            kw["kind"] = kind
            self._write(kw)
            self.n_event += 1
            self.last_active = now()      # 조작은 언제나 '활동'

    def sample(self, view):
        """텔레메트리 1샘플 — 화면에 뜨는 판정 지표를 그대로 남긴다."""
        with self.lock:
            if self.f is None:
                return
            imu = view.get("imu") or {}
            # '실제 활동' = 스택이 살아 있거나(stage8 상태 수신), 정책이 돌거나,
            # 로봇이 하중을 받고 있을 때. 매달린 채 방치된 상태는 활동이 아니다.
            if (view["fresh"].get("status") or view["fresh"].get("bridge")):
                self.last_active = now()
            self._write({
                "kind": "s",
                "pct": view["foot"]["pct"],
                "fn": [view["foot"]["left_n"], view["foot"]["right_n"]],
                "fwd": imu.get("fwd"), "lat": imu.get("lat"),
                "tilt": imu.get("tilt"), "gyro": imu.get("gyro"),
                "quat": imu.get("quat"),
                "armed": view["stage8"]["armed"],
                "base": view["stage8"]["baseline"],
                "live": view["bridge"]["live"],
                "ckpt": view["bridge"]["checkpoint"],
                "blend": view["bridge"]["blend"],
                "contact": view["bridge"]["contact"],
                "hang_state": view["bridge"].get("hang_state"),   # 0911 R5: 리프트 가드 무장 여부 기록
                "sensor_contact": view["foot"].get("contact"),
                "foot_raw_sum": view["foot"].get("raw_sum"),
                "foot_pads_on": view["foot"].get("pads_on"),
                "stage8_running": view["stage8"].get("running"),
                "stage8_command_seq": view["stage8"].get("command_seq"),
                "stage8_last_command": view["stage8"].get("last_command"),
                "joints": {name: {key: joint.get(key) for key in (
                    "latest_actual_deg", "baseline_deg", "commanded_relative_deg",
                    "desired_relative_deg", "feedback_valid", "feedback_stale", "feedback_age_sec")}
                    for name, joint in view["stage8"].get("joints", {}).items() if isinstance(joint, dict)},
                "trial": {k: view.get("trial", {}).get(k) for k in
                          ("number", "mode", "checkpoint", "phase", "owner", "fault", "measurement", "stop")},
                "unverified_force_fields": True,
                "fresh": [int(view["fresh"]["imu"]), int(view["fresh"]["foot"]),
                          int(view["fresh"]["status"])],
            })
            self.n_sample += 1

    def open_proc_log(self, key):
        """프로세스 출력을 세션 폴더에 파일로도 남긴다 (링버퍼는 잘린다)."""
        with self.lock:
            if self.dir is None:
                return None
            try:
                return open(os.path.join(
                    self.dir, f"{key}_{time.strftime('%H%M%S')}.log"), "a")
            except OSError:
                return None

    def _close_locked(self):
        if self.f is not None:
            self._write({"kind": "session_end", "samples": self.n_sample,
                         "events": self.n_event})
            try:
                self.f.close()
            except OSError:
                pass
            self.f = None

    def close(self, reason=""):
        with self.lock:
            if self.f is None:
                return None
            name = self.name
            self.event("session_close", reason=reason)
            self._close_locked()
            print(f"[세션] 종료 {name}")
            return name

    def maybe_autoclose(self):
        """유휴 자동 닫힘 — 단, 사람이 직접 시작한 세션은 건드리지 않는다."""
        with self.lock:
            if self.f is None or self.manual:
                return
            idle = now() - self.last_active
            if idle > IDLE_CLOSE_SEC:
                self.close(f"유휴 {int(idle/60)}분 자동 종료")

    def status(self):
        with self.lock:
            return {"name": self.name, "active": self.f is not None,
                    "manual": bool(self.manual),
                    "dur": (round(now() - self.t0, 1) if self.t0 else None),
                    "idle": (round(now() - self.last_active, 1)
                             if self.f is not None else None),
                    "samples": self.n_sample, "events": self.n_event,
                    "dir": self.dir}


SESSION = SessionLog()


# ---------------------------------------------------------------------------
# 프로세스 관리 — 콘솔이 띄운 것만 콘솔이 내린다
# ---------------------------------------------------------------------------
class Proc:
    """장기 실행 프로세스 하나. 출력은 링버퍼(웹 tail)와 세션 파일 양쪽에."""

    def __init__(self, key, argv, cwd=None, env=None):
        self.key = key
        self.argv = argv
        self.lines = collections.deque(maxlen=300)
        self.started = now()
        self.rc = None
        self.hold = {}
        self.loaded_checkpoint = None
        self.warmup = None            # 0911 R4: 브리지 인계 워밍업 {state: start|waiting|done|timeout, ts, detail}
        self.all_pass = False
        self.fault = None
        self.blackbox_path = None
        self.release_ok = False
        self.release_error = None
        self.logf = SESSION.open_proc_log(key)
        self.p = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.p.stdout:
                s = line.rstrip("\n")[:400]
                self.lines.append(s)
                hm = re.search(r"\[(에어유지|에어램프|게인램프|운용대기|하중유지)\s+[^]]*\]\s+kp\s*([\d.]+)", s)
                if hm:
                    self.hold = {"phase": hm[1], "kp": float(hm[2]), "ts": now()}
                cm = re.search(r"RL 브리지 기동.*LIVE.*checkpoint=([A-Za-z0-9_]+)", s)
                if cm:
                    self.loaded_checkpoint = cm[1]
                if self.key == "bridge":   # 0911 R4: 워밍업 상태를 stdout 계약에서 뽑아 화면·기록에 남긴다
                    if "인계 워밍업 최대" in s:
                        self.warmup = {"state": "start", "ts": now(), "detail": ""}
                    elif "[인계 워밍업 대기]" in s:
                        self.warmup = {"state": "waiting", "ts": now(), "detail": s.split("]", 1)[-1].strip()[:160]}
                    elif "인계 워밍업 완료" in s:
                        self.warmup = {"state": "done", "ts": now(), "detail": ""}
                    elif "sensor_stale_warmup_timeout" in s:
                        self.warmup = {"state": "timeout", "ts": now(), "detail": s[-200:]}
                if self.key == "bridge" and "블랙박스 기록: " in s:
                    self.blackbox_path = s.split("블랙박스 기록: ", 1)[1].strip()
                if "selftest: ALL PASS" in s:
                    self.all_pass = True
                if self.key == "bridge" and "STOP_ALL (" in s:
                    self.fault = s[s.index("STOP_ALL ("):]
                if self.key == "release":
                    if "12모터 소프트 릴리즈+MIT 해제 완료" in s:
                        self.release_ok = True
                    if "송신실패" in s or "읽기 실패" in s:
                        self.release_error = s
                if self.logf:                       # 링버퍼는 300줄에서 잘린다 —
                    try:                            # 판정 근거는 파일에 남겨야 한다
                        self.logf.write(f"{time.strftime('%H:%M:%S')} {s}\n")
                        self.logf.flush()
                    except OSError:
                        self.logf = None
        except Exception:
            pass
        self.rc = self.p.wait()
        self.lines.append(f"[종료] 반환코드 {self.rc}")
        SESSION.event("proc_exit", key=self.key, rc=self.rc,
                      dur=round(now() - self.started, 1))
        if self.logf:
            try:
                self.logf.write(f"[종료] 반환코드 {self.rc}\n")
                self.logf.close()
            except OSError:
                pass
            self.logf = None

    @property
    def alive(self):
        return self.p.poll() is None

    def stop(self, sig=15):
        """추적 중인 PID에만 신호를 보낸다 (pkill 금지 규칙)."""
        if self.alive:
            try:
                os.killpg(os.getpgid(self.p.pid), sig)
            except OSError:
                try:
                    self.p.send_signal(sig)
                except OSError:
                    pass

    def snapshot(self, n=25):
        return {"key": self.key, "pid": self.p.pid, "alive": self.alive,
                "rc": self.rc, "age": round(now() - self.started, 1),
                "argv": " ".join(shlex.quote(a) for a in self.argv),
                "lines": list(self.lines)[-n:], "hold": dict(self.hold),
                "loaded_checkpoint": self.loaded_checkpoint, "all_pass": self.all_pass,
                "warmup": getattr(self, "warmup", None),   # 테스트는 __init__ 없이 Proc 를 만들 수 있음
                "fault": self.fault, "started": self.started, "blackbox_path": self.blackbox_path,
                "release_ok": self.release_ok, "release_error": self.release_error}


class Runner:
    def __init__(self):
        self.procs = {}          # key -> Proc
        self.lock = threading.Lock()

    # 0910 현장: 콘솔 재기동 뒤엔 자식 추적이 사라져 같은 노드를 두 번 띄울 수 있었다(센서 어댑터·stage8·브리지 중복 = 같은 버스/토픽 이중 발행).
    # 추적 밖 프로세스도 명령행 패턴으로 검출해 거부한다. 종료는 콘솔이 띄운 것만(명시 PID) — 정리는 스택 종료 버튼.
    DUP_GUARD = {"stage8": ["stage8_12axis_mit_control_node.py"], "hang_stand": ["hang_stand.py"],
                 "bridge": ["rl_bridge_node.py"], "sensors": ["imu_adapter_node.py", "foot_force_adapter_node.py"]}

    @staticmethod
    def _foreign_pids(patterns):
        found = []
        for pat in patterns:
            try:
                out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=3).stdout.split()
            except (OSError, subprocess.TimeoutExpired):
                out = []
            found += [int(x) for x in out if x.isdigit() and int(x) != os.getpid()]
        return sorted(set(found))

    def start(self, key, argv, cwd=None):
        with self.lock:
            old = self.procs.get(key)
            if old is not None and old.alive:
                return False, f"{key} 이미 실행 중 (pid {old.p.pid})"
            if key in self.DUP_GUARD:
                pids = self._foreign_pids(self.DUP_GUARD[key])
                if pids:
                    return False, (f"{key} 이미 실행 중 (pid {pids}, 콘솔 추적 밖) — 중복 기동 안 함. "
                                   "기존 소유 콘솔/터미널에서 상태를 확인하세요. 이 콘솔은 외부 PID를 종료하지 않습니다")
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            try:
                self.procs[key] = Proc(key, argv, cwd=cwd, env=env)
            except OSError as exc:
                return False, f"{key} 기동 실패: {exc}"
            return True, f"{key} 기동 (pid {self.procs[key].p.pid})"

    def stop(self, key):
        with self.lock:
            p = self.procs.get(key)
            if p is None or not p.alive:
                return False, f"{key} 실행 중 아님"
            p.stop()
            return True, f"{key} 종료 신호 (pid {p.p.pid})"

    def alive(self, key):
        p = self.procs.get(key)
        return bool(p and p.alive)

    def snapshot(self):
        with self.lock:
            return {k: p.snapshot() for k, p in self.procs.items()}


# ---------------------------------------------------------------------------
# 텔레메트리 — ROS2 구독 (없으면 콘솔은 조작반으로만 동작)
# ---------------------------------------------------------------------------
class Telemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"status": None, "imu": None, "foot": None, "bridge": None}
        self.t = {k: 0.0 for k in self.d}
        self.events = collections.deque(maxlen=40)
        self.ok = False
        self.node = None
        self._pub_stage8 = None
        self._pub_bridge = None
        self.bridge_fault = None

    def start(self):
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except Exception as exc:
            self.events.append(f"[경고] rclpy 없음 — 텔레메트리 비활성 ({exc})")
            return
        rclpy.init(args=None)
        node = Node("operator_console")

        def mk(key):
            def cb(msg):
                try:
                    obj = json.loads(msg.data)
                except ValueError:
                    return
                self.receive(key, obj)
            return cb

        node.create_subscription(String, STATUS_TOPIC, mk("status"), 10)
        node.create_subscription(String, IMU_TOPIC, mk("imu"), 20)
        node.create_subscription(String, FOOT_TOPIC, mk("foot"), 20)
        node.create_subscription(String, BRIDGE_DEBUG_TOPIC, mk("bridge"), 20)
        self._pub_stage8 = node.create_publisher(String, COMMAND_TOPIC, 10)
        self._pub_bridge = node.create_publisher(String, BRIDGE_CMD_TOPIC, 10)
        self.node = node
        self.ok = True

        def spin():
            try:
                rclpy.spin(node)
            except Exception:
                pass
        threading.Thread(target=spin, daemon=True).start()

    def receive(self, key, obj):
        """Event packets do not replace a full telemetry snapshot or its age."""
        if not isinstance(obj, dict):
            return
        with self.lock:
            if key == "bridge" and "event" in obj:
                reason = obj.get("reason") or obj.get("why") or obj["event"]
                self.events.append(f"{time.strftime('%H:%M:%S')} {obj['event']} {reason}")
                if obj["event"] in ("stop_all", "lift_guard", "osc_guard"):
                    self.bridge_fault = str(reason)
                return
            self.d[key] = obj
            self.t[key] = now()
            if key == "bridge" and (obj.get("stop_reason") or obj.get("osc_tripped") or obj.get("stopped_for_stale")):
                self.bridge_fault = str(obj.get("stop_reason") or obj.get("osc_tripped") or "sensor_stale")

    def publish(self, which, obj):
        pub = self._pub_stage8 if which == "stage8" else self._pub_bridge
        if pub is None:
            return False
        from std_msgs.msg import String
        pub.publish(String(data=json.dumps(obj)))
        return True

    def view(self):
        """Normalize observed status; missing/invalid data is never a healthy zero."""
        with self.lock:
            d = dict(self.d)
            t = dict(self.t)
            ev = list(self.events)[-12:]
        n = now()
        fresh = {k: -0.5 <= (n - t[k]) < STALE_SEC and t[k] > 0 for k in t}
        for key, data in d.items():
            source_ts = (data or {}).get("timestamp")
            if not finite(source_ts) or not -0.5 <= n - source_ts < STALE_SEC:
                fresh[key] = False

        out = {"fresh": fresh, "events": ev, "ros": self.ok, "received": t,
               "server_time": n}

        foot = d.get("foot") or {}
        ln, rn = foot.get("left_n", 0.0), foot.get("right_n", 0.0)
        if not finite(ln) or not finite(rn):
            ln, rn = 0.0, 0.0
        source_ts = foot.get("timestamp")
        if finite(source_ts) and not -0.5 <= n - source_ts < STALE_SEC:
            fresh["foot"] = False
        contacts = [foot.get("left_contact"), foot.get("right_contact")]
        contact_valid = all(type(v) is bool for v in contacts)
        # Old adapter data remains visible in diagnostics, but cannot certify
        # contact from an uncalibrated force scale.
        if not contact_valid:
            fresh["foot"] = False
        out["foot"] = {
            "left_n": round(ln, 1), "right_n": round(rn, 1),
            "total_n": round(ln + rn, 1),
            "pct": round(100.0 * (ln + rn) / BODY_WEIGHT_N, 1),
            "contact": contacts if contact_valid and fresh["foot"] else None,
            "raw_sum": [foot.get("left_raw_sum"), foot.get("right_raw_sum")],
            "pads_on": [foot.get("left_pads_on"), foot.get("right_pads_on")],
            "age": round(n - t["foot"], 2) if t["foot"] else None,
            "reason": "패드 접촉 필드 없음 — 어댑터 버전 확인" if not contact_valid else (None if fresh["foot"] else "접촉 데이터 오래됨/비유한 값"),
        }

        # IMU는 **항상** 값을 내보낸다. 끊기면 None 으로 지우지 않고 마지막
        # 값에 age 를 붙여 보낸다 — 화면에서 갑자기 "—"가 되면 자세를 잃은
        # 것인지 센서가 끊긴 것인지 구분이 안 된다(원격 오판 사고의 씨앗).
        imu = d.get("imu") or {}
        q = imu.get("quat") or imu.get("quat_wxyz")
        gyro = imu.get("gyro_rad_s") or imu.get("gyro") or []
        valid_imu = (isinstance(q, (list, tuple)) and len(q) == 4 and all(finite(v) for v in q)
                     and 0.8 < sum(v*v for v in q) < 1.2
                     and isinstance(gyro, (list, tuple)) and len(gyro) == 3 and all(finite(v) for v in gyro))
        if not valid_imu:
            fresh["imu"] = False
        imu_ts = imu.get("timestamp")
        if finite(imu_ts) and not -0.5 <= n - imu_ts < STALE_SEC:
            fresh["imu"] = False
        if valid_imu:
            w, x, y, z = (float(v) for v in q)
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            # 방위(yaw) — 기울기와 달리 중력만으로는 못 구하므로 quat 에서
            yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                          1.0 - 2.0 * (y * y + z * z)))
            self._imu_last = {
                "yaw": round(yaw, 1),
                # 부호 규약(0809 검증): fwd>0 = 앞으로 기울임,
                # lat>0 = 왼쪽으로 기울임 (base +X 전방 / +Y 좌측)
                "fwd": round(math.degrees(math.atan2(gx, -gz)), 1),
                "lat": round(math.degrees(math.atan2(gy, -gz)), 1),
                "tilt": round(math.degrees(
                    math.acos(max(-1.0, min(1.0, -gz)))), 1),
                "quat": [round(float(v), 4) for v in (w, x, y, z)],
                "gyro": [round(float(v), 3) for v in gyro[:3]],
                "grav": [round(gx, 3), round(gy, 3), round(gz, 3)],
            }
        last = getattr(self, "_imu_last", None)
        if last is not None:
            out["imu"] = dict(last)
            out["imu"]["age"] = (round(n - t["imu"], 1) if t["imu"] else None)
        else:
            out["imu"] = None

        st = d.get("status") or {}
        if not all(type(st.get(k)) is bool for k in ("armed", "running", "accepted")):
            fresh["status"] = False
        joints = st.get("joints") if isinstance(st.get("joints"), dict) else {}
        tk = st.get("takeover") if isinstance(st.get("takeover"), dict) else {}
        n_base = sum(1 for v in joints.values()
                     if isinstance(v, dict) and finite(v.get("baseline_deg")))
        out["stage8"] = {
            "armed": bool(st.get("armed")), "holding": bool(tk.get("holding")),
            "waiting": bool(tk.get("waiting")), "baseline": n_base,
            "hold_age": tk.get("hold_age_sec"), "hold_joint_count": tk.get("hold_joint_count", 0),
            "last_event": tk.get("last_event"), "running": bool(st.get("running")),
            "last_command": st.get("last_command"), "accepted": st.get("accepted") is True,
            "command_seq": st.get("command_seq"), "errors": st.get("errors", []),
            # 0911 호환검수 R3: stage8 의 feedback_stale(3 s)는 브리지 라이브 게이트(0.2 s, 관절별 worst_age)보다 15배
            # 느슨해 콘솔이 초록인 채 stand 첫 틱이 sensor_stale 로 죽을 수 있다 → 브리지와 같은 0.2 s 를 관절별로 요구.
            "feedback_ok": (fresh["status"] and len(joints) == 12 and
                            all(isinstance(j, dict) and j.get("feedback_valid") is True
                                and j.get("feedback_stale") is False and finite(j.get("latest_actual_deg"))
                                and isinstance(j.get("feedback_age_sec"), (int, float))
                                and float(j.get("feedback_age_sec")) < BRIDGE_JOINT_FRESH_SEC
                                for j in joints.values())),
            "joints": joints,
        }

        br = d.get("bridge") or {}
        if not all(type(br.get(k)) is bool for k in ("live", "armed")):
            fresh["bridge"] = False
        out["bridge"] = {
            "live": bool(br.get("live")), "armed": bool(br.get("armed")),
            "checkpoint": br.get("checkpoint"),
            "blend": br.get("blend_alpha"),
            "contact": br.get("contact"),
            "fault": self.bridge_fault,
            "hang_state": br.get("hang_state"),
        }
        return out


class CanHealth:
    """CAN 버스 건강 — bus-off 폭주를 화면에 띄운다.

    0809 실기: 로봇이 서 있는 동안 bus-off 가 200회 가까이 발생해 모터가
    통신을 잃었다 되찾으며 "무릎이 쫙 나갔다가 점프하듯 튀는" 사고가 났는데,
    **화면에는 아무 표시도 없었다.** 조작자는 원인을 알 수 없었고 진단에
    한참 걸렸다. 버스 건강은 발력·기울기와 같은 급의 안전 지표다.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.busoff = None
        self.prev = None
        self.rate = 0.0          # 최근 60초 bus-off 증가분
        self.hist = collections.deque(maxlen=60)

    def poll(self):
        try:
            out = subprocess.run(["ip", "-d", "-s", "link", "show", "can1"],
                                 capture_output=True, text=True, timeout=4).stdout
        except (OSError, subprocess.SubprocessError):
            return
        st = None
        m = re.search(r"can state (\S+)", out)
        if m:
            st = m.group(1)
        bo = None
        m = re.search(r"re-started\s+bus-errors.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+"
                      r"(\d+)\s+(\d+)\s+(\d+)", out, re.S)
        if m:
            bo = int(m.group(6))
        with self.lock:
            self.state = st
            if bo is not None:
                if self.prev is not None:
                    self.hist.append(max(0, bo - self.prev))
                self.prev = bo
                self.busoff = bo
                self.rate = sum(self.hist)

    def view(self):
        with self.lock:
            return {"state": self.state, "busoff": self.busoff,
                    "rate60": self.rate}


RUN = Runner()
TEL = Telemetry()
CAN = CanHealth()
TRIAL = TrialState(CKPT_DEFAULT)
STATE_LOCK = threading.RLock()
REQUESTS = collections.OrderedDict()
BUSY = False
DEMO = False
PROCESS_HEALTH = {"ts": 0, "external": [], "error": "아직 검사 안 됨"}


def poll_processes():
    """Read-only check of known local senders. Never kill/adopt foreign PIDs."""
    patterns = ["stage8_12axis_mit_control_node.py", "hang_stand.py", "rl_bridge_node.py",
                "posture_probe.py", "zero_kinematic_check.py", "release_all.py", "zero_anchor.py", "stage8_cmd.py"]
    try:
        result = subprocess.run(["pgrep", "-f", "|".join(re.escape(p) for p in patterns)],
                                capture_output=True, text=True, timeout=3)
        if result.returncode not in (0, 1):
            raise OSError("프로세스 목록 검사 실패")
        # Hold the runner lock only during the short ownership comparison.
        with RUN.lock:
            owned = {p.p.pid for p in RUN.procs.values()}
        external = [int(v) for v in result.stdout.split() if v.isdigit() and int(v) not in owned | {os.getpid()}]
        health = {"ts": now(), "external": external, "error": None}
    except (OSError, subprocess.SubprocessError) as exc:
        health = {"ts": now(), "external": [], "error": str(exc)}
    with STATE_LOCK:
        PROCESS_HEALTH.update(health)


# ---------------------------------------------------------------------------
# 액션 화이트리스트 — 웹에서 실행 가능한 것의 전부
# ---------------------------------------------------------------------------
def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def act_can_up():
    p = subprocess.run(
        ["sudo", "-S", "bash", os.path.join(CALIB, "can_up.sh")],
        input=SUDO_PW + "\n", capture_output=True, text=True, timeout=90)
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    return p.returncode == 0, "\n".join(tail) or "완료"


def act_gate(key, argv, timeout=300):
    ok, msg = RUN.start(key, argv)
    return ok, msg


def act_marker(path):
    with open(path, "w"):
        pass
    return True, f"마커 생성: {path}"


def act_stage8_cmd(sub):
    ok, msg = RUN.start(f"stage8_{sub}",
                        [sys.executable, os.path.join(CALIB, "stage8_cmd.py"), sub])
    return ok, msg


def bridge_cmd_payload(cmd, extra=None):
    """브리지 명령 JSON. extra 는 명시 플래그만 (0908: stand 의 hang_arm — 접지 전 무접촉 허용, E·F 단계 전용)."""
    d = {"cmd": cmd}
    if extra:
        d.update({k: v for k, v in extra.items() if v})
    return d


def act_bridge_cmd(cmd, extra=None):
    st = TEL.view()
    # 0910 A1 현장 버그: 브리지는 stand 이후에만 debug 토픽을 내므로 '토픽 기준 live' 로는 첫 stand 를 영영 못 보낸다.
    # 콘솔이 띄운 브리지 프로세스가 살아 있으면(RUN 추적) 허용. 실물에선 ros2 CLI(/tmp/fire_stand.sh)로 우회했었음.
    _need(st["bridge"]["live"] or RUN.alive("bridge") or cmd == "estop",
          "브리지가 live 상태가 아닙니다 — 먼저 브리지를 기동하세요")
    payload = bridge_cmd_payload(cmd, extra)
    ok = TEL.publish("bridge", payload)
    tag = cmd + (" [행잉 ARM]" if payload.get("hang_arm") else "")
    return ok, f"브리지 명령 발행: {tag}" if ok else "발행 실패 (ROS 미연결)"


def act_estop():
    """항상 실행 가능해야 한다 — 어떤 선행조건도 걸지 않는다."""
    msgs = []
    sent = 0
    # Cancel owned retrying command senders, so a delayed ARM retry cannot undo
    # the freeze. Do not terminate the torque-holding process or release motors.
    for key in ("stage8_arm", "stage8_baseline"):
        try:
            if RUN.alive(key):
                RUN.stop(key)
                msgs.append(f"{key} 재시도 송신자 종료 요청")
        except Exception as exc:
            msgs.append(f"{key} 종료 요청 실패: {exc}")
    try:
        if TEL.publish("bridge", {"cmd": "estop"}):
            msgs.append("브리지 estop 발행")
            sent += 1
    except Exception as exc:
        msgs.append(f"브리지 estop 실패: {exc}")
    try:
        if TEL.publish("stage8", {"command": "STOP_ALL"}):
            msgs.append("stage8 STOP_ALL 발행")
            sent += 1
    except Exception as exc:
        msgs.append(f"STOP_ALL 실패: {exc}")
    return sent == 2, (" / ".join(msgs) or "발행 경로 없음 (ROS 미연결)") + " — 실제 동결 응답 확인 필요"


def act_stack_down():
    """Only terminate console-owned processes after explicit support confirmation."""
    messages = []
    for key in ("stage8_arm", "stage8_baseline", "bridge", "hang_stand", "stage8"):
        if RUN.alive(key):
            ok, message = RUN.stop(key)
            messages.append(message)
    return True, " / ".join(messages) or "추적 중인 제어 프로세스 없음 — 잔존 상태 확인 중"


#: hang_stand 트림 채널 파일 (0.5°/s 로 램프되므로 급변 없음)
TRIM_FILES = {"a": "/tmp/ankle_trim_deg", "h": "/tmp/hip_trim_deg",
              "k": "/tmp/knee_trim_deg"}
#: 한계 — hang_stand.TRIM_CH 와 일치시킬 것 (0809 확대: a16/h10/k14)
TRIM_LIM = {"a": (-4.0, 16.0), "h": (-4.0, 10.0), "k": (-2.0, 14.0)}


def act_trim(payload):
    """트림 3채널 설정. 한계 밖 값은 거부한다(자동 클램프하지 않는다 —
    조작자가 의도한 값이 안 들어간 것을 모르는 편이 더 위험하다)."""
    wrote = []
    for ch, path in TRIM_FILES.items():
        if ch not in payload:
            continue
        v = float(payload[ch])
        lo, hi = TRIM_LIM[ch]
        _need(lo <= v <= hi, f"{ch} 트림 {v:+.1f}° 는 한계 [{lo}, {hi}] 밖입니다")
        with open(path, "w") as f:
            f.write(f"{v}\n")
        wrote.append(f"{ch}={v:+.1f}°")
    _need(wrote, "설정할 트림 채널이 없습니다 (a/h/k 중 하나 이상 필요)")
    return True, "트림 " + " ".join(wrote) + " (0.5°/s 램프)"


def act_bridge_live(payload):
    ckpt = str(payload.get("checkpoint", CKPT_DEFAULT))
    _need(ckpt in CKPT_CHOICES, "허용되지 않은 체크포인트입니다")
    return RUN.start("bridge",
                     [sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"),
                      "--live", "--checkpoint", ckpt])


def act_analyze():
    path = RUN.snapshot().get("bridge", {}).get("blackbox_path")
    _need(path and os.path.isfile(path), "이 시험 브리지의 블랙박스 경로를 확인하지 못했습니다 — 최신 파일로 대체하지 않습니다")
    _need(Path(LOGDIR).resolve() in Path(path).resolve().parents, "시험 로그 경로가 로그 디렉터리 밖입니다")
    return RUN.start("analyze",
                     [sys.executable, os.path.join(SCRIPTS, "analyze_fall.py"),
                      path])


ACTIONS = {
    # 게이트
    "can_up":     (lambda p: act_can_up(), "CAN 브링업"),
    "boot_pose":  (lambda p: act_gate("boot_pose", [
        sys.executable, os.path.join(CALIB, "boot_pose_check.py"),
        "--ref", "stand"]), "영점 판정 12/12"),
    "sensors":    (lambda p: act_gate("sensors", [
        "bash", os.path.join(CALIB, "sensors_up.sh")]), "센서 기동"),
    # 0907: 체크포인트를 넘기지 않으면 브리지가 기본값(구 walk)을 **필수**로 로드해
    # 실패한다 — 젯슨 구 자산 정리로 walk 는 더 이상 배포되지 않는다. 드롭다운 기본값을 넘긴다.
    "selftest":   (lambda p: act_gate("selftest", [
        sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"),
        "--selftest", "--checkpoint",
        (p.get("checkpoint") if p.get("checkpoint") in CKPT_CHOICES
         else CKPT_DEFAULT)]), "브리지 셀프테스트"),
    # 영점 재앵커 — 미리보기는 읽기 전용, 적용은 zero_anchor.py 가 자체 차단
    "zero_preview": (lambda p: RUN.start("zero", [
        sys.executable, os.path.join(SCRIPTS, "zero_anchor.py"),
        "--preview"]), "영점 미리보기(읽기만)"),
    "zero_multiple": (lambda p: RUN.start("zero", [
        sys.executable, os.path.join(SCRIPTS, "zero_anchor.py"),
        "--apply", "multiple"]), "영점 배수 보정"),
    "zero_recapture": (lambda p: RUN.start("zero", [
        sys.executable, os.path.join(SCRIPTS, "zero_anchor.py"),
        "--apply", "recapture", "--in-reference-pose"]), "영점 재캡처"),
    # 스택
    "stage8":     (lambda p: RUN.start("stage8", [
        sys.executable, os.path.join(SCRIPTS, "stage8_12axis_mit_control_node.py"),
        "--can-channel", "can1", "--wait-takeover"]), "stage8 선기동(인수대기)"),
    "hang_stand": (lambda p: RUN.start("hang_stand", [
        sys.executable, os.path.join(CALIB, "hang_stand.py")]), "기립 스크립트"),
    # 트림 — hang_stand 의 3채널을 콘솔에서 직접. 부호를 손으로 계산하다
    # 반대로 밀었던 사고(0809: 발목을 하한 −2로 밀고 있었는데 답은 +12.5)를
    # 막기 위해 **정책 평형 자세 프리셋**을 함께 둔다.
    "trim":       (act_trim, "트림 조정"),
    "trim_preset": (lambda p: act_trim({"a": 12.5, "h": 6.4, "k": 10.4}),
                    "정책 평형 자세 트림(v8)"),
    "trim_zero":  (lambda p: act_trim({"a": 0.0, "h": 0.0, "k": 0.0}),
                   "트림 전부 0"),
    # 게인 램프 — **착지 전에** 공중에서 kp20→kp150 (hang_stand 문서 3단계).
    # 전 모터 오차 3° 미만일 때만 수락된다(아니면 스크립트가 파일 삭제+거부).
    "gain_ramp":  (lambda p: act_marker("/tmp/hang_gain"), "게인 램프 kp150"),
    "ground":     (lambda p: act_marker("/tmp/hang_ground"), "접지 선언(스톨가드 해제)"),
    "handoff":    (lambda p: act_marker("/tmp/hang_stand_handoff"), "핸드오프"),
    "baseline":   (lambda p: act_stage8_cmd("baseline"), "베이스라인 12/12"),
    "arm":        (lambda p: act_stage8_cmd("arm"), "ARM_ALL"),
    "disarm":     (lambda p: act_stage8_cmd("disarm"), "DISARM_ALL"),
    "bridge":     (act_bridge_live, "브리지 live"),
    # 영점 기구학 검증 — 정책 없이 발목 ±3° 스윕 → 절편(=영점 오차)
    "zero_kin": (lambda p: RUN.start("zerokin", [
        sys.executable, os.path.join(SCRIPTS, "zero_kinematic_check.py")]),
        "영점 기구학 검증"),
    # 정적 자세 시험 — 정책 없이 stage8 홀드만으로 목표 자세 진입
    "posture":    (lambda p: RUN.start("posture", [
        sys.executable, os.path.join(SCRIPTS, "posture_probe.py"),
        "--pose", (p.get("pose") if p.get("pose") in
                   ("policy_v8", "policy_v7", "zero", "knee_only", "ankle_only")
                   else "policy_v8")]), "정적 자세 시험"),
    # 정책
    # 0908: hang_arm 체크 시에만 브리지 HangGuard 가 접지 전 무접촉을 허용한다(미체크 = 종전과 동일)
    "stand":      (lambda p: act_bridge_cmd("stand", {"hang_arm": bool(p.get("hang_arm"))}), "stand 발사"),
    "march":      (lambda p: act_bridge_cmd("march"), "march 발사"),
    "estop":      (lambda p: act_estop(), "비상정지"),
    # 회수·판정
    "stack_down": (lambda p: act_stack_down(), "콘솔 소유 제어 프로세스 종료"),
    "release":    (lambda p: RUN.start("release", [
        sys.executable, os.path.join(CALIB, "release_all.py")]), "릴리즈"),
    "analyze":    (lambda p: act_analyze(), "최신 로그 판정"),
    # 운영 윈치 이벤트(0910 A4) — 동작 없이 세션 타임라인에 시각 기록만. 블랙박스(wall 시각)와 사후 정렬해 '방출/재인장/슬랙'을 판별.
    "op_release_start": (lambda p: (True, "기록: 방출 시작"), "▶ 방출 시작"),
    "op_release_stop":  (lambda p: (True, "기록: 방출 정지"), "⏸ 방출 정지"),
    "op_retension":     (lambda p: (True, "기록: 재인장"), "↥ 재인장"),
    "op_slack_done":    (lambda p: (True, "기록: 줄 완전 슬랙"), "✓ 슬랙 완료"),
    "op_touchdown":     (lambda p: (True, "기록: 착지"), "▼ 착지"),
    "op_note":          (lambda p: (True, f"기록: {str(p.get('note',''))[:80]}"), "메모"),
    # 세션 — 자동 개시/종료가 있지만 사람이 명시로 잡을 수 있어야 한다
    "session_start": (lambda p: (True, f"기록 시작: "
                                 f"{SESSION.ensure('수동 시작', manual=True)}"),
                      "기록 시작"),
    "session_end": (lambda p: (True, f"기록 중지: {SESSION.close('수동') or '없음'}"),
                    "기록 중지"),
}
STOPPABLE = {"stage8", "hang_stand", "bridge"}

PAGE = Path(__file__).with_name("console_ui.html").read_text(encoding="utf-8")
DEMO_SOURCE = None
CONFIRM_ACTIONS = {"hang_stand", "gain_ramp", "ground", "handoff", "arm", "stand",
                   "release", "support_confirm", "op_slack_done", "new_trial"}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def console_snapshot():
    """Called by the recorder too: browser refresh is never the state machine."""
    with STATE_LOCK:
        st = DEMO_SOURCE.snapshot() if DEMO else TEL.view()
        if not DEMO:
            st["procs"] = RUN.snapshot()
            st["can"] = CAN.view()
            bp = st["procs"].get("bridge", {})
            st["bridge"]["loaded_checkpoint"] = bp.get("loaded_checkpoint") if bp.get("alive") else None
            st["bridge"]["fault"] = st["bridge"].get("fault") or bp.get("fault")
            st["external_controls"] = list(PROCESS_HEALTH["external"])
            st["process_check_ok"] = not PROCESS_HEALTH["error"] and now() - PROCESS_HEALTH["ts"] < 4.0
        st["session"] = SESSION.status()
        st["demo"] = DEMO
        TRIAL.update(st)
        st["trial"] = TRIAL.snapshot()
        if BUSY:
            for key, item in st["trial"]["allowed"].items():
                if key not in EVENTS | {"estop", "support_confirm"}:
                    item.update(enabled=False, reason="서버에서 앞 요청을 처리 중입니다")
        for event in TRIAL.drain_events():
            SESSION.event("trial_event", **event)
        return json_safe(st)


def dispatch_action(payload):
    """Single server gate for every UI/API request. Never retry hardware commands."""
    global BUSY, TRIAL, DEMO_SOURCE
    if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
        return 400, {"ok": False, "message": "잘못된 요청"}
    action = payload["action"]
    if action not in LABELS:
        return 403, {"ok": False, "message": "기립 시험에서 허용하지 않는 동작"}
    if action == "estop":
        # No revision, recorder, pending-action or sensor gate may block this.
        with STATE_LOCK:
            TRIAL.stop_request_at = now()
            TRIAL.stop_request_seq = TRIAL.view.get("stage8", {}).get("command_seq") or 0
            TRIAL.stop_delivery = None
            TRIAL.latch("운영자 정지 요청 — 회수 후 새 시험 필요")
            if TRIAL.pending:
                TRIAL.finish(False, "정지 요청으로 진행 취소 · 이미 보낸 명령은 실제 상태 확인 필요")
        try:
            ok, msg = DEMO_SOURCE.act(action, {}) if DEMO else act_estop()
        except Exception as exc:
            ok, msg = False, f"정지 발행 오류: {exc}"
        with STATE_LOCK:
            TRIAL.stop_delivery = {"ok": bool(ok), "message": msg}
            TRIAL.emit("stop_delivery", ok=bool(ok), message=msg)
        return 200, {"ok": bool(ok), "message": msg, "confirmed": False}

    immediate = action in EVENTS | {"support_confirm", "configure", "new_trial", "session_start", "session_end"}
    if not DEMO and not immediate:
        poll_processes()
    with STATE_LOCK:
        console_snapshot()
        if payload.get("boot_id") != TRIAL.boot_id:
            return 409, {"ok": False, "message": "서버가 재시작됐거나 페이지가 오래됐습니다. 현재 상태를 다시 확인하세요"}
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
            return 400, {"ok": False, "message": "요청 식별자가 필요합니다"}
        fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if request_id in REQUESTS:
            old, response = REQUESTS[request_id]
            if old != fingerprint:
                return 409, {"ok": False, "message": "같은 요청 식별자를 다른 동작에 사용할 수 없습니다"}
            return response
        if payload.get("revision") != TRIAL.revision:
            return 409, {"ok": False, "message": "다른 조작으로 단계가 바뀌었습니다. 현재 상태를 확인하고 다시 선택하세요"}
        reason = TRIAL.reason(action)
        if reason or (BUSY and action not in EVENTS | {"support_confirm"}):
            return 409, {"ok": False, "message": reason or "앞 요청 처리 중"}
        if action in CONFIRM_ACTIONS and payload.get("confirmed") is not True:
            return 400, {"ok": False, "message": "운영자의 명시 확인이 필요합니다"}
        if action == "configure":
            if payload.get("mode") not in ("ground", "air") or payload.get("checkpoint") not in CKPT_CHOICES:
                return 400, {"ok": False, "message": "허용된 시험 방식·정책을 선택하세요"}
            if not payload["checkpoint"].startswith("stand"):
                return 400, {"ok": False, "message": "이 콘솔은 기립 시험용입니다. 보행 정책은 별도 절차가 필요합니다"}
        record_warning = ""
        try:
            if action != "session_end":
                SESSION.ensure(LABELS[action], manual=True)
                SESSION.event("action_request", action=action, trial=TRIAL.number, request_id=request_id)
                if not SESSION.status()["active"]:
                    raise OSError("세션 파일 쓰기 실패")
        except OSError as exc:
            if action not in {"support_confirm", "stack_down", "release", "session_end"}:
                return 503, {"ok": False, "message": f"기록을 열지 못해 실행하지 않았습니다: {exc}"}
            record_warning = " · 기록 불가 — 회수 동작을 우선합니다"
            TRIAL.latch("기록 불가 — 회수 후 저장 장치 확인 필요")
        REQUESTS[request_id] = (fingerprint, (202, {"ok": True, "message": "동일 요청 처리 중 · 재실행하지 않음"}))
        while len(REQUESTS) > 256:
            REQUESTS.popitem(last=False)
        if immediate:
            if action in EVENTS | {"support_confirm"}:
                TRIAL.operator_event(action, payload)
                if DEMO:
                    DEMO_SOURCE.act(action, payload)
            elif action == "configure":
                TRIAL.mode, TRIAL.checkpoint = payload["mode"], payload["checkpoint"]
                TRIAL.completed.discard("selftest")
                TRIAL.revision += 1
                TRIAL.emit("configured", mode=TRIAL.mode, checkpoint=TRIAL.checkpoint)
            elif action == "new_trial":
                previous = TRIAL
                SESSION.event("trial_summary", **previous.snapshot())
                TRIAL = TrialState(previous.checkpoint)
                TRIAL.boot_id = previous.boot_id
                TRIAL.revision = previous.revision + 1
                TRIAL.number = previous.number + 1
                TRIAL.mode = previous.mode
                with TEL.lock:
                    TEL.bridge_fault = None
                    TEL.d["bridge"] = None
                    TEL.t["bridge"] = 0
                # Preserve old process log objects; don't reuse their faults/readiness.
                if not DEMO:
                    with RUN.lock:
                        RUN.procs = {k: p for k, p in RUN.procs.items() if p.alive}
                else:
                    DEMO_SOURCE = type(DEMO_SOURCE)()
                TRIAL.emit("new_trial")
            elif action == "session_end":
                TRIAL.completed.add("session_end")
                TRIAL.revision += 1
                SESSION.event("trial_summary", **TRIAL.snapshot())
                SESSION.close("운영자 회수 후 종료")
            else:
                TRIAL.revision += 1
            result = (200, {"ok": True, "message": LABELS[action] + " 완료" + record_warning})
            REQUESTS[request_id] = (fingerprint, result)
            console_snapshot()
            return result
        BUSY = True
        TRIAL.begin(action, now())
        pending = TRIAL.pending
        # API cannot silently change the selected policy or airborne flag.
        effective = {"checkpoint": TRIAL.checkpoint, "hang_arm": TRIAL.mode == "air"}
    try:
        if DEMO:
            ok, msg = DEMO_SOURCE.act(action, effective)
        elif action == "sensors" and not TRIAL._sensor_reason():
            ok, msg = True, "기존 센서 정상 수신 확인 — 중복 기동하지 않음"
        else:
            ok, msg = ACTIONS[action][0](effective)
    except Exception as exc:
        ok, msg = False, f"{LABELS[action]} 실행 실패: {exc}"
    finally:
        with STATE_LOCK:
            BUSY = False
    with STATE_LOCK:
        if TRIAL.pending is pending:
            if not ok or action == "can_up":
                TRIAL.finish(bool(ok), msg)
            else:
                pending["launched"] = True
        else:
            ok, msg = False, msg + " / 진행이 취소되었습니다. 잔존 프로세스·실제 상태를 확인하세요"
        SESSION.event("action_dispatched", action=action, ok=bool(ok), message=msg)
        result = (200, {"ok": bool(ok), "message": msg + (" · 실제 응답 대기" if ok and TRIAL.pending else "") + record_warning})
        REQUESTS[request_id] = (fingerprint, result)
        console_snapshot()
        return result


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, render_page(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._send(200, json.dumps(console_snapshot(), ensure_ascii=False, allow_nan=False))
        return self._send(404, '{"ok":false,"message":"not found"}')

    def do_POST(self):
        if self.path != "/api/action":
            return self._send(404, '{"ok":false,"message":"not found"}')
        # Reject cross-origin browser commands; this remains a trusted-LAN tool,
        # not an authenticated Internet service.
        from urllib.parse import urlsplit
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            return self._send(403, '{"ok":false,"message":"cross-origin request rejected"}')
        if self.headers.get_content_type() != "application/json":
            return self._send(415, '{"ok":false,"message":"JSON required"}')
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= 4096:
                raise ValueError("request size")
            payload = json.loads(self.rfile.read(n))
        except (ValueError, TypeError):
            return self._send(400, '{"ok":false,"message":"invalid request"}')
        code, result = dispatch_action(payload)
        return self._send(code, json.dumps(result, ensure_ascii=False))


def selftest():
    """Small deployment smoke test. No ROS, subprocesses, markers or real logs."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix="console-selftest-") as directory:
        with patch.dict(globals(), SESSION_DIR=directory):
            recorder = SessionLog()
            recorder.ensure("offline selftest", manual=True)
            recorder.event("check")
            recorder.sample(Telemetry().view())
            assert recorder.status()["samples"] == 1
            recorder.close("test complete")
        t = TrialState(CKPT_DEFAULT)
        assert t.reason("estop") is None
        assert t.reason("arm") is not None
        assert t.reason("release") is not None
        assert bridge_cmd_payload("stand") == {"cmd": "stand"}
        assert bridge_cmd_payload("stand", {"hang_arm": True})["hang_arm"] is True
        with patch.object(TEL, "publish", return_value=False) as publish:
            assert act_estop()[0] is False
            assert publish.call_count == 2
        page = render_page()
        assert "<!--CKPT_OPTIONS-->" not in page
        assert f"<option>{CKPT_DEFAULT}</option>" in page
        src = Path(__file__).with_name("rl_bridge_node.py").read_text(encoding="utf-8")
        body = src.split("CHECKPOINTS = {", 1)[1].split("\n}\n", 1)[0]
        known = set(re.findall(r'^\s+"([A-Za-z0-9_]+)": os\.path\.join', body, re.M))
        assert set(CKPT_CHOICES) <= known, "console checkpoint not in bridge"
    print("console selftest: ALL PASS (offline only; hardware not exercised)")
    return 0


def main():
    global DEMO, DEMO_SOURCE, SESSION_DIR
    ap = argparse.ArgumentParser(description="실기 운영 콘솔 — 로컬 모의 확인은 --demo")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="합성 데이터만 사용; ROS/CAN/명령 실행 없음")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    DEMO = a.demo
    if DEMO:
        try:
            from .console_demo import Demo
        except ImportError:
            from console_demo import Demo
        DEMO_SOURCE = Demo()
        a.host = "127.0.0.1"
        SESSION_DIR = tempfile.mkdtemp(prefix="humanoid-console-demo-")
        print(f"[DEMO] 실제 장비 연결 없음. 합성 시험 기록: {SESSION_DIR}", flush=True)
    else:
        TEL.start()
        poll_processes()
    stopped = threading.Event()

    def process_watch():
        while not stopped.wait(1.0):
            poll_processes()
    if not DEMO:
        threading.Thread(target=process_watch, daemon=True).start()

    def sampler():
        tick = 0
        while not stopped.wait(1.0 / SAMPLE_HZ):
            tick += 1
            try:
                if not DEMO and tick % int(SAMPLE_HZ) == 0:
                    before = CAN.view().get("busoff")
                    CAN.poll()
                    after = CAN.view().get("busoff")
                    if before is not None and after is not None and after > before:
                        SESSION.event("can_busoff", count=after, delta=after-before)
                view = console_snapshot()
                SESSION.sample(view)
            except Exception as exc:
                with STATE_LOCK:
                    TRIAL.latch(f"상태·기록 처리 오류: {type(exc).__name__}")
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True

    def bye(signum, _frame):
        print(f"\n신호 {signum} — 콘솔 종료. 장비 제어 자식 프로세스는 자동 종료하지 않습니다.", flush=True)
        stopped.set()
        SESSION.close(f"콘솔 종료(신호 {signum})")
        threading.Thread(target=srv.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, bye)
    print(f"운영 콘솔: http://{a.host}:{a.port}/  {'[DEMO]' if DEMO else '[LIVE / 신뢰된 내부망 전용]'}", flush=True)
    try:
        srv.serve_forever()
    finally:
        stopped.set()
        thread.join(timeout=5)
        SESSION.close("콘솔 종료")
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
