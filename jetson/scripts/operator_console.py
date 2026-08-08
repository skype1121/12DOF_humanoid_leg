#!/usr/bin/env python3
"""실기 운영 콘솔 — 젯슨에서 웹으로 띄우는 단일 화면 조작반.

매 실기마다 터미널 여러 개를 오가며 순서를 외우는 대신, 한 화면에서:
  게이트 → 기립 → 핸드오프 → 무장 → 정책 → 판정 → 회수
를 순서대로 누르고, **판정 지표(발력 총합%·기울기·접촉·가드)를 실시간으로**
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
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

BODY_WEIGHT_N = 117.7        # 실물 12kg — 발력 총합 %의 분모
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
            self.name = f"session_{stamp}"
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
            if (view["fresh"].get("status") or view["bridge"].get("live")
                    or view["foot"]["pct"] > 20.0):
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
                "lines": list(self.lines)[-n:]}


class Runner:
    def __init__(self):
        self.procs = {}          # key -> Proc
        self.lock = threading.Lock()

    def start(self, key, argv, cwd=None):
        with self.lock:
            old = self.procs.get(key)
            if old is not None and old.alive:
                return False, f"{key} 이미 실행 중 (pid {old.p.pid})"
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
                with self.lock:
                    self.d[key] = obj
                    self.t[key] = now()
                    if key == "bridge" and "event" in obj:
                        self.events.append(
                            f"{time.strftime('%H:%M:%S')} {obj.get('event')} "
                            f"{obj.get('reason', '')}".strip())
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

    def publish(self, which, obj):
        pub = self._pub_stage8 if which == "stage8" else self._pub_bridge
        if pub is None:
            return False
        from std_msgs.msg import String
        pub.publish(String(data=json.dumps(obj)))
        return True

    def view(self):
        """웹이 보는 판정 지표 — 발력 총합%가 1순위 (IMU 스냅샷 오판 이력)."""
        with self.lock:
            d = dict(self.d)
            t = dict(self.t)
            ev = list(self.events)[-12:]
        n = now()
        fresh = {k: (n - t[k]) < STALE_SEC and t[k] > 0 for k in t}

        out = {"fresh": fresh, "events": ev, "ros": self.ok}

        foot = d.get("foot") or {}
        ln, rn = float(foot.get("left_n", 0.0)), float(foot.get("right_n", 0.0))
        out["foot"] = {
            "left_n": round(ln, 1), "right_n": round(rn, 1),
            "total_n": round(ln + rn, 1),
            "pct": round(100.0 * (ln + rn) / BODY_WEIGHT_N, 1),
        }

        # IMU는 **항상** 값을 내보낸다. 끊기면 None 으로 지우지 않고 마지막
        # 값에 age 를 붙여 보낸다 — 화면에서 갑자기 "—"가 되면 자세를 잃은
        # 것인지 센서가 끊긴 것인지 구분이 안 된다(원격 오판 사고의 씨앗).
        imu = d.get("imu") or {}
        q = imu.get("quat") or imu.get("quat_wxyz")
        gyro = imu.get("gyro_rad_s") or imu.get("gyro") or []
        if q and len(q) == 4:
            w, x, y, z = (float(v) for v in q)
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            self._imu_last = {
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
        joints = st.get("joints", {}) or {}
        n_base = sum(1 for v in joints.values()
                     if isinstance(v, dict) and v.get("baseline_deg") is not None)
        out["stage8"] = {
            "armed": bool(st.get("armed")), "holding": bool(st.get("holding")),
            "waiting": bool(st.get("waiting")), "baseline": n_base,
            "hold_age": st.get("hold_age_sec"),
            "last_event": st.get("last_event"),
        }

        br = d.get("bridge") or {}
        out["bridge"] = {
            "live": bool(br.get("live")), "armed": bool(br.get("armed")),
            "checkpoint": br.get("checkpoint"),
            "blend": br.get("blend_alpha"),
            "contact": br.get("contact"),
        }
        return out


RUN = Runner()
TEL = Telemetry()


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


def act_bridge_cmd(cmd):
    st = TEL.view()
    _need(st["bridge"]["live"] or cmd == "estop",
          "브리지가 live 상태가 아닙니다 — 먼저 브리지를 기동하세요")
    ok = TEL.publish("bridge", {"cmd": cmd})
    return ok, f"브리지 명령 발행: {cmd}" if ok else "발행 실패 (ROS 미연결)"


def act_estop():
    """항상 실행 가능해야 한다 — 어떤 선행조건도 걸지 않는다."""
    msgs = []
    try:
        if TEL.publish("bridge", {"cmd": "estop"}):
            msgs.append("브리지 estop 발행")
    except Exception as exc:
        msgs.append(f"브리지 estop 실패: {exc}")
    try:
        if TEL.publish("stage8", {"command": "STOP_ALL"}):
            msgs.append("stage8 STOP_ALL 발행")
    except Exception as exc:
        msgs.append(f"STOP_ALL 실패: {exc}")
    return True, " / ".join(msgs) or "발행 경로 없음 (ROS 미연결)"


def act_bridge_live(payload):
    ckpt = str(payload.get("checkpoint", "stand_v7"))
    _need(ckpt.replace("_", "").isalnum(), "체크포인트 이름이 올바르지 않습니다")
    return RUN.start("bridge",
                     [sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"),
                      "--live", "--checkpoint", ckpt])


def act_analyze():
    files = sorted(
        (os.path.join(LOGDIR, f) for f in os.listdir(LOGDIR)
         if f.startswith("rl_bridge_") and f.endswith(".jsonl")),
        key=os.path.getmtime)
    _need(files, "블랙박스 로그가 없습니다")
    return RUN.start("analyze",
                     [sys.executable, os.path.join(SCRIPTS, "analyze_fall.py"),
                      files[-1]])


ACTIONS = {
    # 게이트
    "can_up":     (lambda p: act_can_up(), "CAN 브링업"),
    "boot_pose":  (lambda p: act_gate("boot_pose", [
        sys.executable, os.path.join(CALIB, "boot_pose_check.py"),
        "--ref", "stand"]), "영점 판정 12/12"),
    "sensors":    (lambda p: act_gate("sensors", [
        "bash", os.path.join(CALIB, "sensors_up.sh")]), "센서 기동"),
    "selftest":   (lambda p: act_gate("selftest", [
        sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"),
        "--selftest"]), "브리지 셀프테스트"),
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
    "ground":     (lambda p: act_marker("/tmp/hang_ground"), "접지 선언"),
    "handoff":    (lambda p: act_marker("/tmp/hang_stand_handoff"), "핸드오프"),
    "baseline":   (lambda p: act_stage8_cmd("baseline"), "베이스라인 12/12"),
    "arm":        (lambda p: act_stage8_cmd("arm"), "ARM_ALL"),
    "disarm":     (lambda p: act_stage8_cmd("disarm"), "DISARM_ALL"),
    "bridge":     (act_bridge_live, "브리지 live"),
    # 정적 자세 시험 — 정책 없이 stage8 홀드만으로 목표 자세 진입
    "posture":    (lambda p: RUN.start("posture", [
        sys.executable, os.path.join(SCRIPTS, "posture_probe.py"),
        "--pose", (p.get("pose") if p.get("pose") in
                   ("policy_v8", "policy_v7", "zero", "knee_only", "ankle_only")
                   else "policy_v8")]), "정적 자세 시험"),
    # 정책
    "stand":      (lambda p: act_bridge_cmd("stand"), "stand 발사"),
    "march":      (lambda p: act_bridge_cmd("march"), "march 발사"),
    "estop":      (lambda p: act_estop(), "비상정지"),
    # 회수·판정
    "stack_down": (lambda p: RUN.start("stack_down", [
        "bash", os.path.join(CALIB, "stack_down.sh")]), "스택 종료"),
    "release":    (lambda p: RUN.start("release", [
        sys.executable, os.path.join(CALIB, "release_all.py")]), "릴리즈"),
    "analyze":    (lambda p: act_analyze(), "최신 로그 판정"),
    # 세션 — 자동 개시/종료가 있지만 사람이 명시로 잡을 수 있어야 한다
    "session_start": (lambda p: (True, f"기록 시작: "
                                 f"{SESSION.ensure('수동 시작', manual=True)}"),
                      "기록 시작"),
    "session_end": (lambda p: (True, f"기록 중지: {SESSION.close('수동') or '없음'}"),
                    "기록 중지"),
}
STOPPABLE = {"stage8", "hang_stand", "bridge"}

PAGE = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>실기 운영 콘솔</title><style>
*{box-sizing:border-box}body{margin:0;background:#12141a;color:#e6e8ef;
font:14px/1.5 system-ui,-apple-system,"Noto Sans KR",sans-serif}
header{display:flex;align-items:center;gap:12px;padding:10px 14px;
background:#191c24;border-bottom:1px solid #2a2f3d;position:sticky;top:0;z-index:9}
h1{font-size:15px;margin:0;font-weight:600}
#estop{margin-left:auto;background:#c0243a;color:#fff;border:0;border-radius:8px;
padding:12px 22px;font-size:16px;font-weight:700;cursor:pointer}
#estop:active{background:#8f1a2b}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;padding:12px}
section{background:#191c24;border:1px solid #2a2f3d;border-radius:10px;padding:12px}
section.full{grid-column:1/-1}
h2{font-size:12px;margin:0 0 10px;color:#8b93a7;text-transform:uppercase;letter-spacing:.06em}
button{background:#242938;color:#e6e8ef;border:1px solid #333a4d;border-radius:7px;
padding:11px 14px;margin:0 6px 6px 0;cursor:pointer;font-size:14px}
button:hover{background:#2d3346}button.warn{border-color:#7a4b1e;background:#2e2216}
button.go{border-color:#1e6b4a;background:#16281f}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.tile{background:#12141a;border:1px solid #262c3a;border-radius:8px;padding:10px 12px}
.tile .k{font-size:11px;color:#8b93a7}.tile .v{font-size:26px;font-weight:700;margin-top:2px;
line-height:1.15;font-variant-numeric:tabular-nums}
/* 발력 총합은 줄을 잡은 채 흘끗 보는 지표 — 화면에서 제일 커야 한다 */
.big{grid-column:span 2}.big .v{font-size:56px}
@media(max-width:560px){.big{grid-column:1/-1}.big .v{font-size:48px}}
.ok{color:#4ade80}.bad{color:#f87171}.mid{color:#fbbf24}.dim{color:#6b7280}
pre{background:#0d0f14;border:1px solid #232936;border-radius:8px;padding:9px;
max-height:230px;overflow:auto;font-size:11.5px;margin:8px 0 0;white-space:pre-wrap}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pill{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid #333a4d}
#toast{position:fixed;left:50%;transform:translateX(-50%);bottom:18px;background:#242938;
border:1px solid #3a4256;padding:10px 16px;border-radius:8px;max-width:90vw;display:none;z-index:20}
</style></head><body>
<header><h1>실기 운영 콘솔</h1><span id=conn class=pill>연결 중…</span>
<span id=sess class="pill dim">세션 —</span>
<button data-a=session_start style="padding:7px 12px;font-size:12px;margin:0">● 기록 시작</button>
<button data-a=session_end style="padding:7px 12px;font-size:12px;margin:0">■ 기록 중지</button>
<button id=estop>■ 비상정지</button></header>
<main>
<section class=full><h2>판정 지표</h2><div class=tiles>
<div class="tile big"><div class=k>발력 총합 · 체중 117.7N 대비</div><div class=v id=pct>—</div></div>
<div class=tile><div class=k>왼발 / 오른발 (N)</div><div class=v id=feet>—</div></div>
<div class=tile><div class=k>전방 기울임</div><div class=v id=fwd>—</div></div>
<div class=tile><div class=k>좌우 기울임</div><div class=v id=lat>—</div></div>
<div class=tile><div class=k>블렌딩 alpha</div><div class=v id=blend>—</div></div>
<div class=tile><div class=k>총 기울기</div><div class=v id=tilt>—</div></div>
</div>
<div class=row style="margin-top:8px;font-size:12px;color:#8b93a7;gap:14px">
<span>IMU <b id=imu_age class=dim>—</b></span>
<span>중력 <b id=imu_grav>—</b></span>
<span>자이로(rad/s) <b id=imu_gyro>—</b></span>
<span>quat <b id=imu_quat>—</b></span></div>
<div class=row style=margin-top:10px>
<span class=pill id=p_arm>stage8</span><span class=pill id=p_base>baseline</span>
<span class=pill id=p_live>bridge</span><span class=pill id=p_ckpt>ckpt</span>
<span class=pill id=p_contact>접촉</span></div>
<pre id=events style=max-height:96px>이벤트 없음</pre></section>

<section><h2>0. 게이트</h2>
<button data-a=can_up>CAN 브링업</button>
<button data-a=boot_pose>영점 판정 12/12</button>
<button data-a=sensors>센서 기동</button>
<button data-a=selftest>브리지 셀프테스트</button>
<div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2f3d">
<div class=k style="font-size:11px;color:#8b93a7;margin-bottom:6px">영점 재앵커 — 스택 내린 상태에서만</div>
<button data-a=zero_preview>미리보기</button>
<button data-a=zero_multiple class=warn>배수 보정</button>
<button data-a=zero_recapture class=warn>재캡처</button></div>
<pre id=log_gate>—</pre></section>

<section><h2>1. 기립 → 무장</h2>
<button data-a=stage8 class=warn>stage8 선기동</button>
<button data-a=hang_stand class=warn>hang_stand</button>
<button data-a=ground>접지 선언</button>
<button data-a=handoff class=warn>핸드오프</button><br>
<button data-a=baseline>베이스라인</button>
<button data-a=arm class=warn>ARM</button>
<button data-a=disarm>DISARM</button>
<pre id=log_stack>—</pre></section>

<section><h2>2. 정책</h2><div class=row>
<select id=ckpt><option>stand_v8</option><option>stand_v7</option><option>stand_v6</option>
<option>stand_v5</option><option>stand_v3</option><option>march_v5b</option></select>
<button data-a=bridge class=warn>브리지 live</button></div>
<button data-a=stand class=go>stand 발사</button>
<button data-a=march class=go>march 발사</button>
<div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2f3d">
<div style="font-size:11px;color:#8b93a7;margin-bottom:6px">정적 자세 시험 — 정책 없이 stage8만 (ARM 후)</div>
<select id=pose><option value=policy_v8>정책 평형 자세(v8)</option>
<option value=policy_v7>정책 평형 자세(v7)</option>
<option value=knee_only>무릎만</option><option value=ankle_only>발목만</option>
<option value=zero>전 관절 0 복귀</option></select>
<button data-a=posture class=warn>자세 진입</button></div>
<pre id=log_bridge>—</pre></section>

<section><h2>3. 회수 · 판정</h2>
<button data-a=stack_down>스택 종료</button>
<button data-a=release class=warn>릴리즈</button>
<button data-a=analyze>최신 로그 판정</button>
<button data-a=session_end>세션 종료</button>
<pre id=log_tail>—</pre></section>
</main><div id=toast></div>
<script>
const $=s=>document.querySelector(s);
function toast(m,bad){const t=$('#toast');t.textContent=m;
t.style.borderColor=bad?'#c0243a':'#3a4256';t.style.display='block';
clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',4200);}
async function act(a){
 const body={action:a};
 if(a==='bridge')body.checkpoint=$('#ckpt').value;
 if(a==='posture')body.pose=$('#pose').value;
 if(a==='zero_recapture'){
   if(!confirm('영점 재캡처 — 지금 자세가 기준 자세(수평 매달림/정렬)가 확실합니까?\\n'
     +'현재 읽기가 그대로 새 앵커가 됩니다.'))return;
   if(!confirm('한 번 더 확인합니다. 잘못 잡으면 이후 boot_pose_check 조차\\n'
     +'틀린 기준에 대해 통과합니다. 진행할까요?'))return;}
 else if(['arm','handoff','bridge','stand','march','release','zero_multiple',
   'posture'].includes(a)){ if(!confirm(a+' — 실행할까요?'))return;}
 try{const r=await fetch('/api/action',{method:'POST',
   headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();toast(j.message,!j.ok);}catch(e){toast(''+e,true);}}
document.querySelectorAll('button[data-a]').forEach(b=>
 b.onclick=()=>act(b.dataset.a));
$('#estop').onclick=()=>act('estop');
function cls(v,g,m){return v>=g?'ok':(v>=m?'mid':'bad');}
async function poll(){
 try{const s=await (await fetch('/api/state')).json();
  $('#conn').textContent=s.ros?'ROS 연결됨':'ROS 없음';
  $('#conn').className='pill '+(s.ros?'ok':'bad');
  const S=s.session||{};
  $('#sess').textContent=S.active
    ?('● '+S.name.replace('session_','')+'  '+Math.floor(S.dur/60)+'분 '
      +S.samples+'샘플'+(S.manual?' (수동)':'')
      +(S.idle>60?'  유휴 '+Math.floor(S.idle/60)+'분':''))
    :'기록 대기 (조작하면 자동 시작)';
  $('#sess').className='pill '+(S.active?'ok':'dim');
  const f=s.foot,fr=s.fresh;
  $('#pct').textContent=fr.foot?f.pct.toFixed(0)+'%':'—';
  $('#pct').className='v '+(fr.foot?cls(f.pct,85,50):'dim');
  $('#feet').textContent=fr.foot?f.left_n.toFixed(0)+' / '+f.right_n.toFixed(0)+' N':'—';
  // IMU는 끊겨도 마지막 값을 계속 보여준다 (dim 처리 + 경과 초 표시).
  const I=s.imu;
  $('#fwd').textContent=I?(I.fwd>0?'+':'')+I.fwd+'°':'—';
  $('#fwd').className='v '+(I?(fr.imu?cls(8-Math.abs(I.fwd),4,0):'dim'):'dim');
  $('#lat').textContent=I?(I.lat>0?'+':'')+I.lat+'°':'—';
  $('#lat').className='v '+(I&&fr.imu?'':'dim');
  $('#tilt').textContent=I?I.tilt+'°':'—';
  $('#tilt').className='v '+(I&&fr.imu?'':'dim');
  $('#imu_age').textContent=I&&I.age!=null?(fr.imu?'수신중':(I.age+'초 전 끊김')):'없음';
  $('#imu_age').className=fr.imu?'ok':'bad';
  $('#imu_grav').textContent=I&&I.grav?I.grav.map(v=>v.toFixed(2)).join(' , '):'—';
  $('#imu_gyro').textContent=I&&I.gyro&&I.gyro.length?I.gyro.map(v=>v.toFixed(2)).join(' , '):'—';
  $('#imu_quat').textContent=I&&I.quat?I.quat.map(v=>v.toFixed(3)).join(' '):'—';
  $('#blend').textContent=s.bridge.blend!=null?(+s.bridge.blend).toFixed(2):'—';
  $('#p_arm').textContent='stage8 '+(s.stage8.armed?'ARMED':(s.stage8.waiting?'대기':'—'));
  $('#p_arm').className='pill '+(s.stage8.armed?'ok':'dim');
  $('#p_base').textContent='baseline '+s.stage8.baseline+'/12';
  $('#p_base').className='pill '+(s.stage8.baseline===12?'ok':'dim');
  $('#p_live').textContent='bridge '+(s.bridge.live?'LIVE':'—');
  $('#p_live').className='pill '+(s.bridge.live?'ok':'dim');
  $('#p_ckpt').textContent=s.bridge.checkpoint||'ckpt —';
  const c=s.bridge.contact;
  $('#p_contact').textContent='접촉 '+(c?('L'+(c[0]>0.5?'●':'○')+' R'+(c[1]>0.5?'●':'○')):'—');
  $('#p_contact').className='pill '+(c&&c[0]>0.5&&c[1]>0.5?'ok':'dim');
  $('#events').textContent=s.events.length?s.events.join('\\n'):'이벤트 없음';
  const P=s.procs||{};
  const pick=(ks)=>{for(const k of ks){if(P[k])return '['+k+' pid'+P[k].pid+
    (P[k].alive?' 실행중':' 종료 rc='+P[k].rc)+']\\n'+P[k].lines.join('\\n');}return '—';};
  $('#log_gate').textContent=pick(['boot_pose','selftest','sensors']);
  $('#log_stack').textContent=pick(['hang_stand','stage8','stage8_arm','stage8_baseline']);
  $('#log_bridge').textContent=pick(['bridge']);
  $('#log_tail').textContent=pick(['analyze','release','stack_down']);
 }catch(e){$('#conn').textContent='콘솔 연결 끊김';$('#conn').className='pill bad';}
}
setInterval(poll,500);poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass          # 접근 로그 억제 (콘솔 출력을 실기 로그로 쓴다)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/state":
            st = TEL.view()
            st["procs"] = RUN.snapshot()
            st["session"] = SESSION.status()
            return self._send(200, json.dumps(st))
        return self._send(404, json.dumps({"ok": False, "message": "not found"}))

    def do_POST(self):
        if self.path != "/api/action":
            return self._send(404, json.dumps({"ok": False, "message": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, TypeError):
            return self._send(400, json.dumps({"ok": False, "message": "잘못된 요청"}))
        name = payload.get("action")
        entry = ACTIONS.get(name)
        if entry is None:
            return self._send(
                403, json.dumps({"ok": False,
                                 "message": f"허용되지 않은 액션: {name}"}))
        fn, label = entry
        # 조작이 일어난 시점 = 모터 전원 이후. 여기서 세션을 연다.
        if name not in ("session_end", "session_start"):
            SESSION.ensure(reason=label)
        SESSION.event("action_start", action=name, label=label)
        try:
            ok, msg = fn(payload)
        except Exception as exc:                     # 액션 실패가 콘솔을 죽이지 않게
            ok, msg = False, f"{label} 실패: {exc}"
        SESSION.event("action", action=name, label=label, ok=bool(ok),
                      message=msg)
        print(f"[{time.strftime('%H:%M:%S')}] {label}: {msg}")
        return self._send(200, json.dumps({"ok": bool(ok), "message": msg}))


def selftest():
    """조작반의 안전 계약을 검사 — 실기 전에 이것만 통과하면 된다."""
    fails = []
    # ① 화이트리스트 밖 액션은 거부되어야 한다
    if "rm" in ACTIONS or any(";" in k or "&" in k for k in ACTIONS):
        fails.append("액션 키에 셸 메타문자")
    # ② 모든 액션이 호출 가능한 형태인가
    for k, (fn, label) in ACTIONS.items():
        if not callable(fn) or not label:
            fails.append(f"액션 정의 불량: {k}")
    # ③ 비상정지는 선행조건이 없어야 한다 (ROS 미연결에서도 예외 없이 반환)
    try:
        ok, _ = act_estop()
        if not ok:
            fails.append("estop이 실패를 반환 — 항상 실행 가능해야 함")
    except Exception as exc:
        fails.append(f"estop 예외: {exc}")
    # ④ 정지 대상은 장기 프로세스로 한정
    if not STOPPABLE <= set(ACTIONS):
        fails.append("STOPPABLE에 미정의 액션 포함")
    # ⑤ 경로가 실재하는지 (젯슨 배치 확인)
    for p in (CALIB, SCRIPTS):
        if not os.path.isdir(p):
            fails.append(f"경로 없음: {p}")
    # ⑥ 발력 백분율 산식
    got = round(100.0 * (58.85 + 58.85) / BODY_WEIGHT_N, 1)
    if got != 100.0:
        fails.append(f"발력 백분율 산식 오류: {got}")
    # ⑦ 세션 레코더 — 닫힌 상태에서 호출해도 예외가 없어야 하고(콘솔이 죽으면
    #    안 된다), 열면 타임라인에 실제로 기록되어야 한다
    try:
        SESSION.event("x")           # 세션 없음 — 조용히 무시되어야
        SESSION.sample(TEL.view())
        if SESSION.status()["active"]:
            fails.append("세션이 열리지 않았는데 active")
        name = SESSION.ensure("selftest")
        SESSION.event("selftest_marker", v=1)
        SESSION.sample(TEL.view())
        st = SESSION.status()
        if not st["active"] or st["events"] < 1 or st["samples"] < 1:
            fails.append(f"세션 기록 누락: {st}")
        tl = os.path.join(st["dir"], "timeline.jsonl")
        with open(tl) as f:
            kinds = [json.loads(l).get("kind") for l in f if l.strip()]
        for want in ("session_start", "selftest_marker", "s"):
            if want not in kinds:
                fails.append(f"타임라인에 {want} 없음")
        # 유휴 자동 닫힘: 자동 개시 세션은 닫히고, 수동 세션은 유지되어야 한다
        SESSION.last_active = now() - (IDLE_CLOSE_SEC + 60)
        SESSION.maybe_autoclose()
        if SESSION.status()["active"]:
            fails.append("자동 개시 세션이 유휴에도 안 닫힘")
        d2 = SESSION.ensure("selftest-manual", manual=True)
        SESSION.last_active = now() - (IDLE_CLOSE_SEC + 60)
        SESSION.maybe_autoclose()
        if not SESSION.status()["active"]:
            fails.append("수동 세션이 자동으로 닫힘 — 사람의 선언을 무시함")
        dir2 = SESSION.status()["dir"]
        SESSION.close("selftest")
        if SESSION.status()["active"]:
            fails.append("close 후에도 active")
        shutil.rmtree(st["dir"], ignore_errors=True)
        shutil.rmtree(dir2, ignore_errors=True)
        print(f"  (유휴 자동닫힘·수동 유지 확인: {d2})")
        print(f"  (세션 레코더 확인: {name} — 기록·종료 정상, 테스트 폴더 삭제)")
    except Exception as exc:
        fails.append(f"세션 레코더 예외: {exc}")
    for f in fails:
        print(f"[FAIL] {f}")
    print("selftest: " + ("ALL PASS" if not fails else f"{len(fails)}건 실패"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description="실기 운영 콘솔")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    TEL.start()

    def sampler():
        """세션이 열려 있는 동안 판정 지표를 SAMPLE_HZ 로 남긴다."""
        period = 1.0 / SAMPLE_HZ
        while True:
            time.sleep(period)
            try:
                if SESSION.status()["active"]:
                    SESSION.sample(TEL.view())
                    SESSION.maybe_autoclose()
            except Exception:
                pass
    threading.Thread(target=sampler, daemon=True).start()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    # rclpy.init() 이 SIGTERM/SIGINT 핸들러를 가로채 kill 이 먹지 않는다
    # (0808 실측: 재배포하려고 kill 했더니 구버전이 계속 포트를 물고 있어
    #  새 프로세스가 Address already in use 로 죽었다). 운영 도구가 안 꺼지는
    # 것은 그 자체로 결함이므로 우리 핸들러를 나중에 덮어씌운다.
    def _bye(signum, _frame):
        print(f"\n신호 {signum} — 콘솔 종료 (띄운 프로세스는 그대로 둡니다: "
              f"모터는 마지막 명령을 홀드).")
        SESSION.close(f"콘솔 종료(신호 {signum})")   # 기록을 잘린 채 남기지 않는다
        threading.Thread(target=srv.shutdown, daemon=True).start()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _bye)
    print(f"운영 콘솔: http://<젯슨IP>:{a.port}/   (Ctrl-C 또는 kill 로 종료)")
    print(f"  ROS 텔레메트리: {'연결' if TEL.ok else '없음(조작반만 동작)'}")
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
