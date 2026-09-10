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
import re
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

BODY_WEIGHT_N = 172.9        # 실물 17.62kg(저울 0904) — 발력 총합 %의 분모. 어댑터 npc 0.02650(0906 재보정) 과 짝.
#: ⚠ 구 어댑터(npc 0.01804, 12kg 가정)와 짝지으면 완전 하중이 68% 로 나온다 — 착지 후 %가 70% 근처에서 안 오르면 구판 의심.
#:    백분율은 카운트 앵커 대비라 4차 "90% 자립" 은 유효(0905 카드의 "실제 61%" 는 분모만 바꾼 오류 — 철회).

#: 정책 드롭다운 — 첫 항목이 기본값(7차 데뷔 후보). 브리지 CHECKPOINTS 의 부분집합이어야 한다(selftest ⑧).
#: 0907: stand_v14 제거 — 패키지 GENS(v15~v18b)에 없어 젯슨에 파일이 없다(선택 시 로드 실패). 젯슨 구 자산 정리로
#: walk/march/drv2/march_drv2/march_v3/stand_v3/v5/v6 도 삭제됨 — 젯슨 잔존 = v15~v18b + v7/v8/v9/march_v5b.
CKPT_CHOICES = ["stand_hang_v5", "stand_v18b", "stand_hang_v4", "stand_hang_v3", "stand_hang_v2", "stand_hang_v1", "stand_v17", "stand_v15", "stand_v18", "stand_v16",
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
.flow .grp{margin:12px 0 4px;font-size:13px;font-weight:600;color:#c7cbd6}.flow .grp:first-of-type{margin-top:0}
.gd{font-weight:400;color:#8b93a7;font-size:11.5px;margin-left:6px}
.step{display:flex;align-items:center;gap:10px;padding:5px 8px;border-radius:8px;border:1px solid transparent}
.step.next{border-color:#3b82f6;background:#141a2b}
.step .n{width:22px;height:22px;border-radius:50%;background:#242938;color:#8b93a7;font-size:11px;display:flex;align-items:center;justify-content:center;flex:none}
.step.next .n{background:#3b82f6;color:#fff}
.step button{margin:0;min-width:150px}
.step .st{font-size:11px;padding:2px 9px;border-radius:99px;border:1px solid #333a4d;color:#8b93a7;flex:none}
.st.prog{color:#fbbf24;border-color:#7a5b1e}.st.ok{color:#4ade80;border-color:#1e6b4a}.st.run{color:#4ade80;border-color:#1e6b4a}.st.fail{color:#f87171;border-color:#7a1e2b}
.step .msg{font-size:11.5px;color:#8b93a7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chk{font-size:11px;color:#fbbf24}
input[type=number]{width:58px;background:#12141a;color:#e6e8ef;border:1px solid #333a4d;border-radius:5px;padding:4px;margin-right:4px}
button[title]:hover{outline:1px solid #3b82f6}
</style></head><body>
<header><h1>실기 운영 콘솔</h1><span id=conn class=pill>연결 중…</span>
<span id=sess class="pill dim">세션 —</span>
<button data-a=session_start style="padding:7px 12px;font-size:12px;margin:0">● 기록 시작</button>
<button data-a=session_end style="padding:7px 12px;font-size:12px;margin:0">■ 기록 중지</button>
<button id=estop>■ 비상정지</button></header>
<main>
<section class=full><h2>판정 지표</h2><div class=tiles>
<div class="tile big"><div class=k>발력 총합 · 체중 172.9N 대비</div><div class=v id=pct>—</div></div>
<div class=tile><div class=k>왼발 / 오른발 (N)</div><div class=v id=feet>—</div></div>
<div class=tile><div class=k>전방 기울임</div><div class=v id=fwd>—</div></div>
<div class=tile><div class=k>좌우 기울임</div><div class=v id=lat>—</div></div>
<div class=tile><div class=k>블렌딩 alpha</div><div class=v id=blend>—</div></div>
<div class=tile><div class=k>총 기울기</div><div class=v id=tilt>—</div></div>
</div>
<div class=row style="margin-top:10px;gap:16px;align-items:center">
<canvas id=tiltmap width=132 height=132 style="border:1px solid #2a2f3d;border-radius:10px;background:#0d0f14"></canvas>
<div style="display:flex;flex-direction:column;gap:6px">
<div style=font-size:13px><span class=dim>앞뒤 (X)</span> &nbsp;<b id=t_fwd style=font-size:17px>—</b></div>
<div style=font-size:13px><span class=dim>좌우 (Y)</span> &nbsp;<b id=t_lat style=font-size:17px>—</b></div>
<div style=font-size:13px><span class=dim>방위 (Z)</span> &nbsp;<b id=t_yaw style=font-size:17px>—</b></div>
<div style=font-size:11px;color:#8b93a7>IMU <b id=imu_age class=dim>—</b> · 흔들림 <b id=imu_gyro>—</b></div>
</div></div>
<div class=row style=margin-top:10px>
<span class=pill id=p_arm>stage8</span><span class=pill id=p_base>baseline</span>
<span class=pill id=p_live>bridge</span><span class=pill id=p_ckpt>ckpt</span>
<span class=pill id=p_contact>접촉</span><span class=pill id=p_hang>가드</span>
<span class=pill id=p_can>CAN</span></div>
<pre id=events style=max-height:96px>이벤트 없음</pre></section>

<section class=flow><h2>절차 — 위에서 아래로 순서대로 (다음 차례가 밝게 표시됨)</h2>
<div class=grp>0. 게이트 <span class=gd>전원·통신·영점·센서 — 하나라도 실패면 진행 금지</span></div>
<div class=step data-a=can_up><span class=n>1</span><button data-a=can_up title="peak_usb 드라이버 로드 + can1 1Mbps 기동 (부팅 후 1회, sudo)">CAN 브링업</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=sensors><span class=n>2</span><button data-a=sensors title="IMU·발압력 어댑터 기동 — 판정 지표 신선(초록) 확인. 이미 떠 있으면 다시 누르지 말 것(중복)">센서 기동</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=boot_pose><span class=n>3</span><button data-a=boot_pose title="제로게인 읽기로 12모터 영점 일치 판정(허용 8°). 홀드 중엔 누르지 말 것 — 같은 버스에 제로게인 프레임을 보내 홀드가 풀림">영점 판정 12/12</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=selftest><span class=n>4</span><button data-a=selftest title="브리지 셀프테스트(하드웨어맵·게이트·가드·학습 계약·정책 로드). 홀드 중엔 누르지 말 것">브리지 셀프테스트</button><span class=st>대기</span><span class=msg></span></div>

<div class=grp>1. 공중 홀드 <span class=gd>매달린 채 스탠드 자세를 kp150 으로 잡는다</span></div>
<div class=step data-a=stage8><span class=n>5</span><button data-a=stage8 class=warn title="stage8 제어 노드를 인수 대기 모드로 띄운다 (토크 없음, 핸드오프 때 넘겨받음)">stage8 선기동</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=hang_stand><span class=n>6</span><button data-a=hang_stand class=warn title="기립 스크립트: 현재각→스탠드 자세 kp20 에어램프(6 s) 후 유지. 트림 0">hang_stand 홀드</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=gain_ramp><span class=n>7</span><button data-a=gain_ramp class=warn title="kp20→kp150 8 s 램프 (전 모터 오차 3° 미만일 때만 수락). 반드시 착지 전 공중에서">게인 램프 kp150</button><span class=st>대기</span><span class=msg></span></div>

<div class=grp>2. 착지 · 인계 <span class=gd>윈치로 발끝 연직선에 평평하게 착지 → 발력 30~50% 에서 멈춘 뒤</span></div>
<div class=step data-a=ground><span class=n>8</span><button data-a=ground title="접지 선언: hang_stand 의 무부하 스톨가드 해제 (착지 후, 또는 공중 발사 절차 때는 인계 직전)">접지 선언</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=handoff><span class=n>9</span><button data-a=handoff class=warn title="hang_stand 가 릴리즈 없이 종료 → 모터는 현 자세·kp150 유지, stage8 이 인수 (접지 선언 + kp150 + stage8 대기 필요)">핸드오프</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=baseline><span class=n>10</span><button data-a=baseline title="stage8 베이스라인 12/12 캡처 (현재 각도를 기준점으로)">베이스라인 12/12</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=arm><span class=n>11</span><button data-a=arm class=warn title="ARM_ALL — stage8 이 명령 스트리밍 시작 (토크 유지 상태로 정책 대기)">ARM</button><span class=st>대기</span><span class=msg></span></div>

<div class=grp>3. 정책 <span class=gd>체크포인트 선택 → 브리지 → 발사. 공중 발사는 행잉 ARM 체크(접지 전 무접촉 120 s 허용)</span></div>
<div class=step data-a=bridge><span class=n>12</span><select id=ckpt title="배포된 정책 체크포인트"><!--CKPT_OPTIONS--></select> <button data-a=bridge class=warn title="선택한 정책으로 RL 브리지를 LIVE 로 띄운다 (stand 전까지 대기, 블랙박스 기록 시작)">브리지 live</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=stand><span class=n>13</span><button data-a=stand class=go title="stand 명령 발행 — 중력 게이트 통과 시 블렌딩 1.5 s 후 정책 제어 시작">stand 발사</button>
<label class=chk title="체크 시 접지 전 무접촉을 최대 120 s 허용(HangGuard). 매달린 채 발사할 때만"><input type=checkbox id=hang_arm> 행잉 ARM</label><span class=st>대기</span><span class=msg></span></div>

<div class=grp>4. 회수 <span class=gd>줄 인장 → 스택 종료 → 그 다음에만 릴리즈 (동시 실행 금지)</span></div>
<div class=step data-a=stack_down><span class=n>14</span><button data-a=stack_down title="브리지·stage8·hang_stand 종료 (콘솔이 띄운 것만, 명시 PID)">스택 종료</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=release><span class=n>15</span><button data-a=release class=warn title="12모터 2 s 게인 램프다운 → 제로게인 + MIT 해제(무토크). 스택이 살아 있으면 거부(exit 3)">릴리즈</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=session_end><span class=n>16</span><button data-a=session_end title="세션 기록(타임라인) 닫기">세션 종료</button><span class=st>대기</span><span class=msg></span></div>
<div class=step data-a=analyze><span class=n>17</span><button data-a=analyze title="최신 블랙박스 analyze_fall.py 판정(A~D 등급·토크표)">최신 로그 판정</button><span class=st>대기</span><span class=msg></span></div>
</section>

<section><h2>실행 로그 <span class=gd id=log_key>—</span></h2><pre id=log_main style=max-height:420px>—</pre>
<details style="margin-top:10px"><summary style="cursor:pointer;color:#8b93a7;font-size:12px">고급 (트림·영점·측정·march) — 절차 밖 도구</summary>
<div style="margin-top:8px">
<div class=gd style="margin:6px 0">트림 (hang_stand 홀드 중, 0.5°/s)</div>
<button data-a=trim_zero title="발목·힙·무릎 트림 전부 0 (Hanging-to-Stand 정책은 0 이 기본)">트림 0</button>
<button data-a=trim_preset class=warn title="v18b 계열 평형 자세 트림 a+12.5 h+6.4 k+10.4 — Hanging-to-Stand 정책엔 쓰지 않음">정책 평형 자세(v8)</button>
<span style=font-size:12px;color:#8b93a7> 수동: </span>
<input id=t_a type=number step=0.5 value=0 title="발목 [°]"><input id=t_h type=number step=0.5 value=0 title="힙 [°]"><input id=t_k type=number step=0.5 value=0 title="무릎 [°]">
<button data-a=trim title="입력한 발목/힙/무릎 트림 적용">적용</button>
<div class=gd style="margin:10px 0 6px">영점 재앵커 — 스택 내린 상태에서만. 배수 보정이 항상 우선, 재캡처는 기준 자세일 때만</div>
<button data-a=zero_preview title="현재 읽기 vs 앵커 비교 (쓰기 없음)">미리보기</button>
<button data-a=zero_multiple class=warn title="36°/10° 배수 프레임 시프트만 보정 (안전)">배수 보정</button>
<button data-a=zero_recapture class=warn title="현재 읽기를 새 영점으로 씀 — 가장 위험한 쓰기, 백업 자동">재캡처</button>
<div class=gd style="margin:10px 0 6px">측정 (정책 없이 stage8 만, ARM 후)</div>
<button data-a=zero_kin class=warn title="발목 ±3° 스윕으로 영점 오차 절편 추정">영점 기구학 검증</button>
<select id=pose title="자세 프로브 목표"><option value=policy_v8>정책 평형 자세(v8)</option><option value=policy_v7>정책 평형 자세(v7)</option><option value=knee_only>무릎만</option><option value=ankle_only>발목만</option><option value=zero>전 관절 0 복귀</option></select>
<button data-a=posture class=warn title="stage8 홀드만으로 목표 자세 진입">자세 진입</button>
<div class=gd style="margin:10px 0 6px">기타</div>
<button data-a=march class=go title="march 정책 발사 (보행 계열 체크포인트 전용)">march 발사</button>
<button data-a=disarm title="DISARM_ALL — stage8 스트리밍 중단(동결 홀드)">DISARM</button>
<button data-a=session_start title="세션 기록 수동 시작">기록 시작</button>
</div></details></section>
</main><div id=toast></div>
<script>
const $=s=>document.querySelector(s);
function toast(m,bad){const t=$('#toast');t.textContent=m;
t.style.borderColor=bad?'#c0243a':'#3a4256';t.style.display='block';
clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',4200);}
// 액션 → 감시할 프로세스 키. null = 즉시 응답형
const PROC={can_up:null,sensors:'sensors',boot_pose:'boot_pose',selftest:'selftest',
 stage8:'stage8',hang_stand:'hang_stand',gain_ramp:'hang_stand',ground:'hang_stand',handoff:'hang_stand',
 baseline:'stage8_baseline',arm:'stage8_arm',disarm:'stage8_disarm',bridge:'bridge',stand:null,march:null,
 stack_down:'stack_down',release:'release',analyze:'analyze',zero_preview:'zero',zero_multiple:'zero',zero_recapture:'zero',
 zero_kin:'zerokin',posture:'posture',trim:null,trim_preset:null,trim_zero:null,session_start:null,session_end:null,estop:null};
const DAEMON=new Set(['stage8','hang_stand','bridge']);
const FLOW=['can_up','sensors','boot_pose','selftest','stage8','hang_stand','gain_ramp','ground','handoff','baseline','arm','bridge','stand','stack_down','release','session_end','analyze'];
const pending={};   // action → {t0, key, pid0, phase}
const done={};      // action → 'ok'|'fail'|'run'
let lastKey=null, lastState=null;
function setSt(a,cls,txt,msg){
 document.querySelectorAll('.step[data-a="'+a+'"]').forEach(el=>{
  const s=el.querySelector('.st'); s.className='st '+cls; s.textContent=txt;
  const m=el.querySelector('.msg'); m.textContent=msg?msg.slice(0,90):''; m.title=msg||'';});
 if(cls==='ok'||cls==='run')done[a]=cls; else if(cls==='fail')done[a]='fail'; else if(cls==='idle')delete done[a];
 markNext();}
function markNext(){
 let next=null;
 for(const a of FLOW){ if(done[a]==='ok'||done[a]==='run')continue; next=a; break; }
 document.querySelectorAll('.step').forEach(el=>el.classList.toggle('next',el.dataset.a===next));}
async function act(a){
 const body={action:a};
 if(a==='bridge')body.checkpoint=$('#ckpt').value;
 if(a==='posture')body.pose=$('#pose').value;
 if(a==='trim'){body.a=+$('#t_a').value;body.h=+$('#t_h').value;body.k=+$('#t_k').value;}
 if(a==='stand'){body.hang_arm=$('#hang_arm').checked;
   if(body.hang_arm&&!confirm('행잉 ARM: 접지 전 무접촉 120 s 허용. 매달린 채 발사가 맞습니까?'))return;}
 if(a==='zero_recapture'){
   if(!confirm('영점 재캡처: 지금 자세가 기준 자세(수평 매달림·정렬)가 확실합니까? 현재 읽기가 새 앵커가 됩니다.'))return;
   if(!confirm('한 번 더: 잘못 잡으면 이후 영점 판정도 틀린 기준에 통과합니다. 진행?'))return;}
 else if(['arm','handoff','release','zero_multiple','posture','zero_kin','gain_ramp','trim','trim_preset','march'].includes(a)){ if(!confirm(a+' — 실행할까요?'))return;}
 const key=PROC[a]; const P=(lastState&&lastState.procs)||{};
 pending[a]={t0:Date.now(),key,pid0:(key&&P[key])?P[key].pid:null,phase:'sent'};
 if(key)lastKey=key;
 setSt(a,'prog','진행 중…','');
 try{const r=await fetch('/api/action',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json(); toast(j.message,!j.ok);
  if(!j.ok){setSt(a,'fail','실패',j.message);delete pending[a];return;}
  pending[a].msg=j.message;
  if(!key){ // 즉시 응답형
   if(a==='stand'||a==='march'){pending[a].phase='verify';}
   else{setSt(a,'ok','성공',j.message);delete pending[a];}
  }
 }catch(e){toast(''+e,true);setSt(a,'fail','실패',''+e);delete pending[a];}}
document.querySelectorAll('button[data-a]').forEach(b=>b.onclick=()=>act(b.dataset.a));
$('#estop').onclick=()=>act('estop');
function judge(s){ // 폴링마다 대기 중인 액션 판정
 const P=s.procs||{}; const now=Date.now();
 for(const a of Object.keys(pending)){
  const p=pending[a]; const dt=(now-p.t0)/1000;
  if(p.phase==='verify'){ // stand/march: 4 s 안에 브리지 armed
   if(s.bridge.live&&s.bridge.armed){setSt(a,'ok','성공','정책 제어 중 · ckpt '+s.bridge.checkpoint);delete pending[a];}
   else if(dt>6){setSt(a,'fail','실패','브리지가 armed 로 안 넘어감 — 중력 게이트/접촉/live 확인');delete pending[a];}
   continue;}
  const pr=P[p.key]; if(!pr)continue;
  const fresh=(p.pid0==null)||(pr.pid!==p.pid0)||['gain_ramp','ground','handoff'].includes(a);
  if(!fresh)continue;
  const last=(pr.lines||[]).filter(l=>l.trim()).slice(-1)[0]||'';
  if(a==='gain_ramp'){ const hit=(pr.lines||[]).some(l=>/운용대기|하중유지/.test(l)&&/kp150/.test(l));
   if(hit){setSt(a,'ok','성공','kp150 운용대기');delete pending[a];}
   else if((pr.lines||[]).some(l=>/거부/.test(l))||dt>25||!pr.alive){setSt(a,'fail','실패',last);delete pending[a];}
   continue;}
  if(a==='ground'){ setSt(a,'ok','성공','접지 선언됨 (스톨가드 해제)');delete pending[a];continue;}
  if(a==='handoff'){ if(!pr.alive&&(pr.rc===3||(pr.lines||[]).some(l=>/\[핸드오프\]/.test(l)))){setSt(a,'ok','성공','인계 완료 — 모터 kp150 유지, stage8 인수');delete pending[a];}
   else if((pr.lines||[]).some(l=>/핸드오프 거부/.test(l))||dt>15){setSt(a,'fail','실패',(pr.lines||[]).filter(l=>/핸드오프 거부/.test(l)).slice(-1)[0]||'시간 초과 — 접지 선언·kp150·stage8 대기 확인');delete pending[a];}
   continue;}
  if(pr.alive){ if(DAEMON.has(p.key)&&dt>4){setSt(a,'run','실행 중',last);delete pending[a];} else setSt(a,'prog','진행 중…',last); continue;}
  // 종료됨
  if(pr.rc===0||(p.key==='hang_stand'&&pr.rc===3)){
   const bad=(pr.lines||[]).some(l=>/불일치|FAIL|\[거부됨\]|실패/.test(l)&&!/0 ·|불량 0/.test(l));
   setSt(a,bad?'fail':'ok',bad?'실패':'성공',(pr.lines||[]).filter(l=>/\[OK\]|ALL PASS|완료|불일치|FAIL|거부|실패/.test(l)).slice(-1)[0]||last);
  } else setSt(a,'fail','실패 rc='+pr.rc,last);
  delete pending[a];
 }
 // 데몬 상태 반영(클릭 없이도): 죽으면 표시
 for(const a of ['stage8','hang_stand','bridge']){ const pr=P[PROC[a]]; if(!pr||pending[a])continue;
  if(pr.alive&&done[a]!=='run'&&done[a]!=='ok')setSt(a,'run','실행 중','');
  if(!pr.alive&&done[a]==='run'){ if(a==='hang_stand'&&pr.rc===3)setSt(a,'ok','인계 완료',''); else if(pr.rc===0)setSt(a,'idle','종료',''); else setSt(a,'fail','종료 rc='+pr.rc,(pr.lines||[]).slice(-1)[0]||''); }
 }}
function cls(v,g,m){return v>=g?'ok':(v>=m?'mid':'bad');}
function drawTilt(I,fresh){
 const c=$('#tiltmap'),x=c.getContext('2d'),W=c.width,H=c.height,cx=W/2,cy=H/2,R=W/2-10;
 x.clearRect(0,0,W,H);x.strokeStyle='#2a2f3d';x.lineWidth=1;
 [R,R*0.3].forEach(r=>{x.beginPath();x.arc(cx,cy,r,0,7);x.stroke();});
 x.beginPath();x.moveTo(cx-R,cy);x.lineTo(cx+R,cy);x.moveTo(cx,cy-R);x.lineTo(cx,cy+R);x.stroke();
 x.fillStyle='#6b7280';x.font='9px system-ui';x.textAlign='center';
 x.fillText('앞',cx,cy-R-2);x.fillText('뒤',cx,cy+R+8);x.textAlign='right';x.fillText('왼',cx-R-3,cy+3);x.textAlign='left';x.fillText('오',cx+R+3,cy+3);
 if(!I)return; const s=R/10, px=cx-(I.lat||0)*s, py=cy-(I.fwd||0)*s; const t=Math.hypot(I.fwd||0,I.lat||0);
 const col=!fresh?'#6b7280':(t<3?'#4ade80':(t<6?'#fbbf24':'#f87171'));
 x.strokeStyle=col;x.lineWidth=1.5;x.beginPath();x.moveTo(cx,cy);x.lineTo(px,py);x.stroke();x.fillStyle=col;x.beginPath();x.arc(px,py,5,0,7);x.fill();}
async function poll(){
 try{const s=await (await fetch('/api/state')).json(); lastState=s;
  $('#conn').textContent=s.ros?'ROS 연결됨':'ROS 없음';$('#conn').className='pill '+(s.ros?'ok':'bad');
  const S=s.session||{};
  $('#sess').textContent=S.active?('● '+S.name.replace('session_','')+'  '+Math.floor(S.dur/60)+'분 '+S.samples+'샘플'+(S.manual?' (수동)':'')):'기록 대기 (조작하면 자동 시작)';
  $('#sess').className='pill '+(S.active?'ok':'dim');
  const f=s.foot,fr=s.fresh;
  $('#pct').textContent=fr.foot?f.pct.toFixed(0)+'%':'—';$('#pct').className='v '+(fr.foot?cls(f.pct,85,50):'dim');
  $('#feet').textContent=fr.foot?f.left_n.toFixed(0)+' / '+f.right_n.toFixed(0)+' N':'—';
  const I=s.imu;
  $('#fwd').textContent=I?(I.fwd>0?'+':'')+I.fwd+'°':'—';$('#fwd').className='v '+(I?(fr.imu?cls(8-Math.abs(I.fwd),4,0):'dim'):'dim');
  $('#lat').textContent=I?(I.lat>0?'+':'')+I.lat+'°':'—';$('#lat').className='v '+(I&&fr.imu?'':'dim');
  $('#tilt').textContent=I?I.tilt+'°':'—';$('#tilt').className='v '+(I&&fr.imu?'':'dim');
  $('#imu_age').textContent=I&&I.age!=null?(fr.imu?'수신중':(I.age+'초 전 끊김')):'없음';$('#imu_age').className=fr.imu?'ok':'bad';
  const word=(v,pos,neg)=>Math.abs(v)<0.35?'거의 수평':(v>0?pos:neg)+' '+Math.abs(v).toFixed(1)+'°';
  const cl=v=>Math.abs(v)<3?'ok':(Math.abs(v)<6?'mid':'bad');
  if(I){$('#t_fwd').textContent=word(I.fwd,'앞으로','뒤로');$('#t_fwd').className=fr.imu?cl(I.fwd):'dim';
   $('#t_lat').textContent=word(I.lat,'왼쪽으로','오른쪽으로');$('#t_lat').className=fr.imu?cl(I.lat):'dim';
   $('#t_yaw').textContent=(I.yaw>0?'+':'')+(I.yaw!=null?I.yaw.toFixed(0):'—')+'°';$('#t_yaw').className=fr.imu?'':'dim';
   const g=I.gyro||[];const mag=Math.hypot(g[0]||0,g[1]||0,g[2]||0);$('#imu_gyro').textContent=mag<0.02?'정지':mag.toFixed(2)+' rad/s';}
  else{['t_fwd','t_lat','t_yaw'].forEach(k=>{$('#'+k).textContent='—';$('#'+k).className='dim';});}
  const C=s.can||{};
  if(C.busoff==null){$('#p_can').textContent='CAN —';$('#p_can').className='pill dim';}
  else{const bad=C.rate60>0;$('#p_can').textContent='CAN '+(C.state||'?')+(bad?('  ⚠ bus-off +'+C.rate60+'/60s'):('  bus-off '+C.busoff));$('#p_can').className='pill '+(bad?'bad':'ok');
   if(bad&&!window._canWarned){window._canWarned=1;toast('CAN bus-off 발생 중 — 모터 통신이 끊기고 있습니다. 즉시 중단하세요',true);} if(!bad)window._canWarned=0;}
  drawTilt(I,fr.imu);
  $('#blend').textContent=s.bridge.blend!=null?(+s.bridge.blend).toFixed(2):'—';
  $('#p_arm').textContent='stage8 '+(s.stage8.armed?'ARMED':(s.stage8.waiting?'대기':'—'));$('#p_arm').className='pill '+(s.stage8.armed?'ok':'dim');
  $('#p_base').textContent='baseline '+s.stage8.baseline+'/12';$('#p_base').className='pill '+(s.stage8.baseline===12?'ok':'dim');
  const brAlive=!!(s.procs&&s.procs.bridge&&s.procs.bridge.alive);
  $('#p_live').textContent='bridge '+(s.bridge.armed?'정책 제어 중':(s.bridge.live||brAlive?'LIVE 대기':'—'));$('#p_live').className='pill '+(s.bridge.armed?'ok':(brAlive?'mid':'dim'));
  $('#p_ckpt').textContent=s.bridge.checkpoint||('ckpt '+$('#ckpt').value);
  const c=s.bridge.contact;$('#p_contact').textContent='접촉 '+(c?('L'+(c[0]>0.5?'●':'○')+' R'+(c[1]>0.5?'●':'○')):'—');$('#p_contact').className='pill '+(c&&c[0]>0.5&&c[1]>0.5?'ok':'dim');
  const ev=s.events||[]; const lastEv=ev.length?ev[ev.length-1]:'';
  $('#p_hang').textContent=/stop_all|lift_guard|osc_guard/.test(lastEv)?'가드 정지: '+lastEv.slice(9,60):'가드 정상';$('#p_hang').className='pill '+(/stop_all|lift_guard|osc_guard/.test(lastEv)?'bad':'dim');
  $('#events').textContent=ev.length?ev.join('\\n'):'이벤트 없음';
  const P=s.procs||{}; const k=lastKey&&P[lastKey]?lastKey:Object.keys(P).sort((x,y)=>(P[x].age-P[y].age))[0];
  if(k&&P[k]){$('#log_key').textContent=k+' pid '+P[k].pid+(P[k].alive?' 실행중':' 종료 rc='+P[k].rc);$('#log_main').textContent=P[k].lines.join('\\n')||'(출력 없음)';}
  judge(s);
 }catch(e){$('#conn').textContent='콘솔 연결 끊김';$('#conn').className='pill bad';}
}
setInterval(poll,500);poll();markNext();
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
            return self._send(200, render_page(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            st = TEL.view()
            st["procs"] = RUN.snapshot()
            st["session"] = SESSION.status()
            st["can"] = CAN.view()
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


def _selftest_hang_arm():
    assert bridge_cmd_payload("stand") == {"cmd": "stand"}, "기본 = 종전과 동일 (플래그 없음)"
    assert bridge_cmd_payload("stand", {"hang_arm": False}) == {"cmd": "stand"}, "False 는 보내지 않는다"
    assert bridge_cmd_payload("stand", {"hang_arm": True}) == {"cmd": "stand", "hang_arm": True}
    assert 'id=hang_arm' in PAGE and "body.hang_arm" in PAGE, "페이지 체크박스·JS 누락"


def selftest():
    _selftest_hang_arm()
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
    got = round(100.0 * (86.45 + 86.45) / BODY_WEIGHT_N, 1)
    if got != 100.0:
        fails.append(f"발력 백분율 산식 오류: {got}")
    # ⑧ 드롭다운 정책은 전부 브리지 CHECKPOINTS 에 있어야 하고 기본값은 데뷔 후보 (0905 감사 A1:
    #    구판은 정적 목록 stand_v8~ 에 기본값 legacy stand_v7 — 10.49kg 자산 정책이 실물에 오르는 경로)
    try:
        bridge_py = os.path.join(SCRIPTS, "rl_bridge_node.py")
        if not os.path.isfile(bridge_py):   # 젯슨 밖(개발 PC)에선 같은 폴더의 브리지로 대조
            bridge_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_bridge_node.py")
        src = open(bridge_py, encoding="utf-8").read()
        body = src.split("CHECKPOINTS = {", 1)[1].split("\n}\n", 1)[0]
        known = set(re.findall(r'^\s+"([A-Za-z0-9_]+)": os\.path\.join', body, re.M))
        missing = [c for c in CKPT_CHOICES if c not in known]
        if missing:
            fails.append(f"드롭다운 정책이 브리지 CHECKPOINTS 에 없음: {missing}")
        if CKPT_DEFAULT != "stand_hang_v5":   # 0910: 실물 첫 후보 v5 (승윤님 확정 0909 밤)
            fails.append(f"기본 정책이 실물 첫 후보(stand_hang_v5, 승윤님 확정 0909)가 아님: {CKPT_DEFAULT}")
        page = render_page()
        _sel = page.index("<select id=ckpt"); _sel_end = page.index(">", _sel) + 1
        if "<!--CKPT_OPTIONS-->" in page or f"<option>{CKPT_DEFAULT}</option>" not in page \
                or page.index(f"<option>{CKPT_DEFAULT}</option>") != _sel_end:
            fails.append("페이지 드롭다운 렌더 실패 (기본값이 첫 항목이 아님)")
    except Exception as exc:
        fails.append(f"브리지 CHECKPOINTS 대조 실패: {exc}")
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
        n_tick = 0
        while True:
            time.sleep(period)
            n_tick += 1
            try:
                if n_tick % int(SAMPLE_HZ) == 0:      # 1Hz — CAN 통계는 무겁다
                    before = CAN.view().get("busoff")
                    CAN.poll()
                    after = CAN.view().get("busoff")
                    if (before is not None and after is not None
                            and after > before):
                        SESSION.event("can_busoff", count=after,
                                      delta=after - before)
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
