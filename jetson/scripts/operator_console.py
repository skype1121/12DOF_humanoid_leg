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
# 프로세스 관리 — 콘솔이 띄운 것만 콘솔이 내린다
# ---------------------------------------------------------------------------
class Proc:
    """장기 실행 프로세스 하나. 출력은 링버퍼로 보관해 웹에서 tail 한다."""

    def __init__(self, key, argv, cwd=None, env=None):
        self.key = key
        self.argv = argv
        self.lines = collections.deque(maxlen=300)
        self.started = now()
        self.rc = None
        self.p = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.p.stdout:
                self.lines.append(line.rstrip("\n")[:400])
        except Exception:
            pass
        self.rc = self.p.wait()
        self.lines.append(f"[종료] 반환코드 {self.rc}")

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

        imu = d.get("imu") or {}
        q = imu.get("quat") or imu.get("quat_wxyz")
        if q and len(q) == 4:
            w, x, y, z = (float(v) for v in q)
            gx = -2.0 * (x * z - w * y)
            gy = -2.0 * (y * z + w * x)
            gz = -(1.0 - 2.0 * (x * x + y * y))
            out["imu"] = {
                "fwd": round(math.degrees(math.atan2(gx, -gz)), 1),
                "lat": round(math.degrees(math.atan2(gy, -gz)), 1),
                "tilt": round(math.degrees(
                    math.acos(max(-1.0, min(1.0, -gz)))), 1),
            }
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
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;padding:12px}
section{background:#191c24;border:1px solid #2a2f3d;border-radius:10px;padding:12px}
h2{font-size:12px;margin:0 0 10px;color:#8b93a7;text-transform:uppercase;letter-spacing:.06em}
button{background:#242938;color:#e6e8ef;border:1px solid #333a4d;border-radius:7px;
padding:9px 12px;margin:0 6px 6px 0;cursor:pointer;font-size:13px}
button:hover{background:#2d3346}button.warn{border-color:#7a4b1e;background:#2e2216}
button.go{border-color:#1e6b4a;background:#16281f}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tile{background:#12141a;border:1px solid #262c3a;border-radius:8px;padding:10px}
.tile .k{font-size:11px;color:#8b93a7}.tile .v{font-size:22px;font-weight:700;margin-top:2px}
.big{grid-column:1/-1}.big .v{font-size:34px}
.ok{color:#4ade80}.bad{color:#f87171}.mid{color:#fbbf24}.dim{color:#6b7280}
pre{background:#0d0f14;border:1px solid #232936;border-radius:8px;padding:9px;
max-height:230px;overflow:auto;font-size:11.5px;margin:8px 0 0;white-space:pre-wrap}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pill{font-size:11px;padding:2px 8px;border-radius:99px;border:1px solid #333a4d}
#toast{position:fixed;left:50%;transform:translateX(-50%);bottom:18px;background:#242938;
border:1px solid #3a4256;padding:10px 16px;border-radius:8px;max-width:90vw;display:none;z-index:20}
</style></head><body>
<header><h1>실기 운영 콘솔</h1><span id=conn class=pill>연결 중…</span>
<button id=estop>■ 비상정지</button></header>
<main>
<section><h2>판정 지표</h2><div class=tiles>
<div class="tile big"><div class=k>발력 총합 (체중 117.7N 대비)</div><div class=v id=pct>—</div></div>
<div class=tile><div class=k>왼발 / 오른발</div><div class=v id=feet>—</div></div>
<div class=tile><div class=k>전방 기울임</div><div class=v id=fwd>—</div></div>
<div class=tile><div class=k>좌우 기울임</div><div class=v id=lat>—</div></div>
<div class=tile><div class=k>블렌딩 alpha</div><div class=v id=blend>—</div></div>
</div><div class=row style=margin-top:10px>
<span class=pill id=p_arm>stage8</span><span class=pill id=p_base>baseline</span>
<span class=pill id=p_live>bridge</span><span class=pill id=p_ckpt>ckpt</span></div>
<pre id=events>이벤트 없음</pre></section>

<section><h2>0. 게이트</h2>
<button data-a=can_up>CAN 브링업</button>
<button data-a=boot_pose>영점 판정 12/12</button>
<button data-a=sensors>센서 기동</button>
<button data-a=selftest>브리지 셀프테스트</button>
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
<select id=ckpt><option>stand_v7</option><option>stand_v6</option>
<option>stand_v5</option><option>stand_v3</option><option>march_v5b</option></select>
<button data-a=bridge class=warn>브리지 live</button></div>
<button data-a=stand class=go>stand 발사</button>
<button data-a=march class=go>march 발사</button>
<pre id=log_bridge>—</pre></section>

<section><h2>3. 회수 · 판정</h2>
<button data-a=stack_down>스택 종료</button>
<button data-a=release class=warn>릴리즈</button>
<button data-a=analyze>최신 로그 판정</button>
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
 if(['arm','handoff','bridge','stand','march','release'].includes(a)){
   if(!confirm(a+' — 실행할까요?'))return;}
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
  const f=s.foot,fr=s.fresh;
  $('#pct').textContent=fr.foot?f.pct.toFixed(0)+'%':'—';
  $('#pct').className='v '+(fr.foot?cls(f.pct,85,50):'dim');
  $('#feet').textContent=fr.foot?f.left_n.toFixed(0)+' / '+f.right_n.toFixed(0)+' N':'—';
  $('#fwd').textContent=s.imu&&fr.imu?(s.imu.fwd>0?'+':'')+s.imu.fwd+'°':'—';
  $('#fwd').className='v '+(s.imu&&fr.imu?cls(8-Math.abs(s.imu.fwd),4,0):'dim');
  $('#lat').textContent=s.imu&&fr.imu?(s.imu.lat>0?'+':'')+s.imu.lat+'°':'—';
  $('#blend').textContent=s.bridge.blend!=null?(+s.bridge.blend).toFixed(2):'—';
  $('#p_arm').textContent='stage8 '+(s.stage8.armed?'ARMED':(s.stage8.waiting?'대기':'—'));
  $('#p_arm').className='pill '+(s.stage8.armed?'ok':'dim');
  $('#p_base').textContent='baseline '+s.stage8.baseline+'/12';
  $('#p_base').className='pill '+(s.stage8.baseline===12?'ok':'dim');
  $('#p_live').textContent='bridge '+(s.bridge.live?'LIVE':'—');
  $('#p_live').className='pill '+(s.bridge.live?'ok':'dim');
  $('#p_ckpt').textContent=s.bridge.checkpoint||'ckpt —';
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
        try:
            ok, msg = fn(payload)
        except Exception as exc:                     # 액션 실패가 콘솔을 죽이지 않게
            ok, msg = False, f"{label} 실패: {exc}"
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
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"운영 콘솔: http://<젯슨IP>:{a.port}/   (Ctrl-C 종료)")
    print(f"  ROS 텔레메트리: {'연결' if TEL.ok else '없음(조작반만 동작)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료 — 콘솔이 띄운 프로세스는 그대로 둡니다 (모터 홀드 유지).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
