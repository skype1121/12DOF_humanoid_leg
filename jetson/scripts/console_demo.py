"""Synthetic console data only. No ROS, CAN, subprocess or marker-file access."""
import copy
import os
import sys
import time

try:
    from .console_state import WALK_KEYS
except ImportError:
    from console_state import WALK_KEYS


def _w3_module():
    """합성 W3 세션이 쓰는 원본 모듈(rl_walking/deploy/w3_deploy) — 걷기 실험 키를 고른 데모에서만 지연 import."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from rl_walking.deploy import w3_deploy
    return w3_deploy


class DemoW3Bridge:
    """합성 W3 세션(1007, 데모·콘솔 selftest 전용 — 실물도 브리지 프로세스도 아니다).

    판정은 **실제** 감독기(w3_deploy.W3Supervisor: seq·정지 래치·켜기 선행조건·기한)와 **실제** 래치 클록(W3GaitClock)이 한다.
    여기서 합성하는 것은 조용한 양발 서기 관측, 걷는 동안의 한발 지지 접촉, 합성 러너의 모드 전환(켜기 다음 틱 w3, 끄기 뒤 조용한
    양발 서기 0.5 s 뒤 ke), 그리고 응답 봉투다(키는 브리지 _w3_ack·_w3_burst_end 와 같게 맞췄다).
    실제 브리지 노드 코드와의 대조는 tests/test_console_walk_bridge.py 가 한다 — 이 모형은 화면·상태기 확인용이다.
    """
    TICK = 0.02
    SETTLE_S = 0.5          # 합성 러너 W3 → KE 정착 시간(브리지 KEW3_SPECS 와 같은 값 — 여기서는 화면용 모사)
    LAND_S = 0.1            # 래치가 꺼진 뒤 양발 접촉까지의 착지 지연(브리지 노드 selftest 의 가정 모형과 같은 값)

    def __init__(self, composite, now, guard_mode="data"):
        W3 = _w3_module()
        self.composite, self.guard_mode = bool(composite), guard_mode
        self.sup = W3.W3Supervisor(cmd_limits=W3.W3S3D_CMD_LIMITS)
        self.clk = W3.W3GaitClock(W3.W3S3D_CONFIG)
        self.tick = 2                    # 정책이 이미 도는 중(래치 클록의 fresh 규칙 틱 ≤ 1 은 지났다)
        self.t = now                     # 합성 틱을 돌린 마지막 시각
        self.cmd = (0.0, 0.0, 0.0)
        self.active = True
        self.mode = "ke" if self.composite else None
        self.settle = 0.0
        self.n_to_w3 = self.n_to_ke = 0
        self.off_t = now - 10.0          # 마지막으로 래치가 꺼진 시각
        self.acks = []
        self.gyro = (0.02, 0.01, 0.0)    # 감독기에 넣는 합성 각속도(조용한 서기). 시험이 키워서 '정착 미달' 거부를 만든다
        self.drop_acks = 0               # 시험용: 다음 N 개 응답을 버린다(응답 유실 모의)
        self.deaf = False                # 시험용: 명령이 브리지에 닿지 않는다(명령 유실 모의)

    def contact(self, base, tt):
        if self.clk.ge:                  # 걷는 중 = 한발 지지(위상으로 교대)
            return (True, False) if float(self.clk.u) >= 0.5 else (False, True)
        if tt - self.off_t < self.LAND_S - 1e-9:
            return (True, False)
        return (bool(base[0]), bool(base[1]))

    def kew3(self):
        return {"mode": self.mode, "event": "", "n_to_w3": self.n_to_w3, "n_to_ke": self.n_to_ke, "ticks": 0,
                "want": bool(self.clk.want), "block": None, "since_latch_s": None, "settle_s": None}

    def _ack(self, env):
        env.update(active=bool(self.active), w3=self.clk.snapshot(), sup=self.sup.snapshot(), guard_mode=self.guard_mode)
        if self.composite:
            env["kew3"] = self.kew3()
        if self.drop_acks > 0:
            self.drop_acks -= 1
            return
        self.acks.append(env)

    def advance(self, now, base):
        """합성 정책 틱(50 Hz)을 now 까지 돌린다 — 브리지 _tick 과 같은 순서: 관측 → 기한 판정 → 러너(래치 클록)."""
        if now - self.t > 20.0:          # 오래 쉰 뒤에는 최근 20 s 만 모사
            self.t = now - 20.0
        while self.t + self.TICK <= now + 1e-9:
            self.t += self.TICK
            if not self.active:
                continue
            tt = self.t
            feet = self.contact(base, tt)
            self.sup.observe(tt, [1.0 if c else 0.0 for c in feet], self.gyro, (0.0, 0.0, -1.0))
            due = self.sup.burst_due(tt, bool(self.clk.want))
            if due is not None:          # 기한 도달 — 브리지가 스스로 끄기(명령 0·gait 0) + 같은 req_id 의 burst_end 응답
                rid, dur, _ran = due
                self.cmd = (0.0, 0.0, 0.0)
                self.clk.set_want(False)
                self._ack({"timestamp": tt, "cmd": "step", "event": "burst_end", "req_id": rid, "accepted": True,
                           "why": f"기한 {dur:g}s 경과 — 끄기(반주기 경계에서 서기)", "action": "set_want", "dur_s": dur})
            was_ge = bool(self.clk.ge)
            if self.composite and self.mode == "ke":
                if self.clk.want:        # 켜기 다음 틱에 W3 로(이 틱은 W3 STAND — 래치는 그다음 틱에 켜진다)
                    self.mode, self.settle = "w3", 0.0
                    self.n_to_w3 += 1
            else:
                self.clk.update(self.tick, self.cmd)
                if was_ge and not self.clk.ge:
                    self.off_t = tt
                if self.composite and not self.clk.want and not self.clk.ge:
                    self.settle = self.settle + self.TICK if feet == (True, True) else 0.0
                    if self.settle >= self.SETTLE_S - 1e-9:
                        self.mode, self.settle = "ke", 0.0
                        self.n_to_ke += 1
                        self.clk.reset()
                else:
                    self.settle = 0.0
            self.tick += 1

    def handle(self, d, now, base):
        """명령 한 건 — 브리지 _on_cmd_w3 와 같은 순서(감독기 판정 → 실행 → 기한 걸기/지우기 → 응답)."""
        self.advance(now, base)
        if self.deaf:
            return
        dec = self.sup.handle(d, now, active=self.active, warmed=True, blend_done=True, ge_now=bool(self.clk.ge),
                              want_now=bool(self.clk.want), cmd_now=tuple(self.cmd))
        act = dec["action"]
        if act == "estop":
            self.sup.on_frozen("estop")
            self.clk.set_want(False)
            self.active = False
        elif act == "stop":
            self.cmd = (0.0, 0.0, 0.0)
            self.clk.set_want(False)
        elif act == "set_want":
            self.cmd = tuple(dec["cmd"]) if dec["want"] else (0.0, 0.0, 0.0)
            self.clk.set_want(bool(dec["want"]))
        elif act == "set_cmd":
            self.cmd = tuple(dec["cmd"])
        if dec["accepted"]:
            if (act == "set_want" and dec["want"]) or act == "set_cmd":
                self.sup.burst_arm(now, d.get("req_id"), dec.get("dur_s"))
            elif act in ("set_want", "stop", "estop"):
                self.sup.burst_clear()
        self._ack({"timestamp": now, "cmd": d.get("cmd"), "req_id": d.get("req_id"), "seq": d.get("seq"),
                   "accepted": bool(dec["accepted"]), "why": str(dec["why"]), "action": act})

    def frame(self, now):
        out = {"w3": self.clk.snapshot(), "w3_sup": self.sup.snapshot(),
               "w3_cmd": {"lead": self.clk.last_lead, "hl_on": False, "hl": None},
               "guard_mode": self.guard_mode if self.guard_mode != "safe" else None, "ts": now, "cmd": list(self.cmd)}
        if self.composite:
            out["kew3"] = self.kew3()
        return out


class Demo:
    def __init__(self, clock=None):
        self.clock = clock or time.time     # 시험에서 가짜 시계를 넣을 수 있다(기본은 실제 시각)
        self.procs = {}
        self.sensors = False
        self.contact = [False, False]
        self.waiting = self.holding = self.armed = self.running = False
        self.baseline = 0
        self.command = None
        self.seq = 0
        self.phase = None
        self.kp = 20
        self.loaded = None
        self.policy_started = False
        self.fault = None
        self.start_gate = None        # 1005 착지 자동 인수(stand touchdown_start) 합성 상태
        self.w3 = None                # 1007 합성 W3 세션(걷기 실험 키로 정책을 시작했을 때만)

    def act(self, action, payload):
        proc = {"alive": False, "rc": 0, "lines": ["[DEMO] 합성 응답 — 실물 검증 아님"], "pid": None, "started": time.time()}
        if action == "sensors":
            self.sensors = True
        elif action == "stage8":
            self.waiting = True
            proc.update(alive=True, rc=None)
        elif action == "hang_stand":
            self.phase = "에어유지"
            proc.update(alive=True, rc=None)
        elif action == "gain_ramp":
            self.phase, self.kp = "운용대기", 150
        elif action == "ground":
            self.phase = "하중유지"
        elif action == "handoff":
            self.procs["hang_stand"].update(alive=False, rc=3)
            self.waiting, self.holding = False, True
        elif action == "baseline":
            self.baseline = 12
            self.command = "SET_BASELINE_FROM_CURRENT_ALL"
            self.seq += 1
        elif action == "arm":
            self.armed = True
            self.holding = False
            self.command = "ARM_ALL"
            self.seq += 1
        elif action == "bridge":
            self.loaded = payload["checkpoint"]
            proc.update(alive=True, rc=None, loaded_checkpoint=self.loaded)
        elif action == "stand":
            self.running = True
            self.policy_started = True
            if payload.get("touchdown_start"):
                self.start_gate = "POLICY" if self.contact == [True, True] else "WAIT_CONTACT"
            self.command = "SET_ALL"
            self.seq += 1
            if self.loaded in WALK_KEYS:     # 1007: 걷기 실험 키 = 합성 W3 세션(실제 감독기·래치 클록으로 판정)
                self.w3 = DemoW3Bridge(WALK_KEYS[self.loaded]["composite"], self.clock())
        elif action == "estop":
            self.running = False
            self.command = "STOP_ALL"
            self.seq += 1
            if self.w3 is not None:          # W3 세션의 estop = 즉시 동결 + 정지 래치(응답은 req_id 없이 온다)
                self.w3.handle({"cmd": "estop"}, self.clock(), self.contact)
        elif action == "stack_down":
            for key in ("bridge", "hang_stand", "stage8"):
                if key in self.procs:
                    self.procs[key].update(alive=False, rc=0)
            self.running = self.armed = self.holding = False
        elif action == "op_touchdown":
            self.contact = [True, True]
            if self.start_gate is not None:
                self.start_gate = "POLICY"
        elif action == "selftest":
            proc["all_pass"] = True
        elif action == "release":
            proc["release_ok"] = True
        key = {"baseline": "stage8_baseline", "arm": "stage8_arm"}.get(action, action)
        if action not in ("gain_ramp", "ground", "handoff", "estop", "stack_down") and not action.startswith("op_"):
            self.procs[key] = proc
        return True, "데모 합성 응답 — 장비 명령 없음"

    # ---- 1007 걷기 조작(합성 브리지 명령·응답) ----
    def _w3_live(self):
        """합성 W3 세션이 명령을 듣는 상태인가 — 브리지 프로세스가 살아 있고 정책이 시작됐고 착지 대기 중이 아닐 때."""
        return (self.w3 is not None and self.procs.get("bridge", {}).get("alive", False) and self.policy_started
                and self.start_gate in (None, "POLICY"))

    def bridge_cmd(self, cmd):
        """콘솔이 브리지 명령 토픽에 발행한 것과 같은 자리. 반환 = 발행 여부(듣는 브리지가 없어도 발행은 된다 → 응답 없음)."""
        if self._w3_live():
            self.w3.handle(dict(cmd), self.clock(), self.contact)
        return True

    def drain_acks(self):
        if self.w3 is None:
            return []
        acks, self.w3.acks = self.w3.acks, []
        stamp = self.clock()
        return [{"t": stamp, "ack": ack} for ack in acks]

    def snapshot(self):
        stamp = self.clock()
        stage = self.procs.get("stage8", {}).get("alive", False)
        bridge = self.procs.get("bridge", {}).get("alive", False) and self.policy_started
        fresh = {"imu": self.sensors, "foot": self.sensors, "status": stage, "bridge": bridge}
        procs = copy.deepcopy(self.procs)
        if procs.get("hang_stand", {}).get("alive"):
            procs["hang_stand"]["hold"] = {"phase": self.phase, "kp": self.kp, "ts": stamp}
        contact = list(self.contact)
        walk = {}
        if self._w3_live():
            self.w3.advance(stamp, self.contact)
            walk = self.w3.frame(stamp)
            if self.w3.active:
                contact = list(self.w3.contact(self.contact, stamp))     # 걷는 동안은 한발 지지가 화면에도 보인다
        # 접촉 시 '살아있는 채널 2개만' 눌린 합성값 — pads_on(=2) 과 어긋나지 않게 만든다
        pads = [[620, 540, 0, 0] if contact[0] else [0, 3, 0, 0],
                [700, 480, 0, 0] if contact[1] else [0, 0, 0, 0]]
        mask = [[v > 200 for v in row] for row in pads]
        joints = {f"joint_{i+1:02}": {"feedback_valid": True, "feedback_stale": False, "feedback_age_sec": 0.02,
                  "latest_actual_deg": 0.2, "baseline_deg": 0 if self.baseline else None,
                  "commanded_relative_deg": 0, "desired_relative_deg": 0}
                  for i in range(12)}
        return {"demo": True, "ros": False, "server_time": stamp,
                "external_controls": [], "process_check_ok": True,
                "received": {k: stamp if v else 0 for k, v in fresh.items()}, "fresh": fresh,
                "imu": {"fwd": 1.2, "lat": -0.4, "tilt": 1.3, "gyro": [0.01, 0.02, 0], "age": 0} if self.sensors else None,
                "foot": {"contact": contact if self.sensors else None,
                         "pads_on": [int(x)*2 for x in contact], "raw_sum": [800, 900], "age": 0 if self.sensors else None,
                         # 0913 합성: 채널별 카운트·눌림·죽은 채널. 0912 실물 모양을 본떠
                         # 왼[2]는 bias 판정제외(0), 왼[3]은 dead 후보로 둔다 (하중 아님).
                         "pads_counts": pads if self.sensors else None,
                         "pads_on_mask": mask if self.sensors else None,
                         "dead_mask": [[False, False, False, True], [False] * 4] if self.sensors else None,
                         "dead_note": ("dead 후보 L3 / 판정제외(bias) L2 — 계측용, 접촉 판정 불변"
                                       if self.sensors else None)},
                "stage8": {"armed": self.armed, "running": self.running, "waiting": self.waiting,
                           "holding": self.holding, "hold_joint_count": 12 if self.holding else 0,
                           "baseline": self.baseline, "feedback_ok": stage, "last_command": self.command,
                           "accepted": True, "command_seq": self.seq, "joints": joints if stage else {}},
                "bridge": {"live": bridge, "armed": self.running, "checkpoint": self.loaded if bridge else None,
                           "loaded_checkpoint": self.loaded if procs.get("bridge", {}).get("alive") else None,
                           "blend": 1 if self.running else None, "contact": contact, "fault": self.fault,
                           "start_gate": ({"state": self.start_gate} if self.start_gate else None), **walk},
                "can": {"state": "DEMO", "busoff": 0, "rate60": 0}, "procs": procs, "events": []}
