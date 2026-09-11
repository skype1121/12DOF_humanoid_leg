"""Synthetic console data only. No ROS, CAN, subprocess or marker-file access."""
import copy
import time


class Demo:
    def __init__(self):
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
            self.command = "SET_ALL"
            self.seq += 1
        elif action == "estop":
            self.running = False
            self.command = "STOP_ALL"
            self.seq += 1
        elif action == "stack_down":
            for key in ("bridge", "hang_stand", "stage8"):
                if key in self.procs:
                    self.procs[key].update(alive=False, rc=0)
            self.running = self.armed = self.holding = False
        elif action == "op_touchdown":
            self.contact = [True, True]
        elif action == "selftest":
            proc["all_pass"] = True
        elif action == "release":
            proc["release_ok"] = True
        key = {"baseline": "stage8_baseline", "arm": "stage8_arm"}.get(action, action)
        if action not in ("gain_ramp", "ground", "handoff", "estop", "stack_down") and not action.startswith("op_"):
            self.procs[key] = proc
        return True, "데모 합성 응답 — 장비 명령 없음"

    def snapshot(self):
        stamp = time.time()
        stage = self.procs.get("stage8", {}).get("alive", False)
        bridge = self.procs.get("bridge", {}).get("alive", False) and self.policy_started
        fresh = {"imu": self.sensors, "foot": self.sensors, "status": stage, "bridge": bridge}
        procs = copy.deepcopy(self.procs)
        if procs.get("hang_stand", {}).get("alive"):
            procs["hang_stand"]["hold"] = {"phase": self.phase, "kp": self.kp, "ts": stamp}
        joints = {f"joint_{i+1:02}": {"feedback_valid": True, "feedback_stale": False,
                  "latest_actual_deg": 0.2, "baseline_deg": 0 if self.baseline else None,
                  "commanded_relative_deg": 0, "desired_relative_deg": 0}
                  for i in range(12)}
        return {"demo": True, "ros": False, "server_time": stamp,
                "external_controls": [], "process_check_ok": True,
                "received": {k: stamp if v else 0 for k, v in fresh.items()}, "fresh": fresh,
                "imu": {"fwd": 1.2, "lat": -0.4, "tilt": 1.3, "gyro": [0.01, 0.02, 0], "age": 0} if self.sensors else None,
                "foot": {"left_n": 0, "right_n": 0, "pct": 0, "contact": list(self.contact) if self.sensors else None,
                         "pads_on": [int(x)*2 for x in self.contact], "raw_sum": [800, 900], "age": 0 if self.sensors else None},
                "stage8": {"armed": self.armed, "running": self.running, "waiting": self.waiting,
                           "holding": self.holding, "hold_joint_count": 12 if self.holding else 0,
                           "baseline": self.baseline, "feedback_ok": stage, "last_command": self.command,
                           "accepted": True, "command_seq": self.seq, "joints": joints if stage else {}},
                "bridge": {"live": bridge, "armed": self.running, "checkpoint": self.loaded if bridge else None,
                           "loaded_checkpoint": self.loaded if procs.get("bridge", {}).get("alive") else None,
                           "blend": 1 if self.running else None, "contact": list(self.contact), "fault": self.fault},
                "can": {"state": "DEMO", "busoff": 0, "rate60": 0}, "procs": procs, "events": []}
