"""Hardware-free operator workflow. A request is never an acknowledgement.

This module only inspects snapshots; it never starts a process or sends a command.
The console samples it independently of browser connections.
"""
import math
import time
import uuid


LABELS = {
    "can_up": "CAN 준비", "sensors": "센서 기동", "boot_pose": "영점 확인",
    "selftest": "정책 계약 검사", "stage8": "인수 제어기 준비",
    "hang_stand": "공중 자세 유지 시작", "gain_ramp": "운용 게인으로 전환",
    "ground": "접지 확인", "handoff": "제어 인계", "baseline": "기준점 확인",
    "arm": "제어기 ARM", "bridge": "선택 정책 로드", "stand": "정책 제어 시작",
    "support_confirm": "회수 줄 지지 확인", "stack_down": "제어 프로세스 종료",
    "release": "토크 해제", "session_start": "기록 시작", "session_end": "기록 종료",
    "op_touchdown": "착지 기록", "op_release_start": "방출 시작 기록",
    "op_release_stop": "방출 정지 기록", "op_retension": "재인장 기록",
    "op_slack_done": "모든 줄 느슨·무보조 확인", "op_assist": "외부 보조 기록",
    "op_note": "메모 기록", "analyze": "이 시험 로그 분석", "estop": "정지 요청",
    "configure": "시험 설정", "new_trial": "새 시험 준비",
}
PREP = ("can_up", "sensors", "boot_pose", "selftest", "stage8", "hang_stand", "gain_ramp")
CONTROL = ("stage8", "hang_stand", "bridge", "stage8_arm", "stage8_baseline")
EVENTS = {k for k in LABELS if k.startswith("op_")}
RECOVERY = {"support_confirm", "stack_down", "release", "session_end", "analyze"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class TrialState:
    def __init__(self, checkpoint, clock=time.monotonic):
        self.clock = clock
        self.boot_id = uuid.uuid4().hex
        self.revision = 0
        self.number = 1
        self.mode = "ground"
        self.checkpoint = checkpoint
        self.completed = set()
        self.pending = None
        self.fault = None
        self.last_result = None
        self.support = False
        self.stop_request_at = None
        self.stop_request_seq = -1
        self.stop_delivery = None
        self.measurement = {"state": "waiting", "elapsed": 0.0, "reason": "모든 줄 느슨·무보조 확인 전"}
        self.slack_at = None
        self.max_elapsed = 0.0
        self.view = {}
        self.history = []
        self.changes = []
        self._pose_stamps = None

    def emit(self, event, **data):
        row = {"event": event, "trial": self.number, **data}
        self.changes.append(row)
        self.history.append(row)
        self.history = self.history[-60:]

    def drain_events(self):
        rows, self.changes = self.changes, []
        return rows

    def flow(self):
        # The legacy ground marker also enables an airborne handoff. It is an
        # explicit operator confirmation here, never a fabricated sensor reading.
        return PREP + ("ground", "handoff", "baseline", "arm", "bridge", "stand")

    def active_processes(self):
        return [k for k in CONTROL if self.view.get("procs", {}).get(k, {}).get("alive")]

    def invalidate(self, reason):
        if self.slack_at is not None and self.measurement["state"] == "measuring":
            self.max_elapsed = max(self.max_elapsed, self.clock() - self.slack_at)
            self.measurement = {"state": "interrupted", "elapsed": round(self.max_elapsed, 1),
                                "reason": reason, "review_required": True}
            self.slack_at = None
            self.emit("measurement_interrupted", reason=reason, elapsed=round(self.max_elapsed, 1))

    FAULT_HINTS = (   # 0911 R8: 정지 사유 원문 → 운영 문구 (부분 일치, 앞에서부터)
        ("sensor_stale_warmup_timeout", "인계 실패 — 관절 피드백 미확보. 브리지 로그의 낡은 관절 목록/status 나이 확인 후 CAN·stage8 점검, 재명령"),
        ("sensor_stale", "센서 신선도 정지 — IMU/발압력/관절 피드백 공백. 원인 확인 전 재개 금지"),
        ("lift_guard", "양발 무접촉 정지 — 접촉 비트·패드·영상으로 '실제 리프트'인지 '센서'인지 판별"),
        ("hang_guard", "행잉 허용 시간 초과 정지"),
        ("osc_guard", "발진 가드 정지"),
        ("nonfinite", "비유한 관측 정지 — 센서 점검"),
        ("STOP_ALL", "제어기가 STOP_ALL 수락 — 목표 동결(토크 유지), 정지 사유 확인"),
    )

    def latch(self, reason):
        if not self.fault:
            self.fault = reason
            self.fault_hint = next((h for k, h in self.FAULT_HINTS if k in str(reason)), None)
            self.revision += 1
            self.emit("fault", reason=reason, hint=self.fault_hint)
        self.invalidate(reason)

    def _sensor_reason(self, need_status=False):
        fresh = self.view.get("fresh", {})
        for key, name in (("imu", "IMU"), ("foot", "발 접촉 센서")):
            if not fresh.get(key):
                return f"{name} 정상 수신 확인 필요"
        if need_status:
            if not fresh.get("status") or not self.view.get("stage8", {}).get("feedback_ok"):
                return "제어기·12관절 피드백 확인 필요"
        return None

    def _contact_reason(self):
        if not self.view.get("fresh", {}).get("foot"):
            return "발 접촉 센서 상태 불명"
        if self.view.get("foot", {}).get("contact") != [True, True]:
            return "양발 접촉 확인 필요 (하중 비율은 사용하지 않음)"
        return None

    def reason(self, action):
        if action == "estop":
            return None
        if action not in LABELS:
            return "현재 기립 시험에서 제공하지 않는 동작입니다"
        if self.pending and action not in (EVENTS - {"op_slack_done"}) | {"support_confirm"}:
            return f"{LABELS[self.pending['action']]} 응답 확인 중"
        s, b = self.view.get("stage8", {}), self.view.get("bridge", {})
        fresh = self.view.get("fresh", {})
        owners = self.active_processes()
        external = (bool(self.view.get("external_controls"))
                    or (fresh.get("status") and "stage8" not in owners)
                    or (fresh.get("bridge") and "bridge" not in owners))
        if action == "support_confirm":
            return None
        if action in EVENTS:
            if not self.view.get("session", {}).get("active"):
                return "기록 시작 필요"
            if action == "op_slack_done":
                if self.fault:
                    return "정지·고장 후에는 회수하고 새 시험을 시작하세요"
                if getattr(self, "releasing", False):
                    return "방출 정지(⏸)를 먼저 기록하세요 — 윈치 조작 중에는 느슨 확인 불가"
                if "stand" not in self.completed or not fresh.get("bridge") or not b.get("armed"):
                    return "정책 제어 중인 상태 확인 필요"
                if self.slack_at is not None or self.measurement["state"] != "waiting":
                    return "이미 측정한 시험입니다. 회수 후 새 시험을 시작하세요"
                return self._sensor_reason(True) or self._contact_reason()
            return None
        if action == "session_start":
            return "이미 기록 중입니다" if self.view.get("session", {}).get("active") else None
        if action == "session_end":
            if owners or (fresh.get("status") and (s.get("armed") or s.get("holding"))):
                return "회수·토크 해제 후 기록을 종료하세요"
            return None if "release" in self.completed or not self.completed.intersection(CONTROL) else "토크 해제 확인 필요"
        if action == "stack_down":
            return None if self.support else "먼저 회수 줄이 로봇을 지지함을 확인하세요"
        if action == "release":
            if not self.support or "stack_down" not in self.completed:
                return "줄 지지 확인 → 제어 프로세스 종료 순서가 필요합니다"
            if owners or external or (fresh.get("status") and (s.get("armed") or s.get("holding"))):
                return "제어 프로세스 종료 확인 중 — 토크 해제 불가"
            if not self.view.get("process_check_ok"):
                return "잔존 제어 프로세스 확인이 오래됐거나 실패했습니다"
            return "이미 토크 해제했습니다" if "release" in self.completed else None
        if action == "analyze":
            return None if "release" in self.completed else "회수 후 분석할 수 있습니다"
        if action == "new_trial":
            return None if "release" in self.completed and not owners and not external else "회수·토크 해제 완료 후 새 시험을 준비하세요"
        if self.fault:
            return "정지·고장 기록이 남아 있습니다. 회수 후 새 시험을 시작하세요"
        if self.support:
            return "회수 중입니다. 회수 완료 후 새 시험을 준비하세요"
        if external:
            return "콘솔 추적 밖 제어기 감지 — 기존 제어 상태를 확인하세요"
        if action == "configure":
            return "제어 시작 후에는 설정·영점을 바꿀 수 없습니다" if owners or self.completed.intersection(CONTROL) else None
        if action in self.completed:
            return "이 단계는 완료됐습니다"
        if action not in self.flow():
            return "현재 단계에서 허용되지 않는 동작입니다"
        if not self.view.get("process_check_ok"):
            return "잔존 제어 프로세스 확인 필요"
        preceding = self.flow()[:self.flow().index(action)]
        missing = next((k for k in preceding if k not in self.completed), None)
        if missing:
            return f"먼저 {LABELS[missing]} 단계를 완료하세요"
        if action not in ("can_up", "sensors"):
            # 0912 R9(현장 발견): 인계 '전'의 stage8 은 --wait-takeover 로 CAN 침묵(피드백 없음)이라 12관절 신선을
            # 요구하면 인계가 영원히 막힌다. 인계는 아래의 '대기 중(waiting)+status 신선+kp150 홀드' 게이트로 판정하고,
            # 12관절 피드백은 인계 뒤 단계(기준점·ARM·정책 로드·발사)에만 요구한다.
            r = self._sensor_reason(action in ("baseline", "arm", "bridge", "stand"))
            if r:
                return r
        if action in ("boot_pose", "selftest") and owners:
            return "자세 유지 중 실행할 수 없는 점검입니다"
        if action == "hang_stand" and not (fresh.get("status") and s.get("waiting")):
            return "인수 제어기가 실제로 대기 중이어야 합니다"
        if action in ("gain_ramp", "ground", "handoff"):
            proc = self.view.get("procs", {}).get("hang_stand", {})
            hold = proc.get("hold", {})
            if not proc.get("alive") or self.view.get("server_time", 0) - hold.get("ts", 0) >= 3:
                return "자세 유지 프로세스의 신선한 상태 확인 필요"
            if action in ("ground", "handoff") and hold.get("kp", 0) < 150:
                return "실제 운용 게인 도달 확인 필요"
        if action == "ground" and self.mode == "ground":
            return self._contact_reason()
        if action == "handoff" and not (fresh.get("status") and s.get("waiting")):
            return "인수 제어기 대기 상태 확인 필요"
        if action in ("baseline", "arm") and not s.get("holding"):
            return "인수 홀드 상태 확인 필요"
        if action == "arm" and s.get("baseline") != 12:
            return "기준점 12/12 확인 필요"
        if action in ("bridge", "stand") and not s.get("armed"):
            return "제어기 ARM 상태 확인 필요"
        if action == "stand":
            if b.get("loaded_checkpoint") != self.checkpoint:
                return "실제 로드된 정책이 선택 정책과 다릅니다"
            if self.mode == "ground":
                return self._contact_reason()
        return None

    def begin(self, action, wall_time):
        self.revision += 1
        self.pending = {"action": action, "started": self.clock(), "wall": wall_time,
                        "launched": False, "fault_at_start": self.fault,
                        "status_seq": self.view.get("stage8", {}).get("command_seq", -1)}
        self.last_result = None
        self.emit("request", action=action)

    def finish(self, ok, message):
        if not self.pending:
            return
        action = self.pending["action"]
        self.pending = None
        self.revision += 1
        self.last_result = {"action": action, "ok": ok, "message": message}
        if ok:
            self.completed.add(action)
        self.emit("action_confirmed" if ok else "action_failed", action=action, message=message)
        if not ok and action in ("hang_stand", "gain_ramp", "ground", "handoff", "baseline", "arm", "stand", "stack_down", "release"):
            self.latch(f"{LABELS[action]} 확인 실패: {message}")

    def operator_event(self, action, payload):
        if action == "support_confirm":
            self.support = True
            self.invalidate("회수 줄 지지 확인")
        elif action in ("op_retension", "op_assist"):
            self.invalidate(LABELS[action])
        elif action == "op_release_start":
            self.support = False
            self.releasing = True          # 0911: 방출 중 표시 — '모든 줄 느슨' 확인은 방출 정지 기록 후에만
            if self.slack_at is not None:
                self.invalidate("자립 측정 중 윈치 조작")
        elif action == "op_release_stop":
            self.releasing = False
        elif action == "op_slack_done":
            self.support = False
            self.releasing = False
            self.slack_at = self.clock()
            self._pose_stamps = None
            self.measurement = {"state": "measuring", "elapsed": 0.0,
                                "reason": "운영자 무보조 확인 · 영상 검토 필요", "review_required": True}
        self.revision += 1
        self.emit(action, note=str(payload.get("note", ""))[:300])

    def update(self, view):
        self.view = view
        s, b = view.get("stage8", {}), view.get("bridge", {})
        fresh = view.get("fresh", {})
        if b.get("fault"):
            self.latch(b["fault"])
        in_handoff = self.pending and self.pending["action"] == "handoff"
        if not in_handoff and len([k for k in self.active_processes() if k in ("hang_stand", "stage8")]) == 2 and fresh.get("status") and (s.get("armed") or s.get("holding")):
            self.latch("자세 유지와 제어기 송신 상태가 겹칩니다")
        if self.active_processes() and not view.get("session", {}).get("active"):
            self.latch("시험 기록 중단 — 회수 후 저장 장치를 확인하세요")
        if self.active_processes() and view.get("can", {}).get("state") == "BUS-OFF":
            self.latch("CAN BUS-OFF 감지")
        recovering = self.support or (self.pending and self.pending["action"] in RECOVERY)
        if not recovering and self.completed.intersection(CONTROL):
            if view.get("external_controls"):
                self.latch("추적 밖 제어 송신 가능 프로세스 감지")
            if "stage8" in self.completed and not view.get("procs", {}).get("stage8", {}).get("alive"):
                self.latch("인수 제어 프로세스 종료 — 실제 상태 확인 필요")
            if "hang_stand" in self.completed and "handoff" not in self.completed and not in_handoff:
                hp = view.get("procs", {}).get("hang_stand", {})
                if not hp.get("alive"):
                    self.latch("자세 유지 프로세스 종료 — 회수 상태 확인 필요")
            if "handoff" in self.completed and not fresh.get("status"):
                self.latch("인수 후 제어기 상태 수신 끊김")
        if "stand" in self.completed and "stack_down" not in self.completed and not self.support:
            if not fresh.get("bridge"):
                self.latch("정책 상태 수신 끊김 — 실제 제어 상태 불명")
            elif b.get("checkpoint") != self.checkpoint:
                self.latch("실행 중 정책 불일치")
            r = self._sensor_reason(True)
            if r:
                self.latch(r)
        if self.slack_at is not None:
            reason = self._sensor_reason(True) or self._contact_reason()
            if not view.get("session", {}).get("active"):
                reason = "기록 중단"
            if not s.get("armed") or not s.get("running") or not b.get("armed"):
                reason = "정책 제어 상태 상실"
            if reason:
                self.invalidate(reason)
            else:
                self.max_elapsed = self.clock() - self.slack_at
                self.measurement["elapsed"] = round(self.max_elapsed, 1)
                # Timer only establishes observed duration, never an automatic
                # claim of physical independence or acceptable posture.
                for threshold in (5, 30):
                    key = f"observed_{threshold}s"
                    if self.max_elapsed >= threshold and not self.measurement.get(key):
                        self.measurement[key] = True
                        self.emit(key, review_required=True)
        p = self.pending
        if not p or not p["launched"]:
            return
        action = p["action"]
        procs = view.get("procs", {})
        key = {"baseline": "stage8_baseline", "arm": "stage8_arm",
               "gain_ramp": "hang_stand", "ground": "hang_stand", "handoff": "hang_stand"}.get(action, action)
        proc = procs.get(key, {})
        received = view.get("received", {})
        status_new = fresh.get("status") and received.get("status", 0) > p["wall"]
        ok = False
        if self.fault and self.fault != p["fault_at_start"] and action not in RECOVERY:
            return self.finish(False, "정지·고장이 발생해 진행을 중단했습니다")
        # 0911 호환검수 R2: 콘솔이 정지를 요청하지 않았는데 제어기가 요청 이후 STOP_ALL 을 수락했다면(브리지발
        # 워밍업 타임아웃·센서 stale·가드 정지 등) 브리지 stdout 소유 여부와 무관하게 즉시 래치한다.
        if (action == "stand" and self.stop_request_at is None and status_new
                and s.get("last_command") == "STOP_ALL" and s.get("accepted")
                and (s.get("command_seq") or 0) > (p.get("status_seq") or 0)):
            self.latch("제어기가 브리지/외부 STOP_ALL 을 수락 — 브리지 정지 사유(워밍업 타임아웃·센서 stale·가드) 확인 필요")
            return self.finish(False, "정책 시작 중 제어기 STOP_ALL 수신")
        if action == "sensors":
            ok = not self._sensor_reason()
        elif action == "stage8":
            ok = status_new and s.get("waiting") and proc.get("alive")
        elif action in ("hang_stand", "gain_ramp", "ground"):
            hold = proc.get("hold", {})
            valid = proc.get("alive") and hold.get("ts", 0) > p["wall"] and view.get("server_time", 0) - hold.get("ts", 0) < 3.0
            if action == "hang_stand":
                ok = valid and hold.get("phase") == "에어유지"
            elif action == "gain_ramp":
                ok = valid and hold.get("kp", 0) >= 150 and hold.get("phase") in ("운용대기", "하중유지")
            else:
                ok = valid and hold.get("phase") == "하중유지"
        elif action == "handoff":
            ok = not proc.get("alive") and proc.get("rc") == 3 and status_new and s.get("holding") and s.get("hold_joint_count") == 12
        elif action == "baseline":
            ok = (status_new and s.get("baseline") == 12 and proc.get("rc") == 0
                  and s.get("accepted") and s.get("last_command") == "SET_BASELINE_FROM_CURRENT_ALL"
                  and (s.get("command_seq") or 0) > (p.get("status_seq") or 0))
        elif action == "arm":
            ok = (status_new and s.get("armed") and proc.get("rc") == 0
                  and s.get("accepted") and s.get("last_command") == "ARM_ALL"
                  and (s.get("command_seq") or 0) > (p.get("status_seq") or 0))
        elif action == "bridge":
            ok = proc.get("alive") and b.get("loaded_checkpoint") == self.checkpoint
        elif action == "stand":
            ok = (fresh.get("bridge") and received.get("bridge", 0) > p["wall"]
                  and b.get("armed") and b.get("checkpoint") == self.checkpoint
                  and status_new and s.get("armed") and s.get("running"))
        elif action == "stack_down":
            ok = not self.active_processes() and not fresh.get("status") and not fresh.get("bridge")
        elif action in ("boot_pose", "selftest", "release", "analyze"):
            ok = proc.get("rc") == 0 and not proc.get("alive")
            if action == "selftest":
                ok = ok and proc.get("all_pass", False)
            if action == "release":
                if not proc.get("alive") and proc.get("release_error"):
                    return self.finish(False, "릴리즈 과정의 피드백·송신 오류 — 실제 토크 상태 확인 필요")
                ok = ok and proc.get("release_ok", False)
        if ok:
            message = "해제 명령 전송 완료 보고 · 실제 토크 상태는 현장 확인 필요" if action == "release" else "실제 응답 확인 완료"
            return self.finish(True, message)
        if proc and not proc.get("alive") and proc.get("rc") not in (None, 0, 3):
            return self.finish(False, f"프로세스 종료 코드 {proc['rc']}")
        # 0911 호환검수 R2: stand 는 브리지 인계 워밍업(최대 4 s)+블렌딩 안에 응답하므로 30 s 는 브리지발 정지를 26 s 동안 가린다
        timeout = 300 if action in ("boot_pose", "selftest", "analyze") else (8 if action == "stand" else 30)
        if self.clock() - p["started"] > timeout:
            self.finish(False, "확인 시간 초과 — 실제 상태를 확인하세요")

    def snapshot(self):
        s, b = self.view.get("stage8", {}), self.view.get("bridge", {})
        fresh = self.view.get("fresh", {})
        owners = self.active_processes()
        if "release" in self.completed:
            phase, owner = "회수 명령 완료", "토크 해제 전송 완료 보고 · 실토크 미검증"
        elif self.fault:
            phase, owner = "정지·상태 확인 필요", "실제 상태 확인 필요"
        elif self.measurement["state"] == "interrupted":
            phase, owner = "무보조 관찰 중단 · 회수 필요", "정책 실행 상태는 별도 확인"
        elif fresh.get("bridge") and b.get("armed") and fresh.get("status") and s.get("running"):
            phase, owner = "자립 관찰" if self.slack_at is not None else "정책 제어 · 지지 제거", "정책 → 12축 제어기"
        elif fresh.get("status") and s.get("armed"):
            phase, owner = "정책 시작 준비", "12축 제어기 · ARM"
        elif fresh.get("status") and s.get("holding"):
            phase, owner = "정책 인계", "12축 제어기 · 인수 홀드"
        elif "hang_stand" in owners:
            phase, owner = "공중 자세 유지", "자세 유지 스크립트"
        elif fresh.get("status") and s.get("waiting"):
            phase, owner = "자세 유지 준비", "인수 대기 · 송신 전"
        else:
            phase, owner = "준비 확인", "미확인" if fresh.get("status") or owners else "제어 시작 전 / 외부 상태 미확인"
        stop = None
        if self.stop_request_at is not None:
            # 0911 호환검수 R1: stage8 는 armed 인 한 STOP_ALL 다음 제어틱에 running=True 로 돌아온다(동결 목표를 계속
            # 스트리밍 = 토크 유지). 'not running' 을 요구하면 동결 확인이 구조적으로 실패해 경보가 상시화되므로,
            # 확인은 "STOP_ALL 이 요청 이후 수락됐고(command_seq 증가) 그 뒤 status 가 왔다" 로 판정한다.
            ack = (fresh.get("status") and self.view.get("received", {}).get("status", 0) > self.stop_request_at
                   and s.get("last_command") == "STOP_ALL" and s.get("accepted")
                   and (s.get("command_seq") or 0) > self.stop_request_seq)
            stop = {"confirmed": bool(ack), "delivery": self.stop_delivery,
                    "message": "STOP_ALL 수락 확인 · 제어기는 동결 목표를 계속 스트리밍(토크 유지), 다른 송신자 확인 필요" if ack else "정지 요청 후 STOP_ALL 수락 미확인"}
        allowed = {k: {"enabled": self.reason(k) is None, "reason": self.reason(k), "label": v} for k, v in LABELS.items()}
        if self.mode == "air":
            allowed["ground"]["label"] = "공중 인계 조건 확인"
        if self.support or self.fault or self.measurement["state"] == "interrupted":
            next_action = next((k for k in ("support_confirm", "stack_down", "release", "session_end")
                                if k not in self.completed and (k != "support_confirm" or not self.support)), None)
        elif self.slack_at is not None:
            next_action = None
        else:
            next_action = next((k for k in self.flow() if k not in self.completed), "op_slack_done")
        return {"boot_id": self.boot_id, "revision": self.revision, "number": self.number,
                "mode": self.mode, "checkpoint": self.checkpoint, "phase": phase, "owner": owner,
                "completed": sorted(self.completed), "pending": self.pending, "fault": self.fault,
                "fault_hint": getattr(self, "fault_hint", None),
                "last_result": self.last_result, "next_action": next_action, "allowed": allowed,
                "measurement": dict(self.measurement), "stop": stop, "support_confirmed": self.support,
                "history": list(self.history)}
