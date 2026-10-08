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
    "op_note": "메모 기록", "op_push": "밀기 기록", "analyze": "이 시험 로그 분석", "estop": "정지 요청",
    "configure": "시험 설정", "new_trial": "새 시험 준비", "force_reset": "강제 초기화(무토크 확인)",
    "joint_trim": "개별 관절 조정", "joint_trim_zero": "개별 조정 전부 0",
    "joint_trim_preset": "수평 기준 자세 적용", "hold_kd": "홀드 감쇠 kd",
    "air_gait_on": "에어 보행 시작", "air_gait_off": "에어 보행 정지",
    "auto_level_on": "자동 수평 ON", "auto_level_off": "자동 수평 OFF",
    "zero_multiple": "영점 배수 보정", "zero_recapture": "영점 재캡처", "zero_kin": "영점 기구학 검증",
    "policy_on": "정책 ON(착지 자동 인수)",
    # 1007 걷기 조작(W3 세션 전용) — 버튼 하나 = 시간 제한 걷기 한 번(끄기 기한은 브리지 몫)
    "w3_move": "걷기 버튼(시간 제한 걷기)", "w3_off": "걷기 끄기(서기)", "w3_stop": "걷기 정지 래치", "w3_rearm": "걷기 재무장",
}
MANUAL = {"joint_trim", "joint_trim_zero", "joint_trim_preset", "hold_kd", "air_gait_on", "air_gait_off", "auto_level_on", "auto_level_off"}
ZERO_TOOLS = {"zero_multiple", "zero_recapture", "zero_kin"}   # 0914 영점 도구 — 제어 시작 전(준비)에서만   # 0914 12축 개별 제어기(자세 유지 스크립트의 관절 트림). 운용 게인 뒤에만.
PREP = ("can_up", "sensors", "boot_pose", "selftest", "stage8", "hang_stand", "gain_ramp")
CONTROL = ("stage8", "hang_stand", "bridge", "stage8_arm", "stage8_baseline")
#: 1005 단순화 기동: '정책 ON' 한 번이 서버 안에서 기존 단계를 같은 게이트·같은 응답 확인으로 순서대로 실행한다.
#: 정책은 착지 확인 전까지 구동되지 않는다(브리지 stand touchdown_start — 착지 전 홀드 유지). 기존 단계별 절차는 그대로 남는다.
#: 1008: 시험 방식이 '공중'(mode "air")이면 마지막 stand 를 hang_arm 으로 보낸다 — 정책이 공중에서 바로 구동된다(착지 대기 없음). 접지 방식은 위 그대로다.
POLICY_ON_STEPS = ("ground", "handoff", "baseline", "arm", "bridge", "stand")
POLICY_ON_LABEL_AIR = "정책 ON(공중에서 바로 구동)"   # 공중 방식에서 화면·robotctl 에 보이는 이름(접지 방식은 LABELS["policy_on"] 그대로)
POLICY_ON_LEVEL_TOL_DEG = 5.0     # 홀드 자세 vs 수평 기준 자세(level_pose) 허용 편차 — '수평 자세 적용 누락' 검출용
POLICY_ON_RAMP_TOL_DEG = 0.5      # 개별 조정 램프 완료(요청 vs 적용) 허용 편차

#: 0918 프로세스 종료 코드 → (뜻, 할 일). 키 = 액션 이름 또는 프로세스 키. 3 = 정상(핸드오프)라 실패로 안 잡음.
RC_HINTS = {
    "zero_multiple": {
        1: ("불량 모터 있음(무응답·폴트·불안정)", "robotctl motors 로 응답·표 확인 → 커넥터·전원"),
        2: ("차단: CAN 쓰는 프로세스가 살아 있거나, 창 어긋남이 없어 보정할 것이 없음(자세불일치만 남음)",
            "robotctl motors 표의 조치 열대로 — 자세불일치면 그 관절을 기준 자세로 맞추기, 지그에서 맞는데도 남으면 zero-recap"),
    },
    "zero_recapture": {
        1: ("불량 모터 있음(무응답·폴트·불안정)", "robotctl motors 로 응답 확인 → 커넥터·전원"),
        2: ("차단: 창 어긋남이 남아 있음(재캡처하면 어긋난 프레임이 영구 박힘) 또는 CAN 쓰는 프로세스 살아 있음",
            "robotctl zero-fix 먼저 → 다시 motors → 그 뒤 zero-recap"),
    },
    "zero_kin": {
        1: ("가드로 중단(현 자세 홀드 유지)", "자세·줄 확인 뒤 재실행"),
        2: ("선행조건 미충족(stage8 ARMED 아님 / 센서 미수신 / 브리지 live)", "순서 확인: 인계·ARM 뒤, 정책 시작 전"),
    },
    "boot_pose": {
        1: ("영점 확인 실패: 잔차 초과 관절 또는 읽기 실패", "robotctl motors 표의 조치 열대로 (창 어긋남 zero-fix / 자세 맞추기 / 무응답 커넥터)"),
    },
    "hang_stand": {
        1: ("시작 실패: 모터 시작각 읽기 실패 또는 스탠드 대비 12° 초과 이탈", "영점 확인(boot_pose)부터 다시, 매달림 직립 정렬 후 재실행"),
        2: ("가드 릴리즈(관절 이탈 DEV_ABORT·오차 초과) — 전 모터 무토크", "줄 확인 → robotctl down (모터 전원 OFF 면 robotctl reset) → motors 로 창 어긋남 확인"),
    },
    "stage8": {
        1: ("12축 제어기 기동 실패(인자·ROS2) 또는 비정상 종료", "콘솔 로그 확인 → 회수 후 prep 부터 다시"),
    },
    "selftest": {
        1: ("셀프테스트 FAIL 항목 있음", "로그의 [FAIL] 줄 확인(정책 등록·계약·경로)"),
    },
    "release": {
        4: ("버스 무응답 — 응답 0/12(모터 전원 차단 추정)", "CAN 으로 토크를 걸 수도 해제할 수도 없는 상태입니다. 전원 차단을 현장에서 확인하면 단계는 자동으로 넘어갑니다"),
        5: ("일부 모터만 응답/송신 실패", "허브 커넥터·분기 배선을 확인하고 robotctl motors 로 응답 수를 다시 보세요"),
        1: ("토크 해제 송신/피드백 오류", "모터 전원 OFF·이미 무토크면 robotctl reset, 아니면 전원 확인 뒤 재시도"),
    },
    "bridge": {
        1: ("정책 브리지 종료(가드 정지·estop·오류)", "회수(robotctl down) → 블랙박스 로그 확인"),
    },
}


def explain_rc(key, rc):
    """종료 코드 설명 문자열. 모르는 조합이면 코드만."""
    try:
        meaning, todo = RC_HINTS.get(key, {})[int(rc)]
    except (KeyError, TypeError, ValueError):
        return f"프로세스 종료 코드 {rc}"
    return f"프로세스 종료 코드 {rc}: {meaning} → {todo}"
EVENTS = {k for k in LABELS if k.startswith("op_")}
RECOVERY = {"support_confirm", "stack_down", "release", "session_end", "analyze"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# 1007 걷기 조작 — 시간 제한 걷기 버튼 한 번의 수명 상태기 (W3 세션 전용)
# ---------------------------------------------------------------------------
#: 걷기 실험 키: 브리지가 실험 플래그(--experimental-w3 [--experimental-kew3])로만 등록하는 정책 키.
#: 값은 여기 한 곳에 적고 콘솔 selftest 가 원본(w3_deploy.W3S3D_CKPT_KEY · 브리지 KEW3_CKPT_KEY·KEW3Q_CKPT_KEY)과 대조한다.
#:   composite = 합성 러너(서 있는 동안 KE, 걷는 동안 W3) — '서기 복귀' 판정에 kew3.mode == "ke" 를 더 요구한다.
#: 1008: 셋째 걷기 키(브리지의 둘째 합성 키) = 첫 합성 키와 같은 구성에 KE 감지기 갇힘 해소(회복 모드 중 조용한 양발 서기 2 s → 기준 재설정·서기 복귀)만 켠 것 —
#:   **시뮬 합성 평가가 돌린 구성이 이 키다**(첫 합성 키는 그 옵션이 꺼져 있다 — 회복 모드에 갇히면 걷기 켜기가 계속 거부될 수 있다). 실물 미검증.
#:   **뒤에** 덧붙였다 — 앞의 두 키(이름·표시·순서)는 그대로다. 어느 키를 쓸지는 사용자가 정한다.
#: 1008 오후: 넷째 걷기 키(브리지의 셋째 합성 키) = 셋째 걷기 키(서기 KE·걷기 W3·갇힘 해소 그대로)에 **공중·착지 구간 서기 정책**(G21h — 매달림 시작을
#:   학습 분포에 넣은 서기)을 더한 3 단 러너(AIR → KE ⇄ W3). 공중에서 시작하면(stand + hang_arm) 공중·착지 정책이 맡다가 착지가 가라앉으면 브리지 러너가
#:   스스로 서기 KE 로 넘긴다(디버그 kew3.mode "air" → "ke"). 지면에서 시작하면 셋째 걷기 키와 같은 동작이다.
#:   air_start = 공중 정책 시작(시험 방식 '공중')을 받는 키. **이 표시가 없는 키는 False 로 읽는다(.get)** — 앞의 세 항목은 그대로다.
#:   브리지 명세(KEW3_SPECS 의 air 가 있는 키)와 같은 키여야 한다(콘솔 selftest 가 브리지 원문과 대조). 승윤님 결정(1008): 정책을 공중에서 구동한 채 내린다 —
#:   앞의 세 키는 지면 시작으로만 학습돼 브리지가 hang_arm 을 거부한다(시뮬 AIR1: 착지 낙상 40–62 %). 실물 미검증.
WALK_KEYS = {
    "stand_w3s3d_15800": {"composite": False, "label": "걷기 실험 · W3 단독"},
    "stand_kew3_g21a_g24ra_w3s3d15800": {"composite": True, "label": "걷기 실험 · 서기 KE + 걷기 W3 합성"},
    "stand_kew3q_g21a_g24ra_w3s3d15800": {"composite": True, "label": "걷기 실험 · 서기 KE + 걷기 W3 합성 + 갇힘 해소(시뮬 평가 구성) · 실물 미검증"},
    "stand_kew3qa_g21a_g24ra_w3s3d15800": {"composite": True, "air_start": True,
                                           "label": "걷기 실험 · 서기 KE + 걷기 W3 합성 + 갇힘 해소 + 공중·착지 정책(G21h) · 공중 정책 시작 가능 · 실물 미검증"},
}


def walk_air_start(checkpoint):
    """이 정책 키가 공중 정책 시작(stand + hang_arm)을 받는 걷기 키인가 — WALK_KEYS 의 air_start(없으면 False)."""
    return bool(WALK_KEYS.get(checkpoint, {}).get("air_start"))


def default_mode(checkpoint):
    """시험 방식의 기본값: 공중 정책 시작을 받는 걷기 키면 "air"(공중에서 정책 시작 → 착지), 그 밖은 종전대로 "ground"(접지 후 정책 인계).
    새 TrialState 가 이 값으로 시작한다. '새 시험 준비'는 앞 시험의 방식을 그대로 넘기고, '시험 설정'은 운영자가 고른 값을 쓴다."""
    return "air" if walk_air_start(checkpoint) else "ground"


WALK_KIND = {"w3_move": "move", "w3_off": "off", "w3_stop": "stop", "w3_rearm": "rearm"}
WALK_SAFE = {"w3_off", "w3_stop"}      # 안전 방향(걷기 끄기·정지 래치) — 단계·고장 래치·앞 요청과 무관하게 받는다
WALK_ACK_TIMEOUT_S = 2.0               # 응답 대기 [s]. 넘으면 '응답 없음' — 같은 명령을 다시 보내지 않는다
WALK_END_GRACE_S = 3.0                 # 표시 전용: 수락 뒤 기한 + 이 시간이 지나도 걷기 끝이 확인되지 않으면 경고(콘솔은 끄기 타이머를 갖지 않는다)
WALK_STAND_GRACE_S = 6.0               # 표시 전용: 걷기 끝 요청 뒤 이 시간이 지나도 서기가 확인되지 않으면 경고(반주기 0.45 s + 합성 정착 상한 3 s)
WALK_SETTLE_MARGIN_S = 0.15            # 콘솔(20 Hz 표본)과 브리지(정책 틱)가 세는 '조용한 양발 서기' 시간의 차이 여유 [s]
WALK_ACTIVE = ("sent", "walking", "ending")
WALK_STAGE_LABEL = {
    "sent": "발행함 · 브리지 응답 대기", "walking": "수락됨 · 걷는 중(서기 복귀 미확인)",
    "ending": "걷기 끝 요청됨 · 서기 복귀 확인 중", "done": "서기 복귀 확인됨", "rejected": "거부됨",
    "no_ack": "응답 없음 — 상태 확인 필요", "interrupted": "중단됨",
    "accepted": "수락됨",              # 보조 명령(끄기·정지 래치·재무장)에만 쓰는 끝 단계
}


class WalkState:
    """걷기 버튼(시간 제한 걷기) 한 번의 수명. 서버가 갖는 상태라 새로고침·재접속·중복 클릭과 무관하다.

        sent(발행함) → walking(같은 req_id 의 수락 응답) → ending(같은 req_id 의 burst_end 응답 = 브리지가 스스로 끄기 요청)
                     → done(디버그 토픽이 서기를 보임: want 0·ge 0, 합성 키면 mode ke)
        · rejected(거부 응답) · no_ack(응답 대기 초과 — 재전송 없음, 디버그가 서기를 보일 때까지 새 버튼 차단)
        · interrupted(정지 요청·고장 래치·브리지 정지 래치)

    이 클래스는 명령을 보내지 않는다. 콘솔이 발행 직전에 begin() 으로 알리고, 응답은 feed(), 디버그 상태는 update() 로 받는다.
    '수락됨'(walking)과 '서기 복귀 확인됨'(done)은 다른 단계다 — 클릭 성공·수락 응답을 동작 완료로 취급하지 않는다.
    끄기 기한은 브리지 몫이다(step 켜기의 dur_s). 여기에는 끄기 타이머가 없고, 기한이 실제로 걸렸는지를 수락 응답에서 확인만 한다.
    디버그 프레임과 응답은 서로 다른 토픽이라 도착 순서가 보장되지 않는다 → 프레임이 응답보다 '뒤의 것'인지는 둘의 브리지 timestamp 로 가린다.
    """

    def __init__(self):
        self.cur = None          # 현재/마지막 걷기 버튼 기록
        self.cmd = None          # 마지막 보조 명령(끄기·정지 래치·재무장) 기록
        self.rev = 0             # 기록 단계가 바뀔 때마다 증가(화면 갱신·진단용)
        self.acks = []           # 최근 응답 요약(표시용, 최대 8)
        self.guard_mode = None   # 마지막 응답이 알려 준 guard_mode (디버그 토픽은 safe 일 때 이 필드를 싣지 않는다)
        self._inbox = []
        self._steady_since = None   # 콘솔이 본 '조용한 양발 서기'가 이어지기 시작한 시각(끊기면 None) — 브리지 켜기 조건을 따라 세는 표시용 시계
        self.broken = None       # 이 상태기 안에서 난 예외(첫 것) — 걷기 버튼·재무장을 잠그되 기립 시험 상태기로 번지지 않게 가둔다(fail)

    # ---- 입력 ----
    def feed(self, items):
        """수신한 응답 [{"t": 수신 시각, "ack": dict}, …] 을 쌓아 둔다 — 처리는 update() 가 수신 순서대로 한다."""
        self._inbox.extend(i for i in items if isinstance(i, dict) and isinstance(i.get("ack"), dict))

    def begin(self, trial, kind, req_id, seq, cmd, name=None, label=None):
        """콘솔이 발행 **직전에** 부른다(발행은 한 번뿐). kind = "move" | "off" | "stop" | "rearm"."""
        rec = {"kind": kind, "req_id": req_id, "seq": int(seq), "name": name, "label": label or kind, "stage": "sent",
               "why": None, "note": None, "accepted": None, "cmd": dict(cmd),
               "wall_sent": self._wall(trial), "_t_sent": trial.clock()}
        if kind == "move":
            rec.update(dur_s=cmd.get("dur_s"), deadline=None, end_by=None, late_ack=False, seen_walking=False, stand_seen=None)
            self.cur = rec
        else:
            self.cmd = rec
        self.rev += 1
        trial.emit("walk_request", req_id=req_id, walk=name or kind, stage="sent", cmd=dict(cmd),
                   message=f"{rec['label']} 발행(1 회) · seq {int(seq)}")
        return rec

    def not_sent(self, trial, rec, why):
        """발행 자체가 실패했다(명령이 나가지 않음) — 브리지의 거부와 구분해 by=console 로 남긴다."""
        self._set(trial, rec, "rejected", "walk_not_sent", why=why, accepted=False, by="console")

    def interrupt(self, trial, reason):
        """정지 요청·고장 래치 — 진행 중인 걷기 버튼(응답 없음으로 서기 확인을 기다리던 것 포함)을 중단으로 닫는다."""
        cur = self.cur
        if cur is not None and (cur["stage"] in WALK_ACTIVE or (cur["stage"] == "no_ack" and not cur.get("stand_seen"))):
            self._set(trial, cur, "interrupted", "walk_interrupted", why=str(reason), stand_seen=None)

    def fail(self, trial, exc):
        """걷기 패널(실험 기능) 안에서 난 예외를 가둔다 — 걷기 버튼·재무장은 잠그고(끄기·정지 래치·정지 요청은 그대로), 기립 시험의
        단계·고장 래치·기록은 건드리지 않는다. 첫 예외만 남기고 타임라인에 한 번 적는다."""
        if self.broken is None:
            self.broken = f"{type(exc).__name__}: {exc}"[:300]
            self.rev += 1
            trial.emit("walk_internal_error", message=f"걷기 패널 내부 오류 — 걷기 버튼 잠금: {self.broken}")

    # ---- 내부 ----
    @staticmethod
    def _wall(trial):
        stamp = (trial.view or {}).get("server_time")
        return stamp if finite(stamp) else time.time()

    @staticmethod
    def describe(rec):
        """화면·기록에 쓰는 한 줄 요약. 사유(why)는 거부·중단·응답 없음과 보조 명령의 수락에만 붙인다 — 걷기 버튼의 수락 사유는
        길어서 기록의 why 필드와 '최근 응답' 줄에 두고, 끝난 뒤에는 무엇으로 끝났는지를 보인다."""
        stage = rec["stage"]
        text = f"{rec['label']} · {WALK_STAGE_LABEL.get(stage, stage)}"
        if stage in ("rejected", "interrupted", "no_ack", "accepted") and rec.get("why"):
            text += f" — {rec['why']}"
        elif stage == "ending":
            text += " — " + {"burst": "브리지 기한 종료", "off": "운영자 걷기 끄기",
                            "observed": "디버그에서 걷기 요청 해제 관측"}.get(rec.get("end_by"), str(rec.get("end_by")))
        elif stage == "done":
            # 콘솔이 본 시간이다(브리지 틱이 아니라 콘솔 상태 주기 기준): 수락 응답 → 걷기 끝 요청 → 서기 확인
            parts = ([f"수락→끝 요청 {rec['walked_s']:.2f} s"] if finite(rec.get("walked_s")) else []) \
                + ([f"끝 요청→서기 확인 {rec['settle_s']:.2f} s"] if finite(rec.get("settle_s")) else [])
            if parts:
                text += " — " + " · ".join(parts)
        if stage == "walking" and rec.get("deadline") is False:
            text += " · ⚠ 브리지가 이 요청으로 기한(burst)을 걸지 않았습니다 — 스스로 서지 않을 수 있음. '걷기 끄기(서기)'나 정지 요청으로 끝내세요"
        if rec.get("note"):
            text += f" · {rec['note']}"
        return text

    def _set(self, trial, rec, stage, event, **fields):
        rec.update(fields)
        rec["stage"] = stage
        rec["wall_" + stage] = self._wall(trial)
        rec["_t_" + stage] = trial.clock()
        self.rev += 1
        trial.emit(event, req_id=rec["req_id"], walk=rec.get("name") or rec["kind"], stage=stage, message=self.describe(rec),
                   **({"why": rec["why"]} if rec.get("why") else {}))      # 브리지·콘솔이 준 사유 원문(타임라인용)

    def _end(self, trial, cur, by, ts, wall, **fields):
        """걷기 끝 '요청'이 확인됐다(브리지 기한 burst_end · 운영자 끄기 · 디버그의 want 0). 서기 복귀 확인은 아직 아니다."""
        self._set(trial, cur, "ending", "walk_burst_end" if by == "burst" else "walk_end",
                  end_by=by, _end_ts=ts, _end_wall=wall, **fields)

    @staticmethod
    def bridge_status(trial):
        """**신선한** 디버그 프레임에서만 걷기 상태를 읽는다 — 오래된 프레임의 want 0·ge 0 을 '서 있음'으로 쓰지 않는다."""
        view = trial.view or {}
        b = view.get("bridge") or {}
        fresh = bool((view.get("fresh") or {}).get("bridge"))
        w3 = b.get("w3") if fresh and isinstance(b.get("w3"), dict) else None
        sup = b.get("w3_sup") if fresh and isinstance(b.get("w3_sup"), dict) else None
        kew3 = b.get("kew3") if fresh and isinstance(b.get("kew3"), dict) else None
        key = b.get("checkpoint") if b.get("checkpoint") in WALK_KEYS else trial.checkpoint
        composite = kew3 is not None or bool(WALK_KEYS.get(key, {}).get("composite"))
        want, ge = (w3 or {}).get("want"), (w3 or {}).get("ge")
        mode = (kew3 or {}).get("mode")
        return {"fresh": fresh, "w3": w3, "sup": sup, "kew3": kew3, "composite": composite, "mode": mode,
                # 서기 = 걷기 요청 없음·래치 꺼짐(합성 키는 KE 로 인계까지). 필드가 없거나 bool 이 아니면 서기로 보지 않는다.
                "standing": w3 is not None and want is False and ge is False and (not composite or mode == "ke"),
                "walking": w3 is not None and (want is True or ge is True or (composite and mode == "w3")),
                "ts": b.get("ts") if finite(b.get("ts")) else None,
                "recv": (view.get("received") or {}).get("bridge")}

    @staticmethod
    def _end_stand_observation(trial, why):
        """서기(무보조) 관찰 구간은 걷기가 시작되면 끝난다 — 한발 지지의 접촉 상실로 무효 처리되기 전에 사유를 정확히 남긴다."""
        if trial.slack_at is not None and trial.measurement.get("state") == "measuring":
            trial.invalidate(f"{why} — 서기(무보조) 관찰 구간 종료")
            trial.measurement["ended_by"] = "walk"

    @staticmethod
    def _after(st, ref_ts, ref_wall):
        """디버그 프레임이 기준 시점(응답)보다 뒤의 것인가 — 브리지 시계끼리 비교하고, 없으면 콘솔 수신 시각으로."""
        if finite(st["ts"]) and finite(ref_ts):
            return st["ts"] > ref_ts
        return finite(st["recv"]) and finite(ref_wall) and st["recv"] > ref_wall

    def _on_ack(self, trial, item):
        ack = item["ack"]
        rid, event, ts, wall = ack.get("req_id"), ack.get("event"), ack.get("timestamp"), item.get("t")
        accepted = ack.get("accepted") is True
        why = str(ack.get("why") if ack.get("why") is not None else "")[:300]
        if isinstance(ack.get("guard_mode"), str):
            self.guard_mode = ack["guard_mode"]
        brief = {"t": wall, "req_id": rid, "cmd": ack.get("cmd"), "event": event, "accepted": accepted,
                 "why": why, "action": ack.get("action"), "match": None}
        self.acks = (self.acks + [brief])[-8:]
        cur, cmd = self.cur, self.cmd
        if cur is not None and isinstance(rid, str) and rid == cur["req_id"]:
            brief["match"] = "walk"
            if event == "burst_end":
                if cur["stage"] in ("sent", "walking", "no_ack"):
                    lost = cur["stage"] != "walking"      # 수락 응답은 못 받았지만 같은 req_id 의 기한 종료가 왔다 = 수락됐었다
                    self._end(trial, cur, "burst", ts, wall, end_why=why, accepted=True,
                              **({"late_ack": True, "note": "수락 응답 미수신 — burst_end 로 수락 확인"} if lost else {}))
            elif cur["stage"] in ("sent", "no_ack"):
                late = cur["stage"] == "no_ack"
                if accepted:
                    sup = ack.get("sup") if isinstance(ack.get("sup"), dict) else {}
                    burst = sup.get("burst")
                    armed = isinstance(burst, dict) and burst.get("req_id") == rid
                    # deadline = 브리지가 이 요청으로 끄기 기한(burst)을 실제로 걸었는가. 아니면(옛 브리지·이미 켜져 있던 걷기 등)
                    # 스스로 서지 않을 수 있다 — 콘솔은 끄기를 대신 보내지 않고 경고만 띄운다(describe).
                    self._set(trial, cur, "walking", "walk_ack", accepted=True, why=why, late_ack=late, deadline=armed,
                              ack_action=ack.get("action"), _ack_ts=ts, _ack_wall=wall, stand_seen=None)
                    self._end_stand_observation(trial, f"걷기 버튼 실행({cur['label']})")
                else:
                    self._set(trial, cur, "rejected", "walk_ack", accepted=False, why=why, late_ack=late, by="bridge")
            return
        if cmd is not None and isinstance(rid, str) and rid == cmd["req_id"]:
            brief["match"] = "cmd"
            if cmd["stage"] in ("sent", "no_ack"):
                self._set(trial, cmd, "accepted" if accepted else "rejected", "walk_cmd_ack", accepted=accepted, why=why)
                if accepted and cur is not None:
                    if cmd["kind"] == "off" and cur["stage"] == "walking":
                        self._end(trial, cur, "off", ts, wall, end_why="운영자 걷기 끄기")
                    elif cmd["kind"] == "stop" and cur["stage"] in WALK_ACTIVE:
                        self._set(trial, cur, "interrupted", "walk_interrupted", why="운영자 정지 래치(stop)")

    # ---- 주기 갱신(브라우저와 무관하게 콘솔 샘플러가 부른다) ----
    def update(self, trial):
        inbox, self._inbox = self._inbox, []
        for item in inbox:
            self._on_ack(trial, item)
        now = trial.clock()
        st = self.bridge_status(trial)
        w3, sup = st["w3"], st["sup"] or {}
        latched = sup.get("stop_latched") is True
        if st["walking"]:        # 응답을 못 받았거나 콘솔 밖 명령이어도, 브리지가 걷는 중이면 서기 관찰은 끝난 것이다
            self._end_stand_observation(trial, "걷기 시작(브리지 want/ge 1)")
        # 브리지 감독기의 켜기 조건(양발 접촉·|ω_xy| 조용함이 각각 연속 N 초)을 콘솔 표본으로 따라 센다. 걷기 직후 곧바로 누른 버튼이
        # 브리지에서 '양발 접촉 0.5s < 1s' 로 거부되는 일을 줄이는 **표시용** 문턱이다 — 최종 판정은 브리지가 한다(reason 참조).
        view = trial.view or {}
        ready = view.get("walk_ready") if isinstance(view.get("walk_ready"), dict) else None
        gyro = (view.get("imu") or {}).get("gyro")
        quiet = (ready is None or not finite(ready.get("quiet_w"))
                 or (isinstance(gyro, (list, tuple)) and len(gyro) >= 2 and finite(gyro[0]) and finite(gyro[1])
                     and math.hypot(gyro[0], gyro[1]) < ready["quiet_w"]))
        steady = st["standing"] and (view.get("foot") or {}).get("contact") == [True, True] and quiet
        self._steady_since = (self._steady_since if self._steady_since is not None else now) if steady else None
        cur = self.cur
        if cur is not None:
            if trial.fault and (cur["stage"] in WALK_ACTIVE or (cur["stage"] == "no_ack" and not cur.get("stand_seen"))):
                self.interrupt(trial, f"정지·고장: {trial.fault}")
            if cur["stage"] == "sent" and now - cur["_t_sent"] > WALK_ACK_TIMEOUT_S:
                self._set(trial, cur, "no_ack", "walk_no_ack",
                          why=f"{WALK_ACK_TIMEOUT_S:g} s 안에 응답 없음 — 재전송하지 않음. 디버그가 서기를 보일 때까지 새 걷기 버튼 차단")
            elif cur["stage"] == "walking":
                after = self._after(st, cur.get("_ack_ts"), cur.get("_ack_wall"))
                last_end, burst = sup.get("last_burst_end"), sup.get("burst")
                if isinstance(last_end, dict) and last_end.get("req_id") == cur["req_id"]:
                    self._end(trial, cur, "burst", st["ts"], st["recv"], note="burst_end 응답 미수신 — 디버그의 기한 종료 기록으로 확인")
                elif after and isinstance(burst, dict) and burst.get("req_id") != cur["req_id"]:
                    self._set(trial, cur, "interrupted", "walk_interrupted", why="다른 요청이 걷기 명령을 넘겨받았습니다 — 콘솔 밖 명령 확인")
                elif after and w3 is not None and w3.get("want") is False:
                    if latched:
                        self._set(trial, cur, "interrupted", "walk_interrupted", why=f"브리지 정지 래치({sup.get('latch_reason')})")
                    else:       # 끄기·기한 종료 응답은 못 받았지만 브리지가 걷기 요청을 내렸다(want 0)
                        self._end(trial, cur, "observed", st["ts"], st["recv"])
            elif cur["stage"] == "ending":
                after = self._after(st, cur.get("_end_ts"), cur.get("_end_wall"))
                if after and st["standing"]:
                    took = cur["_t_ending"] - cur["_t_walking"] if "_t_walking" in cur else None
                    self._set(trial, cur, "done", "walk_done", stand_seen=self._wall(trial),
                              walked_s=round(took, 2) if took is not None else None,
                              settle_s=round(now - cur["_t_ending"], 2))
                elif after and latched:
                    self._set(trial, cur, "interrupted", "walk_interrupted", why=f"브리지 정지 래치({sup.get('latch_reason')})")
                elif after and w3 is not None and w3.get("want") is True:
                    self._set(trial, cur, "interrupted", "walk_interrupted", why="걷기가 다시 켜졌습니다 — 콘솔 밖 명령 확인")
            if cur["stage"] in ("no_ack", "interrupted") and not cur.get("stand_seen"):
                if st["walking"] and not cur.get("seen_walking"):
                    cur["seen_walking"] = True          # 응답은 없었지만 디버그상 걷는 중 — 화면에 그대로 보인다
                    self.rev += 1
                t_ref = cur.get("wall_" + cur["stage"])
                if st["standing"] and finite(st["recv"]) and finite(t_ref) and st["recv"] > t_ref:
                    cur["stand_seen"] = self._wall(trial)
                    self.rev += 1
                    trial.emit("walk_standing_confirmed", req_id=cur["req_id"], walk=cur.get("name"), stage=cur["stage"],
                               message=f"{cur['label']} · 디버그에서 서기(want 0·ge 0) 확인")
        cmd = self.cmd
        if cmd is not None and cmd["stage"] == "sent" and now - cmd["_t_sent"] > WALK_ACK_TIMEOUT_S:
            self._set(trial, cmd, "no_ack", "walk_cmd_no_ack", why=f"{WALK_ACK_TIMEOUT_S:g} s 안에 응답 없음 — 디버그 상태로 확인")

    # ---- 서버 선행조건(브리지가 최종 판정, 콘솔도 거른다) ----
    def session(self, trial):
        """이 시험이 걷기 세션인가 — 실행 중(디버그)·로드(프로세스 출력)·선택 순으로 본 걷기 키, 없으면 None."""
        b = (trial.view or {}).get("bridge") or {}
        return next((k for k in (b.get("checkpoint"), b.get("loaded_checkpoint"), trial.checkpoint) if k in WALK_KEYS), None)

    def safe_reason(self, trial):
        """걷기 끄기·정지 래치: 걷기 세션이고 브리지가 떠 있으면 언제든(고장 래치·회수 중·진행 중 단계와 무관)."""
        view = trial.view or {}
        b = view.get("bridge") or {}
        if self.session(trial) is None and not isinstance(b.get("w3_sup"), dict):
            return "걷기 세션이 아닙니다 — 선택·로드된 정책이 걷기 실험 키가 아님"
        if not ((view.get("procs") or {}).get("bridge", {}).get("alive") or (view.get("fresh") or {}).get("bridge")):
            return "브리지가 실행 중이 아닙니다"
        return None

    def reason(self, action, trial):
        """걷기 버튼(w3_move)·재무장(w3_rearm). 고장 래치·회수 중·추적 밖 제어기·진행 중 단계는 TrialState.reason 이 먼저 거른다."""
        view = trial.view or {}
        if view.get("walk_error"):
            return f"걷기 모듈을 불러오지 못했습니다 — {view['walk_error']}"
        if self.broken:
            return f"걷기 패널 내부 오류 — 걷기 버튼을 잠갔습니다({self.broken}). 콘솔 로그 확인"
        b, s, fresh = view.get("bridge") or {}, view.get("stage8") or {}, view.get("fresh") or {}
        ck = trial.checkpoint
        if ck not in WALK_KEYS:
            return "선택한 정책이 걷기 실험 키가 아닙니다 — 설정·점검에서 걷기 실험 키를 고른 시험에서만"
        if "stand" not in trial.completed:
            return "정책 실행 단계가 아닙니다 — 정책 제어 시작 확인 뒤에만"
        if not (view.get("procs") or {}).get("bridge", {}).get("alive"):
            return "브리지 프로세스가 콘솔 추적 아래 실행 중이 아닙니다"
        if not view.get("process_check_ok"):
            return "잔존 제어 프로세스 확인 필요"
        r = trial._sensor_reason(True)
        if r:
            return r
        if not fresh.get("bridge"):
            return "정책 상태(디버그 토픽) 수신 없음/오래됨"
        if trial.start_gate_waiting(b):
            return "착지 대기 중 — 정책이 아직 구동되지 않았습니다"
        if b.get("live") is not True:
            return "브리지가 live 가 아닙니다"
        if not (b.get("armed") and s.get("armed") and s.get("running")):
            return "제어기 ARM·정책 명령 실행 상태 확인 필요"
        if b.get("loaded_checkpoint") != ck or b.get("checkpoint") != ck:
            return "실제 로드·실행 중인 정책이 선택한 걷기 키와 다릅니다"
        st = self.bridge_status(trial)
        w3, sup, kew3 = st["w3"], st["sup"], st["kew3"]
        if w3 is None or sup is None:
            return "브리지가 걷기 세션 상태(w3·w3_sup)를 보내지 않습니다 — 실험 플래그 기동 확인"
        if WALK_KEYS[ck]["composite"] and kew3 is None:
            return "합성 러너 상태(kew3)를 받지 못했습니다"
        cur, cmd = self.cur, self.cmd
        if cur is not None and cur["stage"] in WALK_ACTIVE:
            return f"앞 걷기 버튼 진행 중 — {cur['label']} · {WALK_STAGE_LABEL[cur['stage']]}"
        if cur is not None and cur["stage"] == "no_ack" and not cur.get("stand_seen"):
            return "앞 걷기 버튼 응답 없음 — 디버그가 서기(want 0·ge 0)를 보일 때까지 대기"
        if cmd is not None and cmd["stage"] == "sent":
            return f"앞 명령({cmd['label']}) 응답 대기 중"
        if sup.get("freeze"):
            return f"가드 동결 대기 중({sup.get('freeze_why')})"
        want, ge, latched = w3.get("want"), w3.get("ge"), sup.get("stop_latched")
        if type(want) is not bool or type(ge) is not bool or type(latched) is not bool:
            return "브리지 걷기 상태(want·ge·정지 래치)를 읽지 못했습니다"      # 읽지 못한 값을 '서 있음·래치 없음'으로 보지 않는다
        if action == "w3_rearm":
            if not latched:
                return "정지 래치가 없습니다 — 재무장할 것이 없음"
            if want or ge:
                return "서기 전환 뒤에만 재무장할 수 있습니다(지금 want/ge 1)"
            return None
        if latched:
            return f"정지 래치({sup.get('latch_reason')}) — 재무장 뒤에만 걷기 가능"
        if want or ge:
            return "지금 걷는 중입니다(want/ge 1) — 서기 복귀 뒤에"
        if WALK_KEYS[ck]["composite"] and kew3.get("mode") == "air" and kew3.get("air_hold"):
            # 1008: 공중 폭주 차단 래치(러너가 목표를 고정했다) — 이 세션은 자세 고정만 한다. 정지·회수 뒤 새로 시작해야 풀린다
            return "공중 폭주 차단(기본자세 유지 중) — 정지한 뒤 처음부터 다시 시작하세요"
        if WALK_KEYS[ck]["composite"] and kew3.get("mode") == "air":
            # 1008: 공중 정책 시작 세션의 공중·착지 구간 — 착지가 가라앉아 브리지 러너가 KE 로 넘길 때까지 걷기 불가(브리지도 러너 게이트·감독기로 거부한다)
            return "공중·착지 정책 구간(착지 정착 뒤 KE 인계 대기) — 서기(KE) 인계 뒤에"
        if WALK_KEYS[ck]["composite"] and kew3.get("mode") != "ke":
            return "합성 러너가 아직 걷기 정책(W3) 모드입니다 — 서기(KE) 인계 뒤에"
        if WALK_KEYS[ck]["composite"] and kew3.get("block"):
            return f"브리지 러너 선행조건: {kew3['block']}"
        if (view.get("foot") or {}).get("contact") != [True, True]:
            return "양발 접촉 확인 필요"
        if finite(b.get("blend")) and b["blend"] < 1.0:
            return "정책 인계 직후의 온셋 블렌딩 진행 중"
        ready = view.get("walk_ready") if isinstance(view.get("walk_ready"), dict) else None
        if ready is not None and finite(ready.get("contact_s")) and finite(ready.get("quiet_s")):
            need = max(ready["contact_s"], ready["quiet_s"]) + WALK_SETTLE_MARGIN_S
            held = trial.clock() - self._steady_since if self._steady_since is not None else 0.0
            if held < need:
                return f"서기 정착 대기 중 — 조용한 양발 서기 {max(0.0, held):.1f} / {need:.2f} s (브리지 켜기 조건)"
        return None

    # ---- 화면·기록용 ----
    @staticmethod
    def _public(rec):
        if rec is None:
            return None
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        out["stage_label"] = WALK_STAGE_LABEL.get(rec["stage"], rec["stage"])
        out["text"] = WalkState.describe(rec)
        return out

    def snapshot(self, trial):
        view = trial.view or {}
        b = view.get("bridge") or {}
        st = self.bridge_status(trial)
        w3, sup = st["w3"] or {}, st["sup"] or {}
        now, stamp = trial.clock(), view.get("server_time")
        cur = self._public(self.cur)
        if cur is not None:
            raw = self.cur
            overdue = False                      # 표시 전용 — 콘솔은 끄기를 보내지 않는다(운영자가 '걷기 끄기'·정지 요청으로 판단)
            if raw["stage"] == "walking" and finite(raw.get("dur_s")):
                overdue = now - raw["_t_walking"] > raw["dur_s"] + WALK_END_GRACE_S
            elif raw["stage"] == "ending":
                overdue = now - raw["_t_ending"] > WALK_STAND_GRACE_S
            cur["overdue"] = overdue
            cur["blocking"] = raw["stage"] in WALK_ACTIVE or (raw["stage"] == "no_ack" and not raw.get("stand_seen"))
        burst = sup.get("burst") if isinstance(sup.get("burst"), dict) else None
        remaining = None
        if burst is not None and finite(burst.get("t_off")) and finite(st["ts"]):
            age = stamp - st["recv"] if finite(stamp) and finite(st["recv"]) else 0.0
            remaining = round(max(0.0, burst["t_off"] - st["ts"] - max(0.0, age)), 2)
        w3c = b.get("w3_cmd") if st["fresh"] and isinstance(b.get("w3_cmd"), dict) else None
        key = self.session(trial)
        if st["sup"] is not None:
            guard = b.get("guard_mode") if isinstance(b.get("guard_mode"), str) else "safe"
        else:
            guard = self.guard_mode
        return {
            "rev": self.rev,
            "session": key is not None or isinstance(b.get("w3_sup"), dict),
            "profile": ({"checkpoint": key, **WALK_KEYS[key]} if key else None),
            "current": cur,
            "command": self._public(self.cmd),
            "bridge": {
                "fresh": st["w3"] is not None, "want": w3.get("want"), "ge": w3.get("ge"), "ge_obs": w3.get("ge_obs"),
                "standing": st["standing"], "walking": st["walking"],
                "stop_latched": sup.get("stop_latched"), "latch_reason": sup.get("latch_reason"),
                "freeze": sup.get("freeze"), "freeze_why": sup.get("freeze_why"), "last_seq": sup.get("last_seq"),
                "burst": ({"req_id": burst.get("req_id"), "dur_s": burst.get("dur_s"), "remaining_s": remaining}
                          if burst is not None else None),
                "last_burst_end": sup.get("last_burst_end") if isinstance(sup.get("last_burst_end"), dict) else None,
                "contact": sup.get("contact"), "tilt_deg": sup.get("tilt_deg"), "w_xy": sup.get("w_xy"),
                "cmd": b.get("cmd") if st["fresh"] and isinstance(b.get("cmd"), list) else None,
                "heading_loop": (bool(w3c.get("hl_on")) and w3c.get("hl") is not None) if w3c is not None else None,
            },
            # 합성 키: 지금 도는 러너("ke" 서기 | "w3" 걷기 | 1008 "air" 공중·착지 구간 — 공중 정책 시작 키만). W3 단독 키는 서기·걷기가 같은 정책이라 늘 "w3".
            "mode": st["mode"] if st["composite"] else ("w3" if st["w3"] is not None else None),
            "composite": st["composite"],
            "kew3": st["kew3"],
            "guard_mode": guard,
            "recent_acks": list(self.acks[-5:]),
        }


class TrialState:
    def __init__(self, checkpoint, clock=time.monotonic):
        self.clock = clock
        self.boot_id = uuid.uuid4().hex
        self.revision = 0
        self.number = 1
        self.mode = default_mode(checkpoint)   # 1008: 공중 정책 시작을 받는 키면 "air", 그 밖은 종전대로 "ground"
        self.checkpoint = checkpoint
        self.completed = set()
        self.pending = None
        self.fault = None
        self.last_result = None
        self.support = False
        self.release_no_bus = False
        self.stop_request_at = None
        self.stop_request_seq = -1
        self.stop_delivery = None
        self.measurement = {"state": "waiting", "elapsed": 0.0, "reason": "모든 줄 느슨·무보조 확인 전"}
        self.slack_at = None
        self.commander_at_slack = None   # 무보조 확인 시점의 명령 주체 — 홀드/정책 혼동 방지
        self.max_elapsed = 0.0
        self.view = {}
        self.history = []
        self.changes = []
        self._pose_stamps = None
        self.macro = None        # 1005 정책 ON 자동 순서 진행 상태 {action, step, started} — 진행 중 다른 조작 차단
        self.policy_on = False   # 1005 이 시험이 '정책 ON(착지 자동 인수)' 경로인지 — 접지 마커·stand 의 접촉 게이트를 브리지 착지 게이트가 대신한다
        #                          (1008 공중 방식의 정책 ON 에서도 True — 그때 접촉 게이트는 방식 자체가 요구하지 않고, 브리지는 HangGuard 행잉 모드 + 러너 AIR 모드로 받는다)
        self.walk = WalkState()  # 1007 걷기 버튼 수명 상태기(W3 세션에서만 움직인다) — 시험마다 새로

    def emit(self, event, **data):
        row = {"event": event, "trial": self.number, **data}
        self.changes.append(row)
        self.history.append(row)
        self.history = self.history[-60:]

    def drain_events(self):
        rows, self.changes = self.changes, []
        return rows

    @staticmethod
    def start_gate_waiting(b):
        """1005: 브리지가 착지 자동 인수 대기(정책 미구동, 홀드 명령 스트리밍) 중인가."""
        g = (b or {}).get("start_gate")
        return isinstance(g, dict) and g.get("state") not in (None, "POLICY")

    def policy_commanding(self, b):
        """브리지가 실제로 '정책' 목표를 내고 있는가 — 착지 대기 중의 브리지 홀드는 정책이 아니다."""
        return bool((b or {}).get("armed")) and not self.start_gate_waiting(b)

    def commander(self):
        """지금 실제로 명령을 내고 있는 제어기 — "policy" | "hold" | None.

        0928: 무보조 확인(op_slack_done)이 정책 armed 만 허용하고 있어서, 홀드만으로 선
        자립을 기록할 방법이 없었다(70개 세션 전수 op_slack_done 0건 — 못 섰기 때문이 아니라
        누를 수 없었기 때문). 프로젝트 판정 기준은 '모든 줄 느슨·무보조 5초'이고 어떤 제어기가
        명령 중인지는 규정하지 않는다. 다만 **홀드 자립과 정책 자립은 절대 섞지 않는다** —
        그래서 제어기를 반환하고 측정 기록에 남긴다.
        """
        s = self.view.get("stage8", {}) or {}
        b = self.view.get("bridge", {}) or {}
        fresh = self.view.get("fresh", {}) or {}
        if "stand" in self.completed and fresh.get("bridge") and self.policy_commanding(b):
            return "policy"
        # 홀드: stage8 이 ARM 되어 스트리밍 중이고 정책은 명령하지 않는 상태 (1005: 착지 대기 브리지 홀드 포함)
        if ("arm" in self.completed and fresh.get("status")
                and s.get("armed") and s.get("running") and not self.policy_commanding(b)):
            return "hold"
        return None

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
        self.walk.interrupt(self, f"정지·고장: {reason}")   # 1007 진행 중인 걷기 버튼은 '중단'으로 닫는다(없으면 무동작)

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

    def reason(self, action, _macro=False):
        """_macro=True: 정책 ON 순서기가 하위 단계를 실행할 때만 — 순서기 진행 중 차단을 건너뛰고 나머지 게이트는 그대로."""
        if action == "estop":
            return None
        if action not in LABELS:
            return "현재 기립 시험에서 제공하지 않는 동작입니다"
        if action in WALK_SAFE:
            # 1007 걷기 끄기·정지 래치 = 안전 방향: 진행 중 단계·고장 래치·회수 중에도 받는다(걷기 세션·브리지 실행만 확인)
            try:
                return self.walk.safe_reason(self)
            except Exception as exc:
                self.walk.fail(self, exc)
                return None                            # 판정 자체의 오류가 안전 방향 명령을 막지 않게
        if self.macro and not _macro and not self.fault and action not in EVENTS | {"support_confirm"}:
            return (f"정책 ON 자동 진행 중({LABELS.get(self.macro.get('step'), '시작')}) — 중단은 '정지 요청'")
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
                if self.commander() is None:
                    return ("명령 중인 제어기 확인 필요 — 정책 제어 중이거나, "
                            "stage8 홀드가 ARM·스트리밍 중이어야 합니다")
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
        if action == "force_reset":
            # 0918: 가드 릴리즈·모터 전원 차단 뒤 토크 해제 단계가 CAN 무응답으로 못 끝나 새 시험이 막히는 경우.
            # 운영자가 '무토크(전원 꺼짐)'를 선언하고 단계를 초기화한다. 살아 있는 제어 프로세스가 있으면 거부(먼저 stack_down).
            if owners or external:
                return "제어 프로세스가 살아 있습니다 — 먼저 제어 프로세스 종료(stack_down) 뒤 강제 초기화"
            if fresh.get("status") and (s.get("armed") or s.get("holding")):
                return "stage8 가 무장/홀드 중으로 보고합니다 — 제어 프로세스 종료 먼저"
            return None
        if self.fault:
            return "정지·고장 기록이 남아 있습니다. 회수 후 새 시험을 시작하세요"
        if self.support:
            return "회수 중입니다. 회수 완료 후 새 시험을 준비하세요"
        if external:
            return "콘솔 추적 밖 제어기 감지 — 기존 제어 상태를 확인하세요"
        if action in ("w3_move", "w3_rearm"):          # 1007 걷기 버튼·재무장 — 브리지가 최종 판정, 콘솔도 거른다
            try:
                return self.walk.reason(action, self)
            except Exception as exc:                   # 판정 중 예외 = 허용하지 않는다(잠금) — 기립 시험 쪽으로는 번지지 않게 가둔다
                self.walk.fail(self, exc)
                return f"걷기 패널 내부 오류 — 걷기 버튼을 잠갔습니다({self.walk.broken}). 콘솔 로그 확인"
        if action == "policy_on":
            return self._policy_on_reason()
        if action in MANUAL:
            # 0914 개별 제어기: 자세 유지 스크립트가 살아 있고 운용 게인(kp150)에 도달한 뒤, 정책 브리지가 없는 동안만.
            proc = self.view.get("procs", {}).get("hang_stand", {})
            hold = proc.get("hold", {})
            if not proc.get("alive") or self.view.get("server_time", 0) - hold.get("ts", 0) >= 3:
                return "자세 유지 프로세스의 신선한 상태 확인 필요"
            if hold.get("kp", 0) < 150:
                return "운용 게인(kp150) 도달 뒤에만 개별 조정이 가능합니다"
            if "handoff" in self.completed or self.view.get("procs", {}).get("bridge", {}).get("alive") or b.get("live"):
                return "제어 인계·정책 브리지 이후에는 개별 조정이 불가합니다"
            if not self.view.get("process_check_ok"):
                return "잔존 제어 프로세스 확인 필요"
            if action == "air_gait_on":
                # 0915 사진용 공중 제자리걸음: 접지 선언·발 접촉이 있으면 거부(지면에서 다리를 휘두르지 않게)
                hh = self.view.get("hang_hold") or {}
                if hh.get("grounded") or hh.get("phase") == "하중유지":
                    return "접지 상태에서는 에어 보행을 켤 수 없습니다"
                if any(bool(c) for c in (self.view.get("foot", {}).get("contact") or [])):
                    return "발 접촉 감지 — 공중(발이 바닥에서 떨어진 상태)에서만"
            return None
        if action in ZERO_TOOLS:
            # 영점 쓰기/검증은 제어 시작 전에만(인수 대기 stage8 은 무방). 어긋난 모터가 있을 때 boot_pose 결과 보고 누른다.
            if self.active_processes() or self.completed.intersection({"hang_stand", "bridge"}) or b.get("live"):
                return "제어 시작 전에만 영점 도구를 쓸 수 있습니다"
            if not self.view.get("process_check_ok"):
                return "잔존 제어 프로세스 확인 필요"
            return None
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
        if action == "ground" and self.mode == "ground" and not self.policy_on:
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
            if self.mode == "ground" and not self.policy_on:
                return self._contact_reason()
        return None

    def _policy_on_reason(self):
        """1005 정책 ON 선행조건 — 공중 수평 자세 유지(kp150)·인수 대기 상태에서만. 기존 단계 게이트는 순서기가 단계마다 다시 검사한다."""
        if "policy_on" in self.completed or self.macro:
            return "이미 정책 ON 을 실행했습니다"
        missing = next((k for k in PREP if k not in self.completed), None)
        if missing:
            return f"먼저 {LABELS[missing]} 단계를 완료하세요"
        done = [k for k in POLICY_ON_STEPS if k in self.completed]
        if done:
            return f"단계별 절차가 이미 진행 중입니다({LABELS[done[-1]]} 완료) — 그 절차로 계속하세요"
        if not str(self.checkpoint).startswith("stand"):
            return "기립 정책만 정책 ON 을 쓸 수 있습니다"
        if not self.view.get("process_check_ok"):
            return "잔존 제어 프로세스 확인 필요"
        r = self._sensor_reason()
        if r:
            return r
        s, fresh = self.view.get("stage8", {}), self.view.get("fresh", {})
        proc = self.view.get("procs", {}).get("hang_stand", {})
        hold = proc.get("hold", {})
        if not proc.get("alive") or self.view.get("server_time", 0) - hold.get("ts", 0) >= 3:
            return "자세 유지 프로세스의 신선한 상태 확인 필요"
        if hold.get("kp", 0) < 150:
            return "운용 게인(kp150) 도달 확인 필요"
        if not (fresh.get("status") and s.get("waiting")):
            return "인수 제어기가 실제로 대기 중이어야 합니다"
        if self.view.get("foot", {}).get("contact") == [True, True]:
            return "양발 접촉 중 — 정책 ON 은 매달린 상태(발이 바닥에서 떨어짐)에서만. 지상 시작은 기존 단계별 절차를 쓰세요"
        if self.view.get("demo"):
            return None
        hh = self.view.get("hang_hold")
        if not isinstance(hh, dict):
            return "자세 유지 상태 파일(관절 조정·보행) 확인 필요"
        if (hh.get("gait") or {}).get("active"):
            return "에어 보행 정지 후 정책 ON"
        jt = hh.get("joint_trim") or {}
        ref = self.view.get("level_pose_ref") or {}
        if not ref:
            return "수평 기준 자세(level_pose) 확인 필요"
        for j, r in ref.items():
            v = jt.get(j) or {}
            req, cur = v.get("req"), v.get("cur")
            if not (finite(req) and finite(cur)):
                return f"{j} 개별 조정 상태 확인 필요"
            if abs(req - cur) > POLICY_ON_RAMP_TOL_DEG:
                return f"수평 자세 램프 진행 중({j.replace('_joint', '')} {cur:+.1f}→{req:+.1f}°)"
            if abs(cur - r) > POLICY_ON_LEVEL_TOL_DEG:
                return (f"수평 기준 자세 적용 필요 — {j.replace('_joint', '')} {cur:+.1f}° vs 기준 {r:+.1f}° "
                        f"(허용 ±{POLICY_ON_LEVEL_TOL_DEG:g}°)")
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
            self.commander_at_slack = self.commander()
            _who = {"policy": "정책", "hold": "홀드(정책 없음)"}.get(self.commander_at_slack, "미확인")
            self.measurement = {"state": "measuring", "elapsed": 0.0,
                                "commander": self.commander_at_slack,
                                "reason": f"운영자 무보조 확인 · 명령 주체 {_who} · 영상 검토 필요",
                                "review_required": True}
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
                    rc = hp.get("rc")
                    self.latch("자세 유지 프로세스 종료 — 회수 상태 확인 필요"
                               + (f" ({explain_rc('hang_stand', rc)})" if rc not in (None, 0, 3) else ""))
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
        # 1007 걷기 버튼 수명 갱신(응답·디버그 상태 반영) — 위의 고장 래치 판정 뒤에, 매번(아래 단계 응답 확인의 조기 반환과 무관).
        # 서기 관찰(아래)보다 먼저 돈다: 걷기가 시작되면 관찰 구간을 '걷기 시작' 사유로 끝낸다(한발 지지의 접촉 상실 사유로 남기지 않게).
        # 걷기 패널은 실험 기능이다 — 그 안의 예외가 기립 시험 상태기를 멈추거나 '상태·기록 처리 오류' 고장 래치로 번지지 않게 가둔다.
        try:
            self.walk.update(self)
        except Exception as exc:
            self.walk.fail(self, exc)
        if self.slack_at is not None:
            reason = self._sensor_reason(True) or self._contact_reason()
            if not view.get("session", {}).get("active"):
                reason = "기록 중단"
            _who = getattr(self, "commander_at_slack", None)
            if _who == "policy":
                if not s.get("armed") or not s.get("running") or not self.policy_commanding(b):
                    reason = "정책 제어 상태 상실"
            elif _who == "hold":
                # 홀드 측정 중 정책이 명령을 잡으면 그 시점부터는 다른 시험이다 — 무효 처리.
                if not s.get("armed") or not s.get("running"):
                    reason = "홀드 제어 상태 상실"
                elif self.policy_commanding(b):
                    reason = "측정 중 정책이 명령을 잡음 — 홀드 자립 측정 무효"
            else:
                reason = "명령 주체 미확인"
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
               "gain_ramp": "hang_stand", "ground": "hang_stand", "handoff": "hang_stand",
               "zero_multiple": "zero", "zero_recapture": "zero", "zero_preview": "zero", "zero_kin": "zerokin"}.get(action, action)
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
        elif action in ("zero_multiple", "zero_recapture", "zero_preview", "zero_kin"):
            # 0915: 영점 도구는 프로세스 정상 종료(rc 0)가 완료. (이전엔 분기가 없어 성공해도 30 s 뒤 '시간 초과' 오판)
            ok = proc.get("rc") == 0 and not proc.get("alive")
        elif action in ("boot_pose", "selftest", "release", "analyze"):
            ok = proc.get("rc") == 0 and not proc.get("alive")
            if action == "selftest":
                ok = ok and proc.get("all_pass", False)
            if action == "release":
                # 0922: 버스 무응답(응답 0/12)은 '실패' 가 아니라 **CAN 으로 해제할 수 없는 상태**다.
                #   모터 전원이 꺼지면 ACK 가 없어 송신이 전부 실패하므로, 종전 로직은 토크 해제 단계를 영구히 막았고
                #   운영자가 robotctl reset(강제 초기화)을 따로 찾아야 했다(0918·0921 두 번 발생).
                #   CAN 으로 토크를 걸 수도 없는 상태이므로 단계는 넘기되, 사유를 남기고 실토크 현장 확인을 요구한다.
                verdict = proc.get("release_verdict")
                if not proc.get("alive") and verdict == "NO_BUS":
                    self.release_no_bus = True
                    return self.finish(True, "버스 무응답(응답 0/12) — CAN 으로 토크를 걸 수도 해제할 수도 없음. "
                                             "모터 전원 차단을 현장에서 확인하세요")
                if not proc.get("alive") and (proc.get("release_error") or verdict == "PARTIAL"):
                    return self.finish(False, "일부 모터만 응답/송신 실패 — 실제 토크 상태 확인 필요")
                ok = ok and proc.get("release_ok", False)
        if ok:
            message = "해제 명령 전송 완료 보고 · 실제 토크 상태는 현장 확인 필요" if action == "release" else "실제 응답 확인 완료"
            return self.finish(True, message)
        if proc and not proc.get("alive") and proc.get("rc") not in (None, 0, 3) and not (action == "release" and proc.get("rc") == 4):
            return self.finish(False, explain_rc(action, proc["rc"]))
        # 0911 호환검수 R2: stand 는 브리지 인계 워밍업(최대 4 s)+블렌딩 안에 응답하므로 30 s 는 브리지발 정지를 26 s 동안 가린다
        timeout = 300 if action in ("boot_pose", "selftest", "analyze", "zero_multiple", "zero_recapture", "zero_preview", "zero_kin") else (8 if action == "stand" else 30)
        if self.clock() - p["started"] > timeout:
            self.finish(False, "확인 시간 초과 — 실제 상태를 확인하세요")

    def snapshot(self):
        s, b = self.view.get("stage8", {}), self.view.get("bridge", {})
        fresh = self.view.get("fresh", {})
        owners = self.active_processes()
        if "release" in self.completed:
            phase = "회수 명령 완료"
            owner = ("버스 무응답으로 해제 송신 불가(전원 차단 추정) · 실토크 현장 확인 필요"
                     if getattr(self, "release_no_bus", False) else "토크 해제 전송 완료 보고 · 실토크 미검증")
        elif self.fault:
            phase, owner = "정지·상태 확인 필요", "실제 상태 확인 필요"
        elif self.measurement["state"] == "interrupted":
            if self.measurement.get("ended_by") == "walk" and self.commander() == "policy":
                # 1007: 걷기 버튼이 서기 관찰 구간을 끝낸 경우 — 보조·가드로 깨진 것이 아니므로 '회수 필요'로 표시하지 않는다
                phase, owner = "정책 제어 · 걷기 조작(서기 관찰 종료)", "정책 → 12축 제어기"
            else:
                phase, owner = "무보조 관찰 중단 · 회수 필요", "정책 실행 상태는 별도 확인"
        elif self.macro:
            phase, owner = f"정책 ON 진행 중 · {LABELS.get(self.macro.get('step'), '시작')}", "기존 단계 자동 진행(같은 게이트·응답 확인)"
        elif fresh.get("bridge") and self.start_gate_waiting(b) and fresh.get("status") and s.get("running"):
            phase, owner = "착지 대기 · 수평 자세 홀드(정책 미구동)", "브리지 홀드 → 12축 제어기 · 양발 착지 확인 뒤 정책 자동 시작"
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
            allowed["policy_on"]["label"] = POLICY_ON_LABEL_AIR     # 1008: 공중 방식의 정책 ON 은 착지를 기다리지 않는다 — 이름으로 드러낸다
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
                "macro": (dict(self.macro) if self.macro else None), "policy_on": self.policy_on,
                "history": list(self.history)}
