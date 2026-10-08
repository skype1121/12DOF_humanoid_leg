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
import html
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
    from .console_state import TrialState, LABELS, EVENTS, CONTROL, POLICY_ON_STEPS, finite
    from .console_state import WALK_KEYS, WALK_KIND, WALK_SAFE, WALK_ACK_TIMEOUT_S
except ImportError:
    from console_state import TrialState, LABELS, EVENTS, CONTROL, POLICY_ON_STEPS, finite
    from console_state import WALK_KEYS, WALK_KIND, WALK_SAFE, WALK_ACK_TIMEOUT_S

# --- 1007 걷기 조작(시간 제한 걷기 버튼) 모듈 ---
# 버튼 값의 단일 원본 = rl_walking/deploy/w3_deploy.W3_MOVES·w3_move_cmd, 명령 순번 = w3_cmd.next_seq(파일 잠금 단조 seq, CLI 와 같은 파일).
# 이 파일들이 배포에 없거나 옛 판(W3_MOVES 없음)이면 걷기 패널만 '사용 불가'로 두고 콘솔의 나머지는 종전대로 뜬다.
try:
    try:
        from . import w3_cmd as _W3CMD
    except ImportError:
        import w3_cmd as _W3CMD
    _W3 = _W3CMD._W3
    for _name in ("W3_MOVES", "w3_move_cmd", "W3S3D_CKPT_KEY"):
        getattr(_W3, _name)
    getattr(_W3CMD, "next_seq")
    WALK_IMPORT_ERROR = None
except Exception as _exc:      # ImportError(파일·numpy 없음)·AttributeError(옛 판) 모두 — 콘솔 기동을 막지 않는다
    _W3CMD = _W3 = None
    WALK_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

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
BRIDGE_ACK_TOPIC = "/humanoid/rl_bridge_ack"     # 1007: W3 세션 브리지만 발행하는 명령 응답(rl_bridge_node.BRIDGE_ACK_TOPIC) — 걷기 버튼 수락/거부/기한 종료

# 0922: BODY_WEIGHT_N 제거 — 감압 패드는 접지 판단용이며, 카운트를 체중으로 나눈 '발력 %' 는 만들지 않는다.
BRIDGE_JOINT_FRESH_SEC = 0.2  # rl_bridge_node.SENSOR_STALE_SEC 와 동일 — '12/12 신선' 판정 기준 (0911 R3)

#: 정책 드롭다운 — 첫 항목이 기본값(7차 데뷔 후보). 브리지 CHECKPOINTS 의 부분집합이어야 한다(selftest ⑧).
#: 0907: stand_v14 제거 — 패키지 GENS(v15~v18b)에 없어 젯슨에 파일이 없다(선택 시 로드 실패). 젯슨 구 자산 정리로
#: walk/march/drv2/march_drv2/march_v3/stand_v3/v5/v6 도 삭제됨 — 젯슨 잔존 = v15~v18b + v7/v8/v9/march_v5b.
# 0922: stand_crouch_v5 를 목록 **맨 앞(기본값)** 으로. 기본자세가 실물 v5 스탠바이(힙 ±30.26·무릎 ∓38.50·발목 ±28.00)와
#   일치하는 유일한 정책이다. stand_crouch_v1* 는 기본자세 힙 15° 라 인계 블렌딩이 '힙 15° 펴기' 명령이 되어
#   0918 실물 stand 낙상을 만들었다 — 목록에는 남기되 기본값에서 뺀다.
# 0929: 승윤님 지시 — 의미 없는 이전 세대는 목록에서 뺀다. 오늘 비교 시험에 쓰는 두 개만 남긴다.
#   stand_crouch_g11a = 고친 플랜트(발목 정지 마찰·편향 절반)로 학습한 비교 후보(승격 아님)
#   stand_crouch_v6   = 같은 날 대조군(0922 배포본)
#   브리지 CHECKPOINTS 등록은 그대로 둔다 — 과거 기록 재현·셀프테스트용. 콘솔 선택지만 줄인다.
# 1001: stand_crouch_g12_ema0 = 같은 G12 정책 파일, 관측 joint_vel_ema 만 0(학습 계약 복원 — 학습 중 이 채널은 0+잡음이었다).
#   이름 그대로의 stand_crouch_g12 는 0929·1002 실물과 같은 동작(실제 속도 EMA).
# 1003 승윤님 결정(배포 시): 교대 진단의 G12 = v2 그대로(1002 실물과 같은 동작) → 기본값(맨 앞) 유지. G21a = 학습 관측 필터 계약(진단 전용, 승격 아님).
#   stand_crouch_g12_ema0(학습 관측 계약) 은 선택지로만 둔다.
CKPT_CHOICES = ["stand_crouch_g12", "stand_crouch_g21a", "stand_c5_g21a_g24ra", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra", "stand_crouch_g12_ema0", "stand_crouch_g11a"]   # 1004: C5(G21a+G24R-a 전환) 구조 진단 선택지
# 1008 승윤님 지시: 안 쓰는 정책은 다 정리한다 — **고를 수 있는 정책은 합성 키 하나**(서기 KE + 걷기 W3 + 갇힘 해소 = 시뮬에서 평가한 구성).
#   화면 드롭다운·기본값·시험 설정·정책 로드·계약 검사가 받는 키 = CKPT_MENU(아래) 뿐이다. 그 밖의 키는 서버가 거부한다.
#   위 CKPT_CHOICES 와 걷기 키 목록은 "콘솔이 기동 인자를 아는 키"로만 남긴다(0929 와 같은 방식: 브리지 CHECKPOINTS 등록·정책 파일은 그대로 —
#   과거 기록 재현·selftest 용. 고를 수는 없다). 다시 늘리려면 CKPT_MENU_KEYS 에 키를 더한다.
CKPT_MENU_KEYS = ("stand_kew3q_g21a_g24ra_w3s3d15800",)
CKPT_DEFAULT = CKPT_MENU_KEYS[0]

# 1007 걷기 실험 키(콘솔 걷기 조작 패널) — 기본 선택지(CKPT_CHOICES)·기본값은 그대로 두고 **뒤에** 덧붙인다.
#   키 목록 = console_state.WALK_KEYS(selftest 가 w3_deploy·브리지 원본과 대조). 브리지는 아래 실험 플래그를 붙여야만 이 키를 등록한다
#   (정책 로드와 셀프테스트 게이트 양쪽). 기존 키의 브리지 기동 인자는 한 글자도 바뀌지 않는다(bridge_extra_args 가 빈 목록 — selftest 고정).
WALK_BRIDGE_ARGS = {
    "stand_w3s3d_15800": ["--experimental-w3"],
    "stand_kew3_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"],
    # 1008 둘째 합성 키(첫 합성 키 + KE 감지기 갇힘 해소 = 시뮬 합성 평가 구성 · 실물 미검증) — 첫 합성 키와 같은 실험 플래그. 위 두 키의 인자는 그대로다.
    "stand_kew3q_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"],
}
#: 정책 로드(--live)에만 더 붙는 인자. 헤딩 고리는 기본 off — 시뮬 판정: 왕복에서 기준 미달(reports/gen_w3/HEADING_SESSION_1007.md §7).
#: 셀프테스트 게이트에는 붙이지 않는다: 브리지 --selftest 의 'W3 감시 2 차 대비책'·'헤딩 고리 고장 판별 분리' 항목은 고리 설정이 켜진 채로만
#: 통과한다(1007 실행 대조: 두 걷기 키 모두 off 를 붙이면 2 FAILED, 빼면 30/30 ALL PASS — 원인은 이 플래그 하나). 게이트가 보는 것은
#: 정책 파일·계약이라 고리 설정과 무관하다. 브리지 selftest 가 off 를 견디게 되면 게이트에도 붙일 수 있다.
WALK_LIVE_ARGS = ["--w3-heading-loop", "off"]
#: 걷기 모듈을 못 불러왔으면 걷기 키를 선택지에 올리지 않는다(패널 없이 걷기 세션만 뜨는 상태를 만들지 않는다).
WALK_CKPTS = [k for k in WALK_KEYS if WALK_IMPORT_ERROR is None and k in WALK_BRIDGE_ARGS]
CKPT_ALL = CKPT_CHOICES + WALK_CKPTS
#: 고를 수 있는 정책(1008: 하나). 걷기 모듈을 못 불러오면 비어 있다 — 그때는 정책 로드·계약 검사가 사유를 대고 거부한다(옛 정책으로 조용히 넘어가지 않는다).
CKPT_MENU = [k for k in CKPT_MENU_KEYS if k in CKPT_ALL]


def ckpt_menu_reason(ckpt):
    """이 키를 지금 고를 수 없는 이유(없으면 None)."""
    if ckpt in CKPT_MENU:
        return None
    if ckpt in CKPT_MENU_KEYS and WALK_IMPORT_ERROR is not None:
        return f"걷기 모듈을 불러오지 못해 이 정책을 쓸 수 없습니다 — {WALK_IMPORT_ERROR}"
    if ckpt in CKPT_ALL:
        return f"정리된 정책입니다 — 콘솔에서 고를 수 있는 정책은 {', '.join(CKPT_MENU) or '없음'} 뿐입니다(1008)"
    return "허용되지 않은 체크포인트입니다"
#: 화면 방향 패드 배치(이름만) — 값(vx·vy·wz·dur_s)과 표시 이름은 w3_deploy.W3_MOVES 가 유일한 원본이다. 콘솔에 값을 다시 적지 않는다.
WALK_PAD = (("turn_l", "fwd1", "turn_r"), ("left1", "inplace", "right1"), ("fwd3", "back1"))
WALK_REQ_MAX_AGE_S = 5.0     # 걷기 버튼·재무장 요청의 허용 나이 [s](화면이 본 서버 시각 기준) — 끊겼다 뒤늦게 도착한 요청으로 걷지 않게


def _walk_ready_cfg():
    """브리지 감독기의 켜기 문턱(기본값 — 콘솔은 브리지에 --w3-safety 를 넘기지 않는다): 양발 접촉·조용한 서기의 연속 시간 [s]과 |ω_xy| 문턱 [rad/s].
    원본 = w3_deploy.W3_SAFETY_DEFAULT. 콘솔은 이 값으로 '서기 정착 대기'를 표시할 뿐이고 켜기의 최종 판정은 브리지가 한다."""
    cfg = getattr(_W3, "W3_SAFETY_DEFAULT", None) if _W3 is not None else None
    try:
        return {"contact_s": float(cfg.enable_contact_s), "quiet_s": float(cfg.enable_quiet_s), "quiet_w": float(cfg.enable_quiet_w)}
    except (AttributeError, TypeError, ValueError):
        return None


WALK_READY = _walk_ready_cfg()
W3_SEQ_FILE = _W3CMD.SEQ_FILE if _W3CMD is not None else None   # 명령 순번 파일(~/.w3_cmd_seq, CLI w3_cmd.py 와 공유). DEMO·selftest 는 임시 폴더로 바꾼다


def bridge_extra_args(ckpt, live):
    """체크포인트별 브리지 추가 기동 인자. 걷기 실험 키에만 붙고(실험 플래그 + 정책 로드면 헤딩 고리 off), 그 밖의 키는 빈 목록(종전 인자 그대로)."""
    if ckpt not in WALK_CKPTS:
        return []
    return list(WALK_BRIDGE_ARGS[ckpt]) + (list(WALK_LIVE_ARGS) if live else [])


def bridge_live_argv(ckpt):
    return [sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"), "--live", "--checkpoint", ckpt] + bridge_extra_args(ckpt, live=True)


def bridge_selftest_argv(ckpt):
    return [sys.executable, os.path.join(SCRIPTS, "rl_bridge_node.py"), "--selftest", "--checkpoint", ckpt] + bridge_extra_args(ckpt, live=False)


#: X14(1007 시뮬 평가 — 합성 키 배포 코드 폐루프, reports/gen_w3/TRANSITION_W3KE.md §5.18·§5.18b)에서 잰 '버튼 한 번'의 이동량.
#: **시뮬 중앙값이고 실물에서는 확인하지 않았다.** 화면의 참고 표시일 뿐 명령 값이 아니다(명령 값의 원본은 W3_MOVES).
#: 버튼 값(vx·vy·wz·기한)이나 정책이 바뀌면 이 숫자는 낡는다 — 다시 재서 고치거나 지운다. 비활성 버튼에는 보이지 않는다.
#: 1008: X14 는 KE 감지기 갇힘 해소를 켠 구성(= 걷기 키 stand_kew3q_…)으로 쟀다. 이 표는 걷기 키 전부에 같은 값으로 보인다(키별 표 아님) —
#: 갇힘 해소가 꺼진 첫 합성 키·W3 단독 키의 표시는 종전 그대로 두었다(따로 둘지는 이 변경의 범위 밖).
WALK_SIM_REF = {"fwd1": "+4.8 cm", "fwd3": "+16 cm", "turn_l": "+17.6°", "turn_r": "−13.8°", "left1": "+4.4 cm"}
WALK_SIM_REF_NOTE = "X14 시뮬 중앙값 · 실물 미확인"


def walk_moves():
    """걷기 버튼 목록(원본 정의 순서) — [{name, label, vx, vy, wz, dur_s, enabled, why_disabled}].
    원본의 공개 목록(w3_deploy.w3_moves_public — 시뮬 평가 X14 의 enabled·why_disabled 포함)을 그대로 쓴다.
    그 함수가 없는 판(1007 초판)이면 W3_MOVES 에서 같은 모양으로 만든다(enabled 필드가 없으면 활성 — 원본과 같은 규칙)."""
    if _W3 is None:
        return []
    public = getattr(_W3, "w3_moves_public", None)
    rows = list(public()) if callable(public) else [dict(m, name=k) for k, m in _W3.W3_MOVES.items()]
    return [{"name": m["name"], "label": str(m["label"]), "vx": m["vx"], "vy": m["vy"], "wz": m["wz"], "dur_s": m["dur_s"],
             "enabled": bool(m.get("enabled", True)), "why_disabled": m.get("why_disabled")} for m in rows]


def walk_move_block(name):
    """걷기 버튼 이름 하나를 서버가 막는 이유 → (HTTP 코드, 문구) 또는 None. 화면을 우회한 API 호출도 여기서 걸린다.
    모르는 이름 400 · 시뮬 평가에서 비활성(원본 enabled False) 409 + 사유. 원본 w3_move_cmd 가 비활성을 거부하지 않는 옛 판이어도
    콘솔이 먼저 거른다(원본이 거부하는 판에서는 두 겹). 평가 전용 우회 인자(allow_disabled)는 콘솔이 넘기지 않는다."""
    entry = next((m for m in walk_moves() if m["name"] == name), None) if isinstance(name, str) else None
    if entry is None:
        return 400, f"모르는 걷기 버튼입니다: {name!r}"
    if not entry["enabled"]:
        return 409, f"{entry['label']}: 비활성 걷기 버튼 — {entry['why_disabled'] or '사유 미기록'}"
    return None


def walk_pad_rows():
    """방향 패드 행(이름). 원본 목록에 없는 이름은 빼고, 패드에 자리가 없는 원본 항목은 마지막 줄에 덧붙인다(조용히 버리지 않는다)."""
    moves = [m["name"] for m in walk_moves()]
    rows = [[n for n in row if n in moves] for row in WALK_PAD]
    extra = [n for n in moves if not any(n in row for row in WALK_PAD)]
    return [r for r in rows + [extra] if r]


def render_walk_pad():
    """방향 패드 HTML — 원본 목록으로 그린다. 비활성 버튼은 표시(class walkOff·툴팁에 사유)를 달고, 활성 버튼에는 시뮬 참고 이동량을 적는다.
    누를 수 있는지(disabled)·툴팁·작은 글씨는 화면이 /api/state 의 walk.buttons 로 계속 맞춘다(여기는 첫 그림)."""
    if _W3 is None:
        return f'<p class="note">걷기 모듈을 불러오지 못했습니다 — {html.escape(str(WALK_IMPORT_ERROR))}</p>'
    moves = {m["name"]: m for m in walk_moves()}

    def button(n):
        m = moves[n]
        if m["enabled"]:
            attrs, small = "", (f"시뮬 {WALK_SIM_REF[n]}" if n in WALK_SIM_REF else "")
        else:
            attrs = ' class="walkOff" title="{}"'.format(html.escape("비활성 — " + str(m["why_disabled"] or "사유 미기록"), quote=True))
            small = "비활성(시뮬 평가 미달)"
        return (f'<button data-walk="{html.escape(n, quote=True)}"{attrs} disabled>'
                f'<span>{html.escape(m["label"])}</span><small>{html.escape(small)}</small></button>')
    return "".join('<div class="walkRow">' + "".join(button(n) for n in row) + "</div>" for row in walk_pad_rows())


def render_page():
    """PAGE 의 정책 드롭다운을 CKPT_MENU 로 채운다 (1008: 고를 수 있는 정책만 — 지금은 합성 키 하나. 구판은 정적 목록 stand_v8~ 로 기본값이 legacy stand_v7 였음).
    1007: 걷기 실험 키는 표시를 붙여 뒤에 덧붙이고, 걷기 방향 패드는 W3_MOVES 의 표시 이름으로 채운다."""
    options = "".join((f'<option value="{k}">{k} — {html.escape(WALK_KEYS[k]["label"])}</option>' if k in WALK_KEYS else f"<option>{k}</option>") for k in CKPT_MENU)
    return PAGE.replace("<!--CKPT_OPTIONS-->", options).replace("<!--WALK_PAD-->", render_walk_pad())
STALE_SEC = 1.5              # 이보다 오래된 텔레메트리는 '끊김'으로 표시
#: can_up.sh 는 sudo가 필요하다. 젯슨 계정 암호(공개본에서 기본값 제거)를 환경변수로
#: 덮어쓸 수 있게 둔다 — 소스에 상수로 박아두지 않기 위함.
SUDO_PW = os.environ.get("JETSON_SUDO_PW", "")


def now():
    return time.time()


def pad_ints(value):
    """어댑터의 [4] 채널 카운트 → [4] int, 형식이 어긋나면 None (0으로 위장하지 않는다).

    0913 추가. **진단·화면 전용** — 하중/CoP/체중 지지율이 아니라 "이 채널이
    눌렸는가·살아있는가"를 사람이 보기 위한 값이다. 접촉 판정에는 쓰이지 않는다.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out = []
    for v in value:
        if not finite(v):
            return None
        out.append(int(v))
    return out


def pad_bools(value):
    """어댑터의 [4] bool 마스크 → [4] bool, 형식이 어긋나면 None (진단·화면 전용)."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(type(v) is not bool for v in value):
        return None
    return list(value)


# ---------------------------------------------------------------------------
# 모터 오류 기록(1007) — stage8 상태의 오류 바이트·warnings 를 타임라인에 남긴다(기록 전용)
# ---------------------------------------------------------------------------
#: 배경(reports/real_1007): 발목 롤 드라이버(모터 6·12)가 회복 동작 중 재시작했는데, 원인(전원 강하·커넥터 순간 단절·
#: 드라이버 보호)을 가를 모터 오류 바이트(stage8 상태 joints[j].feedback_error_raw)와 stage8 warnings 가 타임라인에 없었다.
#: 화면·판정·가드·명령에는 쓰지 않는다. 오류가 없고 warnings 가 비어 있는 보통 행에는 아무것도 더하지 않는다(행 크기 불변).
#: · 왜 메시지마다 보는가: stage8 상태는 100 Hz 이상, 타임라인 행은 20 Hz 다. stage8 은 명령을 받을 때마다(브리지 구동 중
#:   50 Hz) warnings 를 비우고, 오류 바이트는 그 모터의 마지막 프레임 값이라 한 프레임(20 ms)짜리면 행 사이에 사라진다.
#:   행 시점의 최신 메시지만 보면 이를 놓치므로 수신 콜백에서 메시지마다 받아 두고(feed) 행을 쓸 때 꺼낸다(drain).
#: · 행에 적는 것은 **직전 행 이후 받은 상태 메시지에서 본 것뿐**이다. 상태 메시지를 하나도 못 받은 구간(stage8 종료·수신
#:   공백)의 행에는 오류·warnings 를 적지 않는다 — 오류가 사라진 것도 warnings 가 비워진 것도 아니라 '모름'이다.
#:   마지막으로 받은 메시지의 값을 그 뒤 행마다 되풀이해 적지 않는다(첫 판은 그렇게 했다가 같은 값이 두 행에 실리거나 순서가
#:   뒤집히고, 끊긴 뒤에도 옛 값이 계속 실려서 뺐다).
#: · stage8_nomsg: '모름' 구간의 경계. 상태 메시지를 받다가 **한 건도 못 받은 첫 행에 1**, **다시 받은 첫 행에 0** 을 한 번씩 적는다.
#:   1 이 적힌 행부터 0 이 적힌 행 전까지의 '필드 없음'은 '오류 없음'이 아니다. 행의 fresh[2] 는 1.5 s 뒤에야 0 이 되므로
#:   끊긴 직후를 가리지 못한다(실물 1004 타임라인: 끊긴 첫 행 25 개 중 24 개가 fresh[2] = 1). 세션 안에서만 센다.
#: · 관절별 feedback_error_raw: 그 구간에 한 번이라도 0 이 아니었던 관절에만 넣는다(값 = 가장 최근의 0 아닌 값).
#:   서로 다른 값이 둘 이상이면 feedback_error_seen 에 전부(오래된 것 → 최근 순).
#: · stage8_warnings: 그 구간에 본 warnings 의 합집합(정렬). 내용이 바뀐 행에만 적고(비워진 것을 본 행에 [] 한 번),
#:   비어 있지 않은 채 그대로면 S8_WARN_KEYFRAME_SEC 마다 한 번 다시 적는다 — 선기동 대기처럼 warnings 가 몇 분씩 남는
#:   정상 구간에서 행마다 적으면 행이 커지고, 바뀐 행에만 적으면 정지 전후 일부 구간만 읽을 때 그 전에 켜진 것을 못 본다.
#: · session_start 의 "errlog": 1 = 이 세션은 위 필드를 기록한다(필드 부재 = 관측 없음. 이 표식이 없는 옛 세션은 '기록 안 함').
S8_WARN_KEYFRAME_SEC = 1.0   # warnings 가 그대로 남아 있을 때 다시 적는 주기 [s]
S8_WARN_MAX = 48             # 한 행에 싣는 warnings 최대 개수(넘치면 버리고 "+more" 한 건으로 표시)
S8_WARN_MAXLEN = 160         # warnings 한 건 최대 길이
S8_ERR_JOINTS_MAX = 16       # 받아 두는 관절 수 상한(형식이 어긋난 메시지 방어)
S8_ERR_NAME_MAXLEN = 64      # 받아 두는 관절 이름은 ASCII·이 길이 이내만 — 행의 키가 되므로 파일에 쓸 수 없는 이름(짝 없는 서러게이트 등)을 막는다
S8_ERR_CODES_MAX = 8         # 관절 하나에 받아 두는 서로 다른 값 수 상한


def s8_err_code(value):
    """모터 오류 바이트 → 0 이 아닌 정수면 그 값, 아니면 None(0·없음·bool·비정수·비유한)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
        return None
    return int(value) or None


class Stage8ErrLatch:
    """타임라인 행 사이에 지나간 stage8 모터 오류 바이트·warnings 를 붙잡아 둔다. **어떤 입력에도 예외를 내지 않는다**
    (ROS 수신 콜백 안에서 불린다 — 여기서 예외가 나가면 텔레메트리 수신 스레드가 죽는다. 내부 예외는 세어서 행에 남긴다)."""

    def __init__(self):
        self.lock = threading.Lock()
        self._clear()

    def _clear(self):
        self.n = 0          # 받은 상태 메시지 수(형식이 어긋난 것 포함) — 0 이면 그 구간은 통째로 '모름'(stage8_nomsg)
        self.nw = 0         # warnings 목록이 실린 상태 메시지 수 — 0 이면 warnings 는 '모름'(비워졌다고 적지 않는다)
        self.err = {}       # {관절: [0 아닌 오류 바이트 …]} — 마지막으로 본 순서(끝이 최신)
        self.warn = {}      # {warning: True} — 처음 본 순서
        self.more = 0       # 상한을 넘어 버린 warnings 건수
        self.fail = 0       # 처리 중 예외가 난 메시지 수

    def feed(self, status):
        """stage8 상태 메시지 1 건(dict)."""
        try:
            with self.lock:
                self.n += 1
                joints, warns = status.get("joints"), status.get("warnings")
                if isinstance(joints, dict):
                    for name, joint in joints.items():
                        code = s8_err_code(joint.get("feedback_error_raw")) if isinstance(joint, dict) else None
                        if code is None:
                            continue
                        seen = self.err.get(name)
                        if seen is None:
                            if (len(self.err) >= S8_ERR_JOINTS_MAX or not isinstance(name, str)
                                    or len(name) > S8_ERR_NAME_MAXLEN or not name.isascii()):
                                continue
                            seen = self.err[name] = []
                        if code in seen:
                            seen.remove(code)
                        elif len(seen) >= S8_ERR_CODES_MAX:
                            del seen[0]
                        seen.append(code)
                if isinstance(warns, list):
                    self.nw += 1
                    for w in warns[:4 * S8_WARN_MAX]:
                        if not isinstance(w, str):
                            continue
                        w = w[:S8_WARN_MAXLEN].encode("utf-8", "replace").decode("utf-8")   # 짝 없는 서러게이트 → '?'(파일 쓰기 보호)
                        if w in self.warn:
                            continue
                        if len(self.warn) >= S8_WARN_MAX:
                            self.more += 1
                            continue
                        self.warn[w] = True
        except Exception:
            with self.lock:
                self.fail += 1

    def drain(self):
        """직전 호출 이후 받아 둔 것을 꺼내고 비운다 → (받은 메시지 수, warnings 가 실린 메시지 수, err, warns, more, fail)."""
        with self.lock:
            out = (self.n, self.nw, self.err, list(self.warn), self.more, self.fail)
            self._clear()
        return out


ERRLATCH = Stage8ErrLatch()


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
        self.s8_warn_sent = ()      # 1007 모터 오류 기록: 이 세션 타임라인에 마지막으로 적은 stage8 warnings(정렬 튜플)
        self.s8_warn_ts = 0.0       #   그 시각 — 그대로 남아 있으면 S8_WARN_KEYFRAME_SEC 마다 다시 적는다
        self.s8_msg = None          #   직전 행 구간에 stage8 상태 메시지를 받았는가(None = 이 세션에서 아직 받은 적 없음) — stage8_nomsg 표식용

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
            self.s8_warn_sent, self.s8_warn_ts, self.s8_msg = (), 0.0, None
            self._write({"kind": "session_start", "reason": reason,
                         "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "errlog": 1})   # 1007: 이 세션은 모터 오류 바이트·stage8 warnings 를 기록한다(필드 부재 = 관측 없음)
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
            drained = ERRLATCH.drain()   # 1007: 세션이 닫혀 있어도 비운다(세션 밖에서 지나간 값이 새 세션 첫 행에 실리지 않게)
            if self.f is None:
                return
            imu = view.get("imu") or {}
            # '실제 활동' = 스택이 살아 있거나(stage8 상태 수신), 정책이 돌거나,
            # 로봇이 하중을 받고 있을 때. 매달린 채 방치된 상태는 활동이 아니다.
            if (view["fresh"].get("status") or view["fresh"].get("bridge")):
                self.last_active = now()
            row = {
                "kind": "s",

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
                # 0913: 채널별 보정 카운트·죽은 채널 후보 — "착지 뒤 살아있는 채널이
                # 실제로 눌렸는가"를 사후에 확인하기 위한 기록. 하중·CoP 아님.
                "foot_pads_counts": view["foot"].get("pads_counts"),
                "foot_dead_mask": view["foot"].get("dead_mask"),
                "stage8_running": view["stage8"].get("running"),
                "stage8_command_seq": view["stage8"].get("command_seq"),
                "stage8_last_command": view["stage8"].get("last_command"),
                "joints": {name: {key: joint.get(key) for key in (
                    "latest_actual_deg", "baseline_deg", "commanded_relative_deg",
                    "desired_relative_deg", "feedback_valid", "feedback_stale", "feedback_age_sec",
                    "feedback_torque_est", "feedback_velocity_rad_s", "feedback_temperature_raw")}   # 0919: 토크·속도 추정치도 타임라인에(모터별 흡수 가설 검증용)
                    for name, joint in view["stage8"].get("joints", {}).items() if isinstance(joint, dict)},
                "trial": {k: view.get("trial", {}).get(k) for k in
                          ("number", "mode", "checkpoint", "phase", "owner", "fault", "measurement", "stop")},
                "unverified_force_fields": True,
                "fresh": [int(view["fresh"]["imu"]), int(view["fresh"]["foot"]),
                          int(view["fresh"]["status"])],
            }
            try:   # 1007 모터 오류 기록(기록 전용) — 여기서 무슨 일이 나도 행은 그대로 쓴다
                self._s8_errlog(row, drained, now())
            except Exception:
                row["stage8_errlog_fail"] = -1
            self._write(row)
            self.n_sample += 1

    def _s8_errlog(self, row, drained, now_ts):
        """1007 모터 오류 기록: 행에 오류 바이트·stage8 warnings 를 덧붙인다(직전 행 이후 받은 상태 메시지에서 본 것만 — 보통 행은 그대로)."""
        n, nw, err, warns, more, fail = drained
        if n:        # '모름' 구간의 경계: 받다가 끊긴 첫 행에 1, 다시 받은 첫 행에 0 (세션 안에서 한 번씩 — 끊긴 채·받는 채 이어지는 행은 그대로)
            if self.s8_msg is False:
                row["stage8_nomsg"] = 0
            self.s8_msg = True
        elif self.s8_msg:
            self.s8_msg = False
            row["stage8_nomsg"] = 1
        for name, seen in err.items():
            entry = row["joints"].setdefault(name, {})
            entry["feedback_error_raw"] = seen[-1]
            if len(seen) > 1:
                entry["feedback_error_seen"] = seen
        if nw:   # warnings 가 실린 메시지를 하나도 못 받은 구간은 판단하지 않는다 — '비워짐'이 아니라 '모름'(다시 받으면 이어서 비교)
            cur = tuple(sorted(warns)) + (("+more",) if more else ())
            if cur != self.s8_warn_sent or (cur and abs(now_ts - self.s8_warn_ts) >= S8_WARN_KEYFRAME_SEC):
                self.s8_warn_sent, self.s8_warn_ts = cur, now_ts
                row["stage8_warnings"] = list(cur)
        if fail:
            row["stage8_errlog_fail"] = fail

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
        self.release_verdict = None
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
                    # 0922: 기계 판독 판정 우선. (구) 문자열 매칭은 하위호환으로 남긴다.
                    #   '읽기 실패' 는 오류로 보지 않는다 — 한 모터가 안 읽혀도 release_all 이 즉시 제로게인으로 처리한다.
                    if s.startswith("RELEASE_VERDICT="):   # noqa: SIM102
                        v = s.split("=", 1)[1].split()[0]
                        self.release_verdict = v
                        self.release_ok = (v == "OK")
                        self.release_error = None if v in ("OK", "NO_BUS") else s
                    elif "12모터 소프트 릴리즈+MIT 해제 완료" in s:
                        self.release_ok = True
                    elif "송신실패" in s and not getattr(self, "release_verdict", None):
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
                "release_ok": self.release_ok, "release_error": self.release_error,
                "release_verdict": getattr(self, "release_verdict", None)}   # 테스트는 __init__ 없이 Proc 를 만든다


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

    #: sensors_up.sh 가 자기 노드의 PID 를 적는 곳 — 이 파일들의 PID 만 '콘솔 계열이 띄운 센서'로 인정한다.
    SENSOR_PID_DIR = os.path.expanduser("~/run")

    @classmethod
    def _sensor_pidfile_pids(cls):
        found = set()
        try:
            for fn in os.listdir(cls.SENSOR_PID_DIR):
                if fn.endswith(".pid"):
                    with open(os.path.join(cls.SENSOR_PID_DIR, fn)) as f:
                        v = f.read().strip()
                    if v.isdigit():
                        found.add(int(v))
        except OSError:
            pass
        return found

    def start(self, key, argv, cwd=None):
        with self.lock:
            old = self.procs.get(key)
            if old is not None and old.alive:
                return False, f"{key} 이미 실행 중 (pid {old.p.pid})"
            note = ""
            if key in self.DUP_GUARD:
                pids = self._foreign_pids(self.DUP_GUARD[key])
                if pids:
                    owned = self._sensor_pidfile_pids() if key == "sensors" else set()
                    if key == "sensors" and set(pids) <= owned:
                        # 0919: 센서 노드가 살아 있어도 DDS 가 멈추면(사용자 세션 종료 등) 데이터가 안 온다.
                        # sensors_up.sh 는 자기 PID파일의 노드를 서명 검증 후 개별 종료하고 재기동하므로,
                        # PID파일에 적힌 것과 같은 PID 들이면 재기동을 허용한다(외부 PID 는 여전히 거부).
                        note = f"기존 센서 pid {pids}(PID파일 소유 확인) 정리 후 재기동 — "
                    else:
                        return False, (f"{key} 이미 실행 중 (pid {pids}, 콘솔 추적 밖) — 중복 기동 안 함. "
                                       "기존 소유 콘솔/터미널에서 상태를 확인하세요. 이 콘솔은 외부 PID를 종료하지 않습니다")
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            try:
                self.procs[key] = Proc(key, argv, cwd=cwd, env=env)
            except OSError as exc:
                return False, f"{key} 기동 실패: {exc}"
            return True, f"{note}{key} 기동 (pid {self.procs[key].p.pid})"

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
        #: 1007 W3 세션 브리지의 명령 응답(수신 순서대로). 상태 스냅샷이 아니라 사건이라 d/t 에 넣지 않고, 걷기 상태기가 console_snapshot 에서 비운다.
        self.acks = collections.deque(maxlen=64)

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
        node.create_subscription(String, BRIDGE_ACK_TOPIC, mk("ack"), 20)   # 1007 걷기 버튼 응답(W3 세션 브리지만 발행 — 그 밖 세션에서는 조용하다)
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
        if key == "status":
            ERRLATCH.feed(obj)   # 1007 모터 오류 기록(기록 전용·예외 없음): 행(20 Hz) 사이에 지나가는 값을 놓치지 않게 메시지마다 본다
        with self.lock:
            if key == "ack":      # 1007: 응답은 수신 시각과 함께 쌓기만 한다(판정은 걷기 상태기 — req_id 가 다르면 표시만 하고 무시)
                self.acks.append({"t": now(), "ack": obj})
                return
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

    def drain_acks(self):
        """쌓인 브리지 응답을 수신 순서대로 꺼낸다(1007 걷기 상태기 전용)."""
        with self.lock:
            items = list(self.acks)
            self.acks.clear()
        return items

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
        source_ts = foot.get("timestamp")
        if finite(source_ts) and not -0.5 <= n - source_ts < STALE_SEC:
            fresh["foot"] = False
        contacts = [foot.get("left_contact"), foot.get("right_contact")]
        contact_valid = all(type(v) is bool for v in contacts)
        # Old adapter data remains visible in diagnostics, but cannot certify
        # contact from an uncalibrated force scale.
        if not contact_valid:
            fresh["foot"] = False
        # 0913 추가(진단·화면 전용, 키 추가만): 채널별 카운트·눌림·죽은 채널.
        # 접촉 판정(contact)·신선도·게이트는 위에서 이미 끝났고 아래 값들은 그것을
        # 바꾸지 않는다. 어댑터가 구버전이면 전부 None — 0 으로 위장하지 않는다.
        pads_counts = [pad_ints(foot.get("left_pads_counts")),
                       pad_ints(foot.get("right_pads_counts"))]
        pads_on_mask = [pad_bools(foot.get("left_pads_on_mask")),
                        pad_bools(foot.get("right_pads_on_mask"))]
        dead_raw = foot.get("dead_mask")
        dead_mask = None
        if isinstance(dead_raw, (list, tuple)) and len(dead_raw) == 2:
            rows = [pad_bools(r) for r in dead_raw]
            dead_mask = rows if all(r is not None for r in rows) else None
        dead_note = foot.get("dead_note")
        if not isinstance(dead_note, str) or not dead_note.strip():
            dead_note = None
        out["foot"] = {
            # 0922 승윤님: 감압 패드는 **접지 여부 판단용**이다. 미보정 카운트를 힘(N)·체중 비율로 바꿔 보여주면
            #   그 값이 하중처럼 읽히고 판단에 섞인다(0922 1회차 분석에서 실제로 그렇게 오용했다).
            #   → left_n/right_n/total_n/pct 를 상태에서 뺀다. 접지 비트와 채널 카운트(고장 패드 진단용)만 남긴다.
            "contact": contacts if contact_valid and fresh["foot"] else None,
            "raw_sum": [foot.get("left_raw_sum"), foot.get("right_raw_sum")],
            "pads_on": [foot.get("left_pads_on"), foot.get("right_pads_on")],
            # 채널별 진단 — 값이 없으면 None (없음과 0 을 구분해서 보여주기 위함)
            "pads_counts": pads_counts if any(v is not None for v in pads_counts) else None,
            "pads_on_mask": pads_on_mask if any(v is not None for v in pads_on_mask) else None,
            "dead_mask": dead_mask,
            "dead_note": dead_note,
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
            # 1005 착지 자동 인수 게이트(브리지 stand touchdown_start 경로에서만 옴) — 정책 미구동 홀드 대기 여부 표시·명령 주체 판정
            "start_gate": br.get("start_gate") if isinstance(br.get("start_gate"), dict) else None,
            # 0913: 롤 보정기(브리지 맵 키 roll_comp) 표시 전용 — 맵 키가 없으면 브리지가 필드를
            # 아예 안 보내므로 None = 보정기 OFF. 판정·자동 진행에는 쓰지 않는다(읽기 전용 표시).
            "roll_comp": br.get("roll_comp_deg"),
            "roll_comp_gate": br.get("roll_comp_gate"),
            # 1007 걷기 조작: W3 세션 브리지만 싣는 필드를 그대로 옮긴다(없으면 None — 0·False 로 위장하지 않는다).
            #   w3 = 래치 클록(want·ge·ge_obs·u·f), w3_sup = 감독기(last_seq·stop_latched·latch_reason·freeze·burst·last_burst_end…),
            #   kew3 = 합성 러너(mode ke/w3·block…), w3_cmd = 리드 발·헤딩 고리, guard_mode 는 safe 가 아닐 때만 실린다,
            #   ts = 프레임의 브리지 timestamp(응답 timestamp 와 같은 시계 — '응답 뒤의 프레임인가' 판정용), cmd = 정책 명령(vx, vy, wz).
            "w3": br.get("w3") if isinstance(br.get("w3"), dict) else None,
            "w3_sup": br.get("w3_sup") if isinstance(br.get("w3_sup"), dict) else None,
            "kew3": br.get("kew3") if isinstance(br.get("kew3"), dict) else None,
            "w3_cmd": br.get("w3_cmd") if isinstance(br.get("w3_cmd"), dict) else None,
            "guard_mode": br.get("guard_mode") if isinstance(br.get("guard_mode"), str) else None,
            "ts": br.get("timestamp") if finite(br.get("timestamp")) else None,
            "cmd": br.get("cmd") if isinstance(br.get("cmd"), list) else None,
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
        cand = [int(v) for v in result.stdout.split() if v.isdigit() and int(v) not in owned | {os.getpid()}]
        # 0921: **읽기 전용 호출만** 외부 제어에서 뺀다. 그 외는 전부 남긴다(fail closed).
        #   사고: robotctl motors 가 Ctrl+Z/터미널 종료로 멈추면 자식 zero_anchor.py --preview(읽기 전용)가 T 상태로 남고,
        #   콘솔이 이를 외부 제어로 세어 강제 초기화(force_reset)가 영구히 막혔다(토크는 이미 없는 상태였다).
        #   --preview/--json 은 zero_anchor 가 자체적으로 쓰기를 차단하는 모드라 CAN 송신이 없다.
        #   ※ 정지(T) 상태는 제외하지 않는다 — 멈춘 stage8 은 모터가 마지막 명령(토크)을 물고 있을 수 있다.
        #   ※ /proc 을 못 읽으면 남긴다(확인 못 한 것을 안전하다고 보지 않는다).
        external = []
        for pid in cand:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cmd = fh.read().decode("utf-8", "replace").replace("\0", " ")
            except OSError:
                external.append(pid)          # 확인 불가 → 보수적으로 제어 프로세스로 간주
                continue
            if "zero_anchor.py" in cmd and ("--preview" in cmd or "--json" in cmd) and "--write" not in cmd:
                continue                      # 읽기 전용 미리보기 — 송신 없음
            external.append(pid)
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


# 1006: stage8 실효 지연 단축 옵션(73889d5f, 기본 꺼짐) — 환경변수 STAGE8_LATENCY_OPTS 에 허용 목록 키만(쉼표 구분).
#   fas = --feedback-after-send, soc = --send-on-command, w<ms> = --feedback-wait-ms <ms>(0.5–8). 비우면 종전 기동 인자 그대로.
#   권장 단계(승인 후): "fas" → "fas,soc". 100 Hz(--control-hz)는 여기서 켜지 않는다(별도 승인).
_STAGE8_LAT_FLAGS = {"fas": ["--feedback-after-send"], "soc": ["--send-on-command"]}


STAGE8_LATENCY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "stage8_latency_opts")


def stage8_latency_raw():
    """옵션 원문: 환경변수 STAGE8_LATENCY_OPTS 가 있으면 그것, 없으면 config/stage8_latency_opts 첫 줄(없거나 비면 '' = 종전 경로).
    서비스(systemd) 변경 없이 파일 한 줄로 켜고 끈다 — 되돌리기 = 파일 비우기/삭제 후 stage8 재기동."""
    env = os.environ.get("STAGE8_LATENCY_OPTS")
    if env is not None:
        return env
    try:
        with open(STAGE8_LATENCY_FILE) as f:
            line = f.readline().split("#", 1)[0].strip()
        return line
    except OSError:
        return ""


def stage8_latency_args(raw=None):
    raw = stage8_latency_raw() if raw is None else raw
    out = []
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        if tok in _STAGE8_LAT_FLAGS:
            out += _STAGE8_LAT_FLAGS[tok]
        elif tok.startswith("w"):
            ms = float(tok[1:])
            if not (0.5 <= ms <= 8.0):
                raise ValueError(f"STAGE8_LATENCY_OPTS 대기 범위 밖: {tok}")
            out += ["--feedback-wait-ms", f"{ms:g}"]
        else:
            raise ValueError(f"STAGE8_LATENCY_OPTS 허용되지 않은 키: {tok}")
    return out


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

# ── 0914 12축 개별 제어기(승윤님 요청): 콘솔은 관절 프레임 요청 파일만 쓰고, 자세 유지 스크립트(hang_stand.py)가
#    운용 게인(kp150) 도달 뒤에만 0.5°/s 로 램프해 적용한다. 정책 브리지가 살아 있으면 콘솔 상태기가 거부한다.
JOINT_TRIM_FILE = "/tmp/joint_trim_deg.json"
HANG_STATUS_FILE = "/tmp/hang_stand_status.json"
JOINT_TRIM_LIM = 100.0       # [deg] 관절당 상한 (hang_stand 도 같은 값; 실효 한계는 ROM 마진 클램프) — 0914 승윤님 요청
JOINT_TRIM_STEP_MAX = 2.0    # [deg] 한 번 클릭에 허용하는 최대 변화
JOINT_NAMES = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
               "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
               "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def _read_joint_trim():
    try:
        with open(JOINT_TRIM_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: float(raw.get(k, 0.0)) for k in JOINT_NAMES} if isinstance(raw, dict) else {k: 0.0 for k in JOINT_NAMES}
    except (OSError, ValueError, TypeError):
        return {k: 0.0 for k in JOINT_NAMES}


def _write_joint_trim(req):
    tmp = JOINT_TRIM_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: round(float(req.get(k, 0.0)), 3) for k in JOINT_NAMES}, f)
    os.replace(tmp, JOINT_TRIM_FILE)


def act_joint_trim(payload):
    """관절 하나의 개별 조정 요청. delta(클릭) 또는 value(절대). 한계 밖은 거부(자동 클램프 없음 — 트림과 같은 원칙)."""
    joint = payload.get("joint")
    _need(joint in JOINT_NAMES, "허용되지 않은 관절 이름입니다")
    req = _read_joint_trim()
    if "delta" in payload:
        d = float(payload["delta"])
        _need(math.isfinite(d) and abs(d) <= JOINT_TRIM_STEP_MAX + 1e-9,
              f"한 번에 ±{JOINT_TRIM_STEP_MAX:g}° 까지만 바꿀 수 있습니다")
        v = req[joint] + d
    else:
        v = float(payload.get("value", 0.0))
    _need(math.isfinite(v) and abs(v) <= JOINT_TRIM_LIM + 1e-9,
          f"{joint} 조정 {v:+.1f}° 는 한계 ±{JOINT_TRIM_LIM:g}° 밖입니다")
    req[joint] = v
    _write_joint_trim(req)
    return True, f"개별 조정 {joint.replace('_joint', '')} → {v:+.1f}° (관절 프레임, 0.5°/s 램프)"


# 0914 수평 기준 자세 프리셋 — 수동 튜닝으로 찾은 자세(22:25, fwd −2·lat +1.4·양발접촉). 관절 프레임 deg.
LEVEL_POSE_0914 = {"left_hip_f_joint": 15.0, "right_hip_f_joint": -15.0, "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0, "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0, "left_knee_joint": -30.0, "right_knee_joint": 30.0, "left_ankle_f_joint": 20.5, "right_ankle_f_joint": -20.5, "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0}


HOLD_KD_FILE = "/tmp/hold_kd"
HOLD_KD_MIN, HOLD_KD_MAX = 1.0, 30.0


def act_hold_kd(payload):
    """자세 홀드 감쇠 kd 라이브 조절(운용 게인 kp150 도달 후에만 hang_stand 가 반영). kp 는 정책 호환 위해 150 고정."""
    v = float(payload.get("value", 5.0))
    _need(math.isfinite(v) and HOLD_KD_MIN <= v <= HOLD_KD_MAX, f"kd 는 {HOLD_KD_MIN}~{HOLD_KD_MAX} 범위여야 합니다")
    tmp = HOLD_KD_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{v}\n")
    os.replace(tmp, HOLD_KD_FILE)
    return True, f"홀드 kd → {v:.0f} (운용 게인에서 반영)"


AIR_GAIT_FILE = "/tmp/air_gait.json"
AIR_GAIT_LIM = {"freq_hz": (0.2, 1.0), "hip_deg": (0.0, 40.0), "hip_ext_deg": (0.0, 25.0), "knee_deg": (0.0, 45.0), "ankle_deg": (0.0, 15.0)}
AIR_GAIT_DEFAULT = {"freq_hz": 0.5, "hip_deg": 30.0, "hip_ext_deg": 12.0, "knee_deg": 30.0, "ankle_deg": 8.0}


def _read_air_gait():
    try:
        with open(AIR_GAIT_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {k: float(raw.get(k, v)) for k, v in AIR_GAIT_DEFAULT.items()} | {"on": bool(raw.get("on", False))}
    except (OSError, ValueError, TypeError):
        pass
    return dict(AIR_GAIT_DEFAULT) | {"on": False}


def _write_air_gait(on, params):
    tmp = AIR_GAIT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"on": bool(on), **{k: round(float(params[k]), 3) for k in AIR_GAIT_DEFAULT}}, f)
    os.replace(tmp, AIR_GAIT_FILE)


def act_air_gait_on(payload):
    """에어 보행(사진용, 공중 제자리 걸음) 시작 요청. hang_stand 가 운용 게인·미접지에서만 2 s 페이드로 적용. 한계 밖은 거부."""
    cur = _read_air_gait()
    params = {k: cur[k] for k in AIR_GAIT_DEFAULT}
    for k, (lo, hi) in AIR_GAIT_LIM.items():
        if k in payload:
            v = float(payload[k])
            _need(math.isfinite(v) and lo <= v <= hi, f"{k}={v} 는 허용 범위 [{lo}, {hi}] 밖입니다")
            params[k] = v
    _write_air_gait(True, params)
    return True, (f"에어 보행 시작 요청 {params['freq_hz']:.2f} Hz · 힙 앞 {params['hip_deg']:.0f}°/뒤 {params['hip_ext_deg']:.0f}° · "
                  f"무릎 {params['knee_deg']:.0f}° · 발목 {params['ankle_deg']:.0f}° (공중·kp150 에서 2 s 페이드인)")


def act_air_gait_off(payload):
    cur = _read_air_gait()
    _write_air_gait(False, {k: cur[k] for k in AIR_GAIT_DEFAULT})
    return True, "에어 보행 정지(2 s 페이드아웃 → 현재 개별 조정 자세)"


LEVEL_POSE_FILE = os.path.join(CALIB, "level_pose.json")   # 0917: 프리셋을 파일에서 매번 읽음 → 갱신에 콘솔 재시작 불필요


def _level_pose():
    """~/rl_calib/level_pose.json 의 trims_joint_frame_deg(12관절 전부, 유한값)만 채택; 아니면 코드 상수."""
    try:
        with open(LEVEL_POSE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        t = raw.get("trims_joint_frame_deg") if isinstance(raw, dict) else None
        if isinstance(t, dict) and set(t) == set(JOINT_NAMES) and all(math.isfinite(float(v)) for v in t.values()):
            kd = raw.get("hold_kd")
            kd = float(kd) if isinstance(kd, (int, float)) and HOLD_KD_MIN <= float(kd) <= HOLD_KD_MAX else None
            return {k: float(t[k]) for k in JOINT_NAMES}, kd
    except (OSError, ValueError, TypeError):
        pass
    return dict(LEVEL_POSE_0914), None


def _level_pose_file_ref():
    """1005 정책 ON 선행조건용: level_pose.json 의 12관절 수평 기준 자세(관절 프레임 deg). 파일이 없거나 형식이 틀리면 None
    (코드 상수 0914 폴백을 기준으로 삼지 않는다 — 정책 기본자세와 다른 옛 자세라 '적용 확인' 이 거짓 통과/거부가 된다)."""
    try:
        with open(LEVEL_POSE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        t = raw.get("trims_joint_frame_deg") if isinstance(raw, dict) else None
        if isinstance(t, dict) and set(t) == set(JOINT_NAMES) and all(math.isfinite(float(v)) for v in t.values()):
            return {k: float(t[k]) for k in JOINT_NAMES}
    except (OSError, ValueError, TypeError):
        pass
    return None


# ── 0918 자동 수평(auto level): 접지 상태에서 IMU fwd 를 목표(기본 0°)로 맞추도록 힙앞뒤 트림을 천천히 돌린다.
#    근거: 같은 관절각이라도 착지마다 몸통 기울기가 −1~+9° 로 달랐고(0917), 고정 홀드엔 수평 피드백이 없다.
#    부호(실측 0914~17): 힙앞뒤 굴곡(왼 +, 오른 −)을 키우면 fwd(앞 기울기)가 +로 간다(≈1:1).
#    안전: kp150·hang_stand 살아 있을 때만, 발 접촉이 하나라도 있을 때만, 편차 |err|>AL_PAUSE_DEG 이면 일시정지(넘어짐/줄 지지 의심),
#          총 조정폭 ±AL_MAX_DEG, 속도 ≤AL_RATE_DEG_S. 파일 쓰기는 개별 조정과 같은 경로(hang_stand 램프·소프트 한계 클램프).
AL_STATE_FILE = "/tmp/auto_level.json"
AL_RATE_DEG_S = 0.5
AL_MAX_DEG = 10.0
AL_DEADBAND_DEG = 0.3
AL_GAIN = 0.6              # [deg 힙 / deg 오차] per tick 비례(속도 제한 안에서)
AL_PAUSE_DEG = 8.0
AL_QUIET_GYRO = 0.06      # [rad/s] 피치 각속도 이 이하일 때만 '정지' (0919: 밀기 중 적분 금지)
AL_QUIET_DFWD = 0.4       # [deg/틱] fwd 변화 이 이하
AL_QUIET_TICKS = 4        # 연속 정지 틱 수(AL_TICK_S 0.5 s → 2 s) 뒤에만 적분
AL_TICK_S = 0.5
AUTO_LEVEL = {"on": False, "target": 0.0, "offset": 0.0, "base": None, "status": "off", "err": None, "ts": 0.0}


def _al_write(req):
    _write_joint_trim(req)


def act_auto_level_on(payload):
    target = float(payload.get("target", 0.0))
    _need(math.isfinite(target) and -5.0 <= target <= 5.0, "목표 fwd 는 −5~+5° 안이어야 합니다")
    with STATE_LOCK:
        req = _read_joint_trim()
        AUTO_LEVEL.update(on=True, target=target, offset=0.0, status="armed", err=None,
                          base={"left_hip_f_joint": req["left_hip_f_joint"], "right_hip_f_joint": req["right_hip_f_joint"]})
    return True, f"자동 수평 ON — 목표 fwd {target:+.1f}° (힙앞뒤 트림 ≤{AL_RATE_DEG_S}°/s, 최대 ±{AL_MAX_DEG}°, 접촉 있을 때만)"


def act_auto_level_off(payload):
    with STATE_LOCK:
        AUTO_LEVEL.update(on=False, status="off")
    return True, f"자동 수평 OFF (현재 힙 오프셋 {AUTO_LEVEL['offset']:+.1f}° 는 유지)"


def auto_level_tick(view):
    """샘플러에서 AL_TICK_S 마다 호출. view = console_snapshot()."""
    if not AUTO_LEVEL["on"]:
        return
    hh = view.get("hang_hold") or {}
    fresh = view.get("fresh") or {}
    im = view.get("imu") or {}
    contact = (view.get("foot") or {}).get("contact") or [False, False]
    if not (hh.get("kp", 0) >= 150 and fresh.get("imu")):
        AUTO_LEVEL["status"] = "대기(운용 게인·IMU 필요)"; return
    if not any(bool(c) for c in contact):
        AUTO_LEVEL["status"] = "대기(발 접촉 없음)"; AUTO_LEVEL["quiet"] = 0; return
    fwd = float(im.get("fwd", 0.0)); err = AUTO_LEVEL["target"] - fwd
    AUTO_LEVEL["err"] = round(err, 2)
    if abs(err) > AL_PAUSE_DEG:
        AUTO_LEVEL["status"] = f"일시정지(편차 {err:+.1f}° > {AL_PAUSE_DEG})"; AUTO_LEVEL["quiet"] = 0; return
    if abs(err) <= AL_DEADBAND_DEG:
        AUTO_LEVEL["status"] = "수평"; return
    # 0919 결함 수정: 밀기·손 조정 중의 기울기를 영점 오차로 적분해 힙 오프셋이 쌓였음(실측 0 → +4.3°).
    # 양발 접촉 + 자이로 조용 + fwd 변화 작음 이 AL_QUIET_TICKS 연속 유지될 때만 적분한다.
    if not all(bool(c) for c in contact):
        AUTO_LEVEL["status"] = "대기(양발 접촉 필요)"; AUTO_LEVEL["quiet"] = 0; return
    gy = im.get("gyro") or [0.0, 0.0, 0.0]
    rate = abs(float(gy[1])) if len(gy) > 1 else 0.0
    last = AUTO_LEVEL.get("last_fwd"); AUTO_LEVEL["last_fwd"] = fwd
    moving = rate > AL_QUIET_GYRO or (last is not None and abs(fwd - last) > AL_QUIET_DFWD)
    AUTO_LEVEL["quiet"] = 0 if moving else AUTO_LEVEL.get("quiet", 0) + 1
    if AUTO_LEVEL["quiet"] < AL_QUIET_TICKS:
        AUTO_LEVEL["status"] = f"대기(외란/정착 중 {AUTO_LEVEL['quiet']}/{AL_QUIET_TICKS}, |ω| {rate:.2f})"; return
    step = max(-AL_RATE_DEG_S * AL_TICK_S, min(AL_RATE_DEG_S * AL_TICK_S, AL_GAIN * err))
    new_off = max(-AL_MAX_DEG, min(AL_MAX_DEG, AUTO_LEVEL["offset"] + step))
    if new_off == AUTO_LEVEL["offset"]:
        AUTO_LEVEL["status"] = f"한계(±{AL_MAX_DEG}°)"; return
    AUTO_LEVEL["offset"] = new_off
    req = _read_joint_trim(); b = AUTO_LEVEL["base"] or {"left_hip_f_joint": 0.0, "right_hip_f_joint": 0.0}
    req["left_hip_f_joint"] = b["left_hip_f_joint"] + new_off      # 앞으로 더 기울이려면(err>0) 굴곡 증가
    req["right_hip_f_joint"] = b["right_hip_f_joint"] - new_off
    _al_write(req)
    AUTO_LEVEL["status"] = f"조정 중 (오프셋 {new_off:+.1f}°)"
    AUTO_LEVEL["ts"] = time.time()


def act_joint_trim_preset(payload):
    """캡처해 둔 수평 기준 자세를 개별 조정 요청 파일에 그대로 쓴다(hang_stand 가 램프·소프트한계 클램프)."""
    pose, kd = _level_pose()
    _write_joint_trim(pose)
    if kd is not None:
        act_hold_kd({"value": kd})
    return True, (f"수평 기준 자세 적용 — 힙앞뒤 {pose['left_hip_f_joint']:+.1f}/{pose['right_hip_f_joint']:+.1f} · "
                  f"무릎 {pose['left_knee_joint']:+.1f}/{pose['right_knee_joint']:+.1f} · 발목앞뒤 {pose['left_ankle_f_joint']:+.1f}/{pose['right_ankle_f_joint']:+.1f}"
                  + (f" · kd {kd:.0f}" if kd is not None else ""))


def act_joint_trim_zero(payload):
    AUTO_LEVEL.update(on=False, status="off", offset=0.0)
    _write_joint_trim({k: 0.0 for k in JOINT_NAMES})
    return True, "개별 조정 12관절 전부 0 (0.5°/s 로 복귀)"


def read_hang_status(max_age=2.0):
    try:
        with open(HANG_STATUS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or time.time() - float(d.get("ts", 0)) > max_age:
            return None
        return d
    except (OSError, ValueError, TypeError):
        return None


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


def act_selftest_gate(payload):
    """정책 계약 검사(브리지 --selftest). 키가 없거나 메뉴 밖이면 기본값(= 메뉴의 정책)으로 돌린다 — 1008 전과 같은 규칙, 기본값만 바뀌었다.
    기본값조차 못 쓰는 상태(걷기 모듈 적재 실패)면 사유를 대고 거부한다."""
    ckpt = payload.get("checkpoint") if payload.get("checkpoint") in CKPT_MENU else CKPT_DEFAULT
    _need(ckpt_menu_reason(ckpt) is None, ckpt_menu_reason(ckpt) or "")
    return act_gate("selftest", bridge_selftest_argv(ckpt))


def act_bridge_live(payload):
    ckpt = str(payload.get("checkpoint", CKPT_DEFAULT))
    _need(ckpt_menu_reason(ckpt) is None, ckpt_menu_reason(ckpt) or "")
    # 1007: 걷기 실험 키면 실험 플래그 + 헤딩 고리 off 가 붙는다(bridge_extra_args). 그 밖의 키는 종전 인자 그대로.
    return RUN.start("bridge", bridge_live_argv(ckpt))


# ---------------------------------------------------------------------------
# 1007 걷기 조작 — 시간 제한 걷기 버튼 (W3 세션 전용, 기본 선택지에는 영향 없음)
# ---------------------------------------------------------------------------
#: 버튼 하나 = "잠깐 걷고 스스로 서는" 걷기 한 번. **끄기 기한은 브리지가 책임진다**(step 켜기의 dur_s — 콘솔·통신이 죽어도 선다).
#: 콘솔은 켜기 명령을 **한 번** 발행하고, 응답 토픽(수락/거부/기한 종료)과 디버그 토픽(서기 복귀)으로 상태를 보여 줄 뿐이다.
#: 응답이 없어도 같은 명령을 다시 보내지 않는다(브리지는 형식이 맞는 새 seq 를 거부해도 소비하고, 같은/오래된 seq 는 거부한다).
def walk_cmd_payload(kind, req_id, seq, move=None):
    """브리지 명령 JSON(고정 프로토콜). move 의 vx·vy·wz·dur_s 는 w3_deploy.w3_move_cmd(name) 가 돌려준 검증된 값 그대로다."""
    base = {"req_id": req_id, "seq": int(seq)}
    if kind == "move":
        return dict({"cmd": "step", "enable": True}, **base, heading_loop=False, **_W3.w3_move_cmd(move))
    if kind == "off":
        return dict({"cmd": "step", "enable": False}, **base)
    if kind in ("stop", "rearm"):
        return dict({"cmd": kind}, **base)
    raise ValueError(f"모르는 걷기 명령 {kind!r}")


def walk_req_id(tag, seq):
    """req_id = 콘솔·세션·동작·seq (1~64자). seq 가 단조라 콘솔을 다시 띄워도 겹치지 않는다(seq 자리는 잘리지 않는다)."""
    sid = re.sub(r"[^A-Za-z0-9_]", "", str(SESSION.name or "nosession").replace("session_", ""))[:26]
    return f"con-{sid}-{str(tag)[:10]}-{int(seq)}"


def walk_next_seq():
    """파일 잠금 단조 seq = max(마지막 + 1, 현재 ms) — 콘솔·브리지 재시작에도 증가한다(w3_cmd.next_seq 그대로)."""
    return _W3CMD.next_seq(W3_SEQ_FILE)


def walk_publish(cmd):
    """브리지 명령 토픽에 **1 회** 발행(DEMO 는 합성 브리지로). 반환 = 발행 여부."""
    if DEMO:
        return bool(DEMO_SOURCE.bridge_cmd(cmd))
    return bool(TEL.publish("bridge", cmd))


def dispatch_walk(action, payload, note=""):
    """걷기 명령 하나를 한 번만 발행한다(STATE_LOCK 안, 선행조건 통과 뒤). 응답은 여기서 기다리지 않는다 —
    수락·기한 종료·서기 복귀는 TrialState.walk 가 응답 토픽·디버그 토픽으로 따로 확인한다."""
    try:
        return _dispatch_walk(action, payload, note)
    except Exception as exc:          # 예상 못 한 내부 오류 — 요청 처리 스레드를 죽이지 않고 알린다(다시 보내지 않는다)
        return 500, {"ok": False, "message": f"걷기 명령 처리 오류({type(exc).__name__}: {exc}) — 전송 여부는 걷기 상태 줄·브리지 상태로 확인하세요. 재전송하지 않습니다"}


def _dispatch_walk(action, payload, note=""):
    kind = WALK_KIND[action]
    walk = TRIAL.walk
    name, label = None, LABELS[action]
    if kind == "move":
        name = payload.get("move")
        block = walk_move_block(name)                   # 모르는 이름·비활성 버튼(dispatch_action 이 먼저 거르지만 여기만 불러도 나가지 않게)
        if block is not None:
            return block[0], {"ok": False, "message": block[1]}
        try:
            _W3.w3_move_cmd(name)                       # 이름·비활성·학습 범위·단일 축·기한 범위 검사(원본 함수 — 평가 전용 우회 인자는 넘기지 않는다)
            label = str(_W3.W3_MOVES[name]["label"])
        except Exception as exc:
            return 400, {"ok": False, "message": f"걷기 버튼 거부: {exc}"}
    try:
        seq = walk_next_seq()
    except Exception as exc:
        if action not in WALK_SAFE:
            return 503, {"ok": False, "message": f"명령 순번(seq)을 발급하지 못해 보내지 않았습니다: {exc}"}
        seq = int(now() * 1000)     # 끄기·정지 래치는 브리지가 순번과 무관하게 적용한다(안전 방향) — 순번 파일 고장이 막지 않게
    req_id = walk_req_id(name or kind, seq)
    cmd = walk_cmd_payload(kind, req_id, seq, name)
    rec = walk.begin(TRIAL, kind, req_id, seq, cmd, name=name, label=label)
    try:
        sent = walk_publish(cmd)
    except Exception as exc:
        # 전송 여부를 알 수 없다 — 기록은 '발행함'으로 두고(응답 대기 → 없으면 '응답 없음'), 다시 보내지 않는다
        return 200, {"ok": False, "message": f"{label} 발행 중 오류({exc}) — 전송 여부 불명. 재전송하지 않습니다. 걷기 상태 줄 확인" + note}
    if not sent:
        walk.not_sent(TRIAL, rec, "발행 실패(ROS 미연결) — 명령이 나가지 않았습니다")
        return 200, {"ok": False, "message": f"{label} 발행 실패 (ROS 미연결) — 명령이 나가지 않았습니다" + note}
    return 200, {"ok": True, "message": f"{label} 발행(1 회) — 브리지 응답 대기 · 수락·서기 복귀는 걷기 상태 줄에서 확인" + note}


def dispatch_walk_safe(action, payload):
    """걷기 끄기·정지 래치 = 안전 방향. boot_id·revision·앞 요청(BUSY)·진행 중 단계·고장 래치·기록 실패가 막지 않는다.
    같은 request_id 의 재전송만 한 번으로 묶는다(끄기·정지 래치는 여러 번 가도 해가 없지만 순번·기록을 어지럽히지 않게)."""
    with STATE_LOCK:
        console_snapshot()
        request_id = payload.get("request_id")
        key = request_id if isinstance(request_id, str) and 8 <= len(request_id) <= 100 else None
        fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if key is not None and key in REQUESTS:
            old, response = REQUESTS[key]
            if old != fingerprint:
                return 409, {"ok": False, "message": "같은 요청 식별자를 다른 동작에 사용할 수 없습니다"}
            return response
        reason = TRIAL.reason(action)
        if reason:
            return 409, {"ok": False, "message": reason}
        try:
            SESSION.ensure(LABELS[action], manual=True)
            SESSION.event("action_request", action=action, trial=TRIAL.number, request_id=request_id)
        except OSError:
            pass                     # 기록 실패가 안전 방향 명령을 막지 않는다(기록 중단 자체는 TrialState.update 가 따로 래치한다)
        result = dispatch_walk(action, payload)
        if key is not None:
            REQUESTS[key] = (fingerprint, result)
            while len(REQUESTS) > 256:
                REQUESTS.popitem(last=False)
        console_snapshot()
        return result


def walk_view(st):
    """/api/state 의 walk 항목 — 상태기 스냅샷 + 버튼 표(이름·표시 이름·값은 W3_MOVES 에서, 사용 가능 여부는 서버 선행조건에서).
    걷기 패널은 실험 기능이다: 여기서 난 예외는 가둬서(걷기 버튼 잠금 + 오류 표시) 기립 시험의 상태·기록 주기로 번지지 않게 한다."""
    try:
        return _walk_view(st)
    except Exception as exc:
        first = TRIAL.walk.broken is None
        TRIAL.walk.fail(TRIAL, exc)
        if first:                                   # 로그에는 한 번만(상태 주기마다 되풀이하지 않는다)
            print(f"[걷기 패널] 내부 오류 — 걷기 버튼 잠금: {TRIAL.walk.broken}", flush=True)
        allowed = st["trial"]["allowed"]
        return {"available": False, "reason": f"걷기 패널 내부 오류 — 걷기 버튼을 잠갔습니다({TRIAL.walk.broken}). 콘솔 로그 확인",
                "internal_error": TRIAL.walk.broken, "module_error": WALK_IMPORT_ERROR, "session": False, "profile": None,
                "current": None, "command": None, "bridge": {}, "buttons": [], "pad": [], "recent_acks": [], "guard_mode": None,
                "mode": None, "composite": False,
                "controls": {k: {"enabled": bool(allowed[k]["enabled"]), "reason": allowed[k]["reason"], "label": allowed[k]["label"]}
                             for k in ("w3_off", "w3_stop", "w3_rearm")}}


def _walk_view(st):
    allowed = st["trial"]["allowed"]
    move = allowed["w3_move"]
    moves = {m["name"]: m for m in walk_moves()}
    out = TRIAL.walk.snapshot(TRIAL)
    out.update(
        available=bool(move["enabled"]), reason=move["reason"], module_error=WALK_IMPORT_ERROR,
        pad=walk_pad_rows(),
        # 버튼마다: allowed = 원본 표의 enabled(시뮬 평가에서 비활성이면 False + why_disabled), enabled = 지금 누를 수 있는가(= 패널 사용 가능 ∧ allowed),
        # sim_ref = 한 번 눌렀을 때의 이동량 참고값(시뮬 중앙값 · 실물 미확인 — 활성 버튼에만).
        buttons=[dict(moves[n], allowed=moves[n]["enabled"], enabled=bool(move["enabled"]) and moves[n]["enabled"],
                      sim_ref=WALK_SIM_REF.get(n) if moves[n]["enabled"] else None)
                 for row in walk_pad_rows() for n in row],
        controls={k: {"enabled": bool(allowed[k]["enabled"]), "reason": allowed[k]["reason"], "label": allowed[k]["label"]}
                  for k in ("w3_off", "w3_stop", "w3_rearm")},
        ack_timeout_s=WALK_ACK_TIMEOUT_S, sim_ref_note=WALK_SIM_REF_NOTE, internal_error=TRIAL.walk.broken,
        sign_note="시뮬 규약: +vx 앞 · +vy 왼쪽 · +wz 왼쪽(반시계) 회전")
    return out


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
    # 1007: 걷기 실험 키면 실험 플래그가 붙는다(헤딩 고리 off 는 빼고 — WALK_LIVE_ARGS 주석). 그 밖의 키는 종전 인자 그대로.
    "selftest":   (lambda p: act_selftest_gate(p), "브리지 셀프테스트"),
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
        "--can-channel", "can1", "--wait-takeover"] + stage8_latency_args()), "stage8 선기동(인수대기)"),
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
    # 0914 12축 개별 제어기 — 상태기(console_state.MANUAL)가 '운용 게인 도달·브리지 없음' 을 강제한다.
    "joint_trim": (act_joint_trim, "개별 관절 조정"),
    "joint_trim_zero": (act_joint_trim_zero, "개별 조정 전부 0"),
    "joint_trim_preset": (act_joint_trim_preset, "수평 기준 자세 적용"),
    "hold_kd": (act_hold_kd, "홀드 감쇠 kd"),
    "auto_level_on": (act_auto_level_on, "자동 수평 ON"),
    "auto_level_off": (act_auto_level_off, "자동 수평 OFF"),
    "air_gait_on": (act_air_gait_on, "에어 보행 시작"),
    "air_gait_off": (act_air_gait_off, "에어 보행 정지"),
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
    # 1005: touchdown_start 는 '정책 ON' 순서기만 넘긴다(착지 확인 전 정책 미구동). 단계별 절차의 payload 는 종전과 같다.
    "stand":      (lambda p: act_bridge_cmd("stand", {"hang_arm": bool(p.get("hang_arm")),
                                                      "touchdown_start": bool(p.get("touchdown_start"))}), "stand 발사"),
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
    # 1003 교대 진단: 밀기 직전·직후에 누르는 기록 전용 표시 — 분석기(reports/real_1003/ab_analyze.py)가 이 시각 주변에서 신호 기준 밀기 시작을 찾는다.
    "op_push":          (lambda p: (True, "기록: 밀기"), "⇢ 밀기"),
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
CONFIRM_ACTIONS = {"hang_stand", "gain_ramp", "ground", "handoff", "arm", "stand", "policy_on",
                   "release", "support_confirm", "op_slack_done", "new_trial", "zero_multiple", "zero_recapture", "air_gait_on",
                   "w3_move", "w3_rearm"}   # 1007: 걷기 버튼(로봇이 실제로 걷는다)·재무장(정지 래치 해제). 걷기 끄기·정지 래치는 확인 없이 즉시


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
        st["hang_hold"] = None if DEMO else read_hang_status()
        st["joint_trim_req"] = None if DEMO else _read_joint_trim()
        st["air_gait_req"] = None if DEMO else _read_air_gait()
        st["auto_level"] = dict(AUTO_LEVEL)
        st["level_pose_ref"] = None if DEMO else _level_pose_file_ref()   # 1005 정책 ON 선행조건(수평 자세 적용 확인)용
        # 1007 걷기 조작: 걷기 모듈 적재 오류(선행조건용)와 그동안 받은 브리지 응답을 상태기에 넘긴다. 응답 원문은 타임라인에만 남긴다.
        st["walk_error"] = WALK_IMPORT_ERROR
        st["walk_ready"] = WALK_READY          # 브리지 켜기 문턱(양발 접촉·조용한 서기 연속 시간) — '서기 정착 대기' 표시용
        acks = DEMO_SOURCE.drain_acks() if DEMO else TEL.drain_acks()
        for item in acks:
            SESSION.event("bridge_ack", recv=item.get("t"), ack=item.get("ack"))
        TRIAL.walk.feed(acks)
        TRIAL.update(st)
        st["trial"] = TRIAL.snapshot()
        if BUSY:
            for key, item in st["trial"]["allowed"].items():
                if key not in EVENTS | {"estop", "support_confirm"} | WALK_SAFE:
                    item.update(enabled=False, reason="서버에서 앞 요청을 처리 중입니다")
        st["walk"] = walk_view(st)
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
    if action in WALK_SAFE:
        # 1007 걷기 끄기·정지 래치: 안전 방향 — 아래의 boot_id·revision·BUSY·기록 게이트를 거치지 않는다(걷기 세션·브리지 실행만 확인).
        return dispatch_walk_safe(action, payload)

    immediate = action in EVENTS | {"support_confirm", "configure", "new_trial", "force_reset", "session_start", "session_end",
                                    "joint_trim", "joint_trim_zero", "joint_trim_preset", "hold_kd", "air_gait_on", "air_gait_off", "auto_level_on", "auto_level_off"}
    walk_action = action in ("w3_move", "w3_rearm")     # 1007: 발행 전에 pgrep(최대 3 s)을 기다리지 않는다 — 프로세스 점검은 1 Hz 감시가 따로 한다
    if not DEMO and not immediate and not walk_action:
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
        if action == "w3_move" and WALK_IMPORT_ERROR is None:
            # 1007: 버튼 자체가 막힌 경우(모르는 이름·시뮬 평가에서 비활성)는 지금 상태와 무관한 이유라 먼저 알린다 — 화면을 우회한 호출도 여기서 거부
            block = walk_move_block(payload.get("move"))
            if block is not None:
                return block[0], {"ok": False, "message": block[1]}
        reason = TRIAL.reason(action)
        if reason or (BUSY and action not in EVENTS | {"support_confirm"}):
            return 409, {"ok": False, "message": reason or "앞 요청 처리 중"}
        if action in CONFIRM_ACTIONS and payload.get("confirmed") is not True:
            return 400, {"ok": False, "message": "운영자의 명시 확인이 필요합니다"}
        if action == "configure":
            if payload.get("mode") not in ("ground", "air"):
                return 400, {"ok": False, "message": "허용된 시험 방식·정책을 선택하세요"}
            if ckpt_menu_reason(payload.get("checkpoint")) is not None:
                return 400, {"ok": False, "message": ckpt_menu_reason(payload.get("checkpoint"))}
            if not payload["checkpoint"].startswith("stand"):
                return 400, {"ok": False, "message": "이 콘솔은 기립 시험용입니다. 보행 정책은 별도 절차가 필요합니다"}
            if payload["checkpoint"] in WALK_KEYS and payload["mode"] == "air":
                # 1007: W3 키는 지면 시작 학습뿐이라 브리지가 hang_arm(공중 정책 시작)을 거부한다 — 시작이 조용히 막히기 전에 여기서 알린다
                return 400, {"ok": False, "message": "걷기 실험 키는 '접지 후 정책 인계' 방식만 됩니다 — 브리지가 공중 정책 시작(hang_arm)을 거부합니다"}
        if walk_action:
            # 1007: 화면이 본 서버 시각(seen)이 오래됐으면 보내지 않는다 — 연결이 끊겼다 뒤늦게 도착한 요청으로 로봇이 걷지 않게.
            seen = payload.get("seen")
            age = now() - seen if finite(seen) else None
            if age is None or not -1.0 <= age <= WALK_REQ_MAX_AGE_S:
                return 409, {"ok": False, "message": "요청이 오래됐습니다" + (f"({age:.1f} s)" if age is not None else "(화면 시각 없음)")
                             + " — 걷기 명령을 보내지 않았습니다. 현재 상태를 확인하고 다시 누르세요"}
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
        if action == "policy_on":
            result = start_policy_on(record_warning)
            REQUESTS[request_id] = (fingerprint, result)
            console_snapshot()
            return result
        if walk_action:
            # 1007 걷기 버튼·재무장: 한 번만 발행하고 돌아온다. 같은 request_id 재전송은 위 REQUESTS 가 같은 응답으로 묶는다(재발행 없음).
            result = dispatch_walk(action, payload, record_warning)
            REQUESTS[request_id] = (fingerprint, result)
            console_snapshot()
            return result
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
            elif action in ("new_trial", "force_reset"):
                previous = TRIAL
                if action == "force_reset":
                    # 운영자 선언: 모터 전원 차단/무토크 확인 → 토크 해제 단계 생략하고 단계 초기화 (제어 프로세스 없음은 allowed 에서 검사)
                    SESSION.event("force_reset", note=str((payload or {}).get("note", "")),
                                  skipped_release="release" not in previous.completed, fault=previous.fault)
                    previous.completed.add("release")
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
                    TEL.acks.clear()      # 1007: 지난 시험의 걷기 응답을 새 시험으로 넘기지 않는다(걷기 상태기는 TrialState 와 함께 새로 생긴다)
                # Preserve old process log objects; don't reuse their faults/readiness.
                if not DEMO:
                    with RUN.lock:
                        RUN.procs = {k: p for k, p in RUN.procs.items() if p.alive}
                else:
                    DEMO_SOURCE = type(DEMO_SOURCE)()
                TRIAL.emit("new_trial" if action == "new_trial" else "force_reset")
            elif action in ("joint_trim", "joint_trim_zero", "joint_trim_preset", "hold_kd", "air_gait_on", "air_gait_off", "auto_level_on", "auto_level_off"):
                try:
                    ok, msg = ACTIONS[action][0](payload)
                except Exception as exc:
                    ok, msg = False, f"{LABELS[action]} 실패: {exc}"
                SESSION.event("joint_trim", action=action, ok=bool(ok), message=msg,
                              joint=payload.get("joint"), delta=payload.get("delta"), value=payload.get("value"))
                TRIAL.revision += 1
                result = (200 if ok else 400, {"ok": bool(ok), "message": msg + record_warning})
                REQUESTS[request_id] = (fingerprint, result)
                console_snapshot()
                return result
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


# ---------------------------------------------------------------------------
# 1005 정책 ON — 단순화 기동 순서기 (기존 단계별 절차는 그대로 남는다)
# ---------------------------------------------------------------------------
#: 운영자 조작: 영점 → 수평 자세(공중, kp150) → **정책 ON** → 지그 해제. 정책 ON 한 번이 서버 안에서
#:   접지 마커(공중 인계 허가) → 인계 → 기준점 → ARM → 정책 로드 → stand(touchdown_start)
#: 를 **기존 액션 함수·기존 게이트(TrialState.reason)·기존 응답 확인(TrialState.update)** 그대로 순서대로 실행한다.
#: 단계 하나라도 거부·실패·시간 초과면 즉시 멈추고 고장 래치(회수 절차로) — 자동 재시도 없음.
#: 정책은 브리지 착지 게이트가 양발 착지를 확인한 뒤에만 구동된다(그 전엔 stage8 이 하던 수평 홀드 명령 유지).
POLICY_ON_WORKER = None
POLICY_ON_STEP_TIMEOUT_S = 90.0      # 단계 응답 확인 상한(각 단계 고유 시간 초과는 TrialState.update 가 먼저 판정)
POLICY_ON_SUB_WAIT_S = 10.0          # 정책 로드 뒤 브리지 명령 토픽 구독 연결 대기 상한(DDS 발견) — 넘으면 그대로 진행(stand 8 s 확인이 판정)
POLICY_ON_POLL_S = 0.05


def _bridge_cmd_subscribed():
    """브리지 명령 토픽에 구독자가 붙었는가 (None = 알 수 없음). 첫 stand 발행이 DDS 발견 전에 사라지는 것을 줄인다."""
    pub = getattr(TEL, "_pub_bridge", None)
    if pub is None:
        return None
    try:
        return pub.get_subscription_count() > 0
    except Exception:
        return None


def start_policy_on(record_warning=""):
    """STATE_LOCK 안에서 호출(dispatch). 선행조건은 이미 TRIAL.reason('policy_on') 으로 확인됨."""
    global POLICY_ON_WORKER
    TRIAL.macro = {"action": "policy_on", "step": None, "started": now()}
    TRIAL.policy_on = True
    TRIAL.revision += 1
    TRIAL.emit("policy_on_start", checkpoint=TRIAL.checkpoint)
    SESSION.event("policy_on_start", checkpoint=TRIAL.checkpoint, trial=TRIAL.number)
    if AUTO_LEVEL.get("on"):
        # 자동 수평은 접지 상태 힙 트림 적분기 — 인계 뒤엔 자세 유지 스크립트가 없어 의미가 없다(robotctl handoff 와 같이 끈다)
        AUTO_LEVEL.update(on=False, status="off(정책 ON)")
    POLICY_ON_WORKER = threading.Thread(target=_policy_on_worker, args=(TRIAL,), name="policy_on", daemon=True)
    POLICY_ON_WORKER.start()
    return (200, {"ok": True, "message": "정책 ON 시작 — 인계·기준점·ARM·정책 로드·착지 대기를 기존 게이트로 순서 진행"
                  " (정책은 양발 착지 확인 뒤에만 구동)" + record_warning})


def _policy_on_abort(trial, message):
    with STATE_LOCK:
        if TRIAL is not trial:
            return
        step = (trial.macro or {}).get("step")
        trial.macro = None
        trial.revision += 1
        trial.emit("policy_on_aborted", step=step, message=message)
        SESSION.event("policy_on_aborted", step=step, message=message)
        trial.latch(f"정책 ON 중단({LABELS.get(step, step)}): {message}")


def _policy_on_worker(trial):
    for step in POLICY_ON_STEPS:
        if not DEMO:
            poll_processes()
        with STATE_LOCK:
            console_snapshot()
            if TRIAL is not trial or trial.macro is None:
                return
            if trial.fault:
                return _policy_on_abort(trial, f"고장·정지 래치 — {trial.fault}")
            if step in trial.completed:
                continue
        if step == "stand" and not DEMO:
            t_sub = time.monotonic()
            while time.monotonic() - t_sub < POLICY_ON_SUB_WAIT_S and _bridge_cmd_subscribed() is False:
                time.sleep(0.1)
            time.sleep(0.5)
        with STATE_LOCK:
            console_snapshot()
            if TRIAL is not trial or trial.macro is None:
                return
            if trial.fault:
                return _policy_on_abort(trial, f"고장·정지 래치 — {trial.fault}")
            why = trial.reason(step, _macro=True)
            if why:
                return _policy_on_abort(trial, why)
            trial.macro["step"] = step
            trial.begin(step, now())
            pending = trial.pending
            effective = {"checkpoint": trial.checkpoint, "hang_arm": False, "touchdown_start": True}
            SESSION.event("policy_on_step", step=step, trial=trial.number)
        try:
            ok, msg = DEMO_SOURCE.act(step, effective) if DEMO else ACTIONS[step][0](effective)
        except Exception as exc:
            ok, msg = False, f"{LABELS[step]} 실행 실패: {exc}"
        with STATE_LOCK:
            SESSION.event("action_dispatched", action=step, ok=bool(ok), message=msg, via="policy_on")
            if trial.pending is not pending:
                return _policy_on_abort(trial, "진행이 취소되었습니다 — 잔존 프로세스·실제 상태를 확인하세요")
            if not ok:
                trial.finish(False, msg)
                return _policy_on_abort(trial, msg)
            pending["launched"] = True
        t0 = time.monotonic()
        while True:
            time.sleep(POLICY_ON_POLL_S)
            with STATE_LOCK:
                console_snapshot()
                if TRIAL is not trial:
                    return
                if trial.pending is not pending:
                    break
                if time.monotonic() - t0 > POLICY_ON_STEP_TIMEOUT_S:
                    trial.finish(False, "정책 ON 단계 응답 확인 시간 초과")
                    break
        with STATE_LOCK:
            if step not in trial.completed:
                lr = trial.last_result or {}
                return _policy_on_abort(trial, lr.get("message") or "단계 확인 실패")
    with STATE_LOCK:
        if TRIAL is not trial or trial.macro is None:
            return
        trial.macro = None
        trial.completed.add("policy_on")
        trial.revision += 1
        trial.emit("policy_on_armed", message="착지 대기 — 양발 착지 확인 뒤 정책 자동 시작")
        SESSION.event("policy_on_armed", trial=trial.number)
        console_snapshot()


class WalkBench:
    """걷기 패널 오프라인 시험대(콘솔 --selftest 와 tests/ 가 같이 쓴다). 콘솔을 DEMO 합성 브리지에 붙이고 가짜 시계로 돌린다.

    합성 브리지(console_demo.DemoW3Bridge)의 판정은 실제 감독기·래치 클록(w3_deploy)이 한다. 요청은 전부 dispatch_action 을 지난다.
    ROS·서브프로세스·실제 로그·실제 순번 파일(~/.w3_cmd_seq)을 쓰지 않는다. **계약·순서 검증이지 폐루프 성능 증명이 아니다.**
    """
    TICK = 0.05                 # 콘솔 샘플러와 같은 20 Hz

    def __init__(self, directory, checkpoint):
        from unittest.mock import patch
        try:
            from .console_demo import Demo
        except ImportError:
            from console_demo import Demo
        self.t = 1_800_000_000.0
        self.checkpoint = checkpoint
        self.sent = []          # 콘솔이 브리지 명령 토픽에 발행한 것(순서대로) — '한 번만 발행' 확인용
        self.demo = Demo(clock=lambda: self.t)
        real = self.demo.bridge_cmd
        self.demo.bridge_cmd = lambda cmd: (self.sent.append(json.loads(json.dumps(cmd))), real(cmd))[1]
        # 1008: 운용 메뉴(CKPT_MENU)는 합성 키 하나뿐이다. 시험대는 자기 키를 메뉴에 올려, 콘솔이 기동 인자를 아는 키 전부의 동작을 계속 고정한다.
        self._patch = patch.dict(globals(), DEMO=True, DEMO_SOURCE=self.demo, TRIAL=TrialState(CKPT_DEFAULT, clock=lambda: self.t),
                                 CKPT_MENU=[checkpoint] if checkpoint in CKPT_ALL else list(CKPT_MENU),
                                 SESSION=SessionLog(), SESSION_DIR=os.path.join(directory, f"sessions-{uuid.uuid4().hex[:8]}"),
                                 TEL=Telemetry(), RUN=Runner(), BUSY=False, REQUESTS=collections.OrderedDict(),
                                 PROCESS_HEALTH={"ts": self.t, "external": [], "error": None}, now=lambda: self.t,
                                 W3_SEQ_FILE=os.path.join(directory, "w3_seq"))

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        try:
            SESSION.close("walk bench")
        finally:
            self._patch.stop()

    def state(self):
        return console_snapshot()

    def walk(self):
        return console_snapshot()["walk"]

    def payload(self, action, **extra):
        st = console_snapshot()
        return {"action": action, "boot_id": st["trial"]["boot_id"], "revision": st["trial"]["revision"],
                "request_id": uuid.uuid4().hex, "confirmed": True, "seen": st["server_time"], **extra}

    def req(self, action, **extra):
        return dispatch_action(self.payload(action, **extra))

    def tick(self, seconds=None):
        for _ in range(max(1, int(round((self.TICK if seconds is None else seconds) / self.TICK)))):
            self.t += self.TICK
            console_snapshot()

    def until(self, pred, limit_s, what):
        t0 = self.t
        while not pred(console_snapshot()["walk"]):
            assert self.t - t0 <= limit_s, f"{what}: {limit_s:g} s 안에 일어나지 않음 — {console_snapshot()['walk']['current']}"
            self.tick()
        return round(self.t - t0, 2)

    def prepare(self, settle_s=1.5):
        """설정 → 기존 단계별 절차를 정책 제어 시작까지 → 합성 감독기의 켜기 선행조건(양발 접촉·정착 연속 1.0 s)이 차도록 대기."""
        code, res = self.req("configure", mode="ground", checkpoint=self.checkpoint)
        assert code == 200 and res["ok"], (code, res)
        for action in TRIAL.flow():
            if action == "ground":
                assert self.req("op_touchdown")[0] == 200
            code, res = self.req(action)
            self.tick()
            assert code == 200 and res["ok"] and action in TRIAL.completed, (action, code, res, TRIAL.pending, TRIAL.fault)
        self.tick(settle_s)

    def timeline(self):
        with open(os.path.join(SESSION.dir, "timeline.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def walk_events(self):
        return [r["event"] for r in self.timeline() if r.get("kind") == "trial_event" and str(r.get("event", "")).startswith("walk")]


def selftest_walk(directory):
    """1007 걷기 패널 오프라인 시험 → 결과 요약 줄 목록(실패는 AssertionError). 가짜 응답·디버그 입력으로 상태기의 전 경로를 지난다."""
    from unittest.mock import patch
    assert WALK_IMPORT_ERROR is None, f"걷기 모듈 적재 실패: {WALK_IMPORT_ERROR}"
    W, out = _W3, []
    stage = lambda bn: (bn.walk()["current"] or {}).get("stage")     # noqa: E731
    VAL = ("label", "vx", "vy", "wz", "dur_s")

    # ① 버튼 표 = 원본(W3_MOVES·공개 목록) · 비활성 버튼 = 원본의 enabled False(사유 포함) · 명령 JSON = 고정 프로토콜 · 실제 감독기 통과 · seq 단조
    moves = walk_moves()
    names = [n for row in walk_pad_rows() for n in row]
    assert [m["name"] for m in moves] == list(W.W3_MOVES) and sorted(names) == sorted(W.W3_MOVES) and len(names) == len(set(names)), (names, list(W.W3_MOVES))
    assert sorted(n for row in WALK_PAD for n in row) == sorted(W.W3_MOVES), "방향 패드 배치(WALK_PAD)와 원본 버튼 표의 이름이 어긋남"
    assert walk_pad_rows() == [list(r) for r in WALK_PAD] and set(WALK_SIM_REF) <= set(W.W3_MOVES)
    for m in moves:
        src_m = W.W3_MOVES[m["name"]]
        assert {k: m[k] for k in VAL} == {k: src_m[k] for k in VAL}, m
        assert m["enabled"] is bool(src_m.get("enabled", True)) and m["why_disabled"] == src_m.get("why_disabled"), m
        assert m["enabled"] or m["why_disabled"], f"비활성 버튼 {m['name']} 에 사유가 없음"
    if callable(getattr(W, "w3_moves_public", None)):       # 원본에 공개 목록이 있으면 그것과 같은 내용이어야 한다(없는 판은 위 대체 경로)
        assert moves == [{k: p.get(k) for k in ("name",) + VAL + ("enabled", "why_disabled")} for p in W.w3_moves_public()]
    on, off = [m["name"] for m in moves if m["enabled"]], [m["name"] for m in moves if not m["enabled"]]
    by_dur = sorted((m for m in moves if m["enabled"]), key=lambda m: m["dur_s"])
    mv, mv_long = by_dur[0]["name"], by_dur[-1]["name"]         # 가장 짧은·가장 긴 활성 버튼(시나리오용 — 이름을 박아 두지 않는다)
    mv_b, mv_c = [n for n in on if n not in (mv, mv_long)][:2]
    assert by_dur[-1]["dur_s"] > WALK_ACK_TIMEOUT_S + 0.2, "응답 대기보다 긴 활성 버튼이 없어 '걷는 중 응답 없음' 시나리오를 만들 수 없음"
    page = render_page()
    assert "<!--WALK_PAD-->" not in page and page.count('data-walk="') == len(W.W3_MOVES)
    for m in moves:
        if m["enabled"]:
            small = f"시뮬 {WALK_SIM_REF[m['name']]}" if m["name"] in WALK_SIM_REF else ""
            assert f'<button data-walk="{m["name"]}" disabled><span>{html.escape(m["label"])}</span><small>{html.escape(small)}</small></button>' in page, m
        else:
            tip = html.escape("비활성 — " + str(m["why_disabled"]), quote=True)
            assert f'<button data-walk="{m["name"]}" class="walkOff" title="{tip}" disabled><span>{html.escape(m["label"])}</span>' in page, m
            blocked = walk_move_block(m["name"])
            assert blocked and blocked[0] == 409 and "비활성" in blocked[1] and str(m["why_disabled"]) in blocked[1], blocked
    assert walk_move_block("moonwalk")[0] == 400 and walk_move_block(None)[0] == 400 and all(walk_move_block(n) is None for n in on)
    sup = W.W3Supervisor(cmd_limits=W.W3S3D_CMD_LIMITS)
    for i in range(120):
        sup.observe(100.0 + i * 0.02, (1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    for i, n in enumerate(on):
        cmd = walk_cmd_payload("move", f"rid-{i}", 1000 + i, n)
        assert list(cmd) == ["cmd", "enable", "req_id", "seq", "heading_loop", "vx", "vy", "wz", "dur_s"], list(cmd)
        assert cmd == dict({"cmd": "step", "enable": True, "req_id": f"rid-{i}", "seq": 1000 + i, "heading_loop": False},
                           **{k: W.W3_MOVES[n][k] for k in ("vx", "vy", "wz", "dur_s")}), cmd
        dec = sup.handle(json.loads(json.dumps(cmd)), 102.38, active=True, warmed=True, blend_done=True,
                         ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
        assert dec["accepted"] and dec["action"] == "set_want" and dec["dur_s"] == W.W3_MOVES[n]["dur_s"] and dec["hl"] is False, (n, dec)
    for kind in ("off", "stop", "rearm"):            # 보조 명령은 참고 구현(w3_cmd.build)과 같은 JSON
        ref, rid = _W3CMD.build(kind, 77)
        assert walk_cmd_payload(kind, rid, 77) == ref, (kind, ref)
    ref, rid = _W3CMD.build("on", 78, heading_loop=False, **{k: W.W3_MOVES[mv][k] for k in ("vx", "vy", "wz")})
    mine = walk_cmd_payload("move", rid, 78, mv)
    assert mine.pop("dur_s") == W.W3_MOVES[mv]["dur_s"] and mine == ref, (mine, ref)
    with patch.dict(globals(), W3_SEQ_FILE=os.path.join(directory, "w3_seq_unit")):
        s1, s2 = walk_next_seq(), walk_next_seq()
        s3 = _W3CMD.next_seq(W3_SEQ_FILE, now_ms=1)      # 프로세스를 다시 띄우고 시계가 뒤로 가도 파일이 단조를 지킨다
        assert s1 < s2 < s3, (s1, s2, s3)
    rid = walk_req_id("turn_l", 2 ** 62)
    assert 1 <= len(rid) <= 64 and rid.endswith(f"-turn_l-{2 ** 62}"), rid
    out.append(f"① 버튼 표 = 원본 {len(names)} 항목(값·표시 이름 그대로, 활성 {len(on)}·비활성 {len(off)}"
               + (f" [{', '.join(off)}] 서버 거부·사유 표시" if off else "") + f") · 공개 목록 {'있음' if callable(getattr(W, 'w3_moves_public', None)) else '없음(대체 경로)'}"
               " · 켜기 JSON = 고정 프로토콜·실제 감독기 통과 · 끄기/정지/재무장 = w3_cmd.build · seq 단조 · req_id ≤ 64자")

    # ② 브리지 기동 인자: 기존 키는 종전과 같다(live·selftest 게이트) · 걷기 키에만 실험 플래그 · 키·토픽 = 원본
    script = os.path.join(SCRIPTS, "rl_bridge_node.py")
    assert CKPT_CHOICES == ["stand_crouch_g12", "stand_crouch_g21a", "stand_c5_g21a_g24ra", "stand_c5kr_g21a_g24ra",
                            "stand_c5krh_g21a_g24ra", "stand_crouch_g12_ema0", "stand_crouch_g11a"]
    # 1008 정리: 고를 수 있는 정책 = 합성 키 하나(기본값도 그것). 그 밖의 아는 키는 정책 로드·시험 설정에서 사유를 대고 거부된다.
    assert CKPT_MENU_KEYS == ("stand_kew3q_g21a_g24ra_w3s3d15800",) and CKPT_MENU == list(CKPT_MENU_KEYS) and CKPT_DEFAULT == CKPT_MENU[0]
    _flags = ["--experimental-w3", "--experimental-kew3"]
    started = []
    with patch.object(RUN, "start", side_effect=lambda key, argv, cwd=None: (started.append((key, list(argv))), (True, "mock"))[1]):
        for k in [x for x in CKPT_ALL if x not in CKPT_MENU]:
            assert "정리된 정책" in ckpt_menu_reason(k), k
            try:
                act_bridge_live({"checkpoint": k})
                raise AssertionError(f"정리된 정책 {k} 로 브리지를 띄움")
            except ValueError as exc:
                assert "정리된 정책" in str(exc), exc
            assert not started, started
        for bad in ({}, {"checkpoint": "stand_v7"}, {"checkpoint": "stand_crouch_g12"}):     # 키 없음·선택지 밖·정리된 키 → 계약 검사는 기본값(= 남긴 정책)으로
            ACTIONS["selftest"][0](bad)
            assert started[-1] == ("selftest", [sys.executable, script, "--selftest", "--checkpoint", CKPT_DEFAULT] + _flags), started[-1]
        act_bridge_live({})                                                        # 키를 안 주면 기본값 = 남긴 정책(실험 플래그 + 헤딩 고리 off)
        assert started[-1] == ("bridge", [sys.executable, script, "--live", "--checkpoint", CKPT_DEFAULT] + _flags + ["--w3-heading-loop", "off"]), started[-1]
        with patch.dict(globals(), WALK_IMPORT_ERROR="ImportError: 시험", CKPT_MENU=[]):   # 걷기 모듈 적재 실패 → 옛 정책으로 넘어가지 않고 사유를 대며 거부
            n0 = len(started)
            for fn in (lambda: act_bridge_live({}), lambda: ACTIONS["selftest"][0]({})):
                try:
                    fn()
                    raise AssertionError("걷기 모듈 없이 정책을 띄움")
                except ValueError as exc:
                    assert "걷기 모듈" in str(exc), exc
            assert len(started) == n0
    with WalkBench(directory, CKPT_DEFAULT) as bn:      # 시험 설정(서버): 남긴 정책만 받는다 — 화면을 우회한 요청도 같다
        assert CKPT_MENU == [CKPT_DEFAULT]
        for k in [x for x in CKPT_ALL if x != CKPT_DEFAULT] + ["stand_v7", None]:
            code, res = bn.req("configure", mode="ground", checkpoint=k)
            assert code == 400 and not res["ok"] and TRIAL.checkpoint == CKPT_DEFAULT, (k, code, res)
            assert ("정리된 정책" in res["message"]) == (k in CKPT_ALL), (k, res)
        code, res = bn.req("configure", mode="ground", checkpoint=CKPT_DEFAULT)
        assert code == 200 and res["ok"] and TRIAL.checkpoint == CKPT_DEFAULT, (code, res)
    started = []
    # 아래부터는 "콘솔이 기동 인자를 아는 키" 전부의 인자를 종전대로 고정한다(메뉴를 시험용으로 넓혀서) — 메뉴에 다시 올릴 때 인자가 달라져 있지 않게.
    with patch.dict(globals(), CKPT_MENU=list(CKPT_ALL)), \
            patch.object(RUN, "start", side_effect=lambda key, argv, cwd=None: (started.append((key, list(argv))), (True, "mock"))[1]):
        for k in CKPT_CHOICES:
            assert bridge_extra_args(k, live=True) == [] == bridge_extra_args(k, live=False)
            act_bridge_live({"checkpoint": k})
            ACTIONS["selftest"][0]({"checkpoint": k})
            assert started[-2:] == [("bridge", [sys.executable, script, "--live", "--checkpoint", k]),
                                    ("selftest", [sys.executable, script, "--selftest", "--checkpoint", k])], (k, started[-2:])
        try:
            act_bridge_live({"checkpoint": "stand_v7"})
            raise AssertionError("선택지 밖 체크포인트로 브리지를 띄움")
        except ValueError:
            pass
        # 걷기 키: 실험 플래그는 정책 로드·셀프테스트 게이트 양쪽, 헤딩 고리 off 는 정책 로드에만(브리지 selftest 의 고리 항목은 off 에서 실패한다)
        expect = {W.W3S3D_CKPT_KEY: ["--experimental-w3"],
                  "stand_kew3_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"],
                  "stand_kew3q_g21a_g24ra_w3s3d15800": ["--experimental-w3", "--experimental-kew3"]}       # 1008 둘째 합성 키(허용된 추가)
        for k, flags in expect.items():
            act_bridge_live({"checkpoint": k})
            ACTIONS["selftest"][0]({"checkpoint": k})
            assert started[-2:] == [("bridge", [sys.executable, script, "--live", "--checkpoint", k] + flags + ["--w3-heading-loop", "off"]),
                                    ("selftest", [sys.executable, script, "--selftest", "--checkpoint", k] + flags)], (k, started[-2:])
    src = Path(__file__).with_name("rl_bridge_node.py").read_text(encoding="utf-8")
    kew3 = re.search(r'^KEW3_CKPT_KEY = "([A-Za-z0-9_]+)"', src, re.M)
    kew3q = re.search(r'^KEW3Q_CKPT_KEY = "([A-Za-z0-9_]+)"', src, re.M)
    # 1008: 걷기 키 = 1007 의 두 키(이름·순서·표시·인자 불변) + 허용된 추가 한 키(둘째 합성 키 — 갇힘 해소). 그 밖의 키가 끼면 실패.
    assert kew3 and kew3q and set(WALK_KEYS) == {W.W3S3D_CKPT_KEY, kew3[1], kew3q[1]} == set(WALK_BRIDGE_ARGS) == set(expect) and WALK_CKPTS == list(WALK_KEYS)
    assert list(WALK_KEYS) == [W.W3S3D_CKPT_KEY, kew3[1], kew3q[1]] == list(expect) \
        == ["stand_w3s3d_15800", "stand_kew3_g21a_g24ra_w3s3d15800", "stand_kew3q_g21a_g24ra_w3s3d15800"], list(WALK_KEYS)
    assert WALK_KEYS[W.W3S3D_CKPT_KEY]["composite"] is False and WALK_KEYS[kew3[1]]["composite"] is True
    assert WALK_KEYS[W.W3S3D_CKPT_KEY] == {"composite": False, "label": "걷기 실험 · W3 단독"}
    assert WALK_KEYS[kew3[1]] == {"composite": True, "label": "걷기 실험 · 서기 KE + 걷기 W3 합성"}
    assert WALK_KEYS[kew3q[1]]["composite"] is True and WALK_BRIDGE_ARGS[kew3q[1]] == WALK_BRIDGE_ARGS[kew3[1]]
    # 둘째 합성 키의 표시 이름: 무엇이 다른지(갇힘 해소)·시뮬 평가 구성임·실물 미검증을 적는다. 앞의 두 키 표시에는 그 문구가 없다.
    assert all(s in WALK_KEYS[kew3q[1]]["label"] for s in ("서기 KE + 걷기 W3 합성 + 갇힘 해소(시뮬 평가 구성)", "실물 미검증")), WALK_KEYS[kew3q[1]]["label"]
    assert not any("갇힘 해소" in WALK_KEYS[k]["label"] for k in (W.W3S3D_CKPT_KEY, kew3[1]))
    assert len({v["label"] for v in WALK_KEYS.values()}) == len(WALK_KEYS), "걷기 키의 표시 이름이 겹친다(드롭다운에서 구분 불가)"
    # 합성 키 ↔ 명세 = 브리지 KEW3_SPECS 원문: 두 키 모두 걷기 W3S3d, 갇힘 해소(requiet)는 둘째 키에만
    specs = re.findall(r'(KEW3Q?_CKPT_KEY): KeW3Spec\(w3=_W3\.([A-Z0-9_]+), ke="([a-z0-9_]+)", handoff="carry", settle_quiet_s=0\.5(\)|,\s*requiet=\(2\.0, 0\.3, "both"\)\))', src)
    assert [s[:3] for s in specs] == [("KEW3_CKPT_KEY", "W3S3D_CKPT_KEY", "stand_c5kr_g21a_g24ra"), ("KEW3Q_CKPT_KEY", "W3S3D_CKPT_KEY", "stand_c5kr_g21a_g24ra")], specs
    assert specs[0][3] == ")" and "requiet" in specs[1][3], specs
    assert all(k.startswith("stand") for k in WALK_KEYS) and not set(WALK_KEYS) & set(CKPT_CHOICES) and CKPT_ALL[:len(CKPT_CHOICES)] == CKPT_CHOICES
    for flag in ("--experimental-w3", "--experimental-kew3", "--w3-heading-loop"):
        assert f'ap.add_argument("{flag}"' in src, flag
    for name, value in (("BRIDGE_ACK_TOPIC", BRIDGE_ACK_TOPIC), ("BRIDGE_CMD_TOPIC", BRIDGE_CMD_TOPIC), ("BRIDGE_DEBUG_TOPIC", BRIDGE_DEBUG_TOPIC)):
        assert re.search(rf'^{name} = "([^"]+)"', src, re.M)[1] == value, name
    for k in WALK_CKPTS:
        assert "걷기 실험" in WALK_KEYS[k]["label"], k
        assert (f'<option value="{k}">{k} — {html.escape(WALK_KEYS[k]["label"])}</option>' in page) == (k in CKPT_MENU), k      # 드롭다운에는 남긴 정책만
    assert page.count("<option") - page.count("<option value=\"\"") >= 1 and not any(f"<option>{k}</option>" in page for k in CKPT_CHOICES), "정리된 정책이 드롭다운에 남아 있다"
    out.append(f"②a 정책 메뉴(1008): 고를 수 있는 정책 {CKPT_MENU} 하나 · 그 밖의 아는 키 {len(CKPT_ALL) - len(CKPT_MENU)} 개는 정책 로드·시험 설정 거부(사유 표시) · "
               "키 없는 요청은 남긴 정책으로 · 걷기 모듈 적재 실패 시 거부(옛 정책으로 넘어가지 않음)")
    out.append(f"② 브리지 기동 인자: 기존 {len(CKPT_CHOICES)} 키 live·selftest 종전과 같음 · 걷기 키 {len(expect)} 개에만 실험 플래그(정책 로드에는 헤딩 고리 off 추가, 셀프테스트 게이트에는 없음) · 키·토픽 = w3_deploy·브리지 원본"
               f" · 1007 두 키 불변 + 1008 추가 {kew3q[1]}(첫 합성 키와 같은 인자, 브리지 명세에 갇힘 해소)")

    # ③ 텔레메트리(실제 경로): 응답은 순서대로 쌓였다가 한 번만 꺼내지고, 디버그의 w3·w3_sup·kew3 는 그대로 실린다(없으면 None)
    tel = Telemetry()
    with patch.dict(globals(), now=lambda: 5000.0):
        tel.receive("bridge", {"timestamp": 5000.0, "live": True, "armed": True, "checkpoint": kew3[1], "cmd": [0.0, 0.0, 0.0],
                               "w3": {"want": False, "ge": False, "ge_obs": 0.0, "u": 0.0, "f": 0.0},
                               "w3_sup": {"last_seq": 7, "stop_latched": False, "latch_reason": None, "freeze": None},
                               "kew3": {"mode": "ke", "block": None}, "guard_mode": "data"})
        tel.receive("ack", {"req_id": "a", "accepted": True})
        tel.receive("ack", {"req_id": "b", "accepted": False})
        tel.receive("ack", "not a dict")
        b = tel.view()["bridge"]
        assert tel.view()["fresh"]["bridge"] and "ack" not in tel.view()["fresh"]
    assert b["w3"]["want"] is False and b["w3_sup"]["last_seq"] == 7 and b["kew3"]["mode"] == "ke" and b["guard_mode"] == "data" and b["ts"] == 5000.0
    assert [a["ack"]["req_id"] for a in tel.drain_acks()] == ["a", "b"] and tel.drain_acks() == []
    plain = Telemetry()
    with patch.dict(globals(), now=lambda: 5000.0):
        plain.receive("bridge", {"timestamp": 5000.0, "live": True, "armed": True, "checkpoint": CKPT_DEFAULT})
        pb = plain.view()["bridge"]
    assert pb["w3"] is None and pb["w3_sup"] is None and pb["kew3"] is None and pb["guard_mode"] is None
    with patch.dict(globals(), DEMO=False), patch.object(TEL, "publish", return_value=True) as publish:
        assert walk_publish({"cmd": "stop", "req_id": "x", "seq": 1}) is True
        publish.assert_called_once_with("bridge", {"cmd": "stop", "req_id": "x", "seq": 1})
    # 끊긴(오래된) 디버그 프레임의 want 0·ge 0 은 '서 있음'의 근거가 아니다 — 신선할 때만 읽는다
    probe = TrialState(kew3[1])
    probe.view = {"fresh": {"bridge": False}, "bridge": dict(b), "received": {"bridge": 5000.0}}
    assert probe.walk.bridge_status(probe)["standing"] is False and probe.walk.bridge_status(probe)["w3"] is None
    probe.view = dict(probe.view, fresh={"bridge": True})
    assert probe.walk.bridge_status(probe)["standing"] is True
    probe.view = dict(probe.view, bridge=dict(b, kew3={"mode": "w3", "block": None}))
    assert probe.walk.bridge_status(probe)["standing"] is False and probe.walk.bridge_status(probe)["walking"] is True
    out.append("③ 텔레메트리: 응답 순서대로 1 회만 전달·비객체 무시 · 디버그 w3/w3_sup/kew3/guard_mode 그대로(걷기 세션이 아니면 None) · 끊긴 프레임은 서기 근거 아님 · 발행은 브리지 명령 토픽 1 회")

    for key in WALK_CKPTS:
        composite = WALK_KEYS[key]["composite"]
        # ④ 수락 → 기한 종료(burst_end) → 서기 복귀. '수락됨'과 '서기 복귀 확인됨'은 다른 단계.
        cmd_calls, real_move_cmd = [], W.w3_move_cmd
        with WalkBench(directory, key) as bn, patch.object(
                W, "w3_move_cmd", side_effect=lambda *a, **k: (cmd_calls.append((a, k)), real_move_cmd(*a, **k))[1]):
            bn.prepare()
            w = bn.walk()
            assert w["available"] and w["session"] and w["guard_mode"] == "data" and w["profile"]["checkpoint"] == key, w
            assert [x["name"] for x in w["buttons"]] == names and all(x["enabled"] is x["allowed"] is (x["name"] in on) for x in w["buttons"])
            assert all({k: x[k] for k in VAL} == {k: W.W3_MOVES[x["name"]][k] for k in VAL} for x in w["buttons"])
            assert all(x["sim_ref"] == (WALK_SIM_REF.get(x["name"]) if x["allowed"] else None) for x in w["buttons"]) and "실물 미확인" in w["sim_ref_note"]
            code, res = bn.req("w3_move", move=mv)
            assert code == 200 and res["ok"] and len(bn.sent) == 1, (code, res, bn.sent)
            assert bn.sent[0] == walk_cmd_payload("move", bn.sent[0]["req_id"], bn.sent[0]["seq"], mv)
            cur = bn.walk()["current"]
            assert cur["stage"] == "walking" and cur["deadline"] is True and cur["stand_seen"] is None and not bn.walk()["available"], cur
            walked = bn.until(lambda w: w["current"]["stage"] != "walking", 3.0, "기한 종료")
            w = bn.walk()
            assert w["current"]["stage"] == "ending" and w["current"]["end_by"] == "burst" and not w["available"], w["current"]
            if composite:                             # 래치가 꺼져도(ge 0) 합성 러너가 KE 로 넘기기 전에는 '서기 복귀'가 아니다
                bn.until(lambda w: w["bridge"]["ge"] is False, 2.0, "반주기 래치 해제")
                w = bn.walk()
                assert w["mode"] == "w3" and w["current"]["stage"] == "ending" and not w["bridge"]["standing"], w
            settled = bn.until(lambda w: w["current"]["stage"] == "done", 5.0, "서기 복귀")
            w = bn.walk()
            assert w["bridge"]["standing"] and w["current"]["stand_seen"] and (w["mode"] == "ke") is composite and len(bn.sent) == 1, w
            assert bn.walk_events() == ["walk_request", "walk_ack", "walk_burst_end", "walk_done"], bn.walk_events()
            acks = [r["ack"] for r in bn.timeline() if r.get("kind") == "bridge_ack"]
            assert [a.get("event") for a in acks] == [None, "burst_end"] and {a["req_id"] for a in acks} == {bn.sent[0]["req_id"]}
            json.dumps(bn.state(), ensure_ascii=False, allow_nan=False)
            # 서기 복귀 직후에는 브리지 켜기 조건(조용한 양발 서기 연속)이 차기 전이라 콘솔이 '정착 대기'로 잠가 둔다 → 차면 다시 열리고,
            # 열린 직후의 켜기는 브리지(합성 감독기)도 수락한다 — 걷기 직후 연타가 브리지 거부로 끝나지 않는다
            assert not w["available"] and ("정착 대기" in w["reason"] or "양발 접촉" in w["reason"]), w["reason"]
            reopen = bn.until(lambda w: w["available"], 2.5, "서기 정착 뒤 버튼 다시 열림")
            code, res = bn.req("w3_move", move=mv)
            assert code == 200 and stage(bn) == "walking" and len(bn.sent) == 2, (code, res, bn.walk()["current"])
            bn.until(lambda w: w["current"]["stage"] == "done", 6.0, "둘째 걷기의 서기 복귀")
            assert bn.walk_events() == ["walk_request", "walk_ack", "walk_burst_end", "walk_done"] * 2 and len(bn.sent) == 2
            # 콘솔은 원본 함수에 버튼 이름만 넘긴다 — 범위(limits)도, 평가 전용 우회 인자도 넘기지 않는다
            assert cmd_calls and all(a == (mv,) and not k for a, k in cmd_calls), cmd_calls
        out.append(f"④ {key}: {W.W3_MOVES[mv]['label']} 발행 1 회 → 수락(기한 확인) → {walked:.2f} s 뒤 브리지 기한 종료 → {settled:.2f} s 뒤 서기 복귀"
                   + ("(ge 0 뒤에도 mode ke 전에는 미완료)" if composite else "") + f" · 타임라인 4 사건 · 정착 대기 {reopen:.2f} s 뒤 다시 열려 둘째 걷기도 수락")

        # ⑤ 중복 클릭·진행 중 다른 버튼·새로고침
        with WalkBench(directory, key) as bn:
            bn.prepare()
            p = bn.payload("w3_move", move=mv_long)
            first, again = dispatch_action(p), dispatch_action(json.loads(json.dumps(p)))
            assert first == again and first[0] == 200 and len(bn.sent) == 1, (first, again, bn.sent)
            for move in (mv_long, mv_b):               # 새 요청 식별자로 같은 버튼·다른 버튼
                code, res = bn.req("w3_move", move=move)
                assert code == 409 and "진행 중" in res["message"] and len(bn.sent) == 1, (move, code, res)
            code, res = bn.req("w3_rearm")
            assert code == 409 and len(bn.sent) == 1, (code, res)
            before = bn.walk()["current"]
            fresh_page = json.loads(json.dumps(console_snapshot(), ensure_ascii=False, allow_nan=False))["walk"]   # 새로고침 = 새 /api/state
            assert fresh_page["current"]["req_id"] == before["req_id"] and fresh_page["current"]["stage"] == "walking"
            assert not fresh_page["available"] and not any(x["enabled"] for x in fresh_page["buttons"]) and fresh_page["controls"]["w3_off"]["enabled"]
            bn.until(lambda w: w["current"]["stage"] == "done", 8.0, "서기 복귀")
            assert len(bn.sent) == 1 and bn.walk()["current"]["req_id"] == before["req_id"]
        # ⑥ 브리지 거부(이유 표시) → 다음 클릭은 새 seq
        with WalkBench(directory, key) as bn:
            bn.prepare(settle_s=0.2)                      # 정책 시작 직후: 콘솔이 먼저 '서기 정착 대기'로 거른다(브리지까지 가지 않는다)
            code, res = bn.req("w3_move", move=mv)
            assert code == 409 and "정착 대기" in res["message"] and bn.sent == [], (code, res)
            bn.tick(1.5)
            bn.demo.w3.gyro = (0.6, 0.0, 0.0)             # 콘솔이 볼 수 없는 이유로 브리지가 거부(합성 감독기에 넣는 각속도만 크게)
            bn.tick(0.2)
            code, res = bn.req("w3_move", move=mv)
            cur = bn.walk()["current"]
            assert code == 200 and cur["stage"] == "rejected" and cur["by"] == "bridge" and "정착 미달" in cur["why"], (code, cur)
            assert bn.walk()["bridge"]["want"] is False and bn.walk_events() == ["walk_request", "walk_ack"]
            bn.demo.w3.gyro = (0.02, 0.01, 0.0)
            bn.tick(1.5)
            code, res = bn.req("w3_move", move=mv)        # 거부된 순번은 소비됐다 — 다시 누르면 새 seq·새 req_id
            assert stage(bn) == "walking" and bn.sent[1]["seq"] > bn.sent[0]["seq"] and bn.sent[1]["req_id"] != bn.sent[0]["req_id"]
        # ⑦ 응답 없음: 재전송 없음 · 오래된/남의 응답 무시 · 디버그가 서기를 보일 때까지 차단
        with WalkBench(directory, key) as bn:             # (a) 명령이 브리지에 닿지 않음
            bn.prepare()
            bn.demo.w3.deaf = True
            code, res = bn.req("w3_move", move=mv)
            assert code == 200 and stage(bn) == "sent"
            TRIAL.walk.feed([{"t": bn.t, "ack": {"req_id": "con-old-fwd1-1", "cmd": "step", "accepted": True, "why": "지난 요청", "timestamp": bn.t}},
                             {"t": bn.t, "ack": {"req_id": None, "cmd": "stand", "accepted": True, "why": "남의 응답", "timestamp": bn.t}}])
            bn.tick(1.8)
            assert stage(bn) == "sent" and not bn.walk()["available"]
            bn.tick(0.3)
            w = bn.walk()
            assert w["current"]["stage"] == "no_ack" and "응답 없음" in w["current"]["stage_label"] and len(bn.sent) == 1, w["current"]
            bn.tick(0.1)                                  # 시간 초과 '뒤'의 프레임이 서기를 보임 → 차단 해제(단계는 응답 없음 그대로)
            w = bn.walk()
            assert w["current"]["stage"] == "no_ack" and w["current"]["stand_seen"] and w["available"], w
            bn.tick(5.0)
            assert len(bn.sent) == 1 and "walk_no_ack" in bn.walk_events() and "walk_standing_confirmed" in bn.walk_events()
        with WalkBench(directory, key) as bn:             # (b) 수락·기한 종료 응답이 둘 다 유실(로봇은 실제로 걷는다)
            bn.prepare()
            bn.demo.w3.drop_acks = 2
            bn.req("w3_move", move=mv_long)
            bn.tick(2.1)
            w = bn.walk()
            assert w["current"]["stage"] == "no_ack" and w["current"]["seen_walking"] and not w["current"]["stand_seen"], w["current"]
            assert w["bridge"]["walking"] and not w["available"] and "응답 없음" in w["reason"], w
            code, res = bn.req("w3_move", move=mv)
            assert code == 409 and len(bn.sent) == 1, (code, res)
            bn.until(lambda w: w["current"]["stand_seen"], 5.0, "서기 확인")
            bn.until(lambda w: w["available"], 2.5, "서기 확인·정착 뒤 버튼 다시 열림")
            w = bn.walk()
            assert w["current"]["stage"] == "no_ack" and w["bridge"]["standing"] and len(bn.sent) == 1, w
        with WalkBench(directory, key) as bn:             # (c) 수락 응답만 유실 → 같은 req_id 의 기한 종료 응답으로 수락을 확인
            bn.prepare()
            bn.demo.w3.drop_acks = 1
            bn.req("w3_move", move=mv_long)
            bn.tick(2.1)
            assert stage(bn) == "no_ack"
            bn.until(lambda w: w["current"]["stage"] == "ending", 1.5, "기한 종료 응답")
            cur = bn.walk()["current"]
            assert cur["late_ack"] and cur["accepted"] is True and cur["end_by"] == "burst" and "burst_end" in cur["note"], cur
            bn.until(lambda w: w["current"]["stage"] == "done", 5.0, "서기 복귀")
        with WalkBench(directory, key) as bn:             # (d) 기한 종료 응답만 유실 → 디버그의 last_burst_end 로 끝을 확인
            bn.prepare()
            bn.req("w3_move", move=mv)
            bn.demo.w3.drop_acks = 1
            bn.until(lambda w: w["current"]["stage"] == "ending", 2.0, "기한 종료(디버그)")
            assert "디버그" in bn.walk()["current"]["note"] and bn.walk()["current"]["end_by"] == "burst"
            bn.until(lambda w: w["current"]["stage"] == "done", 5.0, "서기 복귀")
        with WalkBench(directory, key) as bn:             # (e) 수락됐는데 응답에 이 요청의 기한(burst)이 없다 → 경고, 끄기 응답 없이도 디버그로 끝을 본다
            bn.prepare()
            bn.demo.w3.deaf = True
            bn.req("w3_move", move=mv)
            TRIAL.walk.feed([{"t": bn.t, "ack": {"req_id": bn.sent[0]["req_id"], "seq": bn.sent[0]["seq"], "cmd": "step", "accepted": True,
                                                "why": "켜기", "action": "set_want", "timestamp": bn.t, "sup": {"last_seq": bn.sent[0]["seq"]}}}])
            bn.tick()
            cur = bn.walk()["current"]
            assert cur["stage"] in ("walking", "ending") and cur["deadline"] is False, cur
            warned = [r["message"] for r in bn.timeline() if r.get("event") == "walk_ack"]
            assert len(warned) == 1 and "기한(burst)을 걸지 않았습니다" in warned[0], warned
            bn.until(lambda w: w["current"]["stage"] == "done", 2.0, "디버그로 본 걷기 끝·서기")
            assert bn.walk()["current"]["end_by"] == "observed"
        # ⑧ 정지 래치·재무장·걷기 끄기
        with WalkBench(directory, key) as bn:
            bn.prepare()
            code, res = bn.req("w3_rearm")
            assert code == 409 and "정지 래치가 없습니다" in res["message"] and bn.sent == [], (code, res)
            code, res = bn.req("w3_stop")
            assert code == 200 and res["ok"] and bn.sent[-1]["cmd"] == "stop", (code, res)
            bn.tick(0.1)
            w = bn.walk()
            assert w["command"]["kind"] == "stop" and w["command"]["stage"] == "accepted" and w["bridge"]["stop_latched"] is True, w
            assert not w["available"] and "정지 래치" in w["reason"] and not any(x["enabled"] for x in w["buttons"]) and w["controls"]["w3_rearm"]["enabled"]
            n = len(bn.sent)
            code, res = bn.req("w3_move", move=mv)
            assert code == 409 and "정지 래치" in res["message"] and len(bn.sent) == n, (code, res)
            assert bn.req("w3_rearm", confirmed=False)[0] == 400 and len(bn.sent) == n
            code, res = bn.req("w3_rearm")
            assert code == 200 and res["ok"] and bn.sent[-1]["cmd"] == "rearm", (code, res)
            bn.tick(0.1)
            w = bn.walk()
            assert w["command"]["kind"] == "rearm" and w["command"]["stage"] == "accepted" and w["bridge"]["stop_latched"] is False and w["available"], w
            bn.req("w3_move", move=mv_b)
            bn.tick(0.3)
            assert stage(bn) == "walking"
            code, res = bn.req("w3_stop")                 # 걷는 중 정지 래치 → 이 걷기 버튼은 '중단'
            bn.tick(0.1)
            cur = bn.walk()["current"]
            assert code == 200 and cur["stage"] == "interrupted" and "정지 래치" in cur["why"], cur
            bn.until(lambda w: w["bridge"]["standing"], 5.0, "서기")
            bn.tick(0.1)
            assert bn.walk()["current"]["stand_seen"] and not bn.walk()["available"]
        with WalkBench(directory, key) as bn:             # 기한 전 운영자 끄기 → 걷기 끝 요청됨 → 서기 복귀(뒤늦은 burst_end 없음)
            bn.prepare()
            bn.req("w3_move", move=mv_c)
            bn.tick(0.5)
            p = bn.payload("w3_off")
            (code, res), again = dispatch_action(p), dispatch_action(json.loads(json.dumps(p)))
            assert again == (code, res) and sum(1 for s in bn.sent if s.get("enable") is False) == 1, (code, res, again)   # 같은 요청의 재전송 = 한 번만 발행
            bn.tick()
            w = bn.walk()
            assert code == 200 and w["command"]["kind"] == "off" and w["command"]["stage"] == "accepted", (code, w["command"])
            assert w["current"]["stage"] == "ending" and w["current"]["end_by"] == "off", w["current"]
            bn.until(lambda w: w["current"]["stage"] == "done", 5.0, "서기 복귀")
            bn.tick(3.0)
            assert not any(r["ack"].get("event") == "burst_end" for r in bn.timeline() if r.get("kind") == "bridge_ack")
        # ⑨ 정지 요청(estop) → 중단 · 고장 래치 뒤에는 걷기·재무장 불가, 끄기는 여전히 가능
        with WalkBench(directory, key) as bn:
            bn.prepare()
            bn.req("w3_move", move=mv_long)
            bn.tick(0.5)
            assert stage(bn) == "walking"
            dispatch_action({"action": "estop"})
            bn.tick()
            cur = bn.walk()["current"]
            assert cur["stage"] == "interrupted" and "정지" in cur["why"] and "walk_interrupted" in bn.walk_events(), cur
            for action, extra in (("w3_move", {"move": mv}), ("w3_rearm", {})):
                code, res = bn.req(action, **extra)
                assert code == 409 and len(bn.sent) == 1, (action, code, res)
            code, res = bn.req("w3_off")
            assert code == 200 and res["ok"] and bn.sent[-1] == walk_cmd_payload("off", bn.sent[-1]["req_id"], bn.sent[-1]["seq"]), (code, res)
        with WalkBench(directory, key) as bn:             # 브리지는 모르는 콘솔 쪽 고장 래치(센서 신선도 등)도 진행 중인 걷기 버튼을 '중단'으로 닫는다
            bn.prepare()
            bn.req("w3_move", move=mv_long)
            bn.tick(0.3)
            assert stage(bn) == "walking"
            TRIAL.latch("sensor_stale(자가시험)")
            cur = bn.walk()["current"]
            assert cur["stage"] == "interrupted" and "sensor_stale(자가시험)" in cur["why"], cur
            assert bn.walk()["bridge"]["walking"] and bn.req("w3_move", move=mv)[0] == 409 and len(bn.sent) == 1
        # ⑩ 서버가 거르는 요청: 오래된 요청·확인 없음·모르는 버튼·공중 방식 — 아무것도 발행하지 않는다
        with WalkBench(directory, key) as bn:
            bn.prepare()
            for mutate, want in ((lambda p: p.update(seen=bn.t - 30.0), 409), (lambda p: p.pop("seen"), 409),
                                 (lambda p: p.update(confirmed=False), 400), (lambda p: p.update(move="moonwalk"), 400),
                                 (lambda p: p.pop("move"), 400), (lambda p: p.update(boot_id="old"), 409)):
                p = bn.payload("w3_move", move=mv)
                mutate(p)
                code, res = dispatch_action(p)
                assert code == want and not res["ok"], (code, res)
            assert bn.sent == [] and bn.walk()["current"] is None
            assert bn.req("op_slack_done")[0] == 200       # 서기(무보조) 관찰 중 걷기 → 관찰 구간을 '걷기' 사유로 끝낸다
            bn.tick(1.0)
            assert TRIAL.measurement["state"] == "measuring"
            bn.req("w3_move", move=mv)
            m = TRIAL.measurement
            assert m["state"] == "interrupted" and m.get("ended_by") == "walk" and "걷기" in m["reason"], m
            assert "회수 필요" not in bn.state()["trial"]["phase"]
        with WalkBench(directory, key) as bn:
            code, res = bn.req("configure", mode="air", checkpoint=key)
            assert code == 400 and "접지" in res["message"], (code, res)
    out.append(f"⑤–⑩ 걷기 키 {len(WALK_CKPTS)} 개 각각: 중복 클릭·재전송 1 회 발행 · 진행 중 같은/다른 버튼 거부 · 새로고침 뒤 상태 유지 · 브리지 거부 사유 표시 뒤 새 seq · "
               "응답 없음(재전송 없음·서기 확인까지 차단·남의 응답 무시·늦은 기한 종료 응답 수용·디버그로 끝 확인·기한 미확인 경고) · "
               "정지 래치 중 거부·재무장·걷는 중 정지 래치 중단 · 걷기 끄기 · estop 중단 · 오래된 요청·확인 없음·모르는 버튼·공중 방식 거부 · 서기 관찰 종료 사유")

    # ⑪ 걷기 키가 아닌 시험: 걷기 버튼·끄기·정지 래치·재무장 모두 거부, 발행 0
    with WalkBench(directory, CKPT_CHOICES[0]) as bn:
        bn.prepare()
        w = bn.walk()
        assert not w["session"] and not w["available"] and "걷기 실험 키가 아닙니다" in w["reason"] and w["profile"] is None, w
        code, res = bn.req("w3_move", move=mv)
        assert code == 409 and "걷기 실험 키" in res["message"], (code, res)
        for action in ("w3_off", "w3_stop"):
            code, res = bn.req(action)
            assert code == 409 and "걷기 세션이 아닙니다" in res["message"], (action, code, res)
        assert bn.req("w3_rearm")[0] == 409 and bn.sent == [] and bn.walk()["current"] is None
        assert "walk" not in "".join(bn.walk_events())
    out.append(f"⑪ 걷기 키가 아닌 시험({CKPT_CHOICES[0]} — 시험대에서만 고를 수 있음): 걷기 버튼·끄기·정지 래치·재무장 전부 거부(이유 표시) · 발행 0")

    # ⑫ 비활성 버튼(원본 표의 enabled False): 화면에는 비활성 + 사유, 화면을 우회한 API 호출도 서버가 사유와 함께 거부, 다른 버튼은 그대로.
    #    원본 표에 이미 비활성 버튼이 있으면 그것도 보고, 없어도 경로를 지나도록 활성 버튼 하나를 시험 동안만 비활성으로 바꿔 본다.
    target, why = on[-1], "selftest 비활성 사유(시험 전용)"
    with patch.dict(W.W3_MOVES[target], {"enabled": False, "why_disabled": why}):
        with WalkBench(directory, WALK_CKPTS[-1]) as bn:
            bn.prepare()
            w = bn.walk()
            btn = {x["name"]: x for x in w["buttons"]}
            assert w["available"] and btn[target]["allowed"] is False and btn[target]["enabled"] is False, btn[target]
            assert btn[target]["why_disabled"] == why and btn[target]["sim_ref"] is None
            assert all(btn[n]["enabled"] and btn[n]["allowed"] for n in on if n != target)
            for n, reason in [(target, why)] + [(n, str(W.W3_MOVES[n]["why_disabled"])) for n in off]:
                code, res = bn.req("w3_move", move=n)
                assert code == 409 and "비활성" in res["message"] and reason in res["message"], (n, code, res)
                assert dispatch_walk("w3_move", {"move": n})[0] == 409         # 안쪽 함수만 불러도 나가지 않는다
            assert bn.sent == [] and bn.walk()["current"] is None
            tip = html.escape("비활성 — " + why, quote=True)
            assert f'<button data-walk="{target}" class="walkOff" title="{tip}" disabled>' in render_page()
            code, res = bn.req("w3_move", move=mv)                               # 활성 버튼은 그대로 나간다
            assert code == 200 and res["ok"] and len(bn.sent) == 1 and stage(bn) == "walking", (code, res)
    assert walk_move_block(target) is None                                      # 시험용 비활성은 되돌려졌다
    out.append("⑫ 비활성 버튼: 화면 목록에 allowed false + 사유(참고 이동량 없음) · API 직접 호출 409 + 사유 · 발행 0 · 다른 버튼은 정상"
               + (f" (원본 비활성 {', '.join(off)} 포함)" if off else " (원본에 비활성 버튼 없음 — 시험용으로 하나를 꺼서 확인)"))

    # ⑬ 걷기 패널(실험 기능) 안의 예외는 가둔다: 걷기 버튼만 잠기고, 기립 시험의 상태 주기·고장 래치·다른 조작·걷기 끄기는 그대로다
    with WalkBench(directory, WALK_CKPTS[0]) as bn:
        bn.prepare()
        with patch.object(type(TRIAL.walk), "update", side_effect=RuntimeError("자가시험용 상태기 오류")):
            bn.tick(0.2)
        w = bn.walk()
        assert TRIAL.fault is None and not w["available"] and "내부 오류" in w["reason"] and "자가시험용 상태기 오류" in w["internal_error"], w
        assert bn.req("w3_move", move=mv)[0] == 409 and bn.sent == [] and "walk_internal_error" in bn.walk_events()
        assert bn.req("op_push")[0] == 200 and TRIAL.fault is None           # 기립 시험 쪽 조작·기록은 그대로
        code, res = bn.req("w3_off")
        assert code == 200 and res["ok"] and len(bn.sent) == 1, (code, res)  # 안전 방향 명령은 남는다
        with patch.object(type(TRIAL.walk), "snapshot", side_effect=RuntimeError("자가시험용 표시 오류")):
            state = console_snapshot()
        assert state["walk"]["available"] is False and "내부 오류" in state["walk"]["reason"] and state["trial"]["fault"] is None
        with patch.object(type(TRIAL.walk), "begin", side_effect=RuntimeError("자가시험용 발행 준비 오류")):
            code, res = bn.req("w3_stop")
        assert code == 500 and not res["ok"] and "재전송하지 않습니다" in res["message"] and len(bn.sent) == 1, (code, res)
    out.append("⑬ 걷기 패널 내부 오류 가두기: 걷기 버튼만 잠금 + 오류 표시 · 시험 고장 래치 없음 · 다른 조작·걷기 끄기 정상 · 명령 처리 오류는 500 으로 알림(재전송 없음)")
    return out


def selftest_walk_report():
    """콘솔 --selftest 가 부르는 걷기 패널 시험: 자기 임시 폴더에서 selftest_walk 를 돌리고 [PASS] 줄을 찍는다.
    걷기 모듈을 못 불러온 배포에서는 시험을 건너뛰고 그 사실을 찍는다(걷기 조작만 사용 불가, 그 밖의 기능은 종전과 같다)."""
    import contextlib
    import io
    if WALK_IMPORT_ERROR is not None:
        print(f"[경고] 걷기 패널 시험 생략 — 걷기 모듈을 불러오지 못했습니다({WALK_IMPORT_ERROR}). 걷기 조작은 사용 불가, 그 밖의 기능은 종전과 같음")
        return None
    with tempfile.TemporaryDirectory(prefix="console-selftest-walk-") as directory:
        with contextlib.redirect_stdout(io.StringIO()):      # 시험대가 여닫는 임시 세션의 안내 줄은 삼킨다(실패는 예외로 그대로 나온다)
            lines = selftest_walk(directory)
    for line in lines:
        print(f"[PASS] 걷기 패널 {line}")
    return lines


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


def _selftest_errlog():
    """(1007) 모터 오류 기록 — ROS·실물 불필요. ① 오류 0·warnings 없음 → 행에 새 필드 없음(행·관절 키 = 종전 그대로) ② 오류 ≠ 0·warnings 있음 → 필드 있음
    ③ 행 사이에 지나간 값(한 메시지에만 있던 오류·명령으로 비워진 warnings)도 남고, 같은 값을 두 행에 싣지 않음 ④ warnings 는 바뀐 행과 1 s 주기에만,
    상태 메시지를 못 받은 구간은 '모름'(비워졌다고 적지 않음) ⑤ 수신 경로(Telemetry.receive → 래치 → 행) 연결·상태 토픽만 받음
    ⑥ 세션 경계·수신이 끊긴 뒤 옛 값을 싣지 않음 ⑦ 형식이 어긋난 입력에 예외 없음·행은 계속 쓰임
    ⑧ 상태 메시지가 끊긴 첫 행에 stage8_nomsg 1·다시 받은 첫 행에 0 을 한 번씩('모름' 구간의 경계)."""
    from unittest.mock import patch
    row_keys = {"kind", "fwd", "lat", "tilt", "gyro", "quat", "armed", "base", "live", "ckpt", "blend", "contact", "hang_state",
                "sensor_contact", "foot_raw_sum", "foot_pads_on", "foot_pads_counts", "foot_dead_mask", "stage8_running",
                "stage8_command_seq", "stage8_last_command", "joints", "trial", "unverified_force_fields", "fresh", "t", "ts"}
    joint_keys = {"latest_actual_deg", "baseline_deg", "commanded_relative_deg", "desired_relative_deg", "feedback_valid",
                  "feedback_stale", "feedback_age_sec", "feedback_torque_est", "feedback_velocity_rad_s", "feedback_temperature_raw"}
    new_keys = {"stage8_warnings", "stage8_errlog_fail", "stage8_nomsg"}
    AL, AR = "left_ankle_r_joint", "right_ankle_r_joint"
    clock = [1_000_000.0]

    def status(err=None, warnings=()):
        """stage8 상태 토픽 모양 — 관절 항목 키는 stage8 get_status 와 같은 이름."""
        joints = {name: {"allowed": True, "motor_id": i + 1, "baseline_deg": 1.0, "latest_actual_deg": 2.0, "feedback_age_sec": 0.01,
                         "feedback_stale": False, "feedback_valid": True, "desired_relative_deg": 0.0, "commanded_relative_deg": 0.0,
                         "feedback_velocity_rad_s": 0.0, "feedback_torque_est": 0.1, "feedback_temperature_raw": 66,
                         "feedback_error_raw": int((err or {}).get(name, 0)), "feedback_error_ok": not (err or {}).get(name, 0)}
                  for i, name in enumerate(JOINT_NAMES)}
        return {"timestamp": clock[0], "armed": True, "running": True, "accepted": True, "command_seq": 7,
                "last_command": "SET_JOINT_TARGETS", "joints": joints, "warnings": list(warnings), "errors": []}

    with tempfile.TemporaryDirectory(prefix="console-selftest-errlog-") as directory:
        with patch.dict(globals(), SESSION_DIR=directory, now=lambda: clock[0]):
            ERRLATCH.drain()
            tel, rec = Telemetry(), SessionLog()
            rec.ensure("errlog selftest", manual=True)
            path = os.path.join(rec.dir, "timeline.jsonl")

            def rows():
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f]

            def step(*msgs, key="status", view=None):
                """상태 메시지들을 수신 경로로 넣고 50 ms 뒤 행 1 개를 쓴다 → 그 행. view 를 주면 그 화면 상태로 행을 만든다(미리 떠 둔 것)."""
                for m in msgs:
                    tel.receive(key, m)
                clock[0] += 0.05
                rec.sample(tel.view() if view is None else view)
                return rows()[-1]

            def errs(row):
                return {n: j["feedback_error_raw"] for n, j in row["joints"].items() if "feedback_error_raw" in j}

            def size(row):
                """행의 직렬화 길이(시각 필드 t·ts 는 자릿수가 달라지므로 뺀다)."""
                return len(json.dumps({k: v for k, v in row.items() if k not in ("t", "ts")}, ensure_ascii=False))

            # ① 보통 상태: 새 필드 없음(행·관절 키가 종전 그대로 = 행 크기 불변)
            r = step(status(), status(), status())
            assert rows()[0]["kind"] == "session_start" and rows()[0].get("errlog") == 1, rows()[0]
            assert set(r) == row_keys and len(r["joints"]) == 12 and all(set(j) == joint_keys for j in r["joints"].values()), sorted(set(r) ^ row_keys)
            n_plain = size(r)
            # ② ③ 오류 바이트: 한 메시지에만 있었고 최신 메시지는 0 이어도 다음 행에 남는다 → 그다음 행에는 없다
            r = step(status({AL: 4}), status())
            assert errs(r) == {AL: 4} and "feedback_error_seen" not in r["joints"][AL] and set(r) == row_keys, (errs(r), sorted(set(r) ^ row_keys))
            assert set(r["joints"][AL]) == joint_keys | {"feedback_error_raw"} and set(r["joints"][AR]) == joint_keys
            r = step(status())
            assert errs(r) == {} and set(r) == row_keys and size(r) == n_plain, "오류가 사라진 뒤 행은 보통 행 크기"
            #    오류가 계속되면 메시지가 올 때마다 행에 남는다. 메시지가 끊기면 옛 값을 되풀이하지 않고('모름'),
            # ⑧ 끊긴 첫 행에 stage8_nomsg 1 · 다시 받은 첫 행에 0 을 한 번씩 남긴다 — 그 사이 행의 '필드 없음'은 '오류 없음'이 아니다
            r = step(status({AR: 5}))
            assert errs(r) == {AR: 5} and "stage8_nomsg" not in r, "받는 채 이어지는 행에는 표식이 없다"
            assert errs(step(status({AR: 5}))) == {AR: 5}
            r = step()
            assert errs(r) == {} and r.get("stage8_nomsg") == 1 and set(r) == row_keys | {"stage8_nomsg"}, (errs(r), r.get("stage8_nomsg"), "끊긴 첫 행: 옛 값 없이 표식 1")
            r = step()
            assert set(r) == row_keys and size(r) == n_plain, (sorted(set(r) ^ row_keys), "끊긴 채 이어지는 행에는 아무것도 적지 않는다")
            r = step(status({AR: 5}))
            assert errs(r) == {AR: 5} and r.get("stage8_nomsg") == 0, (errs(r), r.get("stage8_nomsg"), "다시 받은 첫 행: 표식 0 + 그 메시지의 값")
            r = step(status({AR: 5}))
            assert errs(r) == {AR: 5} and "stage8_nomsg" not in r, "표식 0 은 한 번만"
            r = step(status())
            assert errs(r) == {} and set(r) == row_keys and size(r) == n_plain, "메시지를 받는 중의 '필드 없음' = 오류 없음(보통 행)"
            #    한 행 구간에 서로 다른 값이 둘: 최근 값 + 본 값 전부(오래된 것 → 최근)
            r = step(status({AL: 4, AR: 5}), status({AL: 2, AR: 5}), status({AR: 5}))
            assert errs(r) == {AL: 2, AR: 5} and r["joints"][AL].get("feedback_error_seen") == [4, 2] and "feedback_error_seen" not in r["joints"][AR], r["joints"][AL]
            assert errs(step(status())) == {}
            #    행을 만드는 화면 상태(view)를 뜬 뒤에 도착한 메시지가 있어도 순서가 뒤집히지 않고, 이미 적은 값이 다음 행에 다시 실리지 않는다
            tel.receive("status", status({AL: 4}))
            old_view = tel.view()
            tel.receive("status", status({AL: 2}))
            r = step(view=old_view)
            assert errs(r) == {AL: 2} and r["joints"][AL].get("feedback_error_seen") == [4, 2], r["joints"][AL]
            assert errs(step(status({AL: 4}))) == {AL: 4}
            old_view = tel.view()                                   # 이 화면 상태의 최신 값은 4(방금 행에 적음)
            assert errs(step(status(), view=old_view)) == {}, "앞 행에 적은 한 메시지짜리 값이 다음 행에 다시 실렸다"
            # ② ③ ④ warnings: 명령으로 비워진 뒤에도(최신 메시지 []) 합집합이 남고, 그대로면 1 s 마다만, 비워진 것을 보면 [] 한 번
            w1, w2 = "feedback_error:%s:4" % AL, "joint_feedback_stale:%s" % AR
            r = step(status({}, [w1, w2]), status({}, []))
            assert r.get("stage8_warnings") == sorted([w1, w2]), r.get("stage8_warnings")
            t_emit, n_emit = [clock[0]], 0
            for _k in range(45):                                   # 2.25 s 동안 같은 warnings 가 계속(명령마다 비워졌다 다시 켜짐)
                r = step(status({}, [w1, w2]), status({}, []))
                if "stage8_warnings" in r:
                    assert r["stage8_warnings"] == sorted([w1, w2])
                    assert S8_WARN_KEYFRAME_SEC - 1e-6 <= clock[0] - t_emit[-1] <= S8_WARN_KEYFRAME_SEC + 0.05 + 1e-6, clock[0] - t_emit[-1]
                    t_emit.append(clock[0]); n_emit += 1
                else:
                    assert set(r) == row_keys, sorted(set(r) ^ row_keys)
            assert n_emit == 2, f"같은 warnings 45 행 중 다시 적은 행 {n_emit} 개(기대 2 = 1 s 주기)"
            r = step(status({}, [w1]))
            assert r.get("stage8_warnings") == [w1], "내용이 바뀐 행"
            #    상태 메시지를 못 받은 구간·warnings 가 실리지 않은 메시지는 '비워짐'이 아니다 → 아무것도 적지 않고, 다시 받으면 이어서 비교한다
            r = step()
            assert "stage8_warnings" not in r and r.get("stage8_nomsg") == 1, "메시지를 못 받은 구간에 [] 가 찍혔다(표식 1 만 있어야 한다)"
            no_key = status()
            del no_key["warnings"]
            r = step(no_key)
            assert "stage8_warnings" not in r and r.get("stage8_nomsg") == 0, "warnings 가 실리지 않은 메시지에 [] 가 찍혔다(받기는 했으므로 표식 0)"
            assert "stage8_warnings" not in step(status({}, [w1])), "수신이 이어졌을 뿐인데 다시 적었다"
            r = step(status())
            assert r.get("stage8_warnings") == [], "비워진 것을 본 행에 [] 한 번"
            r = step(status())
            assert set(r) == row_keys and size(r) == n_plain, "warnings 가 비워진 뒤 행은 보통 행 크기"
            # ⑤ 상태 토픽만 받는다(브리지 디버그 토픽에 같은 모양이 와도 무시)
            r = step(status({AL: 9}, ["x"]), key="bridge")
            assert errs(r) == {} and "stage8_warnings" not in r and r.get("stage8_nomsg") == 1, "다른 토픽은 상태 메시지로 세지 않는다"
            # ⑥ 세션 경계: 닫힌 동안 지나간 값·수신이 끊긴 뒤의 옛 값은 새 세션에 싣지 않고, 이어지는 warnings 는 새 세션에서 다시 적는다
            assert step(status({AL: 3}, ["keep"])).get("stage8_warnings") == ["keep"]
            rec.close("first done")
            tel.receive("status", status({AL: 3}, ["keep", "gone"]))
            rec.sample(tel.view())                                  # 닫힌 세션: 행은 안 쓰지만 받아 둔 값은 비운다
            rec.ensure("errlog selftest 2", manual=True)
            path = os.path.join(rec.dir, "timeline.jsonl")
            r = step()                                              # 새 세션 첫 행, 받은 메시지 없음(마지막 상태 메시지에는 오류 3 이 남아 있다)
            assert rows()[0].get("errlog") == 1 and errs(r) == {} and "stage8_warnings" not in r, (errs(r), r.get("stage8_warnings"))
            assert set(r) == row_keys, "끊김 표식은 세션 안에서만 센다(앞 세션에서 받고 있었어도 새 세션 첫 행에 1 을 적지 않는다)"
            r = step(status({}, ["keep"]))                          # 앞 세션에서 0.1 s 전에 같은 내용을 적었어도(1 s 주기 전) 새 세션에서는 다시 적는다
            assert errs(r) == {} and r.get("stage8_warnings") == ["keep"] and "stage8_nomsg" not in r, (errs(r), r.get("stage8_warnings"), r.get("stage8_nomsg"))
            assert "stage8_warnings" not in step(status({}, ["keep"]))
            assert step(status()).get("stage8_warnings") == []
            # ⑦ 형식이 어긋난 입력: 수신·행 쓰기 모두 예외 없음, 양에 상한, 쓴 행은 JSON 으로 다시 읽힘
            bad = [{}, {"joints": [1, 2], "warnings": "abc"}, {"joints": {AL: "x", AR: None}, "warnings": {"a": 1}},
                   {"joints": {AL: {"feedback_error_raw": "4"}, AR: {"feedback_error_raw": True}}, "warnings": [1, None, ["x"], {"y": 2}]},
                   {"joints": {AL: {"feedback_error_raw": float("nan")}, AR: {"feedback_error_raw": float("inf")}}},
                   {"joints": {AL: {"feedback_error_raw": 2.5}, AR: {"feedback_error_raw": 0.0}}},
                   {"joints": {"j%02d" % i: {"feedback_error_raw": 7} for i in range(40)}},
                   {"joints": {AL: {"feedback_error_raw": 1}}, "warnings": ["ok\ud800tail", "z" * 1000] + ["w%03d" % i for i in range(500)]}]
            n0 = len(rows())
            for b in bad:
                r = step(b)
            assert len(rows()) == n0 + len(bad), "불량 입력에서도 행은 계속 쓰여야 한다"
            assert not any("stage8_nomsg" in x for x in rows()[n0:]), "형식이 어긋난 메시지도 '받은 메시지'로 센다(끊김 표식 없음)"
            assert "stage8_errlog_fail" not in r and errs(r) == {AL: 1}, (r.get("stage8_errlog_fail"), errs(r))
            ws = r.get("stage8_warnings")
            assert len(ws) == S8_WARN_MAX + 1 and "+more" in ws and "ok?tail" in ws and max(len(w) for w in ws) == S8_WARN_MAXLEN, (len(ws), ws[:3])
            assert not any("stage8_errlog_fail" in x for x in rows()[n0:]), "불량 입력이 래치 안에서 예외를 냈다"
            many = rows()[n0 + 6]
            assert sum(1 for j in many["joints"].values() if j.get("feedback_error_raw") == 7) == S8_ERR_JOINTS_MAX, "받아 두는 관절 수 상한"
            #    파일에 쓸 수 없는 관절 이름(짝 없는 서러게이트)·너무 긴 이름·비 ASCII 이름은 받아 두지 않는다 — 행의 키가 되어 행 쓰기를 깨면 안 된다
            n1 = len(rows())
            r = step({"joints": {"bad\ud800name": {"feedback_error_raw": 4}, "x" * (S8_ERR_NAME_MAXLEN + 1): {"feedback_error_raw": 4},
                                 "한글관절": {"feedback_error_raw": 4}}}, status())
            assert len(rows()) == n1 + 1 and errs(r) == {} and len(r["joints"]) == 12, (len(rows()) - n1, errs(r), len(r["joints"]))
            for k in range(S8_ERR_CODES_MAX + 4):                  # 한 관절에 서로 다른 값이 상한보다 많이
                tel.receive("status", status({AL: k + 1}))
            r = step()
            assert r["joints"][AL]["feedback_error_raw"] == S8_ERR_CODES_MAX + 4 and len(r["joints"][AL]["feedback_error_seen"]) == S8_ERR_CODES_MAX

            class Boom(dict):
                def get(self, *a):
                    raise RuntimeError("boom")
            ERRLATCH.feed(Boom()); ERRLATCH.feed(None)
            r = step(status())
            assert r.get("stage8_errlog_fail") == 2, "래치 내부 예외는 세어서 행에 남긴다"
            with patch.object(rec, "_s8_errlog", side_effect=RuntimeError("boom")):
                r = step(status({AL: 4}))
            assert r.get("stage8_errlog_fail") == -1 and set(r) - new_keys == row_keys, "행 덧붙임이 실패해도 행은 쓴다"
            rec.close("errlog selftest done")
    return (f"오류 0 → 새 필드 없음(행·관절 키와 행 크기 그대로) · 오류 ≠ 0 → feedback_error_raw · warnings → stage8_warnings(바뀐 행·{S8_WARN_KEYFRAME_SEC:g} s 주기·비워진 것을 보면 [] 한 번) · "
            f"행 사이 값 유지·같은 값 두 번 안 실림 · 수신 없는 구간은 '모름'(끊긴 첫 행 stage8_nomsg 1·다시 받은 첫 행 0) · 세션 경계 · 불량 입력 {len(bad)} 종·쓸 수 없는 관절 이름 무예외·내부 예외 표시")


def selftest():
    """Small deployment smoke test. No ROS, subprocesses, markers or real logs."""
    from unittest.mock import patch
    selftest_walk_report()       # 1007 걷기 패널(걷기 실험 키) 시험을 먼저 돌려 결과 줄을 찍는다 — 실패는 예외로 그대로 나온다
    with tempfile.TemporaryDirectory(prefix="console-selftest-") as directory:
        with patch.dict(globals(), SESSION_DIR=directory):
            recorder = SessionLog()
            recorder.ensure("offline selftest", manual=True)
            recorder.event("check")
            recorder.sample(Telemetry().view())
            assert recorder.status()["samples"] == 1
            recorder.close("test complete")
        # 0914 12축 개별 제어기: 액션 등록·요청 파일 쓰기·한계 거부·상태기 허용 규칙
        assert {"joint_trim", "joint_trim_zero"} <= set(ACTIONS) and {"joint_trim", "joint_trim_zero"} <= set(LABELS)
        with patch.dict(globals(), JOINT_TRIM_FILE=os.path.join(directory, "joint_trim_deg.json"),
                        HANG_STATUS_FILE=os.path.join(directory, "hang_stand_status.json")):
            ok, msg = act_joint_trim({"joint": "left_ankle_r_joint", "delta": 0.5}); assert ok, msg
            ok, msg = act_joint_trim({"joint": "left_ankle_r_joint", "delta": 2.0}); assert ok, msg
            assert abs(_read_joint_trim()["left_ankle_r_joint"] - 2.5) < 1e-9
            for bad in ({"joint": "nope", "delta": 1.0}, {"joint": "left_knee_joint", "delta": 3.0},
                        {"joint": "left_knee_joint", "value": 101.0}, {"joint": "left_knee_joint", "value": float("nan")}):
                try:
                    act_joint_trim(bad); raise AssertionError(f"거부돼야 함: {bad}")
                except ValueError:
                    pass
            ok, _ = act_joint_trim_zero({}); assert ok and all(v == 0.0 for v in _read_joint_trim().values())
            assert read_hang_status() is None
            with open(HANG_STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "kp": 150.0, "joint_trim_enabled": True}, f)
            assert read_hang_status()["kp"] == 150.0
        # 0915: 영점 배수 보정 완료 판정 — 프로세스 키 'zero' rc 0 → finish(True) (이전: 분기 없음 → 시간 초과 오판)
        _tz = TrialState(CKPT_DEFAULT); _tz.completed.update({"can_up", "sensors"})
        _tz.view = {"server_time": 1000.0, "fresh": {"status": False, "imu": True, "foot": True, "bridge": False}, "stage8": {}, "bridge": {},
                    "procs": {}, "process_check_ok": True, "session": {"active": True}, "external_controls": [], "received": {}}
        assert _tz.reason("zero_multiple") is None, _tz.reason("zero_multiple")
        _tz.begin("zero_multiple", 999.0); _tz.pending["launched"] = True
        _tz.view = dict(_tz.view, procs={"zero": {"alive": False, "rc": 0}}); _tz.update(_tz.view)
        assert _tz.pending is None and "zero_multiple" in _tz.completed and _tz.last_result["ok"], (_tz.pending, _tz.last_result)
        assert _tz.reason("zero_multiple") is None   # 재실행 허용(완료 목록과 무관)
        # 0918 자동 수평: 접촉 없으면 대기, 편차 크면 일시정지, 정상이면 힙앞뒤 트림을 속도 제한 안에서 조정(부호: err>0 → 왼 +/오른 −)
        with patch.dict(globals(), JOINT_TRIM_FILE=os.path.join(directory, "joint_trim_al.json")):
            _write_joint_trim({k: 0.0 for k in JOINT_NAMES}); act_auto_level_on({"target": 0.0})
            base_view = {"hang_hold": {"kp": 150.0}, "fresh": {"imu": True}, "imu": {"fwd": -3.0}, "foot": {"contact": [False, False]}}
            auto_level_tick(base_view); assert "접촉" in AUTO_LEVEL["status"], AUTO_LEVEL["status"]
            auto_level_tick(dict(base_view, foot={"contact": [False, True]}, imu={"fwd": -20.0})); assert "일시정지" in AUTO_LEVEL["status"]
            auto_level_tick(dict(base_view, foot={"contact": [False, True]})); assert "양발" in AUTO_LEVEL["status"], AUTO_LEVEL["status"]   # 0919: 한 발 접촉은 적분 금지
            both = dict(base_view, foot={"contact": [True, True]}, imu={"fwd": -3.0, "gyro": [0.0, 0.0, 0.0]})
            for _ in range(AL_QUIET_TICKS - 1):
                auto_level_tick(both); assert "정착" in AUTO_LEVEL["status"], AUTO_LEVEL["status"]      # 조용한 틱이 AL_QUIET_TICKS 개 쌓일 때까지 대기
            r = _read_joint_trim(); assert abs(r["left_hip_f_joint"]) < 1e-9, r
            auto_level_tick(both); r = _read_joint_trim()
            assert abs(r["left_hip_f_joint"] - 0.25) < 1e-9 and abs(r["right_hip_f_joint"] + 0.25) < 1e-9, r   # 0.5°/s × 0.5 s = 0.25°
            auto_level_tick(dict(both, imu={"fwd": -3.0, "gyro": [0.0, 0.5, 0.0]})); assert "정착" in AUTO_LEVEL["status"] and AUTO_LEVEL["quiet"] == 0   # 밀기(각속도) → 적분 정지·카운터 리셋
            auto_level_tick(dict(base_view, foot={"contact": [True, True]}, imu={"fwd": 0.1})); assert AUTO_LEVEL["status"] == "수평"
            act_auto_level_off({}); assert not AUTO_LEVEL["on"]
        _ts = TrialState(CKPT_DEFAULT); _ts.completed.update({"can_up", "sensors", "boot_pose", "selftest", "stage8", "hang_stand", "gain_ramp"})
        _base = {"server_time": 1000.0, "fresh": {"status": True, "imu": True, "foot": True, "bridge": False}, "stage8": {"waiting": True},
                 "bridge": {"live": False}, "procs": {"hang_stand": {"alive": True, "hold": {"kp": 150.0, "ts": 999.5}}, "stage8": {"alive": True}},
                 "process_check_ok": True, "session": {"active": True}, "external_controls": []}
        _ts.view = _base; assert _ts.reason("joint_trim") is None and _ts.reason("joint_trim_zero") is None
        _v = dict(_base); _v["procs"] = {"hang_stand": {"alive": True, "hold": {"kp": 20.0, "ts": 999.5}}, "stage8": {"alive": True}}
        _ts.view = _v; assert "운용 게인" in _ts.reason("joint_trim")
        _v = dict(_base); _v["procs"] = dict(_base["procs"], bridge={"alive": True}); _ts.view = _v; assert "브리지" in _ts.reason("joint_trim")
        _ts.view = _base; _ts.completed.add("handoff"); assert _ts.reason("joint_trim") is not None; _ts.completed.discard("handoff")
        t = TrialState(CKPT_DEFAULT)
        assert t.reason("estop") is None
        assert t.reason("arm") is not None
        assert t.reason("release") is not None
        assert bridge_cmd_payload("stand") == {"cmd": "stand"}
        assert bridge_cmd_payload("stand", {"hang_arm": True})["hang_arm"] is True
        # 1005 정책 ON: 종전 stand payload 불변 · touchdown_start 는 순서기만 · 착지 대기 브리지 홀드는 '정책' 아님 · 선행조건
        assert bridge_cmd_payload("stand", {"hang_arm": False, "touchdown_start": False}) == {"cmd": "stand"}
        assert bridge_cmd_payload("stand", {"touchdown_start": True}) == {"cmd": "stand", "touchdown_start": True}
        assert "policy_on" in LABELS and "policy_on" in CONFIRM_ACTIONS and tuple(POLICY_ON_STEPS) == ("ground", "handoff", "baseline", "arm", "bridge", "stand")
        _tp = TrialState(CKPT_DEFAULT); _tp.completed.update({"arm", "bridge", "stand"})
        _S8 = {"armed": True, "running": True}; _FR = {"status": True, "bridge": True, "imu": True, "foot": True}
        _tp.view = {"stage8": _S8, "bridge": {"armed": True, "start_gate": {"state": "WAIT_CONTACT"}}, "fresh": _FR}
        assert _tp.commander() == "hold", _tp.commander()
        _tp.view = dict(_tp.view, bridge={"armed": True, "start_gate": {"state": "POLICY"}}); assert _tp.commander() == "policy"
        _tp.view = dict(_tp.view, bridge={"armed": True}); assert _tp.commander() == "policy"          # 종전 경로(필드 없음)
        _tq = TrialState(CKPT_DEFAULT); _tq.view = _ts.view; assert "먼저" in _tq.reason("policy_on")   # 준비 전 거부
        _tq.completed.update(_ts.completed); _tq.view = dict(_base, foot={"contact": [True, True]}, demo=True)
        assert "양발 접촉" in _tq.reason("policy_on"), _tq.reason("policy_on")
        _tq.view = dict(_base, foot={"contact": [False, False]}, demo=True); assert _tq.reason("policy_on") is None, _tq.reason("policy_on")
        _tq.macro = {"action": "policy_on", "step": "arm"}; assert "정책 ON" in _tq.reason("bridge") and _tq.reason("estop") is None
        with patch.object(TEL, "publish", return_value=False) as publish:
            assert act_estop()[0] is False
            assert publish.call_count == 2
        page = render_page()
        assert "<!--CKPT_OPTIONS-->" not in page
        assert f'<option value="{CKPT_DEFAULT}">' in page and page.count("<option value=\"stand_") == len(CKPT_MENU) == 1
        src = Path(__file__).with_name("rl_bridge_node.py").read_text(encoding="utf-8")
        body = src.split("CHECKPOINTS = {", 1)[1].split("\n}\n", 1)[0]
        known = set(re.findall(r'^\s+"([A-Za-z0-9_]+)": os\.path\.join', body, re.M))
        assert set(CKPT_CHOICES) <= known, "console checkpoint not in bridge"
    print("[PASS] 모터 오류 기록(1007): " + _selftest_errlog())
    print("console selftest: ALL PASS (offline only; hardware not exercised)")
    return 0


def main():
    global DEMO, DEMO_SOURCE, SESSION_DIR, W3_SEQ_FILE
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
        W3_SEQ_FILE = os.path.join(SESSION_DIR, "w3_cmd_seq")     # 1007: 데모는 실제 순번 파일(~/.w3_cmd_seq)을 건드리지 않는다
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
                if not DEMO and tick % max(1, int(SAMPLE_HZ * AL_TICK_S)) == 0:
                    try:
                        auto_level_tick(view)
                    except Exception as _exc:
                        AUTO_LEVEL["status"] = f"오류 {type(_exc).__name__}"
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
