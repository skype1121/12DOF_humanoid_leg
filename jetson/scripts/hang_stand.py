"""매달림 기립 — 지그에 매달린 로봇을 스탠드로 홀드 후, 줄을 풀며 체중 이전 (2026-07-30).

절차 (승윤님 수동 하강 + 원격 파일 트리거):
  1. AIR_RAMP   : 현재각→스탠드(하드웨어맵 v5) 저게인 동기 램프 (kp20/kd1, 최대 5°/s)
  2. AIR_HOLD   : 매달림 유지. 트림 프로브 — 채널 3종 (모두 0.5°/s 램프):
                  /tmp/ankle_trim_deg : + = 발등굽힘 (m5 +/m11 −)
                  /tmp/hip_trim_deg   : + = 힙 신전 (m1 −/m7 +) — 발목과 같은 값이면
                                        "다리 앞기울임+몸통 수직" = 골반 전방 이동
                  /tmp/knee_trim_deg  : + = 무릎 굽힘 (m4 +/m10 −)
                  부호 근거 = 웅크림 실측(pose_record_20260801_crouch).
                  무부하에서 하드스톱에 닿으면 추종오차 증분이 벌어짐 → 3틱 연속
                  초과 시 해당 채널 동결+경고 (저토크라 무해). 프로브 후 0으로
                  되돌릴 것 (0 복귀 시 베이스라인 리셋 — 재프로브 가능).
  3. GAIN_RAMP  : /tmp/hang_gain 생성 시 kp20/kd1 → kp150/kd5 선형 램프 (8s).
                  전 모터 오차 3° 미만일 때만 수락 (아니면 파일 삭제+거부 사유 출력).
                  ※ 반드시 발이 땅에 닿기 전에. 완료 후 줄을 천천히 풀어 착지.
  4. 착지 선언  : 발이 닿으면 touch /tmp/hang_ground — 무부하 스톨가드 해제
                  (접지 후엔 부하 오차가 정상이므로). 이후 3채널 트림 라이브 조정
                  (하중 상태에서 손=힘센서 티트레이션, 0.25° 단위 권장).
                  오탐 동결은 touch /tmp/hang_trim_unfreeze 로 해제.
  5. 종료       : /tmp/hang_stand_stop → 소프트 릴리즈: 명령 동결 + 게인만 3s 램프다운
                  (줄 재인장 후 사용!). /tmp/hang_stand_kill → 즉시 릴리즈 (비상).
                  /tmp/hang_stand_handoff → 핸드오프 종료: 릴리즈 없이 즉시 종료
                  — AK 모터는 마지막 MIT 명령(현 자세·운용게인)을 계속 잡고 있으므로
                  로봇이 선 채로 stage8+정책 스택에 무토크 공백 없이 인계됨
                  (0802 프로토콜: 에어 라이브 금지 — 정책은 접지 스탠드에서 시작).
                  인계 실패 시 회수는 release_all.py(줄 인장 후) 또는 hang_stand 재기동.
                  --hold 초과 시: 저게인이면 소프트 릴리즈, 운용게인(하중 가능성)이면
                  자동 릴리즈 없이 경고 반복 — 파일 트리거로만 종료.

안전:
  - 가드: |실측−명령| > 15°(어느 모터든) / 피드백 0.5s 두절 / 모터 폴트 2회 관측
    → 즉시 릴리즈 (줄이 최후 안전망). 폴트 1회는 경고만 (일시 스파이크 디바운스).
  - 시그널: 1회(SIGINT/SIGTERM/SIGHUP) → 소프트 정지 경로, 2회 → 즉시 릴리즈.
    SSH 단절(pty 사망) 시에도 print OSError를 삼켜 소프트 경로가 완주됨.
  - 릴리즈 = 제로게인 3연발 + MIT 해제. 어떤 경로로 끝나도 finally 보장.
  - ROM 클램프는 '스탠드 영점 상대값'(프레임 불변). 발목 트림은 클램프 대신
    무부하 스톨가드로 보호 (m5 상한 41.1°는 스윕 미완 의심 — 실측으로 확인).
  - 게인 상한 kp150/kd5 (하드웨어맵 승인값, kd5=실물 상한 규칙)
  - 제어 시계는 monotonic (NTP 스텝 보정에 면역)
  - 기구 고정(1009): 하드웨어맵에 "mech_fixed": true 로 선언된 모터는 송신·읽기·가드·상태·핸드오프
    파일 대상에서 뺀다("모터 N 기구 고정 — 제외" 출력). 살아 있는 모터의 가드(이탈·피드백 두절·폴트)는 그대로다.

사용: python3 hang_stand.py [--hold 900] [--channel can1]
"""
import argparse
import json
import math
import os
import signal
import statistics
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402

from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402
from mech_fixed import load_fixed, note as fixed_note  # noqa: E402

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
STOP_FILE = "/tmp/hang_stand_stop"
KILL_FILE = "/tmp/hang_stand_kill"
HANDOFF_FILE = "/tmp/hang_stand_handoff"
GAIN_FILE = "/tmp/hang_gain"
GROUND_FILE = "/tmp/hang_ground"
TRIM_FILE = "/tmp/ankle_trim_deg"
HIP_TRIM_FILE = "/tmp/hip_trim_deg"
KNEE_TRIM_FILE = "/tmp/knee_trim_deg"
UNFREEZE_FILE = "/tmp/hang_trim_unfreeze"
JOINT_TRIM_FILE = "/tmp/joint_trim_deg.json"     # 0914 12축 개별 제어기: {관절명: deg(관절 프레임)} — 콘솔이 씀
STATUS_FILE = "/tmp/hang_stand_status.json"
HOLD_KD_FILE = "/tmp/hold_kd"                      # 0914 자세 홀드 kd 라이브 조절(콘솔 슬라이더). 없으면 KD_MAX(=5)      # 0914 콘솔 표시용 상태(0.5 s)
JOINT_TRIM_LIM_DEG = 100.0                         # 관절당 개별 조정 상한 [deg, 관절 프레임] — 0914 승윤님 요청 ±100(실효 한계는 ROM 마진 3° 클램프)
JOINT_LIMIT_FILES = ("/home/mama/rl_calib/joint_limits_12dof.json",
                     "/home/mama/humanoid_repo/config/joint_limits_12dof.json")
JOINT_LIMIT_MARGIN_DEG = 1.0                       # 소프트 한계 안쪽 여유 [deg]
AIR_GAIT_FILE = "/tmp/air_gait.json"               # 0915 에어 보행(사진용): {"on":bool,"freq_hz","hip_deg","knee_deg","ankle_deg"} — 콘솔/robotctl 이 씀
AIR_GAIT_LIM = {"freq_hz": (0.2, 1.0), "hip_deg": (0.0, 40.0), "hip_ext_deg": (0.0, 25.0), "knee_deg": (0.0, 45.0), "ankle_deg": (0.0, 15.0)}
AIR_GAIT_DEFAULT = {"on": False, "freq_hz": 0.5, "hip_deg": 30.0, "hip_ext_deg": 12.0, "knee_deg": 30.0, "ankle_deg": 8.0}
# hip_deg = 앞스윙(굴곡) 진폭, hip_ext_deg = 뒤스윙(신전) 진폭. 힙앞뒤 소프트 한계(왼 −29~45 / 오른 −40.7~29)에 맞춰 상한 40/25.
AIR_GAIT_FADE_S = 2.0                              # 시작/정지 진폭 페이드 [s]
JOINT_TRIM_RATE_DEG_S = 30.0                       # 개별 조정 램프 [deg/s] — 0914 승윤님 "바로 따라오게": 에어램프(5)·posture_probe(10) 사이, 슬루 제한은 유지
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
IDS = list(range(1, 13))

#: 실측 ROM의 '스탠드 영점 상대값' (rom_sweep2 − stand_v4, 프레임 불변).
ROM_REL = {
    1: (-146.4, 4.2), 2: (-15.7, 18.7), 3: (-91.0, 80.3), 4: (-78.5, 38.0),
    5: (-64.9, 6.8), 6: (-30.4, 29.1), 7: (-30.3, 109.1), 8: (-31.5, 18.4),
    9: (-75.8, 63.1), 10: (-89.9, 42.1), 11: (-53.5, 37.3), 12: (-30.8, 28.3),
}

KP_AIR, KD_AIR = 20.0, 1.0
KP_MAX, KD_MAX = 150.0, 5.0
DEV_ABORT_DEG = 15.0
#: 관절별 이탈 한계 [deg]. 승윤님 승인(0928) — 무릎만 따로, 지속 조건은 넣지 않는다.
#:
#: 왜 무릎만 다른가 (실측):
#:   무하중에서 링크각 = 엔코더각(±1°)이고 고정 영점 오차는 없다. 하중이 실리면 엔코더각이
#:   명령에서 **+12.6°** 벌어진다 — 엔코더 뒤 직렬 컴플라이언스의 탄성 처짐이고 고장이 아니다
#:   (k ≈ 35~45 Nm/rad, 대역 무관, 좌우 차 ≤1.5°). 힙·발목은 이만큼 크지 않다.
#:   종전 15° 는 이 정상 처짐 위에 2.4° 밖에 안 남긴다. 검사는 매 틱(50Hz)·한 틱만 넘어도
#:   전 모터 릴리즈라, 착지 충격 한 번이면 터진다. 0928 실기에서 착지 도중 실제로 터져
#:   전 모터가 무토크로 풀렸다(같은 세션 2회: 53s 자세유지 시작, 532.7s 착지 중).
#: 25° 의 근거: 실측 정적 처짐 12.6° × 2 = 25.2° → 하중이 순간 2배가 되는 충격까지 통과.
#:   기구 여유 충분 — 스탠바이 무릎 ∓38.5° 에서 25° 벗어나도 소프트 밴드
#:   (좌 [−83.5,+4.0] / 우 [−4.0,+80.0]) 안이고, 타깃은 ROM 클램프가 따로 막는다.
#: 완화이므로 조용히 넘기지 않는다: 종전 한계를 넘은 관절은 관절별 1회 경고를 남긴다.
#: 0929 승윤님 지시: 힙 앞뒤(1·7)도 25°. 근거 사건 — 0929 13:33:42 착지 중 몸이 비틀리며(요 −0.15 rad/s) 오른발이
#:   먼저 닿아 7번(오른 힙 앞뒤) 이 +17.1° 이탈 → 한계 15° 초과로 전 모터 릴리즈. 무릎처럼 정적 처짐 실측 근거는 없다
#:   (힙 정적 처짐 ~2°) — 착지 충격 허용을 위한 운영 결정. 힙 좌우(2·8)·회전(3·9)·발목은 15° 유지.
DEV_ABORT_DEG_BY_MOTOR = {4: 25.0, 10: 25.0,     # 4 = left_knee, 10 = right_knee
                          1: 25.0, 7: 25.0}      # 1 = left_hip_f, 7 = right_hip_f


def dev_limit_deg(mid):
    """모터별 이탈 한계 [deg]. 표에 없으면 공용 DEV_ABORT_DEG."""
    return DEV_ABORT_DEG_BY_MOTOR.get(mid, DEV_ABORT_DEG)

FB_TIMEOUT_S = 0.5
RAMP_RATE_DEG_S = 5.0
GAIN_RAMP_S = 8.0
GAIN_GATE_DEG = 3.0
SOFT_RELEASE_S = 3.0
TRIM_RATE_DEG_S = 0.5
TRIM_STALL_AIR_DEG = 2.5
TRIM_STALL_TICKS = 3
FAULT_ABORT_N = 2
TICK = 0.02

#: 트림 채널 — gain: 모터별 raw 부호 (좌우 부호 반대 규칙, 웅크림 실측
#: pose_record_20260801_crouch 로 검증: 힙굴곡 m1+/m7−, 무릎굽힘 m4+/m10−).
#: 트림은 ROM 클램프 대신 무부하 스톨가드로 보호 (발목과 동일 설계 —
#: ROM_REL 힙 상한은 웅크림 실측 +10.9°와 모순되어 신뢰 불가).
#: 트림 상한 확대 (0809 실측 근거). stand_v8 이 시뮬에서 학습한 평형 자세를
#: 트림 채널로 환산하면 **발목 +12.5 · 힙 +6.4 · 무릎 +10.4** 로 세 채널 모두
#: 기존 상한(12/6/8)을 넘는다. 곧은 다리(전 관절 0) 자세는 이 기계에서 CoM이
#: 지지면 뒤에 있어 정적으로 못 선다(정책 없이 kp150 고정으로 5초 뒤 −22.6°
#: 후방 낙하 실측). 웅크린 스탠바이가 균형 자세이고, 거기까지 도달하려면
#: 상한이 필요하다. 하한도 대칭으로 조금 넓힌다(과교정 탐색용).
TRIM_CH = {
    "a": {"file": TRIM_FILE, "gain": {5: 1.0, 11: -1.0}, "lo": -4.0, "hi": 16.0},
    "h": {"file": HIP_TRIM_FILE, "gain": {1: -1.0, 7: 1.0}, "lo": -4.0, "hi": 10.0},
    "k": {"file": KNEE_TRIM_FILE, "gain": {4: 1.0, 10: -1.0}, "lo": -2.0, "hi": 14.0},
}

SIG_STOP = []


def on_signal(signum, frame):
    SIG_STOP.append(signum)


def say(*args, **kw):
    """print 래퍼 — SSH 단절로 pty가 죽어도 (OSError) 제어 루프를 못 죽이게."""
    kw["flush"] = True
    try:
        print(*args, **kw)
    except OSError:
        pass


def tx(bus, mid, data):
    bus.send(can.Message(arbitration_id=mid, data=data, is_extended_id=False))


def release_all(bus):
    for _ in range(3):
        for m in IDS:
            try:
                tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass
        time.sleep(0.005)
    for m in IDS:
        try:
            tx(bus, m, EXIT)
        except Exception:
            pass


FB_TAU = {}     # 0919: 모터별 토크 추정(MIT 피드백, 부호=모터 프레임) — 상태파일 진단용(검증 전 참고값)
FB_VEL = {}


def drain(bus, pos, seen, fault, fault_cnt, fault_t):
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            return
        m = msg.data[0] if len(msg.data) == 8 else None
        if msg.is_error_frame or m not in pos:
            continue
        try:
            fb = decode_ak_mit_feedback(bytes(msg.data))
            pos[m] = fb.position_deg
            FB_TAU[m] = fb.torque_est; FB_VEL[m] = fb.velocity_rad_s
            seen[m] = time.monotonic()
            if not fb.error_ok:
                t = time.monotonic()
                # 1s 시간창 디바운스 — 무관한 단발 스파이크 2건의 누적 방지
                fault_cnt[m] = 1 if t - fault_t.get(m, -9.9) > 1.0 \
                    else fault_cnt.get(m, 0) + 1
                fault_t[m] = t
                fault[m] = fb.error_raw
        except Exception:
            pass


def load_stand_zero():
    try:
        hw = json.load(open(HW_MAP))
        zero = {int(j["motor_id"]): float(j["stand_zero_deg"])
                for j in hw["joints"].values()}
    except Exception as e:
        raise SystemExit(f"[중단] 하드웨어맵 읽기 실패 {HW_MAP}: {e} — "
                         f"boot_pose_check로 프레임 확정된 맵이 필요합니다")
    missing = [m for m in IDS if m not in zero]
    if missing:
        raise SystemExit(f"[중단] 하드웨어맵에 모터 {missing} stand_zero 없음")
    return zero


def live_ids(ids, fixed):
    """제어 대상 모터 = ids 에서 기구 고정 제외 (선언 없으면 그대로). 트림 채널 모터가 고정이면 중단."""
    for ch, cfg in TRIM_CH.items():
        hit = sorted(m for m in cfg["gain"] if m in fixed)
        if hit:
            raise SystemExit(f"[중단] 트림 채널 {ch} 의 모터 {hit} 가 기구 고정으로 선언됨 — "
                             f"이 조합은 지원하지 않습니다(하드웨어맵 mech_fixed 확인)")
    return [m for m in ids if m not in fixed]


def load_fixed_ids():
    """하드웨어맵의 기구 고정 모터 ID 집합 — 맵·선언 오류는 중단(추측하지 않는다)."""
    try:
        return set(load_fixed(HW_MAP)[1])
    except Exception as e:
        raise SystemExit(f"[중단] 하드웨어맵 기구 고정 선언 읽기 실패 {HW_MAP}: {e}")


def load_map_extras():
    """하드웨어맵에서 관절명→모터·direction. 12축 개별 조정(관절 프레임)을 모터 프레임으로 바꾸는 데 쓴다.
    joint_deg = direction × (motor − zero)  (stage8 규약) ⇒ motor_delta = direction × joint_delta."""
    hw = json.load(open(HW_MAP))
    motor_of, direction = {}, {}
    for name, j in hw["joints"].items():
        m = int(j["motor_id"]); d = int(j.get("direction", 1))
        if d not in (-1, 1):
            raise SystemExit(f"[중단] 하드웨어맵 {name} direction={d!r} (±1 만 허용)")
        motor_of[name] = m; direction[m] = d
    if len(motor_of) != 12 or len(direction) != 12:
        raise SystemExit("[중단] 하드웨어맵 관절 12개·모터 12개가 아님")
    return motor_of, direction


def load_joint_soft_limits(motor_of):
    """joint_limits_12dof.json 의 소프트 한계(관절 프레임, 좌우 대칭)를 {모터:(min,max)} 로. 없으면 None → ROM 표로 폴백."""
    for path in JOINT_LIMIT_FILES:
        try:
            raw = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        found = {}
        def walk(o):
            if isinstance(o, dict):
                if "soft_min" in o and "soft_max" in o:
                    return (float(o["soft_min"]), float(o["soft_max"]))
                for k, v in o.items():
                    r = walk(v)
                    if isinstance(r, tuple) and k in motor_of:
                        found[motor_of[k]] = r
            return None
        walk(raw)
        if len(found) == 12:
            return found
    return None


def read_air_gait_req(prev):
    """AIR_GAIT_FILE → {on, freq_hz, hip_deg, knee_deg, ankle_deg}. 읽기 실패 시 이전 값. 파라미터는 AIR_GAIT_LIM 으로 클램프."""
    try:
        with open(AIR_GAIT_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return prev
    if not isinstance(raw, dict):
        return prev
    out = dict(prev)
    out["on"] = bool(raw.get("on", False))
    for k, (lo, hi) in AIR_GAIT_LIM.items():
        try:
            v = float(raw.get(k, prev[k]))
        except (TypeError, ValueError):
            v = prev[k]
        if math.isfinite(v):
            out[k] = max(lo, min(hi, v))
    return out


def air_gait_joint_deg(phase, amp, p):
    """공중 걷기 패턴(관절 프레임 deg). 왼다리 위상 phase, 오른다리 phase+π.
    힙: 앞뒤 완전 스윙 h(φ)=a·sinφ+b·sin²φ (a=(F+E)/2, b=(F−E)/2 → 앞 최대 +F, 뒤 최대 −E, 무한 미분 가능=속도 연속).
    무릎·발목: 앞스윙 때만 접힘 sin²(양의 반주기). 부호(FK·수평자세 실측): 힙 굴곡 왼 +/오른 −, 무릎 굴곡 왼 −/오른 +, 발목 배굴 왼 +/오른 −."""
    F, E = p["hip_deg"], p.get("hip_ext_deg", 0.0)
    a, b = (F + E) / 2.0, (F - E) / 2.0
    def swing(ph):
        sn = math.sin(ph)
        return a * sn + b * sn * sn
    def lift(ph):
        sn = math.sin(ph)
        return sn * sn if sn > 0.0 else 0.0
    l_l, l_r = lift(phase), lift(phase + math.pi)
    return {
        "left_hip_f_joint": +amp * swing(phase), "right_hip_f_joint": -amp * swing(phase + math.pi),
        "left_knee_joint": -amp * p["knee_deg"] * l_l, "right_knee_joint": +amp * p["knee_deg"] * l_r,
        "left_ankle_f_joint": +amp * p["ankle_deg"] * l_l, "right_ankle_f_joint": -amp * p["ankle_deg"] * l_r,
    }


def joint_trim_req_motor(raw, motor_of, direction, lim=JOINT_TRIM_LIM_DEG):
    """콘솔 JSON({관절명: deg 관절 프레임}) → {모터: deg 모터 프레임}. 모르는 키·비수치·NaN 은 무시, |값|>lim 은 클램프."""
    out = {}
    if not isinstance(raw, dict):
        return out
    for name, v in raw.items():
        m = motor_of.get(name)
        if m is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        v = max(-lim, min(lim, v))
        out[m] = direction[m] * v
    return out


def ramp_toward(cur, lim, step):
    """한 틱 램프: cur 에서 lim 쪽으로 최대 step 만큼."""
    return min(cur + step, lim) if cur < lim else max(cur - step, lim)


def read_joint_trim_req(prev, motor_of, direction):
    """JOINT_TRIM_FILE 을 읽어 모터별 요청(deg 모터 프레임)으로. 읽기 실패 시 이전 값 유지."""
    try:
        with open(JOINT_TRIM_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return prev
    req = {m: 0.0 for m in IDS}
    req.update(joint_trim_req_motor(raw, motor_of, direction))
    return req


def write_status(path, payload):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass


def selftest():
    motor_of = {"left_ankle_r_joint": 6, "right_ankle_r_joint": 12, "left_knee_joint": 4}
    direction = {6: 1, 12: -1, 4: 1}
    r = joint_trim_req_motor({"left_ankle_r_joint": 3.0, "right_ankle_r_joint": 3.0, "left_knee_joint": 130.0,
                              "bogus": 1.0, "left_hip_f_joint": 2.0}, motor_of, direction)
    assert r == {6: 3.0, 12: -3.0, 4: 100.0}, r           # direction 적용·클램프(±100)·모르는 키 무시
    r2 = joint_trim_req_motor({"left_ankle_r_joint": "x", "right_ankle_r_joint": float("nan")}, motor_of, direction)
    assert r2 == {}, r2                                     # 비수치·NaN 무시
    assert joint_trim_req_motor("notadict", motor_of, direction) == {}
    assert ramp_toward(0.0, 3.0, 0.01) == 0.01 and ramp_toward(2.995, 3.0, 0.01) == 3.0
    assert ramp_toward(-1.0, 0.0, 0.5) == -0.5 and ramp_toward(0.2, -1.0, 0.5) == -0.3
    # 좌우 대칭 소프트 한계 클램프 산식: 관절 프레임 [jmin,jmax] → 모터 프레임 δ 구간 (direction 부호 처리)
    def motor_delta_bounds(direction, tgt, zero, jmin, jmax, margin=1.0):
        sj = direction * (tgt - zero)
        a = jmin + margin - sj; b = jmax - margin - sj
        c1, c2 = direction * a, direction * b
        return min(c1, c2), max(c1, c2)
    # 왼 무릎(dir 가정 +1, zero=tgt) soft[-83.5,4] → δ∈[-82.5,3]; 오른 무릎(dir −1) soft[-4,80] → δ∈[-79,3] (부호 뒤집혀 대칭)
    lo, hi = motor_delta_bounds(+1, 0.0, 0.0, -83.5, 4.0); assert abs(lo + 82.5) < 1e-9 and abs(hi - 3.0) < 1e-9, (lo, hi)
    lo, hi = motor_delta_bounds(-1, 0.0, 0.0, -4.0, 80.0); assert abs(lo + 79.0) < 1e-9 and abs(hi - 3.0) < 1e-9, (lo, hi)
    # 에어 보행 패턴: 위상 π/2 = 왼다리 최대 들림(힙+·무릎−·발목+), 오른다리 0; 3π/2 = 반대. 진폭 amp 스케일.
    P = {"freq_hz": 0.5, "hip_deg": 30.0, "hip_ext_deg": 12.0, "knee_deg": 30.0, "ankle_deg": 8.0}
    g = air_gait_joint_deg(math.pi / 2, 1.0, P)      # 왼다리 앞스윙 최대: 힙 +30, 무릎 −30, 발목 +8; 오른다리 뒤스윙 최대: 힙 +12(신전=오른쪽 +), 무릎 0
    assert abs(g["left_hip_f_joint"] - 30.0) < 1e-9 and abs(g["left_knee_joint"] + 30.0) < 1e-9 and abs(g["left_ankle_f_joint"] - 8.0) < 1e-9, g
    assert abs(g["right_hip_f_joint"] - 12.0) < 1e-9 and g["right_knee_joint"] == 0.0, g
    g = air_gait_joint_deg(3 * math.pi / 2, 0.5, P)  # 반대: 오른 앞스윙(−15 = 굴곡·amp 0.5), 왼 뒤스윙(−6)
    assert abs(g["right_hip_f_joint"] + 15.0) < 1e-9 and abs(g["right_knee_joint"] - 15.0) < 1e-9 and abs(g["left_hip_f_joint"] + 6.0) < 1e-9, g
    assert air_gait_joint_deg(0.0, 1.0, P)["left_hip_f_joint"] == 0.0     # 위상 0 = 양다리 0(연속)
    # 속도 연속성: 위상 0 근방에서 힙 기울기 급변 없음(수치 미분 좌우 차 < 5 %)
    d = 1e-4; f = lambda ph: air_gait_joint_deg(ph, 1.0, P)["left_hip_f_joint"]
    assert abs((f(d) - f(0)) - (f(0) - f(-d))) < 0.05 * abs(f(d) - f(0)), "힙 파형 기울기 불연속"
    r = read_air_gait_req(dict(AIR_GAIT_DEFAULT)); assert r["on"] is False and r["freq_hz"] == 0.5   # 파일 없음 → 이전값
    print("hang_stand selftest: joint_trim_req_motor · ramp_toward · 대칭 소프트한계 클램프 · 에어보행 패턴 — ALL PASS")
    return 0


def read_trim_req(ch, cur_req):
    cfg = TRIM_CH[ch]
    try:
        with open(cfg["file"]) as f:
            v = float(f.read().strip())
        return min(cfg["hi"], max(cfg["lo"], v))
    except Exception:
        return cur_req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=900.0,
                    help="유지 상한(초). 운용게인 상태면 초과해도 자동 릴리즈 안 함")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    # hang_handoff_state.json: 이전 세션의 낡은 핸드오프 잔재가 stage8에
    # 재소비되는 경로 차단 (최종점검 — ts 게이트 120s의 이중 방어)
    for f in (STOP_FILE, KILL_FILE, HANDOFF_FILE, GAIN_FILE, GROUND_FILE,
              TRIM_FILE, HIP_TRIM_FILE, KNEE_TRIM_FILE, UNFREEZE_FILE,
              JOINT_TRIM_FILE, STATUS_FILE, HOLD_KD_FILE, AIR_GAIT_FILE,
              "/tmp/hang_handoff_state.json",
              "/tmp/hang_handoff_state.json.tmp"):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                raise SystemExit(f"[중단] 트리거 파일 삭제 불가 {f}: {e} — "
                                 f"수동 삭제 후 재실행 (타 계정 소유?)")
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    zero = load_stand_zero()
    motor_of, direction = load_map_extras()
    # 기구 고정 모터는 이 뒤 모든 루프(ENTER·시작각 읽기·명령·가드·상태·핸드오프·릴리즈)에서 빠진다.
    global IDS
    fixed = load_fixed_ids()
    IDS = live_ids(IDS, fixed)
    if fixed:
        say(f"[기구 고정] {fixed_note(sorted(fixed))} (대상 {len(IDS)}모터)")
    fixed_names = [n for n, m in sorted(motor_of.items(), key=lambda kv: kv[1]) if m in fixed]
    joint_of = {m: n for n, m in motor_of.items()}
    soft_lim = load_joint_soft_limits(motor_of)   # 관절 프레임(좌우 대칭) 소프트 한계; None 이면 ROM 폴백
    if soft_lim is not None:
        say("[개별제어] 관절 한계표(좌우 대칭 소프트 한계) 로드 — ROM 표 대신 사용")
    else:
        say("[개별제어] 관절 한계표 없음 — ROM 표(좌우 비대칭)로 폴백")
    tgt = {m: min(max(zero[m], zero[m] + ROM_REL[m][0] + 3),
                  zero[m] + ROM_REL[m][1] - 3) for m in IDS}
    for m in IDS:
        if abs(tgt[m] - zero[m]) > 0.01:
            say(f"[주의] 모터{m} 스탠드 영점이 ROM 마진 밖 — "
                f"{zero[m]:+.2f}→{tgt[m]:+.2f} 클램프")

    try:
        bus = can.Bus(channel=a.channel, interface="socketcan")
    except Exception as e:
        raise SystemExit(f"[중단] CAN 열기 실패({a.channel}): {e} — can_up.sh 후 재실행")

    abort = None
    relaxed_noted = set()       # 관절별 '완화 구간 진입' 경고 1회용
    soft_stop = False
    handoff = False
    try:
        for m in IDS:
            tx(bus, m, ENTER)
            time.sleep(0.005)
        time.sleep(0.05)

        p0 = {}
        for m in IDS:
            reads = []
            for _ in range(5):
                tx(bus, m, pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0))
                t_e = time.monotonic() + 0.03
                while time.monotonic() < t_e:
                    r = bus.recv(timeout=0.03)
                    if r is not None and not r.is_error_frame \
                            and len(r.data) == 8 and r.data[0] == m:
                        reads.append(
                            decode_ak_mit_feedback(bytes(r.data)).position_deg)
                        break
                if len(reads) >= 3:
                    break
            if len(reads) < 3:
                say(f"[FAIL] 모터 {m} 시작각 읽기 실패 — 중단")
                return 1
            p0[m] = statistics.median(reads)
            if SIG_STOP:
                say("[중단] 시그널 수신 — 시작 전 종료")
                return 1

        big = [m for m in IDS if abs(tgt[m] - p0[m]) > 12.0]
        if big:
            say(f"[FAIL] 모터 {big} 시작각이 스탠드에서 12° 초과 이탈 — "
                f"매달림 직립 정렬 후 재실행 (boot_pose_check 먼저)")
            return 1

        dmax = max(abs(tgt[m] - p0[m]) for m in IDS)
        t_ramp = max(6.0, dmax / RAMP_RATE_DEG_S)
        say("모터별 이동량: " + " ".join(
            f"{m}:{tgt[m] - p0[m]:+.1f}" for m in IDS))
        say(f"[hang_stand] 에어램프 {t_ramp:.1f}s → 유지 (kp{KP_AIR:.0f} 시작)")

        pos = dict(p0)
        seen = {m: time.monotonic() for m in IDS}
        fault = {}
        fault_cnt = {}
        fault_t = {}
        fault_warned = set()
        kp, kd = KP_AIR, KD_AIR
        gain_t0 = None
        gain_skip_mtime = None      # 삭제 불가한 거부된 gain 파일 mtime (무시 목록)
        unfreeze_skip_mtime = None  # 삭제 불가한 unfreeze 파일 mtime (1회만 적용)
        handoff_skip_mtime = None   # 삭제 불가한 handoff 파일 mtime (재발화 금지)
        gain_done_recaptured = False  # 게인램프 완료 시 트림 베이스라인 1회 재캡처
        trim = {ch: {"req": 0.0, "cur": 0.0, "frozen": None, "fdir": 1,
                     "base": None,
                     "stall": {m: 0 for m in TRIM_CH[ch]["gain"]}}
                for ch in TRIM_CH}
        # 0914 12축 개별 제어기(콘솔): 관절 프레임 요청 → 모터 프레임, 운용 게인(kp150) 도달 뒤에만 램프
        jt = {m: {"req": 0.0, "cur": 0.0, "frozen": None, "fdir": 1, "base": None, "stall": 0} for m in IDS}
        jt_req_raw = {m: 0.0 for m in IDS}
        jt_notice_t = 0.0
        gait = dict(AIR_GAIT_DEFAULT)      # 0915 에어 보행 요청(파일)
        gait_phase, gait_amp, gait_notice_t = 0.0, 0.0, 0.0
        gait_off_m = {m: 0.0 for m in IDS}   # 모터 프레임 δ 오버레이
        t_status = 0.0
        grounded = False
        t0 = time.monotonic()
        t_show = t0
        t_prev = t0
        t_warn = 0.0
        t_warn_ground = 0.0

        def cmd_of(m, frac):
            c = p0[m] + frac * (tgt[m] - p0[m])
            for ch, cfg in TRIM_CH.items():
                g = cfg["gain"].get(m)
                if g:
                    c += g * trim[ch]["cur"]
            c += jt[m]["cur"] + gait_off_m[m]
            return c

        while True:
            now = time.monotonic()
            dt = min(0.1, now - t_prev)
            t_prev = now
            frac = min(1.0, (now - t0) / t_ramp)
            operational = kp > KP_AIR + 1e-6

            if now - t0 > a.hold:
                if operational:
                    if now - t_warn > 30.0:
                        t_warn = now
                        say("[경고] 유지 상한 초과 — 하중 가능성으로 자동 릴리즈 "
                            "안 함. stop/kill 파일로 종료하세요")
                else:
                    soft_stop = True
                    say("[시간초과] 저게인 상태 — 소프트 릴리즈")
                    break

            if not grounded and os.path.exists(GROUND_FILE):
                grounded = True
                say("[착지 선언] 스톨가드 해제 — 트림 라이브 조정 가능")
            if operational and not grounded and now - t_warn_ground > 30.0 \
                    and any(abs(t["cur"]) > 0.0 for t in trim.values()):
                t_warn_ground = now
                say("[안내] 착지했으면 touch /tmp/hang_ground — 선언 전엔 "
                    "스톨가드가 부하 오차를 하드스톱으로 오인할 수 있음")

            if (any(t["frozen"] is not None for t in trim.values())
                    or any(j["frozen"] is not None for j in jt.values())) \
                    and os.path.exists(UNFREEZE_FILE):
                try:
                    mt = os.path.getmtime(UNFREEZE_FILE)
                except OSError:
                    mt = None
                if unfreeze_skip_mtime is None or mt != unfreeze_skip_mtime:
                    try:
                        os.remove(UNFREEZE_FILE)
                    except OSError:
                        unfreeze_skip_mtime = mt
                    # cur≠0 상태라 0크로싱 재캡처가 안 옴 — 즉시 재캡처해
                    # 스톨가드를 재장전한 채로 트림 재조정 허용
                    for ch, t in trim.items():
                        if t["frozen"] is None:
                            continue
                        t["frozen"] = None
                        t["base"] = {m: pos[m] - cmd_of(m, frac)
                                     for m in TRIM_CH[ch]["gain"]}
                        t["stall"] = {m: 0 for m in TRIM_CH[ch]["gain"]}
                    for m, j in jt.items():
                        if j["frozen"] is not None:
                            j["frozen"] = None; j["base"] = pos[m] - cmd_of(m, frac); j["stall"] = 0
                    say("[동결 해제] 베이스라인 재캡처 — 트림 재조정 가능")

            if gain_t0 is None and frac >= 1.0 and os.path.exists(GAIN_FILE):
                try:
                    mt = os.path.getmtime(GAIN_FILE)
                except OSError:
                    mt = None
                if gain_skip_mtime is None or mt != gain_skip_mtime:
                    errs_now = {m: abs(pos[m] - cmd_of(m, frac)) for m in IDS}
                    bad = [m for m in IDS if errs_now[m] >= GAIN_GATE_DEG]
                    if bad:
                        try:
                            os.remove(GAIN_FILE)
                        except OSError:
                            gain_skip_mtime = mt
                        say(f"[거부] 게인램프 — 모터 {bad} 오차 {GAIN_GATE_DEG}° "
                            f"이상 (걸림/스톨 의심). 해소 후 다시 touch")
                    else:
                        gain_t0 = now
                        say("[게인램프] kp20/kd1 → kp150/kd5 (8s) — 완료 후 "
                            "줄을 천천히 풀어 착지시키세요")
            if gain_t0 is not None:
                g = min(1.0, (now - gain_t0) / GAIN_RAMP_S)
                kp = KP_AIR + g * (KP_MAX - KP_AIR)
                kd = KD_AIR + g * (KD_MAX - KD_AIR)
                if g >= 1.0 and not gain_done_recaptured:
                    # 게인램프 완료 시 활성 트림 채널 베이스라인 재캡처 —
                    # kp20에서 잡은 베이스라인 오차가 kp150 수렴으로 줄어들며
                    # 증분 오탐 동결을 일으키는 것 방지 (적대리뷰)
                    gain_done_recaptured = True
                    for ch, t in trim.items():
                        if t["base"] is not None:
                            t["base"] = {m: pos[m] - cmd_of(m, frac)
                                         for m in TRIM_CH[ch]["gain"]}
                            t["stall"] = {m: 0 for m in TRIM_CH[ch]["gain"]}

            if frac >= 1.0:
                step = TRIM_RATE_DEG_S * dt
                for ch, t in trim.items():
                    t["req"] = read_trim_req(ch, t["req"])
                    lim = t["req"]
                    if t["frozen"] is not None:
                        # 동결 당시 진행 방향(fdir) 쪽만 캡 — 0 방향 물러남은 허용
                        lim = min(lim, t["frozen"]) if t["fdir"] > 0 \
                            else max(lim, t["frozen"])
                    cur = t["cur"]
                    new = min(cur + step, lim) if cur < lim \
                        else max(cur - step, lim)
                    crossed = (new == 0.0) or (new * cur < 0.0)
                    if crossed:
                        t["base"] = None    # 0 복귀/부호 전환 — 베이스라인 리셋
                        t["stall"] = {m: 0 for m in TRIM_CH[ch]["gain"]}
                    if t["base"] is None and new != 0.0 \
                            and (crossed or cur == 0.0):
                        # 0 이탈/크로싱에서 베이스라인 캡처 (구 트림 기준 오차)
                        t["base"] = {m: pos[m] - cmd_of(m, frac)
                                     for m in TRIM_CH[ch]["gain"]}
                        t["stall"] = {m: 0 for m in TRIM_CH[ch]["gain"]}
                    t["cur"] = new
                # ── 0914 12축 개별 조정: 운용 게인 완료(kp150) 뒤에만 움직인다 ──
                jt_req_raw = read_joint_trim_req(jt_req_raw, motor_of, direction)
                jt_on = gain_t0 is not None and kp >= KP_MAX - 1e-6
                if not jt_on:
                    if any(abs(v) > 0.0 for v in jt_req_raw.values()) and now - jt_notice_t > 10.0:
                        jt_notice_t = now
                        say("[개별제어] 운용 게인(kp150) 도달 전 — 요청은 보류(0 유지)")
                jstep = JOINT_TRIM_RATE_DEG_S * dt
                for m, j in jt.items():
                    # 요청은 모터 프레임(δ). 관절 프레임 = direction·(motor−zero). 좌우 대칭 소프트 한계로 클램프.
                    want = jt_req_raw.get(m, 0.0) if jt_on else 0.0
                    if soft_lim is not None:
                        sj = direction[m] * (tgt[m] - zero[m])          # 스탠드 목표의 관절각(관절 프레임)
                        jmin, jmax = soft_lim[m]
                        djo_lo = jmin + JOINT_LIMIT_MARGIN_DEG - sj      # 허용 관절 δ 하한
                        djo_hi = jmax - JOINT_LIMIT_MARGIN_DEG - sj      # 허용 관절 δ 상한
                        cm1, cm2 = direction[m] * djo_lo, direction[m] * djo_hi   # 모터 프레임 δ (direction=±1, 부호 뒤집힘 처리)
                        lo, hi = min(cm1, cm2), max(cm1, cm2)
                    else:
                        lo = zero[m] + ROM_REL[m][0] + 3.0 - tgt[m]
                        hi = zero[m] + ROM_REL[m][1] - 3.0 - tgt[m]
                    want = max(lo, min(hi, want))
                    j["lo"], j["hi"] = lo, hi
                    j["req"] = want
                    lim = want
                    if j["frozen"] is not None:
                        lim = min(lim, j["frozen"]) if j["fdir"] > 0 else max(lim, j["frozen"])
                    cur = j["cur"]
                    new = ramp_toward(cur, lim, jstep)
                    crossed = (new == 0.0) or (new * cur < 0.0)
                    if crossed:
                        j["base"] = None; j["stall"] = 0
                    if j["base"] is None and new != 0.0 and (crossed or cur == 0.0):
                        j["base"] = pos[m] - cmd_of(m, frac); j["stall"] = 0
                    j["cur"] = new
                # ── 0915 에어 보행(사진용): 운용 게인·미접지에서만, 진폭 2 s 페이드, 개별 조정과 합산 뒤 소프트 한계 클램프 ──
                gait = read_air_gait_req(gait)
                gait_ok = jt_on and not grounded
                if gait["on"] and not gait_ok and now - gait_notice_t > 10.0:
                    gait_notice_t = now
                    say("[에어보행] 보류 — 운용 게인(kp150)·미접지 상태에서만 켜집니다")
                gait_amp = ramp_toward(gait_amp, 1.0 if (gait["on"] and gait_ok) else 0.0, dt / AIR_GAIT_FADE_S)
                if gait_amp > 0.0:
                    gait_phase = (gait_phase + 2.0 * math.pi * gait["freq_hz"] * dt) % (2.0 * math.pi)
                    gj = air_gait_joint_deg(gait_phase, gait_amp, gait)
                else:
                    gait_phase = 0.0
                    gj = {}
                for m in IDS:
                    gm = direction[m] * gj.get(joint_of[m], 0.0)
                    lo_m, hi_m = jt[m].get("lo", -1e9), jt[m].get("hi", 1e9)
                    gait_off_m[m] = max(lo_m - jt[m]["cur"], min(hi_m - jt[m]["cur"], gm))

            kd_send = kd
            if kp >= KP_MAX - 1e-6:                    # 운용 게인 도달 후에만 kd 라이브 조절 허용(에어/램프는 원래대로)
                try:
                    with open(HOLD_KD_FILE) as _f:
                        kd_send = min(30.0, max(1.0, float(_f.read().strip())))
                except Exception:
                    kd_send = kd
            for m in IDS:
                try:
                    tx(bus, m, pack_ak_mit_command(
                        math.radians(cmd_of(m, frac)), 0.0, kp, kd_send, 0.0))
                except Exception as e:
                    abort = f"송신실패 {type(e).__name__}"
                    break
            if abort:
                break
            drain(bus, pos, seen, fault, fault_cnt, fault_t)

            if not grounded:
                for ch, t in trim.items():
                    if t["frozen"] is not None or t["base"] is None \
                            or abs(t["cur"]) <= 0.3:
                        continue
                    for m in TRIM_CH[ch]["gain"]:
                        inc = abs((pos[m] - cmd_of(m, frac)) - t["base"][m])
                        t["stall"][m] = t["stall"][m] + 1 \
                            if inc > TRIM_STALL_AIR_DEG else 0
                        if t["stall"][m] >= TRIM_STALL_TICKS:
                            fz = t["cur"] - math.copysign(1.0, t["cur"])
                            t["frozen"] = 0.0 if fz * t["cur"] < 0.0 else fz
                            t["fdir"] = 1 if t["cur"] > 0.0 else -1
                            say(f"[프로브] {ch}트림 모터{m} 오차 증분 {inc:+.1f}°"
                                f" — 하드스톱 의심, {t['frozen']:+.1f}°로 동결 "
                                f"(오탐이면 touch {UNFREEZE_FILE})")
                            break

            # 0914: 12축 개별 조정(jt)은 공중에서 동결하지 않는다 — 빠른 램프(30°/s)로 공중 추종오차가
            #        스톨 임계(2.5°)를 넘어 오동결하던 문제(오른발목앞뒤 −6 고착). 실제 하드스톱 보호는
            #        DEV_ABORT(15°)가 담당. 채널 트림(a/h/k)의 공중 스톨가드는 종전대로 유지(위 블록).
            hard_fault = [m for m, c in fault_cnt.items() if c >= FAULT_ABORT_N]
            if hard_fault:
                abort = "모터 폴트 " + " ".join(
                    f"{m}:0x{fault[m]:X}" for m in sorted(hard_fault))
                break
            for m in fault:
                if m not in fault_warned:
                    fault_warned.add(m)
                    say(f"[경고] 모터{m} 폴트 1회 관측 (0x{fault[m]:X}) — "
                        f"{FAULT_ABORT_N}회 시 릴리즈")
            for m in IDS:
                _dev = pos[m] - cmd_of(m, frac)
                _lim = dev_limit_deg(m)
                if abs(_dev) > _lim:
                    abort = f"모터{m} 이탈 {_dev:+.1f}° (한계 {_lim:g}°)"
                    break
                if abs(_dev) > DEV_ABORT_DEG and m not in relaxed_noted:
                    # 완화 구간 진입 = 종전이면 여기서 풀렸을 지점. 조용히 넘기지 않는다.
                    relaxed_noted.add(m)
                    say(f"[주의] 모터{m} 이탈 {_dev:+.1f}° — 종전 한계 {DEV_ABORT_DEG:g}° 초과, "
                        f"관절별 한계 {_lim:g}° 로 계속 진행")
                if now - seen[m] > FB_TIMEOUT_S:
                    abort = f"모터{m} 피드백 두절"
                    break
            if len(SIG_STOP) >= 2:
                abort = "시그널 2회 — 즉시 릴리즈"
            if abort or os.path.exists(KILL_FILE):
                abort = abort or "비상 정지(kill)"
                break
            if os.path.exists(HANDOFF_FILE):
                try:
                    mt = os.path.getmtime(HANDOFF_FILE)
                except OSError:
                    mt = None
                if handoff_skip_mtime is None or mt != handoff_skip_mtime:
                    try:
                        os.remove(HANDOFF_FILE)
                    except OSError:
                        # 삭제 불가(sudo touch 등) — 같은 mtime은 재발화 금지
                        # (적대리뷰: 묵은 트리거가 조건 성립 순간 무단 인계)
                        handoff_skip_mtime = mt
                    # 운용게인 '완료'(kp=KP_MAX)에서만 인계 — 램프 중 kp21~149
                    # 인계 시 가드 없는 저게인 홀드로 주저앉음 (적대리뷰)
                    # + 최종점검 핸드셰이크: stage8 --wait-takeover 가 살아서
                    # 대기 중(마커 3s 이내 갱신)일 때만 — 수신자 없는 핸드오프는
                    # 상태 파일만 남기고 종료 → 모터 자체 타임아웃 릴리즈 사고.
                    # 적대검수 0803: 창 3s → 0.5s. 마커는 50Hz로 갱신되므로
                    # 0.5s면 25회분 여유 — 3s는 stage8 사망 후에도 최대 3초간
                    # 핸드오프를 승인하는 창이었다 (모터 타임아웃 0.3s보다 김).
                    s8_waiting = False
                    try:
                        s8_waiting = (time.time() - os.path.getmtime(
                            "/tmp/stage8_takeover_waiting")) < 0.5
                    except OSError:
                        pass
                    if grounded and kp >= KP_MAX - 1e-6 and s8_waiting:
                        handoff = True
                        break
                    if not s8_waiting:
                        say("[핸드오프 거부] stage8 선기동 대기 미검출 — 먼저 "
                            "stage8을 --wait-takeover로 띄운 뒤 다시 시도")
                    else:
                        say("[핸드오프 거부] 접지+운용게인(kp150 도달) 상태에서만 "
                            "인계 가능 — 공중/램프중 인계 금지 (0802 발진 사고)")
            if os.path.exists(STOP_FILE) or SIG_STOP:
                soft_stop = True
                if SIG_STOP:
                    say(f"[시그널 {SIG_STOP[0]}] 소프트 정지 경로")
                break

            if now - t_status >= 0.5:
                t_status = now
                _phase = ("에어램프" if frac < 1.0 else
                          "게인램프" if gain_t0 and kp < KP_MAX - 1e-6 else
                          "에어유지" if gain_t0 is None else
                          "하중유지" if grounded else "운용대기")
                _st = {
                    "ts": time.time(), "phase": _phase, "kp": kp, "kd": kd_send if "kd_send" in dir() else kd, "grounded": grounded,
                    "joint_trim_enabled": bool(gain_t0 is not None and kp >= KP_MAX - 1e-6),
                    "trim": {ch: trim[ch]["cur"] for ch in trim},
                    # 관절 프레임(deg): joint = direction × motor
                    "joint_trim": {joint_of[m]: {"req": direction[m] * jt[m]["req"], "cur": direction[m] * jt[m]["cur"],
                                                 "frozen": jt[m]["frozen"] is not None} for m in IDS},
                    "joint_pos_deg": {joint_of[m]: direction[m] * (pos[m] - zero[m]) for m in IDS},
                    "joint_cmd_deg": {joint_of[m]: direction[m] * (cmd_of(m, frac) - zero[m]) for m in IDS},
                    "err_deg": {joint_of[m]: pos[m] - cmd_of(m, frac) for m in IDS},
                    "joint_tau_est": {joint_of[m]: FB_TAU.get(m) for m in IDS},          # 0919 진단용(모터 프레임, 미검증)
                    "joint_vel_rad_s": {joint_of[m]: FB_VEL.get(m) for m in IDS},
                    "gait": {"on": gait["on"], "active": gait_amp > 0.0, "amp": round(gait_amp, 3),
                             "freq_hz": gait["freq_hz"], "hip_deg": gait["hip_deg"], "knee_deg": gait["knee_deg"], "ankle_deg": gait["ankle_deg"]}}
                if fixed_names:
                    # 기구 고정 관절: 관절 0° 에 굳어 있다 — 조정 요청은 따르지 않으므로 req·cur·위치·명령 모두 0 으로 적고 표지를 단다
                    # (콘솔의 수평 자세 확인이 12 관절을 본다: 기준이 0° 가 아니면 거기서 불일치로 드러난다). 오차·토크·속도는 싣지 않는다.
                    _st["mech_fixed"] = list(fixed_names)
                    for _n in fixed_names:
                        _st["joint_trim"][_n] = {"req": 0.0, "cur": 0.0, "frozen": False, "mech_fixed": True}
                        _st["joint_pos_deg"][_n] = 0.0
                        _st["joint_cmd_deg"][_n] = 0.0
                write_status(STATUS_FILE, _st)
            if now - t_show >= 1.0:
                t_show = now
                errs = {m: pos[m] - cmd_of(m, frac) for m in IDS}
                worst = max(errs, key=lambda k: abs(errs[k]))
                phase = ("에어램프" if frac < 1.0 else
                         "게인램프" if gain_t0 and kp < KP_MAX - 1e-6 else
                         "에어유지" if gain_t0 is None else
                         "하중유지" if grounded else "운용대기")
                jt_txt = " ".join(f"{joint_of[m].replace('_joint','')}{direction[m]*jt[m]['cur']:+.1f}"
                                  for m in IDS if abs(jt[m]["cur"]) > 0.05)
                say(f"[{phase} {now - t0:5.1f}s] kp{kp:5.1f} 트림 "
                    f"a{trim['a']['cur']:.2f}/h{trim['h']['cur']:.2f}/"
                    f"k{trim['k']['cur']:.2f}° "
                    f"| 최대오차 모터{worst} {errs[worst]:+.2f}° | 평균 "
                    f"{statistics.mean(abs(v) for v in errs.values()):.2f}°"
                    + (f" | 개별 {jt_txt}" if jt_txt else "")
                    + (f" | 보행 {gait['freq_hz']:.2f}Hz amp{gait_amp:.2f}" if gait_amp > 0.0 else ""))
            time.sleep(max(0.0, TICK - (time.monotonic() - now)))

        if handoff:
            # 인수 홀드 상태 파일 — stage8이 첫 틱부터 같은 명령을 스트리밍해
            # 무토크 공백 제로로 이어받게 함 (ENTER 폴링의 명령 리셋 실측 대응)
            try:
                state = {"ts": time.time(), "motors": {
                    str(m): {"deg": cmd_of(m, frac), "kp": kp, "kd": kd}
                    for m in IDS}}
                tmp = "/tmp/hang_handoff_state.json.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(state, f)
                os.replace(tmp, "/tmp/hang_handoff_state.json")
                say("[핸드오프] 상태 파일 기록 → stage8 인수 홀드 준비 완료")
                say("[핸드오프] 릴리즈 생략 — 모터가 현 자세·운용게인을 그대로 "
                    "잡고 있습니다. 즉시 진행: stage8 기동(인수 홀드 자동) → "
                    "SET_BASELINE_FROM_CURRENT_ALL → ARM_ALL → 정책. "
                    "지연·실패 시 회수 = 줄 인장 후 release_all.py")
                return 3
            except OSError as e:
                # 적대검수 0803: 기록 실패인 채 종료하면 --wait-takeover
                # stage8은 영원히 침묵(상태파일 없이는 송신 0) → 모터 자체
                # 타임아웃(~0.3s 브레이크) 하중 낙하 확정. 핸드오프를 취소하고
                # 접지 소프트 릴리즈 경로로 전환한다 (줄 인장 후 재시도).
                say(f"[핸드오프] ⚠ 상태 파일 기록 실패({e}) — 핸드오프 취소, "
                    "접지 소프트 램프다운 경로로 전환. 줄 확인!")
                abort = f"핸드오프 상태파일 기록 실패({e})"
                handoff = False
        if abort:
            if grounded and not os.path.exists(KILL_FILE):
                # 착지(줄 느슨) 상태의 가드 abort는 즉시 무토크가 오히려 위험
                # (12kg 붕괴→자유낙하 회생→SMPS 트립, 적대리뷰) — 명령 동결 +
                # 2s 게인 램프다운을 최선노력으로 시도 (버스 사망이면 send가
                # 조용히 실패해 사실상 즉시 릴리즈와 동일)
                say(f"[ABORT] {abort} — 접지 상태: 소프트 램프다운 시도 후 릴리즈")
                frozen = {m: cmd_of(m, frac) for m in IDS}
                g0 = time.monotonic()
                while time.monotonic() - g0 < 2.0:
                    if os.path.exists(KILL_FILE) or len(SIG_STOP) >= 2:
                        break
                    g = 1.0 - (time.monotonic() - g0) / 2.0
                    for m in IDS:
                        try:
                            tx(bus, m, pack_ak_mit_command(
                                math.radians(frozen[m]), 0.0,
                                max(2.0, kp * g), max(0.3, kd * g), 0.0))
                        except Exception:
                            pass
                    time.sleep(TICK)
            else:
                say(f"[ABORT] {abort} — 즉시 릴리즈 (줄이 받습니다)")
            return 2
        if soft_stop:
            # 명령 동결 + 게인만 램프다운: 위치오차 성분 토크가 스텝 소실 없이
            # 게인에 비례해 서서히 빠짐. kill·폴트·추가 시그널은 계속 감시.
            say("[소프트정지] 명령 동결, 게인 3s 램프다운 — 줄 확인!")
            frozen = {m: cmd_of(m, min(1.0, (time.monotonic() - t0) / t_ramp))
                      for m in IDS}
            kp0, kd0 = kp, kd
            g0 = time.monotonic()
            while time.monotonic() - g0 < SOFT_RELEASE_S:
                if os.path.exists(KILL_FILE) or len(SIG_STOP) >= 2 \
                        or any(c >= FAULT_ABORT_N for c in fault_cnt.values()):
                    say("[소프트정지 중단] 즉시 릴리즈")
                    break
                g = 1.0 - (time.monotonic() - g0) / SOFT_RELEASE_S
                for m in IDS:
                    try:
                        tx(bus, m, pack_ak_mit_command(
                            math.radians(frozen[m]), 0.0,
                            max(2.0, kp0 * g), max(0.3, kd0 * g), 0.0))
                    except Exception:
                        pass
                drain(bus, pos, seen, fault, fault_cnt, fault_t)
                time.sleep(TICK)
            say("[완료] 릴리즈")
        return 0
    finally:
        if not handoff:
            release_all(bus)


if __name__ == "__main__":
    raise SystemExit(main())
