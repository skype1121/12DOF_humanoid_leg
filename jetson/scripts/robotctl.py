#!/usr/bin/env python3
"""robotctl — ssh 터미널에서 운영 콘솔(8800) API 로 로봇을 조작한다 (0915, 사진 촬영·무테더 운용용).

표준 절차(1008 — 공중에서 인계):  robotctl prep  →  robotctl level  →  robotctl policy-on  →  (지그 내림 — 아래)  →  걷기는 콘솔 패널
  시험 방식 air(1008 오후 기본값 — 남긴 정책이 공중 정책 시작을 받는다): policy-on 이 끝나면 정책이 **공중에서 이미 구동 중** → 지그를 천천히 내린다
                               (공중 ~ 착지는 공중·착지 정책이 맡고, 착지가 가라앉으면 브리지가 서기 KE 로 넘긴다 — 걷기는 그 뒤)
  시험 방식 ground: policy-on 뒤 화면 '착지 대기'(정책 미구동·수평 자세 유지) → 지그를 내리면 양발 착지 확인 뒤 정책이 시작한다

모든 명령은 콘솔 서버의 선행조건·허용 규칙을 그대로 통과한다(우회 없음). 버튼 대신 타이핑일 뿐이다.

  robotctl status            현재 상태 한 줄
  robotctl config air|ground [CKPT]   시험 방식·정책 설정 (제어 시작 전). 1008: 고를 수 있는 정책은 합성 키 하나(기본값) — 그 밖의 키는 서버가 거부.
                             1008 오후: 남긴 정책(종전 합성 키 + 공중·착지 구간 정책 G21h)은 air 를 다시 받는다.
                               air    = 공중 정책 시작(hang_arm): 정책이 공중에서 바로 구동된 채 내린다(기본값 · 착지 정착 뒤 서기 KE 로 자동 인계)
                               ground = 착지 자동 인수(touchdown start): 공중에서 인계하되 정책은 양발 착지 확인 뒤 시작
                             공중 정책 시작을 받지 않는 정책(지면 시작 학습 · 시뮬 AIR1 착지 실패 다수)에서는 air 를 서버가 거부한다.
  robotctl watch             1초마다 상태 (Ctrl-C 로 종료)
  robotctl motors            12모터 CAN 응답 일괄 점검(0토크 폴링, ~/rl_calib/ping_all.py) — 제어 스택이 살아 있으면 거부
  robotctl joints            12관절 실제각·명령·오차·신선도 표 (읽기 전용)
  robotctl prep [-y] [--no-hang]   CAN 준비 → 센서 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 → 공중 자세 유지 → 운용 게인(kp150)
                             (완료된 단계는 건너뜀. 1008: hang 을 흡수 — 모터가 움직이기 직전에 한 번 묻는다 ※ 줄이 로봇을 지지해야 함.
                              -y = 묻지 않음 · --no-hang = 모터를 움직이지 않는 준비 단계까지만)
  robotctl zero-fix          영점 배수 보정(전원 재투입 뒤 영점 확인이 '창 어긋남' 이면) → 다시 prep
  robotctl sensors           센서 기동/재기동(IMU·발 어댑터; 데이터가 신선하면 그대로, 멈췄으면 PID파일 소유 노드 정리 후 재기동)
  robotctl codes             프로세스 종료 코드 표(뜻·할 일)
  robotctl reset             강제 초기화: 가드 릴리즈·모터 전원 OFF 로 토크 해제 단계가 안 끝날 때(무토크 확인 뒤). 제어 프로세스 살아 있으면 거부
  robotctl zero-recap        영점 재캡처(지그 기준 자세에서, 시프트 0인데 잔차만 남았을 때; 12관절 앵커를 지금 읽기값으로 — 맵 자동 백업) → 다시 prep
  robotctl hang [-y]         (1008: prep 에 흡수됨) 공중 자세 유지 → 운용 게인(kp150) 단계만 따로 실행  ※ 줄이 로봇을 지지해야 함
  robotctl level             수평 기준 자세(어제 캡처: 힙앞뒤 ±20.5·무릎 ±30·발목앞뒤 ±19)
  robotctl zero              에어 보행 정지 + 개별 조정 전부 0 = 매달림 기준(제로) 자세
  robotctl walk [-y] [--f 0.5 --hip 30 --hipext 12 --knee 30 --ankle 8]   공중 걷기(힙 앞뒤 스윙) 시작 (공중·kp150 에서만)
  robotctl stop              에어 보행 정지(자세는 유지)
  robotctl kd N              홀드 감쇠 kd (1~30, 기본 5)
  robotctl autolevel on [--target 0]   접지 뒤 IMU 기준 자동 수평(힙앞뒤 트림, 0.5°/s, ±10°)
  robotctl autolevel off
  robotctl trim JOINT DEG    관절 하나 개별 조정 (예: trim left_knee -25)
  robotctl ground            접지 확인 (발을 바닥에 대고 서게 할 때만)
  robotctl handoff [-y]      제어 인계 → 기준점 → ARM → 정책 로드 (선택 정책은 콘솔 설정; 자동 수평은 먼저 끔)
  robotctl stand [-y]        정책 시작 (인계·ARM·정책 로드 뒤)
                             ※ 위 세 명령은 단계별 절차다. 1008 오후부터 기본 방식이 air 라, **지면에서 서게 하려면 먼저 robotctl config ground**.
                               방식 air 에서는 ground = 공중 인계 조건 확인(접촉 검사 없음), stand = 정책이 공중에서 바로 구동(hang_arm) — 확인 문구가 그렇게 묻는다.
  robotctl policy-on [-y]    [1005 단순화] 공중 수평 자세(hang·level 뒤)에서 한 번: 인계→기준점→ARM→정책 로드→(방식별 마지막 단계)를
                             콘솔이 기존 게이트로 진행. 방식 ground: 착지 대기 — 정책은 양발 착지 확인 뒤에만 시작 → 그 다음 지그 해제
                             방식 air(1008 오후): **정책이 공중에서 바로 구동**(다리가 움직일 수 있음) → 구동 확인 뒤 지그를 천천히 내림
  robotctl down [-y]         회수: 줄 지지 확인 → 제어 프로세스 종료 → 토크 해제
  robotctl estop             정지 요청
  robotctl new               새 시험 준비 (회수·토크 해제 뒤)
  robotctl raw ACTION [k=v]  임의 콘솔 액션
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8800"
JOINTS = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
          "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
          "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def die(msg, code=2):
    print(f"✗ {msg}", flush=True)
    sys.exit(code)


def state():
    try:
        with urllib.request.urlopen(BASE + "/api/state", timeout=4) as r:
            return json.load(r)
    except Exception as exc:
        die(f"콘솔 서버(127.0.0.1:8800) 응답 없음: {exc} — systemctl status operator-console.service")


def act(action, extra=None, confirmed=True, quiet=False):
    """콘솔 액션 1회. 서버가 거부하면 이유를 출력하고 False."""
    st = state()
    allowed = st["trial"]["allowed"].get(action)
    if allowed is not None and not allowed.get("enabled") and action != "estop":
        if not quiet:
            print(f"✗ {allowed.get('label', action)}: {allowed.get('reason')}", flush=True)
        return False
    body = {"action": action, "boot_id": st["trial"]["boot_id"], "revision": st["trial"]["revision"],
            "request_id": str(uuid.uuid4())}
    if confirmed:
        body["confirmed"] = True
    body.update(extra or {})
    req = urllib.request.Request(BASE + "/api/action", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=110 if action == "can_up" else 15) as r:
            res = json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            res = json.load(exc)
        except Exception:
            res = {"ok": False, "message": f"HTTP {exc.code}"}
    except Exception as exc:
        die(f"요청 실패: {exc}")
    mark = "✓" if res.get("ok") else "✗"
    if not quiet or not res.get("ok"):
        print(f"{mark} {res.get('message')}", flush=True)
    return bool(res.get("ok"))


def line(st):
    # 센서 기동 직후엔 imu/foot/bridge 가 None 일 수 있다(0915 prep 크래시) — 전부 dict 로 정규화
    t = st.get("trial") or {}
    b = st.get("bridge") or {}
    im = st.get("imu") or {}
    ft = st.get("foot") or {}
    hh = st.get("hang_hold") or {}
    g = hh.get("gait") or {}
    jt = hh.get("joint_trim") or {}
    trims = " ".join(f"{k.replace('_joint', '')}{v['cur']:+.0f}" for k, v in jt.items() if abs(v.get("cur", 0)) > 0.5)
    hold = f"{hh.get('phase')} kp{hh.get('kp', 0):.0f} kd{hh.get('kd', 0):.0f}" if hh else "-"
    c = ft.get("contact") or [None, None]
    parts = [f"[{t.get('phase')}]", f"홀드 {hold}",
             f"fwd {im.get('fwd', 0):+.1f} lat {im.get('lat', 0):+.1f}",
             f"접촉 L{int(bool(c[0]))} R{int(bool(c[1]))}"]
    if trims:
        parts.append(f"조정 {trims}")
    if g.get("active"):
        parts.append(f"보행 {g.get('freq_hz')}Hz amp{g.get('amp')}")
    al = st.get("auto_level") or {}
    if al.get("on"):
        parts.append(f"자동수평 {al.get('status')} off{al.get('offset', 0):+.1f}")
    if b.get("live"):
        parts.append(f"정책 LIVE {b.get('loaded_checkpoint')}")
    if (b.get("kew3") or {}).get("mode") == "air":      # 1008 오후(표시 전용): 공중 정책 시작 세션의 공중·착지 구간 — 착지가 가라앉으면 사라진다(서기 KE 인계)
        parts.append("공중·착지 정책 구간(KE 인계 전·걷기 불가)")
    sg = b.get("start_gate") or {}
    if sg.get("state") and sg.get("state") != "POLICY":
        parts.append(f"착지 대기({sg.get('state')}: {sg.get('why')})")
    if t.get("macro"):
        parts.append(f"정책 ON 진행 {(t['macro'] or {}).get('step')}")
    if t.get("fault"):
        parts.append(f"⚠ {t['fault']}")
    if t.get("next_action"):
        parts.append(f"다음: {(t.get('allowed') or {}).get(t['next_action'], {}).get('label', t['next_action'])}")
    return " | ".join(parts)


def wait_until(pred, timeout, what):
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        st = state()
        if pred(st):
            return st
        if st["trial"].get("fault"):
            die(f"{what} 중 정지: {st['trial']['fault']}")
        cur = line(st)
        if cur != last:
            print("   … " + cur, flush=True)
            last = cur
        time.sleep(1.0)
    die(f"{what} 시간 초과({timeout:.0f}s)")


def wait_completed(action, timeout):
    def pred(st):
        t = st["trial"]
        lr = t.get("last_result") or {}
        if lr.get("action") == action and lr.get("ok") is False:
            die(f"{t['allowed'].get(action, {}).get('label', action)} 실패: {lr.get('message')}")
        return action in t.get("completed", []) and not t.get("pending")
    return wait_until(pred, timeout, action)


def confirm(prompt, yes):
    if yes:
        return True
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_prep(a):
    steps = [("can_up", 100), ("sensors", 30), ("boot_pose", 60), ("selftest", 180), ("stage8", 40)]
    for action, tmo in steps:
        st = state()
        if action in st["trial"].get("completed", []):
            print(f"· {st['trial']['allowed'][action]['label']}: 이미 완료", flush=True)
            continue
        print(f"▶ {st['trial']['allowed'][action]['label']}", flush=True)
        if not act(action):
            if action == "boot_pose":
                print("   → 영점이 '창 어긋남'이면:  robotctl zero-fix   후 다시  robotctl prep", flush=True)
            sys.exit(1)
        try:
            wait_completed(action, tmo)
        except SystemExit:
            if action == "boot_pose":
                print("   → 영점이 '창 어긋남'이면:  robotctl zero-fix   후 다시  robotctl prep  (지그로 기준 자세 정렬 확인)", flush=True)
            raise
    if getattr(a, "no_hang", False):
        print("✓ 준비 완료 (--no-hang: 공중 자세 유지는 시작하지 않음) → 이어서 하려면 robotctl prep", flush=True)
        return
    if not hang_phase(getattr(a, "yes", False)):
        print("· 준비 단계까지만 완료 — 공중 자세 유지는 시작하지 않았습니다. 이어서 하려면 다시 robotctl prep", flush=True)
        sys.exit(1)
    print("✓ 준비 + 공중 자세 유지 완료(운용 게인) → robotctl level → robotctl policy-on", flush=True)


def cmd_zero_fix(a):
    if not confirm("영점 배수 보정(영점 쓰기). 로봇이 기준 자세(지그/직립 정렬)인가요?", a.yes):
        die("취소")
    if act("zero_multiple"):
        wait_completed("zero_multiple", 60)
        print("✓ 배수 보정 완료 → robotctl prep (영점 확인부터 이어감)", flush=True)


def cmd_sensors(a):
    st = state(); fr = st.get("fresh") or {}; im = st.get("imu") or {}
    print(f"센서 신선도 imu {fr.get('imu')} foot {fr.get('foot')} (IMU 마지막 값 {im.get('age', '?')} s 전)", flush=True)
    if act("sensors"):
        wait_completed("sensors", 60)
        st = state(); fr = st.get("fresh") or {}
        print(f"✓ 센서 imu {fr.get('imu')} foot {fr.get('foot')}", flush=True)


def cmd_codes(a):
    sys.path.insert(0, "/home/mama/humanoid_repo/jetson/scripts")
    from console_state import RC_HINTS
    print("종료 코드 0 = 정상, 3 = 정상 핸드오프 종료(자세유지). 그 외:")
    for key, m in RC_HINTS.items():
        for rc, (meaning, todo) in sorted(m.items()):
            print(f"  {key:<15} {rc}: {meaning}\n{'':19}→ {todo}")


def cmd_reset(a):
    if not confirm("강제 초기화: 모터에 토크가 없음(전원 꺼짐/가드 릴리즈)을 확인했나요? 살아 있는 제어 프로세스가 있으면 거부됩니다.", a.yes):
        die("취소")
    act("air_gait_off", quiet=True)
    st = state(); alive = [k for k, v in (st.get("procs") or {}).items() if v.get("alive") and k in ("stage8", "hang_stand", "bridge")]
    if alive:
        print(f"제어 프로세스 {alive} 살아 있음 → 지지 확인 + 제어 종료 먼저", flush=True)
        act("support_confirm")
        if act("stack_down"):
            wait_completed("stack_down", 30)
    if act("force_reset"):
        print("✓ 단계 초기화 완료(토크 해제 단계 생략 기록) → robotctl prep", flush=True)


def cmd_zero_recap(a):
    if not confirm("영점 재캡처(가장 위험한 쓰기): 지그로 12관절이 기준 자세에 정확히 있나요? 시프트가 남아 있으면 스크립트가 거부합니다.", a.yes):
        die("취소")
    if act("zero_recapture"):
        wait_completed("zero_recapture", 60)
        print("✓ 재캡처 완료 → robotctl prep (영점 확인 잔차 0 근처인지 확인)", flush=True)


def hang_phase(yes):
    """공중 자세 유지 시작 → 운용 게인(kp150). prep 의 마지막 단계이자 hang 명령의 본문(1008: hang 을 prep 에 흡수).

    콘솔 서버의 게이트·완료 판정을 그대로 쓴다(우회 없음). 이미 끝난 단계는 건너뛰고, 모터가 움직이기 전에 한 번 묻는다(-y 면 생략).
    반환: True = 운용 게인 도달(또는 이미 완료), False = 운영자가 확인에서 취소(아무것도 보내지 않음)."""
    done = state()["trial"].get("completed", [])
    if "hang_stand" in done and "gain_ramp" in done:
        print("· 공중 자세 유지·운용 게인: 이미 완료", flush=True)
        return True
    if not confirm("줄이 로봇을 지지하고 다리 주변에 사람이 없나요? 모터가 공중 기준 자세로 움직입니다.", yes):
        return False
    if "hang_stand" not in done:
        print("▶ 공중 자세 유지 시작", flush=True)
        if not act("hang_stand"):
            sys.exit(1)
        wait_until(lambda st: (st.get("hang_hold") or {}).get("phase") == "에어유지", 60, "에어램프")
    print("▶ 운용 게인 kp150", flush=True)
    if not act("gain_ramp"):
        sys.exit(1)
    wait_until(lambda st: (st.get("hang_hold") or {}).get("kp", 0) >= 150, 30, "게인램프")
    return True


def cmd_hang(a):
    """1008: prep 에 흡수됨 — 남겨 둔 것은 준비 단계가 이미 끝난 상태에서 공중 자세 유지만 다시 걸 때를 위해서다(prep 도 같은 일을 한다)."""
    if not hang_phase(a.yes):
        die("취소")
    print("✓ 운용 게인 도달 → level / walk / zero / kd / trim 가능", flush=True)


def cmd_level(a):
    act("air_gait_off", quiet=True)
    act("joint_trim_preset")


def cmd_zero(a):
    act("air_gait_off", quiet=True)
    act("joint_trim_zero")


def cmd_walk(a):
    if not confirm("에어 보행: 발이 바닥에서 떨어져 있고 줄이 지지하나요?", a.yes):
        die("취소")
    extra = {}
    if a.f is not None:
        extra["freq_hz"] = a.f
    if a.hip is not None:
        extra["hip_deg"] = a.hip
    if a.hipext is not None:
        extra["hip_ext_deg"] = a.hipext
    if a.knee is not None:
        extra["knee_deg"] = a.knee
    if a.ankle is not None:
        extra["ankle_deg"] = a.ankle
    act("air_gait_on", extra)


def cmd_stop(a):
    act("air_gait_off")


def cmd_autolevel(a):
    if a.mode == "on":
        act("auto_level_on", {"target": a.target})
    else:
        act("auto_level_off")


def cmd_kd(a):
    act("hold_kd", {"value": a.value})


def cmd_trim(a):
    j = a.joint if a.joint.endswith("_joint") else a.joint + "_joint"
    if j not in JOINTS:
        die(f"관절 이름 오류: {a.joint} (예: left_knee, right_hip_f, left_ankle_f)")
    act("joint_trim", {"joint": j, "value": a.deg})


def cmd_down(a):
    if not confirm("회수: 줄이 로봇을 받치고 있나요? (제어 종료 → 토크 해제)", a.yes):
        die("취소")
    act("air_gait_off", quiet=True)
    act("support_confirm")
    if act("stack_down"):
        wait_completed("stack_down", 30)
    if act("release"):
        wait_completed("release", 30)
    print("✓ 회수 완료 → 새 시험은 robotctl new", flush=True)


def cmd_handoff(a):
    st = state()
    if (st.get("auto_level") or {}).get("on"):
        act("auto_level_off", quiet=True)
    im = st.get("imu") or {}
    air = st["trial"].get("mode") == "air"      # 1008 오후: 방식 air 의 단계별 절차 = 공중 인계(뒤의 stand 가 공중에서 정책을 구동한다)
    print(f"현재 fwd {im.get('fwd', 0):+.1f}° lat {im.get('lat', 0):+.1f}° | 선택 정책 {st['trial'].get('checkpoint')}"
          + (" | 방식 air(공중 정책 시작)" if air else ""), flush=True)
    if air:
        if not confirm("제어 인계(공중 방식): 줄(지그)이 로봇을 지지하고 발이 바닥에서 떨어져 있으며 수평 자세(level)가 적용됐나요? "
                       "(지면에 세운 채 인계하려면 취소하고 robotctl config ground 먼저)", a.yes):
            die("취소")
    elif not confirm("제어 인계: 발이 바닥에 있고 줄이 느슨하며 몸통이 거의 수평인가요?", a.yes):
        die("취소")
    for action, tmo in (("handoff", 20), ("baseline", 20), ("arm", 20), ("bridge", 60)):
        print(f"▶ {state()['trial']['allowed'][action]['label']}", flush=True)
        if not act(action):
            sys.exit(1)
        wait_completed(action, tmo)
    print("✓ 인계·ARM·정책 로드 완료 → robotctl stand", flush=True)


def cmd_stand(a):
    st = state()
    if st["trial"].get("mode") == "air":         # 1008 오후: 방식 air 의 stand = 공중 정책 시작(hang_arm) — 줄이 받친 채, 다리 주변에 사람 없이
        if not confirm(f"정책 시작(공중 방식 · {st['trial'].get('checkpoint')}): 정책이 공중에서 바로 구동됩니다(다리가 움직일 수 있음). "
                       "줄(지그)이 로봇을 지지하고 다리 주변에 사람이 없나요?", a.yes):
            die("취소")
    elif not confirm(f"정책 시작({st['trial'].get('checkpoint')}): 줄 느슨·손 떼기 확인?", a.yes):
        die("취소")
    if act("stand"):
        wait_completed("stand", 15)
        print("✓ 정책 제어 중 — 'robotctl watch' 로 관찰, 문제 시 robotctl estop", flush=True)


def cmd_policy_on(a):
    """1005 단순화 기동: 서버 순서기가 기존 단계를 같은 게이트로 실행 — 클라이언트는 요청 1회 + 관찰만.
    1008 오후: 콘솔의 시험 방식(state()["trial"]["mode"])에 따라 마지막 단계가 다르다 — 확인 문구·완료 안내가 그 방식을 따른다.
      ground(접지 방식): 착지 자동 인수 — 정책은 양발 착지 확인 뒤에만 시작한다(종전 문구 그대로).
      air(공중 방식)   : 정책이 **공중에서 바로** 구동된다(stand + hang_arm). 착지를 기다리지 않으므로 다리가 움직일 수 있다."""
    st = state()
    im = st.get("imu") or {}
    air = st["trial"].get("mode") == "air"
    print(f"현재 fwd {im.get('fwd', 0):+.1f}° lat {im.get('lat', 0):+.1f}° | 선택 정책 {st['trial'].get('checkpoint')}"
          + (" | 방식 air(공중 정책 시작)" if air else ""), flush=True)
    if air:
        if not confirm("정책 ON(공중 방식): 줄(지그)이 로봇을 지지하고 발이 바닥에서 떨어져 있으며 수평 자세(level)가 적용됐고 "
                       "다리 주변에 사람이 없나요? 정책이 공중에서 바로 구동됩니다(착지를 기다리지 않음 — 다리가 움직일 수 있습니다)", a.yes):
            die("취소")
    elif not confirm("정책 ON: 줄(지그)이 로봇을 지지하고 발이 바닥에서 떨어져 있으며 수평 자세(level)가 적용됐나요? "
                     "정책은 착지 확인 뒤에만 시작합니다", a.yes):
        die("취소")
    if not act("policy_on"):
        sys.exit(1)

    def done(st):
        t = st["trial"]
        if "policy_on" in t.get("completed", []):
            return True
        if not t.get("macro") and t.get("fault"):
            die(f"정책 ON 중단: {t['fault']}")
        return False
    wait_until(done, 180, "정책 ON")
    if air:
        print("✓ 정책이 공중에서 구동 중 — 이제 지그를 천천히 내리세요(착지가 가라앉으면 서기 KE 로 자동 인계 — 걷기는 그 뒤에). "
              "'robotctl watch' 로 관찰, 문제 시 robotctl estop", flush=True)
        return
    print("✓ 착지 대기 — 정책 미구동·수평 자세 유지 중. 이제 지그를 해제하세요(양발 착지 0.3 s·정착 뒤 정책 자동 시작). "
          "'robotctl watch' 로 관찰, 문제 시 robotctl estop", flush=True)


def cmd_config(a):
    st = state()
    ck = a.ckpt or st["trial"].get("checkpoint")
    act("configure", {"mode": a.mode, "checkpoint": ck}, confirmed=False)


def cmd_motors(a):
    import subprocess
    st = state(); alive = [k for k, v in (st.get("procs") or {}).items() if v.get("alive") and k in ("stage8", "hang_stand", "bridge")]
    ext = st.get("external_controls") or []
    if alive or ext:
        print(f"✗ 제어 프로세스가 살아 있어 CAN 폴링 불가({alive or ext}) — 회수(robotctl down) 뒤 실행. 대신 관절 표:", flush=True)
        return cmd_joints(a)
    r = subprocess.run([sys.executable, "/home/mama/rl_calib/ping_all.py", "--channel", "can1"], capture_output=True, text=True)
    print((r.stdout or "") + (r.stderr or ""), flush=True)
    # 0918: 영점 대비 차이 표 (읽기 전용 미리보기, 쓰기 없음) — 안 맞을 때 무엇을 바꿀지 조치 열로 안내
    z = subprocess.run([sys.executable, "/home/mama/humanoid_repo/jetson/scripts/zero_anchor.py", "--preview", "--json"],
                       capture_output=True, text=True)
    rep = None
    for ln in (z.stdout or "").splitlines():
        if ln.startswith("JSON "):
            try:
                rep = json.loads(ln[5:])
            except ValueError:
                rep = None
    if rep is None:
        print("영점 대비 표 없음(zero_anchor --preview 실패):", (z.stdout or z.stderr or "").strip()[-300:], flush=True)
        return r.returncode
    print("\n영점 대비 (읽기 − 현앵커 = 창 k×36°/10° + 잔차; 허용 AK70 8°/AK45 3.5°)", flush=True)
    print(f"{'모터':<4}{'관절':<20}{'읽기°':>8}{'앵커°':>8}{'차이°':>8}{'창k':>4}{'잔차°':>7}  {'판정':<8} 조치", flush=True)
    todo = set()
    for row in rep["rows"]:
        kind = row["kind"]; k = row.get("k"); res = row.get("residual"); rd = row.get("reading"); sz = row.get("stand_zero")
        diff = (rd - sz) if (rd is not None and sz is not None) else None
        if kind == "일치":
            act_ = "-"
        elif kind == "프레임시프트":
            act_ = f"창 어긋남 {k:+d}×{row['window']:g}° → robotctl zero-fix"; todo.add("zero-fix")
        elif kind == "자세불일치":
            act_ = (f"이 관절을 기준 자세로 {-res:+.1f}° 움직여 맞추기(지그) → 다시 motors; 지그에서 맞는데도 남으면 zero-recap"
                    if res is not None else "기준 자세로 맞추기"); todo.add("posture")
        elif kind == "무응답":
            act_ = "CAN 무응답 → 커넥터·전원·허브 접촉"; todo.add("bad")
        else:
            act_ = f"{kind} → 전원 재투입 뒤 다시 motors"; todo.add("bad")
        f = lambda v, w=8: (f"{v:{w}.2f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"{row['motor']:<4}{row['joint']:<20}{f(rd)}{f(sz)}{f(diff)}{(f'{k:+d}' if isinstance(k, int) else '-'):>4}{f(res, 7)}  {kind:<8} {act_}", flush=True)
    if "bad" in todo:
        print("→ 먼저 무응답/폴트 모터 해결(영점 판정 불가)", flush=True)
    elif "zero-fix" in todo:
        print("→ 조치: robotctl zero-fix (창 어긋남만 보정, 앵커 유지) → 다시 motors", flush=True)
    elif "posture" in todo:
        print("→ 조치: 자세불일치 관절을 지그/손으로 기준 자세에 맞춘 뒤 다시 motors. 지그에 정확히 있는데도 남으면 robotctl zero-recap(12관절 앵커 재캡처)", flush=True)
    else:
        print("→ 12/12 영점 일치 — 조치 없음", flush=True)
    return r.returncode


def cmd_joints(a):
    st = state(); hh = st.get("hang_hold") or {}; s8 = st.get("stage8") or {}
    src = "hang_stand" if hh.get("joint_pos_deg") else "stage8"
    print(f"[{(st.get('trial') or {}).get('phase')}] 출처 {src} | fwd {(st.get('imu') or {}).get('fwd', 0):+.1f} lat {(st.get('imu') or {}).get('lat', 0):+.1f} | 접촉 {(st.get('foot') or {}).get('contact')}", flush=True)
    print(f"{'관절':22s} {'실제°':>8s} {'명령°':>8s} {'오차°':>7s}  비고")
    for j in JOINTS:
        if src == "hang_stand":
            q = hh["joint_pos_deg"].get(j); c = (hh.get("joint_cmd_deg") or {}).get(j); e = (hh.get("err_deg") or {}).get(j)
            note = ("동결 " if ((hh.get("joint_trim") or {}).get(j) or {}).get("frozen") else "")
        else:
            x = (s8.get("joints") or {}).get(j) or {}
            q = x.get("latest_actual_deg"); c = x.get("commanded_relative_deg"); e = (q - c) if (q is not None and c is not None) else None
            note = ("STALE " if x.get("feedback_stale") else "") + (f"err0x{x.get('feedback_error_raw'):X} " if x.get("feedback_error_raw") else "")
        f = lambda v: f"{v:+8.1f}" if isinstance(v, (int, float)) else f"{'—':>8s}"
        print(f"{j.replace('_joint',''):22s} {f(q)} {f(c)} {f(e)[1:] if e is not None else '     —'}  {note}", flush=True)


def cmd_status(a):
    print(line(state()), flush=True)


def cmd_watch(a):
    try:
        while True:
            print(time.strftime("%H:%M:%S") + " " + line(state()), flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def cmd_raw(a):
    extra = {}
    for kv in a.kv:
        k, _, v = kv.partition("=")
        try:
            extra[k] = json.loads(v)
        except Exception:
            extra[k] = v
    act(a.action, extra)


def main():
    ap = argparse.ArgumentParser(prog="robotctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("config"); s.add_argument("mode", choices=("air", "ground")); s.add_argument("ckpt", nargs="?", default=None); s.set_defaults(fn=cmd_config)
    sub.add_parser("watch").set_defaults(fn=cmd_watch)
    sub.add_parser("motors").set_defaults(fn=cmd_motors)
    sub.add_parser("joints").set_defaults(fn=cmd_joints)
    s = sub.add_parser("prep"); s.add_argument("-y", dest="yes", action="store_true")
    s.add_argument("--no-hang", dest="no_hang", action="store_true", help="모터를 움직이지 않는 준비 단계까지만(공중 자세 유지 생략)")
    s.set_defaults(fn=cmd_prep)
    s = sub.add_parser("zero-fix"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_zero_fix)
    sub.add_parser("sensors").set_defaults(fn=cmd_sensors)
    sub.add_parser("codes").set_defaults(fn=cmd_codes)
    s = sub.add_parser("reset"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_reset)
    s = sub.add_parser("zero-recap"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_zero_recap)
    s = sub.add_parser("hang"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_hang)
    sub.add_parser("level").set_defaults(fn=cmd_level)
    sub.add_parser("zero").set_defaults(fn=cmd_zero)
    s = sub.add_parser("walk"); s.add_argument("-y", dest="yes", action="store_true")
    s.add_argument("--f", type=float, default=None, help="Hz 0.2~1.0"); s.add_argument("--hip", type=float, default=None, help="앞스윙 deg 0~40")
    s.add_argument("--hipext", type=float, default=None, help="뒤스윙 deg 0~25")
    s.add_argument("--knee", type=float, default=None, help="deg 0~45"); s.add_argument("--ankle", type=float, default=None, help="deg 0~15")
    s.set_defaults(fn=cmd_walk)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    s = sub.add_parser("kd"); s.add_argument("value", type=float); s.set_defaults(fn=cmd_kd)
    s = sub.add_parser("autolevel"); s.add_argument("mode", choices=("on", "off")); s.add_argument("--target", type=float, default=0.0); s.set_defaults(fn=cmd_autolevel)
    s = sub.add_parser("trim"); s.add_argument("joint"); s.add_argument("deg", type=float); s.set_defaults(fn=cmd_trim)
    sub.add_parser("ground").set_defaults(fn=lambda a: act("ground"))
    s = sub.add_parser("handoff"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_handoff)
    s = sub.add_parser("stand"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_stand)
    s = sub.add_parser("policy-on"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_policy_on)
    s = sub.add_parser("down"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_down)
    sub.add_parser("estop").set_defaults(fn=lambda a: act("estop", confirmed=False))
    sub.add_parser("new").set_defaults(fn=lambda a: act("new_trial"))
    s = sub.add_parser("raw"); s.add_argument("action"); s.add_argument("kv", nargs="*"); s.set_defaults(fn=cmd_raw)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
