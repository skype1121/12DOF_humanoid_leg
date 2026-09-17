#!/usr/bin/env python3
"""robotctl — ssh 터미널에서 운영 콘솔(8800) API 로 로봇을 조작한다 (0915, 사진 촬영·무테더 운용용).

모든 명령은 콘솔 서버의 선행조건·허용 규칙을 그대로 통과한다(우회 없음). 버튼 대신 타이핑일 뿐이다.

  robotctl status            현재 상태 한 줄
  robotctl config air|ground [CKPT]   시험 방식·정책 설정 (제어 시작 전; air = 접촉 게이트 대신 HangGuard — 왼발 패드 탈락 상태 권장)
  robotctl watch             1초마다 상태 (Ctrl-C 로 종료)
  robotctl prep              CAN 준비 → 센서 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 (완료된 단계는 건너뜀)
  robotctl zero-fix          영점 배수 보정(전원 재투입 뒤 영점 확인이 '창 어긋남' 이면) → 다시 prep
  robotctl hang [-y]         공중 자세 유지 시작 → 운용 게인(kp150)  ※ 줄이 로봇을 지지해야 함
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
    print("✓ 준비 완료 → robotctl hang", flush=True)


def cmd_zero_fix(a):
    if not confirm("영점 배수 보정(영점 쓰기). 로봇이 기준 자세(지그/직립 정렬)인가요?", a.yes):
        die("취소")
    if act("zero_multiple"):
        wait_completed("zero_multiple", 60)
        print("✓ 배수 보정 완료 → robotctl prep (영점 확인부터 이어감)", flush=True)


def cmd_hang(a):
    if not confirm("줄이 로봇을 지지하고 다리 주변에 사람이 없나요? 모터가 공중 기준 자세로 움직입니다.", a.yes):
        die("취소")
    if not act("hang_stand"):
        sys.exit(1)
    wait_until(lambda st: (st.get("hang_hold") or {}).get("phase") == "에어유지", 60, "에어램프")
    print("▶ 운용 게인 kp150", flush=True)
    if not act("gain_ramp"):
        sys.exit(1)
    wait_until(lambda st: (st.get("hang_hold") or {}).get("kp", 0) >= 150, 30, "게인램프")
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
    print(f"현재 fwd {im.get('fwd', 0):+.1f}° lat {im.get('lat', 0):+.1f}° | 선택 정책 {st['trial'].get('checkpoint')}", flush=True)
    if not confirm("제어 인계: 발이 바닥에 있고 줄이 느슨하며 몸통이 거의 수평인가요?", a.yes):
        die("취소")
    for action, tmo in (("handoff", 20), ("baseline", 20), ("arm", 20), ("bridge", 60)):
        print(f"▶ {state()['trial']['allowed'][action]['label']}", flush=True)
        if not act(action):
            sys.exit(1)
        wait_completed(action, tmo)
    print("✓ 인계·ARM·정책 로드 완료 → robotctl stand", flush=True)


def cmd_stand(a):
    st = state()
    if not confirm(f"정책 시작({st['trial'].get('checkpoint')}): 줄 느슨·손 떼기 확인?", a.yes):
        die("취소")
    if act("stand"):
        wait_completed("stand", 15)
        print("✓ 정책 제어 중 — 'robotctl watch' 로 관찰, 문제 시 robotctl estop", flush=True)


def cmd_config(a):
    st = state()
    ck = a.ckpt or st["trial"].get("checkpoint")
    act("configure", {"mode": a.mode, "checkpoint": ck}, confirmed=False)


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
    sub.add_parser("prep").set_defaults(fn=cmd_prep)
    s = sub.add_parser("zero-fix"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_zero_fix)
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
    s = sub.add_parser("down"); s.add_argument("-y", dest="yes", action="store_true"); s.set_defaults(fn=cmd_down)
    sub.add_parser("estop").set_defaults(fn=lambda a: act("estop", confirmed=False))
    sub.add_parser("new").set_defaults(fn=lambda a: act("new_trial"))
    s = sub.add_parser("raw"); s.add_argument("action"); s.add_argument("kv", nargs="*"); s.set_defaults(fn=cmd_raw)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
