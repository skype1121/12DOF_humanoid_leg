"""전체 릴리즈 — 현재각 홀드에서 게인 램프다운 후 MIT 해제 (estop 프로토콜 최종 단계).

⚠ 반드시 줄(카라비너)이 로봇을 받는 상태에서 실행할 것.
   서 있는 로봇에 쏘면 (부드럽게) 주저앉는다.

v2.1 (2026-08-02, 적대리뷰 반영):
  - 인터리브 홀드: 모터별로 '읽기 성공 즉시' kp12 홀드 1발을 쏘고 다음 모터로
    넘어감 — 종전 일괄 방식은 버스 반죽음(뒷번호 재시도 소진) 시 앞번호가
    ~1초 무토크로 방치되어, 이 도구가 막으려던 자유낙하 회생 트립을 재현했음.
  - 시그널(SIGINT/SIGTERM/SIGHUP) → 즉시 최종 시퀀스로 점프. 최종 시퀀스
    (제로게인 3연발 + MIT 해제)는 finally에 있어 어떤 경로로 죽어도 실행됨 —
    종전엔 램프 중 Ctrl-C면 모터가 kp6 홀드를 문 채 MIT 래치로 방치됐음.
  - 원리 유지: 실측 현재각 홀드(점프 없음) → 2s 게인 램프다운(회생 미미)
    → 제로게인+EXIT. --fast = 즉시 릴리즈 (버스 사망 직전 비상 전용).

v2.2 (2026-09-06, 배포 계약 감사 A3):
  - 회수 순서 게이트: stage8 노드·hang_stand·브리지(--live) 프로세스가 살아 있으면
    거부(종료코드 3). CLAUDE.md "줄 인장 → 스택 종료 → 그 다음에만 release_all" 이
    코드로 강제되지 않아 같은 버스에 상반 명령이 겹칠 수 있었다. 탐지만 하고(pgrep)
    종료는 절대 하지 않는다 — 콘솔이 띄운 것은 콘솔이 내린다. --force '<사유>' 로만 우회.
  - CAN·디코더 import 를 main() 안으로 — --selftest 는 CAN 없는 PC 에서도 돈다.

v2.3 (2026-10-09, 기구 고정):
  - 하드웨어맵에 "mech_fixed": true 로 선언된 모터는 대상에서 빼고 "모터 N 기구 고정 — 제외" 로 알린다.
    판정(RELEASE_VERDICT·종료코드)은 살아 있는 모터만으로 낸다. 맵·선언을 못 읽으면 경고하고
    **12모터 전부** 대상으로 진행한다 — 회수 도구는 어떤 경우에도 실행을 거부하지 않는다.

사용: python3 release_all.py [--channel can1] [--fast] [--force '<사유>'] | --selftest
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")

ENTER = bytes([0xFF] * 7 + [0xFC])
EXIT = bytes([0xFF] * 7 + [0xFD])
IDS = list(range(1, 13))
HW_MAP = "/home/mama/rl_calib/robot_12dof_hardware_map.json"
KP_SOFT = 12.0     # 홀드 시작 게인 — 스윙만 잡을 저강성 (하중 지지 불가 수준)
KD_SOFT = 0.8
RAMP_S = 2.0
TICK = 0.02

#: 회수 순서 게이트 — 이 패턴(pgrep -f, 정규식)의 프로세스가 살아 있으면 릴리즈 거부.
STACK_PATTERNS = (
    ("stage8 노드", r"stage8_12axis_mit_control_node"),
    ("hang_stand", r"hang_stand\.py"),
    ("브리지 live", r"rl_bridge_node\.py.*--live"),
)
GATE_EXIT_CODE = 3

SIG = []


def on_signal(signum, frame):
    SIG.append(signum)


def stack_alive(patterns=STACK_PATTERNS):
    """살아 있는 스택 프로세스 [(라벨, [pid, ...]), ...] — 없으면 []. 탐지 전용(종료 없음)."""
    me, parent = os.getpid(), os.getppid()
    found = []
    for label, pat in patterns:
        r = subprocess.run(["pgrep", "-f", "--", pat], capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
        pids = [p for p in pids if p not in (me, parent)]
        if pids:
            found.append((label, pids))
    return found


def gate_or_die(force_reason=None):
    """스택 생존 시 거부(SystemExit 3). --force '<사유>' 면 경고만 찍고 진행."""
    alive = stack_alive()
    if not alive:
        return []
    lines = "; ".join(f"{lab} pid {','.join(map(str, pids))}" for lab, pids in alive)
    if force_reason:
        print(f"[release_all] ⚠ 스택 생존 상태에서 --force 진행 (사유: {force_reason}): {lines}")
        return alive
    print(f"[release_all] 거부 — 스택이 살아 있음: {lines}")
    print("  회수 순서: ① 줄 인장 → ② bash ~/rl_calib/stack_down.sh (콘솔: '스택 종료') → ③ release_all")
    print("  같은 버스에 상반 명령이 겹치면 회생 과전압으로 SMPS 가 죽는다. 비상이면 --force '<사유>'")
    raise SystemExit(GATE_EXIT_CODE)


def live_ids(path=None, ids=None):
    """(릴리즈 대상 모터, 제외된 기구 고정 모터). 맵·선언 읽기 실패 = 경고 후 전부 대상(회수는 막지 않는다)."""
    ids = list(IDS if ids is None else ids)
    try:
        from mech_fixed import load_fixed
        fixed = set(load_fixed(path or HW_MAP)[1])
    except Exception as e:  # noqa: BLE001 — 어떤 실패도 릴리즈를 막지 않는다
        print(f"[release_all] ⚠ 기구 고정 선언 읽기 실패({type(e).__name__}: {e}) — 12모터 전부 대상으로 진행")
        return ids, []
    return [m for m in ids if m not in fixed], [m for m in ids if m in fixed]


def selftest():
    """CAN·ROS 불필요 — 가짜 stage8 프로세스(exec -a 위장)로 게이트 거부·통과 검증.
    종료는 내가 띄운 명시 PID 만 (pkill 금지 원칙)."""
    pre = stack_alive()
    if pre:
        print(f"[selftest] 실스택 생존 중 — '빈 상태' 검사 생략: {pre}")
    else:
        assert gate_or_die(None) == [], "빈 상태에서 오탐"
    p = subprocess.Popen(["bash", "-c", "exec -a stage8_12axis_mit_control_node sleep 30"])
    try:
        time.sleep(0.3)
        got = stack_alive()
        assert any(lab == "stage8 노드" and p.pid in pids for lab, pids in got), f"가짜 stage8 미탐지: {got}"
        try:
            gate_or_die(None)
        except SystemExit as e:
            assert e.code == GATE_EXIT_CODE, e.code
        else:
            raise AssertionError("스택 생존인데 게이트가 통과시킴")
        forced = gate_or_die("selftest 강제")
        assert forced, "--force 경로가 진행하지 않음"
    finally:
        p.terminate()          # 명시 PID — 내가 띄운 가짜 프로세스만
        p.wait(timeout=5)
    time.sleep(0.2)
    assert not any(p.pid in pids for _, pids in stack_alive()), "가짜 프로세스 잔존"
    print("[selftest] release_all 게이트: 빈상태 통과 · 가짜 stage8 거부(exit 3) · --force 통과 — ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--fast", action="store_true",
                    help="즉시 제로게인 (구버전) — 버스 사망 직전 비상 전용. "
                         "관절이 영점에서 멀면 자유낙하 스윙→회생으로 SMPS 트립 위험")
    ap.add_argument("--force", metavar="사유", default=None,
                    help="스택 생존 게이트 우회(사유 기록 필수) — 버스 사망 직전 비상 전용")
    ap.add_argument("--selftest", action="store_true", help="CAN 없이 게이트 검증만")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    gate_or_die(a.force)   # 스택 종료 선행 강제 (v2.2)
    global IDS
    IDS, skipped = live_ids()
    if skipped:
        print(f"[release_all] 모터 {skipped} 기구 고정 — 제외 (대상 {len(IDS)}모터)")

    import can  # noqa: E402 — 게이트 통과 후에만 버스를 연다
    from ak_mit_command import pack_ak_mit_command  # noqa: E402
    from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402
    ZERO = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    bus = can.Bus(channel=a.channel, interface="socketcan")
    fail = 0

    def tx(m, data):
        bus.send(can.Message(arbitration_id=m, data=data, is_extended_id=False))

    def hold(m, p, g):
        tx(m, pack_ak_mit_command(math.radians(p), 0.0,
                                  KP_SOFT * g, KD_SOFT * g, 0.0))

    try:
        if not a.fast:
            # 인터리브 읽기+홀드: 읽힌 모터는 그 자리에서 즉시 kp12 홀드 점유
            for m in IDS:
                try:
                    tx(m, ENTER)
                except Exception:
                    pass
            time.sleep(0.02)
            pos = {}
            for m in IDS:
                if SIG:
                    break
                got = None
                for _ in range(3):
                    try:
                        tx(m, ZERO)
                    except Exception:
                        break
                    t_end = time.monotonic() + 0.03
                    while time.monotonic() < t_end:
                        r = bus.recv(timeout=0.03)
                        if r is not None and not r.is_error_frame \
                                and len(r.data) == 8 and r.data[0] == m:
                            got = decode_ak_mit_feedback(
                                bytes(r.data)).position_deg
                            break
                    if got is not None:
                        break
                if got is not None:
                    pos[m] = got
                    try:
                        hold(m, got, 1.0)   # 즉시 홀드 — 무토크 공백 최소화
                    except Exception:
                        pass
                # 이미 읽힌 모터들 홀드 재송신 (읽기 지연 동안 명령 신선 유지)
                for pm, pp in pos.items():
                    try:
                        hold(pm, pp, 1.0)
                    except Exception:
                        pass
            missing = sorted(set(IDS) - set(pos))
            if missing:
                print(f"[release_all] 모터 {missing} 읽기 실패 — 해당 모터는 "
                      f"즉시 제로게인 (아마 무전원)")
            if pos and not SIG:
                print(f"[release_all] {len(pos)}모터 현재각 홀드 → "
                      f"{RAMP_S:.0f}s 게인 램프다운")
                t0 = time.monotonic()
                while not SIG:
                    g = 1.0 - (time.monotonic() - t0) / RAMP_S
                    if g <= 0.0:
                        break
                    for m, p in pos.items():
                        try:
                            hold(m, p, g)
                        except Exception:
                            fail += 1
                    time.sleep(TICK)
            if SIG:
                print(f"[release_all] 시그널 {SIG[0]} — 최종 시퀀스로 점프")
        return 0
    finally:
        # 최종 시퀀스는 어떤 경로(정상/시그널/예외)로도 반드시 실행
        for _ in range(3):
            for m in IDS:
                try:
                    tx(m, ZERO)
                except Exception:
                    fail += 1
            time.sleep(0.005)
        for m in IDS:
            try:
                tx(m, EXIT)
            except Exception:
                fail += 1
        # 0922: 결과를 기계 판독 가능한 판정으로 구분한다. 종전엔 '송신실패 N건' 한 줄뿐이라
        #   **모터 전원이 꺼진 경우**(응답 0/12 → ACK 없음 → 송신 전부 실패)와 **일부 모터 이상**을 구분할 수 없었고,
        #   콘솔이 둘 다 실패로 처리해 토크 해제 단계가 영구히 막혔다(0918·0921 두 번 발생).
        #   버스에 응답 노드가 0 이면 CAN 으로 토크를 **걸 수도 해제할 수도** 없다 — 별도 판정으로 알린다.
        n_seen = len(pos)
        if fail == 0:
            print(f"[release_all] {len(IDS)}모터 소프트 릴리즈+MIT 해제 완료 — 무토크")
            print("RELEASE_VERDICT=OK seen=%d fail=0" % n_seen)
            verdict_code = 0
        elif n_seen == 0:
            print(f"[release_all] 버스 무응답 — 응답 0/{len(IDS)}, 송신실패 {fail}건. "
                  f"CAN 으로는 토크를 걸 수도 해제할 수도 없습니다(모터 전원 차단 추정). "
                  f"전원 차단을 현장에서 확인하세요.")
            print("RELEASE_VERDICT=NO_BUS seen=0 fail=%d" % fail)
            verdict_code = 4
        else:
            print(f"[release_all] 완료 — 응답 {n_seen}/{len(IDS)}, 송신실패 {fail}건 (버스 상태 확인!)")
            print("RELEASE_VERDICT=PARTIAL seen=%d fail=%d" % (n_seen, fail))
            verdict_code = 5
        globals()["VERDICT_CODE"] = verdict_code
        try:
            bus.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    _rc = main()
    raise SystemExit(globals().get("VERDICT_CODE", 0) or _rc)
