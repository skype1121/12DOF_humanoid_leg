"""Isaac Sim 쪽 브리지 — main UI의 SIM 경로 상대편 (2026-07-26 신설).

역할 2가지 (종전 미체크인 외부 스크립트를 리포에 정식 편입):
1. 상태 하트비트: /tmp/humanoid_isaac_state.json 에 timestamp/simulation_playing/
   state_valid/state_reason/internal_joints(12관절 deg)를 5Hz로 기록 —
   main UI의 SIM 대상 연결 게이트(_is_sim_ready, staleness 2s)가 이걸 읽는다.
2. 명령 파일 적용: /tmp/humanoid_12dof_sim_command.json (Stage9 슬라이더 경로,
   deg)을 감지해 관절 목표로 적용. 컨트롤러가 STAND/HOLD일 때만 — RL/FSM
   보행 중 슬라이더 명령이 정책과 충돌하지 않게 차단.

사용 (Isaac Sim script editor 또는 MCP execute_script):
    import sys; sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
    from sim_interface import isaac_sim_bridge as B
    B.start()      # 중지: B.stop()   상태: B.status()

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import json
import math
import os
import time

STATE_PATH = "/tmp/humanoid_isaac_state.json"
CMD_PATH = "/tmp/humanoid_12dof_sim_command.json"
STATE_EVERY_FRAMES = 12          # 60fps 기준 5Hz (UI staleness 게이트 2s)

_sub = None
_frame = 0
_last_cmd_ts = None
_err = ""


def _write_state(ctrl):
    import omni.timeline
    playing = bool(omni.timeline.get_timeline_interface().is_playing())
    joints = {}
    valid, reason = True, "ok"
    try:
        jp = ctrl.s.art.get_joint_positions()
        row = jp[0] if getattr(jp, "ndim", 1) > 1 else jp
        for name, val in zip(ctrl.s.names, row):
            joints[name] = math.degrees(float(val))
        if len(joints) != 12:
            valid, reason = False, f"joint_count:{len(joints)}"
    except Exception as ex:
        valid, reason = False, f"{type(ex).__name__}: {ex}"[:80]
    state = {
        "timestamp": time.time(),
        "simulation_playing": playing,
        "state_valid": valid,
        "state_reason": reason,
        "internal_joints": joints,
        "controller_mode": ctrl.mode,
    }
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def _apply_command(ctrl):
    """슬라이더 명령 파일 → HOLD 목표 병합 (STAND/HOLD에서만)."""
    global _last_cmd_ts
    try:
        with open(CMD_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    ts = data.get("timestamp")
    if ts is None or ts == _last_cmd_ts:
        return
    _last_cmd_ts = ts
    if ctrl.mode not in ("STAND", "HOLD"):
        return  # 보행(RL/WALK) 중 슬라이더 무시 — 충돌 방지
    targets_deg = data.get("joint_targets_deg") or {}
    if not targets_deg:
        return
    base = dict(ctrl._hold_targets) if (
        ctrl.mode == "HOLD" and ctrl._hold_targets) else dict(ctrl._neutral)
    for name, deg in targets_deg.items():
        base[name] = math.radians(float(deg))
    ctrl._hold_targets = base
    ctrl.mode = "HOLD"


def _tick(_e):
    global _frame, _err
    _frame += 1
    if _frame % STATE_EVERY_FRAMES:
        return
    try:
        from sim_walking import live_controller as LC
        ctrl = LC.get()
        _write_state(ctrl)
        _apply_command(ctrl)
        _err = ""
    except Exception as ex:
        _err = f"{type(ex).__name__}: {ex}"[:120]


def start():
    """브리지 시작 (idempotent). 컨트롤러가 없으면 생성(STAND)."""
    global _sub
    if _sub is not None:
        return {"ok": True, "note": "already running"}
    import omni.kit.app
    stream = omni.kit.app.get_app().get_update_event_stream()
    _sub = stream.create_subscription_to_pop(
        _tick, name="sim_interface.isaac_sim_bridge")
    return {"ok": True, "state_path": STATE_PATH, "cmd_path": CMD_PATH}


def stop():
    global _sub
    if _sub is not None:
        _sub.unsubscribe()
        _sub = None
    return {"ok": True}


def status():
    return {"running": _sub is not None, "frame": _frame,
            "last_err": _err, "last_cmd_ts": _last_cmd_ts}
