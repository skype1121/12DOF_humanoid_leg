#!/usr/bin/env python3
"""pitch_comp(앞뒤 균형 복원항)·guard_mode 를 하드웨어맵 **모든 사본**에 한 번에 쓴다. 0923.

왜 도구가 필요한가
  브리지는 맵 사본이 하나라도 다르면 기동을 거부한다(구맵 사건 재발 방지). 손으로 한쪽만
  고치면 다음 기동이 막힌다. 이 도구는 존재하는 사본 전부에 같은 값을 쓰고 다시 읽어 확인한다.

왜 맵에 두는가
  이득을 바꿀 때 코드를 다시 배포하지 않아도 된다. 브리지 재기동만 하면 된다 —
  시행마다 이득을 손보는 튜닝 루프에 맞춘 자리다.

부호 (0922 3회차 실물 rp_deg 로 확정, tests/test_pitch_comp.py 가 고정)
  pitch + = 앞기울임. 앞으로 기울면 왼발목 +K·pitch, 오른발목 −K·pitch 를 명령에 더한다.
  = 평발 기하선(−1.00 °/°)보다 위를 써서 버티는 방향. 부호가 반대면 낙상을 가속한다.

  python3 set_pitch_comp.py --K 0.35                 # 켜기(기본값으로)
  python3 set_pitch_comp.py --K 0.5 --tau 0.08       # 이득만 바꾸기
  python3 set_pitch_comp.py --off                    # 끄기
  python3 set_pitch_comp.py --guard-mode data        # 발진·리프트 가드를 기록만 (과속은 그대로 정지)
  python3 set_pitch_comp.py --show                   # 현재 값만 보기
적용에는 **브리지 재기동**이 필요하다(맵은 기동 시 1회 읽는다).
"""
import argparse
import json
import os
import sys

MAPS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "config", "robot_12dof_hardware_map.json"),
    "/home/mama/rl_calib/robot_12dof_hardware_map.json",
]

DEFAULTS = {
    "enabled": True,
    "axis": "pitch",
    "joints": "ankle_f",
    "K": 0.35,            # 시뮬 잔차 +0.346(지연 6틱)에서 출발
    "tau_s": 0.06,        # 빠른 되먹임 — 320ms 루프 지연 위에 지연을 더 얹지 않는다
    "limit_deg": 8.0,
    "sign": 1,
    "lr": [1, -1],        # 왼 +, 오른 − (거울)
    "gate": "any",        # 한 발만 닿아도 작동 — 낙상 중 한 발이 떠도 복원항을 끊지 않는다
    "debounce_s": 0.3,
    "slew_deg_per_tick": 1.0,
    "roll_clamp_deg": 25.0,
}


def existing():
    ps = [p for p in MAPS if os.path.exists(p)]
    if not ps:
        sys.exit("하드웨어맵을 찾지 못했다: " + ", ".join(MAPS))
    return ps


def show(ps):
    for p in ps:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pc = d.get("pitch_comp")
        print(f"  {p}")
        print(f"    guard_mode = {d.get('guard_mode', 'safe')}")
        print("    pitch_comp = " + (json.dumps(pc, ensure_ascii=False) if pc else "(없음 = OFF)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--limit", type=float)
    ap.add_argument("--slew", type=float)
    ap.add_argument("--gate", choices=("both", "any"))
    ap.add_argument("--sign", type=int, choices=(-1, 1))
    ap.add_argument("--off", action="store_true", help="pitch_comp 끄기")
    ap.add_argument("--guard-mode", choices=("safe", "data"))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    ps = existing()
    if a.show or not any((a.K is not None, a.tau is not None, a.limit is not None,
                          a.slew is not None, a.gate, a.sign is not None, a.off, a.guard_mode)):
        print(f"하드웨어맵 사본 {len(ps)}개:")
        show(ps)
        return

    # 기준값 = 첫 사본의 현재 pitch_comp, 없으면 DEFAULTS
    with open(ps[0], encoding="utf-8") as f:
        cur = (json.load(f).get("pitch_comp") or {}).copy()
    new = dict(DEFAULTS)
    new.update({k: v for k, v in cur.items() if k in DEFAULTS})
    if a.K is not None:
        new["K"] = a.K
    if a.tau is not None:
        new["tau_s"] = a.tau
    if a.limit is not None:
        new["limit_deg"] = a.limit
    if a.slew is not None:
        new["slew_deg_per_tick"] = a.slew
    if a.gate:
        new["gate"] = a.gate
    if a.sign is not None:
        new["sign"] = a.sign
    new["enabled"] = not a.off

    # 쓰기 전에 브리지 검증기로 통과시킨다 — 봉투 밖 값이 맵에 실리는 경로 차단
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import importlib.util
        import types
        for nm in ("rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg",
                   "sensor_msgs", "sensor_msgs.msg", "geometry_msgs", "geometry_msgs.msg"):
            if nm not in sys.modules:
                m = types.ModuleType(nm); m.__path__ = []; sys.modules[nm] = m
        sys.modules["rclpy.node"].Node = type("Node", (), {"__init__": lambda s, *x, **k: None})
        for mod, cls in (("std_msgs.msg", "String"), ("sensor_msgs.msg", "Imu"),
                         ("sensor_msgs.msg", "JointState"), ("geometry_msgs.msg", "Vector3")):
            setattr(sys.modules[mod], cls, type(cls, (), {"__init__": lambda s, *x, **k: None}))
        for nm, v in (("QoSProfile", type("Q", (), {"__init__": lambda s, *x, **k: None})),
                      ("ReliabilityPolicy", types.SimpleNamespace(BEST_EFFORT=0, RELIABLE=1)),
                      ("HistoryPolicy", types.SimpleNamespace(KEEP_LAST=0)),
                      ("DurabilityPolicy", types.SimpleNamespace(VOLATILE=0))):
            setattr(sys.modules["rclpy.qos"], nm, v)
        spec = importlib.util.spec_from_file_location(
            "_rlbn", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_bridge_node.py"))
        mod = importlib.util.module_from_spec(spec)
        _argv, sys.argv = sys.argv, ["rl_bridge_node.py", "--selftest"]
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        finally:
            sys.argv = _argv
        if new["enabled"]:
            mod._validate_roll_comp(new, name="pitch_comp")
    except ValueError as e:
        sys.exit(f"거부 — {e}")

    for p in ps:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["pitch_comp"] = new
        if a.guard_mode:
            d["guard_mode"] = a.guard_mode
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    print(f"{len(ps)}개 사본에 기록:")
    show(ps)
    print("\n적용하려면 **브리지 재기동**이 필요하다 (맵은 기동 시 1회 읽는다).")


if __name__ == "__main__":
    main()
