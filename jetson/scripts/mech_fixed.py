#!/usr/bin/env python3
"""기구 고정 관절 선언의 단일 원본 (2026-10-09).

배경: 모터를 떼고 출력물로 0° 에 굳힌 관절(예: 좌우 힙 회전)은 CAN 에 응답하지 않는다.
정책·러너는 12 관절 그대로 두고, 하드웨어맵 관절 항목의 선택 필드로 **명시 선언**한 관절만
예외 처리한다:

    "left_hip_r_joint": { ..., "mech_fixed": true }

  - 필드가 없거나 false = 종전과 완전히 같은 동작(12 모터 전부 살아 있어야 한다).
  - 고정 각도 = 관절 0° (= 모터 프레임 stand_zero_deg). kp/kd·direction·stand_zero_deg·motor_id 는 그대로 둔다.
  - `enabled` 필드와는 무관하다(재활용하지 않는다).
  - 고정 모터 ID 는 맵에서 읽는다 — 번호를 코드에 박지 않는다.

가드·문턱은 이 선언으로 하나도 느슨해지지 않는다: 선언되지 않은 모터의 무응답·stale 은 종전대로
거부·정지다. 선언 값이 bool 이 아니면 조용히 추측하지 않고 ValueError(기동 차단).

stdlib 만 쓴다 — 저장소(jetson/scripts)와 젯슨 실행본(~/rl_calib) 어디서든 같은 파일을 import 한다.

사용:
    from mech_fixed import load_fixed               # ~/rl_calib 스크립트
    names, ids = load_fixed(HW_MAP)                 # (관절명 튜플, 모터 ID 튜플) — 둘 다 모터 ID 순
셀프테스트: python3 mech_fixed.py --selftest
"""
from __future__ import annotations

import json
import sys

FIELD = "mech_fixed"


def fixed_from_map(hw):
    """하드웨어맵 dict → (고정 관절명 튜플, 고정 모터 ID 튜플). 모터 ID 오름차순.

    선언이 없으면 ((), ()). 값이 bool 이 아니거나 motor_id 가 정수가 아니면 ValueError."""
    joints = hw.get("joints") if isinstance(hw, dict) else None
    if not isinstance(joints, dict):
        raise ValueError("하드웨어맵에 joints 객체 없음")
    found = []
    for name, cfg in joints.items():
        if not isinstance(cfg, dict) or FIELD not in cfg:
            continue
        flag = cfg[FIELD]
        if not isinstance(flag, bool):
            raise ValueError(f"{name}: {FIELD}={flag!r} (true/false 만 허용)")
        if not flag:
            continue
        mid = cfg.get("motor_id")
        if isinstance(mid, bool) or not isinstance(mid, int):
            raise ValueError(f"{name}: {FIELD} 선언 관절의 motor_id={mid!r} (정수 필요)")
        found.append((mid, name))
    found.sort()
    return tuple(n for _m, n in found), tuple(m for m, _n in found)


def load_fixed(path):
    """맵 파일 경로 → fixed_from_map 결과. 읽기·파싱 실패는 예외 그대로(호출자가 처리 방침을 정한다)."""
    with open(path, encoding="utf-8") as f:
        return fixed_from_map(json.load(f))


def note(ids):
    """출력용 한 줄 — 고정이 없으면 빈 문자열."""
    return f"모터 {list(ids)} 기구 고정 — 제외" if ids else ""


def _selftest():
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"{name}: {got!r} != {want!r}")

    def hw(**flags):
        js = {f"j{m}": {"motor_id": m} for m in range(1, 13)}
        for k, v in flags.items():
            js[k][FIELD] = v
        return {"joints": js}

    chk("선언 없음", fixed_from_map(hw()), ((), ()))
    chk("false 는 선언 아님", fixed_from_map(hw(j3=False)), ((), ()))
    chk("두 관절·ID 순", fixed_from_map(hw(j9=True, j3=True)), (("j3", "j9"), (3, 9)))
    chk("안내문", note((3, 9)), "모터 [3, 9] 기구 고정 — 제외")
    chk("안내문(없음)", note(()), "")
    for bad in (1, "true", None, 0):
        try:
            fixed_from_map(hw(j3=bad))
            fails.append(f"bool 아닌 값 {bad!r} 통과")
        except ValueError:
            pass
    for bad_map in ({}, {"joints": []}, None):
        try:
            fixed_from_map(bad_map)
            fails.append(f"joints 없는 맵 {bad_map!r} 통과")
        except ValueError:
            pass
    try:
        fixed_from_map({"joints": {"a": {"motor_id": "3", FIELD: True}}})
        fails.append("문자열 motor_id 통과")
    except ValueError:
        pass
    if fails:
        print("[selftest] mech_fixed FAIL\n  " + "\n  ".join(fails))
        return 1
    print("[selftest] mech_fixed: 선언 없음·false·ID 순 정렬·bool 아닌 값 거부·맵 형식 거부 — ALL PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if len(sys.argv) == 2:
        _names, _ids = load_fixed(sys.argv[1])
        print(note(_ids) or "기구 고정 선언 없음", list(_names))
    else:
        print("사용: mech_fixed.py --selftest | <하드웨어맵 경로>")
