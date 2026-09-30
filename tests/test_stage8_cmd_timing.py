"""stage8 명령 지연 기록기 CommandTiming (0930) — 기록 전용, 명령 처리 동작 불변.

Run: python3 -m pytest tests/test_stage8_cmd_timing.py -q
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jetson.scripts import stage8_12axis_mit_control_node as s8  # noqa: E402


def _msg(seq, sent, src="bridge:1:1"):
    return {"command": "SET_JOINT_TARGETS", "targets": {}, "seq": seq, "sent_ts": sent, "src": src}


def test_age_percentiles_and_stale():
    ct = s8.CommandTiming()
    now = 1000.0
    for k in range(50):                       # 1 s 동안 20 ms 간격, 지연 15 ms (마지막 5건은 90 ms)
        t = now + 0.02 * k
        age = 0.090 if k >= 45 else 0.015
        ct.update(t, _msg(k + 1, t - age))
    snap = ct.snapshot(now + 0.98)
    assert snap["n"] == 50 and snap["last_age_ms"] == 90.0
    assert snap["p50_age_ms_1s"] == 15.0 and snap["max_age_ms_1s"] == 90.0
    assert snap["stale"] == 5 and snap["seq_gaps"] == 0 and snap["seq_regress"] == 0
    assert ct.snapshot(now + 5.0)["count_1s"] == 0          # 명령이 끊기면 1 s 창이 비워진다


def test_gaps_regress_and_source_change():
    ct = s8.CommandTiming()
    for seq in (1, 2, 5, 6):                   # 3·4 누락
        ct.update(10.0, _msg(seq, 9.99))
    ct.update(10.0, _msg(6, 9.99))              # 같은 출처에서 순번 역행(중복)
    assert ct.gaps == 2 and ct.regress == 1
    ct.update(10.0, _msg(1, 9.99, src="bridge:2:2"))   # 브리지 재기동 = 새 출처, 순번 1 부터 — 역행으로 세지 않음
    assert ct.src_changes == 1 and ct.regress == 1 and ct.gaps == 2


def test_old_publisher_without_fields_is_ignored():
    ct = s8.CommandTiming()
    ct.update(10.0, {"command": "SET_JOINT_TARGETS", "targets": {}})
    ct.update(10.0, {"command": "SET_JOINT_TARGETS", "targets": {}, "seq": True, "sent_ts": float("nan")})
    snap = ct.snapshot(10.0)
    assert snap["n"] == 0 and snap["last_age_ms"] is None and snap["seq_gaps"] == 0


def test_core_behavior_unchanged_and_status_additive():
    core = s8.Stage8TwelveAxisMitControlCore(now_fn=lambda: 50.0)
    st = core.handle_command_data(json.dumps(_msg(1, 49.98)))
    assert core.last_reject_reason == "not_armed"            # 무장 전 거부는 종전과 같다
    assert st["cmd_timing"]["n"] == 1 and st["cmd_timing"]["last_age_ms"] == 20.0
    st2 = core.handle_command_data(json.dumps({"command": "SET_JOINT_TARGETS", "targets": {}}))
    assert core.last_reject_reason == "not_armed" and st2["cmd_timing"]["n"] == 1
    json.dumps(st2)                                            # 상태 JSON 직렬화 가능
