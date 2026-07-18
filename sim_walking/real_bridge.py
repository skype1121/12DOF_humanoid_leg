"""sim → real 브리지 (Stage8 12축 MIT 프로토콜 기준. 기존 real 스택 무변경).

검증된 실물 구조를 그대로 따른다:
  - 페이로드: {"command": "SET_JOINT_TARGET", "joint": <이름>, "target_deg": <상대각>}
    (jetson/scripts/stage8_12axis_mit_control_node.py — 단일 관절만, 멀티관절 거부)
  - 각도는 '베이스라인 대비 상대각'. 노드가 50Hz 틱에서 4°/tick 슬루로 추종.
  - ARM_ALL / SET_BASELINE_FROM_CURRENT_ALL / STOP_ALL 은 이 브리지가 절대
    자동 송신하지 않는다 — 운영자가 기존 스크립트로 직접 수행.
  - kp/kd 는 노드측(하드웨어맵) 소관. 브리지는 각도 스트림만 만든다.

베이스라인 규약(중요):
  실물을 '시뮬 중립 자세(전 관절 0°)'로 세워둔 상태에서 SET_BASELINE을 잡으면
  tape 의 target_deg = 시뮬 절대각과 동일해진다. (트림/보행각 모두 포함됨)

부호 규약:
  tape 값 = 시뮬 각[rad→deg] × 하드웨어맵 direction × sign.
  실물 모터 방향 캘리브레이션은 하드웨어맵 direction/sign 에만 기록하면
  브리지가 자동 반영한다 (현재 전부 +1).

안전:
  - 이 모듈은 ROS/CAN 을 import 하지 않는다. 출력은 dict tape 와 jsonl 파일.
  - 실 송신은 별도 sink 를 운영자가 명시적으로 붙여야 하며, 그 경우에도
    Stage8 노드의 ARM/베이스라인/리밋/슬루/피드백 신선도 가드가 최종 방어선.
  - 개루프 tape 에는 밸런스 피드백이 없다 → 지면 위 자유 기립 보행용이 아니라
    '거치대/서스펜션 테스트, 스탠딩 체중이동' 검증용이다. (게인 실험 결론:
    kd<=5 제약에서 보행은 추가 작업 필요 — sim_dynamics.json 참조)
"""
import json
import math
import os

from .sim_session import REPO

R2D = 180.0 / math.pi

STAGE8_COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
NODE_TICK_HZ = 50.0
NODE_SLEW_DEG_PER_TICK = 4.0
SAFETY_MAX_DELTA_DEG = 30.0     # control_logic SafetyFilter 와 동일


def _load_hw_map():
    with open(os.path.join(REPO, "config", "robot_12dof_hardware_map.json")) as f:
        return json.load(f)["joints"]


def _load_sim_limits():
    with open(os.path.join(REPO, "config", "sim_joint_limits.json")) as f:
        return json.load(f)["limits_deg"]


class BridgeConfig:
    def __init__(self, stream_hz=12.5, deadband_deg=0.2, use_sim_limits=True):
        self.stream_hz = float(stream_hz)
        self.deadband_deg = float(deadband_deg)
        self.use_sim_limits = bool(use_sim_limits)


def build_tape(gait, duration_s, cfg: BridgeConfig = None):
    """gait.targets(t) → Stage8 SET_JOINT_TARGET 명령 tape.

    returns (tape, stats)
      tape: [{"t": 초, "payload": {...}}, ...]  시간순
      stats: 검증 통계 (violations 가 하나라도 있으면 tape 사용 금지)
    """
    cfg = cfg or BridgeConfig()
    hw = _load_hw_map()
    sim_lim = _load_sim_limits()
    dt = 1.0 / cfg.stream_hz
    n = int(duration_s / dt) + 1

    # 노드 슬루가 스트림 주기 내 낼 수 있는 최대 이동량 (90% 마진)
    slew_capacity = NODE_SLEW_DEG_PER_TICK * (NODE_TICK_HZ / cfg.stream_hz) * 0.9
    max_delta = min(SAFETY_MAX_DELTA_DEG, slew_capacity)

    tape = []
    last_sent = {}
    stats = {
        "stream_hz": cfg.stream_hz, "duration_s": duration_s,
        "commands": 0, "skipped_deadband": 0,
        "max_abs_deg": {}, "max_delta_deg": {},
        "violations": [], "slew_capacity_deg": round(slew_capacity, 2),
    }
    for i in range(n):
        t = i * dt
        tg = gait.targets(t)                     # {joint: rad, 시뮬 부호}
        for joint, rad in tg.items():
            j = hw[joint]
            deg = rad * R2D * float(j["direction"]) * float(j["sign"])
            # 리밋 검증 (하드웨어맵 + 시뮬 실측 리밋 교집합)
            tl = float(j["target_limit_deg"])
            lo, hi = -tl, tl
            if cfg.use_sim_limits:
                lo = max(lo, float(sim_lim[joint]["lower"]))
                hi = min(hi, float(sim_lim[joint]["upper"]))
            if not (lo - 1e-9 <= deg <= hi + 1e-9):
                stats["violations"].append(
                    {"t": round(t, 3), "joint": joint, "deg": round(deg, 2),
                     "reason": f"limit [{lo:.1f},{hi:.1f}]"})
                continue
            prev = last_sent.get(joint)
            if prev is not None:
                d = abs(deg - prev)
                if d < cfg.deadband_deg:
                    stats["skipped_deadband"] += 1
                    continue
                if d > max_delta:
                    stats["violations"].append(
                        {"t": round(t, 3), "joint": joint,
                         "delta": round(d, 2),
                         "reason": f"delta>{max_delta:.1f} (slew/safety)"})
                    continue
                stats["max_delta_deg"][joint] = round(
                    max(stats["max_delta_deg"].get(joint, 0.0), d), 3)
            last_sent[joint] = deg
            stats["max_abs_deg"][joint] = round(
                max(stats["max_abs_deg"].get(joint, 0.0), abs(deg)), 3)
            tape.append({"t": round(t, 4), "payload": {
                "command": "SET_JOINT_TARGET",
                "joint": joint,
                "target_deg": round(deg, 4),
            }})
            stats["commands"] += 1
    return tape, stats


def write_tape_jsonl(tape, stats, path):
    """tape 를 jsonl 로 저장 (첫 줄 = 메타/절차 주석 레코드)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({
            "_meta": {
                "format": "stage8_command_tape_v1",
                "topic": STAGE8_COMMAND_TOPIC,
                "msg_type": "std_msgs/String (data=payload JSON)",
                "operator_procedure": [
                    "1) 로봇을 시뮬 중립 자세(전 관절 0)로 거치/지지",
                    "2) 기존 스크립트로 ARM_ALL → SET_BASELINE_FROM_CURRENT_ALL",
                    "3) tape 재생 (각 레코드의 t 시각에 payload 발행)",
                    "4) 이상 시 STOP_ALL / DISARM_ALL",
                ],
                "warning": "개루프 tape — 자유 기립 보행용 아님 (거치대/체중이동 검증용)",
                "stats": stats,
            }}, ensure_ascii=False) + "\n")
        for rec in tape:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path
