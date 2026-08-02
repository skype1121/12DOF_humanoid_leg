"""RL 브리지 노드 — 정책 실행기 → Stage8 세트포인트 (실물 이식용, 2026-07-27).

구조 (전부 기검증 조각의 조립):
  센서 구독 → PolicyRunnerStage4(관측245 조립·비트일치 검증본·리밋 클램프 내장)
  → 50Hz 틱마다 12관절 목표각 → Stage8 SET_JOINT_TARGET ×12 발행
  (12×50Hz = 600msg/s — 목 벤치 검증 여유 PASS. 노드 측 게이트/슬루/절대각
  리밋 검사 그대로 통과하므로 브리지가 뚫려도 노드가 막는다.)

프레임 변환 (2026-07-27 실측 캘리브레이션 확정 — config/robot_12dof_hardware_map.json):
  관절 j: DIR[j] = direction(±1, 실물 미소동작 검증),
          SZ[j]  = stand_zero_deg(모터 인코더 프레임 deg, stand_snapshot_v4)
  입력 (모터→관절): joint_rad = radians( DIR × (motor_deg − SZ) )
  출력 (관절→모터): motor_target_deg = SZ + DIR × degrees(policy_target_rad)
  발행 상대각 rel = motor_target_deg − baseline_deg → Stage8 절대각 = baseline + rel
  = motor_target 그대로 복원 (baseline이 어떤 자세로 캡처됐든 무관).
  AK45(모터 6·12): 위치 스케일 1.0 확정 — 위치엔 보정 없음. 단 velocity 디코드는
  과대판독 — 관절 6·12의 feedback.velocity_rad_s 는 소비 금지 (qvel은 어떤 관절도
  피드백 속도를 쓰지 않고 관절 프레임 위치의 유한차분만 사용).

토픽 규약:
  구독 /humanoid/joint_states            (JSON, name→deg) — 값은 '모터 인코더
      프레임' deg. 현재 저장소에 발행자 없음(규약 유지) — 실질 피드백 소스는 아래
  구독 /humanoid/stage8_12axis_mit_status  armed·baseline_deg 확인 + 관절 피드백
      겸용: joints[j].latest_actual_deg(모터 프레임 deg)를 feedback_age_sec<0.5s
      일 때만 신선한 관절각으로 흡수 — Stage8 상태만으로 브리지 구동 가능
  구독 /humanoid/imu                     iAHRS 드라이버 (JSON):
      {"timestamp": epoch초, "gyro_rad_s": [wx,wy,wz](body),
       "quat_wxyz": [w,x,y,z](world<-body)}          ※ 드라이버 미구현 — 규약 확정용
  구독 /humanoid/foot_force              RA30P 드라이버 (JSON):
      {"timestamp": epoch초, "left_n": 법선힘N, "right_n": N}   ※ 미구현
  구독 /humanoid/rl_bridge_command       조종 (JSON):
      {"cmd": "stand"|"walk"|"march"|"stop"|"estop",
       "vx":0.0,"vy":0.0,"wz":0.0, "heading_hold": true}
  발행 /humanoid/stage8_12axis_mit_command   (--live 일 때만)
  발행 /humanoid/rl_bridge_debug            매 틱 진단 (드라이런 목표각 포함)

안전 (기본 = 드라이런):
  - --live 없으면 Stage8 명령 발행 안 함 (debug 토픽만) — 이중: 노드 armed 게이트
  - 센서 신선도 0.2s 초과 → 즉시 정지(명령 중단 + STOP_ALL 1회) — 정책은 stale
    관측으로 오동작하므로 멈추는 게 안전
  - estop 명령 → STOP_ALL 발행 + 루프 정지
  - 체크포인트: walk=R4(12994) 기본. march 전환은 접지 테스트 단계에서만.
  - kd는 노드 하드웨어맵 관할 — 브리지는 게인 미접촉. 시뮬 kd25 금지 원칙.
  - 게인 게이트: 정책 라이브는 학습게인 kp150/kd5 일치가 전제 (kd5 = MIT 프로토콜
    상한 = 실물 kd 상한 규칙). 하드웨어맵 kp/kd 불일치 시 --live 기동 거부 —
    --force-gains 로만 우회 (맵 갱신은 사용자 결정 사항).
  - 프레임 게이트: --live 는 boot_pose_check 저널(last_pose_journal.json)에
    12모터 전부 '✓ 일치' 검증 관측이 있고 최고령 관측이 --frame-max-age(기본
    60분) 이내일 때만 기동. 로터절대 엔코더는 전원사이클마다 36°/10° 배수로
    어긋날 수 있으므로 (encoder aliasing) 미검증 프레임에 정책을 얹으면 안 됨.
    전원사이클 자체는 저널로 감지 불가 — 절차(사이클 후 재판정)가 원칙이고 이
    게이트는 백스톱. --force-frame 으로만 우회.
  - 하드웨어맵 결함(12관절 direction/stand_zero_deg 미비) 시 기동 자체 차단.
  - 발진 가드(0802 공중 발진 사고 재발 방지): live 중 어느 관절이든 0.5s 창에
    속도 부호반전 ≥6회+진폭 ≥2° (또는 |qvel|>12rad/s) → 자동 STOP_ALL·비활성.
    이후 절차 = 줄 재인장 → release_all.py. estop 손명령은 발진 속도를 못 따라감.
  - 블랙박스: live면 ~/logs/rl_bridge_*.jsonl 에 틱 단위 관측·타깃·이벤트 기록
    (/tmp 금지 — 재부팅 소실 사고 재발 방지).

드라이런 자가검증 (로봇·ROS 불필요):
  python3 jetson/scripts/rl_bridge_node.py --selftest
  → 하드웨어맵 12/12 검증 + 왕복 항등 + 수치 앵커 + 게인 정책
  → 목 센서(정지 직립)로 500틱 돌려 목표각 유한성·리밋·틱 주기 검증
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rl_walking.deploy.policy_runner_stage4 import (  # noqa: E402
    CONTACT_FORCE_THRESHOLD_N, HeadingHold, PolicyRunnerStage4,
    yaw_from_quat_wxyz)
from rl_walking.deploy.policy_runner import (  # noqa: E402
    DEFAULT_POSE_RAD, JOINT_ORDER)

RATE_HZ = 50.0
TICK_SEC = 1.0 / RATE_HZ
SENSOR_STALE_SEC = 0.2
#: 드라이런(비-live) 관절 신선 한계 — Stage8 비무장 시 CAN 피드백이 12모터
#: 라운드로빈(~240ms/바퀴)이라 0.2s 게이트로는 영원히 stale. live 무장 시엔
#: 명령마다 피드백이 와서(50Hz) 0.2s 엄격 게이트가 그대로 적용된다.
JOINT_STALE_DRYRUN_SEC = 0.5
STATUS_FEEDBACK_FRESH_SEC = 0.5   # status 경유 관절 피드백(feedback_age_sec) 신선 한계
COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
JOINT_STATES_TOPIC = "/humanoid/joint_states"
IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"
BRIDGE_CMD_TOPIC = "/humanoid/rl_bridge_command"
BRIDGE_DEBUG_TOPIC = "/humanoid/rl_bridge_debug"

CHECKPOINTS = {
    "walk": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-07-26_14-53-03/exported"),
    "march": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-07-26_21-41-30/exported"),
    # DRv2 (0802 보강학습, 컴플라이언스 정합): 컴플라이언스 조건 낙상 1.6% vs
    # walk 85.9% — 실물(구조 컴플라이언스 실증) 배포 1순위. 표준조건 푸시 마진은
    # walk가 우위(0.5m/s: 2% vs 28%) — 실물 강성 실측 후 v3에서 회수 예정.
    "drv2": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-02_04-31-34/exported"),
    # march-DRv2 (0802): 컴플라이언스 낙상 4.7% vs 구 march 46.9%, 표준 0% —
    # 실물 제자리걸음 배포 1순위 (구 march와 동일 주의: 명령 0 = 정지 아닌 스텝)
    "march_drv2": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-02_12-22-59/exported"),
}

# ---------------------------------------------------------------------------
# 프레임 변환 (2026-07-27 실측 캘리브레이션 확정)
# ---------------------------------------------------------------------------

#: 하드웨어맵 탐색 순서 — 저장소 상대 우선, 젯슨 배포(/home/mama/rl_calib) 폴백
HARDWARE_MAP_PATHS = (
    os.path.join(REPO, "config", "robot_12dof_hardware_map.json"),
    "/home/mama/rl_calib/robot_12dof_hardware_map.json",
)
#: 수치 앵커 (정책 기본자세 → 모터 목표각, 2026-07-27 산출물) — selftest 대조용
ANCHOR_TARGETS_PATH = os.path.join(
    REPO, "jetson", "calib_data", "sim_default_stand_targets_20260727.json")

POLICY_KP = 150.0   # 학습 게인 — 정책 라이브 전제 (kd5 = MIT 프로토콜 상한 규칙)
POLICY_KD = 5.0     # 시뮬 kd25는 절대 실물 전송 금지


class FrameMap:
    """모터 인코더 프레임 ↔ URDF 관절 프레임 변환 (하드웨어맵 단일 원본).

    확정 규약 (stand_snapshot_v4, 2026-07-27):
      입력: joint_rad = radians( DIR × (motor_deg − SZ) )
      출력: motor_target_deg = SZ + DIR × degrees(joint_rad)
    """

    def __init__(self, path, dir_by_joint, sz_by_joint, motor_id_by_joint,
                 gains_by_joint):
        self.path = path
        self.dir_by_joint = dir_by_joint            # {관절: ±1}
        self.sz_by_joint = sz_by_joint              # {관절: 모터 프레임 deg}
        self.motor_id_by_joint = motor_id_by_joint  # {관절: 1..12}
        self.gains_by_joint = gains_by_joint        # {관절: (kp, kd)}

    def motor_deg_to_joint_rad(self, joint, motor_deg):
        """입력 변환: 모터 인코더 절대각 deg → URDF 관절각 rad."""
        return math.radians(
            self.dir_by_joint[joint]
            * (float(motor_deg) - self.sz_by_joint[joint]))

    def joint_rad_to_motor_deg(self, joint, joint_rad):
        """출력 변환: URDF 관절 목표각 rad → 모터 인코더 목표각 deg."""
        return (self.sz_by_joint[joint]
                + self.dir_by_joint[joint] * math.degrees(float(joint_rad)))

    def gain_mismatches(self):
        """학습게인(kp150/kd5)과 다른 관절 목록 [(관절, kp, kd), ...]."""
        return [(j, kp, kd) for j, (kp, kd) in self.gains_by_joint.items()
                if (kp, kd) != (POLICY_KP, POLICY_KD)]


def load_frame_map():
    """하드웨어맵 로드 + 12관절 전수 검증 — 결함 시 즉시 예외 (기동 차단).

    검증: 12관절 전부 direction ∈ {−1,+1} 그리고 stand_zero_deg 유한 float,
    |SZ| < 120°. 하나라도 빠지면 어떤 변환도 신뢰 불가 → 하드 페일.
    """
    existing = [p for p in HARDWARE_MAP_PATHS if os.path.isfile(p)]
    if not existing:
        raise FileNotFoundError(
            "하드웨어맵 없음 — 다음 경로를 모두 확인함: "
            + ", ".join(HARDWARE_MAP_PATHS))
    # 두 사본이 공존하면 DIR/SZ 일치 검증 (적대리뷰 수정): 드라이런은 A사본으로
    # PASS 받고 브리지는 B사본으로 뜨는 이력(젯슨 구맵 사건)을 원천 차단한다.
    if len(existing) > 1:
        def _frame_sig(p):
            with open(p, encoding="utf-8") as f:
                js = json.load(f).get("joints") or {}
            return {n: (c.get("direction"), round(float(c.get("stand_zero_deg", 1e9)), 2))
                    for n, c in js.items() if isinstance(c, dict) and "motor_id" in c}
        sigs = {p: _frame_sig(p) for p in existing}
        first = sigs[existing[0]]
        for p in existing[1:]:
            if sigs[p] != first:
                raise ValueError(
                    "하드웨어맵 사본 불일치 — direction/stand_zero_deg 가 서로 다름:\n  "
                    + "\n  ".join(existing)
                    + "\n  구맵 사본을 동기화한 뒤 재기동하세요 (기동 차단)")
    path = existing[0]
    print(f"[FrameMap] 하드웨어맵: {path}")
    with open(path, encoding="utf-8") as f:
        joints = json.load(f).get("joints") or {}
    dir_by, sz_by, mid_by, gains_by = {}, {}, {}, {}
    problems = []
    for name in JOINT_ORDER:
        info = joints.get(name)
        if not isinstance(info, dict):
            problems.append(f"{name}: 관절 항목 없음")
            continue
        joint_ok = True
        d = info.get("direction")
        if isinstance(d, bool) or d not in (-1, 1):
            problems.append(f"{name}: direction={d!r} (−1/+1 만 허용)")
            joint_ok = False
        sz = info.get("stand_zero_deg")
        if (isinstance(sz, bool) or not isinstance(sz, (int, float))
                or not math.isfinite(float(sz)) or abs(float(sz)) >= 120.0):
            problems.append(
                f"{name}: stand_zero_deg={sz!r} (유한 float, |SZ|<120° 필요)")
            joint_ok = False
        if not joint_ok:
            continue
        dir_by[name] = int(d)
        sz_by[name] = float(sz)
        mid_by[name] = int(info.get("motor_id", 0))
        gains_by[name] = (float(info.get("kp", 0.0)), float(info.get("kd", 0.0)))
    if problems:
        raise ValueError(
            f"하드웨어맵 검증 실패 ({path}) — 변환 상수 신뢰 불가, 기동 중단:\n  "
            + "\n  ".join(problems))
    if sorted(mid_by.values()) != list(range(1, 13)):
        raise ValueError(
            f"하드웨어맵 motor_id 이상 (1..12 전단사 아님): {mid_by} ({path})")
    return FrameMap(path, dir_by, sz_by, mid_by, gains_by)


class BridgeCore:
    """ROS 무관 코어 — 관측 → 12관절 목표각(rad, URDF 관절 프레임). 셀프테스트 공유.

    모터 인코더 프레임 변환은 코어 밖(FrameMap) 담당 — 코어 입출력은 전부 관절 프레임.
    """

    def __init__(self, checkpoint="walk"):
        self.runner = PolicyRunnerStage4(CHECKPOINTS[checkpoint])
        self.ckpt_name = checkpoint
        self.hh = HeadingHold(kp=1.0, wz_limit=0.6)
        self.hh_ref_set = False
        self.cmd = (0.0, 0.0, 0.0)
        self.heading_hold = True
        self.active = False          # stand/walk/march 시작 여부

    def reset(self, checkpoint=None):
        if checkpoint and checkpoint != self.ckpt_name:
            self.runner = PolicyRunnerStage4(CHECKPOINTS[checkpoint])
            self.ckpt_name = checkpoint
        self.runner.reset()
        self.hh_ref_set = False

    def tick(self, gyro, quat_wxyz, qpos_rad, qvel_rad_s, contact2):
        """관측 → 목표각 12 [rad, 절대] (JOINT_ORDER 순)."""
        w, x, y, z = quat_wxyz
        # projected gravity (world -z 를 body 프레임으로)
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        grav = (gx, gy, gz)
        vx, vy, wz = self.cmd
        if self.heading_hold and abs(wz) < 1e-6:
            yaw = yaw_from_quat_wxyz(w, x, y, z)
            if not self.hh_ref_set:
                self.hh.set_ref(yaw)
                self.hh_ref_set = True
            wz = self.hh.update(yaw)
        return self.runner.step(gyro, grav, (vx, vy, wz),
                                qpos_rad, qvel_rad_s, contact2)


def _selftest_hwmap():
    """(1) 하드웨어맵 검증 12/12 — 로더 하드페일 조건 그대로 통과 확인."""
    fm = load_frame_map()
    dirs = [fm.dir_by_joint[j] for j in JOINT_ORDER]
    sz_max = max(abs(v) for v in fm.sz_by_joint.values())
    return (f"12/12 검증 통과 ({fm.path}) — direction(JOINT_ORDER순)={dirs}, "
            f"|SZ|max={sz_max:.2f}°")


def _selftest_roundtrip():
    """(2) 왕복 항등: 관절→모터→관절 오차 < 1e-9 rad, 12관절 × 표본각 7개."""
    fm = load_frame_map()
    samples = (-1.5, -0.7, -0.10471975511965977, 0.0, 0.0244, 0.9, 1.5)
    worst = 0.0
    for j in JOINT_ORDER:
        for q in samples:
            q2 = fm.motor_deg_to_joint_rad(j, fm.joint_rad_to_motor_deg(j, q))
            err = abs(q2 - q)
            worst = max(worst, err)
            assert err < 1e-9, f"{j}: 왕복 오차 {err:.3e} rad ≥ 1e-9 (q={q})"
    return f"관절→모터→관절 항등 12관절×{len(samples)}각 OK (최대 오차 {worst:.1e} rad)"


def _selftest_anchors():
    """(3) 수치 앵커: 정책 기본자세를 실제 변환 경로에 통과시켜 산출물과 대조.

    산출물 v2(프레임 불변): 비교는 '영점 상대값' (target − 생성당시 SZ) vs
    (변환 결과 − 현재 SZ). 재앵커(전원사이클 배수 시프트 보정)로 SZ가 창 배수만큼
    움직여도 상대값은 불변 — v5 재앵커에서 m7/m9 절대 비교가 깨진 사건의 재발 방지.
    구형(절대각 dict) 산출물이면 절대 비교로 폴백. 허용오차 0.01°.
    """
    fm = load_frame_map()
    with open(ANCHOR_TARGETS_PATH, encoding="utf-8") as f:
        art = json.load(f)
    legacy = "targets_deg" not in art
    anchors = art if legacy else art["targets_deg"]
    sz_gen = None if legacy else art["stand_zero_at_generation_deg"]
    for i, j in enumerate(JOINT_ORDER):
        mid = str(fm.motor_id_by_joint[j])
        got = fm.joint_rad_to_motor_deg(j, float(DEFAULT_POSE_RAD[i]))
        if legacy:
            diff = got - float(anchors[mid])
        else:
            sz_now = fm.sz_by_joint[j]
            diff = (got - sz_now) - (float(anchors[mid]) - float(sz_gen[mid]))
        assert abs(diff) <= 0.01, (
            f"{j}(모터{mid}): {'절대' if legacy else '영점 상대'} 편차 "
            f"{diff:+.4f}° (허용 0.01°)")
    # 확정 규약 4대 앵커 — 산출물 파일과 독립인 하드코딩 재확인 (영점 상대값:
    # 무릎 ±6.00°, 발목F ±1.40° = 정책 기본자세의 물리적 정의라 재앵커 불변)
    named_rel = {"left_knee_joint": +6.00, "right_knee_joint": -6.00,
                 "left_ankle_f_joint": +1.40, "right_ankle_f_joint": -1.40}
    for j, rel in named_rel.items():
        got = fm.joint_rad_to_motor_deg(
            j, float(DEFAULT_POSE_RAD[JOINT_ORDER.index(j)]))
        got_rel = got - fm.sz_by_joint[j]
        assert abs(got_rel - rel) <= 0.01, (
            f"4대 앵커 {j}: 상대 {got_rel:+.4f}° vs {rel:+.2f}°")
    fmt = "구형(절대)" if legacy else "v2(영점 상대)"
    return f"12/12 산출물 일치[{fmt}] + 4대 앵커(무릎±6.00/발목F±1.40) 일치 (±0.01°)"


def _selftest_gains():
    """(4) 게인 정책: 하드웨어맵 kp/kd ≠ 학습게인(150/5)이면 크게 경고."""
    fm = load_frame_map()
    bad = fm.gain_mismatches()
    if bad:
        lines = ", ".join(f"{j}(kp{kp:g}/kd{kd:g})" for j, kp, kd in bad)
        bar = "!" * 72
        print(f"[WARN] {bar}")
        print(f"[WARN] 하드웨어맵 kp/kd ≠ 학습게인(kp{POLICY_KP:g}/kd{POLICY_KD:g})"
              f" — {len(bad)}/12 관절: {lines}")
        print("[WARN] 정책 라이브는 학습게인 PD 추종이 전제 — 현재 맵 값은 레거시."
              " 맵 갱신은 사용자 결정 사항.")
        print("[WARN] 불일치 상태의 --live 는 기동 거부됨 (--force-gains 로만 우회).")
        print(f"[WARN] {bar}")
        return f"게인 불일치 {len(bad)}/12 관절 — 경고 출력 (라이브 게이트가 차단)"
    return f"전 관절 kp{POLICY_KP:g}/kd{POLICY_KD:g} 일치"


def _selftest_policy():
    """(5) 목 센서(정지 직립) 500틱 — 유한성·리밋·주기 (관절 프레임 그대로)."""
    import numpy as np
    core = BridgeCore("walk")
    core.cmd = (0.0, 0.0, 0.0)
    core.reset()
    q = np.array(core.runner._limits_lo) * 0.0 if core.runner._limits_lo is not None \
        else np.zeros(12)
    gyro = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    qvel = np.zeros(12)
    t0 = time.perf_counter()
    for i in range(500):
        out = core.tick(gyro, quat, q, qvel, [1.0, 1.0])
        assert np.isfinite(out).all(), f"틱 {i}: 비유한 목표"
        if core.runner._limits_lo is not None:
            assert (out >= core.runner._limits_lo - 1e-9).all()
            assert (out <= core.runner._limits_hi + 1e-9).all()
        q = out  # 목표 추종 가정 (목)
    dt = (time.perf_counter() - t0) / 500
    assert dt < TICK_SEC, f"틱 연산 {dt*1000:.2f}ms ≥ 20ms — 젯슨에서 재확인 필요"
    return (f"500틱 유한·리밋 준수, 틱 {dt*1000:.2f}ms "
            f"(50Hz 예산 20ms, 백엔드 {core.runner.backend})")


#: boot_pose_check 검증 저널 (젯슨 경로) — 프레임 게이트의 근거 자료
FRAME_JOURNAL = "/home/mama/rl_calib/last_pose_journal.json"
FRAME_MAX_AGE_DEFAULT_S = 3600.0

#: 발진 가드 파라미터 — 2026-08-02 공중 발진 사고("부들부들 떨다 튕김", 사람
#: 전원컷으로 종료; estop은 ssh 왕복 지연으로 무력) 재발 방지. 보행 리듬
#: (~1.4Hz = 0.5s당 부호반전 ~1.4회)과 명확히 분리되는 영역만 트리거한다.
OSC_WINDOW_SEC = 0.5     #: 감시 창
OSC_REVERSALS = 6        #: 창 내 속도 부호반전 임계 (≈6Hz 이상 진동)
OSC_P2P_RAD = 0.035      #: 그리고 창 내 위치 진폭 ≥2° — 미세 노이즈 배제
OSC_VEL_DEADBAND = 0.5   #: rad/s — 이하 속도의 부호는 무시 (양자화·유한차분 노이즈)
OSC_VEL_HARD = 12.0      #: rad/s 즉시 트리거 (정상 보행 스윙 3~6 rad/s의 2배+)


class OscGuard:
    """관절 발진 감지 (순수 로직, ROS 비의존 — selftest 대상).

    피드백 기반: 어느 관절이든 0.5s 창에서 속도 부호반전 ≥6회 그리고 위치
    진폭 ≥2° 이면 발진으로 판정. |qvel| > 12 rad/s 는 즉시 트리거.
    update(t, qpos, qvel) → None(정상) 또는 사유 문자열(트리거).
    """

    def __init__(self, joint_names):
        self.names = list(joint_names)
        self.last_sign = {j: 0 for j in self.names}
        self.revs = {j: collections.deque() for j in self.names}
        self.pos = {j: collections.deque() for j in self.names}

    def update(self, t, qpos, qvel):
        for i, j in enumerate(self.names):
            v = float(qvel[i])
            if abs(v) > OSC_VEL_HARD:
                return f"과속 {j} {v:+.1f} rad/s (한계 {OSC_VEL_HARD})"
            p = self.pos[j]
            p.append((t, float(qpos[i])))
            while p and t - p[0][0] > OSC_WINDOW_SEC:
                p.popleft()
            s = 0 if abs(v) < OSC_VEL_DEADBAND else (1 if v > 0 else -1)
            if s == 0:
                continue
            if self.last_sign[j] != 0 and s != self.last_sign[j]:
                r = self.revs[j]
                r.append(t)
                while r and t - r[0] > OSC_WINDOW_SEC:
                    r.popleft()
                if len(r) >= OSC_REVERSALS:
                    vals = [q for _, q in p]
                    p2p = max(vals) - min(vals)
                    if p2p >= OSC_P2P_RAD:
                        return (f"발진 {j}: {OSC_WINDOW_SEC}s 내 반전 {len(r)}회, "
                                f"진폭 {math.degrees(p2p):.1f}°")
            self.last_sign[j] = s
        return None


def frame_check_verdict(path=FRAME_JOURNAL, max_age_s=FRAME_MAX_AGE_DEFAULT_S,
                        now=None):
    """프레임 게이트 판정 — None=통과, str=거부 사유.

    boot_pose_check 저널의 motor_ts(모터별 '✓ 일치' 검증 시각)를 근거로,
    12모터 전부 검증 관측이 있고 최고령 관측이 max_age_s 이내면 통과.
    저널에는 검증된 관측만 들어가므로 (시프트/폴트/불안정 배제) 존재+신선
    = "최근에 프레임 12/12 판정을 통과했다"와 동치.
    """
    try:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
    except FileNotFoundError:
        return f"저널 없음({path}) — boot_pose_check --ref stand 먼저 실행"
    except (json.JSONDecodeError, ValueError):
        return f"저널 파손({path}) — boot_pose_check 재실행으로 재생성"
    mts = (j.get("motor_ts") or {}) if isinstance(j, dict) else {}
    missing = [m for m in range(1, 13) if str(m) not in mts]
    if missing:
        return f"모터 {missing} 프레임 미검증 — boot_pose_check 12/12 필요"
    now = time.time() if now is None else now
    try:
        ages = {m: now - float(mts[str(m)]) for m in range(1, 13)}
    except (TypeError, ValueError):
        return "저널 motor_ts 형식 이상 — boot_pose_check 재실행으로 재생성"
    worst = max(ages, key=lambda m: ages[m])
    if ages[worst] > max_age_s:
        return (f"프레임 검증 만료 — 모터{worst} 마지막 검증 "
                f"{ages[worst] / 60:.0f}분 전 (허용 {max_age_s / 60:.0f}분). "
                f"boot_pose_check 재실행 후 기동")
    if min(ages.values()) < -60.0:
        return "저널 타임스탬프가 미래 — 시계 이상 의심, 재판정 필요"
    return None


def _selftest_frame_gate():
    import tempfile
    now = 1_700_000_000.0
    fresh = {str(m): now - 60.0 for m in range(1, 13)}

    def write(d):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, f)
        f.close()
        return f.name

    cases = []
    p = write({"ts": now, "pos_deg": {}, "motor_ts": fresh})
    cases.append(("신선 12/12 통과",
                  frame_check_verdict(p, 3600.0, now) is None))
    stale = dict(fresh)
    stale["7"] = now - 7200.0
    p = write({"motor_ts": stale})
    v = frame_check_verdict(p, 3600.0, now)
    cases.append(("만료 거부", v is not None and "모터7" in v))
    p = write({"motor_ts": {k: v for k, v in fresh.items() if k != "11"}})
    v = frame_check_verdict(p, 3600.0, now)
    cases.append(("결측 거부", v is not None and "11" in v))
    v = frame_check_verdict("/nonexistent/journal.json", 3600.0, now)
    cases.append(("저널 없음 거부", v is not None and "없음" in v))
    p = write({"motor_ts": {**fresh, "3": "abc"}})
    cases.append(("형식 이상 거부",
                  frame_check_verdict(p, 3600.0, now) is not None))
    bad = [n for n, ok in cases if not ok]
    if bad:
        raise AssertionError(f"프레임 게이트 케이스 실패: {bad}")
    return f"판정 케이스 {len(cases)}/{len(cases)} 통과"


def _selftest_osc_guard():
    def run(freq_hz, amp_deg, dur_s):
        g = OscGuard(["j"])
        amp = math.radians(amp_deg)
        w = 2 * math.pi * freq_hz
        n = int(dur_s / TICK_SEC)
        for k in range(n):
            t = k * TICK_SEC
            trig = g.update(t, [amp * math.sin(w * t)],
                            [amp * w * math.cos(w * t)])
            if trig:
                return trig, t
        return None, dur_s

    trig, t = run(8.0, 3.0, 2.0)          # 발진 시나리오 (8Hz, ±3°)
    assert trig and "발진" in trig and t <= 1.0, (trig, t)
    trig2, _ = run(1.4, 20.0, 3.0)        # 정상 보행 리듬 — 미트리거
    assert trig2 is None, trig2
    trig3, _ = run(12.0, 0.15, 2.0)       # 미세 노이즈 (데드밴드 이하) — 미트리거
    assert trig3 is None, trig3
    g = OscGuard(["j"])                    # 과속 즉시 트리거
    trig4 = g.update(0.0, [0.0], [15.0])
    assert trig4 and "과속" in trig4, trig4
    return (f"8Hz±3° {t:.2f}s 내 트리거 / 보행 1.4Hz·노이즈 미트리거 / "
            f"과속 즉시 — 4케이스 통과")


def run_selftest():
    """드라이런 자가검증 — ROS·로봇 불필요. 프레임 변환 4종 + 정책 500틱."""
    tests = [("하드웨어맵", _selftest_hwmap),
             ("왕복 항등", _selftest_roundtrip),
             ("수치 앵커", _selftest_anchors),
             ("게인 정책", _selftest_gains),
             ("프레임 게이트", _selftest_frame_gate),
             ("발진 가드", _selftest_osc_guard),
             ("정책 500틱", _selftest_policy)]
    failed = 0
    for name, fn in tests:
        try:
            print(f"[PASS] {name}: {fn()}")
        except Exception as e:  # noqa: BLE001 — selftest는 전 항목 보고가 목적
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("selftest:", "ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="실제 Stage8 명령 발행 (없으면 드라이런 — debug 토픽만)")
    ap.add_argument("--force-gains", action="store_true",
                    help="하드웨어맵 kp/kd가 학습게인(kp150/kd5)과 달라도 --live 허용. "
                         "정책은 kp150/kd5 PD 추종을 전제로 학습됨 — 게인 불일치 시 "
                         "실물 거동이 학습 분포를 벗어난다. 원칙은 맵 갱신(사용자 결정)"
                         "이고, 이 플래그는 의도적 저게인 예비시험 전용 (기본 False)")
    ap.add_argument("--force-frame", action="store_true",
                    help="boot_pose_check 프레임 검증(저널 12/12·신선) 없이도 "
                         "--live 허용. 로터절대 엔코더는 전원사이클마다 배수 "
                         "시프트 가능 — 미검증 프레임 라이브는 원칙 위반. "
                         "벤치(모터 무전원) 시험 전용 (기본 False)")
    ap.add_argument("--frame-max-age", type=float,
                    default=FRAME_MAX_AGE_DEFAULT_S,
                    help="프레임 게이트 허용 최고령(초). 기본 3600")
    ap.add_argument("--checkpoint", default="walk", choices=list(CHECKPOINTS))
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()

    # 프레임 변환 상수 — 하드웨어맵 결함 시 여기서 하드 페일 (드라이런 포함 기동 차단)
    frames = load_frame_map()
    if args.live:
        bad = frames.gain_mismatches()
        if bad and not args.force_gains:
            lines = ", ".join(f"{j}(kp{kp:g}/kd{kd:g})" for j, kp, kd in bad)
            print("[거부] --live 기동 불가: 하드웨어맵 kp/kd가 학습게인"
                  f"(kp{POLICY_KP:g}/kd{POLICY_KD:g})과 불일치 — {lines}\n"
                  "       정책은 학습게인 PD 추종이 전제. 맵 갱신(사용자 결정) 후 "
                  "재시도하거나, 저게인 예비시험 의도라면 --force-gains 를 명시할 것.",
                  file=sys.stderr)
            return 2
        if bad:
            print(f"[경고] --force-gains: 게인 불일치 {len(bad)}/12 관절 상태로 "
                  "라이브 진행 — 학습 분포 밖 거동 주의", file=sys.stderr)
        why = frame_check_verdict(max_age_s=args.frame_max_age)
        if why and not args.force_frame:
            print(f"[거부] --live 기동 불가: 프레임 게이트 — {why}\n"
                  "       미검증 프레임 라이브는 배수 시프트를 그대로 명령하게 "
                  "됨. boot_pose_check 통과 후 재시도하거나, 벤치 시험이면 "
                  "--force-frame 을 명시할 것.", file=sys.stderr)
            return 2
        if why:
            print(f"[경고] --force-frame: {why} 상태로 라이브 진행 — 프레임 "
                  "미보증", file=sys.stderr)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class RlBridgeNode(Node):
        def __init__(self):
            super().__init__("humanoid_rl_bridge")
            self.core = BridgeCore(args.checkpoint)
            self.frames = frames        # 모터↔관절 프레임 변환 (기동 시 검증 완료)
            self.live = bool(args.live)
            self.joint_deg = {}         # {관절: '모터 인코더 프레임' deg} — 변환 전 원본
            self.joint_ts = 0.0
            self.prev_qpos = None
            self.imu = None
            self.imu_ts = 0.0
            self.foot = None
            self.foot_ts = 0.0
            self.baseline = {}
            self.armed = False
            self.stopped_for_stale = False
            self.osc = OscGuard(JOINT_ORDER)
            self.osc_tripped = None
            # 블랙박스: live면 무조건 ~/logs 에 틱 단위 jsonl 기록 (0802 사고 때
            # /tmp 로그가 재부팅으로 소실 — 사후분석 불가 재발 방지)
            self.bb = None
            if self.live:
                d = os.path.expanduser("~/logs")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(
                    d, time.strftime("rl_bridge_%Y%m%d_%H%M%S.jsonl"))
                self.bb = open(path, "a", buffering=1)
                self.get_logger().info(f"블랙박스 기록: {path}")
            self.create_subscription(String, JOINT_STATES_TOPIC, self._on_js, 10)
            self.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
            self.create_subscription(String, IMU_TOPIC, self._on_imu, 50)
            self.create_subscription(String, FOOT_TOPIC, self._on_foot, 50)
            self.create_subscription(String, BRIDGE_CMD_TOPIC, self._on_cmd, 10)
            self.pub_cmd = self.create_publisher(String, COMMAND_TOPIC, 10)
            self.pub_dbg = self.create_publisher(String, BRIDGE_DEBUG_TOPIC, 10)
            self.timer = self.create_timer(TICK_SEC, self._tick)
            self.get_logger().info(
                f"RL 브리지 기동 — {'LIVE' if self.live else 'DRY-RUN'}, "
                f"checkpoint={self.core.ckpt_name}")

        # ---- 입력 콜백 ----
        def _on_js(self, msg):
            """관절각 피드백 (JSON, name→deg) — 값은 '모터 인코더 프레임' deg.

            현재 저장소에 이 토픽 발행자는 없음 (규약 유지용) — 실질 피드백은
            _on_status 의 latest_actual_deg 경로. 두 경로 모두 모터 프레임 deg를
            self.joint_deg 에 저장하고, 관절 프레임 변환은 _tick 에서만 수행.
            """
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            acts = d.get("joints") or d.get("actual_deg_by_joint") or {}
            if not isinstance(acts, dict):
                return
            updated = False
            for k, v in acts.items():
                if k not in JOINT_ORDER:
                    continue
                try:
                    self.joint_deg[k] = float(v)   # 모터 프레임 deg
                except (ValueError, TypeError):
                    continue   # dict/None 등 불량 항목은 개별 스킵 (콜백 사망 방지)
                updated = True
            if updated:
                self.joint_ts = time.time()

        def _on_status(self, msg):
            """Stage8 상태 — armed/baseline 흡수 + 관절 피드백 겸용 소스.

            joints[j].latest_actual_deg(모터 프레임 deg)는 feedback_age_sec <
            STATUS_FEEDBACK_FRESH_SEC(0.5s) 일 때만 신선한 관절각으로 채택.
            joint_ts 는 12관절 전부 신선할 때만 갱신 — 일부만 신선한데 루프가
            계속 도는 안전 구멍 방지 (나머지 관절은 낡은 값이므로 정지가 안전).
            """
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            self.armed = bool(d.get("armed"))
            js = d.get("joints") or {}
            if not isinstance(js, dict):
                return
            fresh = 0
            worst_age = 0.0
            for j in JOINT_ORDER:
                info = js.get(j)
                if not isinstance(info, dict):
                    continue
                b = info.get("baseline_deg")
                if b is not None:
                    try:
                        self.baseline[j] = float(b)   # 모터 프레임 deg
                    except (ValueError, TypeError):
                        pass
                try:
                    age = float(info.get("feedback_age_sec"))
                    val = float(info.get("latest_actual_deg"))
                except (ValueError, TypeError):
                    continue   # 피드백 없음(None 등) — 이 관절은 이번 미갱신
                if age < STATUS_FEEDBACK_FRESH_SEC:
                    self.joint_deg[j] = val           # 모터 프레임 deg
                    fresh += 1
                    worst_age = max(worst_age, age)
            if fresh == len(JOINT_ORDER):
                # 데이터 나이 보정 (적대리뷰 수정): 메시지 '도착' 시각이 아니라
                # 실제 측정 시각으로 기록해야 stale 게이트가 끝단까지 유효하다.
                # 보정 없으면 CAN 두절 후에도 status 도착만으로 최대 ~0.7s 동안
                # 얼어붙은 관절각으로 정책이 계속 계산되는 창이 생긴다.
                self.joint_ts = time.time() - worst_age

        def _on_imu(self, msg):
            try:
                d = json.loads(msg.data)
                gyro = tuple(float(x) for x in d["gyro_rad_s"])
                quat = tuple(float(x) for x in d["quat_wxyz"])
                if len(gyro) != 3 or len(quat) != 4:
                    return   # 형상 불량 — 소비 금지 (콜백 생존이 우선)
                self.imu = (gyro, quat)
                self.imu_ts = float(d.get("timestamp", time.time()))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        def _on_foot(self, msg):
            try:
                d = json.loads(msg.data)
                self.foot = (float(d["left_n"]), float(d["right_n"]))
                self.foot_ts = float(d.get("timestamp", time.time()))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        def _on_cmd(self, msg):
            try:
                d = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            c = d.get("cmd", "")
            if c == "estop":
                self._stop_all("estop")
                self.core.active = False
                return
            if c == "stop":
                self.core.cmd = (0.0, 0.0, 0.0)   # walk 정책 능동 서기 = 정지
                return
            if c in ("stand", "walk", "march"):
                if c in ("stand", "march"):
                    cmd = (0.0, 0.0, 0.0)
                else:
                    try:   # 명령 파싱 실패가 콜백을 죽이면 안 됨 — 거부하고 유지
                        cmd = (float(d.get("vx", 0.3)), float(d.get("vy", 0.0)),
                               float(d.get("wz", 0.0)))
                    except (ValueError, TypeError):
                        self.get_logger().warn(f"walk 명령 vx/vy/wz 파싱 실패 — 무시: {d}")
                        return
                ck = "march" if c == "march" else "walk"
                self.core.reset(checkpoint=ck)
                self.core.cmd = cmd
                self.core.heading_hold = bool(d.get("heading_hold", True))
                self.core.active = True
                self.stopped_for_stale = False
                self.get_logger().info(f"RL {c} 시작 cmd={self.core.cmd}")

        # ---- 50Hz 틱 ----
        def _tick(self):
            now = time.time()
            if not self.core.active:
                return
            joint_stale = SENSOR_STALE_SEC if self.live else JOINT_STALE_DRYRUN_SEC
            fresh = (now - self.joint_ts < joint_stale
                     and now - self.imu_ts < SENSOR_STALE_SEC
                     and now - self.foot_ts < SENSOR_STALE_SEC
                     and len(self.joint_deg) == 12)
            if not fresh:
                if not self.stopped_for_stale:
                    self._stop_all("sensor_stale")
                    self.stopped_for_stale = True
                return
            self.stopped_for_stale = False
            # 입력 변환: joint_deg(모터 인코더 프레임 deg) → 관절 프레임 rad
            #   joint_rad = radians( DIR × (motor_deg − SZ) )
            qpos = [self.frames.motor_deg_to_joint_rad(j, self.joint_deg[j])
                    for j in JOINT_ORDER]
            # qvel: 관절 프레임 위치의 유한차분 (변환된 qpos 기반이라 프레임 자동
            # 상속). AK45(6·12) velocity 디코드 과대판독 때문에 피드백 속도값은
            # 어떤 관절에서도 소비하지 않는다.
            if self.prev_qpos is None:
                qvel = [0.0] * 12
            else:
                qvel = [(a - b) / TICK_SEC for a, b in zip(qpos, self.prev_qpos)]
            self.prev_qpos = qpos
            gyro, quat = self.imu
            contact2 = [1.0 if self.foot[0] > CONTACT_FORCE_THRESHOLD_N else 0.0,
                        1.0 if self.foot[1] > CONTACT_FORCE_THRESHOLD_N else 0.0]
            # 발진 가드 — 피드백 기반이라 정책 출력과 무관하게 몸의 발진을 감지.
            # 트리거 시 STOP_ALL(동결 홀드) 후 비활성: 이후 절차 = 줄 재인장 →
            # release_all.py (0802 프로토콜)
            if self.live and self.osc_tripped is None:
                why = self.osc.update(now, qpos, qvel)
                if why:
                    self.osc_tripped = why
                    if self.bb:
                        self.bb.write(json.dumps(
                            {"timestamp": now, "event": "osc_guard",
                             "reason": why}) + "\n")
                    self._stop_all(f"osc_guard: {why}")
                    self.core.active = False
                    self.get_logger().error(
                        f"발진 가드 트리거 — {why}. 줄 재인장 후 release_all 실행!")
                    return
            targets_rad = self.core.tick(gyro, quat, qpos, qvel, contact2)
            # 출력 변환: 관절 프레임 rad → 모터 인코더 프레임 deg
            #   motor_target_deg = SZ + DIR × degrees(policy_target_rad)
            targets_deg_joint = {}   # URDF 관절 프레임 (진단 표기용)
            targets_deg_motor = {}   # 모터 인코더 프레임 (실제 명령 기준)
            for j, v in zip(JOINT_ORDER, targets_rad):
                targets_deg_joint[j] = math.degrees(float(v))
                targets_deg_motor[j] = self.frames.joint_rad_to_motor_deg(
                    j, float(v))
            dbg = {"timestamp": now, "live": self.live, "armed": self.armed,
                   "checkpoint": self.core.ckpt_name, "cmd": list(self.core.cmd),
                   # 프레임 명시: joint=URDF 관절 프레임 / motor=모터 인코더 프레임
                   "targets_deg_joint": targets_deg_joint,
                   "targets_deg_motor": targets_deg_motor,
                   # 사후분석용 관측 원본 (0802 발진 사고 때 이 데이터가 없었음)
                   "qpos_rad": [round(v, 5) for v in qpos],
                   "qvel_rad_s": [round(v, 4) for v in qvel],
                   "contact": contact2,
                   "foot_n": [round(float(self.foot[0]), 2),
                              round(float(self.foot[1]), 2)]}
            self.pub_dbg.publish(String(data=json.dumps(dbg)))
            if self.bb:
                self.bb.write(json.dumps(dbg) + "\n")
            if not (self.live and self.armed):
                return
            for j in JOINT_ORDER:
                base = self.baseline.get(j)
                if base is None:
                    continue   # baseline 미설정 관절은 노드가 어차피 거부
                # Stage8 절대각 = baseline + rel = motor_target 그대로 복원
                # (baseline이 어떤 자세로 캡처됐든 상쇄되어 무관)
                rel = targets_deg_motor[j] - base
                self.pub_cmd.publish(String(data=json.dumps(
                    {"command": "SET_JOINT_TARGET", "joint": j,
                     "target_deg": rel})))

        def _stop_all(self, reason):
            self.get_logger().warning(f"STOP_ALL ({reason})")
            if self.bb:
                self.bb.write(json.dumps({"timestamp": time.time(),
                                          "event": "stop_all",
                                          "reason": reason}) + "\n")
            if self.live:
                self.pub_cmd.publish(String(
                    data=json.dumps({"command": "STOP_ALL"})))

    rclpy.init()
    node = RlBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
