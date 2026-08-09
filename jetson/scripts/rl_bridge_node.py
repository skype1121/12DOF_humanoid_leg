"""RL 브리지 노드 — 정책 실행기 → Stage8 세트포인트 (실물 이식용, 2026-07-27).

구조 (전부 기검증 조각의 조립):
  센서 구독 → PolicyRunnerStage4(관측245 조립·비트일치 검증본·클램프 없음 —
  0802 발진 수사: 타깃 클램프는 학습 1:1 파괴로 금지, 안전은 stage8 리밋이 담당)
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
#: 발압력 전용 신선 한계 — 10Hz(100ms 주기) 센서라 0.2s 게이트는 1펄스
#: 결손에 트립(0804 5차 스텝테스트 무효화 실증). 0.35s = 3주기 결손까지
#: 허용 — 접촉 이진 관측 350ms 지연은 동적 스텝에서도 허용 범위.
FOOT_STALE_SEC = 0.35
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
    # march_v3 (0802 채터 제거: action_rate 6배): 유의반전율 0.613→0.051, 슬루
    # 포화 스텝 54.7%→0.000%, 컴플라이언스 낙상 0%, 직립 2.8°.
    # ⚠ 0803 공정 재평가: 측정 기반 분포(지연 2~6틱·강성 0.2~·마찰 0.3~)에서
    # 낙상 106회(무낙상 env 33%) — '0%'는 구식 쉬운 분포의 수치였음.
    "march_v3": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-02_19-30-03/exported"),
    # march_v5b (0803 측정 기반 재정합: 실측 루프지연 2~6틱·강성 0.2~1.1·
    # 마찰 0.3~1.2·클록 1.1Hz·착지충격 벌점): play 유의반전 0.027(전 세대
    # 최저=발진 리스크 최소)·낙상 0%·가드재생 0트립, 측정 분포 낙상 23.4%
    # (v3 대비 압도적, 절대치는 개선 과제). 하네스 1.1Hz PASS. 실물 데뷔
    # 1순위 — 검증 사다리(스탠드 확인→march 수 초→연장) 엄수.
    "march_v5b": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-03_03-18-21/exported"),
    # stand_v3 (0803 신규 종목: 조용히 서서 능동 균형 — stage8 조각상 홀드
    # 대체): 측정 DR 하 낙상 3.1%·직립 7.2°·양발접촉 98.3%·가드재생 0트립.
    # 검증 사다리 1단 정책 ("stand" 명령 기본).
    "stand_v3": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-03_06-12-07/exported"),
    # stand_v5 (0803 3차 실기 판정 반영: payload 실측 명시+스텝 회복 합법화+
    # 저용량 관측 현실화): 공정 비교(v5 조건) — 직립 3.54° vs v3 8.72°(2.5배),
    # 낙상 동률 4.7%, 위치 유지 +0.06m vs v3 -0.30m 뒤밀림. 실물 4차 1순위.
    "stand_v5": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-03_18-44-48/exported"),
    # stand_v6 (0804 v6b 클린 재베이크 59899 — 방폭 4중 후 무폭발 완주: 4차 실기
    # 전방 과보상 편향 교정): 낙상 1.6%·직립 2.74°(역대 최고)·양발 97.4%·토크
    # 1.72Nm. 베이크 후반 폭발 전염으로 최종본 대신 붕괴 전 최고점 인양.
    "stand_v6": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-04_11-31-27/exported"),
    # stand_v7 (0804 무개입 낙하 실측 반영: payload '위치' 가정 제거 — 균형 몸
    # 전제 학습): 낙상 1.6%·골반고 완전직립·토크 1.52Nm 최저. v6b의 학습된
    # 전방 편향(균형 실물 위 1초부터 단조 기울임) 제거가 목적 — 실물 6차 1순위.
    "stand_v7": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-04_13-13-49/exported"),
    # stand_v8 (0808 환경 교정 — 학습 환경이 '서 있기만 하기'를 금지하고 있던
    # 것을 되돌린 세대): 푸시 제거·초기교란 ±4.6°→±1.1°·평지 고정·무릎
    # 스탠바이 밴드 강화. v5~v7이 바꾼 축(payload/CoM)은 DR 스윕 0/128 재현으로
    # 기각됐고, 실패의 원인은 '끊임없이 능동 보정하는' 전략을 강제한 외란 설계에
    # 있었다. 실물 절차도 변경: 정책 투입 전 posture_probe 로 평형 자세 정합.
    "stand_v8": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-08_21-24-48/exported"),
    # stand_v9 (0809 캘리브레이션 상수 편향 랜덤화 — 관절 ±3°·IMU ±3°):
    # v8은 영점 오차 3°에서 낙상 44%였다(관측 노이즈가 전부 제로평균이라
    # 상수 편향을 학습한 적이 없었음). v9는 같은 조건에서 16%로 2.7배 개선.
    # 대가: 직립 4.09→9.09°, 낙상 0→3.1%, IMU 3° 오차에서는 오히려 악화.
    # **영점을 실측 검증하기 전까지는 v9가 안전한 선택**이다.
    "stand_v9": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-08-09_03-59-22/exported"),
}

#: 체크포인트별 게이트 클록 주파수 [Hz] — 학습 클록과 반드시 일치 (불일치 =
#: 위상 어긋난 관측 → 보행 파괴). 미등록 = 1.4 (기존 가족 전체의 계약).
#: v5 가족(0803 측정 기반 재정합)부터 1.1Hz — 등록 시 평가 게이트+하네스
#: (--gait-freq 1.1) 통과 필수.
CKPT_GAIT_FREQ = {"march_v5b": 1.1, "stand_v3": 1.1, "stand_v5": 1.1, "stand_v6": 1.1, "stand_v7": 1.1,
                   "stand_v8": 1.1, "stand_v9": 1.1}
CKPT_GAIT_FREQ_DEFAULT = 1.4

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
        self.runner = PolicyRunnerStage4(
            CHECKPOINTS[checkpoint],
            gait_freq=CKPT_GAIT_FREQ.get(checkpoint, CKPT_GAIT_FREQ_DEFAULT))
        self.ckpt_name = checkpoint
        self.hh = HeadingHold(kp=1.0, wz_limit=0.6)
        self.hh_ref_set = False
        self.cmd = (0.0, 0.0, 0.0)
        self.heading_hold = True
        self.active = False          # stand/walk/march 시작 여부

    def reset(self, checkpoint=None):
        if checkpoint and checkpoint != self.ckpt_name:
            self.runner = PolicyRunnerStage4(
                CHECKPOINTS[checkpoint],
                gait_freq=CKPT_GAIT_FREQ.get(checkpoint, CKPT_GAIT_FREQ_DEFAULT))
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
#: 전원컷으로 종료; estop은 ssh 왕복 지연으로 무력) 재발 방지.
#:
#: v2 (최종 점검 후 재조정): 구버전(반전6회+2°)은 march_v3 '컴플라이언스'
#: 시뮬 궤적 재생에서 30초에 85회 오트립 — 실물 보행을 상시 차단할 파라미터
#: 였다. 데이터 기반 재튜닝: 합성 발진 클래스(주기-2 채터 0.7°/틱~,
#: 6/8/12/16Hz 사인) 전부 포착 + march_v3 play·compliant 30초 오트립 0 +
#: 실물 발진 파형 포착을 동시에 만족하는 3중 모드.
#: 커버리지 경계(문서화): 4Hz 미만 대진폭 요동은 보행 3고조파(1.4Hz×3)와
#: 겹쳐 가드 범위 밖 — 운용자 estop·줄 인장 영역.
OSC_VEL_HARD = 12.0      #: rad/s 즉시 트리거 (정상 보행 스윙 3~6 rad/s의 2배+)
OSC_VEL_DEADBAND = 0.5   #: rad/s — 이하 속도의 부호는 무시 (양자화·유한차분 노이즈)

#: 온셋 블렌딩 (0803 stand_v3 2차: 진입 0.8s에 힙 타깃 24° 휘두름 → 9Hz 링잉
#: 기폭. 현재 실측 자세→정책 타깃 크로스페이드로 진입 과도 제거)
#:
#: ⚠ **1.5 → 0.0 (0810 시뮬 실측).** 이 크로스페이드가 다섯 세대 실패의
#: 최대 단일 요인이었다. 기전: 명령이 `착지자세·(1−α) + 정책타깃·α` 라서
#: **정책의 보정량까지 α배로 깎인다**. 0.5초 시점 α=0.26 — 정책이 자기
#: 보정의 26% 밖에 못 낸다. 그 사이 로봇은 착지 자세에 붙들려 있고,
#: 그 자세가 안 서는 자세면 넘어진다. 권한을 되찾을 땐 이미 늦다.
#:
#: 측정 (v6b·64env·DR켬·LPF 0.27 동일, `eval_drift_dr.py --blend/--blend_mode`):
#:     블렌딩 2.5s   58/64 낙상   −24.6°/s
#:     블렌딩 1.5s   47/64        −18.2      ← 종전 값
#:     블렌딩 0.8s   25/64        − 5.5
#:     블렌딩 0.4s   11/64        − 1.2
#:     **블렌딩 없음(슬루 인수)  3/64   − 0.8**   ← 채택
#:     (참고) 래퍼 전혀 없음      1/64   − 0.2
#: 정책 무관이다 — 네 세대 전부 블렌딩에서 47~55/64 로 무너졌다.
#: 실측 −18.2°/s 는 0809 실기 강체 낙하 −19°/s 와 거의 일치한다.
#:
#: **그럼 0803 링잉은 무엇이 막고 있나 — LPF 다, 블렌딩이 아니다.**
#: >5Hz 에너지 비율:  래퍼 없음 25.6% → LPF만 13.3% → LPF+블렌딩 12.4%
#:                  → **LPF+슬루(블렌딩 제거) 12.3%**
#: 블렌딩의 발진 억제 기여는 0 이다. E_smth 는 오히려 블렌딩이 악화시킨다
#: (0.41 vs 0.16). 완화책 5종 중 이 한 종만 순손실이었다.
#:
#: **진입 과도는 stage8 이 이미 막는다.** stage8 제어루프는 매 틱 모든 관절에
#: `_step_toward` 로 4°/틱 슬루를 건다(MAX_TARGET_STEP_DEG_PER_TICK, 646행).
#: 인수 시점의 `commanded_relative_deg_by_joint` 는 hang_stand 가 남긴 착지
#: 명령이므로, 블렌딩을 빼도 명령은 착지 자세에서 4°/틱으로만 출발한다.
#: 즉 브리지 크로스페이드는 **한 층 아래에서 이미 풀린 문제를 위해 덧댄 것**.
#:
#: ══ 철회 (0810 실물 블랙박스 재분석) ══════════════════════════════════
#: 위 시뮬 근거로 1.5→0 수술했다가 **되돌린다**. 실물 로그가 전제를 무너뜨렸다.
#:
#: bb_5차_v6b_무개입낙하_20260804_130927.jsonl 을 쿼터니언 부호 교정 후 재분석:
#:     t 0.00~0.62s (alpha 0→0.37)   앞뒤 −1.7° **정지·안정**
#:     t 0.82s      (alpha 0.57)     −0.47
#:     t 1.22s      (alpha 0.91)     +6.20
#:     t 3.82s      (alpha 1.00)     +13.15
#: **전방 이동이 alpha 0.4~0.6 을 넘는 순간 시작된다.** 즉 실물에서 블렌딩은
#: 낙상을 일으킨 것이 아니라 **정책의 전방 구동을 0.8초간 억제하고 있었다.**
#:
#: 시뮬이 정반대로 나온 이유: 시뮬은 착지 자세(전 관절 0)에서 뒤로 넘어진다
#: (−4.55°/s). 그래서 블렌딩이 그 자세에 붙들면 낙상한다. 그런데 **실물은
#: 그 자세에서 안정**하다(0804 실측 −1.7° 정지, 트림0 자립 3회). 전제가
#: 다르므로 결론이 이전되지 않는다. 블렌딩을 빼면 정책이 t=0 부터 전방
#: 구동을 시작해 **실물은 오히려 더 빨리 넘어질 것**으로 예상된다.
#:
#: (0809 강체 홀드가 −19°/s 로 뒤로 넘어진 것은 0804 와 모순된다. 그 사이
#:  m8 무전원사이클 시프트·재앵커·hwmap v13→v14 가 있었다 — 0809 낙하는
#:  영점 오염 가능성이 크고, 거기서 역산한 CoM −31.6mm 도 그만큼 의심스럽다.)
#:
#: 남은 진짜 문제: **정책이 권한을 얻으면 앞으로 민다.** 래퍼가 아니라 정책이다.
#: ═══════════════════════════════════════════════════════════════════
BLEND_S = 1.5
#: 리프트 가드 — 0.6s (4차 실기: 0.4s가 전방 낙하 중 회복 스텝의 양발
#: 순간 무접촉을 중단시킴 → 스텝 완료 여유 부여. 들림 발버둥 차단은 유지)
#: (0803 실측: 재인장으로 들리자 접촉 [0,0] → 정책 공중 발버둥.
#: 서기/제자리걸음 문맥에서 양발 무접촉 지속 = 들림·완전 공중 → 동결이 안전)
LIFT_STOP_S = 0.6


def blend_alpha(elapsed_s, dur_s=BLEND_S):
    """온셋 블렌딩 가중치 — 스무스스텝 0→1 (양 끝 기울기 0, 단조증가).

    BLEND_S=0 (현행 기본) 이면 크로스페이드는 비활성이고 이 함수는 롤백
    경로·자기검사용으로만 남는다. 0 나눗셈은 즉시 1.0 으로 처리한다.
    """
    if dur_s <= 0.0:
        return 1.0
    a = min(1.0, max(0.0, elapsed_s / dur_s))
    return a * a * (3.0 - 2.0 * a)


#: stand 계열 타깃 저역필터 (0803 발진 분석: 지연 80~100ms 루프에서 5~9Hz
#: 정책 주도 리밋사이클 — 발진 대역 루프게인 컷. 서기 보정은 <2Hz라 손실 미미.
#: EMA 1차: beta=0.27 ≈ 차단 2.5Hz@50Hz. march/walk에는 미적용 (보행 리듬 보존).
STAND_LPF_BETA = 0.27


def lpf_step(state, x, beta=STAND_LPF_BETA):
    """EMA 1차 저역필터 한 스텝 — state None이면 x로 초기화 (점프 없음)."""
    if state is None:
        return list(x)
    return [s + beta * (v - s) for s, v in zip(state, x)]

#: 모드1 '채터': 인접 틱 속도 교대(양쪽 |v|≥CH_VMIN) ≥8회/0.5s — 주기-2
#: 포화 발진(2차 사고 클래스, 스텝 0.7°/틱부터)을 0.2s 내 포착.
OSC_CH_WINDOW = 0.5
OSC_CH_COUNT = 8
OSC_CH_VMIN = 0.55       #: rad/s (0.7°/틱 주기-2 = ±0.61 rad/s)
#: 모드2 '중주파': 속도 부호이벤트 반전(스텝>MF_STEP) ≥9회/0.5s + 진폭≥2.5°
#: — 8~25Hz 성장형 발진(1차 사고 클래스).
OSC_MF_WINDOW = 0.5
OSC_MF_COUNT = 9
OSC_MF_P2P = math.radians(2.5)
OSC_MF_STEP = math.radians(0.7)   #: 반전 카운트 최소 스텝 — 보행 지터 배제
#: 모드3 '저주파 대진폭': 반전 ≥6회/0.6s + 진폭≥5° + 스텝>1.5° — 6Hz대
#: 대진폭 리밋사이클.
OSC_LF_WINDOW = 0.6
OSC_LF_COUNT = 6
OSC_LF_P2P = math.radians(5.0)
OSC_LF_STEP = math.radians(1.5)


class OscGuard:
    """관절 발진 감지 v2 — 3중 모드 (순수 로직, ROS 비의존 — selftest 대상).

    피드백 기반: 정책 출력과 무관하게 몸의 발진을 감지한다.
      과속:  |qvel| > 12 rad/s 즉시.
      채터:  인접 틱 속도 교대(양쪽 |v|≥0.55) ≥8회/0.5s — 주기-2 포화 발진.
      중주파: 부호이벤트 반전(스텝>0.7°) ≥9회/0.5s + 진폭≥2.5° — 8~25Hz 성장형.
      저주파: 반전(스텝>1.5°) ≥6회/0.6s + 진폭≥5° — 6Hz대 대진폭.
    march_v3 play·compliant 시뮬 30초 재생 오트립 0 검증됨 (2026-08-02).
    update(t, qpos, qvel) → None(정상) 또는 사유 문자열(트리거).
    """

    def __init__(self, joint_names):
        self.names = list(joint_names)
        self.prev_v = {j: 0.0 for j in self.names}
        self.alt = {j: collections.deque() for j in self.names}
        self.last_sign = {j: 0 for j in self.names}
        self.rev_mf = {j: collections.deque() for j in self.names}
        self.rev_lf = {j: collections.deque() for j in self.names}
        self.pos = {j: collections.deque() for j in self.names}
        self.prev_q = {j: None for j in self.names}
        self._wmax = max(OSC_CH_WINDOW, OSC_MF_WINDOW, OSC_LF_WINDOW)

    def update(self, t, qpos, qvel):
        for i, j in enumerate(self.names):
            v = float(qvel[i])
            q = float(qpos[i])
            if abs(v) > OSC_VEL_HARD:
                return f"과속 {j} {v:+.1f} rad/s (한계 {OSC_VEL_HARD})"
            # 모드1 채터: 인접 틱 속도 교대
            pv = self.prev_v[j]
            if abs(v) >= OSC_CH_VMIN and abs(pv) >= OSC_CH_VMIN and v * pv < 0:
                a = self.alt[j]
                a.append(t)
                while a and t - a[0] > OSC_CH_WINDOW:
                    a.popleft()
                if len(a) >= OSC_CH_COUNT:
                    return (f"채터 {j}: {OSC_CH_WINDOW}s 내 인접틱 속도교대 "
                            f"{len(a)}회 (|v|≥{OSC_CH_VMIN})")
            self.prev_v[j] = v
            # 위치 창 (중·저주파 진폭 판정용)
            p = self.pos[j]
            p.append((t, q))
            while p and t - p[0][0] > self._wmax:
                p.popleft()
            step = abs(q - self.prev_q[j]) if self.prev_q[j] is not None else 0.0
            self.prev_q[j] = q
            s = 0 if abs(v) < OSC_VEL_DEADBAND else (1 if v > 0 else -1)
            if s == 0:
                continue
            if self.last_sign[j] != 0 and s != self.last_sign[j]:
                # 모드2 중주파
                if step > OSC_MF_STEP:
                    r = self.rev_mf[j]
                    r.append(t)
                    while r and t - r[0] > OSC_MF_WINDOW:
                        r.popleft()
                    if len(r) >= OSC_MF_COUNT:
                        vals = [x for ts_, x in p if t - ts_ <= OSC_MF_WINDOW]
                        p2p = max(vals) - min(vals)
                        if p2p >= OSC_MF_P2P:
                            return (f"발진 {j}: {OSC_MF_WINDOW}s 내 반전 "
                                    f"{len(r)}회, 진폭 {math.degrees(p2p):.1f}°")
                # 모드3 저주파 대진폭
                if step > OSC_LF_STEP:
                    r = self.rev_lf[j]
                    r.append(t)
                    while r and t - r[0] > OSC_LF_WINDOW:
                        r.popleft()
                    if len(r) >= OSC_LF_COUNT:
                        vals = [x for _, x in p]
                        p2p = max(vals) - min(vals)
                        if p2p >= OSC_LF_P2P:
                            return (f"저주파발진 {j}: {OSC_LF_WINDOW}s 내 반전 "
                                    f"{len(r)}회, 진폭 {math.degrees(p2p):.1f}°")
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
        run_ts = float(j.get("ts"))
        ages = {m: now - float(mts[str(m)]) for m in range(1, 13)}
    except (TypeError, ValueError):
        return "저널 ts/motor_ts 형식 이상 — boot_pose_check 재실행으로 재생성"
    # 최신 판정에서의 전수 검증 요구 (적대리뷰 A5): 저널은 '✓ 일치' 모터만 병합
    # 갱신하므로, 최근 판정에서 어긋난 모터는 옛 ts를 유지한 채 신선해 보일 수
    # 있음. 모든 모터가 '마지막 실행'(j.ts)에서 검증됐을 것을 요구해 차단.
    stale_run = [m for m in range(1, 13) if float(mts[str(m)]) < run_ts - 1.0]
    if stale_run:
        return (f"모터 {stale_run} 가 최신 판정에서 미검증 (직전 실행에서 시프트/"
                f"무응답 의심) — boot_pose_check 12/12 재통과 필요")
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
    p = write({"ts": now - 60.0, "pos_deg": {}, "motor_ts": fresh})
    cases.append(("신선 12/12 통과",
                  frame_check_verdict(p, 3600.0, now) is None))
    stale = dict(fresh)
    stale["7"] = now - 7200.0
    p = write({"ts": now - 7200.0, "motor_ts": stale})
    v = frame_check_verdict(p, 3600.0, now)
    cases.append(("만료 거부", v is not None and "모터7" in v))
    p = write({"ts": now - 60.0,
               "motor_ts": {k: v for k, v in fresh.items() if k != "11"}})
    v = frame_check_verdict(p, 3600.0, now)
    cases.append(("결측 거부", v is not None and "11" in v))
    v = frame_check_verdict("/nonexistent/journal.json", 3600.0, now)
    cases.append(("저널 없음 거부", v is not None and "없음" in v))
    p = write({"ts": now - 60.0, "motor_ts": {**fresh, "3": "abc"}})
    cases.append(("형식 이상 거부",
                  frame_check_verdict(p, 3600.0, now) is not None))
    # A5 재현: 최신 판정(ts=now-60)에서 m6만 미검증(옛 ts 유지, 아직 60분 이내)
    merged = dict(fresh)
    merged["6"] = now - 1800.0
    p = write({"ts": now - 60.0, "motor_ts": merged})
    v = frame_check_verdict(p, 3600.0, now)
    cases.append(("최신판정 미검증 거부", v is not None and "6" in v and "최신" in v))
    # ts 자체가 없는 구형 저널 → 형식 이상 거부
    p = write({"motor_ts": fresh})
    cases.append(("ts 결측 거부", frame_check_verdict(p, 3600.0, now) is not None))
    bad = [n for n, ok in cases if not ok]
    if bad:
        raise AssertionError(f"프레임 게이트 케이스 실패: {bad}")
    return f"판정 케이스 {len(cases)}/{len(cases)} 통과"


def _selftest_stand_lpf():
    import cmath
    # 초기화 = 무점프
    s = lpf_step(None, [1.0] * 12)
    assert s == [1.0] * 12
    # 주파수 응답: 5Hz 강감쇠 / 0.5Hz 근통과 (50Hz 샘플링, 정착 후 진폭 측정)
    def gain(freq):
        st = None
        out = []
        for k in range(400):
            t = k / 50.0
            x = math.sin(2 * math.pi * freq * t)
            st = lpf_step(st, [x])
            out.append(st[0])
        tail = out[200:]
        return max(abs(v) for v in tail)
    g5, g05 = gain(5.0), gain(0.5)
    assert g5 < 0.5, g5          # 발진 대역 절반 이하
    assert g05 > 0.9, g05        # 균형 보정 대역 보존
    # 수렴: 스텝 입력 1.0 → 20틱 후 0.95 이상
    st = lpf_step(None, [0.0])
    for _ in range(20):
        st = lpf_step(st, [1.0])
    assert st[0] > 0.95, st[0]
    return (f"EMA beta={STAND_LPF_BETA}: 5Hz 이득 {g5:.2f}<0.5 / "
            f"0.5Hz {g05:.2f}>0.9 / 무점프 초기화·수렴 OK")


def _selftest_blend():
    # ① 크로스페이드 수학은 롤백 경로로 보존 — 명시 dur_s 로 검사한다
    #    (BLEND_S 를 쓰면 기본값 0 에서 검사 자체가 무의미해진다)
    D = 1.5
    a0, a_mid, a1 = blend_alpha(0.0, D), blend_alpha(D / 2, D), blend_alpha(D, D)
    assert a0 == 0.0 and a1 == 1.0, (a0, a1)
    assert abs(a_mid - 0.5) < 1e-9, a_mid
    assert blend_alpha(D * 2, D) == 1.0               # 과경과 클램프
    assert blend_alpha(-1.0, D) == 0.0                # 음수 클램프
    prev = -1.0
    for k in range(0, 151):
        a = blend_alpha(k * 0.01, D)
        assert a >= prev, (k, a, prev)                # 단조증가
        prev = a
    assert blend_alpha(0.02, D) < 0.002               # 양 끝 기울기 0 (부드러운 출발)
    assert 1.0 - blend_alpha(D - 0.02, D) < 0.002
    frm, tgt = 10.0, 20.0
    mix = frm * (1 - a_mid) + tgt * a_mid
    assert abs(mix - 15.0) < 1e-9
    # ② 현행 기본 경로: 크로스페이드 비활성 = 첫 틱부터 정책 권한 100%
    #    (0810 수술. 0 나눗셈 없이 즉시 1.0 이어야 한다)
    assert blend_alpha(0.0, 0.0) == 1.0
    assert blend_alpha(-1.0, 0.0) == 1.0
    if BLEND_S <= 0.0:
        assert blend_alpha(0.0) == 1.0, "기본 경로가 α<1 을 내면 권한이 깎인다"
    # ③ 인수 안전은 stage8 슬루가 담당한다는 계약 명시 (4°/틱)
    #    브리지가 크로스페이드를 안 걸어도 명령은 착지 자세에서 4°/틱로 출발.
    step, gap = 4.0, 24.0
    assert math.ceil(gap / step) == 6, "24° 이탈은 6틱(120ms)에 도달 — 과도 아님"
    return (f"스무스스텝 롤백경로 정상 + 기본경로 α≡1 (BLEND_S={BLEND_S}s, "
            "인수 슬루는 stage8 4°/틱)")


def _selftest_osc_guard():
    def run_seq(qseq, dur_limit=None):
        """위치 시퀀스를 브리지와 동일한 유한차분 qvel로 재생."""
        g = OscGuard(["j"])
        prev = None
        for k, q in enumerate(qseq):
            t = k * TICK_SEC
            v = 0.0 if prev is None else (q - prev) / TICK_SEC
            prev = q
            trig = g.update(t, [q], [v])
            if trig:
                return trig, t
        return None, len(qseq) * TICK_SEC

    def sine(freq_hz, amp_deg, dur_s):
        amp = math.radians(amp_deg)
        w = 2 * math.pi * freq_hz
        return [amp * math.sin(w * k * TICK_SEC)
                for k in range(int(dur_s / TICK_SEC))]

    # 채터 (주기-2, 0.9°/틱 — 2차 사고 클래스): 0.5s 내 트리거
    q, step = [0.0], math.radians(0.9)
    for k in range(1, 100):
        q.append(q[-1] + (step if k % 2 else -step))
    trig0, t0 = run_seq(q)
    assert trig0 and "채터" in trig0 and t0 <= 0.5, (trig0, t0)
    trig, t = run_seq(sine(8.0, 3.0, 2.0))      # 8Hz±3° (1차 사고 클래스)
    assert trig and "발진" in trig and t <= 1.0, (trig, t)
    trig2, _ = run_seq(sine(1.4, 20.0, 3.0))    # 정상 보행 리듬 — 미트리거
    assert trig2 is None, trig2
    trig3, _ = run_seq(sine(12.0, 0.15, 2.0))   # 미세 노이즈 — 미트리거
    assert trig3 is None, trig3
    g = OscGuard(["j"])                          # 과속 즉시 트리거
    trig4 = g.update(0.0, [0.0], [15.0])
    assert trig4 and "과속" in trig4, trig4
    return (f"주기2채터 {t0:.2f}s / 8Hz±3° {t:.2f}s 트리거 / 보행·노이즈 "
            f"미트리거 / 과속 즉시 — 5케이스 통과")


def run_selftest():
    """드라이런 자가검증 — ROS·로봇 불필요. 프레임 변환 4종 + 정책 500틱."""
    tests = [("하드웨어맵", _selftest_hwmap),
             ("온셋 블렌딩", _selftest_blend),
             ("stand LPF", _selftest_stand_lpf),
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
        if not (math.isfinite(args.frame_max_age) and args.frame_max_age > 0):
            print(f"[거부] --frame-max-age 비정상: {args.frame_max_age}",
                  file=sys.stderr)
            return 2
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
            self.qpos_hold_ticks = [1] * 12
            self._spike_count = {}   # 관절별 연속 스파이크 카운트 (필터)
            self._last_raw_val = {}  # 관절별 직전 수신 원시값 (새 샘플 판별)
            self.imu = None
            self.imu_ts = 0.0
            self.foot = None
            self.foot_ts = 0.0
            self.baseline = {}
            self.armed = False
            self.stopped_for_stale = False
            self.osc = OscGuard(JOINT_ORDER)
            self.osc_tripped = None
            self.blend_t0 = None
            self.blend_from = None
            self.nocontact_t0 = None
            self.stage8_cmd_rel = {}
            self.tgt_filt = None
            self.osc_tgt = OscGuard(JOINT_ORDER)
            self.prev_tgt = None
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
                cr = info.get("commanded_relative_deg")
                if cr is not None:
                    try:
                        # 블렌드 시작점용: stage8가 실제 스트리밍 중인 명령
                        # (baseline+rel) — 실측 시작 대비 토크 연속 인수 (리뷰 F2)
                        self.stage8_cmd_rel[j] = float(cr)
                    except (ValueError, TypeError):
                        pass
                try:
                    age = float(info.get("feedback_age_sec"))
                    val = float(info.get("latest_actual_deg"))
                except (ValueError, TypeError):
                    continue   # 피드백 없음(None 등) — 이 관절은 이번 미갱신
                if age < STATUS_FEEDBACK_FRESH_SEC:
                    # 스파이크 필터 (0802 march 데뷔 트립 교훈): 한 갱신에 30°+
                    # 점프는 물리 불가(모터 최대 ~20rad/s×20ms≈23°) — 쓰레기
                    # 프레임(명령 오귀속 등)이 과속 가드를 오발시키는 것을 차단.
                    # 적대검수 0803: '3연속'은 status 메시지가 아니라 '새 피드백
                    # 샘플' 기준이어야 한다 — status는 라이브 중 ~700Hz(명령
                    # 수신마다 발행)라 같은 쓰레기 값이 ~14회 반복 노출돼 4ms
                    # 만에 3연속을 오달성했다. 값이 직전 수신값과 다를 때만
                    # 새 샘플로 계수한다.
                    prev = self.joint_deg.get(j)
                    if prev is not None and abs(val - prev) > 30.0:
                        last_raw = self._last_raw_val.get(j)
                        if last_raw is None or val != last_raw:
                            self._spike_count[j] = self._spike_count.get(j, 0) + 1
                        self._last_raw_val[j] = val
                        if self._spike_count[j] < 3:
                            fresh += 1        # 직전값 유지 — 신선도는 인정
                            worst_age = max(worst_age, age)
                            continue
                    self._spike_count[j] = 0
                    self._last_raw_val[j] = val
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
            if not isinstance(d, dict):
                # 비-dict 유효 JSON("estop" 등)이 콜백을 죽여 라이브 중 브리지
                # 사망(STOP_ALL 없이)하는 경로 차단 (적대리뷰 A6)
                self.get_logger().warn(f"명령이 JSON 객체가 아님 — 무시: {msg.data[:60]}")
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
                # 체크포인트 선택은 기동 플래그(--checkpoint)를 존중 (적대리뷰 A4:
                # 종전 하드코딩은 신규 체크포인트를 도달불가로 만들어 '배제한
                # 구정책'이 무언 교체 실행되는 사고 경로였음 — march_v3에서 재발해
                # 가족 기반으로 구조화). 이름이 march*면 march 가족, 아니면 walk
                # 가족. 명령과 가족이 어긋나면 각 가족의 검증된 최신판으로.
                march_family = {k for k in CHECKPOINTS if k.startswith("march")}
                stand_family = {k for k in CHECKPOINTS if k.startswith("stand")}
                if c == "stand":
                    # 0803 부활: stand_v3 전용 정책 등록 (능동 균형 서기 —
                    # 낙상 3.1%·가드재생 0트립·양발접촉 98.3%). 검증 사다리의
                    # 1단: march 전에 stand 제로샷 확인.
                    ck = args.checkpoint if args.checkpoint in stand_family \
                        else "stand_v7"
                elif c == "march":
                    ck = args.checkpoint if args.checkpoint in march_family \
                        else "march_v5b"
                else:
                    ck = args.checkpoint \
                        if args.checkpoint in (
                            set(CHECKPOINTS) - march_family - stand_family) \
                        else "drv2"
                # 라이브 (재)시작 = 가드·미분 상태 전부 리셋 (적대리뷰 A1·A2:
                # 종전엔 osc_tripped 래치로 재개 시 가드 영구 사망 + prev_qpos
                # 동결로 복귀 첫 틱 qvel 폭주)
                self.osc = OscGuard(JOINT_ORDER)
                self.osc_tripped = None
                self.prev_qpos = None
                self.blend_t0 = None
                self.blend_from = None
                self.nocontact_t0 = None
                self.tgt_filt = None
                self.osc_tgt = OscGuard(JOINT_ORDER)
                self.prev_tgt = None
                self.core.reset(checkpoint=ck)
                self.core.cmd = cmd
                # stand 계열은 heading_hold 강제 OFF (0803 발진 분석: stand_v3
                # 학습 명령은 전 에피소드 (0,0,0) — HeadingHold의 wz≠0 주입은
                # 학습에 없던 입력 채널이자 요 요동과 상관된 추가 피드백 루프)
                if ck in stand_family:
                    self.core.heading_hold = False
                else:
                    self.core.heading_hold = bool(d.get("heading_hold", True))
                self.core.active = True
                self.stopped_for_stale = False
                self.get_logger().info(f"RL {c} 시작 ckpt={ck} cmd={self.core.cmd}")

        # ---- 50Hz 틱 ----
        def _tick(self):
            now = time.time()
            if not self.core.active:
                return
            joint_stale = SENSOR_STALE_SEC if self.live else JOINT_STALE_DRYRUN_SEC
            fresh = (now - self.joint_ts < joint_stale
                     and now - self.imu_ts < SENSOR_STALE_SEC
                     and now - self.foot_ts < FOOT_STALE_SEC
                     and len(self.joint_deg) == 12)
            if not fresh:
                if not self.stopped_for_stale:
                    self._stop_all("sensor_stale")
                    self.stopped_for_stale = True
                    # 최종점검: 종전엔 신선도 복귀 시 정책이 무인 자동재개 —
                    # 공백 원인(버스 이상 등) 미확인 상태의 재개는 위험.
                    # 비활성화하고 운용자의 march 재명령(=가드·상태 전체 리셋
                    # 경로)을 요구한다.
                    self.core.active = False
                    self.get_logger().error(
                        "센서 공백 → STOP_ALL + 정책 비활성. 원인 확인 후 "
                        "march 재명령으로만 재개")
                # 공백 복귀 첫 틱의 유한차분 qvel 폭주 방지 (적대리뷰 A2):
                # 동결된 prev_qpos로 (공백 동안 이동량)/TICK을 계산하면 가드
                # 오탐 트립 또는 분포 밖 관측이 정책에 유입됨
                self.prev_qpos = None
                return
            self.stopped_for_stale = False
            # 입력 변환: joint_deg(모터 인코더 프레임 deg) → 관절 프레임 rad
            #   joint_rad = radians( DIR × (motor_deg − SZ) )
            qpos = [self.frames.motor_deg_to_joint_rad(j, self.joint_deg[j])
                    for j in JOINT_ORDER]
            # 적대검수 0803: 비유한값(NaN/inf) 게이트 — NaN은 발진가드의 모든
            # 비교(>, <)를 False로 통과해 가드를 완전 블라인드로 만든다.
            # 어떤 입력이든 비유한이면 센서 고장으로 보고 즉시 동결 정지.
            gyro, quat = self.imu
            flat = list(qpos) + list(gyro) + list(quat) + [
                float(self.foot[0]), float(self.foot[1])]
            if not all(math.isfinite(v) for v in flat):
                self._stop_all("nonfinite_observation")
                self.core.active = False
                self.get_logger().error(
                    "비유한 관측(NaN/inf) — STOP_ALL + 정책 비활성. 센서 점검 후 "
                    "march 재명령으로만 재개")
                return
            # qvel: 관절 프레임 위치의 유한차분 (변환된 qpos 기반이라 프레임 자동
            # 상속). AK45(6·12) velocity 디코드 과대판독 때문에 피드백 속도값은
            # 어떤 관절에서도 소비하지 않는다.
            if self.prev_qpos is None:
                qvel = [0.0] * 12
                self.qpos_hold_ticks = [1] * 12
            else:
                # 최종점검: 라운드로빈 폴링으로 관절값이 수 틱 동안 안 변하다
                # 한 번에 갱신되면 Δq/TICK이 실제 속도의 N배로 뻥튀기 —
                # 관절별 '값이 머문 틱 수'로 나눠 보정한다.
                qvel = []
                for k, (a, b) in enumerate(zip(qpos, self.prev_qpos)):
                    if a == b:
                        # 적대검수 0803: 캡 12틱(=라운드로빈 최장 240ms) — 캡
                        # 없으면 '진짜 정지'가 홀드로 오인 누적돼, 이후 실제
                        # 대점프의 qvel을 N분의 1로 희석해 과속 가드를 눈멀게
                        # 한다.
                        self.qpos_hold_ticks[k] = min(
                            self.qpos_hold_ticks[k] + 1, 12)
                        qvel.append(0.0)
                    else:
                        qvel.append((a - b) / (TICK_SEC * self.qpos_hold_ticks[k]))
                        self.qpos_hold_ticks[k] = 1
            self.prev_qpos = qpos
            gyro, quat = self.imu
            contact2 = [1.0 if self.foot[0] > CONTACT_FORCE_THRESHOLD_N else 0.0,
                        1.0 if self.foot[1] > CONTACT_FORCE_THRESHOLD_N else 0.0]
            # 리프트 가드 — 양발 무접촉 LIFT_STOP_S 지속 시 동결 (발버둥 차단)
            if self.live and contact2[0] == 0.0 and contact2[1] == 0.0:
                if self.nocontact_t0 is None:
                    self.nocontact_t0 = now
                elif now - self.nocontact_t0 >= LIFT_STOP_S:
                    self._bb_write({"timestamp": now, "event": "lift_guard"})
                    self._stop_all(f"lift_guard: 양발 무접촉 {LIFT_STOP_S}s 지속")
                    self.core.active = False
                    self.get_logger().error(
                        "리프트 가드 — 양발 무접촉 지속, 동결 홀드. 줄·발 확인!")
                    return
            else:
                self.nocontact_t0 = None
            # 발진 가드 — 피드백 기반이라 정책 출력과 무관하게 몸의 발진을 감지.
            # 트리거 시 STOP_ALL(동결 홀드) 후 비활성: 이후 절차 = 줄 재인장 →
            # release_all.py (0802 프로토콜)
            if self.live and self.osc_tripped is None:
                why = self.osc.update(now, qpos, qvel)
                if why:
                    self.osc_tripped = why
                    self._bb_write({"timestamp": now, "event": "osc_guard",
                                    "reason": why})
                    self._stop_all(f"osc_guard: {why}")
                    self.core.active = False
                    self.get_logger().error(
                        f"발진 가드 트리거 — {why}. 줄 재인장 후 release_all 실행!")
                    return
            targets_rad = self.core.tick(gyro, quat, qpos, qvel, contact2)
            if self.core.ckpt_name.startswith("stand"):
                self.tgt_filt = lpf_step(self.tgt_filt, targets_rad)
                targets_rad = self.tgt_filt
            # 타깃 측 발진 가드 (0803 분석: 두 발진 모두 타깃이 몸을 80~100ms
            # 선행 — 정책 출력 감시가 절반 진폭·0.3~1.2s 조기 컷. 러너 슬루
            # 4°/틱=3.5rad/s 상한이라 정상 타깃은 과속 트립 불가)
            if self.live and self.osc_tripped is None:
                tvel = ([0.0] * 12 if self.prev_tgt is None else
                        [(x - p) / TICK_SEC
                         for x, p in zip(targets_rad, self.prev_tgt)])
                self.prev_tgt = list(targets_rad)
                why_t = self.osc_tgt.update(now, targets_rad, tvel)
                if why_t:
                    self.osc_tripped = why_t
                    self._bb_write({"timestamp": now, "event": "osc_guard_tgt",
                                    "reason": why_t})
                    self._stop_all(f"osc_guard_tgt: {why_t}")
                    self.core.active = False
                    self.get_logger().error(
                        f"타깃 발진 가드 — 정책 출력 발진 {why_t}. "
                        "줄 재인장 후 release_all!")
                    return
            else:
                self.prev_tgt = list(targets_rad)
            # 출력 변환: 관절 프레임 rad → 모터 인코더 프레임 deg
            #   motor_target_deg = SZ + DIR × degrees(policy_target_rad)
            targets_deg_joint = {}   # URDF 관절 프레임 (진단 표기용)
            targets_deg_motor = {}   # 모터 인코더 프레임 (실제 명령 기준)
            for j, v in zip(JOINT_ORDER, targets_rad):
                targets_deg_joint[j] = math.degrees(float(v))
                targets_deg_motor[j] = self.frames.joint_rad_to_motor_deg(
                    j, float(v))
            # 온셋 블렌딩: 실측 현재각(모터 프레임) → 정책 타깃 크로스페이드.
            # targets_deg_joint(진단)는 정책 원출력 유지, targets_deg_motor
            # (실명령·블랙박스)만 블렌딩 — 명령 vs 실측 사후분석 정합.
            if self.blend_t0 is None:
                self.blend_t0 = now
                # 시작점 = stage8가 스트리밍 중인 명령(baseline+rel) — 실측으로
                # 잡으면 추종오차만큼 토크가 풀려 하중 관절이 ~1° 침하 (리뷰 F2).
                # 명령값 미보유 관절만 실측 폴백.
                self.blend_from = {}
                for j in JOINT_ORDER:
                    base = self.baseline.get(j)
                    rel = self.stage8_cmd_rel.get(j)
                    self.blend_from[j] = (base + rel
                                          if base is not None and rel is not None
                                          else self.joint_deg[j])
            alpha = 1.0 if BLEND_S <= 0.0 else blend_alpha(now - self.blend_t0)  # noqa: E501
            if alpha < 1.0:
                for j in JOINT_ORDER:
                    targets_deg_motor[j] = (self.blend_from[j] * (1.0 - alpha)
                                            + targets_deg_motor[j] * alpha)
            dbg = {"timestamp": now, "live": self.live, "armed": self.armed,
                   "checkpoint": self.core.ckpt_name, "cmd": list(self.core.cmd),
                   # 프레임 명시: joint=URDF 관절 프레임 / motor=모터 인코더 프레임
                   "targets_deg_joint": targets_deg_joint,
                   "targets_deg_motor": targets_deg_motor,
                   # 사후분석용 관측 원본 (0802 발진 사고 때 이 데이터가 없었음)
                   "qpos_rad": [round(v, 5) for v in qpos],
                   "qvel_rad_s": [round(v, 4) for v in qvel],
                   "contact": contact2,
                   "blend_alpha": round(alpha, 3),
                   "foot_n": [round(float(self.foot[0]), 2),
                              round(float(self.foot[1]), 2)],
                   # 자세 사후분석용 (0802 첫 march: 줄 걸림 vs 자체 기울어짐
                   # 구분에 IMU 부재가 병목이었음)
                   "quat_wxyz": [round(float(v), 5) for v in quat],
                   "gyro_rad_s": [round(float(v), 4) for v in gyro]}
            self.pub_dbg.publish(String(data=json.dumps(dbg)))
            self._bb_write(dbg)
            if not (self.live and self.armed):
                if self.live:
                    # 리뷰 F1: 실송출이 없는 동안 블렌드 클록 정지 — armed 공백
                    # 뒤 재개 첫 실명령이 무블렌드 풀스텝이 되는 경로 차단.
                    # 스트리밍 재개 첫 틱에 그 시점 명령/실측에서 재블렌드.
                    self.blend_t0 = None
                    self.blend_from = None
                return
            # 지연 수술 0803: 배치 명령 1건/틱 (종전 12건/틱 = 600msg/s의
            # DDS·콜백 부하가 젯슨 제어 지터에 기여). 검증은 stage8 쪽에서
            # 단건 경로와 동일하게 관절별 수행.
            targets = {}
            for j in JOINT_ORDER:
                base = self.baseline.get(j)
                if base is None:
                    continue   # baseline 미설정 관절은 노드가 어차피 거부
                # Stage8 절대각 = baseline + rel = motor_target 그대로 복원
                # (baseline이 어떤 자세로 캡처됐든 상쇄되어 무관)
                targets[j] = targets_deg_motor[j] - base
            if targets:
                self.pub_cmd.publish(String(data=json.dumps(
                    {"command": "SET_JOINT_TARGETS", "targets": targets})))

        def _bb_write(self, obj):
            """블랙박스 기록 — 실패(디스크 만충 등)해도 안전동작을 못 막게 격리.

            (적대리뷰 A3: 종전엔 write 예외가 STOP_ALL 발행 전에 콜백을 죽여
            안전동작 유실 + 노드 사망 경로였음)
            """
            if not self.bb:
                return
            try:
                self.bb.write(json.dumps(obj) + "\n")
            except OSError:
                try:
                    self.bb.close()
                except OSError:
                    pass
                self.bb = None
                self.get_logger().error("블랙박스 기록 실패 — 기록 중단 (제어는 계속)")

        def _stop_all(self, reason):
            self.get_logger().warning(f"STOP_ALL ({reason})")
            # 안전동작 먼저, 기록은 다음 (적대리뷰 A3: 순서 역전 금지)
            if self.live:
                self.pub_cmd.publish(String(
                    data=json.dumps({"command": "STOP_ALL"})))
            self._bb_write({"timestamp": time.time(), "event": "stop_all",
                            "reason": reason})

    rclpy.init()
    node = RlBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        if getattr(node, "bb", None):
            try:
                node.bb.close()
            except OSError:
                pass
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
