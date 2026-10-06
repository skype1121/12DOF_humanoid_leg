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
  - 느린 기울기 롤 보정기(맵 선택 키 roll_comp): 기본 OFF — 키가 없으면 종전과
    완전히 동일(계산·기록 없음). 켜면 타깃 발진 가드 **뒤**·프레임 변환 앞에서
    ±limit 이내 롤 오프셋을 최종 관절 목표에 더한다(시뮬 eval_hang2stand.py
    --roll_comp 와 같은 법칙). 가드·임계는 일절 불변 — 가드 입력은 교정된 정책
    체인 출력 그대로 두고, 보정기 출력은 구조적으로 느리게(틱당 ≤slew·|off|≤limit·
    τ≥0.5s·게이트 디바운스·해제 시 감쇠) 묶어 계단·발진을 만들 수 없게 한다.
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
    DEFAULT_POSE_RAD, JOINT_ORDER, SLEW_RAD)
from rl_walking.deploy.c5_switch import C5Runner, SwitchDetector  # noqa: E402 — 1004 C5(서기+회복 전환), 미배포
from rl_walking.deploy import deploy_chain as _DC  # noqa: E402 — 배포 경로 상수·함수의 단일 원본 (판정 스크립트와 공유)
from rl_walking.deploy import hturn2 as _H2  # noqa: E402 — 1006 HTURN2 실험 플래그(기본 꺼짐, 순수 numpy 상수·함수)
from rl_walking.deploy import w3_deploy as _W3  # noqa: E402 — 1006 W3 실험 플래그(기본 꺼짐, 순수 numpy 상수·함수)

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
#: 인계 워밍업(0911): stand 직후 stage8 status 스트림이 유휴 버스트/스톨로 아직 안 붙어
#: joint_ts 가 낡을 수 있다 — tick-1 에서 sensor_stale 로 오정지하던 것(실측: 3회 정상기동 vs
#: 금일 2회 tick-1 실패, 차이=타이밍)을 막는다. 이 창 동안엔 정책 구동 없이 홀드-프라임만 보내
#: stage8 status 에코를 깨우고 신선 피드백을 기다린다. 정상상태 게이트(SENSOR_STALE_SEC)는 불변.
STAND_WARMUP_MAX_S = 4.0   # 관측된 최대 status 갭(~2.9s)보다 넉넉. 초과 시 STOP_ALL.
COMMAND_TOPIC = "/humanoid/stage8_12axis_mit_command"
STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
JOINT_STATES_TOPIC = "/humanoid/joint_states"
IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"
BRIDGE_CMD_TOPIC = "/humanoid/rl_bridge_command"
BRIDGE_DEBUG_TOPIC = "/humanoid/rl_bridge_debug"
BRIDGE_ACK_TOPIC = "/humanoid/rl_bridge_ack"     # 1006 W3 세션 전용 명령 응답(수락/거부+이유) — 종전 세션은 발행하지 않음

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
    # stand_v14 (0904): 실질량 17.62kg(저울)·압력패드 8점 자산으로 학습한 첫 세대.
    # 고정환경(V6) 교차평가 v6b 대비 낙상 65.6→40.6%(com0)·85.9→29.7%(com+16), 전방
    # 밀기 소멸(추세 중앙 +0.1). md5 eed4c3fd… — 실물 데뷔 후보 1순위 (승윤님 결정 후 배포).
    "stand_v14": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-04_02-06-07/exported"),
    # stand_v15 (0904): v14 + 실물 오차 현실화(LPF 내장·착지 기울기·관절군별 영점·착지 비대칭) +
    # 토크 예산 벌점. 버그수정 후 재판정: V6 고정환경 낙상 18.8/17.2/10.9%(v14 40.6/29.7/21.9), 명목 0/64,
    # hip_r 토크포화 95%→0%, 에너지 1/6. **실물 데뷔 후보 1순위** (승윤님 결정 후 배포).
    "stand_v15": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-04_16-43-02/exported"),
    # stand_v16 (0904): v15 + 토크한계 0.6~1.0·관절마찰 0~0.3·발접촉 5틱 지연. V6 고정환경 26.6/26.6/15.6%
    # (v15 보다 열위이나 실물 효과 포함 학습), 플러시 자산 신래퍼 com+16 1.6%. 명목 0/64, 발 요차 7°.
    "stand_v16": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-04_20-05-14/exported"),
    # stand_v17 (0904): v16 + 백래시 ±0.3~1.5°·IMU 자세 지연 0~3틱. V6 고정환경 32.8/10.9/7.8%
    # (com+16 조건 최고 — 실물 CoM 재적합 +16mm 에 해당), 명목 0/64·추세 −0.04, 발 요차 2.7°, hip_r 포화 0%.
    "stand_v17": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-04_21-48-32/exported"),
    # stand_v18 (0905): 정정 env(실질량 17.62kg·명목 노이즈 정정, 새 축 없음) — 판정 reports/gen_*/README.md
    "stand_v18": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-05_17-57-21/exported"),
    # stand_v18b (0905): 정정 env(실질량 17.62kg·명목 노이즈 정정, 새 축 없음) — 판정 reports/gen_*/README.md
    "stand_v18b": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported"),
    # stand_hang_v1 (0908 밤): Hanging-to-Stand S4m(혼합 지지 스케줄+실측 대역 DR, v18b 계보 웜스타트) —
    # 관측 245 계약·클록 1.1Hz 동일. 판정 reports/hang2stand_0908/README.md. **젯슨 미배포(승윤님 결정 후).**
    "stand_hang_v1": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-08_19-00-25/exported"),
    # stand_hang_v2 (0909 00:00): S5v — 안정 자세(대칭·무릎 ∓6°·힙회전 0·몸통 −9°)+실물 현가 수평 모델. 실물유사 98.4/실제행잉 88.3 (256env).
    # 관측 245 계약·클록 1.1Hz·지연 3~8 동일. 판정 reports/hang2stand_0908/README.md. **젯슨 미배포(승윤님 결정 후).**
    "stand_hang_v2": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-08_23-33-51/exported"),
    # stand_hang_v3 (0909 12:00, **권장**): S5w — 몸통 수평 −2.3°·어깨너비 266mm·무릎 ∓5°. 실물유사 100/실제행잉 94.1 (256env).
    # 관측 245 계약·클록 1.1Hz·지연 3~8 동일. 판정 reports/hang2stand_0908/README.md. **젯슨 미배포(승윤님 결정 후).**
    "stand_hang_v3": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-09_11-34-22/exported"),
    # stand_hang_v4 (0909 13:00, **권장**): S6 = v3 + 발목롤 마찰·접촉 임계 DR. 실물유사 99.6/실제행잉 93.0/마찰·임계 조건 93.4(v3 89.8).
    # 관측 245 계약·클록 1.1Hz·지연 3~8 동일. 판정 reports/hang2stand_0908/README.md. **젯슨 미배포(승윤님 결정 후).**
    "stand_hang_v4": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-09_12-33-03/exported"),
    # stand_hang_v5 (0909 21:00, S8ac4): v4 + raw+EMA(α0.1) 자이로·관절속도 관측 **320차원** + 정착 후 정숙 보너스.
    # 1024env 실물유사 99.7/실제행잉 94.8/S6 98.5; v4 대비 몸 잔진동 ω −41%·v −80%·발들기 0; 정착 피치 −4.9°(후방 CoM 적응).
    # 관측 320 은 policy_runner_stage4 가 모델 입력 차원으로 자동 선택(EMA 는 러너 내부 계산) — 브리지 코드 변경 없음. **젯슨 미배포(승윤님 결정 후).**
    "stand_hang_v5": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/exported"),
    # stand_hang_v5p (0909 21:30, S8ac6): v5 + 몸통 밴드 −40 → v4 자세(몸통 −2.9°·무릎 ∓6°·발 간격 272mm) 그대로 + S8 정숙(ω −40%·v −65%).
    # 실제행잉 92.6(v4 동등, v5 94.1). 자세 우선 대안. (S8ac5 밴드 −60 은 전 지표 열세로 폐기.) 젯슨 미배포.
    "stand_hang_v5p": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-09_20-50-01/exported"),
    # stand_hang_v6 (0913, S8ac4r): v5 + 준정적 롤 램프 DR + 실측 잡음·IMU 편향·강성 대역. 관측 320 동일(브리지 변경 없음).
    # 사전등록 결과(reports/real_prep_0913/promotion_criteria_v6.md): 줄 없이 상수 6/9 Nm 100 %(v5 11/0), 연성 ×0.4 느린 방출 100 %(v5 0),
    # 램프 홀드 63/64; 대가 지연 6~9틱 73 %(v5r 84)·접촉 비트 상시 0 에서 46 %. **규칙상 미승격 — 대안 동봉(승윤님 선택). 젯슨 미배포.**
    "stand_hang_v6": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-13_00-40-49/exported"),
    # stand_crouch_v1 (0918 밤, S9c): **크라우치 기본자세**(손 튜닝 v3: 힙 ±15·무릎 ∓30·발목앞뒤 ±20.5) + env 별 물리 기본자세
    # 편향 DR(힙 ±12°·무릎 ±5°·발목 ±8°) + 스폰 피치 ±8° + 지면 연성 k2e3. v6 웜스타트 +3000it. 관측 320 동일.
    # **배포 계약 변경**: 이 정책의 DEFAULT_POSE 는 CKPT_DEFAULT_POSE["stand_crouch_v1"](관측 joint_pos 기준·액션 오프셋).
    # 인계 전 자세 = 콘솔 '수평 기준 자세'(v4) 홀드 → 정책은 자기 기본자세(v3) 근처에서 시작(편향 DR 안).
    "stand_crouch_v1": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-18_00-14-13_s9c/exported"),
    # stand_crouch_v1b (0918 새벽): v1 에서 +6000it 계속 학습(같은 S9c). 기본자세·계약 동일.
    "stand_crouch_v1b": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-18_01-08-37_s9c2/exported"),
    # stand_crouch_v1c (0918 새벽): v1b 에서 +12000it 계속 학습(같은 S9c).
    "stand_crouch_v1c": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-18_02-52-27_s9c3/exported"),
    # stand_crouch_v5 (0921 밤, G9 = G8 + knee_stance_flexion 회복게이트 단일 변경, models/stand_crouch_v5):
    #   **기본자세가 실물 v5 스탠바이(힙 ±30.26·무릎 ∓38.50·발목 ±28.00)와 일치하는 첫 등록 정책.**
    #   기존 stand_crouch_v1* 는 _CROUCH_V1_POSE(힙 15.00)라 인수 순간 '힙을 15° 펴라' 가 되어 0918 실물 stand 낙상을 만들었다.
    #   판정(reports/gen_g6/RECOVERY_EVAL_0920.md §12, 3 seed 384 시행 명목): 앞20N 22/384 · 느린앞 140/384 · 뒤20N 229/384 ·
    #   느린뒤 106/384 (G6-4500 대비 +5.7/+9.9/+17.4/+26.0 %p), 무외란 인계 뒤 생존 128/128, 목 루프(v5) PASS 램프 6틱·28.9°.
    #   약점: 무외란 정지 도달 0/128(평상시 잔움직임), 앞쪽 스텝 미생성(버티기 회복).
    "stand_crouch_v5": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-21_20-30-32_g9/exported"),
    # stand_crouch_v6 (0922 14:00, G10 = G9 + 명령 기준 관절 한계 벌점, models/stand_crouch_v6):
    #   0922 실물 2회 낙상 기전 대응 — 발목 **명령**이 소프트 한계 39° 를 넘어 56.7° 까지 감기며(windup) 접지 상실.
    #   기존 dof_pos_limits 는 실측 관절각 기준이라 이를 벌하지 못했다. 학습 중 cmd_limit_excess −0.125 → −0.072.
    #   기본자세는 v5 와 동일(실물 스탠바이 일치). 목 루프(v5) PASS 램프 10틱·44.5°.
    #   ⚠ **독립 시뮬 판정 없이 배포**(승윤님 지시, 0922 점검 일정) — 성능 근거는 학습 지표뿐이다.
    "stand_crouch_v6": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-22_13-32-53_g10/exported"),
    # stand_crouch_g11a (0929, G11a = G9 + 플랜트 정합: 발목 유격 안 정지 마찰 τs 3~6 Nm(가정값) + 기본자세 편향 폭 절반,
    #   models/stand_crouch_g11a) — **승격 아님, 실물 비교 시험 후보**(승윤님 결정 0929).
    #   사전 기준(reports/gen_g11/PROMOTION_CRITERIA.md) 탈락: 정지 도달 120/192(기준 135)·뒤 18N 회복 37 %(기준 39 %).
    #   v6 대비(같은 판정 환경): 정지 69→120 · 지연11틱 생존 118→152 · 뒤18N 26→34 % · 명령 부호반전 0.258→0.062.
    #   기본자세는 v5/v6 와 동일(CROUCH_V5). 관측·액션 계약은 G9 계열 그대로.
    "stand_crouch_g11a": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported"),
    # stand_crouch_g12 (0929, G12 = G11a + 학습 액션 지연 3~8 → 9~14틱, model_9500) — **승격 아님, 실물 비교 시험 후보**.
    #   근거: 실물 명령→관절 반응 16틱 ≈ 시뮬 지연 11틱, G11a 는 3~8틱 학습(실물 루프가 학습 범위 밖).
    #   사전 기준(reports/gen_g11/G12_CRITERIA.md) B 생존 150 < 홀드 151 (1/192) 로 탈락. 지연 11틱에서 G11a 대비:
    #   정지 116→141 · 몸통 |ω| 0.065→0.035 · 지연14 생존 131→147 · 지연6 정지 120→150. 기본자세 CROUCH_V5 동일.
    "stand_crouch_g12": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported"),
    # stand_crouch_g12_ema0 (1001) — **같은 정책 파일**, 관측 joint_vel_ema 만 0 (CKPT_JVEL_EMA_MODE). G4~G15 는 학습 중 이 채널이
    #   0+잡음이었다(mdp_hang 구버전 encoder_joint_vel_ema 결함, 커밋 9264b5a) → 학습 계약 복원판. 시뮬 단일 변수 대조(311쌍):
    #   무외란·8 N 동일, d3 뒤 18 N 낙상 v2 52.1 % vs 학습 계약 44.7 % (McNemar p=0.019). 이름 그대로 = 0929 과 같은 동작(v2).
    "stand_crouch_g12_ema0": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported"),
    # stand_crouch_g21a (1003, G21a-11300 = G19v 계열 + 보상 결함 교정 a1·a2·a3, 승윤님·외부 검토 설계) — **승격 아님, 진단 실물 시험 전용**.
    #   사전 기준(reports/gen_g18/G21_DESIGN_CRITERIA.md) 정식 C0 탈락(지연 3틱 흔들림 5.6 % > 2.9 %; 무외란 2.2 % 통과).
    #   목적: G12 와 교대로 같은 약한 앞 밀기 → ① 3 Hz 진폭·성장률 ② 이륙→착지→하중 수용 ③ 착지 후 정착 시간 비교(승격 판단 없음).
    #   관측 계약: actions = 원시 액션 EMA(β0.27) · joint_vel = 관절속도 EMA(β0.25) — env.yaml 로 자동 결정(ckpt_obs_filter), G12 는 종전 그대로.
    "stand_crouch_g21a": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported"),
    # stand_c5_g21a_g24ra (1004, 사용자 승인 C5 구조 진단 — **승격 아님**): 서기 G21a-11300 + 회복 G24R-a-21793, 상대 기울기 감지기 RV3.
    #   이 키의 경로 = 서기 정책(G21a) — 이름 기반 계약(기본자세·클록·관측 필터 env.yaml)이 전부 G21a 로 해석된다. 회복 정책은 C5_PAIRS 에서만 붙는다.
    #   근거: reports/gen_c5/REAL_GATE_C5.md (G1 RV3·G2 실물 로그 재생·G3/G4 21793·G5 힙 편향 곡선). 두 정책 관측·액션 계약 env.yaml 동일 확인.
    "stand_c5_g21a_g24ra": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported"),
    # stand_c5kr_g21a_g24ra (1005, **로컬 준비·실물 미검증·미배포**, 최종 krF): C5 + 감지 입력 EMA α0.33 + 회복 cue(켜짐 틱 필터 ω_y>0 → 1 s 회복 입력 vx 0.03)
    #   + 조기 복귀(0.6 s·6°·0.4 rad/s·양발 0.2 s) + 불응기(1.5 s, 6°/1.0). 근거: reports/gen_c5/imu/*, reports/gen_c5/kr/KD_PLAN.md.
    "stand_c5kr_g21a_g24ra": os.path.join(
        REPO, "logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported"),
    # stand_crouch_v2 (0918 저녁, S9d, models/stand_crouch_v2) — **등록 보류**: 기본자세 = 실물 v4 명령 + 무릎 직렬 처짐 편향 DR.
    # 시뮬 명목 94.5 % 이나 목 센서 500틱(IMU 무반응) 시험에서 기본자세 출발도 슬루 최대로 발산(틱 34 비유한) → 공중·줄 고정
    # 몸통에서 관절 폭주 위험. 정책 500틱 selftest 는 등록된 stand_* 전부를 순회하므로 등록 자체가 prep 게이트를 막는다.
    # 재학습(자세 편차 벌점·목 루프 안정성 평가 추가) 후 재등록.
}

#: 1004 C5 (사용자 승인 구조, **실물 미검증·미배포**): 이름 → (서기 정책 이름[CHECKPOINTS 키], 회복 정책 exported 경로).
#: 서기 정책의 관측 계약(기본자세·클록·관측 필터)을 두 정책이 공유한다(G24R 계열 = G21a 계보, 같은 계약).
#: 감지기 = 상대 기울기(기준 자세 편차, τ 10 s) — 절대 기울기는 실물 서기 3–4° 로 회복 모드 고착(1004 로그 재생 70–99 %).
#: 회복 체크포인트는 reports/gen_c5/REAL_GATE_C5.md G3 확정 뒤 등록(그 전엔 비움).
C5_PAIRS = {"stand_c5_g21a_g24ra": ("stand_crouch_g21a", os.path.join(
    REPO, "logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported")),
            "stand_c5kr_g21a_g24ra": ("stand_crouch_g21a", os.path.join(
    REPO, "logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported"))}
#: 1005 kr: C5 키별 추가 설정(감지기 조기 복귀·불응기, 러너 cue) — 시뮬 --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.0,6,1.0 --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu 와 같은 정의.
#: 1005 최종(krF): 감지 입력 EMA α0.33(관측 신호 잡음 대응) + 불응기 1.5 s — reports/gen_c5/kr/KD_PLAN.md (공정 비교·KDF3).
#: 1005 최종 KE: + 조기 복귀는 최대 기울기 편차 ≥ 5° 또는 최대 |ω| ≥ 1.0 이었던 활성화에만(무외란 오탐은 calm 경로) — reports/gen_c5/kr/KE_PLAN.md.
C5_EXTRA = {"stand_c5kr_g21a_g24ra": dict(det=dict(early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0)),
                                           runner=dict(cue_vx=0.03, cue_s=1.0, cue_dir_imu=True))}
C5_DETECTOR_KW = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any")   # RV4(1004 밤): RV3 + 끄기 접촉 한 발 이상 — 실물 시험 4 패드 0 고착 19 s 재생 해소·시뮬 RV3 와 동일(짝 불일치 0) — 젯슨 미반영(다음 배포 승인)


#: 1007 병목 2 (**기본 꺼짐 · 실물 미검증 · 젯슨 미배포**): C5 KE 조기 복귀 판정기. None = 현행 IMU 규칙(종전 경로 비트 동일 —
#: rl_walking/deploy/selftest_odic_equiv.py [A]). 켜는 길은 기동 인자 --c5-settle 하나뿐(set_c5_settle). dict = c5_switch.OdicSettle 설정
#: (ODIC_DEFAULT 위에 덮어씀: v_thr·ds_s·min_s·min_frac·window_s·k_hat·tau_ref). 조기 복귀(early)가 있는 C5 키(stand_c5kr_*)에만 적용 —
#: early 없는 키는 SwitchDetector 가 거부한다. 근거·한계: reports/gen_w3/STOPEST.md §9–§10 (k̂ 42 는 0919 역산 가정값, 실물 k 식별 전).
C5_SETTLE = None
#: odic 무릎 토크 입력 신선도 상한 [s] — analyze_fall 이 tau_est 동결값으로 보는 기준(0.1 s)과 같다. 넘으면 τ 없음(판정 불가 → 조기 복귀 막힘).
KNEE_TAU_MAX_AGE_S = 0.1
_ODIC_KEYS = ("v_thr", "ds_s", "min_s", "min_frac", "window_s", "k_hat", "tau_ref", "w_thr")


def parse_c5_settle(spec):
    """--c5-settle 값 → C5_SETTLE. None/'imu' = None(현행), 'odic' = {}(ODIC_DEFAULT), 'odicw' = {"w_thr": 0.4}(1007, odic + 감지 |ω_xy| < 0.4),
    JSON = 키 검증·범위 검증(OdicSettle 생성으로) 뒤 dict."""
    if spec is None or spec == "imu":
        return None
    if spec == "odic":
        cfg = {}
    elif spec == "odicw":
        cfg = {"w_thr": 0.4}
    else:
        cfg = json.loads(spec)
        if not isinstance(cfg, dict):
            raise ValueError("--c5-settle JSON 은 객체여야 함")
        bad = sorted(set(cfg) - set(_ODIC_KEYS))
        if bad:
            raise ValueError(f"--c5-settle 모르는 키 {bad} (허용 {list(_ODIC_KEYS)})")
    from rl_walking.deploy.c5_switch import ODIC_DEFAULT, OdicSettle
    OdicSettle(**dict(ODIC_DEFAULT, **cfg))      # 범위·tau_ref 검증(ValueError)
    return dict(cfg)


def set_c5_settle(cfg):
    global C5_SETTLE
    C5_SETTLE = None if cfg is None else dict(cfg)


def knee_tau_joint(tau_est_motor, status_age_s, dir_by_joint, max_age_s=KNEE_TAU_MAX_AGE_S):
    """odic 입력: stage8 feedback_torque_est(모터 프레임, 관절명 키) → (왼, 오른) 무릎 관절 프레임 토크 [N·m] = direction × τ_motor.
    미수신(None)·비유한·status 신선도 초과(또는 수신 이력 없음)는 None — 조용히 0 으로 대체하지 않는다(판정기가 '판정 불가' 처리)."""
    if status_age_s is None or not (0.0 <= float(status_age_s) <= float(max_age_s)):
        return None
    out = []
    for j in ("left_knee_joint", "right_knee_joint"):
        v = tau_est_motor.get(j)
        if v is None:
            return None
        v = float(v)
        if not math.isfinite(v):
            return None
        out.append(float(dir_by_joint[j]) * v)
    return tuple(out)


def _make_runner(checkpoint):
    """체크포인트 이름 → 러너. C5 이름이면 C5Runner(서기 정책 계약 공유), 아니면 PolicyRunnerStage4(종전)."""
    if checkpoint in C5_PAIRS:
        stand, rec_path = C5_PAIRS[checkpoint]
        _ex = C5_EXTRA.get(checkpoint, {})
        # 1007 odic(기본 꺼짐 → 인자 자체를 넘기지 않음). 켜져 있어도 조기 복귀(early)가 있는 C5 키에만 — 실행 중 다른 C5 키로 바꿔도 거부·중단 없이 종전 imu.
        _st = {"settle": dict(C5_SETTLE)} if (C5_SETTLE is not None and _ex.get("det", {}).get("early") is not None) else {}
        return C5Runner(CHECKPOINTS[stand], rec_path, detector=SwitchDetector(**C5_DETECTOR_KW, **_ex.get("det", {}), **_st), **_ex.get("runner", {}),
                        gait_freq=CKPT_GAIT_FREQ.get(stand, CKPT_GAIT_FREQ_DEFAULT),
                        default_pose_rad=CKPT_DEFAULT_POSE.get(stand),
                        jvel_ema_mode=ckpt_jvel_ema_mode(stand),
                        act_obs_ema_beta=ckpt_obs_filter(stand)[0], jvel_obs_ema_beta=ckpt_obs_filter(stand)[1])
    _h2 = {"hturn2": HTURN2_CKPTS[checkpoint]} if checkpoint in HTURN2_CKPTS else {}   # 1006 실험 플래그 켜짐 + HTURN2 키에만
    if checkpoint in W3_CKPTS:                                                          # 1006 W3 실험 플래그 켜짐 + W3 키에만
        _h2 = {"w3": W3_CKPTS[checkpoint]}
    return PolicyRunnerStage4(
        CHECKPOINTS[checkpoint],
        gait_freq=CKPT_GAIT_FREQ.get(checkpoint, CKPT_GAIT_FREQ_DEFAULT),
        default_pose_rad=CKPT_DEFAULT_POSE.get(checkpoint),
        jvel_ema_mode=ckpt_jvel_ema_mode(checkpoint),
        act_obs_ema_beta=ckpt_obs_filter(checkpoint)[0], jvel_obs_ema_beta=ckpt_obs_filter(checkpoint)[1], **_h2)


#: 0918 체크포인트별 기본자세 [rad, JOINT_ORDER 순]. 미등록 = 학습 공통 DEFAULT_POSE_RAD.
_CROUCH_V1_POSE = {
    "left_hip_f_joint": 0.2618, "right_hip_f_joint": -0.2618,
    "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0,
    "left_knee_joint": -0.5236, "right_knee_joint": 0.5236,
    "left_ankle_f_joint": 0.3578, "right_ankle_f_joint": -0.3578,
    "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0,
}
#: 0918 저녁 v4 (rl_walking.hang_stand_env_cfg.CROUCH_V4_DEFAULT_RAD 와 동일; stand_crouch_v2 재등록 시 CKPT_DEFAULT_POSE 에 연결)
#: 실물 v5 스탠바이 자세(jetson/calib_data/level_pose_0918.json 의 trims)와 동일 — G 계열(S9c 이후) 학습 기본자세.
#: 인수 블렌딩이 홀드 명령에서 정책 목표로 넘어가므로 이 값이 실물 자세와 다르면 인수 순간 자세 이동 명령이 된다(0918 낙상).
_CROUCH_V5_POSE = dict(_CROUCH_V1_POSE, left_hip_f_joint=0.5282, right_hip_f_joint=-0.5282,
                       left_knee_joint=-0.6720, right_knee_joint=0.6720,
                       left_ankle_f_joint=0.4887, right_ankle_f_joint=-0.4887)
_CROUCH_V4_POSE = dict(_CROUCH_V1_POSE, left_hip_f_joint=0.4538, right_hip_f_joint=-0.4538)   # 힙 ±26°
CKPT_DEFAULT_POSE = {
    "stand_crouch_v1": [float(_CROUCH_V1_POSE[j]) for j in JOINT_ORDER],   # 러너가 np.float32 로 변환
    "stand_crouch_v1b": [float(_CROUCH_V1_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_v1c": [float(_CROUCH_V1_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_v5": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_v6": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_g11a": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_g12": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_g12_ema0": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_crouch_g21a": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_c5_g21a_g24ra": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
    "stand_c5kr_g21a_g24ra": [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER],
}


def ckpt_default_pose(checkpoint):
    """체크포인트의 기본자세(관절 프레임 rad, JOINT_ORDER 순)."""
    return CKPT_DEFAULT_POSE.get(checkpoint, DEFAULT_POSE_RAD)

#: 체크포인트별 게이트 클록 주파수 [Hz] — 학습 클록과 반드시 일치 (불일치 =
#: 위상 어긋난 관측 → 보행 파괴). 미등록 = 1.4 (기존 가족 전체의 계약).
#: v5 가족(0803 측정 기반 재정합)부터 1.1Hz — 등록 시 평가 게이트+하네스
#: (--gait-freq 1.1) 통과 필수.
CKPT_GAIT_FREQ = {"stand_crouch_v1": 1.1, "stand_crouch_v1b": 1.1, "stand_crouch_v1c": 1.1, "stand_crouch_v5": 1.1, "stand_crouch_v6": 1.1, "stand_crouch_g11a": 1.1, "stand_crouch_g12": 1.1, "stand_crouch_g12_ema0": 1.1, "stand_crouch_g21a": 1.1, "stand_c5_g21a_g24ra": 1.1, "stand_c5kr_g21a_g24ra": 1.1, "march_v5b": 1.1, "stand_v3": 1.1, "stand_v5": 1.1, "stand_v6": 1.1, "stand_v7": 1.1,
                   "stand_v8": 1.1, "stand_v9": 1.1, "stand_v14": 1.1, "stand_v15": 1.1, "stand_v16": 1.1, "stand_v17": 1.1, "stand_v18": 1.1, "stand_v18b": 1.1, "stand_hang_v1": 1.1, "stand_hang_v2": 1.1, "stand_hang_v3": 1.1, "stand_hang_v4": 1.1, "stand_hang_v5": 1.1, "stand_hang_v5p": 1.1, "stand_hang_v6": 1.1}
CKPT_GAIT_FREQ_DEFAULT = 1.4

#: 1001 관측 joint_vel_ema 모드 (러너 jvel_ema_mode). 미등록 = 'v2'(실제 속도 EMA, 0929 까지의 동작).
#: 'ema0' 은 학습 env.yaml 의 joint_vel_ema 가 구버전(encoder_joint_vel_ema = 학습 중 0+잡음)인 정책에만 허용 —
#: v2 로 학습한 정책(G16~)에 0 을 넣는 반대 방향 결함은 _jvel_ema_contract 검사가 기동을 거부한다.
CKPT_JVEL_EMA_MODE = {"stand_crouch_g12_ema0": "ema0"}
#: 1003: 관측 필터 계약(학습 env.yaml 로 결정)이 필요한 정책 — env.yaml 이 없으면 원시 관측으로 조용히 도는 대신 기동 거부.
CKPT_REQUIRE_ENV_YAML = {"stand_crouch_g21a", "stand_c5_g21a_g24ra", "stand_c5kr_g21a_g24ra"}
#: stand 명령에 --checkpoint 가 stand 가족이 아닐 때의 대체 정책 = 7차 데뷔 후보(0906).
#: 대체는 경고 로그로 드러낸다 (0905 감사 A1: 무언 대체 'stand_v7' = 10.49kg 자산 정책 탑승 경로).
STAND_DEFAULT = "stand_v18b"

# ---------------------------------------------------------------------------
# 1006 HTURN2 실험 플래그 (**기본 꺼짐 · 실물 미검증 · 젯슨 미배포**) — reports/HTURN2_BRIDGE_FLAG_1006.md
# 앞 보행 W2q 25590 + 명령 연동 cadence 클록 + 회전 피드포워드(hip_r) + HeadingHold 목표 각(|wz| ≤ 0.3).
# 켜는 길은 기동 인자 --experimental-hturn2 하나뿐(enable_experimental_hturn2). 꺼져 있으면 아래 HTURN2_CKPTS 가 비어
# CHECKPOINTS·CKPT_*·LPF 분기·HeadingHold 상수·명령 처리 전부가 종전과 비트 동일하다(selftest 'HTURN2 실험 플래그' 가 고정).
# 플래그를 켜도 HTURN2 키(_H2.HTURN2_CKPT_KEY)를 --checkpoint 로 고른 경우에만 러너 계약이 바뀐다(체크포인트별 설정 + 명시 플래그).
# ---------------------------------------------------------------------------
EXPERIMENTAL_HTURN2 = False
HTURN2_CKPTS = {}          # 켜졌을 때만 {키: Hturn2Config}
HH_DEFAULT_KP, HH_DEFAULT_WZ_MAX = 1.0, 0.6    # 종전 BridgeCore HeadingHold 상수(HTURN2 키가 아니면 항상 이것)


def enable_experimental_hturn2():
    """HTURN2 키 등록 — 경로 = W2q run 의 exported/(없으면 selftest 가 SKIP, 기동 --checkpoint 로 고르면 필수 파일 FAIL).
    계약: 기본자세 = CROUCH_V5(학습 env.yaml init_state), 클록 f0 1.1, 관측 필터 env.yaml 필수(act EMA·jvel LPF), LPF β0.27 적용."""
    global EXPERIMENTAL_HTURN2
    k = _H2.HTURN2_CKPT_KEY
    CHECKPOINTS[k] = os.path.join(REPO, _H2.HTURN2_RUN_REL, "exported")
    CKPT_GAIT_FREQ[k] = _H2.HTURN2_DEFAULT.f0
    CKPT_DEFAULT_POSE[k] = [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER]
    CKPT_REQUIRE_ENV_YAML.add(k)
    HTURN2_CKPTS[k] = _H2.HTURN2_DEFAULT
    EXPERIMENTAL_HTURN2 = True


def disable_experimental_hturn2():
    """등록 되돌림(selftest 격리용) — 꺼진 상태 = 위 레지스트리에 키가 하나도 없다."""
    global EXPERIMENTAL_HTURN2
    k = _H2.HTURN2_CKPT_KEY
    for reg in (CHECKPOINTS, CKPT_GAIT_FREQ, CKPT_DEFAULT_POSE, HTURN2_CKPTS):
        reg.pop(k, None)
    CKPT_REQUIRE_ENV_YAML.discard(k)
    EXPERIMENTAL_HTURN2 = False


def hturn2_target_rad(x):
    """목표 요 변화[deg] → rad. |x| ≤ 180, 유한값만."""
    x = float(x)
    if not (math.isfinite(x) and abs(x) <= 180.0):
        raise ValueError(f"HTURN2: heading_target_deg {x} 는 ±180 밖")
    return math.radians(x)


def hturn2_validate_walk(d):
    """HTURN2 키의 walk 명령 검증 → (cmd, heading_target_rad) 또는 ValueError(거부 사유).
    앞 보행 한정: 0 ≤ vx ≤ vx_max(학습 범위 상한 0.15), vy = 0, wz = 0(회전은 heading_target_deg 로만), heading_hold 필수, vx 명시 필수."""
    c = _H2.HTURN2_DEFAULT
    if "vx" not in d:
        raise ValueError("HTURN2: vx 명시 필수(종전 기본 0.3 은 학습 범위 밖)")
    vx, vy, wz = float(d.get("vx")), float(d.get("vy", 0.0)), float(d.get("wz", 0.0))
    if not (math.isfinite(vx) and 0.0 <= vx <= c.vx_max):
        raise ValueError(f"HTURN2: vx {vx} 는 [0, {c.vx_max}] 밖(앞 보행 한정)")
    if vy != 0.0 or wz != 0.0:
        raise ValueError("HTURN2: vy·wz 는 0 이어야 함 — 회전은 heading_target_deg(HeadingHold) 로만")
    if not bool(d.get("heading_hold", True)):
        raise ValueError("HTURN2: heading_hold 끔 불가(회전 명령 경로가 HeadingHold 뿐)")
    return (vx, 0.0, 0.0), hturn2_target_rad(d.get("heading_target_deg", 0.0))


# ---------------------------------------------------------------------------
# 1006 W3 실험 플래그 (**기본 꺼짐 · 실물 미검증 · 젯슨 미배포**) — reports/W3_BRIDGE_FLAG_1006.md
# W3-A 400 제자리 교대 스텝 엔진: 관측 클록 = [sin 2πu, cos 2πu] × ge_obs(gait_enable 래치·0.2 s 램프), 명령 (0,0,0) + gait_enable.
# 켜는 길은 기동 인자 --experimental-w3 하나뿐(enable_experimental_w3). 꺼져 있으면 W3_CKPTS 가 비어 레지스트리·명령 처리·
# 블랙박스 스키마 전부 종전과 같다(selftest 'W3 실험 플래그' 가 고정). 켜도 --checkpoint 로 W3 키를 고른 경우에만 계약이 바뀐다.
# 키는 stand 가족(stand_ 접두사): 시작 = stand 명령(ge 0 서기, heading_hold 강제 OFF, LPF β0.27), 스텝 = {"cmd":"step","enable":true}.
# ---------------------------------------------------------------------------
EXPERIMENTAL_W3 = False
W3_CKPTS = {}              # 켜졌을 때만 {키: W3Config}
W3_CMD_LIMITS = {}         # 1007: 켜졌을 때만 {명령 채널 키: W3CmdLimits} — 없는 키(W3-A)는 명령 (0,0,0) 고정
W3_SAFETY = _W3.W3_SAFETY_DEFAULT   # W3 세션 감독기 문턱(기동 --w3-safety JSON 으로만 바꿈, 범위 검증) — 감사 H3–H6·L1
W3_HL_CFG = _W3.W3_HL_DEFAULT       # 1007 헤딩 고리(기동 --w3-heading-loop 'off'|JSON). 명령 채널 키에서만 쓰인다
W3_CMD_LIMITS_OVERRIDE = None       # 1007 기동 --w3-cmd-limits(상한·체류만, 범위 검증) — None = 학습 범위
#: 1007 강화 연구 2 차 대비책(기본 꺼짐): --w3-monitor 'on'|'note'|JSON → rl_walking/deploy/w3_contingency.W3MonitorConfig.
#: None 이면 BridgeCore.w3mon 이 None 이라 tick·감독기·블랙박스 스키마가 종전과 같다(selftest 'W3 감시' 가 비트 동일 고정).
#: 모듈은 이 값이 있을 때만 import 한다(젯슨에 파일이 없어도 기본 기동 불변). 문서 reports/gen_w3/CONTINGENCY.md
W3_MON_CFG = None
C5_OVERTIME = None                  # 1007 --c5-overtime 'note:6'|'freeze:6' → RecoveryOvertime(회복 세션 C5 키에서만)


def _w3c():
    from rl_walking.deploy import w3_contingency as _m   # 지연 import — 플래그를 켤 때만
    return _m


def enable_experimental_w3():
    """W3 키 등록(W3-A 400 + 1007 W3S3d-15800) — 경로 = 각 run 의 exported/(없으면 selftest '정책 500틱' SKIP, 기동 --checkpoint 로
    고르면 필수 파일 FAIL). 계약: 기본자세 = CROUCH_V5(학습 env.yaml init_state, G21a 와 같음), env.yaml 필수(관측 필터 act EMA·jvel LPF),
    LPF β0.27(stand 경로). CKPT_GAIT_FREQ 는 stand 가족 레지스트리 요건으로만 둔다(W3 클록은 gait_freq 를 쓰지 않는다)."""
    global EXPERIMENTAL_W3
    for k, (run_rel, _m, _t, cfg, lim) in _W3.W3_SPECS.items():
        CHECKPOINTS[k] = os.path.join(REPO, run_rel, "exported")
        CKPT_GAIT_FREQ[k] = cfg.f_step
        CKPT_DEFAULT_POSE[k] = [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER]
        CKPT_REQUIRE_ENV_YAML.add(k)
        W3_CKPTS[k] = cfg
        if lim is not None:
            W3_CMD_LIMITS[k] = W3_CMD_LIMITS_OVERRIDE if W3_CMD_LIMITS_OVERRIDE is not None else lim
    EXPERIMENTAL_W3 = True


def disable_experimental_w3():
    """등록 되돌림(selftest 격리용) — 꺼진 상태 = 레지스트리에 W3 키가 하나도 없다."""
    global EXPERIMENTAL_W3
    for k in _W3.W3_KEYS:
        for reg in (CHECKPOINTS, CKPT_GAIT_FREQ, CKPT_DEFAULT_POSE, W3_CKPTS, W3_CMD_LIMITS):
            reg.pop(k, None)
        CKPT_REQUIRE_ENV_YAML.discard(k)
    EXPERIMENTAL_W3 = False


def _w3_env_contract(path):
    """1007: 학습 env.yaml 에서 W3 명령·위상 계약만 줄 단위로 뽑는다(PyYAML 불필요) — events.w3_gait.params(f_step·lead_*)·
    events.w3_cmd.params(stage·범위 튜플)·commands.base_velocity.heading_command·actions.joint_pos 지연 대역. 없는 키는 결과에 없음."""
    out = {"w3_gait": {}, "w3_cmd": {}}
    top = ev = None
    in_params = False
    tup_key, tup = None, []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for raw in lines + ["__end__:"]:
        if not raw.strip():
            continue
        ind = len(raw) - len(raw.lstrip())
        st = raw.strip()
        if tup_key is not None:
            if st.startswith("- ") and ind == 6:
                tup.append(float(st[2:]))
                continue
            out[ev][tup_key] = tuple(tup)
            tup_key, tup = None, []
        if ind == 0:
            top, ev, in_params = st.split(":")[0], None, False
            continue
        key, _, val = st.partition(":")
        val = val.strip()
        if top == "events":
            if ind == 2:
                ev, in_params = (key if key in ("w3_gait", "w3_cmd") else None), False
            elif ind == 4 and ev is not None:
                in_params = key == "params"
            elif ind == 6 and ev is not None and in_params:
                if val.startswith("!!python/tuple"):
                    tup_key, tup = key, []
                elif val not in ("", "null"):
                    out[ev][key] = float(val) if key != "stage" else int(val)
        elif top == "commands" and ind == 4 and key == "heading_command":
            out["heading_command"] = (val == "true")
        elif top == "actions" and ind == 4 and key in ("min_delay_steps", "max_delay_steps"):
            out[key] = int(val)
    return out


def w3_monitor_gate(dec, mon):
    """1007 2 차 대비책: 감독기가 수락한 '켜기'(set_want True)를 감시기 판정(접촉 채터·센서 나이·실효 지연)으로 거부할 수 있다.
    mon None(기본)이면 dec 를 그대로 돌려준다. 끄기·정지·명령 변경은 건드리지 않는다(안전 방향만 더함)."""
    if mon is None or dec.get("action") != "set_want" or not dec.get("want"):
        return dec
    ok, why = mon.enable_gate()
    if ok:
        return dec
    return {"accepted": False, "why": why, "action": "none", "want": None, "cmd": None, "hl": None}


def w3_monitor_latch(sup, why):
    """1007 감시 걷기 정지 → 감독기 정지 래치(rearm 전 켜기 거부). 이미 래치면 첫 사유 유지(W3Supervisor.on_frozen 과 같은 규칙)."""
    if not sup.stop_latched:
        sup.latch_reason = str(why)
    sup.stop_latched = True


def w3_parse_step(d):
    """{"cmd":"step","enable":bool} → bool. enable 은 반드시 명시(bool 또는 0/1)."""
    if "enable" not in d:
        raise ValueError("W3: step 명령은 enable 명시 필수")
    v = d["enable"]
    if isinstance(v, bool):
        return v
    if v in (0, 1):
        return bool(v)
    raise ValueError(f"W3: enable 은 true/false 또는 0/1 (받음 {v!r})")

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
                 gains_by_joint, roll_comp=None, pitch_comp=None, guard_mode="safe"):
        self.path = path
        self.dir_by_joint = dir_by_joint            # {관절: ±1}
        self.sz_by_joint = sz_by_joint              # {관절: 모터 프레임 deg}
        self.motor_id_by_joint = motor_id_by_joint  # {관절: 1..12}
        self.gains_by_joint = gains_by_joint        # {관절: (kp, kd)}
        self.roll_comp = roll_comp                  # 느린 기울기 롤 보정기 설정 dict 또는 None(=OFF)
        self.pitch_comp = pitch_comp                # 앞뒤 균형 복원항 설정 dict 또는 None(=OFF)
        self.guard_mode = guard_mode                # "safe"(기본) | "data"(발진·리프트 가드 기록만)

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
                full = json.load(f)
            js = full.get("joints") or {}
            sig = {n: (c.get("direction"), round(float(c.get("stand_zero_deg", 1e9)), 2))
                   for n, c in js.items() if isinstance(c, dict) and "motor_id" in c}
            # 0913 F6: 정책 거동을 바꾸는 보정 키도 사본 일치 검사에 포함 — 한쪽만 켜진(또는 값이 다른)
            # 맵으로 브리지가 뜨는 경로 차단(드라이런은 A사본 PASS·실물은 B사본, 구맵 사건과 같은 구조).
            return (sig,
                    json.dumps(full.get("roll_comp"), sort_keys=True, ensure_ascii=False),
                    json.dumps(full.get("pitch_comp"), sort_keys=True, ensure_ascii=False),
                    json.dumps(full.get("guard_mode"), sort_keys=True, ensure_ascii=False),
                    json.dumps(full.get("imu_mount_bias_deg"), sort_keys=True, ensure_ascii=False))
        sigs = {p: _frame_sig(p) for p in existing}
        first = sigs[existing[0]]
        for p in existing[1:]:
            if sigs[p] != first:
                raise ValueError(
                    "하드웨어맵 사본 불일치 — direction/stand_zero_deg/roll_comp/pitch_comp/"
                    "guard_mode/imu_mount_bias_deg "
                    "중 하나가 서로 다름:\n  "
                    + "\n  ".join(existing)
                    + "\n  구맵 사본을 동기화한 뒤 재기동하세요 (기동 차단)")
    path = existing[0]
    print(f"[FrameMap] 하드웨어맵: {path}")
    with open(path, encoding="utf-8") as f:
        _full = json.load(f)
    joints = _full.get("joints") or {}
    _mb = _full.get("imu_mount_bias_deg")
    if _mb is not None and (not isinstance(_mb, dict) or not {"lat", "fwd"} <= set(_mb.keys())):
        raise ValueError(f"하드웨어맵 imu_mount_bias_deg 형식 오류: {_mb!r} (dict, 키 lat·fwd 필수; 조용히 무시하지 않음 — 0913 감사)")
    if isinstance(_mb, dict):
        _lat, _fwd = _validate_imu_mount_bias(_mb)
        IMU_MOUNT_BIAS_DEG[0], IMU_MOUNT_BIAS_DEG[1] = float(_lat), float(_fwd)
        print(f"[FrameMap] IMU 장착 편향 보정 lat {float(_lat):+.2f}° fwd {float(_fwd):+.2f}° (정책 중력벡터에만 적용)")
    else:
        IMU_MOUNT_BIAS_DEG[0], IMU_MOUNT_BIAS_DEG[1] = 0.0, 0.0
    # 느린 기울기 롤 보정기 (선택 키, 기본 OFF). 키 없음/enabled=false → None = 완전 OFF.
    # 범위·오타는 여기서 ValueError (imu_mount_bias_deg 와 같은 규약 — 조용히 무시하지 않는다).
    _rc_cfg = _validate_roll_comp(_full.get("roll_comp"))
    if _rc_cfg:
        print(f"[FrameMap] roll_comp 활성 — K={_rc_cfg['K']:g} tau={_rc_cfg['tau_s']:g}s "
              f"limit=±{_rc_cfg['limit_deg']:g}° joints={_rc_cfg['joints']} sign={_rc_cfg['sign']:+.0f} "
              f"lr={_rc_cfg['lr']} gate={_rc_cfg['gate']} debounce={_rc_cfg['debounce_s']:g}s "
              f"slew={_rc_cfg['slew_deg_per_tick']:g}°/틱 roll_clamp=±{_rc_cfg['roll_clamp_deg']:g}° "
              f"(타깃 가드 뒤·블렌딩 앞에서 가산; 시뮬 --roll_comp 와 같은 법칙)")
    # 앞뒤 균형 복원항 (선택 키 pitch_comp, 기본 OFF). roll_comp 와 같은 법칙·같은 자리, 축만 pitch.
    _pc_cfg = _validate_roll_comp(_full.get("pitch_comp"), name="pitch_comp")
    if _pc_cfg:
        if _pc_cfg["axis"] != "pitch":
            raise ValueError(f"하드웨어맵 pitch_comp axis={_pc_cfg['axis']!r} 이상 (pitch 이어야 함)")
        print(f"[FrameMap] pitch_comp 활성 — K={_pc_cfg['K']:g} tau={_pc_cfg['tau_s']:g}s "
              f"limit=±{_pc_cfg['limit_deg']:g}° joints={_pc_cfg['joints']} sign={_pc_cfg['sign']:+.0f} "
              f"lr={_pc_cfg['lr']} gate={_pc_cfg['gate']} debounce={_pc_cfg['debounce_s']:g}s "
              f"slew={_pc_cfg['slew_deg_per_tick']:g}°/틱 clamp=±{_pc_cfg['roll_clamp_deg']:g}° "
              f"(앞기울임 + 일 때 왼발목 +, 오른발목 − 로 기하선 위를 더 쓴다)")
    # 가드 모드 — "safe"(기본, 종전) | "data"(발진·리프트 가드는 기록만, 과속 폭주는 그대로 정지)
    _gm = _full.get("guard_mode", "safe")
    if _gm not in ("safe", "data"):
        raise ValueError(f"하드웨어맵 guard_mode={_gm!r} 이상 (safe | data)")
    if _gm == "data":
        print("[FrameMap] ** guard_mode=data ** — 발진 가드·타깃 발진 가드·리프트 가드는 기록만 하고 "
              "정책을 멈추지 않는다. 과속(12 rad/s)·센서 스테일·중력 게이트는 그대로 정지한다. "
              "운영자가 줄과 손으로 회수를 책임지는 데이터 수집 모드다.")
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
    return FrameMap(path, dir_by, sz_by, mid_by, gains_by, roll_comp=_rc_cfg,
                    pitch_comp=_pc_cfg, guard_mode=_gm)


def projected_gravity(quat_wxyz):
    """world −z 를 body 프레임으로 (학습 obs 와 동일식). 직립·정합 IMU 면 (0,0,−1)."""
    w, x, y, z = quat_wxyz
    return (-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y)))


#: IMU 장착 편향 [deg] = [lat(롤, + = 왼쪽), fwd(피치, + = 앞)] — rp_deg/콘솔 규약. 하드웨어맵 선택 키
#: "imu_mount_bias_deg": {"lat": 1.3, "fwd": -0.6} 로 지정(없으면 0 = 종전과 동일). 0912 지그 수평 접지 실측: lat +1.2~1.4, fwd −0.2~−1.1.
#: 정책 관측(중력벡터)에만 적용한다 — 게이트·rp_deg 기록·콘솔 표시는 원시값 유지(대조 가능하도록).
IMU_MOUNT_BIAS_DEG = [0.0, 0.0]
IMU_MOUNT_BIAS_MAX_DEG = 5.0
IMU_MOUNT_CORR_MIN_GZ = 0.3   # −g_z 가 이보다 작으면(기울기 > ~73°) 보정 생략


def _validate_imu_mount_bias(mb):
    """하드웨어맵 imu_mount_bias_deg dict → (lat, fwd) [deg]. 유한 float·|v| ≤ IMU_MOUNT_BIAS_MAX_DEG 아니면 ValueError."""
    lat, fwd = mb.get("lat", 0.0), mb.get("fwd", 0.0)
    for v in (lat, fwd):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or abs(float(v)) > IMU_MOUNT_BIAS_MAX_DEG:
            raise ValueError(f"하드웨어맵 imu_mount_bias_deg 이상: {mb!r} (유한 float, |v| ≤ {IMU_MOUNT_BIAS_MAX_DEG}° 필요)")
    return float(lat), float(fwd)


def correct_gravity(g, bias_deg=None):
    """투영중력 g 에서 장착 편향(롤·피치)을 뺀다. rp 정의(roll=atan2(g_y,−g_z), pitch=atan2(g_x,−g_z))에서 각각 편향을 빼고
    같은 정의로 단위벡터를 복원한다(두 atan2 값이 g_z<0 인 방향을 유일하게 정하므로 정확). 편향 0 이면 g 그대로."""
    b = IMU_MOUNT_BIAS_DEG if bias_deg is None else bias_deg
    if not b or (abs(float(b[0])) < 1e-12 and abs(float(b[1])) < 1e-12):
        return g
    gx, gy, gz = (float(v) for v in g)
    if -gz < IMU_MOUNT_CORR_MIN_GZ:      # 0913 감사: 기울기 ≥ ~73°(넘어짐)에선 (roll,pitch) 재구성이 반구를 접어 뒤집힌 자세가 직립처럼 보임 → 보정 생략(원시값)
        return g
    roll = math.atan2(gy, -gz) - math.radians(float(b[0]))
    pitch = math.atan2(gx, -gz) - math.radians(float(b[1]))
    tx, ty = math.tan(pitch), math.tan(roll)
    c = 1.0 / math.sqrt(1.0 + tx * tx + ty * ty)
    return (c * tx, c * ty, -c)


# ── 느린 기울기 롤 보정기(roll_comp) — 하드웨어맵 선택 키, 기본 OFF (0913) ──────────
#: 시뮬 대조 원본: rl_walking/scripts/eval_hang2stand.py --roll_comp (같은 법칙·같은 자리).
#: 맵 키 예 (없으면 완전 OFF = 종전과 비트 동일):
#:   "roll_comp": {"enabled": false, "K": 1.0, "tau_s": 1.0, "limit_deg": 5.0,
#:                 "joints": "ankle_r", "sign": 1, "lr": [1, 1], "gate": "both",
#:                 "debounce_s": 0.2, "slew_deg_per_tick": 0.25, "roll_clamp_deg": 15.0}
#: 값은 시뮬 검증(--roll_comp 스윕) 으로 확정한 뒤에만 켠다 — 코드에 운용값을 박지 않는다.
#: 범위는 0913 적대검수(F5) 기준 = 시뮬로 실제 훑은 봉투 안쪽으로만 허용한다.
ROLL_COMP_JOINT_CHOICES = ("ankle_r", "hip_a", "ankle_f")
#: 0923 추가 — 보정기 축. roll = 느린 좌우 드리프트 보정(종전). pitch = 앞뒤 균형 복원항.
#: 근거: 0922 3회차 실물 무보조 구간에서 발목 명령이 평발 기하선(−1.00 °/°)을 그대로 따라가
#: 복원 작용이 0(잔차 −0.021 °/°)이었다. 같은 정책이 시뮬에서는 +0.35~0.64, 실물도 손 보조
#: 구간에서는 +0.39 를 냈다. 모자란 그 항을 명령 위에 직접 더한다.
ROLL_COMP_AXIS_CHOICES = ("roll", "pitch")
ROLL_COMP_GATE_CHOICES = ("both", "any")        # "none"(상시 작동)은 실물에서 거부 — 공중·낙상 중 구동 금지
ROLL_COMP_K_MAX = 1.5
ROLL_COMP_TAU_MIN_S, ROLL_COMP_TAU_MAX_S = 0.5, 5.0        # roll 축 봉투(종전, 불변)
#: 축별 봉투. pitch 축은 느린 드리프트 보정이 아니라 빠른 되먹임이라 τ·슬루 범위가 다르다.
#: τ 를 0.5s 로 두면 이미 있는 320ms 루프 지연 위에 또 지연을 얹어 복원항이 늦는다.
ROLL_COMP_TAU_RANGE = {"roll": (0.5, 5.0), "pitch": (0.02, 5.0)}
ROLL_COMP_SLEW_RANGE = {"roll": (0.05, 0.5), "pitch": (0.05, 2.0)}
ROLL_COMP_LIMIT_MAX_DEG = {"ankle_r": 5.0, "hip_a": 3.0, "ankle_f": 15.0}
ROLL_COMP_DEBOUNCE_MIN_S, ROLL_COMP_DEBOUNCE_MAX_S = 0.1, 1.0
ROLL_COMP_SLEW_MIN_DEG, ROLL_COMP_SLEW_MAX_DEG = 0.05, 0.5
ROLL_COMP_ROLLCLAMP_MIN_DEG, ROLL_COMP_ROLLCLAMP_MAX_DEG = 5.0, 30.0
ROLL_COMP_DEFAULTS = {"K": 1.0, "tau_s": 1.0, "debounce_s": 0.2,
                      "slew_deg_per_tick": 0.25, "roll_clamp_deg": 15.0}
ROLL_COMP_KEYS = frozenset(
    ("enabled", "axis", "K", "tau_s", "limit_deg", "joints", "sign", "lr", "gate",
     "debounce_s", "slew_deg_per_tick", "roll_clamp_deg", "note"))


def _rc_num(rc, key, val, lo, hi, name="roll_comp"):
    """보정기 수치 필드 → float. 유한 float·lo~hi 아니면 ValueError (bool 거부)."""
    if (isinstance(val, bool) or not isinstance(val, (int, float))
            or not math.isfinite(float(val)) or not (lo <= float(val) <= hi)):
        raise ValueError(
            f"하드웨어맵 {name} {key}={val!r} 이상 (유한 float, {lo}~{hi} 필요): {rc!r}")
    return float(val)


def _validate_roll_comp(rc, name="roll_comp"):
    """하드웨어맵 roll_comp dict → 정규화 설정 dict. 키 없음 또는 enabled=false → None(완전 OFF).

    범위 밖·오타 키는 조용히 무시하지 않고 기동 시점에 ValueError
    (imu_mount_bias_deg 와 같은 규약 — 잘못된 보정 상수가 말없이 실물에 실리는 경로 차단).
    한계는 시뮬로 훑은 봉투: K ≤ 1.5, τ 0.5~5 s, |offset| ≤ 5°(ankle_r)/3°(hip_a),
    lr 는 ±1 만, gate 는 both/any(상시 none 금지).
    """
    if rc is None:
        return None
    if not isinstance(rc, dict):
        raise ValueError(f"하드웨어맵 {name} 형식 오류: {rc!r} (dict 필요)")
    unknown = set(rc) - ROLL_COMP_KEYS
    if unknown:
        raise ValueError(
            f"하드웨어맵 {name} 미지의 키 {sorted(unknown)} — 오타 의심 "
            f"(허용 {sorted(ROLL_COMP_KEYS)})")
    axis = rc.get("axis", "roll")
    if axis not in ROLL_COMP_AXIS_CHOICES:
        raise ValueError(f"하드웨어맵 {name} axis={axis!r} 이상 ({list(ROLL_COMP_AXIS_CHOICES)} 중 하나)")
    joints = rc.get("joints", "ankle_r" if axis == "roll" else "ankle_f")
    if joints not in ROLL_COMP_JOINT_CHOICES:
        raise ValueError(
            f"하드웨어맵 {name} joints={joints!r} 이상 ({list(ROLL_COMP_JOINT_CHOICES)} 중 하나)")
    lim_max = ROLL_COMP_LIMIT_MAX_DEG[joints]
    _tlo, _thi = ROLL_COMP_TAU_RANGE[axis]
    _slo, _shi = ROLL_COMP_SLEW_RANGE[axis]
    k = _rc_num(rc, "K", rc.get("K", ROLL_COMP_DEFAULTS["K"]), 0.0, ROLL_COMP_K_MAX, name)
    tau = _rc_num(rc, "tau_s", rc.get("tau_s", ROLL_COMP_DEFAULTS["tau_s"]), _tlo, _thi, name)
    lim = _rc_num(rc, "limit_deg", rc.get("limit_deg", lim_max), 0.0, lim_max, name)
    sign = rc.get("sign", 1)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError(f"하드웨어맵 {name} sign={sign!r} 이상 (±1 만 허용)")
    lr = rc.get("lr", [1, 1])
    if not isinstance(lr, (list, tuple)) or len(lr) != 2:
        raise ValueError(f"하드웨어맵 {name} lr={lr!r} 이상 (길이 2 리스트 [좌, 우])")
    for v in lr:
        if isinstance(v, bool) or v not in (-1, 1):
            raise ValueError(f"하드웨어맵 {name} lr={lr!r} 이상 (각 항목 +1 또는 −1 만)")
    lr = [float(v) for v in lr]
    gate = rc.get("gate", "both")
    if gate not in ROLL_COMP_GATE_CHOICES:
        raise ValueError(
            f"하드웨어맵 {name} gate={gate!r} 이상 ({list(ROLL_COMP_GATE_CHOICES)} 중 하나; "
            "'none'(상시 작동)은 실물에서 금지)")
    deb = _rc_num(rc, "debounce_s", rc.get("debounce_s", ROLL_COMP_DEFAULTS["debounce_s"]),
                  ROLL_COMP_DEBOUNCE_MIN_S, ROLL_COMP_DEBOUNCE_MAX_S, name)
    slew = _rc_num(rc, "slew_deg_per_tick",
                   rc.get("slew_deg_per_tick", ROLL_COMP_DEFAULTS["slew_deg_per_tick"]),
                   _slo, _shi, name)
    rclamp = _rc_num(rc, "roll_clamp_deg",
                     rc.get("roll_clamp_deg", ROLL_COMP_DEFAULTS["roll_clamp_deg"]),
                     ROLL_COMP_ROLLCLAMP_MIN_DEG, ROLL_COMP_ROLLCLAMP_MAX_DEG, name)
    enabled = rc.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"하드웨어맵 {name} enabled={enabled!r} 이상 (true/false)")
    if not enabled:
        return None
    return {"axis": axis, "K": k, "tau_s": tau, "limit_deg": lim, "limit_rad": math.radians(lim),
            "joints": joints, "sign": float(sign), "lr": lr, "gate": gate,
            "debounce_s": deb, "slew_deg_per_tick": slew, "slew_rad": math.radians(slew),
            "roll_clamp_deg": rclamp, "roll_clamp_rad": math.radians(rclamp)}


class RollComp:
    """느린 기울기(롤) 보정기 — 기본 OFF. 시뮬 eval_hang2stand.py --roll_comp 와 같은 법칙.

    한 틱(50Hz):
      roll  = clamp(atan2(g_y, −g_z), ±roll_clamp)   g = 정책이 받은 투영중력(장착 편향 보정 후,
                                                        시뮬 robot.data.projected_gravity_b 와 같은 부호 규약)
      raw_gate  = 정책 구동 중 AND 접촉 비트 조건(both/any) AND (브리지) 접지 확인 상태
      eff_gate  = raw_gate 또는 raw-gate-off 가 debounce_s 미만으로 이어진 동안(디바운스 유지)
      f  += (dt/τ)·(roll − f)      raw_gate 일 때만. 디바운스 중엔 f 유지(=오프셋 유지),
            eff_gate 해제 뒤엔 f += (dt/τ)·(0 − f) 로 **같은 τ 로 0 을 향해 감쇠**(계단 없음)
      off = off_prev + clamp(clamp(sign·K·f, ±limit) − off_prev, ±slew)   ← 출력 슬루 제한
      → 선택 관절 2개에 좌 lr[0]·우 lr[1](±1) 배수로 가산

    **오프셋은 절대 계단으로 뛰지 않는다**(0913 적대검수 F2): 게이트 해제·접촉 채터·센서 1틱
    누락 어디에서도 틱당 |Δoff| ≤ slew(기본 0.25°/틱 = 12.5°/s). 즉시 0 으로 떨구는 하드
    리셋은 정책 (재)명령·인계 워밍업·비구동(reset())에서만 — 그 구간은 명령 자체가 없거나
    stage8 슬루가 받는 자리다.

    적용 자리(브리지 _tick): 러너 슬루 → stand LPF → **타깃 발진 가드(osc_tgt) 뒤** →
    [여기서 오프셋 가산] → 관절→모터 프레임 변환 → 온셋 블렌딩 → 송출. 근거(F4):
      · 가드 보정 보존 — osc_tgt 는 '정책 체인 출력'으로 오트립 0 을 맞춰 교정한 가드다
        (march_v3 30초 재생 0회, 채터 0.2s 포착). 그 입력에 외부 루프 신호를 섞으면 교정이
        무효가 된다. 보정기 출력은 구조적으로 느리다(틱당 ≤0.25°·|off| ≤ limit·τ ≥ 0.5s)
        → 발진할 수 없으므로 가드가 볼 필요가 없다. 가드 임계·판정식은 일절 불변.
      · 기록은 최종 명령 기준 — targets_deg_joint/targets_deg_motor 는 오프셋 포함(실제 송출),
        분리 분석용으로 targets_deg_joint_pre_rc·roll_comp_deg·roll_comp_gate·roll_comp_raw_gate
        를 함께 남긴다.
      · 시뮬과의 차이 — 시뮬 pre 자리는 액션 지연 버퍼 앞(오프셋이 지연·슬루·백래시를 통과),
        브리지는 송출 직전이며 그 뒤 stage8 슬루·전송 지연이 같은 몫을 한다. 온셋 블렌딩
        1.5s 동안은 브리지 쪽 오프셋만 α(0→1) 배로 **작아진다**(보수적).
      · LPF 상태 오염 금지 — stand 계열에서 targets_rad 는 self.tgt_filt **그 객체**다.
        제자리 가산하면 오프셋이 LPF 상태로 되먹임돼 누적한다. add_to 는 새 리스트를 낸다.
    """

    def __init__(self, cfg=None, dt=TICK_SEC):
        self.cfg = cfg or None
        self.dt = float(dt)
        self.f = 0.0            # 저역 필터 상태 [rad]
        self.off = 0.0          # 마지막 출력 오프셋 [rad] (슬루 제한 뒤)
        self.gate_on = False    # 실효 게이트(디바운스 포함)
        self.raw_gate = False   # 원시 게이트(접촉·구동 조건 그대로)
        self.gate_off_ticks = 0
        self.debounce_ticks = (max(1, int(round(self.cfg["debounce_s"] / self.dt)))
                               if self.cfg else 0)
        self.cols = ([JOINT_ORDER.index(f"left_{self.cfg['joints']}_joint"),
                      JOINT_ORDER.index(f"right_{self.cfg['joints']}_joint")]
                     if self.cfg else [])

    @property
    def enabled(self):
        return self.cfg is not None

    def reset(self):
        """하드 리셋 — 정책 (재)명령·워밍업 프라임·비구동에서만 (그 구간은 명령 자체가 없다)."""
        self.f = 0.0
        self.off = 0.0
        self.gate_on = False
        self.raw_gate = False
        self.gate_off_ticks = 0

    def update(self, grav, contact2, active=True, extra_gate=True):
        """한 틱 갱신 → (오프셋 [rad], 실효 게이트). 원시 게이트는 self.raw_gate.

        extra_gate: 브리지의 접지 확인(HangGuard 가 PRE_CONTACT 를 벗어난 상태) 등 추가 조건.
        active=False(정책 비구동) 는 하드 리셋 — 명령이 나가지 않는 상태이므로 계단 위험 없음.
        """
        if self.cfg is None:
            return 0.0, False
        if not active:
            self.reset()
            return 0.0, False
        if grav is None:
            grav = (0.0, 0.0, -1.0)      # 코어가 아직 중력벡터를 안 채운 틱: roll 0 으로 취급(게이트만 흐름)
        c0 = float(contact2[0]) > 0.5
        c1 = float(contact2[1]) > 0.5
        raw_gate = bool(extra_gate) and ((c0 and c1) if self.cfg["gate"] == "both" else (c0 or c1))
        self.raw_gate = raw_gate
        self.gate_off_ticks = 0 if raw_gate else self.gate_off_ticks + 1
        eff_gate = raw_gate or self.gate_off_ticks < self.debounce_ticks
        self.gate_on = eff_gate
        # 축: roll = atan2(g_y, −g_z) (종전) / pitch = atan2(g_x, −g_z).
        # 두 식 모두 시뮬 mdp_hang.torso · 브리지 _roll_pitch_deg 와 같은 정의. pitch + = 앞기울임.
        if self.cfg.get("axis", "roll") == "pitch":
            ang = math.atan2(float(grav[0]), -float(grav[2]))
        else:
            ang = math.atan2(float(grav[1]), -float(grav[2]))
        if not math.isfinite(ang):       # 상류 관측 게이트가 이미 막지만 NaN 명령은 절대 금지
            ang = 0.0
        rc = self.cfg["roll_clamp_rad"]
        ang = max(-rc, min(rc, ang))     # 공중 40° 같은 큰 기울기가 필터를 끌고 가지 못하게
        if raw_gate:
            self.f += (self.dt / self.cfg["tau_s"]) * (ang - self.f)
        elif not eff_gate:               # 디바운스 창 밖 = 같은 τ 로 0 을 향해 감쇠
            self.f += (self.dt / self.cfg["tau_s"]) * (0.0 - self.f)
        # (디바운스 중에는 f 유지 = 오프셋 유지)
        lim = self.cfg["limit_rad"]
        tgt = max(-lim, min(lim, self.cfg["sign"] * self.cfg["K"] * self.f))
        sl = self.cfg["slew_rad"]
        self.off += max(-sl, min(sl, tgt - self.off))
        return self.off, eff_gate

    def add_to(self, targets_rad, off):
        """선택 관절 2개에만 lr 배수로 가산한 **새 리스트**. off=0/OFF 면 입력을 그대로 돌려준다."""
        if self.cfg is None or off == 0.0:
            return targets_rad
        out = list(targets_rad)
        out[self.cols[0]] += self.cfg["lr"][0] * off
        out[self.cols[1]] += self.cfg["lr"][1] * off
        return out


#: 중력 정합 게이트 (0905 감사 A4): stand/walk/march 시작 시 |g−(0,0,−1)| 가 이 값을 넘으면 기동 거부.
#: 장착보정 미적용 IMU 는 (±1,0,0) 근처(거리 1.41), 17.5° 이상 기운 착지는 0.3 을 넘는다(거리 = 2·sin(θ/2)).
GRAV_GATE_TOL = 0.3
GRAV_GATE_MAX_AGE_S = 0.5


def _roll_pitch_deg(quat_wxyz):
    """IMU 쿼터니언 → (롤, 피치)[deg] — 시뮬 mdp_hang.torso 식과 같은 정의(투영중력 pg: roll=atan2(pg_y,−pg_z), pitch=atan2(pg_x,−pg_z)).
    A1 기록 전용(0910). 판정용 각도는 콘솔 fwd/lat 와 대조해 부호를 확정한 뒤 쓴다."""
    try:
        g = projected_gravity(tuple(float(v) for v in quat_wxyz))
        return [round(math.degrees(math.atan2(g[1], -g[2])), 3), round(math.degrees(math.atan2(g[0], -g[2])), 3)]
    except Exception:  # noqa: BLE001 — 기록 실패가 제어를 막으면 안 된다
        return None


def gravity_gate(imu, imu_ts, now, tol=GRAV_GATE_TOL, max_age=GRAV_GATE_MAX_AGE_S):
    """(통과 여부, 사유). imu = (gyro, quat_wxyz) 또는 None(미수신)."""
    if imu is None:
        return False, "IMU 미수신"
    if now - imu_ts > max_age:
        return False, f"IMU 스테일 (age {now - imu_ts:.2f}s > {max_age}s)"
    g = projected_gravity(imu[1])
    dist = math.sqrt(g[0] ** 2 + g[1] ** 2 + (g[2] + 1.0) ** 2)
    if not math.isfinite(dist) or dist > tol:
        return False, (f"중력벡터 정합 실패 |g−(0,0,−1)|={dist:.2f} > {tol} "
                       f"(g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f})) — IMU 장착보정 미적용 또는 기울어진 착지")
    return True, f"중력 정합 OK (|g−(0,0,−1)|={dist:.2f})"


def warmup_gate(fresh, warmed, now, warmup_until):
    """stand 직후 인계 워밍업 게이트(순수 함수 — 셀프테스트 공유).

    반환 (proceed, action):
      proceed=True  → 정책 경로로 진행(이미 워밍업 완료거나 방금 신선해짐).
      (False,'prime')   → 홀드-프라임 후 대기 (창 이내, 아직 미신선).
      (False,'timeout') → 워밍업 창 초과, 신선 피드백 미확보 → 정지.
    warmed=True 이후엔 신선도와 무관하게 proceed=True → 정상상태 게이트(0.2s)가 뒤에서 판정(불변)."""
    if warmed or fresh:
        return True, "ok"
    return (False, "prime") if now < warmup_until else (False, "timeout")


class BridgeCore:
    """ROS 무관 코어 — 관측 → 12관절 목표각(rad, URDF 관절 프레임). 셀프테스트 공유.

    모터 인코더 프레임 변환은 코어 밖(FrameMap) 담당 — 코어 입출력은 전부 관절 프레임.
    """

    def __init__(self, checkpoint="walk", runner=None):
        # runner: selftest 전용 주입(스텁 정책) — None(기본·실행 경로)이면 종전대로 _make_runner
        self.runner = _make_runner(checkpoint) if runner is None else runner
        self.ckpt_name = checkpoint
        self.hh = HeadingHold(kp=HH_DEFAULT_KP, wz_limit=HH_DEFAULT_WZ_MAX)
        self.hh_ref_set = False
        self.hh_target = 0.0         # 1006 HTURN2: 목표 요 변화[rad] (시작 요 기준). HTURN2 키가 아니면 늘 0
        self.hh_yaw0 = None          # 1006 HTURN2: HeadingHold 기준을 잡은 시작 요
        self.cmd = (0.0, 0.0, 0.0)
        self.heading_hold = True
        self.active = False          # stand/walk/march 시작 여부
        self.last_grav = None        # 이번 틱에 정책이 받은 투영중력(장착 편향 보정 후) — roll_comp 가 같은 벡터를 쓴다
        self._apply_hh_cfg()
        self.w3hl = None             # 1007 W3 헤딩 고리(명령 채널 키 + 고리 켜짐에서만) — None 이면 tick 은 종전과 같다
        self.w3hl_on = True          # 1007 이번 걷기에서 고리 사용(step 명령 heading_loop) — 고리 객체가 있을 때만 의미
        self.w3hl_events = []        # 1007 고리 고장 → 걷기 정지 사건(노드가 블랙박스·로그로 비움)
        self.w3mon = None            # 1007 2 차 대비책 감시기(--w3-monitor + W3 명령 채널 키에서만) — None 이면 tick 은 종전과 같다
        self._apply_w3_cfg()

    @property
    def w3(self):
        """이 코어가 W3 계약으로 도는지(실험 플래그 켜짐 + W3 키)."""
        return self.ckpt_name in W3_CKPTS

    def set_gait_enable(self, want):
        """1006 W3: 명령이 원하는 gait_enable. 실제 전이는 러너 래치(0→1 다음 틱, 1→0 반주기 경계)."""
        if not self.w3:
            raise ValueError("W3 키가 아님 — gait_enable 입력 불가")
        self.runner.set_gait_enable(bool(want))

    @property
    def w3cmd(self):
        """1007: 이 코어가 W3 명령 채널 키인지(vx·vy·wz 검증 범위 등록)."""
        return self.ckpt_name in W3_CMD_LIMITS

    def _apply_w3_cfg(self):
        self.w3hl = _W3.W3HeadingLoop(W3_HL_CFG) if (self.ckpt_name in W3_CMD_LIMITS and W3_HL_CFG is not None) else None
        self.w3hl_on = True
        if W3_MON_CFG is not None and self.ckpt_name in W3_CMD_LIMITS:
            if self.w3mon is None:       # 세션 상태(요 표류 추정·고리 금지)는 정책 재시작에도 유지, 걷기 상태만 초기화
                self.w3mon = _w3c().W3WalkMonitor(W3_MON_CFG, JOINT_ORDER)
            else:
                self.w3mon.reset_walk_state()
        else:
            self.w3mon = None

    def _w3_cmd_step(self, quat_wxyz, vx, vy, wz):
        """1007 W3 명령 채널: 걷는 중(want 1)·비회전(wz 0)·고리 사용이면 wz = 헤딩 고리 출력, 아니면 ψ* 해제하고 그대로.
        고리 고장(오차 상한·타임아웃·비유한 요) → 걷기 정지(want 0, 명령 0 = 시뮬 정지 계약)·사건 기록."""
        clk = self.runner._w3_clock
        mon = self.w3mon
        hl_ok = True
        if mon is not None and clk.want and self.w3hl_on and self.w3hl is not None:
            hl_ok, _hw = mon.hl_allowed()
            if not hl_ok:                # 1007 감시: IMU 요 이상 → 고리 끔(보정 없음, 운영자 wz₀ 그대로) — 사건 1 회
                self.w3hl_on = False
                self.w3hl_events.append({"event": "w3_hl_banned", "why": _hw})
        if self.w3hl is None or not (clk.want and self.w3hl_on and wz == 0.0):
            if self.w3hl is not None:
                self.w3hl.release()
            return vx, vy, wz
        yaw = yaw_from_quat_wxyz(*quat_wxyz)
        wz_out, fault = self.w3hl.update(yaw, wz)
        if fault is None and mon is not None:
            if self.w3hl.n * self.w3hl.cfg.dt > mon.hl_timeout_s(self.w3hl.cfg.timeout_s) + 1e-9:
                fault = self.w3hl.fault = self.w3hl.last_fault = (
                    f"헤딩 고리 타임아웃 {mon.hl_timeout_s(self.w3hl.cfg.timeout_s):.1f}s(요 표류율 {mon.yaw_rate_dpm}°/min 로 단축)")
            else:
                cap = mon.hl_wz_cap(vx, vy)
                if cap is not None:      # 1007 감시 E: 병진 중 고리 wz 상한(단일 축 학습 — 병진+회전은 학습 밖)
                    wz_out = max(-cap, min(cap, wz_out))
        if fault is None:
            return vx, vy, wz_out
        self.runner.set_gait_enable(False)
        self.cmd = (0.0, 0.0, 0.0)
        self.w3hl_events.append({"event": "w3_hl_fault", "why": fault, "hl": self.w3hl.snapshot()})
        self.w3hl.release()
        return 0.0, 0.0, 0.0

    def _w3_mon_step(self, gyro, grav, quat_wxyz, contact2, vx, vy, wz):
        """1007 감시기 한 틱: 판정(자세·접촉·요·시간) → act 모드 정지 요청이면 헤딩 고리 고장과 같은 경로로 걷기 정지
        (want 0 + 명령 0 → 반주기 경계에서 같은 정책 STAND). 사건은 w3hl_events 로 노드에 넘긴다(노드가 정지 래치를 건다)."""
        clk = self.runner._w3_clock
        mon = self.w3mon
        mon.tick(walking=bool(clk.want or clk.ge), contact2=contact2, gyro=gyro, grav=grav,
                 yaw=yaw_from_quat_wxyz(*quat_wxyz), hl_latched=bool(self.w3hl is not None and self.w3hl.latched))
        while mon.events:
            self.w3hl_events.append(mon.events.pop(0))
        st = mon.take_stop()
        if st is None or not clk.want:
            return vx, vy, wz
        self.runner.set_gait_enable(False)
        self.cmd = (0.0, 0.0, 0.0)
        if self.w3hl is not None:
            self.w3hl.release()
        self.w3hl_events.append({"event": "w3_mon_stop", "kind": st[0], "why": st[1]})
        return 0.0, 0.0, 0.0

    @property
    def hturn2(self):
        """이 코어가 HTURN2 계약으로 도는지(실험 플래그 켜짐 + HTURN2 키)."""
        return self.ckpt_name in HTURN2_CKPTS

    def _apply_hh_cfg(self):
        c = HTURN2_CKPTS.get(self.ckpt_name)
        self.hh.kp, self.hh.wz_limit = (c.hh_kp, c.hh_wz_max) if c is not None else (HH_DEFAULT_KP, HH_DEFAULT_WZ_MAX)

    def set_heading_target(self, rad):
        """1006 HTURN2: 목표 요 = 시작 요 + rad (eval_walk --heading_target 와 같은 정의). 기준을 잡기 전이면 첫 틱에 반영."""
        if not self.hturn2:
            raise ValueError("HTURN2 키가 아님 — 목표 각 입력 불가")
        self.hh_target = float(rad)
        if self.hh_ref_set and self.hh_yaw0 is not None:
            self.hh.set_ref(self.hh_yaw0 + self.hh_target)

    def reset(self, checkpoint=None):
        if checkpoint and checkpoint != self.ckpt_name:
            self.runner = _make_runner(checkpoint)
            self.ckpt_name = checkpoint
        self.runner.reset()
        self.hh_ref_set = False
        self.hh_target = 0.0
        self.hh_yaw0 = None
        self._apply_hh_cfg()
        self._apply_w3_cfg()

    def tick(self, gyro, quat_wxyz, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=None):
        """관측 → 목표각 12 [rad, 절대] (JOINT_ORDER 순).
        tau_knee_nm(1007 odic 전용): (왼, 오른) 무릎 관절 프레임 토크 또는 None — 러너가 요구할 때(needs_knee_tau)만 넘긴다."""
        w, x, y, z = quat_wxyz
        grav = correct_gravity(projected_gravity(quat_wxyz))   # world -z 를 body 프레임으로 (게이트와 동일식) + 장착 편향 보정(맵 키 있을 때만)
        self.last_grav = grav
        vx, vy, wz = self.cmd
        if self.heading_hold and abs(wz) < 1e-6:
            yaw = yaw_from_quat_wxyz(w, x, y, z)
            if not self.hh_ref_set:
                self.hh_yaw0 = yaw
                self.hh.set_ref(yaw if self.hh_target == 0.0 else yaw + self.hh_target)   # 목표 0 = 종전 그대로
                self.hh_ref_set = True
            wz = self.hh.update(yaw)
        if self.w3mon is not None:       # 1007 2 차 대비책 감시(--w3-monitor 일 때만) — 정지 판정은 이번 틱 명령·래치에 반영
            vx, vy, wz = self._w3_mon_step(gyro, grav, quat_wxyz, contact2, vx, vy, wz)
        if self.w3hl is not None:        # 1007 W3 명령 채널 + 헤딩 고리(그 밖 키는 이 줄을 지나지 않는다)
            vx, vy, wz = self._w3_cmd_step(quat_wxyz, vx, vy, wz)
        if getattr(self.runner, "needs_knee_tau", False):
            return self.runner.step(gyro, grav, (vx, vy, wz),
                                    qpos_rad, qvel_rad_s, contact2, tau_knee_nm=tau_knee_nm)
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


def _selftest_policy_one(name):
    import numpy as np
    core = BridgeCore(name)
    core.cmd = (0.0, 0.0, 0.0)
    core.reset()
    q = np.array(core.runner._limits_lo) * 0.0 if core.runner._limits_lo is not None \
        else np.zeros(12)
    gyro = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    qvel = np.zeros(12)
    # 러너 계약(policy_runner_stage4.step 주석, 0802 회귀 수정): 브리지는 타깃을
    # 리밋으로 클램프하지 않는다 — 학습 env 와 1:1. 위치 울타리는 Stage8 노드의
    # 절대각 검사가 '방출 명령' 기준으로 담당. 따라서 여기서 검증할 것은
    # ① 유한성 ② 슬루 상한(4°/틱) 준수 ③ 소프트 리밋 초과량은 정보로만 보고
    # (목 센서 루프에는 물리가 없어 리밋 밖 표류가 정상 — 리밋 준수 단정은 오검사였음).
    prev = q.copy()
    slew_max = 0.0
    exc_max, exc_j = 0.0, -1
    t0 = time.perf_counter()
    for i in range(500):
        out = core.tick(gyro, quat, q, qvel, [1.0, 1.0])
        assert np.isfinite(out).all(), f"{name} 틱 {i}: 비유한 목표"
        d = np.abs(out - prev)
        slew_max = max(slew_max, float(d.max()))
        assert d.max() <= SLEW_RAD * 1.001 + 1e-6, (
            f"{name} 틱 {i}: 슬루 초과 {math.degrees(d.max()):.2f}° > 4°/틱 (관절 {int(d.argmax())})")
        if core.runner._limits_lo is not None:
            e = np.maximum(core.runner._limits_lo - out, out - core.runner._limits_hi)
            if float(e.max()) > exc_max:
                exc_max, exc_j = float(e.max()), int(e.argmax())
        prev = out
        q = out  # 목표 추종 가정 (목)
    dt = (time.perf_counter() - t0) / 500
    assert dt < TICK_SEC, f"{name} 틱 연산 {dt*1000:.2f}ms ≥ 20ms — 젯슨에서 재확인 필요"
    info = (f"소프트리밋 초과 {math.degrees(exc_max):+.1f}°(관절 {exc_j}, Stage8 클램프 영역)"
            if exc_max > 0 else "소프트리밋 내")
    return dt, core.runner.backend, f"슬루 최대 {math.degrees(slew_max):.2f}°/틱, {info}"


def _selftest_policy(required=()):
    """(5) 목 센서(정지 직립) 500틱 — 유한성·리밋·주기 (관절 프레임 그대로).

    0905 감사(low): 'walk' 만 돌려 신규 stand_v14~ 등록(경로·로드·gait_freq·stand LPF 분기)이
    자가검증에서 한 번도 실행되지 않았다 → walk + 레지스트리의 stand_* 전부 순회, stand_* 인데
    CKPT_GAIT_FREQ 에 없으면 FAIL.
    0905 감사 A5: 파일이 없는 등록 항목(미배포 세대)은 FAIL 이 아니라 SKIP 으로 집계 — 단 `required`
    (기동 --checkpoint 와 STAND_DEFAULT)가 없으면 FAIL (배포 패키지 누락).
    """
    names = ["walk"] + [k for k in CHECKPOINTS if k.startswith("stand_")] + [k for k in required if k in HTURN2_CKPTS]
    missing_freq = [k for k in names if k.startswith("stand_") and k not in CKPT_GAIT_FREQ]
    assert not missing_freq, f"CKPT_GAIT_FREQ 누락: {missing_freq}"
    assert STAND_DEFAULT in CHECKPOINTS and STAND_DEFAULT in CKPT_GAIT_FREQ, f"STAND_DEFAULT 미등록: {STAND_DEFAULT}"
    res, skipped = [], []
    backend = "-"
    for n in names:
        d = CHECKPOINTS[n]
        if not (os.path.exists(os.path.join(d, "policy.pt")) or os.path.exists(os.path.join(d, "policy.onnx"))):
            skipped.append(n)
            continue
        dt, backend, note = _selftest_policy_one(n)
        res.append(f"{n} {dt*1000:.2f}ms [{note}]")
    need = [n for n in required if n in skipped or n not in CHECKPOINTS]
    assert not need, f"필수 정책 파일 없음: {need} (배포 패키지 누락?) — SKIP 불가"
    assert res, "실행된 정책이 없음"
    skip_note = f" | SKIP {len(skipped)} 미배포: {', '.join(skipped)}" if skipped else ""
    return (f"500틱 유한·슬루 준수 ×{len(res)}: " + "; ".join(res)
            + f" (50Hz 예산 20ms, 백엔드 {backend}){skip_note}")


def _selftest_roll_pitch():
    """(A1 기록) 직립 → (0,0); y축 +10° 회전 쿼터니언 → 피치 ±10°(부호는 정의상 pg_x 기준), 롤 0; 잘못된 입력 → None."""
    rp = _roll_pitch_deg((1.0, 0.0, 0.0, 0.0)); assert rp is not None and abs(rp[0]) < 1e-6 and abs(rp[1]) < 1e-6, rp
    h = math.radians(5.0); rp = _roll_pitch_deg((math.cos(h), 0.0, math.sin(h), 0.0))
    assert rp is not None and abs(abs(rp[1]) - 10.0) < 1e-3 and abs(rp[0]) < 1e-3, rp
    assert _roll_pitch_deg(None) is None
    return f"직립 (0,0) · y축 10° → 피치 {rp[1]:+.1f}° 롤 {rp[0]:.1f}° · 불량 입력 None"


def _selftest_imu_mount():
    """(0913) 장착 편향 보정: 편향 0 → 항등 / 편향만큼 기운 IMU → 보정 후 (0,0,−1) / 추가 10° 기울기는 보정 뒤에도 10° 로 남음 / 범위 밖 값 거부."""
    g0 = (0.0, 0.0, -1.0)
    assert correct_gravity(g0, [0.0, 0.0]) == g0
    def _q(roll_deg, pitch_deg):   # x축 롤 → y축 피치 순 작은 회전 쿼터니언 (w,x,y,z)
        r, p = math.radians(roll_deg) / 2, math.radians(pitch_deg) / 2
        qr = (math.cos(r), math.sin(r), 0.0, 0.0); qp = (math.cos(p), 0.0, math.sin(p), 0.0)
        w1, x1, y1, z1 = qr; w2, x2, y2, z2 = qp
        return (w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2)
    q_b = _q(-1.3, -0.6); rp_b = _roll_pitch_deg(q_b)          # 실물 편향과 같은 크기의 기울기
    g_c = correct_gravity(projected_gravity(q_b), rp_b)
    assert max(abs(g_c[0]), abs(g_c[1]), abs(g_c[2] + 1.0)) < 1e-4, g_c     # rp_deg 는 0.001° 로 반올림되므로 잔차 ~1e-6
    q_t = _q(-1.3 + 10.0, -0.6); g_t = correct_gravity(projected_gravity(q_t), rp_b)
    roll_t = math.degrees(math.atan2(g_t[1], -g_t[2]))
    assert abs(abs(roll_t) - 10.0) < 0.05, roll_t
    for _bad in ({"lat": 9.0, "fwd": 0.0}, {"lat": float("nan"), "fwd": 0.0}, {"lat": True, "fwd": 0.0}, {"lat": "1.3", "fwd": 0.0}):
        try:
            _validate_imu_mount_bias(_bad); raise AssertionError(f"범위/타입 검사 미작동: {_bad}")
        except ValueError:
            pass
    assert _validate_imu_mount_bias({"lat": 1.3, "fwd": -0.6}) == (1.3, -0.6)
    # 큰 기울기(> ~73°)에선 보정 생략(원시값 그대로)
    g_fall = (0.0, 0.95, -0.2); assert correct_gravity(g_fall, [1.3, -0.6]) == g_fall
    return f"편향 {rp_b[0]:+.2f}/{rp_b[1]:+.2f}° 보정 → (0,0,−1) 오차 <1e-4 · +10° 기울기 → {abs(roll_t):.2f}° · 한계 {IMU_MOUNT_BIAS_MAX_DEG}° · 범위/타입 4종 거부 · 73° 초과 기울기 보정 생략"


def _selftest_roll_comp():
    """(0913) 느린 기울기 롤 보정기 — 시뮬 eval_hang2stand.py --roll_comp 와 같은 법칙인지 검증.

    (a) 맵 키 없음/enabled=false → 완전 OFF (목표 리스트 객체 그대로 = 출력 비트 동일)
    (b) 양발 접촉 + 롤 +3° 를 5s(50Hz) → 오프셋 → sign·K·3° (5% 이내), 선택 관절 2개만 변화
    (c) F2 계단 금지: 1틱 누락 무영향(디바운스) · 10Hz 게이트 채터에도 |Δoff| ≤ slew ·
        게이트 8s 해제 → 초기값의 1% 미만으로 **감쇠**(즉시 0 아님) · 롤 40° → ±roll_clamp
    (d) ±limit 클램프 · sign=−1 반전
    (e) 봉투 밖 설정 거부 (F5: K·τ·limit(관절별)·gate none·lr 비±1·debounce·slew·오타·타입)
    """
    base = [0.1 * (k + 1) for k in range(12)]

    def grav(deg):   # roll=atan2(g_y,−g_z)=deg 인 단위 중력벡터
        return (0.0, math.sin(math.radians(deg)), -math.cos(math.radians(deg)))
    g3, g5, g40 = grav(3.0), grav(5.0), grav(40.0)
    assert abs(math.degrees(math.atan2(g3[1], -g3[2])) - 3.0) < 1e-9
    # (a) OFF
    assert _validate_roll_comp(None) is None
    assert _validate_roll_comp({"enabled": False, "K": 1.2, "joints": "hip_a"}) is None
    rc_off = RollComp(_validate_roll_comp(None))
    assert not rc_off.enabled
    assert rc_off.update(g3, [1.0, 1.0]) == (0.0, False)
    assert rc_off.add_to(base, 0.0) is base          # 새 객체조차 만들지 않음
    # (b) 수렴
    cfg = _validate_roll_comp({"enabled": True, "K": 1.0, "tau_s": 1.0, "limit_deg": 5.0,
                               "joints": "ankle_r", "sign": 1, "lr": [1, 1], "gate": "both"})
    assert cfg["debounce_s"] == 0.2 and cfg["slew_deg_per_tick"] == 0.25 and cfg["roll_clamp_deg"] == 15.0
    rc = RollComp(cfg)
    assert rc.debounce_ticks == 10
    off = gate = None
    for _ in range(250):                            # 5s @ 50Hz
        off, gate = rc.update(g3, [1.0, 1.0])
    conv = math.degrees(off)
    assert gate and abs(conv - 3.0) <= 3.0 * 0.05, conv
    out = rc.add_to(base, off)
    chg = [k for k in range(12) if out[k] != base[k]]
    assert chg == [JOINT_ORDER.index("left_ankle_r_joint"),
                   JOINT_ORDER.index("right_ankle_r_joint")] == [10, 11], chg
    assert abs((out[10] - base[10]) - off) < 1e-12 and abs((out[11] - base[11]) - off) < 1e-12
    assert base == [0.1 * (k + 1) for k in range(12)], "입력 리스트를 건드렸다(LPF 상태 오염 경로)"
    # joints=hip_a → 열 2·3, lr 좌우 배수 각각 (hip_a 기본 한계 3°)
    cfg_h = _validate_roll_comp({"enabled": True, "joints": "hip_a", "lr": [1, -1]})
    assert cfg_h["limit_deg"] == 3.0
    rc_h = RollComp(cfg_h)
    oh = rc_h.add_to(base, 0.05)
    assert [k for k in range(12) if oh[k] != base[k]] == [2, 3], oh
    assert abs((oh[2] - base[2]) - 0.05) < 1e-12 and abs((oh[3] - base[3]) + 0.05) < 1e-12
    # (c1) 1틱 누락 = 디바운스 → 오프셋 불변, 원시 게이트만 내려감
    o_before = rc.off
    o_drop, g_drop = rc.update(g3, [1.0, 0.0])
    assert o_drop == o_before and g_drop is True and rc.raw_gate is False, (o_drop, o_before)
    assert rc.update(g3, [1.0, 1.0])[1] is True
    # (c2) 10Hz 게이트 채터 5s + 롤 5° → 틱당 |Δoff| ≤ slew (계단 없음)
    rc_ch = RollComp(_validate_roll_comp({"enabled": True, "K": 1.0, "tau_s": 1.0,
                                          "limit_deg": 5.0, "joints": "ankle_r"}))
    prev, dmax, eff_off = 0.0, 0.0, 0
    for i in range(250):
        on = (i // 5) % 2 == 0                      # 0.1s on / 0.1s off = 10Hz 토글
        o, g = rc_ch.update(g5, [1.0, 1.0] if on else [1.0, 0.0])
        dmax = max(dmax, abs(o - prev)); prev = o
        eff_off += (not g)
    assert dmax <= rc_ch.cfg["slew_rad"] + 1e-15, math.degrees(dmax)
    assert eff_off == 0, "10Hz 채터는 디바운스(0.2s) 안쪽이라 실효 게이트가 내려가면 안 된다"
    # (c3) 게이트 8s 해제 → 즉시 0 이 아니라 감쇠, 초기값의 1% 미만
    o0 = rc_ch.off
    assert abs(o0) > math.radians(1.0), math.degrees(o0)
    dmax2 = 0.0
    for _ in range(400):                            # 8s
        o, g = rc_ch.update(g5, [0.0, 0.0])
        dmax2 = max(dmax2, abs(o - prev)); prev = o
    assert dmax2 <= rc_ch.cfg["slew_rad"] + 1e-15, math.degrees(dmax2)
    assert abs(o) < 0.01 * abs(o0), (math.degrees(o), math.degrees(o0))
    assert not g, "8s 해제 뒤엔 실효 게이트도 내려가 있어야"
    # (c4) 롤 40° → roll_clamp 15° 로 잘려 필터가 15° 로 수렴
    rc_cl = RollComp(_validate_roll_comp({"enabled": True, "K": 1.0, "tau_s": 0.5,
                                          "limit_deg": 5.0, "joints": "ankle_r"}))
    for _ in range(500):
        rc_cl.update(g40, [1.0, 1.0])
    f_cl = math.degrees(rc_cl.f)
    assert abs(f_cl - 15.0) < 0.05, f_cl
    # (c5) 정책 비구동·접지 미확인(F3) → 하드 리셋 / 게이트 OFF
    assert rc_cl.update(g40, [1.0, 1.0], active=False) == (0.0, False) and rc_cl.f == 0.0
    rc_ex = RollComp(_validate_roll_comp({"enabled": True}))
    assert rc_ex.update(g3, [1.0, 1.0], True, False)[0] == 0.0 and rc_ex.raw_gate is False
    # (c6) F3 접지 확인 게이트 — 행잉 ARM 에선 HangGuard 가 PRE_CONTACT 를 벗어나기 전까진 보정기 OFF
    hg = HangGuard(hang_mode=True, t0=0.0)
    rc_f3 = RollComp(_validate_roll_comp({"enabled": True}))
    assert hg.state == HangGuard.PRE_CONTACT
    # 접지 미확인 = raw 게이트 OFF → 필터가 0 에 머물러 오프셋 0 (디바운스 창이라 실효 게이트는 아직 True)
    assert rc_f3.update(g3, [1.0, 1.0], True, hg.state != HangGuard.PRE_CONTACT)[0] == 0.0
    assert rc_f3.raw_gate is False and rc_f3.f == 0.0
    for _k in range(20):                            # 0.4s 양발 접촉 → 접지 확인(CONTACT_CONFIRM_S 0.3)
        hg.update(0.02 * _k, [1.0, 1.0])
    assert hg.state == HangGuard.CONTACT_ACQUIRED, hg.state
    _o3, _g3 = rc_f3.update(g3, [1.0, 1.0], True, hg.state != HangGuard.PRE_CONTACT)
    assert _g3 and _o3 > 0.0, (_o3, _g3)
    # (d) 클램프 · 부호
    rc_lim = RollComp(_validate_roll_comp({"enabled": True, "K": 1.0, "tau_s": 0.5, "limit_deg": 1.0}))
    for _ in range(500):
        off_lim, _g = rc_lim.update(g3, [1.0, 1.0])
    assert abs(math.degrees(off_lim) - 1.0) < 1e-9, math.degrees(off_lim)
    rc_neg = RollComp(_validate_roll_comp({"enabled": True, "sign": -1, "tau_s": 0.5}))
    for _ in range(500):
        off_neg, _g = rc_neg.update(g3, [1.0, 1.0])
    assert abs(math.degrees(off_neg) + 3.0) < 0.05, math.degrees(off_neg)
    # (e) 봉투 밖 거부
    bads = ({"enabled": True, "K": 2.0}, {"enabled": True, "tau_s": 0.2},
            {"enabled": True, "tau_s": 6.0}, {"enabled": True, "limit_deg": 6.0},
            {"enabled": True, "joints": "hip_a", "limit_deg": 4.0},
            {"enabled": True, "gate": "none"}, {"enabled": True, "lr": [0.5, 1]},
            {"enabled": True, "debounce_s": 2.0}, {"enabled": True, "slew_deg_per_tick": 1.0},
            {"enabled": True, "roll_clamp_deg": 40.0}, {"enabled": True, "joints": "knee"},
            {"enabled": True, "sign": 0}, {"enabled": True, "lr": [1]},
            {"enabled": True, "taus": 1.0}, {"enabled": True, "K": float("nan")},
            {"enabled": "yes"}, [1, 2])
    for bad in bads:
        try:
            _validate_roll_comp(bad)
            raise AssertionError(f"봉투 밖 설정 미거부: {bad}")
        except ValueError:
            pass
    fm = load_frame_map()
    return (f"키 없음 OFF(현재 맵 {'키 없음' if fm.roll_comp is None else fm.roll_comp}) · "
            f"+3° 5s → {conv:+.3f}° (오차 {abs(conv - 3.0) / 3.0 * 100:.2f}%) · 변화 관절 {chg}(ankle_r) · "
            f"hip_a lr[1,−1] 열 2·3(한계 3°) · F3 접지확인 전 OFF·후 ON · 1틱 누락 불변 · 10Hz 채터 Δmax {math.degrees(dmax):.3f}°/틱 ≤ 0.25 · "
            f"8s 해제 감쇠 {math.degrees(o0):+.2f}°→{math.degrees(o):+.3f}° · 롤 40°→필터 {f_cl:.1f}°(클램프 15°) · "
            f"클램프 ±1.0° · sign−1 {math.degrees(off_neg):+.2f}° · 봉투 밖 {len(bads)}종 거부")


def _selftest_grav_gate():
    """(0905 감사 A4) 정합 IMU 통과 / 장착보정 미적용(−1,0,0) 거부 / 스테일·미수신 거부 / 15° 통과·20° 거부."""
    now = 1000.0
    ok, why = gravity_gate(((0, 0, 0), (1.0, 0.0, 0.0, 0.0)), now, now)
    assert ok, why
    s = math.sqrt(0.5)                      # y축 −90° 회전 → g = (−1,0,0): 장착보정 미적용 IMU 의 전형
    g = projected_gravity((s, 0.0, -s, 0.0))
    assert abs(g[0] + 1.0) < 1e-6 and abs(g[2]) < 1e-6, g
    ok, why = gravity_gate(((0, 0, 0), (s, 0.0, -s, 0.0)), now, now)
    assert not ok and "정합 실패" in why, why
    ok, why = gravity_gate(((0, 0, 0), (1.0, 0.0, 0.0, 0.0)), now - 1.0, now)
    assert not ok and "스테일" in why, why
    ok, why = gravity_gate(None, 0.0, now)
    assert not ok and "미수신" in why, why
    for deg, expect in ((15.0, True), (20.0, False)):   # 거리 = 2·sin(θ/2): 0.26 / 0.35
        h = math.radians(deg) / 2.0
        ok, _ = gravity_gate(((0, 0, 0), (math.cos(h), 0.0, math.sin(h), 0.0)), now, now)
        assert ok == expect, (deg, ok)
    return "정합 통과 · (−1,0,0) 거부 · 스테일/미수신 거부 · 15° 통과/20° 거부"


def _selftest_warmup():
    """(0911) 인계 워밍업 게이트: 미신선+창내→prime, 신선→ok, 미신선+창초과→timeout, 워밍업후→ok(정상게이트로)."""
    assert warmup_gate(fresh=False, warmed=False, now=1.0, warmup_until=5.0) == (False, "prime"), "창내 미신선→prime"
    assert warmup_gate(fresh=True, warmed=False, now=1.0, warmup_until=5.0) == (True, "ok"), "신선→진행(워밍업 완료)"
    assert warmup_gate(fresh=False, warmed=False, now=6.0, warmup_until=5.0) == (False, "timeout"), "창초과 미신선→timeout"
    assert warmup_gate(fresh=False, warmed=True, now=9.0, warmup_until=5.0) == (True, "ok"), "워밍업후 미신선→진행(뒤의 0.2s 게이트가 판정)"
    assert warmup_gate(fresh=True, warmed=True, now=9.0, warmup_until=5.0) == (True, "ok")
    return "미신선+창내 prime · 신선 완료 · 창초과 timeout · 워밍업후 정상게이트 위임"


def _parse_env_contract(path):
    """rsl_rl 이 남긴 params/env.yaml 에서 배포 계약 항목만 줄 단위로 뽑는다 (PyYAML·python 태그 불필요).
    scene.robot.init_state.joint_pos(첫 블록) · actions.joint_pos 의 scale/max_delta_per_step/use_default_offset/지연 대역 ·
    decimation · sim.dt."""
    out = {"joint_pos": {}}
    sect = None
    in_init = in_jp = done_jp = False
    jp_indent = 0
    obs_group = obs_term = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            indent = len(s) - len(s.lstrip())
            key, _, val = s.strip().partition(":")
            val = val.strip()
            if indent == 0:
                sect = key
                in_init = in_jp = False
                obs_group = obs_term = None
                if key == "decimation":
                    out["decimation"] = int(val)
                continue
            if sect == "observations":     # 1001: 정책 관측 joint_vel_ema 함수(학습 계약 legacy/v2 판별)
                if indent == 2:
                    obs_group, obs_term = key, None
                elif indent == 4:
                    obs_term = key
                elif indent == 6 and key == "func" and obs_group == "policy" and obs_term == "joint_vel_ema":
                    out["jvel_ema_func"] = val
                elif indent == 6 and key == "func" and obs_group == "policy" and obs_term in ("actions", "joint_vel"):
                    out[f"obs_{obs_term}_func"] = val          # 1003: 관측 필터 계약 판별
                elif indent == 8 and key == "beta" and obs_group == "policy" and obs_term == "joint_vel":
                    out["obs_joint_vel_beta"] = float(val)
                continue
            if sect == "sim" and key == "dt" and indent == 2:
                out["dt"] = float(val)
            elif sect == "scene" and not done_jp:
                if key == "init_state":
                    in_init, in_jp = True, False
                elif in_init and key == "joint_pos" and not val:
                    in_jp, jp_indent = True, indent
                elif in_jp:
                    if indent > jp_indent and val and not val.startswith("!!"):
                        try:
                            out["joint_pos"][key] = float(val)
                        except ValueError:
                            pass
                    elif indent <= jp_indent:
                        in_jp, done_jp = False, True
            elif sect == "actions":
                if key == "max_delta_per_step":
                    out["max_delta"] = float(val)
                elif key == "scale" and val not in ("", "null"):
                    out.setdefault("scale", float(val))
                elif key == "use_default_offset":
                    out["use_default_offset"] = (val.lower() == "true")
                elif key == "min_delay_steps":
                    out["min_delay"] = int(val)
                elif key == "max_delay_steps":
                    out["max_delay"] = int(val)
    return out


def _jvel_ema_contract(func):
    """env.yaml 의 observations.policy.joint_vel_ema.func → 'legacy'(학습 중 0+잡음, 1001 결함) | 'v2'(실제 속도 EMA) | None(항 없음)."""
    if not func:
        return None
    name = str(func).split(":")[-1].strip()
    if name == "encoder_joint_vel_ema":
        return "legacy"
    if name in ("encoder_joint_vel_ema_v2", "joint_vel_ema"):
        return "v2"
    raise ValueError(f"알 수 없는 joint_vel_ema 함수: {func}")


#: 1003 관측 필터 계약: 관측 'actions' 가 last_action_lpf 인 정책은 원시 액션 EMA(β = 배포 stand LPF β) 를 받는다(학습: 같은 env 의 액션 LPF β).
OBS_ACT_EMA_BETA_DEPLOY = 0.27


def obs_filter_contract(env_contract):
    """env.yaml 계약 → (act_obs_ema_beta, jvel_obs_ema_beta). 학습 env 가 필터 함수를 선언한 정책에만 적용 —
    그 밖의 함수(last_action·encoder_joint_vel_v2·joint_vel_rel_dropout 등 종전 정책)는 (None, None) = 종전 원시값 그대로.
    이름에 'lpf' 가 있는데 모르는 함수는 기동 거부(새 필터 계약을 조용히 원시값으로 먹이는 사고 방지)."""
    af = str(env_contract.get("obs_actions_func") or "").split(":")[-1].strip()
    vf = str(env_contract.get("obs_joint_vel_func") or "").split(":")[-1].strip()
    act = jv = None
    if af == "last_action_lpf":
        act = OBS_ACT_EMA_BETA_DEPLOY
    elif "lpf" in af:
        raise ValueError(f"알 수 없는 관측 actions 필터 함수: {af}")
    if vf == "encoder_joint_vel_lpf":
        jv = env_contract.get("obs_joint_vel_beta")
        if jv is None:
            raise ValueError("encoder_joint_vel_lpf 인데 env.yaml 에 beta 없음")
    elif "lpf" in vf:
        raise ValueError(f"알 수 없는 관측 joint_vel 필터 함수: {vf}")
    return act, jv


def ckpt_obs_filter(name):
    """체크포인트의 러너 관측 필터 β. env.yaml 이 없으면 (None, None) — 종전 계약.
    단 CKPT_REQUIRE_ENV_YAML 의 정책은 env.yaml 없이 기동 거부(필터 계약이 조용히 빠지는 것 방지, 1003)."""
    if name not in CHECKPOINTS:
        return None, None
    yml = _ckpt_env_yaml(name)
    if not os.path.isfile(yml):
        if name in CKPT_REQUIRE_ENV_YAML:
            raise ValueError(f"{name}: 관측 필터 계약 정책인데 env.yaml 없음 ({yml}) — 기동 거부")
        return None, None
    return obs_filter_contract(_parse_env_contract(yml))


def _ckpt_env_yaml(name):
    return os.path.join(os.path.dirname(CHECKPOINTS[name].rstrip("/")), "params", "env.yaml")


def ckpt_jvel_ema_mode(name):
    """체크포인트의 러너 jvel_ema_mode. 'ema0' 이면 env.yaml 로 legacy 계약을 확인하고, 아니면 기동 거부(ValueError).
    env.yaml 이 없으면 ema0 도 거부(계약을 확인할 수 없음)."""
    mode = CKPT_JVEL_EMA_MODE.get(name, "v2")
    if mode == "ema0":
        yml = _ckpt_env_yaml(name)
        if not os.path.isfile(yml):
            raise ValueError(f"{name}: ema0 인데 env.yaml 없음 ({yml}) — 학습 계약 확인 불가, 기동 거부")
        c = _jvel_ema_contract(_parse_env_contract(yml).get("jvel_ema_func"))
        if c != "legacy":
            raise ValueError(f"{name}: ema0 은 legacy 계약 정책에만 — 이 정책의 joint_vel_ema 계약 = {c} (반대 방향 결함), 기동 거부")
    return mode



def _selftest_c5():
    """(1004) C5 브리지 경로: _make_runner → C5Runner(서기 G21a 계약 공유) · 300틱 유한·슬루 · 상대 감지기(정적 3.5° 기울기에서 안 켜짐,
    +6° 계단에서 켜짐, 복귀 후 꺼짐) · 블랙박스 c5 필드. 회복 자리에 G21a 내보내기를 임시로 써서 경로만 검증(정책 성능 아님)."""
    import numpy as np
    if not os.path.isdir(CHECKPOINTS["stand_crouch_g21a"]):
        return "SKIP (G21a 미배포)"
    C5_PAIRS["_selftest_c5"] = ("stand_crouch_g21a", CHECKPOINTS["stand_crouch_g21a"])
    global C5_SETTLE
    _saved_settle, C5_SETTLE = C5_SETTLE, None      # 1007: 종전 경로 정의 검증(--c5-settle 켜진 기동에서도 imu 로 격리)
    try:
        core = BridgeCore("_selftest_c5")
        assert isinstance(core.runner, C5Runner) and core.runner.act_obs_ema_beta == 0.27, "C5 러너·관측 필터 계약"
        q = np.asarray(CKPT_DEFAULT_POSE["stand_crouch_g21a"], dtype=np.float32); prev = None; modes = []
        for k in range(300):
            ang = math.radians(3.5 + (6.0 if 100 <= k < 110 else 0.0))
            quat = (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)       # 피치 기울기
            out = core.tick((0.0, 0.0, 0.0), quat, q, np.zeros(12, np.float32), (1.0, 1.0))
            assert np.all(np.isfinite(out))
            if prev is not None:
                assert float(np.max(np.abs(out - prev))) <= math.radians(4.0) + 1e-6, "슬루 초과"
            prev = out; q = out.astype(np.float32); modes.append(core.runner.last_mode)
        bb = core.runner.blackbox()
        assert max(modes[:100]) == 0, "정적 3.5° 에서 회복 모드"
        assert max(modes[100:112]) == 1, "+6° 계단에서 안 켜짐"
        assert modes[-1] == 0, "복귀 뒤 서기로 안 돌아옴"
        assert set(bb) >= {"mode", "switch", "n_on", "ref_deg", "raw_stand", "raw_rec"}
        return f"C5 경로 OK: 켜짐 {bb['n_on']} 회 · 회복 체류 {sum(modes)} 틱 · 기준 자세 {bb['ref_deg']}"
    finally:
        C5_SETTLE = _saved_settle
        C5_PAIRS.pop("_selftest_c5", None)


def _selftest_c5kr():
    """(1005) kr 브리지 경로: C5_EXTRA 가 감지기(조기 복귀·불응기)·러너(cue)로 전달되는지, 앞 기울기(ω_y>0) 켜짐에서 cue 50 틱,
    조기 복귀로 서기 복귀, 블랙박스 cue·n_cue·n_early 필드. 회복 자리에 G21a 내보내기(경로 검증, 정책 성능 아님)."""
    import numpy as np
    if not os.path.isdir(CHECKPOINTS["stand_crouch_g21a"]):
        return "SKIP (G21a 미배포)"
    C5_PAIRS["_selftest_c5kr"] = ("stand_crouch_g21a", CHECKPOINTS["stand_crouch_g21a"])
    C5_EXTRA["_selftest_c5kr"] = C5_EXTRA["stand_c5kr_g21a_g24ra"]
    global C5_SETTLE
    _saved_settle, C5_SETTLE = C5_SETTLE, None      # 1007: 종전 경로 정의 검증(--c5-settle 켜진 기동에서도 imu 로 격리)
    try:
        core = BridgeCore("_selftest_c5kr")
        r = core.runner
        assert isinstance(r, C5Runner) and r.cue_vx == 0.03 and r.det.early is not None and r.det.refrac is not None, "kr 설정 전달"
        q = np.asarray(CKPT_DEFAULT_POSE["stand_crouch_g21a"], dtype=np.float32); prev = None; modes = []; cues = []
        for k in range(300):
            ang = math.radians(3.5 + (6.0 if 100 <= k < 110 else 0.0))
            quat = (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)
            gy = 0.6 if k == 100 else 0.0
            out = core.tick((0.0, gy, 0.0), quat, q, np.zeros(12, np.float32), (1.0, 1.0))
            assert np.all(np.isfinite(out))
            if prev is not None:
                assert float(np.max(np.abs(out - prev))) <= math.radians(4.0) + 1e-6, "슬루 초과"
            prev = out; q = out.astype(np.float32); modes.append(r.last_mode); cues.append(int(r.cue_active))
        bb = r.blackbox()
        assert max(modes[:100]) == 0 and max(modes[100:112]) == 1 and modes[-1] == 0, "켜짐·복귀"
        assert sum(cues) == min(50, sum(modes)) and bb["n_cue"] == 1, f"cue 틱 {sum(cues)} (회복 체류 {sum(modes)})"   # cue 는 회복 모드가 끝나면(조기 복귀) 같이 끝난다
        assert bb["n_early"] >= 1, "조기 복귀 미발동"
        assert {"cue", "n_cue", "n_early"} <= set(bb)
        return f"kr 경로 OK: 켜짐 {bb['n_on']} · cue {sum(cues)} 틱 · 조기 복귀 {bb['n_early']} · 회복 체류 {sum(modes)} 틱"
    finally:
        C5_SETTLE = _saved_settle
        C5_PAIRS.pop("_selftest_c5kr", None); C5_EXTRA.pop("_selftest_c5kr", None)


def _selftest_c5_odic():
    """(1007 병목 2) odic 플래그: 기본 C5_SETTLE None → kr 러너 감지기 settle 없음·needs_knee_tau False(종전 경로) / 켜면 settle 전달·
    tick 이 무릎 τ 를 넘김 · τ None 이면 조기 복귀 없음(calm 경로만) · τ 있으면 조기 복귀 · early 없는 C5 키 거부 · 설정 파싱 거부 ·
    knee_tau_joint = direction × τ_motor, 미수신·낡음 None. 회복 자리에 G21a 내보내기(경로 검증, 정책 성능 아님)."""
    import numpy as np
    if not os.path.isdir(CHECKPOINTS["stand_crouch_g21a"]):
        return "SKIP (G21a 미배포)"
    for bad in ('{"k_hat": -1}', '{"foo": 1}', '{"tau_ref": "x"}', "[1]", "nope"):
        try:
            parse_c5_settle(bad); raise AssertionError(f"--c5-settle {bad} 거부 안 됨")
        except ValueError:
            pass
    assert parse_c5_settle(None) is None and parse_c5_settle("imu") is None and parse_c5_settle("odic") == {}
    assert parse_c5_settle('{"k_hat": 35}') == {"k_hat": 35}
    assert parse_c5_settle("odicw") == {"w_thr": 0.4} and parse_c5_settle('{"w_thr": 0.3}') == {"w_thr": 0.3}
    try:
        parse_c5_settle('{"w_thr": -0.1}'); raise AssertionError("w_thr 음수 거부 안 됨")
    except ValueError:
        pass
    dmap = {"left_knee_joint": -1, "right_knee_joint": -1}
    assert knee_tau_joint({"left_knee_joint": 2.0, "right_knee_joint": -3.0}, 0.005, dmap) == (-2.0, 3.0)
    assert knee_tau_joint({"left_knee_joint": 2.0, "right_knee_joint": -3.0}, 0.2, dmap) is None, "낡은 τ 통과"
    assert knee_tau_joint({"left_knee_joint": 2.0}, 0.005, dmap) is None
    assert knee_tau_joint({"left_knee_joint": 2.0, "right_knee_joint": 1.0}, None, dmap) is None
    assert knee_tau_joint({"left_knee_joint": float("nan"), "right_knee_joint": 1.0}, 0.0, dmap) is None
    saved = C5_SETTLE
    C5_PAIRS["_selftest_odic"] = ("stand_crouch_g21a", CHECKPOINTS["stand_crouch_g21a"])
    C5_EXTRA["_selftest_odic"] = C5_EXTRA["stand_c5kr_g21a_g24ra"]
    C5_PAIRS["_selftest_odic_noearly"] = ("stand_crouch_g21a", CHECKPOINTS["stand_crouch_g21a"])
    try:
        set_c5_settle(None)
        c0 = BridgeCore("_selftest_odic")
        assert c0.runner.det.settle is None and not c0.runner.needs_knee_tau and "odic" not in c0.runner.blackbox(), "꺼짐 기본 경로"
        set_c5_settle({})
        cn = BridgeCore("_selftest_odic_noearly")
        assert cn.runner.det.settle is None and not cn.runner.needs_knee_tau, "early 없는 C5 키에 odic 적용됨(종전 imu 여야 함)"
        try:
            from rl_walking.deploy.c5_switch import SwitchDetector as _SD
            _SD(**C5_DETECTOR_KW, settle={}); raise AssertionError("early 없는 감지기에 odic 허용")
        except ValueError:
            pass
        res = {}
        for nm, tau in (("τ 있음", (-6.0, 6.0)), ("τ 없음", None)):
            core = BridgeCore("_selftest_odic"); r = core.runner
            assert r.det.settle is not None and r.needs_knee_tau
            q0 = np.asarray(CKPT_DEFAULT_POSE["stand_crouch_g21a"], dtype=np.float32); prev = None; modes = []
            for k in range(300):
                ang = math.radians(0.5 + (6.0 if 100 <= k < 110 else (2.0 if 110 <= k < 200 else 0.0)))   # 펄스 뒤 2° 잔류(calm 경로 막힘)
                quat = (math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0)
                out = core.tick((0.0, 0.6 if k == 100 else 0.0, 0.0), quat, q0, np.zeros(12, np.float32), (1.0, 1.0), tau_knee_nm=tau)
                assert np.all(np.isfinite(out))
                if prev is not None:
                    assert float(np.max(np.abs(out - prev))) <= math.radians(4.0) + 1e-6, "슬루 초과"
                prev = out; modes.append(r.last_mode)
            bb = r.blackbox()
            assert "odic" in bb and {"spd", "frac", "ds", "tau_ok", "n_tau_bad"} <= set(bb["odic"])
            res[nm] = (bb["n_early"], sum(modes), bb["odic"]["n_tau_bad"], modes[-1])
        assert res["τ 있음"][0] >= 1 and res["τ 있음"][3] == 0, f"τ 있음 조기 복귀 미발동 {res}"
        assert res["τ 없음"][0] == 0 and res["τ 없음"][2] == 300 and res["τ 없음"][1] > res["τ 있음"][1], f"τ 없음인데 조기 복귀 {res}"
        return (f"odic 플래그 OK: 기본 꺼짐(settle 없음) · early 없는 키 = imu 유지(감지기 직접 생성은 거부) · 설정 파싱 거부 5 종 · knee τ 변환·신선도 · "
                f"τ 있음 조기 복귀 {res['τ 있음'][0]}·체류 {res['τ 있음'][1]} 틱 / τ 없음 조기 0·체류 {res['τ 없음'][1]} 틱(calm 경로)")
    finally:
        set_c5_settle(saved)
        for k in ("_selftest_odic", "_selftest_odic_noearly"):
            C5_PAIRS.pop(k, None); C5_EXTRA.pop(k, None)


def _selftest_obs_filter():
    """(1003) 체크포인트별 관측 필터 계약: G12 = 종전(필터 없음), G21a = actions EMA β0.27·joint_vel EMA β0.25 (env.yaml 기준).
    BridgeCore 가 정책을 바꾸면 러너를 새로 만들어 필터·이력 상태가 비워진다(G21a → G12 전환 후 G12 러너에 필터 없음)."""
    # G21a 는 콘솔 선택지 — env.yaml 이 없으면 SKIP 이 아니라 실패(배포 누락 검출, 1003)
    assert os.path.isfile(_ckpt_env_yaml("stand_crouch_g21a")), f"stand_crouch_g21a env.yaml 없음 ({_ckpt_env_yaml('stand_crouch_g21a')}) — 배포 누락"
    g12 = ckpt_obs_filter("stand_crouch_g12"); g21 = ckpt_obs_filter("stand_crouch_g21a")
    assert g12 == (None, None), f"G12 관측 필터가 생김: {g12} — G12 기준선 오염"
    assert ckpt_obs_filter("stand_crouch_g12_ema0") == (None, None) and ckpt_jvel_ema_mode("stand_crouch_g12_ema0") == "ema0", "G12 ema0 기준선 계약 이상"
    _saved = CHECKPOINTS["stand_crouch_g21a"]                 # env.yaml 없는 G21a 는 기동 거부
    try:
        CHECKPOINTS["stand_crouch_g21a"] = "/nonexistent/g21a/exported"
        try:
            ckpt_obs_filter("stand_crouch_g21a"); raise AssertionError("env.yaml 없는 G21a 가 원시 관측으로 허용됨")
        except ValueError:
            pass
    finally:
        CHECKPOINTS["stand_crouch_g21a"] = _saved
    assert g21 == (OBS_ACT_EMA_BETA_DEPLOY, 0.25), f"G21a 관측 필터 {g21} ≠ (0.27, 0.25)"
    core = BridgeCore("stand_crouch_g21a")
    assert core.runner.act_obs_ema_beta == 0.27 and core.runner.jvel_obs_ema_beta == 0.25
    q = ckpt_default_pose("stand_crouch_g21a")
    for _ in range(3):
        core.tick([0.01, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], q, [0.1] * 12, [1.0, 1.0])
    assert core.runner._ema, "G21a 필터 상태가 쌓이지 않음"
    core.reset("stand_crouch_g12")
    assert core.runner.act_obs_ema_beta is None and core.runner.jvel_obs_ema_beta is None and not core.runner._ema, "전환 후 필터 상태 잔존"
    core.reset("stand_crouch_g21a")
    assert core.runner.act_obs_ema_beta == 0.27 and not core.runner._ema, "G21a 재전환 시 상태가 비어 있지 않음"
    assert obs_filter_contract({"obs_joint_vel_func": "x:joint_vel_rel_dropout", "obs_actions_func": "x:last_action"}) == (None, None)
    for bad in ({"obs_actions_func": "x:last_action_lpf_v9"}, {"obs_joint_vel_func": "x:encoder_joint_vel_lpf"}):
        try:
            obs_filter_contract(bad); raise AssertionError(f"잘못된 계약 허용: {bad}")
        except ValueError:
            pass
    return "G12·G12 ema0 필터 없음 · G21a act β0.27·jvel β0.25 · env.yaml 없는 G21a 거부 · 정책 전환 시 러너 재생성(필터·이력 비움) · 미지 함수 거부"

def _selftest_contract(required=()):
    """(0906 — 반복 감사 질문의 코드화) 배포 stand 정책의 학습 env(params/env.yaml)와 러너 상수의 계약:
    DEFAULT_POSE(관절명 기준 12개) · 액션 스케일 · 슬루 4°/틱 · 제어 주기 50Hz(decimation×dt) · use_default_offset.
    env.yaml 이 없는 세대는 SKIP(구 세대·비 stand). STAND_DEFAULT 와 required 의 stand_* 는 env.yaml 이 있어야 한다
    (배포 패키지가 models/<gen>/env.yaml 을 <run>/params/ 로 싣는다)."""
    names = [n for n in dict.fromkeys([STAND_DEFAULT] + list(required) + list(CKPT_JVEL_EMA_MODE) + ["stand_crouch_g12", "stand_crouch_g21a"]) if n in CHECKPOINTS]
    checked, skipped = [], []
    for name in names:
        yml = os.path.join(os.path.dirname(CHECKPOINTS[name].rstrip("/")), "params", "env.yaml")
        if not os.path.isfile(yml):
            skipped.append(name)
            continue
        c = _parse_env_contract(yml)
        for j, want in zip(JOINT_ORDER, ckpt_default_pose(name)):   # 0918 체크포인트별 기본자세
            got = c["joint_pos"].get(j)
            assert got is not None, f"{name}: env.yaml init_state.joint_pos 에 {j} 없음 ({yml})"
            assert abs(got - float(want)) < 1e-6, (
                f"{name}: DEFAULT_POSE[{j}] 러너 {float(want):+.5f} ≠ 학습 env {got:+.5f} rad")
        assert abs(c.get("scale", -1.0) - _DC.ACTION_SCALE) < 1e-9, f"{name}: 액션 스케일 {c.get('scale')} ≠ {_DC.ACTION_SCALE}"
        assert abs(c.get("max_delta", -1.0) - SLEW_RAD) < 1e-9, f"{name}: 슬루 {c.get('max_delta')} ≠ {SLEW_RAD}"
        hz = 1.0 / (c["dt"] * c["decimation"])
        assert abs(hz - _DC.RATE_HZ) < 1e-6, f"{name}: 제어 주기 {hz:.2f}Hz ≠ {_DC.RATE_HZ}"
        assert c.get("use_default_offset") is True, f"{name}: use_default_offset={c.get('use_default_offset')} (기대 True)"
        ema_c = _jvel_ema_contract(c.get("jvel_ema_func"))
        ema_m = ckpt_jvel_ema_mode(name)          # ema0 인데 legacy 계약이 아니면 여기서 ValueError
        fa, fv = obs_filter_contract(c)             # 1003: 관측 필터 계약(알 수 없는 함수면 ValueError)
        checked.append(f"{name}(지연 {c.get('min_delay', '?')}~{c.get('max_delay', '?')}틱, EMA 계약 {ema_c}·모드 {ema_m}, 관측필터 act={fa} jvel={fv})")
    need = [n for n in names if n in skipped and (n == STAND_DEFAULT or n.startswith("stand_"))]
    assert not need, f"stand 정책의 env.yaml 없음: {need} — 배포 패키지에 params/env.yaml 포함 필요"
    return ("DEFAULT_POSE 12관절·스케일 0.25·슬루 4°/틱·50Hz·default_offset 일치: " + ", ".join(checked)
            + (f" | SKIP {', '.join(skipped)}" if skipped else ""))


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
#: 0906 감사 B1: 값의 원본은 rl_walking/deploy/deploy_chain.py — 판정 스크립트(eval_drift_dr --deploy old)가
#: 같은 상수·같은 순서(러너 슬루 → LPF → 블렌딩)를 import 한다. 여기서 숫자를 바꾸지 말 것.
BLEND_S = _DC.BLEND_S
#: 리프트 가드 — 0.6s (4차 실기: 0.4s가 전방 낙하 중 회복 스텝의 양발
#: 순간 무접촉을 중단시킴 → 스텝 완료 여유 부여. 들림 발버둥 차단은 유지)
#: (0803 실측: 재인장으로 들리자 접촉 [0,0] → 정책 공중 발버둥.
#: 서기/제자리걸음 문맥에서 양발 무접촉 지속 = 들림·완전 공중 → 동결이 안전)
LIFT_STOP_S = 1.0   # 0910 승윤님 지시: 0.6→1.0s (A1 3회에서 두 발 이탈 후 0.6s 에 동결 — 회복 여유 부여)
#: ── 행잉 ARM 리프트 가드 상태기 (0908, 승윤님 지시 Hanging-to-Stand) ──────────────────────
#: 기존 리프트 가드(양발 무접촉 LIFT_STOP_S → 동결)는 0803 공중 발버둥 사고의 방지책이라 **삭제·완화하지 않는다**.
#: stand 명령에 hang_arm:true 를 **명시**한 경우에만, 접촉을 한 번도 얻기 전(PRE_CONTACT_HANG_ALLOWED)
#: 무접촉을 허용한다. 양발 접촉이 CONTACT_CONFIRM_S 지속되면 CONTACT_ACQUIRED — 그 뒤로는 기존 가드 그대로.
#: PRE_CONTACT 에도 상한(HANG_MAX_S)이 있다. 0913 승윤님 결정(실물 5회: 접지 뒤 재조정 때마다 동결): hang_arm 모드에서
#: CONTACT_ACQUIRED 중 양발 무접촉 LIFT_STOP_S 가 지속되면 동결 대신 PRE_CONTACT(새 HANG_MAX_S 창)로 **되돌아간다** —
#: 최대 HANG_MAX_RELIFTS 회, FREE_STAND(줄 느슨·자립 판정 뒤) 재들림은 종전대로 동결(잡힌 상태에서 정책 발버둥 방지).
#: hang_arm 이 없으면 종전과 비트 단위로 같다.
#: 중력·발진·프레임·CAN 게이트는 무관·불변. hang_arm 이 없으면 종전과 비트 단위로 같은 동작이다(셀프테스트로 고정).
HANG_MAX_S = 120.0            # 접지 없이 정책이 공중에 살아 있을 수 있는 최대 시간
HANG_MAX_RELIFTS = 5          # 0913: 접지 뒤 재들림 → 공중 허용 복귀 허용 횟수(초과 시 종전 lift_guard 동결)
CONTACT_CONFIRM_S = 0.3       # 양발 접촉이 이만큼 지속돼야 '접지 획득'
#: 0922: 체중 상수·발력 비율 제거. 감압 패드는 접지 여부만 판단한다.
FREE_STAND_CONFIRM_S = 3.0    # 양발 접지가 이만큼 연속되면 FREE_STAND (그 뒤 재들림은 동결)



class HangGuard:
    """리프트 가드 상태기 — PRE_CONTACT_HANG_ALLOWED → CONTACT_ACQUIRED → FREE_STAND.
    0913: hang_mode 에서 CONTACT_ACQUIRED 중 양발 무접촉 lift_stop_s 지속 = 재들림 → PRE_CONTACT 로 복귀(≤ max_relifts, 새 hang_max 창).
    FREE_STAND 재들림·hang_mode 아님·재들림 초과는 종전대로 동결."""

    PRE_CONTACT = "PRE_CONTACT_HANG_ALLOWED"
    CONTACT_ACQUIRED = "CONTACT_ACQUIRED"
    FREE_STAND = "FREE_STAND"

    def __init__(self, hang_mode=False, t0=None, lift_stop_s=LIFT_STOP_S, hang_max_s=HANG_MAX_S,
                 confirm_s=CONTACT_CONFIRM_S, free_confirm_s=FREE_STAND_CONFIRM_S,
                 max_relifts=HANG_MAX_RELIFTS):
        self.hang_mode = bool(hang_mode)
        self.max_relifts = int(max_relifts)
        self.relifts = 0
        self.state = self.PRE_CONTACT if self.hang_mode else self.CONTACT_ACQUIRED
        self.t0 = t0
        self.lift_stop_s = float(lift_stop_s)
        self.hang_max_s = float(hang_max_s)
        self.confirm_s = float(confirm_s)
        self.free_confirm_s = float(free_confirm_s)
        self.free_t0 = None
        self.nocontact_t0 = None
        self.contact_t0 = None
        self._transitions = []

    def _go(self, state, now):
        self.state = state
        self._transitions.append((state, now))

    def pop_transitions(self):
        out, self._transitions = self._transitions, []
        return out

    def update(self, now, contact2, foot_total_n=None):   # foot_total_n: 0922 폐지(접지 판단만 사용) — 호출부 호환용으로만 남김
        """매 틱. 반환 None = 정상, 문자열 = 즉시 동결 사유 (호출자가 STOP_ALL)."""
        both = contact2[0] > 0.0 and contact2[1] > 0.0
        none = contact2[0] == 0.0 and contact2[1] == 0.0
        if self.state == self.PRE_CONTACT:
            if self.t0 is None:
                self.t0 = now
            if both:
                if self.contact_t0 is None:
                    self.contact_t0 = now
                elif now - self.contact_t0 >= self.confirm_s:
                    self._go(self.CONTACT_ACQUIRED, now)
                    self.nocontact_t0 = None
            else:
                self.contact_t0 = None
            if self.state == self.PRE_CONTACT and now - self.t0 >= self.hang_max_s:
                return f"hang_guard: 행잉 허용 {self.hang_max_s:.0f}s 초과 — 접지 없음"
            return None
        # CONTACT_ACQUIRED / FREE_STAND — 종전 리프트 가드와 동일
        if none:
            if self.nocontact_t0 is None:
                self.nocontact_t0 = now
            elif now - self.nocontact_t0 >= self.lift_stop_s:
                if self.hang_mode and self.state == self.CONTACT_ACQUIRED and self.relifts < self.max_relifts:
                    # 0913 승윤님 결정: 접지 뒤 재들림 = 재착지 허용 (공중 허용 상태로 복귀, 새 hang_max 창)
                    self.relifts += 1
                    self.t0 = now
                    self.contact_t0 = None
                    self.nocontact_t0 = None
                    self._go(self.PRE_CONTACT, now)
                    return None
                return f"lift_guard: 양발 무접촉 {self.lift_stop_s}s 지속"
        else:
            self.nocontact_t0 = None
        # 0922 승윤님: 감압 패드는 **접지 여부 판단용**이다. 종전엔 '발력 ≥ 0.8×체중' 으로 FREE_STAND 를 판정했는데,
        #   그것은 미보정 카운트를 하중으로 읽은 것이다. → **양발 접지가 free_confirm_s 연속** 유지되면 FREE_STAND.
        #   (FREE_STAND 의 효과는 '그 뒤의 재들림은 동결' 뿐이므로 접지 기준으로도 목적이 유지된다.)
        if self.state == self.CONTACT_ACQUIRED:
            if both:
                if self.free_t0 is None:
                    self.free_t0 = now
                elif now - self.free_t0 >= self.free_confirm_s:
                    self._go(self.FREE_STAND, now)
            else:
                self.free_t0 = None
        return None


#: ── 착지 자동 인수 게이트 (1005, 단순화 기동 경로 — stand 명령에 touchdown_start:true 를 **명시**한 경우만) ──────────
#: 배경: 배포 후보(G21a·C5/KE = G21a+G24R-a)는 학습 env 에서 hang_support r0=0(지지 없음)·스폰 z 0~1 cm(지면 출발)
#: 이었다 — 공중(무접촉·몸통이 줄에 매달려 관절 동작이 중력벡터를 못 바꾸는 상태) 관측을 한 번도 본 적이 없다.
#: 실물 guard_mode=data 에서는 리프트 가드·발진 가드(과속 제외)가 정지하지 않으므로, 공중에서 정책을 돌리면 막는 층이 없다.
#: → 이 경로는 착지 전에는 **정책을 계산조차 하지 않고**(러너 이력·EMA·직전 액션·C5 감지기 미접촉) stage8 이 스트리밍하던
#:   수평 홀드 명령을 그대로 유지한다. 양발 접촉 TD_CONFIRM_S 지속 + 중력 게이트 + 짧은 정착(|ω_xy|) 뒤에
#:   종전 stand 시작과 **같은 초기화**(러너 reset = 학습 리셋 계약: 이력 백필·직전 액션 0·EMA 첫 값, 가드·LPF·블렌딩 재시작)
#:   를 그 시점에 수행한다. 이후는 hang_arm 없는 종전 지상 stand 와 비트 단위로 같은 경로(블렌딩 1.5 s 포함 — 평가된 배포 체인).
#: touchdown_start 가 없으면 이 게이트는 만들어지지도 않는다(종전 동작 불변, 셀프테스트 고정).
TD_CONFIRM_S = CONTACT_CONFIRM_S   # 양발 접촉 지속 [s] — 행잉 ARM 의 접지 확인과 같은 값
TD_QUIET_W = 0.3                   # [rad/s] |ω_xy| 정착 문턱 — C5 감지기 off_w 와 같은 값(켜기 0.5 보다 낮다: 시작 직후 오탐 방지)
TD_QUIET_S = 0.1                   # 정착 연속 시간 [s]
TD_SETTLE_MAX_S = 1.0              # 양발 확인 뒤 정착 대기 상한 — 넘으면 시작(계속 흔들리면 실제 외란 → 정책·회복이 맡는다)
TD_WAIT_MAX_S = HANG_MAX_S         # 무장 뒤 착지 대기 상한 — 초과 시 STOP_ALL(동결 = 같은 홀드 유지) + 정책 비활성
#: 홀드 자세(래치) vs 정책 기본자세 허용 편차 [deg, 관절 최대]. 넘으면 정책을 시작하지 않고 정지(동결 = 같은 홀드).
#: 근거: 인수 블렌딩이 홀드→정책 기본자세 이동 명령이 된다 — 0918 실물 낙상(힙 15° 차), 0922 발목 +3.9° 차로 전방 기울기.
#: 8° = '수평 자세 적용 누락'(무릎 0 vs 38.5°)·0918 힙 15° 는 잡고, 손 미세 조정(수 °)은 통과.
TD_POSE_TOL_DEG = 8.0


class TouchdownGate:
    """착지 자동 인수 상태기 — AIR_CHECK → WAIT_CONTACT → SETTLE → POLICY (POLICY 는 되돌아가지 않는다).

    update() 반환 (act, why): act = "hold"(홀드 유지, 정책 미구동) | "start"(이번 틱 정책 시작) | "timeout"(정지).
    AIR_CHECK: 무장 뒤 '양발 동시 접촉이 아닌' 상태를 한 번 봐야 착지 대기로 간다 — 이미 선 채로/패드 1 고착으로
    무장되면 정책이 시작되지 않는다(실제 착지 전이만 인정). 중력 게이트 불통과(>17.5° 기울기)면 시작하지 않는다."""

    AIR_CHECK = "AIR_CHECK"
    WAIT = "WAIT_CONTACT"
    SETTLE = "SETTLE"
    POLICY = "POLICY"

    def __init__(self, t0=None, confirm_s=TD_CONFIRM_S, quiet_w=TD_QUIET_W, quiet_s=TD_QUIET_S,
                 settle_max_s=TD_SETTLE_MAX_S, max_wait_s=TD_WAIT_MAX_S, require_air=True):
        self.t0 = t0
        self.confirm_s, self.quiet_w, self.quiet_s = float(confirm_s), float(quiet_w), float(quiet_s)
        self.settle_max_s, self.max_wait_s = float(settle_max_s), float(max_wait_s)
        self.state = self.AIR_CHECK if require_air else self.WAIT
        self.contact_t0 = None
        self.settle_t0 = None
        self.quiet_t0 = None
        self.why = "착지 대기"
        self.start_t = None
        self.start_kind = None       # "quiet" | "settle_cap"
        self.last_w = None
        self._transitions = []

    @property
    def started(self):
        return self.state == self.POLICY

    def _go(self, state, now):
        self.state = state
        self._transitions.append((state, now))

    def pop_transitions(self):
        out, self._transitions = self._transitions, []
        return out

    def snapshot(self, now):
        return {"state": self.state, "why": self.why,
                "waited_s": (round(now - self.t0, 2) if self.t0 is not None else None),
                "start_kind": self.start_kind,
                "w_xy": (round(self.last_w, 3) if self.last_w is not None and math.isfinite(self.last_w) else None)}

    def update(self, now, contact2, gyro, grav_ok):
        if self.state == self.POLICY:
            return "start", self.why
        if self.t0 is None:
            self.t0 = now
        both = contact2[0] > 0.0 and contact2[1] > 0.0
        try:
            w = math.hypot(float(gyro[0]), float(gyro[1]))
        except (TypeError, ValueError, IndexError):
            w = float("nan")
        self.last_w = w
        if self.state == self.AIR_CHECK:
            if not both:
                self._go(self.WAIT, now)
            else:
                self.why = "양발 접촉 상태로 무장됨 — 공중(양발 동시 접촉 아님) 확인 전에는 시작하지 않음"
        if self.state == self.WAIT:
            if both:
                if self.contact_t0 is None:
                    self.contact_t0 = now
                if now - self.contact_t0 >= self.confirm_s:
                    self._go(self.SETTLE, now)
                    self.settle_t0 = now
                    self.quiet_t0 = None
                else:
                    self.why = "양발 접촉 확인 중"
            else:
                self.contact_t0 = None
                self.why = "착지 대기 — 수평 자세 홀드(정책 미구동)"
        if self.state == self.SETTLE:
            if not both:
                self.contact_t0 = None
                self.settle_t0 = None
                self._go(self.WAIT, now)
                self.why = "접촉 상실 — 착지 대기로 복귀"
            else:
                quiet = math.isfinite(w) and w < self.quiet_w
                self.quiet_t0 = (self.quiet_t0 if self.quiet_t0 is not None else now) if quiet else None
                quiet_ok = self.quiet_t0 is not None and now - self.quiet_t0 >= self.quiet_s - 1e-9
                capped = now - self.settle_t0 >= self.settle_max_s - 1e-9
                if grav_ok and (quiet_ok or capped):
                    self.start_kind = "quiet" if quiet_ok else "settle_cap"
                    self.start_t = now
                    self.why = f"착지 확인 — 정책 시작({self.start_kind})"
                    self._go(self.POLICY, now)
                    return "start", self.why
                self.why = ("중력 게이트 불통과(기울기 과대) — 홀드 유지" if not grav_ok
                            else f"착지 정착 대기 |ω_xy| {w:.2f} rad/s")
        if now - self.t0 >= self.max_wait_s:
            return "timeout", (f"touchdown_wait_timeout: 착지 자동 인수 대기 {self.max_wait_s:.0f}s 초과 "
                               f"({self.state}) — 정책 미시작")
        return "hold", self.why


#: 온셋 블렌딩 가중치 — 스무스스텝 0→1 (양 끝 기울기 0, 단조증가). dur_s<=0 이면 즉시 1.0.
#: 구현은 deploy_chain.blend_alpha (판정 스크립트와 동일 함수).
blend_alpha = _DC.blend_alpha


#: stand 계열 타깃 저역필터 (0803 발진 분석: 지연 80~100ms 루프에서 5~9Hz
#: 정책 주도 리밋사이클 — 발진 대역 루프게인 컷. 서기 보정은 <2Hz라 손실 미미.
#: EMA 1차: beta=0.27 ≈ 차단 2.5Hz@50Hz. march/walk에는 미적용 (보행 리듬 보존).
STAND_LPF_BETA = _DC.STAND_LPF_BETA     # 원본: deploy_chain.py (0906 감사 B1)
lpf_step = _DC.lpf_step                  # EMA 1차 한 스텝 — state None 이면 x 로 초기화 (점프 없음)

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


def _selftest_hang_guard():
    """HangGuard: (1) hang_arm 없음 = 종전 리프트 가드와 동일, (2) 행잉 ARM 은 접지 전 무접촉 허용,
    접지 0.3s 후 기존 가드 복원, 접지 뒤 재들림은 동결, 상한 초과 동결, 상태 역행 없음."""
    def run(g, seq, dt=0.02):
        """seq: [(지속초, contact2, foot_total)], 반환: (정지사유 or None, 정지 시각)."""
        t = 0.0
        for dur, c2, ft in seq:
            n = int(round(dur / dt))
            for _ in range(n):
                t += dt
                why = g.update(t, c2, ft)
                if why:
                    return why, t
        return None, t
    B, NONE, L = [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]
    # (1) 종전 동작: 0.5s 무접촉은 통과, 0.6s 는 동결, 한 발이라도 닿으면 타이머 리셋
    g = HangGuard(hang_mode=False)
    assert g.state == HangGuard.CONTACT_ACQUIRED
    assert run(g, [(1.0, B, 100.0), (LIFT_STOP_S - 0.1, NONE, 0.0), (0.5, L, 40.0), (LIFT_STOP_S - 0.1, NONE, 0.0)])[0] is None
    why, t = run(HangGuard(hang_mode=False), [(0.2, B, 100.0), (LIFT_STOP_S + 0.1, NONE, 0.0)])
    assert why and why.startswith("lift_guard") and abs(t - (0.2 + LIFT_STOP_S)) < 0.03, (why, t)
    # (2) 행잉 ARM: 10s 무접촉도 허용, 한 발 접촉은 접지 아님, 양발 0.3s 로 접지 획득
    g = HangGuard(hang_mode=True, t0=0.0)
    assert g.state == HangGuard.PRE_CONTACT
    assert run(g, [(10.0, NONE, 0.0), (1.0, L, 30.0), (0.2, B, 60.0), (0.5, NONE, 0.0)])[0] is None
    assert g.state == HangGuard.PRE_CONTACT, "0.2s 접촉은 확인 시간 미달"
    assert run(g, [(0.4, B, 60.0)])[0] is None and g.state == HangGuard.CONTACT_ACQUIRED
    assert [s for s, _ in g.pop_transitions()] == [HangGuard.CONTACT_ACQUIRED]
    # (0913) 접지 뒤 재들림(hang_arm) = 동결 대신 PRE_CONTACT 복귀, 재착지하면 다시 CONTACT_ACQUIRED
    why, _ = run(g, [(0.3, B, 120.0), (LIFT_STOP_S + 0.1, NONE, 0.0)])
    assert why is None and g.state == HangGuard.PRE_CONTACT and g.relifts == 1, (why, g.state, g.relifts)
    assert [s for s, _ in g.pop_transitions()] == [HangGuard.PRE_CONTACT]
    assert run(g, [(30.0, NONE, 0.0), (0.4, B, 60.0)])[0] is None and g.state == HangGuard.CONTACT_ACQUIRED, "재착지 30 s 뒤 재접지"
    # 재들림 허용 횟수 초과 → 종전 lift_guard 동결
    for _ in range(HANG_MAX_RELIFTS - 1):
        assert run(g, [(LIFT_STOP_S + 0.1, NONE, 0.0), (0.4, B, 60.0)])[0] is None
    assert g.relifts == HANG_MAX_RELIFTS
    why, _ = run(g, [(LIFT_STOP_S + 0.1, NONE, 0.0)])
    assert why and why.startswith("lift_guard"), why
    # (0913 추가 검증) 재들림 뒤 시간 제한: 새 창에서 접지 없이 hang_max_s 초과 → hang_guard 동결 (재들림 시각 기준)
    g4 = HangGuard(hang_mode=True, t0=0.0, hang_max_s=5.0)   # run() 은 호출마다 시각 0 부터 → 한 호출로 잇는다
    t_relift = 0.4 + LIFT_STOP_S          # 재들림 판정 시각 = 접지 0.4 s + 무접촉 lift_stop_s
    why, t = run(g4, [(0.4, B, 60.0), (LIFT_STOP_S + 0.1, NONE, 0.0), (6.0, NONE, 0.0)])
    assert why and why.startswith("hang_guard") and abs(t - (t_relift + 5.0)) < 0.06 and g4.relifts == 1, (why, t, t_relift, g4.relifts)
    # 접촉 오판 1: 짧은 누락/채터(양발 0 이 lift_stop_s 미만) → 상태·횟수 불변
    g5 = HangGuard(hang_mode=True, t0=0.0)
    run(g5, [(0.4, B, 60.0)]); assert g5.state == HangGuard.CONTACT_ACQUIRED; g5.pop_transitions()   # 착지 전이는 비우고 이후 '전이 없음' 검사
    for _ in range(20):
        assert run(g5, [(0.3, NONE, 0.0), (0.3, B, 60.0)])[0] is None
    assert g5.state == HangGuard.CONTACT_ACQUIRED and g5.relifts == 0 and g5.pop_transitions() == []
    # 접촉 오판 2: 한 발 비트 고착 0(정책은 [0,1]) → 양발 무접촉이 아니므로 재들림도 동결도 없음(종전 동일)
    assert run(g5, [(30.0, L, 60.0)])[0] is None and g5.state == HangGuard.CONTACT_ACQUIRED and g5.relifts == 0
    # 접촉 오판 3: 한 발 비트 고착 1(공중에서도 [1,0]) → 실제 들려도 재들림 판정 불가 = 종전과 같은 사각(문서화)
    assert run(g5, [(30.0, L, 0.0)])[0] is None and g5.state == HangGuard.CONTACT_ACQUIRED
    # hang_arm 아님: 재들림 동결 그대로 (케이스 1 과 동일, 회귀 고정)
    g2 = HangGuard(hang_mode=False)
    why, _ = run(g2, [(0.3, B, 120.0), (LIFT_STOP_S + 0.1, NONE, 0.0)])
    assert why and why.startswith("lift_guard") and g2.relifts == 0, why
    # FREE_STAND 뒤 재들림 = 동결 (잡힌 상태에서 정책 발버둥 방지)
    g3 = HangGuard(hang_mode=True, t0=0.0)
    run(g3, [(0.4, B, None), (3.2, B, None)])          # 0922: 양발 접지 3 s 지속 → FREE_STAND (발력 기준 제거)
    assert g3.state == HangGuard.FREE_STAND
    why, _ = run(g3, [(LIFT_STOP_S + 0.1, NONE, 0.0)])
    assert why and why.startswith("lift_guard") and g3.state == HangGuard.FREE_STAND, why
    # FREE_STAND 표시: 양발 접지 3 s 지속 (0922: 발력 기준 폐지 — 감압은 접지 판단용)
    g = HangGuard(hang_mode=True, t0=0.0)
    run(g, [(0.4, B, None), (3.2, B, None)])
    assert g.state == HangGuard.FREE_STAND
    # 상한: 접지 없이 HANG_MAX_S 초과 → 동결
    g = HangGuard(hang_mode=True, t0=0.0, hang_max_s=5.0)
    why, t = run(g, [(6.0, NONE, 0.0)])
    assert why and why.startswith("hang_guard") and abs(t - 5.0) < 0.03, (why, t)
    return (f"hang_arm 없음=종전 동일({LIFT_STOP_S}s 동결) · 행잉 ARM 10s 무접촉 허용 · 접지 {CONTACT_CONFIRM_S}s 확인 · "
            f"재들림 → 공중 허용 복귀(≤{HANG_MAX_RELIFTS}회, 초과 동결) · 재들림 창 상한 재적용 · 채터/한발 고착0·1 불변 · hang_arm 아님/FREE_STAND 재들림 동결 · 상한 {HANG_MAX_S:.0f}s 동결 — 12케이스 통과")


def _selftest_touchdown_gate():
    """TouchdownGate(1005): 공중 확인 → 양발 0.3 s → 정착 → 시작, 채터·접촉 상실 복귀, 중력 불통과 보류, 정착 상한, 대기 상한, 선 채 무장 거부."""
    dt = 0.02
    B, NONE, L = [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]
    Q, W = (0.0, 0.0, 0.0), (0.0, 0.8, 0.0)

    def run(g, seq, t=0.0, grav=True):
        """seq: [(지속초, contact2, gyro)] → (act, why, t) — 'start'/'timeout' 에서 멈춘다."""
        act, why = "hold", ""
        for dur, c2, gy in seq:
            for _ in range(int(round(dur / dt))):
                t += dt
                act, why = g.update(t, c2, gy, grav)
                if act != "hold":
                    return act, why, t
        return act, why, t
    # (1) 정상: 공중 2 s → 착지(조용) — 양발 0.3 s + 정착 0.1 s 뒤 시작, 그 전엔 hold
    g = TouchdownGate(t0=0.0)
    act, _, t = run(g, [(2.0, NONE, Q), (1.0, B, Q)])
    assert act == "start" and g.started and g.start_kind == "quiet" and abs(t - (2.0 + TD_CONFIRM_S + TD_QUIET_S)) < 0.05, (act, t)
    assert [s for s, _ in g.pop_transitions()] == [TouchdownGate.WAIT, TouchdownGate.SETTLE, TouchdownGate.POLICY]
    # (2) 착지 채터(0.2 s 접촉/0.1 s 이탈 반복) → 시작 없음, 이후 연속 접촉이면 시작
    g = TouchdownGate(t0=0.0)
    act, _, t = run(g, [(0.5, NONE, Q)] + [(0.2, B, Q), (0.1, L, Q)] * 10)
    assert act == "hold" and not g.started
    act, _, _ = run(g, [(1.0, B, Q)], t)
    assert act == "start"
    # (3) 착지 충격(|ω| 0.8) 지속 → 정착 상한 1.0 s 에서 시작(settle_cap), 그 전엔 hold
    g = TouchdownGate(t0=0.0)
    act, _, t = run(g, [(0.5, NONE, Q), (3.0, B, W)])
    assert act == "start" and g.start_kind == "settle_cap" and abs(t - (0.5 + TD_CONFIRM_S + TD_SETTLE_MAX_S)) < 0.05, (t, g.start_kind)
    # (4) 중력 게이트 불통과 → 양발·정숙이어도 시작 안 함
    g = TouchdownGate(t0=0.0)
    act, why, _ = run(g, [(0.5, NONE, Q), (5.0, B, Q)], grav=False)
    assert act == "hold" and not g.started and "중력" in why, why
    # (5) 선 채로 무장(양발 접촉 고착) → 공중 확인 전에는 시작하지 않음 → 상한에서 timeout
    g = TouchdownGate(t0=0.0, max_wait_s=5.0)
    act, why, t = run(g, [(6.0, B, Q)])
    assert act == "timeout" and abs(t - 5.0) < 0.03 and g.state == TouchdownGate.AIR_CHECK, (act, t, g.state)
    # (6) 공중 유지 → 대기 상한 timeout
    g = TouchdownGate(t0=0.0, max_wait_s=3.0)
    act, why, t = run(g, [(4.0, NONE, Q)])
    assert act == "timeout" and why.startswith("touchdown_wait_timeout") and abs(t - 3.0) < 0.03
    # (7) 정착 중 접촉 상실 → WAIT 복귀(확인 시간 재시작), 한 발만 접촉은 시작 아님
    g = TouchdownGate(t0=0.0)
    act, _, t = run(g, [(0.5, NONE, Q), (TD_CONFIRM_S + 0.04, B, W), (0.1, NONE, Q), (2.0, L, Q)])
    assert act == "hold" and g.state == TouchdownGate.WAIT, g.state
    # (8) 시작 뒤 상태 고정(되돌아가지 않음)
    g = TouchdownGate(t0=0.0); run(g, [(0.2, NONE, Q), (1.0, B, Q)])
    assert g.started and g.update(10.0, NONE, Q, False)[0] == "start" and g.started
    return (f"공중 확인→양발 {TD_CONFIRM_S}s→정착 {TD_QUIET_S}s 시작 · 채터 무시 · 충격 지속 시 상한 {TD_SETTLE_MAX_S}s · "
            f"중력 불통과 보류 · 선 채 무장 거부 · 대기 상한 timeout · 접촉 상실 복귀 · 단방향 — 8케이스 통과")


def _selftest_hturn2():
    """1006 HTURN2 실험 플래그: ① 꺼짐 = 레지스트리·HeadingHold(1.0/0.6)·LPF 분기·러너 계약이 종전 그대로(키 없음·피드포워드 없음)
    ② 켜짐 = 키 등록 계약(경로·f0·기본자세·env.yaml 필수), HeadingHold kp 1·|wz|≤0.3·목표 = 시작 요 + 목표,
    walk 명령 검증(앞 보행 한정), cadence 클록·hip_r 피드포워드가 독립 float64 식과 일치, 슬루가 피드포워드 뒤에 걸림
    ③ 되돌림 = 꺼짐 상태 복원. 시뮬 원본 함수와의 비트 동치는 PC 스크립트 rl_walking/deploy/selftest_hturn2_equiv.py(torch)."""
    import numpy as np
    was_on = EXPERIMENTAL_HTURN2
    k = _H2.HTURN2_CKPT_KEY
    cfg = _H2.HTURN2_DEFAULT
    if was_on:
        disable_experimental_hturn2()
    try:
        # ① 꺼짐
        regs = (CHECKPOINTS, CKPT_GAIT_FREQ, CKPT_DEFAULT_POSE, CKPT_REQUIRE_ENV_YAML, HTURN2_CKPTS)
        assert not EXPERIMENTAL_HTURN2 and not HTURN2_CKPTS and all(k not in r for r in regs), "꺼짐인데 HTURN2 키가 등록돼 있음"
        snap = [sorted(r) for r in regs]
        stub = lambda obs: np.zeros(12, np.float32)  # noqa: E731
        core = BridgeCore("walk", runner=PolicyRunnerStage4(model_path=stub))
        assert not core.hturn2 and (core.hh.kp, core.hh.wz_limit) == (1.0, 0.6) and core.runner.hturn2 is None
        ref_hh = HeadingHold(kp=1.0, wz_limit=0.6)
        core.reset(); core.cmd = (0.2, 0.0, 0.0)
        q = np.array(DEFAULT_POSE_RAD, dtype=np.float32)
        for i in range(60):
            yaw = 0.3 + 0.02 * i
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            core.tick((0.0, 0.0, 0.0), quat, q, np.zeros(12), [1.0, 1.0])
            if i == 0:
                ref_hh.set_ref(yaw_from_quat_wxyz(*quat))
            want = ref_hh.update(yaw_from_quat_wxyz(*quat))
            got = float(core.runner.last_obs[30 + 4 * 3 + 2])           # velocity_commands 최신 프레임 wz
            assert got == np.float32(want), f"꺼짐 HeadingHold wz {got} ≠ 종전 {want} (틱 {i})"
            assert core.runner.last_ff is None
        try:
            core.set_heading_target(0.5)
            raise AssertionError("꺼짐인데 목표 각 입력이 받아들여짐")
        except ValueError:
            pass
        # ② 켜짐
        enable_experimental_hturn2()
        assert EXPERIMENTAL_HTURN2 and k in CHECKPOINTS and CHECKPOINTS[k].endswith(os.path.join(_H2.HTURN2_RUN_REL, "exported"))
        assert CKPT_GAIT_FREQ[k] == 1.1 and k in CKPT_REQUIRE_ENV_YAML and not k.startswith(("stand", "march"))
        assert CKPT_DEFAULT_POSE[k] == [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER]
        r = PolicyRunnerStage4(model_path=lambda obs: np.full(12, 3.0, np.float32), obs_dim=320, gait_freq=1.1,
                               default_pose_rad=CKPT_DEFAULT_POSE[k], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25, hturn2=cfg)
        core = BridgeCore(k, runner=r)
        assert core.hturn2 and (core.hh.kp, core.hh.wz_limit) == (cfg.hh_kp, 0.3)
        # walk 명령 검증
        for bad in ({}, {"vx": 0.2}, {"vx": -0.05}, {"vx": 0.1, "vy": 0.05}, {"vx": 0.1, "wz": 0.1},
                    {"vx": 0.1, "heading_hold": False}, {"vx": 0.1, "heading_target_deg": 200}, {"vx": float("nan")}):
            try:
                hturn2_validate_walk(bad)
                raise AssertionError(f"거부돼야 할 walk 명령 통과: {bad}")
            except ValueError:
                pass
        cmd, tgt = hturn2_validate_walk({"vx": 0.1, "heading_target_deg": -90})
        assert cmd == (0.1, 0.0, 0.0) and abs(tgt + math.pi / 2) < 1e-12
        # 시나리오: 서기 40틱 → vx 0.1 목표 +45° → 문턱 0.029/0.03 → vx 0.15
        core.reset(); core.heading_hold = True; core.set_heading_target(math.radians(45.0))
        q = np.array(CKPT_DEFAULT_POSE[k], dtype=np.float32)
        yaw0, u64, n_nz, worst_u, worst_ff = 0.7, 0.0, 0, 0.0, 0.0
        prev_out = None
        for n in range(400):
            vx = 0.0 if n < 40 else (0.1 if n < 200 else (0.029 if n < 240 else (0.03 if n < 280 else 0.15)))
            core.cmd = (vx, 0.0, 0.0)
            yaw = yaw0 + 0.004 * max(0, n - 40)
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            out = core.tick((0.0, 0.0, 0.0), quat, q, np.zeros(12), [1.0, 1.0])
            if n == 0:
                assert abs(core.hh.ref - (yaw_from_quat_wxyz(*quat) + math.radians(45.0))) < 1e-12, "목표 = 시작 요 + 45° 아님"
            wz = float(core.runner.last_obs[30 + 4 * 3 + 2])
            assert abs(wz) <= 0.3 + 1e-7, f"|wz| {wz} > 0.3"
            # 독립 float64 기준: cadence·위상
            f64 = (min(max(0.7 + 4.0 * abs(vx), 0.8), 1.6) if abs(vx) >= 0.03 else 1.1)
            u64 = 0.0 if n <= 1 else (u64 + f64 * 0.02) % 1.0
            u32 = float(core.runner._h2_clock.u)
            du = min(abs(u32 - u64), 1.0 - abs(u32 - u64))
            worst_u = max(worst_u, du)
            assert du < 1e-4, f"틱 {n} 위상 {u32} vs 기준 {u64}"
            clk = core.runner.last_obs[225 + 4 * 2: 225 + 5 * 2]
            assert abs(clk[0] - math.sin(2 * math.pi * u32)) < 1e-6 and abs(clk[1] - math.cos(2 * math.pi * u32)) < 1e-6
            # 독립 float64 기준: 피드포워드(러너 위상 u32 사용)
            ul = (u32 + float(core.runner._h2_clock.f) * 0.12) % 1.0
            v = (2.0 * ul) % 1.0
            ss = v * v * (3.0 - 2.0 * v)
            shl, shr = (ss, 1.0 - ss) if ul < 0.5 else (1.0 - ss, ss)
            amp = 1.5 * -0.455 * wz * (1.0 if abs(vx) >= 0.03 else 0.0)
            fl, fr = core.runner.last_ff
            e = max(abs(fl - amp * shl), abs(fr - amp * shr))
            worst_ff = max(worst_ff, e)
            assert e < 1e-6, f"틱 {n} 피드포워드 {core.runner.last_ff} vs 기준 {(amp * shl, amp * shr)}"
            n_nz += int(fl != 0.0 or fr != 0.0)
            if abs(vx) < 0.03:
                assert fl == 0.0 and fr == 0.0, "서기 모드에서 피드포워드"
            # 슬루는 피드포워드 뒤(러너 슬루 앞에 더함): 방출 변화 ≤ 4°/틱
            if prev_out is not None:
                assert float(np.abs(out - prev_out).max()) <= SLEW_RAD * 1.001 + 1e-6, "피드포워드가 슬루를 우회"
            prev_out = out
            q = out
        assert n_nz > 150, f"0 아닌 피드포워드 틱 {n_nz} — 시나리오 점검"
        core.set_heading_target(math.radians(-30.0))
        assert abs(core.hh.ref - (core.hh_yaw0 + math.radians(-30.0))) < 1e-12, "turn: 목표 = 시작 요 − 30° 아님"
        # ③ 되돌림
        disable_experimental_hturn2()
        assert [sorted(r) for r in regs] == snap and not EXPERIMENTAL_HTURN2, "되돌림 뒤 레지스트리가 꺼짐 상태와 다름"
    finally:
        if was_on and not EXPERIMENTAL_HTURN2:
            enable_experimental_hturn2()
        elif not was_on and EXPERIMENTAL_HTURN2:
            disable_experimental_hturn2()
    return (f"꺼짐: 키 0·HeadingHold 1.0/0.6 종전 wz 60틱 일치·피드포워드 없음 / 켜짐: 키 계약(exported·f0 1.1·CROUCH_V5·env.yaml 필수)·"
            f"|wz|≤0.3·목표=시작요+목표·walk 검증 8종 거부·400틱 위상 오차 {worst_u:.1e}·피드포워드 오차 {worst_ff:.1e}(0 아님 {n_nz})·"
            f"슬루 뒤 4°/틱 / 되돌림 복원" + (" (기동 플래그 켜짐 상태 복원)" if was_on else ""))


def _selftest_w3():
    """1006 W3 실험 플래그: ① 꺼짐 = 레지스트리에 키 없음·러너 w3 None·gait_enable 입력 거부 ② 켜짐 = 키 계약(stand 가족·exported·
    CROUCH_V5·env.yaml 필수·학습 env.yaml 관측 필터/기본자세 일치), step 명령 파싱·선행조건, 래치(0→1 다음 틱·1→0 반주기 경계)·
    클록 = [sin,cos]×ge_obs 가 독립 float64 식과 일치, 명령 관측 0, 리셋 = 서기 ③ 되돌림. 시뮬 원본과의 비트 동치는
    PC 스크립트 rl_walking/deploy/selftest_w3_equiv.py(torch)."""
    import numpy as np
    was_on = EXPERIMENTAL_W3
    k = _W3.W3_CKPT_KEY
    if was_on:
        disable_experimental_w3()
    try:
        regs = (CHECKPOINTS, CKPT_GAIT_FREQ, CKPT_DEFAULT_POSE, CKPT_REQUIRE_ENV_YAML, W3_CKPTS)
        assert not EXPERIMENTAL_W3 and not W3_CKPTS and all(k not in r for r in regs), "꺼짐인데 W3 키가 등록돼 있음"
        snap = [sorted(r) for r in regs]
        stub = lambda obs: np.zeros(12, np.float32)  # noqa: E731
        core = BridgeCore("stand_crouch_g21a", runner=PolicyRunnerStage4(model_path=stub, obs_dim=320))
        assert not core.w3 and core.runner.w3 is None and core.runner._w3_clock is None
        try:
            core.set_gait_enable(True)
            raise AssertionError("꺼짐인데 gait_enable 입력이 받아들여짐")
        except ValueError:
            pass
        # ② 켜짐
        enable_experimental_w3()
        assert EXPERIMENTAL_W3 and CHECKPOINTS[k].endswith(os.path.join(_W3.W3_RUN_REL, "exported"))
        assert k.startswith("stand_") and k in CKPT_REQUIRE_ENV_YAML and CKPT_GAIT_FREQ[k] == 1.1
        assert CKPT_DEFAULT_POSE[k] == [float(_CROUCH_V5_POSE[j]) for j in JOINT_ORDER]
        yml = _ckpt_env_yaml(k)
        env_note = "env.yaml 없음(SKIP)"
        if os.path.isfile(yml):
            c = _parse_env_contract(yml)
            for j, want in zip(JOINT_ORDER, CKPT_DEFAULT_POSE[k]):
                assert abs(c["joint_pos"][j] - want) < 1e-6, f"W3 env.yaml 기본자세 {j} 불일치"
            assert ckpt_obs_filter(k) == (0.27, 0.25), f"W3 관측 필터 {ckpt_obs_filter(k)} ≠ (0.27, 0.25)"
            env_note = "env.yaml 기본자세 CROUCH_V5·관측 필터 (0.27, 0.25) 일치"
        for bad in ({}, {"enable": "yes"}, {"enable": 2}):
            try:
                w3_parse_step(bad)
                raise AssertionError(f"거부돼야 할 step 명령 통과: {bad}")
            except ValueError:
                pass
        assert w3_parse_step({"enable": True}) is True and w3_parse_step({"enable": 0}) is False
        assert not _W3.step_enable_allowed(False, True, True)[0] and not _W3.step_enable_allowed(True, False, True)[0]
        assert not _W3.step_enable_allowed(True, True, False)[0] and _W3.step_enable_allowed(True, True, True)[0]
        r = PolicyRunnerStage4(model_path=lambda obs: np.full(12, 0.5, np.float32), obs_dim=320, gait_freq=1.1,
                               default_pose_rad=CKPT_DEFAULT_POSE[k], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25, w3=W3_CKPTS[k])
        core = BridgeCore(k, runner=r)
        assert core.w3 and not core.hturn2
        core.reset(); core.heading_hold = False; core.cmd = (0.0, 0.0, 0.0)   # stand 가족 경로와 같은 설정
        q = np.array(CKPT_DEFAULT_POSE[k], dtype=np.float32)
        want_at = lambda n: (30 <= n < 77) or (110 <= n < 111) or (150 <= n < 240)   # noqa: E731 — 반주기 중간·1틱 펄스
        ge, u, go, flips, worst = False, 0.0, 0.0, [], 0.0
        for n in range(320):
            if n == 280:                       # 중간 리셋(정책 재시작) — 서기·위상 0 으로
                core.reset(); core.heading_hold = False; ge, u, go = False, 0.0, 0.0
                n0 = n
            core.set_gait_enable(want_at(n))
            w = core.runner._w3_clock.want
            quat = (1.0, 0.0, 0.0, 0.0)
            core.tick((0.0, 0.0, 0.0), quat, q, np.zeros(12), [1.0, 1.0])
            t = n - (n0 if n >= 280 else 0)
            fresh = t <= 1
            un = u + 1.1 * 0.02
            cross = (u < 0.5 and un >= 0.5) or un >= 1.0
            on = w and not ge
            g2 = w if fresh else (True if on else (False if ((not w) and ge and cross) else ge))
            if g2 != ge:
                flips.append((n, int(g2)))
            u = 0.0 if (fresh or on or not g2) else un % 1.0
            go = (1.0 if g2 else 0.0) if fresh else min(max(go + (0.1 if g2 else -0.1), 0.0), 1.0)
            ge = g2
            obs = core.runner.last_obs
            clk = obs[225 + 4 * 2: 225 + 5 * 2]
            e = max(abs(clk[0] - math.sin(2 * math.pi * u) * go), abs(clk[1] - math.cos(2 * math.pi * u) * go))
            worst = max(worst, e)
            assert core.runner._w3_clock.ge == ge and e < 1e-5, f"틱 {n} ge/클록 불일치 (ge {core.runner._w3_clock.ge} vs {ge}, 오차 {e})"
            assert np.all(obs[30:45] == 0.0), "W3 Stage 0 명령 관측이 0 이 아님(heading_hold?)"
        assert len(flips) >= 5, flips
        # ③ 되돌림
        disable_experimental_w3()
        assert [sorted(r) for r in regs] == snap and not EXPERIMENTAL_W3, "되돌림 뒤 레지스트리가 꺼짐 상태와 다름"
    finally:
        if was_on and not EXPERIMENTAL_W3:
            enable_experimental_w3()
        elif not was_on and EXPERIMENTAL_W3:
            disable_experimental_w3()
    return (f"꺼짐: 키 0·러너 w3 None·gait_enable 거부 / 켜짐: stand 가족 키·exported·CROUCH_V5·{env_note}·step 파싱 3종 거부·"
            f"선행조건(정책·워밍업·블렌딩)·래치 전이 {flips}·클록 오차 {worst:.1e}·명령 관측 0·리셋=서기 / 되돌림 복원"
            + (" (기동 플래그 켜짐 상태 복원)" if was_on else ""))


def _w3_sup_closed_loop(trip_tick, *, contact_ok=True, cfg=None, n_ticks=400, enable_tick=60):
    """W3 감독기 + 실제 BridgeCore(W3 러너, 스텁 정책) 폐루프 — 노드 _tick 순서(observe → 지연 동결 확인 → 가드 → core.tick)를 흉내.
    접촉 모형(가정): ge 1 이면 u<0.5 = 왼발 스윙(접촉 L0 R1), u≥0.5 = 오른발 스윙(L1 R0); ge 0 이 된 뒤 5틱(0.1 s) 착지 지연 후 양발.
    contact_ok=False 면 ge 0 뒤에도 오른발 접촉이 안 들어온다(패드 고장 모형). 반환 dict(트립·래치 해제·동결 틱, 동결 종류, 동결 틱 접촉)."""
    import numpy as np
    k = _W3.W3_CKPT_KEY
    r = PolicyRunnerStage4(model_path=lambda obs: np.zeros(12, np.float32), obs_dim=320, gait_freq=1.1,
                           default_pose_rad=CKPT_DEFAULT_POSE[k], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25, w3=W3_CKPTS[k])
    core = BridgeCore(k, runner=r)
    core.reset(); core.heading_hold = False; core.cmd = (0.0, 0.0, 0.0)
    sup = _W3.W3Supervisor(cfg or _W3.W3_SAFETY_DEFAULT)
    q = np.array(CKPT_DEFAULT_POSE[k], dtype=np.float32)
    clk = core.runner._w3_clock
    t0 = 1000.0
    off_t = None
    res = {"trip": None, "latch_off": None, "freeze": None, "kind": None, "contact_at_freeze": None,
           "enable": None, "single_support_at_trip": None}
    seq = 0
    for n in range(n_ticks):
        now = t0 + n * TICK_SEC
        if clk.ge:
            u = float(clk.u)
            contact = [1.0, 0.0] if u >= 0.5 else [0.0, 1.0]
            gyro = (0.6, 0.3, 0.0)
            off_t = None
        else:
            off_t = now if off_t is None else off_t
            landed = now - off_t >= 5 * TICK_SEC - 1e-9 or res["enable"] is None
            contact = [1.0, 1.0 if (landed and (contact_ok or res["trip"] is None)) else 0.0]
            gyro = (0.05, 0.02, 0.0) if landed else (0.5, 0.2, 0.0)
        sup.observe(now, contact, gyro, (0.0, 0.0, -1.0))
        due = sup.freeze_due(now, bool(clk.ge))
        if due is not None:
            res["freeze"], res["kind"], res["contact_at_freeze"] = n, due[0], list(contact)
            sup.on_frozen(due[1])
            break
        if n == enable_tick:
            seq += 1
            d = sup.handle({"cmd": "step", "enable": True, "req_id": "t", "seq": seq}, now, active=True, warmed=True,
                           blend_done=True, ge_now=bool(clk.ge), want_now=bool(clk.want))
            assert d["accepted"] and d["action"] == "set_want", d
            core.set_gait_enable(True)
            res["enable"] = n
        if n == trip_tick:
            res["trip"] = n
            res["single_support_at_trip"] = contact.count(1.0) == 1
            if sup.begin_freeze(now, "lift_guard: 시험"):
                core.set_gait_enable(False)
            due = sup.freeze_due(now, bool(clk.ge))
            if due is not None:
                res["freeze"], res["kind"], res["contact_at_freeze"] = n, due[0], list(contact)
                sup.on_frozen(due[1])
                break
        was_ge = bool(clk.ge)
        core.tick((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), q, np.zeros(12), contact)
        if was_ge and not clk.ge and res["trip"] is not None and res["latch_off"] is None:
            res["latch_off"] = n
    return res


def _selftest_w3_supervisor():
    """1006 감사 H3–H6·L1: W3 세션 감독기 — 명령 상태기(req_id·seq·정지 래치·rearm·응답 결정), 실행 중 재발사·교체 거부,
    켜기 선행조건(양발 접촉·기울기·정착), 지연 동결 순서(래치 해제 → 양발 지지 → 동결, 타임아웃)를 순수 객체 + BridgeCore 폐루프로 고정.
    플래그 꺼짐 기동에서는 노드가 감독기를 만들지 않는다(w3sup None — 종전 경로)."""
    was_on = EXPERIMENTAL_W3
    if not was_on:
        enable_experimental_w3()
    try:
        S = _W3.W3Supervisor()
        ok_ctx = dict(active=True, warmed=True, blend_done=True, ge_now=False, want_now=False)

        def settle(sup, t_end, dur=1.2, contact=(1.0, 1.0), gyro=(0.02, 0.01, 0.0), grav=(0.0, 0.0, -1.0)):
            t = t_end - dur
            while t <= t_end + 1e-9:
                sup.observe(t, contact, gyro, grav)
                t += TICK_SEC

        def step(seq, en=True, rid="op"):
            return {"cmd": "step", "enable": en, "req_id": rid, "seq": seq}

        # (1) 형식: req_id·seq 필수
        settle(S, 10.0)
        for bad in ({"cmd": "step", "enable": True}, {"cmd": "step", "enable": True, "req_id": "a"},
                    {"cmd": "step", "enable": True, "req_id": "a", "seq": True}, {"cmd": "step", "enable": True, "req_id": "", "seq": 1},
                    {"cmd": "step", "enable": True, "req_id": "a", "seq": 0}, {"cmd": "step", "enable": "yes", "req_id": "a", "seq": 1}):
            assert not S.handle(bad, 10.0, **ok_ctx)["accepted"], f"형식 불량 수락: {bad}"
        # (2) 정상 켜기 → 중복 seq·오래된 seq 거부
        d = S.handle(step(5), 10.0, **ok_ctx)
        assert d["accepted"] and d["action"] == "set_want" and d["want"] is True, d
        dup = S.handle(step(5), 10.02, **dict(ok_ctx, want_now=False))
        old = S.handle(step(4), 10.04, **ok_ctx)
        assert not dup["accepted"] and "중복" in dup["why"] and not old["accepted"], (dup, old)
        # (3) 늦은 enable(H3): stop 뒤 0.3 s 에 새 seq 켜기 → 정지 래치로 거부
        assert S.handle({"cmd": "stop"}, 11.0, **dict(ok_ctx, ge_now=True, want_now=True))["action"] == "stop"
        settle(S, 11.3)
        late = S.handle(step(6), 11.3, **ok_ctx)
        assert not late["accepted"] and "정지 래치" in late["why"], late
        # (4) 1 Hz 반복 enable(같은 메시지 seq 7) 중 stop → 전부 거부, rearm 뒤에도 반복 메시지는 거부, 새 seq 만 수락
        S2 = _W3.W3Supervisor()
        settle(S2, 20.0)
        rep = step(7, rid="pub")
        acc = [S2.handle(rep, 20.0, **ok_ctx)["accepted"]]
        assert S2.handle({"cmd": "stop", "req_id": "stop", "seq": 8}, 20.5, **dict(ok_ctx, ge_now=True, want_now=True))["accepted"]
        for i in range(1, 6):
            settle(S2, 20.0 + i)
            acc.append(S2.handle(rep, 20.0 + i, **ok_ctx)["accepted"])
        assert acc == [True, False, False, False, False, False], acc
        assert not S2.handle({"cmd": "rearm", "req_id": "r", "seq": 8}, 26.0, **ok_ctx)["accepted"], "중복 seq rearm 수락"
        assert not S2.handle({"cmd": "rearm", "req_id": "r", "seq": 9}, 26.0, **dict(ok_ctx, ge_now=True))["accepted"], "ge 1 rearm 수락"
        assert S2.handle({"cmd": "rearm", "req_id": "r", "seq": 10}, 26.0, **ok_ctx)["accepted"]
        settle(S2, 27.0)
        assert not S2.handle(rep, 27.0, **ok_ctx)["accepted"], "rearm 뒤 옛 반복 메시지가 켜짐"
        assert S2.handle(step(11), 27.0, **ok_ctx)["accepted"]
        # 끄기는 seq 없거나 오래돼도 적용(안전 방향)
        off = S2.handle({"cmd": "step", "enable": False}, 27.1, **dict(ok_ctx, ge_now=True, want_now=True))
        assert off["accepted"] and off["action"] == "set_want" and off["want"] is False and "주의" in off["why"], off
        # estop 도 래치
        S2.handle({"cmd": "estop"}, 28.0, **ok_ctx)
        settle(S2, 29.0)
        assert not S2.handle(step(12), 29.0, **ok_ctx)["accepted"]
        # (5) 스텝 중 stand/walk/march 재발사·교체 거부(H4/H5), 비활성 stand 만 launch
        S3 = _W3.W3Supervisor()
        for c in ("stand", "walk", "march", "turn", "foo"):
            dd = S3.handle({"cmd": c}, 1.0, **dict(ok_ctx, ge_now=True, want_now=True))
            assert not dd["accepted"] and dd["action"] == "none", (c, dd)
        for c in ("walk", "march"):
            assert not S3.handle({"cmd": c}, 1.0, **dict(ok_ctx, active=False))["accepted"], f"비활성 {c} 수락(가족 대체 경로)"
        assert S3.handle({"cmd": "stand"}, 1.0, **dict(ok_ctx, active=False))["action"] == "launch"
        # (6) 켜기 선행조건(L1): 접촉 없음·한발·접촉 짧음·기울기·흔들림·관측 낡음·ge 1·종전 3종
        rej = {}
        for name, kw, ctx in (("한발", dict(contact=(1.0, 0.0)), ok_ctx), ("무접촉", dict(contact=(0.0, 0.0)), ok_ctx),
                              ("접촉 0.5 s", dict(dur=0.5), ok_ctx),
                              ("기울기 9°", dict(grav=(math.sin(math.radians(9)), 0.0, -math.cos(math.radians(9)))), ok_ctx),
                              ("흔들림", dict(gyro=(0.4, 0.0, 0.0)), ok_ctx),
                              ("ge 1", {}, dict(ok_ctx, ge_now=True)), ("미실행", {}, dict(ok_ctx, active=False)),
                              ("워밍업", {}, dict(ok_ctx, warmed=False)), ("블렌딩", {}, dict(ok_ctx, blend_done=False))):
            s = _W3.W3Supervisor()
            settle(s, 5.0, **kw)
            dd = s.handle(step(1), 5.0, **ctx)
            assert not dd["accepted"], f"{name} 인데 켜기 수락"
            rej[name] = dd["why"].split(" ")[0]
        s = _W3.W3Supervisor()
        settle(s, 5.0)
        assert not s.handle(step(1), 5.2, **ok_ctx)["accepted"], "관측 0.2 s 낡음인데 수락"
        s = _W3.W3Supervisor()
        settle(s, 5.0, grav=(math.sin(math.radians(6)), 0.0, -math.cos(math.radians(6))))
        assert s.handle(step(1), 5.0, **ok_ctx)["accepted"], "6° 직립·정착인데 거부"
        # 설정 파서: 범위 밖·모르는 키 거부
        for bad in ('{"enable_tilt_deg": 30}', '{"foo": 1}', '{"enable_quiet_w": true}', '[1]'):
            try:
                _W3.parse_w3_safety(bad)
                raise AssertionError(f"설정 거부 실패: {bad}")
            except ValueError:
                pass
        assert _W3.parse_w3_safety('{"enable_tilt_deg": 5}').enable_tilt_deg == 5.0
        # (7) 지연 동결(H2): 한발 지지 중 트립 → 래치 해제(반주기 경계) 뒤 양발 지지에서 동결, 즉시 동결 아님
        lat = []
        for trip in range(66, 66 + 46):          # 켜기(60) 뒤 반주기 전 구간 위치에서 트립
            r = _w3_sup_closed_loop(trip)
            assert r["single_support_at_trip"], r
            assert r["kind"] == "support" and r["contact_at_freeze"] == [1.0, 1.0], r
            assert r["latch_off"] is not None and r["freeze"] > r["latch_off"] >= r["trip"], r
            lat.append((r["freeze"] - r["trip"]) * TICK_SEC)
        assert max(lat) < _W3.W3_SAFETY_DEFAULT.freeze_timeout_s, lat
        # 접촉이 끝내 안 들어오면 타임아웃 동결(사유 기록)
        r = _w3_sup_closed_loop(80, contact_ok=False)
        assert r["kind"] == "timeout" and abs((r["freeze"] - r["trip"]) * TICK_SEC - _W3.W3_SAFETY_DEFAULT.freeze_timeout_s) < 0.03, r
        # 이미 서 있으면(ge 0·양발·정착) 트립 틱에 바로 동결
        r = _w3_sup_closed_loop(40, enable_tick=10_000)
        assert r["freeze"] == r["trip"] == 40 and r["kind"] == "support", r
        # 동결 뒤 래치 유지 → rearm 전 켜기 거부
        s = _W3.W3Supervisor()
        settle(s, 3.0)
        s.begin_freeze(3.0, "x"); s.on_frozen("x")
        assert s.stop_latched and s.freeze_state is None and not s.handle(step(1), 3.0, **ok_ctx)["accepted"]
    finally:
        if not was_on:
            disable_experimental_w3()
    return (f"형식 6종 거부·중복/오래된 seq 거부·stop 뒤 0.3 s 늦은 켜기 거부·1 Hz 반복(seq 7) {acc}·rearm 뒤 옛 메시지 거부·"
            f"끄기 무조건 적용·estop 래치 / 실행 중 stand·walk·march·turn 거부, 비활성 walk·march 거부 / 켜기 거부 {list(rej)}·관측 낡음 거부·6° 수락 / "
            f"한발 지지 트립 46위치: 래치 해제 뒤 양발 지지에서 동결, 트립→동결 {min(lat):.2f}–{max(lat):.2f}s, 접촉 부재 = 타임아웃 "
            f"{_W3.W3_SAFETY_DEFAULT.freeze_timeout_s:g}s, 서 있으면 즉시")


def _selftest_w3_cmd():
    """1007 W3S3d 명령 채널·리드 발·헤딩 고리: ① 학습 env.yaml 계약(리드 규칙 상수·명령 범위·heading_command 끔·지연 1–5·기본자세·
    관측 필터)과 배포 상수 일치 ② 명령 검증(단일 축·범위·0 허용·bool/비유한 거부)·상한 설정 범위 ③ 감독기: 켜기 = 명령 동반·
    걷는 중 변경 = 체류·같은 명령 무변화·채널 없는 키는 0 아닌 속도 거부 ④ 리드 발: 켜기 틱 위상 0.5 ⇔ vy<−0.02 또는 (|vy|<0.02 ∧ wz<−0.08)
    ⑤ 헤딩 고리 식 = eval_w3 set_cmd 식(독립 float64 재구현) · 정지 시 ψ* 해제 · 회전 명령 중 고리 끔 · 오차 상한·타임아웃 → 걷기 정지
    ⑥ 실모델(exported 있으면) 걷기 500틱 유한·슬루 4°/틱. 토치 소스 수준 동치는 PC rl_walking/deploy/selftest_w3_equiv.py [E]·[F]."""
    import numpy as np
    was_on = EXPERIMENTAL_W3
    if not was_on:
        enable_experimental_w3()
    notes = []
    try:
        k = _W3.W3S3D_CKPT_KEY
        assert k in W3_CKPTS and k in W3_CMD_LIMITS and k.startswith("stand_") and _W3.W3_CKPT_KEY not in W3_CMD_LIMITS
        assert CHECKPOINTS[k].endswith(os.path.join(_W3.W3S3D_RUN_REL, "exported"))
        cfg, lim = W3_CKPTS[k], W3_CMD_LIMITS[k]
        # ① env.yaml 계약
        yml = _ckpt_env_yaml(k)
        if os.path.isfile(yml):
            c = _parse_env_contract(yml)
            for j, want in zip(JOINT_ORDER, CKPT_DEFAULT_POSE[k]):
                assert abs(c["joint_pos"][j] - want) < 1e-6, f"W3S3d env.yaml 기본자세 {j} 불일치"
            assert ckpt_obs_filter(k) == (0.27, 0.25), ckpt_obs_filter(k)
            w = _w3_env_contract(yml)
            g, cm = w["w3_gait"], w["w3_cmd"]
            assert g.get("f_step") == cfg.f_step and g.get("ramp_s", 0.2) == cfg.ramp_s, g
            assert g.get("lead_vy_cmin", 0.0) == cfg.lead_vy_cmin and g.get("lead_wz_cmin", 0.0) == cfg.lead_wz_cmin, g
            assert cm.get("stage") == 3 and w.get("heading_command") is False, (cm, w.get("heading_command"))
            assert tuple(cm.get("vx_range", (0.03, 0.10))) == lim.vx_fwd and tuple(cm["vx_back_range"]) == lim.vx_back, cm
            if W3_CMD_LIMITS_OVERRIDE is None:
                assert tuple(cm["vy_range"]) == lim.vy and tuple(cm["wz_range"]) == lim.wz, cm
            assert cm.get("p_hloop", 0.0) == 0.0          # 학습에 고리 없음 → 고리는 평가(eval_w3 --heading_loop)로만 검증된 배포 장치
            assert (w.get("min_delay_steps"), w.get("max_delay_steps")) == (1, 5), w
            notes.append("env.yaml 계약(리드 0.02/0.08·범위·heading 끔·지연 1–5·CROUCH_V5·필터) 일치")
        else:
            notes.append("env.yaml 없음(계약 SKIP)")
        # ② 명령 검증
        V = _W3.validate_w3_cmd
        for good, want in (({}, (0.0, 0.0, 0.0)), ({"vx": 0.06}, (0.06, 0.0, 0.0)), ({"vx": -0.03}, (-0.03, 0.0, 0.0)),
                           ({"vy": -0.04}, (0.0, -0.04, 0.0)), ({"wz": 0.2}, (0.0, 0.0, 0.2)), ({"vx": 0.10, "vy": 0}, (0.10, 0.0, 0.0))):
            assert V(good, lim) == want, (good, V(good, lim))
        bad_cmds = ({"vx": 0.02}, {"vx": 0.11}, {"vx": -0.07}, {"vy": 0.05}, {"vy": 0.01}, {"wz": 0.3}, {"wz": -0.05},
                    {"vx": 0.05, "wz": 0.1}, {"vx": 0.05, "vy": 0.03}, {"vx": True}, {"vy": float("nan")}, {"wz": "0.1"})
        for bad in bad_cmds:
            try:
                V(bad, lim)
                raise AssertionError(f"거부돼야 할 명령 통과 {bad}")
            except ValueError:
                pass
        L2 = _W3.parse_w3_cmd_limits('{"vy_max": 0.08, "wz_max": 0.3}')
        assert V({"vy": 0.08}, L2) == (0.0, 0.08, 0.0) and V({"wz": -0.3}, L2) == (0.0, 0.0, -0.3)
        for bad in ('{"vy_max": 0.1}', '{"wz_max": 0.5}', '{"vx_max": 0.2}', '{"min_dwell_s": 0.1}', '[1]'):
            try:
                _W3.parse_w3_cmd_limits(bad)
                raise AssertionError(f"범위 밖 상한 통과 {bad}")
            except ValueError:
                pass
        assert _W3.parse_w3_heading_loop("off") is None and _W3.parse_w3_heading_loop(None) == _W3.W3_HL_DEFAULT
        for bad in ('{"kp": 2}', '{"wz_max": 1.0}', '{"foo": 1}', '{"timeout_s": 1000}'):
            try:
                _W3.parse_w3_heading_loop(bad)
                raise AssertionError(f"범위 밖 고리 설정 통과 {bad}")
            except ValueError:
                pass
        notes.append(f"명령 검증 정상 {6}·거부 {len(bad_cmds)}·상한 설정 범위 거부 5·고리 설정 거부 4")
        # ③ 감독기
        S = _W3.W3Supervisor(cmd_limits=lim)
        t = 100.0
        for i in range(60):
            S.observe(t + i * 0.02, (1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        t += 1.2
        kw = dict(active=True, warmed=True, blend_done=True, ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 1, "vy": 0.06}, t, **kw)
        assert not r["accepted"] and "범위" in r["why"], r
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 2, "vy": -0.04, "heading_loop": False}, t, **kw)
        assert r["accepted"] and r["action"] == "set_want" and r["cmd"] == (0.0, -0.04, 0.0) and r["hl"] is False, r
        kw.update(ge_now=True, want_now=True, cmd_now=(0.0, -0.04, 0.0))
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 3, "vy": -0.04, "heading_loop": False}, t + 0.1, **kw)
        assert r["accepted"] and r["action"] == "none", r
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 4, "vx": 0.05}, t + 0.5, **kw)
        assert not r["accepted"] and "체류" in r["why"], r
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 5, "vx": 0.05}, t + 1.1, **kw)
        assert r["accepted"] and r["action"] == "set_cmd" and r["cmd"] == (0.05, 0.0, 0.0) and r["hl"] is True, r
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 6, "vx": 0.05, "heading_loop": 1}, t + 3, **kw)
        assert not r["accepted"], r
        r = S.handle({"cmd": "step", "enable": False}, t + 3.1, **kw)
        assert r["accepted"] and r["action"] == "set_want" and r["want"] is False, r
        S0 = _W3.W3Supervisor()
        for i in range(60):
            S0.observe(t + i * 0.02, (1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))
        r = S0.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 1, "vx": 0.05}, t + 1.2,
                      active=True, warmed=True, blend_done=True, ge_now=False, want_now=False)
        assert not r["accepted"] and "명령 채널 없음" in r["why"], r
        r = S0.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 2, "vx": 0}, t + 1.2,
                      active=True, warmed=True, blend_done=True, ge_now=False, want_now=False)
        assert r["accepted"] and r["cmd"] is None, r
        notes.append("감독기: 범위 밖 거부·켜기 명령 동반·같은 명령 무변화·체류 1 s 거부/통과·heading_loop 형식·끄기·W3-A 속도 거부")
        # ④ 리드 발 + ⑤ 헤딩 고리 (BridgeCore + 스텁 정책)
        q = np.array(CKPT_DEFAULT_POSE[k], dtype=np.float32)

        def mk():
            rr = PolicyRunnerStage4(model_path=lambda o: np.zeros(12, np.float32), obs_dim=320, gait_freq=1.1,
                                    default_pose_rad=CKPT_DEFAULT_POSE[k], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25, w3=cfg)
            cc = BridgeCore(k, runner=rr)
            cc.reset(); cc.heading_hold = False; cc.cmd = (0.0, 0.0, 0.0)
            return cc

        def yq(yaw):
            return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        leads = []
        for cmd, right in (((0.0, 0.0, 0.0), False), ((0.0, 0.04, 0.0), False), ((0.0, -0.04, 0.0), True), ((0.0, -0.03, 0.0), True),
                           ((0.0, 0.0, 0.2), False), ((0.0, 0.0, -0.1), True), ((0.06, 0.0, 0.0), False), ((-0.04, 0.0, 0.0), False)):
            cc = mk()
            for n in range(20):
                cc.tick((0.0, 0.0, 0.0), yq(0.3), q, np.zeros(12), [1.0, 1.0])
            cc.cmd = cmd
            cc.set_gait_enable(True)
            cc.tick((0.0, 0.0, 0.0), yq(0.3), q, np.zeros(12), [1.0, 1.0])
            clk = cc.runner._w3_clock
            assert clk.ge and float(clk.u) == (0.5 if right else 0.0), (cmd, clk.u)
            obs = cc.runner.last_obs
            assert abs(float(obs[30 + 12]) - cmd[0]) < 1e-7 and abs(float(obs[31 + 12]) - cmd[1]) < 1e-7, "명령 관측 ≠ 명령(켜기 틱)"
            leads.append("R" if right else "L")
        notes.append("리드 발 " + "".join(leads) + "(제자리·+vy·−vy·−0.03·+wz·−wz·앞·뒤)")
        # 고리 식: 독립 float64 재구현(eval_w3 set_cmd: tgt 첫 틱 래치, 이후 tgt=wrap(tgt+wz0·dt), wz=clip(wz0+K·wrap(tgt−ψ), ±max))
        hc = W3_HL_CFG or _W3.W3_HL_DEFAULT
        cc = mk()
        if cc.w3hl is None:
            notes.append("헤딩 고리 꺼짐(--w3-heading-loop off) — 고리 시험 SKIP")
        else:
            for n in range(10):
                cc.tick((0.0, 0.0, 0.0), yq(1.0), q, np.zeros(12), [1.0, 1.0])
            cc.cmd = (0.06, 0.0, 0.0)
            cc.set_gait_enable(True)
            tgt, worst = None, 0.0
            wr = lambda x: math.atan2(math.sin(x), math.cos(x))  # noqa: E731
            for n in range(300):
                psi = 1.0 + 0.12 * math.sin(n * 0.05) + 0.0004 * n
                if n == 150:
                    psi = wr(psi + 2 * math.pi)            # ±π 감싸기 경로
                cc.tick((0.0, 0.0, 0.0), yq(psi), q, np.zeros(12), [1.0, 1.0])
                tgt = psi if tgt is None else wr(tgt + 0.0 * 0.02)
                ref = max(-hc.wz_max, min(hc.wz_max, 0.0 + hc.kp * wr(tgt - psi)))
                got = float(cc.runner.last_obs[32 + 12])
                worst = max(worst, abs(got - ref))
            assert worst < 1e-6, f"고리 wz 오차 {worst}"
            lat = cc.w3hl.latched
            cc.set_gait_enable(False)                      # 정지(want 0) → 다음 틱 ψ* 해제
            cc.cmd = (0.0, 0.0, 0.0)
            cc.tick((0.0, 0.0, 0.0), yq(2.0), q, np.zeros(12), [1.0, 1.0])
            assert lat and not cc.w3hl.latched and float(cc.runner.last_obs[44]) == 0.0, "정지 뒤 ψ* 미해제/wz ≠ 0"
            # 회전 명령 중 고리 끔(wz 그대로·ψ* 없음)
            cc.cmd = (0.0, 0.0, -0.15); cc.set_gait_enable(True)
            for n in range(30):
                cc.tick((0.0, 0.0, 0.0), yq(2.0 - 0.003 * n), q, np.zeros(12), [1.0, 1.0])
            assert not cc.w3hl.latched and abs(float(cc.runner.last_obs[44]) + 0.15) < 1e-7
            # 오차 상한 → 걷기 정지
            cc.cmd = (0.0, 0.0, 0.0)
            for n in range(5):
                cc.tick((0.0, 0.0, 0.0), yq(2.0), q, np.zeros(12), [1.0, 1.0])
            cc.tick((0.0, 0.0, 0.0), yq(2.0 + math.radians(hc.err_max_deg + 1.0)), q, np.zeros(12), [1.0, 1.0])
            assert not cc.runner._w3_clock.want and cc.cmd == (0.0, 0.0, 0.0) and cc.w3hl_events \
                and "오차" in cc.w3hl_events[-1]["why"], "오차 상한에서 걷기 미정지"
            # 타임아웃 → 걷기 정지
            cc2 = mk()
            cc2.cmd = (0.0, 0.0, 0.0); cc2.set_gait_enable(True)
            nmax = int(round(hc.timeout_s / 0.02)) + 3
            for n in range(nmax):
                cc2.tick((0.0, 0.0, 0.0), yq(0.5), q, np.zeros(12), [1.0, 1.0])
                if not cc2.runner._w3_clock.want:
                    break
            assert cc2.w3hl_events and "타임아웃" in cc2.w3hl_events[-1]["why"] and abs(n * 0.02 - hc.timeout_s) < 0.1, n
            notes.append(f"고리 = eval 식(300틱 최대 차 {worst:.1e}, ±π 감싸기 포함)·정지 ψ* 해제·회전 중 끔·오차 상한/타임아웃 {hc.timeout_s:g}s → 걷기 정지")
        # ⑥ 실모델 500틱(걷기)
        d = CHECKPOINTS[k]
        if os.path.exists(os.path.join(d, "policy.pt")) or os.path.exists(os.path.join(d, "policy.onnx")):
            cc = BridgeCore(k)
            cc.reset(); cc.heading_hold = False; cc.cmd = (0.0, 0.0, 0.0)
            assert cc.w3 and cc.w3cmd and cc.runner.obs_dim == 320
            qq = q.copy(); prev = qq.copy(); smax = 0.0
            seq = [(0, (0.0, 0.0, 0.0), False), (50, (0.06, 0.0, 0.0), True), (200, (0.0, -0.04, 0.0), True),
                   (300, (0.0, 0.0, -0.2), True), (400, (0.0, 0.0, 0.0), False)]
            t0 = time.perf_counter()
            for n in range(500):
                for (nn, cmd, en) in seq:
                    if n == nn:
                        cc.cmd = cmd; cc.set_gait_enable(en)
                out = cc.tick((0.0, 0.0, 0.0), yq(0.2), qq, np.zeros(12), [1.0, 1.0])
                assert np.isfinite(out).all(), f"W3S3d 틱 {n} 비유한"
                dd = float(np.abs(out - prev).max()); smax = max(smax, dd)
                assert dd <= SLEW_RAD * 1.001 + 1e-6, f"W3S3d 틱 {n} 슬루 초과"
                prev = out; qq = out
            dt = (time.perf_counter() - t0) / 500
            assert dt < TICK_SEC
            notes.append(f"실모델 500틱(서기→앞→옆−→회전−→정지) 유한·슬루 최대 {math.degrees(smax):.2f}°/틱·{dt*1000:.2f}ms [{cc.runner.backend}]")
        else:
            notes.append("실모델 exported 없음(500틱 SKIP)")
    finally:
        if not was_on:
            disable_experimental_w3()
    return " / ".join(notes)


def _selftest_w3_monitor():
    """1007 2 차 대비책 감시(--w3-monitor) 통합: ① 꺼짐 = core.w3mon None·켜기 판정 그대로 ② 켜짐(act) 정상 입력 = 꺼짐과 목표각 비트 동일
    (W3S3d 키 + 헤딩 고리 + 앞 0.06 걷기 600 틱, 관측 의존 스텁 정책) ③ 자세 위반 → 같은 틱 want 0·명령 0·반주기 경계 ge 0·정지 래치(rearm 필요)
    ④ note 모드 = 같은 입력에서 목표각 비트 동일 + 사건 기록 ⑤ 요 표류 → 고리 금지(wz 보정 없음) ⑥ 병진 중 고리 wz 상한 ⑦ 켜기 거부(센서 나이)
    ⑧ 정책 재시작(reset)에도 세션 상태(고리 금지) 유지. 단위 시험 전체는 rl_walking/deploy/selftest_w3_contingency.py."""
    import numpy as np
    global W3_MON_CFG
    was_on, was_cfg = EXPERIMENTAL_W3, W3_MON_CFG
    M = _w3c()
    try:
        enable_experimental_w3()
        k = _W3.W3S3D_CKPT_KEY
        cfg = W3_CKPTS[k]
        q = np.array(CKPT_DEFAULT_POSE[k], dtype=np.float32)
        Wm = np.random.default_rng(7).normal(0, 0.05, (12, 320)).astype(np.float32)

        def mk(moncfg):
            global W3_MON_CFG
            W3_MON_CFG = moncfg
            rr = PolicyRunnerStage4(model_path=lambda o: np.tanh(Wm @ np.asarray(o, np.float32)).astype(np.float32), obs_dim=320,
                                    gait_freq=1.1, default_pose_rad=CKPT_DEFAULT_POSE[k], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25, w3=cfg)
            cc = BridgeCore(k, runner=rr)
            cc.reset(); cc.heading_hold = False; cc.cmd = (0.0, 0.0, 0.0)
            return cc

        def yq(yaw):
            return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

        def contacts(cc):
            u = float(cc.runner._w3_clock.u)
            if not cc.runner._w3_clock.ge:
                return [1.0, 1.0]
            return [0.0 if 0.05 < u < 0.45 else 1.0, 0.0 if 0.55 < u < 0.95 else 1.0]

        def scenario(cc, n_walk=600, pitch_bad_at=None):
            outs = []
            for n in range(60):
                outs.append(cc.tick((0.01, 0.0, 0.0), yq(0.4), q, np.zeros(12), [1.0, 1.0]))
            cc.cmd = (0.06, 0.0, 0.0); cc.set_gait_enable(True)
            for n in range(n_walk):
                pitch = math.radians(1.5 * math.sin(n * 0.14))
                if pitch_bad_at is not None and n >= pitch_bad_at:
                    pitch = math.radians(14.0)
                quat = (math.cos(pitch / 2) * math.cos(0.2), -math.sin(pitch / 2) * math.sin(0.2),
                        math.sin(pitch / 2) * math.cos(0.2), math.cos(pitch / 2) * math.sin(0.2))   # 요 0.4 + 피치
                outs.append(cc.tick((0.3 * math.sin(n * 0.2), 0.2, 0.0), quat, q + 0.01 * math.sin(n * 0.1), np.zeros(12), contacts(cc)))
                if cc.w3mon is not None:
                    cc.w3mon.health(foot_age_s=0.05)
                    cc.w3mon.track(outs[-1], q)
            return np.asarray(outs)

        notes = []
        # ① 꺼짐
        c0 = mk(None)
        assert c0.w3mon is None
        dec = {"accepted": True, "why": "ok", "action": "set_want", "want": True, "cmd": (0.06, 0, 0), "hl": True}
        assert w3_monitor_gate(dec, None) is dec
        a_off = scenario(c0)
        # ② 켜짐 act 정상 입력 → 비트 동일
        c1 = mk(M.W3MonitorConfig())
        a_on = scenario(c1)
        assert c1.w3mon is not None and a_on.shape == a_off.shape and np.array_equal(a_on, a_off), \
            f"감시 켜짐(정상 입력) 목표각 ≠ 꺼짐 (최대 차 {np.abs(a_on - a_off).max()})"
        assert not [e for e in c1.w3hl_events if e.get("event") in ("w3_mon_stop", "w3_mon")], c1.w3hl_events
        notes.append(f"act 정상 입력 {len(a_on)} 틱 비트 동일·사건 0")
        # ④ note 모드 + 위반 → 비트 동일(꺼짐의 같은 위반 입력과) + 기록
        c_off_bad = mk(None); a_off_bad = scenario(c_off_bad, pitch_bad_at=200)
        c_note = mk(M.W3MonitorConfig(mode="note")); a_note = scenario(c_note, pitch_bad_at=200)
        assert np.array_equal(a_note, a_off_bad), "note 모드가 목표각을 바꿈"
        assert any(e.get("kind") == "tilt" and e.get("action") == "note:stop_walk" for e in c_note.w3hl_events), c_note.w3hl_events
        notes.append("note 위반 입력 비트 동일·기록")
        # ③ act 위반 → 정지
        c2 = mk(M.W3MonitorConfig())
        for n in range(60):
            c2.tick((0.01, 0.0, 0.0), yq(0.4), q, np.zeros(12), [1.0, 1.0])
        c2.cmd = (0.06, 0.0, 0.0); c2.set_gait_enable(True)
        _p = math.radians(14.0)
        bad_q = (math.cos(_p / 2) * math.cos(0.2), -math.sin(_p / 2) * math.sin(0.2),
                 math.sin(_p / 2) * math.cos(0.2), math.cos(_p / 2) * math.sin(0.2))     # 요 0.4 유지 + 피치 14°
        t_stop = t_ge0 = None
        for n in range(200):
            c2.tick((0.0, 0.0, 0.0), yq(0.4) if n < 50 else bad_q, q, np.zeros(12), contacts(c2))
            clk = c2.runner._w3_clock
            if t_stop is None and not clk.want:
                t_stop = n
                assert c2.cmd == (0.0, 0.0, 0.0) and any(e.get("event") == "w3_mon_stop" and e.get("kind") == "tilt"
                                                         for e in c2.w3hl_events), c2.w3hl_events
            if t_stop is not None and t_ge0 is None and not clk.ge:
                t_ge0 = n
        _mc = M.W3MonitorConfig()
        _exp = 0.2 - _mc.walk_tau_s * math.log(1.0 - _mc.walk_tilt_deg / 14.0)   # 저역(EMA) 도달 + 지속 0.2 s
        assert t_stop is not None and abs((t_stop - 50) * 0.02 - _exp) < 0.06, (t_stop, _exp)
        assert t_ge0 is not None and (t_ge0 - t_stop) * 0.02 <= 0.455 + 0.02 + 1e-9, (t_stop, t_ge0)
        S = _W3.W3Supervisor(cmd_limits=W3_CMD_LIMITS[k])
        w3_monitor_latch(S, "monitor tilt: 시험")
        r = S.handle({"cmd": "step", "enable": True, "req_id": "a", "seq": 1, "vx": 0.04}, 0.0, active=True, warmed=True,
                     blend_done=True, ge_now=False, want_now=False, cmd_now=(0.0, 0.0, 0.0))
        assert not r["accepted"] and "정지 래치" in r["why"], r
        notes.append(f"피치 14° 계단 {(t_stop - 50) * 0.02:.2f}s 뒤(저역 도달 + 0.2 s) want 0·명령 0, ge 0 까지 {(t_ge0 - t_stop) * 0.02:.2f}s(≤ 반주기), 정지 래치 → 켜기 거부")
        # ⑤ 요 표류 → 고리 금지
        c3 = mk(M.W3MonitorConfig())
        rate = math.radians(6.0) / 60.0
        for n in range(1300):
            c3.tick((0.0, 0.0, 0.0), yq(0.4 + rate * n * 0.02), q, np.zeros(12), [1.0, 1.0])
        assert not c3.w3mon.hl_allowed()[0]
        c3.cmd = (0.06, 0.0, 0.0); c3.set_gait_enable(True)
        for n in range(20):
            c3.tick((0.0, 0.0, 0.0), yq(0.4 + math.radians(8)), q, np.zeros(12), contacts(c3))
        assert float(c3.runner.last_obs[44]) == 0.0 and not c3.w3hl_on and any(e.get("event") == "w3_hl_banned" for e in c3.w3hl_events)
        c3.reset()                                         # 정책 재시작에도 세션 상태 유지
        assert c3.w3mon is not None and not c3.w3mon.hl_allowed()[0]
        notes.append("요 표류 6°/min → 고리 금지(wz 0)·재시작 뒤 유지")
        # ⑥ 병진 중 고리 wz 상한 0.1
        for capv, want in ((0.0, 0.5 * math.radians(12)), (0.1, 0.1)):
            c4 = mk(M.W3MonitorConfig(hl_wz_translate=capv))
            for n in range(10):
                c4.tick((0.0, 0.0, 0.0), yq(1.0), q, np.zeros(12), [1.0, 1.0])
            c4.cmd = (0.06, 0.0, 0.0); c4.set_gait_enable(True)
            c4.tick((0.0, 0.0, 0.0), yq(1.0), q, np.zeros(12), [1.0, 1.0])
            for n in range(1, 61):                         # 0.2°/틱 램프(요 점프 판정 1°/틱 아래)
                c4.tick((0.0, 0.0, 0.0), yq(1.0 - math.radians(0.2 * n)), q, np.zeros(12), contacts(c4))
            assert abs(float(c4.runner.last_obs[44]) - want) < 1e-6, (capv, float(c4.runner.last_obs[44]))
        notes.append("병진 중 고리 wz 0.105 → 상한 0.1")
        # ⑦ 켜기 거부(센서 나이) — 감독기 수락 뒤 감시기 판정으로만 거부
        c5 = mk(M.W3MonitorConfig())
        for n in range(20):                                # 0.3 s 지속해야 판정(health_hold_s)
            c5.tick((0.0, 0.0, 0.0), yq(0.4), q, np.zeros(12), [1.0, 1.0])
            c5.w3mon.health(foot_age_s=0.3, fb_age_s=0.01, s8_p50_ms=1.0)
        d2 = w3_monitor_gate(dict(dec), c5.w3mon)
        assert not d2["accepted"] and d2["action"] == "none" and "센서 나이" in d2["why"], d2
        assert w3_monitor_gate({"accepted": True, "action": "set_want", "want": False}, c5.w3mon)["accepted"], "끄기는 막지 않음"
        c5.w3mon.health(foot_age_s=0.05)
        assert w3_monitor_gate(dict(dec), c5.w3mon)["accepted"]
        notes.append("센서 나이 → 켜기 거부·끄기 통과·회복 뒤 허용")
        # 다른 키(W3-A·stand)에는 감시기가 붙지 않는다
        W3_MON_CFG = M.W3MonitorConfig()
        rr = PolicyRunnerStage4(model_path=lambda o: np.zeros(12, np.float32), obs_dim=320, gait_freq=1.1,
                                default_pose_rad=CKPT_DEFAULT_POSE[_W3.W3_CKPT_KEY], act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25,
                                w3=W3_CKPTS[_W3.W3_CKPT_KEY])
        assert BridgeCore(_W3.W3_CKPT_KEY, runner=rr).w3mon is None
        return " · ".join(notes)
    finally:
        W3_MON_CFG = was_cfg
        if not was_on and EXPERIMENTAL_W3:
            disable_experimental_w3()


def run_selftest(required=()):
    """드라이런 자가검증 — ROS·로봇 불필요. 프레임 변환 4종 + 게이트 + 정책 500틱.
    required: 파일이 반드시 있어야 하는 체크포인트(기동 --checkpoint, STAND_DEFAULT) — 없으면 FAIL."""
    tests = [("하드웨어맵", _selftest_hwmap),
             ("온셋 블렌딩", _selftest_blend),
             ("stand LPF", _selftest_stand_lpf),
             ("왕복 항등", _selftest_roundtrip),
             ("수치 앵커", _selftest_anchors),
             ("게인 정책", _selftest_gains),
             ("프레임 게이트", _selftest_frame_gate),
             ("중력 게이트", _selftest_grav_gate),
             ("IMU 장착 편향", _selftest_imu_mount),
             ("롤 보정기", _selftest_roll_comp),
             ("인계 워밍업", _selftest_warmup),
             ("롤/피치 기록", _selftest_roll_pitch),
             ("발진 가드", _selftest_osc_guard),
             ("행잉 가드", _selftest_hang_guard),
             ("착지 자동 인수 게이트(1005)", _selftest_touchdown_gate),
             ("학습 계약", lambda: _selftest_contract(required)),
             ("관측 필터 계약(1003)", _selftest_obs_filter),
             ("C5 전환 경로(1004)", _selftest_c5),
             ("C5 kr 경로(1005)", _selftest_c5kr),
             ("C5 odic 조기 복귀 플래그(1007)", _selftest_c5_odic),
             ("HTURN2 실험 플래그(1006)", _selftest_hturn2),
             ("W3 실험 플래그(1006)", _selftest_w3),
             ("W3 세션 감독기(1006 감사)", _selftest_w3_supervisor),
             ("W3S3d 명령 채널·리드 발·헤딩 고리(1007)", _selftest_w3_cmd),
             ("W3 감시 2 차 대비책(1007)", _selftest_w3_monitor),
             ("정책 500틱", lambda: _selftest_policy(required))]
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
    ap.add_argument("--experimental-hturn2", action="store_true",
                    help="1006 실험(기본 꺼짐·실물 미검증): HTURN2 키 등록(W2q 25590 + cadence 클록 + 회전 피드포워드 + "
                         "HeadingHold 목표 각 |wz|≤0.3). 켜도 --checkpoint " + _H2.HTURN2_CKPT_KEY + " 일 때만 계약이 바뀐다")
    ap.add_argument("--experimental-w3", action="store_true",
                    help="1006 실험(기본 꺼짐·실물 미검증): W3 키 등록(W3-A 400 제자리 교대 스텝 엔진 — gait_enable 래치 클록). "
                         "켜도 --checkpoint " + _W3.W3_CKPT_KEY + " 일 때만 계약이 바뀐다. 시작 = stand(서기), "
                         "스텝 = {\"cmd\":\"step\",\"enable\":true,\"req_id\":\"..\",\"seq\":N}(seq 단조 증가), "
                         "stop/estop/가드 = 정지 래치 → {\"cmd\":\"rearm\",\"req_id\":\"..\",\"seq\":N}. 응답 " + BRIDGE_ACK_TOPIC)
    ap.add_argument("--w3-safety", default=None,
                    help="W3 세션 감독기 문턱 JSON(예: '{\"enable_tilt_deg\": 6}'). 키·범위는 w3_deploy.W3_SAFETY_RANGES — "
                         "범위 밖·모르는 키는 기동 거부. 생략 = 보수적 기본값")
    ap.add_argument("--w3-heading-loop", default=None,
                    help="1007 W3 명령 채널 키(" + _W3.W3S3D_CKPT_KEY + ")의 외부 헤딩 고리: 생략 = 켜짐(K 0.5·±0.4 rad/s·오차 상한 15°·"
                         "120 s), 'off' = 끔, JSON = 설정(키·범위 w3_deploy.W3_HL_RANGES)")
    ap.add_argument("--w3-cmd-limits", default=None,
                    help="1007 W3 명령 채널 상한 JSON(vy_max ≤0.08·wz_max ≤0.3·min_dwell_s) — 생략 = W3S3d 학습 범위(|vy| ≤0.04·|wz| ≤0.2)")
    ap.add_argument("--c5-settle", default=None,
                    help="1007 병목 2 (기본 꺼짐·실물 미검증): C5 KE 조기 복귀 판정기. 생략/'imu' = 현행 IMU 규칙, 'odic' = 다리 운동학 오도메트리 "
                         "0.9 s 주기 평균 |v̄| < 0.07 m/s·양발 ≥ 0.3 s·켜진 뒤 ≥ 0.9 s + 무릎 처짐 보상 (τ−τ_인계)/k̂(k̂ 42 가정값), "
                         "'odicw' = odic + 감지 |ω_xy| < 0.4 rad/s(1007, STOPEST §11), "
                         "JSON = 설정 덮어쓰기(키 " + ", ".join(_ODIC_KEYS) + "). 조기 복귀가 있는 C5 키(stand_c5kr_*)에서만 허용")
    ap.add_argument("--w3-monitor", default=None,
                    help="1007 2 차 대비책(기본 꺼짐·실물 미검증, reports/gen_w3/CONTINGENCY.md): W3 명령 채널 키 걷기 감시. "
                         "'on' = 조치(자세·접촉·요·지연 판정 → 걷기 정지·켜기 거부·고리 금지), 'note' = 같은 판정 기록만(오트립 계수), "
                         "JSON = 설정(키·범위 w3_contingency.W3_MON_RANGES). --experimental-w3 와 함께만")
    ap.add_argument("--c5-overtime", default=None,
                    help="1007 2 차 대비책(기본 꺼짐): C5 회복 모드 연속 체류 상한 'note:초'(기록) | 'freeze:초'(양발 접촉일 때 STOP_ALL). C5 키에서만")
    ap.add_argument("--checkpoint", default=STAND_DEFAULT, choices=list(CHECKPOINTS) + [_H2.HTURN2_CKPT_KEY] + list(_W3.W3_KEYS),
                    help="기본값은 배포 데뷔 정책(STAND_DEFAULT). 0907: 구 walk 는 젯슨에서 삭제돼 "
                         "기본값으로 두면 selftest 가 필수 파일 없음으로 실패한다")
    args = ap.parse_args()
    if args.checkpoint == _H2.HTURN2_CKPT_KEY and not args.experimental_hturn2:
        print(f"[거부] --checkpoint {args.checkpoint} 는 --experimental-hturn2 를 명시해야 선택 가능", file=sys.stderr)
        return 2
    if args.checkpoint in _W3.W3_KEYS and not args.experimental_w3:
        print(f"[거부] --checkpoint {args.checkpoint} 는 --experimental-w3 를 명시해야 선택 가능", file=sys.stderr)
        return 2
    if args.experimental_hturn2 and args.experimental_w3:
        print("[거부] --experimental-hturn2 와 --experimental-w3 는 함께 켤 수 없음", file=sys.stderr)
        return 2
    if args.w3_safety is not None and not args.experimental_w3:
        print("[거부] --w3-safety 는 --experimental-w3 와 함께만", file=sys.stderr)
        return 2
    if (args.w3_heading_loop is not None or args.w3_cmd_limits is not None) and not args.experimental_w3:
        print("[거부] --w3-heading-loop·--w3-cmd-limits 는 --experimental-w3 와 함께만", file=sys.stderr)
        return 2
    global W3_MON_CFG, C5_OVERTIME
    W3_MON_CFG, C5_OVERTIME = None, None          # 기동마다 명시(같은 프로세스에서 main 을 다시 부르는 셀프테스트 하네스 포함)
    if args.w3_monitor is not None and not args.experimental_w3:
        print("[거부] --w3-monitor 는 --experimental-w3 와 함께만", file=sys.stderr)
        return 2
    if args.w3_monitor is not None:
        try:
            W3_MON_CFG = _w3c().parse_w3_monitor(args.w3_monitor)
        except ValueError as e:      # json.JSONDecodeError 포함
            print(f"[거부] --w3-monitor: {e}", file=sys.stderr)
            return 2
        if W3_MON_CFG is not None:
            print(f"[경고] --w3-monitor: 2 차 대비책 감시 켜짐(실물 미검증) — {_w3c().config_dict(W3_MON_CFG)}", file=sys.stderr)
    if args.c5_overtime is not None:
        if args.checkpoint not in C5_PAIRS:
            print(f"[거부] --c5-overtime 는 C5 키에서만(현재 --checkpoint {args.checkpoint})", file=sys.stderr)
            return 2
        try:
            C5_OVERTIME = _w3c().parse_c5_overtime(args.c5_overtime)
        except ValueError as e:
            print(f"[거부] --c5-overtime: {e}", file=sys.stderr)
            return 2
        print(f"[경고] --c5-overtime: 회복 모드 체류 상한 {C5_OVERTIME.max_s:g}s · 조치 {C5_OVERTIME.action}", file=sys.stderr)
    if args.experimental_w3:
        global W3_SAFETY, W3_HL_CFG, W3_CMD_LIMITS_OVERRIDE
        try:
            W3_SAFETY = _W3.parse_w3_safety(args.w3_safety)
            W3_HL_CFG = _W3.parse_w3_heading_loop(args.w3_heading_loop)
            W3_CMD_LIMITS_OVERRIDE = (None if args.w3_cmd_limits is None
                                      else _W3.parse_w3_cmd_limits(args.w3_cmd_limits))
        except ValueError as e:      # json.JSONDecodeError 포함
            print(f"[거부] W3 설정: {e}", file=sys.stderr)
            return 2
        enable_experimental_w3()
        print(f"[경고] --experimental-w3: 실험 기능 켜짐(실물 미검증) — 키 {', '.join(_W3.W3_KEYS)}"
              f" · 헤딩 고리 {W3_HL_CFG} · 명령 범위 {W3_CMD_LIMITS_OVERRIDE or 'W3S3d 학습 범위'}", file=sys.stderr)
    if args.experimental_hturn2:
        enable_experimental_hturn2()
        print(f"[경고] --experimental-hturn2: 실험 기능 켜짐(실물 미검증) — 키 {_H2.HTURN2_CKPT_KEY}", file=sys.stderr)
    try:
        _c5s = parse_c5_settle(args.c5_settle)
    except ValueError as e:      # json.JSONDecodeError 포함
        print(f"[거부] --c5-settle: {e}", file=sys.stderr)
        return 2
    if _c5s is not None:
        if args.checkpoint not in C5_PAIRS or C5_EXTRA.get(args.checkpoint, {}).get("det", {}).get("early") is None:
            print(f"[거부] --c5-settle {args.c5_settle} 는 조기 복귀가 있는 C5 키에서만(현재 --checkpoint {args.checkpoint})", file=sys.stderr)
            return 2
        set_c5_settle(_c5s)
        print(f"[경고] --c5-settle: odic 조기 복귀 판정기 켜짐(실물 미검증, k̂ 가정값) — 설정 {_c5s or 'ODIC_DEFAULT'}", file=sys.stderr)
    if args.selftest:
        return run_selftest(required=(args.checkpoint, STAND_DEFAULT))

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
            self._gnote = {}            # guard_mode=data 기록 throttle: {이벤트: (마지막 사유, 마지막 시각)}
            self.core = BridgeCore(args.checkpoint)
            self._launches = 0          # 정책 시작 횟수(1006 W3 응답용 계수 — 제어 무관)
            # 1006 W3 세션 감독기(감사 H3–H6·L1): --experimental-w3 + W3 키 기동일 때만. 그 밖에는 None = 종전 경로 그대로.
            self.w3sup = (_W3.W3Supervisor(W3_SAFETY, **({"cmd_limits": W3_CMD_LIMITS[args.checkpoint]}
                                                         if args.checkpoint in W3_CMD_LIMITS else {}))
                          if (EXPERIMENTAL_W3 and args.checkpoint in W3_CKPTS) else None)
            self.frames = frames        # 모터↔관절 프레임 변환 (기동 시 검증 완료)
            self.roll_comp = RollComp(frames.roll_comp)   # 느린 기울기 롤 보정기 — 맵 키 없으면 OFF(no-op)
            self.pitch_comp = RollComp(frames.pitch_comp)  # 앞뒤 균형 복원항 — 맵 키 없으면 OFF(no-op)
            self.guard_mode = getattr(frames, "guard_mode", "safe")
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
            self.foot_contact = None     # 0910: 패드 단위 접지 (어댑터 필드), None=구판 합산 규칙
            self.foot_ts = 0.0
            self.baseline = {}
            self.fb_age = {}             # 관절별 마지막 feedback_age_sec (워밍업 진단: 어느 관절이 낡았나)
            self.armed = False
            self.stopped_for_stale = False
            self.warmed = True           # 인계 워밍업 완료 여부 — stand 시 False 로 리셋
            self.warmup_until = 0.0      # 워밍업 데드라인(time.time 기준)
            self.osc = OscGuard(JOINT_ORDER)
            self.osc_tripped = None
            self.stop_reason = None      # A1 기록: 마지막 STOP_ALL 사유
            self.grav_last = None        # A1 기록: 마지막 stand 명령 시 중력 게이트 [ok, why]
            self.foot_raw_sum = None     # A1 기록: 어댑터 원카운트 합 [L, R]
            self.blend_t0 = None
            self.blend_from = None
            self.nocontact_t0 = None
            self.hang_guard = HangGuard(hang_mode=False)
            self.td_gate = None          # 1005 착지 자동 인수 게이트 — stand 에 touchdown_start 를 명시할 때만 생성
            self.td_hold_rel = None      # 착지 대기 중 보내는 홀드 상대 명령(첫 대기 틱 래치)
            self.stage8_cmd_rel = {}
            # 블랙박스 전용(제어 무관): stage8 피드백의 토크 추정·모터 온도 원값.
            # 0905 — 시뮬 판정이 토크 피크/포화 %를 지표로 쓰므로 실물도 같은
            # 축으로 대조할 수 있어야 한다 (실물 전류 기록 요구, 승윤님).
            self.tau_est = {}
            self.temp_raw = {}
            self._cmd_seq = 0        # 0930: 목표 명령 순번·발행 시각·출처 — stage8 cmd_timing 이 발행→처리 지연·누락을 잰다
            self._cmd_src = f"bridge:{os.getpid()}:{int(time.time())}"
            self.s8_cmd_timing = None
            self.status_ts = 0.0     # 마지막 stage8 status 수신 시각 — 블랙박스 status_age_s (0905 감사 C4: tau_est 동결값 식별)
            # stage8 soft 클램프 누적·거부 사유 (0905 감사 B7: 클램프 발생이 블랙박스에 없어 사후분석에서 보이지 않았음)
            self.clamp_total = None
            self.clamp_counts = {}
            self.reject_reason = None
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
                # 0929: 제어 루프가 디스크를 기다리지 않도록 블랙박스는 전용 스레드가 쓴다.
                #   사고: 0929 20:04:30 자세유지 루프 122 ms 송신 공백 → 모터가 명령을 잃고 두 무릎 22~25° 꺾임.
                #   /tmp·로그가 90 % 찬 eMMC 위에 있고 쓰기 평균 18 ms — 한 번 막히면 틱 전체가 멈춘다.
                #   브리지도 매 틱(50 Hz) 블랙박스를 직접 write() 하고 있었다 → 큐에 넣고 즉시 돌아간다.
                self.bb = open(path, "a", buffering=1 << 16)
                import queue as _queue, threading as _threading
                self._bbq = _queue.Queue(maxsize=20000)
                self._bb_dropped = 0
                self._bbt = _threading.Thread(target=self._bb_writer, name="bb_writer", daemon=True)
                self._bbt.start()
                self.get_logger().info(f"블랙박스 기록: {path}")
            # 1001 되돌림: 0930 의 '최신값 1칸(KEEP_LAST 1)' 은 실물에서 IMU 수신이 0.5–0.8 s 늦어져 stand 중력 게이트(0.5 s)가
            #   매번 거부했다(젯슨 dry-run A/B: 깊이 1 = 3/3 거부·age 0.68–0.82 s, 깊이 10/50 = 3/3 통과). 0929 실물 검증 값으로 복귀.
            self.create_subscription(String, JOINT_STATES_TOPIC, self._on_js, 10)
            self.create_subscription(String, STATUS_TOPIC, self._on_status, 10)
            self.create_subscription(String, IMU_TOPIC, self._on_imu, 50)
            self.create_subscription(String, FOOT_TOPIC, self._on_foot, 50)
            self.create_subscription(String, BRIDGE_CMD_TOPIC, self._on_cmd, 10)
            self.pub_cmd = self.create_publisher(String, COMMAND_TOPIC, 10)
            self.pub_dbg = self.create_publisher(String, BRIDGE_DEBUG_TOPIC, 10)
            self.pub_ack = None
            if self.w3sup is not None:     # 1006 W3 세션만 응답 토픽을 연다(종전 세션은 토픽·스키마 불변)
                self.pub_ack = self.create_publisher(String, BRIDGE_ACK_TOPIC, 10)
                self.get_logger().warn(
                    f"W3 세션 감독기 켜짐 — 응답 {BRIDGE_ACK_TOPIC}, 문턱 {W3_SAFETY}, guard_mode={self.guard_mode}"
                    + (" (data 모드: 리프트·발진 가드는 정지하지 않음 — 지연 동결은 safe 모드에서만 작동)"
                       if self.guard_mode == "data" else ""))
            self.timer = self.create_timer(TICK_SEC, self._tick)
            self.get_logger().info(
                f"RL 브리지 기동 — {'LIVE' if self.live else 'DRY-RUN'}, "
                f"checkpoint={self.core.ckpt_name} jvel_ema={getattr(self.core.runner, 'jvel_ema_mode', 'v2')}")

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
            self.status_ts = time.time()
            _ct = d.get("cmd_timing")
            self.s8_cmd_timing = _ct if isinstance(_ct, dict) else None
            self.clamp_total = d.get("soft_clamp_total")
            cc = d.get("soft_clamp_counts")
            self.clamp_counts = cc if isinstance(cc, dict) else {}
            self.reject_reason = d.get("last_reject_reason")
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
                for key, store in (("feedback_torque_est", self.tau_est),
                                   ("feedback_temperature_raw", self.temp_raw)):
                    v = info.get(key)
                    if v is not None:
                        try:
                            store[j] = float(v)
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
                self.fb_age[j] = age     # 낡은 관절도 기록 (워밍업 진단용)
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
                # 0922 승윤님: 감압 패드는 접지 여부 판단용. left_n/right_n(미보정 카운트×상수)은 **하중이 아니다**.
                #   블랙박스 진단용으로만 싣고, 판정에는 쓰지 않는다(5 N 폴백도 폐지 — 아래 접지 비트 필수).
                self.foot = (float(d.get("left_n") or 0.0), float(d.get("right_n") or 0.0))
                self.foot_raw_sum = [d.get("left_raw_sum"), d.get("right_raw_sum")]
                # 0910: 어댑터가 패드 단위 접지(4개 중 하나라도)를 주면 그것을 우선 — 없으면 합산 5N 규칙(구판 어댑터 호환)
                self.foot_contact = ((bool(d["left_contact"]), bool(d["right_contact"]))
                                     if "left_contact" in d and "right_contact" in d else None)
                if self.foot_contact is None:
                    return          # 0922: 접지 비트 없는 구판 메시지는 버린다(힘 임계로 접지를 추정하지 않는다)
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
                if self.w3sup is not None:
                    self._w3_ack({}, False, "명령이 JSON 객체가 아님")
                return
            if self.w3sup is not None:      # 1006 W3 세션(플래그 + W3 키 기동)에서만 — 감독기가 먼저 판정
                self._on_cmd_w3(d)
                return
            self._on_cmd_base(d)

        def _w3_ack(self, d, accepted, why, action=None):
            """1006 W3 세션: 모든 명령에 응답(수락/거부+이유) — BRIDGE_ACK_TOPIC + 블랙박스."""
            now = time.time()
            ack = {"timestamp": now, "cmd": d.get("cmd"), "req_id": d.get("req_id"), "seq": d.get("seq"),
                   "accepted": bool(accepted), "why": str(why), "action": action,
                   "active": bool(self.core.active), "w3": self.core.runner._w3_clock.snapshot() if self.core.w3 else None,
                   "sup": self.w3sup.snapshot(), "guard_mode": self.guard_mode}
            try:
                self.pub_ack.publish(String(data=json.dumps(ack, default=str)))
            except Exception:  # noqa: BLE001 — 응답 실패가 명령 처리를 막으면 안 된다
                pass
            self._bb_write(dict(ack, event="w3_cmd_ack"))
            (self.get_logger().info if accepted else self.get_logger().warn)(
                f"W3 명령 {d.get('cmd')!r} seq={d.get('seq')} {'수락' if accepted else '거부'}: {why}")

        def _on_cmd_w3(self, d):
            """1006 W3 세션 명령 처리(감사 H3–H5·L1) — 판단은 W3Supervisor.handle, 실행은 종전 경로."""
            now = time.time()
            w3c = self.core.runner._w3_clock if self.core.w3 else None
            _bd = BLEND_S <= 0.0 or (self.blend_t0 is not None and now - self.blend_t0 >= BLEND_S)
            dec = self.w3sup.handle(d, now, active=bool(self.core.active), warmed=bool(self.warmed), blend_done=_bd,
                                    ge_now=bool(w3c.ge) if w3c is not None else False,
                                    want_now=bool(w3c.want) if w3c is not None else False,
                                    **({"cmd_now": tuple(self.core.cmd)} if self.core.w3cmd else {}))
            dec = w3_monitor_gate(dec, self.core.w3mon)   # 1007 감시(--w3-monitor 일 때만): 켜기 거부만 더한다
            act = dec["action"]
            if act in ("estop", "stop"):
                self._on_cmd_base({"cmd": act})        # 종전 경로(estop = STOP_ALL 즉시·비활성 / stop = ge 0)
            elif act == "set_want":
                if self.core.w3:
                    _ev = {"timestamp": now, "event": "w3_step", "enable": bool(dec["want"]), "by": "cmd",
                           "req_id": d.get("req_id"), "seq": d.get("seq")}
                    if self.core.w3cmd:                # 1007 명령 채널: 켜기 = 명령과 ge 를 같은 틱에(시뮬 리샘플), 끄기 = 명령 0
                        self.core.cmd = tuple(dec["cmd"]) if dec["want"] else (0.0, 0.0, 0.0)
                        if dec["want"]:
                            self.core.w3hl_on = bool(dec["hl"])
                        _ev.update(cmd=list(self.core.cmd), heading_loop=bool(self.core.w3hl_on) and self.core.w3hl is not None)
                    self.core.set_gait_enable(dec["want"])
                    self._bb_write(_ev)
                else:
                    dec = dict(dec, accepted=False, why="W3 러너 아님")
            elif act == "set_cmd":                     # 1007 걷는 중 명령 변경(위상 유지 — 시뮬 STEP→STEP 리샘플과 같음)
                self.core.cmd = tuple(dec["cmd"])
                self.core.w3hl_on = bool(dec["hl"])
                self._bb_write({"timestamp": now, "event": "w3_cmd", "cmd": list(self.core.cmd),
                                "heading_loop": bool(self.core.w3hl_on) and self.core.w3hl is not None,
                                "req_id": d.get("req_id"), "seq": d.get("seq")})
            elif act == "launch":
                n0 = self._launches
                self._on_cmd_base(d)
                if self._launches == n0:
                    g = self.grav_last
                    dec = dict(dec, accepted=False,
                               why=f"stand 시작 거부(종전 게이트) — {g[1] if g and not g[0] else 'hang_arm/touchdown 조건 — 로그 참조'}")
            self._w3_ack(d, dec["accepted"], dec["why"], act)

        def _on_cmd_base(self, d):
            c = d.get("cmd", "")
            if c == "estop":
                self._stop_all("estop")
                self.core.active = False
                return
            if c == "stop":
                self.core.cmd = (0.0, 0.0, 0.0)   # walk 정책 능동 서기 = 정지
                if self.core.w3:                  # 1006 W3: 정지 = gait_enable 0(반주기 경계에서 서기로)
                    self.core.set_gait_enable(False)
                    self._bb_write({"timestamp": time.time(), "event": "w3_step", "enable": False, "by": "stop"})
                return
            if c == "step":
                # 1006 W3 실험: gait_enable 만 바꾼다(정책·러너 리셋 없음). 플래그 꺼짐/다른 키면 무시(종전에도 모르는 명령은 무시).
                if not self.core.w3:
                    self.get_logger().warn(f"step 명령은 --experimental-w3 + {_W3.W3_CKPT_KEY} 실행 중에만 — 무시: {d}")
                    return
                try:
                    want = w3_parse_step(d)
                except (ValueError, TypeError) as e:
                    self.get_logger().warn(f"step 거부: {e}")
                    return
                if want:
                    _now = time.time()
                    _bd = BLEND_S <= 0.0 or (self.blend_t0 is not None and _now - self.blend_t0 >= BLEND_S)
                    ok, why = _W3.step_enable_allowed(self.core.active, self.warmed, _bd)
                    if not ok:
                        self.get_logger().warn(f"step 켜기 거부: {why}")
                        self._bb_write({"timestamp": _now, "event": "w3_step_rejected", "why": why})
                        return
                self.core.set_gait_enable(want)
                self._bb_write({"timestamp": time.time(), "event": "w3_step", "enable": bool(want), "by": "cmd"})
                self.get_logger().info(f"W3 gait_enable 요청 {int(want)} (켜기=다음 틱 위상 0·왼발 스윙, 끄기=반주기 경계)")
                return
            if c == "turn":
                # 1006 HTURN2 실험: 목표 요 변화만 바꾼다(정책·러너 리셋 없음). 플래그 꺼짐/다른 키면 무시(종전에도 모르는 명령은 무시).
                if not (self.core.active and self.core.hturn2):
                    self.get_logger().warn(f"turn 명령은 --experimental-hturn2 + {_H2.HTURN2_CKPT_KEY} 실행 중에만 — 무시: {d}")
                    return
                try:
                    self.core.set_heading_target(hturn2_target_rad(d.get("heading_target_deg", 0.0)))
                except (ValueError, TypeError) as e:
                    self.get_logger().warn(f"turn 거부: {e}")
                    return
                self._bb_write({"timestamp": time.time(), "event": "hturn2_turn",
                                "heading_target_deg": round(math.degrees(self.core.hh_target), 2)})
                self.get_logger().info(f"HTURN2 목표 요 변화 {math.degrees(self.core.hh_target):+.1f}° (시작 요 기준)")
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
                    # 0803 부활: stand 전용 정책 가족 (능동 균형 서기). 검증 사다리의
                    # 1단: march 전에 stand 제로샷 확인. 가족 밖이면 STAND_DEFAULT + 경고.
                    ck = args.checkpoint if args.checkpoint in stand_family \
                        else STAND_DEFAULT
                    if ck != args.checkpoint:
                        self.get_logger().warn(
                            f"--checkpoint {args.checkpoint} 는 stand 가족이 아님 → {ck} 로 대체 실행")
                elif c == "march":
                    ck = args.checkpoint if args.checkpoint in march_family \
                        else "march_v5b"
                else:
                    ck = args.checkpoint \
                        if args.checkpoint in (
                            set(CHECKPOINTS) - march_family - stand_family) \
                        else "drv2"
                _h2_target = None
                if c == "walk" and ck in HTURN2_CKPTS:   # 1006 HTURN2 실험(플래그 켜짐 + 키)에서만
                    try:
                        cmd, _h2_target = hturn2_validate_walk(d)
                    except (ValueError, TypeError) as e:
                        self.get_logger().error(f"[거부] {e} — 무시: {d}")
                        return
                # 1005 착지 자동 인수(touchdown_start): stand 전용, 행잉 ARM 과 동시 금지(공중 정책 경로와 섞지 않는다).
                _td = bool(d.get("touchdown_start", False))
                if ck in W3_CKPTS and bool(d.get("hang_arm", False)):   # 1006 W3: 학습 분포는 지면 시작뿐
                    self.get_logger().error(f"[거부] W3 키는 hang_arm 불가(지면 시작 학습) — 무시: {d}")
                    return
                if _td and (c != "stand" or bool(d.get("hang_arm", False))):
                    self.get_logger().error(
                        f"[거부] touchdown_start 는 stand 전용이며 hang_arm 과 함께 쓸 수 없음 — 무시: {d}")
                    return
                # 중력 정합 게이트 (0905 감사 A4) — 장착보정 미적용 IMU 관측(±1,0,0)이 그대로
                # 정책에 들어가는 경로 차단. 드라이런도 같은 게이트(관측 검증이 목적).
                g_ok, g_why = gravity_gate(self.imu, self.imu_ts, time.time())
                self.grav_last = [bool(g_ok), g_why]
                if not g_ok and not bool(d.get("force_gravity", False)):
                    self.get_logger().error(
                        f"[거부] RL {c} 시작 불가 — {g_why}. IMU 장착보정·착지 자세 확인. "
                        "강제 시작: {\"cmd\": ..., \"force_gravity\": true}")
                    return
                self.get_logger().info(g_why)
                # 라이브 (재)시작 = 가드·미분 상태 전부 리셋 (적대리뷰 A1·A2:
                # 종전엔 osc_tripped 래치로 재개 시 가드 영구 사망 + prev_qpos
                # 동결로 복귀 첫 틱 qvel 폭주)
                # 0908 행잉 ARM: stand + hang_arm:true 를 명시한 경우에만 접지 전 무접촉 허용
                _hang = (c == "stand" and bool(d.get("hang_arm", False)))
                self._reset_chain_state(_hang, time.time())
                if _hang:
                    self.get_logger().warn(
                        f"행잉 ARM — 접지 전 무접촉 허용(최대 {HANG_MAX_S:.0f}s), 양발 접촉 {CONTACT_CONFIRM_S}s 후 "
                        "기존 리프트 가드 복원")
                # 1005: touchdown_start 면 착지 확인 전까지 정책 미구동(홀드 유지) — 없으면 None(종전 경로 그대로)
                self.td_gate = TouchdownGate(t0=time.time()) if _td else None
                self.td_hold_rel = None
                if _td:
                    self.get_logger().warn(
                        f"착지 자동 인수 — 착지 전 정책 미구동(수평 홀드 유지), 양발 접촉 {TD_CONFIRM_S}s·정착 뒤 "
                        f"정책 시작(최대 대기 {TD_WAIT_MAX_S:.0f}s)")
                self.core.reset(checkpoint=ck)
                self.core.cmd = cmd
                # stand 계열은 heading_hold 강제 OFF (0803 발진 분석: stand_v3
                # 학습 명령은 전 에피소드 (0,0,0) — HeadingHold의 wz≠0 주입은
                # 학습에 없던 입력 채널이자 요 요동과 상관된 추가 피드백 루프)
                if ck in stand_family:
                    self.core.heading_hold = False
                else:
                    self.core.heading_hold = bool(d.get("heading_hold", True))
                if _h2_target is not None:   # 1006 HTURN2: 목표 각(시작 요 기준) — reset() 뒤에 설정
                    self.core.set_heading_target(_h2_target)
                self.core.active = True
                self.stopped_for_stale = False
                # 인계 워밍업 시작 — stage8 status 스트림이 붙을 때까지 정책 구동 보류(홀드-프라임만).
                # 실측: 3회 정상기동은 tick-1 이 신선했고, 금일 2회 실패는 tick-1 이 유휴 status 갭에
                # 걸려 sensor_stale 오정지. 워밍업이 이 타이밍 레이스를 없앤다(정상상태 게이트는 불변).
                self.warmed = False
                self.warmup_until = time.time() + STAND_WARMUP_MAX_S
                self._launches += 1      # 시작 횟수(1006 W3 세션 응답이 '실제 시작'을 판정 — 종전 경로 동작 무관)
                self.get_logger().info(f"RL {c} 시작 ckpt={ck} cmd={self.core.cmd} (인계 워밍업 최대 {STAND_WARMUP_MAX_S:.0f}s)")

        def _reset_chain_state(self, hang, t0):
            """정책 (재)시작 시 가드·미분·필터·블렌딩 상태 리셋 — stand 명령과 착지 자동 인수 시작이 같은 함수를 쓴다.
            (종전 _on_cmd 의 리셋 블록을 그대로 옮긴 것; 순서·내용 불변)"""
            self.osc = OscGuard(JOINT_ORDER)
            self.osc_tripped = None
            self.prev_qpos = None
            self.blend_t0 = None
            self.blend_from = None
            self.nocontact_t0 = None
            self.hang_guard = HangGuard(hang_mode=hang, t0=t0)
            self.tgt_filt = None
            self.roll_comp.reset()
            self.pitch_comp.reset()
            self.osc_tgt = OscGuard(JOINT_ORDER)
            self.prev_tgt = None

        def _td_hold(self, now, qpos, contact2, gyro, quat):
            """착지 자동 인수 대기 틱: 정책을 계산하지 않고, 무장 시점에 stage8 이 스트리밍하던 상대 명령(수평 홀드)을 그대로 보낸다.
            래치는 첫 대기 틱 1회 — 스트리밍 명령 에코를 매 틱 다시 읽지 않으므로 표류가 없다."""
            if self.td_hold_rel is None:
                if all(j in self.stage8_cmd_rel for j in JOINT_ORDER):
                    self.td_hold_rel = {j: float(self.stage8_cmd_rel[j]) for j in JOINT_ORDER}
                    src = "stage8 스트리밍 명령"
                elif all(j in self.baseline for j in JOINT_ORDER) and all(j in self.joint_deg for j in JOINT_ORDER):
                    self.td_hold_rel = {j: self.joint_deg[j] - self.baseline[j] for j in JOINT_ORDER}
                    src = "실측 자세(폴백)"
                else:
                    src = None
                if src:
                    self.get_logger().info(f"착지 대기 홀드 목표 래치 — {src}")
                    self._bb_write({"timestamp": now, "event": "touchdown_hold_latch", "src": src,
                                    "rel_deg": [round(self.td_hold_rel[j], 4) for j in JOINT_ORDER]})
                    # 홀드 자세 vs 정책 기본자세 — 블렌딩이 큰 자세 이동 명령이 되는 경로 차단(기준점이 있어야 판정 가능)
                    if all(j in self.baseline for j in JOINT_ORDER):
                        pose = ckpt_default_pose(self.core.ckpt_name)
                        dev = [(abs(math.degrees(self.frames.motor_deg_to_joint_rad(j, self.baseline[j] + self.td_hold_rel[j]) - float(pose[k]))), j)
                               for k, j in enumerate(JOINT_ORDER)]
                        worst, wj = max(dev)
                        if not math.isfinite(worst) or worst > TD_POSE_TOL_DEG:
                            why = (f"touchdown_pose_mismatch: 홀드 자세가 정책 기본자세와 {wj} {worst:.1f}° 차이"
                                   f"(허용 {TD_POSE_TOL_DEG:g}°) — 수평 자세 적용 확인. 정책 미시작")
                            self._bb_write({"timestamp": now, "event": "touchdown_pose_mismatch", "why": why})
                            self._stop_all(why)
                            self.core.active = False
                            self.get_logger().error(why)
                            return
            rel = self.td_hold_rel
            tgt_motor = ({j: self.baseline[j] + rel[j] for j in JOINT_ORDER}
                         if rel is not None and all(j in self.baseline for j in JOINT_ORDER) else None)
            dbg = {"timestamp": now, "live": self.live, "armed": self.armed,
                   "checkpoint": self.core.ckpt_name, "cmd": list(self.core.cmd),
                   "targets_deg_motor": tgt_motor,
                   "targets_deg_joint": ({j: math.degrees(self.frames.motor_deg_to_joint_rad(j, v)) for j, v in tgt_motor.items()}
                                         if tgt_motor else None),
                   "qpos_rad": [round(v, 5) for v in qpos],
                   "contact": contact2, "hang_state": None, "blend_alpha": 0.0,
                   "foot_n": [round(float(self.foot[0]), 2), round(float(self.foot[1]), 2)],
                   "quat_wxyz": [round(float(v), 5) for v in quat],
                   "gyro_rad_s": [round(float(v), 4) for v in gyro],
                   "rp_deg": _roll_pitch_deg(quat),
                   "status_age_s": round(now - self.status_ts, 3) if self.status_ts > 0.0 else None,
                   "obs": None, "raw_action": None, "c5": None,
                   "start_gate": self.td_gate.snapshot(now),
                   "osc_tripped": self.osc_tripped, "stopped_for_stale": self.stopped_for_stale,
                   "stop_reason": self.stop_reason, "grav_gate": self.grav_last,
                   "foot_raw_sum": self.foot_raw_sum, "last_reject_reason": self.reject_reason}
            if self.guard_mode != "safe":
                dbg["guard_mode"] = self.guard_mode
            self.pub_dbg.publish(String(data=json.dumps(dbg)))
            self._bb_write(dbg)
            if self.live and self.armed and rel is not None:
                self.pub_cmd.publish(String(data=self._targets_msg(dict(rel))))

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
            # 인계 워밍업: stand 직후 stage8 status 스트림이 아직 안 붙어 joint_ts 가 낡을 수 있다.
            # 정책을 구동하지 않고 홀드-프라임(현 자세 상대0)만 보내 stage8 를 깨우며 신선 피드백을 기다린다.
            proceed, wact = warmup_gate(fresh, self.warmed, now, self.warmup_until)
            if not proceed:
                self.prev_qpos = None
                self.roll_comp.reset()   # 정책 미구동(홀드-프라임) → 보정기 필터 0
                self.pitch_comp.reset()
                if wact == "prime":
                    self._send_hold_prime()
                    if now - getattr(self, "_warm_log_t", 0.0) >= 0.5:   # 판별 측정: 무엇이 낡았나 (H1 관절 한 분기 vs H3 status 전체)
                        self._warm_log_t = now
                        self.get_logger().info("[인계 워밍업 대기] " + self._warmup_diag(now))
                    return
                # timeout — 워밍업 창 내 관절 피드백 미확보
                self._stop_all("sensor_stale_warmup_timeout")
                self.core.active = False
                self.get_logger().error(
                    f"인계 워밍업 {STAND_WARMUP_MAX_S:.0f}s 내 관절 피드백 미확보 → STOP_ALL + 정책 비활성. "
                    f"{self._warmup_diag(now)}. 낡은 관절이 있으면 그 CAN 분기, 없으면 status 전달 경로를 점검 후 재명령.")
                return
            if not self.warmed:   # 방금 신선해져 워밍업 완료
                self.warmed = True
                self.get_logger().info("인계 워밍업 완료 — 관절 피드백 신선, 정책 구동 시작")
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
            if getattr(self, "foot_contact", None) is not None:
                contact2 = [1.0 if self.foot_contact[0] else 0.0, 1.0 if self.foot_contact[1] else 0.0]
            else:
                contact2 = [1.0 if self.foot[0] > CONTACT_FORCE_THRESHOLD_N else 0.0,
                            1.0 if self.foot[1] > CONTACT_FORCE_THRESHOLD_N else 0.0]
            # 1005 착지 자동 인수: 착지 확인 전에는 정책 경로에 들어가지 않는다(러너·감지기·가드 상태 미접촉).
            # 센서 신선도·비유한 게이트(위)는 대기 중에도 그대로 적용된다(stale → STOP_ALL = 같은 홀드 동결).
            if self.td_gate is not None and not self.td_gate.started:
                _g_ok = gravity_gate(self.imu, self.imu_ts, now)[0]
                _act, _why = self.td_gate.update(now, contact2, gyro, _g_ok)
                for _st, _t in self.td_gate.pop_transitions():
                    self._bb_write({"timestamp": _t, "event": "touchdown_gate", "state": _st})
                    self.get_logger().info(f"착지 자동 인수 상태 → {_st}")
                if _act == "timeout":
                    # 정책이 아직 시작하지 않은 대기 상한 — guard_mode 와 무관하게 정지(동결 = 같은 홀드, 토크 유지).
                    # 그 뒤 착지해도 정책이 갑자기 시작되지 않게 비활성화한다.
                    self._bb_write({"timestamp": now, "event": "touchdown_timeout", "why": _why})
                    self._stop_all(_why)
                    self.core.active = False
                    self.get_logger().error(f"{_why} — STOP_ALL(홀드 동결) + 정책 비활성. 재무장은 회수 후 새 시험")
                    return
                if _act == "hold":
                    self._td_hold(now, qpos, contact2, gyro, quat)
                    return
                # _act == "start": 종전 stand 시작과 같은 초기화를 **지금** 수행 — 러너 reset(이력 백필·직전 액션 0·
                # EMA 첫 값·클록 0·C5 감지기 기준 미설정 → 착지 뒤 첫 틱부터 캡처), 가드·LPF·블렌딩 재시작(블렌딩 시작점 =
                # 지금 스트리밍 중인 홀드 명령). hang_mode=False → 종전 지상 리프트 가드 그대로.
                self._reset_chain_state(False, now)
                self.core.reset()
                self.prev_qpos = qpos          # 종전 첫 틱과 같은 qvel 수열(첫 틱 0)
                self.qpos_hold_ticks = [1] * 12
                qvel = [0.0] * 12
                self._bb_write({"timestamp": now, "event": "touchdown_start", "why": _why,
                                "gate": self.td_gate.snapshot(now)})
                self.get_logger().info(f"{_why} — 정책 구동 시작 (ckpt={self.core.ckpt_name})")
            # 1006 W3 세션: 감독기 관측 갱신(정책이 받는 contact2·투영중력과 같은 값) + 지연 동결 순서 확인(감사 H2)
            if self.w3sup is not None:
                self.w3sup.observe(now, contact2, gyro, correct_gravity(projected_gravity(quat)))
                if self._w3_freeze_check(now):
                    return
            # 리프트 가드 — 양발 무접촉 LIFT_STOP_S 지속 시 동결 (발버둥 차단).
            # 0908: HangGuard 상태기로 이관. hang_arm 미명시면 종전과 동일 동작(셀프테스트 고정),
            # hang_arm 명시면 접지 획득 전에만 무접촉 허용(상한 HANG_MAX_S).
            _why = self.hang_guard.update(
                now, contact2, (float(self.foot[0]) + float(self.foot[1])) if self.foot else None)
            for _st, _t in self.hang_guard.pop_transitions():
                self._bb_write({"timestamp": _t, "event": "hang_guard", "state": _st})
                self.get_logger().info(f"행잉 가드 상태 → {_st}")
            if self.live and _why:
                if self.guard_mode == "data":
                    # 데이터 수집 모드: 양발 무접촉(=낙상 중)에도 멈추지 않는다. 그 구간이
                    # 바로 재려는 구간이기 때문이다. 회수는 운영자의 줄·손이 책임진다.
                    self._guard_note(now, "lift_guard", _why)
                elif self.w3sup is not None:
                    if self._w3_guard_trip(now, "lift_guard", _why):
                        return
                else:
                    self._bb_write({"timestamp": now, "event": "lift_guard", "why": _why})
                    self._stop_all(_why)
                    self.core.active = False
                    self.get_logger().error(f"리프트 가드 — {_why}. 동결 홀드. 줄·발 확인!")
                    return
            # 발진 가드 — 피드백 기반이라 정책 출력과 무관하게 몸의 발진을 감지.
            # 트리거 시 STOP_ALL(동결 홀드) 후 비활성: 이후 절차 = 줄 재인장 →
            # release_all.py (0802 프로토콜)
            if self.live and self.osc_tripped is None:
                why = self.osc.update(now, qpos, qvel)
                if why and self.guard_mode == "data" and not why.startswith("과속"):
                    # 발진 자체는 허용(운영자 판단). 과속 폭주(OSC_VEL_HARD)는 발진이 아니라
                    # 기구 보호 한계라 데이터 모드에서도 그대로 정지한다.
                    self._guard_note(now, "osc_guard", why)
                    why = None
                if why and self.w3sup is not None and not why.startswith("과속"):
                    # 1006 W3 세션: 지연 동결(감사 H2). osc_tripped 를 걸지 않아 대기 중에도 과속(하드) 감시는 계속된다.
                    if self._w3_guard_trip(now, "osc_guard", why):
                        return
                    why = None
                if why:
                    self.osc_tripped = why
                    self._bb_write({"timestamp": now, "event": "osc_guard",
                                    "reason": why})
                    self._stop_all(f"osc_guard: {why}")
                    self.core.active = False
                    self.get_logger().error(
                        f"발진 가드 트리거 — {why}. 줄 재인장 후 release_all 실행!")
                    return
            if getattr(self.core.runner, "needs_knee_tau", False):      # 1007 odic(기본 꺼짐): 무릎 τ 관절 프레임·신선도 검사
                _tk = knee_tau_joint(self.tau_est, (now - self.status_ts) if self.status_ts > 0.0 else None, self.frames.dir_by_joint)
                targets_rad = self.core.tick(gyro, quat, qpos, qvel, contact2, tau_knee_nm=_tk)
            else:
                targets_rad = self.core.tick(gyro, quat, qpos, qvel, contact2)
            while self.core.w3hl_events:   # 1007 W3 헤딩 고리 고장 → 코어가 걷기를 멈췄다(서기로) — 기록·경고
                _e = self.core.w3hl_events.pop(0)
                self._bb_write(dict(_e, timestamp=now))
                _ev = _e.get("event")
                if _ev == "w3_hl_fault":
                    self.get_logger().error(f"W3 헤딩 고리 고장 — {_e['why']}. 걷기 정지(반주기 경계에서 서기), 재개는 새 step(seq)")
                elif _ev == "w3_mon_stop":     # 1007 감시 act: 걷기 정지 + 정지 래치(rearm 전 켜기 거부)
                    if self.w3sup is not None:
                        w3_monitor_latch(self.w3sup, f"monitor {_e['kind']}: {_e['why']}")
                    self.get_logger().error(f"W3 감시 걷기 정지 — {_e['why']}. 반주기 경계에서 서기, 정지 래치(재개 = rearm + 새 step)")
                else:                           # w3_mon(판정 기록)·w3_hl_banned
                    self.get_logger().warn(f"W3 감시 {_e.get('kind', _ev)} — {_e.get('why')} [{_e.get('action', '')}]")
            if self.core.w3mon is not None:   # 1007 감시: 발 센서 나이(접촉 신선도) — 정책 입력은 건드리지 않는다
                _s8 = (self.s8_cmd_timing or {}).get("p50_age_ms_1s") if isinstance(self.s8_cmd_timing, dict) else None
                self.core.w3mon.health(foot_age_s=now - self.foot_ts,
                                       fb_age_s=(now - self.status_ts) if self.status_ts > 0.0 else None,
                                       s8_p50_ms=(float(_s8) if isinstance(_s8, (int, float)) else None))
            if C5_OVERTIME is not None and isinstance(self.core.runner, C5Runner):   # 1007 회복 모드 체류 상한(기본 꺼짐)
                _m = int(self.core.runner.last_mode)
                _ov = C5_OVERTIME.update(_m)
                if _ov is not None:
                    self._bb_write(dict(_ov, timestamp=now))
                    self.get_logger().error(f"C5 회복 모드 체류 상한 — {_ov['why']} [{_ov['action']}]")
                    self._c5_ov_pending = _ov["action"] == "freeze"
                if _m == 0:
                    self._c5_ov_pending = False
                if getattr(self, "_c5_ov_pending", False) and contact2[0] > 0.0 and contact2[1] > 0.0:
                    self._c5_ov_pending = False
                    self._stop_all("c5_overtime")      # 동결 = 같은 홀드(토크 유지). 양발 접촉 틱에만 — 한발 동결 금지
                    self.core.active = False
                    self.get_logger().error("C5 회복 체류 상한 → 양발 접촉 확인, STOP_ALL(동결 유지). 줄 인장 → 스택 종료 → release_all 순서")
                    return
            if self.core.ckpt_name.startswith("stand") or self.core.hturn2:   # 1006 HTURN2: 평가 조건(LPF β0.27) 그대로
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
                if why_t and self.guard_mode == "data" and not why_t.startswith("과속"):
                    self._guard_note(now, "osc_guard_tgt", why_t)
                    why_t = None
                if why_t and self.w3sup is not None and not why_t.startswith("과속"):
                    if self._w3_guard_trip(now, "osc_guard_tgt", why_t):   # 1006 W3 세션: 지연 동결(감사 H2)
                        return
                    why_t = None
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
            # 느린 기울기 롤 보정기 (맵 키 roll_comp, 기본 OFF — 키가 없으면 이 블록 전체가 no-op).
            # 자리(F4): 타깃 발진 가드 **뒤**, 프레임 변환·온셋 블렌딩 **앞**. 가드는 교정된
            # 정책 체인 출력만 보고(오트립 0 교정 보존), 보정기 출력은 구조적으로 느려
            # (틱당 ≤ slew·|off| ≤ limit·τ ≥ 0.5s) 발진할 수 없다 — 근거는 RollComp docstring.
            # add_to 는 새 리스트를 내므로 LPF 상태(self.tgt_filt)는 오염되지 않는다.
            rc_off, rc_gate = 0.0, False
            pc_off, pc_gate = 0.0, False
            rc_pre_deg = None
            if self.roll_comp.enabled or self.pitch_comp.enabled:
                # F3: 접지 확인 상태(HangGuard 가 PRE_CONTACT 를 벗어남)를 게이트에 AND.
                # 지상 인계(hang_arm 없음)에선 상태가 처음부터 CONTACT_ACQUIRED 라 항상 참이고,
                # 행잉 ARM 에선 '양발 접촉 0.3s 확인' 전까지 보정기가 돌지 않는다.
                rc_landed = self.hang_guard.state != HangGuard.PRE_CONTACT
                rc_pre_deg = [round(math.degrees(float(v)), 4) for v in targets_rad]
                if self.roll_comp.enabled:
                    rc_off, rc_gate = self.roll_comp.update(
                        self.core.last_grav, contact2, self.core.active, rc_landed)
                    targets_rad = self.roll_comp.add_to(targets_rad, rc_off)
                if self.pitch_comp.enabled:
                    pc_off, pc_gate = self.pitch_comp.update(
                        self.core.last_grav, contact2, self.core.active, rc_landed)
                    targets_rad = self.pitch_comp.add_to(targets_rad, pc_off)
            if self.core.w3mon is not None:   # 1007 감시 C: 최종 송출 목표(LPF·보정 뒤, 블렌딩 전 관절 프레임)와 엔코더 → 실효 지연 추정
                self.core.w3mon.track(targets_rad, qpos)
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
                   "hang_state": self.hang_guard.state,
                   "blend_alpha": round(alpha, 3),
                   "foot_n": [round(float(self.foot[0]), 2),
                              round(float(self.foot[1]), 2)],
                   # 자세 사후분석용 (0802 첫 march: 줄 걸림 vs 자체 기울어짐
                   # 구분에 IMU 부재가 병목이었음)
                   "quat_wxyz": [round(float(v), 5) for v in quat],
                   "gyro_rad_s": [round(float(v), 4) for v in gyro],
                   # 0905 신설(기록 전용): stage8 토크 추정[Nm]·온도 원값, JOINT_ORDER 순,
                   # 미수신 관절은 None — 시뮬 토크 피크/포화 지표와 동일 축 대조용
                   "tau_est_nm": [self.tau_est.get(j) for j in JOINT_ORDER],
                   "temp_raw": [self.temp_raw.get(j) for j in JOINT_ORDER],
                   # stage8 status 신선도 [s] — analyze_fall 이 0.1s 초과 틱의 tau_est 를 동결값으로 제외
                   "status_age_s": round(now - self.status_ts, 3) if self.status_ts > 0.0 else None,
                   # 0930: stage8 가 잰 '발행 → 처리' 명령 지연·누락 (필드 없으면 옛 stage8)
                   "s8_cmd": ({k: self.s8_cmd_timing.get(k) for k in ("last_age_ms", "p50_age_ms_1s", "max_age_ms_1s",
                                                                     "count_1s", "stale", "seq_gaps", "seq_regress", "src_changes")}
                              if self.s8_cmd_timing else None),
                   # stage8 soft 클램프 누적(총·관절별)·마지막 거부 사유 — 클램프/거부가 낙상과 겹치는지 사후분석 (감사 B7)
                   "soft_clamp_total": self.clamp_total,
                   "soft_clamp_counts": [self.clamp_counts.get(j) for j in JOINT_ORDER],
                   # A1 기록 보강(0910, 기록 전용 — 승윤님 A-stage 요구 필드): 정책 관측 벡터(러너가 이번 틱에 조립한 245/320 그대로),
                   # raw 액션(아핀·슬루·블렌딩 전), IMU 롤/피치, 가드·정지 상태, 어댑터 원카운트 합. 패드 8채널 원값은 cop_from_pads.py --live --save 로 별도 기록.
                   "obs": ([round(float(v), 4) for v in self.core.runner.last_obs]
                           if getattr(self.core.runner, "last_obs", None) is not None else None),
                   "raw_action": [round(float(v), 4) for v in self.core.runner.last_action],
                   # 1004 C5: 모드(0 서기/1 회복)·이번 틱 전환(on/off)·켜짐 누계·감지기 기준 자세·두 정책 원시 출력(전환 매끄러움 판정용)
                   "c5": (self.core.runner.blackbox() if isinstance(self.core.runner, C5Runner) else None),
                   # 1001: 관측 joint_vel_ema 모드(v2 = 실제 속도 EMA / ema0 = 0, 학습 계약 복원)와 실제 속도 EMA RMS(두 모드 모두 계산)
                   "jvel_ema_mode": getattr(self.core.runner, "jvel_ema_mode", "v2"),
                   "obs_filter": [getattr(self.core.runner, "act_obs_ema_beta", None), getattr(self.core.runner, "jvel_obs_ema_beta", None)],   # 1003 관측 필터 계약(act β, jvel β)
                   "jvel_ema_rms": (round(math.sqrt(sum(float(v) * float(v) for v in self.core.runner.last_jvel_ema)
                                                    / len(self.core.runner.last_jvel_ema)), 4)
                                    if getattr(self.core.runner, "last_jvel_ema", None) is not None else None),
                   "rp_deg": _roll_pitch_deg(quat),
                   "imu_mount_bias_deg": [IMU_MOUNT_BIAS_DEG[0], IMU_MOUNT_BIAS_DEG[1]],
                   "osc_tripped": self.osc_tripped,
                   "stopped_for_stale": self.stopped_for_stale,
                   "stop_reason": self.stop_reason,
                   "grav_gate": self.grav_last,
                   "foot_raw_sum": self.foot_raw_sum,
                   "last_reject_reason": self.reject_reason}
            if self.roll_comp.enabled:   # OFF(키 없음)면 기록 스키마도 종전과 동일
                # targets_deg_joint/motor 는 오프셋 포함 = 실제 송출 명령. 아래는 분리 분석용.
                dbg["roll_comp_deg"] = round(math.degrees(rc_off), 4)
                dbg["roll_comp_gate"] = bool(rc_gate)                 # 실효(디바운스 포함)
                dbg["roll_comp_raw_gate"] = bool(self.roll_comp.raw_gate)
                dbg["roll_comp_joints"] = self.roll_comp.cfg["joints"]
                dbg["targets_deg_joint_pre_rc"] = rc_pre_deg          # JOINT_ORDER 순 [deg]
            if self.pitch_comp.enabled:
                dbg["pitch_comp_deg"] = round(math.degrees(pc_off), 4)
                dbg["pitch_comp_gate"] = bool(pc_gate)
                dbg["pitch_comp_raw_gate"] = bool(self.pitch_comp.raw_gate)
                dbg["pitch_comp_K"] = self.pitch_comp.cfg["K"]
                if rc_pre_deg is not None:
                    dbg["targets_deg_joint_pre_rc"] = rc_pre_deg
            if self.guard_mode != "safe":
                dbg["guard_mode"] = self.guard_mode
            if self.td_gate is not None:   # 1005: touchdown_start 경로에서만 필드 추가(종전 스키마 불변)
                dbg["start_gate"] = self.td_gate.snapshot(now)
            if self.core.hturn2:           # 1006 HTURN2 실험 경로에서만 필드 추가(종전 스키마 불변)
                _r = self.core.runner
                dbg["hturn2"] = {"u": float(_r._h2_clock.u), "f": float(_r._h2_clock.f),
                                 "ff_hip_r": (None if _r.last_ff is None else [round(v, 5) for v in _r.last_ff]),
                                 "hh_target_deg": round(math.degrees(self.core.hh_target), 2),
                                 "hh_ref": (None if not self.core.hh_ref_set else round(self.core.hh.ref, 5))}
            if self.core.w3:               # 1006 W3 실험 경로에서만 필드 추가(종전 스키마 불변)
                dbg["w3"] = self.core.runner._w3_clock.snapshot()
            if self.core.w3cmd:            # 1007 W3 명령 채널 키에서만: 리드 발·헤딩 고리 상태(정책이 받은 wz 는 obs[32])
                dbg["w3_cmd"] = {"lead": self.core.runner._w3_clock.last_lead, "hl_on": bool(self.core.w3hl_on),
                                 "hl": (self.core.w3hl.snapshot() if self.core.w3hl is not None else None)}
            if self.w3sup is not None:     # 1006 W3 세션 감독기 상태(정지 래치·동결 대기·켜기 판정 입력)
                dbg["w3_sup"] = self.w3sup.snapshot()
            if self.core.w3mon is not None:   # 1007 감시(--w3-monitor 일 때만 필드 추가 — 종전 스키마 불변)
                dbg["w3_mon"] = self.core.w3mon.snapshot()
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
                self.pub_cmd.publish(String(data=self._targets_msg(targets)))

        def _send_hold_prime(self):
            """인계 워밍업 전용: **현재 실측 자세**를 유지하는 명령(rel = joint_deg − baseline)을 보내
            stage8 명령 경로를 살려 두고(모터 워치독·명령 트리거 status) 신선한 피드백을 기다린다.

            baseline 상대 0 이 아니라 현 자세를 잡는 이유(호환 검수 §2.7): STOP_ALL 로 동결된 뒤 수동
            재발화되면 상대 0 은 동결 목표를 baseline 으로 교체해 슬루 상한(4°/틱=200°/s)으로 끌고 간다.
            현 자세 유지는 어느 상태에서든 무동작이다. 12관절 모두 실측·baseline 이 있어야 보낸다.
            ※ 이 프라임은 tick-1 stale 의 원인을 고치지 않는다(원인은 stage8 쪽 status 전달 갭 — 가설).
              워밍업은 실측된 전달 갭(최대 ~3 s)에 대한 유예이며, 4 s 안에 못 붙으면 정지가 맞다."""
            tg = {}
            if all(j in self.stage8_cmd_rel for j in JOINT_ORDER):
                # 호환검수 R6: stage8 가 지금 스트리밍 중인 상대 명령을 그대로 = 어떤 상태(ARM 직후·STOP_ALL 동결)에서도 '무변화'
                tg = {j: float(self.stage8_cmd_rel[j]) for j in JOINT_ORDER}
            else:
                if not self.baseline or len(self.joint_deg) < 12:
                    return
                for j in JOINT_ORDER:
                    b = self.baseline.get(j); q = self.joint_deg.get(j)
                    if b is None or q is None:
                        return
                    tg[j] = q - b            # 폴백: 실측 자세 유지
            self.pub_cmd.publish(String(data=self._targets_msg(tg)))

        def _targets_msg(self, targets):
            """SET_JOINT_TARGETS + 순번·발행 시각·출처 (0930). stage8 은 targets 만 쓰고 나머지는 지연 기록용."""
            self._cmd_seq += 1
            return json.dumps({"command": "SET_JOINT_TARGETS", "targets": targets,
                               "seq": self._cmd_seq, "sent_ts": time.time(), "src": self._cmd_src})

        def _warmup_diag(self, now):
            stale = [j for j in JOINT_ORDER
                     if j not in self.joint_deg or self.fb_age.get(j, 9e9) >= STATUS_FEEDBACK_FRESH_SEC]
            return (f"joint_ts {now - self.joint_ts:.2f}s status_age {now - self.status_ts:.2f}s "
                    f"imu {now - self.imu_ts:.2f}s foot {now - self.foot_ts:.2f}s 관절 {len(self.joint_deg)}/12 "
                    f"낡은관절 {stale if stale else '없음'}")

        def _bb_write(self, obj):
            """블랙박스 기록 — 제어 틱에서는 직렬화만 하고 큐에 넣는다(디스크 대기 없음).

            실제 쓰기는 bb_writer 스레드. 큐가 가득 차면(디스크가 오래 막힘) 기록을 버리고 개수만 센다 —
            기록을 위해 제어를 멈추지 않는다. 버린 개수는 다음 기록에 bb_dropped 로 남는다.
            (적대리뷰 A3 원칙 유지: 기록 실패가 안전동작을 막지 않는다)
            """
            if not self.bb:
                return
            try:
                if self._bb_dropped:
                    obj = dict(obj); obj["bb_dropped"] = self._bb_dropped
                self._bbq.put_nowait(json.dumps(obj))
                if self._bb_dropped:
                    self._bb_dropped = 0
            except Exception:  # noqa: BLE001 — queue.Full 포함
                self._bb_dropped += 1

        def _bb_writer(self):
            import queue as _queue
            while True:
                try:
                    line = self._bbq.get(timeout=0.5)
                except _queue.Empty:
                    try:
                        if self.bb:
                            self.bb.flush()
                    except OSError:
                        pass
                    continue
                if line is None:
                    break
                try:
                    self.bb.write(line + "\n")
                    if self._bbq.empty():
                        self.bb.flush()
                except (OSError, AttributeError, ValueError):
                    try:
                        self.bb.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self.bb = None
                    self.get_logger().error("블랙박스 기록 실패 — 기록 중단 (제어는 계속)")
                    return

        def _bb_close(self):
            """종료 시 큐에 남은 기록을 최대 3 s 안에 비우고 닫는다."""
            t = getattr(self, "_bbt", None)
            if t is not None and t.is_alive():
                try:
                    self._bbq.put(None, timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
                t.join(timeout=3.0)
            if self.bb:
                try:
                    self.bb.close()
                except OSError:
                    pass
                self.bb = None

        def _guard_note(self, now, event, why):
            """guard_mode=data 에서 가드가 잡았지만 멈추지 않을 때의 기록. 사유가 바뀌거나
            1초가 지났을 때만 쓴다 — 매 틱 기록하면 블랙박스가 가드 로그로 덮인다."""
            prev = self._gnote.get(event)
            if prev is not None and prev[0] == why and (now - prev[1]) < 1.0:
                return
            self._gnote[event] = (why, now)
            self._bb_write({"timestamp": now, "event": event + "_noted",
                            "why": why, "guard_mode": "data", "stopped": False})
            self.get_logger().warn(f"[data] {event} 감지 — {why} (정지하지 않음)")

        def _w3_guard_trip(self, now, event, why):
            """1006 W3 세션 가드 트립(비하드, guard_mode safe) — 즉시 동결 대신 ge 0 → 반주기 래치 해제 → 양발 지지·정착 → 동결.
            반환 True = 이번 틱에 동결했다(호출자 return), False = 대기 중(정책 계속 — 서기로 내려가는 중)."""
            if self.w3sup.begin_freeze(now, f"{event}: {why}"):
                if self.core.w3:
                    if self.core.w3cmd:            # 1007 명령 채널: 정지 = 명령 0 + ge 0 (시뮬 정지 계약)
                        self.core.cmd = (0.0, 0.0, 0.0)
                    self.core.set_gait_enable(False)
                self._bb_write({"timestamp": now, "event": event, "why": why, "w3_deferred": True})
                self._bb_write({"timestamp": now, "event": "w3_step", "enable": False, "by": event})
                self.get_logger().error(f"{event} — {why}. W3 지연 동결 시작(ge 0 → 반주기 래치 해제 → 양발 지지 확인 → 동결)")
            return self._w3_freeze_check(now)

        def _w3_freeze_check(self, now):
            """지연 동결 조건이 차면 STOP_ALL + 비활성 → True."""
            w3c = self.core.runner._w3_clock if self.core.w3 else None
            due = self.w3sup.freeze_due(now, bool(w3c.ge) if w3c is not None else False)
            if due is None:
                return False
            kind, msg = due
            why = f"{self.w3sup.freeze_why} → W3 지연 동결({kind}): {msg}"
            self._bb_write({"timestamp": now, "event": "w3_freeze", "kind": kind, "why": why})
            self._stop_all(why)
            self.core.active = False
            (self.get_logger().error if kind == "timeout" else self.get_logger().warn)(
                f"{why}. 동결 홀드 — 줄 재인장 후 회수 절차")
            return True

        def _stop_all(self, reason):
            if self.w3sup is not None:      # 1006 W3 세션: 어떤 경로의 STOP_ALL 이든 지연 대기 해제 + 정지 래치
                self.w3sup.on_frozen(reason)
            self.stop_reason = str(reason)
            self.get_logger().warning(f"STOP_ALL ({reason})")
            # 안전동작 먼저, 기록은 다음 (적대리뷰 A3: 순서 역전 금지)
            if self.live:
                self.pub_cmd.publish(String(
                    data=json.dumps({"command": "STOP_ALL"})))
            self._bb_write({"timestamp": time.time(), "event": "stop_all",
                            "reason": reason})
            try:   # 0911 호환검수 R7: 브리지발 정지를 ROS 디버그 토픽으로도 알림 — 콘솔 Telemetry.receive 의 event 분기가 소비
                self.pub_dbg.publish(String(data=json.dumps(
                    {"event": "stop_all", "reason": str(reason), "timestamp": time.time()})))
            except Exception:  # noqa: BLE001 — 알림 실패가 안전동작을 막으면 안 된다
                pass

    rclpy.init()
    node = RlBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node._bb_close()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
