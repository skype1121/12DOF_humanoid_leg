# RL 보행 정책 — 실물 이식 가이드 (sim2real)

**시뮬 전용 산출물** — 이 문서는 "이식 준비" 상태를 정의한다. 실물 송신은 승윤님 결정·입회 하에만.
작성: 2026-07-19 (12시간 자율 작업 중). 관련: `docs/RL보행_작업로그.md`, `docs/RL보행_최종보고서.md`

## 1. 왜 이 정책은 이식 가능하게 설계됐나

| 실물 제약 | 학습 환경 반영 | 근거 |
|---|---|---|
| MIT 프로토콜 kd ≤ 5 | **kd=5로 학습** (±20% DR) | kd25→kd5 전이는 FSM에서 실패 확인(게인 정합 실험). 처음부터 kd5 동역학에서 학습 |
| kp ≤ 500 (실용 60~250) | kp=150 (±20% DR) | kd5+kp150 기립·트림 실측 검증값 |
| 토크 ≤ 25Nm (AK70-10) / 24Nm (AK45-36) | effort_limit 25/24 | 모터 공식 스펙 |
| 발목롤 AK45-36 속도 ≤ ~5rad/s | velocity_limit 5.0 + 반사관성 0.0236kg·m² | cubemars.com AK45-36 (감속 36:1) |
| 노드 슬루 4°/tick @50Hz | 동일 슬루를 액션 후처리로 내장 | real_bridge.py NODE_SLEW_DEG_PER_TICK |
| 제어 주기 50Hz | 정책 주기 50Hz (물리 200Hz × decimation 4) | Stage8 NODE_TICK_HZ |
| 관측 = IMU + 엔코더뿐 | 액터 관측에 base_lin_vel 제외 (자이로+중력벡터+관절상태+직전액션만) | iAHRS IMU로 전부 측정 가능 |

## 2. 정책 인터페이스 사양

**파일**: `logs/rsl_rl/biped12_flat/<런>/exported/policy.pt` (TorchScript) / `policy.onnx` (Jetson용)
입력 45차원 (float32, 순서 고정):

| 구간 | 차원 | 내용 | 실물 소스 |
|---|---|---|---|
| 0-2 | 3 | base 각속도 [rad/s] (base 프레임: +X전방 +Y좌 +Z상) | iAHRS 자이로 (프레임 정렬 필요) |
| 3-5 | 3 | 중력 방향 단위벡터 (base 프레임, 직립=(0,0,-1)) | iAHRS 자세 쿼터니언에서 유도 |
| 6-8 | 3 | 속도 명령 (vx[m/s], vy[m/s], wz[rad/s]) | 조작기/상위 제어 |
| 9-20 | 12 | 관절각 − 기본자세 [rad], **JOINT_ORDER 순서** | 모터 엔코더 (MIT 모드 피드백) |
| 21-32 | 12 | 관절 각속도 [rad/s] | 모터 엔코더 |
| 33-44 | 12 | 직전 정책 출력 (raw action) | 자체 버퍼 |

출력 12차원 (raw action) → 관절 목표각: `target = default_pos + 0.25 × action` [rad]
→ **슬루 ±4°/tick 적용 후** 명령 (학습 때 이 슬루가 있었으므로 실물에서도 노드 슬루가 동일 역할).

**JOINT_ORDER** (학습 환경 실측 — 하드웨어맵 모터ID와 다름! 반드시 매핑):
```
[0] left_hip_f  [1] right_hip_f  [2] left_hip_a  [3] right_hip_a
[4] left_hip_r  [5] right_hip_r  [6] left_knee   [7] right_knee
[8] left_ankle_f [9] right_ankle_f [10] left_ankle_r [11] right_ankle_r
```
(하드웨어맵 motor_id: 좌 1-6 / 우 7-12 순차 — 이름 기준으로 매핑할 것)

**기본자세 (default_pos)**: 전관절 0 + `left_knee −0.10472 / right_knee +0.10472`(무릎 6° 굽힘)
+ `left_ankle_f +0.0244 / right_ankle_f −0.0244`(dorsi 트림). 부호 규약 = joint_direction 실측 = ANAT_SIGN (12관절 무중력 프로브로 변환 자산 일치 확인 — log_picture/05).

## 3. 이식 경로 3단계

### 3-1. sim-to-sim 검증 (완료 대상, 즉시 가능)
`python3 scripts/rl_walk_demo.py` — walk_scene(60Hz, 구 자산)에서 재생.
학습 플랜트(200Hz, armature 포함)와 다른 시뮬로의 전이 = 1차 강건성 증거.
게인은 자동으로 kp150/kd5 정합됨.

### 3-2. 실물 오프라인 검증 (하드웨어 무송신)
1. 실물 로봇을 손으로 세워두고 iAHRS+엔코더 로그만 수집 (송신 없음)
2. 로그를 정책에 입력해 출력 목표각의 타당성 검사 (발산·리밋 위반·슬루 포화율)
3. `real_bridge.py` 검증기(리밋∩시뮬, 델타 4°/tick)를 그대로 통과하는지 확인

### 3-3. 실물 온라인 (승윤님 입회, 별도 세션)
- 정책 실행 위치: 제어 PC 또는 Jetson (ONNX Runtime, 45→12 MLP라 1ms 미만)
- **주의: Stage8 노드는 SET_JOINT_TARGET 단일관절 메시지만 수용** → 12관절×50Hz = 600msg/s.
  토픽 대역폭 검증 필요. 부족하면 노드에 멀티관절 명령 추가가 선행 과제
- 베이스라인 규약은 tape와 동일: 실물을 시뮬 중립(전관절 0°)에 두고 SET_BASELINE
  → 목표각 = 시뮬 절대각 [deg] (하드웨어맵 direction·sign 적용, 현재 전부 +1)
- 게인 설정: kp 150 (AK 2.0 Kp 범위 0-500 내), kd 5.0 (상한) — 전 관절
- 시작 절차: 매달아 두고 스탠딩 명령(cmd=0,0,0)부터 → 접지 스탠딩 → 저속 0.3m/s
- 안전: SafetyFilter 스텝당 30° 제한 유지, 낙상 시 STOP_ALL (UI HOLD와 동일 철학)

## 4. 향후 센서 활용 (승윤님 보유 하드웨어)

| 센서 | 용도 | 반영 방법 |
|---|---|---|
| iAHRS IMU | 필수 — obs 0-5 소스 | 축 정렬만 하면 즉시 사용 (학습 노이즈 DR ±0.2rad/s, ±0.05 반영됨) |
| RA30P 압력센서(발) | 접촉 검출 → 관측 보강 | 차기 학습에서 obs에 발 접촉 2bit 추가 시 계단·외란 성능 ↑ (시뮬 접촉센서와 1:1 대응) |
| RealSense D455 | 지각 보행 (계단 높이맵) | Isaac Lab height-scan 관측 재학습 필요. 현 스택은 블라인드 험지까지 (TiledCamera Blackwell 이슈로 비전 학습은 보류) |

## 5. 체크리스트 (이식 준비 완료 판정)

- [ ] 평지 정책: 시뮬 무낙상 + 속도추종 (eval 리포트 log_picture/정책평가*.txt)
- [ ] ONNX/JIT 익스포트 존재
- [ ] sim-to-sim(walk_scene) 재생 PASS
- [ ] 관절 순서/부호/기본자세 문서화 (본 문서 §2)
- [ ] 오프라인 검증 절차 정의 (§3-2)
- [ ] 600msg/s 대역폭 이슈 인지 (§3-3)
