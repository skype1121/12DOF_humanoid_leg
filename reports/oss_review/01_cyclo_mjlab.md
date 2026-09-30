# cyclo_mjlab 조사 보고 (2026-09-30)

**경로 약어** (모두 scratchpad에 shallow clone 또는 압축 해제한 것):
- `[C]` = `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/cyclo_mjlab`. HEAD `e659f69`, v0.0.3.
- `[A]` = `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/ai_sapiens`. `[C]/third_party/ai_sapiens` 서브모듈로 핀된 커밋 `611a070`과 같은 것이며, 실제 배포 코드는 전부 여기에 있다.
- `[M]` = `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/mjlab_pkg/src/mjlab`. mjlab 1.2.0 wheel이다. cyclo가 import하는 보상·DR·액추에이터 함수의 원본이 여기에 있다.

우리 저장소는 수정하지 않았다. 8-④ 항목을 확인하려고 `rl_walking/` 파일 세 개만 읽기로 grep했다.

---

## 1. 요약

ROBOTIS의 **AI Sapiens K1 Rev.1**을 위한 RL 저장소다. K1은 23 DoF 전신 휴머노이드로, MJCF 질량 합계가 약 35.7 kg이고 골반 높이는 약 0.78~0.80 m다.
- **시뮬·학습**: MJLab 1.2.0(MuJoCo 3.5 / MuJoCo-Warp GPU)과 rsl-rl-lib 5.0.1 PPO를 쓴다.
- **태스크**: 평지 속도추종 보행(`Cyclo-Velocity-Flat-K1-Rev1-v0`)과 모션 모방 Dance1/2(whole_body_tracking 이식) 두 가지뿐이다. **서기, 행잉→착지 태스크는 없다.**
- **라이선스**: Apache-2.0. mimic 부분만 whole_body_tracking의 MIT다.
- **성숙도**: 2026-08-27 생성, 커밋 14개, 별 5개, 테스트·평가 도구 없음. 사실상 학습·재생 스크립트와 설정만 있다.
- **실질적 가치는 서브모듈 `ai_sapiens`에 있다.** 이쪽은 2026-07-21 생성, 별 66개, CI와 단위 테스트가 있다. 구성은 C++ ROS 2(Jazzy) 실시간 배포 런타임 `ai_sapiens_sim2real`(1 kHz RT 스레드, ONNX), ros2_control 1 kHz 임피던스 컨트롤러와 Dynamixel MIT 모드 드라이버, 그리고 MuJoCo 갠트리(매달기→내리기→해제) sim2sim이다.
- **학습 쪽 결손**: 지연·백래시·관절마찰·게인 DR이 전혀 없다. 지연 문제는 학습이 아니라 배포 구조를 빠르게 만드는 방식으로 풀고 있다.
- **(추정) 배포 정책의 출처가 다르다.** 동봉된 실물 배포 보행 정책(`walk_default`)은 관측 이력, 스케일, 게인, 키 이름이 이 저장소 설정과 모두 다르다. 이 저장소가 아니라 주석에 나오는 "Cyclo Lab" 파이프라인에서 나온 것으로 보인다(근거는 §2.4).

## 2. 아키텍처

### 2.1 학습 파이프라인
- **학습**: `[C]/scripts/reinforcement_learning/train.py`.
  - tyro로 태스크 ID를 받는다.
  - `ManagerBasedRlEnv` → `RslRlVecEnvWrapper` → `MjlabOnPolicyRunner` 순으로 구성한다.
  - 학습을 시작할 때 `params/sim2real.yaml`을 자동으로 내보낸다(122-125행).
  - `runner.learn(..., init_at_random_ep_len=True)`(168-170행).
- **재생**: `play.py`가 `exported/policy.onnx`와 `policy.pt`를 내보낸다(429-441행).
- **추적성**: `[C]/source/utils/export_git_state.py:59-87`가 실행마다 브랜치, 커밋, `git diff --binary HEAD`를 저장한다.

### 2.2 설정 위치
| 대상 | 파일 |
|---|---|
| 로봇·액추에이터 | `[C]/source/assets/robots/robotis_k1/k1_rev1.py` (MJCF는 `[A]/ai_sapiens_description/mujoco/k1/k1.xml`) |
| Velocity 기본 MDP | `[C]/source/tasks/velocity/velocity_env_cfg.py`: 관측 57-86, 액션 96-98, 명령 102-117, 이벤트·DR 121-190, 보상 194-277, 종료 281-287, 커리큘럼 291-301, 시뮬 327-337 |
| K1 전용 오버라이드 | `[C]/source/tasks/velocity/config/k1_rev1/env_cfgs.py` (보상 교체 102-234, play 236-245) |
| 커스텀 보상·관측 | `[C]/source/tasks/velocity/mdp/rewards.py`, `observations.py` |
| PPO | `[C]/source/tasks/velocity/config/k1_rev1/rl_cfg.py:30-67`, `[C]/source/tasks/mimic/config/k1_rev1/agents/rsl_rl_ppo_cfg.py:30-67` |
| Mimic | `[C]/source/tasks/mimic/tracking_env_cfg.py`, `mdp/commands.py` |
| 배포 계약 export | `[C]/source/utils/export_sim2real_cfg.py:258-321` |

### 2.3 배포 스택 (ai_sapiens)
```
Dynamixel(MIT모드, 펌웨어 PD) ←1kHz sync R/W, e2d2udp 6Mbps→ ros2_control(1kHz, 우선순위 80)
   ├ joint_state_broadcaster / imu_sensor_broadcaster → /joint_states, /imu  (추정: 1 kHz 발행)
   └ joint_group_impedance_controller ← /…/commands (pos, ff, kp, kd)
ai_sapiens_sim2real_node:  콜백 → RealtimeBuffer(최신값 1칸) → RT 스레드 1 kHz (SCHED_FIFO 50, mlockall)
   sense → decide(ModeController) → act(PolicyRuntime, 50 Hz 데시메이션, ONNX) → command(같은 틱 publish)
```
- RMW는 Zenoh다.
- docker 설정: `rtprio 99`, memlock, privileged (`[A]/docker/docker-compose.yml`).

### 2.4 주의: 동봉 정책과 이 저장소 설정의 불일치
`[A]/ai_sapiens_sim2real/assets/k1/locomotion/velocity/walk_default/params/sim2real.yaml`과 이 저장소 설정을 비교하면 다음이 모두 다르다.

| 항목 | 동봉 walk_default | 이 저장소가 만드는 것 |
|---|---|---|
| 관측 이력 | `history_length: 5` | 1 |
| gait_phase | 없음 | 있음 |
| 관측 스케일 | 각속도 0.2, 관절속도 0.05 | 1.0 |
| 관측 항목 이름 | `joint_pos_rel`, `last_action` | `joint_pos`, `actions` (exporter 이름표 62-71행) |
| PD 게인 | hip 100/2, knee 150/4, **ankle 40/2**, waist 200/5 | 76.5/4.87, 22.3/1.42 |
| 액션 스케일 | 0.25 균일, `raw_clip ±5` | 0.317 / 0.530 |
| ONNX | 입력 390 = 5×78, 정규화기 없음, 레거시 키 `actor.0.weight` (scratchpad에서 onnx로 확인) | 정규화 켬 |

- mimic 자산의 발목 스케일 0.265도 이 저장소 공식 결과(0.317/0.530)와 다르다.
- 결론 **(추정)**: 동봉 정책은 다른 파이프라인("Cyclo Lab")의 산출물이다. **따라서 이 저장소의 velocity 설정이 실물에서 검증되었다는 근거는 저장소 안에 없다.**

## 3. 보상 항목

모든 가중치는 reward manager에서 `weight·value·dt`로 곱해진다(`[M]/managers/reward_manager.py:115-130`, `scale_rewards_by_dt=True`는 `[M]/envs/manager_based_rl_env.py:135`). 따라서 가중치는 초당 비율이며, Isaac Lab을 쓰는 우리와 같은 규약이다.

### 3.1 K1 Velocity 최종 활성 항목 (`env_cfgs.py` 적용 후)
| 항목 | w | 식 | 게이팅 | 출처 |
|---|---|---|---|---|
| track_linear_velocity | +1.2 | exp(−(‖v_xy−c_xy‖² **+ v_z²**)/0.25) | 상시 | `[M]/tasks/velocity/mdp/rewards.py:23-40` |
| **track_angular_velocity** | +1.2 | exp(−((ω_z−c_z)² **+ ω_x² + ω_y²**)/0.5) | **상시 (서기 env 포함)** | 同 43-60 |
| upright | +1.0 | exp(−‖g_xy(torso)‖²/0.2) | 상시 | 同 63-85 |
| gait_contact_swing_tracking | +0.5 | 발별 1[스케줄 접촉==실측 접촉] 평균. T=0.6 s, 위상차 0.5, stance 0.55 | ‖cmd‖>0.1일 때만 | `[C]/…/mdp/rewards.py:111-147` |
| termination_penalty | −200 | 1[비시간초과 종료] | – | env_cfgs 112-114 |
| base_height | −10 | (z_root−0.80)² | 상시 | rewards.py:37-44 |
| lin_vel_z_l2 | −2.0 | v_z² | 상시 | 47-53 |
| ang_vel_xy_l2 | −0.1 | ω_x²+ω_y² (루트, 몸체좌표) | 상시 | 56-62 |
| flat_orientation_l2 | −5.0 | ‖g_xy(root)‖² | 상시 | `[M]/envs/mdp/rewards.py:157-163` |
| action_rate_l2 | −0.1 | ‖a_t−a_{t−1}‖² (raw action) | 상시 | 同 58-65 |
| dof_acc_l2 | −5e-7 | Σq̈² (다리) | 상시 | 50-55 |
| dof_torques_l2 | −3e-6 | Στ² (다리) | 상시 | 32-39 |
| dof_pos_limits | −5.0 | soft limit(0.9배) 초과량 합 | 상시 | 81-96 |
| joint_deviation_hip (yaw·roll) | −0.5 | Σ\|q−q0\| | 상시 | rewards.py:65-75 |
| 팔·허리 편차 L1 | −0.1 ~ −1.0 | Σ\|q−q0\| | 상시 | env_cfgs 162-213 |
| foot_clearance | −2.0 | Σ\|z_foot−0.10\|·‖v_foot,xy‖ | ‖cmd‖>0.05 | `[M]` 167-189 |
| foot_slip | −0.1 | Σ‖v_foot,xy‖²·contact | ‖cmd‖>0.05 | 244-271 |
| feet_lateral_separation | −1.0 | (max(0, 0.13−y_sep)/0.13)² | ‖cmd‖>0.1 | rewards.py:78-108 |
| soft_landing | −5e-4 | Σ‖F‖·first_contact | ‖cmd‖>0.05 | `[M]` 274-300 |
| self_collisions | −1.0 | 최근 4 서브스텝 중 10 N 넘은 자기충돌 수 | 상시 | env_cfgs 230-234 |

- **제거된 항목**(env_cfgs 102-107 `pop`):
  - `body_ang_vel`(기본 가중치 0)
  - `angular_momentum`(−0.05)
  - `default_stand_joint_deviation`(variable_posture, +1.0)
- 참고로 upstream G1은 `std_standing={".*":0.05}`, `body_ang_vel −0.05`, `angmom −0.02`를 쓴다(`[M]/tasks/velocity/config/g1/env_cfgs.py:94,140-141`).
- **이벤트형 보너스(회복 보너스 등)는 없다.** 모든 항목이 매 스텝 조밀하게 계산되고, 게이팅은 명령 크기로만 한다. 균형 상실 상태로 끄는 항목은 없다.

### 3.2 흔들림에 대한 정량 비교 (계산, 사인 흔들림 한 주기 평균)
| 흔들림 | cyclo exp 항 손실(/s) | L2 −0.1 (/s) | 비율 |
|---|---|---|---|
| 0.5 Hz ±2° | 0.0143 | 0.0006 | 24× |
| 1 Hz ±2° | 0.056 | 0.0024 | 23× |
| 2 Hz ±2° | 0.201 | 0.0096 | 21× |
| 2 Hz ±5° | 0.697 | 0.060 | 12× |

- 정적 2° 기울기: flat_l2 0.006/s, upright 0.006/s로 작다.
- **action_rate**를 진폭 A로 계산하면 0.5 Hz에서 0.002·A², 2 Hz에서 0.031·A², 10 Hz에서 0.69·A²다. action_acc는 0.5 Hz에서 8e-6·A²다. 즉 **액션 평활 항은 채터(≥5 Hz) 필터이고, 0.5~2 Hz 흔들림에는 사실상 반응하지 않는다.**
- 이 저장소에서 저주파 흔들림을 누르는 주력은 track_angular_velocity의 xy 성분이다.

### 3.3 Mimic (`tracking_env_cfg.py:214-253`)
- 앵커 위치·자세 0.5/0.5, 바디 위치·자세·선속·각속 각 1.0. 모두 exp 커널이다.
- joint_acc −2.5e-7, torque −1e-5, action_rate −0.1, joint_limit −10, undesired_contacts −0.1(base_env_cfg 85-89).

## 4. DR·액추에이터 모델링

### 4.1 주파수와 PD
- **시뮬**: dt 5 ms(200 Hz), implicitfast, solver iterations 10, ls 20. decimation 4로 정책은 50 Hz다(`velocity_env_cfg.py:327-337`). 액션은 매 서브스텝 적용된다(`[M]/envs/manager_based_rl_env.py:368-375`).
- **액추에이터**: `BuiltinPositionActuatorCfg`는 MuJoCo `<position>` 액추에이터다. affine bias로 감쇠를 암시적으로 적분하고, `ctrllimited=False`, forcerange는 ±τmax다(`[M]/utils/spec.py:142-`).
- **게인은 반사관성(armature)에서 유도한다** (`k1_rev1.py:53-64`):
  ```python
  NATURAL_FREQ = 10.0 * 2.0 * math.pi; DAMPING_RATIO = 2.0
  STIFFNESS = ARMATURE * NATURAL_FREQ**2 ; DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
  ```
  - QC080(hip, knee, 발목 pitch, 허리): kp 76.45, kd 4.87, τmax 96.86, armature 0.01937
  - QC060(발목 roll, 팔): kp 22.30, kd 1.42, τmax 47.28, armature 0.00565
- **액션 스케일** = 0.25·τmax/kp → 0.317 rad(18°) / 0.530 rad(30°) (229-241행).
- **MJCF 관절 설정**: `damping=0`, `frictionloss=0`(k1.xml 5행). ActuatorCfg의 frictionloss 기본값도 0이다(`[M]/actuator/actuator.py`).
- **발 모델**: 반경 5 mm 구 9개로 된 점접촉 발이다(k1.xml 110-126). 마찰 0.6, condim 3.

### 4.2 DR (velocity, `velocity_env_cfg.py:121-190`)
| 이벤트 | 시점 | 범위 |
|---|---|---|
| reset_base | reset | x·y ±0.5 m, z +0.01~0.05, yaw ±π. 초기 속도 없음 |
| reset_robot_joints | reset | 오프셋 0 |
| push_robot | 1~3 s 간격 | **현재 속도에 더한다**(`[M]/envs/mdp/events.py:321`). vx·vy ±0.5, vz ±0.4 m/s, **roll·pitch ±0.52**, yaw ±0.78 rad/s |
| foot_friction | startup | 0.3~1.2 절대값. 발 geom 공유 단일값, 접선 축만 |
| encoder_bias | startup | 관절별 ±0.015 rad. **액추에이터 목표에서 뺀다**(`[M]/envs/mdp/actions/actions.py:206-209`). velocity 관측은 unbiased라 정책에 보이지 않는 영점 오차가 된다 |
| base_com | startup | torso ipos x·y ±2.5 cm, z ±3 cm |

- **관측 노이즈**(균등, 가산): 자이로 ±0.2, 중력 ±0.05, 관절각 ±0.01, 관절속도 ±1.5. 우리 S8 기본값과 같은 대역이다.
- **Mimic**:
  - 전 geom 마찰 0.3~1.2
  - encoder bias ±0.01 (관측에도 bias가 보인다, `biased=True`)
  - torso CoM x ±2.5, y·z ±5 cm
  - 푸시 1~3 s 간격(z는 ±0.2)
  - 리셋 자세·속도·관절 노이즈(`tracking_env_cfg.py:146-210`)

### 4.3 지연: cyclo는 없음, mjlab에는 기능이 있다
cyclo는 어떤 지연도 모델링하지 않는다. mjlab 1.2.0이 제공하는 기능은 다음과 같다.
- `DelayedActuatorCfg`(`[M]/actuator/delayed_actuator.py:20-61`): 물리 스텝 단위 lag를 [min, max]에서 균등 샘플한다. 옵션은 `delay_hold_prob`, `delay_update_period`, `per_env_phase`다(시간 상관 지터).
- 관측 항목별 `delay_min/max_lag`(정책 스텝 단위), hold, update_period (`[M]/managers/observation_manager.py:17-63`). 처리 순서는 compute → noise → clip → scale → delay → history다.
- `dr.sync_actuator_delays`, `dr.pd_gains`, `dr.effort_limits` (`[M]/envs/mdp/dr/actuator.py:23,120,201`).

### 4.4 마찰·백래시·액추에이터넷
- cyclo는 쓰지 않는다. mjlab에 있는 도구:
  - `dr.joint_friction`, `dof_frictionloss`, `dof_damping`, `dof_armature` DR
  - `DcMotorActuator`(토크-속도 곡선)
  - `LearnedMlpActuator`(위치오차·속도 이력 → 토크, TorchScript, `[M]/actuator/learned_actuator.py:21-60`)
- **mjlab 액추에이터 모듈(builtin, pd, dc, learned, delayed, xml)에는 백래시(데드존) 모델이 없다.**

### 4.5 sim2sim PD (`[A]/ai_sapiens_mujoco/src/mujoco_simulation.cpp:81-87, 216-229`)
- `ctrl = ff + kp·q*`, `bias=[0,−kp,−kd]` 형태의 affine 액추에이터다. kd를 암시적으로 적분하고, ff+PD 합계에 forcerange를 건다.
- 지연·노이즈 주입은 없다.

## 5. 관측 설계
- **cyclo velocity actor**(`velocity_env_cfg.py:57-72`):
  - 항목: 자이로(IMU site), projected gravity, 명령 3, `gait_phase`(sin/cos, T=0.6 s), 관절각 rel, 관절속도, last_action(raw).
  - **그룹 history_length=1**(83-86행)이라 이력이 없다.
  - `obs_normalization=True`(EmpiricalNormalization).
- **gait_phase**(`[C]/…/mdp/observations.py:29-53`): `clock * is_moving`. **명령 합이 0.1 미만이면 0**이 된다("정지 시 스텝 단서를 주지 않기 위함").
- **Critic(비대칭)**: actor 항목 + base_lin_vel + foot_height, air_time, contact, contact_forces. noise는 끈다.
- **동봉 walk_default**: 이력 5(390 = 5×78), 각속도 ×0.2, 관절속도 ×0.05, 정규화 없음, gait_phase 없음.
- **이력 규약**:
  - term-major, 오래된 것 → 최신 순이다(`[M]/managers/observation_manager.py:57-63`, 배포 `[A]/…/observation/observation_term_cfg.hpp:132-137`).
  - 리셋이나 진입 시 이력을 0이 아니라 **첫 관측으로 채운다.** 학습(`[M]/utils/buffers/circular_buffer.py:210-213`)과 배포(`observation_term_cfg.hpp:67-73`)가 같다.
  - clip → scale 순서는 Isaac Lab 기본과 같다.
- **배포 관측 구현**(`[A]/…/src/observation/observations.cpp:96-180`):
  - 관절속도는 `/joint_states` 원값을 쓴다(필터 없음).
  - 중력 벡터는 IMU 쿼터니언으로 `q̄·[0,0,−1]`을 계산한다.
  - last_action은 raw 출력이다.

## 6. Sim2real·배포

### 6.1 제어 루프와 스레딩
- **센서 핸들**(`imu_sensor_handle.cpp:46-49, 54-74, 81-108` / `joint_states_sensor_handle.cpp:53-56, 70-87, 94-110`):
  - 구독 깊이는 10이지만 **콜백은 파싱 후 `realtime_tools::RealtimeBuffer`에 덮어쓰기만 한다.** 수신 시각(`steady_clock`)도 함께 기록한다.
  - RT 루프는 매 틱 최신값 1개만 읽는다. 큐를 소비하지 않는다.
- **RT 루프**(`control_loop.cpp:227-274, 289-310, 335-355, 435-458`):
  - 별도 `std::thread`에서 mlockall, SCHED_FIFO(50), 1 kHz `sleep_until`로 돈다.
  - 틱 순서는 sense → decide → act → command다. overrun이 나면 따라잡지 않고 기준 시각을 재설정한다.
  - ROS spin은 메인 스레드의 단일 executor가 맡지만, 콜백 비용이 거의 없다.
- **정책 데시메이션**(`policy_runtime.cpp:186-197`):
  - 측정 주기를 누적해서 `step_dt`(0.02 s) 이상이면 추론한다.
  - `fmod`로 밀린 스텝을 버린다(몰아서 실행하지 않는다).
  - 관측 → ONNX → 액션 처리 → **같은 틱에 publish**한다.
- **ONNX Runtime**: intra/inter 스레드 1, sequential 실행("지연 예측 가능성", `onnx_inference.cpp:32-40`).
- **명령 경로**:
  - `JointImpedanceCommand{positions, feedforward, kp, kd}`를 1 kHz로 발행한다(`joint_command_publisher.cpp:85-119`).
  - 임피던스 컨트롤러는 구독 콜백 → RealtimeBuffer → 1 kHz update로 HW 인터페이스에 쓴다(`…impedance_controller.cpp:125-157, 212-223`).
  - Dynamixel은 Operating Mode 6(MIT P/D), sync read 주기 1000 µs, 6 Mbps UDP 브리지다(`k1_left_leg.ros2_control.xacro`).
- **정리**: 정책 출력부터 모터 명령까지 소프트웨어 대기열이 구조적으로 없다. 대기는 약 1 ms에 전송 지연이 더해지는 수준이다 **(추정, 측정값은 저장소에 없음)**.

### 6.2 안전장치
- **failsafe 우선순위**(`mode_controller.cpp:248-266`): 텔레옵 타임아웃 → 텔레옵 Damping → 내부 damping 요청 → 액션 한계 → 자세 불량.
- 조건별 동작:
  - IMU나 관절상태가 **0.5 s 동안 없으면 damping으로 래치하고 재시작 전까지 풀지 않는다.**
  - 텔레옵 0.2 s 무응답이면 damping으로 간다.
  - 추론이 3회 연속 실패하면 damping으로 간다(1~2회는 이전 액션 유지).
  - 처리된 액션이 NaN이거나 |q*|>2π면 커밋을 거부하고 damping으로 간다(`policy_runtime.cpp:269-309`, `.hpp:164-169`).
  - 자세 불량 기준은 acos(−g_z)>**2.0 rad**다(mimic은 진입 500 ms 뒤 3.0 rad, `mode_controller.cpp:942-956`).
  - RT 스레드에서 예외가 나면 비상 damping을 건다: kp=0, kd 유지, 목표=측정각(`control_loop.cpp:276-287`, `mode_controller.cpp:1024-1030`).
- **Damping 모드**: kp=0, kd=3.0, 목표=매 틱 측정각(`k1_config.yaml:133-141`, `mode_controller.cpp:1087-1093`).
- **전이 규칙**(`mode_controller.cpp:963-981`): Damping→Posture만 허용, Posture/Policy 상호 전이 허용, 어디서든 →Damping 허용.
- **ReadyPose**: 측정각에서 목표까지 **3 s 선형 보간**한다(hip·knee·ankle kp 100/150/40, kd 2/4/2, 1095-1109행).
- **정책 진입**(`policy_runtime.cpp:88-95, 131-142`; `mode_controller.cpp:1068-1074`):
  - 같은 틱에 게인·기본자세를 설치하고 이력을 현재 관측으로 채운 뒤 last_action=0으로 첫 추론을 한다. 명령 공백은 없다.
  - **블렌딩은 없다.** ReadyPose 목표(−0.18/0.36/−0.18)와 정책 기본자세(−0.205/0.517/−0.307)가 달라 목표가 계단식으로 바뀐다.
- **시작·권한**:
  - 시작 게이트: 텔레옵이 Damping 또는 ReadyPose 입력과 일치할 때까지 대기한다.
  - API 권한은 스위치 상승 에지가 새로 들어와야 하고, 하트비트 sequence가 계속 증가해야 한다(같은 값이 반복되면 stale 처리).
  - `RealtimeRequestGate`는 단일 슬롯에 epoch 기반 철회를 한다(`[A]/…/mode_runtime/realtime_request_gate.hpp`).
  - mock HW로 전체 스택을 돌리는 smoke test가 있다(`scripts/run_mode_smoke_tests.py`).

### 6.3 MuJoCo 갠트리 sim2sim (`[A]/ai_sapiens_mujoco/`)
- 구성: mocap 바디 + `weld`(gantry↔torso_link, `scene_gantry.xml`).
- 서비스:
  - `set_height`: 0.05 m/s로 이동(`mujoco_simulation.cpp:280-293, 388-400`)
  - `release`: weld 비활성화
  - `attach`: 관절각은 유지하고 직립 행잉 자세로 복원(295-380행)
- ros2_control HW 플러그인으로 실물과 **같은 컨트롤러, 토픽, 런타임**을 그대로 돌린다.
- 실물 운용 순서와 같다: 매단 채 기동 → 발이 닿을 때까지 내리기 → 정책 시작 → 해제.

### 6.4 계약
- 학습 시작 시 해석된 env에서 `sim2real.yaml`을 자동 생성한다. 내용은 관절 순서, step_dt, 관절별 default·kp·kd, 액션 scale·offset, 관측 항목·스케일·clip·이력이다.
- 런타임은 ONNX 입출력 크기와 YAML 합계를 대조해 기동 전에 실패시킨다(`policy_runtime.cpp:423-469`).
- **자체 불일치 한 건**: 학습 쪽 gait_phase는 정지 시 0인데, 배포 `GaitClock`은 자유 진행이다. "standing-aware" 규칙은 주석 처리되어 있다("export가 파라미터를 내보낼 때까지 비활성", `gait_clock.cpp:70-139`).

## 7. PPO·커리큘럼
- **Velocity**(`rl_cfg.py:30-67`):
  - actor/critic 512-256-128 ELU, obs 정규화 켬, Gaussian scalar std, init_std 1.0
  - clip 0.2, entropy 0.01, epoch 5, minibatch 4, lr 1e-3 adaptive(KL 0.01), γ 0.99, λ 0.95, grad norm 1.0
  - env당 24 스텝, 4096 env, 30000 iteration, 저장 간격 50, seed 42(기본)
  - 에피소드 20 s, 종료 조건은 기울기 70°
- **Mimic**: 정규화 끔, std log, entropy 0.005, clip_actions 10, 에피소드 30 s.
- **커리큘럼**:
  - velocity는 명령 단계가 step 0 하나뿐이라 사실상 없다(291-301행). 서기 env 비율 10%(`rel_standing_envs=0.1`), 명령 재샘플 3~8 s.
  - Mimic은 **실패 기반 적응형 시작점 샘플링**이다(`[C]/…/mimic/mdp/commands.py:298-359, 480-484`): 1 s 구간별 실패 EMA(α 0.001)에 균등 0.1을 섞는다.
- 우리 PPO와 같은 계열이라 하이퍼파라미터에서 새로운 정보는 없다.

---

## 8. 우리에게 추출할 것 (우리 문제 기준 기대효과 순)

**① [지연·최우선] ai_sapiens_sim2real의 데이터 경로 패턴**
- **출처**: `[A]/ai_sapiens_sim2real/src/sensor_handles/*.cpp`, `src/control_loop.cpp`, `src/policy/policy_runtime.cpp:165-197`.
- **가져올 것 (rclpy 이식)**:
  - 센서 QoS를 KEEP_LAST depth 1(필요하면 BEST_EFFORT)로 둔다.
  - 콜백은 (메시지, `time.monotonic()` 수신시각)을 최신값 한 칸에 덮어쓰기만 한다.
  - 제어는 전용 스레드(또는 MultiThreadedExecutor + 별도 callback group)의 단조시계 루프에서 최신값 읽기 → 추론 → **같은 틱 송신**으로 처리한다.
  - 밀린 스텝은 버린다(`fmod`).
  - 수신시각 기반 stale 감시를 래치로 둔다.
  - 가능하면 브리지와 stage8을 같은 프로세스나 함수 호출로 합쳐 토픽 한 단계를 없앤다.
  - 단계별 지연은 디버그 토픽처럼 `/policy_input/raw_observation`, `/policy_output/processed_action` 같은 기록점을 두어 측정한다.
- **근거 (계산)**: 순수 지연 330 ms면 위상지연이 0.5 Hz에서 59°, 1.5 Hz에서 178°, 2 Hz에서 238°다. 60 ms면 2 Hz에서 43°다. 약 1.5 Hz 이상의 정책 되먹임은 부호가 뒤집힌다. 실물의 "방향은 맞는데 늦은 보정"과 ~2 Hz 지속 흔들림에 들어맞는다.
- **주의**:
  - 큐 깊이 10 적체가 원인이라는 것은 **아직 가설**이다. 단계별 header.stamp와 수신시각을 먼저 재고, 한 변수만 바꾼 전후 비교로 확인해야 한다(CLAUDE.md 규칙 1·2).
  - Python GIL과 CAN 50 Hz 제약이 있다. stage8 가드는 유지해야 한다.

**② [보상] 게이트 없는 각속도 형상화, 이벤트 보너스 없음**
- **출처**: `[M]/tasks/velocity/mdp/rewards.py:43-60`에 `[C]/…/env_cfgs.py:127-137` 조합.
- **가져올 것**:
  - 균형 상실 게이팅 없이 상시 켜진 유계 커널 w·exp(−(ω_x²+ω_y²)/σ²)을 L2와 **함께** 둔다.
  - 회복 이벤트 보너스를 없애고, 조밀한 항과 종료 페널티로 대체하는 대조안을 둔다.
- **근거 (계산)**: §3.2에서 이 커널은 0.5~2 Hz, ±2° 흔들림에 우리 −0.1 L2보다 약 21~24배 큰 비용을 준다. action_rate와 action_acc는 이 대역에 사실상 반응하지 않으므로 상태 기반 항이 필요하다.
- **주의**:
  - 절대값은 여전히 작다(0.01~0.2/s).
  - exp는 0 근처 기울기가 작고 큰 ω에서 포화하므로 L2를 같이 둬야 한다.
  - 정숙 보상이 전이 중 필요한 보정을 억제하는지 확인해야 한다(CLAUDE.md).
  - 시뮬 지연이 실물보다 짧으면 이 형상화만으로는 지연 발진을 학습으로 피할 수 없다.

**③ [시뮬-실물 지연 정합] mjlab 지연 기능의 설계 요소**
- **출처**: `[M]/actuator/delayed_actuator.py:20-61`, `[M]/utils/buffers/delay_buffer.py:12-118`.
- **가져올 것**:
  - `hold_prob`, `update_period`, `per_env_phase` 방식의 시간 상관 지터
  - 액션 지연과 관측 지연 분리
- **확인 필요**: 커밋 제목에 G12가 "지연 3~8→9~14"틱(180~280 ms)으로 나와 있다. 실측 320~340 ms(16~17틱)보다 짧아 보인다. 다만 모터 40~60 ms 포함 여부와 관측·액션 분리 여부 같은 정의는 확인하지 않았다. cyclo 자체는 지연 DR 없이 배포를 빠르게 만드는 쪽을 택했다. ①이 먼저다.

**④ [저비용 진단] 정지 시 클록 입력**
- **출처**: cyclo는 `gait_phase = clock × is_moving`(`[C]/…/mdp/observations.py:29-53`)이다. 동시에 배포 쪽은 자유 진행 클록이라 같은 저장소 안에서도 계약이 어긋나 있다(§6.4).
- **우리 코드 읽기 결과 (정밀 검증 아님)**:
  - 서기 계통(`hang_stand_env_cfg.py:21,27` → `Stage4V2Stand…` → `MarchDRv5`)은 `gait_clock freq=1.1`을 상속하는 것으로 보인다(`stage4_v2_env_cfg.py:511`, `mdp_stage4.py:36-41`, 명령과 무관하게 자유 진행).
  - 배포 러너의 `CKPT_GAIT_FREQ` 값은 확인하지 않았다.
- **가설**: 서기 정책에 의미 없는 1.1 Hz 주기 입력이 들어가 주기 가진이나 위상 고정원이 될 수 있다. 0.5 Hz와 2 Hz가 0.55와 2.2의 근처라는 것은 **증거가 아니다.**
- **대조 실험**:
  - (a) 기존 실물 로그에서 클록 위상과 pitch rate의 코히어런스를 본다.
  - (b) 같은 seed와 같은 지연 조건에서 클록 고정 (0,1)과 자유 진행을 actor 관측에만 적용해 비교한다.
  - 둘 중 하나라도 유의하면 정지 시 클록을 0으로 두고 재학습하는 안을 검토한다.

**⑤ [PD 게인] 발목 소프트, kd/kp 비**
- **출처**: 동봉 walk_default는 ankle 40/2, knee 150/4, hip 100/2다. mimic은 kp=armature·(2π·10)², ζ=2로 kd/kp=0.064 s다. 우리는 150/4로 0.027 s다.
- **근거**: 펌웨어 PD는 소프트웨어 지연이 없는 유일한 되먹임 경로다.
- **가져올 것**: posture_probe나 시뮬에서 발목 kp를 낮추거나 kd를 높이는 A/B를 해보고, 흔들림 성장률을 비교한다.
- **주의**:
  - 로봇이 다르다(35.7 kg 전신, 저유격 QC 액추에이터).
  - 발목 kp를 낮추면 균형 토크 권한이 줄어든다.
  - 백래시 데드존 안의 운동은 로터 쪽 엔코더 기반 kd로 감쇠되지 않는다 **(추정)**.
  - 게인을 바꾸면 반드시 같은 게인으로 재학습해야 한다(정책-게인 계약).

**⑥ [행잉 sim2sim 하네스] 배포 코드를 그대로 시뮬에 붙이는 구조**
- **출처**: `[A]/ai_sapiens_mujoco/src/mujoco_simulation.cpp`, `…/mujoco_hardware_interface.cpp:162-182`.
- **가져올 것**: 실물용 스택(브리지 + stage8 + 콘솔)을 가짜 CAN/HW 계층 뒤 물리 시뮬에 붙여 종단 지연·인계·상태기계를 실기 전에 검증한다.
- **주의**:
  - weld는 6자유도 강체라 로프 이완이나 뒤꿈치 회전을 재현하지 못한다. 로프가 필요하면 길이 제한 spatial tendon(단방향 구속)이 대안이다(**제안**, 저장소에는 없음).
  - 이 sim2sim에는 지연·노이즈 주입이 없다.

**⑦ [콘솔·안전 상태기계]**
- **출처**: §6.2의 파일들.
- **가져올 것**:
  - failsafe 우선순위 표
  - stale 래치
  - 액션 한계 거부 후 안전모드
  - 3회 추론 실패 규칙
  - 상승 에지 재진입과 하트비트 sequence 증가 검사
  - 단일 슬롯·epoch 요청 게이트: 중복 클릭이나 오래된 응답 방지에 해당하며 CLAUDE.md 콘솔 규칙과 직결된다
  - mock HW smoke test
- **주의**:
  - 자세 불량 2.0 rad는 우리에게 너무 느슨하다.
  - damping 모드(kp=0, kd=3)를 회수 순서 규칙(줄 인장 → 스택 종료 → release_all)이나 회생 과전압 이력과 대조하지 않고 그대로 도입하면 안 된다.

**⑧ [인계]** 같은 틱에 진입과 추론, 이력은 현재 관측으로 채우기, last_action=0, 3 s 선형 자세 보간(`policy_runtime.cpp:88-95`, `mode_controller.cpp:1048-1109`). 이들은 명령 공백을 없애는 패턴이다. 다만 이 스택은 목표 블렌딩이 없어 목표 불연속이 남는다.

**⑨ [재현성·계약]** 학습 시작 시 해석된 env에서 배포 계약 YAML을 자동 export하고, 기동 시 크기를 대조하고, 실행마다 git diff를 저장한다(`[C]/source/utils/*.py`). 흔들림에 대한 영향은 작다.

**⑩ [커리큘럼·DR 세부]**
- mimic의 실패 기반 시작점 샘플링을 행잉·착지 초기조건(착지 기하, 줄 길이 등) 샘플링으로 옮기는 안.
- 푸시에 roll·pitch 각속도 킥(±0.52 rad/s, 1~3 s 간격)이 들어가는지 우리 설정을 대조한다.
- encoder_bias를 "액추에이터 목표에서 빼는" 방식은 영점 오차 모델로 깔끔하다(±0.015 rad).

### 적용 불가 또는 근거 없음
- **백래시·스틱션**: 모델이 없다. MJCF frictionloss와 damping이 0이고 mjlab 액추에이터에도 데드존 모델이 없다. `LearnedMlpActuator`는 가능성은 있지만, 로터 쪽 엔코더만 있는 우리 구조에서는 관절 쪽 실측이 없어 백래시를 학습시킬 데이터가 없다 **(추정)**.
- **무릎 탄성 처짐(10~12°)**: 직렬 탄성 모델이 없다.
- **서기·행잉→착지 태스크, 평가 하네스**: 없다. play는 노이즈와 푸시만 끈 시각화용이다.
- **접촉 비트**: cyclo actor에는 접촉 비트가 없다(critic 전용).
- **프레임워크**: MJLab/MuJoCo-Warp라 Isaac Lab 2.3.2 설정에 그대로 넣을 수 없다. 개념만 이식할 수 있다.
- **PPO 하이퍼파라미터**: 새로운 정보가 없다.
- **동봉 정책**: K1 23 DoF 전용이다.
- **"이 velocity 설정이 실물에서 검증됨"**: 근거가 없다(§2.4 불일치).
