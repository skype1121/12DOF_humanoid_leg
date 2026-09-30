## unitree_rl_lab 조사 보고

**기준:** `main` 4960b84 (2025-11-19)
**클론 위치:** `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/unitree_rl_lab`
같은 `oss/` 폴더에 보조 자료로 `unitree_rl_lab_hist`(전체 이력), `unitree_sdk2`(dds_wrapper만 sparse), `unitree_mujoco`를 받아 두었다.

- 아래에서 `U/`는 `source/unitree_rl_lab/unitree_rl_lab/`를 뜻한다.
- Isaac Lab 함수의 실제 동작은 로컬 `/home/ryu/IsaacLab`(v2.3.2)에서 대조했다.
- 우리 저장소는 읽기만 했고 수정하지 않았다.
- 표기: (계산)은 계산값, (추정)은 코드에서 직접 읽지 않고 추론한 내용이다.

---

### 1. 요약

- **정체:** Unitree 공식 Isaac Lab 확장. Apache-2.0 라이선스, README 배지 기준 Isaac Sim 5.1.0 / Isaac Lab 2.3.0.
- **학습 태스크:** G1-29dof·H1·Go2 속도 추종 3종, G1 BeyondMimic 모션 추종 2종. 알고리즘은 rsl_rl PPO.
- **배포:** C++로 작성. unitree_sdk2 DDS와 ONNX Runtime 1.22 위에 Isaac Lab의 관측·액션 매니저를 재구현한 범용 러너와 조이스틱 FSM이 있다. G1-23dof·H1-2·B2·Go2W는 배포 코드만 있고 학습 cfg는 없다.
- **sim2sim:** 같은 실행 파일을 unitree_mujoco에 DDS로 연결해서 돌린다.
- **규모:** 파일 160개, 커밋 77개(2025-06~2025-11). main은 2025-11-19 이후 멈춰 있다.
- **활성 브랜치:**
  - `isaaclab3.0`(2026-05-25): Newton 백엔드 이식.
  - `dev/add_new_robots`(2026-05-18): R1 추가, G1에 UnitreeActuator 적용.
- **사전학습 정책:** G1 velocity v0와 mimic 2개뿐이다. 테스트 코드는 없다.
- **성숙도:** "템플릿 + 레퍼런스 배포" 수준이다. 보상과 DR은 Isaac Lab 기본 locomotion 수준이다. 지연·백래시·게인 DR은 어느 브랜치에도 없다.
- **핵심:** 가치 있는 부분은 학습 쪽이 아니라 배포 루프 구조다.
- **호환성:** 우리 Isaac Lab 2.3.2에 `DelayedPDActuator`, `reset_joints_by_scale`, 보상 함수들이 모두 있음을 확인했다. 확인한 함수 범위 안에서는 그대로 가져다 쓸 수 있다.

### 2. 아키텍처 (경로)

- **학습 스크립트:** `scripts/rsl_rl/train.py`
  - hydra cfg → `RslRlVecEnvWrapper` → `OnPolicyRunner` 순서로 실행된다.
  - 학습 시작 시 `params/{env,agent,deploy}.yaml`을 저장한다(194-201).
  - `learn(..., init_at_random_ep_len=True)`로 학습한다(204).
- **재생·내보내기:** `play.py`가 jit/onnx를 정규화기 포함으로 내보낸다(149-152). `--real-time` 옵션이 있다(176-179).
- **태스크 등록:** `U/tasks/locomotion/robots/{g1/29dof,h1,go2}/__init__.py`에서 `Unitree-G1-29dof-Velocity` 등으로 등록한다.
- **환경 cfg:** 로봇별 `velocity_env_cfg.py` 한 파일에 전부 들어 있다. Scene, Event, Command, Action, Obs(policy/critic), Reward, Termination, Curriculum, `__post_init__`(dt·decimation).
- **MDP 함수:** `U/tasks/locomotion/mdp/{rewards,observations,curriculums}.py`에 커스텀 함수가 있고, Isaac Lab mdp를 재수출한다(`mdp/__init__.py:1-7`).
- **자산:** `U/assets/robots/unitree.py`(`joint_sdk_names` = SDK 모터 순서)와 `unitree_actuators.py`.
- **액션:** 목표각 = 기본자세 + 0.25·a (G1 cfg 183-185). mimic은 관절별 scale을 0.25·τmax/kp로 둔다(unitree.py:706-717).
- **sim→배포 계약:** `U/utils/export_deploy_cfg.py:22-117`이 다음을 yaml로 내보낸다.
  - joint map, step_dt, kp/kd, default_joint_pos
  - 액션 scale/offset/clip
  - 관측 항목 순서·scale·clip·history
  - C++ 쪽 `deploy/include/isaaclab/envs/manager_based_rl_env.h:25-48`이 이 yaml을 그대로 읽는다.

### 3. 보상 (커맨드가 0이면 서기)

수식은 Isaac Lab v2.3.2 원문으로 확인했다.
위치: G1 `U/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py:242-337`, H1 `.../h1/velocity_env_cfg.py:243-337`, Go2 `.../go2/velocity_env_cfg.py:273-341`.

| 항목 | 수식 | G1 | H1 | Go2 |
|---|---|---|---|---|
| track_lin_vel_xy_exp | exp(−‖v_cmd−v_xy‖²/0.25) (G1·H1은 yaw 프레임) | +1.0 | +1.0 | +1.5 |
| track_ang_vel_z_exp | exp(−(ωz_cmd−ωz)²/0.25) | +0.5 | +0.5 | +0.75 |
| is_alive | 1 | +0.15 | +0.15 | – |
| lin_vel_z_l2 | v_z² | −2.0 | −2.0 | −2.0 |
| ang_vel_xy_l2 | ωx²+ωy² (rad/s, scale 무관) | −0.05 | −0.5 | −0.05 |
| joint_vel_l2 / joint_acc_l2 | Σq̇² / Σq̈² | −1e-3 / −2.5e-7 | – / −2.5e-7 | −1e-3 / −2.5e-7 |
| action_rate_l2 | Σ(a_t−a_{t−1})² | −0.05 | −0.05 | −0.1 |
| joint_pos_limits | soft limit(0.9) 초과량 | −5 | −5 | −10 |
| energy / torques_l2 | Σ\|q̇\|\|τ\| / Στ² | −2e-5 / – | – | −2e-5 / −2e-4 |
| flat_orientation_l2 | g_x²+g_y² | −5.0 | −1.0 | −2.5 |
| base_height_l2 | (z−z*)² | −10 (0.78 m) | −10 (0.9 m) | – |
| joint_deviation_l1 | Σ\|q−q0\| | 힙 roll/yaw −1, 허리 −1, 팔 −0.1 | 힙·몸통·팔 −1 | – |
| joint_position_penalty (커스텀, rewards.py:67-76) | ‖q−q0‖, 정지 시 ×5 | – | – | −0.7 |
| feet_gait (커스텀, 174-200) | 위상과 접촉 일치 수 (주기 0.8/0.6 s) | +0.5 (커맨드>0.1일 때만) | +0.5 (항상) | – |
| foot_clearance_reward (120-128) | exp(−Σ(z−z*)²·tanh(2‖v_xy‖)/0.05) | +1.0 | +20 | – |
| feet_slide / undesired_contacts | Σ‖v_foot‖·접촉 / 발 외 접촉 수 | −0.2 / −1 | −0.2 / −1 | −0.1 / −1 |

**Go2 자세 패널티의 핵심 코드:**
```python
torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)
```
`stand_still_scale=5.0`, 속도 임계 0.3 m/s. 멈춰 있으면 패널티를 5배로 키운다.

**서기와 흔들림에 관한 관찰:**

- **게이팅 구조:**
  - 이벤트형 보너스(회복 보너스 등)가 없다.
  - "균형을 잃으면 패널티를 끈다"는 식의 게이팅도 없다.
  - 게이트는 두 가지뿐이다. G1은 커맨드가 0.1 미만이면 gait 보상을 끈다. Go2는 정지 상태에서 자세 패널티를 5배로 키운다.
- **정숙 신호:** 커맨드가 0일 때의 정숙 신호는 사실상 exp 속도 추종 항(std 0.5)이다.
  - (계산) 사인파 흔들림의 평균 보상 손실:

    | std | V=0.05 m/s | V=0.1 | V=0.2 | V=0.3 |
    |---|---|---|---|---|
    | 0.5 | 0.5% | 2.0% | 7.5% | 15.8% |
    | 0.15 | 5.3% | 18.9% | 50% | 69% |
    | 0.1 | 11% | 36% | – | – |

  - 즉 std 0.5로는 느린 소진폭 흔들림을 거의 벌하지 못한다.
- **G1 보상 이력(커밋 4dfd1af, 2025-07-03):**
  - ang_vel_xy −0.5 → −0.05 (10배 **축소**)
  - action_rate −0.01 → −0.05
  - energy 항 추가
  - 정책 관측에서 `gait_phase`(클럭) 제거
  - 팔 편차 −1 → −0.1
- **(계산) 피치 흔들림에 대한 ang_vel_xy_l2 스텝당 패널티:**
  - 2 Hz·±3°: 0.011 (w 0.05), 0.108 (w 0.5)
  - 0.5 Hz·±2°: 0.0003 (w 0.05), 0.003 (w 0.5)
  - L2 각속도 항은 가중치와 상관없이 느린 모드를 거의 보지 못한다.
- **foot_clearance_reward:** 발이 수평으로 움직이지 않으면 exp(0)=1로 최대가 된다. 결과적으로 "발 정지 보너스" 역할을 한다(미병합 PR #80이 지적).
- **메인테이너 발언(#34):** G1이 서 있을 수 있는 이유는 "disable gait when no cmd"라고 설명했다.
- **Mimic 보상(tracking_env_cfg.py:219-276):** exp 추적 6종(w 0.5~1.0, std 0.3~3.14), action_rate −0.1, torque −1e-5, joint limit −10.
- **종료 조건:** G1은 z<0.2 m 또는 기울기>0.8 rad (344-346).

### 4. DR·액추에이터

**시뮬 설정**
- dt 0.005 s(200 Hz), decimation 4 → 정책 50 Hz.
- 에피소드 20 s, mimic은 30 s.
- PhysX solver pos 8 / vel 4, 자기충돌 on, soft limit 0.9 (unitree.py:24-47).

**지연: 없음**
- `UnitreeActuator`는 `DelayedPDActuator`를 상속한다. 하지만 `min_delay/max_delay`의 기본값이 0이다(Isaac Lab `actuators/actuator_pd_cfg.py:57-61`).
- 전 브랜치에서 grep했지만 이 값을 설정하는 태스크가 없다.
- 관측 지연과 IMU 지연도 없다.

**액추에이터 구성**
- **G1 velocity:** `ImplicitActuatorCfg` (unitree.py:397-510), armature는 모두 0.01.

  | 그룹 | kp | kd | 최대 토크 | 최대 속도 |
  |---|---|---|---|---|
  | hip pitch/yaw | 100 | 2 | 88 Nm | 32 rad/s |
  | hip roll / knee | 100 / 150 | 2 / 4 | 139 Nm | 20 rad/s |
  | **ankle** | **40** | **2** | 25 Nm | 37 rad/s |

- **H1:** `IdealPDActuatorCfg`(명시 PD). knee 200/4, hip 150/2, ankle 40/2 (231-273).
- **G1 mimic (BeyondMimic 방식):** kp = J·(2π·10)², kd = 2·2·J·(2π·10) (513-529).
  - knee 99.1/6.31, hip pitch 40.2/2.56, ankle 28.5/1.81.
- **정리:** Unitree 휴머노이드의 발목 kp는 무릎의 약 1/4~1/5이다.
- **UnitreeActuator (unitree_actuators.py:11-88):**
  - 명시 PD 뒤에 마찰을 뺀다: `applied_effort -= Fs*tanh(q̇/Va) + Fd*q̇` (65-67).
  - 비대칭 토크-속도 포락선을 적용한다. 같은 방향이면 Y1, 제동이면 Y2, |q̇|>X1부터 X2까지 선형 감소 (75-88).
  - armature 공식: `J2 + J1·i2² + Jr·(i1·i2)²` (40-41).
- **dev 브랜치 추가분:**
  - `motor_offset`(발목 0.03 rad): PD가 `joint_pos + offset`을 보게 한다.
  - Fs/Fd/offset 기본값을 "for DR"로 보관하지만, 실제 랜덤화 이벤트는 없다.
- **메인테이너 발언(#14):** "Implicit은 PD를 시뮬 내부에서 풀어서 실물과 다를 수 있다. 교체 예정이다."

**DR 범위**
- **G1 velocity (92-156):**
  - 마찰 0.3–1.0
  - torso 질량 −1…+3 kg
  - 푸시: 루트 속도에 x,y ±0.5 m/s를 **정확히 5 s마다** 더함
  - 리셋: 위치 ±0.5 m, yaw ±π
  - 관절은 기본자세 그대로. `velocity_range (-1,1)`은 기본 속도 0에 곱해지므로 효과가 없다(로컬 코드 확인).
  - 게인·CoM·오프셋 DR은 없다.
- **mimic (177-215):**
  - 마찰 0.3–1.6 / 0.3–1.2
  - 관절 기본자세 ±0.01 rad(액션 offset도 같이 이동, `mimic/mdp/events.py:15-52`)
  - torso CoM x ±2.5 cm, y·z ±5 cm
  - 푸시 1–3 s마다. 선속도 ±0.5/±0.5/±0.2에 더해 **roll/pitch 각속도 ±0.52 rad/s**, yaw ±0.78 rad/s (32-39)

### 5. 관측

- **G1 actor (197-208):** ang_vel×0.2, gravity, cmd, joint_pos_rel, joint_vel_rel×0.05, last_action.
  - history 5, 항목별로 쌓는다.
  - **클럭 없음.**
  - 차원은 480 (계산).
- **G1 critic:** actor 항목 + base_lin_vel, 노이즈 없음, history 5, 495차원 (계산). 접촉 관측은 없다.
- **노이즈:** 균등 가산, raw 단위, scale 적용 전. Isaac Lab 처리 순서는 noise→clip→scale이다(`observation_manager.py:354-357, 392-404`).
  - 각속도 ±0.2 rad/s, 중력 ±0.05, 관절각 ±0.01 rad, 관절속도 ±1.5 rad/s.
- **정규화:** `empirical_normalization=False`. 수동 scale만 쓴다.
- **H1/Go2:** history 없음. H1은 gait_phase(sin/cos, 0.6 s)를 관측한다.
- **C++ 배포 쪽 관측 처리:**
  - 이력은 항목 우선으로 배치한다(과거→현재). Isaac Lab과 같다(`observation_manager.h:63-92`).
  - clip 후 scale 순서다(`manager_term_cfg.h:32-51`).
  - 리셋 시 이력 전체를 현재 관측으로 채운다(27-30). Isaac Lab CircularBuffer의 첫 push 동작과 같다(로컬 확인).

### 6. 배포

**구조: C++ 단일 프로세스, 모든 경로가 "최신값 한 칸"**

1. **수신:** DDS 콜백이 `msg_`를 mutex로 덮어쓰고 `last_update_time_`을 기록한다(unitree_sdk2 `dds_wrapper/common/Subscription.h:28-43`). 큐에 쌓아 두고 소비하는 경로가 없다.
2. **FSM/송신 스레드 1 kHz** (`deploy/include/FSM/CtrlFSM.h:48-57, 80`, dt=0.001):
   - 매 틱 상태 갱신 → `run()` → `lowcmd->unlockAndPublish()` → 전이 검사 (82-113, `FSMState.h:56-65`).
3. **정책 스레드 50 Hz** (`FSM/State_RLBase.h:29-46`): 절대 시각 기준으로 스케줄한다.
   ```cpp
   while (policy_thread_running) { env->step(); std::this_thread::sleep_until(sleepTill); sleepTill += dt; }
   ```
   - `step()` 순서: 최신 상태 복사(`unitree_articulation.h:21-41`) → 관측 → ONNX 추론(`algorithms.h:29-93`) → scale·offset·clip(`joint_actions.h:39-60`).
4. **송신:** `RealTimePublisher`(ros-controls realtime_tools 패턴, `Publisher.h:36-162`)가 최신 msg만 복사해서 보낸다. 새 메시지가 없으면 1 ms 간격으로 폴링한다.

**PD와 지연**
- `run()`은 최신 목표를 모터 q에 복사한다(`robots/g1_29dof/src/State_RLBase.cpp:55-61`).
- kp/kd는 yaml 값을 쓰고 dq=0, tau=0이다. 즉 PD는 모터 드라이버에서 돈다.
- (추정) 소프트웨어 지연은 상태 나이 + 추론 + ≤1 ms + ≤1 ms로 수 ms 수준이다. 학습에서 지연 DR 없이 버티는 전제가 이 구조일 것이다.
- **메인테이너 발언(#31):** "It's not recommended to deploy a locomotion policy by python."

**FSM과 안전장치 (`deploy/robots/g1_29dof/config/config.yaml`)**
- **Passive:** kp 0, kd 3. 매 틱 명령 q를 측정 q로 맞춘다(`State_Passive.h:25-45`).
- **FixStand:** 현재 명령 q에서 목표 자세로 3 s 선형 보간한다(`State_FixStand.h:20-51`).
  - 목표 자세와 kp/kd가 정책의 default_joint_pos·gain과 **동일**하다(config.yaml:39-62 vs `policy/velocity/v0/params/deploy.yaml:4-11`). 그래서 홀드→정책 인계에 불연속이 없다.
- **워치독:**
  - 기울기 >1.0 rad → Passive (`State_RLBase.cpp:47-52`)
  - 상태 1 s 무수신 → Passive (`FSMState.h:47-53`, `Subscription.h:76`)
- **기동 점검:** `rt/lowcmd`를 0.2 s 청취해서 다른 명령 주체가 있으면 거부한다(`main.cpp:11-26`). 로봇 형식도 확인한다(41-45).
- **전이 조건:** 조이스틱 DSL(`unitree_joystick_dsl.hpp`)로 기술한다.

**복사하면 안 되는 결함**
- (a) `_processed_actions`가 0으로 초기화된다(`joint_actions.h:26-27`). 정책 첫 추론 전에 FSM이 `run()`을 돌면 q=0이 송신될 수 있다. 재진입 시에는 오래된 목표가 나갈 수 있다(추정 경합).
- (b) 두 스레드가 action을 잠금 없이 공유한다(추정).
- (c) `trylock` 없이 `unlockAndPublish`를 호출한다.
- (d) 액션 필터나 슬루 제한이 없다.
- (e) **export 버그:** `export_deploy_cfg.py:36`이 env 0의 **랜덤화된** default_joint_pos를 내보낸다.
  - 흔적: mimic `deploy.yaml:10`의 hip pitch가 −0.302/−0.319다(명목값 −0.312).
  - dev 브랜치에서 `default_joint_pos_nominal`로 수정됐다.

**sim2sim (MuJoCo)**
- `g1_ctrl --network lo`로 unitree_mujoco에 붙인다(README:110-133).
- 브리지는 1 kHz로 τ = τff + kp·Δq + kd·Δq̇를 계산한다. 지연 모델은 없다(`unitree_mujoco/simulate/src/unitree_sdk2_bridge.h:168-230`).
- **Elastic band (하네스 대용):** 고정점 (0,0,3)과 몸통 사이의 스프링-댐퍼(`main.cc:53-84, 488-500, 634-641`).
  - k=200 N/m, c=100 N·s/m
  - 7/8 키로 길이 ±0.1 m 조절, 9 키로 해제
  - 힘만 가하고 토크는 없다. 압축되면 밀어내며 느슨해지지 않는다.
  - 절차: FixStand → 8 키로 내려 발 접지 → 정책 시작 → 9 키로 밴드 해제.

### 7. PPO·커리큘럼

- **PPO (`U/tasks/locomotion/agents/rsl_rl_ppo_cfg.py:11-36`):**
  - 4096 env, 24 steps/env, 50000 iter
  - [512,256,128] ELU, init_noise_std 1.0
  - clip 0.2, entropy 0.01 (mimic 0.005)
  - 5 epochs, 4 minibatches
  - lr 1e-3 adaptive (KL 0.01), γ 0.99, λ 0.95, grad_norm 1.0
  - 대칭 손실 없음. `joint_mirror` 보상은 정의만 있고 쓰지 않는다.
- **커리큘럼 (`curriculums.py:11-37`):**
  - 에피소드마다 한 번, 추적 보상이 0.8×weight를 넘으면 x,y 커맨드 범위를 ±0.1씩 넓힌다(초기 ±0.1 → x −0.5…1.0, y ±0.3).
  - 커맨드 재샘플은 10 s마다. `rel_standing_envs=0.02`라서 재샘플마다 2% 확률로 커맨드가 정확히 0이 된다(Isaac Lab velocity_command.py:141, 162-163).
  - mimic은 실패율 기반 적응 샘플링을 쓴다(commands.py:207-241).

### 8. 우리에게 가져올 것 (우리 문제 기준, 영향 순)

**1) 배포 루프의 "최신값 한 칸" 구조 — 지연 대응, 영향 최대**
- **원본:** `Subscription.h:28-51`, `Publisher.h:36-162`, `CtrlFSM.h:48-113`, `State_RLBase.h:15-47`
- **우리 현황:**
  - `jetson/scripts/rl_bridge_node.py:1963-1969`: joint_states/status/cmd는 depth 10, IMU/foot는 depth 50
  - `stage8_12axis_mit_control_node.py:1425-1436`: depth 10, 50 Hz 제어 타이머, 100 Hz status 타이머
  - 둘 다 단일 스레드 `rclpy.spin`
- **가져올 것:**
  - (a) 제어 경로를 KEEP_LAST depth=1로 바꾸고, 콜백은 "덮어쓰기 + 수신 시각 기록"만 하게 한다.
  - (b) 정책 틱마다 사용한 샘플의 나이를 기록하고 가드를 건다.
  - (c) 50 Hz 타이머 두 개가 비동기로 돌면서 생기는 위상 지연(최대 20 ms, 추정)을 없앤다. 방법: stage8에서 수신 즉시 송신, 송신 주기 상향, 또는 정책과 CAN을 같은 프로세스에 두고 스레드만 분리.
  - (d) 절대 시각 스케줄을 쓴다.
- **(계산) 순수 지연의 위상 지연:** 330 ms면 0.5 Hz에서 59°, 2 Hz에서 238°. 50 ms면 9°/36°.
  - 관측된 0.5 Hz 증가 진동, 2 Hz 흔들림과 정합한다. 다만 원인 주장에는 지연만 바꾼 대조 시험이 필요하다.
- **주의:** 서비스 변경과 실물 적용은 별도 승인 대상이다. GIL 때문에 추론과 CAN 지터를 확인해야 한다. 송신 주기 상한은 CAN 대역에 달려 있다(추정).

**2) 배포 스택을 수정 없이 돌리는 sim2sim — 지연 진단과 회귀 시험**
- **원본:** README:110-133, `unitree_sdk2_bridge.h:168-230`
- **가져올 것:** 가짜 CAN/joint_states/IMU 브리지(MuJoCo 또는 Isaac)를 만든다. rl_bridge_node와 stage8을 그대로 붙여서, 목표→관절 지연과 흔들림을 실제 소프트웨어 타이밍으로 재현한다. 현재 Isaac 평가기로는 파이프라인 적체를 볼 수 없다.
- **주의:** elastic band는 우리 로프 모델(느슨함·재인장)보다 단순하다. MJCF가 필요하다.

**3) 서기 보상 패턴**
- (a) 이벤트 보너스를 두지 않고 패널티를 항상 켠다. 우리의 회복 보너스 제거와 게이트 해제를 A/B 후보로 둔다.
- (b) Go2식 "정지 시 ×5 강화" 게이트(rewards.py:67-76, go2 cfg 293-301). 정지 상태에서 패널티를 끄는 우리 방식과 반대 방향이다.
- (c) 영커맨드에 대한 exp 속도 추종을 정숙 신호로 쓴다. 단, std는 우리 흔들림 규모에 맞춰 0.1~0.15로 잡는다(위 계산표).
- (d) G1은 클럭 관측이 없다(4dfd1af). 우리 정책은 클럭을 관측하므로, 클럭 고정/제거 대조 평가 후보로 둔다(가설).
- **주의:** Unitree가 흔들림 억제용 보상을 따로 설계한 증거는 없다. 오히려 ang_vel 가중치를 줄였다. 보상 변경은 물리·관측 변경과 분리해서 같은 조건의 기준선과 비교해야 한다.

**4) 명시 PD 대 Implicit PD 대조 — 시뮬이 실물보다 순한 문제**
- **원본:** `unitree_actuators.py:11-118`, 이슈 #14
- **우리 현황:** 기본 자산이 ImplicitActuatorCfg다(`rl_walking/biped12_cfg.py:73,84`).
- **가져올 것:** 같은 정책을 명시 PD(200 Hz, 토크-속도 포락선, Fs·tanh+Fd 마찰) 플랜트에서 평가한다. 감쇠가 실물 서명 쪽으로 가는지 본다.
- **주의:**
  - Deadzone(백래시형)·Coulomb 액추에이터는 우리 `rl_walking/actuators_play.py`에 이미 있다. Unitree 모델에서 새로 얻는 것은 포락선과 물리스텝(5 ms) 단위 지연 정도다.
  - AK70-10/AK45-36 제원은 이 레포에 없다.
- **참고 데이터(가설, 재학습 필요):** Unitree는 발목 kp를 40으로 둔다(무릎 150~200). "강한 발목 PD + ±4° 백래시 + 지연"은 전형적인 한계 사이클 조건이다.

**5) 인계와 안전 패턴 — 콘솔 규칙과 직결**
- (a) 기동 전 다른 명령 주체의 송신 여부를 확인한다(`main.cpp:11-26`).
- (b) 무충격 인계: Passive에서 명령 q = 측정 q를 유지한다. FixStand의 최종 자세·게인을 정책 기본값과 같게 둔다.
- (c) 상태 신선도 워치독을 둔다.
- **복사 금지:** 6장의 결함 (a)~(d). Passive(kp 0)로 즉시 전환하는 동작은 우리 '굽힌 관절 즉시 릴리즈 금지' 규칙과 충돌한다.

**6) 배포 계약 교훈**
- 관절 오프셋 DR을 넣는다면 명목값을 export해야 한다(main의 버그).
- 이력 배치 방식(항목 우선)과 리셋 시 이력 채우기 방식을 우리 계약과 대조한다.

**7) 피치 흔들림을 직접 자극하는 DR**
- **원본:** mimic 푸시의 pitch 각속도 ±0.52 rad/s, 1–3 s 간격(tracking_env_cfg.py:32-39, 209-215).
- **주의:** 크기는 G1 기준이다. 기본 서기와 외란 내성 평가는 분리해야 한다. 지연 모델이 틀린 상태에서 푸시만 늘리면 효과를 해석할 수 없다.

**레포에 없어서 가져올 수 없는 것**
- 학습 측 지연·지터·관측 지연 모델. 우리 `rl_walking/actions.py:28-49`가 이미 앞서 있다.
- 백래시·로터측 엔코더 오차·무릎 탄성 처짐 모델.
- 게인·모터 강도·IMU 편향 DR.
- 접촉 관측과 압력센서 처리.
- 매달림→착지 학습 환경.
- 서기 전용 태스크. 속도 태스크의 2% 영커맨드 env가 전부다.
- 새로운 PPO 기법이나 대칭 손실.
- 전용 가능한 사전학습 가중치. 있는 것은 G1 전용이다.
