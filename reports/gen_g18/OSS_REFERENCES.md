# 참고한 오픈소스·논문 기록 (G17–G20, 2026-10-02)

목적: 어떤 외부 자료의 **어느 부분**을 보고, 우리 설계·판정에 **어떻게** 반영(또는 기각)했는지 남긴다.
조사는 1002 04시 분석 에이전트 5기(주제별)가 수행했다. 주제별 분석 요약과 그 위에서 돌린 진단 결과는 이 문서의 **부록 1–5** 에 있다(참고 자료는 이 문서 하나로 관리).
저장소는 `git clone --depth 1` 로 읽기만 했다(스크래치 경로, 저장소에 사본 없음). 파일:줄은 아래 커밋 기준.

## 1. 저장소

| 저장소 | URL | 커밋 (날짜) | 참고한 부분 (파일:줄) | 우리 쪽 반영 |
|---|---|---|---|---|
| smooth-humanoid-locomotion (LCP) | github.com/zixuan417/smooth-humanoid-locomotion | 004f670 (2025-06-16) | `rsl_rl/algorithms/ppo_rma.py:156-159` `_calc_grad_penalty`(∇_obs log π 제곱 벌점), `:174-190` 계수 스케줄, `:237` 손실 결합; `g1_walk_phase_config.py:262` λ=0.002, `:148-176` action_rate 항 없음; `humanoid_config.py:298-309` RMA 특권 잠재 | **G18j** SmoothPPO(`rl_walking/smooth_ppo.py`) — 전역 대신 **last_action 블록 한정** 야코비안 벌점으로 변형. 결과: 떨림 0 이나 이득 전반 축소·느린 흔들림(H-j 잠정 지지, 배포 불가) |
| rsl_rl (최신) | github.com/leggedrobotics/rsl_rl | 857de61 (2026-09-09) | `algorithms/ppo.py:405` load 시 학습률 복원(설치본 3.1.2 에는 없음), `modules/distribution.py:147-150,187-190` std 범위·고정, `extensions/symmetry.py:24-26,66-118`, `modules/normalization.py:18`, `algorithms/distillation.py:44-47` | `train.py --restore_lr` 옵션 추가(미사용). 설치본 3.1.2 의 `ppo.py:313,364-365`(손실·zero_grad), `on_policy_runner.py:422`(eval(class_name)) 를 보고 SmoothPPO 주입 지점 결정 |
| legged_gym | github.com/leggedrobotics/legged_gym | 8fa29ac (2025-05-29) | `legged_robot_config.py:138-145`(action_rate −0.01 등), `:166-175` 관측 잡음(자이로 ±0.2, 관절속도 ±1.5), push 15 s ±1 m/s, `legged_robot.py:226,554` | **G19n** 관절속도 잡음 ±0.3→±1.5 의 근거 수치 → 결과: 정숙하나 회복 상실(b18 90 %), 승격 금지. 우리 action_rate 가 legged_gym 의 20배임을 확인 |
| unitree_rl_gym | github.com/unitreerobotics/unitree_rl_gym | 276801e (2025-07-25) | `g1_config.py:79-83,93-106`(보상·PPO), `deploy_mujoco.py:84-122`(sim2sim), `deploy_real.py:153-225`(배포 루프) | 위상 시계 상시(제자리걸음 기본) 구조 확인 → 무외란 발 들기 목표와 충돌해 **채택 안 함**. sim2sim 은 이상 관측·기본 접촉이라 3 Hz 튐 재현 용도로 부적합 판단 |
| humanoid-gym | github.com/roboterax/humanoid-gym | ae46e20 (2024-11-01) | `humanoid_env.py:194-196,209-251`(관측 조립·잡음), `:219-235,250-253` critic 특권(밀기 힘·마찰·질량·접촉 mask), `:530-540` action_smoothness, `humanoid_config.py:88-97,159-160,212,235-245`(entropy 0.001, lr 1e-5), `sim2sim.py:114-160` | critic 특권 정보 아이디어(G18 후보, **미착수**). 매끄러움 항 가중 비교 |
| isaac_berkeley_humanoid | github.com/HybridRobotics/isaac_berkeley_humanoid | ffc7b26 (2024-10-11) | push 커리큘럼(최대 3 m/s), feet_air_time 은 명령 0 이면 0 | 회복 스텝 보상 설계 비교(정지 서기에서 air_time 을 끄는 관행 확인) |
| FALCON | github.com/LeCAR-Lab/FALCON | a967a6d (2026-04-09) | `config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml:57-66`(지속 힘 ±40 N 삼각파), `config/obs/dec_loco/t1_29dof_obs_diff_force_history_wolinvel_ma.yaml:24-38,45`(critic 에 힘), `config/rewards/.../reward_dec_loco_stand_height_ma_diff_force.yaml:209-217` 힘 커리큘럼, `envs/.../...diff_force.py:590-608`; 서기 모드 한 발 들기 −4 | 지속 외력 설계 비교 → G17p(빠른 지속 외력) 근거 일부. **H-p 미지지**. critic 힘 특권·생존 기반 힘 커리큘럼은 G19+ 후보로 기록 |
| HIMLoco | github.com/OpenRobotLab/HIMLoco | ef289ac (2024-05-14) | `legged_robot_config.py:36-39,125-129,152-154`(외력 ±30 N·payload·CoM ±5 cm), `legged_robot.py:259,280,550-555`, `him_estimator.py:60-110`(이력 인코더·대조 학습) | critic 외력 특권·이력 추정기 → 액터 계약 변경이 커서 **보류** |
| HugWBC | github.com/InternRobotics/HugWBC | eb831af (2025-08-02) | `h1_config.py:261-284`(속도 푸시만, push_f_* 미사용), payload −3~+9 kg | 지속 외력 미사용 사례로 비교만 |
| HoST | github.com/InternRobotics/HoST | 70bb580 (2025-06-17) | `host_platform.py:246,457-459`(액션 지연 0–4 틱, 이력 6) | 지연 DR 범위 비교(G18d·G20d 설계 참고) |
| booster_gym | github.com/BoosterRobotics/booster_gym | da396a0 (2025-12-23) | `T1.yaml:147-249`(관측 잡음·접촉 compliance 0.5–1.5·restitution 0.1–0.9 DR), `t1.py:316,445,574-592`, `deploy.py:176`(배포만 EMA — 학습과 다른 계약) | 지면 연성·반발 DR 아이디어(G19+ 후보, 재생 재현 선행 조건). '배포만 필터' 계약 불일치 사례 → 우리 G8 이후 학습 LPF 부재와 같은 문제로 인식 → **G18f**(학습에 LPF 복원 + 필터 거친 명령 관측) |
| rl_sar | github.com/fan-ziqi/rl_sar | 376d42c (2026-08-16) | `rl_sdk.cpp:64-181`(관측 조립·clip_obs), `rl_sim_mujoco.cpp:107-108`, `rl_real_g1.cpp:72-73`(sim·실물 같은 클래스) | 러너 관측 클리핑 부재 발견(현재 무해, 추가 권고). 배포 스택을 시뮬에 그대로 연결하는 구조 참고 |
| IsaacLab | github.com/isaac-sim/IsaacLab | 32b02cc (2026-10-01) | `actuators/actuator_pd.py:310-363`·`actuator_pd_cfg.py:52-61`(DelayedPD), `envs/mdp/events.py:1159-1196`(apply_external_force_torque), `velocity_env_cfg.py:125-132,209-239,267-272`, `sensor_base_cfg.py:34`(update_period), `g1/agents/rsl_rl_ppo_cfg.py:14-36` | 우리 PPO 설정이 G1 보행 레시피 그대로임 확인. 관측 지연 기능 부재 확인 → 자체 구현(자이로 지연·접촉 샘플홀드). 설치본 `manager_based_rl_env.py:216-238` 리셋 경로 확인(접촉 기준 자세 결함 진단에 사용) |

## 2. 논문 (저장소 없음 또는 논문만 참고)

| 자료 | 참고한 부분 | 반영 |
|---|---|---|
| van Marum et al. 2024, *Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking* (arXiv 2404.19173) | 서기 명령에서 접촉 항 상수 1, "양발 접촉 보상은 회복 스텝을 벌한다", 20–200 N·0.2–0.5 s 외란 | **G18q**(외란 창 안 한 발 접촉 허용) 근거 → **H-q 미지지** |
| Ferigo et al. 2021, *On the Emergence of Whole-body Strategies from Humanoid Robot Push-recovery Learning* (arXiv 2104.14534) | "어느 한 발이라도 접지" 접촉 보상, 200 N×0.2 s 구면 균등(1.21 N·s/kg), CoM-지지다각형 항 | G18q 근거, 우리 충격 세기(최대 0.68 N·s/kg)가 절반 수준임을 확인 |
| LCP: arXiv 2410.11825 | Table I(떨림 42.2→3.21), λ 0.002 탐색 | G18j 계수 결정 참고 |
| CAPS: Mysore 2021, arXiv 2012.06644 | 시간·공간 매끄러움 손실 | 비교만(미채택) |
| SN-LCP: arXiv 2504.08246 | 스펙트럼 정규화 정책 | 이어 학습 곤란으로 미채택 |
| Gradient-based regularization: arXiv 2407.04315 | 기울기 정규화 일반 | 배경 |
| Tan et al. 2018 RSS (Minitaur sim-to-real), arXiv 2504.00614 (기어 휴머노이드 발 IMU), MMDR arXiv 2109.14549, arXiv 2512.01996 | 지연 DR 범위(0–40 ms 등), 모달리티별 지연 | 자이로 지연 0–3 틱 DR(G17) 범위 판단 배경. 원문 표는 미확인(요약 기준) |
| RL-Force (ICIRA 2025) | 외력·작용점 추정 | 공개 코드 없음, 수치 미확인 |
| HoST: arXiv 2502.08378 | 일어서기 과제 | 해당 없음 확인 |
| Baisero & Amato 2022 (비대칭 actor-critic 편향) | critic 특권 정보의 위험 | critic 특권 후보의 위험 항목으로 기록 |

## 3. 결과적으로 채택된 것 / 기각된 것

- **채택·지지**: 배포 필터를 학습에 넣고 관측도 필터 거친 값으로(Booster 의 '배포만 필터' 반면교사 → G18f, 두 seed 지지) → 관절속도도 같은 방식(G19v, 지지). 우리 진단(폐루프 관측 EMA 귀속)이 결정적이었고 OSS 는 방향 제시.
- **기각·미지지**: 위상 시계(unitree·humanoid-gym), 관절속도 잡음 확대(legged_gym 수치 → G19n 회복 상실), 한 발 접촉 보상(van Marum·Ferigo → G18q), 빠른 지속 외력 확대(FALCON 류 → G17p), 전역 기울기 벌점의 블록 한정판(LCP → G18j, 흔들림).
- **보류(후보)**: critic 특권 정보(humanoid-gym·FALCON·HIMLoco), 지면 연성 DR(Booster), 힘 커리큘럼(FALCON), sim2sim(unitree·rl_sar — 3 Hz 재현 용도로는 낮은 우선순위).


---
# 부록: 주제별 분석 요약 (에이전트 보고 + 후속 진단 결과)


## 부록 1 — 반응성 유지 + 고주파 떨림 억제
(원 파일 `oss_1_smoothness.md`, 1002 통합)
### 새 오프라인 발견 (CPU, 플랜트 제외 '자기루프' = 다른 관측 고정·출력만 last_action 이력으로 되먹임, 256 상태) — 가설
| ckpt | 자기루프 진동 env | 진동 RMS 중앙 | ‖∂μ/∂last_action‖²_F | ‖∂μ/∂자이로‖²_F |
|---|---|---|---|---|
| G12-9500 | 0.02 | 0.000 | 0.93 | 2.12 |
| G12c-10400 | 0.04 | 0.000 | – | – |
| G16d-10500/11499 | 0.84/0.88 | 0.10 | – | – |
| G17o-10100/10400/10700 | 0.94/1.00/0.90 | 0.11/0.18/0.12 | 5.59(10400) | 1.27 |
| G17r-10100~10700 | 0.18~0.30 | 0.000 | 1.37 | **0.45** |
- G17o 체크포인트 순서가 평가 부호반전 순서(0.392>0.315>0.290)와 일치 → 25 Hz 떨림 = last_action 경유 2주기 자기 진동(가설, 폐루프 대조 필요).
- G17r(×3)은 떨림을 줄였지만 자이로 이득도 G12 의 1/5 로 깎음(반응성 손실 우려, G15 와 같은 방향 가설). 학습 σ 0.27–0.57 이 action_rate 벌점 기대값의 85–90 % 차지(가설).
- 학습 중 LPF 만으로는 못 막음(G8b 0.253 vs G8d 0.303) — last_action 관측은 필터 앞 원시 행동.
- 우리 action_rate 가중 정지 −0.2/회복 −0.06 은 이미 legged_gym 의 20배.
### OSS
LCP(smooth-humanoid-locomotion ppo_rma.py:156-159 grad penalty λ=0.002, action_rate 항 없음; 떨림 42.2→3.21) / CAPS / 스펙트럼 정규화(이어 학습 곤란).
rsl_rl 3.1.2: ppo.py:313 손실, :364-365 zero_grad/backward, on_policy_runner.py:422 eval(class_name) → train.py 에서 opr.PPO 교체로 최소 변경.
### 제안
G18a last_action 블록(관측 165–224) 한정 야코비안 벌점(Hutchinson, λ=0.01) / G18b LCP 원형 λ=0.002 / G18c last_action 관측 잡음 ±0.1.
무비용 대조: 평가에서 last_action 관측을 지연/배포 명령 역변환으로 바꿔 떨림 소멸 여부.
사전검사(CPU): 자기루프 진동 ≤5 %, ‖J_act‖² ≤1.5, ‖J_gyro‖² ≥ G17o×0.7.

### 폐루프 대조 사전 기록 (1002 04:5x, 실행 전) — `--last_action_ema α`
G17o-10400 · eval seed 1 · 실물 계약 d1 · nd·sf10·b18 · α ∈ {0.1, 0.3} vs 없음. 대조 G12-9500 nd α0.1.
- H-loop(25 Hz 떨림 = last_action 경유 자기루프): α0.1 nd 부호반전이 0.392 → ≤0.08(−80 %) 이고 action-rate RMS ≤0.08 이면 지지.
  부호반전이 −30 % 미만 감소면 기각(다른 경로 — 관절속도·자이로).
- 보고: sf10·b18 낙상 변화(관측 분포 밖 입력이라 해석은 제한적).

### 폐루프 대조 결과 (1002 08:4x, G17o-10400 · seed 1 · 실물 계약 d1)
| last_action 관측 | nd 부호반전 | nd action-rate RMS | sf10 낙상 | b18 낙상 |
|---|---|---|---|---|
| 원래(G17o-10400 선별) | 0.392 | 0.255 | 18.8 % | 17.9 % |
| EMA α0.3 | 0.007 | 0.052 | ~17 % (19) | ~41 % (46) |
| EMA α0.1 | 0.002 | 0.047 | ~12.5 % (14) | ~70 % (78) |
- **H-loop 지지**: α0.1 에서 부호반전 −99 %(기준 −80 %), RMS 0.047(≤0.08). 25 Hz 떨림 = last_action 관측 경유 자기루프(폐루프 확인).
- 대가: 같은 경로가 **충격(b18) 회복에 쓰인다** — 느리게 할수록 b18 낙상 18 → 41 → 70 %. 지속 밀기(sf10)는 영향 없음.
  (분포 밖 입력이라 크기 해석은 제한적. 괄호 = 낙상 env 수 / 사전검사 통과 ~112.)
- G18j(블록 야코비안 벌점) 에 대한 예측(가설): 떨림 제거, 단 b18 회복이 G18o 보다 나빠질 위험 → G18j 판정에서 b18 을 핵심 관찰.


## 부록 2 — 회복 스텝/밀기 회복
(원 파일 `oss_2_push_recovery.md`, 1002 통합)
- 정지 서기에서 스텝을 얻은 사례: Ferigo 2021(iCub, 접촉 보상 = "어느 한 발이라도", 200 N×0.2 s 구면 균등 평균 5 s, 1.21 N·s/kg),
  van Marum 2024(Digit, 서기 명령에서 접촉 항 상수 1 — 양발 접촉 보상은 회복 스텝을 벌한다고 명시, 20–200 N 0.2–0.5 s).
  위상 시계 상시(unitree_rl_gym, humanoid-gym) 는 무외란 발 들기 목표와 충돌. FALCON 서기 모드는 한 발 떨어지면 −4(스텝 금지).
- 우리 진단(코드·tensorboard, 가설 포함):
  1. 게이트 열림 동안 양(+) 보상은 recovery_quality(+5) 뿐인데 `q *= both` → 스윙 중 0 = 스텝 구조적 손해(코드 확인).
  2. 학습 충격 5–30 N×0.2–0.4 s (최대 0.68 N·s/kg); 평가 뒤18N(5.4 N·s) 이상 뒤 충격은 슬롯의 ~6 %. push_robot ±0.25 m/s 는 평가 Δv 0.31 미만.
  3. feet_impact 문턱 180 N 은 12 kg 시절 값(17.6 kg 한 발 정하중 173 N) — 측정 크기 작음.
  4. 스텝 뒤 stance_shape·pose_symmetry 가 엇갈린 발 벌(추정 −1.7/s).
  5. 학습 종료 51 % 낙상(base_too_low), termination_penalty 실효 −0.2/회.
  6. `dst_last_end` 는 G8 이후 갱신 안 됨 → 외란 창은 `dist_window_mask`(_dist_end_time) 사용.
- 제안: G18a recovery_quality 접촉 조건 = 외란 창(외력 시작~종료+1.5 s) 안에서만 any_contact (외생 신호라 자기 개방 차단) /
  G18b F_imp_range (5,30)→(10,45) / G18c 외란 창 조건 착지 보상(shaping). 대조 = 같은 출발점 무변경 이어 학습 필수.


## 부록 3 — sim2real 관측·지연·센서 계약
(원 파일 `oss_3_sensor_contract.md`, 1002 통합)
- OSS 6종 모두 actor 에 접촉 관측 없음(humanoid-gym 은 critic 에만). 관절속도 잡음 0.1–1.5 rad/s(정규화 용도). IMU 별도 지연 DR 휴머노이드 OSS 미확인.
  Booster Gym: 접촉 compliance 0.5–1.5·restitution 0.1–0.9 DR, 물리 서브스텝 지연 0–9. rl_sar: sim·실물이 같은 관측 조립/주기 클래스.
- 실물 1001 측정: 틱 p50 20.0 ms·p99 20.4–21.3, 관절 양자 0.00038 rad(속도 0.019 rad/s), 정지 qvel=0 65–99 %, 자이로 정지 σ 0.006–0.008, 튐 |ωy| p99 2.7–3.0, 자이로 동결 최대 8틱.
- 결함(코드 확인): feet_contact_binary(delay 0) 가 _contact_reset_ids 를 안 비움 → 마지막 리셋 묶음 임계 U(2–20 N) 매 틱 재추출(평가에선 전 env). 2–20 N 구간(하중 상실·착지)만 영향.
- 누락: IMU 동결(표본 유지) 모델 없음(브리지는 0.2 s 까지 마지막 값 유지), 러너 관측 클리핑 없음(현재 무해), 자이로·중력 지연 독립 추출.
- 3 Hz 튐: 발목 PD 로 계산한 흔들림 고유주파수 ≈1.1 Hz(가정 I≈4.4) → 3 Hz 는 발 가장자리 구름·패드 탄성·이탈의 비선형 모드 지지(가설). MuJoCo sim2sim 보다 Isaac 내 지면 연성·반발 스윕 + --replay_bb 재생이 먼저.
- 제안: G18d 액션 지연 DR 0–3→1–3(실물 d≈1, 지연 0 은 실물에 없음; 짧은 지연 학습 = 떨림 가설) / G18i IMU 동결 DR(p≈0.5 %/틱, 1–10틱) / 지면 연성·반발 DR(재생 재현 시에만).


## 부록 4 — 지속 외력 내성
(원 파일 `oss_4_sustained_force.md`, 1002 통합)
- 외력 구현: mdp_hang.py:215-252(disturbance_reset), :183-191(base 로컬 프레임), :194-200(상승→유지→즉시 해제). 평가기 동일.
- G17o 분포: 램프 60 % 3–20 N·상승 1–4 s / 충격 40 % 5–30 N 0.2–0.4 s. **상승률 ≥20 N/s 표본 0 %** (평가 sf10=20 N/s, sf15=30 N/s 는 분포 밖).
  G17p: ≥20 N/s 11 %, (≥15 N & ≥30 N/s) 4.5 %.
- 크리틱에 외력 없음(env_cfg.py:124-133 base_lin_vel 만). FALCON·HIMLoco 는 크리틱 특권 관측에 외력.
- 정적 계산(가정 작용점 0.6 m): 발목 토크 한계 25 Nm 는 비병목(필요 11–15 Nm). 자세 고정 최대 지속 힘 앞 ≈30 N / 뒤 ≈13 N(CoM −20 mm).
  자세 보정 시 뒤 15 N 가능(CoM 1–3 cm 앞으로). → **15 N 은 스텝 없이 버틸 수 있는 범위(계산)**.
- 서명(G17o-10700): sf15 89 % 낙상·발 들기 0 %·양발 접촉 100 %·피치 +3.9→+7°·낙상 3.3–3.7 s; sb10 50 % 낙상·84 % 뒤 스텝.
- 앞 밀기 기전 가설: 명목 홀드는 발목 유격 뒤끝 평형(−3.5 Nm) → 앞 15 N(+8.3 Nm) 이 합 토크 부호 반전 → 쿨롱(3–6 Nm) 넘어 유격대 6–10° 횡단 → 발끝 넘어감. 대조 실험 필요.
- 제안: D0 진단(쿨롱 배율·유격 0·com_fix 스윕, sf15/sb10) / G18c 크리틱 특권 관측에 외력(액터 계약 불변) / G18h F_ramp (3,25)→(5,35) 또는 FALCON 생존 기반 힘 커리큘럼.

### D0 사전 기록 (1002 04:2x, 실행 전)
대상 G17o-10700 · eval seed 1 · 실물 계약(--enc_obs v2 --contact_real d1) · 조건 sf15·sb10 · 변형: 기본 / 유격 0(--ankle_play_deg 0) /
쿨롱 고정 1.5·4.0 Nm / com_fix −0.02·+0.02 m.
- H-play(앞 밀기 = 유격대 횡단): sf15 낙상이 유격 0 에서 기본 대비 ≥20 %p 낮고, 쿨롱 4.0 에서 1.5 보다 ≥15 %p 낮으면 지지.
- H-com(뒤 밀기 = 뒤꿈치 CoP 한계): sb10 낙상이 com −0.02 > 기본 > +0.02 로 단조이고 양 끝 차 ≥20 %p 이면 지지.

### D0 결과 (1002 08:43, G17o-10700 · seed 1 · 실물 계약 d1) — 낙상 % (사전검사 통과 n)
| 조건 | 기본 | 유격 0 | 쿨롱 1.5 | 쿨롱 4.0 | CoM −2 cm | CoM +2 cm |
|---|---|---|---|---|---|---|
| sf15 (앞 15 N 지속) | 89.3 (112) | 50.4 (115) | 94.6 (93) | 79.6 (103) | 66.3 (83) | 99.1 (108) |
| sb10 (뒤 10 N 지속) | 50.9 (112) | 52.2 (115) | 54.8 (93) | 18.4 (103) | 78.3 (83) | 31.5 (108) |
- **H-play 지지(경계)**: 유격 0 −38.9 %p(≥20 ✓), 쿨롱 4.0 vs 1.5 −15.0 %p(≥15, 경계 ✓).
- **H-com 지지**: sb10 이 CoM −2 > 기본 > +2 로 단조, 양 끝 차 46.8 %p(≥20 ✓).
- 주의: 플랜트 변형이 홀드 사전검사 통과 집합을 바꿔(n 83–115) 짝 비교가 아님. 발목 쿨롱·유격·CoM 실물값은 미측정(가정 범위).
- 해석(가설): 앞 밀기 실패 = 발목 유격대 횡단, 뒤 밀기 실패 = 뒤꿈치 쪽 CoP 여유(CoM 위치) + 마찰. 쿨롱 4.0 은 뒤 밀기 낙상을 51→18 % 로 줄여
  '정지 마찰이 뒤 밀기를 붙잡는' 기전도 시사. → G19 후보: 학습 DR 의 발목 유격·쿨롱·CoM 대역을 실측으로 좁히기(현장 측정 필요).


## 부록 5 — 학습 틀·적응 구조·평가 방법론
(원 파일 `oss_5_training_framework.md`, 1002 통합)
- "이어 학습이라 떨린다"는 미증명: G12 원 학습도 9500 이후 정숙도 악화(G12_VERDICT:9-11), 9500 은 저장본 중 최상 → 평균 회귀(승자의 저주) 가능.
  가장 유력(가설): 정숙 신호 부재(정착 보너스 죽음 0930·action_rate 는 안정 상태만) → 정숙도가 PPO 무관 방향이라 반복마다 떠돎. G16o seed 간 차가 세대 효과 급.
- 코드 확인 결함: 설치 rsl_rl 3.1.2 는 load 시 학습률 미복원(5.5.1 ppo.py:405 는 복원) → 이어 학습 직후 lr 1e-5→~1e-3 급등 ~100 반복(G12c 9501–9550 로그).
- std: G17o 0.586→0.41, G17r 0.26. action_rate 벌점은 샘플 액션에 걸려 탐색 잡음 자체를 벌함(std 0.59×0.25 ≈ 틱당 8.4° LPF 전).
- critic = actor 관측(잡음·센서 계약 포함) + base_lin_vel 뿐. 외력·지연·실제 접촉·DR 값 없음. humanoid-gym(외력·마찰·질량·접촉 mask)·HIMLoco(외력) 는 critic 특권.
- humanoid-gym PPO: entropy 0.001, lr 1e-5, epoch 2. 우리: entropy 0.008, lr 1e-3 adaptive(IsaacLab G1 보행 레시피).
- 제안: 0순위 G12 원 학습 저장본(8700–10698) vs G12c 분포 비교(평가만) / G18a critic 특권 입력(외력·지연·LPF·유격·실제 접촉력, critic 첫 층 0열 확장 이어 학습) /
  G18b LCP 0.002 / G18c 학습률 복원 수정(G12c 대조). 방법론: 학습 seed ≥2, 기준선 = 체크포인트 분포.
