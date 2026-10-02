# 참고한 오픈소스·논문 기록 (G17–G20, 2026-10-02)

목적: 어떤 외부 자료의 **어느 부분**을 보고, 우리 설계·판정에 **어떻게** 반영(또는 기각)했는지 남긴다.
조사는 1002 04시 분석 에이전트 5기(주제별)가 수행했다. 요약 원문: `oss_1_smoothness.md` · `oss_2_push_recovery.md` · `oss_3_sensor_contract.md` · `oss_4_sustained_force.md` · `oss_5_training_framework.md`.
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
