# OSS 조사 — 첫 스텝 탐색(step discovery) 병목

목적: G21a(강한 서기 정책)에서 이어 학습할 때 첫 스텝이 발견되지 않는 문제(앞 밀기 이륙 ≈0, 속도 명령 착지 0, STEP-B0 하중 보상 평탄)에
직접 쓸 수 있는 공개 코드·논문을 라운드별로 기록한다. 기존 참고(`reports/gen_g18/OSS_REFERENCES.md`: legged_gym·unitree_rl_gym·humanoid-gym·
isaac_berkeley_humanoid·FALCON·HIMLoco·HugWBC·HoST·booster_gym·rl_sar·LCP·rsl_rl·IsaacLab·van Marum 2024·Ferigo 2021)는 중복하지 않는다.
표기: **[코드 확인]** = 클론해 해당 줄을 직접 읽음, **[논문 확인]** = 원문 PDF 텍스트를 읽음, **[계산]** = 우리 설정으로 직접 계산, **미검증** = 요약·추정.
제약(동결): actor 관측 계약, PD 게인, 액션 스케일, 네트워크, PPO 하이퍼파라미터. 아래 제안은 이 제약 안에서 가능한 것(환경 리셋·보상·critic 쪽)을 우선한다.

## 라운드 1 (2026-10-04 03:31)

### 0. 우리 쪽 사실 확인 (조사 중 확인한 것, 결론에 영향)

- **앞으로 디딘 성공 스텝 시연은 시뮬에 0건** — `reports/gen_step/CORRECTION_1004_forward_steps.md`(앞 그룹 20건 전부 월드 −0.130 m, 반동 뒤 스텝).
  따라서 "966 뒤 + 11 앞" 시연 중 앞 정보는 없다. BC·DAPG·AMP 류는 **뒤 스텝만 가르친다**. 게다가 이 시연은 G21a/G22R2 자신의 롤아웃이므로
  같은 상태에서 BC 하면 정책은 이미 그 행동을 하고 있어 **새 정보가 거의 없다**(뒤 스텝은 이미 하고 있음). 시연의 가치는 "앞 밀기 상태에서의 행동"에 있는데 그 데이터가 없다.
- **탐색 잡음 크기 [계산]**: G21a 체크포인트(`logs/rsl_rl/biped12_w1/2026-10-03_21-20-36_w1b_cS0toS3_seed1_from_g21a11300/model_11300.pt`, CPU 로드)의
  행동 std = 0.31–0.59(액션 단위) × scale 0.25 = **0.078–0.15 rad**(관절당 독립 백색잡음). `rl_walking/actions.py:141-152` 의 슬루(±0.0698 rad/틱) → EMA(β 0.2–0.4)를
  numpy 로 통과시키면(정책 평균 일정 가정) **실제 관절 목표 흔들림 1.5–3.4°**, 그리고 **샘플 행동의 46–68 % 틱에서 슬루가 포화**된다.
  의미(가설): ① 스텝에 필요한 수십 °·여러 관절의 협응된 하중 이동은 관절별 독립 2–3° 잡음으로는 사실상 나오지 않는다.
  ② 슬루 포화가 잦아 PPO 가 log-prob 을 계산하는 "샘플 행동"과 실제 실행된 목표가 달라 신용 할당이 흐려진다. → 보상 형태만 바꾸는 시도(B0·B1)의 천장을 설명할 수 있는 후보. 대조 실험 전까지 가설.
- STEP-B0/B1 위상 `u = freq·t`(`rl_walking/mdp_hang.py:2272-2275`)와 actor 의 `gait_clock` 관측(`rl_walking/mdp_stage4.py:25-42`)은 같은 에피소드 시간 기반 → 보상이 actor 에게 비마르코프는 아님(확인). 단 모든 env 가 리셋 직후 φ=0(왼발 스윙 전환 창)에서 시작한다.

### 1. 출처별 분석

#### 1-1. LearningHumanoidWalking (Singh et al., JVRC-1/H1 스텝·보행) — 주기 접촉 보상의 현행 구현
- URL: https://github.com/rohanpsingh/LearningHumanoidWalking · commit **cd8c655 (2026-05-02)** [코드 확인]
- 핵심 줄:
  - `tasks/rewards.py:107-140` `calc_foot_frc_clock_reward`: 발별 힘을 **0.5 mg 로 정규화 → 2f−1 ∈ [−1,1]** 로 바꾼 뒤 `tan(π/4 · clock · f)`. clock 은 지지 +1, 스윙 −1.
    즉 **발별 선형(부호 있는) 항의 합** — 곱이 아니다. 대칭 서기(각 발 0.5 mg → f=+1)에서 예정 스윙발 점수 −1, 지지발 +1, 스윙발 하중을 줄이는 즉시 기울기가 생긴다.
  - `tasks/rewards.py:196-260` `create_phase_reward`: PCHIP 보간 위상함수, `strict_relaxer=0.1` 로 경계를 10 % 완화, 이중지지 구간 모드(grounded/aerial/zero).
  - `tasks/stepping_task.py:98-127` 보상 가중(힘 0.15·속도 0.15·자세 0.05·높이 0.05·**목표 발판 도달 0.45**·상체 0.05); `:105-108` STANDING 모드에서는 clock 을 상수(지지 +1)로 바꿈.
  - `tasks/stepping_task.py:280-287` 리셋 때 위상을 `{0, 주기/2}` 중 무작위.
- 메커니즘: 접촉 일정을 "힘 있으면 좋음/나쁨" 부호 함수로 주고 선형으로 보상. 처음부터(scratch) 학습. 발판 목표(0.45)가 가장 큰 항.
- **우리에게 적용**: STEP-B1(선형 비대칭)과 같은 계열이며, B0 의 곱형(ramp×ramp)이 대칭점에서 기울기 0 이라는 H-B1 가설을 **외부 구현이 지지**한다(근거 수준: 설계 사례, 인과 증명 아님).
  B1 이 실패할 경우의 차이점 두 가지만 단일 변수로 시험 가치가 있다: (a) 스윙 구간 **발 속도 항**(`calc_foot_vel_clock_reward`, 스윙 중 발 속도에 +) — 하중이 아니라 발 움직임에 직접 기울기;
  (b) 리셋 위상 무작위화(`{0, ½주기}`) — 리셋 직후 정착 과도와 첫 전환 창이 겹치는 문제 제거.
  위험: 이 저장소는 scratch 학습이라 "서기 국소해에서 이어 학습" 상황의 탐색 문제를 해결한다는 증거는 없다. 발판 목표 항은 우리 관측 계약에 발판 입력이 없어 그대로 못 쓴다.

#### 1-2. osudrl/apex (Siekmann·Xie, Cassie) — 주기 보상 원형 + 참조 잔차의 역사
- URL: https://github.com/osudrl/apex · commit **82c44af (2020-08-21)** [코드 확인]. 논문: Siekmann et al. ICRA 2021, arXiv 2011.01387.
- 핵심 줄:
  - `cassie/rewards/clock_rewards.py:6-117` `clock_reward`: 정규화 힘(최대 250 N 캡)·발 속도(2 m/s 캡)에 clock(±1)을 곱해 `tan(π/4·x)`; 가중 힘 0.2·발속도 0.2. 캡의 목적은 "위상 중 힘/속도를 최대화하는 꼼수 방지"(주석 :17).
  - `cassie/rewards/clock_rewards.py:119-` `early_clock_reward`: **학습 초기 전용**으로 clock 항 가중을 키우고 매끄러움/자세 항을 완화 — 주석 "prevent bobbing in place local minima".
  - `cassie/deprecated/cassie_env.py:88-90` `target = action + ref_pos` (참조 궤적 + 잔차 행동), `:137-142` 리셋 시 **참조 궤적의 무작위 위상 상태로 초기화**(RSI).
    이후 env 들은 `no_delta=True`(`cassie_footdist_env.py:19`)로 잔차를 버리고 clock 보상만 사용.
- 메커니즘: (i) 초기에는 접촉 일정 항을 지배적으로, 정숙 항을 약하게 하는 2단 보상; (ii) 역사적으로 "참조 잔차 + RSI" → "참조 없는 clock 보상"으로 이동.
- **우리에게 적용**: `early_clock_reward` 는 우리 G15 교훈(정숙 보상이 회복 스텝을 억제)과 같은 방향. 최소 실험: B1 학습의 **처음 N 반복 동안 스텝 모드 env 에서만 action_rate·정숙·자세 항 가중을 0.3배**(단일 변수: "정숙 항 완화 여부"), 이후 원복.
  기대: 첫 비대칭/이륙 시도가 벌점에 즉시 상쇄되는 것을 줄임. 위험: 25 Hz 떨림(G17) 재발 — 완화 기간을 짧게, 떨림 지표 동시 감시. 잔차 구조(`action + ref`)는 Xie 가 결국 폐기했다는 점이 "α→0 소거 시 붕괴" 위험의 간접 증거(**미검증**: 폐기 이유를 문서로 확인하지 못함).

#### 1-3. MimicKit (Peng, DeepMimic/AMP/ASE/ADD 통합, Isaac Lab 엔진 지원)
- URL: https://github.com/xbpeng/MimicKit · commit **2ed1e6c (2026-06-23)** [코드 확인]. DeepMimic 원본: https://github.com/xbpeng/DeepMimic · 1f915c5 (2025-11-27).
- 핵심 줄:
  - `mimickit/envs/deepmimic_env.py:175-197` `_reset_ref_motion`, `:277-285` `_sample_motion_times`: `rand_reset=True` 면 **동작 클립의 무작위 시각 상태로 리셋(RSI)**, False 면 시작 프레임.
  - `:724-784` `compute_done`: 조기 종료 = 허용 접촉체(발) 외 지면 접촉 **또는 몸체 위치 오차 > `pose_termination_dist`**(G1 설정 1.0 m), 첫 틱 제외.
  - `:787-845` `compute_reward`: exp(−k·err) 가중합. G1 설정 `data/envs/deepmimic_g1_env.yaml`: pose 0.5·vel 0.1·root_pose 0.15·root_vel 0.1·key_pos 0.15(발목·머리·손목), scale pose 0.25·key_pos 10.
  - Isaac Lab 엔진: `mimickit/engines/isaac_lab_engine.py` (README: IsaacLab 2ed331a 에서 시험).
- 메커니즘: RSI + 조기 종료 + 추적 보상. DeepMimic 논문 ablation 에서 RSI·ET 없이 어려운 동작 학습 실패(수치는 이번 라운드에서 원문 재확인 안 함 → **미검증**).
- **우리에게 적용**: 우리 계획(STEP-I: 위상 정규화 기준 + 감쇠, `rl_walking/mdp_hang.py:2325-2400`)은 보상만 있고 **RSI 가 없다**. DeepMimic 계열에서 RSI 는 추적 보상과 짝이다 —
  보상만으로는 "스윙 중간 상태"를 한 번도 방문하지 못하면 그 구간의 가치·행동을 배울 수 없다. 최소 실험은 §2 순위 1 참조(보상 동결, 리셋 분포만 변경).
  위험: 우리 시연이 전부 뒤 스텝 → RSI 상태도 뒤 스텝뿐이면 앞 밀기 문제에 직접 기여 못 함. 앞 상태는 합성해야 한다(§2-1).

#### 1-4. BeyondMimic / whole_body_tracking (HybridRobotics, **Isaac Lab + rsl_rl**) — 실패 구간 적응 샘플링 RSI
- URL: https://github.com/HybridRobotics/whole_body_tracking · commit **cd65172 (2025-10-05)** [코드 확인]
- 핵심 줄 (`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`):
  - `:207-241` `_adaptive_sampling`: 동작을 시간 bin 으로 나누고, **종료(실패)가 난 bin 의 횟수**를 EMA(`adaptive_alpha=0.001`, `:296-299`)로 누적,
    확률 = 실패횟수 + `uniform_ratio 0.1`/bin 수, 커널(λ=0.8) 평활 후 다항 샘플링. 샘플링 엔트로피·top1 bin 을 metrics 로 로그.
  - `:243-280` `_resample_command`: 선택된 시각의 기준 상태를 `write_joint_state_to_sim`/`write_root_state_to_sim` 으로 **직접 써서 RSI**, 자세·속도·관절(±0.52 rad) 잡음 추가.
  - `tasks/tracking/mdp/terminations.py:18-58` 기준 대비 앵커 z·자세·발 위치 오차 종료; `tracking_env_cfg.py:202-240` 보상(가중 0.5–1.0, std 0.3–3.14), action_rate −0.1.
- 메커니즘: "어디서 실패하는가"에 비례해 시작 상태를 더 많이 뽑는 자동 커리큘럼(hard-negative mining 의 시간축 버전, PHC 와 같은 계열).
- **우리에게 적용**: Isaac Lab ManagerBased 환경에서 **리셋 이벤트로 상태 은행(state bank)에서 샘플 → sim 에 직접 쓰기**를 하는 그대로의 참고 코드.
  우리 상태 은행의 bin = "스텝 위상(이중지지 하중이동 / 이륙 직후 / 공중 중간 / 착지 직전)". 실패율 비례 샘플링을 넣으면 정책이 못하는 위상에 표본이 몰린다.
  위험: 상태 은행에 들어간 상태가 물리적으로 실현 불가능하면(관입·속도 불일치) 첫 틱 폭발 — BeyondMimic 처럼 첫 틱 종료 제외(MimicKit `:780-782`)와 관입 검사 필요.

#### 1-5. ProtoMotions (NVlabs) — RSI 와 "시작 자세" 혼합 비율
- URL: https://github.com/NVlabs/ProtoMotions · commit **7a8417a (2026-09-22)** [코드 확인]
- 핵심 줄: `protomotions/envs/motion_manager/config.py:20-26` `init_start_prob=0.2` — "무작위 시각 대신 초기 자세로 시작할 확률. **AMP 에서 국소해 방지에 도움**";
  `protomotions/envs/motion_manager/motion_manager.py:90-92`; G1 배포 설정 `data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py:209` 0.2(학습), `:499` 1.0(평가).
- **우리에게 적용**: 상태 은행 RSI 를 넣을 때 **표준 서기 리셋을 일정 비율(예 50–80 %) 유지**하는 혼합 설계의 근거. 서기 성능(G21a 의 강점) 망각 방지용 대조 축으로 "RSI 비율"을 단일 변수로 둔다.

#### 1-6. AMP_for_hardware (Escontrela et al., legged_gym + rsl_rl AMP) — RSI 비율 설정의 함정
- URL: https://github.com/escontra/AMP_for_hardware · commit **bfb0dbd (2022-04-27)** [코드 확인]
- 핵심 줄: `legged_gym/envs/base/legged_robot.py:205-211` `reference_state_initialization` 이면 **매 리셋 100 % 참조 프레임에서 시작**;
  `legged_gym/envs/a1/a1_amp_config.py:44-45` `reference_state_initialization_prob = 0.85` 는 **정의만 있고 코드 어디서도 사용되지 않음**(grep 확인).
  `rsl_rl/rsl_rl/algorithms/amp_discriminator.py:55-71` 판별기 보상 `coef·clamp(1−¼(d−1)², 0)` 과 과제 보상의 선형 보간(`task_reward_lerp 0.3`, 설정 `:161-164`); AMP 관측 `legged_robot.py:285-292`(관절·발 위치·기저 속도·높이).
- **우리에게 적용**: AMP 는 "스텝 스타일"을 판별기로 주는 대안이지만 (i) 앞 스텝 시연이 없고 (ii) rsl_rl 3.1.2 에 AMP 가 없어 러너 수정 필요 → **이번 병목에는 비우선**.
  교훈: 혼합 확률 같은 설정은 **실제로 소비되는지 코드로 확인**(CLAUDE.md 4번 — 하네스가 무엇을 하는지 확인)과 같은 함정 사례.

#### 1-7. rsl_rl 내장 RND (Random Network Distillation, Schwarke et al. CoRL 2023) — 설치본 3.1.2 에 이미 있음
- 설치본: `/home/ryu/isaacsim/isaac-sim-5.1.0/kit/python/lib/python3.11/site-packages/rsl_rl` [코드 확인]
  - `modules/rnd.py:24-112` 가중 스케줄(constant/step/linear), 상태·보상 정규화; `:117-138` 내재 보상 = ‖target−predictor‖ × weight; `:155-157` **`rnd_state` 관측 그룹을 별도로 지정**.
  - `algorithms/ppo.py:63-73,161-174` 외재 보상에 내재 보상을 더함; `runners/on_policy_runner.py:43-44` rnd_cfg 가 있으면 `rnd_state` 그룹 요구.
  - Isaac Lab 쪽 `isaaclab_rl/rsl_rl/rnd_cfg.py`, `rl_cfg.py:125 rnd_cfg`.
  - **함정**: `on_policy_runner.py:312-322` 로드 시 `loaded_dict["rnd_state_dict"]` 를 무조건 읽음 → **RND 없이 저장된 G21a 체크포인트로 이어 학습하면 KeyError**. 로드 래퍼 필요.
- 메커니즘: 처음 보는 상태에 보너스. `rnd_state` 를 actor 관측과 분리할 수 있어 **actor 계약은 그대로**.
- **우리에게 적용**: `rnd_state` = critic 쪽 특권 신호 중 "스텝 관련" 저차원만(양발 높이·발별 Fz/mg·발 전후 위치). 서기 국소해에서는 이 상태가 거의 변하지 않으므로, 한 발을 들거나 하중을 옮기는 순간 큰 보너스.
  최소 실험: 스텝 모드 이어 학습에 RND 하나만 추가(weight 선형 감쇠), 동일 seed 무RND 대조. 기대: 하중 비대칭·이륙 시도 빈도 ↑. 위험: 내재 보상이 넘어짐·발 끌기 같은 "새 상태" 전반을 보상 →
  rnd_state 를 발 관련으로 좁히고 넘어짐 종료 유지. PPO 하이퍼 동결 원칙과 충돌 여부는 사용자 판단 필요(손실·옵티마이저는 별도라 PPO 자체 파라미터는 불변).

#### 1-8. Atalante 반응 스텝 (Duburcq et al., IROS 2022, arXiv 2203.01148) — 서기 정책에 **보행 상태로 초기화**, 밀기 커리큘럼 역효과
- 논문만(코드 비공개) [논문 확인: Sec. IV-A4, IV-B, IV-C, IV-D]
- 내용:
  - IV-B Initialization: 서기 밀기 회복 정책인데 초기 상태를 **서기·전후 보행·회전 참조 궤적에서 균등 샘플 + 가우시안 잡음**으로 뽑음("naive random sampling 은 회복 불가 상태를 많이 만든다").
  - IV-A4 Disturbances: 골반에 종 모양 힘 **최대 800 N·400 ms, 3 s ± 2 s 마다, 구면 방향**. "**힘을 0→800 N 로 점진 증가시키면 조기 수렴(준최적)**이 관찰됨".
  - IV-C5 Transient dynamics 종료: 4 s 창에서 기준과의 최소 오차 < 0.3 이 아니면 종료(크게 벗어나는 건 허용, 오래 머무는 건 금지).
  - IV-D 보상: 캡처 포인트가 지지다각형 밖으로 나가면 골반을 CP 쪽으로(‖p_cp − p̂_cp‖), ZMP 를 지지 중심으로; 스텝 자체엔 보상 없음.
- **우리에게 적용**: (i) "서기 정책도 보행/스텝 상태에서 시작시킨다"는 직접 선례 → §2-1 상태 은행 RSI. (ii) 우리 학습 충격은 작고(OSS_REFERENCES 부록2: 최대 0.68 N·s/kg) 점진 커리큘럼 관행이 있음 —
  Atalante 의 "점진 증가가 준최적 수렴" 관찰은 **앞 밀기만 큰 고정 세기 표본을 섞는** 단일 변수 실험의 근거(800 N×0.4 s / 약 150 kg ≈ 2 N·s/kg 수준, 질량은 **미검증 추정**).
  위험: 큰 충격만으로는 G21a 가 "넘어지기"를 학습할 수도 있음(앞 실패 = 시도 없음이므로) — RSI 와 같이 써야 의미.

#### 1-9. Jump-Start RL (Uchendu et al., ICML 2023, arXiv 2204.02372) — 안내 정책 roll-in
- 프로젝트: https://jumpstart-rl.github.io/ [논문 확인: Algorithm 1, Sec. 5.4]
- 내용: 안내 정책 π_g 가 h 틱 동안 행동하고 이후 탐색 정책 π_e 가 이어받음. 커리큘럼(h 를 성능 임계 β 넘을 때 감소) vs **JSRL-Random(h 균등 무작위)**.
  Sec 5.4: "Random 도 최종 성능은 비슷 — **이점의 대부분은 안내 정책이 만든 좋은 방문 상태에서 온다**, 순서는 부차적". 기반 알고리즘은 IQL/QT-Opt(오프폴리시).
- **우리에게 적용**: 우리 계획의 "policy_action + α·참조 잔차, α 0.5→0"(assisted exploration)는 JSRL 의 연속 버전인데, PPO(온폴리시)에서는 α>0 구간이 **다른 동역학에서의 학습**이 되어 α→0 때 분포 이동이 생긴다.
  JSRL 의 결론(방문 상태가 핵심)을 따르면 더 깔끔한 구현은: **보조(스크립트 잔차)로 롤아웃한 상태를 오프라인으로 상태 은행에 저장 → PPO 학습은 그 상태에서 정책 단독으로 시작(RSI)**. PPO 손실·동역학 불변.
  위험: 저장 상태에는 actor 관측 이력(5틱 history·LPF 상태·지연 버퍼)도 있어야 함 — 리셋 시 이력 버퍼를 일관되게 채우지 않으면 첫 몇 틱 관측이 계약과 다름(§2-1 위험 항).

#### 1-10. DAPG (Rajeswaran et al., RSS 2018) — 시연 BC 항 감쇠의 정량 레시피
- URL: https://github.com/aravindr93/mjrl · commit **3871d93 (2021-06-18)** [코드 확인]
- 핵심 줄: `mjrl/algos/dapg.py:34-35` `lam_0=1.0, lam_1=0.95`; `:62-70` 시연 전이에 **상수 어드밴티지 λ0·λ1^k** 를 주어 PG 배치에 합침(정책 어드밴티지는 표준화 후 1e-2 배).
  0.95^k → 100 반복 후 0.6 %, 사실상 100 반복 내 소거.
- **우리에게 적용**: "BC warm-start ablation"을 rsl_rl 에 넣는다면 PPO 손실에 `−w_k·mean log π(a_demo|o_demo)` 를 더하고 w_k = w0·0.95^k 로 감쇠 — 가장 단순한 감쇠 규칙.
  **단 §0 사실 때문에 효과 기대가 낮다**: 시연이 G21a 계열 자신의 뒤 스텝이면 BC 손실은 이미 거의 0. 앞 상태 시연(합성)이 생긴 뒤에만 의미. 위험: 관측 이력·잡음 계약 불일치 시연은 오히려 해로움.

#### 1-11. 검토했으나 이번 병목에 비우선
- **Reverse Curriculum Generation** (Florensa et al., CoRL 2017, arXiv 1707.05300): 목표 근처 시작 상태에서 점점 멀리. 논문만 참고(코드 rllab 기반, 미클론, **미검증**). §2-1 의 "착지 → 공중 → 이륙 → 이중지지" 역순 RSI 근거.
- **ASE/CALM/PHC/LocoMuJoCo**: 대규모 모션 데이터·잠재 기술 전제. 앞 스텝 데이터 0건인 우리 상황에서 직접 기여 없음.
- **Enhancing Model-Based Step Adaptation for Push Recovery via RL** (arXiv 2411.01000): 모델 기반 스텝 계획 + RL 로 타이밍·영역 조정 — 구조 변경이 커서 보류(**미검증**, 초록만).

### 2. 우리 병목에 대한 실행 순위 (상위 3)

**1위 — 앞 스텝 상태 은행 + 역순 RSI (리셋 분포만 변경; 보상·관측·PPO 불변)**
- 근거: Atalante IV-B(서기 정책을 보행 상태에서 시작), JSRL Sec 5.4(이점은 방문 상태), DeepMimic/MimicKit RSI(`deepmimic_env.py:175-197,277-285`), BeyondMimic 적응 샘플링(`commands.py:207-280`), ProtoMotions 혼합 비율(`config.py:20-26`).
- 구체안: 앞 시연이 0건이므로 상태를 **합성**한다 — (a) 앞 밀기 직후 상태(기저 앞 속도 0.2–0.5 m/s, 기울기 3–8°)에 한 발을 스윙 자세(이륙 직후·공중 중간·착지 직전 3구간)로 둔 기구학 상태, 또는
  (b) 앞 밀기 중 G21a 에 스크립트 잔차(스윙 다리 힙 굴곡 + 지지 쪽 하중 이동)를 더해 롤아웃한 상태(JSRL 식 보조 roll-in, 오프라인 저장). 학습은 bin(착지 직전 → 공중 → 이륙 → 이중지지 하중이동) 순서 또는 실패율 비례 샘플링, 표준 서기 리셋 50–80 % 유지.
- 최소 실험: 같은 G21a 출발·seed 에서 [대조 = 현재 B1 설정] vs [RSI 은행 20–50 % 추가] 단 하나만 변경. 판정은 **은행 시작 env 가 아니라 표준 서기 시작 + 앞 밀기 평가**의 이륙률/회복률(은행 시작 성적은 진단용).
- 기대: 착지·하중 수용 가치가 먼저 학습되고, 이어서 더 이른 위상에서 이륙 행동이 생김. 위험: 합성 상태의 물리 비현실성(관입·첫 틱 폭발 — 첫 틱 종료 제외·관입 검사), **관측 이력/LPF/지연 버퍼 초기화 불일치**, 은행 위상에서만 잘하고 표준 시작에서 전이 안 됨(그래서 판정을 표준 시작으로).

**2위 — 탐색량 진단 + 대조: "잡음이 스텝을 만들 수 있는가"**
- 근거: §0 [계산] — 실효 관절 잡음 1.5–3.4°, 슬루 포화 46–68 %. 보상 형태(B0→B1)를 바꿔도 첫 비대칭이 안 나오는 현상과 정합. LHW/apex 는 scratch 학습(초기 std 큼)이라 이 문제를 겪지 않았을 가능성(**미검증**).
- 최소 실험(학습 없이 가능, GPU 평가 1회): G21a 를 **결정론 vs 확률 정책(학습 때와 같은 샘플링)**으로 앞 밀기·스텝 모드에서 롤아웃해 하중 비대칭 분포·이륙률·슬루 포화율을 측정. 확률 롤아웃에서도 비대칭 꼬리가 B1 성공 기준(+0.05 mg)에 못 미치면 "보상 설계로는 탐색 불가" 가 측정으로 확정.
  이후 단일 변수 후보: RND(1-7, actor 계약 불변, 로드 래퍼 필요) 또는 1위의 RSI.
- 위험: 진단 자체는 무해. 확률 롤아웃의 슬루 포화는 실제 학습 표본의 성질이므로 그대로 둔다(평가 하네스가 학습과 같은 액션 경로를 쓰는지 먼저 확인).

**3위 — 초기 정숙 항 완화(early clock reward) 단일 변수**
- 근거: apex `clock_rewards.py:119-` "bobbing in place 국소해 방지" 초기 전용 보상, 우리 G15 판정(정숙 보상 3종이 회복 스텝 보호를 깸).
- 최소 실험: B1 설정에서 스텝 모드 env 에 한해 처음 300 반복 action_rate·각속도·자세 대칭 항 0.3배, 이후 원복. 대조 = B1 그대로. 판정 = B1 사전 기준(비대칭 +0.05 mg / load_ready ≥ 5 % / 이륙 +5 %p) + 25 Hz 떨림 지표.
- 기대: 하중 이동 시도의 즉시 벌점을 줄여 B1 기울기가 실제 행동으로 이어짐. 위험: 떨림 재발·배포 계약 무관하지만 원복 뒤 망각. 2위 진단에서 "탐색량 자체가 부족"으로 나오면 이 실험의 기대는 낮아진다(그 경우 1위 우선).

출처: [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking) · [apex](https://github.com/osudrl/apex) · [MimicKit](https://github.com/xbpeng/MimicKit) · [DeepMimic](https://github.com/xbpeng/DeepMimic) ·
[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) · [ProtoMotions](https://github.com/NVlabs/ProtoMotions) · [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) · [mjrl/DAPG](https://github.com/aravindr93/mjrl) ·
[Atalante arXiv 2203.01148](https://arxiv.org/abs/2203.01148) · [JSRL arXiv 2204.02372](https://arxiv.org/abs/2204.02372) · [Siekmann arXiv 2011.01387](https://arxiv.org/abs/2011.01387) · [Reverse Curriculum arXiv 1707.05300](https://arxiv.org/abs/1707.05300) · [arXiv 2411.01000](https://arxiv.org/abs/2411.01000)

## 라운드 2 (2026-10-04 03:55)

전제(코디네이터 전달, 측정 사실): 개루프 진단에서 G21a + α·기준 잔차가 α=1 → 비대칭 +0.18 mg·이륙 0.8 %, α=2 → 비대칭 +0.39 mg·load_ready 74 %·이륙 31 %·착지 14.5 %·하중 수용 12.8 %·낙상 3.9 %.
STEP-B1 은 200 반복에서도 비대칭 0. 준비 중인 STEP-A0 = 정책 + α·잔차, α 2.0→1.5→1.0→0.5→0.25→0, 블록 하중 수용 ≥ 50 % 로 감쇠 승격.

### 0. STEP-A0 구현에서 확인한 사실 (읽기만, 수정 없음)

- 보조는 `rl_walking/mdp_hang.py:2402-2423` `install_step_assist` 가 액션 항의 `pre_delay_hook` 에 잔차를 더한다. 훅 위치는 `rl_walking/actions.py:133-134` —
  **지연 버퍼 → 슬루(±4°/틱) → LPF 앞**. 따라서 (i) 보조는 env 쪽 동역학 변경이고 PPO 저장 행동은 정책 샘플 그대로(아래 2-1, on-policy 유지),
  (ii) 잔차가 **정책 잡음과 같은 슬루 예산을 나눠 쓰고 LPF 로 감쇠**된다. α=1 → 이륙 0.8 %, α=2 → 31 % 의 강한 비선형은 이 포화·감쇠와 정합(가설: 인과 미검증).
  → α 를 2.0→1.5 로 한 단계 내리는 것만으로도 이륙이 급락할 수 있다(계단 폭이 비선형 구간에 비해 큼).
- 커리큘럼 `mdp_hang.py:2426-2446` `step_assist_curriculum`: **전 env 공통 α 하나**, 승격 조건 = 블록 하중 수용 ≥ 0.5 & 최소 50 반복, **되돌림(α 증가) 없음**. 개루프 α=2 의 하중 수용은 12.8 % 이므로 첫 감쇠 전에 정책이 약 4배 개선돼야 한다.
- α 는 actor 관측에 없음(`mdp_hang.py:2404` 주석). 잔차는 **에피소드 시간 기반 위상**(`2416-2420`)의 함수이고 actor 는 같은 위상의 `gait_clock`(`mdp_stage4.py:25-42`)을 관측한다 → 원리적으로 actor 가 잔차를 스스로 표현(내재화)할 수 있는 정보는 있다.

### 1. 출처별 분석

#### 1-1. HoST (Huang et al., RSS 2025) — **성공 조건부 보조력 감쇠의 실제 코드**
- URL: https://github.com/InternRobotics/HoST · commit **70bb580 (2025-06-17)** [코드 확인] (라운드 0 OSS_REFERENCES 는 지연 DR 만 참고 — 보조 커리큘럼은 이번이 처음)
- 핵심 줄 (`legged_gym/legged_gym/envs/base/host_ground.py`):
  - `:107-113` 기저에 **위쪽 당김 힘**(env 별 `self.force`) 인가, 단 **몸통이 거의 수직일 때만**(`projected_gravity_z < −0.8`, `:110-112` 의 `host_ground_prone.py` 동형) 그리고 비구동 초기 시간 이후.
  - `:505-508` `update_force_curriculum(env_ids)`: **리셋되는 env 묶음의 평균 머리 높이 > 문턱**이면 그 env 들의 힘 −20 N(0 하한), 행동 배율 −0.02(0.25 하한). 증가 경로 없음. `:201` reset 시 호출.
  - `:589-590` 초기값(G1: `g1/g1_config_ground.py:293-299` force 100 → 실제 200 N 주석, threshold_height 0.9 m), `:290` **행동 배율을 actor 관측에 넣음**(잡음 ±0.025), `:227-228` 힘·배율 로그.
- 메커니즘: 보조 = **env 쪽 외력**(정책 행동과 분리), **env 별 진폭 감쇠**, 성공(목표 높이 도달) 1회당 한 계단(200 N → 10회 성공이면 0), 보조는 **상태 조건부**(이미 섰을 때만).
- **우리에게 적용**: (a) STEP-A0 의 전역 α 대신 **env 별 α** 로 바꾸고 "그 env 의 스텝 사이클이 하중 수용까지 성공하면 그 env 의 α −δ" 규칙(HoST 식)을 쓰면 잘하는 env 부터 보조가 빠지고 못하는 env 는 보조 유지 — 전역 α 의 '평균에 끌려 전원 동시 급락' 위험이 없다.
  (b) HoST 는 보조량(행동 배율)을 actor 관측에 넣었지만 우리는 actor 계약 동결 → **α 를 critic 관측에만** 넣는 것이 대응책(비대칭 actor-critic; 가치 추정의 α 모호성 제거). 위험: Baisero & Amato 2022 의 특권 critic 편향(OSS_REFERENCES §2).
  (c) HoST 의 '수직일 때만' 같은 상태 조건: 우리 잔차는 시간만 보고 상태를 안 봄 → 이미 넘어지는 중에도 잔차가 들어가 낙상 3.9 % 기여 가능(가설).

#### 1-2. Yu, Turk & Liu 2018 (SIGGRAPH, "Learning Symmetric and Low-energy Locomotion", arXiv 1801.08093) — **에피소드 안에서 보조를 줄이는 env-centered 커리큘럼**
- 논문 [논문 확인: Sec. 4.1, 4.2.1, 4.2.2, Algorithm 3]. 공식 코드 저장소는 이번 라운드에서 찾지 못함(**미검증**).
- 내용:
  - 4.1 보조 = 골반에 붙은 **가상 PD 스프링 힘**(좌우 균형 + 전진 추진), 커리큘럼 변수 x = (kp_횡, kp_전진). env 쪽 힘이므로 PPO 는 그대로.
  - 4.2.1 learner-centered: 현재 정책이 원 수익의 h=80 % 에 도달하면 5방향 line-search 로 "정책이 수익 60 % 를 유지하는 가장 큰 보조 감소"를 찾아 다음 레슨. ‖x‖<5 이면 무보조로 최종 학습("매우 빨리 배운다").
  - 4.2.2 environment-centered(논문이 데이터 효율로 선호, Sec 6): **한 롤아웃 안에서 보조를 3 s 마다 25 % 씩 계단 감소**(x_begin → x_end), 진행 조건 = **두 번째 계단(2p s)까지 균형 유지 AND 평균 수익 > 완전보조 정책 수익의 g=70 %**.
    "한 롤아웃 안에서 여러 보조 수준을 겪게 해 **특정 보조에 과적합하는 것을 막는다**". 수익 조건을 둔 이유: "균형 조건만 쓰면 가만히 서 있는 정책이 될 수 있다".
- **우리에게 적용**: STEP-A0 의 가장 큰 위험(α→0 에서 행동 소멸)에 대한 원조 처방. 최소 변경: 스텝 모드 에피소드 안에서 α 를 **반주기마다 계단 감소**(예 2.0 → 1.5 → 1.0 → 0.5, 반주기 23틱 기준) 하고, 전역 감쇠는 이 창 [α_begin, α_end] 를 통째로 내리는 방식.
  같은 에피소드에서 "보조가 빠지는 순간에도 이어서 딛기"가 직접 학습 신호가 된다. 진행 조건은 하중 수용률만이 아니라 **α 가 가장 낮은 반주기의 하중 수용률**(=보조 없이 얼마나 하나)로.
  위험: 반주기마다 동역학이 바뀌어 critic 이 α 를 모르면 가치 잡음 증가(→ critic 에 α). 이 논문은 scratch 학습·토크 제어라 이어 학습에서도 같은 효과가 날지는 미검증.

#### 1-3. Residual Policy Learning (Silver et al. 2018, arXiv 1812.06298) — 잔차 학습의 기본 규칙 2개
- URL: https://github.com/k-r-allen/residual-policy-learning · commit **ab330ee (2019-04-08)** [코드 확인] + 논문 [논문 확인: ar5iv]
- 핵심 줄: `rpl_environments/rpl_environments/envs/residual_fetch_push_env.py:26-34` — **기본 제어기를 env.step 안에서 더함**(`action = residual + get_push_control(obs)`, 이후 clip); `tensorflow/experiment/models/utils.py:27-42` `nn_last_zero` **마지막 층 0 초기화**.
  논문: 기본 제어기는 **끝까지 제거하지 않음**. 강한 초기 정책 + 미학습 critic 이면 초기 actor 갱신이 성능을 망치므로 **critic 만 먼저 학습하는 burn-in**(critic 손실 < β=1.0 이 될 때까지 정책 고정).
- **우리에게 적용**: (a) RPL 은 보조를 '영구 부품'으로 둔다 — 실물 배포 계약상 우리는 α=0 이 필수라 RPL 은 "제거 문제"를 풀어주지 않는다(대신 env 쪽에 더하는 구조가 on-policy 를 지킨다는 점은 STEP-A0 와 동일, 확인).
  (b) **critic burn-in**: α=2 를 켠 순간 G21a 의 critic 은 이 동역학(스텝·착지)을 본 적이 없다. 최소 실험: STEP-A0 시작 후 처음 N 반복(예 50) **actor 학습률 0(또는 actor 고정)·critic 만 갱신**, 대조 = 즉시 공동 갱신. 기대: 초기 actor 붕괴·서기 망각 감소.
  위험: PPO 하이퍼 동결 원칙과의 관계(학습률 스케줄 변경) — 사용자 판단 필요. rsl_rl 3.1.2 에는 actor 고정 옵션이 없어 러너 래퍼 필요.

#### 1-4. Kickstarting (Schmitt et al. 2018, arXiv 1803.03835) — 보조를 **행동이 아니라 손실로** 넣는 대안
- 논문 [논문 확인: ar5iv Sec 2.1, 2.3, 5.1.1]. 공식 코드 없음.
- 내용: 손실 = ℓ_RL + λ_k·H(π_teacher ‖ π_student). λ_k 는 PBT 로 자동 조정하거나 **선형 1→0 감쇠**(Sec 5.1.1: "올바르게 고르면 선형 감쇠도 잘 된다", 상수 λ 는 교사 수준에서 정체). λ→0 이후에도 성능 하락 없이 교사를 넘어섬(보고 수치 +43.4 %, DMLab-30).
- **우리에게 적용**: STEP-A0 의 "보조가 빠지면 행동이 사라진다" 위험은 보조가 **정책 밖**(env)에 있기 때문. Kickstarting 형으로 바꾸면 보조가 **정책 안으로** 들어간다:
  교사 = 고정 G21a 평균 + α·r(φ)/0.25(행동 단위), 학생 손실에 λ_k·‖μ_θ(o) − 교사(o)‖² (스텝 모드 env 만), λ_k 선형 감쇠. actor 가 `gait_clock`·명령을 보므로 r(φ) 를 표현할 정보가 있다(§0).
  최소 실험: STEP-A0(env 보조) vs STEP-K0(손실 보조, 실행 행동은 정책 그대로) — 같은 α 일정·seed. 판정은 둘 다 α/λ=0 에서. 기대: K0 는 보조 제거 절벽이 없음. 위험: 실행 행동에 보조가 없으므로 **초기에는 실제 스텝 상태를 방문 못 함**(탐색 이점 상실) →
  A0 와 결합(env 보조로 방문 + 손실로 내재화)이 현실적. PPO 손실 변경이므로 SmoothPPO(G18j) 와 같은 서브클래스 방식 필요.

#### 1-5. Jump-Start RL (라운드 1 §1-9 보강) — 롤인 길이 감쇠 규칙
- [논문 확인: Algorithm 1] h(안내 정책 단계 수)를 H₁ > H₂ > … 로 줄이되 **조합 정책 평가 ≥ β 일 때만** 다음 단계. JSRL-Random(h 균등 무작위)도 최종 성능은 비슷(Sec 5.4). 공식 코드는 찾지 못함(**미검증**).
- **우리에게 적용**: α(진폭) 대신 **보조 지속 반주기 수 h** 를 줄이는 변형 — 에피소드 처음 h 반주기만 α=2(개루프에서 실제로 스텝이 나는 진폭), 이후 α=0. h 를 env 별 무작위(JSRL-Random)로.
  장점: 진폭의 비선형 절벽(α 1 vs 2)을 피한다 — 보조가 있는 구간은 항상 '효과 있는' 진폭. 위험: 에피소드 초반에만 스텝이 몰려 이후 서기 국소해로 돌아갈 수 있음 → 판정은 보조 종료 후 반주기의 하중 수용.

#### 1-6. DAgger β-혼합 (Ross et al. 2011, AISTATS) — '확률적으로 보조 켜기'의 고전적 일정
- 논문만(코드 없음, 원문 재확인 안 함 → 수치 **미검증**): i 번째 반복에서 전문가 행동을 확률 β_i 로 실행, β₁=1, β_i = p^{i−1} 지수 감쇠.
- **우리에게 적용**: ④ 의 "env 비율 감쇠" 변형 근거. 단 DAgger 는 모방 손실로 학습하므로, PPO 에서 보조를 확률적으로 켜면 보조 없는 env 는 여전히 국소해에 있다(아래 §2-4 논의).

### 2. 질문별 정리

#### 2-1. PPO 의 off-policy 성 — 보조를 동역학으로 둘 때 vs 행동으로 넣을 때
- **확인된 사실 [코드 확인]**: rsl_rl 3.1.2 `algorithms/ppo.py:147-149` 는 `policy.act(obs)` 샘플을 **env.step 전에** 저장하고 그 log-prob 을 기록, `runners/on_policy_runner.py:104-110` 에서 같은 텐서를 env 에 넘긴다. 갱신 시 `ppo.py:251,295` ratio = exp(logπ_new(a) − logπ_old(a)).
  STEP-A0 는 잔차를 env 안(액션 항 훅)에서 더하므로 저장된 a 는 정책 샘플 그대로 → **PPO 는 '보조가 포함된 동역학'에 대해 on-policy**. likelihood ratio 보정 불필요.
- 만약 실행된 행동(정책+보조)을 a 로 저장하거나, 보조를 rsl_rl 쪽에서 행동에 더하면 old log-prob 이 실행 행동과 불일치 → ratio 편향(보조 크기 2·r 은 std 0.08–0.15 rad 보다 큼 → 로그확률이 극단값, 클립이 대부분 표본을 잘라 학습 정지 가능, 계산적 추론·미검증). **지금 구조 유지가 맞다.**
- 남는 문제는 off-policy 가 아니라 **비정상성·부분관측**: (i) α 가 바뀔 때마다 MDP 가 바뀌어 critic 이 낡음(→ α 를 critic 관측에 / 감쇠 직후 critic 우선 갱신), (ii) actor 가 α 를 모르므로 정책은 "평균 α 에 맞춘 행동"을 학습 — 전역 α 가 계단으로 떨어지면 그 순간 정책은 줄어든 보조를 모른다.
- 슬루 포화 연관: 정책 샘플이 슬루(4°/틱)로 잘리면 저장된 a 와 실행 목표가 다르다(라운드 1 계산: 포화 46–68 % 틱). 보조가 같은 슬루 예산을 쓰므로 α=2 구간에서 포화는 더 늘 가능성(미측정).
  Fujita & Maeda 2018(Clipped Action Policy Gradient, arXiv 1802.07564, **미검증**: 원문 미확인)은 잘린 행동에서 미리 자른 값의 log-prob 을 쓰면 분산이 준다고 보고 — 우리 슬루는 시간 의존 클립이라 그대로 적용 불가, 진단 지표(포화율)만 로그 권장.

#### 2-2. α 감쇠 스케줄 실제 사례 요약 (진폭 vs 비율 vs 길이)
| 출처 | 보조 형태 | 감쇠 단위 | 감쇠 속도·조건 | 제거 후 |
|---|---|---|---|---|
| HoST `host_ground.py:505-508` | env 쪽 위 당김 힘(상태 조건부) | env 별 진폭 | 리셋 묶음 평균 머리 높이 > 문턱마다 −20 N (200 N → 10회), 되돌림 없음 | 평가 시 힘 0(`scripts/play.py:26`) |
| Yu 2018 Alg. 3 | env 쪽 PD 스프링 힘 | 진폭, **에피소드 내 계단(3 s 마다 −25 %)** + 창 전체 이동 | 2p s 균형 & 수익 > 완전보조 수익 70 % | 무보조 최종 학습 별도, "빨리 배운다" |
| Yu 2018 4.2.1 | 같음 | 진폭, line-search | 수익 80 % 도달 → 수익 60 % 유지하는 최대 감소 | 같음 |
| Kickstarting | 증류 손실 | λ_k | PBT 또는 선형 1→0 | 성능 유지·초과 |
| JSRL Alg. 1 | 안내 정책 롤인 | 롤인 길이 h | 평가 ≥ β 일 때 h 감소, 또는 무작위 h | 보고상 유지 |
| DAgger | 전문가 행동 실행 | 확률 β_i | p^{i−1} 지수 | (모방 손실이라 해당 없음) |
| RPL | 기본 제어기 + 잔차 | — | 제거 안 함 | — |
| DAPG `dapg.py:62-70` | 시연 어드밴티지 | λ₀·λ₁^k | λ₁=0.95/반복 | — |

공통점: (1) **성공 조건부**로만 감쇠(HoST·Yu·JSRL), (2) 감쇠 판정에 "보조가 적은 상태에서의 성능"을 포함(Yu: 2p s 이후 계단, 수익 조건), (3) 보조를 **env 별 / 에피소드 내에서 다양화**해 특정 보조 수준에 과적합 방지(Yu·JSRL-Random).

#### 2-3. RSI / 상태 은행의 Isaac Lab 구현 줄 단위
- **BeyondMimic** `whole_body_tracking/tasks/tracking/mdp/commands.py`: `:207-241` 실패 bin 비례 샘플링, `:243-280` `_resample_command` 에서 기준 상태 + 잡음을 `self.robot.write_joint_state_to_sim(...)`, `self.robot.write_root_state_to_sim(...)` 로 직접 기록(관절은 soft limit 으로 clip), `:282-` `_update_command` 에서 시간 끝나면 재샘플. [코드 확인, 라운드 1]
- **Isaac Lab 설치본**(`/home/ryu/IsaacLab`, 37ddf62687): `source/isaaclab/isaaclab/envs/mdp/events.py:1074-1115` `reset_root_state_uniform`, `:1278-1315` `reset_joints_by_offset` — 리셋 이벤트에서 `write_root_pose_to_sim`/`write_joint_state_to_sim` 하는 표준 형태. 상태 은행 RSI 는 이 두 함수를 본떠 "은행에서 인덱스 샘플 → 같은 API 로 기록" 하는 reset 이벤트로 충분.
  리셋 순서 `envs/manager_based_rl_env.py:349-394`: scene.reset → **event(reset)** → observation_manager.reset → action_manager.reset → … → episode_length_buf 0.
- **RSI 함정 3가지(우리 코드 기준) [코드 확인]**:
  1. **관측 이력**: `utils/buffers/circular_buffer.py:136-139` — 리셋 뒤 첫 append 값으로 **이력 전체를 채움**. 우리 actor 관측은 5틱 이력이므로 은행 상태(스윙 중, 움직이는 중)에서 시작하면 첫 5틱은 "정지된 이력"이 된다 — 실물·정상 학습에 없는 관측. 대응: 은행에 이력도 저장해 버퍼에 직접 기록하거나, 첫 5틱을 보상·손실에서 제외(미구현 기능, 래퍼 필요).
  2. **액션 항 상태**: `rl_walking/actions.py:121-131` 리셋 env 의 슬루 기준·지연 버퍼·LPF 상태를 **현재 관절 위치**로 초기화 → RSI 가 관절을 먼저 써 두면 일관(이벤트가 액션 리셋보다 먼저 실행되므로 정합, 위 순서). 단 `last_action_lpf` 관측과 `encoder_joint_vel_lpf`·자이로 지연 버퍼(`mdp_hang`)의 내부 상태도 은행 값으로 맞춰야 함(개별 확인 필요, 미검증).
  3. **위상**: STEP-B/A 의 위상·보조 잔차·`gait_clock` 이 모두 `episode_length_buf` 기반(`mdp_stage4.py:32`, `mdp_hang.py:2272,2416`)이고 리셋 시 0 → 은행 상태의 위상(예 스윙 중간)과 시계가 어긋난다. 대응: 은행 상태의 위상 k 만큼 `episode_length_buf` 를 k 틱으로 설정(시간 제한·밀기 타이밍 영향 확인 필요) 또는 위상 오프셋 버퍼 도입.

#### 2-4. 감쇠 실패 시 대안: env 비율 감쇠 vs 진폭 감쇠
- **진폭 감쇠(현 STEP-A0)**: 장점 — α 가 조금 줄면 부족분을 정책이 채워야 보상이 유지되므로 **내재화 기울기**가 연속적으로 생긴다(Yu·HoST 방식). 단점 — 우리 플랜트에서 α 효과가 강한 비선형(α 1 → 이륙 0.8 %, α 2 → 31 %): 한 계단이 절벽이 될 수 있음.
- **env 비율 감쇠(DAgger·JSRL-Random 식, α ∈ {0, 2} 를 확률 p 로)**: 장점 — 보조 env 는 항상 '효과 있는' 진폭이라 스텝 상태 방문이 보장. 단점 — **보조 없는 env 는 여전히 서기 국소해**이고, 보조 env 에서 정책이 배우는 것은 "α=2 잔차 위에서의 보정"이지 잔차 자체가 아니다. actor 가 어느 env 가 보조인지 모르므로 평균에 맞춘 절충(대략 p·r 만큼 내재화되는 방향, 추론·미검증).
  즉 비율 감쇠 단독은 '방문'은 주지만 '내재화' 압력이 약하다.
- **권장 조합(가설)**: env 별 α 를 구간 [α_lo, α_hi] 에서 균등 샘플(예 시작 [1.5, 2.0]) + 에피소드 내 반주기 계단 감소(Yu) + env 별 성공 시 그 env 의 구간 하향(HoST) + 일부 env(예 20 %)는 항상 α=0(현재 성능 감시용, 판정 지표).
- **감쇠 게이트용 진단 지표(새 제안, 계산 가능)**: 스텝 모드에서 정책 평균의 G21a 대비 변화 Δμ(φ) 를 잔차 r(φ) 에 투영한 **내재화 계수 κ = ⟨Δμ, r⟩ / ⟨r, r⟩**(행동 단위 환산). κ 가 0 근처면 정책은 잔차를 전혀 흡수하지 않은 것 → α 를 내리면 행동이 사라진다고 예측. 성능 게이트(하중 수용 ≥ 50 %)와 함께 **κ ≥ (현재 α − 다음 α) 의 일정 비율**을 감쇠 조건으로. 로그된 액션만으로 학습 중 계산 가능(GPU 추가 실행 불필요).

### 3. 실행 순위 (라운드 2, 상위 3)

**1위 — STEP-A0 를 env 별·에피소드 내 감쇠로 바꾸는 단일 변경 (Yu 2018 Alg. 3 + HoST env 별 감쇠)**
- 근거: Yu 4.2.2("롤아웃 안 여러 보조 수준 → 특정 보조 과적합 방지", 진행 조건에 무보조에 가까운 구간 성능 포함), HoST `host_ground.py:505-508`(env 별 성공 조건부 −20 N 계단).
- 최소 실험: 대조 = STEP-A0 원안(전역 α 계단). 변경 = α 를 env 별 [α_lo, α_hi] 샘플 + 스텝 모드 반주기마다 −0.25 계단, 전역은 창을 내림. 두 실험 모두 α=0 고정 env 20 % 를 평가용으로 둠. 판정 = α=0 env 의 하중 수용·이륙(사전 기준).
- 위험: critic 의 α 모호성(→ critic 관측에 α, 2위와 묶지 말고 별도 변수로), 낙상 증가.

**2위 — 감쇠 게이트에 내재화 계수 κ 추가 + 감쇠 직후 critic 우선 갱신(RPL burn-in)**
- 근거: §2-4(성능만 보면 "보조 덕에 성공"과 "정책이 배워서 성공"을 구분 못 함), RPL 논문(강한 초기 정책 + 미학습 critic 의 초기 붕괴, critic 손실 < β 까지 정책 고정).
- 최소 실험: 학습 변경 없이 먼저 **STEP-A0 학습 로그에서 κ 를 사후 계산**(현재 α 계단마다). κ≈0 인데 성능 게이트가 열리면 → 감쇠 실패를 사전에 예측하는 지표로 확정. 이후 게이트에 반영.
- 위험: κ 는 선형 투영이라 정책이 다른 방식(다른 관절 조합)으로 스텝을 배우면 과소평가 — 보조 지표로만.

**3위 — 감쇠가 절벽에서 실패하면: Kickstarting 형 손실 보조(STEP-K0)를 env 보조와 결합**
- 근거: Kickstarting Sec 5.1.1(선형 λ 감쇠, λ→0 후 성능 유지), §0(actor 가 위상을 관측하므로 r(φ) 표현 가능), 보조가 정책 밖에 있는 한 제거 시 소멸 위험이 구조적.
- 최소 실험: STEP-A0 + 스텝 모드 env 에 λ_k‖μ_θ − (μ_G21a + α r/0.25)‖² (λ 선형 감쇠), 대조 = STEP-A0 단독. 판정 α=0.
- 위험: PPO 손실 변경(동결 원칙과의 관계 — 사용자 승인 필요), 교사가 개루프라 착지·하중 수용 구간에서 정책의 폐루프 보정을 방해할 수 있음 → 손실을 이륙 전·스윙 구간 관절에만 적용 검토.

출처: [HoST](https://github.com/InternRobotics/HoST) · [HoST arXiv 2502.08378](https://arxiv.org/abs/2502.08378) · [Yu et al. 2018 arXiv 1801.08093](https://arxiv.org/abs/1801.08093) · [residual-policy-learning](https://github.com/k-r-allen/residual-policy-learning) · [RPL arXiv 1812.06298](https://arxiv.org/abs/1812.06298) ·
[Kickstarting arXiv 1803.03835](https://arxiv.org/abs/1803.03835) · [JSRL arXiv 2204.02372](https://arxiv.org/abs/2204.02372) · [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) · [Clipped Action PG arXiv 1802.07564](https://arxiv.org/abs/1802.07564)(미검증) · DAgger(Ross et al. 2011, 미검증)

## 라운드 3 (2026-10-04 04:20)

전제(코디네이터 전달, 측정 사실): **STEP-B1 FAIL** — 600 반복에서 비대칭 +0.003 mg·ready 0.2 %. 진행 중 STEP-I0(모방, 기준 = 뒤 스텝 489건 공통 primitive, 앞뒤 방향은 "이륙 위치 + 5 cm" 목표), 다음 STEP-A0(전역 계단 감쇠). 라운드 2 의 env 별·에피소드 내 감쇠는 A1 후보로 기록됨.

### 0. 이번 라운드에서 직접 계산·측정한 것

**(a) 성공 스텝은 이미 슬루 한계 근처에서 움직인다 [측정: `reports/gen_step/ref_steps_g22r2_s1.npz`, 509건(뒤 489 + 앞그룹 20), dt 0.02]**
- 스텝 창(45틱)에서 측정 관절 위치의 틱당 변화: **가장 빠른 관절의 최대 |Δq| 중앙값 0.066 rad/틱(3.8°), 상위 10 % 0.089 rad/틱(5.1°)** — 슬루 한계 0.0698 rad/틱(4°)과 같은 수준.
  관절별 최대 속도 중앙값 1.9–2.9°/틱, 관절별 스텝 중 가동 범위 중앙값 9–14°. 4°/틱 초과 틱 비율은 관절별 0–0.4 %(측정치는 PD 응답이라 목표보다 늦고 작을 수 있음 — 목표 기준 포화율은 미측정).
- 해석: (i) 슬루·LPF 체인 **안에서도 스텝은 가능**하다(G21a 계열이 실제로 뒤 스텝을 함 — 가장 강한 내부 증거). (ii) 그러나 스텝은 **가장 빠른 관절에서 슬루 예산을 거의 다 쓴다** → 탐색 잡음·보조 잔차가 같은 예산을 나눠 쓰면(라운드 1: 잡음만으로 46–68 % 틱 포화) 스텝 궤적을 낼 여유가 줄어든다(가설, 대조 실험 없음).

**(b) 앞 스텝 착지 거리의 LIPM/캡처 포인트 계산 [계산; CoM 높이 h_c 는 가정 — 골반고 ≈0.60 m 를 상한으로 0.50–0.60 m]**
- ω = √(g/h_c) = 4.04–4.43 s⁻¹ (1/ω = 0.23–0.25 s). 순간 캡처 포인트 ξ = x_CoM + v/ω → v/ω = **0.1 m/s → 2.3–2.5 cm, 0.2 → 4.5–4.9, 0.3 → 6.8–7.4, 0.5 → 11.3–12.4 cm**.
- 발 기하(발목축 앞 85 mm / 뒤 65 mm, CLAUDE.md): CoM 이 발목 위에 있을 때 **앞 밀기로 스텝이 '필요'해지는 경계는 ξ > 발끝(+8.5 cm), 즉 v ≳ 0.35–0.38 m/s**. 그보다 약하면 발목 전략으로 충분(이론상) — 학습 앞 밀기 세기가 이 경계 아래면 스텝을 배울 이유가 없다(가설: 앞 밀기 후 실제 CoM 속도는 미측정).
- CoP 를 발끝에 둔 채 하중 소거+공중 T 동안 ξ 는 발끝 기준으로 e^{ωT} 배 발산: T = 0.06+0.15 ≈ 0.21 s(측정 서명: 소거 0.06 s·공중 0.14–0.16 s) → **×2.3–2.5**.
  착지 발목 위치 목표 ≈ ξ(T) − 1 cm(발 중심 = 발목 +1 cm), ξ(T) = 8.5 + (ξ₀ − 8.5)·e^{ωT} (cm, 발목 원점).
  예: ξ₀ = 10 cm → 착지 ≈ **11–12 cm 앞**, ξ₀ = 12 cm → ≈ **16 cm 앞**. 성공한 뒤 스텝 수평 변위 중앙값 18.8 cm 와 같은 규모.
- → **STEP-I0 의 "이륙 + 5 cm" 앞 목표는 스텝이 필요한 상황(ξ₀ > 8.5 cm)에서 LIPM 기준 착지 거리의 1/2–1/3** — 착지해도 ξ 가 새 발끝(5+8.5 = 13.5 cm)보다 앞에 남아 두 번째 스텝·낙상이 필요할 수 있다(계산, h_c·CoP 위치 가정, 미검증).

### 1. 출처별 분석

#### 1-1. ① 착지 목표 휴리스틱을 RL 에 쓴 코드
**(a) walk-these-ways (Margolis & Agrawal, Go1) — Raibert 휴리스틱 보상의 대표 구현**
- URL: https://github.com/Improbable-AI/walk-these-ways · commit **0e7236b (2024-06-16)** [코드 확인]
- `go1_gym/envs/rewards/corl_rewards.py:161-200` `_reward_raibert_heuristic`: 몸 좌표 발 위치 vs 목표 = 공칭 자세(폭 0.3·길이 0.45 m, 명령 13/14번으로 가변) + **phase·v_des·(0.5/f)** (`:184-191`, phase ∈ [−0.5, 0.5]), 오차 제곱합. 가중 **−10**(`scripts/train.py:126`; 기본 0, `legged_robot_config.py:332`).
- 메커니즘: 명령 속도·보행 주파수로 정해지는 발 목표(피드포워드 v·T/2 항만, Raibert 의 속도 오차 피드백 항 k(v−v_des) 는 없음). 사족·명령 추종용 — **밀기 회복 용도가 아님**.
- **우리에게 적용**: W2(클록 보행) 분기에서 착지 목표가 필요하면 그대로 이식 가능한 형태(이족: 폭 고정, 길이 = v·T_st/2). 밀기 회복에는 속도 오차 피드백이 없어서 부적합 → (b)·(c) 의 CP 형을 권장.
  위험: 발 목표 오차 벌점은 **발이 떠야만 줄일 수 있어서** 서기 국소해에서는 기울기가 없다(이륙 탐색 문제를 풀어주지 않음).

**(b) apex `cassie_standing_env.py` — CP 기반 서기 보상(밀기 없음)**
- [코드 확인] `apex/cassie/cassie_standing_env.py:170-191`: "capture_point_vel = 발 사이 거리 × √(9.8/h)" 를 CoM 속도와 맞추는 항, 양발 접촉 시에만(`:183-186`), 전부 공중이면 −0.5.
  CP 정의를 변형해 서기 안정에 쓴 예로, **스텝 유도 효과는 없음**(서기 전용 env). 참고 가치 낮음 — 'CP 를 썼다는 것'만으로 스텝을 기대하면 안 된다는 반례.

**(c) "Embedding Classical Balance Control Principles in RL for Humanoid Recovery" (arXiv 2603.08619, H1-2, Isaac Lab)** — [논문 요약 확인(HTML), 코드 공개 링크 없음]
- r_CP = exp(−d_CP²/σ²), d_CP = **CP 와 발 지지 볼록껍질 사이 부호 거리**, ξ = p_c + ṗ_c/√(g/h_c). CoM 지지 보상과 함께 그룹 가중. **critic 에만** CoM 위치·속도·가속도·운동량·CP(actor 는 고유감각만).
  스텝 보상 없이 발목·고관절·**교정 스텝**·낙상·일어서기가 하나의 정책에서 출현. 밀기 50–300 N × 0.1–0.2 s. 3단계 커리큘럼 **1단계: 토크 한계 10배 → 3단계: 1배로 어닐링**. 50 000 반복.
- **우리에게 적용**: 
  - 착지 목표 대신 **"CP–지지다각형 부호 거리" 보상**(우리 발 기하: 앞 85·뒤 65 mm, 폭은 양발 볼록껍질)을 critic 특권 정보와 함께 쓰는 것이 우리 관측 계약(actor 불변)과 맞는다. 앞 밀기에서 ξ 가 발끝 밖이면 서 있는 한 보상이 계속 낮고, 앞으로 디뎌 다각형을 넓히는 순간 회복 — **착지 결과를 직접 채점**(시연 불필요).
  - 최소 실험: STEP-I0 와 같은 출발점에서 r_CP 하나만 추가(앞 밀기 env), 대조 = 무변경. 판정 = 앞 밀기 이륙률·회복률.
  - 위험: B0/B1 이 보여준 것처럼 **결과 보상만으로는 첫 이륙이 탐색되지 않을 수 있다**(r_CP 는 이륙 전엔 '몸을 뒤로 젖히기'로만 개선 가능 → 발목·고관절 전략 강화 쪽으로 기울 수 있음). 그래서 A0(보조로 방문) + r_CP(앞 착지 채점) 결합이 현실적. h_c 는 시뮬 CoM 에서 직접 계산(가정 불필요).
  - 1단계 "토크 한계 10배" 어닐링은 ③ 의 근거로 아래 3-1 에서 다룸.

**(d) STEP-I0 앞 목표에 대한 적용(①의 결론)**: 시연이 없으므로 착지 목표를 **상태 의존 공식**으로 만든다 — 이륙 순간의 CoM 위치·속도로 x_land = x_ankle0 + 8.5 + (ξ₀ − 8.5)·e^{ω·T_air} − 1 (cm), [최소 6, 최대 20] cm 로 clip(최대는 뒤 스텝 측정 상위값 25 cm 이하에서 선택).
  "+5 cm 고정" 대비 단일 변수 실험. 위험: LIPM 은 발목 토크·상체 회전을 무시 → 시뮬에서 공식 착지점과 실제 회복 착지점(G22R2 의 뒤 스텝 489건, 이륙 순간 CoM 속도 로그가 있으면)을 먼저 비교해 **ω·오프셋을 데이터로 맞추는 검증**이 선행돼야 함(뒤 방향으로는 지금 데이터로 바로 가능 — 뒤꿈치 6.5 cm 로 바꿔 같은 식).

#### 1-2. ② 접촉 클록 보행 구현 수치 (W2 분기)
| 저장소 (commit) | 위상·주기 | 접촉/발 보상 | 서기(명령 0) 처리 | 파일:줄 |
|---|---|---|---|---|
| humanoid-gym (ae46e20) | t/0.64 s, sin≥0 왼발 지지, **\|sin\|<0.1 이중지지** | `feet_contact_number` 일치 +1 / 불일치 −0.3 (가중 1.2), **`joint_pos` 기준 관절 사인 궤적 추종**(진폭 0.17 rad, 무릎 2배, 가중 1.6), feet_air_time 1.0, clearance 1.0 | 명령 노름 < 0.2 → 0, **클록은 계속(제자리걸음)** | `humanoid/envs/custom/humanoid_env.py:100-142,272-279,336-344`; `humanoid_config.py:174-207`; 명령 0 처리 `envs/base/legged_robot.py:336` |
| unitree_rl_gym G1 (276801e) | 주기 0.8 s, 반대발 0.5 오프셋, **지지 = 위상 < 0.55** | `_reward_contact` = 발별 XNOR(접촉, 지지) 합 (가중 0.18), `feet_swing_height` = (z−0.08)²·비접촉 (−20) | 명령 0 이어도 클록·보상 그대로 | `legged_gym/envs/g1/g1_env.py:58-63,98-109`; `g1_config.py:85-89`; 명령 0 `envs/base/legged_robot.py:306` |
| MuJoCo Playground Berkeley Humanoid (ef4fefc) | 주파수 **U(1.25, 1.5) Hz 무작위**, 위상 [0, π] | `feet_phase` = exp(−Σ(z−r_z(φ))²/0.01), r_z = 3차 베지어 발 높이(최대 0.1 m) — **명령으로 게이트 안 함(TODO 주석)** | stand_still 가중 0.0 | `_src/locomotion/berkeley_humanoid/joystick.py:73-85,228-229,329-330,648-665`; `_src/gait.py:32-43` |
| isaac_berkeley_humanoid (ffc7b26) | 클록 없음 | `feet_air_time`(**공중 0.2–0.5 s** 만 보상, 가중 2.0) | `rel_standing_envs=0.02` | `velocity_env_cfg.py:96-106,273-280`; 명령 커리큘럼 `:331-333` + `mdp/curriculums.py`(추적 보상 > 최대의 80 % 면 범위 +0.5 m/s, 200 반복마다, 5000 반복 이후) |
| Isaac Lab G1 (설치본 37ddf62687) | 클록 없음 | `feet_air_time_positive_biped`(단일지지 중 min(지지·공중 시간), 상한 0.4 s, 가중 0.25), 명령 노름 ≤ 0.1 이면 0 | `rel_standing_envs=0.02` | `isaaclab_tasks/.../velocity/mdp/rewards.py:49-68`; `config/g1/rough_env_cfg.py:32-38`; `velocity_env_cfg.py:97` |
| LearningHumanoidWalking (cd8c655) | 라운드 1 §1-1 | 발별 부호 선형 힘·속도 | STANDING 모드 5 %(clock 상수) | 라운드 1 참조 |

관찰 및 **우리에게 적용**:
- **서기 정책에서 출발해 클록 보행을 만든 공개 사례는 이번 조사에서 찾지 못했다**(전부 scratch 학습). 클록 감쇠 사례도 apex `early_clock_reward`(라운드 1) 외엔 없음. → W2 의 "서기에서 출발" 자체가 검증되지 않은 경로.
- 공통 설계: (i) 클록은 명령 0 에서도 돌거나(humanoid-gym·unitree·Playground) 서기 env 비율을 2–5 % 로 낮게 둠 → 우리 '정지 서기 우선' 목표와 충돌(OSS_REFERENCES §3 에서 위상 시계 상시는 이미 기각). W2 는 B0 처럼 **명령 게이트된 클록** 유지가 맞다.
  (ii) 발 접촉 일치 보상은 **불일치에도 −0.3(humanoid-gym)·XNOR 합(unitree)** 처럼 "지금 한 발이라도 맞으면 부분 점수" — 곱이 아니라 합. (iii) humanoid-gym 은 **기준 관절 사인 궤적 추종(joint_pos 1.6)을 가장 큰 항 중 하나**로 둔다 → 사실상 모방 부트스트랩(STEP-I0 와 같은 계열).
- **주의(코드 확인, 함정)**: humanoid-gym `use_ref_actions=True` 옵션(`humanoid_config.py:49`, 기본 False)은 `humanoid_env.py:190-191` 에서 `actions += self.ref_action` 을 **제자리(in-place)** 로 수행하는데, 이 텐서는 러너가 넘긴 `transition.actions` 와 같은 객체(`algo/ppo/ppo.py:93-101` 이 그것을 반환, `on_policy_runner.py:129-130`, 저장은 step 이후 `rollout_storage.py:92` copy) → **저장된 행동 = 정책 샘플 + 기준인데 log-prob 은 샘플 기준**(라운드 2 §2-1 의 ratio 편향 그대로). 우리 STEP-A0 훅은 새 텐서를 반환(`mdp_hang.py:2421`)하므로 해당 없음 — 구현을 바꿀 때 in-place 금지 확인 항목.
- **공중 시간 문턱 주의**: Berkeley `feet_air_time` 은 0.2 s 미만 공중을 보상 0(또는 음) 처리. 우리 성공 스텝 공중 0.14–0.16 s 는 **이 문턱이면 벌을 받는다**. W2 에서 air-time 항을 쓰면 문턱 ≤ 0.1 s(또는 Isaac Lab biped 형 min(지지,공중))로.
- 속도 커리큘럼(Berkeley): 추적 보상 > 80 % 일 때만 범위 +0.5 m/s. W2 에서 vx 를 0.05 부근에서 시작해 같은 규칙으로 키우는 근거.

#### 1-3. ③ 슬루 제한·LPF 체인에서의 스텝 학습
- **DeepMind OP3 축구 (Haarnoja et al., Science Robotics 2024, arXiv 2304.13653)** [논문 요약 확인(HTML)]: 40 Hz 관절 위치 목표에 **지수 필터 u_t = 0.8·u_{t−1} + 0.2·a_t**(= 우리 β 0.2 와 같음) 를 걸고도 걷기·회전·차기·일어서기를 학습. 필터가 학습·탐색에 준 영향은 논문이 논하지 않음. 슬루 제한 언급 없음. → **LPF 단독은 스텝 학습을 막지 않는다**는 외부 근거(단 scratch 학습·다른 알고리즘(MPO 계열)·소형 로봇).
- **motion_imitation (Peng et al. 2020, Laikago)** [코드 확인, commit d0e7b96]: 4 Hz 2차 Butterworth 행동 필터 구현(`motion_imitation/robots/action_filter.py:42-44`), 명령을 **현재 관절각 ±0.2 rad 로 클립**(`robots/laikago.py:363-380`, 슬루가 아니라 위치 오차 제한). 그러나 공개 env 빌더 기본값은 둘 다 **꺼짐**(`envs/env_builder.py:54-55,144-145`). 근거로 약함.
- **arXiv 2603.08619(위 1-1c)**: 학습 1단계에서 **토크 한계 10배 → 1배 어닐링** — 구동 제약이 초기 탐색을 막는다는 판단을 커리큘럼으로 반영한 사례(제약 완화 자체의 대조 실험은 요약에서 확인 못 함 → 미검증).
- 슬루 제한이 PPO 탐색을 막는다는 **직접 실험 근거는 찾지 못했다**. 우리 쪽 근거는 라운드 1(포화율 46–68 %) + 위 0(a)(성공 스텝이 한계 근처) 두 계산·측정뿐.
- **우리에게 적용**: 
  - 배포 계약은 슬루 4°/틱이므로 최종 정책은 같은 체인에서 판정해야 한다(CLAUDE.md 3번). 그 안에서 할 수 있는 단일 변수 실험: **학습 초기에만 슬루를 완화(예 8°/틱)했다가 4°/틱으로 어닐링**(2603.08619 의 토크 10배→1배와 같은 구조). 대조 = 슬루 4°/틱 고정, 같은 STEP 설정·seed. 판정은 슬루 4°/틱 평가에서만.
  - 그 전에 싸게 확인할 진단: STEP-A0 α=2 롤아웃의 **목표 기준 슬루 포화율**(관절별, 스텝 창 vs 서기 창)을 로그. 스텝 창에서 포화가 거의 100 % 면 "보조 잔차 + 정책이 예산을 다 씀" 이 측정으로 확인됨.
  - 위험: 슬루 완화 중 배운 빠른 동작이 4°/틱 로 돌아오면 사라지는 '보조 제거' 와 같은 문제(라운드 2), 실물 지연·떨림 특성 변화. 실물 안전 가드와 무관한 학습 전용 변경임을 명시해야 함.

### 2. 실행 순위 (라운드 3, 상위 3)

**1위 — 앞 착지 목표를 상태 의존 CP 공식으로 (STEP-I0 의 "+5 cm 고정" 대체, 단일 변수)**
- 근거: §0(b) 계산 — 스텝이 필요한 앞 밀기에서 LIPM 착지 거리 ≈ 11–16 cm, 고정 +5 cm 는 1/2–1/3. arXiv 2603.08619 의 CP 부호거리 보상(시연 없이 교정 스텝 출현). 뒤 스텝 18.8 cm 와 규모 일치.
- 선행 검증(학습 없음): G22R2 뒤 스텝 489건에서 이륙 순간 CoM 속도로 공식 착지점을 계산해 실측 착지점과 비교(뒤꿈치 6.5 cm 로 바꿔). 오차가 크면 ω·오프셋을 데이터로 맞춤. 그 뒤 STEP-I1 = I0 + 공식 목표.
- 위험: LIPM 단순화, h_c·CoP 가정, 첫 이륙 탐색 문제는 그대로(→ A0 와 결합).

**2위 — 앞 밀기 세기 점검: 학습 앞 밀기가 "스텝 필요 경계"(ξ > 발끝, v ≳ 0.35 m/s 추정)를 넘는가**
- 근거: §0(b). 경계 아래 밀기에서는 발목 전략이 최적이라 스텝을 배울 이유가 없다. 라운드 1 Atalante("점진 증가 → 조기 수렴"). 측정 사실: 앞 실패 = 시도 없음.
- 최소 진단(평가 로그만, 학습 없음): 앞 밀기 평가·학습 표본의 밀기 직후 최대 CoM 속도 분포와 ξ − 발끝 분포를 기록. 대부분이 경계 아래면, 앞 밀기 표본의 일부(예 30 %)를 경계 위 고정 세기로 바꾸는 단일 변수 실험.
- 위험: 경계 위 밀기는 G21a 에겐 낙상만 늘릴 수 있음 → 보조(A0)나 RSI 와 함께여야 학습 신호가 생김.

**3위 — 슬루 포화 진단 → (필요 시) 학습 초기 슬루 완화 어닐링**
- 근거: §0(a) 성공 스텝이 슬루 한계 근처(가장 빠른 관절 3.8°/틱 중앙값), 라운드 1 잡음 포화 46–68 %, arXiv 2603.08619 토크 한계 10배→1배 어닐링, OP3 는 LPF 만으로 스텝 학습(→ 병목은 LPF 보다 슬루일 가능성, 가설).
- 최소 실험: 먼저 A0 α=2 롤아웃의 목표 기준 포화율 로그(학습 없음). 포화가 스텝 창에서 지배적이면 슬루 8→4°/틱 어닐링 vs 4°/틱 고정 대조.
- 위험: 배포 계약과 다른 체인에서의 학습 — 최종 판정은 반드시 4°/틱 체인, 어닐링 끝 성능이 고정 대조보다 나을 때만 채택.

출처: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [apex](https://github.com/osudrl/apex) · [arXiv 2603.08619](https://arxiv.org/abs/2603.08619) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) ·
[mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [OP3 soccer arXiv 2304.13653](https://arxiv.org/abs/2304.13653) · [motion_imitation](https://github.com/google-research/motion_imitation) ·
[A2CF arXiv 2506.23125](https://arxiv.org/abs/2506.23125)(학습형 보조력 에이전트, 초록만 확인 — 라운드 2 보조 감쇠의 추가 사례, 세부 미검증) · Pratt et al. 2006 Capture Point(이론, 원문 미재확인)

## 라운드 4 (2026-10-04 04:45) — RSI/상태 은행 구현 설계서 (A0 실패 대비, 코드 변경 없음)

전제(측정 사실): STEP-I0 FAIL(400 반복 이륙 0; joint_pose_ref 0.53·foot_ref 0.23·contact_ref 0.79 = 서기 자세 근처 값). B0·B1·I0 보상 3종이 같은 방식으로 실패 → 탐색 구조 문제로 판정. A0(보조 α 2→0) 학습 중.
범위: 이 절은 **설계서**다. 아래 파일:줄은 읽기만 했고, 제안한 변경은 하나도 구현하지 않았다. 근거 계보: A0 `rl_walking/hang_stand_env_cfg.py:2174-2182` ← I0 `:2151` ← B1 `:2136` ← B0 `:2102` ← G21a `:1861`.
실행 중인 A0 설정은 `logs/rsl_rl/biped12_stepb0/2026-10-04_03-51-07_stepa0_seed1_from_g21a11300/params/env.yaml` 에서 읽었다(이하 "A0 env.yaml").

### 0. 설계 결론 먼저

- 우리 env 에는 **에피소드 시작 판정(`episode_length_buf` == 0 또는 ≤ 1)에 묶인 내부 상태가 20곳 이상** 있다(§2 표). 스냅샷 RSI(경로 A)를 하려면 이 상태를 모두 은행 값으로 맞추거나, "RSI 로 시작한 env 는 fresh 초기화를 건너뛴다"는 분기를 각 함수에 넣어야 한다. 어느 쪽이든 코드 변경 범위가 크다.
- 그래서 **2단계 설계**를 권장한다.
  - **경로 B(동적 롤인)**: 상태를 쓰지 않는다. 에피소드 안에서 보조(α=2 잔차, A0 훅 재사용)를 **처음 h 반주기만** 켜고 뒤는 α=0. 모든 내부 상태가 자연 진화하므로 동기화 문제가 없다. 라운드 2 §1-5(JSRL 롤인 길이) 와 같은 구조이고 A0 코드 변경이 가장 작다.
  - **경로 C(속도 RSI)**: 앞 스텝 상태가 없으므로, 표준 서기에서 온셋이 끝난 뒤 **몸통 앞 방향 속도**를 캡처 포인트 경계 위로 부여(§3).
  - **경로 A(스냅샷 RSI)**: B·C 가 실패하고 "스윙 중간 상태 방문"이 부족하다는 진단이 나올 때만. §2 의 동기화와 §6 의 재현 검사를 통과한 뒤에만 학습에 쓴다.
- 판정은 셋 다 **표준 서기 시작 + 기존 평가기**에서만 한다(§5).

### 1. ① 상태 은행 출처와 저장 항목

**현재 npz 에 있는 것** [코드 확인: `reports/gen_step/extract_ref.py:5-6,44-63,65-103`]
- 출처: G22R2-13099 롤아웃(앞 15 N 램프 / 뒤 18 N 충격 0.3 s, `t0` 3 s). 유효 스텝 = 이륙 ≥ 60 ms & 재착지 수평 ≥ 2 cm.
- 창: 이륙 기준 [−15, +30) 틱(−300 ms ~ +600 ms), `i_td` 는 창 기준 착지 틱. 509건(뒤 489 + 앞그룹 20; 앞그룹도 전부 뒤로 디딘 반동 스텝 — `CORRECTION_1004_forward_steps.md`).
- 필드: `q`·`qd`(물리 관절각/속도, **오른발 스윙은 미러해서 '왼발 스윙' 형으로 저장** `:83-84`), `fz`·`c`([스윙, 지지] 순서로 재배열), `frel`(골반 기준 요 프레임 발 위치), `roll`·`pitch`(deg), `h`(골반고), 사건 시각.
- **루트 상태는 6-DoF 중 높이·롤·피치만 있다.** 요·수평 위치·선속도·각속도가 없고, 관측·액션 체인의 내부 상태도 없다.

**은행용 재추출 시 추가할 항목** (같은 롤아웃 스크립트에 기록 항목만 추가. 미러는 저장 뒤 별도 처리하고 원본은 미러하지 않은 채 보관)

| 분류 | 항목 | 이유 |
|---|---|---|
| 루트 | `root_pos_w`(z 필수, xy 는 env 원점 상대), `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w` | 물리 상태 재구성. 속도가 없으면 스윙 중 상태가 '정지된 자세'가 된다 |
| 관절 | `joint_pos`, `joint_vel`(현재 있음, 미러 전 원본으로) | 물리 |
| 액션 체인 | `_prev_target`, `_buf`(지연 링버퍼 전체 + `_delay`), `_lpf_state`·`_beta`, `_bl_out`·`_bl`, `_on_t`, 원시 정책 행동 최근 5틱 | §2 동기화(`rl_walking/actions.py:40-79,117-158`) |
| 관측 상태 | `_gyro_lag_st`(buf·lag), `_s8_gyro_ema`, `_s8_enc_jvel_ema`, `_jv_lpf_st.y`, `_enc_prev`/`_enc_vel`, `_la_lpf_st.y`, `_fcr2_st`(held·q·phase·thr·fthr), `_grav_buf`·`_imu_lag`, **정책 관측 그룹 이력 5틱 전체(정규화 전 원값)** | §2 |
| 사건·스케줄 | sb0 상태(`prev_c`·`air_t`·`lift_xy`·`lift_z`·`valid`), 그 틱의 외란 힘(`dst_F_now`, 남은 슬롯), 회복 게이트 상태(`_g2`·`_g22`), 이륙 기준 상대 틱 τ, 스윙발 id(미러 여부) | §2·§3 |
| DR | 그 env 의 DR 표본(질량·CoM·게인·마찰·유격·쿨롱·기본자세 편향) | 재현 검사(§6)용. 학습 때는 새 env 의 DR 을 그대로 쓴다(복원하지 않음) |

**어느 틱을 쓰나**: 이륙 −80 ms(창 인덱스 11) ~ 착지 +100 ms(`i_td`+5). 성공 스텝 서명(하중 소거 0.06 s, 공중 0.14–0.16 s, 하중 수용 0.08 s)을 덮는 범위로, 스텝당 약 17–19 상태이고 489건이면 약 8 900 상태다(좌우 미러 포함 시 2배).
bin 은 5개로 나눈다: B1 소거 전(−80~−40 ms) / B2 소거(−40~0) / B3 공중 전반 / B4 공중 후반 / B5 착지~하중 수용. 실패율 비례 샘플링(§4)의 단위다.

위험
- 출처 정책은 G22R2 이고 학습 출발점은 G21a(11300)다. 은행 상태의 '이전 행동 이력'은 G22R2 의 것이라 G21a 정책이 낼 이력과 다르다. 영향은 작다고 보지만 미검증.
- 뒤 밀기 충격이 이륙 시점에도 남아 있는 건이 있을 수 있다(충격 0.3 s, 밀기→이륙 0.4–0.58 s). 이륙 시점에 `dst_F_now` > 0 인 건은 제외하거나 힘을 함께 복원해야 한다.

### 2. ② 리셋 때 동기화해야 할 내부 상태

리셋 순서 [코드 확인, Isaac Lab 설치본]: `step()` 안에서 `episode_length_buf += 1`(`envs/manager_based_rl_env.py:201`) → 종료·보상 → `_reset_idx`(`:221`; 그 안에서 reset 이벤트 `:362` → 관측 매니저 리셋 `:369` → 액션 매니저 리셋 `:372` → … → `episode_length_buf` = 0) → 명령 계산(`:232`) → **interval 이벤트**(`:235`) → **관측 계산 + 이력 추가**(`:238`).
즉 리셋 직후 첫 관측은 `episode_length_buf == 0` 상태에서 계산되고, 우리 상태 함수 대부분이 이 틱에 "현재값으로 재시작"한다.
**reset 이벤트에서 상태를 써도 관측 함수가 그 직후 fresh 분기로 덮어쓴다.** 경로 A 의 핵심 설계 결정이 여기서 나온다.

| # | 내부 상태 | 위치(파일:줄) | 현재 리셋 동작 | 경로 A 처리 | 위험(처리 안 할 때) |
|---|---|---|---|---|---|
| 1 | 관측 이력 5틱 (정책 그룹 10항목 × 5) | Isaac Lab `utils/buffers/circular_buffer.py:96-106`(reset: `_num_pushes`=0), `:136-139`(첫 append 가 전체를 채움); `managers/observation_manager.py:301-309` | 첫 프레임 5회 복제 | 은행에 저장한 이력 4틱을 버퍼에 쓰고 `_num_pushes`=4 로 두어 첫 append 가 5번째가 되게 한다(비공개 속성이므로 Isaac Lab 버전에 묶임). 이 쓰기는 reset 이벤트가 아니라 **관측 리셋 이후**에 해야 하므로 interval 이벤트(`:235`)에서 해야 한다 | 스윙 중인데 이력은 '정지' → 실물·정상 학습에 없는 관측(속도·시계 이력이 상수) |
| 2 | `_ema_state`(gyro_ema, enc_jvel_ema) | `mdp_hang.py:885-893`(fresh = elb ≤ 1, **2틱 동안** 현재값 재시작) | 현재값으로 재시작 | `fresh & ~rsi_mask` 분기 추가 + 은행 값 주입 | α=0.1(A0 env.yaml `gyro_ema`·`joint_vel_ema`)은 시정수 10틱(200 ms)이다. 스윙 내내 이력 없는 EMA |
| 3 | 관절속도 LPF 관측 | `mdp_hang.py:1559-1570`(elb==0 → y = x) | 현재 유한차분값 | 은행 `y` 주입 | 아래 #4 와 겹치면 첫 틱 0 |
| 4 | 엔코더 유한차분 속도 | `mdp_hang.py:1300-1322`(elb==0 → **vel = 0**, prev = 현재) | 첫 틱 속도 0 | 은행 `qd`(엔코더 기준)로 `_enc_vel`·`_enc_prev` 주입 | 스윙 중 관절속도가 첫 틱 0 → #3 LPF(β 0.25)가 4틱 이상 과소 → 정책이 '멈춘 다리'를 봄 |
| 5 | 엔코더각(유격 모델) θ_enc | `mdp_hang.py:1237-1244`(elb==0 → θ_ref := θ_j) | 엔코더 = 링크 | 은행 `_prev_target`(=θ_ref)을 액션 체인과 같이 복원하면 자연 일치 | 발목 앞뒤 유격 4°(A0 env.yaml `ankle_f_play` play_rad 0.0698 × 0.75–1.25) 안 이동이 첫 틱에 사라짐 |
| 6 | 자이로 지연 버퍼 | `mdp_hang.py:1407-1430`(elb==0 → buf 를 현재값으로, lag 재추출) | 지연 0~3틱 버퍼 = 현재 | 은행 자이로 최근 4틱 주입(lag 은 새로 뽑아도 됨) | 지연 동안 각속도 이력 상수 |
| 7 | 중력(자세) 지연 버퍼 | `mdp_stage4.py:624-642`(`_grav_reset_ids` → 현재값; `resample_calib_bias` `:393-441` 가 설정) | 현재값 | 은행 중력 이력 주입 | 같음(롤·피치 변화가 큰 스윙 구간) |
| 8 | 직전 행동 LPF 관측 | `mdp_hang.py:1539-1556`(elb==0 → y = 원시 행동) | 첫 정책 출력 | 은행 `_la_lpf_st.y` 주입 | 첫 틱 '직전 행동'이 현재 행동과 같음 |
| 9 | 접촉 관측 hold·전송 지연 | `mdp_hang.py:1490-1536`(fresh → 임계·위상 재추출, held = 현재 비트, 지연 큐 = held) | 현재 접촉으로 일관 | **그대로 둬도 된다**(현재 접촉으로 채워지므로 일관). 단 10 Hz hold 위상은 새로 뽑힘 | 작음 |
| 10 | 액션 슬루 기준·지연 버퍼·LPF·백래시 | `actions.py:117-131`(`_needs_init` → `_prev_target`·`_buf`·`_lpf_state`·`_bl_out` = **현재 관절각**), `:175-196` reset | 목표 = 현재 관절각 | 은행 값 주입 + `_needs_init[ids]`=False. 액션 매니저 리셋(`:372`)이 `_needs_init` 을 True 로 만들므로 그 뒤에 써야 함 | 목표 = 현재 관절각 → PD 가 스윙을 '정지' 명령. 슬루 4°/틱 때문에 회복에 수 틱. 성공 스텝은 슬루 한계 근처(라운드 3 §0a)라 치명적일 수 있음 |
| 11 | **온셋 인계(홀드→블렌드)** | `actions.py:153-158`(목표 = 기본자세×(1−α) + 정책×α), 설정 A0 env.yaml `onset_hold_range (0, 0.3)`·`onset_blend_range (1.0, 1.5)` | 리셋 뒤 **최대 1.8 s 동안 정책 권한이 0→1** | 은행 env 는 `_on_t` ≥ hold + blend 로 설정(온셋 끔) | **가장 치명적.** 스윙 중간에 목표가 기본자세로 끌려가 스텝이 반드시 실패. 경로 C(속도 부여)도 온셋 이후에 해야 함 |
| 12 | 클록 위상 | `mdp_stage4.py:25-42`(φ = 2π·f·elb·dt), `mdp_hang.py:2272-2275`(sb0), `:2355`(sbi), `:2416`(보조 훅) | elb=0 → φ=0 | **`episode_length_buf` 를 바꾸지 말 것**(아래). env 별 위상 오프셋 버퍼 `env._phase_off_ticks` 를 두고 4곳이 `elb + off` 를 쓰게 한다 | elb 를 직접 키우면 #2·#4·#6·#8·#13–#16 의 fresh 판정(≤1, ==0)이 전부 꺼지고, 외란 스케줄 t0(2–10 s, `mdp_hang.py:1603,1656`)·'시작 2 s 제외' 통계(`:2064,2207`)·타임아웃이 함께 어긋남 |
| 13 | sb0 사건 상태 | `mdp_hang.py:2238-2250`(상태), `:2267-2271`(fresh → 0, `prev_c`=1, half=−1) | 이중지지 가정 | 은행의 `prev_c`·`air_t`·`lift_xy`(월드 → 새 env 좌표로 변환)·`lift_z`·`valid` 주입 | 공중 상태에서 시작하면 첫 틱에 '이륙'이 은행 지점에서 기록 → 앞축 변위(3–10 cm 판정)·공중 ≥ 60 ms 판정이 틀림. 공중 후반 bin 은 착지가 **집계 안 됨** |
| 14 | sbi 반주기 상태 | `mdp_hang.py:2372`(elb ≤ 1 → 새 반주기) | — | 위상 오프셋(#12)과 함께 | 모방 보상 위상 어긋남 |
| 15 | 회복 게이트·G21/G22 상태 | `mdp_hang.py:1656-1658,1935-1937`(fresh ≤ 1), A0 는 `recovery_gate_step` `:295` | 서기 상태로 시작 | 은행 env 는 게이트를 'RECOVERY 열림'으로 시작(외생 무장 신호와 같은 취급) | 회복 중인데 STAND 벌점(자세·접촉)이 적용됨 |
| 16 | 외란 스케줄·hang 램프 토크 | `disturbance_reset_g22r` `mdp_hang.py:1850-1887`(t0 2–10 s), `hang_support_reset` `:515-`(A0: ramp_start 1–8 s, cap 2–12 Nm) | 에피소드 시간 기준 | 은행 env 는 첫 1 s 동안 외란 슬롯 비우기(이후는 정상) | 착지 직후 새 외란이 겹쳐 실패율이 과대(실패율 비례 샘플링을 왜곡) |
| 17 | 액추에이터 내부 상태 | `actuators_play.py`: `DeadzoneCoulombImplicitActuator.reset` `:155-`(쿨롱 재표본; A0 은 발목 앞뒤만), SEA `_theta_m` `:236-261`(A0 미사용) | 재표본 | 그대로(DR) | 작음. SEA 를 쓰는 env 로 바꾸면 `_theta_m` 복원이 필수 |
| 18 | 기본자세 편향 이벤트 | `resample_default_pose_bias` `mdp_hang.py:1076-` → `:1144` 근처 `write_joint_state_to_sim(q, 0)` | **관절 위치를 다시 쓰고 속도를 0 으로** | RSI 쓰기 이벤트를 cfg 에서 **이 이벤트 뒤**(reset 이벤트 목록 맨 끝)에 둔다 | RSI 관절 상태가 덮어써짐 |
| 19 | 루트 리셋 이벤트 | A0 env.yaml `reset_base`(reset_root_state_uniform, 요 ±π·xy ±0.3·속도 0), Isaac Lab `envs/mdp/events.py:1074-1115` | 서 있는 자세 | RSI 이벤트가 그 뒤에 루트 6-DoF 를 덮어씀. 요는 은행 값 + 무작위 회전(수평 성분만 같이 회전) | 속도를 월드/몸통 프레임 혼동(메모리: sim push frame gotcha) |

권장 구현 형태(경로 A, 미구현):
(1) reset 이벤트 `bank_reset`(목록 맨 끝)에서 물리 상태를 쓰고 `env._rsi_mask[ids]`=True, `env._rsi_idx[ids]`=은행 인덱스로 둔다.
(2) interval 이벤트 `bank_sync`(interval 0, `:235`, 관측 계산 직전)에서 #1·#10·#11·#13·#15 를 주입한다.
(3) 상태 함수 #2–#8 은 `fresh & ~env._rsi_mask` 일 때만 재시작하고, rsi env 는 은행 값을 1회 주입한 뒤 마스크를 내린다.
(4) 클록 4곳은 위상 오프셋을 더한다.
변경 파일은 `mdp_hang.py`(약 10개 함수), `mdp_stage4.py`(2), `actions.py`(1), 새 이벤트 2개다. 범위가 크므로 §6 재현 검사가 합격 조건이다.

경로 B(동적 롤인, 권장 1순위)의 동기화 필요 항목은 **#11 하나**다. 보조 창이 온셋 블렌드가 끝난 뒤 시작해야 한다. 다만 클록 위상이 elb 기반이라 보조 창 시작 시각을 반주기 경계에 맞춰야 하고, A0 커리큘럼 집계(§4 위험)도 함께 손봐야 한다.

### 3. ③ 앞 스텝 상태가 없을 때

- **좌우 미러는 유효하다.** `rl_walking/symmetry.py:70`(`_JOINT_PERM`/`_JOINT_SIGN`), `:26`(각속도 (−wx, wy, −wz)), `:34`(선속도 (vx, −vy, vz)). 루트 쿼터니언은 롤·요 부호 반전, 위치 y 반전. 현재 npz 는 이미 '왼발 스윙' 형으로 표준화돼 있어 미러로 오른발 상태를 만든다. 은행 크기 2배, 정책의 미러 손실(`agents.py:70-72` use_mirror_loss)과도 정합.
- **앞뒤 반전은 쓰지 않는다.** 무릎 굽힘 방향과 발 비대칭(발목축 앞 85 mm / 뒤 65 mm) 때문에 반전 상태는 물리적으로 다른 로봇이다.
- **경로 C: 속도 RSI** — 서기 상태에 앞 방향 루트 속도만 준다.
  - 크기: 캡처 포인트 경계 위. 라운드 3 §0b: CoM 이 발목 위일 때 ξ > 발끝 ⇔ v ≳ 0.35–0.38 m/s(h_c 0.50–0.60 m 가정) → **v ∈ [0.30, 0.60] m/s** 균등(경계 아래 일부 포함, 발목 전략과 대조). 각속도는 강체 낙하와 맞추려면 피치율 ≈ v/h_c(0.6–1.2 rad/s)를 함께 줄 수 있다. 단일 변수 원칙상 1차는 선속도만.
  - **방향은 몸통 앞(+x) 기준으로 월드에 회전해서 쓴다.** Isaac Lab `push_by_setting_velocity` 는 월드 프레임에 그대로 더하고(`envs/mdp/events.py:1065-1071`), 우리 리셋 요는 ±π 무작위다. `disturbance_reset_g22r` 가 이미 같은 계산을 한다(`mdp_hang.py:1880-1883`, 몸통 +x → 월드).
  - **시점은 리셋이 아니라 온셋 블렌드가 끝난 뒤(t ≥ 2 s)**(#11). 즉 "리셋 RSI"가 아니라 시간 지정 1회 속도 부여다. 구현은 기존 `push_with_arm_g22`(`mdp_hang.py:1890-1901`, 현재 ±0.25 m/s 월드 xy 균등 — 경계 아래이고 **A0 branch "w1" 에서는 아예 꺼짐** `:1897-1898`)를 앞 방향 전용·고정 크기 범위로 바꾼 형태다. 회복 게이트 무장 래치(`push_with_arm`)를 그대로 받는다.
  - 경로 B 와 결합할 수 있다. 속도 부여 직후 h 반주기만 보조를 켜면 '앞 밀림 + 디딤' 상태가 만들어진다. 단 A0 잔차는 좌우 방향 관절·무릎만 담고 있어(`mdp_hang.py:2325-2328` 주석) 앞축 착지 거리를 직접 만들지 않는다 → 앞 착지는 라운드 3 1위(CP 공식 목표) 보상과 짝이어야 함(미검증).
  - 위험: (a) G21a 는 경계 위 속도에서 '시도 없이 넘어짐'이 측정된 상태라, C 단독은 낙상 표본만 늘릴 수 있다(라운드 3 2위와 같은 이유로 B 와 결합). (b) 0.6 m/s 상한은 착지 거리 ~16 cm(계산)로 뒤 스텝 실측 범위 안이지만 앞 스텝 기구학 가능성은 미검증(`CORRECTION_1004` 에서 미검증으로 되돌림).

### 4. ④ 샘플 비율·실패율 비례 샘플링·서기 시작 유지

- **비율(초기값 제안)**: 표준 서기 시작 **≥ 60 %**(G21a 의 서기·뒤 회복 능력 보존, ProtoMotions `init_start_prob` 0.2 의 반대 방향 해석 — 라운드 1 §1-5), 경로 B 롤인 env 20 %, 경로 C 속도 RSI env 20 %. 경로 A 를 쓸 때는 B 자리를 대체한다. 비율 자체는 단일 변수로 따로 실험한다.
- **실패율 비례 샘플링**(경로 A 의 bin B1–B5, 경로 B 의 롤인 길이 h, 경로 C 의 속도 구간): BeyondMimic `commands.py:207-241` 방식을 그대로 쓴다. bin 별 실패수 EMA(α 0.001, `:296-299`) + 균등 비율 0.1(`:217`), 커널 평활은 bin 이 5개라 생략. 실패 정의는 "그 env 가 RSI 후 2 s 안에 종료(base_too_low 등)".
  매 반복 bin 확률·엔트로피를 로그한다(`:233-241` 와 같은 지표).
- **커리큘럼·통계 오염 차단(필수)**
  - A0 `step_assist_curriculum`(`mdp_hang.py:2426-2446`)은 전 env 의 `n_sched`·`n_td`·`n_load` **합**으로 감쇠를 판정한다. 은행/롤인/속도 RSI env 의 착지·하중 수용이 섞이면 표준 env 가 못해도 α 가 내려간다 → **판정 집계에서 RSI env 제외**.
  - G22R 단계 커리큘럼·'자발 이륙' 집계(`:1897` 주석: 무외란 env 속도 밀기 회복이 자발 이륙 45 % 로 집계돼 커리큘럼이 막혔던 사례)에서도 같은 마스크를 쓴다.
  - "시작 2 s 제외" 통계(`:2064,2207`)는 RSI env 에서는 의미가 다르다(시작 직후가 핵심 구간).
- 위험: RSI 비율이 크면 서기 성능 망각(G21a 강점 상실). 비율과 함께 **표준 서기 무외란 낙상률·정숙 지표를 매 판정 지점에서 감시**하고, 기준선 대비 퇴보 시 탈락.

### 5. ⑤ 판정

- 판정 평가는 **RSI·롤인·속도 부여를 전부 끈 표준 서기 시작**에서만 한다. 기존 평가기(`eval_step`, 앞 밀기 평가, `eval_hang2stand`)와 같은 시드·초기조건이다. RSI env 의 학습 중 성적은 진단용이며 승격 근거로 쓰지 않는다(CLAUDE.md "산출물로 판정", 3번 "평가기 하나로 승격 금지").
- 사전 기준(학습 전 문서화): 앞 밀기 이륙률·회복률(기준선 G21a 대비 +5 %p 이상 등, 수치는 STEP 기존 기준 재사용), **무외란 서기 퇴보 없음**, 뒤 밀기 회복 퇴보 없음, 25 Hz 떨림 지표, 슬루 포화율(라운드 3).
- 대조: 같은 출발점(G21a-11300)·seed 에서 RSI 경로만 끈 같은 설정. 경로 B·C 는 서로 다른 변수이므로 각각 단독 대조를 먼저 한다.

### 6. 경로 A 채택 전 합격 검사 — 스냅샷 재현 검사 (GPU 1회 평가, 학습 없음)

- 방법: 은행 추출 롤아웃에서 상태를 저장한 env 를 골라, 새 env 에 그 상태를 RSI 로 쓴 뒤 **같은 DR 표본·같은 원시 행동 시퀀스**(은행에 저장)를 10틱 재생한다. 정책 관측 벡터(A0 정책 그룹: 프레임당 64 = 각속도 3·중력 3·명령 3·관절 12·관절속도 12·직전행동 12·시계 2·접촉 2·자이로EMA 3·관절속도EMA 12, × 이력 5 = 320차원)와 관절·루트 궤적을 원 롤아웃과 비교한다.
- 합격: 첫 틱부터 관측 차이 < 잡음 수준(잡음 항은 끈 상태로), 궤적 차이 < 1 mm·0.1°(물리 비결정성 범위). 불합격 항목은 §2 표에서 어떤 상태가 빠졌는지 바로 가리킨다.
- 이 검사는 '로그 재생에 의한 계약 검증'이며 폐루프 성능 증명이 아니다(CLAUDE.md 6번).

### 7. 실행 순위 (라운드 4)

1. **경로 B: 보조 롤인 길이형 A0 변형** — 같은 훅을 쓰고 에피소드 처음 h 반주기만 α=2, 이후 α=0, h 는 env 별 무작위·실패율 비례. 동기화 항목은 온셋(#11) 하나. A0 가 '전역 α 감쇠 절벽'에서 실패할 때의 직접 대안이다(라운드 2 §1-5·§2-4).
2. **경로 C: 온셋 뒤 몸통 앞 방향 속도 부여(0.30–0.60 m/s)** — `push_with_arm_g22` 의 방향·크기만 바꾸는 최소 변경. CP 경계 위 앞 밀림 상태를 시연 없이 만든다. B 와 결합하는 것을 전제로 한다.
3. **경로 A: 스냅샷 은행** — 재추출(§1 표) → 재현 검사(§6) 합격 → 학습. 변경 범위가 가장 크고(약 15곳), 앞 스텝 상태는 여전히 없다(뒤 스텝·미러만). B·C 가 '스윙 중간 상태 방문 부족'으로 실패했다는 진단이 있을 때만 착수한다.

공통 위험: RSI env 가 커리큘럼·통계를 오염(§4)시키는 것, 온셋 블렌드(#11), 위상 오프셋을 `episode_length_buf` 로 구현하는 것(#12) — 세 가지 모두 **구현 전에 테스트로 막아야 할 항목**이다.

## 라운드 5 (2026-10-04 05:10) — 이륙·스윙 보상의 공개 구현, 셔플링 국소해, 종료 벌점

전제(측정 사실, 코디네이터 전달): STEP-A0(α=2 고정, 600 반복) 정책을 α=2 로 평가 → load_ready 87 %·스윙발 Fz < 0.35 mg 89 % 인데 **이륙 0 %**(학습 전 G21a 는 같은 α=2 에서 31 %). 정책이 보조에 맞서 '발을 붙인 채 하중만 빼는' 법을 학습(이하 Case B).
착지 이벤트가 앞 3–10 cm 만 인정해 보조 스텝이 착지·하중·정착 보상을 못 받던 결함 → A0b(방향 무관 2–10 cm) 학습 중.

### 0. A0b 보상표로 본 Case B 의 손익 [코드·설정 확인 + 계산(가정 명시)]

A0b 설정: `logs/rsl_rl/biped12_stepb0/2026-10-04_04-04-57_stepa0b_seed1_from_g21a11300/params/env.yaml`(rewards·terminations), PPO γ = 0.99·λ = 0.95(agent.yaml). Isaac Lab 보상 = 가중 × 값 × dt(0.02).

- **발을 붙인 채 하중을 빼는 행동에 직접 값을 주는 항은 `load_transfer`(r_asym, 가중 1.0) 하나다.**
  - 정의: `mdp_hang.py:2297-2298` r_asym = (지지 Fz − 스윙 Fz)/mg, [0, 0.65] 로 자르고 전환 창(v < 0.16)에서만.
  - 이 항은 접촉 여부를 보지 않는다. 대칭 서기(0) → 스윙 Fz 0.175 mg(지지 0.825)까지 선형으로 오르다가 **포화**(0.65).
  - 측정된 '스윙 Fz < 0.35 mg 89 %·load_ready 87 %'는 이 항이 최대 또는 거의 최대인 상태와 맞는다.
- **이륙해야만 받는 항**: `swing_unload`(r_unload 0.5, 스윙발 비접촉, `:2299`), `swing_clearance`(r_clear 1.0, 이륙 높이 **1.5–4 cm 밴드**, `:2301-2302`), `swing_progress`(r_prog 0.5), 이벤트 `touchdown 5 / load_accept 10 / settle 15`(A0b 에서 방향 무관 2–10 cm 로 수정), 접촉 위상 모방 `contact_phase_reference`(r_cref 2.0, 두 발 접촉 일치 비율).
- **기울기 공백(설계 사실)**: 스윙 Fz 0.175 mg → 접촉 꺼짐(5 N 문턱 ≈ 0.03 mg) 구간에는 r_asym 이 이미 포화돼 **보상 변화가 0**이다. 꺼지는 순간에만 r_unload·r_cref 가 계단식으로 생긴다. 들어 올린 뒤에도 r_clear 는 **1.5 cm 미만이면 0**(밴드)이다. 즉 '조금 들기'에는 기울기가 없다.
- **이득 대 위험 [계산; 가정: 스텝 모드 서기 중 양(+) 항 평균 = survive 1.0 + knee_stance_flexion 4.0×≈1 + 모방 3항(로그 0.53·0.23·0.79) ≈ 7.9/틱]**
  - V(살아 있음) ≈ 7.9 × 0.02 / (1 − 0.99) ≈ **16**. 종료 벌점 −75 × 0.02 = −1.5 는 그중 일부일 뿐이고, **넘어질 때 잃는 것은 대부분 미래 양(+) 보상**이다.
  - 성공한 반주기 스텝 1회의 추가 이득(A0b, 이벤트 포함) ≈ (r_unload 4 + r_clear 5 + r_prog 2.5 + r_cref 8 + 이벤트 30) × 0.02 ≈ **1.0**.
  - → **시도 1회당 낙상 확률이 대략 5–6 %를 넘으면 '들지 않기'가 이득**이다. G21a 의 α=2 개루프 낙상 3.9 %(코디네이터 측정)는 이 손익분기 근처다.
  - 이 계산은 가중치·평균값 가정에 민감하다(미검증). **critic 의 V(s) 를 스텝 모드에서 직접 로그해 확인**하는 것이 싸다(아래 3-2).

### 1. ① 이륙·스윙 보상의 공개 구현 — 정확한 형태와 수치

| 출처 (commit) | 항 | 정확한 식 | 계수·문턱 | 파일:줄 | Case B 관점 |
|---|---|---|---|---|---|
| legged_gym (8fa29ac) | feet_air_time | 착지 첫 틱에 Σ(air_time − **0.5**), 명령 노름 ≤ 0.1 이면 0, 접촉 = Fz > 1 N, 직전 틱 OR 필터 | 가중 1.0, termination 가중 **−0.0** | `legged_gym/envs/base/legged_robot.py:882-893`; `legged_robot_config.py:131-148` | 희소(착지 때만). **0.5 s 미만 공중은 음수** → 우리 0.14–0.16 s 스텝은 −0.35 씩 벌을 받음. 셔플은 공중 0 이라 보상 0(벌도 없음) |
| legged_gym | only_positive_rewards | 틱 총보상 < 0 이면 0 으로 클립(종료 벌점은 클립 후 더함). 주석 "avoids early termination problems" | True | `legged_robot_config.py:148`; `legged_robot.py:201-203` | 종료 회피 문제에 대한 고전적 처방(③) |
| apex Cassie (82c44af) | clock 힘·속도 | 발별 tan(π/4·clock·정규화), 힘 = min(F, 250 N)/250, 속도 = min(‖v‖, 2)/2, clock 지지 +1/스윙 −1 | 힘 0.2·속도 0.2 | `cassie/rewards/clock_rewards.py:13-22,52-68,80-87` | 선형이라 **부분 하중 빼기도 부분 점수**(Case B 를 막지 못함). 속도 항이 스윙 중 발 움직임에 직접 값을 줌 |
| LearningHumanoidWalking (cd8c655) | foot_frc / foot_vel clock | 힘 정규화 = min(F, **0.5 mg**)/(0.5 mg)·2 − 1, 속도 = min(‖v‖, **0.2 m/s**)/0.2·2 − 1, tan(π/4·clock·x) | 각 0.15 | `tasks/rewards.py:107-140,143-175`(속도 정규화 `:162-171`); `tasks/stepping_task.py:110-117` | 힘: 0.5 mg 에서 포화 = 지지 쪽만 포화, 스윙 쪽은 F→0 까지 선형 기울기. **속도 항이 0.2 m/s 에서 포화** → 스윙발이 '움직이기만 하면' 바로 만점 = 셔플(미끄러지며 움직임)에도 점수 |
| walk-these-ways (0e7236b) | tracking_contacts_shaped_force / _vel | 힘: −(1 − c_des)·(1 − exp(−F²/σ_F)), 속도: −c_des·(1 − exp(−‖v‖²/σ_v)) | σ_F = **50 (N²)**, σ_v = 0.5, 가중 0.8/0.8(기본 0), 커리큘럼 문턱 0.9 | `go1_gym/envs/rewards/corl_rewards.py:67-84`; `envs/base/legged_robot_config.py:198-199,292-294`; `scripts/train.py:25-26,139` | 스윙 중 힘 벌점이 **F ≈ 7 N 이상에서 포화**(1 − e⁻¹ = 0.63 @ 7 N) → 60 N(0.35 mg)에서 10 N 으로 줄여도 벌점 변화가 거의 없음 = 우리 Case B 구간에서 기울기 없음 |
| humanoid-gym (ae46e20) | feet_contact_number | 발별 일치 +1 / 불일치 −0.3 의 평균 | 1.2 | `humanoid/envs/custom/humanoid_env.py:336-344`; `humanoid_config.py:192` | 이진 접촉 기준이라 **접촉 유지 = 불일치 −0.3** — 하중 크기와 무관하게 '발 붙이기'를 벌함 |
| humanoid-gym | feet_clearance | 스윙 마스크에서 마지막 접촉 이후 누적 발 높이가 목표 **6 cm ± 1 cm** 안이면 1 | 1.0 | `humanoid_env.py:446-468`; `humanoid_config.py:180,191` | 밴드형(우리 r_clear 와 같은 계열, 낮은 높이에 기울기 없음) |
| humanoid-gym | feet_air_time | 착지 시 min(air, 0.5), 접촉 필터에 **stance_mask OR** | 1.0 | `humanoid_env.py:320-334` | 오프셋 없음(legged_gym 과 달리 짧은 스텝도 양수) |
| unitree_rl_gym G1 (276801e) | contact, feet_swing_height | 발별 XNOR(접촉(Fz > 1 N), 위상 < 0.55) 합 / (z − **0.08**)² × 비접촉 | +0.18 / **−20** | `legged_gym/envs/g1/g1_env.py:98-109`; `g1_config.py:85-89` | XNOR 은 이진 → 접촉 유지 = 0 점(셔플 벌). 스윙 높이는 비접촉일 때만 계산 → 접촉 유지면 높이 벌점도 없음(셔플에 유리한 허점) |
| MuJoCo Playground Berkeley (ef4fefc) | feet_phase, feet_air_time, feet_slip | feet_phase = exp(−Σ(z − r_z(φ))²/0.01), r_z = 3차 베지어(최고 0.1 m); air_time = clip(air − 0.2, ≤ 0.3)(착지 시); slip = ‖v_body‖·접촉 | +1.0 / +2.0 / −0.25, termination −1.0, alive 0 | `_src/locomotion/berkeley_humanoid/joystick.py:60-86,602-646,648-665`; `_src/gait.py:32-43` | **feet_phase 는 접촉 중에도 높이 0 → r_z 오차에 연속 기울기**(1 cm 들면 바로 이득, 밴드 아님). Case B 에 가장 직접적인 연속 신호 |
| CaT (Gepetto, 5fc729b) | air_time 제약 | 착지 시 (0.25 − last_air)·명령 > 0.1 → 위반 크기 비례 **종료 확률**(max_p 0.25) | — | `exts/cat_envs/cat_envs/tasks/utils/cat/constraints.py:122-141`; `config/solo12/cat_flat_env_cfg.py:330-337` | 짧은 스텝·셔플을 확률적 종료로 막음(사족, scratch) |

### 2. ② 셔플·스커핑(발 붙인 채 하중 빼기) 국소해를 막는 기법과 부작용

| 기법 | 대표 구현 | 막는 원리 | 부작용·우리 상황 |
|---|---|---|---|
| (a) 스윙 구간 접촉 **이진** 벌점/불일치 | humanoid-gym −0.3, unitree XNOR, 우리 r_cref | 하중과 무관하게 '닿아 있음'을 벌함 | 문턱(5 N) 바로 아래로 발끝을 끄는 **토 드래그**가 '비접촉'으로 집계될 수 있음 → 발 높이·미끄럼 항과 짝이어야 함. 우리 r_cref 는 이미 있고(2.0) Case B 를 못 막음 → 크기 부족이거나 위험이 더 큼(§0) |
| (b) 스윙 구간 접촉 **힘** 벌점 | Siekmann/apex(선형), walk-these-ways(exp, σ 50) | 힘을 줄이도록 유도 | 선형은 **부분 하중 빼기에 부분 점수 = Case B 를 강화**. exp 는 큰 힘에서 포화돼 Case B 구간(60 N)에 기울기 없음. 우리 r_asym 은 (b) 의 선형형이고 Case B 와 정합 |
| (c) 이륙/공중시간 보상(착지 시) | legged_gym(−0.5 오프셋), humanoid-gym, Playground(−0.2 오프셋), CaT(제약) | 실제 이륙→착지 완주에만 값 | 희소: 첫 성공 전엔 기울기 없음. **오프셋 0.2–0.5 s 는 우리 0.15 s 스텝을 벌함**. 착지 판정 결함이 있으면 0(A0 에서 실제로 발생) |
| (d) 발 높이 **연속** 추적 | Playground feet_phase(exp, σ² 0.01) | 접촉 중에도 r_z 오차가 있어 '조금 들기'에 즉시 이득 | 클록이 actor 관측에 있어야 함(우리 `gait_clock` 있음). 명령 0 에서도 걸면 제자리걸음(Playground 는 TODO 로 방치) → 우리는 스텝 모드 게이트 필수. 너무 강하면 지지 발까지 들려 함 → 스윙발에만 |
| (e) 밴드형 발 높이 | 우리 r_clear(1.5–4 cm), humanoid-gym(6±1 cm) | 목표 높이에서만 점수 | **밴드 아래 기울기 0**(Case B 탈출에 기여 없음) |
| (f) 스윙발 미끄럼 벌점 | Isaac Lab/Berkeley feet_slide, Playground feet_slip, 우리 feet_slide_step −0.15 | 접촉 중 발 이동을 벌함 → 끌며 움직이기 억제 | 셔플이 '정지 상태로 하중만 빼기'면 미끄럼이 없어 효과 없음(우리 Case B 가 그 경우로 보임, 미측정) |
| (g) 확률적 종료로 짧은 스텝 금지 | CaT air_time | 하드 제약 | scratch 학습용. 이어 학습에서 종료를 늘리면 §0 의 위험 비대칭을 더 키움 |

### 3. ③ 종료 벌점과 탐색

- **수치로 본 우리 상황**(§0): 종료 벌점 실효 −1.5 보다 '살아 있을 때의 미래 보상' 상실(≈ 16, 가정)이 10배쯤 크다. 그래서 종료 벌점만 줄여도 위험 회피는 크게 바뀌지 않을 가능성이 높다(가설). 바꿔야 할 것은 **넘어짐 자체의 확률**(보조로 받쳐 주기) 또는 **시도의 기대 이득**이다.
- 공개 사례
  1. legged_gym: termination 가중 **0**, `only_positive_rewards=True`(음의 총보상 클립; 주석 "early termination problems", `legged_robot_config.py:132,148`). 음의 틱 보상이 많은 설계에서 '일찍 죽기'가 이득이 되는 것을 막는 쪽이다(우리 문제의 반대 방향 — 우리는 살아 있기 가치가 커서 시도를 피함).
  2. CaT: 제약 위반 → 보상 × (1 − p) 와 done = p 의 **연속(소프트) 종료**(`utils/cat/cat_env.py:99-107`), p = max_p·clip(위반/위반 이동최댓값)(`constraint_manager.py:54-75`). max_p 를 **1/20 에서 시작해 목표값으로 선형 증가하는 커리큘럼**(`curriculums.py:21-42`, `cat_flat_env_cfg.py:383-`) — 학습 초기에는 제약을 약하게(=종료를 드물게) 하는 사례.
  3. MuJoCo Playground: termination −1.0, alive 0(`joystick.py:77-78`) — 살아 있기 보상을 따로 주지 않아 V(살아 있음)를 작게 유지하는 설계.
  4. Atalante(라운드 1 §1-8, 원문 Sec. IV-C): "보상이 항상 양수라고 가정하면 종료가 수익을 잘라 버리는 효과… 음수면 에이전트가 일부러 죽는다" — 종료를 벌점 대신 **조건**으로 설계해야 한다는 논의. 그리고 큰 외란을 고정 세기로 주어 회복 스텝을 강제.
  5. HoST(라운드 2 §1-1): 넘어질 수 있는 단계에서 **당김 힘(보조)으로 낙상 자체를 줄이고** 성공 시 감쇠. Yu 2018(라운드 2 §1-2): **좌우 균형 보조 스프링 + 추진 보조**, 균형 조건(2p s 유지)을 감쇠 승격 조건으로 사용.
- **우리에게 적용**
  - 3-1. 종료 완화보다 **'넘어지지 않게 받쳐 주는' 보조**(균형 하네스)가 §0 의 손익분기를 직접 바꾼다. 스텝 모드 env 에서 골반에 약한 PD 스프링(롤·피치·수평 위치)이나 부분 체중 지지를 준다. 우리 `hang_support` 인프라(`mdp_hang.py:515-`, A0b 는 r0 = 0 으로 꺼져 있음)가 이미 있다.
  - 3-2. **진단(학습 없음)**: A0b 체크포인트로 스텝 모드 롤아웃에서 critic V(s) 평균, 반주기당 실제 스텝 이득 합, '들기 시도' 뒤 1 s 안의 낙상률을 기록한다. §0 의 손익분기 가설(V ≈ 16, 이득 ≈ 1, 손익분기 낙상률 ≈ 5–6 %)을 측정으로 바꾸는 단계다.

### 4. Case B 탈출용 단일 변경 후보 (순위)

**1위 — `load_transfer`(r_asym)를 '이륙 조건부'로 바꾸기(또는 가중 1.0 → 0)**
- 근거
  - (i) §0: Case B(발 붙인 채 하중 빼기)에 직접 값을 주는 항은 이것 하나다. 측정치(스윙 Fz < 0.35 mg 89 %·load_ready 87 %·이륙 0)가 이 항의 포화 구간과 맞는다.
  - (ii) §2(b): 선형 힘 보상은 부분 하중 빼기에 부분 점수를 줘 셔플을 강화한다는 것이 공개 구현들의 공통 성질이다. 그 구현들은 scratch 학습이라 다른 항(발판 목표 0.45, LHW)이 지배했다.
  - (iii) r_asym 은 B1 단독(보조 없음)에서는 비대칭을 만들지 못했다(측정). 즉 '탐색 유도'로는 실패했고 보조가 있을 때 '안전한 대안'으로만 작동하고 있다(해석, 대조 미실시).
- 형태: 단일 변수로 둘 중 하나만. (a) 가중 0(가장 단순), (b) 전환 창에서 쌓은 r_asym 을 **실제 이륙(스윙발 비접촉 ≥ 60 ms)이 일어난 반주기에만 지급**(지연 지급).
- 기대: '들지 않고 하중만 빼기'의 보상 우위 제거. 학습 전 G21a 의 α=2 이륙 31 % 수준으로 복귀하는지가 첫 신호.
- 위험: 하중 이동 자체가 사라질 수 있음. 판정 지표에 load_ready 와 이륙을 함께 둔다. A0b(이벤트 결함 수정)의 결과를 먼저 보고 Case B 가 남을 때 적용한다(A0b 와 섞지 않음).

**2위 — 스텝 모드 env 에 균형 하네스(낙상 방지 보조, 감쇠형) 추가**
- 근거
  - (i) §0 계산: 들기 시도의 위험(미래 보상 상실 ≈ 16)이 이득(≈ 1)보다 10배 이상 크다. 손익분기 낙상률 ≈ 5–6 % 에 G21a 개루프 3.9 % 가 걸쳐 있다(가정 포함). 진단 3-2 로 먼저 확인한다.
  - (ii) Yu 2018 의 균형 보조 스프링과 HoST 의 상태 조건부 당김 힘은 '넘어짐 위험을 보조로 없애고 성공 조건부로 감쇠'한 실제 사례다.
  - (iii) 종료 벌점 완화(CaT·legged_gym 식)는 미래 보상 상실을 없애지 못한다.
- 형태: 스텝 모드 env 에서만 골반 롤·피치에 약한 PD 토크(또는 부분 체중 지지 10–20 %)를 기존 hang_support 경로로 준다. 기울기 > 문턱일 때만 켜는 상태 조건부(HoST 식)가 의존을 덜 만든다. 감쇠는 env 별 성공 조건부(라운드 2).
- 기대: 들기 시도의 실패 비용 감소 → 이륙 탐색 재개. 위험: 정책이 하네스에 기대는 새 의존(보조 제거 문제 재발). 하네스와 α 잔차를 동시에 감쇠하면 원인 분리가 안 됨 → α 고정 상태에서 하네스만 단일 변수로.

(차순위, 참고) 발 높이 연속 추적 보상(Playground `feet_phase` 형, exp(−(z − r_z(φ))²/0.01), 스윙발·스텝 모드만)으로 r_clear 밴드의 기울기 공백을 메우는 안. 그러나 '조금 들기'의 이득이 위험보다 작다는 §0 구조는 그대로라 1·2위보다 아래로 둔다.

출처: [legged_gym](https://github.com/leggedrobotics/legged_gym) · [apex](https://github.com/osudrl/apex) · [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) ·
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) · [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · [CaT (Gepetto/constraints-as-terminations)](https://github.com/Gepetto/constraints-as-terminations) · [CaT arXiv 2403.18765](https://arxiv.org/abs/2403.18765) · Atalante arXiv 2203.01148 · HoST · Yu et al. 2018(라운드 2)

## 라운드 6 (2026-10-04 05:40) — 우리 코드 감사: 이륙을 벌하거나 양발 지지를 보상하는 구조 (읽기 전용)

전제(측정 사실): **A0c(r_asym 제거)도 α=2 이륙 0** → 라운드 5 1위 가설(r_asym 원인) 기각.
구조 진단 `reports/gen_step/diag_reward_terms_a2.0.md`(G21a-11300, A0c 과제, α=2)에서 한 발 공중 틱 − 양발 접지 틱 차이는 knee_stance_flexion −0.025/틱(최대), foot_position_reference −0.006, contact_phase_reference −0.005, premature_swing −0.002, 스텝 이벤트 합 ≤ +0.0004.
진행 중: A0e(수직 체중 지지 0.2 mg × α/2), 다음 A0f(knee_stance_flexion 스텝 모드 OFF, `hang_stand_env_cfg.py:2229-2239`).
근거 설정 파일: G21a `logs/rsl_rl/biped12_stepb0/2026-10-02_21-23-08_g21a/params/env.yaml`(지정 경로 `biped12_stage4/...` 와 같은 런 이름; stepb0 사본에서 읽음). 스텝 모드 게이트는 `hang_stand_env_cfg.py:2033-2036`(_W1_STAND_ONLY·_W1_SWAP), `:2122-2128`(StepB0 의 swap: feet_slide·foot_yaw_slip·action_rate_l2 → *_step).

### 0. 핵심 발견 (먼저)

1. **knee_stance_flexion 의 '공중 손해'는 무릎 각도보다 회복 게이트 열림 때문일 가능성이 높다** (가설, 근거는 코드 + 측정).
   - 이 항은 `state_gated`(`mdp_hang.py:400-404`: 값 × (1 − rec_open))로 감싸여 있다.
   - 회복 게이트는 **한 발이 떨어지고 기울기 > 3°**(lift_tilt)이면 열린다(`recovery_gate_step` `mdp_hang.py:312`: `(~both) & (tilt > lift_tilt)`). 게이트가 열린 틱에서는 +4 항이 통째로 0 이 된다.
   - 무릎 각도 쪽은 원인이 아니라는 측정이 있다. 성공한 뒤 스텝(`ref_steps_g22r2_s1.npz`, 489건, 보조 없음)에서 지지 무릎의 밴드 점수(G21a 밴드 0.6–1.05 rad, 각도 데이터로 계산)는 이중지지 0.999 → **공중 0.976** 으로 거의 같다.
   - 반면 diag 의 공중/접지 평균은 0.041/0.067(÷0.08 → 점수 0.52/0.84)로 크게 다르다. 차이는 각도가 아니라 '게이트 열림'(또는 α=2 무릎 잔차 과굴곡: primitive 지지 무릎 0.97–1.03 rad + 2·잔차 최대 +0.11 rad → 밴드 상한 1.05 초과 가능, `ref_step_primitive.npz` 계산)에서 온다.
   - 판별은 diag 에서 rec_open 으로 나눠 보면 된다.
   - A0f 는 어느 쪽이든 스텝 모드에서 이 항을 없애므로 유효한 시험이다. 다만 **"접지 발마다 굽힘 보상"이라는 A0f 설명(`hang_stand_env_cfg.py:2231`)은 코드와 다르다.** 실제 식은 접촉한 다리들의 **평균**(`mdp_stage4.py:242-245`)이라 발 수 자체로는 손해가 없다.
2. **보상 창·기준 위상과 실제 이륙 시각이 어긋나는 구조**(가설, 지연 추정 포함, §2-3).
   - 보조 잔차는 기준 **측정 관절각** 궤적 q_ref[k] − q_ref[0] 을 **목표**로 더한다(`mdp_hang.py:2425-2428`). 실제 관절은 지연 1–3틱 + LPF(β 0.2–0.4, 군지연 ≈ 1.5–4틱) + PD 응답만큼 늦는다. 추정 지연 L ≈ 4–7틱.
   - 그러면 기준 이륙 k = 4(스윙 창 k 4–11) 대신 실제 이륙이 k ≈ 8–11 이 된다. 공중 8틱의 절반 이상이 스윙 창 밖(r_unload·r_clear·r_prog 0, r_cref 불일치 벌)으로 밀린다. 착지는 k ≈ 16–19, **하중 수용(100 ms)은 반주기 경계(k 23)를 넘기 쉽다.**
   - 반주기 경계에서 `td_done·load_done·load_t·settle_t` 가 초기화되고(`mdp_hang.py:2281-2284`) 정착 창은 k ≥ 18.4(`_SB0_WIN` `:2235`)라서, **하중 수용·정착 이벤트가 구조적으로 거의 불가능**하다. diag 의 settle 0.00007·load_accept 0.00018 과 정합한다.
3. **diag 스크립트**: '공중 0.0000'은 버그가 아니라 α=0 에서 공중 틱이 0개(분모 max(0,1))인 결과다. 다만 '공중−접지' 열은 해석에 함정 3가지가 있다(§3).

### 1. ① G21a 활성 보상 중 접지 발 수·접촉 상태·발 높이에 의존하는 항

게이트 의미: `state_gated` = 회복 게이트 닫힘(안정)일 때만(`mdp_hang.py:400-404`), `w1_stand_gated` = |vx_cmd| < 0.03 서기 모드만(`:2094-2097`), `w1_walk_gated` = 스텝 모드만(`:2100-2103`).
회복 게이트는 '외력 중 | 기울기 > 6.5° | |ω| > 1.5 | |v| > 0.5 | (한 발 떨어짐 & 기울기 > 3°)'에서 열린다(`:312`).

| 항 (가중) | 정의 파일:줄 | 접촉·발 의존 방식 | 이륙에 미치는 부호 | 서기 모드 | 스텝 모드 A0c | A0f |
|---|---|---|---|---|---|---|
| knee_stance_flexion (+4) | `mdp_stage4.py:216-245` | 접촉한 다리 무릎이 밴드(G21a 0.6–1.05 rad) 안이면 1, 접촉 다리 **평균**, 공중이면 0.5 | 직접 손해는 없음. 단 state_gated 라 **게이트가 열리면 0**(한 발 + 3° 기울기면 열림) | ON(게이트 닫힘 때) | **ON** | 스텝 모드 OFF |
| feet_contact_gated (+2) | `mdp_hang.py:641-648` | **양발 모두 접촉**일 때만 1(× (1 − r)) | 이륙 = 손해(−0.04/틱) | ON(게이트 닫힘) | OFF(_W1_STAND_ONLY) | OFF |
| recovery_quality (+5) | `mdp_hang.py:345-371`(G21a: require_both 기본 True, step_window 0) | 게이트 **열린** 동안 정숙도 × **양발 접촉** | **회복 중 이륙 = 손해(최대 −0.1/틱)** — 앞 밀기 회복의 구조적 손해(OSS_REFERENCES 부록 2 ①, G18q 에서 '외란 창 any-contact'로 시험 → H-q 미지지) | ON | OFF | OFF |
| recovery_event (+30) | `mdp_hang.py:372-374`, 판정 `:312-325` | 게이트가 '안정 settle_s 0.3 s'로 닫히는 틱 1회. 안정 = 위 열림 조건 전부 거짓 → 한 발이고 기울기 > 3° 면 안정 아님 | 중립(착지 후 양발이면 받음) | ON | OFF | OFF |
| settle_quiet_bonus (+2) | `mdp_hang.py:1576-1597`, ramp `:943-` | `settled_ramp` 이 **양발 접촉**(`mdp_hang.py:951,954`) 요구 | 이륙 = 손해 | ON(게이트 닫힘) | OFF | OFF |
| stance_shape (−5) | `mdp_hang.py:700-713` | 발 위치: 좌우 간격 0.22–0.32 m 밖, **앞뒤 어긋남 > 4 cm** 초과분 | **스텝 뒤 엇갈린 발 = 계속 벌**(재정렬 전까지) | ON(게이트 닫힘) | OFF | OFF |
| pose_symmetry (−3), leg_pose_deviation (−1.5), march_anchor (−1) | `mdp_hang.py:692-697`; Isaac joint_deviation_l1; `mdp_stage4.py:163-183` | 좌우 관절 비대칭·기본자세 편차·**제자리 위치 유지**(명령 0 일 때) | 스텝 동작·이동 = 손해 | ON(게이트 닫힘) | OFF | OFF |
| feet_gap (−5) | `mdp_rewards.py:47-59` | 좌우 발 간격 < 0.22 m 초과분(발 위치) | 안쪽 착지 = 손해 | ON(게이트 닫힘) | **ON** | ON |
| torso_pitch_band (−25, 밴드 3°) | `mdp_hang.py:740-752` | 몸통 피치(발 무관) | 스텝 중 기울기 = 손해. 게이트 열리면 0 | ON | **ON** | ON |
| feet_contact_forces (−0.002, 150 N) | `mdp_rewards.py:41-45`, 발(.*_ankle_r_joint) | 발별 힘 > **150 N** 초과분(이력 최댓값) | **한 발 정하중 17.6 kg×g ≈ 173 N > 150 N → 단일 지지 동안 상시 벌**(작음: ≈ −0.001/틱) | ON | **ON** | ON |
| feet_impact (−0.01, 259 N) | `mdp_stage4.py:248-266` | 착지 충격 Fz > 259 N 초과분 | 착지 = 소폭 손해 | ON | **ON** | ON |
| feet_slide (−0.1) / feet_slide_step (−0.15) | Isaac Lab `velocity/mdp/rewards.py:71-` | **접촉 중** 발 속도 | 들면 오히려 이득(셔플 억제) | ON(서기) | _step 으로 ON | ON |
| foot_yaw_slip (−0.5) / _step (−0.5) | `mdp_stage4.py:186-214` | **접촉 중** 발 요 각속도 합 | 들면 이득 | ON(서기) | _step ON | ON |
| undesired_contacts (−1) | Isaac Lab `envs/mdp/rewards.py:267-` | 발 외 몸체 접촉 | 중립 | ON | ON | ON |
| survive_after_release (+1) | `mdp_hang.py:625-630` | 생존 | 넘어짐 위험 = 손해(라운드 5 §0) | ON | ON | ON |
| sb0/sbi 항 | §2 | 접촉·발 높이·위상 | §2 | 스텝 모드만 | ON(r_asym 제외) | ON |

정리
- 스텝 모드에서 '양발 접촉'을 직접 요구하는 양(+) 항(feet_contact_gated·recovery_quality·settle_quiet_bonus)은 이미 꺼져 있다.
- 남은 손해 경로는 세 가지다. (i) **state_gated 양(+) 항이 게이트 열림으로 0 이 되는 것**(A0c 에서는 knee 하나뿐 → A0f 로 제거됨). (ii) sb0/sbi 위상 어긋남(§2). (iii) 단일 지지 상시 벌점 feet_contact_forces(150 N < 정하중 173 N, 작음).
- **서기 모드(앞 밀기 회복 — 원래 병목)에서는 양발 요구 양(+) 항 3개(+2, +5, +2)와 stance_shape(−5, 엇갈림 4 cm)가 모두 살아 있다.** 스텝 모드의 해법이 나와도 서기 모드 앞 밀기 회복에는 이 구조가 그대로 남는다(G18q 기각 이력 있음 — 같은 처방을 반복하지 말고, 이번에는 '양발 요구 항 전체를 회복 게이트 열림 동안 any-contact 로'처럼 범위를 명시한 단일 변경이 필요).

### 2. ② sb0/sbi 에서 공중 틱이 접지 틱보다 불리해지는 구조

| # | 구조 | 위치 | 왜 공중이 불리해지나 | 크기(가중×dt 기준) |
|---|---|---|---|---|
| 2-1 | contact_phase_reference = 두 발 접촉 일치 비율 | `mdp_hang.py:2383`; 기준 `ref_step_primitive.npz` contact_ref: 스윙발 공중 **k 4–11 만**(23틱 중 8), 지지발 항상 접촉 | 기준 창 밖 공중(지연된 이륙 k ≥ 12, 이른 이륙 k ≤ 3, 엉뚱한 발)은 한 발 불일치 → −0.5 × 2.0 = −1.0 → **−0.02/틱** | diag 공중−접지 −0.0046 |
| 2-2 | foot_position_reference = 반주기 시작(k=0) 발 위치 대비 상대 궤적, 골반 요 프레임, σ 3 cm | `:2375-2382` | 기준 상대 궤적: 앞 +6.7 cm(0.073 → 0.140), 옆 −2.5 cm, **위 +1.8 cm 뿐**(z −0.536 → −0.518). (a) 보조 잔차는 앞뒤 관절(hip_f·ankle_f)을 0 으로 둔다(primitive 최대 2·잔차: hip_f 0°·ankle_f 0°, 나머지 6–19°) → 든 발은 앞 6.7 cm 를 못 따라가 오차가 커진다. (b) 들고 있는 동안 r_clear 는 1.5–4 cm 를 요구하는데 r_foot 기준은 +1.8 cm → 3 cm 들면 r_foot 손해(서로 상충). (c) 반주기 경계에서 기준점이 다시 잡힌다(`:2379-2380`). 늦은 스텝이 경계를 넘으면 공중의 발이 새 기준점이 된다 | diag −0.006 |
| 2-3 | **위상 창 고정 vs 실제 이륙 지연** | 창 `_SB0_WIN (0.16, 0.50, 0.80)` `:2235` → 전환 k 0–3, 스윙 k 4–11, 정착 k ≥ 19; 경계 초기화 `:2281-2284`; 보조 훅 k = ⌊v·23⌋ `:2425` | 보조가 '측정 관절각'을 목표로 넣으므로 실제는 L틱(추정 4–7) 늦다. 공중이 스윙 창 밖으로 밀려 r_unload·r_clear·r_prog 는 못 받고 2-1 벌만 받는다. 착지가 k ≈ 16–19 면 하중 수용 100 ms 가 경계를 넘어 **초기화돼 미집계**, 정착(k ≥ 19 + 하중 수용 완료)은 거의 불가능 | 이벤트 합 ≤ +0.0004(diag) |
| 2-4 | premature_swing = 예정 지지발 비접촉, 또는 전환 창(k 0–3)에서 지지 Fz < 0.6 mg 인데 스윙발 공중 | `:2322` | 지연 때문에 직전 반주기 스윙발이 아직 공중인 채 새 반주기(지지발 역할)로 넘어가면 **'예정 지지발 비접촉'으로 −2.0** → −0.04/틱 | diag −0.002 |
| 2-5 | 착지 이벤트 = 현재 반주기의 스윙발(sw)만 | `:2311` | 경계를 넘어 착지한 직전 스윙발은 sw 가 바뀌어 **착지로 집계 안 됨** | — |

판별용 측정(학습 없음, GPU 평가 1회): G21a·α=2 롤아웃에서 ① 이륙 틱 k 와 착지 틱 k 의 히스토그램, ② 하중 수용이 경계를 넘는 비율, ③ 공중 틱을 rec_open 으로 나눈 knee 항, ④ 공중 틱의 k 분포. 2-3 이 맞으면 ①의 이륙 최빈 k 가 4 가 아니라 8–11 근처에 나온다.

### 3. ③ diag 스크립트(`reports/gen_step/diag_reward_terms.py`) 확인

- '공중 0.0000'(α=0): `acc_air.sum() / max(n_air, 1)`(`:52`)에서 n_air = 0 이라 0 이 찍힌다. α=0 에서는 공중 틱이 실제로 없다(파일 2행 "공중인 env-틱 비율 0.0 %"). **버그 아님**. 표기는 'N/A' 가 맞다.
- 기여 계산: Isaac Lab `_step_reward = 가중 × 값`(dt 제외, `managers/reward_manager.py:150-157`) × dt(`:41`) → 실효 기여 맞음.
- **해석 함정(코드 버그는 아니지만 결론에 영향)**
  1. **이벤트는 정의상 접지 틱에 기록된다.** 착지·하중 수용·정착 틱은 발이 닿아 있으므로(`:42` 분류) touchdown/settle 의 '공중 0.00000'은 당연한 값이다. '공중−접지' 열이 이벤트 이득을 지운다. 스텝의 손익은 **반주기 단위 합**(유효 이륙이 있었던 반주기 vs 없던 반주기, 같은 α)으로 비교해야 한다.
  2. **위상·게이트 교란**: 공중 틱은 특정 k(보조 위상)와 게이트 열림 상태에 몰린다. 모방 항·state_gated 항의 차이는 '공중이라서'가 아니라 'k 가 달라서/게이트가 열려서'일 수 있다 → k 별·rec_open 별로 나눠 비교.
  3. 종료 틱 분류: `air` 는 step 이후(리셋된 env 는 리셋 후 상태)의 접촉으로 판정(`:42-43`)하고 alive 는 그 뒤에 갱신(`:47`) → 넘어지는 env 의 마지막 틱이 리셋 자세로 분류됨(낙상 12/256, 영향 작음).
  4. 평가 체인은 LPF 0.27·지연 1 고정(`:23`)이다. 학습 DR(β 0.2–0.4, 지연 0–3)과 다르고, 지연 L(§2-3)도 학습 분포에서는 더 넓게 퍼진다.

### 4. ④ A0f 가 실패할 경우 다음 단일 변경 후보 (순위)

**선행(학습 없음)**: §2 판별 측정 ①–④ 와 반주기 단위 손익(§3-1). 아래 1·2위 중 무엇을 먼저 할지 이것으로 정한다.

**1위 — 보조 잔차 위상 선행(lead L틱): `q_ref[k] − q_ref[0]` → `q_ref[k+L] − q_ref[L]`(L = 측정한 이륙 지연)**
- 근거: §2-3·2-4·2-5 — 보상 창·기준·이벤트는 '성공 스텝의 실제 시각'에 맞춰져 있는데, 보조는 측정 관절각을 목표로 넣어 실제가 L틱 늦다. 이 하나가 공중 틱을 창 밖으로 밀고, 하중 수용·정착 이벤트를 경계 초기화로 지운다는 설명이 diag 의 '이벤트 합 ≈ 0'과 맞는다.
  라운드 2 §0 의 "α 1 → 0.8 %, α 2 → 31 % = PD 추종 지연" 관찰과도 같은 원인이다(진폭이 아니라 위상 문제일 수 있음).
- 기대: 같은 α 에서 공중 틱이 스윙 창 안으로 들어오고 이벤트가 같은 반주기에 집계된다. 판정은 §3-1 반주기 손익과 이륙·하중 수용률.
- 위험: L 을 env 별 지연 DR(0–3틱)·β DR 에 대해 하나로 고정하면 일부 env 는 과보상. 측정 분포의 중앙값을 쓰고 지연 DR 별로 이륙 k 를 따로 본다.

**2위 — A0 계보에서 sbi 모방 3항(joint_pose_reference 1.5·foot_position_reference 2.0·contact_phase_reference 2.0) OFF**
- 근거: I0 단독 FAIL(이륙 0)로 모방 항의 탐색 효과는 이미 부정. diag 상 공중 손해 합 ≈ −0.012/틱(knee 다음으로 큼). r_foot 기준 z(+1.8 cm)와 r_clear(1.5–4 cm)가 상충하고(§2-2), 고정 위상 기준이 지연된 실제 스텝을 벌한다(§2-1).
- 기대: 공중 틱의 상대 손해 제거. 위험: 모방 항이 주던 '서기 자세 근처 양(+) 보상'(합 ≈ 0.058/틱)이 빠지면 V(살아 있음)도 같이 줄어 라운드 5 §0 의 손익분기가 바뀐다. 이것은 부수 효과이자 장점일 수 있다(해석, 미검증).

(3위, 서기 모드 원래 병목용 — 스텝 모드 해법과 별개) 서기 모드의 양발 요구 양(+) 항 3개(feet_contact_gated·recovery_quality·settle_quiet_bonus)를 **회복 게이트 열림 동안** any-contact 로 바꾸고 stance_shape 엇갈림 한도(4 cm)를 게이트 열림 + 착지 후 일정 시간 동안 해제. G18q(외란 창 한정·recovery_quality 하나)가 미지지였던 이력을 감안해, 대상 항 목록을 명시한 단일 변경으로만 시험한다.

## 라운드 7 (2026-10-04 06:10) — 보조 제거 후 행동 유지, 접촉 일정 강제의 크기, 클록 힌트의 이전

전제(측정 사실, 코디네이터 전달): 보조 아래 이륙 억제의 구조 원인 3개를 고친 A0i — (1) 보조 6틱 선행, (2) 스윙 굽힘 펄스 합성 기준(힙 +12·무릎 −25·발목 +13° 사인, 개루프 진짜 스텝 50 %), (3) 스텝 모드 knee_stance_flexion OFF.
학습 블록 착지 0.17→0.30·하중 0.12→0.26. 평가: **α=1 이륙 88 %, α=0 이륙 0**. 감쇠 문턱 0.5 미달로 사전 규칙 FAIL. 이어서 A0i-c(문턱 0.25, α 1.0→0.75→0.5→0.25→0 전역 계단) 진행 중.
스텝 모드에서 서 있어도 받는 양(+) 보상: contact_phase_ref +0.033·pose_ref +0.016·foot_ref +0.009·survive +0.020/틱.

### 0. 이번 라운드의 판단 근거 (계산, 가정 명시)

**(a) 스텝 모드에서 '서 있기'와 '스텝하기'의 보상 차가 작다.**
- contact_phase_reference 는 두 발 접촉 일치 비율 × 2.0 이다(`mdp_hang.py:2385`). 기준(`ref_step_primitive.npz`)은 지지발 23/23틱 접촉, 스윙발 15/23틱 접촉(공중 8틱)이다.
- 서 있기만 해도 일치율 = (1 + 15/23)/2 = **0.826** → 0.826 × 2.0 × 0.02 = **0.033/틱**. 코디네이터가 측정한 +0.033 과 정확히 같다.
- 완벽한 스텝은 1.0 → 0.040/틱. 따라서 **스텝을 전부 빼먹어도 손해는 0.007/틱**이다(서 있을 때 양(+) 합 ≈ 0.078/틱의 약 9 %, 반주기 22.7틱이면 0.16).
- 공개 구현과 비교(같은 방식으로 '클록은 돌고 발은 안 드는' 경우의 손해를 추적 보상 가중에 대한 비율로 계산, 표 §2):
  - humanoid-gym: 접촉 일치 +1 / 불일치 −0.3 의 비대칭이라 손해 ≈ 0.39/틱 = 선속도 추적 가중(1.2)의 **33 %**.
  - unitree G1: XNOR 이라 0.08 = 추적(1.0)의 **8 %**.
  - 우리는 서 있을 때 양(+) 합의 **9 %**이고 비대칭 불일치 벌점이 없다.
- 결론(가설): **α→0 에서 '서 있기'는 스텝과 거의 같은 보상을 받는 끌개**다. 보조가 빠지는 순간 정책이 그쪽으로 돌아가는 것은 보상 구조상 자연스럽다. A0i α=0 이륙 0 과 정합한다.

**(b) actor 는 α 를 모른다.**
- 보조 잔차는 (클록 위상, 명령) 의 결정 함수(`mdp_hang.py:2432-2435` 근처, A0i 수정 후)이고 actor 는 둘 다 관측한다(`gait_clock`, 명령). 그래서 잔차를 '스스로 내는' 표현은 가능하다(라운드 2 §0).
- 그러나 α 자체는 관측에 없어서, 정책은 '현재 α 에서 잔차를 보완하는 출력'만 학습한다. α=1 → 88 %, α=0 → 0 은 내재화 계수 κ(라운드 2 §2-4)가 0 에 가깝다는 신호다(미측정, κ 를 계산하면 확정 가능).

### 1. ① 보조를 걷어낼 때 행동을 유지한 기법 — 코드·문헌 근거

| 출처 | 보조 형태 / α 관측 | 감쇠 속도·문턱 | 감쇠 후 급락·critic 대책 | 파일:줄 / 절 |
|---|---|---|---|---|
| HoST (70bb580) | env 쪽 위 당김 힘(상태 조건부) + 행동 배율. **배율을 actor 관측에 넣음**(잡음 ±0.025) | env 별, 리셋 묶음의 평균 머리 높이 > 문턱일 때마다 −20 N(시작 200 N → 성공 10회면 0), 배율 −0.02(하한 0.25). 되돌림 없음 | 없음(단조 감쇠). 평가 때 힘 0 | `legged_gym/envs/base/host_ground.py:107-113,201,290,505-508,589-590`; `scripts/play.py:26` |
| Yu et al. 2018 (논문, 코드 미확인) | env 쪽 PD 스프링 힘. **관측 안 함** | learner-centered: 원 수익의 h=80 % 도달 → 수익 l=60 % 를 유지하는 최대 감소(line search 5방향). env-centered: 한 롤아웃 안에서 3 s 마다 −25 %, 진행 조건 = 2p s 균형 & 수익 > 완전 보조 수익의 g=70 % | **보조 수준을 롤아웃 안에서 섞어 특정 수준 과적합 방지**. ‖x‖ < 5 이면 무보조 최종 학습("매우 빨리 배운다") | Sec. 4.2.1–4.2.2, Alg. 3 (라운드 2 §1-2) |
| Kickstarting (Schmitt 2018) | 손실(교사 증류). 실행 행동에는 보조 없음 | λ_k: PBT 또는 **선형 1→0**(상수 λ 는 교사 수준에서 정체) | λ→0 후 하락 없음(교사 +43 %) | Sec. 2.1, 5.1.1 (라운드 2 §1-4) |
| DAPG (mjrl 3871d93) | 시연 전이에 상수 어드밴티지 λ0·λ1^k | λ1 = **0.95/반복**(100 반복 후 0.6 %) | — | `mjrl/algos/dapg.py:34-35,62-70` |
| RPL (Silver 2018) | 기본 제어기 + 잔차, **끝까지 제거 안 함** | — | **critic burn-in**: critic 손실 < β(1.0)까지 정책 고정 | `residual_fetch_push_env.py:26-34`; 논문 (라운드 2 §1-3) |
| JSRL | 안내 정책 롤인 길이 h | 평가 ≥ β 일 때 h 감소. Random-h 도 최종 성능 비슷 | 이점의 대부분 = 좋은 방문 상태 | Alg. 1, Sec 5.4 (라운드 1·2) |
| Berkeley Humanoid push 커리큘럼 (ffc7b26) | (보조 아님, 난이도) | **양방향**: 낙상 < 2×타임아웃이면 ×1.5, 낙상 > 타임아웃/2 이면 −0.2 | 되돌림 경로가 있음 | `exts/.../velocity/mdp/curriculums.py:76-130` |
| Isaac Lab terrain 커리큘럼 (설치본) | (난이도) | env 별 양방향 move_up/move_down | 되돌림 | `isaaclab_tasks/.../velocity/mdp/curriculums.py:49-54` |
| rsl_rl 3.1.2 Distillation (설치본) | 학생–교사 행동 손실(MSE 등) | — | PPO 와 결합은 별도 구현 필요 | `rsl_rl/algorithms/distillation.py:28,69-72,116-153` |

관찰
- **α 를 단조·전역으로 내리면서 '관측에 α 없음'인 사례**는 Yu 2018 하나뿐이다. 그마저 (i) 롤아웃 안에서 여러 수준을 섞고, (ii) 진행 조건이 '원래 수익의 60–80 % 유지'처럼 **상대값**이다.
- HoST 는 보조량을 actor 에 보여 준다(우리 계약상 불가). RPL 은 보조를 제거하지 않는다. Kickstarting·DAPG 는 보조를 실행이 아니라 손실에 둔다.
- **되돌림(α 재상승) 경로**는 HoST·A0i-c 에 없고, 난이도 커리큘럼(Berkeley push, Isaac Lab terrain)에는 있다. 단조 감쇠에서 한 계단 뒤 성능이 무너지면 회복 수단이 없다.
- 감쇠 문턱의 근거로 쓸 수 있는 수치: Yu 의 "현재 수준에서 h=80 % 도달 후, 다음 수준에서 l=60 % 유지가 가능한 최대 감소"와 env-centered "g=70 %".
  우리 문턱 0.25(절대값)는 α=1 블록 하중 0.26 의 96 % 라 '다음 계단에서 성능이 약간만 떨어져도 영구 정지'가 될 수 있다. 상대 문턱(직전 α 블록 대비 60–70 %)이 문헌과 맞다.

**우리에게 적용 (A0i-c 진행 중 바로 할 수 있는 진단 — 학습 없음)**
1. 체크포인트마다 **α 곡선**(α = 1, 0.75, 0.5, 0.25, 0 평가의 이륙·하중 수용률)을 그린다. 절벽이 어느 α 에 있는지 보이면 계단 크기를 그 구간에서만 줄일 근거가 된다.
2. **κ(내재화 계수)** = ⟨Δμ(φ), r(φ)⟩/⟨r, r⟩: 스텝 모드에서 정책 평균의 G21a 대비 변화를 잔차에 투영한다. 학습 진행에 따라 κ 가 오르지 않으면 감쇠는 소멸로 끝난다고 예측할 수 있다.
3. **감쇠 직후 value loss 스파이크**(rsl_rl 로그)를 계단마다 기록한다. RPL 식 'critic 손실이 기준선으로 돌아올 때까지 다음 계단 금지' 조건의 근거 자료가 된다.

### 2. ② 접촉 일정 강제(missed step / phase penalty) 사례와 크기

| 출처 | 항 | 강제 방식 | 크기(같은 파일의 선속도 추적 가중 대비) | 파일:줄 |
|---|---|---|---|---|
| humanoid-gym | feet_contact_number | 일치 +1 / **불일치 −0.3**(발 평균) | 1.2 vs 추적 1.2 → **100 %**. '안 들기' 손해 ≈ 33 % | `humanoid_env.py:336-344`; `humanoid_config.py:192,201` |
| unitree G1 | contact (XNOR) | 일치 1 / 불일치 0 | 0.18 vs 1.0 → 18 %. '안 들기' 손해 ≈ 8 % | `g1_env.py:98-104`; `g1_config.py:73,89` |
| walk-these-ways | tracking_contacts_shaped_force/vel | 스윙 중 힘·지지 중 속도 벌(가우스 형, σ_F 50·σ_v 0.5) | 4.0/4.0 vs 1.0 → **800 %**. 커리큘럼 진행 조건에 접촉 추적 ≥ 0.9 | `corl_rewards.py:67-84`; `scripts/train.py:24-26,142-143` |
| HugWBC (eb831af) | tracking_contacts_shaped_force/vel, feet_clearance_cmd_linear | 같은 형(σ_F 50·σ_v 5) + 스윙 중 발 높이 목표 제곱 오차 | 2 / 4 / **−30** vs 추적 2.0 → 300 % | `legged_gym/envs/h1/h1.py:1670-1697`; `h1_config.py:160-162,193,222-224` |
| LearningHumanoidWalking / apex | 발별 부호 클록(힘·속도) | 지지 +1·스윙 −1 의 선형 | 각 0.15/0.2 (발판 목표 0.45 / 속도 0.15) | 라운드 1·5 |
| CaT (Gepetto) | air_time 제약 | 착지 시 공중 < 0.25 s 면 위반 크기 비례 **종료 확률**(최대 0.25) | 확률적 종료 | `constraints.py:122-141`; `cat_flat_env_cfg.py:330-337` |
| **우리 A0i** | contact_phase_reference + premature_swing | 일치 비율(대칭, 불일치 벌 없음) + 예정 지지발 비접촉 −2 | '안 들기' 손해 = 양(+) 합의 **≈ 9 %**(§0a), 반주기당 0.16 | `mdp_hang.py:2385,2324` |

**우리에게 적용**: 공개 구현 중 일정 강제가 약한 쪽(unitree 8 %)과 비슷하다. humanoid-gym(33 %)·walk-these-ways(가우스 벌점 4.0)·HugWBC 처럼 '빼먹은 스윙'에 비대칭 벌을 주는 구성이 아니다. §4 1위의 근거다.

### 3. ③ primitive 를 W2(클록 보행)·G22R-next(외란 사건 구동)로 옮길 때 클록 힌트 처리 사례

- **HugWBC — 서기 env 는 클록을 지지 위상에 고정**
  - `h1.py:776-778`: standing_envs_mask 인 env 는 foot_indices = 0 → 원하는 접촉 = 양발 지지. 서기 env 비율 1/10(`:631-637`).
  - 서기 보상: 양발 접촉 +1(`standing`, `:1519-1523`), 양발 공중 −1(`standing_air`, `:1525-1529`).
  - 보행 주파수 1.5–3.5 Hz 를 명령으로 관측(`h1_config.py:22,138`).
  - → **클록을 없애지 않고 '서기 = 고정 위상'으로 정의**한다. W2 에서 쓸 수 있는 가장 직접적인 패턴이다.
  - 우리에게: 우리 `gait_clock`(`mdp_stage4.py:25-42`)은 서기에서도 1.1 Hz 로 돈다(G21a 는 이를 무시하도록 학습됨). 서기 모드에서 위상을 고정하면 **관측 값의 의미가 바뀌는 계약 변경**(차원 불변)이므로 러너 수정·사용자 승인이 필요하다.
- **LearningHumanoidWalking**: STANDING 모드(5 %)에서 클록 함수를 상수(지지 +1, 속도 −1)로 바꿈(`tasks/stepping_task.py:105-108`). 관측 클록 처리는 이번에 확인 안 함(미검증).
- **van Marum 2024(Digit)**: 클록을 쓰지 않는다. 원문: "the nontrivial question of what to do with the required clock inputs during standing and transitions remains a challenge", "the clock framework incentivizes low foot velocities in standing mode, which directly impedes disturbance rejection".
  대신 서기 = 접촉 항 상수 1, 보행 = 한 발 접촉 틱마다 +1(0.2 s 유예). → **G22R-next(외란 사건 구동)는 클록 없는 설계가 문헌상 자연스럽다.**
- **Isaac Lab G1·Berkeley Humanoid**: 클록 없음, 공중시간 보상(명령 게이트). MuJoCo Playground: 클록 상시(명령 0 처리 TODO).
- **클록 힌트를 '감쇠'시켜 제거한 공개 코드는 이번 조사에서 찾지 못했다.** walk-these-ways 는 반대로 접촉 추적 ≥ 0.9 를 명령 범위 확장의 조건으로 쓴다(일정을 유지한 채 난이도만 올림).
- **우리에게 적용**
  - W2: 클록 유지 + 스텝 모드 일정 강제 강화(§2) + 속도 명령 커리큘럼(Berkeley 식, 추적 > 80 % 일 때 확장)이 사례와 맞는다. 서기에서 클록 고정은 계약 변경이라 별도 승인 항목이다.
  - G22R-next: 외란은 클록과 무관한 시각에 오므로 현재 위상 고정 primitive(보조·창 모두 에피소드 시간 기반)는 그대로 옮길 수 없다.
    대안은 (i) van Marum 식 클록 없는 한 발 접촉 보상을 외란 창(외생 신호)에서만, (ii) 외란 시작 시 보조 primitive 를 '사건 시각 기준'으로 재생(보조 타이머는 env 내부, 관측 안 함 — 실물 밀기도 관측되지 않으므로 계약과 정합)이다.
    (ii) 는 A0i 의 펄스 기준을 그대로 재사용할 수 있다(가설).

### 4. A0i-c 가 α 감쇠 중 소멸할 경우 단일 변경 후보 (순위)

**1위 — 스텝 모드 '빼먹은 스텝' 벌점(반주기 단위 사건)**
- 형태: 스텝 모드에서 예정 반주기가 끝날 때 그 반주기에 유효 이륙(≥ 60 ms)이 없었으면 1회 벌점.
- 크기: 지금 '안 들기' 손해(반주기 0.16)를 humanoid-gym 수준(서 있을 때 양(+) 합의 ≈ 30 %, 반주기 ≈ 0.5)까지 올리는 −0.3 ~ −0.5/반주기(가중 15–25 × dt).
  상한 조건: 서 있을 때 반주기 양(+) 합(≈ 0.078 × 22.7 ≈ 1.8)보다 충분히 작게 → '넘어지는 게 낫다'(legged_gym `only_positive_rewards` 가 막으려던 문제)를 만들지 않도록.
- 근거: §0a(서 있기 = 스텝 대비 9 % 손해뿐이라 α→0 의 끌개), §2(공개 구현의 비대칭 불일치 벌·가우스 일정 벌·CaT 제약), A0i α=0 이륙 0.
- 위험: 아직 스스로 못 드는 env 에도 벌이 쌓여 V 가 낮아진다 → 위험 회피가 줄어드는 쪽(의도)일 수도, 무리한 이륙·낙상 증가일 수도 있다. 낙상률·종료 비율을 같이 판정한다. A0i-c 가 소멸한 지점(α)에서 그 α 를 고정하고 이 벌점만 추가하는 것이 단일 변수다.

**2위 — 전역 계단 대신 env 별 α 표본(보조 DR): α ~ U[0, α_max], α_max 를 성공 조건부로 하향, α=0 env 20 % 상시 감시**
- 근거: Yu 2018 env-centered(롤아웃 안 여러 수준 → 과적합 방지), JSRL-Random, §0b(α 비관측 → 전역 계단은 매번 동역학 충격). Berkeley·Isaac Lab 의 양방향 커리큘럼처럼 α_max 되돌림 경로도 둔다.
- 위험: 1위 없이 쓰면 α≈0 env 는 여전히 서기 끌개라 내재화 기울기가 약하다(§0a). 그래서 2위로 둔다.

**3위 — 보조를 손실로 옮기기(Kickstarting 형): 감쇠 시작 시점 정책의 고정 사본 + α·r(φ) 를 교사로 하는 λ_k MSE, λ 선형 감쇠** — PPO 손실 변경(승인 필요). rsl_rl 3.1.2 `distillation.py` 의 손실 형태를 참고해 SmoothPPO(G18j)처럼 서브클래스로 구현.
- (대안, 계약 변경) RPL 처럼 스텝 모드 피드포워드 primitive 를 **배포 체인에 영구 포함**: 러너가 클록·명령으로 같은 잔차를 더한다. 제거 문제는 사라지지만 액션 처리 계약이 바뀐다. 사용자 결정 사항이며 실물 이식 위험(실물 지연이 시뮬과 달라 6틱 선행이 안 맞을 수 있음)을 함께 평가해야 한다.

출처: [HoST](https://github.com/InternRobotics/HoST) · [HugWBC](https://github.com/InternRobotics/HugWBC) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) ·
[isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [CaT](https://github.com/Gepetto/constraints-as-terminations) · [mjrl/DAPG](https://github.com/aravindr93/mjrl) · [residual-policy-learning](https://github.com/k-r-allen/residual-policy-learning) · [van Marum 2024 arXiv 2404.19173](https://arxiv.org/abs/2404.19173) · Yu et al. 2018 arXiv 1801.08093 · Kickstarting arXiv 1803.03835 · JSRL arXiv 2204.02372
