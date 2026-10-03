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
