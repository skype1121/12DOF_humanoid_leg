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

## 라운드 8 (2026-10-04 06:35) — 비대칭 critic 의 현재 상태와 α 추가 시 체크포인트 호환, env 별 α 섞기의 실패 원인 (읽기 전용)

전제(측정 사실): A0i-c(전역 계단) α 1.0→0.75 첫 감쇠 직후 블록 하중 수용 0.25→0.11, 190 반복 동안 회복 없음(라운드 7 예측과 같음). 12500 에서 중단 예정. 다음은 A0k(env 별 α ~ U[0, α_max]·감시 env 20 % α=0·되돌림 상대 감쇠, A0i-11900 출발).
조사 대상(읽기만): rsl_rl 3.1.2 설치본 `/home/ryu/isaacsim/isaac-sim-5.1.0/kit/python/lib/python3.11/site-packages/rsl_rl`, `rl_walking/agents.py`, `rl_walking/stage4_env_cfg.py`, `rl_walking/symmetry.py`, `rl_walking/scripts/train.py`, A0i 체크포인트 `logs/rsl_rl/biped12_stepb0/2026-10-04_05-06-37_stepa0i_seed1_from_g21a11300/model_11900.pt`(CPU 로드로 형상만 확인), A0i-c `params/agent.yaml`·`env.yaml`.

### 1. ① 비대칭 actor-critic — 지금 켜져 있나, α 를 넣으면 무엇이 깨지나

**현재 상태 [코드·체크포인트 확인]: 이미 켜져 있다(최소형).**
- `rl_walking/agents.py:17-18` `obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}`, `:24-25` actor/critic 관측 정규화 **꺼짐**, `:26-27` 은닉층 [512, 256, 128] 동일. A0i-c `params/agent.yaml:6-10,28-29` 도 같다.
- critic 그룹 내용은 `rl_walking/stage4_env_cfg.py:114-127` `CriticCfg` 의 **base_lin_vel 3 × 이력 5 = 15** 하나뿐이다(A0i-c env.yaml 확인). 그 외 특권 정보(접촉력·외력·DR 표본·보조 상태)는 없다.
- 체크포인트 형상(A0i-11900): `actor.0.weight (512, 320)`, `critic.0.weight (512, 335)` = 정책 320 + critic 15. `log_std (12,)`. 최적화기 상태 17개.
  `rl_walking/symmetry.py:21-22,53-54` 주석·상수는 아직 245/260(구 레이아웃)으로 적혀 있는데, `_POLICY_KIND_BY_DIM` 이 차원으로 레이아웃을 고르므로(`:153-156`) 동작은 한다. critic 쪽은 `CRITIC_GROUP_DIM = 15` 로 고정이다.
- critic 입력 조립: rsl_rl `modules/actor_critic.py:47-50`(그룹 차원 합으로 critic 입력 크기 결정), `:162-173`(obs_groups 순서대로 concat). 따라서 critic 그룹 **끝에 항을 추가하면 새 열은 335 다음(끝)에 붙는다**.

**critic 그룹에 α(·보조 상태)를 추가하면 깨지는 곳 3개** [코드 확인]
1. **체크포인트 로드**: `rl_walking/scripts/train.py:226-229` → `runner.load(resume_path)` → rsl_rl `runners/on_policy_runner.py:309-312` `policy.load_state_dict(...)` → `modules/actor_critic.py:186-199`(strict=True).
   `critic.0.weight` (512, 335) vs 새 모델 (512, 335 + k) 는 **형상 불일치로 RuntimeError**(PyTorch 는 strict=False 라도 형상 불일치는 오류).
   이어서 `:316-319` 최적화기 로드도 Adam 상태(exp_avg/exp_avg_sq)의 형상이 맞지 않아 실패한다. actor 는 그대로라 actor 계약은 영향 없음.
2. **미러 손실**: A0 계보는 `use_mirror_loss=True`(`agents.py:70-75`). rsl_rl `algorithms/ppo.py:316-321` 이 `data_augmentation_func(obs=obs_batch)` 를 **TensorDict 전체**(policy·critic 키 모두)에 부른다.
   우리 `mirror_biped12` 는 키마다 `_mirror` 를 적용하고(`symmetry.py:189-193`), 차원이 `CRITIC_GROUP_DIM`(15)과 다르면 **ValueError**(`:157-162`)를 낸다. 새 그룹 키를 따로 만들면 `_obs_kind` 가 **ValueError**(`:166-171`)를 낸다.
   → `_CRITIC_TERMS`(`:83`)·`CRITIC_GROUP_DIM` 갱신이 필수다. α 는 좌우 미러 불변(부호 +1)이고, 발별 보조 상태를 넣으면 좌우 교환 규칙도 추가해야 한다.
3. **학습률 복원**: `train.py:230-232` `--restore_lr` 는 `optimizer.param_groups[0]["lr"]` 를 읽는다. 최적화기를 새로 만들면 적응 lr 이 cfg 초기값으로 돌아간다(이어 학습 거동 변화).

**우회 방법 — '체크포인트 수술' (오프라인, 구현 안 함; 설계만)**
- (a) `critic.0.weight` 에 **새 입력 k 열을 0 으로 붙임**: W' = [W (512×335) | 0 (512×k)].
  - 첫 출력은 기존 critic 과 **완전히 같다**(새 입력 무시 상태에서 출발). 기울기는 즉시 새 열로 흐른다.
  - critic 관측 정규화가 꺼져 있어(`agents.py:25`) 정규화기 통계 확장은 필요 없다.
- (b) 최적화기: Adam 상태에서 형상 (512, 335) 인 항목(유일 — actor.0 은 (512, 320))의 exp_avg·exp_avg_sq 에 0 열 k 개를 붙인다. lr(`param_groups[0]["lr"]`)·step 은 보존.
  대안으로 `runner.load(path, load_optimizer=False)`(`on_policy_runner.py:309`)도 가능하지만 actor 쪽 Adam 모멘트까지 초기화돼 이어 학습 거동이 바뀐다 → (b) 권장.
- (c) 수술 직후 검증(학습 없음): 같은 관측 배치에서 수술 전후 critic 출력 차 = 0, actor 출력 차 = 0, `mirror_biped12` 왕복(미러 2회 = 원본) 단위 테스트(`rl_walking/tests_symmetry.py` 확장).
- 승인 필요 여부: actor 관측 계약은 불변이지만 **critic 입력(네트워크) 변경**이다. 사용자 사전 지시의 동결 목록(네트워크·PPO 하이퍼파라미터)에 걸리므로 **사전 승인 대상**이다.

### 2. ② A0k(env 별 α 섞기)에서 감시 env(α=0)의 하중 수용이 0 에 머무를 때 — 원인 후보와 대처

| # | 원인 후보 | 근거 | 확인 방법(학습 중 로그, 추가 실행 없음) | 대처 |
|---|---|---|---|---|
| 1 | **α=0 에서 '서 있기'가 스텝과 거의 같은 보상** | 라운드 7 §0a: 안 들기 손해 ≈ 양(+) 합의 9 %(반주기 0.16) | α 구간별 반주기 반환(서기 반주기 vs 이륙 반주기) | 놓친 스텝 벌점(A0j) 결합 — §3 1위 |
| 2 | **α 를 모르는 정책 = 섞인 α 에 맞춘 평균 행동** | actor 는 α 비관측(라운드 7 §0b). U[0, α_max] 에서 학습 기울기 대부분은 보조가 있어 스텝이 실제로 나는 env 에서 오고, 그 기울기는 '잔차를 보완하는 출력'을 강화 | κ(라운드 2 §2-4)를 α 구간별로 계산. κ 가 α_max 를 따라 내려가지 않고 0 근처면 내재화 없음 | α 분포를 낮은 쪽에 몰기(예 α ~ α_max·Beta(1, 2)), 감시 env 비율을 학습에는 넣지 않는 '평가 전용'으로 분리(아래 #5) |
| 3 | **critic 이 α 를 몰라 생기는 가치 오차** | critic 입력에 α 없음(§1). 같은 관측에서 α 에 따라 수익이 크게 다르다(A0i: α=1 이륙 88 % vs α=0 0 %) → V 는 α 평균 → α=0 env 는 V 과대(음의 어드밴티지), 보조 env 는 V 과소(양의 어드밴티지) | α 구간별 (V − 실현 반환) 평균. 0 에서 멀면 가치 오차 확정 | 비대칭 critic 에 α 추가(§1 수술, 승인 필요) |
| 4 | **GAE 부트스트랩 편향** | λ = 0.95·γ = 0.99(A0i-c agent.yaml)라 #3 의 V 오차가 어드밴티지에 그대로 섞인다. 상태만의 함수인 기준선은 정책 기울기의 기댓값을 바꾸지 않지만, **부트스트랩에 쓰이는 V 의 오차는 편향**을 만든다. α 가 숨은 상태이므로 '관측만의 V' 는 진짜 가치가 아니다 | #3 과 같음 | #3 과 같음 |
| 5 | **감시 env 가 학습 표본에도 들어가 서기 쪽으로 당김** | 감시 env 20 % α=0 은 거의 서기만 하므로(#1) 그 표본은 '서기 강화' 기울기를 준다 | 감시 env 의 행동 log-prob 변화 | 감시 env 를 별도 평가 프로세스로(학습 배치 제외) — 단 rsl_rl 은 env 별 손실 마스크가 없어 구현 비용이 있음(미검증). 또는 감시 비율을 5–10 % 로 |
| 6 | **되돌림 상대 감쇠가 α_max 를 높게 묶음** | 감시 env 가 0 이면 상대 조건이 '보조 env 성능'만 보게 돼 α_max 가 내려가도 감시 env 는 변화 없음 | α_max 궤적 vs 감시 env 하중 수용 | 감쇠 조건에 'α 하위 구간(예 α < 0.25) env 의 하중 수용'을 포함 |

추가 관찰: 보조는 지연 6틱 선행·펄스 기준이 '보조가 있을 때'만 정합된다. α 가 작아지면 정책이 내야 할 '선행된 굽힘 펄스'를 스스로 만들어야 하는데, 이 펄스는 슬루 4°/틱 예산의 큰 몫을 쓴다(라운드 3 §0a: 성공 스텝이 이미 슬루 한계 근처). 정책 잡음·보정 출력과 예산을 나눠 쓰므로 α=0 에서 펄스 진폭이 부족할 수 있다(가설, 슬루 포화율 로그로 확인).

### 3. ③ A0k 실패 시 다음 단일 변경 순위 (승인 필요 여부 포함)

| 순위 | 변경 | 근거 | 동결 지시와의 관계 | 위험 |
|---|---|---|---|---|
| **1** | **A0k + 놓친 스텝 벌점(A0j)** — 반주기 끝에 유효 이륙 없으면 1회 벌점(라운드 7 §4 1위 크기: 반주기 −0.3 ~ −0.5, 서기 반주기 양(+) 합 ≈ 1.8 보다 충분히 작게) | 원인 #1(서기 ≈ 스텝 보상)을 직접 제거. 공개 구현 비교: humanoid-gym 비대칭 불일치 벌(안 들기 손해 33 %) vs 우리 9 % | **보상 변경 — 기존 STEP 실험과 같은 범주, 별도 승인 불요**(보상 설계는 이번 연구 범위에서 계속 바꿔 왔음). PPO·네트워크 불변 | 아직 못 드는 env 에도 벌이 쌓여 무리한 이륙·낙상 증가 가능 → 낙상률·종료 비율을 판정에 포함. 보상 합이 음이 되면 '일찍 죽기' 유인(legged_gym `only_positive_rewards` 문제) → 크기 상한 준수 |
| 2 | **비대칭 critic 에 α(·보조 잔차 크기) 추가** — §1 체크포인트 수술(0 열 확장 + Adam 상태 확장) + `symmetry.py` critic 규칙 갱신 | 원인 #3·#4(α 숨은 상태 → 가치 오차·GAE 편향). HoST 는 보조량을 **actor** 에까지 넣는다(`host_ground.py:290`) — 우리는 계약상 critic 까지만 | **네트워크(critic 입력) 변경 → 사전 승인 필요**. actor 계약 불변 | critic 만 α 를 알면 '보조 덕의 성공'을 가치로 설명해 버려 actor 의 내재화 압력이 오히려 줄 수 있음(어드밴티지에서 α 효과가 빠짐 — 장점이자 위험, 미검증). 수술 오류 위험 → §1(c) 검증 필수 |
| 3 | **Kickstarting 형 손실 보조** — 감쇠 시작 시점 정책의 고정 사본 + α·r(φ) 를 교사로 한 λ_k MSE, λ 선형 감쇠 | 보조를 actor 안으로 직접 넣는 유일한 경로(Kickstarting Sec 5.1.1, DAPG 0.95/반복). rsl_rl `algorithms/distillation.py:28,116-153` 의 행동 손실 형태 참고, SmoothPPO(`rl_walking/smooth_ppo.py`) 처럼 서브클래스 | **PPO 손실 변경 → 사전 승인 필요** | 교사가 개루프(위상 고정)라 착지·하중 수용의 폐루프 보정과 충돌 가능 → 손실을 이륙 전·스윙 구간 관절에만 |

순서 근거: 1위는 승인 없이 가능하고, 원인 #1 은 라운드 7 계산으로 이미 수치화돼 있다. 2·3위는 둘 다 동결 대상 변경이므로, A0k + A0j 결과(감시 env 하중 수용이 움직이는지)를 보고 사용자에게 함께 올리는 것이 효율적이다.
A0k 학습 중에 §2 의 α 구간별 로그(반환·κ·V 오차)를 남기면 2위(가치 오차)와 3위(내재화 부족) 중 어느 쪽이 병목인지 승인 요청 전에 근거로 쓸 수 있다.

## 라운드 9 (2026-10-04 07:00) — primitive 를 외란 사건 구동 회복과 W2 보행으로 옮기기 (설계 조사)

전제(측정 사실, 코디네이터 전달): **STEP-A0m**(A0i-11900 출발, 전역 α 계단 1.0→0→, 문턱 블록 하중 수용 0.25, 놓친 스텝 벌점 −0.4/반주기)이 α=0 에 도달했다.
13100 체크포인트 α=0 결정론 평가(eval_step, 스텝 모드 vx 0.05 = 모드 신호, 1.1 Hz 클록 반주기 일정): **이륙 80.5 %·착지 72.8 %·하중 수용 72.8 %·스텝 그룹 낙상 2.3 %·서기 무요청 이륙 0 %**. 최종 평가(독립 seed·서기 흔들림·밀기 회귀·보행 지표) 진행 중.
제약: actor 관측 계약 불변(클록 `gait_clock`·명령 `generated_commands` 채널 재사용만 가능).

### 0. 계산 (가정 명시)

**(a) 고정 클록(1.1 Hz, 반주기 T_h = 0.4545 s)에서 속도별 보폭과 스윙발 이동량**
- 정상 보행에서 반주기마다 한 걸음 → 걸음 길이 L = v·T_h. 스윙발은 '뒤 L → 앞 L' 로 움직이므로 스윙 1회 이동량 = 2L = v·T_cycle.

| v (m/s) | 걸음 L (cm) | 스윙발 이동 2L (cm) | 공중 0.16 s 기준 스윙발 평균 속도 (m/s) | 힙 피치 변화(다리 길이 0.5 m 가정) | 공중 8틱 평균 힙 속도 (°/틱) |
|---|---|---|---|---|---|
| 0.05 | 2.3 | 4.5 | 0.28 | 5.2° | 0.65 |
| 0.10 | 4.5 | 9.1 | 0.57 | 10.4° | 1.3 |
| 0.15 | 6.8 | 13.6 | 0.85 | 15.6° | 1.95 |
| 0.25 | 11.4 | 22.7 | 1.42 | 26° | 3.25 |

- 코디네이터 계산(2–11 cm)과 같다. 단 **0.25 m/s 는 스윙발 이동 22.7 cm** 로, 뒤 회복 스텝 실측 수평 변위(중앙값 18.8 cm, 분포 상단 25.5 cm — `ref_steps_g22r2_s1.txt`)의 상단이다.
- 힙 평균 속도 3.25°/틱에 primitive 자체 힙 펄스(+12° 사인)가 더해지면 **슬루 4°/틱을 넘는 구간이 생긴다**(계산; 라운드 3 §0a: 성공 스텝은 이미 가장 빠른 관절에서 3.8°/틱).
- → W2 1단계 상한은 **0.15 m/s**(평균 힙 2°/틱)가 안전하다. 0.25 m/s 로 가려면 공중 시간을 늘려야 한다(0.25 s 면 2.1°/틱).
  **공중·스윙 비율은 actor 관측에 없다**(클록은 sin/cos 위상뿐, `mdp_stage4.py:36-42`). 그래서 스윙 창(`_SB0_WIN`)과 primitive 길이를 보상·기준 쪽에서 바꾸는 것은 계약 변경이 아니다.

**(b) W2 착지 목표의 속도 비례 계수**
- 피드포워드(중립점): x_land − x_pelvis(착지 시) = v·T_h/2 = **0.227·v** (0.05 → 1.1 cm, 0.15 → 3.4 cm, 0.25 → 5.7 cm).
  walk-these-ways 의 Raibert 항이 정확히 이 형태다: `go1_gym/envs/rewards/corl_rewards.py:184-191` desired_xs_offset = phase·v_des·(0.5/f), phase ∈ [−0.5, 0.5] → 착지(phase 0.5)에서 v/(4f) = v·T_h/2. 이 코드에는 속도 오차 피드백 항이 없다.
- 피드백(속도 오차): LIPM 캡처 포인트 계수 k = 1/ω = √(h_c/g) ≈ **0.23–0.25 s**(h_c 0.50–0.60 m 가정, 라운드 3 §0b). → x_land = 0.227·v_cmd + 0.24·(v − v_cmd).
- 밀기 회복(정지 명령)에서는 라운드 3 §0b 공식 x_land ≈ 8.5 + (ξ₀ − 8.5)·e^{ω·T} − 1 cm(발목 원점, T ≈ 0.21 s)를 그대로 쓴다.

### 1. ① 클록 일정 primitive → 외란 사건 구동 회복 스텝

**공개 사례 상황**
- 'RL 정책의 클록 위상을 외란 사건에서 리셋'하는 공개 코드는 찾지 못했다(검색: phase reset / time-since-disturbance / event-triggered, 결과는 고전 제어·Atalante·Ferigo 류뿐).
- 고전 CPG 보행의 위상 리셋(phase resetting)은 오래된 기법이지만 RL 클록 관측과의 결합 사례는 확인 못 함(미검증).
- 클록 없이 사건 기반으로 회복 스텝을 낸 RL 은 Atalante(라운드 1 §1-8: 큰 고정 외란 + 보행 상태 RSI), Ferigo 2021, van Marum 2024(서기 = 접촉 항 상수, 클록 없음)이다.
- 서기 중 클록을 지지 위상으로 고정한 사례는 HugWBC(`legged_gym/envs/h1/h1.py:776-778`, 라운드 7 §3)다.

**우리 구조의 사실**
- 클록은 배포 러너가 '정책 시작 시각 기준'으로 계산한다(`rl_walking/deploy/policy_runner_stage4.py:138-145`, 학습 `mdp_stage4.py:25-33` 과 같은 식).
- 스텝 모드는 명령 vx ≥ 0.03 으로 켜진다(`mdp_hang.py` `w1_walk_gated` `:2100-2103`, sb0 mode). 명령은 브리지/콘솔이 정한다.
- 즉 **'스텝 모드 힌트'와 '위상'은 둘 다 배포 쪽 소프트웨어가 소유**한다. 사건 구동으로 바꾸려면 그 소프트웨어가 사건을 감지해야 한다.

**설계 대안 비교**

| 안 | 내용 | 배포 시 누가 켜나 | 계약 영향 | 위험 |
|---|---|---|---|---|
| A. env 가 특권 정보(외력 스케줄 `dst_F_now` 등)로 명령을 자동 켬 | 학습만 쉬움 | **배포에서 재현 불가**(실물엔 외력 정보 없음) | — | 시뮬 전용 행동 → **기각** |
| B. **관측 가능 신호 감지기(러너·env 공용 함수)** 가 밀림을 감지하면 (i) vx 힌트를 한 반주기 켜고 (ii) 클록 위상을 '전환 창 시작 + CP 쪽 스윙발'로 리셋 | 러너(브리지)가 같은 함수로 켬. 입력은 actor 가 보는 신호(지연 자이로·중력·접촉 비트)만 | 관측 차원·형식 불변. **클록 위상 의미(시작 시각 기준 → 사건 기준)와 명령 출처가 바뀌므로 배포 계약 변경 = 사전 승인 필요** | 오탐(조용한 서기에서 자발 스텝 — 현재 0 % 를 지켜야 함), 감지 지연(IMU 지연 0–3틱 + 판정), 연속 재트리거, 발 선택 오류, 클록 점프에 따른 5틱 이력 불연속(학습에서도 같은 점프를 겪게 하면 정합) |
| C. 클록 채널을 'time-since-event' 로 재해석 | B 의 위상 리셋이 사실상 이것(리셋 후 φ = 2π·f·(t − t_event) + φ0) | B 와 같음 | B 와 같음 | B 와 같음. 사건 없는 서기 중 클록 처리(계속 돌림 vs 고정)를 함께 정해야 함 |
| D. 힌트 없이 서기 모드에서 스스로 스텝(원래 목표) | B 가 된 뒤 감지기 힌트를 env 비율로 감쇠(라운드 7·8 의 α 감쇠 방식) | 정책 | 계약 불변 | 서기 모드의 양발 요구 양(+) 항·stance_shape(라운드 6 §1)이 그대로면 감쇠 중 소멸 가능성 높음 |

**권장 감지기 형태(설계, 미구현)**
- 입력: actor 와 같은 지연 IMU(피치·롤, 피치율), 접촉 비트.
- 추정 캡처 포인트(몸통 기준 앞쪽) ξ̂ ≈ h_c·(θ + θ̇/ω). θ 는 정착 서기 기준 피치 편차, h_c 0.55 m·ω 4.2 s⁻¹(가정).
- 트리거: ξ̂ > 발끝 여유(예 6–8 cm)가 2틱 연속, 그리고 직전 트리거 후 ≥ 1 반주기.
- 스윙발: 롤 방향으로 하중이 적은 쪽(접촉 비트·롤 부호).
- 오탐 검증 자료: 실물 정지 노이즈(자이로 std 0.006 rad/s — 메모리 기록)와 시뮬 서기 흔들림 평가 로그로 '무외란 트리거율'을 먼저 측정(학습 없음).
- **env 와 러너가 같은 파이썬 함수를 import**(현재 `policy_runner_stage4.py` 가 `mdp_stage4` 공식을 공유하는 방식과 같게)해서 시뮬–실물 불일치를 막는다.

**CP 착지 목표를 기존 primitive 에 얹는 법**
- 보조가 끝났으므로(α=0) 착지 목표는 **보상 쪽**에만 둔다. sbi `foot_position_reference` 의 앞 성분 기준(`mdp_hang.py:2375-2382`, 반주기 시작 대비 상대 궤적)을 고정 +6.7 cm 대신 사건 시점 ξ̂ 로 계산한 x_land(§0b)로 바꾼다.
  또는 라운드 3 §1-1(c) 의 'CP–지지다각형 부호거리' 보상을 회복 창(사건 후 1 반주기)에만 둔다.
- primitive 굽힘 펄스(힙 +12·무릎 −25·발목 +13°)는 들기용이고 앞 배치는 정책 몫이다. 착지 목표가 크면(> 15 cm) 슬루 한계(§0a)에 걸릴 수 있어 x_land 를 [3, 15] cm 로 자르는 편이 안전하다(계산 근거).

### 2. ② W2 보행 — 스텝 모드에 속도 추종을 더할 때

- **착지 목표**: §0b 의 x_land = 0.227·v_cmd + 0.24·(v − v_cmd)를 r_foot 앞 성분 목표로 쓴다(walk-these-ways Raibert 보상 `corl_rewards.py:161-200` 형태, 우리는 가우스 σ 3 cm 형 유지). 피드백 계수 0.24 s 는 LIPM 값이고 실측 보정이 필요하다(미검증).
- **명령 커리큘럼**: Berkeley Humanoid `exts/.../velocity/mdp/curriculums.py:133-170`(추적 보상 > 최대의 80 % 일 때만 범위 +0.5 m/s; 우리는 +0.05 m/s 단위). 시작 0.05(현재 모드 신호) → 0.10 → 0.15 에서 일단 멈춤(§0a 슬루 근거).
- **보폭 충분성**: 0.05–0.25 m/s 에 필요한 걸음 2.3–11.4 cm 는 뒤 회복 스텝(18.8 cm)보다 작아 기구학적으로 가능하다. 그러나 **고정 공중 0.16 s 에서는 0.25 m/s 의 스윙발 이동 22.7 cm 가 슬루 한계에 걸린다**(§0a). 0.15 m/s 까지는 여유가 있다. 그 이상은 스윙 창·primitive 공중 시간을 0.25 s 로 늘리는 보상/기준 변경(계약 불변)이 먼저다.
- **위험**: 속도 추종 항이 지금 스텝 모드 보상(놓친 스텝 벌점 −0.4, 모방 3항)과 경쟁해 다시 '서서 미끄러지기(feet_slide)'로 속도를 내는 국소해가 생길 수 있다. 판정에 착지 변위·feet_slide·놓친 스텝 비율을 같이 둔다.

### 3. ③ 다음 단일 실험 순위

**회복 1개 — R1: 관측 신호 감지기 힌트(B안) + 위상 리셋, 보상은 A0m 스텝 모드 그대로**
- 형태: A0m-13100 출발. 서기 모드 env 에 CP 경계 위 앞 밀기(몸통 앞 0.35–0.6 m/s, 온셋 뒤 — 라운드 4 §3 경로 C)를 준다. 감지기가 켜면 한 반주기 동안 vx = 0.05 + 클록 리셋. 착지 목표 변경은 다음 단계(R1b)로 미룬다.
- 판정: 표준 앞 밀기 평가의 이륙·회복률(원래 병목) + **무외란 서기 트리거율 ≈ 0**(오탐) + 뒤 밀기 회귀 없음.
- 승인: 배포 러너의 클록·명령 의미 변경이 걸리므로 **학습 전에 사용자에게 '감지기 계약' 승인**을 받는다. 시뮬 실험 자체는 env 쪽 함수로 먼저 가능.

**보행 1개 — W2-1: 스텝 모드 속도 명령 0.05→0.15 m/s 커리큘럼 + Raibert/LIPM 착지 목표(§0b)**
- 형태: A0m-13100 출발, 선속도 추적 항을 스텝 모드에서 켬(현재 w1_stand_gated 로 꺼져 있음 — `hang_stand_env_cfg.py:2122-2128`). r_foot 앞 목표 = 0.227·v_cmd + 0.24·(v − v_cmd), 0.15 m/s 상한.
- 판정: 속도 추적 오차, 걸음 길이 vs v·T_h, 놓친 스텝 비율, feet_slide, 슬루 포화율, 서기 모드 회귀 없음.
- 위험: 추종 항과 스텝 항의 상충(§2 위험). 0.15 이상은 공중 시간 연장 실험을 별도로.

출처: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) `corl_rewards.py:161-200` · [HugWBC](https://github.com/InternRobotics/HugWBC) `h1.py:776-778` · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) `curriculums.py:133-170` · van Marum 2024 arXiv 2404.19173 · Atalante arXiv 2203.01148 · Ferigo 2021 arXiv 2104.14534 · [Faraji et al. 2019 time-projection 스텝 제어(IJRR)](https://journals.sagepub.com/doi/10.1177/0278364919835606)(고전 제어, 참고만)

## 라운드 10 (2026-10-04 09:30) — W2 요 표류, '정책 교정 실패 vs 지지발 슬립' 판별, 사건 기준 클록 오프셋의 일관성 요건

전제(측정 사실, 코디네이터 전달)
- STEP-A0m primitive PASS(α=0 착지 94 %), 서기 보존 FAIL(외란 OFF 망각).
- W2-1/2/3(0.05–0.15 m/s): 추종 오차 0.03–0.06, 교대 94 %. 그러나 **요 표류 10–18°/8 s**. 요 추종 보상(std 0.5·0.1)과 평가 HeadingHold(kp 1, ±0.6 rad/s) 모두 효과 없음.
- diag: 표류의 70 % 가 한 발 지지 중, 지지발 요 회전 합 14°(7 s), 한 발 지지 중 7.9°, 방향 무작위. W2-5(스텝 모드 hip_r_deviation OFF) 대기.
- 회복: G23R-a(외란만) 앞 스텝 0 %. G23R-b(밀림 시 스텝 모드 힌트·감쇠)는 외란 낙상 0.50→0.81 — 클록 위상 정렬 없는 스텝이 해로움.

근거 설정: W2d `logs/rsl_rl/biped12_stepb0/2026-10-04_09-06-04_w2d_seed1_from_w2c17796/params/env.yaml`(hip_r_deviation = joint_deviation_l1 `.*_hip_r_joint` −3.0, joint_deviation_hip = `.*_hip_a_joint`·`.*_hip_r_joint` −0.5, foot_yaw_slip_step, heading_command True·stiffness 0.5, 마찰 DR 정지 0.5–1.1·동 0.4–0.9·곱셈 결합·지면 1.0).

### 0. 계산 (우리 자산·설정, 가정 명시)

**(a) 힙 요 관절 벌점의 크기** — 우리 힙 요(hip_r)에는 L1 벌점 −3.0 과 −0.5 가 겹쳐 실효 **−3.5/rad** 이다.
- 힙 요를 10° 쓰면 3.5 × 0.175 × 0.02 = **0.012/틱**, 14° 면 0.017/틱. 서기·스텝 모드 양(+) 합(≈ 0.08/틱, 라운드 7)의 15–20 % 다.
- 공개 구현은 훨씬 작다.
  - Isaac Lab G1: hip_yaw·hip_roll L1 −0.1 → 10° 에 0.00035/틱(우리의 1/35). `isaaclab_tasks/.../velocity/config/g1/rough_env_cfg.py` joint_deviation_hip(설치본 `:57-61`).
  - Berkeley Humanoid: HR·HAA L1 −0.1(`exts/.../velocity/velocity_env_cfg.py:296-299`).
  - unitree G1: 힙 롤·요 위치 **제곱합** −1.0(`legged_gym/envs/g1/g1_env.py:122-123`, `g1_config.py:86`) → 10° 에 0.03 × 0.02 = 0.0006/틱.
  - humanoid-gym: 요·롤 노름이 **0.1 rad(5.7°) 데드밴드** 안이면 벌점 없음, 넘으면 exp(−100·초과)로 양(+) 보상 급감(가중 +0.5, `humanoid_env.py:362-372`, `humanoid_config.py:207`).
- → 우리 힙 요 벌점은 공개 구현보다 **20–35 배** 크다. 스윙 중 발을 다시 겨누는 데 필요한 5–10° 힙 요 사용이 반주기마다 0.1–0.27 손해다. 같은 반주기의 스텝 이벤트 이득(≈ 0.6)의 1/6–1/2 에 해당한다.

**(b) 지지발 요 마찰 한계** — 발 접촉은 RA30P 패드 4개다(`rl_walking/assets/biped12_final3.urdf` 의 `*_RA30P` 조인트 원점: 좌우 ±0.025 m, 앞 +0.0706 m / 뒤 −0.0509 m; `merge_fixed_joints: true` 로 발목 링크에 병합, `usd_final3/config.yaml`).
- 패드 중심에서 각 패드까지 반경 ≈ **0.066 m**. 한 발 정하중 F = 17.62 × 9.81 ≈ 173 N.
- 요 마찰 토크 한계 ≈ μ·F·r: 네 패드 균등 하중이면 μ 0.4/0.65/0.9 → **4.5/7.4/10.2 N·m**. 앞 또는 뒤 두 패드만 하중(밀어내기·착지 직후)이면 반경이 좌우 0.025 m 뿐이라 **1.7/2.8/3.9 N·m**.
- **비틀림(torsional) 마찰은 없다고 보는 것이 맞다.** USD 에 `torsionalPatchRadius` 속성 문자열이 없다(crate 파일 토큰 grep 0건). env.yaml 로봇 `collision_props: null`. Isaac Lab `sim/schemas/schemas_cfg.py:150-158` "If the radius is zero, no torsional friction is applied".
  비교: LearningHumanoidWalking 의 MuJoCo 바닥은 `friction=".8 .1 .1"`(둘째 값 = 비틀림 계수, 단 condim 3 이라 비활성; `envs/h1/gen_xml.py:114`).
- 스윙 다리를 휘두를 때 지지발에 걸리는 요 반작용 토크는 측정하지 않았다(미측정). 그래서 이 한계가 실제로 넘는지는 §2 측정으로 판별해야 한다.

### 1. ① 이족·휴머노이드 RL 보행에서 요 표류를 잡는 공개 수단

| 수단 | 구현 (commit) · 파일:줄 | 수치 | 우리와 비교 |
|---|---|---|---|
| 요 각속도 추종 + 헤딩 명령 | Isaac Lab velocity env `velocity_env_cfg.py:97-100`(rel_heading_envs 1.0, heading_control_stiffness 0.5), G1 track_ang_vel_z_world_exp 2.0·std 0.5; legged_gym·Berkeley 동형 | 헤딩 오차 → 요 각속도 명령(P 0.5) | 우리도 같은 구조(W2d env.yaml heading_command True·0.5). 보상 std 를 바꿔도 불변 → **보상 신호가 아니라 수단(관절·마찰)이 병목**이라는 신호 |
| 힙 요 편차 벌점 | Isaac Lab G1 −0.1 L1, Berkeley −0.1 L1, unitree −1.0 제곱, humanoid-gym 데드밴드 0.1 rad | §0a | 우리 −3.5 L1 은 20–35배 |
| 지지발 미끄럼 벌점 | humanoid-gym foot_slip = Σ √‖v_foot,xy‖ × 접촉, −0.05(`humanoid_env.py:308-318`, `humanoid_config.py:195`); Isaac Lab/Berkeley feet_slide −0.25(Berkeley `velocity_env_cfg.py:283-285`) | 병진 미끄럼만 | 우리 foot_yaw_slip_step(접촉 발 요 각속도) −0.5 는 **회전 슬립을 직접 벌하는 드문 형태**. 정책이 막을 수단이 없으면 벌점만 쌓임 |
| 착지 시 발 방향 겨누기 | LearningHumanoidWalking 발판 목표(요 포함)·몸통 요 목표(orient 0.05, `tasks/stepping_task.py:81,117,128-136`) | 발판 계획이 요를 줌 | 우리는 발 요 목표 없음 |
| 요 각속도 보상 형태 | LHW `tasks/rewards.py:27-37` exp(−10·|e|³), walking_task 0.15 | — | 형태 차이는 부차적(§1 첫 행 해석) |
| 마찰·접촉 모델 | Isaac Lab 비틀림 패치 반경(`schemas_cfg.py:150-158`), MuJoCo condim 4/6 비틀림 계수 | 기본 0 | 우리 0(§0b) |

**정리**: 공개 구현들은 요를 (i) 헤딩 P 명령 + 요 각속도 추종과 (ii) **약한** 힙 요 벌점(또는 데드밴드)으로 잡는다. 정책은 스윙 중 힙 요로 다음 발을 다시 겨눌 수 있다.
우리 설정은 (ii) 가 20–35배 강해서 이 경로가 사실상 막혀 있다. 이것은 W2-5 가설과 같은 방향이다(공개 수치 근거).

### 2. ② '정책 교정 실패' vs '지지발 슬립(물리)' 판별

diag 수치(지지발 요 회전 합 14°/7 s, 한 발 지지 중 7.9°)는 **지지발이 접촉한 채 월드에서 회전했다**는 뜻이므로 **슬립은 이미 일어나고 있다**(정의상 물리). 질문은 두 가지로 나뉜다. (A) 슬립을 왜 막지 못하나(마찰 한계 vs 정책이 큰 요 토크를 만듦), (B) 슬립을 왜 다음 걸음에서 되돌리지 못하나(힙 요 재조준 막힘).

| 시험 (모두 평가만, 학습 없음) | 방법 | 판정 |
|---|---|---|
| T1 분해 | 한 발 지지 구간마다 골반 요 변화 = 지지발 월드 요 변화(슬립) + 지지 다리 힙 요 관절 변화(+ 다른 관절의 요 기여, 순기구학) | 표류 ≈ 지지발 슬립이면 (A) 가 주 경로. 힙 요 변화가 슬립과 반대 부호로 상쇄하지 않으면 정책이 보정하지 않는 것 |
| T2 확실 접촉 한정 | 지지발 요 회전을 Fz > 0.5 mg(확실 지지)와 Fz < 0.3 mg(이륙·착지 가장자리)로 나눔 | 회전이 가벼운 접촉 틱에 몰리면 '두 패드만 닿는 가장자리 슬립'(§0b 한계 1.7–3.9 N·m)이 원인 |
| T3 토크비 | 지지 다리 힙 요 관절 토크 |τ_hip_r| ÷ (μ_동·Fz·r_eff), r_eff 0.066(4패드)/0.025(2패드) | 슬립 순간 비율 ≈ 1 이면 마찰 한계(물리). ≪ 1 인데 회전하면 접촉 모델·측정 의심 |
| T4 반사실 물리 | 같은 정책·seed 로 (i) 동마찰 ×1.5, (ii) 비틀림 패치 반경 0.02–0.04 m(`CollisionPropertiesCfg`)만 바꿔 평가 | 표류가 절반 이하로 줄면 **물리(슬립) 지배**. 불변이면 정책 쪽 |
| T5 정책 응답 | HeadingHold ±0.6 rad/s 명령에 대해 착지 순간 스윙발 요(골반 대비)가 명령과 상관이 있는지 | 상관 ≈ 0 → 정책이 재조준 수단을 안 씀(힙 요 벌점 의심, W2-5 가 직접 시험) |

- 해석 주의: T4 의 물리 변경은 **판별용**이다. 실물 패드 고무의 비틀림 저항은 측정값이 없다(가정 범위로만 표시, CLAUDE.md "실측 없는 파라미터는 가정 범위").
  표류가 물리 지배로 판정돼도 실물에서 같은지는 현장 확인 필요(영상 + 요 IMU 로 정지 보행 표류 측정).
- W2-5(힙 요 벌점 OFF)는 T5 가설의 단일 변수 시험으로 적절하다. 결과 해석은 T1·T4 와 함께 한다. W2-5 로 표류가 줄면서 T4 도 크면 '슬립은 그대로, 재조준으로 상쇄'가 된다. 이 경우 실물에서는 슬립 크기가 달라 보정량이 맞지 않을 수 있다.

### 3. ③ 사건 시점 기준 위상을 '클록 오프셋'으로 주는 설계 — 시뮬–실물 일관성 요건 (승인안 C4 보강)

공개 사례 없음(라운드 9 §1). G23R-b 결과(위상 정렬 없는 스텝 힌트 → 외란 낙상 0.50→0.81)는 **힌트와 위상 정렬이 함께여야 한다**는 근거다.

요건 목록(구현 전 테스트로 고정할 항목)
1. **같은 함수, 같은 입력.** 감지기·오프셋 계산은 env 와 러너가 같은 모듈을 import 한다(현재 클록 공식 공유 방식: `rl_walking/deploy/policy_runner_stage4.py:138-145` ↔ `mdp_stage4.py:25-33`).
   입력은 actor 가 보는 **지연·잡음 포함 관측**(base_ang_vel_lagged·projected_gravity_lagged·feet_contact_real_v2)에서만 얻는다. 특권 신호(`dst_F_now`, 실제 CoM 속도)는 금지.
2. **위상 연산 정의.** φ = 2π·f·(t − t0) + φ_off. 사건 틱에 φ_off 를 '선택 발의 전환 창 시작(v = 0)'이 되도록 갱신하고, 같은 틱 관측부터 적용한다.
   float32 연산 순서·반올림(k = ⌊v·23⌋)·f(체크포인트별 `CKPT_GAIT_FREQ`, `jetson/scripts/rl_bridge_node.py:327`)를 env 와 같게 한다. 기존 러너 셀프테스트(`policy_runner_stage4.py:571-572` sin/cos 일치)를 오프셋 케이스로 확장한다.
3. **이력 버퍼.** 위상 점프가 정책 관측 5틱 이력에 env 와 같은 방식으로 들어가야 한다(러너 이력 링버퍼에 점프 값을 그대로 push, 보간·평활 금지). 학습에서도 같은 점프를 겪게 한다.
4. **명령 힌트 동기.** vx 힌트 ON/OFF 는 오프셋과 **같은 틱**에 시작해 정확히 N 틱(한 반주기 = 23 틱) 뒤 끈다. 재트리거 금지 시간(≥ 1 반주기)과 최대 빈도를 둔다.
5. **env 내부 보상·보조의 위상 원천 일원화.** sb0/sbi/보조 훅/`gait_clock` 이 모두 `episode_length_buf` 기반이다(`mdp_stage4.py:32`, `mdp_hang.py:2272,2355,2416` 계열). 같은 오프셋 버퍼를 더하는 한 곳으로 바꾼다(라운드 4 §2 #12).
   `episode_length_buf` 자체를 바꾸면 fresh 판정 20여 곳·외란 스케줄·통계 제외가 깨진다.
6. **상태 기계와 안전.** 감지기는 콘솔 상태가 '정책 실행·자립' 일 때만 켠다(매달림·인계·회수 중 OFF). 오탐 시 동작 = 한 반주기 스텝 시도이므로, 무외란 서기에서 트리거율 0 을 시뮬·실물 로그 둘 다로 확인한다.
7. **기록과 재생 검증.** 러너 블랙박스에 감지기 입력·트리거 틱·φ_off·힌트 상태를 기록한다. `rl_walking/deploy/replay_bridge_log.py` 류 재생에서 env 함수가 같은 로그로 같은 트리거를 내는지 확인한다(계약 검증 — 폐루프 성능 증명 아님, CLAUDE.md 6번).
8. **DR 정합.** 감지기 문턱은 고정값(배포값)으로 두거나, DR 을 하더라도 배포값을 범위 안에 둔다. IMU 지연 DR(0–3틱)·접촉 10 Hz hold 아래에서 트리거 지연 분포를 측정하고, 위상 리셋 타깃(v = 0)이 지연만큼 늦게 적용되는 것을 학습이 포함하게 한다.
9. **실물 시험 전 현장 확인 항목**(미검증으로 남김): 실물 밀림 시 IMU 기반 ξ̂ 의 상승 시간, 오탐률(걷는 사람 진동·케이블 접촉).

### 4. 이번 라운드의 우선순위 제안

1. **W2-5 실행 전 또는 병행으로 T1·T4(평가만)** — 표류가 물리 슬립 지배인지 먼저 갈라야 W2-5 결과를 해석할 수 있다. T4 는 물리 파라미터만 바꾼 대조군이다(CLAUDE.md 1번).
2. **W2-5 의 크기 대안**: OFF 대신 humanoid-gym 식 데드밴드(0.1 rad 안 무벌) 또는 Isaac Lab 수준(−0.1)을 쓰면 서기 모드 힙 정렬 효과(−3.0 을 넣었던 이유)를 덜 잃는다. 스텝 모드 OFF 가 1차로 맞고, 데드밴드형은 서기 퇴보가 보일 때의 차선.
3. 회복 쪽은 §3 요건으로 C4 를 보강해 승인 요청한다. 그 전에는 G23R-b 식 '정렬 없는 힌트'를 반복하지 않는다.

출처: Isaac Lab 설치본(37ddf62687) `schemas_cfg.py`·G1 rough_env_cfg·velocity_env_cfg · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking) · 우리 URDF/USD·env.yaml

## 라운드 11 (2026-10-04 11:00) — 클록 스텝 보행의 회전 학습, 스텝·회복 획득 뒤 서기 정숙 유지

전제(측정 사실, 코디네이터 전달)
- T4 마찰 ×1.5 → 요 표류 −21 %(슬립은 부분 기여).
- W2-5(스텝 모드 힙 회전 벌점 OFF) 표류 불변.
- W2-6(스텝 모드 요 명령 ±0.3 rad/s 무작위, 요 추종 std 0.1 × 가중 1.5): 1500 반복 동안 track_ang_vel 0.09 평탄 = 회전 추종 미학습. 힙 회전 한계 ±24–29°.
- 회복은 크게 진전했다: W2-4(보행 + G21a 외란)에서 서기 밀림 앞 스텝이 자발 출현했고, G24R-a 에서 sf15 낙상 91→32 %·스텝 93 %·+10 cm, sf12 4.5 %, sf10 1.8 %, sb10 5.4 %, b18 0 %.
- 그러나 서기 흔들림 nd 7.1 %·d3nd 11.6 %(G21a 0.9·3.6), 피치 1–1.5 Hz. 클록 고정 반사실 7.1→5.4 %(주원인 아님). G24R-b(서기 비율 0.7·무외란 0.5) 학습 중.

### 0. W2-6 이 회전을 못 배운 직접 원인 후보 — 보상 폭(std)의 기울기 소실 [설정 확인 + 계산]

- 설정: W2f `logs/rsl_rl/biped12_stepb0/2026-10-04_10-45-08_w2f_seed1_from_w2e20294/params/env.yaml`. `track_ang_vel_walk` = `w1_walk_gated(track_ang_vel_z_world_exp, std 0.1)`, 가중 1.5. 명령 ang_vel_z U(−0.3, 0.3), heading_command False.
  Isaac Lab 식은 exp(−e²/std²)(`isaaclab_tasks/.../velocity/mdp/rewards.py:318-326`).
- **회전하지 않는 정책의 오차 e = |명령|(0–0.3 rad/s)에서의 기울기 |dr/de|**:

| std | e = 0.05 | e = 0.15 | e = 0.30 |
|---|---|---|---|
| 0.1 (우리) | r 0.78 / 7.8 | r 0.11 / 3.2 | **r 0.0001 / 0.01** |
| 0.45 (humanoid-gym 상당) | 0.99 / 0.49 | 0.90 / 1.33 | 0.64 / 1.90 |
| 0.5 (Isaac Lab G1·Berkeley·unitree) | 0.99 / 0.40 | 0.91 / 1.10 | 0.70 / 1.67 |

- 명령의 절반 이상(|명령| > 0.15)에서 std 0.1 은 **보상도 기울기도 거의 0** 이다. '회전을 조금 해 보기'에 대한 신호가 없다. 평탄한 0.09 와 정합한다.
  std 0.1 은 요 표류(작은 오차)를 잡기에는 날카롭지만, 큰 명령을 처음 배우기에는 부적합하다.
- 공개 구현의 요 추종 폭(모두 큰 쪽):
  - Isaac Lab G1: std 0.5, 가중 2.0, 명령 ±1.0(`config/g1/rough_env_cfg.py:29-31`, `flat_env_cfg.py:41`).
  - Berkeley: std √0.25 = 0.5, 가중 0.5(`velocity_env_cfg.py:265-266`).
  - unitree: σ² = 0.25(→ std 0.5), 명령 ±1(`legged_robot_config.py:48,120`; `legged_robot.py:698-701`).
  - humanoid-gym: exp(−e²·5)(→ std 0.45), 가중 1.1, 명령 ±0.3(`humanoid_config.py:171,185,202`; `humanoid_env.py:436-443`).
- 필요한 동작 크기: 0.3 rad/s 면 반주기(0.455 s)마다 **7.8°** 회전 → 힙 요 ±8° 정도. 한계(±24–29°) 안이다.

### 1. (a) 클록 스텝 보행에서 회전을 학습시킨 사례

| 출처 (commit) | 회전용 장치 | 파일:줄 | 우리에게 |
|---|---|---|---|
| walk-these-ways (0e7236b) | **Raibert 발 배치에 요 항**: 요 명령 → 발 좌우 목표 오프셋. y_vel_des = yaw_vel_des × (stance_length/2), desired_ys_offset = phase × y_vel_des × (0.5/f), 뒤 다리 부호 반전(사족) | `go1_gym/envs/rewards/corl_rewards.py:184-191` | 이족 형태: 회전 시 스윙발 착지 요(골반 대비) 목표 = ω_cmd·T_h(0.3 rad/s → 7.8°)를 r_foot 에 요 성분으로 추가. 우리 r_foot 은 위치만 보고 발 요를 보지 않는다(`mdp_hang.py:2375-2382`) |
| walk-these-ways | **명령 공간 격자 커리큘럼**(vx·vy·wz bin). 성공 bin 주변(local_range 0.5)으로 확장, 성공 = 추종 보상 > 문턱 × 가중(선속도 0.8, **요 0.7**, 접촉 0.9). 요 명령 ±1 → 한계 ±5, bin 21개 | `go1_gym/envs/base/curriculum.py:113-150`; `envs/base/legged_robot.py:727-736`; `scripts/train.py:23-26,155,170,185` | 요 명령을 ±0.3 에서 바로 균등으로 주지 말고 ±0.05 에서 시작해 '요 추종 > 0.7' 일 때 넓힌다 |
| LearningHumanoidWalking (cd8c655) | 곡선 발판 시퀀스(CURVED 모드 15 %) + 몸통 요 목표 + 요 각속도 exp(−10·e³) | `tasks/stepping_task.py:81,117,128-136,290-300`; `tasks/rewards.py:27-37`; `tasks/walking_task.py:121-141` | 발판·몸통 요 목표가 회전의 '위치' 신호를 준다. 우리 W2 는 회전에 대한 위치 목표가 없다 |
| HugWBC (eb831af) | 요 명령 + 보행 주파수 명령, 추종 가중 3.0(요) / 2.0(선속도) | `legged_gym/envs/h1/h1_config.py:193-194` | 요 추종 가중을 선속도보다 크게 둔 예 |
| Isaac Lab G1·Berkeley | 클록 없음, std 0.5 추종만으로 ±1 rad/s 회전 학습(scratch) | 위 §0 | 폭 0.5 가 표준 |

- **스텝 primitive 위에 회전을 얹은 공개 사례는 찾지 못했다.** 위 사례는 모두 처음부터 학습이다. 회전용 보조력/잔차를 쓴 사례도 찾지 못했다.

**우리에게 적용 (단일 변경 순위)**
1. **요 추종 std 0.1 → 0.5**(가중·명령 범위 그대로) — §0 의 기울기 소실이 가장 직접적인 원인 후보이고, 공개 구현이 모두 0.45–0.5 다. 표류(작은 오차) 억제력은 약해진다. 표류는 §0 판단상 슬립·재조준 문제(라운드 10)라 별도로 다룬다.
2. (1이 부족하면) **회전 착지 요 목표**(walk-these-ways 식 피드포워드: 스윙발 착지 요 = ω_cmd·T_h)를 r_foot 에 추가.
3. **요 명령 커리큘럼**(±0.05 → ±0.3, 요 추종 > 0.7 일 때 확장).
- 위험: 회전 학습이 교대·착지(94 %)를 깎을 수 있다. 판정에 교대율·착지율·서기 회귀를 같이 둔다.

### 2. (b) 스텝·회복을 얻은 뒤 서기 정숙이 무너지는 문제 — 단일 정책 유지 사례

| 출처 | 서기 유지 장치 | 수치 / 파일:줄 | 비고 |
|---|---|---|---|
| HugWBC | **서기 env 의 클록을 지지 위상에 고정** + 서기 보상(양발 접촉 +1, 양발 공중 −1, 관절 편차 −2) + 서기 env 비율 1/10 | `h1.py:631-637,776-778,1519-1535`; `h1_config.py:227-229` | 우리 클록 고정 반사실(평가만) 7.1→5.4 % 는 '학습은 도는 클록으로 하고 평가만 고정'이다. **학습에서 서기 클록을 고정**하면 더 줄 가능성(미검증). 단 배포 러너가 서기 때 클록을 고정해야 하므로 계약 변경(승인) |
| HugWBC interrupt | 상체 교란을 **서기 env 에만** 줌(`stand_interrupt_only`) | `h1interrupt.py:284-288` | 외란을 서기 표본에 몰아 '서기 + 외란'을 강화 |
| van Marum 2024 | 서기 명령에서 접촉 항 상수 1, 나머지 항이 자연히 서기를 선호 | 논문(라운드 7 §3) | 서기 전용 정숙 항 없이 유지 |
| MuJoCo Playground | stand_still = Σ|q − q_default| × (명령 < 0.1): G1 **−1.0**, T1·Berkeley 0.0 | `_src/locomotion/g1/joystick.py:75`; `t1/joystick.py:74,627-633` | 로봇별로 켜고 끔. 우리 경험(G15: 정숙 항이 회복 낙상 증가)과 같은 상충 |
| legged_gym | stand_still 기본 −0 | `envs/base/legged_robot.py:900`; `legged_robot_config.py:146` | 기본 비활성 |
| CaT | 명령 < 0.1 일 때 관절속도 > 4 rad/s 를 **확률적 종료**(max_p 0.1) | `utils/cat/constraints.py:223-235`; `cat_flat_env_cfg.py:339-346` | 벌점이 아니라 소프트 종료. 정숙 위반의 '미래 보상 상실'을 키운다 → 회복에도 불리할 수 있음 |
| Isaac Lab G1·Berkeley | 서기 env 2 %(`velocity_env_cfg.py:97/100`) | — | 서기 비율이 낮아도 걷기 중심 평가라 서기 정숙 지표가 없음 |
| humanoid-gym | 명령 < 0.2 면 0, 클록 계속(제자리걸음) | `envs/base/legged_robot.py:336` | 서기 정숙을 포기한 설계 |

**우리 기록과의 정합**
- 정숙 보상 확대(G15)는 회복 낙상을 늘렸다. 정착 보너스 연속화(G21a3, state_gated)는 흔들림을 줄였다. 관절속도 관측 지연 DR(c1)은 충격 생존을 회복시켰다.
- 공개 사례도 '서기 정숙 항을 게이트 없이 세게'(Playground G1, CaT)와 '서기 정숙 포기'(humanoid-gym)로 갈린다. **회복 게이트로 닫힌 동안만 주는 우리 state_gated 방식**(`mdp_hang.py:400-404`)이 둘의 상충을 피하는 쪽이다.

**1–1.5 Hz 피치 흔들림 판별 제안(평가만, 학습 없음)**
1. **클록 누설 확인**: 서기 모드(명령 0, 무외란)에서 두 발 Fz 차의 스펙트럼에 1.1 Hz 피크가 있는지 본다(학습 클록 1.1 Hz).
   피크가 있고 피치 흔들림과 같은 위상이면 '스텝 모드 하중 이동 패턴이 서기로 새는 것'(모드 혼동)이다.
   클록 고정 반사실의 −24 % 가 이 성분의 크기를 시사한다(나머지 76 % 는 다른 원인).
2. **피드백 루프 확인**: 같은 정책을 자이로·관절속도 관측 지연 0 고정 vs 학습 DR 상한(3틱)으로 평가해 흔들림이 지연에 민감한지 본다.
   민감하면 '회복 이득이 커진 피치 되먹임 + 지연'의 루프다(G17·G19 의 루프 귀속 방식, 이번엔 1–1.5 Hz 대역).
3. 판별 결과에 따른 단일 변경
   - 1 이 주원인 → (i) 학습 중 서기 env 클록을 지지 위상에 고정(HugWBC, 계약 변경·승인), 또는 (ii) 서기 모드·게이트 닫힘 한정 '좌우 하중 진동' 벌점(1.1 Hz 성분 억제, 보상 변경, 승인 불요).
   - 2 가 주원인 → 서기 표본에서 관측 지연 DR 상한 쪽 비중 증가(c1 계열, 관측 계약 불변). 또는 정착 보너스(G21a3 형) 가중을 서기 모드에서만 소폭 증가. 정숙 항 확대는 G15 처럼 회복을 깎을 수 있으므로 sf15·b18 회귀를 함께 판정.
- G24R-b(서기 비율 0.7·무외란 0.5)는 HugWBC·Isaac Lab 보다 서기 비중이 훨씬 높다. 표본 비율로 해결되면 가장 싸다. 그 결과를 1·2 판별과 함께 해석한다(서기 비율 ↑ → 회복 표본 ↓ → sf15 회귀 위험).

### 3. 다음 단일 실험 (권장)

1. **W2-7 = W2-6 + 요 추종 std 0.1 → 0.5**(단일 변경). 판정: track_ang_vel 상승, 회전 명령 대비 실제 요 각속도 기울기, 교대·착지 유지.
2. **서기 흔들림 판별 평가 2종**(§2: 1.1 Hz 하중 스펙트럼, 지연 민감도) — G24R-b 결과와 같이 해석한 뒤 서기 단일 변경을 고른다.

출처: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [HugWBC](https://github.com/InternRobotics/HugWBC) · [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · [legged_gym](https://github.com/leggedrobotics/legged_gym) · [CaT](https://github.com/Gepetto/constraints-as-terminations) · Isaac Lab 설치본 · van Marum 2024 arXiv 2404.19173

## 라운드 12 (2026-10-04 12:00) — 서기와 보행·회복을 한 정책에 간섭 없이 두는 방법, 직진 기준 모방에서 회전을 가능하게 하는 방법

전제(측정 사실, 코디네이터 전달)
- **W2-7(요 추종 std 0.5) FAIL**: vx 0.10 에서 일정 요 명령 ±0.3 rad/s 에 대한 실제 회전 비율 −0.05 – 0.24, 회전하지 않음. 학습 보상 상승은 넓은 커널의 산물이었다.
- 구조 의심: 스텝 모방 보상이 보행 계보 내내 전체 크기로 켜져 있다. r_qref(힙 요 관절 가중 1.0·σ 0.10 rad), r_foot(스윙발 상대 궤적, 옆 0·σ 3 cm), r_cref(접촉 위상)이고, 기준은 직진 스텝이다. 다음은 W2-8(스텝 모드 모방 OFF).
- 회복 정책(G24R-a)의 서기 흔들림(피치 1–1.5 Hz)이 배포의 주 장애다. 서기 표본 비율을 올리면 흔들림은 해결되지만 회복이 무너졌다(G24R-b). 사용자 판단: 한 연속 정책 안에서 서기와 보행·회복 표현이 간섭한다 → 다음은 모드별 게이팅 또는 조건부 행동.

### 0. 계산 — 직진 모방 3항이 회전을 얼마나 벌하나 (우리 기준 `ref_step_primitive.npz`, 가중 `w_q` 합 9.2, 보상 = 가중 × 값 × dt 0.02)

- 필요한 동작: 0.3 rad/s × 반주기 0.455 s = 반주기당 7.8°(라운드 11). 힙 요를 한 다리 또는 두 다리에서 6–8° 쓴다.
- **r_qref**(가중 1.5, σ 0.10, 힙 요 w 1.0): exp(−Σ w·Δq²/(Σw·σ²)).
  - 힙 요 6° 한 다리 → 0.90(손해 0.0031/틱).
  - 8° 한 다리 → 0.81(0.0058).
  - **8° 두 다리 → 0.65(0.0104/틱)**.
- **r_foot**(가중 2.0, σ 3 cm): 반주기 시작 대비 상대 궤적을 골반 요 프레임으로 비교한다(`mdp_hang.py:2375-2382`). 골반이 7.8° 돌면 앞 7–14 cm 의 발 위치가 옆으로 1–2 cm 어긋난다(기하 추정).
  - 1.5 cm 오차 → 0.78(손해 0.0088/틱), 2 cm → 0.64(0.0144/틱).
- **요 추종 이득**(W2-7: std 0.5, 가중 1.5): 오차 0.3 → 0 으로 줄이면 +0.30 × 1.5 × 0.02 = **0.0091/틱**.
- → **모방 2항의 회전 손해(합 ≈ 0.01–0.025/틱)가 요 추종 이득(0.009/틱)과 같거나 크다.** 회전하지 않는 것이 보상상 이득이고, W2-7 결과와 정합한다(계산, 기하 추정 포함; W2-8 이 직접 시험).

### 1. Q1 — 서기와 보행·회복을 한 정책에 간섭 없이 두는 공개 방법

| 방식 | 출처 (commit) · 파일:줄 | 구체 메커니즘 | 관측/행동 계약 영향 |
|---|---|---|---|
| **명시 서기 플래그(이진 명령)** | FALCON (a967a6d) `humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py:294`; 관측 설정 `config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml:13,31`(command_stand); `config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml:24`(stand_prob 0.4) | commands[:,4] = 1(걷기·제자리 디딤) / 0(서기)를 actor 관측에 넣는다. 서기면 속도 명령 0. 접촉 보상은 서기에서 '항상 지지'로 바꾼다(`:504-512`). **서기 중 한 발이라도 들면 벌점 −4**(`:515-528`, 보상 yaml `:59`). 서기 관절속도 −1e-3·뿌리 이동 −5(`:550-582`, yaml `:46,49`), 힙 위치 벌점은 걷기에서만(`:530-534`). 위상(클록)은 서기에도 계속 돈다(`envs/locomotion/locomotion_ma.py:109-115`) | **관측 차원 +1**(우리 계약 위반). 서기 중 발 들기를 금지하므로 **서기 상태에서의 회복 스텝과 충돌** |
| **모드 원-핫 + 서기 중 무작위 명령** | LearningHumanoidWalking (cd8c655) `tasks/walking_task.py:21-41`(STANDING/INPLACE/FORWARD 인코딩, **STANDING 에서 mode_ref 를 U(−1,1)³ 잡음으로 줌**), `:115-128`(서기면 클록 보상 상수·추종 목표 0); 관측 `envs/jvrc/jvrc_walk.py:43,66-67`, `envs/h1/h1_walk.py:115-123`(clock 2 + 원-핫 3 + ref 3) | 모드를 원-핫으로 알려 주고, 서기에서는 속도 기준값을 일부러 잡음으로 줘 '서기 플래그가 우선'임을 학습시킨다. 관측 클록은 서기에도 돈다 | 관측 +3(위반) |
| **보행 파라미터 명령** | walk-these-ways `go1_gym/envs/base/legged_robot.py`(관측에 주파수·위상·지속비·발높이 등 gait 명령), 명령 커리큘럼 `curriculum.py:113-150` | 보행 형태를 명령으로 조건화(서기 모드는 없음) | 관측 추가(위반) |
| **서기에서 클록 고정** | HugWBC (eb831af) `legged_gym/envs/h1/h1.py:776-778`(standing env 의 foot_indices = 0 → 사인 클록 = 0, 원하는 접촉 = 양발), 서기 보상 `:1519-1535`, 서기 비율 1/10 `:631-637` | 클록 값이 사실상 서기 플래그 역할 | **차원 불변**. 다만 클록 값의 의미가 바뀌어 배포 러너가 서기에서 클록을 고정해야 함(계약 의미 변경 → 승인) |
| **명령 크기로 보상 게이트** | legged_gym `legged_robot.py:882-893`(feet_air_time × (‖cmd‖ > 0.1)); Isaac Lab `velocity/mdp/rewards.py:49-68`(biped 공중시간 × (‖cmd‖ > 0.1)); MuJoCo Playground `_src/locomotion/g1/joystick.py:75` + `t1/joystick.py:627-633`(stand_still × (‖cmd‖ < 0.1), G1 −1.0); humanoid-gym `envs/base/legged_robot.py:336`(‖cmd‖ < 0.2 → 0) | 보상만 모드별로 다르게, 관측은 명령 그대로 | 불변(우리 w1_stand_gated/w1_walk_gated 가 이것) |
| **두 정책 + 전환(학습된 판정)** | ABS (LeCAR-Lab, 9b95329): 민첩 정책 + 회복 정책, 배포 시 reach-avoid 가치망 RA 값이 문턱(−ε)을 넘으면 회복 정책으로 전환 — `deployment/src/abs_src/depth_obstacle_depth_goal_ros.py:483-496` | 별도 정책 2개, 전환 판정기(학습된 RA 망) | 정책별 계약은 그대로. 러너에 전환 로직 추가 |
| **두 정책 묶음(서기·보행 actor 분리)** | FALCON `humanoidverse/agents/ppo_locomanip.py:84-96,189-192`(서기 체크포인트와 보행 체크포인트를 actor[0]/actor[1] 로 각각 로드, 추론 함수 2개 반환 — 선택 로직은 호출 측, 세부 미확인) | 모드별 별도 네트워크 | 계약은 정책별 |
| **전문가 혼합(MoE) actor** | MoE-Loco(arXiv 2503.08564, IROS 2025): 다중 과제 기울기 충돌을 MoE 로 완화. 공식 저장소 `hrh6666/MoE-Loco`(78696e9)는 **README 만 있고 코드 없음**(확인). Yang 2020 'multi-expert learning'(Science Robotics)도 논문만 | 게이트망이 전문가 출력을 섞음 | 네트워크 구조 변경(우리 동결 대상) |

**우리에게 적용** (actor 관측 계약 불변 조건)
- 공개 구현에서 서기·보행을 한 정책에 둔 것들은 대부분 **관측에 모드 정보를 추가**한다(FALCON +1, LHW +3, WTW gait 명령). 우리 계약에서는 불가하다.
  계약 안에서 가능한 것은 (i) 기존 명령 채널 의미(vx < 0.03 = 서기 — 이미 사용), (ii) HugWBC 식 서기 클록 고정(차원 불변·의미 변경), (iii) 정책 분리 + 러너 전환이다.
- 공개 '서기 플래그' 설계(FALCON)는 **서기에서 발 들기를 −4 로 금지**한다. 즉 그들의 서기 모드에는 회복 스텝이 없다. 우리 목표(서기 중 밀리면 디딤)와 정면으로 다르다.
  사용자 판단(서기·회복 간섭)이 맞다면, 공개 사례 중 그대로 맞는 것은 **ABS 형 '두 정책 + 관측 기반 전환'**이다.
  - 조용한 서기 정책: G21a 계열, 흔들림 nd 0.9 %.
  - 회복 정책: G24R-a 계열(sf15 낙상 32 %).
  - 전환 판정기: 라운드 9·10 의 관측 신호 감지기(추정 캡처 포인트 ξ̂, 지연 IMU)나 ABS 처럼 학습된 위험 가치. 히스테리시스를 둔다(회복 정책이 '회복 완료'(라운드 6 게이트 닫힘 조건)를 내면 서기 정책으로 복귀).
- 정책 분리안의 위험(우리 기록 근거)
  - **전환 순간의 목표 불연속**: 우리 배포 블렌딩 경험상 블렌딩이 거동을 바꾼다(메모리: deployment-wrapper-blending). 전환 경로 자체를 시뮬 평가에 넣어야 한다.
  - 두 정책의 관측 이력·LPF 상태 공유(라운드 4 §2 의 내부 상태들 — 두 정책이 같은 관측 파이프라인을 쓰면 이력은 연속).
  - 감지기 오탐(서기 정책 → 회복 정책으로의 잘못된 전환은 흔들림을 다시 부름).
  - 판정은 '전환 포함 시스템'으로, 서기 흔들림(nd·d3nd)과 sf15·b18 을 동시에.
- 단일 정책을 유지하는 차선: HugWBC 식 **학습 중 서기 클록 고정**(차원 불변, 러너 의미 변경 승인 필요). 라운드 11 반사실(평가만 고정 −24 %)보다 클 가능성이 있지만, 회복 스텝이 클록 위상에 의존한다면(G23R-b 교훈) 서기 중 회복 스텝의 위상 정보를 잃는 상충이 있다(가설).

### 2. Q2 — 직진 기준 모방·참조 기반 보행에서 회전을 가능하게 하는 방법

| 방법 | 출처 · 파일:줄 | 메커니즘 | 우리에게 |
|---|---|---|---|
| **요 정렬(heading-invariant) 추적** | BeyondMimic `tasks/tracking/mdp/commands.py:283-293`(delta_ori_w = **yaw_quat**(로봇 앵커 × 기준 앵커⁻¹)로 기준 몸체 위치·자세를 매 틱 로봇 요에 맞춰 재정렬) | 기준이 로봇 방향을 따라 돌므로 '회전 중'이어도 자세·발 궤적 오차가 생기지 않음 | 우리 r_foot 은 골반 요 프레임 상대 궤적이지만 기준점(반주기 시작 발 위치)이 회전 전 프레임이라 회전하면 오차(§0). 기준점도 같은 요로 재정렬하면 회전 중립 |
| **국소 프레임 보상(track_root=False)** | MimicKit `mimickit/envs/deepmimic_env.py:802-818`(not track_root 이면 뿌리 요·위치를 제거한 국소 프레임으로 오차 계산), `:603`(convert_to_local) | 같은 효과 | 같음 |
| **모방에서 힙 요 제외(부분)** | humanoid-gym `humanoid_env.py:121-142`: 사인 기준은 **피치 관절(힙 피치·무릎·발목, idx 2·3·4 / 8·9·10)에만**, 요·롤 기준은 0. 요·롤 편차는 default_joint_pos 의 **0.1 rad 데드밴드** 밖에서만 벌(`:362-372`). 단 joint_pos 보상(`:272-279`)은 전 관절 노름이라 요 편차도 약하게 들어감 | 회전에 필요한 힙 요(±0.1 rad 이내)를 모방이 거의 벌하지 않음 | 우리 w_q 의 힙 요 가중 1.0 을 0 으로(또는 데드밴드) |
| **회전 전용 참조 추가** | AMP_for_hardware `datasets/mocap_motions/`(leftturn0·rightturn0·trot·pace + hopturn 폴더), `legged_gym/envs/a1/a1_amp_config.py:34,46,143`(요 명령 ±1.57), AMP 관측은 **몸 좌표**(관절·몸 기준 발 위치·몸 속도·높이, `legged_gym/envs/base/legged_robot.py:285-292`) | 스타일 판별기가 회전 동작도 '자연스럽다'고 보고, 과제 보상이 회전 명령을 추종 | 우리 기준은 뒤 스텝 primitive 하나 — 회전형 primitive(골반 대비 발 요 ±ω·T_h) 추가는 기준 확장 |
| **회전 전용 모드** | LHW `tasks/walking_task.py:21-41` INPLACE(요 각속도 ±0.5, 전진 0), 발판 과제 CURVED 15 %·몸통 요 목표(`tasks/stepping_task.py:81,117,128-136`) | 회전만 따로 표본화 | 우리 W2 에서 '제자리 회전' 명령 비율을 별도로 |
| **모방 가중 감쇠** | 우리 `mdp_hang.py` `sbi_anneal_curriculum`(배율 1.0→0.75→0.5→0.25→0.05, 블록 하중 수용 조건) — 현재 보행 계보에서는 미사용(코디네이터 전달); AMP 는 감쇠 대신 과제 보상 혼합 고정(`amp_task_reward_lerp 0.3`, `a1_amp_config.py:164`); Kickstarting 선형 감쇠(라운드 2) | 모방은 탐색용으로만, 이후 과제 보상 위주 | W2-8(OFF)이 극단형. 서기·스텝 품질이 무너지면 감쇠형(배율 0.25)이 차선 |

**요 추종 폭·가중의 전형값**: Isaac Lab G1 std 0.5·가중 2.0(명령 ±1.0), Berkeley std 0.5·0.5, unitree σ² 0.25(std 0.5)·0.5, humanoid-gym std ≈ 0.45·1.1(명령 ±0.3), legged_gym/AMP_for_hardware σ² 0.25·0.5(AMP 는 1/(dt·decimation) 배율: `a1_amp_config.py:118-119`), HugWBC 요 3.0·선속도 2.0, LHW exp(−10·|e|³)·0.15. **std 0.5 가 표준**이다. 우리 W2-7 의 실패는 폭이 아니라 §0 의 모방 상충이 원인일 가능성이 높다.

### 3. 다음 단일 실험 제안

1. **W2-8(스텝 모드 모방 OFF)** — 계획대로. §0 계산상 가장 직접적인 원인이다. 판정에 회전 비율과 함께 교대·착지·놓친 스텝 비율을 둔다(모방이 primitive 형태를 지탱하던 몫이 빠질 수 있음).
   대안 단일 변경(OFF 가 스텝을 무너뜨릴 때): r_qref 힙 요 가중 0 + r_foot 기준점 요 재정렬(BeyondMimic 식)만 — 직진 스텝 형태는 유지하고 회전 벌만 제거.
2. **서기–회복 간섭**: 단일 정책 내 게이팅은 공개 사례상 관측 추가가 필요해 계약과 충돌한다. 계약 안에서는 **ABS 형 두 정책 전환(G21a 서기 + G24R 회복, 관측 기반 감지기·히스테리시스)을 시뮬에서 '전환 포함 시스템'으로 평가**하는 것이 먼저다(학습 불필요, 평가 하네스만).
   이 평가로 흔들림·sf15·b18 을 동시에 만족하면, 단일 정책 게이팅(네트워크 변경·승인 필요)보다 위험이 낮은 경로다. 러너 전환 로직은 배포 계약 변경이므로 실물 전에 승인이 필요하다.

출처: [FALCON](https://github.com/LeCAR-Lab/FALCON) · [ABS](https://github.com/LeCAR-Lab/ABS) · [LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking) · [HugWBC](https://github.com/InternRobotics/HugWBC) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [MoE-Loco (코드 없음)](https://github.com/hrh6666/MoE-Loco) · [MoE-Loco arXiv 2503.08564](https://arxiv.org/abs/2503.08564) ·
[whole_body_tracking (BeyondMimic)](https://github.com/HybridRobotics/whole_body_tracking) · [MimicKit](https://github.com/xbpeng/MimicKit) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) · [legged_gym](https://github.com/leggedrobotics/legged_gym) · [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · Isaac Lab 설치본

## 라운드 13 (2026-10-04 17:00) — 앞 밀기 회복 스텝 중 옆(롤) 낙상, 밀기 해제 뒤 낙상

전제(측정 사실, 코디네이터 전달)
- C5 승인(G21a 서기 + G24R 회복, IMU 감지기 전환), C4 보류. C5 기준선은 서기·전환 기준을 통과했다.
- 남은 실패: 앞 밀기 12 N 낙상 12.0 %(8 seed), 15 N 31.8 %. 인계 A/B/C 불일치 증거 없음(같은 경로 대조 p 0.34).
- **앞 밀기 낙상의 대부분이 옆(롤)이다**: sf12 17/20, sf15 46/69. 거의 모두 **밀기 해제 0.4–0.5 s 뒤**(램프 0.5 s + 유지 1 s, 낙상 ≈ 3.9–4.0 s)에 일어난다. 넘어지는 env 도 2–3회 이륙하지만 걸음이 짧다.
- G24R-d(12–15 N 앞 밀기로 학습)는 악화(sf12 15.3 %, sf15 38 %): 걸음은 길어졌으나 횟수가 줄었고 학습 에피소드가 짧아졌다.
- G24R 학습 외란은 앞/뒤만(옆 제외). 50 Hz MLP, 관측 320, 접촉 on/off 만(CoP 없음), 17.6 kg, 발목 롤 AK45(느림).

표기: **[코드]** = 이번에 해당 줄을 읽어 확인, **[논문]** = 원문/요약 기반, **[계산]** = 직접 계산.

### 0. 우리 코드에서 확인한 사실 [코드] — 결론에 직접 영향

1. **회복 중(게이트 열림)에는 발 간격을 지키는 항이 꺼진다.**
   - G24R-d 보상(`logs/rsl_rl/biped12_stepb0/2026-10-04_16-06-52_g24rd_seed1_from_g24ra21793/params/env.yaml`)에서 좌우 발 간격 항은 `feet_gap` = state_gated(feet_too_close, min_gap 0.22 m, −5) 하나다(`rl_walking/mdp_rewards.py:47-59`).
   - state_gated 는 값 × (1 − rec_open)이다(`mdp_hang.py:400-404`). `stance_shape`(간격 0.22–0.32 m·엇갈림 4 cm)도 state_gated + 서기 게이트다.
   - 게이트는 외력이 있는 동안 열리고(`recovery_gate_step` `mdp_hang.py:312`, `dst_F_now > 0`), 해제 후 기울기 < 6.5°·|ω| < 1.5·|v| < 0.5 가 **0.3 s 연속**이어야 닫힌다(`:313-315`).
   - → 밀기 중과 해제 직후(낙상 시점 0.4–0.5 s 포함 가능)에는 **좁은 걸음·교차 걸음을 막는 항이 없다**.
2. **넓게 딛는 것(외전)은 상시 벌을 받는다.** `joint_deviation_hip`(hip_a·hip_r L1, −0.5)은 게이트와 무관하게 켜져 있다(같은 env.yaml). 옆으로 넓게 디뎌 롤을 잡는 동작이 비용을 낸다.
3. **회복 품질 보상은 양발 접지를 요구한다.** `recovery_quality`(+5, 서기 게이트·열림 게이트)는 require_both 기본값으로 양발 접촉일 때만 준다(`mdp_hang.py:345-371`, 라운드 6). 걸음을 짧게 끊고 빨리 양발로 돌아오는 쪽이 이득이다. '짧은 걸음' 관찰과 같은 방향이다(가설).
4. **학습 외란에 옆 성분이 없다.** `disturbance_reset_g22r` 가 `disturbance_reset(..., lateral_prob=0.0, ...)` 로 호출한다(`mdp_hang.py:1861`). G24R-a 설정의 `lateral_prob: 0.3` 은 덮어써져 무효다.
   옆 성분은 외란 범주 env 의 속도 밀기 ±0.25 m/s(x·y 균등, 3–6 s 간격, `push_with_arm_g22`)뿐이다.
5. **해제 방식과 상승 속도.**
   - 램프 외란은 상승 → 유지 → **즉시 0**(계단 해제)이다(`_dist_profile` `mdp_hang.py:194-200`). 학습에도 같은 계단 해제가 있으므로 '해제' 자체는 분포 안이다.
   - 다만 학습 상승 시간은 1–4 s(G24R-a env.yaml `rise_range`)인데 평가 sf12/sf15 는 **0.5 s** 다. 평가가 학습보다 2–8배 빠른 상승이다(분포 밖).
   - 학습 단계 R3(stage_override 2)의 앞 힘은 10–12 N(`_G22R_STAGES` `mdp_hang.py:1819-1824`).

### 1. Q1 — 앞 밀기 회복 스텝 중 옆(롤) 균형을 지키는 수단

| 수단 | 출처 · 위치 | 식·수치 | 확인 |
|---|---|---|---|
| 발 간격 하한·상한 보상(상시) | humanoid-gym `humanoid/envs/custom/humanoid_env.py:282-292` `_reward_feet_distance` | d = ‖p_L − p_R‖_xy. r = ½[exp(−100·|min(d − 0.2, 0)|) + exp(−100·max(d − 0.5, 0))], 가중 0.2(`humanoid_config.py:176-177,196`). knee_distance 같은 형(0.2). 밀기 xy ±0.2 m/s·4 s(`:154-157`) | [코드] |
| 발 간격 벌점(우리) | `mdp_rewards.py:47-59` feet_too_close: (0.22 − 옆 간격)₊ | −5, **state_gated → 회복 중 꺼짐**(§0-1) | [코드] |
| 발 교차 = 종료 | Atalante(arXiv 2203.01148) Sec. IV-C2: 두 발 볼록껍질 거리 D > 0.02 m 위반 시 종료 | 하드 제약 | [논문] |
| 발 상대 자세 보상 | Atalante IV-D: 평균 발 프레임에서 오른발–왼발 상대 위치·방향 오차 ‖ᵣ₊ₗp_{r−l} − p̂‖² | RBF 정규화 가중합 | [논문] |
| 캡처 포인트 기반(두 축) | Atalante IV-D: 골반을 CP 쪽으로 ‖p_cp − p̂_cp‖, ZMP 를 지지 중심으로. arXiv 2603.08619: r_CP = exp(−d_CP²/σ²), d_CP = **CP–지지 볼록껍질 부호거리**(앞뒤·좌우 공통), critic 에 CoM·CP 특권 | — | [논문] |
| 옆 발 배치 목표 | walk-these-ways Raibert: 공칭 폭 0.3 m + 요 명령 오프셋(`go1_gym/envs/rewards/corl_rewards.py:168-191`), 가중 −10 | 오차 제곱합 | [코드] (사족, 밀기 회복 아님) |
| 롤 각속도·자세 벌점 | 우리 ang_vel_xy_l2 −0.1, flat_orientation_l2 −8(상시); Isaac Lab·legged_gym 표준 ang_vel_xy −0.05 | — | [코드] |
| 학습 외란에 옆 방향 포함 | legged_gym/humanoid-gym 밀기 xy 대칭(위). Atalante: 구면 방향 800 N·0.4 s(IV-A4). Ferigo 2021: 구면 200 N·0.2 s. van Marum 2024: 20–200 N·0.2–0.5 s 전 방향 | — | [코드](legged_gym/humanoid-gym) / [논문] |
| 지속 외력의 x·y 무작위 | FALCON `config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml:56-65`(x·y ±40 N, z −50–5, 영힘 확률 0.25, 지속 150–250 스텝) | 손 끝 외력(사족 아님, 상체 조작용) | [코드] |

**해석(가설)**
- 앞 밀기에 대한 스텝 연속은 매 걸음 지지발이 좌우로 바뀐다. 걸음 폭이 좁아지거나 엇갈리면 옆 캡처 포인트가 지지 밖으로 빠진다. 이것이 표준적인 롤 낙상 기전이다.
- 우리 회복 정책은 (i) 그 구간에 걸음 폭 항이 꺼져 있고(§0-1), (ii) 넓게 딛기는 벌을 받으며(§0-2), (iii) 옆 운동량을 다뤄 본 학습 표본이 거의 없다(§0-4).
- G24R-d 에서 '걸음은 길어지고 횟수는 줄어든' 것은 앞뒤 대응만 강화되고 옆 대응은 그대로였다는 해석과 맞는다(미검증).

### 2. Q2 — '밀기 해제 뒤 낙상'과 해제 학습 방식

- 공개 코드에서 해제 시점을 직접 다룬 예
  - **FALCON**: 외력 크기를 삼각파 위상 |mod(t, 2) − 1| 로 매끄럽게 올렸다 내린다(`humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py:138-144`). 지속 시간 150–250 스텝 무작위, 영힘 확률 0.25(위 yaml). 계단 해제가 아니라 **점진 해제 + 무작위 길이**다 [코드].
  - **legged_gym / humanoid-gym / Isaac Lab push_robot**: 순간 속도 변경(충격)이라 '해제'가 따로 없다 [코드].
  - **Atalante**: 종 모양(bell-shaped) 400 ms 펄스, 3 s ± 2 s 간격 — 연속 외란에 회복해야 한다(IV-A4) [논문].
- '지속 외력 해제 뒤 반동 낙상'을 명시적으로 다룬 논문·코드는 찾지 못했다(미검증).
- 우리 데이터 해석
  - 해제는 학습 분포 안에 있다(§0-5). 그러나 **평가 상승 0.5 s 는 학습(1–4 s)보다 빠르다**. 빠른 상승은 정책이 밀기에 맞서 기울인 자세를 짧은 시간에 크게 만든다. 해제 순간의 반동도 그만큼 크다(가설).
  - 해제 후 0.4–0.5 s 는 2–3 걸음이 이어지는 구간이고, 게이트가 아직 열려 있을 가능성이 높다(정착 0.3 s 연속 조건). 즉 §0-1 의 '간격 항 꺼짐' 구간과 겹친다(확인 필요: 낙상 env 의 rec_open 로그).

### 3. Q3 — 먼저 시험할 최소 단일 변경과 근거

**선행 진단(학습 없음, 기존 평가 롤아웃 재사용)** — 넘어진 env 와 산 env 를 비교한다.
- (a) 해제 후 착지마다 좌우 발 간격(몸 기준 옆 간격; 0.22 m 미만·교차 여부)
- (b) 착지 시 롤각·롤 각속도
- (c) 그때 rec_open 값
- (d) hip_a 각도 사용량

(a) 에서 넘어진 env 의 간격이 좁거나 교차하면 1위가 맞다. 간격은 정상인데 (b) 의 롤 운동량이 크면 2위가 맞다.

**1위 — `feet_gap`(feet_too_close, −5)을 회복 게이트 열림 동안에도 켜기(서기 모드에서 state_gated 해제)**
- 근거 [코드]: 옆 걸음 폭을 지키는 유일한 항이 정확히 낙상 구간(밀기 중 ~ 해제 직후, 게이트 열림)에서 꺼져 있다. 공개 구현(humanoid-gym feet_distance 0.2–0.5 m 상시, Atalante 교차 종료)은 걸음 폭 제약을 **회복 중에도** 유지한다.
- 변경 범위: 보상 래퍼 1개(가중·문턱 그대로). 승인 불요(보상 범주).
- 위험: 회복 중 발 배치 자유도가 줄어 앞 낙상이 늘 수 있다(min_gap 0.22 는 서기 기준 — 회복 중 하한으로는 다소 넓을 수 있음, 진단 (a) 의 산 env 분포로 확인). 판정은 sf12/sf15 낙상 중 옆 비율·총 낙상, sb10/b18 회귀, 서기 흔들림.

**2위 — 학습 외란에 옆 램프 밀기 섞기(`disturbance_reset_g22r` 의 lateral 0 고정 해제, 예: 외란 env 의 30 %·크기 × 0.6 — G21a 설정값 그대로)**
- 근거: 옆 운동량 처리 표본이 거의 없다(§0-4). 공개 회복 연구는 모두 전 방향 외란을 쓴다(Atalante·Ferigo·van Marum [논문], legged_gym·humanoid-gym 밀기 xy [코드]).
- 위험: 표본이 분산돼 앞 밀기 학습량이 줄고, G24R-d 처럼 외란을 키우면 에피소드가 짧아진다 → 앞 힘 범위는 G24R-a(R3) 그대로 두고 방향만 섞는다.

**3위(차선) — 평가 상승 0.5 s 를 학습 분포에 넣기(rise_range 하한 1.0 → 0.5 s)**
- 근거: 평가가 학습 분포 밖이다(§0-5). 해제 반동 크기는 상승 속도와 관련 있을 수 있다(가설).
- 다만 낙상이 옆 방향이라는 관찰을 직접 설명하지 않으므로 1·2위 뒤에 둔다.

참고(단일 변경 아님, 기록만): hip_a 외전 벌점(−0.5, 상시)을 회복 게이트 열림 동안 끄는 것은 1위와 같은 방향(넓게 딛기 허용)이다. 1위 결과를 본 뒤 고려한다.

출처: [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [FALCON](https://github.com/LeCAR-Lab/FALCON) · [legged_gym](https://github.com/leggedrobotics/legged_gym) · Atalante arXiv 2203.01148 · arXiv 2603.08619 · Ferigo 2021 arXiv 2104.14534 · van Marum 2024 arXiv 2404.19173 · 우리 `rl_walking/mdp_hang.py`·`mdp_rewards.py`·G24R-a/d env.yaml

## 라운드 14 (2026-10-04 18:00) — 걸음 폭·옆 발 배치 공식, 스텝 후 주저앉음, G24R-e 부분 효과 시 다음 변경

전제(측정 사실, `reports/gen_c5/DX_FALL_MECHANISM_1004.md`)
- C5 앞 밀기 낙상에서 회복 스텝 중 양발 지지 최소 횡간격이 **28 → ~12 cm**(중앙값 11.6–11.7 cm)로 좁아진다. 생존 env 는 ~28 cm 를 유지한다.
- 낙상 env 는 롤 14–17°·골반 횡속도 0.50–0.55 m/s 로 발산하고, 골반이 내려가 대부분 base_too_low(0.45 m)로 끝난다(sf12 28/32, sf15 95/103).
- 생존 env 의 횡속도 최대 중앙값은 0.23(sf12)·0.36(sf15) m/s. 정책 없는 홀드는 sf12 낙상 73.5 %(정책 기여 확인).
- 병행 중: G24R-e(feet_gap 을 회복 게이트 중에도 ON), G24R-f(옆 밀기 30 % 혼합).

표기: [코드] = 이번에 줄을 읽어 확인, [논문/이론] = 원문·고전 이론, [계산].

### 0. 계산 — 측정된 횡속도에 필요한 옆 착지 거리 (LIP 캡처 포인트) [계산]

- 옆 캡처 포인트 오프셋 ξ_y − y_CoM = v_y/ω, ω = √(g/h_c), h_c 0.50–0.55 m 가정.
  - 생존 env(v_y 0.23–0.36): **5.2–8.5 cm**
  - 낙상 env(v_y 0.50–0.55): **11.3–13.0 cm**
- 즉 낙상 env 는 몸이 옆으로 움직이는 쪽으로 스윙발을 CoM 에서 ≥ 11–13 cm 바깥에 놓아야 멈출 수 있었다. 실제로는 발이 안쪽으로 모여 양발 간격이 12 cm 가 됐다(한 발은 CoM 근처나 그 안쪽 착지로 추정).
- **최소 간격 벌점(0.22 m)만으로는 이 '속도 비례 바깥 착지'를 요구하지 못한다.** 22 cm 를 지켜도 v_y 0.5 에서 CoM 기준 바깥 오프셋은 기하적으로 ~11 cm 남짓(간격의 절반)이라 경계선이다(가설).

### 1. Q1 — 최소 간격 벌점 이외의 걸음 폭·옆 발 배치 공식

| 공식 | 출처 · 위치 | 식 | 가중·수치 | 확인 |
|---|---|---|---|---|
| **간격 밴드(하한+상한) 보상** | humanoid-gym `humanoid/envs/custom/humanoid_env.py:282-292` | d = ‖p_L − p_R‖_xy(전체 거리), r = ½[e^{−100·|min(d − 0.20, 0)|} + e^{−100·max(d − 0.50, 0)}]; 무릎 간격 같은 형(상한 0.25, `:295-`) | 0.2(feet_distance)·0.2(knee_distance), min 0.20·max 0.50(`humanoid_config.py:176-177,196-197`) | [코드] |
| **헤딩 프레임 옆 간격 하한(클립)** | booster_gym `envs/t1.py:719-725` | d_lat = |cos ψ·Δy − sin ψ·Δx|, penalty = clip(0.20 − d_lat, 0, 0.1) | −1.0, ref 0.2(`envs/T1.yaml:277,291`). 발 요 차·평균도 −1(`:274-275`) | [코드] |
| 우리 feet_too_close | `rl_walking/mdp_rewards.py:47-59` | (0.22 − d_lat)₊ | −5, state_gated(G24R-e 에서 게이트 중에도 ON) | [코드] |
| 우리 stance_shape(밴드 + 엇갈림) | `rl_walking/mdp_hang.py:700-713` | (0.22 − d_lat)₊ + (d_lat − 0.32)₊ + (d_lon − 0.04)₊ | −5, 서기 게이트 + state_gated | [코드] |
| **몸 기준 공칭 발 위치(옆 포함)** | walk-these-ways `go1_gym/envs/rewards/corl_rewards.py:161-200` | 몸 프레임 발 xy 목표 = (±stance_length/2, ±stance_width/2) + 위상·명령 오프셋, 오차 제곱합 | −10, 폭 0.3 m(명령으로 가변) | [코드] (사족) |
| 발 간격 상한 위반 | HoST `legged_gym/envs/base/host_ground.py:1181-1186` | 반환 = (‖Δp‖ > 0.9) (주석상 tolerance 형이 있으나 실제 반환은 이진 상한) | style −10(`envs/g1/g1_config_ground.py:241`) | [코드] (일어서기 과제) |
| **발 교차 = 종료** | Atalante Sec. IV-C2: 두 발 볼록껍질 거리 > 0.02 m 위반 시 종료 | 하드 제약 | — | [논문] |
| **발 상대 자세 보상** | Atalante IV-D: 평균 발 프레임에서 오른발–왼발 상대 위치·방향 오차 | RBF 정규화 | — | [논문] |
| **CP–지지 볼록껍질 부호거리** | arXiv 2603.08619: r_CP = exp(−d_CP²/σ²), ξ = p_c + ṗ_c/ω, critic 에 CoM·CP | 두 축 공통 | — | [논문] |
| **옆 캡처 스텝(LIP)** | 고전 이론(Pratt 2006 Capture Point; Englsberger DCM). RL 코드로 옆 CP 착지 목표를 구현한 공개 예는 이번 조사에서 찾지 못함 | 스윙발 착지 y 목표 = ξ_y + s·b(s = 스윙발 쪽 부호, b = 바깥 여유 ≈ 공칭 반폭) | — | [이론] |

**우리에게 적용 — 최소 간격 벌점(G24R-e)을 넘는 형태 두 가지**
- (A) **밴드형(하한 + 상한)**: humanoid-gym 처럼 0.22–0.40 m 밴드에서 바깥으로 벗어나면 연속 감점. 넓게 딛기를 막지 않으면서 좁힘만 막는다.
  우리 joint_deviation_hip(hip_a 외전 L1 −0.5, 상시)이 넓게 딛기를 벌하는 문제(라운드 13 §0-2)와 같이 봐야 한다.
- (B) **속도 비례 옆 착지 목표(LIP)**: 회복 게이트 열림 동안 스윙발 착지(첫 접촉 틱)에서 r = exp(−(y_land − y_target)²/σ²).
  y_target = y_CoM + v_y/ω + s·b(몸 헤딩 프레임 옆축, b ≈ 0.12–0.14 m = 공칭 반폭, σ ≈ 0.04 m). §0 계산상 낙상 env 에서 '안쪽 착지'를 직접 벌하고 바깥 착지에 양(+)의 기울기를 준다.
  필요한 v_y·CoM 은 critic/보상 쪽 특권 정보로 계산한다(actor 계약 불변).

### 2. Q2 — 스텝 뒤 주저앉음(골반 하강) 방지 항과 '게이트 열림 → 자세 항 OFF' 와의 양립

| 항 | 출처 · 위치 | 식 | 게이트와의 양립 |
|---|---|---|---|
| 골반 높이 제곱 오차(양방향) | unitree G1 `legged_gym/envs/base/legged_robot.py:649-652`, target 0.78·−10(`envs/g1/g1_config.py:70,78`); 우리 base_height_l2 −2(target 0.55, 상시) | (h − h*)² | 양방향이라 회복 중 무릎을 굽혀 낮추는 것도 벌함 → 게이트 열림과 상충 |
| 지지발 기준 높이 보상 | humanoid-gym `humanoid_env.py:374-` base_height(접지 발 평균 높이 대비), target 0.89·+0.2(`humanoid_config.py:175,209`) | exp 형 | 같은 상충 |
| 서기 높이 벌점(서기에서 2배) | FALCON `decoupled_locomotion_stand_height_waist_wbc_ma.py:428-430`(서기 env 에서 높이 벌점 배율) | (h − h_cmd)² | 모드(명령)로 게이트 — 회복 중 서기 플래그면 그대로 걸림 |
| 무릎 굽힘 밴드(우리) | `mdp_stage4.py:216-245`, G21a 밴드 0.6–1.05 rad, +4, 서기 게이트 + state_gated | 접촉 다리 평균 | 회복 중 꺼짐(라운드 6) |
| 기울기·높이 **소프트 종료**(상시 제약) | CaT `exts/cat_envs/.../solo12/cat_flat_env_cfg.py` base_orientation(limit 0.1·max_p 0.25)·upsidedown 등(라운드 5 §3) | 위반 크기 비례 종료 확률 | 자세 '모양'이 아니라 **위험 한계만** 제약 → 게이트와 독립으로 둘 수 있음 |
| 골반 자세·높이 **종료 경계** | Atalante IV-C1: z_b > 0.3, −0.4 < ψ < 0.4, −0.25 < θ < 0.7 | 하드 종료 | 같음(경계만) |

**양립 원칙(설계 제안)**: 회복 게이트가 열려도 켜 둘 항은 '자세 모양'이 아니라 **위험 한계 쪽 한방향 항**이다.
- (i) 골반 높이 하한 벌점 (h_floor − h)₊, h_floor ≈ 0.52 m. 정상 크라우치 ≈ 0.597–0.60 m(메모리)보다 충분히 낮고 종료 0.45 m 보다 위.
- (ii) 롤 각 한방향 벌점 (|φ| − 8°)₊.
- (iii) 지지 무릎 과굴곡 한방향 벌점(flex > 1.2 rad).
양방향 자세 항(base_height_l2·knee 밴드·stance_shape)은 지금처럼 게이트로 끄는 것이 회복 스텝과 덜 충돌한다(G15 교훈).
단 측정상 주저앉음은 **롤 발산의 결과**(간격 12 cm → 롤 14–17° → 하강)로 보인다. 높이 항은 원인이 아니라 증상 쪽이라 우선순위는 낮다(가설; 거짓 낙상 비율 대조는 보고서 미확인 항목).

### 3. Q3 — G24R-e 가 부분 효과일 때 다음 단일 변경 순위

1. **속도 비례 옆 착지 목표(§1 (B), LIP 캡처 스텝, 회복 게이트 열림 동안만)**
   - 근거: 측정 횡속도 0.5 m/s 에 필요한 바깥 착지 ≈ 11–13 cm(§0). 최소 간격 0.22 m 벌점은 '좁히지 마라'만 말하고 '얼마나 바깥으로'는 말하지 않는다.
   - 공개 RL 코드 예는 없고(이론 근거), 가장 가까운 코드 형태는 walk-these-ways 몸 기준 발 목표(`corl_rewards.py:161-200`)와 2603.08619 CP 보상(논문)이다.
   - 위험: CoM·v_y 추정은 시뮬 특권이라 보상에만 쓰므로 actor 계약은 불변. σ 가 작으면 착지 시점 희소 보상이 된다.
2. **hip_a 외전 벌점(joint_deviation_hip 중 hip_a, −0.5 상시)을 회복 게이트 열림 동안 OFF** — 넓게 딛는 수단을 벌하는 항(라운드 13 §0-2). 단일 래퍼 변경이고 1위보다 단순하다. 1위 전에 먼저 해도 된다.
   판별: G24R-e 낙상 env 의 hip_a 사용량(외전각)이 생존 env 보다 작으면 이 항이 원인 후보다.
3. **밴드형 간격(§1 (A))으로 교체**: G24R-e 의 하한 벌점에 상한(0.40 m)을 더해 연속 보상으로. 효과는 G24R-e 와 비슷할 것으로 보여 3위.
- G24R-f(옆 밀기 30 %) 결과와 겹치는 효과(옆 운동량 표본 증가)가 있으므로, e·f 두 결과를 함께 보고 1위와 2위 중 하나를 고른다.

출처: [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [booster_gym](https://github.com/BoosterRobotics/booster_gym) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [HoST](https://github.com/InternRobotics/HoST) · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) · [FALCON](https://github.com/LeCAR-Lab/FALCON) · [CaT](https://github.com/Gepetto/constraints-as-terminations) · Atalante arXiv 2203.01148 · arXiv 2603.08619 · Pratt et al. 2006(Capture Point, 이론) · 우리 `mdp_rewards.py`·`mdp_hang.py`·`mdp_stage4.py`

## 라운드 15 (2026-10-04 19:00) — 앞뒤 밀기 회복 스텝이 롤 진동을 부르는 이유와 롤 회복 전용 처방

전제(측정 사실, 코디네이터 전달)
- 순서 진단: 발 횡간격 축소는 **롤 > 8° 이후**에 온다(원인이 아니라 결과 쪽). 라운드 14 의 걸음 폭 가설은 약해졌다.
- 롤 > 8° 이탈은 앞 밀기 유지 1 s 동안(≈ 3.0–3.4 s) **생존·낙상 env 모두**에서 흔하다. 낙상 env 만 회복하지 못하고 롤 14–17°·횡속도 0.5 m/s → 골반 하강으로 간다.
- 옆 밀기는 없다. 앞 밀기에 대해 정책과 무관하게 Δroll −2.6 – −4.3° 의 계통 편향이 있다(플랜트 비대칭 의심).
- 발목 롤 = AK45(속도 1/8, 반사관성 13배), 힙 외전 = AK70.
- 라운드 14 측정: 낙상 env 한 발 지지 0.31–0.36 s vs sf12 생존 0.12 s(sf15 생존 0.46 s).

표기: [코드] = 이번에 읽어 확인, [계산], [논문], [가설].

### 0. 우리 자산·설정 점검 [코드·계산]

**(a) 발의 옆 CoP 권한은 기하가 제한한다 — 발목 롤 모터가 아니다.**
- 발 충돌체 = 발목 롤 링크 메시(convex hull, 좌우 동일 STL: x −0.0449 – +0.0451 m, 바닥 y −0.052) + 패드 4개(중심 x ±0.025, 패드 반폭 0.013 m, 바닥이 메시보다 1.6 mm 아래)이다(`urdf/final3_urdf/meshes/*_ankle_r_link.STL` 정점 계산, `rl_walking/assets/biped12_final3.urdf` RA30P 원점).
- 옆 CoP 범위는 ±3.8 cm(패드) – ±4.5 cm(밑창)이다. 한 발 지지에서 발목 롤로 낼 수 있는 최대 롤 모멘트 ≈ m·g·0.038–0.045 = **6.6–7.8 N·m** 다.
- 발목 롤 모터 한계는 이보다 크다(시뮬 24 N·m). 그래서 한 발 지지 롤 권한의 상한은 **발 폭**이 정한다.
- 비교: CoM 이 지지발에서 안쪽으로 6 cm 떨어져 있으면 중력 롤 토크 ≈ 10.4 N·m 로 발목만으로는 못 잡는다. 이 경우 **다음 걸음의 옆 배치나 힙 전략**이 필요하다.

**(b) 한 발 지지 시간과 옆 발산 [계산, LIP ω ≈ 4.2 s⁻¹].** CoM–지지발 옆 오프셋 d 에서 t 동안 생기는 횡속도 v = d·ω·sinh(ωt):
- d 3 cm: 0.12 s → 0.07 m/s, 0.33 s → 0.24 m/s
- d 6 cm: 0.12 s → 0.13 m/s, 0.33 s → **0.48 m/s**
- 낙상 env 의 측정치(한 발 지지 0.31–0.36 s, 횡속도 0.50–0.55 m/s)는 'CoM 이 지지발 위로 충분히 옮겨지지 않은 채 긴 한 발 지지'와 정합한다.
- 우리 성공 스텝 서명은 원래 '옆 CoM 이동 거의 없이 마지막 50 ms 하중 전환'이다(라운드 1 전제). 짧은 스텝(0.12 s)에서는 문제가 없지만 한 발 지지가 길어지면 이 전략이 롤 발산을 만든다(가설).
- 다만 sf15 생존 env 는 한 발 지지 0.46 s 로도 살았다 → 시간보다 **그동안의 CoM–지지발 오프셋 d** 가 판별 변수일 것이다(진단 권장: 이륙 순간 d 분포, 낙상 vs 생존).

**(c) 발목 롤 액추에이터 모델 [코드]** — G24R-d env.yaml `actuators.ankle_roll`:
- ImplicitActuator, kp 150·kd 5, effort_limit_sim 24 N·m, **velocity_limit_sim 5.0 rad/s**, armature 0.0236(다른 관절 0.003의 7.9배), 시작 시 armature ×0.5–1.5(`rl_walking/env_cfg.py:195-201`), 관절 마찰 0.1–0.6(`ankle_r_friction`).
- 근거 주석은 `rl_walking/biped12_cfg.py:10,81-82`(AK45-36 정격 4.19·무부하 5.45 rad/s, 로터관성 × 36²).
- 사양 기반으로 대체로 합리적이지만 두 가지가 낙관적이다.
  1. Implicit 은 5 rad/s 까지 **전 토크**를 낸다. 실제 DC 모터는 속도에 따라 토크가 선형 감소한다. Isaac Lab `DCMotor`(`actuators/actuator_pd.py:299-306`: τ_max = τ_sat·(1 − ω/ω_max))가 이를 모델한다.
  2. '반사관성 13배'(메모리)와 시뮬 armature 비 7.9배가 다르다(어느 기준의 13배인지 미확인).

**(d) 학습·평가 안의 롤 토크 램프 [코드] — 확인 필요 항목**
- `hang_support_reset` 이 G24R 계보에서 **준정적 롤 토크 램프**를 env 마다 뽑는다(G24R-d env.yaml: rate 0.05–0.6 N·m/s, cap 2–12 N·m, 시작 1–8 s, 부호 랜덤 — `mdp_hang.py:550-557` 의 ramp 샘플).
- 이 토크는 지지비 r 과 무관하게 매 틱 base 에 더해진다(`mdp_hang.py:163-167,180-182`).
- `rl_walking/scripts/eval_ground_recovery.py:154-160` 은 r0 만 검사하고 램프는 끄지 않는다. C5 평가(dx_c5b) 하네스가 이 램프를 끄는지 확인해야 한다.
  t ≈ 3.5–4 s 에서 최대 ≈ 1.5–1.8 N·m 는 위 (a) 의 한 발 지지 롤 권한(6.6–7.8 N·m)의 약 1/4 이다. 부호가 랜덤이라 계통 Δroll 을 설명하지는 못하지만, 롤 이탈의 일부일 수 있다.

**(e) 좌우 비대칭 점검 [코드]**
- 관절 원점·축은 좌우 대칭이다(차이 ≤ 1 mm; hip_a 축은 양쪽 (0,0,−1) 같은 부호인 규약). 발 메시도 동일하다.
- 발목 롤 링크 inertial 은 좌우가 **미러되지 않고 동일**하다(COM x −0.0119 양쪽, ixy 같은 부호, `biped12_final3.urdf` left/right_ankle_r_joint 링크). 결과적으로 두 발 COM 이 같은 쪽으로 1.2 cm 치우친다. 전신 CoM 영향은 ≈ 1 mm 로 작다 → 계통 Δroll(−2.6 – −4.3°)의 주원인으로 보기 어렵다.
- 골반 관성 곱 iyz = 0.0069(요–롤 결합, 주축 ≈ 28° 기울기)는 대칭 몸체에서도 생기는 값이다. 요 가속에 의한 롤 토크는 수십 mN·m 수준이라 무시 가능하다(계산).
- **계통 편향의 원인은 이번 점검(질량·기하)으로는 찾지 못했다.** 남은 후보: 접촉 해석 비대칭(PhysX 접촉 순서), DR 표본 편향(예: env 별 마찰·지연이 seed 공유), 외력 적용 프레임.
  판별: 같은 시나리오를 좌우 미러 초기화(요 180° 회전 스폰)로 돌려 Δroll 부호가 바뀌는지(플랜트·물리) 아니면 유지되는지(정책·관측) 본다.

### 1. Q1 — 앞뒤 밀기 회복 스텝이 롤을 흔드는 이유 (후보 정리)

| 후보 | 근거 | 판정 |
|---|---|---|
| 한 발 지지 중 CoM–지지발 옆 오프셋 × 지속 시간 | §0b: 0.33 s·6 cm → 0.48 m/s(측정 0.5 m/s 와 같은 크기) | **주 기전 후보**(진단: 이륙 시 d) |
| 발 폭이 좁아 발목 롤 권한이 작음 | §0a: 6.6–7.8 N·m 상한 | 구조적 제약(학습으로 못 바꿈) → 걸음 배치·힙 전략이 필수 |
| 발목 롤 AK45 의 느림 | 시뮬 5 rad/s·armature 0.0236. 상한은 기하가 먼저 막음 | 보조 요인. 실물은 토크–속도 감소로 더 나쁠 수 있음(§0c) |
| 힙 외전 권한·사용 억제 | hip_a·hip_r L1 −0.5 상시(라운드 13 §0-2) | 회복 수단(옆 배치·힙 전략)을 벌함 → **처방 후보** |
| 비대칭 플랜트 | §0e: 질량·기하는 대칭(≤ 1 mm) | 이번 점검으로는 미지지. 미러 스폰 대조 필요 |
| 학습·평가의 롤 토크 램프 | §0d | 확인 필요(평가에서 켜져 있으면 낙상 일부를 설명) |

### 2. Q2 — 롤 회복을 겨냥한 처방 (관측·행동 계약 유지)

| 처방 | 식 | 공개 근거 | 확인 |
|---|---|---|---|
| (a) 게이트 열림 동안 힙 외전 자유화 | 기존 joint_deviation_hip 의 hip_a 성분에 × (1 − rec_open) | Isaac Lab G1 hip_roll·yaw L1 −0.1(`config/g1/rough_env_cfg.py:57-61`), humanoid-gym 요·롤 0.1 rad 데드밴드(`humanoid_env.py:362-372`) — 우리 −0.5 보다 약함 | [코드] |
| (b) 게이트 열림 동안 옆 착지 목표(LIP) | 착지 틱: r = exp(−(y_land − (y_CoM + v_y/ω + s·b))²/σ²), b ≈ 0.12–0.14 m, σ ≈ 0.04 m | 공개 RL 코드 없음(이론); 가장 가까운 형태 walk-these-ways 발 목표(`corl_rewards.py:161-200`), Atalante·2603.08619 CP 보상 | [코드]/[논문] |
| (c) 게이트 열림 동안 롤각속도 벌점 | −w·ω_x²·rec_open (우리 ang_vel_xy −0.1 은 상시) | legged_gym/Isaac Lab ang_vel_xy_l2 −0.05 표준; HoST style_ang_vel_xy = exp(−2·‖ω_xy‖²), +1(`host_ground.py:1188-`, `g1_config_ground.py:242`) | [코드] |
| (d) 발목 롤 액추에이터 사실성 | Implicit → DCMotor: τ_max(ω) = τ_sat·(1 − ω/ω_max), ω_max 5.45, τ_sat 24 | Isaac Lab `actuator_pd.py:299-306`, `actuator_pd_cfg.py:42-49` | [코드] |
| (e) 롤 외란 학습 | 옆 밀기(G24R-f 진행 중) 또는 롤 토크 램프(이미 존재, §0d) | Atalante 구면 외란 등(라운드 13) | [논문] |

- (c) 는 G15 교훈(정숙·감쇠 항이 회복 스텝을 억제)과 같은 위험이 있다. 롤을 바로잡는 동작 자체가 롤 각속도를 만든다. 감쇠보다 수단 해방(a)이나 목표 제시(b)가 우리 이력과 맞는다.
- (d) 는 문제 해결이 아니라 실물 정합 쪽이다. 정책이 실물보다 빠른 발목 롤을 믿고 있다면 실물 롤 회복은 시뮬보다 나쁘다. 배포 전 별도 평가 축으로 권장한다.

### 3. Q3 — 상위 2 단일 변수 분기

**선행(학습 없음)**
- (i) 평가 하네스의 롤 토크 램프 ON/OFF 확인(§0d). 켜져 있으면 OFF 로 같은 평가를 다시 해서 낙상 중 램프 기여를 분리한다.
- (ii) 이륙 순간 CoM–지지발 옆 오프셋 d 와 hip_a 사용량을 낙상 vs 생존으로 비교한다.
- (iii) 요 180° 스폰 미러 대조로 계통 Δroll 의 출처를 판별한다.

1. **분기 1 — 회복 게이트 열림 동안 hip_a 외전 벌점 OFF(단일 래퍼)**
   - 근거: 낙상 env 는 롤 이탈(생존 env 와 공통)을 **회복하지 못하는** 쪽이고, 회복 수단은 옆 배치와 힙 전략이다. 둘 다 hip_a 를 쓰는데 그것을 상시 벌하고 있다(−0.5, 공개 구현의 5배 이상).
   - 발 폭 제약(§0a)상 발목만으로는 롤을 못 잡으므로 힙이 필수 수단이다.
   - 위험: 서기 정숙은 게이트 닫힘에서 그대로라 영향이 작다. 회복 중 다리 벌어짐 과다 → 종료·관절 한계를 판정에 포함.
2. **분기 2 — 회복 게이트 열림 동안 LIP 옆 착지 목표(§2 b)**
   - 근거: 롤 이탈 후 회복은 '다음 걸음을 v_y/ω 만큼 바깥에'로 정량화된다(라운드 14 §0: v_y 0.5 → 11–13 cm). 목표를 직접 준다.
   - 위험: 착지 순간 희소 보상, 시뮬 특권 CoM 사용(보상 전용이라 계약 불변). 분기 1 로 수단이 풀린 뒤에 더 잘 작동할 것이다(가설) → 순서상 2위.

출처: Isaac Lab 설치본 `actuators/actuator_pd.py`·`actuator_pd_cfg.py`·G1 rough_env_cfg · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [HoST](https://github.com/InternRobotics/HoST) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · Atalante arXiv 2203.01148 · arXiv 2603.08619 · 우리 URDF/STL·`rl_walking/biped12_cfg.py`·`env_cfg.py`·`mdp_hang.py`·`scripts/eval_ground_recovery.py`·G24R-d env.yaml

## 라운드 16 (2026-10-04 20:00) — 관절 영점 편향(보이지 않는 힙 피치 편향)에 대한 sim2real 처리

전제(측정 사실, DX3 인과 개입, 코디네이터 전달)
- 정책이 관측할 수 없는 **공통 힙 피치 영점 편향 ±1.5°** 만으로 앞 밀기 회복 낙상이 단조적으로 2–3배 바뀐다. 신전 편향은 sf12 27 %(기준 10 %), 굴곡 편향은 4 %.
- 학습 DR 은 이미 신전 쪽 공통 힙 편향 1–3° + 독립 ±2° 를 포함한다.
- 라운드 15 의 롤 토크 램프 지적은 코드로 확인됐고, DX4(램프 ON/OFF)가 진행 중이다.

표기: [코드] = 이번에 읽어 확인, [논문], [계산/추론], 미검증.

### 0. 우리 DR 의 실제 구조 [코드]

G24R-d env.yaml(`logs/rsl_rl/biped12_stepb0/2026-10-04_16-06-52_g24rd_seed1_from_g24ra21793/params/env.yaml`):
- `default_pose_bias`(`rl_walking/mdp_hang.py:1076-`, 기본자세를 바꾸고 `default_joint_pos` 에 써서 `:1132-1133`, 관절 상태도 다시 씀 `:1144`)
  - 다리별: hip_f ±0.035 rad(±2°)·knee ±1°·ankle_f ±0.57°
  - **공통: hip_f −0.052 – −0.018 rad(−3.0 – −1.0°)**·knee −0.236 – −0.183 rad(−13.5 – −10.5°, 무릎 하중 처짐 모델)
  - 관측 `encoder_joint_pos_rel_v2` 는 **편향된 default 를 빼므로**(`mdp_hang.py:1256-1270`) 평형에서 관측 0 → 정책은 편향을 직접 보지 못한다. DX3 의 '관측 0' 과 같은 구조다.
- `calib_bias`(`rl_walking/mdp_stage4.py:396-430`): 관절 ±1°(무릎·발목 별도 ±1°), IMU 피치·롤 ±1.5°, IMU 지연 0–3틱.
- 발목 앞뒤 유격: play 4° × 0.75–1.25(라운드 4 §2). 엔코더는 로터를 보므로 유격 안 이동은 관측에 없다.

### 1. Q1 — sim2real 연구의 관절 영점 오차 처리 방식과 우리 계약과의 양립

| 방식 | 출처 · 위치 | 내용 | 320차원·5틱 계약과 양립 |
|---|---|---|---|
| **영점 오프셋 DR(관측 안 함)** | walk-these-ways `scripts/train.py:63-65`(randomize_motor_offset, **±0.02 rad = ±1.15°**, priv_observe_motor_offset False), PD 오차에 더함 `go1_gym/envs/base/legged_robot.py:931,941`, 기본 범위 ±0.05 `legged_robot_config.py:344`; HugWBC `legged_gym/envs/h1/h1_config.py:279-280`(±0.02), `h1.py:554-556,835`(τ = kp·(a + q0 − q + offset) − kd·q̇) | 정책이 모르는 상수 오프셋에 강건하게 | **양립**(우리 default_pose_bias·calib_bias 가 같은 계열, 범위는 우리가 더 넓음) [코드] |
| 특권 관측으로 오프셋 추정(교사–학생/동시 추정) | walk-these-ways 동시 추정기(adaptation module: 관측 이력 → 특권 값, `go1_gym_learn/ppo_cse/actor_critic.py:14,39-`)에 motor_offset 을 특권 항목으로 넣는 플래그가 있으나(`legged_robot.py:444-452`) 공개 학습 설정에서는 **꺼짐** | 이력에서 추정한 잠재값을 actor 입력에 추가 | actor 입력 추가 → **계약 변경**(불가) [코드] |
| RMA 류 이력 잠재 적응 | Kumar 2021(RMA): 약 0.5 s 이력 → 환경 잠재. 대상은 질량·마찰·모터 세기 등. 영점 오프셋을 다룬 실험은 확인 못 함 | 별도 적응 모듈 | 계약 변경 [논문, 영점 대상은 미검증] |
| 이력 기반 암묵 추정(MLP 입력 이력) | HIMLoco(라운드 0: 이력 인코더·대조 학습); 우리 5틱 이력 MLP | 별도 모듈 없이 이력에서 | 양립(지금 구조). 단 식별 가능성은 §2 |
| **보정 절차(측정 후 영점 갱신)** | Contact-Constrained Lower-Limb Joint-Offset Calibration(arXiv 2609.02306): 엔코더 + 골반 IMU 만으로 정적 양발 지지 4자세(스쿼트·좌우 런지·넓은 스쿼트)에서 발 간 일관성 + 평지 제약 + 무릎 사전값. 시뮬 주입 복원 최대 오차 **0.27°**. 실기 발 높이 RMS 4.26→2.20 mm(A3)·8.03→1.43 mm(A2). "**1° 미만 관절 오프셋도 발바닥에서 cm 급 오차**". 3° 피치 오프셋 주입 시 폐루프 궤적 편차가 보정으로 크게 감소. 코드 공개 언급 없음 | 배포 전 측정 → 영점·트림 | **정책 계약과 무관**(배포 체인 쪽). 실물 측정 필요 [논문] |
| 기계 한계 기준 영점 | Berkeley Humanoid Lite 저수준(`HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel`): 모터축 엔코더만 있어 **전원마다** 관절을 손으로 기계 한계까지 움직여 영점 저장(calibration.yaml). 정확도 수치는 문서에서 확인 못 함 | 지그 없이 하드스톱 | 무관(배포 쪽) [문서 요약, 정확도 미검증] |

### 2. Q2 — 이력 MLP 가 정적 자세 편향을 암묵적으로 추정할 수 있나

- **직접 증거(정적 영점 편향을 이력 MLP 가 추정한다는 실험)는 찾지 못했다**(미검증). RMA·HIMLoco·walk-these-ways 의 이력 기반 추정 근거는 동역학 파라미터(질량·마찰·모터 세기·지형)에 대한 것이다. walk-these-ways 는 오프셋을 추정 항목으로 넣을 수 있게 해 두고도 공개 학습에서는 끄고 DR 강건성으로 처리했다 [코드].
- **우리 관측에서의 식별 가능성 [계산/추론]**
  - 발이 평평할 때 몸통 피치(IMU)는 발목·무릎·힙 피치 합의 함수다. 그래서 '엔코더 관절각 vs IMU 피치'의 불일치는 원리적으로 공통 힙 편향을 드러낸다(한 프레임으로도).
  - 그러나 같은 불일치를 만드는 다른 은닉 항이 더 크다. **발목 앞뒤 유격 ±4°(×0.75–1.25)**, IMU 피치 편향 ±1.5°, 관절 편향 ±1° 다.
  - 따라서 1.5° 공통 힙 편향은 **관측상 구별되지 않는다**(유격만으로도 ±4°). 5틱(100 ms) 이력은 이 모호성을 풀 정보가 거의 없다.
  - → 명시적 적응 모듈이 있어도 같은 관측이면 식별이 어렵다. **강건성(DR 범위·가중) 쪽 처방이 맞다**고 본다.
- 결과적으로 정책은 '편향 분포 전체에서 평균적으로 좋은 행동'을 학습한다. 학습 분포 중심이 신전 −2° 인데 평가 기준 0 에서도 성능이 그 근처라면, 신전 꼬리(−3° 넘음)에 표본이 적은 것이 DX3 의 신전 쪽 급악화와 맞는다(가설; DX3 의 ±1.5° 가 학습 공통 편향 위에 더해진 것인지에 따라 해석이 달라짐 — 확인 필요).

### 3. Q3 — 실무 영점 정확도·절차

- arXiv 2609.02306: 정적 자세 기반 보정으로 시뮬 복원 0.27° 이내, 실기 발 높이 RMS 1.4–2.2 mm [논문].
- walk-these-ways·HugWBC 의 DR ±1.15° 는 '보정 후 남는 오차가 이 정도'라는 실무 가정으로 읽힌다(추론).
- Berkeley Humanoid Lite: 전원마다 하드스톱 보정(정확도 미기재).
- 우리: 로터 절대 엔코더 + 감속비로 인한 배수 시프트(메모리: 전원 사이클 36°/10° 배수), `jetson/scripts/zero_anchor.py` 의 배수 보정 우선 절차(CLAUDE.md). 잔여 정확도의 정량값은 이번 조사 범위에서 확인하지 않았다(파일을 열지 않음 — 로봇 쪽 파일 비접촉 지시).
- 시사점: DX3 민감도(±1.5° 에 낙상 2–3배)는 공개 DR 범위(±1.15°)와 같은 크기다. **실물 영점 오차가 1° 급이면 회복 성능이 그만큼 흔들린다**는 뜻이다. 보정 정확도 확인이 배포 전 필수 항목이다(현장 확인 필요).

### 4. Q4 — 다음 단일 변수 분기 순위

후보 평가
- (A) 회복 정책 DR 을 신전 쪽으로 치우치게: shared hip_f 범위 −3.0 – −1.0° → 예 −4.5 – −1.0°(하한만 확장) — 시뮬 전용, 계약 불변. §2(식별 불가 → 강건성)와 DX3(신전 쪽만 급악화)에 직접 대응.
- (B) DR 대칭 확장(예 ±4°): 굴곡 쪽은 이미 성능이 좋아 표본 낭비. 범위가 넓어지면 명목 성능 저하 위험(메모리: 'DR 범위 커지면 nominal 의심').
- (C) 특권 편향 교사–학생: 학생 입력에 추정값을 넣으면 계약 변경이다. 넣지 않으면 §2 식별성 문제로 이득이 불확실하다. 학습 틀 변경(승인 필요).
- (D) 배포 쪽 트림: 실물 편향을 모르면 방향·크기를 정할 수 없다. 메모리상 '트림은 배포 체인을 못 넘음(DEFAULT_POSE 이동 필요)' 이력이 있고, 영점 변경은 CLAUDE.md 에서 가장 위험한 쓰기다. 실물 측정(2609.02306 식 정적 자세 보정) 뒤의 일이다.

**1위 — (A) 신전 꼬리 확장 DR(공통 hip_f 하한 −3° → −4.5°, 나머지 동일)**
- 근거: 편향은 관측으로 식별되지 않으므로 강건성이 유일한 정책 측 수단이다(§2). 악화가 신전 쪽에만 있어(DX3) 그쪽 꼬리 표본을 늘리는 것이 최소 변경이다.
- 판정: DX3 와 같은 편향 스윕(−3, −1.5, 0, +1.5°)에서 단조 민감도 기울기 감소 + 기준(0°) 낙상 비퇴보 + 서기 흔들림·b18 회귀 없음.

**2위 — (D′) 실물 영점 잔여 오차 측정 절차 설계(정책 변경 없음, 현장 단계)**
- 근거: 실물 편향 크기를 모르면 (A) 의 범위도 추정이다. 2609.02306 의 정적 양발 지지 4자세 + IMU 방식은 추가 장비 없이 엔코더·IMU 로그만으로 공통 피치 편향을 추정한다.
- 측정 결과가 신전 쪽 수 도라면 (A) 의 범위를 그 값으로 맞추고, 트림은 그 다음 승인 절차로 다룬다.
- 주의: 실물 자세 시험이 필요하므로 **실물 작업 승인 대상**이다. 여기서는 설계·로그 분석 방법까지만 권장한다(CLAUDE.md: 실물 시험은 준비 작업에 자동 포함되지 않음).

(C) 는 §2 식별성 근거상 우선순위가 낮다. (B) 는 (A) 가 효과 없을 때의 차선이다.

출처: [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [HugWBC](https://github.com/InternRobotics/HugWBC) · [arXiv 2609.02306](https://arxiv.org/abs/2609.02306) · [Berkeley-Humanoid-Lite-Lowlevel](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel) · RMA(Kumar et al. 2021) · HIMLoco(라운드 0) · 우리 `rl_walking/mdp_hang.py`·`mdp_stage4.py`·G24R-d env.yaml

## 라운드 17 (2026-10-04 21:00) — 이어 학습 성능 요동, 보이지 않는 정적 편향에 대한 강건성

전제(측정 사실, 코디네이터 전달)
- DX5/DX5f/DX7: 지속 자동 수평 트림은 앞 회복만 해친다(뒤 밀기 상충 설명 기각). 인계 전 홀드를 3 s → 6 s 로 늘리면 sf15 가 31 → 42 % 로 악화. G24R-a 단독도 숨은 힙 신전 편향에 매우 민감하다(sf12 6.9 → 26.8 %).
- G24R-g(공통 힙 편향 DR −4.5° 로 확대): +407 반복에서 개선(sf12 10.3 vs 13.5 %), +907 에서 역전(17.2 vs 13.5; 신전 조건 37.9 vs 31.4) → fail-fast 중단.
- G24R-h(회복 중 hip_a/hip_r 편차 벌점 OFF): +907 에서 악화(비유의), 진행 중. G24R-e(feet_gap): 실패.
- G24R-a 자체의 이어 학습도 체크포인트마다 요동 → 체크포인트 간 분산 측정 중.

표기: [측정] = 이번에 학습 로그(tensorboard)를 읽어 계산, [코드], [논문], 미검증.

### 0. 이어 학습 로그에서 확인한 사실 [측정·코드]

로그: G24R-a `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294`, G24R-g `.../2026-10-04_19-24-01_g24rg_seed1_from_g24ra21793`(tensorboard scalars).

1. **재개 직후 학습률이 30–60배로 튄다.**
   - G24R-g 첫 반복들의 Loss/learning_rate = **2.96e-4, 6.67e-4**, 그 뒤 1e-5. G24R-a 는 끝까지 대부분 1e-5 였다.
   - 2.96e-4 = 1e-3/1.5³ 이므로 재개 시 학습률이 **cfg 초기값 1e-3**(agent.yaml `learning_rate: 0.001`, `schedule: adaptive`)에서 시작했다. 즉 `--restore_lr` 없이 재개한 것이다.
   - 근거 코드: rsl_rl 3.1.2 는 load 때 학습률을 복원하지 않는다(우리 `rl_walking/scripts/train.py:43-44` 도움말: "rsl_rl 3.1.2 누락, 5.5.1 ppo.py:405"). 복원은 `train.py:230-232` 옵션으로만 된다.
   - 끝값 대비 30–60배 학습률이 수십 미니배치 동안 걸려 **재개 첫 반복에서 정책이 크게 흔들린다**.
2. **적응 학습률이 바닥(1e-5)에 붙어 있다.** 1e-5 비율: G24R-a 67.6 %, **G24R-g 99.8 %**.
   - rsl_rl 적응 스케줄은 미니배치 KL > 2 × desired_kl(0.02)이면 ÷1.5, < ½ 이면 ×1.5, 범위 [1e-5, 1e-2]다(`algorithms/ppo.py:259-292`).
   - 즉 **최소 학습률에서도 KL 목표를 계속 넘는다**. 갱신 크기를 더 줄일 수단이 없는 상태로 매 반복 '원하는 것보다 큰' 정책 변화가 일어난다.
   - G24R-g 의 Loss/surrogate 중앙값 +0.0007(G24R-a −0.0018)도 같은 방향(클립 영역에서 목적함수 악화)이다. KL 자체는 로그되지 않는다.
3. **탐색 잡음이 이어 학습 중 커진다.** G24R-g Policy/mean_noise_std 0.4892 → 0.4937, Loss/entropy 8.18 → 8.30(entropy_coef 0.008).
   G24R-a 는 0.495 → 0.489 로 줄었다. 넓어진 DR 에서 엔트로피 보너스가 std 를 키우는 방향이다.
4. **가치 함수 오차 급등은 재개 직후 1 반복뿐이다.** Loss/value_function: G24R-g 3.92 → 0.26(2번째 반복). G24R-a 도 4.57 → 0.42. critic 불일치는 빠르게 해소되므로 주원인으로 보기 어렵다.

### 1. Q1 — 좋은 체크포인트에서 DR 을 넓혀 이어 학습하면 악화·요동하는 이유와 처방

| 후보 | 우리 증거 | 처방 | 출처 |
|---|---|---|---|
| 재개 시 학습률 리셋 | §0-1: 1e-3 에서 재개(끝값 1e-5) | **`--restore_lr`**(우리 train.py 에 이미 있음) 또는 이어 학습 cfg 학습률을 1e-5 로 | rsl_rl 최신은 load 시 학습률 복원(OSS_REFERENCES: rsl_rl 857de61 `algorithms/ppo.py:405`) [코드] |
| KL 목표 상시 초과(바닥 학습률) | §0-2: 99.8 % 바닥 | (a) desired_kl 상향(0.01 → 0.02, 적응이 다시 작동하는 영역으로) 또는 (b) KL 조기 종료(목표 KL 의 1.5배 넘으면 그 반복의 남은 epoch 중단 — Stable-Baselines3 PPO `target_kl` 방식; 이번에 줄 확인 안 함) 또는 (c) epoch 수 5 → 2 | SB3 문서/구현(미검증 줄), rsl_rl 은 적응 학습률만 |
| 엔트로피 보너스로 std 증가 | §0-3 | 이어 학습에서 entropy_coef 0.008 → 0.002–0.004 또는 std 고정/상한 | rsl_rl 최신 `modules/distribution.py` std 범위·고정(OSS_REFERENCES §1) [코드, 최신판] |
| critic 불일치 | §0-4: 1 반복 내 해소 | (필요 시) critic 우선 갱신 수 반복(RPL burn-in, 라운드 2) | [논문] |
| 망각(분포 이동) | G24R-g 가 +407 개선 → +907 역전 | 넓힌 범위 안에서도 기존 분포 비중 유지(혼합 샘플링), 짧은 이어 학습 + 체크포인트 선택 | 일반 원칙 |
| 체크포인트 간 고유 분산 | 코디네이터 측정 중 | **판정 문턱을 분산 기반으로**(아래 Q3) | — |
| 가중치 평균(EMA·체크포인트 평균) | — | 다리 RL 공개 코드베이스(legged_gym·Isaac Lab·rsl_rl·humanoid-gym)에서 정책 EMA/체크포인트 평균을 쓰는 예는 이번 조사에서 찾지 못함(미검증) | — |

**해석**: G24R-g 의 '초반 개선 → 후반 역전'은 넓힌 DR 자체의 효과일 수도 있지만, (i) 재개 학습률 급등, (ii) KL 목표 상시 초과, (iii) std 증가가 모든 이어 학습 실험(g·h 포함)에 공통으로 섞여 있다. DR 변경의 효과를 깨끗하게 보려면 이 학습 위생을 먼저 고정해야 한다.

### 2. Q2 — 관측 변경 없이 보이지 않는 정적 편향에 대한 민감도를 줄인 접근

| 접근 | 메커니즘 | 근거 | 우리 계약 |
|---|---|---|---|
| **critic 전용 특권 편향(비대칭 actor-critic)** | critic 입력에 공통 힙 편향·관절 편향을 추가. 가치 추정이 편향별로 갈려 어드밴티지 분산·GAE 부트스트랩 편향이 줄어듦(라운드 8 §2 #3–4 와 같은 논리) | 특권 critic 은 humanoid-gym(`humanoid_env.py:219-235` 마찰·질량·밀기 등), FALCON(critic 에 외력), HIMLoco 에서 표준 [코드]. '숨은 정적 편향에 대한 민감도 감소'를 직접 보인 실험은 미발견 | actor 불변, critic 입력 변경 = 체크포인트 수술(라운드 8 §1: critic.0.weight 0 열 확장 + Adam 상태 확장 + `symmetry.py` critic 규칙). 배포 무관 |
| **실패 비례 편향 샘플링(적응 DR)** | 편향 값 구간(bin)별 실패율에 비례해 다음 에피소드 편향을 뽑음. 넓게 균등으로 펴지 않고 어려운 꼬리에 표본 집중 | BeyondMimic `commands.py:207-241` 실패 bin 비례 샘플링(시간축) [코드]; OpenAI ADR(Akkaya 2019, 성능 문턱으로 범위 확장) [논문] | env 쪽만 변경 |
| 편향 크기 커리큘럼 | 성공률 문턱 넘으면 범위 확장(Berkeley push 커리큘럼처럼 양방향) | Berkeley `curriculums.py:76-130`(push 양방향) [코드], ADR [논문] | env 쪽만 |
| 적대적 편향 | 낙상을 최대화하는 편향을 적대자가 선택(RARL, Pinto 2017) | [논문] | env 쪽만, 불안정 위험 |

- G24R-g(균등 확대)가 +907 에서 역전한 것은 '넓은 균등 분포 → 명목 성능 저하'(메모리: DR 범위 커지면 nominal 의심)와 맞는다. **실패 비례 샘플링**은 범위를 넓히지 않고 실패가 많은 신전 꼬리에 표본만 몰아 이 상충을 피하는 형태다.

### 3. Q3 — 다음 단일 변수 분기 2개 (비용·판정 규칙)

**판정 규칙(두 분기 공통)**
- 코디네이터의 체크포인트 간 분산 측정으로 기준선(G24R-a 이어 학습, 같은 학습 위생)의 sf12·신전 조건 낙상률 SD 를 구한다.
- 후보는 **연속 2개 이상 체크포인트(예 +400·+900)에서 기준선 평균보다 2 SD 이상 개선**이고, 0° 조건·서기 흔들림·b18 이 1 SD 이상 퇴보하지 않을 때만 채택한다.
- 한 체크포인트만의 개선은 채택하지 않는다(G24R-g 교훈).

**1위 — 이어 학습 위생 고정: `--restore_lr` 만 켜고 G24R-a 를 그대로 이어 학습(DR 변경 없음)**
- 근거 [측정]: 재개 학습률 30–60배 급등(§0-1)이 모든 이어 학습 실험에 섞여 있다. 이것이 'G24R-a 이어 학습 자체의 요동'의 일부를 설명하는지 직접 본다.
- 비용: 코드 변경 0(옵션), 학습 1회(같은 반복 수).
- 판정: 기준선(복원 없이 이어 학습)보다 체크포인트 간 SD 가 줄고 평균이 같거나 나으면, 이후 모든 이어 학습(g·h 재시도 포함)의 기본값으로 삼는다.
- 위험: 학습률이 1e-5 로 고정돼 개선 속도가 느려질 수 있다. 바닥 비율(§0-2)과 KL 상시 초과는 그대로다.
  → 같은 실험에서 Loss/learning_rate 와 (가능하면) KL 을 로그해 desired_kl 조정이 다음 단계로 필요한지 판단한다.

**2위 — 실패 비례 편향 샘플링(공통 힙 편향 bin, 범위는 G24R-a 그대로 −3 – −1°에 신전 꼬리 bin −4.5 – −3° 만 추가, BeyondMimic 식 EMA 실패율 + 균등 10 %)**
- 근거: 편향은 관측으로 식별되지 않는다(라운드 16 §2). 민감도는 신전 쪽에만 있다(DX3·G24R-a). 균등 확대(G24R-g)는 명목을 해쳤다. 실패가 몰리는 꼬리에만 표본을 늘리는 최소 형태다.
- 비용: env 리셋 이벤트 1개 수정(편향 샘플러) + 실패 집계(종료 사유) — 작음. 1위의 학습 위생 위에서 실행.
- 판정: 위 공통 규칙 + 편향 스윕(−4.5, −3, −1.5, 0, +1.5°)의 기울기 감소.
- 대안(3위): critic 전용 특권 편향. 배포 무관이지만 체크포인트 수술·미러 규칙 갱신 비용이 있다(라운드 8). 그리고 "critic 이 편향을 알면 actor 의 강건화 압력이 줄 수 있다"는 상충(라운드 8 §3)이 있어 2위 결과를 본 뒤로 미룬다.

출처: rsl_rl 3.1.2 설치본 `algorithms/ppo.py:259-292` · 우리 `rl_walking/scripts/train.py:43-44,230-232` · G24R-a/g tensorboard · [whole_body_tracking (BeyondMimic)](https://github.com/HybridRobotics/whole_body_tracking) · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [FALCON](https://github.com/LeCAR-Lab/FALCON) · OpenAI ADR(Akkaya et al. 2019) · RARL(Pinto et al. 2017) · Stable-Baselines3 PPO target_kl(문서)

## 라운드 18 (2026-10-05 00:30) — 충격(impulse) 밀기 회복, 이전 기술을 지키는 자동 난이도 커리큘럼

전제(실물 결과 1004 밤, 코디네이터 전달)
- C5(G21a 서기 + G24R-a 회복, 상대 기울기 감지기)가 실물에서 작동했다. 서기 오탐 0, 밀기 → 전환 → 회복 스텝 → 서기 복귀 왕복 3회.
- 운영자 관찰: **느린 지속 밀기(램프형)는 혼자 재균형, 짧고 강한 밀침(충격)은 넘어짐**.
- G24R-a 학습 분포(g22r R3): 앞 램프 10–12 N(상승 1–4 s·유지 1–3 s), 충격 = 힘 ×1.5(앞 15–18 N) × 0.2–0.4 s = 3–7 N·s, 외란 사건 중 충격 확률 0.4.
- 시뮬 충격 곡선 측정 중: 앞 20/30/40/60 N @0.2 s, 40/80 N @0.1 s.
- 사용자 지시: 자동 커리큘럼 승급. 안정되면 난이도를 자동으로 올리되 서기·램프 회복·보행이 퇴보하면 이전 체크포인트에서 분기.
  축: 크기·속도·방향·지속·충격·센서/구동 지연·잡음/편향·힙 영점 오차·보행 중 밀기·복합 DR.

표기: [코드] = 이번에 줄을 읽어 확인, [논문], [계산], [설계] = 이번 제안.

### 0. 크기 환산 [계산] — 질량 17.62 kg

| 외란 | 충격량 J (N·s) | 몸 전체 Δv = J/m (m/s) |
|---|---|---|
| 우리 학습 충격(R3) | 3.0 – 7.2 | 0.17 – 0.41 |
| 측정 예정 20 / 30 / 40 / 60 N × 0.2 s | 4 / 6 / 8 / 12 | 0.23 / 0.34 / 0.45 / 0.68 |
| 측정 예정 40 / 80 N × 0.1 s | 4 / 8 | 0.23 / 0.45 |
| 우리 속도 밀기(G22R 계열) ±0.25 m/s | 4.4 | 0.25 |
| Ferigo 2021(iCub) 1.21 N·s/kg | 21.3 | 1.21 |

- 공개 '속도 차기'(아래 표)는 Δv 를 직접 주므로 로봇 질량과 무관하게 비교할 수 있다. legged_gym 1.0, unitree G1 1.5, Playground 최대 2.0, Berkeley 커리큘럼 최대 3.0 m/s 다.
- 우리 학습 충격(Δv ≤ 0.41 m/s)은 **공개 보행 정책들의 통상 범위보다 2–7배 작다**. 실물 '짧고 강한 밀침'이 이 범위를 넘었을 가능성이 크다(실물 충격량은 미측정 — IMU 로그에서 Δv 추정 가능, 아래 §3).

### 1. Q1 — 충격 밀기 회복의 크기·적용 방식·커리큘럼

| 출처 | 적용 방식 | 크기 | 간격/커리큘럼 | 확인 |
|---|---|---|---|---|
| legged_gym | **루트 선속도 덮어쓰기**(`_push_robots`: root_states[:, 7:9] = U(−v, v)) | max_push_vel_xy **1.0 m/s** | 15 s 마다 | [코드] `legged_gym/envs/base/legged_robot.py:414-419`, `legged_robot_config.py:126-128` |
| unitree_rl_gym G1 | 같음 | **1.5 m/s** | 5 s | [코드] `envs/g1/g1_config.py:33-35` |
| Isaac Lab 속도 env(G1 등) | `push_by_setting_velocity`(루트 속도에 더함) | ±0.5 m/s | 10–15 s | [코드] `velocity_env_cfg.py:222-227`; 함수 `envs/mdp/events.py:1065-1071` |
| Berkeley Humanoid(Isaac Lab) | 같음 + **양방향 커리큘럼** | 시작 ±0.5 → 최대 3.0 m/s | 낙상 < 2×타임아웃이면 ×1.5, 낙상 > 타임아웃/2 이면 −0.2(200 반복 간격, 1500 반복 이후) | [코드] `velocity_env_cfg.py:249-253`, `mdp/curriculums.py:76-130` |
| MuJoCo Playground Berkeley Humanoid | 베이스 qvel[:2] 에 방향 균등·크기 U(0.1, 2.0) 더함 | 0.1 – 2.0 m/s | 5–10 s | [코드] `_src/locomotion/berkeley_humanoid/joystick.py:86-92,281-295` |
| humanoid-gym | 루트 속도 | 0.2 m/s·각속도 0.4 | 4 s | [코드] `humanoid_config.py:154-157` |
| van Marum 2024(Digit) | 힘 펄스 | 20–200 N × 0.2–0.5 s(전 방향) | — | [논문] |
| Ferigo 2021(iCub) | 힘 펄스 | 200 N × 0.2 s(구면 균등, 1.21 N·s/kg) | 평균 5 s | [논문] |
| Atalante | 종 모양 힘 펄스(골반) | 최대 800 N × 0.4 s | 3 ± 2 s, **점진 증가는 조기 수렴 유발** | [논문] |
| FALCON | 손끝 지속 힘(삼각파) | x·y ±40 N | 150–250 스텝 무작위 | [코드](라운드 13) |

**속도 차기 vs 힘 펄스 — 손으로 미는 밀침에 더 충실한 것**
- 루트 속도를 덮어쓰거나 더하면 관절 속도는 그대로라 **모든 링크에 같은 Δv** 가 더해진다(감축 좌표 관절계). 이는 전신 CoM 에 무한히 짧은 순수 선형 충격이고 **각운동량 변화 0, 지속 0** 이다.
- 사람의 밀침은 몸통(골반·상체)에 0.1–0.3 s 동안 작용하는 힘이다. (i) 적용점이 전신 CoM 보다 높아 피치 모멘트를 함께 주고, (ii) 그 0.1–0.3 s 동안 정책·발목·접촉이 반응한다.
- 우리 힘 펄스는 base(골반) 몸체에 힘만 넣는다(positions 없음, `mdp_hang.py:186-189`). 골반 질량중심이 전신 CoM 보다 위에 있으면 피치 모멘트가 생기므로 손 밀침에 더 가깝다(추론; 골반 COM 높이와 전신 CoM 차이는 미계산).
- → **주 수단은 힘 펄스(지속 0.05–0.3 s 무작위)**, 속도 차기는 '순간 충격' 극한 사례로 소수 혼합이 적절하다(설계 판단).

### 2. Q2 — 이전 기술을 지키는 자동 난이도 커리큘럼

| 방식 | 승급/강등 규칙 | 망각 방지 장치 | 확인 |
|---|---|---|---|
| legged_gym 지형 커리큘럼 | env 별: 이동 거리 > 지형 길이/2 → +1, 명령 대비 이동 < 50 % → −1 | **최고 레벨을 푼 env 는 무작위 레벨로 재배치**(쉬운 레벨 재방문) | [코드] `legged_robot.py:433-441`; Isaac Lab 동일 `terrains/terrain_importer.py:314-326`, `velocity/mdp/curriculums.py:49-54` |
| legged_gym 명령 커리큘럼 | 추종 보상 평균 > 가중의 80 % → 선속도 범위 ±0.5 확장 | 범위를 넓혀도 균등 샘플이라 이전 범위가 계속 표본에 포함 | [코드] `legged_robot.py:450-451` |
| walk-these-ways 격자 커리큘럼 | 명령 bin 별 성공(추종 > 문턱 × 가중; 선속도 0.8·요 0.7·접촉 0.9) → 그 bin 과 인접 bin 의 가중 +0.2(상한 1) | **가중은 줄지 않는다** → 한 번 성공한 쉬운 bin 은 계속 표본됨(암묵적 재연) | [코드] `go1_gym/envs/base/curriculum.py:113-154`, `scripts/train.py:23-26` |
| Berkeley push 커리큘럼 | 낙상 < 2×타임아웃 → ×1.5, 낙상 > 타임아웃/2 → −0.2 | 범위 안 균등 샘플(약한 밀기도 계속 나옴) + **강등 경로** | [코드] `curriculums.py:76-130` |
| HoST 보조력 감쇠 | 성공 묶음마다 −20 N(되돌림 없음) | 없음 | [코드](라운드 2) |
| OpenAI ADR | 파라미터별 범위 경계에서 확률적으로 평가. 경계 성능이 상한 문턱 이상이면 범위 확장, 하한 문턱 이하이면 축소 | 대부분 표본은 범위 내부에서 균등 → 쉬운 값이 계속 포함 | [논문] Akkaya et al. 2019(문턱·버퍼 값은 이번에 원문 재확인 안 함) |
| BeyondMimic 적응 샘플링 | bin 별 실패율 EMA(α 0.001) + 균등 10 % | 균등 바닥 10 % 가 쉬운 bin 표본 유지 | [코드] `commands.py:207-241` |

**공통 원리**
1. 승급은 **성공률 문턱(추종 80 %, 낙상 비율 등)**으로 한다.
2. **강등(되돌림) 경로**를 둔다(Berkeley·legged_gym·ADR).
3. 망각 방지는 별도 리플레이 버퍼가 아니라 **샘플링 분포가 쉬운 영역을 계속 포함**하는 방식이다(균등 범위·무작위 재배치·가중 비감소·균등 바닥).

### 3. Q3 — 회복 전문가(C5 회복 정책)를 위한 커리큘럼 설계안 [설계]

**전제**: C5 회복 정책의 역할은 '감지기가 넘겨준 순간부터 넘어지지 않고 서기 정책으로 돌려주는 것'이다. 서기 정숙은 판정 대상이 아니다. 대신 **복귀 조건(회복 게이트 닫힘)까지의 시간**과 **램프 회복(실물에서 이미 됨)의 비퇴보**를 지킨다.

**외란 축과 레벨(한 번에 한 축만 승급 — 원인 귀속을 위해)**

| 순서 | 축 | 레벨(쉬움 → 어려움) |
|---|---|---|
| 1 | 충격량 J(앞, 지속 0.2 s 힘 펄스) | 4 → 6 → 8 → 10 → 12 → 14 → 16 N·s (Δv 0.23 → 0.91 m/s) |
| 2 | 지속 시간(같은 J) | 0.3 → 0.2 → 0.1 → 0.05 s |
| 3 | 방향 | 앞·뒤 → ±30° → ±60° → 전 방향(옆 포함) |
| 4 | 센서·구동 | 지연 DR 상한 +1틱 → +2틱, 관절 영점 신전 꼬리(라운드 16·17) |
| 5 | 타이밍 | 정지 서기 중 → 회복 스텝 도중 두 번째 밀기 → (보행 정책 쪽 별도) |

**사건별 샘플링(망각 방지) — 각 외란 사건마다**
- **50 %: 현재 축의 프런티어 레벨**(다른 축은 숙달 레벨 중 무작위)
- **30 %: 모든 축의 숙달 레벨에서 균등**(walk-these-ways·ADR 식 '쉬운 영역 계속 포함')
- **20 %: G24R-a 원래 분포(R3 램프 10–12 N + 충격 3–7 N·s)** — 실물에서 된 램프 회복을 지키는 고정 앵커
- 속도 차기는 축 1 의 각 레벨에서 같은 Δv 로 10 % 섞는다(순간 충격 극한).

**승급·강등·되돌림 규칙(블록 = 프런티어 사건 ≥ 2 000)**
- **승급**: 프런티어 회복 성공률 ≥ 85 %(성공 = 3 s 안 무낙상 + 회복 게이트 닫힘) **그리고** 숙달·앵커 표본 성공률이 기준선 − 2 SD 이상. SD 는 코디네이터의 체크포인트 간 분산 측정값(라운드 17 판정 규칙과 같은 값).
- **강등**: 프런티어 성공률 < 60 % 가 3 블록 연속이면 한 레벨 내림(Berkeley 식 양방향).
- **되돌림**: 승급 때마다 체크포인트를 저장한다. 별도 회귀 탐침(256 env, 결정론: sf12·sb18 램프 + 직전 숙달 충격 레벨)이 기준선 + 2 SD 보다 나빠지면 직전 승급 체크포인트로 되돌리고, 그 축을 '정체'로 표시한 뒤 다음 축으로 넘어간다(사용자 지시의 '퇴보 시 이전 체크포인트 분기').
- 학습 위생은 라운드 17 1위(`--restore_lr`)를 기본으로 한다.

**로그(판정·진단용)**: 축·레벨별 성공률, 샘플링 엔트로피(BeyondMimic 식), 회복 게이트 닫힘까지 시간, 낙상 방향(앞/옆 — 라운드 13–15 의 옆 낙상 추적).

**실물 연결(현장 확인 필요)**: 실물 밀침의 크기를 모르면 커리큘럼 목표 레벨을 정할 수 없다. 실물 블랙박스 IMU 가속도 적분으로 밀침 직후 Δv 를 추정해 위 레벨표(Δv)와 대조할 것을 권한다(로그 분석만, 실물 추가 시험 불필요). 운영자 관찰('짧고 강한 밀침 = 넘어짐')이 Δv 몇 m/s 였는지가 축 1 의 최종 목표다.

**확인 범위 요약**
- 코드로 확인한 것: legged_gym·unitree·Isaac Lab·Berkeley·Playground·humanoid-gym 의 밀기 방식과 크기, legged_gym/Isaac Lab/walk-these-ways/Berkeley/BeyondMimic 의 커리큘럼 규칙.
- 논문 근거: ADR 의 경계 평가 규칙, van Marum·Ferigo·Atalante 의 힘 펄스 크기.
- 위 커리큘럼 수치(85 %·60 %·50/30/20 %·블록 2 000)는 [설계] 제안이며 검증되지 않았다.

출처: [legged_gym](https://github.com/leggedrobotics/legged_gym) · [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) · Isaac Lab 설치본 · [isaac_berkeley_humanoid](https://github.com/HybridRobotics/isaac_berkeley_humanoid) · [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) · [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) · [HoST](https://github.com/InternRobotics/HoST) · [FALCON](https://github.com/LeCAR-Lab/FALCON) · OpenAI ADR(Akkaya et al. 2019) · van Marum 2024 · Ferigo 2021 · Atalante 2022

## 라운드 19 (2026-10-05 02:00) — 충격 회복의 물리적 천장(LIP 캡처 가능성)과 스텝 가속 구조

전제(측정 사실): `reports/gen_c5/ENVELOPE_IMPULSE_1004.md`, `reports/gen_c5/tourney1/RESULT.md`
- C5 앞 충격 경계 ≈ **6–8 N·s**: 4 N·s 0.3 %, 6 N·s 2.4 %, 8 N·s(0.2 s) 19.5 %, 8 N·s(0.1 s) 48.3 %, **12 N·s 99.4 %**(모든 후보 ~99 %).
- 같은 충격량이면 짧을수록 나쁘다. 충격 커리큘럼(T2)은 8 N·s 대(fi40·fs80)에서 3/3 체크포인트 일관된 소폭 개선(fs80 −5 – −9 %p).
- 패드 고장(왼발 0 고정) 회귀: RV3 에서 회복 모드 4 s 초과 env 293/329, RV4 150/329.

표기: [계산] = LIP 모델 직접 계산(가정 명시), [측정] = 위 보고서, [코드], [논문].

### 0. (a) 12 N·s 는 물리적으로 회복 가능한가 — N-걸음 캡처 가능성 [계산]

**모델·가정**
- 선형 역진자(LIP), CoM 높이 h_c = 0.55 m → ω = √(g/h_c) = 4.22 s⁻¹. 충격은 순간(실제 0.1–0.2 s 동안 작용 — 무시하면 낙관적).
- 지지발 CoP 는 발끝(발목 앞 t = 8.5 cm)에 고정(최선의 발목 전략). 걸음마다 발목을 최대 L 앞에 두고, 충격(또는 직전 착지)부터 다음 착지까지 T 초(감지 + 인계 + 하중 소거 + 공중).
- 골반 위 밀기로 생기는 앞 피치 회전(각운동량)은 무시(낙관적).
- 캡처 가능 범위(Koolen et al. 2012 의 N-걸음 캡처 가능성과 같은 재귀; 발목 기준 캡처 포인트 거리):
  d₀ = t, d_N = t + (d_{N−1} + L − t)·e^{−ωT}.
- 충격 J 의 캡처 포인트: ξ₀ = x_CoM + (J/m)/ω (m = 17.62 kg). CoM 은 발목 위(0) 또는 2 cm 뒤(메모리: 스탠바이 CoM 발목 뒤 1.3–2.5 cm).

**충격량 → 캡처 포인트**: 4 → 5.4 cm, 6 → 8.1, 8 → 10.8, 10 → 13.4, **12 → 16.1**, 16 → 21.5 cm.

**걸음 시간·보폭별 캡처 한계 d₁(1걸음)·d₂(2걸음)·d∞(무한 걸음)와 대응 충격량(CoM 0 / CoM 2 cm 뒤)**

| L | T | d₁ | d₂ | d∞ (cm) | d∞ 대응 J (N·s) |
|---|---|---|---|---|---|
| 15 cm | 0.16 s | 16.1 | 20.0 | 24.0 | 17.9 / 19.4 |
| 15 | 0.20 | 14.9 | 17.7 | 19.8 | 14.7 / 16.2 |
| 15 | 0.25 | 13.7 | 15.5 | 16.5 | 12.3 / 13.8 |
| 15 | **0.30** | 12.7 | 13.9 | 14.4 | 10.7 / 12.2 |
| 15 | 0.35 | 11.9 | 12.7 | 12.9 | 9.6 / 11.1 |
| 15 | **0.45** | **10.7** | 11.1 | 11.1 | 8.3 / 9.8 |
| 20 cm | 0.30 | 14.1 | 15.7 | 16.3 | 12.2 / 13.6 |
| 20 | 0.25 | 15.5 | 17.9 | 19.2 | 14.3 / 15.8 |

**결론**
- **현재 걸음 시간(0.3–0.45 s)·보폭 15 cm 에서 12 N·s(ξ 16.1 cm)는 1–2 걸음으로 캡처 불가**다. d₂ = 11.1–13.9 cm 이고, 무한 걸음 한계(11.1–14.4 cm)도 넘는다. 낙관적 가정(순간 충격·피치 무시·CoP 항상 발끝)인데도 그렇다.
- 측정 경계 6–8 N·s(ξ 8.1–10.8 cm)는 **유효 T ≈ 0.4–0.45 s 의 1걸음 한계 d₁(10.7 cm)**와 같은 크기다 → 현재 회복은 사실상 '느린 1걸음' 체계로 보인다(추론).
- 12 N·s 를 1–2 걸음으로 잡으려면 **T ≤ 0.2–0.25 s**(감지부터 착지까지)가 필요하다. 보폭을 20 cm 로 늘려도 T 0.3 s 에서 d∞ 16.3 cm 로 경계선이다.
  → **지렛대는 보폭이 아니라 걸음 시간(반응 + 소거 + 공중)**이다. 우리 성공 스텝 서명은 소거 0.06 + 공중 0.14–0.16 s = 0.20–0.22 s 이므로(라운드 1 전제), **감지·인계·준비 지연을 0.05 s 이하로 줄이면** 12 N·s 가 이론상 범위 안에 들어온다.
- 10 N·s(ξ 13.4)는 T 0.3 s 의 2걸음(13.9)으로 경계 — 현재 커리큘럼의 현실적 상한이다. 16 N·s 는 T 0.16 s 수준에서만 가능하다.

### 1. (b) 계약 안에서 회복 스텝을 빠르게 하는 구조 아이디어 — 순위

| 순위 | 아이디어 | 계약 | 근거 | 위험 |
|---|---|---|---|---|
| **1** | **감지·인계 지연 단축**(C5 감지기·전환 쪽). 예: 상대 기울기 문턱·지속 조건 완화, 인계 블렌드 단축, 감지기 입력에 가속도계(IMU 가속도)를 추가해 충격을 첫 틱에 검출 | **정책 관측·행동 계약 불변**. 감지기는 러너 소프트웨어라 시뮬 동일 구현 + 사용자 승인 필요. 가속도계를 정책 관측에 넣는 것이 아니다 | §0: T 가 지배 변수. 감지기만 바꾸므로 학습 없이 시뮬 평가 가능 | 오탐 증가(서기 무오탐이 실물 성공의 핵심). 패드 고장 시 회복 모드 고착(RV3 293/329)과 상호작용 |
| 2 | **사건 구동 반사 첫 걸음 primitive**(A0m 식 보조 잔차를 '전환 시각 기준'으로 재생하고 감쇠 — 라운드 9 §1(ii)) | env 쪽 보조, 계약 불변 | A0m 에서 primitive(소거 0.06 + 공중 0.15 s)를 α=0 까지 내재화한 이력(착지 94 %) | 학습 비용 큼(보조 감쇠 전 과정). 앞 착지 거리(10–15 cm)를 primitive 가 직접 만들지 않음(보조 잔차는 굽힘 펄스) |
| 3 | 빠른 스텝 모방 보상(STEP-I 식) | 보상만, 계약 불변 | I0 단독은 실패 이력(이륙 0) — 보상만으로는 탐색이 안 됨 | 같은 실패 반복 위험 |
| 4 | 클록 위상 리셋으로 스윙 대기 제거(C4) | 클록 의미 변경(승인·보류 중) | 회복 스텝이 1.1 Hz 클록 위상에 묶여 있는지 먼저 측정 필요(묶여 있지 않으면 효과 없음) | 라운드 9·10 의 일관성 요건 |
| 5 | 정책 관측에 몸통 가속도 이력 추가 | **계약 변경(관측 차원 증가) — 승인 필요** | 충격을 빨리 '보게' 하지만 1위(감지기 쪽 가속도)로 같은 이득을 계약 변경 없이 얻을 수 있음 | 재학습 필요, 실물 가속도 잡음 |

### 2. 이번 라운드의 세 가지(독립 판단)

**(1) 실물에서 가장 큰 실패 위험**
- **사람 손 밀침이 ≈ 7 N·s(Δv ≈ 0.4 m/s)를 넘으면 거의 결정적으로 넘어진다.** 이것은 학습 부족이 아니라 현재 걸음 시간(유효 ≈ 0.4–0.45 s)의 **물리적 한계 근처**다(§0).
- 실물 시험에서 '콱 밀기'의 세기를 통제하지 않으면 낙상(하드웨어 손상)이 반복된다. 시험 프로토콜에서 밀침 크기 상한(로그 Δv 로 사후 확인)과 줄 대기(회수 준비)가 필요하다.
- 차순위 위험은 **패드 고장 시 회복 모드 고착**(시뮬 RV3 4 s 초과 293/329, RV4 150/329 — 실물 시험 4 재현)이다.

**(2) 정보량이 가장 큰 다음 실험(평가만, 학습 없음)**
- **시간 분해 + 오라클 감지기 반사실**: 6·8·10 N·s(0.1·0.2 s)에서 충격 시작 → 감지 → 인계 완료 → 하중 소거 시작 → 이륙 → 착지의 시각과 보폭을 기록한다. 같은 조건을 **'충격 시작 틱에 즉시 전환'(오라클 감지기)**으로 다시 평가한다.
- **판정 규칙**: 오라클이 8 N·s 낙상을 상대적으로 ≥ 50 % 줄이면(예 19.5 → < 10 %, fs80 48 → < 24 %), 다음 투자는 **감지·인계 지연 단축(§1 1위)**이다. 줄이지 못하면 병목은 정책의 소거·스윙 속도이므로 **반사 첫 걸음 primitive(§1 2위)**로 간다.
  보조 측정: 이륙이 클록 위상에 몰리면(위상 히스토그램) C4 재검토 근거.

**(3) 멈출 것**
- **12 N·s(fi60) 를 학습 목표로 쫓는 것.** §0 상 현재 걸음 시간으로는 이론 천장 밖이다. 커리큘럼 상한은 10 N·s(0.2 s)로 두고, 지연 단축 효과가 확인된 뒤에 다시 연다.
- 균등 DR 확대형 분기(G24R-g 유형)와 발 간격 최소 벌점(G24R-e)의 재시도 — 이미 실패/역전했고 충격 경계와도 무관하다.

출처: LIP·캡처 포인트 — Pratt et al. 2006, Koolen et al. 2012 'Capturability-based analysis …'(N-걸음 캡처 가능성 재귀) [논문, 수식은 직접 유도·계산] · 우리 측정 `reports/gen_c5/ENVELOPE_IMPULSE_1004.md`, `tourney1/RESULT.md` · STEP primitive 서명(라운드 1 전제) · A0m 결과(라운드 9 전제)

## 라운드 20 (2026-10-05 03:30) — 병목은 회복 정책 자체의 반응: 지연 민감도·떨림·첫 걸음 가속

전제(측정 사실, 코디네이터 전달)
- **오라클 감지기**(충격 시작 틱에 회복 정책으로 전환): 개선 없음. 8 N·s 19.5 vs 17.7 %·45.5 vs 46.4 %, 10 N·s 82 vs 82 % → 감지·인계 지연은 병목이 아니다(라운드 19 1위 기각).
- 실물 루프 지연 ≈ 시뮬 지연 2틱(상호상관 측정). 회복 정책의 지연 민감도: sf12 = **5.5 / 15 / 21 / 43 %**(지연 1 / 3 / 4 / 6틱).
- '9 Hz 힙 진동'은 정책 자체의 고주파 출력 떨림(6–12 Hz, 원시 3–4°)이고 시뮬에도 있으며 실물에서 약 2배 크다.
- G25 충격 커리큘럼(판정 수정판): 레벨 0(4–8 N·s) 성공률 ≈ 0.76 에서 정체.

표기: [측정] = 코디네이터 측정·우리 기록, [코드], [논문], [추론].

### 0. 우리 기록에서 가장 직접적인 근거 [측정, 메모리 g16-generation-1001]

- **G16d(학습 지연 0–6틱)**: 짧은 지연으로 평가한 뒤 18 N 밀기 낙상이 **9–24 %**(기준 G12-9500 54 %)로 크게 개선됐다. 대가는 무외란 서기 흔들림 8–22 %·부호반전 0.17–0.31/틱(약 100배)이었다.
  당시 결론은 '게인–지연 교환: 긴 지연 학습은 밀기 회복↑·정지 떨림↑ → 정숙 항 없이 쓸 수 없다'(서기와 회복이 한 정책이었기 때문에 기각).
- **C5 구조에서는 서기 정숙이 G21a 의 일이다.** 회복 정책은 회복 구간에서만 쓰인다. 따라서 G16d 를 기각시킨 대가(서기 떨림)가 회복 전문가에게는 대부분 해당하지 않는다.
  남는 대가는 '회복 모드 중 떨림'과 '서기로 복귀 직전 떨림'인데, 실물에서 이미 2배인 6–12 Hz 떨림(아래 (1))이 그 위험이다.
- 현재 G24R-a 의 학습 지연: 행동 0–3틱 균등, 자이로 지연 0–3, IMU 자세 지연 0–3(G24R-a env.yaml `max_delay_steps 3`, `max_lag 3`, `imu_lag_steps (0,3)`).
  실물 2틱은 학습 범위 안이지만 **민감도 곡선의 가파른 구간**(1 → 3틱에 낙상 3배)에 있다.

### 1. (4) 첫 회복 걸음을 빠르게 할 정책 측 구조 변경 — 순위

| 순위 | 변경 | 근거(우리 수치·OSS·논문) | 비용·위험 |
|---|---|---|---|
| **1** | **회복 전문가 학습 지연 DR 확대(행동 0–5틱, 자이로·IMU 지연 함께)** | (i) G16d: 지연 0–6 학습이 밀기 낙상 54 → 9–24 %(우리 이력). (ii) 측정 민감도: 실물 2틱이 가파른 구간. 지연에 둔감해지려면 '더 일찍·더 크게' 반응하는 정책이 필요하고, 그것이 곧 첫 걸음 가속이다(추론). (iii) OSS 지연 DR: HoST 행동 지연 **0–4 정책 틱**(50 Hz; `legged_gym/envs/base/host_platform.py:244-246,455-458`, `envs/g1/g1_config_ground.py:291` max_delay_timesteps 5 → randint 0–4), humanoid-gym 행동 지연 혼합 최대 0.5(`humanoid_config.py:159`, `humanoid_env.py:193-195`) [코드]; Tan et al. 2018(Minitaur) 지연 DR [논문] | ≤ 500 반복(G24R-a 에서, `--restore_lr`). 위험: 회복 모드 떨림 증가(G16d 의 100배 부호반전) — 실물 떨림 2배와 겹치면 하드웨어 위험 → 판정에 회복 모드 떨림 포함 |
| 2 | **빠른 유효 착지 사건 보상**(충격 후 첫 유효 착지 시각 t_td 에 r = exp(−t_td/τ), τ ≈ 0.25 s; 착지 x 가 CP 기준 이상일 때만 지급) | 지금은 8 N·s 에서 스텝 env 100 % — 걸음은 이미 나므로 **시점 보상에 기울기가 있다**(B0·B1·I0 실패는 걸음이 없던 상황). CP 조건이 없으면 낙상 env 처럼 '짧은 걸음'(라운드 13 측정)으로 수렴할 위험 → CP 하한 조건 필수(라운드 3·14 의 착지 공식) | 보상 1항 + 사건 판정. 위험: 서두른 짧은 걸음, 부정확 착지 |
| 3 | 충격 시작 기준 반사 보조(A0m 식 잔차 재생 후 감쇠) | A0m 은 primitive 를 α=0 까지 내재화(착지 94 %). 단 primitive 는 '굽힘 펄스'(들기)라 앞 착지 거리·타이밍 단축을 직접 주지 않는다 | 비용 큼(보조 감쇠 전 과정 ≥ 수천 반복). 1·2 가 첫 착지 시각을 못 줄일 때만 |
| 4 | 회복 행동 매끄러움(action smoothness) | 첫 걸음을 빠르게 하는 수단이 아니다. 우리 이력: G17 ×3 벌점 → 느린 흔들림, G18j(LCP 블록) → 이득 축소·느린 흔들림. 매끄러움 벌점은 반응을 늦추는 방향 | 떨림(실물 2배) 대책으로는 별도 검토 대상이지 속도 대책이 아님 |

- 1위와 4위는 같은 축(이득·지연·떨림)의 반대 방향이다. 지연 DR 확대는 떨림을 키울 가능성이 높으므로 **1위 실험의 판정에 떨림 상한을 넣는 것**이 핵심이다.

### 2. 이번 라운드의 세 가지(독립 판단)

**(1) 실물에서 가장 큰 실패 위험**
- **실물 루프 지연의 작은 증가 + 실물에서 2배인 6–12 Hz 떨림의 결합.** 회복 정책은 지연 2 → 3–4틱만 돼도 sf12 낙상이 15–21 %(1틱 대비 3–4배)다.
- 실물은 이미 stage8 큐 적체로 200 ms 지연 사고 이력이 있다(메모리: stage8-latency-root-cause-0930). CAN 부하·로그 기록·발열로 지연이 1–2틱 늘 여지가 있다.
- 실물 떨림이 시뮬의 2배라는 것은 실물 플랜트에서 회복 정책의 루프 이득이 이미 여유가 적다는 신호다(추론). 같은 상태에서 지연이 늘면 진동 증폭·낙상으로 이어질 수 있다.
- 대책(실물 작업 아님, 준비): 실물 블랙박스에서 회복 모드 중 지연(상호상관)과 6–12 Hz 대역 출력 전력을 매 시험 자동 산출해 '지연 ≥ 3틱 또는 떨림 ≥ 시뮬 3배'면 회복 시험 중단 기준으로 쓰는 것.

**(2) 정보량이 가장 큰 다음 실험(≤ 500 반복)**
- **G24R-a → 지연 DR 확대(행동 0–5, 자이로·IMU 0–5) 500 반복, `--restore_lr`**, 대조 = 같은 반복 수의 지연 DR 그대로 이어 학습(T1 식 안전 PPO 와 같은 위생).
- 평가: 지연 1·2·3·4틱 × {sf12, fs80(8 N·s 0.1 s), fi40(8 N·s 0.2 s)} + 회복 모드 떨림(6–12 Hz 대역 출력 전력, 부호반전/틱) + 회복 후 서기 복귀 시간.
- **판정 규칙**: 지연 2·3틱 모두에서 fs80·fi40 낙상이 대조 대비 체크포인트 간 2 SD 이상 감소 **그리고** 회복 모드 떨림 전력 ≤ 대조 × 1.5 → 채택(다음은 충격 커리큘럼과 결합).
  낙상은 줄지만 떨림이 1.5배를 넘으면 → 떨림 대책(회복 모드 한정 action-rate)과 묶어 재시험. 낙상이 안 줄면 → 지연 가설 기각, 2위(빠른 착지 사건 보상)로.
- 학습 없이 같이 할 수 있는 측정: 첫 회복 걸음의 시간 분해(충격 → 하중 소거 시작 → 이륙 → 착지)를 지연 1 vs 3틱에서 비교. 지연이 첫 걸음을 늦추는지(가설)를 직접 본다.

**(3) 멈출 것**
- **감지기·인계 지연 단축 계열**(라운드 19 1위): 오라클 반사실로 효과 0 이 확인됐다.
- **G25 레벨 0 정체(0.76)에서 반복만 늘리는 것**: 구조를 바꾸지 않은 연장은 정보가 적다. 1위 결과를 보고 결합 여부를 정한다.
- **'첫 걸음 가속'을 목적으로 한 매끄러움 벌점**(속도와 반대 방향, 이력상 느린 흔들림). 떨림 대책은 별도 항목으로 다룬다.
- 12 N·s 목표(라운드 19 와 같음).

출처: 우리 기록 — 메모리 g16-generation-1001(G16d 지연 0–6 결과), stage8-latency-root-cause-0930, G24R-a env.yaml · [HoST](https://github.com/InternRobotics/HoST) `host_platform.py`·`g1_config_ground.py` · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · Isaac Lab 설치본 DelayedPDActuator(OSS_REFERENCES §1) · Tan et al. 2018(Sim-to-Real Minitaur, 지연 DR)

## 라운드 21 (2026-10-05 05:00) — 이어 학습이 막힌 회복 정책: 구조 대안 (a)–(e) 순위

전제(측정 사실, `reports/gen_c5/tourney2/RESULT.md` 외)
- G24R-a 21793 은 500–1500 반복 이어 학습으로 개선되지 않는다. feet_gap(e)·힙 편향 DR(g)·hip_a 게이트(h)·충격 커리큘럼(G25)·지연 DR 0–5(G25D) 모두 같은 반복 대조(T1: 고정 lr 1e-4·엔트로피 0 '안전 PPO')와 구별 불가하거나 더 나빴다.
- G25D 는 지연 2–4 에서 fi40 대부분 악화 → kill. G16d(지연 0–6, 다른 계보·긴 학습)의 개선은 500 반복 이어 학습으로 재현되지 않았다.
- 제약: 실물 유효 지연은 아마 시뮬 3–4틱. ~10 N·s 초과 충격은 물리적으로 캡처 불가(라운드 19). 램프 회복은 이미 양호하고, 실물에서 C5 램프 밀기 성공.
- T1 지연 3/4틱 성적: sf12 17–22 / 21–25 %, fs80 71–73 / 81–84 %, fi40 31–39 / 44–50 %.

표기: [측정], [코드], [논문], [추론].

### 0. 판단의 축

- **지금 가장 모르는 것은 '한계가 정보(관측) 부족 때문인가, 물리(지연 3–4틱·발 폭·걸음 시간) 때문인가'다.** 다섯 대안의 기대 이득은 이 답에 따라 갈린다.
  - 정보 부족이면 → 특권 정보를 주는 (b)·(d) 가 효과 있다.
  - 물리 한계면 → 어떤 학습 구조도 효과가 작고, 실물 지연 단축·운용 범위 제한이 답이다.
- 따라서 **이 질문에 가장 싸게 답하는 실험**이 GPU 시간당 정보량이 가장 크다.

### 1. (a)–(e) 순위 — GPU 시간당 기대 이득

| 순위 | 대안 | 배포 계약(actor 관측·행동) | 근거 | 비용 |
|---|---|---|---|---|
| **1** | **(b) 교사–학생** — 1단계 '특권 교사 탐침'부터. 21793 에 특권 입력(참 몸통 선속도, 현재 외력·외력 시작 후 시간, 행동 지연 틱, 공통 힙 편향)을 **0 초기화 열**로 붙여(라운드 8 §1 수술 방식) 500 반복 | 교사는 actor 관측 변경(시뮬 전용, 배포 안 함). **학생은 계약 불변**(같은 320 관측) | 특권 교사 → 학생 증류는 다리 RL 표준: Lee et al. 2020(Science Robotics, 교사–학생), RMA(Kumar 2021) [논문]. rsl_rl 3.1.2 설치본에 Distillation 알고리즘·러너가 있고 우리 train.py 가 이미 분기 지원(`rl_walking/scripts/train.py:189-190,219-220`; `rsl_rl/algorithms/distillation.py:28,116-153`) [코드] | 교사 탐침 500 반복 + 평가. 학생 증류는 탐침이 양성일 때만 |
| 2 | (d) 비대칭 critic — critic 에 외력·지연·편향·참 속도·접촉력·CoM/CP 추가(현재 critic 은 base_lin_vel 15 차원뿐, 라운드 8 §1) | **계약 불변**(critic 만) | 특권 critic 표준: humanoid-gym(`humanoid_env.py:219-235`, 외력·마찰·질량·접촉), FALCON(critic 외력), arXiv 2603.08619(critic CoM·운동량·CP) [코드/논문]. 숨은 변수(외력 시점·지연)가 어드밴티지 분산의 주원인이라면 이득 | 체크포인트 수술(critic 0 열 + Adam 상태 + symmetry critic 규칙) + 500 반복. 기대 이득은 '학습 신호 품질'이라 (b) 보다 간접적 |
| 3 | (a) 새 회복 전문가(처음부터 또는 A0m 에서), 회복 전용 보상(서기 항 없음), 지연 2–5·충격/램프 혼합 | 계약 불변(같은 관측·행동이면) | 회복 정책을 처음부터 학습한 예: Atalante·Ferigo 2021·van Marum 2024·arXiv 2603.08619 [논문]. 우리 G24R-a 는 서기 계보 보상을 이고 있다(라운드 6 §1: 양발 요구 양(+) 항·stance_shape 가 회복 게이트 밖에서 남음) | 수천–수만 반복(가장 비쌈). 천장은 가장 높을 수 있지만 (b) 탐침이 '물리 한계'로 나오면 같은 벽에 부딪힘 |
| 4 | (e) PPO 하이퍼파라미터(clip·epoch·미니배치·가치 손실 계수) 또는 SAC | 계약 불변(학습 전용). **PPO 하이퍼파라미터는 사용자 동결 대상 → 승인 필요** | 이어 학습 위생은 이미 안전 PPO(고정 lr 1e-4·엔트로피 0)로 고정했고 그래도 구별 불가 → 하이퍼파라미터가 병목일 근거 약함. SAC 는 대규모 병렬 Isaac 파이프라인과 맞지 않음(실물 SAC 보행 Haarnoja et al. 2019 는 소수 env) [논문] | 승인 + 탐색 비용, 기대 이득 낮음 |
| 5 | (c) 회복 전문가 2개(램프 vs 충격) | 각 정책 계약 불변. **러너 전환 로직 변경**(램프/충격 판별기 추가 — 배포 로직 변경, 승인 필요) | 램프는 이미 양호(실물 성공). 충격 전문가 학습은 결국 (a) 와 같은 비용. 판별 오류·전환 추가 위험 | (a) 비용 + 전환 위험 |

**순위 이유 요약**: (b) 의 1단계(교사 탐침)는 500 반복 하나로 '정보 한계 vs 물리 한계'에 답한다. 그 답이 (a)·(c)·(d) 의 가치를 정한다. (d) 는 싸지만 간접적이고, (a) 는 비싸며 (b) 의 답 없이 착수하면 물리 벽에 투자할 위험이 있다.

### 2. 이번 라운드의 세 가지(독립 판단)

**(1) 실물에서 가장 큰 실패 위험**
- **실물 유효 지연 3–4틱에서의 회복 신뢰도 저하.** 시뮬 T1 기준 지연 3–4틱이면 sf12 17–25 %, 8 N·s 충격(0.1 s) 71–84 % 낙상이다. 실물 램프 밀기 성공은 이 구간에서도 램프가 비교적 안전하다는 뜻이지만, **충격성 밀침과 강한 램프(sf15 급)는 실물에서 시뮬 명목보다 2–4배 위험**하다고 봐야 한다.
- 운용 대책: 실물 시험 밀기를 램프형·약한 세기로 제한하고(로그 Δv 로 사후 확인), 회복 모드 중 지연·떨림 자동 중단 기준(라운드 20 (1))을 적용한다.

**(2) 정보량이 가장 큰 다음 실험(≤ 500 반복)**
- **특권 교사 탐침**: 21793 의 actor·critic 입력에 특권 열을 0 초기화로 추가한다(참 몸통 선속도 3, 현재 외력 2, 외력 시작 후 경과 1, 행동 지연 틱 1, 공통 힙 편향 1 = 8 차원, 5틱 이력 없이 현재값). 지연 DR 은 2–5 로 두고 안전 PPO 500 반복, 대조 = T1(같은 반복).
- 평가: 지연 3틱에서 sf12·fs80·fi40, 같은 seed 1–2, 체크포인트 3개.
- **판정 규칙**
  - 교사가 fi40·fs80 에서 대조 대비 **2 SD 이상 그리고 상대 30 % 이상** 개선 → '정보 한계' → 다음은 (b) 학생 증류(rsl_rl Distillation, 같은 320 관측), 병행으로 (d).
  - 개선이 없거나 2 SD 미만 → **'물리·지연 한계'** → 회복 정책 학습 분기 중단, 실물 지연 단축(배포 파이프라인)과 운용 범위 설정으로 전환. (a) 도 이 경우엔 보류.
- 주의: 탐침 교사는 배포 불가 정책이다(시뮬 전용). 네트워크 입력 변경은 연구 탐침이며 배포 계약과 무관하다. 다만 '네트워크 동결' 지시의 범위에 연구 탐침이 포함되는지는 사용자 확인이 필요하다.

**(3) 멈출 것**
- **21793 에서의 단일 보상/DR 변경 + 500–1500 반복 이어 학습 분기**(tourney2: 5/5 구별 불가·악화). 같은 형식을 더 시도하는 것은 정보가 적다.
- 지연 DR 0–5 이어 학습(G25D kill)과 충격 커리큘럼 레벨 0 연장(G25 정체).
- (e) 의 하이퍼파라미터 탐색과 SAC 전환 — 이어 학습 위생을 고정한 뒤에도 구별 불가였으므로 근거가 약하다.

출처: 우리 `reports/gen_c5/tourney2/RESULT.md`, `tourney1/PLAN.md`(안전 PPO = 고정 lr 1e-4·엔트로피 0) · rsl_rl 3.1.2 설치본 `algorithms/distillation.py` · 우리 `rl_walking/scripts/train.py` · [humanoid-gym](https://github.com/roboterax/humanoid-gym) · [FALCON](https://github.com/LeCAR-Lab/FALCON) · Lee et al. 2020(Learning quadrupedal locomotion over challenging terrain, 교사–학생) · Kumar et al. 2021(RMA) · arXiv 2603.08619 · Atalante 2022 · Ferigo 2021 · van Marum 2024 · Haarnoja et al. 2019(실물 SAC 보행)
