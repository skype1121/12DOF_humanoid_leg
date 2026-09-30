## 조사 결과: HumanoidVerse / ASAP의 sim2real 측

결론부터 말하면, HumanoidVerse 본체(master)에는 sim2real 코드가 없습니다. README의 TODO 항목 `[ ] Sim-to-Sim and Sim-to-Real pipelines.`(L44)가 아직 체크되지 않았습니다. 실제 sim2real 코드는 HumanoidVerse를 기반으로 만든 ASAP 저장소에 모두 들어 있습니다. 그리고 우리 문제에 가장 영향이 클 것은 delta-action 자체가 아니라, ASAP의 배포 구조(한 루프 안에서 최신 상태 읽기→추론→송신)와 실물 궤적을 시뮬에 되돌려 비교하는 재생 방법론입니다.

**조사 대상 (얕은 복제):**
- `HV` = `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/HumanoidVerse_s2r` (master 101492b, 2025-06-12)
- `ASAP` = `…/scratchpad/oss/ASAP` (main df5320c, 2026-01-05)
- `SPI` = `…/scratchpad/oss/SPI-Active` (edc07d0) — LeCAR의 SysID 후속작(CoRL'25). 액추에이터 식별 범위라서 포함했습니다.
- ASAP 논문 arXiv 2502.01143v3는 PDF 본문 전체를 확인했습니다.
- 우리 저장소는 `rl_walking/actions.py`와 `rl_walking/hang_stand_env_cfg.py`를 비교용으로 읽기만 했고, 아무것도 수정하지 않았습니다.

---

### 1. sim2real 구성요소와 성숙도

| 구성요소 | 위치 | 실제 상태 |
|---|---|---|
| 액션 지연 무작위화(정책 스텝 단위 정수) | `HV/humanoidverse/envs/legged_base_task/legged_robot_base.py` L112-116, L193-198, L711-716 | 구현됨. 다만 단순함 |
| PD 게인·질량·CoM·마찰·RFI·밀기 무작위화 | `HV/…/config/domain_rand/domain_rand_base.yaml` | 구현됨 |
| 다중 시뮬 간 sim2sim(IsaacGym/IsaacSim/Genesis) | `eval_agent.py +simulator=…` | 구현됨 |
| delta-action 학습(오픈루프 재생) | `ASAP/humanoidverse/envs/delta_a/delta_a_open_loop.py` + 설정 파일 | 구현됨(연구용 코드, 디버그 print 잔존) |
| delta-action으로 정책 fine-tune | `ASAP/humanoidverse/agents/delta_a/train_delta_a.py`, `envs/delta_a/delta_a_closed_loop.py` | 구현됨. 단위 불일치가 의심됨(§2.6) |
| 4-DoF 발목 전용 delta(실물 실험 설정) | `delta_a_open_loop.py` L62-64 `anklePR` 플래그 | 출력 마스킹만 있음. 관측 설정은 미공개 |
| 실물 데이터 수집 | `ASAP/sim2real/rl_policy/listener_deltaa.py` | 구현됨. MoCap 필수 |
| 실물 npz → 학습 입력(motion pkl) 변환 | 없음 | 미공개 |
| SysID 비교군(논문 Table VII의 그리드 탐색) | 없음 | 미공개 |
| Delta-dynamics 비교군 | `agents/delta_dynamics/delta_dynamics_model.py` | 불완전. `parse_delta` 등 env 쪽 함수가 없어 실행 불가 |
| MuJoCo 학습 백엔드 | `config/simulator/mujoco.yaml` | 설정만 있고 구현 파일이 없음 |
| MuJoCo sim2sim(실제 통신 경로 사용) | `ASAP/sim2real/sim_env/base_sim.py`, `sim2real/utils/unitree_sdk2py_bridge.py` | 구현됨(G1 전용) |
| 실물 배포(Unitree SDK2 DDS) | `ASAP/sim2real/rl_policy/*.py` | 구현됨. 안전장치가 빈약 |
| 액추에이터 네트워크·백래시·쿨롱/스틱션 모델 | 없음 | IsaacSim의 armature/관절마찰 상수만 있음 |

---

### 2. ASAP delta-action

#### 2.1 알고리즘
- 정의는 Δa_t = π_Δ(s_t, a_t)이고, 시뮬 동역학은 s_{t+1} = f_sim(s_t, a^r_t + Δa_t)입니다.
- 실물에서 기록한 행동 a^r을 시뮬에 재생하면서 실물 궤적과 맞도록 PPO로 π_Δ를 학습합니다.
- 학습이 끝나면 π_Δ를 동결하고, f_sim(s, a + π_Δ(s,a))라는 보정된 시뮬에서 기존 정책을 fine-tune합니다. 실물 배포 때는 π_Δ 없이 정책만 씁니다.
- 코드상으로 보면 PD 위치제어이므로 Δa는 관절별 잔차 토크 kp·0.25·Δa와 같습니다. 50 Hz로 한 번 정해져 200 Hz 물리 4 substep 동안 유지됩니다. 즉 "상태·명령에 의존하는 잔차 토크 모델"입니다.
  ```python
  # delta_a_open_loop.py L50, L53-57, L92
  actions_scaled = actions * action_scale            # Δa*0.25
  motion_action  = open_loop_action * action_scale   # a^r*0.25
  torques = kp*(actions_scaled + motion_action + default_dof_pos - dof_pos) - kd*dof_vel
  ```

#### 2.2 필요한 데이터
- **상태:** 논문 III-A 기준으로 s_t = [베이스 위치(MoCap), 베이스 선속도, 쿼터니언, 각속도, q(23), q̇(23)]과 행동 a^r_t입니다.
- **로거 (`listener_deltaa.py`):**
  - L20: `buffer_rate=50.0`(주석에는 200 Hz로 적혀 있음), 버퍼 50×600(에피소드당 10분).
  - lowstate·lowcmd·`/odometry`의 최신값을 50 Hz 타이머로 샘플링하고, 세 가지가 모두 있어야 기록합니다(L201).
  - 저장 키(L328-348): time, joint_pos, joint_vel, tau_est, IMU_quaternion/gyro/acc, joint_pos_cmd, joint_vel_cmd, kp, kd, tau_cmd, pos, quat, lin_vel, ang_vel.
- **학습 입력 포맷:** motion pkl `{root_trans_offset, pose_aa, dof, root_rot, action, fps}`입니다(`motion_tracking.py` L81-82 저장, `motion_lib_base.py` L384-405에서 `action` 키 감지). `action`은 스케일 전 정책 출력입니다.
  - 실물 npz에서는 a = (q_cmd − q_default)/0.25로 역산해야 합니다(추정). 관절한계 클립이 걸린 구간은 복원할 수 없습니다.
- **기준 속도:** 기록된 속도를 쓰지 않고 위치 차분으로 계산합니다(`torch_humanoid_batch.py` L207-224, L272-279). 몸체 속도는 σ=2프레임 가우시안 필터, 관절속도는 필터 없는 차분입니다.
- **데이터 양:**
  - 실물에서는 23-DoF가 불가능해서(시뮬 기준으로도 ">400 clips" 필요) 발목 pitch/roll ×2의 4-DoF로 줄였습니다. 100 클립으로 충분했다고 보고합니다.
  - 5개 동작을 각 30회 실행했고, 로코모션 데이터 10분을 추가로 수집했습니다. 동작 길이는 3.06~7.66 s이므로 약 6~8분 분량입니다(추정).
  - 수집 중 G1 2대가 과열로 파손됐습니다.
  - Fig.10a: 데이터를 4300에서 43000 샘플로 늘려도 폐루프 개선은 0.65%에 그쳤습니다.

#### 2.3 학습 절차 (README 명령과 설정)
- **명령:** `+exp=train_delta_a_open_loop +domain_rand=NO_domain_rand +obs=delta_a/open_loop num_envs=5000 env.config.max_episode_length_s=1.0 rewards.reward_scales.penalty_minimal_action_norm=-0.1`
- **에피소드:**
  - 클립 안의 임의 시각에서 실물 상태로 리셋합니다(RSI, `motion_tracking.py` L212-219, L395-482; 초기 잡음 0).
  - 1.0 s(50 스텝) 동안 재생합니다. 넘어짐 종료 조건은 모두 꺼져 있습니다(`env/delta_a_open_loop.yaml` L13-18).
- **보상 (`reward_delta_a_openloop.yaml` L10-49):**
  - 가중치: 몸체 위치 1.0, VR 3점 1.0, 발 위치 1.0, 회전 0.5, 각속도 0.5, 선속도 0.5, 관절각 0.5, 관절속도 0.5.
  - 페널티: action rate −0.01, 관절각 한계 −10, 관절속도 한계 −5, 토크 한계 −0.1, 종료 −200.
  - 행동 크기 페널티 exp(‖Δa‖)−1: README −0.1, 논문 −0.2, Fig.10c 최적값 0.1.
  - 커널 exp(−평균 오차²/σ)의 σ: 위치 0.03~0.1, 회전·속도·관절각·관절속도는 모두 1.0.
- **π_Δ 관측** (`open_loop.yaml` L6-20; 키를 알파벳 순으로 이어붙임, `legged_robot_base.py` L545-548):
  - base_pos_z, 발 접촉력, 베이스 선속도·각속도, 중력 방향, dof_pos, dof_vel, 자기 직전 Δa, 이번 스텝에 적용될 a^r.
  - 행동 이력은 없습니다. `history`는 정의만 있고 주석 처리돼 있습니다(L19, L36-41).
- **PPO:** 24 steps/env, 5 epochs, 4 미니배치, clip 0.2, γ 0.99, λ 0.95, 엔트로피 0.01, lr 1e-3(adaptive KL 0.01), 초기 표준편차 0.8, MLP 512-256-128 ELU.
- **Horizon (Fig.10b):** 오픈루프 오차는 1.5 s에서 최소, 폐루프 성능은 1.0 s가 최적이었습니다.
- **진단 산출물:** 관절별 |Δa| 평균 heatmap(`delta_a_open_loop.py` L71-77). 논문 Fig.13에서는 하체 > 상체, 발목 pitch가 최대, 좌우 비대칭이 나타났습니다.

#### 2.4 fine-tune
- **명령:** `+exp=train_delta_a_closed_loop algo.config.policy_checkpoint=<Δ모델> +checkpoint=<정책> +domain_rand=NO_domain_rand_finetune_with_deltaA env.config.add_extra_action=True domain_rand.push_robots=False env.config.noise_to_initial_level=1 algo.config.num_learning_iterations=1000`
- `train_delta_a.py` L44-81에서 Δ 모델을 로드·동결하고, L111-112에서 매 스텝 π_Δ(closed_loop_actor_obs)를 추론합니다.
- 토크는 `delta_a_closed_loop.py` L164에서 τ = kp(0.25·a_t + Δa + q_def − q) − kd·q̇로 계산됩니다.
- fine-tune 중에는 무작위화가 모두 꺼집니다. 지연도 포함입니다(`NO_domain_rand_finetune_with_deltaA.yaml` L43-44).
- **코드에만 있는 옵션**(논문에는 없음, 기본 False; 같은 yaml L72-76, env L37-52, L73-77):
  - `cotrain_with_without_delta_a`: 환경의 50%는 Δa 없이 학습.
  - `rescale_delta_a`: env·관절별로 Δa에 U(0.5, 1.5) 배율.
- 학습 없이 Δ 모델을 쓰는 방법(고정점 반복/경사 탐색, L89-151)도 코드에 있지만, 23차원이 하드코딩돼 있고 비활성 경로입니다. 논문 Fig.11에서는 이 방법들이 기준선보다 나빴습니다.

#### 2.5 논문이 보고한 성과
- **오픈루프 재생 (Table III, 1.0 s horizon, 단위 mm):**

  | 대상 시뮬 | OpenLoop | SysID | DeltaDyn | ASAP |
  |---|---|---|---|---|
  | IsaacSim | 80.8 | 77.6 | 68.1 | 37.9 |
  | Genesis | 82.5 | 76.5 | 50.2 | 36.9 |

  0.25 s horizon에서는 네 방법 모두 19~24로 차이가 없었습니다.
- **폐루프 fine-tune (Table IV):**
  - Hard/IsaacSim: 148 → 129 (Oracle 116).
  - Hard/Genesis: 175(성공률 82.9%) → 129(성공률 100%).
  - SysID는 오히려 악화된 경우도 있습니다(Hard/IsaacSim 165).
- **실물 (Table V):**
  - Kick: 61.2 → 50.2 (−18%).
  - LeBron(학습 분포 밖): 159 → 112 (−30%).
  - 표본 수와 분산은 기재돼 있지 않습니다. 결론의 "최대 52.7%"는 Table V에서 직접 재현되지 않습니다(추정: 다른 지표나 버전).
- **무작위 액션 노이즈 fine-tune (Fig.12):** β ∈ [0.025, 0.2]에서 개선은 있지만 ASAP가 126으로 더 좋았습니다. 노이즈 쪽 수치는 본문 150, 캡션 173으로 논문 안에서도 불일치합니다.
- 논문의 주장: 불일치는 관절별로 구조화돼 있어서 균일 노이즈로는 잡을 수 없고, Δ 모델이 들어가면 시뮬이 실물의 실패를 재현한다.

#### 2.6 코드 판독상 함정 (재구현 시 반드시 시험할 것)
1. **Δa 스케일 4배 불일치 (추정: 버그 또는 비공개 의도).** 학습 때는 0.25·Δa(open_loop L50)인데, fine-tune 때는 1.0·Δa(closed_loop L67-70, L164)입니다. `action_scale`은 0.25입니다.
2. **1스텝 어긋난 입력 (추정).** fine-tune 때 π_Δ가 보는 정책 행동은 직전 스텝 값입니다(`_get_obs_actions_sim2real_policy` = `self.actions`, L187-188). 학습 때는 곧 적용될 a^r을 봅니다(L137-141, +1).
3. **보상 기준이 1프레임 앞섬 (추정).** 에피소드 카운터가 증가한 뒤에 (buf+1)·dt 시점을 기준으로 씁니다(`motion_tracking.py` L236, `legged_robot_base.py` L239-244). 결과적으로 시뮬의 t+1 상태를 실물의 t+2와 비교하므로, 지연 식별에 쓰면 20 ms 편향이 생깁니다.
4. **Δa가 지연 큐를 우회 (추정).** 지연 무작위화를 켜면 정책 행동만 지연되고 Δa는 즉시 적용됩니다.
5. **이름은 같고 물리는 다른 무작위화.** `randomize_friction`이 IsaacSim에서는 관절 마찰 배율(`isaacsim.py` L129-139)이고, IsaacGym에서는 지면 마찰(`isaacgym.py` L266-291)입니다.
6. **네임스페이스 비호환.** ASAP는 `omni.isaac.lab`(IsaacLab 1.x, Isaac Sim 4.2)을 씁니다. 우리 Isaac Lab 2.3.2와 호환되지 않으므로 이식보다 재구현이 맞습니다(추정).

---

### 3. 지연·액추에이터·SysID 처리 (정확한 값)

#### 3.1 지연 구현
```python
# HV legged_robot_base.py L193-198 (ASAP L219-224 동일)
self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
self.action_queue[:, 0]  = self.actions.clone()
self.actions_after_delay = self.action_queue[torch.arange(N), self.action_delay_idx]
```
- **동작 방식:**
  - 지연은 20 ms 정책 스텝 단위의 정수이고, env별로 에피소드마다 고정됩니다. 에피소드 중 지터는 없습니다.
  - 리셋 때 큐를 0으로 채웁니다(L711-716). 따라서 리셋 직후 d스텝 동안은 기본자세가 목표로 들어갑니다.
  - 관측 지연은 모델링하지 않습니다.
- **범위:**
  - 코드 기본값은 `[0,2]` 스텝(0~40 ms)입니다. 주석에는 "max real delay is 90ms"라고 적혀 있습니다.
  - 논문 Table VI는 U(20, 40) ms입니다.
  - 공개된 학습 명령은 NO_domain_rand라서 지연 무작위화가 꺼져 있습니다. 공개 로코 모델 이름에도 "noDR"이 들어 있습니다(추정: 무작위화 없이 학습).
- **우리 코드와 비교:**
  - `rl_walking/actions.py` L115-121은 리셋 때 지연 버퍼를 현재 관절각으로 채웁니다. LPF·백래시·슬루 제한도 이미 있습니다.
  - `hang_stand_env_cfg.py` L1439-1440은 지연 9~14틱입니다.
  - 따라서 우리 구현이 이미 상위호환입니다. HV에서 가져올 것은 없고, 오히려 반면교사입니다(0 초기화, 관측 지연 부재, 지터 부재).

#### 3.2 PD와 액추에이터
- 모든 백엔드에서 env가 PD 토크를 직접 계산합니다(ASAP `legged_robot_base.py` L550-578). 200 Hz(fps 200, decimation 4)로 계산하고 토크 한계로 클립합니다.
- G1 설정(`g1_29dof_anneal_23dof.yaml` L125-162): kp/kd는 고관절 100/2.5, 무릎 200/5, 발목 20/0.2(roll은 0.1), `action_scale` 0.25, 행동 클립 100.
- IsaacSim에서만 IdealPD(강성·감쇠 0 = 토크 직접 입력) + armature 0.03(한 관절 0.04) + 관절마찰 0.03을 씁니다(`isaacsim.py` L306-318). IsaacGym은 asset armature 0.001입니다.
- RFI: τ += U(−1, 1)·0.1·U(0.5, 1.5)·τ_limit를 substep마다 더합니다. 백색 노이즈입니다(HV L552-553).
- **무작위화 기본값 (`domain_rand_base.yaml`):**
  - 밀기 [5, 10] s·1.0 m/s
  - CoM ±0.1 m
  - 링크 질량 ×[0.8, 1.2]
  - kp/kd ×[0.75, 1.25]
  - 지면 마찰 [0.5, 1.25]
  - RFI 0.1
  - 지연 [0, 2] 스텝
- **논문 Table VI:** 마찰 U(0.2, 1.1), P 게인 U(0.925, 1.05), 지연 U(20, 40) ms, 밀기 10 s/0.5 m/s.
- **관측 잡음** (`deepmimic_a2c_nolinvel_LARGEnoise_history.yaml` L67-79, 균등분포, `helpers.py` L101): 각속도 ±0.3 rad/s, 중력 ±0.2, 관절각 ±0.01 rad, 관절속도 ±1.0 rad/s.

#### 3.3 SysID
- **ASAP의 SysID 비교군 (논문만, 코드 없음):**
  - 탐색 범위: CoM 각 축 {−0.02, 0, 0.02} m, 베이스 질량비 {0.95, 1, 1.05}, 관절별 kp/kd비 {0.95, 1, 1.05}.
  - 결과: 짧은 horizon에서만 개선됐습니다.
- **SPI-Active:**
  - `SPI/scripts/eval.py` L101-142, L218-292: 기록된 모든 시각 t를 초기상태로 두고 K=5 스텝 동안 기록 행동을 재생합니다. 베이스 위치/쿼터니언/관절각 오차를 가중치 10/5/1로 비교합니다.
  - `mass_opt.py` L28-35, L203-217: Optuna TPE로 50회 탐색합니다.
  - 모터 모델(`active_sysid_openloop.py` L356-399): τ′ = a·tanh(τ/a)를 관절군별로 적용합니다.
  - 여기(L402-427)는 FIM 보상 계산입니다. 파라미터를 ±δθ 섭동한 보조 env와의 유한차분으로 구하고, 이 값을 최대화하는 명령 탐색(Optuna)은 `active_sysid.md`에 설명돼 있습니다.
  - 실물 데이터 수집 절차와 sim2real 코드는 미공개입니다("stay tuned"). 데모는 시뮬 데이터입니다(Go2 베이스 질량 6.921 kg → 식별값 7.006 kg).

---

### 4. 배포

- **구조:** 정책 프로세스 1개가 외부 PC에서 이더넷으로 G1에 연결됩니다.
  - 메인 스레드: 50 Hz 루프(`base_policy.py` L423-436; `create_rate(50)` L43).
  - rclpy spin 데몬 스레드(`deepmimic_dec_loco_height.py` L238-241).
  - unitree DDS 수신 스레드, 키보드 스레드.
- **한 틱의 처리 (`rl_inference`, L203-231):**
  1. 최신 LowState를 복사합니다(`state_processor.py` L34-52).
  2. 관측을 알파벳 순으로 손으로 이어붙입니다(L149-163). 이력(4프레임)은 관측을 만든 **뒤에** 갱신하며, 학습 때와 같은 순서입니다.
  3. ONNX를 CPU에서 추론하고 ±100으로 클립합니다.
  4. ×0.25 + 기본자세를 더한 뒤 관절한계로 클립합니다(L224-225).
  5. LowCmd(q, dq=0, tau=0, kp, kd, CRC)를 곧바로 보냅니다(`command_sender.py` L130-149). PD는 모터 드라이버가 수행합니다.
- **통신과 지연 최소화:**
  - 로봇 입출력은 ROS2 노드를 거치지 않고 DDS에 직결됩니다.
  - 구독 큐 길이는 1이고, 핸들러는 최신 메시지를 변수에 대입만 합니다(`state_processor.py` L17-21, L54-58).
  - SDK 내부에서 큐가 차면 새 샘플을 버리지만(`BQueue.Put(replace=False)`, upstream `core/channel.py`), 핸들러가 즉시 비우므로 사실상 "최신값 보관"이 되고 지연 상한이 유계입니다.
  - 지연 측정이나 타임스탬프 검사는 없습니다.
- **정책 전환 (`deepmimic_dec_loco.py` L86-96, L365-430):**
  - loco→mimic: 1.5 s 보간 + 1.0 s 간격.
  - mimic→loco: 0.5 s 간격 + 1.5 s 보간.
  - 전환 때 관측 이력을 0으로 리셋합니다(L389).
- **안전장치:**
  - 있는 것: 관절한계 클립, `i` 키로 500틱(10 s) 동안 기본자세로 보간(L170-178), `o` 키 E-stop(목표를 매 틱 측정각으로 바꿔 사실상 댐핑 홀드, L252-255), kp 전체 배율 런타임 조절(키 4/7 ±0.1, 5/6 ±0.01, L280-309).
  - 없는 것: 오래된 상태 감지(마지막 메시지를 계속 반환), 워치독, 슬루 제한, 낙상 가드, 온도 감시.

---

### 5. sim2sim 파이프라인

1. **HV 다중 시뮬:** `eval_agent.py +checkpoint=… +simulator=genesis`처럼 시뮬만 바꿔 평가합니다. env 코드와 PD 계산은 같고 엔진만 다릅니다(접촉, 솔버, armature 0.001 vs 0.03).
2. **논문의 sim2sim:** IsaacGym에서 학습하고 IsaacSim·Genesis를 "가짜 실물"로 삼았습니다. 대상 시뮬의 롤아웃을 `save_motion`(`motion_tracking.py` L485-542; `opt/record.yaml`)으로 저장하는데, 이 저장 포맷이 곧 delta-action 학습 입력 포맷입니다.
3. **MuJoCo 실통신 sim2sim (`base_sim.py`):**
   - `SIMULATE_DT` 0.005 s로 실시간 진행하고, 매 스텝 실제 상태를 LowState로 발행합니다.
   - 최신 LowCmd로 τ = tau + kp·Δq + kd·Δq̇를 계산하고 토크 한계로 클립합니다(L76-104).
   - 정책 코드는 실물과 동일하고, 네트워크 인터페이스만 `INTERFACE: "lo"`로 바뀝니다.
   - ElasticBand(`unitree_sdk2py_bridge.py` L346-376): 몸통을 점 (0,0,3)에 k=200 N/m, c=100 N·s/m 스프링으로 매답니다. 키 7/8로 길이를 ±0.1 m씩 조절하고 키 9로 해제합니다.
   - 지연·잡음·액추에이터 모델은 없습니다.

---

### 6. 우리에게 가져올 것 (영향 순)

1. **정책→모터 경로를 "최신값 하나"로 단순화** — 지연 문제에 직접 해당. 시뮬이 더 순한 문제와 2 Hz 흔들림에도 영향이 있을 수 있음(추정)
   - **근거:** 앞의 `state_processor`, `rl_inference`, `command_sender` 패턴과, 로깅을 별도 프로세스의 수동 탭으로 분리한 구조(`listener_deltaa.py`).
   - **적용:**
     - 정책 브리지→stage8→CAN의 각 구독을 KEEP_LAST depth 1 + 최신값 대입으로 바꿉니다.
     - 가능하면 추론과 송신을 한 스레드로 합칩니다.
   - **280 ms 큐잉의 원인 (추정):** depth가 1보다 큰 큐가 상시 포화된 것(예: depth 10 × 20 ms = 200 ms)에 홉별 타이머 위상차가 더해진 것일 수 있습니다.
   - **필요한 데이터:** 홉별 타임스탬프(정책 관측 시각, 목표 발행, stage8 수신, CAN 송신, 모터 응답). CAN 덤프와 블랙박스로 대부분 확인 가능합니다.
   - **주의:**
     - ASAP는 지연을 재지도, 오래된 상태를 감지하지도 않습니다. 우리 가드는 그대로 유지해야 합니다.
     - 지연만 바꾼 A/B를 대조군과 함께 돌려야 합니다.
     - 지연이 줄면 학습용 지연 무작위화(현재 9~14틱)를 다시 맞춰야 합니다.

2. **오픈루프 재생 대조 하네스** — 2 Hz 미재현과 "시뮬이 더 순함" 진단용
   - **근거:** 논문 Table III / Fig.5의 방법(기록 행동을 재생하고 horizon 0.25/0.5/1.0 s별 오차 비교)과 RSI 코드.
   - **적용:** 같은 로그를 두 가지로 재생합니다.
     - CAN 덤프의 실제 송신 목표를 **CAN 송신 시각 기준으로** 재생 → 소프트웨어 큐잉을 뺀 순수 플랜트 비교.
     - 블랙박스의 브리지 목표로 재생 → 큐잉 포함 비교.
     - 두 결과의 차이가 파이프라인의 기여입니다. 정책을 빼고 비교하므로 "플랜트 탓인지 정책 탓인지"를 분리할 수 있습니다.
   - **필요한 데이터:** 50 Hz q_cmd/kp/kd, q, q̇, IMU, 접촉비트, 전류(토크 추정).
   - **데이터 구간:** 매달림 구간은 베이스를 고정하고 바로 재생할 수 있습니다(추정). 지면 구간은 발 평평 가정의 FK + IMU로 베이스 상태를 복원해야 합니다(추정).
   - **주의:**
     - 지연 버퍼를 기록된 과거 목표로 채워서 시작해야 합니다. HV처럼 0으로 채우면 안 됩니다.
     - horizon은 1 s 이상이어야 합니다(지연 16틱 + 2 Hz 주기 관찰).
     - 줄이나 손 외력이 들어간 구간은 제외해야 합니다.

3. **재생 기반 명시 파라미터 식별** — 백래시·스틱션·무릎 처짐·지연
   - **근거:** SPI의 배치 재생(`eval.py` L218-292) + TPE 탐색(`mass_opt.py`) + 모터 모델 연결 지점(`active_sysid_openloop.py` L174-187).
   - **적용:** 우리 `actions.py`의 LPF·백래시·지연·마찰, 무릎 직렬강성, kp 배율을 파라미터로 두고 재생 오차를 최소화합니다. 비용항은 IMU 자세·자이로에 큰 가중치를 두고, 전류 기반 토크 오차를 추가합니다. 전류 토크는 ASAP·SPI에는 없는 우리 쪽 장점입니다.
   - **주의:**
     - SPI의 K=5(0.1 s)로는 320 ms 지연을 식별할 수 없습니다. K는 20~50 이상이어야 합니다.
     - 엔코더가 로터 쪽이라 관절각 비용으로는 백래시가 보이지 않습니다. 발이 평평할 때 IMU와 고관절·무릎 각으로 링크 쪽 발목각을 기구학적으로 구해 기준으로 쓰는 편이 낫습니다(추정).
     - 채택 여부는 실물 서명(0.5 Hz 성장 → 2 Hz 흔들림)을 재현하는지로만 판정해야 합니다.

4. **실제 통신 경로를 쓰는 sim2sim** — 2 Hz 미재현
   - **근거:** `base_sim.py` + ElasticBand.
   - **적용:** 우리 토픽이나 CAN을 흉내 내는 플랜트 노드를 만들고, 젯슨 소프트웨어 스택을 그대로 실시간으로 돌립니다. 큐잉·스레드·지터가 흔들림을 만드는지 검증하는 인과 시험이 됩니다.
   - **주의:**
     - 우리 로봇의 MJCF 모델이 없습니다.
     - ASAP 시뮬에는 액추에이터 모델이 없으므로 우리 모델을 옮겨야 합니다.
     - 젯슨 쪽 서비스 변경은 별도 승인이 필요합니다. 로컬 PC에 스택을 복제해 해볼 수도 있습니다(추정).

5. **잔차 모델을 구조화된 무작위화로 사용** — `rescale_delta_a` U(0.5, 1.5)와 50% 무잔차 공동학습(`delta_a_closed_loop.py` L37-52, L73-77). 식별 오차에 대한 보험 역할입니다. 영향은 중간 정도로 봅니다.

6. **축소형 delta-action** — 아래 타당성 평가를 참고하십시오. 우선순위는 낮습니다.

7. **기타:**
   - kp 배율 런타임 조절(`base_policy.py` L280-309)을 2 Hz 모드 판별에 활용할 수 있습니다. 지연이 지배하는 진동인지, 강성·백래시가 지배하는 진동인지를 kp에 대한 주파수 의존성으로 가를 수 있습니다(추정). 시뮬에서 먼저 하고, 실물은 승인이 필요합니다.
   - 정책 전환 보간은 참고할 만합니다. 다만 이력을 0으로 리셋하는 부분은 학습 분포 밖 입력을 만들 수 있으므로(추정) 실측값으로 채우는 쪽을 권합니다.

#### delta-action을 우리 50 Hz 로그로 적용할 수 있는가 (정직한 평가)

| 필요 조건 | 우리 상황 | 판정 |
|---|---|---|
| 상태·행동 쌍의 시간 정렬 | 블랙박스 목표가 정책 틱과 정렬됨 | 있음. ASAP(독립 타이머로 최신값 샘플링)보다 좋음 |
| 베이스 위치·선속도 | MoCap 없음 | 매달림 구간은 불필요. 지면 구간은 FK+IMU로 추정해야 하는데, 백래시 ±4°면 오차가 약 L·0.07 rad(L=0.6~0.7 m 가정 시 4~5 cm)로 흔들림 진폭과 같은 규모(추정) |
| 링크 쪽 관절각 | 로터 쪽 엔코더뿐 | G1 발목도 링크리지로 모터 쪽 측정이었지만 ASAP에는 MoCap 루트가 기준으로 들어갔음. 우리는 IMU가 사실상 유일한 링크 쪽 관측 → 보상 재설계 필요 |
| 지연 표현 | 320 ms ≈ 16틱 | **구조적으로 불가.** π_Δ 관측에 행동 이력이 없음. 명시 지연을 먼저 시뮬에 넣고, 관측에 16틱 이상의 행동 이력을 추가해야 함 |
| 외력 없는 데이터 | 줄·손 지지가 섞임 | Δa가 외력을 "액추에이터 잔차"로 흡수해 가짜로 맞아 보일 위험. 무보조 창은 짧음 |
| 데이터 양 | ASAP는 4-DoF에 100 클립(약 6~8분, 추정) | 무보조 지면 데이터는 그보다 적을 가능성(추정, 로그 집계 필요). 매달림 데이터는 풍부하지만 하중 방향이 다름 |
| 보상 감도 | 흔들림이 수 도 규모 | 회전·관절 σ=1.0이면 2° 오차에서도 보상이 약 0.999로 포화 → σ 재조정 필요(추정) |
| 코드 이식 | Isaac Lab 2.3.2 | 재구현 필요. 핵심 로직은 RSI 리셋 + 기록 행동 재생 + Δa PPO(추정 수일) |

**결론:**
- 원형(MoCap + 23-DoF)은 적용할 수 없습니다.
- 축소형은 가능합니다. 명시 지연과 파라미터 식별을 먼저 끝내고, 남은 잔차만 발목 pitch ×2(필요하면 무릎 ×2)로 학습하며, IMU·기구학 기반 보상을 쓰고, 무보조나 공중 창만 사용하는 방식입니다. 그래도 우선순위는 낮습니다.
- 가장 큰 불일치인 280 ms 큐잉은 모델링할 대상이 아니라 없앨 대상입니다. 백래시·스틱션은 명시 모델이 이미 있으므로 3번(재생 기반 식별)이 먼저입니다.
- delta를 쓰더라도 처음에는 관절별 |Δa| heatmap을 보는 진단용으로 쓰고, 채택은 폐루프에서 실물 서명을 재현하는지로 판정해야 합니다.

#### 적용 불가·따라하지 말 것
- MoCap 의존 수집, 23-DoF delta, 하드웨어 파손을 감수한 수집 방식.
- ASAP의 안전 설계 전반.
- §2.6의 결함으로 추정되는 4건. 재구현한다면 단위와 시간 정렬 단위시험이 필수입니다.
- 0~40 ms 지연 범위(G1의 저지연 기준이라 우리와 무관).
- IsaacSim/IsaacGym의 "friction" 의미 불일치.
- HV 본체에서 가져올 것은 사실상 다중 시뮬 평가라는 개념 정도입니다.

**핵심 파일 (절대경로):**
- `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/ASAP/humanoidverse/envs/delta_a/delta_a_open_loop.py`
- `…/oss/ASAP/humanoidverse/envs/delta_a/delta_a_closed_loop.py`
- `…/oss/ASAP/humanoidverse/agents/delta_a/train_delta_a.py`
- `…/oss/ASAP/sim2real/rl_policy/base_policy.py`
- `…/oss/ASAP/sim2real/rl_policy/listener_deltaa.py`
- `…/oss/ASAP/sim2real/sim_env/base_sim.py`
- `…/oss/ASAP/sim2real/utils/unitree_sdk2py_bridge.py`
- `…/oss/HumanoidVerse_s2r/humanoidverse/envs/legged_base_task/legged_robot_base.py`
- `…/oss/SPI-Active/scripts/eval.py`
- `…/oss/SPI-Active/spigym/envs/sysid/active_sysid_openloop.py`
- 논문 PDF와 텍스트: `…/oss/asap.pdf`, `…/oss/asap.txt`

**출처:**
- [ASAP arXiv 2502.01143](https://arxiv.org/abs/2502.01143)
- [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP)
- [LeCAR-Lab/HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse)
- [LeCAR-Lab/SPI-Active](https://github.com/LeCAR-Lab/SPI-Active)
- [SPI-Active arXiv 2505.14266](https://arxiv.org/abs/2505.14266)
- [unitree_sdk2_python channel.py](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/core/channel.py)
