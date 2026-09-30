# HumanoidVerse 학습 코어 조사 보고 (LeCAR-Lab)

- **대상**: https://github.com/LeCAR-Lab/HumanoidVerse, master `101492b` (2025-06-12). 전체 9커밋이며 첫 공개는 2025-02-04.
- **경로 약칭**: `HV/` = `/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/oss/HumanoidVerse_core/humanoidverse`
- **보조 참조**: 요청하신 "진동 억제 트릭"과 서기 설계를 보려고 파생 저장소 두 곳을 같은 폴더(`.../scratchpad/oss/`)에 추가로 받았습니다. ASAP_ref(df5320c)와 FALCON_ref(a967a6d)입니다.
- **검증 수준**: 전부 정적 코드 읽기이고 시뮬레이터는 돌리지 않았습니다. 표기는 [확인]=코드·파일·API로 직접 확인, [계산]=제가 산술로 낸 값, (추정)=추론입니다.
- `/home/ryu/humanoid_leg_test1`은 수정하지 않았습니다.

---

## 1. 요약

HumanoidVerse는 legged_gym에서 출발한 휴머노이드 sim-to-real 학습 "프레임워크"입니다. hydra 설정 그룹(simulator/exp/robot/rewards/obs/domain_rand/terrain)을 나눠 두고, `+simulator=` 한 줄로 IsaacGym(Preview 4), IsaacSim 4.2/Isaac Lab(`omni.isaac.lab` 1.x API), Genesis 0.2.1을 바꿔 끼웁니다.

- **RL**: 자체 PPO. rsl_rl을 옮겨 온 것이고 액터와 크리틱이 Adam을 따로 씁니다.
- **공개 master에 있는 것**: 속도명령 보행 태스크 1종(`LeggedRobotLocomotion`)과 H1 10/19dof, G1 12/29dof 로봇 설정뿐입니다.
- **없는 것**: 서기 전용 태스크, 모션 트래킹, sim2sim/sim2real 파이프라인. README TODO가 미완이고 이슈 #9에서도 확인됩니다.
- **성숙도: 연구 스냅샷 수준입니다.** 테스트가 없고, 보상 루프 안에 ipdb가 있으며, 시드를 호출하지 않고, 백엔드마다 의미가 어긋나고 버그가 여럿입니다(2장).
- **메인테이너 발언(이슈 #4) [확인]**: "G1 기본 보상으로 배포 가능한 컨트롤러가 나오느냐"는 질문에 "Not really. 프레임워크 기여이며 보상은 unitree_rl_gym을 참고하라"고 답했습니다.
- **라이선스와 지표**: MIT, 별 472/포크 32(2026-09-30 GitHub API 조회).

---

## 2. 아키텍처

### 2.1 구성 요소

**진입점**
- `HV/train_agent.py:20`: `@hydra.main(config_name="base")` 후 `instantiate(env)`, `instantiate(algo)`, `learn()` 순서로 돕니다.
- `HV/eval_agent.py`: 체크포인트 옆의 `config.yaml`을 읽어 `eval_overrides`와 CLI를 병합하고, ONNX로 내보낸 뒤(`:284-307`, 입력 이름 `actor_obs` 단일 텐서, opset 13) 평가합니다.

**설정 조합**
- `+simulator= +exp= +robot= +rewards= +obs= +domain_rand= +terrain=` 형태입니다.
- `exp/locomotion.yaml`이 `algo/ppo.yaml`과 `env/locomotion.yaml`을 불러오고, 후자는 `legged_base.yaml`, `base_task.yaml`로 이어집니다.
- 커스텀 resolver(`eval`, `len` 등)는 `HV/utils/config_utils.py`에 있습니다.

**클래스 계층**
- `BaseTask`(`HV/envs/base_task/base_task.py`): 시뮬레이터 생성, dt = decimation × sim_dt 계산.
- `LeggedRobotBase`(`HV/envs/legged_base_task/legged_robot_base.py`, 이하 `lrb`): 지연, PD, RFI, 푸시, 종료, 보상, 커리큘럼, 관측 파서.
- `LeggedRobotLocomotion`(`HV/envs/locomotion/locomotion.py`, 이하 `loco`): 명령과 트래킹/발 보상.

**보상 정의 방식**
- yaml의 `reward_scales.X`가 메서드 `_reward_X`에 자동으로 묶입니다. 값이 0인 항목은 제거됩니다.
- 나머지 가중치에는 dt를 곱합니다(`lrb:141`). 그래서 가중치는 초당 단위이고, Isaac Lab의 weight×dt 관례와 그대로 비교할 수 있습니다.

**관측 정의 방식**
- `obs_dict`의 키마다 `_get_obs_<key>()`를 부르고, 스케일과 잡음을 적용합니다.
- **키를 알파벳순으로 정렬해 이어 붙입니다**(`lrb:521-526`). 히스토리 그룹 안도 정렬됩니다(`lrb:806-840`).
- 차원은 `HV/utils/helpers.py:29 pre_process_config`가 자동으로 계산합니다.

**액션 경로** (`lrb:185-209, 528-559`)
- clip ±100(사실상 무제한) → 지연 큐 → `q_tgt = 0.25·a + q0` → Python에서 PD 토크 계산 → RFI 잡음 추가 → URDF effort 한계로 clip 순서입니다.
- 자산은 effort 모드(`default_dof_drive_mode: 3`)입니다.

**DR 구현 위치**
- 에피소드 단위(kp/kd, RFI, 지연)는 공통 코드 `lrb:700-716`에 있습니다.
- 물성(마찰, 질량, CoM)은 **백엔드마다 따로 구현**돼 있습니다. Genesis에는 DR이 아예 없습니다.

### 2.2 백엔드 간 불일치 (정적 분석, 실행 미확인)

1. **PD 갱신율이 다릅니다.**
   - IsaacGym은 서브스텝마다 `refresh_dof_state_tensor`를 호출합니다(`HV/simulator/isaacgym/isaacgym.py:495-499`). 즉 PD가 200 Hz로 돕니다.
   - IsaacSim은 `self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]`(`isaacsim.py:658`)로 리스트 인덱싱해 사본을 만들고, 정책 스텝당 한 번만 갱신합니다. `simulate_at_each_physics_step`(`:679-692`)에서는 갱신하지 않습니다.
   - 그래서 4개 서브스텝 동안 낡은 q, q̇로 토크를 계산하고, 결과적으로 50 Hz 토크 ZOH가 됩니다 (추정). Genesis도 같습니다(`genesis.py:249-283, 356-360`).
2. **`randomize_friction`의 의미가 백엔드마다 다릅니다.**
   - IsaacGym: 로봇 접촉 형상 마찰을 [0.5,1.25] 절대값으로, 64버킷에서 뽑습니다(`isaacgym.py:266-292`).
   - IsaacSim: **관절 마찰 계수에 배율**을 곱합니다(`mdp.randomize_joint_parameters`, `isaacsim.py:129-139`).
   - Genesis: 없습니다.
3. **링크 질량 DR 범위가 다릅니다.** IsaacGym은 `randomize_link_body_names`에 있는 링크만 바꾸고, IsaacSim은 `body_names=".*"`로 전 링크를 바꿉니다(`:118-127`).
4. **armature와 관절 마찰은 IsaacSim에서만 쓰입니다.** `dof_armature_list`/`dof_joint_friction_list`(G1-12dof 0.01/0.05)는 IsaacSim만 사용하고(`isaacsim.py:288-315`), IsaacGym은 asset `armature: 0.`입니다.
5. **IsaacSim CoM DR이 `"torso_link"`로 하드코딩돼 있습니다**(`:141-160`). torso가 없는 G1-12dof에서는 실패할 것으로 봅니다 (추정).
6. **Genesis 속도 좌표계가 섞입니다.** `robot_root_states`의 속도는 이미 몸체 좌표계인데, `lrb:265-267`에서 다시 역회전합니다. yaw가 0이 아니면 이중 회전이 됩니다 (추정).

→ README의 "IsaacGym에서 학습 → Genesis에서 평가" 데모는 물리뿐 아니라 제어기 이산화까지 달라진 비교입니다 (추정).

---

## 3. 보상 (보행)

### 3.1 항목과 수식 [확인]

| 항목 | 수식 | 위치 |
|---|---|---|
| tracking_lin_vel | exp(−‖v_cmd,xy − v_xy‖²/0.25) | loco:80-83 |
| tracking_ang_vel | exp(−(ω_cmd,z − ω_z)²/0.25) | loco:85-88 |
| penalty_lin_vel_z | v_z² | loco:92-94 |
| penalty_ang_vel_xy | ω_x² + ω_y² (pelvis, 몸체 좌표) | loco:96-98 |
| penalty_ang_vel_xy_torso | torso 링크 ω_xy² | loco:100-104 |
| penalty_orientation | g_x² + g_y² | lrb:657-659 |
| base_height | (z_root − h_des)², 지형 보정 없음 | loco:149-153 |
| penalty_torques / dof_vel / dof_acc | Στ² / Σq̇² / Σ(Δq̇/0.02)² | lrb:641-651 |
| penalty_action_rate | Σ(a_prev − a)², a 단위 = 0.25 rad | lrb:653-655 |
| limits_dof_pos / vel / torque | 소프트 한계(90%) 초과량 합 | lrb:663-676 |
| penalty_slippage | Σ‖v_foot‖·[‖F_foot‖ > 1 N] | lrb:678-681 |
| feet_air_time | Σ(t_air − 0.5)·[첫 접지], ‖cmd_xy‖ > 0.1일 때만 | loco:113-124 |
| penalty_feet_height | clip(min_foot\|h − 0.2\| − 0.02, 0), **명령 게이트 없음** | loco:192-197 |
| penalty_feet_ori | Σ_foot ‖g_xy(발 좌표)‖, 제곱이 아닌 노름 | loco:142-147 |
| feet_heading_alignment | Σ\|wrap(ψ_foot − ψ_root)\| | loco:161-177 |
| penalty_close_feet_xy | [발 간격 < 0.15 m] | loco:199-204 |
| penalty_hip_pos | hip roll·yaw의 q² 합 | loco:155-159 |
| termination | reset ∧ ¬timeout. 보행 설정에서는 모두 주석 처리(미사용) | lrb:637-639 |

### 3.2 가중치 (초당) [확인]

| 항목 | G1-12dof | G1-23dof | H1 |
|---|---|---|---|
| tracking lin / ang | 1.0 / 0.5 | 1.0 / 0.5 | 1.0 / 0.5 |
| lin_vel_z | −2.0 | −2.0 | −2.0 |
| **ang_vel_xy** | **−0.05** (`:14`) | **−1.0** | **−1.0** (`:14`) |
| ang_vel_xy_torso | – | −1.0 (`:36`) | – |
| orientation | −1.0 | −1.0 | −1.0 |
| base_height | −10 (0.728 m) | −10 (0.78) | −100 (0.98) |
| torques | – | −1e-5 | −1e-5 |
| dof_vel / dof_acc | −1e-3 / −2.5e-7 | – / −3.5e-8 | – / −3.5e-8 |
| **action_rate** | **−0.1** (주석 "−0.5") | −0.01 | −0.01 |
| slippage | −1.0 | −1.0 | – |
| feet_ori | −1.0 | −2.0 | −2.0 |
| limits_dof_pos | −5 | −10 | −10 |
| feet_air_time | +1.0 | +1.0 | +1.0 |
| feet_height (목표) | −5 (0.2 m) | −2.5 (0.12) | −5 (0.2) |
| heading / close_feet / hip_pos | −0.3 / −2 / −1 | −0.3 / −10 / −1 | – |

- `only_positive_rewards: False`(세 설정 모두).
- 페널티 커리큘럼은 G1-12dof만 켜져 있습니다.
- 출처: ASAP에 같은 H1 설정이 `reward_h1_locomotion_19dof_2real.yaml`이라는 이름과 `set_reward: Unitree_RL GYM EXACT REWARD` 표기로 들어 있습니다. H1 가중치는 unitree_rl_gym 계열입니다 [확인].

### 3.3 페널티 커리큘럼 [확인]

**동작** (`lrb:608-631` + 곱하는 부분 `lrb:486-488`)
- 리셋 이벤트가 있는 스텝마다 평균 에피소드 길이를 EMA로 갱신합니다(가중치 = 리셋 수/10000).
- 평균이 400스텝(8 s) 미만이면 scale × (1 − degree), 700스텝(14 s) 초과면 scale × (1 + degree)로 바꾸고 [0,1]로 자릅니다.
- `reward_penalty_reward_names`에 든 항목에 이 scale을 곱합니다.
- G1-12dof 설정(`reward_g1_12dof_locomotion.yaml:53-88`): 초기 0.1, degree 5e-5.
- 목록에는 `feet_air_time`(양의 보상), base_height, ang_vel_xy, action_rate 등 거의 모든 항목이 들어 있습니다.

**램프 시간 [계산]**: 4096 env면 거의 매 스텝 리셋이 있습니다. 0.1→1.0까지 ln10/ln(1+5e-5) ≈ **46,053스텝 ≈ 1,919 iteration이 최소로 필요**합니다.

**사용 근거**: README의 Genesis 학습 명령만 `reward_penalty_curriculum=True, reward_initial_penalty_scale=0.5`를 추가합니다. ablation은 없습니다.

(추정) 종료 페널티가 없고 음의 합계를 허용하는 설정이라, 초기에 스텝 보상이 음수가 되어 "빨리 넘어지는 게 이득"인 해로 빠지는 것을 막는 역할도 합니다.

### 3.4 요동 관점 해석 [계산]

- **상태 게이트가 없습니다.** 모든 페널티가 상시 적용됩니다. 발 관련 항목만 접촉 조건이 붙습니다. 균형을 잃었을 때 페널티를 끄는 로직은 없습니다.
- **피치 정현 요동의 초당 비용**(ω² 가중치 기준, 참고로 tracking 보상 최대치는 1.5/s):

| 요동 | mean ω² | G1-12dof (−0.05) | H1 (−1.0) | 우리 (−0.1) |
|---|---|---|---|---|
| 0.5 Hz 3° | 0.0135 | −0.0007 | −0.0135 | −0.0014 |
| 0.5 Hz 6° | 0.054 | −0.0027 | −0.054 | −0.0054 |
| 2 Hz 2° | 0.096 | −0.0048 | −0.096 | −0.0096 |
| 2 Hz 4° | 0.385 | −0.019 | −0.385 | −0.038 |

- **ω² 페널티는 f²A²에 비례합니다.** 그래서 0.5 Hz 저주파 요동은 H1 가중치로도 tracking 최대치의 4% 이하로, 사실상 공짜입니다.
- orientation(−1.0)은 0.5 Hz 6°에서도 −0.0055/s에 그칩니다. 저주파 요동은 사실상 어느 항도 벌하지 않습니다.
- **이벤트 보상의 반농사 구조**: `feet_air_time`은 (t_air − 0.5 s)를 첫 접지 때만 줍니다. 0.5 s보다 짧게 반복되는 이벤트는 음수가 됩니다.

---

## 4. 도메인 랜덤화와 제어

### 4.1 DR 항목

| 항목 | domain_rand_base.yaml | 구현과 주기 |
|---|---|---|
| push | True, 5–9 s 간격(randint(5,10)), 베이스 xy 속도를 U(−1,1) m/s로 **설정**(더하지 않음) | lrb:96, 282-287, 719-728. 카운터가 에피소드 리셋과 무관 |
| base CoM | ±0.1 m (x, y, z) | IsaacGym: torso가 없으면 pelvis 링크에 더함, 생성 시 1회(isaacgym.py:341-361) |
| link mass | ×[0.8,1.2] | 생성 시 1회, pelvis/hip×3/knee만(발목 제외) (:364-375) |
| PD gain | kp·kd 각각 ×[0.75,1.25], 관절·env별 | 에피소드마다(lrb:704-706), 토크식 lrb:544 |
| friction | [0.5,1.25] | IsaacGym 64버킷, 로봇 형상 전체, 지면은 1.0 고정 |
| base mass | False | 코드가 `raise Exception`(isaacgym.py:378-382) |
| torque RFI | τ += U(−1,1) × 0.1 × U[0.5,1.5] × τ_max, 매 물리 서브스텝(200 Hz 백색잡음) | lrb:552-553, 스케일은 에피소드마다 |
| ctrl delay | [0,2] **정책 스텝** = 0/20/40 ms (주석 "max real delay is 90ms") | lrb:112-115, 193-198, 711-716. env별로 에피소드마다 뽑아 고정, 지터 없음, 리셋 시 큐를 0(기본자세 목표)으로 초기화 |

**없는 모델**: 관측/센서 지연, 백래시·데드존, 모터 토크-속도 곡선, 관절 위치 오프셋.

**설정 파일 결함** [확인]
- `domain_rand_base.yaml:12-13`의 `x : [-0.1., 0.1.]`는 오타입니다. PyYAML로 읽으면 문자열이 되고, `np.random.uniform`에서 ValueError가 납니다. 그래서 이 파일을 그대로 쓰면 IsaacGym 환경 생성이 실패할 것으로 봅니다 (추정).
- `NO_domain_rand.yaml:14`의 `friction_range : [-0.6, 1.2]`는 음수 마찰입니다. 이 파일에서는 꺼져 있습니다.

**README의 모든 학습 명령은 `+domain_rand=NO_domain_rand`입니다.** 즉 공개 레시피는 DR 없음 + 관측 잡음 0입니다 [확인].

### 4.2 제어와 시뮬레이션 설정

- **시간**: sim 200 Hz(dt 0.005), decimation 4 → 정책 50 Hz. PhysX TGS, position iteration 4, velocity iteration 0(`HV/config/simulator/*.yaml`).
- **PD**: τ = kp·s_p·(0.25a + q0 − q) − kd·s_d·q̇ (lrb:544).
- **게인**
  - G1-12dof(`g1_12dof.yaml:112-129`): kp hip 100 / knee 200 / **ankle 20**, kd 2.5 / 5 / **ankle pitch 0.2, roll 0.1**.
  - H1: kp 200 / 300 / ankle 40, kd 5 / 6 / 2.
- **기본자세**: G1 hip −0.1, knee 0.3, ankle −0.2 rad. H1 −0.4 / 0.8 / −0.4.
- **리셋**(lrb:736-783)
  - 관절: q = q0 × U(0.5,1.5). 곱셈이라 q0 = 0인 관절(hip roll/yaw, ankle roll)은 무작위화되지 않습니다.
  - 루트: 6자유도 속도를 U(−0.5,0.5)로(각속도로는 ±28.6°/s).
  - G1은 0.8 m 높이에서 스폰합니다.

---

## 5. 관측 설계

**actor에 히스토리를 넣는 설정은 `obs/loco/leggedloco_obs_history_wolinvel.yaml` 하나뿐입니다.**
- actor 항목: base_ang_vel, projected_gravity, command_lin_vel(2), command_ang_vel(1), dof_pos(q−q0), dof_vel, actions(직전 정책 출력, 지연 적용 전).
- 여기에 `short_history`로 위 7개 항목 × 5프레임을 붙입니다.
- G1-12dof 기준 45 + 225 = 270차원 [계산].
- critic은 actor와 같고 base_lin_vel(×2.0)만 더합니다. 다른 특권 정보(마찰, 질량, 지연, 접촉력)는 없습니다.

**히스토리 시점과 초기화 [확인]**
- 관측을 계산한 뒤에 버퍼에 넣습니다(lrb:249-250). 그래서 현재 프레임 + t−1…t−5, 총 6프레임(120 ms)입니다.
- 리셋 시 0으로 채웁니다(`HV/envs/env_utils/history_handler.py:29-34`). 에피소드 시작에 콜드스타트 과도가 생깁니다.

**항목별 가변 길이 히스토리가 동작합니다** [확인]
- HistoryHandler는 그룹들 중 최대 길이로 버퍼를 잡습니다(history_handler.py `__init__`).
- 정의만 되고 쓰이지 않는 예시가 있습니다.
  - `leggedloco_obs_singlestep_withlinvel.yaml:31-36`의 `history_long: {dof_pos: 20, dof_vel: 2, actions: 15}`
  - `leggedloco_obs_singlestep_wolinvel.yaml:30-39`의 전 항목 20프레임

**스케일과 정규화**
- ang_vel 0.25, dof_vel 0.05, lin_vel 2.0, 나머지 1.0. clip ±100.
- empirical 정규화는 없습니다.

**잡음: 모든 보행 설정에서 0이고, 잡음 코드에 결함이 있습니다.**
- `HV/utils/helpers.py:94`: `actor_obs*obs_scale + (torch.randn_like(actor_obs)*2. - 1.)*obs_noise`
  - 균일분포가 아니라 평균 −noise, 표준편차 2·noise인 가우시안입니다.
  - 스케일을 곱한 뒤에 더합니다.
- ASAP는 이것을 고쳤습니다(`ASAP_ref/humanoidverse/utils/helpers.py:97-101`, 주석 "use rand_like ... instead of randn_like"). 고친 식은 `(obs + U(−1,1)·noise)·scale`입니다.
- 현재값과 히스토리 사본이 잡음을 독립적으로 뽑습니다(lrb:511-517). 같은 값이 히스토리로 밀려가는 실물과 다릅니다. 잡음이 0이면 영향은 없습니다.

**순서 계약**: obs와 history 모두 알파벳순으로 이어 붙입니다. 배포 러너도 같은 순서를 지켜야 합니다.

---

## 6. PPO, 커리큘럼, 종료 조건

**PPO** (`HV/config/algo/ppo.yaml`)
- epochs 5, minibatch 4, clip 0.2, γ 0.99, λ 0.95, value coef 1.0, entropy 0.01.
- actor/critic lr 1e-3, Adam을 따로 씁니다.
- adaptive KL: 목표 0.01, lr을 ÷1.5 또는 ×1.5, 범위 [1e-5, 1e-2], 액터·크리틱 동시 조정(ppo.py:402-418).
- grad clip 1.0(네트워크별), value clip 사용, 24스텝/env, `init_noise_std 0.8`(log가 아닌 원값 파라미터, 상태 무관), `init_at_random_ep_len`.
- MLP 512-256-128 ELU.
- 타임아웃 부트스트랩: r += γV·timeout(ppo.py:260-262). advantage는 배치 전체로 정규화합니다(:346-348).
- 대칭 손실과 RNN은 없습니다.

**시드**: `seeding`을 import만 하고 호출하지 않습니다(train_agent.py:49, 97-102 주석). 재현성이 없습니다.

**커리큘럼**: 페널티 스케일(3.3절)뿐입니다. 지형 커리큘럼은 False, 명령 커리큘럼은 주석 처리돼 있습니다.

**명령**
- 10 s마다 새로 뽑습니다. vx, vy ∈ U[−1,1], heading ∈ U[−π,π], ω_cmd = clip(0.5·heading 오차, ±1).
- ‖v_xy‖ < 0.2이면 0으로 둡니다. 비율은 약 3.1%입니다 [계산] (loco:39-64).
- 평가 모드에서는 명령이 0으로 고정됩니다(loco:72-76).

**종료**
- |g_x| > 0.8 또는 |g_y| > 0.8, 약 53° 기울기(lrb:336-339).
- 접촉 종료는 꺼져 있습니다(주석 "problematic in isaacgym when g1+terrain"). 높이 종료도 꺼져 있습니다.
- 에피소드는 20 s(1000스텝)이고 종료 페널티는 없습니다.

---

## 7. 우리에게 추출 가능한 것 (우리 문제에 대한 기대 효과 순, 순위는 제 판단)

### ① 지연 창을 덮는 긴 액션 히스토리 + 크리틱 특권 입력 — 지연·요동 대응

- **근거 코드**
  - HistoryHandler의 항목별 길이 설정(예: `leggedloco_obs_singlestep_withlinvel.yaml:31-42`, actions 15프레임).
  - 지연 큐 `action_queue`/`action_delay_idx`(lrb:112-115, 193-198).
- **왜 필요한가**: 실측 320–340 ms는 16–17틱입니다. 우리 관측 창은 5프레임(100 ms)이라, 이미 보냈지만 아직 도달하지 않은 목표를 정책이 볼 수 없습니다.
- **적용**
  - 액션 항목만 17–20프레임으로 늘립니다. 17프레임이면 12×17 = 204차원, 현재보다 +144차원입니다.
  - 크리틱에는 현재 실제로 적용 중인 지연된 액션(`actions_after_delay`)과 지연 인덱스를 특권 관측으로 줍니다.
- **기대**: "방향은 맞는데 늦는" 보정의 위상을 보상 (추정).
- **주의**
  - HV는 이 목적으로 써 본 적이 없습니다. 지연 DR은 40 ms 이하, actor 히스토리는 5프레임입니다.
  - 차원이 늘어나므로 MLP 대신 1D-conv/GRU 인코더도 대안입니다.
  - 필요한 대조군: 같은 지연 DR(9–14틱)과 seed에서 액션 히스토리 5 vs 17틱을 비교하고, 요동 주파수·감쇠·지지 제거 후 5 s 자립률로 판정합니다.

### ② 상태 게이트 대신 "상시 페널티 + 생존 기반 전역 스케일" — 요동이 싼 보상 구조 대응

- **근거 코드**: lrb:608-631, 486-488. 설정 `reward_g1_12dof_locomotion.yaml:53-88`.
- **적용**
  - 균형을 잃었을 때 꺼지는 자세/정숙 페널티를 상태와 무관한 상시 항목으로 바꿉니다.
  - 학습 초기의 완화는 게이트 대신 전역 스칼라(0.1→1.0)로 합니다.
  - 승급 지표는 에피소드 길이 대신 "지지 제거 후 5 s 무낙상 비율" 같은 우리 목표 지표로 바꿉니다.
- **주의**
  - HV에는 ablation도 실물 증거도 없습니다(이슈 #4).
  - 스칼라는 전 env 공통이고, 리셋 빈도에 따라 속도가 달라집니다.
  - 최종 체크포인트가 scale 1.0에 도달했는지 `penalty_scale` 로그로 확인해야 합니다.
  - 게이트만 바꾼 대조 실험이 필요합니다.

### ③ 회복 보너스의 반농사 구조 — feet_air_time 패턴 (loco:113-124)

- **적용**: 보너스를 "(직전 이벤트 이후 조용히 유지한 시간 − T_min)" 꼴로 바꿉니다. 반복 흔들림으로 보너스를 모으는 경로가 음수가 됩니다 (추정).
- **주의**: "조용함"의 임계가 다시 게이트가 됩니다. 임계는 넉넉히 잡고 ④와 함께 씁니다.

### ④ 조밀한 exp-커널 정숙 보상 — ASAP 모션트래킹(HV 파생)

- **근거 코드**
  - `ASAP_ref/humanoidverse/envs/motion_tracking/motion_tracking.py:617-639`: `exp(−mean‖Δv‖²/σ)`.
  - 가중치: `.../reward_motion_tracking_dm_2real.yaml:15-18`에서 body ang vel 0.5, joint vel 0.5, σ = 1.0.
- **적용**: 참조를 정지(0)로 두면 항상 켜진 유계 양의 정숙 보상이 됩니다. 거의 발동하지 않는 임계형 settle 보너스를 대체할 수 있습니다.
- **시간평균 값 [계산]** (단일축 피치, σ 단위 (rad/s)²):

| 요동 | σ = 1.0 | σ = 0.25 | σ = 0.1 |
|---|---|---|---|
| 0.5 Hz 6° | 0.95 | 0.82 | 0.63 |
| 2 Hz 2° | 0.91 | 0.71 | 0.48 |
| 2 Hz 4° | 0.71 | 0.36 | 0.21 |

  - ASAP 원식은 xyz와 body 평균이라 유효 σ가 약 3배입니다. 그래서 2 Hz 4° 요동에서도 0.88로, 정지 상태(1.0)와 거의 구분하지 못합니다.
- **주의**
  - 0.5 Hz 요동은 속도 커널로 잡기 어렵습니다. 자세(피치/CoM-지지면) 항이 맡아야 합니다.
  - "조용하지만 서지 못하는" 정책 위험이 있으니 자립률과 함께 판정합니다.

### ⑤ action-rate 강화는 양날의 검

- **근거**: ASAP `reward_motion_tracking_dm_2real.yaml:22`의 `-0.5 # need to be tuned up to -0.5 for real robot`. HV 보행 설정은 −0.01~−0.1입니다.
- 고주파 채터 억제에는 유효합니다. 우리 march_v3 경험과도 맞습니다.
- 그러나 320 ms 지연 루프에서는 위상 지연을 더해 0.5–2 Hz 요동을 키울 수 있습니다 (추정). 이 가중치만 바꾼 A/B가 필요합니다.

### ⑥ 노름형 발 평탄·접촉 발 속도 항

- **근거 코드**: `penalty_feet_ori`(loco:142-147), `penalty_slippage`(lrb:678-681).
- 노름형은 작은 진폭에서도 기울기가 남습니다. 제곱형은 0 근처에서 기울기가 사라집니다.
- 발 링크 원점이 발바닥 위에 있으므로, 뒤꿈치/발끝을 축으로 발이 돌기 시작할 때도 속도가 생겨 벌을 받습니다 (추정). 0908–0910 뒤꿈치 회전 양상과 관련이 있습니다.
- **주의**: 우리 시뮬이 실물의 강체 회전을 재현하지 못한 상태라, 플랜트가 피벗을 재현할 때만 의미가 있습니다.

### ⑦ 토크 RFI로 시뮬을 거칠게 (lrb:552-553)

- **적용**: τ_max를 무엇으로 잡을지(AK70-10 정격 vs 피크)를 먼저 정해야 합니다.
- **주의**: 200 Hz 백색잡음은 0.5–2 Hz 대역 에너지가 작습니다. 요동 재현과는 거의 무관하고, 백래시·정지마찰의 대체물도 아닙니다 (추정).

### ⑧ 발목 저게인 설계와의 대비 (확신 낮음)

- G1의 발목 게인은 kp 20 / kd 0.1–0.2로, 무릎 200/5의 1/10입니다(`g1_12dof.yaml:112-129`). H1도 40/2입니다. 우리는 전 관절 kp 150입니다.
- 백래시 ±4° + 고게인 + 지연은 고전적인 한계 사이클 조건입니다 (추정).
- 로봇이 달라 수치를 옮길 수는 없습니다. 백래시 모델을 넣은 시뮬에서 발목 게인만 바꾼 대조가 먼저입니다.

### ⑨ Isaac Lab 관절 마찰 DR API 사용 예

- **근거 코드**: `isaacsim.py:129-139`(`randomize_joint_parameters`, scale [0.5,1.25], startup), 액추에이터 `IdealPDActuatorCfg(stiffness=0, damping=0, armature, friction)`(`:306-317`).
- 발목 정지마찰 근사의 출발점으로 쓸 수 있습니다. Isaac Lab 2.x에서는 모듈 이름이 `isaaclab.*`로 바뀌었습니다.
- **주의**: PhysX 관절 friction은 무차원 계수로 보입니다. N·m 단위의 쿨롱 토크가 아닐 것으로 봅니다 (추정). 정량 모델은 액추에이터 단에서 토크 수준으로 구현하는 편이 낫습니다.

### 적용 불가 / 가져오면 안 되는 것

- **HV 보행 보상 세트 자체.** 서기 태스크가 없습니다.
  - G1 발 접촉구는 발목 원점 아래 0.03 m, 반경 5 mm라, 서 있을 때 발 링크 높이는 약 0.035 m입니다.
  - 그래서 `penalty_feet_height`(게이트 없음, 목표 0.2 m)가 두 발이 다 지면에 있으면 약 −0.73/s를 벌합니다 [계산].
  - 명령이 0이어도 제자리걸음을 유도할 것으로 봅니다 (추정). sim2real도 메인테이너가 미검증이라고 인정했습니다.
- **지연 DR 0–40 ms.** 메커니즘은 우리와 같고, 우리 G12(9–14틱)가 이미 앞서 있습니다. 지터와 관측 지연은 없습니다.
- **관측 잡음 코드**(helpers.py:94)는 편향이 있습니다. 쓰려면 ASAP 수정판을 쓰십시오.
- **`domain_rand_base.yaml`의 DR 수치**는 검증된 레시피가 아닙니다. 오타가 있고, 공개 학습 명령은 DR을 쓰지 않습니다.
- **PPO**: rsl_rl과 사실상 같습니다. 새로운 것이 없습니다.
- **"한 줄로 시뮬 교체" sim2sim**: 백엔드마다 PD 갱신율과 DR 의미가 달라, 강건성의 증거로 쓰기 어렵습니다.
- **HV에 없는 모델**: 백래시, 무릎 탄성 처짐, 토크-속도 곡선.
- **280 ms 소프트웨어 큐잉 자체를 줄이는 방법**: HV 범위 밖입니다(sim2real 쪽).

---

## 8. 부록: 파생 저장소의 서기 설계 (FALCON, 범위 밖 참고)

**FALCON**: `FALCON_ref/humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml`, 환경 `envs/decoupled_locomotion/decoupled_locomotion_stand_ma.py`.

명령 비트 `commands[:,4]`(0 = stance)로 **상태가 아니라 모드로 게이트**합니다. 그래서 요동 중에도 페널티가 꺼지지 않습니다.

| 항목 | 가중치 | 내용 | 위치 |
|---|---|---|---|
| penalty_stance_dof | −1e-3 | 하체 Σq̇² (stance) | :550-552 |
| penalty_contact | −4.0 | stance에서 한 발이라도 비접촉이면 1 → **회복 스텝과 충돌** | :515-528 |
| penalty_stand_still | −0.15 | stance에서 비접촉 발 존재 | :605-608 |
| penalty_stance_root | −5.0 | 골반의 발 중점 대비 횡오프셋 \|y\| (골반 좌표) | :574-582 |
| penalty_stance_symmetry | −0.5 | 좌우 관절 대칭 | :610-615 |
| penalty_contact_no_vel | −0.2 (stance ×10) | 접촉 발 속도² | :584-593 |
| tracking_stance_base_height | +4.0 | exp(−\|Δh\|/0.05) | `..._diff_force.py:621-624` |
| penalty_torso_orientation | −1.0 | 상시 Σ\|g_xy\| | `..._stand_height_waist_wbc_ma.py:418` |

- 그 밖의 가중치: ang_vel_xy −0.05(pelvis), torso −1.0, action_rate −0.1, termination −250.
- 페널티 커리큘럼: 초기 0.5, degree 1e-5, 임계 40/210스텝.
- DR(`domain_rand_rl_gym.yaml`): 지연 [0,1]틱, kp/kd ±10%, 마찰 [0.25,1.25], 추가 질량 [−1,3] kg, 푸시 끔.

**ASAP**(`reward_motion_tracking_dm_2real.yaml`)
- termination −200, limits_torque −5(soft 0.825), 한계 소프트 커리큘럼(1.15→0.95).
- 관측 잡음: ang_vel 0.3, gravity 0.2, dof_pos 0.01, dof_vel 1.0 + 4프레임 히스토리(`obs/motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history.yaml:67-79`).

ASAP와 FALCON에는 `sim2real/` 디렉터리가 있지만 다른 에이전트 범위라 보지 않았습니다.
