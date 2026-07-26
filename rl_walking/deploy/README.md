# rl_walking/deploy — RL 보행 정책 실행기 (실물/단독 PC용)

**시뮬 전용 — 실물 송신은 승윤님 결정.** 이 디렉토리의 코드는 CAN/모터에 아무것도
보내지 않는다. `PolicyRunner`는 관측 → 목표관절각 계산만 하며, 계산 결과를 실물
브리지(Stage8 노드)에 연결하는 것은 별도의 명시적 결정 후에만 한다.

## 구성

| 파일 | 역할 |
|---|---|
| `policy_runner.py` | (Stage1/flat) 45→12 정책 추론 + 아핀(×0.25+기본자세) + 슬루(±4°/틱) 실행기. `--selftest` 내장 |
| `policy_runner_stage4.py` | (Stage4) 245→12 실행기 — 항별 이력 5 링버퍼 + 게이트 클록 + 발 접촉 입력. `--selftest` 내장 |
| `sim_validate_runner.py` | 45차원판 폐루프 시뮬 검증 (Isaac Lab 필요, GPU) |
| `sim_validate_runner_stage4.py` | Stage4판 폐루프 시뮬 검증 — 관측 245·목표각을 env 버퍼와 매 틱 대조 (Isaac Lab 필요, GPU) |

- 의존성: `numpy` 필수, 추론은 `onnxruntime`(우선) 또는 `torch`(폴백) — 자동 선택.
  isaaclab 불필요 → 실물 PC/Jetson에서 단독 실행 가능.
- 체크포인트: `logs/rsl_rl/biped12_flat/2026-07-25_21-59-19/exported/{policy.onnx, policy.pt}`
  / Stage4: `logs/rsl_rl/biped12_stage4/2026-07-26_14-53-03/exported/{policy.onnx, policy.pt}`
- 호출 주기: **50Hz** (호출자가 보장 — 러너 내부에 타이머 없음. Stage4는 클록이
  tick×0.02s로 시간을 재구성하므로 주기 이탈 = 위상 이탈이라 더 엄격히 지킬 것)

## 사용

```python
from rl_walking.deploy.policy_runner import PolicyRunner

r = PolicyRunner()          # exported/ 자동 로드
r.reset()                   # last_action=0, 슬루 기준점은 첫 step의 qpos로
q_cmd = r.step(gyro, gravity, cmd, qpos, qvel)   # → 목표관절각 12개 [rad]
```

자가검증(GPU 불필요):

```bash
# torch가 있는 인터프리터로 (예: Isaac Sim 파이썬)
/home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/deploy/policy_runner.py --selftest
```

## 실물 연결 시 관측 소스 매핑표

`step()`의 각 인자에 들어갈 실물 측정값 (모두 **생측정값** — 학습 노이즈 Unoise는
corruption 전용이므로 실행기에서 노이즈 추가 금지):

| step() 인자 | 크기 | 실물 소스 | 변환/주의 |
|---|---|---|---|
| `gyro_rad_s` | 3 | **iAHRS 자이로** (바디프레임 각속도) | [rad/s]. IMU 축을 베이스 프레임(+X전방, +Y좌, +Z상)에 정렬 필수 |
| `gravity_unit` | 3 | **iAHRS 자세 쿼터니언**에서 유도 | `R_wb^T @ (0,0,-1)` — 중력 **단위벡터**의 베이스 투영. 직립=(0,0,-1). 가속도계 원시값(m/s²) 아님 |
| `cmd` | 3 | 조작 명령 (조이스틱/음성 등) | (vx[m/s], vy[m/s], wz[rad/s]) 베이스 프레임. 학습분포(본 체크포인트 params/env.yaml 실측): vx −0.4~0.8, vy ±0.25, wz ±1.0, 서기명령 10%. **시작은 (0,0,0) 스탠딩부터** |
| `qpos_rad` | 12 | **모터 엔코더 절대각** (MIT 모드 피드백) | [rad], **JOINT_ORDER 순서** — 러너가 내부에서 기본자세를 빼므로 절대각 그대로 입력 |
| `qvel_rad_s` | 12 | **모터 엔코더 각속도** | [rad/s], JOINT_ORDER 순서 |

반환값 `q_cmd`(12, rad, JOINT_ORDER 순)는 **슬루 적용 후 목표각** — 그대로 위치명령으로
PD(kp150/kd5, MIT 모드) 추종. 토크한계는 학습과 동일하게: AK70 10관절 25Nm,
발목롤 AK45-36 24Nm(속도한계 5rad/s).

### 관절 순서 주의 (제일 잘 틀리는 곳)

`JOINT_ORDER`는 Isaac Lab BFS 순서(**좌우 인터리브**: left_hip_f, right_hip_f, …)로,
하드웨어맵 motor_id(좌1-6/우7-12 순차)와 **다르다**. 반드시 관절명 기준으로 매핑할 것
(docs/RL보행_이식가이드.md §2).

### 슬루 이중 적용에 대해

러너 내부에 ±4°/틱 슬루가 있고, 실물 Stage8 노드에도 동일한
`NODE_SLEW_DEG_PER_TICK=4°`가 있다. 러너 출력은 이미 틱당 |Δ|≤4°이므로 같은 50Hz
틱·같은 시작점(현재 관절각)에서 노드 슬루는 **항등 통과** — 이중 적용 무해.
러너 슬루를 유지하는 이유: 학습 파이프라인(`rl_walking/actions.py`)과 1:1 수식 정합
+ 노드 없이 시뮬/재생 시 동일 거동.

### 액션 지연은 넣지 않는다

이 체크포인트는 `max_delay_steps=1`(0~1틱 랜덤 지연 DR)로 학습됐고, 실물 자연
루프지연(제어기 ~15-19ms + 서보 ≈1틱@50Hz)이 그 역할을 이미 한다 — 실행기에
인위적 지연 버퍼를 추가하지 말 것.

## Stage4 실행기 (`PolicyRunnerStage4`) — 관측 245 (이력 5 · 클록 · 접촉)

```python
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4

r = PolicyRunnerStage4()    # biped12_stage4/2026-07-26_14-53-03/exported 자동 로드
r.reset()                   # tick=0, 이력 비움, last_action=0, 슬루 기준점 재초기화
q_cmd = r.step(gyro, gravity, cmd, qpos, qvel, contact2)   # → 목표관절각 12개 [rad]
```

45차원판과의 차이는 **관측뿐** — 액션 파이프라인(아핀·슬루·PD·관절순서·지연 금지)은
전부 동일하고 위 표/주의사항이 그대로 적용된다. `step()` 인자도 앞 5개는 45차원판
매핑표와 동일하고 `contact2` 하나가 추가된다.

관측 245 = 단일 프레임 49 항목 × 이력 T=5. **항 블록 구조**(프레임 인터리브 아님):
항마다 `[프레임0(最古) … 프레임4(最新)]`이 연달아 놓인다 — 정확한 오프셋은
`policy_runner_stage4.py` 모듈 docstring의 표 참조. 러너가 내부에서 전부 관리하는
것: 이력 링버퍼(리셋 후 첫 스텝은 isaaclab CircularBuffer와 동일하게 5프레임
전체 백필), 게이트 클록(φ = 2π×1.4Hz×tick×0.02s, `reset()`시 tick=0 — 외부 시계
불필요), last_action. 호출자는 매 틱 생측정값 6개만 넣으면 된다.

### RA30P 발바닥 힘센서 → `contact2` 입력 매핑

| 항목 | 값 |
|---|---|
| 크기/순서 | 2 — **[왼발, 오른발]** 고정 |
| 변환 | `contact_L = 1.0 if 왼발 법선힘 노름 > 5.0 N else 0.0` (오른발 동일) |
| 문턱 5 N | 학습 관측 `mdp_stage4.feet_contact_binary(threshold=5.0)`과 동일 — 변경 금지 |
| 값 검증 | 러너가 0/1 이외 값을 거부(ValueError) — **생힘[N]을 그대로 넣으면 즉시 에러**. 반드시 이진화 후 입력 |
| 샘플링 | 50Hz 틱마다 그 시점 측정값으로 이진화 (러너가 이력 5틱을 내부 보관하므로 별도 필터/이력 불필요) |
| 센서 유의 | RA30P는 단축 로드셀 — 발바닥 합력 기준으로 문턱 비교. 노이즈로 5 N 근방에서 채터링하면 히스테리시스(예: 상승 5 N/하강 3 N)는 **학습과 어긋나므로 넣지 말 것** (학습도 생비교) |

행잉(공중 매달기) 테스트 시: 양발 비접지 → `contact2 = [0, 0]`이 정상 입력이다.
이때 보행명령은 허우적거림이 사전 예측된 정상 거동 (커밋 124cd46 참조).

### Stage4 자가검증 / 시뮬 검증

```bash
# 자가검증 (GPU 불필요): 조립 골든·이력 백필·클록 공식·슬루·모델 추론 5종
/home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/deploy/policy_runner_stage4.py --selftest

# 폐루프 시뮬 검증 (GPU): Biped12-Velocity-Stage4-Play-v0에서 관측 245·목표각을
# env 버퍼와 매 틱 대조 — 판정: 오차<1e-4 AND 15s 무낙상
OMNI_KIT_ACCEPT_EULA=YES /home/ryu/IsaacLab/isaaclab.sh -p \
  rl_walking/deploy/sim_validate_runner_stage4.py --headless
# 결과: log_picture/실행기_폐루프검증_stage4.txt
```
