# rl_walking/deploy — RL 보행 정책 실행기 (실물/단독 PC용)

**시뮬 전용 — 실물 송신은 승윤님 결정.** 이 디렉토리의 코드는 CAN/모터에 아무것도
보내지 않는다. `PolicyRunner`는 관측 → 목표관절각 계산만 하며, 계산 결과를 실물
브리지(Stage8 노드)에 연결하는 것은 별도의 명시적 결정 후에만 한다.

## 구성

| 파일 | 역할 |
|---|---|
| `policy_runner.py` | 45→12 정책 추론 + 아핀(×0.25+기본자세) + 슬루(±4°/틱) 실행기. `--selftest` 내장 |

- 의존성: `numpy` 필수, 추론은 `onnxruntime`(우선) 또는 `torch`(폴백) — 자동 선택.
  isaaclab 불필요 → 실물 PC/Jetson에서 단독 실행 가능.
- 체크포인트: `logs/rsl_rl/biped12_flat/2026-07-25_21-59-19/exported/{policy.onnx, policy.pt}`
- 호출 주기: **50Hz** (호출자가 보장 — 러너 내부에 타이머 없음)

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
