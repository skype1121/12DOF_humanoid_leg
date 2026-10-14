# 배포 계약 (deployment contract) — 0911 실기 직전 확정본

작성: 2026-09-11 · 브랜치 `hang2stand-prep-0911` · **코드를 직접 읽고 쓴 문서** (다른 보고서에서 옮기지 않음)
대상 체크포인트: `stand_hang_v5` (= S8ac4, run `2026-09-09_19-15-42`, **관측 320**) 및 대안 `stand_hang_v5p`(S8ac6)

## 0. 이 문서의 규칙

- 모든 주장에 태그: **[측정]** = 원본 로그/파일에서 직접 읽은 값 · **[계산]** = 식을 명시한 유도 ·
  **[가설]** = 미검증 추정. 상관은 인과가 아니며, 구별 불가한 설명은 복수로 남긴다.
- **압력 패드는 접지 판정 전용.** `foot_n` / `pct` / `raw_sum` / CoP / 좌우 하중비를 **측정 하중으로 쓰지 않는다.**
  패드에서 유도한 수치는 전부 `무교정 진단값(uncalibrated diagnostic)` 으로 표기했다.
- 검증 불가 항목은 **UNKNOWN** 으로 남겼다 (추정으로 메우지 않음).
- 인용은 `파일:행` 형식. 행 번호는 0911 워킹트리 기준.

### 대조 데이터 (교차검증 원본)

| 항목 | 값 |
|---|---|
| 블랙박스 | `reports/real_0910_A1/trial1/rl_bridge_20260910_154841.jsonl` |
| 정책 틱 수 | 15,830 (+ 이벤트 2행: `lift_guard`, `stop_all`) **[측정]** |
| 구간 | 316.58 s (50 Hz), `live=true`, `armed=true`, `checkpoint=stand_hang_v5` **[측정]** |
| 젯슨이 읽은 하드웨어맵 | `/home/mama/humanoid_repo/config/robot_12dof_hardware_map.json` (`session_.../bridge_154840.log:1`) **[측정]** |

---

## (a) 관절 이름 순서와 좌우 인터리브

**단일 원본**은 `rl_walking/deploy/policy_runner.py:59-66` 이다. 브리지는 이것을 그대로 import 한다
(`jetson/scripts/rl_bridge_node.py:83-84`), Stage4 러너도 같은 심볼을 재수출한다
(`rl_walking/deploy/policy_runner_stage4.py:60-71`).

```
JOINT_ORDER =
 0 left_hip_f_joint     1 right_hip_f_joint
 2 left_hip_a_joint     3 right_hip_a_joint
 4 left_hip_r_joint     5 right_hip_r_joint
 6 left_knee_joint      7 right_knee_joint
 8 left_ankle_f_joint   9 right_ankle_f_joint
10 left_ankle_r_joint  11 right_ankle_r_joint
```

- **좌우 교대(interleaved)**: 6쌍 `(hip_f, hip_a, hip_r, knee, ankle_f, ankle_r)` 각각 `[L, R]`.
  `rl_walking/symmetry.py:29-32` 가 미러 변환을 이 6쌍 구조 위에 정의하며, 쌍 순서의 근거를
  "log_picture/03 실측 순서"로 명시한다. `policy_runner.py:57-58` 은 "Isaac Lab BFS 순서 — 하드웨어맵
  motor_id(좌1-6/우7-12)와 다름. 실물 연결 시 반드시 '관절명' 기준으로 매핑할 것" 이라고 못 박는다.
- **함정 (0909 오독 사고의 재발 방지)**: 배열 인덱스를 `[왼다리 6, 오른다리 6]` 으로 읽으면 좌우가
  섞인다. 예) 인덱스 6·7 은 `left_knee`·`right_knee` 이지 "왼다리 7번째 관절"이 아니다.
- **하드웨어맵의 `isaac_dof_index` 는 이 순서가 아니다** **[측정]**: 맵은 `L:0~5 / R:6~11`
  (`config/robot_12dof_hardware_map.json`, 관절별 `isaac_dof_index`) 로, JOINT_ORDER 와 **다르다**.
  현재 배포 경로(브리지·러너·stage8)는 이 필드를 쓰지 않는다 (`grep isaac_dof_index` → `robot_runtime/dof_mapping.py`,
  `robot_runtime/dof12_mapping.py:106`, `sim_interface/isaac_12dof_command_buffer.py:76` 의 구 경로뿐) **[측정]**.
  그러나 **정책 벡터를 이 인덱스로 다루면 조용히 좌우·관절이 뒤섞인다** — 금지.

### 블랙박스 필드의 순서 규약 **[측정]** (`rl_bridge_node.py:1763-1793`)

| 필드 | 형태 | 순서 |
|---|---|---|
| `qpos_rad`, `qvel_rad_s`, `raw_action`, `tau_est_nm`, `temp_raw`, `soft_clamp_counts` | 길이 12 배열 | **JOINT_ORDER** |
| `targets_deg_joint`, `targets_deg_motor` | dict | **관절명 키** (삽입 순서는 JOINT_ORDER 이나 이름으로 접근할 것) |
| `contact`, `foot_n`, `foot_raw_sum` | 길이 2 | `[왼발, 오른발]` |
| `rp_deg` | 길이 2 | `[roll, pitch]` |

**미검증(UNKNOWN)**: 러너의 `JOINT_ORDER` 가 Isaac 관절체(articulation)의 실제 DOF 순서와 같다는 것은
**코드 안에 런타임 단정이 없다.** 학습 액션항은 `joint_names: ['.*']`, `preserve_order: false`
(`logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/params/env.yaml:537-541`) 이므로 순서는 관절체 내부 순서이고,
이 문서는 Isaac 을 돌리지 않았다. 근거는 `symmetry.py:29-32` 의 실측 주석과 `policy_runner.py:57-58` 의 서술뿐이다.
**완화**: 브리지는 입력·출력·명령 전 구간을 **관절명 키**로 다루므로(`rl_bridge_node.py:1637-1638, 1739-1742, 1815-1821`),
배열 인덱스 오염은 러너 내부(관측 조립 ↔ 액션 12) 한 곳에만 남는다.

---

## (b) 단위·프레임·영점(stand_zero_deg)·변환식

### 두 프레임

| 프레임 | 단위 | 정의 |
|---|---|---|
| **모터 인코더 프레임** | deg | AK 모터 로터 절대 엔코더가 읽는 각. Stage8 `latest_actual_deg` / `baseline_deg` / 명령이 전부 이 프레임 |
| **URDF 관절 프레임** | rad | 정책(학습 env)이 쓰는 각. `DEFAULT_POSE_RAD`·`qpos_rad`·`targets_deg_joint`(deg 표기)가 이 프레임 |

### 변환식 (단일 원본 = `FrameMap`, `rl_bridge_node.py:252-277`)

```
입력  joint_rad       = radians( DIR[j] × (motor_deg − SZ[j]) )     # :268-272
출력  motor_target_deg = SZ[j] + DIR[j] × degrees(joint_rad)        # :274-277
발행  relative_deg     = motor_target_deg − baseline_deg[j]         # :1821
      → Stage8 절대각 = baseline + relative = motor_target 그대로 복원 (baseline 상쇄)
```

- `DIR[j] ∈ {−1,+1}`, `SZ[j] = stand_zero_deg` (모터 프레임 deg).
- 로더는 12관절 전수 검증(direction ∈ {−1,+1}, SZ 유한·|SZ|<120°, motor_id 1..12 전단사)에 실패하면
  **기동 자체를 차단**한다 (`:285-347`).
- 사본이 둘 이상이면(`REPO/config/...` 와 `/home/mama/rl_calib/...`, `:240-243`) **DIR·SZ 불일치 시 기동 거부**
  (`:298-311`) — 젯슨 구맵 사고의 코드화된 백스톱.
- **Stage8 는 독립 로더를 가진다** (`jetson/scripts/stage8_12axis_mit_control_node.py:139-170`,
  맵 경로는 `robot_runtime/dof12_mapping.py:13` 의 `<repo>/config/robot_12dof_hardware_map.json` **하나뿐**).
  즉 브리지가 `/home/mama/rl_calib` 사본을 쓰고 stage8 가 repo 사본을 쓰면 **리밋 검사 프레임이 어긋난다** —
  위 사본 일치 검사가 이 조합만 막아 준다.

### 현재 `stand_zero_deg` 12값 **[측정]** (`config/robot_12dof_hardware_map.json`)

| # | 관절 | motor_id | DIR | **SZ(현재) [deg]** | SZ(0910 재캡처 前) | Δ |
|---|---|---|---|---|---|---|
| 0 | left_hip_f_joint | 1 | +1 | **32.25** | 31.55 | +0.70 |
| 1 | right_hip_f_joint | 7 | +1 | **9.87** | 8.88 | +0.99 |
| 2 | left_hip_a_joint | 2 | −1 | **4.78** | 3.66 | +1.12 |
| 3 | right_hip_a_joint | 8 | −1 | **8.36** | 8.49 | −0.13 |
| 4 | left_hip_r_joint | 3 | +1 | **7.42** | 5.25 | +2.17 |
| 5 | right_hip_r_joint | 9 | +1 | **33.32** | 28.01 | +5.31 |
| 6 | left_knee_joint | 4 | −1 | **14.48** | 11.20 | +3.28 |
| 7 | right_knee_joint | 10 | −1 | **21.32** | 24.73 | −3.41 |
| 8 | left_ankle_f_joint | 5 | +1 | **42.09** | 39.22 | +2.87 |
| 9 | right_ankle_f_joint | 11 | +1 | **20.84** | 25.76 | −4.92 |
| 10 | left_ankle_r_joint | 6 | −1 | **3.03** | 5.17 | −2.14 |
| 11 | right_ankle_r_joint | 12 | +1 | **7.62** | 7.11 | +0.51 |

전 관절 `kp=150.0 / kd=5.0` **[측정]**.

### 영점이 언제 바뀌었나 **[측정]**

- 파일 mtime `2026-09-10 16:28:51 KST`, 맵 `_provenance.zero_anchor_history` 의 **마지막(7번째) 항목**
  `{"at": "20260910_162820", "mode": "recapture"}` 이 위 12개 변경(`from`→`to`)을 그대로 기록한다
  (`config/robot_12dof_hardware_map.json`).
- ⚠ 같은 파일의 자유서술 필드 `_provenance.stand_zero_deg` 는 아직 `"v12_frame_reanchor_20260804"` 에 멈춰 있다 **[측정]** —
  **서술은 낡았고 값만 갱신됐다.** 서술을 근거로 영점을 판단하지 말 것.
- 이 변경이 **git 에 기록된 것은 오늘**이다: 커밋 `46bb737` (2026-09-11 20:58) — 커밋 메시지 자체가
  "내가 만든 변경 아님(영점 재캡처 추정), 보존. **로컬↔젯슨 일치는 미검증(현장 확인)**" 이라고 적고 있다.
- 중간 이력: `20260908_143941` 에 `left_hip_r 41.25→5.25`, `left_knee 47.2→11.2` (36°·36° = 배수 시프트 보정)
  가 있었고 커밋되지 않은 채 0910 재캡처가 덮었다. 그래서 커밋 diff 는 `41.25→7.42` 처럼 두 단계가 합쳐져 보인다.

> **⚠ 현장 점검 항목 (코드로 확인 불가)**: 로컬 `config/robot_12dof_hardware_map.json` 과 젯슨
> `~/humanoid_repo/config/robot_12dof_hardware_map.json`(및 `/home/mama/rl_calib/` 사본)의 SZ·DIR 일치는
> **이 PC 에서 검증할 수 없다 — UNKNOWN.** 브리지가 두 사본 불일치는 막지만(`:298-311`), **로컬 vs 젯슨**
> 불일치는 아무도 막지 않는다. 기동 전 `[FrameMap] 하드웨어맵:` 로그 줄의 경로와 12값을 눈으로 대조할 것.
> **(i) 절의 측정치가 이 위험이 실재함을 보여 준다 — 0910 실기는 위 "재캡처 前" 값으로 돌았다.**

---

## (c) IMU 규약

### 식 (전부 같은 한 식에서 나온다)

```
projected_gravity(quat_wxyz):            # rl_bridge_node.py:350-355  (world −z 를 body 로)
  gx = −2(xz − wy),  gy = −2(yz + wx),  gz = −(1 − 2(x² + y²))
roll  = atan2( gy, −gz )                 # rl_bridge_node.py:362-370 (_roll_pitch_deg → rp_deg[0])
pitch = atan2( gx, −gz )                 #                            (rp_deg[1])
```

- 정책 관측의 `projected_gravity` 도 **같은 함수**를 쓴다 (`rl_bridge_node.py:428` → `BridgeCore.tick`).
  직립·정합 IMU 는 `(0,0,−1)`.
- 콘솔은 **같은 식**을 `fwd`/`lat` 이름으로 쓴다 (`jetson/scripts/operator_console.py:544-545`).

### 중력 정합 게이트 **[측정]**

| 항목 | 값 | 근거 |
|---|---|---|
| 허용 오차 | `|g − (0,0,−1)| ≤ 0.3` | `rl_bridge_node.py:358` |
| 최대 나이 | `0.5 s` | `:359` |
| 적용 시점 | `stand`/`walk`/`march` **시작 명령 1회** (틱마다 아님) | `:1551-1560` |
| 우회 | 명령 JSON 에 `"force_gravity": true` | `:1554` |
| 의미 | 거리 = `2·sin(θ/2)` → 0.3 ≈ **17.2°** 기울기. 장착보정 미적용 IMU(≈(±1,0,0))는 거리 1.41 로 확실히 걸린다 | `:356-357` **[계산]** |

0910 trial1 실측: 시작 시 `"중력 정합 OK (|g−(0,0,−1)|=0.02)"` **[측정]** (`bridge_154840.log`).
IMU 장착회전은 `config/imu_mount_calib.json` 의 `R_base_imu` (2026-07-27, `two_pose_gravity_hang_tiltrec_offline`)
로 어댑터에서 적용된다 (`jetson/scripts/imu_adapter_node.py:128-186`); 출력 JSON 의 `mount_calibrated` 로 노출.

### 부호: 앞기울임 / 좌기울임 — **코드 안에 모순이 있다** ⚠

| 출처 | 주장 |
|---|---|
| `operator_console.py:542-543` (+ `console_ui.html` 라벨 "+ 전방 기울기 / + 좌측 기울기") | **`fwd(=pitch) > 0` = 앞으로 기울임, `lat(=roll) > 0` = 왼쪽으로 기울임 (base +X 전방 / +Y 좌측). "부호 규약(0809 검증)"** |
| `rl_walking/mdp_hang.py:363` (`torso_pitch_band` **독스트링**) | "앞기울임(**pg_x>0**, CoM 을 발 위로 가져오는 방향)" → 콘솔과 **일치** |
| `rl_walking/mdp_hang.py:367` (같은 함수 **행내 주석**) + `:370` 구현 + `:375-392` 셀프테스트 | "실측 부호: **앞기울임 = 음수**" → 콘솔과 **반대**. 비대칭 밴드 구현 `fwd = (−pitch − fwd_band)` 은 **음수 피치**를 관대하게 본다 |

- **[계산]** base 프레임이 `+X 전방 / +Z 상방`(우수계)이면, 앞으로 코가 내려가는 회전은 `+Y` 축 둘레 `+θ` 이고
  `pg = R^T(0,0,−1) = (sinθ, 0, −cosθ)` → `pitch = atan2(sinθ, cosθ) = +θ`. 즉 **앞기울임 = 양수**.
  그리고 이 자산의 `+X` 가 전방이라는 것은 `hang_stand_env_cfg.py:564-567` 의 "후방 CoM" 이
  `com_range["x"] = (−0.10, 0.0)` (음수 x = 뒤) 로 구현된 것과 정합한다.
- 따라서 **`mdp_hang.py:367` 의 행내 주석·`:370` 의 비대칭 구현이 틀렸을 가능성이 높다** **[가설]**.
  영향 범위: 비대칭 밴드를 쓰는 것은 `hang_stand_env_cfg.py:579` (`S10a`, `fwd_band_rad=0.21`) 뿐이다 **[측정]**.
  **`stand_hang_v5`(S8ac4)·`v5p`(S8ac6)는 `fwd_band_rad=None` 대칭 분기라 부호에 무관 — 0911 실기 후보는 영향 없음.**
- **[측정] 수치 일치**: 0910 trial1 의 정착 피치는 `−2.9 → −5.1°` 로 안정했고(아래 표), `stand_hang_v5` 의
  시뮬 정착 피치는 `−4.9°` 다 (`models/stand_hang_v5/meta.json` "정착 피치 −4.9°").
  **시뮬과 실물이 같은 부호·거의 같은 크기**다. 이것은 시뮬·실물 프레임이 서로 정합한다는 증거이지,
  그 −5° 가 앞인지 뒤인지를 결정하지는 **못한다** — 상관은 인과가 아니다.
- `_roll_pitch_deg` 자신의 독스트링이 이렇게 적고 있다: *"A1 기록 전용(0910). 판정용 각도는 콘솔 fwd/lat 와
  대조해 부호를 확정한 뒤 쓴다"* (`rl_bridge_node.py:363-364`). **현재 상태 = 부호 미확정.**

> **현장 1분 점검 (권장)**: 로봇을 손으로 **앞으로** 살짝 기울이고 콘솔 `앞/뒤` 값의 부호를 읽는다.
> 이것 하나로 위 모순이 닫히고, `rp_deg[1]`/`pitch` 를 판정에 쓸 수 있게 된다.

### 0910 trial1 자세 궤적 **[측정]** (판정 아님 — 부호 규약 확정 전까지는 방향을 단정하지 않는다)

| t [s] | roll° | pitch° | contact |
|---|---|---|---|
| 0 | +0.98 | −0.40 | [1,1] |
| 30 | +1.33 | −2.88 | [1,1] |
| 150 | +1.57 | −4.76 | [1,1] |
| 210 | +1.60 | −5.15 | [1,1] |
| 270 | +0.41 | −1.95 | [1,1] |
| 300 | −3.39 | +1.24 | [1,0] |
| 316.58 (마지막) | −5.28 | +1.67 | [0,0] → `lift_guard` |

---

## (d) DEFAULT_POSE / 기본 관절 오프셋

`rl_walking/deploy/policy_runner.py:69-76` (JOINT_ORDER 순, rad):

| 관절 | rad | deg **[계산]** |
|---|---|---|
| left/right_hip_f, hip_a, hip_r | 0.0 | 0.00 |
| left_knee | −0.10471975511965977 | **−6.00** |
| right_knee | +0.10471975511965977 | **+6.00** |
| left_ankle_f | +0.0244 | **+1.398** |
| right_ankle_f | −0.0244 | **−1.398** |
| left/right_ankle_r | 0.0 | 0.00 |

**두 곳에서 동시에 쓰인다** (같은 상수, 다른 역할):
1. 관측 오프셋 — `joint_pos = qpos_rad − DEFAULT_POSE_RAD` (`policy_runner_stage4.py:332`).
2. 액션 오프셋 — `target = DEFAULT_POSE_RAD + 0.25·raw` (`policy_runner_stage4.py:383`).
학습 env 도 동일 (`use_default_offset: true`, env.yaml:542).

**학습 env 와의 일치 검증 [측정]**: 배포 run 의 `params/env.yaml:189-200` `scene.robot.init_state.joint_pos` 가
위 12값과 **비트 단위로 같다**. 이 대조는 코드화되어 있다 — `rl_bridge_node._selftest_contract` (`:680-710`)
가 `DEFAULT_POSE·scale 0.25·슬루 4°/틱·50Hz·use_default_offset` 을 env.yaml 과 대조하고, stand 계열에
env.yaml 이 없으면 **실패**시킨다.

**모터 프레임 앵커 [측정]** (`rl_bridge_node._selftest_anchors`, `:463-500`): DEFAULT_POSE 를 출력 변환에 통과시킨
영점 상대값이 `무릎 ±6.00°`, `발목F ±1.40°` 여야 한다 (허용 0.01°). 재앵커로 SZ 가 움직여도 **상대값은 불변**이라
이 검사는 영점 재캡처를 건너서 유효하다.

**주의**: DEFAULT_POSE 는 "착지 자세"가 아니다. 온셋 블렌딩의 시작점은 stage8 가 스트리밍 중인 명령
(`baseline + commanded_relative_deg`)이다 (`:1747-1757`). 또 hang 계열 세대의 "몸통 −2.7°·무릎 ∓6°·어깨너비"
같은 자세는 **보상으로 유도된 자세**이지 오프셋이 아니다 — 오프셋을 바꾸면 배포 계약이 깨진다.

---

## (e) 320차원 정책 관측 레이아웃

### 구조 규칙

- isaaclab `ObservationManager` 는 **항(term) 블록** 구조다. 항마다 `CircularBuffer.buffer (N, T, D)` 를
  `(N, T·D)` 로 reshape 한 뒤 **항 선언 순서대로** concat 한다 — **프레임 인터리브가 아니다**
  (`policy_runner_stage4.py:7-20`, `symmetry.py:16-22`).
- **이력 T = 5**, 모든 항 `history_length: 5`, `flatten_history_dim: true` (env.yaml:370,383,393,410,426,437,447,486,499,512).
- **블록 안 프레임 순서: index 0 = 最古, index 4 = 最新** (`policy_runner_stage4.py:216-218, 236-237`).
- **이력 백필**: 리셋 후 첫 push 는 버퍼 전체를 현재 프레임으로 채운다 (`:227-234`, isaaclab `circular_buffer.py` L136-139 재현).
- `concatenate_terms: true` (env.yaml:352) → 인덱스 고정.

### 레이아웃 표 (`STAGE4_TERMS_S8A`, `policy_runner_stage4.py:86-102`; env.yaml:357-513 과 1:1)

| # | 항 | D | 블록 | 오프셋 | 내용 |
|---|---|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 15 | **[0:15)** | 베이스 프레임 각속도 [rad/s] = IMU 자이로 |
| 2 | `projected_gravity` | 3 | 15 | **[15:30)** | world −z 의 body 투영 (직립 (0,0,−1)) |
| 3 | `velocity_commands` | 3 | 15 | **[30:45)** | (vx, vy, wz). stand 는 항상 (0,0,0) (`rl_bridge_node.py:1511`) |
| 4 | `joint_pos` | 12 | 60 | **[45:105)** | `qpos_rad − DEFAULT_POSE_RAD`, JOINT_ORDER |
| 5 | `joint_vel` | 12 | 60 | **[105:165)** | 관절 각속도 [rad/s], JOINT_ORDER |
| 6 | `actions` | 12 | 60 | **[165:225)** | **직전 틱 raw** 정책출력 (아핀·슬루·LPF·블렌딩 前). 첫 틱 0 |
| 7 | `gait_clock` | 2 | 10 | **[225:235)** | `[sin φ, cos φ]`, `φ = 2π·f·tick·0.02`. **이 체크포인트 f = 1.1 Hz** |
| 8 | **`feet_contact`** | **2** | **10** | **[235:245)** | **✅ 확인. 5프레임 × `[왼발, 오른발]` 이진 0/1** |
| 9 | `gyro_ema` | 3 | 15 | **[245:260)** | S8a: 자이로의 EMA (α=0.1) |
| 10 | `joint_vel_ema` | 12 | 60 | **[260:320)** | S8a: 관절속도의 EMA (α=0.1), JOINT_ORDER |

**합계 = 5 × (3+3+3+12+12+12+2+2+3+12) = 5 × 64 = 320** **[계산]**
(245판 = 5×49, S8a 는 policy 그룹 **끝에** 두 항을 더한 것 — `hang_stand_env_cfg.py:347-359`.)

### `feet_contact` 235:245 — 3중 확인 **[측정]**

1. **코드**: 항 순서·차원에서 오프셋이 `235`, 블록 길이 `5×2=10` (위 표).
2. **모델**: `models/stand_hang_v5/policy.pt` 를 torch CPU 로 로드해 입력 차원을 탐침 →
   **245 실패 / 320 성공(출력 (1,12)) / 325 실패**.
3. **실물 로그**: trial1 tick 8000 의 `obs[235:245].reshape(5,2)` 가 그 틱과 직전 4틱의 `contact` 필드와
   **완전히 일치**. 비대칭 구간 tick 13040 에서 `obs` = `[[1,1],[1,1],[1,1],[1,1],[0,1]]`, 로그 `contact`
   이력 = `[[1,1],[1,1],[1,1],[1,1],[0,1]]` → **프레임 순서 = 最古→最新, 프레임 내부 = [L, R] 확정**.

### 프레임 내 `[L, R]` 의 근거

- 시뮬: `SceneEntityCfg("contact_forces", body_names=["left_ankle_r_joint","right_ankle_r_joint"], preserve_order=True)`
  (`rl_walking/stage4_env_cfg.py:36-40`) → 배포 run 의 env.yaml:464-476 에 그대로 직렬화되어 있다 **[측정]**.
  `preserve_order=True` 가 필수인 이유(접촉센서 바디 나열은 USD DFS 순서)도 같은 곳에 적혀 있다.
- 실물: 브리지가 `[left, right]` 로 조립 (`rl_bridge_node.py:1677-1681`), 어댑터가 `left_n`/`right_n` 을 낸다.
- **UNKNOWN 아님 — 확정.**

### 학습쪽 손상(corruption)과 실물의 차이 **[측정]** (env.yaml)

| 항 | 학습 함수 / 파라미터 | 실물 러너 |
|---|---|---|
| `projected_gravity` | `mdp_stage4:projected_gravity_lagged` + Unoise ±0.05 | 생값 (IMU 쿼터니언에서 계산) |
| `joint_pos` | `joint_pos_rel_biased` (`quant_step_rad 0.003`, `ankle_dither_rad 0.008`) + Unoise ±0.01, clip ±10 | 생값 |
| `joint_vel` | `joint_vel_rel_dropout` (`zero_prob 0.25`) + Unoise ±1.5, clip ±60 | 생값 (위치 유한차분) |
| `base_ang_vel` | Unoise ±0.2, clip ±20 | 생 자이로 |
| `gait_clock` | `freq: 1.1` | `CKPT_GAIT_FREQ["stand_hang_v5"]=1.1` (`rl_bridge_node.py:228-229`) |
| `feet_contact` | `threshold 5.0`, **`delay_steps: 5`**, **`threshold_range (2.0, 20.0)`** (env별 랜덤 임계) | 인위 지연 0 (실물 패드 10 Hz → 자연 지연 ≤100 ms). 임계는 **패드 단위 접지 우선, 없으면 5 N 합산 폴백** (g 절) |
| `gyro_ema` / `joint_vel_ema` | α 0.1, **함수 내부에서 자체 노이즈(0.2 / 1.5) 후 EMA**, 소스는 dropout **없는** 원값 (`mdp_hang.py:488-513`) | 같은 α 0.1 이나 소스는 **관측항과 같은 값** (`policy_runner_stage4.py:338-344`) |

마지막 행은 **시뮬/실물 미세 불일치**다 **[측정, 코드 대조]**: 시뮬에서 EMA 채널은 dropout 없는 독립 잡음 사본이고,
실물에서는 `joint_vel` 항과 **같은** 신호의 EMA 다. dropout·잡음이 없는 실물에서는 두 정의가 수렴하므로
실무 영향은 작다고 본다 **[가설]**.

EMA 초기화도 양쪽이 같다: 리셋 직후 첫 틱 출력 = 현재값 (`mdp_hang.py:488-495` vs `policy_runner_stage4.py:341-343`).

---

## (f) 액션 처리 (raw → 모터 명령)

### 실물 경로 (순서가 계약이다)

```
raw (정책 12)                                            # obs 의 actions 에는 이 값이 들어간다
 → target  = DEFAULT_POSE_RAD + 0.25 · raw               # policy_runner_stage4.py:383   (클램프 금지)
 → emitted = prev + clamp(target − prev, ±0.0698 rad)    # :385-388  (첫 틱 prev = 실측 qpos)
 → [stand 계열만] filt = filt + 0.27·(emitted − filt)    # rl_bridge_node.py:1711-1713   (첫 틱 filt = emitted)
 → 관절 rad → 모터 deg (SZ, DIR)                          # :1739-1742
 → 블렌딩: land·(1−α) + motor_target·α, α = smoothstep(t/1.5s)   # :1747-1762
 → relative = motor_target − baseline → SET_JOINT_TARGETS (1 메시지/틱, 12관절 배치)  # :1815-1824
 → [stage8] 4°/틱 슬루 → soft 클램프 → MIT 명령            # stage8:838-843, 760-790
```

### 상수표

| 항목 | 값 | 근거 |
|---|---|---|
| `ACTION_SCALE` | **0.25** | `policy_runner.py:78` / env.yaml:539 |
| `SLEW_RAD` (러너) | **0.06981317007977318 rad = 4.000°/틱** | `policy_runner.py:79` / env.yaml:543 |
| `RATE_HZ` | **50** (`TICK_SEC` 0.02) | `policy_runner.py:80`, `rl_bridge_node.py:87-88`; env `dt 0.005 × decimation 4` (env.yaml:21, 84) |
| `STAND_LPF_BETA` | **0.27** (EMA 1차, 차단 ≈2.5 Hz@50 Hz) — **stand 계열에만** 적용 | `deploy_chain.py:26`, `rl_bridge_node.py:870`, 적용 조건 `:1711` (`ckpt_name.startswith("stand")`) |
| `BLEND_S` | **1.5 s**, smoothstep `a²(3−2a)` (양 끝 기울기 0) | `deploy_chain.py:25,30-35`, `rl_bridge_node.py:783,1758` |
| 블렌딩 시작점 | stage8 스트리밍 명령 `baseline + commanded_relative_deg`, 없으면 실측 | `:1747-1757` |
| 실물 인위 지연 | **없음** (러너에 지연 버퍼 없음) | `policy_runner.py:24-25` |
| 학습 DR 액션 지연 | **3~8 틱** (env별 균등) | env.yaml:544-545 (`max_delay_steps 8 / min_delay_steps 3`) |
| 학습 DR 래퍼 LPF β | **0.2 ~ 1.0** (실물 0.27 을 포함) | env.yaml:546-548 |
| 학습 DR 백래시 | **0.005 ~ 0.026 rad (0.3~1.5°)** | env.yaml:549-551 |
| `kp / kd` | **150 / 5** (하드웨어맵 12/12) | `config/robot_12dof_hardware_map.json`, 기대값 `rl_bridge_node.py:248-249` |
| 게인 게이트 | 맵 kp/kd ≠ 150/5 이면 **`--live` 기동 거부**, `--force-gains` 로만 우회 | `:41-45, 502-517` |
| Stage8 슬루 | **4.0°/틱**, 제어 주기 **0.02 s** | `stage8:69-71, 838-843` |
| 러너 타깃 클램프 | **없음 (금지)** — soft 리밋은 로드만 하고 `step()` 에서 쓰지 않는다 | `policy_runner_stage4.py:373-382` (0802 회귀 수정 주석), `:274-294` |
| 실제 위치 울타리 | **Stage8 절대각 soft 클램프** (경계로 클램프 + 카운터 노출), 경계 초과 **25° 이상은 거부**(`limit_runaway`) | `stage8:760-785` |
| 하드 백스톱 | `|모터절대각 − SZ| > 하드리밋 폭 + 여유` → 거부 | `stage8:787-790` |

### 소프트/하드 리밋 (URDF 관절 프레임 deg) **[측정]** (`config/joint_limits_12dof.json` via `robot_runtime/joint_limits.py:17-27`)

| 관절 | soft_min | soft_max | hard_min | hard_max |
|---|---|---|---|---|
| left_hip_f | −29.00 | 45.00 | −30.0 | 100.0 |
| right_hip_f | −40.70 | 29.00 | −100.0 | 30.0 |
| left_hip_a | −34.00 | 13.00 | −35.0 | 13.9 |
| right_hip_a | −13.20 | 23.80 | −14.2 | 35.0 |
| left_hip_r | −24.00 | 29.00 | −30.0 | 30.0 |
| right_hip_r | −29.00 | 29.00 | −30.0 | 30.0 |
| left_knee | −83.50 | 4.00 | −120.0 | 5.0 |
| right_knee | −4.00 | 80.00 | −5.0 | 120.0 |
| left_ankle_f | −22.00 | 29.00 | −45.0 | 30.0 |
| right_ankle_f | −29.00 | 18.50 | −30.0 | 45.0 |
| left_ankle_r | −19.00 | 19.00 | −20.0 | 20.0 |
| right_ankle_r | −19.00 | 19.00 | −20.0 | 20.0 |

### 발진 가드 (액션 경로의 마지막 안전장치) **[측정]**

- **몸(피드백) 가드** `:1696-1709` 와 **타깃 가드** `:1714-1731` 두 개, 둘 다 트립 시 `STOP_ALL` + 정책 비활성.
- 파라미터: 즉시 트립 `|qvel| > 12 rad/s` (`:724`), 데드밴드 0.5 rad/s (`:725`),
  채터 모드 ≥8회/0.5 s (|v|≥0.55 rad/s), 중주파 모드 ≥9회/0.5 s (진폭 ≥2.5°) (`:875-882`).
- **커버리지 밖(문서화됨)**: 4 Hz 미만 대진폭 요동 — 운용자 estop·줄 인장 영역 (`:722-724`).

### 실물 로그에 의한 체인 전수 재현 **[측정]** — (i) 절 참조
`raw_action` 과 `qpos_rad` 만으로 위 체인(0.25 → 슬루 4° → EMA 0.27)을 3001틱 재구성한 결과,
로그의 `targets_deg_joint` 와 **최대 오차 0.00069°** (중앙값 0.00031°; 잔차는 블랙박스의 `raw_action` 4자리 반올림 크기와 일치).
β 감도: 0.20 → 최대 0.74°, **0.27 → 0.0007°**, 0.35 → 0.63°, 1.0(무필터) → 2.92°. **β=0.27 유일 확정.**

---

## (g) 실물 접촉(contact) 정의

### 어댑터 (`jetson/scripts/foot_force_adapter_node.py`)

| 항목 | 값 | 근거 |
|---|---|---|
| 입력 | `/humanoid/pressure/data` 10 Hz, `left[4]`, `right[4]` 카운트(0~1023) | `:3-5, 25` |
| **패드 단위 접지 규칙 (0910 신설)** | 패드별 `(count − pad_bias) > PAD_CONTACT_COUNTS` 면 그 패드 눌림 → **4개 중 하나라도 눌리면 그 발 접지** | `:73-74, 81-85` |
| `PAD_CONTACT_COUNTS` | **80 counts** | `:48` |
| `left_pad_bias` / `right_pad_bias` | ROS 파라미터, 기본 `[0,0,0,0]` (0910 실측: 왼발 패드[2] 예압 700) | `:182-183` |
| 출력 필드 | `left_contact`, `right_contact`(bool), `left_pads_on`, `right_pads_on`(0~4) | `:82-85` |
| `left_n` / `right_n` | `npc × max(0, Σcount − bias)` — **무교정 진단값. 하중으로 쓰지 말 것** | `:77-78` |
| `NEWTON_PER_COUNT` / 바이어스 | 0.02650 / L 328 / R 62 (무교정 진단값) | `:43-45` |

### 브리지

| 규칙 | 값 | 근거 |
|---|---|---|
| **우선**: 어댑터가 `left_contact`/`right_contact` 를 주면 그대로 사용 | — | `rl_bridge_node.py:1677-1678` |
| **폴백**: 없으면 `foot_n > 5.0 N` (구판 어댑터 호환) | `CONTACT_FORCE_THRESHOLD_N = 5.0` | `:1679-1681`, `policy_runner_stage4.py:82` |
| 러너 입력 검증 | `contact2` 는 **0.0/1.0 만** 허용 — 생힘[N] 입력은 즉시 예외 | `policy_runner_stage4.py:308-318` |

### HangGuard (`rl_bridge_node.py:801-860`, `update()` `:830-860`)

상태: `PRE_CONTACT_HANG_ALLOWED → CONTACT_ACQUIRED → FREE_STAND` (**되돌아가지 않음**)

| 상수 | 값 | 의미 | 근거 |
|---|---|---|---|
| `LIFT_STOP_S` | **1.0 s** (0910 지시로 0.6→1.0) | `CONTACT_ACQUIRED` 이후 **양발 무접촉**이 이만큼 지속 → 즉시 동결(`STOP_ALL`) | `:788, 847-853` |
| `HANG_MAX_S` | **120 s** | `PRE_CONTACT` 에서 접지 없이 이 시간 초과 → 동결 | `:795, 842-844` |
| `CONTACT_CONFIRM_S` | **0.3 s** | 양발 접촉이 이만큼 지속돼야 `CONTACT_ACQUIRED` | `:796, 834-841` |
| `HANG_BODY_WEIGHT_N` / `FREE_STAND_LOAD_FRAC` | 172.9 N / 0.8 | `FREE_STAND` 표시 전용 | `:797-798, 856-858` |
| 진입 조건 | `stand` 명령에 **`"hang_arm": true` 명시** 시에만 `PRE_CONTACT` 로 시작 | `:1562-1567` |

> **⚠ 규칙 위반 소지 (플래그)**: `FREE_STAND` 전이는 `foot_total_n ≥ 0.8 × 172.9 N` — **패드 유도 하중값**을
> 임계로 쓴다 (`:856-858`). 제어에는 영향이 없고(동결 조건은 접촉 이진뿐) **표시·구간분할 전용**이지만
> (`operator_console.py:589`, `analyze_hang2stand_trial.py:72`), 블랙박스 `hang_state` 를 "자립했다"의 근거로
> 읽으면 무교정 하중을 판정에 쓰는 셈이 된다. **hang_state = FREE_STAND 는 판정 근거가 아니다.**

### 0910 trial1 에서 실제로 쓰인 규칙 **[측정]**

- 15,830 틱 **전부** `contact == (foot_n > 5.0 N)` — **불일치 0건**. → **패드 단위 규칙이 아니라 5 N 합산 폴백**이 돌았다.
- 어댑터 상수 역산(**무교정 진단값**, `foot_n` vs `foot_raw_sum` 최소제곱, 잔차 ≤0.006):
  `npc = 0.026500` 양발 공통, **왼발 bias = 700 counts**, 오른발 bias = 62 counts.
  → 같은 5 N 임계가 **왼발은 raw_sum > 889, 오른발은 raw_sum > 251** 로 **좌우 비대칭**하게 작동했다 **[계산]**.
- 접촉 분포: `[1,1]` 13,890 / `[1,0]` 1,070 / `[0,1]` 780 / `[0,0]` 90 틱 **[측정]**.
- 종료: `lift_guard: 양발 무접촉 **0.6s** 지속` (이벤트 행) — **당시 젯슨 코드의 `LIFT_STOP_S` 는 0.6 이었고
  현재 저장소는 1.0 이다.** 젯슨 배포본 갱신 여부는 **현장 확인 필요 (UNKNOWN)**.

---

## (h) 센서 신선도 게이트

| 상수 | 값 | 대상 | 근거 |
|---|---|---|---|
| `SENSOR_STALE_SEC` | **0.2 s** | IMU 항상; 관절 피드백은 `--live` 일 때 | `rl_bridge_node.py:89, 1594-1598` |
| `FOOT_STALE_SEC` | **0.35 s** | 발압력 (10 Hz 센서라 3주기 결손 허용) | `:90-93, 1597` |
| `JOINT_STALE_DRYRUN_SEC` | **0.5 s** | 드라이런(비-live)에서만 관절 피드백 | `:94-97, 1594` |
| `STATUS_FEEDBACK_FRESH_SEC` | **0.5 s** | stage8 `joints[j].feedback_age_sec` — 이 미만일 때만 관절각 채택 | `:98, 1436` |
| `STAND_WARMUP_MAX_S` | **4.0 s** | 인계 워밍업 창 | `:99-103` |

### 판정식 (`:1595-1598`)

```
fresh = (now − joint_ts < joint_stale)          # live: 0.2 / dryrun: 0.5
    and (now − imu_ts   < 0.2)
    and (now − foot_ts  < 0.35)
    and len(joint_deg) == 12
```

- `joint_ts` 는 **12관절 전부 신선할 때만** 갱신되고, 갱신값은 도착시각이 아니라
  `time.time() − worst_age` (데이터 나이 보정) 다 (`:1460-1465`) — CAN 두절 후 얼어붙은 각으로
  정책이 계속 도는 창을 막는다.
- 스파이크 필터: 한 갱신에 30° 이상 점프는 물리 불가로 보고 **직전값 유지·신선도는 인정**, 새 샘플 기준 3연속이면 채택 (`:1437-1459`).

### `warmup_gate` 의미론 (`:386-397`, 사용 `:1601-1614`)

```
warmup_gate(fresh, warmed, now, warmup_until) →
  (True,  "ok")       # warmed 이거나 방금 신선해짐 → 정책 구동
  (False, "prime")    # 창 이내·미신선 → _send_hold_prime() (현 자세 상대 0 명령으로 stage8 status 에코를 깨움) 후 대기
  (False, "timeout")  # 창 초과·미신선 → STOP_ALL("sensor_stale_warmup_timeout") + 정책 비활성
```

- **`warmed=True` 가 된 뒤에는 신선도와 무관하게 통과**시키고, 그 뒤는 정상상태 게이트(0.2 s)가 판정한다 —
  워밍업은 **1회성 레이스 제거**이지 게이트 완화가 아니다 (`:396-397`).
- `warmed`/`warmup_until` 은 stand/walk/march **명령마다** 초기화된다 (`:1585-1586`).
- 정상상태에서 `fresh` 가 깨지면 `STOP_ALL("sensor_stale")` + **정책 비활성** — 자동 재개 없음,
  운용자 재명령으로만 복귀 (`:1617-1632`).
- 비유한(NaN/inf) 관측은 즉시 `STOP_ALL("nonfinite_observation")` (`:1645-1652`).

**0910 trial1 실측**: `status_age_s` 중앙값 **0.009 s**, p95 0.013 s, 최대 **0.028 s** → 워밍업/스테일 게이트 트립 0회 **[측정]**.

---

## (i) 교차검증 — 실물 블랙박스 한 틱

### 선택한 틱 **[측정]**

```
파일 : reports/real_0910_A1/trial1/rl_bridge_20260910_154841.jsonl
인덱스: 8000 (정책 틱)   timestamp = 1789024684.020027
live=true  armed=true  checkpoint=stand_hang_v5  blend_alpha=1.0  hang_state=CONTACT_ACQUIRED
obs 길이 = 320
```
`blend_alpha = 1.0` 인 틱을 고른 이유: α<1 구간에서는 `targets_deg_motor` 가 블렌딩으로 덮여
프레임 변환식과 직접 대조할 수 없다 (`:1759-1762`).

### ① 프레임 변환 `motor = SZ + DIR × joint_deg` — 관절명 기준 **[측정 + 계산]**

역산으로 얻은 SZ (α=1 인 두 틱 `t=1.52 s` 와 `t=316.60 s` 에서 `(Δmotor)/(Δjoint)` 로 DIR, 절편으로 SZ):

| 관절 | `targets_deg_joint` | `targets_deg_motor` | **역산 DIR** | **역산 SZ_used** | 로컬 맵 DIR/SZ | 잔차 |
|---|---|---|---|---|---|---|
| left_hip_f | −6.19121 | 25.35879 | +1 | **31.55** | +1 / 32.25 | 0 |
| right_hip_f | +9.21344 | 18.09344 | +1 | **8.88** | +1 / 9.87 | 0 |
| left_hip_a | +1.87449 | 1.78551 | −1 | **3.66** | −1 / 4.78 | 0 |
| right_hip_a | −0.31741 | 8.80741 | −1 | **8.49** | −1 / 8.36 | 0 |
| left_hip_r | +0.97595 | 6.22595 | +1 | **5.25** | +1 / 7.42 | 0 |
| right_hip_r | +0.50851 | 28.51851 | +1 | **28.01** | +1 / 33.32 | 0 |
| left_knee | −5.08159 | 16.28159 | −1 | **11.20** | −1 / 14.48 | 0 |
| right_knee | +4.28778 | 20.44222 | −1 | **24.73** | −1 / 21.32 | 0 |
| left_ankle_f | +8.03460 | 47.25460 | +1 | **39.22** | +1 / 42.09 | 0 |
| right_ankle_f | −10.35796 | 15.40204 | +1 | **25.76** | +1 / 20.84 | 0 |
| left_ankle_r | +1.84981 | 3.32019 | −1 | **5.17** | −1 / 3.03 | 0 |
| right_ankle_r | +1.21607 | 8.32607 | +1 | **7.11** | +1 / 7.62 | 0 |

**검증된 것**
- `targets_deg_motor[j] = SZ_used[j] + DIR[j] × targets_deg_joint[j]` 가 12/12 관절에서 **잔차 정확히 0** (float 표기 한계 내) — **출력 변환식 확정**.
- **DIR 12/12 가 현재 로컬 맵과 일치.**
- **SZ 12/12 가 "0910 재캡처 前" 값과 정확히 일치**한다 (맵 `_provenance` 의 `20260910_162820` 항목 `from` 값 전부와 동일).
  → **0910 실기는 현재 로컬 맵의 영점이 아니라 그 이전 영점으로 돌았다.** (b) 절의 위험이 실측으로 확인된다.

### ② `qpos_rad → 모터 deg` 역변환 **[계산]**

같은 틱의 `qpos_rad` 를 `SZ_used + DIR × degrees(qpos)` 로 되돌린 값(모터 프레임 deg):

```
left_hip_f  25.037 | right_hip_f  18.283 | left_hip_a   1.650 | right_hip_a  8.951
left_hip_r   6.131 | right_hip_r  28.556 | left_knee   16.338 | right_knee  20.360
left_ankle_f 47.637| right_ankle_f 14.961| left_ankle_r 3.158 | right_ankle_r 8.251
```

**검증하지 못한 것 (UNKNOWN)**: 블랙박스에는 원본 `latest_actual_deg`(모터 프레임 실측)와 `baseline_deg` 가
**기록되지 않는다** (`:1763-1793` 의 dbg 딕셔너리에 없음). 따라서 `joint_rad = radians(DIR×(motor_deg − SZ))`
의 **입력 변환은 왕복(round-trip)으로만 확인**했고, 원본 모터각과의 직접 대조는 불가능하다.
위 12값은 그 역변환 결과이며, 목표각(①)과 0.1~0.5° 수준으로 가깝다 — 정상 추종과 부합하지만 증명은 아니다.

### ③ `targets_deg_motor == baseline + relative` **[코드 확정 / 로그 미검증]**

- 코드상 항등식이다: 발행값이 `targets[j] = targets_deg_motor[j] − baseline[j]` (`:1821`) 이고,
  Stage8 가 `absolute_deg = baseline_deg_by_joint[j] + target_deg` 로 되돌린다 (`stage8:760`).
  즉 `baseline + relative ≡ targets_deg_motor` — **baseline 이 어떤 자세로 캡처됐든 상쇄된다.**
- **로그로는 검증 불가 (UNKNOWN)**: `baseline_deg` 도 발행된 `relative` 도 블랙박스에 없다.
  세션 타임라인(`session_20260910_154826/timeline.jsonl`)도 관절별 baseline 을 담지 않는다 **[측정, 전수 스캔]**.
- 간접 증거 **[측정]**: trial1 전 구간 `soft_clamp_total = 0`, `last_reject_reason = null` —
  Stage8 가 12관절 명령을 한 번도 거부·클램프하지 않았다. baseline 미설정 관절이 있었다면 노드가 거부했을 것이다.
- 절차 증거 **[측정]**: `stage8_baseline_160822.log` = `SET_BASELINE_FROM_CURRENT_ALL — 베이스라인 12/12`,
  정책 시작은 그 다음(`16:15:24`).

### ④ 관측 벡터 대조 (같은 틱) **[측정]**

| 검사 | 결과 |
|---|---|
| `obs[45:105]` 最新 프레임 vs `qpos_rad − DEFAULT_POSE_RAD` | 최대 |Δ| = **4.0e−5** (블랙박스 4자리 반올림 크기) |
| `obs[105:165]` 最新 vs `qvel_rad_s` | **0.0** |
| `obs[0:15]` 最新 vs `gyro_rad_s` | 일치 |
| `obs[15:30]` 最新 vs `projected_gravity(quat_wxyz)` | 일치 (`[−0.0868, 0.0270, −0.9959]`) |
| `obs[30:45]` 最新 vs `cmd` | `[0,0,0]` 일치 (stand) |
| `obs[165:225]` 最新 vs **직전 틱** `raw_action` | **0.0** (이번 틱 raw 와는 0.0056 차 → `actions` = 직전 틱 raw 확정) |
| `obs[235:245]` vs `contact` 5틱 이력 | 완전 일치 (最古→最新, `[L,R]`) |
| `gait_clock` 틱당 위상 증분 | **0.138230 rad/틱 → 1.1003 Hz**; 첫 틱 `[sin,cos] = [0,1]` (φ=0) |

### ⑤ 액션 체인 재현 **[측정]**

`raw_action` 과 첫 틱 `qpos_rad` 만으로 `DEFAULT + 0.25·raw → 슬루 4°/틱 → EMA β0.27` 을 3001틱 재구성:

| β | 최대 오차 [deg] | 중앙값 |
|---|---|---|
| 0.20 | 0.7449 | 0.0240 |
| **0.27** | **0.00069** | **0.00031** |
| 0.35 | 0.6252 | 0.0174 |
| 1.00 (무필터) | 2.9203 | 0.0612 |

→ **스케일 0.25 · 슬루 4°/틱 · LPF β 0.27 · "슬루 → LPF" 순서**가 실물 로그로 전부 확인된다.
또한 블랙박스의 `targets_deg_joint` 는 **LPF 적용 後 / 블렌딩 前** 값임이 확인된다 (코드 `:1711-1713` → `:1740-1742`).

### ⑥ 블렌딩 **[측정]**
α<1 인 틱이 **정확히 75개 = 1.50 s**, α 는 `0.0 → 0.001 → 0.002 → …` 로 시작(양 끝 기울기 0),
75번째에서 1.0. → `BLEND_S = 1.5 s` smoothstep 확정.

---

## 종합 — 미확정(UNKNOWN) 목록

1. **로컬 ↔ 젯슨 하드웨어맵 일치.** 이 PC 에서 검증 불가. **(i)①이 0910 실기가 구영점으로 돌았음을 보여 주므로
   이번엔 반드시 기동 로그의 `[FrameMap] 하드웨어맵:` 경로 + 12값을 대조할 것.**
2. **IMU 앞/뒤 부호.** 콘솔(`+ = 앞`, 0809 검증)과 `mdp_hang.py:367` 행내 주석(`앞 = 음수`)이 충돌한다.
   `stand_hang_v5/v5p` 는 대칭 밴드라 무영향이지만, `rp_deg` 를 판정에 쓰려면 현장 1분 점검이 필요하다.
3. **`baseline + relative` 의 로그 검증.** 블랙박스에 `baseline_deg`·발행 `relative` 가 없어 항등식은
   코드로만 확정. (개선 제안: dbg 에 `baseline_deg` 12값을 1회/틱 또는 변경 시에만 기록)
4. **러너 `JOINT_ORDER` ↔ Isaac 관절체 DOF 순서.** 런타임 단정이 없다. Isaac 없이는 확인 불가.
5. **젯슨 배포본의 코드 세대.** `LIFT_STOP_S` 가 0.6(0910 실측) 인지 1.0(현재 저장소)인지,
   어댑터가 패드 단위 접지(`left_contact`) 판을 싣고 있는지 — 둘 다 현장 확인 사항.
6. **근정적(near-static) 데이터의 한계.** trial1 은 대부분 정지·현가 상태다. 이 데이터로는
   관절 마찰·백래시·감쇠·발목 강성 같은 파라미터를 **식별할 수 없다.** 추정하지 않는다.

## 위험 (실기 전 확인 권장)

- **R1 (높음)** 영점 프레임 불일치: 0910 실기가 구 SZ 로 돌았음이 실측 확인됐다. 오늘 젯슨이 어느 SZ 를
  물고 있는지 모르면 `stand_zero` 오차가 그대로 자세 오차가 된다. (CLAUDE.md: "영점은 가장 위험한 쓰기")
- **R2 (중)** `mdp_hang.torso_pitch_band` 의 비대칭 분기 부호. `S10a`(v6 후보) 학습에 영향. 후보 v5/v5p 는 무영향.
- **R3 (중)** 좌우 비대칭 접촉 임계: 5 N 합산 폴백에서 왼발 bias 700 counts 가 왼발을 늦게/빨리 "비접촉"으로
  만든다 (무교정 진단값 기반 관찰). 패드 단위 규칙(`PAD_CONTACT_COUNTS=80` + per-pad bias)이 젯슨에 실려 있는지 확인.
- **R4 (낮음–중)** `hang_state = FREE_STAND` 는 패드 유도 하중 임계다. **판정 근거로 쓰지 말 것** (표시 전용).
- **R5 (낮음)** 하드웨어맵 `isaac_dof_index`(L0-5/R6-11)로 정책 벡터를 인덱싱하면 조용히 좌우가 섞인다.
  현재 배포 경로는 쓰지 않지만, 새 스크립트에서 실수하기 쉬운 필드다.

---

### 부록 — 재현에 쓴 명령 (전부 읽기 전용, GPU·Isaac 불필요)

- `python3 rl_walking/deploy/deploy_chain.py --selftest` (체인 벡터화=참조)
- `python3 rl_walking/deploy/policy_runner_stage4.py --selftest` (245/320 조립 골든)
- `python3 jetson/scripts/rl_bridge_node.py --selftest` (하드웨어맵 12/12·왕복 항등·수치 앵커·게인)
- `python3 jetson/scripts/foot_force_adapter_node.py --selftest` (패드 단위 접지)
- 본 문서의 블랙박스 측정은 `reports/real_0910_A1/trial1/rl_bridge_20260910_154841.jsonl` 을
  표준 라이브러리 + numpy 로 파싱해 산출했다 (일회성 스크립트, 저장소에 남기지 않음).

---
## 부록 (0911 추가): 학습 env ↔ 배포 러너 ↔ 실물 브리지 조립 등가성 (MEASURED)
- `rl_walking/deploy/sim_validate_runner_stage4.py --hang-nominal` (S8ac4, v5, 15 s): 학습 전용 손상 모델을 끄면 기본 8항 오차 0, 슬루 후 목표각 0 rad, EMA 항은 리셋 2 s 이후 ≤5.5e-5. → 러너 320 조립 = env 관측 (레이아웃·이력·클록·접촉·EMA 산술·슬루 1:1).
- `rl_walking/deploy/replay_bridge_log.py` (0910 trial1/3): 러너 조립·정책 출력 = 실물 블랙박스 기록 ≤1e-4. → 러너 = 실물 브리지.
- 계약 주의점(학습에서만 존재, 배포엔 없음): IMU 자세 지연 0~3틱(`projected_gravity_lagged`, `imu_lag_steps`)·IMU 롤/틸트 편향·관절 영점 편향(`calib_bias`)·관절위치 양자화 0.003 rad·발목 디더 0.008 rad(`joint_pos_rel_biased`)·관절속도 드롭아웃 25 %(`joint_vel_rel_dropout`)·접촉 비트 5틱 지연·임계 2~20 N(`feet_contact` params)·액션 지연 3~8틱·LPF β0.2~1.0·백래시(`SlewJointPositionActionCfg`)·EMA 항 내부 노이즈(자이로 0.2, 관절속도 1.5). EMA 재시드: env 는 리셋 후 2틱 동안, 러너는 1회 — (0.9)ⁿ 감쇠 과도.
- 실측 대응: 정책루프 지연 중앙 6.3틱(학습 3~8 안), 접촉 10 Hz S&H(학습 5틱 지연과 정합), 자이로 잡음 σ 0.31 °/s(학습 노이즈 ±0.2 rad/s 대역보다 훨씬 작음).
