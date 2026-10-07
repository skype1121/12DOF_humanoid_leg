# 정책·체크포인트 전수 목록과 분류 초안 (1007 밤)

> 자동 생성: 2026-10-08 00:54:44 · 브랜치 `hang2stand-prep-0911` @ `df775fa0` · 명령:
> `python3 reports/night_1007/policy_inventory.py --md5 all`
> 이 문서는 **초안**이다. 최종 후보 선별은 평가 결과로 주 작업자가 정한다. 이 작업에서는 **아무것도 지우거나 옮기지 않았다**(읽기만).
> 젯슨·ROS·CAN·GPU·Isaac 은 쓰지 않았다. 젯슨에 관한 내용은 저장소 코드·문서와, 같은 폴더에 이미 받아져 있던 md5 목록(스냅숏)을 읽은 것이고 접속해서 확인한 것이 아니다.
> 기계가 읽는 같은 내용: `reports/night_1007/policy_inventory.json` · 다시 만들기: `python3 reports/night_1007/policy_inventory.py [--md5 all]`

## 0. 한눈에

- `logs/rsl_rl/` 의 run 디렉터리 **244 개**, 체크포인트(model_N.pt) **2677 개**, 합계 **20.25 GiB**(체크포인트만 19.19 GiB). run 을 가리키는 심볼릭 링크 63 개와 실험 폴더 링크 1 개는 별칭으로만 세고, 실제 경로를 한 번씩만 셌다.
- 디스크: 전체 450.67 GiB 중 여유 72.26 GiB(16 %) — 측정 시각 2026-10-08 00:54:44.

| 분류 | run | 체크포인트 | 용량 | 그중 핵심 체크포인트 | 솎을 수 있는 체크포인트(남기는 run 안) |
|---|---:|---:|---:|---:|---:|
| 배포 후보 | 3 | 32 | 271.1 MiB | 13 | 19 개 · 145.3 MiB |
| 평가 중 후보 | 6 | 35 | 307.0 MiB | 35 | — |
| 기준선·검증 의존 | 12 | 211 | 1.55 GiB | 32 | 179 개 · 1.23 GiB |
| 기록 보존 | 43 | 694 | 4.90 GiB | 120 | 574 개 · 3.77 GiB |
| 폐기 후보 | 172 | 1624 | 12.76 GiB | — | — |
| 불명 | 8 | 81 | 484.7 MiB | — | — |
| **합계** | **244** | **2677** | **20.25 GiB** | | 772 개 · 5.14 GiB |

- 폐기 후보의 근거별 내역: **문서에 기각·무효 기록이 있는 것 140 개(12.22 GiB)**, 문서 기록 없이 run 이름·설정만으로 무효로 본 것(스모크·점검·디버그·재기동 전 실행) 32 개(552.6 MiB). 뒤쪽은 지시의 정의('문서에 기각 기록')를 엄격히 적용하면 불명이다 — §5.2 에 따로 모았다.
- '솎을 수 있는 체크포인트'는 남기는 run 안에서 **어떤 참조(익스포트 원본·레지스트리·계보·스크립트·문서·평가 산출물)도 걸리지 않은** 중간 저장본이다. run 을 남기더라도 이것만 따로 치울 수 있다. 최근 24 시간 안에 바뀐 run 은 솎기 대상에서 뺐다.

눈여겨볼 것:

- 배포 후보 세 정책: 코드(레지스트리)가 가리키는 run 의 `SOURCE_CHECKPOINT` 가 기대 체크포인트와 같은 것 **3/3**, 익스포트 가중치가 그 체크포인트와 같은 것 **3/3**, `exported/policy.pt`·`policy.onnx` 의 md5 가 배포 매니페스트(`deploy/*.md5`)와 같은 것 **3/3** (§1). 세 정책의 러너 계약 항목은 전부 같다(§1 계약 대조).
- 세 정책 중 0 개가 등록부 `models/` 에 없다. 지금은 git 이 무시하는 `logs/` 와 `deploy/*.md5` 에만 기록이 있다(§8.1·§9).
- **익스포트와 등록부가 어긋난 곳이 있다**: `biped12_stage4/2026-09-22_13-32-53_g10/exported` 는 model_7500 의 익스포트인데 등록부 `stand_crouch_v6` 는 model_6600 (§7.4).
- 브리지 `CHECKPOINTS` 37 키 중 27 키는 가리키는 run 이 로컬에 없다(§7.2).
- 젯슨 스냅숏(2026-10-08 00:25 에 받아 둔 목록, 이 작업에서 접속하지 않음) 기준 필수 파일: 서기(STAND) — 젯슨에 있고 PC 와 같다 · 회복(RECOVERY) — 젯슨에 있고 PC 와 같다 · 걷기(WALK) — 젯슨에 있고 PC 와 같다 (§3.4).
- ⚠ '평가 중'으로 적어 둔 run 에 판정 문구가 생긴 것으로 보인다 — 분류표를 고쳐야 한다: `biped12_stepb0/2026-10-07_22-25-32_w3S3f_from_15800` ← `reports/gen_w3/TORQUE_BUDGET_1007.md`:82; `biped12_stepb0/2026-10-07_22-52-26_w3S3g_from_15800` ← `reports/gen_w3/TRANSITION_W3KE.md`:762

분류 규칙(위에서부터 첫 일치):

1. **배포 후보** — 브리지 `KEW3_SPECS`(합성 구조)가 가리키는 세 run. 이름을 박아 둔 것이 아니라 코드에서 뽑는다.
2. **평가 중 후보** — 사전 등록 문서만 있고 결과가 아직 없는 run, 또는 분류표에 없고 최근 24 시간 안에 생긴 run(자동). 읽기만 했다.
3. **기준선·검증 의존** — 브리지 레지스트리 키가 가리키거나, selftest·tests·현행 스크립트(`jetson/`·`rl_walking/`·`tests/`·`deploy/`·`reports/regress/`·`reports/gen_w3/`)의 **실행 코드 줄**이 직접 읽는 run. 주석·docstring 의 사용 예는 세지 않는다.
4. **기록 보존** — (a) 남기는 run 의 웜스타트 조상(계보), (b) 여러 판정 문서가 수치로 인용하는 대조군, (c) 남기는 정책을 고른 근거 수치(선정 게이트·재현 seed), (d) 이전 최종·2차 후보와 선행 단계, (e) 과거 분석 스크립트가 체크포인트를 직접 읽는 것.
5. **폐기 후보** — 후보(처치)로 평가했고 문서에 기각·중단·무효 기록이 있으며 위 1–4 에 걸리지 않는 run. 근거 문서와 줄 번호를 붙였다. 같은 실험의 처치가 전부 기각된 대조군도 여기에 둔다.
6. **불명** — 판정 기록도 사용처도 찾지 못한 run.

분류표(문서 판정)는 스크립트 안 `CURATED` 에 있고, 실행할 때마다 근거 문구를 문서에서 다시 찾아 줄 번호를 붙인다. 문구가 사라지면 그 run 은 불명으로 내려간다.

## 1. 배포 후보 — 합성 구조의 세 정책

출처: `jetson/scripts/rl_bridge_node.py` 합성 키 stand_kew3_g21a_g24ra_w3s3d15800 (handoff=carry, settle_quiet_s=0.5).

| 역할 | 브리지 키 | run(실제 경로) | 체크포인트 | 학습 태스크 | 학습 시각(체크포인트 첫–끝 저장) | 근거 문서:줄 |
|---|---|---|---|---|---|---|
| 서기(STAND) | `stand_crouch_g21a` | `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a` | model_11300.pt | `Biped12-HangStand-G21a-v0` (확인) | 2026-10-02 21:23 – 2026-10-02 21:50 | `reports/FINAL_REPORT_1005.md`:7 · `reports/gen_g18/G21_VERDICT.md`:24 |
| 회복(RECOVERY) | `(C5_PAIRS[stand_c5kr_g21a_g24ra] 의 회복 경로 — 단독 키 없음)` | `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | model_21793.pt | `Biped12-HangStand-G24Ra-v0` (확인) | 2026-10-04 10:08 – 2026-10-04 10:39 | `reports/gen_c5/G3_SELECT_1004.md`:14 · `reports/FINAL_REPORT_1005.md`:17 |
| 걷기(WALK) | `stand_w3s3d_15800` | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | model_15800.pt | `Biped12-HangStand-W3S3d-v0` (확인) | 2026-10-07 00:11 – 2026-10-07 00:42 | `reports/night_1005/LOG.md`:1453 · `claude2.md`:130 |

| 역할 | 파일 | 크기 [B] | md5 |
|---|---|---:|---|
| 서기(STAND) | `exported/policy.onnx` | 1321751 | `43f113109a369a60e976a391cf8f17cd` |
|  | `exported/policy.pt` | 1332626 | `2464f5cdba2e819d03ffa2d8d03bbb69` |
|  | `exported/SOURCE_CHECKPOINT` | 15 | `5797e0b7b2eabc2f4502900ec8469469` |
|  | `params/env.yaml` | 54800 | `ebda410ec37987f788df920ccac4f264` |
|  | `params/agent.yaml` | 1212 | `8e61e739e6fd6c6536c40e030cb6180c` |
|  | `model_11300.pt (학습 체크포인트 원본)` | 8019007 | `599c47d3c3f190ad91a80a1ab7ff9fa7` |
| 회복(RECOVERY) | `exported/policy.onnx` | 1321751 | `57547697918f679543413a5b24c36555` |
|  | `exported/policy.pt` | 1332626 | `896ca613bc402de02caa8302988f9a68` |
|  | `exported/SOURCE_CHECKPOINT` | 15 | `e24b90e328fadb2a5c7ef9d87721d1a3` |
|  | `params/env.yaml` | 62452 | `7ff27617abdb9dd8ee6155c5d7703eaf` |
|  | `params/agent.yaml` | 1252 | `df9cf6e61457ef195920966519d1a558` |
|  | `model_21793.pt (학습 체크포인트 원본)` | 8019007 | `05e6b8ee4109ef40fd24bd8437bdfcba` |
| 걷기(WALK) | `exported/policy.onnx` | 1321751 | `d908cd92ba0916e790f9e3e28ff1d882` |
|  | `exported/policy.pt` | 1332468 | `10e3ee405e1534ebe67574ddbeb9a2a2` |
|  | `exported/SOURCE_CHECKPOINT` | 15 | `1ad3809e4af40b1334e497b0d3714a7f` |
|  | `params/env.yaml` | 60296 | `41d1e62aed455029be7351a1db43f48f` |
|  | `params/agent.yaml` | 1230 | `8757acfb3435503cec55a32227b75b27` |
|  | `model_15800.pt (학습 체크포인트 원본)` | 8019007 | `3da9f2e5b6c435d6b6df220e3a7969b2` |

점검 결과(오프라인):

- 서기(STAND): `exported/SOURCE_CHECKPOINT` = `model_11300.pt` → 기대값 model_11300.pt 와 **일치**. `exported/policy.pt` 의 가중치(actor 8 텐서)가 `model_11300.pt` 와 **바이트까지 같다** · `policy.onnx` 안에도 같은 가중치 바이트가 들어 있다. 별칭(심볼릭 링크) `logs/rsl_rl/biped12_g22r/2026-10-02_21-23-08_g21a`, `logs/rsl_rl/biped12_stepb0/2026-10-02_21-23-08_g21a`, `logs/rsl_rl/biped12_w1/2026-10-02_21-23-08_g21a`. 평가 산출물 3621 건(그중 model_11300: 3608 건).
- 회복(RECOVERY): `exported/SOURCE_CHECKPOINT` = `model_21793.pt` → 기대값 model_21793.pt 와 **일치**. `exported/policy.pt` 의 가중치(actor 8 텐서)가 `model_21793.pt` 와 **바이트까지 같다** · `policy.onnx` 안에도 같은 가중치 바이트가 들어 있다. 별칭(심볼릭 링크) `logs/rsl_rl/biped12_stage4/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294`. 평가 산출물 2746 건(그중 model_21793: 2702 건).
- 걷기(WALK): `exported/SOURCE_CHECKPOINT` = `model_15800.pt` → 기대값 model_15800.pt 와 **일치**. `exported/policy.pt` 의 가중치(actor 8 텐서)가 `model_15800.pt` 와 **바이트까지 같다** · `policy.onnx` 안에도 같은 가중치 바이트가 들어 있다. 별칭(심볼릭 링크) `logs/rsl_rl/biped12_stage4/2026-10-07_00-11-12_w3S3d_from_s2d`. 평가 산출물 800 건(그중 model_15800: 706 건).

- 가중치 대조가 보는 것은 '같은 숫자가 들어 있다'까지다. onnx 그래프의 연산 구조가 TorchScript 와 같은 출력을 내는지는 이 검사로 알 수 없다(W3S3d 는 `rl_walking/deploy/selftest_w3s3d_onnx.py` 가 따로 본다).

매니페스트 대조(`deploy/*.md5` 의 정책 쪽 줄을 지금 파일과 비교):

| 매니페스트 | 정책·env 줄 | 코드 줄(지금 작업 트리와 같음 / 다름 / 없음) |
|---|---|---|
| `deploy/MANIFEST.md5` | 파일 없음 12 · 일치 13 · 불일치 2 | 28 / 10 / 0 |
| `deploy/c5_1004.md5` | 일치 4 | 0 / 4 / 0 |
| `deploy/c5kr_1005.md5` | 일치 4 | 0 / 4 / 0 |
| `deploy/c5krh_1007.md5` | 정책 줄 없음 | 0 / 0 / 2 |
| `deploy/g21a_1003.md5` | 일치 5 | 0 / 5 / 0 |
| `deploy/head_1008.md5` | 일치 20 | 99 / 1 / 0 |
| `deploy/lat_1006.md5` | 정책 줄 없음 | 4 / 1 / 0 |
| `deploy/w3s3d_1007.md5` | 일치 6 | 0 / 3 / 0 |

- 정책 쪽 '파일 없음'은 0905 이전 세대처럼 run 디렉터리가 이미 로컬에 없는 경우다(§7.2). 코드 줄이 '다름'인 것은 매니페스트가 작성 시점의 코드 스냅숏이기 때문이다 — 지금 코드로 배포하려면 매니페스트를 새로 만들어야 한다.

세 정책의 학습 env.yaml 계약 대조(러너가 한 계약으로 세 정책을 돌릴 수 있는가):

| 항목 | 서기(STAND) | 회복(RECOVERY) | 걷기(WALK) | 같음 |
|---|---|---|---|---|
| 제어 주기 [Hz] (sim dt × decimation) | [50.0, 0.005, 4] | [50.0, 0.005, 4] | [50.0, 0.005, 4] | 예 |
| 기본자세(힙·무릎·발목 앞뒤, 왼쪽 rad) | ["CROUCH_V5", 0.5282, -0.672, 0.4887] | ["CROUCH_V5", 0.5282, -0.672, 0.4887] | ["CROUCH_V5", 0.5282, -0.672, 0.4887] | 예 |
| 액션(클래스·스케일·기본자세 오프셋·슬루 °/틱) | ["SlewJointPositionAction", 0.25, true, 4.0] | ["SlewJointPositionAction", 0.25, true, 4.0] | ["SlewJointPositionAction", 0.25, true, 4.0] | 예 |
| 관측 항(이름:함수×이력) 순서 | ["base_ang_vel:base_ang_vel_lagged×5", "projected_gravity:projected_gravity_lagged×5", "velocity_commands:generated_commands×5", "joint_pos:encoder_jo… | (같음) | (같음) | 예 |
| 관측 필터(actions·joint_vel β·joint_vel_ema·접촉) | ["last_action_lpf", "encoder_joint_vel_lpf", 0.25, "encoder_joint_vel_ema_v2", "feet_contact_real_v2"] | ["last_action_lpf", "encoder_joint_vel_lpf", 0.25, "encoder_joint_vel_ema_v2", "feet_contact_real_v2"] | ["last_action_lpf", "encoder_joint_vel_lpf", 0.25, "encoder_joint_vel_ema_v2", "feet_contact_real_v2"] | 예 |
| 관측 클록 주파수 인자 | 1.1 | 1.1 | 1.1 | 예 |
| 배포 계약 지문(위 항목의 해시) | ecbeb86f | ecbeb86f | ecbeb86f | 예 |
| 체크포인트 차원 obs / critic / act | [320, 335, 12] | [320, 335, 12] | [320, 335, 12] | 예 |
| 자산(USD) | rl_walking/assets/usd_final3/biped12.usd | rl_walking/assets/usd_final3/biped12.usd | rl_walking/assets/usd_final3/biped12.usd | 예 |
| 학습 액션 지연 DR [틱] | [0, 3] | [0, 3] | [1, 5] | 아니오(학습 분포 차이 — 계약 아님) |
| 학습 LPF β 범위 | [0.2, 0.4] | [0.2, 0.4] | [0.2, 0.4] | 예 |
| 명령 범위 vx / vy / wz · 리샘플 [s] · heading_command | [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [8.0, 12.0], true] | [[0.05, 0.15], [0.0, 0.0], [0.0, 0.0], [4.0, 6.0], true] | [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [8.0, 12.0], false] | 아니오(학습 분포 차이 — 계약 아님) |
| 이벤트 수 / 보상 항 수(가중 ≠ 0) / 보상 지문 | [20, 35, "dfb86fd7"] | [23, 51, "5ead3276"] | [23, 53, "f27b1c54"] | 아니오(학습 분포 차이 — 계약 아님) |

- 맞아야 하는 항목이 전부 같다.

계보(웜스타트 체인, `params/agent.yaml` 의 load_run·load_checkpoint — 부모 체크포인트 파일이 전부 로컬에 있다):

- 서기(STAND): g21a ← g19v(model_10700) ← g12(model_9500) ← g11a(model_8699) ← g9(model_5700) ← g6(model_4500)(처음부터 학습)
- 회복(RECOVERY): g24ra ← w2e(model_20294) ← w2d(model_19295) ← w2c(model_17796) ← w2b(model_16797) ← w2a(model_15798) ← stepa0m(model_14299) ← stepa0i(model_11900) ← g21a(model_11300) ← g19v(model_10700) ← g12(model_9500) ← g11a(model_8699) ← g9(model_5700) ← g6(model_4500)(처음부터 학습)
- 걷기(WALK): w3s3d ← w3s2d(model_15694) ← w3s2(model_15195) ← w3s1c(model_14196) ← w3a3(model_13197) ← w3a2(model_12698) ← w3a(model_11699) ← g21a(model_11300) ← g19v(model_10700) ← g12(model_9500) ← g11a(model_8699) ← g9(model_5700) ← g6(model_4500)(처음부터 학습)

## 2. 평가 중 후보 (읽기만 했다)

| run | 체크포인트 | exported* | 근거 | 마지막 변경 |
|---|---|---|---|---|
| `logs/rsl_rl/biped12_stepb0/2026-10-07_22-25-32_w3S3f_from_15800` | 6 (15800–16299) | exported_16100(← model_16100.pt), exported_16299(← model_16299.pt) | W3S3f — 토크 한계 무작위화(0.7–1.0) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음) `reports/gen_w3/TORQUE_BUDGET_1007.md`:55 | 2026-10-07 22:48 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_22-52-26_w3S3g_from_15800` | 6 (15800–16299) | exported_16100(← model_16100.pt), exported_16299(← model_16299.pt) | W3S3g — W3S3e + 쏠림 벌점(w3_lean) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음) `reports/gen_w3/TRANSITION_W3KE.md`:703 | 2026-10-07 23:15 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_23-17-51_w3S3t_from_15800` | 6 (15800–16299) | exported_16299(← model_16299.pt) | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다  | 2026-10-07 23:40 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_23-42-16_w3S3tK_from_15800` | 6 (15800–16299) | exported_16100(← model_16100.pt), exported_16299(← model_16299.pt) | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다  | 2026-10-08 00:26 |
| `logs/rsl_rl/biped12_stepb0/2026-10-08_00-11-19_g24raf_from_21793` | 6 (21800–22292) | exported_22292(← model_22292.pt) | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다  | 2026-10-08 00:35 |
| `logs/rsl_rl/biped12_stepb0/2026-10-08_00-37-31_w3S3fF_from_15800` | 5 (15800–16200) | 없음 | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다  | 2026-10-08 00:54 |

- 학습 중인 run 은 체크포인트가 더 생긴다. 최근 3 분 안에 바뀐 파일은 md5 를 건너뛰었다(쓰는 중일 수 있음).

## 3. 젯슨 쪽 정책 파일 요구 목록

코드에서 확인한 것: 브리지는 정책마다 `<run>/exported/` 에서 `policy.onnx` 를 먼저 찾고(onnxruntime 이 있으면), 없으면 `policy.pt` 를 torch 로 읽는다(`rl_walking/deploy/policy_runner.py` `_load_backend`). 계약(관측 필터·기본자세·W3 명령 범위)은 `<run>/params/env.yaml` 에서 줄 단위로 읽는다(`_ckpt_env_yaml`·`_parse_env_contract`·`_w3_env_contract`). `SOURCE_CHECKPOINT`·`agent.yaml`·`model_N.pt` 는 읽지 않는다. 회복 정책(C5 의 두 번째 정책)은 `policy.*` 만 읽고 계약은 서기 정책 것을 쓴다(`C5Runner`).

현장 확인 필요: 젯슨에 어떤 백엔드가 깔려 있는지는 이 작업에서 확인하지 않았다. 주 작업자 기록(`reports/night_1007/LOG.md` 22:20 상태판)에는 'torch 없음, onnxruntime 1.15.1 → 젯슨 백엔드는 onnx' 로 적혀 있다 — 그렇다면 젯슨에서 실제로 도는 파일은 `policy.onnx` 다.

### 3.1 합성 구조 — 세 정책(이것이 최소 집합)

| 요구하는 쪽 | 파일(저장소 상대 경로 = 젯슨 `~/humanoid_repo` 상대 경로) | 필요 | 크기 [B] | md5(PC) | 젯슨 스냅숏 |
|---|---|---|---:|---|---|
| 서기(STAND)<br>stand_crouch_g21a: '학습 계약' selftest 가 env.yaml 을 필수로 요구<br>콘솔 stand_crouch_g21a<br>콘솔 stand_c5_g21a_g24ra<br>콘솔 stand_c5kr_g21a_g24ra<br>콘솔 stand_c5krh_g21a_g24ra | `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1321751 | `43f113109a369a60e976a391cf8f17cd` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1332626 | `2464f5cdba2e819d03ffa2d8d03bbb69` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/params/env.yaml` | 필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다 | 54800 | `ebda410ec37987f788df920ccac4f264` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | 15 | `5797e0b7b2eabc2f4502900ec8469469` | 없음 |
| 회복(RECOVERY)<br>콘솔 stand_c5_g21a_g24ra 의 회복 정책<br>콘솔 stand_c5kr_g21a_g24ra 의 회복 정책<br>콘솔 stand_c5krh_g21a_g24ra 의 회복 정책 | `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1321751 | `57547697918f679543413a5b24c36555` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1332626 | `896ca613bc402de02caa8302988f9a68` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/params/env.yaml` | 선택 — 코드가 읽지 않는다(회복 정책은 서기 정책 계약을 쓴다). 배포 스크립트는 기록용으로 함께 보냄 | 62452 | `7ff27617abdb9dd8ee6155c5d7703eaf` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | 15 | `e24b90e328fadb2a5c7ef9d87721d1a3` | 같음 |
| 걷기(WALK) | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1321751 | `d908cd92ba0916e790f9e3e28ff1d882` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1332468 | `10e3ee405e1534ebe67574ddbeb9a2a2` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml` | 필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다 | 60296 | `41d1e62aed455029be7351a1db43f48f` | 같음 |
|  | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | 15 | `1ad3809e4af40b1334e497b0d3714a7f` | 같음 |

### 3.2 selftest 필수 — 브리지 `--selftest` 가 PASS 하려면 함께 있어야 하는 것

| 요구하는 쪽 | 파일(저장소 상대 경로 = 젯슨 `~/humanoid_repo` 상대 경로) | 필요 | 크기 [B] | md5(PC) | 젯슨 스냅숏 |
|---|---|---|---:|---|---|
| STAND_DEFAULT=stand_v18b: --selftest '정책 500틱'·'학습 계약'이 파일을 필수로 요구 | `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1168151 | `c577a65a3edbd9f856fbc4b0912f5908` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1179026 | `fac90eb5490b47a77ca76281631af442` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/params/env.yaml` | 필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다 | 41370 | `a2f1f885aa64d01ef77f14e05549df00` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | — | **PC 에 없음** | 없음 |
| stand_crouch_g12: '학습 계약' selftest 가 env.yaml 을 필수로 요구<br>stand_crouch_g12_ema0: '학습 계약' selftest 가 env.yaml 을 필수로 요구<br>콘솔 stand_crouch_g12(기본값)<br>콘솔 stand_crouch_g12_ema0 | `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1321751 | `6c5f8a7e08696cd07b1e5abf328fa616` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1332626 | `dc9cd5867932e15103684055476565ba` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/params/env.yaml` | 필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다 | 54587 | `0c07175148c9fdbbd71d9e1b125d7814` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | — | **PC 에 없음** | 없음 |

### 3.3 콘솔 선택지 — 콘솔 드롭다운에서 고를 수 있는 나머지

| 요구하는 쪽 | 파일(저장소 상대 경로 = 젯슨 `~/humanoid_repo` 상대 경로) | 필요 | 크기 [B] | md5(PC) | 젯슨 스냅숏 |
|---|---|---|---:|---|---|
| 콘솔 stand_crouch_g11a | `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.onnx` | 필수 — 젯슨 실행 파일(onnxruntime 경로) | 1321751 | `a090aa871b654a099ada80b4287046fd` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.pt` | 선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다 | 1332626 | `51f3e2b7b73168ec881312175447684b` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/params/env.yaml` | 필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다 | 54587 | `ab1c12ebc0a55ec3554c669f5eb5680d` | 같음 |
|  | `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/SOURCE_CHECKPOINT` | 선택 — 코드가 읽지 않는다(출처 기록) | — | **PC 에 없음** | 없음 |

### 3.4 젯슨 스냅숏과의 대조 — `reports/night_1007/jetson/inventory_after_md5.txt` (2026-10-08 00:25 에 받아 둔 목록)

이 파일은 이 작업이 만든 것이 아니다(같은 폴더에 있던 기록을 읽기만 했다). **젯슨에는 접속하지 않았고, 스냅숏 뒤에 젯슨이 바뀌었는지는 모른다.**

- 합성 구조에 필요한 파일 12 개 중 스냅숏과 **같음 11**, 다름 0, 스냅숏에 **없음 1**.
  - 서기(STAND) `biped12_stage4/2026-10-02_21-23-08_g21a`: 같음 3 · 없음 1
  - 회복(RECOVERY) `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294`: 같음 4
  - 걷기(WALK) `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d`: 같음 4
- 스냅숏의 `logs/rsl_rl/` 파일 67 개 전체: PC 와 같음 24 · **다름 2** · PC 에 없음 41(PC 에서는 그 run 디렉터리가 이미 없다).
  - **다름**: `logs/rsl_rl/biped12_stage4/2026-09-22_13-32-53_g10/exported/policy.onnx` — 젯슨 `43c4c3c76e0c5e613dadf7a27e30f21b` / PC `8769437fa40017a1ad3620049599bff6`
  - **다름**: `logs/rsl_rl/biped12_stage4/2026-09-22_13-32-53_g10/exported/policy.pt` — 젯슨 `238a62a02771aa22decaa365861e2a48` / PC `5bdf5bff682d73f3b82d81dd983073d7`
  - PC 에 run 이 없는 것 15 개: `biped12_stage4/2026-08-03_03-18-21`, `biped12_stage4/2026-08-04_13-13-49`, `biped12_stage4/2026-08-08_21-24-48`, `biped12_stage4/2026-08-09_03-59-22`, `biped12_stage4/2026-09-04_16-43-02`, `biped12_stage4/2026-09-04_20-05-14`, `biped12_stage4/2026-09-04_21-48-32`, `biped12_stage4/2026-09-05_17-57-21`, `biped12_stage4/2026-09-09_12-33-03`, `biped12_stage4/2026-09-09_19-15-42`, `biped12_stage4/2026-09-13_00-40-49`, `biped12_stage4/2026-09-18_00-14-13_s9c`, `biped12_stage4/2026-09-18_01-08-37_s9c2`, `biped12_stage4/2026-09-18_02-52-27_s9c3`, `biped12_stage4/2026-09-18_19-06-31_s9d`

- 합성 구조 최소 집합의 크기: 7.8 MiB(세 정책의 onnx·pt·env.yaml·SOURCE_CHECKPOINT 전부).
- 합성 키는 `--experimental-w3 --experimental-kew3` 를 함께 줄 때만 등록된다(기본 꺼짐). 이 목록은 정책 쪽 파일만이다 — 코드 파일(브리지·러너·`w3_deploy.py`·`kew3_runner.py`·`c5_switch.py` 등)은 배포 패키지 담당의 몫이다.
- `policy_runner_stage4.py --selftest` 의 실모델 시험은 기본 경로가 없으면 `logs/rsl_rl/biped12_stage4/*/exported/policy.pt` 가운데 **이름순 마지막**을 쓴다(젯슨에서는 `policy.pt` 가 하나도 없으면 실패). 무엇이 마지막인지는 그 폴더의 디렉터리·링크 구성에 달려 있다.

## 4. 분류 전체 표

열 설명 — 체크포인트: 개수(최소–최대) · exported: 디렉터리(← SOURCE_CHECKPOINT) · 평가: log_picture 결과 파일 머리말에 이 run 이 나온 횟수 · 핵심: 참조가 걸린 체크포인트 · 솎기: 남기는 run 안에서 참조가 없는 체크포인트 수.

### 4.1 배포 후보 — 3 개 · 271.1 MiB

| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |
|---|---|---|---:|---|---|---:|---|---:|---|---|
| `biped12_stage4/2026-10-02_21-23-08_g21a` | 2026-10-02 | 10 (10700–11599) | 82.7 MiB | exported(← model_11300.pt) | HangStand-G21a-v0 | 3621 | 11300, 11599 | 8 | 합성 구조의 서기(STAND) 정책 — KEW3_SPECS[stand_kew3_g21a_g24ra_w3s3d15800].ke=stand_c5kr_g21a_g24ra → C5_PAIRS[..][0] / 문서 기록: 최종 STAND 후보 — G21a-11300 = 최종 STAND 후보 | `reports/FINAL_REPORT_1005.md`:7 · `reports/gen_g18/G21_VERDICT.md`:24 |
| `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | 2026-10-04 | 16 (20300–21793) | 131.8 MiB | exported(← model_21793.pt) | HangStand-G24Ra-v0 | 2746 | 21400, 21500, 21600, 21700, 21793 | 11 | 합성 구조의 회복(RECOVERY) 정책 — KEW3_SPECS[stand_kew3_g21a_g24ra_w3s3d15800].ke=stand_c5kr_g21a_g24ra → C5_PAIRS[..][1] / 문서 기록: 최종 RECOVERY 정책 — G24R-a-21793 = C5/KE 의 회복 정책(단독 사전 기준은 FAIL, 전환 구조에서 채택) | `reports/gen_c5/G3_SELECT_1004.md`:14 · `reports/FINAL_REPORT_1005.md`:17 |
| `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | 2026-10-07 | 6 (15700–16193) | 56.5 MiB | exported(← model_15800.pt)<br>exported_16100(← model_16100.pt)<br>exported_16193(← model_16193.pt) | HangStand-W3S3d-v0 | 800 | 15700, 15800, 15900, 16000, 16100, 16193 | — | 합성 구조의 걷기(WALK) 정책 — KEW3_SPECS[stand_kew3_g21a_g24ra_w3s3d15800].w3 / 문서 기록: 최종 WALK/TURN 후보 — W3S3d-15800 = 실물 WALK/TURN 1차 후보 | `reports/night_1005/LOG.md`:1453 · `claude2.md`:130 |

### 4.2 평가 중 후보 — 6 개 · 307.0 MiB

| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |
|---|---|---|---:|---|---|---:|---|---:|---|---|
| `biped12_stepb0/2026-10-07_22-25-32_w3S3f_from_15800` | 2026-10-07 | 6 (15800–16299) | 54.2 MiB | exported_16100(← model_16100.pt)<br>exported_16299(← model_16299.pt) | HangStand-W3S3f-v0 | 0 | 15800, 15900, 16000, 16100, 16200, 16299 | — | W3S3f — 토크 한계 무작위화(0.7–1.0) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음) / ⚠ 문서에 판정 문구가 생긴 것으로 보인다(`reports/gen_w3/TORQUE_BUDGET_1007.md`:82 '## 9. W3S3f 판정 (1007 23:09, `x13_summ.py --prefix tr_s3f<CK>') — 분류표(CURATED)를 고쳐야 한다 | `reports/gen_w3/TORQUE_BUDGET_1007.md`:55 |
| `biped12_stepb0/2026-10-07_22-52-26_w3S3g_from_15800` | 2026-10-07 | 6 (15800–16299) | 54.2 MiB | exported_16100(← model_16100.pt)<br>exported_16299(← model_16299.pt) | HangStand-W3S3g-v0 | 0 | 15800, 15900, 16000, 16100, 16200, 16299 | — | W3S3g — W3S3e + 쏠림 벌점(w3_lean) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음) / ⚠ 문서에 판정 문구가 생긴 것으로 보인다(`reports/gen_w3/TRANSITION_W3KE.md`:762 '### 5.19b W3S3g 판정 — 쏠림 벌점 (1007 23:54, `s3g_judge.py`, seed') — 분류표(CURATED)를 고쳐야 한다 | `reports/gen_w3/TRANSITION_W3KE.md`:703 |
| `biped12_stepb0/2026-10-07_23-17-51_w3S3t_from_15800` | 2026-10-07 | 6 (15800–16299) | 51.7 MiB | exported_16299(← model_16299.pt) | HangStand-W3S3t-v0(추정) | 0 | 15800, 15900, 16000, 16100, 16200, 16299 | — | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다 |  |
| `biped12_stepb0/2026-10-07_23-42-16_w3S3tK_from_15800` | 2026-10-07 | 6 (15800–16299) | 54.3 MiB | exported_16100(← model_16100.pt)<br>exported_16299(← model_16299.pt) | HangStand-W3S3tK-v0(추정) | 0 | 15800, 15900, 16000, 16100, 16200, 16299 | — | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다 |  |
| `biped12_stepb0/2026-10-08_00-11-19_g24raf_from_21793` | 2026-10-08 | 6 (21800–22292) | 51.5 MiB | exported_22292(← model_22292.pt) | HangStand-G24RaF-v0(추정) | 0 | 21800, 21900, 22000, 22100, 22200, 22292 | — | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다 |  |
| `biped12_stepb0/2026-10-08_00-37-31_w3S3fF_from_15800` | 2026-10-08 | 5 (15800–16200) | 41.1 MiB | — | HangStand-W3S3fF-v0(추정) | 0 | 15800, 15900, 16000, 16100, 16200 | — | 분류표에 없고 최근 24 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다 |  |

### 4.3 기준선·검증 의존 — 12 개 · 1.55 GiB

| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |
|---|---|---|---:|---|---|---:|---|---:|---|---|
| `biped12_flat/2026-07-19_05-32-42` | 2026-07-19 | 31 (6000–8997) | 144.5 MiB | exported(← 표시 없음) | 미상 | 5 | 8997 | 30 | 현행 코드가 직접 읽음: rl_walking/scripts/eval_hang.py / 문서 기록: 문서 인용(구세대) — 3단계 '이식용 최종'(구 자산) | `RL보행_명령어모음.txt`:181 |
| `biped12_flat/2026-07-25_21-59-19` | 2026-07-25 | 4 (12000–12300) | 20.4 MiB | exported(← 표시 없음) | 미상 | 1 | 12300 | 3 | 현행 코드가 직접 읽음: rl_walking/deploy/policy_runner.py, rl_walking/deploy/sim_validate_runner.py / 문서 기록: 문서 인용(구세대) — '신자산 이식용 최종'(model_12300) | `RL보행_명령어모음.txt`:185 |
| `biped12_stage4/2026-09-05_19-37-39` | 2026-09-05 | 0 | 2.3 MiB | exported(← 표시 없음) | Velocity-Stage4V2-Stand-V18-v0 | 111 | — | — | 브리지 레지스트리 키 stand_v18b / 문서 기록: 이전 데뷔 후보 — stand_v18b '데뷔 1순위'(0905~0906) — 체크포인트 원본은 없고 exported 만 남아 있음 | `models/README.md`:17 |
| `biped12_stage4/2026-09-21_20-30-32_g9` | 2026-09-21 | 41 (4500–8499) | 330.1 MiB | exported(← 표시 없음) | HangStand-G9-v0 | 74 | 5300, 5700, 8499 | 38 | 브리지 레지스트리 키 stand_crouch_v5 / 문서 기록: 이전 실물 정책 — stand_crouch_v5(G9-5700) — 0922 실물 정책 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:424 |
| `biped12_stage4/2026-09-22_13-32-53_g10` | 2026-09-22 | 31 (5700–8699) | 250.3 MiB | exported(← 표시 없음) | HangStand-G10-v0 | 34 | 6600, 8400, 8699 | 28 | 브리지 레지스트리 키 stand_crouch_v6 / 문서 기록: 이전 실물 정책 — stand_crouch_v6(G10-6600) — 0922 실물 정책 | `reports/real_0922/G10_VERDICT.md`:1 |
| `biped12_stage4/2026-09-28_23-19-13_g11a` | 2026-09-28 | 31 (5700–8699) | 250.2 MiB | exported(← 표시 없음) | HangStand-G11a-v0 | 77 | 6500, 7500, 8699 | 28 | 브리지 레지스트리 키 stand_crouch_g11a / 콘솔 선택지 stand_crouch_g11a / 문서 기록: 이전 비교 시험 후보 — G11a-8699 — 실물 비교 시험 후보(승격 아님), G12 의 부모 | `reports/gen_g11/VERDICT.md`:13 |
| `biped12_stage4/2026-09-29_14-32-50_g12` | 2026-09-29 | 21 (8700–10698) | 170.3 MiB | exported(← 표시 없음) | HangStand-G12-v0 | 319 | 9500, 10000, 10698 | 18 | 브리지 레지스트리 키 stand_crouch_g12, stand_crouch_g12_ema0 / 콘솔 선택지 stand_crouch_g12, stand_crouch_g12_ema0 / 현행 코드가 직접 읽음: deploy/deploy_g21a_1003.sh, tests/test_bridge_ema_contract.py | `reports/gen_g11/G12_VERDICT.md`:9 |
| `biped12_stage4/2026-10-01_01-46-29_g16o` | 2026-10-01 | 21 (9500–11499) | 167.9 MiB | — | HangStand-G16o-v0(추정) | 32 | 10500, 10800, 11100, 11499 | 17 | 현행 코드가 직접 읽음: tests/test_bridge_ema_contract.py / 문서 기록: 기각 — G16 판정 — H-o(관측 v2 학습 비퇴보) 기각 | `reports/gen_g16/G16_VERDICT.md`:13 |
| `biped12_stepb0/2026-10-05_18-18-43_w2q_seed1_from_w2h24791` | 2026-10-05 | 9 (24800–25590) | 72.6 MiB | — | HangStand-W2q-v0(추정) | 21 | 25100, 25300, 25590 | 6 | 브리지 레지스트리 키 walk_w2q25590_hturn2 (exported 없음 — 키만 등록) / 현행 코드가 직접 읽음: rl_walking/deploy/hturn2.py / 문서 기록: 이전 후보(HTURN2) — W2q 25590 = HTURN2(앞 보행 + 회전 피드포워드) 실험 키의 정책 — 1006 '방향 전환 기술' 후보 | `reports/night_1005/LOG.md`:477 · `reports/gen_walk/WALK_FINAL_1005.md`:7 |
| `biped12_stepb0/2026-10-06_11-40-53_w3A_s1_from_g21a` | 2026-10-06 | 5 (11300–11699) | 40.4 MiB | — | HangStand-W3A-v0(추정) | 20 | 11699 | 4 | 브리지 레지스트리 키 stand_w3a_11699 (exported 없음 — 키만 등록) / 현행 코드가 직접 읽음: rl_walking/deploy/w3_deploy.py / 문서 기록: 선행 단계 — W3-A 400 — 보조 0·참조 0 제자리 교대걸음 자립(W3 계보의 뿌리, 실험 키 stand_w3a_11699) | `reports/night_1005/LOG.md`:925 |
| `biped12_stepb0/2026-10-06_15-06-14_w3S1c_from_w3A3` | 2026-10-06 | 11 (13200–14196) | 89.0 MiB | — | HangStand-W3S1c-v0(추정) | 276 | 13400, 13600, 13900, 14196 | 7 | 현행 코드가 직접 읽음: reports/gen_w3/stopest_run.sh / 문서 기록: 선행 단계 — W3S1c-14196 — Stage 1 앞/뒤 통과(W3S2 의 부모) | `reports/night_1005/LOG.md`:1185 |
| `biped12_stepb0/2026-10-07_21-51-10_w3S3e_from_s3d` | 2026-10-07 | 6 (15800–16299) | 54.2 MiB | exported_16100(← model_16100.pt)<br>exported_16299(← model_16299.pt) | HangStand-W3S3e-v0 | 0 | 15800, 15900, 16000, 16100, 16200, 16299 | — | 대조군(진행 중인 실험) — W3S3e — 후보로서는 '통과 후보 없음'(1007 22:52). 다만 진행 중인 W3S3g 의 사전 등록 대조가 이 run 의 같은 체크포인트(16100·16299)다 — W3S3g 판정이 끝나면 폐기 후보 | `reports/gen_w3/S3E_DESIGN.md`:53 · `reports/gen_w3/TRANSITION_W3KE.md`:705 |

### 4.4 기록 보존 — 43 개 · 4.90 GiB

| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |
|---|---|---|---:|---|---|---:|---|---:|---|---|
| `biped12_flat/2026-07-19_04-21-02` | 2026-07-19 | 31 (0–2999) | 148.6 MiB | exported(← 표시 없음) | 미상 | 1 | 2999 | 30 | 문서 인용(구세대) — 런1 — 걸음새(케이던스 2.0/s), 이식 가이드가 exported 를 가리킴 | `RL보행_명령어모음.txt`:180 · `docs/RL보행_이식가이드.md`:137 |
| `biped12_flat/2026-07-19_04-59-54` | 2026-07-19 | 31 (3000–5998) | 143.5 MiB | exported(← 표시 없음) | 미상 | 1 | 5998 | 30 | 계보 — 2026-07-19_05-32-42 의 웜스타트 원본(model_5998.pt) — 문서 판정만 보면 불명 / 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-19_06-04-22` | 2026-07-19 | 26 (9000–11496) | 121.2 MiB | exported(← 표시 없음) | 미상 | 0 | 11496 | 25 | 계보 — 2026-07-19_07-15-42 의 웜스타트 원본(model_11496.pt) — 문서 판정만 보면 불명 / 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-19_07-15-42` | 2026-07-19 | 26 (11500–13995) | 121.2 MiB | exported(← 표시 없음) | 미상 | 1 | 13995 | 25 | 문서 인용(구세대) — 계단 최종(구 자산) | `RL보행_명령어모음.txt`:183 · `docs/RL보행_최종보고서.md`:53 |
| `biped12_flat/2026-07-25_20-58-39` | 2026-07-25 | 8 (9000–9700) | 36.8 MiB | — | 미상 | 0 | 9700 | 7 | 계보 — 2026-07-25_21-07-04 의 웜스타트 원본(model_9700.pt) — 문서 판정만 보면 불명 / 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-25_21-07-04` | 2026-07-25 | 4 (9700–9999) | 19.6 MiB | exported(← 표시 없음) | 미상 | 1 | 9999 | 3 | 문서 인용(구세대) — 신 URDF 미세조정(model_9999) | `RL보행_명령어모음.txt`:184 |
| `biped12_flat/2026-07-25_21-23-38` | 2026-07-25 | 21 (10000–12000) | 98.0 MiB | exported(← 표시 없음) | 미상 | 1 | 12000 | 20 | 계보 — 2026-07-25_21-59-19 의 웜스타트 원본(model_12000.pt) — 문서 판정만 보면 불명 / 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-25_22-04-32` | 2026-07-25 | 13 (12300–13499) | 61.1 MiB | exported(← 표시 없음) | 미상 | 1 | 13499 | 12 | 문서 인용(구세대) — 폴리시 계속(model_13499) — 대안 | `RL보행_명령어모음.txt`:188 |
| `biped12_flat/2026-07-25_23-44-35` | 2026-07-25 | 11 (14000–14994) | 52.3 MiB | exported(← 표시 없음) | 미상 | 1 | 14994 | 10 | 문서 인용(구세대) — 계단 신자산 재정합(model_14994) | `RL보행_명령어모음.txt`:190 |
| `biped12_g22r/2026-10-03_22-10-34_g22r2_cR1toR4_seed1_from_g21a11300` | 2026-10-03 | 19 (11300–13099) | 153.8 MiB | — | HangStand-G22R2-v0(추정) | 24 | 12500, 13099 | 17 | 참조 데이터 출처 — G22R2-13099 롤아웃이 스텝 기준 데이터(ref_steps_g22r2_s1.npz)·이륙 직전 진단의 출처 — 판정 자체는 탈락 | `reports/gen_g22r/verdict_g22r2_13099_s1.md`:31 · `reports/gen_step/CORRECTION_1004_forward_steps.md`:4 외 1 |
| `biped12_stage4/2026-09-20_03-30-06_g2` | 2026-09-20 | 61 (0–5999) | 488.6 MiB | exported(← 표시 없음) | HangStand-G2-v0(추정) | 14 | 4300, 5500, 5600, 5999 | 57 | 문서 인용 — 0920 재판정 표의 비교 행(커리큘럼 의존 결론 철회의 근거) | `reports/gen_g6/SUMMARY.md`:24 |
| `biped12_stage4/2026-09-20_03-38-43_g3` | 2026-09-20 | 61 (0–5999) | 488.6 MiB | exported(← 표시 없음) | HangStand-G3-v0(추정) | 17 | 5300, 5400, 5500, 5600, 5999 | 56 | 문서 인용 — 0920 재판정 표의 비교 행 | `reports/gen_g6/SUMMARY.md`:22 |
| `biped12_stage4/2026-09-20_15-06-59_g6` | 2026-09-20 | 61 (0–5999) | 488.6 MiB | exported(← 표시 없음) | HangStand-G6-v0(추정) | 225 | 3500, 4200, 4500, 4600, 4900, 5999 | 55 | 이전 기준선 — G6-4500 = 0920 재평가 기준선(이후 G9 의 웜스타트 원본) | `reports/gen_g6/SUMMARY.md`:19 |
| `biped12_stage4/2026-09-30_12-05-46_g12c` | 2026-09-30 | 21 (9500–11499) | 167.8 MiB | — | 미상 | 256 | 10000, 10100, 10400, 10500, 10700, 10800, 11000, 11100, 11499 | 12 | 대조군 — 이어 학습 대조(G12-9500 에서 변경 없이 2000 반복) — G13·G14·G15·G16·G17 판정 문서가 수치로 인용 | `reports/gen_g13/G13_VERDICT.md`:4 · `reports/gen_g13/G14_VERDICT.md`:9 외 3 |
| `biped12_stage4/2026-10-02_12-26-09_g19v` | 2026-10-02 | 16 (9500–10999) | 128.1 MiB | — | HangStand-G19v-v0(추정) | 58 | 10100, 10400, 10700, 10999 | 12 | 선행 단계 — G19v-10700 = G21 설계의 기준(G21a 의 웜스타트 원본) | `reports/gen_g18/G21_VERDICT.md`:1 |
| `biped12_stepb0/2026-10-04_05-06-37_stepa0i_seed1_from_g21a11300` | 2026-10-04 | 7 (11300–11900) | 56.5 MiB | — | HangStand-StepA0i-v0(추정) | 4 | 11700, 11900 | 5 | 계보 — 2026-10-04_05-51-40_stepa0m_seed1_from_a0i11900 의 웜스타트 원본(model_11900.pt) — 문서 판정만 보면 폐기 후보 / 기각 — STEP-A0i — 사전 규칙 FAIL(단, 첫 양성 신호 — A0m 의 부모) | `reports/gen_step/stepa0i/decision.md`:1 |
| `biped12_stepb0/2026-10-04_05-51-40_stepa0m_seed1_from_a0i11900` | 2026-10-04 | 25 (11900–14299) | 202.4 MiB | — | HangStand-StepA0m-v0(추정) | 25 | 12300, 12500, 12900, 13100, 13500, 13900, 14299 | 18 | 선행 단계 — STEP-A0m — 스텝 primitive PASS(서기 보존 FAIL, 배포 후보 아님) — W2·G24R·W2om 계열의 출발점 | `reports/gen_step/stepa0m/decision.md`:1 |
| `biped12_stepb0/2026-10-04_06-50-24_w2a_seed1_from_a0m14299` | 2026-10-04 | 16 (14300–15798) | 129.1 MiB | — | HangStand-W2a-v0(추정) | 7 | 14700, 15798 | 14 | 계보 — 2026-10-04_08-04-28_w2b_seed1_from_w2a15798 의 웜스타트 원본(model_15798.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-1 — 부분 실패 | `reports/gen_step/w2a/decision.md`:1 |
| `biped12_stepb0/2026-10-04_08-04-28_w2b_seed1_from_w2a15798` | 2026-10-04 | 11 (15800–16797) | 88.7 MiB | — | HangStand-W2b-v0(추정) | 7 | 16797 | 10 | 계보 — 2026-10-04_08-42-24_w2c_seed1_from_w2b16797 의 웜스타트 원본(model_16797.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-2 — FAIL | `reports/gen_step/w2b/decision.md`:1 |
| `biped12_stepb0/2026-10-04_08-42-24_w2c_seed1_from_w2b16797` | 2026-10-04 | 11 (16800–17796) | 88.8 MiB | — | HangStand-W2c-v0(추정) | 8 | 17796 | 10 | 계보 — 2026-10-04_09-06-04_w2d_seed1_from_w2c17796 의 웜스타트 원본(model_17796.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-3 — FAIL | `reports/gen_step/w2c/decision.md`:1 |
| `biped12_stepb0/2026-10-04_09-06-04_w2d_seed1_from_w2c17796` | 2026-10-04 | 16 (17800–19295) | 129.3 MiB | — | HangStand-W2d-v0(추정) | 11 | 19295 | 15 | 계보 — 2026-10-04_09-42-49_w2e_seed1_from_w2d19295 의 웜스타트 원본(model_19295.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-4 — 사전 기준 FAIL(앞 회복 스텝 첫 출현) | `reports/gen_step/w2d/decision.md`:1 |
| `biped12_stepb0/2026-10-04_09-42-49_w2e_seed1_from_w2d19295` | 2026-10-04 | 11 (19300–20294) | 88.8 MiB | — | HangStand-W2e-v0(추정) | 11 | 20294 | 10 | 계보 — 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 의 웜스타트 원본(model_20294.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-5 — FAIL | `reports/gen_step/w2e/decision.md`:1 |
| `biped12_stepb0/2026-10-04_10-45-08_w2f_seed1_from_w2e20294` | 2026-10-04 | 16 (20300–21793) | 129.3 MiB | — | HangStand-W2f-v0(추정) | 10 | 21793 | 15 | 계보 — 2026-10-04_12-00-09_w2g_seed1_from_w2f21793 의 웜스타트 원본(model_21793.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-6 — FAIL | `reports/gen_step/w2f/decision.md`:1 |
| `biped12_stepb0/2026-10-04_12-00-09_w2g_seed1_from_w2f21793` | 2026-10-04 | 16 (21800–23292) | 129.3 MiB | — | HangStand-W2g-v0(추정) | 10 | 23292 | 15 | 계보 — 2026-10-04_13-26-17_w2h_seed1_from_w2g23292 의 웜스타트 원본(model_23292.pt) — 문서 판정만 보면 폐기 후보 / 기각 — W2-7 — FAIL | `reports/gen_step/w2g/decision.md`:1 |
| `biped12_stepb0/2026-10-04_13-26-17_w2h_seed1_from_w2g23292` | 2026-10-04 | 16 (23300–24791) | 129.0 MiB | — | HangStand-W2h-v0(추정) | 78 | 24600, 24700, 24791 | 13 | 이전 최종 WALK 후보 — W2-8 24791 = 1005 사이클의 최종 WALK 후보(이후 W3 로 대체) — W2q·WT·WV·W2o 의 부모 | `reports/gen_walk/WALK_FINAL_1005.md`:14 · `reports/FINAL_REPORT_1005.md`:11 |
| `biped12_stepb0/2026-10-04_17-30-51_g24rcont_seed1_from_g24ra21793` | 2026-10-04 | 16 (21800–23292) | 129.4 MiB | — | HangStand-G24Ra-v0 | 39 | 21800, 21900, 22000, 22100, 22200, 22300, 22400, 22700, 23292 | 7 | 대조군 — G24R-a-cont — 이어 학습 대조(G24R-e·g·h 판정과 G3 선정 문서가 인용) | `reports/gen_c5/g24rcont/hypothesis.md`:1 · `reports/gen_c5/g24re/decision.md`:8 외 1 |
| `biped12_stepb0/2026-10-04_20-41-39_g24rcontlr_seed1_from_g24ra21793` | 2026-10-04 | 16 (21800–23292) | 129.4 MiB | — | 미상 | 70 | 22200, 22700, 23000, 23100, 23200, 23292 | 10 | 선정 근거 — G24R-a-cont-lr — 회복 체크포인트 선정 게이트(G3)의 비교 대상 4 체크포인트 | `reports/gen_c5/G3_SELECT_1004.md`:4 · `reports/gen_c5/G3_SELECT_1004.md`:14 |
| `biped12_stepb0/2026-10-04_22-53-20_t1_seed1_from_g24ra21793` | 2026-10-04 | 6 (21800–22292) | 48.4 MiB | — | 미상 | 128 | 22000, 22200, 22292 | 3 | 대조군 — T1(안전 PPO 무변경 이어 학습) — 토너먼트 1·2·R′ 세 결과 문서의 같은 반복 대조 | `reports/gen_c5/tourney1/RESULT.md`:5 · `reports/gen_c5/tourney2/RESULT.md`:2 외 1 |
| `biped12_stepb0/2026-10-05_03-06-22_wt0_seed1_from_w2h24791` | 2026-10-05 | 5 (24800–25190) | 40.3 MiB | — | 미상 | 9 | 25190 | 4 | 대조군 — WT0 = W2hSafe 대조 — WT·WV 두 결과 문서가 대조 수치로 인용 | `reports/gen_walk/wt1/RESULT.md`:8 · `reports/gen_walk/wv1/RESULT.md`:4 |
| `biped12_stepb0/2026-10-05_19-14-45_w2qc_seed1_from_w2q25590` | 2026-10-05 | 6 (25600–26089) | 48.4 MiB | — | 미상 | 2 | 25900, 26089 | 4 | 대조군 — w2qc = W2q 이어 학습 대조(W2q2·WR 판정과 보행 최종 표가 인용) | `reports/gen_walk/W2q2_RESULT_WR_PLAN.md`:7 · `reports/gen_walk/WALK_FINAL_1005.md`:10 |
| `biped12_stepb0/2026-10-06_12-03-39_w3A2_from_w3A400` | 2026-10-06 | 11 (11700–12698) | 88.9 MiB | — | HangStand-W3A2-v0(추정) | 24 | 12200, 12698 | 9 | 선행 단계 — W3-A2 — W3 계보(W3-A3 의 부모) | `reports/night_1005/LOG.md`:1046 |
| `biped12_stepb0/2026-10-06_13-03-09_w3A3_from_w3A2` | 2026-10-06 | 6 (12700–13197) | 48.5 MiB | — | HangStand-W3A3-v0(추정) | 46 | 12900, 13197 | 4 | 선행 단계 — W3-A3 — W3 계보(W3S1c 의 부모) | `reports/night_1005/LOG.md`:1111 |
| `biped12_stepb0/2026-10-06_16-17-24_w3S1c2_from_w3A3` | 2026-10-06 | 11 (13200–14196) | 89.0 MiB | — | 미상 | 30 | 13400, 13600, 13900, 14196 | 7 | 선정 근거 — W3S1c 학습 seed 2 — closure 1 재현성 판정의 수치 | `reports/night_1005/LOG.md`:1218 · `reports/night_1005/LOG.md`:1287 |
| `biped12_stepb0/2026-10-06_19-11-43_w3S2_from_s1c` | 2026-10-06 | 11 (14200–15195) | 89.3 MiB | — | HangStand-W3S2-v0(추정) | 82 | 14300, 14600, 15195 | 8 | 선행 단계 — W3S2-15195 — 옆걸음 획득(W3S2d·W3S3s2 의 부모) | `reports/night_1005/LOG.md`:1339 |
| `biped12_stepb0/2026-10-06_19-58-48_w3S1c3_from_w3A3` | 2026-10-06 | 11 (13200–14196) | 89.0 MiB | — | 미상 | 12 | 13600, 14196 | 9 | 선정 근거 — W3S1c 학습 seed 3 — 'Stage 1 레시피는 seed 민감' 판정의 수치 | `reports/night_1005/LOG.md`:1362 |
| `biped12_stepb0/2026-10-06_21-11-23_w3S3s2_from_s2` | 2026-10-06 | 11 (15200–16194) | 89.6 MiB | — | HangStand-W3S3s2-v0(추정) | 56 | 15300, 15600, 16194 | 8 | 선행 단계 — W3S3s2 — 제자리 회전 획득(W3S3d 의 보상 정의 출처, 이후 W3S3d 로 합침) | `reports/night_1005/LOG.md`:1398 · `reports/night_1005/LOG.md`:1433 |
| `biped12_stepb0/2026-10-06_22-14-02_w3S2d_from_s2` | 2026-10-06 | 6 (15200–15694) | 48.7 MiB | — | HangStand-W3S2d-v0(추정) | 24 | 15694 | 5 | 선행 단계 — W3S2d — 지연 DR 1–5(W3S3d 의 부모) | `reports/night_1005/LOG.md`:1422 |
| `biped12_stepb0/2026-10-07_01-44-18_w3S4d_from_s3d` | 2026-10-07 | 2 (15800–15900) | 16.5 MiB | — | HangStand-W3S4d-v0(추정) | 4 | 15800, 15900 | — | 선행 단계 — W3S4d 첫 실행 — 디스크 가득 참으로 15934 에서 끊김, 15900 에서 w3S4d2 로 이어 학습 | `reports/night_1005/LOG.md`:1471 |
| `biped12_stepb0/2026-10-07_01-53-16_w3S4d2_from_s4d` | 2026-10-07 | 5 (15900–16299) | 40.9 MiB | — | 미상 | 181 | 15900, 16000, 16100, 16200, 16299 | — | 2차 후보 — W3S4d-16299 = 혼합 명령 2차 후보 | `claude2.md`:131 · `reports/night_1005/LOG.md`:1509 |
| `biped12_stepb0/2026-10-07_05-33-59_w3S3d_seed2` | 2026-10-07 | 4 (15700–15993) | 32.6 MiB | — | HangStand-W3S3d-v0 | 0 | 15700, 15800, 15900, 15993 | — | 선정 근거 — W3S3d 학습 seed 2–5 재현 — '15800 은 운 좋은 점, 레시피 기대 낙상 2–3 %' 판단의 수치(평가 원자료는 scratchpad repro/) | `reports/gen_w3/REPRO.md`:25 |
| `biped12_stepb0/2026-10-07_06-43-54_w3S3d_seed3` | 2026-10-07 | 4 (15700–15993) | 32.6 MiB | — | HangStand-W3S3d-v0 | 0 | 15700, 15800, 15900, 15993 | — | 선정 근거 — W3S3d 학습 seed 2–5 재현 — '15800 은 운 좋은 점, 레시피 기대 낙상 2–3 %' 판단의 수치(평가 원자료는 scratchpad repro/) | `reports/gen_w3/REPRO.md`:25 |
| `biped12_stepb0/2026-10-07_10-15-20_w3S3d_seed4` | 2026-10-07 | 4 (15700–15993) | 32.7 MiB | — | HangStand-W3S3d-v0 | 0 | 15700, 15800, 15900, 15993 | — | 선정 근거 — W3S3d 학습 seed 2–5 재현 — '15800 은 운 좋은 점, 레시피 기대 낙상 2–3 %' 판단의 수치(평가 원자료는 scratchpad repro/) | `reports/gen_w3/REPRO.md`:25 |
| `biped12_stepb0/2026-10-07_10-28-16_w3S3d_seed5` | 2026-10-07 | 4 (15700–15993) | 32.7 MiB | — | HangStand-W3S3d-v0 | 0 | 15700, 15800, 15900, 15993 | — | 선정 근거 — W3S3d 학습 seed 2–5 재현 — '15800 은 운 좋은 점, 레시피 기대 낙상 2–3 %' 판단의 수치(평가 원자료는 scratchpad repro/) | `reports/gen_w3/REPRO.md`:25 |

### 4.5 폐기 후보 — 172 개 · 12.76 GiB

| run | 체크포인트 | 용량 | 평가 | 역할 — 사유 | 근거(문서:줄) |
|---|---|---:|---:|---|---|
| `biped12_stage4/2026-09-20_13-39-35_g5` | 61 (0–5999) | 488.6 MiB | 9 | 기각 — 명목 0/256 '진짜 실패' | `reports/gen_g6/SUMMARY.md`:25 |
| `biped12_stage4/2026-09-21_09-08-58_g8` | 41 (4500–8499) | 330.0 MiB | 98 | 기각 — G8 체크포인트 선택 — 적격 후보 없음(익스포트 사본은 deploy/cand/g8_8499 에 따로 있음) | `reports/gen_g6/RECOVERY_EVAL_0920.md`:279 |
| `biped12_stage4/2026-09-20_21-20-53_g7` | 41 (4500–8499) | 327.2 MiB | 82 | 기각 — G7 은 승격 대상이 아니다 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:195 |
| `biped12_stage4/2026-09-28_23-19-53_g11b` | 31 (5700–8699) | 247.6 MiB | 27 | 기각 — G11 판정 — 적격 후보 없음, G11b 는 G11a 대비 일관된 이득 없음 | `reports/gen_g11/VERDICT.md`:18 · `reports/gen_g11/VERDICT.md`:31 |
| `biped12_stage4/2026-09-30_12-05-18_g13` | 21 (9500–11499) | 215.0 MiB | 108 | 기각 — G13 판정 — 통과 후보 없음 | `reports/gen_g13/G13_VERDICT.md`:1 |
| `biped12_stage4/2026-09-30_15-26-19_g14b` | 21 (9500–11499) | 214.9 MiB | 108 | 기각 — G14 판정 — 가설 미지지, 통과 후보 없음 | `reports/gen_g13/G14_VERDICT.md`:1 |
| `biped12_w1/2026-10-03_21-20-36_w1b_cS0toS3_seed1_from_g21a11300` | 25 (11300–13699) | 203.3 MiB | 2 | 기각 — W1b — 스텝 0 | `reports/gen_step/w2a/hypothesis.md`:2 |
| `biped12_w1/2026-10-03_19-38-00_w1_cS0toS3_seed1_from_g21a11300` | 25 (11300–13699) | 203.3 MiB | 2 | 기각 — W1 — 착지 0(걷지 않음) | `reports/gen_g22r/VERDICT_1003.md`:23 |
| `biped12_stepb0/2026-10-04_15-13-03_w2j_seed1_from_w2h24791` | 21 (24800–26790) | 169.8 MiB | 2 | 기각 — W2-10 — FAIL, 요(회전) 갈래 일시 정지 | `reports/gen_step/w2j/decision.md`:1 |
| `biped12_stage4/2026-09-30_17-49-39_g15w` | 21 (9500–11499) | 167.9 MiB | 54 | 기각 — G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| `biped12_stage4/2026-10-01_04-46-25_g16d` | 21 (9500–11499) | 167.9 MiB | 32 | 기각 — G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| `biped12_stage4/2026-10-01_02-54-55_g16a` | 21 (9500–11499) | 167.9 MiB | 32 | 기각 — G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| `biped12_stage4/2026-10-01_02-53-25_g16o43` | 21 (9500–11499) | 167.9 MiB | 32 | 기각 — G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| `biped12_stage4/2026-09-30_22-37-30_g15k` | 21 (9500–11499) | 167.8 MiB | 54 | 기각 — G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| `biped12_stage4/2026-09-30_21-06-44_g15d` | 21 (9500–11499) | 167.8 MiB | 54 | 기각 — G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| `biped12_stage4/2026-09-30_19-18-11_g15q` | 21 (9500–11499) | 167.8 MiB | 54 | 기각 — G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| `biped12_stage4/2026-09-29_11-08-21_g11c` | 21 (8700–10698) | 167.7 MiB | 36 | 기각 — G11c 판정 — 승격 없음(정착↔뒤 회복 교환) | `reports/gen_g11/VERDICT.md`:48 |
| `biped12_stage4/2026-09-30_19-22-14_g15u` | 21 (9500–11499) | 167.7 MiB | 54 | 기각 — G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| `biped12_stage4/2026-09-30_15-26-19_g14a` | 21 (9500–11499) | 167.7 MiB | 108 | 기각 — G14 판정 — 가설 미지지, 통과 후보 없음 | `reports/gen_g13/G14_VERDICT.md`:1 |
| `biped12_g22r/2026-10-03_19-01-59_g22r_cR1toR4_seed1_from_g21a11300` | 19 (11300–13099) | 153.8 MiB | 24 | 기각 — G22R·W1 판정 — 통과 0 | `reports/gen_g22r/VERDICT_1003.md`:1 |
| `biped12_stepb0/2026-10-04_12-35-53_g24rc_seed1_from_g24ra21793` | 16 (21800–23292) | 130.1 MiB | 35 | 기각 — G24R-c — 사전 기준 FAIL, C5 안에서도 차이 없음 | `reports/gen_step/g24rc/decision.md`:1 · `reports/gen_c5/c5c/decision.md`:1 |
| `biped12_stepb0/2026-10-04_19-24-24_g24rh_seed1_from_g24ra21793` | 16 (21800–23292) | 129.4 MiB | 18 | 기각 — G24R-h — 잠정 FAIL | `reports/gen_c5/g24rh/decision.md`:1 |
| `biped12_stepb0/2026-10-04_17-20-12_g24re_seed1_from_g24ra21793` | 16 (21800–23292) | 129.4 MiB | 18 | 기각 — G24R-e — FAIL | `reports/gen_c5/g24re/decision.md`:1 |
| `biped12_stepb0/2026-10-04_16-06-52_g24rd_seed1_from_g24ra21793` | 16 (21800–23292) | 129.3 MiB | 32 | 기각 — G24R-d — FAIL(앞 회복 유의 악화) | `reports/gen_c5/g24rd/decision.md`:1 |
| `biped12_stepb0/2026-10-04_11-23-07_g24rb_seed1_from_g24ra21793` | 16 (21800–23292) | 129.3 MiB | 11 | 기각 — G24R-b — FAIL | `reports/gen_step/g24rb/decision.md`:1 |
| `biped12_stepb0/2026-10-06_03-18-36_w2om_from_w2q25590` | 16 (25600–27089) | 129.3 MiB | 5 | 기각 — W2om — 사전 기준 KILL | `reports/night_1005/LOG.md`:165 |
| `biped12_stepb0/2026-10-03_23-49-02_stepb0_seed1_from_g21a11300` | 16 (11300–12799) | 129.3 MiB | 7 | 기각 — STEP-B0 — 탈락: 스텝 0 | `reports/gen_g22r/STEPB0_CRITERIA.md`:23 |
| `biped12_stepb0/2026-10-04_07-21-05_g23ra_seed1_from_a0m14299` | 16 (14300–15798) | 129.0 MiB | 21 | 기각 — G23R-a — FAIL | `reports/gen_step/g23ra/decision.md`:1 |
| `biped12_stepb0/2026-10-05_06-30-46_w2o_seed1_from_w2h24791` | 16 (24800–26290) | 128.5 MiB | 6 | 기각 — W2o — 사건 보상 제거, 1000 에서 보행 붕괴 → 중단 | `reports/gen_walk/W2o_RESULT.md`:1 |
| `biped12_stage4/2026-10-02_14-08-27_g20s` | 16 (9500–10999) | 128.2 MiB | 97 | 기각 — G20d·G20s — 둘 다 정식 C0 탈락 | `reports/gen_g18/G20_DESIGN_CRITERIA.md`:38 |
| `biped12_stage4/2026-10-02_07-54-56_g18j` | 16 (9500–10999) | 128.1 MiB | 56 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_14-07-27_g20d` | 16 (9500–10999) | 128.1 MiB | 97 | 기각 — G20d·G20s — 둘 다 정식 C0 탈락 | `reports/gen_g18/G20_DESIGN_CRITERIA.md`:38 |
| `biped12_stage4/2026-10-02_12-27-11_g19n` | 16 (9500–10999) | 128.1 MiB | 52 | 기각 — G19n — 회복 상실한 정숙 정책, 승격 금지 | `reports/gen_g18/G19_DESIGN_CRITERIA.md`:31 |
| `biped12_stage4/2026-10-02_11-14-11_g18f43` | 16 (9500–10999) | 128.0 MiB | 60 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_10-23-53_g18q` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_09-06-01_g18f` | 16 (9500–10999) | 128.0 MiB | 55 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_09-03-53_g18d` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_06-48-23_g18o` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함) | `reports/gen_g18/G18_VERDICT.md`:16 |
| `biped12_stage4/2026-10-02_05-25-23_g17r43` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stage4/2026-10-02_05-23-24_g17rp` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stage4/2026-10-02_04-03-27_g17o43` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stage4/2026-10-02_04-01-27_g17p` | 16 (9500–10999) | 128.0 MiB | 52 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stage4/2026-10-02_02-41-23_g17r` | 16 (9500–10999) | 128.0 MiB | 56 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stage4/2026-10-02_02-39-53_g17o` | 16 (9500–10999) | 128.0 MiB | 76 | 기각 — G17 판정 — 선별 C0 통과 0/24 | `reports/gen_g17/G17_VERDICT.md`:6 |
| `biped12_stepb0/2026-10-04_19-24-01_g24rg_seed1_from_g24ra21793` | 13 (21800–23000) | 105.3 MiB | 18 | 기각 — G24R-g — fail-fast 중단 | `reports/gen_c5/g24rg/decision.md`:1 |
| `biped12_stage4/2026-10-01_04-43-55_g16a43` | 13 (9500–10700) | 104.0 MiB | 0 | 기각 — G16a seed 43 — 학습 붕괴 재현, 10719 반복에서 중단 | `reports/gen_g16/G16_VERDICT.md`:78 |
| `biped12_stepb0/2026-10-06_01-59-15_woms_seed1_scratch` | 12 (0–1100) | 94.3 MiB | 2 | 기각 — WOMS — KILL | `reports/night_1005/LOG.md`:114 |
| `biped12_stepb0/2026-10-06_06-04-22_pa1_from_g24ra21793` | 11 (21800–22792) | 90.1 MiB | 32 | 기각 — PA1 — KILL | `reports/night_1005/LOG.md`:409 |
| `biped12_stepb0/2026-10-06_05-07-08_rh2_from_g24ra21793` | 11 (21800–22792) | 89.1 MiB | 22 | 기각 — RH2 — KILL | `reports/night_1005/LOG.md`:369 |
| `biped12_stepb0/2026-10-06_04-09-13_rd3_from_g24ra21793` | 11 (21800–22792) | 89.1 MiB | 22 | 기각 — RD(rd5·대조 rd3) — 사전 성공 미달, RDS KILL 로 계보 종료 | `reports/night_1005/LOG.md`:258 · `reports/night_1005/LOG.md`:280 |
| `biped12_stepb0/2026-10-06_02-42-32_rd5_from_g24ra21793` | 11 (21800–22792) | 89.1 MiB | 26 | 기각 — RD(rd5·대조 rd3) — 사전 성공 미달, RDS KILL 로 계보 종료 | `reports/night_1005/LOG.md`:258 · `reports/night_1005/LOG.md`:280 |
| `biped12_stepb0/2026-10-06_17-54-42_w3S1d2_from_w3A3` | 11 (13200–14196) | 89.0 MiB | 24 | 기각 — W3S1d seed 2(대칭 범위) — 명령 범위 비대칭 가설 기각 | `reports/night_1005/LOG.md`:1276 |
| `biped12_stepb0/2026-10-06_13-03-09_w3S1b_from_w3A2` | 11 (12700–13697) | 89.0 MiB | 49 | 기각 — W3S1b — 뒤 추종 FAIL(W3S1c 로 대체) | `reports/night_1005/LOG.md`:1123 |
| `biped12_stepb0/2026-10-06_05-10-43_wsteer_from_w2q25590` | 11 (25600–26589) | 88.9 MiB | 6 | 기각 — WSTEER — KILL | `reports/night_1005/LOG.md`:395 |
| `biped12_stepb0/2026-10-06_06-58-58_cstab_s2_from_w2q25590` | 11 (25600–26589) | 88.9 MiB | 0 | 분리 실험 — CSTAB — 학습 틀 안정성 진단(결론 '확률적 사건'), 후보 아님 | `reports/night_1005/LOG.md`:500 |
| `biped12_stepb0/2026-10-06_06-02-22_w2qC_from_w2q25590` | 11 (25600–26589) | 88.9 MiB | 5 | 무효 — 대조 C(W2q 이어학습) — 학습 붕괴, 대조 무효 | `reports/night_1005/LOG.md`:420 |
| `biped12_stepb0/2026-10-06_11-49-08_w3CI_from_w3C400` | 11 (11700–12698) | 88.9 MiB | 6 | 분리 실험 — W3-CI — 엔진 2 번째 경로 확인으로 종료(추가 학습 없음) | `reports/night_1005/LOG.md`:1059 |
| `biped12_stepb0/2026-10-06_02-37-15_womx_seed1_scratch` | 11 (0–1000) | 86.6 MiB | 2 | 기각 — WOMX — KILL | `reports/night_1005/LOG.md`:160 |
| `biped12_stage4/2026-10-02_20-20-43_g21d` | 10 (10700–11599) | 80.7 MiB | 71 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_18-48-38_g21c2` | 10 (10700–11599) | 80.6 MiB | 71 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_18-46-38_g21c1` | 10 (10700–11599) | 80.6 MiB | 71 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_16-33-38_g21b` | 10 (10700–11599) | 80.6 MiB | 26 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-03_00-35-40_g22a` | 10 (10700–11599) | 80.2 MiB | 26 | 기각 — G22 판정 — H-22 미지지, 정식 C0 탈락 | `reports/gen_g18/G22_DESIGN_CRITERIA.md`:30 · `reports/gen_g18/G22_DESIGN_CRITERIA.md`:31 |
| `biped12_stage4/2026-10-03_01-13-13_g22ws43` | 10 (10700–11599) | 80.2 MiB | 26 | 기각 — G22 판정 — H-22 미지지, 정식 C0 탈락 | `reports/gen_g18/G22_DESIGN_CRITERIA.md`:30 · `reports/gen_g18/G22_DESIGN_CRITERIA.md`:31 |
| `biped12_stage4/2026-10-03_00-38-38_g22w` | 10 (10700–11599) | 80.2 MiB | 71 | 기각 — G22 판정 — H-22 미지지, 정식 C0 탈락 | `reports/gen_g18/G22_DESIGN_CRITERIA.md`:30 · `reports/gen_g18/G22_DESIGN_CRITERIA.md`:31 |
| `biped12_stage4/2026-10-02_19-36-57_g21a3s43` | 10 (10700–11599) | 80.2 MiB | 26 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_17-49-38_g21a3` | 10 (10700–11599) | 80.2 MiB | 71 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_17-48-08_g21a2` | 10 (10700–11599) | 80.2 MiB | 26 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_16-38-08_g21a1` | 10 (10700–11599) | 80.1 MiB | 26 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-10-02_15-16-09_g21a_void_explosion` | 10 (10700–11599) | 80.1 MiB | 0 | 무효 — G21a 1차 — 물리 폭발 전염으로 무효(재실행이 배포본) | `reports/gen_g18/G21_DESIGN_CRITERIA.md`:43 |
| `biped12_stage4/2026-10-02_15-15-08_g21o` | 10 (10700–11599) | 80.1 MiB | 26 | 기각 — G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음) | `reports/gen_g18/G21_VERDICT.md`:27 |
| `biped12_stage4/2026-09-21_03-33-56_g8d` | 9 (4500–5299) | 71.7 MiB | 2 | 분리 실험 — action-rate 배율 × env LPF 분리 실험(각 800 반복) — 결론이 문서에 있고 후보가 아님 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:227 |
| `biped12_stage4/2026-09-21_03-20-41_g8c` | 9 (4500–5299) | 71.7 MiB | 2 | 분리 실험 — action-rate 배율 × env LPF 분리 실험(각 800 반복) — 결론이 문서에 있고 후보가 아님 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:227 |
| `biped12_stage4/2026-09-21_03-08-04_g8b` | 9 (4500–5299) | 71.7 MiB | 2 | 분리 실험 — action-rate 배율 × env LPF 분리 실험(각 800 반복) — 결론이 문서에 있고 후보가 아님 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:227 |
| `biped12_stage4/2026-09-21_02-55-23_g8a` | 9 (4500–5299) | 71.7 MiB | 3 | 분리 실험 — action-rate 배율 × env LPF 분리 실험(각 800 반복) — 결론이 문서에 있고 후보가 아님 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:227 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-40-32_g22r_cR1toR4_seed1_from_g21a11300` | 8 (11300–12000) | 64.9 MiB | 0 | 무효 — G22R 첫 기동분(같은 이름으로 19:01 에 다시 시작) — _void_1003_bench 로 옮겨져 있음 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_03-51-07_stepa0_seed1_from_g21a11300` | 7 (11300–11900) | 57.1 MiB | 3 | 기각 — STEP-A0 — FAIL(fail-fast 600) | `reports/gen_step/stepa0/decision.md`:1 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-04-57_stepa0b_seed1_from_g21a11300` | 7 (11300–11900) | 57.1 MiB | 2 | 기각 — STEP-A0b — A0·A0b·A0c 3회 같은 실패 | `reports/gen_step/stepa0c/decision.md`:3 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_03-26-05_stepb1_seed1_from_g21a11300` | 7 (11300–11900) | 56.8 MiB | 3 | 기각 — STEP-B1 — FAIL(fail-fast) | `reports/gen_step/stepb1/decision.md`:1 |
| `biped12_stepb0/2026-10-04_05-37-09_stepa0k_seed1_from_a0i11900` | 7 (11900–12500) | 56.6 MiB | 2 | 기각 — STEP-A0k — FAIL | `reports/gen_step/stepa0k/decision.md`:1 |
| `biped12_stepb0/2026-10-06_07-38-58_wsteer2_s2_from_w2q25590` | 7 (25600–26189) | 56.6 MiB | 2 | 기각 — WSTEER2 — KILL | `reports/night_1005/LOG.md`:537 |
| `biped12_stepb0/2026-10-06_07-36-58_wsteer2_s1_from_w2q25590` | 7 (25600–26189) | 56.6 MiB | 2 | 기각 — WSTEER2 — KILL | `reports/night_1005/LOG.md`:537 |
| `biped12_stepb0/2026-10-06_10-20-05_w2omF_from_w2omP14898` | 7 (14900–15497) | 56.5 MiB | 2 | 기각 — W2omF — KILL(학습 전체 붕괴) | `reports/night_1005/LOG.md`:735 |
| `biped12_stepb0/2026-10-06_08-46-57_w2omPs2_from_a0m14299` | 7 (14300–14898) | 56.5 MiB | 1 | 기각 — W2omP seed 2 — 양성 신호만 재현, W2omP 계보는 KILL 로 끝남 | `reports/night_1005/LOG.md`:655 · `reports/night_1005/LOG.md`:637 |
| `biped12_stepb0/2026-10-06_09-45-24_w2omR2_from_w2omP14898` | 7 (14900–15497) | 56.5 MiB | 2 | 기각 — W2omR1·R2 — 둘 다 KILL | `reports/night_1005/LOG.md`:716 |
| `biped12_stepb0/2026-10-06_09-43-24_w2omR1_from_w2omP14898` | 7 (14900–15497) | 56.5 MiB | 2 | 기각 — W2omR1·R2 — 둘 다 KILL | `reports/night_1005/LOG.md`:716 |
| `biped12_stepb0/2026-10-06_08-30-20_w2omP_from_a0m14299` | 7 (14300–14898) | 56.5 MiB | 2 | 기각 — W2omP — KILL | `reports/night_1005/LOG.md`:637 |
| `biped12_stepb0/2026-10-04_05-23-28_stepa0ic_seed1_from_a0i11900` | 7 (11900–12500) | 56.4 MiB | 1 | 기각 — STEP-A0i-c — FAIL(진행 정체) | `reports/gen_step/stepa0ic/decision.md`:1 |
| `biped12_flat/2026-07-25_23-00-29` | 11 (12300–13299) | 51.9 MiB | 1 | 기각 — 대칭 교정 시도 2회 — 문서가 '실패 기록'으로 적음 | `RL보행_명령어모음.txt`:189 |
| `biped12_flat/2026-07-25_22-36-10` | 11 (12300–13299) | 51.9 MiB | 1 | 기각 — 대칭 교정 시도 2회 — 문서가 '실패 기록'으로 적음 | `RL보행_명령어모음.txt`:189 |
| `biped12_stepb0/2026-10-07_04-59-14_w3S4dDb_from_s4d` | 6 (16300–16798) | 49.1 MiB | 72 | 기각 — W3S4dD run1·run2 — 통과 후보 없음 | `reports/gen_w3/WD_DESIGN.md`:149 · `claude2.md`:134 |
| `biped12_stepb0/2026-10-07_03-20-15_w3S4dD_from_s4d` | 6 (16300–16798) | 49.1 MiB | 90 | 기각 — W3S4dD run1·run2 — 통과 후보 없음 | `reports/gen_w3/WD_DESIGN.md`:149 · `claude2.md`:134 |
| `biped12_stepb0/2026-10-07_16-49-39_w3S3dR_from_s3d` | 6 (15800–16299) | 49.1 MiB | 44 | 기각 — W3S3dR — 불합격, 기준선 15800 유지 | `reports/gen_w3/ROBUST_DESIGN.md`:49 |
| `biped12_stepb0/2026-10-07_02-43-05_w3S3dC2_from_s3d` | 6 (16200–16692) | 48.8 MiB | 90 | 기각 — c2(W3S3dC2) — KILL | `reports/gen_w3/C2_DESIGN.md`:357 · `claude2.md`:134 |
| `biped12_stepb0/2026-10-06_08-46-40_w2omP2_from_w2omP14898` | 6 (14900–15400) | 48.6 MiB | 1 | 기각 — W2omP2 — KILL | `reports/night_1005/LOG.md`:667 |
| `biped12_stepb0/2026-10-05_04-08-12_r2p_seed1_from_g24ra21793` | 6 (21800–22292) | 48.6 MiB | 20 | 기각 — R′ 토너먼트 — 전부 중단 | `reports/gen_c5/rcap/RESULT.md`:1 |
| `biped12_stepb0/2026-10-06_19-05-14_w3S1e_from_s1c2` | 6 (14200–14695) | 48.6 MiB | 12 | 기각 — W3S1e — 미채택 | `reports/night_1005/LOG.md`:1303 |
| `biped12_stepb0/2026-10-05_03-54-56_r2_seed1_from_g24ra21793` | 6 (21800–22292) | 48.6 MiB | 20 | 기각 — R′ 토너먼트 r1n(run 이름은 r2) — 전부 중단 | `reports/gen_c5/rcap/NOTE_1005.md`:4 · `reports/gen_c5/rcap/RESULT.md`:1 |
| `biped12_stepb0/2026-10-06_18-18-49_w3S1c2x_from_s1c2` | 6 (14200–14695) | 48.6 MiB | 12 | 같은 실험 대조 — S1c2x — W3S1e 의 이어학습 대조(처치 미채택) | `reports/night_1005/LOG.md`:1303 |
| `biped12_stepb0/2026-10-04_23-48-56_t2b_seed1_from_t222292` | 6 (22300–22791) | 48.5 MiB | 36 | 기각 — T2·T2b(G25 충격 커리큘럼) — 중단 | `reports/gen_c5/tourney1/RESULT.md`:19 |
| `biped12_stepb0/2026-10-04_22-53-50_t2_seed1_from_g24ra21793` | 6 (21800–22292) | 48.5 MiB | 36 | 기각 — T2·T2b(G25 충격 커리큘럼) — 중단 | `reports/gen_c5/tourney1/RESULT.md`:19 |
| `biped12_stepb0/2026-10-05_00-48-54_t3_seed1_from_g24ra21793` | 6 (21800–22292) | 48.4 MiB | 72 | 기각 — T3(G25D 지연 DR 0–5) — 중단 | `reports/gen_c5/tourney2/RESULT.md`:1 |
| `biped12_stepb0/2026-10-05_04-28-53_r1p_seed1_from_g24ra21793` | 6 (21800–22292) | 48.4 MiB | 20 | 기각 — R′ 토너먼트 — 전부 중단 | `reports/gen_c5/rcap/RESULT.md`:1 |
| `biped12_stepb0/2026-10-04_23-48-25_t1b_seed1_from_t122292` | 6 (22300–22791) | 48.4 MiB | 36 | 같은 실험 대조 — T1b — 토너먼트 1b 의 대조(처치 T2b = G25 계열 중단) | `reports/gen_c5/tourney1/RESULT.md`:19 |
| `biped12_stepb0/2026-10-05_18-53-19_w2q2_seed1_from_w2q25590` | 6 (25600–26089) | 48.4 MiB | 2 | 기각 — W2q2 — 중단 | `reports/gen_walk/W2q2_RESULT_WR_PLAN.md`:3 |
| `biped12_stage4/2026-10-02_01-53-44_g17o_void` | 6 (9500–10000) | 48.1 MiB | 0 | 무효 — G17 1차 학습 폐기(접촉 관측 계약 불일치로 중단) | `reports/gen_g17/G17_CRITERIA.md`:54 |
| `biped12_stepb0/2026-10-05_22-33-07_wom_seed1_from_w2q25590` | 6 (25600–26089) | 47.2 MiB | 3 | 기각 — WOM — 중단 | `reports/night_1005/LOG.md`:71 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-43-59_stepa0g_seed1_from_g21a11300` | 5 (11300–11700) | 41.0 MiB | 2 | 기각 — STEP-A0g — FAIL | `reports/gen_step/stepa0g/decision.md`:1 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-20-02_stepa0c_seed1_from_g21a11300` | 5 (11300–11700) | 41.0 MiB | 2 | 기각 — STEP-A0c — FAIL | `reports/gen_step/stepa0c/decision.md`:1 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-30-29_stepa0e_seed1_from_g21a11300` | 5 (11300–11700) | 41.0 MiB | 2 | 기각 — STEP-A0e — FAIL | `reports/gen_step/stepa0e/decision.md`:1 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_03-41-02_stepi0_seed1_from_g21a11300` | 5 (11300–11700) | 40.9 MiB | 2 | 기각 — STEP-I0 — FAIL(fail-fast) | `reports/gen_step/stepi0/decision.md`:1 |
| `biped12_stepb0/2026-10-06_09-26-31_w2omI_from_a0m14299` | 5 (14300–14700) | 40.5 MiB | 2 | 기각 — W2omI — KILL | `reports/night_1005/LOG.md`:688 |
| `biped12_stepb0/2026-10-06_09-14-04_w2omP3_from_w2omP14898` | 5 (14900–15300) | 40.4 MiB | 2 | 기각 — W2omP3 — KILL | `reports/night_1005/LOG.md`:670 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-56-07_stepa0h_seed1_from_g21a11300` | 5 (11300–11700) | 40.3 MiB | 2 | 기각 — STEP-A0h — FAIL | `reports/gen_step/stepa0h/decision.md`:1 |
| `biped12_stepb0/2026-10-04_08-27-46_g23rb_seed1_from_a0m14299` | 5 (14300–14700) | 40.3 MiB | 10 | 기각 — G23R-b — FAIL(진행 정체로 중단) | `reports/gen_step/g23rb/decision.md`:1 |
| `biped12_stepb0/2026-10-05_02-50-46_wt1_seed1_from_w2h24791` | 5 (24800–25190) | 40.3 MiB | 4 | 기각 — WT1 — 중단(학습 루프 내 요 제어 이득 없음) | `reports/gen_walk/wt1/RESULT.md`:1 |
| `biped12_stepb0/2026-10-04_14-52-57_w2i_seed1_from_w2h24791` | 5 (24800–25200) | 40.3 MiB | 0 | 기각 — W2-9 — 조기 중단(판정 불가) | `reports/gen_step/w2i/decision.md`:1 |
| `biped12_stepb0/2026-10-06_11-29-33_w3U_s1_from_g21a` | 5 (11300–11699) | 40.3 MiB | 2 | 기각 — W3-U — KILL | `reports/night_1005/LOG.md`:913 |
| `biped12_stepb0/2026-10-06_11-10-14_w3C_s1_from_g21a` | 5 (11300–11699) | 40.3 MiB | 4 | 기각 — W3-C — Stage 0 토너먼트에서 A 가 1위(C 는 참조 0 에서 소멸) | `reports/night_1005/LOG.md`:936 |
| `biped12_stepb0/2026-10-05_05-39-39_wv3_seed1_from_w2h24791` | 5 (24800–25190) | 40.3 MiB | 3 | 기각 — WV3 — 중단 | `reports/gen_walk/wv1/RESULT.md`:17 |
| `biped12_stepb0/2026-10-05_04-39-49_wv2_seed1_from_w2h24791` | 5 (24800–25190) | 40.2 MiB | 2 | 기각 — WV2 — 중단 | `reports/gen_walk/wv1/RESULT.md`:14 |
| `biped12_stepb0/2026-10-05_04-14-50_wv1_seed1_from_w2h24791` | 5 (24800–25190) | 40.2 MiB | 2 | 기각 — WV1 — 중단 | `reports/gen_walk/wv1/RESULT.md`:1 |
| `biped12_stage4/2026-10-02_01-55-14_g17r_void` | 5 (9500–9900) | 40.2 MiB | 0 | 무효 — G17 1차 학습 폐기(접촉 관측 계약 불일치로 중단) | `reports/gen_g17/G17_CRITERIA.md`:54 |
| `biped12_stepb0/2026-10-06_04-47-57_w2omL_from_w2omA26400` | 4 (26400–26700) | 32.6 MiB | 1 | 기각 — W2omL — KILL | `reports/night_1005/LOG.md`:287 |
| `biped12_stepb0/2026-10-06_08-17-53_w2omM_from_w2omA26400` | 4 (26400–26700) | 32.5 MiB | 1 | 기각 — W2omM — KILL | `reports/night_1005/LOG.md`:589 |
| `biped12_stepb0/2026-10-06_04-12-57_w2omA_from_w2om26100` | 4 (26100–26400) | 32.4 MiB | 4 | 기각 — W2omA — KILL | `reports/night_1005/LOG.md`:200 |
| `biped12_stepb0/2026-10-05_22-15-47_w2gs_seed1_from_w2q25590` | 4 (25600–25900) | 32.4 MiB | 1 | 기각 — W2gs — 중단 | `reports/night_1005/LOG.md`:36 |
| `biped12_stepb0/2026-10-05_19-41-44_wr_seed1_from_w2q25590` | 4 (25600–25889) | 32.2 MiB | 2 | 기각 — WR — 같은 반복 대조와 구별되지 않음 | `reports/gen_walk/WALK_FINAL_1005.md`:11 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-57-51_smoke2_W1` | 2 (11300–11359) | 16.2 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-56-56_smoke2_G22R` | 2 (11300–11359) | 16.2 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stepb0/2026-10-04_17-20-35_g24rf_seed1_from_g24ra21793` | 2 (21800–21900) | 16.2 MiB | 0 | 무효 — G24R-f — 중단(옆 외란이 앞/뒤 사건을 대체하는 교란) | `reports/gen_c5/g24rcont/hypothesis.md`:5 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-35-07_bench_duo_w` | 2 (11300–11339) | 16.2 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-34-17_bench_solo_w` | 2 (11300–11339) | 16.2 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-35-07_bench_duo_g` | 2 (11300–11339) | 16.2 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-33-31_bench_solo_g` | 2 (11300–11339) | 16.1 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-37-59_bench_duo2k_w` | 2 (11300–11329) | 16.1 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-37-59_bench_duo2k_g` | 2 (11300–11329) | 16.1 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-37-03_bench_solo4k` | 2 (11300–11329) | 16.1 MiB | 0 | 무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음 | `reports/gen_g22r/DESIGN_CRITERIA.md`:13 |
| `_void_1003_bench/biped12_stepb0/2026-10-03_23-47-55_smoke_sb0` | 2 (11300–11329) | 16.1 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 30, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_15-13-17_smoke_g21d_void` | 2 (10700–10739) | 16.0 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 40, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_w1/2026-10-03_18-33-02_smoke_W1` | 2 (11300–11302) | 16.0 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-32-48_smoke_G22R` | 2 (11300–11302) | 16.0 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_03-31-23_smoke_si0` | 2 (11300–11304) | 16.0 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 5, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_g22r/2026-10-03_21-26-01_smoke_g22r2` | 2 (11300–11301) | 16.0 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_14-08-12_smoke_g20s_void` | 2 (9500–9501) | 15.9 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_14-07-12_smoke_g20d_void` | 2 (9500–9501) | 15.9 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_12-25-54_smoke_g19v_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_12-26-54_smoke_g19n_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_09-05-43_smoke_g18f_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_06-48-05_smoke_g18q_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_06-47-23_smoke_g18j_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_06-47-07_smoke_g18o_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-10-02_06-47-41_smoke_g18d_void` | 2 (9500–9501) | 15.8 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-21_02-48-57_g8chk` | 2 (4500–4559) | 15.7 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-21_02-53-57_g8chk2` | 2 (4500–4539) | 15.6 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 40, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-22_13-32-00_g10chk` | 2 (5700–5714) | 15.6 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 15, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-21_02-52-40_g8dbg3` | 2 (4500–4514) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 15, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-21_02-51-51_g8dbg2` | 2 (4500–4511) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-20_21-18-04_g7smoke2` | 2 (4500–4511) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-20_21-16-37_g7smoke` | 2 (4500–4511) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `biped12_stage4/2026-09-21_02-50-47_g8dbg` | 2 (4500–4502) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_08-03-19_smoke_g23rb` | 2 (11300–11319) | 15.5 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 20, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_05-28-25_smoke_a0k` | 2 (11300–11307) | 15.4 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 8, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_06-23-06_smoke_W2a` | 2 (11300–11302) | 15.4 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_06-23-23_smoke_G23Ra` | 2 (11300–11302) | 15.4 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_12-01-06_smoke_g24rc` | 2 (11300–11302) | 15.4 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_08-38-27_smoke_w2d` | 2 (11300–11302) | 15.4 MiB | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 | (문서 근거 없음 — 이름·설정) |
| `_void_1003_bench/biped12_stepb0/2026-10-04_04-41-35_stepa0f_seed1_from_g21a11300` | 1 (11300–11300) | 8.5 MiB | 0 | 무효 — A0f — 시작 직후 중단, 무효 | `reports/gen_step/stepa0f/decision.md`:1 |
| `biped12_stage4/2026-10-03_00-36-00_g22c_void_stopped` | 1 (10700–10700) | 8.4 MiB | 0 | 무효 — G22c — 권고 순서에 따라 중단(무효) | `reports/gen_g18/G22_DESIGN_CRITERIA.md`:23 |
| `biped12_stepb0/2026-10-05_05-35-47_w2w_seed1_from_w2h24791` | 1 (24800–24800) | 8.3 MiB | 0 | 기각 — W2w — 비구속 변수(폐기 목록) | `reports/FINAL_REPORT_1005.md`:95 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_05-22-16_stepa0j_seed1_from_g21a11300` | 1 (11300–11300) | 7.9 MiB | 0 | 무효 — A0j — 시작 직후 중단·보류 | `reports/gen_step/stepa0i/decision.md`:6 |
| `biped12_stage4/2026-09-30_12-10-00_g13init` | 1 (9500–9500) | 3.3 MiB | 0 | 기각 — G13 의 시작 체크포인트(관측 차원을 넓힌 G12-9500 사본) — G13 과 같은 판정 | `reports/gen_g13/G13_VERDICT.md`:1 |

### 4.6 불명 — 8 개 · 484.7 MiB

| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |
|---|---|---|---:|---|---|---:|---|---:|---|---|
| `biped12_flat/2026-07-19_04-18-49` | 2026-07-19 | 2 (0–99) | 9.0 MiB | — | 미상 | 0 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-19_04-50-06` | 2026-07-19 | 9 (3000–3800) | 41.2 MiB | — | 미상 | 0 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-19_05-28-26` | 2026-07-19 | 1 (6000–6000) | 4.6 MiB | — | 미상 | 0 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-19_06-36-02` | 2026-07-19 | 25 (11500–13900) | 114.9 MiB | — | 미상 | 0 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-25_21-46-08` | 2026-07-25 | 10 (12000–12900) | 47.4 MiB | exported(← 표시 없음) | 미상 | 1 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_flat/2026-07-25_22-34-20` | 2026-07-25 | 1 (12300–12300) | 4.4 MiB | — | 미상 | 0 | — | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다) |  |
| `biped12_stage4/2026-09-20_13-28-21_g4` | 2026-09-20 | 31 (0–3000) | 246.9 MiB | — | HangStand-G4-v0(추정) | 0 | — | — | 판정 기록 없음 — 문서에 '진행 중'으로만 남아 있다(6000 중 3000 반복에서 멈춤, 평가 산출물 0) | `docs/PLANT_ID_SOLUTION_0920.md`:44 |
| `biped12_stepb0/2026-10-04_21-12-52_g24rglr_seed1_from_g24ra21793` | 2026-10-04 | 2 (21800–21900) | 16.2 MiB | — | 미상 | 0 | — | — | 판정 기록 없음 — 가설 문서만 있고 결과·중단 기록이 없다(체크포인트 2 개에서 멈춤) | `reports/gen_c5/g24rglr/hypothesis.md`:1 |

## 5. 폐기 후보 — 근거와 용량 순위

### 5.1 용량 순 상위 20 (문서 근거 있는 것)

| # | run | 용량 | 체크포인트 | 판정 | 근거 문서:줄 |
|---:|---|---:|---|---|---|
| 1 | `biped12_stage4/2026-09-20_13-39-35_g5` | 488.6 MiB | 61 (0–5999) | 명목 0/256 '진짜 실패' | `reports/gen_g6/SUMMARY.md`:25 |
| 2 | `biped12_stage4/2026-09-21_09-08-58_g8` | 330.0 MiB | 41 (4500–8499) | G8 체크포인트 선택 — 적격 후보 없음(익스포트 사본은 deploy/cand/g8_8499 에 따로 있음) | `reports/gen_g6/RECOVERY_EVAL_0920.md`:279 |
| 3 | `biped12_stage4/2026-09-20_21-20-53_g7` | 327.2 MiB | 41 (4500–8499) | G7 은 승격 대상이 아니다 | `reports/gen_g6/RECOVERY_EVAL_0920.md`:195 |
| 4 | `biped12_stage4/2026-09-28_23-19-53_g11b` | 247.6 MiB | 31 (5700–8699) | G11 판정 — 적격 후보 없음, G11b 는 G11a 대비 일관된 이득 없음 | `reports/gen_g11/VERDICT.md`:18 · `reports/gen_g11/VERDICT.md`:31 |
| 5 | `biped12_stage4/2026-09-30_12-05-18_g13` | 215.0 MiB | 21 (9500–11499) | G13 판정 — 통과 후보 없음 | `reports/gen_g13/G13_VERDICT.md`:1 |
| 6 | `biped12_stage4/2026-09-30_15-26-19_g14b` | 214.9 MiB | 21 (9500–11499) | G14 판정 — 가설 미지지, 통과 후보 없음 | `reports/gen_g13/G14_VERDICT.md`:1 |
| 7 | `biped12_w1/2026-10-03_21-20-36_w1b_cS0toS3_seed1_from_g21a11300` | 203.3 MiB | 25 (11300–13699) | W1b — 스텝 0 | `reports/gen_step/w2a/hypothesis.md`:2 |
| 8 | `biped12_w1/2026-10-03_19-38-00_w1_cS0toS3_seed1_from_g21a11300` | 203.3 MiB | 25 (11300–13699) | W1 — 착지 0(걷지 않음) | `reports/gen_g22r/VERDICT_1003.md`:23 |
| 9 | `biped12_stepb0/2026-10-04_15-13-03_w2j_seed1_from_w2h24791` | 169.8 MiB | 21 (24800–26790) | W2-10 — FAIL, 요(회전) 갈래 일시 정지 | `reports/gen_step/w2j/decision.md`:1 |
| 10 | `biped12_stage4/2026-09-30_17-49-39_g15w` | 167.9 MiB | 21 (9500–11499) | G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| 11 | `biped12_stage4/2026-10-01_04-46-25_g16d` | 167.9 MiB | 21 (9500–11499) | G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| 12 | `biped12_stage4/2026-10-01_02-54-55_g16a` | 167.9 MiB | 21 (9500–11499) | G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| 13 | `biped12_stage4/2026-10-01_02-53-25_g16o43` | 167.9 MiB | 21 (9500–11499) | G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음 | `reports/gen_g16/G16_VERDICT.md`:13 |
| 14 | `biped12_stage4/2026-09-30_22-37-30_g15k` | 167.8 MiB | 21 (9500–11499) | G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| 15 | `biped12_stage4/2026-09-30_21-06-44_g15d` | 167.8 MiB | 21 (9500–11499) | G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| 16 | `biped12_stage4/2026-09-30_19-18-11_g15q` | 167.8 MiB | 21 (9500–11499) | G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| 17 | `biped12_stage4/2026-09-29_11-08-21_g11c` | 167.7 MiB | 21 (8700–10698) | G11c 판정 — 승격 없음(정착↔뒤 회복 교환) | `reports/gen_g11/VERDICT.md`:48 |
| 18 | `biped12_stage4/2026-09-30_19-22-14_g15u` | 167.7 MiB | 21 (9500–11499) | G15 판정 — 통과 후보 없음 | `reports/gen_g15/G15_VERDICT.md`:1 |
| 19 | `biped12_stage4/2026-09-30_15-26-19_g14a` | 167.7 MiB | 21 (9500–11499) | G14 판정 — 가설 미지지, 통과 후보 없음 | `reports/gen_g13/G14_VERDICT.md`:1 |
| 20 | `biped12_g22r/2026-10-03_19-01-59_g22r_cR1toR4_seed1_from_g21a11300` | 153.8 MiB | 19 (11300–13099) | G22R·W1 판정 — 통과 0 | `reports/gen_g22r/VERDICT_1003.md`:1 |

### 5.2 문서 기록 없이 이름·설정만으로 무효로 본 것

지시의 정의대로라면 불명이다. run 이름(smoke·chk·dbg·`_void_*` 폴더)과 반복 수, 평가 산출물이 없다는 사실만 근거다.

| run | 용량 | 체크포인트 | max_iterations | 평가 산출물 | 메모 |
|---|---:|---|---:|---:|---|
| `_void_1003_bench/biped12_g22r/2026-10-03_18-40-32_g22r_cR1toR4_seed1_from_g21a11300` | 64.9 MiB | 8 (11300–12000) | 1800 | 0 | 무효 — G22R 첫 기동분(같은 이름으로 19:01 에 다시 시작) — _void_1003_bench 로 옮겨져 있음 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-57-51_smoke2_W1` | 16.2 MiB | 2 (11300–11359) | 60 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-56-56_smoke2_G22R` | 16.2 MiB | 2 (11300–11359) | 60 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-03_23-47-55_smoke_sb0` | 16.1 MiB | 2 (11300–11329) | 30 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 30, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_15-13-17_smoke_g21d_void` | 16.0 MiB | 2 (10700–10739) | 40 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 40, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_w1/2026-10-03_18-33-02_smoke_W1` | 16.0 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_g22r/2026-10-03_18-32-48_smoke_G22R` | 16.0 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_03-31-23_smoke_si0` | 16.0 MiB | 2 (11300–11304) | 5 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 5, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_g22r/2026-10-03_21-26-01_smoke_g22r2` | 16.0 MiB | 2 (11300–11301) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_14-08-12_smoke_g20s_void` | 15.9 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_14-07-12_smoke_g20d_void` | 15.9 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_12-25-54_smoke_g19v_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_12-26-54_smoke_g19n_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_09-05-43_smoke_g18f_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_06-48-05_smoke_g18q_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_06-47-23_smoke_g18j_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_06-47-07_smoke_g18o_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-10-02_06-47-41_smoke_g18d_void` | 15.8 MiB | 2 (9500–9501) | 2 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 2, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-21_02-48-57_g8chk` | 15.7 MiB | 2 (4500–4559) | 60 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 60, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-21_02-53-57_g8chk2` | 15.6 MiB | 2 (4500–4539) | 40 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 40, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-22_13-32-00_g10chk` | 15.6 MiB | 2 (5700–5714) | 15 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 15, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-21_02-52-40_g8dbg3` | 15.5 MiB | 2 (4500–4514) | 15 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 15, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-21_02-51-51_g8dbg2` | 15.5 MiB | 2 (4500–4511) | 12 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-20_21-18-04_g7smoke2` | 15.5 MiB | 2 (4500–4511) | 12 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-20_21-16-37_g7smoke` | 15.5 MiB | 2 (4500–4511) | 12 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 12, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `biped12_stage4/2026-09-21_02-50-47_g8dbg` | 15.5 MiB | 2 (4500–4502) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_08-03-19_smoke_g23rb` | 15.5 MiB | 2 (11300–11319) | 20 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 20, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_05-28-25_smoke_a0k` | 15.4 MiB | 2 (11300–11307) | 8 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 8, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_06-23-06_smoke_W2a` | 15.4 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_06-23-23_smoke_G23Ra` | 15.4 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_12-01-06_smoke_g24rc` | 15.4 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |
| `_void_1003_bench/biped12_stepb0/2026-10-04_08-38-27_smoke_w2d` | 15.4 MiB | 2 (11300–11302) | 3 | 0 | 무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations 3, 체크포인트 2 개, 평가 산출물 0 건). 문서 기록은 없다 |

### 5.3 근거 문서별 묶음

| 근거 문서 | run 수 | 용량 | run(짧은 이름) |
|---|---:|---:|---|
| `reports/night_1005/LOG.md` | 32 | 2.03 GiB | w2om, woms, pa1, rh2, rd3, rd5, w3s1d2, w3s1b, wsteer, cstab, w2qc, w3ci, womx, wsteer2, wsteer2, w2omf, w2omps2, w2omr2, w2omr1, w2omp, w2omp2, w3s1e, w3s1c2x, wom, w2omi, w2omp3, w3u, w3c, w2oml, w2omm, w2oma, w2gs |
| `reports/gen_g6/RECOVERY_EVAL_0920.md` | 6 | 944.2 MiB | g8, g7, g8d, g8c, g8b, g8a |
| `reports/gen_g15/G15_VERDICT.md` | 5 | 839.1 MiB | g15w, g15k, g15d, g15q, g15u |
| `reports/gen_g18/G18_VERDICT.md` | 6 | 768.3 MiB | g18j, g18f43, g18q, g18f, g18d, g18o |
| `reports/gen_g17/G17_VERDICT.md` | 6 | 767.9 MiB | g17r43, g17rp, g17o43, g17p, g17r, g17o |
| `reports/gen_g18/G21_VERDICT.md` | 9 | 723.4 MiB | g21d, g21c2, g21c1, g21b, g21a3s43, g21a3, g21a2, g21a1, g21o |
| `reports/gen_g16/G16_VERDICT.md` | 4 | 607.7 MiB | g16d, g16a, g16o43, g16a43 |
| `reports/gen_g6/SUMMARY.md` | 1 | 488.6 MiB | g5 |
| `reports/gen_g11/VERDICT.md` | 2 | 415.4 MiB | g11b, g11c |
| `reports/gen_g13/G14_VERDICT.md` | 2 | 382.6 MiB | g14b, g14a |
| `reports/gen_g22r/VERDICT_1003.md` | 2 | 357.1 MiB | w1, g22r |
| `reports/gen_g18/G20_DESIGN_CRITERIA.md` | 2 | 256.3 MiB | g20s, g20d |
| `reports/gen_g18/G22_DESIGN_CRITERIA.md` | 4 | 249.0 MiB | g22a, g22ws43, g22w, g22cvoidstopped |
| `reports/gen_g13/G13_VERDICT.md` | 2 | 218.3 MiB | g13, g13init |
| `reports/gen_step/w2a/hypothesis.md` | 1 | 203.3 MiB | w1b |
| `reports/gen_step/w2j/decision.md` | 1 | 169.8 MiB | w2j |
| `reports/gen_c5/tourney1/RESULT.md` | 3 | 145.3 MiB | t2b, t2, t1b |
| `reports/gen_step/g24rc/decision.md` | 1 | 130.1 MiB | g24rc |
| `reports/gen_c5/g24rh/decision.md` | 1 | 129.4 MiB | g24rh |
| `reports/gen_c5/g24re/decision.md` | 1 | 129.4 MiB | g24re |
| `reports/gen_c5/g24rd/decision.md` | 1 | 129.3 MiB | g24rd |
| `reports/gen_step/g24rb/decision.md` | 1 | 129.3 MiB | g24rb |
| `reports/gen_g22r/STEPB0_CRITERIA.md` | 1 | 129.3 MiB | stepb0 |
| `reports/gen_step/g23ra/decision.md` | 1 | 129.0 MiB | g23ra |
| `reports/gen_walk/W2o_RESULT.md` | 1 | 128.5 MiB | w2o |
| `reports/gen_g18/G19_DESIGN_CRITERIA.md` | 1 | 128.1 MiB | g19n |
| `reports/gen_walk/wv1/RESULT.md` | 3 | 120.7 MiB | wv3, wv2, wv1 |
| `reports/gen_g22r/DESIGN_CRITERIA.md` | 7 | 113.0 MiB | benchduow, benchsolow, benchduog, benchsolog, benchduo2kw, benchduo2kg, benchsolo4k |
| `reports/gen_c5/g24rg/decision.md` | 1 | 105.3 MiB | g24rg |
| `RL보행_명령어모음.txt` | 2 | 103.9 MiB | 2026-07-25_23-00-29, 2026-07-25_22-36-10 |
| `reports/gen_w3/WD_DESIGN.md` | 2 | 98.2 MiB | w3s4ddb, w3s4dd |
| `reports/gen_step/stepa0c/decision.md` | 2 | 98.0 MiB | stepa0b, stepa0c |
| `reports/gen_c5/rcap/RESULT.md` | 2 | 97.0 MiB | r2p, r1p |
| `reports/gen_g17/G17_CRITERIA.md` | 2 | 88.3 MiB | g17ovoid, g17rvoid |
| `reports/gen_g18/G21_DESIGN_CRITERIA.md` | 1 | 80.1 MiB | g21avoidexplosion |
| `reports/gen_step/stepa0/decision.md` | 1 | 57.1 MiB | stepa0 |
| `reports/gen_step/stepb1/decision.md` | 1 | 56.8 MiB | stepb1 |
| `reports/gen_step/stepa0k/decision.md` | 1 | 56.6 MiB | stepa0k |
| `reports/gen_step/stepa0ic/decision.md` | 1 | 56.4 MiB | stepa0ic |
| `reports/gen_w3/ROBUST_DESIGN.md` | 1 | 49.1 MiB | w3s3dr |
| `reports/gen_w3/C2_DESIGN.md` | 1 | 48.8 MiB | w3s3dc2 |
| `reports/gen_c5/rcap/NOTE_1005.md` | 1 | 48.6 MiB | r2 |
| `reports/gen_c5/tourney2/RESULT.md` | 1 | 48.4 MiB | t3 |
| `reports/gen_walk/W2q2_RESULT_WR_PLAN.md` | 1 | 48.4 MiB | w2q2 |
| `reports/gen_step/stepa0g/decision.md` | 1 | 41.0 MiB | stepa0g |
| `reports/gen_step/stepa0e/decision.md` | 1 | 41.0 MiB | stepa0e |
| `reports/gen_step/stepi0/decision.md` | 1 | 40.9 MiB | stepi0 |
| `reports/gen_step/stepa0h/decision.md` | 1 | 40.3 MiB | stepa0h |
| `reports/gen_step/g23rb/decision.md` | 1 | 40.3 MiB | g23rb |
| `reports/gen_walk/wt1/RESULT.md` | 1 | 40.3 MiB | wt1 |
| `reports/gen_step/w2i/decision.md` | 1 | 40.3 MiB | w2i |
| `reports/gen_walk/WALK_FINAL_1005.md` | 1 | 32.2 MiB | wr |
| `reports/gen_c5/g24rcont/hypothesis.md` | 1 | 16.2 MiB | g24rf |
| `reports/gen_step/stepa0f/decision.md` | 1 | 8.5 MiB | stepa0f |
| `reports/FINAL_REPORT_1005.md` | 1 | 8.3 MiB | w2w |
| `reports/gen_step/stepa0i/decision.md` | 1 | 7.9 MiB | stepa0j |

- `logs/rsl_rl/_void_1003_bench/` 는 이미 누군가(1003~1004) 무효 실행을 따로 모아 둔 폴더다. 그 안의 run 은 실험 폴더 밖에 있어서 `--load_run` 으로 잡히지 않는다 — 사실상 이미 격리된 상태다.

## 6. 불명

| run | 용량 | 체크포인트 | exported | 왜 불명인가 | 참고 |
|---|---:|---|---|---|---|
| `biped12_flat/2026-07-19_04-18-49` | 9.0 MiB | 2 (0–99) | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | — |
| `biped12_flat/2026-07-19_04-50-06` | 41.2 MiB | 9 (3000–3800) | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | 부모 2026-07-19_04-21-02/model_2999 |
| `biped12_flat/2026-07-19_05-28-26` | 4.6 MiB | 1 (6000–6000) | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | 부모 2026-07-19_04-59-54/model_5998 |
| `biped12_flat/2026-07-19_06-36-02` | 114.9 MiB | 25 (11500–13900) | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | 부모 2026-07-19_06-04-22/model_11496 |
| `biped12_flat/2026-07-25_21-46-08` | 47.4 MiB | 10 (12000–12900) | exported | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | 부모 2026-07-25_21-23-38/model_12000 |
| `biped12_flat/2026-07-25_22-34-20` | 4.4 MiB | 1 (12300–12300) | — | 근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)  | 부모 2026-07-25_21-59-19/model_12300 |
| `biped12_stage4/2026-09-20_13-28-21_g4` | 246.9 MiB | 31 (0–3000) | — | 판정 기록 없음 — 문서에 '진행 중'으로만 남아 있다(6000 중 3000 반복에서 멈춤, 평가 산출물 0) `docs/PLANT_ID_SOLUTION_0920.md`:44 | — |
| `biped12_stepb0/2026-10-04_21-12-52_g24rglr_seed1_from_g24ra21793` | 16.2 MiB | 2 (21800–21900) | — | 판정 기록 없음 — 가설 문서만 있고 결과·중단 기록이 없다(체크포인트 2 개에서 멈춤) `reports/gen_c5/g24rglr/hypothesis.md`:1 | 부모 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/model_21793 |

## 7. 참조 관계

### 7.1 브리지 레지스트리와 콘솔 선택지

- 콘솔 선택지(`operator_console.py` `CKPT_CHOICES`): `stand_crouch_g12`, `stand_crouch_g21a`, `stand_c5_g21a_g24ra`, `stand_c5kr_g21a_g24ra`, `stand_c5krh_g21a_g24ra`, `stand_crouch_g12_ema0`, `stand_crouch_g11a` — 기본값 `stand_crouch_g12`.
- `STAND_DEFAULT` = `stand_v18b` · env.yaml 필수 키(`CKPT_REQUIRE_ENV_YAML`): `stand_c5_g21a_g24ra`, `stand_c5kr_g21a_g24ra`, `stand_c5krh_g21a_g24ra`, `stand_crouch_g21a` · `CKPT_JVEL_EMA_MODE`: {'stand_crouch_g12_ema0': 'ema0'}.

| 표 | 키 | 가리키는 run | exported | 콘솔 | 로컬 |
|---|---|---|---|---|---|
| CHECKPOINTS | `stand_v18b` | `biped12_stage4/2026-09-05_19-37-39` | 있음 |  | 있음 |
| CHECKPOINTS | `stand_crouch_v5` | `biped12_stage4/2026-09-21_20-30-32_g9` | 있음 |  | 있음 |
| CHECKPOINTS | `stand_crouch_v6` | `biped12_stage4/2026-09-22_13-32-53_g10` | 있음 |  | 있음 |
| CHECKPOINTS | `stand_crouch_g11a` | `biped12_stage4/2026-09-28_23-19-13_g11a` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_crouch_g12` | `biped12_stage4/2026-09-29_14-32-50_g12` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_crouch_g12_ema0` | `biped12_stage4/2026-09-29_14-32-50_g12` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_crouch_g21a` | `biped12_stage4/2026-10-02_21-23-08_g21a` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_c5_g21a_g24ra` | `biped12_stage4/2026-10-02_21-23-08_g21a` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_c5kr_g21a_g24ra` | `biped12_stage4/2026-10-02_21-23-08_g21a` | 있음 | 예 | 있음 |
| CHECKPOINTS | `stand_c5krh_g21a_g24ra` | `biped12_stage4/2026-10-02_21-23-08_g21a` | 있음 | 예 | 있음 |
| C5_PAIRS(회복) | `stand_c5_g21a_g24ra` | `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | 있음 | 예 | 있음 |
| C5_PAIRS(회복) | `stand_c5kr_g21a_g24ra` | `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | 있음 | 예 | 있음 |
| C5_PAIRS(회복) | `stand_c5krh_g21a_g24ra` | `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | 있음 | 예 | 있음 |
| W3_SPECS · model_11699.pt | `stand_w3a_11699` | `biped12_stepb0/2026-10-06_11-40-53_w3A_s1_from_g21a` | **없음** |  | 있음 |
| W3_SPECS · model_15800.pt | `stand_w3s3d_15800` | `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | 있음 |  | 있음 |
| HTURN2 · model_25590.pt | `walk_w2q25590_hturn2` | `biped12_stepb0/2026-10-05_18-18-43_w2q_seed1_from_w2h24791` | **없음** |  | 있음 |
| KEW3_SPECS(합성) | `stand_kew3_g21a_g24ra_w3s3d15800` | 걷기 = `stand_w3s3d_15800` · 서기+회복 = `stand_c5kr_g21a_g24ra` | — | | |

### 7.2 이미 끊어져 있는 참조(로컬에 run 이 없는 것)

- `CHECKPOINTS` 의 37 키 중 **27 키**는 run 디렉터리가 로컬 `logs/rsl_rl/` 에 없다: `walk`, `march`, `drv2`, `march_drv2`, `march_v3`, `march_v5b`, `stand_v3`, `stand_v5`, `stand_v6`, `stand_v7`, `stand_v8`, `stand_v9`, `stand_v14`, `stand_v15`, `stand_v16`, `stand_v17`, `stand_v18`, `stand_hang_v1`, `stand_hang_v2`, `stand_hang_v3`, `stand_hang_v4`, `stand_hang_v5`, `stand_hang_v5p`, `stand_hang_v6`, `stand_crouch_v1`, `stand_crouch_v1b`, `stand_crouch_v1c`.
  이 키들은 브리지 selftest 에서 '미배포'로 SKIP 된다(파일이 없는 등록 항목은 FAIL 이 아니라 SKIP). 정책 사본은 `models/<이름>/` 에 일부 남아 있다(§8).
- 코드·스크립트가 이름으로 가리키는데 로컬에 없는 run: 31 개(문서만 가리키는 것까지 합치면 65 개). 코드 쪽:
  - `2026-07-26_04-07-10` ← `scripts/rl_walk_demo_stage4.py:151` · 평가 산출물 3 건
  - `2026-07-26_13-09-16` ← `sim_walking/live_controller.py:210` · 평가 산출물 7 건
  - `2026-07-26_13-35-55` ← `sim_walking/live_controller.py:212` · 평가 산출물 4 건
  - `2026-07-26_14-53-03` ← `rl_walking/policy_adapter_stage4.py:39`, `rl_walking/policy_adapter_stage4.py:41`, `rl_walking/deploy/policy_runner_stage4.py:3`, `rl_walking/deploy/policy_runner_stage4.py:149` 외 4 · 평가 산출물 9 건
  - `2026-07-26_21-41-30` ← `rl_walking/scripts/record_ppt_media.py:48`, `jetson/scripts/rl_bridge_node.py:126`, `sim_walking/live_controller.py:211` · 평가 산출물 4 건
  - `2026-08-02_04-31-34` ← `rl_walking/scripts/probe_stand_entry.py:14`, `jetson/scripts/rl_bridge_node.py:131` · 평가 산출물 2 건
  - `2026-08-02_12-22-59` ← `jetson/scripts/rl_bridge_node.py:135` · 평가 산출물 2 건
  - `2026-08-02_19-30-03` ← `jetson/scripts/rl_bridge_node.py:141` · 평가 산출물 3 건
  - `2026-08-03_03-18-21` ← `jetson/scripts/rl_bridge_node.py:148` · 평가 산출물 4 건
  - `2026-08-03_06-12-07` ← `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:27`, `rl_walking/scripts/night_0906/pkg_verify.sh:13`, `jetson/scripts/rl_bridge_node.py:153` · 평가 산출물 9 건
  - `2026-08-03_18-44-48` ← `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:27`, `rl_walking/scripts/night_0906/pkg_verify.sh:13`, `jetson/scripts/rl_bridge_node.py:158` · 평가 산출물 4 건
  - `2026-08-04_11-31-27` ← `rl_walking/scripts/eval_ladder.py:25`, `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:27`, `rl_walking/scripts/night_0906/pkg_verify.sh:13`, `jetson/scripts/rl_bridge_node.py:163` · 평가 산출물 168 건
  - `2026-08-04_13-13-49` ← `rl_walking/scripts/eval_drift_dr.py:12`, `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:28`, `rl_walking/scripts/night_0906/pkg_verify.sh:14`, `jetson/scripts/rl_bridge_node.py:168` · 평가 산출물 15 건
  - `2026-08-08_21-24-48` ← `rl_walking/scripts/eval_calib_sensitivity.py:19`, `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:28`, `rl_walking/scripts/night_0906/pkg_verify.sh:14`, `jetson/scripts/rl_bridge_node.py:175` · 평가 산출물 15 건
  - `2026-08-09_03-59-22` ← `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:28`, `rl_walking/scripts/night_0906/pkg_verify.sh:14`, `jetson/scripts/rl_bridge_node.py:182` · 평가 산출물 19 건
  - `2026-09-04_02-06-07` ← `rl_walking/scripts/night_0906/repro_jetson_selftest.sh:33`, `rl_walking/scripts/night_0906/pkg_verify.sh:21`, `jetson/scripts/rl_bridge_node.py:187` · 평가 산출물 15 건
  - `2026-09-04_16-43-02` ← `jetson/scripts/rl_bridge_node.py:192` · 평가 산출물 48 건
  - `2026-09-04_20-05-14` ← `jetson/scripts/rl_bridge_node.py:196` · 평가 산출물 31 건
  - `2026-09-04_21-48-32` ← `rl_walking/scripts/register_model.py:5`, `rl_walking/scripts/night_0906/h3_chain.sh:8`, `jetson/scripts/rl_bridge_node.py:200` · 평가 산출물 58 건
  - `2026-09-05_17-57-21` ← `jetson/scripts/rl_bridge_node.py:203` · 평가 산출물 21 건
  - `2026-09-08_19-00-25` ← `jetson/scripts/rl_bridge_node.py:210` · 평가 산출물 16 건
  - `2026-09-08_23-33-51` ← `jetson/scripts/rl_bridge_node.py:214` · 평가 산출물 4 건
  - `2026-09-09_11-34-22` ← `jetson/scripts/rl_bridge_node.py:218` · 평가 산출물 5 건
  - `2026-09-09_12-33-03` ← `jetson/scripts/rl_bridge_node.py:222` · 평가 산출물 35 건
  - `2026-09-09_19-15-42` ← `rl_walking/scripts/run_v5_eval_battery.sh:15`, `jetson/deploy_to_jetson.sh:35`, `jetson/scripts/rl_bridge_node.py:227` · 평가 산출물 469 건
  - `2026-09-09_20-50-01` ← `jetson/scripts/rl_bridge_node.py:231` · 평가 산출물 14 건
  - `2026-09-13_00-40-49` ← `rl_walking/hang_stand_env_cfg.py:674`, `jetson/scripts/rl_bridge_node.py:236` · 평가 산출물 133 건
  - `2026-09-13_12-33-22_v6d` ← `rl_walking/hang_stand_env_cfg.py:687` · 평가 산출물 50 건
  - `2026-09-18_00-14-13_s9c` ← `jetson/scripts/rl_bridge_node.py:242` · 평가 산출물 3 건
  - `2026-09-18_01-08-37_s9c2` ← `jetson/scripts/rl_bridge_node.py:245` · 평가 산출물 4 건
  - `2026-09-18_02-52-27_s9c3` ← `jetson/scripts/rl_bridge_node.py:248` · 평가 산출물 11 건
- run 은 있는데 **참조된 체크포인트 파일이 없는 것**:
  - `biped12_stage4/2026-09-05_19-37-39`: model_120998.pt ← models/stand_v18b/meta.json

### 7.3 심볼릭 링크(별칭)

- run 디렉터리 링크 63 개: 대부분 `biped12_stage4/<run>` → `../biped12_stepb0/<run>` (G11a 지면 평가 태스크가 `biped12_stage4` 에서 run 을 찾기 때문 — `claude2.md` §8). 부모 run 링크(`biped12_{w1,g22r,stepb0}/…_g21a` → `../biped12_stage4/…_g21a`)도 있다.
- 실험 폴더 링크: `logs/rsl_rl/biped12_stage4/biped12_stepb0` → `../biped12_stepb0/` — **자기 자신의 형제를 가리키는 고리**다(`biped12_stage4/biped12_stepb0/…` 로 같은 run 이 한 번 더 보인다). 재귀로 훑는 도구(`find -L`, `du -L`, glob `**`)는 중복 집계한다.
- run 안의 파일 링크 30 개: `model_11600.pt → model_11599.pt`, `model_11000.pt → model_10999.pt` 꼴(마지막 저장본을 평가 격자의 반복 번호로 부르기 위한 것). 링크 대상을 치우면 그 번호를 쓰는 평가가 깨진다. 핵심 체크포인트 계산에서는 링크를 대상 파일로 바꿔 셌다.
- 깨진 링크: 0 개

### 7.4 `exported*/` 가 어느 체크포인트에서 나왔는가(가중치 대조)

`SOURCE_CHECKPOINT` 파일은 1003 이후 run 에만 있다. 없는 run 은 `policy.pt` 의 가중치를 그 run 의 체크포인트들과 맞대어 찾았다. '등록부'는 `models/*/meta.json` 이 적어 둔 체크포인트다.

| run | 디렉터리 | SOURCE_CHECKPOINT | 가중치가 같은 체크포인트 | 등록부가 말하는 체크포인트 | onnx 에 같은 가중치 | 판정 |
|---|---|---|---|---|---|---|
| `biped12_flat/2026-07-19_04-21-02` | exported | — | model_2999.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-19_04-59-54` | exported | — | model_5998.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-19_05-32-42` | exported | — | model_8997.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-19_06-04-22` | exported | — | model_11496.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-19_07-15-42` | exported | — | model_13995.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_21-07-04` | exported | — | model_9999.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_21-23-38` | exported | — | model_12000.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_21-46-08` | exported | — | model_12900.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_21-59-19` | exported | — | model_12300.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_22-04-32` | exported | — | model_13499.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_22-36-10` | exported | — | model_13299.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_23-00-29` | exported | — | model_13299.pt | — | 예 | 일치 |
| `biped12_flat/2026-07-25_23-44-35` | exported | — | model_14994.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-05_19-37-39` | exported | — | — | stand_v18b: model_120998.pt | 예 | 일치하는 체크포인트 없음(체크포인트 원본이 로컬에 없다) |
| `biped12_stage4/2026-09-20_03-30-06_g2` | exported | — | model_5500.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-20_03-38-43_g3` | exported | — | model_5400.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-20_13-39-35_g5` | exported | — | model_5999.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-20_15-06-59_g6` | exported | — | model_4500.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-21_09-08-58_g8` | exported | — | model_8499.pt | — | 예 | 일치 |
| `biped12_stage4/2026-09-21_20-30-32_g9` | exported | — | model_5700.pt | stand_crouch_v5: model_5700.pt | 예 | 일치 |
| `biped12_stage4/2026-09-22_13-32-53_g10` | exported | — | model_7500.pt | stand_crouch_v6: model_6600.pt | 예 | **⚠ 등록부와 다름** |
| `biped12_stage4/2026-09-28_23-19-13_g11a` | exported | — | model_8699.pt | stand_crouch_g11a: model_8699.pt | 예 | 일치 |
| `biped12_stage4/2026-09-29_14-32-50_g12` | exported | — | model_9500.pt | stand_crouch_g12: model_9500.pt | 예 | 일치 |
| `biped12_stage4/2026-10-02_21-23-08_g21a` | exported | model_11300.pt | model_11300.pt | stand_g21a_11300: model_11300.pt | 예 | 일치 |
| `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | exported | model_21793.pt | model_21793.pt | recover_g24ra_21793: model_21793.pt | 예 | 일치 |
| `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | exported | model_15800.pt | model_15800.pt | walk_w3s3d_15800: model_15800.pt | 예 | 일치 |
| `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | exported_16100 | model_16100.pt | model_16100.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | exported_16193 | model_16193.pt | model_16193.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_21-51-10_w3S3e_from_s3d` | exported_16100 | model_16100.pt | model_16100.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_21-51-10_w3S3e_from_s3d` | exported_16299 | model_16299.pt | model_16299.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_22-25-32_w3S3f_from_15800` | exported_16100 | model_16100.pt | model_16100.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_22-25-32_w3S3f_from_15800` | exported_16299 | model_16299.pt | model_16299.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_22-52-26_w3S3g_from_15800` | exported_16100 | model_16100.pt | model_16100.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_22-52-26_w3S3g_from_15800` | exported_16299 | model_16299.pt | model_16299.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_23-17-51_w3S3t_from_15800` | exported_16299 | model_16299.pt | model_16299.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_23-42-16_w3S3tK_from_15800` | exported_16100 | model_16100.pt | model_16100.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-07_23-42-16_w3S3tK_from_15800` | exported_16299 | model_16299.pt | model_16299.pt | — | 예 | 일치 |
| `biped12_stepb0/2026-10-08_00-11-19_g24raf_from_21793` | exported_22292 | model_22292.pt | model_22292.pt | — | 예 | 일치 |

- ⚠ `logs/rsl_rl/biped12_stage4/2026-09-22_13-32-53_g10/exported` 는 **model_7500.pt 의 익스포트**다. 등록부(stand_crouch_v6)는 model_6600.pt 의 익스포트를 담고 있고(가중치로 확인, §8.1), `deploy/MANIFEST.md5` 에 적힌 md5 도 지금 이 디렉터리의 파일과 다르다. 브리지 키 `stand_crouch_v6` 는 이 디렉터리를 읽으므로, **PC 에서 그 키로 도는 정책은 등록부·매니페스트가 말하는 정책과 다르다.**
  - 젯슨 스냅숏(2026-10-08 00:25)의 같은 경로: `policy.onnx` PC 와 다름(= 등록부 사본과 같은 파일), `policy.pt` PC 와 다름(= 등록부 사본과 같은 파일) — 스냅숏 기준으로 젯슨에는 등록부와 같은 파일이 있고, PC 의 `exported/` 만 다르다.

## 8. `logs/rsl_rl/` 밖에 있는 정책 사본

### 8.1 `models/` (등록부 — `rl_walking/scripts/register_model.py` 가 만든다, git 추적)

| 이름 | run / 체크포인트 | 태스크 | 파일 md5 = meta.json | run 이 로컬에 | 사본 가중치 = 그 체크포인트 | 지금 run 의 exported 와 같은 파일 |
|---|---|---|---|---|---|---|
| `models/recover_g24ra_21793` | 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 / model_21793.pt | HangStand-G24Ra-v0 | 일치 | 있음 | 예 | 예 |
| `models/stand_crouch_g11a` | 2026-09-28_23-19-13_g11a / model_8699.pt | HangStand-G11a-v0 | 일치 | 있음 | 예 | 예 |
| `models/stand_crouch_g12` | 2026-09-29_14-32-50_g12 / model_9500.pt | HangStand-G12-v0 | 일치 | 있음 | 예 | 예 |
| `models/stand_crouch_v1` | 2026-09-18_00-14-13_s9c / model_144485.pt | HangStand-S9c-v0 | 일치 | 없음 | — | — |
| `models/stand_crouch_v1b` | 2026-09-18_01-08-37_s9c2 / model_150484.pt | HangStand-S9c-v0 | 일치 | 없음 | — | — |
| `models/stand_crouch_v1c` | 2026-09-18_02-52-27_s9c3 / model_155500.pt | HangStand-S9c-v0 | 일치 | 없음 | — | — |
| `models/stand_crouch_v2` | 2026-09-18_19-06-31_s9d / model_156300.pt | HangStand-S9d-v0 | 일치 | 없음 | — | — |
| `models/stand_crouch_v5` | 2026-09-21_20-30-32_g9 / model_5700.pt | HangStand-G9-v0 | 일치 | 있음 | 예 | 예 |
| `models/stand_crouch_v6` | 2026-09-22_13-32-53_g10 / model_6600.pt | HangStand-G10-v0 | 일치 | 있음 | 예 | **아니오 ⚠**(§7.4) |
| `models/stand_g21a_11300` | 2026-10-02_21-23-08_g21a / model_11300.pt | HangStand-G21a-v0 | 일치 | 있음 | 예 | 예 |
| `models/stand_hang_v1` | 2026-09-08_19-00-25 / model_129494.pt | HangStand-S4m-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v2` | 2026-09-08_23-33-51 / model_133991.pt | HangStand-S5v-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v3` | 2026-09-09_11-34-22 / model_135490.pt | HangStand-S5w-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v4` | 2026-09-09_12-33-03 / model_136989.pt | HangStand-S6-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v5` | 2026-09-09_19-15-42 / model_139987.pt | HangStand-S8ac4-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v5p` | 2026-09-09_20-50-01 / model_139987.pt | HangStand-S8ac6-v0 | 일치 | 없음 | — | — |
| `models/stand_hang_v6` | 2026-09-13_00-40-49 / model_141486.pt | HangStand-S8ac4r-v0 | 일치 | 없음 | — | — |
| `models/legacy/stand_v13` | 2026-08-10_04-05-26 / model_95300.pt | Velocity-Stage4V2-Stand-V13-v0 | 일치 | 없음 | — | — |
| `models/stand_v14` | 2026-09-04_02-06-07 / model_107000.pt | Velocity-Stage4V2-Stand-V14-v0 | 일치 | 없음 | — | — |
| `models/stand_v14s` | 2026-09-04_05-04-39 / model_114999.pt | Velocity-Stage4V2-Stand-V14s-v0 | 일치 | 없음 | — | — |
| `models/stand_v15` | 2026-09-04_16-43-02 / model_114999.pt | Velocity-Stage4V2-Stand-V15-v0 | 일치 | 없음 | — | — |
| `models/stand_v16` | 2026-09-04_20-05-14 / model_120998.pt | Velocity-Stage4V2-Stand-V16-v0 | 일치 | 없음 | — | — |
| `models/stand_v17` | 2026-09-04_21-48-32 / model_126997.pt | Velocity-Stage4V2-Stand-V17-v0 | 일치 | 없음 | — | — |
| `models/stand_v18` | 2026-09-05_17-57-21 / model_132996.pt | Velocity-Stage4V2-Stand-V18-v0 | 일치 | 없음 | — | — |
| `models/stand_v18b` | 2026-09-05_19-37-39 / model_120998.pt | Velocity-Stage4V2-Stand-V18-v0 | 일치 | 있음 | 확인 불가(체크포인트 없음) | 예 |
| `models/legacy/stand_v6b` | 2026-08-04_11-31-27 / model_59899.pt | Velocity-Stage4V2-Stand-V6-v0 | 일치 | 없음 | — | — |
| `models/legacy/stand_v7` | 2026-08-04_13-13-49 / model_63898.pt | Velocity-Stage4V2-Stand-V7-v0 | 일치 | 없음 | — | — |
| `models/walk_w3s3d_15800` | 2026-10-07_00-11-12_w3S3d_from_s2d / model_15800.pt | HangStand-W3S3d-v0 | 일치 | 있음 | 예 | 예 |

- **배포 후보 가운데 `models/` 에 등록되지 않은 것: 없음.** 등록부의 마지막 항목은 `stand_crouch_g12`(0929)다. 그 뒤 세대는 git 이 무시하는 `logs/` 에만 있고, 체크섬은 `deploy/*.md5` 에 흩어져 있다.

### 8.2 `deploy/cand/` (익스포트 보관 사본)

| 폴더 | 파일 | md5 | 이름이 말하는 체크포인트와 가중치가 같은가 | 같은 내용의 exported |
|---|---|---|---|---|
| `deploy/cand/g10_6600` | policy.onnx | `43c4c3c76e0c5e613dadf7a27e30f21b` |  | 없음(그 run 의 `exported/` 는 지금 다른 체크포인트의 익스포트다) |
| `deploy/cand/g10_6600` | policy.pt | `238a62a02771aa22decaa365861e2a48` | 예 | 없음(그 run 의 `exported/` 는 지금 다른 체크포인트의 익스포트다) |
| `deploy/cand/g6_4500` | policy.onnx | `c54717740612ff323d942782a3fb91b4` |  | `biped12_stage4/2026-09-20_15-06-59_g6/exported/policy.onnx` |
| `deploy/cand/g6_4500` | policy.pt | `ef9d39f40b2ba812c48d2d8bbccb87c5` | 예 | `biped12_stage4/2026-09-20_15-06-59_g6/exported/policy.pt` |
| `deploy/cand/g8_8499` | policy.onnx | `f5064242528962b8ba96b23d711012dd` |  | `biped12_stage4/2026-09-21_09-08-58_g8/exported/policy.onnx` |
| `deploy/cand/g8_8499` | policy.pt | `1a912e096b5d0b29e460f36dd0b2653e` | 예 | `biped12_stage4/2026-09-21_09-08-58_g8/exported/policy.pt` |
| `deploy/cand/g9_5700` | policy.onnx | `aefc0477256bbcacf952bf877191f9df` |  | `biped12_stage4/2026-09-21_20-30-32_g9/exported/policy.onnx` |
| `deploy/cand/g9_5700` | policy.pt | `8ce14bf17c033850550cf7a5e0199809` | 예 | `biped12_stage4/2026-09-21_20-30-32_g9/exported/policy.pt` |
| `deploy/cand/g9_8499` | policy.onnx | `c7288dc8492a192427272c8f4b089303` |  | 없음(그 run 의 `exported/` 는 지금 다른 체크포인트의 익스포트다) |
| `deploy/cand/g9_8499` | policy.pt | `41879bbe9926a3f2a2894126d549cd91` | 예 | 없음(그 run 의 `exported/` 는 지금 다른 체크포인트의 익스포트다) |

### 8.3 워크트리 안의 체크포인트

- `.claude/worktrees/agent-ae9d2cc9f2348a2b9/logs` 는 링크가 아니라 **실제 폴더**다: 체크포인트·정책 파일 12 개(91.8 MiB). 본 트리와 내용이 **전부 같다**(md5) — 본 트리에 사본이 있다.

## 9. 명명 제안(초안) — 기존 브리지 키는 그대로 두고 별칭·매니페스트 수준에서

지금 이름의 문제(사실): ① 걷기 정책의 키가 `stand_w3s3d_15800` 로 `stand_` 로 시작한다(서기 가족 경로를 타게 하려는 접두사라 바꾸면 동작이 바뀐다). ② 회복 정책은 단독 키가 없다(`C5_PAIRS` 값 안의 경로로만 존재). ③ `stand_crouch_g21a` 는 체크포인트 번호가 이름에 없다. ④ 합성 키(`stand_c5_*`·`stand_c5kr_*`·`stand_c5krh_*`·`stand_kew3_*`)는 같은 정책 파일에 감지기·전환 설정만 다른 이름이다 — 정책 버전과 구성 버전이 한 이름에 섞여 있다.

제안 규칙:

- 정책 별칭 = `<역할>_<세대>_<체크포인트>` — 역할은 `stand`·`recover`·`walk`, 세대는 run 이름의 태그를 소문자로, 체크포인트는 `SOURCE_CHECKPOINT` 의 반복 번호.
- 버전 꼬리표 = `@<학습 시작일 YYMMDD>-<policy.onnx md5 앞 8 자>` — 젯슨에서 실제로 도는 파일(onnx) 기준. 같은 체크포인트를 다시 내보내 파일이 달라지면 꼬리표가 달라진다.
- 구성(합성) 이름 = `<구성>__<정책 별칭들>` — 구성은 `ke`(서기 + 회복 + 감지기), `kew3`(ke + 걷기). 감지기·전환 값은 구성 매니페스트에 적고 이름에는 넣지 않는다.
- 브리지 키·콘솔 선택지·블랙박스의 `checkpoint` 문자열은 **바꾸지 않는다**. 별칭은 매니페스트(`models/<별칭>/meta.json`, `deploy/<구성>.md5`)와 문서에서만 쓴다.

| 역할 | 제안 별칭 @ 버전 | 기존 브리지 키 | run | 체크포인트 | policy.onnx md5 | policy.pt md5 | model_N.pt md5 |
|---|---|---|---|---|---|---|---|
| 서기(STAND) | `stand_g21a_11300@261002-43f11310` | `stand_crouch_g21a` | `biped12_stage4/2026-10-02_21-23-08_g21a` | 11300 | `43f113109a369a60e976a391cf8f17cd` | `2464f5cdba2e819d03ffa2d8d03bbb69` | `599c47d3c3f190ad91a80a1ab7ff9fa7` |
| 회복(RECOVERY) | `recover_g24ra_21793@261004-57547697` | `(C5_PAIRS[stand_c5kr_g21a_g24ra] 의 회복 경로 — 단독 키 없음)` | `biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294` | 21793 | `57547697918f679543413a5b24c36555` | `896ca613bc402de02caa8302988f9a68` | `05e6b8ee4109ef40fd24bd8437bdfcba` |
| 걷기(WALK) | `walk_w3s3d_15800@261007-d908cd92` | `stand_w3s3d_15800` | `biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d` | 15800 | `d908cd92ba0916e790f9e3e28ff1d882` | `10e3ee405e1534ebe67574ddbeb9a2a2` | `3da9f2e5b6c435d6b6df220e3a7969b2` |

| 구성 | 제안 이름 | 대응하는 기존 키 | 달라지는 것 |
|---|---|---|---|
| 서기 + 회복 | `ke__stand_g21a_11300__recover_g24ra_21793` | `stand_c5_g21a_g24ra` | 감지기·조기 복귀 설정(`C5_DETECTOR_KW`·`C5_EXTRA`) — 정책 파일은 같다 |
| 서기 + 회복 | `ke__stand_g21a_11300__recover_g24ra_21793` | `stand_c5kr_g21a_g24ra` | 감지기·조기 복귀 설정(`C5_DETECTOR_KW`·`C5_EXTRA`) — 정책 파일은 같다 |
| 서기 + 회복 | `ke__stand_g21a_11300__recover_g24ra_21793` | `stand_c5krh_g21a_g24ra` | 감지기·조기 복귀 설정(`C5_DETECTOR_KW`·`C5_EXTRA`) — 정책 파일은 같다 |
| 서기 + 회복 + 걷기 | `kew3__stand_g21a_11300__recover_g24ra_21793__walk_w3s3d_15800` | `stand_kew3_g21a_g24ra_w3s3d15800` | `KEW3_SPECS` 의 전환 규칙(handoff·정착 시간) |

등록 방법 제안(실행하지 않았다 — `models/` 는 git 추적 폴더라 주 작업자 승인 뒤):

```bash
python3 rl_walking/scripts/register_model.py stand_g21a_11300 --experiment biped12_stage4 --run 2026-10-02_21-23-08_g21a --ckpt model_11300.pt \
    --task Biped12-HangStand-G21a-v0 --usd rl_walking/assets/usd_final3/biped12.usd --warm "g19v model_10700" --summary "서기(STAND) · 합성 구조 1007"
python3 rl_walking/scripts/register_model.py recover_g24ra_21793 --experiment biped12_stepb0 --run 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 --ckpt model_21793.pt \
    --task Biped12-HangStand-G24Ra-v0 --usd rl_walking/assets/usd_final3/biped12.usd --warm "w2e model_20294" --summary "회복(RECOVERY) · 합성 구조 1007"
python3 rl_walking/scripts/register_model.py walk_w3s3d_15800 --experiment biped12_stepb0 --run 2026-10-07_00-11-12_w3S3d_from_s2d --ckpt model_15800.pt \
    --task Biped12-HangStand-W3S3d-v0 --usd rl_walking/assets/usd_final3/biped12.usd --warm "w3s2d model_15694" --summary "걷기(WALK) · 합성 구조 1007"
```

- 이 도구는 `exported/policy.*` 를 그대로 복사한다. 복사 전에 `exported/SOURCE_CHECKPOINT` 가 `--ckpt` 와 같은지 봐야 한다(§1 점검이 그 확인이다 — 평가 체인 스크립트가 `exported/` 를 지우고 다른 체크포인트로 다시 만드는 경우가 있다: `reports/gen_w3/scripts/s3e_eval.sh`).
- 관찰: 브리지·러너 코드에는 로드한 정책 파일의 해시를 계산하는 부분이 없다(`hashlib`·`md5` 없음). 실행 기록에 남는 정책 식별자는 키 이름뿐이라, 파일이 바뀌어도 기록으로는 구분되지 않는다. 체크섬 대조는 지금 배포 스크립트 단계에서만 이뤄진다.

## 10. 폐기 후보 분리 계획(초안 — 실행하지 않았다)

### 10.1 옮기면 깨지는 것

폐기 후보 172 개(12.76 GiB)를 `logs/rsl_rl/<실험>/` 밖으로 옮긴다고 할 때:

1. **심볼릭 링크 27 개가 끊어진다**(전부 `biped12_stage4/<run>` → `../biped12_stepb0|g22r/<run>`). run 보다 링크를 먼저 지워야 한다:
   - `logs/rsl_rl/biped12_stage4/2026-10-03_19-01-59_g22r_cR1toR4_seed1_from_g21a11300`
   - `logs/rsl_rl/biped12_stage4/2026-10-03_23-49-02_stepb0_seed1_from_g21a11300`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_07-21-05_g23ra_seed1_from_a0m14299`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_08-27-46_g23rb_seed1_from_a0m14299`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_11-23-07_g24rb_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_12-35-53_g24rc_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_14-52-57_w2i_seed1_from_w2h24791`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_15-13-03_w2j_seed1_from_w2h24791`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_16-06-52_g24rd_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_17-20-12_g24re_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_19-24-01_g24rg_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_19-24-24_g24rh_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_22-53-50_t2_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_23-48-25_t1b_seed1_from_t122292`
   - `logs/rsl_rl/biped12_stage4/2026-10-04_23-48-56_t2b_seed1_from_t222292`
   - `logs/rsl_rl/biped12_stage4/2026-10-05_00-48-54_t3_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-05_03-54-56_r2_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-05_04-08-12_r2p_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-05_04-28-53_r1p_seed1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_02-42-32_rd5_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_04-09-13_rd3_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_05-07-08_rh2_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_06-04-22_pa1_from_g24ra21793`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_13-03-09_w3S1b_from_w3A2`
   - `logs/rsl_rl/biped12_stage4/2026-10-06_17-54-42_w3S1d2_from_w3A3`
   - `logs/rsl_rl/biped12_stage4/2026-10-07_02-43-05_w3S3dC2_from_s3d`
   - `logs/rsl_rl/biped12_stage4/2026-10-07_16-49-39_w3S3dR_from_s3d`
2. **스크립트가 이름으로 가리키는 곳 0 줄**(실행 코드 줄만. 주석·docstring 제외):
   - 없음(run 디렉터리 이름을 그대로 적은 실행 줄은 없다)
   - 이름 꼬리로 run 을 찾는 줄(`ls -d logs/rsl_rl/<실험>/*_<꼬리> | tail -1`) 8 곳 중 폐기 후보에 걸리거나 꼬리가 변수인 것:
     - `reports/gen_w3/scripts/w3train.sh`:17 `biped12_stepb0/*_$RNAME` → 꼬리가 변수라 실행 때 정해진다
     - `reports/gen_w3/scripts/w3train.sh`:21 `biped12_stepb0/*_$RNAME` → 꼬리가 변수라 실행 때 정해진다
     - `reports/gen_w3/scripts/ketrain.sh`:17 `biped12_stepb0/*_$RNAME` → 꼬리가 변수라 실행 때 정해진다
     - `reports/gen_w3/scripts/ketrain.sh`:21 `biped12_stepb0/*_$RNAME` → 꼬리가 변수라 실행 때 정해진다
     - `reports/gen_w3/scripts/w3s3dr.sh`:11 `biped12_stepb0/*_w3S3dR_from_s3d` → `2026-10-07_16-49-39_w3S3dR_from_s3d`
     - `reports/gen_w3/scripts/w3s3dr.sh`:14 `biped12_stepb0/*_w3S3dR_from_s3d` → `2026-10-07_16-49-39_w3S3dR_from_s3d`
3. **문서가 경로·이름을 적어 둔 곳**: 문서 19 개의 27 줄(판정 기록 — 고칠 필요는 없지만 경로가 죽은 링크가 된다). 많은 순: `reports/gen_step/OSS_STEP_RESEARCH.md`(6), `claude2.md`(3), `reports/gen_w3/WD_DESIGN.md`(2), `RL보행_명령어모음.txt`(1), `reports/gen_g22r/verdict_g22r_12500_s1.md`(1), `reports/gen_g22r/VERDICT_1003.md`(1), `reports/gen_g22r/verdict_g22r_13099_s1.md`(1), `reports/gen_w3/TORQUE_W3T_DESIGN.md`(1)
4. **평가 산출물 3440 건**의 머리말이 이 run 들의 체크포인트 경로를 가리킨다(`log_picture/` — 결과 파일은 남지만 같은 평가를 다시 돌릴 수 없게 된다).
5. **웜스타트 기록**: 폐기 후보끼리의 부모–자식 12 쌍은 `params/agent.yaml` 의 `load_run` 이 죽은 이름이 된다(재현용 기록일 뿐 실행에는 영향 없음). 남기는 run 의 부모는 계보 규칙으로 이미 기록 보존에 들어가 있다.
6. **이름순 마지막에 기대는 코드**:
   - `isaaclab_tasks.utils.get_checkpoint_path(log_root, run_regex, ckpt_regex)` 는 실험 폴더의 디렉터리(링크 포함) 가운데 이름이 정규식에 **앞에서부터 맞는**(`re.match`) 것을 이름순으로 세워 **마지막**을 고른다. `--load_run` 을 안 주면(`.*`) 폴더의 마지막 항목이 뽑힌다 — 지금 `biped12_stepb0` 의 마지막은 `2026-10-08_00-37-31_w3S3fF_from_15800`, `biped12_stage4` 의 마지막은 `biped12_stepb0`(고리 링크 — 안에 model_*.pt 가 없어서 `.*` 로 부르면 '체크포인트 없음' 오류로 끝난다).
   - 같은 함수라서 `--load_run <이름>` 도 앞부분 일치다: 옮긴 뒤 같은 접두사의 다른 run 이 남아 있으면 그쪽이 잡힌다(타임스탬프가 붙어 있어 지금은 겹치는 이름이 없다).
   - `rl_walking/deploy/policy_runner_stage4.py --selftest` 실모델 시험의 대체 경로 = `biped12_stage4/*/exported/policy.pt` 의 이름순 마지막 — 지금 PC 에서는 `logs/rsl_rl/biped12_stage4/2026-10-07_00-11-12_w3S3d_from_s2d/exported`.
   - 평가 체인 스크립트의 `ls -d logs/rsl_rl/biped12_stepb0/*_<이름> | tail -1`(`reports/gen_w3/scripts/w3s3e.sh`·`w3s3dr.sh`·`w3train.sh`) — 같은 꼬리 이름의 run 을 옮기면 다른 run 이 잡히거나 비게 된다.
7. **평가 태스크가 실험 폴더를 정한다**: `--load_run`·`--switch_to`·`--ke` 는 `logs/rsl_rl/<태스크의 experiment_name>/<run>` 에서만 찾는다(`eval_ground_recovery.py`·`eval_transition*.py`). 실험 폴더 밖으로 나간 run 은 이름을 알아도 잡히지 않는다 — 분리의 목적이기도 하고, 되돌릴 때는 원래 폴더로 돌려놓아야 한다는 뜻이기도 하다.

### 10.2 제안 순서

0. 조건: 학습·평가가 하나도 돌지 않을 때(이 목록을 만든 시각에는 평가 중 후보 6 개 — w3s3f, w3s3g, w3s3t, w3s3tk, g24raf, w3s3ff — 가 있고 평가 대기열이 돌고 있었다). 평가 중 후보와 그 부모·대조 run 은 건드리지 않는다.
1. `python3 reports/night_1007/policy_inventory.py --md5 all` 로 전 체크포인트 md5 를 남긴다(이 JSON 이 옮기기 전 기록이 된다).
2. 주 작업자가 §11 의 결정 사항을 정하고, JSON 에서 대상 목록을 뽑는다(`runs[].cls == "폐기 후보"`, 필요하면 `sub` 로 좁힘).
3. 옮길 곳: 같은 파일시스템의 `logs/rsl_rl/_archive_1007/<원래 실험 폴더>/<run>` — `_void_1003_bench/` 와 같은 구조다. 같은 디스크 안 이동이라 용량은 줄지 않는다. 용량을 줄이려면 그 폴더를 외부 저장소로 내보낸 뒤 지우는 단계가 따로 필요하고, 그것은 사용자 결정이다.
4. 순서: ① 대상 run 을 가리키는 링크 제거(10.1-1) → ② run 디렉터리 이동 → ③ `find logs/rsl_rl -xtype l` 로 끊어진 링크 0 확인 → ④ 회귀 확인: `python3 rl_walking/deploy/policy_runner_stage4.py --selftest` · `python3 jetson/scripts/rl_bridge_node.py --selftest` · `python3 jetson/scripts/operator_console.py --selftest` · `pytest tests/test_bridge_ema_contract.py` · `reports/regress/run_regress.sh --mode quick` → ⑤ 이 스크립트를 다시 돌려 분류·끊어진 참조가 예상대로인지 본다.
5. (선택) 솎기: 남기는 run 안의 비핵심 체크포인트 772 개(5.14 GiB)는 JSON 의 `runs[].thin_ckpts` 에 있다. `model_N.pt` 링크의 대상(`ckpt_links`)과 `key_ckpts` 는 제외돼 있다.
6. 하지 않는 것: `exported*/`·`params/`·`git/`·tfevents 는 run 과 함께 움직이고 따로 떼지 않는다. `log_picture/` 의 결과 파일은 손대지 않는다.

### 10.3 범위 밖이지만 같이 볼 것

- `deploy/` 의 옛 배포 묶음 `stand_7차_*.tar.gz` 25 개(232.1 MiB) — 안에 당시 정책 파일이 들어 있다(0907~0929 세대). 이 목록에서는 열어 보지 않았다.
- `.claude/worktrees/` — 에이전트 작업 사본. `logs` 는 대부분 본 트리 링크지만 사본 하나가 수 GB 다(추적되는 `log_picture/`·`reports/` 때문). 디스크 압박의 주원인은 체크포인트(20 GiB)가 아니라 이쪽일 수 있다 — 이 작업 중에도 여유 공간이 0 까지 내려갔다 돌아왔다. 정리는 `claude2.md` 의 절차(병합 확인 뒤 `git worktree remove`)를 따르는 주 작업자 몫이다.

## 11. 주 작업자가 정해야 할 것

1. **폐기 후보의 범위** — 문서 근거 140 개(12.22 GiB)만인가, 이름·설정 근거 32 개(552.6 MiB)도 포함인가.
2. **대조군·선정 근거 run 을 남길 것인가** — 여러 판정 문서가 수치로 인용하는 대조군과 선정 근거 12 개(872.3 MiB)를 기록 보존에 뒀다. 문서의 수치는 run 이 없어도 남지만 같은 평가를 다시 돌릴 수는 없게 된다.
3. **계보 조상을 어디까지 남길 것인가** — 문서 판정은 기각이지만 남기는 run 의 웜스타트 원본이라 올린 것이 12 개(1.21 GiB)다. 부모 체크포인트 하나만 남기고 나머지를 솎으면 대부분 줄어든다(§0 의 솎기 수치에 포함).
4. **솎기를 할 것인가** — 772 개 · 5.14 GiB. 폐기 후보 전체(12.76 GiB)와 합치면 `logs/rsl_rl/` 의 88 % 다.
5. **불명 8 개** — 0719·0725 평지 세대 중 문서에 역할이 안 적힌 것, G4, G24R-g-lr. 판정 기록을 남기고 폐기 후보로 내릴지, 그대로 둘지.
6. **배포 후보의 등록·명명** — §9 의 별칭과 `models/` 등록을 할지. 세 정책 모두 지금은 `logs/`(git 무시)에만 있다.
7. **실험 키의 처리** — exported 가 없는 채 등록만 된 키: `walk_w2q25590_hturn2`, `stand_w3a_11699`. 키를 남기면 그 run 은 기준선·검증 의존, 키를 내리면 기록 보존(계보·이전 후보)으로 바뀐다.
8. **끊어진 레지스트리 키 27 개** — 0918 이전 세대 키(`walk`·`march*`·`stand_v*`·`stand_hang_*`·`stand_crouch_v1*`)는 run 이 로컬에 없다. 레지스트리에서 내릴지는 코드 변경이라 이 작업 범위 밖이다.
9. **평가 중 후보(w3s3f, w3s3g, w3s3t, w3s3tk, g24raf, w3s3ff) 판정 뒤 재분류** — 통과하면 걷기 정책이 바뀌므로 §1·§3·§9 가 전부 달라진다. 스크립트는 `KEW3_SPECS`·`W3_SPECS` 를 읽으므로 코드가 바뀐 뒤 다시 돌리면 따라온다. 탈락하면 분류표(`CURATED`)의 그 줄을 근거 문구와 함께 고쳐야 한다(문서에 판정 문구가 생기면 스크립트가 §0 에 경고를 띄운다).
10. **`biped12_stage4/biped12_stepb0` 고리 링크** — 쓰는 곳을 코드에서 찾지 못했다(§7.3). 남길지 지울지.

## 12. 이 목록의 한계

- **학습 태스크 id 는 run 안에 저장되지 않는다**(`params/env.yaml` 에 없다). 표의 태스크는 코드·문서·`models/*/meta.json` 으로 확인한 것과, run 이름이 `rl_walking/__init__.py` 의 등록 단계 이름과 같아서 추정한 것을 구분해 적었다.
- 문서 인용은 run 디렉터리 이름이 문서에 그대로 나올 때만 자동으로 잡힌다. 'G15w'·'W2-8' 같은 약칭으로 쓴 판정은 분류표에 사람이 옮긴 것이고(근거 문구는 매번 다시 확인), 약칭으로만 인용된 다른 문서는 세지 못한다.
- 평가 산출물 수는 `log_picture/**/*.{txt,json,md}` 머리말 6 KB 에서 run·체크포인트를 찾은 것이다. 무효 폴더(`_invalid_*`)는 뺐다. scratchpad 에만 원자료가 있는 평가(예: `REPRO.md` 의 학습 seed 재현)는 0 으로 나온다 — 그런 run 은 분류표에서 따로 처리했다.
- '실행 코드 줄이 직접 읽는다'는 판정은 줄 단위 규칙이다(주석·docstring 제외, `--load_run`·`torch.load`·경로 상수 등). 변수에 담아 넘기는 스크립트는 놓칠 수 있다 — `reports/gen_w3/scripts/*.sh` 는 손으로 한 번 더 읽어 확인했다.
- md5: 모든 체크포인트 계산했다. 최근 3 분 안에 바뀐 파일은 건너뛰었다.
- 젯슨의 실제 파일 상태는 확인하지 않았다(접속 금지). §3 은 '이 파일들이 있어야 한다'는 요구 목록이지 '젯슨에 있다'는 확인이 아니다.
- 분류는 **문서에 적힌 판정을 옮긴 것**이고 정책 성능을 다시 잰 것이 아니다. 시뮬 판정을 실물 성공 가능성으로 읽지 않는다.

