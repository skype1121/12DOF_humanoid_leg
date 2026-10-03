# G21a-11300 진단 실물 준비 (1003) — 1·2단계 (PC, 젯슨 미접촉)

승윤님 지시(1003): 1–2단계 진행 → G21a-11300 진단 배포 → G12 와 교대 시험. **승격 아님.** 비교 항목은
① 3 Hz 진폭·성장률 ② 발 이륙→착지→하중 수용 ③ 착지 후 정착 시간. G21c2 는 두 번째 진단 카드.

## 1단계 — 체크포인트별 관측 계약 (오프라인 검증 완료)
| 체크포인트 | 직전 액션 관측 | 관절속도 관측 | joint_vel_ema 모드 | 근거 |
|---|---|---|---|---|
| `stand_crouch_g12` | 원시(필터 없음) | 원시 | v2 (0929·1001·1002 실물과 동일, 변경 없음) | env.yaml: last_action / encoder_joint_vel |
| `stand_crouch_g21a` | 원시 액션 EMA β 0.27 | 유한차분 EMA β 0.25 | v2 | env.yaml: last_action_lpf / encoder_joint_vel_lpf(beta 0.25) |

- 계약은 브리지가 **체크포인트의 학습 env.yaml 에서** 결정한다(`obs_filter_contract`). 전역 적용 없음. 모르는 `*lpf*` 함수는 기동 거부.
- 정책 전환·reset 시 러너를 새로 만들고 EMA 상태를 비운다(브리지 selftest: g21a → g12 → g21a 전환에서 `_ema` 비어 있음).
- G21a 산출물: model_11300 → exported (ckpt 대비 최대 6.1e-5), md5 policy.pt 2464f5cd…, policy.onnx 43f11310…, env.yaml ebda410e….
- selftest: 러너 `--selftest` ALL PASS(관측 필터 계약 포함 9종), 브리지 `--selftest` ALL PASS.

## 2단계 — 학습 env ↔ 러너 동치 (`runner_equiv.py` / `runner_equiv.sh`)
Isaac env(관측 잡음·자이로 지연 끔, 액션 LPF β 고정)를 정책으로 600 틱 굴리며, 매 틱 env 0 의 원시 입력
(자이로·중력·명령·엔코더 관절각·유한차분 속도·접촉 비트)을 러너에 넣어 관측 블록과 정책 출력(raw)을 비교.
러너 인자는 브리지가 같은 체크포인트에 주는 값과 같다.

| 실행 | 관측 최대 오차 | 정책 출력 최대 오차 |
|---|---|---|
| G21a (act 0.27 · jvel 0.25) | 2.1e-7 | **5.4e-7** |
| G12 ema0 | 2.8e-7 | 2.8e-7 |
| G12 v2 (현재 배포·실물 사용) | joint_vel_ema 0.50 | 0.53 |

발견 2건(둘 다 이번 변경 이전부터 있던 차이, **러너는 바꾸지 않음** — 결정 대기):
1. **EMA 재시작 틱**: 시뮬 `_ema_state` 는 episode_length_buf ≤ 1(첫 두 틱)에서 현재값으로 재시작, 러너는 첫 틱만.
   gyro_ema·joint_vel_ema 의 시작 과도만 다르다(차이 × 0.9ⁿ 로 소멸, 정지 서기 인계에서는 자이로≈0 이라 작음).
   하네스는 `--sim_restart2 1` 로 이 규약만 맞춰 비교했다. 맞추지 않은 원본 결과 = `runner_equiv_g21a_norestart2.txt`.
2. **G12 v2 = 알려진 계약 불일치(1001)**: G12 는 학습 중 joint_vel_ema 가 0+잡음(구버전 함수)이었고, 배포 `stand_crouch_g12` 는 실제 EMA 를 넣는다.
   `stand_crouch_g12_ema0` 은 학습과 일치(위 표). 지시대로 G12 는 기존 그대로 두며, 교대 시험의 G12 는 1002 실물과 같은 v2 다.

이 비교는 **계약 검증**이다(폐루프 실물 성능의 증명 아님).

## 다음 (승인 필요)
젯슨 배포(러너·브리지·G21a exported+params, md5·젯슨 selftest) → 무외란 하네스 서기(3 Hz 즉시 성장 시 중단) →
G12 → G21a → G12 → G21a 같은 약한 앞 밀기·같은 영상 위치·같은 로거.
