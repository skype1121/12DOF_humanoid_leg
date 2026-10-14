# HTURN2 브리지 실험 플래그 (2026-10-06)

> 상태: **로컬 구현과 오프라인 검증까지 마쳤다. 실물 검증은 하지 않았고, 젯슨에도 배포하지 않았다.**
> 기본값은 꺼짐이다. 끈 상태의 브리지·러너 출력은 종전 코드(df9108db)와 비트 단위로 같다.

## 1. 범위와 원칙

- 사용자 방향(1006): 걷기의 실물 범위를 당분간 **앞 보행 + 방향(헤딩) 제어**로 좁힌다.
- HTURN2는 아래 세 가지를 묶은 것이다.
  - 앞 보행 정책 W2q 25590
  - 회전 피드포워드 α 1.5
  - HeadingHold kp 1, |wz| ≤ 0.3
- 시뮬 근거는 reports/night_1005/LOG.md의 "HTURN2 결과"와 "HTURN2-C 결과"다.
  - 오른 회전은 견고했다.
  - 왼 회전 +45° 도달률은 seed에 따라 71–94 %였다.
  - 낙상률은 0–12 %로, 대조군(α 0)과 같은 수준이었다.
- 이 기능은 **실험 플래그로만** 둔다. 기본 구조에 영구히 넣지 않는다.
  - 켜는 방법은 기동 인자 `--experimental-hturn2` 하나뿐이다.
  - 플래그를 켜도 `--checkpoint walk_w2q25590_hturn2`를 골라야 계약이 바뀐다. 즉 플래그와 체크포인트 키가 둘 다 필요하다.

## 2. 계약 (시뮬 원본 → 실물 대응)

### (a) 명령 연동 cadence 클록 (W2q)

원본은 `mdp_hang.cadence_of_cmd`, `mdp_hang.gait_phase_update`, `mdp_stage4._gait_phase`다.

- 주파수 f
  - 걷기 모드(|vx| ≥ 0.03): f = clamp(0.7 + 4|vx|, 0.8, 1.6) Hz
  - 서기 모드: f = 1.1 Hz
- 위상 u (주기 단위)
  - 러너 틱 0과 1에서는 u = 0이다. env의 `episode_length_buf ≤ 1` fresh 규칙과 같다.
  - 그 뒤에는 u ← (u + f·0.02) mod 1이다. 여기서 f는 이번 틱의 명령으로 계산한다.
- 관측 `gait_clock` = [sin 2πu, cos 2πu]
- 클록 주파수 `gait_freq`(1.1)는 이 모드에서 쓰이지 않는다.

### (b) 회전 피드포워드

원본은 `mdp_hang.install_turn_assist`의 hook이다.

- 앞당긴 위상: u' = (u + f·6·0.02) mod 1
- v = (2u') mod 1, ss(x) = x²(3−2x)
- 위상 모양
  - u' < 0.5(왼발 스윙): shL = ss(v), shR = 1−ss(v)
  - u' ≥ 0.5: shL = 1−ss(v), shR = ss(v)
- 진폭: amp = α·κ·wz·[|vx| ≥ 0.03], α = 1.5, κ = −0.455
- 적용: `left_hip_r += amp·shL`, `right_hip_r += amp·shR`
  - 단위는 rad, 관절 프레임이다.
  - 최대 크기는 1.5·0.455·0.3 = 0.205 rad(11.7°)다.
- **적용 위치**
  - `rl_walking/actions.py`의 `process_actions` 순서는 다음과 같다: `target = 기본자세 + 0.25·raw` → **pre_delay_hook** → 지연 버퍼 → 슬루 4°/틱 → LPF → 온셋 → 백래시.
  - 실물 체인 순서는 러너 target → **러너 슬루** → 브리지 LPF β0.27 → 노드 슬루다.
  - 시뮬 지연 버퍼는 실물의 물리적 전송·응답 지연을 모형화한 것이라 실물 쪽에 대응하는 코드가 없다.
  - 따라서 대응 위치는 **러너에서 target을 계산한 직후, 러너 슬루 앞**이다. `policy_runner_stage4.step`이 이 자리에서 잔차를 더한다.
  - 관측 `actions`는 계속 raw 값이다. 시뮬의 `last_action`도 raw다.
- **틱 대응**
  - 관측 n은 (cmd_n, u_n)으로 조립한다. 그 관측에서 나온 행동의 피드포워드는 (u_n, f_n, cmd_n)을 쓴다.
  - 이 대응은 ManagerBasedRLEnv.step의 학습 순서와 같다.
  - 예외: `eval_walk.py`는 env.step 전에 명령을 바꾼다(set_cmd). 그래서 평가에서는 hook의 vx·wz만 관측보다 1틱 새 값이었다. 위상은 같았다. 계약은 학습 순서를 따른다.

### (c) HeadingHold

- 식: wz = clip(kp·wrap(ψ* − ψ), ±0.3), kp = 1
- 목표 ψ* = **시작 요(정책 시작 첫 틱) + 목표 변화**로 정의한다. `eval_walk.py`의 `--heading_target`과 같다.
- 브리지는 기존 `HeadingHold` 클래스를 그대로 쓰고 kp와 상한만 바꾼다. 종전 기본값은 1.0과 0.6이었다.
- wrap 방식은 원본(atan2)과 다르지만(모듈로), ±π 부근을 빼면 차이가 1e-15 이하다.

### 그 밖의 배포 계약

- 기본자세는 CROUCH_V5다. 학습 env.yaml의 init_state와 일치하며, selftest '학습 계약'이 확인한다.
- 관측은 320차원이다.
- 관측 필터는 act EMA β0.27과 jvel LPF β0.25다. 이 계약 때문에 env.yaml이 반드시 있어야 한다.
- 브리지 LPF β0.27을 적용한다. 종전에는 stand 계열에만 적용했다. HTURN2 평가가 LPF 0.27 조건에서 이뤄졌기 때문이다.

## 3. 플래그와 사용법

구현 파일:

- `rl_walking/deploy/hturn2.py`: 계약 상수(`Hturn2Config`)와 numpy float32 식.
- `PolicyRunnerStage4(hturn2=None)`: `None`이 기본이며 종전 경로를 그대로 탄다.
- `rl_bridge_node.py`:
  - `--experimental-hturn2` → `enable_experimental_hturn2()`가 키 `walk_w2q25590_hturn2`를 등록한다.
  - 등록 항목: 경로(W2q run의 `exported/`), f0 1.1, CROUCH_V5, env.yaml 필수.
  - 플래그 없이 이 키를 고르면 기동을 거부한다(exit 2).

명령 형식 (HTURN2 키에서만 받는다):

- `{"cmd":"walk","vx":0.10,"heading_target_deg":45}`
  - vx는 반드시 명시해야 하고 범위는 0 ≤ vx ≤ 0.15다.
  - vy와 wz는 0이어야 한다.
  - heading_hold를 끌 수 없다.
  - 목표 각은 ±180° 이내여야 한다.
  - 이를 어기면 거부한다.
- `{"cmd":"turn","heading_target_deg":-30}`
  - 정책을 리셋하지 않고 목표 요만 바꾼다. 기준은 시작 요다.
  - 플래그가 꺼져 있거나 다른 키로 실행 중이면 경고를 남기고 무시한다.
- `stop`: vx = 0이 되어 서기 모드로 간다. HeadingHold는 계속 동작한다.
- 블랙박스에는 HTURN2 경로에서만 `hturn2` 필드(u, f, ff_hip_r, 목표 각, 기준 요)와 `hturn2_turn` 이벤트가 추가된다. 끈 상태의 스키마는 종전과 같다.

**끄는 법**: `--experimental-hturn2`를 빼고 기동한다. 그것이 기본 상태다. 코드에서 걷어내려면 위 세 파일의 `1006 HTURN2` 블록을 지우면 된다. 다른 경로는 이 블록에 의존하지 않는다.

## 4. 검증 (오프라인 검증 완료)

| 검사 | 결과 |
|---|---|
| 브리지 selftest(플래그 없음) | ALL PASS — 기존 21항목 출력 동일, 신규 1항목 추가 |
| 브리지 selftest `--experimental-hturn2` | ALL PASS |
| 플래그 + HTURN2 키 selftest | '학습 계약' PASS(env.yaml: CROUCH_V5·act 0.27·jvel 0.25·지연 0~3틱). '정책 500틱'은 **FAIL(의도된 결과)** — exported 없음 |
| 러너 selftest | ALL PASS |
| `selftest_hturn2_equiv.py` [A] 꺼짐 러너 | df9108db 러너와 4000틱 비트 동일(스텁 245/320/512·필터·리셋, g21a 실모델) |
| [B] 켜짐 = 시뮬 소스 실행(AST 추출 torch) | 900틱 위상 u·f 비트 일치, hip_r 잔차 899틱 비트 일치, 클록 관측 최대 차 6e-8(1 ulp: numpy와 torch의 sin 차이) |
| [C] HeadingHold | eval 식과 차이 1e-15 |
| [D] 꺼짐 브리지 코어 | df9108db 브리지와 g21a·v18b 실모델 300틱 비트 동일 |

이 결과는 계약 동치를 보인 것이다. 폐루프 성능을 증명하지는 않는다.

## 5. W2q 25590 내보내기 (미실행 — 방법만 기록)

`exported/`가 없어서 이번에는 만들지 않았다. Isaac에서 play.py를 실행해야 하므로 이번 작업 범위 밖이다. 절차:

```
/home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/play.py --task Biped12-HangStand-W2q-v0 --num_envs 1 --headless \
  --checkpoint logs/rsl_rl/biped12_stepb0/2026-10-05_18-18-43_w2q_seed1_from_w2h24791/model_25590.pt
python3 rl_walking/scripts/register_model.py walk_w2q25590 --experiment biped12_stepb0 \
  --run 2026-10-05_18-18-43_w2q_seed1_from_w2h24791 --ckpt model_25590.pt --task Biped12-HangStand-W2q-v0
```

- `--checkpoint`는 반드시 경로로 지정한다. 지정하지 않으면 최신 ckpt를 고를 수 있다(deploy_path_check.md 함정).
- 내보낸 뒤에는 policy.pt 가중치가 model_25590.pt의 actor와 같은지 대조한다. register_model.py는 이 대조를 하지 않는다.
- 그다음 플래그와 키로 selftest '정책 500틱'이 PASS하는지 확인한다.

## 6. 남은 위험

- **실물 미검증.** 회전 피드포워드, cadence 클록, W2q 보행 모두 실물에서 한 번도 돌리지 않았다. 시뮬 성공률을 실물 성공 확률로 쓰지 않는다.
- **왼 회전이 약하다.** +45° 도달률이 seed에 따라 71–94 %, +90°는 60–82 %다. 원인은 자산이나 정책의 비대칭으로 보지만 아직 분리하지 못한 가설이다.
- **지연.**
  - HTURN2-C 기준으로 지연 3틱이 기준 조건이다. 4틱은 정보용 결과로, 낙상 10 %·+90° 60 %였다.
  - 실물의 실효 지연은 stage8 수정(1e31a27) 뒤 아직 실측하지 않았다.
  - 피드포워드는 위상을 6틱 앞당겨 적용하므로, 실효 지연이 다르면 스윙 위상과 어긋난다.
- **hip_r 부호.** κ의 부호(−)는 시뮬 자산에서 실측했다. 실물 hip_r 방향(하드웨어맵 DIR)이 시뮬과 같은지는 이 기능 기준으로 다시 확인하지 않았다. 발목롤에는 반전 전례가 있다.
- **평가와의 1틱 차이.** 평가에서는 hook의 명령이 관측보다 1틱 새 값이었다. HeadingHold는 연속이라 영향이 작다고 보지만 측정하지는 않았다.
- **온셋 블렌딩.** 브리지의 1.5 s 블렌딩은 HTURN2 평가(eval_walk.py)에 없던 조건이다. 학습에는 온셋 홀드 0–0.3 s만 있었다.
- **제자리 회전 불가.** 회전은 앞 걸음에 묶여 있다(SRTURN 기각). vx < 0.03이면 피드포워드가 0이 된다.

## 7. 배포 전 승인 항목 (사용자 결정)

1. W2q 25590 export와 가중치 대조, 모델 등록(로컬).
2. 젯슨 배포: 러너, 브리지, hturn2.py, exported, params/env.yaml.
3. 실물 실험 계획을 사전 등록. 포함할 것:
   - 공중 hip_r 부호와 피드포워드 진폭 확인(정책 없이 또는 매달린 상태)
   - 줄 지지 상태에서 vx 0.05~0.10 직진
   - ±45° 목표 회전
   - 성공·중단 기준
4. 실효 지연 실측(블랙박스)과, 그 지연을 반영한 시뮬 재평가 여부.
5. 콘솔/robotctl에서 `turn` 명령을 노출할지 여부. 지금은 브리지 토픽에서만 받는다.
