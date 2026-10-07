# claude2 인계 문서 (2026-10-07 14:35, claude1 작성)

> 이 문서는 두 번째 Claude 세션(claude2)이 지금 상태를 정확히 이어받기 위한 것이다.
> 이 저장소의 규칙은 `CLAUDE.md`에 있고, **그것이 이 문서보다 우선한다.** 먼저 `CLAUDE.md`를 읽을 것.
> 사용자(승윤님)에게 보이는 모든 출력은 **한국어**로 쓴다.

---

## 0. 가장 먼저 지킬 것 (양보 불가)

1. **젯슨 배포, 실제 모터 구동, 실제 로봇 실행, 실제 push test, 하드웨어 설정 변경은 사용자의 명시적 승인 없이 절대 하지 않는다.**
   CAN 송신, ARM, 토크 인가, 영점(stand_zero_deg) 변경, guard_mode 변경, sensors_up.sh 변경도 마찬가지다.
2. **프로세스 종료는 자기가 띄운 명시 PID로만 한다.** `pkill`은 금지다. `pgrep -f`는 자기 셸까지 매칭되므로 쓰지 않는다.
   claude1이나 claude1의 서브에이전트가 띄운 학습·평가는 절대 끄지 않는다.
3. **실물 회수 순서**는 줄 인장 → 스택 종료 → 그 다음에만 release_all이다. 가드와 슬루는 완화하지 않는다.
4. **미보정 패드 발력%·CoP는 판정에 쓰지 않는다.** 패드는 접지 0/1 확인용이다.
5. 사용자 이메일은 외부로 보내지 않는다.

## ★ 인계 상태 (14:40 갱신) — 모든 작업이 claude2로 넘어간다

- 사용자 지시로 **claude1 계정은 토큰이 부족해 작업을 종료했다.** 이후 연구·병합·LOG 기록·실물 준비를 **claude2가 모두 맡는다.**
- 아래 §1에서 'claude1 담당'이라고 적힌 영역(`rl_walking/deploy/`, `jetson/`, `deploy/`, `LOG.md`, 병합)도 claude2가 맡는다.
- 정리 상태
  - claude1의 서브에이전트는 모두 종료됐다.
  - 남아 있던 감시 셸 6개와 7시간 넘게 멈춰 있던 평가 1개(`repro/eval_w3g.py` model_16000)는 명시 PID로 종료했다.
  - 14:40 기준 GPU에 도는 작업은 없다(613 MiB).
- 모든 결과는 `hang2stand-prep-0911`에 병합돼 있다(최신 `98d14f5a`).
  - 미커밋: `claude2.md`, `deploy/MANIFEST.md5`, `reports/night_1005/JOBS.md`. 마지막 둘은 원래부터 수정 상태였다.
- 남아 있는 worktree(`.claude/worktrees/agent-*`)는 병합이 끝난 것들이다. 디스크가 부족하면 브랜치가 메인에 병합됐는지(`git merge-base --is-ancestor`) 확인한 뒤 `git worktree remove`로 정리해도 된다.
  - 단, worktree 안 `logs/`의 체크포인트는 먼저 본 트리로 옮겨야 한다.
  - W3S4dD 두 실행은 이미 본 트리로 옮겼다.
  - c2 W3S3dC2 실행은 본 트리 `logs/`에 있다.
- 메모리(`~/.claude/projects/-home-ryu-humanoid-leg-test1/memory/`)는 claude2가 링크해서 읽고 써도 된다.

## 1. 두 세션 협업 규칙 (claude1 ↔ claude2 — 참고: 지금은 claude2 단독)

- 두 세션은 같은 폴더 `/home/ryu/humanoid_leg_test1`를 공유하고, 현재 같은 브랜치 `hang2stand-prep-0911` 위에 있다.
- **git**
  - `git add -A`, `git commit -a`는 금지한다. 자기가 고친 파일만 이름을 지정해 `git add <파일>`로 올린다.
  - 큰 작업은 `git worktree add ../h2_c2 -b claude2-<주제>`처럼 별도 worktree·브랜치에서 하는 것을 권장한다.
    병합은 claude1이 하거나, 사용자에게 알린 뒤 직접 한다.
  - worktree를 만들면 디스크 절약을 위해 `logs/`는 원본으로 심볼릭 링크하고, 그 링크는 커밋하지 않는다.
    (1007에 worktree 130 GB로 디스크 99 % 사고가 있었다.)
- **파일 담당**
  - claude1: `rl_walking/deploy/`, `jetson/`, `deploy/`, `reports/night_1005/LOG.md`, 병합, 실물 준비.
  - claude2: 새 평가 스크립트(새 파일로 추가), `reports/` 아래 자기 이름의 새 문서.
  - 기존 공용 파일(`eval_w3.py`, `eval_ground_recovery.py`, `hang_stand_env_cfg.py`, `__init__.py`)을 고쳐야 하면, 먼저 `reports/night_1005/JOBS.md`에 "claude2: <파일> 수정 중"이라고 적고 짧게 끝낸다.
- **GPU는 1장뿐이다**(약 24 GB, 학습 1개가 7–11 GB를 쓴다).
  - 모든 GPU 평가는 공용 잠금 안에서만 돌린다.
    ```bash
    LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
    exec 9>$LOCK; flock 9
    # nvidia-smi 사용량 ≤ 10500 MiB 될 때까지 대기 후 실행
    ...
    flock -u 9
    ```
  - 학습은 잠금 대상이 아니지만, **학습은 동시에 1개만** 돌린다. 기동 전 메모리가 ≤ 7500 MiB인지 확인한다.
  - 로그에 `PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch`가 있으면 그 실행은 **무효**다. 산출물이 생겼어도 재실행한다.
  - 평가 래퍼 `scratchpad/safe_eval.sh`(TASK RUN CK TAG [인자])에 메모리 대기와 무효 재시도가 들어 있다. 경로는 위 LOCK과 같은 scratchpad 폴더다.
- **판정은 산출물로 한다.** `isaaclab.sh`는 에러를 삼키고 exit 0을 낸다. `log_picture/`의 결과 파일, seed·체크포인트·표본 수·완주 여부를 확인한다.
- 진행 상황은 `reports/night_1005/JOBS.md`에 "claude2: 무엇을 하는 중"으로 한 줄 남긴다.

## 2. 최종 목표 (변하지 않음)

실물 로봇 locomotion 시스템을 만든다:
**STAND ↔ WALK(앞/뒤·옆) ↔ TURN ↔ 외란 → 회복 스텝 → 운동량 억제 → 양발 안정 접촉 → 수평 기준 자세 → 정상 STAND/WALK 복귀.**

- 판정 기준은 시뮬 점수가 아니라 **실물 신뢰성**이다.
- **회복 성공**은 외란 → 회복 스텝 → 각속도 억제 → 양발 접촉 → 수평 기준 자세로 복귀 → 정상 서기 유지까지 간 경우다. 이상한 자세로 균형만 잡고 있으면 실패다.
- 원칙
  - 원인 주장에는 대조군을 붙인다.
  - 같은 실패가 두 번 나오면 보상 가중을 만지지 말고 구조를 바꾼다.
  - 승격 기준은 실행 전에 문서에 등록한다.
  - 256 env × seed 1개에서 0–5건 차이로는 결론을 내지 않는다(회복은 seed ≥ 6).
  - 짝 비교는 RNG 지문이 같을 때만 McNemar로 한다.

## 3. 로봇과 스택 요약

- 12자유도 이족 하체다. 모터는 AK70-10 ×10과 발목롤용 AK45-36 ×2이고, 실질량은 17.62 kg이다.
- 센서: iAHRS IMU, RA30P 발 패드 8개(좌우 각 4개), D455.
- 젯슨에서 `jetson/scripts/operator_console.py`(포트 8800)가 실기 조작 전체를 맡는다. 브리지는 `jetson/scripts/rl_bridge_node.py`, 정책 러너는 `rl_walking/deploy/policy_runner_stage4.py`이다.
- 시뮬은 Isaac Lab v2.3.2, `/home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/...`로 돌린다.
- 1006 저녁에 사용자 요청으로 RA30P 패드 8개를 살리고 오프셋을 적용했다. 젯슨 `~/rl_calib/sensors_up.sh`에 `left_pad_bias [0,85,0,0]`, `right_pad_bias [0,0,15,0]`이 들어가 있고, 백업은 `.bak_1006_2032`이다. **이후 센서 설정 변경은 승인 사항이다.**

## 4. 어제(1006)부터 오늘(1007)까지 한 일 — 시간순 요약

### 1006 낮 ~ 저녁: W3 걷기 구조 확립
- **W3 구조**: COMMAND → gait_enable 래치(반주기 경계에서만 정지) → 위상 클록 → 발 착지 목표 → RL 잔차.
  - 보상: w3_stride(스윙발 이륙→착지 변위 목표 v/f), w3_track_rel(한 주기 박스 평균 속도), w3_drift_net, w3_yaw_net.
  - 설계 문서: `reports/gen_w3/W3_DESIGN.md`, `S2_DESIGN.md`, `S3_DESIGN.md`, `S4_DESIGN.md`.
- **W3S2-15195**: 정책 하나로 제자리·앞·뒤·옆 걷기가 된다(옆 ±0.05 부호 97 %).
- **롤 램프 발견**: 숨은 hang_support 롤 토크 램프가 120 s 낙상 12건 중 10건을 설명했다. 그래서 평가기에 `--ramp_off` 옵션을 추가했다.
- **헤딩 고리**: wz = clip(0.5·wrap(ψ*−ψ), ±0.4)를 정책 밖에서 계산한다. 학습할 때 heading_command는 꺼야 한다.
- **W3S2d**(지연 DR 1–5): 지연 5 낙상 80 → 35(이득 유지), 지연 4 개선은 유의하지 않음.
- **KE 회복 재검**: 살아남은 env의 97–99 %가 수평 정상 자세로 돌아온다(KABS2/kabs3). 체인 성공률은 sf15 38 %, fi40 38 %, b18 78 %이고, KE READY를 유지했다.

### 1006 밤 ~ 1007 새벽: 회전·혼합·회복 판정
- **W3S3d-15800**(실물 WALK/TURN 1차 후보)
  - 요 이득 0.84–0.91, 헤딩 고리 120 s 표류 2.2°.
  - export 완료, 배포 러너·브리지에 `--experimental-w3`로 로컬 반영(젯슨 미배포). md5는 `deploy/w3s3d_1007.md5`.
- **W3S4d-16299**(혼합 2차 후보): 앞+회전 혼합은 성립한다. 낙상 4–5 %, 헤딩 고리를 켜면 앞으로 과속한다.
- **odic → odicw**: 오도메트리 기반 회복 종료 판정기를 만들어 KE 조기 복귀 후보로 올렸다(STOPEST §7–§12).
- **c2**(W3 엔진 회복 학습): 사전 규칙으로 GO 판정을 받고 W3S3dC2를 학습했다.
- **W3D**(걷기 중 외란 회복) 설계 → W3S4dD 학습.
- **REAL_TEST1_PLAN.md**(실물 1차 시험 계획, 사전 등록 v1) 작성.

### 1007 05:28 ~ 14:30: 9시간 강화 연구 (사용자 지시)
에이전트 9개가 역할을 나눠 맡았다(envelope, 재현성, 회귀 자동화, odicw 공격, parity, 적대 검토, 보강대비, c2, W3S4dD). 중간에 사용량 한도로 두 번 전원이 멈췄다(07:50–08:55, ~12:30–13:40). 결과는 아래와 같고, 모두 메인 브랜치에 병합되어 있다(최신 커밋 `98d14f5a`).

| 주제 | 결론 | 문서 |
|---|---|---|
| c2 W3S3dC2 | **KILL.** 앞 밀기에서 회복 스텝 대신 버티기를 학습했다(fs80 스텝 98→12 %). 체인 성공이 KE보다 −28 %p, 무외란 걷기 낙상 10–17 %. 회복 중 켜 둔 knee_stance 서기 항이 이륙을 억제한 것으로 본다(가설). | `C2_DESIGN.md` §15 |
| W3S4dD run1·2 | 미승격. 앞 18 N 낙상은 줄었지만 회복 보폭이 늘지 않았고, 밀린 뒤 정지가 계속 걷기보다 위험했다. 같은 실패가 2회 나왔다. | `WD_DESIGN.md` §9, `WD_S4dD_EVAL_1007.md` |
| 구조 변경 rsup(학습 없음) | a 정지 래치 지연, b 포획점 명령 오프셋 **둘 다 효과 없음.** IMU 피치·롤 편차 감지기는 걸음 흔들림(p99.9 12–13°)과 12 N 충격(4.8–7.1°)을 구분하지 못한다. 다음 단계는 **보행 위상별 평균과 비교한 편차 감지기**다. | `WD_DESIGN.md` §10, `RSUP_EVAL_1007.md`, `rl_walking/w3_rsup.py` |
| odicw 공격 | odicw의 낙상 감소는 **회복 모드를 약 0.5 s 늦게 끈 효과**로 설명된다. 시간을 맞춘 대조 imuH(현행 imu, 조기 복귀 유지 0.2→0.7 s)와 구분되지 않는다(지연 2 짝 8/9). **imuH 권장, 승격은 사용자 결정.** 오도메트리·무릎 k̂·τ 결측 차단은 제거를 권고한다. 1003 재흔들림은 합성 입력으로 재현했다. | `STOPEST.md` §13–§15 |
| parity 감사 | 시뮬 ↔ 배포 러너·브리지가 관측 3.6e-7, 행동 1.2e-6 차이로 일치한다. ONNX도 일치. 불일치 2건은 옵션으로 고쳤다(`--runner-ema-fresh 2`, `--c5-imu-f32`, 기본은 기존과 비트 동일). | `PARITY_AUDIT_1007.md`, `rl_walking/deploy/selftest_w3_rollout_parity.py` |
| 재현성 | **15800은 운 좋은 점이다.** 학습 seed×체크포인트 15조합 중 최저(0.99 %)이고, 레시피 기대 낙상은 2–3 %(범위 1.0–4.6)다. 회전·헤딩·추종 기능은 학습 seed 5개 모두에서 재현된다. 실물 1차 후보는 유지. | `REPRO.md` |
| envelope | 가장 민감한 축: **IMU 피치 편향**(±3° 경계, ±6° 붕괴), **CoM 앞뒤**(+3 cm 붕괴). 둘이 같은 방향으로 겹치면 28 % 낙상. 발목 유격은 속도 이득을 좌우한다(유격 ×0이면 2.35, ×2이면 0.51). 120 s에서 왼쪽 힙 영점 −4°·kp 비대칭은 8–18 % 낙상. 둔감한 축: 접촉 오류, 롤 편향, IMU 잡음, kd, 명령 크기. | `ENVELOPE.md`, `rl_walking/scripts/eval_w3_envelope.py` |
| 적대 검토 | 앞 0.06 명령의 실제 직진 속도는 ≈0.077(+28 %)이고, 기존의 '정확 추종'은 요 표류 cos로 생긴 과소값이었다. 걷기 외란 기준선은 지연 1에서 잰 것이라 지연 2로 정정했다. W3S4d 혼합 커리큘럼은 s=0에서 진급하지 않아 혼합 평가는 외삽이다. 헤딩 15° 고장 문턱은 시뮬 과도 꼬리 바로 위에 있다. | `REVIEW_1007.md` |
| 회귀 자동화 | `reports/regress/run_regress.sh` + `summarize.py` + `THRESHOLDS.md`(v2, 지연 2·접촉 v2 기준). 현 후보 빠른 모드 PASS(경계 1: 걷기→정지 정착 90.9 %). G21a 단독 b18 낙상 22.8 %(지연 2). | `reports/regress/` |
| STAND 모드 비교 | 낙상(W3 ge0 / G21a / KE): b18 37 / 23 / 4 %, sf15 99 / 92 / 18 %, 무외란과 착지 과도는 동급. → **밀릴 수 있는 단계와 W3 정지 뒤에는 KE로 인계.** W3 정지→G21a 인계 과도는 미측정. | `reports/regress/STAND_MODE_CMP.md` |
| 보강대비 | 약점 20개 대장과 현장 대응 카드를 만들었다. 대비책은 기본 꺼짐으로 구현했다: `--w3-monitor`(이상 시 서기·정지 래치), `--w3-hl-fault`(헤딩 고장 판별 분리: IMU-자이로 불일치 2 s 6°, 15° 5 s 연속, 반대 회전 2 s, 즉시 30°), `--c5-overtime`, `w3_cmd.py`. 지연 감시 문턱은 8틱(시뮬 지연 + 3). 헤딩 wz 상한은 과속 원인이 아니라서 기각했다. | `CONTINGENCY.md`(CONT_v3), `ANOMALY_CARD.md`, `rl_walking/deploy/w3_contingency.py` |

## 5. 현재 최종 후보 (1007 14:00 기준)

| 역할 | 후보 | 경로 |
|---|---|---|
| STAND | **KE** = G21a + G24Ra 회복 전환 | G21a: `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/model_11300.pt`, G24Ra: `logs/rsl_rl/biped12_stage4/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/`(model_21793) |
| KE 인자 | `--sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33 --contact_ver v2` | `scratchpad/kabs.sh` |
| WALK/TURN 1차 | **W3S3d-15800** (태스크 `Biped12-HangStand-W3S3d-v0`) | `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt` (+ `exported/`) |
| 혼합 2차 | W3S4d-16299 (태스크 W3S4d) | `logs/rsl_rl/biped12_stepb0/2026-10-07_01-53-16_w3S4d2_from_s4d/model_16299.pt` |
| 회복 조기 복귀 판정 | 현행 imu (권장 imuH, odicw는 대안) — **승격 결정 대기** | `STOPEST.md` §15c |

- 기각·보관: c2 W3S3dC2(`2026-10-07_02-43-05_w3S3dC2_from_s3d`), W3S4dD(`2026-10-07_03-20-15_w3S4dD_from_s4d`, `2026-10-07_04-59-14_w3S4dDb_from_s4d`), W3S3d seed 2–5 재학습 실행.

## 6. 사용자 결정 대기 (claude2가 대신 결정하지 말 것)

- **D7 헤딩 처리**: 현행 15° 유지 / `--w3-hl-fault` / 헤딩 고리 끄기 / 이득 1.0. T7은 30 s로 제한하는 안도 있다.
- **imuH 승격** 여부(사전 절차상 독립 seed 7–12 재확인 권장).
- 새 대비책(기본 꺼짐)을 실물에서 켤지.
- guard_mode safe vs data(REAL_TEST1 D1), 현장 측정 M1–M4(실효 지연, IMU 요 표류, 무릎 k, R3 패드 배선).

## 7. 지금 남은 일 = claude2에게 권하는 작업 (시뮬만, 하나씩)

우선순위 순서다. 각 작업은 **실행 전에 사전 기준을 문서에 등록**하고, 끝나면 결과 문서를 커밋한 뒤 사용자에게 한국어로 짧게 보고한다.

1. **`--w3-monitor` 켬 폐루프 재시험(E2 재실행).**
   - 1007 E2는 문턱 계산 스크립트 실수로 '켬' 팔에도 빈 설정이 넘어가 무효였다.
   - `CONTINGENCY.md`의 E2 절과 `reports/gen_w3/cont_thresholds.py`를 고친 뒤 같은 4개 실행으로 켬/끔 짝 비교를 한다.
   - 확인할 것: 해가 없는지(무외란 낙상·추종 동일)와 효과가 있는지.
   - 평가기는 `eval_w3.py --monitor`다.
2. **W3 정지 → KE(G21a) 인계 과도 측정.**
   - STAND_MODE_CMP의 결론은 "W3 정지 뒤 KE로 인계"인데, 인계 순간의 목표 불연속·낙상은 아직 재지 않았다.
3. **imuH 독립 seed 7–12 확인.**
   - imuH를 후보로 다시 등록한 뒤 지연 2·3에서 P1–P5를 확인한다(STOPEST §13 기준).
   - 지연 3 장기 체류 여유가 작은 점(8.4 % vs 문턱 8.6 %)에 주의한다.
4. **위상 정합 외란 감지기**(rsup 다음 단계).
   - 같은 보행 위상의 평균 IMU 편차를 빼서 걸음 흔들림과 12 N 충격을 구분할 수 있는지, 학습 없이 시뮬 로그로 먼저 본다.
5. envelope 미측정 축: seed 2, 반대 방향 명령(옆 −0.05, 회전 −0.2), 좌우 CoM, W3S4d 대부분 축.

## 8. 자주 밟는 함정

- 평가 seed와 학습 seed를 구분한다("seed 2 재현"이 평가 seed였던 적이 있다).
- 롤 램프를 켰는지 껐는지, 지연 몇 틱인지, 접촉 v1/v2인지를 **표마다 적는다.** 조건이 섞이면 결론이 뒤집힌다.
- 앞 속도는 본체 프레임과 직선 변위를 함께 보고한다. 시작 요 프레임 값은 요 표류 때문에 과소하게 나온다.
- 빈 실행 디렉터리를 체크포인트로 잘못 고르는 경우가 있다. `--resume --load_run`일 때 새 디렉터리에 체크포인트가 생기는지 확인한다.
- G11a 지면 평가는 `logs/rsl_rl/biped12_stage4/`에 실행 디렉터리 링크가 있어야 체크포인트를 찾는다.
- 시뮬 외력은 기본이 로컬 프레임이다. 월드 방향으로 밀려면 `is_global=True`를 쓴다.
- 평가 스크립트가 기본으로 본 트리 `log_picture/`에 쓴다. worktree에서 돌릴 때 경로를 확인한다.
- 학습을 동시에 둘 띄우면 OOM이 난다. 순차로 띄운다.

## 9. 핵심 문서 위치

- 전체 기록: `reports/night_1005/LOG.md`(맨 끝 "내부 갱신·9 시간 강화 종료 (14:00)")
- 작업 목록: `reports/night_1005/JOBS.md`
- 실물 계획: `reports/gen_w3/REAL_TEST1_PLAN.md`
- 약점·대비: `reports/gen_w3/CONTINGENCY.md`, `ANOMALY_CARD.md`
- 평가기: `rl_walking/scripts/eval_w3.py`(걷기), `rl_walking/scripts/eval_ground_recovery.py`(서기 회복), `rl_walking/scripts/eval_w3_envelope.py`, `rl_walking/scripts/eval_w3_parity.py`
- 회귀: `reports/regress/run_regress.sh --mode quick|full`
- 메모리: claude1의 메모리는 `~/.claude/projects/-home-ryu-humanoid-leg-test1/memory/`에 있다(claude2에는 자동으로 보이지 않는다). 필요하면 `MEMORY.md` 인덱스를 직접 읽어도 되지만, **수정은 claude1만 한다.**

## 10. 2026-10-08 10:00 상태 — 12 시간 자율 작업(1007 22:00 – 1008 10:00) 뒤

다음 세션은 여기부터 읽는다. 자세한 것은 아래 문서가 정본이다.

- **최종 보고**: `reports/night_1007/FINAL_REPORT.md`(7 항목) · 시간순 기록 `reports/night_1007/LOG.md`
- **최종 후보는 그대로다**: 서기 G21a-11300 · 회복 G24Ra-21793 + IMU 감지기 · 걷기 W3S3d-15800(`models/` 에 역할 이름으로 등록). 밤사이 이어 학습 팔 10 개 중 승격 0.
  실물 이력 주의: 회복 키 `stand_c5kr_…` 은 실물에서 돈 적이 없다. 돌아 본 것은 `stand_c5krh_…` 한 번(1007, 35.6 s)과 1005 의 옛 감지기 키뿐이다.
- **젯슨 = 저장소 판**(묶음 `head_1008`, 원본 커밋 `f1ad8221`, 매니페스트 `8f4373d3…`). 기록 `reports/night_1007/JETSON_DEPLOY_1008.md`.
  새 것: 콘솔 걷기 패널(버튼 6 활성·2 비활성)·모터 오류 바이트 기록·합성 키 2 개(`stand_kew3_…` 갇힘 해소 끔 / `stand_kew3q_…` 켬 = 시뮬 평가 구성).
  재배포는 `deploy/deploy_head_1008.sh`(드라이런 기본, `--apply --restart-console`, 되돌리기 `--rollback --restart-console`). 재적용 전에 `deploy/head_1008.accept` 를 직전 배포 md5 로 맞춘다.
- **하드웨어 P0**(드라이버 재시작): `reports/night_1007/HARDWARE_P0.md` — H0(드라이버 전압 한계 읽기)·H1(허브·발목 전압 파형)이 먼저다. 걷기·충격형 밀기 실물 시험은 그 뒤.
- **실물 계획서**: `reports/gen_w3/REAL_TEST1_PLAN.md` 끝의 `v1.2` 추가분. 운영 명령 `deploy/cand/head_1008/OPERATOR_COMMANDS.md`, 이상 시 카드 `reports/gen_w3/ANOMALY_CARD.md`.
- **다음 세대 재료**: `reports/night_1007/NEXT_GEN_1008.md`. 확실한 효과는 둘 — 토크 한계 무작위화 + 전진 유인(상한 내성, 전환 약화)·무릎 벌점 −4(지지 무릎 64 → 44°). 묶음은 명령 0 에서 뒤로 흐른다.
- **밟은 함정(새로)**:
  1. 평가기 옵션이 배포 러너 객체를 생성 뒤에 바꾸는 경우(`--kw_requiet`)가 있다 — 그 수치는 같은 설정의 배포 키가 있을 때만 배포 구성 수치다(`TRANSITION_W3KE.md` §5.21).
  2. 이어 학습 한 번·체크포인트 하나의 왕복 수치로 처치 효과를 말하지 않는다. 변경 없는 대조 셋이 요 누적 2.0–3.8 배·왕복 낙상 0.29–2.13 % 로 갈렸다(`W3_FINAL_SELECT.md` §15).
  3. 칸당 100 env 로 +3 %p 는 못 가른다(X17 → X17c).
  4. +1000 학습의 중간 저장은 100 단위다(+500 후보 = 16300, 16299 가 아니다).
  5. 학습 전용 태스크는 `rl_walking/tasks_extra_1008.py` + `RL_WALKING_EXTRA_TASKS` 로 등록한다 — `rl_walking/__init__.py` 는 젯슨 묶음에 들어간다.
  6. 젯슨 `--verify` 는 selftest 를 다시 돌려 5 분쯤 걸린다. 짧은 timeout 으로 끊지 않는다.
- 이 세션(claude2)의 메모리: `~/.claude-acc2/projects/-home-ryu-humanoid-leg-test1/memory/`.
