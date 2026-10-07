# W3 ↔ KE 전환 검증 — 사전 등록 v1 (2026-10-07 15:40, claude2)

> 승윤님 1007 15:00 지시 1·2: "블록보다 화살표". W3 WALK → stop → gait 정리/정착 → KE 시작 → NORMAL STAND 를 실제 배포 경로와 최대한 같게 시뮬에서 재고,
> STOP 경유가 직접 전환·다른 방식보다 나은지 데이터로 판정한다. 검증되지 않은 W3 → STOP → KE 를 기본 동작으로 가정하지 않는다.
> 시뮬 판정이며 실물 성공 확률이 아니다. 젯슨·실물 없음. 결과 보고 뒤 아래 기준은 바꾸지 않는다(바꾸면 v2 로 새로 등록).

## 0. 질문
- **Q1 무해·성립**: W3S3d-15800 걷기 → stop 요청 → 반주기 경계 ge 0 → 정착 → KE(G21a + G24Ra 회복 전환, 현행 imu 종료 판정) 시작 → 정상 서기 복귀가
  무외란·지연 2 에서 낙상·과도·감지기 오탐 없이 성립하는가.
- **Q2 방식 비교**: 배포 가능한 **restart**(스택 재기동 = 목표 동결 → 새 러너 → 온셋 블렌딩)와 가상의 hot-swap(**direct** 상태 이어받기 / **reset** 러너 리셋), 정지 없이 걷는 중 **nostop** 직접 전환,
  대조 **none**(W3 STAND 유지) 가운데 무엇이 낙상·과도에서 나은가.
- **Q3 전환 뒤 외란 내성**: 전환 뒤 KE 가 홀드→KE 기준선(sab_imu, 같은 하네스)과 같은 급인가, W3 STAND 유지(none)보다 나은가(STAND_MODE_CMP 의 '권장'이 걷기 뒤에도 성립하는가).
- **Q4(탐색)**: 걷는 중 외란(앞 12 N) 직후 stop → KE 가 '계속 걷기'·'stop 뒤 W3 STAND' 보다 나쁜가(기존 결과: 밀린 뒤 stop 이 더 위험).

## 1. 하네스 (eval_ground_recovery.py 1007 사본 `rl_walking/scripts/eval_transition.py` — 기존 파일 수정 없음)
- 환경 `Biped12-HangStand-G11a-v0`, 128 env, 홀드 사전검사 3 s → 인계(러너 슬루 4°/틱 → LPF β0.27 → 블렌딩 1.5 s, 출발 = 홀드 명령각) → `--after_s` 20 s.
  G21a 배포 계약 `--enc_obs v2 --contact_real --contact_ver v2 --last_action_ema 0.27 --obs_ema joint_vel:0.25`(W3 걷기 구간에도 같은 계약 — W3 배포 러너와 같은 LPF β0.27·EMA; W3 학습은 jvel EMA 없음이지만 STAND_MODE_CMP 와 같은 조건), 학습 DR 켬, 롤 램프 켬(학습 리셋 그대로), 분모 = 사전검사 통과 env.
- W3: `install_w3_cmd`(stage 0, p_step 0) + `w3_gait_update`(f 1.1, lead_vy 0.02, lead_wz 0.08 = W3S3d 학습 설정) + `w3_step_tracker` 를 G11a env 에 설치(STAND_MODE_CMP c2 어댑터와 같은 방식, 단 supervisor 없음), command term heading_command 끔·리샘플 없음.
  정책 관측은 W3S3d 학습과 같은 320 차원(클록 = W3 위상 × ge 램프, 명령 = 평가기가 틱마다 설정).
- KE: 주 정책 G21a-11300 + `--switch_to` G24Ra-21793, 감지기·조기 복귀 = kabs(`--sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33`).
  감지기는 KE 시작 뒤에만 무장, 기준 자세·필터는 KE 시작 시각에 초기화(배포 = KE 브리지 기동 시 감지기 새로 생성). KE 클록 관측 = KE 시작 틱부터 1.1 Hz 회전(배포 러너 tick 0 과 같음).
- 시간축(인계 기준): W3 STAND 0–2 s → **ge 켬 2.0 s**(명령 `--w3_cmd` 기본 앞 0.06) → **stop 요청 t_stop = 6.0 + (env mod 8)/8 × 0.909 s**(한 주기를 8 위상으로 고르게; 같은 env 번호 = 같은 위상)
  → ge 0 (반주기 경계, t_geoff 기록) → 정착 판정 = 양발 접촉 ∧ 감지 |ω_xy| < 0.15 가 0.5 s 유지(IMU·접촉 비트만, 참 속도 불사용), 상한 3 s(초과 시 시각·플래그 기록 후 진행) → t_settle.
- 전환 팔 `--ke_mode`:
  | 팔 | 뜻 | 상태 처리 |
  |---|---|---|
  | **none** | W3 STAND(ge 0) 유지 | 전환 없음(대조) |
  | **restart** | 배포 경로: t_settle 에 명령 동결(stage8 홀드) **3 s**(`--hold_s`, 재기동 시간 가정) → t_sw 에 KE 시작 | 러너 리셋(관측 이력 5 틀 = 현재 프레임 백필, last_action 0, 자이로·관절속도 EMA·obs_ema·act EMA 현재값 재시작), 체인 리셋(슬루 기준 = 실측 qpos, LPF 재초기화, 블렌딩 1.5 s 출발 = 동결 명령각) |
  | **reset** | hot-swap + 러너 리셋 | t_sw = t_settle, 러너 리셋만(체인 슬루·LPF 연속, 블렌딩 없음) |
  | **direct** | hot-swap 상태 이어받기 | t_sw = t_settle, 아무것도 리셋 안 함(이력·last_action·EMA 그대로), 클록만 KE 회전 |
  | **nostop** | 정지 없이 걷는 중 전환 | stop 요청 없이 t_stop(위상 8 종)에 ge 강제 0 + reset 팔과 같은 KE 시작 |
- 외란(`--dist`, 인계 기준 시각 고정 — 모든 팔 같은 시각, 전환 뒤 경과는 팔마다 달라 함께 기록): nd(없음) · **b18** `push:b:16:18:0.3` · **sf15** `ramp:f:16:15:0.5:1.0`. 16 s 는 restart 팔 최대 t_sw(≈ 6.9 + 0.455 + 3 + 3 ≈ 13.4 s) + 감지기 기준 추종 1.5 s 보다 뒤(실행 전 14 → 16 s 로 정정, 15:55). `--after_s 22`.
  Q4: `push:f:5:12:0.3` + stop 요청 5.3 s(위상 분산 없음, `--w3_stop 5.3 --stop_phases 1`) × {restart, none, walk(정지 없이 계속 걷기)}.
- seed: 지연 2 주 격자 seed 1–3, 지연 3·4 seed 1–2. 지연 = `--delay_steps d,d`. 같은 (seed, env) 짝 = 같은 초기조건·DR 표본(팔은 전환 뒤에만 다름 — RNG 지문 bias/com 으로 확인, 깨지면 비짝 Fisher).


## 1b. KE → W3 START (지시 2) — 사전 등록 v1 (1007 16:05, 실행 전)
- 같은 하네스·같은 평가기(`--dir ke2w3`). 인계 → **KE**(G21a + 감지기, 홀드→KE 경로 = 기존 sab 하네스) 4 s 서기 → t_sw0 = 4.0 s 에 W3 로 전환 → W3 STAND(ge 0) → 켜기 선행조건
  (배포 W3Supervisor 와 같은 정의: 양발 접촉(관측) 1.0 s ∧ 기울기 ≤ 7° ∧ |ω_xy|(관측) < 0.25 가 1.0 s) 성립 즉시 `step enable`(명령 앞 0.06) → 5 s 걷기 → stop 요청 → 서기(끝까지).
- 전환 팔(`--ke_mode` 의 KE→W3 판): **restart**(명령 동결 3 s → W3 새 러너: 이력 백필·last_action 0·EMA 재시작, 클록 (0,0), 체인 리셋 블렌딩 1.5 s) ·
  **reset**(W3 러너 리셋만) · **direct**(상태 이어받기 — W3_BRIDGE_FLAG §5 `carry_runner_state` 의미: 이력·last_action·EMA 그대로, 클록 이력만 (0,0)) · **none**(KE 유지, 창 취약성 대조).
- 조건: nd · **b18w** `push:b:4.7:18:0.3`(전환 0.7 s 뒤 = W3 STAND 창 안; restart 는 동결 뒤 W3 시작 0.7 s 뒤가 되도록 `push:b:7.7`) · **sf15w** 같은 시각 램프. seed 1–3, 지연 2(지연 3·4 는 restart·none seed 1–2).
- 지표: 전환 뒤 2 s/5 s 낙상·피크·불연속(위와 같음) · 선행조건 성립까지 시간(W3 시작 기준, 3 s 안 비율) · 켜기 뒤 첫 유효 이륙까지 시간·왼발 리드 비율·교대 · 켜기 뒤 5 s 낙상 · 창 외란 낙상(W3 STAND vs KE).
- 판정(사전): **K1 무해** nd 전환 뒤 5 s 낙상 ≤ 1 % ∧ 선행조건 3 s 안 ≥ 95 % ∧ 켜기 뒤 1 s 안 첫 이륙 ≥ 95 %(왼발) ∧ 켜기 뒤 5 s 낙상 ≤ 3 %. **K2 과도** = R2 와 같음.
  **K3 창**: b18w 에서 W3 STAND 창 낙상률과 KE(none) 낙상률 보고 — 차이가 +5 %p 이상이면 "창 동안 밀기 금지" 규칙을 실물 절차에 명시(이미 가정)·창 길이(선행조건 시간) 중앙 보고.
  **K4 방식** = R4 와 같음(restart 기준). 통과 = K1·K2.

## 2. 지표(env 별 CSV + 전환 ±(0.5 s 전, 2 s 뒤) 틱 추적 npz)
- 시각: t_stop·t_geoff·t_settle(타임아웃 플래그)·t_sw·전환 시 W3 위상 u·KE 시작 뒤 외란까지 경과.
- 낙상: 걷기 중 / 창(ge 0 ~ t_sw) / 전환 뒤 2 s / 5 s / 외란 뒤(기존 d_fell) · 낙상 시각(인계·전환 기준).
- 스파이크(참값, 전환 뒤 2 s 창·창 구간 따로): 최대 |피치|·|롤|·|ω_xy|.
- 불연속: 전환 틱 ±10 틱 안 |Δ raw action| 최대(관절 최대, 액션 단위), |Δ 관절 목표(체인 뒤)| 최대 [deg/틱], KE 첫 출력 목표 − 직전 목표 [deg] 최대.
- 상태 영향: direct vs reset vs restart 같은 env 짝 비교로 이력·last_action·필터 상태 효과를 분리.
- KE 감지기 오탐: KE 시작 뒤 외란 전 회복 켜짐 env 비율·첫 켜짐 시각(nd 에서는 전체).
- 정상 자세 복귀 시간: t_sw 뒤 '정상 자세'(|피치|·|롤| < 3°·|ω_xy| < 0.2·양발·명목 관절 평균 편차 < 5°·높이 ≥ −2 cm, 0.5 s 유지) 첫 시각 — 중앙·p90·5 s 안 비율.
- 걷기 확인: stop 전 이륙 수(걸었는지), 정착 성공률·시간.

## 3. 판정 규칙(사전 고정)
- **R1 restart 무해(Q1)**: nd·지연 2·seed 1–3 합(≈ 300 env): 전환 뒤 5 s 안 낙상 ≤ 1 %(Wilson 상한 < 3 %) ∧ KE 감지기 오탐(외란 없는데 켜짐) ≤ 5 %(홀드→KE nd 오탐 3.2 % + 여유) ∧ 정상 자세 복귀 중앙 ≤ 3 s·5 s 안 ≥ 90 %.
- **R2 과도(Q1)**: 전환 뒤 2 s 피크 |피치|·|롤| 중앙 ≤ none 팔 같은 창 + 2°, 피크 |ω_xy| 중앙 ≤ 0.5·p95 ≤ 1.0 rad/s. 관절 목표 틱당 변화는 슬루(4°/틱)로 묶여 있어 보고만.
- **R3 외란 내성(Q3)**: b18·sf15(14 s)·지연 2·seed 1–3: restart 낙상 ≤ 홀드→KE 기준선(sab_imu seed 1–6, 지연 2: b18 5.9 %·sf15 18.3 %) + 5 %p ∧ none 보다 ≥ 5 %p 낮고 Fisher p < 0.05(b18 기준; sf15 는 방향 일치).
- **R4 방식 비교(Q2)**: 같은 seed·env 짝에서 direct·reset 이 restart 보다 전환 뒤 5 s 낙상(nd+b18+sf15 합)이 적고 차이 ≥ 3 env·McNemar p < 0.05 → "hot-swap 이득 있음(별도 개발 가치)"; 아니면 restart 유지. direct 가 restart 보다 ≥ 3 %p 많으면 "상태 이어받기 위험" 기록.
- **R5 STOP 경유(Q2)**: nostop 전환 뒤 5 s 낙상 ≥ restart + 5 %p(p < 0.05) → "STOP 경유 유리" 확인. 반대면 보고(STOP 이 손해).
- **R6 지연**: 지연 3·4 에서 R1·R3 재보고(퇴보 폭). 정착 상한 3 s 초과 비율 보고.
- **Q4**: 탐색 — 낙상률 3 팔 보고, 결론 없음(n 작음). 기존 '밀린 뒤 stop 이 더 위험' 결과와 방향 비교만.
- 통과 = R1·R2·R3 전부. 하나라도 실패 → 첫 실물 구성에서 'W3 STOP → KE' 를 기본 동작으로 두지 않고(지시 그대로) 실패 원인 분해(상태·과도·감지기·블렌딩)를 다음 단계로.

## 4. 실행
`eval_transition.py --ke_mode {none,restart,reset,direct,nostop} --w3_cmd 0.06,0,0 --w3_on 2 --w3_stop 6 --stop_phases 8 --settle quiet --hold_s 3 --dist … --delay_steps d,d --seed s --out_tag tr_<mode>_<cond>_d<d>_s<s>`.
산출: `log_picture/ground_recovery_tr_*.txt/_envs.csv`, `log_picture/trans_tr_*.npz`. 요약 `reports/gen_w3/trans_summ.py`.

## 5. 결과
(실행 뒤 추가 — 위 규칙은 수정하지 않는다)
