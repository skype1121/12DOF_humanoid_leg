# 고정 회귀 검사 결과 — f1008_0611 (full, 프로필 s3d)

- 종합: **PASS** · PASS 6 · 미실행 91
- W3: `Biped12-HangStand-W3S3d-v0` `2026-10-07_00-11-12_w3S3d_from_s2d:model_15800.pt` · 서기 `2026-10-02_21-23-08_g21a:model_11300.pt` · 회복 `2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt`
- 추가 인자: w3 `` · stand `--after_s 35` · ke ``
- 코드 `d393d771e5` (hang2stand-prep-0911) · 작업 트리 변경 2 건 · 체크포인트 md5 w3 3da9f2e5 · stand 599c47d3 · rec 05e6b8ee
- 시작 2026-10-08 06:11:22 · 끝 2026-10-08 06:24:02 · 문턱 `THRESHOLDS.md` md5 2463c5de · 이전 실행 비교: f1007_0852_d2base_s3d15800
- 시뮬 판정이다. 실물 성공 확률이 아니다. '경계' = 점추정은 문턱 밖이나 95 % 구간이 문턱을 걸침(표본 부족 → 전체 모드로 재확인).

## STAND (G21a 단독, 지연 2·접촉 v2)
| ID | 항목 | 판정 | 문턱 | 값 | 기준선(출처) | 이전 실행 대비 |
|---|---|---|---|---|---|---|
| S-nd.pre | 홀드 사전검사 통과(하네스 건강) | PASS | 비율≥ 0.70 | 306/384 = 79.7 % [75–83] | 81–82 % (sab_* nd 같은 홀드) | 79.7 % → 79.7 % (Δ +0.0 %p) |
| S-nd.fall | 무외란 서기 낙상 | PASS | 비율≤ 0.02 | 0/306 = 0.0 % [0–1] | 0/306 (rg_f1007_0852 s1–3, 지연 2·v2) | 0.0 % → 0.0 % (Δ +0.0 %p) |
| S-nd.quiet | 무외란 최종 정숙 | PASS | 비율≥ 0.90 | 290/306 = 94.8 % [92–97] | 295/306 = 96.4 % (같은 출처) | 96.4 % → 94.8 % (Δ -1.6 %p) |
| S-nd.pose | 무외란 끝 정상 자세(수평 <3°·명목 관절 <5°·양발·abn ≤ 0.1) | PASS | 비율≥ 0.90 | 296/306 = 96.7 % [94–98] | 295/306 = 96.4 % (rg_f1007_0852) | 96.4 % → 96.7 % (Δ +0.3 %p) |
| S-b18.fall | 뒤 18 N 밀기 낙상(서기 정책만, 회복 전환 없음) | PASS | 비율≤ 0.28 | 70/307 = 22.8 % [18–28] | 70/307 = 22.8 % (rg_f1007_0852 s1–3, 지연 2·v2) | 22.8 % → 22.8 % (Δ +0.0 %p) |
| S-b18.pose | 뒤 18 N 뒤 정상 자세 복귀 | PASS | 비율≥ 0.64 | 224/307 = 73.0 % [68–78] | 216/307 = 70.4 % (rg_f1007_0852) | 70.4 % → 73.0 % (Δ +2.6 %p) |

## W3 STAND 모드·WALK·TURN·HEADING·DELAY·STOP·걷기 중 외란 (eval_w3)
| ID | 항목 | 판정 | 문턱 | 값 | 기준선(출처) | 이전 실행 대비 |
|---|---|---|---|---|---|---|
| W-st.warm | STAND 모드 시작 과도(스폰·착지 워밍업 2 s) 낙상 | 미실행 | 비율≤ 0.13 |  | 41/512 = 8.0 % (seed 1 9.8 %) |  |
| W-st.fall | STAND 모드 채점 구간 낙상 | 미실행 | 비율≤ 0.02 |  | 0/471 |  |
| W-st.steps | STAND 모드 유효 스텝 /env/s(걸으면 안 됨) | 미실행 | ≤ 0.05 |  | 0.000 |  |
| W-st.prms | STAND 모드 피치 RMS °(정지 떨림) | 미실행 | ≤ 0.6 |  | 0.285 |  |
| W-st.vx | STAND 모드 앞 표류 m/s | 미실행 | \|x\|≤ 0.01 |  | +0.0001 |  |
| W-z.fall | 제자리(명령 0) 채점 낙상 | 미실행 | 비율≤ 0.03 |  | 1/237 (s3d_15800_c0) · 0/241 (s3dC_c0_s2) |  |
| W-z.warm | 시작 과도(워밍업 2 s) 낙상 | 미실행 | 비율≤ 0.12 |  | 19/256 = 7.4 % (s3d_15800_c0) |  |
| W-z.vx | 제자리 앞 표류 m/s | 미실행 | \|x\|≤ 0.02 |  | +0.000 (s3d_15800_c0) |  |
| W-z.vy | 제자리 옆 표류 m/s | 미실행 | \|x\|≤ 0.02 |  | +0.004 |  |
| W-z.alt | 좌우 교대율 | 미실행 | ≥ 0.97 |  | 0.997 |  |
| W-z.cad | 유효 cadence Hz(보행 주기) | 미실행 | 구간 0.95 1.15 |  | 1.079 |  |
| W-f.vx | 앞 0.06: 실제 앞 속도 m/s | 미실행 | 구간 0.045 0.085 |  | +0.058 (s3d_15800_f06) · 고리 120 s +0.077 |  |
| W-f.fall | 앞 0.06 채점 낙상 | 미실행 | 비율≤ 0.04 |  | 4/237 = 1.7 % |  |
| W-b.vx | 뒤 −0.04: 실제 앞 속도 | 미실행 | 구간 -0.055 -0.025 |  | −0.045 (s3dC_b04_s2) |  |
| W-b.fall | 뒤 −0.04 채점 낙상 | 미실행 | 비율≤ 0.03 |  | 0/241 |  |
| W-l.vy | 왼 0.05: 실제 옆 속도 | 미실행 | 구간 0.035 0.075 |  | +0.053 (s3d_15800_l05) |  |
| W-l.fall | 왼 0.05 채점 낙상 | 미실행 | 비율≤ 0.05 |  | 6/237 = 2.5 % |  |
| W-r.vy | 오른 −0.05: 실제 옆 속도 | 미실행 | 구간 -0.075 -0.035 |  | −0.047 (s3dC_r05_s2) |  |
| W-r.fall | 오른 −0.05 채점 낙상 | 미실행 | 비율≤ 0.04 |  | 2/243 |  |
| W-tl.yg | 좌회전 +0.2: 요 이득 | 미실행 | 구간 0.6 1.3 |  | 0.875 (s3d_15800_tl02) |  |
| W-tl.fall | 좌회전 채점 낙상 | 미실행 | 비율≤ 0.04 |  | 2/237 |  |
| W-tr.yg | 우회전 −0.2: 요 이득 | 미실행 | 구간 0.6 1.3 |  | 0.839 (s3d_15800_tr02) |  |
| W-tr.fall | 우회전 채점 낙상 | 미실행 | 비율≤ 0.05 |  | 6/232 = 2.6 % |  |
| W-mix.gx | 혼합(0.05, 0.04, 0.2): vx 이득(본체) | 미실행 | 구간 0.8 1.35 |  | 1.17 (s4d_16299_all) |  |
| W-mix.gy | 혼합: vy 이득 | 미실행 | ≥ 0.5 |  | 0.70 (s4d_16299_all, seed 2 0.70) |  |
| W-mix.yg | 혼합: 요 이득 | 미실행 | 구간 0.7 1.3 |  | 1.01 |  |
| W-mix.fall | 혼합 채점 낙상 | 미실행 | 비율≤ 0.06 |  | 5/242 = 2.1 % · seed 2 11/237 = 4.6 % |  |
| W-hl.yaw | 헤딩 고리 120 s(명령 0, 램프 끔) 요 표류 ° | 미실행 | ≤ 5.0 |  | 2.2° (s3dC_hl_long) · S4d 1.4° |  |
| W-hl.fall | 헤딩 고리 120 s 채점 낙상 | 미실행 | 비율≤ 0.02 |  | 0/237 |  |
| W-hl.vx | 헤딩 고리 120 s 앞 표류 m/s | 미실행 | \|x\|≤ 0.03 |  | +0.004 · S4d +0.023 |  |
| W-hl.vy | 헤딩 고리 120 s 옆 표류 m/s | 미실행 | \|x\|≤ 0.02 |  | +0.003 |  |
| W-d4.fall | 지연 4 · 앞 0.06 채점 낙상 | 미실행 | 비율≤ 0.07 |  | 9/235 = 3.8 % (s3d_15800_d4f06) |  |
| W-d4.vx | 지연 4 · 앞 0.06 실제 앞 속도 | 미실행 | ≥ 0.035 |  | +0.046 |  |
| W-d4hl.yaw | 지연 4 · 헤딩 고리 30 s 요 표류 ° | 미실행 | ≤ 8.0 |  | 4.1° (s3dC_hl_d4) |  |
| W-d4hl.fall | 지연 4 · 헤딩 고리 채점 낙상 | 미실행 | 비율≤ 0.03 |  | 2/234 |  |
| W-stop.settle | 앞 0.06 걷다 정지 → 정착(양발·\|ω\|<0.15·\|v\|<0.05 0.5 s) | 미실행 | 비율≥ 0.93 |  | 97.9 % (s3dC_stop_f06) |  |
| W-stop.foot | 정지 뒤 발 중점 이동 중앙 cm | 미실행 | ≤ 5.5 |  | 3.7 |  |
| W-stop.sfall | 정지 요청 뒤 낙상 | 미실행 | 비율≤ 0.03 |  | 2/237 |  |
| W-dist.f0.fall | 대조(힘 0, 앞 걷기) 낙상 | 미실행 | 비율≤ 0.03 |  | 0/120 |  |
| W-dist.l0.fall | 대조(힘 0, 옆 걷기) 낙상 | 미실행 | 비율≤ 0.03 |  | 0/118 |  |
| W-dist.f12.fall | 앞 걷기 중 앞 12 N 0.3 s 낙상 | 미실행 | 비율≤ 0.22 |  | 18/124 = 14.5 % |  |
| W-dist.f12.rec | 앞 12 N 뒤 정상 WALK 복귀(≤ 6 s) | 미실행 | 비율≥ 0.78 |  | 106/124 = 85.5 % |  |
| W-dist.b18.fall | 앞 걷기 중 뒤 18 N 0.3 s 낙상 | 미실행 | 비율≤ 0.28 |  | 24/117 = 20.5 % |  |
| W-dist.b18.rec | 뒤 18 N 뒤 정상 WALK 복귀 | 미실행 | 비율≥ 0.72 |  | 93/117 = 79.5 % |  |
| W-dist.l30.fall | 왼 걷기 중 왼 30 N 0.2 s 낙상 | 미실행 | 비율≤ 0.29 |  | 26/119 = 21.8 % |  |
| W-dist.l30.rec | 왼 30 N 뒤 정상 WALK 복귀 | 미실행 | 비율≥ 0.71 |  | 93/119 = 78.2 % |  |

## RECOVERY·SETTLE·POSTURE RESTORE (KE 체인, eval_ground_recovery)
| ID | 항목 | 판정 | 문턱 | 값 | 기준선(출처) | 이전 실행 대비 |
|---|---|---|---|---|---|---|
| K-pair.fall | odicw vs imu 짝 낙상(외란 3 조건 합) | 미실행 | 짝 0 |  | odicw만 24 / imu만 52, p 0.0018 (6 seed) · seed 1: 3 / 11 |  |
| K-ref-odicw.fall | odicw: 기준선 sab_odicw 와 같은 env 짝 낙상 | 미실행 | 짝 0 |  | 0 / 0(같은 코드면 비트 재현) |  |
| K-ref-imu.fall | imu: 기준선 sab_imu 와 같은 env 짝 낙상 | 미실행 | 짝 0 |  | 0 / 0 |  |
| K-odicw-nd.fall | KE odicw nd: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.01 |  | 0/629 = 0.0 % (sab_odicw) |  |
| K-odicw-nd.chain | KE odicw nd: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.90 |  | 95.5 % (sab_odicw) |  |
| K-odicw-nd.pose | KE odicw nd: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.90 |  | 95.5 % (sab_odicw) |  |
| K-odicw-nd.quiet | KE odicw nd: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.90 |  | 96.5 % (sab_odicw) |  |
| K-odicw-nd.false | KE odicw nd: 무외란 회복 모드 오탐 env(sw_on_n>0) | 미실행 | 비율≤ 0.08 |  | 3.2 % (sab_odicw) |  |
| K-odicw-nd.long | KE odicw nd: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.07 |  | 2.4 % (sab_odicw) |  |
| K-odicw-sf15.fall | KE odicw sf15: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.23 |  | 111/630 = 17.6 % (sab_odicw) |  |
| K-odicw-sf15.chain | KE odicw sf15: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.30 |  | 36.2 % (sab_odicw) |  |
| K-odicw-sf15.pose | KE odicw sf15: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.46 |  | 52.1 % (sab_odicw) |  |
| K-odicw-sf15.quiet | KE odicw sf15: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.67 |  | 73.3 % (sab_odicw) |  |
| K-odicw-sf15.long | KE odicw sf15: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.07 |  | 2.5 % (sab_odicw) |  |
| K-odicw-fi40.fall | KE odicw fi40: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.31 |  | 159/630 = 25.2 % (sab_odicw) |  |
| K-odicw-fi40.chain | KE odicw fi40: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.29 |  | 35.2 % (sab_odicw) |  |
| K-odicw-fi40.pose | KE odicw fi40: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.51 |  | 59.5 % (sab_odicw) |  |
| K-odicw-fi40.quiet | KE odicw fi40: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.59 |  | 67.6 % (sab_odicw) |  |
| K-odicw-fi40.long | KE odicw fi40: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.08 |  | 4.0 % (sab_odicw) |  |
| K-odicw-b18.fall | KE odicw b18: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.08 |  | 24/630 = 3.8 % (sab_odicw) |  |
| K-odicw-b18.chain | KE odicw b18: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.71 |  | 78.3 % (sab_odicw) |  |
| K-odicw-b18.pose | KE odicw b18: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.72 |  | 79.5 % (sab_odicw) |  |
| K-odicw-b18.quiet | KE odicw b18: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.76 |  | 84.3 % (sab_odicw) |  |
| K-odicw-b18.long | KE odicw b18: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.08 |  | 3.5 % (sab_odicw) |  |
| K-imu-nd.fall | KE imu nd: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.01 |  | 0/629 = 0.0 % (sab_imu) |  |
| K-imu-nd.chain | KE imu nd: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.90 |  | 95.9 % (sab_imu) |  |
| K-imu-nd.pose | KE imu nd: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.90 |  | 95.9 % (sab_imu) |  |
| K-imu-nd.quiet | KE imu nd: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.90 |  | 96.3 % (sab_imu) |  |
| K-imu-nd.false | KE imu nd: 무외란 회복 모드 오탐 env(sw_on_n>0) | 미실행 | 비율≤ 0.08 |  | 3.5 % (sab_imu) |  |
| K-imu-nd.long | KE imu nd: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.07 |  | 2.4 % (sab_imu) |  |
| K-imu-sf15.fall | KE imu sf15: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.23 |  | 115/630 = 18.3 % (sab_imu) |  |
| K-imu-sf15.chain | KE imu sf15: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.30 |  | 36.5 % (sab_imu) |  |
| K-imu-sf15.pose | KE imu sf15: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.46 |  | 52.9 % (sab_imu) |  |
| K-imu-sf15.quiet | KE imu sf15: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.67 |  | 72.9 % (sab_imu) |  |
| K-imu-sf15.long | KE imu sf15: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.07 |  | 1.9 % (sab_imu) |  |
| K-imu-fi40.fall | KE imu fi40: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.31 |  | 170/630 = 27.0 % (sab_imu) |  |
| K-imu-fi40.chain | KE imu fi40: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.29 |  | 35.1 % (sab_imu) |  |
| K-imu-fi40.pose | KE imu fi40: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.51 |  | 57.1 % (sab_imu) |  |
| K-imu-fi40.quiet | KE imu fi40: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.59 |  | 64.9 % (sab_imu) |  |
| K-imu-fi40.long | KE imu fi40: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.08 |  | 3.7 % (sab_imu) |  |
| K-imu-b18.fall | KE imu b18: 낙상(사전검사 통과 env) | 미실행 | 비율≤ 0.10 |  | 37/630 = 5.9 % (sab_imu) |  |
| K-imu-b18.chain | KE imu b18: 회복 체인 성공(①스텝 ②억제 ③양발 ④정상 자세 ⑤STAND 복귀) | 미실행 | 비율≥ 0.71 |  | 76.8 % (sab_imu) |  |
| K-imu-b18.pose | KE imu b18: 정상 자세 복귀(POSTURE RESTORE) | 미실행 | 비율≥ 0.72 |  | 77.9 % (sab_imu) |  |
| K-imu-b18.quiet | KE imu b18: 최종 정숙(마지막 1 s, SETTLE) | 미실행 | 비율≥ 0.76 |  | 80.8 % (sab_imu) |  |
| K-imu-b18.long | KE imu b18: 회복 모드 > 4 s 장기 체류(SETTLE) | 미실행 | 비율≤ 0.08 |  | 4.0 % (sab_imu) |  |

## 이전 실행과 env 짝(같은 seed·RNG 지문) — 서기·KE
- S-nd: 짝 306 · 이번에만 낙상 0 / 이전에만 0 · McNemar p 1
- S-b18: 짝 307 · 이번에만 낙상 0 / 이전에만 0 · McNemar p 1

## 실행 유효성
| 작업 | 상태 | 시도 | 시간 s | 비고 |
|---|---|---|---|---|
| S-nd-s1 | 유효 | 1 | 165 | 유효 |
| S-b18-s1 | 유효 | 1 | 111 | 유효 |
| S-nd-s2 | 유효 | 1 | 111 | 유효 |
| S-b18-s2 | 유효 | 1 | 111 | 유효 |
| S-nd-s3 | 유효 | 1 | 111 | 유효 |
| S-b18-s3 | 유효 | 1 | 111 | 유효 |
