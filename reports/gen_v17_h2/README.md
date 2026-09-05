# stand_v17 (h2 재판정·정정 하네스) — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v17_h2)

**검증 사다리** (`demo_output/ladder_v17_h2.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 0 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 1 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | -0.18 | 0.048 | 1.02 | ✓ |
| 2 | ON | 13/64 | -0.36 | 0.483 | 3.11 | ✗ |
| 3 | OFF | 0/64 | -0.18 | 0.191 | 1.91 | ✓ |
| 3 | ON | 19/64 | +0.80 | 0.521 | 3.54 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 26/64 (40.6%) | 뒤19·앞6·왼0·오1 | -1.88 / +0.25 | 2.24° | 0.58500 | right_hip_a 2.34% | 1.4° | 20.7% |
| 구래퍼 com+16 | 20/64 (31.2%) | 뒤9·앞9·왼1·오1 | -0.01 / +0.17 | 1.64° | 0.47735 | right_hip_a 2.38% | 1.4° | 18.1% |
| 신래퍼 com0 | 23/64 (35.9%) | 뒤15·앞5·왼1·오2 | -1.06 / -0.38 | 1.51° | 0.44790 | right_hip_a 2.30% | 2.9° | 20.2% |
| 신래퍼 com+16 | 17/64 (26.6%) | 뒤11·앞5·왼0·오1 | -0.32 / -0.24 | 1.60° | 0.43213 | right_hip_a 2.39% | 2.9° | 20.6% |
| 자기 DR ON | 41/64 (64.1%) | 뒤29·앞12·왼0·오0 | -8.50 / -1.29 | 2.33° | 0.72640 | right_hip_a 2.56% | 1.4° | 23.3% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.43 / -0.42 | 1.07° | 0.00955 | left_hip_f 0.00% | 3.3° | 1.8% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.01 / +0.01 | 0.24° | 0.03040 | left_hip_f 0.00% | 5.1° | 0.9% |
| V6고정 구래퍼 com0 | 19/64 (29.7%) | 뒤19·앞0·왼0·오0 | -3.31 / -0.08 | 2.01° | 0.54675 | left_hip_a 1.16% | 1.7° | 19.0% |
| V6고정 구래퍼 com+16 | 7/64 (10.9%) | 뒤6·앞0·왼0·오1 | -0.44 / +0.00 | 1.56° | 0.42287 | right_hip_a 0.76% | 1.6° | 13.8% |
| V6고정 신래퍼 com+16 | 8/64 (12.5%) | 뒤3·앞5·왼0·오0 | -0.31 / -0.34 | 1.16° | 0.37621 | right_hip_a 0.86% | 2.5° | 14.9% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 4.6 | 25 | 0.0 |
| right_hip_f | 6.0 | 25 | 0.0 |
| left_hip_a | 6.4 | 25 | 0.0 |
| right_hip_a | 5.5 | 25 | 0.0 |
| left_hip_r | 8.4 | 25 | 0.0 |
| right_hip_r | 3.1 | 25 | 0.0 |
| left_knee | 8.2 | 25 | 0.0 |
| right_knee | 10.3 | 25 | 0.0 |
| left_ankle_f | 8.8 | 25 | 0.0 |
| right_ankle_f | 8.7 | 25 | 0.0 |
| left_ankle_r | 5.0 | 24 | 0.0 |
| right_ankle_r | 2.4 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v17_h2_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_h2_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_h2_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_h2_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
