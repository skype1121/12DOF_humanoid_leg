# stand_v18c — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v18c)

**검증 사다리** (`demo_output/ladder_v18c.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 0 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 1 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | -0.13 | 0.067 | 1.46 | ✓ |
| 2 | ON | 14/64 | -0.10 | 0.518 | 3.26 | ✗ |
| 3 | OFF | 3/64 | -0.00 | 0.193 | 2.52 | ✓ |
| 3 | ON | 18/64 | +0.25 | 0.549 | 3.63 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 38/64 (59.4%) | 뒤26·앞5·왼6·오1 | -3.24 / -0.04 | 2.83° | 0.69876 | right_hip_a 4.14% | 6.1° | 28.8% |
| 구래퍼 com+16 | 27/64 (42.2%) | 뒤7·앞14·왼2·오4 | -0.13 / +0.12 | 1.87° | 0.56105 | left_hip_a 2.88% | 5.0° | 20.7% |
| 신래퍼 com0 | 21/64 (32.8%) | 뒤6·앞7·왼2·오6 | -0.65 / -0.59 | 1.45° | 0.45166 | right_hip_a 3.17% | 7.8° | 20.5% |
| 신래퍼 com+16 | 21/64 (32.8%) | 뒤4·앞11·왼1·오5 | -0.41 / -0.47 | 1.33° | 0.45991 | right_hip_a 2.56% | 7.8° | 20.7% |
| 자기 DR ON | 45/64 (70.3%) | 뒤32·앞9·왼3·오1 | -10.48 / -1.86 | 2.45° | 0.75112 | left_hip_a 3.96% | 4.5° | 26.6% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.41 / +0.42 | 0.87° | 0.02441 | left_hip_f 0.00% | 8.6° | 14.9% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.08 / -0.08 | 0.71° | 0.03126 | left_hip_f 0.00% | 9.6° | 16.6% |
| V6고정 구래퍼 com0 | 31/64 (48.4%) | 뒤27·앞1·왼0·오3 | -6.14 / -0.65 | 2.95° | 0.72963 | right_hip_a 1.73% | 7.5° | 24.1% |
| V6고정 구래퍼 com+16 | 14/64 (21.9%) | 뒤8·앞1·왼1·오4 | -0.68 / +0.31 | 1.60° | 0.55010 | left_hip_a 1.31% | 5.4° | 16.8% |
| V6고정 신래퍼 com+16 | 7/64 (10.9%) | 뒤1·앞4·왼0·오2 | -0.52 / -0.36 | 1.03° | 0.41667 | right_hip_a 0.54% | 7.9° | 17.2% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 5.4 | 25 | 0.0 |
| right_hip_f | 6.8 | 25 | 0.0 |
| left_hip_a | 2.8 | 25 | 0.0 |
| right_hip_a | 8.8 | 25 | 0.0 |
| left_hip_r | 6.1 | 25 | 0.0 |
| right_hip_r | 6.2 | 25 | 0.0 |
| left_knee | 9.1 | 25 | 0.0 |
| right_knee | 11.5 | 25 | 0.0 |
| left_ankle_f | 8.5 | 25 | 0.0 |
| right_ankle_f | 11.1 | 25 | 0.0 |
| left_ankle_r | 5.4 | 24 | 0.0 |
| right_ankle_r | 6.0 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v18c_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18c_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18c_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18c_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_03-38-51 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
