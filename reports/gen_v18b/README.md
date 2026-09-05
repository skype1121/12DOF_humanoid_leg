# stand_v18b — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-05_19-37-39`
- checkpoint: `model_120998.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V18-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v18b)

**검증 사다리** (`demo_output/ladder_v18b.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 0 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 1 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | +0.09 | 0.054 | 2.51 | ✓ |
| 2 | ON | 15/64 | -0.28 | 0.486 | 3.49 | ✗ |
| 3 | OFF | 1/64 | +0.08 | 0.236 | 3.00 | ✓ |
| 3 | ON | 15/64 | +0.66 | 0.541 | 3.94 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 32/64 (50.0%) | 뒤18·앞10·왼2·오2 | -1.79 / +0.04 | 2.13° | 0.54651 | right_hip_a 2.98% | 2.3° | 22.5% |
| 구래퍼 com+16 | 13/64 (20.3%) | 뒤2·앞11·왼0·오0 | -0.19 / +0.29 | 1.88° | 0.43356 | right_hip_a 1.70% | 1.8° | 17.2% |
| 신래퍼 com0 | 16/64 (25.0%) | 뒤10·앞5·왼0·오1 | -0.99 / -0.46 | 1.39° | 0.44060 | right_hip_a 1.79% | 4.0° | 18.8% |
| 신래퍼 com+16 | 16/64 (25.0%) | 뒤6·앞7·왼2·오1 | -0.24 / -0.16 | 1.41° | 0.42574 | left_hip_a 2.59% | 4.1° | 21.2% |
| 자기 DR ON | 43/64 (67.2%) | 뒤27·앞13·왼2·오1 | -8.80 / -0.55 | 2.50° | 0.61915 | right_hip_a 3.43% | 1.8° | 26.0% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.55 / +0.56 | 0.89° | 0.00811 | left_hip_f 0.00% | 3.1° | 1.9% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.42 / +0.42 | 0.36° | 0.02922 | left_hip_f 0.00% | 3.9° | 2.4% |
| V6고정 구래퍼 com0 | 20/64 (31.2%) | 뒤16·앞4·왼0·오0 | -2.46 / +0.22 | 2.07° | 0.49726 | right_hip_a 1.47% | 3.2° | 18.9% |
| V6고정 구래퍼 com+16 | 9/64 (14.1%) | 뒤5·앞2·왼1·오1 | -0.16 / +0.32 | 1.53° | 0.40742 | left_hip_a 1.03% | 2.4° | 14.3% |
| V6고정 신래퍼 com+16 | 5/64 (7.8%) | 뒤3·앞0·왼1·오1 | -0.13 / -0.01 | 1.08° | 0.38936 | left_hip_a 0.81% | 4.1° | 16.4% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 6.7 | 25 | 0.0 |
| right_hip_f | 4.5 | 25 | 0.0 |
| left_hip_a | 6.0 | 25 | 0.0 |
| right_hip_a | 6.9 | 25 | 0.0 |
| left_hip_r | 3.9 | 25 | 0.0 |
| right_hip_r | 3.8 | 25 | 0.0 |
| left_knee | 10.8 | 25 | 0.0 |
| right_knee | 9.3 | 25 | 0.0 |
| left_ankle_f | 10.7 | 25 | 0.0 |
| right_ankle_f | 8.7 | 25 | 0.0 |
| left_ankle_r | 4.8 | 24 | 0.0 |
| right_ankle_r | 5.8 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v18b_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18b_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18b_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

## 판정 — 플러시 자산(usd_final3f, 패드 눌림) (v18b_flush)

**검증 사다리** (`demo_output/ladder_v18b_flush.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.94 | ✓ |
| 0 | ON | 48/64 | -18.00 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.94 | ✓ |
| 1 | ON | 48/64 | -18.00 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | +0.19 | 0.136 | 2.20 | ✓ |
| 2 | ON | 10/64 | +0.81 | 0.488 | 3.47 | ✓ |
| 3 | OFF | 0/64 | +0.20 | 0.308 | 2.74 | ✓ |
| 3 | ON | 18/64 | -0.24 | 0.530 | 3.89 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 30/64 (46.9%) | 뒤21·앞6·왼0·오3 | -1.82 / +0.08 | 2.22° | 0.54998 | right_hip_a 3.39% | 2.1° | 21.2% |
| 구래퍼 com+16 | 18/64 (28.1%) | 뒤5·앞11·왼0·오2 | -0.17 / +0.33 | 1.91° | 0.44684 | right_hip_a 2.40% | 1.7° | 17.0% |
| 신래퍼 com0 | 17/64 (26.6%) | 뒤10·앞4·왼0·오3 | -0.95 / -0.61 | 1.42° | 0.44277 | right_hip_a 2.31% | 3.9° | 20.0% |
| 신래퍼 com+16 | 22/64 (34.4%) | 뒤7·앞11·왼3·오1 | -0.29 / -0.17 | 1.37° | 0.42514 | right_hip_a 2.76% | 4.0° | 20.0% |
| 자기 DR ON | 44/64 (68.8%) | 뒤30·앞10·왼1·오3 | -9.16 / -0.80 | 2.49° | 0.61033 | right_hip_a 3.26% | 1.7° | 23.0% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.56 / +0.56 | 0.89° | 0.01143 | left_hip_f 0.00% | 2.6° | 1.2% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.45 / +0.45 | 0.38° | 0.03084 | left_hip_f 0.00% | 3.7° | 2.3% |
| V6고정 구래퍼 com0 | 21/64 (32.8%) | 뒤18·앞3·왼0·오0 | -2.53 / +0.11 | 2.10° | 0.52141 | right_hip_a 1.40% | 3.0° | 18.0% |
| V6고정 구래퍼 com+16 | 7/64 (10.9%) | 뒤5·앞0·왼0·오2 | -0.16 / +0.30 | 1.52° | 0.42844 | right_hip_a 1.24% | 2.2° | 15.7% |
| V6고정 신래퍼 com+16 | 9/64 (14.1%) | 뒤4·앞5·왼0·오0 | -0.12 / +0.00 | 1.06° | 0.37393 | left_hip_a 0.73% | 3.8° | 14.5% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 7.0 | 25 | 0.0 |
| right_hip_f | 5.5 | 25 | 0.0 |
| left_hip_a | 5.8 | 25 | 0.0 |
| right_hip_a | 7.2 | 25 | 0.0 |
| left_hip_r | 4.3 | 25 | 0.0 |
| right_hip_r | 4.8 | 25 | 0.0 |
| left_knee | 11.6 | 25 | 0.0 |
| right_knee | 10.1 | 25 | 0.0 |
| left_ankle_f | 17.0 | 25 | 0.0 |
| right_ankle_f | 10.3 | 25 | 0.0 |
| left_ankle_r | 6.8 | 24 | 0.0 |
| right_ankle_r | 7.3 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v18b_flush_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18b_flush_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18b_flush_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18b_flush_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
