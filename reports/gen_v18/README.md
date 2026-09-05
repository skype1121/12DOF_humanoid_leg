# stand_v18 — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-05_17-57-21`
- checkpoint: `model_132996.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V18-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v18)

**검증 사다리** (`demo_output/ladder_v18.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 0 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 1 | ON | 46/64 | -17.85 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | -0.03 | 0.013 | 2.41 | ✓ |
| 2 | ON | 12/64 | +0.54 | 0.510 | 3.41 | ✓ |
| 3 | OFF | 0/64 | -0.03 | 0.093 | 2.82 | ✓ |
| 3 | ON | 20/64 | +0.54 | 0.556 | 3.92 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 24/64 (37.5%) | 뒤12·앞4·왼4·오4 | -0.80 / +0.90 | 2.03° | 0.55854 | left_hip_a 2.56% | 3.5° | 20.7% |
| 구래퍼 com+16 | 16/64 (25.0%) | 뒤5·앞7·왼3·오1 | +0.49 / +0.68 | 1.60° | 0.51101 | left_hip_a 2.39% | 2.9° | 17.7% |
| 신래퍼 com0 | 20/64 (31.2%) | 뒤12·앞5·왼2·오1 | -0.23 / -0.04 | 1.31° | 0.49484 | left_hip_a 2.35% | 5.1° | 21.1% |
| 신래퍼 com+16 | 20/64 (31.2%) | 뒤5·앞13·왼0·오2 | +0.10 / +0.08 | 1.29° | 0.47364 | left_hip_a 3.09% | 5.2° | 20.6% |
| 자기 DR ON | 42/64 (65.6%) | 뒤29·앞9·왼2·오2 | -8.03 / -0.10 | 2.03° | 0.70582 | left_hip_a 2.30% | 2.9° | 24.8% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.63 / +0.63 | 0.70° | 0.01316 | left_hip_f 0.00% | 4.2° | 2.1% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.41 / +0.41 | 0.48° | 0.05082 | left_hip_f 0.00% | 4.7° | 2.6% |
| V6고정 구래퍼 com0 | 19/64 (29.7%) | 뒤16·앞2·왼1·오0 | -1.39 / +0.82 | 1.96° | 0.50612 | right_hip_a 1.29% | 5.5° | 18.8% |
| V6고정 구래퍼 com+16 | 12/64 (18.8%) | 뒤7·앞2·왼2·오1 | +0.37 / +0.79 | 1.38° | 0.46941 | left_hip_a 1.44% | 4.0° | 16.3% |
| V6고정 신래퍼 com+16 | 12/64 (18.8%) | 뒤7·앞3·왼1·오1 | +0.20 / +0.22 | 1.06° | 0.41596 | left_hip_a 1.40% | 6.4° | 17.0% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 6.6 | 25 | 0.0 |
| right_hip_f | 5.4 | 25 | 0.0 |
| left_hip_a | 9.2 | 25 | 0.0 |
| right_hip_a | 9.4 | 25 | 0.0 |
| left_hip_r | 4.2 | 25 | 0.0 |
| right_hip_r | 9.9 | 25 | 0.0 |
| left_knee | 10.7 | 25 | 0.0 |
| right_knee | 11.2 | 25 | 0.0 |
| left_ankle_f | 11.3 | 25 | 0.0 |
| right_ankle_f | 11.2 | 25 | 0.0 |
| left_ankle_r | 5.4 | 24 | 0.0 |
| right_ankle_r | 3.4 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v18_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

## 판정 — 플러시 자산(usd_final3f, 패드 눌림) (v18_flush)

**검증 사다리** (`demo_output/ladder_v18_flush.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.94 | ✓ |
| 0 | ON | 48/64 | -18.00 | 0.000 | 2.69 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.94 | ✓ |
| 1 | ON | 48/64 | -18.00 | 0.000 | 2.69 | ✗ |
| 2 | OFF | 0/64 | -0.10 | 0.011 | 2.38 | ✓ |
| 2 | ON | 11/64 | +0.64 | 0.521 | 3.46 | ✓ |
| 3 | OFF | 0/64 | -0.10 | 0.088 | 2.82 | ✓ |
| 3 | ON | 12/64 | +0.92 | 0.583 | 3.91 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 28/64 (43.8%) | 뒤15·앞8·왼1·오4 | -1.05 / +0.71 | 1.99° | 0.56172 | right_hip_a 2.36% | 3.5° | 19.4% |
| 구래퍼 com+16 | 17/64 (26.6%) | 뒤5·앞9·왼3·오0 | +0.48 / +0.68 | 1.63° | 0.51310 | left_hip_a 2.11% | 2.6° | 16.9% |
| 신래퍼 com0 | 24/64 (37.5%) | 뒤7·앞8·왼5·오4 | -0.24 / -0.03 | 1.35° | 0.49159 | left_hip_a 2.47% | 4.8° | 20.8% |
| 신래퍼 com+16 | 20/64 (31.2%) | 뒤6·앞10·왼3·오1 | -0.02 / +0.08 | 1.27° | 0.44950 | right_hip_a 2.30% | 4.9° | 19.0% |
| 자기 DR ON | 45/64 (70.3%) | 뒤29·앞12·왼0·오4 | -8.37 / -0.22 | 2.05° | 0.68575 | left_hip_a 1.97% | 2.7° | 23.4% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.63 / +0.64 | 0.70° | 0.01540 | left_hip_f 0.00% | 3.6° | 0.6% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.42 / +0.42 | 0.50° | 0.05128 | left_hip_f 0.00% | 4.0° | 2.0% |
| V6고정 구래퍼 com0 | 17/64 (26.6%) | 뒤14·앞2·왼0·오1 | -1.36 / +0.74 | 2.00° | 0.51165 | left_hip_a 1.00% | 5.1° | 17.8% |
| V6고정 구래퍼 com+16 | 11/64 (17.2%) | 뒤5·앞6·왼0·오0 | +0.39 / +0.81 | 1.35° | 0.46163 | left_hip_a 1.12% | 3.7° | 14.8% |
| V6고정 신래퍼 com+16 | 14/64 (21.9%) | 뒤6·앞5·왼0·오3 | +0.18 / +0.19 | 1.08° | 0.43683 | right_hip_a 1.57% | 6.2° | 18.0% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 7.1 | 25 | 0.0 |
| right_hip_f | 5.7 | 25 | 0.0 |
| left_hip_a | 9.6 | 25 | 0.0 |
| right_hip_a | 9.1 | 25 | 0.0 |
| left_hip_r | 4.2 | 25 | 0.0 |
| right_hip_r | 4.0 | 25 | 0.0 |
| left_knee | 11.4 | 25 | 0.0 |
| right_knee | 11.8 | 25 | 0.0 |
| left_ankle_f | 18.0 | 25 | 0.0 |
| right_ankle_f | 10.2 | 25 | 0.0 |
| left_ankle_r | 7.1 | 24 | 0.0 |
| right_ankle_r | 4.1 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v18_flush_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V18-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v18_flush_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18_flush_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v18_flush_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-05_17-57-21 --checkpoint model_132996.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
