# stand_v17 — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-04_21-48-32`
- checkpoint: `model_126997.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V17-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v17)

**검증 사다리** (`demo_output/ladder_v17.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 36/64 | -8.98 | 0.000 | 2.02 | ✓ |
| 0 | ON | 57/64 | -22.17 | 0.000 | 3.01 | ✓ |
| 1 | OFF | 36/64 | -8.98 | 0.000 | 2.02 | ✗ |
| 1 | ON | 57/64 | -22.17 | 0.000 | 3.01 | ✗ |
| 2 | OFF | 0/64 | -0.04 | 0.383 | 1.83 | ✓ |
| 2 | ON | 10/64 | +0.26 | 0.471 | 4.17 | ✓ |
| 3 | OFF | 0/64 | -0.00 | 0.454 | 2.53 | ✓ |
| 3 | ON | 16/64 | +0.50 | 0.534 | 4.71 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 37/64 (57.8%) | 뒤7·앞12·왼10·오8 | -4.37 / -0.42 | 3.15° | 0.52242 | right_hip_a 3.94% | 1.7° | 24.3% |
| 구래퍼 com+16 | 15/64 (23.4%) | 뒤4·앞5·왼4·오2 | -0.07 / +0.27 | 1.78° | 0.45948 | left_hip_a 2.41% | 0.7° | 16.1% |
| 신래퍼 com0 | 21/64 (32.8%) | 뒤8·앞3·왼4·오6 | -0.33 / -0.29 | 1.57° | 0.42957 | right_hip_a 2.75% | 3.0° | 19.2% |
| 신래퍼 com+16 | 18/64 (28.1%) | 뒤4·앞3·왼5·오6 | +0.25 / -0.12 | 1.60° | 0.41084 | right_hip_a 2.62% | 2.8° | 17.6% |
| 자기 DR ON | 39/64 (60.9%) | 뒤5·앞15·왼9·오10 | -9.30 / -0.88 | 2.25° | 0.53536 | right_hip_a 3.21% | 1.1° | 23.1% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.26 / -0.24 | 0.99° | 0.35263 | left_hip_f 0.00% | 2.5° | 2.7% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.04 / -0.04 | 0.52° | 0.31468 | left_hip_f 0.00% | 2.7° | 2.8% |
| V6고정 구래퍼 com0 | 21/64 (32.8%) | 뒤5·앞4·왼7·오5 | -3.50 / -0.08 | 2.24° | 0.41413 | right_hip_a 1.29% | 1.7° | 15.7% |
| V6고정 구래퍼 com+16 | 7/64 (10.9%) | 뒤1·앞2·왼1·오3 | -0.65 / +0.17 | 1.59° | 0.38648 | left_hip_a 0.74% | 1.5° | 11.7% |
| V6고정 신래퍼 com+16 | 5/64 (7.8%) | 뒤2·앞0·왼1·오2 | -0.29 / -0.24 | 1.22° | 0.36977 | right_hip_a 0.69% | 2.5° | 13.1% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 8.7 | 25 | 0.0 |
| right_hip_f | 14.1 | 25 | 0.0 |
| left_hip_a | 11.8 | 25 | 0.0 |
| right_hip_a | 10.4 | 25 | 0.0 |
| left_hip_r | 6.9 | 25 | 0.0 |
| right_hip_r | 10.0 | 25 | 0.0 |
| left_knee | 12.7 | 25 | 0.0 |
| right_knee | 16.3 | 25 | 0.0 |
| left_ankle_f | 15.7 | 25 | 0.0 |
| right_ankle_f | 13.8 | 25 | 0.0 |
| left_ankle_r | 5.2 | 24 | 0.0 |
| right_ankle_r | 5.4 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v17_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

## 판정 — 플러시 자산(usd_final3f, 패드 눌림) (v17_flush)

**검증 사다리** (`demo_output/ladder_v17_flush.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 37/64 | -9.00 | 0.000 | 2.10 | ✓ |
| 0 | ON | 57/64 | -22.28 | 0.000 | 3.15 | ✓ |
| 1 | OFF | 37/64 | -9.00 | 0.000 | 2.10 | ✗ |
| 1 | ON | 57/64 | -22.28 | 0.000 | 3.15 | ✗ |
| 2 | OFF | 0/64 | -0.02 | 0.381 | 1.85 | ✓ |
| 2 | ON | 8/64 | +0.16 | 0.478 | 4.19 | ✓ |
| 3 | OFF | 0/64 | +0.01 | 0.455 | 2.55 | ✓ |
| 3 | ON | 16/64 | -0.96 | 0.561 | 4.80 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 36/64 (56.2%) | 뒤9·앞11·왼11·오5 | -4.39 / -0.32 | 3.11° | 0.50116 | right_hip_a 3.62% | 1.7° | 24.4% |
| 구래퍼 com+16 | 17/64 (26.6%) | 뒤6·앞6·왼2·오3 | -0.01 / +0.19 | 1.79° | 0.49068 | right_hip_a 2.59% | 0.6° | 16.5% |
| 신래퍼 com0 | 13/64 (20.3%) | 뒤5·앞5·왼0·오3 | -0.21 / -0.32 | 1.57° | 0.40195 | right_hip_a 2.22% | 3.0° | 16.1% |
| 신래퍼 com+16 | 21/64 (32.8%) | 뒤4·앞6·왼6·오5 | +0.08 / -0.11 | 1.53° | 0.38636 | right_hip_a 2.50% | 2.9° | 16.2% |
| 자기 DR ON | 42/64 (65.6%) | 뒤9·앞12·왼10·오11 | -9.47 / -0.95 | 2.30° | 0.53128 | right_hip_a 2.52% | 0.9° | 20.4% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.26 / -0.24 | 1.00° | 0.35586 | left_hip_f 0.00% | 2.5° | 1.9% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.07 / -0.07 | 0.52° | 0.31541 | left_hip_f 0.00% | 2.9° | 2.5% |
| V6고정 구래퍼 com0 | 18/64 (28.1%) | 뒤3·앞3·왼5·오7 | -3.49 / -0.11 | 2.20° | 0.43482 | right_hip_a 0.94% | 1.8° | 17.2% |
| V6고정 구래퍼 com+16 | 8/64 (12.5%) | 뒤3·앞0·왼2·오3 | -0.66 / +0.16 | 1.58° | 0.40017 | right_hip_a 0.59% | 1.4° | 11.3% |
| V6고정 신래퍼 com+16 | 5/64 (7.8%) | 뒤3·앞0·왼2·오0 | -0.30 / -0.24 | 1.20° | 0.36717 | right_hip_a 0.61% | 2.5° | 12.5% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 7.8 | 25 | 0.0 |
| right_hip_f | 14.3 | 25 | 0.0 |
| left_hip_a | 12.0 | 25 | 0.0 |
| right_hip_a | 10.5 | 25 | 0.0 |
| left_hip_r | 12.7 | 25 | 0.0 |
| right_hip_r | 5.2 | 25 | 0.0 |
| left_knee | 13.1 | 25 | 0.0 |
| right_knee | 16.8 | 25 | 0.0 |
| left_ankle_f | 15.0 | 25 | 0.0 |
| right_ankle_f | 20.2 | 25 | 0.0 |
| left_ankle_r | 6.4 | 24 | 0.0 |
| right_ankle_r | 5.8 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v17_flush_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v17_flush_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_flush_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v17_flush_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_21-48-32 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
