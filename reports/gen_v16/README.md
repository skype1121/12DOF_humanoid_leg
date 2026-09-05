# stand_v16 — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-04_20-05-14`
- checkpoint: `model_120998.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V16-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v16)

**검증 사다리** (`demo_output/ladder_v16.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.08 | 0.000 | 1.39 | ✓ |
| 0 | ON | 55/64 | -18.32 | 0.000 | 3.25 | ✓ |
| 1 | OFF | 0/64 | -0.08 | 0.000 | 1.39 | ✓ |
| 1 | ON | 55/64 | -18.32 | 0.000 | 3.25 | ✗ |
| 2 | OFF | 0/64 | -0.30 | 0.437 | 3.18 | ✓ |
| 2 | ON | 8/64 | -0.93 | 0.497 | 4.50 | ✓ |
| 3 | OFF | 0/64 | -0.22 | 0.490 | 3.55 | ✓ |
| 3 | ON | 15/64 | +0.04 | 0.577 | 5.39 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 28/64 (43.8%) | 뒤2·앞12·왼7·오7 | -3.15 / -0.90 | 2.31° | 0.43334 | right_hip_a 3.23% | 4.7° | 19.8% |
| 구래퍼 com+16 | 23/64 (35.9%) | 뒤3·앞10·왼5·오5 | +0.23 / -1.01 | 1.84° | 0.47989 | right_hip_a 4.10% | 3.4° | 19.9% |
| 신래퍼 com0 | 20/64 (31.2%) | 뒤4·앞7·왼5·오4 | -0.53 / -0.28 | 1.41° | 0.37930 | right_hip_a 2.74% | 6.0° | 17.5% |
| 신래퍼 com+16 | 15/64 (23.4%) | 뒤5·앞3·왼5·오2 | -0.35 / -0.16 | 1.28° | 0.37172 | left_hip_a 2.41% | 5.5° | 17.2% |
| 자기 DR ON | 46/64 (71.9%) | 뒤13·앞12·왼7·오14 | -6.82 / -2.25 | 2.86° | 0.53547 | left_hip_a 4.47% | 3.9° | 29.1% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.53 / -1.53 | 2.31° | 0.33012 | left_hip_f 0.00% | 7.0° | 6.3% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.29 / -0.30 | 0.55° | 0.31761 | right_ankle_f 0.00% | 7.4° | 4.8% |
| V6고정 구래퍼 com0 | 17/64 (26.6%) | 뒤3·앞3·왼3·오8 | -3.31 / -0.63 | 2.12° | 0.39672 | right_hip_a 1.23% | 3.8° | 18.0% |
| V6고정 구래퍼 com+16 | 17/64 (26.6%) | 뒤8·앞4·왼1·오4 | -0.79 / -0.89 | 1.81° | 0.40738 | right_hip_a 1.51% | 2.7° | 16.5% |
| V6고정 신래퍼 com+16 | 10/64 (15.6%) | 뒤2·앞3·왼5·오0 | -0.45 / -0.56 | 1.33° | 0.36840 | left_hip_a 0.69% | 5.1° | 16.5% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 14.4 | 25 | 0.0 |
| right_hip_f | 12.7 | 25 | 0.0 |
| left_hip_a | 12.2 | 25 | 0.0 |
| right_hip_a | 9.8 | 25 | 0.0 |
| left_hip_r | 6.3 | 25 | 0.0 |
| right_hip_r | 7.9 | 25 | 0.0 |
| left_knee | 17.1 | 25 | 0.0 |
| right_knee | 15.6 | 25 | 0.0 |
| left_ankle_f | 16.3 | 25 | 0.0 |
| right_ankle_f | 24.6 | 25 | 0.0 |
| left_ankle_r | 7.9 | 24 | 0.0 |
| right_ankle_r | 6.8 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v16_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

## 판정 — 플러시 자산(usd_final3f, 패드 눌림) (v16_flush)

**검증 사다리** (`demo_output/ladder_v16_flush.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.09 | 0.000 | 1.40 | ✓ |
| 0 | ON | 55/64 | -18.34 | 0.000 | 3.32 | ✓ |
| 1 | OFF | 0/64 | -0.09 | 0.000 | 1.40 | ✓ |
| 1 | ON | 55/64 | -18.34 | 0.000 | 3.32 | ✗ |
| 2 | OFF | 0/64 | -0.30 | 0.438 | 3.19 | ✓ |
| 2 | ON | 9/64 | -0.89 | 0.468 | 4.51 | ✓ |
| 3 | OFF | 0/64 | -0.23 | 0.500 | 3.56 | ✓ |
| 3 | ON | 13/64 | -0.31 | 0.583 | 5.08 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 26/64 (40.6%) | 뒤7·앞6·왼9·오4 | -3.10 / -0.90 | 2.27° | 0.43306 | right_hip_a 3.81% | 4.4° | 18.1% |
| 구래퍼 com+16 | 28/64 (43.8%) | 뒤9·앞8·왼7·오4 | +0.07 / -1.01 | 1.88° | 0.46154 | left_hip_a 3.57% | 3.3° | 19.1% |
| 신래퍼 com0 | 19/64 (29.7%) | 뒤8·앞3·왼5·오3 | -0.36 / -0.34 | 1.39° | 0.39744 | right_hip_a 2.98% | 6.0° | 17.3% |
| 신래퍼 com+16 | 17/64 (26.6%) | 뒤2·앞6·왼5·오4 | -0.42 / -0.17 | 1.24° | 0.36832 | right_hip_a 2.10% | 5.6° | 16.0% |
| 자기 DR ON | 46/64 (71.9%) | 뒤13·앞16·왼11·오6 | -6.76 / -2.42 | 2.95° | 0.55560 | right_hip_a 5.13% | 3.8° | 29.8% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.53 / -1.51 | 2.31° | 0.33245 | left_hip_f 0.00% | 6.7° | 5.5% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.31 / -0.31 | 0.55° | 0.31702 | left_hip_f 0.00% | 7.2° | 4.2% |
| V6고정 구래퍼 com0 | 17/64 (26.6%) | 뒤2·앞3·왼3·오9 | -3.35 / -0.57 | 2.06° | 0.40861 | right_hip_a 1.02% | 4.0° | 18.1% |
| V6고정 구래퍼 com+16 | 13/64 (20.3%) | 뒤4·앞3·왼2·오4 | -0.83 / -0.87 | 1.77° | 0.44458 | right_hip_a 1.46% | 2.7° | 18.1% |
| V6고정 신래퍼 com+16 | 1/64 (1.6%) | 뒤1·앞0·왼0·오0 | -0.52 / -0.51 | 1.30° | 0.37527 | left_hip_a 1.01% | 4.7° | 15.9% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 14.4 | 25 | 0.0 |
| right_hip_f | 12.9 | 25 | 0.0 |
| left_hip_a | 12.2 | 25 | 0.0 |
| right_hip_a | 10.2 | 25 | 0.0 |
| left_hip_r | 9.0 | 25 | 0.0 |
| right_hip_r | 7.1 | 25 | 0.0 |
| left_knee | 17.7 | 25 | 0.0 |
| right_knee | 16.2 | 25 | 0.0 |
| left_ankle_f | 17.2 | 25 | 0.0 |
| right_ankle_f | 16.9 | 25 | 0.0 |
| left_ankle_r | 8.2 | 24 | 0.0 |
| right_ankle_r | 8.3 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v16_flush_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_flush_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_flush_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_flush_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
