# stand_v19 — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v19)

**검증 사다리** (`demo_output/ladder_v19.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 0 | ON | 46/64 | -18.10 | 0.000 | 2.62 | ✓ |
| 1 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 1 | ON | 46/64 | -18.10 | 0.000 | 2.62 | ✗ |
| 2 | OFF | 0/64 | +0.62 | 0.090 | 1.58 | ✓ |
| 2 | ON | 20/64 | +0.46 | 0.520 | 3.55 | ✗ |
| 3 | OFF | 1/64 | +0.60 | 0.259 | 2.93 | ✓ |
| 3 | ON | 24/64 | +0.82 | 0.549 | 3.82 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 50/64 (78.1%) | 뒤42·앞4·왼2·오2 | -5.61 / -2.82 | 3.64° | 0.76506 | right_hip_a 4.63% | 16.4° | 38.1% |
| 구래퍼 com+16 | 29/64 (45.3%) | 뒤17·앞8·왼1·오3 | +0.19 / +0.64 | 1.89° | 0.56671 | right_hip_a 2.68% | 14.6° | 26.1% |
| 신래퍼 com0 | 35/64 (54.7%) | 뒤15·앞12·왼2·오6 | -1.24 / -0.75 | 1.66° | 0.48426 | right_hip_a 3.33% | 18.2° | 28.3% |
| 신래퍼 com+16 | 36/64 (56.2%) | 뒤14·앞17·왼2·오3 | -0.85 / -0.62 | 1.54° | 0.48271 | left_hip_a 2.46% | 18.3° | 28.2% |
| 자기 DR ON | 50/64 (78.1%) | 뒤37·앞11·왼1·오1 | -10.40 / -4.97 | 2.47° | 0.71964 | left_hip_a 2.91% | 14.8° | 31.4% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.61 / +0.60 | 0.85° | 0.02359 | left_hip_f 0.00% | 21.2° | 18.3% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +0.13 / +0.16 | 0.74° | 0.03648 | left_hip_f 0.00% | 21.5° | 17.6% |
| V6고정 구래퍼 com0 | 38/64 (59.4%) | 뒤33·앞0·왼1·오4 | -5.79 / -1.15 | 3.33° | 0.82442 | right_hip_a 2.60% | 18.4° | 28.8% |
| V6고정 구래퍼 com+16 | 12/64 (18.8%) | 뒤9·앞0·왼2·오1 | -0.41 / +0.69 | 1.48° | 0.55386 | left_hip_a 1.12% | 16.5° | 19.2% |
| V6고정 신래퍼 com+16 | 10/64 (15.6%) | 뒤0·앞4·왼4·오2 | -0.37 / -0.28 | 0.96° | 0.40974 | right_hip_a 0.96% | 18.8° | 18.6% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 6.5 | 25 | 0.0 |
| right_hip_f | 6.6 | 25 | 0.0 |
| left_hip_a | 3.1 | 25 | 0.0 |
| right_hip_a | 5.0 | 25 | 0.0 |
| left_hip_r | 11.0 | 25 | 0.0 |
| right_hip_r | 15.2 | 25 | 0.0 |
| left_knee | 12.0 | 25 | 0.0 |
| right_knee | 9.2 | 25 | 0.0 |
| left_ankle_f | 16.2 | 25 | 0.0 |
| right_ankle_f | 13.7 | 25 | 0.0 |
| left_ankle_r | 5.0 | 24 | 0.0 |
| right_ankle_r | 7.2 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v19_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V19-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v19_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v19_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v19_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_05-22-40 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
