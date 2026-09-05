# stand_v20 — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v20)

**검증 사다리** (`demo_output/ladder_v20.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 0 | ON | 33/64 | -13.56 | 0.000 | 2.85 | ✓ |
| 1 | OFF | 0/64 | -0.03 | 0.000 | 0.93 | ✓ |
| 1 | ON | 33/64 | -13.56 | 0.000 | 2.85 | ✗ |
| 2 | OFF | 0/64 | -1.16 | 0.008 | 1.13 | ✓ |
| 2 | ON | 13/64 | -1.16 | 0.355 | 3.59 | ✗ |
| 3 | OFF | 1/64 | -1.16 | 0.055 | 1.91 | ✓ |
| 3 | ON | 14/64 | +0.96 | 0.415 | 4.31 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 21/64 (32.8%) | 뒤17·앞4·왼0·오0 | -2.01 / -0.13 | 1.89° | 0.43322 | right_hip_a 3.19% | 10.2° | 27.9% |
| 구래퍼 com+16 | 21/64 (32.8%) | 뒤4·앞13·왼2·오2 | -0.05 / -0.05 | 1.59° | 0.39263 | left_hip_a 2.83% | 9.9° | 27.5% |
| 신래퍼 com0 | 20/64 (31.2%) | 뒤6·앞10·왼1·오3 | -0.54 / -0.64 | 1.58° | 0.31487 | left_hip_a 2.99% | 13.1° | 23.9% |
| 신래퍼 com+16 | 16/64 (25.0%) | 뒤3·앞12·왼0·오1 | -0.30 / -0.51 | 1.47° | 0.32478 | left_hip_a 2.11% | 12.7° | 24.7% |
| 자기 DR ON | 37/64 (57.8%) | 뒤26·앞9·왼0·오2 | -7.84 / -0.84 | 2.08° | 0.48290 | right_hip_a 3.33% | 9.9° | 28.0% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.16 / -1.15 | 0.77° | 0.00399 | left_hip_f 0.00% | 12.4° | 8.8% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.40 / -1.40 | 0.50° | 0.00609 | left_hip_f 0.00% | 15.5° | 7.5% |
| V6고정 구래퍼 com0 | 42/64 (65.6%) | 뒤41·앞1·왼0·오0 | -10.93 / -12.70 | 2.70° | 0.57687 | left_hip_a 0.78% | 11.1° | 21.9% |
| V6고정 구래퍼 com+16 | 16/64 (25.0%) | 뒤5·앞7·왼0·오4 | -1.94 / -0.59 | 2.04° | 0.38245 | right_hip_a 1.02% | 10.4° | 15.5% |
| V6고정 신래퍼 com+16 | 13/64 (20.3%) | 뒤1·앞7·왼1·오4 | -0.92 / -0.87 | 1.02° | 0.26675 | right_hip_a 0.84% | 12.3° | 15.8% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 5.3 | 25 | 0.0 |
| right_hip_f | 5.3 | 25 | 0.0 |
| left_hip_a | 2.1 | 25 | 0.0 |
| right_hip_a | 3.4 | 25 | 0.0 |
| left_hip_r | 5.1 | 25 | 0.0 |
| right_hip_r | 4.6 | 25 | 0.0 |
| left_knee | 7.8 | 25 | 0.0 |
| right_knee | 8.9 | 25 | 0.0 |
| left_ankle_f | 8.5 | 25 | 0.0 |
| right_ankle_f | 9.7 | 25 | 0.0 |
| left_ankle_r | 3.6 | 24 | 0.0 |
| right_ankle_r | 3.9 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v20_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V20-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v20_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v20_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v20_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-06_07-02-39 --checkpoint model_126997.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
