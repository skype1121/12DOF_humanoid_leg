# stand_v15 (h2 재판정·정정 하네스) — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v15_h2)

**검증 사다리** (`demo_output/ladder_v15_h2.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 0 | ON | 46/64 | -18.12 | 0.000 | 2.65 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 1 | ON | 46/64 | -18.12 | 0.000 | 2.65 | ✗ |
| 2 | OFF | 0/64 | -0.44 | 0.127 | 2.80 | ✓ |
| 2 | ON | 5/64 | -0.10 | 0.387 | 3.74 | ✓ |
| 3 | OFF | 0/64 | -0.43 | 0.191 | 3.24 | ✓ |
| 3 | ON | 13/64 | +0.38 | 0.511 | 4.34 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 15/64 (23.4%) | 뒤11·앞3·왼1·오0 | -1.29 / -1.12 | 2.19° | 0.43366 | right_hip_a 1.31% | 18.7° | 22.3% |
| 구래퍼 com+16 | 17/64 (26.6%) | 뒤7·앞9·왼1·오0 | +0.42 / -1.10 | 2.53° | 0.47163 | left_hip_a 2.03% | 18.1° | 25.8% |
| 신래퍼 com0 | 7/64 (10.9%) | 뒤4·앞2·왼1·오0 | -0.51 / -0.33 | 1.49° | 0.33847 | left_hip_a 1.21% | 23.7° | 21.5% |
| 신래퍼 com+16 | 4/64 (6.2%) | 뒤2·앞2·왼0·오0 | -0.40 / -0.35 | 1.28° | 0.34182 | left_hip_a 1.22% | 23.1° | 22.2% |
| 자기 DR ON | 38/64 (59.4%) | 뒤28·앞9·왼0·오1 | -4.98 / -1.56 | 2.76° | 0.68634 | left_hip_a 2.47% | 18.8° | 30.3% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.44 / -1.45 | 1.93° | 0.00806 | left_hip_f 0.00% | 23.1° | 6.9% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.34 / -0.31 | 0.45° | 0.04722 | left_hip_f 0.00% | 23.8° | 5.4% |
| V6고정 구래퍼 com0 | 10/64 (15.6%) | 뒤7·앞1·왼0·오2 | -2.17 / -0.79 | 1.91° | 0.42780 | right_hip_a 1.36% | 18.4° | 22.0% |
| V6고정 구래퍼 com+16 | 12/64 (18.8%) | 뒤3·앞6·왼1·오2 | -0.45 / -0.96 | 2.00° | 0.43877 | right_hip_a 2.36% | 17.5° | 23.0% |
| V6고정 신래퍼 com+16 | 5/64 (7.8%) | 뒤1·앞2·왼1·오1 | -0.46 / -0.49 | 1.33° | 0.32571 | right_hip_a 1.56% | 21.8° | 20.7% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 7.4 | 25 | 0.0 |
| right_hip_f | 5.8 | 25 | 0.0 |
| left_hip_a | 13.7 | 25 | 0.0 |
| right_hip_a | 8.7 | 25 | 0.0 |
| left_hip_r | 16.1 | 25 | 0.0 |
| right_hip_r | 5.3 | 25 | 0.0 |
| left_knee | 12.0 | 25 | 0.0 |
| right_knee | 11.7 | 25 | 0.0 |
| left_ankle_f | 13.8 | 25 | 0.0 |
| right_ankle_f | 13.1 | 25 | 0.0 |
| left_ankle_r | 8.5 | 24 | 0.0 |
| right_ankle_r | 5.4 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v15_h2_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_h2_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v15_h2_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v15_h2_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
