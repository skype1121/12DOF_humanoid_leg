# stand_v16 (h2 재판정·정정 하네스) — 세대 판정 보고 (자동 생성: gen_report.py)


## 판정 — 기본 자산(usd_final3, 패드 돌출) (v16_h2)

**검증 사다리** (`demo_output/ladder_v16_h2.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 0 | ON | 46/64 | -17.90 | 0.000 | 2.62 | ✓ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 0.92 | ✓ |
| 1 | ON | 46/64 | -17.90 | 0.000 | 2.62 | ✗ |
| 2 | OFF | 0/64 | -0.36 | 0.082 | 2.49 | ✓ |
| 2 | ON | 9/64 | -0.68 | 0.376 | 3.20 | ✓ |
| 3 | OFF | 0/64 | -0.33 | 0.148 | 2.93 | ✓ |
| 3 | ON | 17/64 | -0.38 | 0.533 | 4.02 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 23/64 (35.9%) | 뒤16·앞5·왼2·오0 | -1.60 / -0.62 | 2.26° | 0.54198 | left_hip_a 2.89% | 4.8° | 23.9% |
| 구래퍼 com+16 | 24/64 (37.5%) | 뒤11·앞12·왼0·오1 | +0.08 / -1.16 | 2.77° | 0.53959 | right_hip_a 3.09% | 4.1° | 22.6% |
| 신래퍼 com0 | 17/64 (26.6%) | 뒤12·앞2·왼2·오1 | -0.83 / -0.42 | 1.32° | 0.38499 | right_hip_a 2.26% | 6.8° | 21.2% |
| 신래퍼 com+16 | 17/64 (26.6%) | 뒤11·앞3·왼2·오1 | -0.64 / -0.27 | 1.27° | 0.41226 | left_hip_a 2.23% | 6.7° | 22.0% |
| 자기 DR ON | 43/64 (67.2%) | 뒤30·앞12·왼1·오0 | -5.92 / -2.07 | 2.87° | 0.72793 | left_hip_a 4.10% | 3.9° | 30.1% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.68 / -1.69 | 2.39° | 0.00854 | left_hip_f 0.00% | 5.6° | 4.8% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.26 / -0.29 | 0.41° | 0.04253 | left_hip_f 0.00% | 4.9° | 2.8% |
| V6고정 구래퍼 com0 | 18/64 (28.1%) | 뒤13·앞4·왼1·오0 | -2.35 / -0.55 | 2.16° | 0.49523 | right_hip_a 1.80% | 4.3° | 22.0% |
| V6고정 구래퍼 com+16 | 17/64 (26.6%) | 뒤5·앞7·왼4·오1 | -0.02 / -0.96 | 2.12° | 0.48192 | right_hip_a 1.20% | 3.2° | 19.6% |
| V6고정 신래퍼 com+16 | 3/64 (4.7%) | 뒤1·앞0·왼1·오1 | -0.48 / -0.45 | 1.21° | 0.38625 | right_hip_a 1.14% | 5.5° | 18.2% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 7.0 | 25 | 0.0 |
| right_hip_f | 7.3 | 25 | 0.0 |
| left_hip_a | 8.9 | 25 | 0.0 |
| right_hip_a | 6.8 | 25 | 0.0 |
| left_hip_r | 4.2 | 25 | 0.0 |
| right_hip_r | 6.6 | 25 | 0.0 |
| left_knee | 11.6 | 25 | 0.0 |
| right_knee | 11.3 | 25 | 0.0 |
| left_ankle_f | 9.2 | 25 | 0.0 |
| right_ankle_f | 10.3 | 25 | 0.0 |
| left_ankle_r | 4.2 | 24 | 0.0 |
| right_ankle_r | 2.8 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v16_h2_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V16-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v16_h2_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_h2_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v16_h2_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_20-05-14 --checkpoint model_120998.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
