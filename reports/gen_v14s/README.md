# stand_v14s — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-04_05-04-39`
- checkpoint: `model_114999.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V14s-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v14s_fix)

**검증 사다리** (`demo_output/ladder_v14s_fix.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 1.39 | ✓ |
| 0 | ON | 56/64 | -27.35 | 0.000 | 3.79 | ✗ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 1.39 | ✓ |
| 1 | ON | 56/64 | -27.35 | 0.000 | 3.79 | ✗ |
| 2 | OFF | 0/64 | +1.06 | 0.376 | 27.24 | ✓ |
| 2 | ON | 3/64 | +1.18 | 0.462 | 26.24 | ✓ |
| 3 | OFF | 0/64 | +1.05 | 0.448 | 27.86 | ✓ |
| 3 | ON | 10/64 | +1.16 | 0.572 | 28.16 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 19/64 (29.7%) | 뒤4·앞4·왼4·오7 | +0.68 / +3.00 | 2.14° | 0.55534 | left_hip_r 72.36% | 60.0° | 24.8% |
| 구래퍼 com+16 | 26/64 (40.6%) | 뒤6·앞7·왼6·오7 | +3.84 / +3.34 | 2.00° | 0.59665 | left_hip_r 67.78% | 59.9° | 27.8% |
| 신래퍼 com0 | 25/64 (39.1%) | 뒤6·앞9·왼8·오2 | +2.98 / +2.08 | 1.92° | 0.48205 | left_hip_r 82.79% | 60.4° | 25.5% |
| 신래퍼 com+16 | 25/64 (39.1%) | 뒤8·앞6·왼5·오6 | +3.90 / +2.38 | 2.22° | 0.53079 | left_hip_r 80.50% | 60.2° | 27.3% |
| 자기 DR ON | 51/64 (79.7%) | 뒤17·앞7·왼9·오18 | -13.54 / -13.26 | 2.93° | 0.77041 | left_hip_r 52.93% | 59.8° | 35.7% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +2.56 / +2.56 | 1.45° | 0.41697 | left_hip_r 84.49% | 60.2° | 8.7% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | +1.92 / +1.93 | 1.49° | 0.28753 | left_hip_r 96.62% | 60.8° | 8.4% |
| V6고정 구래퍼 com0 | 21/64 (32.8%) | 뒤2·앞7·왼5·오7 | +1.01 / +3.24 | 2.13° | 0.51972 | left_hip_r 71.51% | 60.1° | 24.9% |
| V6고정 구래퍼 com+16 | 17/64 (26.6%) | 뒤4·앞6·왼3·오4 | +3.39 / +3.07 | 1.75° | 0.53608 | left_hip_r 71.38% | 60.0° | 23.2% |
| V6고정 신래퍼 com+16 | 10/64 (15.6%) | 뒤4·앞2·왼3·오1 | +2.30 / +1.92 | 1.86° | 0.42446 | left_hip_r 84.47% | 60.5° | 21.7% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 10.7 | 25 | 0.0 |
| right_hip_f | 8.0 | 25 | 0.0 |
| left_hip_a | 25.0 | 25 | 0.2 |
| right_hip_a | 24.4 | 25 | 0.0 |
| left_hip_r | 25.0 | 25 | 96.6 |
| right_hip_r | 25.0 | 25 | 96.5 |
| left_knee | 16.4 | 25 | 0.0 |
| right_knee | 16.1 | 25 | 0.0 |
| left_ankle_f | 15.5 | 25 | 0.0 |
| right_ankle_f | 13.2 | 25 | 0.0 |
| left_ankle_r | 8.8 | 24 | 0.0 |
| right_ankle_r | 8.5 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v14s_fix_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14s-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_f`
- `log_picture/드리프트DR_v14s_fix_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v14s_fix_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v14s_fix_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_05-04-39 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
