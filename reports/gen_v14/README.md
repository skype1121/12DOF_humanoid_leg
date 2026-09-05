# stand_v14 — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-04_02-06-07`
- checkpoint: `model_107000.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V14-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

명목 평가는 별도 태그(v14_nomfix): hip_r 토크 포화 95%, 발 요차 60° (오리발 하드스톱 악용).

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v14)

**검증 사다리** (`demo_output/ladder_v14.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.11 | 0.000 | 1.39 | ✓ |
| 0 | ON | 56/64 | -27.35 | 0.000 | 3.79 | ✗ |
| 1 | OFF | 0/64 | -0.11 | 0.000 | 1.39 | ✓ |
| 1 | ON | 56/64 | -27.35 | 0.000 | 3.79 | ✗ |
| 2 | OFF | 0/64 | +0.89 | 0.551 | 26.82 | ✓ |
| 2 | ON | 10/64 | -1.85 | 0.433 | 18.85 | ✓ |
| 3 | OFF | 0/64 | +0.78 | 0.613 | 27.54 | ✓ |
| 3 | ON | 14/64 | -1.45 | 0.583 | 20.23 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 29/64 (45.3%) | 뒤5·앞6·왼10·오8 | -1.52 / +0.12 | 1.90° | 0.49062 |  | ° | % |
| 구래퍼 com+16 | 36/64 (56.2%) | 뒤1·앞12·왼16·오7 | +1.10 / +0.04 | 2.38° | 0.57812 |  | ° | % |
| 신래퍼 com0 | 29/64 (45.3%) | 뒤9·앞8·왼7·오5 | -0.34 / -0.06 | 1.48° | 0.44869 |  | ° | % |
| 신래퍼 com+16 | 15/64 (23.4%) | 뒤4·앞3·왼1·오7 | +0.53 / +0.69 | 1.44° | 0.40901 |  | ° | % |
| 자기 DR ON | 57/64 (89.1%) | 뒤20·앞23·왼5·오9 | -14.41 / -12.25 | 2.54° | 0.63690 |  | ° | % |
| 명목 구래퍼 | — | | | | | | | |
| 명목 신래퍼 | — | | | | | | | |
| V6고정 구래퍼 com0 | 26/64 (40.6%) | 뒤8·앞4·왼7·오7 | -1.48 / +0.15 | 1.80° | 0.46116 |  | ° | % |
| V6고정 구래퍼 com+16 | 19/64 (29.7%) | 뒤5·앞2·왼3·오9 | +0.58 / +0.08 | 1.87° | 0.51888 |  | ° | % |
| V6고정 신래퍼 com+16 | 14/64 (21.9%) | 뒤5·앞4·왼4·오1 | +0.81 / +0.67 | 1.38° | 0.38105 |  | ° | % |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v14_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v14_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v14_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v14_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v14_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V14-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v14_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v14_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v14_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_02-06-07 --checkpoint model_107000.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

## 판정 — 플러시 자산(usd_final3f, 패드 눌림) (v14_flush)

**검증 사다리** (`demo_output/ladder_v14_flush.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.12 | 0.000 | 1.40 | ✓ |
| 0 | ON | 56/64 | -27.43 | 0.000 | 3.85 | ✗ |
| 1 | OFF | 0/64 | -0.12 | 0.000 | 1.40 | ✓ |
| 1 | ON | 56/64 | -27.43 | 0.000 | 3.85 | ✗ |
| 2 | OFF | 0/64 | +0.86 | 0.578 | 26.85 | ✓ |
| 2 | ON | 7/64 | -1.41 | 0.437 | 18.99 | ✓ |
| 3 | OFF | 0/64 | +0.71 | 0.629 | 27.54 | ✓ |
| 3 | ON | 10/64 | -1.81 | 0.564 | 20.34 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | — | | | | | | | |
| 구래퍼 com+16 | — | | | | | | | |
| 신래퍼 com0 | — | | | | | | | |
| 신래퍼 com+16 | — | | | | | | | |
| 자기 DR ON | — | | | | | | | |
| 명목 구래퍼 | — | | | | | | | |
| 명목 신래퍼 | — | | | | | | | |
| V6고정 구래퍼 com0 | — | | | | | | | |
| V6고정 구래퍼 com+16 | — | | | | | | | |
| V6고정 신래퍼 com+16 | — | | | | | | | |

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
