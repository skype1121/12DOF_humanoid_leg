# stand_v15 — 세대 판정 보고 (자동 생성: gen_report.py)

- run: `2026-09-04_16-43-02`
- checkpoint: `model_114999.pt`
- task: `Biped12-Velocity-Stage4V2-Stand-V15-v0`
- 자산: `rl_walking/assets/usd_final3/biped12.usd`

## 판정 — 기본 자산(usd_final3, 패드 돌출) (v15_fix)

**검증 사다리** (`demo_output/ladder_v15_fix.json`)

| 단계 | DR | 낙하 | 추세 °/s | E_smth | 에너지 | 게이트 |
|---|---|---|---|---|---|---|
| 0 | OFF | 0/64 | -0.08 | 0.000 | 1.39 | ✓ |
| 0 | ON | 55/64 | -18.51 | 0.000 | 3.29 | ✓ |
| 1 | OFF | 0/64 | -0.08 | 0.000 | 1.39 | ✓ |
| 1 | ON | 55/64 | -18.51 | 0.000 | 3.29 | ✗ |
| 2 | OFF | 0/64 | -0.38 | 0.374 | 3.76 | ✓ |
| 2 | ON | 7/64 | -1.14 | 0.426 | 4.74 | ✓ |
| 3 | OFF | 0/64 | -0.29 | 0.426 | 4.14 | ✓ |
| 3 | ON | 6/64 | -0.42 | 0.515 | 5.09 | ✓ |

**배포조건·명목·고정환경 배터리** (eval_drift_dr, 64env·8s)

| 조건 | 낙상 | 방향 | 추세 평균/중앙 | 피치RMS | E_smth | τ포화 최대 | 발 요차 | slip>2cm/s |
|---|---|---|---|---|---|---|---|---|
| 구래퍼 com0 | 18/64 (28.1%) | 뒤5·앞8·왼1·오4 | -2.89 / -0.96 | 2.09° | 0.39340 | left_hip_a 2.07% | 18.8° | 21.4% |
| 구래퍼 com+16 | 14/64 (21.9%) | 뒤2·앞2·왼5·오5 | -0.03 / -0.78 | 2.19° | 0.43160 | right_hip_a 3.39% | 17.5° | 23.8% |
| 신래퍼 com0 | 12/64 (18.8%) | 뒤3·앞4·왼2·오3 | -0.05 / -0.34 | 1.60° | 0.37965 | right_hip_a 2.09% | 23.1° | 23.0% |
| 신래퍼 com+16 | 19/64 (29.7%) | 뒤4·앞5·왼3·오7 | +0.32 / -0.37 | 1.35° | 0.36209 | left_hip_a 1.95% | 22.2° | 22.3% |
| 자기 DR ON | 42/64 (65.6%) | 뒤19·앞9·왼9·오5 | -5.83 / -1.76 | 2.81° | 0.52323 | left_hip_a 4.01% | 18.3° | 31.6% |
| 명목 구래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -1.28 / -1.28 | 1.92° | 0.27193 | left_hip_f 0.00% | 23.6° | 9.0% |
| 명목 신래퍼 | 0/64 (0.0%) | 뒤0·앞0·왼0·오0 | -0.35 / -0.32 | 0.61° | 0.25517 | left_hip_f 0.00% | 25.4° | 7.8% |
| V6고정 구래퍼 com0 | 12/64 (18.8%) | 뒤2·앞1·왼4·오5 | -2.25 / -0.67 | 1.93° | 0.38234 | right_hip_a 1.80% | 18.9° | 20.7% |
| V6고정 구래퍼 com+16 | 11/64 (17.2%) | 뒤2·앞3·왼2·오4 | -0.93 / -0.87 | 1.81° | 0.39223 | right_hip_a 2.43% | 17.1° | 21.9% |
| V6고정 신래퍼 com+16 | 7/64 (10.9%) | 뒤1·앞5·왼0·오1 | -0.51 / -0.55 | 1.32° | 0.33547 | left_hip_a 1.61% | 21.6° | 20.4% |

**물리 실현성 (명목, 관절별)** — τ피크/한계 Nm, 포화%

| 관절 | τ피크 | 한계 | 포화% |
|---|---|---|---|
| left_hip_f | 11.3 | 25 | 0.0 |
| right_hip_f | 11.2 | 25 | 0.0 |
| left_hip_a | 14.5 | 25 | 0.0 |
| right_hip_a | 10.0 | 25 | 0.0 |
| left_hip_r | 15.4 | 25 | 0.0 |
| right_hip_r | 6.5 | 25 | 0.0 |
| left_knee | 17.0 | 25 | 0.0 |
| right_knee | 14.7 | 25 | 0.0 |
| left_ankle_f | 16.1 | 25 | 0.0 |
| right_ankle_f | 15.3 | 25 | 0.0 |
| left_ankle_r | 8.9 | 24 | 0.0 |
| right_ankle_r | 9.7 | 24 | 0.0 |

<details><summary>원본 파일·인자</summary>

- `log_picture/드리프트DR_v15_fix_old_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_old_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_new_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_new_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_dr_on.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_nom_old.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_nom_new.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V15-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fi`
- `log_picture/드리프트DR_v15_fix_xold_com0.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v15_fix_xold_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`
- `log_picture/드리프트DR_v15_fix_xnew_com16.txt` — `--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run 2026-09-04_16-43-02 --checkpoint model_114999.pt --num_envs 64 --steps 400 --still --mass_fix`

</details>

---
판정 기준: 사다리 게이트(0·1 강체홀드 / 2 서기 ≤20% / 3 푸시 ≤40%), 배포조건은 트림0 착지 인수(구래퍼 블렌딩·신래퍼 슬루), V6 고정환경은 세대 간 공정 비교, 명목은 전 DR OFF 결정론(물리 실현성). 회귀 금지: E_smth·에너지 15% 이상 악화.
