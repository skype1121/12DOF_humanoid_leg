# 시뮬 ↔ 젯슨 배포 계약 감사 (2026-10-07 밤)

> 자동 생성: `python3 reports/night_1007/contract_audit.py --report <파일> --run-existing --run-mutants` · 기준 커밋 d7715923 · 2026-10-08 00:22. 숫자는 전부 그 실행의 산출이다(손으로 옮긴 값 없음).

지시: "시뮬과 Jetson 배포 코드의 observation/action/filter/history/previous-action/phase/supervisor 설정이 동일한지도 확인하세요."

- 대상 브리지 키: `stand_crouch_g21a` · `stand_c5kr_g21a_g24ra` · `stand_c5krh_g21a_g24ra` · `stand_w3s3d_15800` · `stand_kew3_g21a_g24ra_w3s3d15800`
- 학습 쪽 = 각 run 의 `params/env.yaml`·`agent.yaml`·체크포인트·학습 로그 + 학습 시점 커밋의 시뮬 소스(읽기·실행). 감지기·감독기·합성 러너처럼 **학습에 없는 층**은 그 층을 검증한 시뮬 평가의 값(평가 산출물 머리에 찍힌 인자)을 '학습 값' 칸에 적었다.
- 배포 쪽 = 브리지 `_make_runner(키)` 로 만든 **실제 러너 객체**의 입출력 + 가짜 ROS 로 돌린 노드 `main()` + 젯슨이 남긴 실물 블랙박스의 재생.
- HEAD 와 젯슨 판(커밋 8af2f6e5 의 러너·c5_switch + `deploy/cand/c5krh_1007/rl_bridge_node.py`)을 따로 올려 같은 표로 비교했다.
- 젯슨 접속·ROS·CAN·GPU·Isaac 실행 없음. 기존 코드 수정 없음. **계약 검증(열린 고리)이며 폐루프 성능·실물 성공의 증명이 아니다**(CLAUDE.md 6).

## 0. 결론 먼저

- 불일치는 중복을 빼면 **11 줄**이다: 종전 문서에 있던 것 10, 이 감사가 새로 찾은 것 1. 확인 불가는 6 줄 — 현장 측정이 있어야 하는 것과 평가 조건이다.
- **새로 찾은 불일치 — 클립**(crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800): 학습 `base_ang_vel ±20; joint_pos ±10; joint_vel ±60; actions ±10; gyro_ema ±20; joint_vel_ema ±60` ↔ 배포 클립 없음. 정상 범위에서는 두 쪽 값이 같고, 범위 밖(낙상·폭주·센서 튐)에서만 갈린다 — 1 절.
- **종전 문서에 '클램프 누계 증가'로만 적혀 있던 소프트 한계 클램프를 관절별로 풀었다**: 학습은 발목 앞뒤만 자르는데 실물 stage8 은 전 관절을 자른다. 지면에 서 있던 구간에서 학습이 자르지 않는 관절이 잘린 틱 — 1007 right_hip_f 9 틱(+10.4°); 1003 right_hip_f 4 틱(+2.9°). soft 범위는 좌우로 다르다(왼/오른: hip_f 45°/40.7° · hip_a 13°/13.2°, 34°/23.8° · hip_r 24°/29° · knee 83.5°/80° · ankle_f 22°/18.5°). KE 회복 시뮬 평가에는 이 클램프가 없다(5 절 빈칸 9) — 영향은 대조 평가가 없어 모른다.
- 관측 320 의 항 순서·이력·필터 계수·직전 행동 정의·클록, 행동 스케일·기본자세·슬루·LPF·블렌딩, 감지기 설정, 합성 러너 인계 규칙은 학습(또는 시뮬 평가)과 일치한다 — 속성값을 읽은 것이 아니라 러너 입출력에서 역산했다(3 절).
- **젯슨 판은 crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra 에서 HEAD 와 관측·원시 행동·목표각·노드 발행이 비트 동일하다**. 레지스트리·상수 차이 없음.
- 실물 블랙박스 재생 `rl_bridge_20261007_200441.jsonl`(c5krh_g21a_g24ra, 1780 틱 = 36 s): HEAD 관측 최대 |Δ| 8.3e-05 · 원시 행동 0.00021 · LPF 뒤 목표 0.0023° · 회복 모드 불일치 0 틱 / 젯슨 판 8.3e-05 · 0.00021 · 0.0023° (기록 반올림 1e-4 한도).
- 실물 블랙박스 재생 `rl_bridge_20261003_174906.jsonl`(crouch_g21a, 15299 틱 = 306 s): HEAD 관측 최대 |Δ| 9e-05 · 원시 행동 0.00039 · LPF 뒤 목표 0.0037° / 젯슨 판 9e-05 · 0.00039 · 0.0037° (기록 반올림 1e-4 한도).

| 브리지 키 | 일치 | 불일치 | 그중 종전 문서에 있던 것 | 그중 이번에 찾은 것 | 꼬리표 없음 | 확인 불가 |
|---|---|---|---|---|---|---|
| `stand_crouch_g21a` | 42 | 7 | 6 | 1 | 0 | 5 |
| `stand_c5kr_g21a_g24ra` | 63 | 9 | 8 | 1 | 0 | 6 |
| `stand_c5krh_g21a_g24ra` | 65 | 9 | 8 | 1 | 0 | 6 |
| `stand_w3s3d_15800` | 42 | 9 | 8 | 1 | 0 | 5 |
| `stand_kew3_g21a_g24ra_w3s3d15800` | 79 | 11 | 10 | 1 | 0 | 6 |

건수는 표의 줄 수다(한 원인이 여러 줄·여러 키에 걸린다). 스크립트 종료 코드는 불일치가 있으면 1 이다. `--allow-known` 은 꼬리표가 붙은(이 문서에 적힌) 불일치만 있을 때 0 — 새 불일치가 생기면 다시 1.

완료 구분: **오프라인 검증 완료** = 이 문서의 일치·불일치 판정 전부 / **미검증** = 5 절의 남은 빈칸(평가가 없는 조건) / **현장 확인 필요** = 2 절.

## 1. 불일치 (11 줄)

비고 칸은 어느 쪽이 학습과 같은지로 시작한다. 측정이 아닌 것은 (추측)으로 적었다. 조치는 하지 않았다(감사 범위).

| # | 항목 | 해당 키 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 비고·영향 | 근거(파일:줄) | 구분 |
|---|---|---|---|---|---|---|---|---|
| 1 | **클립** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | base_ang_vel ±20; joint_pos ±10; joint_vel ±60; actions ±10; gyro_ema ±20; joint_vel_ema ±60 | 클립 없음(프로브: base_ang_vel 100, joint_vel 200, joint_pos 50, gyro_ema 100, joint_vel_ema 200 그대로 통과) | 같음(프로브 비트 동일) | 학습과 같은 쪽은 시뮬(평가)뿐이다. 값이 범위 안이면 두 쪽이 같다. 실물 로그(1007 c5krh_g21a_g24ra 36 s·1003 crouch_g21a 306 s) 최대: base_ang_vel 4.07, joint_pos 1.03, joint_vel 3.07, actions 4.82, gyro_ema 2, joint_vel_ema 1.47, 원시 행동 4.95 — 전부 범위 안이라 이 기록들에서는 차이가 0 이다. 범위 밖(낙상·폭주·센서 튐)에서는 시뮬만 자른다: 시뮬 X2 폭주(원시 행동 2 000 까지)에서 시뮬 관측은 ±10 에 묶였고(TRANSITION_W3KE §5.6) 배포는 묶이지 않는다 → 폭주 양상이 시뮬과 달라질 수 있다(추측) | 2026-10-02_21-23-08_g21a/params/env.yaml:391 (observations.policy.base_ang_vel.clip) 외 5 항 · Isaac Lab /home/ryu/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:405(잡음 → 클립 → 스케일 → 이력 순서 확인; 판: 학습 기록 'HEAD detached at v2.3.2'·바뀐 파일 0 개 / 현재 v2.3.2·이 파일 변경 없음) ↔ rl_walking/deploy/policy_runner_stage4.py:417(조립부에 클립 없음) | 신규(이 감사 — 정상 범위에서는 값이 같다) |
| 2 | **관절속도 원천(유한차분 식)** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 2.7e-07; 단순 차분으로 재구성하면 0.48) | 같은 식(노드 발행 비트 동일) | 피드백 값이 N 틱 머문 뒤 바뀔 때만 다르다: 브리지는 0·…·0·Δq/(N·dt), 학습 식에 같은 신호를 넣으면 0·…·0·Δq/dt. 시뮬 엔코더는 매 틱 갱신이라 머무는 표본이 없다(양자화 미모형). 실물 1007 로그: 직전과 같은 값 — 정지 관절 71%(엔코더 양자 0.00038 rad), 움직이는 관절(> 0.2 rad/s) 8.5%; 기록 qvel 은 브리지 식과 0.0005, 단순 차분과 최대 3.7 rad/s 차. 시뮬 폐루프 영향 신호 없음(PARITY §4.3) | rl_walking/mdp_hang.py:1299 ↔ jetson/scripts/rl_bridge_node.py:5185 | 기지(PARITY #6) |
| 3 | **gyro_ema 시작 규칙** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 0.513, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | 틱 0 만(옵션 없음) | 시뮬 쪽이 학습 계약이다. 정책 시작 뒤 0.9ⁿ 로 사라지는 과도 — 틱 1 의 차 = 0.9·¦x₁ − x₀¦(왼쪽 수치는 합성 신호 기준). 실물 세션 시작에서의 크기: real_1007 0.014 rad/s; real_1003 0.012 rad/s. 합성 러너 carry 는 ema_fresh_ticks 1 을 요구한다(2 면 기동 거부) | rl_walking/mdp_hang.py:892 ↔ rl_walking/deploy/policy_runner_stage4.py:442 | 기지(PARITY #10–11, 옵션 --runner-ema-fresh 2 기본 꺼짐) |
| 4 | **joint_vel_ema 시작 규칙** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 1.61, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | 틱 0 만(옵션 없음) | 시뮬 쪽이 학습 계약이다. 정책 시작 뒤 0.9ⁿ 로 사라지는 과도 — 틱 1 의 차 = 0.9·¦x₁ − x₀¦(왼쪽 수치는 합성 신호 기준). 실물 세션 시작에서의 크기: real_1007 0.017 rad/s; real_1003 0.017 rad/s. 합성 러너 carry 는 ema_fresh_ticks 1 을 요구한다(2 면 기동 거부) | rl_walking/mdp_hang.py:892 ↔ rl_walking/deploy/policy_runner_stage4.py:442 | 기지(PARITY #10–11, 옵션 --runner-ema-fresh 2 기본 꺼짐) |
| 5 | **LPF 초기값** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 리셋 틱의 현재 관절각 | 첫 슬루 출력(노드 세션: 첫 발행 − 첫 슬루 출력 = 3.5e-18 rad) | 같음 | 시뮬 쪽이 학습 계약이다. 첫 틱 차이 ≤ 슬루 × (1−β) = 2.92°, (1−β)ⁿ 로 사라진다. 그 구간은 온셋 블렌딩 α ≈ 0 이라 발행 목표에 거의 안 나온다 | rl_walking/actions.py:128 ↔ rl_walking/deploy/deploy_chain.py:38·jetson/scripts/rl_bridge_node.py:5317 | 기지(PARITY #13) |
| 6 | **소프트 한계 클램프(자르는 관절)** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | clamp_to_soft_limits True: config soft 를 '.*_ankle_f_joint' 관절만(그 밖은 물리 한계까지 통과) · 한계 파일 마지막 변경 2026-09-18(모든 run 학습 시작 전 — 학습과 실물이 같은 표를 읽었다) | stage8 노드: **전 관절**을 config soft 경계로 자른다(러너·브리지는 자르지 않음) | 같음(stage8 — 8af2f6e5 판도 같은 규칙) | 학습과 같은 쪽이 없다(실물이 더 많은 관절을 자른다). 실물 로그에서 목표가 soft 를 넘은 틱 — 1007 c5krh_g21a_g24ra(전 구간 지면): right_hip_f < -40.7° 9 틱(35.02–35.18 s, 최대 +10.4°), left_ankle_f > 39° 16 틱(35.0–35.58 s, 최대 +19.2°), right_ankle_f < -39° 12 틱(35.36–35.58 s, 최대 +11.4°) · stage8 클램프 누계 35(= 전 구간 초과 틱 수; 브리지가 읽는 누계는 한 틱 늦다) / 1003 crouch_g21a(지면 구간 = 245 s 이전): right_hip_f < -40.7° 4 틱(204.18–204.24 s, 최대 +2.9°), left_ankle_f > 39° 11 틱(204.18–205.34 s, 최대 +6.0°), right_ankle_f < -39° 8 틱(204.1–204.24 s, 최대 +5.0°); 그 뒤(줄 회수·공중) 470 틱은 따로 · stage8 클램프 누계 493(= 전 구간 초과 틱 수; 브리지가 읽는 누계는 한 틱 늦다). 그중 **학습에서 자르지 않는 관절: 1007 right_hip_f 9 틱(+10.4°); 1003 right_hip_f 4 틱(+2.9°)**. soft 범위가 좌우로 다른 관절(왼/오른): hip_f 45°/40.7° · hip_a 13°/13.2°, 34°/23.8° · hip_r 24°/29° · knee 83.5°/80° · ankle_f 22°/18.5°. 이 클램프가 1007 두 번째 회복의 결과에 영향을 줬는지는 모른다 — 대조 실험이 없고, 같은 회복에서 35.4 s 무렵 발목 롤 드라이버 두 개가 재시작해 35.60 s 에 스택이 정지했다(reports/real_1007/README.md). KE 회복·전환 시뮬 평가기(eval_ground_recovery·eval_transition*)에는 stage8 식 전 관절 클램프가 없다 — W3S3d 걷기만 대조가 있다: 발목 롤 목표의 12–17 % 틱이 잘리고 폐루프 낙상 차 신호 없음(PARITY §4.4). 종전 문서에는 '클램프 누계 증가'(REAL_1003·PARITY §4.4)와 '좌우 비대칭·stand 세대 기준 재검토 권고'(0905 B7)로 적혀 있다 — 이 줄이 더한 것은 관절·방향별 실측과 KE 평가의 빈칸이다(0905 이후 한계 파일 변경 1 건: 2026-09-18 발목앞뒤(dorsi) 한계 29/30 → 39/40° (승윤님 지시 0918): 왼 soft_m) | rl_walking/mdp_hang.py:476(joints 기본값)·rl_walking/actions.py:166 ↔ jetson/scripts/stage8_12axis_mit_control_node.py:1116 · config/joint_limits_12dof.json · reports/real_1007/rl_bridge_20261007_200441.jsonl · reports/real_1003/rl_bridge_20261003_174906.jsonl | 기지(PARITY #14·§4.4, 0905 B7, REAL_1003) |
| 7 | **한계 초과 명령의 거부** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 없음(클램프만 — 목표가 아무리 넘어도 경계값으로 계속 구동) | stage8: 경계를 25° 넘게 넘으면 그 관절 명령을 거부(limit_runaway) | 같음(stage8) | 학습과 같은 쪽이 없다. 실물 로그의 최대 초과(전 구간): 1007 +19.2°, 1003 +17.7° — 거부 문턱까지 5.8° 남았다(거부 기록 없음). 넘는 순간 실물은 그 관절이 직전 명령에 머물고 시뮬은 경계에서 계속 민다(발목 명령이 soft 39° 를 넘어 56.7° 까지 감긴 0922 실물 낙상과 같은 계열 — rl_bridge_node.py 의 stand_crouch_v6 주석) | rl_walking/actions.py:166 ↔ jetson/scripts/stage8_12axis_mit_control_node.py:1118 | 기지(0905 감사 B7) |
| 8 | **명령 관측(회복 정책)** | c5kr_g21a_g24ra, c5krh_g21a_g24ra, kew3_g21a_g24ra_w3s3d15800 | 서기 50%: (0,0,0) / 그 밖 vx U(0.05, 0.15), vy (0, 0), wz (0, 0) | 회복 정책 입력만 cue vx 0.03 를 1.0 s(켜짐 틱 필터 ω_y>0), 그 밖 0 | 0.03 | 시뮬 평가와는 같고 학습 범위와는 다르다: cue 값이 학습 명령 범위 하한보다 작다(보상 모드 문턱 walk_min 0.03 과 같음). 시뮬 평가 인자 --sw_rec_vx 0.03 와는 같다 — 의도된 선택(reports/gen_c5/act1/PLAN.md 'am: 최소 진폭') | 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/params/env.yaml:2220 (commands.base_velocity.ranges.lin_vel_x) ↔ jetson/scripts/rl_bridge_node.py:323 | 기지(의도·시뮬 평가 동일) |
| 9 | **타이머 정밀도** | c5kr_g21a_g24ra, c5krh_g21a_g24ra, kew3_g21a_g24ra_w3s3d15800 | float32 누적(시뮬 텐서) | float64 누적(imu_f32 False) | float64(옵션 없음) | 평가 쪽(float32)이 기준이다. 꺼짐 시각이 일부 1 틱 이르다(91 건 중 44 건). 짝 비교 낙상 차 없음(PARITY §4.7). 실물 1007 재생에서는 회복 모드 불일치 0 틱 | rl_walking/scripts/eval_ground_recovery.py:989 ↔ rl_walking/deploy/c5_switch.py:264 | 기지(PARITY #24, 옵션 --c5-imu-f32 기본 꺼짐) |
| 10 | **명령 관측: 헤딩 고리(외부 P 고리)** | w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 학습에 없음(p_hloop 0.0, heading_command False) — 걷기 명령은 한 축만 | 기본 켜짐(K 0.5·±0.4 rad/s): 앞·옆 명령에 wz 를 **동시에** 넣는다 — 노드 세션 측정 관측 명령 (0.06, 0, -0.0349) | — | 학습과 같은 쪽이 없다(학습 분포 밖: 두 축 동시·¦wz¦ < 0.1). 시뮬 평가(eval_w3 --heading_loop 0.5 --hl_wzmax 0.4)는 같은 식으로 있음 — PARITY_AUDIT_1007 §4.5(오차 15° 문턱이 정상 걸음에서 30–41 % env 를 멈춘다, 결정 D7); 합성 러너 + 고리의 폐루프 평가는 없음(X6·X7 은 고리 끔 — KEW3_DESIGN §8.3) | 2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml:1214 (events.w3_cmd.params.p_hloop) ↔ rl_walking/deploy/w3_deploy.py:307·jetson/scripts/rl_bridge_node.py:1323 | 기지(PARITY #17–18, D7) |
| 11 | **명령 변경 간격** | w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 리샘플 (8, 12) s(STEP→STEP 위상 유지) | 최소 체류 1.0 s(위상 유지) | — | 학습 범위는 8 s 이상 — 배포는 그보다 잦은 변경을 허용한다. 시뮬 1 s 순환 평가에서 낙상 증가 신호 없음(PARITY §4.6) | 2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml:2139 (commands.base_velocity.resampling_time_range) ↔ rl_walking/deploy/w3_deploy.py:206 | 기지(PARITY #19) |

결정이 필요한 것(이 감사는 고치지 않았다):

- 관측 클립: 러너 조립부에 학습과 같은 클립을 넣으면 일치한다. 정상 범위에서는 출력이 바뀌지 않는다(6 절 민감도 시험의 반대 방향 대조 `obs_clip_added`). 젯슨 배포 파일이 바뀌므로 승인 사항이다.
- 소프트 한계: 0905 감사 B7 이 "soft 표를 stand 세대 기준으로 재검토"를 권고했다. 그 뒤 한계 파일(config/joint_limits_12dof.json)의 변경은 1 건(2026-09-18 발목앞뒤(dorsi) 한계 29/30 → 39/40° (승윤님 지시 0918): 왼 soft_m) — 힙·무릎을 재검토했는지는 기록에서 확인하지 못했다. 선택지 ① KE 회복 시뮬 평가에 stage8 과 같은 전 관절 클램프·25° 거부를 넣어 영향부터 재기 ② 실물 기구 한계를 확인한 뒤 soft 표 재검토(안전 한계이므로 성공률을 높이려는 목적으로 넓히지 않는다 — CLAUDE.md).

## 2. 확인 불가 (6 줄) — 왜 못 했고 무엇이 필요한가

| 항목 | 해당 키 | 학습 값 | 배포 값 | 이유·필요한 것 |
|---|---|---|---|---|
| **IMU 축 정렬(몸통 프레임 +x 앞·+z 위)** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 시뮬 몸통(base) 프레임 | IMU 어댑터가 낸 gyro·quat 을 그대로 사용 — 축 부호는 현장 확인 기록(0912–0914)에 기댄다 | 오프라인으로 알 수 없다(현장 확인 사항) |
| **접촉 0/1 의 물리 정의** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | feet_contact_real_v2: 발 법선력 > U(2, 20) N ∧ 발바닥 기울기 < U(6, 20)° → 5 틱마다 갱신 + 1 틱 지연 | 어댑터: 패드 4 개 중 하나라도 (카운트 − 바이어스) > 200 → 1 (미보정 카운트, 갱신 100 ms). 브리지는 비트를 그대로 쓴다 | 학습 쪽 조건은 실물 패드의 가정 모형이다(평평 조건 θ_off 는 가정값). 패드 비트와 실제 접지 시각의 대조는 현장 확인 필요. 시뮬 폐루프에서는 조건 차이에 둔감(PARITY §4.2) |
| **블렌딩 시작점** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | env 기본자세(편향 DR 포함) 홀드 목표 | 인계 순간 stage8 이 스트리밍하던 명령(노드 세션: 첫 발행 − 가짜 stage8 명령 = 0°) | 실물 스탠바이 자세와 정책 기본자세의 차가 학습 편향 대역 안인지는 현장 자세에 달렸다(착지 자동 인수 경로만 8° 게이트 TD_POSE_TOL_DEG) |
| **행동 지연** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 〔crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra〕 U{0..3} 틱 / 〔w3s3d_15800〕 U{1..5} 틱 / 〔kew3_g21a_g24ra_w3s3d15800〕 g21a: U{0..3} 틱 / g24ra_seed1_from_w2e20294: U{0..3} 틱 / w3S3d_from_s2d: U{1..5} 틱 → 공통 구간 1..3 틱 | 인위 지연 없음 — 실물 루프 지연이 그 몫(총 지연 미측정). 실물 1007 로그의 구간 실측: 브리지 발행 → stage8 처리 중앙 5.2 ms(99 % 9.9, 최대 12.9) · status 나이 중앙 6 ms(최대 10) | 명령 전송·피드백 수신 구간은 한 틱(20 ms) 안으로 측정됐지만 모터 응답까지의 총 지연은 이 기록으로 알 수 없다 → 현장 측정 필요(REAL_TEST1 M1). 학습 하한이 1 틱인 정책(걷기)이 있다 — 걷기 시뮬 평가는 지연 1–7 틱만 했고 0 틱 평가는 없다(PARITY §4.1: ≥ 5 틱에서 낙상 3–7 배); 합성 러너는 한 하드웨어 지연이 두 정책의 학습 범위에 모두 들어야 한다 |
| **센서 관측 지연(IMU·접촉)** | crouch_g21a, c5kr_g21a_g24ra, c5krh_g21a_g24ra, w3s3d_15800, kew3_g21a_g24ra_w3s3d15800 | 자이로 0–3 틱 · 자세(중력) (0, 3) 틱 · 접촉 5 틱 유지 + 1 틱 | 브리지는 최신 수신값을 그대로 쓴다(신선도 한계 IMU·관절 0.2 s, 발 0.35 s). 센서 안쪽·전송 지연은 미측정. 브리지에서 본 것: IMU 표본이 직전 틱과 같은 틱 real_1007 0.39%; real_1003 0.03%(거의 매 틱 새 표본) | 현장 측정 필요. 발 접촉은 0.35 s 묵은 값까지 정상 입력으로 쓴다 — 학습의 최대 지연(6 틱 = 0.12 s)보다 길다(발 메시지가 2 번 넘게 빠진 경우) |
| **평가 하네스의 자이로 지연** | c5kr_g21a_g24ra, c5krh_g21a_g24ra, kew3_g21a_g24ra_w3s3d15800 | 학습 0–3 틱 DR / KE·전환 평가는 태스크 Biped12-HangStand-G11a-v0 + 관측 덮어쓰기(--gyro_delay 0 — 기본값, 평가 인자에 없음) | 실물 IMU 경로 지연 미측정 | 감지기·회복 평가 수치는 자이로 지연 0 조건에서 나왔다 — 실물 지연이 크면 평가가 덮지 않는 영역 |

## 3. 일치 — 핵심 근거(측정값)

판정표(4 절)의 ○ 가운데 지시서의 일곱 낱말(observation · action · filter · history · previous-action · phase · supervisor)과 젯슨 판에 바로 답하는 줄만 뽑았다. 값은 실제 객체의 입출력에서 잰 것이다. 전체는 부록 A.

| 무엇 | 키 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| observation | crouch_g21a | **항 목록·순서·프레임 차원**: base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | 일치 |
| observation | crouch_g21a | **관측 벡터 전체 재구성(600 틱)**: 학습 계약으로 다시 조립한 값(항 순서·이력·필터·기본자세) | 실제 러너 관측과 최대 ¦Δ¦ 0(base_ang_vel); S8a EMA 시작 규칙만 배포식으로 둔 재구성 | HEAD 와 관측·목표 비트 동일 | 일치 |
| observation | c5krh_g21a_g24ra | **구성 정책들의 학습 관측 설정이 서로 같은가**: 서기 g21a·회복 g24ra_seed1_from_w2e20294 | 하나의 관측 조립을 공유(C5Runner 코어 / 합성 러너 이어받기) | 같은 구조 | 일치 |
| observation | crouch_g21a | **스케일**: 없음(전 항 scale: null) | 스케일 없음(프로브: 관측 = 입력 True) | 프로브 비트 동일 | 일치 |
| observation | crouch_g21a | **projected_gravity 정의**: 쿼터니언으로 (0,0,−1) 을 몸통 프레임에(단위벡터) + 장착 편향 DR 피치 (-0.026, 0.026)·롤 (-0.026, 0.026) rad | projected_gravity(q): 독립 회전행렬 식과 최대 ¦Δ¦ 0 · 맵 편향 보정 [1.3, -0.6]° 식 오차 3.6e-15° | ¦Δ¦ 0 · 보정 [1.3, -0.6]° | 일치 |
| observation | crouch_g21a | **joint_pos = 엔코더각 − 기본자세**: encoder_joint_pos_rel_v2; 기본자세(init_state, deg) +30.26 -30.26 +0.00 +0.00 +0.00 +0.00 -38.50 +38.50 +28.00 -28.00 +0.00 +0.00 | 러너 default_pose 와 최대 ¦Δ¦ 2.9e-08 rad · 관측에서 역산한 기본자세와 2.9e-08 rad | 최대 ¦Δ¦ 2.9e-08 rad | 일치 |
| observation | crouch_g21a | **feet_contact 순서·값**: [left_ankle_r_joint, right_ankle_r_joint] (preserve_order True) 0/1 | contact2 [왼, 오] 0/1 그대로(프로브 ¦Δ¦ 0); 0/1 이외 값은 러너가 거부 | 비트 동일 | 일치 |
| observation | crouch_g21a | **명령 관측(서기 정책)**: (0,0,0) 고정 | stand 가족: 명령 (0,0,0)·heading_hold 강제 끔(노드 세션 관측 명령 최대 ¦값¦ 0) | 같음(노드 발행 비트 동일) | 일치 |
| history | crouch_g21a | **이력 길이(항별 프레임 수)**: [5] | [5] | [5] | 일치 |
| history | crouch_g21a | **이력 배열 순서·리셋 첫 틱**: 오래된 것 먼저, 첫 틱 전 칸 백필(시뮬 기록 실측: 순서 True·백필 True) | 최신 = 마지막 칸 True · 첫 틱 백필 True(프로브) | 프로브 비트 동일 | 일치 |
| filter | crouch_g21a | **joint_vel = 엔코더 유한차분 속도의 EMA**: encoder_joint_vel_lpf β 0.25(에피소드 첫 틱 = 원값) | 프로브 역산 β 0.25 · 시뮬 소스 실행 대조 최대 ¦Δ¦ 0 | 0.25 | 일치 |
| filter | crouch_g21a | **gyro_ema 계수**: gyro_ema_lagged α 0.1 | 프로브 역산 α 0.1 · 정상 구간 재구성 오차 0 | 프로브 비트 동일 | 일치 |
| filter | crouch_g21a | **joint_vel_ema 계수**: encoder_joint_vel_ema_v2 α 0.1(원시 속도에서 — v2) | 프로브 역산 α 0.1 · 모드 v2 · 정상 구간 재구성 오차 1.5e-08 | v2 | 일치 |
| filter | crouch_g21a | **액션 LPF β**: env 별 U(0.2, 0.4) | STAND_LPF_BETA 0.27 · 노드 세션 역산 0.27(키가 stand 로 시작 → 적용 True) | 0.27 | 일치 |
| previous-action | crouch_g21a | **actions(직전 행동) 정의·필터**: last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 6.1e-08 vs 슬루 뒤 값을 넣었다면 0.37; 시뮬 소스 실행 대조 1.5e-07 | 0.27 | 일치 |
| action | crouch_g21a | **관절 순서(12)**: 시뮬 기록 joint_names: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | JOINT_ORDER: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | 같음 | 일치 |
| action | crouch_g21a | **차원·행동 클립**: actor 출력 12 · actions.clip None · clip_actions None | 러너 출력 12 · 클립 없음(프로브 원시 행동 최대 ¦값¦ 2.19 가 그대로 목표식에 들어간다 — 재구성 오차는 아래) | 같음 | 일치 |
| action | crouch_g21a | **스케일**: 0.25 | 프로브 역산 0.25(관절 12/12, 포화 안 된 틱 88%) | 0.25 | 일치 |
| action | crouch_g21a | **오프셋 = 기본자세(use_default_offset)**: use_default_offset True · 명목 기본자세(위 joint_pos 줄) | 프로브 역산 오프셋과 학습 기본자세 최대 ¦Δ¦ 2.9e-08 rad | 목표 비트 동일 | 일치 |
| action | crouch_g21a | **슬루 상한(틱당)**: 4.0000° | 프로브 최대 틱 변화 4.0000° · 1 rad 떨어진 첫 틱 이동 4.0000° | 4.0000° | 일치 |
| action | crouch_g21a | **슬루 기준 초기화**: 리셋 틱의 현재 관절각(_needs_init → joint_pos) | 첫 step 의 qpos(프로브: 전 관절이 qpos 에서 정확히 슬루 한 번 True) | 목표 비트 동일 | 일치 |
| action | crouch_g21a | **목표각 재구성(600 틱)**: 기본자세 + 스케일·원시 → 슬루(학습 식) | 실제 러너 출력과 최대 ¦Δ¦ 0 rad · 타깃 클램프 없음 | HEAD 와 비트 동일 | 일치 |
| action | crouch_g21a | **처리 순서**: 지연 → 슬루 → LPF → 온셋 블렌딩 → 백래시 → 소프트 클램프 | 러너 슬루 → 브리지 LPF → (보정기 꺼짐) → 온셋 블렌딩 → 전송 → stage8 슬루·클램프; 노드 세션 재구성 오차 1.3e-07 rad·블렌딩 0° | 노드 발행 비트 동일 | 일치 |
| action | crouch_g21a | **온셋 블렌딩(인계)**: 유지 U(0, 0.3) s + 스무스스텝 U(1, 1.5) s | 유지 0 s + 스무스스텝 BLEND_S 1.5 s — onset_alpha(소스 실행) vs blend_alpha 최대 ¦Δ¦ 6e-08; 노드 세션 α(t) 오차 0.00049(기록 3 자리 반올림) | 1.5 | 일치 |
| action | crouch_g21a | **PD 게인(kp/kd)**: [(150.0, 5.0)] · DR ×(0.85, 1.1)/×(0.8, 1.2) | POLICY_KP 150.0/KD 5.0 · 맵 불일치 관절 0(--live 는 불일치면 기동 거부) | KP 150.0/KD 5.0 · 젯슨 맵 사본(1005) 게인 같음 | 일치 |
| 시간 | crouch_g21a | **제어 주기**: 20.0 ms(sim.dt × decimation 4) | TICK_SEC 20.0 ms · 러너 POLICY_DT 20.0 ms · 실물 1007 로그 틱 간격 평균 20.00 ms(1–99 % 19.9–20.1, 최대 23.3) | 20.0 ms | 일치 |
| 시간 | crouch_g21a | **접촉 갱신 주기**: 5 틱마다 갱신(feet_contact hold_ticks) — 그 사이는 직전 비트 유지 | 브리지는 틱마다 최신 수신 비트를 읽는다. 실물 로그에서 발 원카운트 합이 바뀌는 간격: real_1007 최빈 5 틱(353/354 구간, 더 긴 구간 1 개·최대 10 틱); real_1003 최빈 5 틱(2872/2875 구간, 더 긴 구간 3 개·최대 10 틱) | 같은 실측(젯슨 세션 기록) | 일치 |
| phase | crouch_g21a | **gait_clock(서기 정책)**: gait_clock: [sin φ, cos φ], φ = 2π·1.1·(틱·dt), 리셋 = 위상 0 | 프로브 위상 기울기 1.1 Hz · 첫 틱 (0, 1) · 시뮬 소스 실행 대조 6e-08 · 1 시간 뒤 float32 위상 오차 0.00025 | 1.1 | 일치 |
| phase | w3s3d_15800 | **스텝 주파수 f_step**: 1.1 | W3Config 1.1 · 프로브(걷는 중 위상 증가/틱) 1.1 Hz | —(젯슨 미배포 키) | 일치 |
| phase | w3s3d_15800 | **관측 게이트 램프**: ramp_s 0.2(env.yaml 에 없으면 w3_gait_update 기본값) | W3Config 0.2 · 프로브: 켜진 뒤 게이트 1 까지 [10, 10, 10, 10] 틱 | —(젯슨 미배포 키) | 일치 |
| phase | w3s3d_15800 | **gait_enable 래치**: 켜기: 다음 갱신에 즉시(위상 0 또는 리드 규칙) / 끄기: 반주기 경계에서만 / 에피소드 틱 ≤ 1 은 즉시 반영 | 프로브: 켜기 지연 [0, 0, 0, 0] 틱 · 끄기 [(0, 0.478, True), (4, 0.986, True), (6, 0.49, True)](대기 틱, 직전 위상, 경계 통과) · 서기 클록 (0,0) True | —(젯슨 미배포 키) | 일치 |
| phase | w3s3d_15800 | **리드 발 규칙 문턱(vy, wz, 옆 없음 판정)**: (0.02, 0.08, 0.02) | W3Config (0.02, 0.08, 0.02) · 프로브 시작 위상 [0.0, 0.5, 0.5, 0.0](앞, 옆−, 회전−, 제자리) · 리드 ['left', 'right', 'right', 'left'] | —(젯슨 미배포 키) | 일치 |
| supervisor | w3s3d_15800 | **명령 관측 범위(걷기)**: stage 3 단일 축: vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2); 학습 로그 도달값 vy_max 0.04·wz_max 0.2·보조 α 0 | W3CmdLimits vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2) · 범위 밖·두 축 동시 거부(validate_w3_cmd) | —(젯슨 미배포 키) | 일치 |
| supervisor | w3s3d_15800 | **걷기 켜기 선행조건(양발 s, 기울기 °, ¦ω¦ , 정착 s)**: (1, 7, 0.25, 1)(eval_transition 기본값) — 학습에는 없음(에피소드 70 % 가 곧바로 STEP) | (1, 7, 0.25, 1) | —(젯슨 미배포 키) | 일치 |
| supervisor | c5kr_g21a_g24ra | **평가 설정의 출처**: log_picture/ground_recovery_sab_imu_nd_s7.txt 의 '인자:' 줄 | 브리지 C5_DETECTOR_KW·C5_EXTRA → 실제 감지기 객체에서 읽은 값 | 같은 방식 | 일치 |
| supervisor | c5kr_g21a_g24ra | **켜기 기울기 편차 [°]**: 3 | 3 | 3 | 일치 |
| supervisor | c5kr_g21a_g24ra | **켜기 ¦ω_xy¦ [rad/s]**: 0.5 | 0.5 | 0.5 | 일치 |
| supervisor | c5kr_g21a_g24ra | **끄기 기울기 [°]**: 1.5 | 1.5 | 1.5 | 일치 |
| supervisor | c5kr_g21a_g24ra | **끄기 평온 지속 [s]**: 1 | 1 | 1 | 일치 |
| supervisor | c5kr_g21a_g24ra | **기준 자세 τ [s]**: 10 | 10 | 10 | 일치 |
| supervisor | c5kr_g21a_g24ra | **조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s)**: (0.6, 6, 0.4, 0.2) | (0.6, 6, 0.4, 0.2) | (0.6, 6, 0.4, 0.2) | 일치 |
| supervisor | c5krh_g21a_g24ra | **조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s)**: (0.6, 6, 0.4, 0.7) | (0.6, 6, 0.4, 0.7) | (0.6, 6, 0.4, 0.7) | 일치 |
| supervisor | c5krh_g21a_g24ra | **불응기 (s, 기울기, ¦ω¦)**: (1.5, 6, 1) | (1.5, 6, 1) | (1.5, 6, 1) | 일치 |
| supervisor | c5krh_g21a_g24ra | **감지 입력 EMA α**: 0.33 | 0.33 | 0.33 | 일치 |
| supervisor | c5krh_g21a_g24ra | **회복 cue (vx, 지속 s, 방향 = IMU ω_y)**: (0.03, 1, True) | (0.03, 1.0, True) | (0.03, 1.0, True) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **두 하위 러너 계약 호환**: 세 run 의 관측 설정 동일 · 기본자세 동일 | KE 코어 = W3 코어: 관측 레이아웃·기본자세·필터 β·EMA 모드·ema_fresh True | —(젯슨 미배포 키) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **W3 → KE 인계 규칙(정착 s, ¦ω_xy¦ 문턱, 상한 s, 방식)**: 평가기 정착 판정 (0.5, 0.15, 3)(산출물 머리) · X6(배포 코드 폐루프) --kw_handoff carry --kw_settle 0.5 | (0.5, 0.15, 3.0, carry) · 프로브: 조용한 양발 25 틱 뒤 인계, 시끄러우면 150 틱(상한), 한발이면 대기 True | —(젯슨 미배포 키) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **KE → W3 인계**: 켜기 수락 틱에 W3(첫 틱 STAND, 다음 틱 gait 0→1, 게이트 0.2 s 램프) — 평가 direct 팔 | 프로브: 같은 틱 전환 True · 첫 틱 ge False·클록 이력 0 True · 둘째 틱 ge True(게이트 0.1) · 전환 틱 목표 변화 4°(슬루 상한 이내 — 슬루 기준을 이어받는다. 합성 입력이라 크기 자체는 뜻이 없다) | —(젯슨 미배포 키) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **인계 뒤 KE 상태**: 이력·필터·직전 행동 이어받기, 클록만 위상 0 에서(평가 direct 팔의 클록 패치), 감지기 새로 | 프로브: KE 틱 0 · 클록 이력 전부 (0,1) True · 감지기 켜짐 누계 0 · 첫 틱 목표 변화 4°(슬루 상한 이내) | —(젯슨 미배포 키) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **회복 모드 중 걷기 켜기**: 평가 X6: 감독기 선행조건 + KE 회복 모드면 거부 | 프로브: 회복 모드에서 사유 'KE 회복 모드 켜짐 — 서기 복귀 뒤에 켤 것' | —(젯슨 미배포 키) | 일치 |
| 합성 | kew3_g21a_g24ra_w3s3d15800 | **노드 세션(가짜 ROS 로 main() 그대로, HEAD)**: 학습 식 재구성: 슬루 → LPF β → 스무스스텝 블렌딩 → 모터 프레임 | 정책 틱 558(연속 558) · 체인 6.6e-07 rad · 블렌딩 0° · 발행 = 목표 − 기준선 0° · 명령 응답 6/6 수락 · 전환 KE→W3 2 회·W3→KE 2 회 | —(젯슨 미배포 키) | 일치 |
| 정책 파일 | crouch_g21a | **서기 정책 가중치 = 학습 체크포인트**: 2026-10-02_21-23-08_g21a/model_11300.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | policy.pt = 매니페스트; policy.onnx = 매니페스트; env.yaml = 매니페스트 | 일치 |
| 정책 파일 | c5krh_g21a_g24ra | **회복 정책 가중치 = 학습 체크포인트**: 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/model_21793.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | policy.pt = 매니페스트; policy.onnx = 매니페스트; env.yaml = 매니페스트 | 일치 |
| 정책 파일 | w3s3d_15800 | **걷기 정책 가중치 = 학습 체크포인트**: 2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | —(젯슨 미배포 키) | 일치 |
| 학습 쪽 기준 | crouch_g21a | **서기 학습 시점의 시뮬 소스(계약 함수)**: 커밋 190c03b9(2026-10-02 18:48:49) = run 시작 2026-10-02 21:23:08 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 19 개 같음 · 바뀐 것: SlewJointPositionAction(+15/−10 줄 — 지연 표본에 가중 옵션(delay_probs) 추가 — 비어 있으면 종전 균등 표본); _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | (학습 쪽 항목) | 일치 |
| 학습 쪽 기준 | c5krh_g21a_g24ra | **회복 학습 시점의 시뮬 소스(계약 함수)**: 커밋 e4d2ea28(2026-10-04 09:44:03) = run 시작 2026-10-04 10:08:41 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 1 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 20 개 같음 · 바뀐 것: _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | (학습 쪽 항목) | 일치 |
| 학습 쪽 기준 | w3s3d_15800 | **걷기 학습 시점의 시뮬 소스(계약 함수)**: 커밋 ee1d80e6(2026-10-07 00:11:10) = run 시작 2026-10-07 00:11:12 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 25 개 중 25 개 같음 · 감사가 읽는 규칙 17 개는 두 판에서 같음 | —(젯슨 미배포 키) | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **재구성한 파일 = 저장소가 기록한 배포본**: — | (HEAD 는 젯슨 미배포 변경 다수) | 브리지 md5 15dcabf2 = c5krh_1007.md5 · 러너 21b2416e = 매니페스트 · c5_switch 11cf9641 = · 8af2f6e5 + 패치 = 후보 True | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **레지스트리·상수 값 HEAD 와 같은가**: — | 비교 29 항목 | 전부 같음 | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **러너 프로브(600 틱) HEAD 와 비트 비교**: — | 기준 | 관측 True · 원시 행동 True · 목표각 True · 회복 모드열 True | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교**: — | 정책 틱 340(회복 모드 63 틱) | 발행 목표 True · 관측 True | 일치 |
| 젯슨 판 | crouch_g21a | **실물 블랙박스 재생 — real_1003(젯슨이 실제로 낸 값과 대조)**: 기록 15299 틱(306.0 s), 기록 반올림 1e-4 — 그때의 젯슨 판(현 판 이전)이 낸 기록 | 관측 최대 ¦Δ¦ 9e-05 · 원시 행동 0.00039 · LPF 뒤 목표 0.0037° · 15299 틱 | 관측 최대 ¦Δ¦ 9e-05 · 원시 행동 0.00039 · LPF 뒤 목표 0.0037° · 15299 틱 | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **실물 블랙박스 재생 — real_1007(젯슨이 실제로 낸 값과 대조)**: 기록 1780 틱(35.6 s), 기록 반올림 1e-4 | 관측 최대 ¦Δ¦ 8.3e-05 · 원시 행동 0.00021 · LPF 뒤 목표 0.0023° · 회복 모드 불일치 0/1780 틱 · 기준 자세 0.00079° · 켜짐 2/2 | 관측 최대 ¦Δ¦ 8.3e-05 · 원시 행동 0.00021 · LPF 뒤 목표 0.0023° · 회복 모드 불일치 0/1780 틱 · 기준 자세 0.00079° · 켜짐 2/2 | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **실물 기록의 배포 체인 값 — real_1007**: 학습 온셋 블렌딩 상한 1.5 s · 관측 필터 β (0.27, 0.25) | — | 블렌딩 α(t) 가장 잘 맞는 길이 1.5 s(오차 0.0005) · 기록 obs_filter [0.27, 0.25] · jvel_ema v2 · 장착 편향 [1.3, -0.6]° · 명령 최대 ¦값¦ 0.0 · 틱 간격 평균 20.00 ms(최대 23.3) | 일치 |
| 젯슨 판 | c5krh_g21a_g24ra | **HEAD 에만 있는 선택 기능**: — | --runner-ema-fresh, --c5-imu-f32·--c5-settle(odic)·requiet | 없음(기본 꺼짐 기능이라 기본 동작은 위 비트 비교대로 같다) | 일치 |

## 4. 항목 × 정책 판정표와 기존 검증 대응

○ 일치 · ✕ 불일치 · ? 확인 불가 · – 해당 없음. 마지막 칸 = 그 항목을 덮는 기존 검증('없음' = 이 감사가 처음). [P1]–[P6] = selftest_w3_rollout_parity, PARITY #n = reports/gen_w3/PARITY_AUDIT_1007.md 표의 번호.

| 분류 | 항목 | g21a | c5kr | c5krh | w3s3d_15800 | kew3_w3s3d15800 | 기존 검증(없음 = 이 감사가 처음) |
|---|---|---|---|---|---|---|---|
| 0 정책 파일 | 서기 정책 가중치 = 학습 체크포인트 | ○ | ○ | ○ | – | ○ | 없음(W3S3d 만 selftest_w3s3d_onnx — onnx vs pt 출력) |
| 0 정책 파일 | 서기 관측 정규화·행동 클립 | ○ | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 서기 학습 시점의 시뮬 소스(계약 함수) | ○ | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 서기 시뮬 평가 체크포인트 = 배포 파일 | ○ | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 회복 정책 가중치 = 학습 체크포인트 | – | ○ | ○ | – | ○ | 없음(W3S3d 만 selftest_w3s3d_onnx — onnx vs pt 출력) |
| 0 정책 파일 | 회복 관측 정규화·행동 클립 | – | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 회복 학습 시점의 시뮬 소스(계약 함수) | – | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 회복 시뮬 평가 체크포인트 = 배포 파일 | – | ○ | ○ | – | ○ | 없음 |
| 0 정책 파일 | 걷기 정책 가중치 = 학습 체크포인트 | – | – | – | ○ | ○ | 없음(W3S3d 만 selftest_w3s3d_onnx — onnx vs pt 출력) |
| 0 정책 파일 | 걷기 관측 정규화·행동 클립 | – | – | – | ○ | ○ | 없음 |
| 0 정책 파일 | 걷기 학습 시점의 시뮬 소스(계약 함수) | – | – | – | ○ | ○ | 없음 |
| 0 정책 파일 | 걷기 시뮬 평가 체크포인트 = 배포 파일 | – | – | – | ○ | ○ | 없음 |
| 1 관측 | 항 목록·순서·프레임 차원 | ○ | ○ | ○ | ○ | ○ | selftest_w3_rollout_parity [P1](W3S3d) · 러너 --selftest 골든(245) |
| 1 관측 | 이력 길이(항별 프레임 수) | ○ | ○ | ○ | ○ | ○ | [P1] |
| 1 관측 | 이력 배열 순서·리셋 첫 틱 | ○ | ○ | ○ | ○ | ○ | [P1]·러너 골든 |
| 1 관측 | 스케일 | ○ | ○ | ○ | ○ | ○ | [P1] |
| 1 관측 | 클립 | ✕ | ✕ | ✕ | ✕ | ✕ | 없음 |
| 1 관측 | base_ang_vel(자이로) 처리 | ○ | ○ | ○ | ○ | ○ | [P1] |
| 1 관측 | IMU 축 정렬(몸통 프레임 +x 앞·+z 위) | ? | ? | ? | ? | ? | 없음(PARITY #2) |
| 1 관측 | projected_gravity 정의 | ○ | ○ | ○ | ○ | ○ | [P5]·브리지 selftest 'IMU 장착 편향' |
| 1 관측 | 명령 관측(서기 정책) | ○ | ○ | ○ | – | ○ | 브리지 selftest W3('명령 관측 0')·없음(G21a) |
| 1 관측 | joint_pos = 엔코더각 − 기본자세 | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '학습 계약'(G21a·C5 키)·'W3 실험 플래그'(W3) |
| 1 관측 | 엔코더 관측 방식 | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '왕복 항등'·'수치 앵커'(변환식) |
| 1 관측 | joint_vel = 엔코더 유한차분 속도의 EMA | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '관측 필터 계약'(값)·[P1] |
| 1 관측 | 관절속도 원천(유한차분 식) | ✕ | ✕ | ✕ | ✕ | ✕ | PARITY #6·[P3] |
| 1 관측 | actions(직전 행동) 정의·필터 | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '관측 필터 계약'·[P1] |
| 1 관측 | gait_clock(서기 정책) | ○ | ○ | ○ | – | ○ | 러너 --selftest(식)·실물 로그 재생(이 감사) |
| 1 관측 | feet_contact 순서·값 | ○ | ○ | ○ | ○ | ○ | [P1]·[P6] 좌우 뒤바꿈 변이 |
| 1 관측 | 접촉 0/1 의 물리 정의 | ? | ? | ? | ? | ? | PARITY #9(문서) |
| 1 관측 | gyro_ema 계수 | ○ | ○ | ○ | ○ | ○ | [P1]·브리지 selftest '학습 계약'(EMA 계약 v2) |
| 1 관측 | gyro_ema 시작 규칙 | ✕ | ✕ | ✕ | ✕ | ✕ | PARITY #10·#11·[P1] |
| 1 관측 | joint_vel_ema 계수 | ○ | ○ | ○ | ○ | ○ | [P1]·브리지 selftest '학습 계약'(EMA 계약 v2) |
| 1 관측 | joint_vel_ema 시작 규칙 | ✕ | ✕ | ✕ | ✕ | ✕ | PARITY #10·#11·[P1] |
| 1 관측 | 관측 벡터 전체 재구성(600 틱) | ○ | ○ | ○ | ○ | ○ | [P1]–[P2](W3S3d, 시뮬 기록 원천) |
| 1 관측 | 구성 정책들의 학습 관측 설정이 서로 같은가 | – | ○ | ○ | – | ○ | 브리지 주석(수동 확인) |
| 1 관측 | 명령 관측(회복 정책) | – | ✕ | ✕ | – | ✕ | 브리지 selftest 'C5 kr 경로'(값 전달) |
| 1 관측 | 명령 관측 범위(걷기) | – | – | – | ○ | ○ | 브리지 selftest 'W3S3d 명령 채널' ①② |
| 1 관측 | 명령 관측: 헤딩 고리(외부 P 고리) | – | – | – | ✕ | ✕ | PARITY #17·#18(문서)·selftest_w3_equiv [F](식) |
| 1 관측 | 명령 변경 간격 | – | – | – | ✕ | ✕ | PARITY #19 |
| 2 행동 | 관절 순서(12) | ○ | ○ | ○ | ○ | ○ | sim_validate_runner_stage4.py(Isaac 필요, 단언)·[P1] |
| 2 행동 | 차원·행동 클립 | ○ | ○ | ○ | ○ | ○ | 러너 --selftest |
| 2 행동 | 스케일 | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '학습 계약' |
| 2 행동 | 오프셋 = 기본자세(use_default_offset) | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '학습 계약' |
| 2 행동 | 슬루 상한(틱당) | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '학습 계약'·'정책 500틱' |
| 2 행동 | 슬루 기준 초기화 | ○ | ○ | ○ | ○ | ○ | 러너 --selftest |
| 2 행동 | 목표각 재구성(600 틱) | ○ | ○ | ○ | ○ | ○ | [P1]·[P4] |
| 2 행동 | 처리 순서 | ○ | ○ | ○ | ○ | ○ | deploy_chain --selftest·[P4] |
| 2 행동 | 액션 LPF β | ○ | ○ | ○ | ○ | ○ | 브리지 selftest 'stand LPF'(주파수 응답) |
| 2 행동 | LPF 초기값 | ✕ | ✕ | ✕ | ✕ | ✕ | PARITY #13·[P4] |
| 2 행동 | 온셋 블렌딩(인계) | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '온셋 블렌딩'·deploy_chain --selftest |
| 2 행동 | 블렌딩 시작점 | ? | ? | ? | ? | ? | 없음 |
| 2 행동 | 소프트 한계 클램프(자르는 관절) | ✕ | ✕ | ✕ | ✕ | ✕ | PARITY #14(문서, 발목 롤만)·0905 감사 B7 |
| 2 행동 | 한계 초과 명령의 거부 | ✕ | ✕ | ✕ | ✕ | ✕ | 0905 감사 B7(문서) |
| 2 행동 | 노드(stage8) 슬루·주기 | ○ | ○ | ○ | ○ | ○ | policy_runner 문서(이중 슬루 항등)·없음(시험) |
| 2 행동 | PD 게인(kp/kd) | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '게인 정책' |
| 2 행동 | 학습에 없는 가산 보정(roll/pitch comp) | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '롤 보정기'(꺼짐 = 비트 동일) |
| 2 행동 | 노드 세션(가짜 ROS 로 main() 그대로, HEAD) | – | – | – | ○ | – | selftest_w3_bridge_node(스텁 정책) |
| 3 시간 | 제어 주기 | ○ | ○ | ○ | ○ | ○ | 브리지 selftest '학습 계약'(50 Hz) |
| 3 시간 | 행동 지연 | ? | ? | ? | ? | ? | 없음(PARITY #15) |
| 3 시간 | 접촉 갱신 주기 | ○ | ○ | ○ | ○ | ○ | 없음 |
| 3 시간 | 센서 관측 지연(IMU·접촉) | ? | ? | ? | ? | ? | 없음 |
| 4 위상 | 스텝 주파수 f_step | – | – | – | ○ | ○ | 브리지 selftest 'W3S3d 명령 채널' ①·selftest_w3_equiv [B] |
| 4 위상 | 관측 게이트 램프 | – | – | – | ○ | ○ | selftest_w3_equiv [B] |
| 4 위상 | gait_enable 래치 | – | – | – | ○ | ○ | selftest_w3_equiv [B](시뮬 소스 실행, 켜짐 길이 1–50 전수)·[P1] |
| 4 위상 | 리드 발 규칙 문턱(vy, wz, 옆 없음 판정) | – | – | – | ○ | ○ | selftest_w3_equiv [E]·[P1](옆 − 시작) |
| 4 위상 | 행동 보조·참조 잔차 | – | – | – | ○ | ○ | PARITY #22(기록 최대 0) |
| 5 감독기 | 걷기 켜기 선행조건(양발 s, 기울기 °, ¦ω¦ , 정착 s) | – | – | – | ○ | ○ | 브리지 selftest 'W3 세션 감독기'·시뮬 X6 |
| 5 감독기 | 명령 상한 덮어쓰기(--w3-cmd-limits) | – | – | – | ○ | ○ | 브리지 selftest 'W3S3d 명령 채널' ② |
| 5 감지기 | 평가 설정의 출처 | – | ○ | ○ | – | ○ | 없음 |
| 5 감지기 | 켜기 기울기 편차 [°] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 켜기 ¦ω_xy¦ [rad/s] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 끄기 기울기 [°] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 끄기 ¦ω_xy¦ | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 끄기 평온 지속 [s] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 켜진 뒤 최소 [s] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 기준 자세 τ [s] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 기준 추종 창 [s] | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s) | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 불응기 (s, 기울기, ¦ω¦) | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 조기 복귀 자격 (최대 기울기, 최대 ¦ω¦) | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 감지 입력 EMA α | – | ○ | ○ | – | ○ | 브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식) |
| 5 감지기 | 끄기 접촉 요구 | – | ○ | ○ | – | ○ | selftest_c5_imu_f32 [B] |
| 5 감지기 | 기준 자세 상한·갱신 조건 | – | ○ | ○ | – | ○ | selftest_c5_imu_f32 [B] |
| 5 감지기 | 회복 cue (vx, 지속 s, 방향 = IMU ω_y) | – | ○ | ○ | – | ○ | selftest_odic_equiv [A] |
| 5 감지기 | 감지 입력 원천 | – | ○ | ○ | – | ○ | 없음(구조) |
| 5 감지기 | 타이머 정밀도 | – | ✕ | ✕ | – | ✕ | selftest_c5_imu_f32 [B] |
| 5 감지기 | 평가 하네스의 관측 덮어쓰기(직전 행동 EMA, 관절속도 EMA, 엔코더 판, 접촉 판) | – | ○ | ○ | – | ○ | 없음 |
| 5 감지기 | 평가 하네스의 자이로 지연 | – | ? | ? | – | ? | 없음 |
| 6 합성 | 두 하위 러너 계약 호환 | – | – | – | – | ○ | kew3_runner --selftest [6]·브리지 selftest 'KE↔W3 합성' |
| 6 합성 | W3 → KE 인계 규칙(정착 s, ¦ω_xy¦ 문턱, 상한 s, 방식) | – | – | – | – | ○ | kew3_runner --selftest [8]·시뮬 X6(배포 코드 폐루프) |
| 6 합성 | KE → W3 인계 | – | – | – | – | ○ | kew3_runner --selftest [7]·시뮬 X1/X6 |
| 6 합성 | 인계 뒤 KE 상태 | – | – | – | – | ○ | kew3_runner --selftest [7] |
| 6 합성 | 회복 모드 중 걷기 켜기 | – | – | – | – | ○ | kew3_runner --selftest [4] |
| 6 합성 | 서기 쪽 KE 변형 | – | – | – | – | ○ | 없음 |
| 6 합성 | 서기·회복(KE) 하위 키 러너 프로브 | – | – | – | – | ○ | 브리지 selftest 'KE↔W3 합성' |
| 6 합성 | 걷기(W3) 하위 키 러너 프로브 | – | – | – | – | ○ | 브리지 selftest 'KE↔W3 합성' |
| 6 합성 | 노드 세션(가짜 ROS 로 main() 그대로, HEAD) | – | – | – | – | ○ | selftest_w3_bridge_node(스텁 정책) |
| 7 젯슨 판 | 재구성한 파일 = 저장소가 기록한 배포본 | ○ | ○ | ○ | – | – | deploy_c5krh_1007.sh(md5 확인 절차) |
| 7 젯슨 판 | 레지스트리·상수 값 HEAD 와 같은가 | ○ | ○ | ○ | – | – | 브리지 selftest 'C5 krh 키'(kr 대비) |
| 7 젯슨 판 | 러너 프로브(600 틱) HEAD 와 비트 비교 | ○ | ○ | ○ | – | – | selftest_odic_equiv [A]·[C](기준 커밋 비트 비교, 다른 기준) |
| 7 젯슨 판 | 노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교 | ○ | ○ | ○ | – | – | selftest_w3_bridge_node(기준 커밋 비교, 다른 기준) |
| 7 젯슨 판 | 실물 블랙박스 재생 — real_1003(젯슨이 실제로 낸 값과 대조) | ○ | – | – | – | – | reports/gen_c5/replay_kr*.py(이전 로그, 감지기만) |
| 7 젯슨 판 | 실물 기록의 배포 체인 값 — real_1003 | ○ | – | – | – | – | 없음 |
| 7 젯슨 판 | HEAD 에만 있는 선택 기능 | ○ | ○ | ○ | – | – | 없음 |
| 7 젯슨 판 | 실물 블랙박스 재생 — real_1007(젯슨이 실제로 낸 값과 대조) | – | – | ○ | – | – | reports/gen_c5/replay_kr*.py(이전 로그, 감지기만) |
| 7 젯슨 판 | 실물 기록의 배포 체인 값 — real_1007 | – | – | ○ | – | – | 없음 |

## 5. 기존 검증 — 재실행 결과와 빈칸

| 기존 검증 | 무엇을 고정하나 | 종료 코드 | [PASS] / [FAIL] 줄 수(그 표기를 쓰는 검증만) | 마지막 판정 줄 | 시간 |
|---|---|---|---|---|---|
| 러너 --selftest | 245 골든 조립(항 블록·이력 오래된 것 먼저·첫 틱 백필)·클록 식·슬루·직전 행동 = 원시 | 0 | 10 / 0 | selftest: ALL PASS | 1 s |
| deploy_chain --selftest | 러너 슬루 → LPF → 블렌딩(벡터화 = 스칼라 참조)·순서 감수성 | 0 | — | [selftest] deploy_chain: 벡터화=참조(4구성×3env×220틱, 리셋 포함) · 슬루 ≤4°/틱 · 첫 틱 α=0 · 순서 감수성 9.18° (슬루→LPF vs LPF→슬루, 28.6° 스텝 10틱) — ALL PASS | 1 s |
| c5_switch --selftest | 감지기 히스테리시스·조기 복귀·불응기·cue·odic·갇힘 해소(배포 구현 내부 일관성) | 0 | — | C5 selftest PASS | 1 s |
| kew3_runner --selftest | 합성 러너 전환 틱·이어받기·정착 대기·계약 불일치 거부(스텁 정책) | 0 | — | [selftest] kew3_runner: KE 모드 = 단독 KE(60 틱 비트 동일) · 켜기 틱 W3 fresh(ge·게이트 즉시, 이력 백필) = 단독 W3(80 틱) · 슬루 기준 이어받기(qpos 28.6° 떨어져도 전환 틱 ¦Δ목표¦ ≤ 4°) · 끄기 = 반주기 래치(13 | 0 s |
| 브리지 --selftest | 30 항목: '학습 계약'(기본자세·스케일·슬루·50 Hz·default_offset, 서기 run)·'관측 필터 계약'·C5·W3·합성 등록·정책 500 틱 | 0 | 30 / 0 | selftest: ALL PASS | 5 s |
| 브리지 --selftest(합성 키 기동 인자) | 같은 30 항목 + 합성·하위 키를 필수로 | 0 | 30 / 0 | selftest: ALL PASS | 5 s |
| selftest_w3_equiv | W3 클록·래치·리드 발·헤딩 고리 = 시뮬 소스 실행([B]·[E]·[F]), 플래그 꺼짐 비트 동일([A]·[C]) | 0 | 6 / 0 | w3 equiv: ALL PASS | 4 s |
| selftest_w3_rollout_parity | W3S3d 시뮬 롤아웃 기록 원천 → 러너·브리지 코어 관측 320·행동·체인([P1]–[P6]) | 0 | — | w3 rollout parity: ALL PASS | 216 s |
| selftest_odic_equiv | 감지기 기본(imu) 경로 기준 커밋 비트 동일([A]·[C])·odic = 시뮬 평가 소스 실행([B]) | 0 | 8 / 0 | odic equiv: ALL PASS | 115 s |
| selftest_c5_imu_f32 --ref 6260e281~1 | KE imu 조기 복귀: 시뮬 평가 소스 실행 vs 배포(float32 옵션 0 불일치·기본 float64 는 −1 틱) | 0 | 3 / 0 | c5 imu_f32: ALL PASS | 4 s |
| selftest_w3_bridge_node | 가짜 ROS 로 노드 main(): W3·합성 세션 명령·감독기·지연 동결, 플래그 꺼짐 발행 = 기준 커밋(스텁 정책) | 0 | 14 / 0 | w3 node selftest: ALL PASS | 6 s |

재실행 결과: **전부 종료 코드 0**.

문서·평가 형태의 기존 검증(여기서 재실행하지 않음): `reports/gen_w3/PARITY_AUDIT_1007.md`(W3S3d·KE 26 항목 + 폐루프 영향 평가), `reports/audits/deploy_contract_0905_summary.md`(stand_v18 세대 37 건), `rl_walking/deploy/selftest_w3s3d_onnx.py`(W3S3d onnx 대 pt — Isaac 파이썬 필요), `rl_walking/deploy/sim_validate_runner_stage4.py`(Isaac 필요 — 관절 순서 단언), TRANSITION_W3KE §5.10c X6(배포 코드 합성 러너를 시뮬 폐루프에 넣은 평가).

**기존 검증의 빈칸** — 1–8 은 이 감사가 메웠고 9–12 는 남아 있다(시뮬 평가가 있어야 한다):

1. 관측 클립 — 어떤 기존 검증도 다루지 않는다(이 감사가 처음: 1 절의 '클립').
2. 서기·회복(G21a·G24Ra) 관측 320 의 시뮬 대조 — 롤아웃 parity([P1]–[P4])는 W3S3d 만이다. 시간식 클록(1.1 Hz)·C5 공유 조립 경로는 시뮬 원천 대조가 없었다 → 이 감사: 세 run 관측 설정 동일성 + 시뮬 소스 실행 + 학습 계약 재구성 + 실물 재생.
3. 회복 정책(G24Ra) run 의 env.yaml 자동 검사 — 브리지 '학습 계약'은 서기 run(G21a)만 읽는다(C5 키의 경로 = 서기 정책). 종전에는 주석의 수동 확인뿐.
4. G21a·G24Ra 의 policy.onnx(젯슨 추론 경로) — onnx 대 pt 시험은 W3S3d 만 있고 Isaac 파이썬이 필요하다 → 이 감사: 세 run 모두 체크포인트 = pt = onnx 가중치 비트 비교(패키지 불필요) + 실물 기록의 원시 행동 재생.
5. 감지기 설정값 ↔ 시뮬 평가 인자의 자동 대조 — 종전에는 브리지 주석으로만 대응시켰다 → 이 감사: 평가 산출물 머리의 '인자:' 줄과 실제 감지기 객체를 대조.
6. 젯슨 판 ↔ HEAD 직접 비교 — 기존 비트 비교들은 기준 커밋이 제각각(68493c11·95946e41·b79370ca·f44e8701 …)이고 젯슨 판을 기준으로 삼은 것이 없다.
7. 실물 블랙박스로 관측·행동 전체를 재생하는 상시 도구 — 감지기 재생(reports/gen_c5/replay_kr*.py)만 있었다.
8. 노드(_tick) 수준의 LPF·블렌딩·유한차분을 실모델로 재구성하는 시험 — 스텁 정책의 기준 커밋 비트 비교만 있었다.
9. stage8 식 전 관절 soft 클램프·25° 거부가 들어간 **KE 회복** 시뮬 평가 — W3 걷기만 있다(PARITY §4.4). 실물 1007 에서는 회복 중 힙이 잘렸다(1 절의 '소프트 한계 클램프').
10. W3S3d 의 지연 0 틱 평가 — 학습 1–5, 평가 1–7 틱. 실물 명령 전송 구간은 한 틱 안으로 측정됐다(2 절의 '행동 지연').
11. KE·전환 평가의 자이로 지연 — 0 틱 조건뿐(2 절).
12. 합성 러너 + 헤딩 고리의 폐루프 평가 — 없음(KEW3_DESIGN §8.3 에 이미 적힘).

## 6. 감사 방법과 감사 자체의 민감도

| 단계 | 무엇으로 | 덮는 것 |
|---|---|---|
| 학습 값 추출 | env.yaml·agent.yaml(PyYAML), 체크포인트 가중치, tfevents 커리큘럼 스칼라, 시뮬 소스 AST | 관측 항·이력·클립·필터 인자, 행동 스케일·슬루·지연·LPF·온셋 대역, 명령 범위·커리큘럼 도달값, W3 위상 인자 |
| 학습 시점 소스 확인 | run 폴더 시각 → 그때의 브랜치 커밋, run 이 남긴 git status·diff, 계약 함수(관측 항 함수·행동 항 클래스·한계/위상 이벤트와 보조 정의)의 소스를 현재와 비교 | '학습 값' 칸의 소스 근거가 학습 때와 같은지 |
| 시뮬 소스 실행 | `_ema_state`·`last_action_lpf`·`encoder_joint_vel_lpf`·`gait_clock`·`onset_alpha` 를 **학습 시점 커밋의 소스 그대로** torch CPU 로 | 필터 시작 규칙·계수, 클록 식, 블렌딩 식 |
| 러너 프로브 | `_make_runner(키)` 실모델에 합성 입력 600 틱(기울기 펄스 3 회·한발 구간) + 클립·슬루 기준 프로브 | 관측 320 재구성, β·α·주파수·스케일·기본자세·슬루 역산, 직전 행동 정의, 클립 유무, W3 래치·리드 발, 합성 인계 규칙 |
| 노드 세션 | 가짜 rclpy 로 `rl_bridge_node.main()` 그대로(--live, 발행은 기록만, HOME 은 임시 폴더) | 유한차분 속도·LPF·블렌딩·모터 프레임·발행, W3 명령·헤딩 고리, 합성 인계 |
| 젯슨 판 | `git archive 8af2f6e5` + 후보 브리지로 임시 트리, 같은 워커 | HEAD 대 젯슨 판 비트 비교, 레지스트리·상수 |
| 실물 재생 | 젯슨 블랙박스의 입력을 BridgeCore 에 넣고 기록된 관측·원시 행동·목표·회복 모드와 대조 | 젯슨이 실제로 낸 값 = 재구성한 코드의 값 |

**민감도 시험**(`contract_audit_mutants.py`): 배포 코드 사본에 결함을 하나씩 심어 감사가 그 항목을 불일치로 바꾸는지 본다 — **변이 26 종 중 26 통과 / 0 실패**(종료 코드 0).

- 대조군(변이 없는 사본의 판정 = 저장소 직접 감사의 판정이어야 한다 — 다르면 사본 과정이 판정을 바꾼 것): **같음**(행 364·364, `--allow-known` 종료 코드 0·0)
- 통과 조건: 변이마다 ① '잡혀야 하는 항목'이 대조군에서는 불일치가 아니었고 변이에서는 불일치이고 ② 회귀 관문(`--allow-known`)의 종료 코드가 0(대조군) → 1(변이)로 바뀌어야 한다. '새 불일치 줄 수'는 같은 결함이 다른 줄(전체 재구성·비트 비교·실물 재생)에서도 잡힌 것까지 센 수다.
- '기지 줄의 변화'는 종전부터 불일치이던 줄이 그 변이에서 다른 판정으로 바뀐 것이다 — 결함 때문에 그 줄의 재구성이 성립하지 않으면 '확인 불가'가 된다(`cue_vx_005` 만 값이 학습 범위 안으로 들어가 '일치'가 된다).

| 결과 | 변이 | 심은 결함 | 잡혀야 하는 항목 → 실제로 잡힌 항목 | 새 불일치 줄 수 | 관문 종료 코드 | 기지 줄의 변화 |
|---|---|---|---|---|---|---|
| 통과 | `act_obs_beta_030` | 관측 직전 행동 EMA β 0.27 → 0.30 | actions(직전 행동) → actions(직전 행동) | 12 | 1 | — |
| 통과 | `jvel_obs_beta_05` | 관측 관절속도 EMA β 0.25 → 0.5 | joint_vel = 엔코더 유한차분 속도의 EMA → joint_vel = 엔코더 유한차분 속도의 EMA | 11 | 1 | 관절속도 원천(유한차분 식) → 확인 불가 |
| 통과 | `lpf_beta_045` | 액션 LPF β 0.27 → 0.45(학습 대역 0.2–0.4 밖) | 액션 LPF β → 액션 LPF β | 9 | 1 | — |
| 통과 | `blend_2s` | 온셋 블렌딩 1.5 → 2.0 s(학습 대역 1.0–1.5 밖) | 온셋 블렌딩(인계) → 온셋 블렌딩(인계) | 5 | 1 | — |
| 통과 | `gait_freq_14` | G21a 클록 1.1 → 1.4 Hz | gait_clock(서기 정책) → gait_clock(서기 정책) | 8 | 1 | — |
| 통과 | `clock_one_tick_ahead` | 클록 위상 한 틱 앞섬 | gait_clock(서기 정책) → gait_clock(서기 정책) | 7 | 1 | — |
| 통과 | `default_pose_hip` | 기본자세 힙 0.5282 → 0.5182 rad(0.57°) | joint_pos = 엔코더각 − 기본자세 / 오프셋 = 기본자세 → joint_pos = 엔코더각 − 기본자세 / 오프셋 = 기본자세 | 14 | 1 | LPF 초기값 → 확인 불가 |
| 통과 | `history_reversed` | 이력 배열을 최신 먼저로 | 이력 배열 순서 → 이력 배열 순서 | 17 | 1 | 관절속도 원천(유한차분 식) → 확인 불가 |
| 통과 | `term_order_swap` | 관측 끝 두 항 순서 바꿈 | 항 목록·순서 → 항 목록·순서 | 11 | 1 | — |
| 통과 | `contact_lr_swap` | 접촉 좌우 뒤바꿈 | feet_contact 순서 → feet_contact 순서 | 7 | 1 | — |
| 통과 | `ema_alpha_02` | S8a EMA α 0.1 → 0.2 | gyro_ema 계수 / joint_vel_ema 계수 → gyro_ema 계수 / joint_vel_ema 계수 | 10 | 1 | — |
| 통과 | `prev_action_emitted` | 직전 행동을 슬루 뒤 목표(행동 단위)로 되먹임 | actions(직전 행동) → actions(직전 행동) | 5 | 1 | — |
| 통과 | `scale_020` | 행동 스케일 0.25 → 0.20 | 스케일 → 스케일 | 13 | 1 | LPF 초기값 → 확인 불가 |
| 통과 | `slew_3deg` | 슬루 4 → 3°/틱 | 슬루 상한 → 슬루 상한 | 15 | 1 | LPF 초기값 → 확인 불가 |
| 통과 | `obs_clip_added` | (반대 방향 대조) 자이로 관측에 학습과 같은 ±20 클립을 넣음 — '클립' 줄의 통과 값이 20 으로 바뀌어야 한다 | '클립' 줄의 배포 값이 바뀜(판정은 그대로 불일치 — 6 항 중 1 항만 고친 것) → 클립 (반대 방향 대조) | 0 | 0 | — |
| 통과 | `det_on_tilt_35` | 감지기 켜기 문턱 3.0 → 3.5° | 켜기 기울기 편차 → 켜기 기울기 편차 | 3 | 1 | — |
| 통과 | `det_early_hold_05` | krh 조기 복귀 유지 0.7 → 0.5 s | 조기 복귀 (최소 s → 조기 복귀 (최소 s | 5 | 1 | — |
| 통과 | `det_in_filt_05` | 감지 입력 EMA 0.33 → 0.5 | 감지 입력 EMA α → 감지 입력 EMA α | 6 | 1 | — |
| 통과 | `cue_vx_005` | 회복 cue vx 0.03 → 0.05(평가와 달라짐 — 반대로 학습 범위 안으로는 들어온다) | 회복 cue → 회복 cue | 6 | 1 | 명령 관측(회복 정책) → 일치 |
| 통과 | `w3_fstep_12` | W3 스텝 주파수 1.1 → 1.2 Hz | 스텝 주파수 f_step → 스텝 주파수 f_step | 2 | 1 | — |
| 통과 | `w3_ramp_04` | W3 게이트 램프 0.2 → 0.4 s | 관측 게이트 램프 → 관측 게이트 램프 | 1 | 1 | — |
| 통과 | `w3_lead_wz` | 리드 발 wz 문턱 0.08 → 0.25(−0.15 회전이 왼발로 시작) | 리드 발 규칙 → 리드 발 규칙 | 1 | 1 | — |
| 통과 | `w3_vy_limit` | 옆걸음 상한 0.04 → 0.08(학습 밖) | 명령 관측 범위(걷기) → 명령 관측 범위(걷기) | 1 | 1 | — |
| 통과 | `w3_enable_tilt` | 켜기 기울기 상한 7 → 9° | 걷기 켜기 선행조건 → 걷기 켜기 선행조건 | 1 | 1 | — |
| 통과 | `kew3_settle_02` | 합성 정착 0.5 → 0.2 s | W3 → KE 인계 규칙 → W3 → KE 인계 규칙 | 1 | 1 | — |
| 통과 | `kew3_settle_w` | 합성 정착 ¦ω¦ 문턱 0.15 → 0.6(시끄러워도 인계) | W3 → KE 인계 규칙 → W3 → KE 인계 규칙 | 2 | 1 | — |

이 시험이 보이는 것은 '이 26 가지 결함은 잡는다'까지다 — 목록에 없는 종류의 결함(예: 현장 센서 축, stage8 노드 안쪽)은 7 절의 한계에 적었다.

## 7. 한계

- **젯슨의 현재 파일·프로세스를 읽지 않았다**(접속 금지). '젯슨 판'은 저장소의 배포 기록(커밋 8af2f6e5, 후보 브리지, md5 매니페스트)으로 재구성한 것이다. 실물 블랙박스 재생이 그 재구성과 맞는 것은 1003(G21a)·1007(krh) 두 세션에 한한다 — 저장소에 c5kr 키의 실물 기록은 없다(1004·1005 의 기록 6 개는 앞선 키 stand_c5_g21a_g24ra 의 것이고 이 감사의 대상 키가 아니어서 재생하지 않았다).
- 젯슨의 **stage8 노드 판**은 확인 범위 밖이다(지시서는 브리지·러너만 지정). HEAD 와 커밋 8af2f6e5 의 stage8 소스에서 슬루·주기·클램프·거부 규칙이 같음을 읽었고, 실물 1007 기록의 클램프 누계가 그 규칙(soft 초과 틱마다 한 번)과 맞는 것만 확인했다.
- 민감도 시험은 **배포 쪽** 결함만 심는다. 학습 쪽 값을 잘못 읽는 종류의 감사 결함은 '일치' 줄이 실제 러너 값과 맞아떨어지는 것으로만 간접 확인된다.
- 열린 고리 계약 검증이다. 같은 입력 → 같은 관측·행동임을 보였을 뿐, 실물 센서의 분포(잡음·지연·패드 오판)가 학습 분포 안인지와 폐루프 성능은 말하지 못한다.
- 학습 쪽 관절 순서·이력 배열 순서의 근거는 W3S3d 태스크에서 남긴 시뮬 기록(log_picture/w3parity_rec_*.npz) 하나다. G21a·G24Ra 는 같은 로봇 자산(usd_path)·같은 관측 설정이라는 점으로 잇는다 — 그 두 태스크에서 직접 남긴 시뮬 기록은 없다.
- 노드 세션의 플랜트는 '발행 목표를 1 차 지연으로 따라가는' 가정 모형이고 접촉은 항상 양발이다. LPF·블렌딩·유한차분·발행 식의 재구성에만 쓴다.
- 학습 쪽 값 가운데 env.yaml 에 없는 것(필터 함수의 시작 규칙, 행동 체인 순서, 소프트 클램프 관절 기본값, w3_gait_update 의 ramp_s·리드 발 상수)은 소스에서 읽는다. 기준은 **학습 시작 시점 커밋**이다 — 단 rsl_rl 은 커밋 해시를 남기지 않아 run 폴더 이름의 시각 직전 커밋으로 추정했다(run 이 남긴 git status·diff 에 rl_walking 변경이 없음은 확인). 그 커밋 이후 바뀐 계약 함수는 '학습 시점의 시뮬 소스' 줄에 적었고, 차이를 읽어 확인한 것만 통과시킨다(함수가 다시 바뀌면 '미검토'로 떨어진다).
- 실물 블랙박스는 관측 4 자리·쿼터니언 5 자리로 반올림돼 있다. 재생 허용오차(관측 2e-4·원시 행동 2e-3)보다 작은 계약 차이는 이 재생으로 보이지 않는다(그런 차이는 프로브·비트 비교가 맡는다).
- 젯슨 추론 백엔드(onnxruntime / torch)는 블랙박스에 기록되지 않는다. 가중치 비트 비교와 원시 행동 재생으로 어느 쪽이든 같은 출력임을 보였다.
- 평가 산출물·시뮬 기록·실물 블랙박스는 저장소의 `log_picture/`·`reports/real_*` 파일에 기댄다(워크트리에 없으면 주 체크아웃에서 읽는다). 파일이 없으면 해당 줄은 '확인 불가'로 떨어진다.

## 부록 A. 정책별 표

칸: 분류 · 항목 · 학습 값 · 배포 값(HEAD) · 배포 값(젯슨 판) · 판정. 기준 정책 표와 완전히 같은 줄은 생략했다. 비고·기존 검증 칸까지 있는 전체 표는 스크립트를 인자 없이 실행하면 나온다.

### stand_crouch_g21a

일치 42 · 불일치 7 · 확인 불가 5

| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| 0 정책 파일 | 서기 정책 가중치 = 학습 체크포인트 | 2026-10-02_21-23-08_g21a/model_11300.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | policy.pt = 매니페스트; policy.onnx = 매니페스트; env.yaml = 매니페스트 | 일치 |
| 0 정책 파일 | 서기 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | 같은 파일 | 일치 |
| 0 정책 파일 | 서기 학습 시점의 시뮬 소스(계약 함수) | 커밋 190c03b9(2026-10-02 18:48:49) = run 시작 2026-10-02 21:23:08 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 19 개 같음 · 바뀐 것: SlewJointPositionAction(+15/−10 줄 — 지연 표본에 가중 옵션(delay_probs) 추가 — 비어 있으면 종전 균등 표본); _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | (학습 쪽 항목) | 일치 |
| 0 정책 파일 | 서기 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_11300.pt | SOURCE_CHECKPOINT model_11300.pt | 같은 파일 | 일치 |
| 1 관측 | 항 목록·순서·프레임 차원 | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | 일치 |
| 1 관측 | 이력 길이(항별 프레임 수) | [5] | [5] | [5] | 일치 |
| 1 관측 | 이력 배열 순서·리셋 첫 틱 | 오래된 것 먼저, 첫 틱 전 칸 백필(시뮬 기록 실측: 순서 True·백필 True) | 최신 = 마지막 칸 True · 첫 틱 백필 True(프로브) | 프로브 비트 동일 | 일치 |
| 1 관측 | 스케일 | 없음(전 항 scale: null) | 스케일 없음(프로브: 관측 = 입력 True) | 프로브 비트 동일 | 일치 |
| 1 관측 | 클립 | base_ang_vel ±20; joint_pos ±10; joint_vel ±60; actions ±10; gyro_ema ±20; joint_vel_ema ±60 | 클립 없음(프로브: base_ang_vel 100, joint_vel 200, joint_pos 50, gyro_ema 100, joint_vel_ema 200 그대로 통과) | 같음(프로브 비트 동일) | 불일치 [신규] |
| 1 관측 | base_ang_vel(자이로) 처리 | base_ang_vel_lagged(몸통 각속도 rad/s, 지연 0–3 틱 DR) + 잡음 | IMU gyro_rad_s 그대로(프로브 ¦Δ¦ 0) | 비트 동일 | 일치 |
| 1 관측 | IMU 축 정렬(몸통 프레임 +x 앞·+z 위) | 시뮬 몸통(base) 프레임 | IMU 어댑터가 낸 gyro·quat 을 그대로 사용 — 축 부호는 현장 확인 기록(0912–0914)에 기댄다 | 같음 | 확인 불가 |
| 1 관측 | projected_gravity 정의 | 쿼터니언으로 (0,0,−1) 을 몸통 프레임에(단위벡터) + 장착 편향 DR 피치 (-0.026, 0.026)·롤 (-0.026, 0.026) rad | projected_gravity(q): 독립 회전행렬 식과 최대 ¦Δ¦ 0 · 맵 편향 보정 [1.3, -0.6]° 식 오차 3.6e-15° | ¦Δ¦ 0 · 보정 [1.3, -0.6]° | 일치 |
| 1 관측 | 명령 관측(서기 정책) | (0,0,0) 고정 | stand 가족: 명령 (0,0,0)·heading_hold 강제 끔(노드 세션 관측 명령 최대 ¦값¦ 0) | 같음(노드 발행 비트 동일) | 일치 |
| 1 관측 | joint_pos = 엔코더각 − 기본자세 | encoder_joint_pos_rel_v2; 기본자세(init_state, deg) +30.26 -30.26 +0.00 +0.00 +0.00 +0.00 -38.50 +38.50 +28.00 -28.00 +0.00 +0.00 | 러너 default_pose 와 최대 ¦Δ¦ 2.9e-08 rad · 관측에서 역산한 기본자세와 2.9e-08 rad | 최대 ¦Δ¦ 2.9e-08 rad | 일치 |
| 1 관측 | 엔코더 관측 방식 | encoder_joint_pos_rel_v2·encoder_joint_vel_lpf: 로터(모터)측 각 — 유격 안 링크 이동·직렬 처짐은 관측에 안 보임 | 모터 엔코더 피드백(로터측)을 관절 프레임으로 변환(DIR × (모터 deg − 영점))해 그대로 사용 | 같음 | 일치 |
| 1 관측 | joint_vel = 엔코더 유한차분 속도의 EMA | encoder_joint_vel_lpf β 0.25(에피소드 첫 틱 = 원값) | 프로브 역산 β 0.25 · 시뮬 소스 실행 대조 최대 ¦Δ¦ 0 | 0.25 | 일치 |
| 1 관측 | 관절속도 원천(유한차분 식) | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 2.7e-07; 단순 차분으로 재구성하면 0.48) | 같은 식(노드 발행 비트 동일) | 불일치 [기지] |
| 1 관측 | actions(직전 행동) 정의·필터 | last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 6.1e-08 vs 슬루 뒤 값을 넣었다면 0.37; 시뮬 소스 실행 대조 1.5e-07 | 0.27 | 일치 |
| 1 관측 | gait_clock(서기 정책) | gait_clock: [sin φ, cos φ], φ = 2π·1.1·(틱·dt), 리셋 = 위상 0 | 프로브 위상 기울기 1.1 Hz · 첫 틱 (0, 1) · 시뮬 소스 실행 대조 6e-08 · 1 시간 뒤 float32 위상 오차 0.00025 | 1.1 | 일치 |
| 1 관측 | feet_contact 순서·값 | [left_ankle_r_joint, right_ankle_r_joint] (preserve_order True) 0/1 | contact2 [왼, 오] 0/1 그대로(프로브 ¦Δ¦ 0); 0/1 이외 값은 러너가 거부 | 비트 동일 | 일치 |
| 1 관측 | 접촉 0/1 의 물리 정의 | feet_contact_real_v2: 발 법선력 > U(2, 20) N ∧ 발바닥 기울기 < U(6, 20)° → 5 틱마다 갱신 + 1 틱 지연 | 어댑터: 패드 4 개 중 하나라도 (카운트 − 바이어스) > 200 → 1 (미보정 카운트, 갱신 100 ms). 브리지는 비트를 그대로 쓴다 | 같음 | 확인 불가 |
| 1 관측 | gyro_ema 계수 | gyro_ema_lagged α 0.1 | 프로브 역산 α 0.1 · 정상 구간 재구성 오차 0 | 프로브 비트 동일 | 일치 |
| 1 관측 | gyro_ema 시작 규칙 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 0.513, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | 틱 0 만(옵션 없음) | 불일치 [기지] |
| 1 관측 | joint_vel_ema 계수 | encoder_joint_vel_ema_v2 α 0.1(원시 속도에서 — v2) | 프로브 역산 α 0.1 · 모드 v2 · 정상 구간 재구성 오차 1.5e-08 | v2 | 일치 |
| 1 관측 | joint_vel_ema 시작 규칙 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 1.61, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | 틱 0 만(옵션 없음) | 불일치 [기지] |
| 1 관측 | 관측 벡터 전체 재구성(600 틱) | 학습 계약으로 다시 조립한 값(항 순서·이력·필터·기본자세) | 실제 러너 관측과 최대 ¦Δ¦ 0(base_ang_vel); S8a EMA 시작 규칙만 배포식으로 둔 재구성 | HEAD 와 관측·목표 비트 동일 | 일치 |
| 2 행동 | 관절 순서(12) | 시뮬 기록 joint_names: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | JOINT_ORDER: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | 같음 | 일치 |
| 2 행동 | 차원·행동 클립 | actor 출력 12 · actions.clip None · clip_actions None | 러너 출력 12 · 클립 없음(프로브 원시 행동 최대 ¦값¦ 2.19 가 그대로 목표식에 들어간다 — 재구성 오차는 아래) | 같음 | 일치 |
| 2 행동 | 스케일 | 0.25 | 프로브 역산 0.25(관절 12/12, 포화 안 된 틱 88%) | 0.25 | 일치 |
| 2 행동 | 오프셋 = 기본자세(use_default_offset) | use_default_offset True · 명목 기본자세(위 joint_pos 줄) | 프로브 역산 오프셋과 학습 기본자세 최대 ¦Δ¦ 2.9e-08 rad | 목표 비트 동일 | 일치 |
| 2 행동 | 슬루 상한(틱당) | 4.0000° | 프로브 최대 틱 변화 4.0000° · 1 rad 떨어진 첫 틱 이동 4.0000° | 4.0000° | 일치 |
| 2 행동 | 슬루 기준 초기화 | 리셋 틱의 현재 관절각(_needs_init → joint_pos) | 첫 step 의 qpos(프로브: 전 관절이 qpos 에서 정확히 슬루 한 번 True) | 목표 비트 동일 | 일치 |
| 2 행동 | 목표각 재구성(600 틱) | 기본자세 + 스케일·원시 → 슬루(학습 식) | 실제 러너 출력과 최대 ¦Δ¦ 0 rad · 타깃 클램프 없음 | HEAD 와 비트 동일 | 일치 |
| 2 행동 | 처리 순서 | 지연 → 슬루 → LPF → 온셋 블렌딩 → 백래시 → 소프트 클램프 | 러너 슬루 → 브리지 LPF → (보정기 꺼짐) → 온셋 블렌딩 → 전송 → stage8 슬루·클램프; 노드 세션 재구성 오차 1.3e-07 rad·블렌딩 0° | 노드 발행 비트 동일 | 일치 |
| 2 행동 | 액션 LPF β | env 별 U(0.2, 0.4) | STAND_LPF_BETA 0.27 · 노드 세션 역산 0.27(키가 stand 로 시작 → 적용 True) | 0.27 | 일치 |
| 2 행동 | LPF 초기값 | 리셋 틱의 현재 관절각 | 첫 슬루 출력(노드 세션: 첫 발행 − 첫 슬루 출력 = 3.5e-18 rad) | 같음 | 불일치 [기지] |
| 2 행동 | 온셋 블렌딩(인계) | 유지 U(0, 0.3) s + 스무스스텝 U(1, 1.5) s | 유지 0 s + 스무스스텝 BLEND_S 1.5 s — onset_alpha(소스 실행) vs blend_alpha 최대 ¦Δ¦ 6e-08; 노드 세션 α(t) 오차 0.00049(기록 3 자리 반올림) | 1.5 | 일치 |
| 2 행동 | 블렌딩 시작점 | env 기본자세(편향 DR 포함) 홀드 목표 | 인계 순간 stage8 이 스트리밍하던 명령(노드 세션: 첫 발행 − 가짜 stage8 명령 = 0°) | 같음 | 확인 불가 |
| 2 행동 | 소프트 한계 클램프(자르는 관절) | clamp_to_soft_limits True: config soft 를 '.*_ankle_f_joint' 관절만(그 밖은 물리 한계까지 통과) · 한계 파일 마지막 변경 2026-09-18(모든 run 학습 시작 전 — 학습과 실물이 같은 표를 읽었다) | stage8 노드: **전 관절**을 config soft 경계로 자른다(러너·브리지는 자르지 않음) | 같음(stage8 — 8af2f6e5 판도 같은 규칙) | 불일치 [기지] |
| 2 행동 | 한계 초과 명령의 거부 | 없음(클램프만 — 목표가 아무리 넘어도 경계값으로 계속 구동) | stage8: 경계를 25° 넘게 넘으면 그 관절 명령을 거부(limit_runaway) | 같음(stage8) | 불일치 [기지] |
| 2 행동 | 노드(stage8) 슬루·주기 | 슬루는 한 번(4.0°/틱) | 러너 슬루 뒤 stage8 이 한 번 더: 4.0°/틱 @ 0.02 s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 3.901841° | 8af2f6e5 판 stage8 상수 같음(젯슨에 어느 판이 있는지는 미확인) | 일치 |
| 2 행동 | PD 게인(kp/kd) | [(150.0, 5.0)] · DR ×(0.85, 1.1)/×(0.8, 1.2) | POLICY_KP 150.0/KD 5.0 · 맵 불일치 관절 0(--live 는 불일치면 기동 거부) | KP 150.0/KD 5.0 · 젯슨 맵 사본(1005) 게인 같음 | 일치 |
| 2 행동 | 학습에 없는 가산 보정(roll/pitch comp) | 없음 | 맵 roll_comp False · pitch_comp False | 젯슨 맵 사본(1005)도 꺼짐 | 일치 |
| 3 시간 | 제어 주기 | 20.0 ms(sim.dt × decimation 4) | TICK_SEC 20.0 ms · 러너 POLICY_DT 20.0 ms · 실물 1007 로그 틱 간격 평균 20.00 ms(1–99 % 19.9–20.1, 최대 23.3) | 20.0 ms | 일치 |
| 3 시간 | 행동 지연 | U{0..3} 틱 | 인위 지연 없음 — 실물 루프 지연이 그 몫(총 지연 미측정). 실물 1007 로그의 구간 실측: 브리지 발행 → stage8 처리 중앙 5.2 ms(99 % 9.9, 최대 12.9) · status 나이 중앙 6 ms(최대 10) | 같음(위 실측이 젯슨 세션 값) | 확인 불가 |
| 3 시간 | 접촉 갱신 주기 | 5 틱마다 갱신(feet_contact hold_ticks) — 그 사이는 직전 비트 유지 | 브리지는 틱마다 최신 수신 비트를 읽는다. 실물 로그에서 발 원카운트 합이 바뀌는 간격: real_1007 최빈 5 틱(353/354 구간, 더 긴 구간 1 개·최대 10 틱); real_1003 최빈 5 틱(2872/2875 구간, 더 긴 구간 3 개·최대 10 틱) | 같은 실측(젯슨 세션 기록) | 일치 |
| 3 시간 | 센서 관측 지연(IMU·접촉) | 자이로 0–3 틱 · 자세(중력) (0, 3) 틱 · 접촉 5 틱 유지 + 1 틱 | 브리지는 최신 수신값을 그대로 쓴다(신선도 한계 IMU·관절 0.2 s, 발 0.35 s). 센서 안쪽·전송 지연은 미측정. 브리지에서 본 것: IMU 표본이 직전 틱과 같은 틱 real_1007 0.39%; real_1003 0.03%(거의 매 틱 새 표본) | 같음 | 확인 불가 |
| 7 젯슨 판 | 재구성한 파일 = 저장소가 기록한 배포본 | — | (HEAD 는 젯슨 미배포 변경 다수) | 브리지 md5 15dcabf2 = c5krh_1007.md5 · 러너 21b2416e = 매니페스트 · c5_switch 11cf9641 = · 8af2f6e5 + 패치 = 후보 True | 일치 |
| 7 젯슨 판 | 레지스트리·상수 값 HEAD 와 같은가 | — | 비교 29 항목 | 전부 같음 | 일치 |
| 7 젯슨 판 | 러너 프로브(600 틱) HEAD 와 비트 비교 | — | 기준 | 관측 True · 원시 행동 True · 목표각 True · 회복 모드열 True | 일치 |
| 7 젯슨 판 | 노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교 | — | 정책 틱 340 | 발행 목표 True · 관측 True | 일치 |
| 7 젯슨 판 | 실물 블랙박스 재생 — real_1003(젯슨이 실제로 낸 값과 대조) | 기록 15299 틱(306.0 s), 기록 반올림 1e-4 — 그때의 젯슨 판(현 판 이전)이 낸 기록 | 관측 최대 ¦Δ¦ 9e-05 · 원시 행동 0.00039 · LPF 뒤 목표 0.0037° · 15299 틱 | 관측 최대 ¦Δ¦ 9e-05 · 원시 행동 0.00039 · LPF 뒤 목표 0.0037° · 15299 틱 | 일치 |
| 7 젯슨 판 | 실물 기록의 배포 체인 값 — real_1003 | 학습 온셋 블렌딩 상한 1.5 s · 관측 필터 β (0.27, 0.25) | — | 블렌딩 α(t) 가장 잘 맞는 길이 1.5 s(오차 0.00049) · 기록 obs_filter [0.27, 0.25] · jvel_ema v2 · 장착 편향 [1.3, -0.6]° · 명령 최대 ¦값¦ 0.0 · 틱 간격 평균 20.00 ms(최대 24.3) | 일치 |
| 7 젯슨 판 | HEAD 에만 있는 선택 기능 | — | --runner-ema-fresh | 없음(기본 꺼짐 기능이라 기본 동작은 위 비트 비교대로 같다) | 일치 |

### stand_c5kr_g21a_g24ra

일치 63 · 불일치 9 · 확인 불가 6 — 아래는 stand_crouch_g21a 표와 다른 33 줄(같은 45 줄 생략)

| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| 0 정책 파일 | 회복 정책 가중치 = 학습 체크포인트 | 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/model_21793.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | policy.pt = 매니페스트; policy.onnx = 매니페스트; env.yaml = 매니페스트 | 일치 |
| 0 정책 파일 | 회복 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | 같은 파일 | 일치 |
| 0 정책 파일 | 회복 학습 시점의 시뮬 소스(계약 함수) | 커밋 e4d2ea28(2026-10-04 09:44:03) = run 시작 2026-10-04 10:08:41 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 1 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 20 개 같음 · 바뀐 것: _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | (학습 쪽 항목) | 일치 |
| 0 정책 파일 | 회복 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_21793.pt | SOURCE_CHECKPOINT model_21793.pt | 같은 파일 | 일치 |
| 1 관측 | 구성 정책들의 학습 관측 설정이 서로 같은가 | 서기 g21a·회복 g24ra_seed1_from_w2e20294 | 하나의 관측 조립을 공유(C5Runner 코어 / 합성 러너 이어받기) | 같은 구조 | 일치 |
| 1 관측 | 명령 관측(회복 정책) | 서기 50%: (0,0,0) / 그 밖 vx U(0.05, 0.15), vy (0, 0), wz (0, 0) | 회복 정책 입력만 cue vx 0.03 를 1.0 s(켜짐 틱 필터 ω_y>0), 그 밖 0 | 0.03 | 불일치 [기지] |
| 1 관측 | 관절속도 원천(유한차분 식) | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 1.9e-07; 단순 차분으로 재구성하면 0.48) | 같은 식(노드 발행 비트 동일) | 불일치 [기지] |
| 1 관측 | actions(직전 행동) 정의·필터 | last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 6.7e-08 vs 슬루 뒤 값을 넣었다면 0.37; 시뮬 소스 실행 대조 1.5e-07 | 0.27 | 일치 |
| 2 행동 | 차원·행동 클립 | actor 출력 12 · actions.clip None · clip_actions None | 러너 출력 12 · 클립 없음(프로브 원시 행동 최대 ¦값¦ 2.23 가 그대로 목표식에 들어간다 — 재구성 오차는 아래) | 같음 | 일치 |
| 2 행동 | 노드(stage8) 슬루·주기 | 슬루는 한 번(4.0°/틱) | 러너 슬루 뒤 stage8 이 한 번 더: 4.0°/틱 @ 0.02 s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 3.908863° | 8af2f6e5 판 stage8 상수 같음(젯슨에 어느 판이 있는지는 미확인) | 일치 |
| 3 시간 | 제어 주기 | 20.0 ms(sim.dt × decimation 4) | TICK_SEC 20.0 ms · 러너 POLICY_DT 20.0 ms · 감지기 dt 20.0 ms · 실물 1007 로그 틱 간격 평균 20.00 ms(1–99 % 19.9–20.1, 최대 23.3) | 20.0 ms | 일치 |
| 5 감지기 | 평가 설정의 출처 | log_picture/ground_recovery_sab_imu_nd_s7.txt 의 '인자:' 줄 | 브리지 C5_DETECTOR_KW·C5_EXTRA → 실제 감지기 객체에서 읽은 값 | 같은 방식 | 일치 |
| 5 감지기 | 켜기 기울기 편차 [°] | 3 | 3 | 3 | 일치 |
| 5 감지기 | 켜기 ¦ω_xy¦ [rad/s] | 0.5 | 0.5 | 0.5 | 일치 |
| 5 감지기 | 끄기 기울기 [°] | 1.5 | 1.5 | 1.5 | 일치 |
| 5 감지기 | 끄기 ¦ω_xy¦ | 0.3 | 0.3 | 0.3 | 일치 |
| 5 감지기 | 끄기 평온 지속 [s] | 1 | 1 | 1 | 일치 |
| 5 감지기 | 켜진 뒤 최소 [s] | 0.4 | 0.4 | 0.4 | 일치 |
| 5 감지기 | 기준 자세 τ [s] | 10 | 10 | 10 | 일치 |
| 5 감지기 | 기준 추종 창 [s] | 1.5 | 1.5 | 1.5 | 일치 |
| 5 감지기 | 조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s) | (0.6, 6, 0.4, 0.2) | (0.6, 6, 0.4, 0.2) | (0.6, 6, 0.4, 0.2) | 일치 |
| 5 감지기 | 불응기 (s, 기울기, ¦ω¦) | (1.5, 6, 1) | (1.5, 6, 1) | (1.5, 6, 1) | 일치 |
| 5 감지기 | 조기 복귀 자격 (최대 기울기, 최대 ¦ω¦) | (5, 1) | (5, 1) | (5, 1) | 일치 |
| 5 감지기 | 감지 입력 EMA α | 0.33 | 0.33 | 0.33 | 일치 |
| 5 감지기 | 끄기 접촉 요구 | any | any | any | 일치 |
| 5 감지기 | 기준 자세 상한·갱신 조건 | ±8°, 서기·¦ω¦ < 0.1·양발일 때만 갱신(평가기 소스 상수) | ref_cap 8.0°, ¦ω¦ < 0.1·양발(c5_switch 소스) | 8 | 일치 |
| 5 감지기 | 회복 cue (vx, 지속 s, 방향 = IMU ω_y) | (0.03, 1, True) | (0.03, 1.0, True) | (0.03, 1.0, True) | 일치 |
| 5 감지기 | 감지 입력 원천 | 정책 관측 최신 프레임(지연·잡음 자이로·중력, 접촉 0/1) — --det_from_obs True | 정책이 받는 값과 같은 중력(장착 편향 보정 뒤)·자이로·접촉 비트 | 같음 | 일치 |
| 5 감지기 | 타이머 정밀도 | float32 누적(시뮬 텐서) | float64 누적(imu_f32 False) | float64(옵션 없음) | 불일치 [기지] |
| 5 감지기 | 평가 하네스의 관측 덮어쓰기(직전 행동 EMA, 관절속도 EMA, 엔코더 판, 접촉 판) | (0.27, joint_vel:0.25, v2, v2) | 배포 러너 (0.27, joint_vel:0.25, v2) · 학습 접촉 함수 v2 | 같음 | 일치 |
| 5 감지기 | 평가 하네스의 자이로 지연 | 학습 0–3 틱 DR / KE·전환 평가는 태스크 Biped12-HangStand-G11a-v0 + 관측 덮어쓰기(--gyro_delay 0 — 기본값, 평가 인자에 없음) | 실물 IMU 경로 지연 미측정 | 같음 | 확인 불가 |
| 7 젯슨 판 | 노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교 | — | 정책 틱 340(회복 모드 40 틱) | 발행 목표 True · 관측 True | 일치 |
| 7 젯슨 판 | HEAD 에만 있는 선택 기능 | — | --runner-ema-fresh, --c5-imu-f32·--c5-settle(odic)·requiet | 없음(기본 꺼짐 기능이라 기본 동작은 위 비트 비교대로 같다) | 일치 |

### stand_c5krh_g21a_g24ra

일치 65 · 불일치 9 · 확인 불가 6 — 아래는 stand_c5kr_g21a_g24ra 표와 다른 9 줄(같은 71 줄 생략)

| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| 1 관측 | 관절속도 원천(유한차분 식) | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 2.4e-07; 단순 차분으로 재구성하면 0.48) | 같은 식(노드 발행 비트 동일) | 불일치 [기지] |
| 1 관측 | actions(직전 행동) 정의·필터 | last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 6.5e-08 vs 슬루 뒤 값을 넣었다면 0.37; 시뮬 소스 실행 대조 1.5e-07 | 0.27 | 일치 |
| 2 행동 | 오프셋 = 기본자세(use_default_offset) | use_default_offset True · 명목 기본자세(위 joint_pos 줄) | 프로브 역산 오프셋과 학습 기본자세 최대 ¦Δ¦ 3e-08 rad | 목표 비트 동일 | 일치 |
| 2 행동 | 노드(stage8) 슬루·주기 | 슬루는 한 번(4.0°/틱) | 러너 슬루 뒤 stage8 이 한 번 더: 4.0°/틱 @ 0.02 s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 3.748558° | 8af2f6e5 판 stage8 상수 같음(젯슨에 어느 판이 있는지는 미확인) | 일치 |
| 5 감지기 | 평가 설정의 출처 | log_picture/ground_recovery_sab_imuH_nd_s7.txt 의 '인자:' 줄 | 브리지 C5_DETECTOR_KW·C5_EXTRA → 실제 감지기 객체에서 읽은 값 | 같은 방식 | 일치 |
| 5 감지기 | 조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s) | (0.6, 6, 0.4, 0.7) | (0.6, 6, 0.4, 0.7) | (0.6, 6, 0.4, 0.7) | 일치 |
| 7 젯슨 판 | 노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교 | — | 정책 틱 340(회복 모드 63 틱) | 발행 목표 True · 관측 True | 일치 |
| 7 젯슨 판 | 실물 블랙박스 재생 — real_1007(젯슨이 실제로 낸 값과 대조) | 기록 1780 틱(35.6 s), 기록 반올림 1e-4 | 관측 최대 ¦Δ¦ 8.3e-05 · 원시 행동 0.00021 · LPF 뒤 목표 0.0023° · 회복 모드 불일치 0/1780 틱 · 기준 자세 0.00079° · 켜짐 2/2 | 관측 최대 ¦Δ¦ 8.3e-05 · 원시 행동 0.00021 · LPF 뒤 목표 0.0023° · 회복 모드 불일치 0/1780 틱 · 기준 자세 0.00079° · 켜짐 2/2 | 일치 |
| 7 젯슨 판 | 실물 기록의 배포 체인 값 — real_1007 | 학습 온셋 블렌딩 상한 1.5 s · 관측 필터 β (0.27, 0.25) | — | 블렌딩 α(t) 가장 잘 맞는 길이 1.5 s(오차 0.0005) · 기록 obs_filter [0.27, 0.25] · jvel_ema v2 · 장착 편향 [1.3, -0.6]° · 명령 최대 ¦값¦ 0.0 · 틱 간격 평균 20.00 ms(최대 23.3) | 일치 |

### stand_w3s3d_15800

일치 42 · 불일치 9 · 확인 불가 5 — 아래는 stand_crouch_g21a 표와 다른 25 줄(같은 31 줄 생략, 젯슨 판 칸은 전부 '젯슨 미배포 키')

| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| 0 정책 파일 | 걷기 정책 가중치 = 학습 체크포인트 | 2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | — | 일치 |
| 0 정책 파일 | 걷기 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | — | 일치 |
| 0 정책 파일 | 걷기 학습 시점의 시뮬 소스(계약 함수) | 커밋 ee1d80e6(2026-10-07 00:11:10) = run 시작 2026-10-07 00:11:12 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 25 개 중 25 개 같음 · 감사가 읽는 규칙 17 개는 두 판에서 같음 | — | 일치 |
| 0 정책 파일 | 걷기 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_15800.pt | SOURCE_CHECKPOINT model_15800.pt | — | 일치 |
| 1 관측 | 명령 관측 범위(걷기) | stage 3 단일 축: vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2); 학습 로그 도달값 vy_max 0.04·wz_max 0.2·보조 α 0 | W3CmdLimits vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2) · 범위 밖·두 축 동시 거부(validate_w3_cmd) | — | 일치 |
| 1 관측 | 명령 관측: 헤딩 고리(외부 P 고리) | 학습에 없음(p_hloop 0.0, heading_command False) — 걷기 명령은 한 축만 | 기본 켜짐(K 0.5·±0.4 rad/s): 앞·옆 명령에 wz 를 **동시에** 넣는다 — 노드 세션 측정 관측 명령 (0.06, 0, -0.0349) | — | 불일치 [기지] |
| 1 관측 | 명령 변경 간격 | 리샘플 (8, 12) s(STEP→STEP 위상 유지) | 최소 체류 1.0 s(위상 유지) | — | 불일치 [기지] |
| 1 관측 | 관절속도 원천(유한차분 식) | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 3.5e-07; 단순 차분으로 재구성하면 0.035) | — | 불일치 [기지] |
| 1 관측 | actions(직전 행동) 정의·필터 | last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 1.4e-07 vs 슬루 뒤 값을 넣었다면 0.84; 시뮬 소스 실행 대조 1.5e-07 | — | 일치 |
| 2 행동 | 차원·행동 클립 | actor 출력 12 · actions.clip None · clip_actions None | 러너 출력 12 · 클립 없음(프로브 원시 행동 최대 ¦값¦ 4.56 가 그대로 목표식에 들어간다 — 재구성 오차는 아래) | — | 일치 |
| 2 행동 | 스케일 | 0.25 | 프로브 역산 0.25(관절 12/12, 포화 안 된 틱 65%) | — | 일치 |
| 2 행동 | 오프셋 = 기본자세(use_default_offset) | use_default_offset True · 명목 기본자세(위 joint_pos 줄) | 프로브 역산 오프셋과 학습 기본자세 최대 ¦Δ¦ 3.1e-08 rad | — | 일치 |
| 2 행동 | 처리 순서 | 지연 → 슬루 → LPF → 온셋 블렌딩 → 백래시 → 소프트 클램프 | 러너 슬루 → 브리지 LPF → (보정기 꺼짐) → 온셋 블렌딩 → 전송 → stage8 슬루·클램프; 노드 세션 재구성 오차 4.6e-07 rad·블렌딩 0° | — | 일치 |
| 2 행동 | LPF 초기값 | 리셋 틱의 현재 관절각 | 첫 슬루 출력(노드 세션: 첫 발행 − 첫 슬루 출력 = 1.4e-20 rad) | — | 불일치 [기지] |
| 2 행동 | 노드(stage8) 슬루·주기 | 슬루는 한 번(4.0°/틱) | 러너 슬루 뒤 stage8 이 한 번 더: 4.0°/틱 @ 0.02 s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 4.000005° | — | 일치 |
| 3 시간 | 제어 주기 | 20.0 ms(sim.dt × decimation 4) | TICK_SEC 20.0 ms · 러너 POLICY_DT 20.0 ms · W3Config.dt 20.0 ms · 실물 1007 로그 틱 간격 평균 20.00 ms(1–99 % 19.9–20.1, 최대 23.3) | — | 일치 |
| 3 시간 | 행동 지연 | U{1..5} 틱 | 인위 지연 없음 — 실물 루프 지연이 그 몫(총 지연 미측정). 실물 1007 로그의 구간 실측: 브리지 발행 → stage8 처리 중앙 5.2 ms(99 % 9.9, 최대 12.9) · status 나이 중앙 6 ms(최대 10) | — | 확인 불가 |
| 4 위상 | 스텝 주파수 f_step | 1.1 | W3Config 1.1 · 프로브(걷는 중 위상 증가/틱) 1.1 Hz | — | 일치 |
| 4 위상 | 관측 게이트 램프 | ramp_s 0.2(env.yaml 에 없으면 w3_gait_update 기본값) | W3Config 0.2 · 프로브: 켜진 뒤 게이트 1 까지 [10, 10, 10, 10] 틱 | — | 일치 |
| 4 위상 | gait_enable 래치 | 켜기: 다음 갱신에 즉시(위상 0 또는 리드 규칙) / 끄기: 반주기 경계에서만 / 에피소드 틱 ≤ 1 은 즉시 반영 | 프로브: 켜기 지연 [0, 0, 0, 0] 틱 · 끄기 [(0, 0.478, True), (4, 0.986, True), (6, 0.49, True)](대기 틱, 직전 위상, 경계 통과) · 서기 클록 (0,0) True | — | 일치 |
| 4 위상 | 리드 발 규칙 문턱(vy, wz, 옆 없음 판정) | (0.02, 0.08, 0.02) | W3Config (0.02, 0.08, 0.02) · 프로브 시작 위상 [0.0, 0.5, 0.5, 0.0](앞, 옆−, 회전−, 제자리) · 리드 ['left', 'right', 'right', 'left'] | — | 일치 |
| 4 위상 | 행동 보조·참조 잔차 | ref_scale 0.0 · 보조 α 학습 로그 0(커리큘럼 start_stage 4) | 배포에 잔차 없음(러너 w3 경로는 클록 관측만 바꾼다) | — | 일치 |
| 5 감독기 | 걷기 켜기 선행조건(양발 s, 기울기 °, ¦ω¦ , 정착 s) | (1, 7, 0.25, 1)(eval_transition 기본값) — 학습에는 없음(에피소드 70 % 가 곧바로 STEP) | (1, 7, 0.25, 1) | — | 일치 |
| 5 감독기 | 명령 상한 덮어쓰기(--w3-cmd-limits) | 학습 ¦vy¦ ≤ 0.04, ¦wz¦ ≤ 0.2 | 기본 = 학습 범위. 기동 인자로 vy_max ≤ 0.08, wz_max ≤ 0.3 까지 열 수 있다 | — | 일치 |
| 2 행동 | 노드 세션(가짜 ROS 로 main() 그대로, HEAD) | 학습 식 재구성: 슬루 → LPF β → 스무스스텝 블렌딩 → 모터 프레임 | 정책 틱 558(연속 558) · 체인 4.6e-07 rad · 블렌딩 0° · 발행 = 목표 − 기준선 0° · 명령 응답 6/6 수락 | — | 일치 |

### stand_kew3_g21a_g24ra_w3s3d15800

일치 79 · 불일치 11 · 확인 불가 6 — 아래는 stand_c5kr_g21a_g24ra·stand_w3s3d_15800 표와 다른 96 줄(같은 0 줄 생략)

| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |
|---|---|---|---|---|---|
| 0 정책 파일 | 서기 정책 가중치 = 학습 체크포인트 | 2026-10-02_21-23-08_g21a/model_11300.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 서기 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 회복 정책 가중치 = 학습 체크포인트 | 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/model_21793.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 회복 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 걷기 정책 가중치 = 학습 체크포인트 | 2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt actor 8 텐서 | policy.pt 8/8 비트 동일 · policy.onnx 8/8 비트 동일(그래프 Gemm–Elu–Gemm–Elu–Gemm–Elu–Gemm) | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 걷기 관측 정규화·행동 클립 | actor_obs_normalization False · clip_actions None · actions.clip None | 정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 서기 학습 시점의 시뮬 소스(계약 함수) | 커밋 190c03b9(2026-10-02 18:48:49) = run 시작 2026-10-02 21:23:08 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 19 개 같음 · 바뀐 것: SlewJointPositionAction(+15/−10 줄 — 지연 표본에 가중 옵션(delay_probs) 추가 — 비어 있으면 종전 균등 표본); _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 회복 학습 시점의 시뮬 소스(계약 함수) | 커밋 e4d2ea28(2026-10-04 09:44:03) = run 시작 2026-10-04 10:08:41 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 1 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 22 개 중 20 개 같음 · 바뀐 것: _gait_phase(+4/−0 줄 — 위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t); gait_clock(+10/−1 줄 — 관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]) · 감사가 읽는 규칙 12 개는 두 판에서 같음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 걷기 학습 시점의 시뮬 소스(계약 함수) | 커밋 ee1d80e6(2026-10-07 00:11:10) = run 시작 2026-10-07 00:11:12 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록(humanoid_leg_test1.diff, 브랜치 hang2stand-prep-0911): 바뀐 추적 파일 2 개 중 rl_walking/*.py 0 개 · 미추적 rl_walking 0 개 | 현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 25 개 중 25 개 같음 · 감사가 읽는 규칙 17 개는 두 판에서 같음 | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 서기 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_11300.pt | SOURCE_CHECKPOINT model_11300.pt | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 회복 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_21793.pt | SOURCE_CHECKPOINT model_21793.pt | —(젯슨 미배포 키) | 일치 |
| 0 정책 파일 | 걷기 시뮬 평가 체크포인트 = 배포 파일 | 평가 인자 model_15800.pt | SOURCE_CHECKPOINT model_15800.pt | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 구성 정책들의 학습 관측 설정이 서로 같은가 | 서기 g21a·회복 g24ra_seed1_from_w2e20294·걷기 w3S3d_from_s2d | 하나의 관측 조립을 공유(C5Runner 코어 / 합성 러너 이어받기) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 항 목록·순서·프레임 차원 | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | base_ang_vel(3) projected_gravity(3) velocity_commands(3) joint_pos(12) joint_vel(12) actions(12) gait_clock(2) feet_contact(2) gyro_ema(3) joint_vel_ema(12) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 이력 길이(항별 프레임 수) | [5] | [5] | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 이력 배열 순서·리셋 첫 틱 | 오래된 것 먼저, 첫 틱 전 칸 백필(시뮬 기록 실측: 순서 True·백필 True) | 최신 = 마지막 칸 True · 첫 틱 백필 True(프로브) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 스케일 | 없음(전 항 scale: null) | 스케일 없음(프로브: 관측 = 입력 True) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 클립 | base_ang_vel ±20; joint_pos ±10; joint_vel ±60; actions ±10; gyro_ema ±20; joint_vel_ema ±60 | 클립 없음(프로브: base_ang_vel 100, joint_vel 200, joint_pos 50, gyro_ema 100, joint_vel_ema 200 그대로 통과) | —(젯슨 미배포 키) | 불일치 [신규] |
| 1 관측 | base_ang_vel(자이로) 처리 | base_ang_vel_lagged(몸통 각속도 rad/s, 지연 0–3 틱 DR) + 잡음 | IMU gyro_rad_s 그대로(프로브 ¦Δ¦ 0) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | IMU 축 정렬(몸통 프레임 +x 앞·+z 위) | 시뮬 몸통(base) 프레임 | IMU 어댑터가 낸 gyro·quat 을 그대로 사용 — 축 부호는 현장 확인 기록(0912–0914)에 기댄다 | —(젯슨 미배포 키) | 확인 불가 |
| 1 관측 | projected_gravity 정의 | 쿼터니언으로 (0,0,−1) 을 몸통 프레임에(단위벡터) + 장착 편향 DR 피치 (-0.026, 0.026)·롤 (-0.026, 0.026) rad | projected_gravity(q): 독립 회전행렬 식과 최대 ¦Δ¦ 0 · 맵 편향 보정 [1.3, -0.6]° 식 오차 3.6e-15° | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 명령 관측(서기 정책) | (0,0,0) 고정 | KE 모드: 외부 명령과 무관하게 0(합성 러너 프로브 관측 명령 [0.0, 0.0, 0.0]) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 명령 관측(회복 정책) | 서기 50%: (0,0,0) / 그 밖 vx U(0.05, 0.15), vy (0, 0), wz (0, 0) | 회복 정책 입력만 cue vx 0.03 를 1.0 s(켜짐 틱 필터 ω_y>0), 그 밖 0 | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | 명령 관측 범위(걷기) | stage 3 단일 축: vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2); 학습 로그 도달값 vy_max 0.04·wz_max 0.2·보조 α 0 | W3CmdLimits vx 앞 (0.03, 0.1), vx 뒤 (0.03, 0.06), ¦vy¦ (0.03, 0.04), ¦wz¦ (0.1, 0.2) · 범위 밖·두 축 동시 거부(validate_w3_cmd) | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 명령 관측: 헤딩 고리(외부 P 고리) | 학습에 없음(p_hloop 0.0, heading_command False) — 걷기 명령은 한 축만 | 기본 켜짐(K 0.5·±0.4 rad/s): 앞·옆 명령에 wz 를 **동시에** 넣는다 — 노드 세션 측정 관측 명령 (0.06, 0, -0.0349) | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | 명령 변경 간격 | 리샘플 (8, 12) s(STEP→STEP 위상 유지) | 최소 체류 1.0 s(위상 유지) | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | joint_pos = 엔코더각 − 기본자세 | encoder_joint_pos_rel_v2; 기본자세(init_state, deg) +30.26 -30.26 +0.00 +0.00 +0.00 +0.00 -38.50 +38.50 +28.00 -28.00 +0.00 +0.00 | 러너 default_pose 와 최대 ¦Δ¦ 2.9e-08 rad · 관측에서 역산한 기본자세와 2.9e-08 rad | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 엔코더 관측 방식 | encoder_joint_pos_rel_v2·encoder_joint_vel_lpf: 로터(모터)측 각 — 유격 안 링크 이동·직렬 처짐은 관측에 안 보임 | 모터 엔코더 피드백(로터측)을 관절 프레임으로 변환(DIR × (모터 deg − 영점))해 그대로 사용 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | joint_vel = 엔코더 유한차분 속도의 EMA | encoder_joint_vel_lpf β 0.25(에피소드 첫 틱 = 원값) | 프로브 역산 β 0.25 · 시뮬 소스 실행 대조 최대 ¦Δ¦ 0 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 관절속도 원천(유한차분 식) | Δθ_enc / dt (틱마다, 리셋 틱 0) | 브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 3.3e-07; 단순 차분으로 재구성하면 0.48) | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | actions(직전 행동) 정의·필터 | last_action_lpf: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U(0.2, 0.4) | 프로브 역산 β 0.27(= 브리지 액션 LPF β 0.27); 원시 출력 재귀 잔차 6.7e-08 vs 슬루 뒤 값을 넣었다면 0.37; 시뮬 소스 실행 대조 1.5e-07 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | gait_clock(서기 정책) | gait_clock: [sin φ, cos φ], φ = 2π·1.1·(틱·dt), 리셋 = 위상 0 | 프로브 위상 기울기 1.1 Hz · 첫 틱 (0, 1) · 시뮬 소스 실행 대조 6e-08 · 1 시간 뒤 float32 위상 오차 0.00025 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | feet_contact 순서·값 | [left_ankle_r_joint, right_ankle_r_joint] (preserve_order True) 0/1 | contact2 [왼, 오] 0/1 그대로(프로브 ¦Δ¦ 0); 0/1 이외 값은 러너가 거부 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | 접촉 0/1 의 물리 정의 | feet_contact_real_v2: 발 법선력 > U(2, 20) N ∧ 발바닥 기울기 < U(6, 20)° → 5 틱마다 갱신 + 1 틱 지연 | 어댑터: 패드 4 개 중 하나라도 (카운트 − 바이어스) > 200 → 1 (미보정 카운트, 갱신 100 ms). 브리지는 비트를 그대로 쓴다 | —(젯슨 미배포 키) | 확인 불가 |
| 1 관측 | gyro_ema 계수 | gyro_ema_lagged α 0.1 | 프로브 역산 α 0.1 · 정상 구간 재구성 오차 0 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | gyro_ema 시작 규칙 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 0.513, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | joint_vel_ema 계수 | encoder_joint_vel_ema_v2 α 0.1(원시 속도에서 — v2) | 프로브 역산 α 0.1 · 모드 v2 · 정상 구간 재구성 오차 1.5e-08 | —(젯슨 미배포 키) | 일치 |
| 1 관측 | joint_vel_ema 시작 규칙 | 에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1) | 틱 0 에서만 재시작(ema_fresh_ticks 1) — 시뮬 소스 실행 대조 시작 구간 최대 ¦Δ¦ 1.61, 정상 구간 0; 옵션 ema_fresh_ticks 2 면 0 | —(젯슨 미배포 키) | 불일치 [기지] |
| 1 관측 | 관측 벡터 전체 재구성(600 틱) | 학습 계약으로 다시 조립한 값(항 순서·이력·필터·기본자세) | 실제 러너 관측과 최대 ¦Δ¦ 0(base_ang_vel); S8a EMA 시작 규칙만 배포식으로 둔 재구성 | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 관절 순서(12) | 시뮬 기록 joint_names: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | JOINT_ORDER: left_hip_f right_hip_f left_hip_a right_hip_a left_hip_r right_hip_r left_knee right_knee left_ankle_f right_ankle_f left_ankle_r right_ankle_r | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 차원·행동 클립 | actor 출력 12 · actions.clip None · clip_actions None | 러너 출력 12 · 클립 없음(프로브 원시 행동 최대 ¦값¦ 2.23 가 그대로 목표식에 들어간다 — 재구성 오차는 아래) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 스케일 | 0.25 | 프로브 역산 0.25(관절 12/12, 포화 안 된 틱 88%) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 오프셋 = 기본자세(use_default_offset) | use_default_offset True · 명목 기본자세(위 joint_pos 줄) | 프로브 역산 오프셋과 학습 기본자세 최대 ¦Δ¦ 2.9e-08 rad | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 슬루 상한(틱당) | 4.0000° | 프로브 최대 틱 변화 4.0000° · 1 rad 떨어진 첫 틱 이동 4.0000° | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 슬루 기준 초기화 | 리셋 틱의 현재 관절각(_needs_init → joint_pos) | 첫 step 의 qpos(프로브: 전 관절이 qpos 에서 정확히 슬루 한 번 True) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 목표각 재구성(600 틱) | 기본자세 + 스케일·원시 → 슬루(학습 식) | 실제 러너 출력과 최대 ¦Δ¦ 0 rad · 타깃 클램프 없음 | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 처리 순서 | 지연 → 슬루 → LPF → 온셋 블렌딩 → 백래시 → 소프트 클램프 | 러너 슬루 → 브리지 LPF → (보정기 꺼짐) → 온셋 블렌딩 → 전송 → stage8 슬루·클램프; 노드 세션 재구성 오차 6.6e-07 rad·블렌딩 0° | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 액션 LPF β | env 별 U(0.2, 0.4) | STAND_LPF_BETA 0.27 · 노드 세션 역산 0.27(키가 stand 로 시작 → 적용 True) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | LPF 초기값 | 리셋 틱의 현재 관절각 | 첫 슬루 출력(노드 세션: 첫 발행 − 첫 슬루 출력 = 3.5e-18 rad) | —(젯슨 미배포 키) | 불일치 [기지] |
| 2 행동 | 온셋 블렌딩(인계) | 유지 U(0, 0.3) s + 스무스스텝 U(1, 1.5) s | 유지 0 s + 스무스스텝 BLEND_S 1.5 s — onset_alpha(소스 실행) vs blend_alpha 최대 ¦Δ¦ 6e-08; 노드 세션 α(t) 오차 0.00049(기록 3 자리 반올림) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 블렌딩 시작점 | env 기본자세(편향 DR 포함) 홀드 목표 | 인계 순간 stage8 이 스트리밍하던 명령(노드 세션: 첫 발행 − 가짜 stage8 명령 = 0°) | —(젯슨 미배포 키) | 확인 불가 |
| 2 행동 | 소프트 한계 클램프(자르는 관절) | clamp_to_soft_limits True: config soft 를 '.*_ankle_f_joint' 관절만(그 밖은 물리 한계까지 통과) · 한계 파일 마지막 변경 2026-09-18(모든 run 학습 시작 전 — 학습과 실물이 같은 표를 읽었다) | stage8 노드: **전 관절**을 config soft 경계로 자른다(러너·브리지는 자르지 않음) | —(젯슨 미배포 키) | 불일치 [기지] |
| 2 행동 | 한계 초과 명령의 거부 | 없음(클램프만 — 목표가 아무리 넘어도 경계값으로 계속 구동) | stage8: 경계를 25° 넘게 넘으면 그 관절 명령을 거부(limit_runaway) | —(젯슨 미배포 키) | 불일치 [기지] |
| 2 행동 | 노드(stage8) 슬루·주기 | 슬루는 한 번(4.0°/틱) | 러너 슬루 뒤 stage8 이 한 번 더: 4.0°/틱 @ 0.02 s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 4.000005° | —(젯슨 미배포 키) | 일치 |
| 2 행동 | PD 게인(kp/kd) | [(150.0, 5.0)] · DR ×(0.85, 1.1)/×(0.8, 1.2) | POLICY_KP 150.0/KD 5.0 · 맵 불일치 관절 0(--live 는 불일치면 기동 거부) | —(젯슨 미배포 키) | 일치 |
| 2 행동 | 학습에 없는 가산 보정(roll/pitch comp) | 없음 | 맵 roll_comp False · pitch_comp False | —(젯슨 미배포 키) | 일치 |
| 3 시간 | 제어 주기 | 20.0 ms(sim.dt × decimation 4) | TICK_SEC 20.0 ms · 러너 POLICY_DT 20.0 ms · 감지기 dt 20.0 ms · W3Config.dt 20.0 ms · 실물 1007 로그 틱 간격 평균 20.00 ms(1–99 % 19.9–20.1, 최대 23.3) | —(젯슨 미배포 키) | 일치 |
| 3 시간 | 행동 지연 | g21a: U{0..3} 틱 / g24ra_seed1_from_w2e20294: U{0..3} 틱 / w3S3d_from_s2d: U{1..5} 틱 → 공통 구간 1..3 틱 | 인위 지연 없음 — 실물 루프 지연이 그 몫(총 지연 미측정). 실물 1007 로그의 구간 실측: 브리지 발행 → stage8 처리 중앙 5.2 ms(99 % 9.9, 최대 12.9) · status 나이 중앙 6 ms(최대 10) | —(젯슨 미배포 키) | 확인 불가 |
| 3 시간 | 접촉 갱신 주기 | 5 틱마다 갱신(feet_contact hold_ticks) — 그 사이는 직전 비트 유지 | 브리지는 틱마다 최신 수신 비트를 읽는다. 실물 로그에서 발 원카운트 합이 바뀌는 간격: real_1007 최빈 5 틱(353/354 구간, 더 긴 구간 1 개·최대 10 틱); real_1003 최빈 5 틱(2872/2875 구간, 더 긴 구간 3 개·최대 10 틱) | —(젯슨 미배포 키) | 일치 |
| 3 시간 | 센서 관측 지연(IMU·접촉) | 자이로 0–3 틱 · 자세(중력) (0, 3) 틱 · 접촉 5 틱 유지 + 1 틱 | 브리지는 최신 수신값을 그대로 쓴다(신선도 한계 IMU·관절 0.2 s, 발 0.35 s). 센서 안쪽·전송 지연은 미측정. 브리지에서 본 것: IMU 표본이 직전 틱과 같은 틱 real_1007 0.39%; real_1003 0.03%(거의 매 틱 새 표본) | —(젯슨 미배포 키) | 확인 불가 |
| 4 위상 | 스텝 주파수 f_step | 1.1 | W3Config 1.1 · 프로브(걷는 중 위상 증가/틱) 1.1 Hz | —(젯슨 미배포 키) | 일치 |
| 4 위상 | 관측 게이트 램프 | ramp_s 0.2(env.yaml 에 없으면 w3_gait_update 기본값) | W3Config 0.2 · 프로브: 켜진 뒤 게이트 1 까지 [10, 10, 10, 10] 틱 | —(젯슨 미배포 키) | 일치 |
| 4 위상 | gait_enable 래치 | 켜기: 다음 갱신에 즉시(위상 0 또는 리드 규칙) / 끄기: 반주기 경계에서만 / 에피소드 틱 ≤ 1 은 즉시 반영 | 프로브: 켜기 지연 [0, 0, 0, 0] 틱 · 끄기 [(0, 0.478, True), (4, 0.986, True), (6, 0.49, True)](대기 틱, 직전 위상, 경계 통과) · 서기 클록 (0,0) True | —(젯슨 미배포 키) | 일치 |
| 4 위상 | 리드 발 규칙 문턱(vy, wz, 옆 없음 판정) | (0.02, 0.08, 0.02) | W3Config (0.02, 0.08, 0.02) · 프로브 시작 위상 [0.0, 0.5, 0.5, 0.0](앞, 옆−, 회전−, 제자리) · 리드 ['left', 'right', 'right', 'left'] | —(젯슨 미배포 키) | 일치 |
| 4 위상 | 행동 보조·참조 잔차 | ref_scale 0.0 · 보조 α 학습 로그 0(커리큘럼 start_stage 4) | 배포에 잔차 없음(러너 w3 경로는 클록 관측만 바꾼다) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 평가 설정의 출처 | log_picture/ground_recovery_tr_kew3_nd_d2_s1.txt 의 '인자:' 줄 | 브리지 C5_DETECTOR_KW·C5_EXTRA → 실제 감지기 객체에서 읽은 값 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 켜기 기울기 편차 [°] | 3 | 3 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 켜기 ¦ω_xy¦ [rad/s] | 0.5 | 0.5 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 끄기 기울기 [°] | 1.5 | 1.5 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 끄기 ¦ω_xy¦ | 0.3 | 0.3 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 끄기 평온 지속 [s] | 1 | 1 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 켜진 뒤 최소 [s] | 0.4 | 0.4 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 기준 자세 τ [s] | 10 | 10 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 기준 추종 창 [s] | 1.5 | 1.5 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 조기 복귀 (최소 s, 기울기, ¦ω¦, 유지 s) | (0.6, 6, 0.4, 0.2) | (0.6, 6, 0.4, 0.2) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 불응기 (s, 기울기, ¦ω¦) | (1.5, 6, 1) | (1.5, 6, 1) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 조기 복귀 자격 (최대 기울기, 최대 ¦ω¦) | (5, 1) | (5, 1) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 감지 입력 EMA α | 0.33 | 0.33 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 끄기 접촉 요구 | any | any | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 기준 자세 상한·갱신 조건 | ±8°, 서기·¦ω¦ < 0.1·양발일 때만 갱신(평가기 소스 상수) | ref_cap 8.0°, ¦ω¦ < 0.1·양발(c5_switch 소스) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 회복 cue (vx, 지속 s, 방향 = IMU ω_y) | (0.03, 1, True) | (0.03, 1.0, True) | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 감지 입력 원천 | 정책 관측 최신 프레임(지연·잡음 자이로·중력, 접촉 0/1) — --det_from_obs True | 정책이 받는 값과 같은 중력(장착 편향 보정 뒤)·자이로·접촉 비트 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 타이머 정밀도 | float32 누적(시뮬 텐서) | float64 누적(imu_f32 False) | —(젯슨 미배포 키) | 불일치 [기지] |
| 5 감지기 | 평가 하네스의 관측 덮어쓰기(직전 행동 EMA, 관절속도 EMA, 엔코더 판, 접촉 판) | (0.27, joint_vel:0.25, v2, v2) | 배포 러너 (0.27, joint_vel:0.25, v2) · 학습 접촉 함수 v2 | —(젯슨 미배포 키) | 일치 |
| 5 감지기 | 평가 하네스의 자이로 지연 | 학습 0–3 틱 DR / KE·전환 평가는 태스크 Biped12-HangStand-G11a-v0 + 관측 덮어쓰기(--gyro_delay 0 — 기본값, 평가 인자에 없음) | 실물 IMU 경로 지연 미측정 | —(젯슨 미배포 키) | 확인 불가 |
| 5 감독기 | 걷기 켜기 선행조건(양발 s, 기울기 °, ¦ω¦ , 정착 s) | (1, 7, 0.25, 1)(eval_transition 기본값) — 학습에는 없음(에피소드 70 % 가 곧바로 STEP) | (1, 7, 0.25, 1) | —(젯슨 미배포 키) | 일치 |
| 5 감독기 | 명령 상한 덮어쓰기(--w3-cmd-limits) | 학습 ¦vy¦ ≤ 0.04, ¦wz¦ ≤ 0.2 | 기본 = 학습 범위. 기동 인자로 vy_max ≤ 0.08, wz_max ≤ 0.3 까지 열 수 있다 | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 두 하위 러너 계약 호환 | 세 run 의 관측 설정 동일 · 기본자세 동일 | KE 코어 = W3 코어: 관측 레이아웃·기본자세·필터 β·EMA 모드·ema_fresh True | —(젯슨 미배포 키) | 일치 |
| 6 합성 | W3 → KE 인계 규칙(정착 s, ¦ω_xy¦ 문턱, 상한 s, 방식) | 평가기 정착 판정 (0.5, 0.15, 3)(산출물 머리) · X6(배포 코드 폐루프) --kw_handoff carry --kw_settle 0.5 | (0.5, 0.15, 3.0, carry) · 프로브: 조용한 양발 25 틱 뒤 인계, 시끄러우면 150 틱(상한), 한발이면 대기 True | —(젯슨 미배포 키) | 일치 |
| 6 합성 | KE → W3 인계 | 켜기 수락 틱에 W3(첫 틱 STAND, 다음 틱 gait 0→1, 게이트 0.2 s 램프) — 평가 direct 팔 | 프로브: 같은 틱 전환 True · 첫 틱 ge False·클록 이력 0 True · 둘째 틱 ge True(게이트 0.1) · 전환 틱 목표 변화 4°(슬루 상한 이내 — 슬루 기준을 이어받는다. 합성 입력이라 크기 자체는 뜻이 없다) | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 인계 뒤 KE 상태 | 이력·필터·직전 행동 이어받기, 클록만 위상 0 에서(평가 direct 팔의 클록 패치), 감지기 새로 | 프로브: KE 틱 0 · 클록 이력 전부 (0,1) True · 감지기 켜짐 누계 0 · 첫 틱 목표 변화 4°(슬루 상한 이내) | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 회복 모드 중 걷기 켜기 | 평가 X6: 감독기 선행조건 + KE 회복 모드면 거부 | 프로브: 회복 모드에서 사유 'KE 회복 모드 켜짐 — 서기 복귀 뒤에 켤 것' | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 서기 쪽 KE 변형 | 전환 평가: 조기 복귀 유지 0.2 s | KEW3_SPECS ke = stand_c5kr_g21a_g24ra(유지 0.2 s) | 젯슨 콘솔에는 krh(0.7 s)가 따로 있다 | 일치 |
| 6 합성 | 서기·회복(KE) 하위 키 러너 프로브 | 학습 계약 재구성(stand_c5kr_g21a_g24ra 표와 같은 검사) | 관측 최대 ¦Δ¦ 0 · 목표각 0 rad | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 걷기(W3) 하위 키 러너 프로브 | 학습 계약 재구성(stand_w3s3d_15800 표와 같은 검사) | 관측 최대 ¦Δ¦ 0 · 목표각 0 rad | —(젯슨 미배포 키) | 일치 |
| 6 합성 | 노드 세션(가짜 ROS 로 main() 그대로, HEAD) | 학습 식 재구성: 슬루 → LPF β → 스무스스텝 블렌딩 → 모터 프레임 | 정책 틱 558(연속 558) · 체인 6.6e-07 rad · 블렌딩 0° · 발행 = 목표 − 기준선 0° · 명령 응답 6/6 수락 · 전환 KE→W3 2 회·W3→KE 2 회 | —(젯슨 미배포 키) | 일치 |


(감사 실행 755 s — 기존 검증·민감도 시험 시간 포함 · 학습 산출물 /home/ryu/humanoid_leg_test1/logs · torch 2.12.0+cu130)
