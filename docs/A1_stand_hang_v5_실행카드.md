# A1 실행카드 — stand_hang_v5 · 70% 지지 핸드오프 (2026-09-10)

**결정(승윤님 0909 밤):** S8 = 최종 시뮬 베이스라인. 실물 첫 후보 **stand_hang_v5**(S8ac4, run 2026-09-09_19-15-42/model_139987, policy.pt md5 `2a20b7b1cd4ec3da4428edb9a859ce0b`).
v5p 는 보존만(이번 패키지·콘솔 목록에서 제외). 시뮬 보상/정책 튜닝·S9 학습 금지. 목표 = 실물에서 HANG→TOUCHDOWN→LOAD TRANSFER→FREE STAND 검증.
**배포·ARM·토크 인가·윈치 방출은 승윤님 직접. Claude 는 명령·체크리스트·기록만.**

v5 가 시뮬에서 보이는 모습(실물 대조 기대치): 정착 몸통 피치 **−4.9°(앞으로 기움; 콘솔 fwd 로는 +4~5° 로 보일 것)**, 무릎 ∓5.5°, 발 간격 273 mm,
정상상태 각속도 0.10 rad/s·수평속도 0.013 m/s(학습 노이즈 하), 실물 수준 노이즈면 action-rate 0.004. 실제 행잉 94.8%·실물유사 99.7%(1024 env).

---
## 1. 배포 전 최종 체크 (PC, 완료 상태를 승윤님이 눈으로 확인)
| 항목 | 명령 | 기대 |
|---|---|---|
| 저장소 | `git status --short \| grep -v '^??'` | 빈 출력(전부 커밋) |
| v5 등록 md5 | `md5sum models/stand_hang_v5/policy.pt` | `2a20b7b1…` = meta.json `files.policy.pt` |
| 브리지 selftest | `python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5` | `selftest: ALL PASS`, 「학습 계약 … stand_hang_v5(지연 3~8틱)」, 「롤/피치 기록」 PASS |
| 러너 selftest | `python3 rl_walking/deploy/policy_runner_stage4.py --selftest` | 6항 ALL PASS(「S8a 320 조립」 포함) |
| 패키지 | `GENS="stand_v18b stand_hang_v4 stand_hang_v5" bash jetson/deploy_package.sh` | `deploy/stand_7차_<stamp>.tar.gz` + `deploy/MANIFEST.md5` |
| 로컬 풀기 검증 | (Claude 수행) 젯슨형 트리에 풀어 md5 종료코드 0 + selftest ALL PASS | MD5_OK |

## 2. 젯슨 deploy 명령 (승윤님 실행 — 파일 복사·md5·selftest·~/rl_calib 사본 갱신·콘솔 재기동만. CAN·모터·stage8 무접촉)
```bash
cd /home/ryu/humanoid_leg_test1
bash jetson/deploy_to_jetson.sh --host 172.16.35.89 --pkg $(ls -t deploy/stand_7차_*.tar.gz | head -1)
```
단계 [1/7]~[7/7] 이 전부 찍히고 마지막 `systemctl is-active operator-console` 가 `active` 면 끝. 어느 단계든 `— 중단` 이 뜨면 진행 금지(Claude 에게 로그 보여주기).

## 3. checksum 확인 (젯슨에서 다시 한 번)
```bash
ssh mama@172.16.35.89 'cd ~/humanoid_repo && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK; md5sum logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/exported/policy.pt'
```
기대: `MD5_OK` 와 `2a20b7b1cd4ec3da4428edb9a859ce0b`. (한국어 로케일에서 md5sum 문자열 grep 금지 — 종료코드/echo 로 판정.)

## 4. selftest (젯슨)
```bash
ssh mama@172.16.35.89 'cd ~/humanoid_repo && python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5 2>&1 | grep -E "^\[(PASS|FAIL)\]|^selftest:" | cut -c1-120'
ssh mama@172.16.35.89 'cd ~/humanoid_repo && python3 rl_walking/deploy/policy_runner_stage4.py --selftest 2>&1 | tail -3'
ssh mama@172.16.35.89 'cd ~/humanoid_repo && python3 jetson/scripts/foot_force_adapter_node.py --selftest && python3 ~/rl_calib/release_all.py --selftest | tail -1'
```
콘솔 http://172.16.35.89:8800 새로고침 → 정책 드롭다운에 `stand_hang_v5` 가 보이고 기본값은 `stand_v18b` 그대로. **드롭다운을 stand_hang_v5 로 바꾼다.**

## 5. A1 체크리스트 (11단계) — 발은 지면 접촉, 윈치가 체중의 ~70% 지지(콘솔 발력 ~30% = 52 N)
| # | 단계 | 조작 | 통과 기준 | 기록 |
|---|---|---|---|---|
| 1 | 안전 절차 | 이전 잔재 「스택 종료」「기록 종료」 → `bash ~/rl_calib/can_up.sh` → `(candump -ta can1 > /tmp/live_candump_$(date +%H%M).log 2>&1 &)` | 12 모터 피드백, 허브 커넥터 고정 | candump 시작 |
| 2 | boot pose | `python3 ~/rl_calib/boot_pose_check.py --ref stand` | **12/12**(△ 경고면 자세 재잡고 재실행). 불일치면 **배수 보정 우선**(`zero_anchor.py`), 맵 갱신 시 stage8 재기동 | — |
| 3 | FSR 8채널 | `bash ~/rl_calib/sensors_up.sh` → `python3 ~/humanoid_repo/jetson/scripts/cop_from_pads.py --identify` (패드 하나씩 누름) → 매달린 무부하 5s: `cop_from_pads.py --live --seconds 5 --save ~/logs/pads_A1_tare_$(date +%H%M).jsonl --no-report` | 8패드 전부 반응, **캡 8개 육안**(0908 왼앞 캡 이탈 오독), 무부하 합 L≈328/R≈62 카운트 근방 | pads tare |
| 4 | IMU | 매달린 채 앞으로 밀기 → 콘솔 fwd **+**(0908 규약), 뒤/좌/우 부호 확인, `fresh` 초록 | 4방향 부호 일치, 정지 시 gyro 잡음 ≤0.01 rad/s | 세션 이벤트 |
| 5 | CAN/모터 피드백 | 콘솔 stage8 status: `status_age` < 0.1 s, 12관절 각·`tau_est` 12개 non-None, 온도 정상 | 스테일 없음 | — |
| 6 | stand_hang_v5 로드 | 콘솔 드롭다운 = stand_hang_v5 → 「브리지 기동」(`--live --checkpoint stand_hang_v5`) → 콘솔 「● 기록 시작」 → PC 에서 폰 영상 시작 + **"기록 시작" 누르며 시각을 소리 내어 말한다**(영상↔로그 정렬) | 브리지 debug 에 `checkpoint=stand_hang_v5`, `obs` 길이 **320**, 블랙박스 파일 생성 | 블랙박스·세션·영상 |
| 7 | policy ARM | 착지(발끝 연직선 조준·평평·트림 全0) → 윈치로 발력 ~30% → hang_stand kp150 홀드 → 핸드오프 → 베이스라인 → **ARM** → `stand` 발사(**hang_arm 체크 안 함** — 접촉 있음) | 중력 게이트 통과(거부되면 IMU·착지 재확인), 블렌딩 1.5 s 완료(`blend_alpha`=1) | 이벤트 |
| 8 | policy 출력 안정 | 10 s 무개입 관찰(70% 지지 그대로) | 발진 가드 미트립, 관절 진동 눈에 안 보임, 무릎 ∓6° 근방 유지, `soft_clamp_total` 증가 없음, fwd 가 단조로 커지지 않음 | — |
| 9 | support 매우 천천히 감소 | 윈치 방출: **발력 5%p 당 10 s 이상**(30→50→70→90%), 각 고원에서 3 s 정지 관찰 | 각 고원에서 abort 조건 없음, 발 들기 없음, 좌우 발력 비 30/70 안 | 발력 % 구간 이벤트(콘솔 메모) |
| 10 | support = 0 | 줄이 **완전히 느슨**(육안 처짐) + 콘솔 발력 ≥ 95%(≥164 N) | 줄 느슨 확인 시각 기록 | 이벤트 「support0」 |
| 11 | 5 s 자립 | 무개입 5 s → 계속 30 s 까지 관찰 | 발력 ≥95%·\|fwd\|,\|lat\| < 10°·발 들기 0·진동 없음 5 s 이상 | 판정 |
회수(순서 엄수): **① 줄 재인장(사람) → ② 스택 종료(`bash ~/rl_calib/stack_down.sh`) → ③ 그 다음에만 `python3 ~/rl_calib/release_all.py`**. 굽힌 관절 즉시 릴리즈 금지. 프로세스 종료는 명시 PID 로만.
동일 착지에서 **재시도 2회**까지(성공 재현 확인). 실패 시 새 학습 없음 — 로그 vs 시뮬 mismatch 규명이 먼저.

## 6. 실시간 abort 조건
**자동(브리지 → STOP_ALL, 블랙박스 event 기록):** 중력 게이트(기동 시 17° 초과) · 발진 가드(관절 8 Hz±3°/주기2 채터/과속 12 rad/s) · 타깃 발진 가드 · 리프트 가드(무접촉 0.6 s 동결; hang_arm 미사용) · 센서 스테일(IMU/발력/status) · 비유한 관측 · stage8 soft/hard 클램프 거부 · estop.
**수동(승윤님 — 즉시 윈치 재인장이 최후 안전망, 손으로 로봇을 잡지 않음):**
- \|fwd\| > 12° 또는 \|lat\| > 10° (v5 기대 fwd +4~5° 는 정상 — 거기서 **더 커지는 단조 추세**가 abort 신호)
- 발력 총합이 5 s 안에 20%p 이상 감소(뒤꿈치 회전 징후) · 한 발 0 N 이 1 s · 발 들기/스텝 시도
- 눈에 보이는 관절 진동(주기 0.5 s 미만) · 이상음 · `tau_est` 한 관절이 한계의 80% 를 3 s 이상 · 모터 온도 경고
- 줄 꼬임/횡력·후크가 몸통을 옆으로 당김 · 콘솔 fresh 지표 적색 · 8단계에서 출력 불안정 → **방출 시작 금지**
**abort 후:** 줄 재인장 → 관찰 정지 → 스택 종료 → release_all → 블랙박스 복사(아래) → 재착지 전 원인 확인.

## 7. 로그 파일 (젯슨) — 분석 전 PC 로 복사
| 로그 | 경로 | 주기 | 내용 |
|---|---|---|---|
| 브리지 블랙박스 | `~/logs/rl_bridge_<YYYYmmdd_HHMMSS>.jsonl` | 50 Hz | timestamp·qpos/qvel 12·targets(joint/motor)·**obs(320)**·**raw_action(12)**·quat·gyro xyz·**rp_deg**·contact·foot_n(L,R)·**foot_raw_sum**·tau_est·temp·hang_state·**osc_tripped/stopped_for_stale/stop_reason/grav_gate**·soft_clamp·events(lift_guard/osc_guard/stop_all) |
| 세션 타임라인 | `~/logs/sessions/session_<stamp>/timeline.jsonl` (+ 프로세스 로그) | 20 Hz + 이벤트 | 발력 %·fwd/lat/tilt·gyro·armed/baseline/live/ckpt/blend/contact/fresh, 조작 이벤트, `session_start.wall`(영상 정렬 기준) |
| 관절·토크·발력 | `~/logs/stiff_A1_<stamp>.jsonl` — `python3 ~/humanoid_repo/jetson/scripts/stiffness_log.py --seconds 600 --hz 20 --save ~/logs/stiff_A1_$(date +%H%M).jsonl` (6단계 직후 별도 터미널) | 20 Hz | 실효강성 분석용 |
| 패드 8채널 원값 | `~/logs/pads_A1_<stamp>.jsonl` — `cop_from_pads.py --live --seconds 600 --save … --no-report` (6단계 직후 별도 터미널) | 센서 주기 | values[8]·raw_ads1115[8]·voltages[8] → CoP·접촉 타이밍 |
| CAN 원문 | `/tmp/live_candump_<HHMM>.log` | 버스 | 통신 두절·재전송 분석 |
| 영상 | 폰 (기록 시작 시 시각 발화·콘솔 화면 한 컷) | 30 fps | `session_start.wall` 과 발화 시각으로 정렬 |
복사: `scp -r mama@172.16.35.89:~/logs/rl_bridge_<stamp>.jsonl mama@172.16.35.89:~/logs/sessions/session_<stamp> mama@172.16.35.89:~/logs/{stiff,pads}_A1_* reports/real_0910_A1/`
1차 분석: `python3 jetson/scripts/analyze_fall.py <블랙박스>` (등급·토크표·루프지연), `cop_from_pads.py --file <pads> --tare <tare>`, `stiffness_log.py --file <stiff>`.

## 8. 첫 시험 핵심 질문 → 어느 로그로 답하나
| 질문 | 데이터 | 시뮬 기대치 |
|---|---|---|
| 1 어느 축이 먼저 흔들리나 | 블랙박스 gyro xyz·rp_deg vs 발력 % 시간축 | 접지 직후 피치, 롤은 0.4~0.8° RMS |
| 2 발목롤 stuck | qpos[10,11] vs targets_deg_joint[10,11] (JOINT_ORDER 좌우 교대: 발목롤 = 10,11) 추종오차·tau_est 부호 반전 없이 정체 | 시뮬 DR 마찰 0.1~0.6 |
| 3 발목피치가 CoM 을 앞으로 | qpos[8,9](발목피치) 변화 vs rp_deg 피치·CoP(패드) 전진 | 정착 피치 −4.9°, CoP 발 중앙 |
| 4 FSR 접촉 타이밍 | foot_n>5 N 시각 vs 패드 원값 상승 vs 영상 | 시뮬 임계 2~20 N DR·5틱 지연 |
| 5 자이로 노이즈 | 정지 구간 gyro std | 실측 0.006 rad/s(학습 ±0.2) |
| 6 액션 채터 | raw_action 스텝 차 RMS(정착 3 s) | 시뮬 0.083(노이즈 하)·0.004(노이즈 0) |
| 7 −5° 평형 피치 | 정착 rp_deg 피치 평균·발력 좌우/앞뒤 비 | −4.9° |
| 8 support 감소 시 CoM 재배치 | 발력 % 구간별 qpos 평균 이동·CoP | 지지비 ↓ 에 따라 무릎·발목 미세 조정 |
