# 실물 시험 데이터 인벤토리 (real-robot test-data inventory)
**작성 2026-09-11 · READ-ONLY 감사 · 목표: 다음 실기(5s 무보조 자립)의 증거로 재사용할 기존 기록의 엄밀한 목록화**

원본 코드/모델/설정은 일절 수정하지 않았다. 로컬 저장소만 조사했고 ssh 하지 않았다.

---

## METHOD + CAVEATS (먼저 읽을 것)

### 규칙 1 — 압력 패드는 접지 판정 전용 (로드셀 아님)
`jetson/scripts/foot_force_adapter_node.py` 를 직접 읽어 확인한 발력 산출식:

```
left_n  = NEWTON_PER_COUNT × max(0, Σ(left 4패드 카운트)  − LEFT_BIAS_COUNTS)
right_n = NEWTON_PER_COUNT × max(0, Σ(right 4패드 카운트) − RIGHT_BIAS_COUNTS)
pct(발력%) = 100 × (left_n + right_n) / BODY_WEIGHT_N
NEWTON_PER_COUNT = 0.02650   (= 172.9N / 6525카운트, 단일 스탠딩 스냅샷 L3356/R3559 에 앵커)
LEFT_BIAS=328, RIGHT_BIAS=62, BODY_WEIGHT_N=172.9
```

즉 **발력 N·발력%·L/R 비·CoP 는 전부 "미보정 패드 카운트 합 × 단일 스칼라"** 다.
패드별 감도 차이(CLAUDE.md·MEMORY 명시), 포화(카운트가 ~900대에서 클리핑됨 — 관측됨), 탈락/캡이탈로 인한 0카운트를 전혀 보정하지 않는다.
→ **패드에서 유도된 발력%·CoP·CoM·L/R 하중비를 측정된 사실로 쓴 결론은 모두 "SUSPECT: 미보정 패드 기반" 으로 표기한다.**
패드에서 **허용되는 사용은 오직 "접지 여부(0 인가 아닌가)"** 뿐이다(코드도 접지판정을 `패드별 (count−bias)>80카운트` 로 별도 정의).

### 규칙 2 — 모든 주장을 3범주로 구분
- **MEASURED** — 원시 센서/로그에서 직접 (예: IMU rp_deg, gyro_rad_s, qpos_rad, tau_est_nm, 접지 비트, 패드 원카운트).
- **COMPUTED** — 공식으로 유도 (공식을 명시). 예: 발력% = 위 식; 추종오차 = `cmd − (act − base)`; 장력 ≈ `mg − 발력`.
- **HYPOTHESIS** — 해석 (예: "뒤꿈치 강체 회전", "CoM 이 지지면 뒤").

### 규칙 3 — 경계를 넘어 기록을 병합하지 않는다
분리 경계: **리마운트 / 모터영점 변경 / 하드웨어맵 변경 / 정책 유무**. 각각 별도 행.
확인된 영점·하드웨어 에폭(줄 하나가 경계):
- **E0** 2026-07-27~08-02 (calib_data): 발목 리마운트 **이전**, 구 발력스케일(npc 0.01804/117.7N). 정책 없음.
- **E1** 2026-09-07 (real_0907): 발목 모터 기구적 **리마운트**(발목피치 왼+10.49°/오+15.89°) 후 재앵커. 왼 힙R·무릎 +36°창 시프트 보정. 정책 없음.
- **E2** 2026-09-08 (real_0908): m3·m4 창 k=−1 재앵커(5.25/11.20), m12(발목롤) 중력 미정착. 정책 없음(hang_stand kp150 홀드).
- **E3** 2026-09-10 (real_0910_A1): **정책 stand_hang_v5 투입**. trial1 = session_154826. trial2·3 = session_162605 (이 세션 안에서 `zero_162605`·`zero_162817` 로 **재앵커** → trial2/3 은 trial1 과 영점 에폭이 다름).

### 규칙 4 — 신호는 배열 순서가 아니라 관절 이름으로 매칭
`rl_walking/deploy/policy_runner.py` `JOINT_ORDER` 는 **좌우 인터리브**:
`[Lhip_f, Rhip_f, Lhip_a, Rhip_a, Lhip_r, Rhip_r, Lknee, Rknee, Lankle_f, Rankle_f, Lankle_r, Rankle_r]`.
`rl_bridge` 블랙박스의 `qpos_rad·qvel_rad_s·tau_est_nm·raw_action·obs`(list[12]/[320])는 이 순서다(딕셔너리 순 아님).
`targets_deg_joint·targets_deg_motor` 는 관절명 키 딕셔너리(12개 전수). **A1 최초 분석이 0–5=왼/6–11=오른 으로 오독해 "비대칭 자세·오른무릎 영점 10°" 오결론을 냈다가 철회**(plan_next_test §0). 이 인벤토리의 관절 수치는 전부 JOINT_ORDER 로 재매핑해 확인함.

### 부수 CAVEAT (문서·코드 드리프트)
- **체중 상수**: CLAUDE.md 는 "판정 1순위 = 발력 총합 % (체중 **117.7N** 대비)". 그러나 배포 코드·0908·0910 리포트는 전부 **172.9N**(17.62kg 저울 0904). MEMORY 는 172.9N 이 현재 진실, 117.7N(12kg 추정) 폐기로 확인. → 발력% 절대값은 어느 분모를 썼는지에 따라 ×1.469 차이.
- **로컬 유일 발력보정 파일 `jetson/calib_data/foot_force_calib_20260802.json` 는 폐기된 구값**(npc 0.01804, 117.7N) 을 담고 있다. 배포 코드(0.02650)와 불일치 — 로컬 유물은 낡음.
- **타임스탬프 정합**: pads/rl_bridge/timeline = epoch(`timestamp`/`ts`). stiff_* 로그 = **상대초 `t`**(0 부터). 파일 간 동기화는 별도 정렬 필요.

---

## 발견 기록 수 (per directory)

| 디렉터리 | 로컬 존재 | 데이터 기록 수 | 비고 |
|---|---|---|---|
| `reports/real_0907/` | ✅ (README만) | **0 raw** | 원시 세션은 젯슨 `~/logs/sessions/session_20260907_212628` — **로컬에 없음** |
| `reports/real_0908/` | ✅ | **12 raw** (jsonl×6, gz×2, txt×2, jpg×2) + README | 정책 없음(hang_stand 홀드), 착지 4회 |
| `reports/real_0910_A1/` | ✅ | **9 raw 데이터** (rl_bridge×3, pads×1, stiff×1, timeline×2, session 텍스트로그 다수) + 문서 6 | 정책 v5, 시도 3회 |
| `jetson/calib_data/` | ✅ | **14 파일** (json×11, log×2, tgz×2) | E0 에폭, 발목 리마운트 이전·구 발력스케일 |
| (참고) `reports/real_*` glob | — | real_0907 / real_0908 / real_0910_A1 **3개뿐** | `hang2stand_0908/` 은 시뮬 eval(실기 아님, 범위 밖) |

---

## 인벤토리 — E3: reports/real_0910_A1 (2026-09-10, 정책 stand_hang_v5)

> 정책 유무 경계로 다른 에폭과 절대 병합 금지. trial1(session_154826) ↔ trial2·3(session_162605, 중간 재앵커) 도 영점 경계로 분리.

| Record (경로) | Date | HW/asset·질량 | 영점/hwmap | 정책 | 게인/러너/배포 | 가용 신호 (MEASURED 굵게) | 결측/사망 신호 | 1줄 요약 |
|---|---|---|---|---|---|---|---|---|
| `trial1/rl_bridge_20260910_154841.jsonl` (15832줄) | 09-10 16:15–16:20 | UNKNOWN — 로그에 asset/URDF/질량 없음 (문맥상 17.62kg·final3) | session_154826 (재앵커 후 12/12, 로그엔 값 없음 → UNKNOWN) | **stand_hang_v5** (`checkpoint` 필드 확인) | kp150 홀드→v5 블렌딩; 러너 JOINT_ORDER; obs 320계약 (로그에 obs[320] 존재) | **quat_wxyz, gyro_rad_s, rp_deg, qpos_rad[12], qvel_rad_s[12], tau_est_nm[12], raw_action[12], targets_deg_joint/motor(관절명 12), contact[2], obs[320], soft_clamp, grav_gate** · foot_n[2]/foot_raw_sum[2](=패드) | `osc_tripped`·`stop_reason`·`last_reject_reason`=None (미발생); foot_n=SUSPECT | 20Hz 풀 블랙박스, 정책 ON. **가장 풍부한 재사용 증거.** lift_guard(양발무접촉0.6s)로 정지 |
| `trial2/rl_bridge_20260910_163011.jsonl` (7994줄) | 09-10 16:31–16:35 | UNKNOWN | session_162605 **재앵커됨**(`zero_162605`,`zero_162817`) | **stand_hang_v5** | 상동 | 상동 | 공중발사; hang_guard(120s초과)+lift_guard 정지 |
| `trial3/rl_bridge_20260910_164841.jsonl` (7646줄) | 09-10 16:48–16:51 | UNKNOWN | session_162605 (trial2 와 동일 세션·영점) | **stand_hang_v5** | 상동 | 상동 | 공중발사; lift_guard 정지 |
| `stiff_A1_1540.jsonl` (44841줄, ~20Hz, 상대 t 0–2400s, pct 0–65.1) | 09-10 (파일명 15:40) | UNKNOWN | UNKNOWN (로그 무) | **정책과 병행 기록** (stiffness_log 는 구독전용) | **joints{12, 관절명키}: act·cmd·des·base·tau·vel·kp·stale · fwd(트렁크피치) · fn[2]** · pct | `fwd` 일부 None 가능; fn/pct=SUSPECT | 20Hz 관절·토크·트렁크피치 전 세션. 관절모델 ID 최상급 |
| `pads_A1_1540.jsonl` (23880줄, epoch 15:40:24–16:20:24) | 09-10 | — | — | — (센서 기록) | **left[4]·right[4]·values[8]·raw_ads1115[8]·voltages[8] (원카운트)** | **trial1 구간만 포함 — trial2(16:31)·trial3(16:48) 미포함** (아래 SUSPECT S5) | 패드 8ch 원값. **접지판정에만** 사용 가능 |
| `trial1/session_20260910_154826/timeline.jsonl` (25784줄) | 09-10 | — | — | — | **kind='s' 20Hz 텔레메트리(25743) · action/action_start(라벨·pid·ts) · proc_exit(rc·dur)** | 없음 | 조작·프로세스·텔레메트리 통합 타임라인. 이벤트 시각정합용 |
| `trial2/session_20260910_162605/timeline.jsonl` (36404줄) | 09-10 | — | — | — | 상동 + `zero_*`·`stack_down`·`release` 액션 | 없음 | trial2·3 세션 타임라인(재앵커·회수 포함) |
| session 텍스트 로그 (`hang_stand_*.log`, `stage8_*.log`, `bridge_*.log`, `boot_pose_*.log`, `zero_*.log`, `sensors_*.log`, `stack_down_*.log`, `release_*.log`) | 09-10 | — | — | — | 콘솔 stdout (홀드오차·게이트·rc) | 수치 텔레메트리 아님 | 절차·게이트 서사 (boot_pose 12/12, 홀드오차 등) |

---

## 인벤토리 — E2: reports/real_0908 (2026-09-08, 정책 없음 / hang_stand kp150 홀드)

> session_20260908_143316. 착지 4회. 각 착지·측정은 별도 행. m3·m4 재앵커(5.25/11.20), m12 미정착.

| Record | Date | HW/asset·질량 | 영점/hwmap | 정책 | 게인/러너 | 가용 신호 (MEASURED 굵게) | 결측/사망 신호 | 1줄 요약 |
|---|---|---|---|---|---|---|---|---|
| `data/stiff_landing3_joints.jsonl.gz` (3312줄, 18.2Hz, 182s, pct 0–72) | 09-08 (착지3) | UNKNOWN(문맥 17.62kg) | E2 (5.25/11.20 재앵커) | 없음 (stage8 홀드) | hang_stand **kp150**/kd5 | **joints{12 관절명키}: act·cmd·des·base·tau·vel·kp·stale** · fn[2] · pct · armed | `fwd`=None(트렁크피치 미기록); joints None필드 ~13.6%; fn/pct=SUSPECT | 착지3 20Hz 관절·토크. **"관절 안 휨" 검증됨(아래).** 최상급 강성증거 |
| `data/stiff_landing4_joints.jsonl.gz` (3200줄, 18.3Hz, 175s) | 09-08 (착지4) | UNKNOWN | E2 | 없음 | kp150 | 상동 (`fwd`=일부 float) | fn/pct=SUSPECT | 착지4 관절·토크. 발산·회수(−16°) 구간 포함 |
| `data/imu_4dir_push.jsonl` (2340줄) | 09-08 | — | E2 | 없음 | **fwd·lat·tilt·grav[3]·yaw·age** (IMU, 매달림 손밀기) | 없음 | 4방향 손밀기 IMU 부호검증. fwd[−9.0,+22.3] lat[−20.8,+24.4] — 부호규약 일치 확인 |
| `data/landing1_foot_imu.jsonl` (2069줄, 5Hz, pct 0–68) | 09-08 (착지1) | — | E2 | 없음 | **fwd·lat·tilt** (IMU) · L·R·pct(=패드) | L/R/pct=SUSPECT | 착지1 5Hz. **fwd(피치)[−16.1,+0.7]** — 리포트 "최대후방 −16.1°" 와 정확 일치 |
| `data/landing2_foot_imu.jsonl` (4432줄, 5Hz, pct 0–72) | 09-08 (착지2) | — | E2 | 없음 | **fwd·lat·tilt** · L·R·pct | L/R/pct=SUSPECT | 착지2. **fwd[−17.3,+3.3]** — 리포트 "−17.3°" 일치 |
| `data/pads_landing1_sweep.jsonl` (3991줄) | 09-08 (착지1) | — | — | — | **left[4]·right[4]·values[8]·raw_ads1115·voltages (원카운트)** | **왼 index0·index2 = 전구간 0** (캡 이탈, 리포트와 일치) | 착지1 패드 8ch. 접지판정용 |
| `data/pads_landing4_sweep.jsonl` (1912줄) | 09-08 (착지4) | — | — | — | 상동 + `t_rx` | **왼 index0·index2 = 전구간 0** | 착지4 패드 8ch |
| `data/pads_identify.jsonl` (1301줄) | 09-08 | — | — | — | 원카운트 8ch (손가락 개별 가압) | 왼 index0 max=396(손누름엔 반응) | 패드 식별(레이아웃 lf,lb,rf,rb 확정). 캡없이 센서는 살아있음 |
| `data/pads_tare_hang.jsonl` (80줄) | 09-08 | — | — | — | 원카운트 8ch (무부하 매달림) | 우 index0 만 소량(max 11), 나머지 0 | 무부하 바이어스. 정적·초단 |
| `data/alarm_landing3.txt` / `alarm_landing4.txt` (25/24행) | 09-08 | — | E2 | 없음 | t · 발력%(=패드) · **fwd(피치)** | 발력%=SUSPECT | −3° 문턱 알람 트레이스. 짧음 |
| `data/hang_rig_photo.jpg` / `hang_rig_eyes.jpg` | 09-08 | — | — | — | 현가 리그 사진(아이볼트·2줄) | — | 뒷줄 아이볼트↔발목축 거리(미측정) 참조 이미지 |

**E2 핵심 MEASURED 검증(착지3, 재계산 `err=cmd−(act−base)`):** 하중 0→72% 전 구간에서 관절 추종오차 최대 = `left_ankle_f 0.284°`(리포트값 0.284 정확 일치), 그 외 전 관절 |err|≤0.153°, |τ|≤0.665Nm. → **"관절은 안 휘었다 / 강성 무죄" 는 깨끗한 측정으로 뒷받침됨**(이 인벤토리에서 재현 확인). 단, 이 로그에 트렁크피치(fwd)=None 이라 "몸통 3.52° 회전" 비교는 이 파일만으로는 불가.

---

## 인벤토리 — E1: reports/real_0907 (2026-09-07, 정책 없음)

| Record | Date | HW/asset | 영점/hwmap | 정책 | 신호 | 결측 | 요약 |
|---|---|---|---|---|---|---|---|
| `reports/real_0907/README.md` (서사만) | 09-07 21:26 | UNKNOWN | E1: 발목 리마운트+재앵커(12/12) | 없음 | **로컬 원시 로그 없음** — 젯슨 `session_20260907_212628`(465s,9197샘플,20Hz) 미회수 | 전부 | 착지 중단·하중 미완. hang_stand 홀드 오차 최대 0.24°(서사). **재사용할 로컬 원시데이터 0** |

---

## 인벤토리 — E0: jetson/calib_data (2026-07-27~08-02, 발목 리마운트 이전, 구 발력스케일)

> 이 에폭은 E1 의 발목 리마운트·다중 재앵커 **이전**이라 현재 로봇과 영점·발목기구가 다르다. 값은 참조상수(부호·ROM·기본자세)로만 유효, 현재 자세/발력 판정에 직접 쓰면 안 됨.

| Record | Date | 내용/신호 | 관절 매핑 | 재사용 가치 |
|---|---|---|---|---|
| `sign_verification_20260727.json` | 07-27 | 모터별 방향부호(홀수+1/짝수−1, m12=+1 예외) | motor_id→관절명 명시 | ✅ 부호규약 참조(현재도 유효 가능성 높으나 미재확인) |
| `stand_snapshot_v{1..4}_20260727.json` | 07-27 | 12모터 영점/자세 스냅샷(motor_id+관절명, min≈max=정적) | 관절명 포함 | 구 영점 스냅샷. **현재 영점 아님**(E1~E3 재앵커) |
| `stand_hold_20260727.log` (64KB) | 07-27 | 정적 홀드 로그 | — | 단일 자세 홀드, 동역학 없음 |
| `rom_sweep_20260727.log`·`rom_sweep2/salvage_20260727.json`·`rom_symmetry_analysis_20260727.json` | 07-27 | ROM 스윕·좌우대칭 분석 | pairs 관절명 | ROM 한계 참조 |
| `ak45_scale_verification_20260727.json` | 07-27 | AK45(발목롤) 위치스케일 검증 | 모터6·12 | 발목롤 스케일 참조 |
| `hang_snapshot_20260727.json` | 07-27 | 매달림 자세 스냅샷 | 관절 | 매달림 기준자세(구) |
| `foot_force_calib_20260802.json` | 08-02 | **구 발력보정: npc 0.01804, 체중 117.7N(12kg)** | — | ⚠ **폐기값** — 배포코드는 0.02650/172.9N. 단일점 앵커 |
| `sim_default_stand_targets_20260727.json` | 07-27→08-01 | 시뮬 기본자세 목표각(v2 프레임불변, 생성시 stand_zero 동봉) | motor_id | 목표자세 참조 |
| `jetson_backup_*.tgz`·`jetson_humanoid_ws_src_*.tgz` | 07-27 | 코드 백업 아카이브 | — | 데이터 아님 |

---

## SUSPECT CONCLUSIONS (미보정 패드 발력%/CoP 또는 트렁크피치 단독 방향추론 기반)

| # | 결론 (인용) | 파일 | 왜 SUSPECT 인가 |
|---|---|---|---|
| **S1** | "로봇은 체중의 **68~72%** 를 넘기지 못하고 뒤로 넘어간다(4회 재현)" · "1·2회차 임계 하중분율 **68.6% / 69.4%**" | real_0908/README.md | 발력% = `0.0265×(패드카운트−bias)/172.9` (미보정 패드·단일점 앵커). 패드 포화·왼앞 2채널 사망(0)으로 카운트가 구조적으로 낮게 나옴 → "천장 68~72%" 는 로드가 아니라 **패드가 셀 수 있는 최대**일 수 있다. COMPUTED, SUSPECT |
| **S2** | (헤드라인) "**뒤꿈치 강체 회전** → 뒤로 넘어간다" vs (같은 폴더 후속) "실패 = **방출 중 앞으로(발끝) 쏠림** … '뒤로' 는 트렁크피치만 본 오독" | real_0910_A1/README.md·plan_next_test.md ↔ success_prep.md | **방향(앞/뒤)이 폴더 내에서 정반대로 결론남**. '뒤' 는 트렁크피치(rp) 단독, '앞' 은 미보정 패드 CoP. 어느 쪽도 독립 로드 측정이 아님. HYPOTHESIS, 미해결 |
| **S3** | "고원 62%: **CoP +8mm(중앙)** … 방출 62→10%: CoP **+8→+70mm(발끝 edge)**, 뒤패드 0" (앞뒤 CoP 궤적으로 방향 확정) | real_0910_A1/success_prep.md (★★, ★★★) | CoP = 미보정 패드카운트 × 패드좌표(앞+70.6/뒤−50.9). **게다가 같은 문서 §7 이 "p1 이 앞인지 뒤인지 라벨 미확정" 이라고 명시** — 부호 있는 CoP 궤적을 계산할 좌표 자체가 미확정. COMPUTED, SUSPECT(이중) |
| **S4** | "**좌우 30:70 쏠림** … 하중은 오른발에 2배 이상" · "총합이 체중의 **72%** 를 넘지 못함" | real_0907/README.md | L/R 하중비·발력% 모두 미보정 패드. 패드 감도 좌우차·왼앞 캡이탈이 곧장 "쏠림" 으로 보임. (0908·A1 모두 "좌우비는 무효" 로 자기부정) SUSPECT |
| **S5** | "index3 은 **세 발 모두 상시 0 = 탈락 패드**(확정)" | real_0910_A1/plan_next_test.md §7 | 원시 검증 결과 **모순**: `pads_A1_1540.jsonl` 는 16:20:24 에 끝나 **trial2(16:31)·trial3(16:48) 을 아예 포함하지 않음** → "세 발 모두" 를 이 파일로 입증 불가. 또 왼 index3 은 파일 전체(15:40–16:20)에서 **7218/23880 표본 비영**(max 646). 스탠스 중 0 을 "센서 사망 확정" 으로 승격 — 패드규칙 위반 |
| **S6** | "**방향 = 뒤로(BACKWARD, 확신 높음)**: 피치가 −5°에서 +1~2°로" | real_0910_A1/success_prep.md (실패트리거 서브2) | 트렁크피치(IMU rp) 단독 방향추론. 같은 문서 ★★★ 가 "발목이 회전하며 트렁크는 펴짐 → 오독" 으로 자기반박. HYPOTHESIS |
| **S7** | "x_com = f_c·x_뒤패드 + (1−f_c)·x_뒤아이 = −35 + 0.305·x_뒤아이 (f_c=0.695)" (CoM 위치 역산) | real_0908/README.md | 모멘트 균형에 **f_c=0.695(=발력분율, 미보정 패드)** 를 하중배분으로 대입 → CoM(mm) 역산. 입력이 SUSPECT 라 출력 mm 도 SUSPECT. COMPUTED |
| **S8** | "왼발 합력이 5N 아래로 가서야 접촉0 → 왼발이 먼저 뜬다(횡 오프셋)" (초기 서사) | real_0910_A1/plan_next_test §1, summary_quick foot_n | foot_n(패드) 좌우 스케일차의 인공물. 후속(§7·success_prep)에서 "패드 독립센서는 좌우 1.00, per-foot foot_n 캘리브 오류" 로 정정. 초기 '왼발 먼저' = SUSPECT |

> 위와 대조적으로 **신뢰 가능**: real_0908 "관절은 안 휘고 토크가 없다(강성 무죄)" 는 관절 추종오차·τ(MEASURED)로 이 감사에서 재현·확인됨. 단 여기서 파생된 "그러므로 CoM 전후가 본선/뒤" 는 여전히 HYPOTHESIS.

---

## REUSABLE EVIDENCE (깨끗한 MEASURED 신호 — 제어/관측모델 식별용)

### 최상급 — 모델 식별에 바로 쓸 수 있음
| Record | 왜 좋은가 | 주의 |
|---|---|---|
| **`real_0910_A1/trial{1,2,3}/rl_bridge_*.jsonl`** (20Hz, 정책 v5 ON) | **명령목표(targets_deg_joint/motor, 관절명 12) + 측정상태(qpos/qvel) + tau_est + raw_action + obs[320] + IMU(quat·gyro·rp) + 접지비트** 동기 기록. **배포 레짐(정책 ON)과 동일** → 관측모델·액추에이터모델·sim2real 관측괴리 식별에 이상적. 정숙성 지표(gyro std 0.006, action-rate 0.002)도 여기서 나옴 | qpos/qvel/tau/raw_action/obs = **JOINT_ORDER 배열** → 관절명 재매핑 필수. foot_n=SUSPECT(접지비트만) |
| **`real_0910_A1/stiff_A1_1540.jsonl`** (20Hz, 44840행, 관절명키) | 전 A1 세션 12관절 `cmd·act·base·tau·vel·kp` + 트렁크피치(fwd). PD 추종·마찰·백래시·강성 식별에 최적. 관절이 명령 지키는지 확인 | 상대초 `t`; fn/pct=SUSPECT |
| **`real_0908/data/stiff_landing3_joints.jsonl.gz`** (18Hz, 관절명키) | 하중 0→72% 에서 **관절이 0.3° 안에서 명령유지·τ<0.7Nm 임을 이 감사가 재현확인**. 가장 깨끗한 토크/강성 증거 | 정책 없음(홀드); fwd=None(트렁크피치 미기록); fn/pct=SUSPECT |
| `real_0908/data/stiff_landing4_joints.jsonl.gz` | 상동 + −16° 발산·회수 구간 → 대변위 관절거동 | 상동 |
| **`real_0908/data/imu_4dir_push.jsonl`** (2340행) | 4방향 손밀기 IMU fwd/lat/tilt/grav/yaw → **IMU 장착부호·마운트각·편향 식별**(부호뒤집힘 기각에 이미 사용) | 손밀기 각도 오차 혼입 |

### 보조 — 일부 신호만, 저레이트/단시간
| Record | 쓸 수 있는 것 | 한계 |
|---|---|---|
| `real_0908/data/landing{1,2}_foot_imu.jsonl` (5Hz) | **fwd/lat/tilt(IMU) 착지 피치궤적** — fwd 범위가 리포트와 정확 일치(검증됨) | 5Hz 저레이트, 짧음; L/R/pct=SUSPECT |
| `real_0910_A1/trial{1,2}/…/timeline.jsonl` (20Hz) | 조작 이벤트 시각(label·pid·ts), proc_exit rc, 재앵커/회수 마커 → **이벤트 정합·절차 재구성** | 's' 텔레메트리는 bridge 보다 저정밀; 그 자체로 모델ID 불가 |
| `real_0910_A1/pads_A1_1540.jsonl`, `real_0908/data/pads_*` | 패드 **원카운트 → 접지 여부(0/비0)·접지순서** 만 | 발력·CoP·비율 금지(SUSPECT). A1 파일은 trial1 만 |

### 식별에 부적합 — 너무 정적/짧음/에폭불일치
- `jetson/calib_data/*` (E0): 발목 **리마운트 이전** 에폭 + 구 발력스케일 → 부호·ROM·기본자세 **참조상수**로만. 정적 홀드/스윕이라 동역학 식별 불가.
- `real_0908/data/pads_tare_hang.jsonl` (80행), `pads_identify.jsonl`: 바이어스/식별용, 동역학 없음.
- `real_0908/data/alarm_landing{3,4}.txt`: 발력%(SUSPECT)+피치 짧은 트레이스.
- **`real_0907/`: 로컬 원시데이터 0** (README 서사만; 세션은 젯슨). 회수 전엔 재사용 불가.
- session 텍스트 로그(`*.log`): 게이트·절차 서사, 수치 텔레메트리 아님.

---

## 다음 실기(5s 자립)를 위한 실무 함의 (인벤토리에서 직접 나오는 것)
1. **방향(앞/뒤)이 미확정**이고, 그 판정에 쓰인 모든 근거(발력%·CoP·트렁크피치)가 SUSPECT. plan/README(뒤)와 success_prep(앞)이 충돌 → **로드셀 없이 방향을 확정한 척하는 결론 위에 시험/학습 설계 금지.** 필요한 것은 접지순서(패드 0/비0)·동기영상, 그리고 패드 앞/뒤 라벨 확정(`cop_from_pads.py --identify`).
2. **재사용 1순위 = A1 rl_bridge(v5) + stiff_A1** — 정책 ON 배포레짐의 명령목표↔측정상태↔토크↔IMU 동기 기록. 관측모델/케이블지지 갭 식별의 실측 앵커.
3. **강성 무죄는 확정적(measured)** — 다음 세대에서 강성/컴플라이언스 재수술은 근거 없음. 남은 미지수는 지지기하(CoM·현가), 이는 패드로는 못 잰다.
4. 발력% 를 계속 판정에 쓰려면 **분모(117.7 vs 172.9)와 npc(0.01804 vs 0.02650) 를 기록마다 명시**하고, 여전히 로드셀이 아님을 표기.
