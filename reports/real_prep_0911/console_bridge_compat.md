# 콘솔 ↔ 브리지 호환 적대검수 (실물준비 item 7) — 2026-09-11

대상: 오늘 갱신된 콘솔(`jetson/scripts/operator_console.py`, `console_state.py`,
`console_ui.html`, `console_demo.py`, `tests/test_operator_console.py`,
`docs/operator_console_renewal.md`)과 **같은 날 따로 바뀐 브리지**
(`jetson/scripts/rl_bridge_node.py`, 커밋 `2a88452` — stand 인계 워밍업 추가).

질문: **"브리지는 살아 있지만 정책은 아직 구동 전"인 4초 창을 콘솔이 어떻게 읽는가.**

모든 주장에 `MEASURED`(원시 로그/코드 실행) · `COMPUTED`(코드에서 도출, 식 명시) ·
`HYPOTHESIS` 를 붙였다. 상관은 인과로 쓰지 않았고, 구분 불가한 설명은 여러 가설로 남겼다.

---

## 0. 검증 방법과 한계

**한 일** (전부 이 저장소 안, 읽기 전용 + 오프라인 실행):

| 무엇 | 결과 |
|---|---|
| 코드 정독 | 콘솔 3종 + 브리지 + `stage8_12axis_mit_control_node.py` + `hang_stand.py` / `release_all.py` / `stage8_cmd.py` 의 출력 계약 |
| `python3 -m pytest -q tests/test_operator_console.py` | **35 passed in 0.17s** (§4) |
| 오프라인 프로브 A — stage8 코어 | `/tmp/.../scratchpad/stop_probe.py`, `prime_after_stop.py`. 모듈 전역 `REAL_CAN_WRITE_ENABLED=False`, `READ_ONLY_CAN_ENABLED=False` 로 CAN 경로 전면 차단(버스 오픈·송신·수신 전부 미호출) |
| 오프라인 프로브 B — 콘솔 상태기 | `/tmp/.../scratchpad/warmup_probe.py`. `console_state.TrialState` 순수 상태기에 워밍업 창 뷰를 주입 |

**안 한 일**: 젯슨 접속, ROS 기동, 실물, `--demo` 서버 실행, 파일 수정(0건).

**구조적 한계 (UNVERIFIED)**
- 콘솔이 실제로 띄우는 스크립트 중 `hang_stand.py` · `boot_pose_check.py` ·
  `release_all.py` · `stage8_cmd.py` · `can_up.sh` 는 `~/rl_calib/` (젯슨) 사본이다.
  이 PC에 `~/rl_calib` 는 **없다**. 아래 문자열·종료코드 계약은 **저장소 사본**으로만
  확인했다. 배포본이 다르면 콘솔은 "완료 확인 보류 → 시간 초과"로 떨어진다.
- 커밋 메시지의 실측치(“status 갭 최대 2.9 s”, “0910 3회 정상 / 금일 2회 실패”)의
  원본 로그(`~/logs/rl_bridge_*.jsonl`, 세션 타임라인)는 이 PC에 없다 → 재확인 불가.

---

## 1. 콘솔이 실제로 파싱하는 것 (계약 전수)

### 1.1 ROS 구독 필드 (`Telemetry.receive` / `Telemetry.view`)

| 토픽 | 쓰는 필드 | 무효 판정(= `fresh=False`) |
|---|---|---|
| `/humanoid/stage8_12axis_mit_status` | `timestamp`, `armed`, `running`, `accepted`, `last_command`, `command_seq`, `errors`, `joints{baseline_deg, latest_actual_deg, feedback_valid, feedback_stale, feedback_age_sec, commanded_relative_deg, desired_relative_deg}`, `takeover{waiting, holding, hold_joint_count, hold_age_sec, last_event}` | `armed/running/accepted` 중 하나라도 bool 이 아니면 status 전체 무효 (op_console:566) |
| `/humanoid/imu` | `timestamp`, `quat`\|`quat_wxyz`(4, 유한, 0.8<‖q‖²<1.2), `gyro_rad_s`\|`gyro`(3, 유한) → `fwd/lat/tilt/yaw/grav` | 형상·노름 위반 시 무효. 단 **마지막 값은 age 와 함께 계속 표시**(원격 오판 방지) |
| `/humanoid/foot_force` | `timestamp`, `left_contact`/`right_contact`(**진짜 bool 만**), `left_raw_sum`/`right_raw_sum`, `left_pads_on`/`right_pads_on`, (진단용) `left_n`/`right_n` | 접촉 필드가 bool 이 아니면 접촉 `None` + foot 무효 → **힘 값으로 접촉을 만들지 않는다** (테스트 2건이 고정) |
| `/humanoid/rl_bridge_debug` | `timestamp`, `live`, `armed`(둘 다 bool 필수), `checkpoint`, `blend_alpha`, `contact`, `hang_state`, 그리고 fault 트리거 `stop_reason`\|`osc_tripped`\|`stopped_for_stale` | bool 아니면 무효 |
| 〃 (별도 형상) | `{event: stop_all\|lift_guard\|osc_guard, reason\|why}` → `bridge_fault` 래치, **`t[]` 나이는 갱신 안 함** | — |

공통 신선도: **도착 시각 기준 `STALE_SEC=1.5 s`** *그리고* **메시지 자체 `timestamp`
기준 1.5 s** (미래 0.5 s 초과도 거부). `foot` 는 접촉 필드 유효성까지 AND.

발행(콘솔 → 장비)은 두 개뿐: `/humanoid/rl_bridge_command` 에
`{"cmd":"stand","hang_arm":bool}` / `{"cmd":"estop"}`, `/humanoid/stage8_12axis_mit_command`
에 `{"command":"STOP_ALL"}` (estop 전용). 베이스라인/ARM 은 `stage8_cmd.py` 서브프로세스
경로다. 게인램프/접지/핸드오프는 마커 파일(`/tmp/hang_gain`, `/tmp/hang_ground`,
`/tmp/hang_stand_handoff`).

### 1.2 프로세스 stdout 문자열 계약 (`Proc._pump`, op_console:285–301)

| 정규식/부분문자열 | 산출 | 생산자(저장소 사본 확인) |
|---|---|---|
| `\[(에어유지\|에어램프\|게인램프\|운용대기\|하중유지)\s+[^]]*\]\s+kp\s*([\d.]+)` | `hold={phase,kp,ts}` | `hang_stand.py:509` `say(f"[{phase} {t:5.1f}s] kp{kp:5.1f} 트림 …")` ✓ MEASURED |
| `RL 브리지 기동.*LIVE.*checkpoint=([A-Za-z0-9_]+)` | `loaded_checkpoint` | `rl_bridge_node.py:1351-1353` ✓ |
| `"블랙박스 기록: "` (bridge 키만) | `blackbox_path` | `rl_bridge_node.py:1342` ✓ (live 일 때만 출력) |
| `"selftest: ALL PASS"` | `all_pass` | `rl_bridge_node.py:1225` ✓ |
| `"STOP_ALL ("` (bridge 키만) | `fault = "STOP_ALL (…)"` | `_stop_all()` 의 `get_logger().warning(f"STOP_ALL ({reason})")` ✓ — **사유 무관 포괄 매칭** |
| `"12모터 소프트 릴리즈+MIT 해제 완료"` (release 키) | `release_ok` | `release_all.py:226` ✓ |
| `"송신실패"` \| `"읽기 실패"` (release 키) | `release_error` | `release_all.py:224,190` ✓ |

stderr 는 `stderr=subprocess.STDOUT` 으로 합류하므로 rclpy 로거(WARN/ERROR)도 잡힌다.

### 1.3 종료코드

- `rc == 3` : hang_stand 핸드오프 성공의 **필수** 조건 (`hang_stand.py:532 return 3`) ✓
- `rc == 0` : boot_pose / selftest / release / analyze / stage8_baseline / stage8_arm
- `rc ∉ (None,0,3)` : 즉시 실패 `"프로세스 종료 코드 N"` (SIGTERM 시 −15 도 여기)
- **`rc==0` 만으로는 릴리즈를 성공으로 안 본다** — `release_ok` 출력 + `release_error` 부재 필수
  (CLAUDE.md 규칙 5 "산출물로 판정한다" 가 코드로 들어가 있음)

### 1.4 단계 완료 술어 (`TrialState.update`, console_state:310–356)

| 단계 | 완료 술어 |
|---|---|
| sensors | IMU·발접촉 모두 신선 |
| stage8 | 새 status(요청 후) + `takeover.waiting` + 프로세스 생존 |
| hang_stand / gain_ramp / ground | 프로세스 생존 + `hold.ts` 가 요청 이후 + 3 s 이내 + phase(`에어유지`/`운용대기`\|`하중유지`/`하중유지`) (+gain_ramp 는 `kp≥150`) |
| handoff | 프로세스 **죽음** + `rc==3` + 새 status + `holding` + `hold_joint_count==12` |
| baseline / arm | 새 status + `rc==0` + `accepted` + `last_command=="SET_BASELINE_FROM_CURRENT_ALL"\|"ARM_ALL"` + `command_seq > 요청직전 seq` + (baseline 12) |
| bridge | 프로세스 생존 + **stdout 로 확인한** `loaded_checkpoint == 선택 정책` |
| **stand** | `fresh.bridge` **and** `received.bridge > 요청시각` **and** `bridge.armed` **and** `bridge.checkpoint==선택` **and** 새 status **and** `stage8.armed` **and** `stage8.running` |
| stack_down | 추적 프로세스 전멸 + `fresh.status==False` + `fresh.bridge==False` |
| release | `rc==0` + 죽음 + `release_ok` + `release_error` 없음 |

공통: 30 s(점검류 300 s) 안에 술어가 성립 안 하면 실패 래치.

### 1.5 phase / owner 결정 순서 (console_state:358–379)

`release 완료` → `fault` → `측정 중단` → **`fresh.bridge && bridge.armed && fresh.status && stage8.running` → "정책 제어 · 지지 제거"(slack 전) / "자립 관찰"(slack 후)** → `stage8.armed` → "정책 시작 준비" → `stage8.holding` → "정책 인계" → hang_stand 생존 → "공중 자세 유지" → `takeover.waiting` → "자세 유지 준비" → "준비 확인".

### 1.6 정지 / 동결 / 토크 해제의 구분

- **정지 요청(estop)**: 게이트 0개. 브리지 `estop` + stage8 `STOP_ALL` 동시 발행,
  추적 중인 ARM/기준점 **재시도 송신자만** 종료(홀드 프로세스·모터는 안 건드림).
  즉시 `fault` 래치 + pending 취소. 응답은 `stop.delivery`(발행 성공 여부)로 분리.
- **목표 동결 확인**: `last_command=="STOP_ALL" && accepted && command_seq>요청직전 && **running==False**` (console_state:382–386). → **§5 R1 에서 이 술어가 구조적으로 거의 성립하지 않음을 실측했다.**
- **토크 해제**: 별도 액션. 순서 강제 = 줄 지지 확인 → 콘솔 소유 프로세스 종료 →
  외부 프로세스 0 + 프로세스 검사 신선(4 s) → 릴리즈. 메시지에 "전송 완료 보고 ·
  실제 토크 상태는 현장 확인 필요" 를 못 박음. 세 개념이 코드·문구에서 분리돼 있다. ✓

### 1.7 측정(무보조) 시작·중단

- 시작: **`op_slack_done` 조작 이벤트 하나뿐** (`slack_at = clock()`, console_state:239).
  선행 조건 = 기록 중 + `stand` 완료 + `fresh.bridge` + `bridge.armed` + IMU/발/제어기
  12축 피드백 + **양발 접촉** + 고장 없음 + 미측정. HTTP `confirmed:true` 필수.
- 중단(`invalidate`): 센서 상실, 접촉 상실, 기록 중단, `stage8.armed/running` 또는
  `bridge.armed` 상실, 재인장/외부보조/방출조작/줄지지확인, 모든 `latch(fault)`.
- **성공 판정 없음** — `observed_5s`/`observed_30s` 플래그에 전부 `review_required`.

---

## 2. 워밍업 창이 콘솔을 속일 수 있는가

### 2.0 워밍업 동안 브리지가 내는 것 / 안 내는 것 (MEASURED, 코드)

`_tick()` 은 워밍업 분기에서 **`return` 으로 조기 종료**한다
(`rl_bridge_node.py:1601-1613`). 디버그 발행(`pub_dbg.publish`, 1801)·블랙박스 기록은
그 **뒤**에 있다. 따라서 워밍업 4 초 동안:

- `/humanoid/rl_bridge_debug` : **한 건도 안 나감** → 콘솔 `fresh.bridge=False`
- `/humanoid/stage8_12axis_mit_command` : `SET_JOINT_TARGETS`(전 관절 상대 0) **50 Hz**
- stdout : `"RL stand 시작 ckpt=… (인계 워밍업 최대 4s)"` 1줄, 성공 시 `"인계 워밍업 완료 …"` 1줄
- 타임아웃 시 : `_stop_all("sensor_stale_warmup_timeout")` → stage8 `STOP_ALL` +
  stdout `STOP_ALL (sensor_stale_warmup_timeout)` + `core.active=False`
  → **이후 디버그는 영원히 0건**(틱 첫 줄에서 return)

### 2.1 "정책 미구동인데 stand 완료" → **일어나지 않는다** (MEASURED, 프로브 B)

`stand` 완료 술어의 `fresh.bridge && received.bridge > 요청시각` 은 워밍업 중 절대
성립할 수 없다(디버그 0건). 프로브 B-A: 워밍업 뷰를 0/1/3/6.9 s 주입 →
`stand_done=False`, phase `정책 시작 준비`, owner `12축 제어기 · ARM`, pending 유지,
`op_slack_done` 은 `"정책 제어 시작 응답 확인 중"` 으로 차단. 대조군 B-B(디버그 흐름):
2 s 만에 `stand_done=True`, phase `정책 제어 · 지지 제거`, `op_slack_done` 허용.
→ **콘솔이 워밍업을 "정책 제어 중"으로 잘못 승격하는 경로는 없다.** 이 한 가지는 안전하다.

### 2.2 그러나 워밍업은 **화면에 전혀 표시되지 않는다** (MEASURED)

phase/owner/next 어디에도 워밍업 개념이 없다. 조작자가 보는 것은
`정책 시작 준비 / 12축 제어기 · ARM / 응답 확인 중` 이며, 이는
(a) 정상 워밍업, (b) stand 명령 유실, (c) 브리지 정지 상태와 **화면상 동일**하다.
줄을 잡고 있는 사람은 "지금 기다리면 되는지 / 이미 실패했는지"를 구분할 수 없다.

### 2.3 홀드-프라임은 stage8 계약상 정책 명령과 **구분 불가** (MEASURED, 프로브 A)

```
hold-prime batch:  running=True last_cmd=SET_JOINT_TARGETS accepted=True seq=1
```
콘솔이 기록·표시하는 stage8 필드(`stage8_running`, `stage8_last_command`,
`stage8_command_seq`, 관절 `commanded_relative_deg`/`desired_relative_deg`)만으로는
홀드-프라임과 정책 첫 명령을 사후에도 구분할 수 없다. 정책 첫 명령 역시 온셋
블렌딩(α≈0)으로 "현 명령 유지"에서 출발하므로 **값 자체도 닮는다**.
20 Hz 세션 타임라인(사후 판정의 1차 사료)에 이 4 초 구간의 표식이 없다.

덧붙여 `stage8.running` 은 **ARM 직후부터 항상 True** 다(프로브 A: `after tick (armed):
running=True last_cmd=None`). UI 의 `"명령 실행 중"` 은 "정책이 돈다"가 아니라
"armed 상태로 스트리밍 중"의 뜻이다.

### 2.4 새 정지 사유는 "분류"되지 않고 **포괄 문자열로만** 잡힌다 (MEASURED)

`sensor_stale_warmup_timeout` 전용 처리는 콘솔 어디에도 없다. 잡히는 경로는
`Proc._pump` 의 `"STOP_ALL (" in s` 하나뿐이며, 그 결과 화면에는 원문
`STOP_ALL (sensor_stale_warmup_timeout)` 이 `fault` 로 뜬다(프로브 B-A2 확인).
운영 문구·다음 행동 안내는 없고, 발진가드/리프트가드 정지와 외형이 같다.

### 2.5 브리지가 콘솔 소유가 아니면 **이 정지는 통째로 안 보인다** (MEASURED, 프로브 B-A3)

워밍업 타임아웃은 ROS 로 아무 것도 내지 않는다(디버그 0건, `_stop_all` 의 이벤트 행은
**블랙박스 파일에만** 기록된다 — `_bb_write`, 1862). 유일한 증거가 stdout 이므로,
브리지를 콘솔 밖에서 띄운 경우(0910 현장에서 실제로 쓴 우회 경로, 그리고 `DUP_GUARD`
가 외부 브리지 존재 시 콘솔 기동을 거부하므로 **강제되는 상황**) 콘솔은:

```
t+4.2s fault=None pending=True phase='정책 시작 준비'
t+29.2s fault=None pending=True phase='정책 시작 준비'
after 30 s: {'ok': False, 'message': '확인 시간 초과 — 실제 상태를 확인하세요'}
```
→ **브리지가 포기한 뒤 26 초 동안 무징후**, 그 뒤에도 사유 없이 "시간 초과"만 뜬다.
그 사이 로봇은 STOP_ALL 동결 홀드로 줄에 매달려 있고, 조작자는 pending 때문에
estop·기록 이벤트 외 아무 것도 누를 수 없다.

### 2.6 워밍업 창 = 가드 미평가 구간 (COMPUTED)

`hang_guard.update()` / 발진가드 / `nonfinite` 게이트는 전부 워밍업 `return` **뒤**에
있다. 즉 stand 직후 최대 4 초 동안 리프트 가드(양발 무접촉 `LIFT_STOP_S=1.0 s` 동결)와
발진 가드는 **평가되지 않는다**. 이 구간에는 정책이 구동하지 않으므로 위험은 낮지만,
`HangGuard` 의 `t0` 는 stand 시각이므로 `HANG_MAX_S=120 s` 예산에서 4 초가 먼저 깎이고,
콘솔이 파싱은 하지만 어차피 표시·기록하지 않는 `hang_state`(§5 R5)는 이 구간 내내 공백이다.

### 2.7 STOP_ALL 뒤의 홀드-프라임은 **동결을 해제한다** (MEASURED, 프로브 A2)

```
frozen:           desired=10.00 commanded=10.00 running=False
3 ticks later:    desired=10.00 commanded=10.00 running=True   (동결 유지)
after hold-prime: desired=3.25  commanded=10.00                (동결 목표가 교체됨)
  tick 1: commanded=6.00 deg   (슬루 상한 4.0°/틱 = 200°/s)
  tick 2: commanded=3.25 deg
```
`STOP_ALL` 은 `desired := commanded` 로 얼리는데, 상대 0 배치가 오면 `desired` 가
**baseline(착지 자세)** 로 바뀌고 `commanded` 가 `MAX_TARGET_STEP_DEG_PER_TICK=4.0°/20 ms
= 200°/s` 로 끌려간다(위 3.25° 는 내 합성 baseline 에 걸린 절대리밋 클램프 값이다).
도달성: 브리지는 STOP_ALL 후 `core.active=False` 라 스스로 다시 프라임하지 않고,
콘솔은 2차 `stand` 를 막는다 → **콘솔만 쓰면 도달 불가**. 그러나 0910 처럼
`ros2 topic pub`/`fire_stand.sh` 로 stand 를 재발화하면 도달한다 (HYPOTHESIS: 재현 필요).

### 2.8 워밍업의 **전제 자체**가 코드와 어긋난다 (HYPOTHESIS — 다음 실기에서 확인 필요)

커밋 근거는 "stage8 는 명령 수신마다 status 를 발행 → 유휴 시 버스트+수초 갭"이다.
그러나 stage8 에는 **`STATUS_PERIOD_SEC=0.01` 의 100 Hz status 타이머**가 있고
(`stage8_…node.py:1339`), 명령 트리거 발행은 오히려 5 ms 로 **캡**돼 있다(1341–1348).
또 armed+baseline 이면 `control_tick` 이 **매 틱 12관절 전부에 CAN 프레임을 쓴다**
(640–655) — 브리지가 아무 것도 안 보내도 피드백은 50 Hz 로 돌아온다.
그러므로 "유휴라서 status 가 안 온다"로는 tick-1 stale 이 설명되지 않는다. 살아 있는 대안:

- **H1**: 한 관절만 `feedback_age_sec ≥ 0.5 s` 여도 브리지는 `joint_ts` 를 **아예 갱신하지
  않는다**(`fresh == 12` 조건, 1461–1466). 이 경우 원인은 CAN 한 분기이고 홀드-프라임은
  아무 것도 못 고친다 → 4 초 뒤 타임아웃이 "정상 동작"이다.
- **H2**: `STALE_DISARM_SEC=60 s` 자동 무장해제 후처럼 stage8 가 스트리밍을 멈춘 상태
  (`_ever_streamed` 때문에 폴링도 금지)라면 CAN 피드백이 영구 침묵 → 동일 증상.
- **H3**: 커밋 실측대로 정말 status 스트림이 끊겼다면, 원인은 발행 주기가 아니라
  DDS/실행기 포화다 → 워밍업은 증상만 가린다.

세 가설 모두 **홀드-프라임이 원인을 고치지 않는다**는 점에서 일치한다. 워밍업은
타이밍 레이스를 4 초 유예하는 완충재이지 원인 규명이 아니다. (CLAUDE.md 규칙 2)
판별 측정: stand 직전/직후 `status_age_s` 와 관절별 `feedback_age_sec` 를 동시에 남기고,
실패 시 어느 관절이 낡았는지 본다.

---

## 3. 스탠드 타이머와 "정책 구동" vs "무보조 자립"

**타이머 시작은 조작자 확인 후에만 맞다 (MEASURED).** `slack_at` 을 세팅하는 곳은
`operator_event("op_slack_done")` **한 곳**(console_state:239)뿐이고, 그 경로는
`dispatch_action` → `CONFIRM_ACTIONS` 의 `confirmed:true` 모달 → `reason()` 게이트를
전부 통과해야 한다. 자동 시작·센서 기반 시작·타임아웃 시작은 없다.
pending 이 남아 있으면 `op_slack_done` 자체가 차단된다(프로브 B-A). 재측정도 차단
("이미 측정한 시험입니다").

**코드상 구분은 있다 (MEASURED).**
`phase`: `정책 제어 · 지지 제거`(slack 전) ↔ `자립 관찰`(slack 후),
`measurement.state`: `waiting` → `measuring` → `interrupted`,
`support_confirmed` 불리언, `observed_5s/30s` + `review_required:true`.
UI 도 "이것은 자립 성공이 아닙니다"(stand 확인 문구)와
"관측 시간 도달 표시"를 분리해 쓴다.

**구분의 근거는 전부 자기보고다 (구조적 한계).** 줄 장력·외부 보조를 재는 채널이
없으므로 "무보조"는 `op_slack_done` 버튼 한 번이 전부다. `owner` 문자열은 두 상태에서
동일(`정책 → 12축 제어기`)이라, 화면에서 "정책이 돈다"와 "혼자 섰다"를 가르는 것은
phase 한 줄과 경과초뿐이다. 문서도 영상 대조를 요구하고 있고 이는 옳다.
다만 다음 두 가지는 **코드로 좁힐 수 있다**:
- `op_release_stop`(방출 정지) 없이 `op_slack_done` 을 누를 수 있다 → 윈치 조작 중
  슬랙 선언 가능.
- 타이머 0점은 **버튼이 서버에 닿은 시각**이지 줄이 실제로 느슨해진 시각이 아니다.
  (영상 동기화용으로 타임라인에 event 시각이 남는 것은 ✓)

---

## 4. 회귀 테스트 실행 결과 (MEASURED)

```
$ cd /home/ryu/humanoid_leg_test1 && python3 -m pytest -q tests/test_operator_console.py
...................................                                      [100%]
35 passed in 0.17s
```
Python 3.10.12 / pytest 6.2.5. 하드웨어·ROS·subprocess 불필요(`subprocess.run`/`Popen`
자체가 `AssertionError` 로 막혀 있음). 재실행 2회 동일.

**커버리지 공백 (이번 변경과 직결)**: `console_demo.Demo.act("stand")` 는
`running=True, policy_started=True` 를 **즉시** 세팅한다(console_demo:50-54).
즉 데모/회귀 테스트에는 **워밍업 창이 존재하지 않는다** — 디버그 공백 4 초,
홀드-프라임 50 Hz, 브리지발 STOP_ALL 중 어느 것도 테스트가 건드리지 않는다.
§2 의 결론들은 전부 내가 따로 주입한 프로브로만 확인한 것이다.

---

## 5. 위험 목록 (심각도 · 최소 수정안) — **파일은 수정하지 않았다**

| # | 위험 | 심각도 | 근거 | 최소 수정안 |
|---|---|---|---|---|
| **R1** | **estop 동결 확인이 거의 항상 실패한다.** `stop.confirmed` 는 `stage8.running==False` 를 요구하는데, stage8 는 armed 인 한 **다음 제어틱(20 ms)에 `running=True` 로 되돌린다**(프로브 A: `STOP_ALL → running=False`, `one tick later → running=True`). 콘솔 샘플링은 20 Hz(UI 폴링은 더 느림) → 비상정지 때 "실제 동결 미확인" 이 정상 상태처럼 붙박이가 된다(경보 피로). | **높음** | MEASURED (프로브 A) | `console_state.py:382-386` 에서 `not s.get("running")` 을 빼고 `last_command=="STOP_ALL" and accepted and command_seq>stop_request_seq` 로 확정하되 문구를 "STOP_ALL 수락 확인 · 제어기는 동결 목표를 계속 스트리밍(토크 유지)"로. (프로브 A 에서 이 세 필드는 정지 후에도 안정적으로 유지됨을 확인) 근본책은 stage8 가 `frozen`(마지막 STOP_ALL 이후 desired==commanded) 을 status 에 노출하는 것. |
| **R2** | **브리지발 STOP_ALL 무징후 26 초.** 브리지가 콘솔 소유가 아니면(=DUP_GUARD 가 강제하는 상황 포함) 워밍업 타임아웃·센서스테일·가드 정지가 ROS 로 전혀 안 보이고, pending 30 s 타임아웃까지 화면이 정상처럼 보인다. | **높음** | MEASURED (프로브 B-A3) | ① `TrialState.update` 에 "stand pending/완료 중 `last_command=="STOP_ALL"` 이고 `command_seq>pending.status_seq` 인데 콘솔이 정지를 요청하지 않았다 → `latch("제어기에 외부/브리지 STOP_ALL 수신")`" 3줄 추가. stdout 의존 없이 **모든** 브리지발 정지를 1 메시지 안에 잡는다. ② `stand` 의 pending 타임아웃 30 s → 6 s(=`STAND_WARMUP_MAX_S`+여유). |
| **R3** | **"12/12 신선"의 기준이 브리지보다 15배 느슨하다.** 콘솔 `feedback_ok` 는 stage8 의 `feedback_stale`(= `FEEDBACK_TIMEOUT_SEC=3.0 s`)을 쓰고, 브리지 라이브 게이트는 `SENSOR_STALE_SEC=0.2 s`(전 12관절, `worst_age` 로 뒤보정)다. 즉 **콘솔이 초록불인 채로 브리지는 tick-1 에 sensor_stale 로 죽을 수 있다** — 오늘 고친 그 실패를 stand 전에 못 거른 이유. | **중상** | MEASURED (상수·코드) | 뷰에 `worst_feedback_age = max(joints[].feedback_age_sec)` 를 넣고(이미 status 에 들어오는 값·세션 기록에도 남는 값) `reason("bridge")`/`reason("stand")` 에서 `worst_feedback_age < 0.2` 를 요구 + 화면에 숫자 표시. 콘솔 단독 수정으로 끝난다. |
| **R4** | **홀드-프라임 4 초가 정책 구동과 기록상 구분 불가.** 20 Hz 타임라인·관절 추종 표에 표식이 없어, 사후에 "정책이 첫 4 초에 뭘 했나"를 물으면 답이 갈린다. | 중 | MEASURED (프로브 A, 세션 sample 필드) | `Proc._pump` 에 이미 나오는 두 줄(`인계 워밍업 최대`, `인계 워밍업 완료`)을 파싱해 `proc.warmup={state,ts}` 로 올리고 ① phase 에 `인계 워밍업 (정책 미구동)` 추가 ② `SessionLog.sample` 에 `warmup` 필드 추가. 콘솔 단독. |
| **R5** | **`bridge.hang_state` 를 파싱만 하고 버린다.** 화면·기록 어디에도 없다(`SessionLog.sample` 미포함). 행잉 시험에서 "리프트 가드가 무장됐는가(CONTACT_ACQUIRED 인가)"를 조작자가 볼 수 없다. | 중 | MEASURED | UI 의 브리지 줄과 `sample()` 에 `hang_state` 한 필드 추가. |
| **R6** | **STOP_ALL 뒤 홀드-프라임이 동결을 풀고 baseline 으로 200°/s 슬루.** 콘솔 경로로는 도달 불가하나 콘솔 밖 재발화(0910 실제 우회)로 도달. | 중 (조건부) | MEASURED(기전, 프로브 A2) / HYPOTHESIS(도달) | 브리지 1줄: `_send_hold_prime` 이 상대 0 대신 **이미 보유 중인 `self.stage8_cmd_rel[j]`**(status 에서 받는 현재 스트리밍 명령)를 보내게 한다 → 진짜 무동작 프라임이 되고 동결 목표도 안 건드린다. |
| **R7** | **브리지 이벤트 → 콘솔 fault 경로에 생산자가 없다.** `Telemetry.receive` 의 `{"event":"stop_all"…}` 분기(op_console:459)는 이 브리지가 **debug 토픽에 이벤트를 안 내므로** 사문이다(블랙박스 파일에만 기록). 테스트가 이 분기를 통과시켜 "가드 정지가 ROS 로 보고된다"는 착시를 만든다. 실제 가드 정지의 유일한 ROS 신호는 "디버그 신선도 상실(≥1.5 s 지연)"이다. | 중 | MEASURED | 브리지 `_stop_all` 에서 `_bb_write` 와 같은 dict 를 `pub_dbg` 로도 1회 발행(콘솔 파서가 **이미** 그 형상을 지원한다). 2줄. |
| **R8** | 새 정지 사유가 원문 문자열로만 노출 — 운영 안내·등급 구분 없음. | 중하 | MEASURED (프로브 B-A2) | `fault` 문자열 → 운영 문구 매핑 소사전(`sensor_stale_warmup_timeout` = "인계 실패: 관절 피드백 미확보. CAN/stage8 확인 후 재명령"). |
| **R9** | 리뉴얼이 `trim/trim_preset/posture/zero_*/disarm/march` 를 UI·API 에서 제거(403)했는데, 같은 스크립트들이 `poll_processes` 의 **외부 제어 프로세스 패턴**이다. 즉 이제 터미널에서 트림/영점/자세시험을 돌리는 순간, stage8 단계 이후라면 콘솔이 `추적 밖 제어 송신 가능 프로세스 감지` 로 시험을 고장 래치한다. | 중하 | MEASURED (op_console ACTIONS/LABELS/patterns, 테스트가 403 을 고정) | 의도된 축소면 문서에 "시험 중 트림·영점·자세시험 불가, 하려면 회수 후"를 명시. 필요하면 `trim`/`trim_preset` 만 LABELS 에 되살리고 `CONFIRM_ACTIONS` 에 넣는다. |
| **R10** | `Runner._foreign_pids` 가 콘솔 **자기 자식**도 외부로 센다(제외는 `os.getpid()` 뿐). 같은 스크립트를 다른 key 로 띄우는 경우(예: `selftest`=`rl_bridge_node.py --selftest` 가 살아 있는 채 `bridge` 기동) 오탐으로 기동 거부. 죽이지는 않으므로 안전측. | 하 | MEASURED (op_console:334-343) | `owned = {p.p.pid for p in self.procs.values()}` 를 제외 집합에 추가. |
| **R11** | 브리지가 50 Hz 로 `SET_JOINT_TARGETS` 를 쏘는 동안에는 `last_command`/`command_seq` 근거의 baseline·arm 확인이 성립 불가(20 ms 안에 덮인다). 현재 순서 강제 때문에 도달하지 않지만, 정지 후 재무장 절차를 만들면 바로 터진다. | 하 | COMPUTED | 재무장 절차를 넣을 때는 브리지 프로세스 종료를 선행 조건으로. |
| **R12** | 데모/회귀 테스트에 워밍업 모델이 없어 이 창 전체가 미검증. | 중하 | MEASURED (console_demo:50-54) | `Demo` 에 `warmup_until` 을 넣어 stand 후 N 초는 `policy_started=False` + `running=True` + `last_command="SET_JOINT_TARGETS"` 로 두고, 회귀 3건 추가: ① 그 창에서 stand 미완료 ② `op_slack_done` 차단 ③ 창 중 STOP_ALL 수신 시 fault 래치(R2 수정과 세트). |

---

## 6. 결론

- 질문 (2)의 핵심 — **"워밍업 중 stand 가 완료로 찍히는가"는 아니오**(MEASURED, 대조군 포함).
  홀드-프라임이 "정책 모션"으로 **화면에서** 승격되는 경로도 없다.
- 대신 진짜 위험은 **침묵**이다: 워밍업은 ROS 로 아무 신호도 내지 않고(R2·R4),
  그 실패(4 s 타임아웃)는 브리지 stdout 을 콘솔이 잡고 있을 때만 보인다.
  브리지를 콘솔 밖에서 띄우는 현장 우회가 그 조건을 깬다.
- 그리고 오늘의 버그를 **stand 전에** 걸러줄 수 있었던 지표(관절 피드백 나이)는
  콘솔이 이미 받고 있으면서 15배 느슨한 기준으로 초록불을 켜고 있다(R3).
- 정지·동결·토크해제의 개념 분리는 코드·문구 모두 좋다. 다만 동결 **확인** 술어는
  실제 stage8 거동과 맞지 않아 사실상 작동하지 않는다(R1).
- 무보조 타이머는 조작자 확인 뒤에만 시작하며, 자동 성공 판정이 없다. ✓

### 다음 실기 전에 값싸게 얻을 것 (측정 제안)

1. stand 직전 10 초 + 직후 10 초의 **관절별 `feedback_age_sec`** 와 `status_age_s`
   동시 기록 → §2.8 의 H1/H2/H3 판별 (콘솔은 이미 `feedback_age_sec` 를 타임라인에 남긴다).
2. 같은 구간에서 stage8 status **수신 간격 히스토그램** → "버스트+수초 갭"의 실재 여부.
3. 워밍업 성공 사례에서 `인계 워밍업 완료` 까지 걸린 시간 → 4 s 창이 타당한지.

---
부록. 오프라인 프로브 3개는
`/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/`
의 `stop_probe.py`, `prime_after_stop.py`, `warmup_probe.py` (세션 임시 디렉터리, 저장소 밖).
CAN 전역 플래그를 꺼서 버스 접근 경로가 호출되지 않음을 코드로 확인한 뒤 실행했다.
