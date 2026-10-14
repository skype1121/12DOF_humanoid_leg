# head_1008 — 젯슨 저장소 사본을 HEAD 와 맞추는 배포 패키지 (2026-10-07 밤 준비)

> 상태: **패키지·스크립트 준비와 로컬(가짜 젯슨) 검증까지. 젯슨에는 접속하지 않았고 아무것도 올리지 않았다.**
> `--apply` 는 승인 사항이다. 이 패키지는 파일 복사·오프라인 selftest·콘솔 서비스 재기동만 한다 — CAN 송신·ARM·토크 인가·영점 변경·센서 기동·
> 제어 프로세스 기동/종료는 하지 않는다. 물리적 로봇 실행은 범위 밖이다.

**재생성 한 줄(메인 체크아웃, 병합 뒤 아무 커밋에서나):**

```
python3 deploy/make_head_manifest.py [--inventory deploy/cand/head_1008/jetson_inventory_before_md5.txt] [--extra-policy-run logs/rsl_rl/<실험>/<run>]
```

- "젯슨 현재 md5 목록 파일" = `--inventory`(기본값이 위 경로 — `md5sum` 출력 형식, 경로는 `./` 기준). 젯슨 상태가 바뀌면 이 파일만 새로 받아 넣고 다시 돌린다.
- 같은 규칙으로 `head_1008.md5`·`.plan`·`.VERSION`·`DIFF_LIST.md`·`ref_bridge/*.ref` 를 전부 다시 만든다. 이 문서의 수치는 메인 `aa53c615`(모터 오류 기록 a18f6c25·콘솔 걷기 패널 d7d7963b·러너 동치 기준 40fc5653 까지)를 병합한 상태 기준이다 — 그 뒤 커밋이 생기면 다시 만든다.
- **자동으로 들어가는 새 파일**: 콘솔·브리지·stage8·도구가 import 하는 모듈(사슬 추적), `jetson/scripts/*.py|*.sh|*.html`·`rl_walking/deploy/*.py`·`robot_runtime/*.py`·
  `protocols/*.py`, `tests/test_stage8_*.py`·`tests/test_console_*.py`·`tests/test_operator_console.py`·`tests/test_latency_check.py`·… 무늬에 맞는 **git 추적** 파일
  (커밋 전 파일은 안 잡힌다), `reports/night_1007/jetson_parity.py` + `reports/night_1007/jetson/parity_ref.npz`(둘 다 있을 때).
- **손으로 넣어야 하는 것**: 새 진입점(`ENTRY_*`), 위 무늬 밖 폴더, 새 정책 키(`POLICY_KEYS`·`--keys`·`--extra-policy-run`), 젯슨 게이트에서 실제로 돌릴 새 selftest
  (`deploy_head_1008.sh` 의 `remote_selftest` 목록 — 시험 파일은 실리기만 하고 자동으로 돌지는 않는다).

지시(1007 22:00): 젯슨 배포 파일·정책 정리, 배포 패키지 구성, 코드·설정 동기화, 실행 스크립트·콘솔 명령 정리, 버전·체크섬 기록, 롤백 가능한 이전 판 보존.

## 1. 무엇이 가고 무엇이 안 가는가

젯슨의 현재 상태는 주 작업자가 22:17 에 읽기 전용으로 뽑은 목록(`jetson_inventory_before_md5.txt`, 173 파일)을 기준으로 삼았다.
전체 표는 `DIFF_LIST.md`(생성물), 기계가 읽는 목록은 `deploy/head_1008.plan`, 배포 뒤 젯슨이 가져야 할 md5 는 `deploy/head_1008.md5`.

| 구분 | 수 | 내용 |
|---|---|---|
| 전송 — 변경 | 14 | 콘솔 4(`operator_console.py`·`console_state.py`·`console_ui.html`·`console_demo.py`), `rl_bridge_node.py`, `stage8_12axis_mit_control_node.py`, `rl_walking/__init__.py`, `rl_walking/deploy/c5_switch.py`·`policy_runner_stage4.py`, `robotctl.py`, 저장소 사본만 오래된 4(`sensors_up.sh`·`pose_hold.py`·`imu_mount_calib.py`·`analyze_cop_transfer.py`) |
| 전송 — 신규 | 50 (+2) | `rl_walking/deploy/` 12(`w3_deploy`·`kew3_runner`·`w3_contingency`·`hturn2`·`leg_odometry` + selftest 7), `jetson/scripts/` 19(`w3_cmd`·`latency_check`·`selftest_w3_bridge_node` + 분석·보조 도구), 시험 7(콘솔 걷기 시험 2 포함), 러너 동치 2(`reports/night_1007/jetson_parity.py`·`jetson/parity_ref.npz`), W3S3d 정책 4, selftest 기준 원문 3 + 실행기 1, 운영 명령 문서 1, (+ 매니페스트·VERSION) |
| 확인만 | 56 | 젯슨 md5 가 이미 HEAD 와 같은 파일(예: `hang_stand.py`·`boot_pose_check.py`·`zero_anchor.py`·`release_all.py`·`robot_runtime/*`·`protocols/*`·G21a·G24Ra·G12·G11a·v18b 정책) — 복사하지 않고, 배포 때 다시 읽어 다르면 중단 |
| 삭제 | 0 | |

**절대 보내지 않는 것**(생성기 `forbidden_reason` 과 배포 스크립트 `forbidden_path` 가 각각 강제 — 걸리면 중단):
`config/` 전체(하드웨어맵·`stand_zero_deg`·IMU 장착 보정·지연 옵션 파일), `jetson/calib_data/`, `~/rl_calib/*`(level_pose 등 — 스크립트는 이 폴더를 읽기만 한다),
`.json`·정책 외 `.yaml`·확장자 없는 파일 전부, 허용 폴더(`jetson/scripts`·`rl_walking`·`robot_runtime`·`protocols`·`tests`·`logs/rsl_rl`·`deploy`) 밖.

젯슨 전용이라 손대지 않는 것: `config/robot_12dof_hardware_map.json`(젯슨 `41b68a6f`, 저장소 이력에 없음 = 젯슨 정본), `~/rl_calib/` 의 실행본·보정
(`level_pose.json`·`last_pose_journal.json`·`can_channel` 은 **젯슨 정본·저장소 사본 없음**), 구 세대 정책 47 파일, 젯슨에 남은 옛 학습 코드 11 파일(실행하지 않음).

정책 파일(상세·md5 는 `DIFF_LIST.md` §7):

| 키 | 필요한 파일 | 젯슨 |
|---|---|---|
| `stand_crouch_g21a` | `…/2026-10-02_21-23-08_g21a/exported/policy.onnx`(+`.pt`), `params/env.yaml` | 이미 있음(md5 같음) |
| `stand_c5kr_g21a_g24ra`·`stand_c5krh_g21a_g24ra` | 위 + `…/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.onnx`(+`.pt`·`SOURCE_CHECKPOINT`), `params/env.yaml` | 이미 있음(md5 같음) |
| `stand_w3s3d_15800` | `…/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.onnx` 1321751 B `d908cd92…`, `policy.pt` 1332468 B `10e3ee40…`, `SOURCE_CHECKPOINT` 15 B `1ad3809e…`, `params/env.yaml` 60296 B `41d1e62a…` | **없음 — 새로 간다** |
| `stand_kew3_g21a_g24ra_w3s3d15800` | 자체 파일 없음 — W3S3d + G21a + G24Ra 의 파일을 쓴다 | 위 셋으로 충족 |
| `stand_kew3q_g21a_g24ra_w3s3d15800`(1008) | 자체 파일 없음 — 위 합성 키와 같은 세 파일 | 위 셋으로 충족 |

**1008 추가 — 둘째 합성 키 `stand_kew3q_g21a_g24ra_w3s3d15800` [실물 미검증]**: 첫 합성 키와 같은 구성(걷기 W3S3d-15800 · 서기 `stand_c5kr_g21a_g24ra` · 상태 이어받기 ·
정착 0.5 s)에 **KE 감지기의 갇힘 해소(`requiet` 2.0 s · 0.3 rad/s · 양발 접촉)만 켠 키**다. 시뮬 합성 평가(`eval_transition_kwx.py --kw_requiet 2.0,0.3`)가 돌린 구성이 이 키이고,
첫 합성 키는 그 옵션이 **꺼져 있다**(평가 구성 아님). 같은 실험 플래그(`--experimental-w3 --experimental-kew3`) 뒤에 있고 콘솔 드롭다운 맨 뒤에 붙는다. 기존 키(단독 KE 키·첫 합성 키 포함)의
감지기·기동 인자·발행은 그대로다(확인: 브리지 selftest 의 '갇힘 해소' 감지기 항목 · 러너 동치 5 키가 종전 기준과 차 0 · 기존 키 세션 발행이 변경 전 커밋과 비트 동일 — 커밋 메시지). 이 문서의 다른 표·수치(파일 수·selftest 20/20 등)는 이 키를 더하기 전의
기록이다 — 매니페스트를 다시 만든 뒤 갱신한다. 러너 동치 기준(`parity_ref.npz`)은 5 키 그대로면 새 키를 대조하지 않는다(6 키로 `--gen` 해야 본다).

브리지가 읽는 것은 `policy.onnx`(젯슨은 torch 가 없어 onnxruntime) 와 `params/env.yaml` 이다. `SOURCE_CHECKPOINT` 는 출처 표시, `params/agent.yaml`·`model_*.pt` 는 읽지 않아 싣지 않는다.

## 2. 기존 세션(서기 G21a·회복 krh)이 HEAD 판으로 바뀌면 달라지는 것

| 항목 | 종전(젯슨 판) | HEAD | 기본값에서 종전과 같은가 · 근거 |
|---|---|---|---|
| 브리지 기동 인자(콘솔이 넘기는 것) | `--live --checkpoint <키>` | 기존 7 키는 같음. 걷기 실험 키 2 개에만 `--experimental-w3 [--experimental-kew3] --w3-heading-loop off` | 기존 키는 같음 — `bridge_extra_args()` 가 걷기 키가 아니면 빈 목록, 콘솔 selftest ②("기존 7 키 live·selftest 종전과 같음")가 고정. 준비 단계의 계약 검사도 기존 키는 `--selftest --checkpoint <키>` 그대로 |
| 브리지 새 인자 | 없음 | `--experimental-hturn2`·`--experimental-w3`·`--experimental-kew3`·`--w3-safety`·`--w3-heading-loop`·`--w3-cmd-limits`·`--w3-monitor`·`--w3-hl-fault`·`--c5-settle`·`--c5-overtime`·`--c5-imu-f32`·`--runner-ema-fresh` | 전부 기본 꺼짐(None·False·1). 콘솔은 넘기지 않는다 |
| 브리지가 내는 명령·기록(기존 키) | — | 블랙박스 틱 줄에 기록 필드 6 종 추가(아래 '모터 오류 기록') | **발행은 같음 — 젯슨 판 ↔ HEAD 를 같은 입력열로 돌린 발행 24,590 건이 비트 동일, 블랙박스 11,304 줄은 새 필드 6 종을 뺀 나머지 키·값·순서가 같음**(`replay_compare.py`: 서기 g21a·g12·g12_ema0·g11a·v18b, 공중 시작(hang_arm), 회복 c5·kr·krh × live/dry × 가드 data/safe 23 경우 + 오류 대본 4 경우, 실제 정책 파일). 저장소 selftest 의 기준 커밋(e8489de4·b58a39cb·443ef820)은 젯슨 판보다 뒤라 이 구간을 덮지 못해 따로 쟀다 |
| 모터 오류 기록(a18f6c25·f5633095·9c191007 — 1007 밤 병합) | 없음 | 브리지 블랙박스 틱 줄에 `fb_age_max_s`·`fb_age_max_joint`(항상), `err_raw`·`s8_warn`·`fb_age_over`·`errlog_fail`(있을 때만). 콘솔 타임라인 줄의 `joints[관절].feedback_error_raw`(+`feedback_error_seen`)·`stage8_warnings`·`stage8_errlog_fail`·`stage8_nomsg` | 기록 추가만 — **오류 대본**(stage8 상태에 오류 바이트·warnings·피드백 나이 0.06/0.12 s·정지 기준 초과 0.35 s 를 실은 입력열, g21a·krh × 가드 data/safe)에서도 발행이 젯슨 판과 비트 동일(새 필드가 실제로 찍힌 줄: `err_raw` 16·`s8_warn` 16·`fb_age_over` 40). 콘솔 타임라인 쪽은 이 비교의 범위 밖 — 콘솔 selftest·`tests/test_operator_console.py` 43 OK 로만 확인 |
| 브리지 새 명령 | — | `step`·`turn`·`rearm`(실험 세션 전용), `stand` 의 `touchdown_start` | 기존 세션에서 `step`·`turn`·`rearm` 은 무시(위 비교의 입력열에 포함). `touchdown_start` 는 콘솔 '정책 ON' 만 보낸다 |
| 브리지 import | `policy_runner*`·`c5_switch`·`deploy_chain` | + `hturn2`·`w3_deploy`(항상), `w3_contingency`·`kew3_runner`(켤 때만) | 새 파일이 없으면 브리지가 뜨지 않는다 → 한 묶음으로 보내고 md5·selftest 로 확인 |
| 콘솔 드롭다운 | g12(기본)·g21a·c5·c5kr·c5krh·g12_ema0·g11a | 같은 7 개 + **뒤에 걷기 실험 키 2 개**(`stand_w3s3d_15800`·`stand_kew3_g21a_g24ra_w3s3d15800`, 표시 문구 포함) | 기본값·기존 키 순서 그대로(`CKPT_CHOICES` 변화 없음, `CKPT_ALL = CKPT_CHOICES + WALK_CKPTS`). 걷기 키를 고르지 않으면 종전과 같다 |
| 콘솔 단계별 절차·`stand` 발행 | `{"cmd":"stand"}`(+hang_arm) | 같음 | 같음 — 콘솔 selftest 의 `bridge_cmd_payload("stand", {"hang_arm": False, "touchdown_start": False}) == {"cmd": "stand"}`, `tests/test_operator_console.py` 는 기준 판의 시험이 삭제·수정 0 줄로 남아 있고 HEAD 에서 통과(43) |
| ⚠ 콘솔 새 버튼 **'정책 ON(착지 자동 인수)'** + 확인 창 | 없음 | 공중 수평 자세에서 한 번 → 접지 마커·인계·기준점·ARM·정책 로드·`stand(touchdown_start)` 자동 진행, 정책은 양발 착지 확인 뒤 시작 | **새 기능 · 실물에서 한 번도 돌지 않았다**(1005 22:54 커밋, 젯슨에는 그 전 판이 있었다). 누르지 않으면 종전과 같다. 선행조건은 `OPERATOR_COMMANDS.md` §2 |
| ⚠ 콘솔 **걷기 조작 패널**(d7d7963b, 1007 23:53 병합) | 없음 | 걷기 실험 키로 시작한 시험에서 시간 제한 걷기 버튼 6 개(+ 비활성 2 개 표시·서버 거부)·걷기 끄기·정지 래치·재무장. 서버 상태기·선행조건·기록, 새 API 동작 `w3_move`·`w3_off`·`w3_stop`·`w3_rearm` | **새 기능 · 실물에서 한 번도 돌지 않았다.** 걷기 키가 아닌 시험에서는 버튼이 전부 거부된다(`WalkState.reason`: "선택한 정책이 걷기 실험 키가 아닙니다"). 콘솔은 걷기 브리지에 `--w3-monitor`·`--w3-safety`·`--w3-hl-fault` 를 넘기지 않는다(감시 꺼짐). 확인한 것: 젯슨 모사 환경에서 콘솔 `--selftest` ALL PASS · pytest 70 passed(`test_operator_console` 43 + 걷기 시험 2 파일) — 전부 오프라인, ROS 실통신·실물 미검증 |
| 콘솔 상태 표시 | — | 진행 중 "정책 ON 진행 중 · …", 착지 대기 "착지 대기 · 수평 자세 홀드(정책 미구동)"; 자동 진행 중 다른 조작 차단(정지 요청·이벤트·줄 지지 확인은 가능) | '정책 ON' 을 쓸 때만 |
| ⚠ 콘솔 준비 단계 '정책 계약 검사' | 브리지 selftest 20 항목 | **31 항목, 로컬 측정으로 약 3.3 배 오래 걸린다**(1.4 → 4.7 s). 콘솔의 이 단계 제한은 300 s | 젯슨에서의 실제 시간은 미측정. 배포 스크립트가 젯슨에서 재서 240 s 를 넘으면 실패 처리·자동 롤백. 미리 보려면 `--probe-selftest-time` |
| stage8 기동 인자 | `--can-channel can1 --wait-takeover` | 같음 + 옵션 파일이 있으면 지연 옵션 | 같음 — 젯슨에 `config/stage8_latency_opts` 가 없거나 비어 있으면 추가 인자 0(`stage8_latency_raw()` 가 파일 없음 → `""`, `stage8_latency_args("")` → `[]`). **이 패키지는 그 파일을 만들지도 바꾸지도 않는다**(`config/` 전체가 전송 금지). dry-run 이 파일·서비스 환경변수를 읽어 켜져 있으면 ★ 로 알리고, 젯슨 selftest 가 콘솔의 판정 함수를 불러 인자 0 개인지 확인한다 |
| stage8 가 버스로 내는 프레임 | — | — | **같음 — 젯슨 판 ↔ HEAD CAN 프레임 39,056 개 비트 동일**(`stage8_frame_compare.py`: 인수 대기 → 인수 홀드 → 기준점 → ARM → 목표(슬루 초과·한계 밖·모르는 관절) → STOP_ALL → 피드백 두절 62 s(램프 동결·60 s 자동 무장 해제) → DISARM, 가짜 can·가짜 ROS) |
| stage8 status | — | 키 4 개 추가(`latency_options`·`send_counts`·`fb_after_send`·`tick_timing`), 공통 키는 7,393 건 전부 같음 | 추가만. status 1 건 직렬화 비용 +22 %(94 → 115 µs, 개발 PC 상대값 — 젯슨 절대값 미측정) |
| stage8 새 인자 | 없음 | `--feedback-after-send`·`--feedback-wait-ms`·`--send-on-command`·`--control-hz` | 전부 기본 꺼짐. `tests/test_stage8_latency_options.py` 의 `test_defaults_are_legacy`·`test_default_slew_is_4deg_per_tick_regardless_of_timing`·`test_default_does_not_wait_for_replies` |
| 러너·c5_switch | — | 새 선택 인자(hturn2·w3·ema_fresh_ticks·settle·imu_f32·requiet) | 기본 None/1/False — 위 브리지 발행 비교가 두 모듈의 차이까지 포함한다(트리마다 자기 모듈 사용) |
| 세션 타임라인 | — | 브리지 보기 항목에 `start_gate` 한 칸 추가(평소 null), 모터 오류 기록 필드(위) | 기록 추가만 — 오류 필드는 평소 줄에는 생기지 않는다. `stage8_nomsg` 1↔0 사이 줄의 '필드 없음'은 '모름'이다 |
| `robotctl` | — | `policy-on` 명령, 상태 줄에 착지 대기·정책 ON 진행 표시 | 추가만 |
| `rl_walking/__init__.py` | 0929 판 | 학습 태스크 등록 추가 | 젯슨에는 gymnasium 이 없어 등록은 무동작(파일 머리의 대체 객체) |
| `~/rl_calib` 실행본 | — | 바꾸지 않음 | 저장소 사본 `jetson/scripts/sensors_up.sh` 만 HEAD(1001 실시간 우선순위·1006 패드 오프셋)로 간다. 콘솔이 실제로 띄우는 것은 `~/rl_calib/sensors_up.sh` 다 — 그 내용은 dry-run 이 읽어 보여 준다 |

위험해 보이는 변화(따로 표시): ① '정책 ON' — 실물 미검증 새 경로, ①-b 콘솔 걷기 패널 — 실물 미검증 새 경로(걷기 키 시험에서만 동작, 걷기 감시 인자 미전달), ② 콘솔 준비 단계의 selftest 시간 증가(제한 300 s), ③ stage8 status 발행 비용 소폭 증가
(지연 재측정 `latency_check --expect off` 권고), ④ selftest 가 도는 수 분 동안 콘솔이 '추적 밖 제어기'를 표시.

## 3. 검증 — 무엇을 어디까지 했나

**오프라인 검증 완료**(이 PC, CPU 만. ROS·CAN·GPU·젯슨 미사용):

| 검사 | 결과 |
|---|---|
| `deploy/test_head_1008_local.sh` — 가짜 젯슨(인벤토리대로 재구성한 152 파일 + 자리표시 `~/rl_calib`) 왕복 | 전체 모드 37/37 · 빠른 모드 27/27 PASS(병합 직후·수정 전에는 33/34 — 실패 1 건이 블랙박스 새 필드, §2): dry-run 무변경 · 금지 경로 14 종 중단 · 제어 프로세스·활성 세션·예상 밖 md5·디스크 부족 거부 · selftest 실패 시 자동 롤백 · 적용(백업 14·신규 52·md5 122/122) · 재적용·`--verify`·재기동 게이트(제어 프로세스·selftest 통과 기록) · **롤백(복원 14·삭제 52) 뒤 파일 md5 152 개·폴더 목록이 처음과 동일** · 부분 적용 복구 · 정책 폴더 추가/제거 · 로컬 트리가 계획서와 달라져도 롤백은 진행(적용은 거부) |
| 같은 시험, 젯슨 모사 파이썬(python 3.10.12 · numpy 1.21.5 · onnxruntime 1.15.1 · torch 없음 · git 없는 트리) | 젯슨 selftest 20 항목 전부 PASS(러너 동치 포함), 추론 백엔드 onnx 기록 |
| parity 묶음까지 넣은 왕복 — 메인의 `reports/night_1007/jetson/parity_ref.npz`(`50b33308…`, 626,860 B)를 넣고 재생성 | 젯슨 게이트 20/20 — `jetson parity: ALL PASS`(5 키, 기준 torch ↔ 젯슨 모사 onnxruntime, 목표각 최대 차 ≤ 7.2e-07 rad · 원시 행동 ≤ 4.1e-06 · 모드 불일치 0 틱), 롤백 → 처음과 동일. 40fc5653 로 기준 npz 가 저장소에 들어온 뒤로는 커밋된 매니페스트에도 parity 2 파일이 들어 있다(같은 `50b33308…`) |
| `check_numpy_compat.py --plan` — numpy 1.21.5 에 없는 이름·인자(정적) | 89 파일 위반 0(심어 둔 위반 6 종은 검출 — 병합 전 판에서) |
| `replay_compare.py` — 브리지 젯슨 판 ↔ HEAD | 27/27 같음(§2 — 발행 비트 동일, 블랙박스는 새 기록 필드 6 종 밖 동일). 대조군: 같은 판 두 번 = 같음 27, 변이 2 종(krh 유지 0.7 → 0.6 s, LPF β 0.27 → 0.2701 — 병합 전 판에서) = 검출, 새 필드 옵션의 대조군 3 종(새 필드 밖 블랙박스 값 1 개 변이 = 검출 · 새 필드 이름의 키를 발행에 끼움 = 검출 · 기준 판에 그 키가 있으면 비교 거부) |
| `stage8_frame_compare.py` — stage8 젯슨 판 ↔ HEAD | 프레임·공통 status 전부 같음(§2). 변이(슬루 4.0 → 4.01°/틱) = 검출 |
| 콘솔 단위 시험(`tests/test_operator_console.py` + 걷기 시험 2 파일, pytest), stage8 단위 시험(가짜 버스) | 70 passed · 28 OK(젯슨 모사 환경). 걷기 시험 2 파일은 pytest 형식이라 `python3 -m unittest` 로는 0 건이 돈다 — 젯슨 게이트에는 넣지 않았다(젯슨의 pytest 유무 미확인) |

**미검증**: 실제 젯슨에서의 어떤 것도(ssh·sudo·systemd 경로, aarch64 onnxruntime 수치, 젯슨의 selftest 소요 시간, 젯슨의 실제 하드웨어맵으로 도는 selftest).
위 비교는 전부 개루프 계약 비교다 — 폐루프 성능·실물 안전을 말하지 않는다.

**현장(젯슨) 확인 필요** — dry-run 에서 볼 것:
1. `예상 밖 0` 인가(젯슨이 22:17 인벤토리와 같은가). 아니면 인벤토리를 다시 뽑아 매니페스트를 재생성한다.
2. 파이썬 환경 줄에 `can`(python-can)이 있는가 — 없으면 stage8 단위 시험 2 항목이 건너뛰어져 게이트가 실패한다.
3. `~/rl_calib` 실행본 대조 표에서 `★다름` 인 것(특히 `sensors_up.sh`·`stage8_cmd.py`·`boot_pose_check.py`·`hang_stand.py`). 이 패키지는 고치지 않는다 — 어느 판이 도는지 알고 가기 위한 표다.
4. `--probe-selftest-time` 의 현재 selftest 시간 × 3.3 이 240 s 아래인가.
5. 디스크 여유(지난 배포는 타임라인 4 개가 디스크를 채워 백업 단계에서 멈췄다), 제어 프로세스 없음, 콘솔 유휴, `KillMode=process`.

## 4. 쓰는 법

```
python3 deploy/make_head_manifest.py              # (실행 파일이 바뀌었으면) 매니페스트·계획서·차이 목록·기준 원문 재생성 — 한 명령
deploy/test_head_1008_local.sh [--jetenv <폴더>]  # 가짜 젯슨 왕복 시험. --make-jetenv <폴더> 로 젯슨 모사 환경을 만들 수 있다
deploy/deploy_head_1008.sh                        # dry-run (젯슨에 쓰지 않는다). --probe-selftest-time 을 붙이면 현재 selftest 시간도 잰다
deploy/deploy_head_1008.sh --apply                # 승인 뒤. 백업 → 전송 → 교체 → md5 → 젯슨 selftest, 실패하면 자동 롤백
deploy/deploy_head_1008.sh --restart-console      # 콘솔 유휴·제어 프로세스 없음·selftest 통과 기록이 있을 때만
deploy/deploy_head_1008.sh --verify               # 복사 없이 상태 확인 + selftest
deploy/deploy_head_1008.sh --rollback [--restart-console]
deploy/deploy_head_1008.sh --add-policy=logs/rsl_rl/<실험>/<run>     # 정책 폴더 하나만 추가(밤사이 새 후보) — 기존 파일은 덮어쓰지 않음
deploy/deploy_head_1008.sh --remove-policy=logs/rsl_rl/<실험>/<run>  # 그 추가의 롤백
```

- **콘솔 재기동은 자동으로 하지 않는다.** `--restart-console` 을 명시할 때만 하고, 그때도 제어 프로세스 0·콘솔 유휴·`KillMode=process`·센서 PID 가 PID 파일 소유·
  이 매니페스트의 selftest 통과 기록이 있을 때만 한다. system 유닛이라 **sudo 가 필요하다**(`echo "$JETSON_PW" | sudo -S systemctl restart operator-console` — 기존 스크립트와 같은 방식).
  주 작업자가 손으로 재기동해도 된다 — 스크립트는 그 경우 아무것도 하지 않는다.
- **롤백 한 줄**: `deploy/deploy_head_1008.sh --rollback`(파일만) 또는 `--rollback --restart-console`. 젯슨에서 손으로 할 때는
  `cd ~/humanoid_repo_bak_1008 && while read -r m f; do cp -p "files/$f" ~/humanoid_repo/"$f"; done < BACKUP_MANIFEST.md5 && while read -r f; do rm -f ~/humanoid_repo/"$f"; done < CREATED.txt`.
  새로 생긴 파일은 `CREATED.txt`, 새 폴더는 `CREATED_DIRS.txt` 에 적혀 있고 롤백이 지운다(가짜 젯슨 시험: 복원 14·삭제 52·남은 것 0, 스냅샷 동일).
- 젯슨 selftest 목록(전부 오프라인): 문법 · 러너 · c5_switch · kew3_runner · w3_cmd · latency_check · 콘솔 · 브리지 7 종(기본·g21a·krh·`--experimental-w3`·W3S3d·합성 키·1008 둘째 합성 키) ·
  W3 노드(실행기) · stage8 기본 인자·단위 시험 2 · 지연 옵션 꺼짐 · 매니페스트 `md5sum -c` · (묶음에 있으면) `reports/night_1007/jetson_parity.py --check reports/night_1007/jetson/parity_ref.npz`.
  하나라도 실패하면 자동 롤백한다. 러너 동치(parity)도 같은 게이트다 — 기준 npz 가 잘못 만들어졌다면 롤백 사유가 되므로 메인에서 `--gen` 한 직후의 파일을 실을 것.

- 메인 체크아웃(정책 `logs/` 가 있는 곳)에서 돌린다. 워크트리에서는 `HEAD1008_POLICY_ROOT=<메인>` 과 `--skip-local-gate`(시험 전용)가 필요하다.
- 로컬 원본을 쓰는 것은 dry-run 과 `--apply` 뿐이다. **`--rollback`·`--verify`·`--restart-console`·정책 폴더 모드는 로컬 트리가 계획서와 달라졌어도 진행한다**(젯슨 쪽 백업·계획서 md5 만 쓴다, 로컬 selftest 게이트도 생략) — 적용 뒤 다른 작업이 메인을 고쳤다고 롤백이 막히지 않게.
- 백업은 젯슨 `~/humanoid_repo_bak_1008/`: `files/<경로 그대로>`, `BACKUP_MANIFEST.md5`(원본 md5), `CREATED.txt`·`CREATED_DIRS.txt`(롤백 때 지울 새 파일·폴더),
  `APPLIED.plan`, `STATE`(이력), `selftest_*.log`. 원본 백업은 한 번 만들어지면 덮어쓰지 않는다 — 여러 번 적용해도 롤백은 **첫 적용 전 상태**로 간다.
- 젯슨 파일이 인벤토리와 다르면 멈춘다. 사람이 확인하고 덮어써도 된다고 판단한 구판은 `deploy/head_1008.accept` 에 `md5  경로` 로 적는다(없으면 쓰지 않는 파일).
- 롤백 뒤에도 콘솔 서비스는 재기동 전까지 메모리의 코드로 돈다 — 되돌릴 때는 `--restart-console` 을 함께.
- 적용에서 selftest 까지 수 분 동안 콘솔을 조작하지 않는다(파일이 섞여 있는 순간이 있고, selftest 프로세스를 콘솔이 '추적 밖 제어기'로 본다).

**다른 작업을 병합한 뒤**(콘솔 걷기 버튼, 브리지 블랙박스·오류 기록 등): ① `python3 deploy/make_head_manifest.py` ② `deploy/test_head_1008_local.sh`
③ T9b 가 '다름'이면 무엇이 달라졌는지 읽는다 — 블랙박스 줄에 의도해서 더한 기록 필드면 소스에서 그 키를 확인하고 시험 스크립트의 `NEW_BB_KEYS`(또는 `HEAD1008_REPLAY_NEW_BB_KEYS=키1,키2`)에 넣어 다시 돌린 뒤 §2 표에 적는다(그 키는 새 판의 블랙박스 줄에서만 빠지고 발행은 전부 비교된다 — 지금 6 개가 들어 있다).
의도하지 않은 차이면 배포하지 않는다. ④ 젯슨 상태가 바뀌었으면 인벤토리(`jetson_inventory_before_md5.txt`)를 새로 받아 ①부터.

## 5. 이 작업에서 찾은 것

1. **`selftest_w3_bridge_node.py` 는 젯슨에서 그대로는 통과할 수 없다.** 비트 비교 3 건이 `git show <기준 커밋>` 으로 브리지 원문을 읽는데 젯슨 사본은 git 저장소가 아니다
   (git 없는 트리에서 재현: 3 건 FAIL, 나머지 PASS). 원본은 고치지 않고 실행기 `selftest_w3_node_refs.py` 가 `_load_ref` 만 파일 읽기로 바꿔 끼운다. 기준 원문 3 개는 생성기가 git 에서 뽑아 md5 와 함께 싣는다.
2. **콘솔 준비 단계의 selftest 가 약 3.3 배 길어진다**(§2). 젯슨 시간은 재 보지 않았다.
3. 젯슨 저장소 사본은 "e7710c63 + krh 패치"와 정확히 같지 않았다: `rl_walking/__init__.py`(0929 판), `sensors_up.sh`(0913 판), `pose_hold.py`·`imu_mount_calib.py`(0727 판),
   `analyze_cop_transfer.py`(0911 판)가 더 오래됐고, `stage8_cmd.py`·`ping_all.py`·`stack_down.sh`·`can_channel.py` 등 19 개는 저장소 사본 자체가 없다(실행본은 `~/rl_calib`).
   전부 git 이력에서 출처를 확인했다(젯슨 로컬 수정본 0).
4. 젯슨의 `…_g10/exported/policy.*`(`stand_crouch_v6`)는 0922 배포본(`deploy/cand/g10_6600`)이고 로컬 `logs/` 의 같은 경로는 그 뒤 다시 내보낸 다른 파일이다. 이번 범위 밖이라 그대로 뒀다 — v6 를 다시 쓸 일이 있으면 어느 쪽이 맞는지 먼저 정해야 한다.
5. `deploy/deploy_lat_1006.sh --apply` 는 이제 쓸 수 없다(그 매니페스트의 콘솔이 HEAD 보다 옛 판이라 로컬 대조에서 멈춘다). 파일은 이 패키지가 올리므로 지연 단계는 `--stage off|fas|soc` 만 쓰면 된다.

## 6. 이 폴더

| 파일 | 역할 |
|---|---|
| `README.md` | 이 문서 |
| `OPERATOR_COMMANDS.md` | 세션 종류별 기동·걷기 명령·정지/회수·selftest 정리(초안, 젯슨에도 간다) |
| `DIFF_LIST.md` | 차이 목록·import 사슬·정책 파일 표(생성물) |
| `jetson_inventory_before_md5.txt`·`…_sizes.txt` | 젯슨 현재 상태(주 작업자 제공, `reports/night_1007/jetson/` 의 사본) |
| `selftest_w3_node_refs.py` · `ref_bridge/*.ref` | 젯슨용 W3 노드 selftest 실행기와 기준 브리지 원문(생성물) |
| `make_fake_jetson.py` | 인벤토리대로 가짜 젯슨 홈 만들기(로컬 시험용) |
| `replay_compare.py` · `stage8_frame_compare.py` · `replay_mutate.py` | 젯슨 판 ↔ 새 판 동작 비교(브리지 발행·stage8 프레임)과 그 변이 대조군 |
| `check_numpy_compat.py` · `numpy_1.21.5_api.json` | numpy 1.21.5 정적 호환 점검과 그 기준 |
| (상위) `deploy/make_head_manifest.py` · `deploy_head_1008.sh` · `test_head_1008_local.sh` · `head_1008.md5`·`.plan`·`.VERSION` | 생성기 · 배포 스크립트 · 왕복 시험 · 매니페스트 |
