# 젯슨 배포 기록 — head_1008 (2026-10-08 00:15–00:25, claude2, 12 시간 자율 작업 지시 범위)

**한 일: 파일 복사·오프라인 selftest·콘솔 서비스 재기동. 하지 않은 일: CAN 송신·ARM·토크 인가·영점 변경·센서 기동·제어 프로세스 기동/종료·물리 실행.**
하드웨어맵·영점·보정 파일(`config/`·`jetson/calib_data/`·`~/rl_calib/`)은 보내지 않았다(스크립트가 경로로 강제). 배포 전후 md5 로 확인했다.

## 버전
| 항목 | 값 |
|---|---|
| 묶음 | `head_1008` (`deploy/deploy_head_1008.sh`·`deploy/make_head_manifest.py`, 에이전트 작성·주 작업자 검토) |
| 원본 커밋 | `42da0c21`(실행 코드를 마지막으로 바꾼 커밋 — 콘솔 걷기 패널·모터 오류 기록 병합) |
| 매니페스트 | `deploy/head_1008.md5` md5 `8c210340a7a64e13da28c663c980809f` — 122 파일(전송 66·확인만 56) |
| 젯슨 위치 | `/home/mama/humanoid_repo` · 버전 파일 `deploy/head_1008.VERSION` |
| 백업(되돌리기용) | `/home/mama/humanoid_repo_bak_1008/` — 덮어쓴 원본 14 개(md5 검증)·새로 만든 파일 목록 52 개 |
| 되돌리기 | 개발 PC 에서 `deploy/deploy_head_1008.sh --rollback --restart-console` (가짜 젯슨 왕복 시험에서 "처음 스냅샷과 동일" 확인) |
| 배포 전후 파일 목록 | `reports/night_1007/jetson/inventory_before_md5.txt`(173) · `inventory_after_md5.txt`(218): 바뀜 14·새로 45·사라짐 0 |

## 무엇이 바뀌었나
- **바뀐 14 파일**: 콘솔 4(`operator_console.py`·`console_state.py`·`console_ui.html`·`console_demo.py`), `rl_bridge_node.py`, `stage8_12axis_mit_control_node.py`,
  `rl_walking/__init__.py`, `rl_walking/deploy/c5_switch.py`·`policy_runner_stage4.py`, `robotctl.py`, 저장소 사본만 오래됐던 4(`analyze_cop_transfer.py`·`imu_mount_calib.py`·`pose_hold.py`·`sensors_up.sh`).
- **새 파일 52**: 걷기 배포 모듈(`w3_deploy`·`kew3_runner`·`w3_contingency`·`hturn2`·`leg_odometry`), 도구(`w3_cmd`·`latency_check`·`selftest_w3_bridge_node`), 시험 7, 분석 도구,
  **W3S3d-15800 정책 4 파일**(`policy.onnx` `d908cd92…`·`policy.pt` `10e3ee40…`·`SOURCE_CHECKPOINT`·`params/env.yaml`), 러너 동치 도구·기준, 운영 문서.
- **안 바뀐 것(md5 확인)**: 하드웨어맵 `41b68a6f…`, `config/` 나머지, `jetson/calib_data/`, `~/rl_calib/*`, G21a·G24Ra·G12·G11a·v18b 정책, `hang_stand.py`·`boot_pose_check.py`·`zero_anchor.py`·`release_all.py`.

## 기본 동작이 종전과 같은가 (오프라인 근거)
| 항목 | 판정 | 근거 |
|---|---|---|
| 기존 키(서기 G21a·회복 kr/krh)의 브리지 발행 | 같음 | 젯슨 구판 ↔ 새 판을 같은 입력열로 돌린 27 경우·발행 24,590 건 비트 동일(`replay_compare.py`) |
| stage8 CAN 프레임 | 같음 | 구판 ↔ 새 판 39,056 프레임 비트 동일(`stage8_frame_compare.py`). 지연 옵션 3 종은 기본 꺼짐, 젯슨에 옵션 파일 없음(게이트 확인) |
| 브리지 블랙박스·콘솔 타임라인 | 기록 필드만 추가 | 모터 오류 바이트·stage8 warnings·피드백 나이(기록 전용 — 발행·가드 판정 불변) |
| 콘솔 드롭다운 | 기존 7 키·기본값 그대로 + 걷기 실험 키 2 개 뒤에 추가 | 콘솔 selftest ② |
| **새 경로(명시해야 켜짐, 실물 미검증)** | — | 콘솔 '정책 ON(착지 자동 인수)' · 콘솔 걷기 패널 · 브리지 `--experimental-w3/-kew3` · `--c5-*` 옵션 |

## 젯슨에서 돌린 검증 (전부 오프라인, 29 항목 통과·실패 0 — 로그 `~/humanoid_repo_bak_1008/selftest_20261008_*.log`)
- 문법(전송한 .py 89 개) · 러너 · c5_switch · kew3_runner · w3_cmd · latency_check · 콘솔 selftest
- 브리지 selftest × 6(기본·서기 G21a·회복 krh·`--experimental-w3`·W3S3d·합성 키) — 각 24–26 s(콘솔 준비 단계 제한 300 s 안)
- W3 노드 selftest(기준 커밋 원문 비교) · stage8 기본 인자 = 종전 · stage8 단위 시험 2 종 · 콘솔이 stage8 에 붙이는 지연 인자 0 개
- **러너 동치(`jetson_parity.py`)**: 로컬(torch, 시뮬 폐루프 평가에 쓴 코드) 기준 ↔ 젯슨(onnxruntime 1.15.1, aarch64) 러너, 같은 입력열 600 틱 × 5 키
  | 키 | 목표각 최대 차 [rad] | 원시 행동 최대 차 | 모드 불일치 틱 |
  |---|---|---|---|
  | `stand_crouch_g21a` | 5.4e-07 | 4.1e-06 | 0 |
  | `stand_c5kr_g21a_g24ra` | 4.2e-07 | 2.9e-06 | 0 |
  | `stand_c5krh_g21a_g24ra` | 3.6e-07 | 2.4e-06 | 0 |
  | `stand_w3s3d_15800` | 7.2e-07 | 2.9e-06 | 0 |
  | `stand_kew3_g21a_g24ra_w3s3d15800` | 7.2e-07 | 2.9e-06 | 0 |
- 매니페스트 대조 `md5sum -c` 122/122.

## 콘솔 서비스
- 00:19:51 재기동(`sudo systemctl restart operator-console`, 스크립트가 제어 프로세스 0·콘솔 유휴·KillMode=process 를 확인한 뒤). 새 PID 25363, active.
- 재기동 뒤 읽기 전용 확인: 걷기 패널 상태(`/api/state.walk`)가 나오고 버튼 8 개 중 6 개 허용·2 개 비활성(오른쪽 옆걸음·후진 한 걸음), 드롭다운에 걷기 키 2 개.

## 현장 확인 필요 (오프라인으로는 못 본 것)
1. 콘솔 준비 단계 '정책 계약 검사'가 실제 세션에서 제한 안에 끝나는지(오프라인 측정 24–26 s).
2. 걷기 패널·'정책 ON' 은 실물에서 한 번도 돌지 않았다. ROS 응답 토픽의 도착 순서·지연도 미확인.
3. `~/rl_calib/ak_mit_command.py` 실행본이 저장소 `protocols/ak_mit_command.py` 와 다르다(실행본 `a3582fa2`, 종전부터 그랬고 이번 배포와 무관) — 어느 쪽이 맞는지 확인.
4. 새 블랙박스 필드(`fb_age_max_s` 등)와 타임라인의 `"errlog": 1` 이 첫 세션에 찍히는지.

## 재동기화 (2026-10-08 04:32–04:37) — 둘째 합성 키 추가, 젯슨 = 저장소 최신 판
**한 일은 처음과 같다: 파일 복사·오프라인 selftest·콘솔 서비스 재기동. CAN 송신·ARM·토크·영점·센서 기동·제어 프로세스 조작·물리 실행은 없다.**
| 항목 | 값 |
|---|---|
| 원본 커밋 | `f1ad8221`(실행 코드를 마지막으로 바꾼 커밋 — 둘째 합성 키) |
| 매니페스트 | `deploy/head_1008.md5` md5 `8f4373d380d9e07a97408a69f3e3acdc` — 122 파일(전송 66·확인만 56). 직전 배포(`8c210340…`) 대비 바뀐 파일 12 개 |
| 바뀐 파일 | `rl_bridge_node.py` · `operator_console.py` · `console_state.py` · `selftest_w3_bridge_node.py` · `jetson_parity.py` · `parity_ref.npz`(5 → 6 키) · `rl_walking/__init__.py`(학습 태스크 등록 줄) · 시험 2 · `OPERATOR_COMMANDS.md` · 매니페스트·VERSION |
| 백업·되돌리기 | 그대로 `~/humanoid_repo_bak_1008/` — **처음(1005 판 + 패치) 상태의 원본**을 보존한다(재적용은 원본 백업을 덮어쓰지 않는다 — 로컬 왕복 시험 T14). 되돌리기 한 줄 동일 |
| 사전 검증 | 로컬 왕복 시험(가짜 젯슨, 배포 → 되돌리기 → 재적용) 37 / 37 · 드라이런 "예상 밖 0" · 재적용 허용 목록 = 직전 배포 매니페스트의 md5 |
| 젯슨 selftest | 21 묶음 전부 통과(실패 0): 브리지 selftest 7 종(새 키 포함, 각 28–29 s)·W3 노드·콘솔·stage8 기본 인자·러너 동치·md5sum -c 122 / 122. 로그 `~/humanoid_repo_bak_1008/selftest_20261008_043240.log` |
| 러너 동치(6 키) | 목표각 최대 차 5.4e-07 · 4.2e-07 · 3.6e-07 · 7.2e-07 · 7.2e-07 · **새 키 7.2e-07 rad**, 모드 불일치 0 |
| 콘솔 | 04:36:58 재기동(제어 프로세스 0·유휴 확인 뒤), PID 68121. 첫 화면 드롭다운에 새 키 표시 확인 |
| 불변 확인 | 하드웨어맵 md5 `41b68a6f`(같음) · `config/`·`jetson/calib_data/` 에서 바뀐 파일 0(배포 전후 목록 대조 `jetson/inventory_after2_md5.txt`) |

**새 키** `stand_kew3q_g21a_g24ra_w3s3d15800`(콘솔 표시 "걷기 실험 · 서기 KE + 걷기 W3 합성 + 갇힘 해소(시뮬 평가 구성) · 실물 미검증"):
- 첫 합성 키와 같은 구성에 KE 감지기의 갇힘 해소 `(2.0 s, 0.3 rad/s, 양발 접촉)`만 켠 것이다. **오늘 밤 합성 러너 시뮬 평가가 돌린 구성이 이 키다**(`reports/gen_w3/TRANSITION_W3KE.md` §5.21).
- 기존 키 5 개의 동작은 그대로다: 기존 키 세션 12 종의 노드 발행이 변경 전 커밋과 비트 동일(에이전트 확인), 동치 기준의 앞 5 키 배열 50 개가 종전 기준과 비트 동일(주 작업자 확인).
- 알려 둘 것: 브리지 소스의 이 키 주석에 "젯슨 미배포"라는 문구가 남아 있다(작성 시점 기준). 동작과 무관한 주석이라 재배포를 피하려고 그대로 두었다 — 다음 변경 때 함께 고친다.
- 현장 확인 필요: 이 키로 실물을 돈 적이 없다. 갇힘 해소가 실물에서 발동하는 모양(회복 모드 중 조용히 2 s 서 있으면 서기로 복귀)은 블랙박스의 `requiet` 필드로 확인한다.
