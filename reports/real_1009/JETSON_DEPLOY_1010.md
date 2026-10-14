# 젯슨 배포·정리 기록 (2026-10-10 16:35–17:05, 승윤님 "젯슨 싹 정리하고 배포진행해줘")

시작 조건: 콘솔 `/api/state` 로 세션 없음·stage8 꺼짐·브리지 꺼짐 확인. 물리 실행·CAN 송신·ARM 없음.

## 1. 정리 (지우기 전 로컬 보관 `/home/ryu/jetson_archive_1010/home_mama/`, 바이트 합 일치 확인)
- `~/logs/sessions/session_*` 29 개(6.95 GB)·`~/logs/rl_bridge_*.jsonl` 25 개·`candump*.log` — 삭제. 살아 있는 콘솔·센서 로그와 pid 파일은 남김.
- 홈: `stand_7차_*.tar.gz` 14 개·빈 파일 `8`·옛 pid 2 개·`ads1115_8ch_test.py` — 삭제.
- `~/rl_calib`: 코드 백업(`*.py.bak_*`·`*.sh.bak_*`)·옛 시험 로그(ak45_*·wiggle·dryrun 등) 53 개 — 삭제. **맵·맵 백업·level_pose·IMU 장착 보정·관절 한계·자세 기록은 손대지 않음.**
- 남긴 것: `humanoid_repo_bak_1008`(배포 도구의 롤백 백업), 무선랜·PCAN 드라이버 빌드, `humanoid_ws`. 디스크 29 → 22 GB 사용.

## 2. 저장소 배포
- `deploy_head_1008.sh` dry-run(예상 밖 0) → `--apply --restart-console`: 바뀜 17·새로 5, 매니페스트 181d53c4 → **55ce09cc**(원본 커밋은 VERSION), 젯슨 selftest 전부 PASS, 콘솔 재기동.
- 콘솔 선택지 2 개: `stand_kew3qa_…_w3s3d15800`(기본, 원래 몸 걷기) · `stand_kew3qa_…_w3nbj2_16299`(새 몸 종종걸음).

## 3. `~/rl_calib` 실행본
- hang_stand·boot_pose_check·release_all·ping_all·zero_anchor 를 저장소 판으로 교체(직전 사본 = 옛 저장소 판과 md5 같음, 백업 `~/rl_calib/bak_1010/`), `mech_fixed.py` 추가. 컴파일 확인.

## 4. 하드웨어맵 기구 고정 선언 (승윤님 승인 "지금 추가")
- 두 사본(`~/humanoid_repo/config/`·`~/rl_calib/`)의 left/right_hip_r_joint 에 `"mech_fixed": true` 한 줄씩. diff = 그 두 줄뿐, 선언을 뺀 JSON 은 직전과 동일(코드로 확인).
  md5 15ba675c → **e2cc21d3**(두 사본 같음). 백업 `robot_12dof_hardware_map.json.bak_pre_mechfixed_1010`.
- 확인: `load_fixed` = 모터 3·9, 콘솔 준비 단계와 같은 브리지 `--selftest`(전체) 두 키 모두 ALL PASS(선언된 실제 맵에서).

## 5. 알려진 문제 — 배포 검증 도구의 시험 2 종이 선언된 맵에서 실패
- `deploy_head_1008.sh --verify` 에서 `selftest_w3_bridge_node.py`(14 항)·`tests/test_stage8_latency_options.py -k test_default` 실패.
  두 시험이 젯슨의 실제 맵을 읽으면서 "기구 고정 없음"을 전제한다(원래 몸 걷기 키의 걷기 켜기 수락·12 모터 프레임 비트 비교) → 가드와 프레임 제외가 설계대로 동작한 결과.
  콘솔이 돌리는 시험은 아니다. 시험을 맵과 무관하게(선언 없는 임시 맵으로) 고치기 전에는 `--verify`·재배포 selftest 가 이 두 항에서 걸린다(재배포 때 자동 롤백 주의).

## 미검증·현장 확인
- 실물: `robotctl motors` 10 개 응답 + 3·9 "기구 고정 — 제외", boot_pose_check, stage8 기동(합성 피드백), 새 키 걷기. 7번 자리 영점은 승윤님이 재캡처.
- 걷기는 메뉴에서 **둘째 키**를 골라야 한다. 기본 키는 선언된 맵에서 서기만 되고 걷기 켜기는 거부된다.

## 6. 정책 하나로 정리 (1010 20:20–20:45, 승윤님 "정책도 깔끔하게 한 개" → "진행해")
- 콘솔 정책 메뉴 = `stand_kew3qa_g21a_g24ra_w3nbj2_16299` **하나**(기본값도 그 키). 원래 몸 키는 고를 수 없다.
- §5 의 시험 2 종 수정(현장 맵의 기구 고정 선언과 무관하게) + 예전부터 깨져 있던 `test_console_walk` 단언 수정. 커밋은 git log.
- 재배포: 바뀜 7(콘솔·W3 노드 시험·tests 3·기록 2), 매니페스트 55ce09cc → **ad6b7e44**, 젯슨 selftest 전부 PASS(선언된 맵에서 W3 노드·stage8 기본 시험 통과 확인), 콘솔 재기동.
- 정책 폴더: 25 → **8**(묶음 매니페스트가 요구하는 것만). 지운 17 개(8–9 월 세대)는 로컬 `/home/ryu/jetson_archive_1010/home_mama/humanoid_repo/logs` 에 보관. 66 → 25 MB.
  남긴 8 = 새 키가 쓰는 4(G21a 서기·G24Ra 회복·G21h2 공중·NBj 종종걸음) + 젯슨 자체 시험이 읽는 4(W3S3d 원래 몸 걷기·G11a·G12·0905 stand_v18b).
- 정리 뒤 `--verify` 전부 통과, 새 키 브리지 `--selftest`(전체) ALL PASS, 맵 e2cc21d3 불변.

## 7. 브리지 로그 등급 결함 수정 (1010 22:00 전후) · 요구 토크 상한 (1011 00:1x, 승윤님 "무조건 해야하지않아?")
- 22:0x: `rl_bridge_node.py` 한 파일(매니페스트 1bbb6ded). 승윤님 지시로 젯슨 selftest 는 중간에 끊음(로컬 selftest·왕복 통과). 그 뒤 실물에서 정책 시작·자립 정상.
- 00:1x: stage8 요구 토크 상한 옵션·콘솔 옵션 토큰·W3 노드 시험 헬퍼·tests 1(매니페스트 **84888155**), 젯슨 selftest 35 항 전부 PASS, 콘솔 재기동.
- **상한 켬**: 젯슨 `~/humanoid_repo/config/stage8_latency_opts` = `tc25,tcj.hip_a=20`(전 관절 25 N·m, 힙 외전 20 N·m — kp 150 에서 허용 |목표−실측| 9.55°·7.64°).
  콘솔이 stage8 기동 인자에 붙이는 것 확인(`--target-torque-cap-nm 25 --target-torque-cap-joint hip_a=20`), 젯슨에서 tests/test_stage8_torque_cap.py 6 항 OK. 하드웨어맵 dd1bc31d(승윤님 재캡처 값) 불변.
  끄기 = 그 파일 삭제 뒤 stage8 재기동. 이 파일이 있는 동안 `deploy_head_1008.sh` 의 selftest '지연 옵션 기본 0 개' 항은 `HEAD1008_ALLOW_LATENCY_OPTS=1` 로 돌려야 한다.
- 미검증: 상한을 켠 실물 동작 전부(서기·착지·걷기). 실물 첫 확인 = stage8 status 의 `target_torque_cap_counts`(서기 중에는 0 근처여야 함).

## 8. 1011 05:47–05:53 걷기 키 교체 배포 (NBj-s2 → NBjD-s2-17798) — 승윤님 "아침까지 종종걸음 젯슨까지 배포 완료… 승인받지 말고"

- 배경·판정: `NEWBODY_WALK.md` §19–§20. 사전 기준을 전부 넘지는 못했고(과도 요구 p95 30–34 > 23 N·m, 명목 왕복 낙상 5.6 %), 모터 8 을 멈춘 지표(25 N·m 초과 틱)가 11.1 → 0.4 % 라서 교체했다.
- 사전 확인: 콘솔 `/api/state` 제어 프로세스 0·세션 비활성·외부 제어 0. dry-run 예상 밖 0(바뀜 12·새로 4).
- 적용: `HEAD1008_ALLOW_LATENCY_OPTS=1 deploy_head_1008.sh --apply --restart-console` — 매니페스트 84888155 → **5b98d495**, 젯슨 selftest 36 항 전부 PASS(러너 동치 포함), 콘솔 재기동 active.
- 적용 뒤 확인: 콘솔 메뉴 = `stand_kew3qa_g21a_g24ra_w3nbjd2_17798` 하나, 그 키 지정 브리지 `--selftest`(젯슨, onnx) ALL PASS(정책 500 틱 ×15 포함),
  하드웨어맵 md5 dd1bc31d 불변, `config/stage8_latency_opts` = `tc25,tcj.hip_a=20` 불변(상한 켬 유지). 옛 정책 폴더 `…w3s3dnbj_s2_deploy16299` 는 남아 있다.
- 되돌리기: `deploy/deploy_head_1008.sh --rollback --restart-console`(직전 84888155 = NBj-s2 키).
- 이번에 같이 올라간 것: stage8 기동 로그의 상한 표시 줄, W3 노드 시험대의 로그 등급 규칙, 걷기 켜기 관련 tests.
- 미검증(현장): 상한 켠 stage8 의 실물 동작, 새 정책의 실물 걸음, 모터 8 정지의 직접 원인.
- 현장 확인 순서(제안): ① 모터 점검에서 10 개 응답(3·9 제외) ② stage8 기동 로그에 "요구 토크 상한 켬" ③ 서기 자립 ④ 줄을 느슨히 잡은 채 **제자리 걸음** 먼저
  ⑤ 끝난 뒤 상태의 `target_torque_cap_counts`(hip_a 가 몇 번 걸렸는지)와 블랙박스의 right_hip_a 목표−실측을 본다. 걸음이 눌려 무너지면 `tcj.hip_a=23`.

## 9. 1011 07:20–07:29 걷기 키 재교체 배포 (NBjD-s2 → NBjE-s1-18997)

- 배경·판정: `NEWBODY_WALK.md` §21. NBjD-s2 보다 모든 힙 외전 요구 지표가 낮고(25 초과 틱 0.4 → 0.0 %, 걷기 켠 뒤 솟음 중앙 19–20 → 16–17 N·m) 낙상 기준을 넘는다. 과도 p95 26–27(기준 23)·표류 0.031(기준 0.03)은 미달로 남는다.
- 사전 확인: `/api/state` 제어 프로세스 0·세션 비활성·외부 제어 0, 로컬 왕복 PASS 37, dry-run 예상 밖 0(바뀜 10·새로 4).
- 적용: 매니페스트 5b98d495 → **66297d40**, 젯슨 selftest 전부 PASS, 콘솔 재기동.
- 적용 뒤 확인: 콘솔 메뉴·걷기 프로필 = `stand_kew3qa_g21a_g24ra_w3nbje1_18997` 하나, 그 키 지정 브리지 `--selftest`(젯슨 onnx) ALL PASS, 맵 md5 dd1bc31d 불변, 상한 파일 `tc25,tcj.hip_a=20` 불변.
- 젯슨에 남은 옛 걷기 정책 폴더: `…w3s3dnbj_s2_deploy16299`, `…w3s3dnbjd_s2_deploy17798`(키는 등록돼 있지 않다 — 되돌리려면 그 커밋의 묶음을 다시 배포).
  `--rollback` 이 되돌리는 지점은 백업 폴더(`~/humanoid_repo_bak_1008`)의 원본 기준이므로 쓰기 전에 dry-run 출력의 백업 매니페스트를 확인한다(§8 의 "직전 84888155 로 롤백" 표현은 확인하지 않은 것).
- §8 의 현장 확인 순서는 그대로다. 미검증 항목도 같다(상한 켠 stage8 실물 동작·새 걸음·모터 8 정지 원인).

## 10. 1011 17:10–17:20 stage8 토크 상한 감쇠 항 포함 배포 (매니페스트 80f85880)

- 계기(승윤님): "절대로 25 를 넘으면 안 돼." 종전 상한은 위치 항 kp·(목표−실측) 만 묶었다 → 밀려 움직이는 관절에서 상한 + kd·|속도| 가 나온다.
  시뮬 왕복 기록(NBjE-s1, 214 env)으로 계산: 종전 상한 뒤에도 힙 외전 최대 29, 무릎 35 N·m, 25 초과 틱 0.01–0.08 %(발목 굽힘 0.05–0.08 % 는 kp 150·kd 5 가정 계산).
- 변경: 허용 창을 kd·실측속도/kp 만큼 옮겨 kp·(목표−실측) − kd·속도 의 합을 상한 안에 둔다(`torque_cap_damping_shift_deg`). 속도를 모르면 종전. 상한 꺼짐 경로 불변. 옵션 파일(`tc25,tcj.hip_a=20`) 불변.
- 검증: 로컬 `tests/test_stage8_torque_cap.py` 7·`test_walk_on_closed_loop.py` 4 통과, 로컬 왕복 PASS 37, dry-run 예상 밖 0(바뀜 4), 젯슨 selftest 전부 PASS, 젯슨에서 상한 시험 7 통과(가짜 버스), 맵 md5 dd1bc31d 불변, 콘솔 유휴.
- 이론상 남는 초과 경로(미검증): ① 실측 위치·속도가 한 송신 주기(10–20 ms) 낡은 동안 관절이 더 움직인 만큼 ② 실측이 낡아 동결된 관절은 묶지 않음(종전 설계) ③ 모터 펌웨어 내부 루프·자체 전류 제한 동작은 모름. 힙 외전은 상한 20 이라 25 까지 5 N·m(속도 오차 1 rad/s 분) 여유가 있고, 나머지 관절은 상한 25 라 여유가 없다.

## 11. 1011 18:05 토크 상한 값 변경: `tc25,tcj.hip_a=20` → `tc20,tcj.knee=23`

- 계기(승윤님): "나머지도 20 으로 자르는 게 낫지 않아?" → 권고(무릎만 23) 뒤 "ㄱㄱ".
- 근거(NBjE-s1 왕복 214 env, 한계만 바꿈): 힙 외전만 20 = 낙상 6, 전 관절 25 = 10, 전 관절 23 = 12, 전 관절 20 = 18. 20 을 넘겨 쓰는 관절은 무릎뿐(걷는 틱의 2.5–2.7 %, 23 초과 0.5–0.6 %), 힙 굽힘·발목 굽힘은 0 %, 발목 롤 0.2–0.6 %.
- 적용: 젯슨 `config/stage8_latency_opts` 한 줄 교체(직전본 `.bak_1011_tc25`), 콘솔 유휴 확인 뒤. 콘솔이 붙일 인자 확인: `--target-torque-cap-nm 20 --target-torque-cap-joint knee=23`. 로컬에서 관절별 상한 풀이 확인(무릎 23, 나머지 20). 맵 md5 불변. 다음 stage8 기동부터 적용.
- 미검증: "전 관절 20 + 무릎 23" 조합 자체는 시뮬에서 돌리지 않았다(평가기가 관절별 혼합 한계 미지원 — 전 관절 23 과 20 사이로 추정). 낙상 수 차이는 seed 한 번씩이라 흩어짐이 섞여 있다. 실물 미검증.
- 되돌리기: 젯슨에서 `cp config/stage8_latency_opts.bak_1011_tc25 config/stage8_latency_opts` 뒤 stage8 재기동.

## 12. 1011 18:13 실물 사고와 되돌림 (§10 의 감쇠 항 포함이 원인)

- 사고: 18:13 제자리 걸음(콘솔 버튼, 2 s 기한) 직후 left_ankle_r(모터 6, AK45-36)이 ±35° 를 최고 속도(4.3–4.5 rad/s)로 3 회 왕복하며 기구 끝에 부딪히다(끝에서 토크 ±14 N·m), 1.8 s 부터 한 방향으로 계속 돌아 −586° 까지 감. 5.3 s 에 sensor_stale 로 전체 정지. 승윤님 보고: 배터리 폭발.
  기록: 젯슨 `~/logs/rl_bridge_20261011_181136.jsonl`, 세션 `session_20261011_180920_413b6f`.
- 원인(블랙박스로 확인한 기전): §10 의 `torque_cap_damping_shift_deg` 는 허용 창 중심을 kd·속도/kp 만큼 옮긴다. kd·|속도| > 상한(5 × 4 rad/s = 20 N·m)이면 창이 실측각을 포함하지 않아 보내는 목표가 항상 움직이는 방향 쪽에 놓인다 → 양의 되먹임. 들린 발의 발목 롤은 관성이 작아 0.1 s 만에 4 rad/s 를 넘었다.
  모터 8(힙 외전) 추정 토크는 걷는 동안 5 N·m 이하였다 — 이번 사고는 걷기 정책의 과토크가 아니다.
- 잘못: 개루프 토크 부등식만 시험하고 폐루프 안정성을 시험하지 않은 채 실물 직전에 제어 경로를 바꿨고, "오프라인 검증 완료"로 보고했다. §10 의 "검증" 항목은 이 결함을 볼 수 없는 시험이었다.
- 조치(18:55–19:00): 콘솔 유휴 확인 → 기록 세션 종료(API) → stage8·시험을 99652353 판으로 되돌려 배포(매니페스트 **66297d40** = 1011 07:29 판과 동일, 젯슨 selftest PASS 36·실패 0). 젯슨 stage8 에 `torque_cap_damping_shift_deg` 0 건 확인. 옵션 파일도 `tc25,tcj.hip_a=20` 으로 복원(§11 의 `tc20,tcj.knee=23` 은 `.bak_1011_tc20`).
- 젯슨 맵: 18:10 승윤님이 모터 4 배수 보정 → md5 f2eef8aa(저장소 사본·`~/rl_calib` 사본 일치). 배포는 맵을 건드리지 않았다.
- 승윤님 결정(18:50): "서기만 하자", 랜선 없이 `robotctl` 로만 제어. 지면 걷기는 콘솔 웹 패널에서만 켤 수 있고 robotctl 에는 그 명령이 없다(`robotctl walk` 는 공중 다리 스윙 — 지면 걷기 아님). 걷기 코드를 막는 추가 수정은 하지 않았다(실물 직전에 코드를 또 바꾸지 않기 위해).
- 현장 확인 필요: 모터 6 기구·배선(여러 바퀴 회전), 배터리·전원, 모터 6 영점 프레임(prep 의 영점 확인에서 드러남 — 어긋나면 zero-fix).
- 미검증: 위치 항만 묶는 상한(`tc25,tcj.hip_a=20`)은 실물에서 한 번도 돌지 않았다(1010 밤 추가 뒤 첫 실물이 오늘의 §10 판이었다).

## 13. 배포 묶음 준비 — 새 몸 걷기 키 W1c(0.8 Hz) + 전환 완화 1 s (10-12 02:15, **젯슨 미배포 — 승인 대기**)

- 내용: 새 몸 합성 키 `stand_kew3qa_g21a_g24ra_w3nbje1_18997` → `stand_kew3qa_g21a_g24ra_w3nbjy1_19996`(걷기 정책 W3S3dNBjY-19996, 걸음 시계 0.8 Hz,
  서기 → 걷기 전환 완화 1 s·hold). 서기·회복·공중 정책(G21a·G24Ra·G21h)과 stage8·토크 상한·하드웨어맵은 그대로. 근거 `WADDLE_W1.md` §4·§6.
- 매니페스트 d8e7a171(직전 젯슨 = 66297d40). 젯슨에서 바뀌는 파일 8: `rl_walking/deploy/kew3_runner.py`·`w3_deploy.py`, `jetson/scripts/rl_bridge_node.py`·`operator_console.py`·
  `console_state.py`, `tests/test_console_walk.py`·`test_operator_console.py`, `reports/night_1007/jetson_parity.py`. 새 정책 폴더 4 파일 추가(옛 NBjE 폴더는 매니페스트에서 빠짐).
  **stage8 노드는 바뀌지 않는다**(T9b: CAN 프레임 39056 개 젯슨 판과 동일).
- 허용 목록 `deploy/head_1008.accept`: 02:10 젯슨 무응답이라 실기에서 읽지 못하고 커밋 07df021a 의 매니페스트(= 66297d40)에서 옮겼다. 배포 dry-run 이 실기 md5 와 대조한다(★예상 밖 0 이어야 적용).
- 로컬 모의 배포(`deploy/test_head_1008_local.sh`): **PASS 37 · FAIL 0** — selftest 23/23, 자동 롤백, 종전 서기·회복 키의 브리지 발행 27 건 젯슨 판과 동일.
- 자가검증: kew3_runner·rl_bridge_node(정책 500 틱 ×17)·selftest_w3_bridge_node·operator_console ALL PASS. 등록 키 폐루프 = 평가 구성과 수치 일치.
- 시뮬(계획 그대로: 제자리걸음 5 s × 2, 힙 외전 한계 20): 낙상 2/214·다른 초기조건 0/204(종전 키 3/214), 8번 제한 뒤 최대 20.0·요구 최대 29.7, 걷기 켠 뒤 0.2 s 목표 변화 0.5°(종전 11.4°).
- 미검증: 젯슨 추론 시간(전환 완화 hold 는 추론 추가 없음), 실물에서 처음 도는 것 = 이 러너 변경·stage8 위치 항 상한(10/10 판)·사고 뒤의 모터 6.
- 되돌리기: 커밋 07df021a 의 묶음(66297d40)을 다시 배포.
- 배포 절차(승인 뒤, 젯슨이 켜져 있고 콘솔 유휴일 때): `HEAD1008_ALLOW_LATENCY_OPTS=1 deploy/deploy_head_1008.sh`(dry-run, ★예상 밖 0 확인) → `--apply --restart-console`.

**03:40 갱신(같은 묶음에 robotctl 걷기 명령 추가)** — 매니페스트 d8e7a171 → **1339b4f6**. 승윤님 "콘솔은 이제 안써 robotctl로만 제어".
- 추가: 걷기 동작 `inplace5`(제자리·기한 5 s — 기한이 지나면 브리지 감독기가 끔), `robotctl march [-y] [--move inplace5] [--list]`·`robotctl march-off`.
  march 는 콘솔 걷기 버튼과 같은 서버 액션(w3_move)을 보낸다(선행조건·운영자 확인·기한 그대로). 콘솔 패드에 inplace5 자리 추가.
- 젯슨에서 바뀌는 파일 10: 앞의 8 + `jetson/scripts/robotctl.py`·`tests/test_robotctl.py`. stage8·상한·맵 불변.
- 로컬 모의 배포 PASS 37 · FAIL 0(매니페스트 1339b4f6). 자가검증 4 종 ALL PASS, test_robotctl 27·test_console_walk 21·test_operator_console 60 통과.
- 주의: 젯슨의 `~/bin/robotctl` 이 저장소 파일을 가리키는지(링크) 사본인지 확인하지 못했다(02:10 젯슨 무응답) — 배포 뒤 `robotctl march --list` 로 새 명령이 보이는지 확인할 것.
- 실물 순서(승인 뒤): 모터 6·전원 점검 → `robotctl prep` → `level` → `policy-on` → 지그 내림 → 서기 확인 → `robotctl march`(확인 1 회 뒤 5 s 제자리걸음, 스스로 서기 복귀). 멈춤 `march-off`, 비상 `estop`.

**03:40 젯슨 배포 완료(승윤님 승인 "ㅇㅇ 배포해도 되고")** — 매니페스트 66297d40 → **1339b4f6**.
- 사전 점검(dry-run): 일치 133/133 · 바뀔 파일 12 · 새 파일 4 · ★예상 밖 0. 콘솔 유휴(제어 프로세스·세션 없음) 확인 뒤 `--apply --restart-console`.
- 젯슨 selftest PASS 35 · FAIL 0(러너·콘솔·브리지 각 키·W3 노드·stage8 기본/단위·jetson_parity·md5sum -c). 백업 `~/humanoid_repo_bak_1008`.
- 배포 뒤 읽기 확인: 선택 키 `stand_kew3qa_g21a_g24ra_w3nbjy1_19996`·방식 air, 걷기 동작에 inplace5(5 s), `~/bin/robotctl march --list` 에 새 동작 표시,
  `KEW3_BLEND = {…: (1.0, "hold")}`·`f_step=0.8` 반영, stage8 md5 c039fc02(불변)·`torque_cap_damping_shift_deg` 0 건, 옵션 파일 `tc25,tcj.hip_a=20`, 하드웨어맵 두 사본 f2eef8aa(불변).
- 실물 미실행. 되돌리기 `deploy/deploy_head_1008.sh --rollback --restart-console`(직전 판 66297d40).
