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
