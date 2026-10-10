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
