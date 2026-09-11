# 실행카드 — stand_hang_v5 · 매달림→착지→지지 제거→무보조 자립 (2026-09-11 확정판)

이전판(0910, "70% 지지 핸드오프·발력% 판정·CoP")은 **폐기**한다. 0910 3회 원시 기록 재검증 결과, 실패는 낙상이 아니라 접촉 상실 정지였고 발력%·CoP 기반 결론은 미보정 패드 위의 추론이었다(`reports/real_prep_0911/hypotheses_and_field_checks.md`). 판정 규칙은 CLAUDE.md(0911 개정)를 따른다.

## 후보(묶음) — 코드·모델·설정 해시 고정
| 항목 | 값 |
|---|---|
| 정책 | `models/stand_hang_v5/policy.pt` md5 `2a20b7b1cd4ec3da…` (run `2026-09-09_19-15-42/model_139987`, 관측 320) |
| 브리지 | `jetson/scripts/rl_bridge_node.py` — **인계 워밍업 4 s**(0911), any-pad 접촉, LIFT_STOP 1.0 s, 신선도 게이트 불변 |
| 콘솔 | Codex 리뉴얼(0911): 실제 응답 기반 단계, 접지 인계/공중 시작 분리, 윈치 이벤트, 무보조 확인 후 관측 타이머, `--demo` |
| 게인·필터 | kp150/kd5, 슬루 4°/틱, LPF β0.27, 블렌딩 1.5 s (변경 없음) |
| 절차 | 접지 후 정책 인계(행잉 ARM 미체크), 좌우 대칭 저속 방출, 모든 줄 느슨 확인 후 관측 |
| 커밋 | 브랜치 `hang2stand-prep-0911` (패키지 MANIFEST.md5 와 대조) |

이전 3회 대비 달라지는 점과 근거: `reports/real_prep_0911/hypotheses_and_field_checks.md` §C.

## 1. 배포 전 최종 체크 (PC)
| 항목 | 명령 | 기대 |
|---|---|---|
| 저장소 | `git status --short \| grep -v '^??'` | 빈 출력 |
| v5 md5 | `md5sum models/stand_hang_v5/policy.pt` | `2a20b7b1…` = meta.json |
| 브리지 selftest | `python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5` | `ALL PASS` (「인계 워밍업」·「학습 계약 … stand_hang_v5」 포함) |
| 콘솔 회귀 | `python3 -m pytest -q tests/test_operator_console.py` | 35 passed |
| 콘솔 모의 | `python3 jetson/scripts/operator_console.py --demo --port 8811` → 절차 클릭 | 정상·정지·회수 전이 (`reports/real_prep_0911/console_mock_validation.md`) |
| 패키지 | `GENS="stand_v18b stand_hang_v4 stand_hang_v5" bash jetson/deploy_package.sh` | `deploy/stand_7차_<stamp>.tar.gz` + `deploy/MANIFEST.md5` |

## 2. 젯슨 반영 (승윤님 실행 — 파일 복사·md5·selftest·콘솔 재기동만; CAN·모터 무접촉)
```bash
cd /home/ryu/humanoid_leg_test1
bash jetson/deploy_to_jetson.sh --host 172.16.35.89 --pkg $(ls -t deploy/stand_7차_*.tar.gz | head -1)
# 젯슨: LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK ; python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5
```
되돌리기: 직전 `deploy/stand_7차_<이전 stamp>.tar.gz` 를 같은 방법으로 풀고 md5·selftest 확인 (`reports/real_prep_0911/deploy_path_check.md`).

## 3. 현장 사전 확인 (승윤님)
- 모터 전원 ON → 콘솔 `영점 확인 12/12`. 불일치면 배수 보정 우선, 재캡처 금지.
- 하드웨어맵 stand_zero 12개 로컬↔젯슨 일치(0911 변경). 갱신 시 stage8 재기동.
- **IMU 4방향 부호 확인**(매달린 채 앞·뒤·왼·오른으로 살짝 밀어 콘솔 앞/뒤·좌/우 부호가 방향과 일치하는지, gyro 잡음 ≤0.01 rad/s). 0910 로그의 롤 부호 해석은 이 확인이 있어야 확정된다.
- 후크가 CoM 연직인지(정지 상태 좌우 기울기 0 ±1°), 줄이 옆으로 당기지 않는지. 착지 직후 양 발목 피치 추종오차가 같은 방향으로 커지는지(0910 전조) 콘솔 관절표에서 관찰.
- 오른발 패드 응답 확인(`cop_from_pads.py --identify`), 캡 탈락 여부. 영상 카메라: 발·줄·후크가 보이게.
- 시험 방식 = **접지 후 정책 인계**, 선택 정책 = stand_hang_v5, 행잉 ARM 미체크. `기록 시작`(첫 제어 전).

## 4. 콘솔 절차 (서버가 실제 응답으로 완료 판정 — 클릭 성공 ≠ 완료)
1. `CAN 준비` → `센서 기동` → `브리지 셀프테스트`(ALL PASS).
2. `stage8 선기동` → `hang_stand` → `게인 램프 kp150`(최대 오차 < 0.3°).
3. 후크 정렬 후 천천히 내려 **양발 접촉** 표시 → `▼ 착지`.
4. `접지 선언` → `핸드오프`(인수 홀드 12/12) → `기준점 12/12` → `ARM` → `브리지 live`(선택 = 로드 = v5 확인).
5. `stand 발사` → 브리지 로그 **"인계 워밍업 완료 — 정책 구동 시작"** 까지(≤ 4 s) 줄을 건드리지 않는다.
   - `sensor_stale_warmup_timeout`: 방출 금지 → stage8 status/CAN 점검 → `ARM`·`stand` 재요청. `sensor_stale` 반복 시 중단.
6. `▶ 방출 시작` → 좌우 대칭·저속·저크 없이. 콘솔 **좌/우 기울기**가 3° 를 넘어 커지거나 한 발이 눈에 띄게 뜨면 `⏸ 방출 정지`/`↥ 재인장`(시도 무효 기록).
7. 줄이 완전히 느슨하고 손·장치 보조가 없음을 확인 → **`✓ 모든 줄 느슨·무보조 확인`** → 관측 타이머 시작. 손대지 않는다.
8. 5 s → 30 s 관측. 넘어질 조짐(전후 ±10°, 좌우 ±8° 접근)이면 회수 우선.
9. 회수: `↥ 재인장`(`support_confirm`) → `스택 종료` → **그 다음** `릴리즈` → `기록 종료` → `이 시험 로그 분석`.

## 5. 판정 (발력%·CoP 사용 금지)
- **정책 시작 성공**: 워밍업 완료 로그 + 정책 제어 응답.
- **부분지지 유지**: 방출 중 가드 정지 없이 유지.
- **무보조 자립 5 s / 30 s**: `✓ 모든 줄 느슨·무보조 확인` 이후 관측 시간 + **영상**으로 줄 느슨·무보조 재검토. 재인장·손 보조·가드 정지·데이터 소실 시 해당 시도 무효.
- 단일 IMU 화면·접촉 비트만으로 성공을 선언하지 않는다.

## 6. 즉시 중단
전후 기울기 게이트 초과 · 발진 · 가드 정지 · 센서 stale 반복 · 줄이 몸을 옆으로 당김 · 스포터 개입 필요.

## 7. 시험 후 (PC, Claude)
```bash
scp mama@172.16.35.89:~/logs/rl_bridge_<stamp>.jsonl  reports/real_<날짜>/   # 블랙박스
scp -r mama@172.16.35.89:~/logs/sessions/session_<stamp> reports/real_<날짜>/  # 타임라인(윈치 이벤트)
python3 jetson/scripts/analyze_hang2stand_trial.py reports/real_<날짜>/rl_bridge_<stamp>.jsonl --timeline reports/real_<날짜>/session_<stamp>/timeline.jsonl
python3 jetson/scripts/analyze_fall.py <블랙박스>            # 낙하 등급(성공 판정 아님)
python3 jetson/scripts/analyze_cop_transfer.py <pads> <bb>   # 미보정 진단만
```
판별 규칙(가설 H1~H5, 무엇이 무엇을 지지/반박하는지): `reports/real_prep_0911/hypotheses_and_field_checks.md` §B.

## 8. 시뮬 기대치 (실물 성공률 아님)
명목 99.6 %(5 s)·99.6 %(30 s), DR·푸시 OFF 100 %, 현실적 접촉 교란 100 %, 횡 편심 ±2 cm 명목 100 %. 실물 무보조 자립은 **미검증**이다.
