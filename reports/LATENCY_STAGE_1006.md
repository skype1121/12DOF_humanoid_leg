# stage8 지연 단축 단계 적용 — 현장 절차 카드 (1006)

> 승인된 방향: **1단계 = `--feedback-after-send`(fas)만**. 들어 올린 무부하 상태에서 이상이 없을 때만
> 2단계 `--send-on-command`(soc). **두 기능을 한 번에 켜지 않는다.** 언제든 옵션 파일을 비우면 종전 경로.
>
> 이 카드는 실행 허가가 아니다. 젯슨 배포(`--apply`)·옵션 변경(`--stage`)·stage8 기동은 승윤님이 직접 실행한다.

## 0. 무엇이 바뀌나

| 구성 | 내용 |
|---|---|
| `config/stage8_latency_opts` (젯슨) | 첫 줄 `''`(비움/없음) = 종전 · `fas` = 1단계 · `fas,soc` = 2단계. 콘솔이 **stage8 기동 순간**에 읽는다 → 바꾼 뒤 stage8 재기동해야 반영 |
| `jetson/scripts/latency_check.py` | **읽기 전용** 판정기. status 토픽을 N 초 구독 → PASS / FAIL / INCONCLUSIVE, JSON 저장 |
| stage8 노드 | 기존 옵션 그대로 + `send_counts.hold`(인수 홀드 송신 틱 수, 계측 전용 — 동작 불변) |
| 배포 스크립트 | `deploy/deploy_lat_1006.sh` — 기본 점검(dry-run), `--apply`, `--stage off|fas|soc`, `--rollback` |

하드웨어맵은 배포하지 않는다. 서비스 설정(systemd)은 바꾸지 않는다.

## 1. 사전 조건

- 로봇은 **들어 올린 상태(발 무접촉·무부하)**, 지지 줄 인장 확인, 손 보조 인원 대기.
- 측정 상태 = 콘솔 정상 순서로 **stage8 선기동 → 기립 스크립트 → 인수 홀드**까지. **ARM·정책 발사는 하지 않는다.**
  (판정기는 인수 홀드 송신(`send_counts.hold`)과 armed 송신을 모두 센다. 송신이 없으면 INCONCLUSIVE.)
- 젯슨 콘솔이 옵션 파일을 읽는 판인지: `deploy/deploy_lat_1006.sh`(dry-run) 출력의 "콘솔이 옵션 파일을 읽는 판인가" ≥ 1.
  0 이면 `--apply --restart-console` 이 필요(제어 프로세스 없을 때만).
- 콘솔 서비스 환경변수에 `STAGE8_LATENCY_OPTS` 가 **없어야** 한다(있으면 파일보다 우선 — 스크립트가 거부).

## 2. 순서

| # | 할 일 | 명령 (PC: 저장소 루트 / 젯슨: `~/humanoid_repo`) |
|---|---|---|
| 1 | 점검(읽기만) | PC `bash deploy/deploy_lat_1006.sh` |
| 2 | 배포(승인 뒤) | PC `bash deploy/deploy_lat_1006.sh --apply` (콘솔이 구판이면 `--apply --restart-console`) |
| 3 | 옵션 off 확인 | PC `bash deploy/deploy_lat_1006.sh --stage off` (파일 비움 = 종전) |
| 4 | **기준 측정 (off)** | 콘솔 정상 순서로 스택 기동 → 인수 홀드 안정 뒤 젯슨에서<br>`source /opt/ros/humble/setup.bash && python3 jetson/scripts/latency_check.py --expect off --duration 60` |
| 5 | 회수 | 콘솔: **줄 인장 확인 → 스택 종료 → 그 다음에만 release_all**(필요 시). 동시 실행 금지 |
| 6 | 1단계 적용 | PC `bash deploy/deploy_lat_1006.sh --stage fas` (제어 프로세스 없을 때만 허용) |
| 7 | **fas 측정** | 콘솔 정상 순서로 스택 재기동 → 인수 홀드 안정 뒤 젯슨에서<br>`python3 jetson/scripts/latency_check.py --expect fas --duration 60` (4번과 같은 시간·같은 상태) |
| 8 | 판정 | 7번이 **PASS** 이고 4번 대비 퇴보(아래 표 '비교')가 없을 때만 1단계 유지. FAIL/INCONCLUSIVE → 3절 즉시 복귀 |
| 9 | 회수 | 5번과 같은 순서 |

결과 파일: 젯슨 `~/logs/latency_check/latcheck_<off|fas>_<시각>.json`, 최신 사본 `last_off.json`·`last_fas.json`.
종료 코드: 0 PASS · 1 FAIL · 2 INCONCLUSIVE · 3 구독 실패. **INCONCLUSIVE 는 PASS 가 아니다.**

## 3. 판정 기준 (`lat_1006_v1`, 측정 전에 고정 — `latency_check.py` 의 `CRITERIA`)

| 항목 | 기준 | 무엇을 보나 |
|---|---|---|
| 실제 로드 옵션 = 기대 | `--expect` 와 일치, 50 Hz | 파일만 바꾸고 재기동 안 한 경우를 잡는다 |
| 측정 중 재기동/옵션 변경 | 0 | 카운터 역행·옵션 변화 |
| 틱 간격 p99 (250틱 창별 최악) | ≤ 25 ms | 지터 |
| 틱 간격 최대 (창별 최악) | ≤ 45 ms | 명령 공백 |
| 틱 대역 밖(20 ± 5 ms) 비율 | ≤ 2 % | 지터 |
| 송신 틱/초 | 50 ± 10 % | 송신 주기 |
| 응답/송신 (missed response) | ≥ 0.99 | `can_rx_count` 증분 / (송신 틱 × 관절 수) |
| fas 같은 틱 응답 완결률 | ≥ 0.99 (fas 일 때만) | `fb_after_send.complete / ticks` |
| stale 표본 수 | 0 | 관절별 `feedback_stale` |
| 관절 피드백 나이 최대 | ≤ 0.10 s | 관절별 `feedback_age_sec` |
| CAN 송신 타임아웃 증분 | 0 | `can_send_timeouts` |
| CAN 디코드 오류 표본 | 0 | `can_decode_error` |
| CAN 상태 비정상 표본 | 0 | `can_status` ∉ {OK, BUS_OPEN…, LINK_UP…, BRINGUP_OK…} |
| 새 stage8 errors | 0 | 측정 중 새로 생긴 `errors` |
| status 수신 공백 최대 | ≤ 0.10 s | 콘솔·브리지가 보는 신선도 |
| 판정 불가 | 측정 < 20 s · 송신 틱 < 500 · tick 창 < 2 | → INCONCLUSIVE |

**비교(기준 off ↔ fas, 같은 시간·상태)**: fas 의 틱 p99·최대·대역 밖·응답/송신·피드백 나이 최대가 off 보다 나빠지면
PASS 여도 1단계 유지를 보류하고 기록을 검토한다. 보고서 `참고` 줄의 `fas 최대 대기`(상한 4 ms)도 기록한다 —
상한에 붙어 있으면 실버스에서 응답이 대기창을 넘는 틱이 생길 수 있다.

## 4. 즉시 복귀 (언제든)

1. 이상(FAIL·떨림·경고음·오류)이 보이면 **먼저 회수 순서**: 줄 인장 → 콘솔 스택 종료 → 그 다음에만 release_all.
2. PC `bash deploy/deploy_lat_1006.sh --rollback` — 옵션 파일 비움(= 종전 경로, 실행 중이어도 허용; 실행 중 stage8 은
   재기동 전까지 기존 옵션 유지) + 코드 되돌리기 명령(`.bak_1006` 복원) 안내 출력(자동 실행 안 함).
   젯슨에서 직접: `: > ~/humanoid_repo/config/stage8_latency_opts`
3. 다음 stage8 기동 뒤 `latency_check.py --expect off` 로 실제 종전 경로인지 확인.

코드 자체를 되돌릴 일은 드물다 — 옵션이 꺼지면 stage8 은 종전 동작(옵션 코드 경로 미사용). 추가된 `send_counts.hold` 는 계측뿐.

## 5. 2단계(soc) 조건

- 1단계 fas 측정이 **PASS**, 기준 대비 퇴보 없음, 승윤님 승인.
- `--stage soc` 은 젯슨 `~/logs/latency_check/last_fas.json` 이 PASS(기대 fas·실로드 fas 단독)이고, 현재 옵션 파일이 `fas` 이며,
  그 결과가 fas 설정 **이후** 측정일 때만 파일을 `fas,soc` 로 바꾼다. 그 외에는 거부한다.
- soc 의 효과(명령 즉시 송신)는 **armed + 명령 스트림**에서만 나타난다(인수 홀드는 종전 타이머 경로). 2단계 측정 상태
  (ARM 여부·브리지 명령 유무)는 별도로 정해 승인받는다. 측정은 `--expect fas,soc`.
- `--control-hz 100` 은 이 절차에 포함되지 않는다(별도 승인).

## 6. 오프라인 검증 (1006, 로컬 모의 벤치 — 실물 수치 아님)

`python3 jetson/bench/latency_mock/latency_bench.py --variants base,fas --duration 30 --latcheck --out <dir>`
(실제 stage8 노드 + python-can 가상 버스 + 모의 모터 12개, 격리 ROS 도메인 87, armed 스트리밍, 29 s 측정)

| 항목 | base(off) | fas |
|---|---|---|
| 판정 | PASS | PASS |
| 틱 p99 / 최대 (최악 창) | 20.25 / 22.76 ms | 20.18 / 20.34 ms |
| 대역 밖 | 0 | 0 |
| 송신 틱/초 | 50.006 | 50.013 |
| 응답/송신 | 1.0 | 1.0 |
| fas 완결률 · 최대 대기 | — | 1.0 · 3.95 ms |
| 피드백 나이 최대 / p99 | 20.1 / 19.7 ms | 18.7 / 17.8 ms |
| stale · CAN 오류 | 0 · 0 | 0 · 0 |
| (벤치 별도 계측) 관측 나이 p50 | 32.0 ms | 15.1 ms |
| (벤치) 루프 틱 평균 | 2.16 | 1.32 |

음성 대조: status 발행자 없는 도메인 → INCONCLUSIVE(rc 2). 합성 열 단위 테스트에서 옵션 불일치·응답 누락·stale·
CAN 타임아웃·지터·재기동은 FAIL, 짧은 측정·송신 없음은 INCONCLUSIVE.

| 구분 | 항목 |
|---|---|
| 오프라인 검증 완료 | 판정 로직(단위 테스트 17), 모의 버스에서 base·fas 실행·수치 산출, 배포 스크립트 문법·로컬 단계(`--local-only`) |
| 미검증 | 젯슨 단계(md5·selftest·옵션 파일 쓰기·soc 게이트의 원격 실행) — 접속하지 않았다 |
| 현장 확인 필요 | 실버스(1 Mbps·실모터 응답 시간)에서 fas 대기 4 ms 안 완결률, 인수 홀드 상태의 실제 수치 |
