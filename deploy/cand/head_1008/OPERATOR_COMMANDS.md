# 운영 명령 정리 — head_1008 배포 뒤 (초안)

> 상태: **초안 · 실행 허가 문서가 아니다.** 젯슨 배포·서비스 변경·CAN 송신·ARM·토크 인가·영점 변경·실물 시험은 각각 승인 사항이다(`CLAUDE.md`).
> 근거: 코드(`jetson/scripts/operator_console.py`·`console_state.py`·`rl_bridge_node.py`·`w3_cmd.py`·`robotctl.py`, `rl_walking/deploy/w3_deploy.py`),
> `reports/gen_w3/REAL_TEST1_PLAN.md`(v1.1), `reports/gen_w3/ANOMALY_CARD.md`, `reports/W3_BRIDGE_FLAG_1006.md` §3·§9·§10, `reports/gen_w3/KEW3_DESIGN.md` §8.
> 표기: **[실물 미검증]** = 시뮬·오프라인 검증만 있음 · **[결정]** = 사용자 결정 항목 · **[추가 예정]** = 다른 작업에서 구현 중.

## 0-1. 1008 변경(승윤님 지시) — 아래 본문보다 이 절이 우선한다

- **고를 수 있는 정책은 하나다**: `stand_kew3qa_g21a_g24ra_w3s3d15800` = 종전 합성(서기 KE + 걷기 W3 + 갇힘 해소) + **공중·착지 구간 정책(G21h2-13498)**. 콘솔 드롭다운·기본값이 이 키다. **[실물 미검증]**
  지면에서 시작하면(접지 방식) 종전 `stand_kew3q_…` 와 같은 동작이다(같은 입력열 600 틱에서 목표각 비트 동일 — `jetson_parity`).
  그 밖의 키(§2 표의 7 개, `stand_w3s3d_15800`, `stand_kew3_…`, `stand_kew3q_…`)는 **정리됐다** — 화면에 없고, `robotctl config … <키>`·API 로 줘도 서버가 "정리된 정책입니다"로 거부한다.
  브리지의 정책 등록과 정책 파일은 그대로 남아 있다(과거 기록 재생·selftest 용). 다시 올리려면 `operator_console.py` 의 `CKPT_MENU_KEYS` 에 키를 더한다.
- **`robotctl prep` 이 `hang` 을 흡수했다**: `robotctl prep` 한 번 = CAN 준비 → 센서 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 → **공중 자세 유지 → 운용 게인(kp150)**.
  모터가 움직이기 직전에 한 번 묻는다("줄이 로봇을 지지하고 다리 주변에 사람이 없나요?"). `-y` = 묻지 않음, `--no-hang` = 모터를 움직이지 않는 준비 단계까지만.
  완료된 단계는 건너뛴다. `robotctl hang` 은 같은 일을 하는 별칭으로 남아 있다.
- **인계는 공중에서 하고, 정책도 공중에서 바로 돈다(시험 방식 기본값 = air)** — 절차:
  1. `robotctl prep` — 공중 자세 유지·kp150 까지.
  2. `robotctl level` — 수평 자세.
  3. 콘솔에서 두 가지를 본다. ① **양발 접촉 비트가 둘 다 0**(떠 있는데 1 이 보이면 패드 문제 — 공중 방식으로 시작하지 않는다. 둘 다 1 이면 서버가 정책 ON 을 거부한다).
     ② **몸이 가만히 매달려 있고 fwd(피치)가 −3 ~ +6° 안**(시뮬로 확인한 범위. 밖이면 줄 길이를 고치거나 접지 방식으로).
  4. `robotctl policy-on` — 확인 질문 뒤 제어 인계·기준점·ARM·정책 로드·**정책 구동**까지 공중에서 끝난다. 정책이 켜지면 다리가 조금 움직인다(시뮬: 공중 관절 이동폭 4–6°).
  5. **곧바로 지그를 천천히 내린다.** 양발이 닿고 1–3 s 가라앉으면 러너가 서기 KE 로 스스로 넘긴다(한 번만).
     넘어가기 전에는 걷기 버튼이 "공중·착지 정책 구간" 사유로 막혀 있고, 넘어가면 그 사유가 사라진다.
  6. 줄을 다 풀고 모든 줄 느슨을 확인한 뒤 자립 타이머(성공 판정은 종전 그대로 — 운영자 확인·영상).
  7. 걷기·회전은 KE 인계 뒤 콘솔 걷기 패널에서(§4·§5).
- **접지 방식으로 되돌리기**: `robotctl config ground` — 홀드(자세 유지)로 내려 양발 착지 확인 뒤에 정책이 시작한다(§2 의 '정책 ON'). 이 키에서도 종전과 같은 동작이다. 공중 방식이 불안하면 이쪽을 쓴다.
- **공중 방식의 주의**
  - **공중 폭주 차단**: 매달린 동안(양발 무접촉 0.5 s 뒤) 정책의 목표가 기본자세에서 30° 넘게 벗어나면 러너가 정책을 끊고 다리를 기본자세로 천천히(50°/s) 되돌려 유지한다.
    걷기 버튼 사유와 `robotctl status` 에 "공중 폭주 차단 — 정지한 뒤 처음부터 다시 시작" 이 뜬다. 그 세션은 내려도 정책이 서지 않는다(자세 고정) — **줄을 유지한 채 정지·회수(§6)하고 새로 시작한다.**
    시뮬 1,800 env 에서 발동 0 이었다(평소 목표 이탈 4° 안팎). 발동했다면 그 자체가 이상 신호다 — 블랙박스(`kew3.air_dev`)·영상을 본 뒤 재시도한다.
  - 차단이 걸리지 않아도 공중에서 다리 움직임·몸 흔들림이 **커지는** 것이 보이면 내리지 말고 `robotctl estop` → 줄 유지 → 회수.
  - **AIR → KE 인계는 양발 접촉 비트와 몸통 각속도만으로 판정한다.** 한쪽 패드가 0 으로 죽어 있으면 넘어가지 않는다 — 서기는 하지만 회복 동작과 걷기가 없다.
    착지하고 5 s 가 지나도 걷기 버튼 사유가 "공중·착지 정책 구간" 이면 밀기·걷기 시험을 하지 말고 회수해 패드를 점검한다(특히 왼발 — 1005 22:03 세션 내내 0 이었다).
  - **KE 로 넘어간 뒤에는 다시 들어 올려도 공중 정책으로 돌아가지 않는다.** 다시 매달아야 하면 정지·회수 뒤 처음부터 새로 시작한다(지면 정책을 공중에서 돌리지 않는다).
  - 하드웨어맵이 `guard_mode = "data"` 라 행잉 상한(120 s)·리프트 가드는 **기록만** 한다(§0). 공중 폭주 차단은 러너 안에 있어 guard_mode 와 무관하게 동작한다.
  - 단계별 절차(인계 → 기준점 → ARM → 브리지 → stand)를 손으로 밟아도 방식이 air 면 마지막 stand 가 공중 정책 시작이다.
- **근거와 한계**: `reports/gen_w3/AIR_START_1008.md`. 지면 시작 정책을 공중에서 돌리면 착지 뒤 39–62 % 가 넘어졌다(AIR1). 매달림을 넣어 학습한 첫 공중 정책(G21h-12299)은 시뮬 기본 조건에서는 섰지만
  매달림 피치가 2° 만 달라져도 공중에서 다리를 20–40° 움직였고 12 % 가 착지 뒤 인계에 실패해 버렸다(AIR4). 기운 채 매달림(±8°)·긴 매달림·공중 자세 유지 보상으로 다시 학습한 G21h2 가 지금의 공중 정책이다.
  시뮬 판정(AIR5, 사전 등록 기준): 수평에 가깝게 매달린 5 조건 1,280 개체 낙상 0·KE 인계 100 %·걷기 버튼 수락 100 %, 기운 채 매달린(피치 −2.9 ~ +6.2°) 512 개체 낙상 0·인계 99.8 %·수락 99.4 %,
  공중 관절 이동폭 p90 4–8°, 공중 폭주 차단 발동 0 / 1,792. 매달림 20 s 까지 확인(60 s 는 앞 세대 정책으로만 확인).
  시뮬 성공률은 실물 성공 확률이 아니다. 줄·지그 모형은 가정이고(줄의 감쇠·비틀림·지그 기하 실측 없음), 공중에서 사람·지그에 닿는 경우와 줄이 걸리는 경우는 시뮬에 없다.
  실물 매달림 피치 범위(−1 ~ +5°)는 젯슨 세션 기록 네 개에서 읽은 것이다.

## 0. 먼저 알아 둘 것

- **걷기(T2 이상)와 충격형 밀기는 선행조건 P0 가 풀리기 전에는 보류 권고**(REAL_TEST1_PLAN v1.1): 회복 동작 중 발목 롤 드라이버(모터 6·12)가 동시에
  재시작한 사건이 4 세션에서 반복됐고, 시뮬의 보통 걷기 부하가 그 구간에 걸친다. P0 전에 할 수 있는 것 = 기동·관측 확인, W3 STAND 서기(T1), 지긋이 미는 회복 시험.
  재시작이 한 번이라도 나면 그 세션의 정책 시험을 멈추고 매달림에서 영점 확인(배수 보정 우선).
- 하드웨어맵은 `guard_mode = "data"` 다 — 리프트·발진·타깃 발진 가드가 **기록만 하고 정지하지 않는다**. 즉시 정지는 과속(12 rad/s)·센서 공백·비유한·estop 만 남는다.
  W3 세션의 '늦춘 동결'도 data 모드에서는 실행되지 않는다. safe 로 바꿀지는 [결정](계획서 D1). 맵을 바꾸면 **stage8 재기동 필수**.
- 이번 배포로 **기본 동작은 바뀌지 않는다**: 콘솔 드롭다운의 기본값과 기존 키 7 개(뒤에 걷기 실험 키 2 개가 **추가**된다), 기존 키의 브리지 기동 인자, stage8 기동 인자, 단계별 절차는 종전과 같다(근거는 `README.md` 의 동작 변화 표).
  새 기능은 전부 명시해야 켜진다 — 콘솔의 '정책 ON' 버튼, 드롭다운에서 걷기 실험 키를 고른 시험의 걷기 패널, 브리지의 `--experimental-*`·`--w3-*`·`--c5-*` 인자, stage8 지연 옵션 파일.

## 1. 공통

| 일 | 명령·위치 |
|---|---|
| 콘솔 | `http://<젯슨>:8800` (systemd `operator-console.service`, 상시). 터미널에서는 `python3 ~/humanoid_repo/jetson/scripts/robotctl.py status` |
| 상태 한 줄 / 계속 보기 | `robotctl status` · `robotctl watch` |
| 준비(완료된 단계는 건너뜀) | `robotctl prep [-y] [--no-hang]` = CAN 준비 → 센서 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 → 공중 자세 유지 → 운용 게인(1008: `hang` 흡수 — 모터가 움직이기 전 한 번 묻는다) |
| 정책·시험 방식 설정 | 정책은 하나라 고를 것이 없다(§0-1). `robotctl config ground` 만 받는다 — `air` 는 남긴 정책에서 거부 |
| 비상정지 | 콘솔 **정지 요청**(항상 눌린다 — 브리지 estop + stage8 STOP_ALL 동시) 또는 `robotctl estop` |
| 기록 | 세션 `~/logs/sessions/session_*/timeline.jsonl`(20 Hz) · 브리지 블랙박스 `~/logs/rl_bridge_*.jsonl`(live 일 때) · 영상 2 대 |
| 사후 판정 | `python3 jetson/scripts/analyze_fall.py <블랙박스>` |

콘솔이 띄우는 프로세스(코드 그대로):

```
stage8 : python3 ~/humanoid_repo/jetson/scripts/stage8_12axis_mit_control_node.py --can-channel can1 --wait-takeover [+ 지연 옵션 — §8, 기본 없음]
브리지 : python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --checkpoint <드롭다운 키>
검사   : python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --selftest --checkpoint <드롭다운 키>     (준비 단계 '정책 계약 검사', 제한 300 s)
~/rl_calib 에서: can_up.sh · sensors_up.sh · boot_pose_check.py --ref stand · hang_stand.py · stage8_cmd.py · release_all.py   (이번 배포 대상 아님)
```

## 2. 세션 A — 서기·회복 (콘솔에서, 종전과 같음)

**1008: 아래 표의 키는 전부 정리됐다(고를 수 없음 — §0-1).** 기록을 위해 남겨 둔다. 지금의 서기·회복은 합성 키의 KE 모드가 맡는다(§4).

드롭다운 키(1008 전의 콘솔 `CKPT_CHOICES`):

| 키 | 내용 | 비고 |
|---|---|---|
| `stand_crouch_g12` | 서기 G12-9500(v2) | 기본값 |
| `stand_crouch_g21a` | 서기 G21a-11300 | 1003 실물 1 회 |
| `stand_c5_g21a_g24ra` | 서기 G21a + 회복 G24Ra, 감지기 RV4 | 구조 진단 |
| `stand_c5kr_g21a_g24ra` | KE = 위 + cue·조기 복귀(유지 0.2 s)·불응기 | 1005 배포 |
| `stand_c5krh_g21a_g24ra` | KE 에서 조기 복귀 유지만 0.7 s | 1007 배포, 승격 아님 |
| `stand_crouch_g12_ema0` · `stand_crouch_g11a` | 대조용 | |

절차는 두 가지다. **어느 쪽이든 버튼 클릭이 아니라 화면의 실제 응답으로 진행을 판단한다.**

1. 단계별(종전): CAN 준비 → 센서 기동 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 → 공중 자세 유지 시작 → 운용 게인으로 전환 →
   접지 확인 → 제어 인계 → 기준점 확인 → 제어기 ARM → 선택 정책 로드 → 정책 제어 시작.
   터미널: `robotctl prep`(1008: 공중 자세 유지·운용 게인까지 포함) → `robotctl level` → (착지) → `robotctl ground` → `robotctl handoff` → `robotctl stand`.
2. **'정책 ON(착지 자동 인수)' [실물 미검증] — 1008: 이것이 표준 절차다(공중에서 인계)**: 운용 게인(kp150)의 **공중** 수평 자세에서 한 번 누르면 콘솔이
   접지 마커 → 인계 → 기준점 → ARM → 정책 로드 → `stand`(`touchdown_start`)를 기존 게이트·응답 확인으로 순서대로 실행한다(하나라도 실패하면 멈추고 고장 래치).
   브리지는 **양발 접촉 0.3 s + 정착(|ω_xy| < 0.3 rad/s 0.1 s, 상한 1.0 s) 전에는 정책을 계산하지 않고** stage8 이 내던 수평 홀드 명령을 그대로 보낸다.
   화면이 "착지 대기 · 수평 자세 홀드(정책 미구동)"를 표시한 뒤에 지그를 내린다. 터미널: `robotctl policy-on`.
   - 거부되는 경우: 준비 단계 미완 · 단계별 절차를 이미 시작함 · 양발이 이미 접촉 중 · 에어 보행 중 · `~/rl_calib/level_pose.json` 의 12 관절 수평 기준 자세가
     없거나 지금 자세와 5° 넘게 다름 · 개별 조정 램프 진행 중.
   - 착지 대기 상한 120 s 를 넘기거나 홀드 자세가 정책 기본자세와 8° 넘게 다르면 STOP_ALL(같은 홀드 동결) 뒤 정책 비활성 — 회수 후 새 시험.
   - 자동 진행 중에는 다른 조작이 막힌다(정지 요청·운영자 이벤트·줄 지지 확인은 가능).

회복 세션 옵션(브리지 인자, **콘솔이 넘기지 않는다** — 쓰려면 콘솔 통합 또는 손 기동[§3 주의]이 필요, 전부 [실물 미검증]):
`--c5-settle odic|odicw`(종료 판정을 다리 오도메트리로) · `--c5-overtime note:초|freeze:초`(회복 모드 체류 상한) · `--c5-imu-f32` · `--runner-ema-fresh 2`.

자립 판정: 모든 줄 느슨 + 외부 보조 없음으로 5 s(별도 집계 30 s). 타이머는 운영자의 **모든 줄 느슨·무보조 확인** 이벤트 뒤부터. 손 보조·재인장·가드 정지가 들어간 시도는 무효.

## 3. 세션 B — W3 단독(걷기·서기) [실물 미검증]

브리지 인자:

```
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --experimental-w3 --checkpoint stand_w3s3d_15800 \
        --w3-heading-loop off [--w3-monitor note|on] [--w3-hl-fault on] [--w3-safety '{…}'] [--w3-cmd-limits '{…}']
```

- 정책: W3S3d-15800 한 개가 서기(걷기 끔)·제자리·앞뒤·옆·제자리 회전을 맡는다. 시작은 항상 `stand`(서기), 걷기는 `step` 명령으로 켠다. `hang_arm` 시작은 거부(지면 시작만 학습).
- `--w3-heading-loop off` 를 적는 이유: 1007 22:14 HD1 판정 불합격으로 헤딩 고리를 끄기로 했다(`reports/night_1007/LOG.md`). 인자를 빼면 고리가 **켜진다**
  (K 0.5·±0.4 rad/s·오차 15°·120 s) — 켜기 명령마다 `--no-heading-loop` 를 붙여도 된다.
- `--w3-monitor note` = 걷기 감시 판정을 기록만(오트립 계수용), `on` = 조치(걷기 정지·켜기 거부·고리 금지). 계획서 기준으로 첫 시험은 note 부터.
- W3 세션에는 회복 정책이 없다 — 걷는 중 밀기 금지, 줄 운영자 상시 대기.

**기동 주체 [결정](계획서 D4)**
- (a) 콘솔 걷기 패널 — **1007 23:53 병합(42da0c21) [실물 미검증]**. 아래는 코드(`operator_console.py`·`console_state.py`)에서 읽은 것이다.
  - 드롭다운 **뒤쪽**에 걷기 실험 키 2 개가 생긴다: `stand_w3s3d_15800`(W3 단독)·`stand_kew3_g21a_g24ra_w3s3d15800`(서기 KE + 걷기 W3 합성). 기본값과 기존 키 7 개,
    그 키들의 브리지 기동 인자는 그대로다(콘솔 selftest 가 고정). 걷기 모듈을 불러오지 못하면 걷기 키는 목록에 올라오지 않는다.
    **1008 추가(이 묶음이 다시 배포된 뒤에 보인다)**: 맨 뒤에 `stand_kew3q_g21a_g24ra_w3s3d15800`(서기 KE + 걷기 W3 합성 + 갇힘 해소 — 시뮬 평가 구성 · 실물 미검증).
    브리지 기동 인자는 첫 합성 키와 같다. 무엇이 다른지는 §4.
  - 걷기 키를 고른 시험에서 콘솔이 띄우는 브리지: `--live --checkpoint <키> --experimental-w3 [--experimental-kew3] --w3-heading-loop off`.
    준비 단계의 계약 검사는 `--selftest --checkpoint <키> --experimental-w3 [--experimental-kew3]`(헤딩 고리 인자 없이).
    **콘솔은 `--w3-monitor`·`--w3-hl-fault`·`--w3-safety`·`--w3-cmd-limits`·`--c5-overtime` 를 넘기지 않는다** → 걷기 감시는 꺼진 채(기록 `note` 도 없음) 돈다.
    계획서가 "첫 시험은 note 부터"라고 한 것과 다르다 — 감시 기록이 필요하면 콘솔 쪽 인자 추가가 먼저다 **[결정]**.
  - 패널: 시간 제한 걷기 버튼 6 개 — `fwd1` 전진 한 걸음(vx 0.06·0.6 s)·`fwd3` 전진 세 걸음(vx 0.06·2.4 s)·`left1` 왼쪽 옆걸음(vy 0.035·0.6 s)·
    `turn_l`/`turn_r` 좌·우 회전(wz ±0.15·2.0 s)·`inplace` 제자리 걸음(2.0 s) + **비활성 2 개**(`back1`·`right1` — 화면에 사유 표시, 서버가 409 로 거부).
    값의 단일 원본은 `w3_deploy.W3_MOVES`. 버튼 옆 '시뮬 +4.8 cm' 등은 X14 시뮬 중앙값(실물 미확인)이고 명령 값이 아니다. 그 밖에 걷기 끄기·정지 래치·재무장.
  - 서버가 걷기 버튼을 받는 조건(화면을 우회한 호출도 같다): 걷기 키로 시작한 시험 · 정책 제어 시작 확인 뒤 · 브리지가 콘솔 추적 아래 live·ARM · 실제 로드된 정책 = 선택한 키 ·
    브리지가 걷기 상태(w3·w3_sup, 합성은 kew3 도)를 보냄 · 앞 버튼·명령이 끝남 · 가드 동결·정지 래치 없음 · 지금 걷는 중 아님 · (합성) KE 모드·차단 없음 · 양발 접촉 ·
    온셋 블렌딩 끝 · 조용한 양발 서기가 브리지 켜기 조건만큼 이어짐 · 요청 나이 5 s 이내. **켜기의 최종 판정은 브리지 감독기**다(§5 의 선행조건).
    걷기 끄기·정지 래치는 단계·고장 래치·앞 요청과 무관하게 받는다. 응답을 2 s 안에 못 받으면 '응답 없음'으로 두고 **같은 명령을 다시 보내지 않는다**.
  - 버튼이 '수락됨'이어도 걷기 완료가 아니다 — 화면의 '서기 복귀 확인됨'(디버그 토픽의 want 0·ge 0, 합성은 KE 모드)까지 본다. 콘솔은 끄기 타이머를 갖지 않는다(기한은 브리지가 센다).
  - 명령 순번 파일 `~/.w3_cmd_seq` 를 CLI `w3_cmd.py` 와 같이 쓴다. 패널과 CLI 를 한 세션에서 섞어 쓸 때의 화면 표시는 확인하지 않았다 — 한 세션에서는 한쪽만 쓴다.
- (b) 손 기동 — 권고하지 않는다. 콘솔은 자기가 띄우지 않은 `rl_bridge_node.py` 를 **'콘솔 추적 밖 제어기'** 로 보고 이후 단계 버튼을 거부하며, 스택 종료도 그 프로세스를 내리지 않는다.
  그래도 한다면: 콘솔로 **제어기 ARM** 까지 진행 → 터미널에서 위 명령으로 브리지를 띄우고 **PID 를 적는다** →
  `ros2 topic pub --once /humanoid/rl_bridge_command std_msgs/msg/String "{data: '{\"cmd\": \"stand\"}'}"` → §5 의 `w3_cmd.py` →
  끝낼 때는 줄 인장 뒤 그 **명시 PID** 에 `kill -INT <PID>`(`pkill` 금지) → 콘솔 스택 종료 → 그 다음에만 토크 해제. 비상정지는 콘솔 버튼이 그대로 듣는다.

## 4. 세션 C — KE ↔ W3 합성 [실물 미검증]

```
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --experimental-w3 --experimental-kew3 \
        --checkpoint stand_kew3_g21a_g24ra_w3s3d15800 --w3-heading-loop off [--w3-monitor note|on] [--c5-overtime note:6]
```

- 서 있는 동안은 KE(`stand_c5kr_g21a_g24ra` = 서기 G21a + 회복 G24Ra + 감지기, 조기 복귀 유지 0.2 s — **krh 아님**), 걷는 동안만 W3S3d-15800.
- 시작 = `stand`(KE 모드, 종전 인계 워밍업·블렌딩). 걷기 켜기 = `step` 켜기 → 다음 정책 틱에 W3. 끄기 = 반주기 경계에서 걸음 끔 → W3 서기로 조용한 양발 서기 0.5 s
  (|ω_xy| < 0.15, 상한 3 s) → KE 가 이어받음.
- **KE 회복 모드가 켜져 있는 동안 걷기 켜기는 거부된다**(응답 사유에 표시). 회복 모드가 풀리지 않으면 운영자가 풀 수단은 재시작뿐이다.
- **둘째 합성 키 `stand_kew3q_g21a_g24ra_w3s3d15800`(1008) [실물 미검증]** — 위 명령에서 `--checkpoint` 만 이 키로 바꾼다(플래그 같음).
  첫 합성 키와 정책 3 개·전환 규칙이 같고, **KE 감지기의 갇힘 해소만 켠다**: 회복 모드 중 |ω_xy| < 0.3 rad/s·양발 접촉이 2.0 s 이어지면 기준 자세를 지금 자세로
  다시 잡고 서기 모드로 돌아간다(블랙박스·디버그 `c5.requiet` = 발동 횟수 n·진행 run). **시뮬 합성 평가의 판정 구성(X6b·X7·X14 …, `--kw_requiet 2.0,0.3`)이 이 구성이다.**
  첫 합성 키는 이 옵션이 **꺼져 있다** — 시뮬에서 평가한 구성이 아니고, 조용히 선 채 회복 모드에 갇히면 위 항목대로 걷기 켜기가 계속 거부된다.
  한쪽 발 패드가 0 을 내는 고장에서는 갇힘 해소가 발동하지 않는다(양발 접촉 요구). 어느 키를 쓸지는 [결정]. 시뮬 성공률은 실물 성공 확률이 아니다.
- 가드 트립은 KE 모드에서도 W3 세션 규칙(늦춘 동결)이다 — 종전 회복 세션의 즉시 동결과 다르다. `--c5-settle`·`--c5-imu-f32`·`--runner-ema-fresh 2` 는 합성 키에서 거부.
- 명령·응답·감독기·기동 주체는 세션 B 와 같다.

## 5. 걷기 명령 — `w3_cmd.py` (세션 B·C — 콘솔 걷기 패널을 쓰지 않을 때의 CLI. 패널의 버튼도 같은 `step` 명령을 보낸다)

```
cd ~/humanoid_repo && source /opt/ros/humble/setup.bash
python3 jetson/scripts/w3_cmd.py on                      # 제자리 걷기 켜기(명령 0)
python3 jetson/scripts/w3_cmd.py on --vx 0.04            # 앞.  뒤는 --vx -0.03
python3 jetson/scripts/w3_cmd.py on --vy 0.03            # 옆(+ = 왼쪽).  --wz 0.1 = 제자리 회전(+ = 왼쪽)
python3 jetson/scripts/w3_cmd.py on --vx 0.04 --no-heading-loop
python3 jetson/scripts/w3_cmd.py off                     # 다음 반주기 경계에서 서기(정책 계속)
python3 jetson/scripts/w3_cmd.py stop                    # 서기 + 정지 래치
python3 jetson/scripts/w3_cmd.py rearm                   # 래치 해제(켜지는 않는다) — 래치 사유를 없앤 뒤에만
python3 jetson/scripts/w3_cmd.py status                  # 디버그 토픽 한 건: w3 · w3_cmd · w3_sup · w3_mon
```

- 한 번만 발행하고 같은 `req_id` 의 응답(`/humanoid/rl_bridge_ack`)을 기다린다. 종료 코드: 0 수락 · 1 거부 · 2 인자/로컬 검사 거부 · 3 응답 없음 · 4 ROS 오류.
  **응답이 없으면 같은 명령을 다시 보내지 않는다** — `status` 로 실제 상태를 본 뒤 판단. `ros2 topic pub -r` 반복 발행 금지.
- 명령 범위(학습 범위만, 한 번에 한 축): 앞 vx 0.03–0.10 · 뒤 0.03–0.06 · 옆 |vy| 0.03–0.04 · 회전 |wz| 0.1–0.2 [m/s, rad/s]. 0 은 제자리. 걷는 중 변경은 1 s 간격 이상.
  부호(+vx 앞·+vy 왼·+wz 왼쪽 회전)의 실물 확인은 계획서 T4–T6 에서 한다(현장 확인 필요).
- 켜기 선행조건(브리지 감독기): 정책 실행 중 · 인계 워밍업·블렌딩(1.5 s) 완료 · 정지 래치 없음 · 직전 끄기 완료 · 관측 0.1 s 안 · 양발 접촉 연속 1.0 s ·
  기울기 ≤ 7° · |ω_xy| < 0.25 rad/s 연속 1.0 s. 합성 세션은 + KE 회복 모드 꺼짐. 문턱은 [가정값] — 성공률을 위해 완화하지 않는다.
- 리드 발: 명령 0·앞뒤 = 왼발 먼저. 오른쪽 옆(vy < −0.02), 또는 옆 명령 없이 오른쪽 회전(wz < −0.08) = 오른발 먼저.
- 시간 제한 걷기: step 켜기에 `dur_s`(0.3–10 s)를 주면 그 시간 뒤 **브리지가 스스로** 끈다(응답 `event: burst_end`). `w3_cmd.py` 에는 이 인자가 없다 —
  콘솔 걷기 버튼이 쓴다. 손으로 보내려면 `{"cmd":"step","enable":true,"req_id":"…","seq":N,"vx":0.06,"dur_s":0.6,"heading_loop":false}` 를 한 번 발행(seq 는 직전보다 커야 한다).
- 버튼 표(`w3_deploy.W3_MOVES`)의 `back1`(후진 한 걸음)·`right1`(오른쪽 옆걸음)은 **비활성**이다 — X14 시뮬에서 방향 부호 일치 80 %·63 %(기준 95 %) 미달.
  버튼 경로(`w3_move_cmd`)는 이 둘을 거부한다. 위의 연속 명령(`w3_cmd.py on --vx -0.03`·`--vy -0.03`)은 이 표를 거치지 않아 막히지 않는다 —
  뒤·오른쪽 **한 걸음**은 방향이 시뮬에서도 불확실했다는 뜻으로 읽고, 실물에서 먼저 시도하지 않는다. [실물 미검증]
- `stand`·`walk`·`march`·`turn` 은 W3 세션에서 정책 실행 중 거부된다. 비상정지는 이 도구에 없다 — 콘솔 버튼.

명령의 뜻(ANOMALY_CARD): `step enable:false`(= `off`) 다음 반주기에 서기 · `stop` 서기 + 정지 래치 · `estop` 즉시 동결(토크 유지) + 래치 · `release_all` 토크 해제.

| 응답·징후 | 조치 |
|---|---|
| 거부 "정지 래치(…)" | 사유(자세·접촉·지연·센서)를 읽고 없앤 뒤에만 `rearm` + 새 켜기 |
| 거부 "감시: stage8 명령 처리 나이 …"·"관절 피드백 나이 …" | 걷기 중단 → 회수 → `latency_check`(§8) |
| 거부 "러너 선행조건: …"(합성) | KE 회복 모드가 켜져 있다 — 서기 복귀를 기다린다 |
| 응답 없음 | 재전송 금지, `status` 확인 → 필요하면 `stop` |
| 걷는 중 크게 흔들림 | `stop` + 줄 준비. 넘어가는 중이면 **estop + 줄 즉시 인장** |

## 6. 정지·회수 순서 (언제나, 양보 불가)

① 줄 인장(운영자 확인 — 콘솔 **① 줄 지지 확인**) → ② 콘솔 **② 제어 프로세스 종료**(콘솔이 띄운 것만, 명시 PID — `pkill` 금지) → ③ **그 다음에만** **③ 토크 해제**.
①–③ 동시 실행 금지(같은 버스에 상반 명령). 굽힌 관절 즉시 릴리즈 금지(회생 과전압 이력). 터미널: `robotctl down`.
낙하·큰 충격·전원 깜빡임 뒤에는 다음 시도 전에 매달림에서 영점 확인(`boot_pose_check`) — 영점 쓰기는 별도 승인, 배수 보정 우선(`zero_anchor.py`).
넘어지는 중이면 회수가 데이터보다 우선이다. 보조가 들어간 시도는 무효로 기록한다.

## 7. selftest (젯슨, 전부 오프라인 — CAN·ROS 노드를 쓰지 않는다)

```
cd ~/humanoid_repo
python3 rl_walking/deploy/policy_runner_stage4.py --selftest          # 마지막 줄 selftest: ALL PASS (모델 추론 줄에 백엔드=onnx)
python3 rl_walking/deploy/c5_switch.py --selftest                     # C5 selftest PASS
python3 rl_walking/deploy/kew3_runner.py --selftest                   # … ALL PASS
python3 jetson/scripts/w3_cmd.py --selftest                           # [PASS] w3_cmd …
python3 jetson/scripts/latency_check.py --selftest
python3 jetson/scripts/operator_console.py --selftest                 # console selftest: ALL PASS
python3 jetson/scripts/rl_bridge_node.py --selftest [--checkpoint <키>]
python3 jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 [--checkpoint stand_w3s3d_15800]
python3 jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint stand_kew3_g21a_g24ra_w3s3d15800
python3 jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint stand_kew3q_g21a_g24ra_w3s3d15800   # 1008 둘째 합성 키(갇힘 해소)
python3 deploy/cand/head_1008/selftest_w3_node_refs.py                # w3 node selftest: ALL PASS
LC_ALL=C md5sum -c --quiet deploy/head_1008.md5 && echo MD5_OK        # 배포된 파일 전체 대조
```

- 젯슨의 `jetson/scripts/selftest_w3_bridge_node.py` 를 **직접** 돌리면 기준 커밋 비교 3 건이 FAIL 로 나온다(젯슨 사본에는 git 이 없다). 젯슨에서는 위의 실행기를 쓴다.
- selftest 가 도는 동안 콘솔은 '콘솔 추적 밖 제어기 감지'를 잠깐 표시할 수 있다(명령행에 `rl_bridge_node.py` 가 보이기 때문) — 끝나면 사라진다. 그동안 콘솔 조작을 하지 않는다.
- 배포 스크립트가 같은 묶음을 자동으로 돌린다: `deploy/deploy_head_1008.sh --verify`(개발 PC 에서).

## 8. stage8 지연 단축 옵션 — 이번 배포로 켜지지 않는다

코드는 올라가지만 옵션 파일 `~/humanoid_repo/config/stage8_latency_opts` 가 없거나 비어 있으면 stage8 기동 인자는 종전 그대로다(환경변수 `STAGE8_LATENCY_OPTS` 가 있으면 그것이 우선).
켜는 절차는 `reports/LATENCY_STAGE_1006.md` 와 `deploy/deploy_lat_1006.sh --stage off|fas|soc` 다(별도 승인). 한 번에 한 단계, 정책 시험과 섞지 않는다.
판정: 들어 올린 무부하 인수 홀드에서 `python3 jetson/scripts/latency_check.py --expect off|fas --duration 60`.
※ `deploy_lat_1006.sh --apply` 는 쓰지 않는다 — 그 매니페스트(`lat_1006.md5`)의 콘솔은 HEAD 보다 옛 판이다. 파일은 이 배포가 이미 올렸으므로 `--stage` 만 쓴다.

## 9. 배포·되돌리기 (개발 PC 에서, 승인 뒤)

```
python3 deploy/make_head_manifest.py                 # 실행 파일이 바뀌었으면 매니페스트 재생성
deploy/test_head_1008_local.sh                       # 가짜 젯슨 왕복 시험(젯슨 접속 없음)
deploy/deploy_head_1008.sh                           # dry-run — 젯슨에 쓰지 않는다 (--probe-selftest-time: 현재 selftest 시간도 잰다)
deploy/deploy_head_1008.sh --apply                   # 백업 → 복사 → md5 → 젯슨 selftest(실패 시 자동 롤백). 콘솔은 재기동하지 않는다
deploy/deploy_head_1008.sh --restart-console         # 별도 단계(sudo systemctl restart operator-console). 콘솔 유휴·제어 프로세스 없음·selftest 통과 기록이 있을 때만
deploy/deploy_head_1008.sh --verify                  # 복사 없이 상태 확인 + selftest
deploy/deploy_head_1008.sh --rollback [--restart-console]
deploy/deploy_head_1008.sh --add-policy=logs/rsl_rl/<실험>/<run>      # 정책 폴더 하나만 추가(기존 파일은 덮어쓰지 않음, md5 기록, 젯슨 추론 확인)
deploy/deploy_head_1008.sh --remove-policy=logs/rsl_rl/<실험>/<run>   # 그렇게 추가한 폴더만 되돌림
```

백업은 젯슨 `~/humanoid_repo_bak_1008/`(원본 `files/`, `BACKUP_MANIFEST.md5`, 새 파일 목록 `CREATED.txt`, 이력 `STATE`, `selftest_*.log`, 정책 추가 기록 `POLICY_ADDED.md5`).
묶음에 `reports/night_1007/jetson_parity.py` 와 `reports/night_1007/jetson/parity_ref.npz` 가 있으면 젯슨 selftest 끝에
`python3 reports/night_1007/jetson_parity.py --check reports/night_1007/jetson/parity_ref.npz`(로컬 torch 기준 ↔ 젯슨 onnxruntime 러너의 목표각·모드 대조)도 돈다.

기록에 새로 생기는 것(1007 모터 오류 기록 — 기록 전용, 제어·가드·발행 불변): 브리지 블랙박스 틱 줄에 `fb_age_max_s`·`fb_age_max_joint`(항상),
`err_raw`·`s8_warn`·`fb_age_over`·`errlog_fail`(있을 때만) · 세션 타임라인 줄의 `joints[관절]` 에 `feedback_error_raw`(여러 값이면 `feedback_error_seen` 도)·줄에 `stage8_warnings`·`stage8_errlog_fail`, 그리고 stage8 상태 수신이
끊긴 첫 줄에 `stage8_nomsg: 1`·다시 받은 첫 줄에 `0`. **`stage8_nomsg: 1` 과 `0` 사이 줄에 오류 필드가 없는 것은 '오류 없음'이 아니라 '모름'이다.**

배포 뒤 첫 세션 전 확인: 콘솔 준비 단계 '정책 계약 검사'가 제한(300 s) 안에 끝나는지 · 드롭다운 키·기본값 · 화면에 '정책 ON' 버튼이 생겼는지(쓰지 않아도 된다) · 드롭다운 뒤쪽에 걷기 실험 키 2 개와 걷기 패널이 보이는지(걷기 키를 고르지 않으면 패널 버튼은 전부 거부된다).
