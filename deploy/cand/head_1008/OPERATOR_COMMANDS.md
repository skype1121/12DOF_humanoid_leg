# 운영 명령 정리 — head_1008 배포 뒤 (초안)

> 상태: **초안 · 실행 허가 문서가 아니다.** 젯슨 배포·서비스 변경·CAN 송신·ARM·토크 인가·영점 변경·실물 시험은 각각 승인 사항이다(`CLAUDE.md`).
> 근거: 코드(`jetson/scripts/operator_console.py`·`console_state.py`·`rl_bridge_node.py`·`w3_cmd.py`·`robotctl.py`, `rl_walking/deploy/w3_deploy.py`),
> `reports/gen_w3/REAL_TEST1_PLAN.md`(v1.1), `reports/gen_w3/ANOMALY_CARD.md`, `reports/W3_BRIDGE_FLAG_1006.md` §3·§9·§10, `reports/gen_w3/KEW3_DESIGN.md` §8.
> 표기: **[실물 미검증]** = 시뮬·오프라인 검증만 있음 · **[결정]** = 사용자 결정 항목 · **[추가 예정]** = 다른 작업에서 구현 중.

## 0. 먼저 알아 둘 것

- **걷기(T2 이상)와 충격형 밀기는 선행조건 P0 가 풀리기 전에는 보류 권고**(REAL_TEST1_PLAN v1.1): 회복 동작 중 발목 롤 드라이버(모터 6·12)가 동시에
  재시작한 사건이 4 세션에서 반복됐고, 시뮬의 보통 걷기 부하가 그 구간에 걸친다. P0 전에 할 수 있는 것 = 기동·관측 확인, W3 STAND 서기(T1), 지긋이 미는 회복 시험.
  재시작이 한 번이라도 나면 그 세션의 정책 시험을 멈추고 매달림에서 영점 확인(배수 보정 우선).
- 하드웨어맵은 `guard_mode = "data"` 다 — 리프트·발진·타깃 발진 가드가 **기록만 하고 정지하지 않는다**. 즉시 정지는 과속(12 rad/s)·센서 공백·비유한·estop 만 남는다.
  W3 세션의 '늦춘 동결'도 data 모드에서는 실행되지 않는다. safe 로 바꿀지는 [결정](계획서 D1). 맵을 바꾸면 **stage8 재기동 필수**.
- 이번 배포로 **기본 동작은 바뀌지 않는다**: 콘솔 드롭다운의 키·기본값, 브리지·stage8 기동 인자, 단계별 절차는 종전과 같다(근거는 `README.md` 의 동작 변화 표).
  새 기능은 전부 명시해야 켜진다 — 콘솔의 '정책 ON' 버튼, 브리지의 `--experimental-*`·`--w3-*`·`--c5-*` 인자, stage8 지연 옵션 파일.

## 1. 공통

| 일 | 명령·위치 |
|---|---|
| 콘솔 | `http://<젯슨>:8800` (systemd `operator-console.service`, 상시). 터미널에서는 `python3 ~/humanoid_repo/jetson/scripts/robotctl.py status` |
| 상태 한 줄 / 계속 보기 | `robotctl status` · `robotctl watch` |
| 준비(완료된 단계는 건너뜀) | `robotctl prep` = CAN 준비 → 센서 → 영점 확인 → 정책 계약 검사 → 인수 제어기 준비 |
| 정책·시험 방식 설정 | 콘솔 드롭다운 또는 `robotctl config air|ground [키]` (제어 시작 전) |
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

드롭다운 키(콘솔 `CKPT_CHOICES`, 첫 항목이 기본값):

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
   터미널: `robotctl prep` → `robotctl hang` → `robotctl level` → (착지) → `robotctl ground` → `robotctl handoff` → `robotctl stand`.
2. **'정책 ON(착지 자동 인수)' [실물 미검증 — 이번 배포로 처음 젯슨에 올라간다]**: 운용 게인(kp150)의 **공중** 수평 자세에서 한 번 누르면 콘솔이
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
- (a) 콘솔 통합 — **[추가 예정]**. 지금 HEAD 콘솔은 드롭다운에 W3 키가 없고 `--experimental-w3` 를 넘기지 않는다. 걷기 버튼(전진 한 걸음 등, `w3_deploy.W3_MOVES` 표)도 다른 작업에서 구현 중이다.
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
- 가드 트립은 KE 모드에서도 W3 세션 규칙(늦춘 동결)이다 — 종전 회복 세션의 즉시 동결과 다르다. `--c5-settle`·`--c5-imu-f32`·`--runner-ema-fresh 2` 는 합성 키에서 거부.
- 명령·응답·감독기·기동 주체는 세션 B 와 같다.

## 5. 걷기 명령 — `w3_cmd.py` (세션 B·C, 콘솔 걷기 버튼은 [추가 예정])

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
  콘솔 걷기 버튼 [추가 예정]이 쓴다. 손으로 보내려면 `{"cmd":"step","enable":true,"req_id":"…","seq":N,"vx":0.06,"dur_s":0.6,"heading_loop":false}` 를 한 번 발행(seq 는 직전보다 커야 한다).
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

배포 뒤 첫 세션 전 확인: 콘솔 준비 단계 '정책 계약 검사'가 제한(300 s) 안에 끝나는지 · 드롭다운 키·기본값 · 화면에 '정책 ON' 버튼이 생겼는지(쓰지 않아도 된다).
