# stand_v7 6차 무개입 실행 카드 (2026-08-08)

**이 세션이 답할 질문은 하나다 — v6b의 "1초부터 전방 단조 밀기"가 사라졌는가.**

**5차(v6b) 기준선 — 원본 블랙박스 재분석 확정치**
(`~/logs/rl_bridge_20260804_130927.jsonl`, 로컬 사본 `.scratch/bb/`):

| 지표 | v6b 실측 |
|---|---|
| 투입 시 발력 / 자세 | **96%** / −1.7° (완전 자립, 줄 무부하) |
| 하중 이탈 개시 | **1.12s — 그때 정책권한 alpha=0.84** |
| 실효 자립 | **1.4s** (발력 50% 미회복) |
| 초기 3s 전방 추세 | **+4.06°/s** (R²=0.77), 단조증가 64% |
| 순변화 | **−1.8° → +19.2°** |
| 가드 | lift_guard 8.5s (낙상 **후**, 원인 아님) |

읽는 법: 로봇은 발력 96%로 **혼자 서 있었고**, 정책 권한(블렌딩 alpha)이
0.84로 올라가는 순간 하중이 빠지기 시작했다. 발진도 스텝도 없는 단조 전방
누적 — **정책이 밀어낸 것**이지 착지나 균형의 문제가 아니다.
→ 원인 가설 = payload **뒤 배치**로 학습된 전방 과보상. v7은 그 위치 가정을
제거(com_range x −0.03~+0.02 중앙 대역)하고 재베이크한 판이다.

정책: **`stand_v7`** (2026-08-04_13-13-49 / model_63898, 보상 122.60 최고,
낙상 1.6%, 골반고 0.649 완전직립, 토크 1.52Nm 최저, 폭발 0건).
브리지는 `stand` 명령 시 자동으로 stand_v7 폴백 — `--checkpoint` 생략 가능하나
**debug 토픽에서 `checkpoint=stand_v7` 육안 확인은 필수**.

---

## 프로토콜 — 무개입 (5차에서 확립)
- **넘어질 때까지 손대지 않는다.** 손 개입 = 데이터 오염. 5차의 "9초 자립"은
  개입 순간 줄이 팽팽해진 산물이었다.
- 안전망은 **줄 + 브리지 가드**이지 사람 손이 아니다. 사람은 줄 옆 스탠바이.
- 판정은 **발력 총합 %** 1순위 (체중 117.7N 대비). IMU 스냅샷 1장으로
  "섰다"고 판정했다가 오판한 사고가 있었다 — 스냅샷 금지.

## ⚠ 젯슨 스크립트 경로 — 두 디렉토리로 갈려 있다 (0808 실측 확인)
| 스크립트 | 위치 |
|---|---|
| `boot_pose_check.py` `sensors_up.sh` `stack_down.sh` `release_all.py` `stage8_cmd.py` `wiggle_watch.py` `can_up.sh` | **`~/rl_calib/`** (여기에만 있음) |
| `rl_bridge_node.py` `stage8_12axis_mit_control_node.py` `analyze_fall.py` | **`~/humanoid_repo/jetson/scripts/`** (여기에만 있음) |
| `hang_stand.py` | 양쪽 동일 (md5 4b7a59a1) — 아무거나 |

`cd ~/humanoid_repo` 후 전부 실행하려 하면 절반이 "파일 없음"으로 막힌다.

## 0. 전원·게이트 (하나라도 실패 시 진행 금지)
```bash
bash ~/rl_calib/can_up.sh                       # 젯슨 부팅 후 1회 (sudo 필요)
(candump -ta can1 > /tmp/live_candump_$(date +%H%M).log 2>&1 &)   # 0단계
python3 ~/rl_calib/boot_pose_check.py --ref stand   # 12/12 (필수·60분 유효)
bash ~/rl_calib/sensors_up.sh                      # IMU·발압력 (멱등)
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --selftest   # ALL PASS
```
- **boot_pose_check는 매 전원사이클 필수** — AK 로터절대 엔코더가 36°/10°
  배수로 시프트한다. m6는 사이클마다 코인플립.
- **m8은 전원사이클 없이도 시프트한다** (5차 중 +36°, 오른분기 브라운아웃 추정).
  낙하·충격 후에는 전원을 안 껐어도 재판정.
- 12/12 아니면 배수 보정 후 재판정. **재캡처보다 배수 보정이 원칙.**
- 하드웨어맵 갱신했다면 **stage8 재기동 필수** (구맵 메모리 → 인수 거부 →
  브레이크 모드 방치 붕괴 사고 1건).

## 1~3. 기립 → 핸드오프 → 무장
`docs/march_v3_실행카드.md` 1~3단계와 동일. 요약:
```bash
python3 ~/humanoid_repo/jetson/scripts/stage8_12axis_mit_control_node.py \
        --can-channel can1 --wait-takeover
python3 ~/rl_calib/hang_stand.py              # 별도 터미널, 에어램프→kp150/kd5
# ── 착지: 매달린 발끝의 연직선을 조준, 평평하게, 천천히 ──
touch /tmp/hang_ground
touch /tmp/hang_stand_handoff                 # 접지+운용게인에서만
python3 ~/rl_calib/stage8_cmd.py SET_BASELINE_FROM_CURRENT_ALL   # 12/12
python3 ~/rl_calib/stage8_cmd.py ARM_ALL                         # 30초 내
```

### 착지가 세션 성패를 지배한다
아침 평평 착지 = 혼자 섬 / 밤 첫 착지 앞 23.5° = 자립 불가.
**혼자 설 때까지 착지를 다시 한다.** 기울어진 채로 정책을 투입하면 그 세션의
데이터는 정책이 아니라 착지 기하를 측정한 것이 된다.
트림은 **전부 0**이 정답 (0802 확정 — 뒤 5.7°는 트림 잔류의 산물이었다).

## 4. 정책 투입 → 무개입 관찰
```bash
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --checkpoint stand_v7
# 브리지가 실제로 떴는지 확인한 뒤에 발사 (-w 1 pub은 구독자 없으면 무한 대기)
ros2 topic pub --once -w 1 /humanoid/rl_bridge_command std_msgs/String \
  "data: '{\"cmd\": \"stand\"}'"
```
- debug 토픽에서 `checkpoint=stand_v7`, `blend_alpha`가 1.5초에 걸쳐 0→1 확인.
- **그 다음은 아무것도 하지 않는다.** 넘어지거나 가드가 걸릴 때까지 관찰.
- 관찰 항목(육안): 기울임 방향과 **가속하는지 vs 왕복하는지**. v6b는 왕복 없이
  단조로 밀렸다. 왕복(보정 시도)이 보이면 그 자체가 개선 신호다.

## 5. 판정 — 블랙박스 자동 분석
```bash
cd ~/humanoid_repo
python3 jetson/scripts/analyze_fall.py ~/logs/rl_bridge_<최신>.jsonl \
    --compare ~/logs/rl_bridge_20260804_130927.jsonl   # 5차 v6b 기준선
```
등급 기준 (스크립트가 자동 산출):

| 등급 | 의미 | 다음 행동 |
|---|---|---|
| **A** | 8초 이상 자립, 종료 기울임 5° 이내 | **march 사다리 진입** (5초 → 연장) |
| **B** | 단조 밀기는 사라짐, 자립은 짧음 | 무엇이 끝냈는지 규명 (가드/횡동요/발력) |
| **C** | 전방 단조 밀기 재현 = **payload 가설 기각** | 다음 용의자로 이동 (아래) |
| **D** | 후방 단조 = v7 과교정 | com_range를 v6b·v7 중간(x −0.05~0.0)으로 재베이크 |

## 6. 종료·회수 (순서 엄수)
```bash
# ① 줄 재인장 (사람)  ② 스택 종료  ③ 그 다음에만 릴리즈
bash ~/rl_calib/stack_down.sh
python3 ~/rl_calib/release_all.py
```
- **release_all을 stage8 armed 스트리밍과 동시 실행 금지** (버스 명령 충돌).
- 굽힌 관절 즉시 릴리즈 금지 — 자유낙하 회생으로 SMPS OVP 래치 사망 사례.
- 프로세스 종료는 **명시 PID로만** (pkill 금지).

---

## C급이 나왔을 때의 다음 용의자 (우선순위)
payload 위치가 원인이 아니라면, 전방 편향은 **몸이 아니라 파이프라인**에서 온다.
1. **관측 지연 +1틱** — stage8 폴링-선행 구조가 정책에 이미 한 틱 묵은 각도를
   준다. 측정 sysid에서 루프지연 4~8틱 확인됨. 학습 측 지연 분포 재정합.
2. **컴플라이언스(로터 사각)** — 기울기 5.7°가 관절오차 0.7°로만 보인다.
   정책은 실제보다 덜 기울어진 것으로 관측한다 → 보정 부족 → 전방 누적.
   관측을 로터각(SEA) 모델로 바꾸는 재학습이 정공법.
3. **발압력 10Hz 지연** — 접촉 관측이 늦어 체중 이동 판단이 밀린다. 50Hz 상향.
4. **qvel 유한차분 절단** — 정지 시 48~71%가 0으로 관측된다. 재설계 항목.

## 다음 단계 (A급 통과 시)
사다리 2단 = `{"cmd": "march"}` + `--checkpoint march_v5b` 5초 → estop →
자세·괴리 점검 → 이상 없으면 연장. march_v5b는 이미 젯슨 배포·하네스 PASS.
