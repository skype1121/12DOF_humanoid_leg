# stand_v18b 7차 무개입 실행 카드 (2026-09-06 04:30 갱신판 — 0905 21:30 확정판에 배포 계약 감사 A/B/C·h3/h3g 재판정·정착 실험 반영)

**이 세션이 답할 질문 — 실질량 17.62kg·패드 접촉 자산으로 학습한 첫 세대가 실물에서
(1) 어느 방향으로, (2) 어떤 토크로, (3) 몇 초 만에 무너지는가.** "서는가" 는 부차.
시뮬 배포조건(구래퍼 com+16, V6 고정환경) 낙상 v18b 14.1% — 실물 오차 축을 다 켠 자기환경에선
36%(0906 h3 배포 경로 정합 판정) 이므로 **첫 투입에서 서지 못해도 데이터는 유효**하다. 판정은 블랙박스로만.

**정책 후보 (h3/h3g 재판정 0906 — `reports/README.md` 표, `reports/0905_감사_정정_보고.md` §5)**

| 순위 | 정책 | 근거 | 우려 |
|---|---|---|---|
| 1 | **`stand_v18b`** (run 2026-09-05_19-37-39 / model_120998, md5 fac90eb5…) | **실질량 17.62kg·정정 관측으로 학습된 유일한 후보**(v15 계보). h3(0906 배포 경로 정합 판정) V6 고정환경 구com+16 **14.1%**(v17 15.6%), 명목(지연 4틱) E_smth **0.022**·τ피크/한계 39%, 발 요차 3.9° | 자기환경 배포조건 36%(h3), 배포조건 낙상 방향 **뒤** 우세(V6 뒤5·앞2, 자기환경 뒤13·앞9) |
| 대조 | `stand_v17` (2026-09-04_21-48-32 / model_126997, md5 4dc9fd58…) | h3 V6 고정환경 구com+16 15.6%(v18b 와 동률권), 명목 τ피크 38%·E_smth 0.026, slip 0.9% | **19~21kg 로 학습**(link_mass_calib 상속) — 실물보다 무거운 몸을 기대 |
| 예비 | `stand_v15` (16-43-02 / model_114999) | 자기환경 배포조건 최강 | 발 요차 24° 오리발 |

6차(v7) 카드와 달라진 전제: **체중 172.9N**(저울 17.62kg). 발압력 어댑터 npc 를 0.01804→**0.02650** 으로 재보정
(0906 — 0802 자립 카운트 앵커 6525 = 172.9N). 백분율은 카운트 앵커 대비라 **4차 "90% 자립" 은 유효**
(0905 판의 "실제 61%" 는 분모만 바꾼 오류 — 철회). 블랙박스에 `tau_est_nm`·`temp_raw` 신설 →
`analyze_fall.py` 가 관절별 토크 피크/포화(문턱 95%) 표를 **자립 구간만** 집계해 시뮬 명목 판정과 같은 축으로 낸다.

---

## 프로토콜 — 무개입 (5차에서 확립, 변경 없음)
- **넘어질 때까지 손대지 않는다.** 손 개입 = 데이터 오염. 안전망은 **줄 + 브리지 가드**.
- 판정은 **발력 총합 %** (172.9N 대비) 1순위. IMU 스냅샷 1장 판정 금지.
- 낙하 후에는 전원을 안 껐어도 `boot_pose_check` 재판정 (m8 무전원사이클 시프트 전례).

## 운영 콘솔 (권장) — http://172.16.35.64:8800 (LAN) / http://100.80.46.80:8800 (tailscale)
콘솔 발력 총합% 는 **신 콘솔(분모 172.9N) + 신 어댑터(npc 0.02650)** 짝이어야 한다. 구 어댑터가 떠 있으면 완전 하중이
**68%** 로 나와 착지 게이트(≥90%)를 영영 못 넘는다 — 그것이 곧 구판 검출이다. 콘솔 정책 드롭다운 기본값 = **stand_v18b**
(구판은 stand_v7 — 구판 콘솔로 브리지를 띄우면 10.49kg 자산 정책이 올라간다. 콘솔 재기동 후 드롭다운 첫 항목을 확인).

## ⚠ 젯슨 스크립트 경로 (0808 실측)
| 스크립트 | 위치 |
|---|---|
| `boot_pose_check.py` `sensors_up.sh` `stack_down.sh` `release_all.py` `stage8_cmd.py` `can_up.sh` | `~/rl_calib/` |
| `rl_bridge_node.py` `stage8_12axis_mit_control_node.py` `analyze_fall.py` `posture_probe.py` `operator_console.py` | `~/humanoid_repo/jetson/scripts/` |

## −1. 배포 패키지 (젯슨 접속 가능해지면 — 현재 No route to host)
로컬에서 `bash jetson/deploy_package.sh` → `deploy/stand_7차_<날짜>.tar.gz` + `MANIFEST.md5`. 젯슨에서:
```bash
cd ~/humanoid_repo && tar xzf ~/stand_7차_*.tar.gz && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK   # 한 줄도 안 뜨고 MD5_OK
python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_v18b   # ALL PASS — stand_v14 는 "SKIP 미배포" 표기가 정상
python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -o "stand_v18b[^;]*"     # 등록·로드 확인
python3 jetson/scripts/foot_force_adapter_node.py --selftest                   # 체중 앵커 172.9N (npc 0.02650)
python3 jetson/scripts/release_all.py --selftest                               # 회수 순서 게이트
# ~/rl_calib 사본 갱신 — 젯슨은 이 경로의 사본을 실행한다(0808 실측 표). 어댑터는 sensors_up.sh 가 띄우는 경로를 확인.
diff ~/rl_calib/release_all.py jetson/scripts/release_all.py | head -5; cp jetson/scripts/release_all.py ~/rl_calib/release_all.py
grep -n 'foot_force_adapter' ~/rl_calib/sensors_up.sh      # 그 경로의 파일에 NEWTON_PER_COUNT = 0.02650 이 있어야 한다
```
패키지 내용: `models/stand_v15|v16|v17|v18*/policy.pt` → `logs/rsl_rl/biped12_stage4/<run>/exported/policy.pt` (브리지 CHECKPOINTS 경로),
`jetson/scripts/{rl_bridge_node,analyze_fall,posture_probe,operator_console,foot_force_adapter_node,release_all,zero_kinematic_check}.py`,
`rl_walking/deploy/*.py`, `robot_runtime/*.py`. 콘솔은 systemd 상시라 **콘솔 재기동 필요** (`sudo systemctl restart operator-console`).
0906 변경(배포 계약 감사 A1~A5): 콘솔 드롭다운 기본 stand_v18b · 어댑터 npc 재보정 · release_all 회수 게이트 · 브리지 중력 게이트 · selftest SKIP.

## 0. 전원·게이트 (하나라도 실패 시 진행 금지)
```bash
bash ~/rl_calib/can_up.sh                                   # 부팅 후 1회 (sudo)
(candump -ta can1 > /tmp/live_candump_$(date +%H%M).log 2>&1 &)
python3 ~/rl_calib/boot_pose_check.py --ref stand           # 12/12 (60분 유효)
bash ~/rl_calib/sensors_up.sh                               # IMU·발압력
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_v18b   # ALL PASS (학습 계약·중력 게이트 포함)
```
- 12/12 아니면 **배수 보정 우선**(`zero_anchor.py`), 재캡처는 최후. 하드웨어맵 갱신 시 **stage8 재기동 필수**.
- boot_pose_check 에 **△ 경고**(잔차 AK70 4°/AK45 2° 초과, 허용치 안)가 뜨면 통과이긴 하나 학습 영점편향 대역 경계다 — 자세를 다시 잡고 재실행,
  반복되면 영점 상태 의심(허용치 8°→4° 하향은 승윤님 결정).
- **중력 게이트(0906 신설):** 브리지는 stand 명령 시 IMU 중력벡터가 (0,0,−1) 에서 0.3(≈17°) 넘게 벗어나면 기동을 거부한다
  (`[거부] RL stand 시작 불가 — 중력벡터 정합 실패`). 장착보정 미적용 IMU 나 기울어진 착지의 검출이다 — 거부되면 IMU 보정 상태를
  확인하고 **착지를 다시 한다**. `force_gravity` 는 벤치 전용, 실기에서 쓰지 않는다.
- **패드 압축 실측(7차 신규, 1분):** 발을 든 상태에서 패드를 손가락으로 눌러 1mm 이상 들어가면 지면강성 1e4급 —
  시뮬(`--ground_k` 스윕)에서 k≤1e4 는 정책을 전멸시킨 조건이므로 **결과와 무관하게 기록**한다.

## 1~3. 기립 → 핸드오프 → 무장 (6차 카드와 동일)
```bash
python3 ~/humanoid_repo/jetson/scripts/stage8_12axis_mit_control_node.py --can-channel can1 --wait-takeover
python3 ~/rl_calib/hang_stand.py                            # 별도 터미널, 에어램프→kp150/kd5
# 착지: 매달린 발끝의 연직선을 조준, 평평하게, 천천히. 트림 全0.
touch /tmp/hang_ground; touch /tmp/hang_stand_handoff
python3 ~/rl_calib/stage8_cmd.py SET_BASELINE_FROM_CURRENT_ALL   # 12/12
python3 ~/rl_calib/stage8_cmd.py ARM_ALL                         # 30초 내
```
**착지가 세션을 지배한다** — 혼자 설 때까지(발력 ≥90%·앞뒤 ±3° 이내) 착지를 다시 한다. 기울어진 채 투입하면 정책이 아니라 착지를 재는 것이 된다.
시뮬 판정 조건이 정확히 "트림0·평평 착지 인수"다.
**정착 피치 기록(0906 신규):** ARM 뒤·정책 전, 혼자 선 상태의 IMU 피치를 적는다 — 기대 **−1~−3°**(시뮬 명목 −1.9°, 0804 실측 −1.7°).
−5° 이하로 뒤로 처지면 실물 유효 강성이 시뮬 명목보다 낮다는 뜻(구조 컴플라이언스) → 그 경우 §4 기대치(h3g)가 아니라 h3(강성 DR 포함) 수치가 대표.

**정책 투입 직전 소리 내어 읽는 5줄**
1. 트림 全0 · 발력 총합 ≥90% · 앞뒤 기울임 ±3° 이내 (아니면 착지 다시).
2. boot_pose_check 12/12 가 60분 안 · 경고(△) 관절 없음 · 하드웨어맵 갱신했으면 stage8 재기동함.
3. 콘솔 드롭다운 = stand_v18b · 어댑터 npc 0.02650(완전 하중 100% 근처) · 브리지 selftest ALL PASS.
4. 줄 느슨(받치되 당기지 않음) · 손 뗌 · 넘어질 때까지 무개입 · 카메라/블랙박스 기록 중.
5. 회수 순서 = 줄 인장 → 스택 종료 → 그 다음 release_all (동시 실행 금지, pkill 금지).

## 4. 정책 투입 → 무개입 관찰
```bash
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --checkpoint stand_v18b
ros2 topic pub --once -w 1 /humanoid/rl_bridge_command std_msgs/String "data: '{\"cmd\": \"stand\"}'"
```
- debug 토픽 `checkpoint=stand_v18b`, `blend_alpha` 1.5s 에 0→1, `tau_est_nm` 12개가 None 이 아님·`status_age_s` < 0.1 을 확인
  (None/스테일이면 stage8 status 미수신 — 토크 대조 불가). `soft_clamp_total` 이 자립 중 늘면 정책 목표가 soft 경계 밖(사후 표에 자동 표기).
- 그 다음은 **아무것도 하지 않는다.** 육안 기록: 기울임 방향, 가속 vs 왕복, 발 요 회전(오리발) 여부.
- 시뮬 예측(v18b, **h3 배포 경로 정합 판정 0906**): 배포조건 낙상은 **뒤** 우세 — V6 고정환경 뒤5·앞2, 자기환경 뒤13·앞9, DR ON 뒤 3:1.
  (0905 판의 "앞 우세(뒤2·앞11)" 는 정합 전 판정 래퍼(LPF→블렌딩→슬루 순서)의 산물 — 철회.)
  실물이 **뒤**로 가면 시뮬 예측과 일치 = 정책의 후방 회복 한계(D급 경로, 세대 과제). 실물이 **앞**으로 가면 시뮬에 없는 요인 =
  CoM 이 +16mm 보다 더 앞이거나 관측 지연·컴플라이언스·발압력 파이프라인(C급 경로, B/C 구분은 단조 밀기 여부).
- **기대치(0906 정착 실험 후):** 시뮬 낙상의 대부분은 서보 강성 0.2~0.5× DR 표본(실물 정적 자세 −1.7° 를 재현 못 하는 몸체)에서 난다. 실물은 kp150 그대로이므로
  대표 조건은 **강성 DR 을 뺀 h3g**: v18b V6 고정환경 com+16 **0/64**, 자기환경 20%, 정착+게이트 통과 후 0/45(V6)·23%(자기). 즉 **혼자 선 착지에서
  인수하면 서는 쪽이 기대치**다. 첫 투입에서 즉시 뒤로 무너지면 실물 유효 강성이 시뮬 명목보다 훨씬 낮다는 뜻(구조 컴플라이언스) — 블랙박스 루프지연 표의 τ 로 확인.

## 5. 판정 — 블랙박스
```bash
cd ~/humanoid_repo
python3 jetson/scripts/analyze_fall.py ~/logs/rl_bridge_<최신>.jsonl --compare ~/logs/rl_bridge_20260804_130927.jsonl
```
| 등급 | 의미 | 다음 행동 |
|---|---|---|
| **A** | 8s+ 자립, 종료 기울임 5° 이내 | 2회 재현 → march 사다리 진입 검토 |
| **B** | 자립 3~8s, 왕복 보정 관찰됨 | 토크표·방향으로 무너진 관절 특정 → 같은 착지에서 v17 대조 투입 |
| **C** | 전방 단조 밀기(v6b 형) 재현 | 정책 문제 아님 — 관측 지연·컴플라이언스·발압력 지연 순으로 파이프라인 규명 |
| **D** | 후방 붕괴(시뮬 배포조건·DR-ON 예측 방향과 일치) | 정책의 후방 회복 한계 — 후속 세대(v19 지연 3~8틱 등) 판정으로; CoM 재적합(+16mm) 재검토는 부차 |
추가 판정축(7차 신규): **τ피크/한계 최대 %** — 시뮬 명목(지연 4틱) 39% 대비 실물이 80% 를 넘으면 모터 부하 현실화(v16 축)가 부족한 것(발목롤 AK45 행은 스케일 미확정 — 제외).
**루프지연 표**(분석기 자동): 순수 지연 d 중앙 3틱·τ 40~80ms(5차 실측)에서 크게 벗어나면 통신·PD 상태부터 의심.
**같은 착지에서 v17 을 대조 투입**(무거운 몸 기대 정책이 실물에서 어느 방향으로 가는지)은 v18b 판정이 끝난 뒤 별도 착지로.

## 6. 종료·회수 (순서 엄수)
```bash
# ① 줄 재인장(사람) ② 스택 종료 ③ 그 다음에만 릴리즈
bash ~/rl_calib/stack_down.sh
python3 ~/rl_calib/release_all.py
```
release_all 과 stage8 armed 스트리밍 동시 실행 금지. 굽힌 관절 즉시 릴리즈 금지(회생 OVP 사망 전례). 프로세스 종료는 명시 PID 로만.
**0906 부터 release_all 은 stage8·hang_stand·브리지(--live) 가 살아 있으면 거부(exit 3)한다** — 이 거부는 절차 위반 신호이지 버그가
아니다. stack_down 후 재실행. `--force '<사유>'` 는 버스 사망 직전 비상 전용.

## 세션 후
블랙박스를 `.scratch/bb/` 로 복사 → `analyze_fall.py` 토크표를 `reports/gen_v18b/README.md` 명목 관절표 옆에 붙여 sim↔real 토크 대조 기록.
