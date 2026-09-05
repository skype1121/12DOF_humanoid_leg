# stand_v18b 7차 무개입 실행 카드 (2026-09-05 21:30 확정판)

**이 세션이 답할 질문 — 실질량 17.62kg·패드 접촉 자산으로 학습한 첫 세대가 실물에서
(1) 어느 방향으로, (2) 어떤 토크로, (3) 몇 초 만에 무너지는가.** "서는가" 는 부차.
시뮬 배포조건(구래퍼 com+16, V6 고정환경) 낙상 v18b 14.1% — 실물 오차 축을 다 켠 자기환경에선
20~25% 이므로 **첫 투입에서 서지 못해도 데이터는 유효**하다. 판정은 블랙박스로만.

**정책 후보 (h2 재판정 + 정정 env 세대 판정, `reports/0905_감사_정정_보고.md`)**

| 순위 | 정책 | 근거 | 우려 |
|---|---|---|---|
| 1 | **`stand_v18b`** (run 2026-09-05_19-37-39 / model_120998, md5 fac90eb5…) | **실질량 17.62kg·정정 관측으로 학습된 유일한 후보**(v15 계보). V6 고정환경 구com+16 14.1% / 신com+16 7.8%, τ피크/한계 43%, 발 요차 3.9°, E_smth 명목 0.029 최저 | 자기환경 배포조건 20~25%, 배포조건 낙상 방향 **앞** 우세 |
| 대조 | `stand_v17` (2026-09-04_21-48-32 / model_126997, md5 4dc9fd58…) | 구래퍼 com+16 10.9%(v18b 와 동률권), τ피크 41%, slip 0.9% | **19~21kg 로 학습**(link_mass_calib 상속) — 실물보다 무거운 몸을 기대 |
| 예비 | `stand_v15` (16-43-02 / model_114999) | 자기환경 배포조건 최강 | 발 요차 24° 오리발 |

6차(v7) 카드와 달라진 전제: **체중 172.9N**(저울 17.62kg — 4차 "90% 자립" 은 실제 61%). 블랙박스에
`tau_est_nm`·`temp_raw` 신설 → `analyze_fall.py` 가 관절별 토크 피크/포화 표를 시뮬 명목 판정과 같은 축으로 낸다.

---

## 프로토콜 — 무개입 (5차에서 확립, 변경 없음)
- **넘어질 때까지 손대지 않는다.** 손 개입 = 데이터 오염. 안전망은 **줄 + 브리지 가드**.
- 판정은 **발력 총합 %** (172.9N 대비) 1순위. IMU 스냅샷 1장 판정 금지.
- 낙하 후에는 전원을 안 껐어도 `boot_pose_check` 재판정 (m8 무전원사이클 시프트 전례).

## 운영 콘솔 (권장) — http://172.16.35.64:8800 (LAN) / http://100.80.46.80:8800 (tailscale)
콘솔 상단 발력 총합% 표기는 172.9N 기준으로 갱신된 `operator_console.py` 여야 한다(아래 배포 목록). 구판이면 %가 1.47배 부풀려 나온다.

## ⚠ 젯슨 스크립트 경로 (0808 실측)
| 스크립트 | 위치 |
|---|---|
| `boot_pose_check.py` `sensors_up.sh` `stack_down.sh` `release_all.py` `stage8_cmd.py` `can_up.sh` | `~/rl_calib/` |
| `rl_bridge_node.py` `stage8_12axis_mit_control_node.py` `analyze_fall.py` `posture_probe.py` `operator_console.py` | `~/humanoid_repo/jetson/scripts/` |

## −1. 배포 패키지 (젯슨 접속 가능해지면 — 현재 No route to host)
로컬에서 `bash jetson/deploy_package.sh` → `deploy/stand_7차_<날짜>.tar.gz` + `MANIFEST.md5`. 젯슨에서:
```bash
cd ~/humanoid_repo && tar xzf ~/stand_7차_*.tar.gz && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK   # 한 줄도 안 뜨고 MD5_OK
python3 jetson/scripts/rl_bridge_node.py --selftest                                   # ALL PASS (walk + stand_* 전부 순회)
python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -o "stand_v18b[^;]*"     # 등록·로드 확인
```
패키지 내용: `models/stand_v15|v16|v17|v18*/policy.pt` → `logs/rsl_rl/biped12_stage4/<run>/exported/policy.pt` (브리지 CHECKPOINTS 경로),
`jetson/scripts/{rl_bridge_node,analyze_fall,posture_probe,operator_console}.py`, `rl_walking/deploy/*.py`, `robot_runtime/*.py`.
콘솔은 systemd 상시라 **콘솔 재기동 필요** (`sudo systemctl restart operator-console`).

## 0. 전원·게이트 (하나라도 실패 시 진행 금지)
```bash
bash ~/rl_calib/can_up.sh                                   # 부팅 후 1회 (sudo)
(candump -ta can1 > /tmp/live_candump_$(date +%H%M).log 2>&1 &)
python3 ~/rl_calib/boot_pose_check.py --ref stand           # 12/12 (60분 유효)
bash ~/rl_calib/sensors_up.sh                               # IMU·발압력
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --selftest   # ALL PASS
```
- 12/12 아니면 **배수 보정 우선**(`zero_anchor.py`), 재캡처는 최후. 하드웨어맵 갱신 시 **stage8 재기동 필수**.
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

## 4. 정책 투입 → 무개입 관찰
```bash
python3 ~/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --checkpoint stand_v18b
ros2 topic pub --once -w 1 /humanoid/rl_bridge_command std_msgs/String "data: '{\"cmd\": \"stand\"}'"
```
- debug 토픽 `checkpoint=stand_v18b`, `blend_alpha` 1.5s 에 0→1, `tau_est_nm` 12개가 None 이 아님을 확인(None 이면 stage8 status 미수신 — 토크 대조 불가).
- 그 다음은 **아무것도 하지 않는다.** 육안 기록: 기울임 방향, 가속 vs 왕복, 발 요 회전(오리발) 여부.
- 시뮬 예측(v18b): 배포조건 낙상은 **앞** 우세(뒤2·앞11), DR ON 은 뒤 2:1. 실물이 **앞**으로 가면 정책 예측과 일치(B/C 구분은 단조 밀기 여부), **뒤**로 가면 CoM 재적합(+16mm) 가정을 의심.

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
| **D** | 후방 붕괴(시뮬 DR-ON 방향과 일치) | 실물 CoM 재적합(+16mm 가정 검증) → com_range 재베이크 |
추가 판정축(7차 신규): **τ피크/한계 최대 %** — 시뮬 명목 43% 대비 실물이 80% 를 넘으면 모터 부하 현실화(v16 축)가 부족한 것.
**같은 착지에서 v17 을 대조 투입**(무거운 몸 기대 정책이 실물에서 어느 방향으로 가는지)은 v18b 판정이 끝난 뒤 별도 착지로.

## 6. 종료·회수 (순서 엄수)
```bash
# ① 줄 재인장(사람) ② 스택 종료 ③ 그 다음에만 릴리즈
bash ~/rl_calib/stack_down.sh
python3 ~/rl_calib/release_all.py
```
release_all 과 stage8 armed 스트리밍 동시 실행 금지. 굽힌 관절 즉시 릴리즈 금지(회생 OVP 사망 전례). 프로세스 종료는 명시 PID 로만.

## 세션 후
블랙박스를 `.scratch/bb/` 로 복사 → `analyze_fall.py` 토크표를 `reports/gen_v18b/README.md` 명목 관절표 옆에 붙여 sim↔real 토크 대조 기록.
