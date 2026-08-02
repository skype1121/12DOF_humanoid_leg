# march 실물 실행 카드 (v3, 2026-08-03 — 1차 데뷔 실측 반영)

정책: **`march_v5b`** (0803 측정 기반 재정합판 — 게이트·하네스 통과, 젯슨 배포 완료).
브리지 기동: `--live --checkpoint march_v5b` (클록 1.1Hz는 CKPT_GAIT_FREQ로 자동 적용).
march 명령: `{"cmd": "march"}` + 브리지 기동 플래그가 march_v5b — debug 토픽에서 checkpoint=march_v5b 확인 필수.
로봇 상태 전제: 에어 행잉, 모터 전원 ON, 젯슨 ON, can1 UP.

## v3 개정 요지 (0802 데뷔 1차 실측)
- **모터 명령 타임아웃 ~200-300ms (브레이크 모드)** — 하중 지지 중 어떤 이유로든 스트리밍이 끊기면 수백 ms 내 붕괴. 침묵을 안전 상태로 착각 금지.
- **candump 상시 녹화가 0단계** — 이번 진단의 결정타. 라이브 전 필수.
- 검증 사다리: **스탠드(정지) 확인 → march 수 초 → 정지 → 연장** — 첫 발화에서 길게 돌리지 않는다.
- 스탠딩 기울기 확인: IMU + 육안 (1차 실측: 전방 ~2.8° 기운 채 시작 — 힙 트림 ±1~2° 티트레이션 검토).

## 0-a. candump 녹화 시작 (모든 라이브 절차의 0단계)
```bash
(candump -ta can1 > /tmp/live_candump_$(date +%H%M).log 2>&1 &)
```

## 0. 사전 게이트 (한 번이라도 실패하면 진행 금지)
```bash
# 젯슨에서
cd ~/humanoid_repo
python3 jetson/scripts/boot_pose_check.py --ref stand      # 12/12 판정 (60분 유효)
bash jetson/scripts/sensors_up.sh                          # IMU·발압력 멱등 기동
python3 jetson/scripts/rl_bridge_node.py --selftest        # ALL PASS 확인
```
- boot_pose_check 12/12 아니면: 배수 보정(m6 ±10°·m11 ±36° 성향) → 재판정.
- 발진가드 v2 확인: selftest에 "주기2채터/8Hz±3° 트리거" 문구.

## 1. stage8 선기동 대기 (핸드오프 **전에**)
```bash
python3 jetson/scripts/stage8_12axis_mit_control_node.py --can-channel can1 --wait-takeover
```
- CAN 완전 침묵으로 대기. `/tmp/stage8_takeover_waiting` 마커를 매 틱 갱신 —
  hang_stand는 이 마커가 3s 이내일 때만 핸드오프를 발화한다 (수신자 없는 핸드오프 방지).

## 2. hang_stand 기립 → 핸드오프
```bash
python3 jetson/scripts/hang_stand.py   # 별도 터미널
# 접지 + 게인램프 완료(kp150) 확인 후:
touch /tmp/hang_stand_handoff
```
- 거부 메시지 나오면 원인 해소 전 재시도 금지 (공중/램프중/stage8 미대기).
- 성공 시 hang_stand 즉시 종료, stage8이 한 틱 내 동일 명령 인수 스트리밍.
- 인수홀드 자체 가드: 실측-홀드 20° 괴리·피드백 두절 60s·방치 300s → 자동 캔슬
  (캔슬 후 모터는 마지막 명령 유지, ENTER 폴링 영구 금지 — 회수는 7단계 절차).

## 3. 베이스라인 → 무장 (30초 내 연속 실행)
```bash
python3 jetson/scripts/stage8_cmd.py SET_BASELINE_FROM_CURRENT_ALL   # 12/12 확인
python3 jetson/scripts/stage8_cmd.py ARM_ALL                         # 인수홀드 무봉합 전환
```
- ARM은 12/12 베이스라인 + 30s 신선도 필수 (부분 무장 구조적 거부).

## 4. 브리지 라이브 + 정책 투입
```bash
python3 jetson/scripts/rl_bridge_node.py --live --checkpoint march_v3
# 별도 터미널 — 검증 전송 (-w 1: 구독자 확인 후 1회 발행):
ros2 topic pub --once -w 1 /rl_bridge/cmd std_msgs/String "data: '{\"mode\": \"march\"}'"
# 직후 /rl_bridge/debug 에서 checkpoint=march_v3·live·armed 확인
```
- 사람은 줄 옆 스탠바이. 발진가드 v2가 0.2~0.7s 내 자동 STOP_ALL (march 보행 오트립 0 검증).
- 센서 공백 시 자동 STOP_ALL + 정책 비활성 — **자동 재개 없음**, 원인 확인 후 march 재명령.
- stage8 status의 `soft_clamp_counts` 관찰: 무릎(니락 스톱 포화) 외 관절에서 지속 증가하면 이상 신호.

## 5. 정지
march 정지 = 브리지 estop(STOP_ALL 동결 홀드) → 로봇은 서 있는 채 유지.

## 6. 종료·회수 (순서 엄수 — E2E 최종점검: release_all은 stage8 종료 후에만)
```bash
# ① 줄 재인장 (사람)
# ② 스택 종료 — stage8·bridge를 먼저 죽인다 (모터는 마지막 명령을 물고 유지됨)
bash jetson/scripts/stack_down.sh
# ③ 그 다음에만 릴리즈 (버스 단독 점유 상태에서)
python3 jetson/scripts/release_all.py
```
- **release_all을 stage8 armed/홀드 스트리밍과 동시 실행 금지** — 같은 버스에 상반 명령 충돌.

## 7. 이상 시 회수 (발진가드 트립·홀드 캔슬·토크 이상 공통)
6단계와 동일: 줄 인장 → stack_down.sh → release_all.py. 원인 파악 전 재시도 금지.
블랙박스: `~/rl_bridge_bb/*.jsonl` (홈 디렉토리 — /tmp 금지).

## 다음 세션 벤치 항목
- 모터 명령 타임아웃(무명령 방치 시 자체 릴리즈까지 시간) 2분 벤치 실측 — 인수홀드 공백 마진 산정 근거.
