# sim_walking — Isaac Sim 12DOF 하체 보행 데모

**시뮬레이션 전용.** 실물 모터/CAN/Jetson에는 아무것도 보내지 않는다.
2026-07-18, Isaac Sim 5.1.0 + walk_scene.usd 에서 개발·검증.

## 실행 (원커맨드)

전제: Isaac Sim이 MCP 확장과 함께 실행 중 (`localhost:8766`):
```bash
ISAACSIM_ROOT=/home/ryu/isaacsim/isaac-sim-5.1.0 /home/ryu/isaacsim-mcp-server/scripts/run_isaac_sim.sh
```

데모 (배치 실행 + 판정):
```bash
python3 scripts/walk_demo.py                    # 16초 연속 보행 (판정 출력)
python3 scripts/walk_demo.py --steps 5          # 5보 걷고 우아하게 정지
python3 scripts/walk_demo.py --speed 0.9        # 속도 배율 (0.8~1.1 검증)
python3 scripts/walk_demo.py --video            # 프레임 캡처 + demo_output/walk_demo.mp4
```
판정 기준: **3보 이상 + 전진 0.2m 이상 + 무낙상** → PASS/FAIL 및 지표 출력.

걷기 전용 UI (리모컨 — Main UI와 별개):
```bash
python3 scripts/walk_ui.py
```
- 스탠딩 시작/유지 · 계속 걷기 · N보 걷고 서기 · **걷다가 서기**(진행 중 정지)
- 외란 주입: 앞/뒤/왼쪽/오른쪽 (세기 5~40N 슬라이더, 0.15s)
- 실시간 상태: 모드(스탠딩/보행/넘어짐)·전진·기울기·방향·골반높이
- 리셋 버튼: 넘어졌을 때 씬 초기화
- 내부: `sim_walking/live_controller.py`(Isaac 상주, Kit 업데이트 콜백)를
  TCP 8766 단문 명령으로 제어 — 명령 즉시 반환이라 걷는 중 조작 가능.
  콜백 구동이라 시뮬이 실시간보다 빠르게(~3.6x) 돈다.
- 그래프: `python3 scripts/plot_walk_telemetry.py [--tag demo|--list]`

오프라인 테스트 (Isaac 불필요):
```bash
python3 -m pytest tests/test_sim_walking.py -v
```

## 검증된 성능 (2026-07-18)

| 항목 | 결과 |
|---|---|
| 기본 데모 16s | 7보, 전진 0.68m, 무낙상, yaw +2° |
| 장거리 36s | 21보, 전진 1.0m, 무낙상 |
| --steps 5 정지 | 무낙상, 정지 후 직립 유지(피드백 유지 시) |
| 속도 배율 | 0.8/1.0/1.1 PASS, 1.2 전도 |
| 외란(0.15s 골반 힘) | 측방 25N 복구 / 후방 10N 복구·20N 전도 / 전방 15N 전도 |

## 구조

```
config/sim_walk_params.json   검증 파라미터 동결(근거 주석 포함)
config/sim_joint_limits.json  관절 리밋(자기충돌 실측+해부학) — walk_scene에 반영됨
config/sim_dynamics.json      질량 감사·토크상한 25Nm(AK70-10)·시뮬 게인·직립 트림
sim_walking/
  sim_session.py       Isaac 세션 attach/명령/텔레메트리/캡처 (이름 키 기반)
  gait_static.py       ★정적 4상 보행 FSM: LEAN_L→STEP_R→LEAN_R→STEP_L
  runner.py            60Hz 구동 + 밸런스 피드백 + 낙상감지 + 지표/외란
  policy_interface.py  RL 정책 교체 경계(obs 스키마·규약)
  gait_fsm.py          (구) 주기 로킹 방식 — 참고용
  batch.py             파라미터 비교 배치
scripts/walk_demo.py   원커맨드 데모(TCP 8766로 실행 중 Isaac 구동)
```

## 원리 요약

- **체중이동**: hip_a 평행사변형(골반 평행이동) + ankle_r 전신 롤.
  hip_a만으론 CoM이 한 발 위에 못 감(내전한계 13.9°→86mm < 발 안쪽모서리
  100mm — 실측). 주기적 로킹은 측방 공진으로 전도(3회 실측)해 **정적 4상**으로 감.
- **전진**: 지지힙 견인(+S→−S)을 **양발지지 구간에만** 배치. 한발지지 중
  견인은 반작용 피치 스파이크로 장거리 전도를 유발(실측 18.3s 전도 → 재배치
  후 36s 완주).
- **피드백**(runner): 시상면 lean_ap→양발목 도르시(P+I 0.010/0.03),
  관상면 lean_lat→hip_a(P+D 0.012/0.010, FSM의 lat_ref 추종),
  요 yaw→hip_r(P −0.008, 부호 실측 캘리브레이션).
- **정지**: 스텝 종료 후 롤/무릎/발목만 중립 복귀, hip_f는 착지 자세 유지
  (벌린 발에 힙 0 강제 시 후방 전도 — 실측).

## 제한사항 (알려진 것)

1. **직립 유지에도 능동 밸런스 필요** — 피드백 루프 정지 시 수 초 내 서서히
   기울어짐(실물 로봇과 동일한 특성). 데모 러너는 정지 후에도 루프 유지.
2. **피치(전후) 축이 약축** — STEP 후반 lean_ap 피크 ~15°(회복됨),
   전후방 외란 내성 10~15N×0.15s. 발이 전후 대칭이라 인체식 푸시오프가
   역효과(실측)로 미사용.
3. 측방 드리프트 ~0.08m/36s, yaw 드리프트 ~12°/36s (피드백으로 유계).
4. 스텝 카운터는 발 링크 z 기반 보수적 집계(명령 5보 → 집계 4~5보 가능).
5. 젯슨/카메라 상부 질량 미모델링(URDF 교체 예정) — 추가 시 재튜닝 필요.
6. 시뮬 전용 파라미터 — 실물 이식 시 게인/토크 재검토 필수.
7. **sim-to-real 게인 갭 (실측, 2026-07-18)**: MIT 프로토콜 kd≤5 제약에서
   현 보행(kd25 튜닝)은 재현 불가 — kp 100~250 전 범위에서 측방 체중이동
   오버슈트로 전도. **스탠딩·정적 체중이동은 kd5로 가능** (kp60+).
   상세와 이식 로드맵: `config/sim_dynamics.json`의
   `sim2real_gain_experiment_2026_07_18` 참고.

## RL로 교체하려면

`policy_interface.py` 참고. `targets(t)`(+선택 `observe(obs)`)만 구현하면
runner가 그대로 구동한다. 관절 이름·부호 규약은 하드웨어맵/joint_direction과 동일.
