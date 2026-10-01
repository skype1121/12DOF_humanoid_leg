# OSS 분석 2 — 회복 스텝/밀기 회복 (에이전트 보고 요약, 1002 04시)
- 정지 서기에서 스텝을 얻은 사례: Ferigo 2021(iCub, 접촉 보상 = "어느 한 발이라도", 200 N×0.2 s 구면 균등 평균 5 s, 1.21 N·s/kg),
  van Marum 2024(Digit, 서기 명령에서 접촉 항 상수 1 — 양발 접촉 보상은 회복 스텝을 벌한다고 명시, 20–200 N 0.2–0.5 s).
  위상 시계 상시(unitree_rl_gym, humanoid-gym) 는 무외란 발 들기 목표와 충돌. FALCON 서기 모드는 한 발 떨어지면 −4(스텝 금지).
- 우리 진단(코드·tensorboard, 가설 포함):
  1. 게이트 열림 동안 양(+) 보상은 recovery_quality(+5) 뿐인데 `q *= both` → 스윙 중 0 = 스텝 구조적 손해(코드 확인).
  2. 학습 충격 5–30 N×0.2–0.4 s (최대 0.68 N·s/kg); 평가 뒤18N(5.4 N·s) 이상 뒤 충격은 슬롯의 ~6 %. push_robot ±0.25 m/s 는 평가 Δv 0.31 미만.
  3. feet_impact 문턱 180 N 은 12 kg 시절 값(17.6 kg 한 발 정하중 173 N) — 측정 크기 작음.
  4. 스텝 뒤 stance_shape·pose_symmetry 가 엇갈린 발 벌(추정 −1.7/s).
  5. 학습 종료 51 % 낙상(base_too_low), termination_penalty 실효 −0.2/회.
  6. `dst_last_end` 는 G8 이후 갱신 안 됨 → 외란 창은 `dist_window_mask`(_dist_end_time) 사용.
- 제안: G18a recovery_quality 접촉 조건 = 외란 창(외력 시작~종료+1.5 s) 안에서만 any_contact (외생 신호라 자기 개방 차단) /
  G18b F_imp_range (5,30)→(10,45) / G18c 외란 창 조건 착지 보상(shaping). 대조 = 같은 출발점 무변경 이어 학습 필수.
