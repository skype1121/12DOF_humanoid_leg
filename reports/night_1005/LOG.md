# 자율 연구 기록 (1005 밤~)

## 현재 best
- STAND: G21a-11300 · RECOVERY: KE(젯슨 배포됨, 실물 미시험) · WALK: W2-8 24791(고정 ~0.09 m/s, 실물 미연결)

## 활성 branch (트랙당 baseline · best · challenger)
- Walk: baseline W2-8 · best(cadence 연동) W2q 25590 · challenger **W2gs**(클록 접촉 일정 보상)
- Recovery: baseline C5 · best KE · challenger (posture restoration 측정 후 결정)

## 실험 W2gs (보행 challenger)
- Hypothesis: sb0 사건 보상(고정 착지 창 2–10 cm·놓친 스텝 −20)이 보폭을 묶는다. 클록 접촉 일정으로 바꾸면 걸음은 강제되고 보폭은 속도 추종을 따라 자유로워진다.
- Changed variable: 걷기 모드 sb0 사건·스윙 9항 → gait_schedule_walk(가중 2). 부모 W2q 25590, 500 반복.
- Expected signal: 스윙당 이동(vx/cadence)이 명령과 함께 증가.
- Success: 실제 vx 기울기(0.05→0.15) ≥ 0.3, 걷기 낙상 ≤ 8 %, 교대 ≥ 90 %.
- Kill: +300(25900)에서 낙상 > 15 % 또는 교대 < 80 % 또는 보행 소실(vx < 0.03).
