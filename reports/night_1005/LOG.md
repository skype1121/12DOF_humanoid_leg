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

## 측정 PR0 (회복 자세 복귀, 평가 전용)
- Hypothesis: 회복 뒤 '정상 서기 복귀'가 덜 된 생존 env 가 상당수다(관절·높이·기울기 편차 잔류).
- Changed variable: 없음(측정). KE vs 관측 대조 C5, 지연 3, seed 1·2, sf12 sf15 fs80 fi40 b18 bi30.
- 지표: rest_t(외란 시작→복귀 0.5 s 유지), 끝 시점 관절 평균 편차·골반 높이·기울기·ω(기준 = 외란 직전 자세).
- 결정: 미복귀(rest_t 없음) 생존 비율이 > 20 % 면 자세 복귀를 회복 트랙의 다음 병목으로 삼는다. 아니면 회복 A(짧은 첫 걸음)·지연을 유지한다.

## PR0 결과 — 자세 복귀는 주 병목 아님
- 외란 뒤 생존 env 미복귀(KE): sf12 3 % · sf15 8 % · fs80 11 % · fi40 9 % · b18 12 % · bi30 11 %(기준 20 % 미만). 복귀 시간 중앙 2.7–3.7 s.
- 끝 자세: 관절 평균 편차 ~2°, 골반 높이 −0.2 cm, 기울기 0.7–0.8°, ω 0.02 → 정상 서기 근처. G21a 로의 인계가 자연 복귀시킨다.
- nd 의 '미복귀 100 %'는 지표 산물이다(외란 없으면 rest_t 미기록).
- 결정: 회복 병목은 A(짧은 첫 걸음)·지연. 다음 = A 원인 분해(평가 전용).

## 실험 A-DEC (평가 전용)
- Hypothesis: 짧은 첫 걸음은 시간 부족(지연 때문에 스윙이 짧다)이다.
- Changed variable: 지연 1 vs 4(KE, 관측 신호, seed 1·2, sf15 fs80 fi40).
- Signal: 낙상 env 의 td1_len·td1_t·여유 분포.
- 판정: 지연 4 에서 첫 걸음 길이가 줄고 착지가 늦으면 → 시간 부족(지렛대 = 실물 지연 단축·예측). 길이·시각이 같으면 → 정책 선택(지렛대 = 스텝 목표 cue·학습).
