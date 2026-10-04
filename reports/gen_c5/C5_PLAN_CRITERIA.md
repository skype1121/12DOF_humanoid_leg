# C5 계획·사전 기준 (2026-10-04 오후, 사용자 승인 — 결과 보기 전 고정)
C5 승인 / C4 보류. 핵심 문제 = G21a → G24R 전환 시 handoff distribution mismatch 제거(서기·회복 능력 새로 만들기 아님).

## 구조
- 정상: G21a-11300 · 회복: G24R 계열 · 감지기(IMU): 켜기 tilt > 4° 또는 |ω_xy| > 0.5 rad/s
- 복귀(히스테리시스): tilt < 2.5° & |ω_xy| < 0.3 rad/s 가 calm 시간 연속(+ 양발 접촉 — SW-1 유지) · 회복 진입 뒤 최소 dwell 0.3–0.5 s (기본 0.4 s)
- calm 시간: SW-1 값 1.0 s 유지(사용자 미지정 — 변경 시 별도 기록)

## 순서
1. W2-8 평가 완료(보행 축 — C5 와 결과 섞지 않음)
2. C5 runtime state handoff 분석(전환 직전/직후 action·prev_action·action_ema·q·qdot raw/filtered·gyro·tilt·ω·contact·Fz·policy id·시각) → action/obs jump 정량화
3. state continuity 수정 후 SW 재평가 (개선되면 재학습보다 우선)
4. 앞 12 N 손실 남으면 G24R-H(C5-H) 학습 — 변경 변수 = 초기 상태 분포 하나(보상 불변)
   분포: 기존 G24R 시작 40 % · 실제 G21a→회복 전환 snapshot 50 % · snapshot 주변 작은 섭동 10 %
5. C5-H 전체 SW 평가 → 6. 합격 판정 → 7. C4 필요성 재검토

## 앞 12 N 손실 원인 분해 (반드시 셋으로)
- A 상태 인계 불일치(관측/액션/필터 튐) → runtime handoff 수정
- B G21a 가 버틴 상태가 G24R 학습 분포 밖 → 전환 순간 tilt·ω·vxy·q·qdot·발 하중을 G24R-a 자체 분포와 비교 → handoff-state 학습
- C 회복 정책이 G21a 의 이전 action/context 를 해석 못 함 → history/action 연속성 또는 handoff 조건 학습

## C5-H 합격 기준 (고정)
Standing: nd 흔들림 ≤ 2.3 % (4 seed 합산) · d3nd 는 G21a 대비 유의 악화 없음(짝 부호검정 p ≥ 0.05) — C0 2.9 % 는 별도 강건성 세대에서 재해결
Recovery: sf10 ≤ 5 % · sf12 ≤ 6 % 목표 / ≤ 8 % 최소 허용 · sf15 ≤ 30 % · b18 ≤ 2 %
Switching: 정상 서기 오탐(인계 과도 뒤, t > 0.5 s 첫 켜짐) < 2 % · chattering 사실상 0 · 회복→서기 복귀 뒤 재발산 < 2 %
품질(기록): switch→takeoff · takeoff→touchdown · touchdown→settle · 스텝 변위 · 하중 수용(완주 창 기준) · 정착 시간

## 금지
Jetson 배포·설정 변경, 실물 모터·정책 실행, 실물 push, 하드웨어맵 변경. C5-H 사전 평가 통과 시 실물 진단 후보로 정리해 승인 요청.
