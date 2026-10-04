# 우선순위·결정 규칙 (2026-10-04 12:26, 사용자 합의 — 결과 보기 전 기록)

현재 위치: "걷기/스텝을 못 찾는 단계"는 거의 끝. 이제 locomotion·recovery·standing 을
한 정책 안에서 충돌 없이 공존시키는 단계.

## 우선순위
1. **G24R-c — 회복 후 하중수용/정착** (스텝 자체는 sf15 93 % — 발 들기 강화 불필요)
   - 목표: 앞15 하중 수용/착지 72 → ≥ 90 %, 정착 시간 중앙 1.36 → ≤ 1.0 s
2. **standing oscillation — 최대 배포 blocker**
   - G24R-a nd 7.1 / d3nd 11.6 %. G24R-b(서기 비율↑)가 회복을 무너뜨림
     → 데이터 비율 문제가 아니라 stand objective 와 recovery/walking 표현의 간섭 신호.
   - 다음 방향은 "서기 샘플 추가"가 아니라 mode-specific gating / conditional behavior.
3. **walking yaw drift** — W2-7 학습 중 yaw 보상 0.009 → ~0.34 (첫 gradient).
   8 s 표류가 13~19° → 한 자릿수 초반 이하면 별도 축으로 정리.

## 순서
W2-7 평가 → G24R-c 평가 → 두 결과 확인 → C4 승인 여부 결정(사용자).

## 결정 규칙 (사전)
- G24R-c 가 회복 품질(하중 ≥ 90 %, 정착 ≤ 1.0 s)을 개선해도 nd ≥ 7 % 가 유지되면
  → 병목을 "단일 연속 정책 안 standing ↔ locomotion/recovery 모드 간섭"으로 좁히고
  C4(밀림 감지 → 클록 위상 리셋 → 회복 스텝 모드)를 승인 요청 1순위로 올린다.
- C1(critic α 관측)·C2(kickstarting)는 원인 분석 비용·기존 행동 훼손 위험이 커서 보류.
- G24R-a 앞15 32 % 는 15 N 부근 강건성 절벽(12 N 4.5 %, 10 N 1.8 %) — STEP primitive 재설계 대상 아님.
