# STEP-A0c decision — **FAIL (fail-fast 400)** — H-A0c 기각
- 관측: 11700 α=0: 비대칭 +0.024 mg·이륙 0 %; α=2 평가 이륙 0.0 %(학습 전 30 %). 학습 블록 착지 0.000.
- 결론: 접촉 무관 하중 비대칭 보상(r_asym)을 없애도 이륙 억제로 수렴 → r_asym 은 원인 아님. A0·A0b·A0c 3회 같은 실패 → 구조 진단(diag_reward_terms).
- 구조 진단 결과: 공중 틱 손해 최대 항 = knee_stance_flexion(+4, 접지 발마다) −0.025/틱 → 다음 A0f(스텝 모드에서 이 항 OFF).
