# 작업 장부 (상태: RUNNING → COMPLETED → ANALYZED → DECISION RECORDED → NEXT ACTION STARTED)

| job | 시작 | PID(스크립트) | 완료 조건(이정표) | 결과 파일 | 완료 뒤 분석 | 상태 |
|---|---|---|---|---|---|---|
| WOM(이어 학습 500) | 1005 22:2x | — | WOM 끝 · WOMOMNI 끝 | walk_eval_wom_26089_none / omni_wom26089 | 단계 분해·전방향 | DECISION(중단) → NEXT=WOMS |
| KAP(발목 유격) | 1005 22:5x | — | KAP 끝 | ground_recovery_kap* | fall_pool | DECISION(큰 위험 아님) → NEXT=DROP |
| WOMS(처음부터 3000) | 1006 02:0x | 2202509 | WOMS 500/1000/1500/2000/2500 평가 끝 · WOMS 학습 끝 | walk_eval_woms_<ck>_omni.txt | 전방향 표 + kill(1000: 유효 cadence<0.3·낙상>30 %·무반응) | RUNNING (대기 감시 b5golm89i: 500) |
| DROP(착지 출발) | 1006 02:1x | 2204384 | DROP 끝 | ground_recovery_dr0*d3 | watch_drop.sh | DECISION(1.2 s 시작 이름·6 cm 미달) → NEXT=DROP2 |
| DROP2(시작 지연) | 02:12 | 2214996 | DROP2 끝 | ground_recovery_dq*d3 | watch_drop.sh(TAGS) | RUNNING |
| KAP0(유격 0 같은 경로) | 02:22 | 2225561 | KAP0 끝 | ground_recovery_kap[kc]0d3 | fall_pool kapk{0,2,4} vs 짝 | RUNNING |
| DSW(착지 시점 지연 전환) | 02:22 | 2226881 | DSW 끝 | ground_recovery_dsw*_ | fall_pool 4팔 | RUNNING |
