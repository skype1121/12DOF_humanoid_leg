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

| 1006 | WOMS 학습 | — | KILLED(1000 판정) → ANALYZED → DECISION RECORDED(LOG 1006) → NEXT=WOMX | — |
| 1006 | WOMX 학습+이정표 평가(womx.sh PID 2242018) | "이정표 WOMX {CK} 평가 끝"/"학습 끝" in scratchpad/womx.out | log_picture/walk_eval_womx_{CK}_omni.txt | 감시 bkuuiwdqz(500) → omni_judge; 1000에서 KILL 판정 |
| 1006 | KAP0 | — | COMPLETED → ANALYZED → DECISION(LOG: KE 유격 무관) | — |
| 1006 | DSW | — | COMPLETED → ANALYZED → DECISION(LOG: 착지 앞·뒤 지연 모두 해) → NEXT=RD | — |
| 1006 | RD: rd5(G24RaD5, PID 2247390) → rd3(G24Ra 대조, rd5 뒤 순차 — 첫 동시 기동 OOM 사망, run 은 scratchpad/dead_rd3_oom_run 로 격리) → KE 평가(rd2.sh PID 2250343) | "RD 끝" in scratchpad/rd.out | log_picture/ground_recovery_rd{5,3}d{4,3,1}_* | 감시 bbk2ox40x(짝 비교), b7ct0id8x(rd3 OOM 감시) |
| 1006 | W2om 조건부 기동(w2om.sh PID 2253932): WOMX 1000 KILL → WOMX PID 종료 후 W2om 이어 학습 1500 + 26000/26500/27000 omni 평가 | "WOMX KILL/계속", "W2om {CK} 평가 끝" in scratchpad/w2om.out | log_picture/walk_eval_w2om_{CK}_omni.txt | 감시 b57r4cfog(분기) |
