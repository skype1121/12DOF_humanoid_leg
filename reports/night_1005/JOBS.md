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
| 1006 | WOMX | — | KILLED(1000, 명시 PID) → ANALYZED → DECISION(보상 튜닝 2회 → 구조 전환) → NEXT=W2om 시작(03:18) | — |
| 1006 | W2om 학습(train PID 는 w2om.sh 자식) + 사전 등록 +500 평가(w2om500.sh PID 2270638, model_26100) | "W2om 26100 평가 끝" in scratchpad/w2om500.out | log_picture/walk_eval_w2om_26100_omni.txt | 감시 bo40izja9 → 사전 kill 판정(26000 평가는 중간값, 판정 아님) |
| 1006 | RD 재배치(rd3.sh PID 2270408, rd2.sh 2250343 종료): rd5 끝 → rd5 KE 평가 → W2om 학습 끝 → rd3 학습(GPU 2 학습 한도) → rd3 평가 | "RD 끝" in scratchpad/rd.out | ground_recovery_rd{5,3}d* | 감시 bbk2ox40x 유지, OOM 감시 b7ct0id8x 폐기 |
| 1006 | rd5 평가 1차 실패(평가가 biped12_stage4 에서 run 탐색 → 링크 없음, Traceback) → 링크 생성 후 재평가 rd5ev.sh PID 2294705(+rd3 링크 자동) | "rd5 재평가 끝" in scratchpad/rd5ev.out | ground_recovery_rd5d* | 감시 bcusnyv30; ⚠ rd5d4_sf15_s1 로그 PhysX 할당 실패 → 재실행 필요 |
| 1006 | W2omA 1차 기동 PhysX OOM(ABORT 뒤에도 학습 진행, 착지 0.000) → 무효, 명시 PID 종료·scratchpad/dead_w2omA_physx_oom 격리 → w2oma2.sh PID 2303107(rd5 평가·W2om 끝 + GPU ≤7.5 GB 대기, 할당 실패 시 재시도) | "W2omA 학습 시작(정상)/평가 끝" in scratchpad/w2oma.out | walk_eval_w2omA_*_omni·w2om_27000_omni_ctl | 감시 추가 예정 |
| 1006 | W2omA 2차(정상 기동 04:14) | — | +300 KILL(하중 수용 0.167) → 명시 PID 종료 → DECISION(LOG) | — |
| 1006 | 평가: w2omA_26400_omni·w2om_27000_omni_ctl(w2oma2.sh) + 진단 α=1 강제 w2omA_26400_a1diag(diag_a1.sh PID 2330665) | "W2omA 평가 끝"·"진단 a1 끝" | walk_eval_*.txt | 감시 bjn2h2llm |
| 1006 | rd3 학습(rd3.sh, 04:0x 시작) → rd3 KE 평가 → "RD 끝" | rd.out | ground_recovery_rd3d* | 감시 bbk2ox40x(짝 비교) |
| 1006 | DIRCAL(dircal.sh; 1차는 --cmd_set 음수 인자 argparse 오류로 40분 대기 위험 → PID 종료·수정 재기동) | "DIRCAL 끝" | walk_eval_dircal_{A,B} | COMPLETED → ANALYZED → DECISION(옆 B 채택, 뒤·회전 기각) → NEXT=W2omL |
| 1006 | W2omL 학습+평가(w2oml.sh PID 2345967, GPU 가드·기동 실패 재시도) | "W2omL 학습 시작(정상)/학습 끝/평가 끝" in scratchpad/w2oml.out | walk_eval_w2omL_*_omni | 감시 bqtx6ccn7(+300 블록) |
| 1006 | RD | — | COMPLETED → ANALYZED → DECISION(사전 성공 미달·kill 아님, KE 유지; 이어학습 표류 발견) → NEXT=RDS | — |
| 1006 | RDS 선별(rds.sh PID 2350685: rd5 21900/22100/22300/22500 × sf15·fi40 × seed 1, d3) | "RDS 선별 끝" in scratchpad/rds.out | ground_recovery_rds*d3_* | 감시 bv3w6plul |
| 1006 | RDS | — | COMPLETED → KILL(통과 0) → RD 계보 종료 | — |
| 1006 | W2omL | — | +300 KILL(하중 수용 0.001) → 명시 PID 종료(w2oml.sh 는 끝 체크포인트 정보용 평가로 진행) | — |
| 1006 | 서브에이전트 2(보행 구조 검토 · 회복 파이프라인 정렬 학습 설계) | 완료 알림 | 대화 내 보고 | 결과 대조 후 선별 반영 |
| 1006 | STEER 진단(steer.sh PID 2358420: W2q 25590 × TURN_ALPHA 0/1/2) | "STEER 끝" in scratchpad/steer.out | walk_eval_steer_a*.txt | 감시 b7vyg0dnk |
| 1006 | STEER | — | COMPLETED(α0 재실행 포함) → ANALYZED → DECISION(부분 효과, 채택·기각 아님) | — |
| 1006 | W2qT | — | 시작 전 철회(WT1 중복) | — |
| 1006 | RH2(rh2.sh PID 2365498: G24RaR4 이어학습 1000 → KE d3·d4 + 22300) | "RH2 끝" in scratchpad/rh2.out | ground_recovery_rh2* | 감시 b1jk9tu6e |
| 1006 | WSTEER(wsteer.sh PID 2368272): W2qS 학습(05:12 정상) + 25900 평가 + 대조 C(W2q 이어학습, RH2 학습 끝 뒤) + 끝 평가(α1 s1·s2, α0 s1) | "WSTEER 300 평가 끝"·"WSTEER 끝" in scratchpad/wsteer.out | walk_eval_wsteer_*·w2qC_* | 감시 +300 bdtb… (아래) |
| 1006 | PA1 구현 서브에이전트(worktree 격리 — 실행 중 학습이 쓰는 train.py 보호) | 완료 알림 | 브랜치 커밋 | 검토 후 병합·스모크 확인 |
| 1006 | WSTEER +300 평가 수동(태그 _s1m) → kill 아님, 계속 | — | walk_eval_wsteer_*_25900_a1.0_s1m | DECISION 기록 |
| 1006 | 메모리 경합 방지: wsteer.sh 하위 서브셸 2368274(+300 평가 대기)·2368275(대조 C 대기) 명시 PID 종료 → seqC.sh PID 2387614(RH2 끝 → 대조 C 학습 → 평가 α1 s1·s2, α0 s1) | "대조 C 평가 끝" in scratchpad/seqC.out | walk_eval_w2qC_* | 감시 추가 |
| 1006 | RH2 | — | COMPLETED → KILL(sf15 +13 pp) → H2 기각 → NEXT=PA1 | — |
| 1006 | PA1 병합(39653e9f, worktree 브랜치 cherry-pick) | — | — | — |
| 1006 | PA1(pa1.sh PID 2410779: GPU ≤ 7.5 GB 대기 → 학습 1000 → 22300 d3 평가 → 끝 d3·d4) | "PA1 학습 시작/22300 평가 끝/PA1 끝" in scratchpad/pa1.out | ground_recovery_pa1* | 감시 b7zadq0nc(초기 신호), bjdfqw9m0(22300 kill 판정) |
| 1006 | 대조 C 학습 시작 06:03(seqC.sh) | "대조 C 평가 끝" | walk_eval_w2qC_* | 감시 b0txw20d2(WSTEER·C 최종) |
| 1006 | PA1 | — | 22300 KILL → H1 미지지 → NEXT=WRAPFIT | — |
| 1006 | 대조 C | — | 학습 붕괴(대조 무효) → WSTEER KILL 유지 + 사후 진단 26300 | — |
| 1006 | diag2.sh PID 2446326: PA1 끝 평가 뒤 WRAPFIT(rd3·rh2·pa1 × C5 래퍼 없음) → WS26300(WSTEER·C α1/α0) | "WRAPFIT 끝"·"WS26300 끝" in scratchpad/diag2.out | ground_recovery_wf*·walk_eval_*_26300_* | 감시 bz32m027q, (WS26300) |
| 1006 | CSTAB(cstab.sh PID 2447388: W2q 이어학습 seed 2, 1000) | "CSTAB 끝" in scratchpad/cstab.out | train_cstab.log 에피소드 길이 | 감시 bs6dp647h |
| 1006 | 지연 감소 구현 서브에이전트(worktree, 젯슨·실CAN 금지, vcan 벤치) | 완료 알림 | 브랜치 커밋 | 검토 후 병합(배포는 승인 대상) |
| 1006 | HTURN(hturn.sh PID 2470347: W2q 25590 + HeadingHold 목표 ±45/±90° × α 1.5/0) | "HTURN 끝" in scratchpad/hturn.out | walk_eval_hturn_a*.txt | 감시 bvdj5x2ml |
| 1006 | HTURN·HTURN2 | — | COMPLETED → ANALYZED → HTURN2 통과(후보) → NEXT=HTURN2-C | — |
| 1006 | HTURN2-C(hturnc.sh PID 2476625: seed 2·3 d3, seed 2 d4·d2) | "HTURN2-C 끝" in scratchpad/hturnc.out | walk_eval_hturn2c_* | 감시 bo5ct7zmi |
| 1006 | HTURN2-C·CSTAB | — | COMPLETED → ANALYZED → DECISION(견고성 미확인 — 왼 회전 약점 / 확률적 붕괴) → NEXT=WSTEER2 | — |
| 1006 | WSTEER2(wsteer2.sh PID 2482537: W2qS seed 1·2 × 600 → 선택 평가 26100/26189 seed1 d3) | "WSTEER2 선택 평가 끝" in scratchpad/wsteer2.out | walk_eval_ws2sel_* | 감시 b30dinzhz → 선택 후 확인 4 run 별도 기동 |
| 1006 | WSTEER2 | — | COMPLETED → KILL(+45° 기준선 미달) | — |
| 1006 | 지연 감소 구현 병합(73889d5f) | — | 오프라인 검증 완료, 배포 승인 대기 | — |
| 1006 | DLYGAIN(dlygain.sh PID 2505693: KE d2 전 조건 + d4 fs80·fi40) | "DLYGAIN 끝" in scratchpad/dlygain.out | ground_recovery_fked2_*·fked4_f* | 감시 brb3j51gx |
| 1006 | SRTURN(srturn.sh PID 2506895: vx 0.03/0.05 + HTURN2 ±90°) | "SRTURN 끝" in scratchpad/srturn.out | walk_eval_srturn_v*.txt | 감시 bbrmnuyld |
| 1006 | SRTURN | — | COMPLETED → 기각 → NEXT=W2omM | — |
| 1006 | W2omM(w2omm.sh PID 2513099: 600 반복 → 26700·끝 평가 α_turn 1.5) | "W2omM 학습 시작/끝/W2omM 끝" in scratchpad/w2omm.out | walk_eval_w2omM_* | 감시 btopb2u94(+300 블록) |
| 1006 | DLYGAIN | — | COMPLETED → ANALYZED → DECISION(지연 단축 배포 근거, 승인 요청 대상) | — |
| 1006 | 콘솔 STAGE8_LATENCY_OPTS(5814c5c7, 로컬) | — | 테스트 51 passed | — |
| 1006 | W2omM | — | +300 KILL(걸음 억제), 명시 PID 종료 | — |
| 1006 | A0MT(a0mt.sh: A0m 14299 × omni 게이트 × α 0/1.5) | "A0MT 끝" in scratchpad/a0mt.out | walk_eval_a0mt_a*.txt | 감시 btqo17yor |
| 1006 | A0MT | — | 기각(vx 관측이 걸음 촉발) → NEXT=W2omP | — |
| 1006 | W2omP(w2omp.sh PID 2524339: A0m 14299 → 600 반복 → 끝 평가) | "W2omP 학습 시작/끝/W2omP 끝" | walk_eval_w2omP_* | 감시 bgqzqiv0t(+300 블록) |
| 1006 | W2omP | — | 끝 KILL(α 0 제자리 cadence 0; α 0.5 진단 0.36–0.49 Hz) → NEXT=W2omP2 | — |
| 1006 | W2omP2(w2omp2.sh PID 2534312: W2omP 14898 → 1000, α 단계 이어받기) | "W2omP2 학습 시작/끝/W2omP2 끝" | walk_eval_w2omP2_* | 감시 bdpihsz6f(+500 블록) |
| 1006 | W2omP-s2 재현(w2omps2.sh PID 2535099) | "W2omP-s2 학습 시작/끝" | train_w2omPs2.log 블록 | 감시 bkcne9ndn(+300) |
| 1006 | W2omP3(w2omp3.sh PID 2544606: GPU 슬롯 대기 → 1000 반복 상대 미세 감쇠) | "W2omP3 학습 시작/끝/W2omP3 끝" | walk_eval_w2omP3_* | 감시 buoofsyit(+500) |
| 1006 | W2omP3 | — | KILL(α 비례 — 보조 산물), 명시 PID 종료 | — |
| 1006 | W2omI(w2omi.sh PID 2559147: A0m → 600, 14600·끝 α0 평가) | "W2omI 14600 평가 끝"·"W2omI 끝" | walk_eval_w2omI_* | 감시 b5gs3pg6q(+300) |
| 1006 | W2omI | — | +300 KILL, 명시 PID 종료 | — |
| 1006 | 제자리 걸음 실패 원인 독립 진단(서브에이전트, 읽기·분석) | 완료 알림 | 대화 내 보고 | 결과로 다음 보행 실험 결정 |
| 1006 | DLYGAIN2(dlygain2.sh PID 2566183: KE sf12·bi30 × d2·d3) | "DLYGAIN2 끝" | ground_recovery_fked{2,3}_{sf12,bi30}_* | 감시 bzj91gw8a |
| 1006 | 진단자 보고 | — | ANALYZED → NEXT=W2omR1·R2 | — |
| 1006 | W2omR1·R2(w2omr.sh PID 2573842: 병렬 600 반복 → 15200·끝 α0 평가) | "W2omR 끝" in scratchpad/w2omr.out | walk_eval_w2omR*_* | 감시 brfi70r76 |
| 1006 | W2omR1·R2 | — | KILL 둘 다(원장 부분 지지) → NEXT=W2omF | — |
| 1006 | W2omF(w2omf.sh PID 2592682: 서기 클록 정지, 600 → 15200·끝 α0 평가, 서기 인계 없음) | "W2omF 끝" | walk_eval_w2omF_* | 감시 bxer8jw62 |
| 1006 | W2omF 재기동(w2omf.sh PID 2595606, 1차 NameError) | "W2omF 끝" | walk_eval_w2omF_* | 감시 bc5sfz1xc |
| 1006 | W2omF | — | KILL(학습 붕괴), 새 방향 이어학습 계열 중단 | — |
| 1006 | LANDSCAPE(landscape.sh PID 2612777: 5 정책 반사실 보상 재생) | "LANDSCAPE 끝" | log_picture/landscape_*.{txt,csv} | 감시 bw9xzexcn |
| 1006 | 서브에이전트 5(감사·구조조사✓·A0m 분석·지연 배포 준비·HTURN2 플래그) | 완료 알림 | — | — |
| 1006 | 보상 지형 배치 | — | COMPLETED → ANALYZED → DECISION(W3 착수) | — |
| 1006 | 클록 대조(G21a 서기: 정상/0,0/정지) | — | 57/57/58 % — 서기 = (0,0) 게이트 안전 | — |
| 1006 | 병합: 지연 1단계 도구(517aed4d)·HTURN2 플래그(6b21120f)·확장 로더(aa39791f)·W3 코어(8a4a8885) | — | — | — |
| 1006 | 트랙 B LPFR(lpfr.sh PID 2639886: KE × 체인 β 0.5/0.4/1.0 × d3 × 7조건) | "LPFR 끝" | ground_recovery_lpf*d3_* | 감시 bk7i8t3zn |
| 1006 | 트랙 A W3 단위시험(w3unit.sh PID 2640739) | "W3UNIT 끝" | landscape_u_* | 감시 bj54x9yd2 |
| 1006 | 서브에이전트: 회복 병목(진행)·W3 채점표 도구(진행) | 완료 알림 | — | — |
| 1006 | 트랙 B ARREST-HO(arrest.sh PID 2642866: LPFR 끝 뒤 aho3·aho2) | "ARREST 끝" | ground_recovery_aho*_* | 감시 b8wsu9ivx(stages_cmp) |
| 1006 | W3 단위시험 1·2차 | — | COMPLETED → ANALYZED → DECISION(높이 공중 한정·가중 3, 토너먼트 착수) | — |
| 1006 | W3 Stage0 토너먼트(w3tour.sh PID 2662103: C·A 병렬 → U, 각 400 → eval_w3) | "W3TOUR 끝" | log_picture/w3eval_w3{C,A,U}_400*.txt | 감시 bidpc106v |
| 1006 | LPFR | — | COMPLETED → KILL(LPF 완화 전 조건 악화) → NEXT = ARREST-HO(자동 대기 중) | — |
| 1006 | ARREST-HO | — | COMPLETED → ANALYZED → DECISION(채택 아님, 정지는 학습 능력 문제) | — |
| 1006 | 트랙 B SWMON(swmon.sh PID 2682338: d4·d3 반복 전환 감시) | "SWMON 끝" | ground_recovery_smon*_* | 감시 bqgpkea6v |
| 1006 | 서브에이전트: 회복 정지 구조(포획점) 설계 | 완료 알림 | — | — |
| 1006 | W3-C 400 | — | COMPLETED → ANALYZED(참조 켬 교대 93 %·참조 0 소멸) → NEXT = W3-CI | — |
| 1006 | W3-A 재실행(w3A_rerun.sh PID 2684025, 1차 GPU 할당 실패) | "W3A 끝" | w3eval_w3A_400 | 감시 b9p9vakrk(+U) |
| 1006 | W3-CI(w3ci.sh PID 2685179: W3-C 400 → 1000, 참조 내재화) | "W3CI 500 끝"·"W3CI 끝" | w3eval_w3CI_* | 감시 bq1w4g1x1 |
| 1006 | 트랙 B SGATE(sgate.sh PID 2686659: SWMON 끝 뒤 sg3·sg2) | "SGATE 끝" | ground_recovery_sg*_* | 감시 bwm90n5b3 |
| 1006 | SWMON·SGATE 재기동(옵션 충돌 수정: swmon PID 2690331, sgate PID 2690332) | "SWMON 끝"→"SGATE 끝" | smon*/sg* | 감시 byy4iwhwq·bwm90n5b3 |
| 1006 | W3-U | — | KILL(낙상 100 %, 학습 붕괴) | — |
| 1006 | SWMON | — | 효과 없음(미발동) → 중단 | — |
| 1006 | W3-A 400 | — | 보조 0 자립 제자리 교대(4/7 통과) → NEXT = W3-A2 | — |
| 1006 | W3-A2(w3a2.sh PID 2720822: W3-A 400 → 1000, 보조 → 0 + 요 벌점) | "W3A2 500 끝"·"W3A2 끝" | w3eval_w3A2_* | 감시 b5fzmogfy |
| 1006 | SGATE | — | KILL(효과 없음) → 인계 계열 폐기 → NEXT = 회복 구조 (c) 설계 | — |
