#!/bin/bash
# W3S3fF-16299 최종 후보 판정(W3_FINAL_SELECT.md §3 S1–S5, §11 등록) — 세 갈래 병렬: A = S1(X7) → S5(X14) · B = S3(E5) → 왕복 ff(보고) · C = S2(E4) → S4(E6)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
RN=2026-10-08_00-37-31_w3S3fF_from_15800; export EVAL_POLL=0.2 EVAL_YIELD=0 W3C_RUN=$RN
( $S/w3cand_eval2.sh s3ff $RN 16299 "e3"; $S/w3cand_eval1.sh s3ff "16299" "x14"; echo "$(date +%H:%M) S3FFC_A_DONE" ) >> $S/s3ffc/a.out 2>&1 &
( $S/w3cand_eval2.sh s3ff $RN 16299 "e5"; $S/w3cand_eval1.sh s3ff "16299" "ff"; echo "$(date +%H:%M) S3FFC_B_DONE" ) >> $S/s3ffc/b.out 2>&1 &
( $S/w3cand_eval2.sh s3ff $RN 16299 "e4 e6"; echo "$(date +%H:%M) S3FFC_C_DONE" ) >> $S/s3ffc/c.out 2>&1 &
wait; echo "$(date +%H:%M) S3FFC_ALL_DONE"
