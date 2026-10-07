#!/bin/bash
# (가) W3S3fF-16100: X13(P1–P4) → X7(S1) — W3_FINAL_SELECT §11d   (나) 상한 0.8 배 X7 왕복: 기준선 대 W3S3fF-16299 — §12
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
RN=2026-10-08_00-37-31_w3S3fF_from_15800; EX=logs/rsl_rl/biped12_stepb0/$RN/exported_16299
( export EVAL_POLL=0.2 EVAL_YIELD=0 W3C_RUN=$RN; $S/w3cand_eval1.sh s3ff "16100" "x13"; $S/w3cand_eval2.sh s3ff $RN 16100 "e3"; echo "$(date +%H:%M) S3FF16100_DONE" ) >> $S/s3ffc/d.out 2>&1 &
( export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=0.5 EVAL_YIELD=1
  ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
  run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
  CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128 --kw_effort_scales 0.8"
  SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
  for SD in 1 2; do
    run tr_s3ff16299_cap08_x7_d2_s${SD} $CM --kw_script "$SO" --after_s 48 --seed $SD --kw_w3_export $EX
    run tr_base_cap08_x7_d2_s${SD} $CM --kw_script "$SO" --after_s 48 --seed $SD
  done; echo "$(date +%H:%M) CAP08_X7_DONE" ) >> $S/s3ffc/e.out 2>&1 &
wait; echo "$(date +%H:%M) S3FF_MORE_DONE"
