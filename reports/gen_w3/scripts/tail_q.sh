#!/bin/bash
# 마무리 대기열(1008 05:45): ① 기준선의 X7 혼합 왕복을 후보들과 같은 표본(256 env × seed 1–2)으로 ② W3S3fF-16299 의 남은 S5(X14 seed 2)·S4(걷는 중 외란) — 판정은 이미 확정, 기록 완결용
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
( export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=0.5 EVAL_YIELD=1
  ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
  run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
  SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
  for SD in 1 2; do run tr_base_x7_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128 --kw_script "$SO" --after_s 48 --seed $SD; done
  echo "$(date +%H:%M) BASE_X7_DONE" ) >> $S/tr/grid.out 2>&1
RN=2026-10-08_00-37-31_w3S3fF_from_15800
( EVAL_POLL=2 EVAL_YIELD=3 W3C_RUN=$RN $S/w3cand_eval1.sh s3ff "16299" "x14"; EVAL_POLL=2 EVAL_YIELD=3 $S/w3cand_eval2.sh s3ff $RN 16299 "e6"; echo "$(date +%H:%M) S3FF_REST_DONE" ) >> $S/s3ffc/rest.out 2>&1
echo "$(date +%H:%M) TAIL_Q_DONE"
