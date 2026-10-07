#!/bin/bash
# W3S3fF-16299 — 토크 한계 0.8 배를 전 구간(서기 KE·걷기·전환)에 건 X14 버튼 대본·X7 왕복(W3_FINAL_SELECT.md §11 의 "승격되면 추가로 볼 것"). 256 env, seed 1–2.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
RN=2026-10-08_00-37-31_w3S3fF_from_15800; EX=logs/rsl_rl/biped12_stepb0/$RN/exported_16299
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.5} EVAL_YIELD=${EVAL_YIELD:-1}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128 --kw_w3_export $EX --kw_effort_scales ${CAP:-0.8}"
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
T=cap$(echo ${CAP:-0.8} | tr -d .)
for SD in 1 2; do
  run tr_s3ff16299_${T}_x14_d2_s${SD} $CM --kw_script "$S14" --after_s 58 --seed $SD
  run tr_s3ff16299_${T}_x7_d2_s${SD} $CM --kw_script "$SO" --after_s 48 --seed $SD
done
echo "$(date +%H:%M) S3FF_CAP_DONE ${CAP:-0.8}"
