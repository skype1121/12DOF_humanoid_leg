#!/bin/bash
# K1b(W3_FINAL_SELECT.md §8 O2): 새 seed 5–6 왕복 짝
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
EX=logs/rsl_rl/biped12_stepb0/2026-10-07_23-42-16_w3S3tK_from_15800/exported_16299
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=0.3 EVAL_YIELD=0
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
SFF="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for SD in 5 6; do
  run tr_s3tk16299_ff_d2_s${SD} $CM --kw_script "$SFF" --after_s 48 --seed $SD --kw_w3_export $EX
  run tr_c0x_ff_d2_s${SD} $CM --kw_script "$SFF" --after_s 48 --seed $SD
done
echo "$(date +%H:%M) S3TKC_K1B_DONE"
