#!/bin/bash
# X7b(reports/gen_w3/TRANSITION_W3KE.md §5.11c) — 마지막 앞 걷기 낙상의 원인 분리. 배포 코드 폐루프, nd, 갇힘 해소 옵션, seed 1–2. 기록이 추가된 사본 평가기(kwn).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=${X7B_SCRIPT:-rl_walking/scripts/eval_transition_kwn.py} EVAL_TIMEOUT=2400
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
SF="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
SB="walk@2:-0.04,0,0;stop@6;walk@11:0.06,0,0;stop@15"
for SD in 1 2; do
  run tr_x7b_bf_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SB" --after_s 20 --delay_steps 2,2 --seed $SD --trace_all 128
  run tr_x7b_ff_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SF" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128
  run tr_x7b_orig_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SO" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128
done
echo "$(date +%H:%M) X7B_DONE"
