#!/bin/bash
# X15(reports/gen_w3/TORQUE_BUDGET_1007.md §8) — KE(서기·회복)의 토크 한계 민감도. 걷기 없음. 256 env, seed 1–2, 배율 1.0/0.8/0.7/0.6.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A X=( [nd]="" [b18]="--dist push:b:16:18:0.3" [sf15]="--dist ramp:f:16:15:0.5:1.0" )
for SD in 1 2; do for C in b18 sf15 nd; do
  run tr_x15_${C}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@900:0,0,0" --after_s 24 --delay_steps 2,2 --seed $SD ${X[$C]} --kw_effort_scales 1.0,0.8,0.7,0.6 --trace_all 128
done; done
echo "$(date +%H:%M) X15_DONE"
