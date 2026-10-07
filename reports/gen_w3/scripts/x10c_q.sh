#!/bin/bash
# X10c 탐색(reports/gen_w3/ANKLE_RATE_X10.md §2b) — 전 관절 슬루 3·2°/틱, KE 단독, seed 21.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
for A in 3 2; do
  run x10_A${A}_b18_d2_s21 --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed 21 --dist push:b:4:18:0.3 --slew_deg $A --trace_all 128
  run x10_A${A}_sf15_d2_s21 --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed 21 --dist ramp:f:4:15:0.5:1.0 --slew_deg $A --trace_all 128
done
echo "$(date +%H:%M) X10C_DONE"
