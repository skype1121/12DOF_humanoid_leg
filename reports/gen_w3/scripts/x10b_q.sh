#!/bin/bash
# X10 보조: 제한 없는 팔(R0)을 관절각·토크 추적 포함으로 다시(통계용 — 결과 판정에는 기준선 keq_KE 를 쓴다).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
run x10_R0t_b18_d2_s21 --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed 21 --dist push:b:4:18:0.3 --trace_all 128
run x10_R0t_sf15_d2_s21 --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed 21 --dist ramp:f:4:15:0.5:1.0 --trace_all 128
echo "$(date +%H:%M) X10B_DONE"
