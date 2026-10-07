#!/bin/bash
# X10 발목 명령 속도 제한(reports/gen_w3/ANKLE_RATE_X10.md) — KE 단독, 사본 평가기.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A ARM=( [R0]="" [R2]="--slew_joint ankle_r:2.0" [R1]="--slew_joint ankle_r:1.0" [RF2]="--slew_joint ankle_r:2.0,ankle_f:2.0" )
cond() { case $1 in b18) echo "--dist push:b:4:18:0.3";; fi40) echo "--dist push:f:4:40:0.2";; sf15) echo "--dist ramp:f:4:15:0.5:1.0";; bb) echo "--dist push:b:4:12:0.3;ramp:f:12:15:0.5:1.0";; esac; }
one() { run x10_${1}_${2}_d2_s${3} --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed $3 $(cond $2) ${ARM[$1]} --trace_all 128; }
for C in b18 sf15; do one R0 $C 21; done
for SD in 21 22; do for C in sf15 b18 fi40 bb; do for A in R2 RF2 R1; do one $A $C $SD; done; done; done
echo "$(date +%H:%M) X10_DONE"
