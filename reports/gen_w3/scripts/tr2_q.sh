#!/bin/bash
# KE→W3 START 격자(TRANSITION_W3KE.md §1b) — stage: a=지연2 (restart/reset/direct/none × nd/b18w/sf15w × seed 1–3), c=지연3·4 (restart/none × seed 1–2), all
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
dist() { local M=$1 C=$2; local T0=4.7 T2=6.0; [ $M = restart ] && { T0=7.7; T2=9.0; }
  case $C in nd) echo "";; b18w) echo "--dist push:b:$T0:18:0.3";; sf15w) echo "--dist ramp:f:$T0:15:0.5:1.0";; b18s) echo "--dist push:b:$T2:18:0.3";; sf15s) echo "--dist ramp:f:$T2:15:0.5:1.0";; esac; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
ST=${1:-all}
if [ $ST = a ] || [ $ST = an ] || [ $ST = all ]; then
  for SD in 1 2 3; do for M in restart none reset direct; do run k2w_${M}_nd_d2_s${SD} --dir ke2w3 --ke_mode $M --delay_steps 2,2 --seed $SD; done; done
fi
if [ $ST = a ] || [ $ST = ab ] || [ $ST = all ]; then
  for SD in 1 2 3; do for C in b18s sf15s b18w sf15w; do for M in restart none reset direct; do run k2w_${M}_${C}_d2_s${SD} --dir ke2w3 --ke_mode $M --delay_steps 2,2 --seed $SD $(dist $M $C); done; done; done
fi
if [ $ST = c ] || [ $ST = all ]; then
  for D in 3 4; do for SD in 1 2; do for C in nd b18s sf15s; do for M in restart none; do run k2w_${M}_${C}_d${D}_s${SD} --dir ke2w3 --ke_mode $M --delay_steps $D,$D --seed $SD $(dist $M $C); done; done; done; done
fi
echo "$(date +%H:%M) TR2Q_DONE $ST"
