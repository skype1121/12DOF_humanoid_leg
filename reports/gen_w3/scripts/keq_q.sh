#!/bin/bash
# KE 감지기 갇힘 해소 후보 확증 배터리(reports/gen_w3/KE_REQUIET.md §4) — KE 단독, 사본 평가기. 이미 유효한 태그는 건너뜀.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A ARM=( [KE]="" [KEq]="--sw_requiet 2.0,0.3,off" [KEH]="--sw_early_imu 0.6,6,0.4,0.7" [KEHq]="--sw_early_imu 0.6,6,0.4,0.7 --sw_requiet 2.0,0.3,off" )
cond() { case $1 in
  nd60) echo "--after_s 60";;
  b18) echo "--dist push:b:4:18:0.3";; fi40) echo "--dist push:f:4:40:0.2";; sf15) echo "--dist ramp:f:4:15:0.5:1.0";;
  bb) echo "--dist push:b:4:12:0.3;ramp:f:12:15:0.5:1.0";; fb) echo "--dist push:f:4:12:0.3;ramp:f:12:15:0.5:1.0";;
  padL) echo "--contact_drop L:3:19 --dist push:b:4:12:0.3;ramp:f:12:15:0.5:1.0";; esac; }
one() { local A=$1 C=$2 D=$3 SD=$4; local T=1500; [ $C = nd60 ] && T=2700
  EVAL_TIMEOUT=$T run keq_${A}_${C}_d${D}_s${SD} --dir ke2w3 --ke_mode none --delay_steps $D,$D --seed $SD $(cond $C) ${ARM[$A]} --trace_all 128; }
ST=${1:-all}
if [ $ST = main ] || [ $ST = all ]; then
  for SD in 21 22 23; do for C in bb fb sf15 b18 fi40 padL nd60; do for A in KE KEq; do one $A $C 2 $SD; done; done; done
  for SD in 21 22; do for C in bb sf15 nd60; do for A in KE KEq; do one $A $C 3 $SD; done; done; done
fi
if [ $ST = aux ] || [ $ST = all ]; then
  for SD in 21 22 23; do for C in bb fb nd60; do for A in KEH KEHq; do one $A $C 2 $SD; done; done; done
fi
echo "$(date +%H:%M) KEQ_DONE $ST"
