#!/bin/bash
# AIR1b(reports/gen_w3/AIR_START_1008.md §3) — 민감도: 지지를 빨리 거두면(방출 1·2 s) 공중 구동 팔이 서는가. 판정 불변, 실패 기전(부분 지지 체류) 확인용.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-4} EVAL_YIELD=${EVAL_YIELD:-3}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
for SD in ${1:-1}; do for A in ${2:-Cr1 A2ir1 A2ir2 A2r1}; do
  case $A in
    Cr1)   X="--precheck_s 4.5 --air_start clear:0.05,0.15;hold:3;rel:1 --after_s 40 --kw_script move@33:fwd1";;
    A2ir1) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1 --after_s 43 --kw_script move@36:fwd1 --air_det_hold";;
    A2ir2) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:2 --after_s 44 --kw_script move@37:fwd1 --air_det_hold";;
    A2r1)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1 --after_s 43 --kw_script move@36:fwd1";;
  esac
  run tr_air1_${A}_s${SD} $CM --seed $SD $X
done; done
echo "$(date +%H:%M) AIR1B_DONE"
