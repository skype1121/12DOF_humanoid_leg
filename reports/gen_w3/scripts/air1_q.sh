#!/bin/bash
# AIR1(reports/gen_w3/AIR_START_1008.md) — 합성 키(kew3q 구성)를 공중에서 구동한 채 내려 착지시키면 서는가. 대조 = 홀드로 착지한 뒤 인계.
# 사용: air1_q.sh "<seed 순서>" "<팔 순서>"   (팔: C A2 A6 A2i A6i)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
for SD in ${1:-1 2}; do for A in ${2:-C A2 A2i A6 A6i}; do
  case $A in
    C)   X="--precheck_s 4.5 --air_start clear:0.05,0.15;hold:3;rel:4 --after_s 42 --kw_script move@35.5:fwd1";;
    A2)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1";;
    A2i) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1 --air_det_hold";;
    A6)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1";;
    A6i) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1 --air_det_hold";;
  esac
  run tr_air1_${A}_s${SD} $CM --seed $SD $X
done; done
echo "$(date +%H:%M) AIR1_DONE"
