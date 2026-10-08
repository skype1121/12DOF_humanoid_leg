#!/bin/bash
# AIR1c(reports/gen_w3/AIR_START_1008.md §5) — 진단: 공중에서 다리를 내미는 움직임이 접촉 관측 (0, 0) 때문인가. 접지 확인 전 정책의 접촉 관측을 (1, 1) 로 고정.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128 --air_det_hold --air_contact_one"
for SD in ${1:-1 2}; do for A in ${2:-A2ic A6ic}; do
  case $A in
    A2ic) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1";;
    A6ic) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1";;
  esac
  run tr_air1_${A}_s${SD} $CM --seed $SD $X
done; done
echo "$(date +%H:%M) AIR1C_DONE"
