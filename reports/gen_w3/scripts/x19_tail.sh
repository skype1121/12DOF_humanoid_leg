#!/bin/bash
# X19 남은 seed 2 실행을 뒤에서부터(PW → PL → D2F5) — 앞 체인과 겹치지 않게
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=0.5 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 58 --delay_steps 2,2 --trace_all 128"
run tr_x19_PW_d2_s2 $CM --kw_script "$S14" --seed 2 --contact_drop "L:35:2"
run tr_x19_PL_d2_s2 $CM --kw_script "$S14" --seed 2 --contact_drop "L:30:40"
run tr_x19_D2F5_d2_s2 $CM --kw_script "$S14" --seed 2 --contact_pert "delay:2;flip:0.05"
echo "$(date +%H:%M) X19_TAIL_DONE"
