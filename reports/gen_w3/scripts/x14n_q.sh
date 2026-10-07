#!/bin/bash
# X14n(TRANSITION_W3KE.md §5.21) — 콘솔 버튼 대본을 **젯슨에 배포된 그대로의 합성 키**(감지기 갇힘 해소 옵션 없음)로. X14 와 같은 대본·seed·env, --kw_requiet 만 뺀다.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.3} EVAL_YIELD=${EVAL_YIELD:-0}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
for SD in 1 2; do
  run tr_x14n_d2_s${SD} --ke_mode direct --runner_loop --kw_script "$S14" --after_s 58 --delay_steps 2,2 --seed $SD --trace_all 128
done
echo "$(date +%H:%M) X14N_DONE"
