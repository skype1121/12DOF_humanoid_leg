#!/bin/bash
# X14b(TRANSITION_W3KE.md §5.18c) — 짝수 반주기에서 끝나는 버튼 변형(오른쪽 옆걸음·후진·제자리·회전, 기한 1.5 s = 두 주기). 256 env, seed 1–2. 버튼 표는 건드리지 않고 대본에 값을 직접 준다.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.3} EVAL_YIELD=${EVAL_YIELD:-0}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
SC="mv@2:0,-0.035,0,1.5;mv@8:-0.04,0,0,1.5;mv@14:0,0,0,1.5;mv@20:0,0,0.15,1.5;mv@26:0,0,-0.15,1.5;mv@32:0,-0.035,0,1.5;mv@38:-0.04,0,0,1.5;mv@44:0.06,0,0,0.6;mv@49:0,-0.035,0,1.5;mv@55:-0.04,0,0,1.5"
for SD in 1 2; do
  run tr_x14b_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SC" --after_s 61 --delay_steps 2,2 --seed $SD --trace_all 128
done
echo "$(date +%H:%M) X14B_DONE"
