#!/bin/bash
# X7d(reports/gen_w3/TRANSITION_W3KE.md §5.11e) — 자세 되돌리기 개입(시뮬 전용 진단). S_ff 대본, nd, 갇힘 해소 옵션, seed 1–2, 256 env. 대조 = X7c 의 C0.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwd.py EVAL_TIMEOUT=3000 EVAL_ENVS=256
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S0="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for SD in 1 2; do
  run tr_x7d_d1_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 48 --delay_steps 2,2 --trace_all 128 --kw_script "$S0" --kw_restore_at 8.5,17.5,26.5,35.5 --seed $SD
done
echo "$(date +%H:%M) X7D_DONE"
