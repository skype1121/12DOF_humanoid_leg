#!/bin/bash
# HD1(reports/gen_w3/HEADING_SESSION_1007.md §7) — 헤딩 고리 켬 대 끔, X7 대본, nd, 갇힘 해소 옵션, seed 1–3. 사본 평가기 kwh.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwh.py EVAL_TIMEOUT=2400
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for SD in 1 2 3; do
  run tr_hd1on_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_hl on --kw_script "$SO" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128
  run tr_hd1off_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SO" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128
done
echo "$(date +%H:%M) HD1_DONE"
