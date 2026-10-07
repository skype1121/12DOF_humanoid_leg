#!/bin/bash
# X19b(TRANSITION_W3KE.md §5.22c) — 켜기 선행조건의 "양발 접촉 연속 시간"을 허용 범위 안에서 줄이면(1.0 → 0.5·0.3 s) 채터에서 수락이 돌아오는가, 그 대가는. 버튼 대본, 256 env, seed 1–2.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.5} EVAL_YIELD=${EVAL_YIELD:-1}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 58 --delay_steps 2,2 --trace_all 128"
for SD in ${X19B_ORDER:-1 2}; do for CS in ${X19B_CS:-03 05}; do
  run tr_x19b_S${CS}C_d2_s${SD} $CM --kw_script "$S14" --seed $SD --kw_safety "{\"enable_contact_s\": 0.${CS#0}}"
  run tr_x19b_S${CS}F2_d2_s${SD} $CM --kw_script "$S14" --seed $SD --kw_safety "{\"enable_contact_s\": 0.${CS#0}}" --contact_pert "flip:0.02"
done; done
echo "$(date +%H:%M) X19B_DONE"
