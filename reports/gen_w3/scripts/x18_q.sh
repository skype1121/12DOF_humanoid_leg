#!/bin/bash
# X18(TORQUE_BUDGET_1007.md §19) — 부분지지(체중의 0/20/35/50 % 를 위로 받침, 이상화)에서 W3 걷기의 부하·걸음. 배포 코드 폐루프(합성 러너), nd, 128 env, seed 21·22. 판정 없는 측정 + 미리 적은 읽는 법.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=2400
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
SC="walk@2:0,0,0;walk@8:0.03,0,0;walk@14:0.06,0,0;walk@20:0.10,0,0;stop@26"
for SD in 21 22; do for U in 00 20 35 50; do
  UA=""; [ $U != 00 ] && UA="--kw_unload 0.$U"
  run x18_u${U}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SC" --after_s 32 --delay_steps 2,2 --seed $SD --trace_all 128 $UA
done; done
echo "$(date +%H:%M) X18_DONE"
