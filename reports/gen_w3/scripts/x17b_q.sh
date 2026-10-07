#!/bin/bash
# X17b(TRANSITION_W3KE.md §5.20c) — soft 한계 3 개를 넓힌 제안 표(deploy/cand/softlimit_1008)로 전 관절 클램프를 다시: KE 회복 b18·sf15, 256 env, seed 1–2. 대조 = X15(클램프 없음)·X17(현 표).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.5} EVAL_YIELD=${EVAL_YIELD:-1}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A X=( [b18]="--dist push:b:16:18:0.3" [sf15]="--dist ramp:f:16:15:0.5:1.0" )
for SD in 1 2; do for C in b18 sf15; do
  run tr_x17b_${C}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@900:0,0,0" --after_s 24 --delay_steps 2,2 --seed $SD ${X[$C]} --kw_effort_scales 1.0,0.8,0.7,0.6 --trace_all 128 --kw_soft_clamp_all --kw_soft_clamp_json deploy/cand/softlimit_1008/joint_limits_12dof.json
done; done
echo "$(date +%H:%M) X17B_DONE"
