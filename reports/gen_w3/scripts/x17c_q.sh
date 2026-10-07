#!/bin/bash
# X17c(TRANSITION_W3KE.md §5.20e) — 전 관절 클램프의 회복 영향을 큰 표본으로: 토크 한계 1.0 배만(256 env 전부), 새 seed 3–6, KE 회복 b18·sf15.
#   팔: c0 클램프 없음(대조) · c1 현 표 · c2 힙 넓힌 제안 표 · c3 발목만(힙·무릎 풀기, 원인 분리용)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-2}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A X=( [b18]="--dist push:b:16:18:0.3" [sf15]="--dist ramp:f:16:15:0.5:1.0" )
declare -A A=( [c0]="" [c1]="--kw_soft_clamp_all" [c2]="--kw_soft_clamp_all --kw_soft_clamp_json deploy/cand/softlimit_1008/joint_limits_12dof.json" [c3]="--kw_soft_clamp_all --kw_soft_clamp_json deploy/cand/softlimit_1008/joint_limits_ankle_only_SIMONLY.json" )
for SD in 3 4 5 6; do for C in b18 sf15; do for K in c0 c1 c2 c3; do
  run tr_x17c_${K}_${C}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@900:0,0,0" --after_s 24 --delay_steps 2,2 --seed $SD ${X[$C]} --trace_all 64 ${A[$K]}
done; done; done
echo "$(date +%H:%M) X17C_DONE"
