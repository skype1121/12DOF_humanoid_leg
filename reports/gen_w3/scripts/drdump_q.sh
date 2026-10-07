#!/bin/bash
# 개체별 무작위화 값 덤프(판정 없는 측정): seed 1·2, 128 env — X7·X7b 와 같은 개체. 짧은 실행(걷기 한 번).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwe.py EVAL_TIMEOUT=1500
for SD in 1 2; do
  $S/safe_tr.sh tr_drdump_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@2:0.06,0,0;stop@6" --after_s 12 --delay_steps 2,2 --seed $SD --trace_all 128
done
echo "$(date +%H:%M) DRDUMP_DONE"
