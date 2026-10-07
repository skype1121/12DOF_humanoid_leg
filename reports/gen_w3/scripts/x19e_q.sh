#!/bin/bash
# X19e(TRANSITION_W3KE.md §5.22j) — 시뮬 전용 수정안: 감지기 기준 자세 갱신의 접촉 요구를 '양발' → '한 발 이상'. 배포 감지기 불변.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--dir ke2w3 --ke_mode none --delay_steps 2,2 --trace_all 128 --sw_ref_contact any"
for SD in ${1:-21 22 23}; do
  EVAL_TIMEOUT=2700 run keq_KELr_nd60_d2_s$SD $CM --seed $SD --after_s 60 --contact_drop "L:0:60"
  EVAL_TIMEOUT=2700 run keq_KEr_nd60_d2_s$SD $CM --seed $SD --after_s 60
  EVAL_TIMEOUT=1500 run keq_KEr_b18_d2_s$SD $CM --seed $SD --dist "push:b:4:18:0.3"
  EVAL_TIMEOUT=1500 run keq_KEr_sf15_d2_s$SD $CM --seed $SD --dist "ramp:f:4:15:0.5:1.0"
done
echo "$(date +%H:%M) X19E_DONE"
