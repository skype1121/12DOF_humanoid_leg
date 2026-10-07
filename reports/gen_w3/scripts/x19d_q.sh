#!/bin/bash
# X19d(TRANSITION_W3KE.md §5.22h) — 회복 키 KE(kr 구성)가 왼발 패드 0 고착에서 60 s 서는가·회복 모드 오발동이 느는가. KE 확증 배터리(keq_q.sh)의 nd60 과 같은 구성에 --contact_drop 만 더한다.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py EVAL_TIMEOUT=2700 EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
for SD in ${1:-21 22 23}; do for A in ${2:-KEL KEB}; do
  case $A in KEL) D="L:0:60";; KEB) D="B:0:60";; esac
  run keq_${A}_nd60_d2_s${SD} --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed $SD --after_s 60 --contact_drop "$D" --trace_all 128
done; done
echo "$(date +%H:%M) X19D_DONE"
