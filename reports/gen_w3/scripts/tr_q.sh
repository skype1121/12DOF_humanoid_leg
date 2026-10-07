#!/bin/bash
# W3→KE 전환 격자(TRANSITION_W3KE.md §1) — 우선순위 순, 이미 유효한 태그는 건너뜀. 사용: tr_q.sh [stage]  (stage: a=지연2 restart/none, b=지연2 nostop/reset/direct, c=지연3·4, d=Q4, all)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
declare -A X=( [nd]="" [b18]="--dist push:b:16:18:0.3" [sf15]="--dist ramp:f:16:15:0.5:1.0" )
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
ST=${1:-all}
if [ $ST = a ] || [ $ST = all ]; then
  for SD in 1 2 3; do for C in nd b18 sf15; do for M in restart none; do run tr_${M}_${C}_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD ${X[$C]}; done; done; done
fi
if [ $ST = b ] || [ $ST = all ]; then
  for SD in 1 2 3; do for C in nd b18 sf15; do for M in nostop reset direct; do run tr_${M}_${C}_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD ${X[$C]}; done; done; done
fi
if [ $ST = c ] || [ $ST = all ]; then
  for D in 3 4; do for SD in 1 2; do for C in nd b18 sf15; do for M in restart none; do run tr_${M}_${C}_d${D}_s${SD} --ke_mode $M --delay_steps $D,$D --seed $SD ${X[$C]}; done; done; done; done
fi
if [ $ST = d ] || [ $ST = all ]; then
  for SD in 1 2; do
    run tr_restart_q4_d2_s${SD} --ke_mode restart --delay_steps 2,2 --seed $SD --dist push:f:5:12:0.3 --w3_stop 5.3 --stop_phases 1
    run tr_none_q4_d2_s${SD} --ke_mode none --delay_steps 2,2 --seed $SD --dist push:f:5:12:0.3 --w3_stop 5.3 --stop_phases 1
    run tr_walk_q4_d2_s${SD} --ke_mode none --w3_walk_only --delay_steps 2,2 --seed $SD --dist push:f:5:12:0.3 --w3_stop 5.3 --stop_phases 1
  done
fi
if [ $ST = e ]; then   # Q5: 걷는 중 외란 직후 KE 즉시 인계 vs 계속 걷기 vs stop→W3 vs stop→KE (on_phases 8)
  for SD in 1 2; do for DD in f12 b18; do
    [ $DD = f12 ] && DS="push:f:5:12:0.3" || DS="push:b:5:18:0.3"
    run tr_keKE_q5${DD}_d2_s${SD} --ke_mode nostop --on_phases 8 --w3_stop 5.2 --stop_phases 1 --delay_steps 2,2 --seed $SD --dist $DS
    run tr_walk_q5${DD}_d2_s${SD} --ke_mode none --w3_walk_only --on_phases 8 --delay_steps 2,2 --seed $SD --dist $DS
    run tr_stopW3_q5${DD}_d2_s${SD} --ke_mode none --on_phases 8 --w3_stop 5.3 --stop_phases 1 --delay_steps 2,2 --seed $SD --dist $DS
    run tr_stopKE_q5${DD}_d2_s${SD} --ke_mode restart --on_phases 8 --w3_stop 5.3 --stop_phases 1 --delay_steps 2,2 --seed $SD --dist $DS
  done; done
fi
if [ $ST = f ]; then   # 창 외란: 재기동 동결(수동 홀드) 중 뒤 18 N(9 s) — restart(동결 중) vs none(W3 STAND) vs reset(KE 즉시)
  for SD in 1 2; do for M in restart none reset; do run tr_${M}_hb18_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD --dist push:b:9:18:0.3; done; done
fi
echo "$(date +%H:%M) TRQ_DONE $ST"
