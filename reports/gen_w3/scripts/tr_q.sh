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
if [ $ST = b ] || [ $ST = h ] || [ $ST = all ]; then   # h = hot-swap 팔만(reset/direct)
  for SD in 1 2 3; do for C in nd b18 sf15; do for M in reset direct; do run tr_${M}_${C}_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD ${X[$C]}; done; done; done
fi
if [ $ST = b ] || [ $ST = n ] || [ $ST = all ]; then   # n = nostop(R5)
  for SD in 1 2 3; do for C in nd b18 sf15; do run tr_nostop_${C}_d2_s${SD} --ke_mode nostop --delay_steps 2,2 --seed $SD ${X[$C]}; done; done
fi
if [ $ST = c ] || [ $ST = all ]; then
  for D in 3 4; do for SD in 1 2; do for C in nd b18 sf15; do for M in restart none; do run tr_${M}_${C}_d${D}_s${SD} --ke_mode $M --delay_steps $D,$D --seed $SD ${X[$C]}; done; done; done; done
fi
if [ $ST = d ]; then echo "Q4 생략: Q5(같은 팔 + 위상 분산 + keKE)로 대체 — TRANSITION_W3KE.md"; fi
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
if [ $ST = g ]; then   # 창 외란 보강: 홀드 10 s 무외란 / 홀드 중 뒤 8 N·앞 12 N
  for SD in 1 2; do run tr_restart_h10nd_d2_s${SD} --ke_mode restart --hold_s 10 --after_s 30 --delay_steps 2,2 --seed $SD; done
  for SD in 1 2; do for M in restart none reset; do
    run tr_${M}_hb8_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD --dist push:b:9:8:0.3
    run tr_${M}_hf12_d2_s${SD} --ke_mode $M --delay_steps 2,2 --seed $SD --dist push:f:9:12:0.3
  done; done
fi
if [ $ST = q6a ]; then   # Q6a: 뒤 18 N 뒤 KE 인계 지연 스윕(오라클 방향) +0.4/+0.6/+1.0 s
  for SD in 1 2; do for LT in 04:5.4 06:5.6 10:6.0; do
    run tr_keKE${LT%%:*}_q6b18_d2_s${SD} --ke_mode nostop --on_phases 8 --w3_stop ${LT##*:} --stop_phases 1 --delay_steps 2,2 --seed $SD --dist push:b:5:18:0.3
  done; done
fi
if [ $ST = q6b ]; then   # Q6b: 걷기 추적 수집(개입 없음, trace_all) — 감지기 오프라인 분석용
  export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
  for SD in 1 2; do
    run tr_walk_q6nd_d2_s${SD} --ke_mode none --w3_walk_only --on_phases 8 --delay_steps 2,2 --seed $SD --trace_all 128
    for DD in b18:push:b:5:18:0.3 b12:push:b:5:12:0.3 f12:push:f:5:12:0.3 f18:push:f:5:18:0.3; do
      run tr_walk_q6${DD%%:*}_d2_s${SD} --ke_mode none --w3_walk_only --on_phases 8 --delay_steps 2,2 --seed $SD --dist ${DD#*:} --after_s 12 --trace_all 128
    done
  done
  unset TR_SCRIPT
fi
if [ $ST = x2 ]; then   # X2: stop → ge 0 래치 즉시 KE(정착 대기 0)
  for SD in 1 2 3; do for C in nd b18 sf15; do run tr_latch_${C}_d2_s${SD} --ke_mode reset --settle fixed --settle_s 0.02 --delay_steps 2,2 --seed $SD ${X[$C]}; done; done
fi
if [ $ST = x2nd ]; then   # X2(reset) 무외란 재현 확인만
  for SD in 2 3; do run tr_latch_nd_d2_s${SD} --ke_mode reset --settle fixed --settle_s 0.02 --delay_steps 2,2 --seed $SD; done
fi
if [ $ST = x2d ] || [ $ST = x2dd ]; then   # X2d: 래치 다음 틱 KE, 상태 이어받기 — x2d = 무외란, x2dd = 외란
  CS="nd"; [ $ST = x2dd ] && CS="b18 sf15"
  for SD in 1 2 3; do for C in $CS; do run tr_latchd_${C}_d2_s${SD} --ke_mode direct --settle fixed --settle_s 0.02 --delay_steps 2,2 --seed $SD ${X[$C]}; done; done
fi
if [ $ST = x2e ]; then   # X2e: X2d + 기준 추종 창 3.0 s(H1 대조) — 사본 평가기·추적. 대조 = 같은 사본으로 다시 돌린 X2d·restart 의 sf15
  export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
  LD="--ke_mode direct --settle fixed --settle_s 0.02 --delay_steps 2,2"
  for SD in 1 2 3; do
    run tr_latchd3_sf15_d2_s${SD} $LD --seed $SD ${X[sf15]} --sw_ref_init_s 3.0 --trace_all 128
    run tr_latchdT_sf15_d2_s${SD} $LD --seed $SD ${X[sf15]} --trace_all 128
    run tr_restartT_sf15_d2_s${SD} --ke_mode restart --delay_steps 2,2 --seed $SD ${X[sf15]} --trace_all 128
  done
  for SD in 1 2 3; do for C in nd b18; do run tr_latchd3_${C}_d2_s${SD} $LD --seed $SD ${X[$C]} --sw_ref_init_s 3.0 --trace_all 128; done; done
  unset TR_SCRIPT
fi
if [ $ST = x3 ]; then   # X3: 정착 창에 오는 외란(7.3 s) — A = direct(정착 뒤) / B = latchd / B' = latchd3 / none(W3 STAND)
  declare -A XW=( [b18x]="--dist push:b:7.3:18:0.3" [sf15x]="--dist ramp:f:7.3:15:0.5:1.0" [f12x]="--dist push:f:7.3:12:0.3" )
  LD="--ke_mode direct --settle fixed --settle_s 0.02 --delay_steps 2,2"
  for SD in 1 2; do for C in b18x sf15x f12x; do
    run tr_direct_${C}_d2_s${SD} --ke_mode direct --delay_steps 2,2 --seed $SD ${XW[$C]}
    run tr_latchd_${C}_d2_s${SD} $LD --seed $SD ${XW[$C]}
    run tr_latchd3_${C}_d2_s${SD} $LD --seed $SD ${XW[$C]} --sw_ref_init_s 3.0
    run tr_none_${C}_d2_s${SD} --ke_mode none --delay_steps 2,2 --seed $SD ${XW[$C]}
  done; done
fi
if [ $ST = x4 ]; then   # X4: 깨끗한 KE 서기에서 가벼운 밀기로 회복 모드를 한 번 켠 뒤 앞 램프(H3) — reset 팔, 정지 4 s, 사본 평가기·추적
  export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
  RS="--ke_mode reset --w3_stop 4 --delay_steps 2,2"
  for SD in 1 2 3; do
    run tr_x4C_sf15_d2_s${SD} $RS --seed $SD --dist "ramp:f:16:15:0.5:1.0" --trace_all 128
    run tr_x4Pb_sf15_d2_s${SD} $RS --seed $SD --dist "push:b:9:12:0.3;ramp:f:16:15:0.5:1.0" --trace_all 128
    run tr_x4Pf_sf15_d2_s${SD} $RS --seed $SD --dist "push:f:9:12:0.3;ramp:f:16:15:0.5:1.0" --trace_all 128
  done
  unset TR_SCRIPT
fi
if [ $ST = x5 ]; then   # X5: 회복 모드 갇힘 해소 변형(--sw_requiet 2.0,0.3) — 시험 1 = X4 조건, 시험 2 = direct 표준 조건
  export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py
  RS="--ke_mode reset --w3_stop 4 --delay_steps 2,2 --sw_requiet 2.0,0.3"
  for SD in 1 2 3; do
    run tr_x5Pb_sf15_d2_s${SD} $RS --seed $SD --dist "push:b:9:12:0.3;ramp:f:16:15:0.5:1.0" --trace_all 128
    run tr_x5Pf_sf15_d2_s${SD} $RS --seed $SD --dist "push:f:9:12:0.3;ramp:f:16:15:0.5:1.0" --trace_all 128
    run tr_x5C_sf15_d2_s${SD} $RS --seed $SD --dist "ramp:f:16:15:0.5:1.0" --trace_all 128
  done
  for SD in 1 2 3; do for C in sf15 b18 nd; do run tr_directq_${C}_d2_s${SD} --ke_mode direct --delay_steps 2,2 --seed $SD ${X[$C]} --sw_requiet 2.0,0.3; done; done
  unset TR_SCRIPT
fi
if [ $ST = x6 ]; then   # X6: 배포 코드 합성 러너 폐루프(--runner_loop) — direct 팔과 동등성
  export TR_SCRIPT=rl_walking/scripts/eval_transition_kw.py
  for SD in 1 2 3; do for C in nd sf15 b18; do run tr_kew3_${C}_d2_s${SD} --ke_mode direct --runner_loop --delay_steps 2,2 --seed $SD ${X[$C]} --trace_all 128; done; done
  unset TR_SCRIPT
fi
echo "$(date +%H:%M) TRQ_DONE $ST"
