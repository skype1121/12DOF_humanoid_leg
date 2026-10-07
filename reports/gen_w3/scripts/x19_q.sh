#!/bin/bash
# X19(TRANSITION_W3KE.md §5.22) — 걷기·합성 구조의 접촉 관측 교란 내성. 콘솔 버튼 대본(X14), 합성 러너(갇힘 해소 켬 = kew3q 구성), 256 env. 교란은 관측 비트에만.
#   C0 대조(seed 3·4 보강 — 1·2 는 X14) · D2 비트 2 틱 추가 지연 · F2 틱당 2 % 뒤집힘 · D2F5 지연 2 + 5 % · PL 왼 패드 30 s 부터 0 고착 · PW 걷는 중(35 s) 왼 패드 2 s 누락
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-0.5} EVAL_YIELD=${EVAL_YIELD:-1}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 58 --delay_steps 2,2 --trace_all 128"
ORD=${X19_ORDER:-"1 2 3 4"}
for SD in $ORD; do
  [ $SD -ge 3 ] && run tr_x14_d2_s${SD} $CM --kw_script "$S14" --seed $SD
  run tr_x19_D2_d2_s${SD} $CM --kw_script "$S14" --seed $SD --contact_pert "delay:2"
  run tr_x19_F2_d2_s${SD} $CM --kw_script "$S14" --seed $SD --contact_pert "flip:0.02"
  run tr_x19_D2F5_d2_s${SD} $CM --kw_script "$S14" --seed $SD --contact_pert "delay:2;flip:0.05"
  if [ $SD -le 2 ]; then
    run tr_x19_PL_d2_s${SD} $CM --kw_script "$S14" --seed $SD --contact_drop "L:30:40"
    run tr_x19_PW_d2_s${SD} $CM --kw_script "$S14" --seed $SD --contact_drop "L:35:2"
  fi
done
echo "$(date +%H:%M) X19_DONE ${ORD// /}"
