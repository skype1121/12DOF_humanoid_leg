#!/bin/bash
# W3 후보 2 단계 평가(S3E_DESIGN.md §4 E3–E6 와 같은 구성) — 사용: w3cand_eval2.sh <이름표(s3e)> <run 폴더 이름> <체크포인트 번호> [단계 목록 "e3 e4 e5 e6"]
# E3 X7 왕복 대본(256 env, seed 1–2) · E4 단일 걷기 회귀(eval_w3 9 명령) · E5 전환 회귀(X6b 조건, 128 env, seed 1–3) · E6 걷는 중 외란(Q6 조건, seed 1–2).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
NM=$1; RN=$2; CK=$3; STAGES=${4:-"e3 e4 e5 e6"}; R=logs/rsl_rl/biped12_stepb0/$RN; EX=$R/exported_$CK
[ -f $EX/policy.pt ] || { echo "내보내기 없음 $EX"; exit 2; }
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
export EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-3}
for st in $STAGES; do case $st in
e3) export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256
    SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
    for SD in 1 2; do run tr_${NM}${CK}_x7_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SO" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128 --kw_w3_export $EX; done
    unset EVAL_ENVS; echo "$(date +%H:%M) ${NM}_E3_DONE $CK";;
e4) for c in "tl01:0,0,0.1" "tr01:0,0,-0.1" "tl02:0,0,0.2" "tr02:0,0,-0.2" "c0:0,0,0" "l05:0,0.05,0" "f06:0.06,0,0"; do
      [ -s log_picture/w3eval_${NM}_${CK}_${c%%:*}.txt ] || $S/safe_eval.sh Biped12-HangStand-W3S3d-v0 $RN $CK ${NM}_${CK}_${c%%:*} --cmd=${c#*:} --turn_metrics
    done
    [ -s log_picture/w3eval_${NM}_${CK}_d4f06.txt ] || $S/safe_eval.sh Biped12-HangStand-W3S3d-v0 $RN $CK ${NM}_${CK}_d4f06 --cmd=0.06,0,0 --delay 4 --turn_metrics
    echo "$(date +%H:%M) ${NM}_E4_DONE $CK";;
e5) export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=1800
    declare -A X=( [nd]="" [sf15]="--dist ramp:f:16:15:0.5:1.0" [b18]="--dist push:b:16:18:0.3" )
    for SD in 1 2 3; do for C in nd sf15 b18; do run tr_${NM}${CK}_kew3q_${C}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --seed $SD ${X[$C]} --trace_all 128 --kw_w3_export $EX; done; done
    echo "$(date +%H:%M) ${NM}_E5_DONE $CK";;
e6) export TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py EVAL_TIMEOUT=1500
    for SD in 1 2; do
      run tr_${NM}${CK}_walk_q6nd_d2_s${SD} --ke_mode none --w3_walk_only --on_phases 8 --delay_steps 2,2 --seed $SD --trace_all 128 --load_run $RN --checkpoint model_$CK.pt
      for DD in b18:push:b:5:18:0.3 b12:push:b:5:12:0.3 f12:push:f:5:12:0.3 f18:push:f:5:18:0.3; do
        run tr_${NM}${CK}_walk_q6${DD%%:*}_d2_s${SD} --ke_mode none --w3_walk_only --on_phases 8 --delay_steps 2,2 --seed $SD --dist ${DD#*:} --after_s 12 --trace_all 128 --load_run $RN --checkpoint model_$CK.pt
      done
    done
    echo "$(date +%H:%M) ${NM}_E6_DONE $CK";;
esac; done
echo "$(date +%H:%M) ${NM}_EVAL2_DONE $CK"
