#!/bin/bash
# X16(TORQUE_BUDGET_1007.md §13) — 회복 정책 후보의 토크 한계 민감도. 사용: kef_eval.sh <이름표(kef)> "<체크포인트들>" [조건 "b18 sf15 nd"]
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
NM=$1; CKS=$2; CONDS=${3:-"b18 sf15 nd"}; TAG=$(echo $NM | tr a-z A-Z); mkdir -p $S/$NM
until grep -q ${TAG}_TRAIN_DONE $S/$NM/chain.out 2>/dev/null; do sleep 30; done
RN=$(sed 's/^run //' $S/$NM/run.txt); R=logs/rsl_rl/biped12_stepb0/$RN; echo "$(date +%H:%M) ${TAG}_EVAL 시작 run $RN · 체크포인트 $CKS"
for CK in $CKS; do
  [ -f $R/model_$CK.pt ] || { echo "$(date +%H:%M) ${TAG}_EVAL 체크포인트 없음 $CK ($(ls $R | grep model_ | tr '\n' ' '))"; continue; }
  if [ ! -f $R/exported_$CK/policy.pt ]; then
    /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $R model_$CK.pt --out_name exported_$CK > $S/$NM/export_$CK.log 2>&1
    [ -f $R/exported_$CK/policy.pt ] || { echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 실패 $CK"; continue; }
    echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 $CK: $(md5sum $R/exported_$CK/policy.pt | cut -c1-8) onnx $(md5sum $R/exported_$CK/policy.onnx | cut -c1-8)"
  fi
done
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-3}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
declare -A X=( [nd]="" [b18]="--dist push:b:16:18:0.3" [sf15]="--dist ramp:f:16:15:0.5:1.0" )
for CK in $CKS; do
  EX=$R/exported_$CK; [ -f $EX/policy.pt ] || continue
  for SD in 1 2; do for C in $CONDS; do
    run tr_${NM}${CK}_x15_${C}_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@900:0,0,0" --after_s 24 --delay_steps 2,2 --seed $SD ${X[$C]} --kw_effort_scales 1.0,0.8,0.7,0.6 --trace_all 128 --kw_rec_export $EX
  done; done
  echo "$(date +%H:%M) ${TAG}_EVAL x15 끝 $CK"
done
echo "$(date +%H:%M) ${TAG}_EVAL1_DONE"
