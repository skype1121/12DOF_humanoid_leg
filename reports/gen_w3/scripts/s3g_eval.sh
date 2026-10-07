#!/bin/bash
# W3S3g 평가 1 단계(TRANSITION_W3KE.md §5.19 G1–G4): 학습 끝 대기 → 16299·16100 내보내기 → S3E 의 E1 과 같은 왕복 대본(256 env, seed 1–2).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
NM=${1:-s3g}; TAG=$(echo $NM | tr a-z A-Z); CKS=${2:-"16299 16100"}
until grep -q ${TAG}_TRAIN_DONE $S/$NM/chain.out 2>/dev/null; do sleep 30; done
RN=$(sed 's/^run //' $S/$NM/run.txt); R=logs/rsl_rl/biped12_stepb0/$RN; echo "$(date +%H:%M) ${TAG}_EVAL 시작 run $RN"
for CK in $CKS; do
  [ -f $R/model_$CK.pt ] || { echo "$(date +%H:%M) ${TAG}_EVAL 체크포인트 없음 $CK"; continue; }
  if [ ! -f $R/exported_$CK/policy.pt ]; then
    while [ -d $R/exported ]; do sleep 5; done
    /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $R model_$CK.pt > $S/$NM/export_$CK.log 2>&1
    [ -f $R/exported/policy.pt ] && mv $R/exported $R/exported_$CK || { echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 실패 $CK"; continue; }
    echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 $CK: $(md5sum $R/exported_$CK/policy.pt | cut -c1-8)"
  fi
done
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
SC="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for CK in $CKS; do
  [ -f $R/exported_$CK/policy.pt ] || continue
  for SD in 1 2; do
    run tr_${NM}${CK}_ff_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SC" --after_s 48 --delay_steps 2,2 --seed $SD --trace_all 128 --kw_w3_export $R/exported_$CK
  done
  echo "$(date +%H:%M) ${TAG}_EVAL E1 실행 끝 $CK"
done
echo "$(date +%H:%M) ${TAG}_EVAL1_DONE"
