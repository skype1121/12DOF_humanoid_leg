#!/bin/bash
# W3S3f 평가 1 단계(TORQUE_BUDGET_1007.md §7 F1–F3): 학습 끝 대기 → 16299·16100 내보내기 → X13 대본(토크 한계 배율 1.0/0.8/0.7/0.6, 256 env, seed 1–2, W3 러너만 새 정책).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
until grep -q S3F_TRAIN_DONE $S/s3f/chain.out 2>/dev/null; do sleep 30; done
RN=$(sed 's/^run //' $S/s3f/run.txt); R=logs/rsl_rl/biped12_stepb0/$RN; echo "$(date +%H:%M) S3F_EVAL 시작 run $RN"
for CK in 16299 16100; do
  [ -f $R/model_$CK.pt ] || { echo "$(date +%H:%M) S3F_EVAL 체크포인트 없음 $CK"; continue; }
  if [ ! -f $R/exported_$CK/policy.pt ]; then
    rm -rf $R/exported; /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $R model_$CK.pt > $S/s3f/export_$CK.log 2>&1
    [ -f $R/exported/policy.pt ] && mv $R/exported $R/exported_$CK || { echo "$(date +%H:%M) S3F_EVAL 내보내기 실패 $CK"; continue; }
    echo "$(date +%H:%M) S3F_EVAL 내보내기 $CK: $(md5sum $R/exported_$CK/policy.pt | cut -c1-8)"
  fi
done
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
for CK in 16299 16100; do
  [ -f $R/exported_$CK/policy.pt ] || continue
  for SD in 1 2; do
    run tr_s3f${CK}_x13_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "walk@2:0,0,0;walk@8:0.06,0,0;stop@16" --after_s 24 --delay_steps 2,2 --seed $SD --dist "push:b:20:12:0.3" --kw_effort_scales 1.0,0.8,0.7,0.6 --trace_all 128 --kw_w3_export $R/exported_$CK
  done
  echo "$(date +%H:%M) S3F_EVAL F1 실행 끝 $CK"
done
echo "$(date +%H:%M) S3F_EVAL1_DONE"
