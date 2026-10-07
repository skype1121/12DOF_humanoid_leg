#!/bin/bash
# W3S3e 평가 1 단계(reports/gen_w3/S3E_DESIGN.md §4 E1–E2): 학습 끝 대기 → 체크포인트 16299·16100 내보내기 → 앞 0.06 × 5 주기 왕복(256 env, seed 1–2, 배포 코드 폐루프, W3 러너만 새 정책).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
until grep -q S3E_TRAIN_DONE $S/s3e/chain.out 2>/dev/null; do sleep 30; done
RN=$(sed 's/^run //' $S/s3e/run.txt); R=logs/rsl_rl/biped12_stepb0/$RN; echo "$(date +%H:%M) S3E_EVAL 시작 run $RN · 체크포인트 $(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tr '\n' ' ')"
for CK in 16299 16100; do
  [ -f $R/model_$CK.pt ] || { echo "$(date +%H:%M) S3E_EVAL 체크포인트 없음 model_$CK.pt"; continue; }
  if [ ! -f $R/exported_$CK/policy.pt ]; then
    rm -rf $R/exported; /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $R model_$CK.pt > $S/s3e/export_$CK.log 2>&1
    [ -f $R/exported/policy.pt ] && mv $R/exported $R/exported_$CK || { echo "$(date +%H:%M) S3E_EVAL 내보내기 실패 $CK (로그 $S/s3e/export_$CK.log)"; continue; }
    echo "$(date +%H:%M) S3E_EVAL 내보내기 $CK: $(md5sum $R/exported_$CK/policy.pt | cut -c1-8) · $(tail -1 $S/s3e/export_$CK.log | cut -c1-120)"
  fi
done
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
S0="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for CK in 16299 16100; do
  [ -f $R/exported_$CK/policy.pt ] || continue
  for SD in 1 2; do
    run tr_s3e${CK}_ff_d2_s${SD} --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 48 --delay_steps 2,2 --trace_all 128 --kw_script "$S0" --kw_w3_export $R/exported_$CK --seed $SD
  done
  echo "$(date +%H:%M) S3E_EVAL E1 실행 끝 $CK"
done
echo "$(date +%H:%M) S3E_EVAL1_DONE"
