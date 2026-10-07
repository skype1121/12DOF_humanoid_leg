#!/bin/bash
# W3 후보 1 단계 평가(공용) — 사용: w3cand_eval1.sh <이름표(s3t)> "<체크포인트들>" "<단계들: x13 ff x14>" [seed 목록 "1 2"]
#   학습 끝(<TAG>_TRAIN_DONE) 대기 → 내보내기 → 단계별 실행(전부 배포 코드 폐루프, W3 러너만 새 정책, 256 env).
#   x13 = 토크 한계 민감도 대본(제자리 6 s → 앞 0.06 8 s → 정지 → 뒤 12 N, 배율 1.0/0.8/0.7/0.6) · ff = 앞 0.06 × 5 주기 왕복 · x14 = 콘솔 버튼 10 번 대본.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
NM=$1; CKS=${2:-"16299 16100"}; STAGES=${3:-"x13 ff"}; SEEDS=${4:-"1 2"}; TAG=$(echo $NM | tr a-z A-Z)
mkdir -p $S/$NM
if [ -n "${W3C_RUN:-}" ]; then RN=$W3C_RUN; else until grep -q ${TAG}_TRAIN_DONE $S/$NM/chain.out 2>/dev/null; do sleep 30; done; RN=$(sed 's/^run //' $S/$NM/run.txt); fi
R=logs/rsl_rl/biped12_stepb0/$RN; echo "$(date +%H:%M) ${TAG}_EVAL 시작 run $RN · 체크포인트 $CKS · 단계 $STAGES"
for CK in $CKS; do
  [ -f $R/model_$CK.pt ] || { echo "$(date +%H:%M) ${TAG}_EVAL 체크포인트 없음 $CK"; continue; }
  if [ ! -f $R/exported_$CK/policy.pt ]; then
    /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $R model_$CK.pt --out_name exported_$CK > $S/$NM/export_$CK.log 2>&1
    [ -f $R/exported_$CK/policy.pt ] || { echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 실패 $CK"; continue; }
    echo "$(date +%H:%M) ${TAG}_EVAL 내보내기 $CK: $(md5sum $R/exported_$CK/policy.pt | cut -c1-8) onnx $(md5sum $R/exported_$CK/policy.onnx | cut -c1-8)"
  fi
done
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-3}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
SFF="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
for st in $STAGES; do for CK in $CKS; do
  EX=$R/exported_$CK; [ -f $EX/policy.pt ] || continue
  for SD in $SEEDS; do case $st in
    x13) run tr_${NM}${CK}_x13_d2_s${SD} $CM --kw_script "walk@2:0,0,0;walk@8:0.06,0,0;stop@16" --after_s 24 --seed $SD --dist "push:b:20:12:0.3" --kw_effort_scales 1.0,0.8,0.7,0.6 --kw_w3_export $EX;;
    ff)  run tr_${NM}${CK}_ff_d2_s${SD} $CM --kw_script "$SFF" --after_s 48 --seed $SD --kw_w3_export $EX;;
    x14) run tr_${NM}${CK}_x14_d2_s${SD} $CM --kw_script "$S14" --after_s 58 --seed $SD --kw_w3_export $EX;;
  esac; done
  echo "$(date +%H:%M) ${TAG}_EVAL $st 끝 $CK"
done; done
echo "$(date +%H:%M) ${TAG}_EVAL1_DONE"
