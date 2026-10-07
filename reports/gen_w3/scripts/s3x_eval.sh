#!/bin/bash
# 묶음 W3S3x 평가(W3_FINAL_SELECT.md §16 M1): +500 체크포인트는 학습 중에 나오는 대로, +1000 은 학습 끝난 뒤 — X13(seed 1–2)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
until [ -s $S/s3x/run.txt ]; do sleep 20; done; RN=$(sed 's/^run //' $S/s3x/run.txt); R=logs/rsl_rl/biped12_stepb0/$RN
until [ -f $R/model_16299.pt ] || grep -q S3X_TRAIN_DONE $S/s3x/chain.out; do sleep 20; done; sleep 20
EVAL_POLL=0.3 EVAL_YIELD=0 W3C_RUN=$RN $S/w3cand_eval1.sh s3x "16299" "x13"
until grep -q S3X_TRAIN_DONE $S/s3x/chain.out; do sleep 20; done
LAST=$(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1); echo "$(date +%H:%M) S3X 마지막 체크포인트 $LAST"
EVAL_POLL=0.3 EVAL_YIELD=0 W3C_RUN=$RN $S/w3cand_eval1.sh s3x "$LAST" "x13"
echo "$(date +%H:%M) S3X_M1_DONE $LAST"
