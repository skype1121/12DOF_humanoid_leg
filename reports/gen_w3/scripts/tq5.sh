#!/bin/bash
# 학습 대기열 5(1008 01:35): 재현 학습 — 처치(W3S3tK)·대조(W3S3d) × 학습 seed 2·3, 이어서 용량 반응 W3S3tK4. 직렬.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
W3SEED=2 $S/w3train_sd.sh W3S3tK s3tk2 >> $S/s3tk2/chain.out 2>&1
W3SEED=2 $S/w3train_sd.sh W3S3d s3d2 >> $S/s3d2/chain.out 2>&1
W3SEED=3 $S/w3train_sd.sh W3S3tK s3tk3 >> $S/s3tk3/chain.out 2>&1
W3SEED=3 $S/w3train_sd.sh W3S3d s3d3 >> $S/s3d3/chain.out 2>&1
$S/w3train.sh W3S3tK4 s3tk4 >> $S/s3tk4/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE5_DONE"
