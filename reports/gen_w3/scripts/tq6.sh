#!/bin/bash
# 학습 대기열 6(1008 03:00): 등록 범위 복원 — 재현 쌍 학습 seed 3(처치 W3S3tK·대조 W3S3d) → 용량 반응 W3S3tK4. 직렬.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
W3SEED=3 $S/w3train_sd.sh W3S3tK s3tk3 >> $S/s3tk3/chain.out 2>&1
W3SEED=3 $S/w3train_sd.sh W3S3d s3d3 >> $S/s3d3/chain.out 2>&1
$S/w3train.sh W3S3tK4 s3tk4 >> $S/s3tk4/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE6_DONE"
