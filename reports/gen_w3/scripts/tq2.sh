#!/bin/bash
# 학습 대기열 2: S3tK 뒤 → W3S3fF → G24RaF(회복) 순서로 직렬
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
until grep -q S3TK_TRAIN_DONE $S/s3tk/chain.out 2>/dev/null; do sleep 15; done
$S/w3train.sh W3S3fF s3ff >> $S/s3ff/chain.out 2>&1
$S/ketrain.sh G24RaF kef 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 21793 500 >> $S/kef/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE2_DONE"
