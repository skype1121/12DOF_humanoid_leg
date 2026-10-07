#!/bin/bash
# 학습 대기열 3(순서 변경 1008 00:07): S3tK 뒤 → G24RaF(회복, X16) → W3S3fF 순서로 직렬. 근거: 미세 조정 팔들의 왕복 낙상이 기준선보다 나빠(W3S3t 5.5 %) 걷기 묶음의 승격 가능성이 낮아짐 → 하드웨어 문제에 직접 닿는 X16 을 앞으로.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
until grep -q S3TK_TRAIN_DONE $S/s3tk/chain.out 2>/dev/null; do sleep 10; done
$S/ketrain.sh G24RaF kef 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 21793 500 >> $S/kef/chain.out 2>&1
$S/w3train.sh W3S3fF s3ff >> $S/s3ff/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE3_DONE"
