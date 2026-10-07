#!/bin/bash
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
until grep -q S3FF_TRAIN_DONE $S/s3ff/chain.out 2>/dev/null; do sleep 15; done
$S/w3train.sh W3S3tK4 s3tk4 >> $S/s3tk4/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE4_DONE"
