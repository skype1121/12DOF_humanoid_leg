#!/bin/bash
# 학습 대기열 7(1008 04:45): 묶음 W3S3x(W3S3fF + 무릎 벌점 −4.0), +1000 — W3_FINAL_SELECT.md §16
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; mkdir -p $S/s3x
RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1008 $S/w3train.sh W3S3x s3x 2026-10-07_00-11-12_w3S3d_from_s2d 15800 1000 >> $S/s3x/chain.out 2>&1
echo "$(date +%H:%M) TRAIN_QUEUE7_DONE"
