#!/bin/bash
# 세대 학습 런치: train_gen.sh <태그> <task> <warm_run> <warm_ckpt> <iters>
TAG=$1; TASK=$2; WRUN=$3; WCKPT=$4; ITERS=$5
cd /home/ryu/humanoid_leg_test1
export PYTHONPATH=/home/ryu/humanoid_leg_test1 OMNI_KIT_ACCEPT_EULA=YES
LOG=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/train_$TAG.log
echo "[$(date '+%F %T')] $TAG 학습 시작 ($TASK, warm $WRUN/$WCKPT, $ITERS it)" | tee -a "$LOG"
/home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --task $TASK --headless \
  --num_envs 4096 --max_iterations $ITERS --resume --load_run $WRUN --checkpoint $WCKPT >> "$LOG" 2>&1
echo "[$(date '+%F %T')] $TAG 학습 프로세스 종료 exit=$?" | tee -a "$LOG"
