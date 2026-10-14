#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
until grep -q "학습 종료" .wd_tmp/train2.out; do sleep 30; done
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S4dDb_from_s4d | tail -1); RN=$(basename $R)
until grep -q "체인 끝" .wd_tmp/chain.out; do sleep 30; done
LAST=$(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
.wd_tmp/evset.sh $RN $LAST e500 full
.wd_tmp/evset.sh $RN 16500 e200 dist
echo "$(date +%H:%M) 체인2 끝"
