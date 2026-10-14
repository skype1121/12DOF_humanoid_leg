#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
until grep -q "학습 종료" .wd_tmp/train.out; do sleep 30; done
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S4dD_from_s4d | tail -1); RN=$(basename $R)
ls $R/model_*.pt | xargs -n1 basename
until ! pgrep -f "evset.sh 2026-10-07_01-53-16_w3S4d2_from_s4d" >/dev/null; do sleep 30; done
for k in f b; do .wd_tmp/ev.sh 2026-10-07_01-53-16_w3S4d2_from_s4d 16299 base_dist_${k}_s1 --cmd 0.06,0,0 --ramp_off --seed 1 --num_envs 320 --dist_jitter 0.91 --dist "push:$k:6:0:0.3|push:$k:6:8:0.3|push:$k:6:12:0.3|push:$k:6:15:0.3|push:$k:6:18:0.3"; done
LAST=$(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
.wd_tmp/evset.sh $RN $LAST d500 full
for CK in 16500 16600; do [ -f $R/model_$CK.pt ] && .wd_tmp/evset.sh $RN $CK d$((CK-16299)) dist; done
echo "$(date +%H:%M) 체인 끝"
