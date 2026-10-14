#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
G="push:f:6:0:0.3|push:f:6:12:0.3|push:f:6:18:0.3|push:b:6:18:0.3|push:l:6:18:0.3|push:r:6:18:0.3|push:f:6:12:0.3;stop:0.3|push:f:6:18:0.3;stop:0.3"
S4=2026-10-07_01-53-16_w3S4d2_from_s4d; S3=2026-10-07_00-11-12_w3S3d_from_s2d
for S in 1 2; do for D in 2 4; do for M in off a b; do
  C="--cmd 0.06,0,0 --ramp_off --seed $S --num_envs 384 --dist_jitter 0.91 --delay $D --dist $G --rsup $M --rsup_params th_on=16.1"
  [ -s log_picture/w3eval_rg_s4_d${D}_${M}_s$S.txt ] || .wd_tmp/ev.sh $S4 16299 rg_s4_d${D}_${M}_s$S $C --rsup_params th_on=16.1,vy_max=0.06
  [ -s log_picture/w3eval_rg_s3_d${D}_${M}_s$S.txt ] || TASK=Biped12-HangStand-W3S3d-v0 .wd_tmp/ev.sh $S3 15800 rg_s3_d${D}_${M}_s$S $C --rsup_params th_on=16.1,vy_max=0.04
done; done; done
echo "격자 끝"
