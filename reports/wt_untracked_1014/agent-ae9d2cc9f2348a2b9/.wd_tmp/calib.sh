#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
G="push:f:6:0:0.3|push:f:6:12:0.3|push:b:6:12:0.3|push:l:6:12:0.3|push:r:6:12:0.3"
S4=2026-10-07_01-53-16_w3S4d2_from_s4d; S3=2026-10-07_00-11-12_w3S3d_from_s2d
C="--cmd 0.06,0,0 --ramp_off --seed 3 --num_envs 320 --dist_jitter 0.91"
.wd_tmp/ev.sh $S4 16299 rs_cal_s4_d2 $C --dist "$G" --rsup ab --rsup_params th_on=1e9
.wd_tmp/ev.sh $S4 16299 rs_off_s4_d2 $C --dist "$G"
.wd_tmp/ev.sh $S4 16299 rs_cal_s4_d4 $C --dist "$G" --rsup ab --rsup_params th_on=1e9 --delay 4
TASK=Biped12-HangStand-W3S3d-v0 .wd_tmp/ev.sh $S3 15800 rs_cal_s3_d2 $C --dist "$G" --rsup ab --rsup_params th_on=1e9
TASK=Biped12-HangStand-W3S3d-v0 .wd_tmp/ev.sh $S3 15800 rs_cal_s3_d4 $C --dist "$G" --rsup ab --rsup_params th_on=1e9 --delay 4
echo "보정 끝"
