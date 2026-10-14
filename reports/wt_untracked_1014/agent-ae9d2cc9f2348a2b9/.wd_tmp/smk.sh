#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
CK=$1; TG=$2
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -le 11000 ]; do sleep 20; done
timeout 900 /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/smoke_w3d.py --headless --task Biped12-HangStand-W3S4dD-v0 --ckpt $CK --num_envs 256 --level 1 --p_dist 1.0 --seconds 60 --tag $TG $3 > .wd_tmp/smk_$TG.log 2>&1
python3 -c "
import json;d=json.load(open('log_picture/smoke_w3d_$TG.json'));print('$TG',{k:d[k] for k in ('fall_dist','ret_paid','sret_paid','stop_issued','applied_env','gain_mean','cap_mean')})"
