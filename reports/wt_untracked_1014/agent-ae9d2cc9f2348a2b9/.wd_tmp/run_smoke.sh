#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
CK=/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/model_15800.pt
for k in 1 2; do
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -le 10500 ]; do sleep 20; done
  rm -f log_picture/smoke_w3d_s3d15800_L2f.json
  timeout 900 /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/smoke_w3d.py --headless --ckpt $CK --num_envs 256 --level 2 --p_dist 1.0 --tag s3d15800_L2f > .wd_tmp/smoke_L2.log 2>&1
  if grep -qE "PxgCudaDeviceMemoryAllocator failed|CUDA error|out of memory|PhysX error|Traceback" .wd_tmp/smoke_L2.log || [ ! -s log_picture/smoke_w3d_s3d15800_L2f.json ]; then echo "무효 $k"; mv .wd_tmp/smoke_L2.log .wd_tmp/smoke_L2.dead$k; sleep 30; continue; fi
  echo "유효"; exit 0
done
