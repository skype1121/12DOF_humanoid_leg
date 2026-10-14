#!/bin/bash
# ev.sh RUN CK TAG [args] — GPU ≤ 11000 대기, 메모리 실패 무효 재시도 3
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD W3EVAL_REPO=$PWD
RN=$1 CK=$2 TG=$3; shift 3
for k in 1 2 3; do
  rm -f log_picture/w3eval_$TG.txt log_picture/w3eval_$TG.json
  LK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
  flock $LK bash -c 'until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -le 10500 ]; do sleep 20; done; exec "$@"' _ timeout 1500 /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_w3.py --headless --task ${TASK:-Biped12-HangStand-W3S4dD-v0} --load_run $RN --checkpoint model_$CK.pt --delay 2 --out_tag $TG "$@" > .wd_tmp/ev_$TG.l 2>&1
  if grep -qE "PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch" .wd_tmp/ev_$TG.l || [ ! -s log_picture/w3eval_$TG.txt ]; then
    echo "$(date +%H:%M) 무효 $TG 재시도 $k"; mv .wd_tmp/ev_$TG.l .wd_tmp/ev_$TG.dead$k; sleep 60; continue; fi
  echo "$(date +%H:%M) 평가 $TG 끝"; exit 0
done
echo "$(date +%H:%M) ✗ $TG 3회 무효"
