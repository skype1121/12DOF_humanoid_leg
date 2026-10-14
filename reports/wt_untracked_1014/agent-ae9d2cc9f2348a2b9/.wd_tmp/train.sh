#!/bin/bash
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
PN=2026-10-07_01-53-16_w3S4d2_from_s4d; PC=16299; T=Biped12-HangStand-W3S4dD-v0; L=.wd_tmp/train_w3S4dDb.log
for try in 1 2 3 4 5 6; do
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -le 7500 ]; do sleep 30; done
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed 1 --max_iterations 500 --resume --load_run $PN --checkpoint model_$PC.pt --run_name w3S4dDb_from_s4d > $L 2>&1 &
  P=$!; sleep 180
  if grep -qE "failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError" $L || ! kill -0 $P 2>/dev/null; then
    kill $P 2>/dev/null; sleep 20; D=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S4dDb_from_s4d 2>/dev/null | tail -1); [ -n "$D" ] && mv $D .wd_tmp/dead_$try; echo "$(date +%H:%M) 기동 실패 재시도 $try"; continue; fi
  echo "$(date +%H:%M) 학습 시작 PID $P"; break
done
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S4dDb_from_s4d | tail -1); echo "실행 디렉터리 $R"
grep -m1 "Loading model checkpoint" $L
wait $P; echo "$(date +%H:%M) 학습 종료 $(ls $R/model_*.pt | tail -2 | xargs -n1 basename)"
