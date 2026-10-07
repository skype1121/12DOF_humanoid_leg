#!/bin/bash
# W3S3dR 학습(ROBUST_DESIGN §2): W3S3d-15800 → +500, seed 1, 4096 env. GPU ≤ 7500 MiB 대기, 기동 실패 재시도. 끝나면 stage4 링크 + 이정표 표시.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/rob
PN=2026-10-07_00-11-12_w3S3d_from_s2d; PC=15800; T=Biped12-HangStand-W3S3dR-v0
for try in 1 2 3 4 5; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 7500 ]; do sleep 30; done
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed 1 --max_iterations 500 --resume --load_run $PN --checkpoint model_$PC.pt --run_name w3S3dR_from_s3d > $S/train_w3S3dR.log 2>&1 &
  P=$!; echo "$(date +%H:%M) 학습 PID $P (시도 $try)"; sleep 180
  if ! kill -0 $P 2>/dev/null || grep -qE "failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError|illegal memory" $S/train_w3S3dR.log; then
    kill $P 2>/dev/null; sleep 20; D=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S3dR_from_s3d 2>/dev/null | tail -1); [ -n "$D" ] && mv $D $S/dead_w3S3dR_$try; echo "$(date +%H:%M) 기동 실패 재시도 $try"; continue; fi
  echo "$(date +%H:%M) W3S3dR 학습 시작(정상)"; break
done
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S3dR_from_s3d | tail -1); RN=$(basename $R); ln -sfn ../biped12_stepb0/$RN logs/rsl_rl/biped12_stage4/$RN; echo "run $RN" | tee $S/run.txt
wait $P; echo "$(date +%H:%M) W3S3dR 학습 끝: $(ls $R/model_*.pt | wc -l) 체크포인트"; echo "ROB_TRAIN_DONE $RN"
