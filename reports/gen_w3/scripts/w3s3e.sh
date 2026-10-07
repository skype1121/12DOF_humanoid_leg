#!/bin/bash
# W3S3e 학습(reports/gen_w3/S3E_DESIGN.md §3): W3S3d-15800 → +500, seed 1, 4096 env. 평가 레인 2 개를 잠시 잡아 GPU 를 비운 뒤 기동, 기동 확인 뒤 레인 반납.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/s3e
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
PN=2026-10-07_00-11-12_w3S3d_from_s2d; PC=15800; T=Biped12-HangStand-W3S3e-v0
echo "$(date +%H:%M) 레인 2·3 확보 대기"
exec 7>"$LOCK.lane2"; flock 7; exec 6>"$LOCK.lane3"; flock 6
echo "$(date +%H:%M) 레인 2·3 확보(평가는 레인 1 만) — GPU ≤ 7500 MiB 대기"
for try in 1 2 3 4 5; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 7500 ]; do sleep 20; done
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed 1 --max_iterations 500 --resume --load_run $PN --checkpoint model_$PC.pt --run_name w3S3e_from_s3d > $S/train_w3S3e.log 2>&1 &
  P=$!; echo "$(date +%H:%M) 학습 PID $P (시도 $try)"; sleep 180
  if ! kill -0 $P 2>/dev/null || grep -qE "failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError|illegal memory" $S/train_w3S3e.log; then
    kill $P 2>/dev/null; sleep 20; D=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S3e_from_s3d 2>/dev/null | tail -1); [ -n "$D" ] && mv $D $S/dead_w3S3e_$try; echo "$(date +%H:%M) 기동 실패 재시도 $try"; continue; fi
  echo "$(date +%H:%M) W3S3e 학습 시작(정상)"; break
done
flock -u 7; flock -u 6; echo "$(date +%H:%M) 레인 2·3 반납"
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_w3S3e_from_s3d | tail -1); RN=$(basename $R); ln -sfn ../biped12_stepb0/$RN logs/rsl_rl/biped12_stage4/$RN; echo "run $RN" | tee $S/run.txt
wait $P; echo "$(date +%H:%M) W3S3e 학습 끝: $(ls $R/model_*.pt | wc -l) 체크포인트"; echo "S3E_TRAIN_DONE $RN"
