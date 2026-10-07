#!/bin/bash
# W3 계열 이어 학습 공용 스크립트 — 사용: w3train.sh <태스크 접미(W3S3f)> <작업 폴더 이름(s3f)> [부모 run] [부모 체크포인트] [반복 수]
# 평가 레인 2 개를 잠시 잡아 GPU 를 비운 뒤 기동, 기동 확인 뒤 레인 반납. 끝나면 "<TAG>_TRAIN_DONE <run>" 을 chain.out 에 남긴다.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
ST=$1; W=$2; PN=${3:-2026-10-07_00-11-12_w3S3d_from_s2d}; PC=${4:-15800}; IT=${5:-500}
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/$W; mkdir -p $S
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
T=Biped12-HangStand-$ST-v0; RNAME=w3${ST#W3}${W3SEED:+_sd$W3SEED}_from_$PC; TAG=$(echo $W | tr a-z A-Z)
echo "$(date +%H:%M) [$TAG] 레인 2·3 확보 대기"
exec 7>"$LOCK.lane2"; flock 7; exec 6>"$LOCK.lane3"; flock 6
echo "$(date +%H:%M) [$TAG] 레인 2·3 확보 — GPU ≤ 7500 MiB 대기"
for try in 1 2 3 4 5; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 7500 ]; do sleep 20; done
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed ${W3SEED:-1} --max_iterations $IT --resume --load_run $PN --checkpoint model_$PC.pt --run_name $RNAME > $S/train.log 2>&1 &
  P=$!; echo "$(date +%H:%M) [$TAG] 학습 PID $P (시도 $try) 태스크 $T"; sleep 180
  if ! kill -0 $P 2>/dev/null || grep -qE "failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError|illegal memory" $S/train.log; then
    kill $P 2>/dev/null; sleep 20; D=$(ls -d logs/rsl_rl/biped12_stepb0/*_$RNAME 2>/dev/null | tail -1); [ -n "$D" ] && mv $D $S/dead_$try; echo "$(date +%H:%M) [$TAG] 기동 실패 재시도 $try"; continue; fi
  echo "$(date +%H:%M) [$TAG] 학습 시작(정상)"; break
done
flock -u 7; flock -u 6; echo "$(date +%H:%M) [$TAG] 레인 2·3 반납"
R=$(ls -d logs/rsl_rl/biped12_stepb0/*_$RNAME | tail -1); RN=$(basename $R); ln -sfn ../biped12_stepb0/$RN logs/rsl_rl/biped12_stage4/$RN; echo "run $RN" > $S/run.txt
wait $P; echo "$(date +%H:%M) [$TAG] 학습 끝: $(ls $R/model_*.pt | wc -l) 체크포인트 · 마지막 $(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)"; echo "${TAG}_TRAIN_DONE $RN"
