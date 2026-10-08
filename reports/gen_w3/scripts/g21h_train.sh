#!/bin/bash
# G21h(매달림 시작 혼합 서기) 이어 학습 — 사용: g21h_train.sh [반복 수=1000] [seed=1] [태스크 접미=G21h] [run 이름 꼬리=g21h]
# 평가 레인 2·3 을 잠시 잡아 GPU 를 비운 뒤 기동, 기동 확인 뒤 반납. 끝나면 "<TAG>_TRAIN_DONE <run>" 을 chain.out 에 남긴다. 종료는 명시 PID 로만.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1008
IT=${1:-1000}; SD=${2:-1}; ST=${3:-G21h}; NM=${4:-g21h}
PN=2026-10-02_21-23-08_g21a; PC=11300
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/g21h; mkdir -p $S
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
T=Biped12-HangStand-$ST-v0; RNAME=${NM}_s${SD}_from_$PC; TAG=$(echo ${NM}_s$SD | tr a-z A-Z)
echo "$(date +%H:%M) [$TAG] 레인 2·3 확보 대기"
exec 7>"$LOCK.lane2"; flock 7; exec 6>"$LOCK.lane3"; flock 6
for try in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 6500 ]; do sleep 20; done
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed $SD --max_iterations $IT --resume --load_run $PN --checkpoint model_$PC.pt --run_name $RNAME > $S/train_$RNAME.log 2>&1 &
  P=$!; echo "$(date +%H:%M) [$TAG] 학습 PID $P (시도 $try) 태스크 $T"; echo $P > $S/train_$RNAME.pid; sleep 200
  if ! kill -0 $P 2>/dev/null || grep -qE "failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError|illegal memory" $S/train_$RNAME.log; then
    kill $P 2>/dev/null; sleep 20; D=$(ls -d logs/rsl_rl/biped12_stage4/*_$RNAME 2>/dev/null | tail -1); [ -n "$D" ] && mv $D $S/dead_${RNAME}_$try; echo "$(date +%H:%M) [$TAG] 기동 실패 재시도 $try"; continue; fi
  echo "$(date +%H:%M) [$TAG] 학습 시작(정상)"; break
done
flock -u 7; flock -u 6; echo "$(date +%H:%M) [$TAG] 레인 2·3 반납"
R=$(ls -d logs/rsl_rl/biped12_stage4/*_$RNAME | tail -1); RN=$(basename $R); echo "run $RN" > $S/run_$RNAME.txt
wait $P; echo "$(date +%H:%M) [$TAG] 학습 끝: $(ls $R/model_*.pt | wc -l) 체크포인트 · 마지막 $(ls $R/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)"; echo "${TAG}_TRAIN_DONE $RN"
