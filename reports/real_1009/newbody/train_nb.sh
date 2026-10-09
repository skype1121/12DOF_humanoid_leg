#!/bin/bash
# 새 몸 회복 이어 학습(사전 기준 reports/real_1009/NEWBODY_TRAIN.md) — 학습 seed 를 순차로(한 번에 하나). 사용: train_nb.sh <seed> [<seed> ...]
# 기동 동안만 공용 평가 잠금을 잡아 평가와 기동이 겹치지 않게 한다. 성공 판정은 산출물(새 run 폴더의 체크포인트)로만. 종료는 명시 PGID 로만.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; cd $R || exit 2
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$R RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1009
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
T=Biped12-HangStand-G24RaNB-v0; PN=2026-10-04_10-08-41_g24ra_seed1_from_w2e20294; PC=21793; IT=1500
BAD='failed to allocate|ALLOC_FAILED|Traceback|PxgCudaDeviceMemoryAllocator failed|OutOfMemoryError|illegal memory|CUDA error|out of memory'
mkdir -p $HERE/logs
for SEED in "$@"; do
  RNAME=g24ranb_s${SEED}_from_$PC; LG=$HERE/logs/train_$RNAME.log; ok=0
  for try in 1 2 3; do
    exec 9>$LOCK; flock 9
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 7500 ]; do sleep 20; done
    t0=$(date +%s)
    setsid /home/ryu/IsaacLab/isaaclab.sh -p -u rl_walking/scripts/train.py --headless --task $T --num_envs 4096 --seed $SEED --max_iterations $IT --resume --load_run $PN --checkpoint model_$PC.pt --run_name $RNAME < /dev/null > $LG 2>&1 &
    P=$!; echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 학습 PID=PGID $P (시도 $try) 로그 $LG"; echo "$P" > $HERE/logs/train_$RNAME.pgid
    sleep 240; flock -u 9
    D=$(ls -d logs/rsl_rl/biped12_stepb0/*_$RNAME 2>/dev/null | tail -1)
    if ! kill -0 $P 2>/dev/null || grep -qE "$BAD" $LG || [ -z "$D" ] || [ "$(stat -c %Y $D)" -lt $t0 ] || ! ls $D/model_*.pt >/dev/null 2>&1; then
      echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 기동 실패(시도 $try) — 그룹 $P 종료"; kill -TERM -- -$P 2>/dev/null; sleep 15; kill -KILL -- -$P 2>/dev/null
      [ -n "$D" ] && [ "$(stat -c %Y $D)" -ge $t0 ] && mv $D $HERE/logs/dead_${RNAME}_$try; cp $LG $LG.dead$try; sleep 30; continue; fi
    ok=1; break
  done
  [ $ok = 1 ] || { echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 기동 3 회 실패 — 중단"; exit 3; }
  RN=$(basename $D); ln -sfn ../biped12_stepb0/$RN logs/rsl_rl/biped12_stage4/$RN; echo "$RN" > $HERE/logs/train_$RNAME.run
  echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 학습 시작(정상) run $RN"
  while kill -0 $P 2>/dev/null; do
    sleep 30
    if [ -s $D/model_$((PC+IT-1)).pt ] && [ $(( $(date +%s) - $(stat -c %Y $D/model_$((PC+IT-1)).pt) )) -gt 120 ]; then   # 마지막 체크포인트 뒤 Kit 종료가 멈추면 그룹 종료
      echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 마지막 체크포인트 저장 뒤 프로세스 잔류 → 그룹 $P 종료"; kill -TERM -- -$P 2>/dev/null; sleep 15; kill -KILL -- -$P 2>/dev/null; fi
  done
  kill -KILL -- -$P 2>/dev/null
  echo "$(date '+%m-%d %H:%M:%S') [seed $SEED] 학습 끝: 체크포인트 $(ls $D/model_*.pt | wc -l) 개 · 마지막 $(ls $D/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)"; echo "NB_TRAIN_DONE seed $SEED $RN"
  [ -s $D/model_$((PC+IT-1)).pt ] || { echo "[seed $SEED] 마지막 체크포인트 없음 — 다음 seed 로 넘어가지 않음"; exit 4; }
done
