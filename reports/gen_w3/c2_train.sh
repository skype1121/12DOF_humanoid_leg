#!/bin/bash
# c2 GO 학습(1007): W3S3dC2 를 W3S3d 끝에서 --resume, seed 1, +500. 워크트리 코드 + 본 트리 logs/(학습 산출물 위치, 코드 쓰기 아님).
# 규칙: 기동 전 GPU 사용 메모리 ≤ 7500 MiB 대기 · 첫 3 분 안 메모리 실패 로그면 실행 디렉터리 삭제 없이 재시도(최대 4) · 실행 디렉터리는 기동 뒤 새로 생긴 *_${RUN_NAME} 로 확인
# · 종료 감시는 PID 로. 사용: c2_train.sh  (출력: scratchpad/c2train.out, 실행 디렉터리 경로는 c2train.rundir)
WT=$(cd "$(dirname "$0")/../.." && pwd); DATA=/home/ryu/humanoid_leg_test1
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
RUN_NAME=${RUN_NAME:-w3S3dC2_from_s3d}; SEED=${SEED:-1}; ITERS=${ITERS:-500}
LOGROOT=$DATA/logs/rsl_rl/biped12_stepb0
export OMNI_KIT_ACCEPT_EULA=YES
for TRY in 1 2 3 4; do
  while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 7500 ] && break; sleep 30; done
  before=$(ls -d $LOGROOT/*_${RUN_NAME} 2>/dev/null)
  cd $DATA
  setsid /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/train.py --headless --task Biped12-HangStand-W3S3dC2-v0 --num_envs 4096 --seed $SEED \
    --resume --load_run 2026-10-07_00-11-12_w3S3d_from_s2d --checkpoint model_16193.pt --run_name $RUN_NAME --max_iterations $ITERS > $S/c2train_try$TRY.log 2>&1 &
  pg=$!; echo "$(date +%H:%M:%S) 학습 기동 시도 $TRY PID $pg" | tee -a $S/c2train.out
  bad=0
  for k in $(seq 1 18); do
    sleep 10
    if grep -aqE 'Unable to allocate|PxgCudaDeviceMemoryAllocator failed|CUDA out of memory|out of memory' $S/c2train_try$TRY.log; then bad=1; break; fi
    kill -0 $pg 2>/dev/null || break
  done
  if [ $bad = 1 ] || ! kill -0 $pg 2>/dev/null; then
    echo "$(date +%H:%M:%S) 시도 $TRY 실패(메모리 실패 $bad 또는 조기 종료) — PID 그룹 $pg 종료 후 재시도" | tee -a $S/c2train.out
    kill -KILL -- -$pg 2>/dev/null; sleep 60; continue
  fi
  after=$(ls -d $LOGROOT/*_${RUN_NAME} 2>/dev/null)
  rd=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | tail -1)
  echo "$rd" > $S/c2train.rundir; echo "$(date +%H:%M:%S) 실행 디렉터리: $rd" | tee -a $S/c2train.out
  while kill -0 $pg 2>/dev/null; do sleep 60; done
  last=$(ls $rd/model_*.pt 2>/dev/null | sort -V | tail -1)
  echo "$(date +%H:%M:%S) 학습 프로세스 종료 · 마지막 체크포인트 $last" | tee -a $S/c2train.out
  grep -aE 'Traceback' $S/c2train_try$TRY.log | head -3 | tee -a $S/c2train.out
  echo TRAIN_DONE | tee -a $S/c2train.out
  exit 0
done
echo TRAIN_FAILED | tee -a $S/c2train.out
