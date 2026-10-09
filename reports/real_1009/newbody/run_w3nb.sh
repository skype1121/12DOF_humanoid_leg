#!/bin/bash
# 새 몸 걷기 평가 실행(eval_w3_nb.py) — 한 번에 하나·공용 GPU 잠금·산출물(이번 실행 시각의 json)로 판정·종료는 명시 PGID.
# 사용: run_w3nb.sh <작업목록>  (줄: TAG<TAB>태스크 접미<TAB>run<TAB>ckpt<TAB>seed<TAB>나머지 인자)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)/w3; R=/home/ryu/humanoid_leg_test1
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
mkdir -p $HERE/logs $HERE/out
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$R RL_WALKING_EXTRA_TASKS=${RL_WALKING_EXTRA_TASKS:-rl_walking.tasks_extra_1009}
BAD='PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|Traceback'
while IFS=$'\t' read -r TAG ST RUN CK SEED REST; do
  [ -z "$TAG" ] && continue
  J=$R/log_picture/w3eval_${TAG}.json; T=$R/log_picture/w3eval_${TAG}.txt
  if [ -s $HERE/out/w3eval_${TAG}.json ]; then echo "$TAG 이미 있음 — 건너뜀"; continue; fi
  exec 9>$LOCK; flock 9; ok=0
  for TRY in 1 2; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 10500 ]; do sleep 20; done
    rm -f $J $T; t0=$(date +%s); LG=$HERE/logs/$TAG.l
    setsid /home/ryu/IsaacLab/isaaclab.sh -p -u $R/rl_walking/scripts/eval_w3_nb.py --headless --task Biped12-HangStand-$ST-v0 --load_run $RUN --checkpoint $CK --seed $SEED $REST --out_tag $TAG < /dev/null > $LG 2>&1 &
    PG=$!; echo "$(date +%H:%M:%S) 시작 $TAG (PGID $PG, 시도 $TRY)"
    while kill -0 $PG 2>/dev/null; do sleep 3; if [ $(( $(date +%s) - t0 )) -gt 900 ]; then echo "  시간 초과 → 그룹 $PG 종료"; kill -TERM -- -$PG 2>/dev/null; sleep 10; kill -KILL -- -$PG 2>/dev/null; fi; done
    kill -KILL -- -$PG 2>/dev/null
    if [ -s $J ] && [ -s $T ] && [ $(stat -c %Y $J) -ge $t0 ] && grep -q '"nb"' $J && ! grep -Eq "$BAD" $LG; then ok=1; cp $J $T $HERE/out/; echo "$(date +%H:%M:%S)   유효 ($(( $(date +%s) - t0 )) s)"; break; fi
    echo "$(date +%H:%M:%S)   무효 → 재시도"; cp $LG $LG.dead$TRY; sleep 20
  done
  flock -u 9
  printf "%s\t%s\t%s\n" $TAG $([ $ok = 1 ] && echo 유효 || echo 무효) $TRY >> $HERE/status.tsv
done < "$1"
echo "$(date +%H:%M:%S) 끝"
