#!/bin/bash
# 새 몸 학습 환경 점검 실행(공용 GPU 잠금·한 번에 하나·산출물로 판정). 사용: run_check.sh <TAG> <태스크 접미> [envs] [steps] [run] [ckpt]
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
TAG=$1; ST=$2; NE=${3:-1024}; STEPS=${4:-1500}; RUN=${5:-2026-10-04_10-08-41_g24ra_seed1_from_w2e20294}; CK=${6:-model_21793.pt}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$R RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1009
BAD='PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch'
mkdir -p $HERE/out $HERE/logs "$(dirname $LOCK)"; O=$HERE/out/check_$TAG.txt; LG=$HERE/logs/check_$TAG.l
exec 9>$LOCK; flock 9
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt ${MEMLIM:-7500} ]; do sleep 20; done
rm -f $O; t0=$(date +%s)
setsid /home/ryu/IsaacLab/isaaclab.sh -p -u $HERE/check_newbody_env.py --headless --task Biped12-HangStand-$ST-v0 --num_envs $NE --seed 1 --load_run $RUN --checkpoint $CK --steps $STEPS --tag $TAG ${XARGS:-} < /dev/null > $LG 2>&1 &
PG=$!; echo "$(date +%H:%M:%S) 시작 $TAG (PGID $PG)"
while kill -0 $PG 2>/dev/null; do sleep 3; if [ $(( $(date +%s) - t0 )) -gt 900 ]; then echo "시간 초과 → 그룹 $PG 종료"; kill -TERM -- -$PG 2>/dev/null; sleep 10; kill -KILL -- -$PG 2>/dev/null; fi; done
kill -KILL -- -$PG 2>/dev/null
flock -u 9
if [ -s $O ] && [ $(stat -c %Y $O) -ge $t0 ] && tail -1 $O | grep -q '^DONE' && ! grep -Eq "$BAD" $LG; then echo "$(date +%H:%M:%S) 유효 $TAG ($(( $(date +%s) - t0 )) s)"; else echo "$(date +%H:%M:%S) 무효 $TAG"; grep -n "Traceback" -A 25 $LG | head -60; tail -5 $LG; fi
