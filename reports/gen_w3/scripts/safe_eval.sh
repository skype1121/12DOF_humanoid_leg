#!/bin/bash
# claude2 평가 래퍼 — 사용: safe_eval.sh TASK RUN CK TAG [eval_w3 추가 인자...]
# 공용 GPU 잠금(eval17.gpu.lock, claude1 scratchpad 와 같은 파일) 안에서 메모리 ≤ MEM_MAX MiB 대기 → eval_w3.py 실행.
# PhysX/CUDA 메모리 실패·산출물 누락 = 무효 → 재시도(최대 3). 종료는 timeout 이 명시 PID 로만. pkill 없음.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
LOCK=${EVAL_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
MEM_MAX=${MEM_MAX:-10500}; TMO=${EVAL_TIMEOUT:-900}; ENVS=${EVAL_ENVS:-256}; SEC=${EVAL_SEC:-30}; DLY=${EVAL_DELAY:-2}; SEED=${EVAL_SEED:-1}
T=$1 RN=$2 CK=$3 TG=$4; shift 4
mkdir -p "$(dirname $LOCK)" $S; exec 9>"$LOCK"; flock 9
BAD="PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|illegal memory access"
for k in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le $MEM_MAX ]; do sleep 30; done
  rm -f log_picture/w3eval_$TG.txt log_picture/w3eval_$TG.json
  echo "$(date +%H:%M) 평가 $TG 시작(시도 $k): --task $T --load_run $RN --checkpoint model_$CK.pt --num_envs $ENVS --seconds $SEC --delay $DLY --seed $SEED $*"
  timeout $TMO /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_w3.py --headless --task "$T" --load_run "$RN" --checkpoint model_$CK.pt \
    --num_envs $ENVS --seconds $SEC --delay $DLY --seed $SEED --out_tag "$TG" "$@" > $S/ev_$TG.l 2>&1
  if grep -qE "$BAD" $S/ev_$TG.l || [ ! -s log_picture/w3eval_$TG.txt ]; then
    echo "$(date +%H:%M) ⚠ $TG 무효(GPU 메모리/산출물 없음) 재시도 $k"; mv $S/ev_$TG.l $S/ev_$TG.dead$k; sleep 60; continue; fi
  echo "$(date +%H:%M) 평가 $TG 끝(유효)"; flock -u 9; exit 0
done
echo "$(date +%H:%M) ✗ $TG 3회 무효"; flock -u 9; exit 1
