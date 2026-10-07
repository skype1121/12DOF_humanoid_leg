#!/bin/bash
# envelope 평가 래퍼 — 사용: safe_env.sh TAG RUN CK [eval_w3_envelope 추가 인자...] (공용 잠금·메모리 대기·무효 재시도 3)
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/rob
LOCK=${EVAL_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
BAD="PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|illegal memory access|Unable to allocate|Traceback"
TG=$1; RN=$2; CK=$3; shift 3
acquire_lane() {   # 레인 1 = 기존 공용 잠금(다른 도구와 직렬), 2..N = 추가 레인 — 동시 평가 최대 ${EVAL_LANES:-3} 개
  local i n=${EVAL_LANES:-3}
  while true; do
    for i in $(seq 1 $n); do
      if [ $i = 1 ]; then exec 9>"$LOCK"; else exec 9>"$LOCK.lane$i"; fi
      if flock -n 9; then return 0; fi
    done
    sleep $((4 + RANDOM % 6))
  done
}
mkdir -p $S; acquire_lane
for k in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 10500 ]; do sleep 30; done
  rm -f log_picture/w3env_$TG.json log_picture/w3env_$TG.txt
  setsid timeout ${EVAL_TIMEOUT:-2400} /home/ryu/IsaacLab/isaaclab.sh -p ${ENV_SCRIPT:-rl_walking/scripts/eval_w3_envelope.py} --headless --task Biped12-HangStand-W3S3d-v0 --per_group 256 --warmup 3 --base_delay 2 --load_run $RN --checkpoint model_$CK.pt "$@" --out_tag $TG > $S/env_$TG.l 2>&1 &
  pg=$!; wait $pg; kill -KILL -- -$pg 2>/dev/null
  if [ -f log_picture/w3env_$TG.json ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('log_picture/w3env_$TG.json')).get('completed', True) else 1)" && ! grep -aqE "$BAD" $S/env_$TG.l; then echo "$(date +%H:%M) 유효 $TG"; flock -u 9; exit 0; fi
  echo "$(date +%H:%M) ⚠ 무효 $TG 재시도 $k"; mv $S/env_$TG.l $S/env_$TG.dead$k; sleep 20
done
echo "$(date +%H:%M) ✗ $TG 3회 무효"; flock -u 9; exit 1
