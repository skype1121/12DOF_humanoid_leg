#!/bin/bash
# 전환 평가 래퍼 — 사용: safe_tr.sh TAG [eval_transition 추가 인자...]  (공용 잠금, 메모리 대기, 무효 재시도 3)
cd ${EVAL_REPO:?EVAL_REPO 필요} || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/tr
LOCK=${EVAL_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
MEM_MAX=${MEM_MAX:-10500}; TMO=${EVAL_TIMEOUT:-1500}; ENVS=${EVAL_ENVS:-128}
BAD="PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|illegal memory access|Unable to allocate|Traceback"
BASE="--headless --task Biped12-HangStand-G11a-v0 --precheck_s 3 --after_s 22 --enc_obs v2 --contact_real --contact_ver v2 --load_run 2026-10-07_00-11-12_w3S3d_from_s2d --checkpoint model_15800.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25"
KE="--sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33 --ke 2026-10-02_21-23-08_g21a:model_11300.pt"
TG=$1; shift
# 1007: 같은 태그의 이중 실행 방지 — 다른 체인이 같은 태그를 돌리는 중이면 그 실행이 끝날 때까지 기다렸다가 그대로 돌아간다(산출물을 지우지 않는다)
mkdir -p $S; exec 8>"$S/$TG.taglock"
if ! flock -n 8; then echo "$(date +%H:%M) 대기 $TG (같은 태그가 다른 체인에서 실행 중 — 끝나면 건너뜀)"; flock 8; exit 0; fi
if [[ " $* " == *" --ke_mode none "* && " $* " != *" --dir ke2w3 "* ]]; then EXTRA=""; else EXTRA="$KE"; fi     # w3ke none 팔 = W3 STAND 유지(KE 정책 없음); ke2w3 none = KE 유지(KE 필요)
acquire_lane() {   # 레인 1 = 기존 공용 잠금(다른 도구와 직렬), 2..N = 추가 레인 — 동시 평가 최대 ${EVAL_LANES:-3} 개
  local i n=${EVAL_LANES:-3}
  if [ -z "${EVAL_POLL:-}" ]; then sleep $((3 + RANDOM % 4)); else sleep ${EVAL_YIELD:-3}; fi   # 양보 뒤 시도 — 우선 체인도 3 s 양보(1007: 방금 끝난 우선 체인이 레인을 즉시 되잡아 다른 우선 체인이 굶던 것 방지)
  while true; do
    for i in $(seq 1 $n); do
      if [ $i = 1 ]; then exec 9>"$LOCK"; else exec 9>"$LOCK.lane$i"; fi
      if flock -n 9; then return 0; fi
    done
    sleep ${EVAL_POLL:-$((4 + RANDOM % 6))}      # EVAL_POLL=1 이면 우선(자주 확인)
  done
}
mkdir -p "$(dirname $LOCK)" $S; acquire_lane
for k in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le $MEM_MAX ]; do sleep 30; done
  rm -f log_picture/ground_recovery_$TG.txt log_picture/ground_recovery_${TG}_envs.csv log_picture/trans_$TG.npz log_picture/transall_$TG.npz log_picture/kwloop_$TG.npz log_picture/rqlog_$TG.json
  echo "$(date +%H:%M) 시작 $TG (시도 $k): $BASE --num_envs $ENVS $EXTRA $* --out_tag $TG"
  setsid timeout $TMO /home/ryu/IsaacLab/isaaclab.sh -p ${TR_SCRIPT:-rl_walking/scripts/eval_transition.py} $BASE --num_envs $ENVS $EXTRA "$@" --out_tag $TG > $S/$TG.l 2>&1 &
  pg=$!; wait $pg; kill -KILL -- -$pg 2>/dev/null
  if grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$TG.txt 2>/dev/null && ! grep -aqE "$BAD" $S/$TG.l; then
    echo "$(date +%H:%M) 유효 $TG"; flock -u 9; exit 0; fi
  echo "$(date +%H:%M) ⚠ 무효 $TG 재시도 $k"; mv $S/$TG.l $S/$TG.dead$k; sleep 20
done
echo "$(date +%H:%M) ✗ $TG 3회 무효"; flock -u 9; exit 1
