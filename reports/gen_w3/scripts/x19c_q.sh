#!/bin/bash
# X19c(TRANSITION_W3KE.md §5.22f) — 서기 단독 키(stand_crouch_g21a 구성)가 패드 신호 끊김(왼발 0 고착 / 양발 0 고착)에서 35 s 를 서는가.
# 구성은 회귀 실행 f1008_0611_stand35 의 S-nd 와 같고(128 env, seed 1–3, 지연 2 틱) --contact_drop 만 더한다(관측 비트만 0, 물리·채점 불변).
# 사용: x19c_q.sh "<seed 순서>" "<팔 순서: SL SB>"
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
BAD="PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|illegal memory access|Unable to allocate|Traceback"
BASE="--headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --enc_obs v2 --contact_real --contact_ver v2 --delay_steps 2,2 --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --after_s 35"
mkdir -p $S/x19c
run() {  # TAG 인자...
  local TG=$1; shift
  grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$TG.txt 2>/dev/null && [ -f log_picture/ground_recovery_${TG}_envs.csv ] && { echo "skip $TG"; return; }
  exec 8>"$S/x19c/$TG.taglock"; flock -n 8 || { echo "대기 $TG"; flock 8; return; }
  sleep $((1 + RANDOM % 4))
  while true; do for i in 1 2 3; do if [ $i = 1 ]; then exec 9>"$LOCK"; else exec 9>"$LOCK.lane$i"; fi; flock -n 9 && break 2; done; sleep 2; done
  for k in 1 2 3; do
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 10500 ]; do sleep 30; done
    rm -f log_picture/ground_recovery_$TG.txt log_picture/ground_recovery_${TG}_envs.csv
    echo "$(date +%H:%M) 시작 $TG (시도 $k)"
    setsid timeout 900 /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py $BASE "$@" --out_tag $TG > $S/x19c/$TG.l 2>&1 &
    pg=$!; wait $pg; kill -KILL -- -$pg 2>/dev/null
    if grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$TG.txt 2>/dev/null && ! grep -aqE "$BAD" $S/x19c/$TG.l; then echo "$(date +%H:%M) 유효 $TG"; flock -u 9; flock -u 8; return; fi
    echo "$(date +%H:%M) ⚠ 무효 $TG 재시도 $k"; mv $S/x19c/$TG.l $S/x19c/$TG.dead$k; sleep 20
  done
  echo "$(date +%H:%M) ✗ $TG 3회 무효"; flock -u 9; flock -u 8
}
for SD in ${1:-1 2 3}; do for A in ${2:-SL SB}; do
  case $A in SL) D="L:0:35";; SB) D="B:0:35";; esac
  run x19c_${A}_s${SD} --seed $SD --contact_drop "$D"
done; done
echo "$(date +%H:%M) X19C_DONE"
