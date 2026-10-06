#!/bin/bash
# 병목 2 기록 롤아웃 3 회(학습 없음). 규칙: GPU 사용 ≤ 10500 MiB 대기, 1 개씩(eval17 잠금 공유),
# 메모리/PhysX 실패 로그면 무효 → 재실행, 종료는 명시 PID(프로세스 그룹)로만.
# 사용: stopest_run.sh <출력 디렉터리> [R12 R3]
# (1006 첫 시도 R1 은 $BASE 누락으로 G6·seed 42·사전검사 29 % = 무효, 폐기. 예산 3 회를 지키려고 KE 2 외란을 한 번에 기록)
WT=$(cd "$(dirname "$0")/../.." && pwd); DATA=/home/ryu/humanoid_leg_test1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$WT EVAL_DATA_REPO=$DATA
OUT=$1; shift; mkdir -p $OUT
LOCK=${EVAL_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
wait_mem() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 10500 ] && return; sleep 20; done; }
bad_log() { grep -aqE "PxgCudaDeviceMemoryAllocator failed|CUDA error|ALLOC|PhysX error|fail to launch|Unable to allocate" "$1"; }
BASE="--headless --task Biped12-HangStand-G11a-v0 --num_envs 256 --precheck_s 3 --seed 1 --delay_steps 2,2 --enc_obs v2 --contact_real --contact_ver v2 --last_action_ema 0.27 --obs_ema joint_vel:0.25"
# KE(= kabs.sh): G21a 서기 + G24Ra 회복 전환, 배포 가능 감지기
KE="--load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33"
# W3 S1c + c2 supervisor, 정지 안 함(회복 뒤 제자리 걸음이 계속 = '걸으며 정착' 음성 표본)
C2NS="--load_run 2026-10-06_15-06-14_w3S1c_from_w3A3 --checkpoint model_14196.pt --c2 --c2_arm deploy --c2_params min_td=1e6,tmax_s=1e9"
declare -A R=(
  [R12]="$KE --after_s 16 --dist push:f:2:40:0.2;push:b:10:18:0.3"
  [R3]="$C2NS --after_s 14 --dist push:b:2:18:0.3;push:f:8:40:0.2"
)
for k in ${@:-R12 R3}; do
  f=$OUT/stopest_$k.npz; lg=$OUT/stopest_$k.l; rm -f $f
  exec 9>$LOCK; flock 9
  for TRY in 1 2 3 4; do
    wait_mem
    setsid /home/ryu/IsaacLab/isaaclab.sh -p $WT/reports/gen_w3/stopest_rec.py $BASE ${R[$k]} --out_tag stopest_$k --rec_out $f > $lg 2>&1 &
    pg=$!; echo "$(date +%H:%M:%S) 시작 $k PID $pg (시도 $TRY)"
    while kill -0 $pg 2>/dev/null; do sleep 3; done
    kill -KILL -- -$pg 2>/dev/null
    if [ -f $f ] && ! bad_log $lg && grep -aq "HangStand-G11a-v0 |" $lg && grep -aq "seed=1 |" $lg; then echo "$(date +%H:%M:%S) 완료 $k"; break; fi
    echo "$(date +%H:%M:%S) 무효(산출물 없음 또는 메모리/PhysX 실패) → 재실행 $k"; rm -f $f; sleep 60
  done
  flock -u 9
done
echo "$(date +%H:%M:%S) STOPEST 기록 끝"
