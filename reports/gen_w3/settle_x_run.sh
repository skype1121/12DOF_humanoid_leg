#!/bin/bash
# STOPEST §13(1007): odicw 개선 후보·조건 제거 감사·진동 주입 — 동적 큐 실행기(동시 1 개). 학습 없음.
# 큐 파일 한 줄 = "태그|추가 인자" (태그 = <접두>_<팔>_<조건>_s<seed>, 접두 sab = 지연 2·sabd3 = 지연 3·osc* = 진동 주입 실험).
# 기본 인자(BASE)는 settle_ab_run.sh 와 같다(KE kabs 인자). 팔 인자는 BASE 뒤에 붙인다(argparse 마지막 값 우선 → --sw_early_imu·--sw_refrac 덮어쓰기).
# GPU: 공용 잠금(GPU_LOCK) 안에서만 실행 · 사용 ≤ 10500 MiB 대기 · 메모리/PhysX 실패 로그·산출물 없음·seed/과제/사전검사 불일치 = 무효 → 재실행(최대 3) · 종료는 명시 PID(프로세스 그룹).
# 사용: settle_x_run.sh <큐 파일>   (큐가 비면 30 s 마다 다시 보고, <큐>.stop 파일이 있으면 끝)
WT=$(cd "$(dirname "$0")/../.." && pwd); DATA=/home/ryu/humanoid_leg_test1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$WT EVAL_DATA_REPO=$DATA
Q=$1; S=$(dirname "$Q"); mkdir -p $S/logs
GPU_LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
wait_mem() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 10500 ] && return; sleep 20; done; }
bad_log() { grep -aqE "PxgCudaDeviceMemoryAllocator failed|CUDA error|ALLOC|PhysX error|fail to launch|Unable to allocate|Traceback" "$1"; }
BASE="--headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --after_s 8 --enc_obs v2 --contact_real --contact_ver v2 \
 --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 \
 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any \
 --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33"
while true; do
  [ -f $Q.stop ] && { echo "$(date +%H:%M:%S) 정지 파일 — 끝"; exit 0; }
  exec 7>$Q.lock; flock 7; LINE=$(head -1 $Q); [ -n "$LINE" ] && sed -i 1d $Q; flock -u 7
  if [ -z "$LINE" ]; then sleep 30; continue; fi
  t=${LINE%%|*}; ARGS=${LINE#*|}
  SD=$(echo $t | sed -E 's/.*_s([0-9]+)$/\1/'); PFX=${t%%_*}
  DLY=2; [ "$PFX" = "sabd3" ] && DLY=3; [[ "$PFX" == *d3 ]] && DLY=3
  f=$DATA/log_picture/ground_recovery_${t}_envs.csv; tx=$DATA/log_picture/ground_recovery_${t}.txt; lg=$S/logs/$t.l
  ok() { [ -f $f ] && ! bad_log $lg && grep -aq "HangStand-G11a-v0 |" $tx && grep -aq "seed=$SD |" $tx && grep -aq "사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]" $tx; }
  for TRY in 1 2 3 4 5; do
    # 1007 코디네이터 지시: 공용 GPU 잠금 안에서 실행 — 잠금 → 사용량 ≤ 10500 MiB 확인 → 실행 → 해제
    exec 6>$GPU_LOCK; flock 6
    wait_mem; rm -f $f $tx
    T0=$(date +%s)
    setsid /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/eval_ground_recovery.py $BASE --delay_steps $DLY,$DLY --seed $SD $ARGS --out_tag $t > $lg 2>&1 &
    pg=$!
    while kill -0 $pg 2>/dev/null; do sleep 3; done; kill -KILL -- -$pg 2>/dev/null
    flock -u 6; exec 6>&-
    if ok 2>/dev/null; then echo "$(date +%H:%M:%S) 완료 $t ($(( $(date +%s) - T0 )) s)"; break; fi
    echo "$(date +%H:%M:%S) 무효 → 재실행 $t ($TRY)"; sleep 30
  done
done
