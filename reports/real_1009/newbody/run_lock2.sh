#!/bin/bash
# run_lock.sh 사본(1010): 회복 정책 자리 REC=run:ckpt · 추가 인자 EXTRA="..." · 산출물 newbody/ev/out · 낙상 시 모드 파일 복사. 나머지 동일.
# 사용: run_lock.sh <작업목록 파일>   (줄: TAG<TAB>NUM_ENVS<TAB>AFTER_S<TAB>LOCK(없으면 -)<TAB>DIST(없으면 -)[<TAB>MASS(없으면 -)<TAB>SEED(기본 1)])
set -u
HERE=$(cd "$(dirname "$0")" && pwd)/ev; R=/home/ryu/humanoid_leg_test1
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
mkdir -p "$(dirname $LOCK)" $HERE/logs $HERE/out
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$R EVAL_DATA_REPO=$R
BASE="--headless --task Biped12-HangStand-G11a-v0 --precheck_s 3 --enc_obs v2 --contact_real --contact_ver v2 --delay_steps 2,2 --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to ${REC:-2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt} --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33"
BAD='PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch'
while IFS=$'\t' read -r TAG NE AFT LK DIST MASS SEED; do
  MASS=${MASS:--}; SEED=${SEED:-1}
  [ -z "$TAG" ] && continue
  T=$R/log_picture/ground_recovery_${TAG}.txt; C=$R/log_picture/ground_recovery_${TAG}_envs.csv
  if [ -s $HERE/out/ground_recovery_${TAG}.txt ] && [ -s $HERE/out/ground_recovery_${TAG}_envs.csv ]; then echo "$TAG 이미 있음 — 건너뜀"; continue; fi
  ARGS="$BASE --seed $SEED --num_envs $NE --after_s $AFT"
  case "$MASS" in -) ;; urdf=*) ARGS="$ARGS --link_urdf ${MASS#urdf=}";; *) ARGS="$ARGS --mass_delta $MASS";; esac
  [ "$LK" != "-" ] && ARGS="$ARGS --lock_joints $LK"
  [ -n "${LOCK_BAND:-}" ] && ARGS="$ARGS --lock_band_deg $LOCK_BAND"; [ -n "${EXTRA:-}" ] && ARGS="$ARGS $EXTRA"
  [ "$DIST" != "-" ] && ARGS="$ARGS --dist $DIST"
  exec 9>$LOCK; flock 9; ok=0
  for TRY in 1 2; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 10500 ]; do sleep 20; done
    rm -f $T $C; t0=$(date +%s); LG=$HERE/logs/$TAG.l
    setsid /home/ryu/IsaacLab/isaaclab.sh -p -u $R/rl_walking/scripts/eval_ground_recovery_lock.py $ARGS --out_tag $TAG < /dev/null > $LG 2>&1 &
    PG=$!; echo "$(date +%H:%M:%S) 시작 $TAG (PID $PG, 시도 $TRY)"
    while kill -0 $PG 2>/dev/null; do sleep 3; if [ $(( $(date +%s) - t0 )) -gt 900 ]; then echo "  시간 초과 → 그룹 $PG 종료"; kill -TERM -- -$PG 2>/dev/null; sleep 10; kill -KILL -- -$PG 2>/dev/null; fi; done
    kill -KILL -- -$PG 2>/dev/null
    if [ -s $T ] && [ -s $C ] && [ $(stat -c %Y $T) -ge $t0 ] && [ $(($(wc -l < $C))) -eq $((NE+1)) ] && ! grep -Eq "$BAD" $LG && grep -q "^\[고정\]" $T && grep -q "^\[질량\]" $T && ! grep -q "무효!" $T; then
      ok=1; cp $T $C $HERE/out/; M=${T%.txt}_modefall.csv; [ -s $M ] && [ $(stat -c %Y $M) -ge $t0 ] && cp $M $HERE/out/; echo "$(date +%H:%M:%S)   유효 ($(( $(date +%s) - t0 )) s)"; break; fi
    echo "$(date +%H:%M:%S)   무효 → 재시도"; cp $LG $LG.dead$TRY; sleep 30
  done
  flock -u 9
  printf "%s\t%s\t%s\n" $TAG $([ $ok = 1 ] && echo 유효 || echo 무효) $TRY >> $HERE/status.tsv
done < "$1"
echo "$(date +%H:%M:%S) 끝"
