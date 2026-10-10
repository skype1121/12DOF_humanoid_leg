#!/bin/bash
# 1010 실물 전 점검(NEWBODY_CHECKS.md) — run_trnb.sh 사본. 다른 점: 평가기 = eval_transition_kwx_nbc.py(환경변수 CHK_EVAL 로 바꿈), 산출물 newbody/chk/,
# 작업 줄에 몸(nb|ob)·추가 인자 열. 공용 GPU 잠금 파일은 run_trnb.sh·run_lock2.sh 와 같다. 한 번에 하나·산출물로 판정·종료는 명시 PGID.
# 사용: run_chk.sh <작업목록>  (줄: TAG<TAB>env 수<TAB>지연<TAB>seed<TAB>after_s<TAB>걷기 exported(-)<TAB>회복 exported(-)<TAB>대본<TAB>몸 nb|ob<TAB>추가 인자(- = 없음))
set -u
HERE=$(cd "$(dirname "$0")" && pwd)/chk; R=/home/ryu/humanoid_leg_test1
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
EV=${CHK_EVAL:-eval_transition_kwx_nbc.py}
mkdir -p $HERE/logs $HERE/out; cd $R
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$R EVAL_DATA_REPO=$R
U=$R/rl_walking/assets/biped12_final3_hiprpla.urdf:left_hip_a_joint,right_hip_a_joint,left_hip_r_joint,right_hip_r_joint
BASE0="--headless --task Biped12-HangStand-G11a-v0 --precheck_s 3 --enc_obs v2 --contact_real --contact_ver v2 --load_run 2026-10-07_00-11-12_w3S3d_from_s2d --checkpoint model_15800.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33 --ke 2026-10-02_21-23-08_g21a:model_11300.pt --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 "
BAD='PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|Traceback'
while IFS=$'\t' read -r TAG NE DL SEED AFT W3X RECX SCR BODY EXTRA; do
  [ -z "$TAG" ] && continue
  T=$R/log_picture/ground_recovery_${TAG}.txt; C=$R/log_picture/ground_recovery_${TAG}_envs.csv; K=$R/log_picture/kwloop_${TAG}.npz
  if [ -s $HERE/out/ground_recovery_${TAG}.txt ]; then echo "$TAG 이미 있음 — 건너뜀"; continue; fi
  BASE="$BASE0"
  [ "$BODY" = nb ] && BASE="$BASE --lock_joints left_hip_r_joint,right_hip_r_joint --lock_band_deg 0 --lock_obs_zero --link_urdf $U"
  ARGS="$BASE --trace_all ${TR_TRACE:-256} --num_envs $NE --delay_steps $DL,$DL --seed $SEED --after_s $AFT"
  [ "$EXTRA" != "-" ] && ARGS="$ARGS $EXTRA"
  [ "$W3X" != "-" ] && ARGS="$ARGS --kw_w3_export $W3X"; [ "$RECX" != "-" ] && ARGS="$ARGS --kw_rec_export $RECX"
  exec 9>$LOCK; flock 9; ok=0
  for TRY in 1 2; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 10500 ]; do sleep 20; done
    rm -f $T $C $K; t0=$(date +%s); LG=$HERE/logs/$TAG.l
    setsid /home/ryu/IsaacLab/isaaclab.sh -p -u $R/rl_walking/scripts/$EV $ARGS --kw_script "$SCR" --out_tag $TAG < /dev/null > $LG 2>&1 &
    PG=$!; echo "$(date +%H:%M:%S) 시작 $TAG (PGID $PG, 시도 $TRY)"
    while kill -0 $PG 2>/dev/null; do sleep 3; if [ $(( $(date +%s) - t0 )) -gt ${TR_TIMEOUT:-1500} ]; then echo "  시간 초과 → 그룹 $PG 종료"; kill -TERM -- -$PG 2>/dev/null; sleep 10; kill -KILL -- -$PG 2>/dev/null; fi; done
    kill -KILL -- -$PG 2>/dev/null
    if [ -s $T ] && [ -s $C ] && [ -s $K ] && [ $(stat -c %Y $T) -ge $t0 ] && [ $(($(wc -l < $C))) -eq $((NE+1)) ] && ! grep -Eq "$BAD" $LG && grep -q "^\[고정\]" $T && grep -q "^\[질량\]" $T && ! grep -q "무효!" $T; then
      ok=1; cp $T $C $HERE/out/; echo "$(date +%H:%M:%S)   유효 ($(( $(date +%s) - t0 )) s)"; break; fi
    echo "$(date +%H:%M:%S)   무효 → 재시도"; cp $LG $LG.dead$TRY; sleep 20
  done
  flock -u 9
  printf "%s\t%s\t%s\n" $TAG $([ $ok = 1 ] && echo 유효 || echo 무효) $TRY >> $HERE/status.tsv
done < "$1"
echo "$(date +%H:%M:%S) 끝 $1"
