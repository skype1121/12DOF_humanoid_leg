#!/bin/bash
# imuH 독립 seed 7–12 (STOPEST §16) — imu(기준)+imuH × 지연 2·3 × nd/sf15/fi40/b18. 공용 잠금 안 1개씩, 무효 재시도 3.
cd /home/ryu/humanoid_leg_test1 || exit 2; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad/imuh
LOCK=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock
BAD="PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|PhysX error|fail to launch|Create Stream|Failed to fetch|illegal memory access|Unable to allocate"
BASE="--headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --after_s 8 --enc_obs v2 --contact_real --contact_ver v2 --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33 --trace_npz"
declare -A X=( [nd]="" [sf15]="--dist ramp:f:2:15:0.5:1.0" [fi40]="--dist push:f:2:40:0.2" [b18]="--dist push:b:2:18:0.3" )
declare -A ARM=( [imu]="--sw_early_imu 0.6,6,0.4,0.2" [imuH]="--sw_early_imu 0.6,6,0.4,0.7" )
ok_txt() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' "$1" 2>/dev/null; }
for D in 2 3; do PFX=$([ $D = 2 ] && echo sab || echo sabd3)
 for SD in 7 8 9 10 11 12; do for A in imu imuH; do for C in nd sf15 fi40 b18; do
  t=${PFX}_${A}_${C}_s${SD}
  ok_txt log_picture/ground_recovery_${t}.txt && [ -f log_picture/ground_recovery_${t}_envs.csv ] && { echo "skip $t"; continue; }
  exec 9>"$LOCK"; flock 9
  for TRY in 1 2 3; do
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 10500 ]; do sleep 30; done
    rm -f log_picture/ground_recovery_${t}.txt log_picture/ground_recovery_${t}_envs.csv
    setsid /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py $BASE ${ARM[$A]} --delay_steps $D,$D --seed $SD ${X[$C]} --out_tag $t > $S/$t.l 2>&1 &
    pg=$!; wait $pg; kill -KILL -- -$pg 2>/dev/null
    if ok_txt log_picture/ground_recovery_${t}.txt && ! grep -aqE "$BAD" $S/$t.l; then echo "$(date +%H:%M) 유효 $t"; break; fi
    echo "$(date +%H:%M) ⚠ 무효 $t 재시도 $TRY"; mv $S/$t.l $S/$t.dead$TRY; sleep 20
  done
  flock -u 9
 done; done; done
done
echo "$(date +%H:%M) IMUH_DONE"
