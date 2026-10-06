#!/bin/bash
# 병목 2 폐루프 A/B(1006 밤): KE 조기 복귀 판정기 — imu(현행) vs odo θ0.07 vs odo θ0.05. 학습 없음.
# 조건 nd sf15 fi40 b18 × seed 1–6 × 팔 3, 128 env, 지연 2, 접촉 v2(kabs.sh 와 같은 KE 인자). 출력 log_picture/ground_recovery_sab_<팔>_<조건>_s<seed>{.txt,_envs.csv}
# GPU 규칙: 사용 ≤ 10500 MiB 대기, 레인 2 개(각자 잠금), 메모리/PhysX 실패 로그 또는 산출물 없음 = 무효 → 재실행, 종료는 명시 PID(프로세스 그룹).
# 사용: settle_ab_run.sh [팔...]   (기본 imu o07 o05)
WT=$(cd "$(dirname "$0")/../.." && pwd); DATA=/home/ryu/humanoid_leg_test1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$WT EVAL_DATA_REPO=$DATA
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/settle_ab; mkdir -p $S
wait_mem() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 10500 ] && return; sleep 20; done; }
bad_log() { grep -aqE "PxgCudaDeviceMemoryAllocator failed|CUDA error|ALLOC|PhysX error|fail to launch|Unable to allocate" "$1"; }
declare -A X=( [nd]="" [sf15]="--dist ramp:f:2:15:0.5:1.0" [fi40]="--dist push:f:2:40:0.2" [b18]="--dist push:b:2:18:0.3" )
declare -A ARM=( [imu]="" [o07]="--settle_judge odo --settle_v 0.07" [o05]="--settle_judge odo --settle_v 0.05" [odo]="--settle_judge odo --settle_v 0.07" [odoc]="--settle_judge odo --settle_v 0.07 --settle_knee_comp 42" [odi]="--settle_judge odo --settle_v 0.07 --settle_knee_inject 42" [odic]="--settle_judge odo --settle_v 0.07 --settle_knee_inject 42 --settle_knee_comp 42" [odicw]="--settle_judge odo --settle_v 0.07 --settle_knee_inject 42 --settle_knee_comp 42 --settle_w 0.4" )
# SAB_PFX(태그 접두, 기본 sab)·SAB_DLY(지연, 기본 2)·SAB_EXTRA(추가 인자, 예 "--knee_sea 42")
PFX=${SAB_PFX:-sab}; DLY=${SAB_DLY:-2}
BASE="--headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --after_s 8 --enc_obs v2 --contact_real --contact_ver v2 --delay_steps $DLY,$DLY ${SAB_EXTRA:-} \
 --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 \
 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --sw_off_contact any \
 --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33"
ARMS=${@:-imu o07 o05}
J=$S/jobs_${PFX}.txt; : > $J
# SAB_SEEDS·SAB_CONDS(기본 전부)·SAB_LANES(동시 실행 수, 기본 2 — 1007 다른 학습과 GPU 공유 시 1)
NL=${SAB_LANES:-2}
for SD in ${SAB_SEEDS:-1 2 3 4 5 6}; do for C in ${SAB_CONDS:-nd sf15 fi40 b18}; do for A in $ARMS; do echo "$A|$C|$SD" >> $J; done; done; done
lane() { local L=$1; local i=0
  while IFS='|' read -r A C SD; do
    if [ $((i % NL)) -eq $L ]; then
      t=${PFX}_${A}_${C}_s${SD}; f=$DATA/log_picture/ground_recovery_${t}_envs.csv; lg=$S/$t.l
      tx=$DATA/log_picture/ground_recovery_${t}.txt
      ok() { [ -f $f ] && ! bad_log $lg && grep -aq "HangStand-G11a-v0 |" $tx && grep -aq "seed=$SD |" $tx && grep -aq "사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]" $tx; }
      if ok 2>/dev/null; then i=$((i+1)); continue; fi
      exec 9>$S/lane$L.lock; flock 9
      for TRY in 1 2 3 4; do
        exec 8>$S/launch.lock; flock 8; wait_mem; rm -f $f $tx
        setsid /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/eval_ground_recovery.py $BASE --seed $SD ${ARM[$A]} ${X[$C]} --out_tag $t > $lg 2>&1 &
        pg=$!; sleep 75; flock -u 8      # 기동 뒤 메모리가 잡힐 때까지 다른 레인 기동 보류(동시 할당 OOM 방지)
        while kill -0 $pg 2>/dev/null; do sleep 3; done; kill -KILL -- -$pg 2>/dev/null
        if ok 2>/dev/null; then echo "$(date +%H:%M:%S) 완료 $t"; break; fi
        echo "$(date +%H:%M:%S) 무효 → 재실행 $t ($TRY)"; sleep 45
      done
      flock -u 9
    fi
    i=$((i+1)); done < $J; }
for ((L = 0; L < NL; L++)); do lane $L & done; wait
echo "$(date +%H:%M:%S) SETTLE_AB 끝"
