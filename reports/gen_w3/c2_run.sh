#!/bin/bash
# c2 GPU 실행기(1006): 워크트리 코드 + 본 트리 자료(logs/·log_picture/). 학습 없음.
# 규칙: GPU 사용 메모리 ≤ 10500 MiB 가 될 때까지 대기, 시뮬 1개씩(eval17 과 같은 잠금), PhysX/CUDA 메모리 실패 로그면 무효 → 재실행(최대 3),
#       종료는 명시 PID(프로세스 그룹)로만.
# 사용:
#   c2_run.sh eval <load_run> <ckpt> <태그접두> <delay> <seed...>      — 회복 평가(nd sf15 fs80 fi40 b18, eval17 정의)  [EVAL_CONDS 로 조건 바꿈]
#   c2_run.sh cal  <load_run> <ckpt> <태그접두> <seed>                 — C2CAL 궤적(nd march·nd deploy·sf15/fi40/b18 nostop)
#   c2_run.sh unit <load_run> <ckpt> <태그접두> <seed>                 — C2UNIT 반사실 보상(B1–B6, 학습 env W3C2)
WT=$(cd "$(dirname "$0")/../.." && pwd); DATA=/home/ryu/humanoid_leg_test1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$WT EVAL_DATA_REPO=$DATA W3EVAL_REPO=$DATA
LOCK=${EVAL_LOCK:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock}
LOGD=${C2_LOGDIR:-/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/c2logs}; mkdir -p $LOGD
declare -A X=( [nd]="" [sf15]="--dist ramp:f:2:15:0.5:1.0" [fs80]="--dist push:f:2:80:0.1" [fi40]="--dist push:f:2:40:0.2" [b18]="--dist push:b:2:18:0.3" )
COMMON="--task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --after_s 8 --enc_obs v2 --contact_real --last_action_ema 0.27 --obs_ema joint_vel:0.25 --contact_ver v2"   # 1006 밤: 접촉 관측 v2(학습·배포 계약, KABS 와 같음) — 그 전 c2 평가는 v1

wait_mem() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le 10500 ] && return; sleep 20; done; }
bad_log() { grep -aqE 'Unable to allocate|PhysX.*(error|Error).*(memory|alloc)|CUDA error|out of memory|Failed to allocate' "$1"; }

# run_one <완료 판정 명령(문자열)> <로그> <명령...>
run_one() {
  local okcmd=$1 lg=$2; shift 2
  exec 9>$LOCK; flock 9
  for TRY in 1 2 3 4 5 6; do
    wait_mem
    setsid "$@" > $lg 2>&1 &
    local pg=$!; echo "$(date +%H:%M:%S) 시작 PID $pg: ${lg##*/}"
    while kill -0 $pg 2>/dev/null; do sleep 3; done
    kill -KILL -- -$pg 2>/dev/null
    if eval "$okcmd" && ! bad_log $lg; then echo "$(date +%H:%M:%S) 완료 ${lg##*/}"; break; fi
    echo "$(date +%H:%M:%S) 무효(메모리 실패 또는 산출물 없음) → 재시도 $TRY: ${lg##*/}"; sleep 60
  done
  flock -u 9
}

MODE=$1; RUN=$2; CK=$3; PFX=$4; shift 4
case $MODE in
eval)
  D=$1; shift
  for SD in "$@"; do for C in ${EVAL_CONDS:-nd sf15 fs80 fi40 b18}; do
    t=${PFX}_${C}_s${SD}; f=$DATA/log_picture/ground_recovery_${t}.txt
    rm -f $f
    run_one "grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' $f 2>/dev/null && [ -f $DATA/log_picture/ground_recovery_${t}_envs.csv ]" $LOGD/$t.l \
      /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/eval_ground_recovery.py --headless $COMMON --load_run $RUN --checkpoint $CK \
      --delay_steps $D,$D --seed $SD --c2 ${C2_EXTRA:-} ${X[$C]} --out_tag $t
  done; done ;;
cal)
  SD=$1
  for spec in ${CAL_SPECS:-"ndmarch:nd:march:" "nddeploy:nd:deploy:" "nddeprel:nd:deploy:tilt_ref_tau=10" "sf15ns:sf15:deploy:min_td=1e6,tmax_s=1e9" "fi40ns:fi40:deploy:min_td=1e6,tmax_s=1e9" "b18ns:b18:deploy:min_td=1e6,tmax_s=1e9"}; do
    IFS=: read nm C ARM PRM <<< "$spec"
    t=${PFX}_${nm}_s${SD}; f=$DATA/log_picture/c2trace_${t}.npz; rm -f $f
    run_one "[ -f $f ]" $LOGD/$t.l \
      /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/eval_ground_recovery.py --headless $COMMON --load_run $RUN --checkpoint $CK \
      --delay_steps 2,2 --seed $SD --c2 --c2_arm $ARM ${PRM:+--c2_params $PRM} --c2_trace ${X[$C]} --out_tag $t
  done ;;
unit)
  SD=$1
  for B in ${UNIT_ARMS:-B1 B2 B3 B4 B5 B6}; do
    t=${PFX}_${B}_s${SD}; f=$DATA/log_picture/c2unit_${t}.npz; rm -f $f
    run_one "[ -f $f ]" $LOGD/$t.l \
      /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/c2_landscape.py --headless --task ${UNIT_TASK:-Biped12-HangStand-W3C2-v0} --load_run $RUN --checkpoint $CK --arm $B --seed $SD --out_tag $t
  done ;;
ke)   # KE 기준선 재측정(KABS 인자 그대로, 고친 평가기) — c2_run.sh ke - - <태그접두> <seed...>
  for SD in "$@"; do for C in ${EVAL_CONDS:-nd sf15 fs80 fi40 b18}; do
    t=${PFX}_${C}_s${SD}; f=$DATA/log_picture/ground_recovery_${t}.txt; rm -f $f
    run_one "grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' $f 2>/dev/null && [ -f $DATA/log_picture/ground_recovery_${t}_envs.csv ]" $LOGD/$t.l \
      /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/eval_ground_recovery.py --headless --task Biped12-HangStand-G11a-v0 --num_envs 128 \
      --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt --precheck_s 3 --after_s 8 --seed $SD --enc_obs v2 --contact_real \
      --last_action_ema 0.27 --obs_ema joint_vel:0.25 --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 \
      --switch_to 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt --delay_steps 2,2 --sw_off_contact any --sw_rec_vx 0.03 --sw_rec_vx_s 1 \
      --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33 --contact_ver v2 ${X[$C]} --out_tag $t
  done; done ;;
sweep)
  SD=$1
  for CUE in B3 B6; do for spec in ${SWEEP_SPECS:-k1:stop_td_k=1 k2:stop_td_k=2 k3:stop_td_k=3 k4:stop_td_k=4 or:stop_oracle=1 ov:stop_oracle=2}; do
    IFS=: read nm PRM <<< "$spec"
    t=${PFX}_${CUE}_${nm}_s${SD}; f=$DATA/log_picture/c2unit_${t}.npz; rm -f $f
    run_one "[ -f $f ]" $LOGD/$t.l \
      /home/ryu/IsaacLab/isaaclab.sh -p $WT/rl_walking/scripts/c2_landscape.py --headless --task ${UNIT_TASK:-Biped12-HangStand-W3C2v-v0} --load_run $RUN --checkpoint $CK \
      --arm X --c2_params $PRM $([ $CUE = B6 ] && echo --cue) --seed $SD --out_tag $t
  done; done ;;
esac
