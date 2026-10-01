#!/bin/bash
# G17 사전 기준 평가 조건 10개(G17_CRITERIA.md §1) — 이미 있는 결과는 건너뜀.
# 사용: eval17.sh <load_run> <checkpoint 파일> <태그 접두> <seed...>
# 출력: log_picture/ground_recovery_<접두>_<조건>_s<seed>.txt
cd /home/ryu/humanoid_leg_test1; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
RUN=$1; CK=$2; PFX=$3; shift 3
declare -A X=( [nd]="" [ndg2]="--gyro_delay 2" [f8]="--dist push:f:2:8:0.3" [b8]="--dist push:b:2:8:0.3"
  [sl8]="--dist ramp:l:2:8:0.5:1.0" [sf10]="--dist ramp:f:2:10:0.5:1.0" [sb10]="--dist ramp:b:2:10:0.5:1.0"
  [sf15]="--dist ramp:f:2:15:0.5:1.0" [b18]="--dist push:b:2:18:0.3" [sf10g2]="--gyro_delay 2 --dist ramp:f:2:10:0.5:1.0" )
L=${EVAL_LOGDIR:-/tmp}
for SD in "$@"; do for C in nd ndg2 f8 b8 sl8 sf10 sb10 sf15 b18 sf10g2; do
  t=${PFX}_${C}_s${SD}
  [ -f log_picture/ground_recovery_${t}.txt ] && grep -q '사전 검사 통과' log_picture/ground_recovery_${t}.txt && continue
  setsid /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py --headless --task Biped12-HangStand-G11a-v0 \
    --num_envs 128 --load_run $RUN --checkpoint $CK --delay_steps 1,1 --precheck_s 3 --after_s 8 --seed $SD \
    --enc_obs v2 --contact_real ${X[$C]} --out_tag $t > $L/ev17_$t.l 2>&1 &
  pg=$!; while kill -0 $pg 2>/dev/null; do sleep 3; done; kill -KILL -- -$pg 2>/dev/null
  echo "$(date +%H:%M) done $t $(grep -c '사전 검사 통과' log_picture/ground_recovery_${t}.txt 2>/dev/null)"
done; done
