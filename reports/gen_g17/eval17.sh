#!/bin/bash
# G17 사전 기준 평가 조건 10개(G17_CRITERIA.md §1) — 이미 있는 결과는 건너뜀.
# EVAL_EXTRA: 후보 계약 인자(예: G18f 는 "--last_action_ema 0.27")
# 사용: eval17.sh <load_run> <checkpoint 파일> <태그 접두> <seed...>
# 출력: log_picture/ground_recovery_<접두>_<조건>_s<seed>.txt
cd /home/ryu/humanoid_leg_test1; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
RUN=$1; CK=$2; PFX=$3; shift 3
declare -A X=( [nd]="" [ndg2]="--gyro_delay 2" [f8]="--dist push:f:2:8:0.3" [b8]="--dist push:b:2:8:0.3"
  [sl8]="--dist ramp:l:2:8:0.5:1.0" [sf10]="--dist ramp:f:2:10:0.5:1.0" [sb10]="--dist ramp:b:2:10:0.5:1.0"
  [sf15]="--dist ramp:f:2:15:0.5:1.0" [b18]="--dist push:b:2:18:0.3" [sf10g2]="--gyro_delay 2 --dist ramp:f:2:10:0.5:1.0"
  [ndg3]="--gyro_delay 3" [d0nd]="" [d3nd]="" [d5nd]="" [d5b8]="--dist push:b:2:8:0.3" )
declare -A DL=( [d0nd]=0 [d3nd]=3 [d5nd]=5 [d5b8]=5 )
CONDS=${EVAL_CONDS:-"nd ndg2 ndg3 d0nd d3nd f8 b8 sl8 sf10 sb10 sf15 b18 sf10g2"}   # d5nd d5b8 = 보고 전용, 정식에서 EVAL_CONDS 로 추가
L=${EVAL_LOGDIR:-/tmp}
for SD in "$@"; do for C in $CONDS; do D=${DL[$C]:-1}
  t=${PFX}_${C}_s${SD}
  [ -f log_picture/ground_recovery_${t}.txt ] && grep -q '사전 검사 통과' log_picture/ground_recovery_${t}.txt && continue
  for TRY in 1 2; do   # Kit 시작 크래시(1002 04:55 1건) 재시도
  setsid /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py --headless --task Biped12-HangStand-G11a-v0 \
    --num_envs 128 --load_run $RUN --checkpoint $CK --delay_steps $D,$D --precheck_s 3 --after_s 8 --seed $SD \
    --enc_obs v2 --contact_real ${EVAL_EXTRA:-} ${X[$C]} --out_tag $t > $L/ev17_$t.l 2>&1 &
  pg=$!; while kill -0 $pg 2>/dev/null; do sleep 3; done; kill -KILL -- -$pg 2>/dev/null
  grep -q '사전 검사 통과' log_picture/ground_recovery_${t}.txt 2>/dev/null && break; echo "$(date +%H:%M) 재시도 $t"; sleep 10; done
  echo "$(date +%H:%M) done $t $(grep -c '사전 검사 통과' log_picture/ground_recovery_${t}.txt 2>/dev/null)"
done; done
