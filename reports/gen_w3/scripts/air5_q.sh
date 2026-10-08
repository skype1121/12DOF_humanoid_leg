#!/bin/bash
# AIR5(AIR_START_1008.md §15) — G21h2 판정. 작업 트리에서 돈다. 사용: air5_q.sh "<seed 목록>" "<팔 목록>" [접두=tr_air5] [추세 확인용 exported 폴더]
#   판정 팔(등록 키 그대로): T(기운 매달림 −6~+6°) A2 A2r1 L2 L20 L2s   추세 팔(AIR2 구성 + --kw_air_exported): Tx
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
export EVAL_REPO=${EVAL_REPO:-$S/wt2}; cd $EVAL_REPO || exit 2
SEEDS=${1:-3}; ARMS=${2:-T}; PF=${3:-tr_air5}; AX=${4:-}
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=4500 EVAL_ENVS=256 EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr_wt.sh $t "$@"; }
KEY=stand_kew3qa_g21a_g24ra_w3s3d15800
CMK="--ke_mode direct --runner_loop --delay_steps 2,2 --kw_composite_key $KEY"
CMX="--ke_mode direct --runner_loop --delay_steps 2,2 --kw_requiet 2.0,0.3 --kw_air_exported $AX"
M=";fix:1;yaw:1"
TA="--precheck_s 3 --after_s 54 --kw_script move@47:fwd1 --air_start clear:0.05,0.15;hold:11;rel:6;model:len$M;p0:-6,6 --trace_all 256"
for SD in $SEEDS; do for A in $ARMS; do
  CM=$CMK
  case $A in
    T)    X="$TA";;
    Tx)   CM=$CMX; X="$TA";;
    A2)   X="--trace_all 128 --precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4$M --after_s 46 --kw_script move@39:fwd1";;
    A2r1) X="--trace_all 128 --precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1$M --after_s 43 --kw_script move@36:fwd1";;
    L2)   X="--trace_all 128 --precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:6;model:len$M --after_s 48 --kw_script move@41:fwd1";;
    L20)  X="--trace_all 128 --precheck_s 3 --air_start clear:0.05,0.15;hold:23;rel:6;model:len$M --after_s 66 --kw_script move@59:fwd1";;
    L2s)  X="--trace_all 128 --precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:15;model:len$M --after_s 57 --kw_script move@50:fwd1";;
    *) echo "모르는 팔 $A"; exit 2;;
  esac
  run ${PF}_${A}_s${SD} $CM --seed $SD $X
done; done
echo "$(date +%H:%M) AIR5_DONE [$SEEDS] [$ARMS] [$PF]"
