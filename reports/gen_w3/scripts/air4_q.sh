#!/bin/bash
# AIR4 탐색(AIR_START_1008.md §14) — 기운 채 매달림(2 점 현수 목표 피치 p0)에서 3 단 러너의 공중 거동. 작업 트리에서 돈다(EVAL_REPO).
#   사용: air4_q.sh "<seed 목록>" "<팔 목록>"   팔: PToff(차단 끔·p0 −6~+6°) PTon(차단 기본) PT0off(p0 0 = 대조) PXoff(p0 −10~+10°)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
export EVAL_REPO=${EVAL_REPO:-$S/wt2}; cd $EVAL_REPO || exit 2
SEEDS=${1:-3}; ARMS=${2:-PToff}; PF=${3:-tr_air4}
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=4500 EVAL_ENVS=256 EVAL_POLL=1 EVAL_YIELD=1
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr_wt.sh $t "$@"; }
KEY=stand_kew3qa_g21a_g24ra_w3s3d15800
CM="--ke_mode direct --runner_loop --delay_steps 2,2 --trace_all 256 --kw_composite_key $KEY"
B="--precheck_s 3 --after_s 54 --kw_script move@47:fwd1"
A="clear:0.05,0.15;hold:11;rel:6;model:len;fix:1;yaw:1"
for SD in $SEEDS; do for X in $ARMS; do
  case $X in
    PToff)  E="--air_start $A;p0:-6,6 --kw_air_guard off";;
    PTon)   E="--air_start $A;p0:-6,6";;
    PT0off) E="--air_start $A --kw_air_guard off";;
    PXoff)  E="--air_start $A;p0:-10,10 --kw_air_guard off";;
    PXon)   E="--air_start $A;p0:-10,10";;
    *) echo "모르는 팔 $X"; exit 2;;
  esac
  run ${PF}_${X}_s${SD} $CM --seed $SD $B $E
done; done
echo "$(date +%H:%M) AIR4_DONE [$SEEDS] [$ARMS]"
