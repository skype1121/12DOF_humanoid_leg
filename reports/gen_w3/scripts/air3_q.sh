#!/bin/bash
# AIR3(reports/gen_w3/AIR_START_1008.md §13) — 정정한 지지 모형(fix:1 = 제어 주기 고정 보정, yaw:1 = 2 줄 현수 요 복원)에서 3 단 러너(AIR → KE ⇄ W3) 재평가.
#   러너 = 브리지에 등록된 합성 키 그대로(--kw_composite_key — 평가용 덮어쓰기 없음). 사용: air3_q.sh "<seed 목록>" "<팔 목록>" [태그 접두=tr_air3]
#   공중 팔: A2 A6 A2r1(힘 모형) · L2 L20 L60 L2f L2s(줄 길이 모형)   대조 팔(홀드로 착지 뒤 인계, 매달림 시간 맞춤): C Cr1 CL6 CL20 CL60 CLf CLs
#   원인 대조(AIR2 구성 그대로·seed 1): xL20m(정정 모형) · xCL20o(종전 모형·홀드 20 s) · xCL20m(정정 모형·홀드 20 s)
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
SEEDS=${1:-3 4}; ARMS=${2:-A2 L2}; PF=${3:-tr_air3}
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=4500 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-1} EVAL_LANES=${EVAL_LANES:-4}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
KEY=stand_kew3qa_g21a_g24ra_w3s3d15800
AX=logs/rsl_rl/biped12_stage4/2026-10-08_13-16-02_g21h_s1_from_11300/exported_12299
CMK="--ke_mode direct --runner_loop --delay_steps 2,2 --trace_all 128 --kw_composite_key $KEY"
CM2="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
M=";fix:1;yaw:1"
for SD in $SEEDS; do for A in $ARMS; do
  CM=$CMK
  case $A in
    A2)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4$M --after_s 46 --kw_script move@39:fwd1";;
    A6)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4$M --after_s 50 --kw_script move@43:fwd1";;
    A2r1) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1$M --after_s 43 --kw_script move@36:fwd1";;
    L2)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:6;model:len$M --after_s 48 --kw_script move@41:fwd1";;
    L20)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:23;rel:6;model:len$M --after_s 66 --kw_script move@59:fwd1";;
    L60)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:63;rel:6;model:len$M --after_s 106 --kw_script move@99:fwd1";;
    L2f)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:2;model:len$M --after_s 44 --kw_script move@37:fwd1";;
    L2s)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:15;model:len$M --after_s 57 --kw_script move@50:fwd1";;
    C)    X="--precheck_s 4.5 --air_start clear:0.05,0.15;hold:3;rel:4$M --after_s 42 --kw_script move@35.5:fwd1";;
    Cr1)  X="--precheck_s 4.5 --air_start clear:0.05,0.15;hold:3;rel:1$M --after_s 40 --kw_script move@33:fwd1";;
    CL6)  X="--precheck_s 10.5 --air_start clear:0.05,0.15;hold:3;rel:6;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    CL20) X="--precheck_s 30.5 --air_start clear:0.05,0.15;hold:23;rel:6;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    CL60) X="--precheck_s 70.5 --air_start clear:0.05,0.15;hold:63;rel:6;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    CLf)  X="--precheck_s 6.5 --air_start clear:0.05,0.15;hold:3;rel:2;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    CLs)  X="--precheck_s 19.5 --air_start clear:0.05,0.15;hold:3;rel:15;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    xL20m)  CM="$CM2 --kw_air_exported $AX"; X="--precheck_s 3 --air_start clear:0.05,0.15;hold:23;rel:6;model:len$M --after_s 66 --kw_script move@59:fwd1";;
    xCL20o) CM="$CM2"; X="--precheck_s 30.5 --air_start clear:0.05,0.15;hold:23;rel:6;model:len --after_s 40 --kw_script move@33:fwd1";;
    xCL20m) CM="$CM2"; X="--precheck_s 30.5 --air_start clear:0.05,0.15;hold:23;rel:6;model:len$M --after_s 40 --kw_script move@33:fwd1";;
    *) echo "모르는 팔 $A"; exit 2;;
  esac
  run ${PF}_${A}_s${SD} $CM --seed $SD $X
done; done
echo "$(date +%H:%M) AIR3_DONE [$SEEDS] [$ARMS]"
