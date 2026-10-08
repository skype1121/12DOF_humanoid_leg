#!/bin/bash
# AIR2(reports/gen_w3/AIR_START_1008.md §10) — 3 단 러너(AIR → KE ⇄ W3) 평가. 사용: air2_q.sh <공중 정책 exported 폴더> "<seed 목록>" "<팔 목록>" [태그 접두=tr_air2]
#   공중 팔: A2 A6 A2r1(힘 모형) · L2 L20 L60 L2f L2s(줄 길이 모형)   대조 팔(공중 러너 없음, 홀드로 착지 뒤 인계): CL6 CLf CLs   지면 동등 확인: GX14
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
AX=$1; SEEDS=${2:-1 2}; ARMS=${3:-A2 L2}; PF=${4:-tr_air2}
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=4500 EVAL_ENVS=256 EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-1}
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ] && [ -f log_picture/kwloop_$1.npz ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
CM="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128"
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
for SD in $SEEDS; do for A in $ARMS; do
  AIR="--kw_air_exported $AX"
  case $A in
    A2)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1";;
    A6)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1";;
    A2r1) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1 --after_s 43 --kw_script move@36:fwd1";;
    L2)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:6;model:len --after_s 48 --kw_script move@41:fwd1";;
    L20)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:23;rel:6;model:len --after_s 66 --kw_script move@59:fwd1";;
    L60)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:63;rel:6;model:len --after_s 106 --kw_script move@99:fwd1";;
    L2f)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:2;model:len --after_s 44 --kw_script move@37:fwd1";;
    L2s)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:15;model:len --after_s 57 --kw_script move@50:fwd1";;
    CL6)  AIR=""; X="--precheck_s 10.5 --air_start clear:0.05,0.15;hold:3;rel:6;model:len --after_s 40 --kw_script move@33:fwd1";;
    CLf)  AIR=""; X="--precheck_s 6.5 --air_start clear:0.05,0.15;hold:3;rel:2;model:len --after_s 40 --kw_script move@33:fwd1";;
    CLs)  AIR=""; X="--precheck_s 19.5 --air_start clear:0.05,0.15;hold:3;rel:15;model:len --after_s 40 --kw_script move@33:fwd1";;
    GX14) X="--after_s 58 --kw_script $S14";;
  esac
  run ${PF}_${A}_s${SD} $CM --seed $SD $AIR $X
done; done
echo "$(date +%H:%M) AIR2_DONE [$SEEDS] [$ARMS]"
