#!/bin/bash
# G21h 후보 평가(reports/gen_w3/AIR_START_1008.md §8) — 사용: g21h_eval.sh <run 이름> <체크포인트 번호> "<단계 목록>"
#   단계: A1·A2(공중 시작 5 팔, seed 1·2) · AR(견고성: 공중 20 s·방출 1 s) · A3·A4(확인용 새 seed) · B1(서기 35 s) · B2(KE 회복) · B3(걷기 버튼) · B4(혼합 왕복)
#   태그: 러너 루프 = tr_h<N>_…, 서기 단독 = rg_h<N>_s_nd_s#, KE = keq_KEh<N>_<조건>_d2_s#. 이미 유효한 태그는 건너뛴다.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
RUN=$1; N=$2; STAGES=${3:-A1}; RD=logs/rsl_rl/biped12_stage4/$RUN; EX=$RD/exported_$N
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD EVAL_POLL=${EVAL_POLL:-1} EVAL_YIELD=${EVAL_YIELD:-1}
[ -f $RD/model_$N.pt ] || { echo "체크포인트 없음 $RD/model_$N.pt"; exit 2; }
if [ ! -f $EX/policy.onnx ]; then      # 내보내기(시뮬 없이) — 같은 폴더를 여러 체인이 동시에 만들지 않게 잠금
  exec 5>"$S/g21h/export_$N.lock"; flock 5
  [ -f $EX/policy.onnx ] || /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/export_ckpt_offline.py $RD model_$N.pt --ref logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.pt --out_name exported_$N > $S/g21h/export_$N.log 2>&1
  flock -u 5
fi
[ -f $EX/policy.onnx ] && [ -f $EX/policy.pt ] || { echo "내보내기 실패 — $S/g21h/export_$N.log"; exit 2; }
okr() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
runtr() { local t=$1; shift; okr $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
KW="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --delay_steps 2,2 --trace_all 128 --kw_stand_exported $EX"
air() {  # 팔 seed
  local A=$1 SD=$2 X
  case $A in
    C)     X="--precheck_s 4.5 --air_start clear:0.05,0.15;hold:3;rel:4 --after_s 42 --kw_script move@35.5:fwd1";;
    A2)    X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1";;
    A2i)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:4 --after_s 46 --kw_script move@39:fwd1 --air_det_hold";;
    A6)    X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1";;
    A6i)   X="--precheck_s 3 --air_start clear:0.05,0.15;hold:9;rel:4 --after_s 50 --kw_script move@43:fwd1 --air_det_hold";;
    A20i)  X="--precheck_s 3 --air_start clear:0.05,0.15;hold:23;rel:4 --after_s 64 --kw_script move@57:fwd1 --air_det_hold";;
    A2ir1) X="--precheck_s 3 --air_start clear:0.05,0.15;hold:5;rel:1 --after_s 43 --kw_script move@36:fwd1 --air_det_hold";;
  esac
  TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 runtr tr_h${N}_${A}_s${SD} $KW --seed $SD $X
}
S14="move@2:fwd1;move@7:turn_l;move@13:turn_r;move@19:left1;move@24:right1;move@29:back1;move@34:fwd3;move@41:inplace;move@47:fwd1;move@52:fwd1"
SO="walk@2:0.06,0,0;stop@6;walk@11:0,0,0.15;stop@15;walk@20:0,0.035,0;stop@24;walk@29:-0.04,0,0;stop@33;walk@38:0.06,0,0;stop@42"
for ST in $STAGES; do
  case $ST in
    A1|A2|A3|A4) SD=${ST#A}; for A in C A2i A6i A2 A6; do air $A $SD; done;;
    AR) air A20i 1; air A2ir1 1;;
    B1) for SD in 1 2 3; do T=rg_h${N}_s_nd_s$SD
          okr $T && { echo "skip $T"; continue; }
          exec 9>"/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eval17.gpu.lock.lane3"; flock 9
          until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 10500 ]; do sleep 30; done
          echo "$(date +%H:%M) 시작 $T"
          timeout 900 /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_ground_recovery.py --headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --enc_obs v2 --contact_real --contact_ver v2 --delay_steps 2,2 --load_run $RUN --checkpoint model_$N.pt --last_action_ema 0.27 --obs_ema joint_vel:0.25 --seed $SD --after_s 35 --out_tag $T > $S/g21h/$T.l 2>&1
          okr $T && echo "$(date +%H:%M) 유효 $T" || echo "$(date +%H:%M) ⚠ 무효 $T"; flock -u 9
        done;;
    B2) for SD in 21 22 23; do for C in nd60 b18 sf15; do
          case $C in nd60) X="--after_s 60"; TM=2700;; b18) X="--dist push:b:4:18:0.3"; TM=1500;; sf15) X="--dist ramp:f:4:15:0.5:1.0"; TM=1500;; esac
          TR_SCRIPT=rl_walking/scripts/eval_transition_dev.py EVAL_TIMEOUT=$TM runtr keq_KEh${N}_${C}_d2_s$SD --dir ke2w3 --ke_mode none --delay_steps 2,2 --seed $SD $X --trace_all 128 --ke $RUN:model_$N.pt
        done; done;;
    B3) for SD in 1 2; do TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 runtr tr_h${N}_x14_d2_s$SD $KW --seed $SD --after_s 58 --kw_script "$S14"; done;;
    B4) for SD in 1 2; do TR_SCRIPT=rl_walking/scripts/eval_transition_kwx.py EVAL_TIMEOUT=3000 EVAL_ENVS=256 runtr tr_h${N}_x7_d2_s$SD $KW --seed $SD --after_s 48 --kw_script "$SO"; done;;
  esac
done
echo "$(date +%H:%M) G21H_EVAL_DONE $N [$STAGES]"
