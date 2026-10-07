#!/bin/bash
# ROBUST_DESIGN §3 평가 체인: E2 코너(선택) → E1 스윕 → E3 120 s 비대칭 → 회귀 빠른 모드 → seed 2 확인 → 판정
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
until grep -q ROB_TRAIN_DONE $S/rob/train.out 2>/dev/null; do sleep 60; done
RN=$(grep ROB_TRAIN_DONE $S/rob/train.out | tail -1 | awk '{print $2}'); B=2026-10-07_00-11-12_w3S3d_from_s2d; BC=15800
E2="--cmds f06:0.06,0,0 --grid imu_pitch=-3,-1.5 --grid com_x=0.02 --seconds 15"
SW="imu_pitch=-6,-3,3,6;com_x=0.03,0.04,-0.04;play=0.5,2"
run() { local t=$1; shift; [ -f log_picture/w3env_$t.json ] && { echo "skip $t"; return; }; $S/safe_env.sh $t "$@"; }
CKS=""; for ck in 16000 16100 16300; do [ -f logs/rsl_rl/biped12_stepb0/$RN/model_$ck.pt ] && CKS="$CKS $ck"; done
[ -z "$CKS" ] && CKS=$(ls logs/rsl_rl/biped12_stepb0/$RN/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
echo "run $RN 후보 $CKS"
for ck in $CKS; do run rob_r${ck}_e2_s1 $RN $ck $E2 --seed 1; done
run rob_b_e2_s1 $B $BC $E2 --seed 1
PK=$(python3 reports/gen_w3/rob_select.py pick $(for ck in $CKS; do echo -n "rob_r${ck}_e2_s1 "; done) | grep '^PICK' | awk '{print $2}'); CK=$(echo $PK | sed 's/rob_r//;s/_e2_s1//')
echo "$(date +%H:%M) 선택 체크포인트 $CK (E2 코너 최소)"; echo "$CK" > $S/rob/pick.txt
for who in "rob_b $B $BC" "rob_r$CK $RN $CK"; do set -- $who
  run ${1}_e1f_s1 $2 $3 --cmds f06:0.06,0,0 --sweep "$SW" --seconds 15 --seed 1
  run ${1}_e1z_s1 $2 $3 --cmds z:0,0,0 --sweep "$SW" --seconds 15 --seed 1
  run ${1}_e3h_s1 $2 $3 --cmds "z:0,0,0|f06:0.06,0,0" --fix hipf_L=-4 --heading_loop 0.5 --ramp_off --seconds 120 --seed 1
  run ${1}_e3k_s1 $2 $3 --cmds "z:0,0,0|f06:0.06,0,0" --fix kp_L=1.3 --heading_loop 0.5 --ramp_off --seconds 120 --seed 1
done
reports/regress/run_regress.sh --mode quick --w3 $RN:model_$CK.pt --only w3,st,dist --tag rob$CK > $S/rob/regress_$CK.out 2>&1; echo "$(date +%H:%M) 회귀 빠른 모드 끝: $(grep -E 'PASS|FAIL|경계' $S/rob/regress_$CK.out | tail -3 | tr '\n' ' ')"
for who in "rob_b $B $BC" "rob_r$CK $RN $CK"; do set -- $who
  run ${1}_e2_s2 $2 $3 $E2 --seed 2
  run ${1}_e1f_s2 $2 $3 --cmds f06:0.06,0,0 --sweep "$SW" --seconds 15 --seed 2
done
python3 reports/gen_w3/rob_select.py judge rob_b rob_r$CK 1 > $S/rob/judge_s1.txt 2>&1; python3 reports/gen_w3/rob_select.py judge rob_b rob_r$CK 2 > $S/rob/judge_s2.txt 2>&1
echo "$(date +%H:%M) ROB_EVAL_DONE $RN $CK"
