#!/bin/bash
# 축소판(1010 01:58 사용자 결정): seed 2 학습 최종본(23292) 한 팔만 — 같은 19 회 구성·--lock_obs_zero. 순차·공용 잠금(run_lock2.sh).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1
U=urdf=$R/rl_walking/assets/biped12_final3_hiprpla.urdf:left_hip_a_joint,right_hip_a_joint,left_hip_r_joint,right_hip_r_joint; J=left_hip_r_joint,right_hip_r_joint
until grep -q "NB_TRAIN_DONE seed 2" $HERE/logs/train_chain.out 2>/dev/null; do
  kill -0 $(cat $HERE/logs/train_chain.pid) 2>/dev/null || { echo "학습 연쇄가 seed 2 완료 없이 끝남"; echo "EVAL_CHAIN2_DONE(no s2)"; exit 0; }
  sleep 30; done
S2=$(cat $HERE/logs/train_g24ranb_s2_from_21793.run); f=$HERE/ev/jobs_c2_23292.tsv; : > $f
for m in fi40 sf15 b18 nd30 l18 r18; do
  case $m in nd30) A=30; D=-;; l18) A=8; D=push:l:2:18:0.3;; r18) A=8; D=push:r:2:18:0.3;; b18) A=8; D=push:b:2:18:0.3;; fi40) A=8; D=push:f:2:40:0.2;; sf15) A=8; D=ramp:f:2:15:0.5:1.0;; esac
  for s in 1 2 3 $([ $m = b18 ] && echo 4); do printf "nb_c2_23292_%s_s%s\t384\t%s\t%s\t%s\t%s\t%s\n" $m $s $A $J $D $U $s >> $f; done
done
echo "$(date '+%m-%d %H:%M:%S') === 팔 c2_23292 (REC=$S2:model_23292.pt)"
REC=$S2:model_23292.pt EXTRA="--lock_obs_zero" LOCK_BAND=0 $HERE/run_lock2.sh $f
echo "$(date '+%m-%d %H:%M:%S') EVAL_CHAIN2_DONE"
