#!/bin/bash
# 1010 종종걸음: (가) 외란 보충 '앞 18 N × 0.2 s' 9 회 → (나) NBj 학습 seed 별 최종본 W1–W5 평가(8 회/seed). 순차·공용 잠금.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1
U=urdf=$R/rl_walking/assets/biped12_final3_hiprpla.urdf:left_hip_a_joint,right_hip_a_joint,left_hip_r_joint,right_hip_r_joint; J=left_hip_r_joint,right_hip_r_joint
until grep -q "seed 1\] 학습 시작(정상)" $HERE/logs/train_chain_nbj.out 2>/dev/null; do sleep 15; done
for arm in "bz|" "c1_23292|2026-10-10_00-46-17_g24ranb_s1_from_21793:model_23292.pt" "c2_23292|2026-10-10_01-29-17_g24ranb_s2_from_21793:model_23292.pt"; do
  a=${arm%%|*}; rec=${arm#*|}; f=$HERE/ev/jobs_${a}_f18.tsv; : > $f
  for s in 1 2 3; do printf "nb_%s_f18_s%s\t384\t8\t%s\tpush:f:2:18:0.2\t%s\t%s\n" $a $s $J $U $s >> $f; done
  echo "$(date '+%H:%M:%S') === 앞 18 N 팔 $a"; REC=$rec EXTRA="--lock_obs_zero" LOCK_BAND=0 $HERE/run_lock2.sh $f
done
for SEED in 1 2; do
  until grep -q "NB_TRAIN_DONE seed $SEED" $HERE/logs/train_chain_nbj.out 2>/dev/null; do sleep 20; done
  RUN=$(cat $HERE/logs/train_w3s3dnbj_s${SEED}_from_15800.run); f=$HERE/w3/jobs_nbj_s$SEED.tsv; : > $f
  for s in 1 2; do
    printf "nbj%s_16299_z_s%s\tW3S3dNB\t%s\tmodel_16299.pt\t%s\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0\n" $SEED $s $RUN $s
    printf "nbj%s_16299_f_s%s\tW3S3dNB\t%s\tmodel_16299.pt\t%s\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0.06,0,0\n" $SEED $s $RUN $s
    printf "nbj%s_16299_stop_s%s\tW3S3dNB\t%s\tmodel_16299.pt\t%s\t--warmup 2 --num_envs 384 --seconds 14 --delay 2 --cmd=0.06,0,0 --stop_at 8\n" $SEED $s $RUN $s
    printf "nbj%s_16299_st_s%s\tW3S3dNB\t%s\tmodel_16299.pt\t%s\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0 --gait_enable 0\n" $SEED $s $RUN $s
  done >> $f
  echo "$(date '+%H:%M:%S') === NBj seed $SEED ($RUN) 평가"; $HERE/run_w3nb.sh $f
done
echo "$(date '+%H:%M:%S') CHAIN_NBJ_DONE"
