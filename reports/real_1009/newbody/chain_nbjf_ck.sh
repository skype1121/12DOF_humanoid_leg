#!/bin/bash
# 1010 NBjF 붕괴 전 체크포인트 평가(seed 1 은 반복 ~16950 뒤 학습 폭발 → 최종본 무효). 사용: chain_nbjf_ck.sh <seed> <N> [<N> ...] — 앞 0.06 걷기 단독 + 왕복 45 s.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; PY=/home/ryu/IsaacLab/_isaac_sim/python.sh; F=1.5; SEED=$1; shift
RUN=$(cat $HERE/logs/train_w3s3dnbjf_s${SEED}_from_16299.run); D=$R/logs/rsl_rl/biped12_stepb0/$RUN
for N in "$@"; do
  (cd $R && $PY rl_walking/scripts/export_ckpt_offline.py $D model_$N.pt --out_name exported_$N 2>&1 | tail -2 && $PY $HERE/make_deploy_contract.py $D $N 2>&1 | tail -1)
  [ -s $D/deploy_$N/exported/policy.onnx ] || { echo "내보내기 실패 $N"; exit 4; }
  f=$HERE/w3/jobs_nbjf_s${SEED}_$N.tsv; : > $f
  printf "nbjf%s_%s_z_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0\n" $SEED $N $RUN $N >> $f
  printf "nbjf%s_%s_f_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0.06,0,0\n" $SEED $N $RUN $N >> $f
  printf "nbjf%s_%s_stop_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 14 --delay 2 --cmd=0.06,0,0 --stop_at 8\n" $SEED $N $RUN $N >> $f
  $HERE/run_w3nb.sh $f
  g=$HERE/tr/jobs_jf_s${SEED}_$N.tsv
  printf "trnb_JF%s_%s_d2_s1\t256\t2\t1\t45\t%s\t-\twalk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39\n" $SEED $N $D/deploy_$N/exported > $g
  TR_EXTRA="--kw_w3_fstep $F" $HERE/run_trnb.sh $g
done
echo "$(date '+%H:%M:%S') CK_DONE $SEED $*"
