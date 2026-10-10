#!/bin/bash
# 1010 빠른 걸음 NBjF: seed 별 학습 끝 → 내보내기·계약 폴더 → 걷기 단독 3 종(제자리·앞 0.06·정지) + 왕복 45 s(지연 2). 최종본만. 순차·공용 잠금.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; PY=/home/ryu/IsaacLab/_isaac_sim/python.sh; F=1.5; N=17798
for SEED in "$@"; do
  until grep -q "NB_TRAIN_DONE seed $SEED" $HERE/logs/train_chain_nbjf.out 2>/dev/null; do
    grep -qE "중단|다음 seed 로 넘어가지 않음" $HERE/logs/train_chain_nbjf.out 2>/dev/null && { echo "학습 실패 — 중단"; exit 3; }; sleep 20; done
  RUN=$(cat $HERE/logs/train_w3s3dnbjf_s${SEED}_from_16299.run); D=$R/logs/rsl_rl/biped12_stepb0/$RUN
  (cd $R && $PY rl_walking/scripts/export_ckpt_offline.py $D model_$N.pt --out_name exported_$N 2>&1 | tail -4 && $PY $HERE/make_deploy_contract.py $D $N 2>&1 | tail -2)
  [ -s $D/deploy_$N/exported/policy.onnx ] || { echo "내보내기 실패 seed $SEED"; exit 4; }
  f=$HERE/w3/jobs_nbjf_s$SEED.tsv; : > $f
  printf "nbjf%s_z_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0\n" $SEED $RUN $N >> $f
  printf "nbjf%s_f_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0.06,0,0\n" $SEED $RUN $N >> $f
  printf "nbjf%s_stop_s1\tW3S3dNBjF\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 14 --delay 2 --cmd=0.06,0,0 --stop_at 8\n" $SEED $RUN $N >> $f
  echo "$(date '+%H:%M:%S') === NBjF seed $SEED ($RUN) 걷기 단독"; $HERE/run_w3nb.sh $f
  g=$HERE/tr/jobs_jf_s$SEED.tsv
  printf "trnb_JF%s_d2_s1\t256\t2\t1\t45\t%s\t-\twalk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39\n" $SEED $D/deploy_$N/exported > $g
  echo "$(date '+%H:%M:%S') === NBjF seed $SEED 왕복"; TR_EXTRA="--kw_w3_fstep $F" $HERE/run_trnb.sh $g
done
echo "$(date '+%H:%M:%S') CHAIN_NBJF_DONE"
