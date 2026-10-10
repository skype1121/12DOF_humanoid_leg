#!/bin/bash
# 1010 밤 NBjT(과요구 벌점) 최종본 평가: seed 별 내보내기 → 걷기 단독 3 종 → 왕복(추적 qd) → 왕복(hip_a 한계 0.8 배). 순차·공용 잠금.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; PY=/home/ryu/IsaacLab/_isaac_sim/python.sh; N=17098
J="walk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39"
for SEED in "$@"; do
  RUN=$(cat $HERE/logs/train_w3s3dnbjt_s${SEED}_from_16299.run); D=$R/logs/rsl_rl/biped12_stepb0/$RUN
  (cd $R && $PY rl_walking/scripts/export_ckpt_offline.py $D model_$N.pt --out_name exported_$N 2>&1 | tail -2 && $PY $HERE/make_deploy_contract.py $D $N 2>&1 | tail -1)
  [ -s $D/deploy_$N/exported/policy.onnx ] || { echo "내보내기 실패 $SEED"; exit 4; }
  g=$HERE/chk/jobs_nbjt_s$SEED.tsv
  { printf "t_nbjt%s_e10_s1\t256\t2\t1\t45\t%s\t-\t%s\tnb\t--trace_qd\n" $SEED $D/deploy_$N/exported "$J"
    printf "t_nbjt%s_ha08_s1\t256\t2\t1\t45\t%s\t-\t%s\tnb\t--trace_qd --kw_effort_scales 0.8 --kw_effort_joints hip_a\n" $SEED $D/deploy_$N/exported "$J"; } > $g
  $HERE/run_chk.sh $g
  f=$HERE/w3/jobs_nbjt_s$SEED.tsv; : > $f
  printf "nbjt%s_z_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0\n" $SEED $RUN $N >> $f
  printf "nbjt%s_f_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0.06,0,0\n" $SEED $RUN $N >> $f
  printf "nbjt%s_stop_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 14 --delay 2 --cmd=0.06,0,0 --stop_at 8\n" $SEED $RUN $N >> $f
  $HERE/run_w3nb.sh $f
done
echo "$(date '+%H:%M:%S') CHAIN_NBJT_DONE"
