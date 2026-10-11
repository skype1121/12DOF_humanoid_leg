#!/bin/bash
# 1011 NBjD 최종본 평가: seed 별 학습 끝 → (폭발 여부 확인은 사람이) 내보내기 → 왕복 3 종(명목·hip_a 0.7·hip_a 0.6, ptg 추적) → 걷기 단독 3 종. 순차·공용 잠금.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; PY=/home/ryu/IsaacLab/_isaac_sim/python.sh; N=${NBJH_N:-20196}
J="walk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39"
for SEED in "$@"; do
  until grep -q "NB_TRAIN_DONE seed $SEED" $HERE/logs/train_chain_nbjh_s${SEED}.out 2>/dev/null; do
    grep -qE "중단|넘어가지 않음" $HERE/logs/train_chain_nbjh_s${SEED}.out 2>/dev/null && { echo "학습 실패 — 중단"; exit 3; }; sleep 20; done
  RUN=$(cat $HERE/logs/train_w3s3dnbjh_s${SEED}_from_18997.run); D=$R/logs/rsl_rl/biped12_stepb0/$RUN
  (cd $R && $PY rl_walking/scripts/export_ckpt_offline.py $D model_$N.pt --out_name exported_$N 2>&1 | tail -2 && $PY $HERE/make_deploy_contract.py $D $N 2>&1 | tail -1)
  [ -s $D/deploy_$N/exported/policy.onnx ] || { echo "내보내기 실패 $SEED"; exit 4; }
  g=$HERE/chk/jobs_nbjh_s$SEED.tsv; X=$D/deploy_$N/exported
  { printf "x_nbjh%s_e10_s1\t256\t2\t1\t45\t%s\t-\t%s\tnb\t--trace_qd\n" $SEED $X "$J"
    printf "x_nbjh%s_ha08_s1\t256\t2\t1\t45\t%s\t-\t%s\tnb\t--trace_qd --kw_effort_scales 0.8 --kw_effort_joints hip_a\n" $SEED $X "$J"
    printf "x_nbjh%s_ha07_s1\t256\t2\t1\t45\t%s\t-\t%s\tnb\t--trace_qd --kw_effort_scales 0.7 --kw_effort_joints hip_a\n" $SEED $X "$J"; } > $g
  $HERE/run_chk.sh $g
  f=$HERE/w3/jobs_nbjh_s$SEED.tsv; : > $f
  printf "nbjh%s_z_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0,0,0\n" $SEED $RUN $N >> $f
  printf "nbjh%s_f_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 2 --cmd=0.06,0,0\n" $SEED $RUN $N >> $f
  printf "nbjh%s_stop_s1\tW3S3dNB\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 14 --delay 2 --cmd=0.06,0,0 --stop_at 8\n" $SEED $RUN $N >> $f
  $HERE/run_w3nb.sh $f
done
echo "$(date '+%H:%M:%S') CHAIN_NBJL_DONE"
