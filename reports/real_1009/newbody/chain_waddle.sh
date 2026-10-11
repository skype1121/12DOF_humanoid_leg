#!/bin/bash
# 1011 WADDLE-W1: 학습 끝 → 내보내기·계약 폴더 → 걷기 단독(제자리·앞) + 왕복(서기→제자리→정지→앞→정지→앞→정지, 지연 2, 걸음 시계 맞춤). 최종본만. 공용 잠금.
# 사용: chain_waddle.sh <팔 글자 W|X> <걸음 시계 Hz> <앞 속도>
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1; PY=/home/ryu/IsaacLab/_isaac_sim/python.sh; A=$1; F=$2; V=$3; N=19996; a=$(echo $A | tr 'A-Z' 'a-z'); SEED=1
J="walk@2:0,0,0;stop@14;walk@19:$V,0,0;stop@34;walk@39:$V,0,0;stop@49"
if [ "$A" = BASE ]; then
  X=$R/logs/rsl_rl/biped12_stepb0/2026-10-11_05-09-59_w3s3dnbje_s1_from_17798/deploy_18997/exported; [ -s $X/policy.onnx ] || { echo "기준선 내보내기 없음 $X"; exit 4; }
  g=$HERE/chk/jobs_wad_base.tsv; printf "w_wadbase_v%s_s1\t256\t2\t1\t55\t%s\t-\t%s\tnb\t--trace_qd\n" ${V#0.} $X "$J" > $g
  TR_TIMEOUT=2400 $HERE/run_chk.sh $g; echo "$(date '+%H:%M:%S') CHAIN_WADDLE_BASE_DONE"; exit 0; fi
O=$HERE/logs/train_chain_w3s3dnbj${a}_s1.out
until grep -q "NB_TRAIN_DONE seed $SEED" $O 2>/dev/null; do grep -qE "중단|넘어가지 않음" $O 2>/dev/null && { echo "학습 실패 — 중단"; exit 3; }; sleep 20; done
RUN=$(cat $HERE/logs/train_w3s3dnbj${a}_s${SEED}_from_18997.run); D=$R/logs/rsl_rl/biped12_stepb0/$RUN
(cd $R && $PY rl_walking/scripts/export_ckpt_offline.py $D model_$N.pt --out_name exported_$N 2>&1 | tail -2 && $PY $HERE/make_deploy_contract.py $D $N 2>&1 | tail -1)
[ -s $D/deploy_$N/exported/policy.onnx ] || { echo "내보내기 실패 $A"; exit 4; }
g=$HERE/chk/jobs_wad_$a.tsv; printf "w_wad%s_s1\t256\t2\t1\t55\t%s\t-\t%s\tnb\t--trace_qd --kw_w3_fstep %s\n" $A $D/deploy_$N/exported "$J" $F > $g
echo "$(date '+%H:%M:%S') === WADDLE $A 왕복"; TR_TIMEOUT=2400 $HERE/run_chk.sh $g
f=$HERE/w3/jobs_wad_$a.tsv; : > $f
printf "wad%s_z_s1\tW3S3dNBj%s\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 30 --delay 2 --cmd=0,0,0\n" $A $A $RUN $N >> $f
printf "wad%s_f_s1\tW3S3dNBj%s\t%s\tmodel_%s.pt\t1\t--warmup 2 --num_envs 384 --seconds 30 --delay 2 --cmd=%s,0,0\n" $A $A $RUN $N $V >> $f
echo "$(date '+%H:%M:%S') === WADDLE $A 걷기 단독"; $HERE/run_w3nb.sh $f
echo "$(date '+%H:%M:%S') CHAIN_WADDLE_${A}_DONE"
