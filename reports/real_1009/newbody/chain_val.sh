#!/bin/bash
# 1010 종종걸음 실물 직전 검증(NEWBODY_WALK.md §10): T1·T2 전환 왕복 → T3·T4 걷기 단독 → 팔 R(회복 = NB-s1 23292). 순차.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); L=/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stepb0
R1=2026-10-10_02-51-28_w3s3dnbj_s1_from_15800; R2=2026-10-10_03-05-58_w3s3dnbj_s2_from_15800
X1=$L/$R1/deploy_16299/exported; X2=$L/$R2/deploy_16299/exported; XR=$L/2026-10-10_00-46-17_g24ranb_s1_from_21793/deploy_23292/exported
SJ="walk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39"
f=$HERE/tr/jobs_p1.tsv; : > $f
for s in 1 2; do
  printf "trnb_J1_d2_s%s\t256\t2\t%s\t45\t%s\t-\t%s\n" $s $s $X1 "$SJ" >> $f
  printf "trnb_J2_d2_s%s\t256\t2\t%s\t45\t%s\t-\t%s\n" $s $s $X2 "$SJ" >> $f
  printf "trnb_B_d2_s%s\t256\t2\t%s\t45\t-\t-\t%s\n" $s $s "$SJ" >> $f
done
printf "trnb_J1_d4_s1\t256\t4\t1\t45\t%s\t-\t%s\n" $X1 "$SJ" >> $f; printf "trnb_J2_d4_s1\t256\t4\t1\t45\t%s\t-\t%s\n" $X2 "$SJ" >> $f
$HERE/run_trnb.sh $f
g=$HERE/w3/jobs_val.tsv; : > $g
for c in "1 $R1" "2 $R2"; do set -- $c
  printf "nbj%s_16299_d4z_s1\tW3S3dNB\t%s\tmodel_16299.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 4 --cmd=0,0,0\n" $1 $2 >> $g
  printf "nbj%s_16299_d4f_s1\tW3S3dNB\t%s\tmodel_16299.pt\t1\t--warmup 2 --num_envs 384 --seconds 15 --delay 4 --cmd=0.06,0,0\n" $1 $2 >> $g
  printf "nbj%s_16299_long_s1\tW3S3dNB\t%s\tmodel_16299.pt\t1\t--warmup 2 --num_envs 128 --long --ramp_off --delay 2 --cmd=0.06,0,0\n" $1 $2 >> $g
done
$HERE/run_w3nb.sh $g
# 팔 R: 지연 2 왕복 낙상이 낮은 쪽(같으면 J1)
PICK=$(python3 - <<PY
import csv
def fr(a):
    k=n=0
    for s in (1,2):
        try: R=[r for r in csv.DictReader(open(f"$HERE/tr/out/ground_recovery_trnb_{a}_d2_s{s}_envs.csv")) if r["pre_pass"]=="1"]
        except Exception: return 9.0
        k+=sum(r["fell"]=="1" for r in R); n+=len(R)
    return k/max(n,1)
print("J2" if fr("J2") < fr("J1") else "J1")
PY
)
echo "$(date +%H:%M:%S) 팔 R 대상 = $PICK"; XP=$([ $PICK = J2 ] && echo $X2 || echo $X1); h=$HERE/tr/jobs_p2.tsv; : > $h
for s in 1 2; do printf "trnb_%sR_d2_s%s\t256\t2\t%s\t45\t%s\t%s\t%s\n" $PICK $s $s $XP $XR "$SJ" >> $h; done
$HERE/run_trnb.sh $h
echo "$(date +%H:%M:%S) CHAIN_VAL_DONE"
