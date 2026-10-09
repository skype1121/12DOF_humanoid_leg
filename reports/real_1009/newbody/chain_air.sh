#!/bin/bash
# 1010 새 몸 공중 시작 평가(NEWBODY_WALK.md §13): NB seed 3·5 + 원래 몸 대조 seed 3. 순차.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); L=/home/ryu/humanoid_leg_test1/logs/rsl_rl
X2=$L/biped12_stepb0/2026-10-10_03-05-58_w3s3dnbj_s2_from_15800/deploy_16299/exported; AX=$L/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported
SC="walk@47:0,0,0;stop@52"
export TR_TRACE=256 TR_TIMEOUT=4500 TR_EXTRA="--kw_air_exported $AX --air_start clear:0.05,0.15;hold:11;rel:6;model:len;fix:1;yaw:1;p0:-6,6"
f=$HERE/tr/jobs_air_nb.tsv; : > $f; for s in 3 5; do printf "trnbair_T_s%s\t256\t2\t%s\t60\t%s\t-\t%s\n" $s $s $X2 "$SC" >> $f; done
TR_BODY=nb $HERE/run_trnb.sh $f
g=$HERE/tr/jobs_air_ob.tsv; printf "trobair_T_s3\t256\t2\t3\t60\t-\t-\t%s\n" "$SC" > $g
TR_BODY=ob $HERE/run_trnb.sh $g
echo "$(date +%H:%M:%S) CHAIN_AIR_DONE"
