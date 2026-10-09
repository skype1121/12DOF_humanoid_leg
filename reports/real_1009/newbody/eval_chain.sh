#!/bin/bash
# 새 몸 후보 평가 연쇄(사전 기준 NEWBODY_TRAIN.md §5): 팔마다 6 측정 × 평가 seed 1–3(뒤 18 N 은 1–4, 무효 대비). 순차·공용 잠금(run_lock2.sh).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=/home/ryu/humanoid_leg_test1
U=urdf=$R/rl_walking/assets/biped12_final3_hiprpla.urdf:left_hip_a_joint,right_hip_a_joint,left_hip_r_joint,right_hip_r_joint; J=left_hip_r_joint,right_hip_r_joint
S1=2026-10-10_00-46-17_g24ranb_s1_from_21793; CT=2026-10-04_17-30-51_g24rcont_seed1_from_g24ra21793
jobs() {   # $1 = 팔 이름, $2 = 측정 목록
  local f=$HERE/ev/jobs_$1.tsv; : > $f
  for m in $2; do
    case $m in nd30) A=30; D=-;; l18) A=8; D=push:l:2:18:0.3;; r18) A=8; D=push:r:2:18:0.3;; b18) A=8; D=push:b:2:18:0.3;; fi40) A=8; D=push:f:2:40:0.2;; sf15) A=8; D=ramp:f:2:15:0.5:1.0;; esac
    for s in 1 2 3 $([ $m = b18 ] && echo 4); do printf "nb_%s_%s_s%s\t384\t%s\t%s\t%s\t%s\t%s\n" $1 $m $s $A $J $D $U $s >> $f; done
  done; echo $f
}
arm() {   # $1 팔, $2 REC(빈칸 = 기존 G24Ra-21793), $3 EXTRA, $4 측정 목록
  echo "$(date '+%m-%d %H:%M:%S') === 팔 $1 (REC=${2:-기존} EXTRA=$3)"
  REC=$2 EXTRA="$3" LOCK_BAND=0 $HERE/run_lock2.sh $(jobs $1 "$4")
}
waitf() { until [ -s "$1" ]; do sleep 30; done; sleep 20; }
mkdir -p $HERE/ev
ALL="fi40 sf15 b18 nd30 l18 r18"
arm bz "" "--lock_obs_zero" "$ALL"                                   # 기준선(관측 0 켬)
waitf $R/logs/rsl_rl/biped12_stepb0/$S1/model_23292.pt
for c in 23292 22800 22300; do arm c1_$c "$S1:model_$c.pt" "--lock_obs_zero" "$ALL"; done
for c in 23292 22800 22300; do arm ct_$c "$CT:model_$c.pt" "--lock_obs_zero" "$ALL"; done
arm bo "" "" "fi40 sf15 b18 nd30"                                    # 기준선(관측 0 끔, 같은 반폭 0) — 켬/끔 비교용
until grep -q "NB_TRAIN_DONE seed 2" $HERE/logs/train_chain.out 2>/dev/null; do
  kill -0 $(cat $HERE/logs/train_chain.pid 2>/dev/null) 2>/dev/null || { echo "학습 연쇄가 seed 2 완료 없이 끝남 — seed 2 평가 생략"; echo "EVAL_CHAIN_DONE(no s2)"; exit 0; }
  sleep 30; done
S2=$(cat $HERE/logs/train_g24ranb_s2_from_21793.run)
for c in 23292 22800 22300; do arm c2_$c "$S2:model_$c.pt" "--lock_obs_zero" "$ALL"; done
echo "$(date '+%m-%d %H:%M:%S') EVAL_CHAIN_DONE"
