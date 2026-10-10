#!/bin/bash
# 1010 점검 연쇄 1: 동일성 확인 → P1 → P7(전 층 4 + 양발 0 층 분리 3)
D=$(cd "$(dirname "$0")" && pwd)
CHK_EVAL=eval_transition_kwx_nb.py $D/run_chk.sh $D/chk/jobs_eq_a.tsv; $D/run_chk.sh $D/chk/jobs_eq_b.tsv; echo EQ_DONE
$D/run_chk.sh $D/chk/jobs_p1.tsv; echo P1_DONE
head -7 $D/chk/jobs_p7.tsv > $D/chk/jobs_p7a.tsv; $D/run_chk.sh $D/chk/jobs_p7a.tsv; echo P7A_DONE
