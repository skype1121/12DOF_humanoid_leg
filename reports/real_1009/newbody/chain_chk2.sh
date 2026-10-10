#!/bin/bash
# 1010 점검 연쇄 2: 연쇄 1(P1·P7) 끝난 뒤 P2 → P3 → P6 → P4 → P5. 연쇄 1 이 죽으면(PID 소멸) 그대로 진행.
D=$(cd "$(dirname "$0")" && pwd)
until grep -q P7A_DONE $D/chk/chain1.out 2>/dev/null; do kill -0 $1 2>/dev/null || break; sleep 15; done
for p in p2 p3 p6 p4 p5; do $D/run_chk.sh $D/chk/jobs_$p.tsv; echo ${p}_DONE; done
echo CHAIN2_DONE
