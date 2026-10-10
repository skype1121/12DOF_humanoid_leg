#!/bin/bash
# 1010 밤: 힙 외전 토크 한계만 낮춘 왕복(실물 모터 8 정지 뒤 — 목표·실측 오차 제한의 시뮬 대응). 0.8·0.9 배, seed 1.
HERE=$(cd "$(dirname "$0")" && pwd)
TR_EXTRA="--kw_effort_scales 0.8 --kw_effort_joints hip_a" $HERE/run_trnb.sh $HERE/tr/jobs_ha08.tsv
TR_EXTRA="--kw_effort_scales 0.9 --kw_effort_joints hip_a" $HERE/run_trnb.sh $HERE/tr/jobs_ha09.tsv
echo CHAIN_HA_DONE
