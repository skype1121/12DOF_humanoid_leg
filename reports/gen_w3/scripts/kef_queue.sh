#!/bin/bash
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
until grep -q S3TK_TRAIN_DONE $S/s3tk/chain.out 2>/dev/null; do sleep 15; done
mkdir -p $S/kef; exec $S/ketrain.sh G24RaF kef 2026-10-04_10-08-41_g24ra_seed1_from_w2e20294 21793 500
