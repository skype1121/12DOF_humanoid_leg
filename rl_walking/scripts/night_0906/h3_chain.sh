#!/bin/bash
# h3 재판정 체인: v18b → v17 (순차, GPU 독점). 결과: judge_v18b_h3.txt / judge_v17_h3.txt
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
echo $$ > $S/h3_chain.pid
echo "h3_chain $$ $(date +%F_%T)" >> $S/pids.txt
USD=rl_walking/assets/usd_final3/biped12.usd
bash $S/judge_gen_h3.sh v18b_h3 2026-09-05_19-37-39 model_120998.pt Biped12-Velocity-Stage4V2-Stand-V18-v0 $USD
bash $S/judge_gen_h3.sh v17_h3  2026-09-04_21-48-32 model_126997.pt Biped12-Velocity-Stage4V2-Stand-V17-v0 $USD
echo "[$(date +%T)] h3 체인 종료" | tee -a $S/judge_v17_h3.txt
