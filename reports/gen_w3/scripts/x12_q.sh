#!/bin/bash
# x12: W3 걷기 명령별 부하(Σ τ²) — 배포 코드 폐루프, nd, seed 21, 128 env, 추적(관절 토크 포함). 판정 없는 측정.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwc.py EVAL_TIMEOUT=3000
SC="walk@2:0,0,0;walk@8:0.03,0,0;walk@14:0.06,0,0;walk@20:0.10,0,0;stop@26;walk@31:-0.04,0,0;walk@37:-0.06,0,0;stop@43;walk@48:0,0.035,0;stop@54;walk@59:0,0,0.15;stop@65"
$S/safe_tr.sh x12_w3load_cmds_d2_s21 --ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --kw_script "$SC" --after_s 70 --delay_steps 2,2 --seed 21 --trace_all 128
echo "$(date +%H:%M) X12L_DONE"
