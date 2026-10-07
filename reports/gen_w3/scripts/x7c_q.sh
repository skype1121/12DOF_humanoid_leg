#!/bin/bash
# X7c(reports/gen_w3/TRANSITION_W3KE.md §5.11d) — 둘째 이후 걷기 낙상의 구조 후보 3 종 + 대조. 배포 코드 폐루프, nd, 갇힘 해소 옵션, seed 1–2, 256 env. 사본 평가기 kwc.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad; cd /home/ryu/humanoid_leg_test1 || exit 2
export TR_SCRIPT=rl_walking/scripts/eval_transition_kwc.py EVAL_TIMEOUT=3000 EVAL_ENVS=256
ok() { grep -q '사전 검사(무개입 자립 [0-9.]*s) 통과: [1-9]' log_picture/ground_recovery_$1.txt 2>/dev/null && [ -f log_picture/ground_recovery_$1_envs.csv ]; }
run() { local t=$1; shift; ok $t && { echo "skip $t"; return; }; $S/safe_tr.sh $t "$@"; }
# 대조: 걷기 4 s → 정지
S0="walk@2:0.06,0,0;stop@6;walk@11:0.06,0,0;stop@15;walk@20:0.06,0,0;stop@24;walk@29:0.06,0,0;stop@33;walk@38:0.06,0,0;stop@42"
# 부드러운 정지: 걷기 4 s → 명령 0 제자리 걸음 1.5 s → 정지
S2="walk@2:0.06,0,0;walk@6:0,0,0;stop@7.5;walk@11:0.06,0,0;walk@15:0,0,0;stop@16.5;walk@20:0.06,0,0;walk@24:0,0,0;stop@25.5;walk@29:0.06,0,0;walk@33:0,0,0;stop@34.5;walk@38:0.06,0,0;walk@42:0,0,0;stop@43.5"
# 부드러운 시작: 제자리 걸음 1.5 s → 앞 4 s → 정지
S3="walk@2:0,0,0;walk@3.5:0.06,0,0;stop@7.5;walk@11:0,0,0;walk@12.5:0.06,0,0;stop@16.5;walk@20:0,0,0;walk@21.5:0.06,0,0;stop@25.5;walk@29:0,0,0;walk@30.5:0.06,0,0;stop@34.5;walk@38:0,0,0;walk@39.5:0.06,0,0;stop@43.5"
C="--ke_mode direct --runner_loop --kw_requiet 2.0,0.3 --after_s 48 --delay_steps 2,2 --trace_all 128"
for SD in 1 2; do
  run tr_x7c_c0_d2_s${SD} $C --kw_script "$S0" --seed $SD
  run tr_x7c_c1_d2_s${SD} $C --kw_script "$S0" --kw_settle 1e9 --kw_settle_max 1e9 --seed $SD
  run tr_x7c_c2_d2_s${SD} $C --kw_script "$S2" --seed $SD
  run tr_x7c_c3_d2_s${SD} $C --kw_script "$S3" --seed $SD
done
echo "$(date +%H:%M) X7C_DONE"
