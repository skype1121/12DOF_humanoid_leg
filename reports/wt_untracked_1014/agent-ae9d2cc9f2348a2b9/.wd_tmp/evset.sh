#!/bin/bash
# evset.sh RUN CK PFX [full|dist]
cd /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9
RN=$1 CK=$2 P=$3 M=${4:-full}; E=.wd_tmp/ev.sh
for S in 1 2; do for D in f b l r; do [ -s log_picture/w3eval_${P}_dist_${D}_s$S.txt ] && continue
  $E $RN $CK ${P}_dist_${D}_s$S --cmd 0.06,0,0 --ramp_off --seed $S --num_envs 320 --dist_jitter 0.91 --dist "push:$D:6:0:0.3|push:$D:6:8:0.3|push:$D:6:12:0.3|push:$D:6:15:0.3|push:$D:6:18:0.3"
done
  $E $RN $CK ${P}_stop_s$S --cmd 0.06,0,0 --ramp_off --seed $S --num_envs 320 --dist_jitter 0.91 --dist "push:f:6:0:0.3|push:f:6:12:0.3|push:f:6:12:0.3;stop:0.3|push:f:6:18:0.3|push:f:6:18:0.3;stop:0.3"
done
[ $M = dist ] && exit 0
$E $RN $CK ${P}_distmix_s1 --cmd 0.06,0,0.2 --ramp_off --seed 1 --num_envs 320 --dist_jitter 0.91 --dist "push:f:6:0:0.3|push:f:6:12:0.3|push:f:6:18:0.3|push:l:6:12:0.3|push:b:6:15:0.3"
for c in "c0:0,0,0" "f06:0.06,0,0" "b04:-0.04,0,0" "l05:0,0.05,0" "r05:0,-0.05,0" "tl02:0,0,0.2" "tr02:0,0,-0.2" "fl02:0.06,0,0.2" "fr02:0.06,0,-0.2" "all3:0.05,0.05,0.2"; do
  $E $RN $CK ${P}_reg_${c%%:*} --cmd ${c#*:} --seed 1 --num_envs 256 --seconds 30 --turn_metrics
done
$E $RN $CK ${P}_reg_d4f06 --cmd 0.06,0,0 --seed 1 --num_envs 256 --seconds 30 --delay 4
$E $RN $CK ${P}_reg_d4all3 --cmd 0.05,0.05,0.2 --seed 1 --num_envs 256 --seconds 30 --delay 4 --turn_metrics
$E $RN $CK ${P}_hl_c0 --cmd 0,0,0 --seed 1 --num_envs 256 --long --ramp_off --heading_loop 0.5
$E $RN $CK ${P}_hl_f06 --cmd 0.06,0,0 --seed 1 --num_envs 256 --long --ramp_off --heading_loop 0.5
$E $RN $CK ${P}_reg_stop --cmd 0.06,0,0 --seed 1 --num_envs 256 --seconds 20 --stop_at 12
$E $RN $CK ${P}_reg_stand --cmd 0,0,0 --seed 1 --num_envs 256 --seconds 30 --gait_enable 0
