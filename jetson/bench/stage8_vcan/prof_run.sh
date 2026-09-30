#!/bin/bash
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=77
W=/tmp/s8bench; BT=~/humanoid_repo/jetson/bench_tmp
python3 $W/responder.py can9 $W/out/can_prof.json 36 & RP=$!
sleep 1
python3 -m cProfile -o $W/out/prof_v0.out $BT/stage8_bench_v0.py --can-channel can9 > $W/out/s8_prof.log 2>&1 & SP=$!
sleep 2
python3 $W/driver.py $W/out/drv_prof.json 25 > $W/out/drv_prof.log 2>&1
kill -INT $SP; wait $SP 2>/dev/null; wait $RP 2>/dev/null
python3 -c "
import pstats; p=pstats.Stats('$W/out/prof_v0.out'); p.sort_stats('cumulative').print_stats(28)" 2>&1 | tail -45
