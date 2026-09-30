#!/bin/bash
# 젯슨에서 실행. 실물 can1 은 절대 쓰지 않는다 (가상 can9 + ROS_DOMAIN_ID 77).

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
R=~/humanoid_repo; BT=$R/jetson/bench_tmp; W=/tmp/s8bench; mkdir -p $BT $W/out
python3 $W/make_variants.py $R/jetson/scripts/stage8_12axis_mit_control_node.py $BT
ip link show can9 >/dev/null 2>&1 || { echo "$JETSON_PW" | sudo -S modprobe vcan; echo "$JETSON_PW" | sudo -S ip link add dev can9 type vcan; echo "$JETSON_PW" | sudo -S ip link set up can9; }
ip -br link show can9
for v in ${VARIANTS:-v0 v1 v2 v3}; do
  python3 $W/responder.py can9 $W/out/can_$v.json 36 & RP=$!
  sleep 1
  python3 $BT/stage8_bench_$v.py --can-channel can9 > $W/out/s8_$v.log 2>&1 & SP=$!
  sleep 2
  python3 $W/driver.py $W/out/drv_$v.json 25 > $W/out/drv_$v.log 2>&1
  kill $SP; wait $SP 2>/dev/null; wait $RP 2>/dev/null
  echo "$v 완료 $(date +%H:%M:%S)"; tail -2 $W/out/drv_$v.log
done
