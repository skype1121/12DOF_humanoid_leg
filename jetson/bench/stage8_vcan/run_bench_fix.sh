#!/bin/bash
# 캐시 수정본 벤치: 배포본(~/humanoid_repo)을 건드리지 않고 ~/s8bench_root 사본에서 실행. 가상 can9 + 도메인 77.
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=77
R=~/s8bench_root; BT=$R/jetson/bench_tmp; W=/tmp/s8bench; mkdir -p $W/out
python3 $W/make_variants.py $R/jetson/scripts/stage8_12axis_mit_control_node.py $BT >/dev/null
python3 -c "import sys; sys.path.insert(0,'$R'); import robot_runtime.dof12_mapping as m; print('캐시판' if hasattr(m,'_cached_config') else '원본')"
for v in ${VARIANTS:-v0 v1}; do
  python3 $W/responder.py can9 $W/out/can_fix_$v.json 36 & RP=$!
  sleep 1
  python3 $BT/stage8_bench_$v.py --can-channel can9 > $W/out/s8_fix_$v.log 2>&1 & SP=$!
  sleep 2
  python3 $W/driver.py $W/out/drv_fix_$v.json 25 > $W/out/drv_fix_$v.log 2>&1
  kill $SP; wait $SP 2>/dev/null; wait $RP 2>/dev/null
  echo "fix_$v 완료 $(date +%H:%M:%S)"; tail -1 $W/out/drv_fix_$v.log
done
