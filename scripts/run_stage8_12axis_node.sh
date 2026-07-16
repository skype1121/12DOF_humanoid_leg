#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

grep "REAL_CAN_WRITE_ENABLED =" jetson/scripts/stage8_12axis_mit_control_node.py
grep "CAN_CHANNEL =" jetson/scripts/stage8_12axis_mit_control_node.py
python3 jetson/scripts/stage8_12axis_mit_control_node.py
