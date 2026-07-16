#!/usr/bin/env bash
set -e

PROJECT_ROOT="/home/ryu/humanoid_leg_test1"

cd "$PROJECT_ROOT"
source /opt/ros/humble/setup.bash
export PYTHONUNBUFFERED=1

echo "[run_ui_ros] project = $PROJECT_ROOT"
echo "[run_ui_ros] ROS_DISTRO = ${ROS_DISTRO:-unknown}"
echo "[run_ui_ros] ROS_DOMAIN_ID = ${ROS_DOMAIN_ID:-0}"
echo "[run_ui_ros] UI ROS mode enabled = True"
echo "[run_ui_ros] node name = humanoid_12dof_control_ui"
echo "[run_ui_ros] publish topic = /humanoid/stage8_12axis_mit_command"
echo "[run_ui_ros] subscribe topic = /humanoid/stage8_12axis_mit_status"
echo "[run_ui_ros] starting ui/main_ui.py --ros"

python3 ui/main_ui.py --ros
