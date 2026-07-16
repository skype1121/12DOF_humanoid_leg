# Main PC Initial Setup

## ROS2

```bash
source /opt/ros/humble/setup.bash
```

## CAN

Expected adapter:

```text
can0: PEAK PCAN-USB
```

Bringup:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0
```

Expected:

```text
state UP
bitrate 1000000
can state ERROR-ACTIVE
```

## Run

Stage8 node:

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_stage8_12axis_node.sh
```

UI:

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_ui_ros.sh
```

## First Motion

1. Confirm CAN status in the UI.
2. Connect target.
3. Confirm actual values.
4. Set baseline from current posture.
5. Arm.
6. Move one joint by a small amount first.
