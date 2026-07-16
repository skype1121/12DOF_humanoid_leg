# Humanoid 12DOF Real Control

Main PC에서 UI와 ROS2 topic을 통해 12축 AK MIT 모터를 직접 제어하는 workspace입니다.

## Current State

- UI -> ROS2 -> Stage8 node -> `can0` -> 12축 실물 제어 확인 완료
- `REAL_CAN_WRITE_ENABLED = True`
- CAN channel: `can0`
- CAN bitrate: `1 Mbps`
- command topic: `/humanoid/stage8_12axis_mit_command`
- status topic: `/humanoid/stage8_12axis_mit_status`
- 12축 motor ID: `1..12`

## Run

터미널 1:

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_stage8_12axis_node.sh
```

터미널 2:

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_ui_ros.sh
```

## CAN Bringup

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0
```

## Runtime Settings

- control tick: `20 ms`
- command ramp: `4 deg/tick` (`~200 deg/s`)
- UI command delta limit: `30 deg/command`
- status publish: `100 Hz`
- read request: one motor every `20 ms`
- 12-axis feedback sweep: about `0.24 s`

## Safe Start Order

1. Power and wiring check.
2. Bring up `can0`.
3. Start Stage8 node.
4. Start UI.
5. `CAN 상태확인`.
6. `대상 연결`.
7. Actual values 확인.
8. `현재자세 기준설정`.
9. `제어 준비`.
10. Small single-joint command first.

## Keep

Do not edit `건들지마 메모장이야.txt`.
