# Stage8 12-Axis Real Control Quickstart

## Topics

- command: `/humanoid/stage8_12axis_mit_command`
- status: `/humanoid/stage8_12axis_mit_status`

## Main PC Run

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_stage8_12axis_node.sh
```

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_ui_ros.sh
```

## Current Real Settings

- `REAL_CAN_WRITE_ENABLED = True`
- `CAN_CHANNEL = "can0"`
- `CAN_BITRATE = 1000000`
- all 12 joints are accepted
- mapping is joint-name based

## Motion Sequence

1. Bring up `can0`.
2. Start Stage8 node.
3. Start UI.
4. `CAN 상태확인`.
5. `대상 연결`.
6. Confirm actual values.
7. `현재자세 기준설정`.
8. `제어 준비`.
9. Move one joint first.

## CLI Commands

```bash
python3 scripts/send_12axis_mit_test_command.py --arm
python3 scripts/send_12axis_mit_test_command.py --baseline
python3 scripts/send_12axis_mit_test_command.py --joint left_knee_joint --deg 1.0
python3 scripts/send_12axis_mit_test_command.py --stop
```
