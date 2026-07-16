# 12DOF Real Control Current State

## Verified Path

```text
UI --ros
  -> /humanoid/stage8_12axis_mit_command
  -> stage8_12axis_mit_control_node
  -> can0
  -> AK MIT motors 1..12
```

Status path:

```text
can0 feedback
  -> stage8_12axis_mit_control_node
  -> /humanoid/stage8_12axis_mit_status
  -> UI motor status / actual display
```

## Runtime Values

- `REAL_CAN_WRITE_ENABLED = True`
- `CAN_CHANNEL = "can0"`
- `CAN_BITRATE = 1000000`
- `CONTROL_PERIOD_SEC = 0.02`
- `MAX_TARGET_STEP_DEG_PER_TICK = 4.0`
- `CAN_FEEDBACK_REQUEST_PERIOD_SEC = 0.02`
- `STATUS_PERIOD_SEC = 0.01`
- UI ROS spin: `10 ms`

## Feedback Rate

- one read request every `20 ms`
- 12 motors per full sweep
- full sweep: about `0.24 s`
- per-axis update: about `4.17 Hz`
- status topic publish: `100 Hz`

## Notes

- 12-axis real control has been confirmed through the UI.
- Baseline is set from current actual positions.
- Joint commands are relative to the captured baseline.
- UI command delta limit is `30 deg/command`.
