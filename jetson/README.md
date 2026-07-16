# 12DOF Real Control Node

`jetson/scripts/stage8_12axis_mit_control_node.py` is the active 12-axis AK MIT control node.

Current deployment is main PC first:

```bash
cd /home/ryu/humanoid_leg_test1
./scripts/run_stage8_12axis_node.sh
```

The node uses:

- command topic: `/humanoid/stage8_12axis_mit_command`
- status topic: `/humanoid/stage8_12axis_mit_status`
- CAN channel: `can0`
- bitrate: `1 Mbps`
- `REAL_CAN_WRITE_ENABLED = True`
