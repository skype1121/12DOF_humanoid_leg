[3gH    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H   [INFO] Using python from: /home/ryu/IsaacLab/_isaac_sim/python.sh
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: /home/ryu/IsaacLab/apps/isaaclab.python.headless.kit
[INFO]: Parsing configuration from: rl_walking.hang_stand_env_cfg:Biped12HangStandW3S4dDEnvCfg
[INFO]: Parsing configuration from: rl_walking.agents:Biped12StepB0PPORunnerCfg

[36m======================================================================================[0m
[36m[1m[INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_2026-10-07_05-03-32.log[0m
[36m======================================================================================[0m

[33m05:03:32 [simulation_context.py] WARNING: The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False. If you are experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to improve the accuracy of velocity updates.[0m
[INFO]: Base environment:
	Environment device    : cuda:0
	Environment seed      : 1
	Physics step-size     : 0.005
	Rendering step-size   : 0.02
	Environment step-size : 0.02
[INFO]: Time taken for scene creation : 1.107815 seconds
[INFO]: Scene manager:  <class InteractiveScene>
	Number of environments: 256
	Environment spacing   : 2.5
	Source prim name      : /World/envs/env_0
	Global prim paths     : ['/World/ground']
	Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
[INFO]: Time taken for simulation start : 0.733462 seconds
[33m05:03:34 [velocity_command.py] WARNING: The velocity command has the 'ranges.heading' attribute set to '(-3.141592653589793, 3.141592653589793)' but the heading command is not active. Consider setting the flag for the heading command to True.[0m
[INFO] Command Manager:  <CommandManager> contains 1 active terms.
+-------------------------------------------------+
|               Active Command Terms              |
+-------+---------------+-------------------------+
| Index | Name          |           Type          |
+-------+---------------+-------------------------+
|   0   | base_velocity | Stage4V2VelocityCommand |
+-------+---------------+-------------------------+

[INFO] Event Manager:  <EventManager> contains 3 active terms.
+--------------------------------------+
| Active Event Terms in Mode: 'startup' |
+---------+----------------------------+
|  Index  | Name                       |
+---------+----------------------------+
|    0    | physics_material           |
|    1    | add_base_mass              |
|    2    | base_com                   |
|    3    | actuator_gains             |
|    4    | joint_friction             |
|    5    | joint_armature             |
|    6    | link_mass_calib            |
|    7    | ankle_r_friction           |
|    8    | joint_limits_config        |
|    9    | w3_cmd                     |
|    10   | w3_ref                     |
+---------+----------------------------+
+---------------------------------------+
|  Active Event Terms in Mode: 'reset'  |
+--------+------------------------------+
| Index  | Name                         |
+--------+------------------------------+
|   0    | base_external_force_torque   |
|   1    | reset_base                   |
|   2    | reset_robot_joints           |
|   3    | calib_bias                   |
|   4    | stance_asym                  |
|   5    | hang_support_reset           |
|   6    | default_pose_bias            |
+--------+------------------------------+
+-----------------------------------------------------+
|        Active Event Terms in Mode: 'interval'       |
+-------+-------------------+-------------------------+
| Index | Name              | Interval time range (s) |
+-------+-------------------+-------------------------+
|   0   | hang_support_step |        (0.0, 0.0)       |
|   1   | w3_gait           |        (0.0, 0.0)       |
|   2   | w3_track          |        (0.0, 0.0)       |
|   3   | w3_lat            |        (0.0, 0.0)       |
|   4   | w3_turn           |        (0.0, 0.0)       |
|   5   | w3_mix            |        (0.0, 0.0)       |
|   6   | w3d_step          |        (0.0, 0.0)       |
|   7   | w3d_track         |        (0.0, 0.0)       |
+-------+-------------------+-------------------------+

[INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
+---------------------+
| Active Recorder Terms |
+-----------+---------+
|   Index   | Name    |
+-----------+---------+
+-----------+---------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+------------------------------------+
|  Active Action Terms (shape: 12)   |
+--------+-------------+-------------+
| Index  | Name        |   Dimension |
+--------+-------------+-------------+
|   0    | joint_pos   |          12 |
+--------+-------------+-------------+

[INFO] Observation Manager: <ObservationManager> contains 2 groups.
+---------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (320,)) |
+-----------+---------------------------------+-----------+
|   Index   | Name                            |   Shape   |
+-----------+---------------------------------+-----------+
|     0     | base_ang_vel                    |   (15,)   |
|     1     | projected_gravity               |   (15,)   |
|     2     | velocity_commands               |   (15,)   |
|     3     | joint_pos                       |   (60,)   |
|     4     | joint_vel                       |   (60,)   |
|     5     | actions                         |   (60,)   |
|     6     | gait_clock                      |   (10,)   |
|     7     | feet_contact                    |   (10,)   |
|     8     | gyro_ema                        |   (15,)   |
|     9     | joint_vel_ema                   |   (60,)   |
+-----------+---------------------------------+-----------+
+-------------------------------------------------------+
| Active Observation Terms in Group: 'critic' (shape: (15,)) |
+-------------+---------------------------+-------------+
|    Index    | Name                      |    Shape    |
+-------------+---------------------------+-------------+
|      0      | base_lin_vel              |    (15,)    |
+-------------+---------------------------+-------------+

[INFO] Termination Manager:  <TerminationManager> contains 4 active terms.
+------------------------------------+
|      Active Termination Terms      |
+-------+-----------------+----------+
| Index | Name            | Time Out |
+-------+-----------------+----------+
|   0   | time_out        |   True   |
|   1   | base_contact    |  False   |
|   2   | bad_orientation |  False   |
|   3   | base_too_low    |  False   |
+-------+-----------------+----------+

[INFO] Reward Manager:  <RewardManager> contains 58 active terms.
+-----------------------------------------------+
|              Active Reward Terms              |
+-------+----------------------------+----------+
| Index | Name                       |   Weight |
+-------+----------------------------+----------+
|   0   | track_lin_vel_xy_exp       |      2.0 |
|   1   | track_ang_vel_z_exp        |      2.0 |
|   2   | flat_orientation_l2        |     -8.0 |
|   3   | base_height_l2             |     -2.0 |
|   4   | lin_vel_z_l2               |     -0.2 |
|   5   | ang_vel_xy_l2              |     -0.1 |
|   6   | feet_slide                 |     -0.1 |
|   7   | feet_contact_forces        |   -0.002 |
|   8   | joint_deviation_hip        |     -0.5 |
|   9   | joint_deviation_ankle_roll |     -0.2 |
|   10  | joint_vel_l2               |  -0.0005 |
|   11  | dof_acc_l2                 | -2.5e-07 |
|   12  | dof_torques_l2             |  -0.0001 |
|   13  | action_rate_l2             |     -0.2 |
|   14  | dof_pos_limits             |     -5.0 |
|   15  | undesired_contacts         |     -1.0 |
|   16  | termination_penalty        |    -75.0 |
|   17  | feet_gap                   |     -5.0 |
|   18  | foot_yaw_slip              |     -0.5 |
|   19  | march_anchor               |     -1.0 |
|   20  | knee_stance_flexion        |      4.0 |
|   21  | feet_impact                |    -0.01 |
|   22  | torque_saturation          |     -2.0 |
|   23  | feet_contact_gated         |      2.0 |
|   24  | survive_after_release      |      1.0 |
|   25  | hang_quiet                 |   -0.002 |
|   26  | pose_symmetry              |     -3.0 |
|   27  | stance_shape               |     -5.0 |
|   28  | torso_pitch_band           |    -25.0 |
|   29  | hip_r_deviation            |     -3.0 |
|   30  | settle_quiet_bonus         |      2.0 |
|   31  | leg_pose_deviation         |     -1.5 |
|   32  | action_rate_window         |    -0.06 |
|   33  | recovery_quality           |      5.0 |
|   34  | recovery_event             |     30.0 |
|   35  | w3_alive                   |      0.5 |
|   36  | w3_periodic                |     -1.5 |
|   37  | w3_height                  |      3.0 |
|   38  | w3_valid                   |     15.0 |
|   39  | w3_place                   |     10.0 |
|   40  | w3_miss                    |    -10.0 |
|   41  | w3_wrong                   |     -5.0 |
|   42  | w3_drift                   |     -3.0 |
|   43  | w3_orient                  |     -2.0 |
|   44  | w3_angvel                  |    -0.05 |
|   45  | w3_arate                   |    -0.05 |
|   46  | w3_imit                    |      1.5 |
|   47  | w3_yaw                     |     -4.0 |
|   48  | w3_track                   |      3.0 |
|   49  | w3_stride                  |     10.0 |
|   50  | w3_width                   |    -10.0 |
|   51  | w3_turn                    |     10.0 |
|   52  | w3_ytrack                  |      3.0 |
|   53  | w3d_cap                    |     10.0 |
|   54  | w3d_rplace                 |     10.0 |
|   55  | w3d_rstride                |     10.0 |
|   56  | w3d_ret                    |     20.0 |
|   57  | w3d_sret                   |     20.0 |
+-------+----------------------------+----------+

[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+

[limits] config → {'left_ankle_f_joint': (-22.0, 39.0, -45.0, 40.0), 'right_ankle_f_joint': (-39.0, 18.5, -40.0, 45.0)}
[INFO]: Completed setting up the environment...
--------------------------------------------------------------------------------
Resolved observation sets: 
	 policy :  ['policy']
	 critic :  ['policy', 'critic']
--------------------------------------------------------------------------------
Actor MLP: MLP(
  (0): Linear(in_features=320, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=12, bias=True)
)
Critic MLP: MLP(
  (0): Linear(in_features=335, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
[eval_w3] 1600 틱 완료 · 136 s
# W3 Stage 0 채점표 · d500_reg_l05
- 태스크 Biped12-HangStand-W3S4dD-v0 · 정책 /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9/logs/rsl_rl/biped12_stepb0/2026-10-07_03-20-15_w3S4dD_from_s4d/model_16798.pt
- env 256 · seed 1 · 지연 2틱 · LPF β 0.27 · 워밍업 2.0s + 채점 30s · 외란 없음 · 학습 DR 유지
- 명령 (vx,vy,wz) = (+0.00,+0.05,+0.00) · gait_enable 1 · STEP_ASSIST_ALPHA 0.0 · W3_REF_SCALE 미설정 · TURN_ALPHA 0 · 위상 출처 env._gu · 보조 훅 없음

| 지표 | 값 |
|---|---|
| 이륙 성공률(예정 스윙 반주기 중 이륙) | 99.5 % (15400/15480) |
| 착지 성공률(이륙 대비) | 98.5 % (15491/15725) |
| 의미 있는 스윙(착지 대비, ≥60 ms·≥1.5 cm) | 98.2 % (15207) |
| 하중 수용률(착지 대비, 0.3 s 안 Fz>0.5 mg 0.1 s) | 99.8 % (15460) |
| 유효 스텝 | 15196 (왼 7437 / 오른 7759) · 2.13 /env/s |
| 교대율(연속 유효 스텝 좌우 교대) | 98.3 % (14700/14956) |
| 유효 cadence(보행 주기) | 1.063 Hz |
| 수평 표류(명령 대비 잔차 속도) | 0.026 m/s · 총 이동 1.72 m · 실제 앞 +0.019 / 옆 +0.051 m/s |
| 요 표류(명령 대비 |Δ요|) | 33.6° · 평균 요 속도 -0.98°/s |
| 낙상률 | 8.6 % (22/256) — 워밍업(시작 과도) 16 · 채점 구간 6 · 시각 0.5-1s:4 1-1.5s:11 1.5-2s:1 5-1000000000.0s:6 |
| 10 s 창별 유효 cadence | 1.05 → 1.09 → 1.05 Hz · 붕괴 없음 |
| 스텝 높이 cm (10/50/90 %) | 3.6/5.2/6.6 |
| 옆 변위 cm (착지, 10/50/90 %) | +2.2/+4.8/+7.4 · 명령 방향 부호 일치 99.2 % |
| 착지 발 간격 cm (10/50/90 %) | 30.2/33.8/37.5 · 23.8 미만 0.00 % · 교차 0.00 % · 좌/우 유효 스텝 비 0.96 · 롤 평균 -0.49° |
| 보폭 cm (수평 변위 10/50/90 %) | 2.6/5.0/7.6 · 앞축 -1.0/+0.6/+2.2 |
| 이륙 위상(예정 스윙 시작 대비 주기, 10/50/90 %) | +0.03/+0.06/+0.09 |
| 피치 RMS(정지 떨림, 참고) | 1.08° · 피치각속도 RMS 0.250 rad/s |
| 정책 행동 RMS | 0.766 (×scale 0.25 = 10.97°) |
| 참조 잔차 RMS(env._w3_ref_last) | 0.00° |

| 발 | 이륙 | 예정창 | 예정창 이륙 | 착지 | 의미 스윙 | 하중 수용 | 유효 |
|---|---|---|---|---|---|---|---|
| 왼 | 7855 | 7622 | 7610 | 7623 | 7438 | 7611 | 7437 |
| 오른 | 7870 | 7858 | 7790 | 7868 | 7769 | 7849 | 7759 |

## 회전 지표(Stage 3 — 명령 wz +0.00)
- 착지 발 요 변위 Δψ(유효 스텝, 10/50/90 %) -3.2/-0.9/+1.3° · 목표 +0.0° · 명령 부호 일치 nan %
- 요 이득(실제/명령) nan · 평균 요 속도 -0.98°/s
- 회전 출처(명령 방향 +): 스윙 착지 -1.89°/s · 지지발 피벗 +0.18°/s · 피벗 몫(부호) -0.11 · |피벗| 4.50°/s(떨림 포함 참고)
- 몸통 비틀림(골반 − 발 연결선) 평균 +1.5° · RMS 2.6°

## Stage 0 사전 기준 판정(STEP): FAIL
- PASS · 교대 > 90 % · 98.3 %
- PASS · 이륙 > 90 % · 99.5 %
- PASS · 착지 > 90 % · 98.5 %
- PASS · 하중 ≥ 85 % · 99.8 %
- FAIL · 채점 구간 낙상 < 2 % (시작 과도 제외 — 1006 판정 정의) · 6/240
- PASS · 표류 ≤ 0.03 m/s · 0.026 m/s
- PASS · 장시간 붕괴 없음 · 창 1.05/1.09/1.05 Hz

주: 접촉 = 발 링크 |F| > 5 N. 유효 cadence = 유효 스텝/(생존 env·s)/2. 표류 = 시작 요 프레임 변위/채점 시간 − 명령(vx,vy). 낙상 env 는 낙상 전까지의 사건만 집계. 이 판정은 시뮬 채점이며 실물 성공을 뜻하지 않는다.
[eval_w3] 저장: log_picture/w3eval_d500_reg_l05.txt/.json
