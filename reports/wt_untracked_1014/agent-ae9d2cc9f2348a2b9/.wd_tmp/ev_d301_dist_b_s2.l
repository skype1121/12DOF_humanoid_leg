[3gH    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H    H   [INFO] Using python from: /home/ryu/IsaacLab/_isaac_sim/python.sh
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: /home/ryu/IsaacLab/apps/isaaclab.python.headless.kit
[INFO]: Parsing configuration from: rl_walking.hang_stand_env_cfg:Biped12HangStandW3S4dDEnvCfg
[INFO]: Parsing configuration from: rl_walking.agents:Biped12StepB0PPORunnerCfg

[36m======================================================================================[0m
[36m[1m[INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_2026-10-07_06-55-54.log[0m
[36m======================================================================================[0m

[33m06:55:54 [simulation_context.py] WARNING: The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False. If you are experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to improve the accuracy of velocity updates.[0m
[INFO]: Base environment:
	Environment device    : cuda:0
	Environment seed      : 2
	Physics step-size     : 0.005
	Rendering step-size   : 0.02
	Environment step-size : 0.02
[INFO]: Time taken for scene creation : 1.213114 seconds
[INFO]: Scene manager:  <class InteractiveScene>
	Number of environments: 320
	Environment spacing   : 2.5
	Source prim name      : /World/envs/env_0
	Global prim paths     : ['/World/ground']
	Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
[INFO]: Time taken for simulation start : 1.174281 seconds
[33m06:55:56 [velocity_command.py] WARNING: The velocity command has the 'ranges.heading' attribute set to '(-3.141592653589793, 3.141592653589793)' but the heading command is not active. Consider setting the flag for the heading command to True.[0m
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
[eval_w3] --dist 그룹: G0=push:b:6:0:0.3(WALK) | G1=push:b:6:8:0.3(WALK) | G2=push:b:6:12:0.3(WALK) | G3=push:b:6:15:0.3(WALK) | G4=push:b:6:18:0.3(WALK)
[eval_w3] 1600 틱 완료 · 131 s
# W3 Stage 0 채점표 · d301_dist_b_s2
- 태스크 Biped12-HangStand-W3S4dD-v0 · 정책 /home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ae9d2cc9f2348a2b9/logs/rsl_rl/biped12_stepb0/2026-10-07_03-20-15_w3S4dD_from_s4d/model_16600.pt
- env 320 · seed 2 · 지연 2틱 · LPF β 0.27 · 워밍업 2.0s + 채점 30s · 외란 --dist push:b:6:0:0.3|push:b:6:8:0.3|push:b:6:12:0.3|push:b:6:15:0.3|push:b:6:18:0.3 · 학습 DR 유지
- 명령 (vx,vy,wz) = (+0.06,+0.00,+0.00) · gait_enable 1 · STEP_ASSIST_ALPHA 0.0 · W3_REF_SCALE 미설정 · TURN_ALPHA 0 · 위상 출처 env._gu · 보조 훅 없음

| 지표 | 값 |
|---|---|
| 이륙 성공률(예정 스윙 반주기 중 이륙) | 99.9 % (18400/18413) |
| 착지 성공률(이륙 대비) | 98.5 % (18419/18701) |
| 의미 있는 스윙(착지 대비, ≥60 ms·≥1.5 cm) | 100.0 % (18410) |
| 하중 수용률(착지 대비, 0.3 s 안 Fz>0.5 mg 0.1 s) | 100.0 % (18414) |
| 유효 스텝 | 18410 (왼 9062 / 오른 9348) · 2.16 /env/s |
| 교대율(연속 유효 스텝 좌우 교대) | 100.0 % (18116/18116) |
| 유효 cadence(보행 주기) | 1.082 Hz |
| 수평 표류(명령 대비 잔차 속도) | 0.025 m/s · 총 이동 2.10 m · 실제 앞 +0.066 / 옆 +0.002 m/s |
| 요 표류(명령 대비 |Δ요|) | 25.8° · 평균 요 속도 +0.02°/s |
| 낙상률 | 12.2 % (39/320) — 워밍업(시작 과도) 23 · 채점 구간 16 · 시각 0.5-1s:3 1-1.5s:14 1.5-2s:6 2-3s:3 5-1000000000.0s:13 |
| 10 s 창별 유효 cadence | 1.05 → 1.10 → 1.10 Hz · 붕괴 없음 |
| 스텝 높이 cm (10/50/90 %) | 5.5/6.4/7.1 |
| 옆 변위 cm (착지, 10/50/90 %) | -1.6/+0.3/+2.4 · 명령 방향 부호 일치 nan % |
| 착지 발 간격 cm (10/50/90 %) | 33.0/34.5/36.1 · 23.8 미만 0.00 % · 교차 0.00 % · 좌/우 유효 스텝 비 0.97 · 롤 평균 +0.26° |
| 보폭 cm (수평 변위 10/50/90 %) | 4.8/6.4/8.1 · 앞축 +4.5/+6.2/+7.9 |
| 이륙 위상(예정 스윙 시작 대비 주기, 10/50/90 %) | +0.04/+0.06/+0.08 |
| 피치 RMS(정지 떨림, 참고) | 1.16° · 피치각속도 RMS 0.243 rad/s |
| 정책 행동 RMS | 0.710 (×scale 0.25 = 10.18°) |
| 참조 잔차 RMS(env._w3_ref_last) | 0.00° |

| 발 | 이륙 | 예정창 | 예정창 이륙 | 착지 | 의미 스윙 | 하중 수용 | 유효 |
|---|---|---|---|---|---|---|---|
| 왼 | 9348 | 9064 | 9058 | 9067 | 9062 | 9066 | 9062 |
| 오른 | 9353 | 9349 | 9342 | 9352 | 9348 | 9348 | 9348 |

## 외란 회복 체인(1006 우선순위 4 — 외란 → 회복 스텝 → 운동량·각속도 억제 → 접촉 → 수평 → 정상 STAND/WALK 복귀)
- 시각 = 외란 시작 기준 s · 복귀 한도 rec_T 6.0 s · 분모 = 외란 시작 시 생존 env(외란 전 낙상은 제외·따로 표기) · 낙상 = 미도달
- WALK 정상 = 1.8 s 창이 외란 끝 뒤에 있고 외란 전 2.7 s 같은 env 기준: 창 평균 속도 추종 오차 ≤ 전 오차 + 0.03 · 유효 스텝률 ≥ 전 × 0.6 · 창 안 교대 비율 ≥ 0.75·유효 ≥ 2 · 평균 피치·롤 전 ± 2.0° · ω_xy RMS ≤ 1.5×전 + 0.05 · 높이 ≥ 전 − 2 cm, 1.0 s 유지. 복귀 시각 = 유지 시작 창의 시작(하한 = 외란 길이). 억제·수평 = 0.5 s 창(전 기준 ω RMS·속도 ±0.05 / 피치·롤 ± 2.0°), 접촉 = 외란 끝 뒤 교대 유효 스텝 2 연속
- STAND 정상(정지 그룹, eval_ground_recovery 와 같은 절대 기준) = |피치|·|롤| < 3° · |ω_xy| < 0.2 · 양발 · 명목 관절 평균 편차 < 5° · 높이 ≥ 외란 전 − 2 cm, 0.5 s 지속(외란 끝 + 0.3 s·정지 요청 뒤). 억제 = |ω_xy| < 0.2·|v| < 0.1 0.2 s, 접촉 = 양발 0.5 s, 수평 = |피치|·|롤| < 3° 0.5 s
- 대조: '외란 전 정상' = 외란 시작 순간 WALK 판정이 이미 참인 비율(판정 기준 자체의 잡음 확인). 방향 검증 Δv = 외란 끝−시작 골반 속도의 힘 방향 성분(걷기 흔들림 포함 — 같은 방향·같은 길이 힘 0 대조군 값을 빼서 읽는다; 기대 ≈ 힘·시간/17.6 kg). 외란 종료 뒤 base 합성 외력 최대 0.00 N(0 이어야 함 — 힘 누적 감시)

| 그룹 | 시나리오 | n(외란 전 낙상) | 외란 전 정상 | Δv 힘 방향 m/s | 낙상 | 회복 스텝(첫 유효) | 억제 | 접촉 | 수평 | **정상 복귀** ≤rec_T | 복귀 s 10/50/90 | ≤1/2/3/5 s | 추가 스텝 중앙/90 | 비정상 체류 | 확정 뒤 유지 | 정점 |Δ피치|/|Δ롤|°·ω·Δh cm(중앙) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 `push:b:6:0:0.3` | WALK(유지) | 59(5) | 59/59 | -0.003 | 0/59 | 59/59 · 0.23s | 59/59 · 1.04s | 59/59 · 0.97s | 59/59 · 0.94s | **59/59** | 0.30/0.31/0.32 | 59/59/59/59 | -0.2/+0.8 | 0.1 % | 99.9 % | 2.4/11.8·1.91·-1.9 |
| G1 `push:b:6:8:0.3` | WALK(유지) | 54(10) | 54/54 | +0.082 | 0/54 | 54/54 · 0.20s | 54/54 · 1.12s | 54/54 · 0.99s | 54/54 · 0.97s | **54/54** | 0.30/0.31/0.32 | 53/54/54/54 | -0.2/+0.8 | 0.2 % | 99.8 % | 4.3/11.7·1.87·-2.0 |
| G2 `push:b:6:12:0.3` | WALK(유지) | 63(1) | 63/63 | +0.151 | 0/63 | 63/63 · 0.20s | 63/63 · 1.35s | 63/63 · 0.97s | 63/63 · 1.08s | **63/63** | 0.30/0.31/0.62 | 60/62/63/63 | -0.2/+0.8 | 0.6 % | 99.8 % | 5.3/12.0·1.93·-2.0 |
| G3 `push:b:6:15:0.3` | WALK(유지) | 59(5) | 59/59 | +0.203 | 3/59 | 59/59 · 0.29s | 56/59 · 1.31s | 59/59 · 0.91s | 57/59 · 1.23s | **56/59** | 0.30/0.31/0.99 | 50/56/56/56 | -0.2/+0.8 | 0.9 % | 99.7 % | 7.3/12.1·1.99·-2.0 |
| G4 `push:b:6:18:0.3` | WALK(유지) | 59(5) | 59/59 | +0.264 | 10/59 | 59/59 · 0.24s | 49/59 · 1.67s | 56/59 · 0.95s | 52/59 · 1.37s | **49/59** | 0.30/0.63/1.40 | 36/47/48/49 | -0.2/+0.8 | 1.8 % | 99.8 % | 9.0/12.2·2.13·-2.2 |

주: 외란 그룹은 env 를 round-robin 으로 나눈다(같은 명령·같은 seed·같은 시각, 힘 0 그룹 = 대조군). 위 Stage 0 표·판정은 전 그룹 합산(외란 포함)이라 외란 실행에서는 참고만.
정점 = 외란 시작~rec_T 동안 외란 전 평균 대비 최대 이탈(높이는 최저). 추가 스텝(WALK) = rec_T 안 유효 스텝 − 2·f_클록(외란 시작 env._gf, 없으면 외란 전 실측 스텝률) × rec_T — 대조군(힘 0) 값과 비교해 읽는다.
hang_support 롤 토크 램프: 끔(--ramp_off)

### WALK 판정 하위 기준별 실패 틱 비율(외란 끝 + 창 이후, 생존 틱 — 대조군 값 = 판정 잡음)
| 그룹 | 속도 추종 | 스텝률·교대 | 피치 | 롤 | ω RMS | 높이 |
|---|---|---|---|---|---|---|
| G0 `push:b:6:0:0.3` | 0.1 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| G1 `push:b:6:8:0.3` | 0.2 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| G2 `push:b:6:12:0.3` | 0.6 % | 0.0 % | 0.1 % | 0.0 % | 0.0 % | 0.0 % |
| G3 `push:b:6:15:0.3` | 0.9 % | 0.0 % | 0.1 % | 0.0 % | 0.0 % | 0.0 % |
| G4 `push:b:6:18:0.3` | 1.7 % | 0.0 % | 0.7 % | 0.0 % | 0.0 % | 0.0 % |

### 외란 시작 위상별 낙상 / 복귀(φ = env._gu, 0–0.5 왼발 스윙·0.5–1 오른발 스윙 예정 · 시각 오프셋 U[0,0.91) s)
| 그룹 | φ 0.00–0.25 낙상/복귀/n | φ 0.25–0.50 낙상/복귀/n | φ 0.50–0.75 낙상/복귀/n | φ 0.75–1.00 낙상/복귀/n |
|---|---|---|---|---|
| G0 `push:b:6:0:0.3` | 0/19/19 | 0/14/14 | 0/15/15 | 0/11/11 |
| G1 `push:b:6:8:0.3` | 0/9/9 | 0/18/18 | 0/15/15 | 0/12/12 |
| G2 `push:b:6:12:0.3` | 0/13/13 | 0/18/18 | 0/18/18 | 0/14/14 |
| G3 `push:b:6:15:0.3` | 1/14/15 | 0/11/11 | 0/22/22 | 2/9/11 |
| G4 `push:b:6:18:0.3` | 0/16/16 | 4/8/12 | 1/14/15 | 5/11/16 |

## Stage 0 사전 기준 판정(STEP): FAIL
- PASS · 교대 > 90 % · 100.0 %
- PASS · 이륙 > 90 % · 99.9 %
- PASS · 착지 > 90 % · 98.5 %
- PASS · 하중 ≥ 85 % · 100.0 %
- FAIL · 채점 구간 낙상 < 2 % (시작 과도 제외 — 1006 판정 정의) · 16/297
- PASS · 표류 ≤ 0.03 m/s · 0.025 m/s
- PASS · 장시간 붕괴 없음 · 창 1.05/1.10/1.10 Hz

주: 접촉 = 발 링크 |F| > 5 N. 유효 cadence = 유효 스텝/(생존 env·s)/2. 표류 = 시작 요 프레임 변위/채점 시간 − 명령(vx,vy). 낙상 env 는 낙상 전까지의 사건만 집계. 이 판정은 시뮬 채점이며 실물 성공을 뜻하지 않는다.
[eval_w3] 저장: log_picture/w3eval_d301_dist_b_s2.txt/.json
