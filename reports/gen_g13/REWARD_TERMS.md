# 보상 항 목록 (G12 계열 = 현재 실물 정책 G12-9500 의 학습 환경, G14a 는 recovery_event 만 제거)

출처: logs/rsl_rl/biped12_stage4/2026-09-30_15-26-19_g14a/params/env.yaml. Isaac Lab 에서 실제 보상 = 가중치 × 값 × dt(0.02).
'안정 시만' = 회복 게이트가 닫힌 동안만 적용(state_gated). 균형을 잃으면(기울기>5°·|ω|>0.35 등) 꺼진다.

에피소드당 기여 = 최근 50 반복 평균. G12c = 보너스 있는 대조군 학습 끝(11499), G14a = 보너스 제거 학습 진행 중(it 9751).

| 항 | 가중치 | 함수 | 설명 | 기여 G12c | 기여 G14a |
|---|---|---|---|---|---|
| track_lin_vel_xy_exp | 2.0 | `track_lin_vel_xy_yaw_frame_exp` | 속도 명령(0) 추종 = 제자리 유지 | +0.694 | +0.714 |
| track_ang_vel_z_exp | 2.0 | `track_ang_vel_z_world_exp` | 요 회전 0 추종 | +0.398 | +0.412 |
| flat_orientation_l2 | -8.0 | `flat_orientation_l2` | 몸통 기울기 벌점 | -0.024 | -0.022 |
| base_height_l2 | -2.0 | `base_height_l2` | 골반 높이 목표 이탈 | -0.002 | -0.002 |
| lin_vel_z_l2 | -0.2 | `lin_vel_z_l2_capped` | 상하 속도 | -0.002 | -0.002 |
| ang_vel_xy_l2 | -0.1 | `ang_vel_xy_l2_capped` | 롤·피치 각속도 (흔들림에 직접 걸리는 유일한 항, 가중 작음) | -0.025 | -0.025 |
| feet_slide | -0.1 | `state_gated(feet_slide)` | 발 미끄러짐 (안정 시만) | -0.002 | -0.002 |
| feet_contact_forces | -0.002 | `contact_forces_capped` | 과도한 접촉력 | -0.006 | -0.005 |
| joint_deviation_hip | -0.5 | `joint_deviation_l1` | 힙 기본자세 이탈 | -0.035 | -0.034 |
| joint_deviation_ankle_roll | -0.2 | `joint_deviation_l1` | 발목 롤 이탈 | -0.008 | -0.008 |
| joint_vel_l2 | -0.0005 | `joint_vel_l2_capped` | 관절 속도 | -0.002 | -0.002 |
| dof_acc_l2 | -2.5e-07 | `joint_acc_l2_capped` | 관절 가속도 | -0.002 | -0.002 |
| dof_torques_l2 | -0.0001 | `joint_torques_l2` | 토크 | -0.021 | -0.021 |
| action_rate_l2 | -0.2 | `state_gated(action_rate_l2_capped)` | 명령 변화량 (안정 시만) | -0.357 | -0.343 |
| dof_pos_limits | -5.0 | `joint_pos_limits` | 관절 한계 근접 | -0.001 | -0.001 |
| undesired_contacts | -1.0 | `undesired_contacts` | 발 외 접촉 | +0.000 | -0.000 |
| termination_penalty | -10.0 | `is_terminated` | 넘어짐 | -0.009 | -0.009 |
| feet_gap | -5.0 | `state_gated(feet_too_close)` | 발 간격 좁음 (안정 시만) | -0.000 | -0.000 |
| foot_yaw_slip | -0.5 | `state_gated(foot_yaw_slip)` | 발 요 미끄러짐 (안정 시만) | -0.107 | -0.113 |
| march_anchor | -1.0 | `state_gated(march_position_hold)` | 위치 유지 (안정 시만) | -0.004 | -0.004 |
| knee_stance_flexion | 4.0 | `state_gated(knee_stance_flexion)` | 무릎 크라우치 밴드 (안정 시만) | +0.675 | +0.730 |
| feet_impact | -0.01 | `feet_impact_force` | 착지 충격 | -0.004 | -0.004 |
| torque_saturation | -2.0 | `torque_saturation` | 토크 포화 | -0.001 | -0.001 |
| feet_contact_gated | 2.0 | `state_gated(feet_contact_gated)` | 양발 접지 (안정 시만) | +0.320 | +0.348 |
| survive_after_release | 1.0 | `survive_after_release` | 지지 제거 뒤 생존 | +0.400 | +0.409 |
| hang_quiet | -0.002 | `hang_quiet` | 매달림 중 정숙 | +0.000 | +0.000 |
| pose_symmetry | -3.0 | `state_gated(pose_symmetry)` | 좌우 대칭 (안정 시만) | -0.014 | -0.014 |
| stance_shape | -5.0 | `state_gated(stance_shape)` | 스탠스 모양 (안정 시만) | -0.003 | -0.002 |
| torso_pitch_band | -25.0 | `state_gated(torso_pitch_band)` | 몸통 피치 밴드 (안정 시만) | -0.025 | -0.024 |
| hip_r_deviation | -3.0 | `joint_deviation_l1` | 힙 요 이탈 | -0.114 | -0.110 |
| settle_quiet_bonus | 2.0 | `state_gated(settle_quiet_bonus)` | 정착 후 정숙 보너스 (안정 시만) | +0.001 | +0.000 |
| leg_pose_deviation | -1.5 | `state_gated(joint_deviation_l1)` | 다리 자세 이탈 (안정 시만) | -0.075 | -0.083 |
| action_rate_window | -0.06 | `open_gated(action_rate_l2_capped)` | 명령 변화량 (균형 상실 중, 약하게) | -0.141 | -0.125 |
| recovery_quality | 5.0 | `recovery_quality` | 균형 상실 중 안정에 가까울수록 보너스 | +0.285 | +0.289 |
| recovery_event | 30.0 (G14 에서 제거) | `recovery_event` | 균형 상실→회복 1회당 보너스 | +0.142 | — |

총 보상(에피소드 평균): G12c 38.4 · G14a 38.6
에피소드 길이(스텝): G12c 399 · G14a 407
