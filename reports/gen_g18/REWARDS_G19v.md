# 보상 항 일람 — G19v (= G20 계열의 바탕) · 1002

출처: `logs/rsl_rl/biped12_stage4/*_g19v/params/env.yaml` (실제 학습에 쓰인 값). 가중치는 Isaac Lab 규약상 매 스텝 `weight × 값 × dt(0.02)`.
**게이트**: 회복 게이트(`mdp_hang.recovery_gate_step`) — 외력 작용 중·기울기 > 6.5°·|ω| > 1.5·|v| > 0.5·(한 발 뜸 & 기울기 > 3°) 이면 **열림**,
정상 상태가 0.3 s 이어지면 닫힘, 4 s 넘게 열려 있으면 안정될 때까지 강제 닫힘.

## A. 항상 켜짐
| 항 | 함수 | 가중치 | 의미 |
|---|---|---|---|
| track_lin_vel_xy_exp | track_lin_vel_xy_yaw_frame_exp | +2.0 | 명령 속도(서기 = 0) 추종 |
| track_ang_vel_z_exp | track_ang_vel_z_world_exp | +2.0 | 요 각속도 추종(0) |
| flat_orientation_l2 | flat_orientation_l2 | −8.0 | 몸통 기울기² |
| base_height_l2 | base_height_l2 | −2.0 | 골반 높이 편차² |
| lin_vel_z_l2 | lin_vel_z_l2_capped | −0.2 | 수직 속도² |
| ang_vel_xy_l2 | ang_vel_xy_l2_capped | −0.1 | 몸통 롤·피치 각속도² |
| feet_contact_forces | contact_forces_capped | −0.002 | 과도한 발 접촉력 |
| joint_deviation_hip | joint_deviation_l1 | −0.5 | 힙 벌림·회전 기본자세 편차 |
| joint_deviation_ankle_roll | joint_deviation_l1 | −0.2 | 발목 롤 편차 |
| hip_r_deviation | joint_deviation_l1 | −3.0 | 힙 회전 편차 |
| joint_vel_l2 | joint_vel_l2_capped | −0.0005 | 관절 속도² |
| dof_acc_l2 | joint_acc_l2_capped | −2.5e-7 | 관절 가속도² |
| dof_torques_l2 | joint_torques_l2 | −0.0001 | 토크² |
| dof_pos_limits | joint_pos_limits | −5.0 | 관절 한계 근접 |
| torque_saturation | torque_saturation | −2.0 | 토크 포화 |
| feet_impact | feet_impact_force | −0.01 | 착지 충격(문턱 180 N — 12 kg 시절 값, 17.6 kg 와 불일치: OSS 분석 2) |
| undesired_contacts | undesired_contacts | −1.0 | 발 외 접촉 |
| termination_penalty | is_terminated | −10.0 | 낙상 종료 |
| survive_after_release | survive_after_release | +1.0 | 지지 해제 뒤 생존 |
| hang_quiet | hang_quiet | −0.002 | 매달림 중 정숙 |

## B. 안정 상태에서만 (`state_gated` — 게이트 닫힘일 때만 적용)
| 항 | 내부 함수 | 가중치 | 의미 |
|---|---|---|---|
| torso_pitch_band | torso_pitch_band | **−25.0** | 몸통 피치 허용 대역 이탈 |
| stance_shape | stance_shape | −5.0 | 발 앞뒤 엇갈림·자세 모양 |
| feet_gap | feet_too_close | −5.0 | 발 간격 너무 좁음 |
| pose_symmetry | pose_symmetry | −3.0 | 좌우 대칭 |
| leg_pose_deviation | joint_deviation_l1 | −1.5 | 다리 기본자세 편차 |
| march_anchor | march_position_hold | −1.0 | 제자리 유지 |
| foot_yaw_slip | foot_yaw_slip | −0.5 | 발 요 미끄럼 |
| feet_slide | feet_slide | −0.1 | 발 미끄럼 |
| action_rate_l2 | action_rate_l2_capped | −0.2 | 명령 변화²(정지 상태 떨림 억제) |
| knee_stance_flexion | knee_stance_flexion | +4.0 | 서기 무릎 굽힘 유지 |
| feet_contact_gated | feet_contact_gated | +2.0 | 양발 접촉 |
| settle_quiet_bonus | settle_quiet_bonus | +2.0 | 정착 정숙 보너스 — **0930 확인: 골반고 문턱 0.60 vs 크라우치 0.597 로 사실상 미발동** |

## C. 회복 중에만 (게이트 열림)
| 항 | 함수 | 가중치 | 의미 |
|---|---|---|---|
| recovery_quality | recovery_quality | +5.0 | 열린 동안 |ω|·|v|·q̇ 작을수록 0–1 (양발 접촉 & 기울기 < 6.5° 조건 — **스윙 중 0**, OSS 분석 2 지적) |
| recovery_event | recovery_event | +30.0 | 균형 상실 → 실제 회복(1회성) |
| action_rate_window | open_gated(action_rate_l2_capped) | −0.06 | 회복 중 명령 변화(약하게) |

## G20s 에 추가된 항 (G19v 대비 단일 변경)
| quiet_ang_vel | state_gated(ang_vel_xy_l2_capped) | −1.0 | 안정 상태에서만 몸통 롤·피치 각속도² (느린 흔들림 감쇠 목적) |

## 1002 관찰 메모
- 회복 스텝은 C 의 recovery_event(+30)·quality 로 유도되나, quality 가 양발 조건이라 스윙 자체는 구조적 손해(G18q 로 완화 시도 → 미지지).
- 정지 정숙은 B 의 action_rate(−0.2)와 A 의 ang_vel(−0.1)뿐, settle 보너스는 죽어 있음 → 느린 흔들림을 직접 벌하는 항이 약함(G20s 가 겨냥).
- 3 Hz 실물 튐은 보상이 아니라 정책 되먹임 이득(자이로·관절속도) 문제로 귀속 — 관측 필터(G18f·G19v)로 다룸.
