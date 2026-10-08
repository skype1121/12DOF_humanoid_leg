# 차이 목록 — 젯슨 현재 판 ↔ HEAD (head_1008)

> `python3 deploy/make_head_manifest.py` 가 만든 파일이다(손으로 고치지 말 것). 젯슨 현재 md5 는
> `deploy/cand/head_1008/jetson_inventory_before_md5.txt`(주 작업자가 젯슨에서 읽기 전용으로 뽑은 목록 173 파일)에서 왔다.

```
package: head_1008
source_commit: 385686de411ecb153fb790628d9e3178ef272d06 2026-10-08T16:44:22+09:00
source_uncommitted: none
manifest: deploy/head_1008.md5 md5 b6d9793c095182ea3b4da1d469355844
files: 127 (send 71 incl. this file and the manifest, verify 56)
jetson_inventory: deploy/cand/head_1008/jetson_inventory_before_md5.txt md5 35a129c9f16a082e1cb6d3ba9cb13855
note: 파일 md5(매니페스트)가 정본이다. source_commit = 실행 코드(콘솔·브리지·stage8·러너·도구·시험)를 마지막으로 바꾼 커밋,
      source_uncommitted = 생성 시점에 그 커밋과 달랐던 실행 코드 파일(none 이어야 커밋과 일치).
```

## 요약

| 구분 | 파일 수 | 크기 합 |
|---|---|---|
| 전송 — 변경(젯슨에 구판이 있음) | 14 | 1115 KB |
| 전송 — 신규(젯슨에 없음) | 55 (+ 매니페스트·VERSION 2) | 8019 KB |
| 확인만(젯슨과 같음 — 복사하지 않음) | 56 | 13343 KB |
| 삭제 | 0 | — |

묶음: session = 콘솔·브리지·stage8 의 import 사슬 · tool = 운영 도구·selftest · test = 오프라인 단위 시험 · sync = 저장소 사본 맞춤 · policy = 정책 파일 · ref = selftest 기준 원문 · record = 기록·문서

## 1. 전송 — 변경 (젯슨의 구판을 백업한 뒤 덮어쓴다)

| 경로 | 묶음 | 젯슨 md5 | 새 md5 | 젯슨 판의 출처 | 쓰는 진입점 |
|---|---|---|---|---|---|
| `jetson/scripts/analyze_cop_transfer.py` | sync | `4182c582` | `b9ed7cac` | eb15a99a 2026-09-11 | - |
| `jetson/scripts/console_demo.py` | session | `6f305da7` | `8bcd50e6` | 98ecfc9b 2026-09-22 | operator_console.py |
| `jetson/scripts/console_state.py` | session | `446a7c2b` | `d4bf8d03` | 432d1176 2026-10-03 | operator_console.py, robotctl.py |
| `jetson/scripts/console_ui.html` | session | `9f6a963d` | `9c902322` | 432d1176 2026-10-03 | - |
| `jetson/scripts/imu_mount_calib.py` | sync | `b8aa00e9` | `7c3bd93c` | 29b984d2 2026-07-27 | - |
| `jetson/scripts/operator_console.py` | session | `90b7ef28` | `a976fb13` | 8af2f6e5 + 1007 krh 패치(deploy/cand/c5krh_1007) | operator_console.py |
| `jetson/scripts/pose_hold.py` | sync | `83e4f599` | `16f57ffd` | 9c4750d3 2026-07-27 | - |
| `jetson/scripts/rl_bridge_node.py` | session | `15dcabf2` | `8938e790` | 8af2f6e5 + 1007 krh 패치(deploy/cand/c5krh_1007) | posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py, zero_kinematic_check.py |
| `jetson/scripts/robotctl.py` | tool | `f761802c` | `61327e6b` | 83faa61b 2026-09-19 | robotctl.py |
| `jetson/scripts/sensors_up.sh` | sync | `73fec1f3` | `3b372a2b` | 54425ab0 2026-09-13 | - |
| `jetson/scripts/stage8_12axis_mit_control_node.py` | session | `2509bd99` | `18e1ec3e` | c93b45d1 2026-09-30 | stage8_12axis_mit_control_node.py |
| `rl_walking/__init__.py` | session | `51ed3c5e` | `2413781a` | 6761a690 2026-09-29 | kew3_runner.py, operator_console.py, posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py, w3_cmd.py |
| `rl_walking/deploy/c5_switch.py` | session | `11cf9641` | `eaf66687` | 332dcf38 2026-10-05 | c5_switch.py, kew3_runner.py, posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py, zero_kinematic_check.py |
| `rl_walking/deploy/policy_runner_stage4.py` | session | `21b2416e` | `bdbe69eb` | b9cd2816 2026-10-03 | c5_switch.py, kew3_runner.py, policy_runner_stage4.py, posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py |

## 2. 전송 — 신규 (젯슨에 없는 파일, 롤백 때 지운다)

| 경로 | 묶음 | 새 md5 | 크기 | 비고 |
|---|---|---|---|---|
| `deploy/cand/head_1008/OPERATOR_COMMANDS.md` | record | `3dfbf672` | 31707 | - |
| `deploy/cand/head_1008/ref_bridge/rl_bridge_node.68493c11.ref` | ref | `2ddfc1e7` | 234333 | 기준 커밋 68493c11 의 jetson/scripts/rl_bridge_node.py 원문 |
| `deploy/cand/head_1008/ref_bridge/rl_bridge_node.95946e41.ref` | ref | `621cb77f` | 255072 | 기준 커밋 95946e41 의 jetson/scripts/rl_bridge_node.py 원문 |
| `deploy/cand/head_1008/ref_bridge/rl_bridge_node.b79370ca.ref` | ref | `9d78084d` | 318778 | 기준 커밋 b79370ca 의 jetson/scripts/rl_bridge_node.py 원문 |
| `deploy/cand/head_1008/selftest_w3_node_refs.py` | ref | `40edb00b` | 2883 | - |
| `jetson/scripts/analyze_a1_blackbox.py` | sync | `4d75432e` | 5032 | - |
| `jetson/scripts/analyze_correction_chain.py` | sync | `9ba80eaa` | 22906 | - |
| `jetson/scripts/analyze_hang2stand_trial.py` | sync | `6ff852b8` | 32693 | - |
| `jetson/scripts/analyze_roll_response.py` | sync | `9a8eeb14` | 9300 | - |
| `jetson/scripts/analyze_takeover.py` | sync | `3553ca0e` | 7114 | - |
| `jetson/scripts/can_channel.py` | sync | `2e7816b1` | 4824 | - |
| `jetson/scripts/collect_trial_logs.sh` | sync | `0b5c54cc` | 3362 | - |
| `jetson/scripts/ground_stand.py` | sync | `32b32fbd` | 23137 | - |
| `jetson/scripts/hang_leg_swing.py` | sync | `7ab49173` | 14018 | - |
| `jetson/scripts/latency_check.py` | tool | `35a134e9` | 25224 | latency_check.py |
| `jetson/scripts/level_pose_retarget.py` | sync | `e71d340b` | 3805 | - |
| `jetson/scripts/ping_all.py` | sync | `18c4b94a` | 5594 | ping_all.py |
| `jetson/scripts/play_probe.py` | sync | `6b06ae5e` | 46233 | - |
| `jetson/scripts/plot_trial_recon.py` | sync | `78a9ca95` | 25298 | - |
| `jetson/scripts/selftest_w3_bridge_node.py` | tool | `6a27719b` | 97562 | selftest_w3_bridge_node.py |
| `jetson/scripts/stack_down.sh` | sync | `7f8f7925` | 1765 | - |
| `jetson/scripts/stage8_cmd.py` | sync | `344aae83` | 4737 | stage8_cmd.py |
| `jetson/scripts/takeover_capture.sh` | sync | `4653296c` | 1887 | - |
| `jetson/scripts/w3_cmd.py` | session | `e06de3d4` | 9569 | operator_console.py, w3_cmd.py |
| `jetson/scripts/wiggle_watch.py` | sync | `cfccb0cc` | 2996 | - |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/SOURCE_CHECKPOINT` | policy | `bdd15e01` | 15 | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/policy.onnx` | policy | `2b718875` | 1321751 | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/policy.pt` | policy | `7bbf24f5` | 1332468 | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/params/env.yaml` | policy | `dcc9ceb5` | 57004 | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/SOURCE_CHECKPOINT` | policy | `1ad3809e` | 15 | stand_w3s3d_15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.onnx` | policy | `d908cd92` | 1321751 | stand_w3s3d_15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.pt` | policy | `10e3ee40` | 1332468 | stand_w3s3d_15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml` | policy | `41d1e62a` | 60296 | stand_w3s3d_15800 |
| `reports/night_1007/jetson/parity_ref.npz` | parity | `425e48cf` | 1128300 | 러너 동치 확인(로컬 기준 ↔ 젯슨 onnxruntime) |
| `reports/night_1007/jetson_parity.py` | parity | `029cb8b2` | 10754 | 러너 동치 확인(로컬 기준 ↔ 젯슨 onnxruntime) |
| `rl_walking/deploy/hturn2.py` | session | `d0b283ab` | 7418 | c5_switch.py, kew3_runner.py, policy_runner_stage4.py, posture_probe.py |
| `rl_walking/deploy/kew3_runner.py` | session | `a6f500e4` | 50539 | kew3_runner.py, posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py |
| `rl_walking/deploy/leg_odometry.py` | session | `5f5c8132` | 12224 | c5_switch.py, kew3_runner.py, posture_probe.py, rl_bridge_node.py |
| `rl_walking/deploy/selftest_c5_imu_f32.py` | sync | `61536386` | 7309 | - |
| `rl_walking/deploy/selftest_hturn2_equiv.py` | sync | `902d9f8d` | 14962 | - |
| `rl_walking/deploy/selftest_odic_equiv.py` | sync | `19e8fc83` | 24418 | - |
| `rl_walking/deploy/selftest_w3_contingency.py` | sync | `1741f3ee` | 18349 | - |
| `rl_walking/deploy/selftest_w3_equiv.py` | sync | `6f1a5188` | 25398 | - |
| `rl_walking/deploy/selftest_w3_rollout_parity.py` | sync | `bae0c5ee` | 19395 | - |
| `rl_walking/deploy/selftest_w3s3d_onnx.py` | sync | `85cc9498` | 2188 | - |
| `rl_walking/deploy/w3_contingency.py` | session | `3105a219` | 36676 | posture_probe.py, rl_bridge_node.py, selftest_w3_bridge_node.py, w3_contingency.py |
| `rl_walking/deploy/w3_deploy.py` | session | `aba079dd` | 46499 | c5_switch.py, kew3_runner.py, operator_console.py, policy_runner_stage4.py |
| `tests/test_console_walk.py` | test | `7339e13d` | 33298 | - |
| `tests/test_console_walk_bridge.py` | test | `3282c2f1` | 37053 | - |
| `tests/test_latency_check.py` | test | `db1ba370` | 7263 | - |
| `tests/test_operator_console.py` | test | `513c4789` | 54578 | - |
| `tests/test_robotctl.py` | test | `ca9b74be` | 14372 | - |
| `tests/test_stage8_can_send_timeout.py` | test | `955b8a97` | 7549 | - |
| `tests/test_stage8_latency_options.py` | test | `70d08af6` | 12404 | - |
| `tests/test_startup_simplify_bridge.py` | test | `491c7085` | 24527 | - |

## 3. 확인만 (젯슨 md5 = HEAD md5 — 배포 때 다시 읽어 같지 않으면 중단)

| 경로 | 묶음 | md5 |
|---|---|---|
| `jetson/scripts/__init__.py` | session | `85f77480` |
| `jetson/scripts/analyze_fall.py` | tool | `a9efae45` |
| `jetson/scripts/boot_pose_check.py` | tool | `cf24508e` |
| `jetson/scripts/can_up.sh` | sync | `42f27f3a` |
| `jetson/scripts/cop_from_pads.py` | sync | `c16ae314` |
| `jetson/scripts/foot_force_adapter_node.py` | sync | `9b13d698` |
| `jetson/scripts/hang_stand.py` | sync | `2692298a` |
| `jetson/scripts/imu_adapter_node.py` | sync | `7feaf579` |
| `jetson/scripts/obs_dryrun.py` | sync | `abc2c009` |
| `jetson/scripts/pitch_sweep.py` | sync | `79d0dac5` |
| `jetson/scripts/posture_probe.py` | tool | `35344671` |
| `jetson/scripts/propose_ankle_r_direction_fix.py` | sync | `e899f245` |
| `jetson/scripts/release_all.py` | sync | `e06b80d1` |
| `jetson/scripts/set_pitch_comp.py` | sync | `1a9fe738` |
| `jetson/scripts/sign_check.py` | sync | `483b4e54` |
| `jetson/scripts/static_march.py` | sync | `194e79b6` |
| `jetson/scripts/stiffness_log.py` | sync | `b65f0a4e` |
| `jetson/scripts/zero_anchor.py` | tool | `00340915` |
| `jetson/scripts/zero_kinematic_check.py` | tool | `bbec70cd` |
| `jetson/scripts/zero_torque_read.py` | sync | `fd7eec04` |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.onnx` | policy | `c577a65a` |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.pt` | policy | `fac90eb5` |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/params/env.yaml` | policy | `a2f1f885` |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.onnx` | policy | `a090aa87` |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.pt` | policy | `51f3e2b7` |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/params/env.yaml` | policy | `ab1c12eb` |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.onnx` | policy | `6c5f8a7e` |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.pt` | policy | `dc9cd586` |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/params/env.yaml` | policy | `0c071751` |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.onnx` | policy | `43f11310` |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.pt` | policy | `2464f5cd` |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/params/env.yaml` | policy | `ebda410e` |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/SOURCE_CHECKPOINT` | policy | `e24b90e3` |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.onnx` | policy | `57547697` |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.pt` | policy | `896ca613` |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/params/env.yaml` | policy | `7ff27617` |
| `protocols/__init__.py` | session | `3820717c` |
| `protocols/ak_mit_command.py` | session | `52543d3d` |
| `protocols/ak_mit_decoder.py` | session | `23d9de66` |
| `rl_walking/deploy/__init__.py` | session | `01666387` |
| `rl_walking/deploy/bench_stage8_mock.py` | sync | `fe1f7e82` |
| `rl_walking/deploy/deploy_chain.py` | session | `4e9f917e` |
| `rl_walking/deploy/policy_runner.py` | session | `5ae6d91f` |
| `rl_walking/deploy/replay_bridge_log.py` | sync | `cb5f3b62` |
| `rl_walking/deploy/sim_validate_runner.py` | sync | `40571ecf` |
| `rl_walking/deploy/sim_validate_runner_stage4.py` | sync | `189d558e` |
| `robot_runtime/__init__.py` | session | `0ba2afc7` |
| `robot_runtime/command_types.py` | sync | `6d387487` |
| `robot_runtime/dof12_mapping.py` | session | `f1b3db6c` |
| `robot_runtime/dof_mapping.py` | sync | `03ed46bc` |
| `robot_runtime/joint_limits.py` | session | `63d466c9` |
| `robot_runtime/mode_types.py` | sync | `d77ae8dd` |
| `robot_runtime/robot_model.py` | sync | `3002c11c` |
| `tests/test_dof12_mapping_cache.py` | test | `b486903a` |
| `tests/test_stage8_baseline_from_hold.py` | test | `b92cf07a` |
| `tests/test_stage8_cmd_timing.py` | test | `27845170` |

## 4. 전송하지 않는 차이 (젯슨 인벤토리에 있고 HEAD 와 다르거나 HEAD 에 없는데 계획에 넣지 않은 것)

| 경로 | 젯슨 md5 | HEAD | 사유 |
|---|---|---|---|
| `config/robot_12dof_hardware_map.json` | `41b68a6f` | `0630c018` | config/ — 하드웨어맵·영점·보정·지연 옵션 파일(젯슨이 정본) |
| `rl_walking/actions.py` | `a52f687f` | `6232ced2` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/agents.py` | `0a6ce7cf` | `0627b3ef` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/biped12_cfg.py` | `c3ac3049` | `46f13c60` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/env_cfg.py` | `2531b750` | `b8d2fb2d` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/mdp_rewards.py` | `cee5b7d5` | `3a268ab1` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/mdp_stage4.py` | `83225f61` | `46de8d50` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/scripts/make_base_urdf.py` | `b6ee93f7` | `2ad80043` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/scripts/train.py` | `47c73fdb` | `bf28723e` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/stage4_env_cfg.py` | `aa7f0225` | `c74fd26b` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/stage4_v2_env_cfg.py` | `0604daa5` | `0af183fa` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |
| `rl_walking/symmetry.py` | `8a27b8d5` | `f21ef06c` | 학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요) |

젯슨에만 있거나 이번 배포와 무관한 정책 파일 47 개(구 세대 exported·env.yaml)는 건드리지 않는다.

HEAD 의 `jetson/` 아래에 있지만 계획에 없는 것: `jetson/calib_data/`(보정 자료 — 금지), `jetson/bench/`(개발 PC 벤치), `jetson/systemd/`(서비스 정의 — 이번 배포는 서비스를 바꾸지 않는다), `jetson/__init__.py`·`README.md`·`deploy_*.sh`(구 배포 도구).

## 5. 진입점별 import 사슬 (저장소 사본 기준, ast 로 추적 — 함수 안 지연 import 포함)

### 콘솔이 저장소 사본에서 띄우는 것 + 콘솔

| 진입점 | 저장소 안 의존 | 저장소 밖 모듈(제3자·젯슨 전용) |
|---|---|---|
| `jetson/scripts/operator_console.py` | `jetson/scripts/__init__.py`, `jetson/scripts/console_demo.py`, `jetson/scripts/console_state.py`, `jetson/scripts/w3_cmd.py`, `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/w3_deploy.py` | gymnasium, numpy, rclpy, std_msgs |
| `jetson/scripts/rl_bridge_node.py` | `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/c5_switch.py`, `rl_walking/deploy/deploy_chain.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/kew3_runner.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_contingency.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | gymnasium, numpy, onnxruntime, rclpy, std_msgs, torch |
| `jetson/scripts/stage8_12axis_mit_control_node.py` | `protocols/__init__.py`, `protocols/ak_mit_command.py`, `protocols/ak_mit_decoder.py`, `robot_runtime/__init__.py`, `robot_runtime/dof12_mapping.py`, `robot_runtime/joint_limits.py` | can, rclpy, std_msgs |

### 운영 도구·selftest

| 진입점 | 저장소 안 의존 | 저장소 밖 모듈(제3자·젯슨 전용) |
|---|---|---|
| `jetson/scripts/robotctl.py` | `jetson/scripts/console_state.py` | 없음 |
| `jetson/scripts/w3_cmd.py` | `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/w3_deploy.py` | gymnasium, numpy, rclpy, std_msgs |
| `jetson/scripts/latency_check.py` | 없음 | rclpy, std_msgs |
| `jetson/scripts/selftest_w3_bridge_node.py` | `jetson/scripts/rl_bridge_node.py`, `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/c5_switch.py`, `rl_walking/deploy/deploy_chain.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/kew3_runner.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_contingency.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | gymnasium, numpy, onnxruntime, rclpy, std_msgs, torch |
| `jetson/scripts/posture_probe.py` | `jetson/scripts/rl_bridge_node.py`, `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/c5_switch.py`, `rl_walking/deploy/deploy_chain.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/kew3_runner.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_contingency.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | gymnasium, numpy, onnxruntime, rclpy, std_msgs, torch |
| `jetson/scripts/zero_kinematic_check.py` | `jetson/scripts/rl_bridge_node.py`, `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/c5_switch.py`, `rl_walking/deploy/deploy_chain.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/kew3_runner.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_contingency.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | gymnasium, numpy, onnxruntime, rclpy, std_msgs, torch |
| `jetson/scripts/zero_anchor.py` | `jetson/scripts/boot_pose_check.py` | ak_mit_command, ak_mit_decoder, can |
| `jetson/scripts/analyze_fall.py` | 없음 | numpy |
| `rl_walking/deploy/kew3_runner.py` | `rl_walking/__init__.py`, `rl_walking/deploy/__init__.py`, `rl_walking/deploy/c5_switch.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | gymnasium, numpy, onnxruntime, torch |
| `rl_walking/deploy/c5_switch.py` | `rl_walking/deploy/__init__.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/leg_odometry.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/policy_runner_stage4.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | numpy, onnxruntime, torch |
| `rl_walking/deploy/policy_runner_stage4.py` | `rl_walking/deploy/__init__.py`, `rl_walking/deploy/hturn2.py`, `rl_walking/deploy/policy_runner.py`, `rl_walking/deploy/w3_deploy.py`, `robot_runtime/__init__.py`, `robot_runtime/joint_limits.py` | numpy, onnxruntime, torch |
| `rl_walking/deploy/w3_contingency.py` | 없음 | numpy |

### 젯슨에서 `~/rl_calib` 사본으로 도는 것(저장소 사본의 사슬 — 이번 배포 대상 아님)

| 진입점 | 저장소 안 의존 | 저장소 밖 모듈(제3자·젯슨 전용) |
|---|---|---|
| `jetson/scripts/hang_stand.py` | 없음 | ak_mit_command, ak_mit_decoder, can |
| `jetson/scripts/boot_pose_check.py` | 없음 | ak_mit_command, ak_mit_decoder, can |
| `jetson/scripts/release_all.py` | 없음 | ak_mit_command, ak_mit_decoder, can |
| `jetson/scripts/stage8_cmd.py` | 없음 | rclpy, std_msgs |
| `jetson/scripts/ping_all.py` | 없음 | ak_mit_command, ak_mit_decoder, can |
| `jetson/scripts/imu_adapter_node.py` | 없음 | rclpy, sensor_msgs, std_msgs |
| `jetson/scripts/foot_force_adapter_node.py` | 없음 | rclpy, std_msgs |

## 6. 콘솔이 띄우는 스크립트의 위치

- 저장소 사본(`~/humanoid_repo/jetson/scripts`)에서: `analyze_fall.py`, `posture_probe.py`, `rl_bridge_node.py`, `stage8_12axis_mit_control_node.py`, `zero_anchor.py`, `zero_kinematic_check.py` → 이번 배포로 바뀐다(위 표).
- `~/rl_calib` 에서: `boot_pose_check.py`, `can_up.sh`, `hang_stand.py`, `level_pose.json`, `release_all.py`, `sensors_up.sh`, `stage8_cmd.py` → **배포 대상이 아니다**(젯슨 정본). 저장소에 같은 이름의 사본이 `jetson/scripts/` 에 있는지는 아래 표. `~/rl_calib` 의 실제 md5 는 인벤토리에 없다 — 배포 스크립트 dry-run 이 읽어 저장소 사본과 나란히 보여 준다(읽기만).

| `~/rl_calib` 의 파일 | 저장소 사본 | HEAD md5 | 젯슨 저장소 사본(인벤토리) |
|---|---|---|---|
| `ak_mit_command.py` | `protocols/ak_mit_command.py` | `52543d3d` | `52543d3d` |
| `ak_mit_decoder.py` | `protocols/ak_mit_decoder.py` | `23d9de66` | `23d9de66` |
| `boot_pose_check.py` | `jetson/scripts/boot_pose_check.py` | `cf24508e` | `cf24508e` |
| `can_channel` | 없음 — **젯슨 정본·저장소 사본 없음** | - | - |
| `can_up.sh` | `jetson/scripts/can_up.sh` | `42f27f3a` | `42f27f3a` |
| `foot_force_adapter_node.py` | `jetson/scripts/foot_force_adapter_node.py` | `9b13d698` | `9b13d698` |
| `hang_stand.py` | `jetson/scripts/hang_stand.py` | `2692298a` | `2692298a` |
| `imu_adapter_node.py` | `jetson/scripts/imu_adapter_node.py` | `7feaf579` | `7feaf579` |
| `imu_mount_calib.json` | `config/imu_mount_calib.json` (배포 금지 — 젯슨 정본) | `1617b7e8` | `1617b7e8` |
| `joint_limits_12dof.json` | `config/joint_limits_12dof.json` (배포 금지 — 젯슨 정본) | `648bcb7f` | `648bcb7f` |
| `last_pose_journal.json` | 없음 — **젯슨 정본·저장소 사본 없음** | - | - |
| `level_pose.json` | 없음 — **젯슨 정본·저장소 사본 없음** (날짜 붙은 옛 사본 `jetson/calib_data/level_pose_0914.json`, `jetson/calib_data/level_pose_0918.json` 은 있으나 실행본과의 일치는 미확인) | - | - |
| `ping_all.py` | `jetson/scripts/ping_all.py` | `18c4b94a` | 없음 |
| `release_all.py` | `jetson/scripts/release_all.py` | `e06b80d1` | `e06b80d1` |
| `robot_12dof_hardware_map.json` | `config/robot_12dof_hardware_map.json` (배포 금지 — 젯슨 정본) | `0630c018` | `41b68a6f` |
| `sensors_up.sh` | `jetson/scripts/sensors_up.sh` | `3b372a2b` | `73fec1f3` |
| `stack_down.sh` | `jetson/scripts/stack_down.sh` | `7f8f7925` | 없음 |
| `stage8_cmd.py` | `jetson/scripts/stage8_cmd.py` | `344aae83` | 없음 |
| `zero_torque_read.py` | `jetson/scripts/zero_torque_read.py` | `fd7eec04` | `fd7eec04` |

## 7. 정책 파일

브리지가 실제로 읽는 것은 `exported/policy.onnx`(onnxruntime 이 있으면 우선) 또는 `policy.pt`, 그리고 `params/env.yaml`(관측 필터·기본자세·W3 명령 범위 계약)이다. `SOURCE_CHECKPOINT` 는 출처 표시일 뿐 읽지 않는다. `params/agent.yaml`·`model_*.pt` 는 싣지 않는다.

| 파일 | 크기 | md5 | 젯슨(인벤토리) | 동작 | 쓰는 키 |
|---|---|---|---|---|---|
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/SOURCE_CHECKPOINT` | 15 | `bdd15e0132b559c38d09d986ed08c88b` | **없음 — 새로 간다** | send | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/policy.onnx` | 1321751 | `2b718875abe606685dcccb9d08704935` | **없음 — 새로 간다** | send | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported/policy.pt` | 1332468 | `7bbf24f5d5da1a85704db999b7a1f330` | **없음 — 새로 간다** | send | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/params/env.yaml` | 57004 | `dcc9ceb51fad439196e00e57816a56f7` | **없음 — 새로 간다** | send | stand_crouch_g21h |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/SOURCE_CHECKPOINT` | 15 | `1ad3809e4af40b1334e497b0d3714a7f` | **없음 — 새로 간다** | send | stand_w3s3d_15800, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.onnx` | 1321751 | `d908cd92ba0916e790f9e3e28ff1d882` | **없음 — 새로 간다** | send | stand_w3s3d_15800, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.pt` | 1332468 | `10e3ee405e1534ebe67574ddbeb9a2a2` | **없음 — 새로 간다** | send | stand_w3s3d_15800, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800 |
| `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml` | 60296 | `41d1e62aed455029be7351a1db43f48f` | **없음 — 새로 간다** | send | stand_w3s3d_15800, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800 |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.onnx` | 1168151 | `c577a65a3edbd9f856fbc4b0912f5908` | 같음(이미 있음) | verify | stand_v18b(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.pt` | 1179026 | `fac90eb5490b47a77ca76281631af442` | 같음(이미 있음) | verify | stand_v18b(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/params/env.yaml` | 41370 | `a2f1f885aa64d01ef77f14e05549df00` | 같음(이미 있음) | verify | stand_v18b(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.onnx` | 1321751 | `a090aa871b654a099ada80b4287046fd` | 같음(이미 있음) | verify | stand_crouch_g11a(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/exported/policy.pt` | 1332626 | `51f3e2b7b73168ec881312175447684b` | 같음(이미 있음) | verify | stand_crouch_g11a(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-28_23-19-13_g11a/params/env.yaml` | 54587 | `ab1c12ebc0a55ec3554c669f5eb5680d` | 같음(이미 있음) | verify | stand_crouch_g11a(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.onnx` | 1321751 | `6c5f8a7e08696cd07b1e5abf328fa616` | 같음(이미 있음) | verify | stand_crouch_g12(확인), stand_crouch_g12_ema0(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported/policy.pt` | 1332626 | `dc9cd5867932e15103684055476565ba` | 같음(이미 있음) | verify | stand_crouch_g12(확인), stand_crouch_g12_ema0(확인) |
| `logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/params/env.yaml` | 54587 | `0c07175148c9fdbbd71d9e1b125d7814` | 같음(이미 있음) | verify | stand_crouch_g12(확인), stand_crouch_g12_ema0(확인) |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.onnx` | 1321751 | `43f113109a369a60e976a391cf8f17cd` | 같음(이미 있음) | verify | stand_crouch_g21a, stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported/policy.pt` | 1332626 | `2464f5cdba2e819d03ffa2d8d03bbb69` | 같음(이미 있음) | verify | stand_crouch_g21a, stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/params/env.yaml` | 54800 | `ebda410ec37987f788df920ccac4f264` | 같음(이미 있음) | verify | stand_crouch_g21a, stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/SOURCE_CHECKPOINT` | 15 | `e24b90e328fadb2a5c7ef9d87721d1a3` | 같음(이미 있음) | verify | stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.onnx` | 1321751 | `57547697918f679543413a5b24c36555` | 같음(이미 있음) | verify | stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported/policy.pt` | 1332626 | `896ca613bc402de02caa8302988f9a68` | 같음(이미 있음) | verify | stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |
| `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/params/env.yaml` | 62452 | `7ff27617abdb9dd8ee6155c5d7703eaf` | 같음(이미 있음) | verify | stand_c5kr_g21a_g24ra, stand_c5krh_g21a_g24ra, stand_kew3_g21a_g24ra_w3s3d15800, stand_kew3q_g21a_g24ra_w3s3d15800, stand_kew3qa_g21a_g24ra_w3s3d15800, stand_c5_g21a_g24ra(확인) |

키가 가리키는 폴더(브리지 레지스트리):

- `stand_crouch_g21a` = `logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported`
- `stand_c5kr_g21a_g24ra` = 서기 `stand_crouch_g21a` + 회복 `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported`
- `stand_c5krh_g21a_g24ra` = 서기 `stand_crouch_g21a` + 회복 `logs/rsl_rl/biped12_stepb0/2026-10-04_10-08-41_g24ra_seed1_from_w2e20294/exported`
- `stand_w3s3d_15800` = `logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported`
- `stand_kew3_g21a_g24ra_w3s3d15800` = 걷기 `stand_w3s3d_15800` + 서기·회복 `stand_c5kr_g21a_g24ra` (전환 carry, 정착 0.5 s) — 자체 파일 없음, 하위 키의 파일을 쓴다
- `stand_kew3q_g21a_g24ra_w3s3d15800` = 걷기 `stand_w3s3d_15800` + 서기·회복 `stand_c5kr_g21a_g24ra` (전환 carry, 정착 0.5 s) — 자체 파일 없음, 하위 키의 파일을 쓴다
- `stand_crouch_g21h` = `logs/rsl_rl/biped12_stage4/2026-10-08_16-04-12_g21h2_s1_from_12299/exported`
- `stand_kew3qa_g21a_g24ra_w3s3d15800` = 걷기 `stand_w3s3d_15800` + 서기·회복 `stand_c5kr_g21a_g24ra` (전환 carry, 정착 0.5 s) — 자체 파일 없음, 하위 키의 파일을 쓴다

## 8. selftest 기준 원문

`jetson/scripts/selftest_w3_bridge_node.py` 의 비트 비교 기준 커밋 `68493c11`, `95946e41`, `b79370ca` 의 브리지 원문을 `deploy/cand/head_1008/ref_bridge/` 에 둔다(젯슨에는 git 이 없어 `git show` 가 실패한다 — 실행기 `deploy/cand/head_1008/selftest_w3_node_refs.py` 가 파일에서 읽는다).
