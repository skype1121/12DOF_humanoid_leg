#!/bin/bash
# 젯슨 상태 재현: 패키지(GENS=v15~v18b) + 기배포 세대(walk, stand_v3~v9) 는 있고 stand_v14 만 없음.
set -euo pipefail
REPO=/home/ryu/humanoid_leg_test1
STAGE=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/jetson_sim
rm -rf "$STAGE"; mkdir -p "$STAGE"
cd "$REPO"
# (1) 패키지가 싣는 것 — deploy_package.sh:16-29 와 동일 논리
GENS="stand_v15 stand_v16 stand_v17 stand_v18 stand_v18b"
for g in $GENS; do
  run=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "models/$g/meta.json")
  dst="logs/rsl_rl/biped12_stage4/$run/exported"
  mkdir -p "$STAGE/$dst"; cp "models/$g/policy.pt" "$STAGE/$dst/policy.pt"
  [ -f "models/$g/policy.onnx" ] && cp "models/$g/policy.onnx" "$STAGE/$dst/policy.onnx"
done
for f in jetson/scripts/rl_bridge_node.py jetson/scripts/analyze_fall.py jetson/scripts/posture_probe.py \
         jetson/scripts/operator_console.py jetson/calib_data/sim_default_stand_targets_20260727.json; do
  mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"
done
mkdir -p "$STAGE/rl_walking/deploy" "$STAGE/robot_runtime" "$STAGE/config"
cp rl_walking/deploy/*.py "$STAGE/rl_walking/deploy/"
cp rl_walking/__init__.py "$STAGE/rl_walking/__init__.py" 2>/dev/null || true
cp robot_runtime/*.py "$STAGE/robot_runtime/"
cp config/joint_limits_12dof.json "$STAGE/config/"
# (2) 젯슨에 이미 있는 것 — 하드웨어맵(젯슨 원본) + 기배포 세대 walk/stand_v3~v9
cp config/robot_12dof_hardware_map.json "$STAGE/config/"
for run in 2026-07-26_14-53-03 2026-08-03_06-12-07 2026-08-03_18-44-48 2026-08-04_11-31-27 \
           2026-08-04_13-13-49 2026-08-08_21-24-48 2026-08-09_03-59-22; do
  mkdir -p "$STAGE/logs/rsl_rl/biped12_stage4/$run/exported"
  cp logs/rsl_rl/biped12_stage4/$run/exported/policy.pt "$STAGE/logs/rsl_rl/biped12_stage4/$run/exported/"
done
echo "== staged exported dirs =="; ls "$STAGE/logs/rsl_rl/biped12_stage4/"
echo "== stand_v14 dir present? =="; ls -d "$STAGE/logs/rsl_rl/biped12_stage4/2026-09-04_02-06-07" 2>&1 || true
echo "== selftest =="
cd "$STAGE"
python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -v "^\[WARN\]" | cut -c1-600
echo "== exit: ${PIPESTATUS[0]} =="
echo "== card grep =="
python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -o "stand_v18b[^;]*" || echo "(grep 출력 없음)"
