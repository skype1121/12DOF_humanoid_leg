#!/bin/bash
# 7차 실기 배포 패키지 조립 (로컬) — 젯슨 접속 불가 상태에서도 미리 만들어 두고, 접속되면 복사·검증.
#   bash jetson/deploy_package.sh            → deploy/stand_7차_<YYYYMMDD_HHMM>.tar.gz + deploy/MANIFEST.md5
#   젯슨: cd ~/humanoid_repo && tar xzf ~/stand_7차_*.tar.gz && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK
#   (한국어 로케일에선 md5sum 이 'OK' 대신 '성공'을 찍으므로 문자열 grep 금지 — 종료코드로 판정)
#   ⚠ config/robot_12dof_hardware_map.json 은 일부러 넣지 않는다 — 젯슨의 맵(실물 영점)이 유일한 원본.
# 내용: models/<gen>/policy.pt → 브리지 CHECKPOINTS 경로(logs/rsl_rl/biped12_stage4/<run>/exported/policy.pt),
#       브리지·분석기·콘솔·프로브, rl_walking/deploy, robot_runtime, 실행카드.
# 정책 등록 근거는 models/<gen>/meta.json (run·ckpt·md5). 젯슨 복사는 승윤님 결정 후.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
STAMP=$(date +%Y%m%d_%H%M)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
GENS=${GENS:-"stand_v15 stand_v16 stand_v17 stand_v18 stand_v18b"}

n=0
for g in $GENS; do
  meta="models/$g/meta.json"
  [ -f "$meta" ] || { echo "  (건너뜀: $g 미등록)"; continue; }
  run=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "$meta")
  dst="logs/rsl_rl/biped12_stage4/$run/exported"
  mkdir -p "$STAGE/$dst"
  cp "models/$g/policy.pt" "$STAGE/$dst/policy.pt"
  [ -f "models/$g/policy.onnx" ] && cp "models/$g/policy.onnx" "$STAGE/$dst/policy.onnx"
  # 학습 env 계약(params/env.yaml) — 브리지 selftest '학습 계약' 이 DEFAULT_POSE·스케일·슬루·주기를 대조 (0906)
  if [ -f "models/$g/env.yaml" ]; then mkdir -p "$STAGE/logs/rsl_rl/biped12_stage4/$run/params"; cp "models/$g/env.yaml" "$STAGE/logs/rsl_rl/biped12_stage4/$run/params/env.yaml"; fi
  echo "  $g → $dst/policy.pt ($(md5sum models/$g/policy.pt | cut -c1-8)…)"
  n=$((n+1))
done
[ "$n" -gt 0 ] || { echo "정책이 하나도 없음"; exit 1; }

# 브리지가 CHECKPOINTS 에 실제로 아는 정책만 의미가 있다 — 등록 누락 검출
for g in $GENS; do
  [ -f "models/$g/meta.json" ] || continue
  grep -q "\"$g\"" jetson/scripts/rl_bridge_node.py || echo "  ⚠ $g 가 rl_bridge_node.py CHECKPOINTS 에 없음 — 등록 필요"
done

for f in jetson/scripts/rl_bridge_node.py jetson/scripts/analyze_fall.py jetson/scripts/posture_probe.py \
         jetson/scripts/operator_console.py docs/stand_v18_7차_실행카드.md docs/hang_to_stand_실물절차.md docs/A1_stand_hang_v5_실행카드.md \
         jetson/scripts/stiffness_log.py \
         jetson/scripts/foot_force_adapter_node.py jetson/scripts/release_all.py jetson/scripts/zero_kinematic_check.py \
         jetson/scripts/cop_from_pads.py jetson/scripts/boot_pose_check.py \
         jetson/calib_data/sim_default_stand_targets_20260727.json; do   # selftest 수치 앵커 대조 파일
  mkdir -p "$STAGE/$(dirname "$f")"; cp "$f" "$STAGE/$f"
done
mkdir -p "$STAGE/rl_walking/deploy" "$STAGE/robot_runtime" "$STAGE/config"
cp rl_walking/deploy/*.py "$STAGE/rl_walking/deploy/"
cp rl_walking/__init__.py "$STAGE/rl_walking/__init__.py" 2>/dev/null || true
cp robot_runtime/*.py "$STAGE/robot_runtime/"
cp config/joint_limits_12dof.json "$STAGE/config/" 2>/dev/null || true

mkdir -p "$STAGE/deploy" deploy
( cd "$STAGE" && find . -type f ! -path ./deploy/MANIFEST.md5 -print0 | sort -z | xargs -0 md5sum > deploy/MANIFEST.md5 )
cp "$STAGE/deploy/MANIFEST.md5" deploy/MANIFEST.md5
OUT="deploy/stand_7차_$STAMP.tar.gz"
tar czf "$OUT" -C "$STAGE" .
echo "패키지: $OUT ($(du -h "$OUT" | cut -f1), 파일 $(wc -l < deploy/MANIFEST.md5)개)"
echo "복사: bash jetson/deploy_to_jetson.sh (또는 scp $OUT mama@<젯슨IP>:~/) → 젯슨에서 tar xzf + LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 + --selftest"
