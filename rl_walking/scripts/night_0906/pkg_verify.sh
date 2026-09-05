#!/bin/bash
# A6 검증: 패키지 조립 → 젯슨형 트리(하드웨어맵 + 기배포 세대 walk/stand_v3~v9, stand_v14 없음)에 풀기
#          → md5 종료코드 → 브리지 selftest(--checkpoint stand_v18b, SKIP 표기 확인) → 어댑터·release_all selftest
set -uo pipefail
REPO=/home/ryu/humanoid_leg_test1
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
STAGE=$S/jetson_sim
cd "$REPO"
OUT=$(bash jetson/deploy_package.sh | tee /dev/stderr | grep -o 'deploy/stand_7차_[0-9_]*\.tar\.gz' | head -1)
[ -n "$OUT" ] && [ -f "$OUT" ] || { echo "패키지 없음"; exit 1; }
rm -rf "$STAGE"; mkdir -p "$STAGE/config"
cp config/robot_12dof_hardware_map.json "$STAGE/config/"
for run in 2026-07-26_14-53-03 2026-08-03_06-12-07 2026-08-03_18-44-48 2026-08-04_11-31-27 \
           2026-08-04_13-13-49 2026-08-08_21-24-48 2026-08-09_03-59-22; do
  mkdir -p "$STAGE/logs/rsl_rl/biped12_stage4/$run/exported"
  cp "logs/rsl_rl/biped12_stage4/$run/exported/policy.pt" "$STAGE/logs/rsl_rl/biped12_stage4/$run/exported/"
done
cd "$STAGE"
tar xzf "$REPO/$OUT"
if LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5; then echo "MD5_OK ($(wc -l < deploy/MANIFEST.md5) files)"; else echo MD5_FAIL; exit 1; fi
[ -d logs/rsl_rl/biped12_stage4/2026-09-04_02-06-07 ] && echo "⚠ stand_v14 디렉토리 있음 — 젯슨 재현 아님"
echo "== bridge selftest (--checkpoint stand_v18b) =="
python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_v18b > "$S/pkg_selftest.out" 2>&1
grep -E '^\[FAIL\]|^selftest:' "$S/pkg_selftest.out"
grep -o 'SKIP [0-9]* 미배포: [a-z0-9_, ]*' "$S/pkg_selftest.out" || echo "(SKIP 표기 없음)"
grep -o 'stand_v18b [0-9.]*ms' "$S/pkg_selftest.out" || echo "(stand_v18b 로드 흔적 없음)"
echo "== adapter/release_all =="
python3 jetson/scripts/foot_force_adapter_node.py --selftest
python3 jetson/scripts/release_all.py --selftest | tail -1
echo "== 콘솔 드롭다운 렌더 =="
python3 - <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("oc", "jetson/scripts/operator_console.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
page = m.render_page()
i = page.index("<select id=ckpt>")
print(page[i:i + 120])
EOF
echo "PKG=$OUT"
