#!/bin/bash
# 젯슨 배포 원샷 (0906): 패키지 scp → 풀기 → md5 → selftest 3종 → ~/rl_calib/release_all.py 갱신 → 어댑터 경로·상수 확인/갱신 → 콘솔 재기동
#   bash jetson/deploy_to_jetson.sh [--host 172.16.35.64] [--pkg deploy/stand_7차_XXXX.tar.gz] [--wait] [--no-restart]
#   --wait : 젯슨이 응답할 때까지 60초마다 재시도한 뒤 배포 (승윤님 결정 후 백그라운드로 띄워 두는 용도)
# 하는 일은 파일 복사·검증·콘솔 서비스 재기동뿐이다 — CAN·모터·stage8 에는 손대지 않는다. 실패 단계에서 즉시 중단(종료코드 ≠0).
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=172.16.35.64; PKG=""; WAIT=0; RESTART=1; PASS="${JETSON_PW:-}"
while [ $# -gt 0 ]; do case "$1" in
  --host) HOST=$2; shift 2;; --pkg) PKG=$2; shift 2;; --wait) WAIT=1; shift;; --no-restart) RESTART=0; shift;;
  *) echo "인자 오류: $1"; exit 2;; esac; done
[ -n "$PKG" ] || PKG=$(ls -t deploy/stand_7차_*.tar.gz 2>/dev/null | head -1)
[ -f "$PKG" ] || { echo "패키지 없음: $PKG (bash jetson/deploy_package.sh 먼저)"; exit 2; }
SSH="sshpass -p $PASS ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no mama@$HOST"
SCP="sshpass -p $PASS scp -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
log() { echo "[$(date +%H:%M:%S)] $*"; }
if [ $WAIT = 1 ]; then
  until $SSH true 2>/dev/null; do log "젯슨 $HOST 응답 없음 — 60초 후 재시도"; sleep 60; done
fi
$SSH true 2>/dev/null || { log "젯슨 $HOST 접속 불가"; exit 1; }
log "젯슨 접속됨: $($SSH hostname)"
B=$(basename "$PKG")
log "[1/7] 패키지 $PKG → ~/"; $SCP "$PKG" "mama@$HOST:~/" || { log "scp 실패"; exit 1; }
log "[2/7] 풀기 + md5 (종료코드로 판정)"
$SSH "cd ~/humanoid_repo && tar xzf ~/$B && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5 && echo MD5_OK" || { log "md5 불일치 — 중단"; exit 1; }
log "[3/7] 브리지 selftest (--checkpoint stand_v18b)"
$SSH "cd ~/humanoid_repo && python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_v18b 2>&1 | grep -E '^\[(PASS|FAIL)\]|^selftest:' | cut -c1-150"
$SSH "cd ~/humanoid_repo && python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_v18b 2>&1 | grep -q '^selftest: ALL PASS'" || { log "브리지 selftest FAIL — 중단"; exit 1; }
log "[4/7] 어댑터·release_all selftest (저장소 사본)"
$SSH "cd ~/humanoid_repo && python3 jetson/scripts/foot_force_adapter_node.py --selftest && python3 jetson/scripts/release_all.py --selftest | tail -1" || { log "selftest FAIL — 중단"; exit 1; }
log "[5/7] ~/rl_calib/release_all.py 갱신 (백업 후 복사, 게이트 selftest)"
$SSH "cd ~/humanoid_repo && { [ -f ~/rl_calib/release_all.py ] && cp ~/rl_calib/release_all.py ~/rl_calib/release_all.py.bak_\$(date +%m%d_%H%M); } ; cp jetson/scripts/release_all.py ~/rl_calib/release_all.py && python3 ~/rl_calib/release_all.py --selftest | tail -1" || { log "release_all 갱신 실패 — 중단"; exit 1; }
log "[6/7] 발압력 어댑터: sensors_up.sh 가 띄우는 경로의 NEWTON_PER_COUNT 확인 (0.02650 이어야 함)"
$SSH 'grep -n "foot_force_adapter" ~/rl_calib/sensors_up.sh | head -3
for f in $(grep -o "[^ \"]*foot_force_adapter_node\.py" ~/rl_calib/sensors_up.sh | sort -u); do
  f2=$(eval echo "$f"); [ -f "$f2" ] || { echo "  $f2: 파일 없음"; continue; }
  v=$(grep -m1 "^NEWTON_PER_COUNT = " "$f2")
  if [ "$f2" -ef ~/humanoid_repo/jetson/scripts/foot_force_adapter_node.py ]; then echo "  $f2 (저장소 사본): $v"
  else
    echo "  $f2 (별도 사본): $v"
    if ! echo "$v" | grep -q "0.02650"; then cp "$f2" "$f2.bak_$(date +%m%d_%H%M)"; cp ~/humanoid_repo/jetson/scripts/foot_force_adapter_node.py "$f2"; echo "  → 갱신: $(grep -m1 "^NEWTON_PER_COUNT = " "$f2")"; fi
  fi
done'
if [ $RESTART = 1 ]; then
  log "[7/7] 콘솔 재기동 (systemd)"
  $SSH "echo $PASS | sudo -S systemctl restart operator-console 2>/dev/null; sleep 2; systemctl is-active operator-console"
else
  log "[7/7] 콘솔 재기동 생략(--no-restart) — 드롭다운 기본값 stand_v18b 는 재기동 후 반영"
fi
log "배포 완료 — 다음: 카드 0단계(can_up → boot_pose_check → sensors_up → selftest)는 승윤님이 현장에서."
