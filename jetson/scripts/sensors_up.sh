#!/bin/bash
# 센서 스택 멱등 기동 — 구 인스턴스 검증-종료 후 4종 재기동 + 토픽 실검증
# (0802 새벽 사고 재발 방지: 중복 기동 → ROS 컨텍스트 붕괴/토픽 이중화로 15분 소모)
#
# 종료 정책: PID파일 우선, 없으면 pgrep 후보를 /proc cmdline 정확 대조로 검증
# 후에만 개별 kill (명시 PID 원칙 — 패턴 일괄 킬 금지).
# 검증: hz 도구는 데몬 캐시 꼬임으로 거짓 경고 이력 — echo 실데이터로만 판정.
set -u
source /opt/ros/humble/setup.bash
source ~/humanoid_ws/install/setup.bash
mkdir -p ~/logs ~/run

declare -A NODES=(
  [imu_node]="/home/mama/humanoid_ws/install/humanoid_imu/lib/humanoid_imu/imu_node"
  [pressure]="/home/mama/humanoid_ws/install/humanoid_pressure/lib/humanoid_pressure/pressure_serial_node"
  [imu_adapter]="python3 /home/mama/rl_calib/imu_adapter_node.py"
  [foot_adapter]="python3 /home/mama/rl_calib/foot_force_adapter_node.py"
)

kill_verified() {  # $1=pid $2=서명(실행파일 basename) — argv[0..1] 위치 정밀 매칭
  # (부분문자열 매칭은 'bash -c "...서명..."' 부모 셸 오인 종료 — stack_down에서
  #  실전 발견된 버그 클래스. 실제 노드는 argv[0] 또는 argv[1]이 서명 그 자체)
  local pid=$1 sig=$2 arg match=0
  [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && return 0
  [ -r "/proc/$pid/cmdline" ] || return 0
  while IFS= read -r arg; do
    [ "$(basename "$arg")" = "$sig" ] && match=1
  done < <(tr '\0' '\n' < "/proc/$pid/cmdline" | head -2)
  if [ "$match" = 1 ]; then
    kill "$pid" 2>/dev/null && echo "  종료: PID $pid ($sig)"
  fi
}

for name in "${!NODES[@]}"; do
  sig=$(basename "${NODES[$name]##* }")
  # 1) PID파일 기반
  if [ -f ~/run/"$name".pid ]; then
    kill_verified "$(cat ~/run/"$name".pid)" "$sig"
    rm -f ~/run/"$name".pid
  fi
  # 2) 유령 인스턴스 (cmdline 대조 검증 후 개별 종료)
  for pid in $(pgrep -f "$sig" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    kill_verified "$pid" "$sig"
  done
done
sleep 1

for name in imu_node pressure; do
  nohup ${NODES[$name]} > ~/logs/"$name".log 2>&1 < /dev/null &
  echo $! > ~/run/"$name".pid
done
sleep 4
for name in imu_adapter foot_adapter; do
  nohup ${NODES[$name]} > ~/logs/"$name".log 2>&1 < /dev/null &
  echo $! > ~/run/"$name".pid
done
sleep 5

ok=0
for i in 1 2 3; do
  IMU=$(timeout 3 ros2 topic echo /humanoid/imu --field data 2>/dev/null | head -1)
  FOOT=$(timeout 3 ros2 topic echo /humanoid/foot_force --field data 2>/dev/null | head -1)
  if [ -n "$IMU" ] && [ -n "$FOOT" ]; then ok=1; break; fi
  sleep 2
done
if [ "$ok" = 1 ]; then
  echo "[sensors_up] OK — imu·foot_force 실데이터 확인"
  echo "  IMU:  ${IMU:0:70}"
  echo "  FOOT: ${FOOT:0:70}"
else
  echo "[sensors_up] FAIL — 토픽 무응답. ~/logs/*.log 확인"
  exit 1
fi
