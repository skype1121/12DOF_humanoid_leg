#!/bin/bash
# 정책 스택 정리 — stage8·rl_bridge·hang_stand·레그스윙 잔여 인스턴스 검증-종료
# (0802 감사에서 발견: 세션 중단 후 살아남은 armed stage8 = 차기 전원 인가 시
#  낡은 베이스라인으로 명령 발사 위험. 세션 시작·종료 시 이 스크립트로 청소.)
# 종료 정책: /proc cmdline 정확 대조 후 개별 kill (명시 PID 원칙).
set -u
SIGS=(
  "stage8_12axis_mit_control_node.py"
  "rl_bridge_node.py"
  "hang_stand.py"
  "hang_leg_swing.py"
  "wiggle_watch.py"
)
found=0
for sig in "${SIGS[@]}"; do
  for pid in $(pgrep -f "$sig" 2>/dev/null); do
    [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && continue
    [ -r "/proc/$pid/cmdline" ] || continue
    # 정밀 매칭: argv[0..1]에서만 시그니처 확인 — 'bash -c "...stage8..."' 같은
    # 부모 셸(긴 문자열 인자에만 포함)을 오인 종료하는 버그 방지 (0802 실전 발견:
    # 자기 ssh 부모를 죽여 exit 255). 실제 노드는 'python3 /경로/시그니처.py'라
    # argv[1]의 basename이 정확히 일치한다.
    argv01=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -2)
    match=0
    while IFS= read -r arg; do
      [ "$(basename "$arg")" = "$sig" ] && match=1
    done <<< "$argv01"
    if [ "$match" = 1 ]; then
      kill "$pid" 2>/dev/null && { echo "종료: PID $pid ($sig)"; found=1; }
    fi
  done
done
[ "$found" = 0 ] && echo "[stack_down] 잔여 스택 없음 — 깨끗함"
sleep 1
LEFT=$(ps -C python3 -o pid,args --no-headers 2>/dev/null | grep -E "stage8|rl_bridge|hang_" | head -3)
if [ -n "$LEFT" ]; then
  echo "[stack_down] ⚠ 잔존:"; echo "$LEFT"
  exit 1
fi
echo "[stack_down] OK"
