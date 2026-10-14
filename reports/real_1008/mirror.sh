#!/usr/bin/env bash
# 젯슨 세션 기록·브리지 블랙박스·콘솔 상태를 로컬로 따라 받는다.
# 젯슨에서 하는 일은 읽기뿐이다: find · rsync(보내기) · GET /api/state. 제어·ROS CLI·쓰기 없음.
# 사용: mirror.sh <control-socket 경로> ["YYYY-mm-dd HH:MM:SS" 이후 것만]
set -u
HOST=mama@172.16.35.89
DST=/home/ryu/humanoid_leg_test1/reports/real_1008/raw
CM=$1
SINCE=${2:-"2026-10-08 21:16:00"}
mkdir -p "$DST"
# 유닉스 소켓 경로 길이 제한(108 자) 때문에 소켓 폴더로 옮겨 상대 경로로 쓴다.
cd "$(dirname "$CM")" || exit 1
CM=./$(basename "$CM")
SSH="ssh -o ControlPath=$CM -o BatchMode=yes -o ConnectTimeout=5"
while true; do
  if ! $SSH -O check $HOST 2>/dev/null; then
    sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ControlMaster=yes -o ControlPath=$CM \
      -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=5 -N -f $HOST 2>>"$DST/mirror.err" \
      || { echo "$(date '+%T') 접속 실패" >>"$DST/mirror.err"; sleep 3; continue; }
  fi
  $SSH $HOST "cd ~/logs && find . -maxdepth 1 -name 'rl_bridge_*.jsonl' -newermt '$SINCE' -printf '%P\n'; find sessions -mindepth 1 -maxdepth 1 -name 'session_*' -newermt '$SINCE' -printf '%p\n'" \
    >"$DST/.list.tmp" 2>>"$DST/mirror.err" \
    && rsync -a -r --append --timeout=10 -e "$SSH" --files-from="$DST/.list.tmp" $HOST:logs/ "$DST/" 2>>"$DST/mirror.err"
  $SSH $HOST "curl -s -m 2 http://127.0.0.1:8800/api/state" 2>/dev/null | python3 -c '
import json,sys,time
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for p in d.get("procs",{}).values():
    if isinstance(p,dict) and "lines" in p: p["lines"]=p["lines"][-3:]
d["_t"]=round(time.time(),2)
print(json.dumps(d,ensure_ascii=False,separators=(",",":")))' >>"$DST/state.jsonl"
  date +%s >"$DST/.heartbeat"
  sleep 2
done
