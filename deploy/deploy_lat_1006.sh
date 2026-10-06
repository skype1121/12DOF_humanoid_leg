#!/bin/bash
# 1006 stage8 지연 단축 단계 적용 — stage8 노드(지연 옵션·tick_timing·hold 송신 계수) · 콘솔(config/stage8_latency_opts 읽기) ·
# 판정기 latency_check.py(읽기 전용) + 회귀 테스트 2종 → 젯슨.  절차 카드: reports/LATENCY_STAGE_1006.md
#
# **사용자(승윤님) 승인 뒤에만 --apply / --stage 실행.** 기본은 점검만(dry-run):
#   (인자 없음)        : 로컬 매니페스트(deploy/lat_1006.md5) 대조 → 로컬 selftest·단위 테스트 → 젯슨 현재 md5·제어 프로세스·
#                        옵션 파일·콘솔 환경변수 확인(읽기만)
#   --apply            : 제어 프로세스(브리지·stage8) 없음 확인 → 백업(.bak_1006) → 복사 → md5 대조 → 젯슨 selftest·테스트
#   --restart-console  : (--apply 와 함께) 콘솔 서비스 재기동 — 젯슨 콘솔이 옵션 파일을 읽는 판(a9bdc83c 이후)이 아닐 때만 필요
#   --stage off|fas|soc: 젯슨 config/stage8_latency_opts 내용 설정 (off=비움, fas='fas', soc='fas,soc').
#                        제어 프로세스가 없을 때만. 효과는 **다음 stage8 기동부터**(콘솔 '스택' 정상 순서로 재기동).
#                        soc 는 젯슨 ~/logs/latency_check/last_fas.json 이 PASS(기대 fas·실로드 fas 단독)이고
#                        현재 파일이 'fas' 이며 그 결과가 fas 설정 이후 측정일 때만 허용. 두 기능을 한 번에 켜는 경로는 없다.
#   --rollback         : 젯슨 옵션 파일 비우기(= 종전 경로, 다음 stage8 기동부터) + .bak_1006 복원 명령 안내(실행하지 않음)
# 하드웨어맵은 보내지 않는다(젯슨 맵 = 실물 영점의 유일한 원본).
# 이 스크립트는 CAN 송신·모터 구동·ARM 을 하지 않는다. 프로세스 종료도 하지 않는다(pkill 없음).
set -euo pipefail
cd "$(dirname "$0")/.."
J=mama@172.16.35.89; R=/home/mama/humanoid_repo
SSH="sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 $J"
SCP="sshpass -p "$JETSON_PW" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8"
FILES="jetson/scripts/stage8_12axis_mit_control_node.py jetson/scripts/operator_console.py jetson/scripts/latency_check.py
tests/test_latency_check.py tests/test_stage8_latency_options.py"
OPTS=config/stage8_latency_opts
MANIFEST=deploy/lat_1006.md5
APPLY=0; RESTART=0; STAGE=""; ROLLBACK=0; LOCAL_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --apply) APPLY=1;;
    --restart-console) RESTART=1;;
    --stage) shift; STAGE=${1:-};;
    --rollback) ROLLBACK=1;;
    --local-only) LOCAL_ONLY=1;;   # 로컬 점검만(젯슨 접속 없음) — 오프라인 검증용
    *) echo "알 수 없는 인자 $1"; exit 2;;
  esac; shift
done
case "$STAGE" in ""|off|fas|soc) ;; *) echo "--stage 는 off|fas|soc"; exit 2;; esac
[ $((APPLY + ROLLBACK + (${#STAGE} > 0 ? 1 : 0))) -le 1 ] || { echo "--apply / --stage / --rollback 은 한 번에 하나만"; exit 2; }
[ $RESTART -eq 0 ] || [ $APPLY -eq 1 ] || { echo "--restart-console 은 --apply 와 함께만"; exit 2; }
[ $LOCAL_ONLY -eq 0 ] || [ $((APPLY + ROLLBACK + ${#STAGE})) -eq 0 ] || { echo "--local-only 는 점검(dry-run) 전용"; exit 2; }

control_procs() { $SSH "ps -eo pid,args | grep -E '[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py|[h]ang_stand.py' || true"; }

# ------------------------------------------------------------------ --stage / --rollback (파일 한 줄만 바꾼다)
if [ -n "$STAGE" ] || [ $ROLLBACK -eq 1 ]; then
  $SSH true || { echo "젯슨 접속 불가"; exit 1; }
  CTRL=$(control_procs)
  if [ -n "$CTRL" ]; then
    if [ $ROLLBACK -eq 1 ] || [ "$STAGE" == "off" ]; then
      # 복귀는 언제든 허용 — 파일만 비운다. 실행 중 stage8 은 기동 때 옵션을 유지하므로, 즉시 종전 경로가 필요하면
      # 회수 순서(줄 인장 → 콘솔 '스택 종료' → 그 다음에만 release_all) 뒤 정상 순서로 재기동.
      echo "⚠ 제어 프로세스 실행 중 — 파일은 지금 비우지만 실행 중 stage8 은 재기동 전까지 옵션 유지:"; echo "$CTRL"
    else
      echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 콘솔 '스택 종료' → 그 다음에만 release_all) 뒤 다시 실행:"; echo "$CTRL"; exit 1
    fi
  fi
  CUR=$($SSH "head -1 $R/$OPTS 2>/dev/null | cut -d'#' -f1 | tr -d ' \t\r'" || true)
  ENVV=$($SSH "systemctl show operator-console -p Environment --value 2>/dev/null" || true)
  if echo "$ENVV" | grep -q STAGE8_LATENCY_OPTS; then
    echo "⚠ 콘솔 서비스 환경변수 STAGE8_LATENCY_OPTS 가 설정돼 있어 파일보다 우선한다: $ENVV — 파일 변경이 무효. 중단"; exit 1
  fi
  echo "   현재 옵션 파일: '${CUR}'"
  if [ $ROLLBACK -eq 1 ] || [ "$STAGE" == "off" ]; then
    $SSH "mkdir -p $R/config && : > $R/$OPTS"
    echo "== 옵션 파일 비움(= 종전 경로). 다음 stage8 기동부터 적용 — 기동 후 latency_check --expect off 로 실로드 확인"
    if [ $ROLLBACK -eq 1 ]; then
      echo "== 코드 되돌리기가 필요할 때만(옵션 off 로 충분하면 불필요) — 제어 프로세스 없는 상태에서 젯슨에서 직접:"
      for f in $FILES; do echo "   [ -e $R/$f.bak_1006 ] && cp -p $R/$f.bak_1006 $R/$f"; done
      echo "   (콘솔 파일을 되돌렸으면 콘솔 재기동 필요. latency_check.py·테스트는 .bak 이 없으면 새 파일 — 남아도 무해)"
    fi
    exit 0
  fi
  if [ "$STAGE" == "fas" ]; then
    $SSH "mkdir -p $R/config && printf 'fas\n' > $R/$OPTS"
    echo "== 1단계: 옵션 파일 = 'fas' (--feedback-after-send 만). 다음 stage8 기동부터 적용"
    echo "   다음: 콘솔에서 정상 순서로 스택 기동 → 들어 올린 무부하 홀드에서 latency_check --expect fas (기준 측정과 같은 시간)"
    exit 0
  fi
  # soc = 2단계 — 1단계 결과 게이트
  [ "$CUR" == "fas" ] || { echo "2단계 거부: 현재 옵션 파일이 'fas' 가 아님('${CUR}') — 1단계부터"; exit 1; }
  GATE=$($SSH "python3 - <<'PY'
import json, os
p = os.path.expanduser('~/logs/latency_check/last_fas.json')
o = '$R/$OPTS'
try:
    d = json.load(open(p))
except Exception as e:
    print('NO', 'last_fas.json 없음/읽기 실패', e); raise SystemExit
lo = (d.get('summary') or {}).get('latency_options') or {}
ok = (d.get('verdict') == 'PASS' and (d.get('summary') or {}).get('expect') == 'fas'
      and lo.get('feedback_after_send') is True and lo.get('send_on_command') is False)
newer = d.get('wall_time', 0) > os.path.getmtime(o)
print('OK' if ok and newer else 'NO', 'verdict=%s expect=%s fas=%s soc=%s 측정후설정=%s criteria=%s' % (
    d.get('verdict'), (d.get('summary') or {}).get('expect'), lo.get('feedback_after_send'), lo.get('send_on_command'),
    newer, d.get('criteria_version')))
PY")
  echo "   1단계 결과: $GATE"
  [ "${GATE%% *}" == "OK" ] || { echo "2단계 거부: fas 판정 PASS 결과(옵션 'fas' 설정 이후 측정)가 없다"; exit 1; }
  $SSH "printf 'fas,soc\n' > $R/$OPTS"
  echo "== 2단계: 옵션 파일 = 'fas,soc'. 다음 stage8 기동부터 적용 — 측정은 latency_check --expect fas,soc"
  exit 0
fi

# ------------------------------------------------------------------ 점검(dry-run) / --apply
echo "== 로컬 매니페스트 대조 ($MANIFEST — 이 파일들이 오프라인 검증본과 같은지)"
LC_ALL=C md5sum -c --quiet $MANIFEST || { echo "로컬 파일이 매니페스트와 다름 — 검증본 아님"; exit 1; }
echo "   일치 $(wc -l < $MANIFEST)/$(echo $FILES | wc -w)"
[ "$(wc -l < $MANIFEST)" -eq "$(echo $FILES | wc -w)" ] || { echo "매니페스트 항목 수가 배포 파일 수와 다름"; exit 1; }
echo "== 로컬 selftest·테스트 (배포 전 게이트)"
python3 jetson/scripts/latency_check.py --selftest | tail -1 | grep -q "ALL PASS" || { echo "latency_check selftest 실패"; exit 1; }
timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1 | grep -q "ALL PASS" || { echo "콘솔 selftest 실패"; exit 1; }
python3 -m unittest discover -s tests -p 'test_*latency*.py' 2>&1 | tail -1 | grep -q "^OK" || { echo "단위 테스트 실패"; exit 1; }
python3 -c "import sys; sys.path.insert(0,'.'); from jetson.scripts import stage8_12axis_mit_control_node as s; a,_=s.build_arg_parser().parse_known_args(['--feedback-after-send']); print('stage8 import·인자 OK')" | grep -q OK || { echo "stage8 import 실패"; exit 1; }
echo "   latency_check·콘솔 selftest ALL PASS · 단위 테스트 OK · stage8 import OK"
[ $LOCAL_ONLY -eq 0 ] || { echo "== --local-only 끝 (젯슨 단계 생략)"; exit 0; }

echo "== 젯슨 현재 상태 (읽기만)"
$SSH true || { echo "젯슨 접속 불가"; exit 1; }
for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f 2>/dev/null" | cut -c1-32 || true)
  echo "   $f  젯슨 ${r:0:8}${r:+ }${r:-없음 } → 새 ${l:0:8} $([ "$l" == "$r" ] && echo '(같음)')"
done
echo "   옵션 파일 $OPTS: '$($SSH "head -1 $R/$OPTS 2>/dev/null" || true)' (비었거나 없음 = 종전 경로)"
echo "   콘솔이 옵션 파일을 읽는 판인가: $($SSH "grep -c STAGE8_LATENCY_FILE $R/jetson/scripts/operator_console.py 2>/dev/null" || echo 0) (0 이면 --apply --restart-console 필요)"
echo "   콘솔 서비스 환경: $($SSH "systemctl show operator-console -p Environment --value 2>/dev/null" || true) (STAGE8_LATENCY_OPTS 가 있으면 파일보다 우선 — 없어야 함)"
CTRL=$(control_procs)
echo "   제어 프로세스: ${CTRL:-없음}"
echo "   최근 판정 결과: $($SSH "ls -1t ~/logs/latency_check/*.json 2>/dev/null | head -3 | tr '\n' ' '" || true)"
[ $APPLY -eq 1 ] || { echo "== dry-run 끝 (승인 뒤 --apply [--restart-console], 그 다음 --stage fas)"; exit 0; }

[ -z "$CTRL" ] || { echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 스택 종료) 뒤 다시 실행"; exit 1; }
echo "== 백업(.bak_1006) → 복사"
for f in $FILES; do
  $SSH "mkdir -p $R/$(dirname $f); [ ! -e $R/$f ] || [ -e $R/$f.bak_1006 ] || cp -p $R/$f $R/$f.bak_1006"
  $SCP "$f" "$J:$R/$f"
done
n=0; for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f" | cut -c1-32)
  [ "$l" == "$r" ] || { echo "md5 불일치 $f — 되돌리기 필요(--rollback 안내 참조)"; exit 1; }; n=$((n+1))
done
echo "   md5 일치 $n/$n"
echo "== 젯슨 selftest·테스트"
$SSH "cd $R && source /opt/ros/humble/setup.bash && python3 jetson/scripts/latency_check.py --selftest | tail -1"
$SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1"
$SSH "cd $R && source /opt/ros/humble/setup.bash && python3 -m unittest discover -s tests -p 'test_*latency*.py' 2>&1 | tail -1"
if [ $RESTART -eq 1 ]; then
  KM=$($SSH "systemctl show operator-console -p KillMode --value")
  SPIDS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter' | awk '{print \$1}' | sort | tr '\n' ' '")
  RPIDS=$($SSH "cat ~/run/*.pid 2>/dev/null | sort | tr '\n' ' '")
  for p in $SPIDS; do echo " $RPIDS " | grep -q " $p " || { echo "센서 pid $p 가 PID파일에 없음(콘솔 추적 밖) — 재기동 보류"; exit 1; }; done
  [ "$KM" == "process" ] || { echo "KillMode=$KM — 콘솔 재기동이 센서를 함께 죽일 수 있어 보류"; exit 1; }
  echo "== 콘솔 재기동 (KillMode=process)"
  $SSH "echo "$JETSON_PW" | sudo -S systemctl restart operator-console && sleep 4 && systemctl is-active operator-console"
fi
echo "== 배포 완료 — 옵션 파일은 바꾸지 않았다(현재 '$($SSH "head -1 $R/$OPTS 2>/dev/null" || true)'). 다음: 기준 측정(off) → --stage fas"
