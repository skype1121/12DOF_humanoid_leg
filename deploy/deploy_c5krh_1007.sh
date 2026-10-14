#!/bin/bash
# 1007 콘솔 정책 선택지 추가 배포 — stand_c5krh_g21a_g24ra (KE 와 같은 두 정책·같은 감지기, 조기 복귀 유지 시간만 0.2 → 0.7 s = "imuH").
# 승윤님 1007 19:50 지시("콘솔에 정책 하나 추가 … 바로 젯슨에 콘솔배포")에 따른 배포. 실물 미검증·승격 아님(기존 선택지 stand_c5kr 는 그대로 남는다).
# 보내는 것은 **젯슨에 지금 있는 판(1005 배포본 = 커밋 e7710c63)에 키 등록만 더한** 브리지·콘솔 두 파일(deploy/cand/c5krh_1007/, 패치 c5krh_1007.patch).
# 저장소 HEAD 의 브리지·콘솔(W3·감시·지연 옵션 등 젯슨 미배포 변경 포함)은 보내지 않는다. 러너·c5_switch·정책 파일·하드웨어맵은 건드리지 않는다.
#   (인자 없음)        : 점검만(dry-run) — 후보 md5·로컬 게이트 기록·젯슨 현재 md5·프로세스·콘솔 상태.
#   --apply            : 젯슨 파일이 기대한 1005 판인지 확인 → 제어 프로세스·활성 세션 없음 확인 → 백업(.bak_1007) → 복사 → md5 → 젯슨 selftest 3 종(실패 시 자동 복원)
#   --restart-console  : (--apply 와 함께) 콘솔 서비스 재기동 — 드롭다운 반영에 필요.
# 되돌리기: 젯슨에서 두 파일의 .bak_1007 을 원래 이름으로 복사하고 콘솔 재기동(이 스크립트 --rollback [--restart-console]).
set -euo pipefail
cd "$(dirname "$0")/.."
J=mama@172.16.35.89; R=/home/mama/humanoid_repo
SSH="sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 $J"
SCP="sshpass -p "$JETSON_PW" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8"
SRC=deploy/cand/c5krh_1007; KEY=stand_c5krh_g21a_g24ra
T_BR=jetson/scripts/rl_bridge_node.py; T_CO=jetson/scripts/operator_console.py
OLD_BR=cadbc98516acf9dc17ffccc1662e39c7; OLD_CO=147f322ff63b8336d61d0c958c54d769       # 젯슨 1005 배포본(deploy/c5kr_1005.md5)
KEEP="rl_walking/deploy/c5_switch.py:11cf9641c8d00bbd5898ae755e9af07e rl_walking/deploy/policy_runner_stage4.py:21b2416efe2108e507caeb363916062b jetson/scripts/console_state.py:446a7c2bf5df35f3fe5069a53e2f7aa4"
APPLY=0; RESTART=0; ROLLBACK=0
for a in "$@"; do case $a in --apply) APPLY=1;; --restart-console) RESTART=1;; --rollback) ROLLBACK=1;; *) echo "알 수 없는 인자 $a"; exit 2;; esac; done
rmd5() { $SSH "md5sum $R/$1 2>/dev/null" | cut -c1-32 || true; }

echo "== 후보 매니페스트 (deploy/c5krh_1007.md5) · 로컬 게이트 기록"
(cd $SRC && LC_ALL=C md5sum -c --quiet ../../c5krh_1007.md5) || { echo "후보 파일이 매니페스트와 다름"; exit 1; }
NEW_BR=$(md5sum $SRC/rl_bridge_node.py | cut -c1-32); NEW_CO=$(md5sum $SRC/operator_console.py | cut -c1-32)
grep -q "$NEW_BR" $SRC/GATE_PASS.txt && grep -q "$NEW_CO" $SRC/GATE_PASS.txt && grep -q '^bridge: selftest: ALL PASS' $SRC/GATE_PASS.txt && grep -q '^runner: selftest: ALL PASS' $SRC/GATE_PASS.txt \
  && grep -q '^console: console selftest: ALL PASS' $SRC/GATE_PASS.txt || { echo "로컬 게이트 기록($SRC/GATE_PASS.txt)이 이 후보와 맞지 않음"; exit 1; }
echo "   후보 브리지 ${NEW_BR:0:8} · 콘솔 ${NEW_CO:0:8} — 게이트 기록 일치(러너·브리지·콘솔 selftest ALL PASS)"

echo "== 젯슨 현재 상태"
$SSH true || { echo "젯슨 접속 불가"; exit 1; }
CUR_BR=$(rmd5 $T_BR); CUR_CO=$(rmd5 $T_CO)
echo "   $T_BR  젯슨 ${CUR_BR:0:8} (1005 판 ${OLD_BR:0:8} / 새 ${NEW_BR:0:8})"
echo "   $T_CO  젯슨 ${CUR_CO:0:8} (1005 판 ${OLD_CO:0:8} / 새 ${NEW_CO:0:8})"
for kv in $KEEP; do p=${kv%%:*}; m=${kv##*:}; c=$(rmd5 $p); echo "   $p  젯슨 ${c:0:8} (기대 ${m:0:8}) $([ "$c" == "$m" ] && echo 같음 || echo '★다름')"; [ "$c" == "$m" ] || KEEP_BAD=1; done
CTRL=$($SSH "ps -eo pid,args | grep -E '[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py' | grep -v 'bash -c' || true")
SENS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter' | grep -v 'bash -c' || true")
STATE=$($SSH "curl -s -m 5 http://127.0.0.1:8800/api/state | python3 -c \"import json,sys; d=json.load(sys.stdin); print('session_active=%s bridge_live=%s armed=%s procs=%s phase=%s' % (d['session']['active'], d['bridge']['live'], d['bridge']['armed'], ','.join(sorted(d.get('procs', {}).keys())) or '-', d['trial']['phase']))\"" 2>/dev/null || echo "상태 읽기 실패")
echo "   제어 프로세스: ${CTRL:-없음}"
echo "   센서 프로세스: ${SENS:-없음}"
echo "   콘솔 상태: $STATE · 서비스 $($SSH 'systemctl is-active operator-console') · 선택지 $($SSH "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option>[a-z0-9_]*</option>' | sed 's/<[^>]*>//g' | tr '\n' ' '")"

restart_console() {
  local KM SPIDS RPIDS p
  KM=$($SSH "systemctl show operator-console -p KillMode --value")
  SPIDS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter' | grep -v 'bash -c' | awk '{print \$1}' | sort | tr '\n' ' '")
  RPIDS=$($SSH "cat ~/run/*.pid 2>/dev/null | sort | tr '\n' ' '")
  for p in $SPIDS; do echo " $RPIDS " | grep -q " $p " || { echo "센서 pid $p 가 PID파일에 없음(콘솔 추적 밖) — 재기동 보류"; return 1; }; done
  [ "$KM" == "process" ] || { echo "KillMode=$KM — 콘솔 재기동이 자식을 함께 죽일 수 있어 보류"; return 1; }
  echo "== 콘솔 재기동 (KillMode=process)"
  $SSH "echo "$JETSON_PW" | sudo -S systemctl restart operator-console 2>/dev/null && sleep 4 && systemctl is-active operator-console"
  OPTS=$($SSH "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option>[a-z0-9_]*</option>' | sed 's/<[^>]*>//g' | tr '\n' ' '")
  echo "   선택지: $OPTS"
}
rollback() {
  echo "== 복원(.bak_1007 → 원래 이름)"
  $SSH "cp -p $R/$T_BR.bak_1007 $R/$T_BR && cp -p $R/$T_CO.bak_1007 $R/$T_CO"
  [ "$(rmd5 $T_BR)" == "$OLD_BR" ] && [ "$(rmd5 $T_CO)" == "$OLD_CO" ] && echo "   복원 md5 일치(1005 판)" || { echo "복원 md5 불일치 — 수동 확인 필요"; exit 1; }
}
safe_now() {   # 제어 프로세스·활성 세션 없음
  [ -z "$CTRL" ] || { echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 스택 종료) 뒤 다시 실행"; return 1; }
  echo "$STATE" | grep -q 'session_active=False' && echo "$STATE" | grep -q 'bridge_live=False' && echo "$STATE" | grep -q 'armed=False' || { echo "콘솔에 활성 세션·브리지가 있음($STATE) — 보류"; return 1; }
}

if [ $ROLLBACK -eq 1 ]; then
  safe_now || exit 1
  $SSH "[ -e $R/$T_BR.bak_1007 ] && [ -e $R/$T_CO.bak_1007 ]" || { echo "백업 없음"; exit 1; }
  rollback; [ $RESTART -eq 1 ] && restart_console; exit 0
fi
[ $APPLY -eq 1 ] || { echo "== dry-run 끝 (--apply [--restart-console])"; exit 0; }

[ -z "${KEEP_BAD:-}" ] || { echo "러너·c5_switch·console_state 가 기대한 1005 판이 아님 — 이 패치의 전제가 깨짐, 중단"; exit 1; }
if [ "$CUR_BR" == "$NEW_BR" ] && [ "$CUR_CO" == "$NEW_CO" ]; then echo "== 이미 새 판이 올라가 있음 — 복사 생략"
else
  [ "$CUR_BR" == "$OLD_BR" ] && [ "$CUR_CO" == "$OLD_CO" ] || { echo "젯슨의 브리지·콘솔이 기대한 1005 판이 아님 — 패치 전제가 깨짐, 중단"; exit 1; }
  safe_now || exit 1
  echo "== 백업(.bak_1007) → 복사"
  $SSH "[ -e $R/$T_BR.bak_1007 ] || cp -p $R/$T_BR $R/$T_BR.bak_1007; [ -e $R/$T_CO.bak_1007 ] || cp -p $R/$T_CO $R/$T_CO.bak_1007"
  $SCP $SRC/rl_bridge_node.py "$J:$R/$T_BR"; $SCP $SRC/operator_console.py "$J:$R/$T_CO"
  [ "$(rmd5 $T_BR)" == "$NEW_BR" ] && [ "$(rmd5 $T_CO)" == "$NEW_CO" ] || { echo "md5 불일치 — 복원"; rollback; exit 1; }
  echo "   md5 일치 2/2"
fi
echo "== 젯슨 selftest (실패하면 자동 복원)"
RS=$($SSH "cd $R && source /opt/ros/humble/setup.bash && python3 rl_walking/deploy/policy_runner_stage4.py --selftest 2>&1 | tail -1"); echo "   러너: $RS"
BS=$($SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 900 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -E 'C5 전환|C5 kr|C5 krh|FAIL|selftest:' | cut -c1-260" || true); echo "$BS" | sed 's/^/   브리지: /'
CS=$($SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1"); echo "   콘솔: $CS"
if ! echo "$RS" | grep -q 'ALL PASS' || ! echo "$BS" | grep -q 'selftest: ALL PASS' || ! echo "$BS" | grep -q 'PASS\] C5 krh' || ! echo "$CS" | grep -q 'ALL PASS'; then
  echo "젯슨 selftest 실패 — 복원"; rollback; exit 1
fi
if [ $RESTART -eq 1 ]; then
  # 재기동 직전 상태 재확인(사용자가 그사이 조작을 시작했을 수 있다)
  CTRL=$($SSH "ps -eo pid,args | grep -E '[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py' | grep -v 'bash -c' || true")
  STATE=$($SSH "curl -s -m 5 http://127.0.0.1:8800/api/state | python3 -c \"import json,sys; d=json.load(sys.stdin); print('session_active=%s bridge_live=%s armed=%s' % (d['session']['active'], d['bridge']['live'], d['bridge']['armed']))\"" 2>/dev/null || echo "상태 읽기 실패")
  safe_now || { echo "파일은 올라갔으나 콘솔 재기동은 보류 — 조작이 끝난 뒤 --apply --restart-console 을 다시 실행"; exit 1; }
  restart_console || exit 1
  echo "$OPTS" | grep -q "$KEY" && echo "   새 선택지 $KEY 확인" || { echo "선택지에 $KEY 없음 — 확인 필요"; exit 1; }
fi
echo "== 배포 완료 — 콘솔 정책 선택지 $KEY (KE + 조기 복귀 유지 0.7 s, 실물 미검증·승격 아님). 되돌리기: $0 --rollback --restart-console"
