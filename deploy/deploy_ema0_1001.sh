#!/bin/bash
# 1001 아침 배포 — 관측 joint_vel_ema 계약 모드(v2 / ema0): 러너·브리지·콘솔 3파일 → 젯슨.
# **승윤님 승인 뒤에만 --apply 로 실행.** 기본은 점검만(dry-run): 젯슨 현재 md5·정책 실행 여부·바뀔 파일을 보여 준다.
#   --apply            : 제어 프로세스(브리지·stage8) 없음 확인 → 백업(.bak_1001) → 복사 → md5 대조 → 젯슨 selftest 3종
#   --restart-console  : (--apply 와 함께) 콘솔 서비스 재기동 — 드롭다운에 stand_crouch_g12_ema0 가 나타나려면 필요.
#                        센서·CAN 이 떠 있으면 재기동하지 않는다(콘솔이 띄운 프로세스는 콘솔이 내린다).
# 되돌리기: 각 파일의 .bak_1001 을 원래 이름으로 복사(콘솔은 재기동).
# 기본 동작 불변: 이름 그대로의 stand_crouch_g12 는 0929 과 같은 v2(실제 속도 EMA). ema0 은 별칭을 골랐을 때만.
set -euo pipefail
cd "$(dirname "$0")/.."
J=mama@172.16.35.89; R=/home/mama/humanoid_repo
SSH="sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 $J"
SCP="sshpass -p "$JETSON_PW" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8"
FILES="rl_walking/deploy/policy_runner_stage4.py jetson/scripts/rl_bridge_node.py jetson/scripts/operator_console.py"
APPLY=0; RESTART=0
for a in "$@"; do case $a in --apply) APPLY=1;; --restart-console) RESTART=1;; *) echo "알 수 없는 인자 $a"; exit 2;; esac; done

echo "== 로컬 selftest (배포 전 게이트)"
python3 rl_walking/deploy/policy_runner_stage4.py --selftest | tail -1 | grep -q "ALL PASS" || { echo "러너 selftest 실패"; exit 1; }
timeout 600 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | tail -1 | grep -q "ALL PASS" || { echo "브리지 selftest 실패"; exit 1; }
echo "   러너·브리지 selftest ALL PASS"

echo "== 젯슨 현재 상태"
for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f" | cut -c1-32)
  echo "   $f  젯슨 ${r:0:8} → 새 ${l:0:8} $([ "$l" == "$r" ] && echo '(같음)')"
done
# 제어 프로세스 확인 — pgrep 자기매칭 함정 회피([r] 패턴)
CTRL=$($SSH "ps -eo pid,args | grep -E '[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py' || true")
SENS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter' || true")
echo "   제어 프로세스: ${CTRL:-없음}"
echo "   센서 프로세스: ${SENS:-없음}"
[ $APPLY -eq 1 ] || { echo "== dry-run 끝 (--apply 로 실행)"; exit 0; }

[ -z "$CTRL" ] || { echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 스택 종료) 뒤 다시 실행"; exit 1; }
echo "== 백업(.bak_1001) → 복사"
$SSH "cd $R && for f in $FILES; do [ -e \$f.bak_1001 ] || cp -p \$f \$f.bak_1001; done"
for f in $FILES; do $SCP "$f" "$J:$R/$f"; done
for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f" | cut -c1-32)
  [ "$l" == "$r" ] || { echo "md5 불일치 $f — 되돌리기 필요"; exit 1; }
done
echo "   md5 일치 3/3"
echo "== 젯슨 selftest"
$SSH "cd $R && source /opt/ros/humble/setup.bash && python3 rl_walking/deploy/policy_runner_stage4.py --selftest | tail -2"
$SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 600 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -E '학습 계약|정책 500틱|selftest:' | cut -c1-300"
$SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1"
if [ $RESTART -eq 1 ]; then
  # 센서가 떠 있어도 재기동해도 되는 조건(1001 확인): KillMode=process(콘솔만 재시작, 자식 센서는 유지) +
  # 떠 있는 센서 PID 가 전부 ~/run/*.pid(sensors_up.sh 소유)에 있음(새 콘솔이 PID파일 소유로 인수·재기동 가능, 0919 수정).
  KM=$($SSH "systemctl show operator-console -p KillMode --value")
  SPIDS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter' | awk '{print \$1}' | sort | tr '\n' ' '")
  RPIDS=$($SSH "cat ~/run/*.pid 2>/dev/null | sort | tr '\n' ' '")
  for p in $SPIDS; do echo " $RPIDS " | grep -q " $p " || { echo "센서 pid $p 가 PID파일에 없음(콘솔 추적 밖) — 재기동 보류"; exit 1; }; done
  [ "$KM" == "process" ] || { echo "KillMode=$KM — 콘솔 재기동이 센서를 함께 죽일 수 있어 보류"; exit 1; }
  echo "== 콘솔 재기동 (KillMode=process, 센서 pid [$SPIDS] 는 PID파일 소유 → 유지)"
  $SSH "echo "$JETSON_PW" | sudo -S systemctl restart operator-console && sleep 4 && systemctl is-active operator-console"
  $SSH "curl -s -m 5 http://127.0.0.1:8800/api/state" | python3 -c "import json,sys; d=json.load(sys.stdin); print('   콘솔 신선도', d['fresh'], '· 센서 pid 유지:', d.get('ros'))"
  $SSH "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option>[a-z0-9_]*</option>' | tr '\n' ' '"; echo
fi
echo "== 배포 완료 — 콘솔 드롭다운: stand_crouch_g12(v2, 0929 동일) / stand_crouch_g12_ema0(학습 계약)"
