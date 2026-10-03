#!/bin/bash
# 1003 G21a-11300 진단 배포 — 러너·브리지·콘솔(밀기 기록 버튼·기본값 g12_ema0) + G21a exported·학습 env.yaml(+ G12 env.yaml: ema0 계약 확인용) → 젯슨.
# **승윤님 승인 뒤에만 --apply 로 실행.** 기본은 점검만(dry-run): 로컬 selftest·로컬 매니페스트 대조·젯슨 현재 md5·프로세스·맵 해시.
#   --apply            : 제어 프로세스(브리지·stage8) 없음 확인 → 백업(.bak_1003) → 복사 → md5 대조 → 젯슨 selftest 3종
#   --restart-console  : (--apply 와 함께) 콘솔 서비스 재기동 — 드롭다운 기본값 stand_crouch_g12_ema0·stand_crouch_g21a 반영에 필요.
# 하드웨어맵은 보내지 않는다(젯슨 맵 = 실물 영점의 유일한 원본). dry-run 에서 젯슨 맵을 deploy/hwmap_jetson_1003.json 로 **읽어 오기만** 한다.
# 되돌리기: 각 파일의 .bak_1003 을 원래 이름으로 복사(콘솔은 재기동). 새로 생긴 G21a 디렉토리는 남아도 무해(선택지에서 빼면 됨).
set -euo pipefail
cd "$(dirname "$0")/.."
J=mama@172.16.35.89; R=/home/mama/humanoid_repo
SSH="sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 $J"
SCP="sshpass -p "$JETSON_PW" scp -o StrictHostKeyChecking=no -o ConnectTimeout=8"
G21=logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a
G12=logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12
FILES="rl_walking/deploy/policy_runner_stage4.py jetson/scripts/rl_bridge_node.py jetson/scripts/operator_console.py jetson/scripts/console_state.py jetson/scripts/console_ui.html
$G21/exported/policy.pt $G21/exported/policy.onnx $G21/params/env.yaml $G12/params/env.yaml $G12/exported/policy.pt"
APPLY=0; RESTART=0
for a in "$@"; do case $a in --apply) APPLY=1;; --restart-console) RESTART=1;; *) echo "알 수 없는 인자 $a"; exit 2;; esac; done

echo "== 로컬 매니페스트 대조 (deploy/g21a_1003.md5 — 이 파일들이 오프라인 검증본과 같은지)"
LC_ALL=C md5sum -c --quiet deploy/g21a_1003.md5 || { echo "로컬 파일이 매니페스트와 다름 — 검증본 아님"; exit 1; }
echo "   일치 $(wc -l < deploy/g21a_1003.md5)/$(wc -l < deploy/g21a_1003.md5)"
echo "== 로컬 selftest (배포 전 게이트)"
python3 rl_walking/deploy/policy_runner_stage4.py --selftest | tail -1 | grep -q "ALL PASS" || { echo "러너 selftest 실패"; exit 1; }
timeout 900 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | tail -1 | grep -q "ALL PASS" || { echo "브리지 selftest 실패"; exit 1; }
timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1 | grep -q "ALL PASS" || { echo "콘솔 selftest 실패"; exit 1; }
echo "   러너·브리지·콘솔 selftest ALL PASS"

echo "== 젯슨 현재 상태"
$SSH true || { echo "젯슨 접속 불가"; exit 1; }
for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f 2>/dev/null" | cut -c1-32 || true)
  echo "   $f  젯슨 ${r:0:8}${r:+ }${r:-없음 } → 새 ${l:0:8} $([ "$l" == "$r" ] && echo '(같음)')"
done
$SCP "$J:$R/config/robot_12dof_hardware_map.json" deploy/hwmap_jetson_1003.json
echo "   젯슨 하드웨어맵 $(md5sum deploy/hwmap_jetson_1003.json | cut -c1-8) (로컬 config $(md5sum config/robot_12dof_hardware_map.json | cut -c1-8)) — 로컬 동기화는 확인 후 수동 cp"
CTRL=$($SSH "ps -eo pid,args | grep -E '[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py' || true")
SENS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter' || true")
echo "   제어 프로세스: ${CTRL:-없음}"
echo "   센서 프로세스: ${SENS:-없음}"
[ $APPLY -eq 1 ] || { echo "== dry-run 끝 (승인 뒤 --apply [--restart-console])"; exit 0; }

[ -z "$CTRL" ] || { echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 스택 종료) 뒤 다시 실행"; exit 1; }
echo "== 백업(.bak_1003) → 복사"
for f in $FILES; do
  $SSH "mkdir -p $R/$(dirname $f); [ ! -e $R/$f ] || [ -e $R/$f.bak_1003 ] || cp -p $R/$f $R/$f.bak_1003"
  $SCP "$f" "$J:$R/$f"
done
n=0; for f in $FILES; do
  l=$(md5sum "$f" | cut -c1-32); r=$($SSH "md5sum $R/$f" | cut -c1-32)
  [ "$l" == "$r" ] || { echo "md5 불일치 $f — 되돌리기 필요"; exit 1; }; n=$((n+1))
done
echo "   md5 일치 $n/$n"
echo "== 젯슨 selftest"
$SSH "cd $R && source /opt/ros/humble/setup.bash && python3 rl_walking/deploy/policy_runner_stage4.py --selftest | tail -1"
$SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 900 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | grep -E '관측 필터|학습 계약|정책 500틱|selftest:' | cut -c1-300"
$SSH "cd $R && source /opt/ros/humble/setup.bash && timeout 300 python3 jetson/scripts/operator_console.py --selftest 2>&1 | tail -1"
if [ $RESTART -eq 1 ]; then
  KM=$($SSH "systemctl show operator-console -p KillMode --value")
  SPIDS=$($SSH "ps -eo pid,args | grep -E '[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter' | awk '{print \$1}' | sort | tr '\n' ' '")
  RPIDS=$($SSH "cat ~/run/*.pid 2>/dev/null | sort | tr '\n' ' '")
  for p in $SPIDS; do echo " $RPIDS " | grep -q " $p " || { echo "센서 pid $p 가 PID파일에 없음(콘솔 추적 밖) — 재기동 보류"; exit 1; }; done
  [ "$KM" == "process" ] || { echo "KillMode=$KM — 콘솔 재기동이 센서를 함께 죽일 수 있어 보류"; exit 1; }
  echo "== 콘솔 재기동 (KillMode=process)"
  $SSH "echo "$JETSON_PW" | sudo -S systemctl restart operator-console && sleep 4 && systemctl is-active operator-console"
  $SSH "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option>[a-z0-9_]*</option>' | tr '\n' ' '"; echo
fi
echo "== 배포 완료 — 공식 교대: stand_crouch_g12_ema0 ↔ stand_crouch_g21a (stand_crouch_g12 v2 = 보조 대조군, A/B 에 섞지 않음)"
