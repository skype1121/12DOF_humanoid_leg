#!/bin/bash
# 1008 HEAD 동기화 배포 — 젯슨 저장소 사본(~/humanoid_repo)의 실행 코드·정책을 저장소 HEAD 와 맞춘다.
#   내용: 콘솔('정책 ON'·지연 옵션 읽기)·브리지(W3 걷기·KE↔W3 합성·감시·시간 제한 걷기 — 전부 실험 플래그, 기본 꺼짐)·stage8(지연 옵션, 기본 꺼짐)·
#         러너·c5_switch·w3_deploy·kew3_runner·w3_contingency·도구(w3_cmd·latency_check·robotctl)·오프라인 시험·W3S3d-15800 정책.
#   목록·md5·젯슨 현재 md5 는 deploy/head_1008.plan (python3 deploy/make_head_manifest.py 가 만든다). 표로 보려면 deploy/cand/head_1008/DIFF_LIST.md.
#
# **사용자(승윤님) 승인 뒤에만 --apply.** 이 스크립트는 파일 복사·오프라인 selftest·콘솔 서비스 재기동만 한다.
# CAN 송신·ARM·토크 인가·영점 변경·센서 기동·제어 프로세스 기동/종료는 하지 않는다(pkill 없음, 프로세스 종료 없음).
#
#   (인자 없음)        : 점검만(dry-run) — 로컬 계획서·원본 md5 대조 → 금지 파일 검사 → 젯슨 현재 md5 를 읽어 계획서의 '젯슨 현재 md5' 와 비교
#                        → 보낼 파일(바뀔 것·새로 생길 것)·이미 같은 것·예상 밖 목록, 디스크 여유, 제어 프로세스·콘솔 상태, ~/rl_calib 실행본 대조(읽기만)
#   --apply            : 예상 밖 0 · 제어 프로세스 없음 · 콘솔 유휴 · 디스크 여유 확인 → 백업(~/humanoid_repo_bak_1008/files, md5 검증) →
#                        임시 폴더로 전송·md5 검증 → 제자리 교체 → 젯슨 전체 md5 대조 → 젯슨 오프라인 selftest → **실패하면 자동 롤백**
#   --restart-console  : (--apply 또는 --rollback 과 함께, 또는 단독) 콘솔 서비스 재기동 — 별도 단계. 제어 프로세스(stage8·브리지·hang_stand 등)가
#                        살아 있거나 콘솔에 활성 세션이 있으면 거부. 새 콘솔 화면('정책 ON' 버튼)은 재기동 뒤에 보인다.
#   --rollback         : 백업에서 원래 파일을 되돌리고 새로 생긴 파일을 지운다(백업 매니페스트로 md5 검증). 콘솔까지 되돌리려면 --restart-console 를 함께.
#   --verify           : 복사 없이 젯슨 상태 = 매니페스트인지 확인하고 selftest 만 다시 돌린다(실패해도 롤백하지 않음).
#   --add-policy=logs/rsl_rl/<실험>/<run> : 정책 폴더 하나만 추가 전송(exported·params/env.yaml, md5 기록, 젯슨 추론 확인). 기존 파일은 덮어쓰지 않는다.
#   --remove-policy=<같은 경로>           : 그렇게 추가한 폴더만 되돌린다(기록된 md5 와 같을 때만 삭제).
#   --skip-local-gate  : 로컬 selftest 게이트 생략(정책 폴더 logs/ 가 없는 워크트리에서만 — 실제 배포에서는 쓰지 말 것).
#   --probe-selftest-time : (dry-run 과 함께) 지금 젯슨에 있는 브리지 selftest 를 한 번 돌려 시간을 잰다 — 새 selftest(약 3.3 배)가
#                        콘솔 준비 단계의 300 s 제한에 걸릴지 미리 가늠. 오프라인 실행이며 파일을 쓰지 않는다.
#
# 절대 보내지 않는 것(코드로 강제 — forbidden_path): config/(하드웨어맵·영점 stand_zero_deg·IMU 장착 보정·지연 옵션), jetson/calib_data/,
#   ~/rl_calib/*(level_pose 등 — 이 스크립트는 그 폴더를 읽기만 한다), .json·확장자 없는 파일 전부. 계획서에 그런 경로가 있으면 즉시 중단한다.
#
# 시험 모드: JETSON_FAKE_ROOT=<폴더> 를 주면 ssh 대신 그 폴더를 젯슨의 /home/mama 로 보고 로컬에서 실행한다(젯슨 접속 없음).
#   가짜 폴더는 deploy/cand/head_1008/make_fake_jetson.py 로 만든다. 왕복 시험 전체는 deploy/test_head_1008_local.sh.
#   JETSON_FAKE_PYTHONPATH=<site 폴더>(selftest 용 numpy·onnxruntime 또는 torch), JETSON_FAKE_PYTHON="python3 -s"(기본).
# 환경변수: HEAD1008_POLICY_ROOT=<logs/ 가 있는 폴더>(기본 = 저장소 루트), HEAD1008_HOST(기본 mama@172.16.35.89),
#   HEAD1008_ALLOW_LATENCY_OPTS=1(젯슨의 stage8 지연 옵션 파일이 켜져 있어도 selftest 를 실패로 보지 않는다 — 지연 단계를 일부러 켠 뒤에만).
#   시험 전용: HEAD1008_PLAN(다른 계획서 — 금지 파일 검사 시험), HEAD1008_MIN_FREE_KB(디스크 여유 문턱 — 부족 거부 시험).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=head_1008
PLAN=${HEAD1008_PLAN:-deploy/$NAME.plan}
MANIFEST=deploy/$NAME.md5
ACCEPT=deploy/$NAME.accept          # (선택) "md5  경로" — 계획서의 젯슨 md5 와 달라도 덮어써도 된다고 사람이 확인한 구판
POLICY_ROOT=${HEAD1008_POLICY_ROOT:-.}
J=${HEAD1008_HOST:-mama@172.16.35.89}
SSH="sshpass -p "$JETSON_PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 $J"
FAKE=${JETSON_FAKE_ROOT:-}
if [ -n "$FAKE" ]; then RH=$(cd "$FAKE" && pwd); PY=${JETSON_FAKE_PYTHON:-python3 -s}; ROS_SETUP=""
else RH=/home/mama; PY=python3; ROS_SETUP=/opt/ros/humble/setup.bash; fi
R=$RH/humanoid_repo; BAK=$RH/humanoid_repo_bak_1008; CALIB=$RH/rl_calib
MIN_FREE_KB=${HEAD1008_MIN_FREE_KB:-204800}   # 복사에 필요한 양 말고도 남겨 둘 여유(기본 200 MB)
# 콘솔의 준비 단계 '브리지 셀프테스트' 시간 제한(console_state.py: timeout = 300 if action in ("boot_pose", "selftest", …))과 배포 게이트의 허용치(80 %)
CONSOLE_SELFTEST_TIMEOUT_S=300; CONSOLE_SELFTEST_LIMIT_S=240
# selftest 에 쓰는 키(계획서의 정책과 맞아야 한다)
KEY_G21A=stand_crouch_g21a; KEY_KRH=stand_c5krh_g21a_g24ra; KEY_W3=stand_w3s3d_15800; KEY_KEW3=stand_kew3_g21a_g24ra_w3s3d15800
# 1008 둘째 합성 키(첫 합성 키 + KE 감지기 갇힘 해소 = 시뮬 합성 평가 구성) — 콘솔 선택지에 올라가므로 젯슨 selftest 도 같은 인자로 돌린다(정책 파일은 첫 합성 키와 같다)
KEY_KEW3Q=stand_kew3q_g21a_g24ra_w3s3d15800
# 제어(=CAN 을 쓰거나 모터에 명령하는) 프로세스 — 하나라도 살아 있으면 복사·재기동·롤백 거부
CTRL_RE='[r]l_bridge_node.py|[s]tage8_12axis_mit_control_node.py|[h]ang_stand.py|[p]osture_probe.py|[g]round_stand.py|[s]tatic_march.py|[p]ose_hold.py|[h]ang_leg_swing.py|[z]ero_anchor.py|[b]oot_pose_check.py|[r]elease_all.py|[p]lay_probe.py'
SENS_RE='[i]mu_node|[f]oot_force_adapter|[p]ressure_serial_node|[i]mu_adapter'
CALIB_NAMES="boot_pose_check.py can_up.sh hang_stand.py release_all.py sensors_up.sh stage8_cmd.py stack_down.sh ping_all.py imu_adapter_node.py foot_force_adapter_node.py zero_torque_read.py ak_mit_command.py ak_mit_decoder.py"

APPLY=0; RESTART=0; ROLLBACK=0; VERIFY=0; SKIP_GATE=0; PROBE=0; ADDPOL=""; RMPOL=""
for a in "$@"; do case $a in
  --apply) APPLY=1;; --restart-console) RESTART=1;; --rollback) ROLLBACK=1;; --verify) VERIFY=1;; --skip-local-gate) SKIP_GATE=1;;
  --probe-selftest-time) PROBE=1;;
  --add-policy=*) ADDPOL=${a#*=}; SKIP_GATE=1;;      # 정책 폴더만(코드가 아니므로 로컬 코드 게이트는 돌리지 않는다)
  --remove-policy=*) RMPOL=${a#*=}; SKIP_GATE=1;;
  -h|--help) sed -n '2,32p' "$0"; exit 0;;
  *) echo "알 수 없는 인자 $a"; exit 2;; esac; done
[ $((APPLY + ROLLBACK + VERIFY)) -le 1 ] || { echo "--apply / --rollback / --verify 는 한 번에 하나만"; exit 2; }
[ -z "$ADDPOL$RMPOL" ] || [ $((APPLY + ROLLBACK + VERIFY + RESTART)) -eq 0 ] || { echo "--add-policy / --remove-policy 는 단독으로만"; exit 2; }
# 로컬 원본을 쓰는 동작 = dry-run(적용 예행)과 --apply 뿐이다. 롤백·확인·재기동·정책 폴더 모드는 젯슨 쪽 기록(백업 폴더·계획서의 md5)만 쓴다 →
# 그 사이 로컬 트리가 바뀌었어도(다른 작업의 수정·병합) 막지 않고, 로컬 selftest 게이트도 돌리지 않는다. **롤백이 로컬 사정으로 막히면 안 된다.**
NEED_LOCAL=1
if [ $((ROLLBACK + VERIFY)) -gt 0 ] || [ -n "$ADDPOL$RMPOL" ] || { [ $RESTART -eq 1 ] && [ $APPLY -eq 0 ]; }; then NEED_LOCAL=0; fi

# ------------------------------------------------------------------ 원격(또는 가짜) 실행
rsh_in() {   # $1 = 명령 문자열, stdin 통과
  if [ -n "$FAKE" ]; then
    env -i HOME="$RH" PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${JETSON_FAKE_PYTHONPATH:-}" bash -c "$1"
  else
    $SSH "$1"
  fi
}
rsh() { rsh_in "$1" </dev/null; }

# ------------------------------------------------------------------ 금지 파일 규칙 (make_head_manifest.py 의 forbidden_reason 과 같은 규칙 — 따로 검사)
forbidden_path() {   # $1 = 경로 → 금지면 사유 출력
  local p="$1"
  [[ "$p" =~ ^[A-Za-z0-9_./-]+$ ]] || { echo "경로 형식(허용 밖 문자)"; return; }
  if [[ "$p" == /* || "$p" == *..* ]]; then echo "경로 형식(절대 경로·..)"; return; fi
  case "$p" in reports/night_1007/jetson_parity.py|reports/night_1007/jetson/parity_ref.npz) return;; esac   # 러너 동치 확인 도구·기준(명시 허용 2 개)
  case "$p" in
    config/*) echo "config/ — 하드웨어맵·영점·보정·지연 옵션(젯슨이 정본)"; return;;
    jetson/calib_data/*) echo "jetson/calib_data/ — 보정 자료"; return;;
  esac
  case "$p" in
    jetson/scripts/*|rl_walking/*|robot_runtime/*|protocols/*|tests/*|logs/rsl_rl/*|deploy/*) ;;
    *) echo "허용 목록 밖 폴더"; return;;
  esac
  if [[ "$p" == logs/* ]]; then
    [[ "$p" =~ ^logs/rsl_rl/[A-Za-z0-9_]+/[A-Za-z0-9_.-]+/(exported/(policy\.pt|policy\.onnx|SOURCE_CHECKPOINT)|params/env\.yaml)$ ]] \
      || echo "logs/ 아래는 정책 exported·params/env.yaml 만"
    return
  fi
  case "$p" in
    *.py|*.sh|*.html|*.md|*.ref|*.md5|*.VERSION) ;;
    *) echo "허용 확장자 밖(.json·.yaml·확장자 없는 파일 금지 — 맵·영점·보정 차단)"; return;;
  esac
  case "$p" in *.py|*.sh) return;; esac
  local low=${p,,}
  case "$low" in
    *hardware_map*|*level_pose*|*imu_mount_calib*|*pose_journal*|*stand_snapshot*|*stand_zero*|*foot_force_calib*|*latency_opts*|*rl_calib*)
      echo "하드웨어맵·영점·보정 류 이름";;
  esac
}

# ------------------------------------------------------------------ 계획서 읽기
[ -f "$PLAN" ] && [ -f "$MANIFEST" ] || { echo "계획서 없음($PLAN) — python3 deploy/make_head_manifest.py 먼저"; exit 1; }
P_PATH=(); declare -A P_ACT P_GRP P_ROOT P_MD5 P_SIZE P_JET
while IFS=$'\t' read -r p act grp root md5 size jet _note; do
  [[ -z "$p" || "$p" == \#* ]] && continue
  P_PATH+=("$p"); P_ACT[$p]=$act; P_GRP[$p]=$grp; P_ROOT[$p]=$root; P_MD5[$p]=$md5; P_SIZE[$p]=$size; P_JET[$p]=$jet
done < "$PLAN"
[ ${#P_PATH[@]} -gt 0 ] || { echo "계획서가 비어 있음"; exit 1; }
src_of() { if [ "${P_ROOT[$1]}" == policy ]; then echo "$POLICY_ROOT/$1"; else echo "$1"; fi; }
declare -A ACC
if [ -f "$ACCEPT" ]; then while read -r m p; do [[ -z "${m:-}" || "$m" == \#* ]] && continue; ACC["$p"]="${ACC[$p]:-} $m"; done < "$ACCEPT"; fi

echo "== [1] 금지 파일 검사 (하드웨어맵·영점·보정·~/rl_calib — 계획서 ${#P_PATH[@]} 항목)"
nbad=0
for p in "${P_PATH[@]}"; do
  why=$(forbidden_path "$p"); [ -z "$why" ] || { echo "   ★ 금지: $p — $why"; nbad=$((nbad + 1)); }
  case "${P_ACT[$p]}" in send|verify) ;; *) echo "   ★ 알 수 없는 동작 ${P_ACT[$p]}: $p"; nbad=$((nbad + 1));; esac
done
[ $nbad -eq 0 ] || { echo "금지 파일이 전송 목록에 있음 — 중단(아무것도 하지 않았다)"; exit 1; }
case "$R" in */humanoid_repo) ;; *) echo "대상 폴더 이름이 humanoid_repo 가 아님($R) — 중단"; exit 1;; esac
echo "   통과 — 금지 0"

echo "== [2] 로컬 원본 ↔ 계획서 md5 ($PLAN)"
nbad=0; nsend=0; nver=0; send_bytes=0
for p in "${P_PATH[@]}"; do
  s=$(src_of "$p")
  if [ ! -f "$s" ]; then echo "   ★ 원본 없음: $s"; nbad=$((nbad + 1)); continue; fi
  m=$(md5sum "$s" | cut -c1-32)
  [ "$m" == "${P_MD5[$p]}" ] || { echo "   ★ md5 다름: $s (계획서 ${P_MD5[$p]:0:8} / 지금 ${m:0:8})"; nbad=$((nbad + 1)); }
  if [ "${P_ACT[$p]}" == send ]; then nsend=$((nsend + 1)); send_bytes=$((send_bytes + ${P_SIZE[$p]})); else nver=$((nver + 1)); fi
done
if [ $nbad -ne 0 ] && [ $NEED_LOCAL -eq 1 ]; then
  echo "로컬 파일이 계획서와 다름 — python3 deploy/make_head_manifest.py 로 다시 만든 뒤(그리고 로컬 왕복 시험 뒤) 실행. 정책 폴더가 다른 곳이면 HEAD1008_POLICY_ROOT"; exit 1
elif [ $nbad -ne 0 ]; then
  echo "   ※ 로컬 파일 $nbad 개가 계획서와 다르다 — 이 동작(롤백·확인·재기동·정책 폴더)은 로컬 원본을 쓰지 않으므로 계속한다(젯슨 대조는 계획서의 md5 로)"
else
  echo "   일치 ${#P_PATH[@]}/${#P_PATH[@]} — 전송 대상 $nsend 개($((send_bytes / 1024)) KB) · 확인만 $nver 개 · 매니페스트 $(md5sum $MANIFEST | cut -c1-8)"
fi
sed 's/^/   /' deploy/$NAME.VERSION | head -3

echo "== [3] 로컬 selftest 게이트 (CPU 만 — GPU 를 쓰지 않는다)"
local_gate() {
  local ok=1 out
  lg() { local name=$1 tok=$2; shift 2
    out=$(CUDA_VISIBLE_DEVICES="" timeout 900 "$@" 2>&1) || true
    if echo "$out" | grep -qF -- "$tok" && ! echo "$out" | grep -q '^\[FAIL\]'; then echo "   PASS $name"; else echo "   ★FAIL $name: $(echo "$out" | tail -1 | cut -c1-160)"; ok=0; fi; }
  lg "러너" "selftest: ALL PASS" python3 rl_walking/deploy/policy_runner_stage4.py --selftest
  lg "c5_switch" "C5 selftest PASS" python3 rl_walking/deploy/c5_switch.py --selftest
  lg "kew3_runner" "ALL PASS" python3 rl_walking/deploy/kew3_runner.py --selftest
  lg "w3_cmd" "[PASS] w3_cmd" python3 jetson/scripts/w3_cmd.py --selftest
  lg "latency_check" "selftest: ALL PASS" python3 jetson/scripts/latency_check.py --selftest
  lg "콘솔" "console selftest: ALL PASS" python3 jetson/scripts/operator_console.py --selftest
  lg "브리지(기본)" "selftest: ALL PASS" python3 jetson/scripts/rl_bridge_node.py --selftest
  lg "브리지(합성 키)" "selftest: ALL PASS" python3 jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint $KEY_KEW3
  lg "브리지(둘째 합성 키 — 갇힘 해소)" "selftest: ALL PASS" python3 jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint $KEY_KEW3Q
  lg "W3 노드(git 기준 커밋 비교 포함)" "w3 node selftest: ALL PASS" python3 jetson/scripts/selftest_w3_bridge_node.py
  [ $ok -eq 1 ]
}
if [ $NEED_LOCAL -eq 0 ]; then echo "   생략 — 이 동작(롤백·확인·재기동·정책 폴더)은 로컬 원본을 쓰지 않는다"
elif [ $SKIP_GATE -eq 1 ]; then echo "   생략(--skip-local-gate) — 젯슨(또는 가짜 트리) selftest 로만 판정"
elif [ ! -d logs/rsl_rl ]; then
  echo "   ★ 이 폴더에는 정책(logs/rsl_rl)이 없어 로컬 게이트를 돌릴 수 없다 — 메인 체크아웃에서 실행하거나 --skip-local-gate"
  [ $APPLY -eq 0 ] || exit 1
else local_gate || { echo "로컬 selftest 실패 — 중단(젯슨에는 손대지 않았다)"; exit 1; }; fi

# ------------------------------------------------------------------ 젯슨 읽기
echo "== [4] 젯슨 현재 상태 (읽기만) — ${FAKE:+가짜 트리 }$([ -n "$FAKE" ] && echo "$RH" || echo "$J:$RH")"
if [ -z "$FAKE" ]; then command -v sshpass >/dev/null || { echo "sshpass 없음"; exit 1; }; fi
rsh true || { echo "젯슨 접속 불가"; exit 1; }
rsh "[ -f '$R/jetson/scripts/operator_console.py' ] && [ -f '$R/jetson/scripts/rl_bridge_node.py' ]" || { echo "$R 가 저장소 사본처럼 보이지 않음 — 중단"; exit 1; }

declare -A CUR
read_remote_md5() {
  local line m f
  CUR=()
  while read -r m f; do [ -n "${f:-}" ] && CUR["$f"]=$m; done < <(printf '%s\n' "${P_PATH[@]}" | rsh_in "cd '$R' && while IFS= read -r f; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo \"-  \$f\"; fi; done")
  for f in "${P_PATH[@]}"; do [ -n "${CUR[$f]:-}" ] || { echo "젯슨 md5 를 읽지 못함: $f"; exit 1; }; done
}
read_remote_md5
L_SAME=(); L_REPLACE=(); L_CREATE=(); L_BAD=()
classify() {
  local p c n e
  L_SAME=(); L_REPLACE=(); L_CREATE=(); L_BAD=()
  for p in "${P_PATH[@]}"; do
    c=${CUR[$p]}; n=${P_MD5[$p]}; e=${P_JET[$p]}
    if [ "$c" == "$n" ]; then L_SAME+=("$p")
    elif [ "${P_ACT[$p]}" == verify ]; then L_BAD+=("$p|확인 대상인데 젯슨 ${c:0:8} ≠ 기대 ${n:0:8}")
    elif [ "$c" == "-" ] && [ "$e" == "-" ]; then L_CREATE+=("$p")
    elif [ "$c" == "-" ]; then L_BAD+=("$p|인벤토리에는 있었는데(${e:0:8}) 지금 젯슨에 없음")
    elif [ "$c" == "$e" ] || [[ " ${ACC[$p]:-} " == *" $c "* ]]; then L_REPLACE+=("$p")
    else L_BAD+=("$p|젯슨 ${c:0:8} — 인벤토리 ${e:0:8}·새 ${n:0:8} 어느 쪽도 아님"); fi
  done
}
classify
echo "   이미 새 판과 같음 ${#L_SAME[@]} · 바뀔 파일(백업 뒤 덮어씀) ${#L_REPLACE[@]} · 새로 생길 파일 ${#L_CREATE[@]} · ★예상 밖 ${#L_BAD[@]}"
for p in "${L_REPLACE[@]}"; do echo "     바뀜  [${P_GRP[$p]}] $p  ${CUR[$p]:0:8} → ${P_MD5[$p]:0:8}"; done
for p in "${L_CREATE[@]}"; do echo "     새로  [${P_GRP[$p]}] $p  → ${P_MD5[$p]:0:8} (${P_SIZE[$p]} B)"; done
for x in "${L_BAD[@]}"; do echo "     ★예상 밖 ${x%%|*} — ${x#*|}"; done

probe() {   # 제어·센서 프로세스, 콘솔 상태, 서비스, 디스크, 파이썬 환경
  local PSTXT STATE_JSON SYSD
  if [ -n "$FAKE" ]; then
    PSTXT=$(cat "$RH/.fake/ps.txt" 2>/dev/null || true); STATE_JSON=$(cat "$RH/.fake/state.json" 2>/dev/null || true)
    SYSD=$(cat "$RH/.fake/systemd.txt" 2>/dev/null || true)
  else
    PSTXT=$(rsh "ps -eo pid,args")
    STATE_JSON=$(rsh "curl -s -m 5 http://127.0.0.1:8800/api/state" || true)
    SYSD=$(rsh "systemctl show operator-console -p KillMode -p ActiveState -p MainPID" || true)
  fi
  CTRL=$(echo "$PSTXT" | grep -E "$CTRL_RE" | grep -v 'bash -c' || true)
  SENS=$(echo "$PSTXT" | grep -E "$SENS_RE" | grep -v 'bash -c' || true)
  STATE=$(echo "$STATE_JSON" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print('session_active=%s bridge_live=%s armed=%s procs=%s phase=%s' % (d['session']['active'], d['bridge']['live'], d['bridge']['armed'],
          ','.join(sorted(d.get('procs', {}).keys())) or '-', d.get('trial', {}).get('phase')))
except Exception as e:
    print('상태 읽기 실패(%s)' % type(e).__name__)
" 2>/dev/null || echo "상태 읽기 실패")
  KM=$(echo "$SYSD" | sed -n 's/^KillMode=//p'); SVC=$(echo "$SYSD" | sed -n 's/^ActiveState=//p'); MAINPID=$(echo "$SYSD" | sed -n 's/^MainPID=//p')
  FREE_KB=$(rsh "df -Pk '$RH' | tail -1 | awk '{print \$4}'")
}
probe
echo "   제어 프로세스: ${CTRL:-없음}"
echo "   센서 프로세스: $( [ -n "$SENS" ] && echo "$SENS" | awk '{print $1}' | tr '\n' ' ' || echo 없음)"
echo "   콘솔: $STATE · 서비스 ${SVC:-?}(KillMode=${KM:-?}, PID ${MAINPID:-?})"
need_kb=$(( (2 * send_bytes) / 1024 + 1 ))
for p in "${L_REPLACE[@]}"; do need_kb=$((need_kb + ${P_SIZE[$p]} / 1024 + 1)); done
echo "   디스크 여유 $((FREE_KB / 1024)) MB — 필요 약 $((need_kb / 1024 + 1)) MB(임시 사본 + 백업) + 남길 여유 $((MIN_FREE_KB / 1024)) MB"
ENVLINE=$(rsh "cd '$R' && $PY -c \"
import sys
v = ['python ' + sys.version.split()[0]]
for m in ('numpy', 'onnxruntime', 'torch', 'can', 'rclpy'):
    try:
        mod = __import__(m); v.append(m + ' ' + str(getattr(mod, '__version__', '있음')))
    except Exception as e:
        v.append(m + ' 없음')
print(' · '.join(v))
\"" 2>/dev/null || echo "파이썬 환경 읽기 실패")
echo "   파이썬 환경(ROS 를 불러오지 않은 셸): $ENVLINE"
echo "   ~/rl_calib 실행본 ↔ 저장소 HEAD (읽기만 — 이 스크립트는 ~/rl_calib 를 바꾸지 않는다):"
CAL=$(rsh "cd '$CALIB' 2>/dev/null && for f in $CALIB_NAMES; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo \"-  \$f\"; fi; done" || true)
for f in $CALIB_NAMES; do
  rm=$(echo "$CAL" | awk -v f="$f" '$2 == f {print $1}'); lp=jetson/scripts/$f; [ -f "$lp" ] || lp=protocols/$f
  lm=$([ -f "$lp" ] && md5sum "$lp" | cut -c1-32 || echo "-")
  if [ -z "$rm" ] || [ "$rm" == "-" ]; then st="젯슨에 없음"; elif [ "$rm" == "$lm" ]; then st="같음"; else st="★다름(실행본 ${rm:0:8})"; fi
  echo "     $f  HEAD $lp ${lm:0:8} — $st"
done
# stage8 지연 옵션은 젯슨의 옵션 파일(config/stage8_latency_opts)이 없거나 비어 있고 서비스 환경변수가 없을 때 꺼진 채다. 이 스크립트는 그 파일을 만들지도 바꾸지도 않는다(읽기만).
LATF=$(rsh "f='$R/config/stage8_latency_opts'; if [ -s \"\$f\" ]; then head -1 \"\$f\" | cut -d'#' -f1 | tr -d ' \t\r'; fi" || true)
if [ -n "$FAKE" ]; then SVCENV=$(sed -n 's/^Environment=//p' "$RH/.fake/systemd.txt" 2>/dev/null || true)
else SVCENV=$(rsh "systemctl show operator-console -p Environment --value" || true); fi
if [ -z "$LATF" ] && ! echo "$SVCENV" | grep -q STAGE8_LATENCY_OPTS; then echo "   stage8 지연 옵션: 꺼짐(옵션 파일 없음/빈 파일 · 서비스 환경변수 없음) — 콘솔이 stage8 에 붙이는 추가 인자 0 개"
else echo "   ★ stage8 지연 옵션이 켜져 있다: 옵션 파일 '${LATF}' · 서비스 환경 '${SVCENV}' — 새 stage8 은 다음 기동부터 이 옵션으로 뜬다(§ deploy_lat_1006.sh --stage off 로 끈다)"; fi
BAKSTATE=$(rsh "cat '$BAK/STATE' 2>/dev/null | tail -1" || true)
echo "   백업 폴더 $BAK: ${BAKSTATE:-없음(첫 적용)}"
grep -q "timeout = $CONSOLE_SELFTEST_TIMEOUT_S if action in (\"boot_pose\", \"selftest\"" jetson/scripts/console_state.py \
  || echo "   ★ console_state.py 의 selftest 단계 제한이 ${CONSOLE_SELFTEST_TIMEOUT_S}s 가 아니다 — 이 스크립트의 CONSOLE_SELFTEST_TIMEOUT_S 를 맞출 것"
if [ $PROBE -eq 1 ]; then
  # 지금 젯슨에 있는 브리지의 selftest 를 그대로 한 번 돌려 시간을 잰다(오프라인, 파일을 쓰지 않는다). 새 판은 항목이 20 → 30 으로 늘어
  # 로컬(젯슨 모사 환경) 측정으로 약 3.3 배 걸린다(1.4 s → 4.7 s) — 콘솔 준비 단계 제한을 넘을지 미리 가늠한다. 실제 판정은 --apply 의 selftest 가 한다.
  echo "== [4b] 젯슨의 현재 브리지 selftest 소요 시간 (--probe-selftest-time)"
  PR=$(rsh "cd '$R' && { [ -z '$ROS_SETUP' ] || [ ! -f '$ROS_SETUP' ] || source '$ROS_SETUP'; }; export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=; t0=\$(date +%s); timeout 900 $PY jetson/scripts/rl_bridge_node.py --selftest --checkpoint $KEY_G21A 2>&1 | tail -1; echo \"SECS \$(( \$(date +%s) - t0 ))\"" || true)
  T_OLD=$(echo "$PR" | sed -n 's/^SECS //p'); echo "   $(echo "$PR" | head -1 | cut -c1-120) — ${T_OLD:-?} s"
  if [ -n "${T_OLD:-}" ]; then
    T_NEW=$(( T_OLD * 33 / 10 ))
    echo "   새 판 예상 약 ${T_NEW} s(× 3.3) — 콘솔 제한 ${CONSOLE_SELFTEST_TIMEOUT_S} s, 배포 게이트 허용 ${CONSOLE_SELFTEST_LIMIT_S} s $([ $T_NEW -gt $CONSOLE_SELFTEST_LIMIT_S ] && echo '★ 넘을 수 있다' || echo '(여유 있음)')"
  fi
fi

safe_now() {   # 제어 프로세스 없음 + 콘솔 유휴
  [ -z "$CTRL" ] || { echo "제어 프로세스 실행 중 — 회수 순서(줄 인장 → 콘솔 '스택 종료' → 그 다음에만 release_all) 뒤 다시 실행:"; echo "$CTRL"; return 1; }
  echo "$STATE" | grep -q 'session_active=False' && echo "$STATE" | grep -q 'bridge_live=False' && echo "$STATE" | grep -q 'armed=False' \
    || { echo "콘솔 상태가 유휴가 아니거나 읽지 못함($STATE) — 보류"; return 1; }
}

# ------------------------------------------------------------------ 젯슨 selftest (오프라인 — CAN·ROS 를 쓰지 않는다)
remote_selftest() {   # 반환 0 = 전부 통과. 결과 요약은 표준 출력, 전체 출력은 젯슨 $BAK/selftest_*.log
  local sent_py="" p res rc=0 line name r tok nf last
  for p in "${P_PATH[@]}"; do [[ "$p" == *.py ]] && sent_py="$sent_py $p"; done
  local pol_dirs; pol_dirs=$(for p in "${P_PATH[@]}"; do [[ "$p" == logs/*/exported/policy.onnx ]] && dirname "$p"; done | sort -u | tr '\n' ' ')
  res=$(rsh_in "bash -s -- '$R' '$BAK' '$PY' '$ROS_SETUP' '$KEY_G21A' '$KEY_KRH' '$KEY_W3' '$KEY_KEW3' '$sent_py' '$pol_dirs' '${HEAD1008_ALLOW_LATENCY_OPTS:-0}' '$KEY_KEW3Q'" <<'EOS'
R=$1; BAK=$2; PY=$3; ROS_SETUP=$4; KEY_G21A=$5; KEY_KRH=$6; KEY_W3=$7; KEY_KEW3=$8; SENT_PY=$9; POL_DIRS=${10}; ALLOW_LAT=${11}; KEY_KEW3Q=${12}
cd "$R" || { echo "ST|cd|90|0|0|0|저장소 폴더 없음"; exit 0; }
# 콘솔 서비스와 같은 환경(ROS 경로)에서 돌린다 — selftest 는 rclpy 를 초기화하지 않는다(브리지 노드 시험은 가짜 rclpy)
if [ -n "$ROS_SETUP" ] && [ -f "$ROS_SETUP" ]; then set +u; source "$ROS_SETUP"; fi
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
mkdir -p "$BAK" 2>/dev/null; LOG="$BAK/selftest_$(date +%Y%m%d_%H%M%S).log"; : > "$LOG" 2>/dev/null || LOG=/dev/null
echo "LOG|$LOG"
run() { name=$1; tmo=$2; token=$3; shift 3
  t0=$(date +%s); out=$(timeout "$tmo" "$@" 2>&1); rc=$?; secs=$(( $(date +%s) - t0 ))
  { echo "===== $name (rc=$rc, ${secs}s): $*"; echo "$out"; } >> "$LOG"
  nfail=$(printf '%s\n' "$out" | grep -c '^\[FAIL\]')
  tok=0; printf '%s\n' "$out" | grep -qF -- "$token" && tok=1
  printf '%s\n' "$out" | grep -qiE 'skipped=' && tok=0        # 단위 시험이 건너뛰면 통과로 치지 않는다
  last=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -1 | tr '|' '/' | cut -c1-220)
  echo "ST|$name|$rc|$tok|$nfail|$secs|$last"
  printf '%s\n' "$out" | grep -E '백엔드=|^\[FAIL\]' | tr '|' '/' | cut -c1-300 | sed "s/^/NOTE|$name|/"
}
$PY - $POL_DIRS <<'PYEOF' 2>&1 | sed 's/^/ENV|/'
import sys
sys.path.insert(0, ".")
v = ["python " + sys.version.split()[0]]
for m in ("numpy", "onnxruntime", "torch"):
    try:
        mod = __import__(m); v.append(m + " " + str(mod.__version__))
    except Exception:
        v.append(m + " 없음")
print(" · ".join(v))
from rl_walking.deploy.policy_runner import _load_backend
for d in sys.argv[1:]:
    try:
        print("추론 백엔드 %s ← %s" % (_load_backend(d)[0], d))
    except Exception as e:
        print("추론 백엔드 실패 %s ← %s: %s" % (type(e).__name__, d, e))
PYEOF
run "문법(전송한 .py 전부, 메모리 컴파일)" 120 "SYNTAX_OK" $PY -c 'import sys
for f in sys.argv[1:]:
    compile(open(f, encoding="utf-8").read(), f, "exec")
print("SYNTAX_OK", len(sys.argv) - 1)' $SENT_PY
run "러너" 300 "selftest: ALL PASS" $PY rl_walking/deploy/policy_runner_stage4.py --selftest
run "c5_switch" 300 "C5 selftest PASS" $PY rl_walking/deploy/c5_switch.py --selftest
run "kew3_runner" 300 "ALL PASS" $PY rl_walking/deploy/kew3_runner.py --selftest
run "w3_cmd" 120 "[PASS] w3_cmd" $PY jetson/scripts/w3_cmd.py --selftest
run "latency_check" 120 "selftest: ALL PASS" $PY jetson/scripts/latency_check.py --selftest
run "콘솔" 300 "console selftest: ALL PASS" $PY jetson/scripts/operator_console.py --selftest
run "브리지(기본)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest
run "브리지(서기 $KEY_G21A)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --checkpoint "$KEY_G21A"
run "브리지(회복 $KEY_KRH)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --checkpoint "$KEY_KRH"
run "브리지(--experimental-w3)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --experimental-w3
run "브리지(W3 $KEY_W3)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --checkpoint "$KEY_W3"
run "브리지(합성 $KEY_KEW3)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint "$KEY_KEW3"
run "브리지(합성 $KEY_KEW3Q)" 1200 "selftest: ALL PASS" $PY jetson/scripts/rl_bridge_node.py --selftest --experimental-w3 --experimental-kew3 --checkpoint "$KEY_KEW3Q"
run "W3 노드(기준 원문은 파일에서)" 1200 "w3 node selftest: ALL PASS" $PY deploy/cand/head_1008/selftest_w3_node_refs.py
run "stage8 기본 인자 = 종전(지연 옵션 꺼짐)" 120 "STAGE8_DEFAULT_OK" $PY -c 'import sys
sys.path.insert(0, ".")
from jetson.scripts import stage8_12axis_mit_control_node as s
a, _ = s.build_arg_parser().parse_known_args(["--can-channel", "can1", "--wait-takeover"])
s.apply_latency_args(a)
o = s.latency_options_status()
assert o["feedback_after_send"] is False and o["send_on_command"] is False and o["control_hz"] == 50.0 and o["time_based_slew"] is False, o
assert s.MAX_TARGET_STEP_DEG_PER_TICK == 4.0 and abs(s.CONTROL_PERIOD_SEC - 0.02) < 1e-12
print("STAGE8_DEFAULT_OK", o)'
run "stage8 기본 = 종전 단위 시험(가짜 버스)" 300 "OK" $PY -m unittest discover -s tests -p 'test_stage8_latency_options.py' -k test_default
run "stage8 기준점 단위 시험(가짜 버스)" 300 "OK" $PY -m unittest discover -s tests -p 'test_stage8_baseline_from_hold.py'
# 콘솔이 stage8 에 붙일 지연 옵션 인자 = 없음이어야 한다(옵션 파일 config/stage8_latency_opts 없음·빈 파일). 콘솔 코드의 판정 함수를 그대로 부른다.
# 지연 단계를 일부러 켠 뒤에 다시 배포·확인할 때만 HEAD1008_ALLOW_LATENCY_OPTS=1 (그때는 켜진 인자를 적기만 한다). 서비스 환경변수는 dry-run 이 따로 본다.
run "stage8 지연 옵션(콘솔이 붙일 인자 — 기본 0 개여야 함)" 120 "LATENCY_ARGS_OK" $PY -c 'import os, sys
sys.path.insert(0, "jetson/scripts")
os.environ.pop("STAGE8_LATENCY_OPTS", None)
import operator_console as c
a = c.stage8_latency_args()
assert a == [] or sys.argv[1] == "1", "옵션 파일 %s 이 지연 옵션을 켠다: %r" % (c.STAGE8_LATENCY_FILE, a)
print("LATENCY_ARGS_OK 인자", a, "· 옵션 파일", "있음" if os.path.exists(c.STAGE8_LATENCY_FILE) else "없음")' "$ALLOW_LAT"
# 러너 동치(로컬 torch 기준 ↔ 여기 러너): 도구와 기준 npz 가 묶음에 있을 때만
if [ -f reports/night_1007/jetson_parity.py ] && [ -f reports/night_1007/jetson/parity_ref.npz ]; then
  run "러너 동치 jetson_parity(로컬 기준 ↔ 젯슨 러너)" 1800 "jetson parity: ALL PASS" $PY reports/night_1007/jetson_parity.py --check reports/night_1007/jetson/parity_ref.npz
else
  echo "NOTE|러너 동치 jetson_parity|묶음에 없음(reports/night_1007/jetson_parity.py·jetson/parity_ref.npz) — 건너뜀"
fi
run "매니페스트 대조(md5sum -c)" 300 "MD5_ALL_OK" bash -c 'LC_ALL=C md5sum -c --quiet deploy/head_1008.md5 && echo MD5_ALL_OK'
EOS
) || true
  echo "$res" | sed -n 's/^ENV|/     환경: /p'
  while IFS='|' read -r tag name r tok nf secs last; do
    [ "$tag" == ST ] || continue
    if [ "$r" == 0 ] && [ "$tok" == 1 ] && [ "$nf" == 0 ]; then echo "     PASS $name (${secs}s) — $last"
    else echo "     ★FAIL $name (rc=$r 토큰=$tok [FAIL] 줄=$nf, ${secs}s) — $last"; rc=1; fi
    # 콘솔의 '브리지 셀프테스트' 준비 단계는 이 selftest 를 그대로 돌리고 CONSOLE_SELFTEST_TIMEOUT_S 뒤 시간 초과로 실패 처리한다
    # (console_state.py). 새 selftest 는 항목이 늘어 종전보다 오래 걸린다 → 콘솔 선택지 키의 selftest 가 여유 없이 길면 배포 실패로 본다.
    # 1007 밤(42da0c21): 걷기 실험 키 2 개도 콘솔 선택지가 됐다 — 콘솔이 같은 인자(--experimental-w3 [--experimental-kew3])로 돌리므로 같이 본다.
    # 1008: 둘째 합성 키($KEY_KEW3Q)도 같은 이름 무늬("브리지(합성 ")라 이 시간 제한을 같이 받는다.
    case "$name" in "브리지(기본)"|"브리지(서기 "*|"브리지(회복 "*|"브리지(W3 "*|"브리지(합성 "*)
      if [ "${secs:-0}" -gt "$CONSOLE_SELFTEST_LIMIT_S" ]; then
        echo "     ★FAIL $name: ${secs}s — 콘솔 selftest 단계 제한 ${CONSOLE_SELFTEST_TIMEOUT_S}s 의 여유(${CONSOLE_SELFTEST_LIMIT_S}s)를 넘는다. 콘솔에서 준비 단계가 시간 초과로 막힌다"; rc=1
      fi;; esac
  done <<< "$res"
  echo "$res" | sed -n 's/^NOTE|\([^|]*\)|/     · \1: /p' | cut -c1-260
  echo "     전체 출력: $(echo "$res" | sed -n 's/^LOG|//p' | head -1)"
  echo "$res" | grep -q '^ST|' || { echo "     ★ selftest 결과를 받지 못함"; rc=1; }
  return $rc
}

# ------------------------------------------------------------------ 롤백
do_rollback() {   # 백업에서 되돌리고 새로 생긴 파일을 지운다. 반환 0 = 백업 매니페스트와 일치
  local out
  echo "== 롤백: $BAK/files → $R (백업 매니페스트로 검증)"
  out=$(rsh_in "bash -s -- '$R' '$BAK'" <<'EOS'
R=$1; BAK=$2
[ -f "$BAK/BACKUP_MANIFEST.md5" ] || { echo "NOBAK"; exit 0; }
n=0; bad=0
while read -r m f; do
  [ -n "$f" ] || continue
  if [ ! -f "$BAK/files/$f" ] || [ "$(md5sum "$BAK/files/$f" | cut -c1-32)" != "$m" ]; then echo "BAD 백업 손상: $f"; bad=$((bad + 1)); continue; fi
  mkdir -p "$R/$(dirname "$f")" && cp -p "$BAK/files/$f" "$R/$f.tmp_head1008" && mv -f "$R/$f.tmp_head1008" "$R/$f" || { echo "BAD 복원 실패: $f"; bad=$((bad + 1)); continue; }
  [ "$(md5sum "$R/$f" | cut -c1-32)" == "$m" ] || { echo "BAD 복원 뒤 md5 다름: $f"; bad=$((bad + 1)); continue; }
  n=$((n + 1))
done < "$BAK/BACKUP_MANIFEST.md5"
d=0
if [ -f "$BAK/CREATED.txt" ]; then while read -r f; do [ -n "$f" ] || continue; case "$f" in /*|*..*) continue;; esac
  if [ -e "$R/$f" ]; then rm -f "$R/$f" && d=$((d + 1)); fi; done < "$BAK/CREATED.txt"; fi
if [ -f "$BAK/CREATED_DIRS.txt" ]; then sort -r "$BAK/CREATED_DIRS.txt" | while read -r dd; do [ -n "$dd" ] || continue; case "$dd" in /*|*..*) continue;; esac
  rmdir "$R/$dd" 2>/dev/null || true; done; fi
left=0
if [ -f "$BAK/CREATED.txt" ]; then while read -r f; do [ -n "$f" ] && [ -e "$R/$f" ] && left=$((left + 1)); done < "$BAK/CREATED.txt"; fi
rm -rf "$BAK/staging"
echo "RESTORED $n DELETED $d LEFT $left BAD $bad"
[ $bad -eq 0 ] && [ $left -eq 0 ] && echo "rolled_back $(date +%Y-%m-%dT%H:%M:%S)" >> "$BAK/STATE"
EOS
) || true
  echo "$out" | sed 's/^/   /'
  if echo "$out" | grep -q '^NOBAK'; then echo "   백업 없음 — 되돌릴 것이 없다"; return 1; fi
  echo "$out" | grep -q 'BAD 0$' && echo "$out" | grep -q 'LEFT 0 ' || { echo "   ★ 롤백 불완전 — 위 목록을 수동 확인"; return 1; }
  echo "   롤백 완료 — 원래 파일 md5 일치·새 파일 삭제"
}

restart_console() {
  local p SPIDS RPIDS t0 t1
  probe
  safe_now || { echo "콘솔 재기동 보류"; return 1; }
  if [ -n "$FAKE" ]; then echo "restart $(date +%s)" >> "$RH/.fake/restart.log"; echo "== 콘솔 재기동(가짜 — 기록만)"; return 0; fi
  SPIDS=$(echo "$SENS" | awk '{print $1}' | sort | tr '\n' ' ')
  RPIDS=$(rsh "cat ~/run/*.pid 2>/dev/null | sort | tr '\n' ' '" || true)
  for p in $SPIDS; do echo " $RPIDS " | grep -q " $p " || { echo "센서 pid $p 가 PID파일에 없음(콘솔 추적 밖) — 재기동 보류"; return 1; }; done
  [ "$KM" == "process" ] || { echo "KillMode=$KM — 콘솔 재기동이 자식(센서)을 함께 죽일 수 있어 보류"; return 1; }
  echo "== 콘솔 재기동 (KillMode=process — 센서 노드는 유지)"
  rsh "echo "$JETSON_PW" | sudo -S systemctl restart operator-console 2>/dev/null && sleep 4 && systemctl is-active operator-console" || { echo "★ 재기동 실패 — journalctl -u operator-console 확인"; return 1; }
  t0=$(rsh "stat -c %Y '$R/jetson/scripts/operator_console.py'"); t1=$(rsh "date -d \"\$(ps -o lstart= -p \$(systemctl show operator-console -p MainPID --value))\" +%s")
  [ "$t1" -ge "$t0" ] || echo "   ★ 콘솔 프로세스 시작 시각이 파일보다 이르다 — 재기동 확인 필요"
  echo "   선택지: $(rsh "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option>[a-z0-9_]*</option>' | sed 's/<[^>]*>//g' | tr '\n' ' '" || true)"
  echo "   '정책 ON' 버튼: $(rsh "curl -s -m 5 http://127.0.0.1:8800/ | grep -c 'data-action=\"policy_on\"'" || echo 0) (1 = 새 화면)"
  echo "   걷기 실험 키 선택지 수: $(rsh "curl -s -m 5 http://127.0.0.1:8800/ | grep -o '<option value=\"stand_[a-z0-9_]*\"' | wc -l" || echo '?') (2 = 걷기 패널이 있는 새 화면)"
}

# ------------------------------------------------------------------ 정책 폴더만 추가/제거 (매니페스트와 별개 — 밤사이 새 후보용)
#   --add-policy=logs/rsl_rl/<실험>/<run>    : 그 폴더의 exported/policy.onnx(필수)·policy.pt·SOURCE_CHECKPOINT·params/env.yaml(필수)만 보낸다.
#       젯슨에 같은 경로의 다른 파일이 있으면 거부(기존 정책을 덮어쓰지 않는다). md5 는 젯슨 $BAK/POLICY_ADDED.md5 에 적는다.
#       보낸 뒤 젯슨에서 추론 백엔드 로드 + 0 벡터 추론 1 회로 읽히는지 확인한다(실패하면 방금 보낸 파일을 지운다).
#   --remove-policy=<같은 경로>              : POLICY_ADDED.md5 에 적힌 그 폴더의 파일만(md5 가 그대로일 때) 지운다 = 추가의 롤백.
#   그 키로 돌리려면 브리지에 키 등록 코드가 있어야 한다(코드는 이 모드로 가지 않는다). --rollback 은 이 파일들을 건드리지 않는다.
policy_mode() {
  local run=$1 mode=$2 sub p out why lm rmd st files=() todo=()
  [[ "$run" =~ ^logs/rsl_rl/[A-Za-z0-9_]+/[A-Za-z0-9_.-]+$ ]] || { echo "정책 폴더 형식이 아님: $run (logs/rsl_rl/<실험>/<run>)"; return 1; }
  safe_now || return 1
  if [ "$mode" == remove ]; then
    echo "== 정책 폴더 제거: $run (POLICY_ADDED.md5 에 적힌 파일만)"
    out=$(rsh_in "bash -s -- '$R' '$BAK' '$run'" <<'EOS'
R=$1; BAK=$2; RUN=$3; L="$BAK/POLICY_ADDED.md5"
[ -f "$L" ] || { echo "PRM none"; exit 0; }
n=0; bad=0; : > "$L.keep"
while read -r m f rest; do
  [ -n "$f" ] || continue
  case "$f" in "$RUN"/*) ;; *) echo "$m  $f $rest" >> "$L.keep"; continue;; esac
  [ -e "$R/$f" ] || continue
  if [ "$(md5sum "$R/$f" | cut -c1-32)" == "$m" ]; then rm -f "$R/$f" && n=$((n + 1))
  else echo "PRMBAD md5 가 달라져 지우지 않음: $f"; echo "$m  $f $rest" >> "$L.keep"; bad=$((bad + 1)); fi
done < "$L"
mv -f "$L.keep" "$L"
rmdir "$R/$RUN/exported" "$R/$RUN/params" "$R/$RUN" 2>/dev/null
echo "PRM done $n $bad"
echo "policy_removed $(date +%Y-%m-%dT%H:%M:%S) $RUN n=$n bad=$bad" >> "$BAK/STATE"
EOS
) || true
    echo "$out" | sed 's/^/   /'
    echo "$out" | grep -q '^PRM done [0-9]* 0$' || { echo "   ★ 제거 불완전 또는 기록 없음"; return 1; }
    return 0
  fi
  echo "== 정책 폴더 추가: $run"
  for sub in exported/policy.onnx exported/policy.pt exported/SOURCE_CHECKPOINT params/env.yaml; do
    p="$run/$sub"
    if [ -f "$POLICY_ROOT/$p" ]; then
      why=$(forbidden_path "$p"); [ -z "$why" ] || { echo "★ 금지 경로: $p — $why"; return 1; }
      files+=("$p")
    else
      case "$sub" in exported/policy.onnx|params/env.yaml) echo "필수 파일 없음: $POLICY_ROOT/$p"; return 1;; esac
    fi
  done
  out=$(printf '%s\n' "${files[@]}" | rsh_in "cd '$R' && while IFS= read -r f; do if [ -f \"\$f\" ]; then md5sum \"\$f\"; else echo \"-  \$f\"; fi; done")
  for p in "${files[@]}"; do
    lm=$(md5sum "$POLICY_ROOT/$p" | cut -c1-32); rmd=$(echo "$out" | awk -v f="$p" '$2 == f {print $1}')
    if [ "$rmd" == "$lm" ]; then echo "   같음(생략) $p ${lm:0:8}"
    elif [ "$rmd" == "-" ]; then todo+=("$p"); echo "   보냄      $p ${lm:0:8} ($(stat -c %s "$POLICY_ROOT/$p") B)"
    else echo "   ★ 젯슨에 다른 파일이 이미 있다: $p (젯슨 ${rmd:0:8} / 로컬 ${lm:0:8}) — 덮어쓰지 않는다. 중단"; return 1; fi
  done
  if [ ${#todo[@]} -gt 0 ]; then
    for p in "${todo[@]}"; do echo "$(md5sum "$POLICY_ROOT/$p" | cut -c1-32)  $p"; done \
      | rsh_in "mkdir -p '$BAK' && cat > '$BAK/POLICY_TODO.md5'" || { echo "기록 폴더를 만들 수 없음 — 중단"; return 1; }
    st=$(mktemp -d)
    for p in "${todo[@]}"; do mkdir -p "$st/$(dirname "$p")"; cp -p "$POLICY_ROOT/$p" "$st/$p"; done
    tar -C "$st" -cf - . | rsh_in "rm -rf '$BAK/staging_policy' && mkdir -p '$BAK/staging_policy' && tar -xf - -C '$BAK/staging_policy'" \
      || { rm -rf "$st"; echo "전송 실패 — 중단(젯슨 저장소 사본은 그대로)"; return 1; }
    rm -rf "$st"
    out=$(rsh_in "bash -s -- '$R' '$BAK' '$run'" <<'EOS'
R=$1; BAK=$2; RUN=$3
cd "$BAK/staging_policy" && LC_ALL=C md5sum -c --quiet "$BAK/POLICY_TODO.md5" || { echo "PADD_BADSTAGE"; exit 0; }
n=0
while read -r m f; do
  [ -n "$f" ] || continue
  if [ -e "$R/$f" ]; then echo "PADD_EXISTS $f"; continue; fi
  if mkdir -p "$R/$(dirname "$f")" && cp -p "$f" "$R/$f.tmp_head1008" && mv -f "$R/$f.tmp_head1008" "$R/$f"; then
    echo "$m  $f $(date +%Y-%m-%dT%H:%M:%S)" >> "$BAK/POLICY_ADDED.md5"; n=$((n + 1))
  else rm -f "$R/$f.tmp_head1008"; echo "PADD_FAIL $f"; fi
done < "$BAK/POLICY_TODO.md5"
cd /; rm -rf "$BAK/staging_policy" "$BAK/POLICY_TODO.md5"
echo "PADD_DONE $n"
echo "policy_added $(date +%Y-%m-%dT%H:%M:%S) $RUN n=$n" >> "$BAK/STATE"
EOS
) || true
    echo "$out" | sed 's/^/   /'
    echo "$out" | grep -q "^PADD_DONE ${#todo[@]}\$" || { echo "   ★ 정책 폴더 추가 실패 — 젯슨의 $run 확인(되돌리기 --remove-policy=$run)"; return 1; }
  fi
  out=$(rsh "cd '$R' && export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= && $PY -c \"
import sys
sys.path.insert(0, '.')
import numpy as np
from rl_walking.deploy.policy_runner import _load_backend
from rl_walking.deploy.policy_runner_stage4 import _probe_obs_dim
name, net = _load_backend('$run/exported')
n = _probe_obs_dim(net)
o = np.asarray(net(np.zeros(n, np.float32))).reshape(-1)
assert o.shape == (12,) and np.all(np.isfinite(o))
print('POLICY_OK backend=%s obs=%d' % (name, n))
\" 2>&1 | tail -1" || true)
  echo "   젯슨 추론 확인: $out"
  if ! echo "$out" | grep -q '^POLICY_OK'; then
    echo "   ★ 젯슨에서 정책을 읽지 못함 — 방금 보낸 파일을 지운다"; policy_mode "$run" remove || true; return 1
  fi
  echo "== 정책 폴더 추가 완료 — 기록 $BAK/POLICY_ADDED.md5 · 되돌리기: $0 --remove-policy=$run"
}
if [ -n "$ADDPOL" ]; then policy_mode "$ADDPOL" add || exit 1; exit 0; fi
if [ -n "$RMPOL" ]; then policy_mode "$RMPOL" remove || exit 1; exit 0; fi

# ------------------------------------------------------------------ 동작 분기
if [ $ROLLBACK -eq 1 ]; then
  safe_now || exit 1
  do_rollback || exit 1
  if [ $RESTART -eq 1 ]; then restart_console || exit 1; else echo "   콘솔 서비스는 재기동 전까지 메모리의 새 코드로 돈다 — 되돌리려면 --restart-console"; fi
  exit 0
fi
MMD5=$(md5sum $MANIFEST | cut -c1-32)
if [ $VERIFY -eq 1 ]; then
  [ ${#L_SAME[@]} -eq ${#P_PATH[@]} ] || { echo "젯슨이 매니페스트와 같지 않다(같음 ${#L_SAME[@]}/${#P_PATH[@]}) — --verify 는 적용 뒤에만"; exit 1; }
  echo "== [5] 젯슨 selftest (복사 없음)"
  if remote_selftest; then
    rsh "[ -d '$BAK' ] && echo \"selftest_pass \$(date +%Y-%m-%dT%H:%M:%S) manifest $MMD5 (verify)\" >> '$BAK/STATE'" || true
    echo "== 확인 끝 — 전부 통과"
  else echo "== ★ selftest 실패(롤백하지 않음)"; exit 1; fi
  exit 0
fi
if [ $APPLY -eq 0 ]; then
  if [ $RESTART -eq 1 ]; then
    [ ${#L_SAME[@]} -eq ${#P_PATH[@]} ] || { echo "젯슨이 매니페스트와 같지 않다 — 재기동은 적용(또는 롤백) 뒤에"; exit 1; }
    # 새 코드로 콘솔을 띄우는 것은 이 매니페스트의 젯슨 selftest 가 통과한 기록이 있을 때만
    rsh "grep -q 'selftest_pass .* manifest $MMD5' '$BAK/STATE' 2>/dev/null" \
      || { echo "이 매니페스트($MMD5)의 젯슨 selftest 통과 기록이 없다($BAK/STATE) — $0 --verify 로 먼저 통과시킬 것"; exit 1; }
    restart_console || exit 1; exit 0
  fi
  echo "== dry-run 끝 — 젯슨에 쓴 것 없음. 적용: $0 --apply (승인 뒤) → 콘솔 유휴일 때 $0 --restart-console"
  [ ${#L_BAD[@]} -eq 0 ] || echo "   ★ 예상 밖 ${#L_BAD[@]} 건이 있어 지금은 --apply 가 거부된다(젯슨 상태가 인벤토리와 달라짐 — 인벤토리·매니페스트를 다시 만들거나 $ACCEPT 에 확인한 md5 를 적는다)"
  exit 0
fi

# ------------------------------------------------------------------ --apply
[ ${#L_BAD[@]} -eq 0 ] || { echo "예상 밖 ${#L_BAD[@]} 건 — 전제가 깨짐, 중단(아무것도 쓰지 않았다)"; exit 1; }
safe_now || exit 1
[ "$FREE_KB" -ge $((need_kb + MIN_FREE_KB)) ] || { echo "디스크 여유 부족($((FREE_KB / 1024)) MB) — 중단. 세션 타임라인(~/logs/sessions) 용량 확인"; exit 1; }
TODO=("${L_REPLACE[@]}" "${L_CREATE[@]}")
if [ ${#TODO[@]} -eq 0 ]; then echo "== [5] 이미 전부 새 판 — 복사 생략"
else
  echo "== [5] 백업 → 전송 → 교체 (${#TODO[@]} 파일: 바뀜 ${#L_REPLACE[@]} · 새로 ${#L_CREATE[@]})"
  STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
  for p in "${TODO[@]}"; do
    why=$(forbidden_path "$p"); [ -z "$why" ] || { echo "★ 금지 파일이 전송 직전에 발견됨: $p — $why. 중단"; exit 1; }
    mkdir -p "$STAGE/$(dirname "$p")"; cp -p "$(src_of "$p")" "$STAGE/$p"
    [ "$(md5sum "$STAGE/$p" | cut -c1-32)" == "${P_MD5[$p]}" ] || { echo "임시 사본 md5 다름: $p — 중단"; exit 1; }
  done
  # 할 일 목록을 젯슨에 둔다: "R <지금 md5> <경로>"(덮어씀 — 백업 대상) / "C - <경로>"(새 파일 — 롤백 때 삭제)
  { for p in "${L_REPLACE[@]}"; do echo "R ${CUR[$p]} $p"; done; for p in "${L_CREATE[@]}"; do echo "C - $p"; done; } \
    | rsh_in "mkdir -p '$BAK/files' && cat > '$BAK/TODO.txt'" || { echo "백업 폴더를 만들 수 없음 — 중단(원본은 건드리지 않았다)"; exit 1; }
  # (a) 백업: 덮어쓸 파일을 $BAK/files 로 복사하고 md5 를 확인한 뒤에만 매니페스트에 적는다. 이미 백업된 원본은 덮어쓰지 않는다(재실행 안전).
  out=$(rsh_in "bash -s -- '$R' '$BAK'" <<'EOS'
R=$1; BAK=$2
touch "$BAK/BACKUP_MANIFEST.md5" "$BAK/CREATED.txt" "$BAK/CREATED_DIRS.txt" || { echo "BKFAIL 백업 기록 파일을 만들 수 없음"; echo "BKDONE new=0 kept=0 create=0 bad=1"; exit 0; }
in_manifest() { awk -v f="$1" '$2 == f {ok = 1} END {exit ok ? 0 : 1}' "$BAK/BACKUP_MANIFEST.md5"; }
nb=0; nk=0; nc=0; bad=0
while read -r kind m f; do
  [ -n "$f" ] || continue
  d=$(dirname "$f")
  while [ "$d" != . ] && [ ! -d "$R/$d" ]; do grep -qxF "$d" "$BAK/CREATED_DIRS.txt" || echo "$d" >> "$BAK/CREATED_DIRS.txt"; d=$(dirname "$d"); done
  if [ "$kind" == C ]; then
    if ! grep -qxF "$f" "$BAK/CREATED.txt" && ! in_manifest "$f"; then echo "$f" >> "$BAK/CREATED.txt"; fi
    nc=$((nc + 1)); continue
  fi
  if in_manifest "$f" && [ -f "$BAK/files/$f" ]; then nk=$((nk + 1)); continue; fi       # 원본이 이미 백업돼 있다
  if grep -qxF "$f" "$BAK/CREATED.txt"; then nk=$((nk + 1)); continue; fi               # 이 배포가 전에 만든 파일(원래 없던 것)
  mkdir -p "$BAK/files/$(dirname "$f")" && cp -p "$R/$f" "$BAK/files/$f" || { echo "BKFAIL 복사 실패: $f"; bad=$((bad + 1)); continue; }
  if [ "$(md5sum "$BAK/files/$f" | cut -c1-32)" != "$m" ]; then echo "BKFAIL 백업 md5 다름: $f"; rm -f "$BAK/files/$f"; bad=$((bad + 1)); continue; fi
  echo "$m  $f" >> "$BAK/BACKUP_MANIFEST.md5"; nb=$((nb + 1))
done < "$BAK/TODO.txt"
echo "BKDONE new=$nb kept=$nk create=$nc bad=$bad"
[ $bad -eq 0 ] && echo "backup_complete $(date +%Y-%m-%dT%H:%M:%S)" >> "$BAK/STATE"
EOS
) || true
  echo "$out" | sed 's/^/   /'
  echo "$out" | grep -q '^BKDONE .*bad=0$' || { echo "백업 실패 — 중단(원본은 건드리지 않았다)"; exit 1; }
  # (b) 전송: 임시 폴더($BAK/staging)에 풀고 md5 검증 — 여기까지 저장소 사본은 그대로다
  tar -C "$STAGE" -cf - . | rsh_in "rm -rf '$BAK/staging' && mkdir -p '$BAK/staging' && tar -xf - -C '$BAK/staging'" \
    || { echo "전송 실패 — 중단(원본은 건드리지 않았다)"; exit 1; }
  out=$(for p in "${TODO[@]}"; do echo "${P_MD5[$p]}  $p"; done | rsh_in "cd '$BAK/staging' && LC_ALL=C md5sum -c --quiet - && echo STAGE_OK") || true
  echo "$out" | grep -q '^STAGE_OK' || { echo "임시 폴더 md5 불일치 — 중단(원본은 건드리지 않았다)"; echo "$out" | head -5; rsh "rm -rf '$BAK/staging'" || true; exit 1; }
  echo "   전송·임시 폴더 md5 일치 ${#TODO[@]}/${#TODO[@]}"
  # (c) 제자리 교체: 파일마다 같은 폴더의 임시 이름으로 복사한 뒤 mv(파일 단위 원자적 — 잘린 파일이 남지 않는다)
  out=$(rsh_in "bash -s -- '$R' '$BAK'" <<'EOS'
R=$1; BAK=$2; n=0; bad=0
echo "applying $(date +%Y-%m-%dT%H:%M:%S)" >> "$BAK/STATE"
while read -r kind m f; do
  [ -n "$f" ] || continue
  if mkdir -p "$R/$(dirname "$f")" && cp -p "$BAK/staging/$f" "$R/$f.tmp_head1008" && mv -f "$R/$f.tmp_head1008" "$R/$f"; then n=$((n + 1))
  else rm -f "$R/$f.tmp_head1008"; echo "INFAIL $f"; bad=$((bad + 1)); fi
done < "$BAK/TODO.txt"
echo "INDONE n=$n bad=$bad"
EOS
) || true
  echo "$out" | sed 's/^/   /'
  if ! echo "$out" | grep -q '^INDONE .*bad=0$'; then echo "교체 실패 — 롤백"; do_rollback || true; exit 1; fi
fi
read_remote_md5; classify
if [ ${#L_SAME[@]} -ne ${#P_PATH[@]} ]; then
  echo "★ 적용 뒤 md5 가 매니페스트와 다름(같음 ${#L_SAME[@]}/${#P_PATH[@]}) — 롤백"; do_rollback || true; exit 1
fi
rsh_in "mkdir -p '$BAK' && cat > '$BAK/APPLIED.plan'" < "$PLAN" || true
rsh "rm -rf '$BAK/staging'; echo \"applied \$(date +%Y-%m-%dT%H:%M:%S) manifest $MMD5\" >> '$BAK/STATE'" || true
echo "   적용 뒤 젯슨 md5 = 매니페스트 ${#L_SAME[@]}/${#P_PATH[@]}"
echo "== [6] 젯슨 selftest (오프라인 — 실패하면 자동 롤백)"
if remote_selftest; then
  rsh "echo \"selftest_pass \$(date +%Y-%m-%dT%H:%M:%S) manifest $MMD5\" >> '$BAK/STATE'" || true
else
  echo "젯슨 selftest 실패 — 롤백"; do_rollback || true
  echo "   콘솔 서비스는 재기동하지 않았으므로 메모리의 옛 코드 그대로다."; exit 1
fi
if [ $RESTART -eq 1 ]; then
  restart_console || { echo "파일은 올라갔고 selftest 도 통과했으나 콘솔 재기동은 보류 — 조작이 끝난 뒤 $0 --restart-console"; exit 1; }
fi
echo "== 배포 완료 — $NAME (매니페스트 ${MMD5:0:8}). 백업 $BAK · 되돌리기: $0 --rollback --restart-console"
[ $RESTART -eq 1 ] || echo "   콘솔 서비스는 아직 옛 코드로 돌고 있다(브리지·stage8 은 다음 기동부터 새 파일). 콘솔 유휴일 때: $0 --restart-console"
