#!/bin/bash
# deploy_head_1008.sh 의 로컬 왕복 시험 — **젯슨에 접속하지 않는다**(JETSON_FAKE_ROOT 가짜 모드만 쓴다). ROS·CAN·GPU 를 쓰지 않는다.
#
#   deploy/test_head_1008_local.sh [--work <폴더>] [--jetenv <site 폴더>] [--policy-root <logs/ 가 있는 폴더>] [--hwmap <맵 파일>] [--quick] [--keep]
#   deploy/test_head_1008_local.sh --make-jetenv <폴더>     # 젯슨 모사 파이썬 환경 만들기(pip, 네트워크 필요): numpy 1.21.5 · onnxruntime 1.15.1 · python-can
#
# 하는 일: 젯슨 인벤토리대로 가짜 젯슨 홈을 만들고(make_fake_jetson.py), 그 위에서 배포 스크립트의
#   dry-run(무변경) → 금지 파일 강제 → 제어 프로세스·활성 세션·예상 밖 md5·디스크 부족 거부(무변경) → selftest 실패 시 자동 롤백 →
#   정상 적용(백업·전송·교체·md5·selftest) → 젯슨 판 ↔ 새 판 동작 비교(브리지 발행·stage8 CAN 프레임 — 가짜 ROS·가짜 CAN) →
#   재적용(멱등) → --verify → 콘솔 재기동(가짜) → 롤백 → **처음 스냅샷과 비교** → 재적용·부분 적용 복구·롤백
# 을 차례로 돌리고 PASS/FAIL 을 적는다. 매니페스트를 다시 만든 뒤에는 이 시험을 다시 돌린다.
# 다른 작업이 기존 세션의 블랙박스 줄에 기록 필드를 의도해서 더했다면 T9b 가 '다름'을 낸다 — 더한 키를 소스에서 확인한 뒤 NEW_BB_KEYS(아래)에 넣는다.
# 지금 들어 있는 6 개 = 2522dc26(모터 오류 기록)이 틱 기록에 더한 필드. 이 키들은 **새 판의 블랙박스 줄에서만** 빼고, 나머지 키·값·순서와 모든 발행은 그대로 비교한다.
#
# 파이썬 환경: --jetenv(또는 HEAD1008_JETENV) 를 주면 가짜 젯슨의 selftest 가 그 폴더만 쓴다(사용자 site 차단 = torch 없음 → onnxruntime 백엔드,
#   젯슨과 같은 numpy 1.21.5). 안 주면 호출자의 사용자 site(torch·numpy 2.x)를 쓴다 — 왕복 논리는 같지만 젯슨 환경 모사는 아니다.
#   --quick: selftest 를 돌리는 경우를 한 번(정상 적용)으로 줄인다.
# 한계: 가짜 트리의 하드웨어맵은 젯슨 실물 맵이 아니고(저장소에 없다), ~/rl_calib 는 자리표시 파일이다. ssh·sudo·systemd 경로는 시험하지 않는다.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
WORK=""; JETENV=${HEAD1008_JETENV:-}; POLICY_ROOT=${HEAD1008_POLICY_ROOT:-}; HWMAP=""; KEEP=0; QUICK=0
while [ $# -gt 0 ]; do case $1 in
  --work) WORK=$2; shift;; --jetenv) JETENV=$2; shift;; --policy-root) POLICY_ROOT=$2; shift;; --hwmap) HWMAP=$2; shift;;
  --keep) KEEP=1;; --quick) QUICK=1;;
  --make-jetenv) D=$2; mkdir -p "$D"
     python3 -m pip install --quiet --no-deps --target "$D" numpy==1.21.5 onnxruntime==1.15.1 && python3 -m pip install --quiet --target "$D" python-can \
       && echo "젯슨 모사 환경: $D (numpy 1.21.5 · onnxruntime 1.15.1 · python-can) — --jetenv $D 로 쓴다"; exit $?;;
  *) echo "알 수 없는 인자 $1"; exit 2;; esac; shift; done
[ -n "$POLICY_ROOT" ] || { [ -d logs/rsl_rl ] && POLICY_ROOT=$REPO || { echo "정책 폴더(logs/rsl_rl)가 없다 — --policy-root <메인 체크아웃>"; exit 2; }; }
POLICY_ROOT=$(cd "$POLICY_ROOT" && pwd)
[ -n "$WORK" ] || WORK=$(mktemp -d "${TMPDIR:-/tmp}/head1008_rt.XXXXXX")
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
FH=$WORK/fake_home; LOGD=$WORK/logs; mkdir -p "$LOGD"
if [ -n "$JETENV" ]; then FPP=$(cd "$JETENV" && pwd); ENVDESC="젯슨 모사 환경 $FPP (사용자 site 차단)"
else FPP=$(python3 -c "import site; print(site.getusersitepackages())"); ENVDESC="호출자 사용자 site $FPP (젯슨 모사 아님 — torch·numpy 2.x 일 수 있다)"; fi
SCRIPT=./deploy/deploy_head_1008.sh
NPASS=0; NFAIL=0
ok() { NPASS=$((NPASS + 1)); echo "[PASS] $1"; }
ng() { NFAIL=$((NFAIL + 1)); echo "[FAIL] $1"; }
chk() { if [ "$1" -eq 0 ]; then ok "$2"; else ng "$2"; fi; }
dep() {   # $1 = 로그 이름, 나머지 = 스크립트 인자. 환경은 호출 쪽에서 덧붙인다. 종료 코드를 돌려준다.
  local name=$1; shift
  JETSON_FAKE_ROOT="$FH" JETSON_FAKE_PYTHONPATH="${PP_OVERRIDE:-$FPP}" HEAD1008_POLICY_ROOT="$POLICY_ROOT" "$SCRIPT" --skip-local-gate "$@" > "$LOGD/$name.log" 2>&1
}
snap() {  # 가짜 홈의 파일 md5 + 폴더 목록(백업 폴더·가짜 재기동 기록 제외)
  (cd "$FH" && { find . -type f ! -path './humanoid_repo_bak_1008/*' ! -path './.fake/restart.log' -print0 | sort -z | xargs -0 md5sum
                 find . -type d ! -path './humanoid_repo_bak_1008' ! -path './humanoid_repo_bak_1008/*' | sort; }) > "$1"
}
same_as_start() { snap "$WORK/snap_now.txt"; cmp -s "$WORK/snap_start.txt" "$WORK/snap_now.txt"; }
has() { grep -q -- "$2" "$LOGD/$1.log"; }

echo "== 준비: 저장소 $REPO · 정책 $POLICY_ROOT · 작업 폴더 $WORK"
echo "   가짜 젯슨 selftest 환경: $ENVDESC"
bash -n "$SCRIPT" && bash -n deploy/test_head_1008_local.sh; chk $? "T0 bash -n (배포 스크립트·이 시험)"
python3 deploy/make_head_manifest.py --check --policy-root "$POLICY_ROOT" > "$LOGD/manifest_check.log" 2>&1; chk $? "T0 매니페스트가 현재 트리와 일치(make_head_manifest.py --check) — $(tail -1 "$LOGD/manifest_check.log")"
python3 deploy/cand/head_1008/check_numpy_compat.py --plan deploy/head_1008.plan > "$LOGD/numpy_static.log" 2>&1; chk $? "T0 numpy 1.21.5 정적 점검 — $(tail -1 "$LOGD/numpy_static.log")"

rm -rf "$FH"
python3 deploy/cand/head_1008/make_fake_jetson.py --out "$FH" --policy-root "$POLICY_ROOT" ${HWMAP:+--hwmap "$HWMAP"} > "$LOGD/make_fake.log" 2>&1
chk $? "T1 가짜 젯슨 홈 생성 — $(tail -1 "$LOGD/make_fake.log" | sed 's/^ *//')"
if [ $QUICK -eq 0 ]; then    # 기존 세션 동작 비교(T9b)의 기준 = 손대지 않는 두 번째 사본
  rm -rf "$WORK/fake_ref"; python3 deploy/cand/head_1008/make_fake_jetson.py --out "$WORK/fake_ref" --policy-root "$POLICY_ROOT" ${HWMAP:+--hwmap "$HWMAP"} >> "$LOGD/make_fake.log" 2>&1
fi
snap "$WORK/snap_start.txt"
echo "   처음 스냅샷: 파일 $(grep -c '^[0-9a-f]\{32\} ' "$WORK/snap_start.txt") 개 · 폴더 $(grep -c '^\./\|^\.$' "$WORK/snap_start.txt") 개"

dep t2_dryrun; rc=$?
[ $rc -eq 0 ] && has t2_dryrun '★예상 밖 0' && has t2_dryrun 'dry-run 끝' && same_as_start
chk $? "T2 dry-run: 종료 0 · 예상 밖 0 · 가짜 젯슨 무변경 — $(grep -o '이미 새 판과 같음.*' "$LOGD/t2_dryrun.log")"

# --- T3 금지 파일 강제(계획서에 금지 경로를 끼워 넣으면 중단해야 한다)
n=0; nbad=0
for fp in config/robot_12dof_hardware_map.json config/imu_mount_calib.json config/stage8_latency_opts jetson/calib_data/level_pose_0918.json \
          jetson/scripts/level_pose.json ../rl_calib/level_pose.json /home/mama/rl_calib/hang_stand.py rl_calib/sensors_up.sh \
          logs/rsl_rl/biped12_stepb0/x/model_15800.pt logs/rsl_rl/biped12_stepb0/x/params/agent.yaml deploy/stand_zero_backup.md \
          jetson/scripts/can_channel jetson/scripts/robot_12dof_hardware_map.json.py.json robot_runtime/last_pose_journal.json; do
  n=$((n + 1)); cp deploy/head_1008.plan "$WORK/plan_bad.plan"
  printf '%s\tsend\tsync\trepo\t00000000000000000000000000000000\t1\t-\t-\n' "$fp" >> "$WORK/plan_bad.plan"
  HEAD1008_PLAN="$WORK/plan_bad.plan" dep t3_forbid_$n --apply; rc=$?
  if [ $rc -ne 0 ] && has t3_forbid_$n '★ 금지' && has t3_forbid_$n '아무것도 하지 않았다' && same_as_start; then :; else nbad=$((nbad + 1)); echo "     금지 안 됨: $fp (rc=$rc)"; fi
done
chk $nbad "T3 금지 파일 강제(배포 스크립트): 금지 경로 $n 종을 계획서에 넣으면 --apply 가 중단·무변경"
python3 - > "$LOGD/t3_gen.log" 2>&1 <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mk", "deploy/make_head_manifest.py"); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
bad = ["config/robot_12dof_hardware_map.json", "config/imu_mount_calib.json", "config/stage8_latency_opts", "jetson/calib_data/level_pose_0918.json",
       "jetson/scripts/level_pose.json", "../rl_calib/level_pose.json", "/home/mama/rl_calib/hang_stand.py", "rl_calib/sensors_up.sh",
       "logs/rsl_rl/biped12_stepb0/x/model_15800.pt", "logs/rsl_rl/biped12_stepb0/x/params/agent.yaml", "deploy/stand_zero_backup.md",
       "jetson/scripts/can_channel", "robot_runtime/last_pose_journal.json", "jetson/scripts/a b.py", "docs/x.md"]
good = ["jetson/scripts/rl_bridge_node.py", "jetson/scripts/sensors_up.sh", "jetson/scripts/console_ui.html", "robot_runtime/joint_limits.py",
        "jetson/scripts/imu_mount_calib.py", "logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported/policy.onnx",
        "logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/params/env.yaml", "deploy/head_1008.md5", "tests/test_latency_check.py"]
e = [p for p in bad if mk.forbidden_reason(p) is None] + [p for p in good if mk.forbidden_reason(p) is not None]
print("오판:", e); sys.exit(1 if e else 0)
PY
chk $? "T3 금지 파일 규칙(생성기 forbidden_reason): 금지 15 종 거부 · 정상 9 종 통과"
awk -F'\t' '!/^#/ && ($1 ~ /^(config\/|jetson\/calib_data\/)/ || $1 ~ /\.json$/ || $1 ~ /rl_calib/ ||
                       ($1 !~ /\.(py|sh)$/ && $1 ~ /(hardware_map|level_pose|imu_mount_calib|pose_journal|stand_zero|stand_snapshot)/)) {print $1}' \
  deploy/head_1008.plan > "$LOGD/t3_plan_scan.log"; [ ! -s "$LOGD/t3_plan_scan.log" ]
chk $? "T3 실제 계획서($(grep -vc '^#' deploy/head_1008.plan) 항목)에 config/·calib_data/·.json·rl_calib·맵·영점·보정 데이터 경로 0 건"

# --- T4 제어 프로세스가 살아 있으면 거부
cp "$FH/.fake/ps.txt" "$WORK/ps_orig.txt"; nbad=0
for line in "  4321 python3 /home/mama/humanoid_repo/jetson/scripts/stage8_12axis_mit_control_node.py --can-channel can1 --wait-takeover" \
            "  4322 python3 /home/mama/humanoid_repo/jetson/scripts/rl_bridge_node.py --live --checkpoint stand_crouch_g21a" \
            "  4323 python3 /home/mama/rl_calib/hang_stand.py"; do
  { cat "$WORK/ps_orig.txt"; echo "$line"; } > "$FH/.fake/ps.txt"
  for mode in --apply --rollback; do
    dep t4_ctrl $mode; rc=$?
    if [ $rc -ne 0 ] && has t4_ctrl '제어 프로세스 실행 중'; then :; else nbad=$((nbad + 1)); echo "     거부 안 됨: $mode / $line (rc=$rc)"; fi
  done
done
cp "$WORK/ps_orig.txt" "$FH/.fake/ps.txt"; same_as_start || nbad=$((nbad + 1))
chk $nbad "T4 제어 프로세스(stage8·브리지·hang_stand) 생존 시 --apply·--rollback 거부·무변경"

# --- T5 콘솔에 활성 세션이 있으면 거부
cp "$FH/.fake/state.json" "$WORK/state_orig.json"
echo '{"session": {"active": true}, "bridge": {"live": false, "armed": false}, "procs": {}, "trial": {"phase": "기립"}}' > "$FH/.fake/state.json"
dep t5_session --apply; rc1=$?
echo 'not json' > "$FH/.fake/state.json"; dep t5_nostate --apply; rc2=$?
cp "$WORK/state_orig.json" "$FH/.fake/state.json"
[ $rc1 -ne 0 ] && has t5_session '유휴가 아니거나' && [ $rc2 -ne 0 ] && has t5_nostate '유휴가 아니거나' && same_as_start
chk $? "T5 콘솔 활성 세션·상태 읽기 실패 시 --apply 거부·무변경"

# --- T6 젯슨 파일이 인벤토리와 다르면(로컬 수정본) 거부
f1=$FH/humanoid_repo/jetson/scripts/operator_console.py; f2=$FH/humanoid_repo/jetson/scripts/hang_stand.py; f3=$FH/humanoid_repo/jetson/scripts/robotctl.py
cp -p "$f1" "$WORK/f1.orig"; echo "# jetson local hotfix" >> "$f1"; dep t6_unexp_send --apply; rc1=$?; cp -p "$WORK/f1.orig" "$f1"
cp -p "$f2" "$WORK/f2.orig"; echo "# jetson local hotfix" >> "$f2"; dep t6_unexp_verify --apply; rc2=$?; cp -p "$WORK/f2.orig" "$f2"
mv "$f3" "$WORK/f3.orig"; dep t6_missing --apply; rc3=$?; mv "$WORK/f3.orig" "$f3"
[ $rc1 -ne 0 ] && has t6_unexp_send '★예상 밖 1' && has t6_unexp_send '아무것도 쓰지 않았다' && [ $rc2 -ne 0 ] && has t6_unexp_verify '확인 대상인데' \
  && [ $rc3 -ne 0 ] && has t6_missing '지금 젯슨에 없음' && same_as_start
chk $? "T6 예상 밖 md5(덮어쓸 파일의 로컬 수정본·확인 대상 변경·사라진 파일) 시 --apply 거부·무변경"

# --- T7 디스크 여유 부족이면 거부
HEAD1008_MIN_FREE_KB=99999999999 dep t7_disk --apply; rc=$?
[ $rc -ne 0 ] && has t7_disk '디스크 여유 부족' && same_as_start
chk $? "T7 디스크 여유 부족 시 --apply 거부·무변경"

# --- T8 젯슨 selftest 실패 → 자동 롤백
mkdir -p "$WORK/empty_site"
PP_OVERRIDE="$WORK/empty_site" dep t8_selftest_fail --apply; rc=$?; unset PP_OVERRIDE
[ $rc -ne 0 ] && has t8_selftest_fail '젯슨 selftest 실패 — 롤백' && has t8_selftest_fail '롤백 완료' && same_as_start
chk $? "T8 selftest 실패(추론 백엔드 없는 환경) → 자동 롤백 → 처음과 동일 — $(grep -c '★FAIL' "$LOGD/t8_selftest_fail.log") 항목 실패, $(grep -o 'RESTORED.*' "$LOGD/t8_selftest_fail.log")"

# --- T9 정상 적용
dep t9_apply --apply; rc=$?
( cd "$FH/humanoid_repo" && LC_ALL=C md5sum -c --quiet deploy/head_1008.md5 ) > "$LOGD/t9_md5.log" 2>&1; rcm=$?
[ $rc -eq 0 ] && has t9_apply '배포 완료' && ! has t9_apply '★FAIL' && [ $rcm -eq 0 ]
chk $? "T9 --apply: 종료 0 · $(grep -o 'BKDONE.*' "$LOGD/t9_apply.log") · $(grep -o 'INDONE.*' "$LOGD/t9_apply.log") · selftest PASS $(grep -c '^     PASS' "$LOGD/t9_apply.log")/$(grep -cE '^     (PASS|★FAIL)' "$LOGD/t9_apply.log") · 가짜 젯슨에서 md5sum -c 통과"
snap "$WORK/snap_applied.txt"
diff <(grep -E ' \./(rl_calib|humanoid_repo/config|humanoid_repo/jetson/calib_data|logs|run)/' "$WORK/snap_start.txt") \
     <(grep -E ' \./(rl_calib|humanoid_repo/config|humanoid_repo/jetson/calib_data|logs|run)/' "$WORK/snap_applied.txt") > "$LOGD/t9_untouched.log"
chk $? "T9 적용 뒤 ~/rl_calib·config(하드웨어맵)·jetson/calib_data 무변경($(grep -cE ' \./(rl_calib|humanoid_repo/config|humanoid_repo/jetson/calib_data)/' "$WORK/snap_start.txt") 파일 md5 동일)"
[ -z "$(find "$FH" -name __pycache__ -o -name '*.tmp_head1008' | head -1)" ] && [ -z "$(ls -A "$FH/logs" 2>/dev/null)" ]
chk $? "T9 selftest 가 가짜 젯슨에 __pycache__·임시 파일·~/logs 기록을 남기지 않음"
grep -E '환경: |백엔드' "$LOGD/t9_apply.log" | sed 's/^ */   /' | cut -c1-200

if [ $QUICK -eq 0 ]; then
  # --- T9b 기존 세션의 동작 불변 확인: 젯슨 판(가짜 기준 트리) ↔ 새 판(방금 적용한 가짜 젯슨) — 실제 노드 코드·실제 정책 파일, 가짜 ROS·가짜 CAN
  fpy() { env -i HOME="$WORK/pyhome" PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FPP" python3 -s "$@"; }
  mkdir -p "$WORK/pyhome"; REFT=$WORK/fake_ref/humanoid_repo; NEWT=$FH/humanoid_repo; RC=deploy/cand/head_1008/replay_compare.py; SC=deploy/cand/head_1008/stage8_frame_compare.py
  fpy $RC --repo "$REFT" --dump "$WORK/replay_before.json" > "$LOGD/t9b_replay_dump.log" 2>&1; r1=$?
  fpy $RC --repo "$REFT" --dump "$WORK/replay_before2.json" >> "$LOGD/t9b_replay_dump.log" 2>&1; r2=$?
  fpy $RC --repo "$NEWT" --dump "$WORK/replay_after.json" >> "$LOGD/t9b_replay_dump.log" 2>&1; r3=$?
  python3 $RC --compare "$WORK/replay_before.json" "$WORK/replay_before2.json" > "$LOGD/t9b_replay_control.log" 2>&1; r4=$?
  NEW_BB_KEYS=${HEAD1008_REPLAY_NEW_BB_KEYS-fb_age_max_s,fb_age_max_joint,fb_age_over,err_raw,s8_warn,errlog_fail}
  python3 $RC --compare "$WORK/replay_before.json" "$WORK/replay_after.json" --new-bb-keys "$NEW_BB_KEYS" --ignore-keys "${HEAD1008_REPLAY_IGNORE_KEYS:-}" > "$LOGD/t9b_replay_cmp.log" 2>&1; r5=$?
  # 변이 대조군: 새 필드가 아닌 블랙박스 값 하나 / 발행 하나를 바꾼 사본은 같은 옵션에서 '다름'이어야 한다(옵션이 차이를 가리지 않는지)
  python3 deploy/cand/head_1008/replay_mutate.py "$WORK/replay_after.json" "$WORK/replay_mut_bb.json" "$WORK/replay_mut_pub.json" "$NEW_BB_KEYS" > "$LOGD/t9b_replay_mut.log" 2>&1; rmu=$?
  python3 $RC --compare "$WORK/replay_before.json" "$WORK/replay_mut_bb.json" --new-bb-keys "$NEW_BB_KEYS" > "$LOGD/t9b_replay_mut_bb.log" 2>&1; m1=$?
  python3 $RC --compare "$WORK/replay_before.json" "$WORK/replay_mut_pub.json" --new-bb-keys "$NEW_BB_KEYS" > "$LOGD/t9b_replay_mut_pub.log" 2>&1; m2=$?
  python3 $RC --compare "$WORK/replay_after.json" "$WORK/replay_before.json" --new-bb-keys "$NEW_BB_KEYS" > "$LOGD/t9b_replay_mut_rev.log" 2>&1; m3=$?
  rm -f "$WORK"/replay_mut_bb.json "$WORK"/replay_mut_pub.json
  if [ -n "$NEW_BB_KEYS" ] && [ $r5 -eq 0 ]; then
    [ $rmu -eq 0 ] && [ $m1 -ne 0 ] && grep -q '다름 1' "$LOGD/t9b_replay_mut_bb.log" && [ $m2 -ne 0 ] && grep -q '다름 1' "$LOGD/t9b_replay_mut_pub.log" \
      && [ $m3 -ne 0 ] && grep -q 'A(기준 판) 블랙박스에도 있다' "$LOGD/t9b_replay_mut_rev.log"
    chk $? "T9b 대조군(새 블랙박스 필드 옵션이 차이를 가리지 않는가): 새 필드 밖 블랙박스 값 1 개 변이 = 검출 · 새 필드 이름의 키를 발행에 끼운 변이 = 검출 · 기준 판에 그 키가 있으면 비교 거부"
  fi
  [ $((r1 + r2 + r3 + r4)) -eq 0 ]; chk $? "T9b 대조군: 젯슨 판 브리지를 두 번 돌리면 발행이 같다 — $(tail -1 "$LOGD/t9b_replay_control.log")"
  chk $r5 "T9b 브리지 발행·블랙박스 젯슨 판 ↔ 새 판(서기 g21a·g12·g12_ema0·g11a·v18b, 공중 시작, 회복 c5·kr·krh × live/dry × 가드 data/safe) — $(tail -1 "$LOGD/t9b_replay_cmp.log") · $(grep -o '경우 [0-9]* · 발행 [0-9]* 건 · 블랙박스 [0-9]* 줄' "$LOGD/t9b_replay_dump.log" | tail -1) · $(grep -o '백엔드 [a-z]*' "$LOGD/t9b_replay_dump.log" | tail -1)"
  [ $r5 -eq 0 ] || grep '^\[다름\]' "$LOGD/t9b_replay_cmp.log" | head -5 | cut -c1-400
  fpy $SC --repo "$REFT" --dump "$WORK/s8_before.json" > "$LOGD/t9b_s8_dump.log" 2>&1; r1=$?
  fpy $SC --repo "$NEWT" --dump "$WORK/s8_after.json" >> "$LOGD/t9b_s8_dump.log" 2>&1; r2=$?
  python3 $SC --compare "$WORK/s8_before.json" "$WORK/s8_after.json" > "$LOGD/t9b_s8_cmp.log" 2>&1; r3=$?
  [ $((r1 + r2 + r3)) -eq 0 ]
  chk $? "T9b stage8 CAN 프레임 젯슨 판 ↔ 새 판(콘솔 기동 인자 --can-channel can1 --wait-takeover, 인수 대기 없는 기동) — $(tail -1 "$LOGD/t9b_s8_cmp.log") · $(grep -o '프레임 [0-9]* · status [0-9]*' "$LOGD/t9b_s8_dump.log" | tail -1) · 새 판에만 있는 status 키 $(sed -n 's/^status 에서 B 에만 있는 키: //p' "$LOGD/t9b_s8_cmp.log")"
  rm -f "$WORK"/s8_before.json "$WORK"/s8_after.json "$WORK"/replay_before2.json
  same_applied() { snap "$WORK/snap_now.txt"; cmp -s "$WORK/snap_applied.txt" "$WORK/snap_now.txt"; }
  same_applied; chk $? "T9b 비교 도구가 가짜 젯슨 트리에 아무것도 쓰지 않음"

  # --- T10 재적용(멱등)·T11 --verify
  cp "$FH/humanoid_repo_bak_1008/BACKUP_MANIFEST.md5" "$WORK/bm1.md5"
  dep t10_reapply --apply; rc=$?
  [ $rc -eq 0 ] && has t10_reapply '복사 생략' && has t10_reapply '배포 완료' && cmp -s "$WORK/bm1.md5" "$FH/humanoid_repo_bak_1008/BACKUP_MANIFEST.md5"
  chk $? "T10 재적용: 복사 생략·selftest 통과·백업 매니페스트 불변"
  dep t11_verify --verify; rc=$?
  [ $rc -eq 0 ] && has t11_verify '전부 통과'; chk $? "T11 --verify: 젯슨 = 매니페스트 · selftest 통과"
fi

# --- T12 콘솔 재기동(가짜 — 기록만): 제어 프로세스가 있으면 거부, 없으면 1 회
rm -f "$FH/.fake/restart.log"
{ cat "$WORK/ps_orig.txt"; echo "  4321 python3 /home/mama/humanoid_repo/jetson/scripts/stage8_12axis_mit_control_node.py --can-channel can1 --wait-takeover"; } > "$FH/.fake/ps.txt"
dep t12_restart_ctrl --restart-console; rc=$?; cp "$WORK/ps_orig.txt" "$FH/.fake/ps.txt"
[ $rc -ne 0 ] && has t12_restart_ctrl '제어 프로세스 실행 중' && [ ! -e "$FH/.fake/restart.log" ]; chk $? "T12 --restart-console: 제어 프로세스(stage8) 생존 시 거부(재기동 0 회)"
mv "$FH/humanoid_repo_bak_1008/STATE" "$WORK/STATE.orig"; grep -v selftest_pass "$WORK/STATE.orig" > "$FH/humanoid_repo_bak_1008/STATE"
dep t12_restart_nopass --restart-console; rc=$?; mv "$WORK/STATE.orig" "$FH/humanoid_repo_bak_1008/STATE"
[ $rc -ne 0 ] && has t12_restart_nopass 'selftest 통과 기록이 없다' && [ ! -e "$FH/.fake/restart.log" ]; chk $? "T12 --restart-console: 이 매니페스트의 selftest 통과 기록이 없으면 거부"
dep t12_restart --restart-console; rc=$?
[ $rc -eq 0 ] && [ "$(wc -l < "$FH/.fake/restart.log")" == 1 ]; chk $? "T12 --restart-console(적용 뒤 단독·유휴): 재기동 1 회 기록"

# --- T13 롤백 → 처음과 비교
dep t13_rollback --rollback --restart-console; rc=$?
[ $rc -eq 0 ] && has t13_rollback '롤백 완료' && same_as_start
chk $? "T13 --rollback --restart-console: $(grep -o 'RESTORED.*' "$LOGD/t13_rollback.log") → **처음 스냅샷과 동일**(파일 md5 $(grep -c '^[0-9a-f]\{32\} ' "$WORK/snap_now.txt") 개·폴더 목록)"
dep t13_restart_after_rollback --restart-console; rc=$?
[ $rc -ne 0 ] && has t13_restart_after_rollback '매니페스트와 같지 않다'; chk $? "T13 롤백 뒤 단독 --restart-console 은 거부(적용 상태가 아님)"

# --- T15 정책 폴더만 추가 → 제거(매니페스트와 별개 경로). 롤백된 상태(젯슨 판 코드)에서 W3S3d 폴더로 시험
PRUN=$(awk -F'\t' '!/^#/ && $3 == "policy" && $2 == "send" && $1 ~ /exported\/policy\.onnx$/ {sub(/\/exported\/policy\.onnx$/, "", $1); print $1; exit}' deploy/head_1008.plan)
if [ -n "$PRUN" ]; then
  dep t15_addpol "--add-policy=$PRUN"; r1=$?
  [ $r1 -eq 0 ] && has t15_addpol 'POLICY_OK' && [ -f "$FH/humanoid_repo/$PRUN/exported/policy.onnx" ] && grep -q "$PRUN/exported/policy.onnx" "$FH/humanoid_repo_bak_1008/POLICY_ADDED.md5"
  chk $? "T15 --add-policy: $(grep -o 'PADD_DONE [0-9]*' "$LOGD/t15_addpol.log") · 젯슨 추론 확인 $(grep -o 'POLICY_OK.*' "$LOGD/t15_addpol.log") · md5 기록"
  echo "tampered" >> "$FH/humanoid_repo/$PRUN/params/env.yaml"; dep t15_addpol_again "--add-policy=$PRUN"; r2=$?
  [ $r2 -ne 0 ] && has t15_addpol_again '덮어쓰지 않는다'; chk $? "T15 --add-policy: 젯슨에 다른 내용의 파일이 있으면 거부(덮어쓰지 않음)"
  dep t15_rmpol_tampered "--remove-policy=$PRUN"; r3=$?
  [ $r3 -ne 0 ] && has t15_rmpol_tampered 'md5 가 달라져 지우지 않음' && [ -f "$FH/humanoid_repo/$PRUN/params/env.yaml" ]; chk $? "T15 --remove-policy: 기록과 md5 가 다른 파일은 지우지 않고 알린다"
  cp -p "$POLICY_ROOT/$PRUN/params/env.yaml" "$FH/humanoid_repo/$PRUN/params/env.yaml"
  dep t15_rmpol "--remove-policy=$PRUN"; r4=$?
  [ $r4 -eq 0 ] && same_as_start; chk $? "T15 --remove-policy: $(grep -o 'PRM done.*' "$LOGD/t15_rmpol.log") → 처음 스냅샷과 동일"
fi

# --- T16 로컬 트리가 계획서와 달라진 경우(다른 작업의 수정·병합 흉내: 전송 대상 한 줄의 md5 를 바꾼 계획서) — 적용은 거부, 롤백은 막히지 않아야 한다
awk -F'\t' 'BEGIN{OFS="\t"} !/^#/ && $2 == "send" && $1 ~ /operator_console\.py$/ {$5 = "00000000000000000000000000000000"} {print}' deploy/head_1008.plan > "$WORK/plan_drift.plan"
HEAD1008_PLAN="$WORK/plan_drift.plan" dep t16_apply_drift --apply; r1=$?
[ $r1 -ne 0 ] && has t16_apply_drift '로컬 파일이 계획서와 다름' && same_as_start; chk $? "T16 로컬 원본이 계획서와 다르면 --apply 거부·무변경"
HEAD1008_PLAN="$WORK/plan_drift.plan" dep t16_rollback_drift --rollback; r2=$?
[ $r2 -eq 0 ] && has t16_rollback_drift '로컬 원본을 쓰지 않으므로 계속한다' && has t16_rollback_drift 'RESTORED 14' && same_as_start
chk $? "T16 같은 상황에서 --rollback 은 막히지 않는다(로컬 게이트도 생략) — $(grep -o 'RESTORED.*' "$LOGD/t16_rollback_drift.log" | head -1)"

if [ $QUICK -eq 0 ]; then
  # --- T14 롤백 뒤 재적용(백업 재사용) → 부분 적용 상태 복구 → 롤백
  dep t14_apply2 --apply; rc=$?
  [ $rc -eq 0 ] && has t14_apply2 'new=0 kept=14' && has t14_apply2 '배포 완료'; chk $? "T14 롤백 뒤 재적용: 원본 백업 재사용($(grep -o 'BKDONE.*' "$LOGD/t14_apply2.log"))"
  cp -p "$FH/humanoid_repo_bak_1008/files/jetson/scripts/rl_bridge_node.py" "$FH/humanoid_repo/jetson/scripts/rl_bridge_node.py"   # 중단된 적용 흉내: 하나는 옛 판,
  rm -f "$FH/humanoid_repo/rl_walking/deploy/w3_deploy.py" "$FH/humanoid_repo/jetson/scripts/w3_cmd.py"                           # 둘은 아직 안 생김
  dep t14_partial_dry; has t14_partial_dry '바뀔 파일(백업 뒤 덮어씀) 1 · 새로 생길 파일 2' ; r1=$?
  dep t14_partial_apply --apply; rc=$?
  [ $r1 -eq 0 ] && [ $rc -eq 0 ] && has t14_partial_apply 'new=0 kept=1 create=2' && cmp -s "$WORK/bm1.md5" "$FH/humanoid_repo_bak_1008/BACKUP_MANIFEST.md5"
  chk $? "T14 부분 적용 상태에서 재실행: 남은 3 파일만 처리·원본 백업을 새 판으로 덮어쓰지 않음"
  HEAD1008_PLAN="$WORK/plan_drift.plan" dep t14_rollback2 --rollback; rc=$?
  [ $rc -eq 0 ] && has t14_rollback2 '로컬 원본을 쓰지 않으므로 계속한다' && same_as_start
  chk $? "T14 다시 롤백(로컬 트리가 계획서와 달라진 상태에서) → $(grep -o 'RESTORED.*' "$LOGD/t14_rollback2.log" | head -1) → 처음 스냅샷과 동일"
fi

echo "== 결과: PASS $NPASS · FAIL $NFAIL — 로그 $LOGD"
if [ $KEEP -eq 0 ] && [ $NFAIL -eq 0 ]; then
  rm -rf "$FH" "$WORK/fake_ref" "$WORK/empty_site" "$WORK/pyhome" "$WORK"/*.orig "$WORK"/plan_bad.plan "$WORK"/plan_drift.plan "$WORK"/replay_*.json
  echo "   가짜 젯슨 홈 삭제(로그·스냅샷은 $WORK 에 남김). --keep 으로 보존"
fi
[ $NFAIL -eq 0 ]
