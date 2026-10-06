#!/bin/bash
# 고정 회귀 검사(1007) — 새 개선이 들어올 때마다 STAND·WALK·TURN·RECOVERY·SETTLE·POSTURE RESTORE 를 한 번에 돌려 PASS/FAIL 표를 낸다.
# 학습 없음 · 젯슨/실물 없음 · 평가 코드는 이 스크립트가 있는 체크아웃(워크트리 포함), 체크포인트·산출물은 본 트리(logs/·log_picture/).
#
# 사용:
#   reports/regress/run_regress.sh [옵션]
#     --mode quick|full        빠른(기본, env 축소·seed 1·KE odicw 팔만, 약 25–35 분) / 전체(seed 확장·KE imu+odicw 6 seed, 약 2–2.5 시간)
#     --w3 RUN:CKPT            W3 체크포인트(기본 W3S3d-15800)            --w3_task TASK  (기본 Biped12-HangStand-W3S3d-v0)
#     --profile s3d|s4d        문턱 프로필(s4d = 혼합 항목도 판정, s3d = 혼합은 참고). 기본: 과제 이름에 S4d 가 있으면 s4d
#     --stand RUN:CKPT         서기 정책(기본 G21a-11300)                 --rec RUN:CKPT  회복 정책(기본 G24Ra-21793)
#     --ke_arms "imu odicw"    KE 종료 판정기 팔(기본: 모드 값)
#     --w3_extra / --stand_extra / --ke_extra "인자"   후보 개선의 추가 인자(해당 평가기에 그대로 전달 — 매니페스트에 기록)
#     --only w3,dist,stand,ke  일부 묶음만(기본 전부)       --tag 이름  실행 이름 꼬리표       --prev DIR  비교할 이전 실행(기본 자동)
#     --resume DIR             중단된 실행 이어 하기(유효 완료 작업은 건너뜀)   --dry  계획만
# 출력: reports/regress/runs/<RID>_<tag>/{manifest.json, jobs.tsv, status.tsv, summary.md, results.json, logs/}
#       평가 산출물은 본 트리 log_picture/ 의 rg_<RID>_* 태그.
# GPU 규칙: 사용 메모리 ≤ 10500 MiB 될 때까지 대기 · 다른 평가와 같은 잠금(eval17.gpu.lock)으로 동시 1 개 ·
#           메모리/PhysX 실패·산출물 누락·물리 정지 = 무효 → 재실행(최대 3) · 종료는 명시 PID(프로세스 그룹)로만(pkill 금지).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$HERE/../.." && pwd); DATA=${REGRESS_DATA_REPO:-/home/ryu/humanoid_leg_test1}
SP=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
LOCK=${EVAL_LOCK:-$SP/eval17.gpu.lock}; MEM_MAX=${REGRESS_MEM_MAX:-10500}; ISAAC=${ISAACLAB_SH:-/home/ryu/IsaacLab/isaaclab.sh}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$WT EVAL_DATA_REPO=$DATA W3EVAL_REPO=$DATA REGRESS_DATA_REPO=$DATA

MODE=quick; W3=2026-10-07_00-11-12_w3S3d_from_s2d:model_15800.pt; W3T=Biped12-HangStand-W3S3d-v0; PROFILE=""
STAND=2026-10-02_21-23-08_g21a:model_11300.pt; REC=2026-10-04_10-08-41_g24ra_seed1_from_w2e20294:model_21793.pt
KEARMS=""; W3X=""; STX=""; KEX=""; ONLY=w3,dist,stand,ke; TAG=cand; PREV=""; RESUME=""; DRY=0
while [ $# -gt 0 ]; do case $1 in
  --mode) MODE=$2; shift 2;; --w3) W3=$2; shift 2;; --w3_task) W3T=$2; shift 2;; --profile) PROFILE=$2; shift 2;;
  --stand) STAND=$2; shift 2;; --rec) REC=$2; shift 2;; --ke_arms) KEARMS=$2; shift 2;;
  --w3_extra) W3X=$2; shift 2;; --stand_extra) STX=$2; shift 2;; --ke_extra) KEX=$2; shift 2;;
  --only) ONLY=$2; shift 2;; --tag) TAG=$2; shift 2;; --prev) PREV=$2; shift 2;; --resume) RESUME=$2; shift 2;; --dry) DRY=1; shift;;
  -h|--help) sed -n 2,22p "$0"; exit 0;;
  *) echo "모르는 인자: $1"; exit 2;; esac; done
[ -z "$PROFILE" ] && { case $W3T in *S4d*) PROFILE=s4d;; *) PROFILE=s3d;; esac; }

if [ -n "$RESUME" ]; then RD=$(cd "$RESUME" && pwd); RID=$(python3 -c "import json;print(json.load(open('$RD/manifest.json'))['rid'])")
else
  RID=${MODE:0:1}$(date +%m%d_%H%M); RD=$HERE/runs/${RID}_${TAG}; mkdir -p $RD/logs
  CFG=$(python3 - "$MODE" "$RID" "$W3" "$W3T" "$PROFILE" "$STAND" "$REC" "$KEARMS" "$W3X" "$STX" "$KEX" "$ONLY" "$TAG" <<'EOF'
import json, sys
k = "mode rid w3 w3_task profile stand rec ke_arms w3_extra stand_extra ke_extra only tag".split()
d = dict(zip(k, sys.argv[1:])); d["ke_arms"] = d["ke_arms"].split() or None
print(json.dumps(d, ensure_ascii=False))
EOF
)
  python3 $HERE/regress_lib.py plan $RD "$CFG" || { echo "계획 실패"; exit 2; }
fi
echo "실행 디렉터리: $RD"
[ $DRY = 1 ] && { cat $RD/jobs.tsv | cut -f1,5 | sed 's/^/  /'; exit 0; }

ST=$RD/status.tsv; touch $ST
wait_mem() { while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -le $MEM_MAX ] && return; sleep 20; done; }
done_ok() { grep -q "^$1	유효	" $ST; }

N=$(wc -l < $RD/jobs.tsv); I=0; T_START=$(date +%s)
while IFS=$'\t' read -r JID KIND TAGJ TMO ARGS; do
  I=$((I+1))
  if done_ok $JID; then echo "[$I/$N] $JID 이미 유효 — 건너뜀"; continue; fi
  SCRIPT=$WT/rl_walking/scripts/$([ $KIND = w3 ] && echo eval_w3.py || echo eval_ground_recovery.py)
  LG=$RD/logs/$JID.l; ok=0; note=""
  exec 9>$LOCK; flock 9
  for TRY in 1 2 3; do
    wait_mem
    if [ $KIND = w3 ]; then rm -f $DATA/log_picture/w3eval_${TAGJ}.json $DATA/log_picture/w3eval_${TAGJ}.txt
    else rm -f $DATA/log_picture/ground_recovery_${TAGJ}.txt $DATA/log_picture/ground_recovery_${TAGJ}_envs.csv $DATA/log_picture/ground_recovery_${TAGJ}_handoff.npz; fi
    t0=$(date +%s)
    eval "setsid $ISAAC -p $SCRIPT $ARGS --out_tag $TAGJ" < /dev/null > $LG 2>&1 &
    PG=$!; echo "$(date +%H:%M:%S) [$I/$N] 시작 $JID (PID $PG, 시도 $TRY)"
    while kill -0 $PG 2>/dev/null; do
      sleep 3
      if [ $(( $(date +%s) - t0 )) -gt $TMO ]; then echo "  시간 초과 ${TMO}s → PID 그룹 $PG 종료"; kill -TERM -- -$PG 2>/dev/null; sleep 10; kill -KILL -- -$PG 2>/dev/null; fi
    done
    kill -KILL -- -$PG 2>/dev/null    # 같은 그룹의 남은 자식(Kit)만 정리 — 명시 PID 그룹
    dt=$(( $(date +%s) - t0 ))
    note=$(python3 $HERE/regress_lib.py check $RD $JID $LG $t0); rc=$?
    if [ $rc = 0 ]; then ok=1; echo "$(date +%H:%M:%S)   유효 ($dt s)"; break; fi
    echo "$(date +%H:%M:%S)   무효: $note → 재시도"; cp $LG $LG.dead$TRY; sleep 45
  done
  flock -u 9
  sed -i "/^$JID	/d" $ST
  printf "%s\t%s\t%s\t%s\t%s\n" $JID $([ $ok = 1 ] && echo 유효 || echo 무효) $TRY "$note" $dt >> $ST
done < $RD/jobs.tsv

python3 $HERE/regress_lib.py finish $RD
echo "$(date +%H:%M:%S) 실행 끝 ($(( ($(date +%s) - T_START) / 60 )) 분) → 요약"
python3 $HERE/summarize.py $RD ${PREV:+--prev $PREV}
