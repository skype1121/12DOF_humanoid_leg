#!/bin/bash
# 세대 판정 배터리: judge_gen.sh <태그> <run_dir> <checkpoint.pt> <task> <usd>
#   ladder 0~3 (DR OFF/ON) + 배포조건 드리프트 (deploy old·new, com 0/+0.016, payload 중립)
set -u
TAG=$1; RUN=$2; CKPT=$3; TASK=$4; USD=$5
cd /home/ryu/humanoid_leg_test1
export PYTHONPATH=/home/ryu/humanoid_leg_test1 OMNI_KIT_ACCEPT_EULA=YES
LAB=/home/ryu/IsaacLab/isaaclab.sh
SCRATCH=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
OUT=$SCRATCH/judge_$TAG.txt
: > "$OUT"
echo "[$(date +%T)] 판정 시작 $TAG ($RUN/$CKPT, $TASK)" | tee -a "$OUT"
rm -f "demo_output/ladder_$TAG.json"
for dr in 0 1; do for st in 0 1 2 3; do
  $LAB -p rl_walking/scripts/eval_ladder.py --headless --task $TASK --load_run $RUN --checkpoint $CKPT \
    --stage $st --dr $dr --usd $USD --payload_neutral --out_tag $TAG > "$SCRATCH/j_${TAG}_s${st}_dr${dr}.log" 2>&1
done; done
python3 - "$TAG" <<'EOF' | tee -a "$OUT"
import json, sys
rows = json.load(open(f"demo_output/ladder_{sys.argv[1]}.json"))
_have = {(d["stage"], d["dr"]) for d in rows}
for st in (0,1,2,3):
    for dr in (False, True):
        if (st, dr) not in _have: print(f"  ⚠ stage{st} DR{'ON' if dr else 'OFF'} 결과 없음 (크래시?)")
for d in rows:
    print(f"  stage{d['stage']} DR{'ON ' if d['dr'] else 'OFF'}: 낙하 {d['fall']:2d}/{d['n']} 추세 {d['trend_dps']:+7.2f} E_smth {d['e_smth']:.3f} 에너지 {d['energy']:.2f} gate={d['pass_gate']}")
EOF
BASE="--headless --task $TASK --load_run $RUN --checkpoint $CKPT --num_envs 64 --steps 400 --still --mass_fix 0.0 --usd $USD"
run() { tag=$1; shift
  rm -f "log_picture/드리프트DR_${TAG}_$tag.txt"   # 0905 감사: 스테일 리포트가 새 판정으로 둔갑하지 않게
  $LAB -p rl_walking/scripts/eval_drift_dr.py $BASE --out_tag "${TAG}_$tag" "$@" > "$SCRATCH/j_${TAG}_$tag.log" 2>&1
  f="log_picture/드리프트DR_${TAG}_$tag.txt"
  if [ -f "$f" ]; then
    echo "  == $tag: $(grep -m1 '낙상(조기' "$f" | tr -d '\n') | $(grep -m1 '초기 3s' "$f" | sed 's/.*평균/평균/' | tr -d '\n') | $(grep -m1 'E_smth (' "$f" | tr -s ' ' | tr -d '\n')" | tee -a "$OUT"
  else echo "  FAIL $tag" | tee -a "$OUT"; fi
}
run old_com0   --deploy old --com_fix 0.0
run old_com16  --deploy old --com_fix 0.016
run new_com0   --deploy new --com_fix 0.0
run new_com16  --deploy new --com_fix 0.016
run dr_on      --deploy old
# 결정론적 명목 평가 (0904 승윤님 원칙): 모든 DR OFF, 실물 nominal 모델 — 물리 실현성 지표는 리포트 안
run nom_old    --nominal --deploy old
run nom_new    --nominal --deploy new
# 고정 평가환경 교차평가 (V6 태스크 DR 대역 — 세대 간 공정 비교; v6b F3b 기준선과 동일 조건)
XBASE="--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run $RUN --checkpoint $CKPT --num_envs 64 --steps 400 --still --mass_fix 0.0 --usd $USD"
xrun() { tag=$1; shift
  rm -f "log_picture/드리프트DR_${TAG}_x$tag.txt"
  $LAB -p rl_walking/scripts/eval_drift_dr.py $XBASE --out_tag "${TAG}_x$tag" "$@" > "$SCRATCH/j_${TAG}_x$tag.log" 2>&1
  f="log_picture/드리프트DR_${TAG}_x$tag.txt"
  if [ -f "$f" ]; then echo "  == X$tag(V6env): $(grep -m1 '낙상(조기' "$f" | tr -d '\n') | $(grep -m1 '초기 3s' "$f" | sed 's/.*평균/평균/' | tr -d '\n') | $(grep -m1 'E_smth (' "$f" | tr -s ' ' | tr -d '\n')" | tee -a "$OUT"
  else echo "  FAIL X$tag" | tee -a "$OUT"; fi
}
xrun old_com0   --deploy old --com_fix 0.0
xrun old_com16  --deploy old --com_fix 0.016
xrun new_com16  --deploy new --com_fix 0.016
echo "[$(date +%T)] 판정 종료 $TAG" | tee -a "$OUT"
