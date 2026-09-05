#!/bin/bash
# h3 재판정 (0906 감사 B1~B3 정합 후): 배포 경로(--deploy old = chain) 만 — 미배포 new 경로·사다리(코드 불변) 생략.
#   judge_gen_h3.sh <태그> <run_dir> <checkpoint.pt> <task> <usd>
#   묶음: old_com0 · old_com16 · dr_on · nom_old(명목 지연 4틱) · V6 고정환경 xold_com0 · xold_com16
#         + 대조군 xold_com16_h2 (--deploy old_h2 = 0905 래퍼 그대로 → h2 수치 재현 여부로 하네스 불변 확인)
set -u
TAG=$1; RUN=$2; CKPT=$3; TASK=$4; USD=$5
cd /home/ryu/humanoid_leg_test1
export PYTHONPATH=/home/ryu/humanoid_leg_test1 OMNI_KIT_ACCEPT_EULA=YES
LAB=/home/ryu/IsaacLab/isaaclab.sh
SCRATCH=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
OUT=$SCRATCH/judge_$TAG.txt
: > "$OUT"
echo "[$(date +%T)] h3 판정 시작 $TAG ($RUN/$CKPT, $TASK)" | tee -a "$OUT"
EXTRA=${EXTRA:-}   # 예: EXTRA="--ablate gains" (강성 DR 제거 = h3g)
BASE="--headless --task $TASK --load_run $RUN --checkpoint $CKPT --num_envs 64 --steps 400 --still --mass_fix 0.0 --usd $USD $EXTRA"
XBASE="--headless --task Biped12-Velocity-Stage4V2-Stand-V6-v0 --load_run $RUN --checkpoint $CKPT --num_envs 64 --steps 400 --still --mass_fix 0.0 --usd $USD $EXTRA"
echo "  EXTRA=[$EXTRA]" | tee -a "$OUT"
summ() { f=$1
  echo "$(grep -m1 '낙상(조기' "$f" | tr -d '\n') | $(grep -m1 '초기 3s' "$f" | sed 's/.*평균/평균/' | tr -d '\n') | $(grep -m1 'E_smth (' "$f" | tr -s ' ' | tr -d '\n') | $(grep -m1 '낙상 방향' "$f" | tr -s ' ' | tr -d '\n')"
}
run() { tag=$1; base=$2; shift 2
  rm -f "log_picture/드리프트DR_${TAG}_$tag.txt"
  $LAB -p rl_walking/scripts/eval_drift_dr.py $base --out_tag "${TAG}_$tag" "$@" > "$SCRATCH/j_${TAG}_$tag.log" 2>&1
  f="log_picture/드리프트DR_${TAG}_$tag.txt"
  if [ -f "$f" ]; then echo "  == $tag: $(summ "$f")" | tee -a "$OUT"
  else echo "  FAIL $tag (tail: $(tail -2 "$SCRATCH/j_${TAG}_$tag.log" | tr '\n' ' ' | cut -c1-200))" | tee -a "$OUT"; fi
}
run old_com0      "$BASE"  --deploy old --com_fix 0.0
run old_com16     "$BASE"  --deploy old --com_fix 0.016
run dr_on         "$BASE"  --deploy old
run nom_old       "$BASE"  --nominal --deploy old
run xold_com0     "$XBASE" --deploy old --com_fix 0.0
run xold_com16    "$XBASE" --deploy old --com_fix 0.016
run xold_com16_h2 "$XBASE" --deploy old_h2 --com_fix 0.016
echo "[$(date +%T)] h3 판정 종료 $TAG" | tee -a "$OUT"
