#!/bin/bash
# 후속 체인 (0906): train_chain_h3 종료 대기 → v18c·v19 에 h3g(강성 DR 제거) 판정 추가 → v20(V20: 강성 하한 0.6) 학습 → judge_gen + h3g.
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
cd /home/ryu/humanoid_leg_test1
echo $$ > "$S/train_chain_v20.pid"; echo "train_chain_v20 $$ $(date +%F_%T)" >> "$S/pids.txt"
USD=/home/ryu/humanoid_leg_test1/rl_walking/assets/usd_final3/biped12.usd
WRUN=2026-09-05_19-37-39; WCKPT=model_120998.pt
PREV=$(cat "$S/train_chain.pid" 2>/dev/null)
while [ -n "$PREV" ] && kill -0 "$PREV" 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] train_chain_h3 종료 감지"
for spec in "v18c Biped12-Velocity-Stage4V2-Stand-V18-v0" "v19 Biped12-Velocity-Stage4V2-Stand-V19-v0"; do
  set -- $spec; TAG=$1; TASK=$2
  [ -f "$S/${TAG}_pick.txt" ] || { echo "[$(date +%T)] $TAG pick 없음 — h3g 생략"; continue; }
  read RUN CKPT < "$S/${TAG}_pick.txt"
  EXTRA="--ablate gains" bash "$S/judge_gen_h3.sh" "${TAG}_h3g" "$RUN" "$CKPT" "$TASK" "$USD"
done
H=$(date +%H%M); if [ "$H" -gt 1000 ]; then echo "[$(date +%T)] 마감 임박(>10:00) — v20 런치 생략"; echo "[$(date +%T)] v20 chain 종료"; exit 0; fi
TAG=v20; TASK=Biped12-Velocity-Stage4V2-Stand-V20-v0
bash "$S/train_gen.sh" "$TAG" "$TASK" "$WRUN" "$WCKPT" 6000
LOG=$S/train_$TAG.log
RUN=$(grep -oE "biped12_stage4/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}" "$LOG" | tail -1 | sed 's|.*/||')
CKPT=$(ls logs/rsl_rl/biped12_stage4/$RUN/model_*.pt 2>/dev/null | sed 's|.*/||' | sort -t_ -k2 -n | tail -1)
if [ -n "$RUN" ] && [ -n "$CKPT" ]; then
  echo "$RUN $CKPT" > "$S/${TAG}_pick.txt"; echo "[$(date +%T)] $TAG run=$RUN ckpt=$CKPT"
  bash "$S/judge_gen.sh" "$TAG" "$RUN" "$CKPT" "$TASK" "$USD"
  EXTRA="--ablate gains" bash "$S/judge_gen_h3.sh" "${TAG}_h3g" "$RUN" "$CKPT" "$TASK" "$USD"
else
  echo "[$(date +%T)] $TAG 체크포인트 없음"
fi
echo "[$(date +%T)] v20 chain 종료"
