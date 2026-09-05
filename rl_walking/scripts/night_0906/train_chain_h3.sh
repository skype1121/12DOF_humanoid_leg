#!/bin/bash
# 0906 야간 개선 루프 학습 체인: v18c(V18 env, 대조군) → 판정 → v19(V19: 지연 3~8틱) → 판정.
#   둘 다 웜스타트 v18b(2026-09-05_19-37-39/model_120998.pt), 6000it, 4096env. 판정 = judge_gen.sh(사다리+배포 경로 h3+명목 지연4).
#   10:00 이후엔 새 학습을 시작하지 않는다(11:45 새 GPU 런치 금지 규칙, 세대 ≈1.6h).
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
cd /home/ryu/humanoid_leg_test1
echo $$ > "$S/train_chain.pid"; echo "train_chain $$ $(date +%F_%T)" >> "$S/pids.txt"
USD=/home/ryu/humanoid_leg_test1/rl_walking/assets/usd_final3/biped12.usd
WRUN=2026-09-05_19-37-39; WCKPT=model_120998.pt
pick() { tag=$1; LOG=$S/train_$tag.log
  RUN=$(grep -oE "biped12_stage4/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}" "$LOG" | tail -1 | sed 's|.*/||')
  [ -n "$RUN" ] || { echo "[$(date +%T)] $tag: run 디렉토리 못 찾음"; return 1; }
  CKPT=$(ls logs/rsl_rl/biped12_stage4/$RUN/model_*.pt 2>/dev/null | sed 's|.*/||' | sort -t_ -k2 -n | tail -1)
  [ -n "$CKPT" ] || { echo "[$(date +%T)] $tag: 체크포인트 없음"; return 1; }
  echo "$RUN $CKPT" > "$S/${tag}_pick.txt"; echo "[$(date +%T)] $tag run=$RUN ckpt=$CKPT"
}
for spec in "v18c Biped12-Velocity-Stage4V2-Stand-V18-v0" "v19 Biped12-Velocity-Stage4V2-Stand-V19-v0"; do
  set -- $spec; TAG=$1; TASK=$2
  H=$(date +%H%M); if [ "$H" -gt 1000 ]; then echo "[$(date +%T)] 마감 임박(>10:00) — $TAG 런치 생략"; continue; fi
  bash "$S/train_gen.sh" "$TAG" "$TASK" "$WRUN" "$WCKPT" 6000
  pick "$TAG" || continue
  read RUN CKPT < "$S/${TAG}_pick.txt"
  bash "$S/judge_gen.sh" "$TAG" "$RUN" "$CKPT" "$TASK" "$USD"
done
echo "[$(date +%T)] train chain 종료"
