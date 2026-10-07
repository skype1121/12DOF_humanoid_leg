#!/bin/bash
# 학습 직렬 대기열 — 사용: train_queue.sh <앞선 완료 표지 파일> <앞선 완료 표지> "<태스크접미:폴더[:부모run:부모ck:반복]> ..."
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/f0e513e9-71ac-45c3-820b-1b06b07d54af/scratchpad
until grep -q "$2" "$1" 2>/dev/null; do sleep 15; done
for item in $3; do
  IFS=: read ST W PN PC IT <<< "$item"; mkdir -p $S/$W
  $S/w3train.sh $ST $W $PN $PC $IT >> $S/$W/chain.out 2>&1
done
echo "$(date +%H:%M) TRAIN_QUEUE_DONE $3"
