#!/bin/bash
# 인수(핸드오프→stage8) "빡" 재현·캡처 — CAN 전 프레임 녹화 (2026-08-02 준비)
#
# 배경: 핸드오프 후 stage8이 명령을 시작하는 순간 모터가 튀는 사건 2회 (0802).
#   코드 수사로는 게인·패킹·베이스라인 전부 정상 — 모터 펌웨어 레벨 거동
#   (ENTER 폴링/명령 교대) 의심. 바이트 증거로 확정한다.
#
# 벤치 재현 절차 (매달림 = 무하중이라 튀어도 무해):
#   1. 로봇 매달기 (발 무접촉) → 이 스크립트 실행 (캡처 시작)
#   2. hang_stand 기동 → touch /tmp/hang_gain (운용게인)
#   3. touch /tmp/hang_ground   (선언만 — 매달림이라 실착지 아님, 핸드오프 게이트용)
#   4. touch /tmp/hang_stand_handoff → stage8 기동 → SET_BASELINE → ARM
#   5. "빡" 발생 여부 관찰 (눈+귀) → 이 스크립트 재실행 아닌 kill로 캡처 종료
#   6. 분석: 모터별 '마지막 hang_stand 프레임' vs '첫 stage8 프레임' 바이트 비교
#      grep " 00A\b" 등 ID별 추출, 위치/게인 비트 디코드는 ak_mit_decoder 참조
#
# 사용: bash takeover_capture.sh          # 캡처 시작 (백그라운드)
#       kill $(cat /tmp/candump.pid)      # 캡처 종료
set -e
mkdir -p ~/logs
LOG=~/logs/takeover_$(date +%Y%m%d_%H%M%S).candump
if [ -f /tmp/candump.pid ] && kill -0 "$(cat /tmp/candump.pid)" 2>/dev/null; then
  echo "[takeover_capture] 이미 녹화 중 (PID $(cat /tmp/candump.pid)) — 먼저 종료하세요"
  exit 1
fi
candump -L can1 > "$LOG" 2> "$LOG.err" &
CPID=$!
echo $CPID > /tmp/candump.pid
sleep 0.5
if ! kill -0 $CPID 2>/dev/null; then
  echo "[takeover_capture] FAIL — candump 즉사 (can1 상태 확인): $(head -1 "$LOG.err" 2>/dev/null)"
  exit 1
fi
echo "[takeover_capture] 녹화 시작: $LOG (PID $CPID)"
echo "[takeover_capture] 종료: kill \$(cat /tmp/candump.pid)"
